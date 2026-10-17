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
/* serving-capstone (the "SDE → LLM serving" path, last stop): How do you rescue a congested chat service?
   Fidelity: Toy run. Everything on the page comes from SC_ENGINE above (pasted unchanged from capstone/engine.js; the Python reference is
   capstone/engine.py, and tests/serving-capstone_engine.py compares the two on every load and a grid of settings):
   SC_ENGINE.simulate(load, cfg) gives the run (R, steps, mem, events) and SC_ENGINE.metrics(run) its numbers. The views only read the run.
   Derived here, and checked against the engine's own totals by tests/serving-capstone_engine.py:
   - why a waiting request waits in each step: static batching → batch; running ≥ cap → cap; the step's tokens reached the budget → budget;
     otherwise memory (the engine's first blocked request sets the reason for everyone behind it); arrived during a step → step;
   - the KV pool after each step: running requests' own blocks (from their chunks and decode tokens), each app's shared prompt blocks in use
     (the longest prefix a running request of that app holds) and idle (present minus in use); an eviction belongs to the step whose memory
     snapshot first lacks the block.
   Words (EN / ZH): part / 部分 for guided cards, step / 步 for engine steps; initial setup / 初始设置 (the service as it runs today);
   token budget / token 预算; cap on running requests / 运行请求上限; prefix caching / 前缀缓存.
   window.__sc is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A||typeof SC_ENGINE==='undefined') return;
const {esc,$,$$}=A, ID='serving-capstone', E=SC_ENGINE, C=E.C, G=E.GOALS, EPS=1e-9;
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Llama‑3.1‑8B', GPUN='H100\u00a0SXM', NB='\u00a0';      // no-break hyphens and spaces: a name or a value never breaks

/* =====================================================================
   formatting: stage values keep their digits; prose rounds (R6.20)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
/* a time on the stage: 7.77 ms, 32.4 ms, 126 ms, 1.49 s, 11.5 s */
const fS=sec=>{const v=sec*1e3; if(v<9.995) return fixd(v,2)+NB+'ms'; if(v<99.95) return fixd(v,1)+NB+'ms'; if(v<999.5) return fint(v)+NB+'ms'; if(sec<9.995) return fixd(sec,2)+NB+'s'; return fixd(sec,1)+NB+'s';};
/* a time in prose: two significant digits (1.5 s, 130 ms, 32 ms) */
const fR=sec=>{const v=sec*1e3; if(sec>=9.95) return fint(sec)+NB+'s'; if(v>=995) return fixd(sec,1)+NB+'s'; if(v>=99.5) return fint(Math.round(v/10)*10)+NB+'ms'; if(v>=9.95) return fint(v)+NB+'ms'; return fixd(v,1)+NB+'ms';};
const fC=sec=>fixd(sec,2)+NB+'s';
const sv=sec=>{const t=fS(sec); return parseFloat(t)*(t.includes('ms')?1e-3:1);};   // a time as shown: ▲▼ reproduce from the shown values (R6.6)                               // the playhead's clock
const fG=gib=>fixd(gib,1)+NB+'GiB';
const pct0=x=>fint(x*100)+'%';
const aN=n=>{const g=fint(n).split(',')[0]; return g[0]==='8'||g==='11'||g==='18'?'an':'a';};
const tokK=n=>n>=1024&&n%1024===0?(n/1024)+'K':fint(n);
const bigN=n=>n>=1e6?(A.ui.big?A.ui.big(n):A.fmt.big(n)):fint(n);   // R6.18: more than 6 digits → 3 significant digits
const SUP='⁰¹²³⁴⁵⁶⁷⁸⁹';
/* 2.04 × 10¹⁵ (review r2, N2); no hover with the exact value (the owner's clarification of 2026-10-07: nobody needs it), like the kit's A.ui.big */
const bigE=n=>{if(n<1e12) return bigN(n); let e=Math.floor(Math.log10(n)), m=n/Math.pow(10,e); if(+m.toFixed(2)>=10){e++; m/=10;} const t=`${m.toFixed(2)} × 10${String(e).split('').map(d=>SUP[+d]).join('')}`;
  return `<span class="num-big">${t}</span>`;};
const BLK_B=C.BLOCK*C.KV_TOK;                                    // one 16-token block of KV cache: 2 MiB
const bGiB=b=>b*BLK_B/C.GiB;
const fGs=b=>b>0&&bGiB(b)<0.05?fint(b*BLK_B/1048576)+NB+'MiB':fG(bGiB(b));   // blocks → GiB, or MiB when under 0.05 GiB
const fCB=b=>{const g=bGiB(b), r=Math.round(g); return (Math.abs(g-r)<1e-9?String(r):fixd(g,1))+NB+'GiB';};   // a cache budget: whole GiB without “.0” (review r3, finding 6)
const POOL_GIB=bGiB(C.POOL_B), POOL92=Math.floor((80*C.GiB*0.92-C.W_MEM-2*C.GiB)/C.KV_TOK);   // vLLM's current gpu_memory_utilization default
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const GIB=()=>DEF('GiB',kt('gib')||tx('A GiB is 2^30 bytes.','GiB 是 2^30 字节。'));
/* a change against the initial setup, from the values as shown (R6.6): percentage points, "2.1 times", or a percentage decrease */
function dFmt(a,b,o){o=o||{}; if(!isFinite(a)||!isFinite(b)) return '';
  if(o.dp!=null){const k=Math.pow(10,o.dp); a=Math.round(a*k)/k; b=Math.round(b*k)/k;}
  if(a===b) return '';
  if(o.pts){const d=b-a, n=fixd(Math.abs(d),o.dp||0); return (d>0?'▲ ':'▼ ')+tx(n==='1'?`${n} pt`:`${n} pts`,`${n} 个百分点`);}
  if(a===0) return tx('▲ up from 0','▲ 原来是 0'); if(b===0) return tx('▼ down to 0','▼ 降到 0');
  const q=b/a; if(q>=1.995){const n=q>=9.95?fint(q):fixd(q,1); return tx(`▲ ${n} times`,`▲ ${n} 倍`);}
  const r=(q-1)*100, p=Math.sign(r)*Math.round(Math.abs(r)+1e-9); if(p===0) return '';
  if(p<=-100) return '▼ −99.9%'; return (p>0?'▲ +':'▼ −')+Math.abs(p)+'%';}

/* =====================================================================
   the loads, the settings and the runs
   ===================================================================== */
const LOADS=['stall','shared','apps','challenge'];
const SPEC={}; const spec=n=>SPEC[n]||(SPEC[n]=E.LOADS[n]());
/* what a load holds, read from its requests, so no count or size is typed twice */
const LF=load=>{const sp=spec(load), ch=sp.filter(x=>x.cls==='chat'), ap=APPS.filter(a=>ch.some(x=>x.app===a));
  return {n:ch.length, docs:sp.filter(x=>x.cls==='doc'), apps:ap, pre:(ch.find(x=>x.pre)||{pre:0}).pre};};
const WEN=['no','one','two','three','four','five','six','seven','eight','nine'], WZH=['零','一','两','三','四','五','六','七','八','九'];
const wEN=n=>WEN[n]||String(n), wZH=n=>WZH[n]||String(n), cap1=s=>s.charAt(0).toUpperCase()+s.slice(1);
const CAPS=[8,16,32,64,128], BUDS=[0,8192,2048,512,128], CBS=[512,1536,3072,6144,null], CBS5=[1536,3072,6144,null], BMIN=Math.min(...BUDS.filter(Boolean));   // cache budgets in blocks: 1 / 3 / 6 / 12 GiB / no limit (review r3, finding 6)
const BASE={policy:E.BASELINE.policy,cap:E.BASELINE.cap,budget:E.BASELINE.budget,cache:E.BASELINE.cache,cb:E.BASELINE.cacheBlocks};   // the service as it runs today
const cfgOf=s=>({policy:s.policy,cap:s.cap,budget:s.budget,cache:!!s.cache,cacheBlocks:s.cache?s.cb:null});
const keyOf=(load,s)=>[load,s.policy[0],s.cap,s.budget,s.cache?1:0,s.cache&&s.cb!=null?s.cb:'n'].join('_');
const KORDER=['chat','doc','support','code','contracts','manual','policy','codebase','paper'], APPS=['support','code','contracts','manual','policy','codebase','paper'];
const DOCAPPS=['manual','policy','codebase','paper'];   // part 5: questions about four long documents, each a shared 24K-token prefix
const KIND={chat:{en:'short chat',zh:'短对话'},doc:{en:'long document',zh:'长文档'},support:{en:'support app',zh:'客服应用'},code:{en:'code app',zh:'代码应用'},contracts:{en:'contracts app',zh:'合同应用'},
  manual:{en:'the manual',zh:'手册'},policy:{en:'the policy',zh:'政策文件'},codebase:{en:'the codebase',zh:'代码库'},paper:{en:'the paper',zh:'论文'}};
const APPN={support:{en:'support',zh:'客服'},code:{en:'code',zh:'代码'},contracts:{en:'contracts',zh:'合同'},manual:{en:'manual',zh:'手册'},policy:{en:'policy',zh:'政策'},codebase:{en:'codebase',zh:'代码库'},paper:{en:'paper',zh:'论文'}};
const kindOf=r=>r.cls==='doc'?'doc':(r.app||'chat');
/* "short chat" means every chat (they share one goal); in a load with apps, the chats without an app prompt are "plain chats" (review r1, S9) */
const kindL=(k,apps)=>k==='chat'&&apps?tx('plain chat','普通对话'):L(KIND[k]);
const rName=r=>{const k=kindOf(r), n='#'+(r.id+1); return k==='chat'?tx(`chat ${n}`,`对话 ${n}`):(k==='doc'?tx(`document ${n}`,`文档 ${n}`):tx(`${APPN[k].en} ${n}`,`${L(APPN[k])} ${n}`));};
const gapGoal=cls=>G[cls]&&G[cls].gap!=null?G[cls].gap:Infinity;

/* one run with everything the views read, cached by load and settings */
const RC=new Map();
function getRun(load,s){const key=keyOf(load,s); let o=RC.get(key); if(o){RC.delete(key); RC.set(key,o); return o;}
  o=build(spec(load),cfgOf(s)); o.key=key; o.load=load; o.s={...s}; RC.set(key,o); if(RC.size>90) RC.delete(RC.keys().next().value); return o;}
function build(sp,cfg){
  const run=E.simulate(sp,cfg); E.check(run); const m=E.metrics(run), R=run.R, st=run.steps, cacheOn=!!cfg.cache;
  const why=st.map(x=>cfg.policy==='static'?'batch':(x.nrun>=cfg.cap?'cap':((cfg.budget||0)&&x.T>=cfg.budget?'budget':'memory')));
  const t0M=new Map(), t1M=new Map(); st.forEach((x,i)=>{t0M.set(x.t0,i); t1M.set(x.t1,i);});
  /* an eviction during a step's allocations has t = its start; one after a release has t = its end and is followed by that finish */
  const evAt=new Map(), evOrder=[], ev=run.events, evTrim=new Map(), evAlloc=new Map(), evAtT=new Map(), evAtA=new Map();
  for(let p=0;p<ev.length;p++){const e=ev[p]; if(e.type!=='evict') continue; let q=p+1; while(q<ev.length&&ev[q].type==='evict') q++;
    const trim=q<ev.length&&ev[q].type==='finish'&&ev[q].t===e.t, s=trim?t1M.get(e.t):t0M.get(e.t); if(s==null) continue;
    evOrder.push([s,e.app,e.i,trim]);
    [evAt,trim?evAtT:evAtA].forEach(M=>{if(!M.has(s)) M.set(s,{}); const d=M.get(s); d[e.app]=(d[e.app]||0)+1;});
    const em=trim?evTrim:evAlloc; em.set(s,(em.get(s)||0)+1);}
  evOrder.sort((x,y)=>x[0]-y[0]);
  const se=r=>(cacheOn&&r.app)?Math.floor(r.pre/C.BLOCK)*C.BLOCK:0;
  /* shared prompt blocks present per app after every step: (present − evicted) ∪ computed in the step */
  const byStep=new Map();
  if(cacheOn) R.forEach(r=>{const e=se(r); if(!e) return; r.chunks.forEach(([si,a,b])=>{if(a>=e) return; if(!byStep.has(si)) byStep.set(si,[]); byStep.get(si).push([r.app,Math.floor(a/C.BLOCK),Math.floor(Math.min(b,e)/C.BLOCK)]);});});
  /* presPre: during step s (its allocations done, before its finished requests release: the engine's peak is taken there);
     pres: after step s (after the releases and the cache trims) */
  const pres=[], presPre=[], sets={}, evS=new Map();
  evOrder.forEach(e=>{if(!evS.has(e[0])) evS.set(e[0],[]); evS.get(e[0]).push(e);});
  const count=()=>{const cnt={}; for(const a in sets) if(sets[a].size) cnt[a]=sets[a].size; return cnt;};
  for(let s=0;s<st.length;s++){const es=evS.get(s)||[];
    es.forEach(([,app,i,trim])=>{if(!trim&&sets[app]) sets[app].delete(i);});
    (byStep.get(s)||[]).forEach(([app,i0,i1])=>{const x=sets[app]||(sets[app]=new Set()); for(let i=i0;i<i1;i++) x.add(i);});
    presPre.push(count());
    es.forEach(([,app,i,trim])=>{if(trim&&sets[app]) sets[app].delete(i);});
    pres.push(count());}
  /* per request: its kind, worst gap, whether it meets its goal, the index of its worst gap */
  R.forEach(r=>{r.kind=kindOf(r); r.ttft=r.first-r.arr; let g=0, gi=0; for(let i=1;i<r.toks.length;i++){const d=r.toks[i]-r.toks[i-1]; if(d>g){g=d; gi=i;}}
    r.gap=g; r.gi=gi; r.ok=r.ttft<=G[r.cls].ttft+EPS&&g<=gapGoal(r.cls)+EPS; r.se=se(r);});
  /* the worst stall of the short chats and the step that caused it */
  let worst=null; R.forEach(r=>{if(r.cls!=='chat'||!r.gi) return; if(!worst||r.gap>worst.g+1e-12) worst={g:r.gap,id:r.id,s:r.dsteps[r.gi-1]};});
  const end=Math.max(...R.map(r=>r.fin)), kinds=KORDER.filter(k=>R.some(r=>r.kind===k));
  const docs=R.filter(r=>r.cls==='doc'), chats=R.filter(r=>r.cls==='chat');
  /* GPU time used: the step times added up, the GPU's work for these requests (review r2, N1: a share of the run is 100% whenever
     requests keep the GPU going, so it cannot show saved work) */
  const t0r=Math.min(...R.map(r=>r.arr)), gpuT=st.reduce((a,x)=>a+(x.t1-x.t0),0);
  return {run,m,R,st,why,evAt,evAtT,evAtA,evTrim,evAlloc,pres,presPre,worst,end,t0r,gpuT,kinds,docs,chats,cfg,mdc:new Map()};
}
const baseOf=load=>getRun(load,BASE);
/* the time axis of a load: the same for every setting of it, so a change of setting moves the marks, not the scale */
const SPAN={};
function spanOf(load){if(SPAN[load]) return SPAN[load]; let m=0;
  [BASE,{...BASE,budget:2048},{...BASE,budget:2048,cache:true}].forEach(s=>{m=Math.max(m,getRun(load,s).end);});
  return (SPAN[load]=Math.ceil(m*2+0.2)/2);}
const spanFor=o=>Math.max(spanOf(o.load),Math.ceil(o.end*2+0.2)/2);

/* the step whose time range holds t (−1: none, the GPU is idle or the run has not started) */
function stepAt(o,t){const st=o.st; let lo=0, hi=st.length-1; if(!st.length||t<st[0].t0-EPS) return -1;
  while(lo<hi){const m=(lo+hi+1)>>1; if(st[m].t0<=t+EPS) lo=m; else hi=m-1;} return t<st[lo].t1-EPS?lo:-1;}
/* the last step that has ended by t (−1: none) */
function doneAt(o,t){const st=o.st; let lo=0, hi=st.length-1; if(!st.length||st[0].t1>t+EPS) return -1;
  while(lo<hi){const m=(lo+hi+1)>>1; if(st[m].t1<=t+EPS) lo=m; else hi=m-1;} return lo;}
/* where a request stands at t: waiting (with why), running, done; and its goal: met, missed or not known yet */
/* a step's end is the next one's start. After ▸ (edge 'end') the stage shows the step that just ended, done: its outputs, the memory
   after its releases and the queue before the next step's admissions, so the GPU box and part 7's code name the same step (review A
   of part 7, finding 1). After ◂ (edge 'start') it shows the next step starting. −1: the playhead is not on a step's end */
let codeEdge='end';
function endView(o){const t=play.t; if(t==null) return -1; const d=doneAt(o,t); if(d<0||Math.abs(o.st[d].t1-t)>=1e-7) return -1; return codeEdge==='end'||stepAt(o,t)<0?d:-1;}
/* admitted by t: at a step's end shown as ended (e ≥ 0), the next step's admissions have not happened yet */
const admittedAt=(r,t,e)=>e>=0?r.admit<t-EPS:r.admit<=t+EPS;
/* why a request waits at t (e: the step shown ended): it arrived during the step (it waits for its end), or the step's admission
   stopped before it (the lab's why[]) */
function reasonAt(o,r,t,e){const s=e>=0?e:stepAt(o,t); if(s>=0&&r.arr>o.st[s].t0+EPS) return 'step'; return s>=0?o.why[s]:o.why[Math.max(0,doneAt(o,t)+1)]||'step';}
function goalAt(r,t){
  const g=G[r.cls];
  if(r.first<=t+EPS){if(r.ttft>g.ttft+EPS) return 'miss';} else if(t-r.arr>g.ttft+EPS) return 'miss';
  const gg=gapGoal(r.cls);
  if(isFinite(gg)){let last=null, mx=0; for(const x of r.toks){if(x>t+EPS) break; if(last!=null) mx=Math.max(mx,x-last); last=x;}
    if(mx>gg+EPS) return 'miss'; if(last!=null&&r.fin>t+EPS&&t-last>gg+EPS) return 'miss';}
  return r.fin<=t+EPS?'ok':'pend';}
/* memory after step k: each running request's own blocks, and per app the shared blocks in use and how many requests use them */
function memAt(o,k){
  if(k<0) return {own:[],inuse:{},users:{},mem:{used:0,idle:0,resv:0,shared:0},pres:{}};
  if(o.mdc.has(k)) return o.mdc.get(k);
  const s=o.st[k], own=[], inuse={}, users={};
  o.R.forEach(r=>{if(!(r.admit<=s.t0+EPS&&r.fin>s.t1+EPS)) return;
    let done=r.hit, ob=0; for(const [si,a,b] of r.chunks){if(si>k) break; done=b; const oa=Math.max(a,r.se); if(b>oa) ob+=E.blocks(b-r.se)-E.blocks(oa-r.se);}
    let nd=0; for(const d of r.dsteps){if(d>k) break; nd++;}
    if(nd>0) ob+=Math.floor((r.P+nd-1)/C.BLOCK)-Math.floor((r.P-1)/C.BLOCK);
    own.push({id:r.id,b:ob,admit:r.admit});
    if(r.se){const n=Math.floor(Math.min(done,r.se)/C.BLOCK); inuse[r.app]=Math.max(inuse[r.app]||0,n); if(n>0) users[r.app]=(users[r.app]||0)+1;}});
  own.sort((x,y)=>x.admit-y.admit||x.id-y.id);
  const v={own,inuse,users,mem:o.run.mem[k],pres:o.pres[k]}; o.mdc.set(k,v); if(o.mdc.size>400) o.mdc.delete(o.mdc.keys().next().value); return v;}
/* memory DURING step k: its allocations done, before the requests that finish in it release their blocks. This is the moment the
   engine takes its peak (SC_ENGINE's run.peak), so the peak has one definition everywhere: table, plan, memory bar, Show the math */
function memPre(o,k){
  if(k<0) return memAt(o,-1);
  const key='p'+k; if(o.mdc.has(key)) return o.mdc.get(key);
  const s=o.st[k], own=[], inuse={}, users={}; let ownSum=0, resvFin=0;
  o.R.forEach(r=>{if(!(r.admit<=s.t0+EPS&&r.fin>s.t0+EPS)) return;
    let done=r.hit, ob=0; for(const [si,a,b] of r.chunks){if(si>k) break; done=b; const oa=Math.max(a,r.se); if(b>oa) ob+=E.blocks(b-r.se)-E.blocks(oa-r.se);}
    let nd=0; for(const d of r.dsteps){if(d>k) break; nd++;}
    if(nd>0) ob+=Math.floor((r.P+nd-1)/C.BLOCK)-Math.floor((r.P-1)/C.BLOCK);
    own.push({id:r.id,b:ob,admit:r.admit}); ownSum+=ob;
    if(Math.abs(r.fin-s.t1)<1e-12) resvFin+=Math.max(0,E.blocks(r.P+r.out)-r.hit/C.BLOCK-r.sharedNew-ob);
    if(r.se){const n=Math.floor(Math.min(done,r.se)/C.BLOCK); inuse[r.app]=Math.max(inuse[r.app]||0,n); if(n>0) users[r.app]=(users[r.app]||0)+1;}});
  own.sort((x,y)=>x.admit-y.admit||x.id-y.id);
  const pres=o.presPre[k]; let sh=0, idle=0; for(const a in pres){sh+=pres[a]; idle+=Math.max(0,pres[a]-(inuse[a]||0));}
  const v={own,inuse,users,mem:{used:ownSum+sh,idle,resv:o.run.mem[k].resv+resvFin,shared:sh},pres,pre:true};
  o.mdc.set(key,v); if(o.mdc.size>400) o.mdc.delete(o.mdc.keys().next().value); return v;}
/* the step during which the engine's peak is reached (the first one if several tie) */
function peakStepPre(o){if(o._pp!=null) return o._pp; let b=0, bu=-1; for(let k=0;k<o.st.length;k++){const u=memPre(o,k).mem.used; if(u>bu){bu=u; b=k;}} return (o._pp=b);}
/* what a step does: its decode tokens, its prefill chunks (the biggest first), and who gets a token at its end */
function stepInfo(o,k){const s=o.st[k], R=o.R, dec=[], pre=[];
  s.work.forEach(w=>{if(w.k==='d') dec.push(w.id); else pre.push({id:w.id,a:w.a,b:w.b,n:w.b-w.a,whole:w.a===R[w.id].hit&&w.b===R[w.id].P,last:w.b===R[w.id].P});});
  const big=pre.slice().sort((x,y)=>y.n-x.n)[0]||null;
  const logits=dec.length+pre.filter(p=>p.last).length, att=s.fl-2*C.P_LAYERS*s.T-2*C.P_HEAD*logits, kv=Math.round((s.by-C.W_READ)/C.KV_TOK);
  return {s,k,dec,pre,big,logits,att,kv,out:dec.concat(pre.filter(p=>p.last).map(p=>p.id)),first:pre.filter(p=>p.last).map(p=>p.id)};}
/* a request's waiting, in time order: [from, to, why] */
function waitSegs(o,r){const out=[], st=o.st; if(r.admit<=r.arr+EPS) return out;
  let lo=0, hi=st.length-1; while(lo<hi){const m=(lo+hi)>>1; if(st[m].t1>r.arr+EPS) hi=m; else lo=m+1;}
  let s=lo; if(st[s].t0<r.arr-EPS&&st[s].t1>r.arr+EPS){out.push([r.arr,st[s].t1,'step']); s++;}
  for(;s<st.length&&st[s].t0<r.admit-EPS;s++) if(st[s].t0+EPS>=r.arr){const w=o.why[s], p=out[out.length-1]; if(p&&p[2]===w&&Math.abs(p[1]-st[s].t0)<1e-9) p[1]=st[s].t1; else out.push([st[s].t0,st[s].t1,w]);}
  return out;}
/* the requests that took the budget in the steps a request waited for it: who went first */
function aheadOf(o,r){const tot={}; o.st.forEach((s,i)=>{if(s.t0<r.arr-EPS||s.t0>=r.admit-EPS||o.why[i]!=='budget') return; s.work.forEach(w=>{if(w.k==='p') tot[w.id]=(tot[w.id]||0)+w.b-w.a;});});
  return Object.entries(tot).sort((x,y)=>y[1]-x[1]).map(([id,n])=>({id:+id,n}));}

/* =====================================================================
   diagnosis: what limited a run, who gained, what it cost (derived from the run: the dominant wait of the requests that missed, and the
   step that caused the worst gap)
   ===================================================================== */
/* the result of an option, against the part's reference run: two numbers, no more (R6.20; review r3, finding 7) */
function cmpHead(o,ref,refEN,refZH){const a=o.m.all.ok, n=o.m.all.n, r=ref.m.all.ok;
  if(o.key===ref.key) return tx(`<p><b>${a} of ${n} requests meet their goal.</b></p>`,`<p><b>${n} 个请求里 ${a} 个达标。</b></p>`);
  return tx(`<p><b>${a} of ${n} requests meet their goal, ${a>r?'more than':a<r?'fewer than':'as many as'} with ${refEN}.</b></p>`,`<p><b>${n} 个请求里 ${a} 个达标，${a===r?'和':'比'}${refZH}${a>r?'多':a<r?'少':'一样多'}。</b></p>`);}
function diagnose(o){
  const miss=o.R.filter(r=>!r.ok), n=miss.length;
  if(!n) return {kind:'none'};
  const gapMiss=miss.filter(r=>r.gap>gapGoal(r.cls)+EPS).length, w={};
  miss.forEach(r=>Object.entries(r.waits).forEach(([k,v])=>{w[k]=(w[k]||0)+v;}));
  if(gapMiss*2>=n&&o.worst) return {kind:'long',n,gapMiss,info:stepInfo(o,o.worst.s)};
  let dom=null; Object.entries(w).forEach(([k,v])=>{if(!dom||v>w[dom]) dom=k;});
  if(dom==='budget'){
    const appsIn=o.R.some(r=>r.app&&r.pre);
    if(!o.cfg.cache&&appsIn) return {kind:'nocache',n};
    /* the steps the missed requests waited in for the budget: mostly set by reading the weights → prompts crawl */
    let mem=0, all=0; o.st.forEach((s,i)=>{if(o.why[i]!=='budget') return; if(!miss.some(r=>s.t0>=r.arr-EPS&&s.t0<r.admit-EPS)) return; all++; if(s.tm>=s.tc) mem++;});
    if(all&&mem*2>=all) return {kind:'crawl',n,budget:o.cfg.budget};
    const r0=miss.find(r=>(r.waits.budget||0)>0), ah=r0?aheadOf(o,r0):[];
    return {kind:'ahead',n,who:ah.length?o.R[ah[0].id]:null,r:r0};}
  return {kind:dom||'step',n,cap:o.cfg.cap};
}
/* a request by what it is, for text that leaves the page (the plan): "the 24K-token document", "a support chat" */
function descName(r){const k=kindOf(r); if(k==='doc') return tx(`the ${tokK(r.own)}-token document`,`${tokK(r.own)} 个 token 的长文档`);
  return k==='chat'?tx('a short chat','一个短对话'):tx(`a ${APPN[k].en} chat`,`一个${L(APPN[k])}对话`);}
/* the long step: one prompt, or several prompts at once (B4) */
const nmOf=plain=>r=>(plain||r.cls==='doc')?descName(r):rName(r);
function longWhat(o,info,plain){const nm=nmOf(plain), pre=info.pre, big=info.big, T=pre.reduce((a,p)=>a+p.n,0), many=pre.filter(p=>p.n>=512).length;
  if(!big) return tx('long steps','很长的步');
  if(many>1&&big.n*2>T){const k=pre.length-1, bn=nm(o.R[big.id]);   /* one prompt carries most of the step (review r2, N3) */
    return big.whole?tx(`a long step (${bn}’s whole prompt, plus ${wEN(k)} smaller prompt${k>1?'s':''})`,`很长的一步（${bn}的整个 prompt，外加${wZH(k)}个小一些的 prompt）`)
      :tx(`a long step (a big chunk of ${bn}, plus ${wEN(k)} smaller prompt${k>1?'s':''})`,`很长的一步（${bn}的一大块 chunk，外加${wZH(k)}个小一些的 prompt）`);}
  if(many>1) return tx(`a long step that prefills ${many} whole prompts at once`,`很长的一步，一次 prefill ${many} 个完整的 prompt`);
  return big.whole?tx(`a long step (${nm(o.R[big.id])}’s whole prompt in one step)`,`很长的一步（${nm(o.R[big.id])}的整个 prompt 在一步里算完）`)
    :tx(`a long step (a big chunk of ${nm(o.R[big.id])})`,`很长的一步（${nm(o.R[big.id])}的一大块 chunk）`);}
/* what limits a run, as a noun phrase (plain: names that make sense off the page) */
function missShort(o,plain){const d=diagnose(o), nm=nmOf(plain), docs=o.load==='apps';
  switch(d.kind){
   case 'none': return tx('no limit that makes a request miss, since every request meets its goal','没有发现让请求不达标的限制，每个请求都达标');
   case 'long': return longWhat(o,d.info,plain);
   case 'cap': return tx(`the cap of ${d.cap} running requests, so new requests wait in the queue until a place frees up`,`${d.cap} 个的运行上限，新请求只能在队列里等空出位置`);
   case 'batch': return tx('static batching, where a new request waits until the whole batch has finished','static batching，新请求要等整批都做完才能开始');
   case 'memory': return tx('KV memory, because the pool cannot yet hold the next request’s prompt and answer','KV 显存，因为显存池还放不下下一个请求的 prompt 和回答');
   case 'crawl': return tx(`a budget of only ${fint(d.budget)} tokens per step, which cuts the document’s prefill into so many small steps that most of each step goes to reading the weights`,`每步只有 ${fint(d.budget)} 个 token 的预算，长文档的 prefill 被切成很多小步，每一步大部分时间都花在读权重上`);
   case 'nocache': return docs?tx('caching turned off, so every question computes its whole document again','关掉的缓存，每个问题都要把整份文档重新算一遍'):tx('caching turned off, so every request computes its app’s system prompt again','关掉的缓存，每个请求都要把应用的 system prompt 重新算一遍');
   case 'ahead': return d.who?tx(`the token budget, because ${nm(d.who)}’s chunks take it first (first come, first served)`,`token 预算，因为${nm(d.who)}的 chunk 先把它占满了（先来先服务）`):tx('the token budget, because the requests ahead take it first','token 预算，因为排在前面的请求先把它占满了');
   default: return tx('long steps, which every running request has to wait out','很长的步，每个正在跑的请求都得等它们结束');}}
/* "in the initial setup, …" / "初始设置下是…". A run where every request meets its goal says only that: meeting the goal does not show the run
   has no limit at all, only that no limit made a request miss (GPT review of B4 v4, P2; R6.22) */
const NONE_EN='every request meets its goal, so this run shows no limit that makes a request miss', NONE_ZH='每个请求都达标，这次运行里没有发现让请求不达标的限制';
function limAt(o,plain,en,zh){return diagnose(o).kind==='none'?tx(`${en}, ${NONE_EN}`,`${zh}${NONE_ZH}`):tx(`${en}, ${missShort(o,plain)}`,`${zh}是${missShort(o,plain)}`);}
function missLine(o){const d=diagnose(o);
  if(d.kind==='none') return tx(`${NONE_EN}.`,`${NONE_ZH}。`);
  if(d.kind==='long') return tx(`the run has ${longWhat(o,d.info)}, and every running chat has to wait for that step to end.`,`这次运行里有${longWhat(o,d.info)}，每个正在跑的对话都得等这一步结束。`);
  if(d.kind==='ahead'&&d.who) return tx(`the scheduler serves requests first come, first served, so ${nmOf(false)(d.who)}’s chunks take each step’s budget first, and the chats that arrive later wait until that prefill ends.`,`调度按先来先服务，所以${nmOf(false)(d.who)}的 chunk 先占满每一步的预算，后来的对话只能等这段 prefill 结束。`);
  return missShort(o)+tx('.','。');}
/* the numbers a change can move, compared as shown (so a sentence never contradicts the table): lower is better for all of them */
function effectRows(o){const n=o.chats.length, nd=o.docs.length, dT=x=>x.docs.length?Math.max(...x.docs.map(r=>r.ttft)):null;
  return [
   {k:'p50',g:x=>x.m.chat.ttft_p50,f:fS,l:tx('short chats’ median first token (queueing included)','短对话首 token 的中位数（含排队）')},
   {k:'p95',g:x=>x.m.chat.ttft_p95,f:fS,l:tx(`short chats’ first-token p95 of ${n} (queueing included)`,`短对话首 token 的 p95（${n} 个，含排队）`)},
   {k:'gap',g:x=>x.m.chat.gap_max,f:fS,l:tx('the longest gap between tokens','最长的 token 间隔')},
   ...(nd?[{k:'doc',g:dT,f:fS,l:nd>1?tx('documents’ first token, the slower one','长文档首 token（较慢的那份）'):tx('the document’s first token','长文档的首 token')}]:[]),
   {k:'gput',g:x=>x.gpuT,f:fS,l:tx('GPU time used','GPU 用时')},
   {k:'pre',g:x=>x.m.all.prefill_tok,f:fint,l:tx('prefill tokens computed','prefill 算过的 token')},
   {k:'peak',g:x=>x.m.all.peak_gib,f:fG,l:tx('peak KV cache','KV cache 峰值')}];}
/* from a to y, as shown; when rounding would hide the change, one more digit (R6.10) */
const pairTxt=(e,a,y)=>{let A=e.f(a), Y=e.f(y); if(A===Y&&e.f===fS){A=fixd(a,3)+NB+'s'; Y=fixd(y,3)+NB+'s';} return `${A} → ${Y}`;};
function effects(w,o){const gain=[], cost=[];
  effectRows(o).forEach(e=>{const a=e.g(w), y=e.g(o); if(a==null||y==null||e.f(a)===e.f(y)) return; (y<a?gain:cost).push({e,a,y,rel:Math.abs(y-a)/Math.max(Math.abs(a),1e-12)});});
  return {gain,cost};}
function gainLine(o,b){const n=o.chats.length, cOk=o.chats.filter(r=>r.ok).length, bc=b.chats.filter(r=>r.ok).length, dOk=o.docs.filter(r=>r.ok).length, bd=b.docs.filter(r=>r.ok).length, nd=o.docs.length;
  const c=cOk===n?tx(`all ${n} short chats meet their goal`,`${n} 个短对话全部达标`):tx(`${cOk} of ${n} short chats meet their goal`,`${n} 个短对话里 ${cOk} 个达标`);
  const cb=cOk===bc?tx(' (as in the initial setup)','（和初始设置一样）'):(bc===0?tx(' (none in the initial setup)','（初始设置时一个也没有）'):tx(` (${bc} in the initial setup)`,`（初始设置时 ${bc} 个）`));
  const dw=k=>nd===1?(k?tx('the document meets its goal','长文档达标'):tx('the document misses its goal','长文档没达标')):(k===nd?tx(`${nd===2?'both':'all '+nd} documents meet theirs`,`${nd===2?'两份':nd+' 份'}长文档都达标`):(k===0?tx(`no document meets its goal`,`长文档都没达标`):tx(`${wEN(k)} of the ${wEN(nd)} documents meets its goal`,`${wZH(nd)}份长文档里${wZH(k)}份达标`)));
  const d=!nd?'':tx('; ','；')+dw(dOk)+(dOk===bd?tx(', as before',`，和原来一样`):tx(` (${wEN(bd)} before)`,`（原来是${wZH(bd)}份）`));
  return c+cb+d+tx('.','。');}
/* the cost of a change: the number that got worse the most, as shown; the cache's idle memory only when that is what grew */
/* the cost of a change as full sentences, rounded, with the table named (review B4, R4 and R8). lead: "The cost is that …" for the
   observation box; without it the sentence stands under the label "What it cost" */
const FRAC=[[1/4,'a quarter','四分之一'],[1/3,'a third','三分之一'],[1/2,'half','一半'],[2/3,'two thirds','三分之二'],[3/4,'three quarters','四分之三']];
const fracTxt=f=>{const w=FRAC.find(([v])=>Math.abs(f-v)<=0.04); return w?tx(`about ${w[1]}`,`约${w[2]}`):tx(`about ${pct0(f)}`,`约 ${pct0(f)}`);};
function costLine(o,b,lead){const ef=effects(b,o).cost.sort((x,y)=>y.rel-x.rel);
  if(!ef.length){const gs=effects(b,o).gain, g=gs.find(x=>x.e.k==='gput'&&x.y<x.a*0.97)||gs.find(x=>x.e.k==='pre'&&x.y<x.a*0.95);
    if(!g) return tx('No cost shows in these numbers.','这些数字里看不出代价。');
    const fr=fracTxt(1-g.y/g.a);
    return g.e.k==='gput'?tx(`No cost shows here. The GPU even computes much less: its total time drops by ${fr}.`,`这里看不出有什么代价。GPU 反而少算了很多，总用时少了${fr}。`)
      :tx(`No cost shows here. The GPU even computes much less: it prefills ${fr} fewer tokens.`,`这里看不出有什么代价。GPU 反而少算了很多，prefill 的 token 少了${fr}。`);}
  const x=ef[0], more=ef.length-1, nd=o.docs.length, k=x.e.k;
  const rT=n=>n>=1e4?`${Math.round(n/1000)}K`:fint(n), rf=k==='pre'?rT:k==='peak'?fG:fR;
  let A=rf(x.a), Y=rf(x.y); if(A===Y){A=x.e.f(x.a); Y=x.e.f(x.y);}   /* rounding would hide the change: the table's digits */
  const core={p50:tx('short chats wait longer for their first token (the median, queueing included)','短对话拿到首 token 的时间变长了（含排队，取中位数）'),
    p95:tx('the slowest short chats wait longer for their first token (the p95, queueing included)','最慢的那些短对话拿到首 token 的时间变长了（含排队，取 p95）'),
    gap:tx('the longest gap between two tokens grows','两个 token 之间最长的间隔变长了'),
    doc:nd>1?tx('the slower document gets its first token later','较慢的那份长文档拿到首 token 更晚了'):tx('the document gets its first token later','长文档拿到首 token 更晚了'),
    gput:tx('the GPU works longer for the same requests','同样这些请求，GPU 要多干一会儿'),
    pre:tx('the GPU prefills more tokens','GPU 要 prefill 的 token 变多了'),
    peak:tx('the KV cache peaks higher','KV cache 的峰值变高了')}[k]||x.e.l;
  const why=k==='peak'&&o.cfg.cache?tx(', because idle cached prompts stay in memory','，因为空闲的缓存 prompt 一直占着显存'):'';
  const body=tx(`${core}: from about ${A} to about ${Y}${why}.`,`${core}，从约 ${A} 变成约 ${Y}${why}。`);
  const t=lead?tx(`The cost is that ${body}`,`代价是${body}`):tx(cap1(body),body);
  return t+(more?tx(' The table below the stage shows how the other numbers change.','其他指标的变化见舞台下方的表格。'):'');}

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={load:'stall',...BASE}, mode='guided', step=0, pred=null, sel=null, root=null, app=null, speed=null, runTok=0;
let sbState=null, sbTouched=false, chCfg={...BASE}, chRunKey=null, chSubmitted=false, chReveal=false, chTier=null, drillSeed=1, drillOpen=false, whyOpen=false;
const tried={}, seen=new Set(), shownKey={v:null};
const play={t:0,playing:false,raf:0,last:0,rate:0,k:-2,dk:''};
const cur=()=>getRun(S.load,S);
const SPEEDS=[0.5,1,2], BASE_SECS=15;                    // 1× plays the whole run in about 15 s, whatever its length
/* R3.9 (product owner, 2026-10-04): a whole run of many requests plays at 2× by default (about 7.5 s); part 7's code, read step by step,
   at 1× (0.8 s a step). Once the learner picks a speed, that speed holds everywhere */
const spd=()=>speed!=null?speed:(codeOn()?1:2);
const RM=()=>A.reduceMotion();
const bump=()=>{runTok++;};                             // every state change: an old animation or timeout can never draw over the new state

/* =====================================================================
   definitions shown on hover (R2.8): a term is defined where a reader first meets it
   ===================================================================== */
const TIP={
 step:()=>tx('A step is one pass of the model over everything the server scheduled together: one new token for each running request that is decoding, and prompt tokens of the requests being prefilled. Every request in the step gets its result only when the whole step ends.',
             '一步是模型把服务端排在一起的工作跑一遍：给每个正在 decode 的请求各算一个新 token，再加上正在 prefill 的请求的 prompt token。这一步里的每个请求，都要等整步结束才拿到结果。'),
 ttft:()=>tx('Time to first token (TTFT): from the moment a request arrives to its first answer token. Here it includes the time it waited in the queue, as a user sees it.',
             '首 token 延迟（TTFT）：从请求到达到它的第一个回答 token。这里包含它在队列里等的时间，和用户感受到的一样。'),
 gap:()=>tx('The gap between tokens: how long a user waits for the next token once the answer has started. Each step gives a running request one token, so the gap is the length of the step it waits for.',
            '两个 token 之间的间隔：回答开始以后，用户等下一个 token 要多久。每一步给正在跑的请求一个 token，所以间隔就是它等的那一步的长度。'),
 budget:()=>tx('A token budget caps how many tokens one step may compute, decode tokens included (vLLM’s max_num_batched_tokens). A prompt longer than what is left is prefilled in chunks, one per step.',
               'token 预算限制一步最多算多少个 token，decode 的 token 也算在内（vLLM 的 max_num_batched_tokens）。prompt 比剩下的预算长时，就切成 chunk，每步 prefill 一个。'),
 cap:()=>tx('The cap on running requests: how many requests the server keeps in its batch at once (vLLM’s max_num_seqs). A request that finds the batch full waits in the queue.',
            '运行请求上限：服务端的 batch 里最多同时放多少个请求（vLLM 的 max_num_seqs）。batch 满了，新请求就在队列里等。'),
 cache:()=>tx('Prefix caching keeps the KV cache of a prompt’s leading blocks after its request ends. A later request whose prompt starts with the same tokens reuses those blocks instead of prefilling them again.',
              'prefix caching（前缀缓存）在请求结束后留下 prompt 开头那些 block 的 KV cache。后来的请求 prompt 开头一样时，直接复用这些 block，不用再 prefill 一遍。'),
 block:()=>{const kib=fint(C.KV_TOK/1024), mib=fint(BLK_B/1048576); return tx(`A block is ${C.BLOCK} tokens of KV cache, ${C.BLOCK} × ${kib} KiB = ${mib} MiB for ${MODEL}. Memory is handed out in whole blocks.`,`一个 block 是 ${C.BLOCK} 个 token 的 KV cache，${MODEL} 是 ${C.BLOCK} × ${kib} KiB = ${mib} MiB。显存按整个 block 分配。`);},
 p95:nn=>{const n=nn||LF('stall').n, k=Math.max(1,Math.ceil(0.95*n-1e-9)), out=n-k, ord=k+(k%100>10&&k%100<14?'th':({1:'st',2:'nd',3:'rd'})[k%10]||'th');
   return tx(`p95 is the value that 95% of the requests stay at or below: sort the values and take the one at rank ⌈0.95 × n⌉. With ${n} short chats it is the ${ord}, so ${out===1?'only the slowest does':`the ${out} slowest do`} not count.`,
             `p95 是 95% 的请求不超过的值：把值排好序，取第 ⌈0.95 × n⌉ 个。${n} 个短对话时取第 ${k} 个，${out===1?'只有最慢的那一个':`最慢的 ${out} 个`}不算。`);},
 continuous:()=>tx('Continuous batching: the server rebuilds the batch at every step, so a new request joins as soon as there is room, and a finished one leaves at once.','continuous batching：服务端每一步都重新组 batch，有空位新请求就加入，做完的请求马上离开。'),
 static:()=>tx('Static batching: a batch starts only when the previous one has finished entirely, so a new request waits for the whole batch.','static batching：上一批全部做完，下一批才开始，所以新请求要等整批。')
};

/* =====================================================================
   the goals (locked, the same everywhere) and the guided parts
   ===================================================================== */
const goalCard=()=>`<div class="sc-goalcard" role="group" aria-label="${esc(tx('The goals, locked','目标（已锁定）'))}"><div class="sc-gc-h"><span class="sc-lock" aria-hidden="true">${ICON.lock}</span>${tx('The goals, locked before any tuning','目标（调参之前就锁定）')}</div>`+
  `<ul><li><i class="sc-sw" style="background:var(--sc-chat)"></i><span>${tx(`<b>Short chat:</b> first token within ${fS(G.chat.ttft)} of arriving (${DEF('queueing included',TIP.ttft())}), and no ${DEF('gap',TIP.gap())} over ${fS(G.chat.gap)} between tokens.`,`<b>短对话：</b>到达后 ${fS(G.chat.ttft)} 内出首 token（${DEF('含排队',TIP.ttft())}），两个 token 之间的${DEF('间隔',TIP.gap())}不超过 ${fS(G.chat.gap)}。`)}</span></li>`+
  `<li><i class="sc-sw" style="background:var(--sc-doc)"></i><span>${tx(`<b>Long document:</b> first token within ${fS(G.doc.ttft)} of arriving.`,`<b>长文档：</b>到达后 ${fS(G.doc.ttft)} 内出首 token。`)}</span></li>`+
  `<li><i class="sc-sw sc-sw-ok"></i><span>${tx(`<b>The service</b> meets its goal when at least ${pct0(E.TARGET)} of all requests meet theirs.`,`<b>整个服务</b>达标：至少 ${pct0(E.TARGET)} 的请求各自达标。`)}</span></li></ul></div>`;
const PRED=[{v:'a',t:{en:'KV memory filled up, so running chats were pushed out and had to wait.',zh:'KV cache 显存满了，正在跑的对话被挤出去，只能等。'}},
  {v:'b',t:{en:`The ${tokK(LF('stall').docs[0].own)} document was prefilled in one step, so every chat had to wait for that step to end.`,zh:`${tokK(LF('stall').docs[0].own)} 的文档在一步里 prefill 完，每个对话都只能等这一步结束。`}},
  {v:'c',t:{en:`The cap of ${BASE.cap} running requests was reached, so chats waited in the queue.`,zh:`同时运行的请求到了 ${BASE.cap} 个的上限，对话在队列里等。`}},
  {v:'d',t:{en:'The network slowed down, so tokens reached the chats late.',zh:'网络变慢了，token 很晚才到对话那里。'}}];
const PRED_RIGHT='b';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const revealed=()=>mode!=='guided'||step>=2||!!pred;        // the stall's cause is named only after the prediction (R3.8)
const PART_CFG={mission:{load:'stall',...BASE},predict:{load:'stall',...BASE},budget:{load:'stall',...BASE},cache:{load:'shared',...BASE,budget:2048},
  cbudget:{load:'apps',...BASE,budget:2048,cache:true,cb:1536},cap:{load:'shared',...BASE,budget:2048,cache:true},code:{load:'stall',...BASE,budget:2048},remember:{load:'stall',...BASE,budget:2048}};
const docOf=load=>spec(load).find(x=>x.cls==='doc');
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'tl',
  lead:()=>{const d=docOf('stall'), n=LF('stall').n; return tx(`Peak hour: ${n} short chats, then a ${tokK(d.own)}-token document at ${fixd(d.t,1)}${NB}s.`,`高峰期：${n} 个短对话，${fixd(d.t,1)}${NB}s 时来了一份 ${tokK(d.own)} 个 token 的文档。`);},
  body:()=>{const d=docOf('stall'), n=LF('stall').n, last=fixd(Math.max(...spec('stall').filter(x=>x.cls==='chat').map(x=>x.t)),0);
   return tx(`<p>One ${GPUN} serves ${MODEL} to a chat app. Over about ${last}${NB}s, ${n} short chats arrive, and at ${fixd(d.t,1)}${NB}s a user sends a ${tokK(d.own)}-token document to summarize. Every request first waits in the queue. When its turn comes, the server computes its whole prompt (the prefill) in a ${DEF('step',TIP.step())}, and after that the request gets one more token in every step.</p>`,
             `<p>一张 ${GPUN} 给一个聊天应用跑 ${MODEL}。约 ${last}${NB}s 内陆续来了 ${n} 个短对话，${fixd(d.t,1)}${NB}s 时，一个用户发来一份 ${tokK(d.own)} 个 token 的文档要写摘要。每个请求先在队列里等；轮到它时，服务端在某一${DEF('步',TIP.step())}里算完它的整个 prompt（这叫 prefill），之后这个请求每一步多拿到一个 token。</p>`)+goalCard()+
    `<div class="inline-ctl sc-ic-play" id="sc-ic-play"><span class="lbl">${tx('Play today’s service','运行今天的服务')}</span><div class="sc-prow">${playBtn('sc-pb-card')}</div></div>`;},
  obs:()=>{const o=cur(); return tx(`<p><b>Only ${o.m.all.ok} of ${o.m.all.n} requests meet their goal. The service needs at least nine in ten to meet it.</b></p><p>At ${fixd(docOf('stall').t,1)}${NB}s every chat stops getting tokens at the same moment. The timeline marks the longest gap between two tokens of a chat.</p>`,
    `<p><b>${o.m.all.n} 个请求里只有 ${o.m.all.ok} 个达标，而服务要求至少九成达标。</b></p><p>${fixd(docOf('stall').t,1)}${NB}s 时，所有对话在同一刻都拿不到 token 了。时间线上标出了对话两个 token 之间最长的间隔。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'tl',gate:()=>!!pred,
  lead:()=>{const w=baseOf('stall').worst, at=fixd(docOf('stall').t,1)+NB+'s'; return tx(`At ${at} every chat stopped for about ${fR(w.g)}. Why?`,`${at} 时，所有对话停了约 ${fR(w.g)}。为什么？`);},
  body:()=>{const w=baseOf('stall').worst, at=fixd(docOf('stall').t,1)+NB+'s', ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>At ${at} every chat stopped getting tokens for about ${fR(w.g)}, and that is the longest gap on the timeline. What caused it?</p>${ch}`,
             `<p>${at} 时，所有对话大约 ${fR(w.g)} 拿不到 token，这是时间线上最长的间隔。是什么造成的？</p>${ch}`);},
  obs:()=>{if(!pred) return ''; const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
   const v=pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, is right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
     :tx(`<p>You predicted “${pt}” The right answer is “${rt}”</p>`,`<p>你预测的是“${pt}”正确答案是“${rt}”</p>`);
   return v+tx(`<p>The server hands out the tokens a step produces only when all of the step’s work is done. The server put the document’s whole prompt into one step, and every running chat’s next token was in that same step, so every chat had to wait until the document’s prefill was finished. ${J('stall','Show me that step')}</p>`,
               `<p>服务端要等一步里的工作全部做完，才把这一步产生的 token 交出去。服务端把文档的整个 prompt 放进了一步，每个正在跑的对话的下一个 token 也在这一步里，所以每个对话都要等文档的 prefill 算完才能拿到下一个 token。${J('stall','给我看这一步')}</p>`);}},
 {id:'budget',title:{en:'Cut it into chunks',zh:'切成 chunk'},focus:'tl',lever:'budget',opts:BUDS,showMe:{budget:2048},expect:()=>S.budget!==0,label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('A token budget cuts the document’s prefill into short steps, but a budget that is too small makes the prefill slow.','token 预算把文档的 prefill 切成短步；但预算太小，prefill 又会变得很慢。'),
  body:()=>tx(`<p>A ${DEF('token budget',TIP.budget())} caps how many tokens one step may compute. With a budget, the server splits the document’s prompt into chunks and puts one chunk into each step, next to the chats’ decode tokens.</p>`,
              `<p>${DEF('token 预算',TIP.budget())}限制一步最多算多少个 token。设了预算，服务端就把文档的 prompt 切成一个个 chunk，每一步放一个 chunk，和对话的 decode token 一起算。</p>`)+leversHTML('budget')},
 {id:'cache',title:{en:`One system prompt, ${LF('shared').n} times`,zh:`同一个 system prompt，${LF('shared').n} 遍`},focus:'mem',lever:'cache',opts:[false,true],showMe:{cache:true},expect:()=>!!S.cache,label:{en:'Observed',zh:'观察到'},
  lead:()=>{const f=LF('shared'); return tx(`${f.n} support chats all start with the same ${fint(f.pre)}-token system prompt. Can the server compute and keep it only once?`,`${f.n} 个客服对话都以同一个 ${fint(f.pre)} 个 token 的 system prompt 开头。服务端能不能只算一次、只存一份？`);},
  body:()=>{const p=spec('shared')[0].pre;
   return tx(`<p>In a support app, every request starts with the same ${fint(p)}-token system prompt, and the budget is now ${fint(PART_CFG.cache.budget)} tokens per step. Without ${DEF('prefix caching',TIP.cache())}, the server prefills that prompt again for every request and keeps one copy of it per request in memory.</p>`,
             `<p>在一个客服应用里，每个请求都以同一个 ${fint(p)} 个 token 的 system prompt 开头，预算现在是每步 ${fint(PART_CFG.cache.budget)} 个 token。不开 ${DEF('prefix caching',TIP.cache())} 时，服务端要为每个请求重新 prefill 这个 prompt，并在显存里为每个请求各存一份。</p>`)+leversHTML('cache');}},
 {id:'cbudget',title:{en:'When the cache is too small',zh:'缓存不够大时'},focus:'mem',lever:'cb',opts:CBS5,showMe:{cb:6144},expect:()=>S.cb!==1536,label:{en:'Observed',zh:'观察到'},
  lead:()=>{const f=LF('apps'), k=f.apps.length; return tx(`Questions about ${wEN(k)} long documents, and a cache that holds only some of them.`,`用户在问${wZH(k)}份长文档的问题，而缓存只放得下其中几份。`);},
  body:()=>{const f=LF('apps'), k=f.apps.length, nm=f.apps.map(a=>L(KIND[a]));
   return tx(`<p>Users ask ${f.n} questions about ${wEN(k)} long documents (${nm.join(', ')}), and each question starts with its document’s ${fint(f.pre)} tokens. After a question is answered, the document it asked about stays in the cache. When the idle documents together take more than the cache budget, the server evicts the one that was used least recently; after that, a question about that document has to prefill the whole document again. Real engines have no separate setting for this: idle prefixes use whatever KV memory is free, so the budget here stands for how much memory is free.</p>`,
             `<p>用户对${wZH(k)}份长文档（${nm.join('、')}）一共提了 ${f.n} 个问题，每个问题都以它所问文档的 ${fint(f.pre)} 个 token 开头。一个问题答完以后，它问的那份文档留在缓存里。空闲的文档加起来超过缓存预算时，服务端就淘汰最久没用过的那一份；之后再有人问这份文档，就只能把整份文档重新 prefill 一遍。真实引擎没有单独的这项设置：空闲的前缀直接用空闲的 KV 显存，所以这里的预算代表空闲的显存有多少。</p>`)+leversHTML('cb');}},
 {id:'cap',title:{en:'How many at once',zh:'同时跑多少个'},focus:'strip',lever:'cap',opts:CAPS,showMe:{cap:8},expect:()=>S.cap!==32||S.policy!=='continuous',label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('The cap limits how many requests run at once; if it is too low, new requests wait in the queue.','上限限制同时跑多少个请求；上限太低，新请求就只能在队列里等。'),
  body:()=>tx(`<p>Back to the support app, with caching on and a ${fint(PART_CFG.cap.budget)}-token budget. The ${DEF('cap on running requests',TIP.cap())} limits how many requests share each step, and a request that finds the batch full waits in the queue.</p>`,
              `<p>回到客服应用，缓存打开，预算每步 ${fint(PART_CFG.cap.budget)} 个 token。${DEF('运行请求上限',TIP.cap())}限制每一步最多有几个请求一起算；新请求来的时候如果 batch 已经满了，它就在队列里等。</p>`)+leversHTML('cap')+
    A.ui.seg({id:'sc-ic-pol',key:'policy',label:tx('Batching','批处理方式'),options:[['continuous','continuous'],['static','static']],value:S.policy,inline:true,
      note:tx(`${DEF('Continuous',TIP.continuous())}: a new request joins at the next step. ${DEF('Static',TIP.static())}: a new request waits until the whole batch has finished.`,`${DEF('continuous',TIP.continuous())}：新请求在下一步就能加入。${DEF('static',TIP.static())}：新请求要等整批做完。`)})+`<p class="small sc-polres" id="sc-polres" aria-live="polite"></p>`},
 {id:'code',title:{en:'The scheduler, step by step',zh:'调度器的每一步'},focus:'code',expect:()=>codeSeen.size>0,label:{en:'Observed',zh:'观察到'},
  lead:()=>tx('The stalls, the queues and the cache hits all come from one scheduler loop. Watch which lines each step runs.','卡顿、排队和缓存命中，都来自同一个调度循环。看看每一步跑了哪几行。'),
  body:()=>tx(`<p>The code on the stage is the scheduler this lab runs, once per ${DEF('step',TIP.step())}. Pick a moment, then press ▸ to see the step end, and again for the next one.</p>`,
              `<p>舞台上的代码就是这个 Lab 用的调度器，每一${DEF('步',TIP.step())}跑一遍。选一个时刻，然后按 ▸ 看这一步结束，再按一次就到下一步。</p>`)+momentsHTML()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('Three rules for a congested service, and three traps.','拥堵服务的三条规律，和三个误区。'),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
/* the options of a part: one lever each. The card shows only the result the learner cares about, how many requests meet their goal, under
   the title so the title never gets squeezed (R3.17, R7.8); every other number is in the table and on the stage */
const metTxt=o=>tx(`${o.m.all.ok} / ${o.m.all.n} meet the goal`,`${o.m.all.ok} / ${o.m.all.n} 达标`);
const LEVER={
 budget:{aria:{en:'Token budget per step, pick one',zh:'每步的 token 预算，选一个'},
   lead:{en:'Pick a budget and see how many requests meet their goal. Each option starts from the initial setup.',zh:'选一个预算，看看有多少个请求达标。每个选项都从初始设置出发。'},
   title:v=>v?tx(`Budget ${fint(v)}`,`预算 ${fint(v)}`):tx('No budget','不设预算'),
   desc:v=>{const o=getRun('stall',{...BASE,budget:v}), d=o.docs[0]; return v?tx(`The document is prefilled in ${fint(d.chunks.length)} steps`,`文档分 ${fint(d.chunks.length)} 步 prefill`):tx('The whole prompt in one step (initial setup)','整个 prompt 一步算完（初始设置）');},
   res:v=>esc(metTxt(getRun('stall',{...BASE,budget:v}))),
   set:v=>({budget:v}), val:()=>S.budget},
 cache:{aria:{en:'Prefix caching, pick one',zh:'前缀缓存，选一种'},
   lead:{en:'Caching off or on? Pick one and see how many requests meet their goal.',zh:'关闭还是打开缓存？选一种，看看有多少个请求达标。'},
   title:v=>v?tx('Caching on','打开缓存'):tx('Caching off','关闭缓存'),
   desc:v=>v?tx('The first request computes the prompt, and later requests reuse it','第一个请求算出 prompt，后来的请求直接复用'):tx('Every request prefills the prompt again (initial setup of this part)','每个请求都重新 prefill 这个 prompt（这一部分的初始设置）'),
   res:v=>esc(metTxt(getRun('shared',{...BASE,budget:2048,cache:v}))),
   set:v=>({cache:v,cb:null}), val:()=>!!S.cache},
 cb:{aria:{en:'Cache budget, pick one',zh:'缓存预算，选一个'},
   lead:{en:'Pick a cache budget and see how many requests meet their goal.',zh:'选一个缓存预算，看看有多少个请求达标。'},
   title:v=>v==null?tx('No limit','不设上限'):fCB(v),
   desc:v=>{const n=v==null?null:Math.round(v/(LF('apps').pre/C.BLOCK)); return v==null?tx('Every idle document stays cached','空闲的文档全部留在缓存里'):tx(`About ${n} document${n>1?'s':''}${v===PART_CFG.cbudget.cb?' (initial setup of this part)':''}`,`约 ${n} 份文档${v===PART_CFG.cbudget.cb?'（这一部分的初始设置）':''}`);},
   res:v=>esc(metTxt(getRun('apps',{...BASE,budget:2048,cache:true,cb:v}))),
   set:v=>({cb:v}), val:()=>S.cb},
 cap:{aria:{en:'Cap on running requests, pick one',zh:'运行请求上限，选一个'},
   lead:{en:'Pick a cap on running requests and see how many requests meet their goal.',zh:'选一个运行请求上限，看看有多少个请求达标。'},
   title:v=>tx(`Cap ${v}`,`上限 ${v}`),
   desc:v=>({8:tx('A quarter of the initial cap, so each step is small, but more requests queue at the cap','是初始上限的四分之一，所以每一步都小，但更多请求要在上限处排队'),16:tx('Half the initial cap, so each step is smaller, but more requests queue at the cap','是初始上限的一半，所以每一步小一些，但更多请求要在上限处排队'),32:tx('Initial setup','初始设置'),
     64:tx('Twice the initial cap, so steps get longer if the load fills it','初始上限的两倍，负载填满它时每一步更长'),128:tx('Four times the initial cap, so it takes more KV memory if the load fills it','初始上限的四倍，负载填满它时占的 KV 显存更多')})[v],
   res:v=>esc(metTxt(getRun('shared',{...BASE,budget:2048,cache:true,cap:v}))),
   set:v=>({cap:v,policy:'continuous'}), val:()=>S.policy==='static'?null:S.cap}
};
function partKeyFor(id,v){const p={...PART_CFG[id]}; return keyOf(p.load,{...p,...LEVER[STEPS[SIX(id)].lever].set(v)});}
function leversHTML(k){const Lv=LEVER[k], id={budget:'budget',cache:'cache',cb:'cbudget',cap:'cap'}[k], cfg=PART_CFG[id];
  return `<p class="sc-lvlead">${esc(L(Lv.lead))}</p>`+
   A.ui.levers({aria:L(Lv.aria),value:String(Lv.val()),options:STEPS[SIX(id)].opts.map(v=>({v:String(v),title:Lv.title(v),desc:Lv.desc(v),result:seen.has(partKeyFor(id,v))?Lv.res(v):''}))});}
const leverVal=(k,s)=>k==='budget'?+s:(k==='cache'?s==='true':(k==='cb'?(s==='null'?null:+s):+s));

/* the observations: the result first, then why in full sentences (R6.21), then what it cost (R3.10, R6.20) */
function obsBudget(){const o=cur(), b=baseOf('stall');
  if(!S.budget) return tx('<p>Without a budget, the server prefills the document’s whole prompt in one step, and that step is the stall on the timeline. Pick a budget.</p>','<p>不设预算时，服务端在一步里算完文档的整个 prompt，这一步就是时间线上的那次卡顿。选一个预算试试。</p>');
  const head=cmpHead(o,b,'the initial setup','初始设置时');
  const d=diagnose(o), gap=o.m.chat.gap_max, dc=o.docs[0];
  let why;
  if(d.kind==='crawl') why=tx(`<p>With a budget this small, the document’s prefill needs hundreds of steps. Every step reads all the weights once but computes only a small piece of the prompt, so most of its time goes to that read. As a result, no chat stalls any more, but the document and the requests queued behind it get their first token late.</p>`,
    `<p>预算这么小，文档的 prefill 要跑几百步才能算完。每一步都要把权重完整读一遍，却只算 prompt 的一小段，所以大部分时间花在读权重上。结果对话不再卡顿，但文档和排在它后面的请求都要等很久才拿到首 token。</p>`);
  else if(gap>G.chat.gap+EPS) why=tx(`<p>The stall is shorter, but it is still too long. Each step still holds a large chunk of the document, so every running chat waits about ${fR(gap)} for such a step to end before it gets its next token.</p>`,`<p>卡顿变短了，但还是太长。每一步里仍然有一大块文档要算，所以每个正在跑的对话都要等这样的一步结束，才能拿到下一个 token，大约要等 ${fR(gap)}。</p>`);
  else why=tx('<p>The stall is gone. Every step is now short, so every running chat stays within its gap goal.</p>','<p>卡顿消失了。现在每一步都很短，所以正在跑的对话都没有超过间隔目标。</p>');
  const miss=o.chats.filter(r=>!r.ok), ahead=miss.filter(r=>(r.waits.budget||0)>0&&r.gap<=G.chat.gap+EPS);
  let who='';
  if(d.kind==='ahead'&&ahead.length) who=tx(`<p>${cap1(wEN(ahead.length))} chat${ahead.length>1?'s':''} still miss${ahead.length>1?'':'es'} the goal. ${ahead.length>1?'They arrive':'It arrives'} during the document’s prefill. The scheduler serves requests first come, first served, so the document’s chunks fill each step’s budget, and ${ahead.length>1?'these chats start':'this chat starts'} only when the prefill ends. Once the document is decoding, it takes one token per step and blocks nobody. <button type="button" class="linkbtn" data-sel="${ahead[0].id}">See why</button></p>`,
    `<p>还有${wZH(ahead.length)}个对话没达标。${ahead.length>1?'它们':'它'}在文档 prefill 的时候到达。调度器先来先服务，文档的 chunk 占满了每一步的预算，所以${ahead.length>1?'这些对话':'这个对话'}要等文档的 prefill 算完才能开始。文档一进入 decode，每步只占一个 token，就不再挡住谁了。<button type="button" class="linkbtn" data-sel="${ahead[0].id}">看原因</button></p>`);
  /* the trade-off, both ends joined in one conclusion (R6.21), behind an expander so the box fits one screen (GPT review of B4 v4, P3; R3.10) */
  const both=`<details class="sc-why"><summary>${tx('Why a middle budget is best','为什么中间的预算最好')}</summary>`+tx(`<p>The budget is a trade-off. If it is too large, one step holds a long chunk of the document, and every chat that is already answering stalls until that step ends. If it is too small, the document’s prefill needs hundreds of steps that spend most of their time reading the weights, so the document and the requests behind it get their first token late. A middle budget keeps every step short while the prefill still moves fast, so it gives the best result.</p>`,
    `<p>预算是一个取舍。预算太大，一步里装着一大块文档，已经在回答的对话都要等这一步结束，于是它们都会卡住一下。预算太小，文档的 prefill 要跑几百步，每一步大部分时间在读权重，于是文档和排在它后面的请求都很晚才拿到首 token。中间的预算既让每一步都短，又让 prefill 走得够快，所以结果最好。</p>`)+`</details>`;
  const cost=`<p class="small">${costLine(o,b,true)}</p>`;
  void dc; return head+why+who+cost+both;}
function obsCache(){const o=cur(), off=getRun('shared',{...BASE,budget:2048}), p=spec('shared')[0].pre;
  if(!S.cache) return tx(`<p><b>Caching off: ${o.m.all.ok} of ${o.m.all.n} requests meet their goal.</b> The server prefills the same ${fint(p)}-token prompt again for every request, so the GPU cannot keep up and the queue keeps growing. Memory also holds one copy of the prompt per request. Turn caching on.</p>`,
                         `<p><b>关闭缓存：${o.m.all.n} 个请求里 ${o.m.all.ok} 个达标。</b>服务端为每个请求把同一个 ${fint(p)} 个 token 的 prompt 重新 prefill 一遍，所以 GPU 跟不上，队列越排越长。显存里也为每个请求各存一份 prompt。打开缓存试试。</p>`);
  return cmpHead(o,off,'caching off','关闭缓存时')+tx(`<p>The first request computes the prompt, and every later request finds it in the cache, so a later request’s prefill bar shrinks to its own few hundred tokens. Memory keeps one shared copy of the prompt, marked ×N in the bar, instead of one copy per request.</p><p class="small">${costLine(o,off,true)}</p>`,
            `<p>第一个请求算出 prompt，之后的每个请求都在缓存里找到它，所以后来的请求的 prefill 条缩短到只剩它自己的几百个 token。显存里只存一份共享的 prompt，显存条上标着 ×N，不再为每个请求各存一份。</p><p class="small">${costLine(o,off,true)}</p>`);}
function obsCb(){const o=cur(), f=LF('apps'), k=f.apps.length, lo=getRun('apps',{...BASE,budget:2048,cache:true,cb:PART_CFG.cbudget.cb});
  const name=S.cb==null?tx('No limit','不设上限'):fCB(S.cb), fit=S.cb==null?k:Math.min(k,Math.round(S.cb/(f.pre/C.BLOCK)));
  const head=cmpHead(o,lo,'this part’s initial setup','这一部分的初始设置');
  const missFirst=o.R.filter(r=>!r.ok).every(r=>r.hit===0);
  const why=fit<k?tx(`<p>With ${name} the cache holds about ${wEN(fit)} idle document${fit>1?'s':''}, so a question about another document often finds its document evicted. The server then prefills that whole document again, so the question gets its first token only after more than a second, and the requests queued behind it wait too.</p>`,
                     `<p>${name} 的缓存只放得下大约${wZH(fit)}份空闲的文档，所以问别的文档的问题常常发现文档已经被淘汰了。服务端只能把整份文档重新 prefill 一遍，这个问题要一秒多才拿到首 token，排在它后面的请求也跟着等。</p>`)
    :tx(`<p>All ${wEN(k)} documents fit, so after each document’s first question, every later question finds its document in the cache.${missFirst?` The requests that still miss are the first questions about each document: when one arrives, its document is not in the cache yet, so the server prefills all ${tokK(f.pre)} tokens from scratch, and the first token takes more than a second. No cache can help a first question, so more budget helps nothing.`:''}</p>`,
        `<p>${wZH(k)}份文档都放得下，所以每份文档的第一个问题之后，后来的问题都能在缓存里找到文档。${missFirst?`还没达标的只有每份文档的第一个问题：它来的时候缓存里还没有这份文档，只能把 ${tokK(f.pre)} 个 token 从头 prefill，首 token 要一秒多。缓存帮不了第一次，所以预算再大也没有用。`:''}</p>`);
  return head+why+tx(`<p class="small">The cost is that a bigger cache keeps more idle documents in KV memory, but here that memory would be free anyway.</p>`,`<p class="small">代价是缓存越大，留在 KV 显存里的空闲文档就越多，不过这里这些显存本来就空着。</p>`);}
function obsCap(){const o=cur(), b=getRun('shared',{...BASE,budget:2048,cache:true}), co=getRun('shared',{...BASE,budget:2048,cache:true,cap:S.cap});
  const vs=o.m.all.ok<co.m.all.ok?tx('fewer than','比'):o.m.all.ok>co.m.all.ok?tx('more than','比'):tx('as many as','和'), vz=o.m.all.ok<co.m.all.ok?'少':o.m.all.ok>co.m.all.ok?'多':'一样多';
  const head=S.policy==='static'?tx(`<p><b>Static batching: ${o.m.all.ok} of ${o.m.all.n} requests meet their goal, ${vs} with continuous batching at the same cap.</b></p>`,`<p><b>static batching：${o.m.all.n} 个请求里 ${o.m.all.ok} 个达标，${vs}同一上限下的 continuous batching ${vz}。</b></p>`)
    :cmpHead(o,b,'the initial cap','初始上限时');
  const d=diagnose(o); let why;
  if(S.policy==='static') why=tx('<p>With static batching, a new request has to wait until every request in the current batch has finished, even when most of them are already done, so new requests get their first token late.</p>','<p>用 static batching 时，新请求要等当前这一批的每个请求都做完才能开始，哪怕这一批大部分已经做完了，所以新请求很晚才拿到首 token。</p>');
  else if(d.kind==='cap') why=tx(`<p>The requests in the queue wait because the batch is at its cap, not because of memory or the budget; the chips in the queue are marked “cap”. Each extra running request adds only a little time to a step, so raising the cap costs little here.</p>`,`<p>队列里的请求在等，是因为 batch 到了上限，而不是因为显存或预算；队列里的方块上标着“上限”。每一步多跑一个请求，这一步只多花一点时间，所以这里调高上限的代价很小。</p>`);
  else {const mx=Math.max(...o.st.map(x=>x.nrun)); why=S.cap>BASE.cap?tx(`<p>Nothing changes. This load never has more than ${mx} requests running at once, so the initial cap of ${BASE.cap} was already enough, and memory is not the limit here either.</p>`,`<p>没有变化。这个负载同时在跑的请求从没超过 ${mx} 个，所以初始上限 ${BASE.cap} 已经够了，这里显存也不是限制。</p>`)
    :tx(`<p>No request waits at the cap, because this load never has more than ${mx} requests running at once.</p>`,`<p>没有请求卡在上限，因为这个负载同时在跑的请求从没超过 ${mx} 个。</p>`);}
  return head+why;}
function remember(){
  const b128=getRun('stall',{...BASE,budget:BMIN});
  const pics=[['stall',tx('<b>One step, one wait.</b> A step hands out its tokens only when all of its work is done, so one huge prefill makes every running chat wait. A token budget cuts that prefill into chunks, so every step stays short.','<b>一步，一起等。</b>一步要等所有工作做完才交出 token，所以一次巨大的 prefill 会让每个正在跑的对话都等着。token 预算把这次 prefill 切成 chunk，于是每一步都很短。')],
    ['share',tx('<b>Compute it once, keep it once.</b> When requests share a prompt, the server needs to prefill it only once and keep one copy in memory. But a cache pays off only for prompts that come back while they are still cached: a document that is evicted before its next question has to be computed again.','<b>算一次，存一份。</b>请求共用一段 prompt 时，服务端只需要 prefill 一次，在显存里存一份。但缓存只对“还缓存着的时候又被用到”的 prompt 有用：一份文档如果在下一个问题来之前就被淘汰了，就只能重新算。')],
    ['queue',tx('<b>Find out what a waiting request waits for, then change that setting.</b> Each waiting request’s trace says what it waits for: the token budget, KV memory, or the cap on running requests. If it waits for the token budget, change the budget; with a higher cap it would wait just as long. Only when a trace says that the running requests have reached the cap is the cap the setting to raise.','<b>先查清请求在等什么，再调对应的那个设置。</b>每个排队的请求都记着它在等什么，是 token 预算、KV 显存，还是运行请求上限。如果它在等 token 预算，该调的就是预算。这时把上限调高，它照样要等那么久。只有当记录写着“同时运行的请求到了上限”，才该调高上限。')]];
  const REMPIC=REMPIC_();
  return `<div class="sc-rem">${pics.map(([k,t])=>`<div class="sc-rem-i"><span class="sc-rem-pic" aria-hidden="true">${REMPIC[k]}</span><p>${t}</p></div>`).join('')}</div>`+
   `<div class="obs"><div class="eyebrow">${tx('Common misconceptions','常见误解')}</div>`+
   tx(`<p>✗ “Push every setting to its extreme.” Take the finest chunks, a budget of only ${fint(BMIN)} tokens: then only ${b128.m.all.ok} of ${b128.m.all.n} requests meet their goal, because the document’s prefill needs hundreds of short steps, and each step spends most of its time reading the weights. The budget works best in the middle, where every step stays short and the prefill still moves fast.</p><p>✗ “A cache hit makes a request free.” The hit skips only the shared prompt; the request’s own tokens and its whole answer still take steps.</p><p>✗ “The p50 tells you whether chats are smooth.” One long step makes every running chat wait at the same time, but the median hides that; the longest gap shows it.</p>`,
      `<p>✗ “每个设置都推到最极端就最好。”比如把 chunk 切到最细，预算只有 ${fint(BMIN)}：这时 ${b128.m.all.n} 个请求里只有 ${b128.m.all.ok} 个达标，因为文档的 prefill 要跑几百个短步，每一步的大部分时间都花在读权重上。预算取中间最好：每一步都短，prefill 又走得够快。</p><p>✗ “缓存命中的请求就不花时间了。”命中只省掉共享的那段 prompt；请求自己的 token 和整个回答照样要一步步算。</p><p>✗ “看 p50 就知道对话顺不顺。”一次很长的步会让所有正在跑的对话同时等，但中位数看不出来；最长的间隔才看得出来。</p>`)+
   `</div>${A.shell.toScenarios()}`;
}

/* =====================================================================
   icons (drawn for this lab: 10 × 10, stroke = currentColor)
   ===================================================================== */
const ic=(p,w)=>`<svg viewBox="0 0 10 10" width="${w||10}" height="${w||10}" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
const ICON={
 cap:ic('<rect x="1.2" y="2" width="7.6" height="6" rx="1"/><path d="M3.2 3.8v2.4M5 3.8v2.4M6.8 3.8v2.4"/>'),
 budget:ic('<path d="M7.8 2H2.4L5.4 5 2.4 8h5.4"/>'),
 memory:ic('<rect x="2.2" y="2.2" width="5.6" height="5.6" rx=".6"/><path d="M1 3.8h1.2M1 6.2h1.2M7.8 3.8H9M7.8 6.2H9"/>'),
 step:ic('<circle cx="5" cy="5" r="3.7"/><path d="M5 2.9v2.3l1.6 1"/>'),
 batch:ic('<path d="M2.6 1.4h4.8M2.6 8.6h4.8M3.1 1.4c0 2.6 3.8 2.6 3.8 7.2M6.9 1.4c0 2.6-3.8 2.6-3.8 7.2"/>'),
 lock:ic('<rect x="2" y="4.4" width="6" height="4.4" rx=".8"/><path d="M3.4 4.4V3.2a1.6 1.6 0 0 1 3.2 0v1.2"/>',11)
};
const WHY={step:{en:'step end',zh:'等这步结束'},cap:{en:'cap',zh:'上限'},budget:{en:'budget used',zh:'预算用完'},memory:{en:'memory full',zh:'显存满'},batch:{en:'whole batch',zh:'等整批'}};
const whyLong=(w,o)=>({step:tx('for the step in progress to end','等正在跑的这一步结束'),cap:tx(`for a place, because the batch is full at its cap of ${o.cfg.cap}`,`等空位，因为 batch 已经到了 ${o.cfg.cap} 个的上限`),
  budget:tx('for room in a step’s token budget, which earlier requests take first','等某一步的 token 预算有空余，因为先到的请求先用预算'),memory:tx('for KV memory, because the pool cannot hold its prompt and answer yet','等 KV 显存，因为显存池暂时放不下它的 prompt 和回答'),
  batch:tx('for the whole batch to finish (static batching)','等整批做完（static batching）')})[w];
/* pictures for "What to remember" */
const REMPIC_=()=>({
 stall:`<svg viewBox="0 0 96 40" width="96" height="40"><rect x="2" y="10" width="7" height="20" fill="var(--sc-chat)"/><rect x="11" y="10" width="7" height="20" fill="var(--sc-chat)"/><rect x="20" y="10" width="72" height="20" fill="var(--sc-doc)"/><text x="56" y="24" text-anchor="middle" font-size="9" fill="var(--bg)" font-family="var(--font-mono)">prefill</text><path d="M20 36h72" stroke="var(--ink-3)" stroke-dasharray="2 2"/></svg>`,
 share:`<svg viewBox="0 0 96 40" width="96" height="40"><rect x="2" y="12" width="34" height="16" fill="var(--sc-kvd)"/><rect x="38" y="12" width="6" height="16" fill="var(--c-kv)"/><rect x="46" y="12" width="6" height="16" fill="var(--c-kv)"/><rect x="54" y="12" width="6" height="16" fill="var(--c-kv)"/><text x="19" y="8" text-anchor="middle" font-size="9" fill="var(--ink)" font-family="var(--font-mono)">×12</text><rect x="62" y="12" width="32" height="16" fill="var(--surface-2)"/></svg>`,
 queue:`<svg viewBox="0 0 96 56" width="96" height="56"><rect x="2" y="21" width="13" height="13" rx="2" fill="var(--sc-chat)"/><path d="M18 27.5h11M26 24.5l3 3-3 3" stroke="var(--ink)" stroke-width="1.3" fill="none"/>`+
   [[tx('cap','上限'),3,false],[tx('budget','预算'),20,true],[tx('memory','显存'),37,false]].map(([t,y,on])=>`<rect x="33" y="${y}" width="61" height="15" rx="2" fill="${on?'var(--ink)':'none'}" stroke="${on?'var(--ink)':'var(--ink-3)'}"/><text x="63.5" y="${y+11}" text-anchor="middle" font-size="9.5" font-weight="${on?650:400}" fill="${on?'var(--bg)':'var(--ink-3)'}" font-family="var(--font-mono)">${esc(t)}</text>`).join('')+`</svg>`
});

/* =====================================================================
   scenarios: the three guided loads; the settings are the learner's own
   ===================================================================== */
const PRESETS=[
 {load:'stall',t:{en:'A document arrives at peak',zh:'高峰期来了一份长文档'},d:()=>{const f=LF('stall'); return tx(`${f.n} short chats, and a ${tokK(f.docs[0].own)}-token document to summarize`,`${f.n} 个短对话，加一份 ${tokK(f.docs[0].own)} 个 token 的文档要写摘要`);}},
 {load:'shared',t:{en:'One support app',zh:'一个客服应用'},d:()=>{const f=LF('shared'); return tx(`${f.n} chats that all start with the same ${fint(f.pre)}-token system prompt`,`${f.n} 个对话，都以同一个 ${fint(f.pre)} 个 token 的 system prompt 开头`);}},
 {load:'apps',t:{en:`Questions about ${wEN(LF('apps').apps.length)} documents`,zh:`${wZH(LF('apps').apps.length)}份文档的提问`},d:()=>{const f=LF('apps'), k=f.apps.length; return tx(`${f.n} questions about ${wEN(k)} long documents; each question starts with its document’s ${fint(f.pre)} tokens`,`${f.n} 个问题，问的是${wZH(k)}份长文档；每个问题都以它所问文档的 ${fint(f.pre)} 个 token 开头`);}}];
const presetNow=()=>PRESETS.findIndex(p=>p.load===S.load);
const loadDesc=load=>{const f=LF(load), k=f.apps.length, nd=f.docs.length;
  return ({stall:()=>tx(`${f.n} short chats and a ${tokK(f.docs[0].own)}-token document`,`${f.n} 个短对话和一份 ${tokK(f.docs[0].own)} 个 token 的文档`),
   shared:()=>tx(`${f.n} support chats with one shared ${fint(f.pre)}-token system prompt`,`${f.n} 个客服对话，共用一个 ${fint(f.pre)} 个 token 的 system prompt`),
   apps:()=>tx(`${f.n} questions about ${wEN(k)} long documents of ${fint(f.pre)} tokens each`,`${f.n} 个问题，问的是${wZH(k)}份各 ${fint(f.pre)} 个 token 的长文档`),
   challenge:()=>tx(`${f.n} chats from ${wEN(k)} apps and plain chats, and ${wEN(nd)} long documents`,`${f.n} 个对话（${wZH(k)}个应用和普通对话），加${wZH(nd)}份长文档`)})[load]();};
const cfgDesc=s=>[s.policy==='static'?tx('static batching','static batching'):tx('continuous batching','continuous batching'),tx(`cap ${s.cap}`,`上限 ${s.cap}`),
  s.budget?tx(`token budget ${fint(s.budget)}`,`token 预算 ${fint(s.budget)}`):tx('no token budget','不设 token 预算'),
  s.cache?(s.cb==null?tx('caching on','缓存打开'):tx(`caching on, ${fCB(s.cb)} for idle prompts`,`缓存打开，空闲 prompt 最多 ${fCB(s.cb)}`)):tx('caching off','缓存关闭')].join(tx(', ','，'));
function scenHTML(){const o=cur(), b=baseOf(S.load), same=o.key===b.key;
  if(!seen.has(o.key)) return `<p class="small">${runLine(o)}</p>`;
  const head=same?tx(`<p><b>The initial setup: ${o.m.all.ok} of ${o.m.all.n} meet their goal.</b> Change a setting to compare on the same requests.</p>`,`<p><b>初始设置：${o.m.all.n} 个里 ${o.m.all.ok} 个达标。</b>改一个设置，在同样的请求上比一比。</p>`)
    :tx(`<p><b>${o.m.all.ok} of ${o.m.all.n} meet their goal (initial setup: ${b.m.all.ok}).</b></p>`,`<p><b>${o.m.all.n} 个里 ${o.m.all.ok} 个达标（初始设置是 ${b.m.all.ok} 个）。</b></p>`);
  return head+feedbackHTML(o,b,same);}
function feedbackHTML(o,b,same){
  const lim=same?missLine(o):limAt(b,false,'in the initial setup','初始设置下')+tx('; ','；')+limAt(o,false,'with yours','你的设置下')+tx('.','。');
  return `<ul class="sc-fb"><li><b>${tx('What limits it','限制它的是')}</b>${tx(': ','：')}${lim}</li>`+(same?'':`<li><b>${tx('Who gained','谁受益')}</b>${tx(': ','：')}${gainLine(o,b)}</li><li><b>${tx('What it cost','代价')}</b>${tx(': ','：')}${costLine(o,b)}</li>`)+`</ul>`;}
/* while a run plays: where it is (a live value, not the result: R3.8) */
function runLine(o){const t=play.t==null?o.end:play.t, done=o.R.filter(r=>r.fin<=t+EPS).length;
  return tx(`Running: ${fC(Math.min(t,o.end))} of ${fC(o.end)}, ${done} of ${o.R.length} requests done.`,`运行中：${fC(Math.min(t,o.end))} / ${fC(o.end)}，${o.R.length} 个请求完成了 ${done} 个。`)+
    (RM()?'':` <button type="button" class="linkbtn" data-act="toend">${tx('Skip to the end ⏭','直接看结果 ⏭')}</button>`);}

/* =====================================================================
   challenge: the load and the goals first, then the five settings and Run
   ===================================================================== */
const CHS=()=>spec('challenge');
const chDocs=()=>CHS().filter(x=>x.cls==='doc');
const chApps=()=>{const m={}; CHS().forEach(x=>{if(x.app) m[x.app]=x.pre;}); return APPS.filter(a=>m[a]).map(a=>[a,m[a]]);};
const changes=s=>(s.policy!==BASE.policy)+(s.cap!==BASE.cap)+(s.budget!==BASE.budget)+(!!s.cache!==BASE.cache)+(s.cache&&s.cb!=null?1:0);
/* every setting of the grid (static batching ignores the budget) */
const GRID=(()=>{const g=[]; ['continuous','static'].forEach(p=>CAPS.forEach(cap=>(p==='static'?[0]:BUDS).forEach(budget=>[[false,null],...CBS.map(v=>[true,v])].forEach(([cache,cb])=>g.push({policy:p,cap,budget,cache,cb}))))); return g;})();
/* tiers (R4.4): met with the fewest changes; met, but fewer changes also pass; not met. The search runs in small slices so the page stays
   responsive, and stops if the state changes (the run token) */
function chTierCompute(s,done){
  const me=getRun('challenge',s), ok=me.m.all.ok_frac>=E.TARGET-EPS, k=changes(s), my=runTok;
  if(!ok){done({tier:3}); return;}
  const cand=GRID.filter(g=>changes(g)<k); let i=0, best=null;
  const slice=()=>{if(my!==runTok) return; const t0=performance.now();
    while(i<cand.length&&performance.now()-t0<12){const g=cand[i++], o=getRun('challenge',g); if(o.m.all.ok_frac>=E.TARGET-EPS){const sh=[g.policy===s.policy,g.cap===s.cap,g.budget===s.budget,g.cache===s.cache].filter(Boolean).length; if(!best||changes(g)<changes(best.g)||(changes(g)===changes(best.g)&&sh>best.sh)) best={g,sh};}}
    if(i<cand.length){setTimeout(slice,0); return;}
    done(best?{tier:2,alt:best.g}:{tier:1});};
  slice();}
function chLoadCard(){const ds=chDocs(), ap=chApps(), sp=CHS(), n=sp.filter(x=>x.cls==='chat').length, span=Math.ceil(Math.max(...sp.map(x=>x.t))+1);
  const dots=sp.map(x=>{const k=x.cls==='doc'?'doc':(x.app||'chat'); return `<i class="sc-ld-${x.cls==='doc'?'doc':'dot'}" style="left:${(x.t/span*100).toFixed(2)}%;background:var(--sc-${k})"></i>`;}).join('');
  const axis=[0,Math.round(span/2),span].map(v=>`<span style="left:${(v/span*100).toFixed(1)}%">${v}${v===span?' s':''}</span>`).join('');
  const lastT=Math.max(...sp.filter(x=>x.cls==='chat').map(x=>x.t));
  return `<div class="sc-load"><div class="sc-gc-h"><span class="sc-lock" aria-hidden="true">${ICON.lock}</span>${tx('The load, fixed: the same requests every run','负载（固定）：每次运行都是同一批请求')}</div>`+
   `<p>${tx(`${n} chats in ${fixd(lastT,0)}${NB}s: ${wEN(ap.length)} apps with their own system prompts (${ap.map(([a,p])=>`${L(APPN[a])} ${tokK(p)}`).join(', ')} tokens) and plain chats; ${wEN(ds.length)} documents (${ds.map(d=>`${tokK(d.own)} at ${fixd(d.t,0)}${NB}s`).join(', ')}).`,
            `${fixd(lastT,0)}${NB}s 内来 ${n} 个对话：${wZH(ap.length)}个应用各有自己的 system prompt（${ap.map(([a,p])=>`${L(APPN[a])} ${tokK(p)}`).join('、')} 个 token），另有普通对话；还有${wZH(ds.length)}份长文档（${ds.map(d=>`${fixd(d.t,0)}${NB}s 时 ${tokK(d.own)}`).join('、')}）。`)}</p>`+
   `<div class="sc-ldstrip" role="img" aria-label="${esc(tx('When each request arrives, coloured by kind','每个请求到达的时刻，按种类着色'))}">${dots}</div><div class="sc-ldaxis mono">${axis}</div>`+
   `<div class="tz-leg sc-ldleg">${['chat','doc',...ap.map(a=>a[0])].map(k=>`<span><i class="sc-sw" style="background:var(--sc-${k})"></i>${esc(kindL(k,true))}</span>`).join('')}</div></div>`;}
const chSeg=(key,label,opts,val)=>A.ui.seg({id:'sc-ch-'+key,key:'ch-'+key,label,options:opts,value:String(val),inline:true});
function chSettingsHTML(){const c=chCfg;
  return `<div class="sc-chset">`+chSeg('policy',tx('Batching','批处理方式'),[['continuous','continuous'],['static','static']],c.policy)+
   chSeg('cap',tx('Cap on running requests','运行请求上限'),CAPS.map(v=>[String(v),String(v)]),c.cap)+
   chSeg('budget',tx('Token budget per step','每步的 token 预算'),BUDS.map(v=>[String(v),v?fint(v):tx('none','不设')]),c.budget)+
   A.ui.seg({id:'sc-ch-cache',key:'ch-cache',label:tx('Prefix caching','前缀缓存'),options:[['false',tx('off','关')],['true',tx('on','开')]],value:String(!!c.cache),inline:true,note:c.cache?'':tx('On, it adds a fifth setting: the cache budget.','打开后多一项设置：缓存预算。')})+
   (c.cache?chSeg('cb',tx('Cache budget for idle prompts','空闲 prompt 的缓存预算'),CBS.map(v=>[String(v),v==null?tx('no limit','不设上限'):fCB(v)]),c.cb):'')+`</div>`;}
function challengeHTML(){
  const last=chRunKey?RC.get(chRunKey):null, dirty=last&&keyOf('challenge',chCfg)!==chRunKey;
  return `<div class="railcard sc-chcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`One ${GPUN}, a new peak: get at least ${pct0(E.TARGET)} of requests to meet their goal`,`一张 ${GPUN}，新的高峰：让至少 ${pct0(E.TARGET)} 的请求达标`)}</h3>`+
   `<p class="small">${tx(`${MODEL} in BF16 on one ${GPUN}. Neither this load nor any setting’s result has appeared in the guided parts.`,`一张 ${GPUN} 用 BF16 跑 ${MODEL}。这个负载和任何设置的结果都没有在分步引导里出现过。`)}</p>`+
   chLoadCard()+goalCard()+`<div class="sc-chh">${tx('Your settings','你的设置')}</div>`+chSettingsHTML()+
   `<div class="sc-chrun"><button type="button" class="btn" data-act="run">${chSubmitted?tx('▶ Run again','▶ 再运行一次'):tx('▶ Run and check','▶ 运行并核对')}</button><span class="small muted">${tx('The stage plays your settings on these requests; the check against the goal comes when the run ends.','舞台会用你的设置跑这批请求；跑完就对照目标给出结果。')}</span></div>`+
   (dirty?`<p class="small sc-dirty">${tx('The settings changed since your last run.','上次运行之后，设置改过了。')}</p>`:'')+
   `<div id="sc-ch-res" class="sc-ch-res" aria-live="polite">${chResultHTML()}</div></div>`;}
function chResultHTML(){
  if(!chRunKey) return ''; const o=RC.get(chRunKey)||getRun('challenge',chRunKey?parseKey(chRunKey):chCfg), b=baseOf('challenge');
  if(!seen.has(o.key)||!chTier) return `<p class="small">${runLine(o)}</p>`;
  const ok=o.m.all.ok_frac>=E.TARGET-EPS, t=chTier;
  let h=ok?`<p class="okline"><b>${tx(`Goal met: ${o.m.all.ok} of ${o.m.all.n} requests meet theirs.`,`达成目标：${o.m.all.n} 个请求里 ${o.m.all.ok} 个达标。`)}</b></p>`
    :`<p class="badline"><b>${tx(`Not yet: ${o.m.all.ok} of ${o.m.all.n} meet their goal, under ${pct0(E.TARGET)}.`,`还没达到：${o.m.all.n} 个里 ${o.m.all.ok} 个达标，不到 ${pct0(E.TARGET)}。`)}</b></p>`;
  if(t.tier===1) h+=`<p class="small">${tx(`And with ${changes(o.s)} changes from the initial setup, the fewest that pass.`,`而且只改了 ${changes(o.s)} 处，是能达标的最少改动。`)}</p>`;
  if(t.tier===2){const x=o.s, a=t.alt, extra=[];
    if(x.policy!==a.policy&&x.policy!==BASE.policy) extra.push(tx('static batching','static batching'));
    if(x.cap!==a.cap&&x.cap!==BASE.cap) extra.push(tx(`cap ${x.cap}`,`上限 ${x.cap}`));
    if(x.budget!==a.budget&&x.budget!==BASE.budget) extra.push(tx(`budget ${fint(x.budget)}`,`预算 ${fint(x.budget)}`));
    if(x.cache&&!a.cache) extra.push(tx('prefix caching','前缀缓存'));
    if(x.cache&&x.cb!=null&&a.cb!==x.cb) extra.push(tx(`cache budget ${fCB(x.cb)}`,`缓存预算 ${fCB(x.cb)}`));
    h+=`<p class="small">${tx(`Fewer changes pass too: ${cfgDesc(a)}.`,`更少的改动也能达标：${cfgDesc(a)}。`)}</p>`+(extra.length?`<p class="small">${tx(`Your ${extra.join(' and ')} ${extra.length>1?'change':'changes'} nothing in this load, so the fewer changes above are enough.`,`你的${extra.join('和')}在这个负载里没有带来变化，所以上面那组更少的改动就够了。`)}</p>`:'');}
  h+=feedbackHTML(o,b,false);
  if(!ok){h+=`<p class="small">${tx('Hint: look at what limits it above, find it in the trace of a request that misses, and relieve that one limit first.','提示：看上面“限制它的是”，在一个没达标的请求的记录里找到它，先解决这一个限制。')}</p>`;
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(chReveal||ok) h+=chChartHTML(o);
  h+=planHTML(o,b)+jobsHTML();
  return h;}
const parseKey=k=>{const p=k.split('_'); return {policy:p[1]==='s'?'static':'continuous',cap:+p[2],budget:+p[3],cache:p[4]==='1',cb:p[5]==='n'?null:+p[5]};};
/* named settings, each run on the challenge's requests, against the 90% goal. Brief (the owner's review of batch 8): the chart keeps
   the settings that pass, the pick and the initial setup, and says how many of the others it left out */
function chChartHTML(o){
  const rows=[[tx('Initial setup','初始设置'),BASE],[tx('Cap 16, budget 2,048, caching on','上限 16，预算 2,048，缓存打开'),{...BASE,cap:16,budget:2048,cache:true}],
   [tx('Cap 64, budget 8,192, caching on','上限 64，预算 8,192，缓存打开'),{...BASE,cap:64,budget:8192,cache:true}],[tx('Cap 64, no budget, caching on','上限 64，不设预算，缓存打开'),{...BASE,cap:64,cache:true}],
   [tx('Cap 64, budget 2,048, caching off','上限 64，预算 2,048，缓存关闭'),{...BASE,cap:64,budget:2048}],[tx('Largest cap, finest chunks: cap 128, budget 128, caching on','上限最大、chunk 最细：上限 128，预算 128，缓存打开'),{...BASE,cap:128,budget:128,cache:true}],
   [tx('Static batching, cap 32, caching on','static batching，上限 32，缓存打开'),{...BASE,policy:'static',cache:true}],[tx('Cap 64, budget 2,048, caching on','上限 64，预算 2,048，缓存打开'),{...BASE,cap:64,budget:2048,cache:true}],
   [tx('Cap 32, budget 512, caching on with 1 GiB','上限 32，预算 512，缓存打开，1 GiB'),{...BASE,budget:512,cache:true,cb:512}]];
  const mine=keyOf('challenge',o.s); if(!rows.some(r=>keyOf('challenge',r[1])===mine)) rows.push([tx('Your settings','你的设置'),o.s]);
  const gp=pct0(E.TARGET), nR=o.R.length, okOf=s=>getRun('challenge',s).m.all.ok_frac>=E.TARGET-EPS, base=keyOf('challenge',BASE);
  const shown=rows.filter(([,s])=>okOf(s)||keyOf('challenge',s)===mine||keyOf('challenge',s)===base);
  return A.ui.cmpChart({title:tx(`Share of the ${nR} requests that meet their goal, ${MODEL} on one ${GPUN} with the challenge’s requests (goal: at least ${gp})`,`一张 ${GPUN} 跑 ${MODEL}、服务挑战题这批请求时，${nR} 个请求里达标的比例（目标是至少 ${gp}）`),
    aria:tx('Share of requests that meet their goal: ','请求达标的比例：')+shown.map(([lab,s])=>`${lab}${tx(': ','：')}${pct0(getRun('challenge',s).m.all.ok_frac)}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(n===1?`The other setting compared here gets under ${gp} of the requests to their goal and is not shown.`:`The other ${n} settings compared here get under ${gp} of the requests to their goal and are not shown.`,
                                `这里比较的另外 ${n} 种设置达标的请求不到 ${gp}，图里没有列出。`),
    rows:rows.map(([lab,s])=>{const r=getRun('challenge',s); return {label:lab,value:r.m.all.ok_frac*100,text:pct0(r.m.all.ok_frac),ok:r.m.all.ok_frac>=E.TARGET-EPS,me:keyOf('challenge',s)===mine,
      tags:(keyOf('challenge',s)===keyOf('challenge',BASE)?[{text:tx('initial setup','初始设置')}]:[]).concat(keyOf('challenge',s)===mine?[{text:tx('your pick','你的选择'),me:true}]:[])};}),
    goal:E.TARGET*100,goalLabel:tx(`goal: ${gp}`,`目标 ${gp}`),max:100,okText:tx(`${gp} or more meet their goal`,`${gp} 以上达标`),noText:tx(`under ${gp}`,`不到 ${gp}`),goalText:tx(`goal: ${gp}`,`目标：${gp}`)});}
/* the plan, to copy: plain text for interview notes (no company names) */
function planText(o,b){
  const ds=chDocs(), ap=chApps(), n=o.chats.length, m=o.m, bm=b.m, lines=[];
  const pair=(a,y)=>`${a} → ${y}`;
  lines.push(tx(`Serving plan: ${MODEL.replace(/‑/g,'-')} on one H100 SXM (BF16); estimates from a step model, queueing included.`,`服务方案：一张 H100 SXM 用 BF16 跑 ${MODEL.replace(/‑/g,'-')}；数字来自一步一步的耗时模型估算，含排队。`));
  lines.push(tx(`Load: ${n} chats (${wEN(ap.length)} apps with system prompts of ${ap.map(a=>tokK(a[1])).join(', ')} tokens, and plain chats); documents of ${ds.map(d=>`${tokK(d.own)} tokens at ${fixd(d.t,0)} s`).join(' and ')}.`,
                `负载：${n} 个对话（${wZH(ap.length)}个应用，system prompt 分别是 ${ap.map(a=>tokK(a[1])).join('、')} 个 token，另有普通对话）；长文档 ${ds.map(d=>`${fixd(d.t,0)} s 时 ${tokK(d.own)} 个 token`).join('、')}。`));
  lines.push(tx(`Goals: short chats get their first token within ${fS(G.chat.ttft)} of arriving and never wait over ${fS(G.chat.gap)} between tokens; documents within ${fS(G.doc.ttft)}; at least ${pct0(E.TARGET)} of requests.`,
                `目标：短对话到达后 ${fS(G.chat.ttft)} 内出首 token，两个 token 之间不超过 ${fS(G.chat.gap)}；长文档 ${fS(G.doc.ttft)} 内；至少 ${pct0(E.TARGET)} 的请求达标。`));
  lines.push(tx(`Settings: ${cfgDesc(o.s)}.`,`设置：${cfgDesc(o.s)}。`));
  lines.push(tx('Initial setup → mine:','初始设置 → 我的：'));
  lines.push('- '+tx(`goal met: ${pair(`${bm.all.ok} / ${bm.all.n} (${pct0(bm.all.ok_frac)})`,`${m.all.ok} / ${m.all.n} (${pct0(m.all.ok_frac)})`)}`,`达标：${pair(`${bm.all.ok} / ${bm.all.n}（${pct0(bm.all.ok_frac)}）`,`${m.all.ok} / ${m.all.n}（${pct0(m.all.ok_frac)}）`)}`));
  lines.push('- '+tx(`short chats' first token (queueing included), p50: ${pair(fS(bm.chat.ttft_p50),fS(m.chat.ttft_p50))}; p95 of ${m.chat.n}: ${pair(fS(bm.chat.ttft_p95),fS(m.chat.ttft_p95))}`,`短对话的首 token（含排队）p50：${pair(fS(bm.chat.ttft_p50),fS(m.chat.ttft_p50))}；${m.chat.n} 个的 p95：${pair(fS(bm.chat.ttft_p95),fS(m.chat.ttft_p95))}`));
  lines.push('- '+tx(`longest gap between tokens: ${pair(fS(bm.chat.gap_max),fS(m.chat.gap_max))}`,`最长的 token 间隔：${pair(fS(bm.chat.gap_max),fS(m.chat.gap_max))}`));
  if(m.doc) lines.push('- '+tx(`documents' first token (queueing included), the slower of ${m.doc.n}: ${pair(fS(bm.doc.ttft_max),fS(m.doc.ttft_max))}`,`长文档的首 token（含排队，${m.doc.n} 份里较慢的）：${pair(fS(bm.doc.ttft_max),fS(m.doc.ttft_max))}`));
  lines.push('- '+tx(`GPU time used (the step times added up): ${pair(fS(b.gpuT),fS(o.gpuT))}`,`GPU 用时（所有步的耗时之和）：${pair(fS(b.gpuT),fS(o.gpuT))}`));
  lines.push('- '+tx(`prefill tokens computed: ${pair(fint(bm.all.prefill_tok),fint(m.all.prefill_tok))}`,`prefill 算过的 token：${pair(fint(bm.all.prefill_tok),fint(m.all.prefill_tok))}`));
  lines.push('- '+tx(`output tokens per second: ${pair(fint(bm.all.thr),fint(m.all.thr))}`,`每秒输出 token：${pair(fint(bm.all.thr),fint(m.all.thr))}`));
  lines.push('- '+tx(`peak KV cache: ${pair(fG(bm.all.peak_gib),fG(m.all.peak_gib))}`,`KV cache 峰值：${pair(fG(bm.all.peak_gib),fG(m.all.peak_gib))}`));
  lines.push(tx('What limits it: ','限制它的是：')+limAt(b,true,'initial setup','初始设置下')+tx('; ','；')+limAt(o,true,'mine','我的设置下')+tx('.','。'));
  /* what each change bought: my settings with that one change undone. A counterfactual gives the answer away, so it waits until the goal
     is met or the answer was shown (review r1, A2) */
  const okRun=m.all.ok_frac>=E.TARGET-EPS;
  if(!okRun&&!chReveal) return lines.join('\n');
  const ch=[];
  if(o.s.budget!==BASE.budget) ch.push([{...o.s,budget:BASE.budget},tx(`token budget ${fint(o.s.budget)}`,`token 预算 ${fint(o.s.budget)}`)]);
  if(!!o.s.cache!==BASE.cache) ch.push([{...o.s,cache:false,cb:null},tx('prefix caching','前缀缓存')]);
  if(o.s.cap!==BASE.cap) ch.push([{...o.s,cap:BASE.cap},tx(`cap ${o.s.cap}`,`上限 ${o.s.cap}`)]);
  if(o.s.cache&&o.s.cb!=null) ch.push([{...o.s,cb:null},tx(`cache budget ${fCB(o.s.cb)}`,`缓存预算 ${fCB(o.s.cb)}`)]);
  if(o.s.policy!==BASE.policy) ch.push([{...o.s,policy:BASE.policy},tx('static batching','static batching')]);
  if(ch.length){lines.push(tx('What each change does (my settings with only that change undone):','每处改动的作用（只撤回这一处，其余保持我的设置）：'));
    ch.forEach(([s2,name])=>{const w=getRun('challenge',s2), ef=effects(w,o), same=w.m.all.ok===m.all.ok;
      const it=e=>tx(`${e.e.l} ${pairTxt(e.e,e.a,e.y)}`,`${e.e.l}${/[A-Za-z0-9)]$/.test(e.e.l)?' ':''}${pairTxt(e.e,e.a,e.y)}`);
      const kw=diagnose(w).kind, ko=diagnose(o).kind, none=missShort(w,true)===missShort(o,true);
      const rel=kw==='none'&&ko!=='none'?tx('it is what makes requests miss now','现在让请求不达标的正是它')
        :none?tx('it relieves no limit here','它在这里没有解决任何限制'):tx(`it relieves ${missShort(w,true)}`,`它解决的是${missShort(w,true)}`);
      let t='- '+tx(`${name}: undo it and ${w.m.all.ok} / ${w.m.all.n} meet the goal (${same?'no change':`now ${m.all.ok}`}); ${rel}.`,
                     `${name}：撤回后 ${w.m.all.n} 个里 ${w.m.all.ok} 个达标（${same?'没有变化':`现在是 ${m.all.ok} 个`}）；${rel}。`);
      if(ef.gain.length) t+=tx(` Gains: ${ef.gain.map(it).join('; ')}.`,`收益：${ef.gain.map(it).join('；')}。`);
      if(ef.cost.length) t+=tx(` Costs: ${ef.cost.map(it).join('; ')}.`,`代价：${ef.cost.map(it).join('；')}。`);
      if(!ef.gain.length&&!ef.cost.length) t+=tx(' No change in these numbers.','这些数字没有变化。');
      lines.push(t);});}
  return lines.join('\n');}
const strip=h=>String(h).replace(/<[^>]+>/g,'');
function planHTML(o,b){const t=planText(o,b);
  return `<div class="sc-plan"><div class="sc-chh">${tx('Your plan, to copy','你的方案（可复制）')}</div><pre class="sc-plantext" id="sc-plan">${esc(t).split(MODEL.replace(/‑/g,'-')).join(MODEL)}</pre><button type="button" class="btn ghost sm" data-act="copyplan">${tx('Copy the plan','复制方案')}</button></div>`;}
/* the two practice problems: which part to do, and the skill this challenge trained (the notes live on GitHub; no company names) */
const PRACTICE_URL={'inference-serving':'OpenAI-Interview-Notes/blob/main/system-design/llm-inference-serving/','inference-engine':'Anthropic-Interview-Notes/blob/main/coding/oa-inference-engine/'};
const pURL=id=>`https://github.com/Schuture/${PRACTICE_URL[id]}${tx('README.md','README.zh.md')}`;
const JOBS=()=>[
 {id:'inference-serving',t:tx('LLM Inference Serving (design)','大模型推理服务（设计题）'),
  d:tx('Do the first-token latency breakdown in “Requirements and scale”, then deep dive 1: continuous batching, interleaving prefill with decode, admission control. The skill: explaining first-token time and stalls by queueing, chunking and caching, and choosing a chunk size by its trade-off.',
       '先做“需求与规模”里的首 token 延迟拆分，再做“深入话题”的第一项（连续批处理，以及把 prefill 交织进 decode，含准入控制）。练的是：用排队、切 chunk 和缓存讲清首 token 延迟和卡顿，并按取舍选 chunk 大小。')},
 {id:'inference-engine',t:tx('Completing a Small LLM Inference Engine (coding)','补全一个小型 LLM 推理引擎（编程题）'),
  d:tx('Do Level 2 (a budget-constrained decode loop) and Level 4 (prefix caching). The skill: the scheduling rules used here (decode first, first come first served, cap and budget, stop at the first request that does not fit) and shared blocks with reference counts, kept when idle and evicted when needed. One difference: that problem admits a whole prompt or nothing, while this lab, like vLLM, splits a prompt to fit the budget.',
       '做“Level 2 —— 带预算约束的解码循环”和“Level 4 —— 前缀缓存”。练的是这里用的调度规则（先 decode、先来先服务、上限和预算、遇到第一个放不下的请求就停），以及带引用计数的共享 block（空闲时留着，需要时淘汰）。有一处不同：那道题的 prompt 要么整个接纳、要么不接纳；这个 Lab 和 vLLM 一样，把 prompt 切开来放进预算。')}];
const jobsHTML=()=>`<div class="sc-jobs"><div class="sc-chh">${tx('Where this goes in an interview','面试里会用到哪儿')}</div><p class="small muted">${tx('Examples of frontier-lab interview questions, reconstructed from second-hand accounts:','Frontier lab 真题示例（据面经整理）：')}</p><ul>`+
  JOBS().map(j=>`<li><a href="${pURL(j.id)}" target="_blank" rel="noopener">${esc(j.t)} <span aria-hidden="true">↗</span></a><span>${esc(j.d)}</span></li>`).join('')+`</ul></div>`;

/* =====================================================================
   the capability layer (R3.16): what the learner can do after the lab, and three ways to show it transfers
   ===================================================================== */
const OUTCOME=()=>tx('diagnose a slow LLM service from its timeline and tune its cap, chunk size and cache by their costs.',
                     '从一个变慢的 LLM 服务的时间线找出卡住它的原因，并按各自的代价选出运行请求上限、token 预算和前缀缓存。');
const XC=new Map();
function xRun(name){if(XC.has(name)) return XC.get(name); let sp, cfg;
  if(name==='doc64'){sp=spec('stall').map(x=>x.cls==='doc'?{...x,own:65536}:x); cfg=cfgOf(BASE);}
  else if(name==='sys12'){sp=E.LOADS.shared(12,48,6.0,12288); cfg=cfgOf({...BASE,budget:2048,cache:true});}
  else if(name==='sys12off'){sp=E.LOADS.shared(12,48,6.0,12288); cfg=cfgOf({...BASE,budget:2048});}
  const o=build(sp,cfg); XC.set(name,o); return o;}
/* 3 significant digits with a unit (R3.19, R6.18): 65.5K, 6.98B, 525M, 2.04 × 10¹⁵ */
const sig3=n=>{const a=Math.abs(n); if(a>=1e12){let e=Math.floor(Math.log10(a)), m=a/Math.pow(10,e); if(+m.toFixed(2)>=10){e++; m/=10;} return `${m.toFixed(2)} × 10${String(e).split('').map(d=>SUP[+d]).join('')}`;}
  const u=a>=1e9?[1e9,'B']:a>=1e6?[1e6,'M']:a>=1e3?[1e3,'K']:[1,''], v=a/u[0]; return (v>=100?v.toFixed(0):v>=10?v.toFixed(1):v.toFixed(2))+u[1];};
const kTok=n=>`${Math.round(n/1000)}K`;   // "about 28K": a token count as the page rounds it
const EXITS=()=>{
  const b=baseOf('stall'), w=b.worst, d64=xRun('doc64'), w64=d64.worst, st64=stepInfo(d64,w64.s), s12=xRun('sys12'), s12off=xRun('sys12off'), pre12=s12.R[0].pre;
  const s6=getRun('shared',{...BASE,budget:2048,cache:true}), pre6=spec('shared')[0].pre, p6=s6.m.all.prefill_tok, p12=s12.m.all.prefill_tok, p12off=s12off.m.all.prefill_tok, n12=s12.R.length;
  const g32=w.g, g64=st64.s.t1-st64.s.t0, ratio=g64/g32, mults=[1,2,Math.round(ratio*10)/10,4], near=(xs,v)=>xs.reduce((bi,x,i)=>Math.abs(Math.log(x/v))<Math.abs(Math.log(xs[bi]/v))?i:bi,0);
  const kx=st64.s.T-65536;   /* the chats' own tokens in the step, beside the document's 64K (review B4, N9) */
  const tip64=tx(`The step’s arithmetic: the document’s ${tokK(65536)} (65,536) tokens${kx>0?` plus ${kx} chat token${kx>1?'s':''}, ${sig3(st64.s.T)} in all,`:''} × 2 × ${sig3(C.P_LAYERS)} parameters, plus 2 × ${sig3(C.P_HEAD)} × ${st64.logits} for the LM head (once for each of the ${st64.logits} requests that get a token from this step), plus ${sig3(st64.att)} attention FLOPs, is ${sig3(st64.s.fl)} FLOPs; at ${sig3(C.FS/1e12)} TFLOP/s that takes ${fS(st64.s.tc)}. Its memory traffic takes only ${fS(st64.s.tm)}.`,
                 `这一步的计算量：文档的 ${tokK(65536)}（65,536）个 token${kx>0?` 加上对话的 ${kx} 个 token，一共 ${sig3(st64.s.T)} 个`:''} × 2 × ${sig3(C.P_LAYERS)} 个参数，加上 LM head 的 2 × ${sig3(C.P_HEAD)} × ${st64.logits}（这一步有 ${st64.logits} 个请求各拿到一个 token，每个算一次），再加上 ${sig3(st64.att)} 次 attention FLOP，一共 ${sig3(st64.s.fl)} FLOP；按 ${sig3(C.FS/1e12)} TFLOP/s 算要 ${fS(st64.s.tc)}。它读写显存只要 ${fS(st64.s.tm)}。`);
  const v4=[p6,p6+(pre12-pre6),2*p6,p12off], how=DEF(tx('How the step is computed','这一步怎么算'),tip64), a1=near(mults,ratio), sec=m=>m===mults[a1]?fR(g64):fR(g32*m);
  /* the answers as ranges (R3.19): 10K wide under 100K; the last one open-ended */
  const bin=v=>{const w=v<1e5?1e4:1e5, lo=Math.floor(v/w)*w; return [lo,lo+w];}, r4=v4.map(bin), a2=(i=>i>=0?i:near(v4,p12))(r4.findIndex(([lo,hi],i)=>i<3&&p12>=lo&&p12<hi)), rg=i=>`${kTok(r4[i][0])}–${kTok(r4[i][1])}`;
  const bx=p12off/p6, bt=bx>=10?Math.floor(bx/10)*10:Math.floor(bx);   /* the last option against the 28K the page shows (review B4, N10) */
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Without a token budget, one long prompt stalls every chat that is decoding. Why?','不设 token 预算时，一个长 prompt 会让所有正在 decode 的对话都卡住。为什么？'),
   o:[tx('The server puts the whole prompt into one step, and a step hands out its tokens only when all of its work ends, so every running chat waits for the whole prompt.','服务端把整个 prompt 放进一步，而一步要等所有工作结束才交出 token，所以每个正在跑的对话都要等整个 prompt 算完。'),
     tx('The long prompt fills the KV memory, so the running chats are pushed out and have to wait until its memory is freed again.','长 prompt 占满了 KV 显存，正在跑的对话被挤了出去，要等它的显存释放才能继续。'),
     tx('The long prompt takes the running chats’ places in the batch, so they sit in the queue until its prefill is done.','长 prompt 占了正在跑的对话在 batch 里的位置，所以它们只能在队列里等它 prefill 完。')],a:0,
   n:[null,tx('Memory stays mostly free here, and nobody is pushed out. The chats keep running, but they get no token until the step ends.','这里显存大部分都空着，也没有谁被挤出去。对话一直在跑，只是这一步结束之前拿不到 token。'),
     tx('The chats stay in the batch. The step computes their next tokens too, but it hands them out only when the whole step ends.','对话一直在 batch 里。这一步也在算它们的下一个 token，只是要等整步结束才交出来。')],
   why:tx('A step is one pass over everything the server scheduled together: the chats’ next tokens and the whole prompt. The prompt’s huge amount of arithmetic sets how long the step takes, and nothing in the step comes out before it ends. A budget splits the prompt into many short steps, so the chats get their tokens between them.',
          '一步是把排在一起的所有工作跑一遍：对话的下一个 token 和整个 prompt。prompt 巨大的计算量决定了这一步有多长，而这一步结束之前谁也拿不到结果。预算把 prompt 切成许多短步，对话就能在这些短步之间拿到 token。')},
  {q:tx('Users ask questions about four long documents that share one cache. Past a point, a bigger cache budget brings no more requests that meet their goal. Why?','用户在问四份长文档的问题，这些文档共用一个缓存。缓存预算大到一定程度以后，再大也不会让更多请求达标。为什么？'),
   o:[tx('A cache helps only documents that are asked about again while they are still cached. Once all four documents fit, the only misses left are each document’s first question, and no cache can serve those.','缓存只对“还缓存着的时候又被问到”的文档有用。四份文档都放得下以后，剩下没命中的只有每份文档的第一个问题，任何缓存都帮不了它们。'),
     tx('A bigger cache makes every lookup slower, so past a point the time a hit saves is lost again in searching the cache.','缓存越大，每次查找越慢，所以到了某个大小，命中省下的时间又在查找里花掉了。'),
     tx('Past a point the cached documents fill the KV memory, so running requests get less room and start to queue.','到了某个大小，缓存的文档占满了 KV 显存，正在跑的请求空间变小，开始排队。')],a:0,
   n:[null,tx('A lookup costs almost nothing here, whatever the cache size. The hits simply stop growing.','这里查找几乎不花时间，缓存多大都一样，只是命中不再增加。'),
     tx(`The four documents take ${fCB(4*LF('apps').pre/C.BLOCK)} of a ${fG(POOL_GIB)} pool, so running requests still find room. The hits stop growing because no other document comes back.`,`四份文档只占 ${fG(POOL_GIB)} 显存池里的 ${fCB(4*LF('apps').pre/C.BLOCK)}，正在跑的请求照样有空间。命中不再增加，是因为没有别的文档会再被问到。`)],
   why:tx('A hit needs the document to come back while it is still cached. With four documents, those four are the working set: once the cache keeps all four, every later question hits, and more room holds nothing that anyone asks about again.',
          '命中的前提是文档还缓存着的时候又被问到。只有四份文档时，这四份就是工作集：缓存把四份都留住以后，后来的问题都能命中，再多的空间放的也是没人再问的东西。')}],
 transfer:[
  {q:tx(`With no token budget, a ${tokK(docOf('stall').own)}-token document is prefilled in one step, and every running chat stops for about ${fR(g32)}. If the document had ${tokK(65536)} tokens instead, still with no token budget, about how long would that stop be?`,
        `不设 token 预算时，一份 ${tokK(docOf('stall').own)} 个 token 的文档在一步里 prefill 完，所有正在跑的对话会停约 ${fR(g32)}。如果文档换成 ${tokK(65536)} 个 token，仍然不设 token 预算，这次停顿大约多长？`),
   o:mults.map(m=>m===1?tx(`About the same, about ${fR(g32)}`,`差不多一样，约 ${fR(g32)}`):tx(`About ${m} times as long, about ${sec(m)}`,`约 ${m} 倍，约 ${sec(m)}`)),a:a1,
   n:[tx('The document is twice as long, so the step that holds all of it cannot stay the same.','文档长了一倍，装着整份文档的那一步不可能不变。'),
      tx('Twice the tokens would double the time only if every part of the work grew in step with the length. The attention part grows faster, because each token also looks back at twice as many earlier tokens.','只有当每一部分工作都和长度同比例增长时，token 翻倍，时间才会翻倍。但 attention 那部分增长得更快，因为每个 token 要回看的前文也多了一倍。'),null,
      tx('Only the attention part grows 4 times; the per-token part only doubles, so the whole step grows less than 4 times.','只有 attention 那部分变成 4 倍；按 token 算的那部分只翻一倍，所以整步的增长不到 4 倍。')],
   why:tx(`About ${mults[a1]} times as long. The per-token part of the work doubles with the length, but the attention part grows four times, because every token also looks back at twice as many earlier tokens. So the whole step grows by more than twice but less than four times. ${how}`,
          `约 ${mults[a1]} 倍。按 token 算的那部分工作随长度翻一倍，而 attention 那部分变成四倍，因为每个 token 要回看的前文也多了一倍。所以整步的增长比两倍多、比四倍少。${how}`)},
  {q:tx(`In a support app, ${s6.R.length} chats all start with the same ${tokK(pre6)}-token system prompt. With caching on, the GPU prefills about ${kTok(p6)} prompt tokens in all. If the system prompt grew from ${tokK(pre6)} to ${tokK(pre12)} tokens, with caching still on, about how many prompt tokens would the GPU prefill?`,
        `一个客服应用里，${s6.R.length} 个对话都以同一个 ${tokK(pre6)} 个 token 的 system prompt 开头。打开缓存时，GPU 一共 prefill 大约 ${kTok(p6)} 个 prompt token。如果 system prompt 从 ${tokK(pre6)} 个 token 变成 ${tokK(pre12)} 个，缓存仍然打开，GPU 一共大约要 prefill 多少个 prompt token？`),
   o:[tx(`${rg(0)}, about the same`,`${rg(0)}，基本不变`),tx(`${rg(1)}, about ${tokK(pre12-pre6)} more`,`${rg(1)}，多了约 ${tokK(pre12-pre6)}`),tx(`${rg(2)}, about twice as many`,`${rg(2)}，大约翻一倍`),tx(`More than ${bt} times as many, because every request computes the prompt again`,`是现在的 ${bt} 多倍，因为每个请求都要重算 prompt`)],a:a2,
   n:[tx(`The first request still has to compute the longer prompt once, so the total grows by about ${tokK(pre12-pre6)}.`,`第一个请求还是要把更长的 prompt 算一遍，所以总数会多出约 ${tokK(pre12-pre6)}。`),null,
      tx('Twice the prompt would double the total only if every request computed it. With caching on, only the first request does.','只有每个请求都要算 prompt，prompt 翻倍总数才会翻倍。打开缓存后，只有第一个请求要算。'),
      tx('That would be the total without caching, because then every request prefills the whole prompt.','那是不开缓存时的总数，因为那时每个请求都要 prefill 整个 prompt。')],
   why:tx(`Only the first request computes the system prompt; every later request finds it in the cache. So the ${tokK(pre12-pre6)} extra tokens are computed only once, by the first request, and the total grows by just that much, to about ${kTok(p12)}. Without caching, every request would prefill the longer prompt again.`,
          `只有第一个请求要算 system prompt，之后的每个请求都在缓存里找到它。所以多出来的 ${tokK(pre12-pre6)} 个 token 只由第一个请求算一次，总数也只多这么多，变成约 ${kTok(p12)}。如果不开缓存，每个请求都要把更长的 prompt 再 prefill 一遍。`)}],
 practice:JOBS().map(j=>({id:j.id,d:esc(j.d)}))};
};

/* =====================================================================
   copy the kit shows: mode descriptions, fidelity, primer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Eight short parts. Watch today’s service miss its goal, predict why, then add one fix at a time: a token budget, prefix caching, a cache budget, the cap on running requests. Last, read the scheduler’s code step by step. Start here.',
            '一共 8 个部分。先看今天的服务怎样达不到目标，预测原因，再一次加一个办法：token 预算、前缀缓存、缓存预算、运行请求上限；最后一步一步读调度器的代码。从这里开始。'),
  sandbox:tx(`Three workloads on one ${GPUN} with ${MODEL}: a long document at peak hour, a support app whose chats share one system prompt, and questions about ${wEN(LF('apps').apps.length)} long documents. Pick one, change any of the five settings, and compare with the initial setup on the same requests.`,
             `三个负载，都是一张 ${GPUN} 跑 ${MODEL}，分别是高峰期来了一份长文档、对话共用一个 system prompt 的客服应用，以及对${wZH(LF('apps').apps.length)}份长文档的提问。选一个，改五项设置中的任何一项，在同样的请求上和初始设置比较。`),
  challenge:(f=>tx(`A new peak on one ${GPUN} with ${MODEL}, with ${f.n} chats from ${wEN(f.apps.length)} apps and plain chats, and ${wEN(f.docs.length)} long documents. Choose all five settings so that at least ${pct0(E.TARGET)} of requests meet their goal.`,
               `一张 ${GPUN} 跑 ${MODEL}，迎来新的高峰，来了 ${f.n} 个对话（${wZH(f.apps.length)}个应用和普通对话）和${wZH(f.docs.length)}份长文档。选好五项设置，让至少 ${pct0(E.TARGET)} 的请求达标。`))(LF('challenge'))
});
const SRC={
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 dbx:'https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices',
 vsched:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py',
 vcfg:'https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py',
 design:'https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md',
 radix:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py',
 orca:'https://www.usenix.org/conference/osdi22/presentation/yu',
 serve:'https://www.usenix.org/conference/osdi24/presentation/agrawal',
 sglang:'https://arxiv.org/abs/2312.07104',
 vllm:'https://arxiv.org/abs/2309.06180'
};
const SIMPL=()=>((en,zh)=>A.lang()==='zh'?zh.map(zfix):en)(
 [`Step time from the shared step model: arithmetic at 50% of the ${GPUN}’s dense BF16 peak, memory traffic at 60% of its bandwidth, the longer of the two; no kernel-launch, scheduling, tokenizer or network time. Every event happens at a step’s end.`,
  'The scheduler serves requests first come, first served, so the document’s chunks take each step’s budget first, as they do in vLLM’s default scheduler.',
  'Admission sets aside room for the whole answer, so nothing is preempted. vLLM admits more and preempts a request when memory runs out.',
  'Prefix caching covers the apps’ system prompts and the documents in whole 16-token blocks. When two requests compute the same block in the same step, this model keeps one copy; vLLM V1 keeps both until the requests end (its block tables only grow), and SGLang’s radix cache frees the duplicate. Here that changes the peak memory by a fraction of a GiB and no result. Real engines also cache conversation histories.',
  'The cache budget stands for the free KV memory that idle prefixes may use.',
  'One GPU, no tensor parallelism. Answer lengths are fixed by the load; the scheduler does not know them.',
  'Numbers are teaching estimates. TTFT here always includes queueing, unlike the single-request labs.'],
 [`每一步的耗时来自共用的耗时模型：计算按 ${GPUN} 稠密 BF16 峰值的 50%，读写显存按带宽的 60%，取较长的那个；不计启动 kernel、调度、tokenizer 和网络的时间。所有事件都发生在一步结束时。`,
  '调度按先来先服务，所以文档的 chunk 先占用每一步的预算，vLLM 默认的调度器也是这样。',
  '接纳请求时给整个回答预留空间，所以不会有请求被抢占。vLLM 会接纳更多请求，显存不够时再抢占其中一个。',
  '前缀缓存只缓存各应用的 system prompt 和长文档，按整个 16 个 token 的 block 缓存。两个请求在同一步里算出同一个 block 时，这里只留一份；vLLM V1 会两份都留着，直到请求结束（它的 block 表只追加），SGLang 的 radix cache 会释放重复的那份。在这里，这只让峰值显存差不到 1 GiB，不改变任何结果。真实引擎还会缓存对话历史。',
  '缓存预算代表空闲前缀能用的空闲 KV 显存。',
  '只有一张 GPU，没有张量并行。回答长度由负载固定，调度器事先不知道。',
  '这些数是教学用的估算。这里的 TTFT 总是包含排队，和只看单个请求的那些 Lab 不同。']);
function FIDPOP(){
  return tx(
 `<p><b>Toy run:</b> your browser runs a whole serving scheduler step by step on fixed, seeded requests, and prices every step with the step model of ${labLink('prefill-decode')}. ${J('math','“Show the math”')} redoes the key numbers.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>${MODEL} in BF16 on one ${GPUN}: ${C.L} layers, ${C.KVH} KV heads of ${C.HD} dimensions, ${fint(C.KV_TOK/1024)} KiB of KV cache per token, a pool of ${fint(C.POOL_T)} tokens (${fG(POOL_GIB)}) after the weights and 2 GiB of runtime overhead in 90% of 80 GB (read as GiB).</li>`+
 `<li>Continuous batching as in vLLM V1: the running requests first, then the waiting ones, first come first served under the cap, the token budget and the KV memory; the first request that does not fit stops admission. Here the running requests are their decodes plus at most one unfinished prompt, so taking the decodes first gives the same steps.</li>`+
 `<li>KV memory in 16-token blocks; shared system-prompt blocks with reference counts, idle ones kept within the cache budget and evicted least recently used first; blocks in use are never evicted. Static batching for comparison.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>${SIMPL().map(x=>`<li>${x}</li>`).join('')}</ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${MODEL}: ${cite(SRC.cfg,'config.json')}.</li>`+
 `<li>${GPUN}, from ${cite(SRC.h100,'NVIDIA’s product page')}: BF16 1,979 TFLOPS “with sparsity”, so 989.5 dense; 3.35 TB/s; 80 GB.</li>`+
 `<li>vLLM (main branch, checked 2026-09-29): ${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')} and ${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')}; ${cite(SRC.design,'the prefix caching design')}.</li>`+
 `<li>${cite(SRC.orca,'Orca (OSDI 2022)')}, ${cite(SRC.serve,'Sarathi-Serve (OSDI 2024)')}, ${cite(SRC.sglang,'SGLang (2024)')}, ${cite(SRC.vllm,'vLLM (SOSP 2023)')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-29. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器在固定的、有随机种子的请求上，一步一步地跑一个完整的服务调度器，每一步的耗时用 ${labLink('prefill-decode')} 那个 Lab 的耗时模型计算。${J('math','“展开公式”')}里重算了关键的数。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>一张 ${GPUN} 用 BF16 跑 ${MODEL}：${C.L} 层，${C.KVH} 个 KV head，每个 ${C.HD} 维，每个 token 的 KV cache ${fint(C.KV_TOK/1024)} KiB；80 GB（按 GiB 计）的 90% 减去权重和 2 GiB 运行时开销，剩下的显存池放得下 ${fint(C.POOL_T)} 个 token（${fG(POOL_GIB)}）。</li>`+
 `<li>和 vLLM V1 一样的 continuous batching：先调度在跑的请求，再在上限、token 预算和 KV 显存之内先来先服务地接纳排队的请求；第一个放不下的请求让接纳停下。这里在跑的请求就是各自的 decode，加上最多一个没算完的 prompt，所以先排 decode 得到的步完全一样。</li>`+
 `<li>KV 显存按 16 个 token 的 block 分配；共享的 system prompt block 带引用计数，空闲的在缓存预算之内留着，最久没用的先淘汰；在用的 block 永远不淘汰。另有 static batching 作对比。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>${SIMPL().map(x=>`<li>${x}</li>`).join('')}</ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${MODEL}：${cite(SRC.cfg,'config.json')}。</li>`+
 `<li>${GPUN}：${cite(SRC.h100,'NVIDIA 产品页')}，BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；3.35 TB/s；80 GB。</li>`+
 `<li>vLLM（main 分支，2026-09-29 核对）：${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')} 和 ${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')}；${cite(SRC.design,'前缀缓存的设计说明')}。</li>`+
 `<li>${cite(SRC.orca,'Orca（OSDI 2022）')}、${cite(SRC.serve,'Sarathi-Serve（OSDI 2024）')}、${cite(SRC.sglang,'SGLang（2024）')}、${cite(SRC.vllm,'vLLM（SOSP 2023）')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-29。原型，尚未经专家审阅。</p></div></div>`);
}
const PRIPOP=()=>tx(`<p>This lab puts the serving path together. A request is prefilled in one step, then decoded one token per step (${labLink('prefill-decode')}); the server rebuilds its batch every step (${labLink('continuous-batching')}); a token budget splits a long prompt into chunks (${labLink('chunked-prefill')}); a shared prompt prefix is cached and reused (${labLink('prefix-caching')}); all of it lives in the KV cache memory (${labLink('kv-cache')}).</p>`,
                    `<p>这个 Lab 把推理服务这条路线串起来。请求先在一步里 prefill，再每步 decode 一个 token（${labLink('prefill-decode')}）；服务端每一步都重组 batch（${labLink('continuous-batching')}）；token 预算把长 prompt 切成 chunk（${labLink('chunked-prefill')}）；共享的 prompt 前缀被缓存、复用（${labLink('prefix-caching')}）；这一切都放在 KV cache 显存里（${labLink('kv-cache')}）。</p>`);

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Serving capstone','Serving 综合挑战'),title:l?L(l.q):tx('How do you rescue a congested chat service?','怎么救活一个拥堵的聊天服务？'),
    minutes:20,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: the serving path','前置知识：推理服务这条路线'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
const playLabel=()=>play.playing?tx('❚❚ Pause','❚❚ 暂停'):(play.t!=null&&play.t>1e-9?tx('▶ Play','▶ 继续'):tx('▶ Play','▶ 播放'));
const playBtn=id=>`<button type="button" class="btn sm sc-pbtn"${id?` id="${id}"`:''} data-act="play">${playLabel()}</button>`;
function stageHTML(){
  return `<div class="panel stage sc-stage" id="sc-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="sc-st-title"></div><div class="meter-sub" id="sc-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="sc-status" tabindex="0"></span><div class="sc-m2d" id="sc-m2d" aria-live="off"></div></div></div>`+
   `<div class="sc-cfgrow"><div class="sc-pills" id="sc-pills"></div><div class="sc-kinds" id="sc-kinds"></div></div>`+
   `<div class="stage-graph graph sc-graph" id="sc-graph">`+
    `<div class="sc-strip" id="sc-strip">`+
     `<div class="sc-box sc-q" id="sc-q"><div class="sc-bh"><span>${tx('Queue','队列')}</span><b class="mono" id="sc-qn"></b></div><div class="sc-chips" id="sc-chips"></div></div>`+
     `<div class="sc-arr" aria-hidden="true">→</div>`+
     `<div class="sc-box sc-gpu" id="sc-gpu"><div class="sc-bh"><span id="sc-gh"></span></div><div class="sc-meters" id="sc-meters"></div><div class="sc-gbar" id="sc-gbar"></div><div class="sc-gdur" id="sc-gdur"></div></div>`+
     `<div class="sc-arr" aria-hidden="true">→</div>`+
     `<div class="sc-box sc-out" id="sc-out"><div class="sc-bh"><span><span class="sc-lw">${tx('Tokens out','输出的 token')}</span><span class="sc-ls">${tx('Out','输出')}</span></span><b class="mono" id="sc-on"></b></div><div class="sc-dots" id="sc-dots"></div><div class="sc-onote" id="sc-onote"></div></div>`+
    `</div>`+
    `<div class="sc-play" id="sc-play">${playBtn()}<button type="button" class="btn ghost sm" data-act="back" aria-label="${esc(tx('One step back','后退一步'))}" data-tipx="${esc(tx('Back to the start of this engine step, or of the previous one when the playhead is already there.','回到这一个引擎步的开头；已经在开头时，回到上一步的开头。'))}">◂</button>`+
     `<button type="button" class="btn ghost sm" data-act="fwd" aria-label="${esc(tx('One step forward','前进一步'))}" data-tipx="${esc(tx('On to the end of the engine step under the playhead: every request in it gets its output there.','前进到播放头所在的这一步结束：这一步里的每个请求都在那时拿到结果。'))}">▸</button>`+
     `<button type="button" class="btn ghost sm" data-act="replay" aria-label="${esc(tx('Replay','重放'))}">↺<span class="sc-rt">${tx(' Replay',' 重放')}</span></button>`+
     `<div class="seg-ctl sc-speed" role="group" aria-label="${esc(tx('Replay speed','回放速度'))}" data-key="speed" data-tipx="${esc(tx(`1× plays the whole run in about ${BASE_SECS} s, whatever its length; 0.5× takes twice as long, 2× half.`,`1× 用约 ${BASE_SECS} s 放完整次运行，不管它多长；0.5× 慢一倍，2× 快一倍。`))}">${SPEEDS.map(v=>`<button type="button" data-v="${v}" aria-pressed="${v===spd()}">${v}×</button>`).join('')}</div>`+
     `<input type="range" id="sc-scrub" min="0" max="1000" step="1" value="0" aria-label="${esc(tx('Time in the run','运行中的时刻'))}"></div>`+
    `<p class="sc-clock" id="sc-clock"></p>`+codePanelHTML()+
    `<div class="sc-tlw" id="sc-tlw"><div class="sc-cap" id="sc-tlcap"></div><div class="sc-tlbox" id="sc-tlbox"><canvas id="sc-tl" tabindex="0" role="img"></canvas><button type="button" class="sc-stall" id="sc-stall" hidden></button><span class="sc-pflab" id="sc-pflab" hidden></span><div class="sc-tcard" id="sc-tcard" role="region" aria-live="polite" aria-label="${esc(tx('The trace of the selected request','所选请求的记录'))}" hidden></div></div><div class="sc-axis mono" id="sc-axis" aria-hidden="true"></div><div class="legend-row sc-tlleg" id="sc-tlleg"></div></div>`+
    `<div class="sc-mem" id="sc-mem"><div class="sc-memh"><span class="sc-cap" id="sc-memt"></span><span class="mono sc-memv" id="sc-memv"></span></div><div class="sc-mlab" id="sc-mlab"></div><div class="sc-mbar" id="sc-mbar"></div><div class="sc-axis sc-maxis mono" id="sc-maxis" aria-hidden="true"></div><div class="legend-row sc-mleg" id="sc-mleg"></div></div>`+
   `</div>`+
   `<div class="sc-m2" id="sc-m2"></div>`+
   `<div class="sc-low"><div class="sc-trace" id="sc-trace"></div><div class="sc-mtw"><div class="sc-cap" id="sc-mtcap"></div><div id="sc-mt"></div></div></div>`+
   `<p class="sc-scope small muted" id="sc-scope" tabindex="0"></p>`+
   `<details class="math" id="sc-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="sc-mathbody"></div></details></div>`;
}
const segCtl=(key,label,opts,val,note)=>A.ui.seg({id:'sc-ctl-'+key,key,label,options:opts,value:String(val),note});
function controlsHTML(){
  return `<aside class="panel controls" id="sc-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="sc-lock" hidden></p><div id="sc-ctls">`+
   segCtl('policy',tx('Batching','批处理方式'),[['continuous','continuous'],['static','static']],S.policy,tx('Continuous rebuilds the batch at every step; static waits for the whole batch to finish.','continuous 每一步都重组 batch；static 要等整批做完。'))+
   segCtl('cap',tx('Cap on running requests','运行请求上限'),CAPS.map(v=>[String(v),String(v)]),S.cap,tx('How many requests share each step (vLLM’s max_num_seqs). A request that finds it full waits in the queue.','每一步最多放几个请求（vLLM 的 max_num_seqs）。满了，新请求就在队列里等。'))+
   segCtl('budget',tx('Token budget per step','每步的 token 预算'),BUDS.map(v=>[String(v),v?fint(v):tx('none','不设')]),S.budget,tx('The most tokens one step may compute, decode tokens included (vLLM’s max_num_batched_tokens). “None” prefills a whole prompt in one step.','一步最多算多少个 token，decode 的 token 也算在内（vLLM 的 max_num_batched_tokens）。“不设”就是整个 prompt 一步算完。'))+
   segCtl('cache',tx('Prefix caching','前缀缓存'),[['false',tx('off','关')],['true',tx('on','开')]],!!S.cache,tx('When caching is on, a shared prefix (an app’s system prompt or a document) is computed once, kept in shared blocks and reused by later requests.','打开后，共享的前缀（应用的 system prompt 或文档）只算一次，存在共享的 block 里，后来的请求直接复用。'))+
   `<div id="sc-cbwrap">`+segCtl('cb',tx('Cache budget for idle prompts','空闲 prompt 的缓存预算'),CBS.map(v=>[String(v),v==null?tx('no limit','不设上限'):fCB(v)]),S.cb,tx('How much KV memory idle cached prompts may keep; beyond it the least recently used is evicted. Real engines use whatever memory is free.','空闲的缓存 prompt 最多能占多少 KV 显存；超出就淘汰最久没用的。真实引擎用的是所有空闲显存。'))+`</div>`+
   `</div><div class="sc-grp" id="sc-ctl-eff"><div class="sc-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Arithmetic: 50% of the peak FLOP/s','计算：峰值 FLOP/s 的 50%')}</span><div class="ctl-note">${tx(`This lab assumes a step reaches 50% of the ${GPUN}’s dense BF16 peak (989.5 TFLOP/s), that is ${fixd(C.FS/1e12,2)} TFLOP/s. It sets how long a prefill takes, and so how long a stall lasts.`,`本 Lab 假设一步能用到 ${GPUN} 稠密 BF16 峰值（989.5 TFLOP/s）的 50%，即 ${fixd(C.FS/1e12,2)} TFLOP/s。它决定 prefill 要多久，也就决定卡顿有多长。`)}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Memory: 60% of the peak bandwidth','读写显存：峰值带宽的 60%')}</span><div class="ctl-note">${tx(`This lab assumes a step reaches 60% of the 3.35 TB/s peak, that is ${fixd(C.BS/1e12,2)} TB/s. Databricks (2023) measured 60% at batch size 1 on 2 × H100-80GB, serving MPT-7B. This ${fixd(C.BS/1e12,2)} TB/s sets a decode step’s time: the weights are read once per step.`,`本 Lab 假设一步能用到 3.35 TB/s 峰值带宽的 60%，即 ${fixd(C.BS/1e12,2)} TB/s。Databricks（2023）在 2 × H100-80GB 上跑 MPT-7B 实测，batch 大小为 1 时是 60%。这个 ${fixd(C.BS/1e12,2)} TB/s 决定一个 decode 步的耗时：每一步都要把权重读一遍。`)}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx(`KV pool: ${fint(C.POOL_T)} tokens`,`KV 显存池：${fint(C.POOL_T)} 个 token`)}</span><div class="ctl-note">${tx(`What is left of 90% of the 80 GB (read as GiB) after the weights and 2 GiB of runtime overhead, as in the lab ${labLink('kv-cache')}: ${fG(POOL_GIB)}, ${fint(C.KV_TOK/1024)} KiB per token. The 90% is vLLM’s gpu_memory_utilization; its default is now 0.92, a pool of ${fint(POOL92)} tokens (${fG((POOL92-C.POOL_T)*C.KV_TOK/C.GiB)} more). A request waits for memory only in the four-document scenario with caching off and a budget of 8,192 or none; everywhere else a larger pool would change no result.`,`80 GB（按 GiB 计）的 90% 减去权重和 2 GiB 运行时开销后剩下的，和 ${labLink('kv-cache')}这个 Lab 一样：${fG(POOL_GIB)}，每个 token ${fint(C.KV_TOK/1024)} KiB。90% 是 vLLM 的 gpu_memory_utilization；它现在的默认值是 0.92，那样显存池有 ${fint(POOL92)} 个 token（多 ${fG((POOL92-C.POOL_T)*C.KV_TOK/C.GiB)}）。只有在四份文档的场景里、关闭缓存并且预算是 8,192 或不设时，才会有请求等显存；其余情况下，更大的显存池不会改变任何结果。`)}</div></div></div>`+
   `<div id="sc-resetw"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=[
 ['# One scheduler step, as this lab runs it (continuous batching)',''],
 ['def step(t, running, waiting, cache, pool):','def'],
 ['    budget = BUDGET or float("inf")    # tokens this step may compute','budget'],
 ['    work = []',''],
 ['    for r in running:                  # 1. decode first: 1 token each','decode'],
 ['        if r.prefill_done and r.tokens_left:','decode'],
 ['            work.append((r, 1)); budget -= 1','decode'],
 ['    for r in running:                  # 2. prefills already running','prefill'],
 ['        if not r.prefill_done and budget > 0:','prefill'],
 ['            n = min(r.prompt_left, budget)','prefill'],
 ['            work.append((r, n)); budget -= n','prefill'],
 ['    for r in list(waiting):            # 3. admit, first come first served','admit'],
 ['        if len(running) >= CAP or budget <= 0:','admit'],
 ['            break                      # the first that waits stops all','admit'],
 ['        hit = cache.lookup(r.app, r.prompt)    # leading cached blocks','lookup'],
 ['        if not pool.can_hold(r.blocks - hit):','memory'],
 ['            break                      # room for prompt + whole answer','memory'],
 ['        cache.ref(r.app, hit)          # shared blocks: one more user','ref'],
 ['        r.done = hit * BLOCK','ref'],
 ['        waiting.remove(r); running.append(r)','admit'],
 ['        n = min(r.prompt_left, budget)','admit'],
 ['        work.append((r, n)); budget -= n','admit'],
 ['    t += step_time(work)               # max(arithmetic, memory)','time'],
 ['    for r in finished(work):           # every event at the step’s end','release'],
 ['        pool.free(r.own_blocks)','release'],
 ['        cache.unref(r.app, r.shared)   # idle prompts stay cached','release'],
 ['    cache.evict_lru(keep=CACHE_BUDGET) # least recently used first','evict'],
 ['    return t','']];
/* =====================================================================
   part 7: the scheduler's code, step by step. The stage shows the Connect card's code with this step's values beside the lines it
   runs (product owner, 29 Sep: “播放某一步时高亮对应代码行”). Everything comes from the engine's step record and the requests.
   ===================================================================== */
/* the Connect card's statements, line for line (the same line numbers), without the long comments, so a column of this step's values
   fits beside them; the three loops keep a short comment. [indent level, code, key] */
const WCODE=[[0,'# one scheduler step (continuous batching)','h'],[0,'def step(t, running, waiting, cache, pool):','def'],
 [1,'budget = BUDGET or float("inf")','bud'],[1,'work = []','work'],
 [1,'for r in running:  # 1. decode first','dl'],[2,'if r.prefill_done and r.tokens_left:','di'],[3,'work.append((r, 1)); budget -= 1','da'],
 [1,'for r in running:  # 2. prefills going on','pl'],[2,'if not r.prefill_done and budget > 0:','pi'],[3,'n = min(r.prompt_left, budget)','pn'],[3,'work.append((r, n)); budget -= n','pa'],
 [1,'for r in list(waiting):  # 3. admit in order','al'],[2,'if len(running) >= CAP or budget <= 0:','ac'],[3,'break','ab'],
 [2,'hit = cache.lookup(r.app, r.prompt)','lk'],[2,'if not pool.can_hold(r.blocks - hit):','mc'],[3,'break','mb'],
 [2,'cache.ref(r.app, hit)','rf'],[2,'r.done = hit * BLOCK','rd'],[2,'waiting.remove(r); running.append(r)','ad'],[2,'n = min(r.prompt_left, budget)','an'],[2,'work.append((r, n)); budget -= n','aa'],
 [1,'t += step_time(work)','tm'],[1,'for r in finished(work):','fl'],[2,'pool.free(r.own_blocks)','fr'],[2,'cache.unref(r.app, r.shared)','un'],
 [1,'cache.evict_lru(keep=CACHE_BUDGET)','ev'],[1,'return t','rt']];
const blocksOf=n=>Math.floor((n+C.BLOCK-1)/C.BLOCK);
/* the blocks a request allocates while it decodes: one each time its context c = P + gen − 1 reaches a multiple of 16 (gen = 1 … out − 1) */
const ansBlocks=r=>r.out>=2?Math.floor((r.P+r.out-2)/C.BLOCK)-Math.floor((r.P-1)/C.BLOCK):0;
/* what the scheduler did in step s: the step record, the requests admitted at its start, the ones still waiting and why (the lab's why[]),
   the ones that finished at its end and the idle blocks evicted (at its start while allocating, or at its end over the cache budget) */
const CODEC=new WeakMap();
function codeInfo(o,s){let m=CODEC.get(o); if(!m) CODEC.set(o,m=new Map()); if(m.has(s)) return m.get(s);
  const st=o.st[s], R=o.R, B0=o.cfg.budget||0, at=(x,y)=>Math.abs(x-y)<1e-12, ord=(a,b)=>a.arr-b.arr||a.id-b.id;
  const adm=R.filter(r=>at(r.admit,st.t0)).sort(ord), aset=new Set(adm.map(r=>r.id));
  const wait=R.filter(r=>r.arr<=st.t0+EPS&&r.admit>st.t0+EPS).sort(ord);
  const dec=[], cont=[], apre=new Map();
  st.work.forEach(w=>{if(w.k==='d') dec.push(w.id); else if(aset.has(w.id)) apre.set(w.id,w); else cont.push(w);});
  let b=B0?B0:Infinity; const b0=b; b-=dec.length; const bDec=b;
  const contN=cont.map(w=>{const r=R[w.id], v={id:w.id,left:r.P-w.a,bud:b,n:w.b-w.a,last:w.b===r.P}; b-=v.n; return v;}), bPre=b;
  const admN=adm.map(r=>{const w=apre.get(r.id), v={id:r.id,hit:r.hit,left:r.P-r.hit,bud:b,n:w?w.b-w.a:0,whole:!!w&&w.a===r.hit&&w.b===r.P,P:r.P}; b-=v.n; return v;}), bEnd=b;
  const fin=R.filter(r=>at(r.fin,st.t1)).sort(ord).map(r=>({id:r.id,own:blocksOf(r.P-r.se)+ansBlocks(r),shared:r.se/C.BLOCK,app:r.app}));
  const v={s,st,b0,bDec,bPre,bEnd,dec,cont:contN,adm:admN,wait:wait.map(r=>r.id),why:wait.length?o.why[s]:null,fin,
    evT:o.evTrim.get(s)||0,evA:o.evAlloc.get(s)||0,evApps:o.evAtT.get(s)||{},nrun0:st.nrun-adm.length,cap:o.cfg.cap,cacheOn:!!o.cfg.cache,cb:o.cfg.cacheBlocks};
  m.set(s,v); return v;}
/* which step the stage and the panel show: the one under the playhead (endView, before the strip). The panel shows it running, or
   ended when the playhead sits on its end after ▸ */
function codeStepAt(o){if(play.t==null) return null; const e=endView(o); if(e>=0) return {s:e,phase:'end'}; const s=stepAt(o,play.t); return s>=0?{s,phase:'run'}:null;}
const nmR=(o,id)=>rName(o.R[id]);
/* a document's name in a sentence (“the policy” alone reads as a rule), and a shared prompt's */
const docNm=app=>app==='policy'?tx('the policy document','政策文件'):L(KIND[app]||{en:app,zh:app});
const prefNm=app=>DOCAPPS.includes(app)?docNm(app):tx(`the ${APPN[app].en} app’s system prompt`,`${APPN[app].zh}应用的 system prompt`);
/* the request whose prefill put the cached prefix there: the last one of the app admitted before t that computed shared blocks */
function srcOf(o,app,t){let x=null; o.R.forEach(r=>{if(r.app===app&&r.sharedNew>0&&r.admit<t-EPS&&(!x||r.admit>x.admit)) x=r;}); return x?rName(x):tx('an earlier request','之前的请求');}
/* every line: its state (ran, now, next, skip, stop, plain), the value beside it (at most 12 characters, so it fits the column at every
   width) and a full sentence for its tooltip (R6.21). Values are those of the step's start, when its decisions are made */
function codeLines(o,ci,phase){
  const out={}, inf=v=>!isFinite(v), fb=v=>inf(v)?tx('no limit','不设'):fint(Math.max(0,v)), end=phase==='end';
  const set=(k,st,v,tip)=>{out[k]={st,v:v||'',tip:tip||''};};
  const nw=ci.adm.length+ci.wait.length, a0=ci.adm[0], c0=ci.cont[0], w0=ci.wait.length?nmR(o,ci.wait[0]):'', nb=ci.wait.length-1, n0=ci.nrun0;
  const behind=nb>0?tx(`, and so ${nb===1?'does the request':`do the ${nb} requests`} behind it`,`，它后面的 ${nb} 个请求也一样`):'';
  /* “even if it would fit” is true only when memory stops the loop: after a cap or budget stop nobody behind fits either (review B, B7) */
  const brk=mem=>nb>0?tx(`The loop breaks here: ${w0} is first in the queue, and the ${nb===1?'request behind it waits':`${nb} requests behind it wait`} too${mem?`, even if ${nb===1?'it':'they'} would fit`:''}.`,`循环在这里跳出：${w0}排在队首，它后面的 ${nb} 个请求${mem?'就算放得下也只能':'也'}一起等。`)
    :tx(`The loop breaks here, and ${w0} waits for a later step.`,`循环在这里跳出，${w0}要等到后面的步。`);
  const nWait=n=>tx(n===1?'1 waits':`${n} wait`,`${n} 个要等`);
  set('h','plain');
  set('def','ran',`t = ${fC(ci.st.t0)}`,tx(`This step starts at ${fC(ci.st.t0)}, with ${n0} request${n0===1?'':'s'} running and ${nw?`${nw} waiting in the queue`:'nobody waiting'}.`,`这一步在 ${fC(ci.st.t0)} 开始，此时 ${n0} 个请求在跑，${nw?`${nw} 个在队列里等`:'队列是空的'}。`));
  set('bud','ran',inf(ci.b0)?tx('no limit','不设'):fint(ci.b0),inf(ci.b0)?tx('There is no token budget, so a whole prompt can go into one step.','没有设 token 预算，一个 prompt 可以整个放进一步。'):tx(`The budget is ${fint(ci.b0)} tokens per step, decode tokens included.`,`预算是每步 ${fint(ci.b0)} 个 token，decode 的 token 也算在内。`));
  set('work','ran');
  const nd=ci.dec.length;
  set('dl','ran',nd?tx(`${nd} decode${nd>1?'s':''}`,`${nd} 个 decode`):tx('none','没有'),nd?tx(nd===1?'1 running request is past its prefill, so it decodes one more token in this step.':`${nd} running requests are past their prefill, so each decodes one more token in this step.`,`${nd} 个在跑的请求已经做完 prefill，这一步${nd===1?'':'各'} decode 一个 token。`):tx('No running request is past its prefill yet, so nothing decodes in this step.','在跑的请求都还没做完 prefill，这一步没有 decode。'));
  set('di',nd?'ran':'skip'); set('da',nd?'ran':'skip',nd&&!inf(ci.b0)?tx(`left ${fb(ci.bDec)}`,`剩 ${fb(ci.bDec)}`):'',nd&&!inf(ci.b0)?tx(`Each decode token takes one token of the budget: ${fint(ci.b0)} − ${nd} = ${fint(ci.bDec)} left.`,`每个 decode token 用掉 1 个 token 的预算：${fint(ci.b0)} − ${nd} = ${fint(ci.bDec)}，还剩这么多。`):'');
  const nc=ci.cont.length;
  set('pl','ran',nc?tx(nc===1?'1 goes on':`${nc} go on`,`${nc} 个继续`):tx('none','没有'),nc?tx(`${cap1(nmR(o,c0.id))}’s prefill started in an earlier step and goes on in this one${nc===2?', and so does one more':(nc>2?`, and so do ${nc-1} more`:'')}.`,`${nmR(o,c0.id)}的 prefill 在前面的步里就开始了，这一步接着算${nc>1?`；另外还有 ${nc-1} 个也一样`:''}。`):tx('No prefill goes on from an earlier step.','没有从前面的步延续下来的 prefill。'));
  set('pi',nc?'ran':'skip');
  set('pn',nc?'ran':'skip',nc?`n = ${fint(c0.n)}`:'',nc?tx(`${cap1(nmR(o,c0.id))} has ${fint(c0.left)} prompt tokens left and the budget has ${fb(c0.bud)}, so this step takes ${fint(c0.n)} of them${c0.last?', its last chunk':''}.`,`${nmR(o,c0.id)}还剩 ${fint(c0.left)} 个 prompt token，预算还剩 ${fb(c0.bud)}，所以这一步算其中 ${fint(c0.n)} 个${c0.last?'，这是它的最后一个 chunk':''}。`):'');
  set('pa',nc?'ran':'skip',nc&&!inf(ci.b0)?tx(`left ${fb(ci.bPre)}`,`剩 ${fb(ci.bPre)}`):'',nc&&!inf(ci.b0)?tx(`After ${nc===1?'the prefill that goes on':'the prefills that go on'}, ${fb(ci.bPre)} tokens of the budget are left for new requests.`,`继续的 prefill 算完以后，预算还剩 ${fb(ci.bPre)} 个 token 留给新请求。`):'');
  set('al','ran',nw?tx(`${nw} waiting`,`${nw} 个在等`):tx('empty','空'),nw?tx(nw===1?'When this step starts, 1 request waits in the queue, and the scheduler looks at it.':`When this step starts, ${nw} requests wait in the queue. The scheduler takes them in the order they arrived.`,nw===1?'这一步开始时，队列里有 1 个请求在等，调度器看它能不能进来。':`这一步开始时，队列里有 ${nw} 个请求在等。调度器按到达的顺序一个个看。`):tx('Nobody is waiting when this step starts, so there is nobody to admit.','这一步开始时队列是空的，没有要接纳的请求。'));
  const why=ci.why, stopCB=why==='cap'||why==='budget', stopM=why==='memory';
  if(stopCB) set('ac','stop',why==='cap'?tx(`cap ${ci.cap}: stop`,`上限 ${ci.cap}：停`):tx(`${fb(ci.bEnd)} left: stop`,`剩 ${fb(ci.bEnd)}：停`),
    why==='cap'?tx(`${ci.st.nrun} requests are running, which is the cap, so the scheduler stops admitting. ${cap1(w0)} waits${behind}, although ${inf(ci.b0)?'there is no budget to run out':`${fb(ci.bEnd)} tokens of the budget are left`}.`,`已经有 ${ci.st.nrun} 个请求在跑，到了上限，所以调度器停止接纳。${w0}只能等${behind}，尽管${inf(ci.b0)?'并没有预算限制':`预算还剩 ${fb(ci.bEnd)} 个 token`}。`)
      :tx(`No budget is left, so the scheduler stops admitting. ${cap1(w0)} waits${behind}, although only ${ci.st.nrun} of the ${ci.cap} places are taken.`,`预算已经用完，所以调度器停止接纳。${w0}只能等${behind}，尽管上限 ${ci.cap} 个里才用了 ${ci.st.nrun} 个。`));
  else if(ci.adm.length||stopM){const who=nmR(o,(a0||{id:ci.wait[0]}).id), bl=a0?a0.bud:ci.bEnd;
    set('ac','ran',tx('room','有空'),tx(`The scheduler checks ${who}: ${n0===0?'no request is running':(n0===1?`the 1 running request is under the cap of ${ci.cap}`:`the ${n0} running requests are under the cap of ${ci.cap}`)}, and ${inf(bl)?'there is no budget limit':`${fb(bl)} tokens of the budget are left`}, so it goes on.`,`调度器检查${who}：${n0===0?'还没有请求在跑':`在跑的 ${n0} 个请求没到 ${ci.cap} 个的上限`}，${inf(bl)?'也没有预算限制':`预算还剩 ${fb(bl)} 个 token`}，所以接着往下。`));}
  else set('ac','skip');
  set('ab',stopCB?'stop':'skip',stopCB?nWait(ci.wait.length):'',stopCB?brk(false):'');
  const look=ci.adm.length||stopM, hit=a0?a0.hit:0, hb=hit/C.BLOCK, lw=nmR(o,(a0||{id:ci.wait[0]||0}).id);
  set('lk',look?'ran':'skip',look?(!ci.cacheOn?tx('off: 0','关：0'):(hit?tx(`${fint(hb)} blocks`,`${fint(hb)} 块`):tx('miss: 0','未命中：0'))):'',
    look?(!ci.cacheOn?tx('Prefix caching is off, so the lookup finds nothing: hit = 0.','前缀缓存是关的，所以什么也找不到：hit = 0。'):(hit?tx(`The first ${fint(hb)} blocks of ${nmR(o,a0.id)}’s prompt are already in the cache: ${fint(hit)} tokens that ${srcOf(o,o.R[a0.id].app,ci.st.t0)} computed.`,`${nmR(o,a0.id)}的 prompt 开头 ${fint(hb)} 个 block 已经在缓存里，也就是${srcOf(o,o.R[a0.id].app,ci.st.t0)}算好的 ${fint(hit)} 个 token。`):tx(`None of ${lw}’s prompt is in the cache.`,`${lw}的 prompt 不在缓存里。`))):'');
  if(stopM) set('mc','stop',tx('full: stop','放不下：停'),tx(`The KV pool cannot hold ${w0}’s prompt beyond its cache hit and its whole answer, so the scheduler stops admitting. ${cap1(w0)} waits${behind}.`,`KV 显存池放不下${w0}的 prompt（除去缓存命中的部分）加上它的整个回答，所以调度器停止接纳。${w0}只能等${behind}。`));
  else set('mc',ci.adm.length?'ran':'skip',ci.adm.length?(ci.evA?tx(`evicts ${fint(ci.evA)}`,`淘汰 ${fint(ci.evA)} 块`):tx('fits','放得下')):'',ci.adm.length?tx(`${ci.evA?`To make room, ${fint(ci.evA)} idle cached blocks are evicted, and then the`:'The'} KV pool can hold the prompt beyond the cache hit and the whole answer, so the scheduler admits ${nmR(o,a0.id)}.`,`${ci.evA?`为了腾出空间，先淘汰 ${fint(ci.evA)} 个空闲的缓存 block，之后`:''}KV 显存池放得下 prompt（除去缓存命中的部分）和整个回答，所以调度器接纳${nmR(o,a0.id)}。`):'');
  set('mb',stopM?'stop':'skip',stopM?nWait(ci.wait.length):'',stopM?brk(true):'');
  const na=ci.adm.length, ad=na?'ran':'skip', names=ci.adm.map(x=>nmR(o,x.id));
  /* several admitted in one step: the loop runs lines 15–22 once for each; the values of 15, 18 and 19 are the first one's, line 21 shows
     every n so the budget adds up (review B, B3) */
  const each=na>1?tx(` The loop runs this line for each of the ${na} requests it admits; the value is ${names[0]}’s.`,`循环对接纳的 ${na} 个请求各跑一次这一行；这里是${names[0]}的值。`):'';
  if(na>1&&out.lk) out.lk.tip+=each;
  set('rf',ad,na?(ci.cacheOn&&hit?tx(`+1 on ${fint(hb)}`,`${fint(hb)} 块 +1`):tx('none','无')):'',na?(ci.cacheOn&&hit?tx(`The ${fint(hb)} shared blocks get one more user: their reference count goes up by 1, and nothing is copied.`,`这 ${fint(hb)} 个共享 block 多了一个使用者：引用计数加 1，不复制任何东西。`):tx('Nothing is shared, so no reference count changes.','没有共享的 block，引用计数不变。'))+each:'');
  /* at most 12 characters: 24,576 tokens is written 24K tokens, the lab's K of 1,024 (review B, B5) */
  const doneV=(()=>{const v=tx(`${fint(hit)} tokens`,`${fint(hit)} token`); return v.length<=12?v:tx(`${tokK(hit)} tokens`,`${tokK(hit)} token`);})();
  set('rd',ad,na?doneV:'',na?(hit?tx(`${fint(hb)} blocks × ${C.BLOCK} tokens = ${fint(hit)} tokens count as done, so only ${fint(a0.P-hit)} of the prompt’s ${fint(a0.P)} tokens still need a prefill.`,`${fint(hb)} 个 block × ${C.BLOCK} 个 token = ${fint(hit)} 个 token 算作已完成，所以 prompt 的 ${fint(a0.P)} 个 token 里只剩 ${fint(a0.P-hit)} 个要 prefill。`):tx('Nothing is cached, so the whole prompt still has to be prefilled.','没有缓存可用，整个 prompt 都还要 prefill。'))+each:'');
  set('ad',ad,na?tx(na===1?'admit 1':`admit ${na}`,`接纳 ${na} 个`):'',na?tx(`${cap1(andList(names))} ${na>1?'move':'moves'} from the queue into the running batch.`,`${names.join('、')}从队列进入正在跑的 batch。`):'');
  const nsum=ci.adm.reduce((x,y)=>x+y.n,0), nlist=ci.adm.map(x=>fint(x.n)).join(' + ');
  set('an',ad,na?(na===1?`n = ${fint(a0.n)}`:(nlist.length<=12?nlist:`Σ ${fint(nsum)}`)):'',
    na===1?tx(`${cap1(nmR(o,a0.id))} has ${fint(a0.left)} prompt tokens to prefill and the budget has ${fb(a0.bud)} left, so this step takes ${fint(a0.n)}${a0.whole?(hit?': the rest of its prompt':': its whole prompt'):''}.`,`${nmR(o,a0.id)}有 ${fint(a0.left)} 个 prompt token 要 prefill，预算还剩 ${fb(a0.bud)}，所以这一步算 ${fint(a0.n)} 个${a0.whole?(hit?'，也就是它 prompt 里剩下的全部':'，也就是它的整个 prompt'):''}。`)
      :(na?tx(`${cap1(andList(ci.adm.map(x=>`${nmR(o,x.id)} takes ${fint(x.n)}`)))}: ${fint(nsum)} tokens of the budget in all.`,`${ci.adm.map(x=>`${nmR(o,x.id)}算 ${fint(x.n)} 个`).join('，')}，一共用掉 ${fint(nsum)} 个 token 的预算。`):''));
  set('aa',ad,na&&!inf(ci.b0)?tx(`left ${fb(ci.bEnd)}`,`剩 ${fb(ci.bEnd)}`):'',na&&!inf(ci.b0)?tx(`After ${na===1?'the admission':'the admissions'}, ${fb(ci.bEnd)} tokens of the budget are left.`,`接纳完以后，预算还剩 ${fb(ci.bEnd)} 个 token。`):'');
  const dt=ci.st.t1-ci.st.t0;
  set('tm',end?'ran':'now','+'+fS(dt),tx(`The step takes the longer of its arithmetic, ${fS(ci.st.tc)}, and its memory traffic, ${fS(ci.st.tm)}: ${fS(dt)}. Every request in it gets its output only when it ends.`,`这一步的耗时取计算（${fS(ci.st.tc)}）和读写显存（${fS(ci.st.tm)}）中较长的那个：${fS(dt)}。这一步里的每个请求都要等它结束才拿到结果。`));
  const nf=ci.fin.length, f0=ci.fin[0];
  if(!end){['fl','fr','un','ev','rt'].forEach(k=>set(k,'next'));
    set('fl','next','▸',tx('These lines run when the step ends, when its requests get their output: press ▸ to see them.','这几行在这一步结束时才跑，那时这一步里的请求拿到结果：按 ▸ 就能看到。'));}
  else {
    set('fl','ran',nf?tx(`${nf} done`,`${nf} 个完成`):tx('none','没有'),nf?tx(`${cap1(andList(ci.fin.map(x=>nmR(o,x.id))))} ${nf>1?'get':'gets'} the last answer token in this step and ${nf>1?'finish':'finishes'}.`,`${ci.fin.map(x=>nmR(o,x.id)).join('、')}在这一步拿到最后一个回答 token，做完了。`):tx('No request finishes in this step.','这一步没有请求做完。'));
    const own=ci.fin.reduce((a,x)=>a+x.own,0), others=ci.fin.slice(1).map(x=>nmR(o,x.id));
    /* “beyond the shared part” only when the request shares a prefix; the others named (review B, B9 and B14) */
    set('fr',nf?'ran':'skip',nf?tx(`frees ${fint(own)}`,`释放 ${fint(own)} 块`):'',nf?tx(`${cap1(nmR(o,f0.id))}’s own blocks, for its ${f0.shared>0?'prompt beyond the shared part':'prompt'} and for its answer, go back to the pool: ${fint(f0.own)} blocks${nf>1?`, and ${fint(own-f0.own)} more from ${andList(others)}`:''}.`,`${nmR(o,f0.id)}自己的 block（${f0.shared>0?'共享部分以外的 prompt 和它的回答':'它的 prompt 和回答'}）还给显存池：${fint(f0.own)} 个${nf>1?`；${others.join('、')}再还 ${fint(own-f0.own)} 个`:''}。`):'');
    /* the shared blocks each finished request gives up, summed per prefix: two requests of one app take its count down by 2 */
    const grp=[]; ci.fin.filter(x=>x.shared>0).forEach(x=>{const g=grp.find(y=>y.app===x.app); if(g) g.ids.push(x.id); else grp.push({app:x.app,n:x.shared,ids:[x.id]});});
    const g0=grp[0];
    set('un',nf?'ran':'skip',nf?(ci.cacheOn&&g0?tx(`−${g0.ids.length} on ${fint(g0.n)}`,`${fint(g0.n)} 块 −${g0.ids.length}`):tx('none','无')):'',
      nf?(ci.cacheOn&&g0?grp.map(g=>tx(`${cap1(andList(g.ids.map(i=>nmR(o,i))))} no longer ${g.ids.length>1?'use':'uses'} the ${fint(g.n)} shared blocks of ${prefNm(g.app)}: their reference count goes down by ${g.ids.length}.`,`${g.ids.map(i=>nmR(o,i)).join('、')}不再使用${prefNm(g.app)}的 ${fint(g.n)} 个共享 block：它们的引用计数减 ${g.ids.length}。`)).join(tx(' ',''))+tx(' Blocks nobody uses any more stay in the cache, idle, for a later request.','没人再用的 block 留在缓存里空闲着，等后面的请求复用。'):tx('Nothing shared to give up.','没有要放手的共享 block。')):'');
    const apps=Object.entries(ci.evApps), one=apps.length===1?apps[0]:null, whole=one&&(()=>{const r=o.R.find(x=>x.app===one[0]); return r&&r.se/C.BLOCK===one[1];})();
    set('ev','ran',!ci.cacheOn?tx('off','关'):(ci.evT?tx(`evicts ${fint(ci.evT)}`,`淘汰 ${fint(ci.evT)} 块`):tx('nothing','无')),
      !ci.cacheOn?tx('Prefix caching is off, so nothing is kept to evict.','前缀缓存是关的，没有要淘汰的东西。'):(ci.evT?tx(`The idle cached blocks now take more than the cache budget${ci.cb==null?'':` of ${fCB(ci.cb)}`}, so the least recently used go: ${fint(ci.evT)} blocks${one?`, ${whole?'all of ':'of '}${prefNm(one[0])}`:''}.`,`空闲的缓存 block 超过了缓存预算${ci.cb==null?'':`（${fCB(ci.cb)}）`}，所以淘汰最久没用的：${fint(ci.evT)} 个 block${one?`，也就是${whole?'整份':''}${prefNm(one[0])}`:''}。`)
        :(ci.cb==null?tx('There is no cache budget in this setting, so idle blocks stay cached and nothing is evicted.','这个设置没有缓存预算，空闲的 block 一直留在缓存里，不用淘汰。'):tx('The idle cached blocks fit within the cache budget, so nothing is evicted.','空闲的缓存 block 没有超过缓存预算，不用淘汰。'))));
    set('rt','ran',`t = ${fC(ci.st.t1)}`,tx(`The step ends at ${fC(ci.st.t1)}, and the next one starts there.`,`这一步在 ${fC(ci.st.t1)} 结束，下一步从这里开始。`));}
  return out;}
/* the five moments of part 7, one per lever of parts 3 to 6 plus a usual step; each is found in its run by a rule, never typed */
const MOMENTS=[
 {id:'usual',part:'budget',cfg:{load:'stall',...BASE,budget:2048},
  t:{en:'A usual step',zh:'平常的一步'},d:{en:'Chats decode, and a new chat is admitted',zh:'对话在 decode，一个新对话被接纳'},
  find:o=>o.st.findIndex((x,i)=>{const c=codeInfo(o,i); return c.dec.length>=3&&c.adm.length>=1&&!c.wait.length&&!c.cont.length;})},
 {id:'budget',part:'budget',cfg:{load:'stall',...BASE,budget:2048},
  t:{en:'A chat waits for the budget',zh:'对话在等预算'},d:{en:'The document’s chunk takes the rest of the budget',zh:'文档的 chunk 占满了剩下的预算'},
  find:o=>o.st.findIndex((x,i)=>{const c=codeInfo(o,i); return c.wait.length>0&&c.why==='budget';})},
 {id:'cache',part:'cache',cfg:{load:'shared',...BASE,budget:2048,cache:true},
  t:{en:'A cache hit',zh:'缓存命中'},d:{en:'The shared prompt is not prefilled again',zh:'共享的 prompt 不用再 prefill'},
  find:o=>o.st.findIndex((x,i)=>codeInfo(o,i).adm.some(a=>a.hit>0))},
 {id:'evict',part:'cbudget',at:'end',cfg:{load:'apps',...BASE,budget:2048,cache:true,cb:1536},   // what it shows happens at the step's end
  t:{en:'An eviction',zh:'淘汰'},d:{en:'An idle document leaves the cache',zh:'一份空闲的文档被移出缓存'},
  find:o=>o.st.findIndex((x,i)=>codeInfo(o,i).evT>0)},
 {id:'cap',part:'cap',cfg:{load:'shared',...BASE,budget:2048,cache:true,cap:8},
  t:{en:'The cap is full',zh:'上限满了'},d:{en:'A request waits although the budget has room',zh:'预算还有空，请求也只能等'},
  find:o=>o.st.findIndex((x,i)=>{const c=codeInfo(o,i); return c.wait.length>0&&c.why==='cap';})}];
/* the line that decides each moment's step, drawn heavier while the panel shows that step (review A, suggestion 1) */
const KEYLN={usual:'an',budget:'ac',cache:'lk',evict:'ev',cap:'ac'};
let codeM=null;                                          // the moment picked last (its id), for the card's observation
const codeSeen=new Set();
const momentOf=id=>MOMENTS.find(m=>m.id===id);
function momentStep(m){const o=getRun(m.cfg.load,m.cfg); return {o,s:m.find(o)};}
/* pick a moment: its setting, the playhead a little into its step (the GPU box shows the step running), or at its end when what it
   shows happens there (the eviction) */
function pickMoment(id){const m=momentOf(id); if(!m) return; codeM=id; codeSeen.add(id);
  bump(); stopPlay(); Object.assign(S,{...m.cfg}); sel=null; relCache={k:'',v:new Set()};
  const {o,s}=momentStep(m); seen.add(o.key); const st=o.st[s]; codeEdge='end'; play.t=m.at==='end'?st.t1:st.t0+0.45*(st.t1-st.t0); render();}
/* part 7: the code panel comes into view after a moment, ▸ or ◂ (review A, findings 2 and 3): the span from the GPU box to the panel's
   last line when it fits, else the panel alone. On a desktop the sticky stage scrolls on its own; the span is placed in the stage's own
   coordinates, so a page scroll in the same frame (the card's result coming into view) cannot throw it off. On a phone, or a window too
   short for the sticky stage, the page scrolls, below the top bar and the mini bar (R3.10, R7.6) */
function showCode(flash){const wc=$('#sc-wc',app); if(!wc||wc.hidden) return;
  requestAnimationFrame(()=>{if(!app||wc.hidden||!wc.isConnected) return;
    const st=$('#sc-stage',app), strip=$('#sc-strip',app), r=wc.getBoundingClientRect(), g=strip?strip.getBoundingClientRect():r, beh=RM()?'auto':'smooth';
    const fit=(a,b,lo,hi)=>b-a>hi-lo||a<lo?a-lo:(b>hi?b-hi:0);
    if(innerWidth>760&&st&&getComputedStyle(st).position==='sticky'&&st.scrollHeight>st.clientHeight+2){
      const sr=st.getBoundingClientRect(), off=y=>y-sr.top+st.scrollTop, lo=st.scrollTop+4, hi=st.scrollTop+st.clientHeight-6;
      const dy=off(r.bottom)-off(g.top)<=hi-lo?fit(off(g.top),off(r.bottom),lo,hi):fit(off(r.top),off(r.bottom),lo,hi);
      if(Math.abs(dy)>2) st.scrollBy({top:dy,behavior:beh});
      /* a stage not stuck yet (the page near its top) ends below the window: the page scrolls until it sticks, never past the head of the
         part's card (R3.7; review B, B1) */
      const stick=parseFloat(getComputedStyle(st).top)||70, bottom=off(r.bottom)-(st.scrollTop+(Math.abs(dy)>2?dy:0))+sr.top;
      if(bottom>innerHeight-8&&sr.top>stick+1){const bar=document.querySelector('.topbar'), card=$('#sc-rail .step.current',app);
        let py=Math.min(bottom-(innerHeight-8),sr.top-stick); if(card) py=Math.min(py,card.getBoundingClientRect().top-((bar?bar.getBoundingClientRect().bottom:58)+8));
        if(py>2) window.scrollBy({top:py,behavior:beh});}}
    else {const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), lo=(bar?bar.getBoundingClientRect().bottom:58)+4, hi=innerHeight-8, dkH=dk&&!dk.hidden?dk.offsetHeight:49;
      const dy=r.bottom-g.top<=hi-lo?fit(g.top,r.bottom,lo,hi):fit(r.top,r.bottom,lo+dkH,hi);
      if(Math.abs(dy)>2) window.scrollBy({top:dy,behavior:beh});}
    if(flash) A.ui.flash(wc);});}
function momentsHTML(){return `<div class="levers sc-moms" role="group" aria-label="${esc(tx('Moments to look at, pick one','要看的时刻，选一个'))}">`+MOMENTS.map(m=>`<button type="button" class="choice lever sc-mom" data-moment="${m.id}" aria-pressed="${codeM===m.id}"><span class="lv-t"><b>${esc(L(m.t))}<span class="sc-momok">${codeSeen.has(m.id)?' ✓':''}</span></b><span class="lv-d">${esc(L(m.d))}</span></span></button>`).join('')+`</div>`;}
/* the card's observation for the picked moment: a bold sentence and at most two more, naming the requests and the deciding line
   (R3.10, R6.23; review A, finding 7) */
const andList=xs=>xs.length<2?(xs[0]||''):tx(`${xs.slice(0,-1).join(', ')} and ${xs[xs.length-1]}`,`${xs.slice(0,-1).join('、')}和${xs[xs.length-1]}`);
function obsCode(){const m=codeM&&momentOf(codeM); if(!m) return '';
  const {o,s}=momentStep(m), ci=codeInfo(o,s), S1=fint(s+1), R=o.R, nm=id=>nmR(o,id), nd=ci.dec.length;
  const w0=ci.wait.length?nm(ci.wait[0]):'', nb=ci.wait.length-1;
  /* where the stage is against the moment's step: the same step (in either phase), before it, after it, or elsewhere; the note says
     which way and comes first (review of v10, S4) */
  const cs=o.key===cur().key?codeStepAt(cur()):null, where=o.key!==cur().key||!cs?'off':(cs.s<s?'before':(cs.s>s?'after':'here'));
  const note=where==='here'?'':`<p class="small muted sc-obnote">${where==='before'?tx('The stage is before this step. Pick the moment again to come back to it.','舞台还没到这一步。再选一次这个时刻就能回来。')
    :(where==='after'?tx('The stage has moved on past this step. Pick the moment again to come back to it.','舞台已经走过了这一步。再选一次这个时刻就能回来。')
    :tx('The stage no longer shows this step. Pick the moment again to come back to it.','舞台已经不在这一步了。再选一次这个时刻就能回来。'))}</p>`;
  const waitsEN=nb>0?`${w0} and the ${nb===1?'request behind it wait':`${nb} requests behind it wait`}`:`${w0} waits`, waitsZH=`${w0}${nb>0?`和它后面的 ${nb} 个请求`:''}只能等`;
  let h='';
  if(m.id==='usual'){const a=ci.adm[0];
    h=tx(`<p><b>Step ${S1}: ${nd} requests decode, and ${nm(a.id)} is admitted with its whole prompt.</b></p><p>No prefill goes on from an earlier step, so the admission loop gets the rest of the budget. ${cap1(nm(a.id))} passes the checks for the cap, the budget and memory, and <code>n = min(r.prompt_left, budget)</code> puts its whole prompt into this same step.</p>`,
         `<p><b>第 ${S1} 步：${nd} 个请求在 decode，${nm(a.id)}被接纳，整个 prompt 一起放进这一步。</b></p><p>没有从前面的步延续下来的 prefill，所以剩下的预算都留给接纳循环。${nm(a.id)}通过了上限、预算和显存三项检查，<code>n = min(r.prompt_left, budget)</code> 把它的整个 prompt 放进了同一步。</p>`);}
  else if(m.id==='budget'){const c=ci.cont[0], all=c.n===ci.bDec;
    h=tx(`<p><b>Step ${S1}: ${nm(c.id)}’s chunk takes ${all?'the rest of the budget':`${fint(c.n)} tokens of the budget`}, so ${waitsEN}.</b></p><p>The decodes take one token each and ${all?`${nm(c.id)}’s chunk`:'the prefills'} ${all?'takes':'take'} all the rest, so the admission loop stops at <code>budget &lt;= 0</code>, although the cap still has room. This is how a new chat waits for the budget while a document prefills.</p>`,
         `<p><b>第 ${S1} 步：${nm(c.id)}的 chunk ${all?'占满了剩下的预算':`用掉了 ${fint(c.n)} 个 token 的预算`}，所以${waitsZH}。</b></p><p>每个 decode 用掉一个 token，${all?`${nm(c.id)}的 chunk`:'prefill'}用掉剩下的全部，所以接纳循环停在 <code>budget &lt;= 0</code>，尽管上限还有空位。文档在 prefill 时，新来的对话就是这样在等预算。</p>`);}
  else if(m.id==='cache'){const a=ci.adm.find(x=>x.hit>0), src=srcOf(o,R[a.id].app,ci.st.t0);
    h=tx(`<p><b>Step ${S1}: ${nm(a.id)} finds its app’s system prompt in the cache, so this step prefills only the rest of its prompt.</b></p><p><code>cache.lookup</code> finds the blocks of the system prompt that ${src} computed, and they get one more user instead of a new copy. This is what prefix caching saves.</p>`,
         `<p><b>第 ${S1} 步：${nm(a.id)}在缓存里找到了它所在应用的 system prompt，所以这一步只 prefill 它 prompt 里剩下的部分。</b></p><p><code>cache.lookup</code> 找到${src}算好的那段 system prompt 的 block，它们只是多了一个使用者，不用再复制一份。前缀缓存省下的就是这些。</p>`);}
  else if(m.id==='evict'){const f=ci.fin[0], mine=f.app, evs=Object.keys(ci.evApps), evn=andList(evs.map(docNm)), mem=memAt(o,s);
    const still=R.some(r=>r.id!==f.id&&r.app===mine&&r.admit<=ci.st.t1+EPS&&r.fin>ci.st.t1+EPS);
    const idle=DOCAPPS.filter(a=>evs.includes(a)||((mem.pres&&mem.pres[a])||0)>(mem.inuse[a]||0)), others=idle.filter(a=>a!==mine).map(docNm);
    /* the eviction happens at the step's end: when ◂ brings the stage back to the same step while it runs, the card says what will
       happen, not what has (GPT-6 review of v9, #1; R7.10) */
    const ahead=where==='before'||(where==='here'&&cs.phase==='run');
    if(ahead) h=tx(`<p><b>Step ${S1}: when this step ends, ${nm(f.id)} will finish, and <code>cache.evict_lru</code> will evict ${evn}.</b></p><p>${still?`Other questions will still use ${docNm(mine)}, but the idle documents, ${andList(others)}, will take more than the ${fCB(ci.cb)} cache budget`:`Nobody will use ${docNm(mine)} then, so it will stay cached, idle, next to ${andList(others)}, and together they will take more than the ${fCB(ci.cb)} cache budget`}; ${evn} was used least recently, so it will go first. ${where==='before'?`Press ▸ until step ${S1} ends to see it happen.`:'Press ▸ to see it happen.'}</p>`,
         `<p><b>第 ${S1} 步：这一步结束时，${nm(f.id)}将完成，随后 <code>cache.evict_lru</code> 淘汰${evn}。</b></p><p>${still?`那时别的问题还在用${docNm(mine)}，但空闲的${andList(others)}会超过 ${fCB(ci.cb)} 的缓存预算`:`那时没有请求再用${docNm(mine)}，它会空闲着留在缓存里，和${andList(others)}加起来超过 ${fCB(ci.cb)} 的缓存预算`}；${evn}最久没用，所以会先被淘汰。${where==='before'?`按 ▸ 直到第 ${S1} 步结束，就能看到这一刻。`:'按 ▸ 看这一刻。'}</p>`);
    else h=tx(`<p><b>Step ${S1}: ${nm(f.id)} finishes, and <code>cache.evict_lru</code> evicts ${evn}.</b></p><p>${still?`Other questions still use ${docNm(mine)}, but the idle documents, ${andList(others)}, take more than the ${fCB(ci.cb)} cache budget.`:`Nobody uses ${docNm(mine)} now, so it stays cached, idle, next to ${andList(others)}, and together they take more than the ${fCB(ci.cb)} cache budget.`} ${cap1(evn)} was used least recently, so it goes, and the next question about it prefills the whole document again, which is what a cache budget that is too small costs.</p>`,
         `<p><b>第 ${S1} 步：${nm(f.id)}做完了，<code>cache.evict_lru</code> 淘汰了${evn}。</b></p><p>${still?`别的问题还在用${docNm(mine)}，但空闲的${andList(others)}已经超过了 ${fCB(ci.cb)} 的缓存预算。`:`现在没有请求再用${docNm(mine)}，它空闲着留在缓存里，和${andList(others)}加起来超过了 ${fCB(ci.cb)} 的缓存预算。`}${evn}最久没用，所以先被淘汰；之后再问它，就得把整份文档重新 prefill，这就是缓存预算太小的代价。</p>`);}
  else if(m.id==='cap'){
    h=tx(`<p><b>Step ${S1}: ${ci.st.nrun} requests run, which is the cap, so ${waitsEN} although ${2*ci.bEnd>ci.b0?'most of the budget is unused':`${fint(ci.bEnd)} tokens of the budget are left`}.</b></p><p>The admission loop checks the cap first: <code>len(running) &gt;= CAP</code> is true, so it stops, and ${w0} waits until a running request finishes. This is how a queue builds up when the cap is too low.</p>`,
         `<p><b>第 ${S1} 步：${ci.st.nrun} 个请求在跑，到了上限，所以${waitsZH}，尽管${2*ci.bEnd>ci.b0?'大部分预算都没用上':`预算还剩 ${fint(ci.bEnd)} 个 token`}。</b></p><p>接纳循环先查上限：<code>len(running) &gt;= CAP</code> 成立，于是停下，${w0}要等到有请求做完才能进来。上限设得太低时，排队就是这样来的。</p>`);}
  return note+h+`<p class="small">${J('code',tx('See these lines on the stage','看舞台上的这几行'))}</p>`;}
/* the panel on the stage, only in part 7 */
const codeOn=()=>mode==='guided'&&STEPS[step]&&STEPS[step].id==='code';
function codePanelHTML(){
  return `<div class="sc-wc" id="sc-wc" hidden><div class="sc-wch"><span class="sc-cap" id="sc-wct"></span></div><div class="legend-row sc-wleg" id="sc-wleg"></div>`+
   `<div class="sc-wcb" id="sc-wcb">${WCODE.map(([i,c,k],n)=>{const h=c.indexOf('#'), code=h>0?`${esc(c.slice(0,h))}<span class="sc-wcm">${esc(c.slice(h))}</span>`:(h===0?`<span class="sc-wcm">${esc(c)}</span>`:esc(c));
     return `<div class="sc-wl" data-k="${k}" data-i="${n+1}"><span class="sc-wn mono" aria-hidden="true">${n+1}</span><code class="sc-wx" style="--i:${i}">${code}</code><span class="sc-wv mono" tabindex="-1"></span></div>`;}).join('')}</div>`+
   `<p class="small sc-wobs" id="sc-wobs" hidden>${J('obs',tx('About this step: in the card below ↓','这一步的说明：见下面的卡片 ↓'))}</p></div>`;}
/* the legend sits under the panel's header, so it is on screen whenever the code is (review A, finding 3); one entry per line state (R5.7) */
const wLeg=()=>[['sc-sw-wran',tx('ran this step','这一步跑过'),tx('A tinted line ran in this step, and the value beside it is this step’s.','有底色的行在这一步跑过，旁边的值是这一步的。')],
  ['sc-sw-wnow',tx('GPU computing','GPU 正在算'),tx('The outlined line is what the GPU is doing now: computing the step. Every decision above it was made at the step’s start.','描边的这一行是 GPU 正在做的事，也就是算这一步。它上面的决定都是在这一步开始时做出的。')],
  ['sc-sw-wnext',tx('runs at step end','结束时才跑'),tx('A line with a dashed bar runs when the step ends, when its requests get their output. Press ▸ to see it.','左边是虚线的行在这一步结束时才跑，那时这一步里的请求拿到结果。按 ▸ 就能看到。')],
  ['sc-sw-wskip',tx('skipped','没跑'),tx('A faded line did not run in this step.','半透明的行在这一步里没跑。')],
  ['sc-sw-wstop',tx('admission stops','接纳在这里停下'),tx('The admission loop stops at a red line: the first waiting request, and every one behind it, waits for a later step.','接纳循环在红色的行停下：排在最前面的请求和它后面的每一个，都要等到后面的步。')]]
  .map(([c,t,tip])=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${c}"></i><span>${esc(t)}</span></span>`).join('');
let wcKey='', codeTouched=false;          // codeTouched: ▸, ◂ or Play used in part 7 (the header then no longer says “nothing picked yet”)
const END_K=['fl','fr','un','ev','rt'];
function drawCode(o){const wc=$('#sc-wc',app); if(!wc) return; const on=codeOn(); wc.hidden=!on; if(!on){wcKey=''; return;}
  const phone=innerWidth<=760, cs=codeStepAt(o), km=codeM&&momentOf(codeM), k=[o.key,cs?cs.s:'-',cs?cs.phase:(play.t==null?'n':fC(play.t)),A.lang(),phone,codeM,codeTouched].join('|'); if(k===wcKey) return; wcKey=k;
  const t=$('#sc-wct',wc), N=fint(o.st.length), ob=$('#sc-wobs',wc); if(ob) ob.hidden=!(phone&&codeM);
  /* the header says what the stage shows (review B, B4): nothing picked yet (on a phone the moments are in the card below the chart, B1),
     the whole run, or an idle GPU between steps */
  if(!cs){t.textContent=play.t==null?(!codeM&&!codeTouched?(phone?tx('Pick a moment on the card below','在下面的卡片里选一个时刻'):tx('No step picked yet: pick a moment on the card, or press ▸','还没选中哪一步：在卡片上选一个时刻，或按 ▸'))
        :(phone?tx('Run done: press ▸ for step 1','运行已结束：按 ▸ 看第 1 步'):tx('The whole run is done: press ▸ for step 1, or pick a moment','整次运行已结束：按 ▸ 看第 1 步，或选一个时刻')))
      :(phone?tx('GPU idle: press ▸ for the next step','GPU 空闲：按 ▸ 看下一步'):tx(`At ${fC(play.t)} the GPU is idle: press ▸ for the next step`,`${fC(play.t)} 时 GPU 空闲：按 ▸ 看下一步`));
    wc._ps=null;
    $$('.sc-wl',wc).forEach(l=>{l.className='sc-wl'+(l.dataset.k==='h'?'':' skip'); l._st='skip'; const v=$('.sc-wv',l); v.textContent=''; v.removeAttribute('data-tipx'); v.tabIndex=-1;});}
  else {const ci=codeInfo(o,cs.s), ls=codeLines(o,ci,cs.phase);
    /* the moment's deciding line, while the panel shows the moment's step (at its end too; the eviction's line only at the end) */
    let key=''; if(km){const ms=momentStep(km); if(ms.o.key===o.key&&ms.s===cs.s&&(cs.phase==='end'||km.at!=='end')) key=KEYLN[km.id];}
    t.textContent=cs.phase==='end'?tx(`Step ${fint(cs.s+1)} of ${N}: ended at ${fC(ci.st.t1)}`,`第 ${fint(cs.s+1)} 步（共 ${N} 步）：在 ${fC(ci.st.t1)} 结束`):tx(`Step ${fint(cs.s+1)} of ${N}: the GPU is computing it`,`第 ${fint(cs.s+1)} 步（共 ${N} 步）：GPU 正在算`);
    /* the lines that run at the step's end flash once whenever a step's end comes up, background only (review A, suggestion 5; review B;
       R7.9) */
    const fresh=cs.phase==='end'&&(wc._ps!==cs.s||wc._pp!=='end'||wc._pk!==o.key); wc._ps=cs.s; wc._pp=cs.phase; wc._pk=o.key;
    $$('.sc-wl',wc).forEach(l=>{const x=ls[l.dataset.k]||{st:'skip',v:'',tip:''}; l._st=x.st; l.className='sc-wl'+(x.st==='plain'?'':' '+x.st)+(l.dataset.k===key?' key':'')+(fresh&&x.st==='ran'&&END_K.includes(l.dataset.k)?' sc-wlf':''); const v=$('.sc-wv',l); v.textContent=x.v;
      if(x.tip&&x.v){v.dataset.tipx=x.tip; v.tabIndex=0;} else {v.removeAttribute('data-tipx'); v.tabIndex=-1;}});}
  const lg=$('#sc-wleg',wc), lh=wLeg(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}}

function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="sc-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="sc-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The scheduler’s step behind every picture on this page: decode first, then running prefills, then admission first come first served while the cap, the budget and memory allow; shared prompt blocks with reference counts; idle ones evicted least recently used first.','这一页每张图背后的一步调度：先 decode，再继续在跑的 prefill，然后在上限、预算和显存允许时先来先服务地接纳；共享的 prompt block 带引用计数；空闲的按最久没用先淘汰。')}</p>`+
   `<pre class="code" id="sc-code">${CODE.map(([l,k],i)=>`<span class="sc-cl" data-line="${i+1}"${k?` data-part="${k}"`:''}>${esc(l)}</span>`).join('\n')}</pre><div class="cta-row"><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button><button class="btn ghost sm" type="button" data-act="gocode">${tx('Watch it run, step by step (part 7)','看它一步一步跑（第 7 部分）')}</button></div></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="sc-prod">`+
   tx(`<li>vLLM’s V1 scheduler, ${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')}, schedules the running requests first and then the waiting ones, all within one token budget per step. The budget is <code>max_num_batched_tokens</code> and the cap <code>max_num_seqs</code>, in ${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')}; <code>long_prefill_token_threshold</code> (0, off, by default) caps how many tokens one long prompt may take in a step.</li>`+
      `<li>Unlike this lab, vLLM does not set room aside for a whole answer: it admits more requests and preempts a running one when its blocks run out.</li>`+
      `<li>vLLM’s automatic prefix caching keys each full block by its own tokens and every token before it (chained through the parent block’s hash), shares it with a reference count, and keeps a freed block cached until a new allocation evicts it, least recently used first: ${cite(SRC.design,'<code>docs/design/prefix_caching.md</code>')}. SGLang keeps prefixes in a radix tree, ${cite(SRC.radix,'<code>radix_cache.py</code>')}, and evicts its leaves first.</li>`,
      `<li>vLLM 的 V1 调度器（${cite(SRC.vsched,'<code>vllm/v1/core/sched/scheduler.py</code>')}）先调度在跑的请求，再调度排队的请求，都在每步同一个 token 预算之内。预算是 <code>max_num_batched_tokens</code>，上限是 <code>max_num_seqs</code>，都在 ${cite(SRC.vcfg,'<code>vllm/config/scheduler.py</code>')} 里；<code>long_prefill_token_threshold</code>（默认 0，即关闭）限制一个长 prompt 在一步里最多占多少 token。</li>`+
      `<li>和这个 Lab 不同，vLLM 不给整个回答预留空间：它接纳更多请求，block 不够时再抢占一个在跑的请求。</li>`+
      `<li>vLLM 的自动前缀缓存用这个 block 自己的 token 和它之前的全部 token 算出 key（通过父 block 的哈希串起来），带引用计数共享，释放后的 block 留在缓存里，直到新的分配按最久没用先淘汰它：${cite(SRC.design,'<code>docs/design/prefix_caching.md</code>')}。SGLang 把前缀存在一棵 radix tree 里（${cite(SRC.radix,'<code>radix_cache.py</code>')}），先淘汰叶子。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.orca,'Orca: A Distributed Serving System for Transformer-Based Generative Models')}<span>${tx('Yu et al., OSDI 2022: scheduling at every step (iteration-level scheduling), the idea behind continuous batching','Yu 等，OSDI 2022：每一步都调度（iteration-level scheduling），continuous batching 的来源')}</span></li>`+
   `<li>${cite(SRC.serve,'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve')}<span>${tx('Agrawal et al., OSDI 2024: chunked prefills and stall-free schedules that add new requests without pausing ongoing decodes','Agrawal 等，OSDI 2024：chunked prefill 加上不卡顿的调度，加入新请求时不暂停正在进行的 decode')}</span></li>`+
   `<li>${cite(SRC.sglang,'SGLang: Efficient Execution of Structured Language Model Programs')}<span>${tx('Zheng et al., NeurIPS 2024: RadixAttention, which keeps finished requests’ KV cache in a radix tree and reuses the longest matching prefix','Zheng 等，NeurIPS 2024：RadixAttention，把已完成请求的 KV cache 存在一棵 radix tree 里，复用最长的相同前缀')}</span></li>`+
   `<li>${cite(SRC.vllm,'Efficient Memory Management for Large Language Model Serving with PagedAttention')}<span>${tx('Kwon et al., SOSP 2023: a KV cache in fixed-size blocks, which lets requests share blocks','Kwon 等，SOSP 2023：按固定大小的 block 存 KV cache，让不同请求可以共用 block')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="sc-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="sc-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="sc-app m2" id="sc-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="sc-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){const cur_=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many requests meet their goal.','选一个场景，看看有多少个请求达标。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur_}"><b>${esc(L(p.t))}</b><span>${esc(p.d())}</span></button>`).join('')}</div>`+
   `<div class="obs" id="sc-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;}
const showMeOn=()=>{const s=STEPS[step]; if(s.id==='predict') return !!pred; if(s.id==='code') return !codeSeen.size; if(!s.showMe) return false; return Object.entries(s.showMe).some(([k,v])=>S[k]!==v);};
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!s.expect()&&!tried[s.id])||(!!s.gate&&!s.gate());}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:showMeOn(),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#sc-rail',app); if(!rail) return; rail.innerHTML=railFull(); A.ui.initRanges(rail); updateRail(); }
/* the parts of the rail that follow the state and the playhead */
function obsOf(s){
  if(s.id==='mission') return seen.has(cur().key)?s.obs():'';
  if(s.id==='predict') return s.obs();
  if(s.id==='code') return obsCode();
  if(!s.lever) return '';
  if(!s.expect()&&!tried[s.id]) return '';
  const o=cur(); if(!seen.has(o.key)) return `<p class="small">${runLine(o)}</p>`;
  return {budget:obsBudget,cache:obsCache,cb:obsCb,cap:obsCap}[s.lever]();}
function updateRail(){
  const rail=$('#sc-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    if(box){const h=obsOf(s), html=h?`<div class="eyebrow">${L(s.label||{en:'Result',zh:'结果'})}</div>`+h:'';
      box.hidden=!html; if(box._h!==html){const again=box._h&&html&&box._step===step; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);
        const dw=$('details.sc-why',box); if(dw){dw.open=whyOpen; dw.addEventListener('toggle',()=>{whyOpen=dw.open;});}}}
    if(s.lever){const Lv=LEVER[s.lever];
      $$('[data-lever]',rail).forEach(b=>{const v=leverVal(s.lever,b.dataset.lever); b.setAttribute('aria-pressed',String(String(v)===String(Lv.val())&&(s.lever!=='cap'||true)));
        const r=$('.lv-r',b), h=seen.has(partKeyFor(s.id,v))?Lv.res(v):''; if(r&&r._h!==h){r.innerHTML=h; r._h=h;}});
      const pr=$('#sc-polres',rail); if(pr){const st=getRun('shared',{...BASE,budget:2048,cache:true,cap:S.cap,policy:'static'}), co=getRun('shared',{...BASE,budget:2048,cache:true,cap:S.cap});
        pr.textContent=seen.has(st.key)?tx(`Static batching, cap ${S.cap}: ${st.m.all.ok} / ${st.m.all.n} meet the goal (continuous batching: ${co.m.all.ok} / ${co.m.all.n}).`,`static batching，上限 ${S.cap}：${st.m.all.ok} / ${st.m.all.n} 达标（continuous batching 时 ${co.m.all.ok} / ${co.m.all.n}）。`):'';}
      $$('.seg-ctl[data-key="policy"]',rail).forEach(g=>A.ui.setPressed(g,S.policy));}
    if(s.id==='code') $$('[data-moment]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.moment===codeM)); const r=$('.sc-momok',b), v=codeSeen.has(b.dataset.moment)?' ✓':''; if(r&&r.textContent!==v) r.textContent=v;});
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!showMeOn();
    $$('.sc-pbtn',rail).forEach(b=>{b.textContent=playLabel();});
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#sc-insight',rail); if(ins){const html=`<div class="eyebrow">${esc(c>=0?L(PRESETS[c].t):'')}</div>`+scenHTML(); if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again&&seen.has(cur().key)) A.ui.obsBump(ins);}}
  } else {
    const res=$('#sc-ch-res',rail); if(res){const html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
const tNow=()=>play.t;                                  // null: the whole run
const full=()=>play.t==null||RM();                      // the timeline is drawn complete
function statusOf(o){
  if(mode==='challenge'&&!chRunKey) return {cls:'neutral',txt:tx('Initial setup','初始设置'),short:tx('Initial setup','初始设置'),tip:tx('The stage shows the service as it runs today. Choose your settings in the task card and run them.','舞台上是今天的服务。在任务卡里选好设置再运行。')};
  if(play.t!=null){const txt=play.playing?tx('Running','运行中'):(RM()?tx(`Drawn whole; playhead at ${fC(play.t)}`,`整次运行已画出，播放头在 ${fC(play.t)}`):(play.t>1e-9?tx(`Paused at ${fC(play.t)}`,`停在 ${fC(play.t)}`):tx('Not run yet','还没开始跑')));
    return {cls:'neutral',txt,short:txt,tip:tx('Whole-run numbers appear when the replay reaches the end.','重放到结尾时，才显示整次运行的数字。')};}
  const ok=o.m.all.ok_frac>=E.TARGET-EPS;
  return {cls:ok?'ok':'bad',txt:tx(`${o.m.all.ok} / ${o.m.all.n} meet their goal`,`${o.m.all.n} 个里 ${o.m.all.ok} 个达标`),short:tx(`${o.m.all.ok} / ${o.m.all.n} met`,`${o.m.all.ok} / ${o.m.all.n} 达标`),
    tip:tx(`Green when at least ${pct0(E.TARGET)} of the requests meet their goal: ${o.m.all.ok} ÷ ${o.m.all.n} = ${pct0(o.m.all.ok_frac)}.`,`至少 ${pct0(E.TARGET)} 的请求达标时为绿色：${o.m.all.ok} ÷ ${o.m.all.n} = ${pct0(o.m.all.ok_frac)}。`)};}
function drawHead(o){
  $('#sc-st-title',app).textContent=tx(`${MODEL} on one ${GPUN}: a chat service`,`一张 ${GPUN} 跑 ${MODEL}：一个聊天服务`);
  $('#sc-st-sub',app).textContent=loadDesc(o.load);
  const s0=o.s, pill=(t,tip,on)=>`<span class="sc-pill${on?' on':''}" tabindex="0" data-tipx="${esc(tip)}">${esc(t)}</span>`;
  const ph=pill(s0.policy==='static'?'static':'continuous',s0.policy==='static'?TIP.static():TIP.continuous(),s0.policy!==BASE.policy)+pill(tx(`cap ${s0.cap}`,`上限 ${s0.cap}`),TIP.cap(),s0.cap!==BASE.cap)+
    pill(s0.budget?tx(`budget ${fint(s0.budget)}`,`预算 ${fint(s0.budget)}`):tx('no budget','不设预算'),TIP.budget(),s0.budget!==BASE.budget)+
    pill(s0.cache?(s0.cb==null?tx('caching on','缓存开'):tx(`caching on, ${fCB(s0.cb)}`,`缓存开，${fCB(s0.cb)}`)):tx('caching off','缓存关'),TIP.cache(),!!s0.cache);
  const pe=$('#sc-pills',app); if(pe._h!==ph){pe.innerHTML=ph; pe._h=ph;}
  const s=statusOf(o), st=$('#sc-status',app); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
  const kd=$('#sc-kinds',app), kh=o.kinds.map(k=>`<span tabindex="0" data-tipx="${esc(kindTip(k,o))}"><i class="sc-sw" style="background:var(--sc-${k})"></i>${esc(kindL(k,o.R.some(r=>r.app)))}</span>`).join('');
  if(kd._h!==kh){kd.innerHTML=kh; kd._h=kh;}}
function kindTip(k,o){const n=o.R.filter(r=>r.kind===k).length;
  if(k==='doc') return tx(`${n} long document${n>1?'s':''}: goal first token within ${fS(G.doc.ttft)}. The same colour in every view.`,`${n} 份长文档：目标是 ${fS(G.doc.ttft)} 内出首 token。每个视图里都是这个颜色。`);
  if(k==='chat') return o.R.some(r=>r.app)?tx(`${n} plain chat${n>1?'s':''}, with no app prompt: the goal of every short chat, first token within ${fS(G.chat.ttft)} and no gap over ${fS(G.chat.gap)}. The same colour in every view.`,`${n} 个普通对话，不带应用 prompt：目标和所有短对话一样，${fS(G.chat.ttft)} 内出首 token，间隔不超过 ${fS(G.chat.gap)}。每个视图里都是这个颜色。`)
    :tx(`${n} short chats: goal first token within ${fS(G.chat.ttft)} and no gap over ${fS(G.chat.gap)}. The same colour in every view.`,`${n} 个短对话：目标是 ${fS(G.chat.ttft)} 内出首 token，间隔不超过 ${fS(G.chat.gap)}。每个视图里都是这个颜色。`);
  const p=o.R.find(r=>r.kind===k).pre;
  if(DOCAPPS.includes(k)) return tx(`${n} question${n>1?'s':''} about ${L(KIND[k])}; ${n>1?'each question starts':'it starts'} with the document’s ${fint(p)} tokens, and ${n>1?'they share':'it has'} the goal of every short chat. The same colour in every view.`,`${n} 个问${L(KIND[k])}的问题；每个问题都以这份文档的 ${fint(p)} 个 token 开头，目标和所有短对话一样。每个视图里都是这个颜色。`);
  return tx(`${n} short chat${n>1?'s':''} of the ${L(APPN[k])} app, ${n>1?'each ':''}starting with its ${fint(p)}-token system prompt; the goal of every short chat. The same colour in every view.`,`${L(APPN[k])}应用的 ${n} 个短对话，${n>1?'每个都':'它'}以 ${fint(p)} 个 token 的 system prompt 开头；目标和所有短对话一样。每个视图里都是这个颜色。`);}
/* the live strip: queue → GPU step → tokens out, at the playhead (the whole run: its end) */
function drawStrip(o){
  const tt=play.t==null?o.end+1:play.t, ev=endView(o), s=ev>=0?ev:stepAt(o,tt), k=doneAt(o,tt), phone=innerWidth<=760, narrow=innerWidth<1100, R=o.R;
  const waiting=R.filter(r=>r.arr<=tt+EPS&&!admittedAt(r,tt,ev)), groups=[];
  waiting.forEach(r=>{const w=reasonAt(o,r,tt,ev), g=groups.find(x=>x.w===w); if(g) g.rs.push(r); else groups.push({w,rs:[r]});});
  const qw=($('#sc-q',app)||{}).clientWidth||200, maxC=Math.max(4,Math.floor((qw-10)/15)*2-groups.length*4-2); let shown=0;
  const chips=groups.map(g=>{const cs=g.rs.map(r=>{if(shown>=maxC) return ''; shown++; return `<button type="button" class="sc-chip${sel===r.id?' sel':''}${relOf().has(r.id)?' rel':''}" data-sel="${r.id}" style="--k:var(--sc-${r.kind})" aria-label="${esc(rName(r)+tx(', waiting: ',' 在等：')+whyLong(g.w,o))}" data-tipx="${esc(rName(r)+tx(' waits ',' 在')+whyLong(g.w,o)+tx('.','。'))}"></button>`;}).join('');
    return `<span class="sc-qg" data-why="${g.w}"><span class="sc-qw" data-tipx="${esc(tx(`${g.rs.length} waiting ${whyLong(g.w,o)}.`,`${g.rs.length} 个在${whyLong(g.w,o)}。`))}" tabindex="0">${ICON[g.w]}${esc(g.w==='cap'?tx(`cap ${o.cfg.cap}`,`上限 ${o.cfg.cap}`):(phone&&g.w==='budget'?tx('budget','预算用完'):L(WHY[g.w])))}</span>${cs}</span>`;}).join('');
  const more=waiting.length-shown;
  const ch=$('#sc-chips',app), chh=(waiting.length?chips+(more>0?`<span class="sc-more mono">+${more}</span>`:''):`<span class="sc-empty">${tx('empty','空')}</span>`);
  if(ch._h!==chh){ch.innerHTML=chh; ch._h=chh;}
  $('#sc-qn',app).textContent=String(waiting.length);
  const why=new Set(groups.map(g=>g.w));
  /* the GPU step */
  const gh=$('#sc-gh',app), bar=$('#sc-gbar',app), dur=$('#sc-gdur',app), mt=$('#sc-meters',app), gpu=$('#sc-gpu',app);
  if(s<0){gh.textContent=play.t==null||tt>=o.end?tx('GPU: idle, all done','GPU：空闲，请求全部完成'):tx('GPU: idle','GPU：空闲');
    const bh=`<div class="sc-idle">${play.t==null?`<span class="sc-lw">${tx(`${o.R.length} requests done at ${fC(o.end)}`,`${o.R.length} 个请求在 ${fC(o.end)} 全部完成`)}</span><span class="sc-ls">${fC(o.end)}</span>`:'—'}</div>`; if(bar._h!==bh){bar.innerHTML=bh; bar._h=bh;}
    dur.innerHTML='<span class="sc-gr1"></span><span class="sc-gr2"></span>'; mt.innerHTML=''; gpu.classList.remove('warn','hot');}
  else {const info=stepInfo(o,s), st=info.s, dt=st.t1-st.t0, warn=dt>G.chat.gap+EPS;
    gh.innerHTML=tx(`GPU step ${fint(s+1)}<span class="sc-of"> of ${fint(o.st.length)}</span>${ev>=0?', done':''}`,`GPU 第 ${fint(s+1)} 步<span class="sc-of">（共 ${fint(o.st.length)} 步）</span>${ev>=0?'，已结束':''}`);
    const bud=o.cfg.budget||0, hide=!revealed();
    /* at a step's end the meter counts the requests that ran in it; the clock under the controls counts those still running (review B, B11) */
    const runL=ev>=0?tx('ran','跑过'):tx('running','在跑'), runTip=ev>=0?tx(`${st.nrun} requests ran in this step; the cap is ${o.cfg.cap}.`,`这一步有 ${st.nrun} 个请求跑过；上限是 ${o.cfg.cap}。`):tx(`${st.nrun} requests run in this step; the cap is ${o.cfg.cap}.`,`这一步有 ${st.nrun} 个请求在跑；上限是 ${o.cfg.cap}。`);
    mt.innerHTML=hide?`<span class="sc-meter${why.has('cap')?' hot':''}" tabindex="0" data-tipx="${esc(runTip)}"><span>${runL}</span><i><b style="width:${Math.min(100,st.nrun/o.cfg.cap*100).toFixed(1)}%"></b></i><em class="mono">${st.nrun}/${o.cfg.cap}</em></span>`:`<span class="sc-meter${why.has('cap')?' hot':''}" tabindex="0" data-tipx="${esc(runTip)}"><span>${runL}</span><i><b style="width:${Math.min(100,st.nrun/o.cfg.cap*100).toFixed(1)}%"></b></i><em class="mono">${st.nrun}/${o.cfg.cap}</em></span>`+
      `<span class="sc-meter${why.has('budget')?' hot':''}" tabindex="0" data-tipx="${esc(bud?tx(`This step computes ${fint(st.T)} tokens; the budget is ${fint(bud)} per step, decode tokens included.`,`这一步算 ${fint(st.T)} 个 token；每步的预算是 ${fint(bud)} 个，decode 的 token 也算在内。`):tx(`This step computes ${fint(st.T)} tokens; there is no budget, so a whole prompt fits in one step.`,`这一步算 ${fint(st.T)} 个 token；没有预算，整个 prompt 可以放进一步。`))}"><span>${tx('tokens','token')}</span><i><b style="width:${bud?Math.min(100,st.T/bud*100).toFixed(1):100}%"></b></i><em class="mono">${fint(st.T)}${bud?'/'+fint(bud):''}</em></span>`;
    const W=Math.max(120,bar.clientWidth||300), nd=info.dec.length, pre=info.pre, preT=pre.reduce((a,p)=>a+p.n,0);
    const dw=pre.length?Math.max(2,Math.min(7,Math.floor(W*0.32/Math.max(1,nd)))):(W-2)/Math.max(1,nd);
    let x=0; const parts=[];
    info.dec.forEach(id=>{const r=R[id]; parts.push(`<i class="sc-sl${sel===id?' sel':''}${relOf().has(id)?' rel':''}" data-sel="${id}" style="left:${x.toFixed(1)}px;width:${Math.max(1,dw-1).toFixed(1)}px;--k:var(--sc-${r.kind})"></i>`); x+=dw;});
    const rest=Math.max(0,W-x), pw=pre.map(p=>Math.max(16,rest*p.n/Math.max(1,preT))), sc=pw.reduce((a,b)=>a+b,0)>rest&&rest>0?rest/pw.reduce((a,b)=>a+b,0):1;
    pre.forEach((p,i)=>{const r=R[p.id], w=pw[i]*sc, name=r.kind==='doc'?tx('doc','文档'):(r.kind==='chat'?tx('chat','对话'):L(APPN[r.kind])), lab=`${name} ${fint(p.n)}`;
      parts.push(`<i class="sc-pb${sel===p.id?' sel':''}${relOf().has(p.id)?' rel':''}" data-sel="${p.id}" style="left:${x.toFixed(1)}px;width:${Math.max(2,w-1).toFixed(1)}px;--k:var(--sc-${r.kind})" data-tipx="${esc(tx(`${rName(r)}: prompt tokens ${fint(p.a+1)} to ${fint(p.b)}${p.whole?', its whole prompt in this one step':''}.`,`${rName(r)}：prompt 的第 ${fint(p.a+1)} 到 ${fint(p.b)} 个 token${p.whole?'，整个 prompt 都在这一步里':''}。`))}"><em>${w>lab.length*6.4+8?esc(lab):(w>fint(p.n).length*6.4+6?fint(p.n):'')}</em></i>`); x+=w;});
    const frac=ev>=0?1:Math.max(0,Math.min(1,(tt-st.t0)/dt));
    const bh=(hide?`<i class="sc-pb sc-anon" style="left:0;width:${Math.max(2,W-1).toFixed(1)}px"></i>`:parts.join(''))+`<i class="sc-sweep" style="width:${(frac*100).toFixed(2)}%"></i>`;
    bar.innerHTML=bh; bar._h=bh;
    const lim=G.chat.gap*2, gw=Math.min(100,dt/lim*100);
    const inStep=play.t!=null&&tt>st.t0+EPS&&tt<st.t1-EPS;
    dur.innerHTML=`<span class="sc-gr1"><span class="mono sc-dt">${fS(dt)}</span>${inStep?`<span class="mono sc-elap">${tx(`${fS(tt-st.t0)} so far`,`已过 ${fS(tt-st.t0)}`)}</span>`:''}<span class="sc-gauge" tabindex="0" data-tipx="${esc(hide?tx(`This step takes ${fS(dt)}; the dashed line is the chats’ gap goal, ${fS(G.chat.gap)}.`,`这一步要 ${fS(dt)}；虚线是对话的间隔目标 ${fS(G.chat.gap)}。`):tx(`This step takes ${fS(dt)}. Every running request gets its next token only when it ends, so a step longer than ${fS(G.chat.gap)} (the dashed line) breaks every running chat’s gap goal.`,`这一步要 ${fS(dt)}。每个在跑的请求都要等它结束才拿到下一个 token，所以超过 ${fS(G.chat.gap)}（虚线）的一步，会让每个在跑的对话超过间隔目标。`))}"><i style="width:${gw.toFixed(1)}%"></i><b></b></span>`+
      `</span><span class="sc-gr2">`+((warn?`<span class="sc-warn">${narrow?tx('long step: nobody gets a token','长步：谁也拿不到 token'):tx('long step: no token until it ends','长步：结束前谁也拿不到 token')}${dt>lim&&!narrow?' ▸':''}</span>`:(hide?'':`<span class="sc-dnote">${nd?tx(`${nd} decode token${nd>1?'s':''}`,`${nd} 个 decode token`):''}${pre.length?(nd?tx(' + ','，'):'')+tx(`${fint(preT)} prefill`,`prefill ${fint(preT)} 个`):''}</span>`))+`</span>`);
    gpu.classList.toggle('warn',warn); gpu.classList.toggle('hot',why.has('step')&&warn);}
  /* tokens out: who got a token when the last step ended */
  const on=$('#sc-on',app), dots=$('#sc-dots',app), note=$('#sc-onote',app), outBox=$('#sc-out',app);
  let total=0; R.forEach(r=>{let lo=0, hi=r.toks.length; while(lo<hi){const m=(lo+hi)>>1; if(r.toks[m]<=tt+EPS) lo=m+1; else hi=m;} total+=lo;});
  on.textContent=fint(total);
  if(k<0){if(dots._k!==-1){dots.innerHTML=`<span class="sc-empty">${tx('none yet','还没有')}</span>`; dots._k=-1;} note.textContent=''; outBox.classList.remove('hot');}
  else {const info=stepInfo(o,k), ow=(outBox.clientWidth||160)-12, maxD=Math.max(6,Math.floor(ow/(phone?11:12))*2-2), fs=new Set(info.first);
    if(dots._k!==k||dots._sel!==sel){dots._k=k; dots._sel=sel;
      dots.innerHTML=info.out.slice(0,maxD).map(id=>`<button type="button" class="sc-dot${fs.has(id)?' f':''}${sel===id?' sel':''}" data-sel="${id}" style="--k:var(--sc-${R[id].kind})" aria-label="${esc(rName(R[id])+(fs.has(id)?tx(': first token',': 首 token'):tx(': a token',': 一个 token')))}"></button>`).join('')+(info.out.length>maxD?`<span class="sc-more mono">+${info.out.length-maxD}</span>`:'');}
    const since=tt-o.st[k].t1, stalled=s>=0&&since>G.chat.gap+EPS;
    note.textContent=play.t==null?(phone?tx(`all ${fint(total)} out`,`已全部输出`):tx(`all ${fint(total)} answer tokens out`,`全部 ${fint(total)} 个回答 token 已输出`)):(stalled?(phone?tx(`no token for ${fS(since)}`,`已经 ${fS(since)} 没有新 token`):tx(`no new token for ${fS(since)}`,`已经 ${fS(since)} 没有新 token`)):tx(`${info.out.length} at the last step’s end`,`上一步结束时 ${info.out.length} 个`));
    outBox.classList.toggle('hot',stalled);}
  $('#sc-mem',app).classList.toggle('hot',why.has('memory'));
}
/* the KV pool after the last step that has ended (the whole run: at its peak) */
/* the memory bar's scale: the same for every option the view compares, so 26 GiB against 2 GiB still shows (a nice ceiling of their peaks) */
function memScale(o){const ks=[o];
  if(mode==='guided'){const s=STEPS[step]; if(s.lever){const c=PART_CFG[s.id]; s.opts.forEach(v=>ks.push(getRun(c.load,{...c,...(s.id==='cap'?{policy:S.policy}:{}),...LEVER[s.lever].set(v)})));} else ks.push(baseOf(o.load));}
  else ks.push(baseOf(o.load));
  const pk=Math.max(...ks.map(x=>x.m.all.peak_gib)); for(const g of [2,4,8,16,32]) if(pk<=g*0.97) return g; return POOL_GIB;}
function placeLabels(box){const W=box.clientWidth; if(!W) return; const els=[...box.children]; if(!els.length) return;
  const it=els.map(e=>{const w=e.offsetWidth, x=parseFloat(e.style.left)/100*W; return {e,w,l:x-w/2};}).sort((a,b)=>a.l-b.l);
  let prev=-Infinity; it.forEach(i=>{i.l=Math.max(i.l,prev+3,0); prev=i.l+i.w;});
  let next=W; for(let j=it.length-1;j>=0;j--){const i=it[j]; if(i.l+i.w>next){i.l=Math.max(0,next-i.w);} next=i.l-3;}
  it.forEach(i=>{i.e.style.left=i.l.toFixed(1)+'px'; i.e.style.transform='none';});}
/* what an app's shared prefix is: a document (the four-document load) or a system prompt (review r3, finding 6) */
const preOf=(a,low)=>DOCAPPS.includes(a)?tx(low?KIND[a].en:cap1(KIND[a].en),KIND[a].zh):tx(low?'the app’s system prompt':`The ${L(APPN[a])} app’s system prompt`,low?'应用的 system prompt':`${L(APPN[a])}应用的 system prompt`);
function drawMem(o){
  const tt=play.t==null?null:play.t, ev=endView(o), cs=tt==null||ev>=0?-1:stepAt(o,tt), k=tt==null?peakStepPre(o):(ev>=0?ev:(cs>=0?cs:doneAt(o,tt))), d=(tt==null||cs>=0)?memPre(o,k):memAt(o,k), scG=memScale(o), P=scG*C.GiB/BLK_B, bar=$('#sc-mbar',app), labs=$('#sc-mlab',app), segs=[], lab=[];
  let x=0; const pc=b=>b/P*100;
  const push=(cls,b,extra,style)=>{if(b<=0) return; segs.push(`<i class="sc-ms ${cls}" style="left:${pc(x).toFixed(3)}%;width:${pc(b).toFixed(3)}%;${style||''}"${extra||''}></i>`); const mid=x+b/2; x+=b; return mid;};
  /* evictions of the last 0.6 s, each at its own time: while allocating at a step's start, or after a release at its end. Inside a step,
     its end's evictions have not happened yet (review B, B2) */
  const evNow=[]; if(tt!=null){const tMin=tt-0.6; for(let s=k;s>=0&&o.st[s].t1>tMin;s--){
    [[o.evAtA.get(s),o.st[s].t0],[o.evAtT.get(s),o.st[s].t1]].forEach(([e,t])=>{if(e&&t<=tt+EPS) Object.entries(e).forEach(([a,n])=>evNow.push({app:a,n,age:tt-t}));});}}
  APPS.forEach(a=>{const all=(d.pres&&d.pres[a])||0, inU=d.inuse[a]||0, idle=Math.max(0,all-inU);
    if(inU){const mid=push('sc-ms-sh',inU,` data-app="${a}" data-tipx="${esc(tx(`${preOf(a)} takes ${fint(inU)} shared blocks (${fGs(inU)}), and the ${d.users[a]===1?'one running request that uses it holds':`${d.users[a]} running requests that use it share`} this one copy.`,zfix(`${preOf(a)}占 ${fint(inU)} 个共享 block（${fGs(inU)}），${d.users[a]} 个在跑的请求共用这一份。`)))}"`,`--k:var(--sc-${a})`);
      lab.push(`<span class="sc-mb" style="left:${pc(mid).toFixed(2)}%;--k:var(--sc-${a})">${esc(L(APPN[a]))} ×${d.users[a]}</span>`);}
    if(idle){const mid=push('sc-ms-idle',idle,` data-tipx="${esc(tx(`${preOf(a)} stays cached in ${fint(idle)} blocks (${fGs(idle)}) while no request uses it.`,zfix(`${preOf(a)}没有请求在用，仍缓存着，占 ${fint(idle)} 个 block（${fGs(idle)}）。`)))}"`,`--k:var(--sc-${a})`);
      if(!inU) lab.push(`<span class="sc-mb sc-mb-idle" style="left:${pc(mid).toFixed(2)}%;--k:var(--sc-${a})">${esc(L(APPN[a]))}</span>`);}
    evNow.filter(e=>e.app===a).forEach(e=>{const op=Math.max(0.15,1-e.age/0.6); segs.push(`<i class="sc-ms sc-ms-ev" style="left:${pc(x).toFixed(3)}%;width:${pc(e.n).toFixed(3)}%;opacity:${op.toFixed(2)}"></i>`);
      lab.push(`<span class="sc-mb sc-mb-ev" style="left:${pc(x+e.n/2).toFixed(2)}%;opacity:${op.toFixed(2)}">${tx(`${L(APPN[a])} evicted`,`${L(APPN[a])}被淘汰`)}</span>`);});});
  d.own.forEach(q=>{const r=o.R[q.id]; push('sc-ms-own'+(sel===q.id?' sel':'')+(relOf().has(q.id)?' rel':''),q.b,` data-sel="${q.id}"`,`--k:var(--sc-${r.kind})`);});
  if(d.mem.resv) push('sc-ms-resv',d.mem.resv,` data-tipx="${esc(tx(`Set aside for the running requests’ answers and prompts still to come: ${fint(d.mem.resv)} blocks (${fGs(d.mem.resv)}). Admission reserves it, so nothing is ever preempted.`,`为在跑的请求还没写入的 prompt 和回答预留：${fint(d.mem.resv)} 个 block（${fGs(d.mem.resv)}）。接纳时就预留好，所以不会有请求被抢占。`))}"`);
  const bh=segs.join(''); if(bar._h!==bh){bar.innerHTML=bh; bar._h=bh;}
  const lh=lab.join('')+'|'+bar.clientWidth; if(labs._h!==lh){labs.innerHTML=lab.join(''); labs._h=lh; placeLabels(labs);}
  const zoom=scG<POOL_GIB-0.01, mt=zoom?tx(`KV cache memory: the first ${scG} ${GIB()} of the ${fG(POOL_GIB)} pool (${fint(C.POOL_T)} tokens)`,`KV cache 显存：${fG(POOL_GIB)} 显存池（${fint(C.POOL_T)} 个 token）的前 ${scG} ${GIB()}`)
    :tx(`KV cache memory: the pool of ${fint(C.POOL_T)} tokens (${fG(POOL_GIB).replace('GiB','')}${GIB()})`,`KV cache 显存：${fint(C.POOL_T)} 个 token 的显存池（${fG(POOL_GIB).replace('GiB','')}${GIB()}）`);
  const me=$('#sc-memt',app); if(me._h!==mt){me.innerHTML=mt; me._h=mt;}
  const narrow=(bar.clientWidth||600)<480, ax=$('#sc-maxis',app), ak=scG+'|'+A.lang()+'|'+narrow; if(ax&&ax._k!==ak){ax._k=ak; const n=4, v=[]; for(let i=0;i<=n;i++) if(!narrow||i%2===0) v.push([i,scG*i/n]);
    ax.innerHTML=v.map(([i,g])=>`<span style="left:${(i/n*100).toFixed(1)}%"${i===n?' class="sc-ax-r"':(i===0?' class="sc-ax-l"':'')}>${fixd(g,g%1?1:0)}${i===n?(zoom?tx(` of ${fixd(POOL_GIB,1)}${NB}GiB`,` / ${fixd(POOL_GIB,1)}${NB}GiB`):NB+'GiB'):''}</span>`).join('');}
  $('#sc-memv',app).textContent=k<0?tx('empty','空'):(tt==null?tx(`at the peak, during the step ending at ${fC(o.st[k].t1)}: ${fG(bGiB(d.mem.used))} held`,`峰值（在 ${fC(o.st[k].t1)} 结束的那一步里）：占用 ${fG(bGiB(d.mem.used))}`):tx(`${fG(bGiB(d.mem.used))} held, ${fGs(d.mem.resv)} set aside`,`占用 ${fG(bGiB(d.mem.used))}，预留 ${fGs(d.mem.resv)}`));
}
function memLegend(o){const it=(cls,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i><span>${esc(t)}</span></span>`, cache=!!o.cfg.cache;
  return (cache?it('sc-sw-sh',tx('shared prompt ×N users','共享 prompt ×N 个在用'),tx('A system prompt’s blocks, one copy shared by the N running requests of that app; the label names the app.','一个 system prompt 的 block，这个应用的 N 个在跑的请求共用这一份；标签写着应用名。')):'')+
   it('sc-sw-own',tx('a request’s own KV','请求自己的 KV'),tx('The KV cache of one running request’s own prompt tokens and answer so far; the thin top stripe is the request’s colour.','一个在跑的请求自己的 prompt token 和已生成回答的 KV cache；上沿的细条是这个请求的颜色。'))+
   (cache?it('sc-sw-idle',tx('hatched: cached, idle','斜纹：缓存着，没人在用'),tx('A system prompt no running request uses, kept for the next request of its app; evicted first when room runs short.','没有在跑的请求在用的 system prompt，留给这个应用的下一个请求；空间不够时最先被淘汰。')):'')+
   it('sc-sw-resv',tx('outlined: set aside','空心框：预留'),tx('Room set aside for the running requests’ answers and the rest of their prompts.','给在跑的请求的回答和剩下的 prompt 预留的空间。'))+
   it('sc-sw-free',tx('grey: free','灰色：空闲'),tx('Free KV memory.','空闲的 KV 显存。'))+
   (cache&&o.run.evictions?it('sc-sw-ev',tx('dashed: just evicted','虚线框：刚被淘汰'),tx('Blocks evicted in the last moment of the run: the least recently used idle prompt goes first.','刚刚被淘汰的 block：最久没用的空闲 prompt 先走。')):'');}
/* the timeline: one row per request in arrival order, drawn on a canvas */
const PAT=new Map();
function hatch(g,color){const key=color; if(PAT.has(key)) return PAT.get(key); const c=document.createElement('canvas'); c.width=c.height=6; const x=c.getContext('2d');
  x.strokeStyle=color; x.globalAlpha=0.75; x.lineWidth=1.2; x.beginPath(); x.moveTo(-1,7); x.lineTo(7,-1); x.stroke(); const p=g.createPattern(c,'repeat'); PAT.set(key,p); return p;}
let tlGeo=null, focusRow=-1;
/* the most height the timeline's rows may take on a desktop: --sc-tlh in serving-capstone.css (review r5, B4: 300 px before, and the stage
   did not fit the window in any part) */
const tlBudget=box=>{const v=parseFloat(getComputedStyle(box).getPropertyValue('--sc-tlh')); return v>=150&&v<=300?v:300;};
function drawTL(){
  const o=cur(), cv=$('#sc-tl',app), box=$('#sc-tlbox',app); if(!cv||!box) return;
  const W=Math.max(200,box.clientWidth), n=o.R.length, phone=innerWidth<=760;
  const rh=phone?Math.max(2,Math.min(5,Math.floor(156/n))):Math.max(4,Math.min(8,Math.floor(tlBudget(box)/n))), dpr=Math.min(2,window.devicePixelRatio||1);
  /* the document-prefill label, both halves (R5.8; review B4, N1): in the empty corner below the document's row and left of its arrival
     when it fits there (rows are in arrival order, so no row reaches that corner); otherwise in a band of its own above the rows, with a
     thin leader line down to the prefill, so it never covers the rows of the chats that keep answering. The band is kept whether or not
     the label shows yet, so the page does not jump when it appears */
  const pl=$('#sc-pflab',app), dq=o.docs.find(r=>r.chunks.length), span0=spanFor(o), X0=t=>9+t/span0*(W-12); let band=0, lab=null;
  if(pl&&dq){const s0=o.st[dq.chunks[0][0]].t0, s1=o.st[dq.chunks[dq.chunks.length-1][0]].t1, dur=fR(s1-s0), h2=tx(`document prefill ${dur}<br>new requests wait`,`文档 prefill ${dur}<br>新请求要等`);
    if(pl._h!==h2){pl.innerHTML=h2; pl._h=h2;} pl.classList.add('sc-pf2'); const was=pl.hidden; pl.hidden=false; const lw=pl.offsetWidth||170, lh=pl.offsetHeight||30; pl.hidden=was;
    const xa=X0(dq.arr), corner=xa-6-lw>=9&&4+(dq.id+1)*rh+2+lh<=n*rh+8; if(!corner) band=lh+8; lab={s0,s1,lw,lh,xa,corner};}
  const top=4+band, H=n*rh+top+4;
  if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); cv.style.width=W+'px'; cv.style.height=H+'px';}
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const cs=getComputedStyle(app), cv_=k=>cs.getPropertyValue(k).trim(), ink=cv_('--ink'), ink3=cv_('--ink-3'), line=cv_('--line'), okC=cv_('--c-ok'), badC=cv_('--c-oom');
  const col={}; KORDER.forEach(k=>{col[k]=cv_('--sc-'+k)||ink3;});
  const span=spanFor(o), GX=9, X=t=>GX+t/span*(W-GX-3), tt=play.t==null?Infinity:play.t, cut=full()?Infinity:tt, bh=Math.max(1,rh-(rh>=5?1:0));
  tlGeo={X,GX,top,rh,n,W,span};
  /* the worst stall's step, as a band across every row, once the playhead has passed it */
  const w=o.worst, wst=w?o.st[w.s]:null, showW=w&&wst.t1<=cut+EPS&&w.g>G.chat.gap+EPS;
  if(showW){g.fillStyle=ink; g.globalAlpha=0.08; g.fillRect(X(wst.t0),0,Math.max(2,X(wst.t1)-X(wst.t0)),H); g.globalAlpha=0.55; g.setLineDash([3,3]); g.strokeStyle=ink; g.lineWidth=1;
    g.beginPath(); g.moveTo(X(wst.t0)+.5,0); g.lineTo(X(wst.t0)+.5,H); g.moveTo(X(wst.t1)-.5,0); g.lineTo(X(wst.t1)-.5,H); g.stroke(); g.setLineDash([]); g.globalAlpha=1;}
  const rel=relOf();
  o.R.forEach((r,i)=>{const y=top+i*rh, c=col[r.kind]||ink3;
    /* its goal: met, missed, or not known yet */
    const st=cut===Infinity?(r.ok?'ok':'miss'):(r.arr>cut?null:goalAt(r,cut));
    if(st){g.fillStyle=st==='ok'?okC:(st==='miss'?badC:line); g.fillRect(0,y,5,bh);}
    g.fillStyle=c; g.globalAlpha=0.45; g.fillRect(X(r.arr)-0.5,y,1.5,bh); g.globalAlpha=1;
    if(r.arr>cut) return;
    const wEnd=Math.min(r.admit,cut); if(wEnd>r.arr){g.fillStyle=hatch(g,c); g.fillRect(X(r.arr),y,Math.max(1,X(wEnd)-X(r.arr)),bh);}
    if(r.admit<=cut){const aEnd=Math.min(r.first,cut); g.fillStyle=c; g.globalAlpha=0.35; g.fillRect(X(r.admit),y+bh*0.4,Math.max(0.5,X(aEnd)-X(r.admit)),Math.max(1,bh*0.2)); g.globalAlpha=1;
      r.chunks.forEach(([si])=>{const s=o.st[si]; if(s.t0>cut) return; const e=Math.min(s.t1,cut); g.fillRect(X(s.t0),y,Math.max(1,X(e)-X(s.t0)),bh);});}
    if(r.first<=cut){const dh=Math.max(1,Math.round(bh*0.22)), dy=y+(bh-dh)/2, gg=G.chat.gap, tk=r.toks;
      const solid=(a,b)=>{if(b>a){g.globalAlpha=0.8; g.fillStyle=c; g.fillRect(X(a),dy,Math.max(0.6,X(b)-X(a)),dh); g.globalAlpha=1;}};
      const hollow=(a,b)=>{g.strokeStyle=c; g.lineWidth=1; g.setLineDash([2,1.5]); g.strokeRect(X(a)+0.5,dy-0.5,Math.max(1,X(b)-X(a)-1),dh+1); g.setLineDash([]);};
      let s0=tk[0], last=tk[0];
      for(let j=1;j<tk.length&&tk[j]<=cut+EPS;j++){if(tk[j]-tk[j-1]>gg+EPS){solid(s0,tk[j-1]); hollow(tk[j-1],tk[j]); s0=tk[j];} last=tk[j];}
      solid(s0,last); if(r.fin>cut&&cut!==Infinity){if(cut-last>gg+EPS) hollow(last,cut); else solid(last,cut);}
      g.fillStyle=ink; g.fillRect(X(r.first)-1,y-1,2.2,bh+2);}
    if(sel===r.id||rel.has(r.id)||focusRow===i){g.strokeStyle=ink; g.lineWidth=sel===r.id?1.6:1; if(sel!==r.id) g.setLineDash([2,2]);
      g.strokeRect(X(r.arr)-2,y-1.5,Math.max(4,X(Math.min(r.fin,cut===Infinity?r.fin:Math.max(cut,r.arr)))-X(r.arr)+4),bh+3); g.setLineDash([]);
      if(sel===r.id){g.fillStyle=ink; g.globalAlpha=0.9; r.toks.forEach(t=>{if(t<=cut) g.fillRect(X(t)-0.4,y+bh+1.5,0.9,2);}); g.globalAlpha=1;}}});
  if(play.t!=null){const x=X(Math.min(play.t,span)); g.fillStyle=ink; g.fillRect(x-1,0,2,H); g.beginPath(); g.moveTo(x-5,0); g.lineTo(x+5,0); g.lineTo(x,6); g.closePath(); g.fill();}
  /* the stall's label: a button on the stalled row, linked to the strip */
  const lb=$('#sc-stall',app);
  if(showW){const y=top+w.id*rh; lb.hidden=false; lb.textContent=tx(`longest gap ${fS(w.g)}`,`最长间隔 ${fS(w.g)}`);
    lb.dataset.tipx=revealed()?tx(`The longest gap between two tokens of a short chat (${rName(o.R[w.id])}): step ${fint(w.s+1)} took ${fS(w.g)}. Click to put the playhead there and see that step in the strip.`,`短对话最长的 token 间隔（${rName(o.R[w.id])}）：第 ${fint(w.s+1)} 步用了 ${fS(w.g)}。点击把播放头放到那里，在上方看这一步。`)
      :tx(`The longest gap between two tokens of a short chat (${rName(o.R[w.id])}): ${fS(w.g)}. Click to put the playhead there.`,`短对话最长的 token 间隔（${rName(o.R[w.id])}）：${fS(w.g)}。点击把播放头放到那里。`);
    const lw=lb.offsetWidth||120, xr=X(wst.t1)+6; lb.style.left=(xr+lw<=W?xr:Math.max(0,X(wst.t0)-lw-6)).toFixed(0)+'px'; lb.style.top=Math.max(0,Math.min(H-18,y-7)).toFixed(0)+'px';}
  else lb.hidden=true;
  /* the document's prefill, the stage that blocks others, labelled beside it once the prediction is made (R5.8) */
  if(pl){const pd=revealed()&&dq&&o.st[dq.chunks[0][0]].t0<=cut?dq:null;
    if(pd&&lab){pl.hidden=false; pl.classList.toggle('sc-pfband',!lab.corner);
      if(lab.corner){pl.style.left=(lab.xa-4-lab.lw).toFixed(0)+'px'; pl.style.top=(top+(pd.id+1)*rh+2).toFixed(0)+'px';}
      else {const xs=X(lab.s0), xe=X(Math.min(lab.s1,cut)), xl=Math.round(xs+Math.min(6,Math.max(1,(xe-xs)/2)))+.5, lx=Math.max(0,Math.min(W-lab.lw,xl-12)), ly=Math.round((band-lab.lh)/2);
        pl.style.left=lx.toFixed(0)+'px'; pl.style.top=ly+'px';
        g.strokeStyle=ink; g.globalAlpha=0.6; g.lineWidth=1; g.setLineDash([2,2]); g.beginPath(); g.moveTo(xl,ly+lab.lh); g.lineTo(xl,top+pd.id*rh); g.stroke(); g.setLineDash([]); g.globalAlpha=1;}}
    else pl.hidden=true;}
  const ax=$('#sc-axis',app), ak=span+'|'+W; if(ax._k!==ak){ax._k=ak; let stp=span>12?2:1; while((X(stp)-X(0))<30&&stp<span) stp*=2; const v=[]; /* ticks at least 30 px apart (phones) */ for(let s=0;s<=span+1e-9;s+=stp) v.push(s);
    ax.innerHTML=v.map((s,i)=>`<span style="left:${X(s).toFixed(1)}px"${i===v.length-1?' class="sc-ax-r"':''}>${s}${i===v.length-1?NB+'s':''}</span>`).join('');}
  cv.setAttribute('aria-label',tx(`Timeline of ${n} requests in arrival order, 0 to ${fixd(span,1)} s: waiting hatched, prefill solid, first token a thick tick, decoding a thin bar. Use the arrow keys to move between requests and Enter to trace one.`,`${n} 个请求按到达顺序排列的时间线，0 到 ${fixd(span,1)} s：斜纹是排队，实心是 prefill，粗竖线是首 token，细条是 decode。用方向键在请求之间移动，按回车查看它的记录。`));
}
function tlLegend(){const it=(cls,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i><span>${esc(t)}</span></span>`;
  return it('sc-sw-wait',tx('hatched: waiting','斜纹：排队'),tx('From arrival to admission: the request waits in the queue. Its trace says what for.','从到达到被接纳：请求在队列里等。它的记录里写着在等什么。'))+
   it('sc-sw-pre',tx('thick block: prefill','粗块：prefill'),tx('Each thick block is a step that computes part or all of the request’s prompt. While a long prompt is being prefilled, requests that arrive have to wait for it.','每个粗块是算这个请求 prompt 的一步（一部分或全部）。长 prompt 在 prefill 的时候，新到的请求要等它。'))+
   it('sc-sw-first',tx('thick tick: first token','粗竖线：首 token'),tx('The distance from the request’s arrival to this tick is its TTFT, queueing included.','从请求到达，到这条竖线的距离，就是 TTFT（含排队）。'))+
   it('sc-sw-dec',tx('thin line: decode, one token per step','细线：decode，每步一个 token'),tx('Once its prompt is prefilled, a request takes one token per step until its answer is done, so it blocks nobody.','prompt 算完以后，请求每一步只占一个 token，直到回答写完，所以它不会挡住别人。'))+
   it('sc-sw-hol',tx(`dashed box: a gap over ${fS(G.chat.gap)}`,`虚线框：超过 ${fS(G.chat.gap)} 的间隔`),tx(`A wait between two tokens longer than the chats’ gap goal, ${fS(G.chat.gap)}: the request got no token in that time.`,`两个 token 之间的等待超过了对话的间隔目标 ${fS(G.chat.gap)}：这段时间里这个请求一个 token 也没拿到。`))+
   it('sc-sw-band',tx('shaded band: the longest gap’s step','阴影带：最长间隔的那一步'),tx('The step that caused the longest gap between two tokens of a short chat: every running request waited for it.','造成短对话最长 token 间隔的那一步：每个在跑的请求都在等它。'))+
   it('sc-sw-goal',tx('left edge: green met, red missed','左边：绿色达标，红色没达标'),tx('The small bar at the left of each row shows whether that request meets its goal: green once it is met, red as soon as it is missed (a first token too late, or a gap too long), and grey while it is not known yet.','每一行左边的小条表示这个请求是否达标：达标时是绿色，一旦错过目标（首 token 太晚或间隔太长）就变成红色，还看不出来时是灰色。'))+
   (play.t!=null?it('sc-sw-head',tx('line with a triangle: now','带三角的竖线：当前时刻'),tx('The playhead: the views show the service at this moment.','播放头：各个视图显示服务在这一刻的状态。')):'');}
/* metrics: initial setup | yours, the same requests (R3.8: whole-run values only once the run has been shown to its end) */
function metricRows(o){const c=o.m.chat, d=o.m.doc, n=c.n, rows=[
  {g:tx('The service','整个服务')},
  {k:'ok',l:tx(`goal met, of ${o.m.all.n}`,`达标（共 ${o.m.all.n} 个）`),v:x=>[`${x.m.all.ok} (${pct0(x.m.all.ok_frac)})`,x.m.all.ok_frac*100,0,true],tip:tx(`Requests that meet their own goal, and their share; the service needs ${pct0(E.TARGET)}.`,`达到自己目标的请求数和占比；服务需要 ${pct0(E.TARGET)}。`)},
  {g:tx(`Short chats (${n})`,`短对话（${n} 个）`)},
  {k:'p50',l:tx('first token, p50 (queueing incl.)','首 token 的 p50（含排队）'),v:x=>[fS(x.m.chat.ttft_p50),sv(x.m.chat.ttft_p50)],tip:tx(`The median time to first token of the ${n} short chats, queueing included.`,`${n} 个短对话首 token 延迟的中位数，含排队。`)},
  {k:'p95',l:tx(`first token, p95 of ${n} (queueing incl.)`,`首 token 的 p95（${n} 个，含排队）`),v:x=>[fS(x.m.chat.ttft_p95),sv(x.m.chat.ttft_p95)],tip:TIP.p95(n)+tx(' Queueing included.','含排队。')},
  {k:'gap',l:tx('longest gap between tokens','最长的 token 间隔'),v:x=>[fS(x.m.chat.gap_max),sv(x.m.chat.gap_max)],tip:TIP.gap()}];
  if(d){rows.push({g:d.n>1?tx(`Long documents (${d.n})`,`长文档（${d.n} 份）`):tx('Long document','长文档')});
    rows.push({k:'doc',l:d.n>1?tx('first token, the slower one (queueing incl.)','首 token（较慢的那份，含排队）'):tx('first token (queueing incl.)','首 token（含排队）'),v:x=>[fS(x.m.doc.ttft_max),sv(x.m.doc.ttft_max)],tip:tx(`From arrival to the first token, queueing included; the goal is ${fS(G.doc.ttft)}.`,`从到达到首 token，含排队；目标是 ${fS(G.doc.ttft)}。`)});}
  rows.push({g:tx('The GPU','GPU')},
   {k:'thr',l:tx('output tokens per second','每秒输出 token'),v:x=>[fint(x.m.all.thr),Math.round(x.m.all.thr)],tip:tx('All answer tokens divided by the time from the first arrival to the last finish. It depends on when the requests arrive as much as on the GPU; the work a setting saves shows in the GPU time used.','所有回答 token 数除以从第一个请求到达到最后一个完成的时间。它既取决于 GPU，也取决于请求什么时候到达；一个设置省下多少工作，看“GPU 用时”。')},
   {k:'gput',l:tx('GPU time used','GPU 用时'),v:x=>[fS(x.gpuT),sv(x.gpuT)],tip:tx('The step times added up: how long the GPU computed for these requests. Less means less work for the same requests.','所有步的耗时之和：GPU 为这些请求算了多久。越少，同样的请求花的工作越少。')},
   {k:'peak',l:tx('peak KV cache','KV cache 峰值'),v:x=>[fG(x.m.all.peak_gib),Math.round(x.m.all.peak_gib*10)/10],tip:tx('The most KV cache held during any step, before the requests that finish in it release their blocks; cached idle prompts included.','任何一步进行中占用 KV cache 的最大值（这一步完成的请求还没释放 block 时），含缓存着的空闲 prompt。')},
   {k:'pre',l:tx('prefill tokens computed','prefill 算过的 token'),v:x=>[fint(x.m.all.prefill_tok),x.m.all.prefill_tok],tip:tx('Prompt tokens the GPU actually computed; a cache hit skips its tokens.','GPU 实际算过的 prompt token；缓存命中的部分不用算。')},
   {k:'hit',l:tx('cache hits, tokens','缓存命中的 token'),v:x=>[fint(x.m.all.hit_tok),x.m.all.hit_tok],tip:tx('Prompt tokens found in the prefix cache instead of computed.','在前缀缓存里找到、不用再算的 prompt token。')});
  return rows;}
/* the reference column: in a guided part, the part's own initial setup, so ▲▼ appear only after the learner's own change (review r1,
   S2); in Scenarios and the challenge, the initial setup */
function refRun(o){if(mode==='guided'){const id=STEPS[step].id, m=id==='code'&&codeM?momentOf(codeM):null, c=PART_CFG[m?m.part:id];   // part 7: the part its moment comes from
  return c.load===o.load?getRun(c.load,{...c,policy:'continuous'}):baseOf(o.load);} return baseOf(o.load);}
/* part 7 compares each moment with the initial setup of the part it comes from */
const refPart=()=>{if(mode!=='guided') return null; const id=STEPS[step].id; return id==='code'?(codeM?momentOf(codeM).part:null):id;};
const refName=()=>{if(mode!=='guided') return tx('Initial setup','初始设置'); const rp=refPart(); if(rp&&rp!==STEPS[step].id){const k=STEPS[SIX(rp)].lever, v=k?PART_CFG[rp][k]:null, nm=k==='cb'?tx(`cache budget ${LEVER.cb.title(v)}`,`缓存预算 ${LEVER.cb.title(v)}`):(k?LEVER[k].title(v):''); return nm?tx(`Initial setup, ${nm.charAt(0).toLowerCase()+nm.slice(1)}`,`${nm}${/[\x21-\x7e]$/.test(nm)?' ':''}的初始设置`):tx('Initial setup','初始设置');} return tx('This part’s initial setup','这一部分的初始设置');};
function drawMetrics(o){
  const b=refRun(o), showY=play.t==null&&(mode!=='challenge'||!!chRunKey), showB=seen.has(b.key)||o.key!==b.key, host=$('#sc-mt',app);
  $('#sc-mtcap',app).textContent=tx(`The whole run (all ${o.R.length} requests), the same requests for both`,`整次运行（全部 ${o.R.length} 个请求），两列是同一批请求`);
  const rows=metricRows(o);
  const h=`<table class="sc-tab"><thead><tr><th></th><th data-tipx="${esc(mode==='guided'?tx(`${refName()}: ${cfgDesc(b.s)}.`,`${refName()}：${cfgDesc(b.s)}。`):tx(`The service as it runs today: ${cfgDesc(BASE)}.`,`今天的服务：${cfgDesc(BASE)}。`))}" tabindex="0">${refName()}</th><th>${tx('Yours','你的设置')}</th><th>${tx('vs initial setup','比初始设置')}</th></tr></thead><tbody>`+
    rows.map(r=>{if(r.g) return `<tr class="sc-grp"><th colspan="4">${esc(r.g)}</th></tr>`; const [bt,bv,dp,pts]=r.v(b), [yt,yv]=r.v(o), dd=showY&&showB?dFmt(bv,yv,{dp:dp||(r.k==='peak'?1:(r.k==='ok'?0:null)),pts}):'';
      const cls=r.k==='ok'?(v=>v>=E.TARGET*100-1e-6?' sc-okc':' sc-badc'):()=>'';
      return `<tr data-k="${r.k}"><th tabindex="0" data-tipx="${esc(r.tip)}">${esc(r.l)}</th><td class="mono${showB?cls(bv):''}">${showB?esc(bt):'…'}</td><td class="mono${showY?cls(yv):''}">${showY?esc(yt):(play.t!=null?tx('after the run','跑完后显示'):'—')}</td><td class="mono sc-dd">${esc(dd)}</td></tr>`;}).join('')+`</tbody></table>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  /* phones: two lines (goal met initial → yours; the one number this part is about) */
  const key=mode==='guided'?({cache:'p50',cbudget:'hit',cap:'p95'}[refPart()||STEPS[step].id]||'gap'):'gap', kr=rows.find(r=>r.k===key), ok=rows.find(r=>r.k==='ok');
  const nC=o.m.chat.n, lab=r=>({ok:tx('Goal met','达标'),p50:tx('Chats’ first token p50, queueing incl.','短对话首 token p50（含排队）'),p95:tx(`Chats’ first token p95 of ${nC}, queueing incl.`,`短对话首 token p95（${nC} 个，含排队）`),gap:tx('Chats’ longest gap','短对话最长间隔'),hit:tx('Cache hits, tokens','缓存命中 token')})[r.k]||r.l;
  const line=(r)=>`<div class="sc-m2l"><span>${esc(lab(r))}</span><b class="mono">${showB?esc(r.v(b)[0]):'…'} → ${showY?esc(r.v(o)[0]):'…'}</b></div>`;
  const rn=refName();
  const mh=line(ok)+line(kr)+`<div class="sc-m2n small muted">${mode==='guided'?tx(`${rn.charAt(0).toLowerCase()+rn.slice(1)} → yours`,`${rn} → 你的设置`):tx('initial setup → yours','初始设置 → 你的设置')}</div>`;
  ['#sc-m2','#sc-m2d'].forEach(q=>{const m2=$(q,app); if(m2&&m2._h!==mh){m2.innerHTML=mh; m2._h=mh;}});
}
/* the trace of one request, in plain words, up to the moment shown */
let relCache={k:'',v:new Set()};
function relOf(){const o=cur(); if(sel==null||!o.R[sel]) return new Set(); const key=o.key+'|'+sel; if(relCache.k===key) return relCache.v;
  const r=o.R[sel], v=new Set(); if(r.gi){const inf=stepInfo(o,r.dsteps[r.gi-1]); if(inf.big&&inf.big.id!==r.id&&(r.gap>G.chat.gap)) v.add(inf.big.id);}
  if((r.waits.budget||0)>0) aheadOf(o,r).slice(0,1).forEach(x=>{if(x.id!==r.id) v.add(x.id);});
  relCache={k:key,v}; return v;}
const traceHint=()=>`<p class="small muted sc-trhint"><b>${tx('Trace a request:','查看一个请求的记录：')}</b>${tx(' pick one anywhere, such as a chip in the queue, a slice in the GPU step, a row of the timeline or a block of memory.','在任何地方点它，比如队列里的方块、GPU 步里的一段、时间线上的一行，或显存里的一块。')}</p>`;
function traceHTML(o){
  if(sel==null||!o.R[sel]) return `<p class="small muted sc-trhint"><b>${tx('Trace a request:','查看一个请求的记录：')}</b>${tx(' pick one anywhere, such as a chip in the queue, a slice in the GPU step, a row of the timeline or a block of memory.','在任何地方点它，比如队列里的方块、GPU 步里的一段、时间线上的一行，或显存里的一块。')}</p>`;
  const r=o.R[sel], tt=play.t==null?Infinity:play.t, g=G[r.cls], it=[], kname=kindL(r.kind,o.R.some(x=>x.app));
  const head=`<div class="sc-trh"><i class="sc-sw" style="background:var(--sc-${r.kind})"></i><b>${esc(rName(r))}</b><span class="small muted">${esc(tx(`${kname}: ${aN(r.P)} ${fint(r.P)}-token prompt${r.pre?` (${fint(r.pre)} of them are ${preOf(r.app,true)})`:''}, ${aN(r.out)} ${fint(r.out)}-token answer`,`${kname}：prompt ${fint(r.P)} 个 token${r.pre?`（其中 ${fint(r.pre)} 个是${preOf(r.app,true)}）`:''}，回答 ${fint(r.out)} 个 token`))}</span><button type="button" class="linkbtn sc-unsel" data-act="unsel" aria-label="${esc(tx('Close the trace','关闭记录'))}">✕</button></div>`;
  it.push([r.arr,tx('arrived','到达')]);
  if(r.arr<=tt){const ws=waitSegs(o,r).filter(x=>x[0]<tt).map(x=>[x[0],Math.min(x[1],tt),x[2]]), tot={};
    ws.forEach(([a,b,w])=>{tot[w]=(tot[w]||0)+b-a;});
    const parts=Object.entries(tot).sort((x,y)=>y[1]-x[1]).map(([w,v])=>`${fS(v)} ${whyLong(w,o)}`);
    if(parts.length){const sum=Object.values(tot).reduce((a,b)=>a+b,0), ah=(tot.budget||0)>0?aheadOf(o,r)[0]:null;
      if(ah&&revealed()){const an=o.R[ah.id], rest=Object.entries(tot).filter(([w])=>w!=='budget').map(([w,v])=>`${fS(v)} ${whyLong(w,o)}`);   /* finding 4: say who took the budget and why */
        it.push([null,tx(`waited ${fS(sum)} before it could start. For ${fS(tot.budget)}, the prefill of ${rName(an)}, which arrived earlier, took each step’s whole token budget (first come, first served), so this request could start only when that prefill ended.`,
                         `开始前等了 ${fS(sum)}。其中 ${fS(tot.budget)} 里，先到的${rName(an)}在 prefill，占满了每一步的 token 预算（先来先服务），所以这个请求要等那段 prefill 结束才能开始。`)+(rest.length?tx(` It also waited ${rest.join('; ')}.`,`另外，它还花了${rest.join('；')}。`):''),'w']);}
      else it.push([null,tx(`waited ${fS(sum)}: `,`等了 ${fS(sum)}：`)+parts.join(tx('; ','；')),'w']);}}
  if(r.admit<=tt) it.push([r.admit,r.hit?tx(`admitted: ${fint(r.hit)} prompt tokens found in the cache, prefill starts after them`,`被接纳：${fint(r.hit)} 个 prompt token 在缓存里找到，从它们后面开始 prefill`):tx(`admitted: prefill starts (${r.chunks.length>1?`${fint(r.chunks.length)} steps`:'one step'})`,`被接纳：开始 prefill（${r.chunks.length>1?`${fint(r.chunks.length)} 步`:'一步'}）`)]);
  if(r.first<=tt){const ok=r.ttft<=g.ttft+EPS; it.push([r.first,tx(`first token: ${fS(r.ttft)} after arriving (queueing included), `,`首 token：到达后 ${fS(r.ttft)}（含排队），`)+`<b class="${ok?'sc-okc':'sc-badc'}">${ok?tx(`within the ${fS(g.ttft)} goal`,`在 ${fS(g.ttft)} 的目标以内`):tx(`over the ${fS(g.ttft)} goal`,`超过了 ${fS(g.ttft)} 的目标`)}</b>${tx('.','。')}`]);}
  if(r.gi&&r.toks[r.gi]<=tt){const inf=stepInfo(o,r.dsteps[r.gi-1]), big=inf.big, gg=gapGoal(r.cls), ok=r.gap<=gg+EPS;
    const sd=inf.s.t1-inf.s.t0, bud=o.cfg.budget||0, T=inf.pre.reduce((a,p)=>a+p.n,0), bn=big?rName(o.R[big.id]):'';
    const piece=big?tx(big.whole?`the whole prompt of ${bn} (${fint(big.n)} tokens)`:`a chunk of ${bn} (${fint(big.n)} tokens)`,big.whole?`${bn}的整个 prompt（${fint(big.n)} 个 token）`:`${bn}的一块 chunk（${fint(big.n)} 个 token）`):'';
    /* what the request waited for during its longest gap, as a full sentence (review B4, R3) */
    const what=!(big&&big.id!==r.id)?tx('During that time it waited for a usual step to finish.','这段时间它在等一个平常的步算完。')
      :!revealed()?(sd>G.chat.gap+EPS?tx(`During that time it waited for a long step (${fS(sd)}) to finish.`,`这段时间它在等一个很长的步（${fS(sd)}）算完。`):tx('During that time it waited for a usual step to finish.','这段时间它在等一个平常的步算完。'))   /* before the prediction: the effect, not the cause (R3.8) */
      :inf.pre.length>1?(bud&&inf.s.T>=bud?tx(`During that time it waited for a step to finish; that step used the whole ${fint(bud)}-token budget, and its largest piece was ${piece}.`,`这段时间它在等一步算完，那一步用满了 ${fint(bud)} 个 token 的预算，其中最大的一块是${piece}。`)
                                           :tx(`During that time it waited for a step to finish; that step held ${inf.pre.length} prompts, ${fint(T)} tokens in all, and its largest piece was ${piece}.`,`这段时间它在等一步算完，那一步里有 ${inf.pre.length} 段 prompt，一共 ${fint(T)} 个 token，其中最大的一块是${piece}。`))
      :tx(`During that time it waited for a step that held ${piece}.`,`这段时间它在等一步算完，那一步里是${piece}。`);
    it.push([r.toks[r.gi],tx(`longest gap: ${fS(r.gap)}`,`最长的间隔是 ${fS(r.gap)}`)+(isFinite(gg)?`${tx(', ','，')}<b class="${ok?'sc-okc':'sc-badc'}">${ok?tx(`within ${fS(gg)}`,`没有超过 ${fS(gg)}`):tx(`over ${fS(gg)}`,`超过了 ${fS(gg)}`)}</b>`:'')+tx('. ','。')+what]);}
  if(r.fin<=tt){const gg=gapGoal(r.cls), late=r.ttft>g.ttft+EPS, gapx=isFinite(gg)&&r.gap>gg+EPS;   /* review B4, R2: the answer, then the verdict and its reason */
    const why=late&&gapx?tx('its first token came too late and one gap between its tokens was too long','首 token 来得太晚，而且有一次 token 间隔太长'):late?tx('its first token came too late','首 token 来得太晚'):gapx?tx('one gap between its tokens was too long','有一次 token 间隔太长'):'';
    it.push([r.fin,tx(`done: it answered ${fint(r.out)} tokens.`,`完成：回答了 ${fint(r.out)} 个 token。`)+(play.t==null?` <b class="${r.ok?'sc-okc':'sc-badc'}">${r.ok?tx('This request met its goal.','这个请求达标了。'):(why?tx(`This request missed its goal because ${why}.`,`这个请求没达标，因为${why}。`):tx('This request missed its goal.','这个请求没达标。'))}</b>`:'')]);}
  else if(play.t!=null) it.push([null,tx('…still to come','……之后的还没发生'),'w']);
  return head+`<ol class="sc-tr">${it.map(([t,h,c])=>`<li${c?' class="sc-tr-'+c+'"':''}><span class="sc-trt mono">${t==null?'':fC(t)}</span><span>${h}</span></li>`).join('')}</ol>`;}
/* Show the math (R6.6: every line redoes by hand) */
/* H1/H2 (the owner, 2026-10-09): Show the math as short blocks, like a page of a good textbook. Each block has one lead line in words,
   then a small table: the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the
   result, each on its own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>Math.max(g.terms.length,...g.rows.map(r=>r.c?r.c.length:1)))), eq=`<td class="sc-meq">${MX('=')}</td>`;
  const cells=(a,f)=>a.map((x,i)=>`<td${i===a.length-1&&a.length<n?` colspan="${n-i}"`:''}>${f(x)}</td>`).join('');
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?cells(r.c,x=>x):`<td colspan="${n}" class="sc-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="sc-mname"><th scope="row">${g.name}</th>${eq}${cells(g.terms,MX)}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="sc-mb">${lead?`<p class="sc-mlead">${lead}</p>`:''}<div class="sc-mfx"><table class="sc-mt">${body}</table></div>${note?`<p class="sc-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.sc-mt'); if(!tb) return; const e0=tb.querySelector('td.sc-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('sc-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('sc-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="sc-mc${r?'':' sc-mc0'}">${x}</span>`;
    h+=`<tr class="sc-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="sc-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="sc-mband${g.tall?' sc-mtall':''}"><td class="sc-meq"><span class="sc-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="sc-mfl">${units.map(u=>`<span class="sc-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="sc-meq">${E}</td><td class="sc-mres">${r.l?`<span class="sc-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='sc-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.sc-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.sc-mfx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
function mathHTML(o){
  const t=T_, rows=[], w=o.worst, r=sel!=null&&o.R[sel]?o.R[sel]:(w?o.R[w.id]:o.R[0]);
  const fsN=fixd(C.FS/1e12,2)+NB+'TFLOP/s', bsN=fixd(C.BS/1e12,2)+NB+'TB/s';
  const FXT=[tx('\\max(\\text{compute time},\\ \\text{memory time})','\\max(\\text{计算时间},\\ \\text{读写显存时间})')];
  rows.push(mathBlock(tx('Every step’s time comes from one model. The GPU computes and moves memory at the same time, so a step takes the longer of the two.','每一步的耗时都来自同一个模型。GPU 计算和读写显存同时进行，所以一步取两者中较长的那个。'),
    [{name:tx('One step’s time','一步的耗时'),terms:[tx('\\max\\left(\\frac{\\text{FLOP}}{\\text{FLOP/s reached}},\\right.','\\max\\left(\\frac{\\text{FLOP}}{\\text{能达到的 FLOP/s}},\\right.'),tx('\\left.\\frac{\\text{bytes}}{\\text{bandwidth reached}}\\right)','\\left.\\frac{\\text{字节}}{\\text{能达到的带宽}}\\right)')],
      rows:[{c:[`max(FLOP ÷ ${t('fs',fsN)},`,`${tx('bytes','字节')} ÷ ${t('bs',bsN)})`]}]},
     {name:'FLOP',terms:[tx('2 \\times \\text{layer weights} \\times \\text{tokens}','2 \\times \\text{各层权重} \\times \\text{token 数}'),'+',tx('2 \\times \\text{output-layer weights} \\times \\text{new tokens}','2 \\times \\text{输出层权重} \\times \\text{新 token 数}'),'+','\\text{attention}'],
      rows:[{c:[`2 × ${t('pl',fint(C.P_LAYERS))} × ${tx('tokens','token 数')}`,'+',`2 × ${t('ph',fint(C.P_HEAD))} × ${tx('new tokens','新 token 数')}`,'+',t('att',tx('attention','attention'))]}]},
     {name:tx('bytes','字节'),terms:[tx('\\text{weights}','\\text{权重}'),'+',tx('\\text{KV cache per token} \\times \\text{KV tokens read}','\\text{每个 token 的 KV cache} \\times \\text{读到的 KV token 数}')],
      rows:[{c:[t('wr',fint(C.W_READ)),'+',`${t('kvt',fint(C.KV_TOK))} × ${tx('KV tokens read','读到的 KV token 数')}`]}]},
     {name:'attention',terms:[`2 \\times ${C.L} \\times ${fint(C.D)} \\times (b(b + 1) - a(a + 1))`],rows:[]}],
    tx(`The attention term counts the new tokens at positions ${MX('a')} to ${MX('b - 1')}, summed over the step. A decode token at position ${MX('c')} has ${MX('a = c')} and ${MX('b = c + 1')}.`,`attention 这一项算的是位置 ${MX('a')} 到 ${MX('b - 1')} 的新 token，整步求和。位置 ${MX('c')} 的 decode token，${MX('a = c')}，${MX('b = c + 1')}。`)));
  const ws=w&&(sel==null||!o.R[sel])?w.s:(r.gi?r.dsteps[r.gi-1]:null);
  if(ws!=null){const inf=stepInfo(o,ws), s=inf.s, nPre=inf.pre.reduce((a,p)=>a+p.n,0);
    rows.push(mathBlock(`<b>${sel!=null&&o.R[sel]?tx(`The step behind ${rName(r)}’s longest gap: step ${fint(ws+1)}`,`${rName(r)} 最长间隔背后的那一步：第 ${fint(ws+1)} 步`):tx(`The step behind the longest gap: step ${fint(ws+1)}`,`最长间隔背后的那一步：第 ${fint(ws+1)} 步`)}</b> (${fC(s.t0)} → ${fC(s.t1)})${tx('. ','。')}`+
        tx(`It has ${fint(inf.dec.length)} decode token${inf.dec.length===1?'':'s'} and ${fint(nPre)} prefill tokens, ${fint(s.T)} in all, of which ${fint(inf.logits)} ${inf.logits===1?'is':'are'} new; its attention is ${fint(inf.att)} FLOPs, and it reads the KV cache of ${fint(inf.kv)} tokens.`,
           `这一步有 ${fint(inf.dec.length)} 个 decode token 和 ${fint(nPre)} 个 prefill token，一共 ${fint(s.T)} 个，其中新 token ${fint(inf.logits)} 个。attention 是 ${fint(inf.att)} FLOP，要读 ${fint(inf.kv)} 个 token 的 KV cache。`),
      [{name:tx('Compute time','计算时间'),terms:[tx('\\text{FLOP}','\\text{运算量}'),'\\div',tx('\\text{FLOP/s reached}','\\text{能达到的 FLOP/s}')],
        rows:[{r:`(2 × ${fint(C.P_LAYERS)} × ${fint(s.T)} + 2 × ${fint(C.P_HEAD)} × ${fint(inf.logits)} + ${fint(inf.att)}) ÷ ${fsN}`},{c:[`${fint(s.fl)} FLOP`,'÷',fsN]},{r:fS(s.tc)}]},
       {name:tx('Memory time','读写显存时间'),terms:[tx('\\text{bytes}','\\text{字节}'),'\\div',tx('\\text{bandwidth reached}','\\text{能达到的带宽}')],
        rows:[{r:`(${fint(C.W_READ)} + ${fint(C.KV_TOK)} × ${fint(inf.kv)}) ÷ ${bsN}`},{c:[`${fint(s.by)} ${tx('bytes','字节')}`,'÷',bsN]},{r:fS(s.tm)}]},
       {name:tx('Step time','这一步的耗时'),terms:FXT,rows:[{c:[`max(${fS(s.tc)}, ${fS(s.tm)})`]},{r:`<b>${fS(s.t1-s.t0)}</b>${s.tc>=s.tm?tx(', set by arithmetic','，由计算决定'):tx(', set by memory traffic','，由读写显存决定')}`}]}]));}
  if(r){const ws2=waitSegs(o,r), tot={}; ws2.forEach(([a,b,x])=>{tot[x]=(tot[x]||0)+b-a;});
    /* every term in ms with one decimal, and every sum made of the terms as shown (R6.6) */
    const m1=v=>Math.round(v*1e4)/10, fm=v=>fixd(v,1), add=xs=>Math.round(xs.reduce((a,v)=>a+v,0)*10)/10;
    const pre=o.st.filter(s=>s.t0>=r.admit-EPS&&s.t1<=r.first+EPS), wv=Object.entries(tot).map(([x,v])=>[x,m1(v)]), wsum=add(wv.map(x=>x[1]));
    const pv=pre.length<=6?pre.map(s=>m1(s.t1-s.t0)):null, psum=pv?add(pv):m1(r.first-r.admit), all=add([wsum,psum]);
    const wt=wv.length?wv.map(([x,v])=>`${fm(v)} (${esc(L(WHY[x]))})`).join(' + '):'0';
    const preS=pv?pv.map(fm).join(' + '):tx(`${fint(pre.length)} steps from ${fC(r.admit)} to ${fC(r.first)}`,`从 ${fC(r.admit)} 到 ${fC(r.first)} 的 ${fint(pre.length)} 步`);
    rows.push(mathBlock(tx(`${rName(r)}’s first token comes after its waits in the queue and its prefill steps. All times are in ms.`,`${rName(r)} 的首 token 要等它在队列里的等待，加上它 prefill 的各步。时间都以 ms 计。`),
      [{name:tx(`${rName(r)}’s first token`,`${rName(r)} 的首 token`),terms:[tx('\\text{its waits}','\\text{等待}'),'+',tx('\\text{its prefill steps}','\\text{prefill 的各步}')],
        rows:[{c:[wt,'+',preS]},{c:[fm(wsum),'+',fm(psum)]},{r:`<b>${fm(all)}${NB}ms</b>${tx(', queueing included','，含排队')}`}]}]));}
  const c=o.m.chat, rank=Math.ceil(0.95*c.n-1e-9);
  rows.push(mathBlock(tx('How many requests meet their goal, and how long the slowest 5% of the short chats wait for their first token.','有多少请求达标，以及最慢的那 5% 短对话要等多久才拿到首 token。'),
    [{name:tx('Goal met','达标'),terms:[tx('\\frac{\\text{requests that meet their goal}}{\\text{requests}}','\\frac{\\text{达标的请求数}}{\\text{请求数}}')],rows:[{c:[`${o.m.all.ok} ÷ ${o.m.all.n}`]},{r:`<b>${fixd(o.m.all.ok_frac*100,1)}%</b>`}]},
     {name:tx('p95 of the short chats’ first token','短对话首 token 的 p95'),terms:[tx('\\text{value at rank } ⌈0.95 \\times n⌉ \\text{ of the sorted list}','\\text{排序后第 } ⌈0.95 \\times n⌉ \\text{ 个值}')],
      rows:[{l:`n = ${c.n}`,c:[tx(`the value at rank ⌈0.95 × ${c.n}⌉ = ${rank}`,`排序后第 ⌈0.95 × ${c.n}⌉ = ${rank} 个值`)]},{r:`<b>${fS(c.ttft_p95)}</b>${tx(', queueing included','，含排队')}`}]}],
    tx('⌈ ⌉ rounds up.','⌈ ⌉ 表示向上取整。')));
  const ref=refRun(o), toks=o.R.reduce((a,r)=>a+r.out,0), span=o.end-o.t0r;
  rows.push(mathBlock(tx('The work the GPU did, and the tokens it wrote per second.','GPU 做了多少工作，每秒写出多少 token。'),
    [{name:tx('GPU time used','GPU 用时'),terms:[tx('\\text{the step times added up}','\\text{各步耗时之和}')],rows:[{c:[tx(`the ${fint(o.st.length)} step times of this run added up`,`这次运行 ${fint(o.st.length)} 步的耗时之和`)]},{r:`<b>${fS(o.gpuT)}</b>${tx(` (${(r=>r.charAt(0).toLowerCase()+r.slice(1))(refName())}: ${fS(ref.gpuT)})`,`（${refName()}：${fS(ref.gpuT)}）`)}`}]},
     {name:tx('Output tokens per second','每秒输出 token'),terms:[tx('\\frac{\\text{answer tokens}}{\\text{last finish} - \\text{first arrival}}','\\frac{\\text{回答 token 数}}{\\text{最后一个完成} - \\text{第一个到达}}')],rows:[{c:[`${fint(toks)} ÷ ${fixd(span,2)}${NB}s`]},{r:`<b>${fint(toks/span)}</b>`}]}],
    tx('It depends on when the requests arrive as much as on the GPU, so the work a setting saves shows in the GPU time used.','它既取决于 GPU，也取决于请求什么时候到达，所以一个设置省下多少工作，看 GPU 用时。')));
  const usable=Math.round(80*C.GiB*0.9), mib=fint(BLK_B/1048576);
  rows.push(mathBlock(tx('GPU memory holds the weights and a pool of blocks for the KV cache.','显存里放着权重，以及存 KV cache 的一池 block。'),
    [{name:tx('A block','一个 block'),terms:[tx('\\text{tokens per block} \\times \\text{KV cache per token}','\\text{每个 block 的 token 数} \\times \\text{每个 token 的 KV cache}')],rows:[{c:[`${C.BLOCK} × ${fint(C.KV_TOK)}`]},{r:`${fint(BLK_B)} ${tx('bytes','字节')} (${mib}${NB}MiB)`}]},
     {name:tx('The pool','显存池'),terms:[tx('\\frac{\\text{usable memory} - \\text{weights} - \\text{overhead}}{\\text{KV cache per token}}','\\frac{\\text{可用显存} - \\text{权重} - \\text{开销}}{\\text{每个 token 的 KV cache}}')],rows:[{c:[`(${fint(usable)} − ${t('wm',fint(C.W_MEM))} − ${fint(2*C.GiB)}) ÷ ${fint(C.KV_TOK)}`]},{r:`${fint(C.POOL_T)} ${tx('tokens','个 token')} (${fint(C.POOL_B)} ${tx('blocks','个 block')})`}]},
     {name:tx('This run’s peak','这次运行的峰值'),terms:[tx('\\text{blocks held} \\times \\text{block size}','\\text{占用的 block 数} \\times \\text{block 大小}')],rows:[{c:[`${fint(o.run.peak)} × ${mib}${NB}MiB`]},{r:`<b>${fG(o.m.all.peak_gib)}</b>`}]}],
    tx('The peak is the most held during any step, before the requests that finish in it release their blocks.','峰值是任何一步进行中占用的最大值，这时这一步完成的请求还没释放 block。')));
  rows.push(`<div class="sc-simpl"><b>${tx('Model and simplifications','模型和简化')}</b><ul>${SIMPL().map(x=>`<li>${x}</li>`).join('')}</ul></div>`);
  return rows.join('');}
function drawPlayUI(o){
  const sp=spanFor(o), sc=$('#sc-scrub',app); if(sc){sc.value=String(play.t==null?1000:Math.round(Math.min(1,play.t/sp)*1000)); A.ui.setRange(sc);}
  $$('[data-act="play"]',app).forEach(b=>{b.textContent=playLabel();});
  const tt=play.t, clk=$('#sc-clock',app);
  if(clk){if(tt==null) clk.textContent=tx(`The whole run: all ${o.R.length} requests done at ${fC(o.end)}.`,`整次运行：${o.R.length} 个请求在 ${fC(o.end)} 全部完成。`);
    else {const ev=endView(o); let run=0, wait=0, done=0; o.R.forEach(r=>{if(r.arr>tt+EPS) return; if(r.fin<=tt+EPS) done++; else if(admittedAt(r,tt,ev)) run++; else wait++;});
      clk.textContent=tx(`At ${fC(tt)}: ${run} running, ${wait} waiting, ${done} done.`,`${fC(tt)}：${run} 个在跑，${wait} 个在排队，${done} 个已完成。`);}}
}
/* one frame of the stage at the playhead: the strip, memory, timeline, trace and the play controls */
function frame(){if(!app) return; const o=cur(); drawStrip(o); drawCode(o); drawMem(o); drawTL(); drawPlayUI(o);
  const s_=statusOf(o), st=$('#sc-status',app); if(st&&st.textContent!==s_.txt){st.className='status '+s_.cls; st.textContent=s_.txt; st.dataset.tipx=s_.tip;}
  /* the trace: on a desktop a card beside the selected row, where the learner is looking (A1); on a phone in the flow below */
  const has=sel!=null&&!!o.R[sel], desk=innerWidth>760, tr=$('#sc-trace',app), th=has&&!desk?traceHTML(o):traceHint();
  if(tr._h!==th){tr.innerHTML=th; tr._h=th; const lw=tr.parentNode; if(lw) lw.classList.toggle('has-trace',has&&!desk); tr.classList.toggle('sc-trempty',!(has&&!desk));}
  const tc=$('#sc-tcard',app);
  if(tc){if(has&&desk){const h=traceHTML(o); if(tc._h!==h){tc.innerHTML=h; tc._h=h;} tc.hidden=false; placeCard(o);} else if(!tc.hidden||tc._h){tc.hidden=true; tc._h=''; tc._sel=null;}}
  const lg=$('#sc-tlleg',app), lh=tlLegend(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const dk=(play.t==null?'end':fC(play.t))+'|'+mode+'|'+sel; if(dk!==play.dk){play.dk=dk; A.dock.render();}}
function render(){
  if(!app) return; const o=cur();
  drawHead(o); frame(); drawMetrics(o);
  const ml=$('#sc-mleg',app), mh=memLegend(o); if(ml._h!==mh){ml.innerHTML=mh; ml._h=mh;}
  $('#sc-tlcap',app).textContent=tx(`Every request, in arrival order: from arriving to its last token`,`每个请求按到达顺序排列：从到达到最后一个 token`);
  const sc=$('#sc-scope',app); sc.textContent=tx(`One ${GPUN}: estimates from the step model, queueing included.`,`一张 ${GPUN}：用耗时模型估算，含排队。`);
  sc.dataset.tipx=tx('Every time here comes from one event/time model: the scheduler above and the step model (arithmetic at 50% of the peak FLOP/s, memory traffic at 60% of the peak bandwidth, whichever is longer). The single-request labs leave out queueing; here every first token includes it.','这里的每个时间都来自同一个事件/时间模型：上面的调度器和耗时模型（计算按峰值 FLOP/s 的 50%，读写显存按峰值带宽的 60%，取较长的）。只看单个请求的那些 Lab 不含排队；这里每个首 token 都包含排队。');
  const mb=$('#sc-mathbody',app), mhh=mathHTML(o); if(mb._h!==mhh){mb.innerHTML=mhh; mb._h=mhh;} fitMath(mb);
  const gr=$('#sc-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].focus||''):'';
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce(o);
}
const speedTip=()=>codeOn()?tx(`In this part, 1× shows one engine step every 0.8 s; 0.5× one every 1.6 s, and 2× about one every 0.5 s.`,`这一部分里，1× 每 0.8 s 放一个引擎步；0.5× 每 1.6 s 一步，2× 大约每 0.5 s 一步。`)
  :tx(`1× plays the whole run in about ${BASE_SECS} s, whatever its length; 0.5× takes twice as long, 2× half.`,`1× 用约 ${BASE_SECS} s 放完整次运行，不管它多长；0.5× 慢一倍，2× 快一倍。`);
function syncControls(){
  $$('.sc-speed',app).forEach(g=>{const v=speedTip(); if(g.dataset.tipx!==v) g.dataset.tipx=v;});   // review B, B6
  $$('.seg-ctl[data-key]',app).forEach(g=>{const k=g.dataset.key; if(k==='speed') A.ui.setPressed(g,String(spd())); else if(k in S&&!g.closest('#sc-rail')) A.ui.setPressed(g,k==='cache'?String(!!S.cache):String(S[k]));
    else if(k.startsWith('ch-')) A.ui.setPressed(g,k==='ch-cache'?String(!!chCfg.cache):String(chCfg[k.slice(3)]));});
  const lock=mode!=='sandbox', ln=$('#sc-lock',app), cl=$('#sc-ctls',app), rs=$('#sc-resetw',app);
  ln.hidden=!lock; cl.hidden=lock; rs.hidden=lock;
  if(lock) ln.innerHTML=mode==='guided'?tx(`In the guided parts each part’s card holds the one option it introduces. Every setting is in ${`<button type="button" class="linkbtn" data-mode-go="sandbox">Scenarios</button>`}.`,`分步引导里，每一部分的卡片只放它介绍的那一个选项。所有设置都在“<button type="button" class="linkbtn" data-mode-go="sandbox">应用场景</button>”里。`)
    :tx('In the challenge you choose all five settings in the task card. The model, the GPU and the requests are fixed.','挑战中在任务卡里选五项设置。模型、GPU 和请求都固定不变。');
  const cw=$('#sc-cbwrap',app); if(cw) cw.hidden=!S.cache;
}
let annT=null;
function announce(o){clearTimeout(annT); const my=runTok; annT=setTimeout(()=>{if(!app||my!==runTok) return;
  A.announce(play.t==null?tx(`${loadDesc(o.load)}, ${cfgDesc(o.s)}: ${o.m.all.ok} of ${o.m.all.n} requests meet their goal.`,`${loadDesc(o.load)}，${cfgDesc(o.s)}：${o.m.all.n} 个请求里 ${o.m.all.ok} 个达标。`):tx(`Playing ${loadDesc(o.load)}, ${cfgDesc(o.s)}.`,`正在运行：${loadDesc(o.load)}，${cfgDesc(o.s)}。`));},500);}

/* =====================================================================
   playback: time-based (1× plays a whole run in about 15 s), one token checked in every frame (R3.9)
   ===================================================================== */
function stopPlay(){play.playing=false; if(play.raf) cancelAnimationFrame(play.raf); play.raf=0;}
function revealObs(){if(mode!=='guided'||innerWidth<=760||!app) return; const rail=$('#sc-rail',app), box=rail&&$('[data-obs]',rail); if(!box||box.hidden||!box.innerHTML) return;
  const bar=document.querySelector('.topbar'), top=(bar?bar.offsetHeight:56)+8, r=box.getBoundingClientRect(), want=Math.min(r.height,innerHeight-top-16), d=r.top+want-(innerHeight-12); if(d<=4) return;
  const keep=$('[data-lever][aria-pressed="true"]',rail)||$('.choice[aria-pressed="true"]',rail)||$('.seg-ctl',rail); let lim=r.top-top; if(keep) lim=Math.min(lim,keep.getBoundingClientRect().top-top);
  const by=Math.max(0,Math.min(d,lim)); if(by<=4) return;
  /* an instant scroll, then a check once laid out: results that grew above can add a scroll-anchoring shift; the option stays in view */
  window.scrollBy({top:by,behavior:'auto'});
  requestAnimationFrame(()=>{const k2=keep&&keep.isConnected?keep.getBoundingClientRect().top:null; if(k2!=null&&k2<top) window.scrollBy({top:k2-top,behavior:'auto'});});}
function onEnd(quiet){const o=cur(); play.t=null; stopPlay(); seen.add(o.key); if(mode==='guided'&&STEPS[step].lever) tried[STEPS[step].id]=tried[STEPS[step].id]||o.key!==keyOf(o.load,{...PART_CFG[STEPS[step].id]});
  if(mode==='challenge'&&chRunKey===o.key&&!chTier){const my=runTok; chTierCompute(o.s,t=>{if(my!==runTok) return; chTier=t; if(t.tier!==3) chReveal=true; render();}); }
  render(); if(!quiet) requestAnimationFrame(()=>revealObs());}
function startPlay(fromZero){
  const o=cur();
  if(play.playing&&!fromZero){stopPlay(); frame(); updateRail(); return;}
  if(RM()){onEnd(); return;}
  if(fromZero||play.t==null||play.t>=o.end-1e-9) play.t=0;
  if(codeOn()){codePlay(); return;}
  stopPlay(); play.playing=true; play.last=performance.now(); const my=runTok; frame(); updateRail(); drawMetrics(o);
  const tick=now=>{if(!app||!play.playing||my!==runTok) return; const o2=cur(), dt=Math.min(0.1,(now-play.last)/1000); play.last=now;
    play.t+=dt*spd()*spanFor(o2)/BASE_SECS; if(play.t>=o2.end){onEnd(); return;} frame(); if(Math.floor(play.t*4)!==play.q){play.q=Math.floor(play.t*4); updateRail();} play.raf=requestAnimationFrame(tick);};
  play.raf=requestAnimationFrame(tick);}
/* part 7 plays one engine step at a time, so every line and value can be read (review A, finding 10; R7.11): at 1× 0.48 s computing
   the step (the sweep, the panel's step_time line), then 0.32 s at its end (the release lines), 0.8 s in all; the end stays at least
   0.3 s at every speed (review B, B12). After the last step's end comes the whole run */
const CODE_RUN_S=0.48, CODE_END_S=0.32, CODE_END_MIN=0.3;
const codePace=()=>({run:CODE_RUN_S/spd(), end:Math.max(CODE_END_MIN,CODE_END_S/spd())});
function codePlay(){const o=cur(), st=o.st, n=st.length, t=play.t, e=endView(o); let s, ph=0;
  if(e>=0) s=e+1; else {const k=stepAt(o,t); if(k>=0){s=k; ph=codePace().run*Math.max(0,t-st[k].t0)/(st[k].t1-st[k].t0);} else s=st.findIndex(x=>x.t0>=t-EPS);}
  if(s<0||s>=n){onEnd(); return;}
  codeTouched=true; stopPlay(); play.playing=true; play.last=performance.now(); const my=runTok;
  const place=()=>{const x=st[s], p=codePace(); if(ph<p.run){codeEdge='start'; play.t=x.t0+(x.t1-x.t0)*Math.min(ph/p.run,0.999);} else {codeEdge='end'; play.t=x.t1;}};
  place(); frame(); updateRail(); drawMetrics(o);
  let lastEnd=ph>=codePace().run;          // the card follows the phase as well as the step (review of v10, S3)
  const tick=now=>{if(!app||!play.playing||my!==runTok) return; const dt=Math.min(0.1,(now-play.last)/1000); play.last=now;
    ph+=dt; let moved=false; for(let p=codePace();ph>=p.run+p.end;p=codePace()){ph-=p.run+p.end; s++; moved=true; if(s>=n) break;}
    if(s>=n){onEnd(); return;}
    place(); frame(); const e2=ph>=codePace().run; if(moved||e2!==lastEnd){lastEnd=e2; updateRail();} play.raf=requestAnimationFrame(tick);};
  play.raf=requestAnimationFrame(tick);}
/* a setting changed: run it and replay from 0 (reduced motion: the whole run at once). R3.20: a run already watched to the end (an
   option, a scenario, Settings or a challenge plan picked again, A → B → A) shows whole at once, and its end counts (the challenge's
   verdict shows); ↺ Replay plays it again */
function replayNew(){bump(); stopPlay(); const o=cur(); if(RM()||seen.has(o.key)){play.t=null; seen.add(o.key); onEnd(); return;} play.t=0; render(); startPlay(true);}
function stepMove(dir){
  bump(); stopPlay(); codeEdge=dir>0?'end':'start'; if(codeOn()) codeTouched=true; const o=cur(), st=o.st; let t=play.t;
  if(dir>0){if(t==null) t=st[0].t1; else {const s=stepAt(o,t), e=endView(o); if(e>=0) t=e+1<st.length?st[e+1].t1:null; else if(s>=0) t=st[s].t1; else {const nx=st.find(x=>x.t0>t+EPS); t=nx?nx.t1:null;}}
    if(t==null||t>o.end+1e-9){play.t=null; onEnd(); return;}}   // the last step's end shows too (review B, B4); the next ▸ goes to the whole run
  else {if(t==null) t=st[st.length-1].t0; else {const s=stepAt(o,t), d=doneAt(o,t); if(s>=0&&t>st[s].t0+1e-6) t=st[s].t0; else if(s>=1) t=st[s-1].t0; else if(d>=0) t=st[d].t0; else t=0;}}
  play.t=Math.max(0,t); render();}
function scrub(v){bump(); stopPlay(); const o=cur(), sp=spanFor(o), t=+v/1000*sp; if(t>=o.end-1e-9){play.t=null; onEnd(); return;} play.t=t; render();}
function toStall(){const o=cur(), w=o.worst; if(!w) return; bump(); stopPlay(); const s=o.st[w.s]; play.t=s.t0+0.45*(s.t1-s.t0); render();
  const lb=$('#sc-stall',app); if(lb){lb.classList.remove('sc-pulse'); void lb.offsetWidth; lb.classList.add('sc-pulse');}
  const el=$('#sc-strip',app); if(el&&innerWidth>760) A.ui.flash(el);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p,noPlay){
  const k0=keyOf(S.load,S), was=S.load; Object.assign(S,p); if(!LOADS.includes(S.load)) S.load='stall'; if(!CAPS.includes(S.cap)) S.cap=32; if(!BUDS.includes(S.budget)) S.budget=0; if(!CBS.includes(S.cb)) S.cb=null;
  if(S.policy!=='static') S.policy='continuous'; S.cache=!!S.cache;
  if(S.load!==was) sel=null; relCache={k:'',v:new Set()};
  if(noPlay){bump(); stopPlay(); play.t=seen.has(cur().key)?null:0; render(); return;}
  /* R3.20: the option already picked, Reset at the initial setup, a scenario already open: the run on screen stays as it is (finished,
     playing or paused); only a run that differs plays from 0. ↺ Replay plays it again */
  if(keyOf(S.load,S)===k0){render(); return;}
  replayNew();}
function enterStep(i,wasPl){
  step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step], was=cur().key, pl=wasPl!=null?wasPl:play.playing; focusWant={step,at:Date.now()};
  Object.assign(S,{...PART_CFG[s.id]}); if(s.id!=='cap') S.policy='continuous'; sel=null; relCache={k:'',v:new Set()};
  const st=$('#sc-stage',app); if(st) st.scrollTop=0;
  bump(); stopPlay();
  if(s.id==='code'){if(codeM){renderRail(); pickMoment(codeM); if(innerWidth>760) showCode(false); return;} seen.add(cur().key); play.t=null; renderRail(); render(); if(innerWidth>760) showCode(false); return;}
  const o=cur(); if(o.key===was&&seen.has(o.key)) play.t=null;
  else if(o.key===was&&s.id!=='predict'&&(pl||(play.t!=null&&play.t>1e-9))){renderRail(); render(); if(pl&&!RM()) startPlay(false); return;}   // R3.20: the same run, part-way: it goes on from there
  else if(s.id==='mission'&&!seen.has(o.key)) play.t=0; else if(s.id==='predict'){play.t=null; seen.add(o.key);} else if(seen.has(o.key)) play.t=null; else {play.t=0; renderRail(); render(); if(RM()){onEnd(true);} else startPlay(true); return;}
  renderRail(); render();}
function revealStep(){A.ui.reveal($('#sc-rail .step.current',app)||$('#sc-rail',app));}
function setMode(m){
  if(m===mode) return; const was=mode, k0=cur().key, pl=play.playing; mode=m;
  if(was==='sandbox') sbState={...S};
  bump(); stopPlay(); sel=null;
  A.shell.setMode(app,m,DESCS());
  const st=$('#sc-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step,pl); return;}
  if(m==='sandbox'){S={...(sbState||{load:'stall',...BASE})}; sbTouched=false;}
  if(m==='challenge'){S={load:'challenge',...(chRunKey?parseKey(chRunKey):BASE)}; if(!chRunKey){chCfg={...BASE};}}
  const o=cur();
  if(m==='sandbox'&&o.key===k0&&!seen.has(o.key)&&(pl||(play.t!=null&&play.t>1e-9))){renderRail(); render(); if(pl&&!RM()) startPlay(false); return;}   // R3.20: the same run goes on
  play.t=seen.has(o.key)||m==='challenge'?null:0; if(m==='challenge') seen.add(baseOf('challenge').key);
  renderRail(); render(); if(play.t===0&&!RM()) startPlay(true); else if(play.t===0){onEnd();}}
const token=()=>keyOf(S.load,S);
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6||!LOADS.includes(p[0])||p[0]==='challenge') return null;
  const s=parseKey(tok); if(!CAPS.includes(s.cap)||!BUDS.includes(s.budget)||!['c','s'].includes(p[1])||!['0','1'].includes(p[4])||(p[5]!=='n'&&!CBS.includes(+p[5]))) return null;
  return {load:p[0],...s};}

/* =====================================================================
   interview drill: seeded, computed with the engine on a small load of its own (never the lab's loads)
   ===================================================================== */
function drillCase(seed){const rnd=E.mulberry(seed*7919+17), pick=a=>a[Math.floor(rnd()*a.length)];
  const U=pick([8,16,24]), c=pick([1024,2048]), N=pick([16384,24576,49152]), B=pick([1024,4096]);
  const sp=[]; for(let i=0;i<U;i++) sp.push({t:0,cls:'chat',own:c,out:400}); sp.push({t:3,cls:'doc',own:N,out:50});
  const k='drill|'+[U,c,N,B].join('_'); if(XC.has(k)) return XC.get(k);
  const a=build(sp,cfgOf(BASE)), b=build(sp,cfgOf({...BASE,budget:B})), v={U,c,N,B,a,b}; XC.set(k,v); return v;}
function renderDrill(){if(!app) return; const d=drillCase(drillSeed), q=$('#sc-drill-q',app), an=$('#sc-drill-a',app);
  q.innerHTML=tx(`<b>${MODEL}</b> on one ${GPUN} (BF16; 50% of 989.5 TFLOP/s, 60% of 3.35 TB/s). <b>${d.U}</b> chats are decoding, each with about <b>${fint(d.c)}</b> tokens of context, when a <b>${fint(d.N)}</b>-token document arrives. With no token budget, how long do the chats wait for their next token? And with a budget of <b>${fint(d.B)}</b> tokens per step?`,
                 `<b>${MODEL}</b> 跑在一张 ${GPUN} 上（BF16；989.5 TFLOP/s 的 50%，3.35 TB/s 的 60%）。<b>${d.U}</b> 个对话在 decode，每个约 <b>${fint(d.c)}</b> 个 token 的上下文，这时来了一份 <b>${fint(d.N)}</b> 个 token 的文档。不设 token 预算时，对话要等多久才拿到下一个 token？每步预算 <b>${fint(d.B)}</b> 个 token 时呢？`);
  an.hidden=!drillOpen; if(!drillOpen) return;
  const one=(o,label)=>{const w=o.worst, inf=stepInfo(o,w.s), s=inf.s;
    return `<p><b>${label}</b> ${inf.big&&inf.big.whole?tx('The longest wait is the step with the document’s whole prompt:','最长的等待是装着文档整个 prompt 的那一步：'):tx(`The longest wait is the step with the document’s tokens ${fint(inf.big?inf.big.a+1:0)} to ${fint(inf.big?inf.big.b:0)}:`,`最长的等待是装着文档第 ${fint(inf.big?inf.big.a+1:0)} 到 ${fint(inf.big?inf.big.b:0)} 个 token 的那一步：`)} ${fint(inf.dec.length)} ${tx('decode tokens and','个 decode token 加上')} ${fint(inf.pre.reduce((x,p)=>x+p.n,0))} ${tx('prompt tokens','个 prompt token')} (${tx('the biggest piece','最大的一块')} ${fint(inf.big?inf.big.n:0)}).<br>`+
      `${tx('arithmetic','计算')} = 2 × ${fint(C.P_LAYERS)} × ${fint(s.T)} + 2 × ${fint(C.P_HEAD)} × ${fint(inf.logits)} + ${fint(inf.att)} = ${fint(s.fl)} FLOP ÷ ${fixd(C.FS/1e12,2)} TFLOP/s = ${fS(s.tc)}; `+
      `${tx('memory','读写显存')} = ${fint(C.W_READ)} + ${fint(C.KV_TOK)} × ${fint(inf.kv)} = ${fint(s.by)} ${tx('bytes','字节')} ÷ ${fixd(C.BS/1e12,2)} TB/s = ${fS(s.tm)} → <b>${fS(w.g)}</b>.</p>`;};
  an.innerHTML=`<p><b>${tx(`No budget ≈ ${fS(d.a.worst.g)}; budget ${fint(d.B)} ≈ ${fS(d.b.worst.g)}.`,`不设预算 ≈ ${fS(d.a.worst.g)}；预算 ${fint(d.B)} ≈ ${fS(d.b.worst.g)}。`)}</b></p>`+one(d.a,tx('No budget.','不设预算。'))+one(d.b,tx(`Budget ${fint(d.B)}.`,`预算 ${fint(d.B)}。`))+
    `<p>${tx(`Without a budget the whole prompt is one step, set by its arithmetic; with one, the longest step holds one of the document’s last chunks, whose tokens attend to the most earlier prompt tokens. The gap goal of ${fS(G.chat.gap)} is ${d.b.worst.g<=G.chat.gap+EPS?'met':'still missed'} with the budget.`,`不设预算时整个 prompt 是一步，由计算决定；设了预算，最长的一步装着文档靠后的一个 chunk，它的 token 要对前面最多的 prompt token 做 attention。有预算时，${fS(G.chat.gap)} 的间隔目标${d.b.worst.g<=G.chat.gap+EPS?'达到了':'仍然没达到'}。`)}</p>`;}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const TERM={
 fs:()=>tx(`${fixd(C.FS/1e12,2)} TFLOP/s is 50% of the ${GPUN}’s dense BF16 peak, 989.5 TFLOP/s (its datasheet’s 1,979 TFLOPS with sparsity, halved). The 50% is this lab’s assumption.`,`${fixd(C.FS/1e12,2)} TFLOP/s 是 ${GPUN} 稠密 BF16 峰值 989.5 TFLOP/s（规格表上 with sparsity 的 1,979 TFLOPS 减半）的 50%。50% 是这个 Lab 的假设。`),
 bs:()=>tx(`${fixd(C.BS/1e12,2)} TB/s is 60% of the ${GPUN}’s 3.35 TB/s. The 60% is this lab’s assumption.`,`${fixd(C.BS/1e12,2)} TB/s 是 ${GPUN} 3.35 TB/s 的 60%。60% 是这个 Lab 的假设。`),
 pl:()=>tx(`${fint(C.P_LAYERS)} parameters in the ${C.L} layers (and the final RMSNorm) multiply every token a step computes: 2 FLOPs each.`,`${C.L} 层（加最后的 RMSNorm）里的 ${fint(C.P_LAYERS)} 个参数，要和一步算的每个 token 相乘：每个 2 FLOP。`),
 ph:()=>tx(`${fint(C.P_HEAD)} parameters in the LM head (${fint(C.D)} × ${fint(C.VOCAB)}), used once per new token.`,`LM head 的 ${fint(C.P_HEAD)} 个参数（${fint(C.D)} × ${fint(C.VOCAB)}），每个新 token 用一次。`),
 att:()=>tx(`Attention: a decode token at context c compares with c + 1 keys, a prompt chunk a..b−1 with all earlier prompt tokens: 2 × ${C.L} × ${fint(C.D)} × (b(b+1) − a(a+1)) FLOPs.`,`attention：上下文为 c 的 decode token 要和 c + 1 个 key 比较；prompt 的 chunk a..b−1 要和前面所有 prompt token 比较：2 × ${C.L} × ${fint(C.D)} × (b(b+1) − a(a+1)) FLOP。`),
 wr:()=>tx(`${fint(C.W_READ)} bytes: every parameter a step multiplies (all but the embedding table), 2 bytes each, read once per step however many tokens it computes.`,`${fint(C.W_READ)} 字节：一步要用到的全部参数（除 embedding 表），每个 2 字节；不管算多少 token，每步读一遍。`),
 kvt:()=>tx(`${fint(C.KV_TOK)} bytes (${fint(C.KV_TOK/1024)} KiB): one token’s keys and values, 2 × ${C.L} layers × ${C.KVH} KV heads × ${C.HD} × 2 bytes.`,`${fint(C.KV_TOK)} 字节（${fint(C.KV_TOK/1024)} KiB）：一个 token 的 key 和 value，2 × ${C.L} 层 × ${C.KVH} 个 KV head × ${C.HD} × 2 字节。`),
 wm:()=>tx(`${fint(C.W_MEM)} bytes: all the weights in GPU memory, the embedding table included, 2 bytes per parameter.`,`${fint(C.W_MEM)} 字节：显存里的全部权重，包括 embedding 表，每个参数 2 字节。`)
};
const FIXED=new Set(Object.keys(TERM));
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${jumpAct(k)}</div>`:'';
function jumpAct(k){
  if(k==='controls') return mode==='sandbox'?tx('Click to jump to Settings.','点击跳到“自由调参”。'):tx('Click to jump to Settings; in this mode it explains where the settings are.','点击跳到“自由调参”；这个模式下那里说明了设置在哪。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='stall') return tx('Click to put the playhead in the step behind the longest gap.','点击把播放头放到最长间隔的那一步。');
  if(k==='tl') return tx('Click to scroll to the timeline.','点击滚到时间线。');
  if(k==='code') return tx('Click to scroll to the code on the stage.','点击滚到舞台上的代码。');
  if(k==='obs') return tx('Click to scroll to what the card says about this step.','点击滚到卡片里对这一步的说明。');
  if(FIXED.has(k)) return tx(`Fixed in this lab: it follows from ${MODEL} on one ${GPUN}.`,`这个 Lab 里不能改：它由一张 ${GPUN} 跑 ${MODEL} 决定。`);
  return '';}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#sc-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:RM()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='stall'){A.tip.hide(); if(mode==='guided'&&STEPS[step].id!=='predict'&&STEPS[step].id!=='mission'){} toStall(); const el=$('#sc-strip',app); if(el&&innerWidth<=760) el.scrollIntoView({block:'center',behavior:RM()?'auto':'smooth'}); return;}
  if(k==='tl'){const el=$('#sc-tlw',app); A.tip.hide(); el.scrollIntoView({block:'center',behavior:RM()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(k==='code'){const el=$('#sc-wc',app); A.tip.hide(); if(el&&!el.hidden) showCode(true); return;}
  if(k==='obs'){const el=$('#sc-rail [data-obs]',app); A.tip.hide(); if(el&&!el.hidden){A.ui.jumpToEl(el);} return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#sc-controls',app)); return;}
  if(FIXED.has(k)&&src) A.tip.show(src,termTipHTML(k));}

/* =====================================================================
   events
   ===================================================================== */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop(); let lastY=null, waited=0, pulled=false; const my=runTok;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#sc-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#sc-rail .step.current',app)||$('#sc-rail .railcard',app)||$('#sc-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    if(c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    if(!pulled){pulled=true; let dy=r.top-want; if(dy<-2){dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2){window.scrollBy({top:dy,behavior:RM()?'auto':'smooth'}); lastY=null; waited=0; stageT=setTimeout(tick,120); return;}}}
    /* then, once the page is still, the block the part's card talks about (review r5) */
    const f=focusWant; focusWant=null; if(f&&f.step===step&&Date.now()-f.at<4000&&mode==='guided') fitFocus(); void my;};
  stageT=setTimeout(tick,RM()?30:150);}
/* review r5 (B4; the owner: "the playback panel was pushed down, at first I did not see there was more below"): the stage is taller than
   the window in every part on a desktop, even with its spacing tightened and the timeline drawn shorter (the three views, the table of the
   whole run and Show the math do not fit in 814 px). Entering a part brings the block its card talks about fully on screen: the timeline
   (parts 1-3), the KV pool (4-5) or the strip (6); part 7's code panel has showCode. The page scrolls until the stage sticks, never past
   the head of the part's card (as showCode), then the stage scrolls inside; the rest of the stage (the table, the math) stays below */
let focusWant=null;
function fitFocus(){const q={tl:'#sc-tlw',mem:'#sc-mem',strip:'#sc-strip'}[STEPS[step].focus], el=q&&$(q,app), st=$('#sc-stage',app);
  if(!el||!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
  const beh=RM()?'auto':'smooth', bar=document.querySelector('.topbar'), barB=bar?bar.getBoundingClientRect().bottom:58, card=$('#sc-rail .step.current',app);
  const sr=st.getBoundingClientRect(), r=el.getBoundingClientRect(), stick=parseFloat(getComputedStyle(st).top)||70, bw=st.clientTop||0;
  const hiAt=py=>Math.min(sr.bottom-py-bw,innerHeight)-8, loAt=py=>Math.max(sr.top-py+bw,barB)+6;   // room for the block's dashed outline
  if(r.top>=loAt(0)&&r.bottom<=hiAt(0)) return;
  /* all the way until the stage sticks, its whole box on screen: scrolling only as far as the block needed (85 px at part 4) left the stage's
     foot below the window, and the cards after it then stayed where the kit's reveal leaves a card already on screen, so parts 7 and 8 were
     entered with the stage at 117-126 px instead of its sticky line */
  let py=0; if(r.bottom>hiAt(0)){py=Math.max(0,sr.top-stick); if(card) py=Math.max(0,Math.min(py,card.getBoundingClientRect().top-(barB+8)));}
  let sy=r.bottom-py-hiAt(py); if(r.top-py-sy<loAt(py)) sy=r.top-py-loAt(py);   // a block taller than the room: its top
  sy=Math.max(-st.scrollTop,Math.min(sy,st.scrollHeight-st.clientHeight-st.scrollTop));
  if(py>2) window.scrollBy({top:py,behavior:beh});
  if(Math.abs(sy)>2) st.scrollBy({top:sy,behavior:beh});}
/* the timeline and the strip are drawn for the stage's width, and the stage changes width without a window resize (the page puts the step
   list beside it, or leaves the primer), so they are drawn again when it does (as labs/ppo.js, review r5) */
let stRO=null, stW=0;
function watchStage(){unwatchStage(); const st=app&&$('#sc-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w;
    const ax=$('#sc-axis',app); if(ax) ax._k=''; const g=$('#sc-gbar',app); if(g) g._h=''; frame();}));
  stRO.observe(st);}
function unwatchStage(){if(stRO){stRO.disconnect(); stRO=null;}}
/* the trace card: beside the selected row's bar, on the side with more room, so it never covers the row it explains nor the steps the row
   waited for (they lie inside the bar's time range); at least 250 px wide, else under the timeline in the flow. Its height follows its
   text, never cut off (review r2, A1/N4). Brought into view once per selection */
function placeCard(o){const tc=$('#sc-tcard',app), g=tlGeo, box=$('#sc-tlbox',app); if(!tc||tc.hidden||!g||!box||sel==null) return; const r=o.R[sel]; if(!r) return;
  const H=box.clientHeight, y=g.top+sel*g.rh, x0=g.X(r.arr), x1=g.X(Math.min(r.fin,g.span)), gp=10, room=[g.W-x1-gp-2, x0-g.GX-gp-2], side=room[0]>=room[1]?0:1, w=Math.min(380,room[side]);
  if(w>=250){tc.classList.remove('sc-below'); tc.style.width=w.toFixed(0)+'px'; const ch=tc.offsetHeight, L=side===0?x1+gp:x0-gp-w, vis=Math.min(H,innerHeight-8-box.getBoundingClientRect().top), T=Math.max(0,Math.min(y-ch/2,vis-ch));   /* within the part of the timeline on screen */
    tc.style.left=L.toFixed(0)+'px'; tc.style.top=T.toFixed(0)+'px';}
  else {tc.classList.add('sc-below'); tc.style.width=''; tc.style.left=''; tc.style.top='';}
  const pl=$('#sc-pflab',app); if(pl&&!pl.hidden){const a=pl.getBoundingClientRect(), c=tc.getBoundingClientRect(); if(!(a.right<=c.left||a.left>=c.right||a.bottom<=c.top||a.top>=c.bottom)) pl.hidden=true;}   /* the card is the focus: never on top of the label */
  if(tc._sel!==sel){tc._sel=sel; const bar=document.querySelector('.topbar'), topL=(bar?bar.offsetHeight:56)+4;
    requestAnimationFrame(()=>{if(tc.hidden) return; const rc=tc.getBoundingClientRect(); if(rc.top<topL||rc.bottom>innerHeight-4){tc.scrollIntoView({block:'nearest',behavior:'auto'});   /* checked once laid out */
      const ov=tc.getBoundingClientRect().bottom-(innerHeight-2); if(ov>0){const sp=scrollHost(tc); if(sp) sp.scrollTop+=Math.ceil(ov); else window.scrollBy(0,Math.ceil(ov));}}});}}   /* “nearest” can leave a fraction of a pixel below the screen */
/* the element that scrolls a node: the sticky stage on a desktop, else the page */
function scrollHost(e){for(let p=e.parentElement;p&&p!==document.body;p=p.parentElement){const s=getComputedStyle(p).overflowY; if((s==='auto'||s==='scroll')&&p.scrollHeight>p.clientHeight) return p;} return null;}
function selectReq(id){const o=cur(); sel=(id==null||sel===id)?null:id; relCache={k:'',v:new Set()}; if(sel!=null) focusRow=-1; frame(); const mb=$('#sc-mathbody',app), mh=mathHTML(o); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} fitMath(mb);
  if(sel!=null){const tr=$('#sc-trace',app); if(tr&&innerWidth<=760) tr.scrollIntoView({block:'nearest',behavior:RM()?'auto':'smooth'});}}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function copyText(txt,btn,back){const done=()=>flashBtn(btn,tx('Copied','已复制'),back);
  const fb=()=>{const ta=document.createElement('textarea'); ta.value=txt; ta.setAttribute('readonly',''); ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta); ta.select(); let ok=false; try{ok=document.execCommand('copy');}catch(e){} ta.remove(); if(ok) done(); else flashBtn(btn,tx('Select the text and press Ctrl+C','选中文字后按 Ctrl+C'),back);};
  try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done,fb); else fb();}catch(e){fb();}}
function act(a,btn){
  if(a==='play'){if(btn&&btn.closest('#sc-rail')&&innerWidth>760){} startPlay(false); return;}
  if(a==='replay'){bump(); startPlay(true); return;}
  /* ▸ and ◂ are pressed again and again: their tooltips stay away until the pointer moves (review A, finding 11), and nothing scrolls,
     so the button stays under the pointer (the panel is in view since the moment was picked) */
  if(a==='fwd'||a==='back'){if(A.tip.quiet) A.tip.quiet(); else A.tip.hide(); stepMove(a==='fwd'?1:-1); return;}
  if(a==='toend'){bump(); stopPlay(); onEnd(); return;}
  if(a==='unsel'){selectReq(null); return;}
  if(a==='reset'){byLearner(()=>setState({...BASE})); return;}
  if(a==='run'){const k0=cur().key, pl=play.playing; chSubmitted=true; chTier=null; chReveal=false; chRunKey=keyOf('challenge',chCfg); S={load:'challenge',...chCfg};
    /* R3.20: the plan on screen submitted again is not played again: its result shows at once (a run still playing goes on, and its end
       shows the result) */
    if(chRunKey===k0){renderRail(); if(!pl){bump(); stopPlay(); onEnd();} return;}
    sel=null; renderRail(); replayNew(); return;}
  if(a==='reveal'){chReveal=true; updateRail(); return;}
  if(a==='copyplan'){const o=RC.get(chRunKey); if(o) copyText(planText(o,baseOf('challenge')),btn,tx('Copy the plan','复制方案')); return;}
  if(a==='copy'){copyText(CODE.map(x=>x[0]).join('\n'),btn,A.t('copyCode')); return;}
  if(a==='gocode'){if(mode!=='guided') setMode('guided'); enterStep(SIX('code')); const rl=$('#sc-rail',app); if(rl) rl.scrollIntoView({block:'start',behavior:RM()?'auto':'smooth'}); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
}
let userAct=false;
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
/* a change under Settings never moves the page (R3.2) */
function holdPlace(el,f){if(!el||!el.isConnected){f(); return;} const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top; de.style.overflowAnchor='none';
  try{f();} finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}}
function segClick(key,v,b){
  if(key==='speed'){speed=+v; $$('.seg-ctl[data-key="speed"]',app).forEach(g=>A.ui.setPressed(g,String(speed))); return;}
  if(key.startsWith('ch-')){const k=key.slice(3); chCfg={...chCfg,[k]:k==='cache'?v==='true':(k==='policy'?v:(k==='cb'?(v==='null'?null:+v):+v))}; if(k==='cache'&&!chCfg.cache) chCfg.cb=null;
    const rail=$('#sc-rail',app); const y0=b.getBoundingClientRect().top; rail.querySelector('.sc-chcard').outerHTML=challengeHTML(); A.ui.initRanges(rail); const nb=$(`.seg-ctl[data-key="${key}"] button[data-v="${v}"]`,rail); if(nb){const dy=nb.getBoundingClientRect().top-y0; if(Math.abs(dy)>=1) window.scrollBy(0,dy); nb.focus({preventScroll:true});} syncControls(); return;}
  const val=key==='cache'?v==='true':(key==='policy'?v:(key==='cb'?(v==='null'?null:+v):+v));
  const p={[key]:val}; if(key==='cache'&&!val) p.cb=null;
  const inRail=!!b.closest('#sc-rail'); if(inRail&&mode==='guided'){const s=STEPS[step]; if(s.lever) tried[s.id]=true;}
  const f=()=>byLearner(()=>setState(p));
  if(b.closest('#sc-controls')) holdPlace(b,f); else f();}
function bind(el){
  el.addEventListener('input',e=>{const tg=e.target; if(tg.id==='sc-scrub') scrub(tg.value);});
  el.addEventListener('click',e=>{
    const tl=e.target.closest('#sc-tl'); if(tl&&tlGeo){const r=tl.getBoundingClientRect(), y=e.clientY-r.top, i=Math.floor((y-tlGeo.top)/tlGeo.rh); if(i>=0&&i<tlGeo.n) selectReq(i); return;}
    const sm=e.target.closest('[data-sel]'); if(sm&&el.contains(sm)&&!sm.closest('[data-exits]')){e.preventDefault(); selectReq(+sm.dataset.sel); return;}
    const ms=e.target.closest('.sc-ms-sh,.sc-ms-idle'); if(ms) return;
    if(e.target.closest('#sc-stall')){toStall(); return;}
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#sc-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]'); if(seg&&d.v!=null){segClick(seg.dataset.key,d.v,b); return;}
    if(d.lever!=null){const s=STEPS[step]; if(!s.lever) return; tried[s.id]=true; byLearner(()=>setState(LEVER[s.lever].set(leverVal(s.lever,d.lever)))); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.moment){pickMoment(d.moment); requestAnimationFrame(()=>{revealObs(); showCode(true);}); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.id==='predict'){toStall(); return;} if(s.id==='code'){pickMoment('budget'); requestAnimationFrame(()=>{revealObs(); showCode(true);}); return;} if(s.showMe){tried[s.id]=true; byLearner(()=>setState(s.showMe));} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#sc-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); requestAnimationFrame(()=>revealObs());
      const lb=$('#sc-stall',app); if(lb){lb.classList.remove('sc-pulse'); void lb.offsetWidth; lb.classList.add('sc-pulse');} return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({load:p.load})); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#sc-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  el.addEventListener('keydown',e=>{if(e.key==='Escape'&&sel!=null){selectReq(null); return;} const tl=e.target.closest&&e.target.closest('#sc-tl'); if(!tl) return; const n=cur().R.length;
    if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault(); focusRow=Math.max(0,Math.min(n-1,(focusRow<0?(sel!=null?sel:-1):focusRow)+(e.key==='ArrowDown'?1:-1))); drawTL();
      A.announce(rName(cur().R[focusRow]));}
    else if(e.key==='Enter'||e.key===' '){if(focusRow>=0){e.preventDefault(); selectReq(focusRow);}}});
  el.addEventListener('focusout',e=>{if(e.target.id==='sc-tl'){focusRow=-1; drawTL();}});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
// a setup with any load, the challenge's included, checked with the shared settings' own parse(); cb is kept as set even while the cache is off
const sCheck=o=>{if(!o||typeof o!=='object'||!LOADS.includes(o.load)) return null; const p=parse(keyOf('stall',{policy:'',cap:0,budget:0,cache:false,cb:null,...o}));
  if(!p) return null; p.load=o.load; if(!p.cache&&CBS.includes(o.cb)) p.cb=o.cb; return p;};
const keyCheck=k=>{const q=typeof k==='string'?k.split('_'):[]; return q.length===6&&LOADS.includes(q[0])&&!!parse(['stall',...q.slice(1)].join('_'));};
const sOf=x=>({load:x.load,policy:x.policy,cap:x.cap,budget:x.budget,cache:x.cache,cb:x.cb});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:sOf(S),pred,sel,speed,sbState:sbState?sOf(sbState):null,sbTouched,tried:{...tried},seen:[...seen].slice(-200),codeM,codeSeen:[...codeSeen],codeTouched,
    chCfg:{policy:chCfg.policy,cap:chCfg.cap,budget:chCfg.budget,cache:chCfg.cache,cb:chCfg.cb},chRunKey,chSubmitted,chReveal,chTier:chTier?{...chTier}:null,whyOpen,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&!!chTier&&chTier.tier!==3}),
  restore(o){if(!o||o.v!==1) return; const p=sCheck(o.S); if(!p) return;
    const md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; if((md==='challenge')!==(p.load==='challenge')) return;
    mode=md; S=sOf(p); step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    speed=SPEEDS.includes(o.speed)?o.speed:null; const sb=sCheck(o.sbState); sbState=sb&&sb.load!=='challenge'?sOf(sb):null; sbTouched=o.sbTouched===true;
    for(const k in tried) delete tried[k]; if(o.tried&&typeof o.tried==='object') for(const k in o.tried) if(STEPS.some(x=>x.id===k)&&o.tried[k]===true) tried[k]=true;
    seen.clear(); if(Array.isArray(o.seen)) o.seen.slice(-200).forEach(k=>{if(keyCheck(k)) seen.add(k);});
    codeM=momentOf(o.codeM)?o.codeM:null; codeSeen.clear(); if(Array.isArray(o.codeSeen)) o.codeSeen.forEach(id=>{if(momentOf(id)) codeSeen.add(id);}); codeTouched=o.codeTouched===true;
    const cc=sCheck({load:'challenge',...(o.chCfg||{})}); chCfg=cc?{policy:cc.policy,cap:cc.cap,budget:cc.budget,cache:cc.cache,cb:cc.cb}:{...BASE};
    chRunKey=mode==='challenge'&&keyCheck(o.chRunKey)&&o.chRunKey.indexOf('challenge_')===0?o.chRunKey:null; chSubmitted=!!chRunKey&&o.chSubmitted===true;
    const t=o.chTier; chTier=chSubmitted&&t&&typeof t==='object'&&[1,2,3].includes(t.tier)?(t.tier===2?(sCheck({load:'challenge',...(t.alt||{})})?{tier:2,alt:{policy:t.alt.policy,cap:t.alt.cap,budget:t.alt.budget,cache:t.alt.cache,cb:t.alt.cb}}:null):{tier:t.tier}):null;
    chReveal=chSubmitted&&o.chReveal===true; whyOpen=o.whyOpen===true;
    if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true;
    // the playhead, as enterStep() and setMode() leave it: part 7 at the moment picked last; a run watched to the end whole; any other at its start
    stopPlay(); bump(); const sid=mode==='guided'?STEPS[step].id:'', m=sid==='code'&&codeM?momentOf(codeM):null;
    if(m){Object.assign(S,{...m.cfg}); const ms=momentStep(m), st=ms.o.st[ms.s]; seen.add(ms.o.key); codeEdge='end'; play.t=m.at==='end'?st.t1:st.t0+0.45*(st.t1-st.t0);}
    else play.t=seen.has(cur().key)?null:0;
    const R0=cur().R; sel=Number.isInteger(o.sel)&&R0&&R0[o.sel]?o.sel:null; resumed=true;},
  mount(el,tok){
    root=el; const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(!p&&mode==='guided'&&step===0&&!seen.size) Object.assign(S,{...PART_CFG.mission});
    el.innerHTML=viewHTML(); app=$('#sc-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#sc-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    for(const k in RC){} bump(); stopPlay();
    const o=cur(); if(p){play.t=null; seen.add(o.key);} else if(play.t!=null&&!seen.has(o.key)&&play.t>0) play.t=Math.min(play.t,o.end);
    const rail=$('#sc-rail',app); rail.innerHTML=railFull(); A.ui.initRanges(rail);
    render(); watchStage();
    if(p&&!RM()){play.t=0; startPlay(true);}
  },
  leave(){bump(); clearTimeout(annT); stageStop(); unwatchStage(); stopPlay(); root=null; app=null;},
  onLang(){if(root){const d=$('#sc-math',app), open=d&&d.open, t=play.t, was=play.playing; bump(); stopPlay(); this.mount(root,null); play.t=t; if(open) $('#sc-math',app).open=true; render(); if(was) startPlay(false);}},
  onResize(){if(app){const ax=$('#sc-axis',app); if(ax) ax._k=''; const g=$('#sc-gbar',app); if(g) g._h=''; frame();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#sc-strip',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#sc-graph',app),
    render({status,vis,read}){if(!app) return; const o=cur(), s=statusOf(o), tt=play.t==null?Infinity:play.t;
      status.className='status '+s.cls; status.textContent=s.short;
      let ok=0, miss=0, pend=0; o.R.forEach(r=>{if(r.arr>tt) {pend++; return;} const g=tt===Infinity?(r.ok?'ok':'miss'):goalAt(r,tt); if(g==='ok') ok++; else if(g==='miss') miss++; else pend++;});
      const n=o.R.length, w=v=>(v/n*100).toFixed(1)+'%';
      vis.innerHTML=`<div class="sc-dk"><i class="ok" style="width:${w(ok)}"></i><i class="bad" style="width:${w(miss)}"></i><b style="left:${(E.TARGET*100).toFixed(0)}%"></b></div>`;
      read.innerHTML=esc(play.t==null?tx(`${o.m.all.ok} of ${n} requests meet their goal (the service needs ${pct0(E.TARGET)})`,`${n} 个请求里 ${o.m.all.ok} 个达标（服务要求 ${pct0(E.TARGET)}）`)
        :tx(`At ${fC(play.t)}: ${ok} met, ${miss} missed, ${pend} still open`,`${fC(play.t)}：${ok} 个达标，${miss} 个没达标，${pend} 个还没定`))+`<span class="dk-mem">${esc(tx(`; longest gap ${fS(o.m.chat.gap_max)}`,`；最长间隔 ${fS(o.m.chat.gap_max)}`))}</span>`;}
  }
});

/* =====================================================================
   the home tile (CONTRACT §5, R11.1, R11.5): how the congested service is rescued. One control with three options on part 1's load:
   without the 32K document, with it (part 1's run: every running chat stops when it arrives), and with it under the setup the guided
   parts end on (PART_CFG.remember: one scheduler setting changed from the initial setup). One picture (every request as a thin row in
   arrival order: waiting thin, getting tokens thick, red when it misses its goal, the document boxed) and one sentence. The tile never
   says why the chats stop and names no setting (R3.8: part 2's prediction asks why; the challenge, on another load, needs more than
   this one change). Everything comes from SC_ENGINE; tests/serving-capstone_numbers.py redraws every row of the three runs from
   capstone/engine.py; window.__scTz is its hook.
   ===================================================================== */
const TZOPTS=['none','sent','fix'];
let tzOpt='sent';                                        // the tile opens on the congested service; a re-render keeps the choice
const TZ={}, TZH={top:3,rows:108,docH:9,ax:15};          // px: the rows' band (the document's row is taller, for its box), the time axis
const tzFix=()=>PART_CFG.remember;                       // the setup the guided parts end on, on the same load
/* one run. A request keeps its row, its place in the whole load in arrival order, in every run, so switching moves no chat. A row: thin
   from arrival to the first token, thick while tokens come at most one chat gap goal (150 ms) apart, thin again across a longer wait */
function tzRun(k){if(TZ[k]) return TZ[k];
  const full=spec('stall'), doc=k!=='none', sp=doc?full:full.filter(x=>x.cls!=='doc'), run=E.simulate(sp,k==='fix'?cfgOf(tzFix()):E.BASELINE); E.check(run);
  const m=E.metrics(run), gg=G.chat.gap;
  const rows=run.R.map(r=>{const tk=r.toks, seg=[[r.arr,r.first,'w']]; let s0=tk[0];
    for(let j=1;j<tk.length;j++) if(tk[j]-tk[j-1]>gg+EPS){seg.push([s0,tk[j-1],'t'],[tk[j-1],tk[j],'w']); s0=tk[j];}
    seg.push([s0,tk[tk.length-1],'t']);
    const g=G[r.cls], gap=E.gapOf(r), late=r.first-r.arr>g.ttft+EPS, ok=!late&&gap<=(g.gap!=null?g.gap:Infinity)+EPS;
    return {id:r.id,row:full.indexOf(sp[r.id]),cls:r.cls,own:r.own,arr:r.arr,first:r.first,fin:r.fin,ok,late,gap,seg};});
  const d=rows.find(r=>r.cls==='doc');
  /* the chats that were answering when the document arrived and then waited longer than the gap goal */
  const stalled=d?rows.filter(r=>r.cls==='chat'&&r.first<d.arr&&r.fin>d.arr&&r.gap>gg+EPS):[];
  return (TZ[k]={k,doc,n:sp.length,nChat:rows.filter(r=>r.cls==='chat').length,m,rows,stalled:stalled.map(r=>r.id),end:Math.max(...rows.map(r=>r.fin))});}
const tzDocSpec=()=>spec('stall').find(x=>x.cls==='doc');
const tzSpan=()=>Math.ceil(Math.max(...TZOPTS.map(k=>tzRun(k).end)));                 // whole seconds: the same axis for every option
const tzTicks=()=>{const st=tzDocSpec().t, out=[]; for(let v=0;v<=tzSpan()+EPS;v+=st) out.push(Math.round(v*1000)/1000); return out;};   // the document's arrival falls on a tick
const tzX=v=>0.4+v/tzSpan()*99.2;                       // % of the picture's width: x keeps its scale at any tile width
function tzSVG(t){
  const full=spec('stall'), di=full.findIndex(x=>x.cls==='doc'), n=full.length, p=t.doc?(TZH.rows-TZH.docH)/(n-1):TZH.rows/(n-1);
  const y0=i=>TZH.top+(t.doc?(i<=di?i*p:di*p+TZH.docH+(i-di-1)*p):(i<di?i*p:(i-1)*p)), hOf=i=>t.doc&&i===di?TZH.docH:p;
  const P=v=>tzX(v).toFixed(3)+'%', Wd=(a,b,min)=>Math.max(min,tzX(b)-tzX(a)).toFixed(3)+'%', f2=v=>(+v.toFixed(2)).toString();
  let s='';
  t.rows.forEach(r=>{const y=y0(r.row), h=hOf(r.row), cy=y+h/2, doc=r.cls==='doc', th=doc?2.6:1.7, tn=doc?1.1:0.8;
    s+=`<g class="${r.ok?'sc-tz-met':'sc-tz-miss'}" data-row="${r.row}" data-cls="${r.cls}" data-ok="${r.ok?1:0}">`+
      r.seg.map(([a,b,k])=>{const hh=k==='t'?th:tn; return `<rect class="sc-tz-${k}" x="${P(a)}" y="${f2(cy-hh/2)}" width="${Wd(a,b,k==='t'?0.35:0)}" height="${hh}"/>`;}).join('')+
      (doc?`<rect class="sc-tz-box" x="${(tzX(r.arr)-0.7).toFixed(3)}%" y="${f2(y+0.5)}" width="${(tzX(r.fin)-tzX(r.arr)+1.4).toFixed(3)}%" height="${f2(h-1)}" rx="1.5"/>`:'')+`</g>`;});
  const d=t.rows.find(r=>r.cls==='doc');
  if(d){const near=t.rows.filter(r=>Math.abs(r.row-d.row)<=2), x=Math.max(tzX(d.fin)+0.7,...near.map(r=>tzX(r.fin)));   // right of every row the label's height covers
    s+=`<text class="sc-tz-lab" x="${x.toFixed(3)}%" dx="5" y="${f2(y0(d.row)+TZH.docH/2+3.5)}">${esc(tx(`◂ ${tokK(d.own)} document`,`◂ ${tokK(d.own)} 的文档`))}</text>`;}
  /* the rows' caption, right-aligned in the top corner the staircase leaves empty (the early requests are done by then);
     the tests check it clears every row at a 280 px tile */
  s+=tzCap().map((c,i)=>`<text class="sc-tz-cap" x="100%" y="${TZH.top+8+i*11}" text-anchor="end">${esc(c)}</text>`).join('');
  const yb=TZH.top+TZH.rows+2, tk=tzTicks();
  s+=`<g class="sc-tz-ax" aria-hidden="true"><rect class="sc-tz-base" x="0" y="${yb}" width="100%" height="0.8"/>`+
    tk.map((v,i)=>`<rect class="sc-tz-tk" x="${P(v)}" y="${yb}" width="1" height="4"/><text x="${P(v)}" y="${yb+TZH.ax-1}" text-anchor="${i?'middle':'start'}">${v%1?fixd(v,1):String(v)}${i===tk.length-1?NB+'s':''}</text>`).join('')+`</g>`;
  return `<svg class="sc-tz-svg" width="100%" height="${yb+TZH.ax+1}" role="img" aria-label="${esc(tzAria(t))}">${s}</svg>`;}
const tzCap=()=>[tx('one row per request,','每行一个请求，'),tx('in arrival order','按到达顺序排列')];
/* the picture's description: what it shows, never why (R3.8) */
function tzAria(t){const m=t.m, ok=m.all.ok, n=t.n, span=tzSpan(), gg=G.chat.gap, d=tzDocSpec(), at=fixd(d.t,1)+NB+'s', K=tokK(d.own), calm=m.chat.gap_max<=gg+EPS;
  if(!t.doc) return tx(`Timeline of the ${n} short chats, one row each in arrival order, from 0 to ${span} s, without the document. ${calm?`No chat waits more than ${fS(gg)} between two tokens`:`The longest wait between two tokens is ${fS(m.chat.gap_max)}`}, and ${ok===n?`all ${n}`:`${ok} of the ${n}`} meet their goal.`,
                       `${n} 个短对话的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 ${span} s，没有那份文档。${calm?`没有一个对话在两个 token 之间等超过 ${fS(gg)}`:`两个 token 之间最长等了 ${fS(m.chat.gap_max)}`}，${ok===n?`${n} 个全部达标`:`${n} 个里 ${ok} 个达标`}。`);
  if(t.k==='sent') return tx(`Timeline of the ${n} requests, one row each in arrival order, from 0 to ${span} s. When the ${K}-token document arrives at ${at}, the ${t.stalled.length} chats that are answering get no token for about ${fR(m.chat.gap_max)}; ${n-ok} of the ${n} requests miss their goal.`,
                             `${n} 个请求的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 ${span} s。${at} 时那份 ${K} 个 token 的文档到达，正在回答的 ${t.stalled.length} 个对话约 ${fR(m.chat.gap_max)} 拿不到 token；${n} 个请求里 ${n-ok} 个没达标。`);
  const miss=t.rows.filter(r=>!r.ok), lateChats=miss.length>0&&miss.every(r=>r.cls==='chat'&&r.late);
  const mEN=ok===n?`all ${n} requests meet their goal`:`${n-ok} of the ${n} requests miss their goal`+(lateChats?`, ${miss.length===1?'a chat':miss.length===2?'both chats':'all chats'} whose first token comes more than ${fS(G.chat.ttft)} after ${miss.length===1?'it arrives':'they arrive'}`:'');
  const mZH=ok===n?`${n} 个请求全部达标`:`${n} 个请求里 ${n-ok} 个没达标`+(lateChats?`，都是到达后超过 ${fS(G.chat.ttft)} 才出首 token 的对话`:'');
  return tx(`Timeline of the ${n} requests, one row each in arrival order, from 0 to ${span} s, with one scheduler setting changed. When the ${K}-token document arrives at ${at}, ${calm?`no chat waits more than ${fS(gg)} between two tokens`:`the longest wait between two tokens is ${fS(m.chat.gap_max)}`}; ${mEN}.`,
            `${n} 个请求的时间线，每个请求一行，按到达顺序排列，横轴从 0 到 ${span} s，调度器改了一项设置。${at} 时那份 ${K} 个 token 的文档到达，${calm?`没有一个对话在两个 token 之间等超过 ${fS(gg)}`:`两个 token 之间最长等了 ${fS(m.chat.gap_max)}`}；${mZH}。`);}
/* the sentence (R6.8, R11.5): hardware, model, load and the result, at most 25 English words; it never names the setting */
function tzRead(t){const ok=t.m.all.ok, n=t.n, d=tzDocSpec(), K=tokK(d.own), at=fixd(d.t,1)+NB+'s', low=ok/n<E.TARGET-EPS;
  const en=ok===n?`all <b>${n}</b> requests meet their goal`:`${low?'only ':''}<b>${ok} of ${n}</b> requests meet their goal`, zh=ok===n?`${n} 个请求<b>全部达标</b>`:`${n} 个请求里${low?'只有':''} <b>${ok}</b> 个达标`;
  if(t.k==='sent') return tx(`When one ${GPUN} serves ${MODEL} to ${t.nChat} short chats and ${aN(d.own)} ${K}‑token document at ${at}, ${en}.`,`一张 ${GPUN} 用 ${MODEL} 服务 ${t.nChat} 个短对话，${at} 时又来了一份 ${K} 个 token 的文档：${zh}。`);
  if(t.k==='fix') return tx(`When one ${GPUN} serves ${MODEL} to the same ${t.nChat} chats and document with one scheduler setting changed, ${en}.`,`一张 ${GPUN} 用 ${MODEL} 服务同样的 ${t.nChat} 个短对话和那份文档，调度器改了一项设置：${zh}。`);
  return tx(`When one ${GPUN} serves ${MODEL} to the same ${t.nChat} short chats and no document, ${en}.`,`一张 ${GPUN} 用 ${MODEL} 服务同样的 ${t.nChat} 个短对话，没有那份文档：${zh}。`);}
/* the notes behind “Assumptions & sources”: four short sentences, plain text (R11.5, review home-v4 A #7, B-18) */
function tzNotes(){const gg=G.chat.gap;
  return [tx(`Goals: a chat’s first token within ${fS(G.chat.ttft)} of arriving and no gap over ${fS(gg)} between tokens; the document’s first token within ${fS(G.doc.ttft)}; ${pct0(E.TARGET)} of requests meeting theirs.`,
             `目标：对话到达后 ${fS(G.chat.ttft)} 内出首 token，两个 token 之间的间隔不超过 ${fS(gg)}；文档到达后 ${fS(G.doc.ttft)} 内出首 token；至少 ${pct0(E.TARGET)} 的请求各自达标。`),
    tx(`The lab’s seeded toy run: one ${GPUN} serves ${MODEL} in BF16, and each step lasts the longer of its compute (${pct0(C.FS/989.5e12)} of the dense BF16 peak) and its memory reads (${pct0(C.BS/3.35e12)} of 3.35 TB/s).`,
       `这是 Lab 用固定随机种子的玩具实跑：一张 ${GPUN} 用 BF16 跑 ${MODEL}，每一步的耗时取计算（按稠密 BF16 峰值的 ${pct0(C.FS/989.5e12)}）和读显存（按 3.35 TB/s 的 ${pct0(C.BS/3.35e12)}）里较长的那个。`),
    tx('The first two options use the lab’s initial setup; the third changes one scheduler setting, as the lab’s guided parts end up doing.',
       '前两个选项用 Lab 的初始设置；第三个只改了调度器的一项设置，和 Lab 引导部分最后的设置一样。'),
    tx(`Sources: NVIDIA’s H100 page (BF16 1,979 TFLOP/s with sparsity, so 989.5 dense; 3.35 TB/s) and ${MODEL}’s config.json.`,
       `来源：NVIDIA 的 H100 产品页（BF16 1,979 TFLOP/s 是含稀疏的值，稠密取一半，989.5 TFLOP/s；带宽 3.35 TB/s）和 ${MODEL} 的 config.json。`)];}
function tzLegend(){const gg=fS(G.chat.gap), d=tzDocSpec(), at=fixd(d.t,1)+NB+'s', K=tokK(d.own);
  const it=(c,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sc-tz-sw sc-tz-sw-${c}" aria-hidden="true"></i>${esc(t)}</span>`;
  return `<div class="tz-leg sc-tz-leg">`+
    it('w',tx('thin: waiting for a token','细线：在等 token'),tx(`A thin line: the request is waiting, from its arrival to its first token, or for a next token that comes more than ${gg} after the one before.`,`细线：请求在等，从到达一直等到首 token，或者下一个 token 比上一个晚了 ${gg} 以上。`))+
    it('t',tx('thick: getting tokens','粗线：在出 token'),tx(`A thick line: the request gets its answer one token after another, never more than ${gg} apart.`,`粗线：请求一个接一个地拿到回答的 token，间隔都不超过 ${gg}。`))+
    it('miss',tx('red: goal missed','红色：没达标'),tx(`Red: this request misses its goal. A short chat misses it when its first token comes more than ${fS(G.chat.ttft)} after it arrives, or when it waits more than ${gg} between two tokens.`,`红色：这个请求没达标。短对话到达后超过 ${fS(G.chat.ttft)} 才出首 token，或者两个 token 之间等了超过 ${gg}，就没达标。`))+
    it('met',tx('grey: goal met','灰色：达标'),tx('Grey: this request meets its goal.','灰色：这个请求达标了。'))+
    it('box',tx('box: document','方框：那份文档'),tx(`The box marks the row of the ${K}-token document that a user sends at ${at} to be summarized.`,`方框标出那份 ${K} 个 token 的文档那一行：一个用户在 ${at} 时发来它，要写摘要。`))+`</div>`;}
A.registerTeaser(ID,{fid:'toy',render(el){
  const d=tzDocSpec(), at=fixd(d.t,1)+NB+'s', lab={none:tx('not sent','没来'),sent:tx(`sent at ${at}`,`${at} 时来了`),fix:tx('sent, retuned','来了，调过调度器')};
  el.innerHTML=`<div class="tz sc-tz">${A.ui.seg({key:'sc-tz',label:tx(`The ${tokK(d.own)} document`,`那份 ${tokK(d.own)} 的文档`),options:TZOPTS.map(k=>[k,lab[k]]),value:tzOpt,inline:false})}`+
    `<div class="tz-vis sc-tz-vis"></div>${tzLegend()}<div class="sc-tz-rd"><p class="tz-read"></p>${TZOPTS.slice(1).map(()=>'<p class="sc-tz-twin" aria-hidden="true"></p>').join('')}</div>`+
    tzNotes().map(x=>`<p class="tz-note">${esc(x)}</p>`).join('')+`</div>`;
  const seg=el.querySelector('.seg-ctl[data-key="sc-tz"]'), vis=el.querySelector('.sc-tz-vis'), rd=el.querySelector('.tz-read'), tw=[...el.querySelectorAll('.sc-tz-twin')];
  /* the other options' sentences sit invisibly in the same grid cell, so the tile keeps its height when the option changes */
  const upd=()=>{const t=tzRun(tzOpt); A.ui.setPressed(seg,tzOpt); vis.innerHTML=tzSVG(t); rd.innerHTML=tzRead(t); TZOPTS.filter(k=>k!==tzOpt).forEach((k,i)=>{tw[i].innerHTML=tzRead(tzRun(k));});};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||!seg.contains(b)||!TZOPTS.includes(b.dataset.v)||b.dataset.v===tzOpt) return;
    tzOpt=b.dataset.v; upd(); A.ui.flash(rd); A.announce(rd.textContent);});
  upd();
}});
/* read-only hook for the tile's tests */
window.__scTz=Object.freeze({
  get state(){return {opt:tzOpt};},
  opts:TZOPTS.slice(), fix:()=>({...tzFix()}),
  run:k=>JSON.parse(JSON.stringify(tzRun(k))),
  span:()=>tzSpan(), ticks:()=>tzTicks(), geo:Object.freeze({...TZH}),
  read:k=>tzRead(tzRun(k)), aria:k=>tzAria(tzRun(k)), notes:()=>tzNotes(), cap:()=>tzCap()
});

/* =====================================================================
   read-only hook for the tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
const hook={
  get state(){return {part:step,partId:STEPS[step].id,mode,load:S.load,cfg:cfgOf(S),settings:{...S},playhead:play.t,playing:play.playing,selected:sel,pred,speed:spd(),speedPicked:speed,runTok,seen:[...seen],chCfg:{...chCfg},chRunKey,chSubmitted,chReveal,chTier:chTier?{...chTier}:null,stageBusy:stageT!==0};},
  get run(){return deep(cur().m);},
  get base(){return deep(baseOf(S.load).m);},
  engine:E, loads:LOADS.slice(),
  view(load,s){const o=getRun(load,{...BASE,...s}); return {key:o.key,end:o.end,nSteps:o.st.length,worst:o.worst?{...o.worst}:null,ok:o.R.map(r=>r.ok),why:o.why.slice(),kinds:o.kinds.slice(),evAt:[...o.evAt.entries()],diag:diagnose(o).kind};},
  memAt(load,s,k){const o=getRun(load,{...BASE,...s}), d=memAt(o,k); return deep({own:d.own,inuse:d.inuse,users:d.users,mem:d.mem,pres:d.pres});},
  memPre(load,s,k){const o=getRun(load,{...BASE,...s}), d=memPre(o,k); return deep({own:d.own,inuse:d.inuse,users:d.users,mem:d.mem,pres:d.pres});},
  peakPre(load,s){const o=getRun(load,{...BASE,...s}), k=peakStepPre(o); return {k,used:memPre(o,k).mem.used,peak:o.run.peak,t1:o.st[k].t1};},
  pfGeo(){const g=tlGeo, cv=$('#sc-tl',app), o=cur(), d=o&&o.docs.find(r=>r.chunks.length), pl=$('#sc-pflab',app); if(!g||!cv||!d||!pl) return null; const c=cv.getBoundingClientRect(), l=pl.getBoundingClientRect();
    return {hidden:pl.hidden, text:pl.innerText, lab:[l.left,l.top,l.right,l.bottom], rowsTop:c.top+g.top, docTop:c.top+g.top+d.id*g.rh, docBottom:c.top+g.top+(d.id+1)*g.rh, xa:c.left+g.X(d.arr)};},
  rowBox(id){const g=tlGeo, cv=$('#sc-tl',app), o=cur(), r=o&&o.R[id]; if(!g||!cv||!r) return null; const c=cv.getBoundingClientRect(), y=c.top+g.top+id*g.rh;
    return {left:c.left+g.X(r.arr),right:c.left+g.X(Math.min(r.fin,g.span)),top:y,bottom:y+g.rh,admitX:c.left+g.X(r.admit)};},
  exitsText(){const x=EXITS(), q=[...x.explain,...x.transfer]; return q.map(v=>[v.q,...v.o,v.why,...(v.n||[]).filter(Boolean)].join('\n')).join('\n');},
  gpuTime(load,s){const o=getRun(load,{...BASE,...s}); return {sum:o.gpuT,t0:o.t0r,end:o.end};},
  waits(load,s,id){return deep(waitSegs(getRun(load,{...BASE,...s}),getRun(load,{...BASE,...s}).R[id]));},
  goalAt(load,s,id,t){const o=getRun(load,{...BASE,...s}); return goalAt(o.R[id],t);},
  xrun(name){const o=xRun(name); return {m:deep(o.m),worst:o.worst?{...o.worst}:null};},
  drill(seed){const d=drillCase(seed); return {U:d.U,c:d.c,N:d.N,B:d.B,a:d.a.worst.g,b:d.b.worst.g};},
  plan(){const o=chRunKey&&RC.get(chRunKey); return o?planText(o,baseOf('challenge')):null;},
  moments(){return MOMENTS.map(m=>{const {o,s}=momentStep(m); return {id:m.id,part:m.part,at:m.at||'run',cfg:{...m.cfg},key:o.key,s,t0:o.st[s].t0,t1:o.st[s].t1};});},
  codeInfo(load,s,k){const o=getRun(load,{...BASE,...s}), c=codeInfo(o,k); return deep({...c,st:{t0:c.st.t0,t1:c.st.t1,tc:c.st.tc,tm:c.st.tm,T:c.st.T,nrun:c.st.nrun},b0:isFinite(c.b0)?c.b0:null,bDec:isFinite(c.bDec)?c.bDec:null,bPre:isFinite(c.bPre)?c.bPre:null,bEnd:isFinite(c.bEnd)?c.bEnd:null});},
  codeLines(load,s,k,phase){const o=getRun(load,{...BASE,...s}); return deep(codeLines(o,codeInfo(o,k),phase||'end'));},
  codeStep(){const o=cur(), c=codeStepAt(o); return c?{...c}:null;},
  codeObs(id){const was=codeM; codeM=id; const h=obsCode(); codeM=was; const d=document.createElement('div'); d.innerHTML=h; return d.textContent;},
  endView(){return endView(cur());},
  wcode:WCODE.map(x=>x.slice()), code:CODE.map(x=>x.slice()),
  get codeM(){return codeM;}, get codeSeen(){return [...codeSeen];},
  consts:{BASE:{...BASE},CAPS:CAPS.slice(),BUDS:BUDS.slice(),CBS:CBS.slice(),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,SPEEDS:SPEEDS.slice(),BASE_SECS,PART_CFG:deep(PART_CFG)}
};
Object.freeze(hook.consts);
window.__sc=Object.freeze(hook);
})();
