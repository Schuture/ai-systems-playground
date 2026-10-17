/* PagedAttention lab: why most KV cache memory was wasted, and how fixed-size blocks fix it.
   A seeded simulation of 40 requests sharing one KV cache pool, under three allocation strategies. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='paged-attention';

/* ================= engine: pure functions, seeded and deterministic ================= */
const POOL=12288, CELL=16, NCELL=POOL/CELL, NREQ=40, PMIN=100, PMAX=600, SEED=7, SWEEP_N=200;
const BLOCKS=[1,8,16,32,64,128,256], MEANS=[64,128,256,512,1024], LIMITS=[512,1024,1536,2048,4096,8192];
const STRATS=['max','exact','paged'];
const DEFAULT={strat:'max',B:16,mu:256,lim:1536};
const CH_BASE={mu:256,lim:1536}, GOAL_UTIL=0.9, GOAL_TABLE=64;
/* figures quoted from the vLLM paper (Kwon et al., 2023, Figure 2), shown for comparison only */
const PAPER={orcaMax:0.204,orcaOracle:0.382,vllm:0.963};
const FREE=0, TOK=1, RESV=2, INTF=3, EXT=4;

function rng(seed){let a=seed>>>0; return ()=>{a=(a+0x6D2B79F5)>>>0; let t=a; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296;};}
/* 40 requests: prompt uniform 100..600 tokens; output exponential with mean mu, rounded, clamped to [1, lim] */
function workload(mu,lim,seed){
  const r=rng(seed==null?SEED:seed), out=[];
  for(let i=0;i<NREQ;i++){const p=PMIN+Math.floor(r()*(PMAX-PMIN+1)), u=1-r(), o=Math.max(1,Math.min(lim,Math.round(-mu*Math.log(u)))); out.push({i,p,o,F:p+o});}
  return out;
}
function newAgg(){return {tok:[],resv:[],intf:[],ext:[],free:[],run:[],wait:[],done:[]};}
/* contiguous: each request gets one region (prompt + limit, or its exact final length), first fit, FCFS, no preemption */
function simContig(reqs,exact,lim){
  const n=reqs.length, R=reqs.map(q=>exact?q.F:q.p+lim);
  const a=new Int32Array(n).fill(-1), f=new Int32Array(n).fill(-1), s=new Int32Array(n).fill(-1), len=new Int32Array(n);
  const gaps=[[0,POOL]], run=[], g=newAgg();
  let next=0, used=0, done=0, t=0;
  const release=(st,ln)=>{let k=0; while(k<gaps.length&&gaps[k][0]<st) k++; gaps.splice(k,0,[st,ln]);
    if(k+1<gaps.length&&gaps[k][0]+gaps[k][1]===gaps[k+1][0]){gaps[k][1]+=gaps[k+1][1]; gaps.splice(k+1,1);}
    if(k>0&&gaps[k-1][0]+gaps[k-1][1]===gaps[k][0]){gaps[k-1][1]+=gaps[k][1]; gaps.splice(k,1);}};
  while(next<n||run.length){
    while(next<n){const need=R[next], k=gaps.findIndex(x=>x[1]>=need); if(k<0) break;
      const gp=gaps[k]; s[next]=gp[0]; if(gp[1]===need) gaps.splice(k,1); else {gp[0]+=need; gp[1]-=need;}
      a[next]=t; len[next]=reqs[next].p; used+=need; run.push(next); next++;}
    let tok=0,resv=0,intf=0;
    for(const i of run){len[i]++; tok+=len[i]; resv+=reqs[i].F-len[i]; intf+=R[i]-reqs[i].F;}
    const freeAll=POOL-used, wait=n-next, ext=wait>0?freeAll:0;
    g.tok.push(tok); g.resv.push(resv); g.intf.push(intf); g.ext.push(ext); g.free.push(freeAll-ext); g.run.push(run.length); g.wait.push(wait); g.done.push(done);
    for(let k=run.length-1;k>=0;k--){const i=run[k]; if(len[i]>=reqs[i].F){f[i]=t; used-=R[i]; release(s[i],R[i]); run.splice(k,1); done++;}}
    t++; if(t>200000) break;
  }
  return {paged:false,exact,reqs,R,a,f,start:s,agg:g,T:t,pre:0};
}
/* paged: B-token blocks from a free list; running requests grow first (preempting the newest when the pool is empty),
   waiting requests are admitted only in steps without a preemption and need blocks for prompt + generated + 1.
   The free list is plain first in, first out: a simplification. vLLM's free queue (vllm/v1/core/block_pool.py, free_blocks)
   puts freed blocks that carry a prefix-cache hash at the back and all others at the front, to be reused first.
   Which physical block a request gets only moves it in the picture; every number here depends on block counts alone. */
function simPaged(reqs,B){
  const n=reqs.length, NB=POOL/B, ring=new Int32Array(NB);
  for(let k=0;k<NB;k++) ring[k]=k;
  let head=0, cnt=NB;
  const pop=()=>{const id=ring[head]; head=(head+1)%NB; cnt--; return id;};
  const push=id=>{ring[(head+cnt)%NB]=id; cnt++;};
  const queue=[], run=[], gen=new Int32Array(n), len=new Int32Array(n), f=new Int32Array(n).fill(-1);
  const tbl=Array.from({length:n},()=>[]), stints=Array.from({length:n},()=>[]), g=newAgg();
  for(let i=0;i<n;i++) queue.push(i);
  const release=i=>{const b=tbl[i]; for(let k=b.length-1;k>=0;k--) push(b[k]); tbl[i]=[];};
  let pre=0, done=0, t=0;
  while(queue.length||run.length){
    let preempted=false;
    for(let k=0;k<run.length;){
      const i=run[k];
      if(len[i]+1>tbl[i].length*B){
        let self=false;
        while(cnt===0){const v=run.pop(); release(v); stints[v][stints[v].length-1].e=t-1; queue.unshift(v); pre++; preempted=true; if(v===i){self=true; break;}}
        if(self) continue;
        const id=pop(), st=stints[i][stints[i].length-1]; tbl[i].push(id); st.ids.push(id); st.at.push(t);
      }
      k++;
    }
    if(!preempted){
      while(queue.length){const i=queue[0], have=reqs[i].p+gen[i], need=Math.ceil((have+1)/B); if(need>cnt) break;
        const ids=[], at=[]; for(let k=0;k<need;k++){ids.push(pop()); at.push(t);}
        tbl[i]=ids.slice(); len[i]=have; stints[i].push({a:t,e:-1,len0:have,ids,at}); run.push(i); queue.shift();}
    }
    let tok=0,resv=0,intf=0,cap=0;
    for(const i of run){len[i]++; gen[i]++; const c=tbl[i].length*B, F=reqs[i].F; tok+=len[i]; cap+=c; resv+=Math.min(F,c)-len[i]; intf+=Math.max(0,c-F);}
    g.tok.push(tok); g.resv.push(resv); g.intf.push(intf); g.ext.push(0); g.free.push(POOL-cap); g.run.push(run.length); g.wait.push(queue.length); g.done.push(done);
    for(let k=run.length-1;k>=0;k--){const i=run[k]; if(gen[i]>=reqs[i].o){release(i); stints[i][stints[i].length-1].e=t; f[i]=t; run.splice(k,1); done++;}}
    t++; if(t>200000) break;
  }
  return {paged:true,B,reqs,stints,f,agg:g,T:t,pre};
}
const KEYS5=['tok','resv','intf','ext','free'];
function finishRun(r,s){
  const g=r.agg, T=r.T;
  Object.keys(g).forEach(k=>{g[k]=Int32Array.from(g[k]);});
  let busy=[]; for(let t=0;t<T;t++) if(g.wait[t]>0) busy.push(t);
  r.anyWait=busy.length>0; r.lastWait=busy.length?busy[busy.length-1]:-1;
  if(!busy.length) busy=Array.from({length:T},(_,t)=>t);
  const avg=k=>busy.reduce((acc,t)=>acc+g[k][t],0)/busy.length;
  r.cfg={...s}; r.busyN=busy.length;
  r.shares={}; KEYS5.forEach(k=>{r.shares[k]=avg(k)/POOL;});
  r.util=r.shares.tok; r.tokAvg=avg('tok'); r.running=avg('run');
  let ts=0, tm=0; for(let t=0;t<T;t++){ts+=g.tok[t]; if(g.tok[t]>tm) tm=g.tok[t];}
  r.tokSum=ts; r.tokMax=tm;                                  // tokens read per step: for the bytes-read timing check
  const q=r.reqs, sumO=q.reduce((x,y)=>x+y.o,0);
  r.sumO=sumO; r.tps=sumO/T; r.meanF=q.reduce((x,y)=>x+y.F,0)/q.length; r.meanO=sumO/q.length;
  r.capped=q.filter(x=>x.o===s.lim).length;
  if(r.paged){r.unusedEach=q.map(x=>Math.ceil(x.F/s.B)*s.B-x.F); r.table=q.reduce((x,y)=>x+Math.ceil(y.F/s.B),0)/q.length;}
  else {r.unusedEach=q.map((x,i)=>r.R[i]-x.F); r.table=null;}
  r.unused=r.unusedEach.reduce((x,y)=>x+y,0)/q.length;
  // a typical moment: the waiting step whose mix of states is closest to the average mix
  let best=busy[0], bd=Infinity;
  busy.forEach(t=>{let d=0; KEYS5.forEach(k=>{d+=Math.abs(g[k][t]/POOL-r.shares[k]);}); if(d<bd-1e-12){bd=d; best=t;}});
  r.showcase=best;
  return r;
}
const CACHE=new Map();
const runKey=s=>[s.strat,s.strat==='paged'?s.B:0,s.mu,s.lim].join('.');
function simulate(s){
  const key=runKey(s); let r=CACHE.get(key); if(r) return r;
  const reqs=workload(s.mu,s.lim,SEED);
  r=s.strat==='paged'?simPaged(reqs,s.B):simContig(reqs,s.strat==='exact',s.lim);
  r=finishRun(r,s); CACHE.set(key,r);
  if(CACHE.size>80) CACHE.delete(CACHE.keys().next().value);   // runs are deterministic: an evicted one is simply recomputed
  return r;
}
/* how much the step ratio depends on the random draw: the guided setup with seeds 1..200, computed on demand in slices */
let sweepR=[], SWEEP=null, sweepT=0;
function sweepMore(n){
  while(!SWEEP&&n-->0){const sd=sweepR.length+1;
    const a=simContig(workload(DEFAULT.mu,DEFAULT.lim,sd),false,DEFAULT.lim).T, b=simPaged(workload(DEFAULT.mu,DEFAULT.lim,sd),16).T;
    sweepR.push(a/b);
    if(sweepR.length===SWEEP_N){const srt=[...sweepR].sort((x,y)=>x-y), me=sweepR[SEED-1];
      SWEEP={n:SWEEP_N,ratio:me,median:(srt[SWEEP_N/2-1]+srt[SWEEP_N/2])/2,below:sweepR.filter(x=>x<me).length};}}
  return SWEEP;
}
const sweep=()=>SWEEP||sweepMore(Infinity);
function sweepWarm(){if(SWEEP||sweepT) return; const go=()=>{sweepT=0; if(!sweepMore(8)) sweepT=setTimeout(go,40);}; sweepT=setTimeout(go,40);}
/* the pool at step t, slot by slot: state (FREE/TOK/RESV/INTF/EXT) and owner request */
const SL=new Uint8Array(POOL), OW=new Int16Array(POOL);
function stintAt(r,i,t){const a=r.stints[i]; for(let k=0;k<a.length;k++){if(a[k].a<=t&&t<=a[k].e) return a[k];} return null;}
function fillSlots(r,t){
  SL.fill(FREE); OW.fill(-1);
  const q=r.reqs;
  if(r.paged){const B=r.B;
    for(let i=0;i<q.length;i++){const st=stintAt(r,i,t); if(!st) continue;
      const ln=st.len0+(t-st.a+1), F=q[i].F; let nb=0; while(nb<st.at.length&&st.at[nb]<=t) nb++;
      for(let j=0,J=nb*B;j<J;j++){const sl=st.ids[(j/B)|0]*B+j%B; SL[sl]=j<ln?TOK:(j<F?RESV:INTF); OW[sl]=i;}}
  } else {
    for(let i=0;i<q.length;i++){if(r.a[i]<0||t<r.a[i]||t>r.f[i]) continue;
      const ln=q[i].p+(t-r.a[i]+1), F=q[i].F, s0=r.start[i];
      for(let j=0;j<r.R[i];j++){SL[s0+j]=j<ln?TOK:(j<F?RESV:INTF); OW[s0+j]=i;}}
    if(r.agg.wait[t]>0) for(let k=0;k<POOL;k++) if(SL[k]===FREE) SL[k]=EXT;
  }
}
/* per-request details at step t, for tooltips and examples */
function reqAt(r,i,t){
  const q=r.reqs[i];
  if(r.paged){const st=stintAt(r,i,t); if(!st) return null; let nb=0; while(nb<st.at.length&&st.at[nb]<=t) nb++;
    return {i,p:q.p,o:q.o,F:q.F,len:st.len0+(t-st.a+1),nb,ids:st.ids.slice(0,nb),B:r.B};}
  if(r.a[i]<0||t<r.a[i]||t>r.f[i]) return null;
  return {i,p:q.p,o:q.o,F:q.F,len:q.p+(t-r.a[i]+1),R:r.R[i],s:r.start[i]};
}
const headAt=(r,t)=>{if(r.paged||!r.agg.wait[t]) return -1; for(let i=0;i<r.reqs.length;i++) if(r.a[i]>t) return i; return -1;};
function gapsAt(r,t){fillSlots(r,t); const gs=[]; let st=-1; for(let k=0;k<=POOL;k++){const fr=k<POOL&&(SL[k]===FREE||SL[k]===EXT); if(fr&&st<0) st=k; if(!fr&&st>=0){gs.push([st,k-st]); st=-1;}} return gs;}
/* the challenge: every allowed choice, judged on the same 40 requests */
const CH_OPTS=[{strat:'max'},{strat:'exact'}].concat(BLOCKS.map(B=>({strat:'paged',B})));
function chEval(o){const r=simulate({...CH_BASE,B:16,...o}), B=o.strat==='paged'?o.B:16;
  const tableOk=r.table==null||r.table<=GOAL_TABLE+1e-9, utilOk=r.util>=GOAL_UTIL-1e-12;
  return {o,r,util:r.util,table:r.table,utilOk,tableOk,met:utilOk&&tableOk,
    changes:(o.strat!=='max'?1:0)+(B!==16?1:0),dist:Math.abs(Math.log2(B/16))};}
function chBest(){return CH_OPTS.map(chEval).filter(x=>x.met).sort((x,y)=>x.changes-y.changes||x.dist-y.dist||y.util-x.util)[0];}

/* ================= formatting ================= */
const int=n=>A.fmt.int(n), pc=(x,d)=>A.fmt.num(x*100,d==null?1:d)+'%';
/* one decimal, thousands separators, a trailing .0 dropped: 1232.5 -> 1,232.5 */
const n1=v=>{const s=A.fmt.num(v,1), k=s.indexOf('.'); return k<0?int(+s):int(+s.slice(0,k))+s.slice(k);};
const f2=v=>A.fmt.num(v,2);
/* differences and ratios are worked out from the values as displayed, so the text reproduces from the page */
const shown=x=>Math.round(x*1000)/10, shown1=x=>Math.round(x*10)/10;
const kv8b=()=>A.kv.compute({...A.kv.DEFAULT,model:'llama31-8b',gpu:'rtx4090',n:1,ctx:1,users:1,wp:'bf16',kp:'bf16',attn:'gqa'});
const poolBytes=()=>kv8b().kvPerToken*POOL;
const STRAT_NAME={max:{en:'Reserve the maximum',zh:'按最大长度预留'},exact:{en:'Reserve the exact length',zh:'按实际长度预留'},paged:{en:'Paging',zh:'分页'}};
const cfgName=s=>s.strat==='paged'?tx(`Paging, ${s.B}-token blocks`,`分页，${s.B} 个 token 一个 block`):L(STRAT_NAME[s.strat]);
const reqName=i=>tx(`request ${i+1}`,`请求 ${i+1}`);
const stepsTxt=n=>tx(n===1?'1 decode step':`${int(n)} decode steps`,`${int(n)} 个 decode 步`);
const reqsTxt=(n,w)=>tx(`${n} ${n===1?'request':'requests'}${w?' '+w:''}`,`${n} 个请求`);
/* numbers in prose (R6.20): an explanation says the trend and the reason in words, with at most one key number, rounded for
   reading (a share to the whole percent: "about 25%"). The exact values stay on the stage, the metric cards, the lever results,
   the tooltips and Show the math. Every word below is chosen from the engine's numbers, never typed for one case */
const pc0=x=>pc(x,0);
const NUM_EN={2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine'}, NUM_ZH={2:'两',3:'三',4:'四',5:'五',6:'六',7:'七',8:'八',9:'九'};
/* a ratio of two shown values as "more than five times as many" / "五倍多" (from 2 up; below that, "more" / "更多") */
function manyW(q){const f=Math.floor(q+1e-9), over=q>f+1e-9;
  if(f<2) return tx('more','更多');
  return tx(`${over?'over ':''}${f===2?'twice':`${NUM_EN[f]||int(f)} times`} as many`,`${NUM_ZH[f]||int(f)+' '}倍${over?'多':''}`);}
/* a part of the 40 requests in words when it is a simple fraction (10 of 40: a quarter); otherwise the count */
const FRAC={2:{en:'half',zh:'一半'},4:{en:'a quarter',zh:'四分之一'},5:{en:'a fifth',zh:'五分之一'},8:{en:'an eighth',zh:'八分之一'},10:{en:'a tenth',zh:'十分之一'}};
const partW=c=>{const f=NREQ%c===0&&FRAC[NREQ/c]; return f?tx(f.en,`${f.zh}的回答`):tx(`${c} of the ${NREQ}`,`${NREQ} 个里有 ${c} 个`);};
/* a strategy's decode steps next to reserving the maximum's (same requests), in words */
function stepsVsMaxW(T,Tm){const q=Tm/T;
  if(q<0.87) return tx('more decode steps than reserving the maximum','用的 decode 步比按最大长度预留时还多');
  if(q<1.15) return tx('about as many decode steps as reserving the maximum','用的 decode 步和按最大长度预留时差不多');
  if(q<1.8) return tx('fewer decode steps than reserving the maximum','用的 decode 步比按最大长度预留时少一些');
  const w=q<2.5?['about half','约一半']:(q<3.5?['about a third','约三分之一']:(q<4.5?['about a quarter','约四分之一']:(q<10?['a small fraction','一小部分']:['under a tenth','不到十分之一'])));
  return tx(`${w[0]} of the decode steps that reserving the maximum needs`,`用的 decode 步只有按最大长度预留时的${w[1]}`);}
/* part 6: a block size against the default 16, in words: the stored-token share (a change under 2 points "barely changes"),
   and the block table's length */
function shareMoveW(u,u16){const d=shown(u)-shown(u16);
  if(Math.abs(d)<2) return {small:true,t:tx('barely changes','几乎不变')};
  return {small:false,t:d>0?tx(`rises from about ${pc0(u16)} to about ${pc0(u)}`,`从约 ${pc0(u16)} 升到约 ${pc0(u)}`):tx(`drops from about ${pc0(u16)} to about ${pc0(u)}`,`从约 ${pc0(u16)} 降到约 ${pc0(u)}`)};}
function tableW(t,t16){const q=t/t16;
  if(q>=1.1){if(q<1.5) return tx('get a little longer','长了一点'); if(q<2.5) return tx('get about twice as long','长了约一倍'); return tx(`get about ${int(Math.round(q))} times as long`,`变成约 ${int(Math.round(q))} 倍长`);}
  if(q<=0.9){const s=1/q; if(s<1.5) return tx('get a little shorter','短了一点'); if(s<2.5) return tx('get about half as long','只有约一半长'); if(s<3.5) return tx('get about a third as long','只有约三分之一长'); if(s<4.5) return tx('get about a quarter as long','只有约四分之一长'); return tx('get much shorter','短了很多');}
  return tx('stay about as long','差不多长');}
/* the RTX 4090 estimate: the KV cache memory lab's model (90% of 24 GiB, minus BF16 weights and 2 GiB of overhead), and at vLLM's 0.92 default */
function gpuEst(){const k=kv8b(), at=u=>A.kv.compute({...k.s,util:u});
  return {k,t90:Math.round(k.maxCtx/1000)*1000,t92:Math.round(at(0.92).maxCtx/1000)*1000,util:k.s.util,oh:k.s.overhead};}

/* ================= strings for data-tip keys ================= */
A.addStrings({
 en:{pa_mUtil:'This is the share of the pool that stores tokens, averaged over the decode steps in which a request was waiting. Empty room only costs something while a request waits for it.',
  pa_mRun:'This is how many requests generate a token together in one decode step, averaged over the same decode steps. The more requests run at once, the fewer decode steps the same work takes.',
  pa_mUnused:'This is the room a request holds but never uses before it finishes, averaged over the 40 requests. With paging it is at most the unfilled part of the last block.',
  pa_mTable:'This is the number of entries in a request’s block table when it finishes, one per block, averaged over the 40 requests. Reserving by length gives each request a single region, so it needs no block table.',
  pa_status:'This badge shows the share of the pool that stores tokens, averaged over the decode steps in which requests wait. It is green at 90% or more, red below half, and gray in between.'},
 zh:{pa_mUtil:'这个数是显存池里存着 token 的比例，在有请求排队的 decode 步里取平均。排队时空着的空间才是真浪费：排队的请求本来可以用上它。',
  pa_mRun:'这个数是一个 decode 步里一起生成 token 的请求数，也在这些 decode 步里取平均。同时跑的请求越多，同样的活需要的 decode 步越少。',
  pa_mUnused:'这个数是每个请求占着、到结束也没用上的空间，按 40 个请求取平均。分页时它最多是最后一个 block 里没写满的部分。',
  pa_mTable:'这个数是请求结束时 block table 的条数，每个 block 一条，按 40 个请求取平均。按长度预留时每个请求只有一整块，不需要 block table。',
  pa_status:'这个标记显示有请求排队时，显存池平均有多大比例存着 token。90% 以上是绿色，不到一半是红色，其余是灰色。'}
});

/* ================= state (module scope: survives re-mounts and language switches) ================= */
let S={...DEFAULT}, t=0, mode='guided', step=0, pred=null, seen=false, observed={}, hl='all', speed='med', sameNote=null, oneOff=0, userAct=false;
const hlTried=new Set(['all']), stTried=new Set(['start']), bTried=new Set([16]);
let chSubmitted=false, chReveal=false, savedS=null, drillK=0, drillOpen=false, mounted=false;
/* R3.20: the runs the reader has taken to their last decode step (Play, the slider, +1 decode step, Show me), in any part or mode, by
   their settings (runKey), not by part; part 3 shows a run already watched (in Scenarios, say) with its whole-run result, and its gate counts it */
const watched=new Set();
let root=null, host=null, playing=false, raf=0, lastTs=0, acc=0, prevRun=null, hoverOwner=-1, kbd=-1, geo=null, pats=null, themeObs=null, mq=null, mqFn=null;
/* playback speed in decode steps per second; the run advances by elapsed time, so the speed is the same on any display */
const SPEEDS={slow:25,med:100,fast:400};
const run=()=>simulate(S);
/* the answer to the prediction (the run's averages) stays hidden until part 3 shows the run */
const masked=()=>mode==='guided'&&step<=1;
/* part 3 before its run has reached the last decode step: the stage shows the live decode step, not the whole run */
const pending=()=>mode==='guided'&&step===2&&!observed[2];
const hidden=()=>masked()||pending();
const maskNote=()=>pending()?tx('shown after the run','跑完后显示'):(pred?tx('shown in part 3','第 3 部分显示'):tx('shown after you predict','预测后显示'));

/* ================= copy: states, levers, presets ================= */
const STATE_KEYS=['tok','resv','intf','ext','free'];
const STATE_NAME={tok:{en:'stored tokens',zh:'存着的 token'},resv:{en:'reserved for later tokens',zh:'为后面的 token 预留'},intf:{en:'internal fragmentation',zh:'内部碎片'},ext:{en:'external fragmentation',zh:'外部碎片'},free:{en:'free',zh:'空闲'}};
const LEGEND_NAME={tok:{en:'Stored tokens',zh:'存着的 token'},resv:{en:'Reserved for later tokens',zh:'为后面的 token 预留'},intf:{en:'Internal fragmentation',zh:'内部碎片'},ext:{en:'External fragmentation',zh:'外部碎片'},free:{en:'Free',zh:'空闲'}};
const STATE_TIP={
 tok:{en:'Stored tokens are the prompt and the tokens generated so far; their keys and values live here.',zh:'存着的 token 是 prompt 和已生成的 token，它们的 key 和 value 存在这里。'},
 resv:{en:'This is room the request will use later but that is empty now. Until then, no other request can use it.',zh:'这是请求之后会用到、现在空着的空间。在它用上之前，别的请求也用不了。'},
 intf:{en:'Internal fragmentation is room a request holds but never uses before it finishes. A real server cannot tell it from “reserved for later tokens”, because it does not know the final length.',zh:'内部碎片是请求占着、到结束也用不上的空间。真实服务时不知道请求多长，所以分不清它和“为后面的 token 预留”。'},
 ext:{en:'External fragmentation is room nobody holds while a request is waiting: the request at the head of the queue needs one contiguous piece, and no piece is big enough.',zh:'外部碎片是有请求在排队、却没人占用的空间：排在最前面的请求要一整段连续空间，而每一段都不够大。'},
 free:{en:'Free room is room nobody holds that does not count as waste: no request is waiting, or (with paging) the free blocks together are not enough for the request at the head of the queue.',zh:'空闲是没人占用、也不算浪费的空间：没有请求在排队，或者（分页时）空闲 block 加起来也不够排在最前面的请求用。'}
};
const HL_LEVERS=[
 {v:'all',t:{en:'All',zh:'全部'},d:{en:'stored tokens and the three kinds of waste',zh:'存着的 token 和三种浪费'}},
 {v:'resv',t:{en:'Reserved for later tokens',zh:'为后面的 token 预留'},d:{en:'the request will use it, but it is empty now',zh:'请求之后会用到，但现在空着'}},
 {v:'intf',t:{en:'Internal fragmentation',zh:'内部碎片'},d:{en:'held by a request that finishes before using it',zh:'请求占着，到结束也用不上'}},
 {v:'ext',t:{en:'External fragmentation',zh:'外部碎片'},d:{en:'held by nobody, but each piece is too small for the whole region the next request needs (the end of the pool included)',zh:'没人占用，但每一段都放不下下一个请求要的整块（包括显存池末尾）'}}
];
const ST_LEVERS=[
 {v:'start',s:{strat:'max'},t:{en:'Initial setup: reserve the maximum',zh:'初始设置：按最大长度预留'},d:()=>tx(`one region per request, room for its prompt plus ${int(DEFAULT.lim)} tokens`,`每个请求一整块，放得下 prompt 加 ${int(DEFAULT.lim)} 个 token`)},
 {v:'exact',s:{strat:'exact'},t:{en:'Reserve the exact length',zh:'按实际长度预留'},d:()=>tx('as if each answer’s length were known in advance, which a real server cannot know','假设事先知道每个回答多长，真实服务做不到')},
 {v:'paged',s:{strat:'paged',B:16},t:{en:'Paging: 16-token blocks',zh:'分页：16 个 token 一个 block'},d:()=>tx('the next block only when the last is full; vLLM’s default size','写满一个再分配下一个，vLLM 的默认大小')}
];
const PRESETS=[
 {t:{en:'Customer support',zh:'客服问答'},p:{mu:128,lim:1536}},
 {t:{en:'Chat service',zh:'聊天服务'},p:{mu:256,lim:1536},guided:true},
 {t:{en:'Long-form writing',zh:'长文写作'},p:{mu:1024,lim:2048}},
 {t:{en:'API with a large default cap',zh:'API 服务，默认上限很大'},p:{mu:128,lim:8192}},
 {t:{en:'Summaries',zh:'摘要'},p:{mu:256,lim:512}}
];
const presetDesc=pr=>{const r=simulate({...S,...pr.p}), c=r.capped, mu=int(pr.p.mu), lim=int(pr.p.lim);
  return tx(`answer lengths drawn at random with a mean of ${mu} tokens, capped at ${lim} tokens${pr.guided?' (the initial setup’s workload)':''}${c?`; ${partW(c)} reach the cap`:''}`,
            `回答长度按均值 ${mu} 个 token 随机抽取，上限 ${lim} 个 token${pr.guided?'（和初始设置的负载相同）':''}${c?`；${partW(c)}写到上限`:''}`);};
const presetNow=()=>PRESETS.findIndex(pr=>pr.p.mu===S.mu&&pr.p.lim===S.lim);
const stLeverNow=()=>{if(S.mu!==DEFAULT.mu||S.lim!==DEFAULT.lim) return -1; if(S.strat==='max') return 0; if(S.strat==='exact') return 1; return S.B===16?2:-1;};
const PRED=[{v:'lt30',lo:0,hi:0.3,t:{en:'Under 30%',zh:'不到 30%'}},{v:'30to60',lo:0.3,hi:0.6,t:{en:'30% to 60%',zh:'30% 到 60%'}},
 {v:'60to80',lo:0.6,hi:0.8,t:{en:'60% to 80%',zh:'60% 到 80%'}},{v:'gt80',lo:0.8,hi:1.01,t:{en:'Over 80%',zh:'80% 以上'}}];
/* the option that holds a share, as displayed */
const predOf=u=>{const x=shown(u)/100; return PRED.find(o=>x>=o.lo&&x<o.hi);};

const R0=()=>simulate(DEFAULT), RX=()=>simulate({...DEFAULT,strat:'exact'}), RP=B=>simulate({...DEFAULT,strat:'paged',B:B||16});
/* what if decode-step time grew with the bytes each decode step reads: the weights once, plus the running requests' KV cache.
   Llama-3.1-8B's vocabulary and hidden size come from Meta's llama-models (models/sku_list.py). Its embedding table,
   128,256 x 4,096, is only looked up during decode (one row per token), so a decode step reads every weight except that table. */
const VOCAB=128256, HIDDEN=4096;
function timing(){const k=kv8b(), emb=VOCAB*HIDDEN*k.wTotal/k.m.params, W=k.wTotal-emb, kt=k.kvPerToken, b=R0(), p=RP(16), cost=r=>r.T*W+kt*r.tokSum, top=Math.max(b.tokMax,p.tokMax);
  return {W,Wall:k.wTotal,emb,kvMax:top*kt,maxIsPaged:p.tokMax>=b.tokMax,poolB:POOL*kt,share:top*kt/W,steps:b.T/p.T,bytes:cost(b)/cost(p)};}
const lvResultSt=v=>{const lv=ST_LEVERS.find(x=>x.v===v), u=pc(simulate({...DEFAULT,...lv.s}).util); return tx(`${u} stores tokens while requests wait`,`排队时 ${u} 存着 token`);};
const lvResultHl=v=>v==='all'?tx(`${pc(R0().util)} stores tokens while requests wait`,`排队时 ${pc(R0().util)} 存着 token`):tx(`${pc(R0().shares[v])} of the pool while requests wait`,`排队时占显存池 ${pc(R0().shares[v])}`);
const maxO=r=>r.reqs.reduce((m,q)=>Math.max(m,q.o),0);
const J=(key,txt)=>`<button type="button" class="linkbtn" data-jump="${key}">${txt}</button>`;

/* inline controls (the one control a step asks for sits inside its card) */
const speedCtl=()=>`<div class="pa-speed" data-tipx="${esc(tx(`Slow plays ${SPEEDS.slow} decode steps per second, Medium ${SPEEDS.med} and Fast ${SPEEDS.fast}.`,`慢速每秒播放 ${SPEEDS.slow} 个 decode 步，中速 ${SPEEDS.med} 个，快速 ${SPEEDS.fast} 个。`))}"><span class="pa-speed-l">${tx('Speed','速度')}</span>`+
  `<div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</div></div>`;
/* R3.20: the run is never replayed on its own; Play is its replay control, and its tooltip says so */
const PLAYTIP=()=>tx(`Plays the run from the decode step shown. At the last decode step, Play starts the run again from decode step 1.`,`从当前显示的 decode 步开始播放；停在最后一个 decode 步时，点“播放”会从第 1 个 decode 步重新开始。`);
const timeCtl=(id,inline)=>{const r=run();
  return `<div class="${inline?'inline-ctl':'pa-timebox'}" id="${id}">${inline?`<label for="${id}-in"><span>${tx('Decode step','decode 步')}</span><output data-pa-out="time"></output></label>`:''}`+
    `<div class="pa-time"><button type="button" class="btn ghost sm pa-play" data-pa="play" aria-pressed="false" data-tipx="${esc(PLAYTIP())}"></button>${speedCtl()}<button type="button" class="btn ghost sm" data-pa="step">${tx('+1 decode step','+1 decode 步')}</button>`+
    `<input type="range" id="${id}-in" data-pa="time" min="0" max="${r.T-1}" step="1" value="${t}" aria-label="${esc(tx('Decode step shown in the pool','显存池显示的 decode 步'))}"></div>`+
    `${inline?'':`<div class="pa-tread" data-pa-out="tread"></div>`}</div>`;};
const segB=(id,inline,note)=>A.ui.seg({id,key:'B',label:tx('Block size (tokens per block)','block 大小（每个 block 放几个 token）'),options:BLOCKS.map(b=>[String(b),String(b)]),value:String(S.B),inline,note});
/* the buttons carry the names the text uses; the stylesheet stacks them as a list, since the three names do not fit on one row */
const segStrat=(id,inline,note)=>A.ui.seg({id,key:'strat',label:tx('Allocation strategy','分配方式'),options:STRATS.map(s=>[s,L(STRAT_NAME[s])]),value:S.strat,inline,note});
/* part 6: each block size the learner tried keeps its result under its button */
function triedHTML(){
  return `<div class="pa-btried" aria-label="${esc(tx('Results of the block sizes you tried','你试过的 block 大小的结果'))}">`+BLOCKS.map(B=>{if(!bTried.has(B)) return '<span></span>';
    const r=RP(B); return `<span${S.strat==='paged'&&S.B===B?' class="on"':''}><b>${pc(r.util)}</b><i>${n1(r.table)}</i></span>`;}).join('')+`</div>`+
    `<div class="pa-btcap">${tx('Under each size you tried are two numbers: the share of the pool storing tokens while requests wait, then the average number of block-table entries per request.','每个试过的大小下面有两个数：第一个是排队时存着 token 的比例，第二个是每个请求的 block table 平均条数。')}</div>`;}
const changedTxt=()=>tx(`You changed the setup under ${J('controls','Settings')}.`,`你在“${J('controls','自由调参')}”里改了设置。`);

const STEPS=[
 {title:{en:'Your mission',zh:'你的任务'},lead:{en:'Reserving for the worst case, the pool fits only 6 of the 40 requests at first.',zh:'按最坏情况预留，一开始显存池只放得下 40 个请求里的 6 个。'},enter:()=>({s:{...DEFAULT},t:0}),
  body:()=>{const r=R0(), lim=int(DEFAULT.lim);
   // the setup keeps its set values; the bytes per token (derived, not needed for the question) are in Fidelity and Settings (R6.20)
   return tx(`<p>Every token’s ${A.ui.def('KV cache',A.terms.kvcache())} takes GPU memory (see ${A.shell.labLink?A.shell.labLink('kv-basics'):'the KV cache basics lab'}). This lab runs Llama-3.1-8B: the memory set aside for the KV cache holds ${int(POOL)} tokens, and ${NREQ} requests are waiting.</p>`+
     `<p>Each answer may be up to ${lim} tokens long, but nobody knows in advance how long it will be. The simplest approach plans for the worst case: one contiguous region per request, room for its prompt plus ${lim} tokens.</p>`+
     `<p>That way the pool fits only ${r.agg.run[0]} requests at first. This lab shows where the memory goes, and how PagedAttention gets it back.</p>`,
     `<p>每个 token 的 KV cache 都要占显存（见 ${A.shell.labLink?A.shell.labLink('kv-basics'):'KV cache 入门 Lab'}）。这个 Lab 跑的是 Llama-3.1-8B：GPU 留给 KV cache 的显存能放 ${int(POOL)} 个 token，有 ${NREQ} 个请求在排队。</p>`+
     `<p>每个回答最多 ${lim} 个 token，但会写多长，事先谁也不知道。最简单的做法是按最坏情况，给每个请求预留一整块连续空间：prompt 加上 ${lim} 个 token。</p>`+
     `<p>这样一开始显存池只放得下 ${r.agg.run[0]} 个请求。这个 Lab 看显存花在了哪里，以及 PagedAttention 怎么把它省回来。</p>`);}},
 {title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: while requests wait, what share of the pool stores tokens?',zh:'先猜：有请求排队时，显存池平均有多大比例存着 token？'},gate:()=>!!pred,skip:true,enter:()=>({s:{...DEFAULT},t:0}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}"${seen?' disabled':''}>${L(o.t)}</button>`).join('')}</div>`;
   const note=seen?(pred?tx('Saved. Part 3 has shown the answer, so it can no longer be changed.','已记下。第 3 部分已经给出答案，不能再改了。'):tx('Part 3 has already shown the answer.','第 3 部分已经给出答案。'))
     :(pred?tx('Saved. You can still change it before the next part.','已记下。进入下一部分之前，还可以改。'):'');
   const lim=int(DEFAULT.lim);
   return tx(`<p>Every request reserves room for its prompt plus ${lim} tokens. While requests are waiting, what share of the pool actually stores tokens, on average?</p>`,
             `<p>每个请求都按“prompt 加 ${lim} 个 token”预留。有请求在排队时，显存池里平均有多大比例真正存着 token？</p>`)+ch+(note?`<p class="muted small">${note}</p>`:'');}},
 {title:{en:'Watch it run',zh:'看它跑起来'},lead:{en:'Press Play and watch the 40 requests being served.',zh:'点播放，看这 40 个请求怎么被服务。'},enter:()=>({s:{...DEFAULT},t:0}),target:'pa-ic-time',expect:()=>!!observed[2],showMe:true,
  body:()=>tx(`<p>Press Play to watch the ${NREQ} requests being served. In each decode step, every running request generates one more token; when a request finishes, its region is freed, and the request at the head of the queue starts if it fits, with no one behind it jumping ahead.</p><p>Drag to the last decode step (${int(R0().T)}) to see the result of the whole run.</p>`,
              `<p>点“播放”，看这 ${NREQ} 个请求怎么被服务。每个 decode 步，每个在跑的请求多生成 1 个 token；一个请求结束，它那一块就被释放，排在最前面的请求放得下就开始，后面的不插队。</p><p>拖到最后一个 decode 步（${int(R0().T)}），就能看到整次运行的结果。</p>`)+timeCtl('pa-ic-time',true)+`<p class="pa-live" data-pa-out="live" aria-live="off"></p>`,
  /* result (one rounded share; the cards hold the exact run), the verdict, then the reason in words (R3.10, R6.20) */
  obs:()=>{const r=R0(); let pf='';
    if(runKey(S)!==runKey(DEFAULT)){const c=run(); return tx(`<p>${changedTxt()} With “${esc(cfgName(S))}”, about ${pc0(c.util)} of the pool stores tokens while requests wait. To return to the initial setup, press Back and then Next.</p>`,
      `<p>${changedTxt()}当前是“${esc(cfgName(S))}”：有请求排队时，显存池约 ${pc0(c.util)} 存着 token。想回到初始设置，就先点“上一部分”，再点“下一部分”。</p>`);}
    if(pred){const p=L(PRED.find(x=>x.v===pred).t), ok=predOf(r.util);
      pf=pred===ok.v?tx(`<p class="okline">Your prediction, “${p}”, was right.</p>`,`<p class="okline">你预测的“${p}”是对的。</p>`)
        :tx(`<p>You predicted “${p}”; the right answer is “${L(ok.t)}”.</p>`,`<p>你预测的是“${p}”，正确答案是“${L(ok.t)}”。</p>`);}
    return tx(`<p><b>When each request reserves the maximum, only about ${pc0(r.util)} of the pool stores tokens while requests wait</b>, so only a few requests run at once.</p>`,
              `<p><b>按最大长度预留时，有请求排队的那些 decode 步里，显存池平均只有约 ${pc0(r.util)} 存着 token</b>，所以同时只能跑几个请求。</p>`)+pf+
      tx('<p>Why: most of each region is empty, either waiting for later tokens or never used.</p>','<p>原因：每一块大多空着，有的在等后面的 token，有的到结束也用不上。</p>');}},
 {title:{en:'Where the room goes',zh:'空间去哪了'},lead:{en:'The room that stores no tokens is wasted in three ways.',zh:'没存 token 的空间有三种浪费。'},enter:()=>({s:{...DEFAULT},t:R0().showcase,hl:'all'}),target:'pa-lv-hl',expect:()=>!!observed[3],showMe:true,
  body:()=>{const r=R0();
   return tx(`<p>The pool stops at decode step ${int(r.showcase+1)}: its mix of room is the closest to the average while requests wait. The room that stores no tokens is wasted in three ways. Pick one and see how much of the pool it takes.</p>`,
             `<p>显存池停在 decode 第 ${int(r.showcase+1)} 步：这一步各种空间的比例最接近排队时的平均。没存 token 的空间有三种浪费。选一种，看看它占了显存池的多少。</p>`)+
     `<div id="pa-lv-hl">${A.ui.levers({aria:tx('Kinds of room, pick one','空间的种类，选一种'),value:'hl-'+hl,options:HL_LEVERS.map(o=>({v:'hl-'+o.v,title:L(o.t),desc:L(o.d),result:hlTried.has(o.v)?lvResultHl(o.v):''}))})}</div>`+
     tx('<p class="small muted">A real server does not know a request’s final length, so it cannot tell “reserved for later tokens” from “internal fragmentation”: both are just reserved room.</p>',
        '<p class="small muted">真实服务时不知道请求的最终长度，所以分不清“为后面的 token 预留”和“内部碎片”：对服务端来说都只是预留。</p>');},
  obs:()=>{if(runKey(S)!==runKey(DEFAULT)) return tx(`<p>${changedTxt()} The pool no longer shows the initial setup; pick any option in this part to return to it.</p>`,`<p>${changedTxt()}显存池显示的已经不是初始设置，点这一部分里的任意一个选项就能回去。</p>`);
    // one rounded share per kind, the mechanism in words (R6.20); an example request's own numbers are in its tooltip on the pool
    const r=R0(), sh=r.shares;
    if(hl==='resv') return tx(`<p><b>Reserved for later tokens: about ${pc0(sh.resv)} of the pool while requests wait.</b> A request holds room for all its later tokens from the start, but fills it only one token per decode step; until then no other request can use that room.</p>`,
                              `<p><b>为后面的 token 预留：有请求排队时，约占显存池的 ${pc0(sh.resv)}。</b>请求一开始就占着后面所有 token 的空间，却要每个 decode 步写一个 token，慢慢填满；在那之前，这些空间别的请求也用不了。</p>`);
    if(hl==='intf'){const biggest=sh.intf>=sh.resv&&sh.intf>=sh.ext;
      return tx(`<p><b>Internal fragmentation: about ${pc0(sh.intf)} of the pool while requests wait${biggest?', the biggest waste':''}.</b> Most answers are far shorter than the answer cap, yet each request holds room for the whole cap until it finishes, so most of that room is never used.</p>`,
                `<p><b>内部碎片：有请求排队时，约占显存池的 ${pc0(sh.intf)}${biggest?'，是最大的一块浪费':''}。</b>多数回答比回答上限短得多，可每个请求到结束前都按整个上限占着空间，其中大部分永远用不上。</p>`);}
    if(hl==='ext') return tx(`<p><b>External fragmentation: about ${pc0(sh.ext)} of the pool while requests wait.</b> The room nobody holds (gaps between regions and the end of the pool) adds up, but each piece is smaller than the whole region the next request needs, so it stays empty.</p>`,
                             `<p><b>外部碎片：有请求排队时，约占显存池的 ${pc0(sh.ext)}。</b>没人占用的空间（块与块之间的空隙和显存池末尾）加起来不少，但每一段都比下一个请求要的整块小，只能空着。</p>`);
    return tx(`<p>All states: while requests wait, about ${pc0(sh.tok)} stores tokens; the rest is the three kinds of waste.</p>`,`<p>全部状态：有请求排队时，存着 token 的只有约 ${pc0(sh.tok)}，其余是上面三种浪费。</p>`);}},
 {title:{en:'Switch to paging',zh:'换成分页'},lead:{en:'PagedAttention hands out 16-token blocks only when they are needed.',zh:'PagedAttention 用到了才分配能放 16 个 token 的 block。'},enter:()=>({s:{...DEFAULT},t:R0().showcase}),target:'pa-lv-st',expect:()=>!!observed[4],showMe:true,
  body:()=>tx(`<p>PagedAttention stops reserving whole regions. It cuts each request’s KV cache into blocks with room for 16 tokens and hands out the next block only when the last one is full, from anywhere in the pool.</p><p>Each request keeps a block table listing its blocks in order. Blocks and block tables work like an operating system’s memory pages and page tables; attention reads keys and values through this table, hence the name PagedAttention.</p><p>The ${NREQ} requests and the pool stay the same. Pick an allocation strategy and see how much of the pool stores tokens while requests wait.</p>`,
              `<p>PagedAttention 不再一次预留整块，而是把每个请求的 KV cache 切成能放 16 个 token 的 block：上一个 block 写满了才分配下一个，而且 block 可以在显存池的任何位置。</p><p>每个请求用一张 block table 按顺序记下自己的 block 在哪。block 和 block table 就像操作系统的内存页和页表；attention 计算时照着这张表读取 key 和 value，所以叫 PagedAttention。</p><p>这 ${NREQ} 个请求和显存池都不变。选一种分配方式，看看有请求排队时，显存池有多少存着 token。</p>`)+
   `<div id="pa-lv-st">${A.ui.levers({aria:tx('Allocation strategies, pick one','分配方式，选一种'),value:'st-'+(ST_LEVERS[stLeverNow()]||{}).v,options:ST_LEVERS.map(o=>({v:'st-'+o.v,title:L(o.t),desc:o.d(),result:stTried.has(o.v)?lvResultSt(o.v):''}))})}</div>`,
  obs:()=>{const i=stLeverNow(), b=R0();
    if(i<0) return tx(`<p>${changedTxt()} The setup is no longer one of the three options in this part; pick one of them to compare again. Right now, with “${esc(cfgName(S))}”, about ${pc0(run().util)} of the pool stores tokens while requests wait.</p>`,
                      `<p>${changedTxt()}现在的配置不属于这一部分的三个选项，点其中一个就能回来比较。当前是“${esc(cfgName(S))}”：有请求排队时，显存池约 ${pc0(run().util)} 存着 token。</p>`);
    // one key number per paragraph, the share before → after; running requests and decode steps are on the cards (R6.20)
    if(i===0) return tx(`<p><b>Initial setup: while requests wait, only about ${pc0(b.util)} of the pool stores tokens</b>, so few requests run at once and the run takes many decode steps.</p>`,
                        `<p><b>初始设置：有请求排队时，显存池只有约 ${pc0(b.util)} 存着 token</b>，所以同时只能跑几个请求，整次运行要很多个 decode 步。</p>`);
    const r=run();
    if(i===1) return tx(`<p><b>Exact length: while requests wait, about ${pc0(r.util)} of the pool stores tokens (initial setup: about ${pc0(b.util)}).</b> Knowing the lengths is not enough: each region holds room for tokens not generated yet from the start, and the gaps between regions stay unusable.</p>`,
                        `<p><b>按实际长度预留：有请求排队时，显存池约 ${pc0(r.util)} 存着 token（初始设置约 ${pc0(b.util)}）。</b>即使事先知道长度也不够：每一块从一开始就为还没生成的 token 占着空间，块与块之间的空隙也用不上。</p>`);
    // how many times as many requests run at once, from the shown values; the decode steps fall by less (both on the cards)
    const ratio=shown1(r.running)/shown1(b.running), fold=Math.floor(ratio+1e-9), less=b.T/r.T<fold;
    const fracEn={2:'half',3:'a third',4:'a quarter',5:'a fifth',6:'a sixth',7:'a seventh',8:'an eighth',9:'a ninth'}[fold]||`1/${fold}`;
    const fracZh={2:'二分之一',3:'三分之一',4:'四分之一',5:'五分之一',6:'六分之一',7:'七分之一',8:'八分之一',9:'九分之一'}[fold]||`${fold} 分之一`;
    const runs=ratio>=2?tx(`So ${manyW(ratio)} requests run at once as in the initial setup.`,`所以同时跑的请求是初始设置的${manyW(ratio)}。`):tx('So more requests run at once than in the initial setup.','所以同时跑的请求比初始设置多。');
    return tx(`<p><b>Paging: while requests wait, about ${pc0(r.util)} of the pool stores tokens (initial setup: about ${pc0(b.util)}).</b></p><p>${runs} ${less?`The decode steps do not fall to ${fracEn}, though: once every waiting request has started, fewer and fewer are still running.`:'The run needs far fewer decode steps.'}</p><p>Why: a request wastes at most the unfilled part of its last block, room for fewer than ${r.B} tokens, and a free block works anywhere, so no gap is too small.</p>`,
              `<p><b>分页：有请求排队时，显存池约 ${pc0(r.util)} 存着 token（初始设置只有约 ${pc0(b.util)}）。</b></p><p>${runs}${less?`decode 步数却没有降到${fracZh}：排队的请求都开始以后，还在跑的请求越来越少。`:'整次运行需要的 decode 步也少了一大截。'}</p><p>原因：每个请求最多浪费最后一个 block 里没写满的部分，不到 ${r.B} 个 token 的空间；空闲的 block 在哪都能用，所以没有放不下的空隙。</p>`);}},
 {title:{en:'Pick a block size',zh:'选 block 大小'},lead:{en:'Larger blocks waste more room; smaller ones make longer block tables.',zh:'block 越大浪费越多；越小 block table 越长。'},enter:()=>({s:{...DEFAULT,strat:'paged',B:16},t:RP(16).showcase}),target:'pa-ic-B',expect:()=>!!observed[5],showMe:true,
  body:()=>tx('<p>Block size is a trade-off. Larger blocks leave more of each request’s last block unfilled; smaller blocks make each block table longer, so attention has to look up and read more small pieces.</p><p>This lab counts memory only and does not time attention, so the cost of small blocks does not show up in the decode steps. Change the block size and compare with vLLM’s default of 16:</p>',
              '<p>block 大小是取舍。block 越大，每个请求最后一个 block 里没写满的部分越大；block 越小，每个请求的 block table 越长，attention 计算时要查找和读取的小块也越多。</p><p>这个 Lab 只算显存，不计算 attention 的时间，所以小 block 的代价不会出现在 decode 步数里。改一改 block 大小，和 vLLM 默认的 16 比一比：</p>')+segB('pa-ic-B',true,`<div data-pa-tried>${triedHTML()}</div>`),
  obs:()=>{const r=run(), b=RP(16), B=S.B;
    const close=tx('<p>vLLM defaults to 16: the paper found it large enough for the GPU to read efficiently and small enough to avoid much internal fragmentation in most workloads (§7.2).</p>','<p>vLLM 默认用 16：论文发现，它大到能让 GPU 高效读取，又小到在多数负载下不会造成明显的内部碎片（§7.2）。</p>');
    if(S.strat!=='paged') return tx(`<p>The strategy is now “${esc(cfgName(S))}”, so block size has no effect. Pick a block size in this part to switch back to paging.</p>`,`<p>现在的分配方式是“${esc(cfgName(S))}”，block 大小不起作用。在这一部分里选一个 block 大小，就会切回分页。</p>`);
    if(B===16) return tx(`<p><b>16-token blocks: while requests wait, about ${pc0(r.util)} of the pool stores tokens.</b> Try a smaller size and a larger one.</p>`,`<p><b>16 个 token 一个 block：有请求排队时，约 ${pc0(r.util)} 的显存池存着 token。</b>试一个更小的，再试一个更大的。</p>`)+close;
    // against the default: the share in words (or before → after when it moves 2 points or more) and the table length in words;
    // the exact values are on the metric cards, with their ▲▼ against the previous size, and under each size tried (R6.20)
    const mv=shareMoveW(r.util,b.util), conj=B<16||!mv.small?tx('but','但'):tx('and','而且');
    const why=B===1?tx('Why: no block is half written, but every token needs its own entry.','原因：没有写了一半的 block，但每个 token 都要一条 block table 记录。')
      :(B<16?tx('Why: smaller blocks leave less of the last one empty, but more are needed.','原因：block 越小，最后一个 block 空着的越少，但需要的 block 也越多。')
      :tx('Why: bigger blocks leave more of the last one empty, but fewer are needed.','原因：block 越大，最后一个 block 空着的越多，但需要的 block 也越少。'));
    return tx(`<p><b>${B}-token blocks: against the default size, the share of the pool storing tokens ${mv.t}, ${conj} block tables ${tableW(r.table,b.table)}.</b> ${why}</p>`,
              `<p><b>${B} 个 token 一个 block：和默认大小相比，存着 token 的比例${mv.t}，${conj} block table ${tableW(r.table,b.table)}。</b>${why}</p>`)+close;}},
 {title:{en:'What to remember',zh:'要记住的'},enter:()=>({s:{...DEFAULT,strat:'paged',B:16},t:RP(16).showcase}),
  body:()=>{const b=R0(), x=RX(), p=RP(16), nxt='';   // the next lab is the kit's footer under the rail card (R3.11)
   // the trend in words, one key number per bullet (R6.20); "five times as many" (in the misconception) comes from the running requests as the cards show them
   const q=shown1(p.running)/shown1(b.running), big=q>=2;
   const runsNow=big?tx(`${manyW(q)} requests run at once while requests wait`,`分页后排队时同时跑的请求却是原来的${manyW(q)}`):tx('more requests run at once while requests wait','分页后排队时同时跑的请求却更多了');
   return tx(`<ul><li>Reserving the maximum for each request in one region, only about ${pc0(b.util)} of the pool stores tokens on average while requests wait; the rest is room reserved for later tokens, internal fragmentation and external fragmentation.</li><li>PagedAttention cuts the KV cache into fixed-size blocks, hands out the next one only when the last is full, and finds them through a block table. Each request wastes at most the unfilled part of one block, so about ${pc0(p.util)} of the pool stores tokens while requests wait.</li><li>Block size is a trade-off: larger blocks leave more unfilled room, smaller blocks make block tables longer. vLLM defaults to 16.</li></ul>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Fragmentation is a small effect.” Here it took most of the pool.<br>✗ “How many requests can be served at once depends only on how much memory there is.” The pool stayed the same size, yet with paging ${runsNow}.<br>✗ “Knowing each answer’s length would be enough to reserve exactly.” Then still only about ${pc0(x.util)} of the pool stores tokens while requests wait.</p></div>${A.shell.toScenarios()}${nxt}`,
             `<ul><li>按最大长度给每个请求预留一整块，有请求排队时，显存池平均只有约 ${pc0(b.util)} 存着 token；其余是为后面的 token 预留的空间、内部碎片和外部碎片。</li><li>PagedAttention 把 KV cache 切成固定大小的 block，写满一个再分配下一个，用 block table 找到它们。每个请求最多浪费一个 block 里没写满的部分，所以排队时约 ${pc0(p.util)} 的显存池存着 token。</li><li>block 大小是取舍：越大，没写满的部分越大；越小，block table 越长。vLLM 默认 16。</li></ul>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “碎片只是小问题。”这里它占了显存池的一大半。<br>✗ “能同时服务多少请求，只取决于显存有多大。”显存池大小没变，${runsNow}。<br>✗ “只要知道回答多长，按实际长度预留就行。”那样排队时显存池也只有约 ${pc0(x.util)} 存着 token。</p></div>${A.shell.toScenarios()}${nxt}`);}}
];
const DESCS=()=>({
 guided:tx('Seven short parts. Predict first, see where reserving the maximum wastes memory, switch to paging, then pick a block size.','一共 7 个部分。先预测，再看按最大长度预留浪费在哪，然后换成分页，最后选 block 大小。'),
 sandbox:tx('Pick a real workload and see how the three allocation strategies do under it.','选一种真实的负载，看三种分配方式在它下面表现如何。'),
 challenge:tx(`Llama-3.1-8B, a pool with room for ${int(POOL)} tokens, the same ${NREQ} requests. Change only the allocation strategy and block size so that, while requests wait, at least ${pc(GOAL_UTIL,0)} of the pool stores tokens on average, with block tables averaging ${GOAL_TABLE} entries or fewer.`,
              `Llama-3.1-8B，显存池能放 ${int(POOL)} 个 token，还是那 ${NREQ} 个请求。只改分配方式和 block 大小，让有请求排队时显存池平均至少 ${pc(GOAL_UTIL,0)} 存着 token，同时每个请求的 block table 平均不超过 ${GOAL_TABLE} 条。`)
});
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers.
   Every number is worked out from the lab's own run and constants. Nothing here gives the challenge away: no share of the pool and no
   block-table length for any block size, and no word on which setting meets both goals. */
const OUTCOME=()=>tx('explain why reserving each request’s maximum length wastes most KV cache memory, and count the blocks a request takes with paging.',
                     '讲清按最大长度预留为什么会浪费一大半 KV cache 显存，并算出分页时一个请求要几个 block。');
/* a new case opened in Scenarios, as the KV cache lab's kvGo does: the mode first, then the state, then the stage */
function exGo(p){setMode('sandbox'); setS(p); A.ui.reveal($('#pa-stage',host));}
const EXITS=()=>{
  // explain: the initial setup's run (the same workload and pool as the challenge's initial state)
  const b=R0(), sh=b.shares, waste=1-sh.tok;
  // the explanations in words (R6.20): the one number kept is the room never used per request, rounded (the card has 1,232.5)
  const unusedR=int(Math.round(b.unused/100)*100), smallPart=sh.resv<0.3*waste, biggest=sh.intf>=sh.resv&&sh.intf>=sh.ext;
  // a new case 1: one request of 1,000 tokens in 32-token blocks; each wrong option is a mistake a learner makes
  const L1=1000, B1=32, nb=Math.ceil(L1/B1), room=nb*B1, empty=room-L1, held=L1-(nb-1)*B1;
  const c1=[{v:0,n:tx(`Paging leaves the last block partly empty unless the length is a whole number of blocks, and ${int(L1)} is not a multiple of ${B1}.`,`除非长度正好是整数个 block，否则最后一个 block 总有一部分空着；${int(L1)} 不是 ${B1} 的整数倍。`)},
    {v:held,n:tx(`${held} is what the last block holds, not the room it leaves empty.`,`${held} 是最后一个 block 里放着的 token 数，不是它空着的空间。`)},
    {v:B1/2,n:tx('Half a block is roughly the average over many requests of different lengths. One request’s empty room follows from its own length.','半个 block 大约是很多长度不同的请求的平均值。一个请求空着多少，要按它自己的长度算。')},
    {v:empty,n:null}].sort((x,y)=>x.v-y.v);
  // a new case 2: the scenario with the largest answer cap, each request reserving its maximum length
  const api=PRESETS.reduce((x,y)=>y.p.lim>x.p.lim?y:x), cap=api.p.lim, lo=PMIN+cap, fit=Math.floor(POOL/lo);
  const n2=tx('Reserving the maximum holds room for the answer cap, not for the answer’s actual length, so short answers do not help.','按最大长度预留，占的是回答上限的空间，不是回答实际的长度，所以回答再短也没用。');
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does most of the pool store no tokens when each request reserves its maximum length, even though most answers are short?','按最大长度预留时，虽然多数回答都很短，显存池却有一大半没存 token。为什么？'),
   o:[tx('Nobody knows an answer’s length in advance, so each request holds room for its prompt plus the answer cap from the start, and a short answer never fills most of it.','回答会写多长，事先谁也不知道，所以每个请求一开始就占着 prompt 加回答上限的空间；回答短，其中大部分就一直用不上。'),
     tx('Each region fills up by the time its request finishes, but until then the room for its later tokens sits empty, and at any moment that is most of the pool.','每一块到请求结束时都会写满，只是在那之前，留给后面 token 的空间一直空着；任何时刻，这部分都占了显存池的一大半。'),
     tx('Finished requests leave gaps between the regions, and each gap is too small for the region the next request needs, so most of the pool sits in those gaps.','请求结束后，块与块之间会留下空隙，每段空隙都放不下下一个请求要的一整块；所以显存池的一大半都空在这些空隙里。')],a:0,
   n:[null,tx(`No region here ever fills up: even the longest of these answers stays under the answer cap. Room waiting for later tokens is ${smallPart?'only a small part':'part'} of the waste; ${biggest?'the biggest part':'much of it'} is room that is never used.`,
              `这里没有哪一块会写满：这些回答里最长的也没到回答上限。为后面的 token 预留的空间只是浪费里的${smallPart?'一小部分':'一部分'}，${biggest?'最大的一部分':'很大一部分'}是到结束也用不上的空间。`),
     tx(`Those gaps (external fragmentation) are real, but ${biggest?'the biggest waste':'much of the waste'} is inside the regions: room held for tokens that are never generated (internal fragmentation).`,
        `这些空隙（外部碎片）确实存在，但${biggest?'最大的浪费':'很大一部分浪费'}在每一块里面：为永远不会生成的 token 占着的空间（内部碎片）。`)],
   why:tx(`A server cannot know an answer’s length in advance, so reserving the maximum means holding room for the prompt plus the answer cap from the start. A request uses only its prompt plus its actual answer, and most answers are far shorter than the cap. So on average each request leaves about ${unusedR} tokens of room unused.`,
          `服务端事先不知道回答多长，按最大长度预留，就要从一开始占着 prompt 加回答上限的空间。请求只用到 prompt 加它实际的回答，而多数回答比上限短得多。所以平均每个请求约有 ${unusedR} 个 token 的空间到结束也没用上。`)},
  {q:tx('With paging, what does each request’s block table do?','分页时，每个请求的 block table 是做什么用的？'),
   o:[tx('It lists where each of the request’s blocks sits, in order, so attention can find them even though each one came from wherever a block was free.','它按顺序记下这个请求的每个 block 在哪，所以即使 block 都是从任意空闲的地方取来的，attention 也能找到它们。'),
     tx('It keeps each request’s blocks next to each other in the pool, so attention can read all of that request’s keys and values in one piece.','它让这个请求的 block 在显存池里挨在一起，所以 attention 计算时能一次连续读完这个请求的 key 和 value。'),
     tx('It lists the blocks in the pool that are still free, so the server can find room for the next request without searching the whole pool.','它记下显存池里每一个还空闲的 block，所以服务端给下一个请求找空间时，查这张表就行，不用把整个显存池搜一遍。')],a:0,
   n:[null,tx('Paging does not keep a request’s blocks together: each new one comes from wherever a block is free. The table is what lets attention read them in order anyway.','分页不会让一个请求的 block 挨在一起：新的 block 从任意空闲的地方取。正是这张表，让 attention 照样能按顺序读它们。'),
     tx('That is the list of free blocks, one for the whole pool. A block table belongs to one request and lists only the blocks that request holds.','那是空闲 block 列表，整个显存池只有一个。block table 每个请求一张，只记这个请求自己占着的 block。')],
   why:tx('A request gets its next block only when its last one is full, from wherever a block is free, so its blocks end up scattered. Attention finds its keys and values through its block table, which lists them in order. Any free block will do, so no gap is too small, and paging leaves no external fragmentation.',
          '请求写满一个 block，才从任意空闲的地方再取一个，所以它的 block 散落在显存池各处。attention 照着它的 block table 找 key 和 value，这张表按顺序记着这些 block。任何空闲 block 都能用，没有放不下的空隙，所以分页不会产生外部碎片。')}],
 transfer:[
  {q:tx(`A request ends with ${int(L1)} tokens (prompt plus answer), stored in ${B1}-token blocks. How much room, in tokens, does its last block leave empty?`,`一个请求结束时一共 ${int(L1)} 个 token（prompt 加回答），按 ${B1} 个 token 一个 block 存放。它最后一个 block 空着多少个 token 的空间？`),
   o:c1.map(x=>String(x.v)),a:c1.findIndex(x=>x.v===empty),n:c1.map(x=>x.n),
   // a worked computation, as short steps, one per sentence (R6.20)
   why:tx(`It takes ⌈${int(L1)} ÷ ${B1}⌉ = ${nb} blocks. They hold ${nb} × ${B1} = ${int(room)} tokens. So ${int(room)} − ${int(L1)} = ${empty} are left empty. The last block holds ${int(L1)} − ${nb-1} × ${B1} = ${held} tokens. With paging, this is all the room a request never uses: always less than one block.`,
          `它要 ⌈${int(L1)} ÷ ${B1}⌉ = ${nb} 个 block。它们能放 ${nb} × ${B1} = ${int(room)} 个 token。所以空着 ${int(room)} − ${int(L1)} = ${empty} 个 token 的空间。最后一个 block 放了 ${int(L1)} − ${nb-1} × ${B1} = ${held} 个 token。分页时，一个请求用不上的空间就只有这些，总是不到一个 block。`)},
  {q:tx(`An API’s default answer cap (max_tokens) is ${int(cap)} tokens, though its answer lengths have a mean of only ${int(api.p.mu)}. Prompts are ${PMIN} to ${PMAX} tokens. If each request reserves its maximum length in this lab’s pool (room for ${int(POOL)} tokens), how many requests fit at once?`,
        `一个 API 默认的回答上限（max_tokens）是 ${int(cap)} 个 token，而它的回答长度均值只有 ${int(api.p.mu)} 个 token。prompt 长 ${PMIN} 到 ${PMAX} 个 token。如果每个请求都在这个 Lab 的显存池（能放 ${int(POOL)} 个 token）里按最大长度预留，一次能放下几个请求？`),
   o:[tx('1','1 个'),tx('2 or 3','2 到 3 个'),tx('4 to 10','4 到 10 个'),tx('More than 10','10 个以上')],a:fit<=1?0:(fit<=3?1:(fit<=10?2:3)),n:[null,n2,n2,n2],
   why:tx(`Each request holds room for its prompt plus ${int(cap)} tokens, so at least ${PMIN} + ${int(cap)} = ${int(lo)}. Two need at least 2 × ${int(lo)} = ${int(2*lo)}, ${2*lo>POOL?'more':'no more'} than the pool’s ${int(POOL)}. So only ${fit} ${fit===1?'request runs':'requests run'} at a time, however short its answer.`,
          `每个请求要占 prompt 加 ${int(cap)} 个 token 的空间，至少 ${PMIN} + ${int(cap)} = ${int(lo)} 个。两个请求至少要 2 × ${int(lo)} = ${int(2*lo)} 个，${2*lo>POOL?'超过':'不超过'}显存池的 ${int(POOL)} 个。所以一次只能跑 ${fit} 个请求，不管它的回答多短。`),
   go:()=>exGo({strat:'max',...api.p}),goLabel:tx(`Open it in Scenarios: ${L(api.t)}`,`在应用场景里打开：${L(api.t)}`)}],
 practice:[
  {id:'inference-engine',d:tx('Levels 1 and 3: give each new request enough KV cache blocks for its prompt plus the token it is about to generate, and free them all when it stops.','第 1、3 关：给每个新请求分配够用的 KV cache block，放得下 prompt 和即将生成的那个 token；请求停止时，释放它的全部 block。')},
  {id:'memory-allocator',d:tx('Part 1: write allocate and free over one contiguous address space, placing each range first fit (at the lowest address where it fits) and merging free ranges that touch: the contiguous allocation whose gaps paging avoids.','第 1 部分：在一段连续的地址空间上写出 allocate 和 free，每段区间按首次适配放置（放在放得下的最低地址），相邻的空闲区间要合并。分页避开的空隙，就出在这种连续分配上。')}]};};

/* ================= views ================= */
const SRC_VLLM=p=>`<a href="https://github.com/vllm-project/vllm/blob/main/${p}" target="_blank" rel="noopener"><code>${p}</code></a>`;
function fidPop(){const k=kv8b(), g=gpuEst(), m=timing(), p=RP(16), sw=sweep(), b=R0();
 const kt=A.fmt.B(k.kvPerToken), gib=A.fmt.B(poolBytes()), w=A.fmt.num(m.W/A.GiB,0);
 return tx(`<div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>A KV cache pool with room for ${int(POOL)} tokens (${kt} per token for Llama-3.1-8B, ${gib} (${A.fmt.num(poolBytes()/1e9,2)} GB) in total), each cell drawn for ${CELL} tokens.</li><li>${NREQ} requests served first come, first served. In each decode step every running request generates one token, and a finished request frees its room.</li><li>Three allocation strategies: reserve the maximum; reserve the exact length (this needs each answer’s length in advance, which a real server cannot know); paging (the next block only when the last is full). When no block is free, the most recently started request is paused and its KV cache recomputed later, as vLLM does; this happens ${p.pre} ${p.pre===1?'time':'times'} in the paged run.</li></ul></div>`+
   `<div><div class="eyebrow">Simplified and assumed</div><ul><li>The workload is an assumption: prompts of ${PMIN} to ${PMAX} tokens, spread evenly; answer lengths drawn from an exponential distribution with mean ${DEFAULT.mu}, cut at the cap; random seed ${SEED}.</li><li>Every decode step takes the same time. Each step reads about ${w} GiB of weights and at most ${A.fmt.B(m.poolB)} of KV cache, so how many requests run at once barely changes it; counting bytes read, the initial setup takes ${f2(m.bytes)} times as long as paging, against ${f2(m.steps)} times as many decode steps.</li><li>For these requests the initial setup needs ${f2(sw.ratio)} times as many decode steps as paging, more than for ${sw.below} of random seeds 1 to ${sw.n} (median ${f2(sw.median)}).</li><li>The pool is scaled down: a real RTX 4090 holds about ${int(g.t90)} tokens after the weights (estimated with a GPU memory utilization of ${A.fmt.num(g.util,1)} and ${g.oh} GiB of runtime overhead; about ${int(g.t92)} with vLLM’s default of 0.92). With a larger pool and proportionally more requests, the shares stay about the same.</li><li>Attention time is not modeled; the challenge’s block-table cap stands in for the cost of small blocks. Free blocks are reused first in, first out: vLLM’s order differs (a block that is not full goes back to the front of the queue), but that changes no number here.</li></ul></div>`+
   `<div><div class="eyebrow">Sources</div><ul><li>Kwon et al., SOSP 2023, <a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noopener">the vLLM paper</a>: the three kinds of waste (§3.1), PagedAttention (§4), block size 16 (§7.2), and earlier systems storing tokens in only 20.4% to 38.2% of KV cache memory (Figure 2; this simulation’s ${pc(p.util)} agrees with the paper’s 96.3% for vLLM, but that is not a validation).</li><li>vLLM source: ${SRC_VLLM('vllm/v1/core/block_pool.py')}, ${SRC_VLLM('vllm/v1/core/kv_cache_utils.py')}, ${SRC_VLLM('vllm/v1/core/kv_cache_manager.py')}, ${SRC_VLLM('vllm/v1/core/sched/scheduler.py')}, ${SRC_VLLM('vllm/config/cache.py')}, and the block sizes each attention backend accepts in ${SRC_VLLM('vllm/v1/attention/backends/flash_attn.py')} and the other backends.</li><li>Llama-3.1-8B’s shape: its config.json on Hugging Face.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-09-24. Prototype, not yet expert-reviewed.</p></div></div>`,
   `<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>一个能放 ${int(POOL)} 个 token 的 KV cache 显存池（Llama-3.1-8B 每个 token ${kt}，共 ${gib}（${A.fmt.num(poolBytes()/1e9,2)} GB）），每格画 ${CELL} 个 token。</li><li>${NREQ} 个请求先到先服务；每个 decode 步，每个在跑的请求生成 1 个 token；请求结束就释放它的空间。</li><li>三种分配方式：按最大长度预留；按实际长度预留（需要事先知道回答长度，真实服务做不到）；分页（写满一个 block 再分配下一个。没有空闲 block 时，暂停最近开始的请求，之后重算它的 KV cache，vLLM 也这样做；分页那次运行里发生了 ${p.pre} 次）。</li></ul></div>`+
   `<div><div class="eyebrow">简化和假设</div><ul><li>负载是假设的：prompt ${PMIN} 到 ${PMAX} 个 token 均匀分布；回答长度按均值 ${DEFAULT.mu} 的指数分布抽取，截断在上限；随机种子 ${SEED}。</li><li>每个 decode 步耗时相同。每步要读约 ${w} GiB 权重，KV cache 最多 ${A.fmt.B(m.poolB)}，所以同时跑多少请求对耗时影响不大；按读取字节数算，初始设置和分页的耗时之比是 ${f2(m.bytes)}，按 decode 步数是 ${f2(m.steps)}。</li><li>这组请求下，初始设置用的 decode 步数是分页的 ${f2(sw.ratio)} 倍，比随机种子 1 到 ${sw.n} 里的 ${sw.below} 个都高（中位数 ${f2(sw.median)}）。</li><li>显存池按比例缩小：一张 RTX 4090 放下权重后约能放 ${int(g.t90)} 个 token（按 ${A.fmt.num(g.util,1)} 的显存利用率和 ${g.oh} GiB 运行时开销估算；按 vLLM 默认的 0.92 约 ${int(g.t92)} 个）。放大显存池和请求数时，比例基本不变。</li><li>不计算 attention 的时间；挑战里 block table 的上限代表小 block 的代价。空闲 block 按先进先出复用：vLLM 的顺序不同（没写满的 block 放回队首），但不改变这里的数字。</li></ul></div>`+
   `<div><div class="eyebrow">来源</div><ul><li>Kwon 等，SOSP 2023（<a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noopener">vLLM 论文</a>）：三种浪费（§3.1）、PagedAttention（§4）、block 大小 16（§7.2）、以前的系统只有 20.4% 到 38.2% 的 KV cache 显存存着 token（图 2；本模拟的 ${pc(p.util)} 和论文里 vLLM 的 96.3% 一致，但这不算验证）。</li><li>vLLM 源码：${SRC_VLLM('vllm/v1/core/block_pool.py')}、${SRC_VLLM('vllm/v1/core/kv_cache_utils.py')}、${SRC_VLLM('vllm/v1/core/kv_cache_manager.py')}、${SRC_VLLM('vllm/v1/core/sched/scheduler.py')}，${SRC_VLLM('vllm/config/cache.py')}；各 attention 后端接受的 block 大小见 ${SRC_VLLM('vllm/v1/attention/backends/flash_attn.py')} 等。</li><li>Llama-3.1-8B 的结构：Hugging Face 上的 config.json。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-09-24。原型，尚未经专家审阅。</p></div></div>`);}
// the primer is the KV lab's, word for word, so the two labs name their shared prerequisite the same way
const priPop=()=>A.t('priPop')+tx(`<p>The lab ${A.shell.labLink('kv-cache')} works out how much memory it takes per token. This lab is about how it is laid out in memory.</p>`,
                   `<p>每个 token 占多少显存，在 ${A.shell.labLink('kv-cache')}那个 Lab 里算过。这个 Lab 讲它在显存里怎么摆放。</p>`);
function legendHTML(){
  return `<div class="legend-row pa-legend" id="pa-legend">`+STATE_KEYS.map(k=>`<span data-tipx="${esc(L(STATE_TIP[k]))}" data-jump="hl-${k}" role="button" tabindex="0" aria-pressed="${hl===k}"><i class="pa-sw pa-f-${k}"></i><span>${esc(L(LEGEND_NAME[k]))}</span></span>`).join('')+
    `<span id="pa-lg-start" tabindex="0" data-tipx="${esc(tx('The thick line marks where each request’s region begins (drawn only when regions are reserved).','粗竖线标出每个请求那一块的开头（只在按长度预留时画）。'))}"><i class="pa-sw pa-f-start"></i><span>${tx('Region start (thick line)','每块的开头（粗竖线）')}</span></span></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="pa-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
    `<p class="ctl-lock" id="pa-lock" hidden></p>`+
    segStrat('pa-ctl-strat',false,`<span id="pa-strat-note"></span>`)+
    segB('pa-ctl-B',false,`<span id="pa-b-note"></span>`)+
    `<div class="lockable">${A.ui.seg({id:'pa-ctl-mu',key:'mu',label:tx('Answer length (mean)','回答长度（均值）'),options:MEANS.map(v=>[String(v),int(v)]),value:String(S.mu),note:`<span id="pa-mu-note"></span>`})}</div>`+
    `<div class="lockable">${A.ui.seg({id:'pa-ctl-lim',key:'lim',label:tx('Answer cap (max_tokens)','回答上限（max_tokens）'),options:LIMITS.map(v=>[String(v),v>=1024?(v/1024)+'K':String(v)]),value:String(S.lim),note:`<span id="pa-lim-note"></span>`})}</div>`+
    `<details class="adv lockable" id="pa-adv"><summary>${tx('Pool size, prompt lengths, random seed and timing','显存池大小、prompt 长度、随机种子和计时')}</summary><div id="pa-advbody"></div></details>`+
    `<div><button class="btn ghost sm" type="button" data-act="reset">${A.t('reset')}</button></div></aside>`;}
function advHTML(){const k=kv8b(), g=gpuEst();
  return tx(`<p><b>Pool:</b> room for ${int(POOL)} tokens, ${A.fmt.B(poolBytes()).replace(/ GiB$/,' '+A.ui.def('GiB',A.terms.gib()))} (${A.fmt.num(poolBytes()/1e9,2)} GB) of Llama-3.1-8B KV cache. It is scaled down so each cell stays visible; a real RTX 4090 holds about ${int(g.t90)} tokens after the weights.</p>`+
    `<p><b>Requests:</b> ${NREQ}, all waiting at the start and served first come, first served.</p>`+
    `<p><b>Prompts:</b> ${PMIN} to ${PMAX} tokens, spread evenly. This is an assumption and stays fixed.</p>`+
    `<p><b>Random seed:</b> ${SEED}. Every run replays the same ${NREQ} requests, so settings can be compared.</p>`+
    `<p><b>Free blocks</b> are reused first in, first out, a simplification. It only changes where blocks sit in the picture, not any number.</p>`+
    `<p><b>Timing:</b> every decode step is assumed to take the same time (the ${J('fid','notes at the end of the page')} say why). Prefill and recomputation after a pause are not counted.</p>`,
    `<p><b>显存池：</b>能放 ${int(POOL)} 个 token，即 ${A.fmt.B(poolBytes()).replace(/ GiB$/,' '+A.ui.def('GiB',A.terms.gib()))}（${A.fmt.num(poolBytes()/1e9,2)} GB）的 Llama-3.1-8B KV cache。按比例缩小，好让每一格看得清；一张 RTX 4090 放下权重后约能放 ${int(g.t90)} 个 token。</p>`+
    `<p><b>请求：</b>${NREQ} 个，一开始全部在排队，先到先服务。</p>`+
    `<p><b>prompt：</b>${PMIN} 到 ${PMAX} 个 token 均匀分布。这是假设，保持不变。</p>`+
    `<p><b>随机种子：</b>${SEED}。每次都重放同样的 ${NREQ} 个请求，不同设置之间才能比较。</p>`+
    `<p><b>空闲 block</b> 按先进先出复用，是简化。它只影响 block 在图里的位置，不影响任何数字。</p>`+
    `<p><b>计时：</b>假设每个 decode 步耗时相同（原因见${J('fid','页末的“简化、假设和来源”')}）。prefill 和暂停后的重算不计入。</p>`);}
const MET=[
 {k:'util',tip:'pa_mUtil',jump:'strat',label:()=>tx('Stores tokens while requests wait','排队时存着 token'),v:r=>pc(r.util),dp:1,pts:true},
 {k:'run',tip:'pa_mRun',label:()=>tx('Running at once while requests wait','排队时同时在跑'),v:r=>n1(r.running),dp:1},
 {k:'steps',jump:'fid',label:()=>tx('All 40 finish after','全部完成'),v:r=>int(r.T),dp:0},
 {k:'unused',tip:'pa_mUnused',label:()=>tx('Never used per request','每个请求没用上'),v:r=>n1(r.unused),dp:1},
 {k:'table',tip:'pa_mTable',jump:'B',label:()=>tx('Block table','block table'),v:r=>r.table==null?tx('none','无'):n1(r.table),dp:1}
];
function view(){
  const lab=A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:'PagedAttention',title:L(lab?lab.q:{en:'Why is most KV cache memory wasted?',zh:'为什么 KV cache 显存有一大半被浪费？'}),minutes:12,level:tx('Core','核心'),
      fid:'Simulated',fidPop:fidPop(),primer:A.t('priBtn').replace(/\s*▾$/,''),primerPop:priPop(),descs:DESCS(),mode})+
    `<div class="wrap bench"><aside class="rail" id="pa-rail" aria-label="${esc(A.t('railAria'))}"></aside>`+
    `<div class="panel stage" id="pa-stage"><div class="stage-head"><div><div class="meter-title">${tx(`KV cache pool (Llama-3.1-8B, room for ${int(POOL)} tokens)`,`KV cache 显存池（Llama-3.1-8B，能放 ${int(POOL)} 个 token）`)}</div><div class="meter-sub" id="pa-sub"></div></div>`+
    `<div class="stage-actions"><span class="status" id="pa-status" tabindex="0" data-tip="pa_status"></span></div></div><div class="pa-wl" id="pa-wl" tabindex="0" role="button"></div>`+
    legendHTML()+
    `<div class="stage-graph graph" id="pa-graph"><div class="pa-lab"><span id="pa-cellnote"></span></div>`+
    `<div class="pa-pool" id="pa-pool"><canvas class="pa-canvas" id="pa-cv" tabindex="0" role="img"></canvas><div class="pa-anchor" id="pa-anchor"></div></div>`+
    timeCtl('pa-st-time',false)+
    `<div class="pa-bars"><div class="pa-brow"><span class="pa-bl" id="pa-nowlab" tabindex="0"></span><div class="pa-bar" id="pa-barnow"></div></div>`+
    `<div class="pa-brow"><span class="pa-bl" id="pa-avglab" tabindex="0"></span><div class="pa-barwrap"><div class="pa-bar" id="pa-baravg"></div><i class="pa-ghost" id="pa-ghost"></i></div></div>`+
    `<div class="pa-gleg"><i class="pa-ghostsw"></i><span>${tx('Dashed line: the previous setting’s average','虚线：上一个设置的平均值')}</span></div></div>`+
    `<button type="button" class="linkbtn pa-done" id="pa-done" data-act="to-card" hidden></button></div>`+
    `<div class="pa-mhead" id="pa-mhead">${tx(`Whole run (all ${NREQ} requests)`,`整次运行（全部 ${NREQ} 个请求）`)}</div>`+
    `<div class="metrics pa-whole" id="pa-metrics">${MET.map(m=>`<div class="metric" data-k="${m.k}"${m.tip?` data-tip="${m.tip}"`:''} tabindex="0"><div class="k">${m.label()}</div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('')}</div>`+
    `<details class="math" id="pa-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="pa-mathbody"></div></details></div>`+
    controlsHTML()+`</div>`+connectHTML();
}
function connectHTML(){
  const code=`class PagedKVCache:
    """A free list of fixed-size blocks, one block table per request."""
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free = list(range(num_blocks))  # FIFO, simpler than vLLM's
        self.tables = {}   # request -> physical block ids, in order
        self.lengths = {}  # request -> tokens stored

    def admit(self, req, prompt_len):
        need = -(-prompt_len // self.block_size)  # ceil division
        if need > len(self.free):
            return False  # not enough blocks: keep it waiting
        self.tables[req] = [self.free.pop(0) for _ in range(need)]
        self.lengths[req] = prompt_len
        return True

    def append_token(self, req):
        full = len(self.tables[req]) * self.block_size
        if self.lengths[req] == full:  # a new block only when needed
            if not self.free:
                return False  # pool full: the scheduler preempts one
            self.tables[req].append(self.free.pop(0))
        self.lengths[req] += 1
        return True

    def where(self, req, pos):
        """Where token \`pos\` lives in the pool, like a page table."""
        block = self.tables[req][pos // self.block_size]
        return block * self.block_size + pos % self.block_size

    def finish(self, req):
        self.free.extend(reversed(self.tables.pop(req)))  # to the back
        del self.lengths[req]

kv = PagedKVCache(num_blocks=768)  # 768 x 16: room for 12,288 tokens
kv.admit("a", 40); kv.admit("b", 20); kv.admit("c", 10)  # 0-2, 3-4, 5
kv.finish("a")
for _ in range(20):
    kv.append_token("b")  # b passes 32 tokens and gets a third block
print(kv.tables["b"], kv.where("b", 39))  # [3, 4, 6] 103`;
  const gh=SRC_VLLM;
  return `<section class="wrap connect" aria-labelledby="pa-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="pa-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
    `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('PagedAttention’s bookkeeping needs only two things: a list of free blocks and one block table per request.','PagedAttention 的记账只需要两样东西：一个空闲 block 列表，加上每个请求一张 block table。')}</p><pre class="code pa-code" id="pa-code">${esc(code)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
    `<article class="panel c-card pa-prod"><h3>${A.t('cProd')}</h3><ul>`+
      `<li>${gh('vllm/v1/core/block_pool.py')}${tx(': hands out and takes back blocks.','：分配和回收 block。')}</li>`+
      `<li>${gh('vllm/v1/core/kv_cache_utils.py')}${tx(': the queue of free blocks.','：空闲 block 队列。')}</li>`+
      `<li>${gh('vllm/v1/core/kv_cache_manager.py')}${tx(': <code>allocate_slots</code>, which the scheduler calls for each request it schedules.','：<code>allocate_slots</code>，调度器每调度一个请求都会调用它。')}</li>`+
      `<li>${gh('vllm/v1/core/sched/scheduler.py')}${tx(': when allocation fails, it pauses the last running request.','：分配失败时，暂停最后一个在跑的请求。')}</li>`+
      `<li>${gh('vllm/config/cache.py')}${tx(': the default block size of 16.','：默认 block 大小 16。')}</li></ul></article>`+
    `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs"><li><a href="https://arxiv.org/abs/2309.06180" target="_blank" rel="noopener">Efficient Memory Management for Large Language Model Serving with PagedAttention</a><span>${tx('Kwon et al., SOSP 2023','Kwon 等，SOSP 2023')}</span></li>`+
      `<li><a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noopener">Orca: A Distributed Serving System for Transformer-Based Generative Models</a><span>${tx('Yu et al., OSDI 2022 (one of the systems the vLLM paper compares against)','Yu 等，OSDI 2022（vLLM 论文对比的系统之一）')}</span></li></ul></article>`+
    `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="pa-drillq"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-show">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="pa-drilla" hidden></div></article>`+
    `</div></section>`;
}

/* ================= rail ================= */
function railGuided(){
  const s=STEPS[step];
  return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,
    nextDisabled:(!!s.expect&&!s.expect())||(!!s.gate&&!s.gate()),skip:!!s.skip&&!pred&&!seen});
}
function railSandbox(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
    `<p class="small">${tx(`Pick a scenario and see how much of the pool stores tokens while requests wait. A scenario sets only the answer lengths and the answer cap; pick the allocation strategy below.`,
                          `选一个场景，看看有请求排队时，显存池有多少存着 token。每个场景只定回答长度和回答上限，分配方式在下面选。`)}</p>`+
    `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((pr,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(pr.t))}</b><span data-pa-pd="${i}">${esc(presetDesc(pr))}</span></button>`).join('')}</div>`+
    `<p class="small pa-sbnote" id="pa-sbnote" aria-live="polite"></p>`+
    segStrat('pa-ic-strat',true)+`<p class="small">${tx(`Change the block size or anything else under ${J('controls','Settings')}.`,`block 大小和其他设置，在“${J('controls','自由调参')}”里改。`)}</p>`+A.shell.toChallenge()+`</div>`;
}
function railChallenge(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met both goals, and why.','提交后会告诉你两个目标是否都达到，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Make at least ${pc(GOAL_UTIL,0)} of the pool store tokens on average while requests wait, with block tables averaging ${GOAL_TABLE} entries or fewer per request`,`让显存池在有请求排队时平均至少 ${pc(GOAL_UTIL,0)} 存着 token，同时每个请求的 block table 平均不超过 ${GOAL_TABLE} 条`)}</h3>`+
    `<p>${tx(`These stay fixed: Llama-3.1-8B, a pool with room for ${int(POOL)} tokens, the same ${NREQ} requests. You can change only these two:`,`这些保持不变：Llama-3.1-8B，能放 ${int(POOL)} 个 token 的显存池，还是那 ${NREQ} 个请求。你只能改下面两项：`)}</p>`+
    segStrat('pa-ic-strat',true)+segB('pa-ic-B',true,`<span id="pa-ic-bnote"></span>`)+
    `<p class="pa-why">${tx(`Why cap the block table: each entry is a separate piece of memory that attention has to look up and read. This lab does not time attention, so the cap stands in for the cost of small blocks; ${GOAL_TABLE} is this challenge’s own threshold.`,`为什么限制 block table：每一条都是 attention 计算时要单独查找和读取的一块显存。这个 Lab 不计算 attention 的时间，所以用这个上限代表小 block 的代价；${GOAL_TABLE} 条是本题自己定的门槛。`)}</p>`+
    `<div class="goal mono" id="pa-g1" data-tipx="${esc(tx('This is the share of the pool that stores tokens, averaged over the decode steps in which requests wait.','这一项是有请求在排队的 decode 步里，显存池平均存着 token 的比例。'))}" tabindex="0"><span>${tx('Stores tokens while requests wait: ','排队时存着 token：')}<b id="pa-g1v"></b>${tx(` (goal: at least ${pc(GOAL_UTIL,0)})`,`（目标：至少 ${pc(GOAL_UTIL,0)}）`)}</span></div><div class="progress" id="pa-p1" aria-hidden="true"><i></i></div>`+
    `<div class="goal mono pa-goal2" id="pa-g2" data-tipx="${esc(tx('This is the average number of block-table entries per request when it finishes. Reserving by length has no block table, so this goal is met trivially.','这一项是请求结束时 block table 的平均条数。按长度预留时没有 block table，这一项自然达标。'))}" tabindex="0"><span>${tx('Block table, average: ','block table 平均：')}<b id="pa-g2v"></b>${tx(` (goal: at most ${GOAL_TABLE} entries)`,`（目标：最多 ${GOAL_TABLE} 条）`)}</span></div><div class="progress" id="pa-p2" aria-hidden="true"><i></i></div>`+
    submit+`<div id="pa-chres" aria-live="polite"></div></div>`;
}
function renderRail(){
  const rail=$('#pa-rail',host); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';   // the next lab in every mode (R3.11)
  rail.innerHTML=(mode==='guided'?railGuided():(mode==='sandbox'?railSandbox():railChallenge()+(A.ui.exits?A.ui.exits(ID,EXITS()):'')))+nl;   // the challenge's card, then the capability layer, then the next lab
  A.ui.initRanges(rail);
  updateRail();
}
function cue(){$$('.cue',host).forEach(el=>el.classList.remove('cue')); if(mode!=='guided') return;
  const id=STEPS[step].target; const el=id&&document.getElementById(id); if(el){void el.offsetWidth; el.classList.add('cue');}}
function enterStep(i){
  stop(); step=Math.max(0,Math.min(STEPS.length-1,i));
  if(mode==='guided'&&step>=2) seen=true;                  // part 3 shows the run: the prediction is final from here on
  if(step===4) sweepWarm();
  const e=STEPS[step].enter?STEPS[step].enter():null;
  hl='all';
  if(e){S={...e.s}; t=e.t||0; if(e.hl) hl=e.hl;}
  if(mode==='guided'&&step===2&&!observed[2]&&watched.has(runKey(S))) observed[2]=true;   // R3.20: the same run already watched to the end: its result at once, the gate passed
  hoverOwner=-1; renderRail(); update(); clearBadges(); cue();
  const dn=$('#pa-done',host); if(dn) dn.hidden=true;
}
/* after a click that changes the step card or the mode: bring its top edge into view */
const revealStep=()=>A.ui.reveal($('#pa-rail .step.current',host)), revealRail=()=>A.ui.reveal($('#pa-rail',host));

/* ================= update: everything that depends on the settings ================= */
/* why a challenge setup misses (or meets) the goals, in words */
function chWhy(r,c){
  if(c.met) return tx(`while requests wait, ${pc(r.util)} of the pool stores tokens, and block tables average ${n1(r.table)} entries per request`,`排队时显存池 ${pc(r.util)} 存着 token，每个请求的 block table 平均 ${n1(r.table)} 条`);
  const w=[]; if(!c.utilOk) w.push(tx(`while requests wait, only ${pc(r.util)} of the pool stores tokens, under ${pc(GOAL_UTIL,0)}`,`排队时显存池只有 ${pc(r.util)} 存着 token，不到 ${pc(GOAL_UTIL,0)}`));
  if(!c.tableOk) w.push(tx(`block tables average ${n1(r.table)} entries per request, over ${GOAL_TABLE}`,`每个请求的 block table 平均 ${n1(r.table)} 条，超过 ${GOAL_TABLE} 条`));
  return w.join(tx('; ','；'));}
/* the challenge result, short enough for the mini result bar on a phone */
function chShort(r,c){
  const u=c.utilOk?tx(`${pc(r.util)} stores tokens while requests wait`,`排队时 ${pc(r.util)} 存着 token`):tx(`${pc(r.util)} stores tokens while requests wait, under ${pc(GOAL_UTIL,0)}`,`排队时 ${pc(r.util)} 存着 token，不到 ${pc(GOAL_UTIL,0)}`);
  if(r.table==null) return u;
  const tb=c.tableOk?tx(`block tables average ${n1(r.table)} entries`,`block table 平均 ${n1(r.table)} 条`):tx(`block tables average ${n1(r.table)} entries, over ${GOAL_TABLE}`,`block table 平均 ${n1(r.table)} 条，超过 ${GOAL_TABLE} 条`);
  return c.met||(!c.utilOk&&!c.tableOk)?u+tx('; ','，')+tb:(c.utilOk?tb:u);}
function status(r){
  if(masked()) return {cls:'neutral',text:pred?tx('Result in part 3','结果在第 3 部分'):tx('Predict first','先预测'),
    tip:pred?tx('The whole-run result stays hidden until you watch the run in part 3.','整次运行的结果先隐藏，到第 3 部分看运行时再显示。'):tx('The whole-run result stays hidden until you answer part 2.','整次运行的结果在第 2 部分作答前先隐藏。')};
  if(pending()) return {cls:'neutral',text:tx('Running','运行中'),tip:tx('The whole-run averages appear after the last decode step.','整次运行的平均值在最后一个 decode 步之后显示。')};
  if(mode==='challenge'){const c=chEval(chOpt()), why=chWhy(r,c);
    return c.met?{cls:'ok',text:tx('Goal met','达成目标'),tip:chTip(),why}:{cls:'bad',text:tx(`Not yet: ${why}`,`未达标：${why}`),short:tx('Not yet','未达标'),why,tip:chTip()};}
  return {cls:r.util>=0.9?'ok':(r.util<0.5?'bad':'neutral'),text:tx(`While requests wait: ${pc(r.util)} stores tokens on average`,`排队时平均 ${pc(r.util)} 存着 token`),short:tx('While requests wait','排队时平均')};
}
/* the badge: in part 3 it follows the decode step until the run ends, so it is drawn with every frame */
function drawStatus(r){const st=status(r), se=$('#pa-status',host); if(!se) return;
  const cls='status '+st.cls; if(se.className!==cls) se.className=cls; if(se.textContent!==st.text) se.textContent=st.text;
  if(st.tip){if(se.dataset.tipx!==st.tip){se.removeAttribute('data-tip'); se.setAttribute('data-tipx',st.tip);}}
  else if(se.dataset.tip!=='pa_status'){se.removeAttribute('data-tipx'); se.setAttribute('data-tip','pa_status');}}
const chTip=()=>tx(`Two goals: while requests wait, at least ${pc(GOAL_UTIL,0)} of the pool stores tokens on average; block tables average ${GOAL_TABLE} entries or fewer.`,`两个目标：有请求排队时，显存池平均至少 ${pc(GOAL_UTIL,0)} 存着 token；每个请求的 block table 平均不超过 ${GOAL_TABLE} 条。`);
const chOpt=()=>S.strat==='paged'?{strat:'paged',B:S.B}:{strat:S.strat};
function segSync(){
  $$('.seg-ctl[data-key]',host).forEach(g=>{const k=g.dataset.key; A.ui.setPressed(g,String(k==='speed'?speed:S[k]));});
}
function notes(r){
  const set=(id,h)=>{const el=$('#'+id,host); if(el&&el._h!==h){el.innerHTML=h; el._h=h;}}, lim=int(S.lim);
  set('pa-strat-note',S.strat==='max'?tx(`Each request gets one contiguous region with room for its prompt plus ${lim} tokens (the answer cap). Servers did this before PagedAttention because answer lengths are unknown in advance.`,`每个请求一整块连续空间，放得下 prompt 加 ${lim} 个 token（回答上限）。PagedAttention 之前的服务都这样做，因为回答长度事先不知道。`)
    :(S.strat==='exact'?tx('Each request gets one region exactly as long as its final length. No real server can do this, because answer lengths are unknown in advance; it shows what contiguous allocation still wastes even with that knowledge.','每个请求一整块，大小正好等于它的最终长度。真实服务做不到，因为事先不知道回答多长；它展示的是即使知道长度，连续分配还会浪费多少。')
    :tx('Each time a request fills a block, it takes another from anywhere in the pool; its block table records where they are.','请求每写满一个 block，就从显存池任何位置再取一个；它的 block table 记下这些 block 在哪。')));
  set('pa-b-note',tx(`Applies to paging only${S.strat==='paged'?'':', so it has no effect right now'}. vLLM defaults to 16; its main GPU attention backends accept only multiples of 16, so 1 and 8 are here for comparison.`,
    `只对分页起作用${S.strat==='paged'?'':'，现在不起作用'}。vLLM 默认 16；它的主要 GPU attention 后端只接受 16 的倍数，所以 1 和 8 只用来对比。`));
  set('pa-mu-note',tx(`Answer lengths are drawn at random with this mean: most are short, a few are long, and nobody can tell which in advance. These 40 actually average ${int(Math.round(r.meanO))} tokens.`,
    `回答长度按这个均值随机抽取：多数很短，少数很长，事先分不出是哪个。这 40 个实际平均 ${int(Math.round(r.meanO))} 个 token。`));
  set('pa-lim-note',tx(`The most tokens a request may generate (max_tokens in vLLM). Reserving the maximum, each request holds room for its prompt plus this many tokens. ${r.capped?`${r.capped} of these 40 ${r.capped===1?'reaches':'reach'} the cap.`:'None of these 40 answers reaches the cap.'}`,
    `一个请求最多生成多少 token（vLLM 里叫 max_tokens）。按最大长度预留时，每个请求要占 prompt 加上这么多 token 的空间。${r.capped?`这 40 个里有 ${r.capped} 个写到上限。`:'这 40 个回答都没写到上限。'}`));
  const ic=$('#pa-ic-bnote',host); if(ic){const h=S.strat==='paged'?'':tx('Block size applies only to paging.','block 大小只对分页起作用。'); if(ic._h!==h){ic.innerHTML=h; ic._h=h;}}
}
/* a value as the card displays it ("1,232.5", "24.8%", "none" -> NaN), so a badge reproduces from the shown numbers */
const shownNum=s=>{const x=parseFloat(String(s).replace(/[^0-9.\-]/g,'')); return isFinite(x)?x:NaN;};
/* entering a part or switching mode clears the change badges and the dashed previous-average line, even when the run is unchanged */
function clearBadges(){if(!host) return; $$('#pa-metrics .metric .d',host).forEach(d=>{clearTimeout(d._t); d.textContent=''; d.classList.remove('on');});
  const gh=$('#pa-ghost',host); if(gh){gh.style.transition='none'; gh.style.opacity='0';}}
function drawMetrics(r){
  const hostM=$('#pa-metrics',host); if(!hostM) return;
  const hide=hidden();
  MET.forEach(m=>{const el=hostM.querySelector(`[data-k="${m.k}"]`), mk=hide&&m.k!=='table'; el.querySelector('.v').textContent=mk?'?':m.v(r);
    if(m.jump){el.setAttribute('data-jump',m.jump);}
    // a change badge only after the learner's own change; entering a part, switching mode or Show me clears it
    if(prevRun&&prevRun!==r){const d=el.querySelector('.d'), dt=userAct&&!mk?A.fmt.delta(shownNum(m.v(prevRun)),shownNum(m.v(r)),{dp:m.dp,pts:!!m.pts}):'';
      clearTimeout(d._t); d.textContent=dt; d.classList.toggle('on',!!dt); if(dt) d._t=setTimeout(()=>d.classList.remove('on'),2600);}});
  hostM.querySelector('[data-k="steps"]').setAttribute('data-tipx',tx(`This is the number of decode steps until all ${NREQ} requests finish. Every decode step is assumed to take the same time (see the notes at the end of the page for why).`,`这个数是 ${NREQ} 个请求全部完成用的 decode 步数。这里假设每个 decode 步耗时相同（原因见页末的“简化、假设和来源”）。`));
}
/* H1/H2 (the owner, 2026-10-09): Show the math as short blocks, like a page of a good textbook. Each block has one lead line in words,
   then a small table: the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the
   result, each on its own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>g.terms.length)), eq=`<td class="pa-meq">${MX('=')}</td>`;
  const cells=(a,f)=>a.map((x,i)=>`<td${i===a.length-1&&a.length<n?` colspan="${n-i}"`:''}>${f(x)}</td>`).join('');
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?cells(r.c,x=>x):`<td colspan="${n}" class="pa-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="pa-mname"><th scope="row">${g.name}</th>${eq}${cells(g.terms,MX)}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="pa-mb">${lead?`<p class="pa-mlead">${lead}</p>`:''}<div class="pa-mfx"><table class="pa-mt">${body}</table></div>${note?`<p class="pa-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.pa-mt'); if(!tb) return; const e0=tb.querySelector('td.pa-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('pa-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('pa-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="pa-mc${r?'':' pa-mc0'}">${x}</span>`;
    h+=`<tr class="pa-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="pa-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="pa-mband${g.tall?' pa-mtall':''}"><td class="pa-meq"><span class="pa-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="pa-mfl">${units.map(u=>`<span class="pa-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="pa-meq">${E}</td><td class="pa-mres">${r.l?`<span class="pa-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='pa-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.pa-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.pa-mfx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
function mathHTML(r){
  const T=(c,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${c}">${x}</span>`, q=r.reqs[0], par=s=>tx(`(${s})`,`（${s}）`), lim=int(S.lim);
  const len=tx('\\text{length}','\\text{长度}'), cap=tx('\\text{answer cap}','\\text{回答上限}'), ans=tx('\\text{answer length}','\\text{回答长度}');
  let h=mathBlock(tx('How full the pool is while requests wait: the tokens it stores, out of the tokens it can hold.','有请求排队时，显存池有多满：存着的 token 占它能放的 token 的比例。'),
    [{name:tx('Share that stores tokens while requests wait','排队时存着 token 的比例'),terms:[tx('\\frac{\\text{tokens stored on average}}{\\text{tokens the pool holds}}','\\frac{\\text{平均存着的 token 数}}{\\text{显存池能放的 token 数}}')],
      rows:[{c:[`${n1(r.tokAvg)} ÷ ${T('pool',int(POOL))}`]},{r:`<b>${pc(r.util)}</b>`}]}],
    `<span class="muted">${par(tx(`averaged over the ${int(r.busyN)} decode steps in which a request was waiting`,`在有请求排队的 ${int(r.busyN)} 个 decode 步里取平均`))}</span>`);
  if(S.strat==='max') h+=mathBlock(tx('Reserving the maximum: from the start, each request holds room for its prompt plus the answer cap.','按最大长度预留：每个请求一开始就占住 prompt 加回答上限的空间。'),
    [{name:tx('Room per request','每个请求预留'),terms:['\\text{prompt}','+',cap],rows:[{l:tx('request 1','请求 1'),c:[int(q.p),'+',T('lim',lim)]},{r:`<b>${int(q.p+S.lim)}</b> ${tx('tokens (request 1)','个 token 的空间（请求 1）')}`}]},
     {name:tx('Never used','到结束也没用上'),terms:[cap,'-',ans],rows:[{l:tx('request 1','请求 1'),c:[T('lim',lim),'−',int(q.o)]},{r:`<b>${int(S.lim-q.o)}</b> ${tx(`tokens of room (request 1); ${n1(r.unused)} on average over the ${NREQ} requests`,`个 token 的空间（请求 1）；${NREQ} 个请求平均 ${n1(r.unused)} 个`)}`}]}]);
  else if(S.strat==='exact') h+=mathBlock(tx('Reserving the exact length: each request holds room for its prompt plus its own answer.','按实际长度预留：每个请求占住 prompt 加上它自己回答的空间。'),
    [{name:tx('Room per request','每个请求预留'),terms:['\\text{prompt}','+',ans],rows:[{l:tx('request 1','请求 1'),c:[int(q.p),'+',int(q.o)]},{r:`<b>${int(q.F)}</b> ${tx('tokens (request 1), only if the answer length is known in advance','个 token 的空间（请求 1），需要事先知道回答长度')}`}]}]);
  else {const nb=Math.ceil(q.F/S.B);
    h+=mathBlock(tx('Paging: a request takes blocks of a fixed size, one more each time its last block fills. ⌈ ⌉ rounds up to a whole block.','分页：请求按固定大小的 block 占空间，最后一个 block 满了才再要一个。⌈ ⌉ 表示向上取整到整块。'),
      [{name:tx('Blocks per request','每个请求的 block 数'),terms:[`⌈${len} \\div ${tx('\\text{block size}','\\text{block 大小}')}⌉`],rows:[{l:tx('request 1','请求 1'),c:[`⌈${int(q.F)} ÷ ${T('B',S.B)}⌉`]},{r:`<b>${int(nb)}</b> ${tx(`blocks (request 1); ${n1(r.table)} on average over the ${NREQ} requests`,`个（请求 1）；${NREQ} 个请求平均 ${n1(r.table)} 个`)}`}]},
       {name:tx('Room never used','没用上的空间'),terms:[tx('\\text{blocks}','\\text{block 数}'),'\\times',tx('\\text{block size}','\\text{block 大小}'),'-',len],rows:[{l:tx('request 1','请求 1'),c:[int(nb),'×',T('B',S.B),'−',int(q.F)]},{r:`<b>${int(nb*S.B-q.F)}</b> ${tx('tokens of room, fewer than ','个 token 的空间，少于 ')}${T('B',S.B)}${tx('',' 个')}`}]}]);}
  h+=mathBlock(tx('How many tokens a decode step makes on average. The more requests fit in the pool, the more tokens each step makes.','一个 decode 步平均生成多少 token。显存池里放得下的请求越多，每步生成的 token 越多。'),
    [{name:tx('Tokens generated per decode step','平均每个 decode 步生成的 token'),terms:[tx('\\text{answer tokens}','\\text{回答 token 总数}'),'\\div',tx('\\text{decode steps}','\\text{decode 步数}')],rows:[{c:[int(r.sumO),'÷',int(r.T)]},{r:`<b>${n1(r.tps)}</b>`}]}]);
  h+=`<p class="muted">${tx('Workload: ','负载：')}${T('n',tx(`${NREQ} requests`,`${NREQ} 个请求`))}${tx(', answer lengths with ','，回答长度')}${T('mu',tx('mean '+int(S.mu),'均值 '+int(S.mu)))}${tx(', strategy “','，分配方式“')}${T('strat',cfgName(S))}${tx('”.','”。')}</p>`;
  return h;
}
function update(){
  if(!host) return;
  const r=run(); t=Math.max(0,Math.min(r.T-1,t));
  $('#pa-sub',host).textContent=cfgName(S);
  const wl=$('#pa-wl',host);
  wl.textContent=tx(`${NREQ} requests: prompts of ${PMIN} to ${PMAX} tokens; answers are mostly short and a few long. These ${NREQ} answers actually average ${int(Math.round(r.meanO))} tokens; the longest is ${int(maxO(r))} (the cap is ${int(S.lim)}).`,`${NREQ} 个请求：prompt ${PMIN} 到 ${PMAX} 个 token；回答多数很短、少数很长，这 ${NREQ} 个实际平均 ${int(Math.round(r.meanO))} 个 token，最长 ${int(maxO(r))} 个（上限 ${int(S.lim)}）。`);
  wl.setAttribute('data-tipx',tx(`Answer lengths are drawn from an exponential distribution with mean ${int(S.mu)} and cut at the cap. The random seed is fixed, so every replay serves the same ${NREQ} requests.`,`回答长度按均值 ${int(S.mu)} 的指数分布随机抽取，并截断在上限。随机种子固定，所以每次重放的都是同样的 ${NREQ} 个请求。`));
  wl.setAttribute('data-jump','wl');
  $$('#pa-legend [data-jump]',host).forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.jump==='hl-'+hl)));
  $('#pa-lg-start',host).hidden=S.strat==='paged';
  $$('input[data-pa="time"]',host).forEach(i=>{i.max=r.T-1;});
  segSync(); notes(r); drawMetrics(r);
  $('#pa-mathbody',host).innerHTML=hidden()?`<p class="muted">${pending()?tx('Shown after the run.','跑完后显示。'):(pred?tx('Shown in part 3.','第 3 部分显示。'):tx('Shown after you predict.','预测后显示。'))}</p>`:mathHTML(r); fitMath($('#pa-mathbody',host));
  const adv=$('#pa-advbody',host); if(adv&&!adv._done){adv.innerHTML=advHTML(); adv._done=true;}
  // the whole-run average bar, with a dashed ghost of the previous setting's average
  const bavg=$('#pa-baravg',host); bavg.style.visibility=hidden()?'hidden':''; drawShareBar(bavg,r.shares);
  $('#pa-avglab',host).textContent=hidden()?tx(`While requests wait: ${maskNote()}`,`排队时平均：${maskNote()}`):tx(`While requests wait: ${pc(r.util)} on average`,`排队时平均：${pc(r.util)}`);
  $('#pa-avglab',host).setAttribute('data-tipx',tx(`This average is taken over the ${int(r.busyN)} decode steps in which a request was waiting.`,`这个平均值是在有请求排队的 ${int(r.busyN)} 个 decode 步里取的。`));
  const gh=$('#pa-ghost',host);
  if(userAct&&prevRun&&prevRun!==r&&Math.abs(prevRun.util-r.util)>0.004&&!A.reduceMotion()&&!hidden()){gh.style.transition='none'; gh.style.left=(prevRun.util*100)+'%'; gh.style.opacity='1'; void gh.offsetWidth; gh.style.transition='opacity 1.4s ease-out'; gh.style.opacity='0';}
  updateRail();
  drawFrame();
  A.hash.write(token());
  if(prevRun!==r&&!hidden()){A.announce(tx(`${cfgName(S)}: while requests wait, ${pc(r.util)} of the pool stores tokens on average and ${n1(r.running)} requests run at once. All ${NREQ} finish after ${stepsTxt(r.T)}.`,`${cfgName(S)}：有请求排队时，显存池平均 ${pc(r.util)} 存着 token，同时跑 ${n1(r.running)} 个请求；${NREQ} 个请求全部完成用了 ${stepsTxt(r.T)}。`));}
  prevRun=r;
  A.dock.refresh();
}
function drawShareBar(el,sh){
  if(!el) return; if(!el._e){el.innerHTML=STATE_KEYS.map(k=>`<div class="pa-sg pa-f-${k}"></div>`).join(''); el._e=STATE_KEYS.map((k,i)=>el.children[i]);}
  let x=0; STATE_KEYS.forEach((k,i)=>{const w=sh[k]*100, sg=el._e[i]; sg.style.left=x+'%'; sg.style.width=w+'%'; x+=w;});
}
function updateRail(){
  const r=run();
  // inline controls mirror the state
  segSync();
  if(mode==='guided'){
    const s=STEPS[step], box=$('#pa-rail [data-obs]',host);
    $$('#pa-rail [data-lever]',host).forEach(b=>{const v=b.dataset.lever, grp=v.slice(0,2), id=v.slice(3);
      if(grp==='hl'){b.setAttribute('aria-pressed',String(id===hl)); b.querySelector('.lv-r').textContent=hlTried.has(id)?lvResultHl(id):'';}
      else {const cur=ST_LEVERS[stLeverNow()]; b.setAttribute('aria-pressed',String(!!cur&&cur.v===id)); b.querySelector('.lv-r').textContent=stTried.has(id)?lvResultSt(id):'';}});
    if(step===5&&S.strat==='paged'&&S.B!==16) observed[5]=true;
    const tr=$('#pa-rail [data-pa-tried]',host); if(tr){const h=triedHTML(); if(tr._h!==h){tr.innerHTML=h; tr._h=h;}}
    if(box){const show=!!s.obs&&(!s.expect||s.expect());
      box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${step===3||step===4?tx('Result','结果'):tx('Observed','观察到')}</div>`+s.obs();
        if(box._html!==html){const again=box._step===step&&box._html!=null; box.innerHTML=html; box._html=html; box._step=step; if(again) A.ui.obsBump(box);}}}
    $$('#pa-rail [data-nav="next"]',host).forEach(nb=>{nb.disabled=(!!s.expect&&!s.expect())||(!!s.gate&&!s.gate());});
    const sm=$('#pa-rail [data-nav="showme"]',host); if(sm) sm.hidden=!!(s.expect&&s.expect());
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('#pa-rail [data-preset]',host).forEach(b=>{const i=+b.dataset.preset; b.setAttribute('aria-pressed',String(i===cur)); const d=b.querySelector('[data-pa-pd]'); if(d) d.textContent=presetDesc(PRESETS[i]);});
    const n=$('#pa-sbnote',host);
    if(n){const h=scenarioLine(r,cur,sameNote&&sameNote.key===runKey(S)?sameNote:null); if(n.textContent!==h) n.textContent=h;}
  } else {
    const c=chEval(chOpt()), g1=$('#pa-g1v',host); if(!g1) return;
    g1.textContent=pc(r.util); $('#pa-p1 i',host).style.width=Math.min(100,r.util/GOAL_UTIL*100)+'%';
    $('#pa-g1',host).classList.toggle('met',c.utilOk); $('#pa-p1',host).classList.toggle('met',c.utilOk);
    const g2=$('#pa-g2v',host); g2.textContent=r.table==null?tx('none','无'):n1(r.table);
    const p2=$('#pa-p2',host); p2.firstChild.style.width=(r.table==null?100:Math.min(100,GOAL_TABLE/r.table*100))+'%';
    $('#pa-g2',host).classList.toggle('met',c.tableOk); p2.classList.toggle('met',c.tableOk);
    if(chSubmitted&&c.met) chReveal=true;
    const res=$('#pa-chres',host), h=chResultHTML(); if(res._h!==h){res.innerHTML=h; res._h=h;}
  }
}

/* ================= challenge feedback ================= */
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(chOpt()), best=chBest(), r=me.r;
  let h;
  if(me.met&&best&&me.o.strat===best.o.strat&&(me.o.B||16)===(best.o.B||16))
    h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: while requests wait, ${pc(r.util)} of the pool stores tokens, and block tables average ${n1(r.table)} entries.</b> And you changed as little as possible: only the strategy, keeping vLLM’s default block size.</p>`,
         `<p class="okline" style="margin-top:12px"><b>达成目标：排队时显存池 ${pc(r.util)} 存着 token，block table 平均 ${n1(r.table)} 条。</b>而且改动最少：只换了分配方式，block 大小保留 vLLM 的默认值。</p>`);
  else if(me.met)   // the goal line keeps its two values; the smaller change is said in words (R6.20)
    h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: while requests wait, ${pc(r.util)} of the pool stores tokens, and block tables average ${n1(r.table)} entries.</b> You could change less: paging with vLLM’s default block size already meets both goals.</p>`,
         `<p class="okline" style="margin-top:12px"><b>达成目标：排队时显存池 ${pc(r.util)} 存着 token，block table 平均 ${n1(r.table)} 条。</b>其实改得更少也行：分页加 vLLM 默认的 block 大小，就能达到两个目标。</p>`);
  else {
    let hint;
    if(me.o.strat!=='paged') hint=tx(`Hint: with one region per request, which kinds of room fill most of the pool? Look at the ${J('avgbar','“While requests wait” bar')} under the pool and think about where each kind of waste comes from.`,`提示：每个请求一整块时，显存池大部分被哪几种空间占着？看显存池下面的${J('avgbar','“排队时平均”条')}，想想每种浪费是怎么来的。`);
    else if(!me.tableOk) hint=tx(`Hint: requests here end with about ${int(Math.round(r.meanF))} tokens on average. How many tokens must one block hold to keep a block table at ${GOAL_TABLE} entries or fewer?`,`提示：这里的请求最后平均约 ${int(Math.round(r.meanF))} 个 token。一个 block 要放多少 token，block table 才能不超过 ${GOAL_TABLE} 条？`);
    else hint=tx(`Hint: with ${S.B}-token blocks, how much room does each request’s last block leave empty?`,`提示：${S.B} 个 token 一个 block 时，每个请求最后一个 block 平均空着多少空间？`);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet:</b> ${chWhy(r,me)}.</p><p class="small">${hint}</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：</b>${chWhy(r,me)}。</p><p class="small">${hint}</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  const all=CH_OPTS.map(chEval), ev=o=>all.find(x=>x.o.strat===o.strat&&(o.strat!=='paged'||x.o.B===o.B));
  const b=ev({strat:'max'}), x=ev({strat:'exact'}), far=GOAL_UTIL-x.util>0.15;
  const paged=all.filter(e=>e.o.strat==='paged'), okB=paged.filter(e=>e.met).map(e=>e.o.B);
  const list=okB.length<2?String(okB[0]||''):okB.slice(0,-1).join(tx(', ','、'))+tx(' or ','、')+okB[okB.length-1];
  // the sizes on either side of the range that works, and why they miss
  const lo=okB.length?okB[0]:0, hi=okB.length?okB[okB.length-1]:0;
  const smallFail=okB.length&&paged.filter(e=>e.o.B<lo).every(e=>!e.tableOk), largeFail=okB.length&&paged.filter(e=>e.o.B>hi).every(e=>!e.utilOk);
  const label=e=>e.o.strat==='paged'?cfgName({strat:'paged',B:e.o.B}):L(STRAT_NAME[e.o.strat]);
  /* brief (the owner's review of batch 8): the chart keeps the choices that meet both goals, the pick and the initial setup, and says
     how many it left out; the block-table tag marks the pick when its table is the goal it misses */
  const mine=ev(chOpt()), shown=all.filter(e=>e.met||e===mine||e.o.strat==='max');
  const rows=all.map(e=>{const tags=[]; if(e.o.strat==='max') tags.push({text:tx('initial setup','初始设置')});
    if(e===mine&&e.table!=null&&!e.tableOk) tags.push({text:tx(`block table: ${n1(e.table)} entries, over ${GOAL_TABLE}`,`block table ${n1(e.table)} 条，超过 ${GOAL_TABLE} 条`)});
    if(e===mine) tags.push({text:tx('your pick','你的选择'),me:true});
    return {label:label(e),value:e.util*100,text:pc(e.util),ok:e.met,me:e===mine,tags};});
  h+=`<div class="pa-cmpwrap">`+A.ui.cmpChart({title:tx(`Share of the pool storing tokens while requests wait, Llama-3.1-8B with the same ${NREQ} requests (goal: at least ${pc(GOAL_UTIL,0)}, with block tables of at most ${GOAL_TABLE} entries)`,`Llama-3.1-8B 跑同样的 ${NREQ} 个请求、有请求排队时，显存池存着 token 的比例（目标是至少 ${pc(GOAL_UTIL,0)}，block table 不超过 ${GOAL_TABLE} 条）`),
    brief:true,hiddenText:n=>tx(`The other ${n} choices miss at least one goal and are not shown.`,`其余 ${n} 种选择至少差一个目标，图里没有列出。`),
    aria:tx('Share of the pool storing tokens: ','存着 token 的比例：')+shown.map(e=>label(e)+tx(' ','：')+pc(e.util)+(e.table!=null?tx(`, ${n1(e.table)} block-table entries`,`，block table ${n1(e.table)} 条`):'')+(e.met?tx(', meets both goals','，两个目标都达到'):'')).join(tx('; ','；')),
    rows,goal:GOAL_UTIL*100,goalLabel:tx(`goal: ${pc(GOAL_UTIL,0)}`,`目标 ${pc(GOAL_UTIL,0)}`),max:120,
    okText:tx('meets both goals','两个目标都达到'),noText:tx('misses at least one','至少差一个'),goalText:tx(`goal: ${pc(GOAL_UTIL,0)} stores tokens`,`目标：${pc(GOAL_UTIL,0)} 存着 token`)})+`</div>`;
  // baseline → effect → reason in words (R3.10, R6.20): the chart above holds the initial setup, the pick and the choices that pass;
  // the choices that fail are named in words (reserving the exact length: its share on hover)
  const range=okB.length>1&&okB.join()===BLOCKS.filter(B=>B>=lo&&B<=hi).join();
  const xTip=tx(`Reserving the exact length, only ${pc(x.util)} of the pool stores tokens while requests wait.`,`按实际长度预留，有请求排队时显存池只有 ${pc(x.util)} 存着 token。`);
  const passEn=!okB.length?'no block size meets both goals':(okB.length===1?`only ${okB[0]}-token blocks meet both goals`:(range?`blocks of ${lo} to ${hi} tokens meet both goals`:`blocks of ${list} tokens meet both goals`));
  const passZh=!okB.length?'没有哪个 block 大小能同时达到两个目标':(okB.length===1?`只有 ${okB[0]} 个 token 一个 block 能达到两个目标`:(range?`block 为 ${lo} 到 ${hi} 个 token 都能达到两个目标`:`block 为 ${list} 个 token 都能达到两个目标`));
  h+=tx(`<p class="small">Initial setup, reserving the maximum: only about ${pc0(b.util)} of the pool stores tokens while requests wait. ${A.ui.def('Reserving the exact length',xTip)} still falls ${far?'well ':''}short: every region holds room for tokens not generated yet, and the gaps between regions go unused.</p>`+
        `<p class="small">With paging, ${passEn}${smallFail&&largeFail&&okB.length?': smaller blocks need too many block-table entries, larger ones leave too much of their last block empty':''}.</p>`+
        `<p class="small muted">Before changing the block size on a real server, measure throughput on your own traffic: the paper’s two workloads had different best sizes, but 16 did well on both (§7.2).</p>`,
        `<p class="small">初始设置按最大长度预留：有请求排队时，显存池只有约 ${pc0(b.util)} 存着 token。${A.ui.def('按实际长度预留',xTip)}也离目标差${far?'得远':'一些'}，因为每一块都为还没生成的 token 占着空间，块间空隙也用不上。</p>`+
        `<p class="small">分页时，${passZh}。${smallFail&&largeFail&&okB.length?'原因：block 更小，block table 的条数就超过目标；block 更大，最后一个 block 空着的空间又太多。':''}</p>`+
        `<p class="small muted">在真实服务上改 block 大小之前，先用自己的流量测吞吐：论文里两组请求的最佳大小不一样，不过 16 在两组上都不错（§7.2）。</p>`);
  return h;
}

/* ================= the pool: canvas drawing, hover and keyboard ================= */
function colors(el){const cs=getComputedStyle(el), g=k=>cs.getPropertyValue(k).trim();
  const c={kv:g('--c-kv'),line:g('--line'),line2:g('--line-2'),surface:g('--surface'),surface2:g('--surface-2'),ink:g('--ink'),ink2:g('--ink-2'),ink3:g('--ink-3')};
  c.key=Object.values(c).join('|'); return c;}
function makePatterns(ctx,C,dpr){
  const tile=(draw)=>{const s=4, cv=document.createElement('canvas'); cv.width=Math.round(s*dpr); cv.height=Math.round(s*dpr); const x=cv.getContext('2d'); x.scale(dpr,dpr); draw(x,s);
    const p=ctx.createPattern(cv,'repeat'); try{if(p&&p.setTransform&&window.DOMMatrix) p.setTransform(new DOMMatrix([1/dpr,0,0,1/dpr,0,0]));}catch(e){} return p;};
  const intf=tile((x,s)=>{x.fillStyle=C.surface2; x.fillRect(0,0,s,s); x.strokeStyle=C.ink3; x.lineWidth=1; x.beginPath(); for(let k=-1;k<=1;k++){x.moveTo(k*s,s); x.lineTo(k*s+s,0);} x.stroke();});
  const ext=tile((x,s)=>{x.fillStyle=C.surface; x.fillRect(0,0,s,s); x.fillStyle=C.ink2; x.beginPath(); x.arc(s/2,s/2,0.85,0,Math.PI*2); x.fill();});
  return {intf,ext,key:C.key};
}
/* a cell is cs wide and ch tall: ch = cs × --pa-ch, which paged-attention.css sets to 0.8 on a desktop stage wide enough for 48 cells of
   about 14 px (polish round 5: the stage then fits 1440 × 900 without scrolling inside); 1 (square cells) everywhere else */
function cellRatio(cv){const q=parseFloat(getComputedStyle(cv).getPropertyValue('--pa-ch')); return q>=0.5&&q<=1?q:1;}
function geometry(W,q){const cols=W>=440?48:32, rows=NCELL/cols, gap=W>=440?2:1, cs=(W-(cols-1)*gap)/cols, ch=cs*(q||1); return {cols,rows,gap,cs,ch,W,H:Math.round(rows*ch+(rows-1)*gap)};}
function drawPool(r){
  const cv=$('#pa-cv',host); if(!cv) return;
  const W=cv.clientWidth||cv.parentNode.clientWidth; if(!W) return;
  const q=cellRatio(cv), G=geometry(W,q), dpr=window.devicePixelRatio||1;
  if(cv._w!==W||cv._dpr!==dpr||cv._q!==q){cv.width=Math.round(W*dpr); cv.height=Math.round(G.H*dpr); cv.style.height=G.H+'px'; cv._w=W; cv._dpr=dpr; cv._q=q; pats=null;}
  geo=G;
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,G.H);
  const C=colors(cv); if(!pats||pats.key!==C.key) pats=makePatterns(ctx,C,dpr);
  fillSlots(r,t);
  const fillOf=s=>s===TOK?C.kv:(s===RESV?C.line:(s===INTF?pats.intf:(s===EXT?pats.ext:C.surface)));
  const HLC={all:0,tok:TOK,resv:RESV,intf:INTF,ext:EXT,free:FREE}, hc=HLC[hl]||0, hlOn=hl!=='all';
  const {cols,cs,ch,gap}=G;
  for(let k=0;k<NCELL;k++){
    const x=(k%cols)*(cs+gap), y=((k/cols)|0)*(ch+gap), base=k*CELL;
    ctx.globalAlpha=hlOn&&hc!==FREE?0.35:1; ctx.fillStyle=C.surface; ctx.fillRect(x,y,cs,ch);
    let j=0, hit=false;
    while(j<CELL){const s=SL[base+j], o=OW[base+j]; let e=j+1; while(e<CELL&&SL[base+e]===s&&OW[base+e]===o) e++;
      if(hlOn&&s===hc) hit=true;
      if(s!==FREE){const dim=hlOn?(s!==hc):(hoverOwner>=0&&o!==hoverOwner);
        ctx.globalAlpha=dim?(hlOn?0.2:0.45):1; ctx.fillStyle=fillOf(s); ctx.fillRect(x+cs*j/CELL,y,cs*(e-j)/CELL,ch);}
      j=e;}
    ctx.globalAlpha=1;
    if(hit){ctx.strokeStyle=C.ink; ctx.lineWidth=1.2; ctx.strokeRect(x+0.6,y+0.6,cs-1.2,ch-1.2);}
    else {ctx.strokeStyle=C.line2; ctx.lineWidth=1; if(cs>=6) ctx.strokeRect(x+0.5,y+0.5,cs-1,ch-1);}
  }
  // region starts (contiguous) and the hovered request's cells
  ctx.globalAlpha=1;
  if(!r.paged){ctx.fillStyle=C.ink;
    for(let i=0;i<r.reqs.length;i++){if(r.a[i]<0||t<r.a[i]||t>r.f[i]) continue; const sl=r.start[i], k=(sl/CELL)|0, x=(k%cols)*(cs+gap)+cs*(sl%CELL)/CELL, y=((k/cols)|0)*(ch+gap);
      ctx.fillRect(x,y-1,Math.max(1.5,cs/10),ch+2);}}
  if(hoverOwner>=0){ctx.strokeStyle=C.ink; ctx.lineWidth=1.5;
    for(let k=0;k<NCELL;k++){let hit=false; for(let j=0;j<CELL;j++) if(OW[k*CELL+j]===hoverOwner){hit=true; break;} if(!hit) continue;
      const x=(k%cols)*(cs+gap), y=((k/cols)|0)*(ch+gap); ctx.strokeRect(x+0.75,y+0.75,cs-1.5,ch-1.5);}}
  if(kbd>=0){const x=(kbd%cols)*(cs+gap), y=((kbd/cols)|0)*(ch+gap); ctx.setLineDash([2,2]); ctx.strokeStyle=C.ink; ctx.lineWidth=1.5; ctx.strokeRect(x-1,y-1,cs+2,ch+2); ctx.setLineDash([]);}
  const mk=masked();
  cv.setAttribute('aria-label',tx(`KV cache pool at decode step ${int(t+1)} of ${int(r.T)}: ${mk?'':`${pc(r.agg.tok[t]/POOL)} stores tokens, `}${r.agg.run[t]} requests running, ${r.agg.wait[t]} waiting. Use the arrow keys to inspect cells.`,
    `KV cache 显存池，decode 第 ${int(t+1)} 步（共 ${int(r.T)} 步）：${mk?'':`${pc(r.agg.tok[t]/POOL)} 存着 token，`}${r.agg.run[t]} 个请求在跑，${r.agg.wait[t]} 个在排队。用方向键查看各格。`));
}
function nowShares(r){const g=r.agg, o={}; STATE_KEYS.forEach(k=>{o[k]=g[k][t]/POOL;}); return o;}
/* the live line: where the run is, in full sentences; at the last decode step, the requests still running finish in it */
function readout(r,share){const g=r.agg, n=g.run[t];
  if(t>=r.T-1) return tx(`Decode step ${int(t+1)} (the last): ${n===1?'the final request finishes':`the final ${n} requests finish`}.`,`decode 第 ${int(t+1)} 步（最后一步）：最后${n===1?'一':` ${n} `}个请求在这一步完成。`);
  const head=tx(`Decode step ${int(t+1)} of ${int(r.T)}: ${n} ${n===1?'request':'requests'} running, ${g.wait[t]} waiting, ${g.done[t]} finished.`,
            `decode 第 ${int(t+1)} 步（共 ${int(r.T)} 步）：${n} 个请求在跑，${g.wait[t]} 个在排队，${g.done[t]} 个已完成。`);
  return !share||masked()?head:head+tx(` Right now ${pc(g.tok[t]/POOL)} of the pool stores tokens.`,`此刻显存池有 ${pc(g.tok[t]/POOL)} 存着 token。`);}
function drawFrame(){
  if(!host) return;
  const r=run(); t=Math.max(0,Math.min(r.T-1,t));
  drawPool(r);
  const g=r.agg, hide=hidden();
  drawStatus(r);
  $$('input[data-pa="time"]',host).forEach(i=>{if(+i.value!==t) i.value=t; A.ui.setRange(i); i.setAttribute('aria-valuetext',tx(`decode step ${int(t+1)} of ${int(r.T)}`,`decode 第 ${int(t+1)} 步，共 ${int(r.T)} 步`));});
  $$('[data-pa="play"]',host).forEach(b=>{b.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放'); b.setAttribute('aria-pressed',String(playing));});
  const rdS=readout(r,false), rdC=readout(r,true);
  $$('[data-pa-out="tread"]',host).forEach(el=>{if(el.textContent!==rdS) el.textContent=rdS;});
  $$('[data-pa-out="live"]',host).forEach(el=>{if(el.textContent!==rdC) el.textContent=rdC;});
  $$('[data-pa-out="time"]',host).forEach(el=>{el.textContent=tx(`${int(t+1)} of ${int(r.T)}`,`${int(t+1)} / ${int(r.T)}`);});
  $('#pa-cellnote',host).textContent=tx(`Each cell is room for ${CELL} tokens.`,`每格代表 ${CELL} 个 token 的空间。`);
  const sh=nowShares(r), mk=masked(), bn=$('#pa-barnow',host); bn.style.visibility=mk?'hidden':''; drawShareBar(bn,sh);
  const nl=$('#pa-nowlab',host); nl.textContent=mk?tx(`Now: ${maskNote()}`,`此刻：${maskNote()}`):tx(`Now: ${pc(sh.tok)} stores tokens`,`此刻：${pc(sh.tok)} 存着 token`);
  nl.setAttribute('data-tipx',mk?(pred?tx('This bar shows how the pool is split at the current decode step. It appears in part 3, with the run.','这一条显示当前 decode 步显存池里各种空间的比例，到第 3 部分随运行一起显示。')
      :tx('This bar shows how the pool is split at the current decode step. It appears after you predict.','这一条显示当前 decode 步显存池里各种空间的比例，你预测之后才显示。'))
    :tx(`The pool at decode step ${int(t+1)}: ${STATE_KEYS.map(k=>`${L(STATE_NAME[k])} ${pc(sh[k])}`).join(', ')}.${g.wait[t]?'':' No request is waiting at this decode step.'}`,`decode 第 ${int(t+1)} 步的显存池：${STATE_KEYS.map(k=>`${L(STATE_NAME[k])} ${pc(sh[k])}`).join('，')}。${g.wait[t]?'':'这个 decode 步没有请求在排队。'}`));
  const hm=$('#pa-metrics',host);
  if(hm){const sub=(k,h)=>{const el=hm.querySelector(`[data-k="${k}"] .sub`); if(el&&el._h!==h){el.innerHTML=h; el._h=h;}};
    sub('util',hide?maskNote():tx('average share of the pool over the decode steps in which requests wait','有请求在排队的 decode 步里，显存池的平均比例'));
    sub('run',hide?maskNote():tx(`${n1(r.running)==='1'?'request':'requests'} (average over the decode steps in which requests wait)`,'个请求（有请求在排队的 decode 步里的平均）'));
    sub('steps',hide?maskNote():tx(`decode steps, ${n1(r.tps)} ${n1(r.tps)==='1'?'token':'tokens'} generated per step on average`,`个 decode 步，平均每步生成 ${n1(r.tps)} 个 token`));
    sub('unused',hide?maskNote():S.strat==='paged'?(S.B===1?tx('tokens of room: a block holds 1 token, so none is left unfilled','个 token 的空间：每个 block 只放 1 个 token，不会有没写满的部分')
        :tx(`tokens of room, at most ${S.B-1} (the unfilled part of the last block)`,`个 token 的空间，最多 ${S.B-1} 个（最后一个 block 没写满的部分）`))
      :(S.strat==='exact'?tx('tokens of room: exact reservations are used up','个 token 的空间：按实际长度预留，正好用完'):tx('tokens of room: on average each request reserved this much more than it used','个 token 的空间：平均每个请求多预留了这么多，到结束也没用上')));
    sub('table',S.strat==='paged'?tx('entries per request on average','条，每个请求平均'):tx('one region per request needs none','每个请求一整块，不需要'));}
  if(hoverOwner>=0||kbd>=0) refreshTip();
  A.dock.render();
}
function cellAt(ev){const cv=$('#pa-cv',host); if(!cv||!geo) return null; const rc=cv.getBoundingClientRect(), x=ev.clientX-rc.left, y=ev.clientY-rc.top;
  const {cols,rows,cs,ch,gap}=geo, c=Math.min(cols-1,Math.max(0,Math.floor(x/(cs+gap)))), rr=Math.min(rows-1,Math.max(0,Math.floor(y/(ch+gap))));
  if(x<0||y<0||x>rc.width||y>rc.height) return null;
  const xin=Math.max(0,Math.min(cs-0.01,x-c*(cs+gap))); return {k:rr*cols+c,slot:(rr*cols+c)*CELL+Math.floor(xin/cs*CELL)};}
let tipSlot=-1;
function slotTip(slot){
  const r=run(), s=SL[slot], o=OW[slot];
  if(o>=0){const q=reqAt(r,o,t); if(!q) return ''; const gen=q.len-q.p;
    const head=tx(`<div><b>Request ${o+1}</b>: a ${int(q.p)}-token prompt; its answer will be ${int(q.o)} tokens, ${int(gen)} generated so far.</div>`,`<div><b>请求 ${o+1}</b>：prompt ${int(q.p)} 个 token；回答最终会生成 ${int(q.o)} 个 token，目前已生成 ${int(gen)} 个。</div>`);
    if(r.paged) return head+tx(`<div>It holds ${q.nb} ${q.nb===1?'block, so its block table has 1 entry':`blocks, so its block table has ${q.nb} entries`}.</div>`,`<div>它现在有 ${q.nb} 个 block，block table 里就是 ${q.nb} 条。</div>`);
    const cell=tx(`<div class="m">This cell: ${L(STATE_NAME[s===TOK?'tok':(s===RESV?'resv':'intf')])}.</div>`,`<div class="m">这一格：${s===TOK?'存着 token':(s===RESV?'为后面的 token 预留':'内部碎片')}。</div>`);
    if(S.strat==='exact') return head+tx(`<div>Its room is exactly its final length: ${int(q.R)} tokens.</div>`,`<div>它占着的空间正好等于最终长度：${int(q.R)} 个 token。</div>`)+cell;
    return head+tx(`<div>It holds room for ${int(q.R)} tokens (prompt plus ${int(S.lim)}): ${int(q.F-q.len)} wait for its later tokens and ${int(q.R-q.F)} will never be used.</div>`,`<div>它占着 ${int(q.R)} 个 token 的空间（prompt 加 ${int(S.lim)}）：${int(q.F-q.len)} 个留给后面的 token，${int(q.R-q.F)} 个到结束也用不上。</div>`)+cell;}
  if(s===EXT){let a=slot,b=slot; while(a>0&&SL[a-1]===EXT) a--; while(b<POOL-1&&SL[b+1]===EXT) b++; const h=headAt(r,t);
    return tx(`<div>External fragmentation: this gap has room for ${int(b-a+1)} tokens, but ${reqName(h)}, next in line, needs one piece with room for ${int(r.R[h])}.</div>`,`<div>外部碎片：这段空隙能放 ${int(b-a+1)} 个 token，但排在最前面的${reqName(h)} 需要一整段能放 ${int(r.R[h])} 个 token 的空间。</div>`);}
  return r.agg.wait[t]?tx('<div>Free: nobody holds this room. The next request in line needs more blocks than are free.</div>','<div>空闲：没人占用。排在最前面的请求需要的 block 比现在空闲的多。</div>')
                      :tx('<div>Free: nobody holds this room. Nobody is waiting either.</div>','<div>空闲：没人占用。也没有请求在排队。</div>');
}
function showTipAt(k,slot){
  const an=$('#pa-anchor',host); if(!an||!geo) return;
  const {cols,cs,ch,gap}=geo, x=(k%cols)*(cs+gap), y=((k/cols)|0)*(ch+gap);
  an.style.left=x+'px'; an.style.top=y+'px'; an.style.width=cs+'px'; an.style.height=ch+'px';
  const html=slotTip(slot); tipSlot=slot; if(html) A.tip.show(an,html); else A.tip.hide();
}
function refreshTip(){if(tipSlot>=0&&(hoverOwner>=0||kbd>=0)){const k=(tipSlot/CELL)|0; showTipAt(k,tipSlot);}}
function bindPool(){
  const cv=$('#pa-cv',host); if(!cv) return;
  cv.addEventListener('mousemove',ev=>{const c=cellAt(ev); if(!c) return; const o=OW[c.slot];
    if(o!==hoverOwner){hoverOwner=o; drawPool(run());}
    if(c.slot!==tipSlot||$('#tip').hidden) showTipAt(c.k,c.slot);});
  cv.addEventListener('mouseleave',()=>{hoverOwner=-1; tipSlot=-1; A.tip.hide(); drawPool(run());});
  cv.addEventListener('click',ev=>{const c=cellAt(ev); if(!c) return; hoverOwner=OW[c.slot]; drawPool(run()); setTimeout(()=>showTipAt(c.k,c.slot),0);});
  cv.addEventListener('focus',()=>{if(kbd<0){kbd=0; const r=run(); fillSlots(r,t); for(let k=0;k<NCELL;k++) if(OW[k*CELL]>=0){kbd=k; break;}} hoverOwner=OW[kbd*CELL]; drawPool(run()); showTipAt(kbd,kbd*CELL);});
  cv.addEventListener('blur',()=>{kbd=-1; hoverOwner=-1; tipSlot=-1; A.tip.hide(); drawPool(run());});
  cv.addEventListener('keydown',ev=>{if(kbd<0||!geo) return; const {cols}=geo; let k=kbd;
    if(ev.key==='ArrowRight') k++; else if(ev.key==='ArrowLeft') k--; else if(ev.key==='ArrowDown') k+=cols; else if(ev.key==='ArrowUp') k-=cols; else if(ev.key==='Home') k=0; else if(ev.key==='End') k=NCELL-1;
    else if(ev.key==='Escape'){A.tip.hide(); return;} else return;
    ev.preventDefault(); kbd=Math.max(0,Math.min(NCELL-1,k)); hoverOwner=OW[kbd*CELL]; drawPool(run()); showTipAt(kbd,kbd*CELL);});
}

/* ================= playback: advances by elapsed time x speed ================= */
function stop(){playing=false; oneOff=0; if(raf) cancelAnimationFrame(raf); raf=0;}
/* rate: a one-off speed for Show me; the learner's speed setting stays as it is, and picking a speed takes over at once */
function play(rate){
  if(playing) return;                                       // one animation loop at a time
  const r=run();
  if(A.reduceMotion()){stop(); setT(r.T-1); return;}      // reduced motion: jump to the end state
  if(t>=r.T-1) t=0;
  playing=true; oneOff=rate||0; lastTs=0; acc=0; drawFrame();
  const tick=ts=>{if(!playing||!host){raf=0; return;}
    const rr=run();
    if(lastTs){acc+=Math.min(0.1,(ts-lastTs)/1000)*(oneOff||SPEEDS[speed]); const n=Math.floor(acc);   // a long pause (hidden tab) does not jump ahead
      if(n>0){acc-=n; setT(Math.min(rr.T-1,t+n),true);}}
    lastTs=ts;
    if(t>=rr.T-1){stop(); drawFrame(); A.hash.write(token()); return;}
    raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick);
}
function setSpeed(v){if(!SPEEDS[v]) return; speed=v; oneOff=0; $$('.seg-ctl[data-key="speed"]',host).forEach(g=>A.ui.setPressed(g,speed));}
function setT(v,fromPlay){
  const r=run(); t=Math.max(0,Math.min(r.T-1,Math.round(v))); if(t>=r.T-1) watched.add(runKey(S));
  drawFrame();
  // part 3 shows the whole-run result only once the run has reached its last decode step
  if(mode==='guided'&&step===2&&t>=r.T-1&&!observed[2]){observed[2]=true; update(); showDone();}
  else if(t<r.T-1){const dn=$('#pa-done',host); if(dn) dn.hidden=true;}
  if(!fromPlay) A.hash.write(token());
}

/* ================= settings changes ================= */
function setS(p,o){
  o=o||{};
  if(!o.keepNote) sameNote=null;
  const before=runKey(S);
  S={...S,...p};
  if(mode==='challenge') Object.assign(S,CH_BASE);
  if(runKey(S)!==before){stop(); t=o.keepT?t:run().showcase; hoverOwner=-1;}
  update();
}
function onSeg(k,v,el){
  if(k==='speed'){setSpeed(v); return;}
  if(k==='strat'){setS({strat:v}); return;}
  const n=+v; if(k==='B'){bTried.add(n); const p={B:n}; if(mode==='guided'&&step===5&&S.strat!=='paged') p.strat='paged'; setS(p);
    if(S.strat!=='paged'&&el) A.tip.show(el,`<div>${tx(`No change: “${esc(cfgName(S))}” does not use blocks, so the pool stays the same. Block size applies once the strategy is Paging.`,`没有变化：“${esc(cfgName(S))}”不用 block，显存池不变。分配方式换成“分页”时，block 大小才起作用。`)}</div>`);
    return;}
  if(k==='mu'||k==='lim'){if(mode==='challenge') return;
    // a cap change that changes no number gets a "no change" note with the reason in free explore
    if(k==='lim'){const same=n!==S.lim&&sameRun(run(),simulate({...S,lim:n}));
      sameNote=same?{key:runKey({...S,lim:n}),kind:'cap',fromMu:S.mu,fromLim:S.lim,toLim:n}:null; setS({lim:n},{keepNote:true}); return;}
    setS({[k]:n});}
}
/* the line under the scenario list: it leads with what the click changed (R1.9, R6.1) */
function scenarioLine(r,cur,nt){
  const sp=tx(' ',''), sel=cur<0?tx(`No scenario is selected: answer lengths with a mean of ${int(S.mu)} tokens and a cap of ${int(S.lim)} tokens match none of them.`,`没有选中场景：回答长度按均值 ${int(S.mu)} 个 token 抽取、上限 ${int(S.lim)} 个 token，和哪个场景都不一样。`)
    :tx(`Selected: ${L(PRESETS[cur].t)}.`,`已选：${L(PRESETS[cur].t)}。`);
  if(nt){// nothing changed: say so and why, then where reserving the maximum does differ (in words; its steps are on its cards)
    const up=nt.toLim>nt.fromLim, mNow=simulate({...S,strat:'max'}), mWas=simulate({...S,mu:nt.fromMu,lim:nt.fromLim,strat:'max'}), q=mNow.T/mWas.T;
    const why=S.strat==='paged'?tx('paging hands out the next block only when the last one is full','分页只在写满一个 block 后才分配下一个'):tx('reserving the exact length holds only each request’s final length','按实际长度预留只占每个请求的最终长度');
    const tail=tx(`${why}, and no answer here reaches even the lower cap, so ${up?'raising':'lowering'} the cap changes no number.`,`${why}，这些回答连较低的那个上限都没写到，所以上限${up?'提高':'降低'}后，数字一个都没变。`);
    const diff=q>=2?tx('many more','多得多的'):(q>=1.1?tx('more','更多的'):(q<=0.5?tx('far fewer','少得多的'):(q<=0.9?tx('fewer','更少的'):tx('about as many','差不多的'))));
    const maxT=tx(`Reserving the maximum is different: each request then reserves the ${up?'higher':'lower'} cap, so the run takes ${diff} decode steps.`,`按最大长度预留时就不同了：每个请求都按${up?'更高':'更低'}的上限占空间，跑完要${diff} decode 步。`);
    if(nt.kind==='cap') return sel+sp+tx(`Changing the cap changed nothing: ${tail}`,`改上限没有变化：${tail}`)+sp+maxT;
    const pn=nt.from>=0?L(PRESETS[nt.from].t):null;
    return sel+sp+tx(`No change from ${pn?`“${pn}”`:'the previous setup'}: ${tail}`,`和${pn?`“${pn}”`:'刚才的设置'}相比没有变化：${tail}`)+sp+maxT;}
  if(cur<0) return sel;
  // the numbers changed: one rounded share (on the cards), and the decode steps against reserving the maximum in words (R6.20)
  const m0=simulate({...S,strat:'max'});
  const lead=S.strat==='paged'?tx(`With paging, about ${pc0(r.util)} of the pool stores tokens while requests wait, so the requests finish in ${stepsVsMaxW(r.T,m0.T)}.`,`分页时，排队期间显存池约 ${pc0(r.util)} 存着 token，所以跑完这些请求${stepsVsMaxW(r.T,m0.T)}。`)
    :(S.strat==='exact'?tx(`When each request reserves its exact length, about ${pc0(r.util)} of the pool stores tokens while requests wait, and the requests finish in ${stepsVsMaxW(r.T,m0.T)}.`,`按实际长度预留时，排队期间显存池约 ${pc0(r.util)} 存着 token，跑完这些请求${stepsVsMaxW(r.T,m0.T)}。`)
    :tx(`When each request reserves the maximum, only about ${pc0(r.util)} of the pool stores tokens while requests wait: each request holds room for the whole answer cap from the start.`,`按最大长度预留时，排队期间显存池只有约 ${pc0(r.util)} 存着 token：每个请求一开始就按整个回答上限占着空间。`));
  return sel+sp+lead;}
/* after a scenario click the line under the list must be on screen: scroll just enough, so the list stays in view */
function keepInView(el){if(!el) return; const rc=el.getBoundingClientRect(), vh=window.innerHeight;
  if(rc.bottom<=vh-8) return; if(rc.height>vh*0.5){A.ui.reveal(el); return;}
  window.scrollBy({top:rc.bottom-vh+16,behavior:A.reduceMotion()?'auto':'smooth'});}
/* two runs that show the same numbers everywhere */
const sameRun=(a,b)=>a.T===b.T&&a.util===b.util&&a.running===b.running&&a.unused===b.unused&&a.table===b.table;
/* a scenario click; when it changes no number, the card says so and why (R1.9) */
function pickPreset(i){const pr=PRESETS[i], cur=presetNow(), next={...S,...pr.p};
  const same=i!==cur&&(S.mu!==pr.p.mu||S.lim!==pr.p.lim)&&sameRun(run(),simulate(next));
  sameNote=same?{key:runKey(next),kind:'scn',from:cur,fromMu:S.mu,fromLim:S.lim,toLim:pr.p.lim}:null;
  setS({...pr.p},{keepNote:true});}
function onLever(v){
  const grp=v.slice(0,2), id=v.slice(3);
  if(grp==='hl'){hl=id; hlTried.add(id); if(id!=='all'){observed[3]=true;} stop(); t=R0().showcase; if(runKey(S)!==runKey(DEFAULT)){S={...DEFAULT};} update(); return;}
  const lv=ST_LEVERS.find(x=>x.v===id); if(!lv) return;
  stTried.add(id); if(id!=='start') observed[4]=true;
  S={...DEFAULT,...lv.s}; stop(); t=run().showcase; hoverOwner=-1; update();
}
const lockTxt=()=>tx(`In the challenge you can change only the allocation strategy and the block size. Everything else stays fixed: Llama-3.1-8B, a pool with room for ${int(POOL)} tokens, the same ${NREQ} requests.`,`挑战中只能改分配方式和 block 大小。其余保持不变：Llama-3.1-8B，能放 ${int(POOL)} 个 token 的显存池，同样的 ${NREQ} 个请求。`);
function setMode(m){
  if(m===mode) return;
  stop(); const was=mode; mode=m; sameNote=null;
  if(was==='challenge'&&savedS){S={...savedS}; savedS=null; t=run().showcase;}
  if(m==='challenge'){savedS={...S}; chSubmitted=false; chReveal=false; S={...CH_BASE,strat:'max',B:16}; t=run().showcase;}
  hl='all'; A.shell.setMode(host,m,DESCS());
  const c=$('#pa-controls',host); c.classList.toggle('locked',m==='challenge');
  const ln=$('#pa-lock',host); ln.hidden=m!=='challenge';
  if(m==='challenge') ln.textContent=lockTxt();
  if(m==='guided'){enterStep(step); return;}
  renderRail(); update(); clearBadges(); cue();
}
function showMe(){
  if(step===2){const r=run(); if(t>=r.T-1){observed[2]=true; update();} else play(SPEEDS.fast); return;}
  if(step===3){onLever('hl-intf'); return;}
  if(step===4){onLever('st-paged'); return;}
  if(step===5){onSeg('B','256'); return;}
}
/* the run ended while part 3's card, which now holds the result, is off screen (phones: the card is above the pool) */
function showDone(){const btn=$('#pa-done',host); if(!btn) return; btn.hidden=true; if(mode!=='guided'||step!==2) return;
  const card=$('#pa-rail .step.current',host); if(!card) return; const box=$('[data-obs]',card), el=box&&!box.hidden?box:card, rc=el.getBoundingClientRect(), top=64;
  if(rc.bottom>top+24&&rc.top<window.innerHeight-24) return;
  btn.textContent=tx(`Run finished: the result is in part 3 ${rc.top<top?'↑':'↓'}`,`跑完了，结果在第 3 部分 ${rc.top<top?'↑':'↓'}`); btn.hidden=false;}
function toCard(btn){btn.hidden=true; const card=$('#pa-rail .step.current',host); if(!card) return; const box=$('[data-obs]',card), el=box&&!box.hidden?box:card;
  A.ui.reveal(el); el.setAttribute('tabindex','-1'); el.focus({preventScroll:true});}
function submitChallenge(){chSubmitted=true; if(chEval(chOpt()).met) chReveal=true; renderRail(); update();
  const res=$('#pa-chres',host); if(res) res.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'});}

/* ================= jumps, tooltips for formula terms ================= */
const CTL={strat:'pa-ctl-strat',B:'pa-ctl-B',mu:'pa-ctl-mu',lim:'pa-ctl-lim'};
const INL={strat:'pa-ic-strat',B:'pa-ic-B'};
const CTLNAME={strat:{en:'Allocation strategy',zh:'分配方式'},B:{en:'Block size',zh:'block 大小'},mu:{en:'Answer length (mean)',zh:'回答长度（均值）'},lim:{en:'Answer cap',zh:'回答上限'}};
const TERM={
 strat:{en:'This is the current allocation strategy; it decides how much room a request holds before its tokens are generated.',zh:'这是当前的分配方式，它决定请求在 token 生成之前就占多少空间。'},
 B:{en:'Block size is how many tokens one block holds, the unit that paging hands out.',zh:'block 大小是每个 block 能放几个 token，也就是分页时分配的单位。'},
 mu:{en:'Answer lengths are drawn at random with this mean; the mean is an assumption about the workload.',zh:'回答长度按这个均值随机抽取；这个均值是对负载的假设。'},
 lim:{en:'The answer cap (max_tokens) is the most tokens a request may generate. Reserving the maximum holds room for the prompt plus this many tokens.',zh:'回答上限（max_tokens）是一个请求最多生成多少 token。按最大长度预留时，每个请求占 prompt 加这么多 token 的空间。'},
 pool:{en:'The pool has room for 12,288 tokens, 1.5 GiB (1.61 GB) of Llama-3.1-8B KV cache, scaled down so each cell stays visible.',zh:'显存池固定能放 12,288 个 token，即 1.5 GiB（1.61 GB）的 Llama-3.1-8B KV cache；按比例缩小，好让每一格看得清。'},
 n:{en:`These ${NREQ} requests all wait at the start; they were drawn once with random seed ${SEED}.`,zh:`这 ${NREQ} 个请求一开始全部在排队，是用随机种子 ${SEED} 抽一次得到的。`}
};
const locked=k=>mode==='challenge'&&(k==='mu'||k==='lim'||k==='pool'||k==='n'||k==='wl');
function jumpAct(k){
  if(k&&k.slice(0,3)==='hl-') return hl===k.slice(3)?tx('Click to show all kinds again.','点击恢复显示全部。'):tx('Click to highlight this room in the pool.','点击在显存池里高亮这种空间。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(k==='view') return tx('Click to scroll to the pool.','点击滚到显存池。');
  if(k==='avgbar') return tx('Click to scroll to the “While requests wait” bar.','点击滚到“排队时平均”条。');
  if(locked(k)){if(k==='mu') return tx(`Fixed during the challenge: answers are drawn with a mean of ${int(CH_BASE.mu)} tokens.`,`挑战中固定不变：回答长度的均值是 ${int(CH_BASE.mu)} 个 token。`);
    if(k==='lim') return tx(`Fixed during the challenge: the answer cap is ${int(CH_BASE.lim)} tokens.`,`挑战中固定不变：回答上限是 ${int(CH_BASE.lim)} 个 token。`);
    return tx(`Fixed during the challenge: ${NREQ} requests and a pool with room for ${int(POOL)} tokens.`,`挑战中固定不变：${NREQ} 个请求，能放 ${int(POOL)} 个 token 的显存池。`);}
  if(k==='pool'||k==='n'||k==='wl') return tx('Click to open “Pool size, prompt lengths, random seed and timing” under Settings.','点击打开“自由调参”里的“显存池大小、prompt 长度、随机种子和计时”。');
  if(CTLNAME[k]) return tx(`Click to jump to “${L(CTLNAME[k])}” under Settings.`,`点击跳到“自由调参”里的“${L(CTLNAME[k])}”。`);
  return '';
}
/* a legend click that highlights a kind of room no slot is in at this decode step says why instead of doing nothing visible */
function emptyStateMsg(id){const r=run(); fillSlots(r,t); let n=0; const code={tok:TOK,resv:RESV,intf:INTF,ext:EXT,free:FREE}[id]; for(let k=0;k<POOL;k++) if(SL[k]===code) n++;
  if(n) return '';
  const ts=int(t+1);
  if(id==='ext'&&r.paged) return tx('Nothing to highlight: paging never leaves external fragmentation, because any free block works.','没有可高亮的：分页不会产生外部碎片，任何空闲 block 都能用。');
  if(id==='free'&&!r.paged&&r.agg.wait[t]) return tx(`Nothing to highlight at decode step ${ts}: a request is waiting and no gap fits it, so all room nobody holds counts as external fragmentation.`,`decode 第 ${ts} 步没有可高亮的：有请求在排队，而每段空隙都放不下它，所以没人占用的空间都算外部碎片。`);
  return tx(`At decode step ${ts} there is no room of this kind to highlight.`,`decode 第 ${ts} 步没有这种空间，没什么可高亮的。`);}
function jump(k,src){
  if(k&&k.slice(0,3)==='hl-'){const id=k.slice(3); hl=hl===id?'all':id; if(mode==='guided'&&step===3&&id!=='free'&&id!=='tok'){hlTried.add(hl); if(hl!=='all') observed[3]=true;} A.tip.hide(); update(); A.ui.flash($('#pa-pool',host));
    const msg=hl===id?emptyStateMsg(id):''; if(msg&&src) A.tip.show(src,`<div>${msg}</div>`); return;}
  if(k==='math'){const d=$('#pa-math',host); d.open=true; A.ui.jumpToEl(d); return;}
  if(k==='controls'){A.ui.jumpToEl($('#pa-controls',host)); return;}
  if(k==='fid'){const pop=document.getElementById(ID+'-fid'), b=$(`[data-pop="${ID}-fid"]`,host); if(pop&&pop.hidden&&b) b.click(); if(pop) A.ui.jumpToEl(pop); return;}
  if(k==='view'||k==='avgbar'){const el=k==='view'?$('#pa-graph',host):$('#pa-avglab',host).closest('.pa-brow'); A.tip.hide(); el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(locked(k)){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  if(k==='pool'||k==='n'||k==='wl'){const d=$('#pa-adv',host); d.open=true; A.ui.jumpToEl(d); return;}
  const lv=(mode==='guided'&&step===4&&k==='strat')?$('#pa-lv-st',host):null;
  const inl=INL[k]&&document.getElementById(INL[k]), el=lv||((inl&&host.contains(inl))?inl:$('#'+CTL[k],host));
  if(el) A.ui.jumpToEl(el);
}
function termTip(k){return `<div>${esc(L(TERM[k]||{en:'',zh:''}))}</div><div class="act">${jumpAct(k)}</div>`;}

/* ================= hash token ================= */
const token=(s=S,tt=t)=>[s.strat,s.B,s.mu,s.lim,tt].join('.');
function parse(tok){
  const p=String(tok||'').split('.'); if(p.length!==5) return null;
  const o={strat:p[0],B:+p[1],mu:+p[2],lim:+p[3]}, tt=+p[4];
  if(!STRATS.includes(o.strat)||!BLOCKS.includes(o.B)||!MEANS.includes(o.mu)||!LIMITS.includes(o.lim)||!(tt>=0&&Number.isInteger(tt))) return null;
  return {s:o,t:tt};
}

/* ================= interview drill (seeded by question number) ================= */
function drill(k){const r=rng(9001+k*7919);
  if(k%2===0){const Lx=200+Math.floor(r()*1800), B=[8,16,32,64][Math.floor(r()*4)], n=Math.ceil(Lx/B); return {kind:0,L:Lx,B,n,w:n*B-Lx};}
  const lim=[1024,1536,2048][Math.floor(r()*3)], p=100+Math.floor(r()*500), len=p+50+Math.floor(r()*400), B=16, R=p+lim, c=Math.floor(POOL/R), per=Math.ceil(len/B)*B;
  return {kind:1,lim,p,len,B,R,c,per,g:Math.floor(POOL/per)};}
function drillQ(){const d=drill(drillK);
  return d.kind===0?tx(`A request ends with <b>${int(d.L)}</b> tokens (prompt plus answer). With <b>${d.B}</b>-token blocks, how many entries does its block table have, and how much room does its last block leave empty?`,
                      `一个请求结束时一共 <b>${int(d.L)}</b> 个 token（prompt 加回答）。block 为 <b>${d.B}</b> 个 token 时，它的 block table 有几条？最后一个 block 空着多少个 token 的空间？`)
    :tx(`A pool with room for ${int(POOL)} tokens. Every request has a ${int(d.p)}-token prompt and an answer cap of ${int(d.lim)}, and right now each holds <b>${int(d.len)}</b> tokens. How many requests fit at once if each reserves its maximum length in one region, and how many with 16-token blocks?`,
        `一个能放 ${int(POOL)} 个 token 的显存池。每个请求的 prompt 是 ${int(d.p)} 个 token，回答上限 ${int(d.lim)} 个，此刻每个请求都有 <b>${int(d.len)}</b> 个 token。如果每个请求按最大长度预留一整块，一次能放下几个请求？按 16 个 token 一个 block 呢？`);}
function drillA(){const d=drill(drillK);
  return d.kind===0?tx(`<p><b>${int(d.n)} entries; the last block leaves room for ${int(d.w)} ${d.w===1?'token':'tokens'} empty</b></p><p>⌈${int(d.L)} ÷ ${d.B}⌉ = ${int(d.n)} blocks</p><p>${int(d.n)} × ${d.B} − ${int(d.L)} = ${int(d.w)}, always less than ${d.B}</p>`,
                      `<p><b>${int(d.n)} 条；最后一个 block 空着 ${int(d.w)} 个 token 的空间</b></p><p>⌈${int(d.L)} ÷ ${d.B}⌉ = ${int(d.n)} 个 block</p><p>${int(d.n)} × ${d.B} − ${int(d.L)} = ${int(d.w)}，总是小于 ${d.B}</p>`)
    :tx(`<p><b>${d.c} reserving the maximum, ${d.g} with paging</b></p><p>Reserving the maximum: ⌊${int(POOL)} ÷ (${int(d.p)} + ${int(d.lim)})⌋ = ⌊${int(POOL)} ÷ ${int(d.R)}⌋ = ${d.c}, and fewer once gaps appear</p><p>Paging: each needs ⌈${int(d.len)} ÷ 16⌉ × 16 = ${int(d.per)} tokens of room, so ⌊${int(POOL)} ÷ ${int(d.per)}⌋ = ${d.g}, until they grow</p>`,
        `<p><b>按最大长度预留：${d.c} 个；分页：${d.g} 个</b></p><p>按最大长度预留：⌊${int(POOL)} ÷ (${int(d.p)} + ${int(d.lim)})⌋ = ⌊${int(POOL)} ÷ ${int(d.R)}⌋ = ${d.c}，出现空隙后会更少</p><p>分页：每个请求占 ⌈${int(d.len)} ÷ 16⌉ × 16 = ${int(d.per)} 个 token 的空间，⌊${int(POOL)} ÷ ${int(d.per)}⌋ = ${d.g}，请求变长后会减少</p>`);}
function renderDrill(){const q=$('#pa-drillq',host), a=$('#pa-drilla',host); if(!q) return; q.innerHTML=drillQ(); a.hidden=!drillOpen; if(drillOpen) a.innerHTML=drillA();}

/* ================= theme changes redraw the canvas ================= */
function watchTheme(){
  unwatchTheme();
  if(window.MutationObserver){themeObs=new MutationObserver(()=>{pats=null; drawFrame();}); themeObs.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}
  if(window.matchMedia){mq=matchMedia('(prefers-color-scheme: dark)'); mqFn=()=>{pats=null; drawFrame();}; if(mq.addEventListener) mq.addEventListener('change',mqFn);}
}
function unwatchTheme(){if(themeObs){themeObs.disconnect(); themeObs=null;} if(mq&&mqFn&&mq.removeEventListener) mq.removeEventListener('change',mqFn); mq=null; mqFn=null;}

/* the learner's own change: the next update may show change badges */
function learner(f){userAct=true; try{f();} finally{userAct=false;}}

/* keepStage (CONTRACT §11, R3.2, R9.7; as labs/moe.js; polish round 5): after Next, Back or a part's title the kit's reveal scrolls the
   new card's top up under the top bar; near the end of the bench that pushes the sticky stage up with it (part 7 entered from part 6 at
   1440 × 900 in Chinese: the stage's top at −91 px, its head under the top bar). Once the page stops scrolling this pulls the stage back
   under the top bar without pushing the card's top off the screen, and does nothing once the reader has scrolled the card away */
let stageT=0;
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!host) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#pa-stage',host); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#pa-rail .step.current',host)||$('#pa-rail',host); if(!card) return;
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the pool is drawn for the stage's width and, on a wide desktop stage, the cell height in paged-attention.css: the stage changes width
   without a window resize when the page puts the step list beside it or leaves the primer (part 1 at 1440 × 900 was drawn for a 554 px
   stage and stretched, then grew 69 px at the first hover), so it is drawn again when it does (as labs/ppo.js) */
let stRO=null, stW=0;
function watchStage(){unwatchStage(); const st=host&&$('#pa-stage',host); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!host||!host.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w; drawFrame();}));
  stRO.observe(st);}
function unwatchStage(){if(stRO){stRO.disconnect(); stRO=null;}}

/* ================= mount ================= */
function mount(el,tok){
  stop(); root=el;
  const p=tok?parse(tok):null;
  if(p){S={...p.s}; mode='sandbox'; savedS=null; t=p.t;}
  host=document.createElement('div'); host.className='pa-host m2'; el.replaceChildren(host);
  host.innerHTML=view(); A.ui.initRanges(host);
  A.shell.bindHeader(host,{id:ID,onMode:m=>{setMode(m); revealRail();},token});
  const c=$('#pa-controls',host); c.classList.toggle('locked',mode==='challenge');
  const ln=$('#pa-lock',host); ln.hidden=mode!=='challenge';
  if(mode==='challenge') ln.textContent=lockTxt();
  host.addEventListener('click',ev=>{
    const b=ev.target.closest('button'); if(!b||!host.contains(b)) return; const d=b.dataset;
    const seg=b.closest('.seg-ctl[data-key]'); if(seg&&d.v!=null){if(!b.disabled) learner(()=>onSeg(seg.dataset.key,d.v,b)); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){showMe(); return;}
    if(d.pred){if(seen) return; pred=d.pred; renderRail(); update(); const f=$(`#pa-rail [data-pred="${pred}"]`,host); if(f) f.focus(); return;}
    if(d.lever!=null){learner(()=>onLever(d.lever)); return;}
    if(d.preset!=null){learner(()=>pickPreset(+d.preset)); keepInView($('#pa-sbnote',host)); return;}
    if(d.pa==='play'){if(playing){stop(); drawFrame(); A.hash.write(token());} else play(); return;}
    if(d.pa==='step'){stop(); const r=run(); if(t>=r.T-1){const n=r.agg.run[t]; A.tip.show(b,`<div>${tx(`No change: decode step ${int(t+1)} is the last one; ${n===1?'the final request finishes':`the final ${n} requests finish`} in it. Drag the slider back or press Play to start again.`,`没有变化：decode 第 ${int(t+1)} 步已经是最后一步，最后${n===1?'一':` ${n} `}个请求在这一步完成。把滑块拖回去，或者点“播放”重新开始。`)}</div>`); return;} setT(t+1); return;}
    if(d.modeGo){setMode(d.modeGo); revealRail(); return;}
    if(d.act==='submit'){submitChallenge(); return;}
    if(d.act==='to-card'){toCard(b); return;}
    if(d.act==='reveal'){chReveal=true; updateRail(); return;}
    if(d.act==='reset'){learner(()=>{if(mode==='challenge') setS({strat:'max',B:16}); else setS({...DEFAULT});}); return;}
    if(d.act==='drill-new'){drillK++; drillOpen=false; renderDrill(); return;}
    if(d.act==='drill-show'){drillOpen=true; renderDrill(); return;}
    if(d.act==='copy'){const pre=$('#pa-code',host), txt=pre.textContent, ok=()=>{b.textContent=tx('Copied','已复制'); setTimeout(()=>{b.textContent=A.t('copyCode');},1600);},
      sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); b.textContent=tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'); setTimeout(()=>{b.textContent=A.t('copyCode');},1600);};
      try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(ok,sel); else sel();}catch(e){sel();} return;}
  });
  host.addEventListener('input',ev=>{if(ev.target.dataset&&ev.target.dataset.pa==='time'){stop(); setT(+ev.target.value);}});
  // Enter / Space on focusable non-button items that jump (legend items, the workload line); formula terms are handled by the core
  host.addEventListener('keydown',ev=>{if(ev.key!=='Enter'&&ev.key!==' ') return; const el=ev.target; if(!el.dataset||!el.dataset.jump||el.tagName==='BUTTON'||el.classList.contains('t')) return; ev.preventDefault(); jump(el.dataset.jump,el);});
  bindPool(); watchTheme();
  if(!mounted&&!p&&mode==='guided'){mounted=true; enterStep(step);} else {mounted=true; renderRail(); update(); cue();}
  renderDrill(); watchStage();
}
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:{strat:S.strat,B:S.B,mu:S.mu,lim:S.lim},t,pred,seen,observed:{...observed},hl,speed,savedS:savedS?{strat:savedS.strat,B:savedS.B,mu:savedS.mu,lim:savedS.lim}:null,chSubmitted,chReveal,
    hlTried:[...hlTried],stTried:[...stTried],bTried:[...bTried],watched:[...watched],drillK,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(chOpt()).met}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S,o.t)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; S=p.s; if(mode==='challenge') Object.assign(S,CH_BASE);
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; t=Math.min(p.t,run().T-1);
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; seen=o.seen===true||(mode==='guided'&&step>=2); observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    hl=o.hl==='all'||STATE_KEYS.includes(o.hl)?o.hl:'all'; if(SPEEDS[o.speed]) speed=o.speed;
    const sv=o.savedS&&typeof o.savedS==='object'?parse(token(o.savedS,0)):null; savedS=mode==='challenge'&&sv?sv.s:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    const fill=(set,arr,ok)=>{if(Array.isArray(arr)) arr.forEach(x=>{if(ok(x)) set.add(x);});};
    fill(hlTried,o.hlTried,x=>x==='all'||STATE_KEYS.includes(x)); fill(stTried,o.stTried,x=>x==='start'||ST_LEVERS.some(l=>l.v===x)); fill(bTried,o.bTried,x=>BLOCKS.includes(x));
    watched.clear(); fill(watched,o.watched,x=>{const q=typeof x==='string'?x.split('.'):[]; return q.length===4&&STRATS.includes(q[0])&&+q[1]===(q[0]==='paged'?+q[1]:0)&&(q[0]!=='paged'||BLOCKS.includes(+q[1]))&&MEANS.includes(+q[2])&&LIMITS.includes(+q[3])&&x===runKey({strat:q[0],B:+q[1],mu:+q[2],lim:+q[3]});});
    if(Number.isInteger(o.drillK)&&o.drillK>=0&&o.drillK<1e6) drillK=o.drillK; drillOpen=o.drillOpen===true;
    // the first mount then draws this part as a return in the same session does (enterStep() would reset the settings)
    mounted=true; if(mode==='guided'&&step===4) sweepWarm();},
  mount,
  leave(){stop(); unwatchTheme(); unwatchStage(); stageStop(); clearTimeout(sweepT); sweepT=0; A.tip.hide(); hoverOwner=-1; kbd=-1; tipSlot=-1; host=null; root=null;},
  onLang(){if(root) mount(root,null);},
  onResize(){if(host) drawFrame();},
  termTip, jump, jumpAct,
  dock:{
    anchor:()=>host&&$('#pa-cv',host), bench:()=>host&&$('.bench',host), target:()=>host&&$('#pa-graph',host),
    /* the mini result bar, read left to right: badge, bar, sentence. While part 3 runs (and before the prediction) the bar is the pool
       at the current decode step; otherwise it is the whole-run average, the result the badge names */
    render({status:st,vis,read}){if(!host) return; const r=run(), s=status(r), g=r.agg, live=hidden();
      st.className='status '+s.cls; st.textContent=s.short||s.text;
      let bar=vis.querySelector('.pa-bar'); if(!bar){vis.innerHTML='<div class="pa-bar pa-dkbar"></div>'; bar=vis.querySelector('.pa-bar');}
      bar.style.visibility=masked()?'hidden':''; drawShareBar(bar,live?nowShares(r):r.shares); STATE_KEYS.forEach((k,i)=>{bar._e[i].style.opacity=(hl!=='all'&&hl!==k)?'0.25':'';});
      const cfg=cfgName(S); let lead;
      if(mode==='challenge') lead=chShort(r,chEval(chOpt()));
      else if(masked()) lead=tx(`Decode step ${int(t+1)}`,`decode 第 ${int(t+1)} 步`);
      else if(pending()) lead=tx(`Decode step ${int(t+1)}: ${pc(g.tok[t]/POOL)} of the pool stores tokens now`,`decode 第 ${int(t+1)} 步：此刻 ${pc(g.tok[t]/POOL)} 存着 token`);
      else lead=tx(`${pc(r.util)} stores tokens on average`,`${pc(r.util)} 存着 token`);
      read.innerHTML=`<span class="pa-dk">${esc(lead)}<span class="dk-mem">${esc(tx(` (${cfg})`,`（${cfg}）`))}</span></span>`;}
  }
});

/* ================= home tile teaser: the first 8 requests at their final lengths, worked out with arithmetic ================= */
let tzMode='max';
function tzData(){
  const q=workload(DEFAULT.mu,DEFAULT.lim,SEED).slice(0,8), n=q.length, R=q.map(x=>x.p+DEFAULT.lim), alloc=q.map(x=>Math.ceil(x.F/16)*16);
  // served in order, like the lab: the first request that does not fit waits, and so does everyone behind it
  const fits=sizes=>{let used=0, k=0; while(k<n&&used+sizes[k]<=POOL){used+=sizes[k]; k++;} return {k,used};};
  const c=fits(R), p=fits(alloc), tokOf=k=>q.slice(0,k).reduce((a,x)=>a+x.F,0);
  return {n,F:q.map(x=>x.F),P:q.map(x=>x.p),R,alloc,fit:c.k,held:c.used,tokFit:tokOf(c.k),left:POOL-c.used,next:c.k<n?R[c.k]:0,
    pFit:p.k,allocSum:p.used,tokAll:tokOf(p.k),scale:Math.max(...R,...alloc),min:Math.min(...q.map(x=>x.F)),max:Math.max(...q.map(x=>x.F))};}
A.registerTeaser(ID,{fid:'formula',render(el){
  const d=tzData(), lim=int(DEFAULT.lim);
  el.innerHTML=`<div class="tz">${A.ui.seg({id:'tz-pa-mode',key:'tzpa',label:tx('How each request’s KV cache is stored','每个请求的 KV cache 怎么存'),options:[['max',tx('Reserve the maximum','按最大长度预留')],['paged',tx('16-token blocks','16 个 token 一个 block')]],value:tzMode})}`+
    `<div class="tz-pa-cap">${tx(`${d.n} requests, one bar each; the pool holds ${int(POOL)} tokens`,`${d.n} 个请求，每条是一个请求；显存池能放 ${int(POOL)} 个 token`)}</div>`+
    `<div class="tz-pa-rows tz-vis" role="img"></div>`+
    `<div class="tz-leg"><span><i class="sw kv"></i>${tx('tokens it ends up storing','最终存着的 token')}</span><span><i class="sw tz-pa-swheld"></i>${tx('reserved, never used','预留了但没用上')}</span><span class="tz-pa-lgwait"><i class="sw tz-pa-swwait"></i>${tx('does not fit, waits','放不下，排队')}</span></div>`+
    `<p class="tz-read"></p>`+
    `<p class="tz-note">${esc(tx(`The pool: GPU memory set aside for the KV cache, room for ${int(POOL)} tokens (${A.fmt.B(kv8b().kvPerToken)} per token for Llama-3.1-8B, ${A.fmt.B(poolBytes())} in total), as in the lab.`,
      `显存池：一张 GPU 留给 KV cache 的显存，能放 ${int(POOL)} 个 token（Llama-3.1-8B 每个 token ${A.fmt.B(kv8b().kvPerToken)}，共 ${A.fmt.B(poolBytes())}），和 Lab 里一样。`))}</p>`+
    `<p class="tz-note">${esc(tx(`These are the first ${d.n} requests of the lab’s workload: prompts of ${PMIN} to ${PMAX} tokens, answers of at most ${lim} tokens. Each bar is drawn at the request’s final length.`,
      `这 ${d.n} 个请求是 Lab 负载里的前 ${d.n} 个：prompt ${PMIN} 到 ${PMAX} 个 token，回答最多 ${lim} 个 token。每条按请求结束时的长度画。`))}</p>`+
    `<p class="tz-note">${esc(tx('The vLLM paper measured that in earlier systems only 20.4% to 38.2% of KV cache memory actually held tokens (Kwon et al., 2023).','vLLM 论文测得，以前的系统里 KV cache 显存只有 20.4% 到 38.2% 真正存着 token（Kwon 等，2023）。'))}</p></div>`;
  const rows=el.querySelector('.tz-pa-rows'), rd=el.querySelector('.tz-read'), seg=el.querySelector('.seg-ctl'), lgw=el.querySelector('.tz-pa-lgwait');
  const w=v=>(v/d.scale*100).toFixed(2)+'%';
  const upd=()=>{A.ui.setPressed(seg,tzMode); const max=tzMode==='max';
    rows.innerHTML=d.F.map((F,i)=>{if(max&&i>=d.fit) return `<div class="tz-pa-row"><i class="tz-pa-wait" style="width:${w(d.R[i])}"></i></div>`;
      return `<div class="tz-pa-row"><i class="tz-pa-held" style="width:${w(max?d.R[i]:d.alloc[i])}"></i><i class="tz-pa-used" style="width:${w(F)}"></i></div>`;}).join('');
    lgw.hidden=!max||d.fit>=d.n;
    rows.setAttribute('aria-label',max?tx(`${d.n} requests, one bar each: ${d.fit} fit, each reserving room for its prompt plus ${lim} tokens; ${d.n-d.fit} wait.`,`${d.n} 个请求，每条一个：放得下 ${d.fit} 个，每个预留 prompt 加 ${lim} 个 token 的空间；${d.n-d.fit} 个在排队。`)
      :tx(`${d.n} requests, one bar each, stored in 16-token blocks; ${d.pFit===d.n?'all fit':`${d.pFit} fit`}.`,`${d.n} 个请求，每条一个，按 16 个 token 一个 block 存放；${d.pFit===d.n?'都放得下':`放得下 ${d.pFit} 个`}。`));
    const unusedMax=pc(1-d.tokFit/d.held,0), storedPaged=pc(d.tokAll/d.allocSum,0);
    // one key number, in bold; how many bars fit is on the picture (R6.20, R11.5)
    rd.innerHTML=max
      ?tx(`Each request reserves room for its prompt plus the longest answer allowed: ${d.fit<d.n?`not all ${d.n} requests fit`:`all ${d.n} requests fit`}, and <b>${unusedMax}</b> of the reserved room is never used.`,
          `每个请求都按 prompt 加回答上限预留空间：${d.fit<d.n?`显存池放不下全部 ${d.n} 个请求`:`显存池放得下全部 ${d.n} 个请求`}，预留的空间有 <b>${unusedMax}</b> 到结束也没用上。`)
      :tx(`With 16-token blocks, handed out one at a time as they fill: ${d.pFit===d.n?'every request fits':`only some of the ${d.n} requests fit`}, and <b>${storedPaged}</b> of the room they hold stores tokens.`,
          `按 16 个 token 一个 block，写满一个再分配下一个：${d.pFit===d.n?'每个请求都放得下':`${d.n} 个请求只放得下一部分`}，占用的空间有 <b>${storedPaged}</b> 存着 token。`);};
  seg.addEventListener('click',ev=>{const b=ev.target.closest('button[data-v]'); if(!b) return; tzMode=b.dataset.v; upd();});
  upd();
}});

/* read-only hook for tests */
Object.defineProperty(window,'__pa',{configurable:true,value:Object.freeze({
  POOL,NREQ,SEED,BLOCKS,MEANS,LIMITS,DEFAULT,GOAL_UTIL,GOAL_TABLE,SPEEDS,
  workload:(mu,lim,seed)=>workload(mu,lim,seed).map(x=>({...x})),
  stats:s=>{const r=simulate({...DEFAULT,...s}); return {T:r.T,util:r.util,shares:{...r.shares},running:r.running,tps:r.tps,unused:r.unused,table:r.table,pre:r.pre,busyN:r.busyN,showcase:r.showcase,meanF:r.meanF,meanO:r.meanO,capped:r.capped,tokAvg:r.tokAvg,tokSum:r.tokSum,tokMax:r.tokMax};},
  series:(s,k)=>Array.from(simulate({...DEFAULT,...s}).agg[k]),
  slotsAt:(s,tt)=>{const r=simulate({...DEFAULT,...s}); fillSlots(r,tt); const c=[0,0,0,0,0]; for(let k=0;k<POOL;k++) c[SL[k]]++; fillSlots(run(),t); return {free:c[0],tok:c[1],resv:c[2],intf:c[3],ext:c[4]};},
  challenge:()=>CH_OPTS.map(o=>{const e=chEval(o); return {o:{...o},util:e.util,table:e.table,met:e.met,changes:e.changes};}),
  best:()=>{const b=chBest(); return b?{...b.o}:null;},
  sweep:()=>({...sweep()}),
  timing:()=>({...timing()}),
  gpu:()=>{const g=gpuEst(); return {t90:g.t90,t92:g.t92};},
  teaser:()=>tzData(),
  drill:k=>drill(k),
  state:()=>({S:{...S},t,mode,step,hl,pred,seen,chSubmitted,chReveal,playing,speed})
})});
})();
