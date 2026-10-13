/* Continuous batching lab (#continuous-batching), prefix "cb".
   A seeded, step-by-step simulation of 48 requests on one A100 40GB serving OPT-13B,
   scheduled with static (request-level) or continuous (iteration-level) batching.
   Engine, view and narrative are kept apart: every number on the page comes from simulate(). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A;
const ID='continuous-batching';

/* ---------- verified constants ----------
   OPT-13B, facebook/opt-13b config.json: 40 layers, hidden 5,120, 40 heads of 128, ffn 20,480,
   vocab 50,272, 2,048 positions, float16.
   A100 40GB, NVIDIA datasheet: 40GB HBM2, 1,555 GB/s, FP16 Tensor Core 312 TFLOPS dense (624 with sparsity).
   vLLM v0.30.0: gpu_memory_utilization defaults to 0.92. Anyscale (2023): 512-token prompts, answers from an
   exponential with mean 128, keeping only draws at or under a cap of 32, 128, 512 or 1,536. */
const OPT={layers:40,hidden:5120,heads:40,ffn:20480,vocab:50272,pos:2048,bytes:2};
const PARAMS=(()=>{const h=OPT.hidden,f=OPT.ffn;
  const layer=4*(h*h+h)+2*(2*h)+(h*f+f)+(f*h+h);            // q,k,v,out projections, two layer norms, fc1, fc2
  return OPT.layers*layer+OPT.vocab*h+(OPT.pos+2)*h+2*h;})();    // + token and position embeddings (OPT offsets positions by 2), final norm
const W_BYTES=PARAMS*OPT.bytes;                                   // 25.7 GB of FP16 weights
const KV_TOK=2*OPT.layers*OPT.hidden*OPT.bytes;                   // K and V for every layer: 800 KiB per token
// memGiB: the datasheet's 40 GB read as 40 GiB (as in the KV cache memory lab); cudaMiB: the total CUDA reports for an A100-SXM4-40GB
const GPU={memGiB:40,bw:1555e9,peak:312e12,cudaMiB:40536};
const UTIL=0.92, OH=2*A.GiB;
const PROMPT=512, NREQ=48, MEAN=128;
const AVG_CTX=PROMPT+MEAN;                                        // tokens an average running request holds
const kvLeft=(mem,oh)=>mem*UTIL-W_BYTES-oh;                        // bytes left for the KV cache
const slotsFor=(mem,oh)=>Math.floor(kvLeft(mem,oh)/(KV_TOK*AVG_CTX));
const KV_BUDGET=kvLeft(GPU.memGiB*A.GiB,OH);
const B_MAX=slotsFor(GPU.memGiB*A.GiB,OH);                        // most slots that fit
// how the cap moves with the two memory assumptions (for the Fidelity popover)
const KV_CUDA=kvLeft(GPU.cudaMiB*A.MiB,OH), B_CUDA=slotsFor(GPU.cudaMiB*A.MiB,OH), B_OH1=slotsFor(GPU.memGiB*A.GiB,A.GiB), B_OH3=slotsFor(GPU.memGiB*A.GiB,3*A.GiB);
const CAPS=[32,128,512,1536], RATES=[0,2,4], SEEDS=[132,45,123], EFFS=[30,35,40,45,50,55,60,65,70];
const DEF={pol:'static',B:8,cap:1536,rate:0,set:0,eff:50};
const CH={...DEF};                                                // the challenge starts from the initial setup
const GOAL_THR=330, GOAL_TPOT=0.025;

/* ---------- step-time model (an estimate; see "Show the math") ---------- */
const Tw=W_BYTES/GPU.bw;                                          // one read of all weights
const kvRead=tok=>tok*KV_TOK/GPU.bw;                              // reading tok tokens of KV cache
const Tc=eff=>2*PARAMS/(GPU.peak*eff/100);                        // compute per token at eff% of peak
// linear layers: read the weights once or compute every token, whichever is slower; attention: read each decoding request's KV cache.
// bwf < 1 reads memory at that share of the datasheet bandwidth (used only for the challenge's caveat)
const stepTime=(nTok,kvTok,eff,bwf=1)=>Math.max(Tw/bwf,nTok*Tc(eff))+kvRead(kvTok)/bwf;

/* ---------- workload: seeded ---------- */
function mulberry32(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const WL={};
function workload(set,cap,rate){
  const k=set+'.'+cap+'.'+rate; if(WL[k]) return WL[k];
  const seed=SEEDS[set], r=mulberry32(seed), q=1-Math.exp(-cap/MEAN);
  const lens=[]; for(let i=0;i<NREQ;i++) lens.push(Math.max(1,Math.ceil(-MEAN*Math.log(1-r()*q))));   // inverse CDF of the capped exponential
  const arr=[0];
  if(rate>0){const g=mulberry32(seed+7919); for(let i=1;i<NREQ;i++) arr.push(arr[i-1]-Math.log(1-g())/rate);}   // Poisson arrivals
  else for(let i=1;i<NREQ;i++) arr.push(0);
  return (WL[k]={seed,lens,arr});
}

/* ---------- simulation: one engine step at a time ---------- */
const SIM=new Map();
const skey=s=>[s.pol,s.B,s.cap,s.rate,s.set,s.eff,s.bwf||1].join('.');
function simulate(s){
  const key=skey(s); const hit=SIM.get(key); if(hit) return hit;
  const wl=workload(s.set,s.cap,s.rate), B=s.B, bwf=s.bwf||1;
  const reqs=wl.lens.map((len,i)=>({i,L:len,arr:wl.arr[i],start:0,first:0,fin:0,slot:-1,gen:0}));
  const waiting=[], qev=[], ends=[], jn=[]; let nxt=0, t=0, steps=0, peak=0, pre=0;
  const arrive=()=>{while(nxt<NREQ&&reqs[nxt].arr<=t+1e-12){waiting.push(reqs[nxt]); qev.push([reqs[nxt].arr,1]); nxt++;}};
  if(s.pol==='cont'){
    const run=[], used=new Array(B).fill(false);
    for(;;){
      arrive();
      if(!run.length&&!waiting.length){if(nxt>=NREQ) break; t=reqs[nxt].arr; continue;}
      let nNew=0;
      while(waiting.length&&run.length<B){const q=waiting.shift(); q.start=t; q.slot=used.indexOf(false); used[q.slot]=true; run.push(q); qev.push([t,-1]); nNew++;}
      let kv=0, nd=0; for(const q of run) if(q.gen>0){kv+=PROMPT+q.gen; nd++;}
      peak=Math.max(peak,kv+nd+nNew*(PROMPT+1));
      const dt=stepTime(nd+nNew*PROMPT,kv,s.eff,bwf);
      if(nNew) pre+=dt;
      t+=dt; steps++; ends.push(t); jn.push(nNew);
      for(let k=run.length-1;k>=0;k--){const q=run[k]; q.gen++; if(q.gen===1) q.first=t; if(q.gen>=q.L){q.fin=t; used[q.slot]=false; run.splice(k,1);}}
    }
  } else {
    for(;;){
      arrive();
      if(!waiting.length){if(nxt>=NREQ) break; t=reqs[nxt].arr; continue;}
      const batch=waiting.splice(0,B);
      batch.forEach((q,k)=>{q.start=t; q.slot=k; qev.push([t,-1]);});
      let dt=stepTime(PROMPT*batch.length,0,s.eff,bwf); pre+=dt; t+=dt; steps++; ends.push(t); jn.push(batch.length);
      peak=Math.max(peak,batch.length*(PROMPT+1));
      for(const q of batch){q.gen=1; q.first=t; if(q.gen>=q.L) q.fin=t;}
      for(;;){
        let kv=0, na=0; for(const q of batch) if(q.gen<q.L){kv+=PROMPT+q.gen; na++;}
        if(!na) break;
        peak=Math.max(peak,kv+na);
        dt=stepTime(na,kv,s.eff,bwf); t+=dt; steps++; ends.push(t); jn.push(0);
        for(const q of batch) if(q.gen<q.L){q.gen++; if(q.gen>=q.L) q.fin=t;}
      }
    }
  }
  let mk=0, tokens=0, busy=0, lastStart=0, wsum=0; reqs.forEach(q=>{mk=Math.max(mk,q.fin); tokens+=q.L; busy+=q.fin-q.start; lastStart=Math.max(lastStart,q.start); wsum+=q.start-q.arr;});
  const lat=reqs.map(q=>q.fin-q.arr).sort((a,b)=>a-b);
  const mlat=lat.reduce((a,b)=>a+b,0)/NREQ, p90=lat[Math.ceil(0.9*NREQ)-1];
  const tp=reqs.filter(q=>q.L>1).map(q=>(q.fin-q.first)/(q.L-1)), tpot=tp.reduce((a,b)=>a+b,0)/tp.length;
  // queue: waiting count over time, and the intervals in which at least one request waits
  qev.sort((a,b)=>a[0]-b[0]||b[1]-a[1]);
  const qline=[]; const wint=[]; let w=0, open=null;
  for(const [tt,d] of qev){const before=w; w+=d; qline.push([tt,w]);
    if(before===0&&w>0) open=tt; else if(before>0&&w===0&&open!=null){if(tt>open+1e-12) wint.push([open,tt]); open=null;}}
  // per slot: requests, and idle time split into "requests were waiting" (hatched) and "nothing to start" (empty)
  const slots=[]; for(let k=0;k<B;k++) slots.push([]);
  reqs.forEach(q=>slots[q.slot].push(q));
  let idleWait=0; const hatch=[];
  slots.forEach((list,k)=>{list.sort((a,b)=>a.start-b.start); let cur=0; const gaps=[];
    for(const q of list){if(q.start>cur+1e-12) gaps.push([cur,q.start]); cur=Math.max(cur,q.fin);}
    if(mk>cur+1e-12) gaps.push([cur,mk]);
    for(const [a,b] of gaps) for(const [c,d] of wint){const x=Math.max(a,c), y=Math.min(b,d); if(y>x+1e-12){idleWait+=y-x; hatch.push([k,x,y]);}}});
  const res={s:{...s},wl,reqs,slots,hatch,qline,mk,tokens,busy,util:busy/(B*mk),idleWait:idleWait/(B*mk),mlat,p90,tpot,steps,
    peak,peakBytes:peak*KV_TOK,pre,thr:tokens/mk,lastStart,mwait:wsum/NREQ,lastArr:wl.arr[NREQ-1],ends:Float64Array.from(ends),jn:Uint16Array.from(jn)};
  SIM.set(key,res); if(SIM.size>600) SIM.delete(SIM.keys().next().value);
  return res;
}
// while a request writes its answer (after its first token), how many other requests' prompts are prefilled in its steps,
// averaged over the requests with more than one token: derived from the run, the timing model is unchanged
function waitPrefills(r){if(r._wp!=null) return r._wp;
  const cj=new Float64Array(r.jn.length+1); for(let k=0;k<r.jn.length;k++) cj[k+1]=cj[k]+r.jn[k];
  const ubE=x=>{let lo=0, hi=r.ends.length; while(lo<hi){const m=(lo+hi)>>1; if(r.ends[m]<=x+1e-9) lo=m+1; else hi=m;} return lo;};
  let sum=0, n=0; r.reqs.forEach(q=>{if(q.L<2) return; sum+=cj[ubE(q.fin)]-cj[ubE(q.first)]; n++;});
  return (r._wp=n?sum/n:0);}
// the first static batch: shortest and longest answer, and how long the shortest one's slot then sat idle
function firstBatch(r){const b=r.reqs.filter(q=>q.start===0); if(!b.length) return null;
  const lo=b.reduce((a,q)=>q.L<a.L?q:a), hi=b.reduce((a,q)=>q.L>a.L?q:a);
  return {minL:lo.L,maxL:hi.L,minFin:lo.fin,idle:hi.fin-lo.fin};}
// the step-time chart, from the same rounded constants as "Show the math" (TWD, KVD, TCD below)
const stepRows=eff=>[1,2,4,8,16,B_MAX].map(b=>{const tw=Math.max(TWD,b*TCD(eff))/1000, tkv=b*KVD/1000; return {b,tw,tkv,t:tw+tkv,tps:b/(tw+tkv)};});
const bStar=eff=>TWD/TCD(eff);                                   // requests per step at which the matrix multiplies become compute-bound
// A step in which a prompt joins b-1 decoding requests, and a plain decode step with b requests, as the page states them:
// from T_w, T_kv and T_c rounded as "Show the math" prints them, so every sentence and the formula line give the same value
// (at most a few hundredths of a millisecond from stepTime()).
const TWD=+(Tw*1000).toFixed(2), KVD=+(kvRead(AVG_CTX)*1000).toFixed(3), TCD=eff=>+(Tc(eff)*1000).toFixed(4);
const joinStep=(b,eff)=>(Math.max(TWD,(b-1+PROMPT)*TCD(eff))+(b-1)*KVD)/1000, plainStep=b=>(TWD+b*KVD)/1000;

/* ---------- state (module scope, so re-mounts and language switches keep it) ---------- */
let S={...DEF}, mode='guided', step=0, pred=null, observed={}, root=null, bound=null, inited=false, saved=null, prevR=null;
let chSubmitted=false, chReveal=false, drillI=2, drillOpen=false;
const lvTried=new Set(['static']);
const play={t:null,playing:false,raf:0,last:0,rate:0};            // t = null: the whole run is shown; rate: a one-off speed (Show me)
const cur=()=>simulate(S);
const other=()=>simulate({...S,pol:S.pol==='cont'?'static':'cont'});

/* ---------- formatting ---------- */
const num=(v,d)=>A.fmt.num(v,d);
const sec=v=>Math.abs(v)<0.05?'0 s':v.toFixed(1)+' s';            // one decimal always (4.0 s, not 4 s), except 0
const ms=v=>(v*1000).toFixed(1)+' ms';                               // one decimal always (23.0 ms, not 23 ms)
const tps=v=>A.fmt.int(v);
// formatted constants used in the copy, so the text never carries a hand-typed derived number
const F={wgb:num(W_BYTES/1e9,1)+' GB',wgib:A.fmt.B(W_BYTES),kv:A.fmt.B(KV_TOK),kvMB:A.fmt.int(KV_TOK*AVG_CTX/1e6)+' MB',bw:A.fmt.int(GPU.bw/1e9)+' GB/s',peak:(GPU.peak/1e12)+' TFLOPS',
  mem:GPU.memGiB+' GiB',util:UTIL.toFixed(2),utilPc:Math.round(UTIL*100)+'%',headPc:Math.round((1-UTIL)*100)+'%',oh:A.fmt.B(OH),kvReq:A.fmt.B(KV_TOK*AVG_CTX),pf:num(PROMPT*TCD(50),1)+' ms',
  cuda:A.fmt.int(GPU.cudaMiB)+' MiB',cudaGiB:num(GPU.cudaMiB/1024,1)+' GiB'};
// the parameter count with enough digits that T_c = 2 × params ÷ compute reproduces to 4 decimals
// R6.18: 12.9B in text (no exact-value hover, the owner's clarification of 2026-10-07), the exact count in Show the math
const paramsExact=()=>A.fmt.int(PARAMS);
const pc=v=>Math.round(v*100)+'%';
const times=v=>num(v,1);
const gib=b=>A.fmt.B(b);
const slotsTxt=n=>tx(n===1?'1 slot':`${n} slots`,`${n} 个 slot`);
const polName=(p,cap)=>p==='cont'?tx(cap?'Continuous batching':'continuous batching','continuous batching'):tx(cap?'Static batching':'static batching','static batching');
const capTxt=c=>tx(`answers capped at ${A.fmt.int(c)} tokens`,`回答上限 ${A.fmt.int(c)} 个 token`);
const rateTxt=r=>r===0?tx('all waiting at 0 s','0 s 时全部在排队'):tx(`${r} arriving per second`,`平均每秒到达 ${r} 个`);
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}"${k==='mem'?` data-tipx="${A.esc(MEM_HOW())}"`:''}>${t}</button>`;
// how the slot limit is worked out, from the values as shown, so it reproduces: usable − weights − overhead, ÷ one request's KV cache
function MEM_HOW(){const use=num(GPU.memGiB*UTIL,1)+' GiB', req=KV_TOK*AVG_CTX, reqG=num(req/A.GiB,2)+' GiB', left=gib(KV_BUDGET), ratio=num(KV_BUDGET/req,1);
  return tx(`The A100 is read as ${F.mem}, and vLLM uses at most ${F.utilPc} of it by default (gpu_memory_utilization): ${F.mem} × ${F.util} = ${use}. Minus ${F.wgib} (${F.wgb}) of weights and ${F.oh} of runtime overhead, ${left} is left for the KV cache. Each running request is counted as ${AVG_CTX} tokens (a ${PROMPT}-token prompt plus the ${MEAN}-token average answer) at ${F.kv} per token, ${F.kvReq} (${reqG}) per request. ${left} ÷ ${reqG} ≈ ${ratio}, rounded down to ${B_MAX} requests.`,
            `A100 按 ${F.mem} 计，vLLM 默认最多用 ${F.utilPc}（gpu_memory_utilization）：${F.mem} × ${F.util} = ${use}。减去权重 ${F.wgib}（${F.wgb}）和运行时开销 ${F.oh}，剩 ${left} 给 KV cache。每个在跑的请求按 ${AVG_CTX} 个 token 算（${PROMPT} 个 token 的 prompt 加上平均 ${MEAN} 个 token 的回答），每个 token ${F.kv}，一个请求 ${F.kvReq}（${reqG}）。${left} ÷ ${reqG} ≈ ${ratio}，向下取整是 ${B_MAX} 个请求。`);}
const T=(c,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${c}">${x}</span>`;
// ratio and difference of two values as they are displayed (rounded), so a reader can check them
const rnd=(v,d)=>Math.round(v*10**d)/10**d;
/* numbers in prose (R6.20): an explanation says the trend and the reason in words, with at most one key number, rounded for reading
   (23 ms, not 22.6 ms). The exact values stay on the stage, the metric cards, the step-time chart and Show the math. Every word
   below is chosen from the engine's numbers, never typed for one case */
const msR=v=>Math.round(v*1000)+' ms';
const NUM_EN={2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine'}, NUM_ZH={2:'两',3:'三',4:'四',5:'五',6:'六',7:'七',8:'八',9:'九'};
/* a step's length against another's: "more than four times as long" / "四倍多" */
const longW=q=>{const f=Math.floor(q+1e-9), over=q>f+1e-9; if(f<2) return tx('longer','更长');
  return tx(`${over?'over ':'about '}${f===2?'twice':`${NUM_EN[f]||f} times`} as long`,`${NUM_ZH[f]||f+' '}倍${over?'多':''}`);};
/* one time against another, the second shown with one decimal: "a little slower", "noticeably faster" (from the shown values) */
function paceW(p,p0){const d=p/p0-1, a=Math.abs(d);
  if(a<0.02) return {same:true,en:'about as fast',zh:'几乎一样快'};
  if(d>0) return {slower:true,en:a<0.25?'a little slower':(a<0.5?'noticeably slower':'much slower'),zh:a<0.25?'慢了一点':(a<0.5?'明显变慢':'慢了很多')};
  return {faster:true,en:a<0.25?'a little faster':'noticeably faster',zh:a<0.25?'快了一点':'明显更快'};}
/* a latency against a much longer one, as a fraction: "under a fifth of" / "不到……的五分之一" */
const FRAC={2:{en:'half',zh:'一半'},3:{en:'a third',zh:'三分之一'},4:{en:'a quarter',zh:'四分之一'},5:{en:'a fifth',zh:'五分之一'},6:{en:'a sixth',zh:'六分之一'},7:{en:'a seventh',zh:'七分之一'},8:{en:'an eighth',zh:'八分之一'},9:{en:'a ninth',zh:'九分之一'},10:{en:'a tenth',zh:'十分之一'}};
const ST_TXT={live:{en:'live',zh:'已上线'},wip:{en:'in progress',zh:'开发中'},next:{en:'next up',zh:'下一个'},planned:{en:'planned',zh:'规划中'}};
function labRef(id){const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; if(!l) return ''; const st=c.labStatus(l);
  return st==='live'?`<a href="#${id}">${esc(L(l.q))}</a>`:`${esc(L(l.q))}${tx(` (${L(ST_TXT[st]||ST_TXT.planned)})`,`（${L(ST_TXT[st]||ST_TXT.planned)}）`)}`;}

/* ---------- header copy ---------- */
const DESCS=()=>({
  guided:tx('Seven short parts. Predict first, see where static batching leaves the GPU waiting, switch to continuous batching, then run more requests at once.',
            '一共 7 个部分。先预测，再看 static batching 让 GPU 在哪里空等，然后换成 continuous batching，最后让 GPU 同时跑更多请求。'),
  sandbox:tx('Pick a real workload and compare how the two batching policies serve it.',
             '选一种真实的负载，比较两种调度方式怎么服务它。'),
  challenge:tx(`One A100 40GB serves OPT-13B, and 48 requests are waiting. Choose only the batching policy and the number of slots so that throughput reaches ${GOAL_THR} tokens per second while each output token takes at most 25 ms on average.`,
               `一张 A100 40GB 跑 OPT-13B，有 48 个请求在排队。只选调度方式和 slot 数，让吞吐达到每秒 ${GOAL_THR} 个 token，同时平均每个输出 token 不超过 25 ms。`)
});
const SRC={opt:'https://huggingface.co/facebook/opt-13b/blob/main/config.json',a100:'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf',
  anyscale:'https://www.anyscale.com/blog/continuous-batching-llm-inference',vllmcfg:'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py',
  orca:'https://www.usenix.org/conference/osdi22/presentation/yu',vllm:'https://arxiv.org/abs/2309.06180',
  vsched:'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py',ssched:'https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/managers/scheduler.py',
  vschedcfg:'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/scheduler.py',vutils:'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/utils.py',
  vworker:'https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu_worker.py',colab:'https://github.com/googlecolab/colabtools/issues/2287',torch:'https://github.com/pytorch/pytorch/issues/74522'};
const a=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
// side facts live here, not on the cards (R6.16); every number is computed
function fidPop(){
  const kvB=gib(KV_BUDGET), one=R({B:1}), s8=R({}), c8=R({pol:'cont'}), r1=stepRows(50)[0], k=Math.round(c8.thr)/Math.round(s8.thr);
  const ok90=chOkIf({bwf:0.9}), ok40=chOkIf({eff:40});
  const sens=ok90===0&&ok40===0?tx(`<li>The challenge’s goals depend on these estimates: with memory read at 90% of the datasheet’s ${F.bw}, or prefill at 40% of peak, no option would meet both.</li>`,`<li>挑战题的两个目标取决于这些估计：如果显存只按数据手册 ${F.bw} 的 90% 读取，或 prefill 只有峰值的 40%，就没有任何选项能同时达到。</li>`)
    :(ok90===0?tx(`<li>The challenge’s goals depend on these estimates: with memory read at 90% of the datasheet’s ${F.bw}, no option would meet both.</li>`,`<li>挑战题的两个目标取决于这些估计：如果显存只按数据手册 ${F.bw} 的 90% 读取，就没有任何选项能同时达到。</li>`):'');
  const dec1=one.reqs.filter(q=>q.L>1), ctx1=Math.round(PROMPT+dec1.reduce((x,q)=>x+q.L/2,0)/dec1.length);
  return tx(
`<div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>
<li>48 requests on one A100 40GB serving OPT-13B, simulated one step at a time. In each step, every running request generates one token, and a request that joins has its whole prompt prefilled in that step.</li>
<li>Static batching starts as many requests as there are slots together, and the next group starts when the longest answer of the group is done. Continuous batching lets finished requests leave after every step, and waiting ones take the free slots, first come, first served.</li>
<li>A step takes the larger of two times, reading the weights or computing its tokens, plus the time to read each running request’s KV cache (${J('math','Show the math')}).</li></ul></div>
<div><div class="eyebrow">Simplified</div><ul>
<li>The step times are estimates. Memory is read at the datasheet’s full ${F.bw}, and prefill reaches 50% of the ${F.peak} peak (${J('eff','prefill speed under Settings')}). Reading the ${F.wgb} of weights dominates a step until it processes about ${Math.round(bStar(50))} tokens.</li>
<li>A joining request’s whole prompt is prefilled in one step; it is not split over several steps (chunked prefill). Static batching is modelled generously: a request that finishes early drops out of the computation at once, so it does not slow the rest of its batch.</li>
<li>Memory is used here only to cap the slots at ${B_MAX}: each running request’s KV cache is counted as ${AVG_CTX} tokens in the ${kvB} left for it (${J('mem','how that is computed')}), and the KV cache in use fits in every setting on this page. The A100’s 40 GB is read as 40 GiB; CUDA reports ${F.cuda} (${F.cudaGiB}) for this card, and with vLLM’s rule that leaves room for ${B_CUDA} slots. With 1 GiB of runtime overhead instead of 2, the cap would be ${B_OH1} slots; with 3 GiB, ${B_OH3}.</li>
<li>The workload scales down Anyscale’s 2023 benchmark: 48 requests with 512-token prompts; answer lengths drawn from an exponential distribution with mean 128, keeping only draws at or under the cap; three fixed random seeds (${SEEDS.join(', ')}).</li>
<li>Serving one request at a time, each output token takes ${ms(one.tpot)} on average. That is a little under ${ms(r1.t)}, a step for one request with ${AVG_CTX} tokens in its KV cache, because while it decodes, a request holds its prompt plus half its answer, about ${ctx1} tokens on average.</li>
<li>Anyscale measured about 8 times the throughput of naive static batching (Hugging Face Pipelines) with continuous batching alone (text-generation-inference, Ray Serve), and up to 23 times with vLLM, which adds its memory management. Here both policies use the same 8 slots and the same kernels, so continuous batching gives ${times(k)} times the throughput of static batching.</li>${sens}</ul></div>
<div><div class="eyebrow">Sources</div><ul>
<li>OPT-13B’s shape: its ${a(SRC.opt,'config.json')} on Hugging Face.</li>
<li>A100 40GB: NVIDIA’s ${a(SRC.a100,'datasheet')} (40GB, ${F.bw}, ${F.peak} dense FP16). CUDA reports ${F.cuda} for an A100-SXM4-40GB (${a(SRC.colab,'Colab issue 2287')}) and 39.59 GiB (${a(SRC.torch,'PyTorch issue 74522')}).</li>
<li>Workload and the 8 and 23 times: ${a(SRC.anyscale,'Anyscale’s 2023 benchmark')}.</li>
<li>vLLM v0.30.0: ${a(SRC.vllmcfg,'gpu_memory_utilization')} is 0.92 by default and applies to the total CUDA reports (<code>request_memory()</code> in ${a(SRC.vutils,'vllm/v1/worker/utils.py')}); the runtime overhead is measured at startup (<code>determine_available_memory()</code> in ${a(SRC.vworker,'vllm/v1/worker/gpu_worker.py')}); max_num_seqs and max_num_batched_tokens are in ${a(SRC.vschedcfg,'vllm/config/scheduler.py')}.</li></ul>
<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-24. Prototype, not yet expert-reviewed.</p></div></div>`,
`<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>
<li>一张 A100 40GB 跑 OPT-13B，服务 48 个请求，一步一步地模拟。每一步，每个在跑的请求生成 1 个 token；刚加入的请求在这一步里做完整个 prompt 的 prefill。</li>
<li>static batching 让和 slot 数一样多的请求一起开始，这一批最长的回答写完，下一批才开始。continuous batching 在每一步结束后让写完的请求离开，排队的请求按先来先服务补进空出的 slot。</li>
<li>一步的耗时是读权重的时间和计算这一步所有 token 的时间中较长的那个，再加上读每个在跑请求的 KV cache 的时间（${J('math','展开公式')}）。</li></ul></div>
<div><div class="eyebrow">简化和假设</div><ul>
<li>每一步的耗时是估计值。显存按数据手册的满速 ${F.bw} 读取，prefill 按 ${F.peak} 峰值的 50% 计算（${J('eff','“自由调参”里的 prefill 速度')}）。一步处理的 token 不到约 ${Math.round(bStar(50))} 个时，耗时主要花在读 ${F.wgb} 的权重上。</li>
<li>新请求在加入的那一步里一次做完整个 prompt 的 prefill，不把它分到几步里做（chunked prefill）。static batching 按对它有利的方式建模：一批里先写完的请求马上退出计算，不会让这一批后面的步变慢。</li>
<li>显存在这里只用来限制 slot 数，最多 ${B_MAX} 个。每个在跑的请求的 KV cache 按 ${AVG_CTX} 个 token 算，放进留给它的 ${kvB}（${J('mem','怎么算的')}）；本页所有设置下，实际占用的 KV cache 都放得下。A100 的 40 GB 按 40 GiB 计；CUDA 对这张卡报告的是 ${F.cuda}（${F.cudaGiB}），按 vLLM 的算法只能放 ${B_CUDA} 个 slot。运行时开销如果是 1 GiB 而不是 2 GiB，能放 ${B_OH1} 个；3 GiB 时 ${B_OH3} 个。</li>
<li>负载是 Anyscale 2023 年 benchmark 的缩小版：48 个请求，prompt 各 512 个 token；回答长度按均值 128 的指数分布抽取，只保留不超过上限的；三组固定的随机种子（${SEEDS.join('、')}）。</li>
<li>一次只跑 1 个请求时，每个输出 token 平均 ${ms(one.tpot)}。这比 KV cache 里有 ${AVG_CTX} 个 token 时 1 个请求的一步（${ms(r1.t)}）略短，因为请求在 decode 时，KV cache 里平均只有 prompt 加上一半的回答，约 ${ctx1} 个 token。</li>
<li>Anyscale 测得：只用 continuous batching（text-generation-inference、Ray Serve），吞吐约是朴素 static batching（Hugging Face Pipelines）的 8 倍；vLLM 再加上它的显存管理，最高是 23 倍。这里两种方式用同样 8 个 slot、同样的 kernel，所以 continuous batching 的吞吐只是 static batching 的 ${times(k)} 倍。</li>${sens}</ul></div>
<div><div class="eyebrow">来源</div><ul>
<li>OPT-13B 的结构：Hugging Face 上的 ${a(SRC.opt,'config.json')}。</li>
<li>A100 40GB：NVIDIA ${a(SRC.a100,'数据手册')}（40GB，${F.bw}，FP16 稠密 ${F.peak}）。CUDA 对 A100-SXM4-40GB 报告 ${F.cuda}（${a(SRC.colab,'Colab issue 2287')}）和 39.59 GiB（${a(SRC.torch,'PyTorch issue 74522')}）。</li>
<li>负载，以及 8 倍和 23 倍：${a(SRC.anyscale,'Anyscale 2023 年的 benchmark')}。</li>
<li>vLLM v0.30.0：${a(SRC.vllmcfg,'gpu_memory_utilization')} 默认 0.92，乘的是 CUDA 报告的总显存（${a(SRC.vutils,'vllm/v1/worker/utils.py')} 里的 <code>request_memory()</code>）；运行时开销在启动时实测（${a(SRC.vworker,'vllm/v1/worker/gpu_worker.py')} 里的 <code>determine_available_memory()</code>）；max_num_seqs 和 max_num_batched_tokens 在 ${a(SRC.vschedcfg,'vllm/config/scheduler.py')}。</li></ul>
<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-24。原型，尚未经专家审阅。</p></div></div>`);
}
const priPop=()=>tx(
 `<p>A request is served in two phases. <b>Prefill</b> runs the whole prompt through the model in one step and produces the first answer token. <b>Decode</b> follows: each later step produces one more token, until the answer is done.</p><p>A step is one pass of the model. Every step reads all of the model’s weights from GPU memory: ${F.wgb} for OPT-13B. A step for 8 requests still reads them only once, so a server runs many requests in each step; each running request takes one slot.</p><p>Each request also keeps the keys and values of its earlier tokens in GPU memory, its KV cache. The lab ${A.shell.labLink('kv-cache')} works out how big it gets.</p>`,
 `<p>一个请求分两个阶段完成。<b>prefill</b> 在一步里把整个 prompt 送进模型，产出回答的第一个 token。之后是 <b>decode</b>：每一步只多产出一个 token，直到回答写完。</p><p>模型跑一次前向叫一步。每一步都要从显存里把全部权重读一遍，OPT-13B 是 ${F.wgb}。一步里同时算 8 个请求，这些权重也只读一遍，所以服务端每一步都同时跑很多请求，每个在跑的请求占一个 slot。</p><p>每个请求还要在显存里留着之前所有 token 的 key 和 value，也就是它的 KV cache。它会有多大，在 ${A.shell.labLink('kv-cache')}那个 Lab 里算过。</p>`);

/* ---------- guided parts ---------- */
const PRED=[{v:'same',t:{en:'Less than 1.5 times',zh:'不到 1.5 倍'}},{v:'mid',t:{en:'1.5 to 3 times',zh:'1.5 到 3 倍'}},{v:'big',t:{en:'More than 3 times',zh:'3 倍以上'}}];
const predRight=k=>k<1.5?'same':(k<=3?'mid':'big');
const R=p=>simulate({...DEF,...p});
const leverNow=()=>{const keys=['B','cap','rate','set','eff']; if(!keys.every(k=>S[k]===DEF[k])) return null; return S.pol;};
const stats=wl=>{const s=wl.lens.reduce((x,y)=>x+y,0); return {min:Math.min(...wl.lens),max:Math.max(...wl.lens),avg:s/NREQ};};
const SET=()=>J('controls',tx('Settings','“自由调参”'));                 // the settings panel, as a jump link (R1.10)
let ran3=false;
/* R3.20: the runs the reader has played (or scrubbed) to the end, in any part or mode, by their settings (the share token), not by part;
   part 3 shows a run already watched (in Scenarios, say) whole at once, and its gate counts it as run */
const watched=new Set();
const STEPS=[
 {title:{en:'Your mission',zh:'你的任务'},lead:{en:'One A100, OPT-13B, 48 requests waiting: how should they be scheduled?',zh:'一张 A100、OPT-13B、48 个请求在排队：怎么调度？'},enter:()=>({...DEF}),
  body:()=>{const w=stats(workload(0,1536,0));
   // the setup keeps its set values; the answers' range and mean (derived) are on the stage under the title (R6.20)
   const spread=w.min<10&&w.max>=100&&w.max<1000;
   return tx(`<p>You serve OPT-13B on one A100 40GB, and 48 requests are waiting. Each has a 512-token prompt, and the answers ${spread?'range from a few tokens to a few hundred':'vary a lot in length'}.</p><p>As the lab ${A.shell.labLink('kv-cache')} showed, memory caps how many requests can run at once: ${B_MAX} here. The server starts with a limit of 8, and each running request takes one slot. One pass of the model is a step: in each step, every running request generates one more token.</p><p>Your job: finish all 48 requests as fast as this GPU allows. The scheduler decides it: when a request finishes, how soon does its slot go to a waiting request?</p>`,
             `<p>你用一张 A100 40GB 跑 OPT-13B，有 48 个请求在排队。每个请求的 prompt 是 512 个 token；回答${spread?'从几个 token 到几百个 token 不等':'长短差别很大'}。</p><p>${A.shell.labLink('kv-cache')}那个 Lab 讲过，显存决定了最多能同时跑几个请求：这里最多 ${B_MAX} 个。服务端先设成最多同时跑 8 个，每个在跑的请求占一个 slot。模型每跑一次前向叫一步：每一步，每个在跑的请求多生成一个 token。</p><p>你的任务：让这张 GPU 尽快做完这 48 个请求。关键在调度器：一个请求写完后，它的 slot 什么时候交给排队的请求？</p>`);}},
 {title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: how many times static batching’s throughput does continuous batching reach?',zh:'先猜：continuous batching 每秒的 token 数是 static 的几倍？'},gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${L(o.t)}</button>`).join('')}</div>`;
   return tx(`<p>With the same 8 slots, there are two ways to schedule:</p><ul><li><b>Static batching</b> starts 8 requests together; the next 8 start only when the longest answer of the 8 is done.</li><li><b>Continuous batching</b> checks after every step: a request that just finished leaves, and a waiting request takes its slot at once.</li></ul><p>How many times static batching’s tokens per second does continuous batching make?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it until part 4 shows the result.</p>':''}`,
             `<p>同样 8 个 slot，有两种调度方式：</p><ul><li><b>static batching</b>：8 个请求一起开始，等这 8 个里最长的回答写完，下一批 8 个才开始。</li><li><b>continuous batching</b>：每一步结束都看一眼，刚写完的请求离开，排队的请求马上补进它的 slot。</li></ul><p>continuous batching 每秒生成的 token 数，是 static batching 的多少倍？</p>${ch}${pred?'<p class="muted small">已记下。第 4 部分给出结果之前，还可以改。</p>':''}`);}},
 {title:{en:'Run static batching',zh:'先跑 static batching'},lead:{en:'Press Play: short answers finish first, and their slots sit idle.',zh:'点播放：短的回答先写完，它的 slot 就空等着。'},enter:()=>({...DEF}),target:['cb-ic-play'],expect:()=>ran3,showMe:'play',label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Press <b>Play</b> to run the 48 requests through static batching. In the first batch the short answers finish first, and their slots stay idle until the longest answer of the batch is done.</p>`,
              `<p>点<b>播放</b>，让 48 个请求走一遍 static batching。第一批里，短的回答先写完，它们的 slot 要一直空到这一批最长的回答写完。</p>`)
    +`<div class="inline-ctl" id="cb-ic-play"><span class="lbl">${esc(icLabel())}</span><div class="cb-ic-row"><button type="button" class="btn sm" data-act="play" data-tipx="${esc(PLAYTIP())}">${playLabel()}</button>${speedCtl()}</div></div>`,
  // one key number (how busy the slots were); the finish time and the throughput are on the badge and the cards, and each
  // request's own times in its tooltip (R6.20)
  obs:r=>{const lo=r.util<0.6;
    let h=tx(`<p><b>With ${polName(r.s.pol)}, the ${slotsTxt(r.s.B)} were busy ${lo?'only ':''}${pc(r.util)} of the time.</b></p>`,
             `<p><b>用 ${polName(r.s.pol)} 时，${slotsTxt(r.s.B)} ${lo?'只有':'有'} ${pc(r.util)} 的时间在干活。</b></p>`);
    const b=r.s.pol==='static'&&r.s.rate===0?firstBatch(r):null;
    if(b&&b.idle>0){const long=b.idle>b.minFin;
      h+=tx(`<p>In the first batch, the shortest answer was done ${long?'long ':''}before the longest one, and its slot sat idle all that time (hatched) while the other requests waited.</p>`,
            `<p>第一批里，最短的回答比最长的${long?'早很久':'早'}就写完了；之后它的 slot 一直空着（图里的斜纹），其余请求只能排队。</p>`);}
    return h;}},
 {title:{en:'Switch to continuous batching',zh:'换成 continuous batching'},lead:{en:'Same 48 requests, same 8 slots; only the policy changes.',zh:'同样 48 个请求、同样 8 个 slot，只换调度方式。'},enter:()=>({...DEF}),levers:true,target:['cb-ic-pol'],expect:()=>S.pol==='cont',showMe:{pol:'cont'},label:{en:'Result',zh:'结果'},
  body:()=>{const lv=A.ui.levers({aria:tx('Batching policy, pick one','调度方式，选一种'),value:leverNow()||'',options:[
      {v:'static',title:tx('Static','static'),desc:tx('initial setup: the next 8 start only when the whole batch is done','初始设置：整批做完，下一批 8 个才开始'),result:lvResult('static')},
      {v:'cont',title:tx('Continuous','continuous'),desc:tx('a waiting request takes a free slot at the next step','有 slot 空出来，下一步就补进一个排队的请求'),result:lvResult('cont')}]});
   return tx(`<p>Same 48 requests and the same 8 slots; only the batching policy changes.</p>`,
             `<p>同样 48 个请求、同样 8 个 slot，只换调度方式。</p>`)+`<div class="cb-lv" id="cb-ic-pol">${lv}</div>`;},
  obs:()=>{const i=leverNow(), s=R({}), c=R({pol:'cont'});
   if(!i) return tx(`<p>You changed a setting under ${SET()}, so the setup matches neither option. Pick an option to get back to the same comparison.</p>`,`<p>你在${SET()}里改了设置，现在的配置不属于上面任何一个选项。选一个选项，就回到同样的对比。</p>`);
   if(i==='static') return tx(`<p><b>The initial setup, static batching, makes ${tps(s.thr)} tokens per second</b>; all 48 requests are done at ${sec(s.mk)}. Pick continuous batching to compare.</p>`,`<p><b>初始设置 static batching 每秒生成 ${tps(s.thr)} 个 token</b>，48 个请求在 ${sec(s.mk)} 全部完成。选 continuous batching 来比较。</p>`);
   // result (the prediction's answer, the one key number), the verdict, then the reason and the cost in words (R3.10, R6.20);
   // the two runs' own numbers are on the options above, the cards and the dashed line; the step times are in Show the math
   const k=Math.round(c.thr)/Math.round(s.thr), j=joinStep(8,50), d=plainStep(8), pace=paceW(rnd(c.tpot*1000,1),rnd(s.tpot*1000,1));
   const pt=pred?L(PRED.find(p=>p.v===pred).t):'', right=pred&&pred===predRight(k), rt=L(PRED.find(p=>p.v===predRight(k)).t);
   const verdict=!pred?'':(right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
     :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`));
   return tx(`<p><b>Continuous batching makes ${times(k)} times as many tokens per second as the initial setup.</b></p>`,`<p><b>continuous batching 每秒生成的 token 是初始设置的 ${times(k)} 倍。</b></p>`)+verdict+
     tx(`<p>Why: while requests are waiting, no slot is idle. The slots are busy ${pc(c.util)} of the time instead of ${pc(s.util)}; the only idle time is at the end.</p><p>The cost: each output token comes ${pace.en}, about ${msR(c.tpot)} instead of about ${msR(s.tpot)}. When a request joins, its whole prompt runs through the model (its prefill) in the same step as the others’ next tokens, so that step takes ${longW(j/d)} as one with no request joining.</p>`,
        `<p>原因：只要还有请求在排队，就没有 slot 空着。slot 在干活的时间从 ${pc(s.util)} 升到 ${pc(c.util)}，只在最后才有空闲。</p><p>代价：每个输出 token ${pace.zh}，从约 ${msR(s.tpot)} 变成约 ${msR(c.tpot)}。新请求加入的那一步，除了其他请求的下一个 token，还要把它整个 prompt 过一遍模型（prefill），所以这一步的耗时是没有新请求加入时的${longW(j/d)}。</p>`);}},
 {title:{en:'Why more requests barely slow a step',zh:'为什么多放几个请求，一步几乎不变慢'},lead:{en:'Each step reads all the weights once; an extra request adds only its KV cache.',zh:'每一步读一遍全部权重；多一个请求只多读它的 KV cache。'},enter:()=>({...DEF,pol:'cont'}),
  // the principle in words (R6.20); the step times are on the chart right below, the reads and their sizes in Show the math
  body:()=>{const rows=stepRows(S.eff), r1=rows[0], r8=rows.find(x=>x.b===8), little=r8.t<1.25*r1.t;
   return tx(`<p>In every step the GPU reads all of OPT-13B’s weights from memory once, and that read takes most of the step. Each extra request in the step adds only a read of its own KV cache, which is small next to the weights (the two parts of each bar below).</p><p>So a step with 8 requests takes ${little?'only a little longer':'longer'} than a step with one, yet makes 8 times as many tokens. A slot that static batching leaves idle throws away a token that would have cost almost no time.</p>`,
             `<p>每一步，GPU 都要把 OPT-13B 的全部权重从显存读一遍，这占了一步的大部分时间。一步里每多一个请求，只多读这个请求自己的 KV cache，和权重比起来很小（看下面每根条的两段）。</p><p>所以一步里放 8 个请求，耗时只比放一个请求多${little?'一点':'一些'}，产出的 token 却是它的 8 倍。static batching 让 slot 空着，就是白白放弃了这些几乎不花时间的 token。</p>`)+stepChartHTML(S.eff);}},
 {title:{en:'Grow the batch',zh:'把 batch 加大'},lead:{en:'Drag the slots up from 8 and compare throughput and time per token.',zh:'把 slot 数从 8 往上拖，比较吞吐和每个 token 的耗时。'},enter:()=>({...DEF,pol:'cont'}),target:['cb-ic-B'],expect:()=>S.pol==='cont'&&S.B!==8,showMe:{B:16},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Continuous batching keeps every slot busy, so more slots mean more tokens per step. Drag the slots up from 8 and compare the throughput and the time per output token.</p>`,
              `<p>continuous batching 让每个 slot 都有活干，所以 slot 越多，每一步产出的 token 越多。把 slot 数从 8 往上拖，比较吞吐和每个输出 token 的耗时。</p>`)
    +A.ui.range({id:'cb-ic-B',key:'B',label:tx('Slots','slot 数'),min:1,max:B_MAX,value:S.B,out:slotsTxt(S.B),inline:true}),
  obs:r=>{
   if(r.s.pol!=='cont') return tx(`<p>This part compares slot counts under continuous batching, and you switched to static batching under ${SET()}. <button type="button" class="linkbtn" data-act="to-cont">Switch back to continuous batching</button>.</p>`,
                                   `<p>这一部分比较的是 continuous batching 下不同的 slot 数，而你在${SET()}里切到了 static batching。<button type="button" class="linkbtn" data-act="to-cont">切回 continuous batching</button>。</p>`);
   const b=R({pol:'cont',cap:r.s.cap,rate:r.s.rate,set:r.s.set,eff:r.s.eff}), B=r.s.B;
   // against 8 slots, from the values as displayed: the throughput as a ratio (the key number), the time per output token in words;
   // the exact values are on the cards with their ▲▼, the step times and the prefills a request waits through in Show the math (R6.20)
   const tR=Math.round(r.thr), tB=Math.round(b.thr), pace=paceW(rnd(r.tpot*1000,1),rnd(b.tpot*1000,1));
   // the memory cap is said only where the slider stops (part 1 and the Slots note already give the number)
   const cap=B===B_MAX?tx(`<p>${B_MAX} slots is the limit: memory holds the KV cache of only ${B_MAX} requests (${J('mem','how that is computed')}).</p>`,
                          `<p>${B_MAX} 个 slot 是上限：显存只放得下 ${B_MAX} 个请求的 KV cache（${J('mem','怎么算的')}）。</p>`):'';
   if(B===8) return tx(`<p><b>With 8 slots, continuous batching makes ${tps(b.thr)} tokens per second.</b></p>`,`<p><b>8 个 slot 时，continuous batching 每秒生成 ${tps(b.thr)} 个 token。</b></p>`);
   const q=tR/tB, conj=pace.slower?tx('but','但'):tx('and','而且');
   if(B>8){
     // a ratio that rounds to 1.0 says nothing: then it says "a few more"
     const thr=q>=1.05?tx(`${times(q)} times as many tokens per second as with 8`,`每秒生成的 token 是 8 个 slot 时的 ${times(q)} 倍`)
                      :(tR>tB?tx('a few more tokens per second than with 8','每秒生成的 token 比 8 个 slot 时多一点'):tx('about as many tokens per second as with 8','每秒生成的 token 和 8 个 slot 时差不多'));
     const j=joinStep(B,r.s.eff), d=plainStep(B), several=j/d>=2.5;
     return tx(`<p><b>With ${B} slots, continuous batching makes ${thr}</b>, ${conj} each output token comes ${pace.en}.</p>`,
               `<p><b>${B} 个 slot 时，continuous batching ${thr}</b>，${conj}每个输出 token ${pace.zh}。</p>`)+
       (pace.slower?tx(`<p>Each output token gets slower for two reasons. First, every step reads the KV cache of more requests. Second, other requests join more often: each joining prompt’s prefill makes its step ${several?'several times':'much'} longer, and every running request waits through it.</p>`,
                       `<p>每个输出 token 变慢有两个原因。一是每一步要读更多请求的 KV cache。二是别的请求加入得更频繁：每个新 prompt 的 prefill 都让那一步长${several?'好几倍':'很多'}，在跑的请求都得等它。</p>`):'')+cap;}
   const thr=tR<tB?tx('fewer tokens per second than with 8','每秒生成的 token 比 8 个 slot 时少'):(tR>tB?tx('a few more tokens per second than with 8','每秒生成的 token 比 8 个 slot 时多一点'):tx('as many tokens per second as with 8','每秒生成的 token 和 8 个 slot 时一样多'));
   return tx(`<p><b>With ${slotsTxt(B)}, continuous batching makes ${thr}</b>, ${pace.faster?'though':'and'} each output token comes ${pace.en}.</p>`+(tR<tB?'<p>Fewer slots make each step a little shorter, but each step also makes fewer tokens.</p>':''),
             `<p><b>${slotsTxt(B)} 时，continuous batching ${thr}</b>，${pace.faster?'不过':'而且'}每个输出 token ${pace.zh}。</p>`+(tR<tB?'<p>slot 少了，每一步短一点，但每一步产出的 token 也少了。</p>':''));}},
 {title:{en:'What to remember',zh:'要记住的'},
  // the trend in words (R6.20); the bullets state the pattern and what follows, not a number from an earlier part (G2, the owner 2026-10-08);
  // the misconception keeps the latency as a fraction, its conditions restated in full
  body:()=>{const c8=R({pol:'cont'}), sMax=R({B:B_MAX}), one=R({B:1});
   const qf=Math.floor(one.mlat/c8.mlat+1e-9), fr=FRAC[qf], lower=Math.round(sMax.thr)<Math.round(c8.thr);
   const lat=fr?tx(`is under ${fr.en} of that with one request at a time`,`不到一次只跑一个请求时的${fr.zh}`):tx('is far lower than when one request runs at a time','比一次只跑一个请求时低得多');
   return tx(`<ul><li>A step reads all the weights once, so a few more requests barely slow it.</li><li>Static batching leaves a finished request’s slot idle until the longest answer of its batch is done. Continuous batching gives that slot to a waiting request at the next step, so the same slots make more tokens per second. The more the answer lengths differ, the bigger the gain.</li><li>More slots raise throughput and slow each output token; memory fits at most ${B_MAX} slots.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Batching adds latency for everyone.” Each output token does get slower, but a request waits far less for a slot. With 48 requests queued, its mean latency with continuous batching on 8 slots ${lat}.</p><p>✗ “Static batching with a big batch is good enough.” Even with all ${B_MAX} slots, static batching makes ${lower?'fewer tokens per second than':'only about as many tokens per second as'} continuous batching with 8.</p></div>${A.shell.toScenarios()}`,
             `<ul><li>一步只读一遍全部权重，所以多放几个请求，一步几乎不变慢。</li><li>static batching 让写完的请求的 slot 空等这一批最长的回答；continuous batching 在下一步就把它交给排队的请求，所以同样多的 slot，每秒能生成更多的 token。回答的长短差得越多，多出来的越多。</li><li>slot 越多，吞吐越高，每个输出 token 也越慢；显存最多放得下 ${B_MAX} 个 slot。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “batching 会让每个人都变慢。”每个输出 token 确实会变慢，但请求等 slot 的时间少得多。48 个请求排队时，用 8 个 slot 的 continuous batching，平均延迟${lat}。</p><p>✗ “static batching 只要 batch 够大就行。”这里 static batching 用满 ${B_MAX} 个 slot，吞吐${lower?'仍然不如 8 个 slot 的 continuous batching':'也只和 8 个 slot 的 continuous batching 差不多'}。</p></div>${A.shell.toScenarios()}`);}}
];
function lvResult(p){return lvTried.has(p)?tx(`${tps(R({pol:p}).thr)} tokens/s`,`${tps(R({pol:p}).thr)} token/s`):'';}
function playLabel(){return play.playing?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放');}
/* R3.20: the run is never replayed on its own; Play is its replay control, and its tooltip says so */
const PLAYTIP=()=>tx('Plays the run from the moment shown. When the whole run is drawn, Play starts it again from 0 s.','从图上当前的时刻往后播放；整次运行都已画出时，点“播放”会从 0 s 重新开始。');
const icLabel=()=>tx(`${polName(S.pol,true)}, ${slotsTxt(S.B)}`,`${polName(S.pol)}，${slotsTxt(S.B)}`);
/* playback speed, in seconds of the run played per real second; time-based, so it is the same on any display (R3.9) */
const SPEEDS={slow:0.5,med:1.5,fast:5};
let speed='med';
const speedCtl=()=>`<div class="cb-speed" data-tipx="${esc(tx(`Slow plays ${SPEEDS.slow} s of the run in each real second, Medium ${SPEEDS.med} s and Fast ${SPEEDS.fast} s.`,`慢速时，现实中的 1 秒播放这次运行里的 ${SPEEDS.slow} s；中速 ${SPEEDS.med} s，快速 ${SPEEDS.fast} s。`))}"><span class="cb-speed-l">${tx('Speed','速度')}</span>`+
  `<div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="speed">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${speed===v}">${x}</button>`).join('')}</div></div>`;
function setSpeed(v){if(!SPEEDS[v]) return; speed=v; if(root) $$('.seg-ctl[data-key="speed"]',root).forEach(g=>A.ui.setPressed(g,speed));}

/* ---------- step-time chart ---------- */
function stepChartHTML(eff){
  const rows=stepRows(eff), mx=Math.max(...rows.map(r=>r.t));
  const title=tx(`How long is a step with more requests in it? (OPT-13B on one A100 40GB, ${AVG_CTX} tokens of KV cache per request)`,
                 `一步里的请求越多，一步要多久？（一张 A100 40GB 跑 OPT-13B，每个请求的 KV cache 按 ${AVG_CTX} 个 token 算）`);
  const val=r=>tx(`${ms(r.t)} per step, ${r.b===1?'1 token':`${r.b} tokens`}`,`一步 ${ms(r.t)}，产出 ${r.b} 个 token`);
  const aria=title+'. '+rows.map(r=>tx(`${r.b===1?'1 request':`${r.b} requests`}: ${val(r)}`,`${r.b} 个请求：${val(r)}`)).join(tx('; ','；'));
  return `<div class="cb-stc" role="img" aria-label="${esc(aria)}"><div class="cb-stc-title">${title}</div>`+
    rows.map(r=>`<div class="cb-stc-row" aria-hidden="true"><div class="cb-stc-lab"><span>${tx(r.b===1?'1 request':`${r.b} requests`,`${r.b} 个请求`)}</span><span class="mono">${val(r)}</span></div>`+
      `<div class="cb-stc-bar"><i class="cb-stc-w" style="width:${(r.tw/mx*100).toFixed(2)}%"></i><i class="cb-stc-kv" style="width:${(r.tkv/mx*100).toFixed(2)}%"></i></div></div>`).join('')+
    `<div class="cb-stc-key" aria-hidden="true"><span><i class="cb-sw cb-sw-w"></i>${tx('reading the weights','读权重')}</span><span><i class="cb-sw cb-sw-kv"></i>${tx('reading each request’s KV cache','读每个请求的 KV cache')}</span></div></div>`;
}

/* ---------- show the math ---------- */
// every line reproduces from the values it shows: results that are sums or products are computed from the displayed inputs
const dec=(v,d)=>(v).toFixed(d);
// fewest decimals of the run time (and of the busy slot time) with which the shown division gives the shown result
function mkDigits(r,f){for(const d of [2,3,4]){if(Math.round(f(+r.mk.toFixed(d)))===Math.round(f(r.mk))) return d;} return 4;}
function busyDigits(r){const want=Math.round(100*r.util);
  for(const db of [1,2,3]) for(const dm of [2,3,4]){const b=+r.busy.toFixed(db), m=+r.mk.toFixed(dm); if(Math.round(100*b/(r.s.B*m))===want) return [db,dm];} return [3,4];}
/* H1/H2 (the owner, 2026-10-09): Show the math as short blocks, like a page of a good textbook. Each block has one lead line in words,
   then a small table: the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the
   result, each on its own line with the "=" signs lined up. terms: one TeX string per column; rows: {l, c:[html…]} or {r:html}. */
const MX=x=>A.ui.math(x);
function mathBlock(lead,name,terms,rows,note){const n=terms.length, eq=`<td class="cb-meq">${MX('=')}</td>`;
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?r.c.map(x=>`<td>${x}</td>`).join(''):`<td colspan="${n}" class="cb-mres">${r.r}</td>`}</tr>`;
  return `<div class="cb-mb">${lead?`<p class="cb-mlead">${lead}</p>`:''}<div class="cb-fx"><table class="cb-mt"><tr class="cb-mname"><th scope="row">${name}</th>${eq}${terms.map(x=>`<td>${MX(x)}</td>`).join('')}</tr>${rows.map(tr).join('')}</table></div>${note?`<p class="cb-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.cb-mt'); if(!tb) return; const e0=tb.querySelector('td.cb-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('cb-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('cb-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="cb-mc${r?'':' cb-mc0'}">${x}</span>`;
    h+=`<tr class="cb-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="cb-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="cb-mband${g.tall?' cb-mtall':''}"><td class="cb-meq"><span class="cb-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="cb-mfl">${units.map(u=>`<span class="cb-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="cb-meq">${E}</td><td class="cb-mres">${r.l?`<span class="cb-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='cb-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.cb-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.cb-fx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
const mathHead=x=>`<p class="cb-mhead">${x}</p>`;
function mathLines(r){
  const s=r.s;
  const twD=TWD, kvD=KVD, tcD=TCD(s.eff);
  const tbD=plainStep(s.B)*1000, j=s.B-1, joinD=joinStep(s.B,s.eff)*1000;     // t(b) printed with 3 decimals: the exact sum of the printed terms
  const d1=mkDigits(r,m=>r.tokens/m), [db,d2]=busyDigits(r), bL=`b = ${T('B',s.B)}`;
  return [
   mathHead(tx('How long one step takes','一步要多久')),
   mathBlock(tx('Every decode step reads all the weights from GPU memory once, however many requests it serves.','每个 decode 步都要把全部权重从显存读一遍，不管这一步服务多少个请求。'),
     `${tx('Weights read per step','每步读权重的时间')} ${MX('T_w')}`,[tx('\\text{weight bytes}','\\text{权重的字节数}'),'\\div',tx('\\text{bandwidth}','\\text{带宽}')],
     [{c:[T('params',tx(`${paramsExact()} params × 2 bytes`,`${paramsExact()} 个参数 × 2 字节`)),'÷',T('bw',F.bw)]},{c:[F.wgb,'÷',F.bw]},{r:`<b>${dec(twD,2)} ms</b>`}]),
   mathBlock(tx('Each request also reads its own KV cache in every step.','每个请求在每一步还要读自己的 KV cache。'),
     `${tx('KV cache read per request','每个请求读 KV cache 的时间')} ${MX('T_{kv}')}`,[tx('\\text{KV cache per token}','\\text{每个 token 的 KV cache}'),'\\times',tx('\\text{tokens}','\\text{token 数}'),'\\div',tx('\\text{bandwidth}','\\text{带宽}')],
     [{c:[T('kvtok',F.kv),'×',T('ctx',tx(`${AVG_CTX} tokens`,`${AVG_CTX} 个 token`)),'÷',T('bw',F.bw)]},{r:`${F.kvReq}${tx(` (${F.kvMB})`,`（${F.kvMB}）`)} ÷ ${F.bw}`},{r:`<b>${dec(kvD,3)} ms</b>`}],
     tx(`${AVG_CTX} tokens are the 512-token prompt plus 128 tokens, the mean the answer lengths are drawn with.`,`${AVG_CTX} 个 token 是 512 个 token 的 prompt 加上 128 个 token，即抽取回答长度时设定的均值。`)),
   mathBlock(tx('A step with b requests and no new prompt reads the weights once and each request’s KV cache once.','一步里有 b 个请求、没有新 prompt 加入时，权重读一遍，每个请求的 KV cache 各读一遍。'),
     `${tx('Step with b requests, none joining','b 个请求、没有新请求加入的一步')} ${MX('t(b)')}`,['T_w','+','b','\\times','T_{kv}'],
     [{l:bL,c:[`${dec(twD,2)} ms`,'+',`${s.B}`,'×',`${dec(kvD,3)} ms`]},{r:`<b>${+tbD.toFixed(3)} ms</b>`}],
     tx(`So ${s.B} requests get at most ${s.B} tokens every ${+tbD.toFixed(3)} ms, ${tps(s.B/(tbD/1000))} tokens per second.`,`所以 ${s.B} 个请求每 ${+tbD.toFixed(3)} ms 最多拿到 ${s.B} 个 token，也就是每秒最多 ${tps(s.B/(tbD/1000))} 个。`)),
   mathBlock(tx('Computing one token takes 2 FLOPs per parameter, at the share of the peak the GPU reaches.','算一个 token，每个参数要 2 次运算，速度按 GPU 实际达到的那部分峰值算。'),
     `${tx('Compute per token','每个 token 的计算时间')} ${MX('T_c')}`,[tx('2 \\times \\text{params}','2 \\times \\text{参数数}'),'\\div',tx('(\\text{peak} \\times \\text{efficiency})','(\\text{峰值} \\times \\text{效率})')],
     [{c:[`2 × ${T('params',paramsExact())} FLOP`,'÷',`(${T('peak',F.peak)} × ${T('eff',s.eff+'%')})`]},{r:`<b>${dec(tcD,4)} ms</b>`}],
     tx(`A step is compute-bound only when its arithmetic takes longer than reading the weights, that is above ${MX('T_w \\div T_c')} = ${dec(twD,2)} ÷ ${dec(tcD,4)} ≈ ${Math.round(twD/tcD)} tokens per step.`,`只有一步的计算比读权重还久，这一步才受算力限制，也就是每步超过 ${MX('T_w \\div T_c')} = ${dec(twD,2)} ÷ ${dec(tcD,4)} ≈ ${Math.round(twD/tcD)} 个 token 时。`)),
   mathBlock(tx('A new prompt is computed in one step, all its tokens at once.','新 prompt 在一步里算完，它的 token 一起算。'),
     tx('Prefill of one prompt','一个 prompt 的 prefill'),[tx('\\text{prompt tokens}','\\text{prompt 的 token 数}'),'\\times','T_c'],
     [{c:[T('prompt','512'),'×',`${dec(tcD,4)} ms`]},{r:`<b>${dec(512*tcD,1)} ms</b>`}]),
   mathBlock(tx('When a new prompt joins, its step computes the prompt with the other requests’ newest tokens, and takes the longer of computing and reading the weights.','有新 prompt 加入时，这一步要把 prompt 和其他请求的最新 token 一起算。计算和读权重同时进行，取较长的那个。'),
     tx('Step in which a new prompt joins b − 1 decoding requests','一个新 prompt 加入、另有 b − 1 个请求在 decode 的一步'),['\\max(T_w,\\ (b - 1 + 512) \\times T_c)','+','(b - 1)','\\times','T_{kv}'],
     [{l:bL,c:[`max(${dec(twD,2)} ms, ${j+PROMPT} × ${dec(tcD,4)} ms)`,'+',`${j}`,'×',`${dec(kvD,3)} ms`]},{r:`<b>${dec(joinD,1)} ms</b>`}]),
   mathHead(tx('This run','这次运行')),
   // part 6's second reason, counted in this run (R6.20 moved the count out of the card's text)
   `<div class="cb-mb"><p class="cb-mlead">${tx('Other requests’ prefills that one request waits through while it writes its answer (this run, on average)','一个请求写回答期间，平均要等别的请求 prefill 的次数（这次运行）')}${tx(': ','：')}<b>${num(waitPrefills(r),1)}</b></p></div>`,
   mathBlock(`${T('set',tx(`48 requests, set ${s.set+1}`,`48 个请求，第 ${s.set+1} 组`))}${tx(', ','，')}${T('pol',polName(s.pol))}${tx('.','。')}`,
     tx('Tokens per second','每秒的 token 数'),[tx('\\text{answer tokens}','\\text{回答 token 数}'),'\\div',tx('\\text{run time}','\\text{运行时间}')],
     [{c:[`${A.fmt.int(r.tokens)} ${tx('answer tokens','个回答 token')}`,'÷',`${dec(r.mk,d1)} s`]},{r:`<b>${tx(`${tps(r.thr)} tokens/s`,`每秒 ${tps(r.thr)} 个 token`)}</b>`}]),
   mathBlock(tx('The share of the slots’ time spent writing tokens.','slot 的时间里，有多少在写 token。'),
     tx('Slots busy','slot 忙碌比例'),[tx('\\text{slot time in use}','\\text{slot 在干活的时间}'),'\\div',tx('(\\text{slots} \\times \\text{run time})','(\\text{slot 数} \\times \\text{运行时间})')],
     [{c:[`${dec(r.busy,db)} s`,'÷',`(${T('B',slotsTxt(s.B))} × ${dec(r.mk,d2)} s)`]},{r:`<b>${pc(r.util)}</b>`}]),
   mathHead(tx('Memory','显存')),
   mathBlock(tx('The slots are limited by the memory left for the KV cache.','slot 的上限由留给 KV cache 的显存决定。'),
     tx('Most slots','slot 上限'),[tx('\\frac{\\text{GPU memory} \\times \\text{share used} - \\text{weights} - \\text{overhead}}{\\text{KV cache per token} \\times \\text{tokens}}','\\frac{\\text{显存} \\times \\text{可用比例} - \\text{权重} - \\text{开销}}{\\text{每个 token 的 KV cache} \\times \\text{token 数}}')],
     [{c:[`(${T('gmem',F.mem)} × ${T('util',F.util)} − ${F.wgib} (${F.wgb}) − ${T('oh',F.oh)}) ÷ (${F.kv} × ${T('ctx',AVG_CTX)})`]},{r:`${gib(KV_BUDGET)} ÷ ${F.kvReq}`},{r:`<b>${B_MAX}</b>`}]),
   mathBlock(tx('The most tokens this run kept in the KV cache at one time.','这次运行里，KV cache 同时存着的 token 最多时有多少。'),
     tx('KV cache at the peak of this run','这次运行里 KV cache 的峰值'),[tx('\\text{tokens at the peak}','\\text{峰值时的 token 数}'),'\\times',tx('\\text{KV cache per token}','\\text{每个 token 的 KV cache}')],
     [{c:[`${A.fmt.int(r.peak)} ${tx('tokens','个 token')}`,'×',F.kv]},{r:`<b>${gib(r.peakBytes)}</b> `+
     (r.peakBytes<=KV_BUDGET?`<span class="okc">${tx(`fits in ${gib(KV_BUDGET)}`,`放得进 ${gib(KV_BUDGET)}`)}</span>`:`<span class="badc">${tx(`over ${gib(KV_BUDGET)}`,`超出 ${gib(KV_BUDGET)}`)}</span>`)}])
  ];
}

/* ---------- view ---------- */
// legend: [swatch class, label, tooltip, id of an item shown only when its mark is drawn]
const LEG=()=>[
 ['cb-sw-pf',tx('prefill','prefill'),'','cb-lg-pf'],
 ['cb-sw-dec',tx('decode','decode'),tx('Decode is every later step of a request: it generates one more token per step until its answer is done.','decode 是一个请求之后的每一步：每一步多生成 1 个 token，直到回答写完。')],
 ['cb-sw-idle',tx('idle while requests wait','有请求在等时空着'),tx('Hatched means the slot is idle although requests are waiting. Static batching causes this: a new request cannot start until the whole batch is done.','斜纹表示 slot 空着，而还有请求在排队。static batching 会这样：整批做完之前，新请求不能开始。')],
 ['cb-sw-emp',tx('empty','空'),tx('Blank means the slot is idle and no request is waiting, as near the end of a run.','空白表示 slot 空着，也没有请求在等，比如快结束时。')],
 ['cb-sw-q',tx('requests waiting','排队的请求'),tx('The strip above the slots shows how many requests are waiting for a slot; its full height is 48 requests.','slot 上方的这一条显示有多少个请求在等 slot，满格是 48 个。')],
 ['cb-sw-ghost','','','cb-lg-ghost'],
 ['cb-sw-head',tx('now','当前时刻'),tx('The thick line is the moment the replay has reached.','粗线是回放现在走到的时刻。'),'cb-lg-head']];
// dp: the decimals the card shows, so a change (▲▼) is computed from the shown values; pts: a change in percentage points
const MET=[
 {k:'thr',label:()=>tx('Throughput','吞吐'),v:r=>[tps(r.thr),tx('tokens/s','token/s')],raw:r=>r.thr,dp:0,
  sub:r=>{const t=dec(r.mk,mkDigits(r,m=>r.tokens/m))+' s'; return tx(`The 48 answers, ${A.fmt.int(r.tokens)} tokens in all, took ${t}.`,`48 个回答共 ${A.fmt.int(r.tokens)} 个 token，用了 ${t}。`);},
  tip:()=>tx('Throughput is the answer tokens generated per second: all tokens of the 48 answers divided by the time the last one is done. Prompt tokens are not counted.','吞吐是每秒生成的回答 token：48 个回答的 token 总数，除以最后一个请求写完的时刻。prompt 的 token 不算在内。')},
 {k:'util',label:()=>tx('Slots busy','slot 忙碌比例'),v:r=>[pc(r.util),''],raw:r=>r.util*100,pts:true,dp:0,
  sub:r=>tx(`Slots sat idle while requests were waiting for ${pc(r.idleWait)} of all slot time.`,`slot 空着、同时还有请求在排队的时间，占全部 slot 时间的 ${pc(r.idleWait)}。`),
  tip:r=>tx(`Slots busy: of all the time of the ${slotsTxt(r.s.B)} over the whole run, the share spent on a request (prefill or decode).`,`slot 忙碌比例：整次运行里，${slotsTxt(r.s.B)} 的全部时间中，花在处理请求（prefill 或 decode）上的比例。`)},
 {k:'lat',label:()=>tx('Mean latency','平均延迟'),v:r=>[r.mlat.toFixed(1),'s'],raw:r=>r.mlat,dp:1,
  sub:r=>tx(`90% of the requests are done within ${sec(r.p90)} of arriving.`,`90% 的请求在到达后 ${sec(r.p90)} 内写完。`),
  tip:()=>tx('Mean latency is the time from a request’s arrival to its last token, averaged over the 48 requests.','平均延迟是一个请求从到达到最后一个 token 的时间，对 48 个请求取平均。')},
 {k:'tpot',label:()=>tx('Time per output token','每个输出 token 耗时'),v:r=>[(r.tpot*1000).toFixed(1),'ms'],raw:r=>r.tpot*1000,dp:1,
  sub:r=>{const n=Math.round(1000/rnd(r.tpot*1000,1)); return tx(`That is ${n} tokens per second for each request.`,`相当于每个请求每秒收到 ${n} 个 token。`);},
  tip:()=>tx('Time per output token is how long a request waits, on average, between two of its answer tokens after the first. The pauses while other requests’ prompts are prefilled count too.','每个输出 token 耗时是第一个 token 之后，一个请求平均隔多久得到下一个 token。别的请求做 prefill 时它也得等，这段时间也算在内。')}];
function view(){
  const segP=(id,inline)=>A.ui.seg({id,key:'pol',label:tx('Batching policy','调度方式'),options:[['static',tx('Static','static')],['cont',tx('Continuous','continuous')]],value:S.pol,inline,
    note:inline?'':tx('Static batching starts the next group only when the whole batch is done; continuous batching gives a finished request’s slot to a waiting request at the next step.','static batching 等整批做完才开始下一批；continuous batching 在下一步就把写完的请求的 slot 交给排队的请求。')});
  return `<div class="cb-lab m2" data-mode="${mode}">`+A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:'Continuous batching',title:tx('Why does continuous batching raise throughput?','continuous batching 为什么能提高吞吐？'),
      minutes:15,level:tx('Core','核心'),fid:'Simulated',fidPop:fidPop(),primer:tx('Primer: prefill and decode','前置知识：prefill 和 decode'),primerPop:priPop(),descs:DESCS(),mode})+
    `<div class="wrap bench cb-bench"><aside class="rail" id="cb-rail" aria-label="${esc(A.t('railAria'))}"></aside>`+
    `<div class="panel stage" id="cb-stage"><div class="stage-head"><div><div class="meter-title">${tx('OPT-13B on one A100 40GB','一张 A100 40GB 跑 OPT-13B')}</div><div class="meter-sub" id="cb-sub"></div><div class="cb-sub2" id="cb-sub2"></div></div>`+
      `<div class="stage-actions"><span class="status cb-st" id="cb-status"></span></div></div>`+
      `<div class="legend-row cb-legend">${LEG().map(([c,t,tip,id])=>`<span tabindex="0" data-tipx="${esc(tip)}"${id?` id="${id}"${id==='cb-lg-pf'?'':' hidden'}`:''}><i class="cb-sw ${c}"></i><span>${t}</span></span>`).join('')}</div>`+
      `<div class="stage-graph graph" id="cb-graph"><div class="cb-play"><button type="button" class="btn ghost sm cb-pbtn" data-act="play" data-tipx="${esc(PLAYTIP())}">${playLabel()}</button>${speedCtl()}`+
        `<button type="button" class="btn ghost sm" data-act="next" data-tipx="${esc(tx('The chart jumps to the moment the next request finishes.','图上的时刻跳到下一个请求写完的时候。'))}">${tx('Next finish ▸','下一个完成 ▸')}</button>`+
        `<input type="range" id="cb-scrub" min="0" max="1000" step="1" value="1000" aria-label="${esc(tx('Time in the run','运行中的时刻'))}"><span class="cb-clock" id="cb-clock" aria-live="off"></span></div>`+
        `<div class="bar-label cb-glabel">${tx('Each row is a slot and each bar a request, from its start to its last token.','每一行是一个 slot，每一条是一个请求：从它开始，到它的最后一个 token。')}</div>`+
        `<div class="cb-gantt" id="cb-gantt" role="group"></div><button type="button" class="linkbtn cb-done" id="cb-done" data-act="to-card" hidden></button><p class="stage-note" id="cb-note" hidden></p></div>`+
      `<div class="cb-mhead"><span class="eyebrow">${tx(`Whole run (all ${NREQ} requests)`,`整次运行（${NREQ} 个请求）`)}</span></div>`+
      `<div class="metrics cb-metrics" id="cb-metrics">${MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0" data-tipx="${esc(m.tip(cur()))}"><div class="k">${m.label()}</div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('')}</div>`+
      `<details class="math" id="cb-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="cb-mathbody"></div><div id="cb-mathchart"></div></details></div>`+
    `<aside class="panel controls" id="cb-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
      `<p class="ctl-lock" id="cb-lock" hidden></p>`+
      `<div class="ctl cb-lockable" id="cb-ctl-setup"><span class="lbl">${tx('Setup (fixed in this lab)','配置（这个 Lab 里不变）')}</span><div class="cb-fixed">${tx('OPT-13B on one A100 40GB, 512-token prompts','一张 A100 40GB 跑 OPT-13B，每个 prompt 512 个 token')}</div>`+
        `<div class="ctl-note">${tx(`OPT-13B has ${A.ui.big(PARAMS)} ${A.ui.def('parameters',A.terms.param())}, ${F.wgb} (${F.wgib.replace(/ GiB$/,' '+A.ui.def('GiB',A.terms.gib()))}) of FP16 weights, and ${F.kv} of KV cache per token (from its config.json). The A100 40GB reads memory at ${F.bw} and computes ${F.peak} in dense FP16 (NVIDIA’s datasheet).`,
                                    `OPT-13B 有 ${A.ui.big(PARAMS)} 个${A.ui.def('参数',A.terms.param())}，FP16 权重共 ${F.wgb}（${F.wgib.replace(/ GiB$/,' '+A.ui.def('GiB',A.terms.gib()))}），每个 token 的 KV cache 是 ${F.kv}（来自它的 config.json）。A100 40GB 的显存带宽是 ${F.bw}，FP16 稠密算力 ${F.peak}（NVIDIA 数据手册）。`)}</div></div>`+
      segP('cb-ctl-pol',false)+
      A.ui.range({id:'cb-ctl-B',key:'B',label:tx('Slots','slot 数'),min:1,max:B_MAX,value:S.B,out:slotsTxt(S.B),
        note:tx(`The slots are the most requests the GPU runs in one step (vLLM calls this limit max_num_seqs). Memory fits the KV cache of at most ${B_MAX} requests (${AVG_CTX} tokens, ${F.kvReq} each; ${J('mem','how that is computed')}).`,
                `slot 数是 GPU 每一步最多同时跑的请求数（vLLM 里叫 max_num_seqs）。显存最多放得下 ${B_MAX} 个请求的 KV cache（每个按 ${AVG_CTX} 个 token 算，占 ${F.kvReq}；${J('mem','怎么算的')}）。`)})+
      `<div class="cb-lockable">`+A.ui.seg({id:'cb-ctl-cap',key:'cap',label:tx('Answer cap (tokens)','回答上限（token）'),options:CAPS.map(c=>[String(c),A.fmt.int(c)]),value:String(S.cap),
        note:tx('Answer lengths are drawn from an exponential distribution with a mean of 128 tokens, redrawing any above the cap. A lower cap makes the answers shorter and closer in length.',
                '回答长度从均值 128 个 token 的指数分布里随机抽取，超过上限的重新抽。上限越低，回答越短，长短也越接近。')})+`</div>`+
      `<div class="cb-lockable">`+A.ui.seg({id:'cb-ctl-rate',key:'rate',label:tx('Arrivals','到达方式'),options:[['0',tx('At once','同时到达')],['2',tx('2 per s','每秒 2 个')],['4',tx('4 per s','每秒 4 个')]],value:String(S.rate),
        note:tx('With “At once”, all 48 requests are waiting at 0 s, like an offline batch job. With 2 or 4 per second, requests arrive at random at that average rate, like live traffic.',
                '选“同时到达”时，48 个请求 0 s 时全部在排队，像离线批量任务；选每秒 2 个或 4 个时，请求按这个平均速率随机到达，像线上流量。')})+`</div>`+
      `<div class="cb-lockable">`+A.ui.seg({id:'cb-ctl-set',key:'set',label:tx('Request set','请求组'),options:SEEDS.map((sd,i)=>[String(i),String(i+1)]),value:String(S.set),
        note:tx('The three sets are three random draws of the same workload, and the guided parts and the challenge use set 1. Each set has different answer lengths, so the results differ a little.',
                '三个请求组是同一种负载的三次随机抽样，分步引导和挑战题用第 1 组。每组抽到的回答长度不同，结果也会差一些。')})+`</div>`+
      `<details class="adv cb-lockable" id="cb-adv"><summary>${tx('Step-time and memory assumptions','每步耗时和显存的假设')}</summary>`+
        A.ui.range({id:'cb-ctl-eff',key:'eff',label:tx(`Prefill speed (share of the ${F.peak} peak)`,`prefill 速度（占 ${F.peak} 峰值的比例）`),min:30,max:70,step:5,value:S.eff,out:S.eff+'%',
          note:tx(`Prefill speed is the share of the A100’s ${F.peak} peak that prefill’s matrix multiplies reach. It is an estimate, not a measurement: real kernels stay below the peak by an amount that depends on the kernels, the matrix shapes and the tokens per step. 50% has no measured source and is a middle guess; a lower value makes each prefill, and the pause it causes for the other requests, longer.`,
                  `prefill 速度指 prefill 的矩阵乘法能用到 A100 峰值算力（${F.peak}）的多少。它是估计值，不是实测值：真实 kernel 达不到峰值，差多少取决于 kernel、矩阵形状和一步处理的 token 数。50% 没有实测出处，是一个折中的猜测；这个值越低，每次 prefill 越久，其他请求等得也越久。`)})+
        `<div class="ctl" id="cb-ctl-bw"><span class="lbl">${tx('Memory bandwidth used','实际用到的显存带宽')}<output>${F.bw}</output></span><div class="ctl-note">${tx(`This is an optimistic assumption: every step reads the weights and the KV cache at the datasheet’s full ${F.bw}. Real kernels reach somewhat less, so real steps take a little longer; both policies slow down by about the same factor, so the comparison hardly changes.`,`这是偏乐观的假设：每一步按数据手册的满速 ${F.bw} 读权重和 KV cache。真实 kernel 会低一些，每一步会稍慢；两种调度方式慢下来的比例差不多，所以对比几乎不变。`)}</div></div>`+
        `<div class="ctl" id="cb-ctl-mem"><span class="lbl">${tx('Memory left for the KV cache','留给 KV cache 的显存')}<output>${gib(KV_BUDGET)}</output></span><div class="ctl-note">${tx(
          `<p>${F.mem} × ${F.util} − ${F.wgib} (${F.wgb}) of weights − ${F.oh} of runtime overhead = ${gib(KV_BUDGET)}.</p>`+
          `<p>${F.util} is vLLM’s default gpu_memory_utilization: the engine uses at most ${F.utilPc} of the GPU’s memory and keeps the rest as headroom. The A100’s 40 GB is read as ${F.mem}, as in the lab ${A.shell.labLink('kv-cache')}.</p>`+
          `<p>The ${F.oh} of runtime overhead is an estimate for what a step needs besides the weights and the KV cache: temporary activations, captured CUDA graphs and buffers. vLLM measures it at startup; it grows with the most tokens one step may process (max_num_batched_tokens) and is usually 1 to 2 GiB, small because inference frees each layer’s activations once the next layer has used them.</p>`,
          `<p>${F.mem} × ${F.util} − 权重 ${F.wgib}（${F.wgb}）− 运行时开销 ${F.oh} = ${gib(KV_BUDGET)}。</p>`+
          `<p>${F.util} 是 vLLM 默认的 gpu_memory_utilization：引擎最多用 GPU 显存的 ${F.utilPc}，其余留作余量。A100 的 40 GB 按 ${F.mem} 计，和 ${A.shell.labLink('kv-cache')}那个 Lab 一样。</p>`+
          `<p>运行时开销 ${F.oh} 是估计值，指一步除了权重和 KV cache 之外还要的显存：临时激活、捕获的 CUDA graph 和各种缓冲区。vLLM 启动时实测它；一步最多处理的 token 越多（max_num_batched_tokens），它越大，通常 1 到 2 GiB。它不大，是因为推理时每层的激活在下一层用完后就释放。</p>`)}</div></div></details>`+
      `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside></div>`+connectHTML()+`</div>`;
}

/* ---------- connect ---------- */
const CODE=`from collections import deque

EOS = 2  # OPT's end-of-sequence token id (config.json)

def serve(requests, max_slots, model_step):
    """Continuous batching: the batch is rebuilt at every step.

    requests: iterable of (request_id, prompt_tokens, max_new_tokens)
    model_step(batch) -> one new token per request in the batch
    (a new request's first step also prefills its prompt)."""
    waiting = deque(requests)
    running, finished = [], {}
    while waiting or running:
        # 1. Admit: free slots are refilled from the queue at once.
        while waiting and len(running) < max_slots:
            rid, prompt, max_new = waiting.popleft()
            running.append({"id": rid, "prompt": prompt,
                            "out": [], "max_new": max_new})
        # 2. One step: every running request gets exactly one token.
        for req, token in zip(running, model_step(running)):
            req["out"].append(token)
        # 3. Retire: finished requests leave now, not at batch end.
        still = []
        for req in running:
            if req["out"][-1] == EOS or len(req["out"]) >= req["max_new"]:
                finished[req["id"]] = req["out"]
            else:
                still.append(req)
        running = still
    return finished

# Static batching differs in one place: it admits a new group only
# when \`running\` is empty, so early finishers leave their slots idle.`;
function connectHTML(){
  const li=(u,path,en,zh)=>`<li>${a(u,path)}${tx(`: ${en}`,`：${zh}`)}</li>`;
  return `<section class="wrap connect" aria-labelledby="cb-conn-h"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="cb-conn-h">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The scheduler of continuous batching is one loop: fill the free slots from the queue, give every running request one token, then remove the finished ones.','continuous batching 的调度就是一个循环：先从队列里补满空出的 slot，再让每个在跑的请求生成一个 token，最后移走写完的。')}</p><pre class="code" id="cb-code">${esc(CODE)}</pre><button type="button" class="btn ghost sm" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="cb-prod">`+
     li(SRC.vsched,'vllm/v1/core/sched/scheduler.py','schedule() runs once per step. It first gives the running requests their share of this step’s token budget, then admits waiting ones while max_num_batched_tokens and max_num_seqs allow; update_from_output() frees the finished requests (vLLM v0.30.0).',
        'schedule() 每一步运行一次，先给在跑的请求分配这一步的 token 预算，再在 max_num_batched_tokens 和 max_num_seqs 允许的范围内接纳排队的请求；update_from_output() 释放写完的请求（vLLM v0.30.0）。')+
     li(SRC.vschedcfg,'vllm/config/scheduler.py','max_num_seqs, the most requests running at once (the slots here), and max_num_batched_tokens.',
        'max_num_seqs，同时跑的请求上限（就是这里的 slot 数），以及 max_num_batched_tokens。')+
     li(SRC.ssched,'python/sglang/srt/managers/scheduler.py','every iteration calls get_next_batch_to_run(), which admits new requests with get_new_batch_prefill() and otherwise drops the finished ones with update_running_batch() before the next decode step (SGLang v0.5.20).',
        '每一轮都调用 get_next_batch_to_run()，有新请求就用 get_new_batch_prefill() 接纳，否则用 update_running_batch() 去掉写完的请求，再做下一步 decode（SGLang v0.5.20）。')+
   `</ul><p class="small muted">${tx(`The lab ${A.shell.labLink('chunked-prefill')} picks up from here. It cuts a long prompt into chunks and computes one chunk per step, so the requests that are decoding keep getting tokens while the prompt is computed.`,`${A.shell.labLink('chunked-prefill')} 这个 Lab 接着讲：把长 prompt 切成几块，每一步只算一块。这样算 prompt 的同时，正在 decode 的请求也能继续拿到 token。`)}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
     `<li>${a(SRC.orca,'Orca: A Distributed Serving System for Transformer-Based Generative Models')}<span>${tx('Yu et al., OSDI 2022: rescheduling at every step, with 36.9 times the throughput of FasterTransformer at the same latency (GPT-3 175B).','Yu 等，OSDI 2022：每一步都重新调度，同等延迟下吞吐是 FasterTransformer 的 36.9 倍（GPT-3 175B）。')}</span></li>`+
     `<li>${a(SRC.vllm,'Efficient Memory Management for Large Language Model Serving with PagedAttention')}<span>${tx('Kwon et al., SOSP 2023: works on the memory side, so larger batches fit in the same GPU memory.','Kwon 等，SOSP 2023：从显存入手，让同样的显存放下更大的 batch。')}</span></li>`+
     `<li>${a(SRC.anyscale,'How continuous batching enables 23x throughput in LLM inference while reducing p50 latency')}<span>${tx('Anyscale, 2023: the benchmark this lab scales down (OPT-13B on one A100 40GB).','Anyscale，2023：这个 Lab 按它的 benchmark 缩小了规模（一张 A100 40GB 跑 OPT-13B）。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="cb-drill-q"></p><div class="cta-row"><button type="button" class="btn sm" data-act="drill-reveal">${A.t('drillReveal')}</button><button type="button" class="btn ghost sm" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="cb-drill-a" hidden></div></article>`+
   `</div></section>`;
}
const DRILL_B=[2,4,8,16];
const harm=n=>{let h=0; for(let i=1;i<=n;i++) h+=1/i; return h;};
// static batching share of useful slot-steps for this lab's 48 requests (set 1, cap 1,536): tokens ÷ (B × Σ longest answer per batch)
function drillEmp(B){const lens=workload(0,1536,0).lens; let steps=0; for(let k=0;k<lens.length;k+=B) steps+=Math.max(...lens.slice(k,k+B)); const tok=lens.reduce((x,y)=>x+y,0); return {steps:steps*B,tok,share:tok/(steps*B)};}
function drillQ(){const B=DRILL_B[drillI];
  return tx(`Static batching runs batches of <b>${B}</b> requests, and each batch lasts until its longest answer is done. Answer lengths are exponential with a mean of 128 tokens (no cap), and prefill is ignored. On average, what share of the slots’ steps produce a token?`,
            `static batching 每批 <b>${B}</b> 个请求，每批要等最长的回答写完才结束。回答长度服从均值 128 个 token 的指数分布（不设上限），不考虑 prefill。平均来说，slot 在这些步里，有多大比例的步真正产出了 token？`);}
const hSum=B=>B<=4?Array.from({length:B},(_,i)=>i?`\\frac{1}{${i+1}}`:'1').join(' + '):`1 + \\frac{1}{2} + \\ldots + \\frac{1}{${B}}`;
const p1=v=>(v*100).toFixed(1)+'%';                               // one decimal everywhere in the drill, so 39.0% and 33.3% read alike
function drillA(){const B=DRILL_B[drillI], H=harm(B), e=drillEmp(B), n=e.steps/B;
  // the stage's “Slots busy” for the same run counts time, and each batch's long prefill step counts all its slots as busy
  const st=simulate({...DEF,B}), nb=Math.ceil(NREQ/B), pfB=sec(stepTime(PROMPT*B,0,50));
  const fxH=`<div class="cb-fxl">${MX(tx('\\text{longest answer} = \\mu \\times H_B,\\quad H_B = 1 + \\frac{1}{2} + \\ldots + \\frac{1}{B}','\\text{最长的回答} = \\mu \\times H_B,\\quad H_B = 1 + \\frac{1}{2} + \\ldots + \\frac{1}{B}'))}</div>`,
        fxS=`<div class="cb-fxl">${MX(tx('\\text{share of the slots’ steps that produce a token} = \\frac{\\mu}{\\mu \\times H_B} = \\frac{1}{H_B}','\\text{真正产出 token 的步所占比例} = \\frac{\\mu}{\\mu \\times H_B} = \\frac{1}{H_B}'))}</div>`;
  return tx(`<p><b>${p1(1/H)} of the slots’ steps produce a token.</b></p><p>Each answer takes μ = 128 steps on average, but its batch lasts as long as the longest of ${B} answers. Of ${B} exponential lengths with mean μ, the first finishes after ${MX(`\\frac{\\mu}{${B}}`)} steps on average, the next after another ${MX(`\\frac{\\mu}{${B-1}}`)}, and so on. So on average the longest takes</p>${fxH}<p>Here ${MX(`H_{${B}} = ${hSum(B)} = ${H.toFixed(3)}`)}, so the longest answer takes ${Math.round(128*H)} steps on average. The share does not depend on μ:</p>${fxS}<p>${MX(`\\frac{1}{H_{${B}}}`)} = 1 ÷ ${H.toFixed(3)} = <b>${p1(1/H)}</b>.</p><p>Check with this lab’s 48 requests (set 1, answers capped at 1,536): static batching with ${B} slots runs ${A.fmt.int(n)} steps, in which the ${B} slots could make ${A.fmt.int(e.steps)} tokens; they make ${A.fmt.int(e.tok)}, which is ${p1(e.share)}. The same run’s Slots busy is ${pc(st.util)} because it counts time: in the ${nb} long prefill steps (about ${pfB} each) all ${B} slots are busy.</p>`,
            `<p><b>slot 的步里，平均只有 ${p1(1/H)} 真正产出了 token。</b></p><p>每个回答平均 μ = 128 步，但一批要等 ${B} 个回答里最长的那个写完。${B} 个长度都服从均值 μ 的指数分布时，最先写完的平均在 ${MX(`\\frac{\\mu}{${B}}`)} 步后，下一个再过 ${MX(`\\frac{\\mu}{${B-1}}`)} 步，依此类推。所以最长的回答平均要</p>${fxH}<p>这里 ${MX(`H_{${B}} = ${hSum(B)} = ${H.toFixed(3)}`)}，所以最长的回答平均要 ${Math.round(128*H)} 步。这个比例和 μ 无关：</p>${fxS}<p>${MX(`\\frac{1}{H_{${B}}}`)} = 1 ÷ ${H.toFixed(3)} = <b>${p1(1/H)}</b>。</p><p>用本 Lab 的 48 个请求验算（第 1 组，回答上限 1,536）：${B} 个 slot 的 static batching 一共跑 ${A.fmt.int(n)} 步，这些步里 ${B} 个 slot 本可以产出 ${A.fmt.int(e.steps)} 个 token，实际产出 ${A.fmt.int(e.tok)} 个，占 ${p1(e.share)}。同一次运行的 slot 忙碌比例是 ${pc(st.util)}：它按时间算，而 ${nb} 个较长的 prefill 步（每步约 ${pfB}）里 ${B} 个 slot 都在忙。</p>`);}
function drawDrill(){if(!root) return; const q=$('#cb-drill-q',root), an=$('#cb-drill-a',root); if(!q) return; q.innerHTML=drillQ(); an.hidden=!drillOpen; if(drillOpen) an.innerHTML=drillA();}

/* ---------- the capability layer (Codex review 1 #5, Batch 2) ----------
   Under the title: what the learner can do after the lab. In the challenge, after its own task: explain the two core ideas,
   predict a hand-sized case, then go further. The new case is counted in steps by tzSim (the home tile's scheduler, with the rules
   of simulate(): each running request makes one token per step, static batching starts the next group when the longest answer of
   the batch is done, continuous batching gives a free slot to a waiting request at the next step). Nothing here names the slots,
   the throughput or the time per output token that the challenge asks about. */
const OUTCOME=()=>tx('explain why continuous batching fills the slots static batching leaves idle, and what more slots cost each token.',
                     '讲清 continuous batching 为什么能用上 static batching 空着的 slot，以及 slot 多了，每个 token 为什么变慢。');
const NC={B:4,lens:[15,3,3,3],more:6};                              // the new case: 4 slots, a batch of 4 answers, then 6 more short requests waiting
const EXITS=()=>{
  // the explanations in words (R6.20); the step times behind them are in Show the math and on part 5's chart
  const joinX=joinStep(DEF.B,50)/plainStep(DEF.B), farTok=bStar(50)>=2*B_MAX;
  const {B,lens,more}=NC, long=Math.max(...lens), short=Math.min(...lens), nS=lens.filter(x=>x<long).length, nR=lens.length;
  const s1=tzSim('static',lens,B), all=lens.concat(Array(more).fill(short)), c2=tzSim('cont',all,B), s2=tzSim('static',all,B);
  const shortDone=Math.max(...c2.bars.filter(x=>x.len<long).map(x=>x.b));                      // when continuous batching finishes the last short answer
  const sl=c2.bars.find(x=>x.len<long).slot, rounds=c2.bars.filter(x=>x.slot===sl).sort((x,y)=>x.a-y.a);   // one of the short answers' slots, round by round
  const groups=[]; for(let k=0;k<all.length;k+=B) groups.push(Math.max(...all.slice(k,k+B)));   // static batching: each group lasts as long as its longest answer
  // a new case's options in natural order, the right one listed first (it wins if a wrong value coincides); each wrong one is a
  // misconception worked out from the same case, with a note on what it misses
  const opts=(list,fmt)=>{const seen=new Set(), l=list.filter(x=>!seen.has(x.v)&&seen.add(x.v)).sort((x,y)=>x.v-y.v);
    return {o:l.map(x=>fmt(x.v)),a:l.findIndex(x=>x.ok),n:l.map(x=>x.ok?null:x.n)};};
  const lsEn=lens.slice(0,-1).join(', ')+' and '+lens[lens.length-1], lsZh=lens.join('、'), sum=lens.join(' + ');
  const rEn=rounds.map(x=>`${x.a+1} to ${x.b}`), rEnTxt=rEn.length>1?rEn.slice(0,-1).join(', ')+' and '+rEn[rEn.length-1]:rEn[0], rZh=rounds.map(x=>`第 ${x.a+1} 到 ${x.b} 步`).join('、');
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Static batching leaves slots idle even while requests are waiting. Why?','明明还有请求在排队，static batching 也会让 slot 空着。为什么？'),
   o:[tx('It starts the next group only when the longest answer in the batch is done, so a slot whose answer finished early stays empty until then.','它要等这一批里最长的回答写完，才开始下一批，所以回答先写完的那个 slot 会一直空着，直到这一批结束。'),
     tx('A new request’s prompt cannot run in the same step as the other requests’ next tokens, so it has to wait until the whole batch is done.','新请求的 prompt 不能和其他请求的下一个 token 放在同一步里算，所以它只能等这一批全部做完。'),
     tx('A finished request keeps its KV cache in memory until its batch ends, so until then there is no room to start a new request in its slot.','写完的请求要把 KV cache 在显存里留到这一批结束，所以在那之前，它的 slot 没有空间开始新请求。')],a:0,
   n:[null,tx('It can: continuous batching prefills a joining request’s prompt in the same step as the others’ next tokens. Static batching waits because it schedules whole batches.','可以放在同一步：continuous batching 就是在同一步里给新请求的 prompt 做 prefill，同时算其他请求的下一个 token。static batching 要等，是因为它按整批调度。'),
     tx('In this lab memory does not stop it: the KV cache in use fits in every setting. The slot stays empty because static batching starts requests only as a group.','在这个 Lab 里，拦住它的不是显存：所有设置下，用到的 KV cache 都放得下。slot 空着，是因为 static batching 只按整批开始请求。')],
   why:tx(`Static batching schedules whole batches: a batch keeps all its slots until its longest answer is done. So in the initial setup (static batching, ${DEF.B} slots), a slot whose answer finished early sat idle until the longest answer of its batch was done, while requests waited. Continuous batching gives such a slot to a waiting request at the next step.`,
          `static batching 按整批调度：一批要等最长的回答写完，才让出全部 slot。所以初始设置（static batching，${DEF.B} 个 slot）下，回答先写完的 slot 要一直空到这一批最长的回答写完，而这时还有请求在排队。continuous batching 在下一步就把这样的 slot 交给排队的请求。`)},
  {q:tx('With continuous batching and requests waiting, more slots raise the throughput but also the time per output token. Why?','用 continuous batching、又有请求在排队时，slot 越多，吞吐越高，每个输出 token 的耗时也越长。为什么？'),
   o:[tx('Every step reads all the weights once, so extra slots add tokens almost for free. But each step also reads more KV cache, and requests join (prefill) more often, so steps get longer.','每一步都只读一遍全部权重，所以多加 slot，多产出的 token 几乎不多花时间。但每一步也要多读 KV cache，新请求加入（prefill）也更频繁，所以每一步变长了。'),
     tx('The GPU had compute to spare, so extra slots add more tokens to each step. But the requests share that compute, so each one gets a smaller part of it, and its tokens come more slowly.','GPU 原本有富余的算力，所以多加 slot，每一步能多产出 token。但这些请求要分这份算力，每个请求每一步分到的算力更少，它的 token 就来得更慢。'),
     tx('Memory had room for more requests, so extra slots add more tokens to each step. But their KV cache fills the memory, so the GPU pauses some requests, and their tokens come more slowly.','显存原本还放得下更多请求，所以多加 slot，每一步能多产出 token。但它们的 KV cache 会占满显存，GPU 只好让一些请求先停下，它们的 token 就来得更慢。')],a:0,
   n:[null,tx(`A decode step here is limited by reading the weights, not by compute: computing would outlast that read only if a step handled ${farTok?'far ':''}more tokens than any decode step here. An extra request adds only its small KV cache read.`,
              `这里 decode 的一步受限于读权重，不是算力：一步要处理的 token 比这里任何一个 decode 步都多${farTok?'得多':''}，计算才会比读权重更久。多一个请求，只多读它那一小份 KV cache。`),
     tx(`In this lab the running requests’ KV cache always fits (memory caps the slots at ${B_MAX}), so no request is ever paused. Each token still gets slower, because every step does.`,
        `在这个 Lab 里，在跑请求的 KV cache 总能放下（显存把 slot 数限制在最多 ${B_MAX} 个），没有请求会被停下。每个 token 还是变慢了，因为每一步都变长了。`)],
   why:tx(`A step reads OPT-13B’s weights once, however many requests it holds, and each request adds only a small read of its own KV cache. So each extra slot adds a token to every step almost for free. But those small reads add up, and with more slots, requests finish and others join more often: each joining prompt’s prefill makes its step ${joinX>=2.5?'several times':'much'} longer, and every running request waits through it.`,
          `一步只读一遍 OPT-13B 的权重，不管里面有几个请求；每个请求只多读一小份自己的 KV cache。所以多加 slot，每一步多产出的 token 几乎不多花时间。但这些零碎的读取会累加；而且 slot 越多，请求写完、别的请求加入也越频繁：每个新 prompt 的 prefill 都让那一步长${joinX>=2.5?'好几倍':'很多'}，在跑的请求都得等它做完。`)}],
 transfer:[
  {q:tx(`Static batching fills ${B} slots with a batch whose answers need ${lsEn} tokens, while more requests wait. Each running request makes one token per step. Until the batch is done, what share of the ${B} slots’ steps produce a token?`,
        `static batching 用一批 ${nR} 个请求占满 ${B} 个 slot，它们的回答分别要 ${lsZh} 个 token，后面还有请求在排队。每个在跑的请求每步产出 1 个 token。这一批做完之前，${B} 个 slot 在所有步里，真正产出 token 的比例是多少？`),
   ...opts([{v:s1.busy,ok:true},
     {v:short/s1.steps,n:tx(`That is one short answer’s slot alone, busy ${short} of the ${s1.steps} steps. The long answer’s slot is busy all ${s1.steps}.`,`这只是一个短回答的 slot：${s1.steps} 步里忙了 ${short} 步。长回答的 slot 这 ${s1.steps} 步都在忙。`)},
     {v:long/(B*s1.steps),n:tx(`That counts only the long answer’s slot. The ${nS} short answers make tokens too, in steps 1 to ${short}.`,`这只算了长回答的 slot。${nS} 个短回答在第 1 到 ${short} 步也在产出 token。`)},
     {v:1,n:tx(`A slot makes a token only while its request runs: after step ${short}, ${nS} of the ${B} slots make none.`,`slot 只在它的请求还在跑时才产出 token：第 ${short} 步之后，${B} 个 slot 里有 ${nS} 个不再产出。`)}],pc),
   // a worked computation, as short steps, one per sentence (R6.20)
   why:tx(`The batch lasts as long as its longest answer: ${s1.steps} steps. In them the ${B} slots could make ${B} × ${s1.steps} = ${B*s1.steps} tokens. The answers make ${sum} = ${s1.tok}. So ${s1.tok} ÷ ${B*s1.steps} = ${pc(s1.busy)}. After step ${short}, ${nS} of the ${B} slots sit idle for ${s1.steps-short} steps while requests wait.`,
          `这一批要持续到最长的回答写完，共 ${s1.steps} 步。这 ${s1.steps} 步里，${B} 个 slot 最多产出 ${B} × ${s1.steps} = ${B*s1.steps} 个 token。这些回答只有 ${sum} = ${s1.tok} 个。所以是 ${s1.tok} ÷ ${B*s1.steps} = ${pc(s1.busy)}。第 ${short} 步之后，${B} 个 slot 里有 ${nS} 个空等了 ${s1.steps-short} 步，后面还有请求在排队。`)},
  {q:tx(`The same ${nR} requests (${lsEn} tokens) run in ${B} slots, and ${more} more that need ${short} tokens each wait behind them. With continuous batching, after how many steps is every request done?`,
        `还是这 ${nR} 个请求（${lsZh} 个 token）跑在 ${B} 个 slot 上，后面还有 ${more} 个请求在排队，每个要 ${short} 个 token。用 continuous batching，要多少步才能全部做完？`),
   ...opts([{v:c2.steps,ok:true},
     {v:shortDone,n:tx(`By step ${shortDone} the short requests are done, but the ${long}-token request runs until step ${long}.`,`到第 ${shortDone} 步，短的请求都做完了，但 ${long} 个 token 的请求要跑到第 ${long} 步。`)},
     {v:s2.steps,n:tx('That is static batching, where the waiting requests start only when the whole batch is done. Continuous batching starts each one at the step after a slot frees.','这是 static batching 的步数：排队的请求要等整批做完才开始。continuous batching 在 slot 空出来的下一步就让它开始。')}],v=>tx(`${v} steps`,`${v} 步`)),
   why:tx(`${c2.steps} steps. Continuous batching gives a free slot to a waiting request at the next step. The ${long}-token request holds its slot for steps 1 to ${long}. The other ${nS} slots run ${short}-token requests in steps ${rEnTxt}: ${nS} × ${rounds.length} = ${nS*rounds.length} requests, the first ${nS} and the ${more} waiting. Static batching starts the next group only when the whole batch is done: ${groups.join(' + ')} = ${s2.steps} steps.`,
          `${c2.steps} 步。continuous batching 在 slot 空出来的下一步，就把它交给排队的请求。${long} 个 token 的请求占着一个 slot，从第 1 步到第 ${long} 步。另外 ${nS} 个 slot 在${rZh}各跑一轮 ${short} 个 token 的请求，一共 ${nS} × ${rounds.length} = ${nS*rounds.length} 个，就是最初的 ${nS} 个加上排队的 ${more} 个。static batching 要等整批做完才开始下一批，要 ${groups.join(' + ')} = ${s2.steps} 步。`)}],
 practice:[
  {id:'inference-engine',d:tx('Levels 2 and 5: each step, decode every running request, then admit waiting ones in order while slots and a token budget allow; when KV cache memory runs out, preempt the request admitted last.',
                              '第 2、5 关：每一步先给每个在跑的请求 decode 一个 token，再按顺序接纳排队的请求，直到 slot 或 token 预算用完；KV cache 的显存不够时，抢占最后接纳的请求。')},
  {id:'inference-serving',d:tx('Item 4, first deep dive: choose between static and continuous batching, decide how prefill shares the GPU with decode, and control admission and queueing under load.',
                               '第 4 项的第一个深入话题：在 static batching 和 continuous batching 之间做选择，决定 prefill 和 decode 怎样共用 GPU，并在过载时控制准入和排队。')}]};
};

/* ---------- rail: guided, scenarios, challenge ---------- */
// scenarios: real workloads; each sets only the answer cap and how requests arrive
const PRESETS=[
 {t:{en:'Offline batch job',zh:'离线批量生成'},d:{en:'All 48 requests are waiting at the start; answers average 128 tokens, capped at 1,536.',zh:'48 个请求一开始全部在排队；回答平均 128 个 token，上限 1,536 个。'},p:{cap:1536,rate:0}},
 {t:{en:'Classification and tagging',zh:'分类和打标签'},d:{en:'All 48 requests are waiting at the start; every answer is 32 tokens or shorter.',zh:'48 个请求一开始全部在排队；每个回答最多 32 个 token。'},p:{cap:32,rate:0}},
 {t:{en:'Chat service, light traffic',zh:'聊天服务，流量较小'},d:{en:'Requests arrive at random, 2 per second on average; answers average 128 tokens, capped at 1,536.',zh:'请求随机到达，平均每秒 2 个；回答平均 128 个 token，上限 1,536 个。'},p:{cap:1536,rate:2}},
 {t:{en:'Chat service, peak hours',zh:'聊天服务，高峰时段'},d:{en:'Requests arrive at random, 4 per second on average; answers average 128 tokens, capped at 1,536.',zh:'请求随机到达，平均每秒 4 个；回答平均 128 个 token，上限 1,536 个。'},p:{cap:1536,rate:4}}];
const presetNow=()=>PRESETS.findIndex(p=>p.p.cap===S.cap&&p.p.rate===S.rate);
const polSeg=()=>A.ui.seg({id:'cb-ic-pol',key:'pol',label:tx('Batching policy','调度方式'),options:[['static',tx('Static','static')],['cont',tx('Continuous','continuous')]],value:S.pol,inline:true});
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:true,skip:!!s.skip});}
  if(mode==='sandbox'){const pn=presetNow();
    return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
      `<p class="small">${tx(`Pick a scenario and see how many more tokens per second continuous batching makes than static batching. A scenario sets the answer lengths and how the requests arrive.`,`选一个场景，看看 continuous batching 每秒生成的 token 是 static batching 的几倍。每个场景定下回答长度和请求到达的方式。`)}</p>`+
      `<div class="obs" id="cb-sb-res" aria-live="polite"></div>`+
      `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===pn}"><b>${L(p.t)}</b><span>${L(p.d)}</span></button>`).join('')}</div>`+
      polSeg()+`<p class="small">${tx(`Change the slots or anything else under ${SET()}.`,`slot 数和其他设置，在${SET()}里改。`)}</p>`+A.shell.toChallenge()+`</div>`;}
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button>`;
  // the goal stays complete (R6.8); derived numbers that the question does not need (the answers' longest and mean) are on the
  // stage, and the goal's ratio to the initial setup is said in words (R6.20)
  const s0=simulate({...CH}), t0=Math.round(s0.thr), gq=GOAL_THR/t0;
  const vsT0=gq>=1.8&&gq<2.3?tx('about twice','约是初始设置的两倍'):tx(`${times(gq)} times`,`是初始设置的 ${times(gq)} 倍`);
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Reach ${GOAL_THR} tokens per second on one A100 40GB, with at most 25&nbsp;ms per output token on average`,`让一张 A100 40GB 每秒生成 ${GOAL_THR} 个 token，平均每个输出 token 不超过 25&nbsp;ms`)}</h3>`+
    `<p>${tx('These stay fixed: one A100 40GB runs OPT-13B, and 48 requests are waiting, with 512-token prompts and answers capped at 1,536 tokens. You can change only these two:','这些保持不变：一张 A100 40GB 跑 OPT-13B，48 个请求在排队，prompt 各 512 个 token，回答上限 1,536 个。你只能改下面两项：')}</p>`+
    `<p class="small muted">${tx(`Why these goals: ${GOAL_THR} tokens per second is ${vsT0} the initial setup’s; 25&nbsp;ms per output token means ${Math.round(1/GOAL_TPOT)} tokens per second for each request.`,`为什么是这两个目标：每秒 ${GOAL_THR} 个 token ${vsT0}；每个输出 token 25&nbsp;ms，相当于每个请求每秒收到 ${Math.round(1/GOAL_TPOT)} 个 token。`)}</p>`+
    polSeg()+
    A.ui.range({id:'cb-ic-B',key:'B',label:tx('Slots','slot 数'),min:1,max:B_MAX,value:S.B,out:slotsTxt(S.B),inline:true})+
    `<div class="goal" id="cb-g1"><span>${tx(`Throughput is now <b id="cb-g1v"></b> tokens per second; the goal is at least ${GOAL_THR}.`,`吞吐现在是每秒 <b id="cb-g1v"></b> 个 token，目标是至少 ${GOAL_THR} 个。`)}</span></div><div class="progress" id="cb-g1p" aria-hidden="true"><i></i></div>`+
    `<div class="goal cb-g2" id="cb-g2"><span>${tx('Each output token now takes <b id="cb-g2v"></b>; the goal is at most 25&nbsp;ms.','每个输出 token 现在要 <b id="cb-g2v"></b>，目标是不超过 25&nbsp;ms。')}</span></div>`+
    `<div class="cb-subwrap">${submit}</div><div id="cb-ch-res" aria-live="polite"></div></div>`;
}
function renderRail(){if(!root) return; const el=$('#cb-rail',root);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';   // the next lab in every mode (R3.11)
  el.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(el); const box=$('[data-obs]',el); if(box){box._h=null; box._step=step;}}

/* challenge: all policy × slot options, ranked by how little they change the initial setup */
function chOptions(x){const out=[]; for(const pol of ['static','cont']) for(let B=1;B<=B_MAX;B++){const r=simulate({...CH,...(x||{}),pol,B});
  out.push({pol,B,r,thr:r.thr,tpot:r.tpot,ok:r.thr>=GOAL_THR&&r.tpot<=GOAL_TPOT,n:(pol!==CH.pol?1:0)+(B!==CH.B?1:0),d:Math.abs(B-CH.B)});} return out;}
const chBest=()=>chOptions().filter(o=>o.ok).sort((x,y)=>x.n-y.n||x.d-y.d||x.tpot-y.tpot||x.B-y.B)[0];
// how many options would still meet both goals if the step-time estimates were less optimistic
const chOkIf=x=>chOptions(x).filter(o=>o.ok).length;
// the challenge badge: which goal is still missing, in words
const chBadge=r=>{const a=r.thr>=GOAL_THR, b=r.tpot<=GOAL_TPOT;
  return a&&b?tx('Both goals met','两个目标都达到了'):(!a&&!b?tx('Neither goal met yet','两个目标都还没达到')
    :(!a?tx(`Under ${GOAL_THR} tokens per second`,`吞吐还不到每秒 ${GOAL_THR} 个 token`):tx('Over 25 ms per output token','每个输出 token 超过 25 ms')));};
const optName=o=>tx(`${o.pol==='cont'?'Continuous':'Static'} batching, ${slotsTxt(o.B)}`,`${o.pol==='cont'?'continuous':'static'} batching，${slotsTxt(o.B)}`);
// the two reasons more slots slow each token (CB-A r7 #3), in words (R6.20); the step times at these slots are in Show the math
const whySlower=()=>tx('With more slots, every step reads more KV cache and others join (prefill) more often, so each output token gets slower.',
                       'slot 越多，每一步要读的 KV cache 越多，别的请求加入（prefill）也越频繁，所以每个输出 token 越来越慢。');
// "10 or 11" / "10 to 12" / "10"
const listB=a=>a.length===1?String(a[0]):(a.length===2?tx(`${a[0]} or ${a[1]}`,`${a[0]} 或 ${a[1]}`):tx(`${a[0]} to ${a[a.length-1]}`,`${a[0]} 到 ${a[a.length-1]}`));
function chResultHTML(){
  if(!chSubmitted) return '';
  const opts=chOptions(), me=opts.find(o=>o.pol===S.pol&&o.B===S.B), best=chBest();
  let h;
  // the goal line keeps its two values (they are the point); the rest in words (R6.20)
  if(me.ok&&best&&me.pol===best.pol&&me.B===best.B) h=tx(`<p class="okline" style="margin-top:12px"><b>Both goals met: ${tps(me.thr)} tokens per second at ${ms(me.tpot)} per output token.</b> And you changed as little as possible: in this simulation, with fewer slots continuous batching stays under the throughput goal.</p>`,
                                                            `<p class="okline" style="margin-top:12px"><b>两个目标都达到了：每秒 ${tps(me.thr)} 个 token，每个输出 token ${ms(me.tpot)}。</b>而且你的改动已经是最少的：在这个模拟里，slot 再少，continuous batching 的吞吐就达不到目标。</p>`);
  else if(me.ok){const qk=best.tpot<me.tpot;
    h=tx(`<p class="okline" style="margin-top:12px"><b>Both goals met: ${tps(me.thr)} tokens per second at ${ms(me.tpot)} per output token.</b> Fewer slots would do: continuous batching with ${best.B} slots also meets both goals${qk?', and each output token comes a little faster':''}.</p>`,
         `<p class="okline" style="margin-top:12px"><b>两个目标都达到了：每秒 ${tps(me.thr)} 个 token，每个输出 token ${ms(me.tpot)}。</b>其实 slot 少一些也够：${best.B} 个 slot 的 continuous batching 也能达到两个目标${qk?'，每个输出 token 还快一点':''}。</p>`);}
  else {
    // the progress rows above already give both values: the result line says only which goal is missing (CB-A r7 #15)
    const lowT=me.thr<GOAL_THR, slowT=me.tpot>GOAL_TPOT;
    const miss=lowT&&slowT?tx('neither goal is met.','两个目标都还没达到。'):(lowT?tx(`throughput is under ${GOAL_THR} tokens per second.`,`吞吐还不到每秒 ${GOAL_THR} 个 token。`):tx('each output token takes over 25&nbsp;ms.','每个输出 token 超过了 25&nbsp;ms。'));
    let hint;
    if(me.pol==='static') hint=tx('Hint: under static batching, a slot whose answer is done waits for the longest answer in its batch. Which policy gives that slot to a waiting request right away?','提示：static batching 下，回答写完的 slot 要等这一批里最长的回答。哪种调度方式能马上把这个 slot 交给排队的请求？');
    else if(lowT) hint=tx('Hint: continuous batching is the right policy, but the throughput is short of the goal. A step is mostly reading the weights, and each extra request adds only a little. What happens with a few more slots?',
                          '提示：continuous batching 选对了，但吞吐还不到目标。一步的时间主要花在读权重上，每多一个请求只多一点点。多几个 slot 会怎样？');
    else hint=tx(`Hint: the throughput is enough. ${whySlower()} Try fewer slots.`,`提示：吞吐够了。${whySlower()}试试少几个 slot。`);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not there yet: ${miss}</b></p>`,`<p class="badline" style="margin-top:12px"><b>还没达到目标：${miss}</b></p>`)+`<p class="small">${hint}</p>`+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  /* brief (the owner's review of batch 8): every policy × slot option goes to the chart, which keeps only the options that meet both
     goals, the pick and the initial setup, and says how many it left out; “over 25 ms” marks a shown row that is gray for its speed */
  const isMe=o=>o.pol===me.pol&&o.B===me.B, isStart=o=>o.pol===CH.pol&&o.B===CH.B;
  const shown=opts.filter(o=>o.ok||isMe(o)||isStart(o));
  const sMax=opts.find(o=>o.pol==='static'&&o.B===B_MAX);
  h+=A.ui.cmpChart({title:tx(`Tokens per second for 48 requests on one A100 40GB serving OPT-13B (goal: at least ${GOAL_THR}, with at most 25 ms per output token)`,`一张 A100 40GB 跑 OPT-13B、48 个请求时，每秒生成的 token（目标是至少 ${GOAL_THR} 个，而且每个输出 token 不超过 25 ms）`),
    aria:tx('Tokens per second and time per output token: ','每秒生成的 token 和每个输出 token 的耗时：')+shown.map(o=>`${optName(o)}${tx(': ','：')}${tps(o.thr)}${tx(', ','，')}${ms(o.tpot)}${o.ok?tx(', meets both goals','，两项都达标'):''}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(`The other ${n} settings miss at least one goal and are not shown.`,`其余 ${n} 种设置至少有一个目标没达到，图里没有列出。`),
    rows:opts.map(o=>{const mine=isMe(o), tags=[];
      if(isStart(o)) tags.push({text:tx('initial setup','初始设置')});
      if(o.tpot>GOAL_TPOT&&(mine||isStart(o))) tags.push({text:tx('over 25 ms','超过 25 ms')});
      if(mine) tags.push({text:tx('your pick','你的选择'),me:true});
      return {label:tx(`${optName(o)}: ${ms(o.tpot)} per output token`,`${optName(o)}：每个输出 token ${ms(o.tpot)}`),value:o.thr,text:tps(o.thr),ok:o.ok,me:mine,tags};}),
    goal:GOAL_THR,goalLabel:tx(`goal: ${GOAL_THR}`,`目标 ${GOAL_THR}`),max:Math.max(...shown.map(o=>o.thr))*1.15,
    okText:tx('meets both goals','两项都达标'),noText:tx('misses at least one','至少一项未达标'),goalText:tx(`goal: ${GOAL_THR} tokens per second`,`目标：每秒 ${GOAL_THR} 个 token`)});
  // which options pass, and why the others fail, from the grid above (CB-A r7 #15c)
  const okC=opts.filter(o=>o.ok&&o.pol==='cont').map(o=>o.B), okS=opts.some(o=>o.ok&&o.pol==='static');
  const lo=okC.length?Math.min(...okC):0, hi=okC.length?Math.max(...okC):0;
  const fewLow=okC.length&&opts.filter(o=>o.pol==='cont'&&o.B<lo).every(o=>o.thr<GOAL_THR), moreSlow=okC.length&&opts.filter(o=>o.pol==='cont'&&o.B>hi).every(o=>o.tpot>GOAL_TPOT);
  // which options pass, then the reason in words (R3.10, R6.20); the chart above holds only the options that pass, the pick and the
  // initial setup, so the options that fail are named in words (static batching's best throughput on hover)
  const first=!okC.length?tx('No option meets both goals here.','这里没有任何选项能同时达到两个目标。')
    :tx(`${okS?'Continuous':'Only continuous'} batching with ${listB(okC)} slots meets both goals`+(fewLow&&moreSlow?': with fewer slots the throughput stays under the goal, and with more each output token takes too long.':'.'),
        `${okS?'':'只有 '}continuous batching 用 ${listB(okC)} 个 slot 时两个目标都达到`+(fewLow&&moreSlow?'：slot 再少，吞吐达不到目标；再多，每个输出 token 又太慢。':'。'));
  const sShort=Math.round(sMax.thr)<GOAL_THR;
  const sTip=tx(`Static batching with ${B_MAX} slots: ${tps(sMax.thr)} tokens per second.`,`static batching 用 ${B_MAX} 个 slot 时，每秒 ${tps(sMax.thr)} 个 token。`);
  h+=tx(`<p class="small">${first}${sShort?` Static batching falls short even with ${A.ui.def('every slot that memory allows',sTip)}.`:''}</p>`+
        '<p class="small">Why: continuous batching never leaves a slot idle while requests wait, and each extra slot adds a token to every step for only a little more time. But with more slots, every step reads more KV cache and others join (prefill) more often, so each output token gets slower.</p>'+
        `<p class="small muted">This is one simulated set of 48 requests, and the step times are estimates. On a real server, measure with your own traffic before you set a limit such as vLLM’s max_num_seqs.</p>`,
        `<p class="small">${first}${sShort?`static batching 就算用满${A.ui.def('显存放得下的所有 slot',sTip)} 也达不到目标。`:''}</p>`+
        '<p class="small">原因：有请求在排队时，continuous batching 从不让 slot 空着；每多一个 slot，每一步多产出一个 token，只多花一点点时间。但 slot 越多，每一步要读的 KV cache 越多，别的请求加入（prefill）也越频繁，所以每个输出 token 越来越慢。</p>'+
        `<p class="small muted">这只是一组模拟的 48 个请求，每一步的耗时也是估计值。真实部署时，要先用自己的流量实测，再设 vLLM 的 max_num_seqs 这类上限。</p>`);
  return h;
}

/* ---------- stage drawing ---------- */
// The dashed comparison line. Hidden until the learner has run both policies (it would give away the prediction);
// in "Grow the batch" it marks the 8-slot continuous run that the part's text compares with.
function ghost(){
  if(mode==='challenge'&&!(S.pol==='cont'||chReveal||(cur().thr>=GOAL_THR&&cur().tpot<=GOAL_TPOT))) return null;   // it would answer the hint (R3.8)
  if(mode==='guided'){
    if(step<=2) return null;
    if(step===3&&!lvTried.has('cont')&&S.pol!=='cont') return null;
    if(step===5){if(S.pol==='cont'&&S.B===8) return null; return {r:simulate({...S,pol:'cont',B:8}),kind:'b8'};}
  }
  return {r:other(),kind:'pol'};
}
const ghostName=g=>g.kind==='b8'?tx('8 slots','8 个 slot'):polName(g.r.s.pol);
function xMax(){const g=ghost(); return Math.max(cur().mk,g?g.r.mk:0);}
const P100=(v,X)=>(Math.max(0,Math.min(1,v/X))*100).toFixed(3)+'%';
// tokens a request has generated by time t: one per step that ended after it started (the first of them ends its prefill)
const ub=(a,x)=>{let lo=0, hi=a.length; while(lo<hi){const m=(lo+hi)>>1; if(a[m]<=x+1e-9) lo=m+1; else hi=m;} return lo;};
const genAt=(r,q,t)=>Math.min(q.L,Math.max(0,ub(r.ends,t)-ub(r.ends,q.start)));
// a request's tooltip says only what is known at the moment the chart is drawn up to, plus the length its answer will reach (R3.8)
function reqTip(r,q,t){
  const n=q.i+1;
  if(t==null||q.fin<=t+1e-9) return tx(`Request ${n} has a 512-token prompt and a ${q.L}-token answer. It arrived at ${sec(q.arr)}, started at ${sec(q.start)}, generated its first token at ${sec(q.first)} and finished at ${sec(q.fin)}.`,
                                       `请求 ${n} 的 prompt 有 512 个 token，回答有 ${q.L} 个 token。它在 ${sec(q.arr)} 到达，${sec(q.start)} 开始，${sec(q.first)} 生成第一个 token，${sec(q.fin)} 写完。`);
  const head=tx(`Request ${n} has a 512-token prompt, and its answer will have ${q.L} tokens. `,`请求 ${n} 的 prompt 有 512 个 token，回答最终会有 ${q.L} 个 token。`);
  if(q.arr>t+1e-9) return head+tx(`It arrives at ${sec(q.arr)}.`,`它在 ${sec(q.arr)} 到达。`);
  if(q.start>t+1e-9) return head+tx(`It arrived at ${sec(q.arr)} and is still waiting for a slot.`,`它在 ${sec(q.arr)} 到达，还在等 slot。`);
  if(q.first>t+1e-9) return head+tx(`It arrived at ${sec(q.arr)}, started at ${sec(q.start)} and is still in its prefill.`,`它在 ${sec(q.arr)} 到达，${sec(q.start)} 开始，还在做 prefill。`);
  return head+tx(`It arrived at ${sec(q.arr)}, started at ${sec(q.start)} and generated its first token at ${sec(q.first)}; by ${sec(t)} it has generated ${genAt(r,q,t)}.`,
                 `它在 ${sec(q.arr)} 到达，${sec(q.start)} 开始，${sec(q.first)} 生成第一个 token；到 ${sec(t)} 已生成 ${genAt(r,q,t)} 个。`);
}
function refreshReqTip(el){const r=cur(), q=r.reqs[+el.dataset.i]; if(!q) return; const h=reqTip(r,q,partial()?playT():null); el.dataset.tipx=h; el.setAttribute('aria-label',h);}
function ticks(X,wpx){const cands=[0.5,1,2,5,10,20,30,60,120], maxN=wpx<420?4:6; let st=cands[cands.length-1]; for(const c of cands){if(X/c<=maxN){st=c; break;}}
  const out=[]; for(let v=0;v<=X+1e-9;v+=st) out.push(v); return out;}
function drawGantt(r){
  const g=$('#cb-gantt',root); if(!g) return;
  const gh=ghost(), B=r.s.B, X=xMax(), rh=B<=8?18:(B<=12?14:(B<=16?11:9));
  const lab=k=>(B<=12||k===0||k===B-1||(k+1)%5===0)?String(k+1):'';
  // waiting strip: step area of the queue length, 0..48
  let d='M0,100', w=0; for(const [tt,n] of r.qline){const x=(Math.min(tt,X)/X*1000).toFixed(2); d+=`L${x},${(100-w/NREQ*100).toFixed(2)}L${x},${(100-n/NREQ*100).toFixed(2)}`; w=n;}
  d+=`L1000,${(100-w/NREQ*100).toFixed(2)}L1000,100Z`;
  const rows=[]; for(let k=0;k<B;k++) rows.push([]);
  // idle slivers shorter than a pixel or so (a request that arrived mid-step waits for the step to end) are counted but not drawn
  r.hatch.forEach(([k,x,y])=>{if((y-x)/X>=0.004) rows[k].push(`<i class="cb-idle" style="left:${P100(x,X)};width:${P100(y-x,X)}"></i>`);});
  const tNow=partial()?playT():null;
  r.reqs.forEach(q=>{const dur=q.fin-q.start, pf=dur>0?(q.first-q.start)/dur:1, tip=esc(reqTip(r,q,tNow));
    rows[q.slot].push(`<b class="cb-req" style="left:${P100(q.start,X)};width:${P100(dur,X)}" tabindex="${q.i===0?0:-1}" role="img" aria-label="${tip}" data-tipx="${tip}" data-i="${q.i}"><i class="cb-pf" style="width:${(pf*100).toFixed(2)}%"></i></b>`);});
  const wpx=(g.clientWidth||520)-30, tk=ticks(X,wpx);
  const ghostHTML=gh?`<div class="cb-ghost" style="left:${P100(gh.r.mk,X)}"><span data-short="${esc(`${gh.kind==='b8'?ghostName(gh):(gh.r.s.pol==='cont'?'continuous':'static')} ${sec(gh.r.mk)}`)}">${tx(`${ghostName(gh)} done at ${sec(gh.r.mk)}`,`${ghostName(gh)} 在 ${sec(gh.r.mk)} 做完`)}</span></div>`:'';
  g.style.setProperty('--rh',rh+'px');
  g.innerHTML=`<div class="cb-lab-col" aria-hidden="true"><span class="cb-ql">${tx('wait','排队')}</span>${rows.map((_,k)=>`<span class="cb-rl">${lab(k)}</span>`).join('')}</div>`+
    `<div class="cb-plot" id="cb-plot"><div class="cb-q" aria-hidden="true"><svg viewBox="0 0 1000 100" preserveAspectRatio="none" class="cb-clip"><path class="cb-q-area" d="${d}"/></svg></div>`+
    `<div class="cb-rows">${rows.map(x=>`<div class="cb-track"><div class="cb-trk-in cb-clip">${x.join('')}</div></div>`).join('')}</div>`+
    `<div class="cb-axis" aria-hidden="true">${tk.map((v,i)=>{const p=v/X; return `<span style="left:${(p*100).toFixed(2)}%"${p>0.92?' class="cb-ax-r"':(i===0?' class="cb-ax-l"':'')}>${num(v,v<10&&v%1?1:0)}${i===tk.length-1?' s':''}</span>`;}).join('')}</div>`+
    `${ghostHTML}<div class="cb-head" id="cb-head" hidden></div></div>`;
  placeGhostLabel();
  // the legend lists the dashed line only while it is drawn
  const lg=$('#cb-lg-ghost',root);
  if(lg){lg.hidden=!gh; if(gh){lg.querySelector('span').textContent=tx(`when ${gh.kind==='b8'?'8 slots':polName(gh.r.s.pol)} finish${gh.kind==='b8'?'':'es'}`,`${ghostName(gh)} 做完的时刻`);
    lg.dataset.tipx=gh.kind==='b8'?tx(`The dashed line marks when continuous batching with 8 slots finishes the same 48 requests (${sec(gh.r.mk)}), the run this part compares with.`,`虚线标出 8 个 slot 的 continuous batching 做完这 48 个请求的时刻（${sec(gh.r.mk)}），这一部分就是和它比较。`)
      :tx(`The dashed line marks when ${polName(gh.r.s.pol)} finishes the same 48 requests with the same ${slotsTxt(gh.r.s.B)} (${sec(gh.r.mk)}).`,`虚线标出用同样 ${slotsTxt(gh.r.s.B)} 的 ${polName(gh.r.s.pol)} 做完这 48 个请求的时刻（${sec(gh.r.mk)}）。`);}}
}
// the dashed line's label sits right of the line, or left of it when it would stick out of the chart;
// on a narrow chart it shortens to the name and the time (the legend has the rest), and as a last resort it is kept inside the chart over the line
function placeGhostLabel(){const plot=$('#cb-plot',root), gl=plot&&plot.querySelector('.cb-ghost span'); if(!gl) return;
  const pw=plot.clientWidth, x=gl.parentNode.offsetLeft;
  const fit=()=>{const w=gl.offsetWidth; if(x+5+w<=pw){gl.classList.remove('cb-gl-l'); return true;} if(x-5-w>=0){gl.classList.add('cb-gl-l'); return true;} return false;};
  if(fit()) return; gl.textContent=gl.dataset.short; if(fit()) return;
  gl.classList.remove('cb-gl-l'); gl.style.left=Math.max(-x,pw-x-gl.offsetWidth)+'px';}
function playT(){const r=cur(); return play.t==null?r.mk:Math.min(play.t,r.mk);}
// while a run is shown only in part: it plays or it is paused part-way. Before it starts the stage badge stays empty (the
// clock line under Play already says 0 s); the mini result bar, which has no clock line, says it has not started
const runState=dock=>play.playing?tx('Running','运行中'):(playT()>1e-9?tx('Paused','已暂停'):(dock?tx('Not run yet','还没开始跑'):''));
const partial=()=>play.t!=null&&play.t<cur().mk-1e-9;
function counts(r,t){let done=0, wait=0, run=0; r.reqs.forEach(q=>{if(q.fin<=t+1e-9) done++; else if(q.start<=t+1e-9) run++; else if(q.arr<=t+1e-9) wait++;}); return {done,wait,run};}
function drawPlayhead(){
  if(!root) return; const r=cur(), X=xMax(), t=playT(), end=t>=r.mk-1e-9, c=counts(r,t);
  const g=$('#cb-plot',root), head=$('#cb-head',root), sc=$('#cb-scrub',root), clk=$('#cb-clock',root);
  if(g) g.style.setProperty('--cut',end?'0%':((1-t/X)*100).toFixed(3)+'%');
  if(head){head.hidden=end; head.style.left=P100(t,X);}
  if(sc){sc.value=String(Math.round(t/X*1000)); A.ui.setRange(sc);}
  if(clk) clk.textContent=end?tx(`At ${sec(t)}: all 48 requests are done.`,`${sec(t)}：48 个请求全部完成。`)
                            :tx(`At ${sec(t)}: ${c.run} requests running, ${c.wait} waiting, ${c.done} done.`,`${sec(t)}：${c.run} 个请求在跑，${c.wait} 个在排队，${c.done} 个已完成。`);
  $$('[data-act="play"]',root).forEach(b=>{b.textContent=playLabel();});
  // nothing shows the finished run before the playhead gets there: the badge tells the time, the whole-run cards wait (R3.8)
  const st=$('#cb-status',root), gh=ghost();
  if(st&&mode!=='challenge'){st.className='status cb-st';
    const tip=end&&gh?tx(`With ${gh.kind==='b8'?'continuous batching and 8 slots':`${polName(gh.r.s.pol)} and the same ${slotsTxt(gh.r.s.B)}`}, the same 48 requests are done at ${sec(gh.r.mk)} (the dashed line).`,
                         `同样这 48 个请求，${gh.kind==='b8'?'8 个 slot 的 continuous batching':`用同样 ${slotsTxt(gh.r.s.B)} 的 ${polName(gh.r.s.pol)}`} 在 ${sec(gh.r.mk)} 做完（图中的虚线）。`):'';
    if(tip){st.dataset.tipx=tip; st.tabIndex=0;} else {delete st.dataset.tipx; st.removeAttribute('tabindex');}
    st.textContent=end?tx(`All 48 requests done at ${sec(r.mk)}`,`48 个请求在 ${sec(r.mk)} 全部完成`):runState(); st.hidden=!st.textContent;}
  else if(st) st.hidden=false;
  const mt=$('#cb-metrics',root); if(mt&&mt.classList.contains('cb-pending')!==!end){mt.classList.toggle('cb-pending',!end); drawMetrics(r);}
  const lh=$('#cb-lg-head',root); if(lh) lh.hidden=end;
  const ga=$('#cb-gantt',root); if(ga) ga.setAttribute('aria-label',end?tx(`Schedule of the 48 requests in ${slotsTxt(r.s.B)} with ${polName(r.s.pol)}: all done at ${sec(r.mk)}. Use the arrow keys to move between requests.`,`${slotsTxt(r.s.B)}、${polName(r.s.pol)} 下 48 个请求的调度：${sec(r.mk)} 全部完成。用方向键在请求之间移动。`)
    :tx(`Schedule of the 48 requests in ${slotsTxt(r.s.B)} with ${polName(r.s.pol)}, drawn up to ${sec(t)}: ${c.done} of 48 done.`,`${slotsTxt(r.s.B)}、${polName(r.s.pol)} 下 48 个请求的调度，画到 ${sec(t)}：完成了 ${c.done} 个。`));
  const dk=(end?'end':sec(t)+'|'+c.done)+'|'+mode; if(dk!==play.dk){play.dk=dk; A.dock.render(); $$('.cb-req',root).forEach(refreshReqTip);}
}
function stopPlay(){play.playing=false; play.rate=0; if(play.raf) cancelAnimationFrame(play.raf); play.raf=0; if(root) $$('[data-act="play"]',root).forEach(b=>{b.textContent=playLabel();});}
function onPlayEnd(){play.t=null; watched.add(token()); stopPlay(); drawPlayhead(); if(mode==='guided'&&step===2){ran3=true; updateRail(cur());} showDone();}
// guided mode: when a run ends while the step card is off screen (phones: the card is above the chart), offer a way back to it
function showDone(){const btn=$('#cb-done',root); if(!btn) return; btn.hidden=true; if(mode!=='guided') return;
  const card=$('#cb-rail .step.current',root); if(!card) return; const box=$('[data-obs]',card), el=box&&!box.hidden?box:card, rc=el.getBoundingClientRect(), top=64;
  if(rc.bottom>top+24&&rc.top<window.innerHeight-24) return;
  const up=rc.top<top; btn.textContent=tx(`Run finished: the conclusion is in part ${step+1} ${up?'↑':'↓'}`,`跑完了，结论在第 ${step+1} 部分 ${up?'↑':'↓'}`); btn.hidden=false;}
function toCard(btn){btn.hidden=true; const card=$('#cb-rail .step.current',root); if(!card) return; const box=$('[data-obs]',card);
  const el=box&&!box.hidden&&card.offsetHeight>window.innerHeight*0.7?box:card; A.ui.reveal(el);
  const f=box&&!box.hidden?box:card; f.setAttribute('tabindex','-1'); f.focus({preventScroll:true});}
// the run advances by elapsed time × speed (seconds of the run per second), so the speed is the same on any display;
// rate: a one-off speed for this run only ("Show me" plays fast), which leaves the learner's own setting as it was
function startPlay(rate){
  const dn=$('#cb-done',root); if(dn) dn.hidden=true;
  if(play.playing){stopPlay(); drawPlayhead(); return;}
  if(A.reduceMotion()){onPlayEnd(); return;}
  const r=cur(); if(play.t==null||play.t>=r.mk-1e-9) play.t=0;
  play.playing=true; play.rate=rate||0; play.last=performance.now(); drawPlayhead();
  const tick=now=>{if(!root||!play.playing) return; const r2=cur(), dt=Math.min(0.1,(now-play.last)/1000); play.last=now;   // a long pause (hidden tab) does not jump ahead
    play.t+=dt*(play.rate||SPEEDS[speed]); if(play.t>=r2.mk){onPlayEnd(); return;} drawPlayhead(); play.raf=requestAnimationFrame(tick);};
  play.raf=requestAnimationFrame(tick);
}
function nextFinish(){stopPlay(); const r=cur(), t=play.t==null?-1:play.t; const f=r.reqs.map(q=>q.fin).filter(v=>v>t+1e-9).sort((x,y)=>x-y)[0];
  play.t=f==null?0:f; if(play.t>=r.mk-1e-9){onPlayEnd(); return;} drawPlayhead();}
function drawStage(r){
  const w=stats(r.wl);
  $('#cb-sub',root).textContent=tx(`${polName(r.s.pol,true)}: at most ${r.s.B===1?'1 request':`${r.s.B} requests`} at a time`,`${polName(r.s.pol)}：最多同时跑 ${r.s.B} 个请求`);
  $('#cb-sub2',root).textContent=tx(`48 requests, ${r.s.rate?`arriving at random, ${r.s.rate} per second on average`:'all waiting at 0 s'}, with 512-token prompts and answers of ${w.min} to ${w.max} tokens (capped at ${A.fmt.int(r.s.cap)}).${r.s.set?` This is request set ${r.s.set+1}.`:''}${r.s.eff!==50?` Prefill runs at ${r.s.eff}% of peak.`:''}`,
                                    `48 个请求${r.s.rate?`随机到达，平均每秒 ${r.s.rate} 个`:' 0 s 时全部在排队'}，prompt 各 512 个 token，回答 ${w.min} 到 ${w.max} 个 token（上限 ${A.fmt.int(r.s.cap)} 个）。${r.s.set?`这是第 ${r.s.set+1} 组请求。`:''}${r.s.eff!==50?`prefill 按峰值算力的 ${r.s.eff}% 计算。`:''}`);
  const lpf=$('#cb-lg-pf',root), pf1=num(PROMPT*TCD(r.s.eff),1)+' ms', bb=r.s.B, pfB=sec(stepTime(PROMPT*bb,0,r.s.eff)), stc=r.s.pol==='static'&&bb>1;
  if(lpf) lpf.dataset.tipx=tx(`Prefill is a request’s first step: its 512-token prompt runs through the model, and the first answer token comes out. Prefilling one prompt takes about ${pf1}`+(stc?`; static batching prefills a batch of ${r.s.rate?'up to ':''}${bb} prompts in one step of about ${pfB}.`:'.'),
                              `prefill 是一个请求的第一步：它 512 个 token 的 prompt 在这一步里过一遍模型，产出回答的第一个 token。一个 prompt 的 prefill 约 ${pf1}`+(stc?`；static batching 把一批${r.s.rate?'最多':''} ${bb} 个 prompt 放在同一步里 prefill，这一步约 ${pfB}。`:'。'));
  const st=$('#cb-status',root);
  if(mode==='challenge'){const ok=r.thr>=GOAL_THR&&r.tpot<=GOAL_TPOT; st.className='status '+(ok?'ok':'bad'); st.textContent=chBadge(r); st.tabIndex=0;
    st.dataset.tipx=tx(`The goal is at least ${GOAL_THR} tokens per second with at most 25 ms per output token on average. These settings give ${tps(r.thr)} tokens per second at ${ms(r.tpot)} per output token.`,`目标是每秒至少 ${GOAL_THR} 个 token，同时平均每个输出 token 不超过 25 ms。现在的设置是每秒 ${tps(r.thr)} 个 token，每个输出 token ${ms(r.tpot)}。`);}
  const note=$('#cb-note',root), over=r.peakBytes>KV_BUDGET;
  note.hidden=!over;
  note.textContent=over?tx(`The KV cache of this run would peak at ${gib(r.peakBytes)}, above the ${gib(KV_BUDGET)} left for it. A real engine would pause a request here (vLLM preempts one, SGLang retracts one); this simulation does not.`,`这次运行的 KV cache 峰值会到 ${gib(r.peakBytes)}，超过留给它的 ${gib(KV_BUDGET)}。真实引擎这时会让一个请求先停下（vLLM 叫 preempt，SGLang 叫 retract），这个模拟没有这样做。`):'';
}
// metric cards hold whole-run values, so while the run is shown partly they say when the value appears (R3.8).
// ▲▼ badges: only after the learner's own change (a control, an option, a scenario, Show me), computed from the values as
// shown before and after, and cleared when there is no change; entering a part, Back/Next, a mode switch or a state the
// lab sets itself never shows one (CB-A r7 #10, R7-B C1)
let userAct=false; const shownPrev={};
const learnerSet=p=>{userAct=true; setState(p);};
// a re-render the learner did not cause (a new part, a mode switch, the lab setting the state) clears any badge still showing
function clearDeltas(){if(!root) return; $$('#cb-metrics .metric .d',root).forEach(dd=>{clearTimeout(dd._t); clearTimeout(dd._t2); dd.classList.remove('on'); dd.textContent='';});}
function drawMetrics(r){const pend=partial(), host=$('#cb-metrics',root); if(!host) return;
  const key=skey(r.s)+'|'+pend+'|'+A.lang(); if(host._key===key) return; host._key=key;   // the same run drawn twice gives no badge and clears none
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`); if(!el) return; const [v,u]=m.v(r), vv=el.querySelector('.v'), sb=el.querySelector('.sub'), dd=el.querySelector('.d');
    el.dataset.tipx=m.tip(r);
    if(pend){vv.textContent=tx('shown after the run','跑完后显示'); sb.textContent='';}
    else {vv.innerHTML=esc(v)+(u?`<span class="cb-u"> ${u}</span>`:''); sb.textContent=m.sub(r);}
    const now=pend?null:m.raw(r), dt=userAct&&now!=null&&shownPrev[m.k]!=null?A.fmt.delta(shownPrev[m.k],now,{pts:!!m.pts,dp:m.dp}):'';
    clearTimeout(dd._t); clearTimeout(dd._t2);
    if(dt){dd.textContent=dt; dd.classList.add('on'); dd._t=setTimeout(()=>{dd.classList.remove('on'); dd._t2=setTimeout(()=>{dd.textContent='';},400);},2600);}
    else {dd.classList.remove('on'); dd.textContent='';}
    shownPrev[m.k]=now;});
}
function syncControls(r){
  $$('.seg-ctl[data-key]',root).forEach(g=>{const k=g.dataset.key; if(k==='speed') A.ui.setPressed(g,speed); else if(k in S) A.ui.setPressed(g,String(S[k]));});
  $$('input[data-key="B"]',root).forEach(i=>{if(document.activeElement!==i||+i.value!==S.B) i.value=S.B; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=slotsTxt(S.B);});
  $$('input[data-key="eff"]',root).forEach(i=>{i.value=S.eff; A.ui.setRange(i); const o=i.closest('.ctl').querySelector('output'); if(o) o.textContent=S.eff+'%';});
  const lock=mode==='challenge', ln=$('#cb-lock',root);
  $('#cb-controls',root).classList.toggle('cb-locked',lock); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you can change only the batching policy and the number of slots. Everything else stays fixed: one A100 40GB runs OPT-13B, the 48 requests of set 1 wait at 0 s with 512-token prompts and answers capped at 1,536 tokens, and prefill runs at 50% of peak.',
                             '挑战中只能改调度方式和 slot 数。其余保持不变：一张 A100 40GB 跑 OPT-13B；第 1 组的 48 个请求 0 s 时全部在排队，prompt 各 512 个 token，回答上限 1,536 个；prefill 按峰值算力的 50% 计算。');
}
// scenarios with live traffic: what limits each policy, stated from this run's own numbers
const KEEP_UP=0.5;                                                // mean wait (s) before a request starts, below which a policy keeps up with the arrivals
// one key number per paragraph, the mean wait before a request starts (R6.20); each policy's finish time is on the badge and the
// dashed line, and the arrivals on the chart's queue strip
function liveText(sO,cO){
  const keep=cO.mwait<KEEP_UP;
  const wt=v=>v<0.05?tx('under 0.1 s','不到 0.1 s'):sec(v), sp=v=>v<0.05?'':' ';              // Chinese: no space before 不到
  return tx((keep?`<p class="small">Continuous batching keeps up: a request waits ${wt(cO.mwait)} on average before it starts, so its throughput follows the arrival rate.</p>`
                 :`<p class="small">Continuous batching with ${slotsTxt(cO.s.B)} cannot keep up: a request waits ${wt(cO.mwait)} on average before it starts, so requests queue up for a slot.</p>`)+
      `<p class="small">With static batching, a request that arrives during a batch waits at least until the whole batch is done: ${wt(sO.mwait)} on average before it starts.</p>`,
    (keep?`<p class="small">continuous batching 跟得上：请求到达后平均等${sp(cO.mwait)}${wt(cO.mwait)} 就开始，所以它的吞吐取决于请求到达的速度。</p>`
          :`<p class="small">${slotsTxt(cO.s.B)} 的 continuous batching 跟不上：请求到达后平均要等${sp(cO.mwait)}${wt(cO.mwait)} 才开始，请求只能排队等 slot。</p>`)+
      `<p class="small">static batching 下，批次进行中到达的请求至少要等整批做完：平均等${sp(sO.mwait)}${wt(sO.mwait)} 才开始。</p>`);
}
function updateRail(r){
  if(!root) return; const rail=$('#cb-rail',root); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===leverNow())); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvResult(v);});
    $$('.choice[data-pred]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pred===pred)));
    const lbl=$('#cb-ic-play .lbl',rail); if(lbl) lbl.textContent=icLabel();
    if(s.expect&&s.expect(r)) observed[step]=true;
    if(box){const show=!!s.obs&&(!s.expect||!!observed[step]); box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(r);
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}}}
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=(!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const pn=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===pn)));
    const box=$('#cb-sb-res',rail); if(box){const sO=simulate({...S,pol:'static'}), cO=simulate({...S,pol:'cont'}), k=Math.round(cO.thr)/Math.round(sO.thr);
      const nm=pn>=0?esc(L(PRESETS[pn].t)):'';
      const lead=pn>=0?tx(`In the “${nm}” scenario, with ${slotsTxt(S.B)}, `,`在“${nm}”这个场景里，用 ${slotsTxt(S.B)} 时，`)
                      :tx(`Your answer cap and arrivals match none of the scenarios. With ${slotsTxt(S.B)}, `,`当前的回答上限和到达方式不属于任何一个场景。用 ${slotsTxt(S.B)} 时，`);
      // one key number, the throughput ratio from the shown values; the reason and the latency in words (R6.20)
      const thr=tx(`${lead}continuous batching makes ${times(k)} times as many tokens per second as static batching.`,
                   `${lead}continuous batching 每秒生成的 token 是 static batching 的 ${times(k)} 倍。`);
      let html=`<div class="eyebrow">${tx('Result','结果')}</div>`;
      if(S.rate>0) html+=tx(`<p>${thr} The requests arrive one by one, so neither policy can finish before the last one arrives.</p>`,`<p>${thr}请求是陆续到达的，所以两种方式都不可能在最后一个请求到达之前做完。</p>`)+liveText(sO,cO);
      else {
        // every request waits at 0 s: how busy each policy keeps its slots explains the gap
        const busy=S.cap<=32&&sO.util>=0.75
          ?tx(' The answers are all short and close in length, so static batching already keeps its slots busy most of the time, and there is little to gain.','回答都很短，长短也差不多，所以 static batching 的 slot 本来就大部分时间在干活，快不了多少。')
          :tx(` Static batching’s slots are ${cO.util-sO.util>0.15?'far ':''}less busy: each batch waits for its longest answer.`,`static batching 的 slot 干活的时间${cO.util-sO.util>0.15?'少得多':'少一些'}，因为每批都要等最长的回答。`);
        const same=sec(sO.mlat)===sec(cO.mlat), sooner=cO.mlat<sO.mlat;
        html+=`<p>${thr}${busy}</p>`+(same?tx('<p>A request takes about as long from arrival to its last token with either policy.</p>','<p>一个请求从到达到写完，两种方式用的时间差不多。</p>')
          :tx(`<p>With continuous batching, a request is also done ${sooner?'sooner':'later'} on average, counting from its arrival.</p>`,`<p>用 continuous batching 时，请求从到达到写完的平均时间也${sooner?'更短':'更长'}。</p>`));}
      if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again) A.ui.obsBump(box);}}
  } else {
    const v1=$('#cb-g1v',rail); if(!v1) return; const met1=r.thr>=GOAL_THR, met2=r.tpot<=GOAL_TPOT;
    v1.textContent=tps(r.thr); $('#cb-g1',rail).classList.toggle('met',met1);
    const p=$('#cb-g1p',rail); p.classList.toggle('met',met1); p.querySelector('i').style.width=Math.min(100,r.thr/GOAL_THR*100)+'%';
    const v2=$('#cb-g2v',rail); v2.textContent=ms(r.tpot)+(met2?' ✓':' ✗'); v2.className=met2?'okc':'badc';
    const res=$('#cb-ch-res',rail); if(chSubmitted&&met1&&met2) chReveal=true; const html=chResultHTML(); if(res._h!==html){res.innerHTML=html; res._h=html;}
    const sw=$('.cb-subwrap',rail); if(sw&&chSubmitted) sw.innerHTML='';
  }
}
function drawMath(r){$('#cb-mathbody',root).innerHTML=mathLines(r).join(''); fitMath($('#cb-mathbody',root));
  const c=$('#cb-mathchart',root), k='c'+r.s.eff+A.lang(); if(c._k!==k){c.innerHTML=stepChartHTML(r.s.eff); c._k=k;}}
let annT=null;
function render(){
  if(!root) return;
  const r=cur(); if(!userAct) clearDeltas();
  drawStage(r); drawGantt(r); drawPlayhead(); drawMetrics(r); drawMath(r); syncControls(r); updateRail(r);
  const dn=$('#cb-done',root); if(dn) dn.hidden=true;
  A.hash.write(token()); A.dock.refresh();
  const say=partial()?tx(`${polName(r.s.pol,true)} with ${slotsTxt(r.s.B)}: press Play to run the 48 requests.`,`${polName(r.s.pol)}，${slotsTxt(r.s.B)}：点播放，让 48 个请求跑一遍。`)
    :tx(`${polName(r.s.pol,true)} with ${slotsTxt(r.s.B)}: ${tps(r.thr)} tokens per second, slots busy ${pc(r.util)} of the time, ${ms(r.tpot)} per output token.`,`${polName(r.s.pol)}，${slotsTxt(r.s.B)}：每秒 ${tps(r.thr)} 个 token，slot 有 ${pc(r.util)} 的时间在干活，每个输出 token ${ms(r.tpot)}。`);
  clearTimeout(annT); annT=setTimeout(()=>A.announce(say),300);
  prevR=r; userAct=false;
}

/* ---------- state changes ---------- */
const token=(s=S)=>[s.pol,s.B,s.cap,s.rate,s.set+1,s.eff].join('.');
function parse(tok){if(!tok) return null; const p=String(tok).split('.'); if(p.length!==6) return null;
  const o={pol:p[0],B:+p[1],cap:+p[2],rate:+p[3],set:+p[4]-1,eff:+p[5]};
  if(!(o.pol==='static'||o.pol==='cont')||!Number.isInteger(o.B)||o.B<1||o.B>B_MAX||!CAPS.includes(o.cap)||!RATES.includes(o.rate)||!(o.set>=0&&o.set<SEEDS.length)||!EFFS.includes(o.eff)) return null;
  return o;}
function setState(p){
  Object.assign(S,p);
  S.B=Math.max(1,Math.min(B_MAX,Math.round(S.B)));
  if(mode==='challenge') Object.assign(S,{cap:CH.cap,rate:CH.rate,set:CH.set,eff:CH.eff});
  stopPlay(); play.t=startT(); render();
}
// guided steps 1-3 show an empty chart until the static run has been played in step 3: nothing shows a result before its run
const startT=()=>mode==='guided'&&step<=2&&!observed[2]?0:null;
function enterStep(i){
  step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) S={...s.enter()};
  stopPlay(); if(step===2&&!observed[2]) ran3=watched.has(token());   // R3.20: the same run already watched to the end is drawn whole and passes the gate
  play.t=step===2&&ran3?null:startT();
  renderRail(); render(); cue(s.target);
}
function cue(ids){if(!root) return; $$('.cue',root).forEach(el=>el.classList.remove('cue')); if(mode!=='guided') return;
  (ids||[]).forEach(id=>{const el=document.getElementById(id); if(el&&root.contains(el)){void el.offsetWidth; el.classList.add('cue');}});}
function setMode(m){
  if(!root||m===mode) return; const was=mode; mode=m; stopPlay();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH};}
  A.shell.setMode(root,m,DESCS()); const lab=$('.cb-lab',root); if(lab) lab.dataset.mode=m;
  if(m==='guided'){enterStep(step); return;}
  cue([]); play.t=null; renderRail(); render();
}

/* ---------- events (bound once per root; ignored when this lab is not mounted) ---------- */
const mine=e=>root&&e.target&&e.target.closest&&e.target.closest('.cb-lab')&&root.contains(e.target);
function onInput(e){
  if(!mine(e)) return; const el=e.target;
  if(el.id==='cb-scrub'){stopPlay(); const r=cur(), t=(+el.value)/1000*xMax(); play.t=t>=r.mk-1e-9?null:t; drawPlayhead(); if(play.t==null){watched.add(token()); if(mode==='guided'&&step===2){ran3=true; updateRail(r);} showDone();} return;}
  const k=el.dataset&&el.dataset.key;
  if(k==='B') learnerSet({B:+el.value}); else if(k==='eff') learnerSet({eff:+el.value});
}
function onClick(e){
  if(!mine(e)) return; const b=e.target.closest('button'); if(!b||!root.contains(b)) return; const d=b.dataset;
  // header (what A.shell.bindHeader does, bound once and only while this lab is mounted)
  if(d.pop){const pop=document.getElementById(d.pop); if(pop){const open=pop.hidden; pop.hidden=!open; b.setAttribute('aria-expanded',String(open));} return;}
  if(d.mode&&b.closest('.modes')){setMode(d.mode); A.ui.reveal($('#cb-rail',root)); return;}
  if(b.hasAttribute('data-share')){const frag='#'+ID+'~'+token(), box=root.querySelector('[data-sharebox]'), inp=box&&box.querySelector('input'); if(!inp) return;
    inp.value=frag; box.hidden=false; const done=()=>{b.textContent=tx('Copied','已复制'); setTimeout(()=>{b.textContent=A.t('share');},1600);}, fb=()=>{inp.focus(); inp.select();};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(frag).then(done,fb); else fb();}catch(err){fb();} return;}
  const seg=b.closest('.seg-ctl[data-key]');
  if(seg&&d.v!=null&&seg.dataset.key==='speed'){setSpeed(d.v); return;}
  if(seg&&d.v!=null&&seg.dataset.key in S){if(b.disabled) return; const k=seg.dataset.key; learnerSet({[k]:k==='pol'?d.v:+d.v}); return;}
  if(d.lever!=null){lvTried.add(d.lever); learnerSet({...DEF,pol:d.lever}); return;}
  if(d.pred){pred=d.pred; renderRail(); render(); const f=$(`#cb-rail [data-pred="${pred}"]`,root); if(f) f.focus(); return;}
  if(d.go!=null){enterStep(+d.go); A.ui.reveal($('#cb-rail .step.current',root)); return;}
  if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); A.ui.reveal($('#cb-rail .step.current',root)); return;}
  if(d.nav==='back'){enterStep(step-1); A.ui.reveal($('#cb-rail .step.current',root)); return;}
  if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe==='play'){if(play.playing) play.rate=SPEEDS.fast; else playFrom(b,SPEEDS.fast);} else if(s.showMe){if(s.levers) lvTried.add(s.showMe.pol); learnerSet(s.showMe);} return;}
  if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#cb-rail',root)); return;}
  if(d.preset!=null){learnerSet({...PRESETS[+d.preset].p}); const box=$('#cb-sb-res',root); if(box){const rc=box.getBoundingClientRect(); if(rc.top<60||rc.top>window.innerHeight-120) A.ui.reveal(box);} return;}
  const act=d.act; if(!act) return;
  if(act==='play') playFrom(b);
  else if(act==='to-card') toCard(b);
  else if(act==='next') nextFinish();
  else if(act==='reset') learnerSet(mode==='challenge'?{...CH}:{...DEF});
  else if(act==='to-cont') learnerSet({pol:'cont'});
  else if(act==='submit'){chSubmitted=true; if(cur().thr>=GOAL_THR&&cur().tpot<=GOAL_TPOT) chReveal=true; render(); const res=$('#cb-ch-res',root); if(res) res.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'});}
  else if(act==='reveal'){chReveal=true; render();}                // the other policy's line may now be drawn
  else if(act==='drill-new'){drillI=(drillI+1)%DRILL_B.length; drillOpen=false; drawDrill();}
  else if(act==='drill-reveal'){drillOpen=true; drawDrill();}
  else if(act==='copy'){const pre=$('#cb-code',root), done=()=>{b.textContent=tx('Copied','已复制'); setTimeout(()=>{b.textContent=A.t('copyCode');},1600);},
      sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); b.textContent=tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'); setTimeout(()=>{b.textContent=A.t('copyCode');},1600);};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
// Play pressed inside a step card while the chart is scrolled away (narrow screens): bring the chart into view to watch it
function playFrom(b,rate){const gr=$('#cb-graph',root);
  if(!play.playing&&b.closest('#cb-rail')&&gr){const rc=gr.getBoundingClientRect(); if(rc.top<60||rc.bottom>window.innerHeight) gr.scrollIntoView({block:'start',behavior:A.reduceMotion()?'auto':'smooth'});}
  startPlay(rate);}
function onKey(e){
  if(!mine(e)) return; const el=e.target; if(!el.classList||!el.classList.contains('cb-req')) return;
  const all=$$('.cb-req',root).sort((x,y)=>+x.dataset.i-+y.dataset.i), i=all.indexOf(el); let j=i;
  if(e.key==='ArrowRight'||e.key==='ArrowDown') j=Math.min(all.length-1,i+1); else if(e.key==='ArrowLeft'||e.key==='ArrowUp') j=Math.max(0,i-1);
  else if(e.key==='Home') j=0; else if(e.key==='End') j=all.length-1; else return;
  e.preventDefault(); if(j===i) return;
  const to=all[j]; all.forEach(x=>{x.tabIndex=-1;}); to.tabIndex=0; to.focus();
  // focusing may scroll the page, and a scroll hides the tooltip: show it again once the scroll has happened
  requestAnimationFrame(()=>requestAnimationFrame(()=>{if(root&&document.activeElement===to) A.tip.show(to,`<div>${esc(to.dataset.tipx||'')}</div>`);}));
}
// a request's tooltip is brought up to the moment shown just before the page's tooltip reads it (these run before the document's listeners)
function onTipIn(e){if(!mine(e)) return; const q=e.target.closest&&e.target.closest('.cb-req'); if(q) refreshReqTip(q);}
function bind(el){
  if(bound===el) return; bound=el;
  el.addEventListener('input',onInput); el.addEventListener('click',onClick); el.addEventListener('keydown',onKey);
  el.addEventListener('mouseover',onTipIn); el.addEventListener('focusin',onTipIn);
}

/* ---------- formula terms and jumps ---------- */
// one tooltip per value, naming the value first (CB-A r7 #12); built when shown, so the prefill share follows the setting
const TERM=()=>({
 params:{en:`${paramsExact()} (about ${A.fmt.big(PARAMS)}) is OPT-13B’s parameter count, computed from the layer sizes in its config.json; FP16 stores 2 bytes per parameter.`,zh:`${paramsExact()}（约 ${A.fmt.big(PARAMS)}）是 OPT-13B 的参数量，由它 config.json 里的层数和各层大小算出；FP16 每个参数占 2 字节。`},
 kvtok:{en:`${F.kv} is OPT-13B’s KV cache per token: 2 (key and value) × 40 layers × 5,120 × 2 bytes, from its config.json.`,zh:`${F.kv} 是 OPT-13B 每个 token 的 KV cache：2（key 和 value）× 40 层 × 5,120 × 2 字节，来自它的 config.json。`},
 bw:{en:`${F.bw} is the A100 40GB’s memory bandwidth (NVIDIA’s datasheet). Every step is assumed to read memory at this full speed, an optimistic estimate.`,zh:`${F.bw} 是 A100 40GB 的显存带宽（NVIDIA 数据手册）。这里假设每一步都按这个满速读显存，这是偏乐观的估计。`},
 peak:{en:`${F.peak} is the A100 40GB’s dense FP16 peak (NVIDIA’s datasheet). Prefill is assumed to reach ${S.eff}% of it.`,zh:`${F.peak} 是 A100 40GB 的 FP16 稠密算力峰值（NVIDIA 数据手册）。prefill 按它的 ${S.eff}% 计算。`},
 gmem:{en:`The A100 is sold as 40 GB; it is read here as ${F.mem}, as in the lab “KV cache memory”.`,zh:`A100 标称 40 GB 显存，这里按 ${F.mem} 计，和 KV cache 显存那个 Lab 一样。`},
 util:{en:`${F.util} is vLLM’s default gpu_memory_utilization: the engine uses at most ${F.utilPc} of GPU memory and keeps ${F.headPc} as headroom.`,zh:`${F.util} 是 vLLM 默认的 gpu_memory_utilization：引擎最多用 ${F.utilPc} 的显存，留 ${F.headPc} 作余量。`},
 oh:{en:`${F.oh} is the estimated runtime overhead: memory for temporary activations, CUDA graphs and buffers.`,zh:`${F.oh} 是估计的运行时开销：临时激活、CUDA graph 和缓冲区要的显存。`},
 prompt:{en:'Every prompt has 512 tokens, as in Anyscale’s benchmark.',zh:'每个 prompt 都是 512 个 token，和 Anyscale 的 benchmark 一样。'},
 ctx:{en:`Each running request’s KV cache is counted as ${AVG_CTX} tokens: its 512-token prompt plus 128 tokens, the mean the answer lengths are drawn with.`,zh:`每个在跑的请求的 KV cache 按 ${AVG_CTX} 个 token 算：512 个 token 的 prompt，加上抽取回答长度时设定的均值 128 个 token。`},
 B:{en:'The slots are the most requests the GPU runs in one step.',zh:'slot 数是 GPU 每一步最多同时跑的请求数。'},
 pol:{en:'The batching policy decides when a free slot is refilled: static batching waits for the whole batch, and continuous batching refills it at the next step.',zh:'调度方式决定空出的 slot 什么时候补上：static batching 等整批做完，continuous batching 下一步就补上。'},
 set:{en:'The request set says which random draw of the 48 requests is used; set 1 is the default.',zh:'请求组指用哪一次随机抽出的 48 个请求，默认是第 1 组。'},
 eff:{en:`Prefill is assumed to reach this share of the ${F.peak} peak; it is an estimate.`,zh:`假设 prefill 能用到 ${F.peak} 峰值的这个比例，这是估计值。`},
 cap:{en:'The answer cap is the longest answer allowed.',zh:'回答上限是允许的最长回答。'}, rate:{en:'Arrivals says how the requests arrive.',zh:'到达方式指请求怎么到达。'}
});
const SETUP={en:'Setup',zh:'配置'}, MEMN={en:'Memory left for the KV cache',zh:'留给 KV cache 的显存'};
const CTLNAME={pol:{en:'Batching policy',zh:'调度方式'},B:{en:'Slots',zh:'slot 数'},ctx:{en:'Slots',zh:'slot 数'},cap:{en:'Answer cap',zh:'回答上限'},rate:{en:'Arrivals',zh:'到达方式'},set:{en:'Request set',zh:'请求组'},
  eff:{en:'Prefill speed',zh:'prefill 速度'},params:SETUP,kvtok:SETUP,bw:SETUP,peak:SETUP,gmem:SETUP,prompt:SETUP,util:MEMN,oh:MEMN,mem:MEMN};
const CTLID={pol:'pol',B:'B',ctx:'B',cap:'cap',rate:'rate',set:'set',eff:'eff',params:'setup',kvtok:'setup',bw:'setup',peak:'setup',gmem:'setup',prompt:'setup',util:'mem',oh:'mem',mem:'mem'};
const FIXED=()=>{const model=tx('OPT-13B on one A100 40GB','一张 A100 40GB 跑 OPT-13B'), gpu=tx('one A100 40GB','一张 A100 40GB'), mem=tx(`${gib(KV_BUDGET)} is left for the KV cache`,`留给 KV cache 的显存是 ${gib(KV_BUDGET)}`);
  return {params:model,kvtok:model,bw:gpu,peak:gpu,gmem:gpu,prompt:tx('every prompt has 512 tokens','每个 prompt 都是 512 个 token'),util:mem,oh:mem,mem,
    cap:tx('answers are capped at 1,536 tokens','回答上限是 1,536 个 token'),rate:tx('all 48 requests wait at 0 s','48 个请求 0 s 时全部在排队'),set:tx('request set 1 is used','用第 1 组请求'),eff:tx('prefill runs at 50% of the peak','prefill 按峰值算力的 50% 计算')};};
const AREAS=['math','controls','stage'];
const locked=k=>mode==='challenge'&&!AREAS.includes(k)&&!['pol','B','ctx'].includes(k);
// the jump says where it lands: a control in the part's card, or one under Settings (SITE r7 #17b)
function jumpAct(k){
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='stage') return tx('Click to jump to the schedule chart.','点击跳到调度图。');
  if(locked(k)) return tx(`Fixed during the challenge: ${FIXED()[k]}.`,`挑战中固定不变：${FIXED()[k]}。`);
  const n=L(CTLNAME[k]||{en:k,zh:k}), el=ctlEl(k), inPanel=!el||!!el.closest('#cb-controls');
  return inPanel?tx(`Click to jump to “${n}” under Settings.`,`点击跳到“自由调参”里的“${n}”。`):tx(`Click to jump to “${n}”.`,`点击跳到“${n}”。`);
}
function ctlEl(k){const id=CTLID[k]; if(!id||!root) return null; return $('#cb-ic-'+id,root)||$('#cb-ctl-'+id,root);}
// areas land below the top bar and the mini result bar, then flash (SITE r7 #6)
function jump(k,src){
  if(!root) return;
  if(AREAS.includes(k)){const el=k==='math'?$('#cb-math',root):(k==='controls'?$('#cb-controls',root):$('#cb-graph',root)); if(!el) return;
    if(k==='math') el.open=true; A.tip.hide(); A.ui.reveal(el); A.ui.flash(el); return;}
  if(locked(k)){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* ---------- registration ---------- */
// the kit keeps the settings token with a Scenarios snapshot and compares it with the address on reload; this lab binds its own
// header (above), so it hands the kit its token the way A.shell.bindHeader takes it (on a detached element that is never clicked)
A.shell.bindHeader(document.createElement('div'),{id:ID,token:()=>token()});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,ran3,watched:[...watched],
    lvTried:[...lvTried],speed,drillI,drillOpen,passed:mode==='challenge'&&chSubmitted&&cur().thr>=GOAL_THR&&cur().tpot<=GOAL_TPOT}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; S=p; if(mode==='challenge') Object.assign(S,{cap:CH.cap,rate:CH.rate,set:CH.set,eff:CH.eff});
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; ran3=o.ran3===true;
    watched.clear(); if(Array.isArray(o.watched)) o.watched.forEach(t=>{if(typeof t==='string'&&parse(t)) watched.add(t);});
    lvTried.clear(); lvTried.add('static'); if(Array.isArray(o.lvTried)) o.lvTried.forEach(t=>{if(t==='static'||t==='cont') lvTried.add(t);});
    if(SPEEDS[o.speed]) speed=o.speed; if(Number.isInteger(o.drillI)&&o.drillI>=0&&o.drillI<DRILL_B.length) drillI=o.drillI; drillOpen=o.drillOpen===true;
    // what enterStep() sets for the part, without resetting the settings: mount then draws this part as a return in the same session does
    inited=true; stopPlay(); play.t=mode!=='guided'?null:(step===2&&ran3?null:startT());},
  mount(el,tok){
    root=el; const p=parse(tok);
    if(p){S={...p}; mode='sandbox'; saved=null; stopPlay(); play.t=null;}      // a shared setup opens in the scenarios mode with the whole run shown
    el.innerHTML=view(); A.ui.initRanges(el); bind(el);
    drawDrill();
    if(!inited&&!p){inited=true; mode='guided'; enterStep(0); return;}
    inited=true; renderRail(); render(); if(mode==='guided') cue(STEPS[step].target);
  },
  leave(){stopPlay(); clearTimeout(annT); root=null;},
  onLang(){if(root){const t=play.t; stopPlay(); this.mount(root,null); play.t=t; drawPlayhead();}},
  onResize(){if(root){drawGantt(cur()); drawPlayhead();}},
  termTip(k){const t=TERM()[k]; return `<div>${esc(t?L(t):k)}</div><div class="act">${jumpAct(k)}</div>`;},
  jump, jumpAct,
  dock:{anchor:()=>root&&$('#cb-gantt',root), bench:()=>root&&$('.cb-bench',root), target:()=>root&&$('#cb-graph',root),
    // challenge: the goal and how far the settings get; during a run: the clock only; after it: throughput and how busy the slots were
    render({status,vis,read}){if(!root) return; const r=cur(), bar=(p,cls)=>`<div class="cb-dk${cls?' '+cls:''}" aria-hidden="true"><i class="cb-dk-b" style="width:${(Math.max(0,Math.min(1,p))*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const ok1=r.thr>=GOAL_THR, ok2=r.tpot<=GOAL_TPOT, ok=ok1&&ok2; status.className='status '+(ok?'ok':'bad'); status.textContent=ok?tx('Goals met','达到目标'):tx('Not yet','还没达到');
        vis.innerHTML=bar(r.thr/GOAL_THR,ok1?'met':'');
        read.textContent=tx(`${tps(r.thr)} tokens/s, ${ms(r.tpot)} per output token`,`每秒 ${tps(r.thr)} 个 token，每个输出 token ${ms(r.tpot)}`);}   // the bar is the throughput against the goal
      else if(partial()){const t=playT(), c=counts(r,t); status.className='status cb-st'; status.textContent=runState(true);
        vis.innerHTML=bar(c.done/NREQ,'');
        read.textContent=tx(`At ${sec(t)}: ${c.done} of 48 requests done`,`${sec(t)}：48 个请求完成了 ${c.done} 个`);}
      else {status.className='status cb-st'; status.textContent=tx(`${tps(r.thr)} tokens/s`,`每秒 ${tps(r.thr)} 个 token`);
        vis.innerHTML=bar(r.util,'');
        read.innerHTML=tx(`slots busy ${pc(r.util)} of the time<span class="dk-mem"> with ${polName(r.s.pol)} and ${slotsTxt(r.s.B)}</span>`,
                          `slot 有 ${pc(r.util)} 的时间在干活<span class="dk-mem">（${polName(r.s.pol)}，${slotsTxt(r.s.B)}）</span>`);}}}
});

/* ---------- home tile teaser: a toy run of both policies, 4 slots, 10 made-up answers, one token per step ---------- */
const TZ_LENS=[4,11,2,6,3,8,5,2,9,3], TZ_B=4; let tzPol='static';
// lens and B default to the tile's requests and slots; the capability layer's new case passes its own (EXITS above)
function tzSim(pol,lens=TZ_LENS,B=TZ_B){
  const bars=[], idle=[]; let t=0;
  // idle: a slot is idle while requests are still waiting (hatched, as in the lab); idle time with nobody waiting is left blank
  if(pol==='static'){for(let k=0;k<lens.length;k+=B){const g=lens.slice(k,k+B), m=Math.max(...g), waiting=k+B<lens.length;
      g.forEach((len,j)=>{bars.push({slot:j,a:t,b:t+len,len}); if(len<m&&waiting) idle.push({slot:j,a:t+len,b:t+m});});
      if(waiting) for(let j=g.length;j<B;j++) idle.push({slot:j,a:t,b:t+m}); t+=m;}
  } else {const run=new Array(B).fill(null); let nxt=0;
    while(nxt<lens.length||run.some(x=>x)){
      for(let j=0;j<B;j++) if(!run[j]&&nxt<lens.length){run[j]={len:lens[nxt],a:t,done:0}; nxt++;}
      t++;
      for(let j=0;j<B;j++){const q=run[j]; if(!q) continue; q.done++; if(q.done>=q.len){bars.push({slot:j,a:q.a,b:t,len:q.len}); run[j]=null;}}}
  }   // continuous batching: a slot is idle only after the last request has started, when nobody waits: blank
  const tok=lens.reduce((x,y)=>x+y,0);
  return {steps:t,bars,idle,tok,busy:tok/(B*t)};
}
A.registerTeaser(ID,{render(el){
  const s=tzSim('static'), c=tzSim('cont'), r=tzPol==='cont'?c:s, X=s.steps;          // one scale for both policies, so the shorter run looks shorter
  const pp=v=>(v/X*100).toFixed(2)+'%';
  const rows=[]; for(let j=0;j<TZ_B;j++) rows.push(
    r.idle.filter(x=>x.slot===j).map(x=>`<i class="tz-cb-idle" style="left:${pp(x.a)};width:${pp(x.b-x.a)}"></i>`).join('')+
    r.bars.filter(x=>x.slot===j).map(x=>`<b class="tz-cb-req" style="left:${pp(x.a)};width:${pp(x.b-x.a)}">${x.len}</b>`).join(''));
  // one key number, in bold: how busy the slots are; each run's length is on the axis under the bars (R6.20, R11.5)
  const read=tzPol==='cont'
    ?tx(`Continuous batching gives a free slot to the next request at the very next step: the slots are busy <b>${pc(c.busy)}</b> of the time, so these requests finish ${c.steps<s.steps?'sooner':'no later'}.`,
        `continuous batching 在 slot 空出来的下一步就补进一个请求：slot 有 <b>${pc(c.busy)}</b> 的时间在干活，这些请求也就${c.steps<s.steps?'更早':'同样'}做完。`)
    :tx(`Static batching starts these requests a batch at a time, and each batch waits for its longest answer: the slots are busy only <b>${pc(s.busy)}</b> of the time.`,
        `static batching 让这些请求一批一批地开始，每批都要等最长的回答写完：slot 只有 <b>${pc(s.busy)}</b> 的时间在干活。`);
  const aria=tx(`${tzPol==='cont'?'Continuous':'Static'} batching: 10 requests in 4 slots take ${r.steps} steps.`,`${tzPol==='cont'?'continuous':'static'} batching：4 个 slot 跑 10 个请求，用 ${r.steps} 步。`);
  const notes=[
    tx('In each step, every running request generates one token, so a bar is as many steps long as its answer has tokens. The step that reads the prompt (prefill) is left out here.','每一步，每个在跑的请求生成 1 个 token，所以一条有几步长，这个回答就有几个 token。这里不算读入 prompt 的那一步（prefill）。'),
    tx(`The 10 answer lengths (${TZ_LENS.join(', ')} tokens) were hand-picked to show the effect; they are not the lab’s requests.`,`这 10 个回答的长度（${TZ_LENS.join('、')} 个 token）是为了演示手工挑的，不是 Lab 里的请求。`),
    tx(`${pc(s.busy)} and ${pc(c.busy)} are the share of the 4 slots’ steps that generate a token: the 10 answers have ${s.tok} tokens; static batching takes ${s.steps} steps, so ${s.tok} ÷ (4 × ${s.steps}) = ${pc(s.busy)}, and continuous batching ${c.steps}, so ${c.tok} ÷ (4 × ${c.steps}) = ${pc(c.busy)}.`,
       `${pc(s.busy)} 和 ${pc(c.busy)} 是 4 个 slot 在所有步里真正生成 token 的比例：10 个回答共 ${s.tok} 个 token，static batching 用 ${s.steps} 步，${s.tok} ÷（4 × ${s.steps}）= ${pc(s.busy)}；continuous batching 用 ${c.steps} 步，${c.tok} ÷（4 × ${c.steps}）= ${pc(c.busy)}。`),
    tx('The lab itself simulates 48 requests on one A100 40GB serving OPT-13B, with prefill and estimated step times.','Lab 里模拟的是一张 A100 40GB 跑 OPT-13B 的 48 个请求，算上 prefill 和估算的每步耗时。')];
  const endP=r.steps/X, ax=`<span class="tz-cb-end${endP>0.85?' tz-cb-end-r':''}" style="left:${endP>0.85?'100%':(endP*100).toFixed(2)+'%'}">${tx(`${r.steps} steps`,`${r.steps} 步`)}</span>`;
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-cb-pol',key:'tzcb',label:tx('Batching policy','调度方式'),options:[['static',tx('Static','static')],['cont',tx('Continuous','continuous')]],value:tzPol})+
    `<div class="tz-cb-cap">${tx('The GPU runs at most 4 requests at once, one per slot. Each row is a slot, each bar a request.','GPU 一次最多跑 4 个请求，每个在跑的请求占一个 slot。每一行是一个 slot，每一条是一个请求。')}</div>`+
    `<div class="tz-vis tz-cb-g" role="img" aria-label="${esc(aria)}">${rows.map(x=>`<div class="tz-cb-row">${x}</div>`).join('')}`+
    `<div class="tz-cb-ax" aria-hidden="true"><span class="tz-cb-0">0</span>${ax}</div></div>`+
    `<div class="tz-leg"><span><i class="tz-cb-lg" aria-hidden="true">4</i>${tx('number: tokens in that answer','数字：这个回答的 token 数')}</span><span><i class="tz-cb-sw-idle" aria-hidden="true"></i>${tx('idle while requests wait','有请求在等时空着')}</span></div>`+
    `<p class="tz-read">${read}</p>${notes.map(n=>`<p class="tz-note">${esc(n)}</p>`).join('')}</div>`;
  if(!el._cbTz){el._cbTz=true; el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl[data-key="tzcb"] button[data-v]'); if(!b||!el.contains(b)) return; tzPol=b.dataset.v; this.render(el); const f=el.querySelector(`.seg-ctl[data-key="tzcb"] button[data-v="${tzPol}"]`); if(f) f.focus();});}
}});

/* read-only hook for the golden-number tests */
window.__cb={PARAMS,W_BYTES,KV_TOK,KV_BUDGET,B_MAX,B_CUDA,KV_CUDA,B_OH1,B_OH3,Tw,Tc,kvRead,stepTime,joinStep,plainStep,workload,SEEDS,CAPS,RATES,GOAL_THR,GOAL_TPOT,KEEP_UP,
  okIf:x=>chOkIf(x), ghost:()=>{const g=ghost(); return g?{kind:g.kind,pol:g.r.s.pol,B:g.r.s.B,mk:g.r.mk}:null;}, partial:()=>partial(),
  sim:s=>{const r=simulate({...DEF,...s}); return {mk:r.mk,thr:r.thr,util:r.util,idleWait:r.idleWait,mlat:r.mlat,p90:r.p90,tpot:r.tpot,tokens:r.tokens,busy:r.busy,peak:r.peak,pre:r.pre,steps:r.steps,lastStart:r.lastStart,mwait:r.mwait,lastArr:r.lastArr,
    reqs:r.reqs.map(q=>({i:q.i,L:q.L,arr:q.arr,start:q.start,first:q.first,fin:q.fin,slot:q.slot}))};},
  best:()=>{const b=chBest(); return b?{pol:b.pol,B:b.B}:null;}, okSet:()=>chOptions().filter(o=>o.ok).map(o=>o.pol+':'+o.B),
  tz:()=>({static:tzSim('static'),cont:tzSim('cont')}), stepRows, drillEmp, state:()=>({...S,mode,step}),
  play:()=>({t:play.t,playing:play.playing,speed,rate:play.rate}), reqTip:i=>{const r=cur(); return reqTip(r,r.reqs[i],partial()?playT():null);},
  terms:()=>{const T0=TERM(); return Object.keys(T0).map(k=>L(T0[k])+'\n'+jumpAct(k)).join('\n');}, waitPrefills:s=>waitPrefills(simulate({...DEF,...s}))};
})();
