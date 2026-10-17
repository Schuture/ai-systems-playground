/* prefix-caching (the "SDE → LLM serving" path, stop 7 of 7): How can a 10,000-token system prompt be nearly free?
   Fidelity: Toy run. Two parts, both in this file:
   - the time model is teasers/set-a.js's step model (the same constants and formulas as the home tile and labs/prefill-decode.js):
     Llama-3.1-8B in BF16 on one H100 SXM, a step takes the longer of its arithmetic (50% of the dense BF16 peak) and its memory
     traffic (60% of the peak bandwidth). The first token of a prompt of N tokens whose first H tokens are cached (H a multiple of 16)
     waits for step({tok:N−H, logits:1, attn:attn(H,N), kv:N}); without a hit H = 0, which is prefill-decode's prefill.
   - a small block-hash prefix cache that really runs in the browser: prompts are lists of segments (content id + token count), each
     token gets an id from its segment's content, a 16-token block's key is a hash of its parent block's key and its 16 token ids
     (vLLM's hash_block_tokens), a hit is the longest run of leading full blocks already cached, capped so that at least one prompt
     token is computed (vLLM's get_computed_blocks). After a request its full blocks of prompt + answer (minus the answer's last token,
     which has no KV yet) are cached; freed blocks are ordered least recently used first, a request's last block first (vLLM's
     KVCacheManager.free), and evicted in that order when the cached blocks exceed a stated budget.
   Words (EN / ZH): prefix caching; cached / 缓存的; computed / 现算; cache hit / 命中; miss / 未命中; block (16 tokens); block key / block 的
   key; evict / 淘汰; time to first token (TTFT) / 首 token 延迟（TTFT）; part / 部分 for guided cards.
   window.__px is a read-only hook for tests/prefix-caching_numbers.py and tests/prefix-caching_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='prefix-caching';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const MODEL='Llama‑3.1‑8B', GPUN='H100 SXM';      // no-break hyphens and space: a model or GPU name never breaks

/* =====================================================================
   engine 1: the step model of teasers/set-a.js (same constants, same formulas)
   ===================================================================== */
const H100={peak:1979e12/2,bw:3.35e12,mem:80};                  // dense BF16 = the datasheet's 1,979 TFLOPS "with sparsity" ÷ 2
const EFF={mfu:0.5,mbu:0.6};                                    // assumed: 50% of peak FLOP/s, 60% of peak bandwidth (as set-a)
const LL={layers:32,hidden:4096,heads:32,kvHeads:8,headDim:128,vocab:128256,ffn:14336};
const P_EMB=LL.vocab*LL.hidden, P_HEAD=LL.vocab*LL.hidden;
const P_LAYER=2*LL.hidden*LL.hidden+2*LL.hidden*LL.kvHeads*LL.headDim+3*LL.hidden*LL.ffn+2*LL.hidden;
const P_LAYERS=LL.layers*P_LAYER+LL.hidden, P_TOTAL=P_LAYERS+P_EMB+P_HEAD;
const KV_TOK=2*LL.layers*LL.kvHeads*LL.headDim*2;              // K and V, BF16: 131,072 bytes = 128 KiB per token
const W_READ=2*(P_LAYERS+P_HEAD);                               // bytes of weights one step reads (not the embedding table: a lookup)
const W_MEM=2*P_TOTAL;
const attn=(a,b)=>2*LL.layers*LL.hidden*(b*(b+1)-a*(a+1));      // QKᵀ and AV FLOPs, all layers, for query positions a..b-1
function stepOf(o){
  const fl=2*P_LAYERS*o.tok+2*P_HEAD*o.logits+o.attn, by=W_READ+KV_TOK*o.kv, fs=H100.peak*EFF.mfu, bs=H100.bw*EFF.mbu;
  const tc=fl/fs, tm=by/bs;
  return {t:Math.max(tc,tm),tc,tm,fl,by,cb:tc>=tm};
}
/* the first token of an N-token prompt whose first H tokens come from the cache */
const ttft=(N,H)=>stepOf({tok:N-H,logits:1,attn:attn(H,N),kv:N});
/* a decode step of one request at context c (the timeline's ticks) */
const decode=c=>stepOf({tok:1,logits:1,attn:attn(c,c+1),kv:c});
/* the KV cache pool on one H100 for Llama-3.1-8B: the KV cache memory lab's model (80 GB read as 80 GiB, 90% usable, 2 GiB overhead) */
const KVD=A.kv&&A.kv.DEFAULT?A.kv.DEFAULT:{util:0.9,overhead:2};
function poolBytes(){
  if(A.kv&&A.kv.compute){const m=A.kv.compute({model:'llama31-8b',gpu:'h100',n:1,ctx:1,users:1,wp:'bf16',kp:'bf16',attn:'gqa',util:KVD.util,overhead:KVD.overhead}); return m.budget;}
  return H100.mem*A.GiB*KVD.util-W_MEM-KVD.overhead*A.GiB;}
const POOL_B=poolBytes(), POOL_T=Math.floor(POOL_B/KV_TOK);

/* =====================================================================
   engine 2: a block-hash prefix cache, run in the browser
   ===================================================================== */
const BLOCK=16, ANS=300, DATE_T=12, NAME_T=8;
function h32(s){let h=0x811c9dc5; for(let i=0;i<s.length;i++){h^=s.charCodeAt(i); h=Math.imul(h,0x01000193);} return h>>>0;}
function fmix(h){h^=h>>>16; h=Math.imul(h,0x85ebca6b); h^=h>>>13; h=Math.imul(h,0xc2b2ae35); h^=h>>>16; return h>>>0;}
/* a segment's token ids follow from its content: the same content id gives the same ids; an edit changes one token */
function segIds(sg,out){const base=h32(sg.cid);
  for(let i=0;i<sg.n;i++){let v=fmix((base+Math.imul(i+1,0x9e3779b1))>>>0)%LL.vocab; if(sg.edit===i) v=(v+1)%LL.vocab; out.push(v);}
  return out;}
function idsOf(segs){const out=[]; segs.forEach(s=>segIds(s,out)); return out;}
/* the key of each full block: a hash of its parent block's key and its 16 token ids (two 32-bit halves, 53 bits kept);
   and a content key, a hash of the 16 token ids alone, to spot a block whose tokens are cached after a different prefix */
function keysOf(ids,nb){
  const key=new Array(nb), ck=new Array(nb); let a=0x243f6a88, b=0x85a308d3;
  for(let k=0;k<nb;k++){let x=a, y=b, z=0x13198a2e;
    for(let j=0;j<BLOCK;j++){const t=ids[k*BLOCK+j]; x=fmix((x^t)+j+0x9e3779b9>>>0); y=fmix(Math.imul(y^t,0x01000193)+j>>>0); z=fmix((z^t)+Math.imul(j+1,0x632be5ab)>>>0);}
    a=x; b=y; key[k]=x*2097152+(y>>>11); ck[k]=z;}
  return {key,ck};}
function newCache(capBlocks){return {cap:capBlocks,m:new Map(),cc:new Map()};}   // m: key → content key, least recently used first
function cPut(c,k,ck){if(c.m.has(k)) c.m.delete(k); else c.cc.set(ck,(c.cc.get(ck)||0)+1); c.m.set(k,ck);}
function cEvict(c){let n=0; while(c.m.size>c.cap){const k=c.m.keys().next().value, ck=c.m.get(k); c.m.delete(k); const v=c.cc.get(ck)-1; if(v) c.cc.set(ck,v); else c.cc.delete(ck); n++;} return n;}
/* one request: find its hit, mark every block (0 cached, 1 computed, 2 computed although the same tokens are cached after another
   prefix), then cache its full blocks of prompt + answer in reverse order (the tail is evicted first) and evict down to the budget */
function runOne(c,req){
  const ids=idsOf(req.segs), N=ids.length, nb=Math.floor(N/BLOCK), K=keysOf(ids,nb);
  const maxHit=Math.floor((N-1)/BLOCK); let hit=0;
  while(hit<Math.min(nb,maxHit)&&c.m.has(K.key[hit])) hit++;
  const total=Math.ceil(N/BLOCK), state=new Array(total);
  for(let b=0;b<total;b++) state[b]=b<hit?0:(b<nb&&c.cc.has(K.ck[b])?2:1);
  const all=idsOf(req.segs.concat([req.ans])).slice(0,N+req.ans.n-1), nb2=Math.floor(all.length/BLOCK), K2=keysOf(all,nb2);
  for(let b=nb2-1;b>=0;b--) cPut(c,K2.key[b],K2.ck[b]);
  const ev=cEvict(c);
  return {N,hit:hit*BLOCK,hitBlocks:hit,nb,total,state,evicted:ev,cached:c.m.size};}
/* how many leading blocks of a segment list are cached right now (which system prompts the cache still holds) */
function leadCached(c,segs){const ids=idsOf(segs), nb=Math.floor(ids.length/BLOCK), K=keysOf(ids,nb); let h=0; while(h<nb&&c.m.has(K.key[h])) h++; return h;}

/* =====================================================================
   the lab's streams of requests
   ===================================================================== */
const P_STOPS=[1000,2000,5000,10000,20000,50000,100000], O_STOPS=[50,100,200,500,1000,2000], N_MAX=8;
const APPS='AABACABA';                                          // part 6: three apps, A the busiest
const CHANGES=['none','dateStart','dateAfter','edit','nameStart'];
const MEMS=['pool',3,2,1];
const BASE={P:10000,O:200,n:2,kind:'same',change:'none',apps:1,mem:'pool'};
/* the memory for cached blocks: the whole pool, or room for K requests' prompts and answers */
const memTokens=s=>s.mem==='pool'?POOL_T:s.mem*(s.P+s.O+ANS);
/* the index of the token that "edit" changes: the middle of the prefix (part 4, Settings), or where part 5 puts it (s.at) */
const editAt=s=>s.at!=null?s.at:Math.floor(s.P/2)-1;
function buildStream(s){
  const reqs=[];
  if(s.kind==='chat'){let hist=[];
    for(let i=0;i<s.n;i++){const msg={k:'msg',cid:'chat-msg-'+i,n:s.O}, ans={k:'ans',cid:'chat-ans-'+i,n:ANS};
      reqs.push({i,app:'A',segs:[{k:'pre',cid:'pre-A',n:s.P}].concat(hist,[msg]),ans});
      hist=hist.concat([{k:'hist',cid:msg.cid,n:msg.n},{k:'hist',cid:ans.cid,n:ans.n}]);}
    return reqs;}
  for(let i=0;i<s.n;i++){
    const app=s.apps===3?APPS[i]:'A';
    const pre={k:'pre',cid:'pre-'+app,n:s.P,edit:(s.change==='edit'&&i>0)?editAt(s):null};
    const own={k:'own',cid:'own-'+i,n:s.O}, day=i===0?1:2;
    const date={k:'date',cid:'date-'+day,n:DATE_T}, name={k:'name',cid:'name-'+i,n:NAME_T};
    const segs=s.change==='dateStart'?[date,pre,own]:(s.change==='dateAfter'?[pre,date,own]:(s.change==='nameStart'?[name,pre,own]:[pre,own]));
    reqs.push({i,app,day,segs,ans:{k:'ans',cid:'ans-'+i,n:ANS}});}
  return reqs;}
/* the whole stream: every request's hit, its time to first token with and without the cache */
const RC=new Map();
function run(s){
  const key=[s.P,s.O,s.n,s.kind,s.change,s.apps,s.mem,s.at==null?'':s.at].join('|'); if(RC.has(key)) return RC.get(key);
  const reqs=buildStream(s), capT=memTokens(s), c=newCache(Math.floor(capT/BLOCK)), res=[];
  reqs.forEach(q=>{const r=runOne(c,q); const w=ttft(r.N,r.hit), wo=ttft(r.N,0);
    const apps={}; if(s.apps===3&&s.kind==='same') ['A','B','C'].forEach(a=>{apps[a]=leadCached(c,[{k:'pre',cid:'pre-'+a,n:s.P}]);});
    res.push({...r,req:q,w,wo,apps});});
  const tot={w:res.reduce((a,r)=>a+r.w.t,0),wo:res.reduce((a,r)=>a+r.wo.t,0),pre:res.reduce((a,r)=>a+r.N-r.hit,0),all:res.reduce((a,r)=>a+r.N,0),
    hits:res.filter(r=>r.hit>0).length,cacheT:res.length?res[res.length-1].cached*BLOCK:0,capT};
  const out={s:{...s},reqs,res,tot}; RC.set(key,out); return out;}

/* part 5, the experiment: request 2 changes one token of the mission's system prompt, at the front (a token of block 1), in the middle
   (token 5,000, part 4's word) or at the end (a token of the system prompt's last full block). Every number below comes from run() on
   the mission (request 1, then request 2), from the block keys of keysOf() and from the step model ttft() */
const OT_POS=[{v:'front',t:10},{v:'middle',t:BASE.P/2},{v:'end',t:BASE.P-10}];   // the changed token, counted from 1
const otTok=v=>(OT_POS.find(p=>p.v===v)||OT_POS[1]).t;
const OTC=new Map();
function otRun(v){
  if(OTC.has(v)) return OTC.get(v);
  const tok=otTok(v), s={...BASE,change:'edit',at:tok-1}, st=run(s), r1=st.res[0], r2=st.res[1];
  const K1=keysOf(idsOf(r1.req.segs),r1.nb), K2=keysOf(idsOf(r2.req.segs),r2.nb);
  const blk=Math.floor((tok-1)/BLOCK), same=r2.state.filter(x=>x===2).length, s0=r2.state.indexOf(2);
  const nw=r2.nb-r2.hitBlocks, ansB=(r2.cached-r1.cached)-nw, preB=Math.floor(BASE.P/BLOCK), dup=preB-r2.hitBlocks;
  /* dup: the system prompt's blocks from the changed one on, which the cache now holds twice (under request 1's keys and request 2's);
     recSys: the system prompt's tokens that request 2 recomputes (the question's are new at every position, so they are left out) */
  const out={v,tok,blk,s,r1,r2,K1,K2,same,s0,nw,bytes:nw*BLOCK*KV_TOK,ansB,dup,dupBytes:dup*BLOCK*KV_TOK,recSys:BASE.P-r2.hit,preB};
  OTC.set(v,out); return out;}
/* the three counters after the first k blocks of request 2 are looked up: the system prompt's tokens recomputed so far, the step model's
   time for the blocks so far (cached ones only read their keys and values, each recomputed one adds its arithmetic), and the system
   prompt's blocks stored a second time so far. recAll and nwAll count the question too (the engine's whole-prompt view) */
function otAt(v,k){
  const o=otRun(v), r=o.r2, hb=r.hitBlocks; k=Math.max(0,Math.min(r.total,k));
  const Nk=Math.min(r.N,k*BLOCK), Hk=Math.min(k,hb)*BLOCK, dup=Math.max(0,Math.min(k,o.preB)-hb);
  return {k,rec:Math.max(0,Math.min(k*BLOCK,BASE.P)-Hk),t:k?ttft(Nk,Hk):null,nw:dup,bytes:dup*BLOCK*KV_TOK,recAll:Nk-Hk,nwAll:Math.max(0,Math.min(k,r.nb)-hb)};}
/* a block's key, short: its last hex digits, more of them when two different keys would look the same */
function keyPair(a,b){let n=4; const sh=(k,m)=>k.toString(16).padStart(m,'0').slice(-m); while(n<13&&a!==b&&sh(a,n)===sh(b,n)) n++; return [sh(a,n),sh(b,n)];}

/* the challenge: six parts, one order for every request, 8 requests from 3 users over 3 days */
const CH={parts:['date','name','sys','shots','hist','msg'],len:{sys:3000,shots:1000,date:DATE_T,name:NAME_T,msg:150},
  pre:{Alice:1500,Bob:800,Carol:2500},stream:[[1,'Alice'],[1,'Bob'],[1,'Carol'],[2,'Alice'],[2,'Carol'],[2,'Bob'],[3,'Bob'],[3,'Alice']],goal:4400};
const CHC=new Map();
function chRun(order){
  const key=order.join(','); if(CHC.has(key)) return CHC.get(key);
  const c=newCache(Math.floor(POOL_T/BLOCK)), hist={}, res=[];
  Object.keys(CH.pre).forEach(u=>{hist[u]=[{k:'hist',cid:'hist0-'+u,n:CH.pre[u]}];});
  CH.stream.forEach(([day,u],i)=>{
    const part={sys:[{k:'sys',cid:'ch-sys',n:CH.len.sys}],shots:[{k:'shots',cid:'ch-shots',n:CH.len.shots}],date:[{k:'date',cid:'ch-date-'+day,n:DATE_T}],
      name:[{k:'name',cid:'ch-name-'+u,n:NAME_T}],hist:hist[u],msg:[{k:'msg',cid:'ch-msg-'+i,n:CH.len.msg}]};
    const segs=[].concat(...order.map(p=>part[p].map(x=>({...x,part:p})))), ans={k:'ans',cid:'ch-ans-'+i,n:ANS};
    const r=runOne(c,{segs,ans}); res.push({...r,req:{i,day,user:u,segs,ans},w:ttft(r.N,r.hit),wo:ttft(r.N,0)});
    hist[u]=hist[u].concat([{k:'hist',cid:'ch-msg-'+i,n:CH.len.msg},{k:'hist',cid:'ch-ans-'+i,n:ANS}]);});
  const n=res.length, avg=res.reduce((a,r)=>a+r.hit,0)/n;
  const out={order:order.slice(),res,avg,avgN:res.reduce((a,r)=>a+r.N,0)/n,avgW:res.reduce((a,r)=>a+r.w.t,0)/n,avgWo:res.reduce((a,r)=>a+r.wo.t,0)/n,ok:avg>=CH.goal,moves:movesOf(order)};
  CHC.set(key,out); return out;}
/* parts moved from the initial order: 6 minus the longest run of parts that kept their relative order */
function movesOf(order){const pos=order.map(p=>CH.parts.indexOf(p)), lis=[];
  pos.forEach(x=>{let lo=0,hi=lis.length; while(lo<hi){const m=(lo+hi)>>1; if(lis[m]<x) lo=m+1; else hi=m;} lis[lo]=x;});
  return order.length-lis.length;}
function perms(a){if(a.length<=1) return [a.slice()]; const out=[]; a.forEach((x,i)=>{perms(a.slice(0,i).concat(a.slice(i+1))).forEach(p=>out.push([x].concat(p)));}); return out;}
let CH_ALL=null;
const chAll=()=>CH_ALL||(CH_ALL=perms(CH.parts).map(chRun));
const chBestAvg=()=>Math.max(...chAll().map(r=>r.avg));
const chMinMoves=()=>Math.min(...chAll().filter(r=>r.ok).map(r=>r.moves));
/* the orders the result chart compares, besides the learner's own */
const CH_SHOW=[['date','name','sys','shots','hist','msg'],['name','sys','shots','hist','date','msg'],['date','sys','shots','name','hist','msg'],
  ['sys','shots','name','hist','date','msg'],['sys','shots','name','hist','msg','date']];
/* the reveal's side-by-side (GPT-6 review of v9, #2): what Alice's day-1 request leaves in the cache (its prompt, then its answer)
   against her day-2 prompt, part by part, for three orders: A the best (the date after the new message), B the date before the new
   message, C the name before the new message. Both sides match while they hold the same part at the same place; the first place they
   differ ends the hit (the engine's hit, res[3].hit, is that point rounded down to whole blocks). Day 2's history is day 1's history
   followed by day 1's message and answer only, so whatever sat between the history and the message on day 1 is missing from it */
const AL_ORDERS=[['sys','shots','name','hist','msg','date'],['sys','shots','name','hist','date','msg'],['sys','shots','hist','name','msg','date']];
function alData(order){
  const r=chRun(order), d1=r.res[0], d2=r.res[3];
  const a=d1.req.segs.map(s=>({cid:s.cid,n:s.n,part:s.part})).concat([{cid:d1.req.ans.cid,n:d1.req.ans.n,part:'ans'}]);
  const b=d2.req.segs.map(s=>({cid:s.cid,n:s.n,part:s.part}));
  let m=0; while(m<a.length&&m<b.length&&a[m].cid===b[m].cid) m++;
  const inA=new Set(a.map(p=>p.cid));
  a.forEach((p,i)=>{p.st=i<m?'hit':'held';});
  b.forEach((p,i)=>{p.st=i<m?'hit':(inA.has(p.cid)?'same':'new');});
  return {order:order.slice(),a,b,m,match:a.slice(0,m).reduce((s,p)=>s+p.n,0),hit:d2.hit,N:d2.N,avg:r.avg,ok:r.ok};}

/* =====================================================================
   formatting (every derived number is computed from the values as shown, R6.6)
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const NS=' ';
/* a time to first token: two decimals below 10 ms (8.13 ms), one below 1 s (342.9 ms), then seconds (8.15 s) */
const fT=sec=>{const v=sec*1e3; return v<9.995?fixd(v,2)+NS+'ms':(v<999.95?fixd(v,1)+NS+'ms':fixd(v/1e3,2)+NS+'s');};
const tMs=sec=>{const v=sec*1e3; return v<9.995?Math.round(v*100)/100:(v<999.95?Math.round(v*10)/10:Math.round(v/10)*10);};   // the value shown, in ms
const ratioS=q=>q>=9.95?fint(q):fixd(q,1);
const ratioOf=(a,b)=>tMs(a)/tMs(b);                              // "42 times sooner": from the values as shown
const gib2=b=>fixd(b/A.GiB,2)+NS+'GiB';
const gib1=b=>fixd(b/A.GiB,1)+NS+'GiB';
/* numbers in prose (R6.20): rounded for reading, about two significant digits (340 ms, 8 ms, 8.1 s); exact values stay on the stage,
   the tooltips and Show the math */
const fR=sec=>{const v=sec*1e3; if(v>=9950) return fint(Math.round(v/1e3))+NS+'s'; if(v>=950) return fixd(v/1e3,1)+NS+'s'; if(v>=99.5) return fint(Math.round(v/10)*10)+NS+'ms'; if(v>=9.95) return fint(Math.round(v))+NS+'ms'; return fint(Math.round(v))+NS+'ms';};
/* a share in words */
const fracW=q=>q<0.15?tx('a small fraction','一小部分'):(q<0.3?tx('about a quarter','约四分之一'):(q<0.42?tx('about a third','约三分之一'):(q<0.6?tx('about half','约一半'):(q<0.85?tx('most','大部分'):tx('almost all','几乎全部')))));
const gbS=b=>{const v=b/1e9; return (v>=9.995?fixd(v,1):fixd(v,2))+NS+'GB';};
const tokN=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);
const reqN=n=>tx(n===1?'1 request':`${fint(n)} requests`,`${fint(n)} 个请求`);
const bigT=n=>n>=1e6?A.fmt.big(n):fint(n);                       // R6.18: token totals over 6 digits in M
const bigTh=n=>n>=1e6&&A.ui.big?A.ui.big(n):fint(n);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, sent=false, sel=0, observed={}, root=null, app=null;
let saved=null, chOrder=CH.parts.slice(), chSubmitted=false, chReveal=false, drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false, sbTouched=false;
let lever='none', memLever='pool', sweepNext=false;
let seqN=null, seqT=0;     // part 6: how many requests have arrived so far while they arrive one at a time (null: all of them)
let seqSeen=null, seqCur=null;   // part 6 (R3.20): the rows last shown in full, and those of the run in progress (seqKeys())
const seqDone=new Set();         // part 6 (R3.20): the arrivals watched to the end, by their rows (seqKeys().join('|')), whatever the part or the option
const seqDoneT=new Set();        // the same arrivals by their settings (token()): what a reload keeps, to rebuild seqDone from
/* part 5: the position picked, the prediction, the positions whose end state has been shown; the sweep over request 2's blocks:
   swK blocks looked up so far (null: the end state), swStep while the learner goes block by block, swRun the run token every frame and
   timeout checks, otG the blocks per cell of the two strips, otRail whether the last pick came from the card */
let otPos=null, otPred=null, swK=null, swStep=false, swRun=0, swRaf=0, otG=1, otRail=false;
const otTried=new Set();
const otDone=new Set();          // part 5 (R3.20): the positions whose lookup was watched to the end (played, or block by block to the last stop)
const leverTried=new Set(), memTried=new Set();
const shownPrev={};
const R=()=>run(S);
/* before request 2 is sent (guided parts 1 to 3), every request after the first is hidden: part 3 answers the prediction with it (R3.8) */
const masked=()=>mode==='guided'&&!sent&&step<=2;
const selIdx=()=>{if(mode==='challenge') return Math.min(sel,CH.stream.length-1); const n=R().res.length; return masked()?0:Math.max(0,Math.min(sel,n-1));};
const selRes=()=>mode==='challenge'?chRun(chOrder).res[selIdx()]:R().res[selIdx()];
/* a request's name: a turn in a chat, the day and the user in the challenge, else its number (reqName starts a sentence, reqOf goes inside one) */
const chatMode=()=>mode!=='challenge'&&S.kind==='chat';
function reqName(i){
  if(mode==='challenge'){const q=CH.stream[i]; return tx(`Day ${q[0]}, ${q[1]}`,`第 ${q[0]} 天 ${q[1]}`);}
  return chatMode()?tx(`Turn ${i+1}`,`第 ${i+1} 轮`):tx(`Request ${i+1}`,`请求 ${i+1}`);}
function reqOf(i){
  if(mode==='challenge'){const q=CH.stream[i]; return tx(`${q[1]}’s request on day ${q[0]}`,`第 ${q[0]} 天 ${q[1]} 的请求`);}
  return chatMode()?tx(`turn ${i+1}`,`第 ${i+1} 轮`):tx(`request ${i+1}`,`请求 ${i+1}`);}
const andJoin=a=>A.lang()==='zh'?a.join('、'):(a.length<2?a.join(''):a.slice(0,-1).join(', ')+' and '+a[a.length-1]);

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 pc:()=>tx('Prefix caching keeps the KV cache of prompts the server has already processed, in blocks of 16 tokens, so that a new request that starts with exactly the same tokens reads those keys and values from GPU memory instead of running prefill on them again.',
           'prefix caching 把服务端已经处理过的 prompt 的 KV cache 按 16 个 token 一个 block 留在显存里；新请求开头的 token 和它完全一样时，直接从显存读这些 key 和 value，不再为它们做 prefill。'),
 prefill:()=>tx('Prefill is the first step of answering a request: the model takes in the whole prompt in one pass, writes every prompt token’s keys and values to the KV cache, and produces the first token of the answer.',
                'prefill 是回答一个请求的第一步：模型一次读入整个 prompt，把每个 prompt token 的 key 和 value 写进 KV cache，并产出回答的第一个 token。'),
 ttft:()=>tx('Time to first token (TTFT): how long a user waits from sending the prompt to seeing the first token of the answer. This lab has no queue, so it is the prefill’s time.',
             '首 token 延迟（TTFT，time to first token）：从发出 prompt 到看到回答的第一个 token 要等多久。这个 Lab 里没有排队，所以它就是 prefill 的耗时。'),
 block:()=>tx('A block holds the keys and values of 16 consecutive tokens, vLLM’s default block size; a request’s KV cache is a list of such blocks (the lab PagedAttention shows why).',
              'block 存放连续 16 个 token 的 key 和 value，这是 vLLM 默认的 block 大小；一个请求的 KV cache 就是一串这样的 block（为什么这样分，见 PagedAttention 那个 Lab）。'),
 key:()=>tx('A block’s key is a hash of its 16 token IDs and of the previous block’s key, so it stands for every token from the start of the prompt to the end of the block. Two blocks get the same key only when all of those tokens are the same.',
            'block 的 key 是它 16 个 token 的编号和前一个 block 的 key 一起算出的哈希值，所以它代表从 prompt 开头到这个 block 末尾的全部 token。只有这些 token 全都相同，两个 block 的 key 才相同。'),
 hit:()=>tx('A cache hit: the request’s first blocks are found in the cache, so their keys and values are read, not computed. The hit ends at the first block that is not found (a miss); prefill computes that block and every block after it.',
            '缓存命中：请求开头的 block 在缓存里找到了，它们的 key 和 value 直接读取，不用计算。命中到第一个找不到的 block（未命中）为止；prefill 要计算这个 block 和它之后的所有 block。'),
 lru:()=>tx('LRU (least recently used): when the cached blocks need more memory than they may use, the server evicts the blocks used longest ago first, and among one request’s blocks its last ones first, as vLLM does.',
            'LRU（least recently used，最近最少使用）：缓存的 block 超出能用的显存时，服务端先淘汰最久没用过的 block；同一个请求的 block 里，先淘汰排在最后的，vLLM 就是这样做的。'),
 same:()=>tx('Hatched: this block’s 16 tokens are already in the cache, but after a different prefix, so its key differs and prefill computes it again. Its keys and values really are different: its tokens attended to the changed tokens before them.',
             '斜纹：这个 block 的 16 个 token 缓存里已经有了，但前面接的内容不同，所以 key 不同，prefill 要重算它。它的 key 和 value 本来也不一样：这些 token 对前面变了的 token 做过 attention。'),
 overlap:()=>tx('A GPU reads memory and does arithmetic at the same time, so a step takes about as long as the slower of the two, not their sum. This is this lab’s estimate; real kernels overlap them only in part.',
                'GPU 读写显存和做计算是同时进行的，所以一步的耗时约等于两者中较慢的那个，而不是两者相加。这是这个 Lab 的估计；真实 kernel 只能让两者部分重叠。'),
 wread:()=>tx(`In every step the GPU streams every parameter it multiplies (${gbS(W_READ)} for ${MODEL}, all but the embedding table) from GPU memory once. That one read serves every token the step computes, so it costs the same for 1 token or 10,000.`,
              `每一步，GPU 都要把参与乘法的全部参数（${MODEL} 是 ${gbS(W_READ)}，embedding 表以外的全部参数）从显存读一遍。这一次读，供这一步算的所有 token 共用：算 1 个 token 和算 10,000 个读的一样多。`),
 gb:()=>tx('GB and TB here are 10^9 and 10^12 bytes, to match the bandwidth in TB/s.','这里的 GB、TB 是 10^9、10^12 字节，和带宽的 TB/s 一致。'),
 gib:()=>kt('gib'), kib:()=>kt('kib'), kv:()=>kt('kvcache'), mask:()=>kt('mask'), token:()=>kt('token')
};
/* the kit's causal mask, plus what it means for this lab (R2.8: the kit's wording first, one sentence of our own) */
const MASK_PC=()=>TIP.mask()+tx(' So a token’s key and value depend only on it and the tokens before it: the same first 10,000 tokens give the same keys and values, whatever follows them.',
                                 '所以一个 token 的 key 和 value 只取决于它自己和它前面的 token：前 10,000 个 token 相同，它们的 key 和 value 就相同，不管后面接什么。');

/* =====================================================================
   names of the segments (grayscale on the stage) and of the prefix in each scenario
   ===================================================================== */
const PRESETS=[
 {id:'support',t:{en:'Support chatbot',zh:'客服机器人'},d:{en:'one 10,000-token system prompt, 5 users’ 200-token questions',zh:'一段 10,000 个 token 的 system prompt，5 个用户各问一个 200 个 token 的问题'},pre:{en:'system prompt',zh:'system prompt'},own:{en:'question',zh:'问题'},s:{P:10000,O:200,n:5,kind:'same',change:'none',apps:1,mem:'pool'}},
 {id:'chat',t:{en:'Multi-turn chat',zh:'多轮对话'},d:{en:'a 2,000-token system prompt, 5 turns of 100-token messages and 300-token answers',zh:'2,000 个 token 的 system prompt，5 轮对话，每轮消息 100 个 token、回答 300 个 token'},pre:{en:'system prompt',zh:'system prompt'},own:{en:'new message',zh:'新消息'},s:{P:2000,O:100,n:5,kind:'chat',change:'none',apps:1,mem:'pool'}},
 {id:'doc',t:{en:'Questions about one long document',zh:'针对同一篇长文档提问'},d:{en:'a 100,000-token document, 4 questions of 200 tokens',zh:'100,000 个 token 的文档，4 个问题，每个 200 个 token'},pre:{en:'document',zh:'文档'},own:{en:'question',zh:'问题'},s:{P:100000,O:200,n:4,kind:'same',change:'none',apps:1,mem:'pool'}},
 {id:'fewshot',t:{en:'Few-shot classification',zh:'few-shot 分类'},d:{en:'2,000 tokens of examples, 5 short texts of 200 tokens to label',zh:'2,000 个 token 的示例，5 段 200 个 token 的短文本等着分类'},pre:{en:'examples',zh:'示例'},own:{en:'text to label',zh:'待分类文本'},s:{P:2000,O:200,n:5,kind:'same',change:'none',apps:1,mem:'pool'}}
];
const sameS=(a,b)=>['P','O','n','kind','change','apps','mem'].every(k=>a[k]===b[k]);
const presetNow=()=>PRESETS.findIndex(p=>sameS(p.s,S));
/* a segment's name on the stage: the guided tour's system prompt and question, a scenario's own words, or the challenge's parts */
function segName(k){
  const p=mode==='sandbox'?PRESETS[presetNow()]:null;
  if(k==='pre') return mode==='guided'||S.kind==='chat'?tx('system prompt','system prompt'):(p?L(p.pre):tx('shared prefix','共享前缀'));
  if(k==='own') return mode==='guided'?tx('question','问题'):(p?L(p.own):tx('own tokens','自己的 token'));
  return {date:tx('today’s date','今天的日期'),name:tx('user’s name','用户名'),hist:tx('chat history','对话历史'),msg:tx('new message','新消息'),ans:tx('answer','回答'),
    sys:tx('system instructions','系统指令'),shots:tx('few-shot examples','few-shot 示例')}[k]||k;}
const CH_NAME={sys:{en:'System instructions',zh:'系统指令'},shots:{en:'Few-shot examples',zh:'few-shot 示例'},date:{en:'Today’s date',zh:'今天的日期'},name:{en:'The user’s name',zh:'用户名'},hist:{en:'The chat history',zh:'对话历史'},msg:{en:'The new message',zh:'新消息'}};
const CH_DESC={sys:()=>tx('3,000 tokens, the same for everyone','3,000 个 token，所有人都一样'),shots:()=>tx('1,000 tokens, the same for everyone','1,000 个 token，所有人都一样'),
  date:()=>tx('12 tokens, changes every day','12 个 token，每天都变'),name:()=>tx('8 tokens, one per user','8 个 token，每个用户一个'),
  hist:()=>tx('the user’s earlier turns: 800 to 2,500 tokens at first, and each request adds its message and its 300-token answer','这个用户之前的对话：开始时 800 到 2,500 个 token，每个请求再加上它的消息和 300 个 token 的回答'),
  msg:()=>tx('150 tokens, new in every request','150 个 token，每个请求都不一样')};
const USERS={Alice:{en:'Alice',zh:'Alice'},Bob:{en:'Bob',zh:'Bob'},Carol:{en:'Carol',zh:'Carol'}};
const orderTxt=o=>o.map(p=>L(CH_NAME[p]).replace(/^The /,'the ')).join(tx(', ','、'));

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'same',t:{en:'About the same (300 ms or more)',zh:'差不多一样（300 ms 或更久）'}},
  {v:'half',t:{en:'Less, but still long (20 to 300 ms)',zh:'少一些，但还是很久（20 到 300 ms）'}},
  {v:'few',t:{en:'Only a few milliseconds (under 20 ms)',zh:'只要几毫秒（不到 20 ms）'}},
  {v:'zero',t:{en:'No wait at all (0 ms)',zh:'完全不用等（0 ms）'}}];
const PRED_RIGHT='few';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const MISSION=()=>run({...BASE});                               // request 1 and request 2 of the mission
const LEVERS=[
 {v:'none',t:{en:'Initial setup: only the question differs',zh:'初始设置：只有问题不同'},d:{en:'request 2 is the same system prompt plus its own question',zh:'请求 2 是同样的 system prompt，加上它自己的问题'}},
 {v:'dateStart',t:{en:'Today’s date at the very start',zh:'今天的日期放在最前面'},d:{en:'a 12-token date line before the system prompt; request 2 comes a day later',zh:'system prompt 前面加一行 12 个 token 的日期；请求 2 晚一天到'}},
 {v:'dateAfter',t:{en:'Today’s date after the system prompt',zh:'今天的日期放在 system prompt 后面'},d:{en:'the same date line, between the system prompt and the question',zh:'同样一行日期，放在 system prompt 和问题之间'}},
 {v:'edit',t:{en:'One word fixed in the middle of the system prompt',zh:'修改 system prompt 中间的一个词'},d:{en:'token 5,000 of the 10,000 changes after request 1',zh:'请求 1 之后，10,000 个 token 里的第 5,000 个变了'}},
 {v:'nameStart',t:{en:'The user’s name at the very start',zh:'用户名放在最前面'},d:{en:'an 8-token greeting with the user’s name before the system prompt',zh:'system prompt 前面加一句 8 个 token、带用户名的问候'}}];
const MEM_LEVERS=[
 {v:'pool',t:{en:'Initial setup: the whole KV cache pool',zh:'初始设置：整个 KV cache 显存池'}},
 {v:2,t:{en:'Room for only 2 requests',zh:'只够放 2 个请求'}},
 {v:1,t:{en:'Room for only 1 request',zh:'只够放 1 个请求'}}];
const memDesc=(v,s)=>{const P=s.P, per=P+s.O+ANS;
  if(v==='pool') return tx(`${gib1(POOL_B)}: all that one H100 has left for the KV cache, room for dozens of such prompts`,`${gib1(POOL_B)}：一张 H100 留给 KV cache 的全部显存，放得下几十段这样的 prompt`);
  return tx(`${gib2(v*per*KV_TOK)}: the KV cache of ${v} request${v>1?'s':''} (prompt and answer), so at most ${v===1?'one app’s':`${v} apps’`} system prompt${v>1?'s':''} can stay cached`,`${gib2(v*per*KV_TOK)}：只放得下 ${v} 个请求的 KV cache（prompt 加回答），同一时间最多留住 ${v} 个应用的 system prompt`);};
/* part 5: where request 2's one changed token sits, and the prediction (the answer stays hidden until a pick, R3.8). The prediction,
   the counters and the result all count the system prompt only: request 2's own question is new, so it is computed at every position */
const OT_T={front:{en:'At the front',zh:'在开头'},middle:{en:'In the middle',zh:'在中间'},end:{en:'At the end',zh:'在末尾'}};
const OT_W={front:{en:'at the front',zh:'开头'},middle:{en:'in the middle',zh:'中间'},end:{en:'at the end',zh:'末尾'}};
const OT_PRED=[{v:'block',t:{en:'All of it except the changed token’s block: only that block’s 16 tokens are recomputed',zh:'除了改动所在的 block 都能找到：只重算这个 block 的 16 个 token'}},
  {v:'before',t:{en:'Every block before the changed one; that block and every block after it are recomputed',zh:'改动之前的每个 block 都能找到，改动所在的 block 和之后的每个 block 都要重算'}},
  {v:'none',t:{en:'None of it: one change anywhere means that the whole system prompt is recomputed',zh:'一个也找不到：不管改在哪里，整段 system prompt 都要重算'}}];
const OT_RIGHT='before';
const otPredT=v=>L(OT_PRED.find(o=>o.v===v).t);
/* how many of the system prompt's blocks each prediction says are recomputed: A the changed block alone, B that block and every block
   after it, C all of them. At the front B and C say the same thing, and at the end A and B do, so there both count as right (review r1 #3, #11) */
const otSays=(v,o)=>v==='block'?1:(v==='none'?o.preB:o.preB-o.r2.hitBlocks);
const otRight=(v,o)=>otSays(v,o)===o.dup;
const OT_LOCK=()=>tx(`In this experiment you can change only where request 2’s one different token sits, in the card or under request 2 in the chart. Everything else stays fixed: ${MODEL} on one ${GPUN}, the mission’s 10,000-token system prompt, the 200-token questions and the 2 requests.`,
                     `这个实验里只能改请求 2 那个不同的 token 放在哪，在卡片里或图中请求 2 下面选。其余设置固定不变：一张 ${GPUN} 跑 ${MODEL}，任务里 10,000 个 token 的 system prompt、200 个 token 的问题和 2 个请求。`);
const isOT=()=>mode==='guided'&&!!STEPS[step]&&STEPS[step].id==='onetoken';
const otOpen=()=>isOT()&&!!otPos&&!!otPred;                        // request 2's blocks are shown only after the prediction
const otS=()=>otPos?{...BASE,change:'edit',at:otTok(otPos)-1}:{...BASE};
const otName=v=>tx(`${L(OT_T[v])}: token ${fint(otTok(v))}`,`${L(OT_T[v])}：第 ${fint(otTok(v))} 个 token`);
const otBlkT=v=>{const b=Math.floor((otTok(v)-1)/BLOCK)+1, last=Math.floor(BASE.P/BLOCK);
  return b===1?tx('It sits in block 1, the very first.','它在第 1 个 block 里，也就是最前面。')
    :(b===last?tx('It sits right before the question.','它紧挨在问题前面。')
      :tx(`It sits in block ${fint(b)}, halfway through the system prompt.`,`它在第 ${fint(b)} 个 block 里，正好在 system prompt 的中间。`));};
/* a position's result on its title row once its end state has been shown (R3.6): only the conclusion, how much of the system prompt
   request 2 still finds in the cache (R3.17); the times and the memory are on the stage */
const otConcl=v=>{const r=otRun(v).r2, q=r.hit/BASE.P; return r.hit===0?tx('none of it cached','一点也没命中'):tx(`${fracW(q)} cached`,`命中${fracW(q)}`);};
const otLvRes=v=>otPred&&otTried.has(v)?otConcl(v):'';
const otLevers=()=>`<div class="px-otlv">`+A.ui.levers({aria:tx('Where request 2’s changed token sits, pick one','请求 2 改动的 token 在哪，选一个'),value:otPos||'',
  options:OT_POS.map(p=>({v:p.v,title:otName(p.v),desc:otBlkT(p.v),result:otLvRes(p.v)}))})+`</div>`;
/* the question names the position picked, and says that the question's own tokens are not counted */
const otQ=()=>tx(`Request 2 changes token ${fint(otTok(otPos))}, ${L(OT_W[otPos])}. How much of its system prompt does it still find in the cache?`,`请求 2 改的是${L(OT_W[otPos])}的第 ${fint(otTok(otPos))} 个 token。它还能在缓存里找到自己的 system prompt 的多少？`);
const OT_QNOTE=()=>tx('Request 2’s own question is new, so request 2 computes it at every position; the choices and the counters leave it out.','请求 2 自己的问题是新的，不管改在哪里都要现算，所以选项和计数都不算它。');
const otChoices=()=>`<div class="choices px-otch" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${OT_PRED.map(o=>`<button type="button" class="choice" data-otpred="${o.v}" aria-pressed="false">${esc(L(o.t))}</button>`).join('')}</div>`;
const otQHTML=()=>`<p class="px-otq"><b>${esc(otQ())}</b></p><p class="small muted px-otqn">${esc(OT_QNOTE())}</p>`+otChoices();
function otBody(){
  const intro=tx(`<p>Request 1 has already run, so its system prompt is in the cache <span class="px-nw">(${J('pair','see the chart')})</span>. Request 2 sends the same system prompt with one token changed, followed by its own question. Pick where that token sits:</p>`,
                 `<p>请求 1 已经跑完，所以它的 system prompt 在缓存里（${J('pair','见图里的两个请求')}）。请求 2 发来同样的 system prompt，只改了其中一个 token，后面接着它自己的问题。选一个位置，决定这个改动的 token 放在哪：</p>`)+otLevers();
  if(!otPos||otPred) return intro;                                     // after the prediction the verdict is in the result box (review r1 #8)
  return intro+`<div class="px-otqc">${otQHTML()}</div>`;             // on a phone the same question sits in the chart, under the positions
}
/* the result, once request 2's blocks have all been looked up: what request 2 finds, how long its first token waits (one rounded number,
   shown on the stage), what the cache now holds twice; then the verdict; then why (R3.10, R6.20, R6.21) */
function otObs(){
  if(!otOpen()) return '';
  /* while another position's blocks are looked up, the box keeps its place and height (the card must not shrink under the sticky chart)
     but shows no result yet, only which position is being looked up (R3.8) */
  if(swK!=null&&!(swStep&&otTried.has(otPos))) return otTried.size?`<p class="px-otwait"><b>${esc(otName(otPos))}</b></p>`:'';
  const o=otRun(otPos), r=o.r2, m=MISSION().res[1], P=BASE.P;
  const found=r.hit===0?tx('At the front, the changed token sits in the very first block, so request 2 finds none of its system prompt in the cache.','在开头，改动的 token 就在第一个 block 里，所以请求 2 在缓存里找不到它的 system prompt 的任何部分。')
    :(o.dup===1?tx('At the end, request 2 finds every block of its system prompt in the cache except the last one.','在末尾，请求 2 在缓存里找到了它的 system prompt 除最后一个 block 之外的每个 block。')
      :tx(`In the middle, request 2 finds every block before the change in the cache, ${fracW(r.hit/P)} of its system prompt, but it misses the changed block and every block after it.`,`在中间，请求 2 在缓存里找到了改动之前的每个 block，约占它的 system prompt 的${fracW(r.hit/P).replace(/^约/,'')}；但改动所在的 block 和之后的每个 block 它都没找到。`));
  const wait=tMs(r.w.t)===tMs(r.wo.t)?tx(`Its first token waits about ${fR(r.w.t)}, which is as long as it would wait without the cache.`,`它的首 token 要等约 ${fR(r.w.t)}，和不用缓存时一样久。`)
    :(fR(r.w.t)===fR(m.w.t)?tx(`Its first token comes after about ${fR(r.w.t)}, almost as soon as with no change at all.`,`它的首 token 约 ${fR(r.w.t)} 后出现，和完全没有改动时差不多。`)
      :tx(`Its first token waits about ${fR(r.w.t)}.`,`它的首 token 要等约 ${fR(r.w.t)}。`));
  const twice=o.dup>=o.preB?tx('The cache now holds the whole system prompt twice.','现在缓存里有两份完整的 system prompt。')
    :(o.dup===1?tx('The cache now holds only the system prompt’s last block twice.','现在缓存里只有 system prompt 的最后一个 block 存了两份。')
      :tx('The cache now holds the blocks from the change on twice.','从改动起的这些 block，现在缓存里存了两份。'));
  const head=`<p><b>${tx(`${found} ${wait}`,`${found}${wait}`)}</b>${tx(' ','')}${twice}</p>`;
  const why=o.same>0?tx(`<p>Why: a block’s ${DEF('key',TIP.key())} is computed from its own tokens and every token before it, so from the changed block on, none of request 2’s keys is in the cache, not even for the ${DEF('hatched',TIP.same())} blocks that hold the same tokens. So the parts that change belong at the end, where they make the fewest blocks miss.</p>`,
                        `<p>原因：一个 block 的 ${DEF('key',TIP.key())} 由它自己的 token 和它之前的全部 token 一起算出，所以从改动所在的 block 起，请求 2 的 key 没有一个在缓存里，连装着同样 token 的${DEF('斜纹',TIP.same())} block 也不在。所以会变的内容应该放在最后，这样未命中的 block 最少。</p>`)
    :tx(`<p>Why: a block’s ${DEF('key',TIP.key())} is computed from its own tokens and every token before it, so the changed block misses, and so does every block after it; here that is only the question, which request 2 computes anyway. So the parts that change belong at the end, where they make the fewest blocks miss.</p>`,
        `<p>原因：一个 block 的 ${DEF('key',TIP.key())} 由它自己的 token 和它之前的全部 token 一起算出，所以改动所在的 block 未命中，它之后的每个 block 也未命中；这里它后面只有问题，而问题请求 2 本来就要算。所以会变的内容应该放在最后，这样未命中的 block 最少。</p>`);
  return head+otVerdict(o)+why;}
/* the verdict, one paragraph: right or wrong, with the right answer; at an edge position also why two of the choices give the same result
   there, and a link to the middle, which tells them apart (review r1 #3, #8, #11) */
function otVerdict(o){
  const sh=v=>esc(otPredT(v).split(/[:;：，]/)[0]);                   // a choice by its first words, as the learner read them
  const B=sh(OT_RIGHT), pt=sh(otPred), front=o.r2.hitBlocks===0, end=o.dup===1&&!front, edge=front||end;
  /* at an edge two choices give the same result (review r1 #11): the right one says which other choice agrees here, the other one of
     the pair says why it is right here and what the answer is in general; the middle position tells them apart */
  const why=front?tx('no block comes before the changed one','改动的 block 之前没有别的 block'):tx('no other block of the system prompt follows the changed one','改动的 block 后面没有别的 system prompt block 了');
  let v;
  if(otPred===OT_RIGHT) v=tx(`<span class="okc">Your prediction, “${pt}”, was right.</span>`,`<span class="okc">你的预测“${pt}”是对的。</span>`)+
    (edge?tx(` ${front?'At the front':'At the end'} ${why}, so “${sh(front?'none':'block')}” gives the same result here.`,`${front?'在开头':'在末尾'}，${why}，所以“${sh(front?'none':'block')}”在这里结果一样。`):'');
  else if(otRight(otPred,o)) v=tx(`<span class="okc">Your prediction, “${pt}”, gives the right result here, because ${why}.</span> In general the answer is “${B}”.`,`<span class="okc">你的预测“${pt}”在这里给出的结果是对的，因为${why}。</span>一般情况下，答案是“${B}”。`);
  else v=tx(`You predicted “${pt}”, but the right answer is “${B}”.`,`你预测的是“${pt}”，但正确答案是“${B}”。`);
  const go=edge?`${tx(' ','')}<button type="button" class="linkbtn" data-otpos="middle" data-otrail="1" data-tipx="${esc(tx('Click to change the token in the middle instead: token 5,000.','点击改成中间的第 5,000 个 token。'))}">${tx('The middle position tells the choices apart →','换到中间，才分得出这几个选项 →')}</button>`:'';
  return `<p>${v}${go}</p>`;}
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'tl',
  lead:()=>{const m=MISSION().res[0]; return tx(`A 10,000-token system prompt: ${fT(m.wo.t)} before the first token. A second user asks next.`,`10,000 个 token 的 system prompt：首 token 要等 ${fT(m.wo.t)}。接着第二个用户来提问。`);},
  enter:()=>({s:{...BASE},sel:0}),
  body:()=>{const m=MISSION().res[0];
   return tx(`<p>A support chatbot runs ${MODEL} on one ${GPUN}. Every request starts with the same 10,000-token system prompt (instructions and product manuals) and adds the user’s own 200-token question.</p><p>The first request finds nothing to reuse, so ${DEF('prefill',TIP.prefill())} computes all 10,200 tokens: the ${DEF('time to first token',TIP.ttft())} is ${fT(m.wo.t)} (${J('tl','the timeline in the chart')}). Now a second user sends a question with the same system prompt.</p>`,
             `<p>一个客服机器人在一张 ${GPUN} 上跑 ${MODEL}。每个请求都以同一段 10,000 个 token 的 system prompt（说明和产品手册）开头，再加上用户自己 200 个 token 的问题。</p><p>第一个请求没有可以复用的东西，所以 ${DEF('prefill',TIP.prefill())} 要算完全部 10,200 个 token：${DEF('首 token 延迟',TIP.ttft())}是 ${fT(m.wo.t)}（见${J('tl','图里的时间线')}）。现在第二个用户发来一个问题，system prompt 还是同一段。</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'tl',gate:()=>!!pred,skip:true,
  lead:()=>tx('How long will the second request wait for its first token?','第二个请求的首 token 要等多久？'),
  enter:()=>({s:{...BASE},sel:0}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>With ${DEF('prefix caching',TIP.pc())}, the server keeps the KV cache of the first request’s prompt. How long does the second request wait for its first token?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>开启 ${DEF('prefix caching',TIP.pc())} 后，服务端会留着第一个请求的 prompt 的 KV cache。第二个请求的首 token 要等多久？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'send',title:{en:'Same prefix, same KV cache',zh:'前缀相同，KV cache 就相同'},focus:'strip',
  lead:()=>tx('Send request 2: its first 10,000 tokens come from the cache.','发出请求 2：它的前 10,000 个 token 来自缓存。'),
  enter:()=>({s:{...BASE},sel:sent?1:0}),expect:()=>sent,showMe:true,label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The server cut request 1’s KV cache into ${DEF('blocks',TIP.block())} of 16 tokens and gave each a ${DEF('key',TIP.key())}. Request 2 looks up its own blocks’ keys, from the first block on.</p>`,
              `<p>服务端把请求 1 的 KV cache 切成 16 个 token 一个的 ${DEF('block',TIP.block())}，每个 block 有一个 ${DEF('key',TIP.key())}。请求 2 从第一个 block 起，逐个查找自己 block 的 key。</p>`)+
    `<div class="inline-ctl px-send" id="px-ic-send"><button type="button" class="btn sm" data-act="send"${sent?' disabled':''}>${sent?tx('Request 2 sent','请求 2 已发出'):tx('Send request 2','发出请求 2')}</button></div>`,
  obs:()=>obsSend()},
 {id:'break',title:{en:'What breaks a hit',zh:'什么会让缓存失效'},focus:'strip',
  lead:()=>tx('Only tokens that are the same from the very start count. Try each change and see how much of the cache it breaks.','只有从开头起完全相同的 token 才算数。试一试每种改动，看它让多少缓存失效。'),
  enter:()=>({s:{...BASE,change:lever},sel:1}),expect:()=>leverTried.size>0,showMe:{lever:'dateStart'},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>A ${DEF('hit',TIP.hit())} is the longest run of blocks, from the first, whose keys are cached.</p><p>Pick a change and see how much of the cache request 2 still hits. Each change starts from the initial setup: request 1, then request 2.</p>`,
              `<p>${DEF('命中',TIP.hit())}的是从第一个 block 起、key 都在缓存里的最长一串 block。</p><p>选一种改动，看看请求 2 还能命中多少缓存。每种改动都从初始设置出发，先发请求 1，再发请求 2。</p>`)+leversHTML(),
  obs:()=>obsBreak()},
 {id:'onetoken',title:{en:'Experiment: change one token',zh:'实验：只改一个 token'},focus:'pair',
  lead:()=>tx('Change one token of a 10,000-token system prompt: how much of the cache still helps?','10,000 个 token 的 system prompt 只改一个 token：缓存还能帮上多少？'),
  enter:()=>({s:otS(),sel:1}),gate:()=>!!otPred,expect:()=>otTried.size>0,label:{en:'Result',zh:'结果'},
  body:()=>otBody(),
  obs:()=>otObs()},
 {id:'cost',title:{en:'The cost: GPU memory',zh:'代价：显存'},focus:'rows',
  lead:()=>{const m=MISSION().res[1]; return tx(`Each cached 10,000-token prompt holds ${gib2(m.hit*KV_TOK)}; when memory runs short, old blocks are evicted.`,`每段缓存的 10,000 个 token 的 prompt 占 ${gib2(m.hit*KV_TOK)}；显存不够时，旧的 block 会被淘汰。`);},
  enter:()=>({s:{...BASE,n:8,apps:3,mem:memLever},sel:7}),expect:()=>memTried.size>0,showMe:{mem:2},label:{en:'Result',zh:'结果'},
  body:()=>{const m=MISSION().res[1], per=m.hit*KV_TOK, gt=tx(`10,000 × 128 KiB = ${gbS(per)} = ${fixd(per/A.GiB,2)} GiB`,`10,000 × 128 KiB = ${gbS(per)} = ${fixd(per/A.GiB,2)} GiB`);
   return tx(`<p>Cached blocks stay in GPU memory: each cached 10,000-token prompt takes ${DEF(`about ${fixd(per/A.GiB,1)} GiB`,gt)}.</p><p>Here three apps share one GPU: <b>A</b>, <b>B</b> and <b>C</b> (say a support bot, a writing assistant and a code assistant), each with its own 10,000-token system prompt. Eight requests arrive one after another, from apps A, A, B, A, C, A, B, A. When a new prompt does not fit, the server first evicts the ${DEF('least recently used',TIP.lru())} one.</p><p>How much GPU memory may the cache use? Pick one and see how many of the 8 requests hit.</p>`,
             `<p>缓存的 block 一直占着显存：每段缓存的 10,000 个 token 的 prompt 占${DEF(`约 ${fixd(per/A.GiB,1)} GiB`,gt)}。</p><p>这里有三个应用共用一张 GPU：<b>A</b>、<b>B</b>、<b>C</b>（比如客服机器人、写作助手、代码助手），每个应用都有自己的一段 10,000 个 token 的 system prompt。8 个请求依次到来，分别来自应用 A、A、B、A、C、A、B、A。放不下新的 prompt 时，服务端先淘汰${DEF('最久没用过的',TIP.lru())}那一段。</p><p>缓存最多能用多少显存？选一种，看看 8 个请求里有几个命中。</p>`)+memLeversHTML();},
  obs:()=>obsCost()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('Same prefix, same KV cache; only exact matches from the start count; the cache costs GPU memory.','前缀相同，KV cache 就相同；只有从开头起完全相同才算数；缓存要占显存。'),
  enter:()=>({s:{...BASE},sel:1}),
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
/* a lever's result on its title row, once tried: the second request's first token */
function leverRes(v){const r=run({...BASE,change:v}).res[1]; return fT(r.w.t);}
function leversHTML(){return A.ui.levers({aria:tx('Changes between request 1 and request 2, pick one','请求 1 和请求 2 之间的改动，选一种'),value:lever,
  options:LEVERS.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:(o.v==='none'||leverTried.has(o.v))?leverRes(o.v):''}))});}
function memRes(v){const r=run({...BASE,n:8,apps:3,mem:v}); return tx(`${r.tot.hits} of 8 ${r.tot.hits===1?'hits':'hit'}`,`8 个命中 ${r.tot.hits} 个`);}
function memLeversHTML(){return A.ui.levers({aria:tx('Memory for cached blocks, pick one','缓存能用的显存，选一种'),value:memLever,
  options:MEM_LEVERS.map(o=>({v:String(o.v),title:L(o.t),desc:memDesc(o.v,BASE),result:(o.v==='pool'||memTried.has(o.v))?memRes(o.v):''}))});}

/* part 3: request 2, the verdict on the prediction, then why (R3.10: result, verdict, reason) */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);}
function obsSend(){
  if(!sent) return '';
  const m=MISSION().res[1];
  return tx(`<p><b>Request 2: the whole system prompt comes from the cache, and prefill computes only its own question. The first token comes after about ${fR(m.w.t)} instead of about ${fR(m.wo.t)}.</b></p>`,
            `<p><b>请求 2：整段 system prompt 都来自缓存，prefill 只算它自己的问题。首 token 从约 ${fR(m.wo.t)} 降到约 ${fR(m.w.t)}。</b></p>`)+verdict()+
    tx(`<p>Why: under the ${DEF('causal mask',MASK_PC())} a token’s keys and values depend only on the tokens before it, so both requests have exactly the same keys and values for the system prompt. It is not free: the new question still looks back at the whole prompt, and the step still ${DEF('reads the weights',TIP.wread())} once.</p>`,
       `<p>原因：有${DEF('因果 mask',MASK_PC())}，一个 token 的 key 和 value 只取决于它前面的 token，所以两个请求 system prompt 部分的 key 和 value 完全一样。但也不是零成本：新的问题还要回看整段 prompt，这一步也还要${DEF('把权重读一遍',TIP.wread())}。</p>`);}
/* part 4: the learner's lever against the initial setup */
function obsBreak(){
  if(!leverTried.size&&lever==='none') return tx(`<p><b>Initial setup: request 2 finds its whole system prompt in the cache.</b> Pick a change to see what it breaks.</p>`,`<p><b>初始设置：请求 2 在缓存里找到了整段 system prompt。</b>选一种改动，看它破坏了什么。</p>`);
  const b=run({...BASE}).res[1], r=run({...BASE,change:lever}).res[1], nm=esc(L(LEVERS.find(o=>o.v===lever).t));
  if(lever==='none') return tx(`<p><b>${nm}: request 2 finds its whole system prompt in the cache, and its first token comes after about ${fR(r.w.t)}.</b></p>`,`<p><b>${nm}：请求 2 在缓存里找到了整段 system prompt，首 token 约 ${fR(r.w.t)} 后出现。</b></p>`);
  const head=r.hit===0?tx(`<p><b>${nm}: not one of request 2’s tokens comes from the cache, so its first token waits for the full prefill again, about ${fR(r.w.t)}.</b></p>`,
                          `<p><b>${nm}：请求 2 一个 token 也没从缓存里拿到，首 token 又要等完整的 prefill，约 ${fR(r.w.t)}。</b></p>`)
    :(r.hit>=b.hit?tx(`<p><b>${nm}: request 2 still finds its whole system prompt in the cache, and its first token comes after about ${fR(r.w.t)}.</b></p>`,
                      `<p><b>${nm}：请求 2 仍在缓存里找到整段 system prompt，首 token 约 ${fR(r.w.t)} 后出现。</b></p>`)
      :tx(`<p><b>${nm}: request 2 finds only ${fracW(r.hit/b.hit)} of the system prompt in the cache, so its first token waits about ${fR(r.w.t)}.</b></p>`,
          `<p><b>${nm}：请求 2 只在缓存里找到了 system prompt 的${fracW(r.hit/b.hit)}，首 token 要等约 ${fR(r.w.t)}。</b></p>`));
  let why='';
  if(lever==='dateStart') why=tx(`<p>Why: the date is in block 1, and every later key includes it. So the 10,000 system-prompt tokens are cached, but after yesterday’s date: none of their keys match (the ${DEF('hatched',TIP.same())} blocks).</p>`,
                                 `<p>原因：日期在第 1 个 block 里，之后每个 block 的 key 都包含它。所以 system prompt 的 10,000 个 token 虽然在缓存里，前面接的却是昨天的日期：一个 key 都对不上（${DEF('斜纹',TIP.same())}的 block）。</p>`);
  if(lever==='dateAfter') why=tx('<p>Why: everything before the date is unchanged, so the whole system prompt hits; only the date and the question are computed. Parts that change belong at the end.</p>',
                                 '<p>原因：日期之前的内容没变，所以整段 system prompt 都命中，只有日期和问题要现算。会变的部分应该放在最后。</p>');
  /* the one-word change: what happens, and a question; the reason is what part 5 shows block by block, so it is not given away here (review r1 #4) */
  if(lever==='edit') why=tx(`<p>Only one word of the 10,000 tokens changed, yet request 2 still has to compute about half of them again. Where does that miss start, and why does it not stop?</p>`,
                            `<p>10,000 个 token 里只改了一个词，请求 2 却还要重算其中约一半。这段未命中从哪里开始，为什么一直不停？</p>`)+
    `<p><button type="button" class="linkbtn" data-go="${SIX('onetoken')}" data-tipx="${esc(tx(`Click to go to part ${SIX('onetoken')+1}, “${L(STEPS[SIX('onetoken')].title)}”.`,`点击去第 ${SIX('onetoken')+1} 部分“${L(STEPS[SIX('onetoken')].title)}”。`))}">${tx('The next part looks at it block by block →','下一部分逐个 block 来看 →')}</button></p>`;
  if(lever==='nameStart') why=tx(`<p>Why: a different user’s name in block 1 changes every key after it, so no block of the system prompt matches: every user pays the full prefill.</p>`,
                                 `<p>原因：第 1 个 block 里换成了另一个用户的名字，之后每个 key 都跟着变，所以 system prompt 没有一个 block 能对上，每个用户都要付完整的 prefill。</p>`);
  return head+why;}
/* part 6: the eviction demo */
function obsCost(){
  if(seqN!=null) return '';                                     // the requests are still arriving: the result waits for the last (R3.8)
  const r=R(), b=run({...BASE,n:8,apps:3,mem:'pool'}), v=S.mem, nm=v==='pool'?tx('The whole KV cache pool','整个 KV cache 显存池'):tx(`Room for only ${v} request${v>1?'s':''}`,`只够放 ${v} 个请求`);
  const missed=r.res.map((x,i)=>x.hit===0&&r.res.slice(0,i).some(y=>y.req.app===x.req.app)?i+1:0).filter(Boolean);
  const head=v==='pool'?tx(`<p><b>${nm}: ${r.tot.hits} of the 8 requests ${r.tot.hits===1?'hits':'hit'} the cache.</b></p>`,`<p><b>${nm}：8 个请求里 ${r.tot.hits} 个命中缓存。</b></p>`)
    :tx(`<p><b>${nm}: only ${r.tot.hits} of the 8 requests ${r.tot.hits===1?'hits':'hit'} the cache (${b.tot.hits} with the whole pool).</b></p>`,`<p><b>${nm}：8 个请求里只有 ${r.tot.hits} 个命中缓存（整个显存池时是 ${b.tot.hits} 个）。</b></p>`);
  if(v==='pool'||!missed.length) return head+tx(`<p>Nothing is evicted: every time A or B comes back, its system prompt is still cached. C comes only once, so it never hits.${v==='pool'?' Shrink the memory to see eviction.':''}</p>`,`<p>没有淘汰：A 和 B 每次再来，它的 system prompt 都还在缓存里。C 只来了一次，所以从不命中。${v==='pool'?'把显存调小，看看淘汰。':''}</p>`);
  const lst=andJoin(missed.map(String)), one=missed.length===1;
  const aKept=!r.res.some((x,i)=>i>0&&x.req.app==='A'&&x.hit===0), hitsAt=andJoin(r.res.map((x,i)=>x.hit>0?String(i+1):'').filter(Boolean));
  const why=tx(`<p>Why: when ${one?'request':'requests'} ${lst} ${one?'comes':'come'} back, ${one?'its':'their'} system prompt has been evicted (it was the ${DEF('least recently used',TIP.lru())} when another app needed room), so ${one?'it pays':'each pays'} the full prefill again.</p>`,
               `<p>原因：请求 ${lst} 回来时，${one?'它':'它们'}的 system prompt 已被淘汰（别的应用需要空间时，${one?'它':'它们'}是${DEF('最久没用过的',TIP.lru())}），所以要重新做完整的 prefill。</p>`);
  return head+why+(aKept?tx(`<p>A, the busiest app, stays cached throughout: it is used again before it becomes the least recently used.</p>`,`<p>最忙的应用 A 一直留在缓存里：它总在变成最久没用过的之前又被用到。</p>`)
    :tx(`<p>With room this small, only request${hitsAt.includes(' ')?'s':''} ${hitsAt} ${hitsAt.includes(' ')?'hit':'hits'}: each new app evicts the one before it, even the busiest.</p>`,`<p>显存这么小，只有请求 ${hitsAt} 命中：每来一个别的应用，前一个就被淘汰，最忙的 A 也一样。</p>`));}
function remember(){
  const m=MISSION().res[1], per=m.hit*KV_TOK;
  return tx(`<ul><li><b>Same prefix, same KV cache.</b> Under the causal mask a token’s keys and values depend only on the tokens before it, so a later request with the same prefix reuses its cached keys and values and prefills only its own new tokens. Its first token comes much sooner, and the longer the shared prefix, the more it saves.</li><li><b>Exact match, block by block, from the start.</b> One changed token misses its block and every block after it: stable parts first, parts that change (a date, a name, the question) last.</li><li><b>The cost is GPU memory:</b> about ${fixd(per/A.GiB,1)} GiB per 10,000 cached tokens. When memory runs short, the least recently used blocks are evicted, and the next request with that prefix pays the full prefill again.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Prefix caching stores the answers.” It stores the prompt’s keys and values. A request that hits still computes its own question and writes its own answer.</p><p>✗ “A similar prompt hits the cache.” Only the same tokens from the start count. A few different tokens at the very start, such as today’s date, make every block after them miss, even when the whole system prompt is cached.</p><p>✗ “A cache hit makes the request free.” A request that hits still waits a few milliseconds for its first token (its own tokens, one read of the weights), and every decode step after it costs what it costs without the cache.</p></div>${A.shell.toScenarios()}`,
    `<ul><li><b>前缀相同，KV cache 就相同。</b>有因果 mask，一个 token 的 key 和 value 只取决于它前面的 token，所以后来的请求只要前缀相同，就能复用缓存里这段 key 和 value，只 prefill 自己新加的 token。它的首 token 来得快得多，相同的前缀越长，省得越多。</li><li><b>从开头起、逐个 block 完全一致才算。</b>一个 token 变了，它所在的 block 和之后的所有 block 都未命中：不变的部分放前面，会变的部分（日期、名字、问题）放最后。</li><li><b>代价是显存：</b>每缓存 10,000 个 token 约占 ${fixd(per/A.GiB,1)} GiB。显存不够时，最久没用过的 block 被淘汰，下一个带这段前缀的请求又要付完整的 prefill。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “prefix caching 存的是回答。”它存的是 prompt 的 key 和 value。命中的请求仍要算它自己的问题，自己写回答。</p><p>✗ “差不多的 prompt 也能命中。”只有从开头起完全相同的 token 才算。最前面差几个 token（比如今天的日期），后面的 block 就全都对不上，哪怕整段 system prompt 都在缓存里。</p><p>✗ “命中缓存，请求就免费了。”命中的请求，首 token 仍要等几毫秒（算它自己的 token，读一遍权重），之后每个 decode 步的耗时和不用缓存时一样。</p></div>${A.shell.toScenarios()}`);}

/* =====================================================================
   scenarios: real workloads, one plain phrase each (R2.1)
   ===================================================================== */
function scenHTML(i){
  const p=PRESETS[i], r=run(p.s), name=esc(L(p.t)), f=r.res[0], l=r.res[r.res.length-1];
  const chat=p.s.kind==='chat';
  const head=tx(`<p><b>${name}: the first ${chat?'turn':'request'} waits for the full prefill, about ${fR(f.w.t)}; the last one only about ${fR(l.w.t)}.</b></p>`,
                `<p><b>${name}：${chat?'第一轮':'第一个请求'}要等完整的 prefill，约 ${fR(f.w.t)}；${chat?'最后一轮':'最后一个'}只要约 ${fR(l.w.t)}。</b></p>`);
  let why='';
  if(p.id==='support') why=tx(`<p>Every request after the first reuses the system prompt and computes only its own question, so prefill computes ${fracW(r.tot.pre/r.tot.all)} of the tokens it would without the cache.</p>`,
                              `<p>第一个之后的每个请求都复用 system prompt，只算自己的问题，所以 prefill 要算的 token 只有不用缓存时的${fracW(r.tot.pre/r.tot.all)}。</p>`);
  if(p.id==='chat') why=tx('<p>Each turn resends the whole conversation, which is the next turn’s prefix, so each turn computes only its new message, plus the previous answer’s last block, which was not full.</p>',
                           '<p>每一轮都把整段对话重发一遍，它正是下一轮的前缀，所以每一轮只需要算新消息，再加上上一轮回答没写满的最后一个 block。</p>');
  if(p.id==='doc') why=tx(`<p>Later questions are much faster, but slower than the chatbot’s: each new question looks back at the whole document, so arithmetic sets the time. The price is memory: the cached document holds about ${fint(Math.round(l.hit*KV_TOK/A.GiB))} GiB for as long as it stays cached.</p>`,
                          `<p>之后的问题快得多，但比客服机器人的慢：每个新问题都要回看整篇文档，所以计算决定耗时。代价是显存：缓存的文档只要还在，就占着约 ${fint(Math.round(l.hit*KV_TOK/A.GiB))} GiB。</p>`);
  if(p.id==='fewshot') why=tx('<p>The examples are short, so the saving is smaller. With only the new text to compute, the step is set by reading the weights once, as a decode step is.</p>',
                              '<p>示例不长，所以省得少一些。只算新的那段文本时，这一步的耗时取决于把权重读一遍，和一个 decode 步一样。</p>');
  return head+why;}

/* =====================================================================
   challenge
   ===================================================================== */
/* the reveal's side-by-side: for each order of AL_ORDERS, day 1 (in the cache) and day 2 (the new prompt) as two columns of parts on
   shared rows; amber where both hold the same part at the same place, a thick line where they first differ, then day 1's parts dashed
   (still cached, no longer a match) and day 2's hatched (cached after other parts) or gray (new); a bracket groups day 2's history */
const AL_KEY=['A','B','C'];
function alName(p){const c=p.cid, i=+c.split('-').pop(), k=n=>n===1?'①':'②';
  if(c==='ch-sys') return tx('System','系统指令'); if(c==='ch-shots') return tx('Examples','示例');
  if(c.startsWith('ch-name-')) return tx('Name','用户名'); if(c.startsWith('hist0-')) return tx('History','历史');
  if(c.startsWith('ch-date-')) return tx(`Date ${k(i)}`,`日期${k(i)}`);
  const d=CH.stream[i][0]; if(c.startsWith('ch-msg-')) return tx(`Message ${k(d)}`,`消息${k(d)}`); return tx(`Answer ${k(d)}`,`回答${k(d)}`);}
function alTip(o,side,i){
  const p=(side===1?o.a:o.b)[i], nm=alName(p), n=tx(`${nm}, ${tokN(p.n)}`,`${nm}，${tokN(p.n)}`);
  if(p.st==='hit'){const r=o.match-o.hit, last=i===o.m-1&&r>0;
    return n+tx(': day 2 has it at the same place, after the same parts, so day 2 reads it from the cache.','：第 2 天同一个位置也是它，前面的部分也都一样，所以第 2 天从缓存读。')
      +(last?tx(` Its last ${tokN(r)} do not fill a whole 16-token block, so they are computed again.`,`它最后 ${fint(r)} 个 token 凑不满一个 16 个 token 的 block，所以要重算。`):'');}
  if(p.st==='held') return n+(i===o.m&&o.b[i]?tx(`: still in the cache, but day 2 has ${alName(o.b[i])} here, so nothing from here on matches.`,`：还在缓存里，但第 2 天这里是${alName(o.b[i])}，所以从这里起都对不上了。`)
    :tx(': still in the cache, but the parts before it no longer match, so day 2 cannot use it.','：还在缓存里，但它前面已经对不上了，所以第 2 天用不上。'));
  if(p.st==='same'){const j=o.a.findIndex(x=>x.cid===p.cid), pa=j>0?alName(o.a[j-1]):'', pb=i>0?alName(o.b[i-1]):'';
    /* when the parts just before it are the same, the prefix differs further up: at the thick line (review of v10, S2) */
    if(pa!==pb) return n+tx(`: cached, but on day 1 it came after ${pa} and here after ${pb}. A different prefix, so it is computed again.`,`：缓存里有，但第 1 天它前面是${pa}，这里前面是${pb}。前缀不同，所以要重算。`);
    return n+tx(`: cached, but what comes before it stops matching day 1 at the thick line (day 1 has ${alName(o.a[o.m])} there, day 2 has ${alName(o.b[o.m])}). A different prefix, so it is computed again.`,
                `：缓存里有，但它前面从粗线起就和第 1 天不同（第 1 天那里是${alName(o.a[o.m])}，第 2 天是${alName(o.b[o.m])}）。前缀不同，所以要重算。`);}
  return n+tx(': new on day 2, so prefill computes it.','：第 2 天新有的，所以 prefill 要算。');}
function alignHTML(){
  const LAB=[tx('Date after the new message','日期在新消息后'),tx('Date before the new message','日期在新消息前'),tx('Name before the new message','用户名在新消息前')];
  const it=(cls,txt)=>`<span><i class="sw ${cls}"></i>${esc(txt)}</span>`;
  /* closed by default (the owner's review of batch 8: no long listing open under the answer); its question is the summary, and the
     link under the verdict opens it */
  let h=`<details class="px-alx" id="px-alx"><summary>${esc(tx('Why not put the date or the name right before the new message? Day 1 and day 2 side by side','为什么日期和用户名不放在新消息前面？看第 1 天和第 2 天的对照'))}</summary>`+
    `<div class="px-al" id="px-al"><p class="px-al-q">${esc(tx('Where does Alice’s day-2 prompt stop matching what her day-1 request left in the cache?','Alice 第 2 天的 prompt，从哪里开始和第 1 天留在缓存里的不一样？'))}</p>`+
    `<div class="px-al-leg">${it('kv',tx('amber: the same as day 1 so far, read from the cache','琥珀色：到这里都和第 1 天相同，从缓存读'))}${it('px-sw-held',tx('dashed: cached, but no longer a match','虚线框：在缓存里，但对不上了'))}${it('px-sw-same',tx('hatched: same tokens after a new prefix, computed again','斜纹：token 一样但前缀变了，要重算'))}${it('px-sw-comp',tx('gray: new, computed','灰色：新的，现算'))}${it('px-sw-cut',tx('thick line: the first difference','粗线：从这里开始不同'))}<span class="px-al-ns">${esc(tx('One row per part, not to scale.','每行一个部分，不按 token 数的比例。'))}</span></div>`;
  AL_ORDERS.forEach((ord,k)=>{
    const o=alData(ord), rowOf=i=>i<o.m?2+i:3+i, me=ord.join()===chOrder.join();
    const cell=(side,i)=>{const p=(side===1?o.a:o.b)[i]; if(!p) return '';
      const cls=p.st==='hit'?'px-s0':(p.st==='held'?'px-al-held':(p.st==='same'?'px-s2':'px-s1'));
      return `<span class="px-al-p ${cls}" style="grid-area:${rowOf(i)}/${side===1?1:3}" data-side="${side}" data-st="${p.st}" data-cid="${p.cid}" tabindex="0" data-tipx="${esc(alTip(o,side,i))}"><b>${esc(alName(p))}</b><em>${fint(p.n)}</em></span>`;};
    let g=`<span class="px-al-ch" style="grid-area:1/1">${esc(tx('① Day 1, cached','① 第 1 天，已缓存'))}</span><span class="px-al-ch" style="grid-area:1/3">${esc(tx('② Day 2 prompt','② 第 2 天 prompt'))}</span>`;
    for(let i=0;i<Math.max(o.a.length,o.b.length);i++) g+=cell(1,i)+cell(2,i);
    g+=`<span class="px-al-cut" style="grid-area:${2+o.m}/1/${3+o.m}/4"><span>${esc(tx('from here on they differ','从这里开始不同'))}</span></span>`;
    const hs=o.b.map((p,i)=>p.part==='hist'?i:-1).filter(i=>i>=0);
    if(hs.length) g+=`<span class="px-al-br" style="grid-area:${rowOf(hs[0])}/4/${rowOf(hs[hs.length-1])+1}/5"><span>${esc(tx('history','对话历史'))}</span></span>`;
    h+=`<figure class="px-al-fig${me?' me':''}" data-o="${ord.join()}" aria-label="${esc(tx(`${AL_KEY[k]}: ${LAB[k]}. Day 1 and day 2 match up to ${alName(o.a[o.m-1])}; day 2 computes ${fint(o.N-o.hit)} tokens and reads ${fint(o.hit)} from the cache.`,`${AL_KEY[k]}：${LAB[k]}。两天一直相同到${alName(o.a[o.m-1])}；第 2 天要算 ${fint(o.N-o.hit)} 个 token，${fint(o.hit)} 个来自缓存。`))}">`+
      `<figcaption><span class="px-al-k">${AL_KEY[k]}</span><span class="px-al-t"><b>${esc(LAB[k])}</b><span class="px-al-avg">${esc(tx(`${fint(o.avg)} cached per request, all 8`,`8 个请求平均缓存 ${fint(o.avg)}`))}${me?` <span class="cmp-you">${esc(tx('your pick','你的选择'))}</span>`:''}</span></span></figcaption>`+
      `<div class="px-al-g">${g}</div><div class="px-al-foot">${esc(tx(`Day 2 reads ${fint(o.hit)} tokens from the cache (the ${fint(o.match)} that match, rounded down to whole 16-token blocks) and computes ${fint(o.N-o.hit)}.`,`第 2 天从缓存读 ${fint(o.hit)} 个 token（相同的 ${fint(o.match)} 个按 16 个 token 的 block 向下取整），要算 ${fint(o.N-o.hit)} 个。`))}</div></figure>`;});
  h+=tx(`<p class="small"><b>Why not put the date or the name right before the new message?</b> Because on day 2 the chat history is the old history followed by message ① and answer ① (marked “history” on the right), and nothing else. Whatever sat between the history and the message on day 1 is missing from it: B put the date there and C the name, so both stop matching at the end of the old history and compute message ① and answer ① again. A puts the date after the message, so it computes only answer ① again. The name never changes for a user, so it breaks nothing before the history.</p>`,
        `<p class="small"><b>为什么日期和用户名不放在新消息前面？</b>因为第 2 天的对话历史只是在旧历史后面接上消息①和回答①（右边标着“对话历史”的那一段）。第 1 天夹在历史和新消息之间的东西不会进到历史里：B 在那里放了日期，C 放了用户名，所以两者的匹配都停在旧历史末尾，消息①和回答①都要重算。A 把日期放在新消息后面，所以只重算回答①。用户名对同一个用户不变，所以放在历史前面不会打断匹配。</p>`);
  h+=tx(`<p class="small muted">This depends on the setup: the history keeps only messages and answers, and every user returns on another day. Change the setup and the ranking changes: if the date were saved in the history with each message, B would match A; if users came back the same day, the date would do better before the name and the history, and answer ① would come from the cache too.</p>`,
        `<p class="small muted">这个结论依赖题设：历史只存消息和回答，不存日期和用户名；每个用户都在另一天回来。题设变了，排名也会变：日期若随消息存进历史，B 和 A 一样好；用户若都当天回来，日期放在用户名和历史前面更好，回答①也能从缓存读。</p>`);
  return h+'</div></details>';}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chRun(chOrder), minM=chMinMoves();
  let h;
  const avgS=r=>fint(r.avg);
  const more=me.ok&&me.avg<chBestAvg()?tx(` Order A below caches a little more (${fint(chBestAvg())} per request).`,`下面的 A 顺序缓存得更多一点（平均每个请求 ${fint(chBestAvg())}）。`):'';
  if(me.ok&&me.moves<=minM) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: on average ${avgS(me)} of each request’s prompt tokens come from the cache.</b> And you moved only ${me.moves} parts, the fewest that meet the goal.${more}</p>`,
                                   `<p class="okline" style="margin-top:12px"><b>达成目标：平均每个请求有 ${avgS(me)} 个 prompt token 来自缓存。</b>而且你只移动了 ${me.moves} 个部分，这是达标所需的最少。${more}</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: on average ${avgS(me)} of each request’s prompt tokens come from the cache.</b> A smaller change would do: moving only today’s date and the user’s name meets the goal too.${more}</p>`,
                      `<p class="okline" style="margin-top:12px"><b>达成目标：平均每个请求有 ${avgS(me)} 个 prompt token 来自缓存。</b>其实改得更少也行：只移动今天的日期和用户名，也能达标。${more}</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: on average ${avgS(me)} of each request’s prompt tokens come from the cache, below 4,400.</b></p><p class="small">Hint: which parts are the same in every request, which only for one user, and which change every day or every request? A part that changes makes every block after it miss.</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：平均每个请求只有 ${avgS(me)} 个 prompt token 来自缓存，不到 4,400 个。</b></p><p class="small">提示：哪些部分每个请求都一样，哪些只对同一个用户一样，哪些每天或每个请求都变？一个会变的部分，会让它之后的每个 block 都未命中。</p>`)
    +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  h+=`<p class="small px-al-go">${J('alq',tx('Why not put the date or the name right before the new message? ↓','为什么日期和用户名不放在新消息前面？↓'))}</p>`;
  const rows=CH_SHOW.map(o=>chRun(o)); if(!rows.some(r=>r.order.join()===me.order.join())) rows.push(me);
  const lab=o=>{const k=o.join(); if(k===CH.parts.join()) return tx('Date and name first','日期和名字在最前');
    if(k===CH_SHOW[1].join()) return tx('Date moved below the history','日期移到历史后面'); if(k===CH_SHOW[2].join()) return tx('Name moved below the examples','名字移到示例后面');
    if(k===CH_SHOW[3].join()) return tx('Both moved: date before the message','两个都移：日期在消息前'); if(k===CH_SHOW[4].join()) return tx('Both moved: date after the message','两个都移：日期在消息后');
    return tx('Your order','你的顺序');};
  /* brief (the owner's review of batch 8): the chart keeps the orders that meet the goal, the pick and the initial setup; the orders
     that move only the date or only the name are left out, counted under the chart and named in words below */
  const isMe=r=>r.order.join()===me.order.join(), isStart=r=>r.order.join()===CH.parts.join(), shownR=rows.filter(r=>r.ok||isMe(r)||isStart(r));
  h+=A.ui.cmpChart({title:tx(`Average cached tokens per request, ${MODEL} on one ${GPUN}, 8 requests over 3 days (goal: at least 4,400)`,`一张 ${GPUN} 跑 ${MODEL}、3 天 8 个请求时，平均每个请求来自缓存的 token（目标：至少 4,400）`),
    aria:tx('Average cached tokens per request: ','平均每个请求来自缓存的 token：')+shownR.map(r=>`${lab(r.order)}${tx(': ','：')}${fint(r.avg)}`).join(tx('; ','；')),
    brief:true,hiddenText:n=>tx(n===1?'The other order compared here moves only the date or only the name; it misses the goal and is not shown.':`The other ${n} orders compared here move only the date or only the name; they miss the goal and are not shown.`,
                                `其余 ${n} 种顺序只移了日期或名字，没达到目标，图里没有列出。`),
    rows:rows.map(r=>({label:lab(r.order),value:r.avg,text:fint(r.avg),ok:r.ok,me:r.order.join()===me.order.join(),
      tags:(r.order.join()===CH.parts.join()?[{text:tx('initial setup','初始设置')}]:[]).concat(r.order.join()===me.order.join()?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    goal:CH.goal,goalLabel:tx('goal: 4,400','目标 4,400'),max:6000,okText:tx('4,400 or more','4,400 个或更多'),noText:tx('fewer than 4,400','不到 4,400 个'),goalText:tx('goal: 4,400 cached tokens per request','目标：每个请求 4,400 个缓存 token')});
  /* baseline → effect → reason in words; the two single moves are not on the brief chart, so their averages are on hover */
  const one=(o,en,zh)=>{const r=chRun(o); return DEF(tx(en,zh),tx(`${en}: ${fint(r.avg)} cached per request on average, under 4,400.`,`${zh}，平均每个请求缓存 ${fint(r.avg)} 个 token，不到 4,400。`));};
  const dOnly=one(CH_SHOW[1],'Moving only the date below the history','只把日期移到历史后面'), nOnly=one(CH_SHOW[2],'Moving only the name below the examples','只把名字移到示例后面');
  h+=tx(`<p class="small">Initial setup, the date and the name first: every block’s key includes today’s date and the user’s name, so nothing hits. ${dOnly} lets a returning user find her own history, but no two users share a block. ${nOnly} lets everyone share the instructions and examples, but only on the same day. Moving both hits the most; with the date after the new message, a returning user’s previous message stays cached too.</p>`,
        `<p class="small">初始设置把日期和名字放在最前，每个 block 的 key 都包含今天的日期和用户名，所以什么都命中不了。${dOnly}，回头的用户能找到自己的历史，但不同用户之间一个 block 也不共享。${nOnly}，大家能共用指令和示例，但只限同一天。两个都移，命中最多。日期再放到新消息后面，回头的用户上一条消息也能留在缓存里。</p>`);
  const best=AL_ORDERS[0], bestTxt=best.map((p,i)=>{const s=L(CH_NAME[p]); return i?s.charAt(0).toLowerCase()+s.slice(1):s;}).join(tx(', ','、'));
  h+=tx(`<p class="small">Why: the instructions and examples (4,000 tokens) are the same in every request, so they go first; a user’s name never changes and her history only grows at its end, so what was cached for her is still a prefix, and they come next; the new message and the date change, so they go last, the date after the message. <b>One best order: ${bestTxt} (${fint(chRun(best).avg)} cached per request).</b></p>`,
        `<p class="small">原因：指令和示例（4,000 个 token）每个请求都一样，所以放最前；用户名不变，历史只在末尾变长，所以之前缓存的部分仍是前缀，它们接着放；新消息和日期会变，所以放最后，日期在新消息后面。<b>最好的顺序之一：${bestTxt}（平均每个请求缓存 ${fint(chRun(best).avg)}）。</b></p>`);
  h+=alignHTML();
  h+=tx(`<p class="small muted">Note: this lab keeps every block for the 3 days. A real server evicts the least recently used blocks when memory runs short, so a user who returns after a busy day may find only the shared instructions cached. Before counting on a hit rate, measure it on your server: vLLM reports it as “Prefix cache hit rate” in its logs and as the counters <code>vllm:prefix_cache_hits</code> and <code>vllm:prefix_cache_queries</code>.</p>`,
        `<p class="small muted">注意：这个 Lab 把每个 block 都保留了 3 天。真实服务在显存不够时会淘汰最久没用过的 block，所以一个用户隔了繁忙的一天再回来，可能只剩共享的指令还在缓存里。估算命中率之前，要在自己的服务上实测：vLLM 在日志里报告“Prefix cache hit rate”，也导出计数器 <code>vllm:prefix_cache_hits</code> 和 <code>vllm:prefix_cache_queries</code>。</p>`);
  return h;}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${['Six','Seven','Eight'][STEPS.length-6]||STEPS.length} short parts. Predict first, then see why a second request with the same system prompt skips most of its prefill, what breaks a cache hit, how far one changed token reaches, and what the cache costs. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看同一段 system prompt 的第二个请求为什么能跳过大部分 prefill、什么会让缓存失效、改一个 token 会影响多远，以及缓存要付出什么代价。第一次学从这里开始。`),
  sandbox:tx(`Four real workloads for ${MODEL} on one ${GPUN}: a support chatbot, a multi-turn chat, questions about one long document, and few-shot classification. Pick one to see which blocks each request finds in the cache.`,
             `四个真实负载，都是一张 ${GPUN} 跑 ${MODEL}，分别是客服机器人、多轮对话、针对同一篇长文档提问和 few-shot 分类。选一个，看每个请求在缓存里找到了哪些 block。`),
  challenge:tx(`A chat assistant runs ${MODEL} on one ${GPUN}. Order the six parts of its prompt so that, over 8 requests from 3 users on 3 days, at least 4,400 prompt tokens per request come from the cache on average.`,
               `一个聊天助手在一张 ${GPUN} 上跑 ${MODEL}。给它的 prompt 的六个部分排个顺序，让 3 个用户 3 天里的 8 个请求平均每个至少有 4,400 个 prompt token 来自缓存。`)
});
const SRC={
 cfg:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 h100:'https://www.nvidia.com/en-us/data-center/h100/',
 dbx:'https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices',
 apc:'https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md',
 design:'https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md',
 utils:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py',
 mgr:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py',
 pool:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py',
 ccfg:'https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py',
 vlog:'https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py',
 radix:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py',
 sglinit:'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/cache_init_params.py',
 sglang:'https://arxiv.org/abs/2312.07104',
 vllm:'https://arxiv.org/abs/2309.06180'
};
const TILE_A=()=>`<a href="#map~inference" data-tipx="${esc(tx('Click to open the tree at Inference serving and see this tile.','点击回到知识树的“推理服务”这一枝，看这个方块。'))}">${esc(tx('“How can a 10,000-token system prompt be nearly free?”','“10,000 个 token 的 system prompt 怎么做到几乎免费？”'))}</a>`;
const FIDPOP=()=>{const m=MISSION().res[1];
 return tx(
 `<p><b>Toy run:</b> your browser runs a small prefix cache on made-up token IDs, and a formula prices each first token. ${J('math','“Show the math”')} works it out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The cache, as in vLLM: 16-token blocks; a block’s key hashes its parent block’s key and its 16 token IDs; a hit is the longest run of cached blocks from the first, and at least one prompt token is always computed. After a request, its full blocks of prompt and answer are cached.</li>`+
 `<li>Eviction, as in vLLM: least recently used blocks first, and a request’s last blocks before its first ones, once the cached blocks exceed the memory they may use.</li>`+
 `<li>The first token’s time: the step model of the home-page tile ${TILE_A()} and of ${labLink('prefill-decode')}. A step takes the longer of its arithmetic and its memory traffic; with H tokens cached, it computes only the N − H new ones, which still attend to all N and read their keys and values (128 KiB each).</li>`+
 `<li>${MODEL} in BF16 on one ${GPUN}: ${A.fmt.big(P_TOTAL)} parameters, 128 KiB of KV cache per token. The KV cache pool, ${gib1(POOL_B)}, comes from the ${labLink('kv-cache')} lab’s model (80 GB read as 80 GiB, ${Math.round(KVD.util*100)}% usable, ${KVD.overhead} GiB of runtime overhead).</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>Requests run one at a time, with no queue: the time to first token is the prefill alone. A busy server also batches requests; a hit still skips the same prefill.</li>`+
 `<li>Token IDs are made up: each segment’s content gives the same IDs every time, so equal text means equal tokens. A real tokenizer can split the same text differently after a different preceding token; this lab keeps segment boundaries fixed.</li>`+
 `<li>50% of the dense BF16 peak and 60% of the peak bandwidth are assumptions (${J('eff','see Settings')}); they set the times, not which blocks hit. With the cache, request 2’s step reads ${gbS(m.w.by)} and does ${fixd(m.w.fl/1e12,2)} TFLOP, so memory sets it at these efficiencies.</li>`+
 `<li>vLLM keeps a duplicate block when two requests compute the same block at the same time, and caches blocks while a request runs; here requests are sequential and a request’s blocks are cached when it ends. Neither changes a hit in this lab.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>vLLM: ${cite(SRC.design,'docs/design/prefix_caching.md')} (block hashes, only full blocks, free in reverse order, LRU eviction); ${cite(SRC.utils,'<code>hash_block_tokens</code> in vllm/v1/core/kv_cache_utils.py')}; ${cite(SRC.mgr,'<code>get_computed_blocks</code> in vllm/v1/core/kv_cache_manager.py')} (the last token is always recomputed).</li>`+
 `<li>${MODEL}: ${cite(SRC.cfg,'config.json')}. ${GPUN}: ${cite(SRC.h100,'NVIDIA’s product page')} (1,979 TFLOPS BF16 “with sparsity”, so 989.5 dense; 3.35 TB/s). Bandwidth share: ${cite(SRC.dbx,'Databricks (2023)')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-28. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器在虚构的 token 编号上真的跑一个小的 prefix cache，再用公式估算每个首 token 的耗时。${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>缓存和 vLLM 一样：16 个 token 一个 block；block 的 key 是父 block 的 key 和它 16 个 token 编号一起算出的哈希；命中的是从第一个 block 起连续缓存的最长一串，而且至少要现算一个 prompt token。请求结束后，它 prompt 和回答里写满的 block 都进缓存。</li>`+
 `<li>淘汰和 vLLM 一样：缓存的 block 超出能用的显存时，先淘汰最久没用过的；同一个请求的 block 里先淘汰靠后的。</li>`+
 `<li>首 token 的耗时：用首页方块${TILE_A()}和 ${labLink('prefill-decode')} 那个 Lab 的耗时模型。一步的耗时取计算和读写显存中较长的那个；缓存了 H 个 token 时，只算 N − H 个新 token，但它们仍要对全部 N 个 token 做 attention，并读它们的 key 和 value（每个 128 KiB）。</li>`+
 `<li>一张 ${GPUN} 用 BF16 跑 ${MODEL}：${A.fmt.big(P_TOTAL)} 个参数，每个 token 128 KiB 的 KV cache。KV cache 显存池 ${gib1(POOL_B)} 用的是 ${labLink('kv-cache')} 那个 Lab 的模型（80 GB 按 80 GiB 计，可用 ${Math.round(KVD.util*100)}%，运行时开销 ${KVD.overhead} GiB）。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>请求一个接一个跑，不排队：首 token 延迟就是 prefill 本身。繁忙的服务还会把请求攒成 batch；命中照样省掉同样的 prefill。</li>`+
 `<li>token 编号是虚构的：每段内容每次都给出同样的编号，所以文字相同，token 就相同。真实的 tokenizer 在前面接的 token 不同时，可能把同一段文字切得不一样；这个 Lab 让各段的边界固定。</li>`+
 `<li>稠密 BF16 峰值的 50% 和峰值带宽的 60% 是假设（${J('eff','见“自由调参”')}）；它们决定耗时长短，不决定哪些 block 命中。用了缓存，请求 2 这一步读 ${gbS(m.w.by)}、算 ${fixd(m.w.fl/1e12,2)} TFLOP，所以按这两个效率，读显存决定耗时。</li>`+
 `<li>两个请求同时算出同一个 block 时，vLLM 会留下一个重复的 block，而且请求运行时就把 block 放进缓存；这里请求依次运行，请求结束时才缓存。两者都不改变这个 Lab 里的命中。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>vLLM：${cite(SRC.design,'docs/design/prefix_caching.md')}（block 哈希、只缓存写满的 block、倒序释放、LRU 淘汰）；${cite(SRC.utils,'vllm/v1/core/kv_cache_utils.py 里的 <code>hash_block_tokens</code>')}；${cite(SRC.mgr,'vllm/v1/core/kv_cache_manager.py 里的 <code>get_computed_blocks</code>')}（最后一个 token 总要重算）。</li>`+
 `<li>${MODEL}：${cite(SRC.cfg,'config.json')}。${GPUN}：${cite(SRC.h100,'NVIDIA 产品页')}（BF16 1,979 TFLOPS 标着“with sparsity”，所以稠密是 989.5；3.35 TB/s）。带宽比例：${cite(SRC.dbx,'Databricks（2023）')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-28。原型，尚未经专家审阅。</p></div></div>`);};
/* the kit's primer word for word (CONTRACT §9), plus one line on what this lab builds on */
const PRIPOP=()=>A.t('priPop')+tx(`<p>This lab builds on ${labLink('prefill-decode')} (the first token waits for prefill) and ${labLink('paged-attention')} (the KV cache lives in blocks of 16 tokens).</p>`,
                                  `<p>这个 Lab 接着 ${labLink('prefill-decode')}（首 token 要等 prefill）和 ${labLink('paged-attention')}（KV cache 存在 16 个 token 一个的 block 里）往下讲。</p>`);
/* the capability layer (Codex review 1 #5, Batch 2) */
const OUTCOME=()=>tx('estimate what prefix caching saves a request, and order a prompt so that it stays cacheable.','估算 prefix caching 能为一个请求省下多少时间，并把 prompt 排成能命中缓存的顺序。');
const EXITS=()=>{
  const m=MISSION().res[1], e=run({...BASE,change:'edit'}).res[1];
  const nc={P:50000,O:500}, N1=nc.P+nc.O, w1=ttft(N1,nc.P), wo1=ttft(N1,0), fs=H100.peak*EFF.mfu, bs=H100.bw*EFF.mbu;
  nc.lay=2*P_LAYERS*nc.O; nc.att=attn(nc.P,N1);
  const M70=A.kv&&A.kv.MODELS&&A.kv.MODELS['llama31-70b']?A.kv.MODELS['llama31-70b']:{layers:80,kvHeads:8,headDim:128};
  const kv70=2*M70.layers*M70.kvHeads*M70.headDim*2, mem70=10000*kv70;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('A second request starts with the same 10,000-token system prompt as the first. Why can it reuse the first request’s KV cache for those tokens?','第二个请求和第一个以同一段 10,000 个 token 的 system prompt 开头。为什么它能复用第一个请求这些 token 的 KV cache？'),
   o:[tx('Under the causal mask a token’s key and value depend only on it and the tokens before it, so the same first 10,000 tokens give the same keys and values, whatever question follows them.','有因果 mask，一个 token 的 key 和 value 只取决于它自己和它前面的 token，所以前 10,000 个 token 相同，它们的 key 和 value 就相同，不管后面跟着什么问题。'),
     tx('The model computes each token ID’s key and value once and stores them in a table, so any request that contains that token reads the same entry, wherever the token appears.','模型给每个 token 编号只算一次 key 和 value，存在一张表里，所以任何包含这个 token 的请求都读同一条记录，不管 token 出现在哪。'),
     tx('The server stores the first request’s prompt together with its answer, so a second request with the same system prompt gets that stored work back and skips the computation.','服务端把第一个请求的 prompt 和回答一起存下，所以同一段 system prompt 的第二个请求能拿回这些存好的结果，跳过计算。')],a:0,
   n:[null,tx('A key and a value depend on the token’s position and on every token before it, not on its ID alone: the same token after a different prefix gets different ones. That is why only an identical prefix is reused.','key 和 value 取决于 token 的位置和它前面的每个 token，而不只是它的编号：同一个 token 前面接的内容不同，key 和 value 就不同。所以只有完全相同的前缀才能复用。'),
     tx('The cache holds keys and values of the prompt, not answers: request 2 still computes its own 200 tokens and writes its own answer.','缓存里存的是 prompt 的 key 和 value，不是回答：请求 2 仍要算它自己的 200 个 token，自己写回答。')],
   why:tx(`Each layer computes a token’s key and value from its vector, which attention built only from that token and the tokens before it. So the first 10,000 tokens’ keys and values, 10,000 × 128 KiB = ${gib2(m.hit*KV_TOK)}, are fixed by those tokens alone, and request 2 reads them instead of recomputing them: ${fT(m.wo.t)} → ${fT(m.w.t)}.`,
          `每一层都从 token 的向量算出它的 key 和 value，而这个向量是 attention 只用这个 token 和它前面的 token 算出来的。所以前 10,000 个 token 的 key 和 value（10,000 × 128 KiB = ${gib2(m.hit*KV_TOK)}）只由这些 token 决定，请求 2 直接读，不用重算：${fT(m.wo.t)} → ${fT(m.w.t)}。`)},
  {q:tx('After request 1, one word in the middle of the 10,000-token system prompt changes. Why does request 2 miss every block from that word on, although the 5,000 tokens after it are the same as before?','请求 1 之后，10,000 个 token 的 system prompt 中间改了一个词。为什么请求 2 从这个词起每个 block 都未命中，尽管它后面的 5,000 个 token 和以前一样？'),
   o:[tx('The tokens after the word attended to it, so their keys and values really differ; each block’s key hashes all tokens up to it, so from the changed block on no key matches.','这个词后面的 token 都对它做过 attention，所以它们的 key 和 value 确实不同；每个 block 的 key 包含它之前的全部 token，所以从改动的 block 起，一个 key 都对不上。'),
     tx('The server compares the new prompt with the cached one token by token and stops at the first difference, because comparing further would take longer than computing the rest.','服务端把新 prompt 和缓存里的逐个 token 比较，在第一个不同处停下，因为继续比较比把剩下的算一遍还慢。'),
     tx('The new word has a different length, so every later token shifts to a new position, and the cached blocks no longer line up with the new ones.','新词的长度不同，后面每个 token 的位置都挪了，所以缓存的 block 和新的 block 对不齐。')],a:0,
   n:[null,tx('The stop is not about the cost of comparing: the later keys and values are genuinely different, because those tokens attended to the changed word. Reusing them would change the model’s output.','停下不是因为比较的代价：后面的 key 和 value 本来就不同，因为那些 token 对改动的词做过 attention。复用它们会改变模型的输出。'),
     tx('Here the change keeps the length: one token for one token. The later tokens are the same and in the same places, yet they still miss, because their keys and values depend on the changed token.','这里的改动长度不变：一个 token 换一个 token。后面的 token 一样、位置也一样，却照样未命中，因为它们的 key 和 value 取决于改动的那个 token。')],
   why:tx(`Token 5,000 sits in block ${Math.floor(4999/BLOCK)+1} of 16 tokens, so the ${fint(e.hitBlocks)} blocks before it, ${fint(e.hit)} tokens, still hit, and request 2 waits ${fT(e.w.t)} instead of ${fT(m.w.t)}. A block’s key stands for every token up to it, so a match can only be a run from the start.`,
          `第 5,000 个 token 在第 ${Math.floor(4999/BLOCK)+1} 个 block 里（每个 16 个 token），所以它前面的 ${fint(e.hitBlocks)} 个 block，也就是 ${fint(e.hit)} 个 token，仍然命中，请求 2 要等 ${fT(e.w.t)}，而不是 ${fT(m.w.t)}。一个 block 的 key 代表它之前的全部 token，所以能命中的只能是从开头起连续的一串。`)}],
 transfer:[
  {q:tx(`${MODEL} on the same ${GPUN}: a 50,000-token codebase is cached, and each request adds a 500-token question. About how long does the first token take?`,`还是一张 ${GPUN} 跑 ${MODEL}：一个 50,000 个 token 的代码库已经在缓存里，每个请求再加一个 500 个 token 的问题。首 token 大约要多久？`),
   o:[tx('Under 10 ms','不到 10 ms'),tx('10 to 30 ms','10 到 30 ms'),tx('30 to 100 ms','30 到 100 ms'),tx('More than 100 ms','超过 100 ms')],a:2,
   why:tx(`Only the 500 new tokens are computed, but each attends to up to 50,500 tokens: ${fixd(nc.lay/1e12,2)} TFLOP for the layers plus ${fixd(nc.att/1e12,2)} TFLOP of attention (and the LM head, once) = ${fixd(w1.fl/1e12,2)} TFLOP ÷ ${fixd(fs/1e12,2)} TFLOP/s = ${fT(w1.tc)}. Reading the weights and 50,500 tokens of KV cache, ${gbS(w1.by)} ÷ ${fixd(bs/1e12,2)} TB/s = ${fT(w1.tm)}, is shorter, so arithmetic sets it: about ${fT(w1.t)}, against ${fT(wo1.t)} without the cache.`,
          `只算 500 个新 token，但每个都要对最多 50,500 个 token 做 attention：各层 ${fixd(nc.lay/1e12,2)} TFLOP，加上 attention ${fixd(nc.att/1e12,2)} TFLOP（LM head 再算一次）= ${fixd(w1.fl/1e12,2)} TFLOP ÷ ${fixd(fs/1e12,2)} TFLOP/s = ${fT(w1.tc)}。读权重和 50,500 个 token 的 KV cache 是 ${gbS(w1.by)} ÷ ${fixd(bs/1e12,2)} TB/s = ${fT(w1.tm)}，更短，所以计算决定耗时：约 ${fT(w1.t)}；不用缓存要 ${fT(wo1.t)}。`),
   go:()=>exGo({P:50000,O:500,n:2,kind:'same',change:'none',apps:1,mem:'pool'}),goLabel:tx('Open it in Scenarios','在应用场景里打开')},
  {q:tx(`Llama‑3.1‑70B keeps 2 × 80 layers × 8 KV heads × 128 × 2 bytes of KV cache per token. How much GPU memory does its cached 10,000-token system prompt hold?`,`Llama‑3.1‑70B 每个 token 的 KV cache 是 2 × 80 层 × 8 个 KV head × 128 × 2 字节。它缓存的 10,000 个 token 的 system prompt 占多少显存？`),
   o:[tx('Under 2 GiB','不到 2 GiB'),tx('2 to 5 GiB','2 到 5 GiB'),tx('5 to 20 GiB','5 到 20 GiB'),tx('More than 20 GiB','超过 20 GiB')],a:1,
   why:tx(`2 × 80 × 8 × 128 × 2 = ${fint(kv70)} bytes = ${fint(kv70/1024)} KiB per token, ${fixd(kv70/KV_TOK,1)} times ${MODEL}’s 128 KiB; 10,000 × ${fint(kv70)} = ${fint(mem70)} bytes = ${gbS(mem70)} = ${gib2(mem70)}, against ${gib2(10000*KV_TOK)} for ${MODEL}.`,
          `2 × 80 × 8 × 128 × 2 = ${fint(kv70)} 字节，每个 token ${fint(kv70/1024)} KiB，是 ${MODEL} 的 128 KiB 的 ${fixd(kv70/KV_TOK,1)} 倍；10,000 × ${fint(kv70)} = ${fint(mem70)} 字节 = ${gbS(mem70)} = ${gib2(mem70)}，${MODEL} 只要 ${gib2(10000*KV_TOK)}。`)}],
 practice:[
  {id:'inference-engine',d:tx('Level 4 adds prefix caching to a small engine: chain block hashes from the start, reuse a cached block by raising its reference count, and keep freed blocks cached until they are evicted.','第 4 关给一个小引擎加上前缀缓存：从开头起链式计算 block 的哈希，命中的 block 只加引用计数来复用，释放的 block 留在缓存里，直到被淘汰。')},
  {id:'inference-serving',d:tx('Part 1 of the solution assumes the server may not cache a conversation’s earlier turns: redo its first-token latency and GPU time per request when it may.','参考解答的第 1 部分假设服务端不能缓存对话之前的几轮：换成可以缓存，重算首 token 延迟和每个请求的 GPU 时间。')}]};
};
function exGo(s){setMode('sandbox'); byLearner(()=>setState({...s})); sel=1; render(); A.ui.reveal($('#px-stage',app));}

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Prefix caching','Prefix caching'),title:l?L(l.q):tx('How can a 10,000-token system prompt be nearly free?','10,000 个 token 的 system prompt 怎么做到几乎免费？'),
    minutes:10,fid:'Simulated',fidPop:FIDPOP(),primer:A.t('priBtn').replace(/\s*▾$/,''),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage px-stage" id="px-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="px-st-title"></div><div class="meter-sub" id="px-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="px-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="px-legend"></div>`+
   `<div class="stage-graph graph px-graph" id="px-graph">`+
    `<div class="px-rowsbox" id="px-rowsbox"><div class="px-cap" id="px-rows-cap"></div><div class="px-rows" id="px-rows" role="group"></div><div class="px-sum small" id="px-sum"></div></div>`+
    `<div class="px-stripbox" id="px-stripbox"><div class="px-cap" id="px-strip-cap"></div><div class="px-pins" id="px-pins-top"></div><div class="px-strip" id="px-strip"><div class="px-band" id="px-band"></div><div class="px-cells" id="px-cells" tabindex="0"></div><i class="px-miss" id="px-miss" hidden></i></div><div class="px-pins" id="px-pins"></div><div class="px-strip-foot small" id="px-strip-foot"></div></div>`+
    `<div class="px-tl" id="px-tl"><div class="px-cap" id="px-tl-cap"></div>`+
     `<div class="px-tlrow" data-k="wo"><div class="px-tll"><span class="px-tln"></span><span class="px-tlv mono"></span></div><div class="px-track" tabindex="0"><i class="px-pre"></i><i class="px-first"></i><span class="px-ticks"></span></div></div>`+
     `<div class="px-tlrow" data-k="w"><div class="px-tll"><span class="px-tln"></span><span class="px-tlv mono"></span></div><div class="px-track" tabindex="0"><i class="px-pre"></i><i class="px-first"></i><span class="px-ticks"></span></div></div>`+
     `<div class="px-axis mono" aria-hidden="true"></div></div>`+
    /* part 5: request 1 and request 2 block by block, the key comparison between them, the three positions, the counters */
    `<div class="px-pairbox" id="px-pairbox"><div class="px-cap" id="px-pair-cap"></div>`+
     `<div class="px-pair" id="px-pair">`+
      `<span class="px-plab"></span><div class="px-pcol px-pbandcol"><div class="px-ppins" id="px-ppins"></div><div class="px-band px-pband" id="px-pband"></div></div>`+
      `<span class="px-plab" id="px-pl1"></span><div class="px-pcol"><div class="px-ptrack" id="px-pt1" tabindex="0"><div class="px-pcells" id="px-pc1"></div></div></div>`+
      `<span class="px-plab"></span><div class="px-pcol"><div class="px-lane" id="px-lane" aria-hidden="true"><i class="px-larrow" id="px-larrow" hidden></i></div></div>`+
      `<span class="px-plab" id="px-pl2"></span><div class="px-pcol"><div class="px-ptrack" id="px-pt2" tabindex="0"><div class="px-pcells" id="px-pc2"></div><div class="px-pcover" id="px-pcover"><span></span></div><i class="px-pmiss" id="px-pmiss" hidden></i><i class="px-ptick" id="px-ptick" hidden></i></div></div>`+
      `<span class="px-plab px-plab3" id="px-pl3"></span><div class="px-pcol"><div class="px-ppos" id="px-ppos" role="group"></div></div>`+
     `</div>`+
     `<div class="px-otqs" id="px-otqs"></div>`+
     `<p class="px-same small" id="px-same"></p>`+
     `<div class="px-ctrs" id="px-ctrs"></div>`+
     `<div class="px-pfoot"><span class="px-prog small" id="px-prog" aria-live="polite"></span><span class="px-otbtns"><button type="button" class="btn ghost sm px-otreplay" data-act="ot-replay" id="px-otreplay" hidden data-tipx="${esc(tx('Plays request 2’s lookup again, from its first block.','从第一个 block 起，把请求 2 的查找再播一遍。'))}">${tx('↻ Replay','↻ 重放')}</button><button type="button" class="btn ghost sm px-otstep" data-act="ot-step" id="px-otstep"></button></span></div>`+
    `</div>`+
   `</div>`+
   `<p class="px-scope small muted" tabindex="0" data-tipx="${esc(tx('Every time here comes from this lab’s step model: the arithmetic at 50% of the GPU’s peak FLOP/s, the memory traffic at 60% of its peak bandwidth, whichever is longer. A real server also queues requests and batches them with others.','这里的每个时间都来自这个 Lab 的耗时模型：计算按 GPU 峰值 FLOP/s 的 50%，读写显存按峰值带宽的 60%，取较长的那个。真实的服务还有排队，也会把请求和别的请求攒成 batch。'))}">${tx('Estimates for one GPU: no queue and no network, and requests run one at a time.','单卡估算：不含排队和网络，请求一个接一个跑。')}</p>`+
   `<div class="metrics px-metrics" id="px-metrics"></div>`+
   `<details class="math" id="px-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="px-mathbody"></div></details></div>`;
}
const EFF_NOTE=()=>tx(`This lab assumes a step reaches 50% of the ${GPUN}’s dense BF16 peak (989.5 TFLOP/s), that is ${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s. It matters when arithmetic sets the time: a prefill without a hit, or a hit followed by many new tokens.`,
  `本 Lab 假设一步能用到 ${GPUN} 稠密 BF16 峰值（989.5 TFLOP/s）的 50%，即 ${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s。计算决定耗时的时候它才起作用：没有命中的 prefill，或者命中之后还有很多新 token 要算。`);
const BW_NOTE=()=>tx(`This lab assumes a step reaches 60% of the peak memory bandwidth (3.35 TB/s), that is ${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s; Databricks (2023) measured 60% at batch size 1 on H100 GPUs. It sets the time of a step that computes few tokens, such as a prefill after a long hit.`,
  `本 Lab 假设一步能用到峰值显存带宽（3.35 TB/s）的 60%，即 ${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s；Databricks（2023）在 H100 上实测，batch 大小为 1 时是 60%。只算少量 token 的一步，比如长命中之后的 prefill，耗时由它决定。`);
const rangeStops=(id,key,label,stops,val,fmt,inline)=>A.ui.range({id,key,label,min:0,max:stops.length-1,value:Math.max(0,stops.indexOf(val)),out:fmt(val),inline});
const segCtl=(id,key,label,opts,val,inline,note)=>A.ui.seg({id,key,label,options:opts,value:String(val),inline,note});
const changeOpts=()=>[['none',tx('Nothing','没有')],['dateStart',tx('Date first','日期在前')],['dateAfter',tx('Date after','日期在后')],['edit',tx('One word','改一个词')],['nameStart',tx('Name first','名字在前')]];
const memOpts=()=>[['pool',tx('Whole pool','整个池')],['3',tx('3 requests','3 个请求')],['2',tx('2 requests','2 个请求')],['1',tx('1 request','1 个请求')]];
function controlsHTML(){
  return `<aside class="panel controls" id="px-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="px-lock" hidden></p><p class="ctl-note" id="px-prelock" hidden>${tx('You can change these settings after you answer part 2.','在第 2 部分作答后，才能改这些设置。')}</p>`+
   `<div class="lockable">`+rangeStops('px-ctl-P','P',tx('Shared prefix','共享前缀'),P_STOPS,S.P,v=>tokN(v),false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The tokens every request starts with: a system prompt, a document or examples. In chat turns it is the system prompt.','每个请求开头都一样的 token：system prompt、文档或示例。多轮对话里它就是 system prompt。')}</div></div>`)+`</div>`+
   `<div class="lockable">`+rangeStops('px-ctl-O','O',tx('Own tokens per request','每个请求自己的 token'),O_STOPS,S.O,v=>tokN(v),false).replace(/<\/div>$/,`<div class="ctl-note">${tx('The question or new message that follows the prefix; no other request has it.','跟在前缀后面的问题或新消息，别的请求都没有。')}</div></div>`)+`</div>`+
   `<div class="lockable">`+A.ui.range({id:'px-ctl-n',key:'n',label:tx('Requests','请求数'),min:1,max:N_MAX,value:S.n,out:reqN(S.n)})+`</div>`+
   `<div class="lockable">`+segCtl('px-ctl-kind','kind',tx('How the requests continue','请求之间的关系'),[['same',tx('Same prefix','同一前缀')],['chat',tx('Chat turns','多轮对话')]],S.kind,false,tx('Chat turns: each turn resends the conversation so far, 300-token answers included.','多轮对话：每一轮都重发之前的整段对话，包括每个 300 个 token 的回答。'))+`</div>`+
   `<div class="lockable" id="px-ctl-change-wrap">`+segCtl('px-ctl-change','change',tx('What differs before the own tokens','自己的 token 之前还有什么不同'),changeOpts(),S.change,false,`<span id="px-change-note"></span>`)+`</div>`+
   `<div class="lockable" id="px-ctl-apps-wrap">`+segCtl('px-ctl-apps','apps',tx('Apps with their own prefix','各有自己前缀的应用数'),[['1','1'],['3',tx('3: A, A, B, A, C, A, B, A','3：A、A、B、A、C、A、B、A')]],S.apps,false,tx('With 3, the requests go to apps A, B and C in that order, each app with its own prefix of the same length.','选 3 时，请求按这个顺序发给应用 A、B、C，每个应用有自己的前缀，长度相同。'))+`</div>`+
   `<div class="lockable" id="px-ctl-mem-wrap">`+segCtl('px-ctl-mem','mem',tx('Memory for cached blocks','缓存能用的显存'),memOpts(),S.mem,false,`<span id="px-mem-note"></span>`)+`</div>`+
   `<div class="px-grp lockable" id="px-ctl-eff"><div class="px-gh">${tx('Assumed values','假设值')}</div>`+
    `<div class="ctl"><span class="lbl">${tx('Block size: 16 tokens','block 大小：16 个 token')}</span><div class="ctl-note">${tx('vLLM’s default. A hit ends at a block boundary: a prefix of 1,000 tokens hits 992 (62 blocks), since its last block also holds the request’s own tokens.','vLLM 的默认值。命中只能停在 block 的边界上：1,000 个 token 的前缀只命中 992 个（62 个 block），因为它最后一个 block 里还有请求自己的 token。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Each answer: 300 tokens','每个回答：300 个 token')}</span><div class="ctl-note">${tx('Assumed. A finished request’s answer is cached too; in chat turns the next prompt includes it.','假设值。请求结束后，它的回答也进缓存；多轮对话里，下一轮的 prompt 包含它。')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Arithmetic: 50% of the peak FLOP/s','计算：峰值 FLOP/s 的 50%')}</span><div class="ctl-note">${EFF_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx('Memory: 60% of the peak bandwidth','读写显存：峰值带宽的 60%')}</span><div class="ctl-note">${BW_NOTE()}</div></div>`+
    `<div class="ctl"><span class="lbl">${A.t('cUtil')}${tx(': ','：')}${Math.round(KVD.util*100)}%</span><div class="ctl-note">${A.t('utilNote')}</div></div>`+
    `<div class="ctl"><span class="lbl">${tx(`Runtime overhead: ${KVD.overhead} GiB`,`运行时开销：${KVD.overhead} GiB`)}</span><div class="ctl-note">${A.t('ohNote')}</div></div></div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`# A block-hash prefix cache: 16-token blocks, LRU eviction
from collections import OrderedDict
BLOCK = 16

def block_keys(tokens):
    keys, parent = [], None
    for i in range(0, len(tokens) - BLOCK + 1, BLOCK):
        parent = hash((parent, tuple(tokens[i:i + BLOCK])))
        keys.append(parent)            # stands for tokens[:i+16]
    return keys

class PrefixCache:
    def __init__(self, max_blocks):
        self.blocks = OrderedDict()    # key -> KV block, LRU first
        self.max_blocks = max_blocks

    def lookup(self, tokens):          # cached tokens from the start
        hit = 0
        for k in block_keys(tokens[:-1]):  # always compute one
            if k not in self.blocks:   # insert() refreshes hits
                break
            hit += BLOCK
        return hit

    def insert(self, tokens):          # after the request
        for k in reversed(block_keys(tokens)):  # tail evicted first
            self.blocks[k] = "KV"
            self.blocks.move_to_end(k)
        while len(self.blocks) > self.max_blocks:
            self.blocks.popitem(last=False)     # least recently used

cache = PrefixCache(max_blocks=700)
system = list(range(10_000))           # a 10,000-token system prompt
for user in range(3):
    prompt = system + [200_000 + user] * 200
    print(user, cache.lookup(prompt))
    cache.insert(prompt)
# 0 0
# 1 10000
# 2 10000`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="px-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="px-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A prefix cache in 40 lines: a key per block that chains the keys before it, a lookup that stops at the first miss, and least-recently-used eviction.','40 行的 prefix cache：每个 block 一个串起前面所有 key 的 key，查找在第一个未命中处停下，再加上最近最少使用淘汰。')}</p><pre class="code" id="px-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="px-prod">`+
   tx(`<li>vLLM turns it on with <code>enable_prefix_caching</code>, which ${cite(SRC.ccfg,'<code>vllm/config/cache.py</code>')} sets to true by default; ${cite(SRC.apc,'its feature page')} names long documents and multi-turn chats as the workloads it helps, and says it saves prefill only, not decode.</li>`+
      `<li>${cite(SRC.utils,'<code>vllm/v1/core/kv_cache_utils.py</code>')}: <code>hash_block_tokens</code> hashes the parent block’s hash, the block’s token IDs and extra keys (LoRA, images, a cache salt). ${cite(SRC.mgr,'<code>vllm/v1/core/kv_cache_manager.py</code>')}: <code>get_computed_blocks</code> finds the longest hit, at most the prompt minus its last token, and <code>free</code> releases a request’s blocks in reverse order so its tail is evicted first.</li>`+
      `<li>${cite(SRC.pool,'<code>vllm/v1/core/block_pool.py</code>')} maps block hashes to blocks (<code>cached_block_hash_to_block</code>) and takes new blocks from the head of a free queue, evicting a cached one when it gets there. The design is explained in ${cite(SRC.design,'<code>docs/design/prefix_caching.md</code>')}.</li>`+
      `<li>SGLang keeps prefixes in a radix tree: ${cite(SRC.radix,'<code>python/sglang/srt/mem_cache/radix_cache.py</code>')} (<code>match_prefix</code>, <code>insert</code>, <code>evict</code>, which removes leaves first); its eviction policy defaults to <code>"lru"</code> in ${cite(SRC.sglinit,'<code>cache_init_params.py</code>')}.</li>`,
      `<li>vLLM 用 <code>enable_prefix_caching</code> 打开它，${cite(SRC.ccfg,'<code>vllm/config/cache.py</code>')} 里默认是 true；${cite(SRC.apc,'它的功能说明')}把长文档和多轮对话列为受益的负载，并说明它只省 prefill，不省 decode。</li>`+
      `<li>${cite(SRC.utils,'<code>vllm/v1/core/kv_cache_utils.py</code>')}：<code>hash_block_tokens</code> 对父 block 的哈希、这个 block 的 token 编号和额外的 key（LoRA、图片、cache salt）一起求哈希。${cite(SRC.mgr,'<code>vllm/v1/core/kv_cache_manager.py</code>')}：<code>get_computed_blocks</code> 找出最长的命中，最多到 prompt 的倒数第二个 token；<code>free</code> 倒序释放一个请求的 block，让它末尾的 block 先被淘汰。</li>`+
      `<li>${cite(SRC.pool,'<code>vllm/v1/core/block_pool.py</code>')} 用 <code>cached_block_hash_to_block</code> 从哈希找到 block，新 block 从空闲队列的队首取，取到缓存着的 block 就把它淘汰。设计说明见 ${cite(SRC.design,'<code>docs/design/prefix_caching.md</code>')}。</li>`+
      `<li>SGLang 把前缀存在一棵 radix tree 里：${cite(SRC.radix,'<code>python/sglang/srt/mem_cache/radix_cache.py</code>')}（<code>match_prefix</code>、<code>insert</code>，以及先删叶子的 <code>evict</code>）；${cite(SRC.sglinit,'<code>cache_init_params.py</code>')} 里的淘汰策略默认是 <code>"lru"</code>。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.sglang,'SGLang: Efficient Execution of Structured Language Model Programs')}<span>${tx('Zheng et al., NeurIPS 2024: RadixAttention, which keeps the KV cache of finished requests in a radix tree and reuses the longest matching prefix','Zheng 等，NeurIPS 2024：RadixAttention，把已完成请求的 KV cache 存在一棵 radix tree 里，复用最长的相同前缀')}</span></li>`+
   `<li>${cite(SRC.vllm,'Efficient Memory Management for Large Language Model Serving with PagedAttention')}<span>${tx('Kwon et al., SOSP 2023: a KV cache in fixed-size blocks, which lets requests share blocks','Kwon 等，SOSP 2023：按固定大小的 block 存 KV cache，让不同请求可以共用 block')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="px-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="px-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="px-app m2" id="px-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="px-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much sooner the first token comes once the cache hits.','选一个场景，看看命中缓存后，首 token 能快多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="px-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function chListHTML(){
  return `<ol class="px-order" id="px-order" aria-label="${esc(tx('The prompt’s parts, in order','prompt 的各部分，按顺序'))}">`+chOrder.map((p,i)=>`<li class="px-part" data-part="${p}"><span class="px-pnum mono">${i+1}</span><span class="px-pt"><b>${esc(L(CH_NAME[p]))}</b><span>${esc(CH_DESC[p]())}</span></span>`+
    `<span class="px-pbtns"><button type="button" class="btn ghost sm" data-act="up" data-part="${p}" aria-label="${esc(tx(`Move “${L(CH_NAME[p])}” up`,`把“${L(CH_NAME[p])}”上移`))}"${i===0?' disabled':''}>▲</button><button type="button" class="btn ghost sm" data-act="down" data-part="${p}" aria-label="${esc(tx(`Move “${L(CH_NAME[p])}” down`,`把“${L(CH_NAME[p])}”下移`))}"${i===chOrder.length-1?' disabled':''}>▼</button></span></li>`).join('')+`</ol>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`One ${GPUN}, 8 requests over 3 days: at least 4,400 cached tokens per request`,`一张 ${GPUN}、3 天 8 个请求：平均每个请求至少 4,400 个 token 来自缓存`)}</h3>`+
   `<p>${tx(`A chat assistant runs ${MODEL} on one ${GPUN}. Every request’s prompt has the six parts below, in one order you choose for all requests. Day 1: Alice, Bob, Carol. Day 2: Alice, Carol, Bob. Day 3: Bob, Alice. Order the parts so that, on average, at least 4,400 of each request’s prompt tokens (about 6,050) come from the cache:`,
            `一个聊天助手在一张 ${GPUN} 上跑 ${MODEL}。每个请求的 prompt 都由下面六个部分组成，顺序由你定，所有请求都一样。第 1 天：Alice、Bob、Carol。第 2 天：Alice、Carol、Bob。第 3 天：Bob、Alice。排好顺序，让平均每个请求的 prompt token（约 6,050 个）里至少 4,400 个来自缓存：`)}</p>`+
   chListHTML()+
   `<div class="goal mono" id="px-ch-goal"><span>${tx('Cached tokens per request now','现在平均每个请求的缓存 token')}</span><span><b id="px-ch-now"></b>${tx(' / at least 4,400',' / 至少 4,400')}</span></div><div class="progress" id="px-ch-prog" aria-hidden="true"><i></i></div>`+
   `<p class="small muted px-moves" id="px-ch-moves"></p>`+
   submit+`<div id="px-ch-res" class="px-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function renderRail(){const rail=$('#px-rail',app); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#px-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const body=s.obs?s.obs():'', show=!!s.obs&&body!==''&&(!s.expect||!!observed[step]||s.id==='break');
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+body;
        /* part 5: a box waiting for the next position's result keeps the height of the result before it (measured before it changes) */
        const wait=s.id==='onetoken'&&/px-otwait/.test(body);
        if(wait){if(!box._mh) box._mh=box.offsetHeight; box.style.minHeight=box._mh+'px';} else {box._mh=0; box.style.minHeight='';}
        if(box._h!==html){const again=box._step===step&&box._h!=null&&!wait; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    const sb=$('[data-act="send"]',rail); if(sb){sb.disabled=sent; sb.textContent=sent?tx('Request 2 sent','请求 2 已发出'):tx('Send request 2','发出请求 2');}
    if(s.id==='onetoken') $$('[data-lever]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.lever===otPos)); const r=$('.lv-r',b); if(r) r.textContent=otLvRes(b.dataset.lever);});
    else $$('[data-lever]',rail).forEach(b=>{const on=s.id==='break'?b.dataset.lever===lever:b.dataset.lever===String(S.mem); b.setAttribute('aria-pressed',String(on));});
    if(s.id==='break') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever, r=$('.lv-r',b); if(r) r.textContent=(v==='none'||leverTried.has(v))?leverRes(v):'';});
    if(s.id==='cost') $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever==='pool'?'pool':+b.dataset.lever, r=$('.lv-r',b); if(r) r.textContent=(v==='pool'||memTried.has(v))?memRes(v):'';});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#px-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the four scenarios above.','从上面四个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const r=chRun(chOrder), now=$('#px-ch-now',rail); if(!now) return;
    now.textContent=fint(r.avg);
    $('#px-ch-prog i',rail).style.width=Math.min(100,r.avg/CH.goal*100).toFixed(1)+'%';
    $('#px-ch-goal',rail).classList.toggle('met',r.ok); $('#px-ch-prog',rail).classList.toggle('met',r.ok);
    $('#px-ch-moves',rail).textContent=r.avg===0?tx(`Parts moved from the initial order: ${r.moves}. Nothing comes from the cache yet, so the first token comes after ${fT(r.avgW)} on average, as without the cache.`,`相对初始顺序移动了 ${r.moves} 个部分。还没有 token 来自缓存，首 token 平均 ${fT(r.avgW)} 后出现，和不用缓存时一样。`)
      :tx(`Parts moved from the initial order: ${r.moves}. The first token comes after ${fT(r.avgW)} on average.`,`相对初始顺序移动了 ${r.moves} 个部分。首 token 平均 ${fT(r.avgW)} 后出现。`);
    if(chSubmitted&&r.ok) chReveal=true;
    const res=$('#px-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}
/* the challenge list re-renders in place after a move, keeping the focus on the moved part's button */
function redrawOrder(p,dir){const ol=$('#px-order',app); if(!ol) return; ol.outerHTML=chListHTML(); const b=$(`#px-order [data-part="${p}"][data-act="${dir}"]`,app)||$(`#px-order [data-part="${p}"] button:not([disabled])`,app); if(b) b.focus({preventScroll:true});}

/* =====================================================================
   stage drawing
   ===================================================================== */
function curStream(){return mode==='challenge'?chRun(chOrder):R();}
function statusOf(){
  if(mode==='challenge'){const r=chRun(chOrder); return {cls:r.ok?'ok':'bad',txt:r.ok?tx(`Goal met: ${fint(r.avg)} cached`,`达成目标：缓存 ${fint(r.avg)} 个`):tx(`Below goal: ${fint(r.avg)} cached`,`未达标：缓存 ${fint(r.avg)} 个`),short:r.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx('Green when, on average over the 8 requests, at least 4,400 of each request’s prompt tokens come from the cache.','8 个请求平均下来，每个请求至少有 4,400 个 prompt token 来自缓存时，徽章是绿色。')};}
  const x=selRes(), i=selIdx();
  if(masked()) return {cls:'neutral',txt:tx('Request 1: nothing cached yet','请求 1：缓存里还没有东西'),short:tx('Empty cache','空缓存'),tip:tx('Request 1 finds an empty cache. Request 2’s result is shown in part 3, where it answers your prediction.','请求 1 遇到的是空缓存。请求 2 的结果在第 3 部分给出，那里揭晓你的预测。')};
  /* part 5: request 2 waits for a position and a prediction (R3.8), then its blocks are looked up */
  if(isOT()){
    if(!otPos) return {cls:'neutral',txt:tx('Request 2: not sent yet','请求 2：还没发出'),short:tx('Not sent','还没发出'),tip:tx('Request 2 is sent once you pick where its one changed token is and answer the question in this part.','选好请求 2 改动的 token 在哪、回答这一部分的问题之后，请求 2 才发出。')};
    if(!otPred) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('Request 2’s blocks stay hidden until you answer the question in this part.','回答这一部分的问题之前，请求 2 的 block 先隐藏。')};
    if(swK!=null&&!swStep) return {cls:'neutral',txt:tx('Request 2: looking up its blocks','请求 2：逐个查找 block'),short:tx('Looking up','查找中'),tip:tx('Request 2 looks up its blocks’ keys in the cache, from the first block on.','请求 2 从第一个 block 起，逐个在缓存里查找 block 的 key。')};
  }
  const txt=x.hit===0?tx(`${reqName(i)}: cache miss`,`${reqName(i)}：未命中`):tx(`${reqName(i)}: ${fint(x.hit)} tokens cached`,`${reqName(i)}：命中 ${fint(x.hit)} 个 token`);
  const tip=x.hit===0?tx(`None of the ${fint(x.N)} prompt tokens of ${reqOf(i)} came from the cache: its first block is not cached.`,`${reqOf(i)}的 ${fint(x.N)} 个 prompt token 没有一个来自缓存：它的第一个 block 就不在缓存里。`)
    :tx(`How many of the ${fint(x.N)} prompt tokens of ${reqOf(i)} came from the cache: ${fint(x.hit)}, in ${fint(x.hitBlocks)} blocks of 16 tokens from the start of the prompt.`,`${reqOf(i)}的 ${fint(x.N)} 个 prompt token 里有多少来自缓存：${fint(x.hit)} 个，也就是从 prompt 开头起的 ${fint(x.hitBlocks)} 个 block（每个 16 个 token）。`);
  return {cls:'neutral',txt,short:x.hit===0?tx('Miss','未命中'):tx('Hit','命中'),tip};
}
function drawHead(){
  const t=$('#px-st-title',app), sub=$('#px-st-sub',app), st=$('#px-status',app);
  t.textContent=tx(`${MODEL} on one ${GPUN}, BF16`,`一张 ${GPUN} 跑 ${MODEL}，BF16`);
  let s;
  const gid=mode==='guided'?STEPS[step].id:'', lv=LEVERS.find(o=>o.v===S.change);
  if(mode==='challenge') s=tx('A chat assistant: 8 requests from 3 users over 3 days, with the order you chose','聊天助手：3 个用户 3 天里的 8 个请求，按你选的顺序');
  else if(gid==='onetoken') s=otPos?tx(`Support chatbot, 2 requests: request${NS}2 changes token${NS}${fint(otTok(otPos))}`,`客服机器人，2 个请求：请求 2 改了第 ${fint(otTok(otPos))} 个 token`)
    :tx(`Support chatbot, 2 requests: request${NS}2 changes one token`,'客服机器人，2 个请求：请求 2 改了一个 token');   // as long as the title with a position, so a pick moves nothing
  else if(gid==='cost'&&S.apps===3&&S.n===8&&sameS(S,{...S,P:10000,O:200,kind:'same',change:'none'})) s=tx('Three apps, each with its own 10,000-token system prompt: 8 requests','三个应用，各有一段 10,000 个 token 的 system prompt：8 个请求');
  else if(gid&&sameS(S,{...BASE,change:S.change})) s=S.change==='none'?tx('Support chatbot: 2 requests, one 10,000-token system prompt','客服机器人：2 个请求，同一段 10,000 个 token 的 system prompt')
    :tx(`Support chatbot, 2 requests: ${L(lv.t).replace(/^./,c=>c.toLowerCase())}`,`客服机器人，2 个请求：${L(lv.t)}`);
  else {const p=PRESETS[presetNow()];
    if(mode==='sandbox'&&p) s=L(p.t)+tx(`: ${L(p.d)}`,`：${L(p.d)}`);
    else if(S.kind==='chat') s=tx(`${reqN(S.n)} as chat turns: a ${fint(S.P)}-token system prompt, ${fint(S.O)}-token messages`,`${reqN(S.n)}，多轮对话：${fint(S.P)} 个 token 的 system prompt，每条消息 ${fint(S.O)} 个 token`);
    else s=tx(`${reqN(S.n)} with the same ${fint(S.P)}-token prefix, ${fint(S.O)} tokens of their own each`,`${reqN(S.n)}，前缀都是同样的 ${fint(S.P)} 个 token，各有自己的 ${fint(S.O)} 个 token`);}
  sub.textContent=s;
  const so=statusOf(); st.className='status '+so.cls; st.textContent=so.txt; st.dataset.tipx=so.tip;
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  const kv=it('kv',tx('cached (KV cache)','缓存的（KV cache）'),tx('An amber block is a 16-token block found in the cache: its keys and values are read from GPU memory, so prefill does not compute them.','琥珀色是在缓存里找到的 16 个 token 的 block：它的 key 和 value 从显存里读，prefill 不用计算。'));
  /* part 5: no timeline, so no first-token lines; the changed token and the keys instead */
  if(isOT()) return kv+it('px-sw-comp',tx('computed','现算'),tx('A gray block is computed by prefill: nothing cached matches it. Request 1 computed all of its blocks.','灰色的 block 由 prefill 现算：缓存里没有能对上的。请求 1 的 block 全是现算的。'))+
    it('px-sw-same',tx('hatched: same tokens, new prefix','斜纹：token 一样，前缀变了'),TIP.same())+
    it('px-sw-miss',tx('thick line on blocks: first miss','block 上的粗线：第一个未命中'),tx('The thick line on request 2’s blocks marks its first block that is not in the cache; the arrow from it runs to the end, since every block after it misses too.','请求 2 的 block 上的粗竖线标出第一个不在缓存里的 block；从它出发的箭头一直到最后，因为之后的每个 block 也都未命中。'))+
    it('px-sw-chg',tx('triangle: the changed token','三角：改动的 token'),tx('The small triangle under request 2’s blocks marks the one token that differs from request 1: its place in the prompt, inside its block.','请求 2 的 block 下面的小三角，标出和请求 1 不同的那一个 token：它在 prompt 里、在它那个 block 里的位置。'))+
    it('px-sw-key',tx('keys: = same, ≠ different','key：= 相同，≠ 不同'),TIP.key()+tx(' Between the two requests, = marks a block whose key is the same in both (request 2 reads it from the cache), ≠ one whose key differs (request 2 computes it again). A key is shown by its last hex digits.',
      '两个请求之间，= 标出 key 相同的 block（请求 2 从缓存读它），≠ 标出 key 不同的 block（请求 2 要重算它）。key 只显示最后几位十六进制数字。'));
  return kv+
    it('px-sw-comp',tx('computed','现算'),tx('A gray block is computed by prefill: nothing cached matches it. On the timeline, the gray bar is the prefill’s time.','灰色的 block 由 prefill 现算：缓存里没有能对上的。时间线上的灰色条是 prefill 的耗时。'))+
    it('px-sw-same',tx('hatched: same tokens, new prefix','斜纹：token 一样，前缀变了'),TIP.same())+
    it('px-sw-first',tx('thick line: first token','粗线：首 token'),tx('The thick line on a timeline is the first token: it appears when prefill ends.','时间线上的粗线是首 token：prefill 结束时出现。'))+
    it('px-sw-tick',tx('thin tick: next token','细线：下一个 token'),tx('Each thin tick is one more token of the answer, one decode step each; the cache does not change them.','每条细线是回答的又一个 token，各用一个 decode 步；缓存不改变它们。'))+
    it('px-sw-miss',tx('thick line on blocks: first miss','block 上的粗线：第一个未命中'),tx('The thick line across the block strip marks the first block that is not in the cache: prefill computes it and every block after it.','block 条上的粗竖线标出第一个不在缓存里的 block：prefill 要计算它和它之后的每个 block。'));
}
/* the requests in order: each row's strip is one prompt on a shared token scale */
function drawRows(){
  const st=curStream(), rows=$('#px-rows',app), cap=$('#px-rows-cap',app), sum=$('#px-sum',app); if(!rows) return;
  const res=st.res, maxN=Math.max(...res.map(r=>r.N)), mk=masked(), si=selIdx(), ch=mode==='challenge';
  const lbl=(r,i)=>ch?tx(`Day ${r.req.day}, ${L(USERS[r.req.user])}`,`第 ${r.req.day} 天 ${L(USERS[r.req.user])}`):((S.apps===3&&S.kind==='same')?tx(`Request ${i+1}, app ${r.req.app}`,`请求 ${i+1}，应用 ${r.req.app}`):(S.kind==='chat'?tx(`Turn ${i+1}`,`第 ${i+1} 轮`):tx(`Request ${i+1}`,`请求 ${i+1}`)));
  const lblH=(r,i)=>ch?esc(lbl(r,i)):((S.apps===3&&S.kind==='same')?`<span class="px-rw">${tx('Request','请求')}${NS}</span>${i+1}${NS}<b>${r.req.app}</b>`:(S.kind==='chat'?`<span class="px-rw">${tx('Turn','第')}${NS}</span>${i+1}${tx('','<span class="px-rw">'+NS+'轮</span>')}`:`<span class="px-rw">${tx('Request','请求')}${NS}</span>${i+1}`));
  const apps3=!ch&&S.apps===3&&S.kind==='same', full=Math.floor(S.P/BLOCK), held=r=>['A','B','C'].filter(a=>r.apps[a]>=full);
  const chips=(r,i)=>{if(!apps3) return ''; const h=held(r), was=i>0?held(res[i-1]):[];
    return `<span class="px-rc" aria-hidden="true">${['A','B','C'].map(a=>`<i class="${h.includes(a)?'on':(was.includes(a)?'ev':'')}">${a}</i>`).join('')}</span>`;};
  rows.classList.toggle('px-rows-ch',ch); rows.classList.toggle('px-rows-apps',apps3);
  rows.innerHTML=res.map((r,i)=>{
    if((mk&&i>0)||(seqN!=null&&i>=seqN)) return `<div class="px-row px-row-wait" data-i="${i}"><span class="px-rl">${lblH(r,i)}</span><span class="px-rt"><span class="px-rwait">${mk?tx('not sent yet','还没发出'):tx('not here yet','还没到')}</span></span>${apps3?'<span class="px-rc"></span>':''}<span class="px-rv mono">?</span></div>`;
    const W=r.N/maxN*100, runs=runsOf(r.state), segsB=segBounds(r.req.segs).slice(1,-1).map(t=>`<b style="left:${(t/(r.total*BLOCK)*100).toFixed(3)}%"></b>`).join('');
    const tip=rowTip(r,i,lbl(r,i));
    const nw=seqN!=null&&i===seqN-1?' px-new':'';
    return `<button type="button" class="px-row${i===si?' on':''}${nw}" data-sel="${i}" aria-pressed="${i===si}" data-tipx="${esc(tip+(apps3?tx(' ',' ')+cacheTip(held(r)):''))}" aria-label="${esc(lbl(r,i))}"><span class="px-rl">${lblH(r,i)}</span><span class="px-rt"><span class="px-rbar" style="width:${W.toFixed(3)}%">${runs}${segsB}</span></span>${chips(r,i)}<span class="px-rv mono">${fT(r.w.t)}</span></button>`;}).join('');
  cap.innerHTML=esc(tx('Requests in order: each bar is one prompt, and the number its time to first token','按到达顺序的请求：每条是一个 prompt，数字是它的首 token 延迟'))+
    (apps3?`<span class="px-rcap">${tx('A B C mark which apps’ system prompts are in the cache after the request.','A B C 标出这个请求之后，缓存里还有哪些应用的 system prompt。')}</span>`+(seqN==null?`<button type="button" class="linkbtn px-replay" data-act="seq">${tx('▶ Send them again','▶ 再发一遍')}</button>`:''):'');
  if(apps3&&seqN!=null){const k=Math.max(0,seqN-1), x=res[k], a=x.req.app, h=held(x), was=k>0?held(res[k-1]):[], ev=was.filter(q=>!h.includes(q)), seen=res.slice(0,k).some(y=>y.req.app===a);
    const hs=h.length?andJoin(h):tx('nothing','没有');
    sum.innerHTML=seqN===0?tx('The requests arrive one after another.','请求一个接一个地到来。')
      :tx(`Request ${k+1}, app ${a}: ${x.hit>0?'hit, its system prompt is read from the cache':(seen?'miss, its system prompt was evicted':'miss, the first time for this app')}${ev.length?`; to make room, the server evicted ${andJoin(ev)}’s system prompt`:''}. Cached now: ${hs}.`,
          `请求 ${k+1}，应用 ${a}：${x.hit>0?'命中，system prompt 从缓存里读':(seen?'未命中，它的 system prompt 已经被淘汰了':'未命中，这个应用第一次来')}${ev.length?`；为了腾出空间，淘汰了应用 ${andJoin(ev)} 的 system prompt`:''}。缓存里现在有：${hs}。`);
    return;}
  if(mk){sum.textContent=tx('Request 2 arrives with the same system prompt.','请求 2 带着同样的 system prompt 来了。'); return;}
  const t=st.tot||{};
  if(ch&&st.avg===0){sum.innerHTML=tx(`None of the 8 requests finds anything in the cache: each prompt (about ${fint(aboutN(st))} tokens) is prefilled in full, and the first token comes after ${fT(st.avgW)} on average.`,`8 个请求没有一个在缓存里找到能用的：每个 prompt（约 ${fint(aboutN(st))} 个 token）都要完整 prefill，首 token 平均 ${fT(st.avgW)} 后出现。`); return;}
  if(ch){const r=st; sum.innerHTML=tx(`On average over the 8 requests, ${fint(r.avg)} of each request’s prompt tokens (about ${fint(aboutN(r))}) come from the cache, and the first token comes after ${fT(r.avgW)}.`,`8 个请求平均下来，每个请求的 prompt token（约 ${fint(aboutN(r))} 个）里有 ${fint(r.avg)} 个来自缓存，首 token 在 ${fT(r.avgW)} 后出现。`); return;}
  const memTxt=apps3?appsHeld(res[res.length-1]):'', chat=chatMode(), n=res.length;
  sum.innerHTML=(n<2?tx(`One ${chat?'turn':'request'}: nothing to reuse yet, so it waits for the full prefill.`,`只有${chat?'一轮':'一个请求'}：还没有可复用的，要等完整的 prefill。`)
    :(t.hits===0?tx(`No ${chat?'turn':'request'} hits the cache: every prompt is prefilled in full.`,`没有一${chat?'轮':'个请求'}命中缓存：每个 prompt 都要完整 prefill。`)
    :(q=>tx(`${t.hits} of the ${n} ${chat?'turns':'requests'} ${t.hits===1?'hits':'hit'} the cache, ${q<0.15?'so prefill saves almost nothing':`so prefill skips ${fracW(q)} of the tokens`}.`,`${n} ${chat?'轮':'个请求'}里 ${t.hits} ${chat?'轮':'个'}命中缓存，${q<0.15?'prefill 几乎没省下什么':`prefill 省下了${fracW(q)}的 token`}。`))(1-t.pre/t.all)))+(memTxt?tx(' ','')+memTxt:'');
}
/* the challenge's average prompt, as its task card rounds it */
const aboutN=r=>Math.round(r.avgN/50)*50;
function appsHeld(last){const full=Math.floor(S.P/BLOCK), held=['A','B','C'].filter(a=>last.apps[a]>=full);
  return held.length?tx(`After the last request the cache holds the system prompt${held.length===1?'':'s'} of ${andJoin(held)}.`,`最后一个请求之后，缓存里留着应用 ${andJoin(held)} 的 system prompt。`)
    :tx('After the last request no app’s whole system prompt is left in the cache.','最后一个请求之后，缓存里没有一个应用的 system prompt 是完整的。');}
const cacheTip=h=>h.length?tx(`After it, the cache holds the system prompt${h.length===1?'':'s'} of ${andJoin(h)}.`,`这个请求之后，缓存里有应用 ${andJoin(h)} 的 system prompt。`):tx('After it, no app’s whole system prompt is in the cache.','这个请求之后，缓存里没有一个应用的 system prompt 是完整的。');
/* part 6: the 8 requests arrive one at a time after a pick, so the learner sees which option does what to each request */
function stopSeq(){clearTimeout(seqT); seqT=0; seqN=null; seqCur=null;}
/* from: the first request that arrives; the rows before it are shown at once (R3.20: only the part that differs is played) */
function startSeq(from){
  stopSeq(); if(!app) return;
  if(A.reduceMotion()){seqSeen=seqKeys(); render(); return;}
  /* the first request arrives in the same frame as the start: the badge, the strip and the cards never show the last request's result
     while later rows still wait (review r1 #9) */
  const n=R().res.length; from=Math.max(0,Math.min(n-1,from||0)); seqCur=seqKeys(); seqN=from+1; sel=from; render();
  const end=()=>{seqN=null; seqCur=null; seqSeen=seqKeys(); seqDone.add(seqSeen.join('|')); seqDoneT.add(token()); sel=n-1; render(); updateRail(); if(mode==='guided'&&STEPS[step].id==='cost') requestAnimationFrame(()=>{if(app) revealBox($('#px-rail [data-obs]',app));});};
  const tick=()=>{if(!app||seqN==null) return; seqN++; sel=seqN-1; render();
    seqT=setTimeout(seqN>=n?end:tick,seqN>=n?700:650);};
  seqT=setTimeout(seqN>=n?end:tick,seqN>=n?700:650);}
/* R3.20: each row as drawn (app, prompt, hit, first token, the apps whose system prompts stay cached, its blocks), to tell which rows a
   change would redraw */
function seqKeys(s=S){const full=Math.floor(s.P/BLOCK); return run(s).res.map(x=>[x.req.app,x.N,x.hit,x.w.t,['A','B','C'].filter(a=>x.apps[a]>=full).join(''),x.state.join('')].join(':'));}
/* the first row of b that is not the same as in a: a.length (b longer) or b.length (nothing differs) when a is a prefix of b or b of a */
function firstDiff(a,b){if(!a) return 0; const m=Math.min(a.length,b.length); for(let i=0;i<m;i++) if(a[i]!==b[i]) return i; return b.length>a.length?a.length:b.length;}
/* R3.20: after a pick (or on entering part 6), the requests arrive again only from the first row that changes; when no row changes, the
   finished rows stay (“Send them again” replays them). was: the rows on screen before the change; a run in progress keeps the rows
   it has already shown while they stay the same */
function playSeq(was){
  const k=seqKeys(), done=seqDone.has(k.join('|'));
  if(seqN!=null&&seqCur){const same=seqCur.join('|')===k.join('|'), d=firstDiff(seqCur,k); seqCur=k; if(done&&!same){seqAtEnd(k); return;} if(d>=seqN) return; startSeq(d); return;}
  const d=firstDiff(was,k);
  if(d>=k.length){seqSeen=k; return;}
  if(done){seqAtEnd(k); return;}
  startSeq(d);}
/* R3.20: rows already watched to the end (A → B → A, part 6 entered again with them) show finished at once; “Send them again” replays */
function seqAtEnd(k){const run=seqN!=null; stopSeq(); seqSeen=k; if(run){sel=k.length-1; render(); updateRail();}}
function rowTip(r,i,lb){
  if(r.hit===0) return tx(`${lb}: ${fint(r.N)} prompt tokens, none from the cache, all computed by prefill. First token after ${fT(r.w.t)}, the same as without the cache.`,
                         `${lb}：${fint(r.N)} 个 prompt token，没有一个来自缓存，全部由 prefill 现算。首 token 在 ${fT(r.w.t)} 后出现，和不用缓存时一样。`);
  return tx(`${lb}: ${fint(r.N)} prompt tokens, ${fint(r.hit)} from the cache (${fint(r.hitBlocks)} blocks), ${fint(r.N-r.hit)} computed by prefill. First token after ${fT(r.w.t)}; without the cache ${fT(r.wo.t)}.`,
            `${lb}：${fint(r.N)} 个 prompt token，${fint(r.hit)} 个来自缓存（${fint(r.hitBlocks)} 个 block），${fint(r.N-r.hit)} 个由 prefill 现算。首 token 在 ${fT(r.w.t)} 后出现；不用缓存要 ${fT(r.wo.t)}。`);}
function runsOf(state){let h='', st=-1, n=0; const flush=()=>{if(n) h+=`<i class="px-s${st}" style="flex:${n}"></i>`;};
  for(let k=0;k<state.length;k++){if(state[k]!==st){flush(); st=state[k]; n=0;} n++;} flush(); return `<span class="px-runs">${h}</span>`;}
/* token boundaries of the displayed segments (consecutive chat-history pieces merge into one) */
function dispSegs(segs){const out=[]; let t=0;
  segs.forEach(s=>{const k=s.part||s.k; const last=out[out.length-1]; if(last&&last.k===k&&k==='hist'){last.e+=s.n;} else out.push({k,s:t,e:t+s.n}); t+=s.n;}); return out;}
function segBounds(segs){const d=dispSegs(segs); return [0].concat(d.map(x=>x.e));}
function drawStrip(){
  const x=selRes(), i=selIdx(), box=$('#px-strip',app), cells=$('#px-cells',app), band=$('#px-band',app), miss=$('#px-miss',app), cap=$('#px-strip-cap',app), foot=$('#px-strip-foot',app); if(!box||!x) return;
  const W=cells.clientWidth||box.clientWidth||480, nb=x.total, maxC=Math.max(20,Math.floor(W/5.5));
  let g=1; while(Math.ceil(nb/g)>maxC) g*=2;
  const nc=Math.ceil(nb/g), anim=sweepNext&&!A.reduceMotion(); sweepNext=false;
  let h='';
  for(let c=0;c<nc;c++){const a=c*g, b=Math.min(nb,a+g); let sp='', st=-1, n=0; const fl=()=>{if(n) sp+=`<i class="px-s${st}" style="flex:${n}"></i>`;};
    for(let k=a;k<b;k++){if(x.state[k]!==st){fl(); st=x.state[k]; n=0;} n++;} fl();
    h+=`<span class="px-cell" style="--i:${c}">${sp}</span>`;}
  cells.innerHTML=h; cells.classList.toggle('px-sweep',anim);
  if(anim){cells.classList.add('px-pre-sweep'); requestAnimationFrame(()=>requestAnimationFrame(()=>{cells.classList.remove('px-pre-sweep');}));}
  const TT=nb*BLOCK, ds=dispSegs(x.req.segs);
  band.innerHTML=ds.map((d,j)=>{const l=d.s/TT*100, w=(d.e-d.s)/TT*100, nm=segName(d.k);
    return `<span class="px-bs px-b-${d.k}${j%2?' px-alt':''}" style="left:${l.toFixed(3)}%;width:${w.toFixed(3)}%" tabindex="0" data-tipx="${esc(tx(`${nm[0].toUpperCase()+nm.slice(1)}: ${fint(d.e-d.s)} tokens, tokens ${fint(d.s+1)} to ${fint(d.e)} of the prompt.`,`${nm}：${fint(d.e-d.s)} 个 token，是 prompt 的第 ${fint(d.s+1)} 到第 ${fint(d.e)} 个 token。`))}"><em>${esc(nm)}</em></span>`;}).join('');
  // a label that does not fit its segment moves under the strip (pinBand); the list under the strip names every segment too
  requestAnimationFrame(()=>pinBand(band));
  const hitB=x.hitBlocks, mx=hitB/nb*100;
  miss.hidden=hitB>=nb; if(!miss.hidden) miss.style.left=mx.toFixed(3)+'%';
  cap.textContent=tx(`${reqName(i)}: ${fint(x.N)} prompt tokens in ${fint(nb)} blocks of 16 tokens (one cell: ${g===1?'one block':`${g} blocks`})`,`${reqName(i)}：${fint(x.N)} 个 prompt token，${fint(nb)} 个 block，每个 16 个 token（每格 ${g===1?'一个 block':`${g} 个 block`}）`);
  const same=x.state.filter(s=>s===2).length;
  const segList=ds.map(d=>`${segName(d.k)} ${fint(d.e-d.s)}`).join(tx('; ','；'));
  foot.innerHTML=(x.hit===0?tx(`First miss at block 1: nothing is cached for this prompt’s start. `,`第 1 个 block 就未命中：这个 prompt 的开头不在缓存里。`)
     :tx(`First miss at block ${fint(hitB+1)}, from token ${fint(x.hit+1)} on. `,`从第 ${fint(hitB+1)} 个 block（第 ${fint(x.hit+1)} 个 token）起未命中。`))+
    (same?tx(`${fint(same)} of the computed blocks hold tokens that are cached after another prefix. `,`现算的 block 里有 ${fint(same)} 个，它们的 token 缓存里有，只是前面接的内容不同。`):'')+
    `<span class="muted">${tx('Parts, in tokens: ','各部分的 token 数：')}${esc(segList)}.</span>`.replace(/\.<\/span>$/,tx('.</span>','。</span>'));
  const grayTx=tx(`gray${same?', hatched when the same tokens are cached after a different prefix':''}`,`灰色${same?'；斜纹表示同样的 token 在缓存里，但前面接的不同':''}`);
  cells.dataset.tipx=hitB===0?tx(`${reqName(i)}: all ${fint(nb)} blocks were computed (${grayTx}); none came from the cache.`,`${reqName(i)}：${fint(nb)} 个 block 全部现算（${grayTx}），没有一个来自缓存。`)
    :tx(`${reqName(i)}: ${fint(hitB)} of ${fint(nb)} blocks came from the cache (amber), ${fint(nb-hitB)} were computed (${grayTx}).`,`${reqName(i)}：${fint(nb)} 个 block 里 ${fint(hitB)} 个来自缓存（琥珀色），${fint(nb-hitB)} 个现算（${grayTx}）。`);
  cells.setAttribute('aria-label',cells.dataset.tipx);
}
/* a part too narrow for its label gets it on a short leader line: under the strip, or above it when it would overlap the label
   before it; parts less than 14 px apart share one label */
const PIN_GAP=5, PIN_H=15;
function pinBand(band){
  const lo=app&&$('#px-pins',app), hi=app&&$('#px-pins-top',app); if(!lo||!hi||!band.isConnected) return;
  const W=band.clientWidth, items=[];
  $$('.px-bs',band).forEach(s=>{const e=s.querySelector('em'); if(!e) return; const fits=e.scrollWidth<=s.clientWidth-4; e.style.visibility=fits?'visible':'hidden';
    if(!fits) items.push({x:s.offsetLeft+s.offsetWidth/2,t:e.textContent,tip:s.dataset.tipx});});
  const g=[]; items.forEach(it=>{const l=g[g.length-1]; if(l&&it.x-l.xs[l.xs.length-1]<14){l.t.push(it.t); l.tip.push(it.tip); l.xs.push(it.x);} else g.push({t:[it.t],tip:[it.tip],xs:[it.x]});});
  const lab=p=>`<span class="px-pin" data-tipx="${esc(p.tip.join(tx(' ','')))}">${esc(p.t.join(tx(', ','、')))}</span>`;
  lo.innerHTML=g.map(lab).join(''); const ws=$$('.px-pin',lo).map(el=>Math.ceil(el.offsetWidth));
  const end=[-1e9,-1e9], out=['',''];
  g.forEach((p,j)=>{const w=ws[j], cx=(p.xs[0]+p.xs[p.xs.length-1])/2, left=Math.round(Math.max(0,Math.min(W-w,cx-w/2)));
    const r=left>=end[0]+6?0:(left>=end[1]+6?1:(end[0]<=end[1]?0:1)); end[r]=left+w;
    out[r]+=p.xs.map(x=>`<i style="left:${x.toFixed(1)}px;${r?'bottom:-2px':'top:0'};height:${PIN_GAP+(r?4:2)}px"></i>`).join('')+lab(p).replace('<span ',`<span style="left:${left}px;top:${r?0:PIN_GAP}px" `);});
  lo.innerHTML=out[0]; hi.innerHTML=out[1];
  lo.style.height=out[0]?(PIN_GAP+PIN_H)+'px':'0px'; hi.style.height=out[1]?(PIN_H+PIN_GAP)+'px':'0px';
}
function niceStep(span,n){const raw=span/(n||5), p=Math.pow(10,Math.floor(Math.log10(raw))), m=raw/p; return (m<1.5?1:(m<3.5?2:(m<7.5?5:10)))*p;}
const TL_K=20;
function drawTL(){
  const x=selRes(), i=selIdx(), tl=$('#px-tl',app); if(!tl||!x) return;
  const ticks=[]; let t=0; for(let k=0;k<TL_K;k++){t+=decode(x.N+k).t; ticks.push(t);}
  const span=(x.wo.t+t)*1.02, X=v=>Math.max(0,Math.min(100,v/span*100)), pos=v=>X(v).toFixed(3)+'%';
  const first=i===0&&mode!=='challenge';
  [['wo',x.wo.t,tx('Without prefix caching','不用 prefix caching')],['w',x.w.t,first?tx('With prefix caching (nothing cached yet)','用 prefix caching（缓存里还没有东西）'):tx('With prefix caching','用 prefix caching')]].forEach(([k,tt,nm])=>{
    const row=$(`.px-tlrow[data-k="${k}"]`,tl); $('.px-tln',row).textContent=nm; $('.px-tlv',row).textContent=tx(`first token at ${fT(tt)}`,`${fT(tt)} 出首 token`);
    $('.px-pre',row).style.width=pos(tt); $('.px-first',row).style.left=pos(tt);
    $('.px-ticks',row).innerHTML=ticks.map(d=>`<i style="left:${pos(tt+d)}"></i>`).join('');
    const tr=$('.px-track',row), comp=k==='wo'?x.N:x.N-x.hit;
    tr.dataset.tipx=k==='wo'?tx(`Without the cache, prefill computes all ${fint(x.N)} prompt tokens in one step: ${fT(tt)}, then the first token (thick line). Each thin tick is one more token, one decode step each.`,`不用缓存时，prefill 在一步里算完全部 ${fint(x.N)} 个 prompt token：${fT(tt)}，然后出首 token（粗线）。每条细线是又一个 token，各用一个 decode 步。`)
      :(x.hit===0?tx(`Nothing in the cache matches the start of this prompt, so prefill computes all ${fint(comp)} tokens, as without the cache: ${fT(tt)} to the first token.`,`缓存里没有和这个 prompt 开头对得上的 block，所以 prefill 仍要算全部 ${fint(comp)} 个 token，和不用缓存时一样：${fT(tt)} 出首 token。`)
      :tx(`With the cache, prefill computes only ${fint(comp)} tokens and reads the other ${fint(x.hit)} from the cache: ${fT(tt)} to the first token. The decode steps after it take as long as without the cache.`,`用了缓存，prefill 只算 ${fint(comp)} 个 token，其余 ${fint(x.hit)} 个从缓存读：${fT(tt)} 出首 token。之后的 decode 步和不用缓存时一样长。`));
    tr.setAttribute('aria-label',tr.dataset.tipx);});
  const tw=(tl.clientWidth||500), spanMs=span*1e3, st=niceStep(spanMs,Math.max(2,Math.min(5,Math.floor(tw/90)))), inS=spanMs>=2000, lab=v=>inS?fixd(v/1000,st>=1000?0:1):fint(v);
  const vs=[]; for(let v=0;v<=spanMs*0.93;v+=st) vs.push(v);
  $('.px-axis',tl).innerHTML=vs.map((v,j)=>`<span style="left:${(v/spanMs*100).toFixed(3)}%">${lab(v)}${j===vs.length-1?NS+(inS?'s':'ms'):''}</span>`).join('');
  $('#px-tl-cap',app).textContent=tx(`${reqName(i)}: the wait for its first token, then ${TL_K} more tokens`,`${reqName(i)}：等首 token 的时间，之后再出 ${TL_K} 个 token`);
}
/* =====================================================================
   part 5: request 1 and request 2 block by block, and the sweep over request 2's blocks
   ===================================================================== */
/* one strip's cells: g blocks each, the same g for both strips so that they line up; a cell is as wide as the blocks it holds */
function pcells(state,g){let h='';
  for(let a=0;a<state.length;a+=g){const b=Math.min(state.length,a+g); let sp='', st=-1, n=0; const fl=()=>{if(n) sp+=`<i class="px-s${st}" style="flex:${n}"></i>`;};
    for(let k=a;k<b;k++){if(state[k]!==st){fl(); st=state[k]; n=0;} n++;} fl();
    h+=`<span class="px-cell" style="flex:${b-a} 1 0">${sp}</span>`;}
  return h;}
/* where block b (from 0), a fraction f of the way through it, sits in a strip, in px: read from request 1's cells, which always exist */
function pX(b,f){const cells=app?$('#px-pc1',app).children:[], c=Math.floor(b/otG), el=cells[Math.min(c,cells.length-1)]; if(!el) return 0;
  const n=parseFloat(el.style.flex)||otG, i=Math.min(n,b-c*otG+f); return el.offsetLeft+i/n*el.offsetWidth;}
const tokX=t=>pX(Math.floor((t-1)/BLOCK),((t-1)%BLOCK+0.5)/BLOCK);  // token t (from 1), in the middle of its slot
const otBlk=v=>Math.floor((otTok(v)-1)/BLOCK)+1;
/* a position button's hover (none on the position picked: its result is on the stage, and a tooltip there would cover request 2's strip,
   review r1 #5); once tried, it adds the conclusion, not a time (R3.17) */
function otPosTip(v){
  return tx(`Click to change token ${fint(otTok(v))} of request 2’s system prompt, ${L(OT_W[v])}, in block ${fint(otBlk(v))}.`,`点击改请求 2 的 system prompt 里${L(OT_W[v])}的第 ${fint(otTok(v))} 个 token，它在第 ${fint(otBlk(v))} 个 block 里。`)+
    (otPred&&otTried.has(v)?(otRun(v).r2.hit===0?tx(' Request 2 then finds none of its system prompt in the cache.','这时请求 2 在缓存里一点也找不到它的 system prompt。')
      :tx(` Request 2 then finds ${fracW(otRun(v).r2.hit/BASE.P)} of its system prompt in the cache.`,`这时请求 2 在缓存里找到它的 system prompt 的${fracW(otRun(v).r2.hit/BASE.P)}。`)):'');}
/* the less than 0.01 GiB of a single block, so that a count of 1 never reads “0.00 GiB” (review r1 #13) */
const gibC=b=>b<=0?'0'+NS+'GiB':(b<0.005*A.GiB?tx('< 0.01','不到 0.01')+NS+'GiB':gib2(b));
function drawPair(){
  if(!app||!isOT()) return;
  const o=otRun(otPos||'middle'), r1=o.r1, nb=r1.total, t1=$('#px-pt1',app), W=t1.clientWidth||480, open=otOpen(), r2=o.r2;
  const maxC=Math.max(20,Math.floor(W/5.5)); let g=1; while(Math.ceil(nb/g)>maxC) g*=2; otG=g;
  const lastT=r1.N-(nb-1)*BLOCK;
  $('#px-pair-cap',app).textContent=tx(`Request 1 was computed on an empty cache, so its blocks are gray, and they are now in the cache. Request 2 is lined up under it block by block: ${fint(nb)} blocks of ${BLOCK} tokens, the last one holding ${lastT} (one cell: ${g===1?'one block':`${g} blocks`}).`,
                                       `请求 1 是在空缓存上算的，所以它的 block 是灰色，现在都在缓存里。请求 2 在它下面逐个 block 对齐：共 ${fint(nb)} 个 block，每个 ${BLOCK} 个 token，最后一个只有 ${lastT} 个（每格 ${g===1?'一个 block':`${g} 个 block`}）。`);
  $('#px-pl1',app).textContent=tx('Request 1','请求 1'); $('#px-pl2',app).textContent=tx('Request 2','请求 2');
  $('#px-pc1',app).innerHTML=pcells(r1.state,g);
  $('#px-pc2',app).innerHTML=open?pcells(r2.state,g):'';
  /* the two prompts' parts (the same lengths in both): the system prompt, then the question; a label too long for its part sits above it */
  const band=$('#px-pband',app), pins=$('#px-ppins',app), ds=dispSegs(r1.req.segs);
  band.innerHTML=ds.map((d,j)=>{const l=pX(Math.floor(d.s/BLOCK),(d.s%BLOCK)/BLOCK), r=d.e>=r1.N?W:pX(Math.floor(d.e/BLOCK),(d.e%BLOCK)/BLOCK), nm=segName(d.k);
    const tip=d.k==='pre'?tx(`The system prompt takes tokens 1 to ${fint(d.e)} of both prompts. Request 2’s system prompt differs from request 1’s in one token only.`,`system prompt 占两个 prompt 的第 1 到第 ${fint(d.e)} 个 token。请求 2 的 system prompt 和请求 1 的只差一个 token。`)
      :tx(`The question takes tokens ${fint(d.s+1)} to ${fint(d.e)}, ${fint(d.e-d.s)} tokens. Each request has its own question, so both requests compute these blocks.`,`问题占第 ${fint(d.s+1)} 到第 ${fint(d.e)} 个 token，共 ${fint(d.e-d.s)} 个。每个请求的问题都不一样，所以这几个 block 两个请求都要现算。`);
    return `<span class="px-bs px-b-${d.k}${j%2?' px-alt':''}" style="left:${l.toFixed(1)}px;width:${Math.max(2,r-l).toFixed(1)}px" tabindex="0" data-tipx="${esc(tip)}"><em>${esc(nm)}</em></span>`;}).join('');
  let ph='';
  $$('.px-bs',band).forEach(s=>{const e=s.querySelector('em'), fits=e.scrollWidth<=s.clientWidth-4; e.style.visibility=fits?'visible':'hidden';
    if(!fits){const cx=s.offsetLeft+s.offsetWidth/2; ph+=`<i style="left:${cx.toFixed(1)}px"></i><span class="px-pin" data-cx="${cx.toFixed(1)}" data-tipx="${esc(s.dataset.tipx)}">${esc(e.textContent)}</span>`;}});
  pins.innerHTML=ph; pins.style.height=ph?(PIN_H+PIN_GAP)+'px':'0px';
  $$('.px-pin',pins).forEach(p=>{const cx=+p.dataset.cx, w=p.offsetWidth; p.style.left=Math.round(Math.max(0,Math.min(W-w,cx-w/2)))+'px';});
  /* tooltips of the strips */
  const t2=$('#px-pt2',app), hb=r2.hitBlocks, hatch=o.same?tx(`; the hatched ones hold the same ${BLOCK} tokens as request 1’s blocks but have different keys`,`；斜纹的 block 和请求 1 的 block 装着同样的 ${BLOCK} 个 token，但 key 不同`):'';
  t1.dataset.tipx=tx(`Request 1 was computed on an empty cache, so all ${fint(nb)} of its blocks are gray. Its full blocks are now in the cache, each under its own key.`,`请求 1 是在空缓存上算的，所以它的 ${fint(nb)} 个 block 都是灰色。它写满的 block 现在都在缓存里，各有自己的 key。`);
  t2.dataset.tipx=!otPos?tx('Request 2 has not been sent yet. First pick where its one changed token sits.','请求 2 还没有发出。先选它那个改动的 token 放在哪。')
    :(!open?tx('Request 2’s blocks are shown after you answer the question in this part.','回答这一部分的问题之后，才显示请求 2 的 block。')
      :(hb===0?tx(`Request 2 found none of its ${fint(nb)} blocks in the cache. The change is in block 1, so request 2 computed every block${hatch}.`,`请求 2 的 ${fint(nb)} 个 block 在缓存里一个也没找到。改动在第 1 个 block，所以请求 2 每个 block 都要现算${hatch}。`)
        :tx(`Request 2 found blocks 1 to ${fint(hb)} in the cache (amber). From block ${fint(hb+1)}, the changed one, it computed all ${fint(nb-hb)} blocks${hatch}.`,`请求 2 在缓存里找到了第 1 到第 ${fint(hb)} 个 block（琥珀色）。从改动所在的第 ${fint(hb+1)} 个 block 起，${fint(nb-hb)} 个 block 它全部要现算${hatch}。`)));
  t1.setAttribute('aria-label',t1.dataset.tipx); t2.setAttribute('aria-label',t2.dataset.tipx);
  /* the three positions under request 2, each with a caret at its token. They are built once and then updated in place, so a button keeps
     its focus after a click */
  const pp=$('#px-ppos',app);
  $('#px-pl3',app).textContent=tx('Changed token','改动的 token');
  if(!pp.children.length) pp.innerHTML=OT_POS.map(p=>`<i class="px-pcaret" data-v="${p.v}" aria-hidden="true"></i><button type="button" class="px-pbtn" data-otpos="${p.v}" aria-label="${esc(otName(p.v))}">`+
    tx(`<span class="px-pw">token </span>${fint(p.t)}`,`<span class="px-pw">第 </span>${fint(p.t)}<span class="px-pw"> 个</span>`)+`</button>`).join('');
  const bx=[]; OT_POS.forEach(p=>{const x=tokX(p.t), b=$(`.px-pbtn[data-otpos="${p.v}"]`,pp), c=$(`.px-pcaret[data-v="${p.v}"]`,pp), on=otPos===p.v, w=b.offsetWidth;
    b.setAttribute('aria-pressed',String(on)); c.classList.toggle('on',on);
    if(on) b.removeAttribute('data-tipx'); else b.dataset.tipx=otPosTip(p.v);
    c.style.left=x.toFixed(1)+'px'; bx.push({b,w,l:Math.max(0,Math.min(W-w,x-w/2))});});
  for(let j=1;j<bx.length;j++) if(bx[j].l<bx[j-1].l+bx[j-1].w+4) bx[j].l=bx[j-1].l+bx[j-1].w+4;       // never on top of each other
  for(let j=bx.length-1;j>=0;j--){if(bx[j].l+bx[j].w>W) bx[j].l=W-bx[j].w; if(j&&bx[j-1].l+bx[j-1].w+4>bx[j].l) bx[j-1].l=bx[j].l-bx[j-1].w-4;}
  bx.forEach(x=>{x.b.style.left=Math.max(0,x.l).toFixed(1)+'px';});
  /* on a phone the question sits right under the positions, so the learner answers it where the strips and the counters are in view, and
     the lookup plays in front of them (review r1 #1, #2); on a wide screen it stays in the card */
  const qs=$('#px-otqs',app), qh=otPos&&!otPred?otQHTML():''; if(qs._h!==qh){qs.innerHTML=qh; qs._h=qh;}
  /* the key comparison between the strips: one block of the hit ("=", the same key) and one after the change ("≠": the same tokens under
     a different key, or at the end the changed block itself), from the lab's own block keys */
  const lane=$('#px-lane',app); $$('.px-kp',lane).forEach(e=>e.remove());
  if(open){
    const stack=(b,eq)=>{const [a,c]=keyPair(o.K1.key[b],o.K2.key[b]), t=otTok(otPos);
      const tip=eq?tx(`Block ${fint(b+1)} holds the same ${BLOCK} tokens in both requests, and every token before it is the same too, so both requests have the same key, #${a}. Request 2 reads this block from the cache.`,`第 ${fint(b+1)} 个 block 在两个请求里的 ${BLOCK} 个 token 相同，之前的 token 也全都相同，所以两个请求的 key 一样，都是 #${a}。请求 2 直接从缓存读这个 block。`)
        :(o.r2.state[b]===2?tx(`Block ${fint(b+1)} holds the same ${BLOCK} tokens in both requests, but request 1’s key is #${a} and request 2’s is #${c}, because a key includes every token before its block and token ${fint(t)} differs. So request 2 computes this block again.`,`第 ${fint(b+1)} 个 block 在两个请求里的 ${BLOCK} 个 token 一模一样，但请求 1 的 key 是 #${a}，请求 2 的是 #${c}，因为 key 包含这个 block 之前的全部 token，而第 ${fint(t)} 个 token 不同。所以请求 2 要重算这个 block。`)
          :tx(`Block ${fint(b+1)} holds the changed token ${fint(t)}, so request 1’s key (#${a}) and request 2’s key (#${c}) differ. Request 2 computes this block again, and every block after it.`,`第 ${fint(b+1)} 个 block 里有改动的第 ${fint(t)} 个 token，所以请求 1 的 key（#${a}）和请求 2 的 key（#${c}）不同。请求 2 要重算这个 block，以及它之后的每个 block。`));
      return `<span class="px-kp${eq?' eq':' ne'}" data-kb="${b}" tabindex="0" data-tipx="${esc(tip)}" aria-label="${esc(tip)}" hidden><b>#${a}</b><i>${eq?'=':'≠'}</i><b>#${c}</b></span>`;};
    lane.insertAdjacentHTML('beforeend',(hb>0?stack(Math.floor(hb/2),true):'')+stack(o.same>0?o.s0+Math.floor(o.same/2):hb,false));
    /* each pair sits on its block, but never on the first-miss line (at the end position the changed block is next to it) nor on the other pair */
    const mx=pX(hb,0), ks=$$('.px-kp',lane).map(e=>{const w=e.offsetWidth||44; let l=Math.max(0,Math.min(W-w,pX(+e.dataset.kb,0.5)-w/2));
      if(l<mx+4&&l+w>mx-4) l=mx-w-4>=0?mx-w-4:mx+4; return {e,w,l};});
    if(ks.length===2&&ks[0].l+ks[0].w+4>ks[1].l) ks[0].l=Math.max(0,ks[1].l-ks[0].w-4);
    ks.forEach(k=>{k.e.style.left=k.l.toFixed(1)+'px';});
    const sl=$('#px-same',app);
    sl.innerHTML=o.same>0?`<i class="sw px-sw-same"></i><b class="px-ne">≠</b>${esc(tx('Same tokens, different key: a block’s key includes every token before it.','token 相同，key 却不同：block 的 key 包含它之前的全部 token。'))}`
      :`<b class="px-ne">≠</b>${esc(tx('One token differs, so this block’s key and every key after it differ.','有一个 token 不同，所以这个 block 和之后的 key 都不同。'))}`;
  } else $('#px-same',app).innerHTML='';
  /* the counters: the system prompt's tokens recomputed, the time to first token, the system prompt's blocks now stored twice (review r1 #3) */
  const cr=$('#px-ctrs',app);
  if(!cr.children.length) cr.innerHTML=['rec','t','nw'].map(k=>`<div class="px-ctr" data-k="${k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="px-cbar" aria-hidden="true"><i></i></div><div class="sub"></div></div>`).join('');
  const lab={rec:tx('System-prompt tokens recomputed','重算的 system prompt token'),t:tx('Time to first token','首 token 延迟'),nw:tx('System-prompt blocks stored twice','存了两份的 system prompt block')};
  const tip={rec:tx(`How many of the system prompt’s tokens request 2 recomputes: every token from its first missed block on. A block that request 2 finds in the cache costs only a read of its keys and values. The question’s ${fint(BASE.O)} tokens are new at every position, so this count leaves them out.`,`请求 2 要重算 system prompt 里的多少个 token：从它第一个未命中的 block 起的每个 token。请求 2 在缓存里找到的 block，只要读出它的 key 和 value。问题的 ${fint(BASE.O)} 个 token 在每个位置都是新的，所以这里不算它们。`),
    t:tx(`Request 2’s time to first token by this lab’s step model, for the blocks that it has looked up so far. A cached block adds only the time to read its keys and values, and a recomputed block adds its arithmetic too. Once the last block is looked up, this is request 2’s time to first token.`,`按这个 Lab 的耗时模型，请求 2 到目前查过的 block 要花的首 token 时间。缓存的 block 只加上读 key 和 value 的时间，重算的 block 还要加上它的计算。查完最后一个 block，这就是请求 2 的首 token 延迟。`),
    nw:tx(`How many of the system prompt’s blocks the cache now holds twice. Request 2 stores each block that it recomputes under its own new key, next to request 1’s copy, and each block takes ${BLOCK} × 128 KiB = ${fint(BLOCK*KV_TOK/A.MiB)} MiB of GPU memory.`,`缓存里现在有多少个 system prompt 的 block 存了两份。请求 2 重算的每个 block 都用自己新的 key 存进缓存，和请求 1 的那份并排；每个 block 占 ${BLOCK} × 128 KiB = ${fint(BLOCK*KV_TOK/A.MiB)} MiB 显存。`)};
  $$('.px-ctr',cr).forEach(c=>{const k=c.dataset.k; $('.k',c).textContent=lab[k]; c.dataset.tipx=tip[k];});
  drawPairFrame();
}
/* block by block: where each click stops, and what the next click shows (review r1 #6): all the blocks before the change at once, the
   changed block, the block after it, the rest of the system prompt, the question */
function otStops(o){const r=o.r2, hb=r.hitBlocks, pb=o.preB, T=r.total, s=[];
  if(hb>0) s.push(hb); s.push(hb+1); if(hb+2<=pb) s.push(hb+2); if(pb>s[s.length-1]) s.push(pb); if(T>s[s.length-1]) s.push(T); return s;}
function otStepLabel(o,next){const hb=o.r2.hitBlocks, pb=o.preB;
  if(next==null) return tx('▸ Block by block','▸ 逐个 block 看');
  if(next===hb) return tx(`Show blocks 1 to ${fint(hb)} ▸`,`看第 1 到第 ${fint(hb)} 个 block ▸`);
  if(next===hb+1) return tx('Show the changed block ▸','看改动所在的 block ▸');
  if(next===hb+2) return tx('Show the block after it ▸','看它后面的一个 block ▸');
  if(next===pb) return tx('Show the rest of the system prompt ▸','看 system prompt 剩下的 block ▸');
  return tx('Show the question’s blocks ▸','看问题的 block ▸');}
/* what the lookup has done so far, in words; at the stops of block by block, also what the last click showed */
function otProgress(o,k){const r=o.r2, T=r.total, hb=r.hitBlocks, pb=o.preB;
  if(k===0) return tx('Request 2 has not looked up any block yet.','请求 2 还没有查找任何 block。');
  let t=tx(`Request 2 has looked up blocks 1 to ${fint(k)} of its ${fint(T)}.`,`请求 2 已经查找了第 1 到第 ${fint(k)} 个 block（共 ${fint(T)} 个）。`);
  if(!swStep) return t;
  if(k===hb) t+=tx(' All of them hit, because their keys are in the cache.','它们全部命中，因为它们的 key 都在缓存里。');
  else if(k===hb+1) t+=tx(` Block ${fint(hb+1)}, the changed one, misses.`,`改动所在的第 ${fint(hb+1)} 个 block 未命中。`);
  else if(k===hb+2&&o.same) t+=tx(` Block ${fint(hb+2)} misses too, although its tokens are the same as request 1’s.`,`第 ${fint(hb+2)} 个 block 也未命中，尽管它的 token 和请求 1 的一样。`);
  else if(k===pb) t+=tx(' Every block of the system prompt after the change missed.','改动之后，system prompt 的每个 block 都未命中。');
  return t;}
/* what changes while request 2's blocks are looked up: the part not looked up yet, the first miss and the arrow to the end, the key
   comparison, the counters, the progress; everything from swK (null: all of them) */
function drawPairFrame(){
  if(!app||!isOT()) return;
  const open=otOpen(), o=otRun(otPos||'middle'), r2=o.r2, T=r2.total, hb=r2.hitBlocks, W=$('#px-pt1',app).clientWidth;
  const k=open?(swK==null?T:swK):0;
  if(open&&swK==null) otTried.add(otPos);                             // the end state is on screen: this position counts as seen
  const cv=$('#px-pcover',app), cs=$('span',cv);
  if(!open){cv.hidden=false; cv.classList.add('wait'); cv.style.left='0px'; cs.textContent=otPos?tx('It appears after your prediction.','预测之后显示请求 2。'):tx('Request 2 has not been sent yet.','请求 2 还没有发出。');}
  else if(k>=T) cv.hidden=true;
  else {cv.hidden=false; cv.classList.remove('wait'); cv.style.left=pX(k,0).toFixed(1)+'px'; cs.textContent='';}
  const tk=$('#px-ptick',app); tk.hidden=!otPos; if(otPos) tk.style.left=tokX(otTok(otPos)).toFixed(1)+'px';
  const ms=$('#px-pmiss',app), ar=$('#px-larrow',app), miss=open&&k>hb;
  ms.hidden=!miss; ar.hidden=!miss;
  if(miss){const mx=pX(hb,0), fx=k>=T?W:pX(k,0); ms.style.left=mx.toFixed(1)+'px'; ar.style.left=mx.toFixed(1)+'px'; ar.style.width=Math.max(0,fx-mx).toFixed(1)+'px'; ar.classList.toggle('end',k>=T);}
  $$('#px-lane .px-kp',app).forEach(e=>{e.hidden=!(open&&k>+e.dataset.kb);});
  $('#px-same',app).style.visibility=open&&k>(o.same>0?o.s0:hb)?'visible':'hidden';
  const a=open?otAt(otPos,k):null, wo=r2.wo.t;
  const set=(key,v,q,sub)=>{const c=$(`#px-ctrs .px-ctr[data-k="${key}"]`,app); if(!c) return; $('.v',c).textContent=v; $('.px-cbar i',c).style.width=(Math.max(0,Math.min(1,q))*100).toFixed(2)+'%'; $('.sub',c).textContent=sub;};
  set('rec',a?fint(a.rec):'?',a?a.rec/BASE.P:0,tx(`of ${fint(BASE.P)}; the question is left out`,`共 ${fint(BASE.P)} 个，问题不算在内`));
  set('t',a?(a.t?fT(a.t.t):'—'):'?',a&&a.t?a.t.t/wo:0,a&&swK!=null?tx(`so far, for ${fint(k)} block${k===1?'':'s'} looked up`,`到目前为止，已查 ${fint(k)} 个 block`):tx(`${fT(wo)} without the cache`,`不用缓存要 ${fT(wo)}`));
  set('nw',a?fint(a.nw):'?',a?a.nw/o.preB:0,a?(a.nw?tx(`${gibC(a.bytes)} more GPU memory`,`多占${/^[一-鿿]/.test(gibC(a.bytes))?'':' '}${gibC(a.bytes)} 显存`):tx('no more GPU memory yet','还没多占显存')):tx('in GPU memory','占的显存'));
  const sb=$('#px-otstep',app), pg=$('#px-prog',app), st=otStops(o), next=swStep?st.find(x=>x>k):null;
  sb.hidden=!open; sb.textContent=otStepLabel(o,swStep?next:null); $('#px-otreplay',app).hidden=!open;
  sb.setAttribute('aria-label',swStep?sb.textContent.replace(/ ▸$/,''):tx('Look up request 2’s blocks again, a few at a time','重新查找请求 2 的 block，一次看一段'));
  /* on a phone the result sits in the card below the chart: once the lookup is over, a line under the chart takes the learner there (R3.2) */
  const hint=open&&swK==null&&innerWidth<=760&&!otStepEnd;
  pg.innerHTML=open&&swK!=null?esc(otProgress(o,k)):(hint?`<button type="button" class="linkbtn px-otres" data-act="ot-res" data-tipx="${esc(tx(`Click to scroll to the result in part ${SIX('onetoken')+1}.`,`点击滚到第 ${SIX('onetoken')+1} 部分的结论。`))}">${tx(`The result is in part ${SIX('onetoken')+1} ↓`,`结论在第 ${SIX('onetoken')+1} 部分 ↓`)}</button>`:'');
}
/* the sweep: request 2's blocks looked up left to right in about 1.8 s, with a short stop at the first miss. Every frame checks the run
   token: a new position, a part, a mode, a language or leaving the lab stops it, so an old sweep never draws over a new state */
const SW_MS=1500, SW_PAUSE=300;
let otStepEnd=false;                                                  // block by block has just ended: no need to point to the result
function swStop(){swRun++; if(swRaf) cancelAnimationFrame(swRaf); swRaf=0; swK=null; swStep=false;}
function swStart(rail,atEnd){
  swStop(); otStepEnd=false; if(!app) return;
  const draw=rail?renderRail:render;                                  // after the prediction the card changes too
  if(!otOpen()||A.reduceMotion()||atEnd){draw(); otAfter(); return;}   // reduced motion, or a lookup already watched (R3.20): the end state at once
  const my=swRun, r=otRun(otPos).r2, T=r.total, hb=r.hitBlocks, tb=SW_MS/T, t0=performance.now();
  swK=0; draw();
  const f=now=>{if(my!==swRun||!app||!isOT()) return;
    const e=now-t0, tm=(hb+1)*tb, k=e<tm?Math.floor(e/tb):(e<tm+SW_PAUSE?hb+1:Math.floor((e-SW_PAUSE)/tb));
    if(k>=T){swRaf=0; swK=null; otDone.add(otPos); render(); otAfter(); return;}
    if(k!==swK){swK=k; drawPairFrame(); dockLive();}
    swRaf=requestAnimationFrame(f);};
  swRaf=requestAnimationFrame(f);}
/* block by block: each click moves to the next stop of otStops() (the counters follow); after the last one, the end state again */
function otStep(){
  if(!otOpen()) return;
  const o=otRun(otPos), T=o.r2.total;
  if(!swStep){swStop(); swStep=true; swK=0; otStepEnd=false; render(); return;}
  const n=otStops(o).find(x=>x>swK);
  if(n==null||n>=T){swStop(); otStepEnd=true; otDone.add(otPos); render(); return;}
  swK=n; drawPairFrame(); updateRail(); dockLive(); if(A.announce) A.announce($('#px-prog',app).textContent);}
const dockLive=()=>{const d=document.getElementById('dock'); if(d&&!d.hidden&&A.dock.render) A.dock.render();};
/* after the end state appears: on a wide screen the result comes into view (R3.10); on a phone the chart stays where it is and the line
   under it points to the result, so the positions never scroll away */
function otAfter(){
  requestAnimationFrame(()=>{if(!app||!isOT()||innerWidth<=760) return; const box=$('#px-rail [data-obs]',app); if(box&&!box.hidden) revealBox(box);});}
/* a pick of a position, from the card or from under request 2 in the chart: before the prediction it only moves the changed token (the
   question names it); after it, request 2 runs and its blocks are looked up (a click on the position already picked changes nothing; ↻ Replay looks them up again, R3.20).
   On a phone a pick in the card first brings the chart into view, so the lookup plays where the learner can see it (review r1 #1) */
function otPick(v,fromRail){
  if(!isOT()||!OT_POS.some(p=>p.v===v)) return;
  otRail=fromRail; A.tip.hide();
  if(v===otPos) return;                                               // R3.20: the position on screen is not looked up again; ↻ Replay does that
  otPos=v; S=otS(); sel=1;
  if(!otPred){swStop(); byLearner(()=>renderRail()); if(fromRail){const b=$(`#px-rail [data-lever="${v}"]`,app); if(b) b.focus({preventScroll:true});} return;}
  if(fromRail&&innerWidth<=760) otShowStage();
  byLearner(()=>{swStart(false,otDone.has(v)); updateRail();});}      // R3.20: a position whose lookup was watched to the end (A → B → A) shows its end state at once
function otShowStage(){const pb=app&&$('#px-pairbox',app); if(!pb) return; const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:0)+8, y=pb.getBoundingClientRect().top;
  if(Math.abs(y-top)>=2) window.scrollTo({top:Math.max(0,scrollY+y-top),behavior:'auto'});}
/* metric cards: ▲▼ only after the learner's own change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
const unitOf=v=>{const m=String(v).match(/(ms|s|GiB)$/); return m?m[1]:'';};
const toBase=(v,u)=>u==='ms'?v/1000:v;
function metricsDef(){
  if(mode==='challenge'){const r=chRun(chOrder), pre=r.res.reduce((a,x)=>a+x.N-x.hit,0), all=r.res.reduce((a,x)=>a+x.N,0);
    return [
     {k:'avg',label:tx('Cached tokens per request','每个请求的缓存 token'),v:fint(r.avg),sub:tx('average of 8; goal 4,400','8 个的平均；目标 4,400'),tip:tx(`On average over the 8 requests, ${fint(r.avg)} of each request’s prompt tokens come from the cache (the goal is at least 4,400).`,`8 个请求平均下来，每个请求有 ${fint(r.avg)} 个 prompt token 来自缓存（目标至少 4,400）。`)},
     {k:'avgw',label:tx('First token, with the cache','首 token（用缓存）'),v:fT(r.avgW),sub:tx('average of 8','8 个的平均'),tip:tx(`The time to first token averaged over the 8 requests, with prefix caching: ${fT(r.avgW)}.`,`用 prefix caching 时，8 个请求首 token 延迟的平均：${fT(r.avgW)}。`)},
     {k:'avgwo',label:tx('First token, without','首 token（不用缓存）'),v:fT(r.avgWo),sub:tx('average of 8','8 个的平均'),tip:tx(`The same average if every request ran its whole prefill: ${fT(r.avgWo)}.`,`如果每个请求都做完整的 prefill，同样的平均是 ${fT(r.avgWo)}。`)},
     {k:'pre',label:tx('Tokens prefilled','prefill 的 token'),v:bigT(pre),sub:tx(`of ${bigT(all)}, all 8`,`8 个一共 ${bigT(all)} 个`),tip:pre===all?tx(`The 8 requests compute all ${fint(all)} of their prompt tokens: none come from the cache.`,`8 个请求要现算全部 ${fint(all)} 个 prompt token：没有一个来自缓存。`)
       :tx(`The 8 requests compute ${fint(pre)} of their ${fint(all)} prompt tokens; the other ${fint(all-pre)} come from the cache.`,`8 个请求的 ${fint(all)} 个 prompt token 里要现算 ${fint(pre)} 个，其余 ${fint(all-pre)} 个来自缓存。`)}];}
  const x=selRes(), i=selIdx(), mem=x.hit*KV_TOK, R0=reqOf(i), none=x.hit===0, by=x.w.cb?tx('arithmetic','计算'):tx('reading memory','读显存');
  return [
   {k:'w',label:tx('Time to first token (TTFT)','首 token 延迟（TTFT）'),v:fT(x.w.t),sub:tx(`${R0}, with the cache`,`${R0}，用缓存`),tip:TIP.ttft()+tx(' ','')+(none?tx(`Here: ${R0} finds nothing in the cache and computes all ${fint(x.N)} tokens, so ${fT(x.w.t)}, set by ${by}.`,`这里：${R0}在缓存里没找到能用的，要现算全部 ${fint(x.N)} 个 token，所以是 ${fT(x.w.t)}，由${by}决定。`)
     :tx(`Here: ${R0} computes ${fint(x.N-x.hit)} tokens and reads ${fint(x.hit)} from the cache, so ${fT(x.w.t)}, set by ${by}.`,`这里：${R0}现算 ${fint(x.N-x.hit)} 个 token，从缓存读 ${fint(x.hit)} 个，所以是 ${fT(x.w.t)}，由${by}决定。`))},
   {k:'wo',label:tx('TTFT without the cache','不用缓存的 TTFT'),v:fT(x.wo.t),sub:tx(`all ${fint(x.N)} tokens`,`全部 ${fint(x.N)} 个 token`),tip:tx(`What ${R0} would wait without prefix caching: a prefill of all ${fint(x.N)} prompt tokens, ${fT(x.wo.t)}.`,`不用 prefix caching 时${R0}要等多久：对全部 ${fint(x.N)} 个 prompt token 做 prefill，${fT(x.wo.t)}。`)},
   {k:'pre',label:tx('Tokens prefilled','prefill 的 token'),v:bigT(x.N-x.hit),sub:tx(`of ${bigT(x.N)} in the prompt`,`prompt 共 ${bigT(x.N)} 个`),tip:none?tx(`The prefill of ${R0} computes all ${fint(x.N)} of its prompt tokens: its first block already misses.`,`${R0}的 prefill 要算全部 ${fint(x.N)} 个 prompt token：它的第一个 block 就未命中。`)
     :tx(`The prefill of ${R0} computes ${fint(x.N-x.hit)} of its ${fint(x.N)} prompt tokens: every token from the first missed block on.`,`${R0}的 prefill 要算 ${fint(x.N)} 个 prompt token 里的 ${fint(x.N-x.hit)} 个：从第一个未命中的 block 起的每个 token。`)},
   {k:'hit',label:tx('Tokens from the cache','来自缓存的 token'),v:bigT(x.hit),sub:none?tx('no block matched','没有 block 命中'):tx(`${fint(x.hitBlocks)} blocks of 16`,`${fint(x.hitBlocks)} 个 block，每个 16 个 token`),tip:none?tx(`${reqName(i)} finds none of its blocks in the cache, so it reads no keys or values from it and computes them all.`,`${reqName(i)}在缓存里一个 block 也没找到，没有可以直接读的 key 和 value，全部要算。`)
     :tx(`${reqName(i)} finds its first ${fint(x.hitBlocks)} blocks in the cache: ${fint(x.hitBlocks)} × 16 = ${fint(x.hit)} tokens whose keys and values it reads instead of computing.`,`${reqName(i)}在缓存里找到了开头的 ${fint(x.hitBlocks)} 个 block：${fint(x.hitBlocks)} × 16 = ${fint(x.hit)} 个 token，它们的 key 和 value 直接读，不用算。`)},
   {k:'mem',label:tx('Their KV cache','这些 token 的 KV cache'),v:none?'0'+NS+'GiB':gib2(mem),sub:tx('in GPU memory','占的显存'),tip:none?tx(`${reqName(i)} reads no tokens from the cache, so none are counted here. Its own full blocks join the cache when it finishes.`,`${reqName(i)}没有从缓存读 token，所以这里是 0。它写满的 block 在它结束后进缓存。`)
     :tx(`The KV cache of the ${fint(x.hit)} cached tokens stays in GPU memory: ${fint(x.hit)} × 131,072 bytes = ${gbS(mem)} = ${gib2(mem)}.`,`这 ${fint(x.hit)} 个缓存 token 的 KV cache 一直占着显存：${fint(x.hit)} × 131,072 字节 = ${gbS(mem)} = ${gib2(mem)}。`)}];
}
function drawMetrics(){
  const host=$('#px-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[mode+m.k];
    if(userAct&&prev!=null&&prev!==m.v){const ua=unitOf(prev), ub=unitOf(m.v), same=ua===ub, a=same?numOf(prev):toBase(numOf(prev),ua), b=same?numOf(m.v):toBase(numOf(m.v),ub);
      if(isFinite(a)&&isFinite(b)&&!/[MB]$/.test(prev)&&!/[MB]$/.test(m.v)) dt=A.fmt.delta(a,b,same?{dp:Math.max(decOf(prev),decOf(m.v))}:{})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[mode+m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6), and every term has its own tooltip (R1.4) */
/* H1/H2 (the owner, 2026-10-09): Show the math as short blocks, like a page of a good textbook. Each block has one lead line in words,
   then a small table: the quantity's name and its formula typeset (A.ui.math), this run's numbers under the terms they replace, and the
   result, each on its own line with the "=" signs lined up. groups: [{name, terms:[TeX per column], rows:[{l, c:[html per column]} | {r:html}]}] */
const MX=x=>A.ui.math(x);
function mathBlock(lead,groups,note){const n=Math.max(...groups.map(g=>Math.max(g.terms.length,...g.rows.map(r=>r.c?r.c.length:1)))), eq=`<td class="px-meq">${MX('=')}</td>`;
  const cells=(a,f)=>a.map((x,i)=>`<td${i===a.length-1&&a.length<n?` colspan="${n-i}"`:''}>${f(x)}</td>`).join('');
  const tr=r=>`<tr><th scope="row">${r.l||''}</th>${eq}${r.c?cells(r.c,x=>x):`<td colspan="${n}" class="px-mres">${r.r}</td>`}</tr>`;
  const body=groups.map(g=>`<tr class="px-mname"><th scope="row">${g.name}</th>${eq}${cells(g.terms,MX)}</tr>${g.rows.map(tr).join('')}`).join('');
  return `<div class="px-mb">${lead?`<p class="px-mlead">${lead}</p>`:''}<div class="px-mfx"><table class="px-mt">${body}</table></div>${note?`<p class="px-mnote">${note}</p>`:''}</div>`;}
/* Show the math on a box too narrow for its table (the owner's review of 2026-10-09: a row the reader has to scroll sideways is not
   read). fitMath sets such a block again: the quantity's name on its own line; under it the formula with this run's numbers under the
   terms they replace, broken before an operator (×, ÷, +, −, =) so a long formula goes on in the next line under its first term; the "="
   signs still line up and the result comes last. It runs after each render and again when the box changes width. */
const MOPS=new Set(['×','÷','+','−','-','=','≈','·']);
function mathNarrow(fx){const tb=fx.querySelector('table.px-mt'); if(!tb) return; const e0=tb.querySelector('td.px-meq'), E=e0?e0.innerHTML:'=', gs=[]; let g=null;
  for(const tr of tb.rows){const cs=[...tr.cells], th=cs[0], tds=cs.slice(2);
    if(tr.classList.contains('px-mname')){g={name:th.innerHTML,terms:tds.map(c=>c.innerHTML),ops:tds.map(c=>MOPS.has(c.textContent.trim())),tall:!!tr.querySelector('mfrac,munderover,munder,mover'),rows:[]}; gs.push(g); continue;}
    if(g) g.rows.push({l:th?th.innerHTML:'',res:tds.length===1&&tds[0].classList.contains('px-mres'),c:tds.map(c=>c.innerHTML)});}
  let h='';
  for(const g of gs){const n=g.terms.length, band=[]; let k=0;
    while(k<g.rows.length&&!g.rows[k].res&&g.rows[k].c.length===n){band.push(g.rows[k]); k++;}
    const lb=band.map(r=>r.l).filter(Boolean), units=[]; for(let i=0;i<n;){const u=[i]; if(g.ops[i]&&i+1<n&&!g.ops[i+1]) u.push(i+1); units.push(u); i=u[u.length-1]+1;}
    const lines=[g.terms,...band.map(r=>r.c)], cell=(x,r)=>`<span class="px-mc${r?'':' px-mc0'}">${x}</span>`;
    h+=`<tr class="px-mname"><th scope="row" colspan="2">${g.name}${lb.length?` <span class="px-mlbl">${lb.join(' ')}</span>`:''}</th></tr>`+
      `<tr class="px-mband${g.tall?' px-mtall':''}"><td class="px-meq"><span class="px-mu">${lines.map((_,r)=>cell(E,r)).join('')}</span></td><td><div class="px-mfl">${units.map(u=>`<span class="px-mu" style="grid-template-columns:repeat(${u.length},auto)">${lines.map((ln,r)=>u.map(j=>cell(ln[j],r)).join('')).join('')}</span>`).join('')}</div></td></tr>`;
    for(const r of g.rows.slice(k)) h+=`<tr><td class="px-meq">${E}</td><td class="px-mres">${r.l?`<span class="px-mlbl">${r.l}</span> `:''}${r.c.join(' ')}</td></tr>`;}
  tb.className='px-mtn'; tb.innerHTML=h;}
function fitMath(mb){if(!mb) return; if(!mb._fwo&&typeof ResizeObserver!=='undefined'){mb._fwo=new ResizeObserver(()=>requestAnimationFrame(()=>fitMath(mb))); mb._fwo.observe(mb);}
  const w=mb.clientWidth; if(!w) return; const nar=!!mb.querySelector('table.px-mtn');
  if(nar&&mb._fw===w) return;
  if(nar&&mb._wide!=null) mb.innerHTML=mb._wide;
  mb._wide=mb.innerHTML; mb._fw=w;
  for(const fx of mb.querySelectorAll('.px-mfx')) if(fx.scrollWidth>fx.clientWidth+1) mathNarrow(fx);}
const kvMath=t=>mathBlock(tx('Every token keeps a key and a value for each KV head in every layer, 2 bytes per number in BF16.','每个 token 在每一层、每个 KV head 上存一个 key 和一个 value，BF16 下每个数值占 2 字节。'),
  [{name:tx('KV cache per token','每个 token 的 KV cache'),terms:['2','\\times',tx('\\text{layers}','\\text{层数}'),'\\times',tx('\\text{KV heads}','\\text{KV head 数}'),'\\times','\\text{head_dim}','\\times',tx('\\text{bytes per number}','\\text{每个数值的字节数}')],
    rows:[{c:[t('kv','2'),'×','32','×','8','×','128','×','2']},{r:`131,072 ${tx('bytes','字节')} (<b>128${NS}${DEF('KiB',TIP.kib())}</b>)`}]}]);
/* one prefill's time: compute time and memory time, the longer of the two (the GPU does both at once) */
const FXC=[tx('\\text{FLOP}','\\text{运算量}'),'\\div',tx('\\text{FLOP/s reached}','\\text{能达到的 FLOP/s}')], FXM=[tx('\\text{bytes moved}','\\text{搬运的字节数}'),'\\div',tx('\\text{bandwidth reached}','\\text{能达到的带宽}')],
  FXT=[tx('\\max(\\text{compute time},\\ \\text{memory time})','\\max(\\text{计算时间},\\ \\text{读写显存时间})')];
function stepMath(lead,o,flDesc,byDesc,ce,be,pk,bw){const TF=v=>fixd(v/1e12,3)+NS+'TFLOP', GB=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gb());
  return mathBlock(lead,[
    {name:tx('Compute time','计算时间'),terms:FXC,rows:[{r:`(${flDesc}) ÷ ${ce}`},{c:[TF(o.fl),'÷',fixd(pk/1e12,2)+NS+'TFLOP/s']},{r:`${fixd(o.tc*1e3,2)}${NS}ms`}]},
    {name:tx('Memory time','读写显存时间'),terms:FXM,rows:[{r:`(${byDesc}) ÷ ${be}`},{c:[GB(o.by),'÷',fixd(bw/1e12,2)+NS+'TB/s']},{r:`${fixd(o.tm*1e3,2)}${NS}ms`}]},
    {name:`TTFT (${DEF(tx('the longer of the two','取两者中较长的'),TIP.overlap())})`,terms:FXT,rows:[{c:[`max(${fixd(o.tc*1e3,2)}${NS}ms, ${fixd(o.tm*1e3,2)}${NS}ms)`]},{r:`<b>${fT(o.t)}</b>${o.cb?tx(', set by arithmetic','，由计算决定'):tx(', set by memory traffic','，由读写显存决定')}`}]}]);}
function mathHTML(){
  if(isOT()) return mathOT();
  const x=selRes(), i=selIdx(), t=T_, N=x.N, H=x.hit, pk=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu;
  const ce=t('effc',fixd(pk/1e12,2)+NS+'TFLOP/s'), be=t('effm',fixd(bw/1e12,2)+NS+'TB/s'), WL=t('wl','2 × '+fint(P_LAYERS)), WH=t('wh','2 × '+fint(P_HEAD));
  const rows=[kvMath(t)];
  rows.push(H===0?`<div class="px-mb"><p class="px-mlead">${tx(`${reqName(i)}: cached = <b>0</b> tokens (no block matched)`,`${reqName(i)}：缓存 = <b>0</b> 个 token（没有 block 命中）`)}</p></div>`
    :mathBlock(tx(`${reqName(i)}: the blocks that match the cache are reused, 16 tokens each.`,`${reqName(i)}：和缓存对得上的 block 直接复用，每个 block 16 个 token。`),
      [{name:tx('Cached tokens','缓存命中的 token'),terms:[tx('\\text{blocks matched}','\\text{命中的 block 数}'),'\\times',tx('\\text{tokens per block}','\\text{每个 block 的 token 数}')],rows:[{c:[t('hitb',fint(x.hitBlocks)),'×','16']},{r:`<b>${fint(H)}</b> ${tx('tokens','个 token')}`}]},
       {name:tx('Their KV cache','它们的 KV cache'),terms:[tx('\\text{cached tokens}','\\text{命中的 token 数}'),'\\times',tx('\\text{KV cache per token}','\\text{每个 token 的 KV cache}')],rows:[{c:[fint(H),'×',`131,072 ${tx('bytes','字节')}`]},{r:`${fint(H*KV_TOK)} ${tx('bytes','字节')} (${gbS(H*KV_TOK)}, <b>${gib2(H*KV_TOK)}</b>)`}]}]));
  const wo=x.wo, w=x.w;
  rows.push(stepMath(tx(`<b>Without prefix caching: all ${fint(N)} tokens.</b> The GPU computes and moves memory at the same time, so the prefill takes the longer of the two.`,`<b>不用 prefix caching：全部 ${fint(N)} 个 token。</b>GPU 计算和读写显存同时进行，所以 prefill 取两者中较长的那个。`),wo,
    `${WL} × ${t('N',fint(N))} + ${WH} + ${t('att0',`2 × 32 × 4,096 × ${fint(N)} × ${fint(N+1)}`)}`,`${t('wr',fint(W_READ))} + ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,pk,bw));
  if(H===0) rows.push(`<div class="px-mb"><p class="px-mlead"><b>${tx('With it: nothing cached','用了缓存：没有命中')}</b><br>${tx(`so prefill computes all ${fint(N)} tokens, the same step as without: TTFT = `,`所以 prefill 仍要算全部 ${fint(N)} 个 token，和不用缓存时是同一步：TTFT = `)}<b>${fT(w.t)}</b></p></div>`);
  else rows.push(stepMath(tx(`<b>With it: ${fint(N-H)} new tokens, positions ${fint(H+1)} to ${fint(N)}.</b> The memory traffic is the same as without: the keys and values of all ${fint(N)} tokens are read or written either way.`,`<b>用了缓存：${fint(N-H)} 个新 token，位置 ${fint(H+1)} 到 ${fint(N)}。</b>读写显存和不用缓存时一样：全部 ${fint(N)} 个 token 的 key 和 value 都要读或写。`),w,
    `${WL} × ${t('new',fint(N-H))} + ${WH} + ${t('att1',`2 × 32 × 4,096 × (${fint(N)} × ${fint(N+1)} − ${fint(H)} × ${fint(H+1)})`)}`,`${t('wr',fint(W_READ))} + ${t('N',fint(N))} × ${t('kvt','131,072')}`,ce,be,pk,bw));
  if(H>0) rows.push(mathBlock('',[{name:tx('Sooner by','快了'),terms:[tx('\\frac{\\text{TTFT without the cache}}{\\text{TTFT with it}}','\\frac{\\text{不用缓存的 TTFT}}{\\text{用缓存的 TTFT}}')],rows:[{c:[`${fT(wo.t)} ÷ ${fT(w.t)}`]},{r:`<b>${ratioS(ratioOf(wo.t,w.t))}</b> ${tx('times','倍')}`}]}]));
  return rows.join('');
}
/* part 5: every position, from the engine: the changed block, the hit (blocks × 16), the recomputed tokens, the step model's time to first
   token, the new blocks and their memory; then what the cache holds once request 2's answer is written. H2 (2026-10-09): each quantity's
   formula once, typeset, with one row of numbers per position under it */
function mathOT(){
  const t=T_, rows=[], o0=otRun('middle'), N=o0.r1.N, nbF=o0.r1.nb, pk=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu;
  const GB=v=>fixd(v/1e9,3)+NS+DEF('GB',TIP.gb());
  const ce=t('effc',fixd(pk/1e12,2)+NS+'TFLOP/s'), be=t('effm',fixd(bw/1e12,2)+NS+'TB/s'), WL=t('wl','2 × '+fint(P_LAYERS)), WH=t('wh','2 × '+fint(P_HEAD));
  rows.push(kvMath(t));
  rows.push(mathBlock(tx('Both requests send the same system prompt and question; only one word differs.','两个请求发的是同样的 system prompt 和问题，只差一个词。'),
    [{name:tx('Each prompt','每个 prompt'),terms:[tx('\\text{system prompt}','\\text{system prompt}'),'+',tx('\\text{question}','\\text{问题}')],rows:[{r:t('otN',`${fint(BASE.P)} + ${fint(BASE.O)}`)},{r:`${fint(N)} ${tx('tokens','个 token')} (${fint(nbF)} ${tx(`full blocks of ${BLOCK}, and ${N-nbF*BLOCK} tokens more`,`个 ${BLOCK} token 的完整 block，再加 ${N-nbF*BLOCK} 个 token`)})`}]}]));
  if(!otOpen()){rows.push(`<p class="muted">${tx('Each position’s lines are shown after your prediction.','预测之后，这里给出每个位置的算式。')}</p>`); return rows.join('');}
  const w0=o0.r2.w;
  rows.push(mathBlock(tx('Request 2 reads or writes the keys and values of all its tokens, wherever the change is.','不管改动在哪，请求 2 都要读或写全部 token 的 key 和 value。'),
    [{name:tx('Memory time, the same at every position','读写显存时间，每个位置都一样'),terms:FXM,rows:[{r:`(${t('wr',fint(W_READ))} + ${fint(N)} × ${t('kvt','131,072')}) ÷ ${be}`},{c:[GB(w0.by),'÷',fixd(bw/1e12,2)+NS+'TB/s']},{r:`<b>${fixd(w0.tm*1e3,2)}${NS}ms</b>`}]}]));
  const per=f=>OT_POS.map(p=>{const o=otRun(p.v); return Object.assign({l:esc(otName(p.v))},f(o,o.r2,p.v,p));});
  rows.push(mathBlock(tx(`Where the change is decides how much request 2 reuses: the blocks before the changed one match, the rest are computed again. ⌊ ⌋ rounds down.`,`改动的位置决定请求 2 能复用多少：改动所在 block 之前的 block 都对得上，从它开始都要重算。⌊ ⌋ 表示向下取整。`),
    [{name:tx('Changed block','改动的 block'),terms:[tx('⌊(\\text{position} - 1) \\div 16⌋ + 1','⌊(\\text{位置} - 1) \\div 16⌋ + 1')],rows:per((o,r,v,p)=>({c:[`⌊(${fint(p.t)} − 1) ÷ ${BLOCK}⌋ + 1`,MX('='),t('otb-'+v,fint(o.blk+1))]}))},
     {name:tx('Cached tokens','缓存命中的 token'),terms:[tx('\\text{blocks matched} \\times 16','\\text{命中的 block 数} \\times 16')],rows:per((o,r,v)=>({c:[`${fint(r.hitBlocks)} × ${BLOCK}`,MX('='),t('oth-'+v,fint(r.hit))]}))},
     {name:tx('System-prompt tokens recomputed','重算的 system prompt token'),terms:[tx('\\text{system prompt} - \\text{cached}','\\text{system prompt} - \\text{命中的}')],rows:per((o,r,v)=>({c:[`${fint(BASE.P)} − ${fint(r.hit)}`,MX('='),`<b>${t('ots-'+v,fint(o.recSys))}</b>`]}))},
     {name:tx('Tokens prefill computes','prefill 要算的 token'),terms:[tx('\\text{prompt} - \\text{cached}','\\text{prompt} - \\text{命中的}')],rows:per((o,r,v)=>({c:[`${fint(N)} − ${fint(r.hit)}`,MX('='),t('otr-'+v,fint(N-r.hit))]}))}]));
  rows.push(mathBlock(tx('The arithmetic shrinks with the tokens recomputed; the memory time above does not, so TTFT is the longer of the two.','计算量随重算的 token 变少，上面的读写显存时间却不变，所以 TTFT 取两者中较长的那个。'),
    [{name:tx('Compute time','计算时间'),terms:FXC,rows:per((o,r,v)=>({c:[`(${WL} × ${fint(N-r.hit)} + ${WH} + ${t('ota-'+v,`2 × ${LL.layers} × ${fint(LL.hidden)} × (${fint(N)} × ${fint(N+1)} − ${fint(r.hit)} × ${fint(r.hit+1)})`)}) ÷ ${ce}`,MX('='),`${fixd(r.w.tc*1e3,2)}${NS}ms`]}))},
     {name:`TTFT (${DEF(tx('the longer of the two','取两者中较长的'),TIP.overlap())})`,terms:FXT,rows:per((o,r)=>({c:[`max(${fixd(r.w.tc*1e3,2)}${NS}ms, ${fixd(r.w.tm*1e3,2)}${NS}ms)`,MX('='),`<b>${fT(r.w.t)}</b>`]}))}]));
  rows.push(mathBlock(tx('From the changed block on, request 2 stores its own copy of the system prompt’s blocks.','从改动的 block 开始，请求 2 要再存一份 system prompt 的 block。'),
    [{name:tx('System-prompt blocks stored twice','存了两份的 system prompt block'),terms:[tx('\\text{system-prompt blocks} - \\text{blocks matched}','\\text{system prompt 的 block 数} - \\text{命中的 block 数}')],rows:per((o,r,v)=>({c:[`${fint(o.preB)} − ${fint(r.hitBlocks)}`,MX('='),t('otw-'+v,fint(o.dup))]}))},
     {name:tx('Their KV cache','它们的 KV cache'),terms:[tx('\\text{blocks} \\times 16 \\times \\text{KV cache per token}','\\text{block 数} \\times 16 \\times \\text{每个 token 的 KV cache}')],rows:per(o=>({c:[`${fint(o.dup)} × ${BLOCK} × 131,072`,MX('='),(o.dupBytes<0.005*A.GiB?`<b>${fint(o.dupBytes/A.MiB)}${NS}MiB</b>${tx(', less than 0.01 GiB','，不到 0.01 GiB')}`:`${gbS(o.dupBytes)} (<b>${gib2(o.dupBytes)}</b>)`)]}))}]));
  /* the whole cache after request 2 and its answer (review r1 #7): request 1's blocks, request 2's new ones, every number with its hover */
  const r=otRun(otPos), qB=r.nw-r.dup;
  rows.push(mathBlock(tx(`What the cache holds once both answers are written, with the change ${L(OT_W[otPos])}.`,`两个回答都写完后，缓存里有什么（改动在${L(OT_W[otPos])}）。`),
    [{name:tx('Blocks request 1 left','请求 1 留下的 block'),terms:[tx('⌊(\\text{prompt} + \\text{answer} - 1) \\div 16⌋','⌊(\\text{prompt} + \\text{回答} - 1) \\div 16⌋')],rows:[{c:[`⌊(${fint(N)} + ${fint(ANS)} − 1) ÷ ${BLOCK}⌋`]},{r:`${t('otc1',fint(r.r1.cached))}${tx(`: its prompt’s ${fint(nbF)} full blocks and `,`：它的 prompt 的 ${fint(nbF)} 个完整 block，加上 `)}${t('otans',fint(r.ansB))}${tx(` that its ${ANS}-token answer filled`,` 个由它 ${ANS} 个 token 的回答写满的 block`)}`}]},
     {name:tx('Blocks in the cache after request 2','请求 2 之后缓存里的 block'),terms:[tx('\\text{request 1’s} + \\text{stored twice} + \\text{question} + \\text{answer}','\\text{请求 1 的} + \\text{存了两份的} + \\text{问题的} + \\text{回答的}')],rows:[{c:[`${fint(r.r1.cached)} + ${fint(r.dup)} + ${t('otq',fint(qB))} + ${fint(r.ansB)}`]},{r:`<b>${t('otc2',fint(r.r2.cached))}</b>`}]}]));
  return rows.join('');}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const lg=$('#px-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const gr=$('#px-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].focus||''):'';
  /* what the part or the mode is about comes first: the requests in order for a stream, the block strip and the timeline for one request;
     part 5 shows the two requests block by block instead, with its own counters */
  const ot=isOT(); $('#px-stage',app).dataset.part=ot?'onetoken':'';
  gr.dataset.lead=ot?'pair':((mode==='challenge'||(mode==='guided'&&STEPS[step].id==='cost')||(mode==='sandbox'&&S.n>2))?'rows':'strip');
  if(ot){drawPair(); $$('#px-rowsbox [data-tipx],#px-stripbox [data-tipx],#px-tl [data-tipx],#px-metrics [data-tipx]',app).forEach(e=>e.removeAttribute('data-tipx'));   // hidden here: no stale tooltip
    $$('#px-metrics .metric .d',app).forEach(d=>{clearTimeout(d._t); clearTimeout(d._t2); d.classList.remove('on'); d.textContent='';});}                  // nor a badge from the part before
  else {drawRows(); drawStrip(); drawTL(); drawMetrics();}
  const mb=$('#px-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;} fitMath(mb);
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const setR=(key,stops,val,fmt)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{const v=stops?stops.indexOf(val):val; if(+i.value!==v) i.value=v; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=fmt(val); i.setAttribute('aria-valuetext',fmt(val));});
  setR('P',P_STOPS,S.P,tokN); setR('O',O_STOPS,S.O,tokN); setR('n',null,S.n,reqN);
  [['kind',S.kind],['change',S.change],['apps',String(S.apps)],['mem',String(S.mem)]].forEach(([k,v])=>$$(`.seg-ctl[data-key="${k}"]`,app).forEach(g=>A.ui.setPressed(g,v)));
  const chat=S.kind==='chat', cw=$('#px-change-note',app); if(cw) cw.textContent=chat?tx('Chat turns keep each earlier turn unchanged, so this applies only to the same-prefix requests.','多轮对话保持之前每一轮不变，所以这一项只用于同一前缀的请求。'):tx('Request 1 comes a day before the others, and each request is from a different user.','请求 1 比其余的早一天，每个请求来自不同的用户。');
  $$('.seg-ctl[data-key="change"] button',app).forEach(b=>{b.disabled=chat&&b.dataset.v!=='none';});
  $$('.seg-ctl[data-key="apps"] button',app).forEach(b=>{b.disabled=chat&&b.dataset.v!=='1';});
  const mn=$('#px-mem-note',app); if(mn) mn.textContent=S.mem==='pool'?tx(`The whole KV cache pool, ${gib1(POOL_B)}: room for ${fint(POOL_T)} tokens.`,`整个 KV cache 显存池 ${gib1(POOL_B)}：能放 ${fint(POOL_T)} 个 token。`)
    :tx(`Room for ${S.mem} request${S.mem>1?'s':''} of ${fint(S.P+S.O+ANS)} tokens (prompt and answer): ${gib2(memTokens(S)*KV_TOK)}. Past that, the least recently used blocks are evicted.`,`能放 ${S.mem} 个 ${fint(S.P+S.O+ANS)} 个 token（prompt 加回答）的请求：${gib2(memTokens(S)*KV_TOK)}。超过就淘汰最久没用过的 block。`);
  const ot=isOT(), lock=mode==='challenge'||ot, pl=mode==='guided'&&!pred&&step<=1&&!sent, c=$('#px-controls',app); c.classList.toggle('locked',lock||pl);
  const ln=$('#px-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=ot?OT_LOCK():tx(`In the challenge you can change only the order of the prompt’s parts, in the task card. Everything else is fixed: ${MODEL} on one ${GPUN}, the six parts and the 8 requests.`,`挑战中只能在任务卡里改 prompt 各部分的顺序。其余固定不变：一张 ${GPUN} 跑 ${MODEL}，六个部分和 8 个请求。`);
  const pn=$('#px-prelock',app); if(pn) pn.hidden=!pl||lock;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const so=statusOf(); A.announce(so.txt);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(p.lever!=null){lever=p.lever; leverTried.add(lever); S={...S,change:lever}; delete p.lever;}
  Object.assign(S,p);
  if(!P_STOPS.includes(S.P)) S.P=BASE.P; if(!O_STOPS.includes(S.O)) S.O=BASE.O;
  S.n=Math.max(1,Math.min(N_MAX,Math.round(S.n)||1)); if(!CHANGES.includes(S.change)) S.change='none'; if(![1,3].includes(+S.apps)) S.apps=1; S.apps=+S.apps;
  if(S.mem!=='pool'){S.mem=+S.mem; if(![1,2,3].includes(S.mem)) S.mem='pool';}
  if(S.kind==='chat'){S.change='none'; S.apps=1;}
  if(mode==='guided'){const id=STEPS[step].id; if(id==='break'&&p.change!=null){lever=S.change; if(lever!=='none') leverTried.add(lever);} if(id==='cost'&&p.mem!=null){memLever=S.mem; if(S.mem!=='pool') memTried.add(S.mem);}}
  sel=Math.min(sel,S.n-1); if(p.n!=null||p.kind!=null||p.apps!=null) sel=S.n-1;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; swStop(); const was=STEPS[step].id; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter){const e=s.enter(); S={...e.s}; sel=e.sel;}
  if(s.id==='send'&&sent) sel=1;
  /* R3.20: part 6 again while its requests arrive (its header clicked): they go on; otherwise the arrival plays only for rows that
     differ from the ones last shown in full, so coming back to part 6 never replays the run already watched */
  const goOn=was==='cost'&&s.id==='cost'&&seqN!=null&&!!seqCur&&firstDiff(seqCur,seqKeys())>=seqN;
  if(goOn){seqCur=seqKeys(); sel=seqN-1;} else stopSeq();
  const st=$('#px-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.id==='cost'&&!goOn) playSeq(seqSeen);
  const tg={send:'#px-ic-send',break:'#px-rail .levers',onetoken:'#px-rail .levers',cost:'#px-rail .levers'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#px-rail .step.current',app)||$('#px-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?Math.min(r.bottom-innerHeight+16,r.top-top):r.top-top;
  if(Math.abs(dy)>=2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#px-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#px-rail .step.current',app)||$('#px-rail .railcard',app)||$('#px-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* polish r5: the strip and the timeline are drawn for their width (blocks per cell, the labels pinned under the strip, the time axis); on a
   wide stage part 6 puts them side by side (prefix-caching.css), and the stage changes width without a window resize when the page puts
   the step list beside it or leaves the primer, so they are drawn again when it does (as watchStage in ppo.js) */
let stRO=null, stW=0;
function watchStage(){if(stRO){stRO.disconnect(); stRO=null;} const st=app&&$('#px-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w;
    if(isOT()) drawPair(); else {drawStrip(); drawTL();}}));
  stRO.observe(st);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#px-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#px-insight',app)); else revealBox($('#px-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  stopSeq(); swStop();
  if(isOT()){S={...BASE}; sel=1;}                                  // part 5's changed token never reaches Scenarios or the challenge
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved.S}; sel=saved.sel; saved=null;}
  if(m==='challenge'){saved={S:{...S},sel}; chSubmitted=false; chReveal=false; chOrder=CH.parts.slice(); sel=CH.stream.length-1;}
  if(m==='sandbox') sbTouched=false;
  A.shell.setMode(app,m,DESCS());
  const st=$('#px-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[s.P,s.O,s.n,s.kind,s.change,s.apps,s.mem].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const P=+p[0], O=+p[1], n=+p[2], kind=p[3], change=p[4], apps=+p[5], mem=p[6]==='pool'?'pool':+p[6];
  if(!P_STOPS.includes(P)||!O_STOPS.includes(O)||!Number.isInteger(n)||n<1||n>N_MAX||!['same','chat'].includes(kind)||!CHANGES.includes(change)||![1,3].includes(apps)||!(mem==='pool'||[1,2,3].includes(mem))) return null;
  return {P,O,n,kind,change,apps,mem};
}

/* =====================================================================
   interview drill: seeded, from the engine; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DRILL_P=[4000,8000,16000,24000], DRILL_O=[100,300,800];   // never the mission's 10,000 + 200 or the new case's 50,000 + 500
function makeDrill(seed){const rnd=mulberry(seed*7919+17), pick=a=>a[Math.floor(rnd()*a.length)];
  const P=pick(DRILL_P), O=pick(DRILL_O), N=P+O, H=Math.floor(P/BLOCK)*BLOCK;
  return {seed,P,O,N,H,wo:ttft(N,0),w:ttft(N,H)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#px-drill-q',app), a=$('#px-drill-a',app);
  qd.innerHTML=tx(`<b>${MODEL}</b> runs in BF16 on one ${GPUN}, at 50% of its 989.5 TFLOP/s dense peak and 60% of its 3.35 TB/s bandwidth. Every request starts with the same <b>${fint(d.P)}</b>-token system prompt and adds <b>${fint(d.O)}</b> tokens of its own. With prefix caching in 16-token blocks, estimate a later request’s time to first token, with and without the cache, and the GPU memory the cached prompt holds.`,
                  `<b>${MODEL}</b> 用 BF16 跑在一张 ${GPUN} 上，按稠密峰值 989.5 TFLOP/s 的 50%、带宽 3.35 TB/s 的 60% 计。每个请求都以同一段 <b>${fint(d.P)}</b> 个 token 的 system prompt 开头，再加上自己的 <b>${fint(d.O)}</b> 个 token。开启 prefix caching、16 个 token 一个 block 时，估算后面一个请求用和不用缓存的首 token 延迟，以及缓存的 prompt 占多少显存。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const pk=H100.peak*EFF.mfu, bw=H100.bw*EFF.mbu, w=d.w, wo=d.wo, mem=d.H*KV_TOK;
  a.innerHTML=tx(`<p><b>With the cache ≈ ${fT(w.t)} (${w.cb?'arithmetic':'memory'}); without ≈ ${fT(wo.t)} (${wo.cb?'arithmetic':'memory'}); the cached prompt holds ${gib2(mem)}.</b></p>`+
    `<p>Cached: ⌊${fint(d.P)} ÷ 16⌋ = ${fint(d.H/BLOCK)} blocks = ${fint(d.H)} tokens; ${fint(d.H)} × 131,072 = ${fint(mem)} bytes = ${gib2(mem)}.</p>`+
    `<p>Without: arithmetic = 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(wo.fl)} FLOP ÷ ${fixd(pk/1e12,2)} TFLOP/s = ${fixd(wo.tc*1e3,2)} ms; memory = ${fint(W_READ)} + ${fint(d.N)} × 131,072 = ${fint(wo.by)} bytes ÷ ${fixd(bw/1e12,2)} TB/s = ${fixd(wo.tm*1e3,2)} ms.</p>`+
    `<p>With: arithmetic = 13,959,176,192 × ${fint(d.N-d.H)} + 1,050,673,152 + 2 × 32 × 4,096 × (${fint(d.N)} × ${fint(d.N+1)} − ${fint(d.H)} × ${fint(d.H+1)}) = ${fint(w.fl)} FLOP ÷ ${fixd(pk/1e12,2)} TFLOP/s = ${fixd(w.tc*1e3,2)} ms; memory as without, ${fixd(w.tm*1e3,2)} ms. The longer of each pair is the time to first token.</p>`,
    `<p><b>用缓存 ≈ ${fT(w.t)}（${w.cb?'计算':'读写显存'}决定）；不用缓存 ≈ ${fT(wo.t)}（${wo.cb?'计算':'读写显存'}决定）；缓存的 prompt 占 ${gib2(mem)}。</b></p>`+
    `<p>缓存：⌊${fint(d.P)} ÷ 16⌋ = ${fint(d.H/BLOCK)} 个 block = ${fint(d.H)} 个 token；${fint(d.H)} × 131,072 = ${fint(mem)} 字节 = ${gib2(mem)}。</p>`+
    `<p>不用缓存：计算 = 13,959,176,192 × ${fint(d.N)} + 1,050,673,152 + 2 × 32 × 4,096 × ${fint(d.N)} × ${fint(d.N+1)} = ${fint(wo.fl)} FLOP ÷ ${fixd(pk/1e12,2)} TFLOP/s = ${fixd(wo.tc*1e3,2)} ms；读写显存 = ${fint(W_READ)} + ${fint(d.N)} × 131,072 = ${fint(wo.by)} 字节 ÷ ${fixd(bw/1e12,2)} TB/s = ${fixd(wo.tm*1e3,2)} ms。</p>`+
    `<p>用缓存：计算 = 13,959,176,192 × ${fint(d.N-d.H)} + 1,050,673,152 + 2 × 32 × 4,096 × (${fint(d.N)} × ${fint(d.N+1)} − ${fint(d.H)} × ${fint(d.H+1)}) = ${fint(w.fl)} FLOP ÷ ${fixd(pk/1e12,2)} TFLOP/s = ${fixd(w.tc*1e3,2)} ms；读写显存和不用缓存时一样，${fixd(w.tm*1e3,2)} ms。每组取较长的那个，就是首 token 延迟。</p>`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const CTLN={P:{en:'Shared prefix',zh:'共享前缀'},N:{en:'Shared prefix',zh:'共享前缀'},new:{en:'Own tokens per request',zh:'每个请求自己的 token'},hitb:{en:'Shared prefix',zh:'共享前缀'}};
const TERM={
 N:()=>{const x=selRes(); return tx(`${fint(x.N)} is the number of prompt tokens of ${reqOf(selIdx())}: every one of them is attended to, and its key and value are read or written.`,`${fint(x.N)} 是${reqOf(selIdx())}的 prompt token 数：每个 token 都参与 attention，它的 key 和 value 都要读或写。`);},
 new:()=>{const x=selRes(); return tx(`${fint(x.N-x.hit)} is the number of tokens prefill computes: the prompt minus the ${fint(x.hit)} cached ones.`,`${fint(x.N-x.hit)} 是 prefill 要算的 token 数：prompt 减去缓存的 ${fint(x.hit)} 个。`);},
 hitb:()=>{const x=selRes(); return tx(`${fint(x.hitBlocks)} is the number of blocks, from the first, whose keys are in the cache. A hit is always whole blocks of 16 tokens.`,`${fint(x.hitBlocks)} 是从第一个 block 起、key 在缓存里的 block 数。命中总是整块的，每块 16 个 token。`);},
 wr:()=>tx(`${fint(W_READ)} bytes are the weights every step reads once: all ${fint(P_LAYERS+P_HEAD)} parameters it multiplies, 2 bytes each in BF16.`,`${fint(W_READ)} 字节是每一步都要读一遍的权重：参与乘法的全部 ${fint(P_LAYERS+P_HEAD)} 个参数，BF16 下每个 2 字节。`),
 kv:()=>tx('Every token keeps a key and a value of 128 numbers for each of the 8 KV heads in each of the 32 layers, 2 bytes per number in BF16 (from the config.json).','每个 token 在 32 层、每层 8 个 KV head 上各存一个 key 和一个 value，都是 128 维，BF16 下每个数值 2 字节（来自 config.json）。'),
 kvt:()=>tx('131,072 bytes (128 KiB) is the KV cache of one token: prefill writes it for every new token and reads it for every cached one.','131,072 字节（128 KiB）是一个 token 的 KV cache：prefill 为每个新 token 写入一份，为每个缓存的 token 读一份。'),
 wl:()=>tx(`2 FLOPs, a multiply and an add, for each of the ${fint(P_LAYERS)} parameters of the layers: every computed token pays this; cached tokens do not.`,`层里的 ${fint(P_LAYERS)} 个参数，每个 2 FLOP（一次乘、一次加）：每个现算的 token 都要这么多，缓存的 token 不用。`),
 wh:()=>tx(`2 FLOPs for each of the LM head’s ${fint(P_HEAD)} parameters: it scores the vocabulary once, for the first token of the answer.`,`LM head 的 ${fint(P_HEAD)} 个参数，每个 2 FLOP：只为回答的第一个 token 给词表打一次分。`),
 att0:()=>{const x=selRes(); return tx(`Attention over the whole prompt: token i meets the keys of tokens 1 to i, so ${fint(x.N)} × ${fint(x.N+1)} ÷ 2 query–key pairs, at 4 × 32 × 4,096 FLOPs a pair (score and value, 32 layers).`,`对整个 prompt 的 attention：第 i 个 token 和第 1 到第 i 个 token 的 key 相遇，一共 ${fint(x.N)} × ${fint(x.N+1)} ÷ 2 对，每对 4 × 32 × 4,096 FLOP（得分和 value，32 层）。`);},
 att1:()=>{const x=selRes(); return tx(`Attention of the new tokens only: tokens ${fint(x.hit+1)} to ${fint(x.N)} each meet every key before them, cached ones included: (${fint(x.N)} × ${fint(x.N+1)} − ${fint(x.hit)} × ${fint(x.hit+1)}) ÷ 2 pairs.`,`只算新 token 的 attention：第 ${fint(x.hit+1)} 到第 ${fint(x.N)} 个 token 各自和它前面的每个 key 相遇，包括缓存的：（${fint(x.N)} × ${fint(x.N+1)} − ${fint(x.hit)} × ${fint(x.hit+1)}）÷ 2 对。`);},
 effc:()=>tx(`${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s is 50% of the ${GPUN}’s dense BF16 peak, 989.5 TFLOP/s. The 50% is this lab’s assumption.`,`${fixd(H100.peak*EFF.mfu/1e12,2)} TFLOP/s 是 ${GPUN} 稠密 BF16 峰值 989.5 TFLOP/s 的 50%。50% 是本 Lab 的假设。`),
 effm:()=>tx(`${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s is 60% of the ${GPUN}’s peak memory bandwidth, 3.35 TB/s. The 60% is this lab’s assumption.`,`${fixd(H100.bw*EFF.mbu/1e12,2)} TB/s 是 ${GPUN} 峰值显存带宽 3.35 TB/s 的 60%。60% 是本 Lab 的假设。`)
};
/* part 5's terms: one per position (otb-front …), so that each line's hover speaks of its own numbers */
const OT_TERM=k=>{const m=/^ot([bhrwas])-(front|middle|end)$/.exec(k); if(!m) return '';
  const o=otRun(m[2]), r=o.r2, t=otTok(m[2]), hb=r.hitBlocks, H=r.hit, N=r.N;
  if(m[1]==='b') return tx(`Token ${fint(t)} sits in block ${fint(o.blk+1)}: ⌊(${fint(t)} − 1) ÷ ${BLOCK}⌋ + 1, since each block holds ${BLOCK} tokens. The blocks before it hold the same tokens as request 1’s.`,`第 ${fint(t)} 个 token 在第 ${fint(o.blk+1)} 个 block 里，即 ⌊(${fint(t)} − 1) ÷ ${BLOCK}⌋ + 1，因为每个 block 放 ${BLOCK} 个 token。它之前的 block 和请求 1 的 token 一样。`);
  if(m[1]==='h') return hb===0?tx('No token comes from the cache, because no block comes before the changed one.','没有 token 来自缓存，因为改动的 block 之前没有别的 block。')
    :tx(`${fint(H)} tokens come from the cache: the ${fint(hb)} blocks before the changed one, ${BLOCK} tokens each.`,`${fint(H)} 个 token 来自缓存：改动的 block 之前的 ${fint(hb)} 个 block，每个 ${BLOCK} 个 token。`);
  if(m[1]==='r') return tx(`${fint(N-H)} tokens are recomputed: the changed block and every block after it, to the end of the ${fint(N)}-token prompt.`,`要重算 ${fint(N-H)} 个 token：改动的 block 和它之后的每个 block，一直到 ${fint(N)} 个 token 的 prompt 末尾。`);
  if(m[1]==='w') return o.dup===1?tx(`1 block: the system prompt’s ${fint(o.preB)} blocks minus the ${fint(hb)} that request 2 found in the cache. Request 2 stores this block under its own new key, next to request 1’s copy, so the cache now holds it twice.`,`1 个 block：system prompt 的 ${fint(o.preB)} 个 block，减去请求 2 在缓存里找到的 ${fint(hb)} 个。请求 2 用自己新的 key 把它存进缓存，和请求 1 的那份并排，所以这个 block 现在存了两份。`)
    :tx(`${fint(o.dup)} blocks: the system prompt’s ${fint(o.preB)} blocks minus the ${fint(hb)} that request 2 found in the cache. Request 2 stores each of them under its own new key, next to request 1’s copy, so the cache now holds these blocks twice.`,`${fint(o.dup)} 个 block：system prompt 的 ${fint(o.preB)} 个 block，减去请求 2 在缓存里找到的 ${fint(hb)} 个。请求 2 用自己新的 key 把它们存进缓存，和请求 1 的那份并排，所以这些 block 现在存了两份。`);
  if(m[1]==='s') return tx(`${fint(o.recSys)} of the system prompt’s tokens are recomputed: the ${fint(BASE.P)} tokens minus the ${fint(H)} found in the cache. The question’s ${fint(BASE.O)} tokens are new at every position, so they are not counted here.`,`system prompt 里要重算 ${fint(o.recSys)} 个 token：${fint(BASE.P)} 个减去缓存里找到的 ${fint(H)} 个。问题的 ${fint(BASE.O)} 个 token 在每个位置都是新的，所以这里不算。`);
  return tx(`Attention of the recomputed tokens only: tokens ${fint(H+1)} to ${fint(N)} each meet every key before them, cached ones included: (${fint(N)} × ${fint(N+1)} − ${fint(H)} × ${fint(H+1)}) ÷ 2 pairs, at 4 × ${LL.layers} × ${fint(LL.hidden)} FLOPs a pair.`,`只算重算的 token 的 attention：第 ${fint(H+1)} 到第 ${fint(N)} 个 token 各自和它前面的每个 key 相遇，包括缓存的：（${fint(N)} × ${fint(N+1)} − ${fint(H)} × ${fint(H+1)}）÷ 2 对，每对 4 × ${LL.layers} × ${fint(LL.hidden)} FLOP。`);};
TERM.otc1=()=>{const r=otRun(otPos||'middle'); return tx(`${fint(r.r1.cached)} blocks: request 1’s prompt and its ${ANS}-token answer, less the answer’s last token (which has no keys and values yet), ${fint(r.r1.N+ANS-1)} tokens in all, fill ⌊${fint(r.r1.N+ANS-1)} ÷ ${BLOCK}⌋ = ${fint(r.r1.cached)} full blocks. Only full blocks are cached.`,`${fint(r.r1.cached)} 个 block：请求 1 的 prompt 加上它 ${ANS} 个 token 的回答，减去回答的最后一个 token（它还没有 key 和 value），一共 ${fint(r.r1.N+ANS-1)} 个 token，写满 ⌊${fint(r.r1.N+ANS-1)} ÷ ${BLOCK}⌋ = ${fint(r.r1.cached)} 个 block。只有写满的 block 才进缓存。`);};
TERM.otans=()=>{const r=otRun(otPos||'middle'); return tx(`${fint(r.ansB)} blocks: the ${fint(r.r1.cached)} full blocks of prompt and answer minus the prompt’s own ${fint(r.r1.nb)}. The answer first fills the prompt’s last, half-full block, then ${fint(r.ansB-1)} more.`,`${fint(r.ansB)} 个 block：prompt 加回答的 ${fint(r.r1.cached)} 个完整 block，减去 prompt 自己的 ${fint(r.r1.nb)} 个。回答先把 prompt 最后那个半满的 block 写满，再写满 ${fint(r.ansB-1)} 个。`);};
TERM.otq=()=>{const r=otRun(otPos||'middle'), q=r.nw-r.dup; return tx(`${fint(q)} blocks: the question’s ${fint(BASE.O)} tokens start right after the system prompt and fill ${fint(q)} blocks of ${BLOCK}; its last ${fint(BASE.O-q*BLOCK)} tokens wait for the answer to fill their block.`,`${fint(q)} 个 block：问题的 ${fint(BASE.O)} 个 token 紧接在 system prompt 后面，写满 ${fint(q)} 个 ${BLOCK} token 的 block；最后 ${fint(BASE.O-q*BLOCK)} 个 token 要等回答把它们的 block 写满。`);};
TERM.otc2=()=>{const r=otRun(otPos||'middle'); return tx(`${fint(r.r2.cached)} blocks in the cache after request 2: request 1’s ${fint(r.r1.cached)} and request 2’s ${fint(r.r2.cached-r.r1.cached)} new ones. The memory is ample here, so nothing is evicted.`,`请求 2 之后缓存里有 ${fint(r.r2.cached)} 个 block：请求 1 的 ${fint(r.r1.cached)} 个，加上请求 2 新存的 ${fint(r.r2.cached-r.r1.cached)} 个。这里显存很充足，所以什么也没被淘汰。`);};
TERM.otN=()=>tx(`${fint(BASE.P+BASE.O)} tokens: the mission’s ${fint(BASE.P)}-token system prompt and a ${fint(BASE.O)}-token question, the same lengths in both requests.`,`${fint(BASE.P+BASE.O)} 个 token：任务里 ${fint(BASE.P)} 个 token 的 system prompt，加上 ${fint(BASE.O)} 个 token 的问题，两个请求一样长。`);
const termText=k=>{const tm=TERM[k]; return tm?tm():OT_TERM(k);};
const termTipHTML=k=>termText(k)?`<div>${esc(termText(k))}</div><div class="act">${jumpAct(k)}</div>`:'';
const FIXED=new Set(['wr','kv','kvt','wl','wh','att0','att1']);
const DER={effc:'eff',effm:'eff'};
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='tl') return tx('Click to scroll to the timeline in the chart.','点击滚到图里的时间线。');
  if(k==='alq') return tx('Click to open day 1 and day 2 side by side, with the answer under them.','点击展开第 1 天和第 2 天的对照图，回答在图下面。');
  if(k==='pair') return tx('Click to scroll to the two requests in the chart.','点击滚到图里的两个请求。');
  const om=/^ot[bhrwas]-(front|middle|end)$/.exec(k); if(om) return tx(`Click to jump to “${otName(om[1])}” in the card.`,`点击跳到卡片里的“${otName(om[1])}”。`);
  if(['otN','otc1','otans','otq','otc2'].includes(k)||(isOT()&&CTLN[k])) return tx('Fixed in this experiment: the mission’s 10,000-token system prompt and 200-token questions; only the changed token’s place changes.','这个实验里固定不变：任务里 10,000 个 token 的 system prompt 和 200 个 token 的问题；只有改动的 token 的位置可以换。');
  if(k==='eff') return mode==='challenge'?tx('Fixed during the challenge: 50% of the peak FLOP/s and 60% of the peak bandwidth.','挑战中固定不变：峰值 FLOP/s 的 50%，峰值带宽的 60%。')
    :(isOT()?tx('Fixed in this experiment: 50% of the peak FLOP/s and 60% of the peak bandwidth.','这个实验里固定不变：峰值 FLOP/s 的 50%，峰值带宽的 60%。'):tx('Click to jump to “Assumed values” under Settings.','点击跳到“自由调参”里的“假设值”。'));
  if(FIXED.has(k)) return tx(`Fixed in this lab: it always uses ${MODEL} on one ${GPUN}.`,`这个 Lab 里固定不变：始终是一张 ${GPUN} 跑 ${MODEL}。`);
  if(DER[k]) return jumpAct(DER[k]);
  if(!CTLN[k]) return '';
  if(mode==='challenge') return tx('Fixed during the challenge: the six parts and their lengths; only their order changes.','挑战中固定不变：六个部分和它们的长度；只能改顺序。');
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({P:$('#px-ctl-P',app),N:$('#px-ctl-P',app),hitb:$('#px-ctl-P',app),new:$('#px-ctl-O',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#px-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='alq'){const el=$('#px-al',app), d=$('#px-alx',app); A.tip.hide(); if(d) d.open=true; if(el){el.scrollIntoView({block:'start',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}   // tall: its top, not its middle; it opens the closed side-by-side first
  if(k==='tl'||k==='pair'){const el=$(k==='tl'?'#px-tl':'#px-pairbox',app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  const om=/^ot[bhrwas]-(front|middle|end)$/.exec(k); if(om){const el=$(`#px-rail [data-lever="${om[1]}"]`,app); A.tip.hide(); if(el) A.ui.jumpToEl(el); return;}
  if(['otN','otc1','otans','otq','otc2'].includes(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(k==='controls'){const c=$('#px-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='eff'){const el=$('#px-ctl-eff',app); if(el&&!el.closest('.locked')) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(jumpAct('eff'))}</div>`); return;}
  if(FIXED.has(k)){if(src) A.tip.show(src,termTipHTML(k)); return;}
  if(DER[k]){jump(DER[k],src); return;}
  if(!CTLN[k]) return;
  if(mode==='challenge'||$('#px-controls',app).classList.contains('locked')){if(src) A.tip.show(src,`<div>${esc(mode==='challenge'||isOT()?jumpAct(k):tx('The settings can change after you answer part 2.','在第 2 部分作答后，才能改设置。'))}</div>`); return;}
  const el=ctlEl(k); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...(mode==='guided'?STEPS[step].enter().s:PRESETS[0].s)}; sel=S.n-1; render();}); return;}
  if(a==='send'){if(sent) return; sent=true; sel=1; sweepNext=true; wantObs=true; byLearner(()=>render()); updateRail(); revealSoon(); wantObs=false; return;}
  if(a==='submit'){chSubmitted=true; if(chRun(chOrder).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='seq'){startSeq(); return;}
  if(a==='ot-step'){otStep(); return;}
  if(a==='ot-replay'){if(otOpen()){A.tip.hide(); swStart(); updateRail();} return;}   // the lookup's replay control (R3.20)
  if(a==='ot-res'){const box=$('#px-rail [data-obs]',app); A.tip.hide(); if(box&&!box.hidden) A.ui.reveal(box); return;}
  if(a==='up'||a==='down'){const p=btn.dataset.part, i=chOrder.indexOf(p), j=a==='up'?i-1:i+1; if(i<0||j<0||j>=chOrder.length) return;
    [chOrder[i],chOrder[j]]=[chOrder[j],chOrder[i]]; byLearner(()=>{redrawOrder(p,a); render();}); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#px-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const selr=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,selr); else selr();}catch(err){selr();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2) */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#px-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='P') bySettings(e.target,()=>setState({P:P_STOPS[v]}));
    else if(k==='O') bySettings(e.target,()=>setState({O:O_STOPS[v]}));
    else if(k==='n') bySettings(e.target,()=>setState({n:v}));});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#px-rail')&&!b.closest('[data-exits]')&&!d.act) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.id==='send'){act('send',b); return;} if(s.showMe){const was=s.id==='cost'?seqKeys():null; wantObs=true; byLearner(()=>setState({...s.showMe})); if(s.id==='cost') playSeq(was); revealSoon(); wantObs=false;} return;}   // part 6: like a pick, only the rows that change arrive (R3.20)
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#px-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    /* part 5: a position (in the card or under request 2 in the chart), the prediction (then request 2's blocks are looked up) */
    if(d.otpos!=null){otPick(d.otpos,!!d.otrail); return;}
    if(d.lever!=null&&isOT()){otPick(d.lever,true); return;}
    if(d.otpred){if(otPred||!otPos||!isOT()) return; const inRail=!!b.closest('#px-rail'); otPred=d.otpred; otRail=inRail; A.tip.hide();
      byLearner(()=>swStart(true)); if(innerWidth<=760) otShowStage();   // on a phone the lookup always plays in view (review r1 #1)
      const fb=inRail?$(`#px-rail [data-lever="${otPos}"]`,app):$(`#px-stage [data-otpos="${otPos}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null){const s=STEPS[step]; wantObs=true;
      if(s.id==='break') byLearner(()=>setState({lever:d.lever}));
      else if(s.id==='cost'){const v=d.lever==='pool'?'pool':+d.lever, was=seqKeys(); byLearner(()=>setState({mem:v})); playSeq(was);}   // R3.20: only rows that change arrive again
      requestAnimationFrame(()=>{if(app) revealBox($('#px-rail [data-obs]',app));}); wantObs=false; return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#px-insight',app));}); return;}
    if(d.sel!=null){sel=+d.sel; byLearner(()=>render()); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#px-rail',app)); return;}
    if(d.v!=null){const g=b.closest('.seg-ctl'); if(g&&g.dataset.key&&!b.disabled){const k=g.dataset.key, v=d.v; bySettings(b,()=>setState({[k]:k==='apps'?+v:(k==='mem'?(v==='pool'?'pool':+v):v)})); return;}}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let resumed=false;   // restore() ran: the next mount draws the saved state as it is
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>({v:1,mode,step,S:{...S},sel,pred,sent,observed:{...observed},saved:saved?{S:{...saved.S},sel:saved.sel}:null,chOrder:chOrder.slice(),chSubmitted,chReveal,
    lever,memLever,leverTried:[...leverTried],memTried:[...memTried],otPos,otPred,otTried:[...otTried],otDone:[...otDone],seqDone:[...seqDoneT],
    sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chRun(chOrder).ok}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    const ots=OT_POS.map(x=>x.v), arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    otPos=ots.includes(o.otPos)?o.otPos:null; otPred=otPos&&OT_PRED.some(x=>x.v===o.otPred)?o.otPred:null;
    S=isOT()?otS():p; sel=Number.isInteger(o.sel)&&o.sel>=0&&o.sel<(mode==='challenge'?CH.stream.length:S.n)?o.sel:Math.min(1,S.n-1);
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; sent=o.sent===true; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'&&o.saved.S&&typeof o.saved.S==='object'?parse(token(o.saved.S)):null;
    saved=mode==='challenge'&&sv?{S:sv,sel:Number.isInteger(o.saved.sel)&&o.saved.sel>=0&&o.saved.sel<sv.n?o.saved.sel:sv.n-1}:null;
    const co=Array.isArray(o.chOrder)?o.chOrder:[]; chOrder=co.length===CH.parts.length&&CH.parts.every(x=>co.includes(x))?co.slice():CH.parts.slice();
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    lever=CHANGES.includes(o.lever)?o.lever:'none'; memLever=o.memLever==='pool'||[1,2,3].includes(o.memLever)?o.memLever:'pool';
    leverTried.clear(); arr(o.leverTried,x=>CHANGES.includes(x)).forEach(x=>leverTried.add(x));
    memTried.clear(); arr(o.memTried,x=>[1,2,3].includes(x)).forEach(x=>memTried.add(x));
    otTried.clear(); arr(o.otTried,x=>ots.includes(x)).forEach(x=>otTried.add(x)); otDone.clear(); arr(o.otDone,x=>ots.includes(x)).forEach(x=>otDone.add(x));
    seqDone.clear(); seqDoneT.clear(); arr(o.seqDone,x=>typeof x==='string'&&!!parse(x)).slice(0,40).forEach(x=>{seqDoneT.add(x); seqDone.add(seqKeys(parse(x)).join('|'));});
    sbTouched=o.sbTouched===true; if(Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6) drillSeed=o.drillSeed; drillOpen=o.drillOpen===true; resumed=true;},
  mount(el,tok){
    wantObs=false; root=el; swStop();                                // a sweep never outlives the view it drew into
    const p=tok&&!resumed?parse(tok):null; resumed=false;   /* after restore() the address names the saved settings: nothing to apply */ if(p){S={...p}; mode='sandbox'; sbTouched=true; sel=S.n-1;}
    if(mode==='challenge'&&!saved) saved={S:{...PRESETS[0].s},sel:PRESETS[0].s.n-1};
    el.innerHTML=viewHTML(); app=$('#px-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#px-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0){const e=STEPS[0].enter(); S={...e.s}; sel=e.sel;}
    const rail=$('#px-rail',app); rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):''); A.ui.initRanges(rail);
    render(); watchStage();
  },
  leave(){clearTimeout(annT); stageStop(); swStop(); stopSeq(); if(stRO){stRO.disconnect(); stRO=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#px-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#px-math',app).open=true;}},
  onResize(){if(app){if(isOT()) drawPair(); else {drawStrip(); drawTL();}}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&(isOT()?$('#px-pair',app):$('#px-stripbox',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#px-graph',app),
    render({status,vis,read}){if(!app) return; const so=statusOf();
      status.className='status '+so.cls; status.textContent=so.short;
      /* part 5: request 1 until the prediction; then request 2 as far as its blocks have been looked up */
      if(isOT()){const o=otRun(otPos||'middle'), r1=o.r1, r=o.r2, T=r.total;
        if(!otOpen()){vis.innerHTML=`<div class="px-dk">${runsOf(r1.state).replace('px-runs','px-runs px-dk-runs')}</div>`;
          read.innerHTML=esc(tx(`Request 1 computed all ${fint(r1.N)} of its tokens. Request 2 appears after your prediction.`,`请求 1 现算了全部 ${fint(r1.N)} 个 token。请求 2 在你预测之后显示。`)); return;}
        const k=swK==null?T:swK;
        vis.innerHTML=`<div class="px-dk">${runsOf(r.state.slice(0,k).concat(new Array(T-k).fill(3))).replace('px-runs','px-runs px-dk-runs')}</div>`;
        /* the same scope as the counters (review r1 #3): the system prompt's tokens that request 2 finds in the cache (R6.21: sentences) */
        if(k<T){read.innerHTML=esc(k===0?tx('Request 2 has not looked up any block yet.','请求 2 还没有查找任何 block。'):tx(`Request 2 has looked up blocks 1 to ${fint(k)} of its ${fint(T)}.`,`请求 2 已经查找了第 1 到第 ${fint(k)} 个 block（共 ${fint(T)} 个）。`)); return;}
        read.innerHTML=(r.hit===0?esc(tx(`Request 2 finds none of its ${fint(BASE.P)} system-prompt tokens in the cache, and its first token comes after ${fT(r.w.t)}`,`请求 2 在缓存里一个 system prompt token 也没找到，首 token 要等 ${fT(r.w.t)}`))+`<span class="dk-mem">${esc(tx(', as long as without the cache','，和不用缓存时一样'))}</span>`
          :esc(tx(`Request 2 finds ${fint(r.hit)} of its ${fint(BASE.P)} system-prompt tokens in the cache, and its first token comes after ${fT(r.w.t)}`,`请求 2 在缓存里找到了 ${fint(BASE.P)} 个 system prompt token 里的 ${fint(r.hit)} 个，首 token 在 ${fT(r.w.t)} 后出现`))+`<span class="dk-mem">${esc(tx(` (${fT(r.wo.t)} without the cache)`,`（不用缓存要 ${fT(r.wo.t)}）`))}</span>`)+tx('.','。');
        return;}
      if(mode==='challenge'){const r=chRun(chOrder);
        vis.innerHTML=`<div class="px-dk"><i class="px-dk-kv" style="width:${Math.min(100,r.avg/r.avgN*100).toFixed(2)}%"></i><b class="px-dk-goal" style="left:${(CH.goal/r.avgN*100).toFixed(2)}%"></b></div>`;
        read.innerHTML=esc(tx(`${fint(r.avg)} of about ${fint(aboutN(r))} prompt tokens per request from the cache`,`平均每个请求约 ${fint(aboutN(r))} 个 prompt token 里有 ${fint(r.avg)} 个来自缓存`))+`<span class="dk-mem">${esc(tx(' (goal: at least 4,400)','（目标：至少 4,400）'))}</span>`; return;}
      const x=selRes(), i=selIdx();
      vis.innerHTML=`<div class="px-dk">${runsOf(x.state).replace('px-runs','px-runs px-dk-runs')}</div>`;
      read.innerHTML=x.hit===0?esc(tx(`${reqName(i)}: none of ${fint(x.N)} tokens cached, first token after ${fT(x.w.t)}`,`${reqName(i)}：${fint(x.N)} 个 token 一个也没缓存，${fT(x.w.t)} 出首 token`))+`<span class="dk-mem">${esc(tx(' (the same without the cache)','（不用缓存也一样）'))}</span>`
        :esc(tx(`${reqName(i)}: ${fint(x.hit)} of ${fint(x.N)} tokens cached, first token after ${fT(x.w.t)}`,`${reqName(i)}：${fint(x.N)} 个 token 里缓存 ${fint(x.hit)} 个，${fT(x.w.t)} 出首 token`))+`<span class="dk-mem">${esc(tx(` (${fT(x.wo.t)} without the cache)`,`（不用缓存要 ${fT(x.wo.t)}）`))}</span>`;}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__px=Object.freeze({
  ttft:(N,H)=>deep(ttft(N,H)), decode:c=>deep(decode(c)),
  run:s=>{const r=run({...BASE,...s}); return deep({tot:r.tot,res:r.res.map(x=>({N:x.N,hit:x.hit,hitBlocks:x.hitBlocks,total:x.total,same:x.state.filter(v=>v===2).length,w:x.w.t,wo:x.wo.t,wc:x.w.cb,cached:x.cached,apps:x.apps,app:x.req.app}))});},
  chRun:o=>{const r=chRun(o||chOrder); return deep({avg:r.avg,avgN:r.avgN,avgW:r.avgW,avgWo:r.avgWo,ok:r.ok,moves:r.moves,hits:r.res.map(x=>x.hit),N:r.res.map(x=>x.N)});},
  chBestAvg, chMinMoves, drill:s=>deep(makeDrill(s)), memTokens:s=>memTokens({...BASE,...s}),
  /* the challenge's reveal: day 1 against day 2 for one order (GPT-6 review of v9, #2) */
  al:o=>{const d=alData(o); return deep({m:d.m,match:d.match,hit:d.hit,N:d.N,avg:d.avg,a:d.a.map(p=>({cid:p.cid,n:p.n,st:p.st,part:p.part})),b:d.b.map(p=>({cid:p.cid,n:p.n,st:p.st,part:p.part}))});},
  AL_ORDERS:deep(AL_ORDERS),
  /* part 5: one position's result from the engine, and the three counters after k blocks */
  ot:v=>{const o=otRun(v), r=o.r2; return deep({tok:o.tok,blk:o.blk+1,hitBlocks:r.hitBlocks,hit:r.hit,firstMiss:r.hitBlocks+1,N:r.N,nb:r.nb,total:r.total,rec:r.N-r.hit,t:r.w.t,tc:r.w.tc,tm:r.w.tm,wo:r.wo.t,
    same:o.same,s0:o.s0,nw:o.nw,bytes:o.bytes,ansB:o.ansB,cached1:o.r1.cached,cached2:r.cached,state:r.state,state1:o.r1.state,k1:o.K1.key,k2:o.K2.key,
    dup:o.dup,dupBytes:o.dupBytes,recSys:o.recSys,preB:o.preB,stops:otStops(o),right:OT_PRED.map(q=>otRight(q.v,o))});},
  otAt:(v,k)=>{const a=otAt(v,k); return deep({k:a.k,rec:a.rec,t:a.t?a.t.t:null,nw:a.nw,bytes:a.bytes,recAll:a.recAll,nwAll:a.nwAll});},
  fmt:Object.freeze({t:fT,gib2,gb:gbS}),
  consts:Object.freeze({H100:Object.freeze({...H100}),EFF:Object.freeze({...EFF}),P_EMB,P_HEAD,P_LAYER,P_LAYERS,P_TOTAL,KV_TOK,W_READ,W_MEM,BLOCK,ANS,DATE_T,NAME_T,POOL_B,POOL_T,
    P_STOPS:P_STOPS.slice(),O_STOPS:O_STOPS.slice(),BASE:{...BASE},CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,LEVERS:LEVERS.map(o=>o.v),CODE,
    OT_POS:deep(OT_POS),OT_RIGHT,OT_PRED:OT_PRED.map(o=>o.v)}),
  state:()=>({...S,mode,step,pred,sent,sel:selIdx(),observed:{...observed},chOrder:chOrder.slice(),chSubmitted,chReveal,lever,memLever,leverTried:[...leverTried],memTried:[...memTried],stageBusy:stageT!==0,seq:seqN,
    otPos,otPred,otTried:[...otTried],sweep:swK,swStep,otG})
});
})();
