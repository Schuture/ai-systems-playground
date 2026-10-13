/* Vector index lab: how do you find the nearest of 1B vectors in milliseconds? Real indexes (exact search, IVF, HNSW, IVF-PQ and a
   k-d tree) are built and searched in the browser on 5,000 seeded 32-dimensional vectors; cost is counted as vectors compared per
   question; memory and time at real scale come from stated formulas and one stated assumption. Seeded and deterministic:
   tests/vector-index_numbers.py rebuilds every index in Python. Also exposes window.__vix for the tests.
   Names (batch 10 brief S6): 精确搜索 / 近似最近邻 (exact search / ANN), 召回率 recall@10 (the share of the true 10 nearest that were
   found). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='vector-index';

/* ---------- engine: pure functions (ENGINE-START) ---------- */
/* Every coordinate is an integer (the vector times 4,096, rounded), so every squared distance is an exact integer and the Python twin
   reproduces every comparison bit for bit. Ties are broken by the smaller id everywhere. */
const SEED=20261010, D=32, N=5000, NQ=100, NCL=200, NOISE=1.2, SC=4096, K=10, NLIST=50, NTRAIN=2000, KM_IT=8, PQ_K=256, PQ_IT=6, PQ_TRAIN=2500, EFC=40;
const NPROBES=[1,2,4,8,16,32], EFS=[10,16,32,64,128,256], MS=[4,8,16,32], PQMS=[4,8,16], KDD=[2,8,32];
const P2N=300, P2Q=40, P2CL=8, P2NOISE=0.35, P2LIST=16, P2M=4, P2EFC=20;
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* clustered data: ncl random directions; a point = its direction + noise (a sum of four uniforms, so no log or cos), scaled to length 1
   when `unit` (the 32-dimensional sets), then times 4,096 and rounded */
function genSet(seed,d,n,ncl,noise,unit){
  const rnd=mulberry(seed), C=[];
  for(let c=0;c<ncl;c++){const v=new Float64Array(d); for(let k=0;k<d;k++) v[k]=rnd()*2-1; C.push(v);}
  const X=new Int32Array(n*d), v=new Float64Array(d);
  for(let i=0;i<n;i++){const c=Math.floor(rnd()*ncl); let s=0;
    for(let k=0;k<d;k++){const e=rnd()+rnd()+rnd()+rnd()-2; v[k]=C[c][k]+noise*e;}
    if(unit){for(let k=0;k<d;k++) s+=v[k]*v[k]; s=Math.sqrt(s);} else s=1;
    for(let k=0;k<d;k++) X[i*d+k]=Math.floor(v[k]/s*SC+0.5);}
  return X;}
function d2(X,i,Y,j,d){let s=0; const a=i*d, b=j*d; for(let k=0;k<d;k++){const t=X[a+k]-Y[b+k]; s+=t*t;} return s;}
const byDI=(a,b)=>a[0]-b[0]||a[1]-b[1];
function topK(arr,k){arr.sort(byDI); return arr.slice(0,k);}
function exact(B,Qs,qi,k){const out=[]; for(let j=0;j<B.n;j++) out.push([d2(Qs.X,qi,B.X,j,B.d),j]); return topK(out,k).map(x=>x[1]);}
/* k-means on rows `idx` of X (columns o … o+w−1): start from the first k of a seeded shuffle, then `it` rounds of assign + mean
   (rounded to integers); an empty cluster keeps its centre */
function kmeans(X,d,idx,o,w,k,it,seed){
  const rnd=mulberry(seed), p=idx.slice();
  for(let i=0;i<k;i++){const j=i+Math.floor(rnd()*(p.length-i)); const t=p[i]; p[i]=p[j]; p[j]=t;}
  const Cn=new Int32Array(k*w); for(let c=0;c<k;c++) for(let t=0;t<w;t++) Cn[c*w+t]=X[p[c]*d+o+t];
  const as=new Int32Array(idx.length);
  for(let r=0;r<it;r++){
    for(let a=0;a<idx.length;a++) as[a]=nearestC(X,d,idx[a],o,w,Cn,k);
    const sum=new Float64Array(k*w), cnt=new Int32Array(k);
    for(let a=0;a<idx.length;a++){const i=idx[a], c=as[a]; cnt[c]++; for(let t=0;t<w;t++) sum[c*w+t]+=X[i*d+o+t];}
    for(let c=0;c<k;c++) if(cnt[c]) for(let t=0;t<w;t++) Cn[c*w+t]=Math.floor(sum[c*w+t]/cnt[c]+0.5);}
  return Cn;}
function nearestC(X,d,i,o,w,Cn,k){let bd=Infinity, bc=0; for(let c=0;c<k;c++){let s=0; for(let t=0;t<w;t++){const z=X[i*d+o+t]-Cn[c*w+t]; s+=z*z;} if(s<bd){bd=s; bc=c;}} return bc;}
const range=n=>{const a=[]; for(let i=0;i<n;i++) a.push(i); return a;};
/* IVF: k-means cells trained on the first `ntrain` vectors, every vector stored in the list of its nearest centre */
function buildIVF(B,nlist,ntrain,it,seed){
  const Cn=kmeans(B.X,B.d,range(ntrain),0,B.d,nlist,it,seed), lists=[]; for(let c=0;c<nlist;c++) lists.push([]);
  const as=new Int32Array(B.n); for(let i=0;i<B.n;i++){const c=nearestC(B.X,B.d,i,0,B.d,Cn,nlist); as[i]=c; lists[c].push(i);}
  return {Cn,lists,as,nlist,d:B.d};}
function probeOrder(I,Qs,qi){const d=I.d, o=[]; for(let c=0;c<I.nlist;c++){let s=0; for(let t=0;t<d;t++){const z=Qs.X[qi*d+t]-I.Cn[c*d+t]; s+=z*z;} o.push([s,c]);} return o.sort(byDI).map(x=>x[1]);}
function searchIVF(B,I,Qs,qi,nprobe,k){
  const ord=probeOrder(I,Qs,qi), out=[]; let comps=I.nlist;
  for(let p=0;p<nprobe;p++) for(const j of I.lists[ord[p]]){out.push([d2(Qs.X,qi,B.X,j,B.d),j]); comps++;}
  return {ids:topK(out,k).map(x=>x[1]),comps,probed:ord.slice(0,nprobe)};}
/* HNSW after hnswlib's hnswalg.h: levels with P(level ≥ l) = M^−l (coin flips instead of −ln(u)/ln M); greedy descent on the upper
   layers; a best-first search keeping the ef closest on the insertion layers; neighbours chosen by hnswlib's heuristic
   (getNeighborsByHeuristic2); at most M links on upper layers and 2 × M on layer 0, a full list re-pruned by the same heuristic.
   Heaps order by (distance, id). */
const lt=(a,b)=>a[0]<b[0]||(a[0]===b[0]&&a[1]<b[1]), gt=(a,b)=>lt(b,a);
function hpush(h,x,less){h.push(x); let i=h.length-1; while(i>0){const p=(i-1)>>1; if(less(h[i],h[p])){const t=h[i]; h[i]=h[p]; h[p]=t; i=p;} else break;}}
function hpop(h,less){const top=h[0], last=h.pop(); if(h.length){h[0]=last; let i=0; for(;;){const l=2*i+1, r=l+1; let m=i; if(l<h.length&&less(h[l],h[m])) m=l; if(r<h.length&&less(h[r],h[m])) m=r; if(m===i) break; const t=h[i]; h[i]=h[m]; h[m]=t; i=m;}} return top;}
function searchLayer(G,dq,ep,dep,ef,L,vis,mark,trace){
  const C=[], W=[]; hpush(C,[dep,ep],lt); hpush(W,[dep,ep],gt); vis[ep]=mark;
  while(C.length){const c=hpop(C,lt); if(W.length>=ef&&gt(c,W[0])) break;
    if(trace) trace.push(c[1]);
    const nb=G.links[L][c[1]]; for(let t=0;t<nb.length;t++){const e=nb[t]; if(vis[e]===mark) continue; vis[e]=mark; const x=[dq(e),e];
      if(W.length<ef||lt(x,W[0])){hpush(C,x,lt); hpush(W,x,gt); if(W.length>ef) hpop(W,gt);}}}
  return W;}
function heuristic(G,cands,M){   // keeps a candidate only if it is closer to the new point than to every candidate kept before it
  const s=cands.slice().sort(byDI); if(s.length<=M) return s;
  const keep=[];
  for(const c of s){if(keep.length>=M) break; let good=true; for(const k of keep){if(G.dist(k[1],c[1])<c[0]){good=false; break;}} if(good) keep.push(c);}
  return keep;}
function buildHNSW(B,M,efc,seed){
  const rnd=mulberry(seed), n=B.n, G={M,links:[],level:new Int32Array(n),entry:0,top:-1,dist:(a,b)=>d2(B.X,a,B.X,b,B.d)};
  const vis=new Int32Array(n); let mark=0;
  for(let i=0;i<n;i++){
    let l=0; while(rnd()<1/M) l++; G.level[i]=l;
    while(G.links.length<=l) G.links.push([]);
    for(let L=0;L<=l;L++) G.links[L][i]=[];
    if(G.top<0){G.entry=i; G.top=l; continue;}
    const dq=e=>d2(B.X,i,B.X,e,B.d);
    let cur=G.entry, dc=dq(cur);
    for(let L=G.top;L>l;L--){let ch=true; while(ch){ch=false; for(const e of G.links[L][cur]){const de=dq(e); if(de<dc||(de===dc&&e<cur)){dc=de; cur=e; ch=true;}}}}
    for(let L=Math.min(l,G.top);L>=0;L--){
      mark++; const W=searchLayer(G,dq,cur,dc,efc,L,vis,mark,null);
      const sel=heuristic(G,W,M); G.links[L][i]=sel.map(x=>x[1]);
      const mmax=L===0?2*M:M;
      for(const x of sel){const nb=G.links[L][x[1]];
        if(nb.length<mmax) nb.push(i);
        else {const cs=nb.map(e=>[G.dist(x[1],e),e]); cs.push([x[0],i]); G.links[L][x[1]]=heuristic(G,cs,mmax).map(y=>y[1]);}}
      cur=sel[0][1]; dc=sel[0][0];}
    if(l>G.top){G.top=l; G.entry=i;}}
  return G;}
/* a search: greedy descent from the entry point through the upper layers, then a best-first search on layer 0 with ef candidates.
   `rec` records, for the picture, every vector measured (in order) and the walk's hops with the number measured so far. */
function searchHNSW(B,G,Qs,qi,ef,k,rec){
  const ev=[], dq=e=>{ev.push(e); return d2(Qs.X,qi,B.X,e,B.d);}, path=[];
  let cur=G.entry, dc=dq(cur); path.push([G.top,cur,1]);
  for(let L=G.top;L>0;L--){let ch=true; while(ch){ch=false; for(const e of G.links[L][cur]){const de=dq(e); if(de<dc||(de===dc&&e<cur)){dc=de; cur=e; ch=true;}}} path.push([L-1,cur,ev.length]);}
  const vis=new Int32Array(B.n), trace=rec?[]:null, W=searchLayer(G,dq,cur,dc,Math.max(ef,k),0,vis,1,trace);
  const ids=W.slice().sort(byDI).slice(0,k).map(x=>x[1]);
  return rec?{ids,comps:ev.length,path,ev}:{ids,comps:ev.length};}
/* PQ: the vector cut into m pieces; each piece replaced by the nearest of 256 centres (k-means per piece), stored as one byte. A search
   builds the table of the question's squared distance to every centre of every piece (m × 256 small distances, together as much
   work as 256 full comparisons, counted as 256) and adds up m table entries per stored code. IVF-PQ keeps these codes in the IVF
   lists. This lab encodes the vector itself (FAISS's IVF-PQ encodes its offset from the cell's centre). */
function buildPQ(B,m,ntrain,seed){
  const w=B.d/m, CB=[]; for(let p=0;p<m;p++) CB.push(kmeans(B.X,B.d,range(ntrain),p*w,w,PQ_K,PQ_IT,seed+p));
  const codes=new Uint8Array(B.n*m);
  for(let i=0;i<B.n;i++) for(let p=0;p<m;p++) codes[i*m+p]=nearestC(B.X,B.d,i,p*w,w,CB[p],PQ_K);
  return {m,w,CB,codes};}
function searchIVFPQ(B,I,P,Qs,qi,nprobe,k){
  const ord=probeOrder(I,Qs,qi), T=new Float64Array(P.m*PQ_K), out=[]; let comps=I.nlist+PQ_K;
  for(let p=0;p<P.m;p++) for(let c=0;c<PQ_K;c++){let s=0; for(let t=0;t<P.w;t++){const z=Qs.X[qi*B.d+p*P.w+t]-P.CB[p][c*P.w+t]; s+=z*z;} T[p*PQ_K+c]=s;}
  for(let q=0;q<nprobe;q++) for(const j of I.lists[ord[q]]){let s=0; for(let p=0;p<P.m;p++) s+=T[p*PQ_K+P.codes[j*P.m+p]]; out.push([s,j]); comps++;}
  return {ids:topK(out,k).map(x=>x[1]),comps};}
/* k-d tree, by the interview problem's rule: split on axis (depth mod d) at the median of (coordinate, id); exact nearest neighbour
   by branch and bound; counts the vectors it measures */
function buildKD(B){const d=B.d;
  function rec(arr,dep){if(!arr.length) return null; const ax=dep%d; arr.sort((a,b)=>B.X[a*d+ax]-B.X[b*d+ax]||a-b); const m=arr.length>>1;
    return {id:arr[m],ax,v:B.X[arr[m]*d+ax],l:rec(arr.slice(0,m),dep+1),r:rec(arr.slice(m+1),dep+1)};}
  return rec(range(B.n),0);}
function searchKD(B,T,Qs,qi){const d=B.d, seen=[]; let best=Infinity, bi=-1;
  (function go(nd){if(!nd) return; const dd=d2(Qs.X,qi,B.X,nd.id,d); seen.push(nd.id); if(dd<best||(dd===best&&nd.id<bi)){best=dd; bi=nd.id;}
    const diff=Qs.X[qi*d+nd.ax]-nd.v, near=diff<0?nd.l:nd.r, far=diff<0?nd.r:nd.l; go(near); if(diff*diff<=best) go(far);})(T);
  return {id:bi,comps:seen.length,seen};}
/* ---------- the lab's data and its measured indexes (cached; each built the first time it is needed) ---------- */
let BASE=null;
function base(){if(BASE) return BASE; const X=genSet(SEED,D,N+NQ,NCL,NOISE,true);
  const B={X:X.subarray(0,N*D),d:D,n:N}, Q={X:X.subarray(N*D),d:D,n:NQ}, GT=[]; for(let q=0;q<NQ;q++) GT.push(exact(B,Q,q,K));
  BASE={B,Q,GT}; return BASE;}
const recallOf=(ids,q)=>{const g=base().GT[q]; let c=0; for(const x of ids) if(g.includes(x)) c++; return c/K;};
let IVF=null; const ivf=()=>IVF||(IVF=buildIVF(base().B,NLIST,NTRAIN,KM_IT,SEED+1));
const HN={}, PQ={}, STAT={};
const hnsw=M=>HN[M]||(HN[M]=buildHNSW(base().B,M,EFC,SEED+2+M));
const pq=m=>PQ[m]||(PQ[m]=buildPQ(base().B,m,PQ_TRAIN,SEED+100+m));
function avg(fn){const b=base(); let r=0, c=0; const per=[]; for(let q=0;q<NQ;q++){const s=fn(b,q), rc=recallOf(s.ids,q); r+=rc; c+=s.comps; per.push(rc);} return {rec:r/NQ,comps:c/NQ,per};}
function stat(ix,a,b){const k=ix+'|'+a+'|'+b; if(STAT[k]) return STAT[k]; let s;
  if(ix==='flat') s={rec:1,comps:N,per:new Array(NQ).fill(1)};
  else if(ix==='ivf') s=avg((B0,q)=>searchIVF(B0.B,ivf(),B0.Q,q,a,K));
  else if(ix==='hnsw') s=avg((B0,q)=>searchHNSW(B0.B,hnsw(a),B0.Q,q,b,K,false));
  else s=avg((B0,q)=>searchIVFPQ(B0.B,ivf(),pq(a),B0.Q,q,b,K));
  return (STAT[k]=s);}
/* a setting: {ix, np, M, ef, m}; ix ∈ flat | ivf | hnsw | pq (IVF-PQ) */
const statOf=s=>s.ix==='ivf'?stat('ivf',s.np,0):s.ix==='hnsw'?stat('hnsw',s.M,s.ef):s.ix==='pq'?stat('pq',s.m,s.np):stat('flat',0,0);
/* bytes per vector (the per-vector part; the shared centres and codebooks are left out): float32 vector 4d; IVF adds FAISS's 8-byte
   id; HNSW adds its layer-0 links, 2 × M neighbours × 4 bytes (hnswlib; upper layers add a few percent more); PQ stores m one-byte codes
   and, in the IVF lists, the 8-byte id */
function bytesPer(s,d){return s.ix==='flat'?4*d:s.ix==='ivf'?4*d+8:s.ix==='hnsw'?4*d+8*s.M:s.m+8;}
const REAL_D=768, REAL_N=1e9, BW=100e9;   // 768 dims as E5-base-v2; 100 GB/s: this lab's assumption for how fast one server scans vectors in memory
let KDS=null;
function kdStats(){if(KDS) return KDS; const out={}; const b=base();
  for(const d of KDD){let B,Q; if(d===D){B=b.B; Q=b.Q;} else {const X=genSet(SEED+7+d,d,N+NQ,NCL,NOISE,true); B={X:X.subarray(0,N*d),d,n:N}; Q={X:X.subarray(N*d),d,n:NQ};}
    const T=buildKD(B); let c=0; for(let q=0;q<NQ;q++) c+=searchKD(B,T,Q,q).comps; out[d]=c/NQ/N;}
  return (KDS=out);}
/* the 2-D picture: 300 points (8 clusters, not scaled to length 1), 40 candidate questions; the question shown is the first one whose
   nearest cell holds between 3 and 7 of its 10 nearest points, so probing one cell misses some of them */
let PIC=null;
function pic(){if(PIC) return PIC; const X=genSet(SEED+20,2,P2N+P2Q,P2CL,P2NOISE,false);
  const P={X:X.subarray(0,P2N*2),d:2,n:P2N}, Qs={X:X.subarray(P2N*2),d:2,n:P2Q}, I=buildIVF(P,P2LIST,P2N,10,SEED+21);
  let q=0; for(let i=0;i<P2Q;i++){const g=exact(P,Qs,i,K), r=searchIVF(P,I,Qs,i,1,K).ids.filter(x=>g.includes(x)).length; if(r>=3&&r<=7){q=i; break;}}
  const G=buildHNSW(P,P2M,P2EFC,SEED+22), T=buildKD(P);
  PIC={P,Qs,q,I,G,T,gt:exact(P,Qs,q,K),pqc:{}}; return PIC;}
/* PQ in the picture: 2 pieces (x and y), c centres each (1-D k-means) */
function picPQ(c){const p=pic(); if(p.pqc[c]) return p.pqc[c]; const P=p.P, cx=kmeans(P.X,2,range(P2N),0,1,c,10,SEED+30), cy=kmeans(P.X,2,range(P2N),1,1,c,10,SEED+31);
  const snap=[]; for(let i=0;i<P2N;i++) snap.push([cx[nearestC(P.X,2,i,0,1,cx,c)],cy[nearestC(P.X,2,i,1,1,cy,c)]]);
  const qx=p.Qs.X[p.q*2], qy=p.Qs.X[p.q*2+1], o=snap.map((s,i)=>[(s[0]-qx)*(s[0]-qx)+(s[1]-qy)*(s[1]-qy),i]);
  return (p.pqc[c]={cx:Array.from(cx),cy:Array.from(cy),snap,ids:topK(o,K).map(x=>x[1])});}
/* ENGINE-END */

/* ---------- words and helpers ---------- */
const f=(v,dg)=>A.fmt.num(v,dg), int=n=>A.fmt.int(Math.round(n));
const pc1=p=>f(p*100,1)+'%', pc0=p=>f(p*100,0)+'%';
/* a count of stored vectors: K / M / B (R6.18); in Chinese body text 1B is 10 亿, as the design asks */
const NB=n=>n>=1e9?tx(`${f(n/1e9,1)}B`,`${f(n/1e8,0)} 亿`):n>=1e6?`${f(n/1e6,1)}M`:n>=1e3?`${f(n/1e3,1)}K`:int(n);
const LG=()=>A.lang()==='zh'?'zh':'en';
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn vix-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="vix-math-line">${A.ui.math(t)}</div>`;
const MI=t=>A.ui.math(t);
const MB=t=>`<span class="vix-math-line vix-mblk">${A.ui.math(t)}</span>`;   // a formula on its own line inside a sentence's block (exit answers)
/* the k-d tree's share at 32 dimensions is 99.9994%, which one decimal would round to 100% (R6.10) */
const kdPc=(v,short)=>v>0.999&&v<1?(short?'> 99.9%':tx('over 99.9%','99.9% 以上')):pc1(v);
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id]; if(l&&l.st&&l.st!=='planned'&&A.shell.labLink) return A.shell.labLink(id); return esc(l?L(l.s):id);};
const BYT=b=>tx(`${int(b)} bytes`,`${int(b)} 字节`);
const GB=b=>{const g=b/1e9; return (g>=100?int(g):f(g,1))+' GB';};
const secs=s=>s>=1?tx(`${f(s,1)} s`,`${f(s,1)} 秒`):tx(`${f(s*1000,s<0.01?2:1)} ms`,`${f(s*1000,s<0.01?2:1)} 毫秒`);
const IXN={flat:{en:'Exact search',zh:'精确搜索'},ivf:{en:'IVF',zh:'IVF'},hnsw:{en:'HNSW',zh:'HNSW'},pq:{en:'IVF-PQ',zh:'IVF-PQ'}};
function setTxt(s){
  if(s.ix==='flat') return tx('exact search (compare with every vector)','精确搜索（和每个向量都比）');
  if(s.ix==='ivf') return tx(`IVF, probing ${s.np} of ${NLIST} cells`,`IVF，${NLIST} 个簇里查 ${s.np} 个`);
  if(s.ix==='hnsw') return tx(`HNSW, M = ${s.M}, ef = ${s.ef}`,`HNSW，M = ${s.M}，ef = ${s.ef}`);
  return tx(`IVF-PQ, ${s.m} bytes per code, probing ${s.np} of ${NLIST} cells`,`IVF-PQ，每个编码 ${s.m} 字节，${NLIST} 个簇里查 ${s.np} 个`);}
const D_NN=()=>tx('Nearest-neighbour search: given a question’s vector, return the k stored vectors closest to it. Here k = 10, and closest means the smallest distance between vectors of length 1, which is the same order as the highest cosine similarity.','最近邻搜索：给一个问题的向量，返回存着的向量里离它最近的 k 个。这里 k = 10；“最近”指长度为 1 的向量之间距离最小，和余弦相似度最高是同一个顺序。');
const D_ANN=()=>tx('Approximate nearest-neighbour search (ANN) compares the question with only a small part of the vectors. It is much faster, and the price is that it sometimes misses one of the true nearest.','近似最近邻（ANN）只和一小部分向量比。它快得多，代价是偶尔漏掉真正最近的那几个。');
const D_REC=()=>tx('Recall@10 here: of the true 10 nearest vectors (found by exact search), the share the index returned, averaged over 100 questions.','这里的 recall@10：精确搜索找出的真正最近的 10 个向量里，索引找回了几成，在 100 个问题上取平均。');
const D_IVF=()=>tx('IVF (inverted file): k-means splits the vectors into cells around centres; each vector is stored in the list of its nearest centre. A search compares the question with every centre, then only with the vectors in the nprobe nearest cells.','IVF（倒排文件）：先用 k-means 把向量分成很多簇，每簇有一个中心；每个向量存进离它最近的中心的列表里。搜索时，问题先和所有中心比，再只和最近的 nprobe 个簇里的向量比。');
const D_KM=()=>tx('k-means: start with k centres, put every vector with its nearest centre, move each centre to the mean of its vectors, and repeat a few rounds.','k-means：先取 k 个中心，每个向量归到离它最近的中心，再把每个中心挪到它那些向量的平均位置，重复几轮。');
const D_HNSW=()=>tx('HNSW (hierarchical navigable small world): every vector is a node linked to a few nearby nodes. Upper layers hold fewer nodes with long links; a search walks greedily from the top layer down, then explores the closest ef candidates on the bottom layer.','HNSW（分层可导航小世界图）：每个向量是一个节点，连着附近的几个节点。上层节点少、连线长；搜索从最上层贪心地往下走，到最底层时保留最近的 ef 个候选，继续扩展。');
const D_PQ=()=>tx('Product quantization (PQ): cut each vector into m pieces; for each piece k-means learns 256 centres in advance, and the piece is stored as the number of its nearest centre, one byte. A vector becomes m bytes; a distance is looked up in a small table and added up.','乘积量化（PQ）：把每个向量切成 m 段，每一段事先用 k-means 学出 256 个中心；存的时候，每段只记离它最近的那个中心的编号，1 个字节。一个向量只剩 m 个字节；距离靠查一张小表再加起来。');
const D_GB=()=>tx('GB here is 10⁹ bytes and TB is 10¹² bytes (1,000 GB), as disk and memory sizes of vector indexes are usually quoted.','这里 GB 是 10⁹ 字节，TB 是 10¹² 字节（1,000 GB），向量索引的大小通常这样计。');
const D_BW=()=>tx('This lab assumes one server scans the vectors in its memory at 100 GB/s, a conservative value the lab picks. Time = bytes read ÷ 100 GB/s. It leaves out the arithmetic and assumes all vectors fit in memory.','这个 Lab 假设一台服务器扫描内存里的向量每秒 100 GB，这是这个 Lab 取的一个保守值。时间 = 读的字节数 ÷ 100 GB/s。没有算运算时间，也假设向量都放得进内存。');
const D_SRV=()=>tx('This lab assumes one server has 512 GB of memory. A server can be fitted with several TB, at a much higher price; with another budget, what fits changes.','这个 Lab 假设一台服务器有 512 GB 内存。服务器也能装到几 TB，但贵得多；预算不同，放得下放不下也跟着变。');
const D_KD=()=>tx('A k-d tree splits the points in half on one coordinate, then each half on the next coordinate, and so on. A search goes down to the question’s region, then backs up and checks every other region that could still hold a closer point.','k-d 树先按一个坐标把点分成两半，每一半再按下一个坐标分，一直分下去。搜索先走到问题所在的那一小块，再往回查每一块可能还有更近点的区域。');

/* ---------- state ---------- */
const DEFAULT={ix:'flat',np:4,M:16,ef:16,m:8};
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0, realN=1e6, kdD=2;
const triedNP=new Set(), triedEF=new Set(), triedPQ=new Set(), triedN=new Set(), triedD=new Set(), shownPrev={}, seenPic=new Set();
let picSpeed='mid', anim=null, raf=0, lastTs=0, picF=-1;
const PIC_DUR={slow:6,mid:3,fast:1.5};   // seconds for one search in the picture (R3.9: a mechanism replay defaults to 中)
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const cur=()=>({...S});

/* ---------- guided parts ---------- */
const PRED=[{v:'miss'},{v:'all'},{v:'ten'}];
const predTxt=v=>({miss:tx('About 0.1% of the vectors, and it sometimes misses one of the true nearest','约 0.1% 的向量，而且有时会漏掉真正最近的'),
  all:tx('About 0.1% of the vectors, and it always finds the true nearest, because they sit in the nearest cell','约 0.1% 的向量，而且总能找到真正最近的，因为它们都在最近的那个簇里'),
  ten:tx('About 10% of the vectors, and it always finds the true nearest','约 10% 的向量，而且总能找到真正最近的')}[v]);
function inSeg(id,key,label,opts,val){return A.ui.seg({id:'vix-ic-'+id,key,label,options:opts,value:String(val),inline:true});}
const inNP=()=>inSeg('np','np',tx('Cells to probe, nprobe','要查的簇数 nprobe'),NPROBES.map(v=>[String(v),String(v)]),S.np);
const inEF=()=>inSeg('ef','ef',tx('Candidates kept on the bottom layer, ef','最底层保留的候选数 ef'),EFS.map(v=>[String(v),String(v)]),S.ef);
const inPQ=()=>inSeg('m','m',tx('Bytes per code, m','每个编码的字节数 m'),PQMS.map(v=>[String(v),String(v)]),S.m);
const inRN=()=>inSeg('rn','rn',tx('Stored vectors','存的向量数'),[['1e6','1M'],['1e8','100M'],['1e9','1B']],String(realN===1e6?'1e6':realN===1e8?'1e8':'1e9'));
const inKD=()=>inSeg('kd','kd',tx('Dimensions','维数'),KDD.map(v=>[String(v),String(v)]),kdD);
const DIST_TEX=()=>`{\\lVert q - x \\rVert}^2 = 2 - 2\\cos(q, x)`;
const REC_TEX=nm=>`\\text{${nm}} = \\frac{\\text{true nearest found}}{10}`;
const REC_TEX_ZH=nm=>`\\text{${nm}} = \\frac{\\text{找回的真正近邻数}}{10}`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'1B vectors; the nearest 10 in milliseconds.',zh:'10 亿个向量；几毫秒内找出最近的 10 个。'},
  pic:'flat',panels:['pic'],enter:()=>({...DEFAULT}),
  body:()=>tx(`<p>Lumo Home, a smart-home company made up for this lab, wants its help-center search to cover a billion (1B) passages: manuals, forum answers and support tickets.</p><p>Each passage is stored as an embedding, a vector of 768 numbers (as in E5-base-v2; see ${LL('embeddings')}). A customer’s question becomes a vector too, and search has to return the 10 passages whose vectors are closest to it, in a few milliseconds.</p><p>This is ${DEF('nearest-neighbour search',D_NN())}. On vectors of length 1, ranking by distance and ranking by cosine similarity give the same order, because the squared distance between a question q and a stored vector x is</p>${M(DIST_TEX())}<p>The higher the cosine, the smaller the distance. Comparing the question with every stored vector takes far too long at this size, so search goes through a <b>vector index</b>: a structure built once, before any question arrives, that groups, links or compresses the vectors so that each search compares the question with only a small part of them. A vector database builds one when you store vectors and searches through it on every query; this lab builds the main kinds and measures what each one gives up.</p><p>${J('pic','The picture')} shows the idea in 2 dimensions: stored points, a question (the cross) and its 10 nearest points, ringed.</p><p>The numbers in this lab all come from 5,000 vectors of 32 dimensions that your browser indexes and searches.</p>`,
              `<p>小鹿家居是为这个 Lab 编的一家智能家居公司。它想让帮助中心的搜索覆盖 10 亿（1B）段文字：说明书、论坛回答和客服工单。</p><p>每一段都存成一个 768 维的 embedding 向量（和 E5-base-v2 一样，见 ${LL('embeddings')}）。用户的问题也会变成一个向量。搜索要在几毫秒内，找出向量离它最近的 10 段。</p><p>这就是${DEF('最近邻搜索',D_NN())}。长度为 1 的向量，按距离排和按余弦相似度排，顺序是一样的，因为问题 q 和存着的向量 x 之间距离的平方是</p>${M(DIST_TEX())}<p>余弦越大，距离就越小。向量这么多，问题和每个都比一次就太慢了，所以搜索要靠<b>向量索引</b>：一种在问题到来之前就建好的结构，把向量分好组、连成图，或者压缩，让每次搜索只和其中一小部分向量比。向量数据库在你存入向量时建好它，每次查询都通过它来搜；这个 Lab 会建出主要的几种，并测出每种放弃了什么。</p><p>${J('pic','图')}在 2 维里画出这件事：存着的点、一个问题（叉号），以及离它最近的 10 个点（带圈）。</p><p>这个 Lab 里的数字，都来自你的浏览器建好索引并搜索过的 5,000 个 32 维向量。</p>`)},
 {id:'exact',title:{en:'Exact search',zh:'精确搜索'},lead:{en:'Compare with every vector: right, and slow.',zh:'和每个向量都比一次：结果对，但是慢。'},
  pic:'flat',panels:['pic','metrics'],enter:()=>({...DEFAULT}),target:'rn',expect:()=>triedN.size>1,showMe:{rn:1e9},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another number of vectors to continue.',zh:'换一个向量数再继续。'},
  body:()=>tx(`<p>The simplest search compares the question with every stored vector and keeps the 10 closest. It is exact: what it returns is always the true 10 nearest, none missed. Its cost is one comparison per stored vector, so it grows with the number of vectors; watch it compare every point in ${J('pic','the picture')}.</p><p>At real scale each 768-dimensional vector is 3,072 bytes of float32.</p><p>So 1B vectors take 3,072 ${DEF('GB',D_GB())} (here a GB is 10⁹ bytes), and every one is read for every question. Pick how many vectors the help center stores:</p>`,
              `<p>最简单的搜索，是把问题和每个存着的向量都比一次，留下最近的 10 个。它是精确的：找到的一定就是真正最近的 10 个，一个不漏。代价是每存一个向量，就要多比一次，所以向量越多越慢；看${J('pic','图')}里它把每个点都比了一遍。</p><p>到了真实规模，每个 768 维向量用 float32 存是 3,072 字节。</p><p>所以 10 亿个向量就是 3,072 ${DEF('GB',D_GB())}（这里 GB 是 10⁹ 字节），每个问题都要把它们全部读一遍。选帮助中心存多少个向量：</p>`)+inRN(),
  obs:()=>obsExact()},
 {id:'trees',title:{en:'Why trees stop working',zh:'为什么树不管用了'},lead:{en:'A k-d tree prunes in 2 dimensions, not in 32.',zh:'k-d 树在 2 维能剪枝，到 32 维就不行了。'},
  pic:'kd',panels:['pic'],enter:()=>({...DEFAULT}),target:'kd',expect:()=>triedD.size>1,showMe:{kd:32},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another number of dimensions to continue.',zh:'换一个维数再继续。'},
  body:()=>tx(`<p>Databases speed up a search with trees. A ${DEF('k-d tree',D_KD())} cuts space in half again and again; in ${J('pic','the picture')} the lines are its cuts and the solid dots are the only points the search had to compare. The search may skip a whole region only when the region’s edge is farther from the question than the nearest point it has found so far.</p><p>In many dimensions that almost never happens. There, the distance from a question to most cut lines is about the same as the distance to its nearest neighbour, so almost no region can be skipped and the search has to go back and check nearly every one. Pick a number of dimensions and see what share of the 5,000 vectors the tree compares:</p>`,
              `<p>数据库常用树来加速搜索。${DEF('k-d 树',D_KD())}把空间一次次对半切开；${J('pic','图')}里的线就是它切的位置，实心的点是搜索真正比较过的点。只有当一块区域的边界离问题，比目前找到的最近点还远时，搜索才能跳过这整块区域。</p><p>维数多了，这种情况几乎不会出现。在高维里，问题到大多数切线的距离，和它到最近邻的距离差不多，所以几乎没有哪块区域能跳过，搜索要回过头去把几乎每一块都查一遍。选一个维数，看树要比较 5,000 个向量里的多少：</p>`)+inKD(),
  obs:()=>obsKD()},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'1,000 cells, search only the nearest one.',zh:'1,000 个簇，只查最近的那一个。'},
  pic:'cells',panels:['pic'],enter:()=>({...DEFAULT,ix:'ivf',np:1}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(predTxt(o.v))}</button>`).join('')}</div>`;
   return tx(`<p>Exact search is slow because it compares the question with every vector. With a vector index a search compares it with only a small part, at the price of sometimes missing one of the true nearest; that is why this is called ${DEF('approximate nearest-neighbour search (ANN)',D_ANN())}.</p><p>A common way to choose that part is ${DEF('IVF',D_IVF())} (inverted file): before any search, ${DEF('k-means',D_KM())} splits the vectors into cells, each with a centre, and every vector is stored in the list of its nearest centre (${J('pic','the picture')} shows 16 cells). A search first compares the question with the centres, then only with the vectors of the nearest cells; how many cells it opens is called nprobe.</p><p>Suppose, to keep the arithmetic simple, that the help center’s 1B vectors sit in 1,000 cells of about the same size.</p><p>It searches with nprobe = 1, only the cell whose centre is nearest. What share of the vectors does a question get compared with, and does it always find the true nearest?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
              `<p>精确搜索慢，是因为问题要和每个向量都比。有了向量索引，搜索只和一小部分向量比，代价是偶尔会漏掉真正最近的，所以这种搜索叫${DEF('近似最近邻（ANN）',D_ANN())}。</p><p>挑出这一小部分，常用的一种办法是 ${DEF('IVF',D_IVF())}（倒排文件）：搜索之前，先用 ${DEF('k-means',D_KM())} 把向量分成很多簇，每簇有一个中心，每个向量存进离它最近的中心的列表（${J('pic','图')}里有 16 个簇）。搜索时，问题先和各个中心比，再只和最近几个簇里的向量比；打开几个簇，叫 nprobe。</p><p>为了好算，假设帮助中心的 10 亿个向量分在 1,000 个大小差不多的簇里。</p><p>搜索时 nprobe = 1，只查中心最近的那一个簇。一个问题要和多大比例的向量比？它总能找到真正最近的吗？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'ivf',title:{en:'IVF: probe more cells',zh:'IVF：多查几个簇'},lead:{en:'More cells, higher recall, more comparisons.',zh:'簇查得越多，召回率越高，比较也越多。'},
  pic:'ivf',panels:['pic','chart','metrics'],curves:['ivf'],enter:()=>({...DEFAULT,ix:'ivf',np:1}),target:'np',expect:()=>triedNP.size>1,showMe:{np:16},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another nprobe to continue.',zh:'换一个 nprobe 再继续。'},
  body:()=>{const ok=pred==='miss', v=pred?tx(`<p class="${ok?'okline':'badline'}"><b>${ok?'Right.':'Not quite.'}</b>${ok?'':` You predicted “${esc(predTxt(pred))}”. The right one is the first.`}</p><p>One cell of 1,000 holds about 0.1% of the vectors, and a true neighbour can sit just across a cell border, in a cell whose centre is farther away.</p>`,`<p class="${ok?'okline':'badline'}"><b>${ok?'对了。':'不对。'}</b>${ok?'':`你预测的是“${esc(predTxt(pred))}”。正确的是第一项。`}</p><p>1,000 个簇里的一个，大约装着 0.1% 的向量；而真正的近邻可能就在簇边界的另一侧，落在中心更远的那个簇里。</p>`):tx('<p>The answer: about 0.1% of the vectors, and a true neighbour across a cell border is missed.</p>','<p>答案：约 0.1% 的向量，而且簇边界另一侧的真正近邻会被漏掉。</p>');
   return v+tx(`<p>In ${J('pic','the picture')} the true neighbours in red rings sit in cells that were not opened. Each extra cell catches more of them and costs more comparisons.</p><p>The lab’s 5,000 vectors are in ${NLIST} cells.</p><p>The ${DEF('recall@10',D_REC())}, the recall this lab reports, counts how many of the true 10 nearest a search returns:</p>${M(REC_TEX('recall@10'))}<p>Pick nprobe and watch ${J('chart','the chart')}: every dot is one setting, recall against the comparisons.</p>`,
                 `<p>${J('pic','图')}里带红圈的真正近邻，都在没打开的簇里。每多查一个簇，就多找回几个，也多比较一些向量。</p><p>这个 Lab 的 5,000 个向量分在 ${NLIST} 个簇里。</p><p>${DEF('recall@10',D_REC())}（这个 Lab 说的召回率）数的是，真正最近的 10 个里，搜索找回了几个：</p>${M(REC_TEX_ZH('recall@10'))}<p>选一个 nprobe，看${J('chart','曲线图')}：每个点是一种设置，横轴是比较的次数，纵轴是召回率。</p>`)+inNP();},
  obs:()=>obsIVF()},
 {id:'hnsw',title:{en:'HNSW: walk a graph',zh:'HNSW：在图上走'},lead:{en:'Follow links toward the question.',zh:'顺着连线往问题走。'},
  pic:'hnsw',panels:['pic','chart','metrics'],curves:['ivf','hnsw'],enter:()=>({...DEFAULT,ix:'hnsw',M:16,ef:10}),target:'ef',expect:()=>triedEF.size>1,showMe:{ef:64},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another ef to continue.',zh:'换一个 ef 再继续。'},
  body:()=>tx(`<p>IVF still compares the question with whole cells. ${DEF('HNSW',D_HNSW())} instead links every vector to a few of its nearest vectors when it is added. A search walks this graph: from an entry point, it moves to whichever linked vector is closer to the question, until none is.</p><p>The graph has several layers. Every vector is on the bottom layer; a few vectors, picked at random, are also put on the layers above, which have fewer points and longer links. So the search first takes a few hops on the upper layers to cross most of the space, then continues on the bottom layer. In ${J('pic','the picture')}, dashed hops are on upper layers and solid dots are the vectors compared; the 2-dimensional picture uses M = 4 (explained below) so its links stay readable.</p><p>A plain walk stops at the first point with no closer neighbour, which may not be the nearest. So on the bottom layer HNSW keeps the ef points closest to the question found so far: in each round, the search takes from these ef points the one closest to the question whose neighbours it has not checked yet, and computes the distance from the question to each of that point’s neighbours; any neighbour closer than the farthest of the ef points replaces that farthest point. It stops once every one of the ef has had its neighbours checked, so a larger ef stops later and misses less.</p><p>When the graph is built, each vector links to at most M neighbours on each upper layer and at most ${MI('2 \\times M')} on the bottom layer (M = 16 here).</p>`,
              `<p>IVF 还是要把问题和整个簇里的向量都比一遍。${DEF('HNSW',D_HNSW())} 换了个办法：每加入一个向量，就把它和离它最近的几个向量连起来。搜索时在这张图上走：从入口出发，每一步走到连着的、离问题更近的那个向量，直到没有更近的为止。</p><p>这张图分好几层。所有向量都在最底层；随机挑出的少数向量，还会放进上面几层，上层的点少，连线长。所以搜索先在上层走几步，跨过大半个空间，再到最底层接着走。${J('pic','图')}里虚线是上层的几步，实心的点是比较过的向量；2 维的示意图用的是 M = 4（M 下面会讲），这样连线看得清。</p><p>只这样走，会停在第一个“邻居都不更近”的点上，它不一定是最近的。所以到了最底层，HNSW 一直留着到目前为止离问题最近的 ef 个点。每一轮，搜索从这 ef 个点里取出一个还没看过邻居、离问题最近的点，算出问题到它每个邻居的距离；如果某个邻居比这 ef 个点里最远的那个还近，就用这个邻居替换掉最远的那个。等这 ef 个点的邻居都看过了，搜索才停下。ef 越大，搜索停得越晚，越不容易漏掉。</p><p>建图时，每个向量在上层最多连 M 个邻居，在最底层最多连 ${MI('2 \\times M')} 个（这里 M = 16）。</p>`)+inEF(),
  obs:()=>obsHNSW()},
 {id:'pq',title:{en:'PQ: compress the vectors',zh:'PQ：压缩向量'},lead:{en:'A few bytes per vector, some recall lost.',zh:'每个向量只剩几个字节，召回率掉一些。'},
  pic:'pq',panels:['pic','chart','metrics'],curves:['ivf','pq'],enter:()=>({...DEFAULT,ix:'pq',m:8,np:16}),target:'m',expect:()=>triedPQ.size>1,showMe:{m:16},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another number of bytes to continue.',zh:'换一个字节数再继续。'},
  body:()=>tx(`<p>Speed is not the only limit: 1B vectors take 3,072 GB, far more than the memory of one server as this lab assumes it.</p><p>${DEF('Product quantization (PQ)',D_PQ())} shrinks each vector to m bytes. The lab’s 32 dimensions are cut into m pieces, and for each piece k-means learns 256 centres in advance.</p><p>To store a vector, each piece keeps only the number of its nearest centre: a number from 0 to 255, exactly one byte. In the 2-dimensional ${J('pic','picture')} the two pieces are the x and the y coordinate. Each piece can only take the value of one of its centres, so every point has moved to the grid the centres form.</p><p>To search, each piece of the question is first compared with that piece’s 256 centres, giving a table. A stored vector’s distance is its m numbers looked up in the table and added, so the search ranks the moved points, not the real ones.</p><p>IVF-PQ combines the two: like IVF it opens only the nearest few cells, and inside them it compares the compressed codes instead of the full vectors. Pick how many bytes each code gets (IVF-PQ probing 16 cells). More bytes cut each vector into more pieces, so points move less and recall rises, but each vector also takes more memory.</p>`,
              `<p>速度不是唯一的限制：10 亿个向量是 3,072 GB，远超这个 Lab 假设的一台服务器的内存。</p><p>${DEF('乘积量化（PQ）',D_PQ())}把每个向量压到 m 个字节。这个 Lab 把 32 维切成 m 段，每一段事先用 k-means 学出 256 个中心。</p><p>存一个向量时，每一段只记下离它最近的那个中心的编号。编号在 0 到 255 之间，正好一个字节。2 维的${J('pic','示意图')}里，两段就是 x 坐标和 y 坐标。每一段只能取它某个中心的值，所以每个点都挪到了这些中心组成的网格上。</p><p>搜索时，先算出问题的每一段到这一段 256 个中心的距离，记成一张表。一个向量的距离，就是按它存的 m 个编号查表，再加起来，所以排的是挪过以后的点，不是原来的点。</p><p>IVF-PQ 把两者合起来：像 IVF 一样只打开最近的几个簇，在这些簇里比较的是压缩后的编码，而不是完整的向量。选每个编码用几个字节（IVF-PQ 查 16 个簇）。字节越多，向量切的段越多，每个点挪得越少，召回率越高，但每个向量占的内存也越多。</p>`)+inPQ(),
  obs:()=>obsPQ()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Speed, memory, recall: each index trades them.',zh:'速度、内存、召回率：每种索引都在取舍。'},
  pic:'hnsw',panels:['chart','metrics'],curves:['ivf','hnsw','pq'],enter:()=>({...DEFAULT,ix:'hnsw',M:16,ef:64}),
  body:()=>rememberHTML()}];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
function obsExact(){const bytes=realN*REAL_D*4, t=bytes/BW;
  return tx(`<p><b>${NB(realN)} vectors: ${GB(bytes)} of float32, ${secs(t)} to read once.</b></p><p>Every question reads all of it, so the time (at an ${DEF('assumed 100 GB/s',D_BW())}) grows in step with the number of vectors.</p><p>At 1B even one question takes far longer than the few milliseconds the mission asks for.</p><p>Memory runs out too: 3,072 GB is far more than one ordinary server holds (${DEF('this lab assumes 512 GB',D_SRV())}).</p>`,
            `<p><b>${realN>=1e9?NB(realN)+'个':NB(realN)+' 个'}向量：float32 共 ${GB(bytes)}，读一遍约 ${secs(t)}。</b></p><p>每个问题都要把它们全读一遍，所以时间（按${DEF('假设的每秒 100 GB',D_BW())} 算）和向量数一起涨。</p><p>到 10 亿个时，一个问题就远远超过任务要的几毫秒。</p><p>内存也不够：3,072 GB 远超一台普通服务器的内存（${DEF('这个 Lab 按一台服务器 512 GB 内存算',D_SRV())}）。</p>`);}
function obsKD(){const k=kdStats(), v=k[kdD];
  return tx(`<p><b>${kdD} dimensions: the tree compares ${kdPc(v)} of the vectors to find the nearest.</b></p><p>${kdD===2?'In 2 dimensions almost every region is pruned.':kdD===8?'In 8 dimensions it already backs up into many regions.':'In 32 dimensions it compares almost every vector, no faster than exact search.'} Embeddings have hundreds of dimensions, so exact trees do not help. The indexes in the next parts take another route: they accept missing a true nearest neighbour now and then, in exchange for far fewer comparisons.</p>`,
            `<p><b>${kdD} 维：为了找到最近的那个，树要比较的向量占 5,000 个里的 ${kdPc(v)}。</b></p><p>${kdD===2?'2 维时，几乎每块区域都被剪掉了。':kdD===8?'8 维时，已经要回头查很多块区域。':'32 维时，它几乎把每个向量都比较了一遍，并不比精确搜索快。'}embedding 有几百维，所以精确的树帮不上忙。后面几部分的索引换了一种思路：允许偶尔漏掉一个真正的最近邻，换来少得多的比较。</p>`);}
function obsIVF(){const s=stat('ivf',S.np,0), s1=stat('ivf',1,0);
  return tx(`<p><b>${S.np===1?'1 cell':`${S.np} cells`} probed: recall ${pc1(s.rec)} with ${int(s.comps)} comparisons.</b></p><p>${S.np===1?'One cell is cheap but misses the neighbours across its borders.':`With 1 cell, recall is ${pc1(s1.rec)}; the neighbours found beyond that sit in cells across its borders.`}</p><p>Each extra cell costs about ${int(N/NLIST)} more comparisons, the size of an average cell; the ${NLIST} comparisons with the centres are included.</p><p>Real systems use far more cells: this lab assumes 1B vectors in 65,536 cells, instead of the 1,000 used earlier to keep the arithmetic simple. More cells make each one smaller, so the same recall takes more probes.</p><p>At that scale, probing 16 cells takes comparisons (the 65,536 centres included) equal to only about 0.03% of the 1B vectors.</p>`,
            `<p><b>查 ${S.np} 个簇时，召回率是 ${pc1(s.rec)}，每个问题比较 ${int(s.comps)} 次。</b></p><p>${S.np===1?'只查一个簇很便宜，但会漏掉边界另一侧的近邻。':`只查 1 个簇时，召回率是 ${pc1(s1.rec)}；多找回的近邻，都在边界另一侧的簇里。`}</p><p>每多查一个簇，就多比较约 ${int(N/NLIST)} 个向量（一个簇平均的大小）；和 ${NLIST} 个中心的比较也算在里面。</p><p>真实系统的簇要多得多：这个 Lab 假设 10 亿个向量分在 65,536 个簇里，而不是前面为了好算用的 1,000 个。簇越多，每个簇越小；要找回同样多的近邻，要打开的簇也越多。</p><p>在这个规模下查 16 个簇，比较的次数（连同和 65,536 个中心的比较）只有 10 亿个向量的约 0.03%。</p>`);}
function obsHNSW(){const s=stat('hnsw',S.M,S.ef), b=stat('hnsw',S.M,10), iv=stat('ivf',16,0), h32=stat('hnsw',S.M,32);
  return tx(`<p><b>ef = ${S.ef}: recall ${pc1(s.rec)} with ${int(s.comps)} vectors compared.</b></p><p>${S.ef===10?'With only 10 candidates the walk often settles near, but not at, the true neighbours.':`More candidates find more true neighbours, for more comparisons (at ef = 10: ${pc1(b.rec)}).`}</p><p>At about the same recall, HNSW compares far fewer vectors than IVF on ${J('chart','the chart')}: at ef = 32 HNSW compares ${int(h32.comps)} and reaches ${pc1(h32.rec)}; IVF compares ${int(iv.comps)} to reach ${pc1(iv.rec)}.</p><p>The price is memory: each vector also stores the ids of up to ${MI('2 \\times M')} neighbours, 4 bytes each.</p><p>At M = ${S.M}: ${8*S.M} more bytes, on a 128-byte vector.</p>`,
            `<p><b>ef = ${S.ef} 时，召回率是 ${pc1(s.rec)}，每个问题比较 ${int(s.comps)} 个向量。</b></p><p>${S.ef===10?'只留 10 个候选时，搜索常常停在真正近邻的附近，而不是它们本身。':`候选留得越多，找回的真正近邻越多，比较也越多（ef = 10 时是 ${pc1(b.rec)}）。`}</p><p>召回率差不多时，HNSW 比${J('chart','曲线图')}上的 IVF 比较得少得多：ef = 32 时，HNSW 比较 ${int(h32.comps)} 次，召回率 ${pc1(h32.rec)}；IVF 要比较 ${int(iv.comps)} 次，召回率才到 ${pc1(iv.rec)}。</p><p>代价是内存：每个向量还要存最多 ${MI('2 \\times M')} 个邻居的编号，每个 4 字节。</p><p>M = ${S.M} 时，每个向量多 ${8*S.M} 字节；这里的向量本身是 128 字节。</p>`);}
function obsPQ(){const s=stat('pq',S.m,16), iv=stat('ivf',16,0);
  return tx(`<p><b>${S.m} bytes per code: recall ${pc1(s.rec)}.</b></p><p>IVF with the full vectors, also probing 16 cells, reaches ${pc1(iv.rec)}. ${S.m===16?'More bytes keep each piece closer to the real vector, so fewer neighbours are swapped.':'Fewer bytes move each point farther, so close vectors swap places in the ranking.'}</p><p>In this lab, comparing with a code counts the same as comparing with a full vector. In fact a code comparison is only m table lookups added up and reads far fewer bytes, so at the same count IVF-PQ is much faster.</p><p>Each vector takes ${S.m} + 8 bytes (code and id), not 136.</p><p>At real scale the vectors have 768 dimensions, and this lab assumes a 64-byte code for each (12 dimensions per piece). 1B vectors each store a 64-byte code and an 8-byte id.</p><p>That is 72 GB in all, which fits in one server’s memory as this lab assumes it. FAISS’s README says compressed codes “can scale to billions of vectors in main memory on a single server”.</p>`,
            `<p><b>每个编码 ${S.m} 字节时，召回率是 ${pc1(s.rec)}。</b></p><p>同样查 16 个簇，用完整向量的 IVF 是 ${pc1(iv.rec)}。${S.m===16?'字节越多，每一段离真实的向量越近，排名里被换掉的近邻越少。':'字节越少，每个点挪得越远，离得近的向量在排名里就会互换位置。'}</p><p>在这个 Lab 里，比较一个编码和比较一个完整向量都算一次。实际上，比较一个编码只是查 m 次表再相加，读的字节也少得多，所以比较次数一样时，IVF-PQ 要快得多。</p><p>每个向量占 ${S.m} + 8 字节（编码和 id），而不是 136 字节。</p><p>到了真实规模，向量是 768 维，这个 Lab 假设每个编码 64 字节（每段 12 维）。10 亿个向量各存 64 字节的编码和 8 字节的 id。</p><p>一共 72 GB，按这个 Lab 的假设，一台服务器的内存放得下。FAISS 的 README 说，压缩编码“can scale to billions of vectors in main memory on a single server”。</p>`);}
function rememberHTML(){const hn=stat('hnsw',16,64), iv=stat('ivf',16,0), q8=stat('pq',8,16);
  return tx(`<ul><li>Only exact search misses nothing, and its cost grows with the number of vectors: at 1B, one question reads 3,072 GB.</li><li>Before picking an index, check which limit you hit first: memory or search speed. If memory holds the full vectors, HNSW needed the fewest comparisons in this lab.<br>At ef = 64, for example, it compares ${int(hn.comps)} and reaches ${pc1(hn.rec)}.</li><li>If memory does not hold them, PQ codes in IVF lists fit 1B vectors in one server as this lab assumes it, and the recall lost is the price.</li><li>Every approximate index has a setting that trades recall for work: nprobe for IVF, ef for HNSW. Raise it until recall is high enough for your use, measured on your own data.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Approximate means inaccurate.” The misses can be measured and reduced: in this lab IVF probing 16 cells finds ${pc1(iv.rec)}, and probing more raises it further.</p><p>✗ “99% recall is always needed.” It depends on the product.</p><p>A RAG answer that needs one good passage among 10 may lose nothing at 90%; a duplicate-detection job may need every match.</p><p>✗ “HNSW uses the least memory.” It stores the full vector plus its links: here M = 16 adds 128 bytes per vector.</p><p>IVF-PQ, 8-byte codes: 16 bytes per vector, recall ${pc1(q8.rec)}.</p></div>${A.shell.toScenarios()}`,
            `<ul><li>只有精确搜索一个不漏，但它的代价随向量数增长：10 亿个向量时，一个问题要读 3,072 GB。</li><li>选索引之前，先看先碰到哪个限制：内存不够，还是搜索太慢。内存放得下完整的向量时，在这个 Lab 里 HNSW 比较的次数最少。<br>比如 ef = 64 时，它只比较 ${int(hn.comps)} 个，召回率就到 ${pc1(hn.rec)}。</li><li>如果放不下，就把 PQ 编码放进 IVF 的列表。按这个 Lab 的假设，10 亿个向量一台服务器放得下，代价是掉一些召回率。</li><li>每种近似索引都有一个可调的参数，用计算量换召回率：IVF 是 nprobe，HNSW 是 ef。把它往上调，直到召回率够你的用途，并且要在你自己的数据上测。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “近似就是不准。”漏掉多少能测出来，也能调低：在这个 Lab 里，IVF 查 16 个簇时召回率是 ${pc1(iv.rec)}，多查几个簇还能更高。</p><p>✗ “召回率一定要 99%。”要看产品。</p><p>RAG 回答只要前 10 段里有一段好的，90% 可能就够了；查重复内容的任务，可能一个都不能漏。</p><p>✗ “HNSW 最省内存。”它要存完整的向量，再加上连线：这里 M = 16 时，每个向量多 128 字节。</p><p>每个编码 8 字节的 IVF-PQ，每个向量只占 16 字节；在这个 Lab 里召回率是 ${pc1(q8.rec)}。</p></div>${A.shell.toScenarios()}`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {t:{en:'Initial setup: exact search',zh:'初始设置：精确搜索'},d:{en:'Compare with all 5,000 vectors.',zh:'和全部 5,000 个向量比。'},p:{ix:'flat'}},
 {t:{en:'IVF, probe 1 cell',zh:'IVF，只查 1 个簇'},d:{en:'FAISS’s default nprobe is 1.',zh:'FAISS 的 nprobe 默认是 1。'},p:{ix:'ivf',np:1}},
 {t:{en:'IVF, probe 16 cells',zh:'IVF，查 16 个簇'},d:{en:'16 of the 50 cells.',zh:'50 个簇里查 16 个。'},p:{ix:'ivf',np:16}},
 {t:{en:'HNSW with hnswlib’s defaults',zh:'HNSW，用 hnswlib 的默认值'},d:{en:'M = 16, ef = 10.',zh:'M = 16，ef = 10。'},p:{ix:'hnsw',M:16,ef:10}},
 {t:{en:'HNSW with FAISS’s defaults',zh:'HNSW，用 FAISS 的默认值'},d:{en:'M = 32, efSearch = 16.',zh:'M = 32，efSearch = 16。'},p:{ix:'hnsw',M:32,ef:16}},
 {t:{en:'IVF-PQ, 8 bytes per code',zh:'IVF-PQ，每个编码 8 字节'},d:{en:'Probe 16 cells; 16 bytes per vector with the id.',zh:'查 16 个簇；加上 id 每个向量 16 字节。'},p:{ix:'pq',m:8,np:16}}];
const presetNow=()=>PRESETS.findIndex(p=>p.p.ix===S.ix&&(S.ix==='flat'||(S.ix==='ivf'&&p.p.np===S.np)||(S.ix==='hnsw'&&p.p.M===S.M&&p.p.ef===S.ef)||(S.ix==='pq'&&p.p.m===S.m&&p.p.np===S.np)));
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how recall@10, the vectors compared and the memory per vector change.','选一个场景，看看 recall@10、比较的向量数和每个向量的内存怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="vix-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(){
  const s=statOf(S), i=presetNow(), by=bytesPer(S,D);
  let h=tx(`<p><b>${esc(setTxt(S))}</b></p><p>Recall ${pc1(s.rec)}, ${int(s.comps)} compared, ${BYT(by)} per vector.</p>`,`<p><b>${esc(setTxt(S))}</b></p><p>召回率 ${pc1(s.rec)}，每个问题比较 ${int(s.comps)} 次，每个向量 ${BYT(by)}。</p>`);
  if(i<0) return h+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  const k=['flat','ivf1','ivf16','hl','hf','pq8'][i];
  if(k==='flat') h+=tx('<p>Every one of the 5,000 vectors is compared, so nothing is missed.</p><p>The cost grows in step with the vectors stored: at 1B it is 200,000 times this.</p>','<p>5,000 个向量每个都比，所以一个都不会漏。</p><p>代价和存的向量数同步增长：到 10 亿个，是这里的 200,000 倍。</p>');
  else if(k==='ivf1') h+=tx(`<p>FAISS opens one cell unless told otherwise. That is fast, but in this lab it finds only ${pc1(s.rec)} of the true 10 nearest: the rest sit across cell borders.</p>`,`<p>不另外设置时，FAISS 只打开一个簇。这很快，但在这个 Lab 里只找回真正最近的 10 个里的 ${pc1(s.rec)}：其余的都在簇边界的另一侧。</p>`);
  else if(k==='ivf16') h+=tx(`<p>A third of the cells finds ${pc1(s.rec)} of the true neighbours and compares about a third of the vectors. With only 5,000 vectors, IVF cannot save much more.</p>`,`<p>查三分之一的簇，找回 ${pc1(s.rec)} 的真正近邻，比较的向量也差不多是三分之一。只有 5,000 个向量时，IVF 省不了更多。</p>`);
  else if(k==='hl') h+=tx(`<p>hnswlib builds with M = 16 and searches with ef = 10 unless told otherwise. To raise recall, raise ef: at ef = 64, recall in this lab goes to ${pc1(stat('hnsw',16,64).rec)}.</p>`,`<p>不另外设置时，hnswlib 用 M = 16 建图，用 ef = 10 搜索。要提高召回率，就调大 ef：在这个 Lab 里调到 64，召回率升到 ${pc1(stat('hnsw',16,64).rec)}。</p>`);
  else if(k==='hf'){const h16=stat('hnsw',16,16);
    h+=tx(`<p>FAISS’s IndexHNSW uses 32 links per layer by default (64 on the bottom).</p><p>That adds ${BYT(8*32)} per vector, twice the vector itself at this lab’s 32 dimensions.</p><p>More links find more neighbours at the same ef, and compare more vectors too.</p><p>At M = 16 (same ef): recall ${pc1(h16.rec)} with ${int(h16.comps)} compared.</p><p>In this lab, at the same number of comparisons the two values of M reach about the same recall, so the extra 128 bytes buy no fewer comparisons.</p>`,
          `<p>FAISS 的 IndexHNSW 默认每层 32 条连线（最底层 64 条）。</p><p>每个向量因此多出 ${BYT(8*32)}，在这个 Lab 的 32 维里，是向量本身的两倍。</p><p>连线多了，同样的 ef 下找回的近邻更多，比较的也更多。</p><p>M = 16 时（ef 一样），召回率是 ${pc1(h16.rec)}，比较 ${int(h16.comps)} 个。</p><p>在这个 Lab 里，按同样的比较次数看，两种 M 的召回率几乎一样。所以多出的 128 字节，没有换来更少的比较。</p>`);}
  else h+=tx(`<p>Each vector is 16 bytes instead of 136, and the search reads codes, not vectors.</p><p>Recall stops lower than IVF’s with full vectors (${pc1(stat('ivf',16,0).rec)}), because the codes move every point.</p><p>A system can recompute the distances of the top results with the full vectors and sort them again, to win some of it back. FAISS’s ${SRC(U_RF,'IndexRefineFlat')} does this, so it has to keep the full vectors as well.</p>`,
             `<p>每个向量 16 字节，而不是 136 字节；搜索读的是编码，不是向量。</p><p>召回率比用完整向量的 IVF（${pc1(stat('ivf',16,0).rec)}）低，因为编码把每个点都挪了位置。</p><p>系统可以用完整的向量，把排在前面的结果重新算一次距离、重新排序，找回一部分。FAISS 的 ${SRC(U_RF,'IndexRefineFlat')} 就是这样做的，所以它也要存完整的向量。</p>`);
  return h;}

/* ---------- challenge ---------- */
const CH_REC=0.95, CH_COMP=1000, CH_INIT={ix:'flat',np:4,M:16,ef:16,m:8};
/* ef can be any whole number in hnswlib (set_ef) and FAISS (efSearch). Outside the menu (EFS), these ef ranges also meet both bars
   in this lab, and no other ef from 10 to 400 does; tests/vector-index_numbers.py scans them (B1 of review b10r1). */
const EF_FREE={8:[70,120],4:[210,244]};
function chAll(){const out=[]; out.push({ix:'flat'}); for(const np of NPROBES) out.push({ix:'ivf',np}); for(const M0 of MS) for(const ef of EFS) out.push({ix:'hnsw',M:M0,ef}); for(const m of PQMS) for(const np of NPROBES) out.push({ix:'pq',m,np});
  return out.map(s=>{const st=statOf(s), by=bytesPer(s,D); return {s,rec:st.rec,comps:st.comps,bytes:by,ok:st.rec>=CH_REC-1e-9&&st.comps<=CH_COMP};});}
const sameSet=(a,b)=>a.ix===b.ix&&(a.ix==='flat'||(a.ix==='ivf'&&a.np===b.np)||(a.ix==='hnsw'&&a.M===b.M&&a.ef===b.ef)||(a.ix==='pq'&&a.m===b.m&&a.np===b.np));
function chCtlHTML(){
  let h=A.ui.seg({id:'vix-ic-chix',key:'ix',label:tx('Index','索引'),options:['flat','ivf','hnsw','pq'].map(k=>[k,L(IXN[k])]),value:S.ix,inline:true});
  if(S.ix==='ivf'||S.ix==='pq') h+=A.ui.seg({id:'vix-ic-chnp',key:'np',label:tx('Cells to probe, nprobe','要查的簇数 nprobe'),options:NPROBES.map(v=>[String(v),String(v)]),value:String(S.np),inline:true});
  if(S.ix==='hnsw') h+=A.ui.seg({id:'vix-ic-chM',key:'M',label:tx('Links per layer, M','每层连线数 M'),options:MS.map(v=>[String(v),String(v)]),value:String(S.M),inline:true})+A.ui.seg({id:'vix-ic-chef',key:'ef',label:tx('Candidates kept, ef','保留的候选数 ef'),options:EFS.map(v=>[String(v),String(v)]),value:String(S.ef),inline:true});
  if(S.ix==='pq') h+=A.ui.seg({id:'vix-ic-chm',key:'m',label:tx('Bytes per code, m','每个编码的字节数 m'),options:PQMS.map(v=>[String(v),String(v)]),value:String(S.m),inline:true});
  return h;}
function challengeHTML(){
  const sub=`<div id="vix-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, the lab searches the 100 questions with your index and tells you recall@10 and the vectors compared.','提交后，Lab 用你的索引搜 100 个问题，再告诉你 recall@10 和比较过的向量数。')}</p></div>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Index the lab’s vectors so that recall reaches 95% with at most 1,000 vectors compared per question, using as few bytes per vector as possible','给这个 Lab 的向量建索引：召回率至少 95%，每个问题最多比较 1,000 个向量，每个向量占的字节越少越好')}</h3>`+
   `<p>${tx('The same 5,000 vectors (32 dimensions) and 100 questions.','还是这 5,000 个 32 维向量、这 100 个问题。')}</p><p>${tx('The task scores bytes per vector because they set the total memory once the data grows.','这道题比的是每个向量占多少字节，因为它决定了规模变大以后的总内存。')}</p><p>${tx('With 1B vectors, each extra byte per vector adds 1 GB in all. Choose the index and its settings:','存 10 亿个向量时，每个向量每多 1 个字节，总共就多 1 GB。选索引和它的设置：')}</p>`+
   `<div id="vix-ch-ctl">${chCtlHTML()}</div>`+
   `<div class="goal mono" id="vix-ch-a"><span>${tx('Recall@10 (goal: at least 95%)','recall@10（目标：至少 95%）')}</span><span><b id="vix-ch-av"></b></span></div>`+
   `<div class="goal mono" id="vix-ch-c"><span>${tx('Vectors compared per question (at most 1,000)','每个问题比较的向量数（最多 1,000）')}</span><span><b id="vix-ch-cv"></b></span></div>`+
   `<div class="goal mono" id="vix-ch-g"><span>${tx('Bytes per vector','每个向量的字节数')}</span><span><b id="vix-ch-gv"></b></span></div>`+
   `<p class="small muted">${tx('The two bars, 95% recall and at most 1,000 comparisons, are this challenge’s own, set for the lab’s 5,000 vectors.','召回率至少 95%、最多比较 1,000 个，这两条线是这道题自己定的，按这个 Lab 的 5,000 个向量定。')}</p>`+
   sub+`<div id="vix-ch-res" class="vix-ch-res" aria-live="polite"></div></div>`;}
const chDone=()=>mode==='challenge'&&chSubmitted;
function chRowLabel(p){const s=p.s; return tx(`${setTxt(s)}: ${int(p.bytes)} B, ${int(p.comps)} compared`,`${setTxt(s)}：${int(p.bytes)} 字节，比较 ${int(p.comps)} 个`);}
function chResultHTML(){
  if(!chDone()) return '';
  const pts=chAll(), me=pts.find(p=>sameSet(p.s,S)), okp=pts.filter(p=>p.ok), minB=Math.min(...okp.map(p=>p.bytes)), best=okp.filter(p=>p.bytes===minB), bb=best[0];
  let h;
  if(me.ok&&me.bytes===minB) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: recall ${pc1(me.rec)}, ${int(me.comps)} compared, ${int(me.bytes)} bytes each.</b></p><p>Among the ef values this task offers, no setting meets both bars with fewer bytes.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：召回率 ${pc1(me.rec)}，比较 ${int(me.comps)} 个向量，每个向量 ${int(me.bytes)} 字节。</b></p><p>在这道题能选的 ef 里，没有哪种设置能用更少的字节同时满足两条线。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: recall ${pc1(me.rec)} with ${int(me.comps)} vectors compared.</b></p><p>But ${esc(setTxt(bb.s))} also meets both bars.</p><p>It takes ${int(bb.bytes)} bytes per vector instead of ${int(me.bytes)}.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：召回率 ${pc1(me.rec)}，比较 ${int(me.comps)} 个向量。</b></p><p>不过 ${esc(setTxt(bb.s))} 也能满足两条线。</p><p>它每个向量只要 ${int(bb.bytes)} 字节，而不是 ${int(me.bytes)} 字节。</p>`);
  else {const why=me.rec<CH_REC?tx(`recall@10 ${pc1(me.rec)}, under 95%`,`recall@10 是 ${pc1(me.rec)}，不到 95%`):tx(`${int(me.comps)} vectors compared, over 1,000`,`比较了 ${int(me.comps)} 个向量，超过 1,000`);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${why}.</b></p><p class="small">Hint: which index finds the true neighbours with the fewest comparisons, and how few links can it keep?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：${why}。</b></p><p class="small">提示：哪种索引用最少的比较找到真正的近邻？它最少能留几条连线？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal&&!me.ok) return h;
  const rows=pts.map(p=>({label:chRowLabel(p),value:p.rec*100,text:pc1(p.rec),ok:p.ok,me:p===me,best:p.ok&&p.bytes===minB,
    tags:(sameSet(p.s,CH_INIT)?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  h+=A.ui.cmpChart({brief:true,title:tx('Recall@10 of the initial setup, your pick and the settings that meet both bars (goal at least 95%, at most 1,000 compared)','初始设置、你的选择和满足两条线的设置，各自的 recall@10（目标是至少 95%，最多比较 1,000 个）'),
    aria:tx('Recall@10 by index and setting.','按索引和设置的 recall@10。'),rows,goal:CH_REC*100,goalLabel:tx('goal: 95%','目标 95%'),max:100,
    hiddenText:n=>tx(`The other ${n} settings miss a bar and are not shown.`,`其余 ${n} 种设置至少没达到一条线，图里没有列出。`),
    okText:tx('meets both bars','两条线都满足'),noText:tx('misses at least one','至少一条没满足'),goalText:tx('goal: 95%','目标：95%')});
  const i16=stat('ivf',16,0), p16=stat('pq',16,32), h8a=stat('hnsw',8,64), h8=stat('hnsw',8,128), h4=stat('hnsw',4,256), P=(e,z)=>`<p>${tx(e,z)}</p>`,
    b8=bytesPer({ix:'hnsw',M:8},D), b4=bytesPer({ix:'hnsw',M:4},D);
  h+=`<div class="small vix-why">`+
    P(`Why: only HNSW can meet both bars. IVF adds about ${int(N/NLIST)} comparisons per cell, and goes over the limit well before recall reaches 95%.`,`原因：只有 HNSW 能同时满足两条线。IVF 每多查一个簇，就多比较约 ${int(N/NLIST)} 个向量；召回率还没到 95%，比较就超过了上限。`)+
    P(`IVF-PQ takes the fewest bytes, but its codes move the points: probing 32 cells it still finds only ${pc1(p16.rec)}.`,`IVF-PQ 每个向量最省，但编码把点挪了位置：查 32 个簇时，召回率也只有 ${pc1(p16.rec)}。`)+
    P(`HNSW keeps the vector’s 128 bytes plus ${MI('8 \\times M')} bytes of links, so a smaller M saves memory.`,`HNSW 要存向量本身的 128 字节，再加 ${MI('8 \\times M')} 字节的连线，所以 M 越小越省。`)+
    P(`With M = 8, ef = 64 reaches only ${pc1(h8a.rec)}, just short.`,`M = 8 时，ef = 64 只到 ${pc1(h8a.rec)}，差一点。`)+
    P(`ef = 128 reaches ${pc1(h8.rec)} but compares ${int(h8.comps)}.`,`ef = 128 到了 ${pc1(h8.rec)}，却要比较 ${int(h8.comps)} 个。`)+
    P(`With M = 4, ef = 256 compares ${int(h4.comps)}.`,`M = 4 时，ef = 256 要比较 ${int(h4.comps)} 个。`)+
    P(`So among the ef values offered here, M = 16 with ef = 64 takes the fewest bytes. But ef can be any whole number (set_ef in hnswlib, efSearch in FAISS).`,`所以在这道题能选的 ef 里，M = 16、ef = 64 用的字节最少。不过 ef 其实可以取任意整数（hnswlib 的 set_ef，FAISS 的 efSearch）。`)+
    P(`With M = 8, any ef from ${EF_FREE[8][0]} to ${EF_FREE[8][1]} passes.`,`M = 8 时，ef 取 ${EF_FREE[8][0]} 到 ${EF_FREE[8][1]} 都能过线。`)+
    P(`That takes only ${b8} bytes per vector: a sparser graph that keeps more candidates still finds the neighbours.`,`这样每个向量只要 ${b8} 字节：连线少的图，多留一些候选也能找全。`)+
    P(`With M = 4, ef from ${EF_FREE[4][0]} to ${EF_FREE[4][1]} also passes.`,`M = 4 时，ef 取 ${EF_FREE[4][0]} 到 ${EF_FREE[4][1]} 也能过线。`)+
    P(`That is ${b4} bytes per vector, the fewest of the graphs this lab builds.`,`每个向量 ${b4} 字节，是这个 Lab 建的几种图里最少的。`)+`</div>`;
  h+=tx(`<div class="small muted vix-why"><p>Note: these numbers come from the lab’s 5,000 vectors.</p><p>At 1B vectors, 3,072 GB of vectors plus links is far more than one server’s memory as this lab assumes it.</p><p>Then there are two ways. Split HNSW over several machines: high recall, more machines. Or compress with IVF-PQ into one machine: fewer machines, lower recall. Measure recall on a sample of your own questions.</p></div>`,
        `<div class="small muted vix-why"><p>注意：这些数来自这个 Lab 的 5,000 个向量。</p><p>到 10 亿个向量时，3,072 GB 的向量加上连线，远超一台服务器的内存（按这个 Lab 的假设）。</p><p>这时有两种做法：把 HNSW 分到多台机器上，召回率高，但机器多；或者用 IVF-PQ 压进一台机器，省机器，召回率低一些。召回率要用你自己的一批问题来测。</p></div>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('explain why vector search is approximate, size an index’s memory, and pick IVF, HNSW or IVF-PQ and set its nprobe or ef for a recall target.','讲清向量搜索为什么是近似的，算出索引要多少内存，并按召回率目标选 IVF、HNSW 或 IVF-PQ，调好它的 nprobe 或 ef。');
const EXITS=()=>({
 explain:[
  {q:tx('IVF with nprobe = 1 sometimes misses one of the true 10 nearest. Why?','IVF 在 nprobe = 1 时，有时会漏掉真正最近的 10 个里的一个。为什么？'),
   o:[tx('A true neighbour can sit just across a cell border, in a cell whose centre is farther from the question, and that cell is never opened.','真正的近邻可能就在簇边界的另一侧，落在中心离问题更远的簇里，而那个簇根本没打开。'),
      tx('k-means puts some vectors in no cell at all.','k-means 会让一些向量不属于任何簇。'),
      tx('The distances inside a cell are rounded, so the order changes.','簇里面的距离被取整了，所以顺序变了。')],a:0,
   n:[null,tx('Every vector is stored in exactly one list, the one of its nearest centre.','每个向量都正好存在一个列表里，就是离它最近的中心的那个。'),tx('IVF compares the full vectors inside a probed cell exactly; only the choice of cells is approximate.','IVF 在打开的簇里用完整的向量精确比较；近似的只是选哪些簇。')],
   why:tx('Cells are chosen by the question’s distance to their centres, not by where its neighbours are; more probes cover the borders.','选簇看的是问题到中心的距离，不是近邻在哪里；多查几个簇，才能盖住边界。')},
  {q:tx('Why does an HNSW index need more memory than the vectors alone?','为什么 HNSW 索引要比向量本身多占内存？'),
   o:[tx(`Each vector also stores the ids of its neighbours: up to ${MI('2 \\times M')} of them on the bottom layer, 4 bytes each.`,`每个向量还要存它邻居的 id：最底层最多 ${MI('2 \\times M')} 个，每个 4 字节。`),
      tx('HNSW keeps several copies of every vector, one per layer.','HNSW 每层都存一份向量。'),
      tx('HNSW stores every distance between pairs of vectors.','HNSW 存了每两个向量之间的距离。')],a:0,
   n:[null,tx('A vector is stored once; the layers only hold more link lists for the few vectors on them.','向量只存一份；上层只是给其中少数向量多存几张连线表。'),tx('That would be N² numbers; HNSW stores only a few links per vector.','那是 N² 个数值；HNSW 每个向量只存几条连线。')],
   why:tx(`The graph is the index: hnswlib puts it at roughly ${MI('M \\times 8')} to ${MI('M \\times 10')} bytes per vector.`,`图就是索引本身：hnswlib 估计每个向量约 ${MI('M \\times 8')} 到 ${MI('M \\times 10')} 字节。`)}],
 transfer:[
  {q:tx('1B vectors of 768 dimensions in float32 take 3,072 GB.</b></p><p><b>An HNSW index on them uses M = 32, so each vector also keeps 64 links on the bottom layer.</b></p><p><b>Each link is 4 bytes. About how much memory does the index need?','10 亿个 768 维的 float32 向量占 3,072 GB。</b></p><p><b>在它们上面建 HNSW，M = 32，所以每个向量在最底层还要存 64 条连线。</b></p><p><b>每条连线 4 字节。这个索引大约要多少内存？'),
   o:[tx('About 3,330 GB: 3,072 GB of vectors and 256 GB of links','约 3,330 GB：3,072 GB 的向量，加 256 GB 的连线'),tx('About 256 GB: only the links','约 256 GB：只有连线'),tx('About 3,072 GB: links take no space','约 3,072 GB：连线不占空间')],a:0,
   why:tx(`${MB('\\text{links} = 10^9 \\times 64 \\times 4\\,\\text{bytes} = 256\\,\\text{GB}')}With the 3,072 GB of vectors, that is far more than one server’s memory as this lab assumes it (512 GB), so the index is split over several machines or compressed.`,
          `${MB('\\text{连线} = 10^9 \\times 64 \\times 4\\,\\text{字节} = 256\\,\\text{GB}')}再加上 3,072 GB 的向量，远超这个 Lab 假设的一台服务器的内存（512 GB），所以要分到多台机器上，或者压缩。`)},
  {q:tx('An IVF index holds 1B vectors in 65,536 cells of about equal size, a number this lab picks.</b></p><p><b>A search probes 64 cells. About how many vectors does one question get compared with?','一个 IVF 索引把 10 亿个向量分在 65,536 个大小差不多的簇里（这是这个 Lab 设的数）。</b></p><p><b>搜索时查 64 个簇，一个问题大约要和多少个向量比较？'),
   o:[tx('About 977K vectors, plus the 65,536 centres','约 977K 个向量，再加 65,536 个中心'),tx('About 64','约 64 个'),tx('About 15,000','约 15,000 个')],a:0,
   why:tx(`${MB('\\text{per cell} = 10^9 \\div 65,536 \\approx 15,259')}${MB('\\text{vectors compared} = 64 \\times 15,259 \\approx 977,000')}That is just under 0.1% of the vectors; the 65,536 centres come on top.`,
          `${MB('\\text{每个簇} = 10^9 \\div 65,536 \\approx 15,259')}${MB('\\text{比较的向量} = 64 \\times 15,259 \\approx 977,000')}这还不到全部向量的 0.1%；另外还要和 65,536 个中心比。`)}],
 practice:[{id:'kd-tree',d:tx('Build and search a k-d tree with branch-and-bound pruning: the tree that compares almost every vector at 32 dimensions in this lab.','建一棵 k-d 树并用分支定界剪枝搜索：在这个 Lab 里，32 维时它几乎要比较每一个向量。')},
           {id:'rag-search-ml',d:tx('Its retrieval part: an ANN index (IVF, PQ, HNSW) and the setting for how widely it searches (nprobe, ef).','其中检索的部分：ANN 索引（IVF、PQ、HNSW），以及决定搜多大范围的参数（nprobe、ef）。')},
           {id:'rag-agent-platform',d:tx('A per-tenant ANN index with permission filters.','按租户分开、带权限过滤的 ANN 索引。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. See why exact search and trees are too slow, predict what IVF misses, then try IVF, HNSW and PQ on the lab’s vectors.','一共 8 个部分。先看精确搜索和树为什么太慢，预测 IVF 会漏掉什么，再在这个 Lab 的向量上试 IVF、HNSW 和 PQ。'),
  sandbox:tx('Pick an index and its settings and compare recall, vectors compared and memory.','选一种索引和它的设置，比较召回率、比较的向量数和内存。'),
  challenge:tx('Choose the index and settings that reach 95% recall@10 with at most 1,000 comparisons and the fewest bytes per vector.','选一个索引和设置，让 recall@10 至少 95%、最多比较 1,000 次，并且每个向量的字节最少。')});
const FPIN='f46050eb56c6458e368496ff74281630fd462e4c', HPIN='ca426729609b3221563047ceb269cc1213e0d376';
const GH=(repo,pin,path)=>`https://github.com/${repo}/blob/${pin}/${path}`;
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const U_IVF=GH('facebookresearch/faiss',FPIN,'faiss/IndexIVF.h'), U_HN=GH('facebookresearch/faiss',FPIN,'faiss/impl/HNSW.h'), U_IHN=GH('facebookresearch/faiss',FPIN,'faiss/IndexHNSW.h'),
  U_PQ=GH('facebookresearch/faiss',FPIN,'faiss/impl/ProductQuantizer.h'), U_CL=GH('facebookresearch/faiss',FPIN,'faiss/Clustering.h'), U_FR=GH('facebookresearch/faiss',FPIN,'README.md'), U_MT=GH('facebookresearch/faiss',FPIN,'faiss/MetricType.h'),
  U_HA=GH('nmslib/hnswlib',HPIN,'ALGO_PARAMS.md'), U_HR=GH('nmslib/hnswlib',HPIN,'README.md'), U_HG=GH('nmslib/hnswlib',HPIN,'hnswlib/hnswalg.h'),
  U_RF='https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing/7f5bf798ab1d07e0a04a79b78714f0db19e653e4';   // FAISS wiki, pinned commit
const FIDPOP=()=>tx(`<p><b>Toy run:</b> your browser builds real indexes on seeded vectors and searches them.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>5,000 vectors of 32 dimensions in 200 clusters, scaled to length 1, and 100 questions drawn the same way (seed ${SEED}). The true 10 nearest come from exact search.</li><li>IVF: k-means (8 rounds) on 2,000 vectors, ${NLIST} cells. HNSW: hnswlib’s insertion, neighbour heuristic and search, efConstruction 40. PQ: 256 centres per piece (k-means on 2,500 vectors), codes in the IVF lists. A k-d tree with branch and bound.</li><li>Cost is the vectors compared per question; memory is bytes per vector by each library’s layout, simplified as below.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>5,000 vectors instead of 1B: the share an index compares, and its recall, change with size. Coordinates are rounded to integers so the Python check matches exactly. HNSW levels use coin flips with the same odds as hnswlib’s formula. PQ encodes the vector itself, not its offset from the cell’s centre as FAISS does, and trains on about 10 vectors per centre, fewer than the 39 below which FAISS warns, so its recall here may be a little low.</li><li>A comparison with a PQ code counts as one, the same as with a full vector, though it is only m table lookups added up and reads far fewer bytes.</li><li>Memory per vector counts the vector, the id and the bottom-layer links. hnswlib also keeps an 8-byte label and a 4-byte link count per vector (3,212 bytes instead of 3,200 at 768 dimensions with M = 16), and the upper layers add a little more.</li><li>Times at real scale assume 100 GB/s of memory reads and leave out the arithmetic. One server is assumed to have 512 GB of memory.</li></ul><div class="eyebrow">Sources</div><ul><li>FAISS 1.15.1: ${SRC(U_IVF,'IndexIVF.h')} (nprobe = 1), ${SRC(U_HN,'impl/HNSW.h')} (efConstruction 40, efSearch 16), ${SRC(U_IHN,'IndexHNSW.h')} (M = 32), ${SRC(U_PQ,'ProductQuantizer.h')}, ${SRC(U_MT,'MetricType.h')} (8-byte ids), ${SRC(U_FR,'README')}.</li><li>hnswlib: ${SRC(U_HA,'ALGO_PARAMS.md')} (M, memory), ${SRC(U_HR,'README')} (M = 16, ef_construction = 200), ${SRC(U_HG,'hnswalg.h')} (2 × M links on layer 0, ef = 10, the per-vector layout).</li><li>FAISS wiki: ${SRC(U_RF,'Pre- and post-processing')} (IndexRefineFlat).</li></ul></div></div>`,
  `<p><b>玩具实跑：</b>页面在你的浏览器里，用随机种子生成的向量建出真实的索引，再去搜索。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>5,000 个 32 维向量，分在 200 个簇附近，缩放成长度 1；100 个问题按同样的方式生成（随机种子 ${SEED}）。真正最近的 10 个由精确搜索给出。</li><li>IVF：在 2,000 个向量上跑 k-means（8 轮），${NLIST} 个簇。HNSW：照 hnswlib 的插入、选邻居的规则和搜索，efConstruction 40。PQ：每段 256 个中心（在 2,500 个向量上跑 k-means），编码放在 IVF 的列表里。k-d 树用分支定界搜索。</li><li>代价按每个问题比较过的向量数算；内存按各个库的存法算每个向量的字节数，简化见下。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul><li>只有 5,000 个向量，不是 10 亿个：索引比较的比例和召回率都会随规模变。坐标取整成整数，方便 Python 精确核对。HNSW 的层数用抛硬币来抽，概率和 hnswlib 的公式一样。PQ 编码的是向量本身，不是 FAISS 那样编码它到簇中心的偏移；每个中心只有约 10 个训练向量，少于 FAISS 会警告的 39 个，所以这里的召回率可能偏低一点。</li><li>比较一个 PQ 编码和比较一个完整向量都算一次，虽然它只是查 m 次表再相加，读的字节也少得多。</li><li>每个向量的内存算了向量、id 和最底层的连线。hnswlib 每个向量还另外存 8 字节的标签和 4 字节的连线计数（768 维、M = 16 时是 3,212 字节，不是 3,200），上层的连线也会多占一点。</li><li>真实规模的时间假设内存每秒读 100 GB，没有算运算的时间。一台服务器假设有 512 GB 内存。</li></ul><div class="eyebrow">来源</div><ul><li>FAISS 1.15.1：${SRC(U_IVF,'IndexIVF.h')}（nprobe = 1）、${SRC(U_HN,'impl/HNSW.h')}（efConstruction 40，efSearch 16）、${SRC(U_IHN,'IndexHNSW.h')}（M = 32）、${SRC(U_PQ,'ProductQuantizer.h')}、${SRC(U_MT,'MetricType.h')}（8 字节的 id）、${SRC(U_FR,'README')}。</li><li>hnswlib：${SRC(U_HA,'ALGO_PARAMS.md')}（M、内存）、${SRC(U_HR,'README')}（M = 16，ef_construction = 200）、${SRC(U_HG,'hnswalg.h')}（第 0 层 2 × M 条连线，ef = 10，每个向量怎么存）。</li><li>FAISS wiki：${SRC(U_RF,'Pre- and post-processing')}（IndexRefineFlat）。</li></ul></div></div>`);
const PRIPOP=()=>tx(`<p>An embedding is the vector a model makes from a text; texts with similar meanings get vectors that point in similar directions. Search compares a question’s vector with the stored ones by cosine similarity, the dot product of vectors of length 1. Each dimension stored as float32 takes 4 bytes. The ${LL('embeddings')} lab trains such a model.</p>`,
                    `<p>embedding 是模型把一段文字变成的向量；意思相近的文字，向量的方向相近。搜索时用余弦相似度，把问题的向量和存着的向量比；对长度为 1 的向量，它就是点积。每一维用 float32 存，占 4 字节。${LL('embeddings')} 这个 Lab 里训练过这样一个模型。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'apps',outcome:esc(OUTCOME()),short:tx('Vector index','向量索引'),title:tx('How do you find the nearest of 1B vectors in milliseconds?','怎么在几毫秒内从 10 亿个向量里找到最近的？'),
  minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: embeddings and cosine','前置知识：embedding 和余弦'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
function stageHTML(){
  return `<div class="panel stage vix-stage" id="vix-stage">`+
   `<div class="stage-head vix-head"><div class="vix-st-title" id="vix-st-title"></div><div class="stage-actions"><span class="status" id="vix-status" tabindex="0"></span></div></div>`+
   `<div class="vix-transport" id="vix-transport"></div>`+
   `<section class="vix-pan" id="vix-pan-pic"><div class="vix-pt" id="vix-p-title"></div><svg class="vix-pic" id="vix-pic" viewBox="0 0 360 260" role="img" aria-label=""></svg><div class="legend-row vix-leg" id="vix-p-leg"></div><p class="small muted vix-pnote" id="vix-p-note" hidden></p><div id="vix-kdbars" class="vix-kdbars"></div></section>`+
   `<section class="vix-pan" id="vix-pan-chart"><div class="vix-pt" id="vix-c-title"></div><svg class="vix-chart" id="vix-chart" viewBox="0 0 440 210" role="img" aria-label=""></svg><div class="legend-row vix-leg" id="vix-c-leg"></div></section>`+
   `<div class="metrics vix-metrics" id="vix-metrics"></div>`+
   `<details class="math" id="vix-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="vix-mathbody"></div></details></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="vix-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="vix-lock" hidden></p>`+
   `<div class="lockable">`+A.ui.seg({id:'vix-ctl-ix',key:'ix',label:tx('Index','索引'),options:['flat','ivf','hnsw','pq'].map(k=>[k,L(IXN[k])]),value:S.ix,
      note:tx('Exact search compares every vector. IVF compares only the vectors of the nearest cells. HNSW walks a graph of links. IVF-PQ keeps compressed codes in the IVF lists.','精确搜索和每个向量都比。IVF 只比最近几个簇里的向量。HNSW 在连线组成的图上走。IVF-PQ 在 IVF 的列表里存压缩后的编码。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'vix-ctl-np',key:'np',label:tx('Cells to probe, nprobe (IVF, IVF-PQ)','要查的簇数 nprobe（IVF、IVF-PQ）'),options:NPROBES.map(v=>[String(v),String(v)]),value:String(S.np),
      note:tx(`How many of the ${NLIST} cells a search opens, nearest centres first. FAISS’s default is 1. More cells, higher recall, more comparisons.`,`搜索打开 ${NLIST} 个簇里的几个，从中心最近的开始。FAISS 默认是 1。查的簇越多，召回率越高，比较也越多。`)})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'vix-ctl-M',key:'M',label:tx('Links per layer, M (HNSW)','每层连线数 M（HNSW）'),options:MS.map(v=>[String(v),String(v)]),value:String(S.M),
      note:tx(`Each vector links to up to M neighbours per layer, and up to ${MI('2 \\times M')} on the bottom layer. hnswlib’s default is 16, FAISS’s 32; hnswlib calls 12–48 fine for most uses. Each link is 4 bytes.`,`每个向量每层最多连 M 个邻居，最底层最多 ${MI('2 \\times M')} 个。hnswlib 默认 16，FAISS 默认 32；hnswlib 说 12–48 适合大多数用途。每条连线 4 字节。`)})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'vix-ctl-ef',key:'ef',label:tx('Candidates kept, ef (HNSW)','保留的候选数 ef（HNSW）'),options:EFS.map(v=>[String(v),String(v)]),value:String(S.ef),
      note:tx('On the bottom layer the search keeps the points closest to the question so far and goes on searching from them; ef is how many it keeps. hnswlib’s default is 10, FAISS’s efSearch 16. Higher ef, higher recall, more comparisons.','搜索到了最底层，会一直留着离问题最近的几个点，从它们出发继续找；ef 就是留几个。hnswlib 默认 10，FAISS 的 efSearch 默认 16。ef 越大，召回率越高，比较也越多。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'vix-ctl-m',key:'m',label:tx('Bytes per code, m (IVF-PQ)','每个编码的字节数 m（IVF-PQ）'),options:PQMS.map(v=>[String(v),String(v)]),value:String(S.m),
      note:tx('The 32 dimensions are cut into m pieces of 32 ÷ m; each piece becomes one byte, the number of its nearest of 256 centres.','32 维切成 m 段，每段 32 ÷ m 维；每段变成 1 个字节，记的是 256 个中心里离它最近的那个的编号。')})+`</div>`+
   `<details class="adv"><summary>${tx('Fixed in this lab: the vectors, the cells, efConstruction, the seed','这个 Lab 里固定的：向量、簇、efConstruction、随机种子')}</summary>`+
   `<p class="ctl-note">${tx(`5,000 vectors and 100 questions of 32 dimensions, made for this lab with seed ${SEED}. IVF has ${NLIST} cells (k-means, 8 rounds, on 2,000 vectors; FAISS warns when a cell gets fewer than 39 training vectors). HNSW is built with efConstruction 40, FAISS’s default (hnswlib’s is 200), to keep the build fast in a browser.`,`5,000 个向量和 100 个问题，都是 32 维，用随机种子 ${SEED} 为这个 Lab 生成。IVF 有 ${NLIST} 个簇（k-means 8 轮，用 2,000 个向量训练；FAISS 在每个簇的训练向量少于 39 个时会警告）。HNSW 用 efConstruction 40 建图，这是 FAISS 的默认值（hnswlib 是 200），为了在浏览器里建得快。`)}</p></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import numpy as np

def kmeans(x, k, rounds=8, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(rounds):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return c

class IVF:
    def __init__(self, x, nlist):
        self.c = kmeans(x[:40 * nlist], nlist)
        self.x = x
        cell = ((x[:, None] - self.c[None]) ** 2).sum(-1).argmin(1)
        self.lists = [np.where(cell == j)[0] for j in range(nlist)]

    def search(self, q, k=10, nprobe=1):
        dc = ((self.c - q) ** 2).sum(1)
        cells = np.argsort(dc)[:nprobe]       # nearest centres
        ids = np.concatenate([self.lists[j] for j in cells])
        d = ((self.x[ids] - q) ** 2).sum(1)   # compare these only
        return ids[np.argsort(d)[:k]]

def recall_at_k(found, true, k=10):
    return len(set(found[:k]) & set(true[:k])) / k`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="vix-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="vix-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('An IVF index in NumPy: k-means cells, inverted lists, a search that probes nprobe cells, and recall@10.','用 NumPy 写的 IVF 索引：k-means 分簇、倒排列表、只查 nprobe 个簇的搜索，以及 recall@10。')}</p><pre class="code" id="vix-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="vix-prod">`+
   tx(`<li>FAISS ${SRC(U_IVF,'<code>faiss/IndexIVF.h</code>')}: <code>nlist</code> cells and <code>size_t nprobe = 1;</code> “number of probes at query time”. ${SRC(U_CL,'<code>Clustering.h</code>')}: k-means with <code>min_points_per_centroid = 39</code>.</li>`+
      `<li>FAISS ${SRC(U_HN,'<code>faiss/impl/HNSW.h</code>')} sets <code>efConstruction = 40</code> and <code>efSearch = 16</code>; ${SRC(U_IHN,'<code>IndexHNSW.h</code>')} defaults to <code>M = 32</code>.</li>`+
      `<li>FAISS ${SRC(U_PQ,'<code>ProductQuantizer.h</code>')}: <code>M</code> subquantizers (the number of pieces, m in this lab) of <code>nbits</code> each (8 bits: 256 centres, one byte per piece); <code>IndexIVFPQ</code> keeps these codes in IVF lists.</li>`+
      `<li>hnswlib ${SRC(U_HG,'<code>hnswlib/hnswalg.h</code>')} keeps <code>maxM0_ = M_ * 2</code> links on layer 0 with 4-byte ids and sets <code>ef_ = 10</code>; ${SRC(U_HA,'<code>ALGO_PARAMS.md</code>')} puts the memory at “roughly M * 8-10 bytes per stored element”.</li>`,
      `<li>FAISS ${SRC(U_IVF,'<code>faiss/IndexIVF.h</code>')}：<code>nlist</code> 个簇，<code>size_t nprobe = 1;</code>，注释写“number of probes at query time”。${SRC(U_CL,'<code>Clustering.h</code>')}：k-means，<code>min_points_per_centroid = 39</code>。</li>`+
      `<li>FAISS 的 ${SRC(U_HN,'<code>faiss/impl/HNSW.h</code>')} 里 <code>efConstruction = 40</code>，<code>efSearch = 16</code>；${SRC(U_IHN,'<code>IndexHNSW.h</code>')} 里默认 <code>M = 32</code>。</li>`+
      `<li>FAISS ${SRC(U_PQ,'<code>ProductQuantizer.h</code>')}：<code>M</code> 个子量化器（也就是这个 Lab 里的段数 m），每个 <code>nbits</code> 位（8 位就是 256 个中心，每段 1 个字节）；<code>IndexIVFPQ</code> 把这些编码放在 IVF 的列表里。</li>`+
      `<li>hnswlib 的 ${SRC(U_HG,'<code>hnswlib/hnswalg.h</code>')} 里，第 0 层有 <code>maxM0_ = M_ * 2</code> 条连线，id 4 字节，<code>ef_ = 10</code>；${SRC(U_HA,'<code>ALGO_PARAMS.md</code>')} 说内存约是“roughly M * 8-10 bytes per stored element”。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: FAISS f46050e (1.15.1), hnswlib ca42672.','固定的 commit：FAISS f46050e（1.15.1），hnswlib ca42672。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/1603.09320" target="_blank" rel="noopener">Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a><span>${tx('Malkov and Yashunin, 2016 (HNSW)','Malkov 和 Yashunin，2016（HNSW）')}</span></li>`+
   `<li><a href="https://hal.inria.fr/inria-00514462v2/document" target="_blank" rel="noopener">Product quantization for nearest neighbor search</a><span>${tx('Jégou et al., PAMI 2011 (PQ)','Jégou 等，PAMI 2011（PQ）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/1702.08734" target="_blank" rel="noopener">Billion-scale similarity search with GPUs</a><span>${tx('Johnson, Douze and Jégou, 2017 (FAISS on GPUs)','Johnson、Douze 和 Jégou，2017（GPU 上的 FAISS）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2401.08281" target="_blank" rel="noopener">The Faiss library</a><span>${tx('Douze et al., 2024','Douze 等，2024')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="vix-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="vix-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="vix-app m2" id="vix-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="vix-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels and which picture the stage shows now */
function ctx(){
  if(mode==='guided'){const s=STEPS[step]; return {panels:s.panels,pic:s.pic,curves:s.curves||[]};}
  if(mode==='challenge') return {panels:['pic','chart','metrics'],pic:picOf(S.ix),curves:chSubmitted?['ivf','hnsw','pq']:[],hideNums:!chSubmitted};
  return {panels:['pic','chart','metrics'],pic:picOf(S.ix),curves:['ivf','hnsw','pq']};}
const picOf=ix=>ix==='ivf'?'ivf':ix==='hnsw'?'hnsw':ix==='pq'?'pq':'flat';

/* ---------- the 2-D picture ---------- */
const PW=360, PH=260;
function picXY(){const p=pic(); if(p.tf) return p.tf; let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(let i=0;i<P2N;i++){const x=p.P.X[i*2], y=p.P.X[i*2+1]; x0=Math.min(x0,x); x1=Math.max(x1,x); y0=Math.min(y0,y); y1=Math.max(y1,y);}
  const pad=10, sc=Math.min((PW-2*pad)/(x1-x0),(PH-2*pad)/(y1-y0)), ox=(PW-(x1-x0)*sc)/2, oy=(PH-(y1-y0)*sc)/2;
  p.tf={X:x=>ox+(x-x0)*sc,Y:y=>PH-(oy+(y-y0)*sc),x0,x1,y0,y1}; return p.tf;}
function voronoi(c,Cn,k){const tf=picXY(); let poly=[[tf.x0-1e4,tf.y0-1e4],[tf.x1+1e4,tf.y0-1e4],[tf.x1+1e4,tf.y1+1e4],[tf.x0-1e4,tf.y1+1e4]];
  const cx=Cn[c*2], cy=Cn[c*2+1];
  for(let o=0;o<k&&poly.length;o++){if(o===c) continue; const ox=Cn[o*2], oy=Cn[o*2+1], ax=ox-cx, ay=oy-cy, b=(ox*ox+oy*oy-cx*cx-cy*cy)/2, out=[];
    for(let i=0;i<poly.length;i++){const P=poly[i], Q=poly[(i+1)%poly.length], fp=ax*P[0]+ay*P[1]-b, fq=ax*Q[0]+ay*Q[1]-b;
      if(fp<=0) out.push(P); if((fp<0&&fq>0)||(fp>0&&fq<0)){const t=fp/(fp-fq); out.push([P[0]+t*(Q[0]-P[0]),P[1]+t*(Q[1]-P[1])]);}}
    poly=out;}
  return poly;}
/* events of one search in the picture: what the frame counter walks through */
function picEvents(kind){const p=pic(), q=p.q;
  if(kind==='flat'){const ord=range(P2N).sort((a,b)=>p.P.X[a*2]-p.P.X[b*2]||a-b); return {n:ord.length,ord,found:p.gt};}
  if(kind==='ivf'){const np=Math.min(picNP(),P2LIST), r=searchIVF(p.P,p.I,p.Qs,q,np,K); return {n:1+np,probed:r.probed,found:r.ids};}
  if(kind==='hnsw'){const r=searchHNSW(p.P,p.G,p.Qs,q,S.ef,K,true); return {n:r.ev.length,ev:r.ev,path:r.path,found:r.ids};}
  return {n:0};}
const picNP=()=>S.np;
const picKey=kind=>kind+'|'+(kind==='ivf'?Math.min(S.np,P2LIST):kind==='hnsw'?S.ef:'');
function picLegend(kind){
  const it=(cls,txt,tip)=>`<span class="lg"${tip?` data-tipx="${esc(tip)}"`:''}><i class="sw ${cls}"></i>${txt}</span>`;
  let h=it('vix-sw-q',tx('question','问题'),tx('The cross is the question’s vector.','叉号是问题的向量。'))+
   it('vix-sw-pt',tx('stored point','存着的点'),tx('A faded dot is a stored vector the search did not compare with the question.','半透明的点是搜索没有比较过的向量。'))+
   it('vix-sw-chk',tx('compared','比较过'),tx('A solid dot is a stored vector the search compared with the question: it computed their distance.','实心的点是搜索比较过的向量：算过它到问题的距离。'));
  if(kind!=='kd'&&kind!=='cells') h+=it('vix-sw-ok',tx('true 10 nearest: found','真正最近的 10 个：找到了'),tx('A green ring: one of the true 10 nearest, and the search returned it.','绿圈：真正最近的 10 个之一，搜索找到了它。'))+it('vix-sw-bad',tx('missed','漏掉了'),tx('A red ring: one of the true 10 nearest that the search did not return.','红圈：真正最近的 10 个之一，搜索没有找到。'));
  if(kind==='ivf'||kind==='cells') h+=it('vix-sw-cell',tx('probed cell','打开的簇'),tx('Lines split the plane into the 16 cells (each the region nearest one centre, the small square). Shaded cells are the ones the search opened.','线把平面分成 16 个簇（每个簇是离一个中心最近的区域，中心是小方块）。带阴影的是搜索打开的簇。'));
  if(kind==='hnsw') h+=it('vix-sw-hop',tx('hop on an upper layer','上层的一步'),tx('Dashed arrows: the greedy walk on the upper layers, from the entry point toward the question.','虚线箭头：在上层的贪心行走，从入口点走向问题。'))+it('vix-sw-link',tx('bottom-layer link','最底层的连线'),tx('Thin lines: the links of the bottom layer. Here M = 4, so each point keeps up to 2 × M = 8.','细线：最底层的连线。这里 M = 4，所以每个点最多 2 × M = 8 条。'));
  if(kind==='pq') h+=it('vix-sw-grid',tx('grid of centres','中心的网格'),tx('Each point moved to its nearest x centre and nearest y centre; the search ranks the moved points.','每个点挪到离它最近的 x 中心和 y 中心；搜索排的是挪过以后的点。'));
  if(kind==='kd') h+=it('vix-sw-cut',tx('cut','切线'),tx('Each line is one cut of the k-d tree (the first 5 levels are drawn).','每条线是 k-d 树的一刀（画出了前 5 层）。'));
  return h;}
function drawPic(){
  const c=ctx(), kind=c.pic, p=pic(), tf=picXY(), svg=$('#vix-pic',app), q=p.q, qx=tf.X(p.Qs.X[q*2]), qy=tf.Y(p.Qs.X[q*2+1]);
  const P=i=>[tf.X(p.P.X[i*2]),tf.Y(p.P.X[i*2+1])];
  const ev=kind==='flat'||kind==='ivf'||kind==='hnsw'?picEventsC(kind):{n:0};
  const fr=ev.n?Math.max(0,Math.min(ev.n,picF<0?ev.n:picF)):0, done=fr>=ev.n;
  let h='', chk=new Set(), found=null;
  if(kind==='cells'||kind==='ivf'){const pr=kind==='ivf'?ev.probed.slice(0,Math.max(0,fr-1)):[];
    for(let cI=0;cI<P2LIST;cI++){const poly=voronoi(cI,p.I.Cn,P2LIST); if(!poly.length) continue;
      h+=`<polygon class="vix-cell${pr.includes(cI)?' on':''}" points="${poly.map(v=>tf.X(v[0]).toFixed(1)+','+tf.Y(v[1]).toFixed(1)).join(' ')}"></polygon>`;}
    for(let cI=0;cI<P2LIST;cI++){const x=tf.X(p.I.Cn[cI*2]), y=tf.Y(p.I.Cn[cI*2+1]); h+=`<rect class="vix-ctr" x="${(x-3).toFixed(1)}" y="${(y-3).toFixed(1)}" width="6" height="6"></rect>`;}
    if(kind==='ivf'){for(const cI of pr) for(const i of p.I.lists[cI]) chk.add(i); if(done) found=ev.found;}}
  if(kind==='flat'){for(let t=0;t<fr;t++) chk.add(ev.ord[t]); if(done) found=ev.found;}
  if(kind==='hnsw'){const G=p.G, seen=new Set();
    for(let i=0;i<P2N;i++) for(const e of G.links[0][i]){const a=Math.min(i,e), b=Math.max(i,e), k=a*P2N+b; if(seen.has(k)) continue; seen.add(k); const A1=P(a), B1=P(b); h+=`<line class="vix-link" x1="${A1[0].toFixed(1)}" y1="${A1[1].toFixed(1)}" x2="${B1[0].toFixed(1)}" y2="${B1[1].toFixed(1)}"></line>`;}
    for(let t=0;t<fr;t++) chk.add(ev.ev[t]);
    const hops=ev.path.filter(x=>x[2]<=fr); for(let t=1;t<hops.length;t++){if(hops[t][1]===hops[t-1][1]) continue; const A1=P(hops[t-1][1]), B1=P(hops[t][1]); h+=`<line class="vix-hop" x1="${A1[0].toFixed(1)}" y1="${A1[1].toFixed(1)}" x2="${B1[0].toFixed(1)}" y2="${B1[1].toFixed(1)}" marker-end="url(#vix-arr)"></line>`;}
    const e0=P(G.entry); h+=`<circle class="vix-entry" cx="${e0[0].toFixed(1)}" cy="${e0[1].toFixed(1)}" r="6"></circle>`;
    if(done) found=ev.found;}
  if(kind==='pq'){const g=picPQ(S.m);
    for(const x of g.cx) h+=`<line class="vix-grid" x1="${tf.X(x).toFixed(1)}" y1="0" x2="${tf.X(x).toFixed(1)}" y2="${PH}"></line>`;
    for(const y of g.cy) h+=`<line class="vix-grid" x1="0" y1="${tf.Y(y).toFixed(1)}" x2="${PW}" y2="${tf.Y(y).toFixed(1)}"></line>`;
    found=g.ids; for(let i=0;i<P2N;i++) chk.add(i);}
  if(kind==='kd'){const r=searchKD(p.P,p.T,p.Qs,q); r.seen.forEach(i=>chk.add(i));
    (function cut(nd,dep,x0,x1,y0,y1){if(!nd||dep>=5) return; if(nd.ax===0){const x=tf.X(nd.v); h+=`<line class="vix-cut" x1="${x.toFixed(1)}" y1="${tf.Y(y0).toFixed(1)}" x2="${x.toFixed(1)}" y2="${tf.Y(y1).toFixed(1)}"></line>`; cut(nd.l,dep+1,x0,nd.v,y0,y1); cut(nd.r,dep+1,nd.v,x1,y0,y1);}
      else {const y=tf.Y(nd.v); h+=`<line class="vix-cut" x1="${tf.X(x0).toFixed(1)}" y1="${y.toFixed(1)}" x2="${tf.X(x1).toFixed(1)}" y2="${y.toFixed(1)}"></line>`; cut(nd.l,dep+1,x0,x1,y0,nd.v); cut(nd.r,dep+1,x0,x1,nd.v,y1);}})(p.T,0,tf.x0,tf.x1,tf.y0,tf.y1);}
  const pqg=kind==='pq'?picPQ(S.m):null;
  for(let i=0;i<P2N;i++){const xy=pqg?[tf.X(pqg.snap[i][0]),tf.Y(pqg.snap[i][1])]:P(i); h+=`<circle class="vix-dot${chk.has(i)?' chk':''}" cx="${xy[0].toFixed(1)}" cy="${xy[1].toFixed(1)}" r="2.6"></circle>`;}
  if(kind!=='kd'&&kind!=='cells') for(const i of p.gt){const xy=pqg?[tf.X(pqg.snap[i][0]),tf.Y(pqg.snap[i][1])]:P(i), cls=found?(found.includes(i)?'ok':'bad'):'';
    h+=`<circle class="vix-ring ${cls}" cx="${xy[0].toFixed(1)}" cy="${xy[1].toFixed(1)}" r="5.5"></circle>`;}
  h+=`<path class="vix-q" d="M${(qx-6).toFixed(1)} ${(qy-6).toFixed(1)}L${(qx+6).toFixed(1)} ${(qy+6).toFixed(1)}M${(qx-6).toFixed(1)} ${(qy+6).toFixed(1)}L${(qx+6).toFixed(1)} ${(qy-6).toFixed(1)}"></path>`;
  svg.innerHTML=`<defs><marker id="vix-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8z" class="vix-arrh"></path></marker></defs>`+h;
  const nf=found?p.gt.filter(i=>found.includes(i)).length:null;
  svg.setAttribute('aria-label',picTitle(kind,fr,ev,nf));
  $('#vix-p-title',app).textContent=picTitle(kind,fr,ev,nf);
  const lg=$('#vix-p-leg',app), lk=A.lang()+kind; if(lg._k!==lk){lg._k=lk; lg.innerHTML=picLegend(kind);}
  const kb=$('#vix-kdbars',app); kb.hidden=kind!=='kd'; if(kind==='kd') drawKDBars(kb);
  const pn=$('#vix-p-note',app); pn.hidden=kind!=='pq'; if(kind==='pq') pn.textContent=tx('A real PQ code has m pieces of 256 centres each. The picture has only 2 pieces, x and y, so a longer code is drawn as more centres per piece: the longer the code, the less each point moves.','真实的 PQ 编码有 m 段，每段 256 个中心。图里只有 x、y 两段，所以用每段多放一些中心来表示更长的编码：编码越长，点挪得越少。');
  return {fr,n:ev.n,nf};}
const PEV={};
function picEventsC(kind){const k=picKey(kind); return PEV[k]||(PEV[k]=picEvents(kind));}
function picTitle(kind,fr,ev,nf){
  if(kind==='flat') return fr<ev.n?tx(`Exact search in 2 dimensions: compared ${fr} of 300 points`,`2 维里的精确搜索：300 个点比较了 ${fr} 个`):tx('Exact search in 2 dimensions: all 300 points compared, all 10 nearest found','2 维里的精确搜索：300 个点都比较过，最近的 10 个全部找到');
  if(kind==='cells') return tx('IVF in 2 dimensions: 300 points in 16 cells, and a question','2 维里的 IVF：300 个点分在 16 个簇里，还有一个问题');
  if(kind==='ivf'){const np=Math.min(S.np,P2LIST), all=S.np>=P2LIST?tx(' (the picture has only 16 cells, so all are open)','（图里只有 16 个簇，所以全部打开了）'):''; return fr<ev.n?tx(`IVF in 2 dimensions: comparing with the 16 centres, then opening ${np===1?'the nearest cell':`the nearest ${np} cells`} (${Math.max(0,fr-1)} open)`,`2 维里的 IVF：先和 16 个中心比，再打开最近的 ${np} 个簇（已打开 ${Math.max(0,fr-1)} 个）`):tx(`IVF in 2 dimensions, ${np} of 16 cells probed: found ${nf} of the 10 nearest${all}`,`2 维里的 IVF，16 个簇查了 ${np} 个：最近的 10 个找到 ${nf} 个${all}`);}
  if(kind==='hnsw') return fr<ev.n?tx(`HNSW in 2 dimensions (M = 4, ef = ${S.ef}): compared ${fr} of 300 points`,`2 维里的 HNSW（M = 4，ef = ${S.ef}）：300 个点比较了 ${fr} 个`):tx(`HNSW in 2 dimensions (M = 4, ef = ${S.ef}): ${ev.n} of 300 points compared, found ${nf} of the 10 nearest`,`2 维里的 HNSW（M = 4，ef = ${S.ef}）：300 个点比较了 ${ev.n} 个，最近的 10 个找到 ${nf} 个`);
  if(kind==='pq') return tx(`PQ in 2 dimensions, a sketch: x and y, ${S.m} centres each; ranking the moved points finds ${nf} of the 10 nearest`,`2 维里的 PQ 示意：x、y 两段，每段 ${S.m} 个中心；按挪过的点排序，最近的 10 个找到 ${nf} 个`);
  return tx('A k-d tree in 2 dimensions: only the solid dots were compared','2 维里的 k-d 树：只有实心的点比较过');}
function drawKDBars(el){const k=kdStats();
  el.innerHTML=`<div class="vix-pt">${tx('Share of the 5,000 vectors a k-d tree compares to find the nearest one (average of 100 questions)','k-d 树为了找到最近的那个，要比较 5,000 个向量里的多少（100 个问题的平均）')}</div>`+KDD.map(d=>`<div class="vix-kb${d===kdD?' on':''}"><span>${tx(`${d} dimensions`,`${d} 维`)}</span><span class="vix-kbt"><i style="width:${Math.max(0.6,k[d]*100).toFixed(1)}%"></i></span><b class="mono">${esc(kdPc(k[d],true))}</b></div>`).join('');}

/* ---------- the chart: recall@10 against vectors compared ---------- */
const CX0=48, CX1=420, CY0=12, CY1=172, LX0=Math.log10(50), LX1=Math.log10(5000);
const cxm=v=>CX0+(CX1-CX0)*(Math.log10(Math.max(50,Math.min(5000,v)))-LX0)/(LX1-LX0), cym=r=>CY1-(CY1-CY0)*r;
function curvePts(kind){if(kind==='ivf') return NPROBES.map(np=>({s:{ix:'ivf',np},st:stat('ivf',np,0)}));
  if(kind==='hnsw') return EFS.map(ef=>({s:{ix:'hnsw',M:S.M,ef},st:stat('hnsw',S.M,ef)}));
  return NPROBES.map(np=>({s:{ix:'pq',m:S.m,np},st:stat('pq',S.m,np)}));}
const CURN=k=>k==='ivf'?tx('IVF (nprobe 1 to 32)','IVF（nprobe 1 到 32）'):k==='hnsw'?tx(`HNSW, M = ${S.M} (ef 10 to 256)`,`HNSW，M = ${S.M}（ef 10 到 256）`):tx(`IVF-PQ, ${S.m} bytes (nprobe 1 to 32)`,`IVF-PQ，${S.m} 字节（nprobe 1 到 32）`);
function drawChart(){
  const c=ctx(), svg=$('#vix-chart',app); let h='';
  for(const r of [0,0.25,0.5,0.75,1]) h+=`<line class="vix-gl" x1="${CX0}" y1="${cym(r).toFixed(1)}" x2="${CX1}" y2="${cym(r).toFixed(1)}"></line><text class="vix-ax" x="${CX0-6}" y="${(cym(r)+4).toFixed(1)}" text-anchor="end">${f(r*100,0)}%</text>`;
  for(const v of [50,100,200,500,1000,2000,5000]) h+=`<line class="vix-gl" x1="${cxm(v).toFixed(1)}" y1="${CY0}" x2="${cxm(v).toFixed(1)}" y2="${CY1}"></line><text class="vix-ax" x="${cxm(v).toFixed(1)}" y="${CY1+14}" text-anchor="middle">${int(v)}</text>`;
  h+=`<text class="vix-ax" x="${(CX0+CX1)/2}" y="${CY1+30}" text-anchor="middle">${tx('vectors compared per question (of 5,000; log scale)','每个问题比较的向量数（共 5,000 个；对数刻度）')}</text>`;
  h+=`<text class="vix-ax" x="12" y="${(CY0+CY1)/2}" text-anchor="middle" transform="rotate(-90 12 ${(CY0+CY1)/2})">recall@10</text>`;
  if(mode==='challenge'&&chSubmitted){h+=`<line class="vix-goal" x1="${CX0}" y1="${cym(CH_REC).toFixed(1)}" x2="${CX1}" y2="${cym(CH_REC).toFixed(1)}"></line><line class="vix-goal" x1="${cxm(CH_COMP).toFixed(1)}" y1="${CY0}" x2="${cxm(CH_COMP).toFixed(1)}" y2="${CY1}"></line>`;}
  for(const k of c.curves){const pts=curvePts(k); h+=`<polyline class="vix-cv vix-k-${k}" points="${pts.map(p=>cxm(p.st.comps).toFixed(1)+','+cym(p.st.rec).toFixed(1)).join(' ')}"></polyline>`+pts.map(p=>`<circle class="vix-cd vix-k-${k}" cx="${cxm(p.st.comps).toFixed(1)}" cy="${cym(p.st.rec).toFixed(1)}" r="3"></circle>`).join('');}
  if(c.curves.length||mode!=='guided') h+=`<circle class="vix-cd vix-k-flat" cx="${cxm(N).toFixed(1)}" cy="${cym(1).toFixed(1)}" r="3.5"></circle>`;
  if(!c.hideNums&&(c.curves.includes(S.ix)||S.ix==='flat')){const st=statOf(S), x=cxm(st.comps), y=cym(st.rec);
    h+=`<circle class="vix-me" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7"></circle><text class="vix-mel" x="${(x>CX1-90?x-10:x+10).toFixed(1)}" y="${(y<CY0+16?y+18:y-9).toFixed(1)}" text-anchor="${x>CX1-90?'end':'start'}">${esc(tx('your setting','你的设置'))}</text>`;}
  svg.innerHTML=h; svg.setAttribute('aria-label',tx('Recall@10 against vectors compared per question, one line per index.','recall@10 与每个问题比较的向量数，每种索引一条线。'));
  $('#vix-c-title',app).textContent=mode==='challenge'&&!chSubmitted?tx('Which settings find the true 10 nearest with the fewest comparisons? (shown after you submit)','哪些设置用最少的比较找到真正最近的 10 个？（提交后显示）'):tx('Which settings find the true 10 nearest with the fewest comparisons? (5,000 vectors, 100 questions)','哪些设置用最少的比较找到真正最近的 10 个？（5,000 个向量，100 个问题）');
  const lg=$('#vix-c-leg',app), lk=A.lang()+c.curves.join()+S.M+S.m+mode+chSubmitted;
  if(lg._k!==lk){lg._k=lk; const it=(cls,txt,tip)=>`<span class="lg" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i>${txt}</span>`;
    lg.innerHTML=c.curves.map(k=>it('vix-sw-'+k,esc(CURN(k)),k==='ivf'?tx('Each dot is one nprobe: more cells probed move it up and to the right.','每个点是一个 nprobe：查的簇越多，点越往右上。'):k==='hnsw'?tx('Each dot is one ef at the M chosen in Settings.','每个点是一个 ef，M 用“自由调参”里选的。'):tx('Each dot is one nprobe with the code size chosen in Settings. A comparison with a code counts as one, like a full vector, though it costs far less.','每个点是一个 nprobe，编码大小用“自由调参”里选的。比较一个编码也算一次，和比较完整向量一样，虽然它便宜得多。'))).join('')+
     ((c.curves.length||mode!=='guided')?it('vix-sw-flat',tx('exact search','精确搜索'),tx('Exact search: all 5,000 compared, recall 100%.','精确搜索：比较全部 5,000 个，召回率 100%。')):'')+
     it('vix-sw-me',tx('your setting','你的设置'),tx('The ring marks the setting chosen now.','圆圈标出现在选的设置。'))+
     (mode==='challenge'&&chSubmitted?it('vix-sw-goal',tx('bars: 95% and 1,000','两条线：95% 和 1,000'),tx('Dashed lines: recall@10 of 95% and 1,000 vectors compared. A setting passes above the first and left of the second.','虚线：recall@10 95% 和比较 1,000 个向量。在第一条线以上、第二条线左边的设置才算过关。')):'');}}

/* ---------- metric cards ---------- */
const MET=[
 {k:'rec',label:()=>tx('Recall@10','recall@10'),v:st=>pc1(st.rec),sub:()=>tx('true 10 nearest found, 100 questions','真正最近的 10 个找回几成，100 个问题'),tip:()=>D_REC(),pts:true},
 {k:'cmp',label:()=>tx('Vectors compared','比较的向量数'),v:st=>int(st.comps),sub:()=>S.ix==='flat'?tx('per question: all 5,000','每个问题：全部 5,000 个'):S.ix==='ivf'?tx(`per question, the ${NLIST} centres included`,`每个问题，含 ${NLIST} 个中心`):S.ix==='pq'?tx(`per question, with the ${NLIST} centres and the table`,`每个问题，含 ${NLIST} 个中心和距离表`):tx('per question','每个问题'),tip:()=>tx('How many vectors (or centres, or codes) the search compares with the question, averaged over 100 questions. Exact search compares all 5,000. In IVF-PQ a code counts as one, like a full vector, though it costs far less, and the table of distances to the 256 centres of every piece counts as 256.','搜索时问题要和多少个向量（或中心、编码）比较，在 100 个问题上取平均。精确搜索比较全部 5,000 个。IVF-PQ 里比较一个编码也算一次，和完整向量一样，虽然它便宜得多；到每一段 256 个中心的距离表算 256 次。')},
 {k:'byt',label:()=>tx('Memory per vector','每个向量的内存'),v:()=>BYT(bytesPer(S,D)),sub:()=>tx('32 dimensions in this lab','这个 Lab 的 32 维'),tip:()=>tx('Bytes stored per vector: the float32 vector (4 bytes × 32), plus an 8-byte id in IVF lists, plus HNSW’s links (2 × M links of 4 bytes), or only the PQ code and id.','每个向量存的字节：float32 向量（4 字节 × 32），IVF 列表里再加 8 字节的 id，HNSW 再加连线（2 × M 条，每条 4 字节）；PQ 只存编码和 id。')},
 {k:'big',label:()=>tx('Memory for 1B vectors of 768 dims','存 10 亿个 768 维向量要的内存'),v:()=>GB(REAL_N*bytesPer(S,REAL_D)),sub:()=>tx('with the index and settings above','按现在的索引和设置算'),tip:()=>tx('The memory the same index, with the same settings, would need for 1B vectors of 768 dimensions (GB = 10⁹ bytes). For IVF-PQ the code keeps the same m bytes; its recall at 768 dimensions is not the one measured here.','同一种索引、同样的设置，存 10 亿个 768 维向量要多少内存（GB = 10⁹ 字节）。IVF-PQ 的编码还是 m 个字节；768 维时的召回率和这里测的不一样。')}];
let metKey=null;
function drawMetrics(){
  const host=$('#vix-metrics',app), c=ctx(), st=statOf(S), key=JSON.stringify(S), cfgCh=metKey!=null&&metKey!==key; metKey=key;
  if(!host.children.length||host._lang!==A.lang()){host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" data-tipx="" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=A.lang();}
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), hide=c.hideNums&&(m.k==='rec'||m.k==='cmp'), v=hide?'—':m.v(st), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=hide?tx('shown after you submit','提交后显示'):m.sub(st); el.dataset.tipx=m.tip();
    let dt=''; if(userAct&&cfgCh&&!hide&&shownPrev[m.k]!=null&&shownPrev[m.k]!==v){
      if(m.k==='rec') dt=A.fmt.delta(parseFloat(shownPrev[m.k])/100,parseFloat(v)/100,{pts:true,dp:1});
      else if(m.k==='cmp') dt=A.fmt.delta(+shownPrev[m.k].replace(/,/g,''),+v.replace(/,/g,''),{dp:0});}
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});}
function drawHead(){
  const st=$('#vix-st-title',app), el=$('#vix-status',app), c=ctx();
  st.textContent=c.pic==='kd'?tx('Why trees stop working: a k-d tree in 2, 8 and 32 dimensions','为什么树不管用了：2 维、8 维和 32 维里的 k-d 树'):
    tx(`Searching the lab’s vectors with ${setTxt(S)}`,`搜索这个 Lab 的向量：${setTxt(S)}`);
  if(mode==='guided'&&['mission','exact','trees','predict'].includes(STEPS[step].id)){st.textContent=STEPS[step].id==='trees'?st.textContent:STEPS[step].id==='predict'?tx('IVF: the vectors split into cells','IVF：向量分成很多簇'):tx('Exact search: compare the question with every vector','精确搜索：问题和每个向量都比');}
  if(mode==='challenge'){const done=chDone(), me=done?chAll().find(p=>sameSet(p.s,S)):null; el.className='status'+(done?(me.ok?' ok':' bad'):''); el.textContent=done?(me.ok?tx('Meets both bars','两条线都满足'):tx('Misses a bar','没过线')):tx('Not submitted','还没提交');
    el.dataset.tipx=tx('Whether the setting reaches 95% recall@10 with at most 1,000 vectors compared. It shows once you submit.','这个设置是否达到 recall@10 95%、并且最多比较 1,000 个向量。提交以后才显示。'); el.hidden=false; return;}
  const show=c.panels.includes('metrics'); el.hidden=!show; if(!show) return;
  const s=statOf(S); el.className='status'; el.textContent=tx(`Recall@10 ${pc1(s.rec)}`,`recall@10 ${pc1(s.rec)}`); el.dataset.tipx=tx('Recall@10 of the current setting on the 100 questions.','现在的设置在 100 个问题上的 recall@10。');}
function drawTransport(){
  const tEl=$('#vix-transport',app), c=ctx(), kind=c.pic, on=(kind==='flat'||kind==='ivf'||kind==='hnsw')&&c.panels.includes('pic'), want=A.lang()+'|'+(anim?1:0)+'|'+picSpeed+'|'+on;
  tEl.hidden=!on;
  if(tEl._k!==want){tEl._k=want;
    tEl.innerHTML=`<button type="button" class="btn sm" data-act="play" data-play data-tipx="${esc(tx('Play the search in the picture; press again to pause.','播放图里的这次搜索；再按一次暂停。'))}">${anim?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Play','▶ 播放')}</button>`+
      A.ui.seg({id:'vix-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(k=>[k,L(SPEED_N[k])]),value:picSpeed})+
      `<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this search again from the start.','从头再看一遍这次搜索。'))}">${tx('Replay','重放')}</button>`+
      `<button type="button" class="btn ghost sm" data-act="stepf" data-tipx="${esc(tx('Show the next step of this search.','显示这次搜索的下一步。'))}">${tx('Step ▸','单步 ▸')}</button>`+
      `<input type="range" class="vix-scrub" id="vix-scrub" min="0" max="1" step="1" value="1" aria-label="${esc(tx('Search progress; drag to any step','搜索进度；拖到任意一步'))}">`;}
  const ev=on?picEventsC(kind):null, fr=ev?(picF<0?ev.n:Math.min(ev.n,picF)):0, sc=$('#vix-scrub',app);
  if(sc&&ev){sc.max=String(Math.max(1,ev.n)); sc.value=String(fr); A.ui.initRanges(tEl);}}
/* the scrubber and Step: show any frame of the search without playing it (R3.9) */
function scrubTo(v){const kind=picKind(); if(!kind) return; stopAnim(); const ev=picEventsC(kind);
  if(v>=ev.n){picF=-1; seenPic.add(picKey(kind));} else picF=Math.max(0,v);
  drawPic(); drawTransport(); A.dock.refresh();}
function panelsShow(){const c=ctx(); ['pic','chart'].forEach(p=>{const el=$('#vix-pan-'+p,app); if(el) el.hidden=!c.panels.includes(p);}); const mt=$('#vix-metrics',app); if(mt) mt.hidden=!c.panels.includes('metrics');
  const ch=$('#vix-pan-chart',app), pc=$('#vix-pan-pic',app), chFirst=mode==='guided'&&STEPS[step].id==='remember';
  if(chFirst&&ch.nextElementSibling!==pc) pc.parentNode.insertBefore(ch,pc); else if(!chFirst&&pc.nextElementSibling!==ch) pc.parentNode.insertBefore(ch,pc.nextSibling);}
/* Show the math: short blocks, each a lead line, the named formula, the numbers of the current setting, the result */
/* 3.2 × 10^12 for a byte count (R6.18: big numbers in a formula as powers of ten) */
const SCI=b=>{const e=Math.floor(Math.log10(b)), m=b/Math.pow(10,e); return `${String(+m.toFixed(4))} \\times 10^{${e}}`;};
function mathHTML(){
  const st=statOf(S), by=bytesPer(S,D), big=bytesPer(S,REAL_D), blk=(lead,tex,num)=>`<div class="vix-mb"><p>${lead}</p>${M(tex)}${num?M(num):''}</div>`;
  let h=blk(tx('Recall@10: the share of the true 10 nearest that the search returned, averaged over the 100 questions. h<sub>q</sub> is how many of question q’s true 10 nearest it returned.','recall@10：真正最近的 10 个里，搜索找回的比例，在 100 个问题上取平均。h<sub>q</sub> 是问题 q 的真正最近的 10 个里，找回了几个。'),
     '\\text{recall@10} = \\frac{1}{100}\\sum_{q} \\frac{h_q}{10}',`= \\frac{${f(st.rec*1000,0)}}{1000} = ${f(st.rec*100,1)}\\%`);
  const cmpTex=S.ix==='flat'?['\\text{compared} = N','= 5000']:S.ix==='ivf'?['\\text{compared} = n_{\\text{cells}} + \\sum_{\\text{probed}} |\\text{list}|',`\\approx ${NLIST} + ${f(st.comps-NLIST,1)} = ${f(st.comps,1)}`]:S.ix==='hnsw'?['\\text{compared} = \\text{vectors compared on the walk}',`\\approx ${f(st.comps,1)}`]:['\\text{compared} = n_{\\text{cells}} + 256 + \\sum_{\\text{probed}} |\\text{list}|',`\\approx ${NLIST} + 256 + ${f(st.comps-NLIST-PQ_K,1)} = ${f(st.comps,1)}`];
  const cmpTexZh=S.ix==='flat'?['\\text{比较数} = N','= 5000']:S.ix==='ivf'?['\\text{比较数} = n_{\\text{簇}} + \\sum_{\\text{打开的簇}} |\\text{列表}|',cmpTex[1]]:S.ix==='hnsw'?['\\text{比较数} = \\text{走的过程中比较过的向量数}',cmpTex[1]]:['\\text{比较数} = n_{\\text{簇}} + 256 + \\sum_{\\text{打开的簇}} |\\text{列表}|',cmpTex[1]];
  h+=blk(tx('Vectors compared per question (average of 100); N is the number of vectors, 5,000. For IVF-PQ the table of distances to the 256 centres of every piece counts as 256, and each code counts as one comparison.','每个问题比较的向量数（100 个问题的平均）；N 是向量数，5,000。IVF-PQ 里，到每一段 256 个中心的距离表算作 256 次，每个编码算一次比较。'),tx(cmpTex[0],cmpTexZh[0]),cmpTex[1]);
  const byT=d=>S.ix==='flat'?['4d',`4 \\times ${d} = ${int(4*d)}`]:S.ix==='ivf'?['4d + 8',`4 \\times ${d} + 8 = ${int(4*d+8)}`]:S.ix==='hnsw'?['4d + 2 \\times M \\times 4',`4 \\times ${d} + 2 \\times ${S.M} \\times 4 = ${int(bytesPer(S,d))}`]:['m + 8',`${S.m} + 8 = ${S.m+8}`];
  const vars=S.ix==='hnsw'?tx(` Here d is the number of dimensions (${D}) and M the links per layer (${S.M}).`,`这里 d 是维数（${D}），M 是每层的连线数（${S.M}）。`):S.ix==='pq'?tx(' Here m is the bytes per code.','这里 m 是每个编码的字节数。'):tx(` Here d is the number of dimensions (${D}).`,`这里 d 是维数（${D}）。`);
  h+=blk(tx('Bytes per vector: float32 is 4 bytes per dimension; IVF lists add an 8-byte id; HNSW adds 2 × M links of 4 bytes; PQ keeps only m code bytes and the id.','每个向量的字节数：float32 每维 4 字节；IVF 列表加 8 字节的 id；HNSW 加 2 × M 条 4 字节的连线；PQ 只存 m 个字节的编码和 id。')+vars,tx(`\\text{bytes per vector} = ${byT(D)[0]}`,`\\text{每个向量的字节数} = ${byT(D)[0]}`),`= ${byT(D)[1]}`);
  h+=`<div class="vix-mb"><p>${tx('The same index with the same settings for N = 1B vectors of 768 dimensions (GB = 10⁹ bytes). First the bytes per vector at d = 768, then the total.','同一种索引、同样的设置，存 N = 10 亿个 768 维向量（GB = 10⁹ 字节）。先算 d = 768 时每个向量的字节数，再算总数。')}</p>`+
     M(tx(`\\text{bytes per vector} = ${byT(REAL_D)[1]}`,`\\text{每个向量的字节数} = ${byT(REAL_D)[1]}`))+
     M(tx('\\text{memory} = N \\times \\text{bytes per vector}','\\text{内存} = N \\times \\text{每个向量的字节数}'))+
     M(tx(`= 10^9 \\times ${int(big)}\\,\\text{bytes} = ${SCI(REAL_N*big)}\\,\\text{bytes} = ${int(REAL_N*big/1e9)}\\,\\text{GB}`,`= 10^9 \\times ${int(big)}\\,\\text{字节} = ${SCI(REAL_N*big)}\\,\\text{字节} = ${int(REAL_N*big/1e9)}\\,\\text{GB}`))+`</div>`;
  if(S.ix==='flat') h+=blk(tx('Exact search reads every vector; at the assumed 100 GB/s of one server (10¹¹ bytes a second), with d = 768:','精确搜索要读每个向量；按假设的一台服务器每秒 100 GB（每秒 10¹¹ 字节），d = 768：'),tx('\\text{time} = \\frac{N \\times 4d}{100\\,\\text{GB/s}}','\\text{时间} = \\frac{N \\times 4d}{100\\,\\text{GB/s}}'),tx(`= \\frac{10^9 \\times 3072\\,\\text{bytes}}{10^{11}\\,\\text{bytes/s}} \\approx ${f(REAL_N*REAL_D*4/BW,1)}\\,\\text{s}`,`= \\frac{10^9 \\times 3072\\,\\text{字节}}{10^{11}\\,\\text{字节/秒}} \\approx ${f(REAL_N*REAL_D*4/BW,1)}\\,\\text{秒}`));
  return h;}
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key;
    if(k==='speed') A.ui.setPressed(gEl,picSpeed); else if(k==='rn') A.ui.setPressed(gEl,realN===1e6?'1e6':realN===1e8?'1e8':'1e9'); else if(k==='kd') A.ui.setPressed(gEl,String(kdD)); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));});
  const lock=mode==='challenge', c=$('#vix-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#vix-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you choose the index and its settings in the task card. The vectors, the cells and the questions are fixed.','挑战中只能在任务卡里选索引和它的设置。向量、簇和问题都是固定的。');
  $$('#vix-controls .lockable',app).forEach(e=>{e.hidden=lock;});}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(){
  const rail=$('#vix-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again) A.ui.obsBump(box);}}}
    A.ui.setNext(rail,isNextDisabled());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#vix-insight',rail); if(ins){const html=`<div class="eyebrow">${c>=0?esc(L(PRESETS[c].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const done=chDone(), st=done?statOf(S):null, av=$('#vix-ch-av',rail); if(!av) return;
    av.textContent=done?pc1(st.rec):tx('after you submit','提交后显示');
    $('#vix-ch-cv',rail).textContent=done?int(st.comps):tx('after you submit','提交后显示');
    $('#vix-ch-gv',rail).textContent=BYT(bytesPer(S,D));
    $('#vix-ch-a',rail).classList.toggle('met',!!(done&&st.rec>=CH_REC)); $('#vix-ch-c',rail).classList.toggle('met',!!(done&&st.comps<=CH_COMP));
    const sub=$('#vix-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    const res=$('#vix-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#vix-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  panelsShow(); drawTransport(); drawHead();
  if(!$('#vix-pan-pic',app).hidden) drawPic();
  if(!$('#vix-pan-chart',app).hidden) drawChart();
  if(!$('#vix-metrics',app).hidden) drawMetrics();
  const mb=$('#vix-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();}
function announce(){clearTimeout(annT); if(mode==='guided'&&['predict','trees'].includes(STEPS[step].id)) return; annT=setTimeout(()=>{if(!app) return; const s=statOf(S); A.announce(tx(`${setTxt(S)}: recall@10 ${pc1(s.rec)}.`,`${setTxt(S)}：recall@10 ${pc1(s.rec)}。`));},400);}
/* ---------- the picture's search (R3.9, R3.20): it plays only when the picture would differ; one already watched shows its end ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} const was=anim; anim=null; lastTs=0; return was;}
function picKind(){const c=ctx(); return c.panels.includes('pic')&&(c.pic==='flat'||c.pic==='ivf'||c.pic==='hnsw')?c.pic:null;}
function picStart(fromZero){stopAnim(); const kind=picKind(); if(!kind){picF=-1; render(); return;}
  const ev=picEventsC(kind); if(fromZero||picF<0||picF>=ev.n) picF=0;
  if(A.reduceMotion()){picF=-1; seenPic.add(picKey(kind)); render(); return;}
  anim={kind,acc:0}; lastTs=0; render(); raf=requestAnimationFrame(tick);}
function tick(ts){
  raf=0; if(!anim||!app) return; const ev=picEventsC(anim.kind);
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts;
  anim.acc+=dt*ev.n/PIC_DUR[picSpeed]; const n=Math.floor(anim.acc);
  if(n>0){anim.acc-=n; picF=Math.min(ev.n,picF+n); if(picF>=ev.n){const k=picKey(anim.kind); stopAnim(); picF=-1; seenPic.add(k); render(); return;} drawPic(); drawTransport(); A.dock.refresh();}
  raf=requestAnimationFrame(tick);}
/* a new picture: play it if it has never been watched, else show its end at once */
function showPic(){const kind=picKind(); if(!kind){stopAnim(); picF=-1; render(); return;} if(seenPic.has(picKey(kind))){stopAnim(); picF=-1; render();} else picStart(true);}

/* ---------- navigation ---------- */
function enterStep(i){
  const was=anim?picKey(anim.kind):null, f0=picF;
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  if(s.id==='exact'){triedN.clear(); triedN.add(realN);}
  if(s.id==='trees'){triedD.clear(); triedD.add(kdD);}
  if(s.id==='ivf'){triedNP.clear(); triedNP.add(S.np);}
  if(s.id==='hnsw'){triedEF.clear(); triedEF.add(S.ef);}
  if(s.id==='pq'){triedPQ.clear(); triedPQ.add(S.m);}
  picF=-1; renderRail();
  const kind=picKind(); if(was&&kind&&picKey(kind)===was){picF=f0; picStart(false); return;}   // the same search still playing: let it go on
  showPic();}
function revealStep(){A.ui.reveal($('#vix-rail .step.current',app)||$('#vix-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  const dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#vix-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#vix-insight',app)); else revealBox($('#vix-ch-res',app));});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH_INIT};}
  if(m==='sandbox'&&was==='guided') S={...DEFAULT};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  picF=-1; renderRail(); showPic();}
function applyPreset(i){const p=PRESETS[i]; stopAnim(); Object.assign(S,p.p); byLearner(()=>{picF=-1; render(); showPic();}); revealObs();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=(s=S)=>[s.ix,s.np,s.M,s.ef,s.m].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null; const o={ix:p[0],np:+p[1],M:+p[2],ef:+p[3],m:+p[4]};
  if(!IXN[o.ix]||!NPROBES.includes(o.np)||!MS.includes(o.M)||!EFS.includes(o.ef)||!PQMS.includes(o.m)) return null; return o;}
function act(a,btn){
  if(a==='play'){if(anim){stopAnim(); render(); return;} picStart(false); return;}
  if(a==='replay'){picStart(true); return;}
  if(a==='stepf'){scrubTo(picF<0?0:picF+1); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{Object.assign(S,DEFAULT); picF=-1; render(); showPic();}); return;}
  if(a==='submit'){chSubmitted=true; render(); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#vix-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function setKey(k,v,inRail){
  if(k==='rn'){realN=+v; triedN.add(realN); byLearner(render); return;}
  if(k==='kd'){kdD=+v; triedD.add(kdD); byLearner(render); return;}
  if(k==='ix'){S.ix=v;} else S[k]=+v;
  if(k==='np') triedNP.add(S.np); if(k==='ef') triedEF.add(S.ef); if(k==='m') triedPQ.add(S.m);
  if(mode==='challenge'){if(!chReveal) chSubmitted=false; const c=$('#vix-ch-ctl',app); if(c&&k==='ix'){c.innerHTML=chCtlHTML();}}
  byLearner(()=>{picF=-1; render(); showPic();});}
function showMe(p){
  if(p.rn!=null){setKey('rn',p.rn); revealObs(); return;}
  if(p.kd!=null){setKey('kd',p.kd); revealObs(); return;}
  for(const k of ['np','ef','m']) if(p[k]!=null){setKey(k,p[k]); revealObs(); return;}}

/* ---------- interview drill (seeded): size an IVF index ---------- */
function makeDrill(seed){const rnd=mulberry(seed*7919+13), pick=a=>a[Math.floor(rnd()*a.length)];
  const n=pick([1e7,1e8,1e9]), d=pick([384,768,1024]), nl=pick([4096,16384,65536]), np=pick([8,16,32,64]);
  const per=n/nl, comps=nl+np*per, mem=n*(4*d+8);
  return {n,d,nl,np,per,comps,mem,share:comps/n};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#vix-drill-q',app), a=$('#vix-drill-a',app);
  q.innerHTML=tx(`An IVF index holds <b>${NB(d.n)}</b> float32 vectors of <b>${d.d}</b> dimensions in <b>${int(d.nl)}</b> cells of about equal size, each vector with an 8-byte id. A search probes <b>${d.np}</b> cells. About how many vectors does one question get compared with, and how much memory do the lists take?`,
                 `一个 IVF 索引把 <b>${NB(d.n)}</b>${d.n>=1e9?'':' '}个 <b>${d.d}</b> 维的 float32 向量分在 <b>${int(d.nl)}</b> 个大小差不多的簇里，每个向量带 8 字节的 id。搜索查 <b>${d.np}</b> 个簇。一个问题大约要和多少个向量比较？这些列表占多少内存？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>about ${A.fmt.big(d.comps)} compared (${f(d.share*100,d.share<0.001?3:2)}% of the vectors); ${GB(d.mem)}</b></p>`,`<p><b>约比较 ${A.fmt.big(d.comps)} 个（占全部向量的 ${f(d.share*100,d.share<0.001?3:2)}%）；${GB(d.mem)}</b></p>`)+
    M(tx(`\\text{compared} = ${int(d.nl)} + ${d.np} \\times 10^{${Math.round(Math.log10(d.n))}} \\div ${int(d.nl)} \\approx ${int(d.comps)}`,`\\text{比较数} = ${int(d.nl)} + ${d.np} \\times 10^{${Math.round(Math.log10(d.n))}} \\div ${int(d.nl)} \\approx ${int(d.comps)}`))+
    M(tx(`\\text{memory} = 10^{${Math.round(Math.log10(d.n))}} \\times (4 \\times ${d.d} + 8)\\,\\text{bytes} = ${GB(d.mem).replace(' GB','\\,\\text{GB}')}`,`\\text{内存} = 10^{${Math.round(Math.log10(d.n))}} \\times (4 \\times ${d.d} + 8)\\,\\text{字节} = ${GB(d.mem).replace(' GB','\\,\\text{GB}')}`))+
    tx('<p>N is the number of vectors. Each cell holds about N ÷ (number of cells) vectors, and a search compares the question with every centre, then reads the cells it probes. The memory does not shrink: IVF stores every full vector, so only PQ codes make the lists smaller.</p>','<p>N 是向量总数。每个簇约有 N ÷ 簇数 个向量；搜索先和每个中心比，再读查的那几个簇。内存不会变小：IVF 存的是每个完整的向量，只有换成 PQ 编码，列表才会变小。</p>');}

/* ---------- jumps ---------- */
const CTLN={ix:{en:'Index',zh:'索引'},np:{en:'Cells to probe, nprobe',zh:'要查的簇数 nprobe'},M:{en:'Links per layer, M',zh:'每层连线数 M'},ef:{en:'Candidates kept, ef',zh:'保留的候选数 ef'},m:{en:'Bytes per code, m',zh:'每个编码的字节数 m'}};
function jumpAct(k){
  if(k==='pic') return tx('Click to show the picture.','点击查看图。');
  if(k==='chart') return tx('Click to show the chart.','点击查看曲线图。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k){
  if(!app) return;
  const pan=(id,mark)=>{const el=$('#vix-pan-'+id,app); if(el&&el.hidden){el.hidden=false; render();} A.tip.hide(); const m=mark&&$(mark,app);
    if(el){(m||el).scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); if(m) A.ui.flash(m);}};
  if(k==='pic'){pan('pic','#vix-pic'); return;}
  if(k==='chart'){pan('chart','#vix-chart'); return;}
  if(k==='controls'){const c=$('#vix-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(!CTLN[k]) return;
  const el=$('#vix-ic-'+k,app)||$('#vix-ctl-'+k,app); if(el) A.ui.jumpToEl(el);}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    const st=$('#vix-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#vix-rail .step.current',app)||$('#vix-rail .railcard',app)||$('#vix-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, rc=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=rc.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#vix-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#vix-rail');
      if(k==='speed'){picSpeed=d.v; render(); return;}
      stopAnim(); setKey(k,d.v,inRail); if(inRail&&mode!=='challenge') revealObs(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#vix-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#vix-rail',app)); return;}
    if(d.act) act(d.act,b);});
  el.addEventListener('input',e=>{if(e.target&&e.target.id==='vix-scrub') scrubTo(+e.target.value);});}
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the sizes and dimensions picked, what was tried, the
     searches watched (one still playing counts as watched), the challenge result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,realN,kdD,
    triedN:[...triedN],triedD:[...triedD],triedNP:[...triedNP],triedEF:[...triedEF],triedPQ:[...triedPQ],
    seenPic:[...new Set([...seenPic,...(anim?[picKey(anim.kind)]:[])])].slice(-100),picSpeed,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&(st=>st.rec>=CH_REC-1e-9&&st.comps<=CH_COMP)(statOf(S))}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(token(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, put=(set,v,ok)=>{set.clear(); if(Array.isArray(v)) v.filter(ok).forEach(x=>set.add(x));};
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    const RN=[1e6,1e8,1e9]; if(RN.includes(o.realN)) realN=o.realN; if(KDD.includes(o.kdD)) kdD=o.kdD;
    put(triedN,o.triedN,v=>RN.includes(v)); put(triedD,o.triedD,v=>KDD.includes(v)); put(triedNP,o.triedNP,v=>NPROBES.includes(v)); put(triedEF,o.triedEF,v=>EFS.includes(v)); put(triedPQ,o.triedPQ,v=>PQMS.includes(v));
    put(seenPic,o.seenPic,k=>typeof k==='string'&&/^(flat|ivf|hnsw)\|\d*$/.test(k));
    if(Object.keys(PIC_DUR).includes(o.picSpeed)) picSpeed=o.picSpeed; drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen); picF=-1;},
  mount(el,tok){
    stopAnim(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#vix-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#vix-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    renderRail(); showPic();},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const dm=$('#vix-math',app), open=dm&&dm.open, a=anim, f0=picF; stopAnim(); this.mount(root,null); if(open) $('#vix-math',app).open=true; if(a){picF=f0; picStart(false);}}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#vix-pan-pic:not([hidden])',app)||$('#vix-pan-chart',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#vix-stage',app),
    render({status,vis,read}){if(!app) return; const c=ctx(), st=statOf(S);
      if(mode==='challenge'){const done=chDone(), me=done?chAll().find(p=>sameSet(p.s,S)):null; status.className='status'+(done?(me.ok?' ok':' bad'):''); status.textContent=done?(me.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):tx('Not submitted','未提交');
        vis.innerHTML=''; read.textContent=done?tx(`${setTxt(S)}: recall@10 ${pc1(st.rec)}, ${int(st.comps)} compared`,`${setTxt(S)}：recall@10 ${pc1(st.rec)}，比较 ${int(st.comps)} 个`):tx(`${setTxt(S)}: ${int(bytesPer(S,D))} bytes per vector`,`${setTxt(S)}：每个向量 ${int(bytesPer(S,D))} 字节`); return;}
      if(mode==='guided'&&STEPS[step].id==='predict'){status.className='status'; status.textContent=tx('IVF cells','IVF 的簇'); vis.innerHTML=''; read.textContent=tx('1,000 cells, one probed: what does it miss?','1,000 个簇，只查 1 个：会漏掉什么？'); return;}
      if(c.pic==='kd'){status.className='status'; status.textContent=tx('k-d tree','k-d 树'); vis.innerHTML=''; read.textContent=tx(`In ${kdD} dimensions the tree compares ${kdPc(kdStats()[kdD])} of the vectors`,`${kdD} 维时，树要比较 ${kdPc(kdStats()[kdD])} 的向量`); return;}
      vis.innerHTML=`<div class="vix-dk"><i style="width:${(st.rec*100).toFixed(1)}%"></i></div>`;
      status.className='status'+(st.rec>=0.95?' ok':''); status.textContent=tx(`Recall@10 ${pc1(st.rec)}`,`recall@10 ${pc1(st.rec)}`);
      read.textContent=tx(`${setTxt(S)}: ${int(st.comps)} vectors compared per question`,`${setTxt(S)}：每个问题比较 ${int(st.comps)} 个向量`);}}   // recall@10 is on the status, left (O6 of review b10r1)
});
/* read-only hook for tests/vector-index_numbers.py */
window.__vix={state:()=>({mode,step,ix:S.ix,np:S.np,M:S.M,ef:S.ef,m:S.m,pred,realN,kdD,chSubmitted,chReveal,picF,anim:!!anim,lang:LG()}),
  seen:()=>[...seenPic], stat:(ix,a,b)=>{const s=stat(ix,a,b); return {rec:s.rec,comps:s.comps};}, gt:()=>base().GT, kd:()=>kdStats(),
  pic:()=>{const p=pic(); return {q:p.q,gt:p.gt,lists:p.I.lists.map(l=>l.slice()),Cn:Array.from(p.I.Cn),entry:p.G.entry,top:p.G.top,level:Array.from(p.G.level)};},
  picEv:(kind,np,ef)=>{const s0={...S}; S.np=np; S.ef=ef; const e=picEvents(kind); Object.assign(S,s0); return e;}, picPQ,
  ivfLists:()=>ivf().lists.map(l=>l.slice()), hnswLinks:M0=>hnsw(M0).links[0].map(a=>a.slice()), hnswLevels:M0=>Array.from(hnsw(M0).level), pqCodes:m=>Array.from(pq(m).codes.slice(0,200)),
  data:(i)=>Array.from(base().B.X.subarray(i*D,i*D+D)), drill:makeDrill, bytesPer, chAll:()=>chAll().map(p=>({s:p.s,rec:p.rec,comps:p.comps,bytes:p.bytes,ok:p.ok})),
  consts:{SEED,D,N,NQ,NCL,NOISE,SC,K,NLIST,NTRAIN,KM_IT,PQ_K,PQ_IT,PQ_TRAIN,EFC,NPROBES,EFS,MS,PQMS,KDD,P2N,P2Q,P2CL,P2NOISE,P2LIST,P2M,P2EFC,REAL_D,REAL_N,BW,CH_REC,CH_COMP,EF_FREE}};
})();
