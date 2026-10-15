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
/* the 2-D picture: 300 points (8 clusters, not scaled to length 1), 40 candidate questions. The question shown is the first one whose
   picture shows what each index can miss: opening its nearest cell finds 3 to 7 of its 10 nearest and opening 2 cells still misses
   one; HNSW with ef = 10 misses one; and the PQ grid finds more of them with 8 centres per piece than with 4, and more with 16 */
let PIC=null;
function pqSnap(P,c){const cx=kmeans(P.X,2,range(P2N),0,1,c,10,SEED+30), cy=kmeans(P.X,2,range(P2N),1,1,c,10,SEED+31), snap=[];
  for(let i=0;i<P2N;i++) snap.push([cx[nearestC(P.X,2,i,0,1,cx,c)],cy[nearestC(P.X,2,i,1,1,cy,c)]]);
  return {cx:Array.from(cx),cy:Array.from(cy),snap};}
function pqRank(sn,Qs,q){const qx=Qs.X[q*2], qy=Qs.X[q*2+1], o=sn.snap.map((s,i)=>[(s[0]-qx)*(s[0]-qx)+(s[1]-qy)*(s[1]-qy),i]); return topK(o,K).map(x=>x[1]);}
function pic(){if(PIC) return PIC; const X=genSet(SEED+20,2,P2N+P2Q,P2CL,P2NOISE,false);
  const P={X:X.subarray(0,P2N*2),d:2,n:P2N}, Qs={X:X.subarray(P2N*2),d:2,n:P2Q}, I=buildIVF(P,P2LIST,P2N,10,SEED+21), G=buildHNSW(P,P2M,P2EFC,SEED+22), T=buildKD(P);
  const SN={}; for(const c of PQMS) SN[c]=pqSnap(P,c);
  let q=0; for(let i=0;i<P2Q;i++){const g=exact(P,Qs,i,K), hit=ids=>ids.filter(x=>g.includes(x)).length;
    const v1=hit(searchIVF(P,I,Qs,i,1,K).ids), v2=hit(searchIVF(P,I,Qs,i,2,K).ids), h10=hit(searchHNSW(P,G,Qs,i,10,K,false).ids), pq=PQMS.map(c=>hit(pqRank(SN[c],Qs,i)));
    if(v1>=3&&v1<=7&&v2<K&&h10<K&&pq[0]<pq[1]&&pq[1]<pq[2]){q=i; break;}}
  PIC={P,Qs,q,I,G,T,gt:exact(P,Qs,q,K),sn:SN}; return PIC;}
/* PQ in the picture: 2 pieces (x and y), c centres each (1-D k-means); every point moves to the grid and the search ranks the moved points */
function picPQ(c){const p=pic(), sn=p.sn[c]||(p.sn[c]=pqSnap(p.P,c)); return {cx:sn.cx,cy:sn.cy,snap:sn.snap,ids:pqRank(sn,p.Qs,p.q)};}
/* the k-d search of searchKD as events, for the picture: each vector compared (with its node's region and cut) and each region skipped
   because its border is farther from the question than the nearest point found so far */
function kdTrace(B,T,Qs,qi){const d=B.d, ev=[]; let best=Infinity, bi=-1;
  (function go(nd,box){if(!nd) return; const dd=d2(Qs.X,qi,B.X,nd.id,d); if(dd<best||(dd===best&&nd.id<bi)){best=dd; bi=nd.id;}
    ev.push({k:'cmp',id:nd.id,ax:nd.ax,v:nd.v,box,best,bi});
    const diff=Qs.X[qi*d+nd.ax]-nd.v, near=diff<0?nd.l:nd.r, far=diff<0?nd.r:nd.l, lo=box.slice(), hi=box.slice();
    lo[nd.ax*2+1]=nd.v; hi[nd.ax*2]=nd.v;   // box = [x0, x1, y0, y1]: the low side keeps coordinates up to the cut, the high side from it
    go(near,diff<0?lo:hi); if(diff*diff<=best) go(far,diff<0?hi:lo); else if(far) ev.push({k:'skip',box:diff<0?hi:lo,best,bi});})(T,[-Infinity,Infinity,-Infinity,Infinity]);
  return {ev,id:bi,comps:ev.filter(x=>x.k==='cmp').length};}
/* the search of searchHNSW as events, for the picture: on each upper layer the neighbours looked at and the hops; on the bottom layer
   each round's expanded candidate, the neighbours it compared, the candidates added and the farthest ones they pushed out. It returns
   the same vectors as searchHNSW (tests/vector-index_numbers.py checks it) */
function hnswTrace(B,G,Qs,qi,ef){
  const ev=[], efx=Math.max(ef,K); let comps=0; const dq=e=>{comps++; return d2(Qs.X,qi,B.X,e,B.d);};
  let cur=G.entry, dc=dq(cur); ev.push({k:'entry',cur});
  for(let L=G.top;L>0;L--){let shown=false, ch=true;
    while(ch){ch=false; const from=cur, nb=G.links[L][from]; if(!nb||!nb.length) break;
      if(!shown){ev.push({k:'layer',L,cur}); shown=true;}
      const seen=[]; for(const e of nb){const de=dq(e); seen.push(e); if(de<dc||(de===dc&&e<cur)){dc=de; cur=e; ch=true;}}
      ev.push({k:'look',L,from,seen}); if(ch) ev.push({k:'hop',L,from,to:cur});}
    if(shown) ev.push({k:'down',L,cur});}
  const C=[], W=[], vis=new Set([cur]); hpush(C,[dc,cur],lt); hpush(W,[dc,cur],gt);
  ev.push({k:'bottom',cur,W:[cur],far:dc});
  while(C.length){const c=hpop(C,lt); if(W.length>=efx&&gt(c,W[0])) break;
    const comp=[], add=[], out=[];
    for(const e of G.links[0][c[1]]){if(vis.has(e)) continue; vis.add(e); const x=[dq(e),e]; comp.push(e);
      if(W.length<efx||lt(x,W[0])){hpush(C,x,lt); hpush(W,x,gt); add.push(e); if(W.length>efx) out.push(hpop(W,gt)[1]);}}
    ev.push({k:'round',c:c[1],comp,add,out,W:W.map(x=>x[1]),far:W[0][0]});}
  return {ev,ids:W.slice().sort(byDI).slice(0,K).map(x=>x[1]),comps};}
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
const D_NN=()=>tx('Nearest-neighbour search: given a question’s vector, return the k stored vectors closest to it; here k = 10. When every vector has length 1, a smaller distance means a higher cosine similarity, so ranking by either gives the same order.','最近邻搜索：给一个问题的向量，返回存着的向量里离它最近的 k 个，这里 k = 10。向量长度都是 1 时，距离越小，余弦相似度就越大，所以按距离排和按余弦相似度排是同一个顺序。');
const D_ANN=()=>tx('Approximate nearest-neighbour search (ANN) compares the question with only a small part of the vectors. It is much faster, and the price is that it sometimes misses one of the true nearest.','近似最近邻（ANN）只和一小部分向量比。它快得多，代价是偶尔漏掉真正最近的那几个。');
const D_REC=()=>tx('Recall@10 here: of the true 10 nearest vectors (found by exact search), the share the index returned, averaged over 100 questions.','这里的 recall@10：精确搜索找出的真正最近的 10 个向量里，索引找回了几成，在 100 个问题上取平均。');
const D_IVF=()=>tx('IVF (inverted file): k-means splits the vectors into cells around centres; each vector is stored in the list of its nearest centre. A search compares the question with every centre, then only with the vectors in the nprobe nearest cells.','IVF（倒排文件）：先用 k-means 把向量分成很多簇，每簇有一个中心；每个向量存进离它最近的中心的列表里。搜索时，问题先和所有中心比，再只和最近的 nprobe 个簇里的向量比。');
const D_KM=()=>tx('k-means: start with k centres, put every vector with its nearest centre, move each centre to the mean of its vectors, and repeat a few rounds.','k-means：先取 k 个中心，每个向量归到离它最近的中心，再把每个中心挪到它那些向量的平均位置，重复几轮。');
const D_HNSW=()=>tx('HNSW (hierarchical navigable small world): every vector is a node linked to a few nearby nodes. Upper layers hold fewer nodes with long links; a search goes from the top layer down, each step moving to a linked node closer to the question, then explores the closest ef candidates on the bottom layer.','HNSW（分层可导航小世界图）：每个向量是一个节点，连着附近的几个节点。上层节点少、连线长；搜索从最上层往下走，每一步走到连着的、离问题更近的点，到最底层时保留最近的 ef 个候选，继续扩展。');
const D_PQ=()=>tx('Product quantization (PQ): cut each vector into m pieces; for each piece k-means learns 256 centres in advance, and the piece is stored as the number of its nearest centre, one byte. A vector becomes m bytes; a distance is looked up in a small table and added up.','乘积量化（PQ）：把每个向量切成 m 段，每一段事先用 k-means 学出 256 个中心；存的时候，每段只记离它最近的那个中心的编号，1 个字节。一个向量只剩 m 个字节；距离靠查一张小表再加起来。');
const D_GB=()=>tx('GB here is 10⁹ bytes and TB is 10¹² bytes (1,000 GB), as disk and memory sizes of vector indexes are usually quoted.','这里 GB 是 10⁹ 字节，TB 是 10¹² 字节（1,000 GB），向量索引的大小通常这样计。');
const D_BW=()=>tx('This lab assumes one server scans the vectors in its memory at 100 GB/s, a conservative value the lab picks. Time = bytes read ÷ 100 GB/s. It leaves out the arithmetic and assumes all vectors fit in memory.','这个 Lab 假设一台服务器扫描内存里的向量每秒 100 GB，这是这个 Lab 取的一个保守值。时间 = 读的字节数 ÷ 100 GB/s。没有算运算时间，也假设向量都放得进内存。');
const D_SRV=()=>tx('This lab assumes one server has 512 GB of memory. A server can be fitted with several TB, at a much higher price; with another budget, what fits changes.','这个 Lab 假设一台服务器有 512 GB 内存。服务器也能装到几 TB，但贵得多；预算不同，放得下放不下也跟着变。');
const D_KD=()=>tx('A k-d tree splits the points in half on one coordinate, then each half on the next coordinate, and so on. A search goes down to the question’s region, then backs up and checks every other region that could still hold a closer point.','k-d 树先按一个坐标把点分成两半，每一半再按下一个坐标分，一直分下去。搜索先走到问题所在的那一小块，再往回查每一块可能还有更近点的区域。');
/* "1M vectors" / "1M 个向量" / "10 亿个向量" (no space between the 10 亿 and 个) */
const NBV=n=>tx(`${NB(n)} vectors`,n>=1e9?`${NB(n)}个向量`:`${NB(n)} 个向量`);
/* bytes per vector at real scale (768 dimensions): IVF-PQ is taken as a 64-byte code (12 dimensions per piece) and an 8-byte id, the
   same assumption in the metric card, the math, the observation and the summary (review polish r1, P15) */
const PQ_REAL=64;
const bigBytes=s=>s.ix==='pq'?PQ_REAL+8:bytesPer(s,REAL_D);

/* ---------- state ---------- */
const DEFAULT={ix:'flat',np:4,M:16,ef:16,m:8};
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0, realN=1e6, kdD=2;
const triedNP=new Set(), triedEF=new Set(), triedPQ=new Set(), triedN=new Set(), triedD=new Set(), shownPrev={}, seenPic=new Set();
let picSpeed='mid', anim=null, raf=0, lastTs=0, picF=-1, pendObs=false;
/* the picture's search is a mechanism replay (R3.9): in the guided tour each step of it stays on screen for its own time (picEvents),
   times this factor; in Scenarios and the challenge, where readers switch back and forth, the part that plays takes 1.5 s at 中 */
const PIC_SPD={slow:2,mid:1,fast:0.5}, SB_DUR={slow:3,mid:1.5,fast:0.75}, PIC_LEAD=0.5;
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const cur=()=>({...S});

/* ---------- guided parts ---------- */
const PRED=[{v:'miss'},{v:'all'},{v:'ten'}];
const predTxt=v=>({miss:tx('About 0.1% of the vectors; it sometimes misses one of the true nearest','约 0.1% 的向量；有时会漏掉真正最近的'),
  all:tx('About 0.1% of the vectors; it always finds the true nearest','约 0.1% 的向量；总能找到真正最近的'),
  ten:tx('About 10% of the vectors; it always finds the true nearest','约 10% 的向量；总能找到真正最近的')}[v]);
function inSeg(id,key,label,opts,val){return A.ui.seg({id:'vix-ic-'+id,key,label,options:opts,value:String(val),inline:true});}
const inNP=()=>inSeg('np','np',tx('Cells to probe, nprobe','要查的簇数 nprobe'),NPROBES.map(v=>[String(v),String(v)]),S.np);
const inEF=()=>inSeg('ef','ef',tx('Candidates kept, ef','候选数 ef'),EFS.map(v=>[String(v),String(v)]),S.ef);
const inPQ=()=>inSeg('m','m',tx('Bytes per code, m','每个编码的字节数 m'),PQMS.map(v=>[String(v),String(v)]),S.m);
const inRN=()=>inSeg('rn','rn',tx('Stored vectors','存的向量数'),[['1e6','1M'],['1e8','100M'],['1e9','1B']],String(realN===1e6?'1e6':realN===1e8?'1e8':'1e9'));
const inKD=()=>inSeg('kd','kd',tx('Dimensions','维数'),KDD.map(v=>[String(v),String(v)]),kdD);
const REC_TEX=nm=>`\\text{${nm}} = \\frac{\\text{true nearest found}}{10}`;
const REC_TEX_ZH=nm=>`\\text{${nm}} = \\frac{\\text{找回的真正近邻数}}{10}`;
/* review polish r1: each card opens with its point in two sentences; the picture of each part plays what the card says (pic) */
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'1B vectors; the nearest 10 in milliseconds.',zh:'10 亿个向量；几毫秒内找出最近的 10 个。'},
  pic:'intro',panels:['pic'],enter:()=>({...DEFAULT}),
  body:()=>tx(`<p>Lumo Home, a company made up for this lab, wants its help-center search to cover 1B passages, each stored as a vector of 768 numbers (see ${LL('embeddings')}). A customer’s question becomes a vector too, and search has to return the 10 passages closest to it within a few milliseconds: this is ${DEF('nearest-neighbour search',D_NN())}.</p><p>Comparing the question with every vector takes far too long, so search goes through a <b>vector index</b>, built in advance, that lets each search compare only a small part of the vectors. This lab tries several kinds and measures how many of the true nearest each one misses in order to be faster or to save memory.</p><p>${J('pic','The picture')} is a 2-dimensional sketch: the dots are stored vectors, the cross is a question and the ringed dots are its 10 nearest. The measured numbers later come from your browser: 5,000 vectors of 32 dimensions, a smaller scale, but the indexes are really built and searched.</p>`,
              `<p>小鹿家居（为这个 Lab 编的一家公司）想让帮助中心的搜索覆盖 10 亿段文字，每段存成一个 768 维的向量（见 ${LL('embeddings')}）。用户的问题也变成向量，搜索要在几毫秒内找出离它最近的 10 段，这叫${DEF('最近邻搜索',D_NN())}。</p><p>和每个向量都比一次太慢，所以要先建好<b>向量索引</b>，让每次搜索只比一小部分。这个 Lab 依次试几种索引，量出每种为了更快或更省内存，会漏掉多少真正最近的向量。</p><p>${J('pic','图')}是 2 维的示意：点是存着的向量，叉号是一个问题，带圈的是离它最近的 10 个。后面的实测数来自你的浏览器：5,000 个 32 维向量，规模缩小了，但索引是真的建、真的搜。</p>`)},
 {id:'exact',title:{en:'Exact search',zh:'精确搜索'},lead:{en:'Compare with every vector: right, and slow.',zh:'和每个向量都比一次：结果对，但是慢。'},
  pic:'flat',panels:['pic','scale'],enter:()=>({...DEFAULT}),target:'rn',expect:()=>triedN.size>1,showMe:{rn:1e9},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another number of vectors to continue.',zh:'换一个向量数再继续。'},
  body:()=>tx(`<p>Exact search compares the question with every vector and keeps the 10 closest, so it misses none; the price is reading every vector for every question. 1B vectors of 768 dimensions (float32, 3,072 bytes each) take 3,072 ${DEF('GB',D_GB())}, and at an ${DEF('assumed 100 GB/s',D_BW())} one read takes about 30 seconds.</p><p>In ${J('pic','the picture')} it compares the points from left to right; the dashed circle holds the 10 nearest so far and shrinks whenever a closer point turns up. Pick how many vectors the help center stores, and see how long one read takes and whether it fits in memory:</p>`,
              `<p>精确搜索把问题和每个向量都比一次，留下最近的 10 个，一个不漏；代价是每个问题都要把全部向量读一遍。10 亿个 768 维向量（float32，每个 3,072 字节）一共 3,072 ${DEF('GB',D_GB())}，按${DEF('每秒读 100 GB',D_BW())} 算，读一遍约 30 秒。</p><p>${J('pic','图')}里它从左到右逐个比较；虚线圆圈住目前最近的 10 个，比到更近的点，圆就缩小。选帮助中心存多少个向量，看读一遍要多久、内存放不放得下：</p>`)+inRN(),
  obs:()=>obsExact()},
 {id:'trees',title:{en:'Why trees stop working',zh:'为什么树不管用了'},lead:{en:'A k-d tree skips whole regions in 2 dimensions, not in 32.',zh:'k-d 树在 2 维能跳过整块区域，到 32 维就跳不过去了。'},
  pic:'kd',panels:['pic','kd'],enter:()=>({...DEFAULT}),target:'kd',expect:()=>triedD.size>1,showMe:{kd:32},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another number of dimensions to continue.',zh:'换一个维数再继续。'},
  body:()=>tx(`<p>A ${DEF('k-d tree',D_KD())} cuts space in half again and again, and a search checks only the few regions near the question, skipping the rest whole. In many dimensions almost every region may hold a closer point, so almost nothing can be skipped and the tree is as slow as exact search.</p><p>Watch ${J('pic','the picture')} (2 dimensions): the search first goes down to the question’s region, and the circle’s radius is the nearest distance found so far; a region the circle does not reach turns grey and is skipped. This part looks for the nearest one only; looking for 10 needs a larger circle, which skips even less.</p><p>Pick a number of dimensions and see how many vectors the tree compares:</p>`,
              `<p>${DEF('k-d 树',D_KD())}把空间一次次对半切开，搜索只查离问题近的几块，其余的整块跳过。维数一高，几乎每一块里都可能有更近的点，跳不过去，树就和精确搜索一样慢。</p><p>看${J('pic','图')}（2 维）：搜索先走到问题所在的那一块，圆的半径是目前找到的最近距离；圆碰不到的块整块变灰，不用查。这一步只找最近的 1 个；找 10 个时圆更大，能跳过的更少。</p><p>选一个维数，看树要比较多少向量：</p>`)+inKD(),
  obs:()=>obsKD()},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'1,000 cells, search only the nearest one.',zh:'1,000 个簇，只查最近的那一个。'},
  pic:'cells',panels:['pic'],enter:()=>({...DEFAULT,ix:'ivf',np:1}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(predTxt(o.v))}</button>`).join('')}</div>`;
   return tx(`<p>${DEF('IVF',D_IVF())} (inverted file) first splits the vectors into cells, and a search opens only the cells nearest the question. Suppose the help center’s 1B vectors sit in 1,000 cells of about equal size and a search opens just 1: what share of the vectors does a question get compared with, and does it always find the true nearest?</p><p>The cells come from ${DEF('k-means',D_KM())}: each cell has a centre, and every vector is stored in the list of its nearest centre. A search compares the question with the centres first, then only with the vectors of the nprobe nearest cells (here nprobe = 1). ${J('pic','The picture')} has 16 cells: the small squares are centres and the lines are cell borders.</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
              `<p>${DEF('IVF',D_IVF())}（倒排文件）先把向量分成很多簇，搜索只打开离问题最近的几个簇。假设帮助中心的 10 亿个向量分在 1,000 个大小差不多的簇里，搜索只打开 1 个：一个问题要和多大比例的向量比？它总能找到真正最近的吗？</p><p>分簇用 ${DEF('k-means',D_KM())}：每簇有一个中心，每个向量存进离它最近的中心的列表。搜索时，问题先和各个中心比，再只和最近 nprobe 个簇里的向量比（这里 nprobe = 1）。${J('pic','图')}里有 16 个簇，小方块是中心，线是簇的边界。</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'ivf',title:{en:'IVF: probe more cells',zh:'IVF：多查几个簇'},lead:{en:'More cells, higher recall, more comparisons.',zh:'簇查得越多，召回率越高，比较也越多。'},
  pic:'ivf',panels:['pic','chart','metrics'],curves:['ivf'],enter:()=>({...DEFAULT,ix:'ivf',np:1}),target:'np',expect:()=>triedNP.size>1,showMe:{np:16},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another nprobe to continue.',zh:'换一个 nprobe 再继续。'},
  body:()=>{const ok=pred==='miss', v=pred?tx(`<p class="${ok?'okline':'badline'}"><b>${ok?'Right.':'Not quite.'}</b>${ok?'':` You predicted “${esc(predTxt(pred))}”. The right one is the first.`}</p>`,`<p class="${ok?'okline':'badline'}"><b>${ok?'对了。':'不对。'}</b>${ok?'':`你预测的是“${esc(predTxt(pred))}”，正确的是第一项。`}</p>`):tx('<p>The answer: about 0.1% of the vectors, and it sometimes misses one of the true nearest.</p>','<p>答案：约 0.1% 的向量，而且有时会漏掉真正最近的。</p>');
   return v+tx(`<p>It compares only about 0.1% of the vectors, but it can miss: a true neighbour may sit just across a cell border, in a cell that is not opened (the red rings in ${J('pic','the picture')} at nprobe = 1). Probing more cells finds more of them and compares more vectors. A search like this, which compares only a part and sometimes misses, is called ${DEF('approximate nearest-neighbour search (ANN)',D_ANN())}.</p><p>${DEF('Recall@10',D_REC())} is the share of the true 10 nearest that a search returns:</p>${M(REC_TEX('recall@10'))}<p>Pick nprobe and watch ${J('chart','the chart')}: every dot is one setting.</p>`,
                 `<p>只比约 0.1% 的向量，但会漏：真正的近邻可能在簇边界的另一侧、没打开的簇里（nprobe = 1 时${J('pic','图')}里的红圈）。多查几个簇，找回的多，比较的也多。这种只比一部分、偶尔会漏的搜索叫${DEF('近似最近邻（ANN）',D_ANN())}。</p><p>${DEF('recall@10',D_REC())} 是真正最近的 10 个里找回的比例：</p>${M(REC_TEX_ZH('recall@10'))}<p>选一个 nprobe，看${J('chart','曲线图')}：每个点是一种设置。</p>`)+inNP();},
  obs:()=>obsIVF()},
 {id:'hnsw',title:{en:'HNSW: walk a graph',zh:'HNSW：在图上走'},lead:{en:'Follow links toward the question.',zh:'顺着连线往问题走。'},
  pic:'hnsw',panels:['pic','chart','metrics'],curves:['ivf','hnsw'],enter:()=>({...DEFAULT,ix:'hnsw',M:16,ef:10}),target:'ef',expect:()=>triedEF.size>1,showMe:{ef:64},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another ef to continue.',zh:'换一个 ef 再继续。'},
  body:()=>tx(`<p>${DEF('HNSW',D_HNSW())} links every vector to a few of its nearest into a graph; a search follows the links toward the question and compares only the points it meets on the way. ef is how many candidates the search keeps: the more it keeps, the less likely it stops halfway, and the more it compares.</p><p>In ${J('pic','the picture')} the search starts at the entry point and takes a few hops on the upper layers, then expands candidates round by round on the bottom layer: each round expands the nearest one, and a neighbour of it closer than the farthest candidate kept is kept too. Once every candidate in the dashed circle has been expanded, the search stops.</p><p>The measured graph links each point to at most M = 16 neighbours per layer (${MI('2 \\times M')} on the bottom layer); the picture uses M = 4. Pick ef:</p>`,
              `<p>${DEF('HNSW',D_HNSW())} 把每个向量和最近的几个连成图；搜索沿着连线走向问题，只比较路上的点。ef 是留着的候选数：越多，越不容易停在半路，比较也越多。</p><p>${J('pic','图')}里，搜索从入口出发，先在上层走几步，再到最底层一轮轮展开最近的候选：它的邻居比留着的最远那个近，就留下。虚线圆里的候选都展开过，搜索就停。</p><p>实测的图每层最多连 M = 16 个邻居（最底层 ${MI('2 \\times M')} 个）；图里 M = 4。选一个 ef：</p>`)+inEF(),
  obs:()=>obsHNSW()},
 {id:'pq',title:{en:'PQ: compress the vectors',zh:'PQ：压缩向量'},lead:{en:'A few bytes per vector, some recall lost.',zh:'每个向量只剩几个字节，召回率掉一些。'},
  pic:'pq',panels:['pic','chart','metrics'],curves:['ivf','pq'],enter:()=>({...DEFAULT,ix:'pq',m:8,np:16}),target:'m',expect:()=>triedPQ.size>1,showMe:{m:16},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another number of bytes to continue.',zh:'换一个字节数再继续。'},
  body:()=>tx(`<p>1B vectors take 3,072 GB, more than one ${DEF('512 GB',D_SRV())} server holds. ${DEF('Product quantization (PQ)',D_PQ())} cuts a vector into m pieces and keeps one byte for each: the number of its nearest of 256 centres.</p><p>The price is that the points move: in ${J('pic','the picture')} each point moves to the nearest grid crossing and the search ranks the moved points, so a true neighbour can be pushed out of the top 10.</p><p>IVF-PQ combines IVF and PQ: it opens only the nearest few cells and compares the codes inside them. Pick m (probing 16 cells):</p>`,
              `<p>10 亿个向量有 3,072 GB，一台 ${DEF('512 GB',D_SRV())} 的服务器放不下。${DEF('乘积量化（PQ）',D_PQ())}把向量切成 m 段，每段只存 1 个字节：256 个中心里最近那个的编号。</p><p>代价是点都挪了位置：${J('pic','图')}里每个点挪到最近的网格交点，搜索按挪过的位置排，真正的近邻可能掉出前 10 名。</p><p>IVF-PQ 把 IVF 和 PQ 合起来：只打开最近的几个簇，比较簇里的编码。选一个 m（查 16 个簇）：</p>`)+inPQ(),
  obs:()=>obsPQ()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Speed, memory, recall: each index trades them.',zh:'速度、内存、召回率：每种索引都在取舍。'},
  pic:'hnsw',panels:['chart','metrics'],curves:['ivf','hnsw','pq'],enter:()=>({...DEFAULT,ix:'hnsw',M:16,ef:64}),
  body:()=>rememberHTML()}];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
function obsExact(){const bytes=realN*REAL_D*4, t=bytes/BW;
  const head=tx(`<b>${NBV(realN)}: ${GB(bytes)}, about ${secs(t)} to read once.</b>`,`<b>${NBV(realN)}：${GB(bytes)}，读一遍约 ${secs(t)}。</b>`);
  const rest=realN===1e6?tx(' One server holds them, but one read already takes several times the few milliseconds wanted; 100 times the vectors take 100 times as long.','一台服务器放得下，但读一遍已经是几毫秒目标的好几倍；向量多 100 倍，时间也多 100 倍。')
    :realN===1e8?tx(` One ${DEF('512 GB',D_SRV())} server still holds them, but one read takes hundreds of times the few milliseconds wanted.`,`一台 ${DEF('512 GB',D_SRV())} 的服务器还放得下，但读一遍的时间是几毫秒目标的几百倍。`)
    :tx(` That is thousands of times the few milliseconds wanted, and ${int(bytes/512e9)} times the memory of one ${DEF('512 GB',D_SRV())} server, so it does not fit either.`,`这是几毫秒目标的几千倍；内存也是一台 ${DEF('512 GB',D_SRV())} 服务器的 ${int(bytes/512e9)} 倍，放不下。`);
  return `<p>${head}${rest}</p>`;}
function obsKD(){const k=kdStats(), v=k[kdD];
  const first=kdD===2?tx(`<b>2 dimensions: the tree compares only ${kdPc(v)} of the 5,000 vectors.</b> The circle is small, so most regions are skipped whole.`,`<b>2 维：树只比较 5,000 个向量里的 ${kdPc(v)}。</b>圆很小，大多数块都整块跳过。`)
    :kdD===8?tx(`<b>8 dimensions: the tree compares ${kdPc(v)} of the 5,000 vectors.</b> It already has to go back into many regions.`,`<b>8 维：树要比较 5,000 个向量里的 ${kdPc(v)}。</b>它已经要回头查很多块。`)
    :tx(`<b>32 dimensions: the tree compares ${kdPc(v)} of the 5,000 vectors.</b> It compares almost every vector, so it is no faster than exact search: in 32 dimensions, the circle reaches nearly every region.`,`<b>32 维：树要比较 5,000 个向量里的 ${kdPc(v)}。</b>几乎每个向量都要比，和精确搜索一样慢：到了 32 维，那个圆几乎碰到每一块。`);
  return `<p>${first}</p>`+tx('<p>Embeddings have hundreds of dimensions, and the more dimensions, the less a tree can skip. So the indexes that follow take another route: they compare the question with only a small part of the vectors. What that costs, the next part asks you to guess first.</p>','<p>embedding 有几百维，维数越高，树越跳不过去。所以后面的索引换一种思路：只和一小部分向量比。代价是什么，下一部分先猜。</p>');}
function obsIVF(){const s=stat('ivf',S.np,0), s1=stat('ivf',1,0);
  return tx(`<p><b>${S.np===1?'1 cell':`${S.np} cells`} probed: recall ${pc1(s.rec)} with ${int(s.comps)} comparisons.</b> ${S.np===1?'One cell is cheap but misses the neighbours across its borders.':`With 1 cell, recall is ${pc1(s1.rec)}.`}</p><p>Each extra cell costs about ${int(N/NLIST)} more comparisons, the size of an average cell.</p>`,
            `<p><b>查 ${S.np} 个簇：召回率 ${pc1(s.rec)}，每个问题比较 ${int(s.comps)} 次。</b>${S.np===1?'只查一个簇很便宜，但会漏掉边界另一侧的近邻。':`只查 1 个簇时是 ${pc1(s1.rec)}。`}</p><p>每多查一个簇，约多比较 ${int(N/NLIST)} 个向量（一个簇的平均大小）。</p>`);}
function obsHNSW(){const s=stat('hnsw',S.M,S.ef), b=stat('hnsw',S.M,10);
  return tx(`<p><b>ef = ${S.ef}: recall ${pc1(s.rec)} with ${int(s.comps)} vectors compared.</b> ${S.ef===10?'With only 10 candidates the walk often stops before it reaches the true neighbours.':`At ef = 10 recall is ${pc1(b.rec)}.`}</p><p>On ${J('chart','the chart')} the HNSW line sits above and to the left of IVF’s: at the same recall it compares far fewer vectors. The price is the links: at M = ${S.M} each vector stores ${8*S.M} more bytes.</p>`,
            `<p><b>ef = ${S.ef}：召回率 ${pc1(s.rec)}，每个问题比较 ${int(s.comps)} 个向量。</b>${S.ef===10?'只留 10 个候选时，搜索常常还没走到真正的近邻就停了。':`ef = 10 时召回率是 ${pc1(b.rec)}。`}</p><p>${J('chart','曲线图')}上，HNSW 的线在 IVF 的左上方：召回率一样时，比较少得多。代价是连线：M = ${S.M} 时，每个向量多存 ${8*S.M} 字节。</p>`);}
function obsPQ(){const s=stat('pq',S.m,16), iv=stat('ivf',16,0), big=GB(REAL_N*(PQ_REAL+8));
  /* the 72 GB says its premise, a 64-byte code at 768 dimensions; 136 is named (review polish check r1, N2) */
  return tx(`<p><b>${S.m} bytes per code: recall ${pc1(s.rec)}; IVF with the full vectors reaches ${pc1(iv.rec)}.</b> Each vector takes ${S.m+8} bytes (the code plus an 8-byte id), where IVF with the full vectors takes ${4*D+8}; ${S.m===16?'more bytes move each point less.':'fewer bytes move each point farther.'}</p><p>At real scale, with a 768-dimensional vector’s code counted as ${PQ_REAL} bytes, 1B vectors take ${big} with their ids, which fits in one server.</p>`,
            `<p><b>每个编码 ${S.m} 字节：召回率 ${pc1(s.rec)}，完整向量的 IVF 是 ${pc1(iv.rec)}。</b>每个向量占 ${S.m+8} 字节（编码加 8 字节 id），完整向量的 IVF 要 ${4*D+8} 字节；${S.m===16?'字节越多，点挪得越少。':'字节越少，点挪得越远。'}</p><p>按 768 维、编码 ${PQ_REAL} 字节算，10 亿个向量连 id 只要 ${big}，一台服务器放得下。</p>`);}
function rememberHTML(){const hn=stat('hnsw',16,64), big=GB(REAL_N*(PQ_REAL+8));
  /* two misconceptions, the RAG one with its premise (review polish r1, P17) */
  return tx(`<ul><li>Only exact search misses nothing, but at 1B vectors each question reads 3,072 GB.</li><li>First see which limit you hit first, memory or speed. If memory holds the full vectors, HNSW compares the fewest: here at ef = 64 it compares ${int(hn.comps)} and reaches ${pc1(hn.rec)}.</li><li>If it does not, use IVF-PQ: with ${PQ_REAL}-byte codes, 1B vectors take only ${big}, at the price of some recall.</li><li>nprobe and ef trade work for recall: raise them until the recall measured on your own data is high enough.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Approximate means inaccurate.” The misses can be measured and reduced: probe more cells or raise ef.</p><p>✗ “99% recall is always needed.” It depends on the product: at 90% recall@10, the 10 nearest passages miss 1 on average. A RAG answer that needs one good passage in the top 10 usually loses nothing by it; a duplicate-detection job may need every match.</p></div>${A.shell.toScenarios()}`,
            `<ul><li>只有精确搜索一个不漏，但 10 亿个向量时每个问题读 3,072 GB。</li><li>先看先碰到哪个限制：内存还是速度。内存放得下完整向量时，HNSW 比较得最少：这里 ef = 64 时比较 ${int(hn.comps)} 个，召回率 ${pc1(hn.rec)}。</li><li>放不下时用 IVF-PQ：编码 ${PQ_REAL} 字节时，10 亿个向量只要 ${big}，代价是掉一些召回率。</li><li>nprobe、ef 用计算量换召回率：往上调，直到自己数据上测出的召回率够用。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “近似就是不准。”漏掉多少能测出来，也能调低：多查几个簇、调大 ef。</p><p>✗ “召回率一定要 99%。”要看产品：recall@10 是 90% 时，最近的 10 段平均漏 1 段。RAG 只要前 10 段里有一段答得上，漏 1 段往往没影响；查重复内容，可能一个都不能漏。</p></div>${A.shell.toScenarios()}`);}

/* ---------- scenarios ---------- */
/* each scenario names the question it answers (review polish r1, P29) */
const PRESETS=[
 {t:{en:'Initial setup: exact search',zh:'初始设置：精确搜索'},d:{en:'All 5,000 vectors compared: what does the exact answer cost?',zh:'和全部 5,000 个向量比：标准答案的代价是多少？'},p:{ix:'flat'}},
 {t:{en:'IVF, probe 1 cell',zh:'IVF，只查 1 个簇'},d:{en:'FAISS’s default nprobe = 1: how much is missed without changing it?',zh:'FAISS 默认 nprobe = 1：不改设置时漏多少？'},p:{ix:'ivf',np:1}},
 {t:{en:'IVF, probe 16 cells',zh:'IVF，查 16 个簇'},d:{en:'16 of the 50 cells: how much is found, and how many comparisons are saved?',zh:'50 个簇里查 16 个：找回几成，省下多少比较？'},p:{ix:'ivf',np:16}},
 {t:{en:'HNSW with hnswlib’s defaults',zh:'HNSW，用 hnswlib 的默认值'},d:{en:'M = 16, ef = 10: how much is missed without changing it?',zh:'M = 16，ef = 10：不改设置时漏多少？'},p:{ix:'hnsw',M:16,ef:10}},
 {t:{en:'HNSW with FAISS’s defaults',zh:'HNSW，用 FAISS 的默认值'},d:{en:'M = 32, efSearch = 16: what do the extra links buy?',zh:'M = 32，efSearch = 16：多存的连线换来了什么？'},p:{ix:'hnsw',M:32,ef:16}},
 {t:{en:'IVF-PQ, 8 bytes per code',zh:'IVF-PQ，每个编码 8 字节'},d:{en:'16 cells probed, 16 bytes per vector: how much recall is lost?',zh:'查 16 个簇，每个向量 16 字节：召回率掉多少？'},p:{ix:'pq',m:8,np:16}}];
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
  else if(k==='ivf16') h+=tx(`<p>A third of the cells finds ${pc1(s.rec)} of the true neighbours and compares about a third of the vectors.</p><p>With only 5,000 vectors each cell holds about ${int(N/NLIST)}, and a question’s true neighbours are spread over several cells, so IVF cannot save much more here.</p>`,`<p>查三分之一的簇，找回 ${pc1(s.rec)} 的真正近邻，比较的向量也差不多是三分之一。</p><p>只有 5,000 个向量时，每个簇平均只有 ${int(N/NLIST)} 个向量，一个问题的真正近邻散在好几个簇里，所以 IVF 在这里省不了更多。</p>`);
  else if(k==='hl') h+=tx(`<p>hnswlib builds with M = 16 and searches with ef = 10 unless told otherwise. To raise recall, raise ef: at ef = 64, recall in this lab goes to ${pc1(stat('hnsw',16,64).rec)}.</p>`,`<p>不另外设置时，hnswlib 用 M = 16 建图，用 ef = 10 搜索。要提高召回率，就调大 ef：在这个 Lab 里调到 64，召回率升到 ${pc1(stat('hnsw',16,64).rec)}。</p>`);
  else if(k==='hf'){const h16=stat('hnsw',16,16);
    /* the chart draws both HNSW lines, M = 16 and M = 32, so “almost the same” can be checked on screen (review polish r1, P19) */
    h+=tx(`<p>FAISS’s IndexHNSW uses 32 links per layer by default (64 on the bottom), so each vector stores ${BYT(8*32)} of links, ${BYT(8*32-8*16)} more than at M = 16.</p><p>More links find more neighbours at the same ef, and compare more vectors too: at M = 16 (same ef), recall is ${pc1(h16.rec)} with ${int(h16.comps)} compared.</p><p>On the chart the two HNSW lines, M = 16 and M = 32, almost coincide: at the same number of comparisons they reach about the same recall, so the extra 128 bytes buy no fewer comparisons.</p>`,
          `<p>FAISS 的 IndexHNSW 默认每层 32 条连线（最底层 64 条），所以每个向量要存 ${BYT(8*32)}的连线，比 M = 16 时多 ${BYT(8*32-8*16)}。</p><p>连线多了，同样的 ef 下找回的近邻更多，比较的也更多：M = 16 时（ef 一样），召回率是 ${pc1(h16.rec)}，比较 ${int(h16.comps)} 个。</p><p>曲线图上 M = 16 和 M = 32 两条 HNSW 线几乎重合。比较次数一样时，两者的召回率差不多，所以多存的 128 字节没有换来更少的比较。</p>`);}
  /* re-ranking moved to "In production" (review polish r1, P20); the scenario points there (review polish check r1, N12) */
  else h+=tx(`<p>Each vector is 16 bytes instead of 136, and the search reads codes, not vectors.</p><p>Recall stops lower than IVF’s with full vectors (${pc1(stat('ivf',16,0).rec)}), because the codes move every point.</p><p>To win back part of that recall, see re-ranking (<code>IndexRefineFlat</code>) under ${J('prod','“In production”')} below the lab.</p>`,
             `<p>每个向量 16 字节，而不是 136 字节；搜索读的是编码，不是向量。</p><p>召回率比用完整向量的 IVF（${pc1(stat('ivf',16,0).rec)}）低，因为编码把每个点都挪了位置。</p><p>想找回一部分召回率，见页面下方${J('prod','“生产中怎么做”')}里的重排（<code>IndexRefineFlat</code>）。</p>`);
  return h;}

/* ---------- challenge ---------- */
const CH_REC=0.95, CH_COMP=1000, CH_INIT={ix:'flat',np:4,M:16,ef:16,m:8};
/* ef can be any whole number in hnswlib (set_ef) and FAISS (efSearch). Outside the menu (EFS), these ef ranges also meet both bars
   in this lab, and no other ef from 10 to 400 does; tests/vector-index_numbers.py scans them (B1 of review b10r1). The task asks for
   a choice among the options shown, and the result says so in an aside (review polish r1, P21). */
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
   `<p>${tx('The same 5,000 vectors (32 dimensions) and 100 questions. Once recall and speed meet their bars, the less memory the better, so the task scores bytes per vector.','还是这 5,000 个 32 维向量、这 100 个问题。召回率和速度都达标以后，内存越省越好，所以这道题比每个向量占多少字节。')}</p><p>${tx('Choose the index and its settings from the options below:','索引和它的设置，从下面的选项里选：')}</p>`+
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
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${why}.</b></p><p class="small">Hint: which index finds the true neighbours with the fewest comparisons, and how small can its links per layer, M, be while it still passes?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：${why}。</b></p><p class="small">提示：哪种索引用最少的比较找到真正的近邻？把它的每层连线数 M 调到多小，还能过线？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal&&!me.ok) return h;
  const rows=pts.map(p=>({label:chRowLabel(p),value:p.rec*100,text:pc1(p.rec),ok:p.ok,me:p===me,best:p.ok&&p.bytes===minB,
    tags:(sameSet(p.s,CH_INIT)?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  h+=A.ui.cmpChart({brief:true,title:tx('Recall@10 of the initial setup, your pick and the settings that meet both bars (goal at least 95%, at most 1,000 compared)','初始设置、你的选择和满足两条线的设置，各自的 recall@10（目标是至少 95%，最多比较 1,000 个）'),
    aria:tx('Recall@10 by index and setting.','按索引和设置的 recall@10。'),rows,goal:CH_REC*100,goalLabel:tx('goal: 95%','目标 95%'),max:100,
    hiddenText:n=>tx(`The other ${n} settings miss a bar and are not shown.`,`其余 ${n} 种设置至少没达到一条线，图里没有列出。`),
    okText:tx('meets both bars','两条线都满足'),noText:tx('misses at least one','至少一条没满足'),goalText:tx('goal: 95%','目标：95%')});
  const p16=stat('pq',16,32), h8a=stat('hnsw',8,64), h8=stat('hnsw',8,128), h4=stat('hnsw',4,256), b8=bytesPer({ix:'hnsw',M:8},D), b4=bytesPer({ix:'hnsw',M:4},D);
  /* the reasons in three paragraphs: why HNSW, why M = 16 with ef = 64 among the options, and an aside on ef outside them (review polish r1, P21) */
  h+=`<div class="small vix-why">`+
    tx(`<p>Why: only HNSW can meet both bars. IVF adds about ${int(N/NLIST)} comparisons per cell and goes over the limit before recall reaches 95%; IVF-PQ takes the fewest bytes, but its codes move the points, and probing 32 cells it still finds only ${pc1(p16.rec)}.</p>`+
       `<p>HNSW keeps the vector’s 128 bytes plus ${MI('8 \\times M')} bytes of links, so the smallest M that passes wins. With M = 8, ef = 64 reaches only ${pc1(h8a.rec)}, just short, and ef = 128 reaches ${pc1(h8.rec)} but compares ${int(h8.comps)}. With M = 4, ef = 256 compares ${int(h4.comps)}.</p>`+
       `<p>An aside: ef can in fact be any whole number (set_ef in hnswlib, efSearch in FAISS). With M = 8, any ef from ${EF_FREE[8][0]} to ${EF_FREE[8][1]} passes, at only ${b8} bytes per vector; with M = 4, ef from ${EF_FREE[4][0]} to ${EF_FREE[4][1]} also passes, at ${b4} bytes per vector, the fewest of the graphs this lab builds.</p>`,
       `<p>原因：只有 HNSW 能同时满足两条线。IVF 每多查一个簇就多比较约 ${int(N/NLIST)} 个向量，召回率还没到 95%，比较就超过了上限；IVF-PQ 每个向量最省，但编码把点挪了位置，查 32 个簇时召回率也只有 ${pc1(p16.rec)}。</p>`+
       `<p>HNSW 要存向量本身的 128 字节，再加 ${MI('8 \\times M')} 字节的连线，所以能过线的 M 越小越好。M = 8 时，ef = 64 只到 ${pc1(h8a.rec)}，差一点；ef = 128 到了 ${pc1(h8.rec)}，却要比较 ${int(h8.comps)} 个。M = 4 时，ef = 256 要比较 ${int(h4.comps)} 个。</p>`+
       `<p>题外：ef 其实可以取任意整数（hnswlib 的 set_ef，FAISS 的 efSearch）。M = 8 时，ef 取 ${EF_FREE[8][0]} 到 ${EF_FREE[8][1]} 都能过线，每个向量只要 ${b8} 字节；M = 4 时，ef 取 ${EF_FREE[4][0]} 到 ${EF_FREE[4][1]} 也能过线，每个向量 ${b4} 字节，是这个 Lab 建的几种图里最少的。</p>`)+`</div>`;
  h+=tx(`<div class="small muted vix-why"><p>Note: these numbers come from the lab’s 5,000 vectors. At another scale or on other data, measure recall again on a sample of your own questions.</p></div>`,
        `<div class="small muted vix-why"><p>注意：这些数来自这个 Lab 的 5,000 个向量。规模和数据变了，要用你自己的一批问题重新测召回率。</p></div>`);
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
  {q:tx('This lab assumes an IVF index holds 1B vectors in 65,536 cells of about equal size.</b></p><p><b>A search probes 64 cells. About how many vectors does one question get compared with?','这个 Lab 假设一个 IVF 索引把 10 亿个向量分在 65,536 个大小差不多的簇里。</b></p><p><b>搜索时查 64 个簇，一个问题大约要和多少个向量比较？'),
   o:[tx('About 977K vectors, plus the 65,536 centres','约 977K 个向量，再加 65,536 个中心'),tx('About 64','约 64 个'),tx('About 15,000','约 15,000 个')],a:0,
   why:tx(`${MB('\\text{per cell} = 10^9 \\div 65,536 \\approx 15,259')}${MB('\\text{vectors compared} = 64 \\times 15,259 \\approx 977,000')}That is just under 0.1% of the vectors; the 65,536 centres come on top.`,
          `${MB('\\text{每个簇} = 10^9 \\div 65,536 \\approx 15,259')}${MB('\\text{比较的向量} = 64 \\times 15,259 \\approx 977,000')}这还不到全部向量的 0.1%；另外还要和 65,536 个中心比。`)}],
 practice:[{id:'kd-tree',d:tx('Build and search a k-d tree with branch-and-bound pruning: the tree that compares almost every vector at 32 dimensions in this lab.','建一棵 k-d 树并用分支定界剪枝搜索：在这个 Lab 里，32 维时它几乎要比较每一个向量。')},
           {id:'rag-search-ml',d:tx('Its retrieval part: an ANN index (IVF, PQ, HNSW) and the setting for how widely it searches (nprobe, ef).','其中检索的部分：ANN 索引（IVF、PQ、HNSW），以及决定搜多大范围的参数（nprobe、ef）。')},
           {id:'rag-agent-platform',d:tx('A per-tenant ANN index with permission filters.','按租户分开、带权限过滤的 ANN 索引。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. See why exact search and trees are too slow, predict what IVF does when it probes only one cell, then try IVF, HNSW and PQ on the lab’s vectors.','一共 8 个部分。先看精确搜索和树为什么太慢，预测 IVF 只查一个簇时会怎样，再在这个 Lab 的向量上试 IVF、HNSW 和 PQ。'),
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
/* the picture and its partner (the chart, the scale bars or the k-d bars) sit side by side when the stage is wide enough, so the
   picture, the chart with its legend and the metric cards all fit in one screen at 1440 × 900 (review polish r1, P23) */
function stageHTML(){
  return `<div class="panel stage vix-stage" id="vix-stage">`+
   `<div class="stage-head vix-head"><div class="vix-st-title" id="vix-st-title"></div><div class="stage-actions"><span class="status" id="vix-status" tabindex="0"></span></div></div>`+
   `<div class="vix-transport" id="vix-transport"></div>`+
   `<div class="vix-figs" id="vix-figs">`+
    `<section class="vix-pan" id="vix-pan-pic"><div class="vix-pt" id="vix-p-title"></div><p class="vix-cap" id="vix-p-cap"></p><svg class="vix-pic" id="vix-pic" viewBox="0 0 360 260" role="img" aria-label=""></svg><div class="legend-row vix-leg" id="vix-p-leg"></div><p class="small muted vix-pnote" id="vix-p-note" hidden></p></section>`+
    `<section class="vix-pan" id="vix-pan-scale"><div class="vix-scale" id="vix-scale"></div></section>`+
    `<section class="vix-pan" id="vix-pan-kd"><div class="vix-kdbars" id="vix-kdbars"></div></section>`+
    `<section class="vix-pan" id="vix-pan-chart"><div class="vix-pt" id="vix-c-title"></div><svg class="vix-chart" id="vix-chart" viewBox="0 0 360 240" role="img" aria-label=""></svg><div class="legend-row vix-leg" id="vix-c-leg"></div></section>`+
   `</div>`+
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
   `<article class="panel c-card" id="vix-prod-card"><h3>${A.t('cProd')}</h3><ul class="vix-prod">`+
   tx(`<li>FAISS ${SRC(U_IVF,'<code>faiss/IndexIVF.h</code>')}: <code>nlist</code> cells and <code>size_t nprobe = 1;</code> “number of probes at query time”. ${SRC(U_CL,'<code>Clustering.h</code>')}: k-means with <code>min_points_per_centroid = 39</code>.</li>`+
      `<li>FAISS ${SRC(U_HN,'<code>faiss/impl/HNSW.h</code>')} sets <code>efConstruction = 40</code> and <code>efSearch = 16</code>; ${SRC(U_IHN,'<code>IndexHNSW.h</code>')} defaults to <code>M = 32</code>.</li>`+
      `<li>FAISS ${SRC(U_PQ,'<code>ProductQuantizer.h</code>')}: <code>M</code> subquantizers (the number of pieces, m in this lab) of <code>nbits</code> each (8 bits: 256 centres, one byte per piece); <code>IndexIVFPQ</code> keeps these codes in IVF lists.</li>`+
      `<li>hnswlib ${SRC(U_HG,'<code>hnswlib/hnswalg.h</code>')} keeps <code>maxM0_ = M_ * 2</code> links on layer 0 with 4-byte ids and sets <code>ef_ = 10</code>; ${SRC(U_HA,'<code>ALGO_PARAMS.md</code>')} puts the memory at “roughly M * 8-10 bytes per stored element”.</li>`+
      `<li>FAISS wiki, ${SRC(U_RF,'Pre- and post-processing')}: <code>IndexRefineFlat</code> recomputes the distances of the top results with the full vectors and sorts them again, winning back some of the recall PQ loses; the price is that it keeps the full vectors as well.</li>`,
      `<li>FAISS ${SRC(U_IVF,'<code>faiss/IndexIVF.h</code>')}：<code>nlist</code> 个簇，<code>size_t nprobe = 1;</code>，注释写“number of probes at query time”。${SRC(U_CL,'<code>Clustering.h</code>')}：k-means，<code>min_points_per_centroid = 39</code>。</li>`+
      `<li>FAISS 的 ${SRC(U_HN,'<code>faiss/impl/HNSW.h</code>')} 里 <code>efConstruction = 40</code>，<code>efSearch = 16</code>；${SRC(U_IHN,'<code>IndexHNSW.h</code>')} 里默认 <code>M = 32</code>。</li>`+
      `<li>FAISS ${SRC(U_PQ,'<code>ProductQuantizer.h</code>')}：<code>M</code> 个子量化器（也就是这个 Lab 里的段数 m），每个 <code>nbits</code> 位（8 位就是 256 个中心，每段 1 个字节）；<code>IndexIVFPQ</code> 把这些编码放在 IVF 的列表里。</li>`+
      `<li>hnswlib 的 ${SRC(U_HG,'<code>hnswlib/hnswalg.h</code>')} 里，第 0 层有 <code>maxM0_ = M_ * 2</code> 条连线，id 4 字节，<code>ef_ = 10</code>；${SRC(U_HA,'<code>ALGO_PARAMS.md</code>')} 说内存约是“roughly M * 8-10 bytes per stored element”。</li>`+
      `<li>FAISS wiki 的 ${SRC(U_RF,'Pre- and post-processing')}：<code>IndexRefineFlat</code> 用完整的向量把排在前面的结果重新算一次距离、重新排序，找回 PQ 丢掉的一部分召回率；代价是它还要存完整的向量。</li>`)+
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
/* review polish r1 (B1, P2, P6, P9, P13, P16, P25): every part's picture shows what its card says. The searches are replays of the
   real search on the 300 points, one event at a time: exact search (the 10 nearest so far in a shrinking circle), the k-d tree (the
   regions it skips), IVF (centres first, then the cells in order), HNSW (the upper layers, then the candidates on the bottom layer)
   and PQ (the points moving onto the grid). A true neighbour's ring turns green the moment the search keeps it, red at the end if it
   was missed. Part 1 and the prediction show still pictures. */
const PW=360, PH=260;
function picXY(){const p=pic(); if(p.tf) return p.tf; let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(let i=0;i<P2N;i++){const x=p.P.X[i*2], y=p.P.X[i*2+1]; x0=Math.min(x0,x); x1=Math.max(x1,x); y0=Math.min(y0,y); y1=Math.max(y1,y);}
  const pad=12, sc=Math.min((PW-2*pad)/(x1-x0),(PH-2*pad)/(y1-y0)), ox=(PW-(x1-x0)*sc)/2, oy=(PH-(y1-y0)*sc)/2;
  p.tf={X:x=>ox+(x-x0)*sc,Y:y=>PH-(oy+(y-y0)*sc),x0,x1,y0,y1,s:sc}; return p.tf;}
function voronoi(c,Cn,k){const tf=picXY(); let poly=[[tf.x0-1e4,tf.y0-1e4],[tf.x1+1e4,tf.y0-1e4],[tf.x1+1e4,tf.y1+1e4],[tf.x0-1e4,tf.y1+1e4]];
  const cx=Cn[c*2], cy=Cn[c*2+1];
  for(let o=0;o<k&&poly.length;o++){if(o===c) continue; const ox=Cn[o*2], oy=Cn[o*2+1], ax=ox-cx, ay=oy-cy, b=(ox*ox+oy*oy-cx*cx-cy*cy)/2, out=[];
    for(let i=0;i<poly.length;i++){const P=poly[i], Q=poly[(i+1)%poly.length], fp=ax*P[0]+ay*P[1]-b, fq=ax*Q[0]+ay*Q[1]-b;
      if(fp<=0) out.push(P); if((fp<0&&fq>0)||(fp>0&&fq<0)){const t=fp/(fp-fq); out.push([P[0]+t*(Q[0]-P[0]),P[1]+t*(Q[1]-P[1])]);}}
    poly=out;}
  return poly;}
const ANIM_K=['flat','kd','ivf','hnsw','pq'];
/* one search in the picture as events, each with its own time on screen at 中 (d, in seconds); for every point the event that
   compares it (cmpAt), for each of the true 10 nearest the event from which the search keeps it (okAt). `from` is where the part that
   depends on the setting starts: another setting of a search already watched plays only from there (R3.20) */
function picEvents(kind){const p=pic(), q=p.q, gi=new Map(p.gt.map((x,j)=>[x,j])), cmpAt=new Int32Array(P2N).fill(1e9), okAt=new Array(K).fill(1e9);
  if(kind==='flat'){const ord=range(P2N).sort((a,b)=>p.P.X[a*2]-p.P.X[b*2]||a-b), W=[], ev=[];
    for(const i of ord){hpush(W,[d2(p.Qs.X,q,p.P.X,i,2),i],gt); if(W.length>K) hpop(W,gt); ev.push({k:'cmp',id:i,far:W.length>=K?W[0][0]:Infinity});}
    ev.forEach((e,t)=>{cmpAt[e.id]=t; if(gi.has(e.id)) okAt[gi.get(e.id)]=t;});
    return {kind,ev,d:ev.map(()=>0.025),n:ev.length,from:0,cmpAt,okAt,found:p.gt.slice(),comps:P2N};}
  if(kind==='kd'){const tr=kdTrace(p.P,p.T,p.Qs,q); tr.ev.forEach((e,t)=>{if(e.k==='cmp'&&cmpAt[e.id]===1e9) cmpAt[e.id]=t;});
    return {kind,ev:tr.ev,d:tr.ev.map(e=>e.k==='cmp'?0.4:0.55),n:tr.ev.length,from:0,cmpAt,okAt,found:[tr.id],comps:tr.comps,skips:tr.ev.length-tr.comps};}
  if(kind==='ivf'){const np=Math.min(S.np,P2LIST), ord=probeOrder(p.I,p.Qs,q), ev=[{k:'ctr'},{k:'rank'}];
    for(let t=0;t<np;t++){const c=ord[t]; ev.push({k:'open',c,t}); for(const i of p.I.lists[c]){cmpAt[i]=2+t; if(gi.has(i)) okAt[gi.get(i)]=2+t;}}
    ev.push({k:'end'});   // the last cell stays open on screen for its own time, then the missed ones turn red
    const r=searchIVF(p.P,p.I,p.Qs,q,np,K);
    return {kind,ev,d:[2,1.6].concat(ev.slice(2,-1).map(()=>Math.min(1.2,8/np)),[0]),n:ev.length,from:2,cmpAt,okAt,found:r.ids,comps:r.comps,ord,np};}
  if(kind==='hnsw'){const tr=hnswTrace(p.P,p.G,p.Qs,q,S.ef), rounds=tr.ev.filter(e=>e.k==='round').length, rd=Math.min(0.8,9/Math.max(1,rounds));
    const D0={entry:0.8,layer:0.5,look:0.4,hop:0.4,down:0.5,bottom:0.9};   // the rounds on the bottom layer, where ef acts, get the most time
    tr.ev.forEach((e,t)=>{const mark=i=>{if(cmpAt[i]===1e9) cmpAt[i]=t;};
      if(e.k==='entry') mark(e.cur); else if(e.k==='look') e.seen.forEach(mark); else if(e.k==='round') e.comp.forEach(mark);
      for(const x of e.k==='bottom'?e.W:e.k==='round'?e.add:[]) if(gi.has(x)&&okAt[gi.get(x)]===1e9) okAt[gi.get(x)]=t;});
    return {kind,ev:tr.ev,d:tr.ev.map(e=>e.k==='round'?rd:D0[e.k]),n:tr.ev.length,from:tr.ev.findIndex(e=>e.k==='bottom'),cmpAt,okAt,found:tr.ids,comps:tr.comps,rounds};}
  if(kind==='pq'){const g=picPQ(S.m), ev=[{k:'orig'},{k:'grid'}]; for(let s=1;s<=8;s++) ev.push({k:'move',t:s/8}); ev.push({k:'rank'});
    cmpAt.fill(ev.length-1);   // PQ alone compares every code, when it ranks them
    return {kind,ev,d:[2,2].concat(new Array(8).fill(0.3),[1]),n:ev.length,from:1,cmpAt,okAt,found:g.ids,comps:P2N,g};}
  return null;}
const picKey=kind=>kind+'|'+(kind==='ivf'?Math.min(S.np,P2LIST):kind==='hnsw'?S.ef:kind==='pq'?S.m:'');
const PEV={};
function picEventsC(kind){const k=picKey(kind); return PEV[k]||(PEV[k]=picEvents(kind));}
/* HNSW at frame fr: the layer shown, the walk's hops, the last look, the candidates W and the farthest one's distance, the expanded */
function hnswState(ev,fr){const G=pic().G, st={L:null,cur:G.entry,hops:[],look:null,W:null,far:null,exp:new Set(),round:null,phase:'up'};
  for(let t=0;t<fr;t++){const e=ev.ev[t];
    if(e.k==='entry') st.cur=e.cur;
    else if(e.k==='layer'){st.L=e.L; st.look=null;}
    else if(e.k==='look') st.look=e;
    else if(e.k==='hop'){st.hops.push(e); st.cur=e.to;}
    else if(e.k==='down'){st.L=e.L-1; st.look=null; st.cur=e.cur;}
    else if(e.k==='bottom'){st.phase='bottom'; st.L=0; st.cur=e.cur; st.W=e.W; st.far=e.far; st.look=null;}
    else if(e.k==='round'){st.exp.add(e.c); st.cur=e.c; st.W=e.W; st.far=e.far; st.round=t===fr-1?e:null;}}
  if(st.L==null){const f=ev.ev.find(e=>e.k==='layer'); st.L=f?f.L:0;}
  return st;}
function edgesOf(L){const p=pic(), G=p.G; p.ed=p.ed||{}; if(p.ed[L]) return p.ed[L]; const out=[], seen=new Set();
  for(let i=0;i<P2N;i++){const nb=G.level[i]>=L?G.links[L][i]:null; if(!nb) continue;
    for(const e of nb){const a=Math.min(i,e), b=Math.max(i,e), k=a*P2N+b; if(seen.has(k)) continue; seen.add(k); out.push([a,b]);}}
  return (p.ed[L]=out);}
/* text inside the picture, kept inside its box (R7.14) */
const twid=t=>[...t].reduce((s,ch)=>s+(/[一-鿿]/.test(ch)?10.5:6.3),0);
function svgLab(x,y,t,cls,anchor){const w=twid(t), a=anchor||'middle', lo=a==='start'?3:a==='end'?w+3:w/2+3, hi=a==='start'?PW-w-3:a==='end'?PW-3:PW-w/2-3;
  return `<text class="${cls}" x="${Math.max(lo,Math.min(hi,x)).toFixed(1)}" y="${Math.max(12,Math.min(PH-4,y)).toFixed(1)}" text-anchor="${a}">${esc(t)}</text>`;}
/* the true 10 nearest crowd together near the question, and their rings overlap at the picture's scale. A box in the empty top-left
   corner shows that crowd 3 times larger, with dots and rings at their usual size, so every ring can be told apart and counted; a
   dashed frame marks the part it enlarges (review polish check r1, P25) */
const ZB={x:4,y:4,w:108,h:88};
function picZoom(){const p=pic(); if(p.zoom!==undefined) return p.zoom; const tf=picXY(), R=p.gt.map(i=>[tf.X(p.P.X[i*2]),tf.Y(p.P.X[i*2+1])]);
  const near=R.filter((a,j)=>R.some((b,k)=>k!==j&&Math.hypot(a[0]-b[0],a[1]-b[1])<10));
  if(near.length<2) return (p.zoom=null);
  const xs=near.map(a=>a[0]), ys=near.map(a=>a[1]), x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  const s=Math.min(3,ZB.w/(x1-x0+8),ZB.h/(y1-y0+8)), w=ZB.w/s, h=ZB.h/s;
  return (p.zoom={s,x0:(x0+x1)/2-w/2,y0:(y0+y1)/2-h/2,w,h});}
/* a label beside its mark, at the first spot that covers none of the boxes (rings, the question, other labels), else the least */
const ovl=(a,b)=>Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
function placeLab(x,y,t,cls,spots,boxes){const w=twid(t); let best=null;
  for(const [dx,dy,a] of spots){const x0=a==='start'?x+dx:x+dx-w, b=[x0,y+dy-9,x0+w,y+dy+1];
    const o=boxes.reduce((s0,c0)=>s0+ovl(b,c0),0)+(b[0]<2||b[2]>PW-2||b[1]<2||b[3]>PH-2?1e4:0);
    if(!best||o<best.o) best={o,dx,dy,a,b}; if(!o) break;}
  boxes.push(best.b); return svgLab(x+best.dx,y+best.dy,t,cls,best.a);}
function drawPic(){
  const c=ctx(), kind=c.pic, p=pic(), tf=picXY(), svg=$('#vix-pic',app), q=p.q, qx=tf.X(p.Qs.X[q*2]), qy=tf.Y(p.Qs.X[q*2+1]);
  const F=v=>v.toFixed(1), P=i=>[tf.X(p.P.X[i*2]),tf.Y(p.P.X[i*2+1])];
  const ev=ANIM_K.includes(kind)?picEventsC(kind):null, n=ev?ev.n:0, fr=ev?Math.max(0,Math.min(n,picF<0?n:picF)):0, done=!ev||fr>=n;
  const line=(a,b,cls,extra)=>`<line class="${cls}" x1="${F(a[0])}" y1="${F(a[1])}" x2="${F(b[0])}" y2="${F(b[1])}"${extra||''}></line>`;
  const circ=(xy,r,cls)=>`<circle class="${cls}" cx="${F(xy[0])}" cy="${F(xy[1])}" r="${F(r)}"></circle>`;
  const cross=(xy,r,cls)=>`<path class="${cls}" d="M${F(xy[0]-r)} ${F(xy[1]-r)}L${F(xy[0]+r)} ${F(xy[1]+r)}M${F(xy[0]-r)} ${F(xy[1]+r)}L${F(xy[0]+r)} ${F(xy[1]-r)}"></path>`;
  const cmp=i=>!!ev&&ev.cmpAt[i]<fr, last=ev&&fr>0?ev.ev[fr-1]:null;
  let bg='', ln='', marks='', top='', pos=P, dotCls=i=>cmp(i)?' chk':'', cur=-1;
  /* boxes a label must not cover: the rings of the true 10 nearest, the question and the zoom box */
  const keep=p.gt.map(i=>{const a=P(i); return [a[0]-5.5,a[1]-5.5,a[0]+5.5,a[1]+5.5];}).concat([[qx-6,qy-6,qx+6,qy+6],[ZB.x,ZB.y,ZB.x+ZB.w,ZB.y+ZB.h]]);
  /* the true 10 nearest: a thin ring until the search keeps one (green), red at the end for each one it missed */
  let ringCls=kind==='intro'?()=>'nn':ev&&kind!=='kd'?(i,j)=>{const ok=done?ev.found.includes(i):ev.okAt[j]<fr; return ok?'ok':done?'bad':'nn';}:null;
  if(kind==='cells'||kind==='ivf'){const opened=new Set(), rank=new Map(), ctr=cI=>[tf.X(p.I.Cn[cI*2]),tf.Y(p.I.Cn[cI*2+1])];
    if(kind==='ivf'){for(let t=2;t<fr;t++) if(ev.ev[t].k==='open') opened.add(ev.ev[t].c); if(fr>=2) ev.ord.slice(0,ev.np).forEach((cI,t)=>rank.set(cI,t+1));}
    for(let cI=0;cI<P2LIST;cI++){const poly=voronoi(cI,p.I.Cn,P2LIST); if(!poly.length) continue;
      bg+=`<polygon class="vix-cell${opened.has(cI)?' on':''}" points="${poly.map(v=>F(tf.X(v[0]))+','+F(tf.Y(v[1]))).join(' ')}"></polygon>`;}
    if(kind==='ivf'&&fr>=1) for(let cI=0;cI<P2LIST;cI++){if(fr>=3&&!rank.has(cI)) continue; ln+=line([qx,qy],ctr(cI),'vix-cl'+(fr>=2?(rank.has(cI)?' on':' off'):''));}
    for(let cI=0;cI<P2LIST;cI++){const xy=ctr(cI); top+=`<rect class="vix-ctr${rank.has(cI)?' on':''}" x="${F(xy[0]-3.5)}" y="${F(xy[1]-3.5)}" width="7" height="7"></rect>`; keep.push([xy[0]-4,xy[1]-4,xy[0]+4,xy[1]+4]);}
    for(const [cI,r] of rank){const xy=ctr(cI); top+=placeLab(xy[0],xy[1],String(r),'vix-cn',[[6,-5,'start'],[-6,-5,'end'],[6,13,'start'],[-6,13,'end'],[0,-7,'start'],[0,15,'start']],keep);}}
  if(kind==='flat'&&last){if(isFinite(last.far)) marks+=circ([qx,qy],Math.sqrt(last.far)*tf.s,'vix-rad'); if(!done) cur=last.id;}
  if(kind==='kd'){const cl=(v,a,b)=>Math.max(a,Math.min(b,v));
    const cutL=(ax,v,b,cls)=>ax===0?line([tf.X(v),cl(tf.Y(b[3]),0,PH)],[tf.X(v),cl(tf.Y(b[2]),0,PH)],cls):line([cl(tf.X(b[0]),0,PW),tf.Y(v)],[cl(tf.X(b[1]),0,PW),tf.Y(v)],cls);
    (function cut(nd,dep,b){if(!nd||dep>=4) return; bg+=cutL(nd.ax,nd.v,b,'vix-cut'); const lo=b.slice(), hi=b.slice(); lo[nd.ax*2+1]=nd.v; hi[nd.ax*2]=nd.v; cut(nd.l,dep+1,lo); cut(nd.r,dep+1,hi);})(p.T,0,[-Infinity,Infinity,-Infinity,Infinity]);
    for(let t=0;t<fr;t++){const e=ev.ev[t];
      if(e.k==='skip'){const x0=cl(tf.X(e.box[0]),0,PW), x1=cl(tf.X(e.box[1]),0,PW), y0=cl(tf.Y(e.box[3]),0,PH), y1=cl(tf.Y(e.box[2]),0,PH);
        bg+=`<rect class="vix-skip${t===fr-1&&!done?' now':''}" x="${F(x0)}" y="${F(y0)}" width="${F(Math.max(0,x1-x0))}" height="${F(Math.max(0,y1-y0))}"></rect>`;}
      else ln+=cutL(e.ax,e.v,e.box,'vix-cut on');}
    if(last){marks+=circ([qx,qy],Math.sqrt(last.best)*tf.s,'vix-rad')+circ(P(last.bi),4.8,done?'vix-ring ok':'vix-best'); if(!done&&last.k==='cmp') cur=last.id;}}
  if(kind==='hnsw'){const G=p.G, st=hnswState(ev,fr), ep=P(G.entry);
    const links=L=>edgesOf(L).map(([a,b])=>line(P(a),P(b),L>0?'vix-ulink':st.phase==='up'?'vix-link dim':'vix-link')).join('');
    bg+=links(0);
    if(st.phase==='up'&&st.L>0){ln+=links(st.L); for(let i=0;i<P2N;i++) if(G.level[i]>=st.L) marks+=circ(P(i),3.6,'vix-unode');}
    if(st.phase==='up'&&st.look) for(const e of st.look.seen) ln+=line(P(st.look.from),P(e),'vix-nbl');
    const R=p.gt.map(P), short=(a,b)=>{const L=Math.hypot(b[0]-a[0],b[1]-a[1]); let tip=null;
      for(let d=0;d<=L-5;d++){const t=d/L, x=a[0]+(b[0]-a[0])*t, y=a[1]+(b[1]-a[1])*t; if(R.some(r=>Math.hypot(r[0]-x,r[1]-y)<9)) break; tip=[x,y];}
      return tip&&Math.hypot(tip[0]-a[0],tip[1]-a[1])>10?tip:null;};
    for(const h of st.hops){const a=P(h.from), b=short(a,P(h.to)); if(b) top+=line(a,b,'vix-hop',' marker-end="url(#vix-arr)"'); else ln+=line(a,P(h.to),'vix-hop');}
    if(st.phase==='bottom'){if(st.far!=null) marks+=circ([qx,qy],Math.sqrt(st.far)*tf.s,'vix-rad h');
      if(st.round&&!done){for(const e of st.round.comp) ln+=line(P(st.round.c),P(e),'vix-nbl'); for(const o of st.round.out) top+=cross(P(o),3.2,'vix-out');}}
    marks+=circ(ep,6.5,'vix-entry'); top+=svgLab(ep[0]+8,ep[1]-8,tx('entry','入口'),'vix-elab','start');
    if(!done) marks+=circ(P(st.cur),5.6,'vix-curn');
    dotCls=i=>st.exp.has(i)?' exp':cmp(i)?' chk':'';}
  if(kind==='pq'){const g=ev.g, tt=done?1:last&&last.k==='move'?last.t:0, S2=[...new Set([...p.gt,...ev.found])];
    const to=i=>[tf.X(g.snap[i][0]),tf.Y(g.snap[i][1])];
    pos=i=>{const a=P(i); if(!tt) return a; const b=to(i); return [a[0]+(b[0]-a[0])*tt,a[1]+(b[1]-a[1])*tt];};
    if(fr>=2){for(const x of g.cx) bg+=line([tf.X(x),0],[tf.X(x),PH],'vix-grid'); for(const y of g.cy) bg+=line([0,tf.Y(y)],[PW,tf.Y(y)],'vix-grid');}
    if(tt>0) for(const i of S2){ln+=line(P(i),pos(i),'vix-trail'); marks+=circ(P(i),2.2,'vix-orig');}
    if(done){const cnt=new Map(); for(let i=0;i<P2N;i++){const k=g.snap[i].join(','); cnt.set(k,(cnt.get(k)||0)+1);}
      /* each count sits beside its crossing where it covers no other count, ring or the question (review polish r1, P25) */
      const ks=new Set(S2.map(i=>g.snap[i].join(','))), boxes=p.gt.map(i=>{const a=P(i); return [a[0]-5,a[1]-5,a[0]+5,a[1]+5];}).concat([[qx-6,qy-6,qx+6,qy+6],[ZB.x,ZB.y,ZB.x+ZB.w,ZB.y+ZB.h]]);
      const SPOTS=[[5,-5,'start'],[-5,-5,'end'],[5,12,'start'],[-5,12,'end'],[9,-13,'start'],[-9,-13,'end'],[9,20,'start'],[-9,20,'end']];
      for(const k of ks){const m=cnt.get(k); if(m<2) continue; const [sx,sy]=k.split(',').map(Number); top+=placeLab(tf.X(sx),tf.Y(sy),'×'+m,'vix-cnt',SPOTS,boxes);}}}
  let dots=''; for(let i=0;i<P2N;i++) dots+=circ(pos(i),i===cur?3.6:2.6,'vix-dot'+dotCls(i)+(i===cur?' cur':''));
  if(ringCls) p.gt.forEach((i,j)=>{marks+=circ(P(i),4.8,'vix-ring '+ringCls(i,j));});
  top+=cross([qx,qy],5.5,'vix-q');
  const Z=ringCls?picZoom():null; let zoom='';
  if(Z){const zp=a=>[ZB.x+(a[0]-Z.x0)*Z.s,ZB.y+(a[1]-Z.y0)*Z.s], inZ=(a,m)=>a[0]>=Z.x0-m&&a[0]<=Z.x0+Z.w+m&&a[1]>=Z.y0-m&&a[1]<=Z.y0+Z.h+m;
    let zi='';
    if(kind==='ivf') for(let cI=0;cI<P2LIST;cI++){if(!ev.ev.slice(2,fr).some(e=>e.k==='open'&&e.c===cI)) continue; const poly=voronoi(cI,p.I.Cn,P2LIST);
      zi+=`<polygon class="vix-cell on" points="${poly.map(v=>zp([tf.X(v[0]),tf.Y(v[1])]).map(F).join(',')).join(' ')}"></polygon>`;}
    if(kind==='pq'){const g=ev.g;
      if(fr>=2){for(const x of g.cx){const X=zp([tf.X(x),0])[0]; if(X>ZB.x&&X<ZB.x+ZB.w) zi+=line([X,ZB.y],[X,ZB.y+ZB.h],'vix-grid');}
        for(const y of g.cy){const Y=zp([0,tf.Y(y)])[1]; if(Y>ZB.y&&Y<ZB.y+ZB.h) zi+=line([ZB.x,Y],[ZB.x+ZB.w,Y],'vix-grid');}}
      for(const i of new Set([...p.gt,...ev.found])){const a=P(i), b=pos(i); if(Math.hypot(a[0]-b[0],a[1]-b[1])>0.3&&(inZ(a,12)||inZ(b,12))) zi+=line(zp(a),zp(b),'vix-trail')+circ(zp(a),2.2,'vix-orig');}}
    for(let i=0;i<P2N;i++){const a=pos(i); if(inZ(a,4)) zi+=circ(zp(a),i===cur?3.6:2.6,'vix-dot'+dotCls(i)+(i===cur?' cur':''));}
    if(kind==='hnsw'&&!done){const a=P(hnswState(ev,fr).cur); if(inZ(a,4)) zi+=circ(zp(a),5.6,'vix-curn');}
    p.gt.forEach((i,j)=>{const a=P(i); if(inZ(a,4)) zi+=circ(zp(a),4.8,'vix-ring '+ringCls(i,j));});
    if(inZ([qx,qy],4)) zi+=cross(zp([qx,qy]),5.5,'vix-q');
    marks+=`<rect class="vix-zr" x="${F(Z.x0)}" y="${F(Z.y0)}" width="${F(Z.w)}" height="${F(Z.h)}"></rect>`;
    zoom=`<g class="vix-zoom"><rect class="vix-zbox" x="${ZB.x}" y="${ZB.y}" width="${ZB.w}" height="${ZB.h}" rx="3"></rect><g clip-path="url(#vix-zc)">${zi}</g>`+
      `<text class="vix-zlab" x="${ZB.x+2}" y="${ZB.y+ZB.h+12}">${esc(tx(`dashed frame ×${+Z.s.toFixed(1)}`,`虚线框放大 ${+Z.s.toFixed(1)} 倍`))}</text></g>`;}
  svg.innerHTML=`<defs><marker id="vix-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8z" class="vix-arrh"></path></marker>`+
    (Z?`<clipPath id="vix-zc"><rect x="${ZB.x+1}" y="${ZB.y+1}" width="${ZB.w-2}" height="${ZB.h-2}" rx="2"></rect></clipPath>`:'')+`</defs>`+bg+ln+dots+marks+top+zoom;
  const nf=ev?p.gt.filter(i=>ev.found.includes(i)).length:K, title=picTitle(kind), cap=ev?picCap(kind,ev,fr,nf):'';
  $('#vix-p-title',app).textContent=title;
  const ce=$('#vix-p-cap',app); ce.textContent=cap; ce.hidden=!cap;
  svg.setAttribute('aria-label',cap?title+tx('. ','。')+cap:title);
  const hasBad=!!ev&&kind!=='kd'&&nf<K, lg=$('#vix-p-leg',app), lk=A.lang()+kind+hasBad; if(lg._k!==lk){lg._k=lk; lg.innerHTML=picLegend(kind,hasBad);}
  const pn=$('#vix-p-note',app), nt=picNote(kind); pn.hidden=!nt; if(nt&&pn.textContent!==nt) pn.textContent=nt;
  return {fr,n,nf};}
function picTitle(kind){const np=Math.min(S.np,P2LIST);
  if(kind==='intro') return tx('The help center’s vectors, as a 2-dimensional sketch','帮助中心的向量（2 维示意）');
  if(kind==='flat') return tx('A 2-D sketch: exact search compares the question with all 300 points','2 维示意：精确搜索把问题和 300 个点都比一次');
  if(kind==='kd') return tx('A 2-D sketch: a k-d tree looks for the nearest point','2 维示意：k-d 树找最近的 1 个点');
  if(kind==='cells') return tx('A 2-D sketch: 300 points in 16 cells, and a question','2 维示意：300 个点分在 16 个簇里，还有一个问题');
  if(kind==='ivf') return S.np>=P2LIST?tx(`A 2-D sketch: the picture has only 16 cells, so probing ${S.np} opens them all`,`2 维示意：图里只有 16 个簇，查 ${S.np} 个就是全部打开`):tx(`A 2-D sketch: IVF opens only the nearest ${np===1?'cell':`${np} cells`} of 16`,`2 维示意：IVF 只打开 16 个簇里最近的 ${np} 个`);
  if(kind==='hnsw') return tx(`A 2-D sketch: HNSW walks the graph (M = 4, ef = ${S.ef})`,`2 维示意：HNSW 在图上走（M = 4，ef = ${S.ef}）`);
  return tx(`A 2-D sketch: PQ alone, no cells (${S.m} centres for x and for y)`,`2 维示意：只画 PQ，没有分簇（x、y 各 ${S.m} 个中心）`);}
/* the caption under the title says what the frame on screen shows (the shared clock: picture, caption, scrubber and dock) */
function picCap(kind,ev,fr,nf){const n=ev.n, last=fr>0?ev.ev[fr-1]:null;
  if(fr<n){const b=capMode(kind,ev,fr); if(b) return capBrief(kind,ev,fr,b);}
  if(kind==='flat'){if(fr<K) return tx('From left to right, the question is compared with every point.','从左到右，把问题和每个点比一次。');
    if(fr>=n) return tx('All 300 points compared: the circle holds exactly the true 10 nearest, none missed.','300 个点都比较过，圆里正好是真正最近的 10 个，一个不漏。');
    return tx(`${fr} of 300 points compared. The dashed circle holds the 10 nearest so far and shrinks whenever a closer point turns up.`,`已比较 ${fr} 个点（共 300 个）。虚线圆圈住目前最近的 10 个，比到更近的点，圆就缩小。`);}
  if(kind==='kd'){if(fr===0) return tx('The search first goes down the tree to the question’s region, then backs up to check the others.','搜索先沿着树走到问题所在的那一块，再往回查别的块。');
    if(fr>=n) return tx(`The nearest point is found: ${ev.comps} of 300 points compared, ${ev.skips} regions skipped whole.`,`找到了最近的点：300 个点只比较了 ${ev.comps} 个，${ev.skips} 块区域整块跳过。`);
    if(last.k==='skip') return tx('This region’s border is farther from the question than the circle, so it cannot hold a closer point: skipped whole.','这一块的边界比圆还远，里面不会有更近的点，整块跳过。');
    const k=ev.ev.slice(0,fr).filter(x=>x.k==='cmp').length, prev=ev.ev.slice(0,fr-1).reverse().find(x=>x.k==='cmp');
    if(!prev) return tx(`Point ${k} compared: the nearest so far; the dashed circle’s radius is the distance to it.`,`比较第 ${k} 个点：它是目前最近的；虚线圆的半径就是到它的距离。`);
    return last.bi!==prev.bi?tx(`Point ${k} compared: the nearest so far, so the circle shrinks to it.`,`比较第 ${k} 个点：它是目前最近的，圆缩小到它。`):tx(`Point ${k} compared: not closer, so the circle stays.`,`比较第 ${k} 个点：没有更近，圆不变。`);}
  if(kind==='ivf'){const np=ev.np;
    if(fr===0) return tx('The question is first compared with the 16 centres.','问题先和 16 个中心比。');
    if(fr===1) return tx('Compared with all 16 centres (the lines).','和 16 个中心都比过了（连线）。');
    if(fr===2) return np===1?tx('The nearest centre is numbered 1; its cell opens next.','最近的中心编为 1 号，下面打开它的簇。'):tx(`The nearest ${np} centres are numbered by distance; their cells open in that order.`,`最近的 ${np} 个中心按远近编号，下面按这个顺序打开它们的簇。`);
    if(fr<n){const sz=pic().I.lists[last.c].length, got=ev.okAt.filter(x=>x===fr-1).length;
      return tx(`Cell ${last.t+1} opened: its ${sz} points compared${got?`, ${got} of the true 10 nearest kept`:''}.`,`打开第 ${last.t+1} 个簇，比较里面的 ${sz} 个点${got?`，留下 ${got} 个真正的近邻`:''}。`);}
    const pts=ev.comps-P2LIST;
    return tx(`${np===1?'1 cell':`${np} cells`} opened, ${pts} points compared besides the 16 centres: ${nf} of the 10 nearest found${nf<K?'; the red rings sit in cells not opened':''}.`,`打开了 ${np} 个簇，除了 16 个中心，比较 ${pts} 个点；最近的 10 个找到 ${nf} 个${nf<K?'，红圈标出的近邻在没打开的簇里':''}。`);}
  if(kind==='hnsw'){const ef=Math.max(S.ef,K);
    if(fr===0) return tx('The search starts at the entry point, on the top layer: few points, long links.','搜索从入口出发，先在最上面一层走；上层的点少，连线长。');
    if(fr>=n) return tx(`Every candidate in the dashed circle has been expanded, so the search stops: ${ev.comps} points compared, ${nf} of the 10 nearest found.`,`虚线圆里的候选都展开过了，搜索停下。比较了 ${ev.comps} 个点，最近的 10 个找到 ${nf} 个。`);
    if(last.k==='entry') return tx('The entry point. On an upper layer each step moves to a linked point closer to the question.','这是入口。在上层，每一步走到连着的、离问题更近的点。');
    if(last.k==='layer') return tx(`Upper layer ${last.L}: its points and links.`,`上层第 ${last.L} 层：这一层的点和连线。`);
    if(last.k==='look') return tx(`The ${last.seen.length} points linked to the current one on this layer are compared.`,`比较当前点在这一层连着的 ${last.seen.length} 个点。`);
    if(last.k==='hop') return tx('One of them is closer to the question, so the search moves there.','其中一个离问题更近，走过去。');
    if(last.k==='down') return tx('No linked point is closer, so the search goes down a layer from here.','连着的点都不更近了，从这里下到下一层。');
    if(last.k==='bottom') return tx(`The bottom layer: the search now keeps up to ef = ${ef} candidates, inside the dashed circle.`,`到了最底层：搜索留着最多 ef = ${ef} 个候选，虚线圆圈住它们。`);
    const r=ev.ev.slice(0,fr).filter(x=>x.k==='round').length, a=last.add.length, o=last.out.length;
    const c=last.comp.length;
    if(!c) return tx(`Round ${r}: the nearest candidate not yet expanded is expanded; every point linked to it was compared before, so nothing new.`,`第 ${r} 轮：展开最近的、还没展开的候选；它连着的点都比较过了，没有新邻居。`);
    if(!a) return tx(`Round ${r}: the nearest candidate not yet expanded is expanded; ${c===1?'its 1 new neighbour is':`its ${c} new neighbours are`} farther than every candidate kept, so none is kept.`,`第 ${r} 轮：展开最近的、还没展开的候选，比较它的 ${c} 个新邻居，${c===1?'它比留着的候选都远，没有留下':'都比留着的候选远，一个也没留下'}。`);
    return tx(`Round ${r}: the nearest candidate not yet expanded is expanded; ${c===1?'1 new neighbour':`${c} new neighbours`} compared, ${a} kept${o?`, ${o} farthest pushed out`:''}.`,`第 ${r} 轮：展开最近的、还没展开的候选，比较它的 ${c} 个新邻居，留下 ${a} 个${o?`，挤掉最远的 ${o} 个`:''}。`);}
  const c=S.m;
  if(fr<=1) return tx('These are the points where they really are.','这是点原来的位置。');
  if(fr===2) return tx(`PQ learns ${c} centres for x and ${c} for y: they form a ${c} × ${c} grid.`,`PQ 给 x、y 各学 ${c} 个中心，组成 ${c} × ${c} 的网格。`);
  if(fr<n) return tx('Each point moves to its nearest grid crossing: only the crossing’s numbers are stored.','每个点挪到离它最近的网格交点：存下的只是交点的编号。');
  /* the English end fits two lines at 1440, so the caption does not grow on the last frame (R7.9) */
  return tx(`The search ranks the moved points: ${nf} of the 10 nearest found. Points on one crossing are equally far, so it cannot tell them apart.`,`搜索按挪过的位置排：最近的 10 个找到 ${nf} 个。挪到同一个交点的点距离一样，搜索分不出谁更近。`);}
/* a sentence stays on screen at least CAP_MIN seconds (R7.11). Where the search on screen plays faster than that (a large ef, the speed
   快, or the 1.5 s of 应用场景, 自由调参 and the challenge), the frames share one sentence in which only the numbers change (B6); in the
   tour, a slow walk on the upper layers or a slow look at the centres keeps its own sentences before the fast part. Paused, stepped or
   dragged, each frame says what it shows (review polish check r1, N1) */
const CAP_MIN=0.4;
/* frames that show the same sentence form a run: the exact search's running count and PQ's move are one sentence each; every other
   frame says what its own step did (this point, this cell, this round), so it is a run of its own. Runs group into phases */
function capRun(kind,ev,f){
  if(kind==='flat') return f<K?'a':'b';
  if(kind==='pq') return f<=1?'orig':f===2?'grid':'move';
  return 'f'+f;}
const capPhase=(kind,ev,f)=>kind==='ivf'?(f<=2?'ctr':'open'):kind==='hnsw'?(f>0&&['bottom','round'].includes(ev.ev[f-1].k)?'low':'up'):'all';
/* while a search plays: null (each frame its own sentence), 'all' (one sentence for the whole play: always in the 1.5 s of 应用场景,
   自由调参 and the challenge), or in the tour, for IVF and HNSW, the name of a fast phase ('ctr', 'open', 'up', 'low'), which gets one
   sentence while the other phase keeps its own sentences; that needs each phase on screen at least CAP_MIN seconds, else the whole
   play gets one sentence */
function capMode(kind,ev,fr){if(!anim||anim.kind!==kind) return null; if(mode!=='guided') return 'all'; const key=anim.mul+'|'+anim.f0;
  if(anim.capK!==key){const fast=new Set(), span={}; let run=null, ph=null, rs=0;
    const end=()=>{if(run!=null&&rs<CAP_MIN-1e-9) fast.add(ph);};
    for(let f=anim.f0;f<ev.n;f++){const r=capRun(kind,ev,f), p=capPhase(kind,ev,f), h=(f===anim.f0?PIC_LEAD:ev.d[f-1])*anim.mul;
      span[p]=(span[p]||0)+h; if(r!==run||p!==ph){end(); run=r; ph=p; rs=0;} rs+=h;}
    end();
    const split=(kind==='ivf'||kind==='hnsw')&&Object.values(span).every(v=>v>=CAP_MIN-1e-9);
    anim.capK=key; anim.capM=!fast.size?null:split?fast:'all';}
  const m=anim.capM; if(!m||m==='all') return m; const p=capPhase(kind,ev,fr); return m.has(p)?p:null;}
function capBrief(kind,ev,fr,m){const done=t=>ev.ev.slice(0,fr).filter(e=>e.k===t).length, pl=(n,one,many)=>`${n} ${n===1?one:many}`;
  const cmpd=()=>{let c=0; for(let i=0;i<P2N;i++) if(ev.cmpAt[i]<fr) c++; return c;};
  if(kind==='flat') return tx(`From left to right: ${fr} of 300 points compared.`,`从左到右，已比较 ${fr} 个点（共 300 个）。`);
  if(kind==='kd'){const k=done('cmp'), s=done('skip');
    return tx(`${pl(k,'point','points')} compared, ${pl(s,'region','regions')} skipped whole: a region the circle does not reach is not checked.`,`已比较 ${k} 个点，整块跳过 ${s} 块区域：圆碰不到的块不用查。`);}
  if(kind==='ivf'){const np=ev.np, t=done('open'), lists=pic().I.lists; let c=0; for(const e of ev.ev.slice(0,fr)) if(e.k==='open') c+=lists[e.c].length;
    const kept=ev.okAt.filter(x=>x<fr).length;
    if(m==='open') return tx(`${t} of ${pl(np,'cell','cells')} opened, ${pl(c,'point','points')} compared, ${kept} of the true 10 nearest kept.`,`已打开 ${t} 个簇（共 ${np} 个），比较了 ${c} 个点，留下 ${kept} 个真正的近邻。`);
    if(m==='ctr') return tx(`The question is compared with the 16 centres first; then the nearest ${np===1?'cell opens':`${np} cells open in order`}.`,`问题先和 16 个中心比，再按远近打开最近的 ${np} 个簇。`);
    return tx(`The question is compared with the 16 centres, then the nearest ${np===1?'cell opens':`${np} cells open in order`}: ${t} opened, ${pl(c,'point','points')} compared.`,`问题先和 16 个中心比，再按远近打开 ${np} 个簇：已打开 ${t} 个，比较了 ${c} 个点。`);}
  if(kind==='hnsw'){let c=0; for(const e of ev.ev.slice(0,fr)) c+=e.k==='entry'?1:e.k==='look'?e.seen.length:e.k==='round'?e.comp.length:0;
    if(m==='up') return tx(`Upper layers: each step moves to a linked point closer to the question; ${pl(c,'point','points')} compared.`,`上层：每一步走到连着的、离问题更近的点，已比较 ${c} 个点。`);
    if(m==='low'){const st=hnswState(ev,fr);
      return tx(`Bottom layer: ${pl(done('round'),'candidate','candidates')} expanded, ${st.W?st.W.length:1} kept, ${pl(c,'point','points')} compared.`,`最底层：已展开 ${done('round')} 个候选，留着 ${st.W?st.W.length:1} 个，比较了 ${c} 个点。`);}
    return tx(`From the entry the search follows links toward the question, then expands candidates on the bottom layer: ${pl(c,'point','points')} compared.`,`从入口沿连线走向问题，再在最底层一轮轮展开候选：已比较 ${c} 个点。`);}
  const g=S.m;
  return tx(`PQ learns ${g} centres for x and ${g} for y, and each point moves to its nearest grid crossing: only the crossing’s numbers are stored.`,`PQ 给 x、y 各学 ${g} 个中心，每个点挪到离它最近的网格交点：存下的只是交点的编号。`);}
/* which numbers are drawn and which are measured (review polish r1, P11) */
function picNote(kind){
  const meas=tx('Recall@10 and the metric cards are measured: 5,000 vectors of 32 dimensions, averaged over 100 questions.','recall@10 和指标卡是实测：5,000 个 32 维向量，100 个问题取平均。');
  if(kind==='flat') return mode==='guided'?tx('A sketch: 300 points in 2 dimensions. The bars are at real scale: 768-dimensional vectors read at 100 GB/s.','示意：300 个 2 维的点。条形按真实规模算：768 维的向量，每秒读 100 GB。'):tx('A sketch: 300 points in 2 dimensions. ','示意：300 个 2 维的点。')+meas;
  if(kind==='kd') return tx('A sketch: 300 points in 2 dimensions, looking for the nearest one. The bars are measured: 5,000 vectors, averaged over 100 questions.','示意：300 个 2 维的点，只找最近的 1 个。条形是实测：5,000 个向量，100 个问题取平均。');
  if(kind==='ivf') return tx(`A sketch: 300 points in 2 dimensions, 16 cells. Recall@10 and the metric cards are measured: 5,000 vectors of 32 dimensions in ${NLIST} cells, averaged over 100 randomly generated questions.`,`示意：300 个 2 维的点、16 个簇。recall@10 和指标卡是实测：5,000 个 32 维向量分在 ${NLIST} 个簇里，在 100 个随机生成的问题上取平均。`);
  if(kind==='hnsw') return tx(`A sketch: 300 points in 2 dimensions, M = 4. Recall@10 and the metric cards are measured: 5,000 vectors of 32 dimensions, M = ${S.M}, averaged over 100 questions.`,`示意：300 个 2 维的点，M = 4。recall@10 和指标卡是实测：5,000 个 32 维向量，M = ${S.M}，100 个问题取平均。`);
  if(kind==='pq') return tx(`A real PQ code has m pieces of 256 centres each. The picture has only 2 pieces, x and y, so a longer code is drawn as more centres per piece. Recall@10 and the metric cards are measured on IVF-PQ: 5,000 vectors of 32 dimensions, ${S.np} cells probed.`,`真实的 PQ 编码有 m 段，每段 256 个中心。图里只有 x、y 两段，所以用每段多放几个中心来表示更长的编码。recall@10 和指标卡是 IVF-PQ 的实测：5,000 个 32 维向量，查 ${S.np} 个簇。`);
  return '';}
/* the legend lists the marks this picture draws, and only those (R5.7, R5.12); a mark the end of the search no longer shows says so,
   as the legend stays the same while the search plays (R7.9; review polish check r1, N11); the English labels stay short enough that
   the legend keeps its rows at 1440 and the stage still fits the screen (P23) */
function picLegend(kind,hasBad){
  const it=(cls,txt,tip)=>`<span class="lg"${tip?` data-tipx="${esc(tip)}"`:''}><i class="sw ${cls}"></i>${txt}</span>`;
  const Q=it('vix-sw-q',tx('question','问题'),tx('The cross is the question’s vector.','叉号是问题的向量。'));
  const PT=it('vix-sw-pt',tx('stored vector','存着的向量'),kind==='intro'||kind==='cells'?tx('Each dot is one stored vector.','每个点是一个存着的向量。'):tx('A faded dot is a stored vector the search has not compared with the question.','半透明的点是搜索还没比较过的向量。'));
  const CHK=it('vix-sw-chk',tx('compared','比较过'),tx('A solid dot is a stored vector the search compared with the question: it computed their distance.','实心的点是搜索比较过的向量，算过它到问题的距离。'));
  const NN=kind==='intro'?it('vix-sw-nn',tx('its 10 nearest','离它最近的 10 个'),tx('The 10 stored vectors closest to the question.','离问题最近的 10 个存着的向量。'))
    :it('vix-sw-nn',tx('true 10 nearest, until kept','真正最近的 10 个：还没留下（播放时）'),tx('A thin ring: one of the true 10 nearest that the search has not kept yet.','细圈：真正最近的 10 个之一，搜索还没留下它。'));
  const OK=it('vix-sw-ok',tx('kept','留下了'),tx('A green ring: one of the true 10 nearest, and the search keeps it.','绿圈：真正最近的 10 个之一，搜索留下了它。'));
  const BAD=it('vix-sw-bad',tx('missed','漏掉了'),tx('A red ring: one of the true 10 nearest that the search did not return.','红圈：真正最近的 10 个之一，搜索没有找到。'));
  const CTR=it('vix-sw-ctr',tx('cell centre','簇的中心'),tx('The small square is the centre of a cell; every point is stored in the cell of its nearest centre.','小方块是簇的中心；每个点存进离它最近的中心的那个簇。'));
  const BD=it('vix-sw-bd',tx('cell border','簇的边界'),tx('Lines split the plane into the 16 cells: each cell is the region nearest one centre.','线把平面分成 16 个簇，每个簇是离一个中心最近的区域。'));
  if(kind==='intro') return Q+PT+NN;
  if(kind==='cells') return Q+PT+CTR+BD;
  if(kind==='flat') return Q+PT+CHK+it('vix-sw-rad',tx('the 10 nearest so far','目前最近的 10 个'),tx('The dashed circle around the question holds the 10 nearest points compared so far.','问题周围的虚线圆，圈住目前比较过的点里最近的 10 个。'))+NN+OK;
  if(kind==='kd') return Q+PT+CHK+it('vix-sw-cut',tx('cut','切线'),tx('Each line is one cut of the k-d tree; the cuts of the regions the search checks are drawn thicker.','每条线是 k-d 树的一刀；搜索查过的区域，它的切线画得更粗。'))+
    it('vix-sw-skip',tx('skipped region','跳过的区域'),tx('A grey region is skipped whole: its border is farther from the question than the nearest point found so far.','灰色的区域整块跳过，它的边界离问题比目前找到的最近点还远。'))+
    it('vix-sw-best',tx('nearest so far (while playing)','目前最近的点（播放时）'),tx('The ringed point is the nearest found so far; the dashed circle’s radius is its distance.','带圈的点是目前找到的最近的点；虚线圆的半径就是它的距离。'))+
    it('vix-sw-ok',tx('the nearest, found','找到的最近点'),tx('A green ring: the true nearest point, found by the search.','绿圈：真正最近的那个点，搜索找到了它。'));
  if(kind==='ivf') return Q+PT+CHK+CTR+BD+it('vix-sw-cl',tx('question to centre','问题到中心'),tx('Lines from the question to the centres: the search compares the question with every centre first.','问题到各个中心的连线：搜索先把问题和每个中心比一次。'))+
    it('vix-sw-cell',tx('opened cell','打开的簇'),tx('A shaded cell has been opened: the search compares the question with every point in it.','带底色的簇已经打开：搜索把问题和里面的每个点都比一次。'))+NN+OK+(hasBad?BAD:'');
  if(kind==='hnsw') return Q+PT+CHK+it('vix-sw-entry',tx('entry','入口'),tx('Every search starts at the entry point, on the top layer.','每次搜索都从入口出发，入口在最上面一层。'))+
    it('vix-sw-up',tx('upper layer (while playing)','上层的点和连线（播放时）'),tx('An upper layer holds few points, with long links.','上层只有少数几个点，连线长。'))+
    it('vix-sw-hop',tx('upper-layer step','上层走的一步'),tx('Dashed arrows: the walk on the upper layers, from the entry toward the question.','虚线箭头：在上层走的每一步，从入口走向问题。'))+
    it('vix-sw-link',tx('bottom-layer link','最底层的连线'),tx('Thin lines: the links of the bottom layer. Here M = 4, so each point keeps up to 2 × M = 8.','细线：最底层的连线。这里 M = 4，所以每个点最多 2 × M = 8 条。'))+
    it('vix-sw-radh',tx('candidates kept','留着的候选'),tx('The dashed circle holds the candidates the search keeps on the bottom layer, up to ef of them.','虚线圆圈住搜索在最底层留着的候选，最多 ef 个。'))+
    it('vix-sw-exp',tx('expanded','展开过的'),tx('A filled point has been expanded: the search compared all its neighbours.','实心的彩色点已经展开过，它的邻居都比较过了。'))+NN+OK+(hasBad?BAD:'');
  return Q+PT+it('vix-sw-grid',tx('grid of centres','中心的网格'),tx('The lines are the centres of the x piece and of the y piece; every point moves to a crossing.','这些线是 x 段和 y 段的中心；每个点都挪到一个交点上。'))+
    it('vix-sw-trail',tx('where a point moved','点挪动的路线'),tx('From where a point really is to the crossing it was stored as; drawn for the true 10 nearest and the 10 found.','从点原来的位置到它被存成的交点；只画了真正最近的 10 个和找到的 10 个。'))+
    `<span class="lg" data-tipx="${esc(tx('How many points moved to that crossing: the search sees them all at one spot.','有几个点挪到了这个交点：搜索看它们都在同一个位置。'))}"><b class="vix-sw-cnt mono">×3</b> ${tx('points on one crossing','同一个交点上的点数')}</span>`+NN+OK+(hasBad?BAD:'');}
/* part 2's bars at real scale, one row for each number of vectors picked so far (review polish r1, P3) */
function drawScale(){const el=$('#vix-scale',app), Ns=[1e6,1e8,1e9].filter(v=>triedN.has(v));
  const tPos=t=>Math.max(0,Math.min(1,(Math.log10(t)+3)/5)), mPos=b=>Math.max(0,Math.min(1,b/3500e9)), lab=v=>v===1e6?'1M':v===1e8?'100M':'1B';
  const bar=(pos,mark)=>`<span class="vix-kbt"><i style="width:${Math.max(0.6,pos*100).toFixed(1)}%"></i><s style="left:${(mark*100).toFixed(1)}%"></s></span>`;
  const rows=fn=>Ns.map(v=>`<div class="vix-kb${v===realN?' on':''}"><span>${lab(v)}</span>${fn(v)}</div>`).join('');
  el.innerHTML=`<div class="vix-sg"><div class="vix-pt">${tx('How long one question takes to read every vector (log scale)','每个问题读一遍要多久（对数刻度）')}</div>`+
    rows(v=>{const t=v*REAL_D*4/BW; return bar(tPos(t),tPos(0.01))+`<b class="mono">${secs(t)}</b>`;})+
    `<p class="vix-snote">${tx('Dashed line: 10 ms, the few milliseconds wanted. The axis runs from 1 ms to 100 s.','虚线：10 毫秒，几毫秒的目标。横轴从 1 毫秒到 100 秒。')}</p></div>`+
   `<div class="vix-sg"><div class="vix-pt">${tx('Memory for all the vectors','存全部向量要多少内存')}</div>`+
    rows(v=>{const b=v*REAL_D*4; return bar(mPos(b),mPos(512e9))+`<b class="mono">${GB(b)}</b>`;})+
    `<p class="vix-snote">${tx('Dashed line: one server’s 512 GB. The axis runs from 0 to 3,500 GB.','虚线：一台服务器的 512 GB。横轴从 0 到 3,500 GB。')}</p></div>`;}
/* part 3's bars: one for each number of dimensions picked so far, so no answer shows before the pick (review polish r1, P6) */
function drawKDBars(){const el=$('#vix-kdbars',app), k=kdStats(), ds=KDD.filter(d=>triedD.has(d));
  el.innerHTML=`<div class="vix-pt">${tx('Share of the 5,000 vectors a k-d tree compares to find the nearest one (average of 100 questions)','k-d 树为了找到最近的那个，要比较 5,000 个向量里的多少（100 个问题的平均）')}</div>`+
    ds.map(d=>`<div class="vix-kb${d===kdD?' on':''}"><span>${tx(`${d} dimensions`,`${d} 维`)}</span><span class="vix-kbt"><i style="width:${Math.max(0.6,k[d]*100).toFixed(1)}%"></i></span><b class="mono">${esc(kdPc(k[d],true))}</b></div>`).join('')+
    (ds.length<KDD.length?`<p class="vix-snote">${tx('Each number of dimensions you pick adds its bar.','每选一个维数，就加上它的一根条。')}</p>`:'');}

/* ---------- the chart: recall@10 against vectors compared ---------- */
const CX0=44, CX1=340, CY0=12, CY1=190, LX0=Math.log10(50), LX1=Math.log10(5000);
const cxm=v=>CX0+(CX1-CX0)*(Math.log10(Math.max(50,Math.min(5000,v)))-LX0)/(LX1-LX0), cym=r=>CY1-(CY1-CY0)*r;
function curvePts(kind){if(kind==='ivf') return NPROBES.map(np=>stat('ivf',np,0));
  if(kind==='hnsw'||kind==='hnsw16') return EFS.map(ef=>stat('hnsw',kind==='hnsw'?S.M:16,ef));
  return NPROBES.map(np=>stat('pq',S.m,np));}
/* with another M than 16, the HNSW line at M = 16 is drawn too, so the two can be compared on screen (review polish r1, P19) */
const chartCurves=c=>c.curves.reduce((a,k)=>a.concat(k==='hnsw'&&S.M!==16?['hnsw','hnsw16']:[k]),[]);
const CURN=k=>k==='ivf'?tx('IVF (nprobe 1 to 32)','IVF（nprobe 1 到 32）'):k==='hnsw'?tx(`HNSW, M = ${S.M} (ef 10 to 256)`,`HNSW，M = ${S.M}（ef 10 到 256）`):k==='hnsw16'?tx('HNSW, M = 16 (ef 10 to 256)','HNSW，M = 16（ef 10 到 256）'):tx(`IVF-PQ, ${S.m} bytes (nprobe 1 to 32)`,`IVF-PQ，${S.m} 字节（nprobe 1 到 32）`);
function drawChart(){
  const c=ctx(), svg=$('#vix-chart',app), cv=chartCurves(c); let h='';
  for(const r of [0,0.25,0.5,0.75,1]) h+=`<line class="vix-gl" x1="${CX0}" y1="${cym(r).toFixed(1)}" x2="${CX1}" y2="${cym(r).toFixed(1)}"></line><text class="vix-ax" x="${CX0-6}" y="${(cym(r)+4).toFixed(1)}" text-anchor="end">${f(r*100,0)}%</text>`;
  /* the axis text at 12 (review polish check r1, N9): a grid line at each step of the log scale, a number at 100, 1,000 and 5,000 */
  for(const v of [50,100,200,500,1000,2000,5000]) h+=`<line class="vix-gl" x1="${cxm(v).toFixed(1)}" y1="${CY0}" x2="${cxm(v).toFixed(1)}" y2="${CY1}"></line>`+([100,1000,5000].includes(v)?`<text class="vix-ax" x="${cxm(v).toFixed(1)}" y="${CY1+15}" text-anchor="middle">${int(v)}</text>`:'');
  h+=`<text class="vix-ax" x="${(CX0+CX1)/2}" y="${CY1+34}" text-anchor="middle">${tx('vectors compared per question (log scale)','每个问题比较的向量数（对数刻度）')}</text>`;
  h+=`<text class="vix-ax" x="12" y="${(CY0+CY1)/2}" text-anchor="middle" transform="rotate(-90 12 ${(CY0+CY1)/2})">recall@10</text>`;   // at 12px its box still sits inside the chart
  if(mode==='challenge'&&chSubmitted){h+=`<line class="vix-goal" x1="${CX0}" y1="${cym(CH_REC).toFixed(1)}" x2="${CX1}" y2="${cym(CH_REC).toFixed(1)}"></line><line class="vix-goal" x1="${cxm(CH_COMP).toFixed(1)}" y1="${CY0}" x2="${cxm(CH_COMP).toFixed(1)}" y2="${CY1}"></line>`;}
  for(const k of cv){const pts=curvePts(k); h+=`<polyline class="vix-cv vix-k-${k}" points="${pts.map(st=>cxm(st.comps).toFixed(1)+','+cym(st.rec).toFixed(1)).join(' ')}"></polyline>`+pts.map(st=>`<circle class="vix-cd vix-k-${k}" cx="${cxm(st.comps).toFixed(1)}" cy="${cym(st.rec).toFixed(1)}" r="${k==='hnsw16'?2.4:3}"></circle>`).join('');}
  if(cv.length||mode!=='guided') h+=`<circle class="vix-cd vix-k-flat" cx="${cxm(N).toFixed(1)}" cy="${cym(1).toFixed(1)}" r="3.5"></circle>`;
  /* the ring marks the setting; the legend names it, so no label sits on the lines (review polish r1, P18) */
  if(!c.hideNums&&(c.curves.includes(S.ix)||S.ix==='flat')){const st=statOf(S); h+=`<circle class="vix-me" cx="${cxm(st.comps).toFixed(1)}" cy="${cym(st.rec).toFixed(1)}" r="7"></circle>`;}
  svg.innerHTML=h; svg.setAttribute('aria-label',tx('Recall@10 against vectors compared per question, one line per index.','recall@10 与每个问题比较的向量数，每种索引一条线。'));
  $('#vix-c-title',app).textContent=mode==='challenge'&&!chSubmitted?tx('Which settings find the true 10 nearest with the fewest comparisons? (shown after you submit)','哪些设置用最少的比较找到真正最近的 10 个？（提交后显示）'):tx('Which settings find the true 10 nearest with the fewest comparisons? (5,000 vectors, 100 questions)','哪些设置用最少的比较找到真正最近的 10 个？（5,000 个向量，100 个问题）');
  const lg=$('#vix-c-leg',app), lk=A.lang()+cv.join()+S.M+S.m+mode+chSubmitted+(c.hideNums?1:0);
  if(lg._k!==lk){lg._k=lk; const it=(cls,txt,tip)=>`<span class="lg" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i>${txt}</span>`;
    lg.innerHTML=cv.map(k=>it('vix-sw-'+k,esc(CURN(k)),k==='ivf'?tx('Each dot is one nprobe: more cells probed move it up and to the right.','每个点是一个 nprobe：查的簇越多，点越往右上。'):k==='hnsw'?tx(`Each dot is one ef, at the current M = ${S.M}.`,`每个点是一个 ef，M 是现在的 ${S.M}。`):k==='hnsw16'?tx('The same line at M = 16, for comparison: each dot is one ef.','M = 16 时的同一条线，用来对比；每个点是一个 ef。'):tx(`Each dot is one nprobe, with the current ${S.m}-byte code. A comparison with a code counts as one, like a full vector, though it costs far less.`,`每个点是一个 nprobe，编码是现在的 ${S.m} 字节。比较一个编码也算一次，和比较完整向量一样，虽然它便宜得多。`))).join('')+
     ((cv.length||mode!=='guided')?it('vix-sw-flat',tx('exact search','精确搜索'),tx('Exact search: all 5,000 compared, recall 100%.','精确搜索：比较全部 5,000 个，召回率 100%。')):'')+
     (c.hideNums?'':it('vix-sw-me',tx('your setting','你的设置'),tx('The ring marks the setting chosen now.','圆圈标出现在选的设置。')))+
     (mode==='challenge'&&chSubmitted?it('vix-sw-goal',tx('bars: 95% and 1,000','两条线：95% 和 1,000'),tx('Dashed lines: recall@10 of 95% and 1,000 vectors compared. A setting passes above the first and left of the second.','虚线：recall@10 95% 和比较 1,000 个向量。在第一条线以上、第二条线左边的设置才算过关。')):'');}}

/* ---------- metric cards ---------- */
/* the real-scale memory of IVF-PQ takes a 64-byte code at 768 dimensions, the assumption the observation and the summary use too
   (review polish r1, P15) */
const MET=[
 {k:'rec',label:()=>tx('Recall@10','recall@10'),v:st=>pc1(st.rec),sub:()=>tx('true 10 nearest found, 100 questions','真正最近的 10 个找回几成，100 个问题'),tip:()=>D_REC(),pts:true},
 {k:'cmp',label:()=>tx('Vectors compared','比较的向量数'),v:st=>int(st.comps),sub:()=>S.ix==='flat'?tx('per question: all 5,000','每个问题：全部 5,000 个'):S.ix==='ivf'?tx(`per question, the ${NLIST} centres included`,`每个问题，含 ${NLIST} 个中心`):S.ix==='pq'?tx(`per question, with the ${NLIST} centres and the table`,`每个问题，含 ${NLIST} 个中心和距离表`):tx('per question','每个问题'),
  tip:()=>tx('How many vectors (or centres, or codes) the search compares with the question, averaged over 100 questions. Exact search compares all 5,000. In IVF-PQ a code counts as one comparison, though it costs far less; the table takes m × 256 small piece distances, as much work as 256 full comparisons, so it counts as 256.','搜索时问题要和多少个向量（或中心、编码）比较，在 100 个问题上取平均。精确搜索比较全部 5,000 个。IVF-PQ 里比较一个编码也算一次，虽然它便宜得多；距离表要算 m × 256 个小段距离，工作量和 256 次完整比较一样，算 256 次。')},
 {k:'byt',label:()=>tx('Memory per vector','每个向量的内存'),v:()=>BYT(bytesPer(S,D)),sub:()=>tx('32 dimensions in this lab','这个 Lab 的 32 维'),tip:()=>tx('Bytes stored per vector: the float32 vector (4 bytes × 32), plus an 8-byte id in IVF lists, plus HNSW’s links (2 × M links of 4 bytes), or only the PQ code and id.','每个向量存的字节：float32 向量（4 字节 × 32），IVF 列表里再加 8 字节的 id，HNSW 再加连线（2 × M 条，每条 4 字节）；PQ 只存编码和 id。')},
 {k:'big',label:()=>tx('Memory for 1B vectors of 768 dims','存 10 亿个 768 维向量要的内存'),v:()=>GB(REAL_N*bigBytes(S)),sub:()=>S.ix==='pq'?tx(`a ${PQ_REAL}-byte code at 768 dims`,`768 维时编码按 ${PQ_REAL} 字节算`):tx('with the index and settings above','按现在的索引和设置算'),
  tip:()=>S.ix==='pq'?tx(`The memory the same index would need for 1B vectors of 768 dimensions (GB = 10⁹ bytes). For IVF-PQ this lab takes a ${PQ_REAL}-byte code (12 dimensions per piece) and an 8-byte id at 768 dimensions, whatever m is chosen here; recall at 768 dimensions is not the one measured here.`,`同一种索引存 10 亿个 768 维向量要多少内存（GB = 10⁹ 字节）。768 维时，这个 Lab 按 IVF-PQ 每个编码 ${PQ_REAL} 字节（每段 12 维）、id 8 字节算，和这里选的 m 无关；768 维时的召回率也和这里测的不一样。`)
    :tx('The memory the same index, with the same settings, would need for 1B vectors of 768 dimensions (GB = 10⁹ bytes).','同一种索引、同样的设置，存 10 亿个 768 维向量要多少内存（GB = 10⁹ 字节）。')}];
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
  const st=$('#vix-st-title',app), el=$('#vix-status',app), c=ctx(), id=mode==='guided'?STEPS[step].id:'';
  st.textContent=id==='mission'?tx('Your mission: the nearest 10 of 1B vectors','你的任务：10 亿个向量里找最近的 10 个')
    :id==='exact'?tx('Exact search: compare the question with every vector','精确搜索：问题和每个向量都比')
    :id==='trees'?tx('Why trees stop working: a k-d tree in 2, 8 and 32 dimensions','为什么树不管用了：2 维、8 维和 32 维里的 k-d 树')
    :id==='predict'?tx('IVF: the vectors split into cells','IVF：向量分成很多簇')
    :tx(`Searching the lab’s vectors with ${setTxt(S)}`,`搜索这个 Lab 的向量：${setTxt(S)}`);
  if(mode==='challenge'){const done=chDone(), me=done?chAll().find(p=>sameSet(p.s,S)):null; el.className='status'+(done?(me.ok?' ok':' bad'):''); el.textContent=done?(me.ok?tx('Meets both bars','两条线都满足'):tx('Misses a bar','没过线')):tx('Not submitted','还没提交');
    el.dataset.tipx=tx('Whether the setting reaches 95% recall@10 with at most 1,000 vectors compared. It shows once you submit.','这个设置是否达到 recall@10 95%、并且最多比较 1,000 个向量。提交以后才显示。'); el.hidden=false; return;}
  /* the recall badge shows only with the metric cards, from the part that explains recall@10 on (review polish r1, P4) */
  const show=c.panels.includes('metrics'); el.hidden=!show; if(!show){el.textContent=''; return;}
  const s=statOf(S); el.className='status'; el.textContent=tx(`Recall@10 ${pc1(s.rec)}`,`recall@10 ${pc1(s.rec)}`); el.dataset.tipx=tx('Recall@10 of the current setting on the 100 questions.','现在的设置在 100 个问题上的 recall@10。');}
/* the transport: Play, the speed, Replay, Step and the scrubber, one row of one height (review polish r1, P26) */
function drawTransport(){
  const tEl=$('#vix-transport',app), kind=picKind(), on=!!kind, want=A.lang()+'|'+(anim?1:0)+'|'+picSpeed+'|'+on;
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
function scrubTo(v){const kind=picKind(); if(!kind) return; stopAnim(); pendObs=false; const ev=picEventsC(kind);
  if(v>=ev.n){picF=-1; seenPic.add(picKey(kind));} else picF=Math.max(0,v);
  drawPic(); drawTransport(); A.dock.refresh();}
/* the panels of the stage now: the picture and its partner side by side when there are two (review polish r1, P23) */
function panelsShow(){const c=ctx(), figs=$('#vix-figs',app); let k=0;
  ['pic','scale','kd','chart'].forEach(p=>{const el=$('#vix-pan-'+p,app), on=c.panels.includes(p); el.hidden=!on; el.classList.toggle('sec',on&&k>0); if(on) k++;});
  figs.classList.toggle('two',k>1); figs.hidden=!k;
  $('#vix-metrics',app).hidden=!c.panels.includes('metrics');}
/* Show the math: short blocks, each a lead line, the named formula, the numbers of the current setting, the result */
/* 3.2 × 10^12 for a byte count (R6.18: big numbers in a formula as powers of ten) */
const SCI=b=>{const e=Math.floor(Math.log10(b)), m=b/Math.pow(10,e); return `${String(+m.toFixed(4))} \\times 10^{${e}}`;};
function mathHTML(){
  const st=statOf(S), big=bigBytes(S), blk=(lead,tex,num)=>`<div class="vix-mb"><p>${lead}</p>${M(tex)}${num?M(num):''}</div>`;
  let h=blk(tx('Recall@10: the share of the true 10 nearest that the search returned, averaged over the 100 questions. h<sub>q</sub> is how many of question q’s true 10 nearest it returned.','recall@10：真正最近的 10 个里，搜索找回的比例，在 100 个问题上取平均。h<sub>q</sub> 是问题 q 的真正最近的 10 个里，找回了几个。'),
     '\\text{recall@10} = \\frac{1}{100}\\sum_{q} \\frac{h_q}{10}',`= \\frac{${f(st.rec*1000,0)}}{1000} = ${f(st.rec*100,1)}\\%`);
  const cmpTex=S.ix==='flat'?['\\text{compared} = N','= 5000']:S.ix==='ivf'?['\\text{compared} = n_{\\text{cells}} + \\sum_{\\text{probed}} |\\text{list}|',`\\approx ${NLIST} + ${f(st.comps-NLIST,1)} = ${f(st.comps,1)}`]:S.ix==='hnsw'?['\\text{compared} = \\text{vectors compared on the walk}',`\\approx ${f(st.comps,1)}`]:['\\text{compared} = n_{\\text{cells}} + 256 + \\sum_{\\text{probed}} |\\text{list}|',`\\approx ${NLIST} + 256 + ${f(st.comps-NLIST-PQ_K,1)} = ${f(st.comps,1)}`];
  const cmpTexZh=S.ix==='flat'?['\\text{比较数} = N','= 5000']:S.ix==='ivf'?['\\text{比较数} = n_{\\text{簇}} + \\sum_{\\text{打开的簇}} |\\text{列表}|',cmpTex[1]]:S.ix==='hnsw'?['\\text{比较数} = \\text{走的过程中比较过的向量数}',cmpTex[1]]:['\\text{比较数} = n_{\\text{簇}} + 256 + \\sum_{\\text{打开的簇}} |\\text{列表}|',cmpTex[1]];
  h+=blk(tx('Vectors compared per question (average of 100); N is the number of vectors, 5,000. In IVF-PQ each code counts as one comparison, and the table of distances to the 256 centres of every piece, m × 256 small distances, counts as 256.','每个问题比较的向量数（100 个问题的平均）；N 是向量数，5,000。IVF-PQ 里每个编码算一次比较；到每一段 256 个中心的距离表，一共 m × 256 个小段距离，算作 256 次。'),tx(cmpTex[0],cmpTexZh[0]),cmpTex[1]);
  const byT=d=>S.ix==='flat'?['4d',`4 \\times ${d} = ${int(4*d)}`]:S.ix==='ivf'?['4d + 8',`4 \\times ${d} + 8 = ${int(4*d+8)}`]:S.ix==='hnsw'?['4d + 2 \\times M \\times 4',`4 \\times ${d} + 2 \\times ${S.M} \\times 4 = ${int(bytesPer(S,d))}`]:['m + 8',`${S.m} + 8 = ${S.m+8}`];
  const vars=S.ix==='hnsw'?tx(` Here d is the number of dimensions (${D}) and M the links per layer (${S.M}).`,`这里 d 是维数（${D}），M 是每层的连线数（${S.M}）。`):S.ix==='pq'?tx(' Here m is the bytes per code.','这里 m 是每个编码的字节数。'):tx(` Here d is the number of dimensions (${D}).`,`这里 d 是维数（${D}）。`);
  h+=blk(tx('Bytes per vector: float32 is 4 bytes per dimension; IVF lists add an 8-byte id; HNSW adds 2 × M links of 4 bytes; PQ keeps only m code bytes and the id.','每个向量的字节数：float32 每维 4 字节；IVF 列表加 8 字节的 id；HNSW 加 2 × M 条 4 字节的连线；PQ 只存 m 个字节的编码和 id。')+vars,tx(`\\text{bytes per vector} = ${byT(D)[0]}`,`\\text{每个向量的字节数} = ${byT(D)[0]}`),`= ${byT(D)[1]}`);
  /* real scale; for IVF-PQ a 64-byte code at 768 dimensions, the same assumption as the metric card and the observation (P15) */
  const byR=S.ix==='pq'?`${PQ_REAL} + 8 = ${PQ_REAL+8}`:byT(REAL_D)[1];
  h+=`<div class="vix-mb"><p>${S.ix==='pq'?tx(`The same index for N = 1B vectors of 768 dimensions (GB = 10⁹ bytes). At 768 dimensions this lab takes a ${PQ_REAL}-byte code (12 dimensions per piece) and an 8-byte id, whatever m is chosen here. First the bytes per vector, then the total.`,`同一种索引，存 N = 10 亿个 768 维向量（GB = 10⁹ 字节）。768 维时，这个 Lab 按每个编码 ${PQ_REAL} 字节（每段 12 维）、id 8 字节算，和这里选的 m 无关。先算每个向量的字节数，再算总数。`)
      :tx('The same index with the same settings for N = 1B vectors of 768 dimensions (GB = 10⁹ bytes). First the bytes per vector at d = 768, then the total.','同一种索引、同样的设置，存 N = 10 亿个 768 维向量（GB = 10⁹ 字节）。先算 d = 768 时每个向量的字节数，再算总数。')}</p>`+
     M(tx(`\\text{bytes per vector} = ${byR}`,`\\text{每个向量的字节数} = ${byR}`))+
     M(tx('\\text{memory} = N \\times \\text{bytes per vector}','\\text{内存} = N \\times \\text{每个向量的字节数}'))+
     M(tx(`= 10^9 \\times ${int(big)}\\,\\text{bytes} = ${SCI(REAL_N*big)}\\,\\text{bytes} = ${int(REAL_N*big/1e9)}\\,\\text{GB}`,`= 10^9 \\times ${int(big)}\\,\\text{字节} = ${SCI(REAL_N*big)}\\,\\text{字节} = ${int(REAL_N*big/1e9)}\\,\\text{GB}`))+`</div>`;
  if(S.ix==='flat') h+=blk(tx('Exact search reads every vector; at the assumed 100 GB/s of one server (10¹¹ bytes a second), with d = 768:','精确搜索要读每个向量；按假设的一台服务器每秒 100 GB（每秒 10¹¹ 字节），d = 768：'),tx('\\text{time} = \\frac{N \\times 4d}{100\\,\\text{GB/s}}','\\text{时间} = \\frac{N \\times 4d}{100\\,\\text{GB/s}}'),tx(`= \\frac{10^9 \\times 3072\\,\\text{bytes}}{10^{11}\\,\\text{bytes/s}} \\approx ${f(REAL_N*REAL_D*4/BW,1)}\\,\\text{s}`,`= \\frac{10^9 \\times 3072\\,\\text{字节}}{10^{11}\\,\\text{字节/秒}} \\approx ${f(REAL_N*REAL_D*4/BW,1)}\\,\\text{秒}`));
  /* IVF at real scale: many more cells than the 50 measured (moved here from the observation, review polish r1, P10) */
  if(S.ix==='ivf'||S.ix==='pq'){const nl=65536, per=Math.round(REAL_N/nl), cR=nl+S.np*per, x=cR/REAL_N, shp=f(x*100,x<1e-4?3:2)+'%';
    h+=`<div class="vix-mb"><p>${tx(`At real scale this lab assumes 1B vectors in 65,536 cells of about equal size, far more than the ${NLIST} cells measured here, so each cell is small. The comparisons for one question:`,`到真实规模，这个 Lab 假设 10 亿个向量分在 65,536 个大小差不多的簇里，比这里实测的 ${NLIST} 个多得多，每个簇就小得多。一个问题要比较的次数：`)}</p>`+
      M(tx(`\\text{per cell} = 10^9 \\div 65,536 \\approx ${int(per)}`,`\\text{每个簇} = 10^9 \\div 65,536 \\approx ${int(per)}`))+
      M(tx(`\\text{compared} = 65,536 + ${S.np} \\times ${int(per)} = ${int(cR)}`,`\\text{比较数} = 65,536 + ${S.np} \\times ${int(per)} = ${int(cR)}`))+
      `<p>${tx(`Probing ${S.np} ${S.np===1?'cell':'cells'}, that is about ${shp} of the 1B vectors.`,`查 ${S.np} 个簇时，约占 10 亿个向量的 ${shp}。`)}</p></div>`;}
  /* the rule HNSW keeps its candidates by on the bottom layer (moved here from the card, review polish r1, P12) */
  if(S.ix==='hnsw') h+=blk(tx('How HNSW keeps candidates on the bottom layer. W holds at most ef points: the nearest to the question found so far. Each round takes the point of W nearest the question that has not been expanded and computes the question’s distance to each of its neighbours e. While W holds fewer than ef points, e joins W; once W is full, e joins only if it is closer than the farthest point of W, and that farthest point leaves. Once every point of W has been expanded, the search stops.','HNSW 在最底层怎么留候选。W 最多留 ef 个点：到目前为止离问题最近的那些。每一轮从 W 里取出离问题最近、还没展开的点，算问题到它每个邻居 e 的距离。W 还不满 ef 个时，e 直接放进 W；满了以后，e 比 W 里最远的点还近，才放进 W，挤掉那个最远的。W 里的点都展开过，搜索就停。'),
    tx('|W| < \\text{ef} \\;\\to\\; e \\text{ joins } W','|W| < \\text{ef} \\;\\to\\; e \\text{ 放进 } W'),   // review polish check r1, N8
    tx('d(q, e) < \\max_{w \\in W} d(q, w) \\;\\to\\; e \\text{ joins } W','d(q, e) < \\max_{w \\in W} d(q, w) \\;\\to\\; e \\text{ 放进 } W'));
  /* distance and cosine (moved here from the first card, review polish r1, P1) */
  h+=blk(tx('Why ranking by distance and by cosine similarity gives the same order: for vectors of length 1, the squared distance between a question q and a stored vector x is 2 minus twice their cosine, so the higher the cosine, the smaller the distance.','为什么按距离排和按余弦相似度排是同一个顺序：长度为 1 的向量，问题 q 和存着的向量 x 之间距离的平方，等于 2 减去 2 倍的余弦相似度；余弦越大，距离越小。'),
    tx('\\text{dist}(q, x)^2 = 2 - 2\\cos(q, x)','\\text{距离}(q, x)^2 = 2 - 2\\cos(q, x)'));
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
function renderRail(after){const rail=$('#vix-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail);
  if(after) after();   // the reveal, before the stage is drawn (review polish r1, P24)
  render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  panelsShow(); drawTransport(); drawHead();
  if(!$('#vix-pan-pic',app).hidden) drawPic();
  if(!$('#vix-pan-scale',app).hidden) drawScale();
  if(!$('#vix-pan-kd',app).hidden) drawKDBars();
  if(!$('#vix-pan-chart',app).hidden) drawChart();
  if(!$('#vix-metrics',app).hidden) drawMetrics();
  const mb=$('#vix-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();}
function announce(){clearTimeout(annT); if(mode==='guided'&&['mission','exact','predict','trees'].includes(STEPS[step].id)) return; annT=setTimeout(()=>{if(!app) return; const s=statOf(S); A.announce(tx(`${setTxt(S)}: recall@10 ${pc1(s.rec)}.`,`${setTxt(S)}：recall@10 ${pc1(s.rec)}。`));},400);}
/* ---------- the picture's search (R3.9, R3.20): a mechanism replay at 中 by default, which plays only when the picture would
   differ; a search already watched shows its end, and another setting of a watched search plays only the part that changes ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} const was=anim; anim=null; lastTs=0; return was;}
function picKind(){const c=ctx(); return c.panels.includes('pic')&&ANIM_K.includes(c.pic)?c.pic:null;}
/* seconds at 中 from frame f0 to the end: the first frame holds PIC_LEAD, each later frame its event's own time */
function segDur(ev,f0){let s=PIC_LEAD; for(let t=Math.max(0,f0);t<ev.n-1;t++) s+=ev.d[t]; return s;}
/* in the tour each step keeps its own time (times the speed); in Scenarios and the challenge the part that plays takes SB_DUR */
const playMul=(ev,f0)=>mode==='guided'?PIC_SPD[picSpeed]:SB_DUR[picSpeed]/segDur(ev,f0);
/* f0: the frame to start from; null goes on from the frame on screen (Play after a pause), or from the start after the end */
function picStart(f0){stopAnim(); const kind=picKind(); if(!kind){picF=-1; render(); return;}
  const ev=picEventsC(kind); picF=Math.max(0,Math.min(f0==null?(picF<0?0:picF):f0,ev.n-1));
  if(A.reduceMotion()){picF=-1; seenPic.add(picKey(kind)); render(); return;}
  anim={kind,acc:0,f0:picF,mul:playMul(ev,picF)}; lastTs=0; render(); raf=requestAnimationFrame(tick);}
function tick(ts){
  raf=0; if(!anim||!app) return; const ev=picEventsC(anim.kind);
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts; anim.acc+=dt; let moved=false;
  while(picF<ev.n){const hold=(picF===anim.f0?PIC_LEAD:ev.d[picF-1])*anim.mul; if(anim.acc<hold) break; anim.acc-=hold; picF++; moved=true;}
  if(picF>=ev.n){endAnim(); return;}
  if(moved){drawPic(); drawTransport(); A.dock.refresh();}
  raf=requestAnimationFrame(tick);}
function endAnim(){const k=picKey(anim.kind); stopAnim(); picF=-1; seenPic.add(k); render();
  if(pendObs){pendObs=false; revealObs();}}
function setSpeed(v){picSpeed=v; if(anim) anim.mul=playMul(picEventsC(anim.kind),anim.f0); render();}
/* a new picture: play it if it has never been watched, else show its end at once */
function showPic(){const kind=picKind(); if(!kind){stopAnim(); picF=-1; render(); return;}
  if(seenPic.has(picKey(kind))){stopAnim(); picF=-1; render(); return;}
  const ev=picEventsC(kind), kindSeen=[...seenPic].some(k=>k.split('|')[0]===kind);
  picStart(kindSeen?ev.from:0);}

/* ---------- navigation ---------- */
/* `after` (a reveal) runs as soon as the rail holds the new part, before the stage is drawn: drawing may first build an index (the
   M = 16 graph, the PQ codes), and the kit sends a phone to the parts' bar and the picture only within 700 ms of the click
   (review polish r1, P24) */
function enterStep(i,after){
  const was=anim?picKey(anim.kind):null, f0=picF; stageWhole=wholeNow();
  stopAnim(); pendObs=false; barsStop(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  if(s.id==='exact'){triedN.clear(); triedN.add(realN);}
  if(s.id==='trees'){triedD.clear(); triedD.add(kdD);}
  if(s.id==='ivf'){triedNP.clear(); triedNP.add(S.np);}
  if(s.id==='hnsw'){triedEF.clear(); triedEF.add(S.ef);}
  if(s.id==='pq'){triedPQ.clear(); triedPQ.add(S.m);}
  picF=-1; renderRail(after);
  const kind=picKind(); if(was&&kind&&picKey(kind)===was){picF=f0; picStart(null); return;}   // the same search still playing: let it go on
  showPic();}
const railCard=()=>$('#vix-rail .step.current',app)||$('#vix-rail',app);
/* the sticky stage on a wide screen sticks inside the bench's content box: the page can scroll down to stageMaxY with the stage whole */
let stageWhole=false;
const stickyStage=()=>{const st=app&&!PHONE()?$('#vix-stage',app):null; return st&&getComputedStyle(st).position==='sticky'?st:null;};
function stageMaxY(st){const ss=getComputedStyle(st), be=st.parentElement, bs=getComputedStyle(be);
  const lim=be.getBoundingClientRect().bottom-(parseFloat(bs.paddingBottom)||0)-(parseFloat(bs.borderBottomWidth)||0)-(parseFloat(ss.marginBottom)||0);
  return scrollY+lim-(parseFloat(ss.top)||70)-st.offsetHeight;}
const wholeNow=()=>{const st=stickyStage(); return !!st&&scrollY<=stageMaxY(st)+1;};
/* the kit's rule on a wide screen (an element already in view stays put, else its top comes up under the top bar), with two changes for
   the sticky stage: the page moves no further than the whole stage still fits beside the rail (maxY), and an element in view lets the
   page move down as far as the rest of the stage needs (review polish r1, P23, P28) */
function revealWith(el,st,maxY){
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:0)+12;
  const r=el.getBoundingClientRect(), sr=st.getBoundingClientRect(), want=parseFloat(getComputedStyle(st).top)||70, shown=el.getClientRects().length>0;
  let y=scrollY;
  if(shown&&!(r.top>=top&&r.top<=innerHeight-160)) y=scrollY+r.top-top;
  else if(sr.bottom>innerHeight-8) y=scrollY+Math.min(sr.bottom-innerHeight+8,sr.top-want);
  y=Math.max(0,Math.min(y,maxY)); if(Math.abs(y-scrollY)>2) window.scrollTo({top:y,behavior:A.reduceMotion()?'auto':'smooth'});}
/* a new part's card: on a phone the kit's reveal (the parts' bar and the picture, R3.7); on a wide screen revealWith, once the stage is
   drawn (the microtask runs after the click's work, before the page is painted). A stage that was whole stays whole: when the new part
   makes it taller or the rail shorter, the page moves up at once (the kit's reveal alone pushed it off the end of the bench for a moment) */
function revealStep(){
  if(PHONE()){A.ui.reveal(railCard()); return;}
  queueMicrotask(()=>{if(!app) return; const c=railCard(), st=stickyStage(); if(!c) return;
    if(!st){A.ui.reveal(c); return;}
    let maxY=stageMaxY(st);
    if(stageWhole&&scrollY>maxY+1){window.scrollTo({top:Math.max(0,maxY),behavior:'auto'}); maxY=stageMaxY(st);}
    revealWith(c,st,maxY);});}
/* a new mode: the rail, the same way. In 自由调参 (the kit's 'free' step of the sandbox) the rail is hidden and the settings panel
   takes its place, as in the kit's reveal; a hidden rail measured as if at the top of the screen moved the page up (review polish r1, P23) */
function modePart(){const rail=$('#vix-rail',app); if(!rail||rail.getClientRects().length) return rail;
  const b=rail.closest('.bench.flowed'), c=b&&b.querySelector(':scope>.controls'); return c&&c.getClientRects().length?c:rail;}
const freeStep=()=>{const b=app&&$('.bench',app); return !!b&&b.dataset.flow==='free';};
/* once the stage is drawn for the new mode (the microtask runs after the click's work): measured earlier, the settings panel a phone
   goes to in 自由调参 was later pushed down by the stage growing, and the page stopped in the middle of the stage */
function revealMode(){
  queueMicrotask(()=>{if(!app) return; const rail=$('#vix-rail',app), st=stickyStage();
    if(PHONE()||!st){A.ui.reveal(rail); return;} revealWith(modePart(),st,stageMaxY(st));});}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  const dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#vix-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#vix-insight',app)); else revealBox($('#vix-ch-res',app));});
const PHONE=()=>innerWidth<=760;
/* on a phone the stage sits above the card, so a pick that starts the picture's search brings the stage to the top while it plays,
   and the observation once it ends (R7.12; review polish r1, P24); on a wider screen the stage is sticky and stays in view */
function revealStage(){requestAnimationFrame(()=>{if(!app) return; const st=$('#vix-stage',app), bar=document.querySelector('.topbar'); if(!st) return;
  const top=(bar?bar.offsetHeight:56)+8, r=st.getBoundingClientRect(); if(Math.abs(r.top-top)<8) return;
  window.scrollTo({top:Math.max(0,scrollY+r.top-top),behavior:A.reduceMotion()?'auto':'smooth'});});}
function afterPick(started){barsStop(); if(started!==false&&anim&&PHONE()){pendObs=true; revealStage();} else if(started===false) revealBars(); else revealObs();}
/* exact search and the k-d tree on a phone: a pick adds a row of bars in the stage, above the card. The bars come up under the top bar and stay about a
   second, then the page goes on to the observation (R3.10; review polish check r1, N10) */
let barsT=0;
function barsStop(){clearTimeout(barsT); barsT=0;}
function revealBars(){barsStop(); const pan=app&&($('#vix-pan-scale:not([hidden])',app)||$('#vix-pan-kd:not([hidden])',app));
  if(!PHONE()||!pan||mode!=='guided'){revealObs(); return;}
  requestAnimationFrame(()=>{if(!app) return; const bar=document.querySelector('.topbar'), top=(bar?bar.offsetHeight:56)+8, r=pan.getBoundingClientRect();
    if(r.top<top||r.bottom>innerHeight-8) window.scrollTo({top:Math.max(0,scrollY+r.top-top),behavior:A.reduceMotion()?'auto':'smooth'});
    barsT=setTimeout(()=>{barsT=0; if(app) revealObs();},A.reduceMotion()?400:1000);});}
/* 自由调参 on a phone: the kit brings the settings panel to the top, so the picture shows its end there instead of playing off screen
   (R7.12; review polish r1, P24); 重放 plays it */
const stillHere=()=>PHONE()&&freeStep();
function setMode(m,after){
  if(m===mode){if(anim&&stillHere()){stopAnim(); picF=-1; render();} if(after) after(); return;} stopAnim(); pendObs=false; barsStop();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH_INIT};}
  if(m==='sandbox'&&was==='guided') S={...DEFAULT};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step,after); return;}
  picF=-1; renderRail(after); if(stillHere()) render(); else showPic();}
function applyPreset(i){const p=PRESETS[i]; stopAnim(); Object.assign(S,p.p); byLearner(()=>{picF=-1; render(); showPic();}); afterPick();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=(s=S)=>[s.ix,s.np,s.M,s.ef,s.m].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null; const o={ix:p[0],np:+p[1],M:+p[2],ef:+p[3],m:+p[4]};
  if(!IXN[o.ix]||!NPROBES.includes(o.np)||!MS.includes(o.M)||!EFS.includes(o.ef)||!PQMS.includes(o.m)) return null; return o;}
function act(a,btn){
  if(a==='play'){if(anim){stopAnim(); render(); return;} picStart(null); return;}
  if(a==='replay'){picStart(0); return;}
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
function setKey(k,v,still){
  if(k==='rn'){realN=+v; triedN.add(realN); byLearner(render); return;}
  if(k==='kd'){kdD=+v; triedD.add(kdD); byLearner(render); return;}
  if(k==='ix'){S.ix=v;} else S[k]=+v;
  if(k==='np') triedNP.add(S.np); if(k==='ef') triedEF.add(S.ef); if(k==='m') triedPQ.add(S.m);
  if(mode==='challenge'){if(!chReveal) chSubmitted=false; const c=$('#vix-ch-ctl',app); if(c&&k==='ix'){c.innerHTML=chCtlHTML();}}
  /* on a phone the reader stays where the control is: in the challenge at the task card, in Settings (still) at the control the kit
     keeps under the finger (R3.2). The picture then shows its end instead of playing off screen (R7.12; review polish r1, P24); the dock
     names the result, and 重放 plays it */
  byLearner(()=>{picF=-1; render(); if(!(still||mode==='challenge'&&PHONE())) showPic();});}
function showMe(p){
  if(p.rn!=null){setKey('rn',p.rn); afterPick(false); return;}
  if(p.kd!=null){setKey('kd',p.kd); afterPick(false); return;}
  for(const k of ['np','ef','m']) if(p[k]!=null){setKey(k,p[k]); afterPick(); return;}}

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
  if(k==='prod') return tx('Click to jump to “In production” below the lab.','点击跳到页面下方的“生产中怎么做”。');
  if(!CTLN[k]) return '';
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k){
  if(!app) return;
  const pan=(id,mark)=>{const el=$('#vix-pan-'+id,app); if(el&&el.hidden){el.hidden=false; render();} A.tip.hide(); const m=mark&&$(mark,app);
    if(el){(m||el).scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); if(m) A.ui.flash(m);}};
  if(k==='pic'){pan('pic','#vix-pic'); return;}
  if(k==='chart'){pan('chart','#vix-chart'); return;}
  if(k==='controls'){const c=$('#vix-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='prod'){const c=$('#vix-prod-card',app); if(c){A.tip.hide(); A.ui.jumpToEl(c);} return;}
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
/* a touch or a wheel turn during the search means the reader is moving on their own: no scroll to the observation at its end */
const dropPend=()=>{pendObs=false;};
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    pendObs=false;
    if(!d.jump&&b.closest('#vix-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#vix-rail');
      if(k==='speed'){setSpeed(d.v); return;}
      /* the number of vectors and the number of dimensions change the bars, not the picture: a search playing goes on */
      const same=k==='rn'||k==='kd'; if(!same) stopAnim(); setKey(k,d.v,!inRail&&PHONE());
      if(mode!=='challenge'&&inRail) afterPick(!same); return;}
    if(d.go!=null){enterStep(+d.go,revealStep); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1,revealStep); return;}
    if(d.nav==='back'){enterStep(step-1,revealStep); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#vix-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo,revealMode); return;}
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
    put(seenPic,o.seenPic,k=>typeof k==='string'&&/^(flat|kd|ivf|hnsw|pq)\|\d*$/.test(k));
    if(Object.keys(PIC_SPD).includes(o.picSpeed)) picSpeed=o.picSpeed; drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen); picF=-1;},
  mount(el,tok){
    stopAnim(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#vix-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m,revealMode); keepStage();},token});
    bind(app); renderDrill();
    window.addEventListener('touchstart',dropPend,{passive:true}); window.addEventListener('wheel',dropPend,{passive:true});
    renderRail(); showPic();},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); pendObs=false; window.removeEventListener('touchstart',dropPend); window.removeEventListener('wheel',dropPend); root=null; app=null;},
  onLang(){if(root){const dm=$('#vix-math',app), open=dm&&dm.open, a=anim, f0=picF; stopAnim(); this.mount(root,null); if(open) $('#vix-math',app).open=true; if(a){picF=f0; picStart(null);}}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#vix-pan-pic:not([hidden])',app)||$('#vix-pan-chart',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#vix-stage',app),
    render({status,vis,read}){if(!app) return; const c=ctx(), st=statOf(S), id=mode==='guided'?STEPS[step].id:'';
      if(mode==='challenge'){const done=chDone(), me=done?chAll().find(p=>sameSet(p.s,S)):null; status.className='status'+(done?(me.ok?' ok':' bad'):''); status.textContent=done?(me.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):tx('Not submitted','未提交');
        vis.innerHTML=''; read.textContent=done?tx(`${setTxt(S)}: recall@10 ${pc1(st.rec)}, ${int(st.comps)} compared`,`${setTxt(S)}：recall@10 ${pc1(st.rec)}，比较 ${int(st.comps)} 个`):tx(`${setTxt(S)}: ${int(bytesPer(S,D))} bytes per vector`,`${setTxt(S)}：每个向量 ${int(bytesPer(S,D))} 字节`); return;}
      /* the mission and exact search show no measured recall yet (review polish r1, P4) */
      if(id==='mission'){status.className='status'; status.textContent=tx('Mission','任务'); vis.innerHTML=''; read.textContent=tx('1B vectors: the nearest 10 within a few milliseconds','10 亿个向量，几毫秒内找出最近的 10 个'); return;}
      if(id==='exact'){const by=realN*REAL_D*4; status.className='status'; status.textContent=tx('Exact search','精确搜索'); vis.innerHTML=''; read.textContent=tx(`${NBV(realN)}: ${secs(by/BW)} to read once, ${GB(by)}`,`${NBV(realN)}：读一遍 ${secs(by/BW)}，${GB(by)}`); return;}
      if(id==='predict'){status.className='status'; status.textContent=tx('IVF cells','IVF 的簇'); vis.innerHTML=''; read.textContent=tx('1,000 cells, one probed: what share is compared? Are all the nearest found?','1,000 个簇，只查 1 个：比多少？找得全吗？'); return;}   // no answer before the prediction (R3.8; review polish check r1, P7)
      if(c.pic==='kd'){status.className='status'; status.textContent=tx('k-d tree','k-d 树'); vis.innerHTML=''; read.textContent=tx(`In ${kdD} dimensions the tree compares ${kdPc(kdStats()[kdD])} of the vectors`,`${kdD} 维时，树要比较 5,000 个向量里的 ${kdPc(kdStats()[kdD])}`); return;}
      vis.innerHTML=`<div class="vix-dk"><i style="width:${(st.rec*100).toFixed(1)}%"></i></div>`;
      status.className='status'+(st.rec>=0.95?' ok':''); status.textContent=tx(`Recall@10 ${pc1(st.rec)}`,`recall@10 ${pc1(st.rec)}`);
      read.textContent=tx(`${setTxt(S)}: ${int(st.comps)} vectors compared per question`,`${setTxt(S)}：每个问题比较 ${int(st.comps)} 个向量`);}}   // recall@10 is on the status, left (O6 of review b10r1)
});
/* read-only hook for tests/vector-index_numbers.py and tests/vector-index_flow.py */
window.__vix={state:()=>({mode,step,ix:S.ix,np:S.np,M:S.M,ef:S.ef,m:S.m,pred,realN,kdD,chSubmitted,chReveal,picF,anim:!!anim,lang:LG()}),
  seen:()=>[...seenPic], stat:(ix,a,b)=>{const s=stat(ix,a,b); return {rec:s.rec,comps:s.comps};}, gt:()=>base().GT, kd:()=>kdStats(),
  pic:()=>{const p=pic(); return {q:p.q,gt:p.gt,lists:p.I.lists.map(l=>l.slice()),Cn:Array.from(p.I.Cn),entry:p.G.entry,top:p.G.top,level:Array.from(p.G.level)};},
  /* one search of the picture with these settings: its events, what it returns and how many points it compares */
  picEv:(kind,np,ef,m)=>{const s0={...S}; S.np=np; S.ef=ef; if(m) S.m=m; const e=picEvents(kind); Object.assign(S,s0); return {n:e.n,from:e.from,found:e.found,comps:e.comps,kinds:e.ev.map(x=>x.k)};},
  picPQ, picDur:f0=>{const k=picKind(); if(!k) return 0; return segDur(picEventsC(k),f0||0)*PIC_SPD[picSpeed];},
  ivfLists:()=>ivf().lists.map(l=>l.slice()), hnswLinks:M0=>hnsw(M0).links[0].map(a=>a.slice()), hnswLevels:M0=>Array.from(hnsw(M0).level), pqCodes:m=>Array.from(pq(m).codes.slice(0,200)),
  data:(i)=>Array.from(base().B.X.subarray(i*D,i*D+D)), drill:makeDrill, bytesPer, chAll:()=>chAll().map(p=>({s:p.s,rec:p.rec,comps:p.comps,bytes:p.bytes,ok:p.ok})),
  consts:{SEED,D,N,NQ,NCL,NOISE,SC,K,NLIST,NTRAIN,KM_IT,PQ_K,PQ_IT,PQ_TRAIN,EFC,NPROBES,EFS,MS,PQMS,KDD,P2N,P2Q,P2CL,P2NOISE,P2LIST,P2M,P2EFC,REAL_D,REAL_N,BW,CH_REC,CH_COMP,EF_FREE,PQ_REAL}};
})();
