/* data-pipeline (world training): What gets thrown away before pretraining?
   Fidelity: Simulated (header: Toy run). A seeded corpus of 1,000 made-up web pages, each with the numbers the real rules read; the
   rules and their thresholds are the ones in datatrove's FineWeb example (pinned 1977fbb, see sources/b8src/data-pipeline/README.md):
   - extraction (Trafilatura): a page without main text is dropped;
   - language ID: English score below 0.65 is dropped;
   - quality, in the order the code runs them: duplicated lines over 30% (Gopher repetition), under 50 words, "#" over 0.1 per word or
     ellipses over 0.1 per word (two separate checks; a page carries the larger of the two ratios, so one test is exact), under 2 of the 8 stop words (Gopher quality), under 12% of lines ending in punctuation (FineWeb);
   - dedup: exact copies by a hash of the text, near copies by MinHash with b bands of r rows: a pair of Jaccard similarity s is
     caught with P(s) = 1 − (1 − s^r)^b. Each copy is compared with its original only; one uniform number per copy decides (caught
     iff u < P), so a stricter setting only ever lets more copies through.
   How many pages of each kind the corpus holds is this lab's assumption (CORPORA). Truth is known per page, so the picture can mark
   in red what a stage got wrong: a good page thrown away, or junk or a near copy (s ≥ 0.8) kept.
   The whole run is an animation of 4 phases (one per stage), Fast by default (about 10 s, R3.9). R3.20: a picture that was watched to
   the end is shown at its end when it comes back; a new picture replays from the first stage that changed; "Replay" replays all.
   The mix: four domains, tokens on hand assumed, a 15T-token target (Llama 3.1's "~15 trillion tokens"); epochs = share × target / on hand.
   Words (EN / ZH): page / 网页 (a document), stage / 环节, kept / 留下, thrown away / 扔掉, near copy / 几乎一样的副本, part / 部分.
   window.__dl is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='data-pipeline';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine: the corpus
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const N=1000;
/* page kinds: junk the pipeline should drop, good pages it should keep, and copies of good pages */
const JUNK=new Set(['nomain','foreign','mixed','rep','short','sym','kw','menu','seo','gen']);
const GOODU=new Set(['enlow','uniq','alike','code']);
const COPY=new Set(['exact','near','border','cexact','cnear']);
const CORPORA={
 web:{seed:11,n:{nomain:150,foreign:330,mixed:20,enlow:10,rep:35,short:55,sym:20,kw:30,menu:40,seo:6,uniq:182,alike:20,exact:45,near:45,border:12}},
 foreign:{seed:23,n:{nomain:150,foreign:640,mixed:30,enlow:5,rep:10,short:15,sym:5,kw:10,menu:10,seo:2,uniq:91,alike:5,exact:12,near:12,border:3}},
 code:{seed:37,n:{code:600,gen:60,cexact:220,cnear:120}}
};
const LANGS=['es','de','fr','ru','zh','ja','pt','it'];
const LANGN={es:{en:'Spanish',zh:'西班牙文'},de:{en:'German',zh:'德文'},fr:{en:'French',zh:'法文'},ru:{en:'Russian',zh:'俄文'},zh:{en:'Chinese',zh:'中文'},ja:{en:'Japanese',zh:'日文'},pt:{en:'Portuguese',zh:'葡萄牙文'},it:{en:'Italian',zh:'意大利文'}};
const memo={};
function corpus(key){
  if(memo[key]) return memo[key];
  const C=CORPORA[key], r=mulberry(C.seed), U=(a,b)=>a+(b-a)*r(), I=(a,b)=>Math.floor(a+(b-a+1)*r()), F=(v,d)=>Math.round(v*Math.pow(10,d))/Math.pow(10,d);
  const docs=[];
  const base=k=>({kind:k,noMain:false,lang:'en',en:F(U(0.72,0.99),2),words:I(250,1600),rep:F(U(0,0.12),2),sym:F(U(0,0.03),3),stop:I(5,8),punct:F(U(0.45,0.95),2),dupOf:-1,sim:0,u:0});
  const add=(k,f)=>{for(let i=0;i<(C.n[k]||0);i++){const d=base(k); if(f) f(d); docs.push(d);}};
  add('nomain',d=>{d.noMain=true; d.words=0;});
  add('foreign',d=>{d.lang=LANGS[I(0,LANGS.length-1)]; d.en=F(U(0,0.25),2); d.stop=0; d.words=I(200,1200);});
  add('mixed',d=>{d.lang=LANGS[I(0,LANGS.length-1)]; d.en=F(U(0.3,0.8),2); d.stop=I(0,3); d.words=I(100,800);});
  add('enlow',d=>{d.en=F(U(0.45,0.64),2); d.words=I(150,600);});
  add('rep',d=>{d.en=F(U(0.8,0.99),2); d.words=I(80,400); d.rep=F(U(0.35,0.8),2); d.stop=I(3,6); d.punct=r()<0.5?F(U(0.03,0.11),2):F(U(0.2,0.5),2);});
  add('short',d=>{d.en=F(U(0.7,0.97),2); d.words=I(8,49); d.stop=I(0,4); d.punct=F(U(0.3,0.9),2);});
  add('sym',d=>{d.sym=F(U(0.12,0.45),3); d.words=I(60,300); d.stop=I(1,4); d.punct=F(U(0.05,0.4),2);});
  add('kw',d=>{d.stop=I(0,1); d.words=I(60,400); d.punct=F(U(0,0.3),2); d.sym=F(U(0,0.05),3);});
  add('menu',d=>{d.punct=F(U(0,0.11),2); d.stop=I(2,5); d.words=I(60,300); d.rep=F(U(0.05,0.25),2);});
  add('seo',d=>{d.en=F(U(0.85,0.97),2);});
  add('code',d=>{d.en=F(U(0.3,0.9),2); d.words=I(50,1500); d.rep=F(U(0,0.35),2); d.sym=F(U(0,0.25),3); d.stop=I(0,4); d.punct=F(U(0,0.35),2);});
  add('gen',d=>{d.en=F(U(0.1,0.6),2); d.words=I(500,5000); d.rep=F(U(0.3,0.9),2); d.sym=F(U(0,0.4),3); d.stop=0; d.punct=F(U(0,0.05),2);});
  add('uniq'); add('alike');
  const orig=docs.map((d,i)=>i).filter(i=>docs[i].kind==='uniq'||docs[i].kind==='code');
  const cp=(k,s0,s1)=>{for(let i=0;i<(C.n[k]||0);i++){const o=orig[I(0,orig.length-1)], d={...docs[o],kind:k,dupOf:o}; d.sim=s0>=1?1:F(U(s0,s1),2); d.u=r(); docs.push(d);}};
  cp('exact',1,1); cp('near',0.8,0.98); cp('border',0.65,0.79); cp('cexact',1,1); cp('cnear',0.8,0.98);
  /* "alike": a heavily edited version of a good page (similarity 0.45–0.6, about two thirds of the 5-grams shared); FineWeb's setting
     should keep it as a page of its own, so catching it is a wrong call */
  docs.forEach(d=>{if(d.kind==='alike'){d.dupOf=orig[I(0,orig.length-1)]; d.sim=F(U(0.45,0.6),2); d.u=r();}});
  /* shuffle into crawl order, then point dupOf at the new positions */
  const ord=docs.map((d,i)=>i); for(let i=ord.length-1;i>0;i--){const j=Math.floor(r()*(i+1)); [ord[i],ord[j]]=[ord[j],ord[i]];}
  const pos=new Array(docs.length); ord.forEach((o,p)=>{pos[o]=p;});
  const out=ord.map(o=>({...docs[o]})); out.forEach(d=>{if(d.dupOf>=0) d.dupOf=pos[d.dupOf];});
  out.forEach((d,i)=>{d.i=i;});
  return memo[key]={key,docs:out};
}
/* the five quality rules, in the order datatrove's FineWeb example runs them */
const RULES=[
 {k:'rep',fail:d=>d.rep>0.3,v:d=>pct(d.rep),need:{en:'≤ 30%',zh:'≤ 30%'},name:{en:'Too many repeated lines',zh:'重复的行太多'},item:{en:'repeated lines',zh:'重复的行'}},
 {k:'short',fail:d=>d.words<50,v:d=>fint(d.words),need:{en:'≥ 50',zh:'≥ 50'},name:{en:'Too short',zh:'太短'},item:{en:'words',zh:'词数'}},
 {k:'sym',fail:d=>d.sym>0.1,v:d=>fixd(d.sym,2),need:{en:'≤ 0.10',zh:'≤ 0.10'},name:{en:'Too many symbols',zh:'符号太多'},item:{en:'“#” or “…” per word (the larger)',zh:'每个词摊到的“#”或“…”（多的那种）'}},
 {k:'stop',fail:d=>d.stop<2,v:d=>String(d.stop),need:{en:'≥ 2',zh:'≥ 2'},name:{en:'Too few stop words',zh:'停用词太少'},item:{en:'stop words present',zh:'出现的停用词'}},
 {k:'punct',fail:d=>d.punct<0.12,v:d=>pct(d.punct),need:{en:'≥ 12%',zh:'≥ 12%'},name:{en:'Too few lines end in punctuation',zh:'以标点结尾的行太少'},item:{en:'lines ending in punctuation',zh:'以标点结尾的行'}}
];
const RK=RULES.map(x=>x.k);
const PofS=(s,b,r)=>1-Math.pow(1-Math.pow(s,r),b);
const thrOf=(b,r)=>Math.pow(1/b,1/r);
const BASE={corpus:'web',thr:0.65,rules:{rep:1,short:1,sym:1,stop:1,punct:1},exact:1,mh:1,b:14,r:8,mix:'prop'};
const cloneS=s=>({...s,rules:{...s.rules}});
/* the fate of every page under the settings: st = the stage that drops it (1 extract, 2 language, 3 quality, 4 dedup), 5 = kept */
function fatesOf(S){
  const C=corpus(S.corpus), D=C.docs, n=D.length, st=new Array(n), why=new Array(n), red=new Array(n).fill(false);
  for(let i=0;i<n;i++){const d=D[i];
    if(d.noMain){st[i]=1; why[i]='nomain'; continue;}
    if(d.en<S.thr-1e-9){st[i]=2; why[i]='lang'; continue;}
    let w=null; for(const R of RULES){if(S.rules[R.k]&&R.fail(d)){w=R.k; break;}}
    if(w){st[i]=3; why[i]=w; continue;}
    st[i]=5; why[i]='kept';}
  for(let i=0;i<n;i++){const d=D[i]; if(st[i]!==5||d.dupOf<0||st[d.dupOf]<4) continue;
    if(S.exact&&d.sim>=1){st[i]=4; why[i]='exact';}
    else if(S.mh&&d.u<PofS(d.sim,S.b,S.r)){st[i]=4; why[i]='mh';}}
  for(let i=0;i<n;i++){const d=D[i];
    if(st[i]<5) red[i]=GOODU.has(d.kind);
    else red[i]=JUNK.has(d.kind)||(COPY.has(d.kind)&&d.sim>=0.8&&st[d.dupOf]===5);}
  const rm=[0,0,0,0,0], byRule={}; RK.forEach(k=>{byRule[k]=0;});
  for(let i=0;i<n;i++){if(st[i]<5) rm[st[i]]++; if(st[i]===3) byRule[why[i]]++;}
  const kept=n-rm[1]-rm[2]-rm[3]-rm[4];
  let wAll=0, wKept=0, dupLeft=0, redKept=0, redBin=0, nearLeft=0;
  for(let i=0;i<n;i++){const d=D[i]; wAll+=d.words; if(st[i]===5) wKept+=d.words;
    if(st[i]===5&&COPY.has(d.kind)&&d.sim>=0.8&&st[d.dupOf]===5){dupLeft++; if(d.sim<1) nearLeft++;}
    if(red[i]){if(st[i]===5) redKept++; else redBin++;}}
  return {st,why,red,rm,byRule,kept,wShare:wAll?wKept/wAll:0,dupLeft,nearLeft,redKept,redBin,
    sig:st.join('')+'|'+red.map(x=>x?1:0).join('')};
}
/* pairs that reach the dedup stage: a copy (or a "bit alike" page) whose original also got there */
function pairsOf(S){
  const C=corpus(S.corpus), f=fatesOf({...S,exact:0,mh:0});
  return C.docs.filter(d=>d.dupOf>=0&&f.st[d.i]===5&&f.st[d.dupOf]===5).map(d=>({i:d.i,s:d.sim,u:d.u,kind:d.kind}));
}
/* the challenge: MinHash settings for a batch of pages */
const CH={Bs:[14,20,34,40,50],Rs:[6,8,11,12],hi:0.8,lo:0.5,needHi:0.95,maxLo:0.02,start:{b:14,r:8}};
function chEval(b,r){const ph=PofS(CH.hi,b,r), pl=PofS(CH.lo,b,r); return {b,r,h:b*r,ph,pl,okHi:ph>=CH.needHi-1e-12,okLo:pl<=CH.maxLo+1e-12,ok:ph>=CH.needHi-1e-12&&pl<=CH.maxLo+1e-12};}
function chAll(){const a=[]; CH.Bs.forEach(b=>CH.Rs.forEach(r=>a.push(chEval(b,r)))); return a;}
function chBest(){return chAll().filter(e=>e.ok).sort((x,y)=>x.h-y.h)[0];}
/* the result after Submit (owner's review b8, R1): the settings that pass, cheapest first; the one change (only b or only r) that
   meets the goal with the fewest hashes, and for a setting that already passes only a change that saves hashes; the main gap of a
   setting that misses: 0.8 missed (the curve too far right), 0.5 caught (too far left), or both (not steep enough) */
function chPass(){return chAll().filter(e=>e.ok).sort((x,y)=>x.h-y.h);}
function chOneChange(me){
  const c=[]; CH.Bs.forEach(b=>{if(b!==me.b) c.push({k:'b',from:me.b,v:b,e:chEval(b,me.r)});}); CH.Rs.forEach(r=>{if(r!==me.r) c.push({k:'r',from:me.r,v:r,e:chEval(me.b,r)});});
  return c.filter(x=>x.e.ok&&(!me.ok||x.e.h<me.h)).sort((x,y)=>x.e.h-y.e.h)[0]||null;}
const chGap=me=>me.ok?null:(!me.okHi&&!me.okLo?'steep':(!me.okHi?'right':'left'));
/* the mix: four domains, tokens on hand (this lab's assumption), a 15T-token run */
const TARGET=15e12;
const DOM=[{k:'web',t:{en:'English web pages',zh:'英文网页'},have:12e12},{k:'multi',t:{en:'Web pages in other languages',zh:'其他语言的网页'},have:4e12},
  {k:'code',t:{en:'Code',zh:'代码'},have:1.5e12},{k:'math',t:{en:'Math',zh:'数学'},have:0.3e12}];
const HAVE=DOM.reduce((s,d)=>s+d.have,0);
const MIXES=[
 {v:'prop',t:{en:'In proportion to what you have (initial setup)',zh:'按手里的数据量分（初始设置）'},d:{en:'Every domain gets the same share of its own data',zh:'每一类都用掉自己同样比例的数据'},sh:DOM.map(d=>d.have/HAVE)},
 {v:'more',t:{en:'More code and math',zh:'多放一点代码和数学'},d:{en:'Code 15%, math 5%, the web the rest',zh:'代码 15%，数学 5%，其余是网页'},sh:[0.65,0.15,0.15,0.05]},
 {v:'math15',t:{en:'Math 15%',zh:'数学占 15%'},d:{en:'Code 15%, math 15%, the web the rest',zh:'代码 15%，数学 15%，其余是网页'},sh:[0.55,0.15,0.15,0.15]},
 {v:'math40',t:{en:'Math 40%',zh:'数学占 40%'},d:{en:'Code 15%, math 40%, the web the rest',zh:'代码 15%，数学 40%，其余是网页'},sh:[0.35,0.10,0.15,0.40]}
];
const mixOf=v=>MIXES.find(m=>m.v===v)||MIXES[0];
const epochsOf=v=>{const m=mixOf(v); return DOM.map((d,i)=>({k:d.k,share:m.sh[i],use:m.sh[i]*TARGET,have:d.have,ep:m.sh[i]*TARGET/d.have}));};
const PILE_EP=3;   // The Pile repeats Wikipedia (en) 3 epochs, its most repeated set (EleutherAI/the-pile README)
/* part 4's example (review codex-b9v3, 应修 3): three short pages written for the lab. A is an original, B a repost of A with only
   the title changed, C another page on the same topic. Everything is computed here, from the words: 5-grams (lower case, punctuation
   dropped), Jaccard, a hash of the whole text (FNV-1a, 32 bits) and a seeded MinHash of 12 values in 3 groups of 4 (value k = the
   smallest of fnv1a("seed:k:5-gram") mod 1000 over the page's 5-grams). The Chinese pages are the same pages in Chinese, cut into
   words by hand, with as many words, so both languages give the same counts (28 of 34 5-grams shared, Jaccard 0.82; C shares none).
   The seed was picked so that one group matches for B and nothing for C; tests/data-pipeline_numbers.py recomputes all of it. */
const XP={
 en:{A:{t:['Simple','Bread','Recipe'],b:'Mix flour, water, salt and a spoon of yeast in a bowl. Knead the dough for ten minutes, let it rise for one hour, and bake it until the crust is brown.'.split(' ')},
     B:{t:['Bread','Made','Easy']},
     C:{t:['Why','Bread','Goes','Stale'],b:'Bread goes stale because the starch inside it slowly hardens as it cools. Keep a loaf you cannot finish in a paper bag at room temperature, and warm old slices in the oven.'.split(' ')}},
 zh:{A:{t:['家常','面包','做法'],b:'把 面粉、 水、 盐 和 一小勺 酵母 一起 放进 大碗里 拌匀。 然后 把 面团 揉 十分钟， 盖 上 布 让 它 发酵 一个 小时， 最后 放进 烤箱， 一直 烤 到 表皮 金黄。'.split(' ')},
     B:{t:['零失败','面包','教程']},
     C:{t:['面包','为什么','会','变硬'],b:'面包 放凉 以后， 里面 的 淀粉 会 慢慢 变硬。 吃 不 完 的 面包 最好 装进 纸袋， 放在 室温 下， 硬 了 的 面包 可以 放进 烤箱 热 一下。'.split(' ')}}
};
const XSEED=4293, XN=12, XR=4, XG=3, XMOD=1000;
function fnv1a(s){let h=0x811c9dc5; for(const c of new TextEncoder().encode(s)){h^=c; h=Math.imul(h,0x01000193)>>>0;} return h>>>0;}
const xnorm=t=>t.toLowerCase().replace(/[^\p{L}\p{N}_]/gu,'');
const xmemo={};
function xOf(lg){
  if(xmemo[lg]) return xmemo[lg];
  const P=XP[lg], sep=lg==='zh'?'':' ', pg={};
  ['A','B','C'].forEach(k=>{const t=P[k].t, b=P[k].b||P.A.b, toks=t.concat(b), w=toks.map(xnorm), sh=[];
    for(let i=0;i+5<=w.length;i++) sh.push(w.slice(i,i+5).join(' '));
    const set=[...new Set(sh)].sort(), sig=[];
    for(let j=0;j<XN;j++){let bv=Infinity, bs=null; for(const s of set){const v=fnv1a(`${XSEED}:${j}:${s}`)%XMOD; if(v<bv||(v===bv&&s<bs)){bv=v; bs=s;}} sig.push({v:bv,s:bs});}
    pg[k]={k,toks,nt:t.length,sh,set,sig,text:t.join(sep)+'\n'+b.join(sep),hash:fnv1a(t.join(sep)+'\n'+b.join(sep))};});
  const pair=k=>{const a=pg.A, x=pg[k], inA=new Set(a.set), shared=x.set.filter(s=>inA.has(s)).length, union=new Set(a.set.concat(x.set)).size;
    const agree=x.sig.map((v,j)=>v.v===a.sig[j].v), groups=[]; for(let g=0;g<XG;g++) groups.push(agree.slice(g*XR,(g+1)*XR).every(Boolean));
    /* a word is shaded when one of the page's 5-grams that holds it is also in A */
    const on=x.toks.map(()=>false); x.sh.forEach((s,i)=>{if(inA.has(s)) for(let q=i;q<i+5;q++) on[q]=true;});
    return {shared,union,jac:shared/union,agree,nAgree:agree.filter(Boolean).length,groups,hit:groups.indexOf(true),caught:groups.some(Boolean),hashSame:x.hash===a.hash,on};};
  return xmemo[lg]={pg,B:pair('B'),C:pair('C')};
}
const hex8=h=>('0000000'+h.toString(16)).slice(-8);

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=v=>Math.round(v*100)+'%';
const pct1=v=>fixd(v*100,1)+'%';
const fT=t=>{const v=t/1e12; return String(+(v>=10?v.toFixed(0):v.toFixed(2)))+'T';};   // 2.25T, not 2.3T (R6.6)
const pshare=v=>{const p=Math.round(v*1000)/10; return (Number.isInteger(p)?String(p):fixd(p,1))+'%';};   // 15%, 1.7%
const fEp=e=>e>=10?fixd(e,0):fixd(e,1);
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const M=t=>`<div class="dl-fx">${A.ui.math(t)}</div>`;
const MS=t=>`<span class="dl-fx">${A.ui.math(t)}</span>`;   // a math line inside a <p> (the kit puts a new case's answer in one)
const NBH=s=>String(s).replace(/Llama 3\.1/g,'Llama 3.1').replace(/RedPajama-V2/g,'RedPajama‑V2');

/* =====================================================================
   module state
   ===================================================================== */
let S=cloneS(BASE), mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let saved=null, chSubmitted=false, chReveal=false, sbTouched=false, userAct=false, wantObs=false, annT=null, remView='run', relang=false;
let lvDup='both', triedDup=new Set(['both']), triedMix=new Set(['prop']), thrTried=false, ruleTried=false, brTried=false;
/* the player: g runs from 0 to 4 (one unit per stage); a picture watched to the end is remembered by its signature (R3.20) */
const PL={g:0,on:false,raf:0,last:0,spd:'fast',sig:null,watched:new Set(),from:0};
const DUR={slow:30000,mid:18000,fast:10000};
const shownPrev={};
const SRC={
 fineweb:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/examples/fineweb.py',
 minhash:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/dedup/minhash.py',
 gopherq:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/filters/gopher_quality_filter.py',
 gopherr:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/filters/gopher_repetition_filter.py',
 fwq:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/filters/fineweb_quality_filter.py',
 lang:'https://github.com/huggingface/datatrove/blob/1977fbb0f3c163d43cace334b073dda17fa3ef26/src/datatrove/pipeline/filters/language_filter.py',
 fw2:'https://github.com/huggingface/fineweb-2/blob/d0defb24f193bb9a5a11b8b14524a03c4858e1b6/README.md',
 rp:'https://github.com/togethercomputer/RedPajama-Data/blob/6d2cee9df2b0204dd2bcb00bf06b5a7b1d7432d7/README.md',
 dedup:'https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/dedup.md',
 pile:'https://github.com/EleutherAI/the-pile/blob/df97f8651ae3da658b19659b3ceaa6a34b0fc014/README.md',
 smol3:'https://github.com/huggingface/blog/blob/998ee73e0594a21b42df14632263266a2831dafe/smollm3.md',
 dclm:'https://github.com/mlfoundations/dclm/blob/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/baselines/baselines_configs/fasttext_filter.yaml',
 dolma:'https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/taggers.md',
 l31:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/MODEL_CARD.md',
 pfw:'https://arxiv.org/abs/2406.17557', prw:'https://arxiv.org/abs/2306.01116', pdc:'https://arxiv.org/abs/2305.16264', plee:'https://arxiv.org/abs/2107.06499'
};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 cc:()=>tx('Common Crawl is a public archive of the web: it crawls billions of pages again and again, and each crawl is published as one snapshot.','Common Crawl 是一个公开的网页存档：它一遍又一遍地抓取大量网页，每抓一轮就发布一个快照。'),
 extract:()=>tx('Extraction keeps the main text of a page’s HTML and drops the menus, ads and footer around it. FineWeb uses the Trafilatura library, set to prefer less text that is surely main text.','抽正文就是从网页的 HTML 里只留下正文，去掉周围的菜单、广告和页脚。FineWeb 用的是 Trafilatura 这个库，设置成“宁可少抽一点，也要抽得准”。'),
 score:()=>tx('A language-ID model reads the text and gives each language a score between 0 and 1: how sure it is that the text is in that language. FineWeb uses fastText’s language-ID model and keeps a page when its English score is at least 0.65.','语言识别模型读一遍正文，给每种语言打一个 0 到 1 之间的分，表示它有多确定这段文字是这种语言。FineWeb 用的是 fastText 的语言识别模型，英文得分不低于 0.65 的网页才留下。'),
 stop:()=>tx('Stop words are the most common words of a language. Gopher’s rule counts how many of these 8 appear at least once: the, be, to, of, and, that, have, with. Normal English prose uses several of them in every paragraph.','停用词是一种语言里最常用的词。Gopher 的规则数这 8 个词里有几个至少出现过一次：the、be、to、of、and、that、have、with。正常的英文段落里，每段都会用到好几个。'),
 hash:()=>tx('A hash turns the whole text into one short number. Two texts that are the same to the last character get the same number; change one character and the number is completely different.','哈希把整篇文字算成一个短短的数。一字不差的两篇，算出来的数一样；改一个字，算出来的数就完全不同。'),
 shingle:()=>tx('A 5-gram is a run of 5 words in a row. A page of 300 words has about 300 of them, one starting at each word. FineWeb cuts pages into 5-grams before comparing them.','5-gram 是连在一起的 5 个词。一篇 300 个词的网页大约有 300 个，从每个词开始各取一个。FineWeb 先把网页切成 5-gram 再比较。'),
 jac:()=>tx('Jaccard similarity of two pages: the 5-grams they share, divided by all the different 5-grams the two have together. 1 means the same text, and a repost with a new title and one added line is still above 0.8. At 0.5 two pages of the same length still share about two thirds of their 5-grams; two articles written separately share almost none.','两篇网页的 Jaccard 相似度，是两篇共有的 5-gram 数，除以两篇加起来一共有多少种不同的 5-gram。1 表示文字完全一样。转载时只改了标题、加了一行，相似度仍在 0.8 以上。相似度是 0.5 时，两篇仍有大约三分之二的 5-gram 相同。两篇各自写的文章，几乎没有相同的 5-gram。'),
 sig:()=>tx('A MinHash signature is a row of numbers per page. For each of many hash functions, it keeps the smallest hash among the page’s 5-grams. For two pages, the numbers in the same position agree with a chance equal to their Jaccard similarity.','MinHash 签名是每篇网页的一排数。对很多个哈希函数中的每一个，它记下这篇网页所有 5-gram 里最小的那个哈希值。两篇网页的签名在同一个位置上相等的机会，正好等于它们的 Jaccard 相似度。'),
 band:()=>tx('Groups: the signature is cut into b groups of r hash values. Two pages become candidate duplicates when all r values of at least one group agree. Grouping saves comparing every pair of pages: the r values of a group form one key, pages go into buckets by key, and only pages that share a bucket are candidates. In LSH the groups are usually called bands, and the values in a group rows.','分组：把签名切成 b 组，每组 r 个哈希值。两篇网页只要至少有一组的 r 个哈希值全部相同，就被当成重复。分组是为了不用把网页两两比较：一组的 r 个哈希值拼成一个键，按键分桶，落进同一个桶的网页才是重复。在 LSH 里，组通常叫 band，组里的每个值叫 row。'),
 epoch:()=>tx('One epoch is one pass over a domain’s data. Taking more tokens of a domain than you have means repeating the same text: 2.5 epochs means each text is seen two or three times.','1 遍（epoch）是把一类数据完整用一遍。某类数据要用的 token 比手里的多，就只能重复用同一批文字：2.5 遍就是每篇文字被看到两三次。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'half',t:{en:'More than half (over 500)',zh:'一半以上（500 个以上）'}},{v:'third',t:{en:'30% to half (300 to 500)',zh:'三成到一半（300 到 500 个）'}},
  {v:'tenth',t:{en:'10% to 30% (100 to 300)',zh:'一成到三成（100 到 300 个）'}},{v:'few',t:{en:'Under 10% (under 100)',zh:'不到一成（不到 100 个）'}}];
const predOf=k=>k>500?'half':(k>=300?'third':(k>=100?'tenth':'few'));
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const F0=()=>fates(BASE);
const thrCtl=(id,inline)=>A.ui.range({id,key:'thr',label:tx('English score threshold','英文得分的门槛'),min:30,max:95,step:5,value:Math.round(S.thr*100),out:fixd(S.thr,2),inline});
const bCtl=(id,inline)=>A.ui.range({id,key:'b',label:tx('Groups b (how many groups the signature is cut into)','组数 b（签名分成几组）'),min:1,max:60,value:S.b,out:String(S.b),inline});
const rCtl=(id,inline)=>A.ui.range({id,key:'r',label:tx('Hashes per group r','每组的哈希数 r'),min:1,max:20,value:S.r,out:String(S.r),inline});
function ruleSw(inline){
  return `<div class="dl-rules${inline?' dl-rules-in':''}" role="group" aria-label="${esc(tx('Quality rules; they stack','质量规则，可以叠加'))}">`+RULES.map(R=>`<button type="button" class="dl-sw" data-rule="${R.k}" aria-pressed="${!!S.rules[R.k]}"><i aria-hidden="true"></i><span><b>${esc(L(R.name))}</b><small>${esc(ruleLine(R.k))}</small></span><em class="mono" data-rule-n="${R.k}"></em></button>`).join('')+`</div>`;
}
function ruleLine(k){return {
  rep:tx('More than 30% of the lines repeat another line (Gopher)','超过 30% 的行和别的行一模一样（Gopher）'),
  short:tx('Fewer than 50 words (Gopher)','不到 50 个词（Gopher）'),
  sym:tx('More than one “#” per 10 words, or more than one “…” per 10 words (counted separately; Gopher)','每 10 个词里，“#”多于 1 个，或者“…”多于 1 个（两种符号分开数，Gopher）'),
  stop:tx('Fewer than 2 of the 8 stop words (Gopher)','8 个停用词里出现不到 2 个（Gopher）'),
  punct:tx('Under 12% of the lines end in a full stop, “!” or “?” (FineWeb)','以句号、问号、感叹号这类标点结尾的行不到 12%（FineWeb）')}[k];}
const LVD=[{v:'exact',t:{en:'Only exact copies (a hash of the whole text)',zh:'只去完全一样的（比较全文的哈希）'},d:{en:'A repost like B, with a new title, stays',zh:'像 B 这样只改了标题的转载会留下来'}},
  {v:'both',t:{en:'Exact copies, then near copies with MinHash (initial setup)',zh:'先去完全一样的，再用 MinHash 去几乎一样的（初始设置）'},d:{en:'FineWeb’s setting: 14 groups of 8 hashes',zh:'FineWeb 的设置：14 组，每组 8 个哈希'}}];
const lvdRes=v=>{const f=fatesOf({...cloneS(BASE),exact:1,mh:v==='both'?1:0}); return tx(`${f.dupLeft} near copies left`,`留下 ${f.dupLeft} 篇副本`);};
const mixLv=()=>`<div class="dl-lv" id="dl-ic-mix">`+A.ui.levers({aria:tx('The mix, pick one','配比，选一种'),value:S.mix,options:MIXES.map(m=>({v:m.v,title:L(m.t),desc:L(m.d),result:esc(triedMix.has(m.v)||m.v===S.mix?mixRes(m.v):'')}))})+`</div>`;
const mixRes=v=>{const e=epochsOf(v)[3]; return tx(`math ${fEp(e.ep)} epochs`,`数学用 ${fEp(e.ep)} 遍`);};
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'funnel',gate:()=>!!pred&&PL.watched.has(F0().sig),skip:true,
  lead:()=>tx('1,000 crawled pages go through four stages. How many are left?','抓来的 1,000 个网页走过四个环节，还剩多少？'),
  enter:()=>cloneS(BASE),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>You are preparing data to pretrain a mainly English model, and the data comes from crawling the web. Most crawled pages cannot go into training as they are. Some hold only menus and ads. Some are in other languages, some are lists of keywords, and the same article is often reposted many times.</p><p>A data pipeline filters in stages. It first extracts the main text, then identifies the language, then drops low-quality pages by rules, and finally removes duplicates. Every stage throws some pages away.</p><p>The chart shows 1,000 crawled pages, one small square each. Take a gut guess: after all four stages, about how many are left?</p>${ch}${pred?`<p class="muted small dl-saved"${PL.watched.has(F0().sig)?' hidden':''}>Saved. You can still change it until the run ends.</p>`:''}`,
             `<p>你们要预训练一个以英文为主的模型，数据从网上抓。抓来的网页大多不能直接拿来训练：有的只有菜单和广告，有的是别的语言，有的只是一串关键词，同一篇文章还常常被转载很多次。</p><p>数据管线一个环节一个环节地筛：先抽出正文，再认语言，再按规则去掉质量差的，最后去重。每个环节都会扔掉一部分网页。</p><p>图里是抓来的 1,000 个网页，每个小方块是一个。凭直觉猜一下：走完这四个环节，大约还剩多少个？</p>${ch}${pred?`<p class="muted small dl-saved"${PL.watched.has(F0().sig)?' hidden':''}>已记下。动画播完之前，还可以改。</p>`:''}`);},
  obs:()=>obsMission()},
 {id:'lang',title:{en:'Main text and language',zh:'抽正文，认语言'},view:'funnel',expect:()=>thrTried,showMe:{thr:0.9},label:{en:'Result',zh:'结果'},hi:[1,2],
  lead:()=>tx('A page without main text is dropped; so is one whose English score is under the threshold.','抽不出正文的扔掉，英文得分低于门槛的也扔掉。'),
  enter:()=>cloneS(BASE),
  body:()=>tx(`<p>Only a small part of a page’s HTML is the main text. ${DEF('Extraction',TIP.extract())} keeps that part. A login page or a photo gallery has none, so the whole page is dropped.</p><p>Next, a language-ID model reads the text and gives an ${DEF('English score',TIP.score())} between 0 and 1: how sure it is that the text is English. A page under the threshold is dropped; FineWeb’s threshold is 0.65. Pages in other languages go through pipelines of their own, one per language, and come back when the mix is set.</p><p>Move the threshold up or down.</p>`,
              `<p>网页的 HTML 里，正文只占一小部分。${DEF('抽正文',TIP.extract())}就是只留下这部分。登录页、图片页抽不出正文，整个网页就扔掉。</p><p>接着，语言识别模型读一遍正文，给出一个 0 到 1 之间的${DEF('英文得分',TIP.score())}，表示它有多确定这是英文。得分低于门槛的网页就扔掉，FineWeb 的门槛是 0.65。别的语言的网页另走按语言分开的管线，定配比时再放进来。</p><p>把门槛调高或调低试试。</p>`)+thrCtl('dl-ic-thr',true),
  obs:()=>obsLang()},
 {id:'rules',title:{en:'Quality rules',zh:'质量规则'},view:'funnel',expect:()=>ruleTried,showMe:()=>setRule('stop',0),label:{en:'Result',zh:'结果'},hi:[3],
  lead:()=>tx('Each rule looks at one number and catches one kind of junk.','每条规则只看一个数值，专抓一种垃圾。'),
  enter:()=>cloneS(BASE),
  body:()=>tx(`<p>Among the English pages left, many are still not real articles: keyword lists, rows of tags, a menu item on every line, one line repeated dozens of times. FineWeb drops them with simple rules, and each rule looks at a single number.</p><p>Turn off any of the rules, several at once if you like, and see where the pages they caught go.</p>`,
              `<p>剩下的英文网页里，还有不少不是正经文章：一串关键词、一排标签、一行一个菜单项、同一行重复几十遍。FineWeb 用几条简单的规则去掉它们，每条规则只看一个数值。</p><p>关掉你想关的规则，可以同时关几条，看看它们原来抓住的网页去了哪里。</p>`)+ruleSw(true),
  obs:()=>obsRules()},
 {id:'dedup',title:{en:'Duplicates, exact and near',zh:'去重：不只是完全一样的'},view:'dedup',expect:()=>triedDup.has('exact'),showMe:()=>setLvDup('exact'),label:{en:'Result',zh:'结果'},hi:[4],
  lead:()=>tx('A repost with only a new title has a different hash. MinHash still finds it.','只改了标题的转载，哈希就完全变了。MinHash 还能找出它。'),
  enter:()=>{lvDup='both'; triedDup=new Set(['both']); brTried=false; return cloneS(BASE);},
  body:()=>tx(`<p>The same article is often reposted, often with only a new title. These near copies must go too: a model trained on many copies of a text tends to repeat it word for word, and the copies waste training steps.</p><p>Exact copies are easy to find: the same ${DEF('hash',TIP.hash())} of the whole text means the same page. But B in the chart changed only its title, and its hash is completely different.</p><p>MinHash cuts each page into ${DEF('5-grams',TIP.shingle())} and gives it a ${DEF('signature',TIP.sig())}: each hash value of two pages’ signatures agrees with a chance equal to their ${DEF('similarity',TIP.jac())} s. The signature is ${DEF('cut into b groups of r hash values',TIP.band())}, and two pages count as duplicates when any one group agrees in full.</p><p>Each dot in the chart is a pair of pages, and a dot under the curve counts as a duplicate. At similarity 0.8 or more the second page is a copy and should go; at 0.6 or less it is a heavily edited version and should stay; in between, either is fine. b and r trade copies missed against pages wrongly dropped.</p><p>First pick “Only exact copies” and see what happens to B and to the dots.</p>`,
              `<p>同一篇文章常被转载，还常常只改个标题。这些几乎一样的副本也要去掉：副本多了，模型容易一字不差地背出原文，训练步数也白花。</p><p>完全一样的副本好找：全文的${DEF('哈希',TIP.hash())}相同，就是同一篇。可图里的 B 只改了标题，哈希就完全不同了。</p><p>MinHash 把网页切成 ${DEF('5-gram',TIP.shingle())}，再给每篇算一个${DEF('签名',TIP.sig())}：两篇签名里每个哈希值相同的机会，正好是两篇的${DEF('相似度',TIP.jac())} s。签名${DEF('分成 b 组、每组 r 个哈希值',TIP.band())}，只要有一组全部相同，两篇就算重复。</p><p>图里每个点是一对网页，落在曲线下面就算重复。相似度 0.8 以上的是副本，该删；不超过 0.6 的是改动很大的另一个版本，该留；中间的删不删都行。b 和 r 就在漏删和误删之间取舍。</p><p>先选“只去完全一样的”，看看 B 和图里的点会怎样。</p>`)+
    `<div class="dl-lv" id="dl-ic-dup">`+A.ui.levers({aria:tx('How to remove duplicates, pick one','怎么去重，选一种'),value:lvDup,options:LVD.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(triedDup.has(o.v)?lvdRes(o.v):'')}))})+`</div>`+
    `<div class="dl-br" id="dl-ic-br"${lvDup==='both'?'':' hidden'}><p class="small">${tx('These two settings change only the curve of the 1,000 pages: more hashes per group move it right, so MinHash is stricter; more groups move it left.','这两个设置只改 1,000 个网页的曲线：每组的哈希越多，它越往右，判得越严；组越多，越往左。')}</p>${bCtl('dl-ic-b',true)}${rCtl('dl-ic-r',true)}</div>`,
  obs:()=>obsDedup()},
 {id:'mix',title:{en:'The mix',zh:'配比'},view:'mix',expect:()=>triedMix.size>1,showMe:()=>setMix('math15'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A small domain given a big share is repeated many times.','数据少的一类占的比例大，就要重复很多遍。'),
  enter:()=>{triedMix=new Set(['prop']); return cloneS(BASE);},
  body:()=>tx(`<p>After dedup comes the mix: what share of the training tokens each kind of data gets. Give a small domain a big share and it is repeated many times.</p><p>The target is 15T tokens, about what Llama 3.1 was trained on; how much each kind has on hand is written on its row in the chart. When a domain’s share asks for more tokens than you have, the same text is used again: its ${DEF('epochs',TIP.epoch())} go above 1. Pick a mix and see how many epochs each kind of data needs.</p>`,
              `<p>去完重，还要定配比：训练用的 token 里，每类数据各占多少。数据少的一类占的比例大，就要重复很多遍。</p><p>目标是训练 15T 个 token，和 Llama 3.1 差不多；每类数据手里有多少，写在图里它那一行。某一类要用的 token 比手里的多，就只能把同一批文字再用一次，它的${DEF('遍数',TIP.epoch())}就超过 1。选一种配比，看看每类数据要用几遍。</p>`)+mixLv(),
  obs:()=>obsMix()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>remView==='mix'?'mix':(remView==='dup'?'dedup':'funnel'),
  lead:()=>tx('Every stage drops a lot; find near copies, not just exact ones; repeat a small domain only a few times.','每个环节都扔掉一大块；去重要找几乎一样的；小的数据别重复太多遍。'),
  enter:()=>{remView='run'; return cloneS(BASE);},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
function verdict(){
  if(!pred) return '';
  const right=predOf(F0().kept), pt=esc(predT(pred)), rt=esc(predT(right));
  return pred===right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
function obsMission(){
  const f=F0(); if(!PL.watched.has(f.sig)) return '';
  const big=[1,2,3,4].reduce((a,k)=>f.rm[k]>f.rm[a]?k:a,1);
  return tx(`<p><b>After the four stages, ${f.kept} of the 1,000 pages are left, about ${Math.round(f.kept/10)}%.</b></p>`,`<p><b>走完四个环节，1,000 个网页还剩 ${f.kept} 个，约 ${Math.round(f.kept/10)}%。</b></p>`)+verdict()+
    tx(`<p>Every stage throws away a large part; the largest bin is “${binName(big)}”. How many pages each stage drops here is this lab’s assumption. Real pipelines lose as much. RedPajama‑V2 has over 100B documents from 84 ${DEF('Common Crawl',TIP.cc())} snapshots. Only 30B of them, the better part, carry quality signals, and 20B are left after dedup (${cite(SRC.rp,'RedPajama-Data README')}).</p><p>Red squares are what a stage got wrong: a good page thrown away, or junk or a duplicate that stayed. Point at any square to see why it went where it went.</p>`,
       `<p>每个环节都扔掉一大块，最大的一堆是“${binName(big)}”。各环节在这里扔掉多少，是这个 Lab 的假设。真实的管线也扔掉这么多。RedPajama‑V2 从 84 个 ${DEF('Common Crawl',TIP.cc())} 快照里整理出 100B 多个文档，只有质量较好的 30B 个带上了质量信号，去重后剩 20B 个（${cite(SRC.rp,'RedPajama-Data 的 README')}）。</p><p>红色的方块是某个环节判错的：好网页被扔掉了，或者垃圾、重复的网页留了下来。鼠标移到任意一个方块上，能看到它为什么去了那里。</p>`);
}
function binName(k){return [null,tx('no main text','抽不出正文'),tx('not English','不是英文'),tx('low quality','质量差'),tx('duplicates','重复')][k];}
function obsLang(){
  const f=fatesOf(S), f0=F0(), D=corpus(S.corpus).docs;
  let wrongOut=0, wrongIn=0; D.forEach((d,i)=>{if(d.noMain) return; const en=d.lang==='en'; if(en&&GOODU.has(d.kind)&&f.why[i]==='lang') wrongOut++; if(!en&&d.en>=S.thr-1e-9) wrongIn++;});
  const t=fixd(S.thr,2), a=f0.rm[2], b=f.rm[2], q=b/a;
  const good=wrongOut?tx(`, and ${wrongOut} of them are good English pages`,`，其中有 ${wrongOut} 个是好的英文网页`):tx(', and no good English page is among them','，好的英文网页一个也没扔');
  const leak=wrongIn?tx(`${wrongIn} ${wrongIn>1?'pages':'page'} in other languages ${wrongIn>1?'get':'gets'} through.`,`还有 ${wrongIn} 个别的语言的网页混了进来。`):tx('No page in another language gets through.','别的语言的网页一个也没混进来。');
  /* the change against FineWeb's 0.65, in words (R6.20): the exact counts are in the bins and Show the math */
  const chg=q>=2.05?tx(`${fixd(q,1)} times as many pages as at 0.65 are dropped as not English`,`因为“不是英文”被扔掉的网页，是 0.65 时的 ${fixd(q,1)} 倍`)
    :q>=1.97?tx('twice as many pages as at 0.65 are dropped as not English','因为“不是英文”被扔掉的网页，比 0.65 时多了一倍')
    :q>=1.8?tx('almost twice as many pages as at 0.65 are dropped as not English','因为“不是英文”被扔掉的网页，比 0.65 时多了将近一倍')
    :q>1.03?tx(`about ${Math.round((q-1)*100)}% more pages than at 0.65 are dropped as not English`,`因为“不是英文”被扔掉的网页，比 0.65 时多了约 ${Math.round((q-1)*100)}%`)
    :q>=0.97?tx('about as many pages as at 0.65 are dropped as not English','因为“不是英文”被扔掉的网页，和 0.65 时差不多')
    :tx(`about ${Math.round((1-q)*100)}% fewer pages than at 0.65 are dropped as not English`,`因为“不是英文”被扔掉的网页，比 0.65 时少了约 ${Math.round((1-q)*100)}%`);
  let s=Math.abs(S.thr-0.65)<1e-9?tx(`<p><b>At FineWeb’s threshold of 0.65, ${b} pages are dropped as not English${good}. ${leak}</b></p>`,`<p><b>门槛是 FineWeb 的 0.65 时，因为“不是英文”被扔掉的有 ${b} 个网页${good}。${leak}</b></p>`)
    :tx(`<p><b>At a threshold of ${t}, ${chg}${good}. ${leak}</b></p>`,`<p><b>门槛${S.thr>0.65?'提到':'降到'} ${t}，${chg}${good}。${leak}</b></p>`);
  s+=tx('<p>The threshold is a trade-off. A higher one keeps out more foreign text, but it also drops English pages the model is less sure about, such as pages full of code or numbers. A lower one keeps those pages and lets in more pages that mix English with another language. Many of those still fall to the stop-word rule of part 3.</p>',
        '<p>门槛定在哪是一个取舍。门槛高，别的语言混进来得少，可是模型把握不大的英文网页也一起被扔了，比如夹着很多代码和数字的网页。门槛低，这些网页能留下，可夹着英文的外语网页也会多混进来一些。混进来的，很多在第 3 部分会被“停用词太少”这条规则再抓住。</p>');
  return s;
}
function obsRules(){
  const f=fatesOf(S), f0=F0(), off=RK.filter(k=>!S.rules[k]), D=corpus(S.corpus).docs;
  let seo=0; D.forEach((d,i)=>{if(d.kind==='seo'&&f0.red[i]&&f0.st[i]===5) seo++;});
  const tail=tx(`<p>Some junk passes every rule. In this lab, ${f0.redKept} red pages stay even with every rule on, and ${seo} of them are stitched-together SEO text. For pages like these, DCLM adds one more step: a fastText classifier scores every page, and pages with a low score are dropped (${cite(SRC.dclm,'DCLM <code>fasttext_filter.yaml</code>')}).</p>`,
                `<p>还有的垃圾每条规则都能通过。在这个 Lab 里，规则全开时也有 ${f0.redKept} 个红色的网页留了下来，其中 ${seo} 个是拼凑出来的 SEO 文章。DCLM 为此又加了一步，用一个 fastText 分类器给每个网页打分，分数低的扔掉（${cite(SRC.dclm,'DCLM 的 <code>fasttext_filter.yaml</code>')}）。</p>`);
  if(!off.length) return tx(`<p><b>With all five rules on, ${f.rm[3]} pages are dropped as low quality.</b> The number on each switch is how many pages that rule dropped. The rules run in order, and a page dropped by one rule is not checked by the rules after it.</p>`,`<p><b>5 条规则全开时，有 ${f.rm[3]} 个网页因为质量差被扔掉。</b>每个开关上的数，是由这条规则扔掉的网页数。规则按顺序检查，一个网页被前面的规则扔掉，后面的规则就不再看它。</p>`);
  let was=0, later=0, kept=0;
  D.forEach((d,i)=>{if(f0.st[i]===3&&off.includes(f0.why[i])){was++; if(f.st[i]===3) later++; else if(f.st[i]===5) kept++;}});
  const many=off.length>1, names=off.map(k=>`“${L(RULES.find(R=>R.k===k).name)}”`).join(tx(' and ','和'));
  const its=tx(many?'the pages those rules dropped':'the pages that rule dropped',many?'原来归它们管的网页':'原来归它管的网页');
  const s0=!kept?tx(`<p><b>With ${names} off, ${its} are all caught by another rule; none gets through.</b></p>`,`<p><b>关掉${names}以后，${its}都被别的规则抓住了，一个也没漏过去。</b></p>`)
    :!later?tx(`<p><b>With ${names} off, all ${kept} of ${its} get through and stay in the data, red in the chart.</b></p>`,`<p><b>关掉${names}以后，${its}全都漏了过去，${kept} 个都留在了数据里，图里是红色。</b></p>`)
    :kept>later?tx(`<p><b>With ${names} off, most of ${its} get through. ${later/was<0.35?'Only a few':'Some'} are caught by a later rule, and the other ${kept} stay in the data, red in the chart.</b></p>`,`<p><b>关掉${names}以后，${its}大多漏了过去。只有${later/was<0.35?'少数':'一部分'}又被后面的规则抓住，其余 ${kept} 个留在了数据里，图里是红色。</b></p>`)
    :tx(`<p><b>With ${names} off, most of ${its} are caught by another rule, but ${kept} stay in the data, red in the chart.</b></p>`,`<p><b>关掉${names}以后，${its}大多又被别的规则抓住，可还有 ${kept} 个留在了数据里，图里是红色。</b></p>`);
  return s0+tx('<p>A junk page often breaks several rules at once, so the rules overlap. Yet every rule catches some pages no other rule catches.</p>','<p>一篇垃圾网页常常同时违反好几条规则，所以规则之间有重叠。可每条规则都有只有它才抓得到的网页。</p>')+tail;
}
/* pages that should stay (heavily edited versions, similarity 0.6 or less) but MinHash caught: red dots under the curve */
const wrongDrop=(Sx,f)=>pairsOf(Sx).filter(p=>GOODU.has(p.kind)&&f.st[p.i]===4).length;
function obsDedup(){
  const f=fatesOf(S), fe=fatesOf({...S,mh:0}), fb=fatesOf({...cloneS(BASE)});
  let s;
  const X=xOf(A.lang());
  if(!S.mh) s=tx(`<p><b>Comparing whole-text hashes keeps B, the repost. Among the 1,000 pages, ${fe.dupLeft} near copies stay in the data the same way, red in the chart.</b></p>`,`<p><b>只比全文哈希时，转载 B 留了下来。1,000 个网页里，也有 ${fe.dupLeft} 篇几乎一样的副本这样留在了数据里，图里是红色。</b></p>`);
  else {const bad=wrongDrop(S,f), left=f.dupLeft, set=tx(`With MinHash at ${S.b} groups of ${S.r} hashes`,`MinHash 用 ${S.b} 组、每组 ${S.r} 个哈希时`);
    const ex=X.B.caught&&!X.C.caught?tx(`<p><b>In the example, MinHash finds B: group ${X.B.hit+1} agrees in full. C stays.</b></p>`,`<p><b>示例里，MinHash 找出了 B，因为第 ${X.B.hit+1} 组全部相同。C 留了下来。</b></p>`):'';
    const kept0=tx('no near copy stays','几乎一样的副本一篇也没留下'), keptN=tx(`${left} near ${left>1?'copies stay':'copy stays'}`,`留下了 ${left} 篇几乎一样的副本`);
    const lost=n=>tx(`${n} heavily edited ${n>1?'versions':'version'} that should stay ${n>1?'are':'is'} thrown away`,`误删了 ${n} 篇改动很大、本该留下的网页`);
    s=ex+(!left&&!bad?tx(`<p><b>${set}, ${kept0}, and no page that should stay is thrown away.</b></p>`,`<p><b>${set}，${kept0}，也没有误删本该留下的网页。</b></p>`)
     :!left?tx(`<p><b>${set}, ${kept0}, ${bad<=2?'and only':'but'} ${lost(bad)}.</b></p>`,`<p><b>${set}，${kept0}，${bad<=2?'只':'可也'}${lost(bad)}。</b></p>`)
     :!bad?tx(`<p><b>${set}, no page that should stay is thrown away, but ${keptN}.</b></p>`,`<p><b>${set}，没有误删本该留下的网页，可${keptN}。</b></p>`)
     :tx(`<p><b>${set}, ${keptN}, and ${lost(bad)}.</b></p>`,`<p><b>${set}，${keptN}，还${lost(bad)}。</b></p>`));}
  /* polish round 2 (2026-10-15): the formula P(s) and where it comes from are in Show the math; what b and r do to the curve is the
     sliders' lead-in in the card */
  s+=
    (S.b===14&&S.r===8&&S.mh?'':(fb.dupLeft?tx(`<p class="small">FineWeb’s setting (14 groups of 8 hashes) leaves ${fb.dupLeft} near copies.</p>`,`<p class="small">FineWeb 的设置（14 组、每组 8 个哈希）留下 ${fb.dupLeft} 篇几乎一样的副本。</p>`)
      :tx('<p class="small">FineWeb’s setting (14 groups of 8 hashes) leaves no near copy.</p>','<p class="small">FineWeb 的设置（14 组、每组 8 个哈希）一篇几乎一样的副本也没留下。</p>')));
  return s;
}
function obsMix(){
  const e=epochsOf(S.mix), m=e[3], c=e[2];
  let s=S.mix==='prop'?tx(`<p><b>In proportion, every domain is used ${fEp(m.ep)} times, less than one full pass: no text is repeated.</b> Try a mix with more math.</p>`,`<p><b>按数据量分时，每一类都只用 ${fEp(m.ep)} 遍，连一遍都不到，没有文字被重复。</b>再试试多放数学的配比。</p>`)
    :(()=>{const q=m.ep/PILE_EP, small=m.share<=0.15+1e-9;
      const vs=q<=1?tx(`, not more than The Pile’s most repeated set (${PILE_EP} times)`,`，还没超过 The Pile 重复最多的 ${PILE_EP} 遍`)
        :q<2?tx(`, more than The Pile’s most repeated set (${PILE_EP} times)`,`，比 The Pile 重复最多的 ${PILE_EP} 遍还多`)
        :q<3?tx(`, more than twice The Pile’s most repeated set (${PILE_EP} times)`,`，比 The Pile 重复最多的 ${PILE_EP} 遍还多一倍多`)
        :tx(`, more than ${Math.floor(q)} times The Pile’s most repeated set (${PILE_EP} times)`,`，是 The Pile 重复最多的 ${PILE_EP} 遍的 ${Math.floor(q)} 倍多`);
      return tx(`<p><b>Math is ${small?'only ':''}${pct(m.share)} of the run, but there is little math data on hand, so the same math text is used ${fEp(m.ep)} times${vs}.</b></p>`,`<p><b>数学${small?'只':''}占 ${pct(m.share)}，可手里的数学数据太少，同一批数学文字要用 ${fEp(m.ep)} 遍${vs}。</b></p>`);})();
  s+=tx(`<p>Teams do raise small high-quality domains on purpose: SmolLM3 raised math from 3% of its first training stage to 13% of its last (${cite(SRC.smol3,'Hugging Face, SmolLM3')}). The Pile, a public pretraining dataset, repeated its smaller sets up to 3 times (${cite(SRC.pile,'the-pile README')}). Repeating a few times costs little; repeating many times buys little for the compute it takes (Muennighoff et al., 2023).</p>`,
        `<p>把数据少、质量高的类别调高，是常见的做法：SmolLM3 分阶段训练，数学从第一阶段的 3% 提到最后阶段的 13%（${cite(SRC.smol3,'Hugging Face 的 SmolLM3 博客')}）。公开的预训练数据集 The Pile，把里面较小的几个子集最多重复了 3 遍（${cite(SRC.pile,'the-pile 的 README')}）。重复几遍代价不大；重复很多遍，花的算力就换不来多少提升（Muennighoff 等，2023）。</p>`);
  return s;
}
const REMS={run:{s:{}},dup:{s:{mh:0}},mix:{s:{mix:'math40'}}};
function remBtn(k){const on=remView===k;
  const tip={run:tx('Shows where the 1,000 pages go at the initial setup.','在图里显示初始设置下，1,000 个网页各去了哪里。'),dup:tx('Shows the run with exact copies removed only.','在图里显示只去完全一样的副本时的结果。'),mix:tx('Shows the mix with math at 40%.','在图里显示数学占 40% 的配比。')}[k];
  return ` <button type="button" class="linkbtn dl-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const f=F0(), fe=fatesOf({...cloneS(BASE),mh:0}), m40=epochsOf('math40')[3];
  return tx(`<ul class="dl-remlist"><li><b>Expect most of a crawl to go.</b> So to get the tokens a run needs, start from several times as much raw web text.${remBtn('run')}</li>`+
    `<li><b>Remove near copies, not only exact ones.</b> A repost often changes only a line, and its hash is then completely different. Comparing hashes keeps such reposts; MinHash finds them.${remBtn('dup')}</li>`+
    `<li><b>A small domain can be raised, a little.</b> Set its share too high and the same text is read many times over, and each extra pass teaches the model less.${remBtn('mix')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “More data is always better, so keep everything.” Most crawled pages are menus, other languages, junk or copies. Training on them costs compute and teaches the model to repeat them.</p><p>✗ “Dedup means deleting identical files.” Most copies differ by a line: comparing hashes misses them, and only MinHash finds them.</p><p>✗ “The stricter the filter, the better.” Set the language threshold too high and good English pages go out along with the foreign ones. Filter too hard and too few tokens are left for the run.</p></div>${A.shell.toScenarios()}`,
    `<ul class="dl-remlist"><li><b>抓来的网页，大部分都留不下。</b>所以训练要多少 token，就得准备好几倍的原始网页。${remBtn('run')}</li>`+
    `<li><b>去重要去几乎一样的，不只是完全一样的。</b>转载常常只改一行字，哈希就完全变了。只比哈希，这些转载都会留下，要靠 MinHash 才找得出来。${remBtn('dup')}</li>`+
    `<li><b>数据少的类别可以多放，但别放太多。</b>比例定得太高，同一批文字就要读很多遍，而每多读一遍，模型学到的就更少。${remBtn('mix')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “数据越多越好，不用筛。”抓来的网页，大部分是菜单、别的语言、垃圾或副本。用它们训练既费算力，又会让模型学着复读它们。</p><p>✗ “去重就是删掉完全一样的文件。”大多数副本只差一行字，只比哈希会漏掉，只有 MinHash 找得出来。</p><p>✗ “过滤越严越好。”语言的门槛定得太高，好的英文网页也会跟着别的语言一起被扔掉。筛得太狠，留给训练的 token 就不够了。</p></div>${A.shell.toScenarios()}`);
}
/* fates are cached by the settings that change them */
const FC=new Map();
function fates(s){const k=[s.corpus,s.thr,RK.map(x=>s.rules[x]?1:0).join(''),s.exact?1:0,s.mh?1:0,s.b,s.r].join('|'); let f=FC.get(k); if(!f){f=fatesOf(s); f._corpus=s.corpus; if(FC.size>120) FC.clear(); FC.set(k,f);} return f;}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'fw',t:{en:'FineWeb’s recipe on an English crawl',zh:'FineWeb 的做法，筛一批英文网页'},d:{en:'The initial setup: all four stages as FineWeb runs them',zh:'初始设置：四个环节都照 FineWeb 的做法'},s:cloneS(BASE)},
 {id:'multi',t:{en:'A crawl mostly in other languages',zh:'大部分不是英文的一批网页'},d:{en:'The same recipe; two in three pages are not English',zh:'做法不变，三分之二的网页不是英文'},s:{...cloneS(BASE),corpus:'foreign'}},
 {id:'code',t:{en:'The same recipe on source code',zh:'同样的做法，用来筛代码'},d:{en:'Source files from a code host, run through the web filters',zh:'代码托管网站上的源文件，用网页的过滤规则来筛'},s:{...cloneS(BASE),corpus:'code'}},
 {id:'exact',t:{en:'Skip near-duplicate removal',zh:'不去几乎一样的副本'},d:{en:'Only exact copies are removed, to save compute',zh:'为了省算力，只去完全一样的副本'},s:{...cloneS(BASE),mh:0}}
];
const sameS=(a,b)=>a.corpus===b.corpus&&Math.abs(a.thr-b.thr)<1e-9&&RK.every(k=>!!a.rules[k]===!!b.rules[k])&&!!a.exact===!!b.exact&&!!a.mh===!!b.mh&&(!a.mh||(a.b===b.b&&a.r===b.r));
const presetNow=()=>PRESETS.findIndex(p=>sameS(p.s,S));
function scenHTML(i){
  const p=PRESETS[i], f=fates(S), D=corpus(S.corpus).docs;
  if(p.id==='fw') return tx(`<p><b>${f.kept} of the 1,000 pages are left. The largest bin is “${binName(maxBin(f))}”.</b></p><p>Each filter in datatrove writes the pages it drops to a folder of its own, so a team can count and read what every stage throws away (${cite(SRC.fineweb,'datatrove <code>examples/fineweb.py</code>')}).</p>`,
                            `<p><b>1,000 个网页留下 ${f.kept} 个，扔得最多的是“${binName(maxBin(f))}”。</b></p><p>datatrove 的每个过滤器都把扔掉的网页写进自己的文件夹，团队可以数一数、读一读每个环节扔了什么（${cite(SRC.fineweb,'datatrove 的 <code>examples/fineweb.py</code>')}）。</p>`);
  if(p.id==='multi') return tx(`<p><b>Only ${f.kept} of 1,000 pages are left; the language stage alone drops ${f.rm[2]}.</b></p><p>For an English model that is right. For other languages these pages are good data. FineWeb 2 started from the pages that scored under 0.65 in English when FineWeb was built, and filtered and deduplicated them per language (${cite(SRC.fw2,'fineweb-2 README')}).</p>`,
                               `<p><b>1,000 个网页只留下 ${f.kept} 个，光是认语言这一个环节就扔掉 ${f.rm[2]} 个。</b></p><p>如果要训练的是英文模型，这样扔没错。可这些网页对别的语言是好数据。FineWeb 2 就是从做 FineWeb 时英文得分低于 0.65 的网页做起的，再按语言分别过滤、去重（${cite(SRC.fw2,'fineweb-2 的 README')}）。</p>`);
  if(p.id==='code'){let good=0, lost=0; D.forEach((d,i)=>{if(d.kind==='code'){good++; if(f.st[i]<5) lost++;}});
    return tx(`<p><b>The web filters throw away ${lost} of the ${good} good source files, red in the chart.</b></p><p>Code has a low English score, few stop words, and few lines that end in a full stop, so rules written for prose drop it. Code datasets run their own filters and their own dedup; The Stack, for example, deduplicates with MinHash on 5-grams at a similarity of 0.7 (${cite(SRC.dedup,'Hugging Face’s dedup post')}).</p>`,
              `<p><b>网页的过滤规则把 ${good} 个好的源文件扔掉了 ${lost} 个，图里是红色。</b></p><p>代码的英文得分低，停用词少，以句号结尾的行也少，为文章写的规则会把它扔掉。代码数据集有自己的过滤规则和去重设置，比如 The Stack 用 MinHash 在 5-gram 上去重，相似度门槛是 0.7（${cite(SRC.dedup,'Hugging Face 的去重博客')}）。</p>`);}
  return tx(`<p><b>With only exact copies removed, ${f.dupLeft} near copies stay in the data, red in the chart.</b></p><p>Near copies are common in real crawls. In OpenWebText2, MinHash run after removing duplicate URLs took 69M documents down to 17M (the table in ${cite(SRC.dedup,'Hugging Face’s dedup post')}).</p>`,
            `<p><b>只去完全一样的副本时，有 ${f.dupLeft} 篇几乎一样的副本留在数据里，图里是红色。</b></p><p>真实的抓取数据里，几乎一样的副本很常见。OpenWebText2 按网址去重后还有 69M 篇文档，再跑一遍 MinHash，只剩 17M 篇（${cite(SRC.dedup,'Hugging Face 去重博客里的表格')}）。</p>`);
}
const maxBin=f=>[1,2,3,4].reduce((a,k)=>f.rm[k]>f.rm[a]?k:a,1);

/* =====================================================================
   challenge
   ===================================================================== */
const chName=(b,r)=>tx(`${b} groups of ${r}`,`${b} 组，每组 ${r} 个`);
const chChip=(k,e)=>k==='b'?tx(`${e.b} groups`,`${e.b} 组`):tx(`${e.r} hashes per group`,`每组 ${e.r} 个哈希`);
function chChange(k,a,b){
  return k==='b'?tx(`${b>a?'raising':'cutting'} the groups from ${a} to ${b}`,`把组数从 ${a} ${b>a?'加':'减'}到 ${b}`)
    :tx(`${b>a?'raising':'cutting'} the hashes per group from ${a} to ${b}`,`把每组的哈希从 ${a} 个${b>a?'加':'减'}到 ${b} 个`);}
/* the summary (R1, as in train-capstone): your setting next to the cheapest one; what they share is written once, only the settings
   that differ are chips (yours highlighted), and a value that misses its goal is red */
function chSumHTML(me,best){
  const isBest=me.b===best.b&&me.r===best.r, diff=isBest?['b','r']:['b','r'].filter(k=>me[k]!==best[k]);
  const same=isBest?'':(me.b===best.b?tx(`Both use ${me.b} groups.`,`两种设置都是 ${me.b} 组。`):(me.r===best.r?tx(`Both use ${me.r} hashes per group.`,`两种设置每组都是 ${me.r} 个哈希。`):''));
  const lab={me:tx('Your setting','你的设置'),mebest:tx('Your setting, the cheapest one','你的设置，就是最省的设置'),
    best:DEF(tx('The cheapest setting','最省的设置'),tx('Of the settings that meet both goals, the one with the fewest hashes per page.','两个目标都达到的设置里，每篇网页算的哈希最少的那一种。'))};
  const cell=(name,val,bad)=>`<span class="dl-c3c${bad?' dl-over':''}"><small>${name}</small><b class="mono">${val}</b></span>`;
  const row=(e,role)=>`<li class="dl-c3r ${role}" data-role="${role}"><div class="dl-c3h"><span class="dl-c3t">${lab[role]}</span>${diff.map(k=>`<span class="dl-chip${role==='me'?' on':''}">${esc(chChip(k,e))}</span>`).join('')}</div>`+
    `<div class="dl-c3v">${cell(tx('Caught at 0.8','0.8 被抓住'),pct1(e.ph),!e.okHi)}${cell(tx('Caught at 0.5','0.5 被抓住'),pct1(e.pl),!e.okLo)}${cell(tx('Hashes per page','每篇的哈希数'),String(e.h),false)}</div></li>`;
  let h=`<div class="dl-sum"><div class="dl-sumcap">${isBest?tx('Your setting','你的设置'):tx('Your setting and the cheapest one','你的设置和最省的设置')}</div>`+(same?`<p class="small dl-same">${same}</p>`:'')+
    `<ul class="dl-c3">${row(me,isBest?'mebest':'me')}${isBest?'':row(best,'best')}</ul>`;
  /* the one change that helps most, computed (chOneChange) */
  if(!isBest){const oc=chOneChange(me), nm=oc?chChange(oc.k,oc.from,oc.v):'', toBest=oc&&oc.e.b===best.b&&oc.e.r===best.r;
    h+=`<p class="dl-next">${me.ok?(oc?tx(`The best single change is ${nm}: ${me.h-oc.e.h} fewer hashes per page${toBest?', and that is the cheapest setting':''}.`,`只改一项的话，${nm}最省，每篇网页少算 ${me.h-oc.e.h} 个哈希${toBest?'，这就是最省的设置':''}。`)
        :tx('No single change saves hashes: the cheapest setting changes both.','只改一项省不下来，最省的设置两项一起改。'))
      :(oc?tx(`One change is enough: ${nm} meets both goals${toBest?', and that is the cheapest setting':''}.`,`只改一项就够了：${nm}，两个目标都能达到${toBest?'，这就是最省的设置':''}。`)
        :tx('No single change meets both goals: the cheapest setting changes both.','只改一项达不到目标，最省的设置两项一起改。'))}</p>`;}
  h+=tx(`<p class="small dl-why">The cheapest setting uses ${best.r} hashes per group, so its curve is steep and pairs at 0.5 are rarely caught. Its ${best.b} groups give pairs at 0.8 enough chances.</p>`,
        `<p class="small dl-why">最省的设置每组 ${best.r} 个哈希，曲线够陡，相似度 0.5 的网页对很少被抓住。${best.b} 组又给了 0.8 的网页对足够多的机会。</p>`);
  return h+`</div>`;
}
/* only the settings that pass, behind a disclosure closed by default (R1) */
function chAllHTML(me){
  const ps=chPass(), n=ps.length;
  const tr=e=>{const mine=e.b===me.b&&e.r===me.r; return `<tr class="${mine?'me':''}"${mine?' data-me="1"':''}><td>${esc(chName(e.b,e.r))}</td><td>${e.h}</td><td>${pct1(e.ph)}</td><td>${pct1(e.pl)}</td></tr>`;};
  return `<details class="dl-all"><summary>${tx(`See all settings that pass (${n})`,`看全部可行的设置（${n} 种）`)}</summary><div class="dl-allb">`+
    `<p class="small">${tx('Of the 20 settings in the task card, these meet both goals. They are sorted by hashes per page, fewest first.','任务卡里的 20 种设置中，下面这几种两个目标都达到，按每篇网页的哈希数从少到多排。')}</p>`+
    `<table class="dl-tbl"><thead><tr><th>${tx('Setting','设置')}</th><th>${tx('Hashes','哈希数')}</th><th>${tx('Caught<br>at 0.8','0.8<br>被抓住')}</th><th>${tx('Caught<br>at 0.5','0.5<br>被抓住')}</th></tr></thead><tbody>${ps.map(tr).join('')}</tbody></table>`+
    (ps.some(e=>e.b===me.b&&e.r===me.r)?`<p class="small muted">${tx('Bold: your setting.','加粗的一行是你的设置。')}</p>`:'')+`</div></details>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.b,S.r), best=chBest(), isBest=me.ok&&me.h===best.h;
  let h;
  /* the verdict first: does the setting meet both goals, and is it the cheapest one */
  if(isBest) h=tx(`<p class="okline dl-mt"><b>Goal met, with the fewest hashes: ${me.h} per page.</b> No setting that meets both goals uses fewer.</p>`,`<p class="okline dl-mt"><b>达成目标，而且哈希最少：每篇网页算 ${me.h} 个。</b>两个目标都达到的设置里，没有哪种用得更少。</p>`);
  else if(me.ok) h=tx(`<p class="okline dl-mt"><b>Goal met, with ${me.h} hashes per page.</b> But a cheaper setting needs only ${best.h}.</p>`,`<p class="okline dl-mt"><b>达成目标，每篇网页算 ${me.h} 个哈希。</b>不过还有更省的设置，只要 ${best.h} 个。</p>`);
  else {
    /* what misses is red (the bad line); then the main gap in one or two sentences, as a mechanism */
    const miss=[!me.okHi?tx(`Pairs at 0.8 are caught only ${pct1(me.ph)} of the time; the goal is at least 95%.`,`相似度 0.8 的网页对只抓住 ${pct1(me.ph)}，目标是不低于 95%。`):'',
      !me.okLo?tx(`Pairs at 0.5 are caught ${pct1(me.pl)} of the time; the goal is at most 2%.`,`相似度 0.5 的网页对也抓住了 ${pct1(me.pl)}，目标是不超过 2%。`):''].filter(Boolean);
    const gap={steep:tx('Both goals are missed, so the curve is not steep enough: it climbs too slowly between 0.5 and 0.8. More hashes per group make it steeper.','两个目标都没达到，说明曲线不够陡，从 0.5 到 0.8 升得太慢。每组的哈希多了，曲线才变陡。'),
      right:tx('The curve sits too far right: at 0.8 it has not risen enough yet. More groups, or fewer hashes per group, move it left.','曲线太靠右了，到 0.8 还没升够。组数多一些，或者每组的哈希少一些，曲线都会往左移。'),
      left:tx('The curve sits too far left: it already rises at 0.5. More hashes per group, or fewer groups, move it right.','曲线太靠左了，在 0.5 就已经升起来了。每组的哈希多一些，或者组数少一些，曲线都会往右移。')}[chGap(me)];
    h=`<div class="badline dl-mt"><p><b>${tx('Not yet.','还没达到。')}</b>${tx(' ','')}${miss.join(tx(' ',''))}</p></div><p class="small dl-gap">${gap}</p>`+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the cheapest setting','直接看最省的设置')}</button>`);
  }
  if(me.ok) chReveal=true;
  if(!chReveal) return h;
  return h+chSumHTML(me,best)+chAllHTML(me);
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Six short parts. Guess how much of a crawl survives, then watch each stage at work: main text and language, quality rules, exact and near duplicates, and the mix. Start here if this is new to you.',
            '一共 6 个部分。先猜抓来的网页能留下多少，再一个个看每个环节怎么筛：抽正文和认语言、质量规则、去重，最后是配比。第一次学从这里开始。'),
  sandbox:tx('Four real situations, from FineWeb’s recipe and a crawl mostly in other languages to source code and a pipeline that skips near-duplicate removal. Pick one and see what is thrown away.',
             '四个真实情况，包括 FineWeb 的做法、大部分不是英文的网页、源代码，以及不去几乎一样的副本。选一个，看扔掉了什么。'),
  challenge:tx('Choose MinHash’s groups and hashes per group so that pages at similarity 0.8 are almost always caught and pages at 0.5 almost never, with as few hashes per page as you can.',
               '选好 MinHash 的组数和每组的哈希数，让相似度 0.8 的网页几乎都被抓住、0.5 的几乎不被抓，同时每篇网页算的哈希越少越好。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> the pages are made up, each with the few numbers the rules read; the rules and their thresholds are FineWeb’s.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Four stages in FineWeb’s order: extraction, language ID (English score of at least 0.65), five quality rules in the order the code runs them, then exact and MinHash dedup.</li>`+
 `<li>MinHash: a pair of similarity s is caught with chance ${A.ui.math('1 - (1 - s^r)^b')}. Each copy gets one seeded random number, and it is caught when that number is under the curve, so a stricter setting only ever lets more copies through.</li>`+
 `<li>The mix: ${A.ui.math('\\text{epochs} = \\frac{\\text{share} \\times 15\\text{T tokens}}{\\text{tokens on hand}}')}.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>How many pages of each kind a crawl holds (for example 330 of 1,000 not in English, 90 copies) is this lab’s assumption; the real shares are in the FineWeb paper, not checked here. The counts of every stage follow from it.</li>`+
 `<li>Each page’s numbers (English score, words, repeated lines and so on) are drawn with a seeded generator (mulberry32).</li>`+
 `<li>Each page carries one symbol ratio, the larger of its “#” per word and its “…” per word. The rule checks the two separately and drops a page when either is above 0.1, so testing the larger one gives the same result.</li>`+
 `<li>Not modeled: the URL block list that runs before extraction, the rarer rules (word length, bullets, C4’s line rules, FineWeb’s two other rules), and the replacement of e-mail and IP addresses after dedup.</li>`+
 `<li>The tokens on hand for the mix (12T, 4T, 1.5T, 0.3T) are assumed.</li>`+
 `<li>The three example pages of the dedup part (A, B and C) are made up; their 5-grams, similarity, whole-text hash and MinHash signatures are computed in the page. The example uses a simple hash (FNV-1a) and a signature of 12 values in 3 groups, with each value cut to 0–999 so it fits in a box; real pipelines keep 32-bit or 64-bit hash values (datatrove offers both).</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>Order of the stages, MinHash on 5-grams with 14 groups of 8 hashes, the PII step: ${cite(SRC.fineweb,'datatrove <code>examples/fineweb.py</code>')} and ${cite(SRC.minhash,'<code>dedup/minhash.py</code>')}, pinned to release 0.10.1.</li>`+
 `<li>Thresholds: ${cite(SRC.lang,'<code>language_filter.py</code>')} (0.65), ${cite(SRC.gopherr,'<code>gopher_repetition_filter.py</code>')} (30% repeated lines), ${cite(SRC.gopherq,'<code>gopher_quality_filter.py</code>')} (50 words; at most 0.1 “#” per word and, separately, 0.1 “…” per word; 2 of 8 stop words), ${cite(SRC.fwq,'<code>fineweb_quality_filter.py</code>')} (12% lines ending in punctuation).</li>`+
 `<li>Why near copies go too: a model repeats text it has seen many copies of (${cite(SRC.plee,'Lee et al., 2021')}), and copies waste training steps (${cite(SRC.dedup,'Hugging Face’s dedup post')}).</li>`+
 `<li>RedPajama‑V2’s counts and MinHash settings: ${cite(SRC.rp,'RedPajama-Data README')}. The Pile’s epochs: ${cite(SRC.pile,'the-pile README')}. SmolLM3’s mix: ${cite(SRC.smol3,'Hugging Face blog')}. Llama 3.1’s ~15T tokens: ${cite(SRC.l31,'its model card')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-04. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>网页是编出来的，每个网页只带规则要读的那几个数值；规则和门槛用的是 FineWeb 的。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>四个环节，顺序和 FineWeb 一样：抽正文，认语言（英文得分不低于 0.65），按代码里的顺序跑 5 条质量规则，最后去完全一样的副本和用 MinHash 去几乎一样的副本。</li>`+
 `<li>MinHash：相似度为 s 的一对网页被抓住的机会是 ${A.ui.math('1 - (1 - s^r)^b')}。每篇副本带一个固定种子生成的随机数，落在曲线下面就被抓住，所以设置越严，漏掉的副本只会越多。</li>`+
 `<li>配比：${A.ui.math('\\text{遍数} = \\frac{\\text{占比} \\times 15\\text{T 个 token}}{\\text{手里的 token 数}}')}。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>一批网页里各类网页各有多少（比如 1,000 个里有 330 个不是英文、90 个是副本），是这个 Lab 的假设；真实的比例在 FineWeb 的论文里，这里没有核对。各环节扔掉的数量都由它决定。</li>`+
 `<li>每个网页的数（英文得分、词数、重复的行等）由带种子的随机数生成器（mulberry32）产生。</li>`+
 `<li>每个网页只带一个符号比例，取“#”和“…”里多的那一种。规则对两种符号分开判断，只要有一种超过 0.1 就扔掉，所以只看多的那一种，结果也一样。</li>`+
 `<li>没有模拟的：抽正文之前按网址黑名单过滤，较少用到的规则（词长、项目符号、C4 的逐行规则、FineWeb 另外两条规则），以及去重之后替换邮箱和 IP 地址。</li>`+
 `<li>配比用到的各类数据手里有多少 token（12T、4T、1.5T、0.3T），是假设的。</li>`+
 `<li>讲去重时的三个示例网页（A、B、C）是编的。它们的 5-gram、相似度、全文哈希和 MinHash 签名，都是在页面里真算出来的。示例用的是简单的 FNV-1a 哈希，签名只有 12 个值、分 3 组，每个值取 0 到 999，好放进小格子里。真实的管线用 32 位或 64 位的哈希值，datatrove 两种都支持。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>环节的顺序、在 5-gram 上用 MinHash（14 组、每组 8 个哈希）、PII 处理：${cite(SRC.fineweb,'datatrove 的 <code>examples/fineweb.py</code>')} 和 ${cite(SRC.minhash,'<code>dedup/minhash.py</code>')}，固定在 0.10.1 版。</li>`+
 `<li>门槛：${cite(SRC.lang,'<code>language_filter.py</code>')}（0.65），${cite(SRC.gopherr,'<code>gopher_repetition_filter.py</code>')}（重复的行 30%），${cite(SRC.gopherq,'<code>gopher_quality_filter.py</code>')}（50 个词，“#”和“…”每个词各不超过 0.1 个，8 个停用词里至少 2 个），${cite(SRC.fwq,'<code>fineweb_quality_filter.py</code>')}（以标点结尾的行 12%）。</li>`+
 `<li>几乎一样的副本为什么也要去：副本多了，模型会背出原文（${cite(SRC.plee,'Lee 等，2021')}），训练步数也白花（${cite(SRC.dedup,'Hugging Face 的去重博客')}）。</li>`+
 `<li>RedPajama‑V2 的数量和 MinHash 设置：${cite(SRC.rp,'RedPajama-Data 的 README')}。The Pile 的遍数：${cite(SRC.pile,'the-pile 的 README')}。SmolLM3 的配比：${cite(SRC.smol3,'Hugging Face 博客')}。Llama 3.1 约 15T 个 token：${cite(SRC.l31,'它的 model card')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-04。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('say what each stage of a pretraining data pipeline throws away and why, choose MinHash settings for near-duplicate removal, and tell when a data mix repeats a domain too often.','说出预训练数据管线每个环节扔掉什么、为什么扔，为去重选好 MinHash 的设置，并判断一种配比会不会把某类数据重复太多遍。');
const EXITS=()=>{
  /* new case 1 (owner's review b8): the question asks how many of 100 reposts each setting misses, so the setting that is harder to
     match gives the bigger number (about 8 for FineWeb, about 60 for 9 groups of 13) and the words and the number move together */
  const rp=PofS(0.8,9,13), fw=PofS(0.8,14,8), rpMiss=Math.round((1-rp)*100), fwMiss=Math.round((1-fw)*100), m15=epochsOf('math15')[3].ep;
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why does MinHash catch a repost with one changed line, when a hash of the whole text does not?','为什么改了一行的转载，比较全文的哈希抓不到，MinHash 却能抓到？'),
   o:[tx('A changed line changes only the few 5-grams around it. Most 5-grams are still shared, so most signature numbers still agree, and very likely some group of r agrees in full.','改了一行，只会改掉它附近的几个 5-gram。大部分 5-gram 还是共有的，所以签名里大部分值仍然相同，很可能至少有一组的 r 个值全部相同。'),
      tx('MinHash skips the first and last lines of a page, where reposts usually make their changes.','MinHash 会跳过网页的第一行和最后一行，转载改动的通常就是那里。'),
      tx('MinHash compares two pages word by word, so it can count how many words differ.','MinHash 把两篇网页逐词比较，所以能数出有几个词不一样。')],a:0,
   n:[null,tx('MinHash reads every 5-gram of the page, first and last lines included. It works because most 5-grams stay the same, wherever the change is.','MinHash 读的是网页里所有的 5-gram，第一行和最后一行也算在内。它能抓到转载，是因为不管改在哪里，大部分 5-gram 都没变。'),
     tx('MinHash never compares two pages directly: it compares short signatures, so it never counts differing words. That is what makes it cheap enough for billions of pages.','MinHash 从不直接比较两篇网页，它比的是短短的签名，所以不会去数有几个词不同。正因为这样，它才便宜到能处理海量的网页。')],
   why:tx('A hash of the whole text changes completely with one changed character. MinHash looks at the set of 5-grams instead: a one-line change touches only a few of them, the similarity stays high, and the signatures still agree in most places.','全文的哈希只要改一个字就完全不同。MinHash 看的是 5-gram 的集合：改一行只碰到其中几个，相似度还是很高，两个签名大部分位置仍然相同。')},
  {q:tx('Why do more hashes per group (a larger r) make MinHash stricter?','为什么每组的哈希越多（r 越大），MinHash 判得越严？'),
   o:[tx('More hashes per group mean more groups to compare, so more pairs find a match.','每组的哈希多了，要比较的组也多了，所以更多的网页对能配上。'),
      tx('All r numbers of a group must agree, and the chance of that is s to the power r. It falls fast as r grows, and fastest for pairs that are less alike.','一组里 r 个值必须全部相同，这件事的机会是 s 的 r 次方。r 越大它掉得越快，相似度低的网页对掉得最厉害。'),
      tx('More hashes make each number more precise, so smaller differences between pages are noticed.','哈希多了，每个值都更精确，所以网页之间更小的差别也能被发现。')],a:1,
   n:[tx('The number of groups is b, a separate setting; more groups do give more chances and make MinHash looser. A larger r makes each group harder to match.','组数是 b，是另一个设置；组多了确实机会更多，判得更松。r 大了，每一组都更难全部相同。'),null,
     tx('Each number is just the smallest hash of a page’s 5-grams; it is not more or less precise. What changes with r is how many numbers must agree at once.','每个值只是这篇网页所有 5-gram 里最小的那个哈希，没有更精确或不精确之分。r 改变的是要同时相同的数有几个。')],
   why:tx(`One number agrees with chance s, so all r numbers of a group agree with chance ${A.ui.math('s^r')}. With r = 8, that is 0.17 for a pair at 0.8 and only 0.004 for a pair at 0.5. The gap widens as r grows, which is why r sets how strict MinHash is and b how many chances it gets.`,`一个值相同的机会是 s，一组的 r 个值全部相同，机会就是 ${A.ui.math('s^r')}。r = 8 时，相似度 0.8 的网页对是 0.17，0.5 的只有 0.004。r 越大，两者差得越远。所以 r 决定 MinHash 判得多严，b 决定它有几次机会。`)}],
 transfer:[
  {q:tx(`Take 100 reposts, each 0.8 similar to its original. FineWeb’s setting (14 groups of 8 hashes) misses about ${fwMiss} of them, and the missed reposts stay in the data. RedPajama‑V2 offers a setting of 9 groups of 13 hashes. Of the same 100 reposts, about how many does it miss?`,
        `拿 100 篇转载来试，每篇和原文的相似度都是 0.8。FineWeb 的设置（14 组、每组 8 个哈希）大约漏掉其中 ${fwMiss} 篇，漏掉的转载会留在数据里。RedPajama‑V2 提供的一种设置是 9 组、每组 13 个哈希。同样这 100 篇转载，它大约漏掉多少篇？`),
   o:[tx('About 1: it uses more hashes in all','约 1 篇，它用的哈希总数更多'),tx(`About ${fwMiss}, about the same as FineWeb`,`约 ${fwMiss} 篇，和 FineWeb 差不多`),tx(`About ${rpMiss}: all 13 values of a group must agree, which is much harder`,`约 ${rpMiss} 篇，一组的 13 个值都要相同，难得多`)],a:2,
   why:tx(`With 13 hashes per group, a whole group agrees far less often. The curve only rises near similarity ${fixd(thrOf(9,13),2)}, to the right of 0.8, so about ${rpMiss} of the 100 reposts are missed and stay. RedPajama provides this setting for dedup at similarity 0.8 (${cite(SRC.rp,'RedPajama-Data README')}). Yet it misses most reposts at exactly 0.8.${MS(`\\text{share missed} = (1 - 0.8^{13})^{9} = ${pct1(1-rp).replace('%','\\%')}`)}`,
          `每组 13 个哈希，整组相同难得多。曲线要到相似度 ${fixd(thrOf(9,13),2)} 左右才升起来，在 0.8 的右边，所以 100 篇转载里大约漏掉 ${rpMiss} 篇，它们都留了下来。这种设置是 RedPajama 给相似度 0.8 的去重准备的（${cite(SRC.rp,'RedPajama-Data 的 README')}）。可相似度正好是 0.8 的转载，它大半都漏掉了。${MS(`\\text{漏掉的比例} = (1 - 0.8^{13})^{9} = ${pct1(1-rp).replace('%','\\%')}`)}`)},
  {q:tx(`With math at 15% of a 15T-token run and only 0.3T on hand, the same math text is used ${fEp(m15)} times. Now a team has 0.5T tokens of code and plans a 10T-token run with code at 20% of the tokens. About how many times is each code file seen?`,`训练 15T 个 token、数学占 15%，而手里只有 0.3T 时，同一批数学文字要用 ${fEp(m15)} 遍。现在一个团队手里有 0.5T 个 token 的代码，要训练 10T 个 token，代码占 20%。每个代码文件大约会被看到几次？`),
   o:[tx('Once: 20% is a small share','1 次，20% 的占比不大'),tx('About 4 times','约 4 次'),tx('About 20 times','约 20 次')],a:1,
   why:tx(`The run needs 4 times the code on hand, so each file is seen about 4 times, a little more than The Pile’s most repeated set (3 times).${MS('\\frac{20\\% \\times 10\\text{T}}{0.5\\text{T}} = 4')}`,`要用的代码是手里的 4 倍，所以每个文件大约被看到 4 次，比 The Pile 重复最多的 3 遍还多一点。${MS('\\frac{20\\% \\times 10\\text{T}}{0.5\\text{T}} = 4')}`)},
  {q:tx('You run FineWeb’s English web filters on a crawl of source code. What happens to the good source files?','你用 FineWeb 给英文网页定的过滤规则去筛一批源代码。好的源文件会怎样？'),
   o:[tx('Almost all stay: code is clean, well-formed text','几乎都留下，代码是干净、格式规范的文字'),tx('Many are thrown away: code has a low English score, few stop words and few lines ending in a full stop','很多被扔掉：代码的英文得分低，停用词少，以句号结尾的行也少'),tx('Only the copies are removed, since the language and quality stages skip code','只有副本被去掉，认语言和质量规则会跳过代码')],a:1,
   why:tx('Every rule of FineWeb was written for English prose. Code fails the language stage and the stop-word and punctuation rules, so a code dataset needs filters of its own.','FineWeb 的每条规则都是为英文文章写的。代码过不了认语言这一关，也过不了停用词和标点那两条规则，所以代码数据集要有自己的过滤规则。'),
   go:()=>{setMode('sandbox'); byLearner(()=>setState(cloneS(PRESETS[2].s)));},goLabel:tx('See it in Scenarios','在应用场景里看')}],
 practice:[
  {id:'file-dedup',d:tx('Group the files of a directory tree by identical content: hash first, then compare bytes. The exact-copy half of dedup, as code.','把一棵目录树里内容完全相同的文件分组：先比哈希，再逐字节确认。也就是去重里“去完全一样的副本”那一半，写成代码。')},
  {id:'llm-fundamentals',d:tx('Question 9: the compute-optimal number of training tokens for a budget of FLOPs, the amount of data your pipeline must leave you with.','第 9 题：给定算力预算，算出最优的训练 token 数，也就是数据管线筛完之后至少要剩下多少数据。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  const pr=tx(`<p>A token is the unit a model reads, a word or part of a word; the lab ${labLink('tokenization-cost')} shows how text is cut into tokens. A pretraining run reads trillions of them, and the lab ${labLink('scaling-law')} shows how many tokens a model of a given size should be trained on. This lab is about where those tokens come from.</p>`,
            `<p>token 是模型读文字的单位，一个词或词的一部分；${labLink('tokenization-cost')} 那个 Lab 讲文字怎么切成 token。一次预训练要读的 token 多达几 T 个，${labLink('scaling-law')} 那个 Lab 讲多大的模型该训练多少个 token。这个 Lab 讲的是这些 token 从哪里来。</p>`);
  return A.shell.header({id:ID,world:'training',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Data pipeline','数据管线'),title:l?L(l.q):tx('What gets thrown away before pretraining?','预训练之前，数据要扔掉哪些？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: tokens and training runs','前置知识：token 和训练规模'),primerPop:pr,descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage dl-stage" id="dl-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="dl-st-title"></div><div class="meter-sub" id="dl-st-sub"></div></div><div class="stage-actions"><span class="status" id="dl-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph dl-graph" id="dl-graph">`+
    `<div class="dl-phone" id="dl-phone"></div>`+
    `<div class="dl-fbox" id="dl-fbox"><div class="dl-cap" id="dl-f-cap"></div><div class="dl-cvwrap" id="dl-cvwrap"><canvas id="dl-cv" role="img"></canvas></div><div class="dl-leg" id="dl-f-leg"></div><div class="dl-player" id="dl-player"></div></div>`+
    `<div class="dl-xbox" id="dl-xbox" hidden></div>`+
    /* the caption and the plot in one box (.dl-cmain), the legend after it: on a wide stage part 4 puts the plot beside the
       signatures and runs the legend under both (data-pipeline.css); stacked, it reads as before */
    `<div class="dl-cbox" id="dl-cbox" hidden><div class="dl-cmain"><div class="dl-cap" id="dl-c-cap"></div><div class="dl-crow"><div class="dl-cplot"><div class="dl-axy" id="dl-c-yt"></div><div class="dl-curve" id="dl-curve"></div><div class="dl-axx" id="dl-c-xt"></div></div></div></div><div class="dl-leg dl-leg-side" id="dl-c-leg"></div></div>`+
    `<div class="dl-mbox" id="dl-mbox" hidden><div class="dl-cap" id="dl-m-cap"></div><div class="dl-mix" id="dl-mix"></div></div>`+
   `</div>`+
   `<div class="metrics dl-metrics" id="dl-metrics"></div>`+
   `<details class="math" id="dl-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="dl-mathbody"></div></details></div>`;
}
const CORP=[['web',{en:'English crawl',zh:'英文网页'}],['foreign',{en:'Mostly other languages',zh:'多数不是英文'}],['code',{en:'Source code',zh:'源代码'}]];
function controlsHTML(){
  const onoff=[['1',tx('On','开')],['0',tx('Off','关')]];
  return `<aside class="panel controls" id="dl-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="dl-lock" hidden></p>`+
   `<div class="lockable" id="dl-ctl-corpus-wrap">${A.ui.seg({id:'dl-ctl-corpus',key:'corpus',label:tx('The crawl','抓来的数据'),options:CORP.map(c=>[c[0],L(c[1])]),value:S.corpus})}</div>`+
   `<div class="lockable" id="dl-ctl-thr-wrap">${thrCtl('dl-ctl-thr',false)}</div>`+
   `<div class="lockable dl-grp" id="dl-ctl-rules-wrap"><div class="dl-gh">${tx('Quality rules','质量规则')}</div>${ruleSw(false)}</div>`+
   `<div class="lockable dl-grp" id="dl-ctl-dup-wrap"><div class="dl-gh">${tx('Dedup','去重')}</div>`+
    A.ui.seg({id:'dl-ctl-exact',key:'exact',label:tx('Exact copies (hash)','完全一样的副本（哈希）'),options:onoff,value:String(S.exact)})+
    A.ui.seg({id:'dl-ctl-mh',key:'mh',label:tx('Near copies (MinHash)','几乎一样的副本（MinHash）'),options:onoff,value:String(S.mh)})+
    `<div id="dl-ctl-br">${bCtl('dl-ctl-b',false)}${rCtl('dl-ctl-r',false)}</div></div>`+
   `<div class="lockable" id="dl-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import hashlib
from collections import defaultdict

B, R = 14, 8                  # FineWeb: 14 groups of 8 hashes

def shingles(text, n=5):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def h(seed, s):
    d = hashlib.sha1(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(d[:8], "big")

def signature(text):
    sh = shingles(text) or {text}
    return [min(h(k, s) for s in sh) for k in range(B * R)]

def near_duplicates(docs):
    """docs: {id: text}. Returns the ids to drop; keeps the first."""
    buckets = defaultdict(list)
    for i, text in docs.items():
        sig = signature(text)
        for b in range(B):
            buckets[(b, tuple(sig[b * R:(b + 1) * R]))].append(i)
    return {i for ids in buckets.values() for i in ids[1:]}

a = " ".join(f"word{i}" for i in range(300))
docs = {1: a, 2: "Reposted: " + a, 3: a[:900] + " new ending"}
print(near_duplicates(docs))  # {2}: 3 shares only ~40% of 5-grams`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="dl-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="dl-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('MinHash with groups: cut each page into 5-grams, keep the smallest hash for each of 112 hash functions, cut the signature into 14 groups of 8, and drop a page when one of its groups matches an earlier page’s.','带分组的 MinHash：把每篇网页切成 5-gram，对 112 个哈希函数各记下最小的哈希值，把签名切成 14 组、每组 8 个，只要有一组和前面某篇网页的相同，就扔掉这一篇。')}</p><pre class="code" id="dl-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="dl-prod-card"><h3>${A.t('cProd')}</h3><ul class="dl-prod">`+
   tx(`<li>FineWeb is built with datatrove. Every filter writes the pages it drops to a folder of its own (<code>removed/1_url</code> to <code>6_fineweb_qual</code>), so a team can count and read what each stage throws away. Before extraction a URL filter drops block-listed domains; after MinHash, e-mail and IP addresses are replaced, not removed (${cite(SRC.fineweb,'<code>examples/fineweb.py</code>')}).</li>`+
      `<li>DCLM runs the same kind of rules (from RefinedWeb), then a fastText classifier that scores how much a page looks like good instruction data (${cite(SRC.dclm,'<code>fasttext_filter.yaml</code>')}).</li>`+
      `<li>RedPajama‑V2 ships quality signals and MinHash signatures with every page instead of one filtered set, so each team picks its own thresholds; for a similarity of 0.8 it gives 9 groups of 13 (${cite(SRC.rp,'RedPajama-Data README')}).</li>`+
      `<li>FineWeb 2 deduplicates each language over all snapshots at once and keeps the size of every duplicate cluster. Pages are then repeated by the size of their cluster, up to 8 times. A cluster of 1,000 pages or more is kept once, because very large clusters tend to be low quality (${cite(SRC.fw2,'fineweb-2 README')}).</li>`+
      `<li>Dolma tags pages with the same rule families (Gopher, C4, PII) and filters on the tags afterwards (${cite(SRC.dolma,'<code>docs/taggers.md</code>')}).</li>`,
      `<li>FineWeb 是用 datatrove 做出来的。每个过滤器都把扔掉的网页写进自己的文件夹（<code>removed/1_url</code> 到 <code>6_fineweb_qual</code>），团队可以数一数、读一读每个环节扔了什么。抽正文之前，先按网址黑名单扔掉一批网站；MinHash 之后，邮箱和 IP 地址被替换掉，而不是整篇删除（${cite(SRC.fineweb,'<code>examples/fineweb.py</code>')}）。</li>`+
      `<li>DCLM 用的是同一类规则（来自 RefinedWeb），之后再用一个 fastText 分类器给网页打分，看它有多像好的问答数据（${cite(SRC.dclm,'<code>fasttext_filter.yaml</code>')}）。</li>`+
      `<li>RedPajama‑V2 不给一份筛好的数据，而是给每个网页附上质量信号和 MinHash 签名，让各团队自己定门槛；相似度 0.8 对应的设置是 9 组、每组 13 个（${cite(SRC.rp,'RedPajama-Data 的 README')}）。</li>`+
      `<li>FineWeb 2 把每种语言所有快照放在一起去重，并记下每一簇重复网页有多大。之后按簇的大小，让网页多出现几次，最多 8 次。1,000 篇以上的超大簇只出现 1 次，因为太大的簇质量往往差（${cite(SRC.fw2,'fineweb-2 的 README')}）。</li>`+
      `<li>Dolma 先用同几类规则（Gopher、C4、PII）给网页打标签，之后再按标签过滤（${cite(SRC.dolma,'<code>docs/taggers.md</code>')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.pfw,'The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale')}<span>${tx('Penedo et al., 2024 (arXiv 2406.17557): how FineWeb was built and tested, stage by stage.','Penedo 等，2024（arXiv 2406.17557）：FineWeb 怎么一步步做出来、怎么检验。')}</span></li>`+
   `<li>${cite(SRC.prw,'The RefinedWeb Dataset for Falcon LLM')}<span>${tx('Penedo et al., 2023 (arXiv 2306.01116): web data alone, filtered and deduplicated hard, as pretraining data.','Penedo 等，2023（arXiv 2306.01116）：只用网页数据，经过严格过滤和去重，来做预训练。')}</span></li>`+
   `<li>${cite(SRC.plee,'Deduplicating Training Data Makes Language Models Better')}<span>${tx('Lee et al., 2021 (arXiv 2107.06499): near-duplicate removal and its effect on memorization.','Lee 等，2021（arXiv 2107.06499）：去掉几乎一样的副本，以及它对模型背诵原文的影响。')}</span></li>`+
   `<li>${cite(SRC.pdc,'Scaling Data-Constrained Language Models')}<span>${tx('Muennighoff et al., 2023 (arXiv 2305.16264): what repeating data for several epochs is worth.','Muennighoff 等，2023（arXiv 2305.16264）：数据重复用几遍还值不值。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="dl-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="dl-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="dl-app m2" id="dl-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="dl-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how much of the data each stage throws away.','选一个场景，看看每个环节扔掉了多少数据。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="dl-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const fw=chEval(CH.start.b,CH.start.r), fwMiss=Math.round((1-fw.ph)*100), fwLo=Math.round(fw.pl*100);
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Set up MinHash for the next crawl','为下一批网页设置 MinHash')}</h3>`+
   `<p>${tx(`In the next crawl, many reposts are only about 0.8 similar to their originals. FineWeb’s setting (14 groups of 8 hashes) misses about ${fwMiss}% of such pairs. It also catches about ${fwLo}% of the heavily edited versions (similarity 0.5), which should stay. Choose the number of groups and the hashes per group again. Every hash is computed for every page, so use as few hashes per page as you can.`,`下一批网页里，很多转载和原文的相似度只有 0.8 左右。用 FineWeb 的设置（14 组、每组 8 个哈希），这样的网页对约有 ${fwMiss}% 会漏掉。改动很大的版本（相似度 0.5）本该留下，也有约 ${fwLo}% 被当成了重复。请重新选组数和每组的哈希数。每个哈希都要给每一篇网页算一遍，所以每篇网页算的哈希越少越好。`)}</p>`+
   `<ul class="dl-asm small"><li>${tx('Pairs at similarity 0.8 (reposts) must be caught at least 95% of the time.','相似度 0.8 的网页对（转载）至少 95% 要被抓住。')}</li><li>${tx('Pairs at similarity 0.5 may be caught at most 2% of the time. In such a pair one page is a heavily edited version of the other: they share about two thirds of their 5-grams and should stay as two pages.','相似度 0.5 的网页对，最多只能有 2% 被抓住。这样的两篇，一篇是另一篇改动很大的版本，大约三分之二的 5-gram 相同，应当当成两篇网页留下。')}</li></ul>`+
   A.ui.seg({id:'dl-ic-chb',key:'chb',label:tx('Groups b','组数 b'),options:CH.Bs.map(n=>[String(n),String(n)]),value:String(S.b),inline:true})+
   A.ui.seg({id:'dl-ic-chr',key:'chr',label:tx('Hashes per group r','每组的哈希数 r'),options:CH.Rs.map(n=>[String(n),String(n)]),value:String(S.r),inline:true})+
   `<div class="goal mono dl-goal" id="dl-ch-g1"><span>${tx('Caught at 0.8','0.8 被抓住')}</span><span><b id="dl-ch-hi"></b>${tx(' (goal: at least 95%)','（目标：不低于 95%）')}</span></div>`+
   `<div class="goal mono dl-goal" id="dl-ch-g2"><span>${tx('Caught at 0.5','0.5 被抓住')}</span><span><b id="dl-ch-lo"></b>${tx(' (goal: at most 2%)','（目标：不超过 2%）')}</span></div>`+
   `<div class="goal mono dl-goal" id="dl-ch-g3"><span>${tx('Hashes per page','每篇网页的哈希数')}</span><span><b id="dl-ch-h"></b>${tx(' (fewer is better)','（越少越好）')}</span></div>`+
   submit+`<div id="dl-ch-res" class="dl-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const id=STEPS[step].id;
  if(id==='mission') return !pred?tx('Pick an answer to continue, or Skip.','选一个答案才能继续，也可以跳过。'):tx('Let the run finish to continue (“Fast” takes about 10 seconds).','等动画播完才能继续（选“快”约 10 秒）。');
  const w={lang:{en:'Move the threshold to continue.',zh:'调一下门槛，才能继续。'},rules:{en:'Turn a rule off to continue.',zh:'关掉一条规则，才能继续。'},
    dedup:{en:'Pick “Only exact copies” to continue.',zh:'选“只去完全一样的”，才能继续。'},mix:{en:'Pick another mix to continue.',zh:'换一种配比，才能继续。'}}[id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#dl-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#dl-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(s.id==='mission'?PL.watched.has(F0().sig):(s.id==='mix'||s.id==='dedup'||!!observed[step]));
    if(isP('dedup')){$$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lvDup)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=triedDup.has(v)?lvdRes(v):'';});
      const br=$('#dl-ic-br',rail); if(br) br.hidden=lvDup!=='both';}
    if(isP('mix')) $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.mix)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=triedMix.has(v)||v===S.mix?mixRes(v):'';});
    if(isP('mission')){const lock=PL.watched.has(F0().sig); $$('[data-pred]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.pred===pred)); b.disabled=lock;}); const sv=$('.dl-saved',rail); if(sv) sv.hidden=lock;}
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const done=PL.watched.has(fates(S).sig);
    const ins=$('#dl-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+(done?scenHTML(cur):`<p class="small muted">${tx('The result appears once the run ends.','动画播完后，这里给出结果。')}</p>`):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.b,S.r), hi=$('#dl-ch-hi',rail); if(!hi) return;
    hi.textContent=pct1(e.ph); $('#dl-ch-lo',rail).textContent=pct1(e.pl); $('#dl-ch-h',rail).textContent=String(e.h);
    const g1=$('#dl-ch-g1',rail), g2=$('#dl-ch-g2',rail); g1.classList.toggle('met',e.okHi); g1.classList.toggle('dl-over',!e.okHi); g2.classList.toggle('met',e.okLo); g2.classList.toggle('dl-over',!e.okLo);
    $$('.seg-ctl[data-key="chb"]',rail).forEach(g=>A.ui.setPressed(g,String(S.b))); $$('.seg-ctl[data-key="chr"]',rail).forEach(g=>A.ui.setPressed(g,String(S.r)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#dl-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage: the funnel (a canvas of 1,000 squares and four bins), the MinHash curve, the mix
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':'sb');
const funnelView=v=>v==='funnel'||v==='sb';
const runHidden=()=>isP('mission')&&!pred;          // R3.8: the run starts once the guess is in
const stageName=k=>[null,tx('main text','抽正文'),tx('language','认语言'),tx('quality rules','质量规则'),tx('dedup','去重')][k];
const SPDS=[['slow',{en:'Slow',zh:'慢'}],['mid',{en:'Medium',zh:'中'}],['fast',{en:'Fast',zh:'快'}]];
function colors(){const cs=getComputedStyle(app), g=(k,d)=>cs.getPropertyValue(k).trim()||d;
  return {ink:g('--ink','#0D141E'),ink2:g('--ink-2','#3C4858'),ink3:g('--ink-3','#687485'),line:g('--line','#CBD3DD'),red:g('--c-oom','#CF2F2A'),font:g('--font-ui','sans-serif'),mono:g('--font-mono','monospace')};}
function layout(W){
  const n=N, phone=W<400, cols=40, rows=Math.ceil(n/cols), my=22;
  let cm, bx, bw, by0;
  if(!phone){cm=Math.max(5,Math.min(9,Math.floor(W*0.44/cols))); bx=cols*cm+24; bw=W-bx; by0=0;}
  else {cm=Math.max(5,Math.floor(W/cols)); bx=0; bw=W; by0=my+rows*cm+16;}
  const cb=3, bcols=Math.max(1,Math.floor(bw/cb)), brows=Math.ceil(n/bcols), bh=brows*cb, lab=18, gap=10;
  const bins=[1,2,3,4].map((k,j)=>({k,x:bx,y:by0+j*(lab+bh+gap)+lab,w:bcols*cb,h:bh}));
  return {phone,cols,rows,cm,my,bins,cb,brows,lab,H:Math.ceil(Math.max(my+rows*cm+4,by0+4*(lab+bh+gap))),W};
}
const ease=p=>p<0?0:(p>1?1:p*p*(3-2*p));
/* each dropped page leaves the grid during its stage's unit of g, in index order, and lands in its bin's next slot */
function plan(f){
  const slot=new Array(N).fill(-1), cnt=[0,0,0,0,0];
  for(let i=0;i<N;i++){const s=f.st[i]; if(s<5) slot[i]=cnt[s]++;}
  return {slot,cnt};
}
let CVS={lay:null,px:null,py:null,sz:null,f:null,pl:null};
function drawFunnel(){
  const cv=$('#dl-cv',app), wrap=$('#dl-cvwrap',app); if(!cv||!wrap) return;
  const W=Math.max(280,Math.floor(wrap.clientWidth||600)), lay=layout(W), dpr=Math.min(2,window.devicePixelRatio||1);
  if(cv._w!==W||cv._h!==lay.H||cv._d!==dpr){cv.width=Math.round(W*dpr); cv.height=Math.round(lay.H*dpr); cv.style.width=W+'px'; cv.style.height=lay.H+'px'; cv._w=W; cv._h=lay.H; cv._d=dpr;}
  const ctx=cv.getContext('2d'), C=colors(), f=PL.f||fates(S), pl=plan(f), g=runHidden()?0:PL.g, end=g>=4-1e-9;
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,lay.H);
  const px=new Float32Array(N), py=new Float32Array(N), sz=new Float32Array(N), landed=[0,0,0,0,0];
  let left=0;
  const hi=mode==='guided'&&STEPS[step].hi?STEPS[step].hi:[];
  for(let i=0;i<N;i++){
    const gx=(i%lay.cols)*lay.cm, gy=lay.my+Math.floor(i/lay.cols)*lay.cm, s=f.st[i];
    let x=gx, y=gy, z=lay.cm-1, p=0;
    if(s<5){const q=pl.cnt[s]>1?pl.slot[i]/(pl.cnt[s]-1):0, t0=(s-1)+0.55*q; p=ease((g-t0)/0.45);
      if(p>0){const b=lay.bins[s-1], j=pl.slot[i], bxp=b.x+Math.floor(j/lay.brows)*lay.cb, byp=b.y+(j%lay.brows)*lay.cb; x=gx+(bxp-gx)*p; y=gy+(byp-gy)*p; z=(lay.cm-1)+((lay.cb-1)-(lay.cm-1))*p;}
      if(p>=1) landed[s]++;}
    if(p<1) left++;
    const red=f.red[i]&&((s<5&&p>=1)||(s===5&&end));
    ctx.fillStyle=red?C.red:(p>0?C.ink3:C.ink2); ctx.globalAlpha=(p>0&&!red)?0.55:1;
    ctx.fillRect(x,y,Math.max(1,z),Math.max(1,z));
    px[i]=x; py[i]=y; sz[i]=Math.max(lay.cb,z);}
  ctx.globalAlpha=1; ctx.textBaseline='alphabetic';
  ctx.font=`650 12px ${C.font}`; ctx.fillStyle=C.ink;
  ctx.fillText(tx(`${fint(left)} of 1,000 pages left`,`1,000 个网页，还剩 ${fint(left)} 个`),0,14);
  lay.bins.forEach(b=>{const on=hi.includes(b.k);
    ctx.font=`${on?700:500} 12px ${C.font}`; ctx.fillStyle=on?C.ink:C.ink2;
    const t=tx(`${b.k}. ${binName(b.k)}`,`${b.k}. ${binName(b.k)}`); ctx.fillText(t,b.x,b.y-6);
    const w=ctx.measureText(t).width; ctx.font=`600 12px ${C.mono}`; ctx.fillText(String(landed[b.k]),b.x+w+8,b.y-6);
    ctx.strokeStyle=C.line; ctx.lineWidth=1; ctx.strokeRect(b.x-0.5,b.y-0.5,b.w+1,b.h+1);});
  CVS={lay,px,py,sz,f,landed,left};
  cv.setAttribute('aria-label',tx(`1,000 crawled pages as squares; ${left} still in. Bins: no main text ${landed[1]}, not English ${landed[2]}, low quality ${landed[3]}, duplicates ${landed[4]}.`,`1,000 个网页，每个是一个小方块，还剩 ${left} 个。抽不出正文 ${landed[1]} 个，不是英文 ${landed[2]} 个，质量差 ${landed[3]} 个，重复 ${landed[4]} 个。`));
  drawPlayer();
}
function kindText(d,D){
  const ln=LANGN[d.lang]?L(LANGN[d.lang]):'', o=d.dupOf>=0?d.dupOf+1:0;
  return {nomain:tx('A login page or a photo gallery: no main text','登录页或图片页，没有正文'),foreign:tx(`A page in ${ln}`,`一篇${ln}网页`),mixed:tx(`English mixed with ${ln}`,`英文夹着${ln}`),
    enlow:tx('A good English page full of code and numbers','一篇好的英文网页，夹着很多代码和数字'),rep:tx('One line repeated many times','同一行重复了很多遍'),short:tx('A few words, such as an error page','只有几个词，比如一个报错页'),
    sym:tx('A list of hashtags or “…”','一串话题标签或省略号'),kw:tx('A list of keywords','一串关键词'),menu:tx('Menu items, one per line','菜单项，一行一个'),seo:tx('Stitched-together SEO text','拼凑出来的 SEO 文章'),
    uniq:tx('An article','一篇文章'),alike:tx(`A heavily edited version of page ${o} (similarity ${fixd(d.sim,2)}); FineWeb’s setting should keep it`,`第 ${o} 个网页改动很大的版本（相似度 ${fixd(d.sim,2)}），按 FineWeb 的设置应当留下`),
    exact:tx(`An exact copy of page ${o}`,`第 ${o} 个网页的完全一样的副本`),near:tx(`A repost of page ${o} with a line changed (similarity ${fixd(d.sim,2)})`,`第 ${o} 个网页的转载，改了一行（相似度 ${fixd(d.sim,2)}）`),
    border:tx(`A copy of page ${o} edited in several places (similarity ${fixd(d.sim,2)})`,`第 ${o} 个网页改了好几处的版本（相似度 ${fixd(d.sim,2)}）`),
    code:tx('A source file','一个源文件'),gen:tx('A generated file: minified code or data','自动生成的文件，比如压缩过的代码或数据'),cexact:tx(`An exact copy of source file ${o}`,`第 ${o} 个文件的完全一样的副本`),cnear:tx(`A near copy of source file ${o} (similarity ${fixd(d.sim,2)})`,`第 ${o} 个文件几乎一样的副本（相似度 ${fixd(d.sim,2)}）`)}[d.kind];
}
function fateText(i,f,D){
  const d=D[i], s=f.st[i], o=d.dupOf+1;
  if(s===1) return tx('Thrown away at stage 1: no main text.','在第 1 个环节被扔掉：抽不出正文。');
  if(s===2) return tx(`Thrown away at stage 2: its English score ${fixd(d.en,2)} is under ${fixd(S.thr,2)}.`,`在第 2 个环节被扔掉：英文得分 ${fixd(d.en,2)}，低于 ${fixd(S.thr,2)}。`);
  if(s===3){const R=RULES.find(x=>x.k===f.why[i]); return tx(`Thrown away at stage 3 by “${L(R.name)}”: ${L(R.item)} ${R.v(d)}, needs ${L(R.need)}.`,`在第 3 个环节被“${L(R.name)}”扔掉，${L(R.item)}是 ${R.v(d)}，要求 ${L(R.need)}。`);}
  if(s===4) return f.why[i]==='exact'?tx(`Thrown away at stage 4: the same hash as page ${o}.`,`在第 4 个环节被扔掉：和第 ${o} 个网页的哈希相同。`):tx(`Thrown away at stage 4: MinHash matched it with page ${o}.`,`在第 4 个环节被扔掉：MinHash 认为它和第 ${o} 个网页重复。`);
  return tx('Kept.','留了下来。');
}
function tipHTML(i){
  const f=CVS.f||fates(S), D=corpus(S.corpus).docs, d=D[i], s=f.st[i];
  const row=(k,v)=>`<tr><th>${esc(k)}</th><td class="mono">${esc(v)}</td></tr>`;
  const t=d.noMain?row(tx('Main text','正文'),tx('none','没有')):row(tx('English score','英文得分'),fixd(d.en,2))+row(tx('Words','词数'),fint(d.words))+row(tx('Repeated lines','重复的行'),pct(d.rep))+row(tx('“#” or “…” per word (the larger)','每个词摊到的“#”或“…”（多的那种）'),fixd(d.sym,2))+row(tx('Stop words present','出现的停用词'),tx(`${d.stop} of 8`,`8 个里有 ${d.stop} 个`))+row(tx('Lines ending in punctuation','以标点结尾的行'),pct(d.punct));
  const end=PL.g>=4-1e-9, shown=s<5||end;
  let red='';
  if(f.red[i]&&shown) red=`<p class="dl-ttr">${esc(s<5?tx('A wrong call: a good page thrown away.','判错了：好网页被扔掉了。'):(COPY.has(d.kind)?tx('A wrong call: a near copy that stayed.','判错了：几乎一样的副本留了下来。'):(d.kind==='mixed'?tx('A wrong call: a page in another language got through.','判错了：别的语言的网页混了进来。'):tx('A wrong call: junk that stayed.','判错了：垃圾网页留了下来。'))))}</p>`;
  return `<div class="dl-tt"><b>${esc(tx(`Page ${i+1}`,`第 ${i+1} 个网页`))}</b><p>${esc(kindText(d,D))}</p><table>${t}</table><p>${esc(shown?fateText(i,f,D):tx('Not decided yet: the run has not reached its stage.','还没轮到它所在的环节。'))}</p>${red}</div>`;
}
function hitTest(ev){
  const cv=$('#dl-cv',app); if(!cv||!CVS.px) return -1; const r=cv.getBoundingClientRect(), x=ev.clientX-r.left, y=ev.clientY-r.top;
  let best=-1, bd=1e9; for(let i=0;i<N;i++){const s=CVS.sz[i], dx=x-(CVS.px[i]+s/2), dy=y-(CVS.py[i]+s/2), dd=dx*dx+dy*dy; if(Math.abs(dx)<=s/2+1.5&&Math.abs(dy)<=s/2+1.5&&dd<bd){bd=dd; best=i;}}
  return best;
}
function drawPlayer(){
  const pl=$('#dl-player',app); if(!pl) return;
  const hide=runHidden(); pl.hidden=hide; if(hide) return;
  const g=PL.g, k=Math.min(4,Math.max(1,Math.ceil(g-1e-9)||1)), done=g>=4-1e-9;
  const btn=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(done?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  const cnt=done?tx('All 4 stages done','4 个环节都走完了'):tx(`Stage ${k} of 4: ${stageName(k)}`,`第 ${k} 个环节（共 4 个）：${stageName(k)}`);
  const sig=[PL.on,PL.spd,Math.round(g*100),A.lang()].join('|'); if(pl._sig===sig) return; pl._sig=sig;
  const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()){scrub.value=Math.round(g*100); A.ui.setRange(scrub); pl.querySelector('[data-act="play"]').textContent=btn;
    $$('.seg-ctl[data-key="spd"]',pl).forEach(gg=>A.ui.setPressed(gg,PL.spd)); pl.querySelector('.dl-count').textContent=cnt; return;}
  pl._built=A.lang();
  pl.innerHTML=`<button type="button" class="btn ghost sm dl-play" data-act="play" data-tipx="${esc(tx('Play the run at the chosen speed; press again to pause. Once it has ended, this button replays it from stage 1.','按选定的速度播放；再按一次暂停。播完以后，这个按钮从第 1 个环节重新播放。'))}">${btn}</button>`+
    A.ui.seg({id:'dl-spd',key:'spd',label:tx('Speed','速度'),options:SPDS.map(x=>[x[0],L(x[1])]),value:PL.spd})+
    `<div class="dl-scrub"><input type="range" min="0" max="400" step="1" value="${Math.round(g*100)}" data-key="scrub" aria-label="${esc(tx('How far the run has got','动画播到哪里'))}"><span class="dl-count mono">${cnt}</span></div>`;
  A.ui.initRanges(pl);
}
/* R3.20: a picture watched to the end shows its end at once; a new one replays from the first stage that changed; Replay replays all */
function showRun(){
  if(runHidden()){stopAnim(); PL.g=0; PL.sig=null; PL.f=fates(S); return;}
  const f=fates(S); if(PL.sig===f.sig){PL.f=f; return;}   // the same picture (also after a remount): keep where it is
  const prev=PL.f, prevSig=PL.sig; PL.f=f; PL.sig=f.sig;
  if(PL.watched.has(f.sig)){stopAnim(); PL.g=4; return;}
  let from=1;
  if(prev&&prevSig&&prev!==f&&corpusOfSig(prev)===S.corpus){from=5; for(let i=0;i<N;i++) if(prev.st[i]!==f.st[i]) from=Math.min(from,prev.st[i],f.st[i]); if(from>4) from=4;}
  PL.g=Math.min(PL.g,from-1); if(!prevSig) PL.g=0; play(true);
}
const corpusOfSig=f=>f._corpus;
function play(keep){
  if(!keep&&PL.g>=4-1e-9) PL.g=0;
  PL.on=true; PL.last=0;
  if(A.reduceMotion()){finishRun(); return;}
  if(!PL.raf) PL.raf=requestAnimationFrame(tick); drawPlayer();
}
function stopAnim(){PL.on=false; if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0;}
function pause(){stopAnim(); drawPlayer();}
function tick(t){
  PL.raf=0; if(!app||!PL.on) return;
  const dt=PL.last?Math.min(100,t-PL.last):16; PL.last=t;
  PL.g=Math.min(4,PL.g+dt/DUR[PL.spd]*4);
  if(PL.g>=4){finishRun(); return;}
  if(funnelView(viewNow())) drawFunnel(); PL.raf=requestAnimationFrame(tick);
}
function finishRun(){stopAnim(); PL.g=4; if(PL.sig) PL.watched.add(PL.sig); wantObs=true; render(); if(isP('mission')) revealSoon(); wantObs=false;}

/* the MinHash curve: x = similarity of a pair, y = chance it is caught; in part 4 every pair that reaches dedup is a dot at (s, u),
   caught when it lies under the curve (u is the pair's seeded random number) */
function curveSVG(o){
  /* laid out for the width it is drawn at (review codex-b9v3, 应修 4): the viewBox is the box's own width, so 11 px text stays 11 px
     on a 390 px phone; a narrow chart gets fewer ticks, and the axis titles are HTML outside the SVG (#dl-c-yt, #dl-c-xt).
     In the challenge (o.goals) a strip under the plot zooms into 0–10%, where the goal at 0.5 (at most 2%) can be seen (R5.10).
     o.ph, when given, is the plot's height: part 4 on a wide stage draws it shorter, beside the signatures (drawCurve), and keeps
     every tick and grid line of the full-width chart (there is room for them; only a phone's chart is narrow) */
  const zoom=!!o.goals, W=Math.max(260,Math.round(o.w||560)), nar=W<440&&!o.ph, ZG=14, ZH=nar?56:64, ZM=0.1, PH=o.ph||(nar?170:210);
  const L0=nar?36:44,R0=nar?12:16,T0=10,B0=22,H=T0+PH+(zoom?ZG+ZH:0)+B0,x0=0.4,x1=1.0, X=s=>L0+(s-x0)/(x1-x0)*(W-L0-R0), Y=p=>T0+(1-p)*PH, f2=v=>v.toFixed(1);
  const Z0=T0+PH+ZG, Yz=p=>Z0+(1-p/ZM)*ZH;
  let g='';
  (nar?[0,0.5,1]:[0,0.25,0.5,0.75,1]).forEach(p=>{g+=`<line class="dl-gl" x1="${L0}" x2="${W-R0}" y1="${f2(Y(p))}" y2="${f2(Y(p))}"/><text class="dl-ax" x="${L0-5}" y="${f2(Y(p)+4)}" text-anchor="end">${Math.round(p*100)}%</text>`;});
  (nar?[4,6,8,10]:[4,5,6,7,8,9,10]).forEach(k=>{const s=k/10; g+=`<line class="dl-tk" x1="${f2(X(s))}" x2="${f2(X(s))}" y1="${H-B0}" y2="${H-B0+4}"/><text class="dl-ax" x="${f2(X(s))}" y="${H-B0+16}" text-anchor="${k===10?'end':'middle'}">${fixd(s,1)}</text>`;});
  const path=(b,r,Yf)=>{Yf=Yf||Y; let d=''; for(let k=0;k<=120;k++){const s=x0+(x1-x0)*k/120; d+=(k?'L':'M')+f2(X(s))+' '+f2(Yf(PofS(s,b,r)));} return d;};
  if(o.ref) g+=`<path class="dl-ref" d="${path(14,8)}"/><text class="dl-reft" x="${f2(X(0.66))}" y="${f2(Y(PofS(0.66,14,8))-8)}" text-anchor="end">${esc(tx('FineWeb’s setting','FineWeb 的设置'))}</text>`;
  if(o.best) g+=`<path class="dl-bestc" d="${path(o.best.b,o.best.r)}"/>`;
  if(o.mh) g+=`<path class="dl-cur" d="${path(o.b,o.r)}"/>`;
  else g+=`<path class="dl-cur" d="M${f2(X(x0))} ${f2(Y(0))}L${f2(X(0.995))} ${f2(Y(0))}L${f2(X(1))} ${f2(Y(1))}"/><text class="dl-reft" x="${f2(X(0.97))}" y="${f2(Y(0.55))}" text-anchor="end">${esc(tx('a hash catches only s = 1','哈希只抓得到 s = 1'))}</text>`;
  if(o.pairs) o.pairs.forEach(p=>{const caught=(o.exact&&p.s>=1)||(o.mh&&p.u<PofS(p.s,o.b,o.r)), red=caught?GOODU.has(p.kind):(p.s>=0.8);
    g+=`<circle class="${red?'dl-dr':(caught?'dl-d1':'dl-d0')}" cx="${f2(X(Math.min(1,p.s)))}" cy="${f2(Y(p.u))}" r="3.2"/>`;});
  if(o.goals){const e=chEval(o.b,o.r), cid='dl-zclip';
    /* the goal at 0.8: a band from 95% to 100%; its reading goes left of the band, so it stays inside the chart at every width */
    g+=`<rect class="dl-goalz" x="${f2(X(0.77))}" y="${f2(Y(1))}" width="${f2(X(0.83)-X(0.77))}" height="${f2(Y(0.95)-Y(1))}"/>`;
    g+=`<circle class="${e.okHi?'dl-gok':'dl-gbad'}" cx="${f2(X(0.8))}" cy="${f2(Y(e.ph))}" r="5"/><text class="dl-gt" x="${f2(X(0.77)-6)}" y="${f2(Y(e.ph)+(e.ph>0.85?14:-6))}" text-anchor="end">${esc(tx(`0.8: ${pct1(e.ph)} (goal ≥ 95%)`,`0.8：${pct1(e.ph)}（目标 ≥ 95%）`))}</text>`;
    g+=`<circle class="${e.okLo?'dl-gok':'dl-gbad'}" cx="${f2(X(0.5))}" cy="${f2(Y(e.pl))}" r="4"/>`;
    /* the zoomed strip: the same x axis, 0 to 10%, named inside it at the right, where the curve has left the strip */
    g+=`<defs><clipPath id="${cid}"><rect x="${L0}" y="${Z0}" width="${W-R0-L0}" height="${ZH}"/></clipPath></defs>`;
    g+=`<rect class="dl-zf" x="${L0}" y="${Z0}" width="${W-R0-L0}" height="${ZH}"/>`;
    [0,0.05,0.1].forEach(p=>{if(nar&&p===0.05) return; g+=`<line class="dl-gl" x1="${L0}" x2="${W-R0}" y1="${f2(Yz(p))}" y2="${f2(Yz(p))}"/><text class="dl-ax" x="${L0-5}" y="${f2(Yz(p)+4)}" text-anchor="end">${Math.round(p*100)}%</text>`;});
    g+=`<text class="dl-axt dl-zt" x="${W-R0-4}" y="${f2(Z0+ZH-6)}" text-anchor="end">${esc(tx('zoomed in: 0 to 10%','放大：0 到 10%'))}</text>`;
    g+=`<rect class="dl-goalz" x="${f2(X(0.5)-9)}" y="${f2(Yz(0.02))}" width="18" height="${f2(Yz(0)-Yz(0.02))}"/>`;
    g+=`<g clip-path="url(#${cid})">`+(o.ref?`<path class="dl-ref" d="${path(14,8,Yz)}"/>`:'')+(o.best?`<path class="dl-bestc" d="${path(o.best.b,o.best.r,Yz)}"/>`:'')+`<path class="dl-cur" d="${path(o.b,o.r,Yz)}"/></g>`;
    const over=e.pl>ZM, cy=over?Z0:Yz(e.pl);
    g+=`<circle class="${e.okLo?'dl-gok':'dl-gbad'}" cx="${f2(X(0.5))}" cy="${f2(cy)}" r="5"/><text class="dl-gt" x="${f2(X(0.5)+16)}" y="${f2(over?Z0+13:Math.min(cy+12,Z0+ZH-20))}">${esc(tx(`0.5: ${pct1(e.pl)} (goal ≤ 2%)${over?' ↑':''}`,`0.5：${pct1(e.pl)}（目标 ≤ 2%）${over?' ↑':''}`))}</text>`;}
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(o.aria)}">${g}</svg>`;
}
function drawCurve(){
  const v=viewNow(), host=$('#dl-curve',app), cap=$('#dl-c-cap',app), leg=$('#dl-c-leg',app), w=Math.round(host.clientWidth||(host.parentNode&&host.parentNode.clientWidth)||560);
  let h, lg, aria;
  const yt=tx('↑ chance to be caught','↑ 被抓住的机会'), xt=tx('similarity of the two pages (Jaccard, on 5-grams) →','两篇网页的相似度（Jaccard，按 5-gram 算）→');
  if($('#dl-c-yt',app).textContent!==yt) $('#dl-c-yt',app).textContent=yt;
  if($('#dl-c-xt',app).textContent!==xt) $('#dl-c-xt',app).textContent=xt;
  if(v==='ch'){const e=chEval(S.b,S.r);
    cap.textContent=tx(`MinHash with ${S.b} groups of ${S.r}: the chance that a pair is caught, by how alike the two pages are`,`MinHash 用 ${S.b} 组、每组 ${S.r} 个时，一对网页被抓住的机会，和两篇有多像的关系`);
    aria=tx(`The curve P(s) for ${S.b} groups of ${S.r}. At 0.8: ${pct1(e.ph)}; at 0.5: ${pct1(e.pl)}.`,`${S.b} 组、每组 ${S.r} 个时的曲线 P(s)。0.8 时 ${pct1(e.ph)}，0.5 时 ${pct1(e.pl)}。`);
    const bst=chBest(), showB=chSubmitted&&chReveal&&!(S.b===bst.b&&S.r===bst.r);   // R1: once shown, the cheapest setting's curve too
    h=curveSVG({w,b:S.b,r:S.r,mh:1,ref:!(S.b===14&&S.r===8),goals:true,best:showB?bst:null,aria});
    lg=`<span><i class="sw dl-sw-cur"></i>${esc(tx('your setting','你的设置'))}</span><span><i class="sw dl-sw-ref"></i>${esc(tx('FineWeb’s setting (14 groups of 8)','FineWeb 的设置（14 组、每组 8 个）'))}</span><span><i class="sw dl-sw-goal"></i>${esc(tx('the goals: at 0.8 the curve must reach the box at the top, at 0.5 it must stay in the box of the zoomed strip below','目标：0.8 处曲线要进顶上的框，0.5 处要留在下方放大图的框里'))}</span>`+
      (showB?`<span><i class="sw dl-sw-best"></i>${esc(tx(`the cheapest setting (${bst.b} groups of ${bst.r})`,`最省的设置（${bst.b} 组、每组 ${bst.r} 个）`))}</span>`:'');}
  else {const pairs=pairsOf(S), f=fates(S);
    cap.textContent=S.mh?tx(`All 1,000 pages: every pair that reaches dedup, and MinHash with ${S.b} groups of ${S.r}`,`1,000 个网页里，进到去重环节的每一对，和 MinHash（${S.b} 组，每组 ${S.r} 个）`):tx('All 1,000 pages: every pair that reaches dedup, with only exact copies removed','1,000 个网页里，进到去重环节的每一对，只去完全一样的副本');
    aria=tx(`${pairs.length} pairs as dots; ${f.dupLeft} near copies stay.`,`${pairs.length} 对网页，每对一个点；留下 ${f.dupLeft} 篇几乎一样的副本。`);
    /* the plot's height from the stylesheet (--dl-ph): set only where the curve sits beside the signatures; otherwise the default */
    const ph=parseFloat(getComputedStyle(host).getPropertyValue('--dl-ph'))||0;
    h=curveSVG({w,ph,b:S.b,r:S.r,mh:S.mh,exact:S.exact,ref:S.mh&&!(S.b===14&&S.r===8),pairs,aria});
    lg=`<span><i class="sw dl-sw-d1"></i>${esc(tx('caught: under the curve, one page of the pair is thrown away','被抓住：在曲线下面，扔掉其中一篇'))}</span><span><i class="sw dl-sw-d0"></i>${esc(tx('not caught: above the curve, both stay','没被抓住：在曲线上面，两篇都留下'))}</span><span><i class="sw dl-sw-dr"></i>${esc(tx('red: a wrong call, a near copy (0.8 or more) that stays, or a heavily edited version (0.6 or less) thrown away','红色：判错了，相似度 0.8 以上的副本留了下来，或者改动很大、本该留下的版本（相似度不超过 0.6）被扔掉'))}</span><span class="dl-legn">${esc(tx('A dot’s height is a random number drawn once per pair; the curve’s height is the chance to be caught.','点的高度是给每对网页抽的一个随机数；曲线的高度是被抓住的机会。'))}</span>`;}
  if(host._h!==h){host.innerHTML=h; host._h=h;}
  if(leg._h!==lg){leg.innerHTML=lg; leg._h=lg;}
}
/* part 4's first layer: the three example pages, their whole-text hashes and MinHash signatures; the lever decides the verdicts
   (only exact copies: B stays, in red; MinHash: group 2 matches and B goes; C stays either way) */
const XNAME={A:{en:'Original',zh:'原文'},B:{en:'Repost: only the title changed',zh:'转载：只改了标题'},C:{en:'Another page on the same topic',zh:'另一篇：同一个话题'}};
function xVerdict(k,X,mh){
  if(k==='A') return {bad:false,t:tx('The original: it stays.','原文，留下。')};
  const q=X[k];
  if(k==='B'&&!mh) return {bad:true,t:tx('B’s hash differs from A’s, so comparing hashes keeps it. A wrong call: B is a repost.','B 的哈希和 A 不同，只比哈希就把它留了下来。判错了，B 是一篇转载。')};
  if(k==='B') return {bad:false,t:q.caught?tx(`All 4 values of group ${q.hit+1} agree with A’s, so B counts as a copy and is thrown away.`,`第 ${q.hit+1} 组的 4 个值和 A 全部相同，B 算重复，被扔掉。`):tx('No group agrees in full, so B stays.','没有一组全部相同，B 留了下来。')};
  return {bad:false,t:mh?(q.caught?tx('A group agrees in full, so C is thrown away.','有一组全部相同，C 被扔掉。'):tx('No value agrees with A’s, so no group does either: C stays.','没有一个值和 A 相同，更没有一组全部相同，C 留下。')):tx('C’s hash differs from A’s too, so C stays, which is right.','C 的哈希也和 A 不同，留了下来，这是对的。')};
}
function drawExample(){
  const host=$('#dl-xbox',app); if(!host) return;
  const lg=A.lang(), X=xOf(lg), sep=lg==='zh'?'':' ', mh=!!S.mh, a=X.pg.A;
  /* owner's review b8, part 4: the shading was a 2 px box-shadow on every word, and in Chinese the words touch, so each word's
     shadow was painted over the last 2 px of the word before it. Now a run of shaded words shares one background (.dl-xon); its
     padding takes room in the line, so no grey is ever painted over a glyph */
  const seq=(pg,on,i0,i1)=>{let h='', run=false;
    for(let i=i0;i<i1;i++){const s=!!(on&&on[i]), sp=i>i0?sep:'';
      if(s&&!run){h+=sp+'<span class="dl-xon">'; run=true;} else if(!s&&run){h+='</span>'+sp; run=false;} else h+=sp;
      h+=`<span class="dl-xw${s?' on':''}">${esc(pg.toks[i])}</span>`;}
    return h+(run?'</span>':'');};
  const words=(k)=>{const pg=X.pg[k], on=k==='A'?null:X[k].on;
    return `<b class="dl-xti">${seq(pg,on,0,pg.nt)}</b><br>`+seq(pg,on,pg.nt,pg.toks.length);};
  const card=k=>{const pg=X.pg[k], q=X[k], v=xVerdict(k,X,mh);
    /* the similarity's hover is HTML (xTip 'xjB' / 'xjC'): the idea in words, then the formula on its own line (R6.29) */
    const jt=lb=>`<span class="t" tabindex="0" role="button" data-ctl="xj${k}">${esc(lb)}</span>`;
    const sim=!q?tx(`${pg.set.length} different 5-grams`,`一共 ${pg.set.length} 种 5-gram`):(q.shared?tx(`${q.shared} of its 5-grams are also in A: `,`有 ${q.shared} 个 5-gram 和 A 相同，`)+jt(tx(`similarity ${fixd(q.jac,2)}`,`相似度 ${fixd(q.jac,2)}`)):tx('No 5-gram in common with A: ','没有一个 5-gram 和 A 相同，')+jt(tx('similarity 0','相似度 0')));
    const hs=`<code class="dl-xhash">${hex8(pg.hash)}</code>`+(k==='A'?'':(q.hashSame?tx(', the same as A’s','，和 A 相同'):tx(', not A’s','，和 A 不同')));
    return `<div class="dl-xpg${v.bad?' bad':''}" data-p="${k}"><div class="dl-xh"><b class="dl-xk">${k}</b><span>${esc(L(XNAME[k]))}</span></div>`+
      `<p class="dl-xt" lang="${lg==='zh'?'zh':'en'}">${words(k)}</p>`+
      `<div class="dl-xm">${tx('Hash of the whole text ','全文哈希 ')}${hs}</div><div class="dl-xm">${sim}</div><p class="dl-xv${v.bad?' dl-xbad':''}">${esc(v.t)}</p></div>`;};
  const cell=(k,j)=>{const same=mh&&k!=='A'&&X[k].agree[j]; return `<span class="t dl-xc${same?' same':''}" tabindex="0" role="button" data-ctl="x${k}${j}">${X.pg[k].sig[j].v}</span>`;};
  const row=k=>`<div class="dl-xrow"><span class="dl-xl">${k}</span>`+[0,1,2].map(g=>`<span class="dl-xg${mh&&k!=='A'&&X[k].groups[g]?' hit':''}">${[0,1,2,3].map(r=>cell(k,g*XR+r)).join('')}</span>`).join('')+`</div>`;
  const sig=`<div class="dl-xsig${mh?'':' off'}"><div class="dl-xsh">${esc(mh?tx(`MinHash signatures: 12 values, cut into 3 groups of 4. B agrees with A in ${X.B.nAgree} places, C in ${X.C.nAgree||'none'}.`,`MinHash 签名：每篇 12 个值，分成 3 组，每组 4 个。B 有 ${X.B.nAgree} 个值和 A 相同，C ${X.C.nAgree?`有 ${X.C.nAgree} 个`:'一个也没有'}。`):tx('MinHash signatures (not used when only exact copies are removed)','MinHash 签名（只去完全一样的副本时，不用它）'))}</div>`+
    `<div class="dl-xgrid">`+row('A')+row('B')+row('C')+`<div class="dl-xrow dl-xax"><span class="dl-xl"></span>${[1,2,3].map(g=>`<span class="dl-xg">${tx(`group ${g}`,`第 ${g} 组`)}</span>`).join('')}</div></div></div>`;
  const leg=`<div class="dl-leg"><span><i class="sw dl-sw-xw"></i>${esc(tx('shaded words: in a 5-gram that A has too','灰底的词：在 A 也有的 5-gram 里'))}</span><span><i class="sw dl-sw-xs"></i>${esc(tx('filled value: the same as A’s in that place','实心的值：和 A 同一位置的值相同'))}</span><span><i class="sw dl-sw-xg"></i>${esc(tx('thick box: a group whose 4 values all agree','粗框：4 个值全部相同的一组'))}</span><span><i class="sw dl-sw-red"></i>${esc(tx('red: a wrong call, a repost that stays','红色：判错了，转载留了下来'))}</span></div>`;
  const h=`<div class="dl-cap">${esc(tx('Three example pages, written for this lab: which one is a copy of A?','三个编出来的示例网页：哪一篇和 A 重复？'))}</div><div class="dl-xpgs">${['A','B','C'].map(card).join('')}</div>`+sig+leg;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
/* the hover of one signature value: which 5-gram gave the smallest hash, in A and in this page (R2.11: an aligned list) */
function xTip(key){
  const mj=/^xj([BC])$/.exec(key);
  if(mj){const k=mj[1], q=xOf(A.lang())[k], v=q.jac?fixd(q.jac,2):'0';   // the similarity of A and this page: the idea, then the formula
    return `<div class="dl-xtip"><b>${esc(tx(`Jaccard similarity of A and ${k}`,`A 和 ${k} 的 Jaccard 相似度`))}</b><p>${esc(tx('Of all the different 5-grams in the two pages, the share that both pages have. Two identical pages give 1; two pages with no 5-gram in common give 0.','两篇都有的 5-gram，占两篇所有不同 5-gram 的比例。完全一样的两篇是 1，没有一个 5-gram 相同的两篇是 0。'))}</p>`+
      MS(tx(`\\frac{\\text{5-grams in both}}{\\text{all different 5-grams}} = \\frac{${q.shared}}{${q.union}} = ${v}`,`\\frac{\\text{都有的 5-gram}}{\\text{所有不同的 5-gram}} = \\frac{${q.shared}}{${q.union}} = ${v}`))+`</div>`;}
  const m=/^x([ABC])(\d+)$/.exec(key); if(!m) return ''; const k=m[1], j=+m[2], X=xOf(A.lang()), a=X.pg.A.sig[j], me=X.pg[k].sig[j];
  const q=s=>`“${esc(s)}”`, row=(p,v)=>`<tr><th>${p}</th><td class="mono">${v.v}</td><td>${q(v.s)}</td></tr>`;
  const head=tx(`Value ${j+1} (group ${Math.floor(j/XR)+1})`,`第 ${j+1} 个值（第 ${Math.floor(j/XR)+1} 组）`), what=tx('Each value is the smallest hash among a page’s 5-grams. The table shows which 5-gram gave it.','每个值是一篇网页所有 5-gram 的哈希里最小的那个。表里列出它来自哪个 5-gram。');
  const end=k==='A'?'':(me.v===a.v?tx('The same 5-gram is the smallest in both pages, so the values agree.','两篇里最小的是同一个 5-gram，所以值相同。'):tx('The smallest 5-gram differs, so the values differ.','两篇里最小的不是同一个 5-gram，所以值不同。'));
  return `<div class="dl-xtip"><b>${esc(head)}</b><p>${esc(what)}</p><table>${row('A',a)}${k==='A'?'':row(k,me)}</table>${end?`<p>${esc(end)}</p>`:''}</div>`;
}
function drawMix(){
  const e=epochsOf(S.mix), mx=Math.max(4,Math.ceil(Math.max(...e.map(x=>x.ep))+0.5)), P=v=>(v/mx*100).toFixed(2)+'%';
  $('#dl-m-cap',app).textContent=tx(`Epochs per domain in a 15T-token run: ${L(mixOf(S.mix).t)}`,`15T 个 token 的训练里，每类数据用几遍：${L(mixOf(S.mix).t)}`);
  const rows=e.map((x,i)=>{const d=DOM[i], over=x.ep>PILE_EP+1e-9;
    const tip=tx(`${L(d.t)}: ${pshare(x.share)} of the run is ${fT(x.use)} tokens. With ${fT(x.have)} on hand, that is ${fEp(x.ep)} epochs.`,`${L(d.t)}：占 ${pshare(x.share)}，要用 ${fT(x.use)} 个 token。手里有 ${fT(x.have)}，就是 ${fEp(x.ep)} 遍。`);
    return `<div class="dl-mrow" tabindex="0" data-tipx="${esc(tip)}"><span class="dl-ml">${esc(L(d.t))}<small>${esc(tx(`${pshare(x.share)} of the run, ${fT(x.have)} on hand`,`占 ${pshare(x.share)}，手里有 ${fT(x.have)}`))}</small></span><span class="dl-mtrk"><i class="${over?'over':''}" style="width:${P(x.ep)}"></i><b class="dl-m1" style="left:${P(1)}"></b><b class="dl-m3" style="left:${P(PILE_EP)}"></b></span><em class="mono${over?' over':''}">${esc(tx(`${fEp(x.ep)} epochs`,`${fEp(x.ep)} 遍`))}</em></div>`;}).join('');
  const lg=`<div class="dl-mleg"><span><i class="dl-m1l"></i>${esc(tx('1 epoch: each text once','1 遍：每篇只用一次'))}</span><span><i class="dl-m3l"></i>${esc(tx('3 epochs: the most The Pile repeated a set','3 遍：The Pile 重复最多的数据集'))}</span><span><i class="dl-mol"></i>${esc(tx('red: more than 3 epochs','红色：超过 3 遍'))}</span></div>`;
  const h=rows+lg, host=$('#dl-mix',app); if(host._h!==h){host.innerHTML=h; host._h=h;}
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.b,S.r); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when pairs at 0.8 are caught at least 95% of the time and pairs at 0.5 at most 2%. Now: ${pct1(e.ph)} and ${pct1(e.pl)}, with ${e.h} hashes per page.`,`0.8 的网页对至少 95% 被抓住、0.5 的最多 2% 被抓住时，徽章是绿色。现在分别是 ${pct1(e.ph)} 和 ${pct1(e.pl)}，每篇网页算 ${e.h} 个哈希。`)};}
  if(v==='mix'){const m=epochsOf(S.mix)[3]; return {cls:'neutral',txt:tx(`Math: ${fEp(m.ep)} epochs`,`数学用 ${fEp(m.ep)} 遍`),short:tx(`Math ${fEp(m.ep)} epochs`,`数学 ${fEp(m.ep)} 遍`),tip:tx('How many times the math text is used in this mix.','这种配比下，数学数据要用几遍。')};}
  const f=fates(S);
  if(v==='dedup') return {cls:'neutral',txt:tx(`${f.dupLeft} near copies left`,`留下 ${f.dupLeft} 篇副本`),short:tx(`${f.dupLeft} copies left`,`留下 ${f.dupLeft} 篇副本`),tip:tx('Near copies (similarity 0.8 or more) whose original is also kept.','相似度 0.8 以上、原文也留下了的副本。')};
  if(runHidden()) return {cls:'neutral',txt:tx('Guess first','先猜'),short:tx('Guess first','先猜'),tip:tx('The run starts once you pick a guess in the card.','在卡片里选好答案，动画就开始。')};
  if(PL.g<4-1e-9) return {cls:'neutral',txt:tx('Running','进行中'),short:tx('Running','进行中'),tip:tx('The run has not reached the last stage yet.','动画还没走到最后一个环节。')};
  return {cls:'neutral',txt:tx(`${f.kept} of 1,000 left`,`留下 ${f.kept} 个`),short:tx(`${f.kept} left`,`留下 ${f.kept} 个`),tip:tx(`${f.kept} of the 1,000 pages are left after the four stages.`,`走完四个环节，1,000 个网页还剩 ${f.kept} 个。`)};
}
/* the settings in a few words: only what differs from FineWeb's */
function setupText(){
  const cn=L(CORP.find(c=>c[0]===S.corpus)[1]), off=RK.filter(k=>!S.rules[k]).length, d=[];
  if(Math.abs(S.thr-BASE.thr)>1e-9) d.push(tx(`English score ≥ ${fixd(S.thr,2)}`,`英文得分 ≥ ${fixd(S.thr,2)}`));
  if(off) d.push(tx(`${off} ${off>1?'rules':'rule'} off`,`关掉 ${off} 条规则`));
  if(!S.exact) d.push(tx('exact copies kept','不去完全一样的'));
  if(!S.mh) d.push(tx('no MinHash','不用 MinHash')); else if(S.b!==BASE.b||S.r!==BASE.r) d.push(tx(`MinHash with ${S.b} groups of ${S.r}`,`MinHash 用 ${S.b} 组、每组 ${S.r} 个`));
  return cn+tx(', ','，')+(d.length?d.join(tx(', ','，')):tx('FineWeb’s settings','FineWeb 的设置'));
}
function drawHead(){
  const v=viewNow(), t=$('#dl-st-title',app), sub=$('#dl-st-sub',app), st=$('#dl-status',app);
  t.textContent=v==='ch'?tx('MinHash for the next crawl','为下一批网页设置 MinHash'):(v==='mix'?tx('The mix of a 15T-token run','15T 个 token 的训练配比'):(v==='dedup'?tx('Dedup: which pairs are caught','去重：哪些网页对被抓住'):tx('1,000 crawled pages through four stages','抓来的 1,000 个网页，走过四个环节')));
  sub.textContent=v==='ch'?chName(S.b,S.r):(v==='mix'?tx('Target 15T tokens; tokens on hand assumed','目标 15T 个 token，手里的数据量是假设'):setupText());
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.b,S.r);
    return [{k:'ph',pts:1,label:tx('Caught at 0.8','0.8 被抓住'),v:pct1(e.ph),sub:tx('goal: at least 95%','目标：不低于 95%'),tip:tx(`Pairs of similarity 0.8 are caught ${pct1(e.ph)} of the time.`,`相似度 0.8 的网页对，有 ${pct1(e.ph)} 被抓住。`)},
      {k:'pl',pts:1,label:tx('Caught at 0.5','0.5 被抓住'),v:pct1(e.pl),sub:tx('goal: at most 2%','目标：不超过 2%'),tip:tx(`Pairs of similarity 0.5 are caught ${pct1(e.pl)} of the time.`,`相似度 0.5 的网页对，有 ${pct1(e.pl)} 被抓住。`)},
      {k:'h',label:tx('Hashes per page','每篇网页的哈希数'),v:String(e.h),sub:tx('groups × hashes per group','组数 × 每组的哈希数'),tip:tx(`${S.b} groups × ${S.r} hashes = ${e.h} hashes for every page.`,`${S.b} 组 × ${S.r} 个 = 每篇网页 ${e.h} 个哈希。`)}];}
  if(v==='mix'){const e=epochsOf(S.mix);
    return [{k:'m',label:tx('Math','数学'),v:fEp(e[3].ep),sub:tx('epochs','遍'),tip:tx(`Math takes ${fT(e[3].use)} tokens of the 0.3T on hand.`,`数学要用 ${fT(e[3].use)} 个 token，手里有 0.3T。`)},
      {k:'c',label:tx('Code','代码'),v:fEp(e[2].ep),sub:tx('epochs','遍'),tip:tx(`Code takes ${fT(e[2].use)} tokens of the 1.5T on hand.`,`代码要用 ${fT(e[2].use)} 个 token，手里有 1.5T。`)},
      {k:'w',label:tx('English web','英文网页'),v:fEp(e[0].ep),sub:tx('epochs','遍'),tip:tx(`English web takes ${fT(e[0].use)} tokens of the 12T on hand.`,`英文网页要用 ${fT(e[0].use)} 个 token，手里有 12T。`)}];}
  const f=fates(S), hide=funnelView(v)&&(runHidden()||!PL.watched.has(f.sig));   // R3.8: a run's results only once it has been watched to the end
  if(v==='dedup'){const bad=wrongDrop(S,f);
    return [{k:'dl',label:tx('Near copies left','留下的副本'),v:String(f.dupLeft),sub:tx('similarity 0.8 or more','相似度 0.8 以上'),tip:tx('Near copies whose original is also kept: red dots above the curve.','原文也留下了的副本：曲线上面的红点。')},
      {k:'bad',label:tx('Pages wrongly thrown away','误删的网页'),v:String(bad),sub:tx('similarity 0.6 or less, should stay','相似度不超过 0.6，本该留下'),tip:tx('Heavily edited versions (similarity 0.6 or less) that should stay but were caught: red dots under the curve.','改动很大的版本（相似度不超过 0.6）本该留下，却被当成重复扔掉了：曲线下面的红点。')},
      {k:'h',label:tx('Hashes per page','每篇网页的哈希数'),v:S.mh?String(S.b*S.r):'—',sub:S.mh?tx(`${S.b} groups × ${S.r}`,`${S.b} 组 × ${S.r} 个`):tx('MinHash off','没用 MinHash'),tip:S.mh?tx('MinHash computes this many hashes for every page.','MinHash 要给每篇网页算这么多个哈希。'):tx('Only one hash of the whole text per page.','每篇网页只算一个全文的哈希。')}];}
  const d='—', sh=tx('shown after the run','播完后显示');
  return [{k:'kept',label:tx('Pages left','留下的网页'),v:hide?d:String(f.kept),sub:hide?sh:tx('of 1,000','共 1,000 个'),tip:hide?tx('Shown once the run has ended.','动画播完后显示。'):tx(`${f.kept} of the 1,000 pages pass all four stages.`,`1,000 个网页里，有 ${f.kept} 个走完了四个环节。`)},
    {k:'w',pts:1,label:tx('Words left','留下的词'),v:hide?d:pct(f.wShare),sub:hide?sh:tx('of all words crawled','占抓来的全部词数'),tip:hide?tx('Shown once the run has ended.','动画播完后显示。'):tx('Kept pages are longer than average, so the share of words left differs from the share of pages.','留下的网页比平均更长，所以留下的词所占的比例和网页的比例不一样。')},
    {k:'dup',label:tx('Near copies left','留下的副本'),v:hide?d:String(f.dupLeft),sub:hide?sh:tx('similarity 0.8 or more','相似度 0.8 以上'),tip:hide?tx('Shown once the run has ended.','动画播完后显示。'):tx('Copies whose original is also kept.','原文也留下了的副本。')},
    {k:'red',label:tx('Wrong calls','判错的'),v:hide?d:String(f.redKept+f.redBin),sub:hide?sh:tx(`${f.redBin} thrown away, ${f.redKept} kept`,`误扔 ${f.redBin} 个，误留 ${f.redKept} 个`),tip:hide?tx('Shown once the run has ended.','动画播完后显示。'):tx('Red squares: good pages thrown away, plus junk and near copies that stayed.','红色方块：被扔掉的好网页，加上留下来的垃圾和副本。')}];
}
/* ▲▼ (R6.6): only after the learner's own change; while a run the learner started is replaying, the cards wait ("shown after the run")
   and the change appears at its end, against the last values shown; a scenario or another crawl is the lab setting the state: none */
let dArm=false;
const resetDeltas=()=>{Object.keys(shownPrev).forEach(k=>{shownPrev[k]=null;}); dArm=false;};
function drawMetrics(){
  const host=$('#dl-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  const hid=defs.some(m=>m.v==='—'&&m.k!=='h'); if(userAct&&hid) dArm=true;
  const act=(userAct||dArm)&&!hid;
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(act&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&(a>0||m.pts)) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v)),pts:!!m.pts})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    if(!hid) shownPrev[m.k]=m.v;});
  if(!hid) dArm=false;
}
const f3=x=>fixd(x,3), f4=x=>fixd(x,4);
function mathHTML(){
  const v=viewNow(), t=T_;
  if(v==='mix'){const e=epochsOf(S.mix);
    return `<p>${tx('Epochs of a domain:','某一类数据的遍数：')}</p>`+M(tx('\\text{epochs} = \\frac{\\text{share} \\times 15\\text{T}}{\\text{tokens on hand}}','\\text{遍数} = \\frac{\\text{占比} \\times 15\\text{T}}{\\text{手里的 token 数}}'))+e.map((x,i)=>`<p>${esc(L(DOM[i].t))}${tx(': ','：')}${pshare(x.share)} × 15T ÷ ${fT(x.have)} = <b>${fEp(x.ep)}</b></p>`).join('');}
  if(v==='dedup'||v==='ch'){const b=S.b, r=S.r;
    if(v==='dedup'&&!S.mh) return `<p>${tx('Only exact copies are removed: a copy is caught only when its hash equals its original’s, that is when s = 1.','只去完全一样的副本：副本的哈希和原文相同才会被抓住，也就是 s = 1 时。')}</p>`;
    const row=s=>{const a=Math.pow(s,r), p=PofS(s,b,r); return `<p>${tx(`A pair at s = ${fixd(s,1)}: one group agrees with chance ${fixd(s,1)} to the power ${r}, and at least one of the ${b} groups agrees with`,`相似度 s = ${fixd(s,1)} 的一对：一组全部相同的机会是 ${fixd(s,1)} 的 ${r} 次方，${b} 组里至少有一组全部相同的机会是`)}</p>`+M(`P(${fixd(s,1)}) = 1 - (1 - ${fixd(s,1)}^{${r}})^{${b}}`)+M(`= 1 - (1 - ${f4(a)})^{${b}} = ${f3(p)}`);};
    return `<p>${tx(`All r hash values of a group agree with chance ${A.ui.math('s^r')}, so a pair of similarity s is caught with chance`,`一组 r 个哈希值全部相同的机会是 ${A.ui.math('s^r')}，所以相似度为 s 的一对网页被抓住的机会是`)}</p>`+M('P(s) = 1 - (1 - s^r)^b')+`<p>${t('b','b')} = ${b}${tx(', ','，')}${t('r','r')} = ${r}</p>`+row(0.8)+row(0.5)+
      `<p>${tx('The curve rises steepest near this similarity:','曲线在这个相似度附近升得最陡：')}</p>`+M('t \\approx (1/b)^{1/r}')+M(`= (1/${b})^{1/${r}} = ${f3(thrOf(b,r))}`);}
  const f=fates(S), hide=runHidden()||!PL.watched.has(f.sig);
  if(hide) return `<p>${tx('The counts appear once the run has ended.','动画播完后，这里给出各环节的数。')}</p>`;
  return `<p>${tx('Pages left = pages crawled − what each stage throws away:','留下的网页 = 抓来的网页 − 每个环节扔掉的：')}</p>`+
    `<p>1,000 − ${f.rm[1]} − ${f.rm[2]} − ${f.rm[3]} − ${f.rm[4]} = <b>${f.kept}</b></p>`+
    `<p class="small">${tx(`${f.rm[1]} no main text, ${f.rm[2]} not English, ${f.rm[3]} low quality, ${f.rm[4]} duplicates.`,`抽不出正文 ${f.rm[1]} 个，不是英文 ${f.rm[2]} 个，质量差 ${f.rm[3]} 个，重复 ${f.rm[4]} 个。`)}</p>`;
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.Bs.includes(S.b)) S.b=CH.start.b; if(!CH.Rs.includes(S.r)) S.r=CH.start.r; S.mh=1;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  const v=viewNow();
  $('#dl-graph',app).dataset.v=v;   // before drawing: part 4's wide layout (data-pipeline.css) sets the width and height the curve is drawn at
  if(funnelView(v)) showRun(); else stopAnim();   // first, so the status, cards and math below know whether this picture has been watched (R3.8)
  drawHead();
  $('#dl-fbox',app).hidden=!funnelView(v);
  $('#dl-xbox',app).hidden=v!=='dedup';
  $('#dl-cbox',app).hidden=!(v==='dedup'||v==='ch');
  if(v==='dedup') drawExample();
  $('#dl-mbox',app).hidden=v!=='mix';
  if(funnelView(v)){$('#dl-f-cap',app).textContent=runHidden()?tx('Pick a guess in the card; the run starts then.','先在卡片里选一个答案，动画就开始。'):tx('Each square is one page. Point at one to see what it really is and why it went where it went.','每个小方块是一个网页。鼠标移到方块上，能看到它到底是什么、为什么去了那里。'); drawFunnel(); drawFLeg();}
  if(v==='dedup'||v==='ch') drawCurve();
  if(v==='mix') drawMix();
  drawMetrics();
  const mb=$('#dl-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token());   // the kit puts the settings in the address only in Scenarios; in the other modes it clears them
  A.dock.refresh(); announce();
}
function drawFLeg(){
  const lg=`<span><i class="sw dl-sw-in"></i>${esc(tx('a page still in','还在的网页'))}</span><span><i class="sw dl-sw-out"></i>${esc(tx('thrown away, in the bin of its stage','被扔掉，进了那个环节的堆'))}</span><span><i class="sw dl-sw-red"></i>${esc(tx('red: a wrong call, a good page thrown away or junk and copies kept','红色：判错了，好网页被扔掉，或者垃圾、副本留了下来'))}</span>`;
  const el=$('#dl-f-leg',app); if(el._h!==lg){el.innerHTML=lg; el._h=lg;}
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the groups and the hashes per group, in the task card.','挑战中只能在任务卡里改组数和每组的哈希数。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='lang') return tx('Fixed in this part: everything but the English score threshold. You can change the rest in Scenarios.','这一部分只改英文得分的门槛，其他设置固定。在应用场景里可以改其他设置。');
  if(id==='rules') return tx('Fixed in this part: everything but the five rules. You can change the rest in Scenarios.','这一部分只开关 5 条规则，其他设置固定。在应用场景里可以改其他设置。');
  if(id==='dedup') return tx('In this part you pick the way to remove duplicates in the card, and MinHash’s groups and hashes there too.','这一部分在卡片里选去重的方法，也在那里改 MinHash 的组数和哈希数。');
  if(id==='mix') return tx('In this part you pick a mix in the card.','这一部分在卡片里选配比。');
  return tx('Fixed in this part: FineWeb’s settings. You can change them in Scenarios.','这一部分固定用 FineWeb 的设置。在应用场景里可以改。');
}
function syncControls(){
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('thr',Math.round(S.thr*100),fixd(S.thr,2)); setR('b',S.b,String(S.b)); setR('r',S.r,String(S.r));
  $$('.seg-ctl[data-key="corpus"]',app).forEach(g=>A.ui.setPressed(g,S.corpus)); $$('.seg-ctl[data-key="exact"]',app).forEach(g=>A.ui.setPressed(g,String(S.exact?1:0))); $$('.seg-ctl[data-key="mh"]',app).forEach(g=>A.ui.setPressed(g,String(S.mh?1:0)));
  const f=fates(S), hideN=!PL.watched.has(f.sig);
  $$('.dl-sw[data-rule]',app).forEach(b=>{const k=b.dataset.rule; b.setAttribute('aria-pressed',String(!!S.rules[k])); const n=b.querySelector('[data-rule-n]'); if(n) n.textContent=S.rules[k]&&!hideN?String(f.byRule[k]):'';});
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#dl-controls',app).classList.toggle('locked',ch);
  hide('#dl-ctl-corpus-wrap',g||ch); hide('#dl-ctl-thr-wrap',ch||(g&&!isP('lang'))); hide('#dl-ctl-rules-wrap',ch||(g&&!isP('rules'))); hide('#dl-ctl-dup-wrap',g||ch); hide('#dl-ctl-reset-wrap',g||ch);
  const br=$('#dl-ctl-br',app); if(br) br.hidden=!S.mh;
  const ln=$('#dl-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const v=viewNow();
  if(v==='ch'){const e=chEval(S.b,S.r); A.announce(tx(`${chName(S.b,S.r)}: caught at 0.8 ${pct1(e.ph)}, at 0.5 ${pct1(e.pl)}, ${e.h} hashes per page.`,`${chName(S.b,S.r)}：0.8 被抓住 ${pct1(e.ph)}，0.5 被抓住 ${pct1(e.pl)}，每篇 ${e.h} 个哈希。`)); return;}
  if(v==='mix'){const m=epochsOf(S.mix)[3]; A.announce(tx(`Math ${fEp(m.ep)} epochs.`,`数学用 ${fEp(m.ep)} 遍。`)); return;}
  if(runHidden()||PL.g<4-1e-9) return; const f=fates(S); A.announce(tx(`${f.kept} of 1,000 pages left; ${f.dupLeft} near copies left.`,`1,000 个网页剩 ${f.kept} 个，留下 ${f.dupLeft} 篇副本。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  S={...S,...p,rules:{...S.rules,...(p.rules||{})}};
  S.thr=Math.max(0.3,Math.min(0.95,Math.round(S.thr*20)/20)); S.b=Math.max(1,Math.min(60,Math.round(S.b))); S.r=Math.max(1,Math.min(20,Math.round(S.r)));
  S.exact=S.exact?1:0; S.mh=S.mh?1:0; if(!CORPORA[S.corpus]) S.corpus='web'; if(!MIXES.some(m=>m.v===S.mix)) S.mix='prop';
  if(isP('lang')&&Math.abs(S.thr-BASE.thr)>1e-9) thrTried=true;
  if(isP('dedup')&&(S.b!==BASE.b||S.r!==BASE.r)) brTried=true;
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setRule(k,on){if(!RK.includes(k)) return; S={...S,rules:{...S.rules,[k]:on?1:0}}; if(isP('rules')) ruleTried=true; render();}
function setLvDup(v){lvDup=v==='exact'?'exact':'both'; triedDup.add(lvDup); S={...S,exact:1,mh:lvDup==='both'?1:0}; render();}
function setMix(v){if(!MIXES.some(m=>m.v===v)) return; triedMix.add(v); S={...S,mix:v}; render();}
function setRem(k){if(!REMS[k]) return; remView=k; resetDeltas(); S={...cloneS(BASE),...REMS[k].s}; renderRail(); if(innerWidth<=760) A.ui.reveal($('#dl-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the four answers in the card.','向下滚到卡片里的四个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='lang') return thrCtl('dl-ph-thr',true);
  if(id==='rules') return `<button type="button" class="btn ghost sm" data-act="to-rules" data-tipx="${esc(tx('Scrolls down to the five switches in the card.','向下滚到卡片里的 5 个开关。'))}">${tx('Switches below ↓','开关在下面 ↓')}</button>`;
  if(id==='dedup'||id==='mix') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the choices in the card.','向下滚到卡片里的选项。'))}">${tx('Pick one below ↓','在下面选 ↓')}</button>`;
  return '';
}
function renderPhone(){const el=$('#dl-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; dArm=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S=cloneS(s.enter());
  const st=$('#dl-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#dl-rail .step.current',app)||$('#dl-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tk=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tk,120); return;}
    const st=$('#dl-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#dl-rail .step.current',app)||$('#dl-rail .railcard',app)||$('#dl-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
/* where part 4 stacks the example over the curve (a stage narrower than the side-by-side layout), moving b or r scrolls the sticky
   stage (not the page) to the curve they change; beside the signatures the curve is already in view, and nothing moves */
function revealCurve(){if(!app||!isP('dedup')) return; const st=$('#dl-stage',app), cv=$('#dl-curve',app); if(!st||!cv||innerWidth<=760||getComputedStyle(st).overflowY!=='auto') return;
  const sr=st.getBoundingClientRect(), cr=cv.getBoundingClientRect(); if(cr.top>=sr.top&&cr.bottom<=sr.bottom) return; st.scrollTop+=cr.top-sr.top-40;}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#dl-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#dl-insight',app)); else revealBox($('#dl-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; dArm=false;
  if(was==='challenge'&&saved){S=cloneS(saved); saved=null;}
  if(m==='challenge'){saved=cloneS(S); chSubmitted=false; chReveal=false; S={...cloneS(BASE),b:CH.start.b,r:CH.start.r};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S=cloneS(PRESETS[0].s);}
  A.shell.setMode(app,m,DESCS());
  const st=$('#dl-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[CORP.findIndex(c=>c[0]===s.corpus),Math.round(s.thr*100),RK.map(k=>s.rules&&s.rules[k]?1:0).join(''),s.exact?1:0,s.mh?1:0,s.b,s.r].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const [ci,th,bits,ex,mh,b,r]=p, n=[ci,th,ex,mh,b,r].map(Number);
  if(n.some(x=>!Number.isInteger(x))||!/^[01]{5}$/.test(bits)) return null;
  if(n[0]<0||n[0]>2||n[1]<30||n[1]>95||n[1]%5||n[2]>1||n[2]<0||n[3]>1||n[3]<0||n[4]<1||n[4]>60||n[5]<1||n[5]>20) return null;
  const rules={}; RK.forEach((k,i)=>{rules[k]=+bits[i];});
  return {...cloneS(BASE),corpus:CORP[n[0]][0],thr:n[1]/100,rules,exact:n[2],mh:n[3],b:n[4],r:n[5]};
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
let drillSeed=1, drillOpen=false;
function makeDrill(seed){const r=mulberry(seed*104729+11), pk=a=>a[Math.floor(r()*a.length)];
  const b=pk([9,14,20,25]), rr=pk([5,8,10,13]), s=pk([0.6,0.7,0.8,0.9]);
  return {seed,b,r:rr,s,p:PofS(s,b,rr),a:Math.pow(s,rr),t:thrOf(b,rr)};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#dl-drill-q',app), a=$('#dl-drill-a',app);
  qd.innerHTML=tx(`MinHash with <b>${d.b}</b> groups of <b>${d.r}</b> hashes. What is the chance that two pages of Jaccard similarity <b>${fixd(d.s,1)}</b> become candidate duplicates? At about what similarity does the curve rise?`,
                  `MinHash 用 <b>${d.b}</b> 组，每组 <b>${d.r}</b> 个哈希。两篇 Jaccard 相似度为 <b>${fixd(d.s,1)}</b> 的网页被当成重复的机会是多少？曲线大约在相似度多少的地方升起来？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>About ${pct1(d.p)}; the curve rises near ${f3(d.t)}.</b></p><p>One group agrees with chance:</p>`+M(`s^r = ${fixd(d.s,1)}^{${d.r}} = ${f4(d.a)}`)+`<p>With ${d.b} groups:</p>`+M(`P = 1 - (1 - ${f4(d.a)})^{${d.b}} = ${f3(d.p)}`)+`<p>The steepest point is near:</p>`+M(`t \\approx (1/${d.b})^{1/${d.r}} = ${f3(d.t)}`),
                 `<p><b>约 ${pct1(d.p)}；曲线在 ${f3(d.t)} 附近升起来。</b></p><p>一组全部相同的机会：</p>`+M(`s^r = ${fixd(d.s,1)}^{${d.r}} = ${f4(d.a)}`)+`<p>有 ${d.b} 组时：</p>`+M(`P = 1 - (1 - ${f4(d.a)})^{${d.b}} = ${f3(d.p)}`)+`<p>最陡的地方在这里附近：</p>`+M(`t \\approx (1/${d.b})^{1/${d.r}} = ${f3(d.t)}`));
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={b:{en:'Groups b (how many groups the signature is cut into)',zh:'组数 b（签名分成几组）'},r:{en:'Hashes per group r',zh:'每组的哈希数 r'},thr:{en:'English score threshold',zh:'英文得分的门槛'}};
const TERM={
 b:()=>tx(`b is the number of groups the signature is cut into, ${S.b} here. More groups give a pair more chances to match.`,`b 是签名分成的组数，这里是 ${S.b}。组越多，一对网页配上的机会越多。`),
 r:()=>tx(`r is the number of hashes in each group, ${S.r} here. All r must agree for the group to match.`,`r 是每组的哈希数，这里是 ${S.r}。一组的 r 个值全部相同，这组才算配上。`)
};
const termTipHTML=k=>/^x(?:[ABC]\d+|j[BC])$/.test(k)?xTip(k):(TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'');
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the setting in the task card.','挑战中固定不变：请在任务卡里改设置。');
  if(mode!=='guided') return '';
  if((k==='b'||k==='r')&&isP('dedup')&&lvDup==='both') return ''; if(k==='thr'&&isP('lang')) return '';
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({b:$('#dl-ic-b',app)||$('#dl-ctl-b',app),r:$('#dl-ic-r',app)||$('#dl-ctl-r',app),thr:$('#dl-ic-thr',app)||$('#dl-ctl-thr',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='controls'){const c=$('#dl-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(/^x(?:[ABC]\d+|j[BC])$/.test(k)){if(src) A.tip.show(src,xTip(k)); return;}   // a signature value or a similarity: a click shows its hover too
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=cloneS(mode==='sandbox'?PRESETS[0].s:BASE); render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.b,S.r).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(PL.on) pause(); else play(false); A.tip.quiet(); return;}
  if(a==='to-q'){const q=$('#dl-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-rules'){const q=$('#dl-rail .dl-rules',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#dl-rail .dl-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#dl-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#dl-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='thr') bySettings(e.target,()=>setState({thr:v/100}));
    else if(k==='b'){bySettings(e.target,()=>setState({b:v})); revealCurve();}
    else if(k==='r'){bySettings(e.target,()=>setState({r:v})); revealCurve();}
    else if(k==='scrub'){stopAnim(); PL.g=v/100; if(v>=400){finishRun(); return;} drawFunnel();}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&k!=='scrub'&&e.target.closest('#dl-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#dl-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){if(PL.watched.has(F0().sig)) return; pred=d.pred; renderRail(); const fb=$(`#dl-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.rule){const f=()=>byLearner(()=>setRule(d.rule,!S.rules[d.rule])); if(b.closest('#dl-controls')) holdPlace(b,f); else f(); return;}
    if(d.lever!=null&&(isP('dedup')||isP('mix'))){const f=()=>byLearner(()=>isP('dedup')?setLvDup(d.lever):setMix(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {wantObs=true; f(); revealSoon(); wantObs=false;} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='corpus'){if(d.v!==S.corpus) resetDeltas(); bySettings(b,()=>setState({corpus:d.v}));}
      else if(k==='exact') bySettings(b,()=>setState({exact:+d.v}));
      else if(k==='mh') bySettings(b,()=>setState({mh:+d.v}));
      else if(k==='spd'){PL.spd=d.v; PL.last=0; drawPlayer();}
      else if(k==='chb') byLearner(()=>setState({b:+d.v}));
      else if(k==='chr') byLearner(()=>setState({r:+d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; resetDeltas(); byLearner(()=>{S=cloneS(p.s); render();}); requestAnimationFrame(()=>{if(app) revealBox($('#dl-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#dl-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}
function bindCanvas(){
  const cv=$('#dl-cv',app); if(!cv) return;
  cv.addEventListener('mousemove',ev=>{const i=hitTest(ev); if(i<0){if(cv._tip!=null){cv._tip=null; A.tip.hide();} return;} if(cv._tip===i) return; cv._tip=i; A.tip.show(cv,tipHTML(i));});
  cv.addEventListener('mouseleave',()=>{cv._tip=null; A.tip.hide();});
  cv.addEventListener('click',ev=>{const i=hitTest(ev); if(i>=0){cv._tip=i; A.tip.show(cv,tipHTML(i));}});
}

/* =====================================================================
   registration
   ===================================================================== */
/* a reload continues where the learner was (round 3 #6, CONTRACT.md): the settings, the part, the answer, the choices tried, the
   runs already played to the end, the playing speed, the challenge's result, the drill. A run's own key is a row of 1,000 page fates,
   too long to keep, so the record keeps a short hash of it. The playhead is not kept: a run still playing is recorded as played to
   its end, so after the reload it shows finished, as a run already played does. */
const SK=Object.keys(BASE), okObj=o=>!!o&&typeof o==='object'&&!Array.isArray(o);
const pickS=o=>({...Object.fromEntries(SK.map(k=>[k,o[k]])),rules:Object.fromEntries(RK.map(k=>[k,o.rules[k]]))});
const cfgOf=x=>{if(!okObj(x)||!okObj(x.rules)) return null; const c=parse(token(x)); if(!c) return null; c.mix=MIXES.some(m=>m.v===x.mix)?x.mix:'prop'; return pickS(c);};
const hsig=t=>{let h=2166136261; for(let i=0;i<t.length;i++){h^=t.charCodeAt(i); h=Math.imul(h,16777619);} return (h>>>0).toString(36);};
const WH=new Set();   // hashes of the runs played to the end before the reload
const plHas=Set.prototype.has.bind(PL.watched); PL.watched.has=k=>plHas(k)||(WH.size>0&&typeof k==='string'&&WH.has(hsig(k)));
A.registerLab({
  id:ID,
  snap(){return {v:1,mode,step,S:pickS(S),pred,observed:{...observed},saved:saved?pickS(saved):null,lvDup,triedDup:[...triedDup],triedMix:[...triedMix],thrTried,ruleTried,brTried,remView,
    spd:PL.spd,watched:[...new Set([...WH,...[...PL.watched].map(hsig),...(PL.on&&PL.sig?[hsig(PL.sig)]:[])])].slice(-60),chSubmitted,chReveal,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.b,S.r).ok};},
  restore(o){if(!okObj(o)||o.v!==1) return;
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=cloneS(cfgOf(o.S)||BASE);
    if(mode==='challenge'){if(!CH.Bs.includes(S.b)) S.b=CH.start.b; if(!CH.Rs.includes(S.r)) S.r=CH.start.r;}
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    observed={}; if(okObj(o.observed)) for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    lvDup=o.lvDup==='exact'?'exact':'both'; triedDup=new Set(['both',...arr(o.triedDup,v=>v==='both'||v==='exact'),lvDup]);
    triedMix=new Set(['prop',...arr(o.triedMix,v=>MIXES.some(m=>m.v===v))]);
    thrTried=o.thrTried===true; ruleTried=o.ruleTried===true; brTried=o.brTried===true; remView=REMS[o.remView]&&Object.prototype.hasOwnProperty.call(REMS,o.remView)?o.remView:'run';
    PL.spd=['slow','mid','fast'].includes(o.spd)?o.spd:'fast'; PL.g=0; PL.on=false; PL.sig=null;
    WH.clear(); arr(o.watched,v=>typeof v==='string'&&/^[0-9a-z]{1,8}$/.test(v)).slice(-60).forEach(v=>WH.add(v));
    saved=mode==='challenge'?cfgOf(o.saved):null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    sbTouched=o.sbTouched===true; drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    wantObs=false; root=el;
    /* a settings address that is the state already here (a reload, or coming back) keeps that state, with the mix the address does
       not carry, and whether the learner has touched the settings */
    const p=tok?parse(tok):null; if(p){if(!(mode==='sandbox'&&token()===tok)){sbTouched=true; S=cloneS(p);} mode='sandbox';}
    if(mode==='challenge'&&!saved) saved=cloneS(BASE);
    el.innerHTML=viewHTML(); app=$('#dl-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#dl-rail',app)); keepStage();},token});
    bind(app); bindCanvas();
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0&&!relang) S=cloneS(s.enter());}
    PL.f=null;   // the canvas is new; showRun keeps g and the watched pictures
    fillRail();
    render();
  },
  leave(){clearTimeout(annT); stageStop(); stopAnim(); A.tip.hide(); root=null; app=null;},
  onLang(){if(root){const d=$('#dl-math',app), open=d&&d.open, was=PL.on, g=PL.g, sig=PL.sig; stopAnim(); relang=true; try{this.mount(root,null);} finally{relang=false;} PL.g=g; if(open) $('#dl-math',app).open=true; if(was&&sig===PL.sig) play(true); else drawFunnel();}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='mix'?$('#dl-mix',app):((v==='dedup'||v==='ch')?$('#dl-curve',app):$('#dl-cvwrap',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#dl-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(v==='ch'){const e=chEval(S.b,S.r); vis.innerHTML=''; read.innerHTML=esc(tx(`${chName(S.b,S.r)}: 0.8 caught ${pct1(e.ph)}, 0.5 caught ${pct1(e.pl)}, ${e.h} hashes`,`${chName(S.b,S.r)}：0.8 被抓住 ${pct1(e.ph)}，0.5 被抓住 ${pct1(e.pl)}，${e.h} 个哈希`)); return;}
      if(v==='mix'){const e=epochsOf(S.mix); vis.innerHTML=''; read.innerHTML=esc(tx(`Math ${fEp(e[3].ep)} epochs, code ${fEp(e[2].ep)}`,`数学 ${fEp(e[3].ep)} 遍，代码 ${fEp(e[2].ep)} 遍`)); return;}
      const f=fates(S);
      if(v==='dedup'){vis.innerHTML=''; read.innerHTML=esc(tx(`${f.dupLeft} near copies left`,`留下 ${f.dupLeft} 篇副本`)); return;}
      if(runHidden()||PL.g<4-1e-9){vis.innerHTML=''; read.innerHTML=esc(tx('1,000 crawled pages, four stages','抓来的 1,000 个网页，四个环节')); return;}
      vis.innerHTML=`<div class="dl-dk"><i style="width:${(f.kept/N*100).toFixed(1)}%"></i></div>`; read.innerHTML=esc(tx(`${f.kept} of 1,000 pages left, ${f.dupLeft} near copies`,`1,000 个网页剩 ${f.kept} 个，副本 ${f.dupLeft} 篇`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__dl=Object.freeze({
  fates:o=>{const f=fates({...cloneS(BASE),...(o||{}),rules:{...BASE.rules,...((o||{}).rules||{})}}); return deep({rm:f.rm,kept:f.kept,byRule:f.byRule,dupLeft:f.dupLeft,nearLeft:f.nearLeft,redKept:f.redKept,redBin:f.redBin,wShare:f.wShare,sig:f.sig});},
  docs:k=>deep(corpus(k||'web').docs), pairs:o=>deep(pairsOf({...cloneS(BASE),...(o||{})})), PofS, thrOf, chEval:(b,r)=>deep(chEval(b,r)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), chPass:()=>deep(chPass()), chNextText:(b,r)=>{const x=document.createElement('div'); x.innerHTML=chSumHTML(chEval(b,r),chBest()); const p=x.querySelector('.dl-next'); return p?p.textContent:'';}, chShown:(b,r)=>{const e=chEval(b,r); return {ph:pct1(e.ph),pl:pct1(e.pl),h:String(e.h),ok:e.ok,okHi:e.okHi,okLo:e.okLo};}, chOne:(b,r)=>deep(chOneChange(chEval(b,r))), chGap:(b,r)=>chGap(chEval(b,r)), epochs:v=>deep(epochsOf(v)), drill:s=>deep(makeDrill(s)),
  consts:Object.freeze({N,CORPORA:deep(CORPORA),BASE:deep(BASE),CH:deep(CH),STEP_IDS:STEPS.map(s=>s.id),PRED:PRED.map(o=>o.v),MIXES:deep(MIXES.map(m=>({v:m.v,sh:m.sh}))),DOM:deep(DOM.map(d=>({k:d.k,have:d.have}))),TARGET,PILE_EP,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),CODE}),
  state:()=>deep({...S,mode,step,pred,observed,lvDup,triedDup:[...triedDup],triedMix:[...triedMix],remView,chSubmitted,chReveal,g:PL.g,playing:PL.on,spd:PL.spd,sig:PL.sig,watched:[...PL.watched],stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),explainA:e.explain.map(q=>q.a)});},
  tip:i=>tipHTML(i),
  xpages:()=>deep(XP), xseed:XSEED, x:lg=>deep(xOf(lg||'en')), xtip:k=>xTip(k)
});
})();
