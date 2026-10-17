/* MHA lab: why attention uses many heads (Transformer series ②). A live toy of multi-head attention (Vaswani et al., 2017,
   §3.2.2): one layer, hidden size d = 8, hand-set Wq, Wk, Wv, Wo, a 9-token sentence (one word = one token). The same four
   matrices run as 4 heads of 2 numbers, 2 heads of 4 or 1 head of 8: only the split changes, exactly as in real MHA.
   Everything the page shows is computed here: q = Wq·x, k = Wk·x, v = Wv·x, score = q·k ÷ √head_dim, the causal mask,
   a softmax per head, the weighted sum of values, the heads side by side, × Wo. The guided parts build up (R3.13): one head,
   then 2, then 4, each step showing what the split changes.
   Names used everywhere (EN / ZH): query / key / value, score 得分, attention weight 注意力权重, head, head_dim,
   causal mask 因果 mask, pattern 模式 (this lab only). Also registers the home-tile teaser. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='mha';
/* Chinese text keeps a space between Chinese characters and Latin words, numbers or <s> (“H2 是”, “在 <s> 上”) */
const zfix=s=>String(s).replace(/([\u4e00-\u9fff])(&lt;s&gt;|<s>|[A-Za-z0-9])/g,'$1 $2').replace(/(&lt;s&gt;|<s>|[A-Za-z0-9%])([\u4e00-\u9fff])/g,'$1 $2').replace(/(&lt;s&gt;|<s>)(\d)/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* ---------- engine: pure functions ---------- */
const D=8, NT=9;
const OM=2*Math.PI/11;            // the position code turns 360°/11 per word: cos 5ω = cos 6ω, so “cat” (5 words before “it” in EN, 6 in ZH) scores the same
const SENT={en:['<s>','The','cat','sat','on','the','mat','because','it'],zh:['<s>','小猫','坐','在','垫子','上','，','因为','它']};
const FEAT={en:{noun:[2,6],animal:[2],pron:[8],conj:[7]},zh:{noun:[1,4],animal:[1],pron:[8],conj:[7]}};
const IT=8, BECAUSE=7;            // the query of the mission and the word just before it, the same index in both sentences
const HC=2;                       // with 4 heads, H3 finds who “it” is: the head that puts its weight on “cat”
const CAT={en:2,zh:1};            // the word “it” refers to in this lab
/* the scale of each hand-set pattern: P1 previous word (a), P2 who “it” is (b for a pronoun's query, b2 for every other query),
   P3 first token (c), P4 nouns (d). P2 gives other nouns −d² for a pronoun's query, which cancels P4's +d² in the same head.
   a and b are set so that at “it” one head gives “because” and “cat” almost the same score (43% and 41%), 2 heads give each
   word its own head (83% and 100%) and 4 heads give 93% and 100%; no shown weight at “it” sits within 1 point of 80%. */
const P={a:5.4,b:7.28,b2:3,c:3,d:2};
const FNAMES=['cos','sin','first','noun','animal','pronoun','one','because'];
function feats(lang,p){const f=FEAT[lang];
  return [Math.cos(p*OM),Math.sin(p*OM),p===0?1:0,f.noun.includes(p)?1:0,f.animal.includes(p)?1:0,f.pron.includes(p)?1:0,1,f.conj.includes(p)?1:0];}
function zeros(){return Array.from({length:D},()=>new Array(D).fill(0));}
/* rows = the numbers of q, k or v, in 4 slots of 2: the previous word (P1), the first token (P3), who “it” is (P2), the nouns (P4).
   4 heads: one slot each (H1–H4). 2 heads: head A gets slots 1–2 and head B slots 3–4, so the two words “it” needs fall in
   different heads. 1 head: all four slots, added up. Columns = the features of x. */
function buildMats(){
  const {a,b,b2,c,d}=P, f=d*d/b, co=Math.cos(OM), si=Math.sin(OM), Wq=zeros(), Wk=zeros(), Wv=zeros(), Wo=zeros();
  Wq[0][0]=a*co; Wq[0][1]=a*si; Wq[1][0]=-a*si; Wq[1][1]=a*co;      // P1: the query's position code turned back one word
  Wk[0][0]=a; Wk[1][1]=a;
  Wq[2][6]=c; Wk[2][2]=c;                                             // P3: the first token
  Wq[4][5]=b;                                                         // P2: a pronoun looks for an animal ...
  Wk[4][4]=b+f; Wk[4][3]=-f;                                          // ... and away from other nouns
  Wq[5][6]=b2; Wq[5][5]=-b2; Wk[5][2]=b2;                             // every other query rests on <s>
  Wq[6][6]=d; Wk[6][3]=d;                                             // P4: the nouns
  for(let h=0;h<4;h++){Wv[2*h][7]=1; Wv[2*h+1][4]=1;}                 // values: (is “because”, is an animal), in every head's slice
  for(let i=0;i<D;i++) Wo[i][i]=1;                                    // Wo: the identity
  return {Wq,Wk,Wv,Wo};
}
const M=buildMats();
const NPAR=4*D*D;                 // the block's parameters: Wq, Wk, Wv and Wo, D × D each, whatever the split
const mv=(W,x)=>W.map(r=>r.reduce((s,w,i)=>s+w*x[i],0));
const HEADS=[4,2,1];
/* one head's weights over the tokens 0..p (later tokens get no weight: the causal mask) */
function softmax(s){let m=-Infinity; s.forEach(v=>{if(v>m) m=v;}); const e=s.map(v=>Math.exp(v-m)), z=e.reduce((a,v)=>a+v,0); return e.map(v=>v/z);}
const cache=new Map();
/* s: {lang, q (the query's index 0..8), H (1, 2 or 4), on ([bool] for 4 heads; ignored for 1 and 2)} */
function compute(s){
  const lang=s.lang==='zh'?'zh':'en', p=Math.max(0,Math.min(NT-1,s.q|0)), H=HEADS.includes(+s.H)?+s.H:4;
  const on=H===4?[0,1,2,3].map(h=>!s.on||s.on[h]!==false&&s.on[h]!==0):new Array(H).fill(true);
  const key=[lang,p,H,on.map(Number).join('')].join('|'); let r=cache.get(key); if(r) return r;
  const X=[...Array(NT)].map((_,j)=>feats(lang,j)), Q=X.map(x=>mv(M.Wq,x)), K=X.map(x=>mv(M.Wk,x)), V=X.map(x=>mv(M.Wv,x));
  const hd=D/H, q=Q[p], heads=[];
  for(let h=0;h<H;h++){const lo=h*hd, hi=lo+hd;
    const dots=[], sc=[];
    for(let j=0;j<=p;j++){let t=0; for(let i=lo;i<hi;i++) t+=q[i]*K[j][i]; dots.push(t); sc.push(t/Math.sqrt(hd));}
    const w0=softmax(sc), w=[...Array(NT)].map((_,j)=>j<=p?w0[j]:0);
    const out=[]; for(let i=lo;i<hi;i++){let t=0; for(let j=0;j<=p;j++) t+=w[j]*V[j][i]; out.push(t);}
    heads.push({h,lo,hi,dots,scores:sc,w,out,on:on[h]});}
  const cat=[].concat(...heads.map(o=>o.on?o.out:o.out.map(()=>0)));   // a head that is off adds nothing (it is pruned)
  const y=mv(M.Wo,cat);
  const best=j=>{let v=0,h=-1; heads.forEach(o=>{if(o.on&&o.w[j]>v){v=o.w[j]; h=o.h;}}); return {v,h};};
  const ci=CAT[lang];
  r={lang,p,H,hd,on,X,Q,K,V,q,heads,cat,y,words:SENT[lang],cat_i:ci,
     prev:p>0?best(p-1):{v:0,h:-1}, bec:best(BECAUSE), catw:best(ci)};
  cache.set(key,r); return r;
}
/* the numbers as Show the math prints them, each recomputed from the rounded values shown before it (R6.6) */
const rd=(v,d)=>{const k=10**d; const x=Math.round(v*k+(v>=0?1e-9:-1e-9))/k; return x===0?0:x;};
const SQRT={2:1.4142,4:2,8:2.8284};
function chainOf(r,h){
  const o=r.heads[h], qs=r.q.slice(o.lo,o.hi).map(v=>rd(v,3)), sq=SQRT[r.hd];
  const rows=[]; for(let j=0;j<=r.p;j++){const ks=r.K[j].slice(o.lo,o.hi).map(v=>rd(v,3)), dot=rd(qs.reduce((s,v,i)=>s+v*ks[i],0),3), sc=rd(dot/sq,3); rows.push({j,ks,dot,sc});}
  const mx=Math.max(...rows.map(x=>x.sc)); rows.forEach(x=>{x.e=rd(Math.exp(x.sc-mx),4);});
  const sum=rd(rows.reduce((s,x)=>s+x.e,0),4); rows.forEach(x=>{x.w=x.e/sum;});
  const vs=j=>r.V[j].slice(o.lo,o.hi);
  const out=Array.from({length:o.hi-o.lo},(_,i)=>rd(rows.reduce((s,x)=>s+rd(x.w,3)*vs(x.j)[i],0),3));
  return {qs,sq,rows,mx,sum,out};
}
/* the challenge: 4 heads at “it”; which heads are on */
const CH_GOAL=0.8;
const onKey=on=>on.map(v=>v?1:0).join('');
function chPoint(lang,on){const r=compute({lang,q:IT,H:4,on}); const bb=r.bec.v, bc=r.catw.v; return {on:on.slice(),bb,bc,okB:bb>=CH_GOAL,okC:bc>=CH_GOAL,ok:bb>=CH_GOAL&&bc>=CH_GOAL,off:on.filter(v=>!v).length};}
const ALL_ON=()=>[true,true,true,true];
function combos(){const out=[]; for(let m=15;m>=0;m--){out.push([8,4,2,1].map(b=>!!(m&b)));} return out;}
/* the real-model numbers of part 7 come from the KV lab's catalog (config.json values), so both labs agree */
function realModel(){const m=A.kv&&A.kv.MODELS&&A.kv.MODELS['llama31-8b']||{name:'Llama-3.1-8B',layers:32,heads:32,kvHeads:8,headDim:128};
  const d=m.heads*m.headDim, gqa=2*m.kvHeads*m.headDim, mha=2*m.heads*m.headDim, bytes=2;
  return {name:m.name,layers:m.layers,heads:m.heads,kv:m.kvHeads,hd:m.headDim,d,group:m.heads/m.kvHeads,gqa,mha,bytes,
    perTok:gqa*m.layers*bytes, perTokMha:mha*m.layers*bytes, half:m.heads/2};}

/* ---------- formatting ---------- */
const LG=()=>A.lang()==='zh'?'zh':'en';
const R=()=>compute({lang:LG(),...S});
const pc=w=>rd(w*100,0)+'%';                                   // whole percent, rounded half up as by hand
const pc1=w=>rd(w*100,1).toFixed(1)+'%';
const MINUS='−';
const fx=(v,d)=>{const s=Math.abs(v).toFixed(d); return (v<0&&+s!==0?MINUS:'')+s;};
const fi=n=>A.fmt.int(n);
const wd=j=>SENT[LG()][j];
const COMMA=j=>LG()==='zh'&&wd(j)==='，';                       // the Chinese comma token reads as punctuation when only quoted (M8)
const wq=j=>j===0?'&lt;s&gt;':(COMMA(j)?'逗号“，”':`“${esc(wd(j))}”`);   // a word in running text (HTML)
const wt=j=>j===0?'<s>':(COMMA(j)?'逗号“，”':`“${wd(j)}”`);        // the same word in a tooltip (plain text)
const DEF=(t,tip)=>A.ui.def?A.ui.def(t,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${t}</span>`;
const ROLE=[{en:'the previous word',zh:'前一个词'},{en:'the first token',zh:'第一个 token'},{en:'who “it” is',zh:'“它”指谁'},{en:'the nouns',zh:'名词'}];
/* what each head of the 4-head split was set to do, as the row labels say it (a job given by hand in this lab) */
const JOB=[{en:'looks at the previous word',zh:'看前一个词'},{en:'looks at the first token',zh:'看第一个 token'},{en:'finds who “it” is',zh:'找“它”指谁'},{en:'looks at the nouns',zh:'看名词'}];
const JOB_G=[{en:'looking at the previous word',zh:'看前一个词'},{en:'looking at the first token',zh:'看第一个 token'},{en:'finding who “it” is',zh:'找“它”指谁'},{en:'looking at the nouns',zh:'看名词'}];
/* a head's name in running text: H1…H4 with 4 heads, head A / head B with 2, “the one head” with 1 */
const hn=(H,h)=>H===4?`H${h+1}`:(H===2?`head ${'AB'[h]}`:tx('the one head','唯一的 head'));
const Hn=(H,h)=>H===4?`H${h+1}`:(H===2?tx(`Head ${'AB'[h]}`,`head ${'AB'[h]}`):tx('The one head','唯一的 head'));
const headsTxt=H=>tx(H===1?'1 head':`${H} heads`,`${H} 个 head`);
/* running text names the head count only; the split itself (of q, k and v, never of the token) is spelled out by splitTitle */
const splitTxt=headsTxt;
const splitTitle=H=>H===1?tx('q, k and v kept whole as 1 head of 8 numbers','q、k、v 不拆，1 个 head，8 维'):tx(`q, k and v split into ${H} heads of ${8/H} numbers`,`q、k、v 各拆成 ${H} 个 head，每个 ${8/H} 维`);
const rowLabel=(H,h)=>H===4?tx(`<b>H${h+1}</b>: ${L(JOB[h])}`,`<b>H${h+1}</b>：${L(JOB[h])}`)
  :(H===2?tx(`<b>Head ${'AB'[h]}</b>: patterns ${2*h+1} and ${2*h+2} (${L(ROLE[2*h])}, ${L(ROLE[2*h+1])})`,`<b>head ${'AB'[h]}</b>：模式 ${2*h+1} 和 ${2*h+2}（${L(ROLE[2*h])}、${L(ROLE[2*h+1])}）`)
  :tx('<b>The one head</b>: the whole q, k and v','<b>唯一的 head</b>：完整的 q、k、v'));
const listJoin=a=>a.length<2?(a[0]||''):tx(a.slice(0,-1).join(', ')+(a.length>2?',':'')+' and '+a[a.length-1],a.slice(0,-1).join('、')+'和'+a[a.length-1]);
/* the words a head puts at least `min` of its weight on, largest first */
const tops=(o,min)=>o.w.map((w,j)=>({j,w})).filter(x=>x.w>=min).sort((a,b)=>b.w-a.w||a.j-b.j);
/* R6.20: explanations say how a head spreads its attention weight in words; the percentages stay on the chart, the metric cards and
   Show the math. From the share as the chart shows it: all (100%), almost all (90% or more), most (60% or more); below that the head
   splits its weight among its top words (10% or more each, up to 3) */
const LV={most:1,almost:2,all:3}, LVK=['','most','almost','all'];
function look(o){const t=tops(o,0.1), top=t[0]; if(!top) return {k:'even'};
  const v=rd(top.w*100,0);
  if(v>=60) return {k:v>=100?'all':(v>=90?'almost':'most'),j:top.j,w:top.w};
  if(t.length>=2) return {k:'split',js:t.slice(0,3).map(x=>x.j),j:top.j,w:top.w};
  return {k:'top',j:top.j,w:top.w};}
const LOOKW={en:{all:'only at',almost:'almost only at',most:'mostly at',top:'most at'},zh:{all:'只看',almost:'几乎只看',most:'主要看',top:'看得最多的是'}};
const AT=k=>tx(LOOKW.en[k],LOOKW.zh[k]);
const wlist=js=>listJoin(js.map(j=>wq(j)));
/* one head's look, no number: "looks almost only at “cat”" / "几乎只看“小猫”"; would: a head that is off */
function lookTxt(lk,would){
  const v=(s,w3)=>would?`would ${s}`:w3, zw=would?'本来会':'';
  if(lk.k==='even') return tx(`${v('spread','spreads')} its attention over every word it can see`,`${zw}把注意力分散在它能看到的所有词上`);
  if(lk.k==='split') return lk.js.length>2?tx(`${v('spread','spreads')} its attention over ${wlist(lk.js)}`,`${zw}把注意力分给了${wlist(lk.js)}`)
    :tx(`${v('split','splits')} its attention between ${wlist(lk.js)}`,`${zw}把注意力分给了${wlist(lk.js)}`);
  return tx(`${v('look','looks')} ${AT(lk.k)} ${wq(lk.j)}`,`${zw}${AT(lk.k)}${wq(lk.j)}`);}
/* the one head's look with its key number, the largest share as the chart shows it (one number per paragraph, R6.20) */
function lookNum(o){const lk=look(o), n=pc(lk.w||0), hd=tx('the one head','唯一的 head');
  if(lk.k==='even') return `${hd}${tx(' ','')}${lookTxt(lk)}`;
  if(lk.k==='split') return tx(`${hd} ${lookTxt(lk)}, ${n} on ${wq(lk.j)}`,`${hd}${lookTxt(lk)}，${wq(lk.j)}最多，${n}`);
  return tx(`${hd} puts ${n} of its attention weight on ${wq(lk.j)}`,`${hd}把 ${n} 的注意力权重放在${wq(lk.j)}上`);}
/* the link to the numbers the words stand for (R6.20: scores and q·k live in Show the math) */
const SEEMATH=()=>tx(` (see ${J('math',A.t('showMath'))})`,`（见${J('math','“'+A.t('showMath')+'”')}）`);

/* ---------- module state (survives re-mounts and language switches) ---------- */
let S={q:IT,H:1,on:ALL_ON()}, mode='guided', step=0, pred=null, observed={}, acted={}, focus=-1, root=null, app=null, saved=null;
let mounted=false, chSubmitted=false, chReveal=false, mathHead=0, drillSeed=3, drillOpen=false, annT=null, userAct=false, wantObs=false, fromRail=false;
const shownPrev={};

/* ---------- terms, defined where they first appear (R2.8) ---------- */
const rm=()=>realModel();
/* the site's shared definitions (A.terms, one wording for the whole site, R2.8), with a sentence added where this lab needs it */
const KT=(k,fb)=>A.terms&&A.terms[k]?A.terms[k]():fb;
const TIP={
 s0:()=>tx('<s> is the start token, placed before the first word. Many models put such a token in front of every text; it stands for no word.','<s> 是起始 token，放在第一个词前面。很多模型会在每段文字前面放一个这样的 token，它不代表任何词。'),
 token:()=>KT('token','Token.')+tx(' Here one word or punctuation mark is one token.','这里一个词或一个标点就是一个 token。'),
 hidden:()=>KT('hidden','Hidden size.')+tx(' Here d = 8; in Llama-3.1-8B it is 4,096.','这里 d = 8；Llama-3.1-8B 是 4,096。'),
 attn:()=>KT('attention','Attention block.')+tx(' This lab has one attention block, with hand-set parameters and no MLP block (MLP: multi-layer perceptron) after it.','这个 Lab 只有一个 attention 块，参数是手工设的，后面没有 MLP 块（MLP 即多层感知机）。'),
 weight:()=>KT('aweight','Attention weight.')+tx(' Here each token is one word, and each bar on the chart is one attention weight.','这里一个 token 就是一个词，图里每一根竖条就是一个注意力权重。'),
 head:()=>KT('head','Head.')+tx(' This lab’s block starts with one head; parts 5 and 6 split it into 2 and 4.','这个 Lab 的块先只有一个 head；第 5、6 部分把它拆成 2 个和 4 个。'),
 qkv:()=>KT('qkv','Query, key and value.')+tx(' In this lab each is 8 numbers, computed from all 8 of the token’s numbers; with several heads, each head takes its own slice of all three.','在这个 Lab 里，三个向量都是 8 维，由 token 的全部 8 维算出；有几个 head 时，每个 head 从三者里各取自己的一段。'),
 score:()=>tx(`A word’s score in a head says how well the head’s slice of the query matches that word’s slice of the key: the two multiplied number by number and added up, then divided by the square root of the slice’s size, head_dim. The softmax gives higher scores larger attention weights.<div class="fx">${A.ui.math('\\text{score} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}</div>`,`一个词在某个 head 里的得分，表示这个 head 那段 query 和这个词那段 key 有多匹配：两段逐维相乘再相加，再除以这一段维数 head_dim 的平方根。softmax 给得分高的词更大的注意力权重。<div class="fx">${A.ui.math('\\text{得分} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}</div>`),
 hd:()=>tx(`head_dim is how many of the 8 numbers of q, k and v each head gets: ${A.ui.math('\\text{head_dim} = 8 \\div \\text{heads}')}, so 8 with 1 head, 4 with 2 and 2 with 4. Dividing the scores by ${A.ui.math('\\sqrt{\\text{head_dim}}')} keeps them from growing just because a head has more numbers (Vaswani et al., 2017).`,`head_dim 是每个 head 从 q、k、v 的 8 维里各分到几维：${A.ui.math('\\text{head_dim} = 8 \\div \\text{head 数}')}，1 个 head 时是 8，2 个时是 4，4 个时是 2。得分除以 ${A.ui.math('\\sqrt{\\text{head_dim}}')}，是为了不让得分只因为 head 的维数多就变大（Vaswani 等，2017）。`),
 softmax:()=>KT('softmax','Softmax.')+tx(' In this lab each head turns its scores for the words it can see into attention weights.','在这个 Lab 里，每个 head 把它能看到的词的得分变成注意力权重。'),
 mask:()=>KT('mask','Causal mask.')+tx(' In this lab each token is a word.','在这个 Lab 里，每个 token 就是一个词。'),
 pattern:()=>tx('In this lab, Wq and Wk are built by hand from four patterns, each in 2 of the 8 numbers of q and k: the previous word, the first token, the word “it” refers to, and the nouns. With 4 heads each head gets one pattern and with 2 heads two; one head’s score runs over all 8 numbers, so it adds up all four.','在这个 Lab 里，Wq 和 Wk 是用四种模式手工拼出来的，每种占 q 和 k 的 8 维里的 2 维：前一个词、第一个 token、“它”指代的词和名词。4 个 head 时每个 head 分到一种，2 个 head 时各分到两种；只有一个 head 时，它的得分要算全部 8 维，所以是四种加起来。'),
 kvhead:()=>KT('kvhead',tx('A KV head is one key vector and one value vector that each token stores in every layer.','KV head 是每个 token 在每一层存的一个 key 向量和一个 value 向量。')),
 gqa:()=>{const m=rm(); return KT('gqa','GQA')+tx(' ','')+tx(`${m.name}, used here, has ${m.heads} query heads in ${m.kv} groups of ${m.group}, so each token stores ${m.kv} keys and ${m.kv} values per layer instead of ${m.heads} of each.`,`这里用的 ${m.name} 有 ${m.heads} 个 query head，分成 ${m.kv} 组、每组 ${m.group} 个，所以每个 token 每层只存 ${m.kv} 个 key 和 ${m.kv} 个 value，而不是各 ${m.heads} 个。`);},
};
const cite=(url,en,zh)=>`<a href="${url}" target="_blank" rel="noopener">${tx(en,zh)}</a>`;
const OLSSON=()=>cite('https://arxiv.org/abs/2209.11895','Olsson et al., 2022','Olsson 等，2022');
const MICHEL=()=>cite('https://arxiv.org/abs/1905.10650','Michel et al., 2019','Michel 等，2019');
const MICHEL_P=()=>cite('https://arxiv.org/abs/1905.10650','Michel et al. (2019)','Michel 等（2019）');
const VASWANI=()=>cite('https://arxiv.org/abs/1706.03762','Vaswani et al., 2017','Vaswani 等，2017');
const VOITA=()=>cite('https://arxiv.org/abs/1905.09418','Voita et al., 2019','Voita 等，2019');
const XIAO=()=>cite('https://arxiv.org/abs/2309.17453','Xiao et al., 2023','Xiao 等，2023');
const SU=()=>cite('https://arxiv.org/abs/2104.09864','Su et al., 2021','Su 等，2021');
const BHOJ=()=>cite('https://arxiv.org/abs/2002.07028','Bhojanapalli et al., 2020','Bhojanapalli 等，2020');
/* another lab by its short name, its question on hover (kit); quoted in Chinese running text */
const LK=id=>{const a=A.shell&&A.shell.labLink?A.shell.labLink(id):`<a href="#${id}">${esc(id)}</a>`; return tx(a,`“${a}”`);};
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
/* R3.12: the guided tour ends in Scenarios, Scenarios end in the challenge (kit buttons, data-mode-go) */
const TOSCEN=()=>A.shell&&A.shell.toScenarios?A.shell.toScenarios():`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="sandbox">${tx('See the scenarios','去看应用场景')}</button></div>`;
const TOCHAL=()=>A.shell&&A.shell.toChallenge?A.shell.toChallenge():`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="challenge">${tx('Take the challenge','去做挑战')}</button></div>`;
/* part 3 is about words other than “it”; “it” is part 4's (the one head there answers the prediction) */
const qLock=j=>mode==='guided'&&STEPS[step]&&STEPS[step].id==='look'&&j===IT;

/* ---------- guided parts ---------- */
/* the prediction is about what one head can do at all: its attention weights add up to 100%, so two words cannot both get 80% (80% + 80% > 100%).
   The answer therefore does not depend on the hand-set numbers; what this lab's one head then does (43% and 41%) is shown in part 4. */
const PRED=[{v:'both',t:{en:'Both can get 80% or more',zh:'两个都能在 80% 以上'}},{v:'one',t:{en:'At most one can get 80% or more',zh:'最多一个能在 80% 以上'}}];
const predRight=()=>'one';
const predT=v=>L(PRED.find(o=>o.v===v).t);
function verdict(){if(!pred) return tx(`<p>One head’s attention weights add up to 100%, so at most one word can get ${pc(CH_GOAL)} or more.</p>`,`<p>一个 head 的注意力权重加起来是 100%，所以最多只有一个词能在 ${pc(CH_GOAL)} 以上。</p>`);
  const right=predRight(), pt=esc(predT(pred)), rt=esc(predT(right));
  const why=tx('one head’s attention weights add up to 100%, so no two words can both get more than half.','一个 head 的注意力权重加起来是 100%，所以不可能有两个词都拿到一半以上。');
  return pred===right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right</span>: ${why}</p>`,`<p><span class="okc">你的预测“${pt}”是对的</span>：${why}</p>`)
                     :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”: ${why}</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”：${why}</p>`);}
const pickerHTML=(id,label)=>`<div class="inline-ctl" id="${id}"><span class="lbl">${esc(label)}</span><div class="mha-words" role="group" aria-label="${esc(label)}">${SENT[LG()].map((w,j)=>`<button type="button" data-q="${j}" aria-pressed="${j===S.q}">${esc(w)}</button>`).join('')}</div></div>`;
const QLABEL=()=>tx('Current word','当前的词');
const HLABEL=()=>tx('Number of heads','head 数');
const headSeg=(id,inline,note,only)=>A.ui.seg({id,key:'H',label:HLABEL(),options:[['1',tx('1 head','1 个')],['2',tx('2 heads','2 个')],['4',tx('4 heads','4 个')]].filter(o=>!only||only.includes(+o[0])),value:String(S.H),inline,note});
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'At “it”, the model needs both “because” and “cat”.',zh:'到了“它”，模型既要看“因为”，也要看“小猫”。'},enter:()=>({q:IT,H:1,on:ALL_ON()}),
  body:()=>tx(`<p>A model has read “The cat sat on the mat because it” and must predict the next word. It sees 9 ${DEF('tokens',TIP.token())}: a start token, ${DEF('&lt;s&gt;',TIP.s0())}, and one token per word; each token is a vector of 8 numbers (the ${DEF('hidden size d',TIP.hidden())}). The model here is one ${DEF('attention block',TIP.attn())} whose parameters are set by hand, so you can check what it does.</p><p>At “it”, the next word depends on two earlier words: “because”, just before it, and the word “it” refers to, which in this lab is “cat”. The block gives every word so far an ${DEF('attention weight',TIP.weight())}; it starts with one ${DEF('head',TIP.head())}, whose attention weights add up to 100%.</p><p>Your job: see what one head can and cannot do at “it”, then split the block into 2 heads and 4, and watch what changes.</p>`,
              `<p>模型读到了“小猫坐在垫子上，因为它”，要预测下一个词。它看到 9 个 ${DEF('token',TIP.token())}：一个起始 token ${DEF('&lt;s&gt;',TIP.s0())}，每个词或标点各一个 token；每个 token 是一个 8 维的向量（这个维数叫${DEF('隐藏维度 d',TIP.hidden())}）。这里的模型只有一个 ${DEF('attention 块',TIP.attn())}，参数是手工设的，方便你检查它在做什么。</p><p>到了“它”，下一个词取决于前面两个词：紧挨着的“因为”，以及“它”指代的词，在这个 Lab 里是“小猫”。attention 块给到目前为止的每个词一个${DEF('注意力权重',TIP.weight())}；它先只有一个 ${DEF('head',TIP.head())}，这个 head 的注意力权重加起来是 100%。</p><p>你的任务：先看一个 head 在“它”这里能做到什么、做不到什么，再把这个块拆成 2 个和 4 个 head，看有什么变化。</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: can one head give both words 80% or more?',zh:'先猜：一个 head 能不能给两个词都分到 80% 以上？'},enter:()=>({q:IT,H:1,on:ALL_ON()}),gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
    return tx(`<p>The block has one head, and at “it” that head needs both “because” and “cat”. How much attention weight can the two words get?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 4 shows the result.</p>':''}`,
              `<p>这个块只有一个 head，而它在“它”这个位置既要看“因为”，又要看“小猫”。这两个词能拿到多少注意力权重？</p>${ch}${pred?'<p class="muted small">已记下。第 4 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'look',title:{en:'How one head looks',zh:'一个 head 怎么看'},lead:{en:'A head scores every word against the query and turns the scores into weights.',zh:'head 拿 query 给每个词打分，再换算成注意力权重。'},enter:()=>({q:3,H:1,on:ALL_ON()}),expect:()=>!!acted[SIX('look')],showMe:{q:3},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>One head uses the whole ${DEF('query, key and value',TIP.qkv())}: Wq, Wk and Wv turn each token’s 8 numbers into a query (what the token looks for), a key (what queries are compared with) and a value (what it hands over).</p><p>The head gives each word a ${DEF('score',TIP.score())} that says how well the query matches that word’s key. It multiplies the two number by number, adds the products up, and divides by the square root of ${DEF('head_dim',TIP.hd())}, the size of each head’s slice:</p><div class="mha-math-line">${A.ui.math('\\text{score} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}</div><p>A ${DEF('softmax',TIP.softmax())} turns the scores into attention weights that add up to 100%, and the head mixes the values by them.</p><p>Pick a word; “it” comes in part 4.</p>`,
              `<p>一个 head 用的是完整的 ${DEF('query、key、value',TIP.qkv())}：Wq、Wk、Wv 把每个 token 的 8 维向量变成 query（这个 token 要找什么）、key（拿来和 query 比较的）和 value（被 head 取用时交出去的信息）。</p><p>head 给每个词打一个${DEF('得分',TIP.score())}，表示 query 和这个词的 key 有多匹配。它把两者逐维相乘再相加，再除以 ${DEF('head_dim',TIP.hd())} 的平方根（head_dim 是每个 head 分到的维数）：</p><div class="mha-math-line">${A.ui.math('\\text{得分} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}</div><p>${DEF('softmax',TIP.softmax())} 把得分换成加起来是 100% 的注意力权重，head 再按这些注意力权重混合各词的 value。</p><p>选一个词；“它”留到第 4 部分。</p>`)+pickerHTML('mha-ic-q',QLABEL()),
  obs:r=>obsLook(r)},
 {id:'one',title:{en:'One head at “it”',zh:'一个 head 看“它”'},lead:{en:'Make “it” the current word: one softmax has to serve both words.',zh:'把“它”设为当前的词：一次 softmax 要同时照顾两个词。'},enter:()=>({q:3,H:1,on:ALL_ON()}),expect:()=>S.q===IT&&S.H===1,showMe:{q:IT,H:1},label:{en:'Observed',zh:'观察到'},
  body:()=>tx('<p>Now make “it” the current word. The one head has to serve both words “it” needs with a single softmax.</p>','<p>现在把“它”设为当前的词。这一个 head 要用同一次 softmax 同时照顾“它”需要的两个词。</p>')+pickerHTML('mha-ic-q',QLABEL()),
  obs:r=>obsOne(r)},
 {id:'two',title:{en:'Two heads',zh:'两个 head'},lead:{en:'Cut q, k and v in half: 2 heads, each with its own softmax.',zh:'把 q、k、v 切成两半：2 个 head，各做各的 softmax。'},enter:()=>({q:IT,H:1,on:ALL_ON()}),expect:()=>S.H===2&&S.q===IT,showMe:{q:IT,H:2},label:{en:'Observed',zh:'观察到'},
  body:()=>tx('<p>Now cut q, k and v in half: 2 heads of 4 numbers, each with its own softmax. Wq, Wk and Wv stay the same; only which numbers each head uses changes.</p><p>Switch to 2 heads.</p>',
              '<p>现在把 q、k、v 切成两半：2 个 head，每个 4 维，各做各的 softmax。Wq、Wk、Wv 都不变，变的只是每个 head 用哪几维。</p><p>切换到 2 个 head。</p>')+headSeg('mha-ic-H',true,undefined,[1,2]),
  obs:r=>obsTwo(r)},
 {id:'four',title:{en:'Four heads',zh:'四个 head'},lead:{en:'4 heads: each holds just one of the four patterns.',zh:'4 个 head：每个只装四种模式中的一种。'},enter:()=>({q:IT,H:2,on:ALL_ON()}),expect:()=>S.H===4&&S.q===IT,showMe:{q:IT,H:4},label:{en:'Observed',zh:'观察到'},
  body:()=>tx('<p>Cut them again: 4 heads of 2 numbers each. Now each head holds just one of the four patterns.</p><p>Switch to 4 heads.</p>',
              '<p>再切一次：4 个 head，每个 2 维。现在每个 head 只装四种模式中的一种。</p><p>切换到 4 个 head。</p>')+headSeg('mha-ic-H',true,undefined,[2,4]),
  obs:r=>obsFour(r)},
 {id:'real',title:{en:'Heads in a real model',zh:'真实模型里的 head'},lead:{en:'Llama‑3.1‑8B cuts each query into 32 heads of 128 numbers.',zh:'Llama‑3.1‑8B 把每个 query 切成 32 个 head，每个 128 维。'},enter:()=>({q:IT,H:4,on:ALL_ON()}),
  body:()=>{const m=rm();
    return tx(`<p>${m.name} splits its queries the same way at full size: in every layer, each token’s query is cut into ${m.heads} heads of ${m.hd} numbers (its head_dim), each with its own softmax.</p><p>${m.name} also lets every ${m.group} heads share one key and one value, which is why its config.json lists ${m.kv} ${DEF('KV heads',TIP.kvhead())}; this is called grouped-query attention (${DEF('GQA',TIP.gqa())}), and the lab ${LK('gqa-mla')} covers it.</p>`,
              `<p>${m.name} 在全尺寸上也这样拆 query：每一层里，每个 token 的 query 切成 ${m.heads} 个 head，每个 ${m.hd} 维（它的 head_dim），每个 head 各做各的 softmax。</p><p>${m.name} 还让每 ${m.group} 个 head 共用一份 key 和 value，所以 config.json 里写着 ${m.kv} 个 ${DEF('KV head',TIP.kvhead())}；这种做法叫 grouped-query attention（${DEF('GQA',TIP.gqa())}），在${LK('gqa-mla')}这个 Lab 里讲。</p>`);}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:()=>({q:IT,H:4,on:ALL_ON()}),
  body:()=>{const lg=LG(), one=compute({lang:lg,q:IT,H:1}).heads[0], two=compute({lang:lg,q:IT,H:2}), ci=two.cat_i, m=rm();
    const mid=compute({lang:lg,q:3,H:4}), b=esc(wd(BECAUSE)), c=esc(wd(ci));
    const masked3=pc(Math.max(...mid.heads.map(o=>o.w.slice(4).reduce((a,v)=>a+v,0))));
    /* R6.20: the ideas in words; the shares are in parts 4 and 5 and on the chart. Each phrase is checked against the engine */
    const even=Math.abs(one.w[BECAUSE]-one.w[ci])<0.05, under=one.w[BECAUSE]<0.5&&one.w[ci]<0.5, near=Math.min(one.w[BECAUSE],one.w[ci])>=0.35, lA=look(two.heads[0]), lB=look(two.heads[1]), own=lA.j===BECAUSE&&!!LV[lA.k]&&lB.j===ci&&!!LV[lB.k];
    const oneEn=`In this lab “${b}” and “${c}” ${under?`each got ${near?'a little ':''}under half`:`got ${even?'almost equal':'close'} shares`}.`, oneZh=`在这个 Lab 里，“${b}”和“${c}”${under?`都${near?'只':''}拿到不到一半`:`的份额${even?'差不多':'相近'}`}。`;
    const twoEn=own?`with 2 heads, head A looks mostly at “${b}” and head B at “${c}”`:`with 2 heads, each head takes from its own words`, twoZh=own?`2 个 head 时，head A 主要看“${b}”，head B 主要看“${c}”`:`2 个 head 时，每个 head 从自己的词取信息`;
    return tx(`<ul><li>One head’s attention weights add up to 100%, so no two words can both get more than half. When a word needs two other words at once, one head has to split its attention between them, and neither gets most of it.</li><li>Several heads cut the same query, key and value into slices, each with its own softmax, so one block can take from several words at once, each head from its own.</li><li>Splitting adds no parameters and no multiplications, so real models use many heads: ${m.name} has ${m.heads} in every layer.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “More heads, more parameters.” Every head here keeps its own key and value, and the block has ${NPAR} parameters however it is split. In ${m.name}, only Wk and Wv change with the head count, because its ${m.kv} KV heads stay fixed.</p><p>✗ “Each head sees only a slice of the token.” Each head’s query, key and value are computed from all 8 of the token’s numbers; only the results are cut into slices.</p><p>✗ “A model that writes text can look at later words.” The causal mask gives every later word ${masked3} of the attention weight, so the model cannot peek at words it has not written yet.</p></div>${TOSCEN()}`,
              `<ul><li>一个 head 的注意力权重加起来是 100%，所以不可能有两个词都拿到一半以上。当前的词同时需要两个词时，一个 head 只能把注意力分给两边，哪一边都拿不到大头。</li><li>多个 head 把同一个 query、key、value 切成几段，每段各做一次 softmax，所以一个块能同时从几个词取信息，每个 head 取自己的那个。</li><li>拆成多个 head 不增加参数，也不增加乘法，所以真实模型用很多 head：${m.name} 每层有 ${m.heads} 个。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “head 越多，参数越多。”这里每个 head 各有 key 和 value，不管怎么拆，这个块都是 ${NPAR} 个参数。${m.name} 改 head 数时，只有 Wk 和 Wv 会变，因为它的 ${m.kv} 个 KV head 不变。</p><p>✗ “每个 head 只看到 token 的一部分。”每个 head 的 query、key、value 都由 token 的全部 8 维算出，拆开的只是算出来的结果。</p><p>✗ “生成文字的模型能看后面的词。”因果 mask 让后面每个词的注意力权重都是 ${masked3}，模型看不到还没写出来的词。</p></div>${TOSCEN()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);

/* part 3: where the one head looks at a word other than “it”, and why: the softmax turns its scores into shares. The result in
   words with one number, the largest share; the scores themselves are in Show the math (R6.20) */
function obsLook(r){
  const p=r.p, later=NT-1-p;
  const maskS=later>0?tx(`The ${later===1?'word after it gets':'words after it get'} no attention weight: the ${DEF('causal mask',TIP.mask())} hides words not written yet.`,
                         `后面的词分不到注意力权重：${DEF('因果 mask',TIP.mask())} 挡住了还没写出来的词。`):'';
  if(r.H!==1) return describeState(r);
  if(p===0) return tx(`<p><b>At &lt;s&gt;, the first token, the one head has nothing before it: it puts ${pc(r.heads[0].w[0])} of its attention weight on &lt;s&gt; itself.</b> ${maskS}</p>`,
                      `<p><b>当前的词是 &lt;s&gt; 时，它是第一个 token，前面什么都没有：唯一的 head 把 ${pc(r.heads[0].w[0])} 的注意力权重放在 &lt;s&gt; 自己身上。</b>${maskS}</p>`);
  const lk=look(r.heads[0]), close=lk.k==='split';
  const whyEn=close?`${wlist(lk.js)} have close scores, so they share most of the attention weight`:`${wq(lk.j)} has the highest score, so it gets the largest share`;
  const whyZh=close?`${wlist(lk.js)}的得分很接近，所以它们分走了大部分注意力权重`:`${wq(lk.j)}的得分最高，所以份额最大`;
  return tx(`<p><b>At ${wq(p)}, ${lookNum(r.heads[0])}.</b> ${maskS}</p><p>Why: the softmax turns the scores into shares that add up to 100%. ${whyEn}${SEEMATH()}.</p>`,
            `<p><b>当前的词是${wq(p)}时，${lookNum(r.heads[0])}。</b>${maskS}</p><p>原因：softmax 把得分换成加起来是 100% 的份额。${whyZh}${SEEMATH()}。</p>`);
}
/* “because” and “cat” as they appear in a head: "43% on “because” and 41% on “cat”" */
const twoTxt=(o,ci)=>tx(`${pc(o.w[BECAUSE])} on “${esc(wd(BECAUSE))}” and ${pc(o.w[ci])} on “${esc(wd(ci))}”`,`在“${esc(wd(BECAUSE))}”上 ${pc(o.w[BECAUSE])}，在“${esc(wd(ci))}”上 ${pc(o.w[ci])}`);
function stateLead(r){   // the first sentence for a head count at “it”
  const ci=r.cat_i, H=r.H, it=esc(wd(IT)), b=esc(wd(BECAUSE)), c=esc(wd(ci));
  if(H===1){const o=r.heads[0], half=o.w[BECAUSE]<0.5&&o.w[ci]<0.5;
    return tx(`<p><b>One head: at “${it}” it puts ${twoTxt(o,ci)}${half?', neither above half':''}.</b></p>`,`<p><b>只有一个 head：当前的词是“${it}”时，它${twoTxt(o,ci)}${half?'，都不到一半':''}。</b></p>`);}
  if(H===2) return tx(`<p><b>Two heads: at “${it}”, head A puts ${pc(r.heads[0].w[BECAUSE])} on “${b}” and head B puts ${pc(r.heads[1].w[ci])} on “${c}”.</b></p>`,
                      `<p><b>两个 head：当前的词是“${it}”时，head A 在“${b}”上 ${pc(r.heads[0].w[BECAUSE])}，head B 在“${c}”上 ${pc(r.heads[1].w[ci])}。</b></p>`);
  return tx(`<p><b>Four heads: at “${it}”, H1 puts ${pc(r.heads[0].w[BECAUSE])} on “${b}” and H${HC+1} ${pc(r.heads[HC].w[ci])} on “${c}”.</b></p>`,
            `<p><b>4 个 head：当前的词是“${it}”时，H1 在“${b}”上 ${pc(r.heads[0].w[BECAUSE])}，H${HC+1} 在“${c}”上 ${pc(r.heads[HC].w[ci])}。</b></p>`);
}
/* one head at “it”: its q·k adds up the four patterns, and the two words it needs come out almost equal. In words (R6.20): the q·k
   values are in Show the math, the two shares in the lead above; where the rest of the 100% goes is named (N4) */
function oneWhy(r,inPart){
  const o=r.heads[0], ci=r.cat_i, g=pc(CH_GOAL), rest=tops(o,0.05).filter(x=>x.j!==BECAUSE&&x.j!==ci).sort((x,y)=>x.j-y.j);
  const both=o.w[BECAUSE]<CH_GOAL&&o.w[ci]<CH_GOAL, half=o.w[BECAUSE]<0.5&&o.w[ci]<0.5, near=Math.abs(o.dots[BECAUSE]-o.dots[ci])<1, b=esc(wd(BECAUSE)), c=esc(wd(ci)), rw=wlist(rest.map(x=>x.j));
  const leadEn=inPart&&both?`Why neither gets ${g} in this lab: with one head, `:'With one head, ', leadZh=inPart&&both?`为什么在这个 Lab 里两个都不到 ${g}：只有一个 head 时，`:'只有一个 head 时，';
  return tx(`<p>${leadEn}${A.ui.math('q \\cdot k')} runs over the whole query and key, so each word’s ${A.ui.math('q \\cdot k')} adds up all four ${DEF('patterns',TIP.pattern())}. “${b}” and “${c}” come out ${near?'almost equal':'close'}${SEEMATH()}, so the one softmax gives them ${near?'almost equal':'similar'} shares${rest.length?`, and what is left goes to ${rw}`:''}. The output carries ${half?'less than half of each one’s value':'only those shares of their values'}.</p>`,
            `<p>${leadZh}${A.ui.math('q \\cdot k')} 要用完整的 query 和 key，所以每个词的 ${A.ui.math('q \\cdot k')} 是四种${DEF('模式',TIP.pattern())}之和。“${b}”和“${c}”的 ${A.ui.math('q \\cdot k')} ${near?'几乎一样高':'很接近'}${SEEMATH()}，一次 softmax 就给了它们差不多的份额${rest.length?`，剩下的给了${rw}`:''}。输出里，${half?'两个词的 value 都只有不到一半':'两个词的 value 也只有这么多'}。</p>`);
}
/* two heads at “it”: head A holds the previous-word and first-token patterns, head B the “it”-refers-to and noun patterns */
function twoWhy(r){const A_=r.heads[0], B_=r.heads[1], most=A_.out[0]>=0.5&&B_.out[1]>=0.5;
  return tx(`<p>Why: head A gets the first half, which holds the previous-word and first-token ${DEF('patterns',TIP.pattern())}; head B gets the second half, with the patterns for the word “it” refers to and for the nouns. Each head has its own softmax, so the block’s output now carries ${most?'most of both words’ values':'some of both words’ values'}.</p>`,
            `<p>原因：head A 分到前一半，里面是“前一个词”和“第一个 token”两种${DEF('模式',TIP.pattern())}；head B 分到后一半，里面是“它指代的词”和“名词”两种。两个 head 各做各的 softmax，所以块的输出里现在有两个词${most?'的大部分 value':'各自的一部分 value'}。</p>`);}
/* the line under a part's result before its action is done: what to do next */
const nudge=(en,zh)=>tx(`<p>${en}</p>`,`<p>${zh}</p>`);
function obsOne(r){
  const seen=!!observed[SIX('one')];
  const sw=pred?nudge('Pick “it” to see whether your prediction was right.','选“它”，看你的预测对不对。'):nudge('Pick “it” to see what one head does there.','选“它”，看一个 head 在那里会怎样。');
  if(r.p!==IT||r.H!==1) return describeState(r)+(seen?verdict():sw);
  return stateLead(r)+verdict()+oneWhy(r,true);
}
function obsTwo(r){
  const it=esc(wd(IT));
  if(r.p!==IT) return describeState(r)+nudge(`Make “${it}” the current word again to see the two words it needs.`,`把“${it}”重新设为当前的词，看它需要的两个词。`);
  if(r.H!==2) return stateLead(r)+nudge('Switch to 2 heads.','切换到 2 个 head。');
  return stateLead(r)+twoWhy(r)+tx(`<p>Nothing was added: the same four matrices (${NPAR} parameters) and the same multiplications to score a word, now split between the heads.</p>`,`<p>什么都没多加：还是这四个矩阵（${NPAR} 个参数），给一个词打分的乘法也一样多，只是分给了两个 head。</p>`);
}
function obsFour(r){
  const it=esc(wd(IT)), ci=r.cat_i;
  if(r.p!==IT) return describeState(r)+nudge(`Make “${it}” the current word again to see the two words it needs.`,`把“${it}”重新设为当前的词，看它需要的两个词。`);
  if(r.H!==4) return stateLead(r)+nudge('Switch to 4 heads.','切换到 4 个 head。');
  return tx(`<p><b>Four heads: at “${it}”, ${fourIt(r)}.</b></p><p>Each head does one job at the same time: the previous word, the first token, the word “${it}” refers to, the nouns. More heads let one block look up more things at once, with the same ${NPAR} parameters.</p><p>Each head does one clean job only because this lab set Wq and Wk that way by hand; in trained models few heads are this clean, though some mostly attend to the previous token (${OLSSON()}).</p>`,
            `<p><b>4 个 head：当前的词是“${it}”时，${fourIt(r)}。</b></p><p>每个 head 同时干一件事：前一个词、第一个 token、“${it}”指代的词、名词。head 多了，一个块就能同时查更多的东西，参数还是那 ${NPAR} 个。</p><p>每个 head 只干一件干净的事，是因为这个 Lab 手工这样设了 Wq 和 Wk；训练出来的模型里，这么干净的 head 很少，不过确实有 head 主要看前一个 token（${OLSSON()}）。</p>`);
}
/* 4 heads at “it” in words (R6.20): H1, H2 and H3 each put nearly all of their attention weight on one word, H4 splits its weight
   between the nouns; the percentages are on the chart and the metric cards. Any other shape falls back to one clause per head */
function fourIt(r){const o=r.heads, L3=[0,1,2].map(h=>look(o[h])), l4=look(o[3]);
  if(L3.some(x=>!LV[x.k])) return describeHeads(r);
  const k=LVK[Math.min(...L3.map(x=>LV[x.k]))];
  return tx(`H1 looks ${AT(k)} ${wq(L3[0].j)}, H2 at ${wq(L3[1].j)} and H3 at ${wq(L3[2].j)}; H4 ${lookTxt(l4)}`,
            `H1、H2、H3 各自${AT(k)}一个词，分别是${wq(L3[0].j)}、${wq(L3[1].j)}和${wq(L3[2].j)}；H4 ${lookTxt(l4)}`);}
/* any state in plain sentences (R6.20): where each head puts its attention weight, in words, and heads that put most of it on the
   same word share one clause; the percentages are on the chart. With one head the clause carries the one number, its largest share.
   A head that is off still computes its weights, but the block drops its output */
function headLabel(r,h){const H=r.H, role=H===4?L(ROLE[h]):'';
  return H===4?(!r.heads[h].on?tx(`${hn(H,h)} (${role}; turned off, so the block drops its output)`,`${hn(H,h)}（${role}；已关掉，块不用它的输出）`):tx(`${hn(H,h)} (${role})`,`${hn(H,h)}（${role}）`)):hn(H,h);}
function describeHeads(r){
  if(r.H===1) return lookNum(r.heads[0]);
  const items=[], grp={};
  r.heads.forEach((o,h)=>{const lk=look(o), lab=headLabel(r,h);
    if(o.on&&LV[lk.k]){const g=grp[lk.j]; if(g){g.labs.push(lab); g.lv=Math.min(g.lv,LV[lk.k]);} else {grp[lk.j]={j:lk.j,labs:[lab],lv:LV[lk.k]}; items.push(grp[lk.j]);}}
    else items.push({lab,lk,off:!o.on});});
  return items.map(x=>{
    if(!x.labs) return tx(`${x.lab} ${lookTxt(x.lk,x.off)}`,`${x.lab}${lookTxt(x.lk,x.off)}`);
    const k=LVK[x.lv];
    return x.labs.length>1?tx(`${listJoin(x.labs)} look ${AT(k)} ${wq(x.j)}`,`${listJoin(x.labs)}都${AT(k)}${wq(x.j)}`)
      :tx(`${x.labs[0]} looks ${AT(k)} ${wq(x.j)}`,`${x.labs[0]}${AT(k)}${wq(x.j)}`);}).join(tx('; ','；'));}
function describeState(r,noMask){
  const lead=tx(`<p><b>At ${wq(r.p)}, with ${splitTxt(r.H)}:</b> ${describeHeads(r)}.</p>`,`<p><b>当前的词是${wq(r.p)}，${splitTxt(r.H)}：</b>${describeHeads(r)}。</p>`);
  const later=NT-1-r.p;
  return lead+(!noMask&&later>0?tx(`<p>The ${later===1?'word after':'words after'} ${wq(r.p)} get${later===1?'s':''} no attention weight: the causal mask.</p>`,`<p>${wq(r.p)}后面的词分不到注意力权重，这是因果 mask。</p>`):'');}

/* ---------- scenarios: situations on the same sentence (R2.1) ---------- */
const PRESETS=[
 {id:'it',t:()=>tx('Who does “it” refer to?','“它”指的是谁？'),d:()=>tx('The word “it”, with the block’s 4 heads (initial setup)','“它”这个词，这个块的 4 个 head（初始设置）'),s:{q:IT,H:4}},
 {id:'mid',t:()=>tx(`Mid-sentence: “${wd(3)}”`,`写到一半：“${wd(3)}”`),d:()=>tx(`When the model writes “${wd(3)}”, the words after it do not exist yet`,`模型写到“${wd(3)}”时，后面的词还没写出来`),s:{q:3,H:4}},
 {id:'first',t:()=>tx('The first word','第一个词'),d:()=>tx(`“${wd(1)}” has almost nothing before it to look at`,`“${wd(1)}”前面几乎没有词可看`),s:{q:1,H:4}},
 {id:'one',t:()=>tx('A model with one head per layer','每层只有一个 head 的模型'),d:()=>tx('What a single-head block does at “it”','单 head 的块在“它”这里怎么分注意力权重'),s:{q:IT,H:1}},
 {id:'two',t:()=>tx('A model with two heads per layer','每层两个 head 的模型'),d:()=>tx('What a two-head block does at “it”','两个 head 的块在“它”这里怎么分注意力权重'),s:{q:IT,H:2}}
];
const allOn=()=>S.on.every(Boolean);
const presetNow=()=>PRESETS.findIndex(p=>p.s.q===S.q&&p.s.H===S.H&&allOn());
function insightHTML(r){
  const i=presetNow(), id=i>=0?PRESETS[i].id:'';
  if(id==='it'){const ci=r.cat_i;
    return tx(`<p><b>At “${esc(wd(IT))}”, each head does the one job it was given in this lab.</b> ${fourIt(r)}.</p><p>Each head has its own softmax, so the block takes “${esc(wd(BECAUSE))}” and “${esc(wd(ci))}” at once.</p>`,
              `<p><b>当前的词是“${esc(wd(IT))}”时，每个 head 各干在这个 Lab 里分给它的那件事。</b>${fourIt(r)}。</p><p>每个 head 各做各的 softmax，所以这个块能同时拿到“${esc(wd(BECAUSE))}”和“${esc(wd(ci))}”。</p>`);}
  if(id==='mid')
    return describeState(r,true)+tx(`<p>The words after ${wq(r.p)} get no attention weight: when a model writes ${wq(r.p)}, they do not exist yet (the causal mask).</p><p>H3 finds nothing to look for at ${wq(r.p)}, so it rests on &lt;s&gt;.</p>`,
                                    `<p>${wq(r.p)}后面的词分不到注意力权重：模型写到${wq(r.p)}时，它们还没写出来（因果 mask）。</p><p>H3 在${wq(r.p)}这里没有要找的东西，就停在 &lt;s&gt; 上。</p>`);
  if(id==='first') return describeState(r)+tx(`<p>H3 has nothing to look for at ${wq(r.p)}, so it rests on &lt;s&gt;, and H2 always looks at &lt;s&gt;; here &lt;s&gt; is also the previous word.</p><p>A head’s attention weights always add up to 100%, even when no earlier word is useful. Trained models often park such attention weight on the first token, an attention sink (${XIAO()}).</p>`,
                                               `<p>H3 在${wq(r.p)}这里没有要找的词，就停在 &lt;s&gt; 上；H2 本来就只看 &lt;s&gt;。而 &lt;s&gt; 在这里恰好是前一个词。</p><p>一个 head 的注意力权重总要加起来是 100%，哪怕前面没有有用的词。训练出来的模型常把这部分注意力权重停在第一个 token 上，叫 attention sink（${XIAO()}）。</p>`);
  if(id==='one') return stateLead(r)+oneWhy(r,false);
  if(id==='two') return stateLead(r)+twoWhy(r);
  return describeState(r)+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
}

/* ---------- views ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. Predict first, then build the block up from one head to two and four, and see what each split changes. Start here if this is new to you.','一共 8 个部分。先预测，再把这个块从一个 head 加到两个、四个，看每次加 head 改变了什么。第一次学从这里开始。'),
  sandbox:tx('Five situations on the same sentence, from the word “it” to the first word. Pick one to see what each head looks at.','同一句话里的五种情况，从“它”到第一个词。选一个，看每个 head 在看什么。'),
  challenge:tx('The block has 4 heads. Turn off as many as you can while, at “it”, some head still puts at least 80% on “because” and some head at least 80% on “cat”.','这个块有 4 个 head。关掉尽量多的 head，同时在“它”这个位置，开着的 head 里仍要有一个在“因为”上至少 80%，也要有一个在“小猫”上至少 80%。')
});
function FIDPOP(){const m=rm();
 return tx(
 `<p><b>Toy run:</b> your browser computes real multi-head attention on a sentence of 9 tokens; the parameters of its matrices are set by hand.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>One attention block as in a decoder-only Transformer: q, k and v from Wq, Wk and Wv, the score ${A.ui.math('\\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}, the causal mask, a softmax per head, each head’s sum of values weighted by its attention weights, the heads’ outputs side by side, times Wo.</li><li>One set of Wq, Wk, Wv and Wo (256 parameters), split into 1, 2 or 4 heads; only the split changes, as in real multi-head attention (MHA).</li><li>A head that is off adds nothing to the output, as in head pruning.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>The parameters are set by hand: Wq and Wk hold four patterns (the previous word, the first token, the word “it” refers to, the nouns), so each of the 4 heads does one clean job. Trained heads rarely do.</li><li>Hidden size 8 and one attention block, with no MLP block (MLP: multi-layer perceptron; ${m.name}: ${fi(m.d)} and ${m.layers} layers). One word is one token here.</li><li>The position is two of the 8 numbers in each token’s vector; ${m.name} uses rotary position embedding (RoPE) instead (${SU()}).</li><li>Values mark only “because” and “cat”, so a head’s output reads as how much of each arrived; Wo is the identity.</li><li>Splitting is free here, but heads cannot be cut arbitrarily thin: with a small head_dim, each head can express fewer attention patterns (${BHOJ()}).</li></ul></div><div><div class="eyebrow">Sources</div><ul><li>Multi-head attention, the division by ${A.ui.math('\\sqrt{\\text{head_dim}}')}, and “With a single attention head, averaging inhibits this”: ${VASWANI()}, §3.2.</li><li>${OLSSON()}: trained transformers have previous-token heads.</li><li>${VOITA()}: in a trained translation model a few specialized heads do most of the work, and most of the others can be pruned with little loss.</li><li>${MICHEL()}: many heads of trained models can be removed at test time with little effect.</li><li>${XIAO()}: trained models park a large share of attention on the first token, an attention sink; in this lab H2 always does, and H3 does at every word except “it”.</li><li>${m.name}: ${m.heads} heads that share ${m.kv} KV heads (part 7), head_dim ${m.hd}, from its config.json.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-09-25. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>浏览器在一个 9 个 token 的句子上真的算一遍多头注意力，矩阵的参数是手工设的。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>decoder-only Transformer 里的一个 attention 块：用 Wq、Wk、Wv 算出 q、k、v，得分 ${A.ui.math('\\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}')}，因果 mask，每个 head 各做一次 softmax，各自按注意力权重把 value 加起来，再把各 head 的输出拼在一起，乘以 Wo。</li><li>同一套 Wq、Wk、Wv、Wo（256 个参数），拆成 1、2 或 4 个 head；只有拆法变了，和真实的多头注意力（MHA）一样。</li><li>关掉的 head 不往输出里加任何东西，和剪掉 head 的做法一样。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>参数是手工设的：Wq 和 Wk 装着四种模式（前一个词、第一个 token、“它”指代的词、名词），所以 4 个 head 各干一件干净的事。训练出来的 head 很少这样。</li><li>隐藏维度只有 8，只有一个 attention 块，没有 MLP 块（MLP 即多层感知机；${m.name} 是 ${fi(m.d)} 维、${m.layers} 层）。这里一个词（或一个标点）就是一个 token。</li><li>位置只是每个 token 的向量里的两维；${m.name} 用的是旋转位置编码（RoPE，${SU()}）。</li><li>value 只标记“因为”和“小猫”，所以 head 的输出直接读成两者各到了多少；Wo 是单位矩阵。</li><li>在这个 Lab 里拆开不花代价，但 head 不能无限切细：head_dim 很小时，每个 head 能表达的注意力模式有限（${BHOJ()}）。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>多头注意力、除以 ${A.ui.math('\\sqrt{\\text{head_dim}}')}，以及“With a single attention head, averaging inhibits this”（只有一个 head 时，求平均会妨碍这一点）：${VASWANI()}，§3.2。</li><li>${OLSSON()}：训练出来的 Transformer 里有 previous-token head。</li><li>${VOITA()}：在一个训练好的翻译模型里，少数专门化的 head 干了大部分活，其余大多数剪掉后几乎不影响效果。</li><li>${MICHEL()}：训练好的模型在测试时可以去掉很多 head 而影响很小。</li><li>${XIAO()}：训练出来的模型会把很大一部分注意力停在第一个 token 上，叫 attention sink；这个 Lab 里 H2 总是这样，H3 在“它”以外的每个词上也这样。</li><li>${m.name}：${m.heads} 个 head 共用 ${m.kv} 个 KV head（见第 7 部分），head_dim ${m.hd}，来自它的 config.json。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-09-25。原型，尚未经专家审阅。</p></div></div>`);}
/* the primer points to the lab that introduces the model's components (R2.8, CONTRACT §11) instead of teaching them again */
function PRIPOP(){
 return tx(`<p>This lab looks inside one attention block, the component of each layer where a token takes information from itself and earlier tokens. The lab ${LK('architecture')} shows every component a token passes through and gives each its name; ${LK('transformer')} shows where their parameters sit.</p>`,
           `<p>这个 Lab 只看一个 attention 块，也就是每一层里 token 从它自己和前面的 token 取信息的地方。token 经过的每个部件和它们的名字，见${LK('architecture')}这个 Lab；这些部件的参数都在哪，见${LK('transformer')}。</p>`);}
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers */
const OUTCOME=()=>tx('explain why one head cannot give most of its attention to two words, and what splitting into heads changes and what it keeps.','讲清一个 head 为什么不能把大部分注意力同时给两个词，以及拆成多个 head 改变了什么、没改变什么。');
const EXITS=()=>({
 core:()=>chSubmitted, explain:[
  {q:tx('At “it”, one head cannot give 80% or more to both “because” and “cat”. Why?','到了“它”，一个 head 没法同时给“因为”和“小猫”各 80% 以上。为什么？'),
   o:[tx('A head’s weights come out of one softmax and add up to 100%, so two words at 80% each would need 160%.','一个 head 的权重来自同一次 softmax，加起来是 100%，两个词各 80% 就要 160%。'),
     tx('The causal mask hides one of the two words from the head, so only the other one can get much weight.','因果 mask 挡住了其中一个词，所以只有另一个词能拿到大的权重。'),
     tx('The two words’ keys are too far apart, so no query can be close to both of them.','两个词的 key 差得太远，没有哪个 query 能同时和两个都接近。')],a:0,
   n:[null,tx('Both words come before “it”, so the mask hides neither of them. The limit is that one softmax shares out 100%.','这两个词都在“它”前面，mask 哪个都不挡。限制在于一次 softmax 只能分出 100%。'),
     tx('Even a query close to both keys only splits the weight between them, about half each: one softmax cannot give both 80%.','就算 query 和两个 key 都接近，也只是把权重在它们之间平分，各一半左右：一次 softmax 给不出两个 80%。')],
   why:tx('Softmax turns a head’s scores into shares that add up to 100%, so one head can put most of its weight on one word, not on two. Two heads can: one on “because”, the other on “cat”.','softmax 把一个 head 的得分换成加起来 100% 的份额，所以一个 head 能把大部分权重给一个词，给不了两个。两个 head 就可以：一个看“因为”，一个看“小猫”。')},
  {q:tx('The block goes from 1 head to 4 heads. What changes?','这个块从 1 个 head 改成 4 个 head。什么变了？'),
   o:[tx('Each head takes 2 of the 8 numbers of every query, key and value and runs its own softmax, while Wq, Wk, Wv and Wo stay as they are.','每个 head 分到 query、key、value 8 维里的 2 维，各做各的 softmax；Wq、Wk、Wv、Wo 保持不变。'),
     tx('Each head gets its own full set of Wq, Wk, Wv and Wo, so the block now holds 4 times as many parameters.','每个 head 有自己完整的一套 Wq、Wk、Wv、Wo，所以块的参数变成原来的 4 倍。'),
     tx('The 4 heads still share one softmax and only split the work between them, so the block runs faster.','4 个 head 还是共用一次 softmax，只是把计算分开做，所以块算得更快。')],a:0,
   n:[null,tx(`The four matrices keep their size and are only cut into slices, one per head: the block keeps its ${NPAR} parameters.`,`四个矩阵的大小不变，只是切成几段分给各个 head：块的参数还是 ${NPAR} 个。`),
     tx('Each head runs its own softmax, and that is what lets the block take from several words at once. Splitting changes what the block can express, not its speed.','每个 head 各做各的 softmax，块才能同时从几个词取信息。拆分改变的是块能表达什么，不是速度。')],
   why:tx(`Splitting only changes which numbers each head uses. The four matrices keep their size, so the block keeps its ${NPAR} parameters however many heads it has.`,`拆分只改变每个 head 用哪几维。四个矩阵的大小不变，所以不管分成几个 head，这个块都是 ${NPAR} 个参数。`)}],
 transfer:[
  {q:tx('Llama‑3.1‑8B cuts each token’s query into 32 heads. For one token in one layer, how many softmaxes does attention compute?','Llama‑3.1‑8B 把每个 token 的 query 切成 32 个 head。一个 token 在一层里，attention 要做几次 softmax？'),
   o:['1','8','32','128'],a:2,
   why:tx('One per head: 32. Each head compares its own slice of the query with the keys and runs its own softmax.','每个 head 一次：32 次。每个 head 拿自己那段 query 和各个 key 比，再各做一次 softmax。')},
  {q:tx('In Llama‑3.1‑8B every 4 query heads share one key and one value (32 heads, 8 KV heads). Can two heads that share them still put their attention on different words?','Llama‑3.1‑8B 里每 4 个 query head 共用一份 key 和 value（32 个 head，8 个 KV head）。共用 key 和 value 的两个 head，注意力还能放在不同的词上吗？'),
   o:[tx('Yes: each head has its own query, so it scores the same keys differently','能：每个 head 有自己的 query，对同样的 key 打出的分不一样'),tx('No: the same keys give the same scores, so the weights are the same','不能：key 一样，得分就一样，权重也就一样'),tx('Only if they also use different values','只有 value 也不一样时才能')],a:0,
   why:tx(`A score is ${A.ui.math('q \\cdot k')}. The keys are shared, but each head’s query is its own, so each head still gets its own scores and its own softmax. That is why heads can share keys and values and lose little. Sharing them is grouped-query attention, which the lab ${A.shell.labLink('gqa-mla')} covers.`,`得分是 ${A.ui.math('q \\cdot k')}。key 是共用的，但每个 head 的 query 是自己的，所以每个 head 仍有自己的得分和 softmax。正因为这样，几个 head 共用 key 和 value，损失也不大。这种做法叫 grouped-query attention，“${A.shell.labLink('gqa-mla')}”这个 Lab 专门讲它。`)}],
 practice:[
  {id:'qkv-einsum',d:tx('Write scaled dot-product and multi-head self-attention with einsum, derive why the scores are divided by '+A.ui.math('\\sqrt{d_{\\text{head}}}')+', then fix four bugs in a buggy version.','用 einsum 写出 scaled dot-product attention 和多头自注意力，推导为什么得分要除以 '+A.ui.math('\\sqrt{d_{\\text{head}}}')+'，再修好一份有 4 个 bug 的实现。')},
  {id:'llm-fundamentals',d:tx(`Questions 1 and 2: why the scores are divided by ${A.ui.math('\\sqrt{d_h}')}, and the time and memory attention takes.`,`第 1、2 题：为什么得分要除以 ${A.ui.math('\\sqrt{d_h}')}，以及 attention 花多少时间、占多少显存。`)}]});
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Multi-head attention','多头注意力'),title:l?L(l.q):tx('Why does attention use many heads?','attention 为什么要分成多个 head？'),
    minutes:l?l.min:15,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: the components of a Transformer','前置知识：Transformer 的部件'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const LEG=()=>[
 ['q','mha-sw-q',tx('current word (its query is used)','当前的词（用它的 query）'),tx('The filled box is the current word: its query is compared with the keys, and the chart shows where its attention goes.','实心的框是当前的词：拿它的 query 去和各个词的 key 比较，图里画的是它的注意力放在哪里。')],
 ['bar','mha-sw-bar',tx('bar: attention weight','竖条：注意力权重'),tx('A bar’s height is the share of that head’s attention on the word above it; each row adds up to 100%.','竖条的高度是这个 head 放在上方那个词上的注意力份额；每一行加起来是 100%。')],
 ['mask','mha-sw-mask',tx('hatched: later word, masked','斜纹：后面的词，被 mask'),tx('Hatched cells are words after the current word. The causal mask gives them no attention weight.','斜纹格是当前词后面的词。因果 mask 不给它们任何注意力权重。')],
 ['need','mha-sw-need',tx('dashed: the two words “it” needs','虚线框：“它”需要的两个词'),tx('At “it”, the next word depends on “because”, the word just before, and “cat”, the word “it” refers to in this lab. They get a dashed outline.','到了“它”，下一个词取决于紧挨着的“因为”和“它”指代的“小猫”（在这个 Lab 里），这两个词带虚线框。')],
 ['off','mha-sw-off',tx('faded row: head off or not shown yet','淡色行：已关掉或还没展示的 head'),tx('A faded row is a head that is turned off, whose output the block drops, or a head this part does not show yet.','淡色的行是已关掉的 head（块不用它的输出），或者这一部分还不展示的 head。')]
];
function stageHTML(){
  const lg=LEG().map(([k,sw,t,tip])=>`<span tabindex="0" data-tipx="${esc(tip)}"${k==='need'?' data-mha-leg="need"':''}><i class="sw ${sw}"></i><span>${t}</span></span>`).join('');
  return `<div class="panel stage mha-stage" id="mha-stage"><div class="stage-head"><div><div class="meter-title" id="mha-st-k"></div><div class="mha-st-title" id="mha-st-title"></div></div><div class="stage-actions"><span class="status" id="mha-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row">${lg}</div>`+
   `<div class="stage-graph graph mha-graph" id="mha-graph"><div class="mha-grid" id="mha-grid"></div><p class="mha-cap" id="mha-cap"></p><div class="mha-mats" id="mha-mats"></div></div>`+
   `<div class="metrics mha-metrics" id="mha-metrics"></div>`+
   `<details class="math" id="mha-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="mha-mathbody"></div></details></div>`;
}
function controlsHTML(){
  return `<aside class="panel controls" id="mha-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div><p class="ctl-lock" id="mha-lock" hidden></p>`+
   `<div class="ctl lockable" id="mha-ctl-q"><span class="lbl">${esc(QLABEL())}</span><div class="mha-words" role="group" aria-label="${esc(QLABEL())}">${SENT[LG()].map((w,j)=>`<button type="button" data-q="${j}" aria-pressed="${j===S.q}">${esc(w)}</button>`).join('')}</div>`+
   `<div class="ctl-note">${tx('The chart shows this word’s attention. It can look at itself and every word before it.','图里显示的是这个词的注意力。它能看它自己和它前面的每个词。')}</div></div>`+
   `<div class="lockable" id="mha-ctl-H-wrap">${headSeg('mha-ctl-H',false,tx('The block splits the 8 numbers of every query, key and value into this many heads, 8, 4 or 2 numbers each; each of those numbers is still computed from all 8 of the token’s numbers. Wq, Wk, Wv and Wo stay the same.','这个块把每个 query、key、value 的 8 维分给这么多个 head，每个分到 8、4 或 2 维；这些维仍由 token 的全部 8 维算出。Wq、Wk、Wv、Wo 都不变。'))}</div>`+
   `<p class="ctl-note lockable" id="mha-H-note" hidden></p>`+
   `<div class="ctl lockable" id="mha-ctl-on">${onCtlInner()}<div class="ctl-note">${tx('Turn off any heads you like, several at once, and see which words the block still takes from. A head that is off adds nothing to the block’s output, as when a head is pruned from a trained model.','想关哪个 head 就关哪个，可以同时关几个，看看这个块还能从哪些词取信息。关掉的 head 不往这个块的输出里加任何东西，相当于把训练好的模型里的一个 head 剪掉。')}</div></div>`+
   `<p class="ctl-note lockable" id="mha-on-note" hidden></p>`+
   `<details class="adv lockable" id="mha-adv"><summary>${tx('Hand-set matrices: what they are','手工设的矩阵是什么')}</summary><div class="ctl-note" style="margin-top:8px">${handNote()}</div></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
/* R2.7 in short, and R6.16/R6.18 (N7): the real block's size as 41.9M with the kit's exact-value hover, its shapes written out */
function handNote(){const m=rm(), att=2*m.d*m.d+2*m.d*m.kv*m.hd, r4=compute({lang:LG(),q:IT,H:4}), w1=pc(r4.heads[0].w[BECAUSE]), w3=pc(r4.heads[HC].w[r4.cat_i]);
  const big=A.ui&&A.ui.big?A.ui.big(att):fi(att), D_=fi(m.d), K_=fi(m.kv*m.hd);
  return tx(`<p>The 256 parameters in Wq, Wk, Wv and Wo are set by hand, so you can check what each head does: Wq and Wk are built from four patterns (the previous word, the first token, the word “it” refers to, the nouns), each in 2 of the 8 numbers.</p><p>In a trained model the parameters come from training on text, and its heads rarely do one clean job. Their size decides how sharp a head is: here they are set so that at “it” H1 and H3 each put more than ${pc(CH_GOAL)} on their word (${w1} and ${w3}).</p><p>The real block is far bigger: ${m.name}’s attention block has ${big} parameters per layer (Wq and Wo ${D_} × ${D_} each, Wk and Wv ${D_} × ${K_} each).</p>`,
            `<p>Wq、Wk、Wv、Wo 的 256 个参数是手工设的，方便你检查每个 head 在做什么：Wq 和 Wk 由四种模式拼成（前一个词、第一个 token、“它”指代的词、名词），每种占 8 维里的 2 维。</p><p>训练出来的模型里，参数来自在文本上的训练，head 很少只干一件干净的事。参数的大小决定 head 看得多集中：这里把它们设成在“它”这个位置，H1 和 H3 各在自己的词上放 ${pc(CH_GOAL)} 以上（${w1} 和 ${w3}）。</p><p>真实的块大得多：${m.name} 的 attention 块每层有 ${big} 个参数（Wq 和 Wo 各 ${D_} × ${D_}，Wk 和 Wv 各 ${D_} × ${K_}）。</p>`);}
const onList=()=>{const on=S.on.map((v,h)=>v?`H${h+1}`:null).filter(Boolean); return on.length?tx(`On: ${on.join(', ')}`,`开着：${on.join('、')}`):tx('On: none','开着：没有');};
function onCtlInner(){return `<span class="lbl">${tx('Heads that are on','开着的 head')}</span><div class="mha-onbtns" role="group" aria-label="${esc(tx('Heads that are on; several can be on','开着的 head，可以同时开几个'))}">${[0,1,2,3].map(h=>`<button type="button" data-on="${h}" aria-pressed="${!!S.on[h]}">H${h+1}</button>`).join('')}</div><div class="mha-onlist" aria-live="polite">${onList()}</div>`;}
const CODE=`import torch

def mha(x, Wq, Wk, Wv, Wo, n_heads):
    # x: (T, d), one row per token
    # Wq, Wk, Wv, Wo: (d, d)
    T, d = x.shape
    hd = d // n_heads                     # head_dim
    def split(m):                         # (heads, T, hd)
        return m.view(T, n_heads, hd).transpose(0, 1)
    q, k, v = split(x @ Wq.T), split(x @ Wk.T), split(x @ Wv.T)
    scores = q @ k.transpose(1, 2) / hd ** 0.5
    # causal mask: no attention weight on later tokens
    later = torch.triu(torch.ones(T, T), 1).bool()
    scores = scores.masked_fill(later, float("-inf"))
    w = scores.softmax(dim=-1)            # rows add up to 1
    out = (w @ v).transpose(0, 1).reshape(T, d)
    return out @ Wo.T, w`;
function connectHTML(){
  const HF='https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py', VL='https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py';
  const hf=`<a href="${HF}" target="_blank" rel="noopener"><code>transformers/models/llama/modeling_llama.py</code></a>`, vl=`<a href="${VL}" target="_blank" rel="noopener"><code>vllm/model_executor/layers/linear.py</code></a>`;
  return `<section class="wrap connect" aria-labelledby="mha-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="mha-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The whole layer in PyTorch: the same three matrix products for any number of heads; view() only cuts each row of numbers into heads.','整层 attention 的 PyTorch 写法：不管几个 head，都是同样的三次矩阵乘法；view() 只是把每一行切成几个 head。')}</p><pre class="code" id="mha-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="mha-prod">`+
   tx(`<li>${hf}, <code>LlamaAttention</code>: projects the token’s numbers with <code>q_proj</code>, <code>k_proj</code> and <code>v_proj</code> and views them as heads of <code>head_dim</code> numbers.</li><li>The same file’s <code>eager_attention_forward</code>: <code>repeat_kv</code> copies each KV head to every head of its group (grouped-query attention, GQA); then ${A.ui.math('q k^{\\text{T}}')} times ${A.ui.math('\\text{head_dim}^{-0.5}')}, the causal mask added to the scores, a softmax, and the values summed with the attention weights.</li><li>vLLM, ${vl}, <code>QKVParallelLinear</code>: q, k and v from one matrix, with the heads split across GPUs under tensor parallelism.</li>`,
      `<li>${hf}，<code>LlamaAttention</code>：用 <code>q_proj</code>、<code>k_proj</code>、<code>v_proj</code> 把 token 的向量投影出来，再按每段 <code>head_dim</code> 维切成几个 head。</li><li>同一个文件里的 <code>eager_attention_forward</code>：<code>repeat_kv</code> 把每个 KV head 复制给它那一组里的每个 head（grouped-query attention，GQA）；然后算 ${A.ui.math('q k^{\\text{T}}')} 乘以 ${A.ui.math('\\text{head_dim}^{-0.5}')}，把因果 mask 加到得分上，做 softmax，再按注意力权重把 value 加起来。</li><li>vLLM，${vl}，<code>QKVParallelLinear</code>：用一个矩阵同时算出 q、k、v，张量并行时把 head 分到几张 GPU 上。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener">Attention Is All You Need</a><span>${tx('Vaswani et al., 2017, §3.2.2: multi-head attention','Vaswani 等，2017，§3.2.2：多头注意力')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/1905.10650" target="_blank" rel="noopener">Are Sixteen Heads Really Better than One?</a><span>${tx('Michel et al., NeurIPS 2019: removing heads from trained models','Michel 等，NeurIPS 2019：从训练好的模型里去掉 head')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2209.11895" target="_blank" rel="noopener">In-context Learning and Induction Heads</a><span>${tx('Olsson et al., 2022: previous-token heads and induction heads','Olsson 等，2022：previous-token head 和 induction head')}</span></li></ul>`+
   `<p class="small muted" style="margin-top:12px">${tx('See also: <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a> (Cho et al., 2024) runs GPT-2 small in the browser and shows the attention of its 12 heads per layer. <a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a> (Brendan Bycroft) walks through every step of a tiny GPT, nano-gpt, in 3D.','另见：<a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a>（Cho 等，2024）在浏览器里跑 GPT-2 small，能看它每层 12 个 head 的注意力。<a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a>（Brendan Bycroft）用 3D 一步步演示一个很小的 GPT（nano-gpt）怎么计算。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="mha-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="mha-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="mha-app m2" id="mha-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="mha-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- rail ---------- */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3><p class="small">${tx('Pick a scenario and see which word each head puts its attention on.','选一个场景，看看每个 head 把注意力放在哪个词上。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(p.t())}</b><span>${esc(p.d())}</span></button>`).join('')}</div>`+
   `<div class="obs" id="mha-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${TOCHAL()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, you see whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  const b=esc(wd(BECAUSE)), c=esc(wd(CAT[LG()]));
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Turn off the most heads while “${b}” and “${c}” each keep 80% in some head`,`关掉尽量多的 head，让“${b}”和“${c}”各在某个 head 里仍有 80%`)}</h3>`+
   `<p>${tx(`The block has 4 heads. At “it”, some head that is still on must put at least 80% on “${b}”, and some head at least 80% on “${c}”. Turn off as many heads as you can:`,
            `这个块有 4 个 head。当前的词是“它”时，开着的 head 里要有一个在“${b}”上至少 80%，也要有一个在“${c}”上至少 80%。尽量多关掉几个 head：`)}</p>`+
   `<div class="inline-ctl" id="mha-ic-on">${onCtlInner()}</div>`+
   `<div class="goal mono mha-goal" id="mha-ch-b"><span>${tx(`Highest attention weight on “${b}” among heads still on`,`开着的 head 给“${b}”的最高注意力权重`)}</span><span><b></b>${tx(' (goal: at least 80%)','（目标至少 80%）')}</span></div><div class="progress" id="mha-ch-bp" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono mha-goal" id="mha-ch-c"><span>${tx(`Highest attention weight on “${c}” among heads still on`,`开着的 head 给“${c}”的最高注意力权重`)}</span><span><b></b>${tx(' (goal: at least 80%)','（目标至少 80%）')}</span></div><div class="progress" id="mha-ch-cp" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="mha-ch-o"><span>${tx('Heads turned off','关掉的 head')}</span><span><b></b>${tx(' of 4',' / 4')}</span></div>`+
   `<div style="margin-top:12px">${submit}</div><div id="mha-ch-res" aria-live="polite"></div></div>`;
}
const onNames=on=>on.map((v,h)=>v?`H${h+1}`:null).filter(Boolean);
const chBest=()=>{const lg=LG(), me=chPoint(lg,S.on); return me.ok&&me.off===Math.max(...combos().map(on=>chPoint(lg,on)).filter(p=>p.ok).map(p=>p.off));};
function chResultHTML(){
  if(!chSubmitted) return '';
  const lg=LG(), me=chPoint(lg,S.on), all=combos().map(on=>chPoint(lg,on)), met=all.filter(p=>p.ok), best=met.sort((a,b)=>b.off-a.off)[0];
  const r4=compute({lang:lg,q:IT,H:4}), ci=r4.cat_i, b=`“${esc(wd(BECAUSE))}”`, c=`“${esc(wd(ci))}”`, names=a=>listJoin(a);
  let h;
  const HCn=`H${HC+1}`, g=pc(CH_GOAL);
  /* the result as the goal lines show it; the reason in words (R6.20): the chart and the goal lines hold the numbers */
  if(me.ok&&me.off===best.off) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: with only ${names(onNames(me.on))} on, “it” still puts ${pc(me.bb)} on ${b} and ${pc(me.bc)} on ${c}.</b> And no more heads can go: ${b} needs H1 and ${c} needs ${HCn}.</p>`,
                                   `<p class="okline" style="margin-top:12px"><b>达成目标：只开着 ${names(onNames(me.on))}，“它”在${b}上仍有 ${pc(me.bb)}，在${c}上 ${pc(me.bc)}。</b>而且不能再关了：${b}离不开 H1，${c}离不开 ${HCn}。</p>`);
  else if(me.ok){const extra=onNames(me.on).filter(n=>n!=='H1'&&n!==HCn), one=extra.length===1, looks=extra.map(n=>{const o=r4.heads[+n[1]-1]; return one?tx(`it ${lookTxt(look(o))}`,`它${lookTxt(look(o))}`):tx(`${n} ${lookTxt(look(o))}`,`${n}${lookTxt(look(o))}`);});
    h=tx(`<p class="okline" style="margin-top:12px"><b>Both still at 80% or more: ${pc(me.bb)} on ${b} and ${pc(me.bc)} on ${c}.</b> But you could turn off more: H1 alone covers ${b} and ${HCn} alone covers ${c}. ${names(extra)} ${extra.length>1?'are':'is'} not needed: ${listJoin(looks)}.</p>`,
         `<p class="okline" style="margin-top:12px"><b>两个词仍都在 80% 以上：${b}上 ${pc(me.bb)}，${c}上 ${pc(me.bc)}。</b>不过还能再关：${b}靠 H1、${c}靠 ${HCn} 就够了。${names(extra)}${extra.length>1?'都':''}用不上：${looks.join('，')}。</p>`);}
  else {const miss=[!me.okB?tx(`at most ${pc(me.bb)} on ${b}`,`在${b}上最多 ${pc(me.bb)}`):null,!me.okC?tx(`at most ${pc(me.bc)} on ${c}`,`在${c}上最多 ${pc(me.bc)}`):null].filter(Boolean);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: the heads still on put ${miss.join(' and ')}.</b></p><p class="small">Hint: look at where each head puts its attention weight at “it”, ${J('grid','in the chart')}.</p>`,
         `<p class="badline" style="margin-top:12px"><b>还没达到：开着的 head ${miss.join('，')}。</b></p><p class="small">提示：看看${J('grid','图里')}每个 head 在“它”这个位置把注意力权重放在哪里。</p>`)
      +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx(`Compare ${chRows(me).length} ways to turn heads off`,`看 ${chRows(me).length} 种开关组合的对比`)}</button>`);}
  if(!chReveal) return h;
  const init=chPoint(lg,ALL_ON()), o1=chPoint(lg,[false,true,true,true]), o3=chPoint(lg,[true,true,false,true]), o2=chPoint(lg,[true,false,true,true]), o4=chPoint(lg,[true,true,true,false]);
  /* baseline → effect → reason (R3.10) in words, each phrase checked against the engine; the numbers are on the chart above (R6.20) */
  const same=p=>pc(p.bb)===pc(init.bb)&&pc(p.bc)===pc(init.bc), far=v=>v<0.4;
  const nEn=same(o2)&&same(o4)?'Turning off H2 or H4 changes neither.':'Turning off H2 or H4 changes them only a little.', nZh=same(o2)&&same(o4)?'关掉 H2 或 H4，两个词都不变。':'关掉 H2 或 H4，两个词只变了一点。';
  h+=chChart(me);
  h+=tx(`<p class="small">Initial setup, all 4 heads on: both words clear the ${g} line. ${nEn} Turning off H1 drops ${b} ${far(o1.bb)?'far below':'below'} that line, and turning off H3 drops ${c} ${far(o3.bc)?'far below':'below'} it too.</p><p class="small">Why: each head has its own softmax, and only H1 and H3 put most of their attention weight on these two words; H2 looks at &lt;s&gt; and H4 spreads over both nouns.</p>`,
        `<p class="small">初始设置是 4 个 head 全开：两个词都过了 ${g} 这条线。${nZh}关掉 H1，${b}${far(o1.bb)?'远远掉到':'掉到'}这条线以下；关掉 H3，${c}也${far(o3.bc)?'远远掉到':'掉到'}线以下。</p><p class="small">原因：每个 head 各做各的 softmax，只有 H1 和 H3 把大部分注意力权重放在这两个词上；H2 看 &lt;s&gt;，H4 分散在两个名词上。</p>`);
  h+=tx(`<p class="small muted">Note: this checks one word in one sentence. A head that looks idle at “it” may matter for other words, which is why ${MICHEL_P()} prune heads while checking accuracy on a whole task.</p>`,
        `<p class="small muted">注意：这里只看了一句话里的一个词。在“它”这里看起来没用的 head，对别的词可能有用，所以 ${MICHEL_P()}剪 head 时，要在整个任务上检查准确率。</p>`);
  return h;
}
/* the challenge chart: the weakest of two bars decides; rows are the choices that teach something, plus the learner's pick */
/* the choices the chart compares: the six that teach something, plus the learner's pick */
function chRows(me){const rows=[[true,true,true,true],[true,false,true,true],[true,true,true,false],[true,false,true,false],[false,true,true,true],[true,true,false,true]];
  if(!rows.some(on=>onKey(on)===onKey(me.on))) rows.push(me.on); return rows;}
function chChart(me){
  const lg=LG(), ci=CAT[lg], rows=chRows(me);
  /* the value label goes where the kit puts it (cmp-cq, cmp-fit, --v: right of the bar's end, or inside it, clear of the 80% line; kit
     polish K1), as in A.ui.cmpChart; it used to sit right of the bar on its own background over the line */
  const bar=(lab,v)=>`<div class="mha-cb"><span>${lab}</span><div class="cmp-track cmp-cq"><i style="width:${(v*100).toFixed(2)}%"></i><b class="g" style="left:80%"></b><span class="cmp-val mono cmp-fit" style="left:${(v*100).toFixed(2)}%;--v:${(v*100).toFixed(2)}">${pc(v)}</span></div></div>`;
  const body=rows.map(on=>{const p=chPoint(lg,on), isMe=onKey(on)===onKey(me.on), init=on.every(Boolean);
    const lab=tx(`${onNames(on).join(', ')||'none'} on (${p.off} off)`,`开着 ${onNames(on).join('、')||'无'}（关掉 ${p.off} 个）`);
    const tags=(init?`<span class="cmp-tag">${tx('initial setup','初始设置')}</span>`:'')+(isMe?`<span class="cmp-you">${tx('your pick','你的选择')}</span>`:'');
    return `<div class="mha-cr${p.ok?' ok':''}${isMe?' me':''}"><div class="cmp-lab"><span>${lab}</span>${tags}</div>${bar(esc(wd(BECAUSE)),p.bb)}${bar(esc(wd(ci)),p.bc)}</div>`;}).join('');
  const title=tx(`The largest attention weight any head still on puts on “${esc(wd(BECAUSE))}” and on “${esc(wd(ci))}” at “it”, for ${rows.length} choices of heads (goal: 80% each)`,`在“它”这个位置，${rows.length} 种开关组合下，开着的 head 给“${esc(wd(BECAUSE))}”和“${esc(wd(ci))}”的最高注意力权重（目标：各 80%）`);
  const aria=title+'. '+rows.map(on=>{const p=chPoint(lg,on); return `${onNames(on).join(', ')||'none'}: ${pc(p.bb)}, ${pc(p.bc)}`;}).join('; ');
  return `<div class="cmp mha-cmp" role="img" aria-label="${esc(aria)}"><div class="cmp-title">${title}</div><div aria-hidden="true">${body}<div class="cmp-axis"><span style="left:80%">${tx('goal: 80%','目标 80%')}</span></div></div>`+
    `<div class="cmp-leg" aria-hidden="true"><span><i class="ok"></i>${tx('meets both','两个都达标')}</span><span><i></i>${tx('misses at least one','至少一个不达标')}</span><span><i class="g"></i>${tx('goal: 80%','目标：80%')}</span></div></div>`;
}

/* ---------- stage drawing ---------- */
const masked=()=>mode==='guided'&&step<=1;
const needOn=()=>S.q===IT;
/* the badge: t on the stage, short (two or three words) in the mini result bar (R3.2) */
function status(r){
  if(masked()) return {cls:'neutral',t:pred?tx('Shown from part 3 on','从第 3 部分起显示'):tx('Predict first','先预测'),short:pred?tx('From part 3','第 3 部分起'):tx('Predict first','先预测'),
    tip:tx('The heads’ attention weights stay hidden in parts 1 and 2, so they do not give the answer away.','第 1、2 部分先隐藏每个 head 的注意力权重，免得提前看到答案。')};
  const b=`“${wd(BECAUSE)}”`, c=`“${wd(r.cat_i)}”`, g=pc(CH_GOAL);
  if(mode==='challenge'){const ok=r.bec.v>=CH_GOAL&&r.catw.v>=CH_GOAL, both=r.bec.v<CH_GOAL&&r.catw.v<CH_GOAL, w=r.bec.v<CH_GOAL?b:c;
    return {cls:ok?'ok':'bad',t:ok?tx(`Both words at ${g} or more`,`两个词都在 ${g} 以上`):(both?tx(`Both words under ${g}`,`两个词都不到 ${g}`):tx(`${w} under ${g}`,`${w}不到 ${g}`)),
      /* N6: before the answer is submitted the mini bar states the constraint, not “Goal met” (the goal is also to turn heads off) */
      short:ok?(chSubmitted&&chBest()?tx('Goal met','达成目标'):tx(`Both ≥ ${g}`,`两个都 ≥ ${g}`)):(both?tx(`Both under ${g}`,`都不到 ${g}`):tx(`${w} under ${g}`,`${w}不到 ${g}`)),
      tip:tx(`Green when some head that is on puts at least ${g} on ${b} and some head at least ${g} on ${c}.`,`开着的 head 里有一个在${b}上至少 ${g}，也有一个在${c}上至少 ${g} 时，徽章是绿色。`)};}
  if(r.p===IT&&focusHead()<0){const kb=r.bec.v>=CH_GOAL, kc=r.catw.v>=CH_GOAL;
    return {cls:kb&&kc?'ok':'bad',t:kb&&kc?tx(`Both words at ${g} or more`,`两个词都在 ${g} 以上`):(kb||kc?tx(`Only ${kb?b:c} at ${g} or more`,`只有${kb?b:c}在 ${g} 以上`):tx(`Neither word at ${g} or more`,`两个词都不到 ${g}`)),
      short:kb&&kc?tx(`Both ≥ ${g}`,`两个都 ≥ ${g}`):(kb||kc?tx(`Only ${kb?b:c}`,`只有${kb?b:c}`):tx(`Neither ≥ ${g}`,`都不到 ${g}`)),
      tip:tx(`At “it”, the next word depends on ${b}, the word just before, and ${c}, the word “it” refers to in this lab. Green when some head that is on puts ${g} or more on each of them; ${g} is this lab’s line for “mostly on one word”.`,`到了“它”，下一个词取决于紧挨着的${b}和“它”指代的${c}（在这个 Lab 里）。开着的 head 里，两个词各有一个 head 放了 ${g} 以上时，徽章是绿色；${g} 是这个 Lab 定的线，表示“大部分放在这个词上”。`)};}
  const n=r.p+1;
  return {cls:'neutral',t:tx(`${wt(r.p)} sees ${n} token${n===1?'':'s'}`,`${wt(r.p)}能看到 ${n} 个 token`),short:tx(`Sees ${n} token${n===1?'':'s'}`,`能看到 ${n} 个 token`),tip:tx('The causal mask lets a word look only at itself and the words before it.','因果 mask 让一个词只能看它自己和它前面的词。')};
}
function drawHead(r){
  $('#mha-st-k',app).textContent=tx('One attention block, hidden size d = 8; the parameters in Wq, Wk, Wv and Wo are set by hand','一个 attention 块，隐藏维度 d = 8，Wq、Wk、Wv、Wo 的参数是手工设的');
  $('#mha-st-title',app).innerHTML=tx(`Where ${wq(r.p)} looks: ${splitTitle(r.H)}`,`${wq(r.p)}看向哪里：${splitTitle(r.H)}`);
  const st=status(r), el=$('#mha-status',app); el.className='status '+st.cls; el.textContent=st.t; el.dataset.tipx=st.tip;
  const ln=$('[data-mha-leg="need"]',app); if(ln) ln.hidden=!needOn();
}
const colsOf=lg=>SENT[lg].map(w=>{let n=0; for(const ch of w) n+=ch.codePointAt(0)>=0x2e80?2:1; return `minmax(0,${Math.max(3,n)}fr)`;}).join(' ');
function gridHTML(r,o){   // o: {tile, heads (indices to draw)} for the home tile
  o=o||{}; const lg=r.lang, H=r.H, p=r.p, mk=!o.tile&&masked(), foc=o.tile?-1:focusHead();
  const toks=SENT[lg].map((w,j)=>{const cls=['mha-tok',j===p?'q':'',j>p?'later':'',(needOn()||o.tile)&&(j===BECAUSE||j===r.cat_i)&&p===IT?'need':''].filter(Boolean).join(' ');
    const tip=j===p?tx(`${wt(j)} is the current word: the rows below show where its attention goes.`,`${wt(j)}是当前的词：下面每一行是它的注意力放在哪里。`)
      :(j===0?TIP.s0():(j>p?tx(`${wt(j)} comes after ${wt(p)}, so the causal mask hides it from ${wt(p)}.`,`${wt(j)}在${wt(p)}后面，因果 mask 让${wt(p)}看不到它。`):tx(`${wt(j)} comes before ${wt(p)}, so ${wt(p)} can look at it.`,`${wt(j)}在${wt(p)}前面，${wt(p)}能看它。`)));
    if(o.tile) return `<span class="${cls}">${esc(w)}</span>`;
    const act=j===p?'':(mode==='challenge'?tx(' In the challenge the current word stays “it”.','挑战中当前的词固定是“它”。'):(qLock(j)?tx(' “it” comes in part 4.','“它”留到第 4 部分。'):tx(` Click to make ${wt(j)} the current word.`,`点击把${wt(j)}设为当前的词。`)));
    return `<button type="button" class="${cls}${qLock(j)?' mha-later':''}" data-q="${j}" aria-pressed="${j===p}"${qLock(j)?' aria-disabled="true"':''} data-tipx="${esc(tip+act)}">${esc(w)}</button>`;}).join('');
  const hs=o.heads||r.heads.map((_,h)=>h);
  const rows=hs.map(h=>{const hd=r.heads[h], hide=mk||(foc>=0&&h!==foc), off=!hd.on, ch=o.tile||hide?null:chainOf(r,h);   // N8: hovers show Show the math's numbers
    const cells=SENT[lg].map((w,j)=>{
      if(j>p) return `<div class="mha-cell masked"${o.tile||hide?'':` data-tipx="${esc(tx(`${wt(j)} comes after ${wt(p)}: the causal mask gives it no attention weight.`,`${wt(j)}在${wt(p)}后面：因果 mask 不给它注意力权重。`))}"`}><span class="mha-pv"></span><div class="mha-bz"></div></div>`;
      if(hide) return `<div class="mha-cell"><span class="mha-pv"></span><div class="mha-bz"></div></div>`;
      const v=hd.w[j], t=rd(v*100,0)>=1?pc(v):'';
      const cw=ch?ch.rows.find(x=>x.j===j).w:v, tip=o.tile?'':` data-tipx="${esc(tx(`${Hn(H,h)} puts ${pc1(cw)} of ${wt(p)}’s attention on ${wt(j)}.`,`${Hn(H,h)}把${wt(p)}的注意力的 ${pc1(cw)} 放在${wt(j)}上。`))}"`;
      return `<div class="mha-cell"${tip}><span class="mha-pv">${t}</span><div class="mha-bz"><i class="mha-bar" style="height:${(v*100).toFixed(2)}%"></i></div></div>`;}).join('');
    const job=H===4?tx(` Its job, ${L(JOB_G[h])}, was set by hand in this lab.`,`它的分工（${L(JOB[h])}）是在这个 Lab 里手工设的。`):'';
    const sum=hide?(mk?tx(`${Hn(H,h)}’s attention weights are shown from part 3 on, after you predict.`,`预测之后，第 3 部分起显示 ${Hn(H,h)} 的注意力权重。`):tx(`${Hn(H,h)} is one of the block’s other heads.`,`${Hn(H,h)}是这个块的另外几个 head 之一。`))
      :tx(`${Hn(H,h)} at ${wt(p)}: `,`${Hn(H,h)}，当前的词是${wt(p)}：`)+tops(hd,0.01).map(x=>tx(`${pc(x.w)} on ${wt(x.j)}`,`${wt(x.j)} ${pc(x.w)}`)).join(tx(', ','，'))+tx('.','。')+job+(off?tx(' It is turned off: it still computes these attention weights, but the block drops its output.','它已关掉：照样算出这些注意力权重，但块不用它的输出。'):'');
    const lab=rowLabel(H,h)+(off?tx(' (off)','（已关掉）'):'');
    return `<div class="mha-row${off?' off':''}${hide?' dim':''}"${o.tile?'':` tabindex="0" data-tipx="${esc(sum)}"`}><div class="mha-rlab">${lab}</div><div class="mha-cells">${cells}</div></div>`;}).join('');
  return `<div class="mha-toks">${toks}</div>${rows}`;
}
function focusHead(){return mode==='guided'&&STEPS[step].focus&&focus>=0&&S.H===4?focus:-1;}
function drawGrid(r){
  const g=$('#mha-grid',app); g.style.setProperty('--mha-cols',colsOf(r.lang)); g.innerHTML=gridHTML(r);
  $('#mha-cap',app).textContent=masked()?tx('Each row will show one head’s attention weights over the words, from part 3 on.','从第 3 部分起，每一行会显示一个 head 在各个词上的注意力权重。')
    :tx('Each row is one head; its bars add up to 100%: the share of that head’s attention on each word.','每一行是一个 head，它的各条加起来是 100%，表示这个 head 把多少注意力放在上方的每个词上。');
}
function matSVG(W,cols,H){
  const c=7, n=8, s=n*c, mx=Math.max(...W.flat().map(Math.abs)); let h='';
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){const v=W[i][j], x=j*c+0.5, y=i*c+0.5, w=c-1.5;
    if(Math.abs(v)<1e-9) h+=`<rect class="mz" x="${x}" y="${y}" width="${w}" height="${w}"/>`;
    else if(v>0) h+=`<rect class="mc" x="${x}" y="${y}" width="${w}" height="${w}" fill-opacity="${(0.28+0.72*v/mx).toFixed(2)}"/>`;
    else h+=`<rect class="mn" x="${x+0.5}" y="${y+0.5}" width="${w-1}" height="${w-1}" stroke-opacity="${(0.4+0.6*(-v)/mx).toFixed(2)}"/>`;}
  const hd=8/H; for(let k=1;k<H;k++){const t=k*hd*c; h+=cols?`<line class="ms" x1="${t}" x2="${t}" y1="-2" y2="${s+2}"/>`:`<line class="ms" x1="-2" x2="${s+2}" y1="${t}" y2="${t}"/>`;}
  return `<svg width="${s+4}" height="${s+4}" viewBox="-2 -2 ${s+4} ${s+4}" aria-hidden="true"><rect class="mf" x="0" y="0" width="${s}" height="${s}"/>${h}</svg>`;
}
function drawMats(r){
  const H=r.H, hd=8/H, rows=[...Array(H)].map((_,h)=>[`${h*hd+1}–${(h+1)*hd}`,hn(H,h)]);
  const grp=tx(listJoin(rows.map(([a,n],i)=>`${i?'':'rows '}${a} to ${n}`)),rows.map(([a,n])=>`第 ${a} 行给 ${n}`).join('，'));
  const tips=H===1?{Wq:tx('Wq turns a word’s 8 numbers into its query: each row makes one number. With one head, all 8 rows go to that head.','Wq 把一个词的 8 维向量变成它的 query：每一行算出一维。只有一个 head 时，8 行全给这个 head。'),
    Wk:tx('Wk turns a word’s 8 numbers into its key the same way; with one head, all 8 rows go to that head.','Wk 用同样的方式把一个词的 8 维向量变成它的 key；只有一个 head 时，8 行全给这个 head。'),
    Wv:tx('Wv turns a word’s 8 numbers into its value. In this lab the value marks only “because” and “cat”, so the head’s output shows how much of each arrived.','Wv 把一个词的 8 维向量变成它的 value。在这个 Lab 里，value 只标记“因为”和“小猫”，所以 head 的输出直接显示两者各到了多少。'),
    Wo:tx('Wo turns the head’s output back into 8 numbers. In this lab it is the identity: 1 on the diagonal, 0 elsewhere.','Wo 把 head 的输出变回 8 维。在这个 Lab 里它是单位矩阵：对角线上是 1，其余是 0。')}
   :{Wq:tx(`Wq turns a word’s 8 numbers into its query: each row makes one number. The lines split the rows into heads: ${grp}.`,`Wq 把一个词的 8 维向量变成它的 query：每一行算出一维。横线把行分给各个 head：${grp}。`),
    Wk:tx(`Wk turns a word’s 8 numbers into its key, split into heads the same way: ${grp}.`,`Wk 把一个词的 8 维向量变成它的 key，按同样的方式分给各个 head：${grp}。`),
    Wv:tx(`Wv turns a word’s 8 numbers into its value, split the same way. In this lab each head’s two value numbers mark “because” and “cat”.`,`Wv 把一个词的 8 维向量变成它的 value，也按同样的方式分。在这个 Lab 里，每个 head 的两维 value 分别标记“因为”和“小猫”。`),
    Wo:tx('Wo turns the heads’ outputs, placed side by side, back into 8 numbers; the lines mark which columns take which head’s output. In this lab it is the identity: 1 on the diagonal, 0 elsewhere.','Wo 把各 head 并排放好的输出变回 8 维；竖线标出哪几列接哪个 head 的输出。在这个 Lab 里它是单位矩阵：对角线上是 1，其余是 0。')};
  const el=$('#mha-mats',app);
  el.innerHTML=`<div class="mha-mats-row">${['Wq','Wk','Wv','Wo'].map(k=>`<div class="mha-mat" tabindex="0" data-tipx="${esc(tips[k])}">${matSVG(M[k],k==='Wo',H)}<span>${k}</span></div>`).join('')}</div>`+
    `<div class="mha-mleg"><span tabindex="0" data-tipx="${esc(tx('Blue cells are the hand-set parameters of each matrix; the stronger the blue, the larger the parameter, each matrix on its own scale.','蓝色格子是每个矩阵里手工设的参数，蓝色越浓参数越大，每个矩阵按自己的最大值换算。'))}"><i class="k1"></i>${tx('parameter (stronger blue = larger)','参数（蓝色越浓越大）')}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('An outlined cell holds a negative parameter.','空心的格子是负的参数。'))}"><i class="k2"></i>${tx('negative','负值')}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('A thick line marks where the rows that make q, k and v split into heads. Changing the number of heads moves only these lines; the matrices stay the same.','粗线标出算出 q、k、v 的那些行从哪里分给不同的 head。改 head 数只会移动这些线，矩阵本身不变。'))}"><i class="k3"></i>${tx('split between heads','head 之间的分界')}</span></div>`+
    `<p class="mha-cap">${tx('The block’s four matrices, 8 × 8 each: 256 hand-set parameters.','这个块的四个矩阵，各 8 × 8，一共 256 个手工设的参数。')}</p>`;
}
/* ▲▼ only after the learner's own change, in points, from the values as shown (R6.6) */
const MET=[
 {k:'H',label:()=>tx('Heads in the block','这个块的 head 数'),v:r=>String(r.H),sub:r=>r.H===1?tx('with all 8 numbers of q, k and v','用 q、k、v 的全部 8 维'):tx(`each gets ${r.hd} of the 8 numbers of q, k and v`,`每个分到 q、k、v 的 ${r.hd} 维`),
  tip:()=>tx('The block splits the 8 numbers of every query, key and value into this many heads. Each head gets head_dim of them and runs its own softmax. Each of those numbers is still computed from all 8 of the token’s numbers.<div class="fx">'+A.ui.math('\\text{head_dim} = 8 \\div \\text{heads}')+'</div>','这个块把每个 query、key、value 的 8 维分给这么多个 head。每个 head 分到的维数就是它的 head_dim，各做各的 softmax。这些维仍由 token 的全部 8 维算出。<div class="fx">'+A.ui.math('\\text{head_dim} = 8 \\div \\text{head 数}')+'</div>')},
 {k:'prev',pts:true,label:()=>tx('Attention weight on the previous word','前一个词上的<span class="mha-nw">注意力权重</span>'),v:r=>r.p===0?'—':pc(r.prev.v),
  sub:r=>r.p===0?tx('<s> has no word before it','<s> 前面没有词'):(r.prev.h<0?tx(`on ${wt(r.p-1)}; no head is on`,`在${wt(r.p-1)}上；没有开着的 head`):tx(`on ${wt(r.p-1)}, from ${hn(r.H,r.prev.h)}`,`在${wt(r.p-1)}上，来自${hn(r.H,r.prev.h)}`)),
  tip:()=>tx('The largest share of attention that any head that is on puts on the word just before the current word.','开着的 head 里，放在当前词的前一个词上的最大注意力份额。')},
 {k:'cat',pts:true,label:()=>tx(`Attention weight on “${wd(CAT[LG()])}”`,`“${wd(CAT[LG()])}”上的<span class="mha-nw">注意力权重</span>`),v:r=>r.cat_i>r.p?pc(0):pc(r.catw.v),
  sub:r=>r.cat_i>r.p?tx(`“${wd(r.cat_i)}” comes later: masked`,`“${wd(r.cat_i)}”在后面，被 mask`):(r.catw.h<0?tx('no head is on','没有开着的 head'):tx(`from ${hn(r.H,r.catw.h)}`,`来自${hn(r.H,r.catw.h)}`)),
  tip:()=>tx(`The largest share of attention that any head that is on puts on “${wd(CAT[LG()])}”. At “it”, it is the word “it” refers to in this lab.`,`开着的 head 里，放在“${wd(CAT[LG()])}”上的最大注意力份额。在这个 Lab 里，“它”指代的就是它。`)},
 {k:'par',label:()=>tx('Parameters in this block','这个块的参数'),v:()=>'256',sub:()=>tx('Wq, Wk, Wv, Wo: 4 × 8 × 8','Wq、Wk、Wv、Wo：4 × 8 × 8'),
  tip:()=>tx('Wq, Wk, Wv and Wo hold 8 × 8 hand-set parameters each, 256 in all. Splitting into heads only changes which of them each head uses, so the count is the same for 1, 2 or 4 heads.','Wq、Wk、Wv、Wo 各有 8 × 8 个手工设的参数，一共 256 个。拆成几个 head 只改变每个 head 用哪些参数，所以 1、2、4 个 head 时都是这么多。')}
];
/* both attention-weight metrics are hidden only in parts 1 and 2, before the prediction (R6.10: one promise) */
function hiddenNote(){return tx('shown from part 3 on','第 3 部分起显示');}
const numOf=v=>{const m=String(v).match(/^(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
function drawMetrics(r){
  const host=$('#mha-metrics',app);
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  const fh=focusHead(), rr=fh<0?r:{...r,prev:r.p>0?{v:r.heads[fh].w[r.p-1],h:fh}:r.prev};
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), hid=(masked()&&(m.k==='prev'||m.k==='cat'))||(fh>=0&&m.k==='cat'), v=hid?'?':m.v(rr), d=el.querySelector('.d');
    el.dataset.tipx=m.tip(); el.querySelector('.k').innerHTML=m.label(); el.querySelector('.v').textContent=v;   // labels are the lab's own strings; the Chinese term stays on one line
    el.querySelector('.sub').textContent=hid?hiddenNote(m.k):m.sub(rr);
    const dt=userAct&&m.pts&&!hid&&shownPrev[m.k]!=null&&shownPrev[m.k]!=='—'&&v!=='—'?(A.fmt.delta(numOf(shownPrev[m.k]),numOf(v),{pts:true,dp:0})||''):'';
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=hid?null:v;});
}

/* ---------- Show the math: every line recomputed from the numbers shown before it (R6.6) ---------- */
const vecTxt=(a,d)=>'('+a.map(v=>fx(v,d)).join(', ')+')';
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const MX=t=>A.ui.math(t);
/* the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): the formula is set as its name with “=”,
   then the right-hand side, which goes on its own line under the name when the two do not fit side by side. The right-hand side is cut
   before each + and − outside brackets into terms that wrap onto the next line under its first term; a term still wider than the box
   is cut again before ×. Every piece starts with an empty letter (\mathrm{}), and the name ends with one after its “=”, so an operator at
   the start or end of a piece (+, ×, Σ, =) keeps the spacing it has inside one formula (Chromium gives an mrow's first or last operator
   no space; an empty {} or a space does not count) */
const FWcut=(t,ops)=>{const out=[]; let d=0, last=0;
  for(let i=0;i<t.length;i++){const c=t[i];
    if(c==='{'||c==='(') d++; else if(c==='}'||c===')') d--;
    else if(d===0&&i>last&&ops.some(o=>t.startsWith(o,i))){out.push(t.slice(last,i)); last=i+1;}}
  out.push(t.slice(last)); return out;};
const FW=tex=>{const [name,...rest]=FWcut(tex,[' = ',' \\approx ']); if(!rest.length) return MX(tex);
  const rel=rest[0].startsWith('=')?'=':'\\approx', rhs=rest.join(' ').slice(rel.length+1);
  const piece=x=>`<span class="mha-fp">${MX('\\mathrm{} '+x)}</span>`;
  return `<span class="mha-fwn">${MX(name+' '+rel+' \\mathrm{}')}</span><span class="mha-fwr">${FWcut(rhs,[' + ',' - ']).map(g=>`<span class="mha-fg">${FWcut(g,[' \\times ']).map(piece).join('<wbr>')}</span>`).join('<wbr>')}</span>`;};
function mathHTML(r){
  if(masked()) return `<p>${tx('The worked numbers are shown from part 3 on, after your prediction.','预测之后，第 3 部分起显示具体的计算。')}</p>`;
  const H=r.H, hd=r.hd, p=r.p, fh=focusHead(), mh=fh>=0?fh:Math.min(mathHead,H-1), c=chainOf(r,mh), sqs=c.sq===2?'2':c.sq.toFixed(4);
  const xs=r.X[p].map((v,i)=>i<2?fx(v,3):String(v)), q=r.q.map(v=>fx(v,3));
  const qs=[...Array(H)].map((_,h)=>q.slice(h*hd,(h+1)*hd).join(', ')).join(' │ ');
  /* the owner's decisions of 2026-10-09 (H1, H2): a few short blocks, each a lead line in words, the formula typeset (A.ui.math), then
     the numbers of the current word; the tables keep their plain columns (the numbers test reads them). The owner's review of 2026-10-09
     (R6.37: a row the reader has to scroll sideways is not read): one table of six columns did not fit, so each block has its own: the
     score block lists each word's key k, q·k and score, the softmax block each word's power of e and attention weight */
  const FL=f=>{const h=FW(f); return `<div class="${h.includes('mha-fwn')?'mha-fw ':''}mha-math-line">${h}</div>`;};
  const MB=(lead,f,body)=>`<div class="mha-mb"><div class="mha-mlead">${lead}</div>${f?FL(f):''}${body}</div>`;
  const nm=v=>v<0?`(${fx(v,3)})`:fx(v,3);
  const best=c.rows.reduce((a,x)=>x.e>a.e?x:a,c.rows[0]), wTd=x=>x.j===0?'&lt;s&gt;':esc(wd(x.j));
  let h=`<p class="mha-mh">${T_('q',tx(`Current word ${wq(p)} (word ${p}, counting &lt;s&gt; as 0)`,`当前的词${wq(p)}（第 ${p} 个词，&lt;s&gt; 算第 0 个）`))}${tx('; ','；')}${T_('H',splitTitle(H))}</p>`;
  h+=MB(tx('Wq turns the current word’s 8 numbers x into its query q, what the word looks for.','Wq 把当前的词的 8 维向量 x 变成它的 query q，也就是这个词要找什么。'),'q = \\text{Wq} \\cdot x',
    `<p>x(${wq(p)}) = (${xs.join(', ')})</p><p class="small muted" style="font-family:var(--font-ui)">${tx('The 8 numbers of x, in order: the position code (2 numbers); first token, noun, animal and pronoun (1 when true, 0 when not); the constant 1; and “because” (1 or 0). In this lab these numbers are hand-made features you can read; in a trained model they are learned and cannot be read one by one.','x 的 8 维依次是：位置编码（2 维）；是不是第一个 token、名词、动物、代词（是为 1，不是为 0）；常数 1；是不是“因为”（1 或 0）。在这个 Lab 里，这 8 维是手工做的、能读懂的特征；真实模型的向量是训练出来的，一维一维读不出意思。')}</p>`+
    `<p>q = (${qs})${H>1?tx(' — one slice per head',' ——每个 head 一段'):''}</p>`);
  if(H>1&&fh<0) h+=`<div class="mha-mhp">${A.ui.seg({id:'mha-mh',key:'mh',label:tx('Head to work out','要算的 head'),options:[...Array(H)].map((_,i)=>[String(i),hn(H,i)]),value:String(mh),inline:false})}</div>`;
  if(fh>=0) h+=`<p class="small muted" style="font-family:var(--font-ui)">${tx('This part works out H1; the other heads are worked out from part 5 on.','这一部分只算 H1；其他 head 的计算从第 5 部分起显示。')}</p>`;
  h+=MB(tx('Each word the head can see gets a score: the query and that word’s key k, multiplied number by number and added up. Dividing by the square root of head_dim keeps the scores from growing just because a head has more numbers.','head 能看到的每个词都有一个得分：query 和这个词的 key k 逐维相乘，再加起来。再除以 head_dim 的平方根，得分就不会只因为 head 的维数多而变大。'),
    tx('\\text{score} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}','\\text{得分} = \\frac{q \\cdot k}{\\sqrt{\\text{head_dim}}}'),
    `<p>head_dim = 8 ÷ ${H} = ${hd}${tx(', ','，')}<span class="mha-sq">${MX(`\\sqrt{${hd}}`)}</span> = ${sqs}</p>`+
    `<p>${H>1?`${Hn(H,mh)}${tx(': its query slice',' 的 query 那段')} = ${vecTxt(c.qs,3)}`:tx('The one head uses all 8 numbers of q.','唯一的 head 用 q 的全部 8 维。')}</p>`+
    `<div class="mha-mtab mha-mtab-s"><table><tr><th>${tx('word','词')}</th><th class="mha-k">k</th><th>${MX('q \\cdot k')}</th><th>${MX(tx(`\\text{score} = \\frac{q \\cdot k}{${sqs}}`,`\\text{得分} = \\frac{q \\cdot k}{${sqs}}`))}</th></tr>`+
      c.rows.map(x=>`<tr${x===best?' class="best"':''}><td>${wTd(x)}</td><td class="mha-k">${vecTxt(x.ks,3)}</td><td>${fx(x.dot,3)}</td><td>${fx(x.sc,3)}</td></tr>`).join('')+`</table></div>`);
  const head=`<tr><th>${tx('word','词')}</th><th>${MX(tx(`e^{\\text{score} - ${nm(c.mx)}}`,`e^{\\text{得分} - ${nm(c.mx)}}`))}</th><th class="mha-aw">${tx('attention weight','注意力权重')}</th></tr>`;
  const rows=c.rows.map(x=>`<tr${x===best?' class="best"':''}><td>${wTd(x)}</td><td>${x.e.toFixed(4)}</td><td>${rd(x.w*100,1).toFixed(1)}%</td></tr>`).join('');
  const later=NT-1-p, bw=best.j===0?'&lt;s&gt;':esc(wd(best.j));
  h+=MB(tx('Softmax turns the scores into attention weights that add up to 100%. Subtracting the highest score first keeps every power of e at most 1 and leaves the weights unchanged. Σ adds up over every word the head can see.','softmax 把得分变成加起来是 100% 的注意力权重。先减去最高分，每个 e 的幂都不超过 1，权重也不变。Σ 表示对 head 能看到的每个词求和。'),
    tx('\\text{attention weight} = \\frac{e^{\\text{score} - \\text{highest score}}}{\\sum e^{\\text{score} - \\text{highest score}}}','\\text{注意力权重} = \\frac{e^{\\text{得分} - \\text{最高分}}}{\\sum e^{\\text{得分} - \\text{最高分}}}'),
    `<div class="mha-mtab mha-mtab-w"><table>${head}${rows}</table></div>`+
    `<p>${MX(tx(`\\sum e^{\\text{score} - ${nm(c.mx)}}`,`\\sum e^{\\text{得分} - ${nm(c.mx)}}`))} = ${c.sum.toFixed(4)}</p>`+
    `<p>${tx(`For ${bw}: `,`${bw}：`)}${MX(`\\frac{${best.e.toFixed(4)}}{${c.sum.toFixed(4)}}`)} = ${rd(best.w*100,1).toFixed(1)}%</p>`+
    (later>0?`<p>${tx(`The ${later} word${later===1?'':'s'} after ${wq(p)} ${later===1?'is':'are'} masked: no score, attention weight 0.`,`${wq(p)}后面的 ${later} 个词被 mask 掉：没有得分，注意力权重是 0。`)}</p>`:''));
  const vB='('+Array.from({length:hd},(_,i)=>i%2===0?'1':'0').join(', ')+')', vC='('+Array.from({length:hd},(_,i)=>i%2===1?'1':'0').join(', ')+')';
  const terms=[], wOf=j=>{const x=c.rows.find(y=>y.j===j); return x?rd(x.w,3):null;};
  if(wOf(BECAUSE)!=null) terms.push(`${wOf(BECAUSE).toFixed(3)} × ${vB}`); if(wOf(r.cat_i)!=null) terms.push(`${wOf(r.cat_i).toFixed(3)} × ${vC}`);
  h+=MB(tx('The head’s output adds up the words’ values v, each multiplied by its attention weight.','head 的输出，是把每个词的 value v 乘上它的注意力权重，再加起来。'),
    tx('\\text{output} = \\sum \\text{attention weight} \\times v','\\text{输出} = \\sum \\text{注意力权重} \\times v'),
    `<p>${tx(`${Hn(H,mh)}’s output`,`${Hn(H,mh)} 的输出`)} = ${terms.length?terms.join(' + ')+' = ':''}${vecTxt(c.out,3)}</p><p class="small muted" style="font-family:var(--font-ui)">${tx(`v(“${esc(wd(BECAUSE))}”) = ${vB}, v(“${esc(wd(r.cat_i))}”) = ${vC}. Every other word’s value is 0.`,`v(“${esc(wd(BECAUSE))}”) = ${vB}，v(“${esc(wd(r.cat_i))}”) = ${vC}。其他词的 value 都是 0。`)}</p>`);
  if(fh<0){const outs=[...Array(H)].map((_,i)=>r.heads[i].on?chainOf(r,i).out.map(v=>fx(v,3)).join(', '):Array(hd).fill(fx(0,3)).join(', ')).join(' │ ');
    h+=MB(tx('The block puts the heads’ outputs side by side and multiplies them by Wo. Here Wo is the identity, so the block’s output is the heads’ outputs as they are.','块把各个 head 的输出拼在一起，再乘以 Wo。这里 Wo 是单位矩阵，所以块的输出就是拼起来的各个 head 的输出。'),
      tx('\\text{block output} = \\text{Wo} \\cdot (\\text{the heads’ outputs, side by side})','\\text{块的输出} = \\text{Wo} \\cdot (\\text{拼起来的各个 head 的输出})'),
      `<p>${tx('Block output','块的输出')} = (${outs})${H===4&&r.on.some(v=>!v)?tx('. A head that is off gives 0s.','。关掉的 head 给出 0。'):''}</p>`);}
  const mx=(W,name,split)=>`<table class="mha-mx"><tr><th></th>${FNAMES.map(f=>`<th>${f}</th>`).join('')}</tr>${W.map((row,i)=>`<tr${split&&i>0&&i%hd===0?' class="split"':''}><th>${name} ${i+1}</th>${row.map(v=>`<td>${Math.abs(v)<1e-9?'0':fx(v,3)}</td>`).join('')}</tr>`).join('')}</table>`;
  h+=`<details class="mha-mxs"><summary>${tx('The matrices Wq and Wk (hand-set)','矩阵 Wq 和 Wk（手工设的）')}</summary><p class="small muted" style="font-family:var(--font-ui)">${tx('Rows are the numbers of q or k, columns the features of x; a thick line starts a new head. Wv puts “because” into every head’s first value number and “animal” into its second; Wo is the identity.','行是 q 或 k 的各维，列是 x 的各个特征；粗线处开始下一个 head。Wv 把“因为”放进每个 head 的 value 的第一维，把“动物”放进第二维；Wo 是单位矩阵。')}</p>${mx(M.Wq,'q',true)}${mx(M.Wk,'k',true)}</details>`;
  return h;
}

/* ---------- controls, rail sync ---------- */
function syncControls(r){
  $$('.seg-ctl[data-key="H"]',app).forEach(g=>A.ui.setPressed(g,String(S.H)));
  $$('[data-q]',app).forEach(b=>{if(b.closest('.mha-grid')) return; const j=+b.dataset.q; b.setAttribute('aria-pressed',String(j===S.q)); b.classList.toggle('mha-later',qLock(j));
    if(qLock(j)){b.setAttribute('aria-disabled','true'); b.dataset.tipx=tx('“it” comes in part 4.','“它”留到第 4 部分。');} else {b.removeAttribute('aria-disabled'); delete b.dataset.tipx;}});
  $$('[data-on]',app).forEach(b=>b.setAttribute('aria-pressed',String(!!S.on[+b.dataset.on])));
  $$('.mha-onlist',app).forEach(el=>{el.textContent=onList();});
  const lock=mode==='challenge', c=$('#mha-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#mha-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you can only turn heads on and off, in the card. Everything else is fixed: q, k and v split into 4 heads of 2 numbers, and the current word is “it”.','挑战中只能在卡片里开关 head。其余设置固定：q、k、v 拆成 4 个 head，每个 2 维，当前的词是“它”。');
  const hLock=mode==='guided'&&step<SIX('two'), hw=$('#mha-ctl-H-wrap',app), hn_=$('#mha-H-note',app);
  hw.hidden=hLock; hn_.hidden=!hLock||lock; if(hLock) hn_.textContent=tx('The number of heads opens in part 5.','到第 5 部分才能改 head 数。');
  const onOk=mode!=='guided'&&S.H===4, ow=$('#mha-ctl-on',app), onn=$('#mha-on-note',app);
  ow.hidden=!onOk||lock; onn.hidden=onOk||lock; if(!onOk) onn.textContent=mode==='guided'?tx('Heads can be turned off in Scenarios and in the challenge.','在“应用场景”和挑战题里可以关掉 head。'):tx('Heads can be turned off with 4 heads.','4 个 head 时才能关掉 head。');
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(r){
  const rail=$('#mha-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(userAct) acted[step]=true;
    if(s.expect&&s.expect(r)) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||!!acted[step]);   // any change the learner makes in a part gets a result (R1.9), with no verdict before its action (N8, M3)
    if(box){box.hidden=!show;
      if(show){const lab=L(s.label||{en:'Observed',zh:'观察到'}), html=`<div class="eyebrow">${lab}</div>`+s.obs(r);
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(wantObs&&fromRail) revealSoon();}}   // Settings and the chart never scroll the page (R3.2)
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#mha-insight',rail); if(ins){const html=`<div class="eyebrow">${cur>=0?esc(PRESETS[cur].t()):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(LG(),S.on), set=(id,v)=>{const g=$('#'+id,rail), pr=$('#'+id+'p',rail); if(!g) return; g.querySelector('b').textContent=pc(v); const ok=v>=CH_GOAL;
      g.classList.toggle('met',ok); g.classList.toggle('mha-bad',!ok); pr.classList.toggle('met',ok); pr.classList.toggle('mha-bad',!ok); pr.querySelector('i').style.width=Math.min(100,v/CH_GOAL*100)+'%';};
    set('mha-ch-b',p.bb); set('mha-ch-c',p.bc);
    const oc=$('#mha-ch-o b',rail); if(oc) oc.textContent=String(p.off);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#mha-ch-res',rail), html=chResultHTML(); if(res&&res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function renderRail(){const rail=$('#mha-rail',app);
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');   // the next lab in every mode (R3.11)
  render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  const r=R();
  drawHead(r); drawGrid(r); drawMats(r); drawMetrics(r);
  const mb=$('#mha-mathbody',app), mh=mathHTML(r); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(r); updateRail(r);
  A.hash.write(token()); A.dock.refresh(); announce(r);
}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return;
  if(masked()){A.announce(tx('The attention weights of “it” are hidden until you answer part 2.','在第 2 部分作答前，“它”的注意力权重先隐藏。')); return;}
  A.announce(tx(`${wt(r.p)} with ${splitTxt(r.H)}: `,`${wt(r.p)}，${splitTxt(r.H)}：`)+r.heads.map((o,h)=>{const t=tops(o,0.3)[0]; return t?tx(`${Hn(r.H,h)} ${pc(t.w)} on ${wt(t.j)}`,`${Hn(r.H,h)} 在${wt(t.j)}上 ${pc(t.w)}`):'';}).filter(Boolean).join(tx('; ','；'))+tx('.','。'));},400);}
function setState(p){
  if(p.H!=null&&+p.H!==S.H){focus=-1; if(+p.H!==4) S.on=ALL_ON();}
  Object.assign(S,p);
  if(mode==='challenge'){S.q=IT; S.H=4;}
  if(![4,2,1].includes(S.H)) S.H=4;
  if(S.H!==4) S.on=ALL_ON();
  S.q=Math.max(0,Math.min(NT-1,S.q|0));
  render();
}
/* a change the learner made: its metric cards may show a ▲▼ badge, and a part's result comes into view */
function byLearner(fn){userAct=true; wantObs=true; try{fn();} finally{userAct=false; wantObs=false;}}
function enterStep(i){
  step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  focus=s.focus?0:-1;
  const st=$('#mha-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.expect){const el=$('#mha-ic-q',app)||$('#mha-ic-H',app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#mha-rail .step.current',app)||$('#mha-rail',app));}
/* keepStage (CONTRACT §11, R3.2, R9.7; as labs/moe.js; polish round 5): after Next, Back or a part's title the kit's reveal scrolls the
   new card's top up under the top bar; near the end of the bench that pushes the sticky stage up with it (part 7 entered from part 6 at
   1440 × 900: the stage's top at −412 px, its head and the four heads' rows above the screen). Once the page stops scrolling this pulls
   the stage back under the top bar without pushing the card's top off the screen, and does nothing once the reader has scrolled the
   card away */
let stageT=0;
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#mha-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#mha-rail .step.current',app)||$('#mha-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the result of a part comes into view after the learner's action (R3.10): scroll only as far as needed to show it,
   never so far that the control just used, or the chart row the card talks about, slides under the top bar or the mini
   result bar (N1). A same-frame probe measures where the sticky chart ends up, since the end of the bench can push it up. */
let lastClick=null;
function anchorEl(){
  const c=lastClick&&lastClick.isConnected?lastClick.closest('.inline-ctl,.preset,.choices,#mha-ic-on'):null;
  if(c&&app.contains(c)&&c.closest('#mha-rail')) return c;
  if(mode==='guided') return $('#mha-rail .step.current .inline-ctl',app);
  if(mode==='challenge') return $('#mha-ic-on',app);
  const pr=$('#mha-rail .preset[aria-pressed="true"]',app); return pr||null;}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), narrow=innerWidth<=760;
  const safe=(bar?bar.getBoundingClientRect().bottom:56)+(dk&&!dk.hidden?dk.offsetHeight:(narrow?52:0))+10;
  const r=box.getBoundingClientRect(), anc=anchorEl(), a=anc&&anc.getBoundingClientRect(), ancIn=!!a&&a.bottom>0&&a.top<innerHeight;
  if(r.top>=safe&&r.bottom<=innerHeight-8&&!(ancIn&&a.top<safe)) return;   // the whole result is in view, the control clear of the bars
  const need=r.top+56-innerHeight;                                         // the least scroll that shows its first line
  let dy=Math.max(0,Math.min(r.bottom-innerHeight+16, r.top-safe));       // scroll down to show its end, never its top under the bars
  if(ancIn) dy=Math.min(dy,a.top-safe);        // keep the control used below the bars; if it is half under them, scroll back up (M4)
  if(dy<need) dy=need;
  const row=$('#mha-grid .mha-row',app), rr=row&&row.getBoundingClientRect();
  if(dy>0&&rr&&rr.top>=safe&&rr.bottom<=innerHeight){                     // the chart row is in view now: keep it there
    const y0=scrollY; window.scrollTo({top:y0+dy,behavior:'instant'}); const r2=row.getBoundingClientRect(); window.scrollTo({top:y0,behavior:'instant'});
    if(r2.top<safe) dy=Math.max(need,dy-(safe-r2.top));}
  if(Math.abs(dy)>=2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
let revRaf=0;
function revealSoon(){if(revRaf) return; revRaf=requestAnimationFrame(()=>{revRaf=0; if(!app) return;
  const box=mode==='guided'?$('#mha-rail [data-obs]',app):(mode==='sandbox'?$('#mha-insight',app):$('#mha-ch-res',app)); revealBox(box);});}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S,on:S.on.slice()}; chSubmitted=false; chReveal=false; S={q:IT,H:4,on:ALL_ON()};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#mha-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  focus=-1; renderRail();
}
/* a scenario's result sits under the list of five: on smaller screens (1280 x 800, phones) it starts below the fold, so it
   comes into view the same way as a part's result, with the pressed scenario kept on screen (R3.10) */
function applyPreset(i){const p=PRESETS[i]; byLearner(()=>{focus=-1; S.on=ALL_ON(); setState({...p.s,on:ALL_ON()});}); revealSoon();}
const token=()=>[S.q,S.H,onKey(S.on)].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==3) return null;
  const q=+p[0], H=+p[1], on=p[2];
  if(!Number.isInteger(q)||q<0||q>=NT||![1,2,4].includes(H)||!/^[01]{4}$/.test(on)) return null;
  return {q,H,on:on.split('').map(v=>v==='1')};
}
function act(a,btn){
  if(a==='reset'){byLearner(()=>{if(mode==='challenge') setState({on:ALL_ON()}); else if(mode==='guided'){focus=-1; const e=STEPS[step].enter; setState(e?e():{q:IT,H:1,on:ALL_ON()});} else {focus=-1; setState({q:IT,H:4,on:ALL_ON()});}}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(LG(),S.on).ok) chReveal=true; renderRail(); revealSoon(); return;}   // anchored on the switches (M4)
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#mha-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}

/* ---------- interview drill: generated and seeded, numbers that redo by hand ---------- */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DRILL=[[2048,16,8],[2048,32,8],[3072,24,8],[4096,32,8],[5120,40,8],[8192,64,8],[4096,32,4]];
function makeDrill(seed){const rnd=mulberry(seed), [d,H,G]=DRILL[Math.floor(rnd()*DRILL.length)], hd=d/H;
  return {seed,d,H,G,hd,params:4*d*d,hd2:hd/2,H2:2*H,kvMha:2*H*hd,kvGqa:2*G*hd};}
function renderDrill(){
  if(!app) return; const k=makeDrill(drillSeed), q=$('#mha-drill-q',app), a=$('#mha-drill-a',app), P_=A.ui&&A.ui.big?A.ui.big(k.params):fi(k.params);
  q.innerHTML=tx(`A layer has hidden size d = <b>${fi(k.d)}</b> and <b>${k.H}</b> heads in its attention block, with no biases. (1) What is head_dim? (2) How many parameters do Wq, Wk, Wv and Wo hold together when every head has its own key and value (multi-head attention, MHA)? (3) With ${k.H2} heads instead, same d: what are head_dim and the parameter count? (4) How many numbers per token does this layer add to the KV cache with MHA, and with ${k.G} KV heads (grouped-query attention, GQA)?`,
                 `一层的隐藏维度 d = <b>${fi(k.d)}</b>，attention 块里有 <b>${k.H}</b> 个 head，没有 bias。（1）head_dim 是多少？（2）每个 head 都有自己的 key 和 value（多头注意力，MHA）时，Wq、Wk、Wv、Wo 一共多少参数？（3）d 不变、改成 ${k.H2} 个 head，head_dim 和参数量各是多少？（4）用 MHA 时，这一层每个 token 要往 KV cache 里存多少个数值？用 ${k.G} 个 KV head（grouped-query attention，GQA）时呢？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>(1) ${k.hd}. (2) ${P_}. (3) ${k.hd2} and still ${P_}. (4) ${fi(k.kvMha)} and ${fi(k.kvGqa)}.</b></p><p>(1) head_dim = d ÷ heads = ${fi(k.d)} ÷ ${k.H} = ${k.hd}.</p><p>(2) Each of the four matrices is d × d: 4 × ${fi(k.d)} × ${fi(k.d)} = ${P_}. The head count does not appear.</p><p>(3) head_dim = ${fi(k.d)} ÷ ${k.H2} = ${k.hd2}. The matrices are still d × d, so the count stays ${P_}: more heads only cut q, k and v into more slices.</p><p>(4) Each KV head keeps a key and a value of head_dim numbers: MHA 2 × ${k.H} × ${k.hd} = ${fi(k.kvMha)}; GQA 2 × ${k.G} × ${k.hd} = ${fi(k.kvGqa)} per token in this layer.</p>`,
                 `<p><b>（1）${k.hd}。（2）${P_}。（3）${k.hd2}，参数还是 ${P_}。（4）${fi(k.kvMha)} 和 ${fi(k.kvGqa)}。</b></p><p>（1）head_dim = d ÷ head 数 = ${fi(k.d)} ÷ ${k.H} = ${k.hd}。</p><p>（2）四个矩阵都是 d × d：4 × ${fi(k.d)} × ${fi(k.d)} = ${P_}。式子里没有 head 数。</p><p>（3）head_dim = ${fi(k.d)} ÷ ${k.H2} = ${k.hd2}。矩阵还是 d × d，所以参数量仍是 ${P_}：head 多了，只是把 q、k、v 切成更多段。</p><p>（4）每个 KV head 为每个 token 存一个 key 和一个 value，各 head_dim 维：MHA 是 2 × ${k.H} × ${k.hd} = ${fi(k.kvMha)} 个数值；GQA 是 2 × ${k.G} × ${k.hd} = ${fi(k.kvGqa)} 个数值，都是这一层每个 token 存的。</p>`);
}

/* ---------- term tooltips and jumps ---------- */
const locked=k=>(mode==='challenge'&&(k==='q'||k==='H'))||(k==='H'&&mode==='guided'&&step<SIX('two'))||(k==='on'&&(mode==='guided'||S.H!==4));
function jumpAct(k){
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='grid') return tx('Click to jump to the chart.','点击跳到图表。');
  if(k==='mats') return tx('Click to jump to the four matrices.','点击跳到四个矩阵。');
  if(k==='q'){if(locked(k)) return tx('Fixed during the challenge: the current word is “it”.','挑战中固定不变：当前的词是“它”。'); return tx(`Click to jump to “${QLABEL()}”.`,`点击跳到“${QLABEL()}”。`);}
  if(k==='H'){if(mode==='challenge') return tx('Fixed during the challenge: q, k and v split into 4 heads of 2 numbers.','挑战中固定不变：q、k、v 拆成 4 个 head，每个 2 维。'); if(locked(k)) return tx('The number of heads opens in part 5.','到第 5 部分才能改 head 数。'); return tx(`Click to jump to “${HLABEL()}”.`,`点击跳到“${HLABEL()}”。`);}
  if(k==='on'){if(mode==='guided') return tx('Heads can be turned off in Scenarios and in the challenge.','在“应用场景”和挑战题里可以关掉 head。'); if(S.H!==4) return tx('Heads can be turned off with 4 heads.','4 个 head 时才能关掉 head。'); return tx('Click to jump to “Heads that are on”.','点击跳到“开着的 head”。');}
  return '';
}
const TERM={q:()=>tx('The current word: its query is compared with the key of every word it can see.','当前的词：拿它的 query 和它能看到的每个词的 key 比较。'),
  H:()=>tx('The 8 numbers of every query, key and value are split into this many heads.','每个 query、key、value 的 8 维分给这么多个 head。')};
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#mha-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#mha-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='grid'||k==='mats'){const el=k==='grid'?$('#mha-graph',app):$('#mha-mats',app); A.tip.hide(); el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(locked(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=k==='q'?($('#mha-ic-q',app)||$('#mha-ctl-q',app)):(k==='H'?($('#mha-ic-H',app)||$('#mha-ctl-H',app)):($('#mha-ic-on',app)||$('#mha-ctl-on',app)));
  if(el) A.ui.jumpToEl(el);
}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset; lastClick=b; fromRail=!!b.closest('#mha-rail');
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key;
      if(k==='mh'){mathHead=+d.v; render(); return;}
      if(k==='H'){byLearner(()=>setState({H:+d.v})); return;}
      return;}
    if(d.q!=null){if(mode==='challenge'){A.tip.show(b,`<div>${esc(jumpAct('q'))}</div>`); return;}
      if(qLock(+d.q)){A.tip.show(b,`<div>${esc(tx('“it” comes in part 4, right after this part.','“它”留到第 4 部分，就在这一部分之后。'))}</div>`); return;}
      byLearner(()=>setState({q:+d.q})); return;}
    if(d.on!=null){const h=+d.on, on=S.on.slice(); on[h]=!on[h]; byLearner(()=>setState({on})); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){acted[step]=true; wantObs=true; setState(s.showMe); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const f=$(`#mha-rail [data-pred="${pred}"]`,app); if(f) f.focus(); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#mha-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S,on:S.on.slice()},pred,observed:{...observed},acted:{...acted},focus,saved:saved?{...saved,on:saved.on.slice()}:null,chSubmitted,chReveal,
    mathHead,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(LG(),S.on).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'&&Array.isArray(x.on)?parse([x.q,x.H,x.on.map(v=>v===true?1:(v===false?0:'x')).join('')].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={}; acted={};
    for(const [src,dst] of [[o.observed,observed],[o.acted,acted]]) if(src&&typeof src==='object') for(const k in src) if(/^\d+$/.test(k)&&+k<STEPS.length&&src[k]===true) dst[k]=true;
    focus=Number.isInteger(o.focus)&&o.focus>=-1&&o.focus<S.H?o.focus:-1;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    mathHead=Number.isInteger(o.mathHead)&&o.mathHead>=0&&o.mathHead<4?o.mathHead:0;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true;
    mounted=true;},   // the first mount keeps S as it was, without the part's starting setup
  mount(el,tok){
    root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; focus=-1;}
    el.innerHTML=viewHTML(); app=$('#mha-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#mha-rail',app));},token});
    bind(app); renderDrill();
    if(!mounted&&!p){const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter()); focus=s.focus?0:-1;}
    if(mode==='guided'&&qLock(S.q)) S.q=3;
    mounted=true;
    renderRail();
  },
  leave(){clearTimeout(annT); stageStop(); if(revRaf){cancelAnimationFrame(revRaf); revRaf=0;} root=null; app=null;},
  onLang(){if(root){const d=$('#mha-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#mha-math',app).open=true;}},
  termTip(k){const f=TERM[k]; if(!f) return ''; return `<div>${esc(f())}</div><div class="act">${esc(jumpAct(k))}</div>`;},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#mha-grid',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#mha-graph',app),
    render({status:stEl,vis,read}){if(!app) return; const r=R(), st=status(r), mem=x=>`<span class="dk-mem mha-dk-mem">${x}</span>`;
      stEl.className='status mha-dk-st '+st.cls; stEl.textContent=st.short||st.t;
      if(masked()){vis.innerHTML=''; read.innerHTML=esc(pred?tx('The attention weights of “it” are shown from part 3 on','“它”的注意力权重从第 3 部分起显示'):tx('The attention weights of “it” are shown after you predict','“它”的注意力权重在预测之后显示')); return;}
      const shown=focusHead()>=0?[r.heads[focusHead()]]:r.heads.filter(o=>o.on);
      vis.innerHTML=`<div class="mha-strip">${SENT[r.lang].map((w,j)=>j>r.p?'<i class="m"></i>':`<i style="height:${(Math.max(0,...shown.map(o=>o.w[j]))*100).toFixed(1)}%"></i>`).join('')}</div>`;
      const b=`“${wd(BECAUSE)}”`, c=`“${wd(r.cat_i)}”`, how=splitTxt(r.H), fh=focusHead();
      if(fh>=0&&r.p>0){read.innerHTML=esc(tx(`At ${wt(r.p)}: H1 puts ${pc(r.heads[fh].w[r.p-1])} on the previous word, ${wt(r.p-1)}`,`当前的词是${wt(r.p)}：H1 在前一个词${wt(r.p-1)}上 ${pc(r.heads[fh].w[r.p-1])}`))+mem(esc(tx(` (${how})`,`（${how}）`))); return;}
      if(r.p===IT&&mode==='challenge'){const off=chPoint(LG(),S.on).off;   // N6: the heads-off count stays in the sentence, which phones also show
        read.innerHTML=esc(tx(`At “it”, ${pc(r.bec.v)} on ${b} and ${pc(r.catw.v)} on ${c}, with ${off} of 4 heads off`,`当前的词是“它”：${b}${pc(r.bec.v)}，${c}${pc(r.catw.v)}，关掉了 ${off} / 4 个 head`))
          +mem(esc(tx(` (the best head that is on; goal ${pc(CH_GOAL)} each)`,`（开着的 head 里最高；目标各 ${pc(CH_GOAL)}）`)));}
      else if(r.p===IT) read.innerHTML=esc(tx(`At “it”: ${pc(r.bec.v)} on ${b}, ${pc(r.catw.v)} on ${c}`,`当前的词是“它”：${b}${pc(r.bec.v)}，${c}${pc(r.catw.v)}`))
        +mem(esc(tx(` (${how}; the best head that is on)`,`（${how}；开着的 head 里最高）`)));
      else read.innerHTML=esc(r.p===0?tx('At <s>: 100% on <s> itself','当前的词是 <s>：<s> 自己 100%'):tx(`At ${wt(r.p)}: ${pc(r.prev.v)} on the previous word, ${wt(r.p-1)}`,`当前的词是${wt(r.p)}：前一个词${wt(r.p-1)} ${pc(r.prev.v)}`))
        +mem(esc(tx(` (${how}; the ${NT-1-r.p} later words are masked)`,`（${how}；后面 ${NT-1-r.p} 个词被 mask）`)));}
  }
});

/* ---------- home tile: one control, the weights of “it”, one sentence (R11.5) ---------- */
let tzH=2;   // the tile opens on 2 heads; the one-head split is the prediction of part 2, so the reader clicks to see it
A.registerTeaser(ID,{fid:'toy',render(el){
  el.innerHTML=`<div class="tz">${A.ui.seg({key:'tzH',label:tx('Heads in the attention block','attention 块的 head 数'),options:[['1',tx('1 head','1 个')],['2',tx('2 heads','2 个')]],value:String(tzH),inline:false})}`+
    `<div class="tz-vis mha-tz"></div><p class="mha-tz-cap"></p><p class="tz-read"></p>`+
    `<p class="tz-note">${tx('This is a toy run: your browser computes real attention on a 9-token sentence (a start token and one token per word), and the parameters of its matrices are set by hand.','这是玩具实跑：浏览器在一个 9 个 token 的句子（一个起始 token，每个词或标点各一个 token）上真的算一遍 attention，矩阵的参数是手工设的。')}</p>`+
    `<p class="tz-note">${tx('A head’s attention weights come from a softmax, so they add up to 100%.','一个 head 的注意力权重来自 softmax，所以加起来是 100%。')}</p>`+
    `<p class="tz-note">${tx('Both settings use the same four 8 × 8 matrices; only how the 8 numbers of q, k and v are split into heads changes (Vaswani et al., 2017, §3.2.2).','两种设置用的是同样的四个 8 × 8 矩阵，只有 q、k、v 的 8 维怎么分给 head 不同（Vaswani 等，2017，§3.2.2）。')}</p></div>`;
  const vis=el.querySelector('.mha-tz'), rdEl=el.querySelector('.tz-read'), cap=el.querySelector('.mha-tz-cap'), seg=el.querySelector('.seg-ctl');
  const upd=()=>{const lg=LG(), r=compute({lang:lg,q:IT,H:tzH}), ci=r.cat_i, b=esc(wd(BECAUSE)), c=esc(wd(ci)), labs=tzH===1?[tx('<b>One head</b>','<b>一个 head</b>')]:[tx('<b>Head A</b>','<b>head A</b>'),tx('<b>Head B</b>','<b>head B</b>')];
    let k=0; vis.style.setProperty('--mha-cols',colsOf(lg)); vis.innerHTML=gridHTML(r,{tile:true}).replace(/<div class="mha-rlab">[^]*?<\/div>/g,()=>`<div class="mha-rlab">${labs[k++]||''}</div>`);
    A.ui.setPressed(seg,String(tzH));
    cap.textContent=tzH===1?tx('Bars: how much of the head’s attention “it” puts on each word. Dashed: the two words “it” needs.','竖条：“它”的注意力有多少放在每个词上。虚线框：“它”需要的两个词。')
      :tx('Bars: in each head, how much of “it”’s attention goes to each word. Dashed: the two words “it” needs.','竖条：在每个 head 里，“它”把多少注意力放在每个词上。虚线框：“它”需要的两个词。');
    const o=r.heads[0], half=o.w[BECAUSE]<0.5&&o.w[ci]<0.5;
    rdEl.innerHTML=tzH===1?tx(`With one head, “it” splits its attention: <b>${pc(o.w[BECAUSE])}</b> on “${b}” and <b>${pc(o.w[ci])}</b> on “${c}”${half?', neither above half':''}.`,`一个 head 时，“它”的注意力被分开了：“${b}”<b>${pc(o.w[BECAUSE])}</b>，“${c}”<b>${pc(o.w[ci])}</b>${half?'，都不到一半':''}。`)
      :tx(`With two heads, each has its own job: head A puts <b>${pc(o.w[BECAUSE])}</b> on “${b}”, head B puts <b>${pc(r.heads[1].w[ci])}</b> on “${c}”.`,`两个 head 时，各司其职：head A 把 <b>${pc(o.w[BECAUSE])}</b> 放在“${b}”上，head B 把 <b>${pc(r.heads[1].w[ci])}</b> 放在“${c}”上。`);};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b) return; tzH=+b.dataset.v; upd();});
  upd();
}});

/* read-only hooks for the tests; exits() gives the capability layer's text in the current language (tests/mha_copy_deck.py) */
window.__mha=Object.freeze({compute:s=>compute(s),chain:(s,h)=>chainOf(compute(s),h),chPoint,combos,realModel,
  exits:()=>{const e=EXITS(); return {explain:e.explain.map(q=>({q:q.q,o:q.o,a:q.a,n:q.n,why:q.why})),transfer:e.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why}))};},
  consts:Object.freeze({D,NT,OM,SENT,FEAT,IT,BECAUSE,CAT,P,FNAMES,CH_GOAL,SQRT}),mats:()=>JSON.parse(JSON.stringify(M)),x:(lang,p)=>feats(lang,p)});
})();
