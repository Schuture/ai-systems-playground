/* eval-ci (world eval): Is a 2-point benchmark gain real?
   Fidelity: Analytical. Every interval on the page comes from a formula (normal approximation):
   - one model, accuracy p on n questions: SE = √(p(1−p)/n), 95% interval p ± 1.96·SE;
   - two models on the same n questions, q = share with exactly one right, Δ = pB − pA: paired SE = √((q − Δ²)/n);
     separate: √(SE_A² + SE_B²);
   - k answers per question, c = share of coin-flip questions (both models right half the time, the rest always right or wrong):
     Var_A = pA(1−pA) − (c/4)(1 − 1/k), paired Var(d) = q − Δ² − (c/2)(1 − 1/k) (tests/eval-ci_numbers.py checks it by simulation);
   - questions for a half-width h: n = 1.96²·p(1−p)/h².
   Part 2's re-drawn benchmarks are a seeded picture of the first formula (mulberry32), not a recording.
   Words (EN / ZH): error bar / 误差线 (the drawn 95% interval), point / 分 (a percentage point), question by question / 逐题对比,
   each on its own / 各算各的, coin-flip questions / 时对时错的题, part / 部分.
   window.__evc is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='eval-ci';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const Z=1.96;
const vOne=(p,c,k)=>p*(1-p)-c/4*(1-1/k);                 // one model's variance per question, k answers averaged
function cmp(o){
  const D=o.pB-o.pA, vA=vOne(o.pA,o.c,o.k), vB=vOne(o.pB,o.c,o.k);
  const seA=Math.sqrt(vA/o.n), seB=Math.sqrt(vB/o.n), seS=Math.sqrt(seA*seA+seB*seB);
  const vP=o.q-D*D-o.c/2*(1-1/o.k), seP=Math.sqrt(Math.max(vP,0)/o.n);
  const se=o.paired?seP:seS, h=Z*se;
  return {D,vA,vB,vP,seA,seB,hA:Z*seA,hB:Z*seB,seS,hS:Z*seS,seP,hP:Z*seP,se,h,lo:D-h,hi:D+h,
    real:D-h>0||D+h<0,overlap:o.pA+Z*seA>o.pB-Z*seB&&o.pB+Z*seB>o.pA-Z*seA};
}
const nFor=(p,h)=>Z*Z*p*(1-p)/(h*h);
const floorH=(p,c,n)=>Z*Math.sqrt((p*(1-p)-c/4)/n);
/* standard normal CDF (Abramowitz–Stegun 7.1.26, error < 1.5e-7), for the challenge's power note */
function Phi(x){const s=x<0?-1:1, t0=Math.abs(x)/Math.SQRT2, t=1/(1+0.3275911*t0);
  const e=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-t0*t0); return 0.5*(1+s*e);}
/* power: the chance that the error bar of B − A stays clear of 0 when the true gap is D (both tails) */
const power=(h,D)=>{const se=h/Z; return Phi(D/se-Z)+Phi(-D/se-Z);};
const H80=D=>D*Z/(Z+0.8416212);                          // the half-width that gives 80% power for a true gap D
/* the joint mix behind q and c must exist: shares of the five kinds of question all ≥ 0 */
function feasible(o){const D=o.pB-o.pA, d0=o.q-o.c/2, w3=(d0+D)/2, w4=(d0-D)/2, w1=o.pA-w4-o.c/2, w2=1-w1-w3-w4-o.c;
  return [w1,w2,w3,w4].every(w=>w>-1e-9);}
function fixQC(s){
  const D=s.pB-s.pA, cMax=Math.min(2*s.pA,2*(1-s.pA),2*s.pB,2*(1-s.pB));
  if(s.c>cMax) s.c=Math.floor(cMax*20+1e-9)/20;
  const qMin=Math.abs(D)+s.c/2, qMax=s.c/2+Math.min(2*s.pA-s.c+D,2*(1-s.pA)-s.c-D);
  if(s.q<qMin-1e-9) s.q=Math.ceil(qMin*100-1e-9)/100;
  if(s.q>qMax+1e-9) s.q=Math.floor(qMax*100+1e-9)/100;
  return s;
}
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* part 2: K benchmarks of 164 questions, each question right with chance 0.8 */
const RD={K:200,n:164,p:0.8,seed:7};
const REDRAWS=(()=>{const r=mulberry(RD.seed), out=[]; for(let i=0;i<RD.K;i++){let c=0; for(let j=0;j<RD.n;j++) if(r()<RD.p) c++; out.push(c);} return out;})();
const RD_H=Z*Math.sqrt(RD.p*(1-RD.p)/RD.n);
const inside=m=>REDRAWS.slice(0,m).filter(c=>Math.abs(c/RD.n-RD.p)<=RD_H+1e-12).length;

/* the size slider's stops; three real benchmarks are ticks on it */
const NS=[100,164,200,300,500,700,1000,1319,2000,3000,4000,5000,6200,8000,10000,14000,20000];
const TICKS=[{n:164,t:'HumanEval'},{n:1319,t:'GSM8K'},{n:14000,t:'MMLU'}];
const KS=[1,2,4,8,16];
/* how often a 95% error bar drawn from one score covers the true level (exact binomial sum), for the fidelity card's claim (S-3):
   the lowest coverage for scores 50–90% (0.5-point grid) on the slider's stops from 164 questions up; computed once, on first use.
   The page floors it (92%): finer scores and question counts between the stops dip to about 92.0% (tests/eval-ci_numbers.py). */
let LF=null, covMemo=null;
const logFact=n=>{if(!LF){LF=[0]; let s=0; for(let i=1;i<=NS[NS.length-1];i++){s+=Math.log(i); LF.push(s);}} return LF[n];};
function coverage(p,n){const sd=Math.sqrt(n*p*(1-p)), a=Math.max(0,Math.floor(n*p-12*sd)), b=Math.min(n,Math.ceil(n*p+12*sd)); let t=0;
  for(let x=a;x<=b;x++){const ph=x/n, h=Z*Math.sqrt(ph*(1-ph)/n); if(ph-h<=p&&p<=ph+h) t+=Math.exp(logFact(n)-logFact(x)-logFact(n-x)+x*Math.log(p)+(n-x)*Math.log(1-p));}
  return t;}
const COV={get min(){if(covMemo==null){covMemo=1; NS.filter(n=>n>=164).forEach(n=>{for(let i=500;i<=900;i+=5) covMemo=Math.min(covMemo,coverage(i/1000,n));});} return covMemo;}};
const nIdx=n=>{let b=0; NS.forEach((v,i)=>{if(Math.abs(v-n)<Math.abs(NS[b]-n)) b=i;}); return b;};

/* the challenge: options and the assumed eval */
const CH={Ns:[200,500,1000,4000],Ms:[0,1],Ks:[1,4],pA:0.8,pB:0.82,q:0.12,c:0.2,goal:0.02,start:{n:200,paired:0,k:1}};
const chOf=(n,paired,k)=>({n,paired,k,pA:CH.pA,pB:CH.pB,q:CH.q,c:CH.c});
function chEval(n,paired,k){const r=cmp(chOf(n,paired,k)); return {n,paired,k,h:r.h,ok:r.h<CH.goal,cost:n*k*2,r};}
function chAll(){const a=[]; CH.Ns.forEach(n=>CH.Ms.forEach(m=>CH.Ks.forEach(k=>a.push(chEval(n,m,k))))); return a;}
/* the fewest graded answers among the passing plans (R4.5), then the narrower error bar */
function chBest(){return chAll().filter(e=>e.ok).sort((a,b)=>a.cost-b.cost||a.h-b.h)[0];}

/* =====================================================================
   formatting
   ===================================================================== */
const fint=A.fmt.int;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fPct=p=>fixd(p*100,1)+'%';
const fPt=x=>fixd(x*100,1);                                          // points, one decimal
const fN=n=>fint(n);
/* an interval D ± h as shown, and its ends from the shown values (R6.6); one more decimal where one decimal would show an end at 0.0 */
function ivl(D,h){let d=1; const r=(v,k)=>Math.round(v*Math.pow(10,k))/Math.pow(10,k);
  const lo1=r(D*100,1)-r(h*100,1), hi1=r(D*100,1)+r(h*100,1);
  if(Math.abs(lo1)<0.05||Math.abs(hi1)<0.05) d=2;
  const Ds=sgn(fixd(D*100,d)), hs=fixd(h*100,d), lo=r(D*100,d)-r(h*100,d), hi=r(D*100,d)+r(h*100,d);
  return {Ds,hs,lo:fixd(lo,d).replace('-','−'),hi:fixd(hi,d).replace('-','−'),d};}
const sgn=s=>String(s).replace(/^-/,'−');
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const M=t=>`<div class="evc-fx">${A.ui.math(t)}</div>`;
/* Show the math as a few short blocks (the owner, 2026-10-09: “like a page of a good textbook”): a lead line in words, then the
   quantity's name and formula, the same formula with this page's numbers, the result. A chain breaks at “=”, each “=” under the one above. */
/* a chain whose left column holds a name in words: on a phone the name gets its own line and the “=” signs start the lines under it (H2) */
const EQN=rows=>rows.some(([l])=>/(?<![_^]\{)(?<![_^])\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="evc-eqs${EQN(rows)?' evc-eqs-nm':''}">${rows.map(([l,r])=>`<span class="evc-eq-l">${l?A.ui.math(l):''}</span><span class="evc-eq-e">${A.ui.math('=')}</span><span class="evc-eq-r">${A.ui.math(r)}</span>`).join('')}</div>`;
const MB=(lead,body,res)=>`<div class="evc-mb"><p>${lead}</p>${body}${res?`<p class="evc-mb-res">${res}</p>`:''}</div>`;
const NM=(en,zh)=>`\\text{${tx(en,zh)}}`;
/* a formula on its own line with its name in words on the left; in a narrow column the formula moves under its name instead of running out of its box */
const MN=(en,zh,t)=>`<div class="evc-fx evc-fxn"><span class="evc-fx-nm">${tx(en,zh)}</span>${A.ui.math(t)}</div>`;

/* =====================================================================
   module state
   ===================================================================== */
const BASE={n:164,pA:0.8,pB:0.82,paired:0,q:0.12,k:1,c:0.2};
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let lv='sep', lvTried=new Set(['sep']), remView='pile', saved=null, chSubmitted=false, chReveal=false, sbTouched=false;
let drillSeed=1, drillOpen=false, annT=null, userAct=false, wantObs=false, kSeen=new Set([1]);
let pileM=0, playing=false, spd='fast', raf=0, lastT=0, pileDone=false, relang=false;   // 200 re-runs are a whole run of many tasks: Fast by default, about 4 s (R3.9)
const shownPrev={};
const DUR={slow:30000,mid:15000,fast:4000};

/* =====================================================================
   definitions on hover (R2.8)
   ===================================================================== */
const TIP={
 bench:()=>tx('A benchmark is a fixed set of questions with checkable answers; a model’s score is the share it answers right.','benchmark 是一组固定的、答案可以核对的题；模型的分数就是它答对的比例。'),
 pt:()=>tx('A point here is a percentage point: from 80.0% to 82.0% is 2 points.','这里的 1 分就是 1 个百分点：从 80.0% 到 82.0% 是高了 2 分。'),
 bar:()=>tx('An error bar is the 95% interval drawn around a score. If the model’s true level were exactly this score, a new batch of questions of the same size and difficulty would give a score inside it 95% of the time. A real eval gives only one score, and a bar drawn this way from it covers the model’s true level about 95% of the time. Statisticians call it a 95% confidence interval.',
            '误差线画的是一个分数两边的 95% 区间。假如模型的真实水平正好等于这个分数，换一批同样多、同样难的题再考，新分数有 95% 的机会落在线内。实际评测时只看得到一次分数，按这种方法画出的误差线，大约有 95% 的机会盖住模型的真实水平。统计上把它叫作 95% 置信区间。'),
 se:()=>tx('The standard error is how much the score typically moves from one batch of questions to the next. The error bar reaches 1.96 standard errors either way.',
           '标准误差（standard error）是分数从一批题换到另一批题时通常摆动多少。误差线向两边各伸出 1.96 个标准误差。'),
 temp:()=>tx('At a temperature above 0 the model samples each answer, so the same question can be answered right one time and wrong the next.','temperature 大于 0 时，模型的每次回答都是抽出来的，所以同一道题可能这次答对、下次答错。'),
 coin:()=>tx('In this lab, a coin-flip question is one the model answers right about half the time when it answers again and again; on every other question it is always right or always wrong.',
             '在这个 Lab 里，时对时错的题是指模型反复作答时，大约一半答对、一半答错的题；其余的题它每次都对，或者每次都错。'),
 pool:()=>tx('For each re-run, the app draws 164 new problems, and A gets each one right with chance 80%. The random numbers are all computed from the same fixed starting value, called a seed, so you see the same picture every time you open the page.',
             '每次重考，程序抽 164 道新题，A 每道题有 80% 的机会答对。随机数都由同一个固定的初始值（叫作种子）算出来，所以每次打开看到的都一样。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'one',t:{en:'Within 1 point either way',zh:'上下不到 1 分'}},{v:'three',t:{en:'1 to 3 points either way',zh:'上下 1 到 3 分'}},{v:'wide',t:{en:'More than 3 points either way',zh:'上下 3 分以上'}}];
const PRED_RIGHT='wide';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const nCtl=(id,inline)=>{const h=A.ui.range({id,key:'n',label:tx('Questions in the benchmark','benchmark 的题数'),min:0,max:NS.length-1,value:nIdx(S.n),out:fN(S.n),inline});
  if(!inline) return h;
  const tk=`<div class="evc-ticks" aria-hidden="true">${TICKS.map((t,i)=>`<span class="${i===0?'l':(i===TICKS.length-1?'r':'')}" style="left:${(NS.indexOf(t.n)/(NS.length-1)*100).toFixed(2)}%">${t.t}<br>${t.n===14000?tx('about 14,000','约 14,000'):fN(t.n)}</span>`).join('')}</div>`;
  return h.replace(/<\/div>$/,tk+'</div>');};
const kCtl=(id,inline)=>A.ui.seg({id,key:'k',label:tx('Answers per question','每道题回答几次'),options:KS.map(k=>[String(k),String(k)]),value:String(S.k),inline});
const LV=[{v:'sep',t:{en:'Each on its own (initial setup)',zh:'各算各的（初始设置）'},d:{en:'Work out A’s and B’s error bars separately, then combine them into one for the difference',zh:'先各自算出 A 和 B 的误差线，再合成差值的误差线'}},
  {v:'pair',t:{en:'Question by question',zh:'逐题对比'},d:{en:'For every question, B minus A is +1, 0 or −1; work out the error bar of these 1,319 differences',zh:'每道题看 B 减 A 是 +1、0 还是 −1，再对这 1,319 个差值算误差线'}}];
const P4={n:1319,pA:0.8,pB:0.82,q:0.10,k:1,c:0};
const p4=paired=>cmp({...P4,paired});
const lvRes=v=>{const r=p4(v==='pair'?1:0), i=ivl(r.D,r.h); return tx(`±${i.hs}: ${r.real?'clear of 0':'crosses 0'}`,`±${i.hs} 分，${r.real?'不碰 0':'跨过 0'}`);};
const lvCtl=()=>`<div class="evc-lv" id="evc-ic-lv">`+A.ui.levers({aria:tx('Pick a way to work out the error bar of B − A','选一种算 B − A 误差线的方法'),value:lv,options:LV.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(lvTried.has(o.v)||o.v===lv?lvRes(o.v):'')}))})+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},view:'axis',gate:()=>!!pred,skip:true,
  lead:()=>tx('B beats A by 2 points on 164 coding problems. Is B really better?','B 在 164 道编程题上比 A 高 2 分。B 真的更好吗？'),
  enter:()=>({...BASE}),
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Your product runs model A. A new model, B, scores 82.0% on a coding ${DEF('benchmark',TIP.bench())} of 164 problems, and A scores 80.0%. B is 2 ${DEF('points',TIP.pt())} ahead. Is B really better?</p><p>A benchmark is only one batch of questions out of many like them. Give A another batch of the same size and difficulty, and its score changes too.</p><p>If A took another 164 problems just as hard, how far from 80.0% could its score land?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 2 shows the result.</p>':''}`,
             `<p>你们线上用的是模型 A。新模型 B 在一个有 164 道题的编程 ${DEF('benchmark',TIP.bench())} 上得了 82.0%，A 得了 80.0%，B 比 A 高 2 ${DEF('分',TIP.pt())}。B 真的更好吗？</p><p>一个 benchmark 只是从同类题目里抽出来的一批题。换一批同样多、同样难的题，A 的分数也会变。</p><p>如果让 A 再考另外 164 道同样难的题，它的分数会在 80.0% 上下多大的范围里变动？</p>${ch}${pred?'<p class="muted small">已记下。第 2 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'pile',title:{en:'Where the swing comes from',zh:'分数为什么会摆动'},view:'axis',expect:()=>pileDone,showMe:()=>finishPile(),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Each problem is right or wrong; the score is the average of 164 ones and zeros.','每道题只有对和错，分数就是 164 个 0 和 1 的平均。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>Each problem is either right or wrong, so A’s score is the average of 164 ones and zeros. Take another batch of problems, and the average changes.</p><p>In the chart, the app draws 164 new problems from the same kind again and again, and A takes each batch. Here A’s true level is assumed to be exactly 80%. Each small dot is the score of one ${DEF('re-run',TIP.pool())}.</p>`,
              `<p>每道题只有对、错两种结果，所以 A 的分数就是这 164 个 0 和 1 的平均。换一批题，平均值就会变。</p><p>图里，程序一次又一次从同类题目里重新抽出 164 道题，让 A 再考一遍。这里假设 A 真实的水平正好是 80%。每个小点是一次${DEF('重考',TIP.pool())}的分数。</p>`),
  obs:()=>obsPile()},
 {id:'size',title:{en:'More questions, less swing',zh:'题多一点，摆动小一点'},view:'axis',expect:()=>cmp(S).hA<=0.01+1e-9,showMe:{n:6200},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Four times the questions only halve the error bar.','题目多 4 倍，误差线才短一半。'),
  enter:()=>({...BASE}),
  body:()=>tx(`<p>The more questions, the shorter the error bars, but they shrink slowly: four times the questions only halve them.</p><p>Drag the number of questions until A’s error bar is within ±1 point. The ticks mark three well-known benchmarks.</p>`,
              `<p>题目越多，误差线越短，但缩得很慢：题目多 4 倍，误差线才短一半。</p><p>拖动题数，把 A 的误差线压到 ±1 分以内。刻度上标出了三个常见 benchmark 的题数。</p>`)+nCtl('evc-ic-n',true),
  obs:()=>obsSize()},
 {id:'pair',title:{en:'Same questions, one by one',zh:'同一批题，逐题对比'},view:'pair',expect:()=>lvTried.has('pair'),showMe:()=>setLv('pair'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Questions both models get right, or both get wrong, cannot tell them apart.','两个都对或两个都错的题，分不出谁更好。'),
  enter:()=>{lv='sep'; lvTried=new Set(['sep']); return {...P4,paired:0};},
  body:()=>{const r=p4(0);
   return tx(`<p>Now a math benchmark of 1,319 problems (as many as GSM8K’s test set). A scores 80.0% and B 82.0%; their error bars, about ±${fPt(r.hA)} points each, still overlap.</p><p>A and B answered the same problems. A problem both got right, or both got wrong, says nothing about which is better. What tells them apart are the questions only one model gets right.</p><p>There are two ways to work out the error bar of B − A. Pick one and see whether the error bar crosses 0.</p>`,
             `<p>现在换成一个有 1,319 道题的数学 benchmark（和 GSM8K 的测试集一样多）。A 得 80.0%，B 得 82.0%，两条误差线各约 ±${fPt(r.hA)} 分，还是有重叠。</p><p>A 和 B 做的是同一批题。两个都对或两个都错的题，分不出谁更好。能分出高低的，只有一个答对、一个答错的题，下面简称“一对一错的题”。</p><p>B − A 的误差线有两种算法。选一种，看看误差线会不会跨过 0。</p>`)+lvCtl();},
  obs:()=>obsPair()},
 {id:'k',title:{en:'Several answers per question',zh:'每道题多答几次'},view:'k',expect:()=>kSeen.has(16),showMe:{k:16},label:{en:'Result',zh:'结果'},
  lead:()=>tx('Averaging answers removes each answer’s own luck, not the luck of the draw.','多答几次只能消掉回答自己的随机性，消不掉换题带来的摆动。'),
  enter:()=>{kSeen=new Set([1]); return {...BASE};},
  body:()=>tx(`<p>Back to the 164 coding problems. In production the model answers at ${DEF('temperature 0.7',TIP.temp())} (see ${labLink('sampling')}), so the same problem can get a different answer each time.</p><p>Answering each problem several times and averaging removes only the luck of each answer. The swing from which problems happened to be drawn stays, however many times you answer.</p>`,
              `<p>回到 164 道编程题。线上产品让模型用 ${DEF('temperature 0.7',TIP.temp())} 作答（见${labLink('sampling')}），所以同一道题每次答得可能不一样。</p><p>把同一道题多答几次再取平均，能消掉的只是每次回答自己的随机性。“这次抽到了哪些题”带来的摆动，答多少遍也消不掉。</p>`)+kCtl('evc-ic-k',true),
  obs:()=>obsK()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},view:()=>({pile:'axis',size:'axis',pair:'pair',k:'k'}[remView]||'axis'),
  lead:()=>tx('Count the questions first; compare on the same questions; more answers do not replace more questions.','先看题数；同一批题逐题对比；多答几次代替不了加题。'),
  enter:()=>{remView='pile'; pileM=RD.K; return {...BASE};},
  body:()=>remember()}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
const viewOf=s=>typeof s.view==='function'?s.view():s.view;
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
function obsPile(){
  if(!pileDone) return '';
  const m=inside(RD.K), lo=fPct(RD.p-RD_H), hi=fPct(RD.p+RD_H), h=fPt(RD_H);
  return tx(`<p><b>${m} of the ${RD.K} re-runs land between ${lo} and ${hi}: 80.0% give or take ${h} points.</b></p>`,`<p><b>${RD.K} 次重考里，有 ${m} 次落在 ${lo} 到 ${hi} 之间，也就是 80.0% 上下 ${h} 分。</b></p>`)+verdict()+
    tx(`<p>B’s 82.0% is inside that range, so 164 problems cannot tell A and B apart. Drawn on the axis, the range is A’s ${DEF('error bar',TIP.bar())}.</p><p>No re-runs are needed to find it; a formula gives it from the score p and the number of questions n:</p>`,
       `<p>B 的 82.0% 就在这个范围里，所以只凭 164 道题，分不出 A 和 B 谁更好。这个范围画在分数轴上，就是 A 的${DEF('误差线',TIP.bar())}。</p><p>不用真的重考，用分数 p 和题数 n 就能算出它：</p>`)+
    MN('standard error','标准误差','\\text{SE} = \\sqrt{\\frac{p(1-p)}{n}}')+M(tx('\\text{error bar} = p ± 1.96 \\times \\text{SE}','\\text{误差线} = p ± 1.96 \\times \\text{SE}'))+
    tx(`<p>SE is the ${DEF('standard error',TIP.se())}: how much the score typically moves from one batch of questions to the next. The error bar reaches 1.96 SEs either side of the score. If the score equals the true level (here assumed to be 80%), the bar covers 95% of the re-run scores.</p><p>With p = 0.8 and n = 164, SE is about ${fPt(RD_H/Z)} points and the error bar is ±${h} points, the same range the re-runs showed.</p>`,
       `<p>SE 是${DEF('标准误差',TIP.se())}，表示换一批题时，分数通常会摆动多少。从分数往两边各伸出 1.96 个 SE，就是误差线。如果分数正好等于真实水平（这里假设是 80%），这条线能盖住 95% 的重考分数。</p><p>代入 p = 0.8、n = 164，SE 约是 ${fPt(RD_H/Z)} 分，误差线是 ±${h} 分，和上面数出来的范围一样。</p>`);
}
function obsSize(){
  const r=cmp(S), h=fPt(r.hA), h0=fPt(cmp(BASE).hA), need=nFor(0.8,0.01);
  if(S.n===164) return tx(`<p><b>With 164 questions, A’s error bar is ±${h} points.</b> Drag to the right.</p>`,`<p><b>164 道题时，A 的误差线是 ±${h} 分。</b>往右拖。</p>`);
  let s=tx(`<p><b>With ${fN(S.n)} questions, A’s error bar is ±${h} points</b> (±${h0} with 164).</p>`,`<p><b>${fN(S.n)} 道题时，A 的误差线是 ±${h} 分</b>（164 道题时是 ±${h0} 分）。</p>`);
  if(r.hA<=0.01+1e-9) s+=tx(`<p><b>To get within ±1 point you need about ${fN(Math.round(need/100)*100)} questions.</b></p>`,`<p><b>要把误差线压到 ±1 分以内，大约要 ${fN(Math.round(need/100)*100)} 道题。</b></p>`);
  else if(S.n<164) s+=tx('<p>Fewer questions, a longer error bar. Drag to the right.</p>','<p>题更少，误差线更长。往右拖。</p>');
  s+=tx('<p>This is because the score is an average over many questions, and an average’s swing is inversely proportional to the square root of their number. So four times the questions only halve the error bar.</p><p>Turned around, the formula gives the number of questions needed to bring the error bar down to ±h:</p>',
        '<p>这是因为分数是很多道题的平均，而平均值的摆动和题数的平方根成反比。所以题数变成 4 倍，误差线才缩短一半。</p><p>把公式倒过来，就能算出要把误差线压到 ±h，需要多少道题：</p>')+
    MN('questions needed','需要的题数','n = \\frac{1.96^2 \\times p(1-p)}{h^2}');
  return s;
}
function obsPair(){
  const r=p4(lv==='pair'?1:0), i=ivl(r.D,r.h);
  if(lv==='sep') return tx(`<p><b>Each on its own, B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}. The error bar crosses 0, so this cannot tell which is better.</b></p><p>Now try question by question.</p>`,
                           `<p><b>各算各的时，B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}。误差线跨过了 0，分不出谁更好。</b></p><p>再试试逐题对比。</p>`);
  const s=p4(0), j=ivl(s.D,s.h);
  return tx(`<p><b>Question by question, B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}. The whole error bar is right of 0, so B’s lead is unlikely to be noise.</b> From the same answers, each on its own gives an error bar of ±${j.hs} points.</p><p>This is because the problems both got right or both got wrong (90%) do not move the difference, so their swing cancels when you subtract. What is left comes only from the 10% that only one model got right:</p>`,
            `<p><b>逐题对比时，B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}。整条误差线都在 0 的右边，B 领先不太可能只是噪声。</b>同一份答卷如果各算各的，误差线是 ±${j.hs} 分。</p><p>这是因为两个都对、两个都错的题（共 90%）不影响差值，它们带来的摆动在相减时抵消了。剩下的摆动只来自那 10% 一对一错的题：</p>`)+
    MN('SE of B − A','B − A 的标准误差','\\text{SE} = \\sqrt{\\frac{q - Δ^2}{n}}')+
    tx('<p>q is the share of problems only one model got right, and Δ is B’s score minus A’s.</p>','<p>q 是一对一错的题的占比，Δ 是 B 的分数减去 A 的分数。</p>');
}
function obsK(){
  const r=cmp(S), h=fPt(r.hA), h1=fPt(cmp({...S,k:1}).hA), fl=fPt(floorH(S.pA,S.c,S.n));
  if(S.k===1) return tx(`<p><b>One answer per problem: A’s error bar is ±${h} points.</b> Raise the number of answers.</p>`,`<p><b>每道题答 1 次：A 的误差线是 ±${h} 分。</b>把回答次数调大试试。</p>`);
  let s=tx(`<p><b>${S.k} answers per problem, averaged: A’s error bar shrinks from ±${h1} to ±${h} points.</b></p>`,`<p><b>每道题答 ${S.k} 次再平均：A 的误差线从 ±${h1} 分缩到 ±${h} 分。</b></p>`);
  if(S.k===16) s+=tx(`<p><b>However many answers, it never goes below ±${fl} points.</b></p>`,`<p><b>答再多次，也低不过 ±${fl} 分。</b></p>`);
  s+=tx(`<p>Why: in this lab, A gets 80% of the problems always right or always wrong, and more answers change nothing there. Only the 20% ${DEF('coin-flip problems',TIP.coin())} get steadier. The ±${fl} points left come from which problems were drawn; only more problems shrink it.</p>`,
        `<p>原因：在这个 Lab 里，A 有 80% 的题每次都对或每次都错，多答几次结果也一样；只有那 20% ${DEF('时对时错的题',TIP.coin())}，平均之后才更稳。剩下的 ±${fl} 分来自“抽到了哪些题”，只有加题才能缩小。</p>`)+
    MN('A’s standard error squared','A 的标准误差的平方','\\text{SE}^2 = \\frac{\\text{Var}(p_i)}{n} + \\frac{\\text{mean}(p_i(1 - p_i))}{n k}')+
    tx(`<p>${A.ui.math('p_i')} is A’s chance on problem i, and k is the number of answers per problem. The first term comes from problems differing in difficulty and does not change with k. The second is each answer’s luck, and it shrinks as k grows.</p>`,
       `<p>${A.ui.math('p_i')} 是 A 答对第 i 道题的机会，k 是每道题回答的次数。第一项来自题和题的难度不同，k 再大也不变。第二项是每次回答的运气，k 越大，它越小。</p>`);
  return s;
}
const REMS=['pile','size','pair','k'];
function remBtn(k){const on=remView===k;
  const tip={pile:tx('Shows the re-runs on 164 problems and A’s error bar.','在图里显示 164 道题的重考和 A 的误差线。'),size:tx('Shows the error bars at 6,200 questions.','在图里显示 6,200 道题时的误差线。'),
    pair:tx('Shows the two models’ answer sheet on 1,319 problems and both error bars of B − A.','在图里显示两个模型做 1,319 道题的答卷，和 B − A 的两种误差线。'),k:tx('Shows A’s error bar at 16 answers per problem, with its floor.','在图里显示每道题答 16 次时 A 的误差线和它的下限。')}[k];
  return ` <button type="button" class="linkbtn evc-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b=cmp(BASE), sz=cmp({...BASE,n:6200}), s=p4(0), p=p4(1), k16=cmp({...BASE,k:16}), big=cmp({...BASE,n:14000});
  const need=fN(Math.round(nFor(0.8,0.01)/100)*100);
  return tx(`<ul class="evc-remlist"><li><b>Count the questions before you trust a gap.</b> With a few hundred questions, the error bar spans several points, so a lead of a few points can be pure luck. Work out the error bar before you switch models.${remBtn('pile')}</li>`+
    `<li><b>Four times the questions halve the error bar.</b> To see a gap half as large, you need four times the questions. So fix the smallest gap you must see first, then size the eval for it.${remBtn('size')}</li>`+
    `<li><b>When both models answer the same questions, compare them question by question.</b> Questions both get right, or both get wrong, cancel when you subtract. So the error bar of B − A is much shorter, and the same questions can show a smaller gap.${remBtn('pair')}</li>`+
    `<li><b>More answers per question do not replace more questions.</b> More answers only remove each answer’s own luck. The swing from which questions were drawn stays, so only more questions shorten the error bar a lot.${remBtn('k')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Two error bars that overlap mean no difference.” Two separate error bars ignore that both models answered the same questions. Question by question, the error bar of B − A is much shorter, and it can stay clear of 0 while the two bars overlap.</p><p>✗ “Run the same 164 problems several times and average, and the score becomes exact.” Averaging removes only each answer’s own luck. The draw of problems stays, so the error bar shrinks only a little.</p><p>✗ “The size of the gap alone tells you whether to trust it.” The number of questions matters too. The same 2 points cannot be told from noise on 164 problems. On about 14,000, each error bar is only about ±${fPt(big.hA)} points, and the 2 points are clearly beyond noise.</p></div>${A.shell.toScenarios()}`,
    `<ul class="evc-remlist"><li><b>先看题有多少道，再信分数差。</b>只有一两百道题时，误差线有好几分长，领先几分很可能只是运气。换模型之前，先算出误差线。${remBtn('pile')}</li>`+
    `<li><b>题目多 4 倍，误差线才短一半。</b>想看清的差距每小一半，题目就得多 4 倍。所以先定下必须看清的最小差距，再按它准备题目。${remBtn('size')}</li>`+
    `<li><b>两个模型做的是同一批题时，要逐题对比。</b>两个都对、两个都错的题，相减时抵消了。所以 B − A 的误差线短得多，同一批题能看清更小的差距。${remBtn('pair')}</li>`+
    `<li><b>多答几次代替不了加题。</b>多答几次只消掉每次回答的运气。抽到哪些题带来的摆动还在，所以想让误差线明显变短，只能加题。${remBtn('k')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “两条误差线重叠，就说明没有差别。”各算各的两条误差线，没有用上两个模型做的是同一批题。逐题对比时，B − A 的误差线短得多。两条误差线重叠时，它也可能不碰 0。</p><p>✗ “同一批 164 道题多跑几遍取平均，分数就准了。”取平均只消掉每次回答的运气。抽到哪些题这件事没有变，所以误差线只短一点。</p><p>✗ “只看差了几分，就能判断可不可信。”还要看题数。同样差 2 分，164 道题时分不出是不是噪声。约 14,000 道题时，每条误差线只有 ±${fPt(big.hA)} 分左右，这 2 分就明显超出了噪声。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios
   ===================================================================== */
const SRC={
 he:'https://github.com/openai/human-eval',
 gsm:'https://github.com/openai/grade-school-math',
 mmlu:'https://arxiv.org/abs/2009.03300',
 mmlucat:'https://github.com/hendrycks/test/blob/master/categories.py',
 lmeval:'https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py',
 miller:'https://arxiv.org/abs/2411.00640',
 codex:'https://arxiv.org/abs/2107.03374',
 gsmp:'https://arxiv.org/abs/2110.14168'
};
const PRESETS=[
 {id:'code',t:{en:'Two code models on 164 problems',zh:'两个代码模型比 164 道编程题'},d:{en:'A benchmark the size of HumanEval; B is 2 points ahead',zh:'和 HumanEval 一样多的题，B 高 2 分'},s:{n:164,pA:0.8,pB:0.82,paired:1,q:0.12,k:1,c:0.2}},
 {id:'exam',t:{en:'A multiple-choice exam of about 14,000 questions',zh:'约 14,000 道题的选择题考试'},d:{en:'As large as MMLU; B is under 1 point ahead',zh:'和 MMLU 差不多大，B 高不到 1 分'},s:{n:14000,pA:0.8,pB:0.808,paired:1,q:0.10,k:1,c:0}},
 {id:'team',t:{en:'Your team’s internal eval of 200 prompts',zh:'你们团队 200 条 prompt 的内部评测'},d:{en:'A hand-written set; B is 2 points ahead',zh:'自己写的评测集，B 高 2 分'},s:{n:200,pA:0.8,pB:0.82,paired:1,q:0.12,k:1,c:0.2}},
 {id:'temp',t:{en:'1,319 math problems at temperature 0.7',zh:'1,319 道数学题，用 temperature 0.7 作答'},d:{en:'The average of 4 answers per problem instead of 1',zh:'每道题答 4 次取平均，而不是只答 1 次'},s:{n:1319,pA:0.8,pB:0.82,paired:1,q:0.12,k:4,c:0.2}}
];
const same=(a,b)=>Math.abs(a-b)<1e-9;
const presetNow=()=>PRESETS.findIndex(p=>['n','pA','pB','paired','q','k','c'].every(k=>same(p.s[k],S[k])));
function scenHTML(i){
  const p=PRESETS[i], r=cmp(S), iv=ivl(r.D,r.h);
  if(p.id==='code') return tx(`<p><b>On 164 problems, even question by question, B − A = ${iv.Ds} ± ${iv.hs} points, and the error bar crosses 0. The 2 points cannot be told from noise.</b></p><p>HumanEval, a common coding benchmark, has exactly 164 problems (${cite(SRC.he,'openai/human-eval')}). When two models are a few points apart on HumanEval, that gap alone says little about which is better.</p>`,
                              `<p><b>164 道题时，就算逐题对比，B − A = ${iv.Ds} ± ${iv.hs} 分，误差线也跨过了 0。这 2 分和噪声分不开。</b></p><p>常用的编程 benchmark HumanEval 正好有 164 道题（${cite(SRC.he,'openai/human-eval')}）。排行榜上两个模型在 HumanEval 上差几分，单凭这一点说明不了谁更好。</p>`);
  if(p.id==='exam'){const s=cmp({...S,paired:0}), js=ivl(s.D,s.h);
    return tx(`<p><b>With about 14,000 questions, B − A question by question is ${iv.Ds} ± ${iv.hs} points, clear of 0, so the ${iv.Ds} points are beyond noise.</b></p><p>Each on its own would give ${js.Ds} ± ${js.hs}, which still crosses 0. MMLU, a multiple-choice exam of 57 subjects, has about 14,000 test questions (${cite(SRC.mmlu,'Hendrycks et al., 2021')}). This lab assumes 10% of them are right for only one of the two models.</p>`,
              `<p><b>约 14,000 道题时，逐题对比的 B − A 是 ${iv.Ds} ± ${iv.hs} 分，不碰 0，所以这 ${iv.Ds} 分超出了噪声。</b></p><p>各算各的会得到 ${js.Ds} ± ${js.hs} 分，还是跨过 0。MMLU 是一套 57 个科目的选择题考试，测试集大约有 14,000 道题（${cite(SRC.mmlu,'Hendrycks 等，2021')}）。这个 Lab 假设其中 10% 的题一对一错。</p>`);}
  if(p.id==='team') return tx(`<p><b>On 200 prompts, one score’s error bar is about ±${fPt(r.hA)} points. Even question by question, the error bar of B − A is ±${iv.hs} points and crosses 0, so the 2-point lead says nothing yet.</b></p><p>A hand-written eval of a few hundred prompts is common. If a model grades the answers, the grader adds noise of its own on top (${labLink('llm-judge')}).</p>`,
                              `<p><b>200 条 prompt 时，一个分数的误差线约 ±${fPt(r.hA)} 分。就算逐题对比，B − A 的误差线也有 ±${iv.hs} 分，跨过了 0，领先的 2 分还说明不了什么。</b></p><p>自己写几百条 prompt 的评测很常见。如果再让一个模型来打分，打分本身还会带来额外的噪声（${labLink('llm-judge')}）。</p>`);
  const r1=cmp({...S,k:1}), fl=fPt(floorH(S.pA,S.c,S.n));
  return tx(`<p><b>With 4 answers per problem averaged, A’s error bar only shrinks from ±${fPt(r1.hA)} to ±${fPt(r.hA)} points, and no number of answers takes it below ±${fl}. Question by question, B − A = ${iv.Ds} ± ${iv.hs} points, clear of 0.</b></p><p>GSM8K’s test set has 1,319 grade-school math problems (${cite(SRC.gsm,'openai/grade-school-math')}). This lab assumes 20% of them are coin-flip problems at temperature 0.7.</p>`,
            `<p><b>每道题答 4 次再平均，A 的误差线只从 ±${fPt(r1.hA)} 分缩到 ±${fPt(r.hA)} 分，答再多次也低不过 ±${fl} 分。逐题对比时，B − A = ${iv.Ds} ± ${iv.hs} 分，不碰 0。</b></p><p>GSM8K 的测试集有 1,319 道小学数学题（${cite(SRC.gsm,'openai/grade-school-math')}）。这个 Lab 假设在 temperature 0.7 下，其中 20% 的题时对时错。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
/* the goal (half-width under 2) only guarantees that a measured 2-point gap clears 0; say how often a true 2-point gain would (S-1) */
const pctOf=pw=>pw>0.995?tx('over 99%','99% 以上'):tx(`about ${Math.round(pw*100)}%`,`约 ${Math.round(pw*100)}%`);
function powerNote(h){const pw=power(h,CH.pB-CH.pA), p=pctOf(pw);
  return tx('<p class="small">The goal only guarantees that a measured gap of exactly 2 points keeps the error bar clear of 0. If B really is 2 points better, the measured gap still swings up and down, and when it comes out low, the error bar touches 0.</p>',
            '<p class="small">这个目标只保证测出的差值正好是 2 分时，误差线不碰 0。B 真的高 2 分时，测出的差值也会上下摆动，偏低时误差线就会碰到 0。</p>')+
    (pw<0.8?tx(`<p class="small">With this plan, if B really is 2 points better, the error bar stays clear of 0 only ${p} of the time; statisticians call this chance the power. The usual standard is 80%, which takes an error bar of about ±${fPt(H80(CH.pB-CH.pA))} points (see ${J('prod','“In production”')}).</p>`,
               `<p class="small">按这个方案，B 真的高 2 分时，误差线不碰 0 的机会只有${p}，统计上叫检验力。常用的标准是 80%，这要把误差线压到约 ±${fPt(H80(CH.pB-CH.pA))} 分，见${J('prod','“生产中怎么做”')}。</p>`)
           :tx(`<p class="small">With this plan, if B really is 2 points better, the error bar stays clear of 0 ${p} of the time; statisticians call this chance the power. That is above the usual standard of 80%.</p>`,
               `<p class="small">按这个方案，B 真的高 2 分时，误差线不碰 0 的机会${p}，统计上叫检验力。它超过了常用的标准 80%。</p>`));}
const chName=(n,paired,k)=>tx(`${fN(n)} questions, ${paired?'question by question':'each on its own'}, ${k===1?'1 answer':`${k} answers`} each`,`${fN(n)} 道题，${paired?'逐题对比':'各算各的'}，每题答 ${k} 次`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(S.n,S.paired,S.k), best=chBest(), i=ivl(me.r.D,me.r.h);
  let h;
  const good=tx(`<p class="okline evc-mt"><b>Goal met: B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}, clear of 0. It needs ${fN(me.cost)} graded answers.</b>`,`<p class="okline evc-mt"><b>达成目标：B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}，不碰 0。要打分的回答是 ${fN(me.cost)} 条。</b>`);
  if(me.ok&&me.cost===best.cost&&me.n===best.n&&me.paired===best.paired&&me.k===best.k) h=good+tx(' And no plan that meets the goal needs fewer.</p>','而且达标的方案里，这已经是最少的。</p>')+powerNote(me.h);
  else if(me.ok) h=good+tx(' But a cheaper plan also works.</p>','不过还有更省的方案。</p>')+tx(`<p>The cheapest is “${esc(chName(best.n,best.paired,best.k))}”, with only ${fN(best.cost)} graded answers.</p>`,`<p>最省的方案是“${esc(chName(best.n,best.paired,best.k))}”，只要打分 ${fN(best.cost)} 条。</p>`)+powerNote(me.h);
  else h=tx(`<p class="badline evc-mt"><b>Not yet: B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}, which still includes 0.</b></p>`,`<p class="badline evc-mt"><b>还没达到：B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}，还包含 0。</b></p>`)+
      tx('<p class="small">Hint: the swing has two sources. One is which questions were drawn. Both models answer the same questions, so comparing question by question cancels most of it. The other is each answer’s own luck at temperature 0.7, which averaging several answers shrinks. Once the first is gone, which one is mostly left?</p>',
         '<p class="small">提示：摆动有两个来源。一个来源是抽到了哪些题。两个模型做的是同一批题，逐题对比能把这部分摆动大多抵消。另一个来源是 temperature 0.7 下每次回答自己的随机性，多答几次取平均能把它压小。第一种抵消之后，剩下的主要是哪一种？</p>')+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the plans that meet the goal, the pick and the initial setup (brief) */
  const isMe=e=>e.n===S.n&&e.paired===S.paired&&e.k===S.k, isStart=e=>e.n===CH.start.n&&e.paired===CH.start.paired&&e.k===CH.start.k;
  const rows=chAll(), shown=rows.filter(e=>e.ok||isMe(e)||isStart(e)), st=chEval(CH.start.n,CH.start.paired,CH.start.k);
  h+=A.ui.cmpChart({title:tx('How wide is the error bar of B − A? (In points. The goal is shorter than ±2, so that a measured 2-point gap keeps it clear of 0.)','B − A 的误差线有多宽？（单位：分。目标是短于 ±2 分，这样测出 2 分的差距时，误差线不碰 0。）'),
    aria:tx('Error bar of B − A: ','B − A 的误差线：')+shown.map(e=>`${chName(e.n,e.paired,e.k)}${tx(': ±','：±')}${fPt(e.h)}${e.ok?tx(', meets the goal','，达标'):''}`).join(tx('; ','；')),
    rows:rows.map(e=>({label:(e.ok?'✓ ':'')+chName(e.n,e.paired,e.k)+tx(`: ${fN(e.cost)} graded answers`,`：打分 ${fN(e.cost)} 条`),value:e.h*100,text:'±'+fPt(e.h),ok:e.ok,
      me:isMe(e),tags:(isStart(e)?[{text:tx('initial setup','初始设置')}]:[]).concat(isMe(e)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,goal:2,goalLabel:tx('goal: under ±2','目标：小于 ±2'),max:8,okText:tx('✓ meets the goal','✓ 达标'),noText:tx('misses it','没达标'),goalText:tx('goal: under ±2','目标：小于 ±2')});
  /* the two single changes of the cheapest plan miss the goal and are not in the chart: name them, their error bars in a hover */
  const b=chEval(best.n,0,best.k), s1=chEval(best.n,best.paired,1);
  h+=tx(`<p class="small">The initial setup (200 questions, each on its own, 1 answer) gives ±${fPt(st.h)} points.</p><p class="small">“${esc(chName(best.n,best.paired,best.k))}” is the cheapest. Question by question, the swing from which questions were drawn mostly cancels, so most of what is left is each answer’s own luck. Averaging 4 answers brings that part down.</p>`,
        `<p class="small">初始设置（200 道题，各算各的，每题答 1 次）的误差线是 ±${fPt(st.h)} 分。</p><p class="small">“${esc(chName(best.n,best.paired,best.k))}”最省。逐题对比时，抽到哪些题带来的摆动大部分抵消了，剩下的主要是每次回答自己的随机性。答 4 次取平均，正好能把这部分压下去。</p>`);
  if(!b.ok&&!s1.ok&&best.paired&&best.k>1) h+=tx(`<p class="small">With the same ${fN(best.n)} questions, changing only one of the other two choices misses the goal. Each on its own, the error bar stays ${DEF('wider than ±2 points',`${chName(b.n,b.paired,b.k)}: ±${fPt(b.h)} points.`)} even with ${best.k} answers each. Question by question with only 1 answer each, it is ${DEF('also wider than ±2 points',`${chName(s1.n,s1.paired,s1.k)}: ±${fPt(s1.h)} points.`)}.</p>`,
        `<p class="small">同样是 ${fN(best.n)} 道题，比较方法和回答次数只改一项，都达不到目标。各算各的，每题答 ${best.k} 次，误差线也${DEF('宽于 ±2 分',`${chName(b.n,b.paired,b.k)}，误差线是 ±${fPt(b.h)} 分。`)}。逐题对比、每题只答 1 次，误差线${DEF('同样宽于 ±2 分',`${chName(s1.n,s1.paired,s1.k)}，误差线是 ±${fPt(s1.h)} 分。`)}。</p>`);
  h+=tx('<p class="small muted">Note: this assumes 20% coin-flip questions. If your models answer at temperature 0, every answer to a question is the same, more answers help nothing, and only more questions shrink the error bar. Before you size the eval, count on your own two models how many questions only one of them gets right.</p>',
        '<p class="small muted">注意：这里假设有 20% 时对时错的题。如果你们的模型用 temperature 0 作答，同一道题每次的回答都一样，多答几次没有用，只能加题。确定评测规模之前，先在自己的模型上数一数，两个模型有多少题一对一错。</p>');
  return h;
}

/* =====================================================================
   header, fidelity, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx(`${STEPS.length===6?'Six':fint(STEPS.length)} short parts. Predict first, then see where a score’s swing comes from, how many questions it takes to shrink it, why comparing on the same questions is sharper, and what answering each question again cannot fix. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看分数的摆动从哪来、要多少道题才能压小它、为什么在同一批题上逐题对比更灵敏，以及每道题多答几次解决不了什么。第一次学从这里开始。`),
  sandbox:tx('Four real situations, from a coding benchmark of 164 problems to an exam of about 14,000 questions. Pick one to see whether its gap is more than noise.',
             '四个真实情况，从 164 道题的编程 benchmark 到约 14,000 道题的考试。选一个，看它的分数差是不是超出了噪声。'),
  challenge:tx('B scores 2 points above A on your internal eval. Choose the number of questions, the way to compare and the answers per question so that the error bar of B − A is shorter than ±2 points; then a measured 2-point gap keeps it clear of 0. Use as few graded answers as you can.',
               '内部评测上 B 比 A 高 2 分。选好题数、比较方法和每道题的回答次数，让 B − A 的误差线短于 ±2 分，这样测出 2 分的差距时，误差线不碰 0。同时，要打分的回答越少越好。')
});
const FIDPOP=()=>tx(
 `<p><b>Formula:</b> every error bar on this page is computed from a formula; ${J('math','“Show the math”')} works them out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>A score is the average of n right-or-wrong answers, and its error bar comes from the normal approximation (the formula is in part 2). For scores between 50% and 90% and at least 164 questions, the approximation is close enough: a bar drawn this way covers the true level at least about ${Math.floor(COV.min*100)}% of the time. For scores near 100% on few questions, the bar comes out too narrow.</li>`+
 `<li>When two models answer the same questions, the standard error of the difference depends only on the questions only one model gets right; the formula is in part 4.</li>`+
 `<li>Several answers per question: averaging k answers removes the part of the swing that comes from each answer, not the part from which questions were drawn.</li>`+
 `<li>Part 2’s re-runs are drawn with a seeded generator (mulberry32, seed 7): a picture of the formula, not a recording.</li></ul></div>`+
 `<div><div class="eyebrow">Assumed</div><ul>`+
 `<li>The questions are drawn independently from a large pool of similar ones. A real benchmark is often written in groups (several questions on one passage); then the true swing is larger.</li>`+
 `<li>Part 4’s answer sheet (76% both right, 4% only A, 6% only B, 14% both wrong) is this lab’s assumption, not a recorded run.</li>`+
 `<li>Coin-flip questions: in part 5 and the challenge, 20% of the questions are answered right half the time at temperature 0.7, and the rest always right or always wrong. Real models vary; measure yours.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>164 problems: HumanEval’s <code>data/HumanEval.jsonl.gz</code> has 164 lines (${cite(SRC.he,'openai/human-eval')}). 1,319 problems: GSM8K’s <code>grade_school_math/data/test.jsonl</code> has 1,319 lines (${cite(SRC.gsm,'openai/grade-school-math')}). About 14,000: MMLU’s test set (${cite(SRC.mmlu,'Hendrycks et al., arXiv 2009.03300')}; the exact count was not checked). Its 57 subjects are listed in ${cite(SRC.mmlucat,'hendrycks/test <code>categories.py</code>')}.</li>`+
 `<li>Standard errors in practice: EleutherAI lm-evaluation-harness, ${cite(SRC.lmeval,'<code>lm_eval/api/metrics.py</code>')}.</li>`+
 `<li>Error bars for evals: ${cite(SRC.miller,'Miller, arXiv 2411.00640')}.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-03. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>公式计算：</b>这一页上的每条误差线都由公式算出；${J('math','“展开公式”')}里有逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>分数是 n 个对错结果的平均，误差线用正态近似算出（公式见第 2 部分）。分数在 50% 到 90% 之间、题数不少于 164 时，这个近似够准，画出的误差线至少有约 ${Math.floor(COV.min*100)}% 的机会盖住真实水平。分数接近 100%、题数又少时，误差线会画得偏窄。</li>`+
 `<li>两个模型做同一批题时，差值的标准误差只取决于一对一错的题，公式在第 4 部分。</li>`+
 `<li>每道题多答几次：k 次回答取平均，消掉的是每次回答带来的那部分摆动，不是抽到哪些题带来的那部分。</li>`+
 `<li>第 2 部分的重考用带种子的随机数生成器（mulberry32，种子 7）画出，是公式的示意，不是真实记录。</li></ul></div>`+
 `<div><div class="eyebrow">假设</div><ul>`+
 `<li>题目是从一大堆同类题里各自独立抽出来的。真实的 benchmark 常常成组出题（一篇材料配几道题），那样真实的摆动会更大。</li>`+
 `<li>第 4 部分的答卷（两个都对 76%，只有 A 对 4%，只有 B 对 6%，两个都错 14%）是这个 Lab 的假设，不是真实记录。</li>`+
 `<li>时对时错的题：第 5 部分和挑战题假设在 temperature 0.7 下，20% 的题一半答对、一半答错，其余的题每次都对或每次都错。真实模型各不相同，要在自己的模型上量。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>164 道题：HumanEval 的 <code>data/HumanEval.jsonl.gz</code> 有 164 行（${cite(SRC.he,'openai/human-eval')}）。1,319 道题：GSM8K 的 <code>grade_school_math/data/test.jsonl</code> 有 1,319 行（${cite(SRC.gsm,'openai/grade-school-math')}）。约 14,000 道：MMLU 的测试集（${cite(SRC.mmlu,'Hendrycks 等，arXiv 2009.03300')}；准确题数没有核对）。57 个科目列在 ${cite(SRC.mmlucat,'hendrycks/test 的 <code>categories.py</code>')} 里。</li>`+
 `<li>实际怎么算标准误差：EleutherAI lm-evaluation-harness 的 ${cite(SRC.lmeval,'<code>lm_eval/api/metrics.py</code>')}。</li>`+
 `<li>给评测加误差线：${cite(SRC.miller,'Miller，arXiv 2411.00640')}。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-03。原型，尚未经专家审阅。</p></div></div>`);
const OUTCOME=()=>tx('tell whether a score gap is noise, estimate how many questions an eval needs, and compare two models question by question on the same questions.','判断一个分数差是不是噪声，估算评测需要多少道题，并学会在同一批题上逐题比较两个模型。');
const EXITS=()=>{
  const x5=cmp({...BASE,n:500,pB:0.81}), r164=cmp(BASE), r1319=cmp({...BASE,n:1319});
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Why is the error bar of B − A shorter when you compare question by question?','为什么逐题对比时，B − A 的误差线更短？'),
   o:[tx('Questions both models get right, or both get wrong, do not move the difference, so the swing from which questions were drawn cancels, and what is left comes from the questions only one model gets right.','两个模型都对或都错的题不影响差值，所以抽到哪些题带来的摆动抵消了，只剩下一对一错的那些题。'),
      tx('Comparing question by question counts every question twice, once for each model, so it works from twice as many questions.','逐题对比把每道题算了两次，每个模型一次，所以等于用了两倍的题。'),
      tx('Comparing question by question averages the two models’ scores first, and an average of two scores swings less than either one.','逐题对比先把两个模型的分数取平均，两个分数的平均值比其中任何一个摆动都小。')],a:0,
   n:[null,tx('Both ways use the same answers to the same n questions. Question by question uses one difference per question, n numbers, not 2n.','两种算法用的是同一份答卷、同样的 n 道题。逐题对比是每道题一个差值，一共 n 个数，不是 2n 个。'),
     tx('Nothing is averaged across the two models: question by question looks at B minus A on each question. Its error bar is shorter because shared ups and downs cancel in the subtraction.','逐题对比没有把两个模型取平均，它看的是每道题上 B 减 A。误差线更短，是因为两个模型共同的起伏在相减时抵消了。')],
   why:tx('Two models tend to find the same questions easy or hard. Each on its own counts that shared swing in both error bars. Question by question it cancels, and the error bar depends only on how many questions only one of the two models gets right.','两个模型往往觉得同样的题容易、同样的题难。各算各的会把这份共同的摆动在两条误差线里各算一次。逐题对比时它抵消了，误差线只取决于两个模型有多少题一对一错。')},
  {q:tx('Why can answering each question more times never bring the error bar down to 0?','为什么每道题多答几次，误差线也降不到 0？'),
   o:[tx('More answers average out each answer’s own luck, but the batch of questions stays the same, and its swing stays with it.','多答几次能平均掉每次回答自己的运气，可是抽到的那批题没有变，它带来的摆动也还在。'),
      tx('At a temperature above 0 every extra answer brings new randomness, so after some point more answers make the error bar longer.','temperature 大于 0 时，每多答一次都会带来新的随机性，所以答到一定次数以后，误差线反而变长。'),
      tx('The error bar depends only on the number of questions, so the number of answers per question does not change it at all.','误差线只取决于题数，所以每道题答几次完全不影响它。')],a:0,
   n:[null,tx('Averaging never adds swing: the more answers per question, the shorter the error bar, though it levels off toward a floor.','取平均不会增加摆动：每道题答的次数越多，误差线越短，只是越来越接近一个下限。'),
     tx('It does change it, a little: more answers shorten the error bar a bit, but the swing from which questions were drawn stays.','它会变，只是变得少：多答几次能让误差线短一点，可抽到哪些题带来的摆动还在。')],
   why:tx('A score’s swing has two parts: which questions were drawn, and each answer’s luck. Averaging answers shrinks only the second; the first shrinks only with more questions.','分数的摆动有两部分：抽到了哪些题，和每次回答的运气。多答几次只能缩小第二部分；第一部分只有加题才能缩小。')}],
 transfer:[
  {q:tx(`For a score near 80%, the error bar is ±${fPt(r164.hA)} points on 164 questions and ±${fPt(r1319.hA)} on 1,319. On a 500-question benchmark B is 1 point ahead of A. Does that 1 point show that B is better?`,`分数在 80% 左右时，164 道题的误差线是 ±${fPt(r164.hA)} 分，1,319 道是 ±${fPt(r1319.hA)} 分。一个 500 道题的 benchmark 上，B 比 A 高 1 分。这 1 分能说明 B 更好吗？`),
   o:[tx('Yes, 500 questions are three times as many as 164, enough for a 1-point gap','能，500 道题是 164 道的三倍，足够分辨 1 分'),tx('No, with 500 questions the error bar of B − A is several points wide even question by question, so 1 point is still inside the noise','不能，500 道题时，就算逐题对比，B − A 的误差线也有好几分，1 分还在噪声里'),tx('Only if B scored above 90%','只有 B 超过 90% 时才能')],a:1,
   why:tx(`500 questions sit between 164 and 1,319, so one score’s error bar is between ±${fPt(r1319.hA)} and ±${fPt(r164.hA)} points, about ±${fPt(x5.hA)}. Comparing question by question makes the error bar of B − A shorter, but with 500 questions it is usually still several points wide, wider than the 1-point gap. So the error bar crosses 0, and the gap does not show that B is better.`,
          `500 道题在 164 和 1,319 之间，所以一个分数的误差线在 ±${fPt(r1319.hA)} 和 ±${fPt(r164.hA)} 分之间，约 ±${fPt(x5.hA)} 分。逐题对比能让 B − A 的误差线短一些，可 500 道题时通常还有好几分，比 1 分宽。所以误差线会跨过 0，这 1 分说明不了 B 更好。`)},
  {q:tx('A benchmark gives an error bar of ±2 points. About how many times as many questions would bring it to ±1 point?','一个 benchmark 的误差线是 ±2 分。要压到 ±1 分，题数大约要变成原来的几倍？'),
   o:[tx('About 2 times','约 2 倍'),tx('About 4 times','约 4 倍'),tx('About 8 times','约 8 倍')],a:1,
   why:tx('The error bar is inversely proportional to the square root of the number of questions, so halving it takes 4 times the questions. Doubling the questions only shortens it by about 30%.','误差线和题数的平方根成反比，所以要缩短一半，题数要变成 4 倍。题数只翻一倍的话，误差线只短三成左右。')}],
 practice:[
  {id:'llm-eval-system',d:tx('Requirements: size each domain’s golden set for a 2-point difference at 95% confidence and 80% power, and see why 2,000 prompts fall short; then the gate across 12 domains.','“需求和规模估算”：按 95% 置信度、80% 检验力，算出分辨 2 分差距需要多大的 golden set，并看清为什么 2,000 条 prompt 不够；再设计 12 个领域的放行规则。')},
  {id:'stats-tests',d:tx('Question 7: two models on the same test set; compare a paired test (McNemar) with treating the two accuracies as independent, and derive why the paired one is more powerful.','第 7 题：两个模型做同一套测试题，比较配对检验（McNemar）和把两个正确率当成独立样本的检验，推导为什么配对检验更灵敏。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'eval',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Benchmark noise','评测的噪声'),title:l?L(l.q):tx('Is a 2-point benchmark gain real?','Benchmark 上高 2 分是真的吗？'),
    minutes:10,fid:'Analytical',fidPop:FIDPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage evc-stage" id="evc-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="evc-st-title"></div><div class="meter-sub" id="evc-st-sub"></div></div><div class="stage-actions"><span class="status" id="evc-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph evc-graph" id="evc-graph">`+
    `<div class="evc-phone" id="evc-phone"></div>`+
    `<div class="evc-axbox" id="evc-axbox"><div class="evc-cap" id="evc-ax-cap"></div><div class="evc-ax" id="evc-ax"></div>`+
     `<div class="evc-pilebox" id="evc-pilebox"><div class="evc-pile" id="evc-pile"></div><div class="evc-player" id="evc-player"></div></div></div>`+
    `<div class="evc-pairbox" id="evc-pairbox" hidden><div class="evc-cap" id="evc-grid-cap"></div><div class="evc-gridwrap"><div class="evc-grid" id="evc-grid"></div><div class="evc-gleg" id="evc-gleg"></div></div></div>`+
    `<div class="evc-kbox" id="evc-kbox" hidden><div class="evc-cap" id="evc-mix-cap"></div><div class="evc-mix" id="evc-mix"></div><div class="evc-cap" id="evc-kc-cap"></div><div class="evc-kc" id="evc-kc"></div></div>`+
    `<div class="evc-dfbox" id="evc-dfbox" hidden><div class="evc-cap" id="evc-df-cap"></div><div class="evc-df" id="evc-df"></div></div>`+
   `</div>`+
   `<div class="metrics evc-metrics" id="evc-metrics"></div>`+
   `<details class="math" id="evc-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="evc-mathbody"></div></details></div>`;
}
const NOTE={
 q:()=>tx('An assumed value: the share of questions only one of the two models gets right, with one answer each. The smaller it is, the shorter the question-by-question error bar. The real value is counted from the two models’ answer sheets, and it is never smaller than the gap between the two scores. This lab uses 12%, and 10% in part 4 and in the exam scenario.',
          '这是假设值，指两个模型各答一次时，一对一错的题所占的比例。它越小，逐题对比的误差线越短。真实值要从两个模型的答卷里数出来，它不会小于两个分数之差。这个 Lab 用 12%，第 4 部分和考试场景用 10%。'),
 c:()=>tx('An assumed value. At a temperature above 0, a model gets some questions right only about half the time. This value is the share of such questions; on the rest the model is always right or always wrong. The larger the share, the more several answers per question help. It depends on the model, the temperature and the benchmark, and it is close to 0 at temperature 0. This lab uses 20% so that the effect of several answers shows; it is not a measured value.',
          '这是假设值。temperature 大于 0 时，有些题模型时对时错，大约一半答对。这个值就是这类题的占比，其余的题模型每次都对或每次都错。这个占比越大，每道题多答几次越有用。它取决于模型、temperature 和 benchmark，temperature 为 0 时接近 0。这个 Lab 用 20%，是为了让多答几次的效果看得出来，不是测出来的值。')
};
const pIdx=p=>Math.round(p*1000), qIdx=q=>Math.round(q*100), cIdx=c=>Math.round(c*20);
function controlsHTML(){
  return `<aside class="panel controls" id="evc-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="evc-lock" hidden></p>`+
   `<div class="lockable" id="evc-ctl-n-wrap">${nCtl('evc-ctl-n',false)}</div>`+
   `<div class="lockable" id="evc-ctl-pA-wrap">${A.ui.range({id:'evc-ctl-pA',key:'pA',label:tx('A’s score','A 的分数'),min:500,max:950,value:pIdx(S.pA),out:fPct(S.pA)})}</div>`+
   `<div class="lockable" id="evc-ctl-pB-wrap">${A.ui.range({id:'evc-ctl-pB',key:'pB',label:tx('B’s score','B 的分数'),min:500,max:950,value:pIdx(S.pB),out:fPct(S.pB)})}</div>`+
   `<div class="lockable" id="evc-ctl-m-wrap">${A.ui.seg({id:'evc-ctl-m',key:'paired',label:tx('How to compare','比较方法'),options:[['0',tx('Each on its own','各算各的')],['1',tx('Question by question','逐题对比')]],value:String(S.paired),note:tx('Question by question needs both models to answer the same questions.','逐题对比要求两个模型做的是同一批题。')})}</div>`+
   `<div class="lockable" id="evc-ctl-k-wrap">${kCtl('evc-ctl-k',false)}</div>`+
   `<div class="evc-grp lockable" id="evc-ctl-as"><div class="evc-gh">${tx('Assumed values','假设值')}</div>`+
    A.ui.range({id:'evc-ctl-q',key:'q',label:tx('Questions only one model gets right','一对一错的题'),min:1,max:40,value:qIdx(S.q),out:fixd(S.q*100,0)+'%',note:NOTE.q()})+
    A.ui.range({id:'evc-ctl-c',key:'c',label:tx('Coin-flip questions','时对时错的题'),min:0,max:8,value:cIdx(S.c),out:fixd(S.c*100,0)+'%',note:NOTE.c()})+`</div>`+
   `<div class="lockable" id="evc-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import numpy as np

Z = 1.96   # 95% interval, normal approximation

def error_bar(p, n):
    """Half-width of the 95% interval of a score p on n questions."""
    return Z * np.sqrt(p * (1 - p) / n)

def paired_diff(a, b):
    """a, b: 0/1 per question, same questions. B - A and its bar."""
    d = np.asarray(b, float) - np.asarray(a, float)
    se = d.std(ddof=1) / np.sqrt(len(d))    # like lm-eval's stderr
    return d.mean(), Z * se

def separate_diff(a, b):
    pa, pb, n = np.mean(a), np.mean(b), len(a)
    return pb - pa, np.hypot(error_bar(pa, n), error_bar(pb, n))

def questions_needed(p, h):
    return int(np.ceil(Z**2 * p * (1 - p) / h**2))

print(round(100 * error_bar(0.80, 164), 1))     # 6.1 points
print(questions_needed(0.80, 0.01))             # 6147
rng = np.random.default_rng(0)
kind = rng.choice(4, 1319, p=[.76, .04, .06, .14])  # both/A/B/none
a, b = np.isin(kind, [0, 1]), np.isin(kind, [0, 2])
print(separate_diff(a, b), paired_diff(a, b))`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="evc-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="evc-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The code computes three things: one score’s error bar, the error bar of B − A both ways, and the number of questions needed to bring an error bar down to ±h.','这段代码算三件事：一个分数的误差线，用两种方法算出的 B − A 的误差线，以及要把误差线压到 ±h 需要多少道题。')}</p><pre class="code" id="evc-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card" id="evc-prod-card"><h3>${A.t('cProd')}</h3><ul class="evc-prod">`+
   tx(`<li>EleutherAI’s lm-evaluation-harness reports a standard error next to each score. For metrics that average over questions, such as accuracy, ${cite(SRC.lmeval,'<code>lm_eval/api/metrics.py</code>')} uses <code>mean_stderr</code>: the sample standard deviation divided by ${A.ui.math('\\sqrt{n}')}, almost exactly this lab’s formula for one score’s error bar. For a fixed list of metrics with no simple formula, such as F1 and BLEU, it uses <code>bootstrap_stderr</code>, which resamples the per-question results.</li>`+
      `<li>The standard paired test for right-or-wrong answers is McNemar’s test: it looks only at the questions only one model gets right. A paired bootstrap, resampling questions and keeping both models’ answers together, gives the same kind of interval for any metric.</li>`+
      `<li>To size an eval, people fix the smallest gain they must see and the confidence they need, then turn the error-bar formula around to solve for the questions; asking also for a chance of detecting a real gain (power, often 80%) roughly doubles the count.</li>`,
      `<li>EleutherAI 的 lm-evaluation-harness 会在每个分数旁边给出标准误差。准确率这类按题求平均的指标，${cite(SRC.lmeval,'<code>lm_eval/api/metrics.py</code>')} 用的是 <code>mean_stderr</code>：样本标准差除以 ${A.ui.math('\\sqrt{n}')}，和这个 Lab 算一个分数误差线的公式几乎一样。F1、BLEU 这类没有简单公式的指标列在一张固定的名单里，改用 <code>bootstrap_stderr</code>，对每道题的结果重抽样来估计。</li>`+
      `<li>对错类结果的标准配对检验是 McNemar 检验：它只看一对一错的题。配对 bootstrap（重抽样时让两个模型在同一道题上的答案一起走）对任何指标都能给出同样意义的区间。</li>`+
      `<li>确定评测规模时，先定下必须看出的最小提升和需要的置信度，再把误差线的公式倒过来，解出题数。如果还要求真有提升时大概率看得出来（检验力，常用 80%），题数大约还要翻一倍。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.miller,'Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations')}<span>${tx('Miller, 2024 (arXiv 2411.00640): a statistical approach to adding error bars to language-model evals.','Miller，2024（arXiv 2411.00640）：用统计方法给语言模型评测加上误差线。')}</span></li>`+
   `<li>${cite(SRC.codex,'Evaluating Large Language Models Trained on Code')}<span>${tx('Chen et al., 2021 (arXiv 2107.03374): introduces HumanEval, whose test file has 164 problems.','Chen 等，2021（arXiv 2107.03374）：提出 HumanEval，它的测试文件有 164 道题。')}</span></li>`+
   `<li>${cite(SRC.gsmp,'Training Verifiers to Solve Math Word Problems')}<span>${tx('Cobbe et al., 2021 (arXiv 2110.14168): introduces GSM8K; its test file has 1,319 problems.','Cobbe 等，2021（arXiv 2110.14168）：提出 GSM8K，它的测试文件有 1,319 道题。')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="evc-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="evc-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="evc-app m2" id="evc-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="evc-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see whether the error bar of B − A crosses 0.','选一个场景，看看 B − A 的误差线会不会跨过 0。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="evc-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Can B replace A? Size the eval','B 能替换 A 吗？定好评测方案')}</h3>`+
   `<p>${tx('Your team must decide whether the new model B can replace A in production. On your internal eval, B scores 2 points above A. Choose an eval plan to test those 2 points. The requirement: when the eval measures a 2-point gap, the 95% error bar of B − A must stay clear of 0, so the error bar must be shorter than ±2 points. Every answer costs money to grade, so among the plans that meet the requirement, use as few graded answers as you can.',
            '你们要决定新模型 B 能不能替换线上的 A。在内部评测上，B 比 A 高 2 分。请选一个评测方案来检验这 2 分。要求是测出 B 高 2 分时，B − A 的 95% 误差线不碰 0，也就是误差线要短于 ±2 分。每条回答都要花钱打分，所以在达到要求的方案里，要打分的回答越少越好。')}</p>`+
   `<ul class="evc-asm small"><li>${tx('A scores 80% and B 82%.','A 得 80%，B 得 82%。')}</li><li>${tx('Both models answer at temperature 0.7, and 20% of the questions are coin-flip questions for both of them.','两个模型都用 temperature 0.7 作答，有 20% 的题对两个模型来说都时对时错。')}</li><li>${tx('With one answer each, 12% of the questions are right for only one of the two models.','每个模型只答一次时，有 12% 的题一对一错。')}</li></ul>`+
   A.ui.seg({id:'evc-ic-chn',key:'chn',label:tx('Questions','题数'),options:CH.Ns.map(n=>[String(n),fN(n)]),value:String(S.n),inline:true})+
   A.ui.seg({id:'evc-ic-chm',key:'chm',label:tx('How to compare','比较方法'),options:[['0',tx('Each on its own','各算各的')],['1',tx('Question by question','逐题对比')]],value:String(S.paired),inline:true})+
   A.ui.seg({id:'evc-ic-chk',key:'chk',label:tx('Answers per question','每道题回答几次'),options:CH.Ks.map(k=>[String(k),String(k)]),value:String(S.k),inline:true})+
   `<div class="goal mono evc-goal" id="evc-ch-g1"><span>${tx('Error bar of B − A','B − A 的误差线')}</span><span><b id="evc-ch-h"></b>${tx(' (goal: under ±2)','（目标：小于 ±2）')}</span></div>`+
   `<div class="goal mono evc-goal" id="evc-ch-g2"><span>${tx('Graded answers','要打分的回答')}</span><span><b id="evc-ch-c"></b>${tx(' (fewer is better)','（越少越好）')}</span></div>`+
   submit+`<div id="evc-ch-res" class="evc-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  const w={mission:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},pile:{en:'Let the re-runs finish to continue (“Fast” takes a few seconds).',zh:'等重考播完才能继续（选“快”只要几秒）。'},
    size:{en:'Drag until A’s error bar is within ±1 point to continue.',zh:'把 A 的误差线拖到 ±1 分以内，才能继续。'},pair:{en:'Pick “Question by question” to continue.',zh:'选“逐题对比”才能继续。'},k:{en:'Set 16 answers per question to continue.',zh:'把回答次数调到 16，才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#evc-rail',app); if(!rail) return null; rail.innerHTML=railFull(); A.ui.initRanges(rail); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#evc-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&((s.id==='pile'&&pileDone)||(s.id==='size'&&S.n!==164)||s.id==='pair'||(s.id==='k')||!!observed[step]);
    $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===lv)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvTried.has(v)||v===lv?lvRes(v):'';});
    if(box){const html=show?s.obs():''; box.hidden=!html;
      if(html){const full=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+html;
        if(box._h!==full){const again=box._step===step&&box._h!=null; box.innerHTML=full; box._h=full; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#evc-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const e=chEval(S.n,S.paired,S.k), h=$('#evc-ch-h',rail), c=$('#evc-ch-c',rail); if(!h) return;
    h.textContent='±'+fPt(e.h); c.textContent=fN(e.cost);
    const g1=$('#evc-ch-g1',rail); g1.classList.toggle('met',e.ok); g1.classList.toggle('evc-over',!e.ok);
    $$('.seg-ctl[data-key="chn"]',rail).forEach(g=>A.ui.setPressed(g,String(S.n))); $$('.seg-ctl[data-key="chm"]',rail).forEach(g=>A.ui.setPressed(g,String(S.paired))); $$('.seg-ctl[data-key="chk"]',rail).forEach(g=>A.ui.setPressed(g,String(S.k)));
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#evc-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage
   ===================================================================== */
const viewNow=()=>mode==='guided'?viewOf(STEPS[step]):(mode==='challenge'?'ch':'sb');
/* before the prediction is answered and the re-runs have finished, no error bar is drawn anywhere (R3.8) */
const barsHidden=()=>mode==='guided'&&(isP('mission')||(isP('pile')&&!pileDone));
const showPile=()=>isP('pile')||(isP('remember')&&remView==='pile');
function axisRange(){
  if(mode==='guided') return [70,90];
  const r=cmp(S), lo=Math.min(S.pA-r.hA,S.pB-r.hB)*100, hi=Math.max(S.pA+r.hA,S.pB+r.hB)*100;
  let a=Math.floor((lo-1)/5)*5, b=Math.ceil((hi+1)/5)*5; if(b-a<20){const m=(a+b)/2; a=Math.round(m-10); b=a+20;}
  if(a<0){b-=a; a=0;} if(b>100){a-=b-100; b=100;} return [Math.max(0,a),Math.min(100,b)];
}
const pos=(v,[a,b])=>Math.max(0,Math.min(100,(v-a)/(b-a)*100));
function rowHTML(name,p,h,rng,hide){
  const x=pos(p*100,rng), l=pos((p-h)*100,rng), r=pos((p+h)*100,rng), lab=hide?`${name} ${fPct(p)}`:`${name} ${fPct(p)} ± ${fPt(h)}`;
  const tip=hide?tx(`${name}’s score: ${fPct(p)}.`,`${name} 的分数：${fPct(p)}。`):tx(`${name}’s score is ${fPct(p)}, and its error bar runs from ${fPct(p-h)} to ${fPct(p+h)}: ${fPt(h)} points either way.`,`${name} 的分数是 ${fPct(p)}，误差线从 ${fPct(p-h)} 到 ${fPct(p+h)}，上下各 ${fPt(h)} 分。`);
  const right=r>50;
  return `<div class="evc-row" tabindex="0" data-tipx="${esc(tip)}">${hide?'':`<i class="evc-bar" style="left:${l.toFixed(2)}%;width:${(r-l).toFixed(2)}%"></i>`}<i class="evc-dot" style="left:${x.toFixed(2)}%"></i>`+
    `<span class="evc-lab${right?' r':''}" style="${right?`right:${(100-(hide?x:r)).toFixed(2)}%`:`left:${(hide?x:r).toFixed(2)}%`}">${esc(lab)}</span></div>`;
}
function scaleHTML(rng){const [a,b]=rng; let s=''; for(let v=a; v<=b+1e-9; v+=5) s+=`<span style="left:${pos(v,rng).toFixed(2)}%">${v}%</span>`; return `<div class="evc-scale">${s}</div>`;}
function drawAxis(){
  const rng=axisRange(), r=cmp(S), hide=barsHidden(), v=viewNow();
  const cap=$('#evc-ax-cap',app);
  cap.textContent=v==='k'?tx(`A and B on ${fN(S.n)} problems, ${S.k===1?'1 answer':`${S.k} answers`} per problem averaged`,`A 和 B 在 ${fN(S.n)} 道题上的分数，每题答 ${S.k} 次取平均`)
    :tx(`A’s and B’s scores on ${fN(S.n)} questions${hide?'':', with their error bars'}`,`A 和 B 在 ${fN(S.n)} 道题上的分数${hide?'':'和误差线'}`);
  const h=`${rowHTML('A',S.pA,r.hA,rng,hide)}${rowHTML('B',S.pB,r.hB,rng,hide)}${scaleHTML(rng)}`;
  const ax=$('#evc-ax',app); if(ax._h!==h){ax.innerHTML=h; ax._h=h;}
  const pb=$('#evc-pilebox',app); pb.hidden=!showPile(); if(!pb.hidden) drawPile(rng);
}
/* part 2: the re-run scores pile up, one column per possible score; the error bar over the pile once it is complete */
let pileMax=0;
function drawPile(rng){
  const host=$('#evc-pile',app), m=Math.round(pileM);
  if(!pileMax){const cnt={}; REDRAWS.forEach(c=>{cnt[c]=(cnt[c]||0)+1; pileMax=Math.max(pileMax,cnt[c]);});}
  const stack={}; let dots='';
  for(let i=0;i<m;i++){const c=REDRAWS[i], y=stack[c]=(stack[c]||0)+1; dots+=`<i style="left:${pos(c/RD.n*100,rng).toFixed(2)}%;bottom:${((y-1)/pileMax*100).toFixed(2)}%"></i>`;}
  const done=m>=RD.K, l=pos((RD.p-RD_H)*100,rng), r=pos((RD.p+RD_H)*100,rng);
  const band=done?`<b class="evc-band" style="left:${l.toFixed(2)}%;width:${(r-l).toFixed(2)}%"></b><em class="evc-bandlab" style="left:${((l+r)/2).toFixed(2)}%">${esc(tx(`${inside(RD.K)} of ${RD.K} inside 80.0% ± ${fPt(RD_H)}`,`${RD.K} 次里有 ${inside(RD.K)} 次落在 80.0% ± ${fPt(RD_H)} 里`))}</em>`:'';
  host.innerHTML=`<div class="evc-dots">${dots}</div>${band}`;
  host.dataset.tipx=tx(`Each dot is the score of one re-run of A on 164 new problems (${m} of ${RD.K} so far). Dots with the same score stack up.`,`每个点是 A 重考 164 道新题的一次分数（目前 ${RD.K} 次里的 ${m} 次）。分数相同的点叠在一起。`);
  host.tabIndex=0; host.setAttribute('role','img'); host.setAttribute('aria-label',host.dataset.tipx);
  drawPlayer();
}
function drawPlayer(){
  const pl=$('#evc-player',app); if(!pl) return; const m=Math.round(pileM);
  const sig=[playing,spd,m,A.lang()].join('|'); if(pl._sig===sig) return; pl._sig=sig;
  const scrub=pl.querySelector('input[data-key="scrub"]');
  if(scrub&&pl._built===A.lang()){scrub.value=m; A.ui.setRange(scrub); const pb=pl.querySelector('[data-act="play"]'); pb.textContent=playing?tx('❚❚ Pause','❚❚ 暂停'):(m>=RD.K?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
    $$('.seg-ctl[data-key="spd"]',pl).forEach(g=>A.ui.setPressed(g,spd)); const c=pl.querySelector('.evc-count'); c.textContent=tx(`${m} of ${RD.K} re-runs`,`已重考 ${m} / ${RD.K} 次`); return;}
  pl._built=A.lang();
  pl.innerHTML=`<button type="button" class="btn ghost sm evc-play" data-act="play" data-tipx="${esc(tx(`Play the re-runs at the chosen speed; press again to pause. Once all ${RD.K} have run, the button plays them again from the first.`,`按选定的速度播放重考；再按一次暂停。${RD.K} 次都放完后，这个按钮从第 1 次重新播放。`))}">${playing?tx('❚❚ Pause','❚❚ 暂停'):(m>=RD.K?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'))}</button>`+
    A.ui.seg({id:'evc-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['mid',tx('Medium','中')],['fast',tx('Fast','快')]],value:spd})+
    `<div class="evc-scrub"><input type="range" min="0" max="${RD.K}" step="1" value="${m}" data-key="scrub" aria-label="${esc(tx('Re-runs shown','显示的重考次数'))}"><span class="evc-count mono">${tx(`${m} of ${RD.K} re-runs`,`已重考 ${m} / ${RD.K} 次`)}</span></div>`;
  A.ui.initRanges(pl);
}
function tick(t){
  raf=0; if(!app||!playing) return;
  const dt=lastT?t-lastT:16; lastT=t;
  pileM=Math.min(RD.K,pileM+dt/DUR[spd]*RD.K);
  if(pileM>=RD.K){playing=false; pileM=RD.K; pileDone=true; render(); revealSoonIf(); return;}
  drawPile(axisRange()); raf=requestAnimationFrame(tick);
}
function play(){if(pileM>=RD.K) pileM=0; playing=true; lastT=0; if(A.reduceMotion()){finishPile(); return;} if(!raf) raf=requestAnimationFrame(tick); drawPlayer();}
function pause(){playing=false; if(raf) cancelAnimationFrame(raf); raf=0; drawPlayer();}
function finishPile(){if(raf) cancelAnimationFrame(raf); raf=0; playing=false; pileM=RD.K; pileDone=true; wantObs=true; render(); revealSoon(); wantObs=false;}
function revealSoonIf(){if(isP('pile')){wantObs=true; updateRail(); wantObs=false; revealSoon();}}
/* part 4: the answer sheet as a 2 × 2 grid, each box's square sized by its share */
function drawGrid(){
  const sh={both:0.76,a:0.04,b:0.06,none:0.14}, n=P4.n, mx=Math.sqrt(0.76);
  $('#evc-grid-cap',app).textContent=tx(`The answer sheet: ${fN(n)} problems, by who got each one right`,`答卷：${fN(n)} 道题，按谁答对了来分`);
  const cell=(k,t,disc)=>{const s=Math.sqrt(sh[k])/mx*100, cnt=fN(Math.round(sh[k]*n));
    const tip=disc?tx(`${t}: ${fixd(sh[k]*100,0)}% of the problems, about ${cnt}. Only these problems move B − A.`,`${t}：占 ${fixd(sh[k]*100,0)}% 的题，约 ${cnt} 道。只有这类题会改变 B − A。`)
      :tx(`${t}: ${fixd(sh[k]*100,0)}% of the problems, about ${cnt}. They add the same to both scores, so they cancel in B − A.`,`${t}：占 ${fixd(sh[k]*100,0)}% 的题，约 ${cnt} 道。它们给两个分数加的一样多，所以在 B − A 里抵消了。`);
    return `<div class="evc-cell${disc?' disc':''}" tabindex="0" data-tipx="${esc(tip)}"><i style="width:${s.toFixed(1)}%;height:${s.toFixed(1)}%"></i><span><b>${fixd(sh[k]*100,0)}%</b> ${esc(t)}</span></div>`;};
  const h=`<span class="evc-grh0"></span><span class="evc-grh">${tx('B right','B 对')}</span><span class="evc-grh">${tx('B wrong','B 错')}</span>`+
    `<span class="evc-gr">${tx('A right','A 对')}</span>${cell('both',tx('both right','两个都对'),false)}${cell('a',tx('only A right','只有 A 对'),true)}`+
    `<span class="evc-gr">${tx('A wrong','A 错')}</span>${cell('b',tx('only B right','只有 B 对'),true)}${cell('none',tx('both wrong','两个都错'),false)}`;
  const g=$('#evc-grid',app); if(g._h!==h){g.innerHTML=h; g._h=h;}
  const lg=`<span tabindex="0" data-tipx="${esc(tx('A dashed box holds problems only one model got right. Question by question, only these move the difference.','虚线框里是一对一错的题。逐题对比时，只有这些题影响差值。'))}"><i class="sw evc-sw-disc"></i>${esc(tx('dashed: only one model right','虚线框：一对一错'))}</span>`+
    `<span tabindex="0" data-tipx="${esc(tx('A plain box: both right or both wrong. These problems cannot tell the two models apart.','实线框：两个都对或两个都错。这些题分不出两个模型谁更好。'))}"><i class="sw evc-sw-same"></i>${esc(tx('plain: both alike','实线框：两个一样'))}</span>`+
    `<span>${esc(tx('square area = share of problems','方块面积 = 题目占比'))}</span>`;
  const gl=$('#evc-gleg',app); if(gl._h!==lg){gl.innerHTML=lg; gl._h=lg;}
}
/* the difference B − A on its own axis, 0 marked: in part 4 both ways, in Scenarios the chosen way, in the challenge the plan's */
function drawDiff(){
  const v=viewNow(), host=$('#evc-df',app), cap=$('#evc-df-cap',app);
  let rows;
  if(v==='pair'){const s=p4(0), p=p4(1); rows=[{k:'sep',t:tx('Each on its own','各算各的'),r:s,on:lv==='sep'}].concat(lvTried.has('pair')?[{k:'pair',t:tx('Question by question','逐题对比'),r:p,on:lv==='pair'}]:[]);}
  else if(v==='ch'){const e=chEval(S.n,S.paired,S.k); rows=[{k:'ch',t:S.paired?tx('Question by question','逐题对比'):tx('Each on its own','各算各的'),r:e.r,on:true}];}
  else {const r=cmp(S); rows=[{k:'sb',t:S.paired?tx('Question by question','逐题对比'):tx('Each on its own','各算各的'),r,on:true}];}
  const ext=Math.max(...rows.map(x=>Math.max(Math.abs(x.r.lo),Math.abs(x.r.hi))))*100;
  let a=Math.min(-2,Math.floor(Math.min(...rows.map(x=>x.r.lo*100))-1)), b=Math.max(4,Math.ceil(Math.max(...rows.map(x=>x.r.hi*100))+1));
  if(v==='pair'||v==='ch'){a=Math.min(a,-4); b=Math.max(b,8);} if(!isFinite(ext)){a=-4;b=8;}
  const rng=[a,b], z=pos(0,rng);
  cap.textContent=v==='ch'?tx('B − A with this plan, when the measured gap is 2 points','这个方案下的 B − A（测出的差距是 2 分时）'):tx(`B − A in points, on the same ${fN(v==='pair'?P4.n:S.n)} questions`,`B − A（分），同一批 ${fN(v==='pair'?P4.n:S.n)} 道题`);
  const step=b-a>16?4:(b-a>8?2:1); let ticks=''; for(let t=Math.ceil(a/step)*step;t<=b;t+=step) ticks+=`<span style="left:${pos(t,rng).toFixed(2)}%">${sgn(String(t))}</span>`;
  const rh=rows.map(x=>{const i=ivl(x.r.D,x.r.h), l=pos(x.r.lo*100,rng), r=pos(x.r.hi*100,rng), d=pos(x.r.D*100,rng), right=r>50;
    const tip=tx(`${x.t}: B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}. ${x.r.real?'The error bar stays clear of 0.':'The error bar crosses 0, so there may be no difference at all.'}`,`${x.t}：B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}。${x.r.real?'误差线不碰 0。':'误差线跨过了 0，也可能根本没有差别。'}`);
    return `<div class="evc-row evc-drow${x.on?' on':''}${x.r.real?' real':''}" tabindex="0" data-tipx="${esc(tip)}" data-k="${x.k}"><i class="evc-bar" style="left:${l.toFixed(2)}%;width:${(r-l).toFixed(2)}%"></i><i class="evc-dot" style="left:${d.toFixed(2)}%"></i>`+
      `<span class="evc-lab${right?' r':''}" style="${right?`right:${(100-r).toFixed(2)}%`:`left:${r.toFixed(2)}%`}">${esc(`${x.t} ${i.Ds} ± ${i.hs}`)}</span></div>`;}).join('');
  const h=`<div class="evc-dfin"><div class="evc-zero" style="left:${z.toFixed(2)}%"><em>${esc(tx('0: no difference','0：没有差别'))}</em></div>${rh}<div class="evc-scale">${ticks}</div></div>`;
  if(host._h!==h){host.innerHTML=h; host._h=h;}
}
/* part 5: the problems' mix, and the error bar against answers per problem with its floor */
function drawK(){
  const n=S.n, c=S.c, ar=1-S.pA-c/2, al=S.pA-c/2;
  $('#evc-mix-cap',app).textContent=tx(`This lab assumes A answers these ${fN(n)} problems like this:`,`这个 Lab 假设 A 做这 ${fN(n)} 道题时：`);
  const row=(w,cls,t,tip)=>`<div class="evc-mrow" tabindex="0" data-tipx="${esc(tip)}"><span class="evc-ml">${esc(t)}</span><span class="evc-mtrk"><i class="${cls}" style="width:${(w*100).toFixed(2)}%"></i></span><b class="mono">${fixd(w*100,0)}%</b></div>`;
  const mh=row(al,'evc-m1',tx('always right','每次都对'),tx(`A answers ${fixd(al*100,0)}% of the problems right every time; answering again changes nothing.`,`A 有 ${fixd(al*100,0)}% 的题每次都答对，多答几次也一样。`))+
    row(c,'evc-m2',tx('coin flip','时对时错'),tx(`On ${fixd(c*100,0)}% of the problems A is right about half the time: only here does averaging several answers help.`,`A 有 ${fixd(c*100,0)}% 的题大约一半答对：只有这些题，多答几次取平均才有用。`))+
    row(ar,'evc-m3',tx('always wrong','每次都错'),tx(`A answers ${fixd(ar*100,0)}% of the problems wrong every time.`,`A 有 ${fixd(ar*100,0)}% 的题每次都答错。`));
  const mx=$('#evc-mix',app); if(mx._h!==mh){mx.innerHTML=mh; mx._h=mh;}
  $('#evc-kc-cap',app).textContent=tx('A’s error bar against answers per problem','A 的误差线和每道题回答次数的关系');
  const top=cmp({...S,k:1}).hA*100, fl=floorH(S.pA,c,n)*100, Y=v=>(v/(Math.ceil(top)+1)*100);
  const cols=KS.map(k=>{const hv=cmp({...S,k}).hA*100, on=k===S.k;
    return `<div class="evc-kcol${on?' on':''}" data-k="${k}" tabindex="0" data-tipx="${esc(tx(`${k} ${k===1?'answer':'answers'} per problem: A’s error bar is ±${fixd(hv,1)} points.`,`每道题答 ${k} 次：A 的误差线是 ±${fixd(hv,1)} 分。`))}"><i style="height:${Y(hv).toFixed(2)}%"><em>±${fixd(hv,1)}</em></i></div>`;}).join('');
  const labs=KS.map(k=>`<span${k===S.k?' class="on"':''}>${k}</span>`).join('');
  const floor=`<div class="evc-floor" style="bottom:${Y(fl).toFixed(2)}%" tabindex="0" data-tipx="${esc(tx(`±${fixd(fl,1)} points is the swing from which problems were drawn. However many times each problem is answered, the error bar cannot go below it.`,`±${fixd(fl,1)} 分是“抽到哪些题”带来的摆动，每道题答多少次都压不到它以下。`))}"><em>${esc(tx(`floor ±${fixd(fl,1)}`,`下限 ±${fixd(fl,1)}`))}</em><small>${esc(tx('from which problems were drawn','来自抽到了哪些题'))}</small></div>`;
  const kh=`<div class="evc-kbars">${floor}${cols}</div><div class="evc-klabs">${labs}</div><div class="evc-kx">${esc(tx('answers per problem','每道题回答次数'))}</div>`;
  const kc=$('#evc-kc',app); if(kc._h!==kh){kc.innerHTML=kh; kc._h=kh;}
}
function statusOf(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.n,S.paired,S.k); return {cls:e.ok?'ok':'bad',txt:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),short:e.ok?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
    tip:tx(`Green when the error bar of B − A is under ±2 points. Now: ±${fPt(e.h)}, with ${fN(e.cost)} graded answers.`,`B − A 的误差线小于 ±2 分时，徽章就是绿色。现在是 ±${fPt(e.h)} 分，要打分的回答 ${fN(e.cost)} 条。`)};}
  if(barsHidden()) return {cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),tip:tx('The error bars stay hidden until part 2 shows the re-runs.','误差线先隐藏，等第 2 部分的重考播完再显示。')};
  /* one rule everywhere (B-1, R2.8): the verdict is B − A's error bar against 0, never the overlap of A's and B's bars.
     Guided parts 2, 3, 5 and 6 combine A's and B's error bars (each on its own) until part 4 introduces question by question. */
  const pair=v==='pair', r=pair?p4(lv==='pair'?1:0):cmp(S), i=ivl(r.D,r.h), byQ=pair?lv==='pair':!!S.paired;
  const how=mode==='guided'&&!pair?tx('combined from A’s and B’s error bars','由 A 和 B 的误差线合成'):(byQ?tx('question by question','逐题对比'):tx('each on its own','各算各的'));
  const bars=!pair&&r.overlap;
  const tip=!r.real?tx(`B − A = ${i.Ds} ± ${i.hs} points (${how}). This error bar crosses 0, so the gap cannot be told from noise.`,`B − A = ${i.Ds} ± ${i.hs} 分（${how}）。这条误差线跨过了 0，分数差和噪声分不开。`)
    :(bars?tx(`A’s and B’s error bars still overlap, yet B − A = ${i.Ds} ± ${i.hs} points (${how}), and that error bar stays clear of 0.`,`A 和 B 的误差线还有重叠，但 B − A = ${i.Ds} ± ${i.hs} 分（${how}），这条误差线不碰 0。`)
          :tx(`B − A = ${i.Ds} ± ${i.hs} points (${how}). This error bar stays clear of 0.`,`B − A = ${i.Ds} ± ${i.hs} 分（${how}）。这条误差线不碰 0。`));
  return {cls:'neutral',txt:r.real?tx('Gap beyond noise','超出了噪声'):tx('Cannot tell','分不出'),short:r.real?tx('Beyond noise','超出噪声'):tx('Cannot tell','分不出'),tip};
}
function drawHead(){
  const v=viewNow(), t=$('#evc-st-title',app), sub=$('#evc-st-sub',app), st=$('#evc-status',app);
  t.textContent=v==='ch'?tx('Your internal eval: B − A for the plan you chose','内部评测：所选方案下的 B − A'):(v==='pair'?tx('1,319 math problems: the same 2 points, two ways','1,319 道数学题：同样的 2 分，两种算法'):(v==='k'?tx('164 coding problems at temperature 0.7','164 道编程题，temperature 0.7'):(mode==='sandbox'?tx('Two models on one benchmark','两个模型，同一个 benchmark'):tx('A coding benchmark: is B really better?','编程 benchmark：B 真的更好吗？'))));
  sub.textContent=v==='ch'?chName(S.n,S.paired,S.k):tx(`${fN(v==='pair'?P4.n:S.n)} questions, A ${fPct(S.pA)}, B ${fPct(S.pB)}`,`${fN(v==='pair'?P4.n:S.n)} 道题，A ${fPct(S.pA)}，B ${fPct(S.pB)}`);
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').replace('±','').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').replace('±','').match(/^\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const v=viewNow();
  if(v==='ch'){const e=chEval(S.n,S.paired,S.k), i=ivl(e.r.D,e.r.h);
    return [{k:'h',label:tx('Error bar of B − A','B − A 的误差线'),v:'±'+fPt(e.h),sub:tx('goal: under ±2','目标：小于 ±2'),tip:tx(`With this plan, a measured 2-point gap gives B − A = ${i.Ds} ± ${i.hs} points.`,`按这个方案，测出 B 高 2 分时，B − A = ${i.Ds} ± ${i.hs} 分。`)},
      {k:'cost',label:tx('Graded answers','要打分的回答'),v:fN(e.cost),sub:tx('questions × answers × 2 models','题数 × 回答次数 × 2 个模型'),tip:tx(`${fN(S.n)} questions × ${S.k} ${S.k===1?'answer':'answers'} × 2 models = ${fN(e.cost)} answers to grade.`,`${fN(S.n)} 道题 × ${S.k} 次 × 2 个模型 = ${fN(e.cost)} 条回答要打分。`)},
      {k:'lo',label:tx('Lower end of B − A','B − A 的下端'),v:i.lo,sub:tx('must stay above 0','要大于 0'),tip:tx(`The error bar of B − A starts at ${i.lo} points; above 0 means it stays clear of 0.`,`B − A 的误差线从 ${i.lo} 分开始；大于 0 就是不碰 0。`)}];}
  if(v==='pair'){const s=p4(0), p=p4(1), si=ivl(s.D,s.h), pi=ivl(p.D,p.h);
    return [{k:'gap',label:tx('B − A','B − A'),v:fPt(s.D),sub:tx('points, same answers','分，同一份答卷'),tip:tx('B’s 82.0% minus A’s 80.0%: 2.0 points either way you work out the error bar.','B 的 82.0% 减去 A 的 80.0%：不管误差线怎么算，差值都是 2.0 分。')},
      {k:'hs',label:tx('Each on its own','各算各的'),v:'±'+si.hs,sub:tx('crosses 0','跨过 0'),tip:tx(`Each on its own: B − A = ${si.Ds} ± ${si.hs} points, from ${si.lo} to ${si.hi}.`,`各算各的：B − A = ${si.Ds} ± ${si.hs} 分，从 ${si.lo} 到 ${si.hi}。`)},
      {k:'hp',label:tx('Question by question','逐题对比'),v:lvTried.has('pair')?'±'+pi.hs:'—',sub:lvTried.has('pair')?tx('clear of 0','不碰 0'):tx('pick it in the card','在卡片里选它'),tip:lvTried.has('pair')?tx(`Question by question: B − A = ${pi.Ds} ± ${pi.hs} points, from ${pi.lo} to ${pi.hi}.`,`逐题对比：B − A = ${pi.Ds} ± ${pi.hs} 分，从 ${pi.lo} 到 ${pi.hi}。`):tx('Shown once you pick “Question by question” in the card.','在卡片里选了“逐题对比”后显示。')},
      {k:'q',label:tx('Only one model right','一对一错的题'),v:'10%',sub:tx('about 132 problems','约 132 道'),tip:tx('4% only A right plus 6% only B right: 10% of the 1,319 problems, about 132.','只有 A 对的 4% 加上只有 B 对的 6%，一共是 1,319 道题里的 10%，约 132 道。')}];}
  const r=cmp(S), hide=barsHidden();
  const m=[{k:'hA',label:tx('A’s error bar','A 的误差线'),v:hide?'—':'±'+fPt(r.hA),sub:hide?tx('shown in part 2','第 2 部分揭晓'):tx('points, 95%','分，95%'),tip:hide?tx('Hidden until part 2 shows the re-runs.','在第 2 部分重考之前先隐藏。'):tx(`A’s score is ${fPct(S.pA)} ± ${fPt(r.hA)} points on ${fN(S.n)} questions.`,`${fN(S.n)} 道题时，A 的分数是 ${fPct(S.pA)} ± ${fPt(r.hA)} 分。`)},
    {k:'gap',label:tx('B − A','B − A'),v:fPt(r.D),sub:tx('points','分'),tip:tx(`B’s ${fPct(S.pB)} minus A’s ${fPct(S.pA)}.`,`B 的 ${fPct(S.pB)} 减去 A 的 ${fPct(S.pA)}。`)},
    {k:'n',label:tx('Questions','题数'),v:fN(S.n),sub:v==='k'?tx(`${S.k} ${S.k===1?'answer':'answers'} each`,`每题答 ${S.k} 次`):tx('in the benchmark','道'),tip:tx(`The benchmark has ${fN(S.n)} questions.`,`这个 benchmark 有 ${fN(S.n)} 道题。`)}];
  if(mode==='sandbox'){const i=ivl(r.D,r.h); m.push({k:'d',label:tx('Error bar of B − A','B − A 的误差线'),v:'±'+i.hs,sub:S.paired?tx('question by question','逐题对比'):tx('each on its own','各算各的'),tip:tx(`B − A = ${i.Ds} ± ${i.hs} points, from ${i.lo} to ${i.hi}.`,`B − A = ${i.Ds} ± ${i.hs} 分，从 ${i.lo} 到 ${i.hi}。`)});}
  if(v==='k'){const fl=floorH(S.pA,S.c,S.n); m.push({k:'fl',label:tx('Floor','下限'),v:'±'+fPt(fl),sub:tx('from the draw of problems','来自抽到哪些题'),tip:tx(`However many answers per problem, A’s error bar never goes below ±${fPt(fl)} points.`,`每道题答多少次，A 的误差线都低不过 ±${fPt(fl)} 分。`)});}
  return m;
}
function drawMetrics(){
  const host=$('#evc-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.v;});
}
/* Show the math: every line redoes from the numbers it shows; every term has a tooltip (R1.4) */
const f4=x=>{if(x===0) return '0'; const d=Math.max(0,3-Math.floor(Math.log10(Math.abs(x)))); return fixd(x,Math.min(d,7));};
function mathHTML(){
  const v=viewNow(), t=T_;
  if(barsHidden()) return `<p>${tx('The formulas appear once part 2 has shown the re-runs.','等第 2 部分的重考播完，这里给出公式。')}</p>`;
  const blocks=[], sq=x=>x<0?`{(${sgn(fixd(x,3))})}^2`:`${fixd(x,3)}^2`;
  /* the error bar from a standard error, in the same grid as the SE rows above it */
  const barRows=(sym,se,h)=>[[NM('error bar','误差线'),`±1.96 \\times ${sym}`],['',`±1.96 \\times ${f4(se)}`],['',`±${f4(h)}`]];
  const ptsRes=hs=>tx(`That is <b>±${hs} points</b>.`,`也就是 <b>±${hs} 分</b>。`);
  if(v==='pair'){const s=p4(0), p=p4(1), n=fN(P4.n), si=ivl(s.D,s.h), pi=ivl(p.D,p.h);
    blocks.push(MB(tx('<b>Each on its own.</b> Add A’s and B’s own swings, and take the sum as the swing of B − A.','<b>各算各的。</b>把 A 和 B 各自的摆动加起来，当作 B − A 的摆动。'),
      EQ([[NM('SE of B − A','B − A 的标准误差')+'\\,\\,\\text{SE}','\\sqrt{\\frac{p_A(1-p_A)}{n} + \\frac{p_B(1-p_B)}{n}}'],['',`\\sqrt{\\frac{0.8 \\times 0.2}{${n}} + \\frac{0.82 \\times 0.18}{${n}}}`],['',f4(s.se)]].concat(barRows('\\text{SE}',s.se,s.h))),ptsRes(si.hs)));
    blocks.push(MB(tx(`<b>Question by question.</b> Only the questions that one model gets right and the other wrong move B − A. ${t('q','q')} is their share, here 4% + 6% = 10% (4% only A right, 6% only B right). Δ is B’s score minus A’s, here 0.82 − 0.80 = 0.02.`,
                      `<b>逐题对比。</b>只有一对一错的题会让 B − A 摆动。${t('q','q')} 是这类题的占比，这里是 4% + 6% = 10%（只有 A 对的 4%，只有 B 对的 6%）。Δ 是 B 的分数减去 A 的分数，这里是 0.82 − 0.80 = 0.02。`),
      EQ([[NM('SE of B − A','B − A 的标准误差')+'\\,\\,\\text{SE}','\\sqrt{\\frac{q - Δ^2}{n}}'],['',`\\sqrt{\\frac{0.10 - 0.02^2}{${n}}}`],['',f4(p.se)]].concat(barRows('\\text{SE}',p.se,p.h))),ptsRes(pi.hs)));
    return blocks.join('');}
  const r=cmp(S), seA=r.seA, terms=tx(`${t('pA','p')} is A’s score, ${fixd(S.pA,3)} here. ${t('n','n')} is the number of questions, ${fN(S.n)} here.`,`${t('pA','p')} 是 A 的分数，这里是 ${fixd(S.pA,3)}。${t('n','n')} 是题数，这里是 ${fN(S.n)}。`);
  if(S.k===1) blocks.push(MB(tx(`<b>A’s error bar.</b> First the standard error SE: how far A’s score typically moves from one batch of questions to the next. The error bar reaches 1.96 SEs either side. ${terms}`,
                                `<b>A 的误差线。</b>先算标准误差 SE，它表示换一批题时，A 的分数通常摆动多少。误差线往两边各伸出 1.96 个 SE。${terms}`),
    EQ([[NM('standard error','标准误差')+'\\,\\,\\text{SE}_A','\\sqrt{\\frac{p(1-p)}{n}}'],['',`\\sqrt{\\frac{${fixd(S.pA,3)} \\times ${fixd(1-S.pA,3)}}{${fN(S.n)}}}`],['',f4(seA)]].concat(barRows('\\text{SE}_A',seA,r.hA))),ptsRes(fPt(r.hA))));
  else blocks.push(MB(tx(`<b>A’s error bar with ${S.k} answers per question.</b> ${terms} ${t('c','c')} is the share of coin-flip questions, ${fixd(S.c,2)} here, and ${t('k','k')} the number of answers per question, ${S.k}. This lab assumes every other question is always right or always wrong, so the formula simplifies to the one below.`,
                         `<b>每道题答 ${S.k} 次时，A 的误差线。</b>${terms}${t('c','c')} 是时对时错的题的占比，这里是 ${fixd(S.c,2)}。${t('k','k')} 是每道题回答的次数，这里是 ${S.k}。这个 Lab 假设其余的题每次都对或每次都错，所以公式可以化简成下面这样。`),
    EQ([[NM('standard error squared','标准误差的平方')+'\\,\\,\\text{SE}_A^2','\\frac{p(1-p) - \\frac{c}{4}(1 - \\frac{1}{k})}{n}'],['',`\\frac{${fixd(S.pA,3)} \\times ${fixd(1-S.pA,3)} - \\frac{${fixd(S.c,2)}}{4}(1 - \\frac{1}{${S.k}})}{${fN(S.n)}}`],['',`\\frac{${fixd(S.pA*(1-S.pA),4)} - ${fixd(S.c/4*(1-1/S.k),4)}}{${fN(S.n)}}`],['\\text{SE}_A',f4(seA)]].concat(barRows('\\text{SE}_A',seA,r.hA))),ptsRes(fPt(r.hA))));
  if(v==='axis'&&isP('size')) blocks.push(MB(tx('<b>Questions for ±1 point.</b> Turn the error bar’s formula around and solve for the number of questions n. h is the error bar you want, here 0.01 (1 point), and p = 0.8.','<b>压到 ±1 分要多少道题。</b>把误差线的公式倒过来，解出题数 n。h 是想要的误差线长度，这里是 0.01（1 分）。p 是 0.8。'),
    EQ([[NM('questions needed','需要的题数')+'\\,\\,n','\\frac{1.96^2 \\times p(1-p)}{h^2}'],['','\\frac{3.8416 \\times 0.8 \\times 0.2}{0.01^2}'],['',fixd(nFor(0.8,0.01),1)]]),
    tx(`Rounded up, <b>${fN(Math.ceil(nFor(0.8,0.01)))} questions</b>.`,`向上取整，要 <b>${fN(Math.ceil(nFor(0.8,0.01)))} 道题</b>。`)));
  if(mode!=='guided'||v==='ch'){const i=ivl(r.D,r.h), res=tx(`B − A = ${i.Ds} ± <b>${i.hs}</b> points, from ${i.lo} to ${i.hi}.`,`B − A = ${i.Ds} ± <b>${i.hs}</b> 分，从 ${i.lo} 分到 ${i.hi} 分。`), nm=NM('SE of B − A','B − A 的标准误差')+'\\,\\,\\text{SE}_{B-A}';
    if(S.paired) blocks.push(MB(tx(`<b>The error bar of B − A, question by question.</b> ${t('q','q')} is the share of questions only one model gets right, ${fixd(S.q,2)} here. Δ is B’s score minus A’s, ${sgn(fixd(r.D,3))} here.`,`<b>逐题对比时，B − A 的误差线。</b>${t('q','q')} 是一对一错的题的占比，这里是 ${fixd(S.q,2)}。Δ 是 B 的分数减去 A 的分数，这里是 ${sgn(fixd(r.D,3))}。`),
      EQ((S.k>1?[[nm,'\\sqrt{\\frac{q - Δ^2 - \\frac{c}{2}(1 - \\frac{1}{k})}{n}}'],['',`\\sqrt{\\frac{${fixd(S.q,2)} - ${sq(r.D)} - \\frac{${fixd(S.c,2)}}{2}(1 - \\frac{1}{${S.k}})}{${fN(S.n)}}}`]]
                :[[nm,'\\sqrt{\\frac{q - Δ^2}{n}}'],['',`\\sqrt{\\frac{${fixd(S.q,2)} - ${sq(r.D)}}{${fN(S.n)}}}`]]).concat([['',f4(r.se)]],barRows('\\text{SE}_{B-A}',r.se,r.h))),res));
    else blocks.push(MB(tx('<b>The error bar of B − A, each on its own.</b> Combine the two scores’ standard errors.','<b>各算各的时，B − A 的误差线。</b>把两个分数的标准误差合起来。'),
      EQ([[nm,'\\sqrt{\\text{SE}_A^2 + \\text{SE}_B^2}'],['',`\\sqrt{${f4(r.seA)}^2 + ${f4(r.seB)}^2}`],['',f4(r.seS)]].concat(barRows('\\text{SE}_{B-A}',r.seS,r.h))),res));}
  return blocks.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.Ns.includes(S.n)) S.n=CH.start.n; if(!CH.Ks.includes(S.k)) S.k=CH.start.k; Object.assign(S,{pA:CH.pA,pB:CH.pB,q:CH.q,c:CH.c});}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true; if(isP('k')) kSeen.add(S.k);}
  const v=viewNow();
  drawHead();
  $('#evc-axbox',app).hidden=!(v==='axis'||v==='k'||v==='sb');
  $('#evc-pairbox',app).hidden=v!=='pair';
  $('#evc-kbox',app).hidden=v!=='k';
  $('#evc-dfbox',app).hidden=!(v==='pair'||v==='ch'||v==='sb');
  if(!$('#evc-axbox',app).hidden) drawAxis();
  if(v==='pair') drawGrid();
  if(v==='k') drawK();
  if(!$('#evc-dfbox',app).hidden) drawDiff();
  $('#evc-graph',app).dataset.v=v;
  drawMetrics();
  const mb=$('#evc-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you change only the questions, the way to compare and the answers per question, in the task card. Everything else is fixed: A 80%, B 82%, 12% of the questions with only one model right, 20% coin-flip questions.','挑战中只能在任务卡里改题数、比较方法和回答次数。其余设置固定：A 80%，B 82%，12% 的题一对一错，20% 的题时对时错。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='size') return tx('Fixed in this part: A 80.0%, B 82.0%, one answer per question. Only the number of questions changes.','这一部分固定：A 80.0%，B 82.0%，每题答 1 次。只改题数。');
  if(id==='k') return tx('Fixed in this part: 164 problems, A 80.0%, B 82.0%, 20% coin-flip problems. Only the answers per problem change.','这一部分固定：164 道题，A 80.0%，B 82.0%，20% 的题时对时错。只改回答次数。');
  if(id==='pair') return tx('In this part you pick one way to compare in the card: 1,319 problems, A 80.0%, B 82.0%, 10% of the problems with only one model right. You can change the settings in Scenarios.','这一部分只在卡片里选比较方法：1,319 道题，A 80.0%，B 82.0%，10% 的题一对一错。在应用场景里可以改这些设置。');
  return tx('Fixed in this part: 164 problems, A 80.0%, B 82.0%, one answer per problem. You can change the settings in Scenarios.','这一部分固定：164 道题，A 80.0%，B 82.0%，每题答 1 次。在应用场景里可以改这些设置。');
}
function syncControls(){
  const setR=(key,v,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==v) i.value=v; A.ui.setRange(i); const w=i.closest('.ctl,.inline-ctl'), o=w&&w.querySelector('output'); if(o) o.textContent=out; i.setAttribute('aria-valuetext',out);});
  setR('n',nIdx(S.n),fN(S.n)); setR('pA',pIdx(S.pA),fPct(S.pA)); setR('pB',pIdx(S.pB),fPct(S.pB)); setR('q',qIdx(S.q),fixd(S.q*100,0)+'%'); setR('c',cIdx(S.c),fixd(S.c*100,0)+'%');
  $$('.seg-ctl[data-key="k"]',app).forEach(g=>A.ui.setPressed(g,String(S.k))); $$('.seg-ctl[data-key="paired"]',app).forEach(g=>A.ui.setPressed(g,String(S.paired)));
  const g=mode==='guided', ch=mode==='challenge', hide=(id,h)=>{const e=$(id,app); if(e) e.hidden=h;};
  $('#evc-controls',app).classList.toggle('locked',ch);
  hide('#evc-ctl-n-wrap',ch||(g&&!isP('size'))); hide('#evc-ctl-k-wrap',ch||(g&&!isP('k'))); ['#evc-ctl-pA-wrap','#evc-ctl-pB-wrap','#evc-ctl-m-wrap','#evc-ctl-as','#evc-ctl-reset-wrap'].forEach(id=>hide(id,g||ch));
  const q=$('#evc-ctl-q',app); if(q) q.hidden=!S.paired; const c=$('#evc-ctl-c',app); if(c) c.hidden=S.k===1;
  const ln=$('#evc-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const v=viewNow();
  if(v==='ch'){const e=chEval(S.n,S.paired,S.k); A.announce(tx(`${chName(S.n,S.paired,S.k)}: error bar ±${fPt(e.h)} points, ${fN(e.cost)} graded answers.`,`${chName(S.n,S.paired,S.k)}：误差线 ±${fPt(e.h)} 分，打分 ${fN(e.cost)} 条。`)); return;}
  if(barsHidden()) return; const r=cmp(S); A.announce(tx(`${fN(S.n)} questions: A’s error bar ±${fPt(r.hA)} points.`,`${fN(S.n)} 道题：A 的误差线 ±${fPt(r.hA)} 分。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  Object.assign(S,p);
  S.n=NS.includes(S.n)||CH.Ns.includes(S.n)?S.n:NS[nIdx(S.n)]; S.pA=Math.round(S.pA*1000)/1000; S.pB=Math.round(S.pB*1000)/1000; S.q=Math.round(S.q*100)/100; S.c=Math.round(S.c*20)/20;
  if(!KS.includes(S.k)) S.k=1; S.paired=S.paired?1:0;
  if(mode==='sandbox') fixQC(S);
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setLv(id){lv=id; lvTried.add(id); render();}
function setRem(k){if(!REMS.includes(k)) return; remView=k; pause();
  S={...BASE}; if(k==='size') S.n=6200; if(k==='pair'){lv='pair'; lvTried.add('pair');} if(k==='k') S.k=16; if(k==='pile'){pileM=RD.K;}
  renderRail(); if(innerWidth<=760) A.ui.reveal($('#evc-stage',app));}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='pair') return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(tx('Scrolls down to the two ways in the card.','向下滚到卡片里的两种算法。'))}">${tx('Pick a way below ↓','在下面选算法 ↓')}</button>`;
  if(id==='size') return nCtl('evc-ph-n',true);
  if(id==='k') return kCtl('evc-ph-k',true);
  return '';
}
function renderPhone(){const el=$('#evc-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h; A.ui.initRanges(el);}}
function enterStep(i){
  wantObs=false; pause(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  const st=$('#evc-stage',app); if(st) st.scrollTop=0;
  if(s.id==='pile'){if(!pileDone){pileM=0;} else pileM=RD.K;}
  renderRail();
  if(s.id==='pile'&&!pileDone) play();
}
function revealStep(){A.ui.reveal($('#evc-rail .step.current',app)||$('#evc-rail',app));}
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
    const st=$('#evc-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#evc-rail .step.current',app)||$('#evc-rail .railcard',app)||$('#evc-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tk,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#evc-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#evc-insight',app)); else revealBox($('#evc-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m; pause();
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,...CH.start,pA:CH.pA,pB:CH.pB,q:CH.q,c:CH.c};}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...PRESETS[0].s};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#evc-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=(s=S)=>[nIdx(s.n),pIdx(s.pA),pIdx(s.pB),s.paired,qIdx(s.q),KS.indexOf(s.k),cIdx(s.c)].join('_');
function parse(tok){
  const p=String(tok||'').split('_').map(Number); if(p.length!==7||p.some(x=>!Number.isInteger(x))) return null;
  const [ni,a,b,m,q,ki,c]=p;
  if(ni<0||ni>=NS.length||a<500||a>950||b<500||b>950||(m!==0&&m!==1)||q<1||q>40||ki<0||ki>=KS.length||c<0||c>8) return null;
  return fixQC({n:NS[ni],pA:a/1000,pB:b/1000,paired:m,q:q/100,k:KS[ki],c:c/20});
}

/* =====================================================================
   interview drill: seeded, from the engine
   ===================================================================== */
function makeDrill(seed){const r=mulberry(seed*104729+11), pk=a=>a[Math.floor(r()*a.length)];
  const n=pk([400,800,1200,2500]), pA=pk([0.6,0.7,0.75]), D=pk([0.02,0.03]), q=pk([0.08,0.12,0.16]);
  return {seed,n,pA,pB:Math.round((pA+D)*100)/100,q,s:cmp({n,pA,pB:Math.round((pA+D)*100)/100,q,k:1,c:0,paired:0}),p:cmp({n,pA,pB:Math.round((pA+D)*100)/100,q,k:1,c:0,paired:1})};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#evc-drill-q',app), a=$('#evc-drill-a',app);
  qd.innerHTML=tx(`Two models answer the same <b>${fN(d.n)}</b> questions: A scores <b>${fPct(d.pA)}</b>, B <b>${fPct(d.pB)}</b>, and <b>${fixd(d.q*100,0)}%</b> of the questions are right for only one of them. Give the 95% error bar of B − A each on its own and question by question. Is B’s lead beyond noise?`,
                  `两个模型做同样的 <b>${fN(d.n)}</b> 道题：A 得 <b>${fPct(d.pA)}</b>，B 得 <b>${fPct(d.pB)}</b>，<b>${fixd(d.q*100,0)}%</b> 的题一对一错。分别用各算各的和逐题对比，给出 B − A 的 95% 误差线。B 的领先超出了噪声吗？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const si=ivl(d.s.D,d.s.h), pi=ivl(d.p.D,d.p.h);
  const v1=d.pA*(1-d.pA), v2=d.pB*(1-d.pB);
  const vd=d.p.real?(d.s.real?tx('Beyond noise both ways','两种算法都超出了噪声'):tx('Beyond noise question by question; each on its own cannot tell','逐题对比时超出了噪声，各算各的分不出')):(d.s.real?tx('Beyond noise only each on its own','只有各算各的超出了噪声'):tx('Cannot tell either way','两种算法都分不出'));
  const dEq=(num,se,h)=>EQ([[NM('SE of B − A','B − A 的标准误差')+'\\,\\,\\text{SE}',`\\sqrt{\\frac{${num}}{${fN(d.n)}}}`],['',f4(se)],[NM('error bar','误差线'),`±1.96 \\times ${f4(se)}`],['',`±${f4(h)}`]]);
  a.innerHTML=tx(`<p><b>Each on its own gives ${si.Ds} ± ${si.hs} points, question by question ${pi.Ds} ± ${pi.hs}. ${vd}.</b></p>`,`<p><b>各算各的是 ${si.Ds} ± ${si.hs} 分，逐题对比是 ${pi.Ds} ± ${pi.hs} 分。${vd}。</b></p>`)+
    `<p>${tx('Each on its own: add the two scores’ p(1 − p).','各算各的：把两个分数各自的 p(1 − p) 加起来。').replace('p(1 − p)',A.ui.math('p(1-p)'))}</p>`+dEq(`${fixd(v1,4)} + ${fixd(v2,4)}`,d.s.se,d.s.h)+
    `<p>${tx('Question by question: q minus Δ squared.','逐题对比：用 q 减去 Δ 的平方。')}</p>`+dEq(`${fixd(d.q,2)} - ${fixd(d.s.D,2)}^2`,d.p.se,d.p.h)+
    `<p>${tx(`The interval question by question runs from ${pi.lo} to ${pi.hi} points.`,`逐题对比的区间从 ${pi.lo} 分到 ${pi.hi} 分。`)}</p>`;
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={n:{en:'Questions in the benchmark',zh:'benchmark 的题数'},pA:{en:'A’s score',zh:'A 的分数'},k:{en:'Answers per question',zh:'每道题回答几次'},q:{en:'Questions only one model gets right',zh:'一对一错的题'},c:{en:'Coin-flip questions',zh:'时对时错的题'}};
const TERM={
 n:()=>tx(`n is the number of questions, ${fN(S.n)} here. The standard error is inversely proportional to its square root.`,`n 是题数，这里是 ${fN(S.n)}。标准误差和它的平方根成反比。`),
 pA:()=>tx(`p is A’s score as a share, ${fixd(S.pA,3)} here. p(1 − p) is largest at 0.5, so a score near 50% swings the most.`,`p 是 A 的分数（小数），这里是 ${fixd(S.pA,3)}。p(1 − p) 在 0.5 时最大，所以分数在 50% 附近时摆动最大。`),
 k:()=>tx(`k is the number of answers per question that are averaged, ${S.k} here.`,`k 是每道题回答并取平均的次数，这里是 ${S.k}。`),
 q:()=>tx(`q is the share of questions only one of the two models gets right, ${fixd(mode==='guided'?P4.q:S.q,2)} here (assumed).`,`q 是一对一错的题的占比，这里是 ${fixd(mode==='guided'?P4.q:S.q,2)}（假设值）。`),
 c:()=>tx(`c is the share of coin-flip questions, ${fixd(S.c,2)} here (assumed).`,`c 是时对时错的题的占比，这里是 ${fixd(S.c,2)}（假设值）。`)
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]()).replace(/p\(1 − p\)/g,A.ui.math('p(1-p)'))}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function ctlLockMsg(k){
  if(mode==='challenge') return tx('Fixed during the challenge: change the plan in the task card.','挑战中固定不变：请在任务卡里改方案。');
  if(mode!=='guided') return '';
  if(k==='n'&&isP('size')) return ''; if(k==='k'&&isP('k')) return '';
  if(isP('pair')) return tx('Fixed in this part: pick a way to compare in the card instead.','这一部分固定不变：请在卡片里选比较方法。');
  return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='prod') return tx('Click to jump to “In production” below the lab.','点击跳到页面下方的“生产中怎么做”。');
  if(!CTLN[k]) return '';
  const m=ctlLockMsg(k); if(m) return m;
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
const ctlEl=k=>({n:$('#evc-ic-n',app)||$('#evc-ctl-n',app),pA:$('#evc-ctl-pA',app),k:$('#evc-ic-k',app)||$('#evc-ctl-k',app),q:$('#evc-ctl-q',app),c:$('#evc-ctl-c',app)}[k]);
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#evc-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#evc-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='prod'){const c=$('#evc-prod-card',app); if(c){A.tip.hide(); A.ui.jumpToEl(c);} return;}
  if(!CTLN[k]) return;
  if(ctlLockMsg(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=ctlEl(k); if(el&&el.getClientRects().length) A.ui.jumpToEl(el); else if(src) A.tip.show(src,`<div>${esc(tx('This setting is hidden with the current choices.','按现在的选择，这个设置不显示。'))}</div>`);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S=mode==='sandbox'?{...PRESETS[0].s}:{...BASE}; render();}); return;}
  if(a==='submit'){chSubmitted=true; if(chEval(S.n,S.paired,S.k).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='play'){if(playing) pause(); else play(); A.tip.quiet(); return;}
  if(a==='to-q'){const q=$('#evc-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#evc-ic-lv',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#evc-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#evc-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(!k) return; const v=+e.target.value;
    if(k==='n') bySettings(e.target,()=>setState({n:NS[v]}));
    else if(k==='pA') bySettings(e.target,()=>setState({pA:v/1000}));
    else if(k==='pB') bySettings(e.target,()=>setState({pB:v/1000}));
    else if(k==='q') bySettings(e.target,()=>setState({q:v/100}));
    else if(k==='c') bySettings(e.target,()=>setState({c:v/20}));
    else if(k==='scrub'){pause(); pileM=v; if(v>=RD.K&&!pileDone){finishPile(); return;} drawPile(axisRange());}});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#evc-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#evc-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>typeof s.showMe==='function'?s.showMe():setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#evc-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('pair')){const f=()=>byLearner(()=>setLv(d.lever)); if(innerWidth<=760) holdPlace(b,f); else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='k') bySettings(b,()=>setState({k:+d.v}));
      else if(k==='paired') bySettings(b,()=>setState({paired:+d.v}));
      else if(k==='spd'){spd=d.v; lastT=0; drawPlayer();}
      else if(k==='chn') byLearner(()=>setState({n:+d.v}));
      else if(k==='chm') byLearner(()=>setState({paired:+d.v}));
      else if(k==='chk') byLearner(()=>setState({k:+d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#evc-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#evc-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the owner's outline, approved 2026-10-12 07:13 UTC): four sections read before the guided parts, text and formulas only.
   They give the why and the mechanism (sampling the questions, the variance of a difference, the law of total variance); how large
   each effect is stays with the guided parts, so no number of the lab's scenarios appears here (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const PM=A.ui.math;
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${h}</p>`;}).join('')});
// formulas on their own line: separate formulas side by side, wrapping under each other in a narrow column (one:true: the pieces of one
// formula, kept close, wrapping only at a + or −), so nothing scrolls sideways
const PFX=(xs,one)=>`<div class="evc-prfx${one?' one':''}">${xs.map(x=>`<span>${PM(x)}</span>`).join('')}</div>`;
const pH='\\hat{p}';
A.registerPrimer(ID,{min:6,sections:[
 {h:()=>tx('Using a benchmark score to decide whether to switch models','用 benchmark 分数决定换不换模型'),parts:[
   prP(()=>tx('Every time a model changes, someone has to decide whether to put the new version into production. It may have new training data, fine-tuning (training an existing model a little further on a small set of data) or quantization (storing the weights as numbers with fewer bits, to save memory and run faster), or it may simply be a newly released model.',
              '模型每改动一次，都要有人决定要不要把新版本换上线：可能是加了新的训练数据，做了微调（在已有模型上用一小批数据接着训练），做了量化（把权重换成位数更少的数来存，省显存、跑得更快），也可能干脆是新发布的模型。'),
       ()=>tx(`That decision usually rests on a benchmark score. A benchmark is a fixed set of ${PM('n')} questions whose answers are graded automatically: a coding question runs the model’s code against prepared unit tests and counts as right only if every test passes; a maths question compares only the final answer. The share of questions the model gets right is its score.`,
              `这个决定通常看 benchmark 上的分数。benchmark 是一组固定的 ${PM('n')} 道题，答案由程序自动判对错：编程题把模型写的代码拿去跑事先写好的单元测试，全部通过才算对；数学题只比对最后的答案。模型答对的比例，就是它的分数。`),
       ()=>tx(`The decision is made on the gap: a few points up, switch; a few points down, keep the old one. But a score comes from these ${PM('n')} questions only. Is a gap of a few points a real change in the model, or did this batch of questions just happen to suit it? That takes three answers: how far one score can be from the model’s true level, how large the difference of two scores must be before it is more than the luck of the questions drawn, and how much the randomness of each answer adds.`,
              `决定靠的是分数差：新版本高出几分就换，低几分就不换。可分数只是在这 ${PM('n')} 道题上考出来的。几分的差距，是模型真的变了，还是这批题恰好对它有利？这需要回答三个问题：一个分数离模型的真实水平能有多远，两个分数之差要多大才不是抽题造成的偶然，以及模型每次回答自带的随机性又添了多少。`))]},
 {h:()=>tx('A score is one estimate of the true accuracy (standard error)','分数是对真实正确率的一次估计（标准误差）'),parts:[
   prP(()=>tx(`The ${PM('n')} questions of a benchmark can be seen as a sample from a large pool of similar questions: of the same kinds and the same difficulty there are far more than ${PM('n')}. On that pool the model has a true accuracy ${PM('p')}, its true level. The decision is really about ${PM('p')}; the benchmark score, written ${PM(pH)}, is one estimate of ${PM('p')} from the ${PM('n')} questions that happened to be drawn.`,
              `benchmark 里的 ${PM('n')} 道题，可以看成从一大堆同类题里抽出的样本：同样的题型、同样的难度，能出的题远不止 ${PM('n')} 道。模型在这一大堆题上有一个真实正确率 ${PM('p')}，也就是它的真实水平。做决定真正关心的是 ${PM('p')}；benchmark 分数 ${PM(pH)} 只是用碰巧抽到的这 ${PM('n')} 道题对 ${PM('p')} 做的一次估计。`),
       ()=>tx(`Write ${PM('X_i = 1')} if question ${PM('i')} is answered right and ${PM('X_i = 0')} if not. The questions are drawn at random, so the ${PM('X_i')} are independent and each is ${PM('\\text{Bernoulli}(p)')}. The score is their average; its expectation is ${PM('p')}, and its variance falls as the number of questions ${PM('n')} grows:`,
              `第 ${PM('i')} 道题答对记 ${PM('X_i = 1')}，答错记 ${PM('X_i = 0')}。题是随机抽的，所以各个 ${PM('X_i')} 相互独立，都服从 ${PM('\\text{Bernoulli}(p)')}。分数是它们的平均，期望等于 ${PM('p')}，方差随题数 ${PM('n')} 增大而变小：`),
       ()=>PFX([`${pH}=\\frac{1}{n}\\sum_{i=1}^{n}X_i`,`\\mathbb{E}[${pH}]=p`,`\\mathrm{Var}(${pH})=\\frac{p(1-p)}{n}`])),
   prP(()=>tx(`The standard deviation of ${PM(pH)} is called the standard error (SE). ${PM('p')} is unknown, so ${PM(pH)} takes its place:`,
              `${PM(pH)} 的标准差叫标准误差（standard error，SE）。${PM('p')} 不知道，就用 ${PM(pH)} 代替：`),
       ()=>PFX([`\\text{SE}=\\frac{\\sqrt{${pH}(1-${pH})}}{\\sqrt{n}}`]),
       ()=>tx(`${PM(pH)} is an average of ${PM('n')} independent variables, so by the central limit theorem it is close to normally distributed, and the interval ${PM(`${pH}\\pm 1.96\\cdot\\text{SE}`)} covers ${PM('p')} about 95% of the time: the 95% confidence interval. Drawn on the score axis, it is the score’s error bar. ${PM('n')} sits under the square root in the denominator of SE, so the width of the error bar is inversely proportional to ${PM('\\sqrt{n}')}.`,
              `${PM(pH)} 是 ${PM('n')} 个独立变量的平均，按中心极限定理近似服从正态分布，所以区间 ${PM(`${pH}\\pm 1.96\\cdot\\text{SE}`)} 大约有 95% 的机会盖住 ${PM('p')}，这就是 95% 置信区间。把它画在分数轴上，就是这个分数的误差线。SE 的分母里，${PM('n')} 在根号下，所以误差线的宽度和 ${PM('\\sqrt{n}')} 成反比。`))]},
 {h:()=>tx('To compare two models, take per-question differences and look at the variance of the difference','比较两个模型：逐题取差值，看差值的方差'),parts:[
   prP(()=>tx(`Call the model in production A and the new one B. The decision is not how precise one score is but how much better B is, so the quantity to estimate is the difference of the two scores, ${PM(`d=${pH}_B-${pH}_A`)}. Its variance is`,
              `记线上的模型为 A，新模型为 B。要回答的不是一个分数有多准，而是 B 比 A 好多少，所以要估计的是两个分数之差 ${PM(`d=${pH}_B-${pH}_A`)}。它的方差是`),
       ()=>PFX([`\\mathrm{Var}(d)=\\mathrm{Var}(${pH}_A)+\\mathrm{Var}(${pH}_B)`,`-\\,2\\,\\mathrm{Cov}(${pH}_A,${pH}_B)`],true),
       ()=>tx(`If A and B each took their own, independently drawn questions, the two scores would be uncorrelated, the covariance would be 0, and the variance of the difference would be the two variances added. In a real comparison both models usually answer the same questions, and then their results are correlated: a question A gets wrong is often one B gets wrong too, and when the drawn questions are harder overall, both scores come out lower together. So ${PM('\\mathrm{Cov}>0')}. The part by which both scores move up or down together does not enter ${PM(`d=${pH}_B-${pH}_A`)}, so ${PM('\\mathrm{Var}(d)')} is smaller than the two variances added.`,
              `如果 A 和 B 各考一批独立抽出的题，两个分数互不相关，协方差是 0，差值的方差就是两个方差相加。实际比较时，两个模型通常做的是同一批题，这时它们的对错是相关的：A 答错的题，B 往往也答错；抽到的题整体偏难，两个分数会一起偏低。所以 ${PM('\\mathrm{Cov}>0')}。两个分数一起偏高或偏低的那部分，在相减时消掉了，不会进入 ${PM(`d=${pH}_B-${pH}_A`)}，因此 ${PM('\\mathrm{Var}(d)')} 比两个方差相加要小。`),
       ()=>tx(`The covariance need not be estimated on its own. On the same questions, ${PM('d')} is exactly the average of the per-question differences: with ${PM('D_i=X_i^B-X_i^A')}, ${PM('d=\\frac{1}{n}\\sum_i D_i')}. The ${PM('D_i')} of different questions are independent, so`,
              `协方差不必单独去估。在同一批题上，${PM('d')} 正好是逐题差值的平均：令 ${PM('D_i=X_i^B-X_i^A')}，就有 ${PM('d=\\frac{1}{n}\\sum_i D_i')}。不同题的 ${PM('D_i')} 相互独立，所以`),
       ()=>PFX(['\\mathrm{Var}(d)=\\frac{\\mathrm{Var}(D)}{n}']),
       ()=>tx(`Here ${PM('\\mathrm{Var}(D)')} is the variance of one question’s difference, estimated by the sample variance of the ${PM('n')} differences ${PM('D_i')}; since ${PM('\\mathrm{Var}(D)=\\mathrm{Var}(X^B)+\\mathrm{Var}(X^A)-2\\,\\mathrm{Cov}(X^A,X^B)')}, the per-question covariance is already in it. The square root of ${PM('\\mathrm{Var}(D)/n')} is the standard error of ${PM('d')}, written ${PM('\\text{SE}(d)')}, and ${PM('d\\pm 1.96\\cdot\\text{SE}(d)')} is its error bar. Only when that error bar stays clear of 0 is the gap between B and A beyond the swing that comes from which questions were drawn.`,
              `其中 ${PM('\\mathrm{Var}(D)')} 是一道题的差值的方差，用这 ${PM('n')} 个 ${PM('D_i')} 的样本方差来估计；因为 ${PM('\\mathrm{Var}(D)=\\mathrm{Var}(X^B)+\\mathrm{Var}(X^A)-2\\,\\mathrm{Cov}(X^A,X^B)')}，两个模型逐题的协方差已经含在里面。${PM('\\mathrm{Var}(D)/n')} 开方就是 ${PM('d')} 的标准误差，记作 ${PM('\\text{SE}(d)')}；${PM('d\\pm 1.96\\cdot\\text{SE}(d)')} 就是 ${PM('d')} 的误差线。这条误差线不碰 0，才能说 B 和 A 的差距超出了抽题带来的摆动。`))]},
 {h:()=>tx('Two kinds of randomness: which questions were drawn, and how each is answered','两种随机性：抽到哪些题，每次怎么回答'),parts:[
   prP(()=>tx(`So far a question asked of a model has always had the same result, and ${PM('X_i')} was random only through which question was drawn. But a model writes its answer one token (a word or a piece of a word) at a time, and each token is picked at random according to its probability; ${A.shell.labLink?A.shell.labLink('sampling','temperature'):'temperature'} is the setting that controls how random that pick is, and in production it is usually above 0. So the same question asked twice can be answered right once and wrong once.`,
              `前面默认同一道题问模型几遍，结果都一样，${PM('X_i')} 的随机只来自抽到了哪道题。可模型的回答是一个 token（一个词或词的一部分）一个 token 往下写的，每一步写哪个 token 都按概率随机挑选；${A.shell.labLink?A.shell.labLink('sampling','temperature'):'temperature'} 是控制这种随机程度的参数，线上通常设成大于 0。所以同一道题问两遍，可能一次答对、一次答错。`),
       ()=>tx(`Each question then has its own chance of being answered right, ${PM('p_i')}: some questions are right almost every time, some wrong almost every time, some right only part of the time. The average of ${PM('p_i')} over all questions is the true accuracy ${PM('p')}. A question’s result is random in two layers: which question is drawn sets ${PM('p_i')}, and answering with chance ${PM('p_i')} sets whether this answer is right. By the law of total variance,`,
              `于是每道题有自己的答对概率 ${PM('p_i')}：有的题几乎每次都对，有的几乎每次都错，有的时对时错。${PM('p_i')} 在所有题上的平均，就是真实正确率 ${PM('p')}。一道题的结果因此有两层随机：抽到哪道题，决定了 ${PM('p_i')}；按 ${PM('p_i')} 作答，决定这一次是对是错。按全方差公式，`),
       ()=>PFX(['\\mathrm{Var}(X_i)=\\mathrm{Var}(p_i)+\\mathbb{E}[p_i(1-p_i)]']),
       ()=>tx(`The first term is how much ${PM('p_i')} differs from question to question, so it comes from which questions were drawn; the second is the randomness of the answer once the question is fixed. Together they make exactly ${PM('p(1-p)')}, because ${PM('X_i')} still takes only the values 0 and 1, with ${PM('P(X_i=1)=\\mathbb{E}[p_i]=p')}, so it is still ${PM('\\text{Bernoulli}(p)')}. So with one answer per question, the formula ${PM(`\\mathrm{Var}(${pH})=p(1-p)/n`)} still holds, and the randomness of the answers was already in it.`,
              `第一项是题与题之间 ${PM('p_i')} 的差别，来自抽到了哪些题；第二项是题目定下以后，回答本身的随机性。两项加起来正好是 ${PM('p(1-p)')}，因为 ${PM('X_i')} 仍然只取 0 和 1，而且 ${PM('P(X_i=1)=\\mathbb{E}[p_i]=p')}，它仍服从 ${PM('\\text{Bernoulli}(p)')}。所以每道题只答一次时，${PM(`\\mathrm{Var}(${pH})=p(1-p)/n`)} 照样成立，回答的随机性已经算在里面了。`)),
   prP(()=>tx(`If each question is answered ${PM('k')} times and the average of the ${PM('k')} results is its score:`,
              `如果每道题答 ${PM('k')} 次，把 ${PM('k')} 次的平均当作这道题的得分：`),
       ()=>PFX([`\\mathrm{Var}(${pH})=\\frac{\\mathrm{Var}(p_i)}{n}+\\frac{\\mathbb{E}[p_i(1-p_i)]}{n\\,k}`]),
       ()=>tx(`${PM('k')} appears only in the denominator of the second term. However large ${PM('k')} is, ${PM(`\\mathrm{Var}(${pH})`)} stays above ${PM('\\mathrm{Var}(p_i)/n')}, a term that only ${PM('n')} makes smaller. At temperature 0 every answer to a question is the same, ${PM('p_i')} is either 0 or 1, and the second term is 0 to begin with.`,
              `${PM('k')} 只出现在第二项的分母上。${PM('k')} 再大，${PM(`\\mathrm{Var}(${pH})`)} 也不会低于 ${PM('\\mathrm{Var}(p_i)/n')}，这一项只随 ${PM('n')} 变小。temperature 为 0 时，同一道题每次的回答都一样，${PM('p_i')} 只有 0 和 1 两种，第二项本来就是 0。`))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the ways and k values tried, whether the pile of
     re-runs was watched (one still playing counts: a reload shows it whole), the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,lv,lvTried:[...lvTried],remView,kSeen:[...kSeen],
    pileDone:pileDone||playing,spd,sbTouched,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chEval(S.n,S.paired,S.k).ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'&&NS.includes(s.n)?parse(token(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, L=LV.map(x=>x.v);
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    lv=L.includes(o.lv)?o.lv:'sep'; lvTried=new Set(['sep',lv,...(Array.isArray(o.lvTried)?o.lvTried.filter(v=>L.includes(v)):[])]);
    remView=REMS.includes(o.remView)?o.remView:'pile'; kSeen=new Set([1,...(Array.isArray(o.kSeen)?o.kSeen.filter(v=>KS.includes(v)):[])]);
    pileDone=bool(o.pileDone); if(Object.keys(DUR).includes(o.spd)) spd=o.spd; sbTouched=bool(o.sbTouched);
    drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);
    playing=false; pileM=isP('pile')&&!pileDone?0:RD.K;},   // part 2 not yet watched plays its pile from the start (mount starts it)
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#evc-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#evc-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p){const s=STEPS[step]; if(s.enter&&step===0) S={...s.enter()};}
    fillRail();
    render();
    if(isP('pile')&&!pileDone&&!playing&&!relang) play();   // a language switch keeps a paused pile paused (R3.20)
  },
  leave(){clearTimeout(annT); stageStop(); pause(); root=null; app=null;},
  onLang(){if(root){const d=$('#evc-math',app), open=d&&d.open, was=playing; pause(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#evc-math',app).open=true; if(was) play();}},
  onResize(){if(app) render();},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>{if(!app) return null; const v=viewNow(); return v==='pair'?$('#evc-grid',app):(v==='ch'?$('#evc-df',app):$('#evc-ax',app));},
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#evc-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), v=viewNow();
      status.className='status '+s.cls; status.textContent=s.short;
      if(v==='ch'){const e=chEval(S.n,S.paired,S.k), i=ivl(e.r.D,e.r.h); vis.innerHTML=miniDiff(e.r); read.innerHTML=esc(tx(`${chName(S.n,S.paired,S.k)}: B − A = ${i.Ds} ± ${i.hs} points, ${fN(e.cost)} graded answers`,`${chName(S.n,S.paired,S.k)}：B − A = ${i.Ds} ± ${i.hs} 分，打分 ${fN(e.cost)} 条`)); return;}
      if(v==='pair'){const r=p4(lv==='pair'?1:0), i=ivl(r.D,r.h); vis.innerHTML=miniDiff(r); read.innerHTML=esc(tx(`1,319 problems, ${lv==='pair'?'question by question':'each on its own'}: B − A = ${i.Ds} ± ${i.hs} points`,`1,319 道题，${lv==='pair'?'逐题对比':'各算各的'}：B − A = ${i.Ds} ± ${i.hs} 分`)); return;}
      const r=cmp(S);
      if(barsHidden()){vis.innerHTML=''; read.innerHTML=esc(tx(`${fN(S.n)} problems: A ${fPct(S.pA)}, B ${fPct(S.pB)}`,`${fN(S.n)} 道题：A ${fPct(S.pA)}，B ${fPct(S.pB)}`)); return;}
      vis.innerHTML=miniBars(r); read.innerHTML=esc(tx(`${fN(S.n)} questions${S.k>1?`, ${S.k} answers each`:''}: A ${fPct(S.pA)} ± ${fPt(r.hA)}, B ${fPct(S.pB)} ± ${fPt(r.hB)} points`,`${fN(S.n)} 道题${S.k>1?`，每题答 ${S.k} 次`:''}：A ${fPct(S.pA)} ± ${fPt(r.hA)} 分，B ${fPct(S.pB)} ± ${fPt(r.hB)} 分`));}
  }
});
function miniBars(r){const rng=axisRange(), b=(p,h)=>`<i style="left:${pos((p-h)*100,rng).toFixed(1)}%;width:${(pos((p+h)*100,rng)-pos((p-h)*100,rng)).toFixed(1)}%"></i>`;
  return `<div class="evc-dk">${b(S.pA,r.hA)}${b(S.pB,r.hB)}</div>`;}
function miniDiff(r){const rng=[-4,8], l=pos(r.lo*100,rng), h=pos(r.hi*100,rng); return `<div class="evc-dk evc-dkd"><b style="left:${pos(0,rng).toFixed(1)}%"></b><i style="left:${l.toFixed(1)}%;width:${(h-l).toFixed(1)}%"></i></div>`;}

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__evc=Object.freeze({
  cmp:o=>deep(cmp({...BASE,...o})), nFor, floorH, power, H80, coverage, covMin:()=>COV.min, feasible:o=>feasible({...BASE,...o}), fixQC:o=>deep(fixQC({...BASE,...o})),
  redraws:()=>REDRAWS.slice(), inside, chEval:(n,m,k)=>deep(chEval(n,m,k)), chAll:()=>deep(chAll()), chBest:()=>deep(chBest()), drill:s=>deep(makeDrill(s)), ivl,
  consts:Object.freeze({NS:NS.slice(),KS:KS.slice(),RD:{...RD},CH:deep(CH),BASE:{...BASE},P4:{...P4},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s})))}),
  state:()=>({...S,mode,step,pred,observed:{...observed},lv,lvTried:[...lvTried],remView,chSubmitted,chReveal,pileM,pileDone,playing,spd,kSeen:[...kSeen],stageBusy:stageT!==0}),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});}
});
})();
