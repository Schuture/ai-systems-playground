/* eval-capstone (world eval): How do you evaluate an LLM assistant before and after release?
   Fidelity: Analytical. A stated world: a current model and a candidate, five categories of questions with a set true difference
   each, a judge with set preferences, set prices. Every number on the page comes from the formulas below (no random draws), so a
   plan always gives the same result. tests/eval-capstone_numbers.py recomputes every number independently.
   - The difference of a category, in points: share of items where the candidate's answer is better minus share where the current
     model's is better (a tie, both right or both wrong, counts for neither side). With pass/fail checks it is the difference of the
     two pass rates. People and program checks measure the true difference exactly (the lab's assumption: the measured shares equal
     the set shares); the interval is eval-ci's paired one: h = 1.96 · sqrt((q − Δ²) / n), q = share of items with a winner.
   - The judge adds a bias b (controls off: position and length preference plus self-preference; controls on, both orders and length
     taken out: only self-preference left). Calibration with m human labels (prediction-powered inference with λ = 1, as
     ppi_py's ppi_mean_ci with alpha=0.05, lam=1): Δ̂ = Δ_judge(the n − m unchecked items) − (Δ_judge(m) − Δ_human(m)) = Δ,
     SE² = (q − Δj²)/(n − m) + (e − b²)/m, e = share of items where the judge and the person disagree (assumed: they then differ
     by one level, a tie against a winner, so Var(people − judge) = e − b²). m = n: people graded every item, SE² = (q − Δ²)/m.
   - A category's verdict: better (lower end > 0), worse (upper end < 0), not worse by more than 3 points (lower end > −3; shown from
     part 3 on), else can't tell yet. Wrong: a verdict the true difference contradicts.
   - Cost: a judge call $0.01 (both orders: two calls), a human label $2, program checks free; people label 400 items a day.
   - After release (part 6): an A/B test, 5% of 100,000 conversations a day on each model; guardrail = regenerate rate, 8% now;
     the candidate +1.2 points in writing conversations (20%) and −0.3 in the rest; two-proportion interval.
   Words (EN / ZH): candidate / 候选模型; current model / 当前模型; category / 类别; program check / 程序判分; judge / 裁判;
   people / 人工; calibration / 人工校准; guardrail / 护栏指标; part / 部分. window.__ecap is a read-only hook for the tests. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='eval-capstone';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const Z=1.96, MARGIN=0.03, BUDGET=1500, DEADLINE=3, JCALL=0.01, HUM=2, PERDAY=400;
const SIZES=[100,300,1000,3000], CALS=[0,100,300,1000], METHS=['prog','judge','human'];
/* jb: the judge's bias [controls off, on]; je: share of items where judge and person disagree [off, on] (all assumed) */
const CATS=[
 {id:'code', w:.30,d:.060, q:.20,prog:true, jb:[.01,0],   je:[.07,.05]},
 {id:'qa',   w:.25,d:.035, q:.10,prog:true, jb:[.01,0],   je:[.07,.05]},
 {id:'write',w:.20,d:-.040,q:.30,prog:false,jb:[.09,.035],je:[.20,.07]},
 {id:'tool', w:.15,d:.025, q:.10,prog:true, jb:[.01,0],   je:[.07,.05]},
 {id:'safe', w:.10,d:.012, q:.06,prog:false,jb:[.03,0],   je:[.09,.04]}];
const CAT=Object.fromEntries(CATS.map(c=>[c.id,c]));
function est(c,meth,n,ctl,cal){
  if(meth!=='judge'){const D=c.d, v=(c.q-D*D)/n; return {D,h:Z*Math.sqrt(v),se:Math.sqrt(v),Dj:null,b:0,mm:0,vj:v};}
  const b=c.jb[ctl?1:0], e=c.je[ctl?1:0], Dj=c.d+b, vj=(c.q-Dj*Dj)/n, mm=Math.min(cal,n);
  if(!mm) return {D:Dj,h:Z*Math.sqrt(vj),se:Math.sqrt(vj),Dj,b,e,mm:0,nu:n,vj};
  /* as ppi_py: the judge's mean over the items people did not check (nu of them), plus people minus judge on the checked ones */
  const nu=n-mm;
  if(!nu){const v=(c.q-c.d*c.d)/mm; return {D:c.d,h:Z*Math.sqrt(v),se:Math.sqrt(v),Dj,b,e,mm,nu:0,vj:0,vd:v};}
  const vu=(c.q-Dj*Dj)/nu, vd=(e-b*b)/mm; return {D:c.d,h:Z*Math.sqrt(vu+vd),se:Math.sqrt(vu+vd),Dj,b,e,mm,nu,vj:vu,vd};
}
function verdict(D,h,mg){const lo=D-h, hi=D+h; if(lo>0) return 'better'; if(hi<0) return 'worse'; if(mg&&lo>-MARGIN) return 'ni'; return 'unk';}
const wrongV=(v,d)=>(v==='better'&&d<=0)||(v==='worse'&&d>=0)||(v==='ni'&&d<=-MARGIN);
function costOf(meth,n,ctl,cal){if(meth==='prog') return {j:0,h:0,H:0,calls:0}; if(meth==='human') return {j:0,h:n*HUM,H:n,calls:0};
  const mm=Math.min(cal,n), calls=n*(ctl?2:1); return {j:calls*JCALL,h:mm*HUM,H:mm,calls};}
/* a plan: {m:{cat:meth}, n:{cat:size}, ctl:0|1, cal}; mg: the 3-point margin is in use (part 3 on) */
function evalPlan(P,mg){
  if(mg===undefined) mg=true;
  const rows=CATS.map(c=>{const meth=(c.prog||P.m[c.id]!=='prog')?P.m[c.id]:'judge', n=P.n[c.id], e=est(c,meth,n,P.ctl,P.cal), v=verdict(e.D,e.h,mg), co=costOf(meth,n,P.ctl,P.cal);
    return {id:c.id,meth,n,...e,lo:e.D-e.h,hi:e.D+e.h,v,wrong:wrongV(v,c.d),truth:c.d,cost:co};});
  const D=rows.reduce((s,r,i)=>s+CATS[i].w*r.D,0), se=Math.sqrt(rows.reduce((s,r,i)=>s+(CATS[i].w*r.se)**2,0)), h=Z*se;
  const tv=verdict(D,h,false), costJ=rows.reduce((s,r)=>s+r.cost.j,0), costH=rows.reduce((s,r)=>s+r.cost.h,0), H=rows.reduce((s,r)=>s+r.cost.H,0);
  const days=Math.max(1,Math.ceil(H/PERDAY)), vs=rows.map(r=>r.v);
  const decision=vs.includes('worse')?'block':(vs.includes('unk')?'unk':(vs.includes('better')?'ship':'unk'));
  const w=rows.find(r=>r.id==='write'), nWrong=rows.filter(r=>r.wrong).length;
  return {rows,total:{D,se,h,lo:D-h,hi:D+h,v:tv},costJ,costH,cost:costJ+costH,H,days,decision,caught:w.v==='worse',nWrong};
}
/* the challenge's goal */
function chOk(e){return e.rows.every(r=>r.v!=='unk'&&!r.wrong)&&e.caught&&e.rows.some(r=>r.v==='better');}
function chPass(e){return chOk(e)&&e.cost<=BUDGET+1e-9&&e.days<=DEADLINE;}
const catOpts=c=>[].concat(...(c.prog?METHS:['judge','human']).map(m=>SIZES.map(n=>[m,n])));
const CHS={};
/* every plan a reader can reach, by judge setting: the cheapest one that meets the goal apart from money and days */
function chSearch(){
  if(CHS.done) return CHS;
  const combos=[]; let nPass=0, best=null;
  [0,1].forEach(ctl=>CALS.forEach(cal=>{
    const per=CATS.map(c=>catOpts(c).map(([m,n])=>{const e=est(c,m,n,ctl,cal), v=verdict(e.D,e.h,true), co=costOf(m,n,ctl,cal);
      return {m,n,ok:v!=='unk'&&!wrongV(v,c.d)&&(c.id!=='write'||v==='worse'),better:v==='better',cost:co.j+co.h,H:co.H};}).filter(o=>o.ok));
    let cb=null, cnt=0;
    const rec=(i,acc,cost,H,bet)=>{
      if(i===CATS.length){if(!bet) return; const days=Math.max(1,Math.ceil(H/PERDAY)), pass=cost<=BUDGET+1e-9&&days<=DEADLINE;
        if(pass) cnt++;
        if(!cb||cost<cb.cost-1e-9) cb={cost,days,plan:acc.slice(),pass};
        return;}
      per[i].forEach(o=>{acc.push(o); rec(i+1,acc,cost+o.cost,H+o.H,bet||o.better); acc.pop();});};
    rec(0,[],0,0,false);
    nPass+=cnt;
    if(cb){const plan={m:{},n:{},ctl,cal}; cb.plan.forEach((o,i)=>{plan.m[CATS[i].id]=o.m; plan.n[CATS[i].id]=o.n;}); combos.push({ctl,cal,cost:cb.cost,days:cb.days,pass:cb.pass,plan});}
    combos.forEach(x=>{if(x.ctl===ctl&&x.cal===cal&&x.pass&&(!best||x.cost<best.cost-1e-9)) best=x;});
  }));
  Object.assign(CHS,{done:true,combos,nPass,best});
  return CHS;
}
/* the answer the page shows: the cheapest plan, with the program-checked categories at 1,000 items */
const CH_ANSWER={m:{code:'prog',qa:'prog',write:'judge',tool:'prog',safe:'human'},n:{code:300,qa:300,write:3000,tool:300,safe:300},ctl:1,cal:300};
/* after release: an A/B test (part 6) */
const AB={perDay:100000,share:.05,sw:.20,p0:.08,dW:.012,dO:-.003,days:[1,3,7]};
const SH=()=>`${Math.round(AB.share*100)}%`, WL=()=>fint(Math.round(-CAT.write.d*100));
const JE=()=>{const v=CATS.filter(c=>c.id!=='write').flatMap(c=>c.je).map(x=>Math.round(x*100)); return tx(`${Math.min(...v)}% to ${Math.max(...v)}%`,`${Math.min(...v)}% 到 ${Math.max(...v)}%`);};
function abRows(days){
  const arm=AB.perDay*AB.share*days, r=(p1,n)=>{const p0=AB.p0, D=p1-p0, se=Math.sqrt((p0*(1-p0)+p1*(1-p1))/n); return {D,se,h:Z*se,lo:D-Z*se,hi:D+Z*se,n,p0,p1,alarm:D-Z*se>0};};
  const pT=AB.sw*(AB.p0+AB.dW)+(1-AB.sw)*(AB.p0+AB.dO);
  return {tot:r(pT,arm),write:r(AB.p0+AB.dW,arm*AB.sw),other:r(AB.p0+AB.dO,arm*(1-AB.sw)),arm};
}

/* =====================================================================
   words and formats
   ===================================================================== */
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fint=A.fmt.int;
const MINUS='−';
const sgn=(x,d)=>{const s=fixd(Math.abs(x),d); return (x<0&&+s!==0?MINUS:(x>0&&+s!==0?'+':''))+s;};
/* a bound in points: one decimal, two when one decimal would sit on 0 or on −3 (R6.10) */
const fB=x=>{const p=x*100; return sgn(p,(Math.abs(p)<0.05||Math.abs(p+3)<0.05)?2:1);};
const fD=x=>sgn(x*100,1);
const fA=x=>fixd(Math.abs(x*100),1);   // a size in points, no sign: “2.4 points higher”
const usd=v=>'$'+fint(Math.round(v));
/* a real evaluation, where the measured difference wanders around the truth with the same standard error: the chance that the
   interval lies wholly below 0 (normal approximation; the erf is Abramowitz and Stegun 7.1.26) */
const erf=x=>{const t=1/(1+0.3275911*Math.abs(x)), y=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x); return x<0?-y:y;};
const Phi=x=>0.5*(1+erf(x/Math.SQRT2));
const pCatch=r=>Phi(-r.D/r.se-Z);
const pct=p=>`${Math.round(p*100)}%`;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const DEF=(t,tip)=>A.ui.def(t,tip);
const MI=t=>A.ui.math(t);
const SN=id=>{const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[id]; return l&&l.s?L(l.s):id;};   // a lab's short name, in plain text
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(!l) return esc(id); if(C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return A.shell.labLink?A.shell.labLink(id):esc(id);};
const J=(k,t)=>`<span class="linkbtn ecap-j" data-jump="${k}">${t}</span>`;
const PIN={ppi:'3d1f0c668444907b39bd045cb9dd38e479ce7dd6',lme:'d6de81643928d653435c431bae19945d41d32520',fc:'587d5cfa1609a43d192cedb8441cac3c17db105d'};
const SRC={
 ppi:`https://github.com/aangelopoulos/ppi_py/blob/${PIN.ppi}/ppi_py/ppi.py`,
 ppiR:`https://github.com/aangelopoulos/ppi_py/blob/${PIN.ppi}/README.md`,
 lme:`https://github.com/EleutherAI/lm-evaluation-harness/blob/${PIN.lme}/lm_eval/api/metrics.py`,
 fc:`https://github.com/lm-sys/FastChat/blob/${PIN.fc}/fastchat/llm_judge/common.py`,
 ppiP:'https://arxiv.org/abs/2301.09633', ppi2:'https://arxiv.org/abs/2311.01453', zheng:'https://arxiv.org/abs/2306.05685', lc:'https://arxiv.org/abs/2404.04475'
};
const CN={code:{en:'Coding',zh:'编程'},qa:{en:'Factual Q&A',zh:'事实问答'},write:{en:'Chinese writing',zh:'中文写作'},tool:{en:'Tool use',zh:'工具调用'},safe:{en:'Safety refusals',zh:'安全拒答'}};
const CHK={code:{en:'unit tests',zh:'跑单元测试'},qa:{en:'compare with the reference answer',zh:'对照标准答案'},tool:{en:'check the final state',zh:'检查最后的状态'}};
const MN={prog:{en:'program check',zh:'程序判分'},judge:{en:'judge',zh:'裁判'},human:{en:'people',zh:'人工'}};
const VN={better:{en:'better',zh:'变好了'},worse:{en:'worse',zh:'变差了'},ni:{en:'not worse by over\u00a03',zh:'没有变差超过 3\u00a0分'},unk:{en:'can’t tell yet',zh:'还看不出'}};
const DN={ship:{en:'Can ship',zh:'能上线'},block:{en:'Cannot ship',zh:'不能上线'},unk:{en:'No verdict yet',zh:'还不能下结论'},tot:{en:'Looks better',zh:'总分变好了'},totU:{en:'Can’t tell',zh:'总分看不出'}};
const nItems=n=>tx(`${fint(n)} items`,`${fint(n)} 题`);
function planLine(r,P){
  if(r.meth==='prog') return tx(`program check, ${fint(r.n)} items`,`程序判分，${fint(r.n)} 题`);
  if(r.meth==='human') return tx(`people, ${fint(r.n)} items`,`人工判 ${fint(r.n)} 题`);
  const c=P.ctl?tx('both orders, length-corrected','两种顺序、按长度校正'):tx('judged once','判一次');
  return tx(`judge, ${fint(r.n)} items, ${c}${r.mm?`, ${fint(r.mm)} checked by people`:''}`,`裁判 ${fint(r.n)} 题，${c}${r.mm?`，人工校准 ${fint(r.mm)} 题`:''}`);
}
const TIP={
 diff:()=>tx('Items where both answers are right, or both wrong, count for neither side. When a program marks each answer right or wrong, the difference is the gap between the two models’ pass rates.',
             '两个回答都对或都错的题，两边都不算。用程序判对错时，差就是两个模型正确率的差。'),
 ci:()=>tx(`If the true difference were exactly the measured one, a new set of items of the same size and kind would give a difference inside this range about 95 times in 100. A real evaluation gives only one measured difference, and a range drawn this way from it covers the true difference about 95% of the time. The lab “${SN('eval-ci')}” explains it.`,
           `假如真实的差正好等于量出来的差，换一批同样多的同类题再评一次，新量出的差 100 次里约有 95 次落在这个范围里。真实评测只量得到一次，按这种方法画出的区间，大约有 95% 的机会盖住真实的差。“${SN('eval-ci')}”那个 Lab 专门讲它。`),
 traffic:()=>tx(`The share of users’ questions that fall in each category: ${CATS.map(c=>`${c.id==='write'?CN[c.id].en:CN[c.id].en.replace(/^./,x=>x.toLowerCase())} ${Math.round(c.w*100)}%`).join(', ')} (set for this lab).`,
                `用户的问题落在每一类的比例：${CATS.map(c=>`${CN[c.id].zh} ${Math.round(c.w*100)}%`).join('，')}（这个 Lab 设定的）。`),
 safe:()=>tx('Requests the assistant should refuse, such as instructions for causing harm. An item is right when the answer refuses such a request and helps with a harmless one.',
             '助手应该拒绝的请求，比如教人害人的步骤。拒绝了这类请求、又没有拒绝无害请求的回答算对。'),
 prog:()=>tx('A program decides whether each answer is right: coding runs unit tests, factual Q&A compares with the reference answer, tool use checks the final state. It costs almost nothing and has no taste of its own; this lab counts it as free.',
             '由程序判断每个回答对不对：编程跑单元测试，事实问答对照标准答案，工具调用检查最后的状态。几乎不花钱，也没有自己的偏好；这个 Lab 把它算作不花钱。'),
 q:()=>tx('q is the share of items where one side’s answer is better (a winner). The more items have a winner, the more each result swings, and the wider the interval.',
          'q 是有一方答得更好的题所占的比例。有输赢的题越多，每题的结果越分散，区间越宽。'),
 margin:()=>tx('How much worse a category may get and still count as acceptable: 3 points in this lab, the dashed line in the chart. An interval wholly right of that line gets this verdict. This kind of test is called a non-inferiority test.',
               '一类题最多能接受变差多少：这个 Lab 定为 3 分，就是图里的虚线。区间整个在虚线右边，就得到这个结论。这种检验叫非劣效检验。'),
 judge:()=>tx(`An LLM shown one question and both answers, asked which is better or whether they tie. In this lab one call costs $0.01. The lab “${SN('llm-judge')}” shows its preferences.`,
              `给另一个 LLM 看一道题和两个回答，问它哪个更好，或者一样好。这个 Lab 里每次调用 $0.01。“${SN('llm-judge')}”那个 Lab 讲它的偏好。`),
 self:()=>tx('A judge may prefer answers it wrote itself (Zheng et al., 2023, call this self-enhancement bias, and say their data is too limited to settle whether it exists). This lab sets the judge and the candidate in the same model family, and the preference adds '+fixd(CAT.write.jb[1]*100,1)+' points to Chinese writing. Swapping the order or correcting for length leaves it in place.',
             '裁判可能偏向自己写的回答（Zheng 等 2023 把它叫 self-enhancement bias，论文说数据还不够下定论）。这个 Lab 设定裁判和候选模型来自同一个模型家族，这个偏好给中文写作加了 '+fixd(CAT.write.jb[1]*100,1)+' 分。换顺序、按长度校正都去不掉它。'),
 len:()=>tx('How the length correction works: fit how the judge’s verdict changes with the length difference between the two answers, then report the verdict as if both answers were equally long. AlpacaEval’s length-controlled win rate works this way (Length-Controlled AlpacaEval, Dubois et al., 2024, arXiv 2404.04475). This lab assumes the correction removes the length preference completely.',
            '按长度校正的做法：拟合裁判的判断怎样随两个回答的长度差变化，再按两个回答一样长时的情形报告结果。AlpacaEval 的长度控制胜率就是这样算的（Length-Controlled AlpacaEval，Dubois 等 2024，arXiv 2404.04475）。这个 Lab 假设这样校正能把长度的偏好完全去掉。'),
 truth:()=>tx('The true difference set for this lab. In a real evaluation nobody knows it; here it shows when a plan’s verdict is wrong.',
              '这个 Lab 设定的真实的差。真实的评测里没人知道它；这里用它看出哪个结论错了。'),
 ppi:()=>tx('Prediction-powered inference (Angelopoulos et al., 2023): many items graded by a model, a few also by people; the people’s items correct the model’s bias, and the interval counts both sources of error. The ppi_py library implements it.',
            'Prediction-powered inference（Angelopoulos 等，2023）：大量的题让模型判，其中一小批再请人判；人判的那批用来修正模型的偏差，区间同时算进两部分的误差。ppi_py 库实现了这种做法。'),
 guard:()=>tx('When its interval lies wholly past the alarm line, the alarm goes off and the rollout stops. Here the alarm line is 0: the candidate must be no worse than the current model.',
              '它的区间整个越过报警线，就报警，停止放量。这里的报警线是 0：候选不能比当前模型差。'),
 regen:()=>tx('The share of conversations in which the user clicks “regenerate” for a new answer. A higher rate suggests users are less satisfied. In this lab it is 8% with the current model.',
              '用户点“重新生成”要一个新回答的对话所占的比例。比例越高，说明用户越不满意。这个 Lab 里当前模型是 8%。'),
 ab:()=>tx('An A/B test sends a random part of the traffic to the candidate and an equal part to the current model, and compares the two groups on the same metrics.',
           'A/B 实验把一部分流量随机交给候选模型，另一份同样大的流量留给当前模型，再在同样的指标上比较两组。')
};

/* =====================================================================
   module state
   ===================================================================== */
const PLAN0=()=>({m:{code:'prog',qa:'prog',write:'human',tool:'prog',safe:'human'},n:{code:1000,qa:1000,write:1000,tool:1000,safe:1000},ctl:0,cal:0});
const CH_START=()=>({m:{code:'prog',qa:'prog',write:'judge',tool:'prog',safe:'judge'},n:{code:300,qa:300,write:300,tool:300,safe:300},ctl:0,cal:0});
let S=PLAN0(), mode='guided', step=0, pred=null, observed={}, root=null, app=null, annT=0, userAct=false, sbTouched=false, wantObs=false;
let split=false, sizeTried=new Set(), sizeClicked=false, p4Tried=new Set(), p4Sel='human', calTried=new Set(), abDays=1, abTried=new Set();
let chSubmitted=false, chReveal=false, saved=null, drillSeed=1, drillOpen=false;
const shownPrev={};
const clone=P=>({m:{...P.m},n:{...P.n},ctl:P.ctl,cal:P.cal});

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'ship',t:{en:'Yes: the total is clearly higher',zh:'能：总分确实变高了'}},
  {v:'split',t:{en:'Not yet: check each category, one may be worse',zh:'还不能下结论：先分开看，可能有一类变差了'}},
  {v:'noise',t:{en:'No: 2.4 points is too small, it may be noise',zh:'不能：2.4 分太少，可能只是噪声'}}];
const PRED_RIGHT='split';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const curId=()=>mode==='guided'?STEPS[step].id:'';
const totalOnly=()=>isP('mission')||(isP('split')&&!split);
const marginOn=()=>mode!=='guided'||step>=SIX('size');
const truthOn=()=>mode==='sandbox'||(mode==='challenge'&&chSubmitted)||(mode==='guided'&&step>=SIX('judge')&&!isP('online'));
const onlineView=()=>isP('online');
const chHide=()=>mode==='challenge'&&!chSubmitted;
const E=()=>evalPlan(S,marginOn());
/* part 4's four ways to grade Chinese writing */
const P4={human:{m:'human',n:1000,ctl:0},once:{m:'judge',n:3000,ctl:0},both:{m:'judge',n:3000,ctl:1}};
const P4T={human:{en:'People grade 1,000 items',zh:'请人判 1,000 题'},once:{en:'Judge 3,000 items, once each',zh:'裁判判 3,000 题，每题判一次'},both:{en:'Judge 3,000 items in both orders, length-corrected',zh:'裁判判 3,000 题，两种顺序都判，按长度校正'}};
const P4D={human:{en:'the initial setup, $2 an item',zh:'初始设置，每题 $2'},once:{en:'2,000 new items narrow the interval',zh:'多准备了 2,000 题，区间更窄'},both:{en:'two calls an item',zh:'每题判两次'}};
function p4Plan(k){const P=PLAN0(), o=P4[k]; P.m.write=o.m; P.n.write=o.n; P.ctl=o.ctl; P.cal=0; return P;}
/* a tried way's verdict on writing: the full verdict name on one line (R6.4), and “wrong” under it in red, as on the chart's rows */
const p4Res=k=>{const r=evalPlan(p4Plan(k)).rows.find(x=>x.id==='write'); return `<span class="ecap-nw">${esc(L(VN[r.v]))}</span>`+(r.wrong?`<em class="wrong">${tx('wrong','看错了')}</em>`:'');};
const p4Ctl=()=>`<div class="ecap-lv" id="ecap-ic-p4"><p class="ecap-lvh">${tx('Pick a way to grade writing, and see whether the verdict is right and what it costs.','选一种判写作的办法，看看结论对不对、花多少钱。')}</p>`+A.ui.levers({aria:tx('How to grade Chinese writing, pick one','中文写作怎样判，选一个'),value:p4Sel,
  options:Object.keys(P4).map(k=>({v:k,title:L(P4T[k]),desc:L(P4D[k]),result:p4Tried.has(k)||k===p4Sel?p4Res(k):''}))})+`</div>`;
const GSIZES=[100,1000,3000];   // part 3 leaves out 300, so the challenge's 300 items for safety is the reader's own inference
const sizeCtl=id=>A.ui.seg({id,key:'gsize',label:tx('Items in each\u00a0category','每类的题数'),options:GSIZES.map(n=>[String(n),fint(n)]),value:String(S.n.code),inline:true});
const splitCtl=id=>A.ui.seg({id,key:'split',label:tx('Show','显示'),options:[['0',tx('the total only','只看总分')],['1',tx('each category','按类别分开看')]],value:split?'1':'0',inline:true});
const calCtl=id=>A.ui.seg({id,key:'gcal',label:tx('Items people also grade','人工校准的题数'),options:CALS.map(n=>[String(n),fint(n)]),value:String(S.cal),inline:true});
const abCtl=id=>A.ui.seg({id,key:'abd',label:tx('Days of the A/B test','A/B 实验的天数'),options:AB.days.map(d=>[String(d),tx(`${d} day${d>1?'s':''}`,`${d} 天`)]),value:String(abDays),inline:true});
const toScen=()=>A.shell.toScenarios?A.shell.toScenarios():`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="sandbox">${tx('See the scenarios','去看应用场景')}</button></div>`;
const toChal=()=>A.shell.toChallenge?A.shell.toChallenge():`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="challenge">${tx('Take the challenge','去做挑战')}</button></div>`;
const tot0=()=>evalPlan(PLAN0(),false).total;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},gate:()=>!!pred,
  lead:()=>tx(`The candidate’s total is ${fA(tot0().D)} points higher. Can it replace the current model?`,`候选模型总分高 ${fA(tot0().D)} 分，能替换当前模型吗？`),
  enter:()=>{split=false; return PLAN0();},
  body:()=>{const t=tot0(), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}"><span>${esc(L(o.t))}</span></button>`).join('')}</div>`;
   /* English runs longer than Chinese, and the page's title takes two lines: so that the question and its three options are on the first
      screen at 1440x900, how the total is made (last week's items, the difference, the weighting) follows the options; the same content */
   return tx(`<p>You own the evaluation of an LLM assistant. The team has trained a candidate model to replace the current one.</p>`+
      `<p>Users’ questions fall into five categories: coding, factual Q&A, Chinese writing, tool use and ${DEF('safety refusals',TIP.safe())}. The report gives only a total over them; the chart’s row is that total: the dot is ${fD(t.D)} points, the horizontal line its ${DEF('95% interval',TIP.ci())} (${fB(t.lo)} to ${fB(t.hi)}), and the vertical line marks 0. Can the candidate replace the current model?</p>`+ch+
      `<p>The total comes from last week’s evaluation: each category got 1,000 items, both models answered each item, and each pair of answers was graded for which is better. A category’s ${DEF('difference',TIP.diff())} is the share of items where the candidate’s answer is better minus the share where the current model’s is, in points (percentage points); the total averages the five by ${DEF('share of traffic',TIP.traffic())}.</p>`,
      `<p>你负责一个 LLM 助手的评测。团队训练出一个候选模型，想用它替换当前模型。</p>`+
      `<p>用户的问题分五类：编程、事实问答、中文写作、工具调用、${DEF('安全拒答',TIP.safe())}。上周每类评了 1,000 题：两个模型答同一道题，判出谁答得更好。</p>`+
      `<p>一类题的${DEF('差',TIP.diff())}，是候选答得更好的题所占的比例，减去当前模型答得更好的比例，单位是分（百分点）。报告只给一个总分：五类的差按${DEF('流量',TIP.traffic())}加权平均。</p>`+
      `<p>图上这一行就是总分：点是 ${fD(t.D)} 分，横线是 ${DEF('95% 区间',TIP.ci())}（${fB(t.lo)} 到 ${fB(t.hi)}），竖线是 0。候选能替换当前模型吗？</p>`+ch);}},
 {id:'split',title:{en:'Look at each category',zh:'分开看每一类'},expect:()=>split,showMe:()=>setSplit(true),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A total is an average: progress in one category can hide a loss in another.','总分是平均数：一类的进步可以盖住另一类的退步。'),
  enter:()=>{split=false; return PLAN0();},
  body:()=>tx(`<p>The total weights each category by its share of traffic, so a category with much progress and much traffic can hide a loss elsewhere. Split the same 5,000 items by category, and give each category its own difference and interval.</p><p>${DEF('A program checks',TIP.prog())} coding, factual Q&A and tool use; people grade writing and safety.</p>`,
              `<p>总分按流量给每一类加权，所以一类进步多、流量又大，就能盖住别处的退步。把同样的 5,000 题按类别分开，每一类单独算差和区间。</p><p>编程、事实问答和工具调用${DEF('用程序判分',TIP.prog())}，写作和安全请人判。</p>`)+splitCtl('ecap-ic-split'),
  obs:()=>obsSplit()},
 {id:'size',title:{en:'How many items each category needs',zh:'每一类要多少题'},expect:()=>sizeTried.has(100),showMe:()=>setSize(100),label:{en:'Result',zh:'结果'},
  lead:()=>tx('The interval must be narrow enough for a verdict; 4 times the items halve it.','区间要窄到能下结论；题数变成 4 倍，区间才窄一半。'),
  enter:()=>{sizeTried=new Set([1000]); sizeClicked=false; return PLAN0();},
  body:()=>tx(`<p>Few items give a wide interval, too wide for a verdict; halving it takes 4 times the items, and 4 times what people cost.</p><p>A category’s verdict depends on where its interval lies:</p><ul class="ecap-vl"><li>wholly right of 0: better;</li><li>wholly left of 0: worse;</li><li>across 0, but wholly right of the dashed line at −3: <span class="ecap-nw">${DEF('not worse by more than 3 points',TIP.margin())};</span></li><li>across both 0 and −3: can’t tell yet.</li></ul>`+
      `<p>The candidate can ship only when all five categories have a verdict, none is worse and at least one <span class="ecap-nw">is better.</span></p><p>In this lab the measured difference equals the true difference, so a dot stays put when the items change; only its interval widens or narrows.</p>`,
      `<p>题少，区间就宽，下不了结论；区间要窄一半，题数和请人判的花费都得变成 4 倍。</p><p>每一类的结论，看区间落在哪：</p><ul class="ecap-vl"><li>整个在 0 右边：变好了；</li><li>整个在 0 左边：变差了；</li><li>跨过 0，但整个在 −3 分虚线右边：<span class="ecap-nw">${DEF('没有变差超过 3 分',TIP.margin())}；</span></li><li>同时跨过 0 和 −3 分：还看不出。</li></ul>`+
      `<p>五类都有结论、都没变差、至少一类变好，<span class="ecap-nw">才能上线。</span></p><p>在这个 Lab 里，量出来的差就等于真实的差：换题数时点不动，只有区间变宽或变窄。</p>`)+sizeCtl('ecap-ic-size'),
  obs:()=>obsSize()},
 {id:'judge',title:{en:'A judge is cheap, but it leans',zh:'裁判便宜，但会偏'},expect:()=>p4Tried.has('both'),showMe:()=>setP4('both'),label:{en:'Result',zh:'结果'},
  lead:()=>tx('A judge costs far less than people, but its preferences can turn writing’s loss into a gain.','裁判比人便宜得多，但它的偏好会把写作的退步变成进步。'),
  enter:()=>{p4Sel='human'; p4Tried=new Set(['human']); return p4Plan('human');},
  body:()=>tx(`<p>People grading 1,000 writing items cost ${usd(1000*HUM)}; another LLM as a ${DEF('judge',TIP.judge())} costs $0.01 a call. But a judge has preferences, and they can turn writing’s loss into a gain.</p>`+
      `<p>A judge has three preferences, for position, length and itself: it favors the answer shown first, the longer answer, and the answer from a model of its own family. This judge comes from the candidate’s family, and the candidate writes longer; judged once, the candidate’s answer is also always shown first. Judging in both orders, with a ${DEF('length correction',TIP.len())}, removes the position and length preferences.</p>`,
              `<p>请人判写作 1,000 题要 ${usd(1000*HUM)}；另一个 LLM 当${DEF('裁判',TIP.judge())}，判一次只要 $0.01。可裁判有偏好，会把退步判成进步。</p>`+
      `<p>裁判有位置、长度和自我三种偏好：偏向排在前面的、更长的、和自己同一家族的模型写的回答。这个裁判和候选模型来自同一家族，候选的回答也更长；只判一次时，候选的回答又总排在前面。两种顺序都判、再${DEF('按长度校正',TIP.len())}，能去掉位置和长度这两种偏好。</p>`)+p4Ctl(),
  obs:()=>obsJudge()},
 {id:'calib',title:{en:'Calibrate the judge with people',zh:'拿人工标注校准裁判'},expect:()=>calTried.has(1000),showMe:()=>setCal(1000),label:{en:'Result',zh:'结果'},
  lead:()=>tx('People grade a few of the judged items; the gap between them is the judge’s bias, and it is taken out.','人工判一小批裁判判过的题，两者的差就是裁判的偏差，再把它减掉。'),
  enter:()=>{calTried=new Set([0]); const P=p4Plan('once'); P.cal=0; return P;},
  body:()=>{const o=evalPlan(p4Plan('once')).rows.find(r=>r.id==='write');
   return tx(`<p>People can measure a judge’s bias, and then it is taken out. Start with the judge that leans most: judged once per item, it puts writing ${fA(o.D)} points up.</p>`+
      `<p>Pick some of the items it graded at random and ask people to grade them too. On those items, the judge’s difference minus the people’s difference is the judge’s bias; subtract it from the judge’s difference on the other items. This is called ${DEF('PPI',TIP.ppi())}.</p><p>The bias is itself estimated from a few items: the more often the judge and a person disagree, and the fewer items people grade, the wider the corrected interval.</p>`,
              `<p>裁判的偏差可以请人量出来，再减掉。先拿偏得最厉害的裁判试：每题判一次，它说写作好了 ${fA(o.D)} 分。</p>`+
      `<p>从它判过的题里随机抽一批，也请人判。同一批题上，裁判的差减去人的差，就是裁判的偏差；把它从裁判在其余题上量出的差里减掉。这种做法叫 ${DEF('PPI',TIP.ppi())}。</p><p>偏差也是从一小批题里估出来的：裁判和人意见不一致的题越多、请人判的题越少，修正后的区间越宽。</p>`)+calCtl('ecap-ic-cal');},
  obs:()=>obsCal()},
 {id:'online',title:{en:'Keep watching after release',zh:'上线以后接着看'},expect:()=>abTried.has(7),showMe:()=>setAb(7),label:{en:'Result',zh:'结果'},
  lead:()=>tx('Offline items differ from real traffic; after release, watch guardrails by category.','离线的题和真实的问题不一样；上线后按类别看护栏指标。'),
  enter:()=>{abDays=1; abTried=new Set([1]); return PLAN0();},
  body:()=>tx(`<p>Offline items differ from what real users ask, so a release starts with an ${DEF('A/B test',TIP.ab())}, and the candidate’s share of traffic grows in steps only while nothing looks wrong (a rollout). ${DEF('Guardrail metrics',TIP.guard())} (metrics that must not get worse during a rollout) must be read by category too: averaged over all conversations, writing’s loss may not show.</p>`+
      `<p>Suppose the team looked only at the total and started an A/B test: ${SH()} of conversations go to the candidate and another ${SH()} stay with the current model.</p><p>The guardrail is the ${DEF('regenerate rate',TIP.regen())}; the chart shows the candidate minus the current model, and right of 0 is worse. When a group’s interval lies wholly right of 0, the alarm goes off and the rollout stops.</p>`,
              `<p>离线的题和真实用户的问题不一样，所以上线先做 ${DEF('A/B 实验',TIP.ab())}，没问题再一步步加大候选的流量（放量）。${DEF('护栏指标',TIP.guard())}（放量时不能变差的指标）也要按类别看：全部对话平均下来，可能看不出写作变差了。</p>`+
      `<p>假设团队只看了总分就开始 A/B：${SH()} 的对话交给候选模型，另 ${SH()} 留给当前模型。</p><p>护栏指标是${DEF('重新生成率',TIP.regen())}，图里画的是候选减当前，在 0 右边是变差。哪一组对话的区间整个落到 0 的右边，就报警，停止放量。</p>`)+abCtl('ecap-ic-ab'),
  obs:()=>obsOnline()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('Judge each category, check the judge against people, and keep watching after release.','每一类分开判，裁判先拿人校准，上线后接着看。'),
  enter:()=>{const P=p4Plan('once'); P.cal=1000; return P;},
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdictP(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  /* a wrong pick also gets the reason (A8): why the total is not noise, or why a higher total is not enough */
  const why={ship:{en:'A higher total does not mean no category got worse.',zh:'总分变高，不等于每一类都没变差。'},
    noise:{en:'The total’s interval excludes 0, so it is not noise; the trouble is that it averages the five categories.',zh:'总分的区间不含 0，不是噪声；问题是它把五类平均了。'}}[pred];
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”. ${L(why)}</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。${L(why)}</p>`);
}
function obsSplit(){
  const e=evalPlan(PLAN0(),false), w=e.rows.find(r=>r.id==='write'), c=e.rows.find(r=>r.id==='code'), wt=id=>pct(CAT[id].w);
  /* the result first, then the verdict on the prediction and its reason, then why the total went up (R3.10, N4) */
  return tx(`<p><b>Split by category, Chinese writing’s difference is ${fD(w.D)} points, and its whole interval lies below 0: the loss is real. One category that got worse is enough to block a release, so this plan cannot ship.</b></p>`,
    `<p><b>分开以后，中文写作的差是 ${fD(w.D)} 分，区间整个在 0 的左边：退步是真的。只要有一类变差就不能上线，所以这个计划不能上线。</b></p>`)+verdictP()+
    tx(`<p>The total still went up because coding gained ${fA(c.D)} points on ${wt('code')} of the traffic, while writing lost ${fA(-w.D)} on only ${wt('write')}. Safety’s interval crosses 0: it cannot tell yet whether safety got better or worse.</p>`,
    `<p>总分还是变好了，因为编程好了 ${fA(c.D)} 分，占 ${wt('code')} 的流量；写作差了 ${fA(-w.D)} 分，只占 ${wt('write')}。安全这一类的区间跨过了 0，还看不出它变好还是变差。</p>`);
}
/* category names inside a sentence: in English lower case but for Chinese, the last joined with “and” */
const catList=rows=>tx((a=>a.length>1?a.slice(0,-1).join(', ')+' and '+a[a.length-1]:(a[0]||''))(rows.map(r=>r.id==='write'?CN[r.id].en:CN[r.id].en.replace(/^./,x=>x.toLowerCase()))),rows.map(r=>CN[r.id].zh).join('、'));
function obsSize(){
  const n=S.n.code, e=E(), w=e.rows.find(r=>r.id==='write'), unkR=e.rows.filter(r=>r.v==='unk'), unk=unkR.map(r=>L(CN[r.id]));
  const qw=pct(CAT.write.q), qc=pct(CAT.code.q);
  if(unk.length) return tx(`<p><b>With ${fint(n)} items per category, ${catList(unkR)} ${unk.length>1?'cannot':'can’t'} tell yet.</b></p><p>${w.v==='unk'?`Writing’s difference is only ${fA(-w.D)} points, and ${qw} of its items have a winner (coding: ${qc}), so its results swing more.</p><p>So writing’s interval crosses both the 0 line and the −3 line. `:''}A smaller difference, or results that swing more, need more items.</p>`,
      `<p><b>每类 ${fint(n)} 题时，${catList(unkR)}还看不出。</b></p><p>${w.v==='unk'?`写作的差只有 ${fA(-w.D)} 分，又有 ${qw} 的题分出了输赢（编程只有 ${qc}），每题结果更分散。</p><p>所以写作的区间同时跨过了 0 和 −3 分这两条线。`:''}差越小、每题结果越分散，要的题就越多。</p>`);
  const sf=e.rows.find(r=>r.id==='safe'), k=n/1000, nar=pct(1/Math.sqrt(k));
  if(sf.v==='ni') return tx(`<p><b>With ${fint(n)} items per category, every category has a verdict, and writing is worse.</b></p><p>Safety counts as not worse by more than 3 points; to be sure it got better too, its interval must narrow further.</p>`,
    `<p><b>每类 ${fint(n)} 题时，每一类都有了结论，写作变差了。</b></p><p>安全这一类算“没有变差超过 3 分”；要确定它也变好了，区间还得再窄。</p>`);
  return tx(`<p><b>With ${fint(n)} items per category, every category has a verdict, and writing is worse.</b></p><p>Safety’s interval now lies wholly above 0 as well: it really got better.</p><p>With ${k} times the items of 1,000 per category, each interval narrows only to ${nar} of its width then (${MI(`1/\\sqrt{${k}}`)}), while what people cost grows ${k} times.</p>`,
    `<p><b>每类 ${fint(n)} 题时，每一类都有了结论，写作变差了。</b></p><p>安全这一类的区间也整个落到了 0 的右边：它确实变好了。</p><p>题数是每类 1,000 题时的 ${k} 倍，区间只窄到那时的 ${nar}（${MI(`1/\\sqrt{${k}}`)}），花费却跟着变成 ${k} 倍。</p>`);
}
function obsJudge(){
  const k=p4Sel, w=evalPlan(p4Plan(k)).rows.find(r=>r.id==='write');
  const wc=usd(w.cost.j+w.cost.h);
  if(k==='human') return tx(`<p><b>People: writing is worse, and the verdict matches the true difference.</b> It costs ${wc}.</p>`,`<p><b>请人判：写作变差了，和真实的差一致。</b>要花 ${wc}。</p>`);
  const bt=fA(CAT.write.jb[0]), bs=fA(CAT.write.jb[1]);
  if(k==='once') return tx(`<p><b>The verdict is wrong: the judge says writing got ${fA(w.D)} points better, but the true difference (the hollow diamond) is ${fD(w.truth)}.</b></p><p>The three preferences together add ${bt} points, drawn under writing’s row.</p>`,
    `<p><b>结论错了：裁判说写作好了 ${fA(w.D)} 分，真实的差<span class="ecap-nw">（空心菱形）</span>却是 ${fD(w.truth)} 分。</b></p><p>三种偏好一共多算了 ${bt} 分，画在写作那一行下面。</p>`);
  /* the judge's number and its verdict, as the result judged once gives them (N9) */
  return tx(`<p><b>Still wrong: the judge says writing lost only ${fA(-w.D)} points (${L(VN[w.v]||VN.unk)}), and the ${bs} points from ${DEF('self-preference',TIP.self())} stay.</b></p><p>Without people to compare with, nothing would show that the judge is wrong.</p>`,
    `<p><b>还是错的：裁判说写作只差了 ${fA(-w.D)} 分（${L(VN[w.v]||VN.unk)}），${DEF('自我偏好',TIP.self())}的 ${bs}\u00a0分还在。</b></p><p>没有人工结果对照，就看不出裁判错了。</p>`);
}
function obsCal(){
  const e=E(), w=e.rows.find(r=>r.id==='write'), m=S.cal, ew=pct(w.e);
  if(!m) return tx(`<p><b>No calibration: the judge says writing got ${fA(w.D)} points better, which is wrong.</b> Pick how many items people also grade.</p>`,`<p><b>不校准：裁判说写作好了 ${fA(w.D)} 分，结论是错的。</b>选一个人工校准的题数。</p>`);
  const cost=w.cost.j+w.cost.h, r=cost/(1000*HUM);
  if(w.v==='unk') return tx(`<p><b>With ${fint(m)} items checked by people, the corrected interval still crosses both 0 and the −3 line, so it cannot tell yet.</b></p><p>The judge and people disagree on ${ew} of writing items, so the bias measured on only ${fint(m)} items is uncertain, and the correction adds a wide error.</p>`,
    `<p><b>人工校准 ${fint(m)} 题，修正后的区间还同时跨着 0 和 −3 分这两条线，所以还看不出。</b></p><p>写作有 ${ew} 的题裁判和人意见不一致，只从 ${fint(m)} 题里量出的偏差把握不大，修正带进来的误差也大。</p>`);
  const same=r>0.9&&r<1.1;
  return tx(`<p><b>With ${fint(m)} items checked by people, the corrected interval for writing lies wholly below 0: the loss is caught.</b></p><p>But grading writing this way costs ${usd(cost)} in all, ${same?'about as much as':`about ${pct(r)} of`} people grading all 1,000 items: this judge disagrees with people too often for a few labels to measure its bias.</p>`,
    `<p><b>人工校准 ${fint(m)} 题，修正后写作的区间整个在 0 的左边：抓到了退步。</b></p><p>可这样判写作一共花 ${usd(cost)}，${same?'和全请人判 1,000 题差不多':`约是全请人判 1,000 题的 ${pct(r)}`}：这个裁判和人意见不一致的题太多，少量人工量不准它的偏差。</p>`);
}
function obsOnline(){
  const a=abRows(abDays);
  if(!a.write.alarm) return tx(`<p><b>After ${abDays} day${abDays>1?'s':''}, the writing guardrail’s interval still crosses 0: no alarm yet.</b></p><p>Each group has only ${fint(a.arm*AB.sw)} writing conversations so far. Run the test longer.</p>`,
    `<p><b>${abDays} 天后，写作类的护栏区间还跨过 0：还没报警。</b></p><p>到现在每组只有 ${fint(a.arm*AB.sw)} 个写作对话，让实验再跑几天。</p>`);
  return tx(`<p><b>After ${abDays} days, writing conversations are regenerated about ${fixd(a.write.D*100,1)} points more often with the candidate, and the interval lies wholly above 0: the guardrail sounds the alarm, and the rollout stops.</b></p><p>Over all conversations the rate barely moves: the other four categories improved a little and average writing’s rise away. So online metrics, too, are read by category.</p>`,
    `<p><b>${abDays} 天后，写作类对话的重新生成率高了约 ${fixd(a.write.D*100,1)} 个百分点，区间整个在 0 的右边：护栏报警，停止放量。</b></p><p>全部对话的重新生成率几乎没变，因为其他四类反而好了一点，正好把写作的上升平均掉了。所以线上指标也要按类别分开看。</p>`);
}
function remember(){
  const t=tot0(), o=evalPlan(p4Plan('once')).rows.find(r=>r.id==='write'), cal=E().rows.find(r=>r.id==='write');
  return tx(`<ul><li><b>Judge each category, not one total.</b> This candidate’s total was ${fA(t.D)} points higher, yet Chinese writing lost ${WL()} points.</li><li><b>Check a judge against people before you trust it.</b> Judged once, the judge put writing ${fA(o.D)} points up; ${fint(cal.mm)} items graded by people measured the ${fA(o.b)} points it added, and only after taking them out was the loss caught (writing’s row in the chart).</li><li><b>Keep watching after release, by category.</b> In the A/B test the overall regenerate rate stayed flat, and only the writing guardrail sounded the alarm. Problems found in user feedback and sampled conversations become offline items.</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “A higher total means it can ship.” A ${WL()}-point loss in a fifth of the traffic fits inside a total that went up.</p><p>✗ “A judge is as good as people.” In both orders the judge still leaned ${fA(CAT.write.jb[1])} points; only people’s labels show how far a judge leans.</p><p>✗ “Once it ships, evaluation is done.” Real users ask what your items did not; guardrails by category find what the offline evaluation missed.</p></div>`+toScen(),
    `<ul><li><b>每一类分开下结论，不看一个总分。</b>这个候选总分高 ${fA(t.D)} 分，写作却差了 ${WL()} 分。</li><li><b>裁判先拿人工校准再用。</b>只判一次的裁判说写作好了 ${fA(o.D)} 分；人工校准 ${fint(cal.mm)} 题量出它多算了 ${fA(o.b)} 分，减掉才抓到退步（图里写作那一行）。</li><li><b>上线后接着按类别看。</b>A/B 实验里全部对话的重新生成率没变，只有写作的护栏报了警。用户反馈和抽查对话里查出的问题，补进离线评测。</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “总分高就能上线。”一类占五分之一的流量，差了 ${WL()} 分，总分照样可以上升。</p><p>✗ “裁判和人一样可靠。”两种顺序都判以后，裁判还偏 ${fA(CAT.write.jb[1])} 分；偏多少，只有拿人工结果对照才知道。</p><p>✗ “上线以后就不用评测了。”用户会问评测题里没有的问题；按类别设的护栏能发现离线评测漏掉的退步。</p></div>`+toScen());
}

/* =====================================================================
   scenarios
   ===================================================================== */
const PRESETS=[
 {id:'people',t:{en:'The initial setup: people grade writing and safety',zh:'初始设置：写作和安全请人判'},d:{en:'1,000 items per category',zh:'每类 1,000 题'},p:PLAN0},
 {id:'cheap',t:{en:'A judge everywhere, judged once',zh:'全部交给裁判，每题判一次'},d:{en:'1,000 items per category, no people to calibrate it',zh:'每类 1,000 题，不请人校准'},
  p:()=>({m:{code:'judge',qa:'judge',write:'judge',tool:'judge',safe:'judge'},n:{code:1000,qa:1000,write:1000,tool:1000,safe:1000},ctl:0,cal:0})},
 {id:'ctl',t:{en:'Program checks, and a judge in both orders',zh:'能用程序就用程序，裁判两种顺序都判'},d:{en:'writing and safety 3,000 items, no people',zh:'写作和安全各 3,000 题，不请人'},
  p:()=>({m:{code:'prog',qa:'prog',write:'judge',tool:'prog',safe:'judge'},n:{code:1000,qa:1000,write:3000,tool:1000,safe:3000},ctl:1,cal:0})},
 {id:'small',t:{en:'A quick check: 100 items per category',zh:'快速检查：每类 100 题'},d:{en:'people grade writing and safety',zh:'写作和安全请人判'},
  p:()=>({m:{code:'prog',qa:'prog',write:'human',tool:'prog',safe:'human'},n:{code:100,qa:100,write:100,tool:100,safe:100},ctl:0,cal:0})}];
const same=(a,b)=>a.ctl===b.ctl&&a.cal===b.cal&&CATS.every(c=>a.m[c.id]===b.m[c.id]&&a.n[c.id]===b.n[c.id]);
const presetNow=()=>PRESETS.findIndex(p=>same(p.p(),S));
function scenHTML(i){
  const e=E(), w=e.rows.find(r=>r.id==='write'), id=PRESETS[i].id;
  const dec=tx(`The plan concludes: <b>${L(DN[e.decision])}</b>.</p><p>`,`这个计划的结论：<b>${L(DN[e.decision])}</b>。</p><p>`);
  if(id==='people') return tx(`<p>${dec}Every verdict matches the true difference, and writing’s loss is caught. But people cost ${usd(e.cost)} and need ${e.days} days.</p>`,`<p>${dec}每个结论都和真实的差一致，写作的退步也抓到了。可是请人要花 ${usd(e.cost)}，用 ${e.days} 天。</p>`);
  /* the other four rows are overstated too (their diamonds sit left of their intervals), but no verdict flips (A8) */
  const ob=CATS.filter(c=>c.id!=='write').map(c=>Math.round(c.jb[0]*100)), oLo=Math.min(...ob), oHi=Math.max(...ob);
  if(id==='cheap') return tx(`<p>${dec}It costs only ${usd(e.cost)}, but the judge’s preferences make writing look ${fA(w.D)} points better, so the plan would ship a model that got worse.</p><p>The other four categories are overstated too, by ${oLo} to ${oHi} points; their direction stays the same, so their verdicts still hold.</p>`,
    `<p>${dec}只花 ${usd(e.cost)}，可裁判的偏好让写作看上去好了 ${fA(w.D)} 分，于是这个计划会把一个变差了的模型放上线。</p><p>其他四类也被多算了 ${oLo} 到 ${oHi} 分，只是方向没变，结论还算对。</p>`);
  if(id==='ctl') return tx(`<p>${dec}Both orders with length taken out remove two preferences, but not the judge’s lean toward its own family: it measures writing at only ${fA(w.D)} points worse, and the loss slips through.</p><p>It costs ${usd(e.cost)}.</p>`,`<p>${dec}两种顺序都判、按长度校正，去掉了位置和长度两种偏好，却去不掉裁判对自家模型的偏向：它量出写作只差 ${fA(w.D)} 分，退步漏了过去。</p><p>花费 ${usd(e.cost)}。</p>`);
  const unk=e.rows.filter(r=>r.v==='unk').length;
  return tx(`<p>${dec}With ${fint(e.rows[0].n)} items each, ${unk} categories cannot tell yet: their intervals cross both 0 and the −3 line.</p><p>It costs ${usd(e.cost)}.</p>`,`<p>${dec}每类只有 ${fint(e.rows[0].n)} 题，有 ${unk} 类还看不出：区间同时跨过了 0 和 −3 分这两条线。</p><p>花费 ${usd(e.cost)}。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
/* the answer's plan, one line per way of grading: categories graded the same way share a line (P27) */
const pickList=P=>{const e=evalPlan(P), g=[];
  e.rows.forEach(r=>{const pl=planLine(r,P), o=g.find(x=>x.pl===pl); if(o) o.rs.push(r); else g.push({pl,rs:[r]});});
  return `<ul class="ecap-pick">${g.map(x=>`<li><b>${esc(x.rs.length>1?catList(x.rs).replace(/^./,z=>z.toUpperCase()):L(CN[x.rs[0].id]))}</b>${tx(': ','：')}${esc(nbN(x.rs.length>1?tx(`${x.pl} each`,x.pl.replace(/(\d[\d,]*) 题/,'各 $1 题')):x.pl))}</li>`).join('')}</ul>`;};
const SAFE_FREE=()=>{const c=CAT.safe; return Math.round(Math.ceil(Z*Z*(c.q-c.d*c.d)/((c.d+MARGIN)**2))/10)*10;};   // safety's people-graded items if any count were allowed
/* R12: a plan that misses says what it still misses and what to change. First the cause in each category that cannot tell or got a
   wrong verdict; then the hint: the one setting that fixes it, with what it gives and costs (when one change is enough), or which
   settings must change together (then no values: the reader works them out, S11). General advice follows only where the plan does
   not already do it; what the plan already does right is said as such. */
const probs=e=>e.rows.filter(r=>r.v==='unk'||r.wrong).map(r=>r.id).concat(e.rows.some(r=>r.v==='better')?[]:['better'],e.cost>BUDGET+1e-9?['cost']:[],e.days>DEADLINE?['days']:[]);
const ctlW=c=>c?tx('in both orders, length-corrected','两种顺序都判、按长度校正'):tx('once per item','每题判一次');
const kLess=(a,b)=>{for(let i=0;i<a.length;i++){if(a[i]<b[i]-1e-9) return true; if(a[i]>b[i]+1e-9) return false;} return false;};
function chMoves(){const p0=probs(E()), out=[];   // a change counts when it fixes something and breaks nothing that was right
  CATS.forEach(c=>catOpts(c).forEach(([m,n])=>[0,1].forEach(ctl=>CALS.forEach(cal=>{
    const P=clone(S); P.m[c.id]=m; P.n[c.id]=n; P.ctl=ctl; P.cal=cal;
    const ch=[]; if(m!==S.m[c.id]) ch.push('m'); if(n!==S.n[c.id]) ch.push('n'); if(ctl!==S.ctl) ch.push('ctl'); if(cal!==S.cal) ch.push('cal');
    if(!ch.length) return; const e=evalPlan(P), p=probs(e); if(p.length<p0.length&&p.every(x=>p0.includes(x))) out.push({c,P,e,ch,k:[chPass(e)?0:1,p.length,ch.length,e.cost]});}))));
  const best=xs=>xs.reduce((a,x)=>!a||kLess(x.k,a.k)?x:a,null);
  return {one:best(out.filter(x=>x.ch.length===1)),any:best(out)};}
const VPH={better:{en:'all of it right of 0, so it is better',zh:'整个在 0 的右边，能看出它变好了'},worse:{en:'all of it left of 0, so it is worse',zh:'整个在 0 的左边，能看出它变差了'},
  ni:{en:'its left end above −3, so it is not worse by more than 3 points',zh:'左端没有低于 −3 分，能确定它没有变差超过 3 分'},unk:{en:'so it still cannot tell',zh:'还是看不出'}};
function missHTML(e,why,withHints){
  /* each paragraph carries at most three numbers (tests/numdense.py): the cause, the change, what it gives and what it costs, apart.
     Once the answer is shown, the hints go (R4.4, P27): the answer says what to change */
  const bad=e.rows.filter(r=>r.v==='unk'||r.wrong), ps=[];
  bad.slice(0,2).forEach(r=>{const cn=L(CN[r.id]);
    if(r.wrong){   // the cause and what it did to the result, in one sentence (P27); the lean itself is drawn under the row
      ps.push(r.meth==='judge'&&!r.mm
        ?tx(`The judge grades ${ctlW(S.ctl)} and no people check it, so it measures ${cn} at ${fD(r.D)} points; the true difference is ${fD(r.truth)}.`,`裁判${ctlW(S.ctl)}，又没有请人校准，把${cn} ${fD(r.truth)} 分的差量成了 ${fD(r.D)} 分。`)
        :tx(`The plan measures ${cn} at ${fD(r.D)} points, but the true difference is ${fD(r.truth)}.`,`这个计划量出${cn}的差是 ${fD(r.D)} 分，可真实的差是 ${fD(r.truth)} 分。`));
      return;}
    ps.push(tx(`${cn}’s 95% interval runs from ${fD(r.lo)} to ${fD(r.hi)} points.`,`${cn}的 95% 区间是 ${fD(r.lo)} 到 ${fD(r.hi)} 分。`));
    ps.push(tx('It crosses both 0 and the −3 line, so the plan can say neither whether it got better or worse, nor that it lost no more than 3 points.',
               '区间同时跨过了 0 和 −3 分那条线，所以既说不出它变好还是变差，也不能确定它的退步不超过 3 分。'));
    if(r.meth==='judge'&&r.mm>=r.n) ps.push(tx(`People check ${fint(S.cal)} items in each judged category, ${S.cal>r.n?'more than':'as many as'} this category has: every item is graded by people, none of the judge’s grades is used, and the verdict rests on these ${fint(r.n)} labels alone.`,
                                               `每个用裁判的类别人工校准 ${fint(S.cal)} 题，${S.cal>r.n?'比这一类的题还多':'和这一类的题数一样多'}：每道题都请人判了，裁判的分数一个也没用上，结论只靠人工判的这 ${fint(r.n)} 题。`));});
  let h=tx(`<p class="badline ecap-mt"><b>Not met yet: ${why.join('; ')}.</b></p>`,`<p class="badline ecap-mt"><b>还没达标：${why.join('；')}。</b></p>`)+ps.map((t,i)=>`<p class="small">${i?'':`<b>${tx('Why: ','原因：')}</b>`}${t}</p>`).join('');
  if(!withHints) return h;
  const mv=chMoves(), hints=[], ch=new Set();
  const nm={m:c=>tx(`how ${L(CN[c.id])} is graded`,`${L(CN[c.id])}的判分方法`),n:c=>tx(`the items for ${L(CN[c.id])}`,`${L(CN[c.id])}的题数`),ctl:()=>tx('how the judge runs','裁判怎么判'),cal:()=>tx('the items people check','人工校准的题数')};
  const multi=mv.any&&mv.any.ch.length>1&&!mv.any.k[0]&&!(mv.one&&!mv.one.k[0]);   // only a change of several settings meets the goal: name them
  if(mv.one&&!multi){const x=mv.one, k=x.ch[0], c=x.c, e1=x.e; ch.add(k);
    const what=k==='m'?tx(`grade ${L(CN[c.id])} by ${L(MN[x.P.m[c.id]])} instead of by ${L(MN[S.m[c.id]])}`,`把${L(CN[c.id])}的判分方法从${L(MN[S.m[c.id]])}改成${L(MN[x.P.m[c.id]])}`)
      :k==='n'?tx(`${x.P.n[c.id]>S.n[c.id]?'raise':'cut'} ${L(CN[c.id])} from ${fint(S.n[c.id])} to ${fint(x.P.n[c.id])} items`,`把${L(CN[c.id])}的题数从 ${fint(S.n[c.id])} 题${x.P.n[c.id]>S.n[c.id]?'加到':'减到'} ${fint(x.P.n[c.id])} 题`)
      :k==='ctl'?tx(`have the judge grade ${ctlW(x.P.ctl)} instead of ${ctlW(S.ctl)}`,`让裁判${ctlW(x.P.ctl)}，不再${ctlW(S.ctl)}`)
      :tx(`have people check ${fint(x.P.cal)} items instead of ${fint(S.cal)} in each judged category`,`把每个用裁判的类别人工校准的题数从 ${fint(S.cal)} 题改成 ${fint(x.P.cal)} 题`);
    hints.push(tx(`${what.replace(/^./,z=>z.toUpperCase())}, and leave everything else as it is.`,`${what}，其余设置不变。`));
    bad.forEach(r=>{const q=e1.rows.find(y=>y.id===r.id); hints.push(tx(`Then ${L(CN[r.id])}’s interval becomes ${fD(q.lo)} to ${fD(q.hi)} points: ${L(VPH[q.v])}.`,`这样${L(CN[r.id])}的区间变成 ${fD(q.lo)} 到 ${fD(q.hi)} 分，${L(VPH[q.v])}。`));});
    const d=e1.cost-e.cost, dc=Math.abs(d)<0.5?tx('the same','不变'):(d>0?tx(`${usd(d)} more`,`多 ${usd(d)}`):tx(`${usd(-d)} less`,`少 ${usd(-d)}`));
    const left=[]; if(!chPass(e1)){const w1=e1.rows.filter(r=>r.v==='unk'||r.wrong).filter(r=>!bad.some(b=>b.id===r.id)); if(w1.length) left.push(tx(`${w1.map(r=>L(CN[r.id])).join(', ')} then misses`,`${w1.map(r=>L(CN[r.id])).join('、')}又不行了`));
      if(e1.cost>BUDGET+1e-9) left.push(tx('it is over budget','超出预算')); if(e1.days>DEADLINE) left.push(tx('people need too many days','人工天数超了'));
      if(!e1.rows.some(r=>r.v==='better')) left.push(tx('no category is shown to be better','没有一类确定变好了'));}
    hints.push(tx(`The whole plan then costs ${usd(e1.cost)} (${dc}) and people need ${e1.days} day${e1.days>1?'s':''}${chPass(e1)?': that meets the goal.':`, but ${left.join('; ')||'the goal is still not met'}.`}`,
                  `整个计划的花费变成 ${usd(e1.cost)}（${dc}），人工要 ${e1.days} 天${chPass(e1)?'：这样就达到目标了。':`，但${left.join('；')||'还没达到目标'}。`}`));}
  else if(mv.any){const x=mv.any; x.ch.forEach(k=>ch.add(k)); const ns=x.ch.map(k=>nm[k](x.c)), j=ns.length<3?ns.join(tx(' and ','和')):ns.slice(0,-1).join(tx(', ','、'))+tx(' and ','和')+ns[ns.length-1];
    hints.push(tx(`Changing one setting is not enough; ${j} must change together.`,`只改一项还不够，${j}要一起改。`));}
  const judged=e.rows.some(r=>r.meth==='judge'), w=e.rows.find(r=>r.id==='write'), ewB=pct(CAT.write.je[1]), ewO=pct(CAT.write.je[0]);
  if(w.meth==='judge'&&!S.cal&&!w.wrong) hints.push(tx('A judge without people to check it misreads writing.','不拿人校准的裁判会看错写作。'));
  if(judged&&!S.ctl) hints.push(tx(`Judged once, the judge disagrees with people on many writing items (${ewO}; in both orders, only ${ewB}), so estimating its bias from people’s labels has a large error, and it takes so many labels that the plan goes over budget.`,`只判一次的裁判和人意见不一致的题多（${ewO}，两种顺序都判时只有 ${ewB}），用人工估它的偏差时误差大，要请很多人判才够，超出预算。`));
  if(w.meth==='human') hints.push(tx('People alone on 1,000 writing items cost too much as well.','写作全请人判 1,000 题也太贵。'));
  h+=hints.map((t,i)=>`<p class="small">${i?'':`<b>${tx('Hint: ','提示：')}</b>`}${t}</p>`).join('');
  if(judged&&S.ctl&&S.cal&&!ch.has('ctl')&&!ch.has('cal')) h+=tx('<p class="small">Already right: the judge grades in both orders with length taken out, and people check its grades.</p>','<p class="small">已经做对的：裁判两种顺序都判、按长度校正，也拿人校准了。</p>');
  return h;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const e=E(), S0=chSearch(), best=S0.best, pass=chPass(e), min=pass&&Math.abs(e.cost-best.cost)<1e-6;
  const why=[]; const unk=e.rows.filter(r=>r.v==='unk'), wr=e.rows.filter(r=>r.wrong);
  if(unk.length) why.push(tx(`${catList(unk)} cannot tell yet`,`${catList(unk)}还看不出`));
  if(wr.length) why.push(tx(`${catList(wr)} got a wrong verdict`,`${catList(wr)}的结论错了`));
  if(!e.caught&&!wr.some(r=>r.id==='write')&&!unk.some(r=>r.id==='write')) why.push(tx('writing’s loss is not caught','没有抓到写作的退步'));
  if(!e.rows.some(r=>r.v==='better')) why.push(tx('no category is shown to be better','没有一类确定变好了'));
  if(e.cost>BUDGET+1e-9) why.push(tx(`it costs ${usd(e.cost)}, over ${usd(BUDGET)}`,`花费 ${usd(e.cost)}，超过 ${usd(BUDGET)}`));
  if(e.days>DEADLINE) why.push(tx(`people need ${e.days} days, over ${DEADLINE}`,`人工要 ${e.days} 天，超过 ${DEADLINE} 天`));
  const ewB=pct(CAT.write.je[1]);
  /* P28: what safety would cost on the judge instead: people check as many of its items as the plan's calibration */
  const safeJ=()=>{const P=clone(CH_ANSWER); P.m.safe='judge'; return evalPlan(P).cost-evalPlan(CH_ANSWER).cost;};
  let h;
  if(min) h=tx(`<p class="okline ecap-mt"><b>Goal met, at the lowest cost among the settings you can pick: ${usd(e.cost)}, ${e.days} days, and writing’s loss is caught.</b></p><p>Giving safety to the judge as well would mean people check ${fint(CH_ANSWER.cal)} of its items too, ${usd(safeJ())} more than people grading them directly.</p><p>If item counts could be any number it could cost less (safety needs only about ${SAFE_FREE()} items graded by people); the reason stays the same. Program checks are free, so the money goes on the two categories no program can check.</p>`,
                 `<p class="okline ecap-mt"><b>达成目标，而且是这道题能选的设置里最便宜的：${usd(e.cost)}，${e.days} 天，抓到了写作的退步。</b></p><p>安全也交给裁判的话，人工校准同样要 ${fint(CH_ANSWER.cal)} 题，比直接请人判还多 ${usd(safeJ())}。</p><p>题数能随便定的话还能再省（安全请人判 ${SAFE_FREE()} 题左右就够），道理不变：程序判分不花钱，钱只花在程序判不了的两类上。</p>`);
  else if(pass) h=tx(`<p class="okline ecap-mt"><b>Goal met: ${usd(e.cost)}, ${e.days} days, and writing’s loss is caught.</b> But among the settings you can pick, a cheaper plan also meets it.</p><p>The cheapest costs ${usd(best.cost)}:</p>${pickList(CH_ANSWER)}`,
                      `<p class="okline ecap-mt"><b>达成目标：${usd(e.cost)}，${e.days} 天，抓到了写作的退步。</b>不过在能选的设置里，有更便宜的计划也能达标。</p><p>最便宜的只要 ${usd(best.cost)}：</p>${pickList(CH_ANSWER)}`);
  else h=missHTML(e,why,!chReveal)
       +(chReveal?tx(`<p><b>The answer:</b> among the settings you can pick, the cheapest plan that meets the goal costs ${usd(best.cost)}.</p>`,`<p><b>答案：</b>在能选的设置里，最便宜的达标计划花 ${usd(best.cost)}。</p>`)+pickList(CH_ANSWER):`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal&&!pass) return h;
  /* the brief chart (R4.6): the initial setup, the reader's plan, and the cheapest plan of the one judge setting that can meet the goal;
     the chart's note says the other judge settings have no plan that does (P26) */
  const ctlN=c=>c?tx('both orders, length-corrected','两种顺序、按长度校正'):tx('judged once','判一次');
  const rows=S0.combos.map(x=>({label:`${x.pass?'✓ ':''}${tx(`Judge ${ctlN(x.ctl)}, people check ${fint(x.cal)}`,`裁判${ctlN(x.ctl)}，人工校准 ${fint(x.cal)} 题`)}${tx(` (${x.days} days)`,`（${x.days} 天）`)}`,value:x.cost,text:usd(x.cost),ok:x.pass,best:x.pass&&Math.abs(x.cost-best.cost)<1e-6,tags:[]}));
  const st=evalPlan(CH_START());
  rows.unshift({label:tx('Judge once, no people checking (writing read wrong)','裁判判一次，不请人校准（写作看错了）'),value:st.cost,text:usd(st.cost),ok:false,tags:[{text:tx('initial setup','初始设置')}]});
  rows.push({label:tx(`Your plan (${e.days} day${e.days>1?'s':''})`,`你的计划（${e.days} 天）`),value:e.cost,text:usd(e.cost),ok:pass,me:true,tags:[]});
  h+=A.ui.cmpChart({title:tx(`Cost: the initial setup, your plan, and the cheapest plan that meets the goal (budget ${usd(BUDGET)}, ${DEADLINE} days)`,`花费：初始设置、你的计划、最便宜的达标计划（预算 ${usd(BUDGET)}、${DEADLINE} 天）`),
    aria:tx('Cost of each plan: ','每个计划的花费：')+rows.filter(r=>r.ok||r.me||r.tags.length).map(r=>`${r.label}${tx(': ','：')}${r.text}`).join(tx('; ','；')),
    rows,brief:true,goal:BUDGET,goalLabel:tx(`budget ${usd(BUDGET)}`,`预算 ${usd(BUDGET)}`),max:4500,
    hiddenText:k=>tx(`The other ${k} combinations of judging and calibration count cannot meet the goal; they are not shown.`,`另外 ${k} 种判法和校准题数的组合都不达标，没有画出。`),
    okText:tx('✓ every verdict right, within budget and 3 days','✓ 结论都对，不超预算、3 天内'),noText:tx('over budget, too slow, or a verdict wrong','超预算、太慢或结论有错'),goalText:tx('the budget line','预算线')});   // the amount is on the line itself (P26)
  /* why only that setting fits (P27): people alone need too many items (N15), the judge in both orders needs few labels */
  const nH=SIZES.find(n=>{const P=clone(CH_ANSWER); P.m.write='human'; P.n.write=n; return evalPlan(P).rows.find(r=>r.id==='write').v==='worse';})||1000;
  h+=tx(`<p class="small">People alone need ${fint(nH)} writing items to catch the loss, which costs ${usd(nH*HUM)}, over the budget.</p><p class="small">Only a judge in both orders fits: it disagrees with people on just ${ewB} of writing items, so ${fint(CH_ANSWER.cal)} labels measure its lean, and ${fint(CH_ANSWER.n.write)} judged items keep the interval narrow.</p>`,
        `<p class="small">写作全请人判，要 ${fint(nH)} 题才抓得到退步，花 ${usd(nH*HUM)}，<span class="ecap-nw">超出预算。</span></p><p class="small">只有两种顺序都判的裁判放得进：它和人只有 ${ewB} 的题意见不一致，${fint(CH_ANSWER.cal)} 题人工就量得准它的偏差，${fint(CH_ANSWER.n.write)} 题裁判让区间足够窄。</p>`);
  /* how a real evaluation differs, one point per sentence (N2) */
  const shown=pass?e:evalPlan(CH_ANSWER), sw=shown.rows.find(r=>r.id==='write'), pc=pct(pCatch(sw));
  h+=tx(`<p class="small">In a real evaluation the measured difference wanders: ${pass?'your':'this'} plan would catch writing’s loss only about ${pc} of the time, so keep extra items. The true differences are unknown there too, so nothing shows how far a judge leans: every category a judge grades needs people to calibrate it.</p>`,
        `<p class="small">真实评测里，量出来的差会晃动：${pass?'你的':'这个'}计划只有约 ${pc} 的机会抓到写作的退步，题要多留一些。真实的差也看不到，就看不出裁判偏多少，所以裁判判的每一类都要请人校准。</p>`);
  return h;
}

/* =====================================================================
   header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
 guided:tx('Seven parts: one total hides a loss, intervals per category, a judge and its preferences, people to calibrate it, and guardrails after release.','七个部分，依次讲一个总分会藏住退步、每一类的区间、裁判和它的偏好、用人工校准裁判、上线后的护栏。'),
 sandbox:tx('Pick a scenario or set every category’s grading in Settings, and see the verdicts, the cost and the days.','选一个场景，或者在自由调参里给每一类定判分方法，看结论、花费和天数。'),
 challenge:tx(`With ${usd(BUDGET)} and ${DEADLINE} days, plan an evaluation whose verdicts are all right and that catches writing’s loss.`,`预算 ${usd(BUDGET)}、${DEADLINE} 天内，设计一个结论都对、能抓到写作退步的评测计划。`)});
const TRUEL=()=>CATS.map(c=>`${L(CN[c.id])} ${fD(c.d)}`).join(tx(', ','，'));
const pts=x=>sgn(x*100,1).replace(/\.0$/,'');
const FIDPOP=()=>tx(
 `<p><b>What is modeled.</b> A stated world: five categories with a set share of traffic and a set true difference each (${TRUEL()} points), and a set share q of items with a winner. People and program checks measure the true difference exactly; the interval is the item-by-item one of ${LL('eval-ci')}. A judge adds a set bias (graded once: position and length preferences plus self-preference; both orders with length taken out: self-preference only). Calibration follows prediction-powered inference with λ = 1 (ppi_py’s <code>ppi_mean_ci</code> with <code>alpha=0.05</code> and <code>lam=1</code>; its default, <code>alpha=0.1</code>, gives a 90% interval, while every interval on this page is 95%). As in ppi_py, the judge’s mean is taken over the items people did not check. The correction assumes that when judge and person disagree they differ by one level: one calls a tie, the other a winner.</p>`+
 `<p><b>Simplified.</b> A measured difference here equals its true value (plus the judge’s bias): a real one also wanders inside its interval, as ${LL('eval-ci')} shows. So the item counts here are just enough; a real evaluation needs more. Calibration items are drawn from the judged items. Generating the answers costs nothing here. Days count only people’s labels. Every share, bias, price and rate is this lab’s assumption, chosen to make each idea visible.</p>`+
 `<p><b>Sources.</b> ${cite(SRC.ppi,'ppi_py, <code>ppi.py</code>')} (pinned commit); ${cite(SRC.lme,'lm-evaluation-harness, <code>metrics.py</code>')} (<code>mean_stderr</code>); ${cite(SRC.fc,'FastChat, <code>common.py</code>')} (both orders). Papers: Angelopoulos et al. 2023 (arXiv 2301.09633); Zheng et al. 2023 (arXiv 2306.05685); Dubois et al. 2024 (arXiv 2404.04475, the length correction).</p>`,
 `<p><b>模拟了什么。</b>一个设定好的世界：五类题，每类有设定的流量比例和真实的差（${TRUEL()} 分），以及有一方答得更好的题所占的比例 q。人工和程序判分量出的就是真实的差；区间按${LL('eval-ci')}里逐题对比的公式算。裁判会加上设定的偏差（判一次：位置偏好、长度偏好加自我偏好；两种顺序都判并按长度校正：只剩自我偏好）。人工校准按 λ = 1 的 prediction-powered inference 算（ppi_py 的 <code>ppi_mean_ci</code>，<code>alpha=0.05</code>、<code>lam=1</code>；它默认 <code>alpha=0.1</code>，给的是 90% 区间，页面上的区间都是 95%）。和 ppi_py 一样，裁判的平均只算人没有判的题。修正时假设裁判和人不一致时只差一级：一个判平手，另一个判出了输赢。</p>`+
 `<p><b>简化了什么。</b>这里量出来的差就等于真实的差（再加上裁判的偏差）；真实评测里它还会在区间里晃动，见${LL('eval-ci')}。所以这里的题数是“刚好够”，真实评测要多留。校准用的题从裁判判过的题里抽。让模型回答题目的花费不计。天数只算人工标注。所有比例、偏差、价格和速度都是本 Lab 的假设，取值是为了让每个道理看得见。</p>`+
 `<p><b>来源。</b>${cite(SRC.ppi,'ppi_py 的 <code>ppi.py</code>')}（固定提交）；${cite(SRC.lme,'lm-evaluation-harness 的 <code>metrics.py</code>')}（<code>mean_stderr</code>）；${cite(SRC.fc,'FastChat 的 <code>common.py</code>')}（两种顺序都判）。论文：Angelopoulos 等 2023（arXiv 2301.09633）；Zheng 等 2023（arXiv 2306.05685）；Dubois 等 2024（arXiv 2404.04475，按长度校正）。</p>`);
const PRIPOP=()=>tx(`<p>A score measured on a few hundred items is an estimate with an interval (${LL('eval-ci')}); an LLM judge has preferences you must cancel and check (${LL('llm-judge')}); an agent’s success is measured over several tries (${LL('pass-k')}); this lab grades each tool-use item once. It puts these ideas together into one plan.</p>`,
  `<p>几百道题上量出来的分数是一个带区间的估计（${LL('eval-ci')}）；LLM 裁判有偏好，要抵消、要检验（${LL('llm-judge')}）；agent 的成功率要看多次尝试（${LL('pass-k')}），这个 Lab 里工具调用的每道题只判一次。这个 Lab 把这些放进同一个计划。</p>`);
const OUTCOME=()=>tx('plan an LLM assistant’s evaluation: per-category size and grading, a judge calibrated by people, guardrails after release.','为 LLM 助手设计上线前后的评测，定每一类的题数和判分方法，用人工校准裁判，上线后设好护栏。');
const EXITS=()=>{
  const t=tot0();
  const nNew=Math.round(Z*Z*(0.2-0.0004)/0.0004), n80=Math.round((Z+0.84)**2*(0.2-0.0004)/0.0004);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`The total rose by ${fA(t.D)} points, yet the candidate cannot ship. Why?`,`总分高了 ${fA(t.D)} 分，候选模型还是不能上线，为什么？`),
   o:[tx(`The total averages the categories by traffic; coding’s larger gain on more traffic hides writing’s ${WL()}-point loss.`,`总分按流量把各类平均；编程进步得多、流量又大，盖住了写作 ${WL()} 分的退步。`),
      tx('The total’s interval is too wide to trust.','总分的区间太宽，不可信。'),
      tx('Writing had too few items, so the total was computed wrong.','写作的题太少，所以总分算错了。')],a:0,
   n:[null,tx('The total’s interval excludes 0; the total is measured well. It just answers a different question.','总分的区间不含 0，量得没问题，只是它回答的不是这个问题。'),
      tx('Each category had 1,000 items, and the total is a correct weighted average of them.','每类都是 1,000 题，总分是它们正确的加权平均。')],
   why:tx('A release needs every category to hold up. An average can go up while one category goes down.','能不能上线，要看每一类都没出问题。平均数上升时，某一类照样可以下降。')},
  {q:tx('In both orders with length taken out, why does the judge still misread writing?','两种顺序都判、按长度校正以后，裁判为什么还是看错了写作？'),
   o:[tx('Those two controls remove the position and length preferences only; the judge’s lean toward its own model family stays, and only people’s labels reveal it.','这两种做法只去掉位置偏好和长度偏好；裁判偏向自家模型的偏差还在，只有拿人工标注对照才看得出。'),
      tx('3,000 items are too few for a judge.','3,000 题对裁判来说太少。'),
      tx('Judging twice makes the interval wider.','判两次会让区间变宽。')],a:0,
   n:[null,tx('More items narrow the interval around the judge’s answer; they do not move the answer toward the truth.','题多只让区间围着裁判的答案变窄，不会把答案挪向真实的差。'),
      tx('The interval with 3,000 items is about ±2 points; the problem is where it sits, not how wide it is.','3,000 题的区间约 ±2 分；问题在它的位置，不在宽窄。')],
   why:tx('A bias shifts every verdict the same way; more of the same items cannot reveal it. Only a reference, people, measures it.','偏差让每个结论往同一个方向偏，同样的题再多也看不出来。只有拿参照（人）去量，才知道偏了多少。')}],
 transfer:[
  {q:tx(`A new category, translation, is graded by a program. A fifth of its items have a winner (q = 0.2), and the candidate is in fact 2 points worse. Suppose the measured difference is exactly that, as everywhere in this lab. The half-width is ${MI('h = 1.96 \\sqrt{(q - \\Delta^2) / n}')}. About how many items does it take for the interval to lie wholly below 0?`,
        `新加一类：翻译，用程序判分。有一方答得更好的题占五分之一（q = 0.2），候选模型其实差 2 分。假设量出来的差也正好是这么多（和这个 Lab 的设定一样）。半宽是 ${MI('h = 1.96 \\sqrt{(q - \\Delta^2) / n}')}。大约要多少题，区间才整个在 0 的左边？`),
   o:[tx('About 300','约 300 题'),tx('About 2,000','约 2,000 题'),tx('About 20,000','约 20,000 题')],a:1,
   why:tx(`The half-width must be under 2 points, 0.02: ${MI(`n = \\frac{1.96^2 \\times (0.2 - 0.02^2)}{0.02^2} \\approx ${tn(nNew)}`)} items. In a real evaluation the measured difference wanders, so with that many items the loss shows only half the time. Seeing it four times in five takes about twice as many: ${MI(`n = \\frac{(1.96 + 0.84)^2 (q - \\Delta^2)}{\\Delta^2} \\approx ${tn(n80)}`)}, about ${fint(Math.round(n80/100)*100)} items; a measured difference lands less than 0.84 standard errors above the true one four times in five.`,
                `半宽要小于 2 分，也就是 0.02：${MI(`n = \\frac{1.96^2 \\times (0.2 - 0.02^2)}{0.02^2} \\approx ${tn(nNew)}`)} 题。真实评测里量出来的差会晃动：约 2,000 题时，只有一半的机会看出这 2 分的退步。要有八成把握，题数要多一倍左右：${MI(`n = \\frac{(1.96 + 0.84)^2 (q - \\Delta^2)}{\\Delta^2} \\approx ${tn(n80)}`)}，约 ${fint(Math.round(n80/100)*100)} 题。0.84 对应八成把握：量出来的差有八成的机会，比真实的差高出不到 0.84 个标准误差。`)},
  {q:tx('In an A/B test, the overall regenerate rate is flat, but the coding guardrail rises with its interval wholly above 0. What do you do?','A/B 实验里，全部对话的重新生成率没变，编程类的护栏却上升了，区间整个在 0 的右边。你怎么做？'),
   o:[tx('Stop the rollout and look into coding, because a guardrail sounded its alarm','停止放量，查编程这一类：护栏报警了'),tx('Keep rolling out, because the overall rate is flat','继续放量：全部对话的指标没变'),tx('Double the traffic and look at the overall rate again','把流量加倍，再看全部对话的指标')],a:0,
   why:tx('A guardrail exists for exactly this: the overall rate averages coding’s rise away, just as the offline total hid writing’s loss.','护栏就是为这种情况设的：全部对话的指标把编程的上升平均掉了，就像离线的总分藏住了写作的退步。')}],
 practice:[{id:'llm-eval-system',d:tx('Design a weekly release gate across many task domains: per-domain sample sizes, judge calibration, and a non-inferiority rule.','设计一个跨多个任务领域的每周上线关卡：每个领域的样本量、裁判校准、非劣效规则。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'eval',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Eval capstone','评测综合挑战'),title:l?L(l.q):tx('How do you evaluate an LLM assistant before and after release?','怎么给一个 LLM 助手设计上线前后的评测？'),
    minutes:20,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: intervals and judges','前置知识：区间和裁判'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage ecap-stage" id="ecap-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="ecap-st-title"></div><div class="meter-sub" id="ecap-st-sub"></div></div><div class="stage-actions"><span class="status" id="ecap-status"></span></div></div>`+
   `<div class="stage-graph graph ecap-graph" id="ecap-graph"><div class="ecap-phone" id="ecap-phone"></div><div class="ecap-leg legend-row" id="ecap-leg"></div><div class="ecap-fp" id="ecap-fp"></div></div>`+
   `<div class="metrics ecap-metrics" id="ecap-metrics"></div>`+
   `<details class="math" id="ecap-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ecap-mathbody"></div></details></div>`;
}
const segM=c=>A.ui.seg({id:`ecap-ctl-m-${c.id}`,key:`m-${c.id}`,label:L(CN[c.id]),options:(c.prog?METHS:['judge','human']).map(m=>[m,L(MN[m])]),value:S.m[c.id]});
const segN=(c,pre)=>A.ui.seg({id:`${pre}-n-${c.id}`,key:`n-${c.id}`,label:tx('Items','题数'),options:SIZES.map(n=>[String(n),fint(n)]),value:String(S.n[c.id])});
const segCtl=(id,inline)=>A.ui.seg({id,key:'ctl',label:tx('How the judge runs','裁判怎么判'),options:[['0',tx('judge once','判一次')],['1',tx('both orders, length-corrected','两种顺序都判，<span class="ecap-nw">按长度校正</span>')]],value:String(S.ctl),inline});
const segCal=(id,inline)=>A.ui.seg({id,key:'cal',label:tx('People check per judged category','每个用裁判的类别，人工校准的题数'),options:CALS.map(n=>[String(n),fint(n)]),value:String(S.cal),inline});
const ASSUME=()=>[
 [tx('Prices','价格'),tx('a judge call $0.01; a person grading one item $2; program checks and generating the answers are not counted','裁判每次 $0.01；人判一题 $2；程序判分和模型回答题目的花费不计')],
 [tx('People’s speed','人工速度'),tx('400 items a day, all categories together','每天 400 题，所有类别合计')],
 [tx('Judge’s bias, judged once (the candidate’s answer shown first)','裁判判一次的偏差（候选的回答排在前面）'),tx(`writing ${pts(CAT.write.jb[0])} points, safety ${pts(CAT.safe.jb[0])}, the checkable categories ${pts(CAT.code.jb[0])}`,`写作 ${pts(CAT.write.jb[0])} 分，安全 ${pts(CAT.safe.jb[0])} 分，能用程序判的三类 ${pts(CAT.code.jb[0])} 分`)],
 [tx('Judge’s bias, both orders, length-corrected','两种顺序、按长度校正后的偏差'),tx(`writing ${pts(CAT.write.jb[1])} points (self-preference), the rest 0; the length correction removes the length preference completely`,`写作 ${pts(CAT.write.jb[1])} 分（自我偏好），其他类 0；按长度校正能把长度的偏好完全去掉`)],
 [tx('Judge and person disagree','裁判和人不一致'),tx(`on ${Math.round(CAT.write.je[1]*100)}% of writing items judged in both orders and length-corrected (${Math.round(CAT.write.je[0]*100)}% judged once); ${JE()} elsewhere`,`两种顺序、按长度校正时写作有 ${Math.round(CAT.write.je[1]*100)}% 的题不一致（判一次时 ${Math.round(CAT.write.je[0]*100)}%）；其他类 ${JE()}`)],
 [tx('Items with a winner, q','有输赢的题 q'),tx(CATS.map(c=>`${CN[c.id].en} ${Math.round(c.q*100)}%`).join(', '),CATS.map(c=>`${CN[c.id].zh} ${Math.round(c.q*100)}%`).join('，'))]];
function controlsHTML(){
  return `<aside class="panel controls" id="ecap-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="ecap-lock" hidden></p>`+
   `<div class="lockable" id="ecap-ctl-wrap"><p class="ctl-hint small muted">${tx('Each category: how to grade it, then how many items. The chart shows its verdict.','每一类先选判分方法，再选题数；图里是它的结论。')}</p>`+
   CATS.map(c=>`<div class="ecap-cg" id="ecap-cg-${c.id}">${segM(c)}${segN(c,'ecap-ctl')}</div>`).join('')+
   `<div class="ecap-cg">${segCtl('ecap-ctl-ctl',false)}${segCal('ecap-ctl-cal',false)}<p class="ctl-note">${tx('Calibration: people also grade this many of the judge’s items in every category the judge grades, and the judge’s bias measured on them is taken out.','人工校准：每个用裁判的类别，都从裁判判过的题里抽这么多题请人判，用它们量出裁判的偏差，再减掉。')}</p></div>`+
   `<div class="ecap-grp"><div class="ecap-gh">${tx('Values this lab assumes','本 Lab 假设的值')}</div><ul class="ecap-asm">${ASSUME().map(([k,v])=>`<li><b>${esc(k)}</b> ${esc(v)}</li>`).join('')}</ul></div>`+
   `<div id="ecap-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></div></aside>`;
}
const CODE=`import math

def interval(win, lose, n, z=1.96):
    """Item-by-item difference with its 95% interval."""
    d = (win - lose) / n
    q = (win + lose) / n
    h = z * math.sqrt((q - d * d) / n)
    return d, d - h, d + h

def ppi(judge_rest, judge_chk, people_chk, z=1.96):
    """Judge on the unchecked items, corrected by people."""
    mean = lambda v: sum(v) / len(v)
    def var(v):
        m = mean(v)
        return sum((x - m) ** 2 for x in v) / len(v)
    gap = [p - j for p, j in zip(people_chk, judge_chk)]
    d = mean(judge_rest) + mean(gap)
    se = math.sqrt(var(judge_rest) / len(judge_rest)
                   + var(gap) / len(gap))
    return d, d - z * se, d + z * se

def verdict(lo, hi, margin=0.03):
    if lo > 0: return "better"
    if hi < 0: return "worse"
    if lo > -margin: return "not worse by over 3"
    return "can't tell yet"

# writing graded by people: 130 wins, 170 losses
d, lo, hi = interval(130, 170, 1000)
print(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", verdict(lo, hi))`;
const CODE_OUT='-0.040 [-0.074, -0.006] worse';
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ecap-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ecap-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A category’s difference with its interval, the judge corrected by people (PPI with λ = 1), and the verdict.','一类题的差和区间、用人工修正的裁判结果（λ = 1 的 PPI），以及结论。')}</p><pre class="code" id="ecap-code">${esc(CODE+'\n# '+CODE_OUT)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ecap-prod">`+
   tx(`<li>${cite(SRC.ppi,'<code>ppi_py/ppi.py</code>')}: <code>ppi_mean_ci</code> takes people’s labels, the model’s predictions on the same items and its predictions on the items people did not label, and returns the interval; the 95% interval used here needs <code>alpha=0.05</code> (the default, <code>alpha=0.1</code>, gives 90%). By default it estimates the best weight λ from the data (PPI++); λ = 1 is plain PPI, as here, and λ = 0 is the classical interval from people’s labels alone.</li>`+
      `<li>${cite(SRC.lme,'lm-evaluation-harness\u00a0<code>lm_eval/<wbr>api/<wbr>metrics.py</code>')}: <code>mean_stderr</code> reports a score’s standard error, the sample standard deviation over ${MI('\\sqrt{n}')}.</li>`+
      `<li>${cite(SRC.fc,'FastChat\u00a0<code>fastchat/<wbr>llm_judge/<wbr>common.py</code>')}: <code>play_a_match_pair</code> judges each pair in both orders, and <code>load_pairwise_model_judgments</code> records two different winners as “inconsistent”.</li>`+
      `<li>A common interview answer for a release gate: check every category, each must be not worse than a margin and at least one better, then watch the rollout with guardrails by category.</li>`+
      `<li>Offline items must be new to both models: items that leaked into training data inflate the score (${LL('contamination')}).</li>`,
      `<li>${cite(SRC.ppi,'<code>ppi_py/ppi.py</code>')}：<code>ppi_mean_ci</code> 接收人工标注、模型在同一批题上的判断和它在人没有标注的题上的判断，返回区间；这里用的 95% 区间要传 <code>alpha=0.05</code>（默认 <code>alpha=0.1</code>，给的是 90% 区间）。默认它会从数据里估一个最好的权重 λ（PPI++）；λ = 1 就是这里用的普通 PPI，λ = 0 就是只用人工标注的经典区间。</li>`+
      `<li>${cite(SRC.lme,'lm-evaluation-harness 的 <code>lm_eval/api/metrics.py</code>')}：<code>mean_stderr</code> 报告分数的标准误差，即样本标准差除以 ${MI('\\sqrt{n}')}。</li>`+
      `<li>${cite(SRC.fc,'FastChat 的 <code>fastchat/llm_judge/common.py</code>')}：<code>play_a_match_pair</code> 把每一对按两种顺序各判一次；<code>load_pairwise_model_judgments</code> 读结果时，两次赢家不同就记为“inconsistent”。</li>`+
      `<li>面试里设计上线关卡，常见的答法是逐类检查：每一类都不能变差超过一个幅度，至少一类要变好；放量时再按类别看护栏指标。</li>`+
      `<li>离线评测要用两个模型都没见过的题：题目混进训练数据的话，分数会虚高（见${LL('contamination')}）。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.ppiP,'Prediction-Powered Inference')}<span>${tx('Angelopoulos et al., 2023, arXiv 2301.09633','Angelopoulos 等，2023，arXiv 2301.09633')}</span></li>`+
   `<li>${cite(SRC.ppi2,'PPI++: Efficient Prediction-Powered Inference')}<span>${tx('Angelopoulos, Duchi and Zrnic, 2023, arXiv 2311.01453','Angelopoulos、Duchi、Zrnic，2023，arXiv 2311.01453')}</span></li>`+
   `<li>${cite(SRC.zheng,'Judging LLM-as-a-Judge with MT‑Bench and Chatbot Arena')}<span>${tx('Zheng et al., 2023, arXiv 2306.05685','Zheng 等，2023，arXiv 2306.05685')}</span></li>`+
   `<li>${cite(SRC.lc,'Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators')}<span>${tx('Dubois et al., 2024, arXiv 2404.04475','Dubois 等，2024，arXiv 2404.04475')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ecap-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ecap-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="ecap-app m2" id="ecap-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ecap-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see which verdicts it gets right, what it costs and how long it takes.','选一个场景，看看它哪些结论对、花多少钱、要几天。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ecap-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change any category’s grading under ${J('controls','Settings')}.`,`想改某一类怎么判，在“${J('controls','自由调参')}”里改。`)}</p>${toChal()}</div>`;
}
function challengeHTML(){
  const submit=`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting runs your plan: it shows each category’s verdict and the true differences, and tells you whether you met the goal.','提交后会运行你的计划：显示每一类的结论和真实的差，并告诉你有没有达成目标。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Right verdicts for ${usd(BUDGET)} in ${DEADLINE} days`,`花不超过 ${usd(BUDGET)}、${DEADLINE} 天，让五类的结论都对`)}</h3>`+
   `<p>${tx(`The same candidate and current model, the same five categories. For each category pick how to grade it and how many items; then set the judge and how many items people check.</p><p>The goal:</p><ul class="ecap-vl"><li>every category gets a verdict (better, worse, or not worse by more than 3 points);</li><li>every verdict matches the true difference, so writing’s loss is caught;</li><li>at least one category is shown to be better;</li><li>grading costs at most ${usd(BUDGET)}: people cost $${HUM} an item;</li><li>the judge costs $${JCALL} a call, and program checks are free;</li><li>people label at most ${PERDAY} items a day, so finish within ${DEADLINE} days.</li></ul><p>Of the plans that meet the goal, the cheapest is best. A plan always gives the same result here.`,
     `还是这个候选模型和当前模型，还是这五类题。给每一类选判分方法和题数，再定裁判怎么判、人工校准多少题。</p><p>目标是：</p><ul class="ecap-vl"><li>每一类都有结论（变好了、变差了，或者没有变差超过 3 分）；</li><li>每个结论都和真实的差一致，也就抓到了写作的退步；</li><li>至少有一类确定变好了；</li><li>判分最多花 ${usd(BUDGET)}，人判一题 $${HUM}，裁判每次 $${JCALL}，程序判分不花钱；</li><li>人工每天最多标 ${PERDAY} 题，要在 ${DEADLINE} 天内完成。</li></ul><p>在能达标的计划里，花钱最少的最好。这里同一个计划每次都得到同样的结果。`)}</p>`+
   `<div class="ecap-chg">`+CATS.map(c=>`<div class="ecap-cg">${A.ui.seg({id:`ecap-ic-m-${c.id}`,key:`m-${c.id}`,label:L(CN[c.id]),options:(c.prog?METHS:['judge','human']).map(m=>[m,L(MN[m])]),value:S.m[c.id],inline:true})}${A.ui.seg({id:`ecap-ic-n-${c.id}`,key:`n-${c.id}`,label:tx('Items','题数'),options:SIZES.map(n=>[String(n),fint(n)]),value:String(S.n[c.id]),inline:true})}</div>`).join('')+
   `<div class="ecap-cg">${segCtl('ecap-ic-ctl',true)}${segCal('ecap-ic-cal',true)}</div></div>`+
   `<div class="goal mono ecap-goal" id="ecap-ch-g1"><span>${tx('Cost','花费')}</span><span><b id="ecap-ch-cost"></b>${tx(` (goal: at most ${usd(BUDGET)})`,`（目标：不超过 ${usd(BUDGET)}）`)}</span></div>`+
   `<div class="goal mono ecap-goal" id="ecap-ch-g2"><span>${tx('Days','天数')}</span><span><b id="ecap-ch-days"></b>${tx(` (goal: at most ${DEADLINE})`,`（目标：不超过 ${DEADLINE} 天）`)}</span></div>`+
   submit+`<div id="ecap-ch-res" class="ecap-ch-res" aria-live="polite"></div></div>`;
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY={mission:{en:'Pick an answer to continue.',zh:'选一个答案才能继续。'},split:{en:'Show each category to continue.',zh:'选“按类别分开看”才能继续。'},
  size:{en:'Try fewer items to continue.',zh:'试一个更小的题数才能继续。'},judge:{en:'Try the judge in both orders to continue.',zh:'试一下两种顺序都判的裁判才能继续。'},
  calib:{en:'Pick a calibration that catches the loss to continue.',zh:'选一个能抓到退步的校准题数才能继续。'},online:{en:'Run the A/B test for 7 days to continue.',zh:'让 A/B 实验跑 7 天才能继续。'}};
const nextWhy=()=>{const w=WHY[STEPS[step].id]; return w?L(w):'';};
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:esc(nextWhy()),skip:false});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#ecap-rail',app); if(!rail) return null; rail.innerHTML=railFull(); renderPhone(); return rail;}
function renderRail(){if(fillRail()) render();}
function setObs(box,html){if(box._h!==html){const again=box._h!=null&&box._h!==''&&html!==''; box.innerHTML=html; box._h=html; if(again) A.ui.obsBump(box);}}
function updateRail(){
  const rail=$('#ecap-rail',app); if(!rail) return;
  $$('.seg-ctl[data-key="split"]',app).forEach(g=>A.ui.setPressed(g,split?'1':'0'));
  $$('.seg-ctl[data-key="gsize"]',app).forEach(g=>A.ui.setPressed(g,String(S.n.code)));
  $$('.seg-ctl[data-key="gcal"]',app).forEach(g=>A.ui.setPressed(g,String(S.cal)));
  $$('.seg-ctl[data-key="abd"]',app).forEach(g=>A.ui.setPressed(g,String(abDays)));
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    if(s.id==='judge'){$$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===p4Sel)); const rr=b.querySelector('.lv-r'), h=p4Tried.has(v)||v===p4Sel?p4Res(v):''; if(rr&&rr._h!==h){rr.innerHTML=h; rr._h=h;}}); renderPhone();}
    if(s.id==='calib') renderPhone();
    const show=!!s.obs&&(s.id!=='split'||split)&&(s.id!=='size'||sizeClicked);   // part 3: no result before the reader picks a size (R3.8)
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    A.ui.setNext(rail,isNextDisabled(),esc(nextWhy()));
    $$('[data-nav="showme"]',app).forEach(sm=>{sm.hidden=!!observed[step];});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#ecap-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):''; ins.hidden=!html; setObs(ins,html);}
  } else {
    const e=E(), c=$('#ecap-ch-cost',rail), d=$('#ecap-ch-days',rail); if(!c) return;
    c.textContent=usd(e.cost); d.textContent=tx(`${e.days} day${e.days>1?'s':''}`,`${e.days} 天`);
    $('#ecap-ch-g1',rail).classList.toggle('ecap-over',e.cost>BUDGET+1e-9); $('#ecap-ch-g2',rail).classList.toggle('ecap-over',e.days>DEADLINE);
    CATS.forEach(cc=>{$$(`.seg-ctl[data-key="m-${cc.id}"]`,app).forEach(g=>A.ui.setPressed(g,S.m[cc.id])); $$(`.seg-ctl[data-key="n-${cc.id}"]`,app).forEach(g=>A.ui.setPressed(g,String(S.n[cc.id])));});
    if(chSubmitted&&chPass(e)) chReveal=true;
    setObs($('#ecap-ch-res',rail),chResultHTML());
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
const AX={lo:-15,hi:15,ticks:[-15,-10,-5,0,5,10,15]}, AXO={lo:-2,hi:4,ticks:[-2,-1,0,1,2,3,4]};
const pos=(v,ax)=>Math.max(0,Math.min(100,(v-ax.lo)/(ax.hi-ax.lo)*100));
const pcs=v=>v.toFixed(2)+'%';
const spanAt=(a,b,ax)=>{const l=pos(Math.min(a,b),ax), r=pos(Math.max(a,b),ax); return `left:${pcs(l)};width:${pcs(Math.max(0.6,r-l))}`;};
/* a tooltip that lists a row's fields, names and values in two aligned columns (R2.11, P20); the kit escapes the text, so none here */
const tipL=(title,pairs,tail)=>`<p><b>${title}</b></p><div class="ecap-tl">${pairs.filter(Boolean).map(([k,v])=>`<div><span>${k}</span><b>${v}</b></div>`).join('')}</div>${tail?`<p>${tail}</p>`:''}`;
const ptsT=x=>tx(`${fD(x)} points`,`${fD(x)} 分`), ivT=(lo,hi)=>tx(`${fB(lo)} to ${fB(hi)}`,`${fB(lo)} 到 ${fB(hi)}`);
/* the judge's lean, drawn under its row from the true difference (R6.25, R6.26, P11): self-preference first (solid), then the position
   and length preferences (dashed), or, in both orders with length taken out, those two as a faint dotted span marked removed. The
   arrowhead sits where the judge's result is. Only an unchecked judge: a calibrated one is drawn by calOf */
function biasOf(r){
  if(r.meth!=='judge'||r.mm) return null;
  const c=CAT[r.id], self=c.jb[1], pl=c.jb[0]-c.jb[1], on=!!S.ctl, segs=[]; let x=c.d;
  if(c.jb[0]<5e-4) return null;
  if(self>5e-4){segs.push({k:'s',a:x,b:x+self}); x+=self;}
  if(pl>5e-4) segs.push({k:on?'x':'p',a:x,b:x+pl});
  return {segs,tip:on?(self>5e-4?c.d+self:null):c.d+c.jb[0]};
}
/* calibration (P15): the judge's own result on all its items, hollow, with that interval dashed; the bias people measured, taken off */
function calOf(r){
  if(r.meth!=='judge'||!r.mm||!r.nu) return null;
  const hj=Z*Math.sqrt((CAT[r.id].q-r.Dj*r.Dj)/r.n);
  return {D:r.Dj,lo:r.Dj-hj,hi:r.Dj+hj};
}
/* the line under writing's row that names what the arrow shows */
function annOf(r,bias,raw){
  if(raw) return `<span class="a1"><i class="ecap-sw rw"></i>${tx(`judge alone ${fD(r.Dj)}`,`裁判量出 ${fD(r.Dj)}`)}</span>`+
    `<span class="a2"><i class="ecap-sw cor"></i>${tx(`minus the bias people found, ${fA(r.b)}`,`减去人工量出的偏差 ${fA(r.b)}`)}</span>`;
  if(!bias) return '';
  const c=CAT[r.id], self=c.jb[1], pl=c.jb[0]-c.jb[1], on=!!S.ctl;
  return (self>5e-4?`<span class="a1"><i class="ecap-sw bs"></i>${tx(`self-preference ${fD(self)}${on?', still there':''}`,`自我偏好 ${fD(self)}${on?'，还在':''}`)}</span>`:'')+
    (pl>5e-4?`<span class="a2"><i class="ecap-sw ${on?'bx':'bp'}"></i>${on?tx(`position and length ${fD(pl)}, removed`,`位置和长度 ${fD(pl)}，已去掉`):tx(`position and length ${fD(pl)}`,`位置和长度 ${fD(pl)}`)}</span>`:'');
}
function rowHTML(o,ax){
  const lo=o.lo*100, hi=o.hi*100, d=o.D*100;
  const bar=o.hide?'':`<i class="ecap-bar ecap-m${o.red?' red':''}" style="${spanAt(lo,hi,ax)}"></i><i class="ecap-pt ecap-m${o.red?' red':''}" style="left:${pcs(pos(d,ax))}"></i>`+
    (lo<ax.lo?`<i class="ecap-cut l"></i>`:'')+(hi>ax.hi?`<i class="ecap-cut r"></i>`:'');
  const gh=o.ghost?`<i class="ecap-ghost" style="${spanAt(o.ghost.lo*100,o.ghost.hi*100,ax)}"></i>`:'';
  const bs=o.bias?o.bias.segs.map(s=>`<i class="ecap-bs ${s.k}" style="${spanAt(s.a*100,s.b*100,ax)}"></i>`).join('')+(o.bias.tip!=null?`<i class="ecap-ah r" style="left:${pcs(pos(o.bias.tip*100,ax))}"></i>`:''):'';
  const cal=o.raw?`<i class="ecap-rawb" style="${spanAt(o.raw.lo*100,o.raw.hi*100,ax)}"></i><i class="ecap-rawp" style="left:${pcs(pos(o.raw.D*100,ax))}"></i>`+
    `<i class="ecap-bs cor" style="${spanAt(d,o.raw.D*100,ax)}"></i><i class="ecap-ah l" style="left:${pcs(pos(d,ax))}"></i>`:'';
  const tr=o.truth!=null?`<i class="ecap-tru" style="left:${pcs(pos(o.truth*100,ax))}" data-tipx="${esc(TIP.truth()+tx(` Here: ${fD(o.truth)} points.`,`这里是 ${fD(o.truth)} 分。`))}"></i>`:'';
  const cls=['ecap-row',o.cls,o.bias||o.raw?'has-b':'',o.raw?'has-r':'',o.anim?'ecap-anim '+o.anim:''].filter(Boolean).join(' ');
  return `<div class="${cls}" data-row="${o.id}" data-tipx="${esc(o.tip)}"><div class="ecap-rn"><b>${o.name}</b><span>${o.sub}</span></div>`+
    `<div class="ecap-tr"><i class="ecap-zero" style="left:${pcs(pos(0,ax))}"></i>${o.mg?`<i class="ecap-mg" style="left:${pcs(pos(-3,ax))}"></i>`:''}${gh}${cal}${bar}${bs}${tr}</div>`+
    `<div class="ecap-vd${o.vcls?' '+o.vcls:''}">${o.vd}</div>${o.ann?`<div class="ecap-ann">${o.ann}</div>`:''}</div>`;
}
function axisHTML(ax,label){
  return `<div class="ecap-row ecap-axis"><div class="ecap-rn"></div><div class="ecap-tr">${ax.ticks.map(t=>`<span class="${t%(ax.hi>10?10:2)?'mn':'mj'}" style="left:${pcs(pos(t,ax))}">${t>0?'+':''}${t<0?MINUS+Math.abs(t):t}</span>`).join('')}</div><div class="ecap-vd"></div></div>`+
    `<div class="ecap-axl">${label}</div>`;
}
/* the build-up (P11, P15): about 1 s, only in parts 4 and 5, only when the reader's click changes what the judge's row draws; the
   point, its interval, the arrow and the verdict share one clock (CSS delays from the same start). A repeated click, or any later
   redraw, drops the class and shows the end state (R3.20); a new mount starts clean, so a language switch does not replay it */
let rowSig={}, buildNow='';
function animOf(id,sig,kind){const changed=rowSig[id]!==sig; rowSig[id]=sig;
  return changed&&kind&&userAct&&mode==='guided'&&(isP('judge')||isP('calib'))?kind:'';}
/* a plan's line keeps each number with its word (“300 题”), so a narrow name column breaks it after a comma (P19) */
const nbN=s=>s.replace(/ (?=\d)|(?<=\d) /g,'\u00a0');
/* No line holds only one or two characters (N5, B5), at any width: in every block of text the last three characters stay on one line,
   and a number stays with its unit (“100 题”). It is done to the page as it is drawn, so every text gets it, now and later; the spans
   change no text. A block whose text sits directly in a flex or grid box is left, and so is one that ends in a long term or formula. */
const GLUE_IN='svg,math,code,pre,kbd,select,textarea,input,.seg-ctl,.ecap-wd', GLUE_STOP='br,img,input,select,textarea,button,svg,pre';
const GLUE_CH=/[A-Za-z0-9$\u3400-\u9fff\uf900-\ufaff]/, GLUE_CHG=/[A-Za-z0-9$\u3400-\u9fff\uf900-\ufaff]/g, GLUE_RUN=/[A-Za-z0-9$,.%]/;
const GLUE_UZ=/\d[\d,.]*%? [分题天倍类次个秒]/g, GLUE_U0Z=/^ [分题天倍类次个秒]/, GLUE_HAN=/[\u3400-\u9fff\uf900-\ufaff]/;
/* in English the number keeps the word after it (“1,000 items”, “3 days”, “2.4 points”, “95% interval”), also when a hyphen joins
   them (“a 4-point loss”) */
const GLUE_UW='(?:items?|days?|d|points?|times|categories|conversations|calls|labels|checked|wrong|per [a-z]+|interval)(?![\\w-])';
const GLUE_UE=new RegExp('\\d[\\d,.]*%?[ -]'+GLUE_UW,'g'), GLUE_U0E=new RegExp('^[ -]'+GLUE_UW);
/* a Chinese word is not cut where the kept end begins (“按长 / 度校正”): the end starts with the whole word */
const GLUE_SEG=typeof Intl!=='undefined'&&Intl.Segmenter?new Intl.Segmenter('zh',{granularity:'word'}):null;
const wordStart=(s,k)=>{if(!GLUE_SEG||k<=0||!GLUE_HAN.test(s[k])||!GLUE_HAN.test(s[k-1])) return k;
  for(const w of GLUE_SEG.segment(s)){if(w.index>=k) break; if(w.index+w.segment.length>k) return k-w.index<=3?w.index:k;} return k;};
let glueMo=null;
const dispOf=e=>getComputedStyle(e).display, boxy=e=>/flex|grid/.test(dispOf(e)), inl=e=>/^(inline|math|contents)$/.test(dispOf(e));
function glueBlockOf(t){let e=t.parentElement; while(e&&e!==app){if(!inl(e)) return e; e=e.parentElement;} return null;}
/* a span is kept only where it changes nothing but the line breaks: a box that styles every span in it (an axis label, a legend) is
   left as it was */
const GLUE_SAME=['fontSize','fontWeight','fontFamily','fontStyle','color','letterSpacing','verticalAlign'], glueNo=new WeakSet();
const gluePlain=sp=>{const c=getComputedStyle(sp), p=getComputedStyle(sp.parentElement), z=k=>!parseFloat(c[k]);
  return c.display==='inline'&&c.position==='static'&&c.float==='none'&&c.transform==='none'&&['marginLeft','marginRight','paddingLeft','paddingRight','borderLeftWidth','borderRightWidth'].every(z)
    &&/^(transparent|rgba\(0, 0, 0, 0\))$/.test(c.backgroundColor)&&GLUE_SAME.every(k=>c[k]===p[k]);};
const glueUndo=sp=>{const par=sp.parentElement; glueNo.add(par); sp.replaceWith(...sp.childNodes); par.normalize();};
const glueSpan=(cls,nodes,tag)=>{const sp=document.createElement(tag||'span'); sp.className=cls; nodes[0].before(sp); sp.append(...nodes); if(gluePlain(sp)) return sp; glueUndo(sp); return null;};
/* “100 题” in one text, or “<b>49</b> 题” across an element and the text after it */
function glueNums(t){
  const p=t.parentElement; if(!p||glueNo.has(p)||p.closest(GLUE_IN+',.ecap-nw')||boxy(p)) return;
  const pv=t.previousSibling, zh=A.lang()==='zh', GLUE_U=zh?GLUE_UZ:GLUE_UE, m0=t.data.match(zh?GLUE_U0Z:GLUE_U0E);
  if(m0&&pv&&pv.nodeType===1&&inl(pv)&&!pv.matches(GLUE_STOP)&&/\d%?$/.test(pv.textContent)&&pv.textContent.length<=10){
    if(t.data.length>m0[0].length) t.splitText(m0[0].length); glueSpan('ecap-nw',[pv,t]); return;}
  const s=t.data; GLUE_U.lastIndex=0; if(!GLUE_U.test(s)) return;
  const f=document.createDocumentFragment(); let i=0, m; GLUE_U.lastIndex=0;
  const made=[];
  while((m=GLUE_U.exec(s))){if(m.index>i) f.append(s.slice(i,m.index)); const sp=document.createElement('span'); sp.className='ecap-nw'; sp.textContent=m[0]; f.append(sp); made.push(sp); i=m.index+m[0].length;}
  if(i<s.length) f.append(s.slice(i)); t.replaceWith(f);
  if(made.length&&!gluePlain(made[0])) made.forEach(glueUndo);}
/* the end of a block: its last three characters (a word or a number kept whole), with any short element among them, on one line */
function glueTail(B){
  if(!B.isConnected||glueNo.has(B)||boxy(B)||B.closest(GLUE_IN)) return;
  const tw=document.createTreeWalker(B,NodeFilter.SHOW_TEXT); let L=null, n;
  while((n=tw.nextNode())) if(n.data.trim()&&glueBlockOf(n)===B&&!n.parentElement.closest('math')) L=n;
  if(!L||L.parentElement.closest(GLUE_IN)) return;
  const nw=L.parentElement.closest('.ecap-nw');
  let x=nw&&B.contains(nw)?nw:L, par=x.parentElement, need=3, at=-1, atoms=[];
  for(;;){
    for(; x; x=x.previousSibling){
      if(x.nodeType===3){const s=x.data; let k=s.length; while(k>0&&need>0){k--; if(GLUE_CH.test(s[k])) need--;}
        atoms.unshift(x); if(need<=0){while(k>0&&GLUE_RUN.test(s[k])&&GLUE_RUN.test(s[k-1])) k--; at=wordStart(s,k); break;}}
      else if(x.nodeType===1){if(x.matches(GLUE_STOP)||!inl(x)||x.textContent.length>10) return;
        atoms.unshift(x); need-=(x.textContent.match(GLUE_CHG)||[]).length; if(need<=0) break;}
    }
    if(need<=0||par===B||!par||!inl(par)) break;
    atoms=[par]; x=par.previousSibling; par=par.parentElement;
  }
  if(need>0) return;
  const len=atoms.reduce((a,y,i)=>a+(i===0&&at>=0?y.data.length-at:y.textContent.length),0); if(len>12) return;
  if(at>0) atoms[0]=atoms[0].splitText(at);
  if(!glueSpan('ecap-wd',atoms)) glueNo.add(B);
}
/* English (todo 6): words, not characters. A block's last three words stay on one line, so no line ends a block with one or two words
   alone (“No / verdict / yet”); where the three would take most of the block's width (a narrow card), the last two; where even those
   would, the block keeps its breaks. A short block keeps two words ahead of its end, so its first line is not one word either. The
   end is counted in what a line can break between (a no-break space joins “3,000 items”). The ends are measured together, once the
   page is laid out; a hidden block keeps a short end only. The end is held by an element of its own name, not a span, so a box that
   styles its spans (a scenario's line under its title) leaves it alone */
const GLUE_BRK=/[ \t\n\r\f\u2009\u200b]/;
function glueEndEn(B,words){
  if(!B.isConnected||glueNo.has(B)||boxy(B)||B.closest(GLUE_IN)) return null;
  words=Math.min(words,B.textContent.trim().split(/\s+/).length-2); if(words<2) return null;
  const tw=document.createTreeWalker(B,NodeFilter.SHOW_TEXT); let L=null, n;
  while((n=tw.nextNode())) if(n.data.trim()&&glueBlockOf(n)===B&&!n.parentElement.closest('math')) L=n;
  if(!L||L.parentElement.closest(GLUE_IN)) return null;
  const nw=L.parentElement.closest('.ecap-nw');
  let x=nw&&B.contains(nw)?nw:L, par=x.parentElement, got=0, inW=false, atoms=[], at=-1;
  const scan=(s,txt)=>{for(let k=s.length-1;k>=0;k--){if(!GLUE_BRK.test(s[k])){inW=true; continue;} if(inW){inW=false; if(++got>=words){at=txt?k+1:-1; return true;}}} return false;};
  walk: for(;;){
    for(; x; x=x.previousSibling){
      if(x.nodeType===3){atoms.unshift(x); if(scan(x.data,true)) break walk;}
      else if(x.nodeType===1){if(x.matches(GLUE_STOP)||!inl(x)||x.textContent.length>30) return null; atoms.unshift(x); if(scan(x.textContent,false)) break walk;}
    }
    if(par===B||!par||!inl(par)) return null;
    atoms=[par]; x=par.previousSibling; par=par.parentElement;
  }
  if(at>=0&&at>=atoms[0].data.length){atoms.shift(); at=-1;}
  if(!atoms.length) return null;
  const len=atoms.reduce((a,y,i)=>a+(i===0&&at>=0?y.data.length-at:y.textContent.length),0); if(len>30) return null;
  if(at>0) atoms[0]=atoms[0].splitText(at);
  const sp=glueSpan('ecap-wd',atoms,'ecap-w'); return sp?{sp,B}:null;
}
const glueLet=sp=>{const par=sp.parentElement; sp.replaceWith(...sp.childNodes); par.normalize();};
function glueEndsEn(bs){
  const wide=list=>{const r=list.map(m=>{const c=getComputedStyle(m.B), cw=m.B.clientWidth-parseFloat(c.paddingLeft)-parseFloat(c.paddingRight);
    return cw>0?(m.sp.getBoundingClientRect().width>cw*0.9?2:0):(m.sp.textContent.length>12?1:0);}); return list.filter((m,i)=>{m.bad=r[i]; return r[i];});};
  let bad=wide([...bs].map(B=>glueEndEn(B,3)).filter(Boolean)); bad.forEach(m=>glueLet(m.sp));
  bad=wide(bad.map(m=>glueEndEn(m.B,2)).filter(Boolean)); bad.forEach(m=>{glueLet(m.sp); if(m.bad===2) glueNo.add(m.B);});
}
function glueIn(roots){
  if(!app) return; const els=new Set();
  roots.forEach(r=>{const e=r&&(r.nodeType===3?r.parentElement:r); if(e&&e.nodeType===1&&e.isConnected&&app.contains(e)) els.add(e);});
  const texts=()=>{const ts=[]; els.forEach(e=>{if(!e.isConnected) return; const tw=document.createTreeWalker(e,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())) ts.push(n);}); return ts;};
  texts().forEach(t=>{if(t.isConnected&&/\d|^ /.test(t.data)) glueNums(t);});
  const bs=new Set(); texts().forEach(t=>{if(t.data.trim()){const b=glueBlockOf(t); if(b) bs.add(b);}});
  if(A.lang()==='zh') bs.forEach(glueTail); else glueEndsEn(bs);
}
/* a block shown later (Show the math) is measured when it opens */
const glueOpen=e=>{if(e.target&&e.target.open&&glueMo){glueIn([e.target]); glueMo.takeRecords();}};
function glueOff(){if(glueMo){glueMo.disconnect(); glueMo=null;} if(app) app.removeEventListener('toggle',glueOpen,true);}
function glueOn(){glueOff(); if(!app||typeof MutationObserver==='undefined') return; glueIn([app]); app.addEventListener('toggle',glueOpen,true);
  glueMo=new MutationObserver(rs=>{const t=[]; rs.forEach(r=>{t.push(r.target); r.addedNodes.forEach(a=>t.push(a));}); glueIn(t); if(glueMo) glueMo.takeRecords();});
  glueMo.observe(app,{childList:true,subtree:true,characterData:true});}
const TOTV=t=>t.v==='better'?DN.tot:(t.v==='worse'?VN.worse:DN.totU);
function drawFP(){
  const host=$('#ecap-fp',app), leg=$('#ecap-leg',app); buildNow='';
  if(onlineView()){
    const a=abRows(abDays), R=[['tot',tx('All conversations','全部对话'),a.tot],['write',tx('Writing conversations','写作类对话'),a.write],['other',tx('The other four categories','其他四类对话'),a.other]];
    const alarmT=r=>r.alarm?tx('alarm','报警'):tx('no alarm yet','还没报警');
    const h=R.map(([id,name,r])=>rowHTML({id,name:esc(name),sub:esc(tx(`${fint(r.n)} per group`,`每组 ${fint(r.n)} 个`)),lo:r.lo,hi:r.hi,D:r.D,red:r.alarm,mg:false,
      vd:`<span class="ecap-vw">${alarmT(r)}</span>`,vcls:r.alarm?'bad':'',
      tip:tipL(name,[[tx('Conversations','每组对话'),tx(`${fint(r.n)} per group`,`${fint(r.n)} 个`)],[tx('Candidate minus current','候选减当前'),tx(`${fD(r.D)} points`,`${fD(r.D)} 个百分点`)],
        [tx('95% interval','95% 区间'),ivT(r.lo,r.hi)],[tx('Guardrail','护栏'),alarmT(r)]])},AXO)).join('');
    const ht=h+axisHTML(AXO,tx('Candidate minus current regenerate rate, in points (right of 0 is worse)','候选减当前的重新生成率（百分点；0 右边是变差）'));
    if(host._h!==ht){host.innerHTML=ht; host._h=ht;}
    const lh=`<span class="lg-m"><i class="ecap-sw pt"></i>${tx('measured difference and its 95% interval','量出来的差和 95% 区间')}</span><span class="lg-w"><i class="ecap-sw red"></i>${tx('alarm: interval wholly above 0','报警：区间整个在 0 右边')}</span>`;
    if(leg._h!==lh){leg.innerHTML=lh; leg._h=lh;}
    return;
  }
  const e=E(), mg=marginOn(), tr=truthOn(), hide=chHide(), only=totalOnly();
  const t=e.total, rows=[];
  /* part 3: each row keeps its interval at 1,000 items as a dashed box, so a new size shows how much it narrowed or widened (P7) */
  const g0=isP('size')&&S.n.code!==1000?evalPlan(PLAN0(),true):null;
  rows.push({id:'total',name:esc(tx('Total','总分')),sub:esc(tx('five categories, weighted by traffic','五类按流量加权')),lo:t.lo,hi:t.hi,D:t.D,cls:'tot',mg,hide,ghost:g0?g0.total:null,
    vd:hide?'':`<span class="ecap-vw">${A.lang()==='zh'?L(TOTV(t)):L(TOTV(t)).replace(/^./,c=>c.toLowerCase())}</span><span class="ecap-vn"><small>${fB(t.lo)} … ${fB(t.hi)}</small></span>`,
    tip:hide?tx('Submit to see the result.','提交后显示结果。'):tipL(tx('Total','总分'),[[tx('Weighting','加权'),tx('five categories by traffic','五类按流量加权')],[tx('Difference','差'),ptsT(t.D)],[tx('95% interval','95% 区间'),ivT(t.lo,t.hi)],[tx('Verdict','结论'),L(TOTV(t))]])});
  if(!only) e.rows.forEach((r,i)=>{const name=L(CN[r.id]), red=!hide&&r.v==='worse', wrong=!hide&&tr&&r.wrong, pl=planLine(r,S);
    const bias=!hide&&tr?biasOf(r):null, raw=!hide?calOf(r):null;
    const anim=animOf(r.id,JSON.stringify([r.D,r.lo,r.hi,bias,raw]),raw?'cal':(bias?'bias':''));
    rows.push({id:r.id,name:esc(name),sub:esc(nbN(pl)),lo:r.lo,hi:r.hi,D:r.D,red,mg,hide,truth:tr&&!hide?r.truth:null,cls:wrong?'wrong':'',ghost:g0?g0.rows[i]:null,bias,raw,anim,
      ann:r.id==='write'?annOf(r,bias,raw):'',
      vd:hide?'':`<span class="ecap-vw">${L(VN[r.v])}</span><span class="ecap-vn">${wrong?`<em class="ecap-wr">${tx('wrong','看错了')}</em>`:''}<small>${fB(r.lo)} … ${fB(r.hi)}</small></span>`,vcls:red?'bad':'',
      tip:hide?tipL(name,[[tx('Grading','判分'),pl]],tx('Submit to see the result.','提交后显示结果。')):
        tipL(name,[[tx('Grading','判分'),pl],[tx('Difference','差'),ptsT(r.D)],[tx('95% interval','95% 区间'),ivT(r.lo,r.hi)],[tx('Verdict','结论'),L(VN[r.v])+(wrong?tx(', wrong','，看错了'):'')],
          tr?[tx('True difference','真实的差'),ptsT(r.truth)]:null,bias?[tx('Judge’s bias','裁判的偏差'),ptsT(r.b)]:null,
          raw?[tx('Judge alone','裁判自己量出'),ptsT(r.Dj)]:null,raw?[tx('Bias people measured','人工量出的偏差'),ptsT(r.b)]:null])});});
  /* N10: while writing's row builds up, the total row, the badge and the verdict card wait for its result and appear with it, on the
     same clock (render() gives the badge and the card the same class) */
  const wb=rows.find(o=>o.id==='write'); if(wb&&wb.anim){rows[0].anim=wb.anim; buildNow=wb.anim;}
  const ht=rows.map(o=>rowHTML(o,AX)).join('')+axisHTML(AX,tx('Candidate minus current, in points (right of 0 is better)','候选减当前模型的差（分，0 右边是候选更好）'));
  if(host._h!==ht){host.innerHTML=ht; host._h=ht;}
  /* the measured and the true difference first, so on a phone the five entries fit three lines (P22) */
  const items=[`<span class="lg-m"><i class="ecap-sw pt"></i>${tx('measured difference and its 95% interval','量出来的差和 95% 区间')}</span>`];
  if(tr&&!hide) items.push(`<span class="lg-t" data-tipx="${esc(TIP.truth())}"><i class="ecap-sw tru"></i>${tx('true difference','真实的差')}</span>`);
  if(g0) items.push(`<span class="lg-g"><i class="ecap-sw gh"></i>${tx('dashed box: interval at 1,000 items each','虚线框：每类 1,000 题时的区间')}</span>`);
  if(!only) items.push(`<span class="lg-w"><i class="ecap-sw red"></i>${tx('worse: interval wholly below 0','变差了：区间整个在 0 左边')}</span>`);
  if(mg) items.push(`<span class="lg-mg"><i class="ecap-sw mg"></i>${tx('−3 points: the most loss accepted','−3 分：最多接受的退步')}</span>`);
  if(tr&&!hide) items.push(`<span class="lg-r"><em class="ecap-wr">${tx('wrong','看错了')}</em>${tx(': truth disagrees','：结论和真实的差不符')}</span>`);
  const lh=items.join('');
  if(leg._h!==lh){leg.innerHTML=lh; leg._h=lh;}
}
/* the rule for a plan's verdict, as a list (P20) */
const RULETIP=()=>tipL(tx('What this plan concludes','这个计划得出的结论'),[[L(DN.block),tx('a category is worse','有一类变差了')],[L(DN.ship),tx('every category has a verdict, none is worse, at least one is better','每一类都有结论，没有一类变差，至少一类变好')],[L(DN.unk),tx('any other case','其他情况')]],
  tx('A wrong verdict in a category makes this one wrong too.','某一类的结论错了，它也会跟着错。'));
function statusOf(){
  if(onlineView()){const a=abRows(abDays); return {cls:a.write.alarm?'bad':'neutral',txt:a.write.alarm?tx('Guardrail alarm: rollout stopped','护栏报警，停止放量'):tx('No alarm yet','还没报警'),short:a.write.alarm?tx('alarm','报警'):tx('no alarm yet','还没报警'),tip:tx('Whether the writing guardrail’s interval lies wholly above 0.','写作类护栏的区间是否整个在 0 的右边。')};}
  const e=E();
  if(mode==='challenge'){if(!chSubmitted) return {cls:'neutral',txt:tx(`Cost ${usd(e.cost)}`,`花费 ${usd(e.cost)}`),short:usd(e.cost),tip:tx('The cost of your plan. Submit to see the verdicts.','你的计划的花费。提交后显示结论。')};
    const p=chPass(e); return {cls:p?'ok':'bad',txt:p?tx('Goal met','达成目标'):tx('Not met yet','还没达标'),short:p?tx('met','达标'):tx('not met yet','还没达标'),tip:tx('Green when every verdict is right, writing’s loss is caught, and the plan fits the budget and the days.','结论都对、抓到写作的退步、不超预算和天数时，徽章是绿色。')};}
  if(totalOnly()){const t=e.total; return {cls:'neutral',txt:L(t.v==='better'?DN.tot:DN.totU),short:L(t.v==='better'?DN.tot:DN.totU),tip:tx('What the total alone says.','只看总分时的结论。')};}
  return {cls:'neutral',txt:L(DN[e.decision]),short:L(DN[e.decision]),tip:RULETIP()};
}
function drawHead(){
  const t=$('#ecap-st-title',app), sub=$('#ecap-st-sub',app), st=$('#ecap-status',app);
  if(onlineView()){t.textContent=tx('After release: did the candidate make users regenerate more?','上线以后：候选模型让用户更常点重新生成了吗？'); sub.textContent=tx(`A/B test, ${abDays} day${abDays>1?'s':''}, ${SH()} of conversations per model`,`A/B 实验 ${abDays} 天，每个模型 5% 的对话`);}
  else {t.textContent=totalOnly()?tx('How much better is the candidate overall?','候选模型总体上好多少？'):tx('How much better is the candidate per category?','候选模型在每一类上好多少？');
    const e=E(); sub.textContent=chHide()?tx('Your plan; submit to see the results','你的计划，提交后显示结果'):tx(`Grading: ${usd(e.cost)}, ${e.days} day${e.days>1?'s':''}`,`判分花费 ${usd(e.cost)}，${e.days} 天`);}
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
const numOf=v=>{const m=String(v).replace(/[,$]/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
/* the judge is named on the cost card only once part 4 has introduced it (P2) */
const judgeSeen=()=>mode!=='guided'||step>=SIX('judge');
function metricsDef(P){
  if(onlineView()){const a=abRows(abDays);
    return [{k:'days',label:tx('A/B test','A/B 实验'),v:tx(`${abDays} d`,`${abDays} 天`),sub:tx('days run so far','已经跑的天数'),tip:tx('How long the A/B test has run.','A/B 实验已经跑了多久。')},
      {k:'conv',label:tx('Group size','每组对话'),v:fint(a.arm),sub:tx(`${fint(AB.perDay*AB.share)} a day`,`每天 ${fint(AB.perDay*AB.share)} 个`),tip:tx(`Conversations each model has served in the test: ${SH()} of ${fint(AB.perDay)} a day each.`,`实验里每个模型服务过的对话：每天 ${fint(AB.perDay)} 个对话各占 ${SH()}。`)},
      {k:'guard',label:tx('Guardrail','写作护栏'),v:a.write.alarm?tx('alarm','报警'):tx('no alarm yet','还没报警'),sub:tx(`${fD(a.write.D)} points in writing`,`${fD(a.write.D)} 个百分点`),tip:tx('The writing conversations’ regenerate rate, candidate minus current.','写作类对话的重新生成率，候选减当前。')}];}
  const e=P?evalPlan(P,marginOn()):E(), hide=chHide(), js=judgeSeen();
  const dec=hide?tx('after Submit','提交后显示'):(totalOnly()?L(e.total.v==='better'?DN.tot:DN.totU):L(DN[e.decision]));
  return [{k:'cost',label:tx('Grading cost','判分花费'),v:usd(e.cost),sub:js?tx(`people ${usd(e.costH)}, judge ${usd(e.costJ)}`,`人工 ${usd(e.costH)}，裁判 ${usd(e.costJ)}`):tx(`people, $${HUM} an item`,`人工每题 $${HUM}`),
      tip:js?tx('People at $2 an item and the judge at $0.01 a call; program checks are free.','人工每题 $2，裁判每次 $0.01，程序判分不花钱。'):tx('People at $2 an item; program checks are free.','人工每题 $2，程序判分不花钱。')},
    {k:'days',label:tx('Time','用时'),v:tx(`${e.days} d`,`${e.days} 天`),sub:tx(`${fint(e.H)}\u00a0items for people`,`人工共 ${fint(e.H)}\u00a0题`),tip:js?tx('People label 400 items a day; the judge and program checks finish within the first day.','人工每天标 400 题；裁判和程序判分第一天之内就能做完。'):tx('People label 400 items a day; program checks finish within the first day.','人工每天标 400 题；程序判分第一天之内就能做完。')},
    {k:'dec',label:tx('Verdict','结论'),v:dec,sub:hide?'':(totalOnly()?tx('from the total alone','只看总分'):(e.nWrong&&truthOn()?tx(`${e.nWrong} wrong`,`${e.nWrong} 类看错了`):tx('every category','看全部五类'))),tip:statusOf().tip}];
}
const twoLine=dt=>{const vs=x=>zfix('比'+x), m1=dt.match(/^▲ 是(.+)的 (\S+) 倍$/), m2=dt.match(/^([▲▼] \S+)（比(.+)）$/);
  return m1?`▲ ${m1[2]} 倍\n${vs(m1[1])}`:m2?`${m2[1]}\n${vs(m2[2])}`:dt;};
/* the same in English: the change, then what it is compared with (“▲ +10%” over “vs last”); where that second line is too long for the
   card, “vs” ends the first line instead (“▼ −90% vs” over “1,000 items”), so no word is left alone on a line */
const twoLineEn=dt=>{const r=dt.match(/^(▲ \S+ times) (.+)$/), p=dt.match(/^([▲▼] \S+(?: pts?)?) vs (.+)$/), h=r?r[1]:p?p[1]:'', o=r?(r[2]==='the last value'?'last':r[2]):p?p[2]:'';
  return h?[`${h}\nvs ${o}`,`${h} vs\n${o}`]:[dt];};
function drawMetrics(){
  const host=$('#ecap-metrics',app), defs=metricsDef(), sg=defs.map(m=>m.k).join()+(onlineView()?'o':'');
  if(host._sig!==sg){host._sig=sg; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  /* part 3 compares every size with 1,000 items each, as its result box does (P8), not with the size picked before */
  const base=isP('size')?(S.n.code!==1000?Object.fromEntries(metricsDef(PLAN0()).map(m=>[m.k,m.v])):{}):null;
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    /* a two-line verdict breaks between its words (“还不能 / 下结论”), not before its last character (old item 4) */
    const ve=el.querySelector('.v'), vh=m.k==='dec'?esc(m.v).replace(/^(还不能|总分)(?=.)/,'$1<wbr>'):esc(m.v); if(ve._h!==vh){ve.innerHTML=vh; ve._h=vh;}
    let dt=''; const prev=base?base[m.k]:shownPrev[m.k];
    if(userAct&&prev!=null&&prev!==m.v&&m.k!=='dec'&&m.k!=='guard'){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,base?{base:{en:innerWidth<=760?'1,000 items':'the 1,000-item plan',zh:innerWidth<=760?'1,000 题时':'每类 1,000 题时'}}:{})||'';
      if(A.lang()==='zh') dt=zfix(dt);}
    clearTimeout(d._t); clearTimeout(d._t2);
    /* N5: in Chinese a change too long for the card's line goes on two lines, the change first, then what it is compared with
       (“▼ −90%” over “比每类 1,000 题时”), so no one or two characters end up alone on a line */
    /* once faded, the change keeps its line, hidden, until the reader acts again: clearing it 3 s later would shrink the card, and on a
       phone the page would jump by a line under the reader */
    const hide=()=>{d._t2=setTimeout(()=>{d.style.visibility='hidden';},400);};
    if(dt){d.style.visibility=''; d.textContent=dt; d._at=mode+step;
      if(d.offsetHeight>22){if(A.lang()==='zh'){const t2=twoLine(dt); if(t2!==dt) d.textContent=t2;} else for(const t2 of twoLineEn(dt)){d.textContent=t2; if(d.offsetHeight<=32) break;}}
      d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); hide();},2600);}
    else if(userAct||!d.textContent||d._at!==mode+step){d.classList.remove('on'); d.textContent=''; d.style.visibility='';}
    else {d.classList.remove('on'); hide();}
    shownPrev[m.k]=m.v;});
}
/* Show the math: blocks a reader can redo from the numbers shown */
const EQN=rows=>rows.some(([l])=>/\\text\{[^}]*?([a-z]{4,}|[\u4e00-\u9fff]{2,})/i.test(l||''));
const EQ=rows=>`<div class="ecap-eqs${EQN(rows)?' ecap-eqs-nm':''}">${rows.map(([l,r,rel])=>`<span class="ecap-eq-l">${l?MI(l):''}</span><span class="ecap-eq-e">${MI(rel||'=')}</span><span class="ecap-eq-r">${MI(r)}</span>`).join('')}</div>`;
const MB=(lead,body)=>`<div class="ecap-mb"><p>${lead}</p>${body}</div>`, inLast=(h,p)=>h.replace(/<\/div>$/,p+'</div>');
const tn=x=>fint(x);   // 1,000 typesets as one number, with no space after its comma (P30)
const f3=x=>(x<0?MINUS:'')+fixd(Math.abs(x),3), f4=x=>(x<0?MINUS:'')+fixd(Math.abs(x),4);
const tm=x=>f3(x).replace(MINUS,'-');
/* a negative number in a substitution sits in parentheses, so “− (−0.040)” never shows two minus signs in a row (O6); its minus leads a
   group of its own, {…}, so it is set as a sign, without the spacing of a subtraction (P30) */
const tv=x=>x<0?`({-${fixd(-x,3)}})`:fixd(x,3), tv5=x=>x<0?`({-${fx(-x)}})`:fx(x);
/* the cards and the chart speak in points; Show the math in decimals: say once, at the top, how one becomes the other (P30) */
const DECN=x=>tx(`Differences are written as decimals here: ${fixd(Math.abs(x),3)} is ${fA(x)} points.`,`这里的差写成小数：${fixd(Math.abs(x),3)} 就是 ${fA(x)} 分。`);
const fx=x=>{let t=fixd(x,6).replace(/0+$/,''); if(t.endsWith('.')) t+='0'; return t;};   // to 6 decimals, no trailing zeros
function mathHTML(){
  if(chHide()) return `<p>${tx('Submit your answer to see the working for your plan.','提交答案后，这里显示你这个计划的算式。')}</p>`;
  if(onlineView()){const a=abRows(abDays), r=a.write;
    return MB(tx(`<b>The writing guardrail.</b> Rates are written as decimals here: ${fixd(r.D,3)} is ${fixd(r.D*100,1)} points. Two separate groups, so the variances add: ${MI('p_0')} is the current model’s regenerate rate, ${MI('p_1')} the candidate’s, ${MI('n')} the writing conversations in each group.`,`<b>写作护栏。</b>这里的比例写成小数：${fixd(r.D,3)} 就是 ${fixd(r.D*100,1)} 个百分点。两组是不同的用户，方差相加：${MI('p_0')} 是当前模型的重新生成率，${MI('p_1')} 是候选的，${MI('n')} 是每组的写作对话数。`),
      EQ([[tx('\\text{difference}','\\text{差}'),'p_1 - p_0'],['',`${fixd(r.p1,3)} - ${fixd(r.p0,3)}`],['',tm(r.D)],
          [tx('\\text{half-width}','\\text{半宽}'),'1.96 \\sqrt{\\frac{p_0 (1 - p_0) + p_1 (1 - p_1)}{n}}'],['',`1.96 \\sqrt{\\frac{${fixd(r.p0,3)} \\times ${fixd(1-r.p0,3)} + ${fixd(r.p1,3)} \\times ${fixd(1-r.p1,3)}}{${tn(r.n)}}}`],['',f4(r.h)]])+
      `<p>${tx(`Interval: ${f3(r.lo)} to ${f3(r.hi)}; ${r.alarm?'wholly above 0, so the guardrail sounds the alarm.':'it crosses 0, so no alarm yet.'}`,`区间：${f3(r.lo)} 到 ${f3(r.hi)}；${r.alarm?'整个在 0 的右边，护栏报警。':'跨过了 0，还不报警。'}`)}</p>`);}
  const e=E(), w=e.rows.find(r=>r.id==='write'), c=CAT.write;
  let h='';
  if(totalOnly()){const t=e.total;   // no category's own difference here: part 1 asks the reader to predict it (R3.8)
    h+=MB(tx(`<b>The total.</b> ${DECN(t.D)} Each category’s difference Δ weighted by its share of traffic w. The categories’ own differences are not listed until you split them by category.`,`<b>总分。</b>${DECN(t.D)}每一类的差 Δ 乘它的流量比例 w 再相加。每一类自己的差，要分开看以后才列出来。`),
      EQ([[tx('\\text{total}','\\text{总分}'),'\\sum_c w_c \\Delta_c'],['',f3(t.D)]]))+
    MB(tx(`<b>Its half-width.</b> The categories’ standard errors SE combine with the same weights into the total’s standard error, here ${f4(t.se)}.`,`<b>它的半宽。</b>各类的标准误差 SE 按同样的权重合成总分的标准误差，这里是 ${f4(t.se)}。`),
      EQ([[tx('\\text{half-width}','\\text{半宽}'),'1.96 \\sqrt{\\sum_c w_c^2\\, \\text{SE}_c^2}'],['',`1.96 \\times ${f4(t.se)}`],['',f4(t.h)]]));
    return h;}
  const qs=fixd(c.q,2);
  if(w.meth!=='judge'){const win=(c.q+w.D)/2, lose=(c.q-w.D)/2;
    h+=MB(tx(`<b>Chinese writing, ${planLine(w,S)}.</b> ${DECN(w.D)} The difference is the share of items the candidate wins minus the share it loses; q is the share of items with a winner, n the items.`,`<b>中文写作，${planLine(w,S)}。</b>${DECN(w.D)}差等于候选赢的题所占比例减去输的题所占比例；q 是有输赢的题所占比例，n 是题数。`),
      EQ([[tx('\\text{difference}\\;\\Delta','\\text{差}\\;\\Delta'),`${fixd(win,3)} - ${fixd(lose,3)}`],['',f3(w.D)],
          [tx('\\text{half-width}\\;h','\\text{半宽}\\;h'),'1.96 \\sqrt{\\frac{q - \\Delta^2}{n}}'],['',`1.96 \\sqrt{\\frac{${qs} - ${fx(w.D*w.D)}}{${tn(w.n)}}}`],['',f4(w.h)]]));
  } else if(!w.mm){const Dj=w.Dj;   // the judge's difference is the true one plus its bias, not the other way round
    h+=MB(tx(`<b>Chinese writing, ${planLine(w,S)}.</b> ${DECN(w.D)} The judge’s difference ${MI('\\Delta_{\\text{j}}')} is the true difference ${MI('\\Delta')} plus the judge’s bias b, here ${f3(w.b)}.`,`<b>中文写作，${planLine(w,S)}。</b>${DECN(w.D)}裁判量出的差 ${MI('\\Delta_{\\text{j}}')} 等于真实的差 ${MI('\\Delta')} 加上裁判的偏差 b，这里 b 是 ${f3(w.b)}。`),
      EQ([[tx('\\text{judge}\\;\\Delta_{\\text{j}}','\\text{裁判}\\;\\Delta_{\\text{j}}'),'\\Delta + b'],['',`${tv(c.d)} + ${fixd(w.b,3)}`],['',f3(Dj)],
          [tx('\\text{half-width}\\;h','\\text{半宽}\\;h'),'1.96 \\sqrt{\\frac{q - \\Delta_{\\text{j}}^2}{n}}'],['',`1.96 \\sqrt{\\frac{${qs} - ${fx(Dj*Dj)}}{${tn(w.n)}}}`],['',f4(w.h)]]));
  } else if(!w.nu){
    h+=MB(tx(`<b>Chinese writing, ${planLine(w,S)}.</b> ${DECN(w.D)} People graded all ${fint(w.n)} items, so the judge adds nothing: the difference is the people’s, ${f3(c.d)}, and the half-width is the item-by-item one with m = ${fint(w.n)}.`,`<b>中文写作，${planLine(w,S)}。</b>${DECN(w.D)}${fint(w.n)} 题全都请人判了，裁判没有起作用：差就是人量出的 ${f3(c.d)}，半宽按逐题对比的公式算，m = ${fint(w.n)}。`),
      EQ([[tx('\\text{half-width}\\;h','\\text{半宽}\\;h'),'1.96 \\sqrt{\\frac{q - \\Delta^2}{m}}'],['',`1.96 \\sqrt{\\frac{${qs} - ${fx(c.d*c.d)}}{${tn(w.n)}}}`],['',f4(w.h)]]));
  } else {
    h+=MB(tx(`<b>Chinese writing, ${planLine(w,S)}.</b> ${DECN(w.D)} The judge measures ${f3(w.Dj)} on the items people did not check and on the checked ones; people measure ${f3(c.d)} on the checked ones. The gap between them is the judge’s bias, and it is taken out.`,`<b>中文写作，${planLine(w,S)}。</b>${DECN(w.D)}裁判在人没有判的题和校准题上量出的差都是 ${f3(w.Dj)}；人在校准题上量出的差是 ${f3(c.d)}。两者之差就是裁判的偏差，把它减掉。`),
      EQ([[tx('\\text{corrected}\\;\\hat{\\Delta}','\\text{修正后的差}\\;\\hat{\\Delta}'),tx('\\Delta_{\\text{judge, rest}} - ( \\Delta_{\\text{judge, chk}} - \\Delta_{\\text{people, chk}} )','\\Delta_{\\text{裁判，其余}} - ( \\Delta_{\\text{裁判，校准}} - \\Delta_{\\text{人，校准}} )')],['',`${tv(w.Dj)} - ( ${tv(w.Dj)} - ${tv(c.d)} )`],['',f3(w.D)]]))+
    MB(tx(`<b>Its half-width</b> combines two errors. The first comes from the n − m = ${fint(w.nu)} items only the judge graded; Δj is the judge’s difference. The second comes from the m = ${fint(w.mm)} items people checked: b is the judge’s bias on them (here ${f3(w.b)}), and e is the share of them where judge and person disagree (here ${pct(w.e)}). This assumes that when they disagree they differ by one level: one calls a tie, the other a winner.`,`<b>它的半宽</b>由两部分误差合起来。第一部分来自只有裁判判的 n − m = ${fint(w.nu)} 题，Δj 是裁判量出的差。第二部分来自人工校准的 m = ${fint(w.mm)} 题：b 是裁判在这批题上的偏差（这里 ${f3(w.b)}），e 是裁判和人意见不一致的题所占的比例（这里 ${pct(w.e)}）。这里假设两边不一致时只差一级：一个判平手，另一个判出了输赢。`),
      EQ([[tx('\\text{half-width}\\;h','\\text{半宽}\\;h'),'1.96 \\sqrt{\\frac{q - \\Delta_{\\text{j}}^2}{n - m} + \\frac{e - b^2}{m}}'],['',`1.96 \\sqrt{\\frac{${qs} - ${fx(w.Dj*w.Dj)}}{${tn(w.nu)}} + \\frac{${fixd(w.e,2)} - ${fx(w.b*w.b)}}{${tn(w.mm)}}}`],['',f4(w.h)]]));
  }
  h=inLast(h,`<p>${tx(`Interval: ${f3(w.lo)} to ${f3(w.hi)}, so the verdict is “${L(VN[w.v])}”.`,`区间：${f3(w.lo)} 到 ${f3(w.hi)}，结论是“${L(VN[w.v])}”。`)}</p>`);
  if(isP('size')){const se=Math.round(w.se*1e5)/1e5, z=Math.abs(w.D)/se-Z, pc=pct(Phi(z));
    h+=MB(tx(`<b>Would a real evaluation catch the loss?</b> There the measured difference wanders around the true one (in this lab it does not), by about its standard error SE, here ${fx(se)}. The interval lies wholly below 0 only when the measured difference is more than 1.96 SE below 0, which happens with chance ${MI('\\Phi(z)')}; ${MI('\\Phi')} is the normal distribution’s cumulative probability.`,
                `<b>真实评测里能不能抓到退步。</b>量出来的差会围着真实的差晃动（这个 Lab 里不晃），晃动的大小约是标准误差 SE，这里是 ${fx(se)}。量出来的差要比 0 低 1.96 个 SE 以上，区间才整个落在 0 的左边，这件事发生的机会是 ${MI('\\Phi(z)')}，${MI('\\Phi')} 是正态分布的累积概率。`),
      EQ([['z',tx('\\frac{|\\Delta|}{\\text{SE}} - 1.96','\\frac{|\\Delta|}{\\text{SE}} - 1.96')],['',`\\frac{${fixd(Math.abs(w.D),3)}}{${fx(se)}} - 1.96`],['',(z<0?'-':'')+fixd(Math.abs(z),3)],
          [tx('\\text{chance}','\\text{机会}'),'\\Phi(z)'],['',`\\Phi(${z<0?`{-${fixd(-z,3)}}`:fixd(z,3)})`],['',fixd(Phi(z),2),'\\approx']])+
      `<p>${Phi(z)>=0.9?tx(`With ${fint(w.n)} items per category, such a plan catches writing’s loss about ${pc} of the time: the items are enough.`,`每类 ${fint(w.n)} 题时，这样的计划约有 ${pc} 的机会抓到写作的退步，题数够了。`)
          :w.v!=='worse'?tx(`With ${fint(w.n)} items per category, such a plan catches writing’s loss only about ${pc} of the time; in this lab it cannot tell yet either.`,`每类 ${fint(w.n)} 题时，这样的计划只有约 ${pc} 的机会抓到写作的退步；在这个 Lab 里也还看不出。`)
          :tx(`With ${fint(w.n)} items per category, such a plan catches writing’s loss only about ${pc} of the time, so a real plan keeps extra items in hand.`,`每类 ${fint(w.n)} 题时，这样的计划只有约 ${pc} 的机会抓到写作的退步，所以真实的计划要多留一些题。`)}</p>`);}   // N1: what the chance means at each size
  const nj=e.rows.filter(r=>r.meth==='judge'), calls=nj.reduce((s,r)=>s+r.cost.calls,0);
  h+=MB(tx('<b>Cost</b>, over all five categories: judge calls at $0.01 each and people’s labels at $2 each.','<b>花费</b>，五类合计：裁判每次 $0.01，人工每题 $2。'),
    EQ([[tx('\\text{cost}','\\text{花费}'),tx('\\text{calls} \\times 0.01 + \\text{labels} \\times 2','\\text{裁判次数} \\times 0.01 + \\text{人工题数} \\times 2')],['',`${tn(calls)} \\times 0.01 + ${tn(e.H)} \\times 2`],['',`\\text{$}${tn(Math.round(e.cost))}`]]));
  return h;
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawFP(); drawMetrics(); lateAll();
  const mb=$('#ecap-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); // the kit drops the settings from the address outside Scenarios, so a reload in another mode is not taken for a shared link
  A.dock.refresh(); announce();
}
/* N10: while writing's row builds up, the badge and the verdict card show their new value when the row's result appears (the total row
   does the same in drawFP): one class each, with the build-up's own delay; any later render takes it off, so nothing replays */
const LATE={bias:'ecap-lb',cal:'ecap-lc'};
function lateIn(el,k){if(!el) return; el.classList.remove('ecap-lb','ecap-lc'); if(k&&LATE[k]){void el.offsetWidth; el.classList.add(LATE[k]);}}
function lateAll(){const dc=$('#ecap-metrics [data-k="dec"]',app);
  lateIn($('#ecap-status',app),buildNow); if(dc){lateIn(dc.querySelector('.v'),buildNow); lateIn(dc.querySelector('.sub'),buildNow);}}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you set the plan in the task card.','挑战中只在任务卡里定计划。');
  if(mode!=='guided') return '';
  return tx('In the guided parts the plan is set by each part. You can change every setting in Scenarios.','分步引导里，计划由每一部分设定。在应用场景里可以改所有设置。');
}
function syncControls(){
  CATS.forEach(c=>{$$(`#ecap-controls .seg-ctl[data-key="m-${c.id}"]`,app).forEach(g=>A.ui.setPressed(g,S.m[c.id])); $$(`#ecap-controls .seg-ctl[data-key="n-${c.id}"]`,app).forEach(g=>A.ui.setPressed(g,String(S.n[c.id])));});
  $$('.seg-ctl[data-key="ctl"]',app).forEach(g=>A.ui.setPressed(g,String(S.ctl)));
  $$('.seg-ctl[data-key="cal"]',app).forEach(g=>A.ui.setPressed(g,String(S.cal)));
  const w=$('#ecap-ctl-wrap',app); if(w) w.hidden=mode!=='sandbox';
  const ln=$('#ecap-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.txt);},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setPlan(P){S=clone(P); render();}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setSplit(on){split=!!on; render();}
function setSize(n){sizeTried.add(n); sizeClicked=true; CATS.forEach(c=>{S.n[c.id]=n;}); render();}
function setP4(k){p4Sel=k; p4Tried.add(k); S=p4Plan(k); render();}
function setCal(m){calTried.add(m); S.cal=m; render();}
function setAb(d){abDays=d; abTried.add(d); render();}
function phoneHTML(){
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='mission'&&!pred) return `<button type="button" class="btn ghost sm" data-act="to-q" data-tipx="${esc(tx('Scrolls down to the three answers in the card.','向下滚到卡片里的三个答案。'))}">${tx('Answer below ↓','在下面作答 ↓')}</button>`;
  if(id==='split') return splitCtl('ecap-ph-split');
  if(id==='size') return sizeCtl('ecap-ph-size');
  if(id==='judge'){const more=p4Tried.size>1;   // after a way is picked, the stage is in view and the verdict is in the card below it (P22)
    return `<button type="button" class="btn ghost sm" data-act="to-lv" data-tipx="${esc(more?tx('Scrolls down to the verdict and the three ways in the card.','向下滚到卡片里的结论和三种判法。'):tx('Scrolls down to the three ways in the card.','向下滚到卡片里的三种判法。'))}">${more?tx('Verdict and the other ways below ↓','结论和另外两种判法在下面 ↓'):tx('Pick below ↓','在下面选 ↓')}</button>`;}
  if(id==='calib'){const more=calTried.size>1;   // as in part 4: the counts are in the card, a pick brings the stage up (N3)
    return `<button type="button" class="btn ghost sm" data-act="to-cal" data-tipx="${esc(more?tx('Scrolls down to the verdict and the counts in the card.','向下滚到卡片里的结论和题数。'):tx('Scrolls down to the counts in the card.','向下滚到卡片里的题数。'))}">${more?tx('Verdict and the other counts below ↓','结论和其他题数在下面 ↓'):tx('Pick below ↓','在下面选 ↓')}</button>`;}
  if(id==='online') return abCtl('ecap-ph-ab');
  return '';
}
function renderPhone(){const el=$('#ecap-phone',app); if(!el) return; const h=phoneHTML(); if(el._h!==h){el.innerHTML=h; el._h=h;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S=clone(s.enter());
  renderRail();
}
function revealStep(){A.ui.reveal($('#ecap-rail .step.current',app)||$('#ecap-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* on a phone the ways of part 4 and the counts of part 5 sit in the card under the stage: after a pick, the whole stage comes up under
   the top bar, so the writing row's build-up plays in view; the button at the stage's top then leads back down to the verdict (P22,
   N3). The build-up, with everything that waits for it, stays paused at its start while the page scrolls, and starts once the stage
   stands still (N11): on the scroll's end, or after the time a smooth scroll of that length takes */
let holdT=0, holdOn=false;
function holdEnd(){clearTimeout(holdT); holdT=0; if(holdOn){document.removeEventListener('scrollend',holdEnd); holdOn=false;}
  const st=app&&$('#ecap-stage',app); if(st) st.classList.remove('ecap-hold');}
function stageUp(){const st=$('#ecap-stage',app); if(!st) return;
  const bar=document.querySelector('.topbar'), dy=st.getBoundingClientRect().top-(bar?A.ui.barB():56)-6;
  if(Math.abs(dy)<=2) return;
  if(A.reduceMotion()){window.scrollBy({top:dy,behavior:'auto'}); return;}
  holdEnd(); st.classList.add('ecap-hold'); holdOn=true; document.addEventListener('scrollend',holdEnd);
  holdT=setTimeout(holdEnd,Math.min(900,350+Math.abs(dy)*0.4));
  window.scrollBy({top:dy,behavior:'smooth'});}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#ecap-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ecap-rail .step.current',app)||$('#ecap-rail .railcard',app)||$('#ecap-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#ecap-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ecap-insight',app)); else revealBox($('#ecap-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S=clone(saved); saved=null;}
  if(m==='challenge'){saved=clone(S); chSubmitted=false; chReveal=false; S=CH_START();}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S=PLAN0();}
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const MC={prog:'p',judge:'j',human:'h'}, CM={p:'prog',j:'judge',h:'human'};
const token=()=>CATS.map(c=>MC[S.m[c.id]]+S.n[c.id]).join('_')+`_c${S.ctl}_m${S.cal}`;
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==7) return null;
  const P={m:{},n:{},ctl:0,cal:0};
  for(let i=0;i<5;i++){const m=/^([pjh])(\d+)$/.exec(p[i]); if(!m) return null; const c=CATS[i], meth=CM[m[1]], n=+m[2];
    if(!SIZES.includes(n)||(meth==='prog'&&!c.prog)) return null; P.m[c.id]=meth; P.n[c.id]=n;}
  const c=/^c([01])$/.exec(p[5]), m=/^m(\d+)$/.exec(p[6]); if(!c||!m||!CALS.includes(+m[1])) return null;
  P.ctl=+c[1]; P.cal=+m[1]; return P;
}

/* =====================================================================
   interview drill: a category graded by people; every line redoes from the numbers it shows
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* n divides 10^5, so Δ and q are exact to 5 decimals; h is rounded to 4, and the interval is Δ ± that h, so every line redoes */
function makeDrill(seed){const r=mulberry(seed*104729+7), n=[400,500,800,1000][Math.floor(r()*4)], win=Math.round(n*(0.08+r()*0.08)), lose=Math.round(n*(0.08+r()*0.08));
  const d=(win-lose)/n, q=(win+lose)/n, h=Math.round(Z*Math.sqrt((q-+fx(d*d))/n)*1e4)/1e4, lo=Math.round((d-h)*1e5)/1e5, hi=Math.round((d+h)*1e5)/1e5;
  return {seed,n,win,lose,d,q,h,lo,hi,v:verdict(d,h,true)};}
const fe=x=>(x<0?MINUS:'')+fx(Math.abs(x));
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#ecap-drill-q',app), a=$('#ecap-drill-a',app);
  qd.innerHTML=tx(`People compare the candidate and the current model on <b>${fint(d.n)}</b> items of one category. The candidate’s answer is better on <b>${d.win}</b>, the current model’s on <b>${d.lose}</b>, the rest tie. Accepting a loss of at most 3 points, what is the verdict?`,
                  `人工在一类题的 <b>${fint(d.n)}</b> 道题上比较候选和当前模型：候选答得更好的有 <b>${d.win}</b> 题，当前模型更好的有 <b>${d.lose}</b> 题，其余平手。最多接受变差 3 分，这一类的结论是什么？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>${L(VN[d.v])}: the difference is ${fD(d.d)} points, interval ${fB(d.lo)} to ${fB(d.hi)}.</b></p>`,`<p><b>${L(VN[d.v])}：差是 ${fD(d.d)} 分，区间 ${fB(d.lo)} 到 ${fB(d.hi)} 分。</b></p>`)+
    EQ([[tx('\\Delta','\\Delta'),`\\frac{${d.win} - ${d.lose}}{${tn(d.n)}}`],['',fe(d.d)],['q',`\\frac{${d.win} + ${d.lose}}{${tn(d.n)}}`],['',fx(d.q)],
        ['h',`1.96 \\sqrt{\\frac{${fx(d.q)} - ${fx(d.d*d.d)}}{${tn(d.n)}}}`],['',f4(d.h)],
        [tx('\\text{interval}','\\text{区间}'),`${tv5(d.d)} \\pm ${f4(d.h)}`],['',`${fe(d.lo)}\\ \\text{${tx('to','到')}}\\ ${fe(d.hi)}`]])+
    tx('<p>Above 0 at the lower end: better. Below 0 at the upper end: worse. Lower end above −3: not worse by more than 3 points. Otherwise, more items.</p>','<p>下端高于 0：变好了。上端低于 0：变差了。下端高于 −3 分：没有变差超过 3 分。都不是，就要更多的题。</p>');
}

/* =====================================================================
   term tooltips and jumps
   ===================================================================== */
const CTLN={ctl:{en:'How the judge runs',zh:'裁判怎么判'},cal:{en:'People check',zh:'人工校准'}};
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!CTLN[k]) return '';
  if(mode==='guided') return tx('Fixed in this part; you can change it in Scenarios.','这一部分固定不变；在应用场景里可以改。');
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#ecap-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#ecap-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(!CTLN[k]) return;
  if(mode==='guided'){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=$(mode==='challenge'?`#ecap-ic-${k}`:`#ecap-ctl-${k}`,app); if(el) A.ui.jumpToEl(el);
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>setPlan(PLAN0())); return;}
  if(a==='submit'){chSubmitted=true; if(chPass(E())) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='to-q'){const q=$('#ecap-rail .step.current .choices',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-lv'){const q=$('#ecap-ic-p4',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='to-cal'){const q=$('#ecap-ic-cal',app); if(q) A.ui.jumpToEl(q); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ecap-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
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
const bySettings=(el,f)=>el&&el.closest('#ecap-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
const inRail=b=>!!b.closest('#ecap-rail, #ecap-phone');
function bind(el){
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ecap-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; byLearner(()=>s.showMe()); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ecap-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null&&isP('judge')){const f=()=>byLearner(()=>setP4(d.lever)); if(innerWidth<=760){f(); stageUp();} else {f(); revealSoon();} return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key, v=d.v, guided=inRail(b)&&mode==='guided';
      const run=f=>{if(b.closest('#ecap-controls')) bySettings(b,f); else {byLearner(f); if(guided) revealSoon();}};
      if(k==='split') run(()=>setSplit(v==='1'));
      else if(k==='gsize') run(()=>setSize(+v));
      else if(k==='gcal'){if(guided&&innerWidth<=760){byLearner(()=>setCal(+v)); stageUp();} else run(()=>setCal(+v));}   // N3: as part 4 does
      else if(k==='abd') run(()=>setAb(+v));
      else if(k==='ctl') run(()=>{S.ctl=+v; render();});
      else if(k==='cal') run(()=>{S.cal=+v; render();});
      else if(k.startsWith('m-')){const id=k.slice(2); run(()=>{S.m[id]=v; render();});}
      else if(k.startsWith('n-')){const id=k.slice(2); run(()=>{S.n[id]=+v; render();});}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setPlan(p.p())); requestAnimationFrame(()=>{if(app) revealBox($('#ecap-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ecap-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
/* the lab state contract (round 3 #6): what the learner did here, saved by the kit and put back once per page load before mount */
let restored=false;
const planOK=o=>{if(!o||typeof o!=='object'||!o.m||!o.n||typeof o.m!=='object'||typeof o.n!=='object') return null;
  return parse(CATS.map(c=>(MC[o.m[c.id]]||'x')+o.n[c.id]).join('_')+`_c${o.ctl}_m${o.cal}`);};
const setOf=(a,ok)=>new Set(Array.isArray(a)?a.filter(ok).slice(0,20):[]);
function snap(){return {v:1,mode,step,S:clone(S),pred,observed:{...observed},saved:saved?clone(saved):null,sbTouched,split,sizeTried:[...sizeTried],sizeClicked,p4Tried:[...p4Tried],p4Sel,
  calTried:[...calTried],abDays,abTried:[...abTried],chSubmitted,chReveal,drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPass(E())};}
function restore(o){if(!o||o.v!==1) return; const P=planOK(o.S); if(!P) return;
  mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; S=P;
  step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
  observed={}; if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
  const sv=planOK(o.saved); saved=mode==='challenge'?(sv||PLAN0()):null;
  sbTouched=o.sbTouched===true; split=o.split===true;
  sizeTried=setOf(o.sizeTried,v=>SIZES.includes(v)); sizeClicked=o.sizeClicked===true; p4Tried=setOf(o.p4Tried,v=>typeof v==='string'&&!!P4[v]); p4Sel=typeof o.p4Sel==='string'&&P4[o.p4Sel]?o.p4Sel:'human';
  calTried=setOf(o.calTried,v=>CALS.includes(v)); abDays=AB.days.includes(o.abDays)?o.abDays:1; abTried=setOf(o.abTried,v=>AB.days.includes(v));
  chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
  drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; restored=true;}
A.registerLab({
  id:ID, snap, restore,
  mount(el,tok){
    wantObs=false; root=el; rowSig={};
    const p=tok?parse(tok):null; if(p){S=clone(p); mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved=PLAN0();
    if(mode==='guided'&&!p&&step===0&&!restored) S=clone(STEPS[0].enter());   // a state just put back keeps its plan
    restored=false;
    el.innerHTML=viewHTML(); app=$('#ecap-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ecap-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    fillRail();
    render();
    glueOn();
  },
  leave(){clearTimeout(annT); stageStop(); holdEnd(); glueOff(); root=null; app=null;},
  onLang(){if(root){const d=$('#ecap-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#ecap-math',app).open=true;}},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#ecap-fp',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#ecap-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf();
      status.className='status '+s.cls; status.textContent=s.short;
      if(onlineView()){const a=abRows(abDays), r=a.write;
        vis.innerHTML=`<div class="ecap-dk"><i class="z" style="left:${pos(0,AXO)}%"></i><i class="b${r.alarm?' red':''}" style="left:${pos(r.lo*100,AXO).toFixed(1)}%;width:${(pos(r.hi*100,AXO)-pos(r.lo*100,AXO)).toFixed(1)}%"></i></div>`;
        /* the unit stays whole on a phone's narrow bar (“个百分 / 点”) */
        read.innerHTML=tx(esc(`Writing conversations: regenerate rate ${fD(r.D)} points (${fB(r.lo)} to ${fB(r.hi)})`),`写作类对话：重新生成率 <span class="ecap-nw">${esc(fD(r.D))} 个百分点</span>（${esc(fB(r.lo))} 到 ${esc(fB(r.hi))}）`); return;}
      const e=E();
      if(chHide()){vis.innerHTML=''; read.innerHTML=esc(tx(`Your plan costs ${usd(e.cost)} and takes ${e.days} day${e.days>1?'s':''}; submit to see the verdicts`,`你的计划花 ${usd(e.cost)}，用 ${e.days} 天；提交后显示结论`)); return;}
      const r=totalOnly()?e.total:e.rows.find(x=>x.id==='write'), nm=totalOnly()?tx('Total','总分'):L(CN.write), red=!totalOnly()&&r.v==='worse';
      vis.innerHTML=`<div class="ecap-dk"><i class="z" style="left:${pos(0,AX)}%"></i><i class="b${red?' red':''}" style="left:${pos(r.lo*100,AX).toFixed(1)}%;width:${(pos(r.hi*100,AX)-pos(r.lo*100,AX)).toFixed(1)}%"></i></div>`;
      read.innerHTML=esc(tx(`${nm}: ${fD(r.D)} points (${fB(r.lo)} to ${fB(r.hi)}), ${L(VN[r.v]||VN.unk)}`,`${nm}：${fD(r.D)} 分（${fB(r.lo)} 到 ${fB(r.hi)}），${L(VN[r.v]||VN.unk)}`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__ecap=Object.freeze({
  evalPlan:(P,mg)=>deep(evalPlan(P,mg)), est:(id,m,n,ctl,cal)=>deep(est(CAT[id],m,n,ctl,cal)), abRows:d=>deep(abRows(d)), chSearch:()=>{const s=chSearch(); return deep({combos:s.combos,nPass:s.nPass,best:s.best});},
  chPass:P=>chPass(evalPlan(P)), drill:s=>deep(makeDrill(s)), fB:x=>fB(x), parse, exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a),transfer:e.transfer.map(q=>({q:q.q,o:q.o,why:q.why}))});},
  consts:Object.freeze({Z,MARGIN,BUDGET,DEADLINE,JCALL,HUM,PERDAY,SIZES,CALS,CATS:deep(CATS),AB:deep(AB),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PLAN0:PLAN0(),CH_START:CH_START(),CH_ANSWER:deep(CH_ANSWER),PRESETS:PRESETS.map(p=>({id:p.id,p:p.p()})),CODE_OUT}),
  state:()=>deep({S,mode,step,pred,observed,split,sizeTried:[...sizeTried],p4Sel,p4Tried:[...p4Tried],calTried:[...calTried],abDays,abTried:[...abTried],chSubmitted,chReveal,stageBusy:stageT!==0})
});
})();
