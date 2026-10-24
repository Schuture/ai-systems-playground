/* RAG failure lab: why does your RAG system answer wrong? A help-center bot of a fictional smart-home company (示例, the Lumo Home /
   小鹿家居 world of the RAG labs; documents and runs written for this lab, not recorded traces) is traced stage by stage: question,
   retrieval, the context budget, the answer, the judging. Seven illustrative runs show one failure kind each; a stated set of 200 graded
   questions gives the failure breakdown and the per-stage checks; five fixes repair stated shares of their failure kinds (the lab's
   assumptions). Everything is deterministic; tests/rag-failure_numbers.py recomputes it in Python.
   Names (batch 10 brief S6): 召回率 here is per question: are all the passages the answer needs among the chunks retrieved (检索召回率)
   or among the chunks sent (上下文召回率). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='rag-failure';

/* ---------- engine (ENGINE-START) ---------- */
/* the help center's chunks (example): id → [section en, zh], [text en, zh], tag (old: a replaced edition still in the index) */
const CH={
 W1:[['Warranty › coverage','保修 › 保修范围'],['Hardware faults are covered for 2 years from the date of purchase.','硬件故障自购买之日起保修 2 年。']],
 W0:[['Warranty › coverage (2024 edition)','保修 › 保修范围（2024 年版）'],['Hardware faults are covered for 1 year from the date of purchase.','硬件故障自购买之日起保修 1 年。'],'old'],
 W2:[['Warranty › exclusions','保修 › 不保修的情况'],['Accidental damage, such as dropping the device or getting it wet, is not covered.','意外损坏不在保修范围内，比如设备摔落或进水。']],
 W3:[['Warranty › making a claim','保修 › 申请保修'],['Start a claim in the app under Support, with a photo of the receipt.','在 App 的“支持”里发起保修申请，附上购物凭证的照片。']],
 E1:[['Error codes › E-104','错误码 › E-104'],['E-104: the lock’s motor is jammed. This is a hardware fault; contact support for a repair.','E-104：门锁电机卡住了。这是硬件故障，请联系客服维修。']],
 E2:[['Error codes › E-201','错误码 › E-201'],['E-201: a wheel of the vacuum is stuck. Lift the robot and clear the wheel.','E-201：扫地机器人的轮子卡住了。把机器人拿起来，清理轮子。']],
 L1:[['Lock › auto-lock','门锁 › 自动上锁'],['Auto-lock locks the door 30 seconds after it closes. In Settings you can choose 5 seconds to 5 minutes.','自动上锁默认在门关上 30 秒后锁门。在设置里可以改成 5 秒到 5 分钟。']],
 L2:[['Lock › batteries','门锁 › 电池'],['The lock runs on four AA batteries; a new set lasts about eight months.','门锁用四节 5 号电池，一套新电池大约能用八个月。']],
 L3:[['Lock › door-open alarm','门锁 › 门没关提醒'],['If the door stays open for more than 2 minutes, the lock beeps and the app sends an alert.','门开着超过 2 分钟，门锁会响，App 也会发提醒。']],
 L4:[['Lock › locking from the app','门锁 › 用 App 开锁'],['You can lock and unlock the door from anywhere in the app, under Lock.','在 App 的“门锁”里，可以随时随地开门、锁门。']],
 V1:[['Vacuum › charging','扫地机器人 › 充电'],['A full charge takes about four hours and cleans for 120 minutes.','充满一次大约要四个小时，能清扫 120 分钟。']],
 V2:[['Vacuum › cliff sensors','扫地机器人 › 防跌落传感器'],['Cliff sensors under the robot stop it at the edge of stairs. Wipe them once a month.','机器人底部的防跌落传感器会让它在楼梯边停下。每月擦一次。']],
 V3:[['Vacuum › filter','扫地机器人 › 滤网'],['Wash the filter with water once a month and let it dry for a day.','滤网每月用水冲洗一次，晾干一天再装回。']],
 V4:[['Vacuum › dust bin','扫地机器人 › 尘盒'],['Empty the dust bin after every run.','每次清扫完都要倒掉尘盒。']],
 A1:[['App › members','App › 成员'],['The owner can invite up to 8 family members in the app, under Members.','主人可以在 App 的“成员”里邀请最多 8 位家人。']],
 T1:[['Thermostat › eco mode','恒温器 › 节能模式'],['Eco mode lowers the temperature by 2 degrees while nobody is home.','节能模式会在家里没人时把温度调低 2 度。']]};
/* the seven runs (example): kind, question, the 6 chunks retrieved with their similarity, the passages the answer needs (gold) with the
   rank search gave them, the answer as claims (each with the chunk that supports it, or null), the reference answer */
const RUNS=[
 {id:'charge',kind:'ok',q:['How long does a full charge of the robot vacuum take?','扫地机器人充满一次电要多久？'],
  ret:[['V1',.86],['V2',.58],['V4',.55],['L2',.49],['V3',.47],['T1',.31]],gold:[['V1',1]],
  ans:[['About four hours; then it cleans for 120 minutes.','大约四个小时，充满后能清扫 120 分钟。','V1']],truth:['About four hours.','大约四个小时。']},
 {id:'stairs',kind:'miss',q:['My vacuum fell down the stairs. Will you fix it for free?','我的扫地机器人从楼梯上摔下去了，你们免费修吗？'],
  ret:[['V2',.71],['W1',.66],['W3',.61],['W0',.57],['V1',.44],['E2',.41]],gold:[['W2',11]],
  ans:[['Hardware faults are covered for 2 years from the date of purchase.','硬件故障自购买之日起保修 2 年。','W1'],['Start a warranty claim in the app under Support.','请在 App 的“支持”里申请保修。','W3']],
  truth:['No. A fall is accidental damage, which the warranty does not cover.','不免费。摔落属于意外损坏，不在保修范围内。']},
 {id:'autolock',kind:'cut',q:['The door locks itself too fast. Can I make it wait longer?','门自己锁得太快了，能让它多等一会儿吗？'],
  ret:[['L3',.68],['L4',.64],['L2',.52],['L1',.51],['A1',.40],['E1',.36]],gold:[['L1',4]],
  ans:[['You can lock and unlock the door from the app at any time.','可以在 App 里随时开门、锁门。','L4']],
  truth:['Yes. In Settings, set the auto-lock wait between 5 seconds and 5 minutes (the default is 30 seconds).','可以。在设置里把自动上锁的等待时间改成 5 秒到 5 分钟之间（默认 30 秒）。']},
 {id:'thermo',kind:'stale',q:['How long is the warranty on my thermostat?','我的恒温器保修多久？'],
  ret:[['W0',.83],['W1',.82],['W3',.63],['W2',.58],['T1',.47],['A1',.30]],gold:[['W1',2]],
  ans:[['Hardware faults are covered for 1 year.','硬件故障保修 1 年。','W0']],
  truth:['2 years. The 2024 page that says 1 year was replaced.','2 年。写着 1 年的是 2024 年的旧版，已经被替换了。']},
 {id:'filter',kind:'unsup',q:['How often should I wash the vacuum’s filter?','扫地机器人的滤网多久洗一次？'],
  ret:[['V3',.84],['V4',.62],['V2',.55],['V1',.50],['E2',.38],['L2',.29]],gold:[['V3',1]],
  ans:[['Wash the filter once a week.','滤网每周洗一次。',null]],truth:['Once a month, then let it dry for a day.','每月一次，洗完晾干一天。']},
 {id:'e104',kind:'multi',q:['My lock shows E-104 and it is 14 months old. Is the repair free?','门锁显示 E-104，买了 14 个月，维修免费吗？'],
  ret:[['E1',.81],['E2',.64],['L2',.52],['L3',.47],['L4',.45],['W3',.44]],gold:[['E1',1],['W1',9]],
  ans:[['E-104 means the motor is jammed; contact support for a repair.','E-104 是电机卡住了，请联系客服维修。','E1']],
  truth:['Yes. E-104 is a hardware fault, and hardware faults are covered for 2 years.','免费。E-104 是硬件故障，硬件故障保修 2 年。']},
 {id:'watch',kind:'noans',q:['Can I unlock the door with my smartwatch?','能用智能手表开门吗？'],
  ret:[['L4',.62],['L1',.48],['A1',.45],['L3',.41],['L2',.38],['E1',.30]],gold:[],
  ans:[['Yes: open the Lumo app on your watch and tap Unlock.','能：在手表上打开小鹿 App，点“开锁”。',null]],
  truth:['The help center does not say. The bot should say so and suggest contacting support.','帮助中心没有写。机器人应该说文档里没有，并建议联系客服。']}];
const NRET=6, BUDGET=3;   // the bot: vector search returns 6 chunks; the context budget holds the first 3
const KINDS=['miss','multi','cut','stale','unsup','noans'];   // in the order of the stage that fails: retrieval, budget, documents, model
/* which check flags a kind: [retrieval recall, context recall, grounded, correct]; null: not counted (no passage to find) */
const FLAG={ok:[0,0,0,0],miss:[1,1,0,1],cut:[0,1,0,1],stale:[0,0,0,1],unsup:[0,0,1,1],multi:[1,1,0,1],noans:[null,null,1,1]};
const CAUSE={ok:null,miss:'ret',cut:'ctx',stale:'ret',unsup:'ans',multi:'ret',noans:'ans'};
const CHK=['ret','ctx','grd','cor'], CAUSE_CHK={miss:'ret',cut:'ctx',stale:'cor',unsup:'grd',multi:'ret',noans:'grd'};
/* the 200 graded questions of each system (the lab's stated mix) */
const MIX={base:{ok:140,miss:6,cut:16,stale:12,unsup:12,multi:4,noans:10},ch:{ok:118,miss:10,cut:8,stale:28,unsup:6,multi:20,noans:10}};
const FIXES=['hyb','ctx','ver','cite','dec'];
/* share of each failure kind a fix repairs, and the wait it adds per question in ms (the lab's assumptions) */
const SHARE={hyb:{miss:.6,cut:.5,multi:.25},ctx:{cut:.75,multi:.25},ver:{stale:.9},cite:{unsup:.5,noans:.8},dec:{multi:.75,miss:.2}};
const WAIT={hyb:200,ctx:100,ver:0,cite:100,dec:600};
const rnd=x=>Math.floor(x+0.5+1e-9);
/* runs left in each kind after the fixes: a run stays wrong only if no chosen fix repairs it, n × Π(1 − share), rounded half up */
function compute(sys,fx){const mix=MIX[sys], left={}, fixed={}; let ok=mix.ok;
  KINDS.forEach(k=>{let p=1; fx.forEach(f=>{p*=1-(SHARE[f][k]||0);}); left[k]=rnd(mix[k]*p); fixed[k]=mix[k]-left[k]; ok+=fixed[k];});
  const N=Object.values(mix).reduce((a,b)=>a+b,0), ans=N-mix.noans;
  const nRet=ans-left.miss-left.multi, nCtx=nRet-left.cut, nGrd=N-left.unsup-left.noans;
  return {N,ans,ok,left,fixed,nRet,nCtx,nGrd,cor:ok/N,ret:nRet/ans,ctx:nCtx/ans,grd:nGrd/N,wrong:N-ok,wait:fx.reduce((a,f)=>a+WAIT[f],0)};}
/* one run's checks, read from its log */
function checks(r){const top=r.ret.map(x=>x[0]), sent=top.slice(0,BUDGET), g=r.gold.map(x=>x[0]);
  const ret=g.length?g.every(x=>top.includes(x)):null, ctx=g.length?g.every(x=>sent.includes(x)):null;
  const grd=r.ans.every(c=>c[2]&&sent.includes(c[2])), cor=r.kind==='ok';
  return {ret,ctx,grd,cor};}
const CH_GOAL=0.75, CH_WAIT=300;
function pairs(){const out=[]; for(let i=0;i<FIXES.length;i++) for(let j=i+1;j<FIXES.length;j++) out.push([FIXES[i],FIXES[j]]); return out;}
function chAll(){const out=pairs().map(p=>{const m=compute('ch',p); return {p,ok:m.ok,cor:m.cor,wait:m.wait,met:m.cor>=CH_GOAL-1e-9&&m.wait<=CH_WAIT};});
  const best=Math.max(...out.filter(x=>x.met).map(x=>x.ok)); out.forEach(x=>{x.best=x.met&&x.ok===best;}); return out;}
/* ENGINE-END */

/* ---------- words and helpers ---------- */
const pc=v=>Math.round(v*100+1e-9)+'%', pc1=v=>A.fmt.num(v*100,1)+'%';
const LG=()=>A.lang()==='zh'?'zh':'en', li=()=>LG()==='zh'?1:0;
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn rgf-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="rgf-math-line" data-tex="${esc(t)}">${A.ui.math(t)}</div>`;   // data-tex: the source, so the numbers test can recompute every line
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(!l) return esc(id); if(C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return A.shell.labLink?A.shell.labLink(id):esc(id);};
const SEE=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id], n=l?L(l.s||{en:id,zh:id}):id;   // no space between 见 and a Chinese name
  return (/^[\u4e00-\u9fff]/.test(n)?'见':'见 ')+LL(id);};
/* a fix in two lines (round-1 polish P13): what it does (the mechanism in its tooltip), then what it repairs and the wait it adds */
const FXLINES=(f,assumed)=>`<span class="rgf-ln"><span class="rgf-do" data-tipx="${esc(FXT[f]())}">${esc(FXS[f]())}</span></span><span class="rgf-ln rgf-fe">${esc(fxEffect(f,assumed))}</span>`;
const ord=n=>n+((n%100>=11&&n%100<=13)?'th':(['th','st','nd','rd'][n%10]||'th'));
const zhList=a=>a.length<2?a.join(''):a.slice(0,-1).join('、')+'和'+a[a.length-1];
const quote=s=>tx(`“${esc(s)}”`,`“${esc(s)}”`);
const KN={ok:{en:'Correct',zh:'答对'},miss:{en:'Not retrieved',zh:'检索没找到'},cut:{en:'Cut by the budget',zh:'被预算挤掉'},stale:{en:'Outdated page',zh:'新旧文档矛盾'},
  unsup:{en:'Not from the context',zh:'没照着上下文答'},multi:{en:'Needs two passages',zh:'要两段合起来'},noans:{en:'No answer in the docs',zh:'文档里没有答案'}};
const KD={miss:{en:'none of the passages the answer needs is among the chunks retrieved.',zh:'回答要用的段落，一段也没检索到。'},
  cut:{en:'it is retrieved but ranks below the budget line, so it never reaches the context.',zh:'检索到了，但排在预算线下面，没放进上下文。'},
  stale:{en:'the index still holds a replaced edition of a page, and the model answers from the old one.',zh:'文档库里还留着被替换的旧版本，模型照着旧的答。'},
  unsup:{en:'the right passage is in the context, but the model writes something the passage does not say.',zh:'对的段落就在上下文里，模型却写了段落里没有的内容。'},
  multi:{en:'the answer is split over two passages; retrieval finds one and misses the other.',zh:'答案分在两段里，检索只找到其中一段，另一段没检索到。'},
  noans:{en:'the documents do not say; the model should say so but makes up an answer.',zh:'文档里根本没写，模型本该说不知道，却编了一个答案。'}};
/* the same six kinds in one short line each, under the options of part 3 (round-1 polish P8) */
const KDS={miss:{en:'None of the needed passages was retrieved',zh:'要用的段落一段也没取回来'},multi:{en:'The answer is split over two passages; only one was retrieved',zh:'答案分在两段里，只取回了一段'},
  cut:{en:'Retrieved, but ranked below the budget line',zh:'取回了，但排在预算线下面'},stale:{en:'An old edition is still indexed, and the model follows it',zh:'旧版本还在库里，模型照着旧的答'},
  unsup:{en:'The passage is in the context, but the answer says something it does not',zh:'段落在上下文里，回答却写了段落没说的'},noans:{en:'The documents do not say, but the model makes up an answer',zh:'文档没写，模型却编了一个答案'}};
/* the stages of the record, by the number the picture gives them */
const STG={ret:{n:2,en:'Retrieval',zh:'检索'},ctx:{n:3,en:'Context',zh:'放进上下文'},ans:{n:4,en:'Answer',zh:'回答'}};
const stgName=s=>`<span class="rgf-sn">${STG[s].n}</span>${esc(L(STG[s]))}`;
const CKN={ret:{en:'Retrieval recall',zh:'检索召回率'},ctx:{en:'Context recall',zh:'上下文召回率'},grd:{en:'Grounded',zh:'有据可依'},cor:{en:'Answer correct',zh:'答案正确率'}};
const CKM={ret:{en:'Retrieval',zh:'检索召回率'},ctx:{en:'Context',zh:'上下文召回率'},grd:{en:'Grounded',zh:'有据可依'},cor:{en:'Correct',zh:'答案正确率'}};
const CKS={ret:{en:'Retrieved',zh:'检索'},ctx:{en:'In context',zh:'上下文'},grd:{en:'Grounded',zh:'有据'},cor:{en:'Correct',zh:'正确'}};
const CKD={ret:()=>tx('Retrieval recall: of the questions the documents can answer, the share whose needed passages are all among the chunks retrieved.','检索召回率：文档里有答案的问题中，回答要用的段落全都被检索取回来的比例。'),
  ctx:()=>tx('Context recall: of the questions the documents can answer, the share whose needed passages all reach the context.','上下文召回率：文档里有答案的问题中，回答要用的段落全都放进了上下文的比例。'),
  grd:()=>tx('Grounded: the share of answers in which every sentence is supported by a chunk in the context. It counts whole answers, so one unsupported sentence fails the answer.','有据可依：回答里的每一句话都能在上下文的段落里找到依据，这样的回答占的比例。这里按整个回答算，有一句没有依据就不算。'),
  cor:()=>tx('Answer correctness: the share of answers that agree with the reference answer. When the documents have no answer, only “the help center does not say” is correct.','答案正确率：和标准答案一致的回答占的比例。文档里没有答案的问题，只有回答“帮助中心没有写”才算对。')};
const FXN={hyb:{en:'Hybrid search + rerank',zh:'混合检索 + rerank'},ctx:{en:'Bigger budget (8 chunks)',zh:'加大预算（放 8 段）'},ver:{en:'Version filter',zh:'文档版本过滤'},
  cite:{en:'Cite, or say “not in the docs”',zh:'要求引用、允许说不知道'},dec:{en:'Split the question',zh:'拆分问题'}};
const shareTxt=f=>KINDS.filter(k=>SHARE[f][k]).sort((a,b)=>SHARE[f][b]-SHARE[f][a]).map(k=>tx(`${Math.round(SHARE[f][k]*100)}% of “${L(KN[k])}”`,`“${L(KN[k])}”的 ${Math.round(SHARE[f][k]*100)}%`)).join(tx(', ','、'));
const waitTxt=ms=>tx(`${A.fmt.int(ms)} ms`,`${A.fmt.int(ms)} 毫秒`);
const WM=ms=>ms?tx(`${waitTxt(ms)} more wait per question`,`每个问题多等 ${waitTxt(ms)}`):tx('no extra wait','不多等');
/* each fix: what it does in one short sentence (FXS), how it works in at most two (FXT, the tooltip), and what it repairs (fxEffect) */
const FXS={hyb:()=>tx('Keyword and vector search together, then a model reranks.','关键词和向量一起检索，再用模型重新排序。'),
  ctx:()=>tx('Retrieve 10 chunks and send the first 8 instead of 3.','取回 10 段，放进上下文的从前 3 段改成前 8 段。'),
  ver:()=>tx('Replaced editions stay out of the index.','被替换的旧版本不进索引。'),
  cite:()=>tx('Cite a chunk for every sentence; with none, say “the help center does not say”.','每句话标出处；没有依据就答“帮助中心没有写”。'),
  dec:()=>tx('Split the question into smaller ones and search each.','先把问题拆成几个小问题，分别检索。')};
const FXT={hyb:()=>tx('Keyword search (BM25) and vector search each find candidate passages, merged into a top 20. A model that reads the question and each passage together rescores the 20 and moves the most relevant to the top.','关键词检索（BM25）和向量检索各自找出候选段落，合并成前 20 段。再用一个同时读问题和段落的模型给这 20 段重新打分，把最相关的排到前面。'),
  ctx:()=>tx('Passages ranked 4th to 8th now reach the context, but with more chunks the model more often overlooks one in the middle (Lost in the Middle), so only part is repaired. The passages of a “Not retrieved” question all rank below 10th, out of reach.','排第 4 到第 8 名的段落也能放进上下文了，可段落一多，模型更容易漏看夹在中间的那一段（Lost in the Middle），所以只修好一部分。“检索没找到”的段落都排在第 10 名以后，加大预算也拿不到。'),
  ver:()=>tx('Every chunk is tagged with its document’s edition when indexing and old editions stay out, so the model sees only the current one. The filtering is done then, so answering waits no longer.','建索引时给每一段标上所属文档的版本，旧版本不进索引，模型只看得到新版本。过滤在建索引时就做完了，所以回答时不多等。'),
  cite:()=>tx('Having to cite a chunk leaves less room for sentences no chunk supports, and “the help center does not say” replaces a made-up answer. Answers get a little longer, so each waits about 100 ms more.','模型要给每句话找出处，没有依据的句子就少了；找不到出处时答“帮助中心没有写”，就不会编答案。回答变长一些，所以多等约 100 毫秒。'),
  dec:()=>tx('“My lock shows E-104; is the repair free?” becomes “what is E-104?” and “how long is the warranty?”, so both passages are found. One more model call and more searches add about 600 ms.','“门锁显示 E-104，维修免费吗”会拆成“E-104 是什么”和“保修多久”，两段都能检索到。多一次模型调用、多几次检索，所以多等约 600 毫秒。')};
const fxEffect=(f,assumed)=>{const tok=f==='ctx'?tx(', and the prompt uses about 1,500 more tokens','，prompt 多用约 1,500 个 token'):'';   // N4 (round-1 check): what grows
  return assumed?tx(`Assumed to repair ${shareTxt(f)}; ${WM(WAIT[f])}${tok}.`,`假设修好${shareTxt(f)}；${WM(WAIT[f])}${tok}。`):tx(`Repairs ${shareTxt(f)}; ${WM(WAIT[f])}${tok}.`,`修好${shareTxt(f)}；${WM(WAIT[f])}${tok}。`);};
const fxList=fx=>fx.length===FIXES.length?tx('all five fixes','五种改法全用'):fx.length?fx.map(f=>L(FXN[f])).join(tx(' + ',' + ')):tx('initial setup, no fix','初始设置，没有改法');
const chunkName=id=>esc(CH[id][0][li()]);

/* ---------- state ---------- */
const DEFAULT={run:1,fx:[],log:1,met:'cor'};
let S={...DEFAULT,fx:[]}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let tri={i:0,picks:[null,null,null]}, chSubmitted=false, chReveal=false, chFx=[], drillSeed=1, drillOpen=false, userAct=false, annT=0;
const seenRun=new Set(), shownPrev={}, tried=new Set(), lvTried=new Set();   // lvTried: fixes tried in part 6 (a result shows only after its pick, R3.8)
let speed='mid', anim=null, raf=0, lastTs=0, runOpen=false, reqPlay=false;   // runOpen: the run record under the cards in Scenarios; reqPlay: the reader asked for the record (▶, a run, a question)
const DUR={slow:16,mid:10,fast:6};   // seconds for one run's replay; at 中 each beat stays about a second (R3.9, R7.11; round-1 polish P3)
const phone=()=>!!(window.matchMedia&&matchMedia('(max-width:760px)').matches);
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const TRI=[2,3,4];   // the triage runs: cut, outdated page, not from the context
const RI=id=>RUNS.findIndex(r=>r.id===id);
const sys=()=>mode==='challenge'?'ch':'base';
const fxNow=()=>mode==='challenge'?(chSubmitted?chFx:[]):S.fx;
const runNow=()=>mode==='guided'&&STEPS[step].id==='triage'?TRI[tri.i]:S.run;
const sameFx=(a,b)=>a.length===b.length&&a.every(x=>b.includes(x));

/* ---------- guided parts ---------- */
const segIn=(key,label,opts,val)=>A.ui.seg({id:'rgf-ic-'+key,key,label,options:opts,value:String(val),inline:true});
const PRED=['ret','after','even'];
const predTxt=v=>({ret:tx('In retrieval: the right passage was not found','错在检索：对的段落没找到'),after:tx('After retrieval: the passage was found, then cut by the budget, outvoted by an old page or not used by the model','错在检索之后：段落找到了，却被预算挤掉、被旧文档带偏，或者模型没照着答'),even:tx('About half and half; these two numbers cannot tell','两边差不多，光看这两个百分比分不出来')})[v];
function triVerdict(i){const k=RUNS[TRI[i]].kind, p=tri.picks[i]; if(!p) return '';
  const ok=p===k;
  return `<div class="rgf-verd">`+(ok?tx(`<p class="okline"><b>Right: ${esc(L(KN[k]))}.</b></p>`,`<p class="okline"><b>对：${esc(L(KN[k]))}。</b></p>`)
    :tx(`<p class="badline"><b>Not quite.</b> You picked “${esc(L(KN[p]))}”; the answer is “${esc(L(KN[k]))}”.</p>`,`<p class="badline"><b>不对。</b>你选了“${esc(L(KN[p]))}”，答案是“${esc(L(KN[k]))}”。</p>`))+`<p>${runWhy(TRI[i])}</p></div>`;}
function runWhy(ri){const r=RUNS[ri], g=r.gold;
  switch(r.kind){
  case 'ok': return tx('The passage on charging ranks first, reaches the context, and the answer says what it says.','讲充电的那段排第 1，放进了上下文，回答照着它说。');
  case 'miss': return tx(`The page that says accidental damage is not covered ranks ${ord(g[0][1])}: search never returns it, because “fell down the stairs” and “dropping the device” share no word and their vectors are not close enough. The model answers from the coverage page it did get: grounded, and wrong.`,`讲“意外损坏不保修”的那段排第 ${g[0][1]}，检索没取回来：“从楼梯上摔下去”和“设备摔落”没有一个词相同，向量也不够近。模型照着取回来的保修范围那段回答：有依据，但答错了。`);
  case 'cut': return tx(`The auto-lock passage ranks ${ord(g[0][1])}, just below the budget line: it was retrieved but never reached the model. The answer is grounded in the app passage and does not answer the question.`,`讲自动上锁的那段排第 ${g[0][1]}，正好在预算线下面：检索到了，却没放进上下文。回答照着讲 App 的那段说，有依据，却没回答问题。`);
  case 'stale': return tx('Both editions of the warranty page reach the context, and the replaced 2024 edition ranks first. The model follows it: the answer is grounded, but in an outdated page. Only the reference answer shows that it is wrong.','保修页的新旧两版都放进了上下文，被替换的 2024 年版排在第 1。模型照着它答：有依据，可依据过时了。只有和标准答案比，才发现答错了。');
  case 'unsup': return tx('The filter passage ranks first and reaches the context; it says once a month. The model writes once a week, which no chunk in the context says: the answer is not grounded.','讲滤网的那段排第 1，放进了上下文，写的是每月一次。模型却写了每周一次，上下文里没有哪段这样说：回答没有依据。');
  case 'multi': return tx(`The E-104 passage ranks first, but the warranty passage ranks ${ord(g[1][1])}: the question never says “warranty”. With half of what it needs, the model answers only the first half of the question.`,`讲 E-104 的那段排第 1，讲保修期的那段却排第 ${g[1][1]}：问题里没说“保修”两个字。只拿到一半材料，模型只回答了前一半。`);
  default: return tx('No chunk in the help center mentions a smartwatch. The model makes up an app on the watch; the right answer was “the help center does not say”.','帮助中心没有哪一段提到智能手表。模型编了一个手表上的 App；对的回答是“帮助中心没有写”。');}}
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'“Wrong answer” is a symptom: find the stage.',zh:'“答错了”只是症状：先找出错在哪一环。'},
  panels:['run'],enter:()=>({run:1,fx:[],log:1}),
  body:()=>{const m=compute('base',[]);
   return tx(`<p>Lumo Home’s help-center bot answers users with retrieval-augmented generation (RAG): it searches the help pages with the question, takes the ${NRET} most similar chunks, puts only the first ${BUDGET} into the model’s prompt (this cap is called the context budget), and the model answers from them.</p><p>The team graded its answers to 200 test questions (example): ${m.ok} are correct (${pc(m.cor)}).</p><p>${J('flow','The picture')} follows one of the wrong answers from the question through retrieval, the context and the answer to the judging; the red frame marks the stage that failed.</p><p>When an answer is wrong, how do you find the stage that failed, and which stage to change?</p>`,
             `<p>小鹿家居的帮助中心机器人用检索增强生成（RAG）回答用户：拿问题去搜帮助文档，取回最像的 ${NRET} 段，只把前 ${BUDGET} 段放进模型的 prompt（这个上限叫上下文预算），再让模型照着这些段落回答。</p><p>团队拿 200 个测试问题（示例）给它判对错，它答对 ${m.ok} 个（${pc(m.cor)}）。</p><p>${J('flow','图里')}是其中一个答错的问题，从问题、检索、放进上下文、回答一直走到判分；红框是出错的那一环。</p><p>答错了，怎么找出错在哪一环、该改哪一环？</p>`);}},
 {id:'log',title:{en:'Log every stage',zh:'每一环都记下来'},lead:{en:'Question and answer alone show that it is wrong, not why.',zh:'只记问题和回答，看得出错了，看不出为什么。'},
  panels:['run'],enter:()=>({run:RI('e104'),fx:[],log:0}),expect:()=>S.log===1,showMe:{log:1},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Log every stage to continue.',zh:'选“每一环都记”再继续。'},
  body:()=>tx(`<p>To find the stage that failed, every stage needs a record. A log of only the question and the answer shows that an answer is wrong, not where it went wrong.</p><p>For every question, log the chunks retrieved and the rank of each, the chunks actually put into the context, and the model’s answer. For test questions, also write a reference answer and mark the passages it needs (★ in the picture); the four marks at the judging stage are computed from these.</p><p>${J('flow','The picture')} now shows another wrong answer: a lock shows E-104, and the user asks whether the repair is free. Choose what the log keeps, and see what is left of the record.</p>`+
              segIn('log',tx('The log keeps','日志里记了什么'),[['0',tx('Question and answer','只记问题和回答')],['1',tx('Every stage','每一环都记')]],S.log),
              `<p>要找出错在哪一环，每一环都得有记录。只记问题和回答，只看得出答错了，看不出错在哪。</p><p>每个问题要记：检索取回了哪几段、各排第几；真正放进上下文的是哪几段；模型怎么答的。测试问题还要写好标准答案，标出要用的段落（图里的 ★），判分那一环的四个标记就是拿这些查出来的。</p><p>${J('flow','图里')}换了一个答错的问题：门锁显示 E-104，用户问维修免不免费。选日志里记什么，看看记录还剩下什么。</p>`+
              segIn('log',tx('The log keeps','日志里记了什么'),[['0',tx('Question and answer','只记问题和回答')],['1',tx('Every stage','每一环都记')]],S.log)),
  obs:()=>{const r=RUNS[RI('e104')];
   return tx(`<p><b>With every stage logged: the E-104 passage ranks first and reaches the context; the warranty passage ranks ${ord(r.gold[1][1])} and is never retrieved.</b></p><p>The answer needs both passages and retrieval found only one, so the model only says “contact support” and never says whether the repair is free.</p>`,
             `<p><b>每一环都记下来以后：讲 E-104 的那段排第 1，放进了上下文；讲保修期的那段排第 ${r.gold[1][1]}，没被检索到。</b></p><p>回答要用这两段，检索只找到一段，所以模型只说了“联系客服维修”，没说免不免费。</p>`);}},
 {id:'triage',title:{en:'Triage, one run at a time',zh:'逐个诊断'},lead:{en:'Name the kind of failure, then check.',zh:'先判断是哪一类错，再看答案。'},
  panels:['run'],enter:()=>({fx:[],log:1}),gate:()=>tri.picks.every(Boolean),skip:true,nextWhy:{en:'Diagnose all three questions to continue, or Skip.',zh:'三个问题都诊断完再继续，或点“跳过”。'},
  body:()=>{const i=tri.i, p=tri.picks[i], k=RUNS[TRI[i]].kind;
   /* the six kinds are the options, each with its short definition; after a wrong pick the right option is marked (round-1 polish P8, P9) */
   const ch=`<div class="choices rgf-tri" role="group" aria-label="${esc(tx('Which kind of failure is it?','这是哪一类错？'))}">${KINDS.map(x=>`<button type="button" class="choice${p&&p!==k&&x===k?' right':''}" data-tri="${x}" aria-pressed="${p===x}"${p?' disabled':''}><span class="rgf-ch"><b>${esc(L(KN[x]))}</b><span class="rgf-kd">${esc(L(KDS[x]))}</span></span></button>`).join('')}</div>`;
   const mk=j=>{const q=tri.picks[j]; return q?(q===RUNS[TRI[j]].kind?' ✓':' ✗'):'';};
   const nav=`<div class="rgf-trinav">${TRI.map((x,j)=>`<button type="button" class="btn ghost sm" data-trigo="${j}" aria-pressed="${j===i}"${j>0&&!tri.picks[j-1]&&!tri.picks[j]?' disabled':''}>${tx(`Question ${j+1}${mk(j)}`,`问题 ${j+1}${mk(j)}`)}</button>`).join('')}</div>`;
   return tx(`<p>${J('flow','The picture')} shows question ${i+1} of 3 that the bot got wrong; ★ marks the passages the reference answer needs. Read the chunks retrieved, the budget line, the answer and the reference answer, and name the kind of failure.</p>`,
             `<p>${J('flow','图里')}是机器人答错的第 ${i+1} 个问题（共 3 个），★ 是标准答案要用的段落。看检索到的段落、预算线、回答和标准答案，判断这是哪一类错。</p>`)+nav+ch+triVerdict(i)+
     (p&&i<TRI.length-1?`<div class="stepnav"><button type="button" class="btn sm" data-trigo="${i+1}">${tx('Next question','下一个问题')}</button></div>`:'');}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'High retrieval recall, many wrong answers: where are they?',zh:'检索召回率很高，答错的却不少：错在哪？'},
  panels:['sum'],enter:()=>({fx:[],log:1}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const m=compute('base',[]);
   const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>This bot’s ${DEF('retrieval recall',CKD.ret())} is ${pc(m.ret)}: for almost all of the ${m.ans} questions the documents can answer, every needed passage was retrieved. Yet only ${pc(m.cor)} of the 200 answers are correct.</p><p>Where are most of the wrong answers?</p>`+ch+(pred?`<p class="small muted">${tx('Saved. You can still change it before the next part.','已记下，进入下一部分之前还可以改。')}</p>`:''),
             `<p>这个机器人的${DEF('检索召回率',CKD.ret())}是 ${pc(m.ret)}：文档里有答案的 ${m.ans} 个问题，回答要用的段落几乎都检索到了。可 200 个回答只答对 ${pc(m.cor)}。</p><p>答错的回答，大多数错在哪？</p>`+ch+(pred?`<p class="small muted">已记下，进入下一部分之前还可以改。</p>`:''));}},
 {id:'metrics',title:{en:'Check every stage',zh:'每一环都检查'},lead:{en:'Answer correctness alone hides the cause.',zh:'只看答案对不对，看不出原因。'},
  panels:['bars','metrics'],enter:()=>({fx:[],met:'cor'}),expect:()=>tried.size>1,showMe:{met:'ret'},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another check to continue.',zh:'换一项检查再继续。'},
  body:()=>{const m=compute('base',[]), lost=m.ans-m.nRet, after=m.wrong-lost, ok=pred==='after';
   /* the verdict and its arithmetic in one sentence, so the part's point comes second (round-1 polish P12, as the check proposed) */
   const v=pred?(ok?tx(`<p class="okline"><b>Right:</b> of the ${m.wrong} wrong answers, at most about ${lost} fail at retrieval and at least ${after} after it.</p>`,`<p class="okline"><b>对：</b>答错的 ${m.wrong} 个里，最多约 ${lost} 个错在检索，至少 ${after} 个错在检索之后。</p>`)
     :tx(`<p class="badline"><b>Not quite:</b> of the ${m.wrong} wrong answers, at most about ${lost} fail at retrieval (with ${pc(m.ret)} retrieval recall, about ${lost} of the ${m.ans} answerable questions were not fully retrieved), and at least ${after} fail after it.</p>`,
         `<p class="badline"><b>不对：</b>答错的 ${m.wrong} 个里，最多约 ${lost} 个错在检索（检索召回率 ${pc(m.ret)}，${m.ans} 个有答案的问题里约 ${lost} 个没检索全），至少 ${after} 个错在检索之后。</p>`))
     :tx(`<p>The answer: of the ${m.wrong} wrong answers, at most ${lost} fail at retrieval and at least ${after} after it.</p>`,`<p>答案：${m.wrong} 个答错的回答里，最多 ${lost} 个错在检索，至少 ${after} 个错在检索之后。</p>`);
   const seg=segIn('met',tx('Check','检查'),CHK.map(c=>[c,L(CKM[c])]),S.met);
   return v+tx(`<p>Answer correctness alone lumps all ${m.wrong} failures together, so it cannot tell you what to change. ${J('bars','The chart')} sorts them into six kinds by where they failed; the four columns on the right are the four checks, and a ✗ means the check catches that kind.</p><p>The ${m.N-m.ans} questions whose answer is not in the documents have no passage to find, so the two recall checks skip them, marked —. Pick a check and see which kinds it catches and which it misses.</p>`+seg,
              `<p>只看答案正确率，${m.wrong} 个错混在一起，看不出该改哪里。${J('bars','图里')}把它们按出错的地方分成六类；右边四列是四项检查，✗ 表示这项检查能发现这一类。</p><p>文档里没有答案的 ${m.N-m.ans} 个问题，没有要找的段落，两项召回率不算它们，记作 —。选一项检查，看它能发现哪几类、漏掉哪几类。</p>`+seg);},
  obs:()=>{const m=compute('base',[]), c=S.met, i=CHK.indexOf(c), ks=KINDS.filter(k=>FLAG[k][i]===1), n=ks.reduce((a,k)=>a+m.left[k],0), v={ret:m.ret,ctx:m.ctx,grd:m.grd,cor:m.cor}[c];
   const q=k=>`“${L(KN[k])}”`;   // one sentence on why the check sees what it sees (round-1 polish P6, P12)
   const why={ret:tx(`It only asks whether the needed passages were retrieved, so it catches only ${q('miss')} and ${q('multi')}.`,`它只看要用的段落有没有被检索取回来，所以只发现${q('miss')}和${q('multi')}这两类。`),
     ctx:tx(`A passage never retrieved cannot reach the context either, so besides ${q('cut')} it also catches the two retrieval kinds.`,`没被取回来的段落也进不了上下文，所以它除了发现${q('cut')}，也发现检索那一环的两类。`),
     grd:tx(`It checks that the answer follows the passages in the context, so it catches only the two answer-stage kinds, ${q('unsup')} and ${q('noans')}. In practice another model scores it against a rubric; see ${LL('llm-judge')}.`,`它查回答是不是照着上下文里的段落说，所以只发现回答那一环的${q('unsup')}和${q('noans')}。实际中通常由另一个模型按评分标准来判，${SEE('llm-judge')}。`),
     cor:tx(`It catches all six kinds but cannot tell them apart. Only it catches ${q('stale')}: the passage was found and sent and the answer follows it, but the passage is out of date.`,`它能发现所有六类，但分不出是哪一类。${q('stale')}只有它发现得了：段落找到了、放进去了，回答也照着段落说，只是段落过时了。`)}[c];
   return tx(`<p><b>${L(CKN[c])} is ${pc(v)}: it catches ${n===m.wrong?`all ${n} wrong answers`:`${n} of the ${m.wrong} wrong answers`}.</b></p><p>${why}</p>`,
             `<p><b>${L(CKN[c])}是 ${pc(v)}：${n===m.wrong?`答错的 ${n} 个它全都发现了`:`它发现了答错的 ${m.wrong} 个里的 ${n} 个`}。</b></p><p>${why}</p>`);}},
 {id:'fixes',title:{en:'Fix the stage that failed',zh:'对症下药'},lead:{en:'Each fix repairs part of a few kinds, at a cost.',zh:'每种改法只管几类错，各有代价。'},
  panels:['bars','metrics'],enter:()=>({fx:[],met:'cor'}),expect:()=>tried.size>1,showMe:{fx:['cite']},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick a fix to continue.',zh:'选一种改法再继续。'},
  body:()=>tx(`<p>Once you know the stage that failed, change that stage. Each fix covers a few kinds of failure and repairs only part of each; most fixes also make every question wait a little longer. How much a fix repairs and how long it adds are this lab’s assumptions. Pick a fix and watch which kinds shrink in ${J('bars','the chart')}.</p>`,
              `<p>知道错在哪一环，就改那一环。每种改法只管几类错，每类也只修好一部分；大多数改法还会让每个问题多等一会儿。修好多少、多等多久，是这个 Lab 的假设。选一种改法，看看${J('bars','图里')}哪几类错变少了。</p>`)+
    `<div class="levers rgf-levers" role="group" aria-label="${esc(tx('Fixes, pick one','改法，选一种'))}">${FIXES.map(f=>`<button type="button" class="choice lever" data-lever="${f}" aria-pressed="${S.fx.length===1&&S.fx[0]===f}"><span class="lv-t"><b>${esc(L(FXN[f]))}</b><span class="lv-d">${FXLINES(f,false)}</span></span><span class="lv-r mono">${lvTried.has(f)?tx(`${pc(compute('base',[f]).cor)} correct`,`答对 ${pc(compute('base',[f]).cor)}`):''}</span></button>`).join('')}</div>`,
  obs:()=>{const m0=compute('base',[]), f=S.fx[0]; if(!f) return tx('<p>No fix chosen: the initial setup.</p>','<p>没选改法：初始设置。</p>');
   const m=compute('base',[f]), ks=KINDS.filter(k=>m.fixed[k]>0), rest=KINDS.filter(k=>!SHARE[f][k]).sort((a,b)=>m0.left[b]-m0.left[a]).slice(0,2);
   return tx(`<p><b>${L(FXN[f])}: ${m.ok-m0.ok} more correct answers, ${pc(m.cor)} in all.</b></p><p>It repairs ${ks.map(k=>`${m.fixed[k]} “${L(KN[k])}”`).join(', ').replace(/, ([^,]*)$/,' and $1')}. The two largest kinds it does not touch, ${rest.map(k=>`“${L(KN[k])}”`).join(' and ')}, need a different fix.</p>`,
             `<p><b>${L(FXN[f])}：答对的从 ${pc(m0.cor)} 变成 ${pc(m.cor)}，多了 ${m.ok-m0.ok} 个。</b></p><p>它修好了${ks.map(k=>`“${L(KN[k])}”${m.fixed[k]} 个`).join('、')}。它管不到的错里，最大的两类是${rest.map(k=>`“${L(KN[k])}”`).join('和')}，要用别的改法。</p>`);}},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Trace, check each stage, pick the fix that repairs the most.',zh:'看记录、每环检查、选修得最多的改法。'},
  panels:['bars','metrics'],enter:()=>({fx:[],met:'cor'}),
  body:()=>{const m0=compute('base',[]), h=compute('base',['hyb']).ok-m0.ok, c=compute('base',['ctx']).ok-m0.ok, ci=compute('base',['cite']).ok-m0.ok, st=RUNS[1];
   const big=KINDS.slice().sort((a,b)=>m0.left[b]-m0.left[a])[0];
   return tx(`<ul class="rgf-ul"><li><b>Find the stage that failed before you change anything.</b> A user asks whether a vacuum that fell down the stairs is repaired for free, and the bot tells them to start a warranty claim, as if a fall were covered. The passage “accidental damage is not covered” ranked ${ord(st.gold[0][1])} and was never retrieved, so retrieval is what to fix.</li><li><b>Log every stage and check every stage.</b> Answer correctness alone mixes every failure together. Retrieval recall, context recall and groundedness watch retrieval, the context and the answer in turn; an outdated page shows up only against the reference answer.</li><li><b>Pick fixes by how many they repair and how long they add.</b> The largest kind is “${L(KN[big])}” (${m0.left[big]}): hybrid search + rerank or a bigger budget ${h===c?`each adds ${h}`:`adds ${h} or ${c}`} correct answers. Asking for citations and allowing “not in the docs” handles the two answer-stage kinds (${m0.left.unsup} and ${m0.left.noans}), adds ${ci} and waits only ${waitTxt(WAIT.cite)} more.</li></ul>`+
              `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “A bigger model fixes it.” If the right passage was not retrieved or was cut by the budget, no model can see it; if an old and a new edition are both in the context and no passage says which one was replaced, a bigger model can only guess.</p><p>✗ “Retrieved means answered right.” Here retrieval recall is ${pc(m0.ret)} and correctness ${pc(m0.cor)}: a found passage can still be cut by the budget, outvoted by an old page or ignored by the model.</p><p>✗ “Saying ‘I don’t know’ is a failure.” When the documents have no answer, “the help center does not say” is the correct answer; making one up is the failure.</p></div>`+A.shell.toScenarios(),
              `<ul class="rgf-ul"><li><b>先找出错的那一环，再动手改。</b>用户问扫地机器人摔下楼梯能不能免费修，机器人让他去申请保修，好像摔坏也在保修范围内。讲“意外损坏不保修”的那段排第 ${st.gold[0][1]}，根本没被检索到，所以该改的是检索。</li><li><b>每一环都记日志、都检查。</b>只看答案正确率，所有的错混在一起。检索召回率、上下文召回率、有据可依分别盯住检索、放进上下文和回答这三环；旧文档带来的错，只有和标准答案比才看得出来。</li><li><b>按修好多少、多等多久来选改法。</b>最大的一类是“${L(KN[big])}”（${m0.left[big]} 个），混合检索 + rerank 或加大预算${h===c?`各多答对 ${h} 个`:`分别多答对 ${h} 个和 ${c} 个`}；“${L(FXN.cite)}”管住回答那一环的两类错（${m0.left.unsup} 个和 ${m0.left.noans} 个），多答对 ${ci} 个，只多等 ${waitTxt(WAIT.cite)}。</li></ul>`+
              `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “换个更大的模型就好。”对的段落没取回来、或者被预算挤掉时，模型再强也看不到它；新旧两版都放进上下文、段落里又没写哪一版已被替换时，模型再强也只能猜。</p><p>✗ “检索到了就会答对。”这里检索召回率是 ${pc(m0.ret)}，答对的只有 ${pc(m0.cor)}：找到的段落还可能被预算挤掉、被旧版本带偏，或者被模型忽略。</p><p>✗ “说不知道就是失败。”文档里没有答案时，“帮助中心没有写”才是对的回答；编一个答案才是失败。</p></div>`+A.shell.toScenarios());}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);

/* ---------- scenarios ---------- */
const PRESETS=[
 {k:'init',t:{en:'Initial setup: no fix',zh:'初始设置：不改'},d:{en:'Vector search, 6 chunks retrieved, 3 sent, a plain prompt.',zh:'向量检索，取回 6 段，放进 3 段，普通的 prompt。'},fx:[]},
 {k:'retr',t:{en:'Better retrieval and ranking',zh:'改检索和排序'},d:{en:'Hybrid search + rerank.',zh:'混合检索 + rerank。'},fx:['hyb']},
 {k:'after',t:{en:'Fix what comes after retrieval',zh:'改检索之后的环节'},d:{en:'Version filter, and a prompt that asks for citations.',zh:'文档版本过滤，加上要求引用的 prompt。'},fx:['ver','cite']},
 {k:'more',t:{en:'More context',zh:'多放内容'},d:{en:'Bigger budget and split questions.',zh:'加大预算，再拆分问题。'},fx:['ctx','dec']},
 {k:'all',t:{en:'All five fixes',zh:'五种全用'},d:{en:'Every fix at once.',zh:'所有改法一起上。'},fx:FIXES.slice()}];
const presetNow=()=>PRESETS.findIndex(p=>sameFx(p.fx,S.fx));
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p>${tx('Pick a set of fixes for the help-center bot and see how its 200 graded answers (example) change.','给帮助中心机器人选一组改法，看看判过对错的 200 个回答（示例）怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="rgf-insight" aria-live="polite"></div>`+A.shell.toChallenge()+`</div>`;}
function insightHTML(){const m=compute('base',S.fx), m0=compute('base',[]), i=presetNow();
  let h=tx(`<p><b>${esc(fxList(S.fx))}: ${pc(m.cor)} correct, ${WM(m.wait)}.</b></p>`,`<p><b>${esc(fxList(S.fx))}：答对 ${pc(m.cor)}，${WM(m.wait)}。</b></p>`), hb=compute('base',['hyb']);
  if(i<0) return h+tx('<p class="small">The current fixes match none of these scenarios.</p>','<p class="small">现在的改法和这几个场景都不一样。</p>');
  const k=PRESETS[i].k, big=KINDS.slice().sort((a,b)=>m.left[b]-m.left[a])[0];
  if(k==='init') h+=tx(`<p>Retrieval recall is high (${pc(m.ret)}), yet ${m.wrong} answers are wrong.</p><p>The largest kind is “${L(KN[big])}” with ${m.left[big]}.</p>`,`<p>检索召回率很高（${pc(m.ret)}），答错的却有 ${m.wrong} 个。</p><p>最大的一类是“${L(KN[big])}”，${m.left[big]} 个。</p>`);
  /* the second sentence brings only numbers the bold line does not give (round-1 polish P20) */
  else if(k==='retr') h+=tx(`<p>Retrieval recall rises to ${pc(m.ret)}.</p><p>Of the ${m.ok-m0.ok} more correct answers, ${m.fixed.cut} were cut by the budget before: rerank moved the needed passage inside the budget line. Questions retrieval truly missed were few to begin with.</p>`,`<p>检索召回率升到了 ${pc(m.ret)}。</p><p>多答对的 ${m.ok-m0.ok} 个里，${m.fixed.cut} 个原来被预算挤掉：rerank 把要用的段落排到了预算线以内。真正没检索到的错本来就不多。</p>`);
  else if(k==='after'){const a3=m0.left.stale+m0.left.unsup+m0.left.noans, r3=m0.left.miss+m0.left.multi+m0.left.cut, ra=m.ok-m0.ok, rh=hb.ok-m0.ok;   // each side's three kinds, the largest one included (P29; N3 of the round-1 check)
    const tail=2*rh===r3?tx(', and it repairs only half of them','，而且只修好其中一半'):(rh*a3<ra*r3?tx(', and it repairs a smaller share of them','，修好的比例也更低'):'');
    h+=tx(`<p>They repair ${ra}, more than hybrid search + rerank (${rh}).</p><p>Their three kinds add up to ${a3} questions; the three that hybrid search + rerank covers (“${L(KN.miss)}”, “${L(KN.multi)}” and “${L(KN.cut)}”) add up to only ${r3}${tail}.</p>`,`<p>这两种改法修好了 ${ra} 个，比混合检索 + rerank（${rh} 个）多。</p><p>它们管的三类加起来有 ${a3} 个；混合检索 + rerank 管的三类（“${L(KN.miss)}”、“${L(KN.multi)}”和“${L(KN.cut)}”）只有 ${r3} 个${tail}。</p>`);}
  else if(k==='more') h+=tx(`<p>Together they repair ${m.ok-m0.ok}, ${m.fixed.cut} of them cut by the budget before; splitting the question accounts for ${waitTxt(WAIT.dec)} of the wait.</p>`,`<p>这两种改法一共修好 ${m.ok-m0.ok} 个，其中 ${m.fixed.cut} 个原来被预算挤掉；多等的时间里，拆分问题占了 ${waitTxt(WAIT.dec)}。</p>`);
  else h+=tx(`<p>${m.wrong} answers stay wrong: in this lab’s assumptions no fix repairs them.</p>`,`<p>还剩 ${m.wrong} 个答错，是这个 Lab 的假设里哪种改法都修不好的。</p>`);
  return h;}

/* ---------- challenge ---------- */
const chDone=()=>mode==='challenge'&&chSubmitted;
function chMe(){const m=compute('ch',chFx); return {p:chFx.slice(),ok:m.ok,cor:m.cor,wait:m.wait,met:chFx.length===2&&m.cor>=CH_GOAL-1e-9&&m.wait<=CH_WAIT};}
function challengeHTML(){const m0=compute('ch',[]);
  const opts=FIXES.map(f=>{const on=chFx.includes(f), dis=!on&&chFx.length>=2;
    return `<button type="button" class="rgf-fxb" data-chfx="${f}" aria-pressed="${on}"${dis?' disabled':''}><b>${esc(L(FXN[f]))}</b><span>${esc(tx(`Repairs ${shareTxt(f)}.`,`修好${shareTxt(f)}。`))}</span><span>${esc(WAIT[f]?tx(`+${waitTxt(WAIT[f])} per question`,`每个问题多等 ${waitTxt(WAIT[f])}`):tx('No extra wait','不多等'))}</span></button>`;}).join('');
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Two fixes, at most 300 ms more per question: get the installer bot to at least 75% correct answers','两种改法、每个问题最多多等 300 毫秒：让安装助手至少答对 75%')}</h3>`+
   tx(`<p>Another Lumo Home bot, the installer assistant, answers electricians’ questions. It is set up like the help-center bot: vector search returns 6 passages and the first 3 go into the context. Of its 200 graded questions (example), only ${pc(m0.cor)} are correct; ${J('bars','the chart')} sorts the wrong ones by kind. Installation manuals change often, and many questions need the model page and the wiring page together, so outdated pages and two-passage questions are the largest kinds.</p><p>The team can ship two fixes this sprint, and users accept at most 300 ms more wait per question. Pick the two fixes.</p>`,
      `<p>小鹿家居的另一个机器人：安装助手，回答电工和安装师傅的问题。它的设置和帮助中心机器人一样：向量检索取回 6 段，前 3 段放进上下文。判过对错的 200 个问题（示例）里只答对 ${pc(m0.cor)}，${J('bars','图里')}把答错的按类别分好了。安装手册常常改版，问题又常要把型号和接线两段合起来，所以旧文档和要两段的错最多。</p><p>团队这一轮只来得及上线两种改法，用户最多能接受每个问题多等 300 毫秒。选两种改法。</p>`)+
   `<div class="rgf-fxs" role="group" aria-label="${esc(tx('Fixes, pick two','改法，选两种'))}">${opts}</div>`+
   `<div class="goal mono" id="rgf-ch-n"><span>${tx('Fixes picked','已选的改法')}</span><span><b id="rgf-ch-nv"></b>${tx(' of 2',' / 2')}</span></div>`+
   `<div class="goal mono" id="rgf-ch-w"><span>${tx('Added wait per question','每个问题多等')}</span><span><b id="rgf-ch-wv"></b>${tx(' (at most 300 ms)','（最多 300 毫秒）')}</span></div>`+
   `<div class="goal mono" id="rgf-ch-c"><span>${tx('Correct answers','答对的比例')}</span><span><b id="rgf-ch-cv"></b>${tx(' (goal: at least 75%)','（目标：至少 75%）')}</span></div>`+
   `<div id="rgf-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit"${chFx.length===2?'':' disabled'}>${tx('Submit my answer','提交答案')}</button><p class="small muted rgf-subw">${chFx.length===2?'':tx('Pick two fixes to submit.','选好两种改法再提交。')}</p></div>`+
   `<div id="rgf-ch-res" class="rgf-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chDone()) return '';
  const pts=chAll(), me=chMe(), best=pts.filter(p=>p.best)[0], m0=compute('ch',[]), why=[];
  if(me.cor<CH_GOAL-1e-9) why.push(tx(`${pc(me.cor)} correct is under 75%`,`答对 ${pc(me.cor)}，不到 75%`));
  if(me.wait>CH_WAIT) why.push(tx(`${waitTxt(me.wait)} more wait per question is over the 300 ms limit`,`每个问题多等 ${waitTxt(me.wait)}，超过了 300 毫秒的上限`));
  const isMe=p=>sameFx(p.p,me.p);
  let h;
  if(me.met&&me.ok===best.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met, and the best pair: ${me.ok} correct (${pc(me.cor)}), ${waitTxt(me.wait)} more per question.</b></p>`,`<p class="okline" style="margin-top:12px"><b>达标，而且是最好的一对：答对 ${me.ok} 个（${pc(me.cor)}），每个问题多等 ${waitTxt(me.wait)}。</b></p>`);
  else if(me.met) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${pc(me.cor)} correct.</b> But ${esc(fxList(best.p))} reaches ${pc(best.cor)} within the same limit.</p>`,`<p class="okline" style="margin-top:12px"><b>达标：答对 ${pc(me.cor)}。</b>不过${esc(fxList(best.p))}在同样的限制下能到 ${pc(best.cor)}。</p>`);
  else {h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${why.join('; ')}.</b></p><p class="small">Hint: start with the largest kinds in the chart, and check each fix’s wait against the 300 ms limit.</p>`,`<p class="badline" style="margin-top:12px"><b>还没达标：${why.join('；')}。</b></p><p class="small">提示：先看图里最大的几类错，再拿每种改法多等的时间和 300 毫秒的上限比一比。</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(me.met) chReveal=true;
  if(!chReveal) return h;
  /* the wait goes with the row's name and the value label is the share alone, so the label stays clear of the goal line (round-1 polish P21) */
  const rows=pts.map(p=>({label:tx(`${fxList(p.p)} (+${waitTxt(p.wait)})`,`${fxList(p.p)}（多等 ${waitTxt(p.wait)}）`),value:p.cor*100,text:pc(p.cor),ok:p.met,best:p.best,me:isMe(p),tags:isMe(p)?[{text:tx('your pick','你的选择'),me:true}]:[]}));
  rows.sort((a,b)=>(b.best-a.best)||(b.ok-a.ok)||(b.value-a.value));
  rows.push({label:tx('No fix (no extra wait)','不用改法（不多等）'),value:m0.cor*100,text:pc(m0.cor),ok:false,best:false,me:false,tags:[{text:tx('initial setup','初始设置')}]});
  h+=A.ui.cmpChart({brief:true,title:tx('Share correct for the initial setup, your pick, and every pair that waits at most 300 ms and reaches 75%','答对的比例：初始设置、你的选择，以及多等不超过 300 毫秒、答对至少 75% 的每一对改法'),aria:tx('Correct answers by pair of fixes.','按改法组合的答对比例。'),rows,goal:CH_GOAL*100,goalLabel:'75%',max:100,
    hiddenText:n=>tx(`The other ${n} pairs wait more than 300 ms or stay under 75%, and are not shown.`,`其余 ${n} 对要么多等超过 300 毫秒，要么不到 75%，图里没有列出。`),okText:tx('within 300 ms and at least 75%','不超过 300 毫秒，且至少 75%'),noText:tx('over 300 ms or under 75%','超过 300 毫秒，或不到 75%'),goalText:tx('goal: at least 75%','目标：至少 75%')});
  const v=compute('ch',['ver']), d=compute('ch',['ver','dec']), hb=compute('ch',['hyb']);
  h+=tx(`<p class="small">Why: the largest kinds are “${L(KN.stale)}” (${m0.left.stale}) and “${L(KN.multi)}” (${m0.left.multi}).</p><p class="small">The version filter repairs ${v.fixed.stale} at no wait, so it belongs in every good pair.</p><p class="small">Splitting the question would repair ${d.fixed.multi} two-passage questions, but it adds 600 ms, over the limit.</p><p class="small">Of the fixes left, hybrid search + rerank repairs the most, ${hb.fixed.miss+hb.fixed.cut+hb.fixed.multi} questions across three kinds.</p>`,
     `<p class="small">原因：最大的两类是“${L(KN.stale)}”（${m0.left.stale} 个）和“${L(KN.multi)}”（${m0.left.multi} 个）。</p><p class="small">版本过滤不多等就修好 ${v.fixed.stale} 个，好的组合里都有它。</p><p class="small">拆分问题能修好 ${d.fixed.multi} 个要两段的问题，可它要多等 600 毫秒，超过了上限。</p><p class="small">剩下的改法里，混合检索 + rerank 修得最多，三类加起来修好 ${hb.fixed.miss+hb.fixed.cut+hb.fixed.multi} 个。</p>`);
  h+=tx(`<p class="small muted">Note: the repair shares and the waits are this lab’s assumptions. On your own system, grade the same questions again after each fix ships: count what it repaired, and what it broke among the answers that were right.</p>`,`<p class="small muted">注意：修好多少、多等多久，都是这个 Lab 的假设。在你自己的系统上，每上线一种改法，都要拿同一批问题重新判分：数一数它修好了几个，又把原来答对的弄错了几个。</p>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('trace a wrong RAG answer stage by stage to the stage that failed and the kind of failure, and pick the fix that repairs the most.','把 RAG 的错答一环一环查下去，找出错在哪一环、属于哪一类错，并选出修得最多的改法。');
const EXITS=()=>({
 explain:[
  {q:tx('A RAG bot has 98% retrieval recall but only 75% correct answers. The team plans to switch to a bigger model. What is wrong with that plan?','一个 RAG 机器人的检索召回率是 98%，答案正确率只有 75%。团队打算换一个更大的模型。这个计划有什么问题？'),
   o:[tx('Nobody knows yet which stage fails. If most failures are passages cut by the budget, a bigger model never sees the right passage; if an old and a new edition clash, a bigger model cannot tell which one is current either.','还不知道错在哪一环。如果大多数错是段落被预算挤掉，模型再大也看不到对的段落；如果是新旧两版文档打架，换模型也分不出哪一版是现行的。'),
      tx('A bigger model lowers retrieval recall.','更大的模型会降低检索召回率。'),
      tx('With 98% retrieval recall the model must be at fault, so a bigger model is sure to help.','检索召回率已经 98%，错一定在模型，换大模型一定有用。')],a:0,
   n:[null,tx('Retrieval runs before the model and does not depend on it.','检索在模型之前运行，和模型无关。'),tx('Retrieved is not sent: the passage can still be cut by the budget, or an outdated page can win.','检索到不等于放进去了：段落还可能被预算挤掉，或者被旧文档带偏。')],
   why:tx('Per-stage checks split the failures; each stage has its own fix.','按环节检查才能把错分开；每一环有自己的改法。')},
  {q:tx('Why do retrieval recall, context recall and groundedness all miss a failure caused by an outdated page?','为什么旧文档带来的错，检索召回率、上下文召回率和有据可依都发现不了？'),
   o:[tx('The new edition was found and sent, and the answer is grounded, only in the old edition; only the reference answer shows that it is wrong.','新版本找到了，也放进去了；回答也有依据，只是依据是旧版本。只有和标准答案比，才看得出答错了。'),
      tx('Because old pages are never retrieved.','因为旧文档不会被检索到。'),
      tx('Because groundedness only checks how long the answer is.','因为有据可依只检查回答的长度。')],a:0,
   n:[null,tx('The old page is retrieved; that is the problem.','旧文档正是被检索到了，问题就出在这里。'),tx('Groundedness checks that every sentence is supported by a chunk in the context.','有据可依检查的是每句话能不能在上下文里找到依据。')],
   why:tx('Support by a chunk is not truth: the chunk itself can be outdated.','有段落支持不等于说得对：段落本身可能过时了。')}],
 transfer:[
  {q:tx('A legal Q&A bot: retrieval recall 92%, context recall 60%, grounded 97%. What should it change first?','一个法律问答机器人：检索召回率 92%，上下文召回率 60%，有据可依 97%。最该先改什么？'),
   o:[tx('The context budget or the ranking (rerank): passages are found but most do not reach the context','上下文预算或排序（rerank）：段落找到了，大多却没放进上下文'),tx('A prompt that asks for citations','要求引用的 prompt'),tx('A bigger model','换更大的模型')],a:0,
   why:tx('The big drop is between retrieved (92%) and sent (60%): the passages are found and then cut. Rerank moves them up; a bigger budget lets more in.','落差在检索到（92%）和放进去（60%）之间：段落找到了，又被挤掉了。rerank 把它们往前排，加大预算让更多段放进去。')},
  {q:tx('A company’s price list changes every week. Users often get last month’s price, and every such answer is grounded. What should it add?','一家公司的价目表每周都变。用户常常拿到上个月的价格，而这些回答都有据可依。该加什么？'),
   o:[tx('A version filter: replaced editions stay out of the index, or only the latest date is kept','文档版本过滤：旧版本不进索引，或者按日期只留最新的'),tx('A bigger context budget','加大上下文预算'),tx('Split the question','拆分问题')],a:0,
   why:tx('Grounded but wrong means the passage itself is outdated; only removing old editions stops the model from following them.','有依据却答错，说明段落本身过时了；只有把旧版本拿掉，模型才不会照着它答。')}],
 practice:[{id:'rag-agent-platform',d:tx('Its evaluation part: recall@k, groundedness per claim and end-to-end task success, measured separately to localise a regression.','其中评测那部分：recall@k、逐句的有据可依和端到端的任务成功率分开测，用来定位出问题的环节。')},
           {id:'llm-eval-system',d:tx('Designing the evaluation of an assistant: reference answers, judge models and what to measure at each stage.','为助手设计评测：标准答案、评分模型，以及每一环测什么。')},
           {id:'rag-search-ml',d:tx('Search and RAG design spoken out loud: retrieval stages, rerankers and how to measure them.','口述搜索和 RAG 的设计：检索分几级、rerank、怎么衡量。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Seven short parts. Follow one wrong answer through every stage, then learn to check each stage and pick fixes.','一共 7 个部分。跟着一个错答走完每一环，再学会按环节检查、选改法。'),
  sandbox:tx('Pick a set of fixes and see how the 200 graded answers of the help-center bot change.','选一组改法，看帮助中心机器人判过对错的 200 个回答怎么变。'),
  challenge:tx('Pick two fixes for the installer bot so that at least 75% of its answers are correct, with at most 300 ms more wait per question.','给安装助手选两种改法，让它至少答对 75%，每个问题最多多等 300 毫秒。')});
const RAGPIN='298b68274234c060deacab3cf5fb52aa3a20e885', TLPIN='96d858212b525450cd9dc091ab1fd9a7e6693eab', ACPIN='d7265d6ae994ccd8429db0594b000073b2f9ad43';
const GH=(repo,pin,path)=>`https://github.com/${repo}/blob/${pin}/${path}`;
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const U_FA=GH('explodinggradients/ragas',RAGPIN,'docs/concepts/metrics/available_metrics/faithfulness.md');
const U_CR=GH('explodinggradients/ragas',RAGPIN,'docs/concepts/metrics/available_metrics/context_recall.md');
const U_TL=GH('truera/trulens',TLPIN,'docs/getting_started/core_concepts/rag_triad.md');
const U_AC=GH('anthropics/anthropic-cookbook',ACPIN,'capabilities/contextual-embeddings/guide.ipynb');
const FIDPOP=()=>tx(`<p><b>Replay:</b> the seven runs are example records written for this lab.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Each run’s stages: the 6 chunks vector search returns with their similarity, the budget line after 3, the answer split into sentences with the chunk that supports each, and the reference answer.</li><li>The four checks are computed from that record: retrieval recall asks whether the needed passages were all retrieved, context recall whether they all reached the context, grounded whether every sentence of the answer has a supporting chunk, and correct whether the answer agrees with the reference.</li></ul></div><div><div class="eyebrow">Simplified (the lab’s assumptions)</div><ul><li>The 200 graded questions follow a mix of kinds this lab sets.</li><li>Each fix repairs a stated share of its kinds; two fixes on the same kind combine as independent chances (the shares left over are multiplied) and the count is rounded. A repaired question becomes correct; a real fix can also break answers that were right.</li><li>The waits per fix are assumptions, of the order the Anthropic cookbook reports for a reranker (100–200 ms). The waits of several fixes add up; the searches of a split question could run in parallel, so a real system may wait less.</li><li>Grounded counts a whole answer, not single sentences.</li></ul></div></div><p class="small"><b>Sources:</b> ${SRC(U_FA,'RAGAS faithfulness')}, ${SRC(U_CR,'RAGAS context recall')}, ${SRC(U_TL,'TruLens RAG triad')}, ${SRC(U_AC,'Anthropic contextual retrieval cookbook')}.</p>`,
  `<p><b>回放：</b>这七次运行是为这个 Lab 写的示例记录。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>每次运行的每一环：向量检索返回的 6 段和它们的相似度、第 3 段后面的预算线、拆成句子的回答（每句标出支持它的段落）和标准答案。</li><li>四项检查都从这份记录算出来：检索召回率看要用的段落是否都被取回，上下文召回率看它们是否都放进了上下文，有据可依看回答的每句话是否都有段落支持，答案正确看回答是否和标准答案一致。</li></ul></div><div><div class="eyebrow">简化了什么（这个 Lab 的假设）</div><ul><li>判过对错的 200 个问题，各类数量由本 Lab 写定。</li><li>每种改法修好它那几类错的固定比例；两种改法修同一类时，当作相互独立的机会（剩下的比例相乘），再取整。修好的问题算答对；真实的改法也可能把原来答对的弄错。</li><li>每种改法多等的时间是假设，和 Anthropic cookbook 给出的 rerank 耗时（100–200 毫秒）同一个量级。几种改法一起用时，多等的时间直接相加；拆分问题的几次检索可以并行，真实情况可能更短。</li><li>有据可依按整个回答算，不按单句算。</li></ul></div></div><p class="small"><b>来源：</b>${SRC(U_FA,'RAGAS faithfulness')}，${SRC(U_CR,'RAGAS context recall')}，${SRC(U_TL,'TruLens RAG triad')}，${SRC(U_AC,'Anthropic contextual retrieval cookbook')}。</p>`);
const PRIPOP=()=>tx(`<p>Retrieval-augmented generation (RAG) searches a document collection with the question and puts the passages found into the model’s prompt. Search compares embeddings (${LL('embeddings')}); the documents are cut into chunks and the best ones kept (${LL('chunk-rerank')}); the prompt has a limited budget (${LL('context-eng')}).</p>`,
                    `<p>检索增强生成（RAG）先拿问题去搜文档库，再把搜到的段落放进模型的 prompt。检索比较的是两段文字的向量（${LL('embeddings')}）；文档先切成块，只留最好的几块（${LL('chunk-rerank')}）；prompt 的预算有限（${LL('context-eng')}）。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'apps',outcome:esc(OUTCOME()),short:tx('Why RAG fails','RAG 为什么答错'),title:tx('Why does your RAG system answer wrong?','你的 RAG 为什么答错？'),
  minutes:15,fid:'Replay',fidPop:FIDPOP(),primer:tx('Primer: RAG','前置知识：RAG'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
/* the stage, top to bottom: the picture that changes with the reader's choices first (R7.12; round-1 polish P18): part 4's two numbers,
   the failure breakdown, the four checks; then the run record, which in Scenarios waits closed under the cards until the reader opens it */
/* phones: the scenarios (in Settings, the fixes) as a row of buttons right under the cards, so a tap and the breakdown and the cards it
   changes share the screen (round-1 polish P18 on phones; R3.2, R7.12). The scenario cards in the rail and the Settings drawer stay, with
   their descriptions; which row shows follows the step the reader is on (the bench's data-flow) */
const pickHTML=()=>`<div class="rgf-pk-g rgf-pk-sc" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}"><span class="rgf-pk-l">${tx('Pick a scenario','选一个应用场景')}</span>${PRESETS.map((p,i)=>`<button type="button" class="rgf-pkb" data-pick="${i}" aria-pressed="false">${esc(L(p.t))}</button>`).join('')}</div>`+
  `<div class="rgf-pk-g rgf-pk-fx" role="group" aria-label="${esc(tx('Fixes, several at once if you like','改法，可以同时开几个'))}"><span class="rgf-pk-l">${tx('Fixes on','打开的改法')}</span>${FIXES.map(f=>`<button type="button" class="rgf-pkb" data-pfx="${f}" aria-pressed="false">${esc(L(FXN[f]))}</button>`).join('')}</div>`;
function stageHTML(){
  return `<div class="panel stage rgf-stage" id="rgf-stage">`+
   `<div class="stage-head rgf-head"><div class="rgf-st-title" id="rgf-st-title"></div><div class="stage-actions"><span class="status" id="rgf-status" tabindex="0"></span></div></div>`+
   `<section class="rgf-pan" id="rgf-pan-sum"><div class="rgf-two" id="rgf-two"></div></section>`+
   `<section class="rgf-pan" id="rgf-pan-bars"><div class="rgf-pt" id="rgf-b-title"></div><div class="rgf-bars" id="rgf-bars"></div><div class="legend-row rgf-leg" id="rgf-b-leg"></div></section>`+
   `<div class="metrics rgf-metrics" id="rgf-metrics"></div>`+
   `<section class="rgf-pan rgf-pk" id="rgf-pk" hidden>${pickHTML()}</section>`+
   `<section class="rgf-pan" id="rgf-pan-run"><button type="button" class="rgf-runtog" id="rgf-runtog" data-act="runtog" aria-expanded="false" aria-controls="rgf-runbody"></button>`+
   `<div class="rgf-runbody" id="rgf-runbody"><div class="rgf-ctl"><div class="rgf-qrow"><label class="rgf-ql" for="rgf-rsel">${tx('Run','运行')}</label><select id="rgf-rsel" class="rgf-rsel"></select></div><div class="rgf-transport" id="rgf-transport"></div></div>`+
   `<p class="rgf-rnote small muted" id="rgf-rnote" hidden></p><div class="rgf-flowbox"><div class="rgf-flow" id="rgf-flow"></div><button type="button" class="btn rgf-go" id="rgf-go" data-act="go" hidden>${tx('▶ Play this run','▶ 播放这次运行')}</button></div><div class="legend-row rgf-leg" id="rgf-r-leg"></div></div></section>`+
   `<details class="math" id="rgf-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="rgf-mathbody"></div></details></div>`;}
function controlsHTML(){
  const seg=f=>`<div class="lockable">`+A.ui.seg({id:'rgf-ctl-'+f,key:'fx-'+f,label:L(FXN[f]),options:[['0',tx('Off','不用')],['1',tx('On','用')]],value:S.fx.includes(f)?'1':'0',note:FXLINES(f,true)})+`</div>`;
  return `<aside class="panel controls" id="rgf-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Turn on the fixes you want, several at once if you like, and watch the failure breakdown and the four checks.','打开你想用的改法，可以同时开几个，看看答错的分类和四项检查怎么变。')}</p><p class="ctl-lock" id="rgf-lock" hidden></p>`+
   FIXES.map(seg).join('')+
   `<details class="adv"><summary>${tx('Assumed in this lab: the bot and its 200 questions','这个 Lab 假设的：机器人和它的 200 个问题')}</summary><p class="ctl-note">${tx(`Vector search returns ${NRET} chunks and the context budget holds ${BUDGET} (about 300 tokens each). The 200 graded questions are a stated mix: ${MIX.base.ok} correct, then ${KINDS.map(k=>`${MIX.base[k]} ${L(KN[k]).toLowerCase()}`).join(', ')}. The lab assumes the ${MIX.base.ok} correct answers pass all four checks. The passages of a “Not retrieved” question all rank below 10th. When several fixes are on, their waits add up (an assumption: the searches of a split question could run in parallel). Real numbers come from grading your own questions.`,`向量检索取回 ${NRET} 段，上下文预算放得下 ${BUDGET} 段（每段约 300 个 token）。判过对错的 200 个问题是写定的：答对 ${MIX.base.ok} 个，${KINDS.map(k=>`“${L(KN[k])}”${MIX.base[k]} 个`).join('、')}。这个 Lab 假设答对的 ${MIX.base.ok} 个，四项检查都通过。“检索没找到”的问题，要用的段落都排在第 10 名以后。几种改法一起用时，多等的时间直接相加（这是假设；拆分问题的几次检索可以并行，真实情况可能更短）。真实的数要拿你自己的问题判对错才知道。`)}</p></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`def triage(run, needed, supported):
    """run: ranked chunk ids retrieved, the ids sent,
    the answer's sentences and the reference verdict.
    needed: ids the reference answer needs ([] if none).
    supported(sentence, ids): a judge model's verdict."""
    got, sent = run["retrieved"], run["sent"]
    checks = {
        "retrieval": all(c in got for c in needed),
        "context": all(c in sent for c in needed),
        "grounded": all(supported(s, sent)
                        for s in run["sentences"]),
        "correct": run["correct"],
    }
    if checks["correct"]:
        return "ok", checks
    if not needed:
        return "no answer in the docs", checks
    if not checks["retrieval"]:
        found = sum(c in got for c in needed)
        return ("needs two passages" if found
                else "not retrieved"), checks
    if not checks["context"]:
        return "cut by the budget", checks
    if not checks["grounded"]:
        return "not from the context", checks
    return "outdated page? check versions", checks

def rates(results):
    n = len(results)
    ans = [c for _, c, need in results if need]
    share = lambda xs, k: sum(x[k] for x in xs) / len(xs)
    return {"retrieval recall": share(ans, "retrieval"),
            "context recall": share(ans, "context"),
            "grounded": sum(c["grounded"] for _, c, _ in results) / n,
            "correct": sum(c["correct"] for _, c, _ in results) / n}`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="rgf-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="rgf-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Per-question triage from a log, and the four rates, in Python.','用 Python 写的：从日志给每个问题分类，再算四个比例。')}</p><pre class="code" id="rgf-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="rgf-prod">`+
   tx(`<li>RAGAS ${SRC(U_FA,'faithfulness')}: “A response is considered faithful if all its claims can be supported by the retrieved context”; the score is the share of supported claims. Its ${SRC(U_CR,'context recall')} is the share of the reference answer’s claims supported by the retrieved context.</li>`+
      `<li>TruLens ${SRC(U_TL,'RAG triad')}: three checks, context relevance, groundedness and answer relevance, one on each edge of the RAG pipeline.</li>`+
      `<li>Anthropic’s ${SRC(U_AC,'contextual retrieval cookbook')}: 248 test queries, each with a “golden chunk”; Pass@k checks whether it is among the first k chunks retrieved. Adding contextual embeddings, hybrid search (BM25) and reranking one after another cuts the Pass@10 failure rate from 12.85% to 4.74%; reranking adds 100–200 ms per query.</li>`+
      `<li>Every retrieved chunk is text written by whoever can edit the documents, and it may hold instructions (“ignore the rules above…”). The model should treat chunks as material, not as commands; see ${LL('prompt-injection')}.</li>`,
      `<li>RAGAS ${SRC(U_FA,'faithfulness')}：回答里的每个说法都能由检索到的上下文支持，这个回答才算 faithful（原文 “A response is considered faithful if all its claims can be supported by the retrieved context”）；分数是有支持的说法占的比例。它的 ${SRC(U_CR,'context recall')} 是标准答案里的说法有多少能在检索到的上下文里找到支持。</li>`+
      `<li>TruLens 的 ${SRC(U_TL,'RAG triad')} 有三项检查，各查 RAG 流程上的一段：上下文相关性、有据可依（groundedness）和回答相关性。</li>`+
      `<li>Anthropic 的 ${SRC(U_AC,'contextual retrieval cookbook')}：248 个测试问题，每个配一个“golden chunk”；Pass@k 检查它在不在检索到的前 k 段里。依次加上 contextual embeddings、混合检索（BM25）和 rerank，前 10 段的 Pass@10 失败率从 12.85% 降到 4.74%；rerank 让每次查询多花 100–200 毫秒。</li>`+
      `<li>检索回来的每一段，都是能改文档的人写的文字，里面可能夹着指令（“忽略上面的规则……”）。模型应该把段落当资料，不当命令，${SEE('prompt-injection')}。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: ragas 298b682, trulens 96d8582, anthropic-cookbook d7265d6.','固定的 commit：ragas 298b682，trulens 96d8582，anthropic-cookbook d7265d6。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</a><span>${tx('Lewis et al., 2020: RAG','Lewis 等，2020：RAG')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2309.15217" target="_blank" rel="noopener">RAGAS: Automated Evaluation of Retrieval Augmented Generation</a><span>${tx('Es et al., 2023: reference-free RAG metrics','Es 等，2023：不需要标准答案的 RAG 指标')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2401.05856" target="_blank" rel="noopener">Seven Failure Points When Engineering a Retrieval Augmented Generation System</a><span>${tx('Barnett et al., 2024: a list of where RAG systems fail','Barnett 等，2024：RAG 系统在哪些地方出错')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2307.03172" target="_blank" rel="noopener">Lost in the Middle: How Language Models Use Long Contexts</a><span>${tx('Liu et al., 2023: where a passage sits in the context matters','Liu 等，2023：段落在上下文里的位置有影响')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="rgf-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="rgf-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="rgf-app m2" id="rgf-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="rgf-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels the stage shows now */
function ctx(){
  if(mode==='guided') return STEPS[step].panels;
  if(mode==='challenge') return ['bars','metrics'];
  return ['bars','metrics','run'];}
const runShown=()=>ctx().includes('run')&&(mode!=='sandbox'||runOpen);

/* ---------- drawing ---------- */
const triHidden=()=>mode==='guided'&&STEPS[step].id==='triage'&&!tri.picks[tri.i];
const logOff=()=>mode==='guided'&&STEPS[step].id==='log'&&S.log===0;
const early=()=>mode==='guided'&&step<2;   // parts 1 and 2 come before the six kinds are named: their verdict names the stage only (round-1 polish P22)
const CIRC='①②③④⑤', stgTxt=s=>CIRC[STG[s].n-1]+' '+L(STG[s]);
/* the outdated page fails at retrieval although retrieval recall passes: the current edition was retrieved, and so was the replaced one,
   ranked first. The verdict says so right after the stage, and the ✓ says why it passes (N1 of the round-1 check) */
const STGWHY={stale:{en:' (it returned the replaced edition too)',zh:'（取回了已被替换的旧版）'}};
const stgWhy=k=>STGWHY[k]?L(STGWHY[k]):'';
const PASSWHY={stale:{ret:{en:'The current edition it needs was retrieved, so retrieval recall passes; it cannot tell that the replaced edition came back too.',zh:'要用的新版取回来了，所以检索召回率算通过；它查不出旧版也被取回来了。'}}};
const passWhy=(kind,k)=>PASSWHY[kind]&&PASSWHY[kind][k]?tx(' ','')+L(PASSWHY[kind][k]):'';
/* the check that a set of fixes moves the most, in shown percentages, against no fix: [check, before, after], or null if none moves */
const bigChange=(m,m0)=>{let best=null; ['ret','ctx','grd'].forEach(k=>{const a=Math.round(m0[k]*100+1e-9), b=Math.round(m[k]*100+1e-9); if(a!==b&&(!best||Math.abs(b-a)>Math.abs(best[2]-best[1]))) best=[k,a,b];}); return best;};
/* the replay's beats, in seconds at 中 (scaled with the speed; round-1 polish P3): the question; the 6 chunks one by one, all alike; a needed
   passage the search never returned; the budget line drops under the 3rd chunk and the rest turn grey; the 3 chunks above the line fly into
   the context; the answer, sentence by sentence with its source; the reference answer; the four checks one by one; then the stage that
   failed turns red with the verdict. Everything is drawn from the start and only shown at its beat, so nothing moves (R7.9) */
const BT={row0:.6,row:.3,miss:2.4,bl:2.9,fly:3.7,flyLen:.8,ans:5.3,ansStep:.6,truth:6.6,chk:7.4,chkStep:.3,verd:9.1,end:10};
/* with only the question and the answer logged, retrieval and the context have nothing to show: both say “not logged” on one beat, the
   answer follows at once, and the replay ends in about 5 s at 中 (N8 of the round-1 check) */
const BT_OFF={row0:.6,row:.3,miss:.6,bl:.6,fly:.6,flyLen:.01,ans:1.5,ansStep:.6,truth:2.4,chk:3,chkStep:.3,verd:4.2,end:5};
const beats=()=>logOff()?BT_OFF:BT;
/* pre: from just before the chips fly until they land, the rows under the budget line and its note step back, so the flying chips are
   not read as those rows moving (N10 of the round-1 check) */
function phase(p,nAns){const B=beats(), t=p*B.end, cnt=(t0,dt,max)=>t<t0?0:Math.min(max,Math.floor((t-t0)/dt+1e-9)+1);
  return {n:cnt(B.row0,B.row,NRET),miss:t>=B.miss,bl:t>=B.bl,fly:t<B.fly?0:Math.min(1,(t-B.fly)/B.flyLen),ctx:t>=B.fly+B.flyLen,
    ans:cnt(B.ans,B.ansStep,nAns),truth:t>=B.truth,nc:cnt(B.chk,B.chkStep,4),verd:t>=B.verd,pre:B===BT&&t>=B.fly-.3&&t<B.fly+B.flyLen};}
const runAnim=()=>anim&&anim.key===RUNS[runNow()].id?anim:null;
function drawRunNote(){   // the record is the initial setup's run; with fixes on, say what they do to runs of this kind
  const el=$('#rgf-rnote',app), r=RUNS[runNow()], fx=fxNow().filter(f=>SHARE[f][r.kind]); el.hidden=!fxNow().length||r.kind==='ok';
  if(el.hidden){el.textContent=''; return;}
  const left=fx.reduce((a,f)=>a*(1-SHARE[f][r.kind]),1), kn=L(KN[r.kind]);
  el.textContent=fx.length?tx(`This record is from the initial setup. With the fixes on (${fxList(fx)}), about ${pc(1-left)} of the questions of this kind, “${kn}”, are repaired.`,`这份记录来自初始设置。打开的改法里，${zhList(fx.map(f=>`“${L(FXN[f])}”`))}${fx.length>1?'合起来':''}能修好这一类问题（“${kn}”）的约 ${pc(1-left)}。`)
    :tx(`This record is from the initial setup. None of the fixes on touches its kind, “${kn}”.`,`这份记录来自初始设置。打开的改法都管不到这一类（“${kn}”）。`);}
/* the record of one run, every part drawn once (data-b, data-cl, data-pk: the beat that shows it; data-bad: red once the verdict shows) */
function runHTML(r,hide,off,ear){
  const c=checks(r), g=r.gold.map(x=>x[0]), sent=r.ret.slice(0,BUDGET).map(x=>x[0]), cause=CAUSE[r.kind], judge=!hide&&!off, BAD=on=>on?' data-bad="1"':'';
  const stg=(s,num,label,inner,cls)=>`<div class="rgf-stg${cls?' '+cls:''}" data-s="${s}"${BAD(judge&&cause===s)}><div class="rgf-sl"><span class="rgf-sn">${num}</span>${label}</div><div class="rgf-sb">${inner}</div></div>`;
  let h=stg('q',1,tx('Question','问题'),`<p class="rgf-qt">${quote(r.q[li()])}</p>`);
  if(off){const nl=b=>`<p class="rgf-nolog rgf-w" data-b="${b}">${tx('Not logged','没有记录')}</p>`;
    h+=stg('ret',2,tx(`Retrieved: ${NRET} chunks`,`检索：取回 ${NRET} 段`),nl('n1'),'nolog')+stg('ctx',3,tx(`Sent: ${BUDGET} chunks`,`放进上下文：${BUDGET} 段`),nl('ctx'),'nolog');}
  else{
    const rows=r.ret.map((x,i)=>{const id=x[0], gold=g.includes(id), old=CH[id][2]==='old';
      return `<li data-i="${i}"${i>=BUDGET&&!gold&&!old?` data-tipx="${esc(CH[id][1][li()])}"`:''} class="rgf-li rgf-w${gold?' gold':''}"${BAD(judge&&((r.kind==='cut'&&gold)||(r.kind==='stale'&&old)))}><span class="rgf-rk">${i+1}</span><span class="rgf-nm"><b>${gold?'★ ':''}${chunkName(id)}</b>${old?`<span class="rgf-tag" data-tipx="${esc(tx('A note in the document index; the chunk sent to the model carries no such mark.','文档库里的记录；发给模型的段落里没有这个标记。'))}">${tx('replaced','已被替换')}</span>`:''}${i<BUDGET||gold||old?` <span class="rgf-tx">${esc(CH[id][1][li()])}</span>`:''}</span><span class="rgf-sc">${A.fmt.num(x[1],2)}</span></li>`;}).join('');
    const missH=r.gold.filter(x=>!r.ret.some(y=>y[0]===x[0])).map(x=>`<p class="rgf-miss rgf-w" data-b="miss"${BAD(judge)}>${tx(`★ ${chunkName(x[0])}: ranked ${ord(x[1])}, not retrieved`,`★ ${chunkName(x[0])}：排第 ${x[1]}，没取回来`)}</p>`).join('');
    h+=stg('ret',2,tx(`Retrieved: the ${NRET} most similar chunks`,`检索：取回最像的 ${NRET} 段`),`<div class="rgf-lh"><span>${tx('Rank and chunk','排名和段落')}</span><span class="rgf-lhs" data-tipx="${esc(tx(`How similar the chunk is to the question: the cosine similarity of their two vectors (1 means the same direction). Retrieval keeps the ${NRET} highest.`,`这一段和问题的相似度：两段文字的向量的余弦相似度，1 是同一个方向。检索按它从高到低取前 ${NRET} 段。`))}">${tx('Similarity','相似度')}</span></div><ol class="rgf-ol">${rows}</ol><p class="rgf-bl rgf-w" data-b="bl">${tx(`Budget line: only the first ${BUDGET} fit in the context`,`预算线：只有前 ${BUDGET} 段放得进上下文`)}</p>${missH}`);
    const chips=sent.map((id,i)=>`<span class="rgf-chip rgf-w${g.includes(id)?' gold':''}" data-c="${i}">${g.includes(id)?'★ ':''}${chunkName(id)}</span>`).join('');
    const cutH=r.kind==='cut'&&judge?`<p class="rgf-miss rgf-w" data-b="ctx"${BAD(judge)}>${tx(`★ ${chunkName(g[0])} ranked ${ord(r.gold[0][1])}: below the line, not sent`,`★ ${chunkName(g[0])} 排第 ${r.gold[0][1]}：在预算线下面，没放进来`)}</p>`:'';
    h+=stg('ctx',3,tx(`Sent to the model: ${BUDGET} chunks`,`放进上下文：${BUDGET} 段`),`<p class="rgf-chips">${chips}</p>${cutH}`);}
  const claims=r.ans.map((x,j)=>{const ok=!!(x[2]&&sent.includes(x[2]));   // each sentence comes with its source (hidden while the reader still has to judge)
    return `<span class="rgf-cl rgf-w" data-cl="${j}"${BAD(judge&&!ok)}>${esc(x[li()])}${judge?`<sup class="rgf-src">${ok?esc(tx(`from ${CH[x[2]][0][0]}`,`出自${CH[x[2]][0][1]}`)):tx('no source in the context','上下文里没有依据')}</sup>`:''}</span>`;}).join(' ');
  h+=stg('ans',4,tx('Answer','回答'),`<p class="rgf-at">${claims}</p><p class="rgf-truth rgf-w" data-b="truth"><b>${tx('Reference answer: ','标准答案：')}</b>${esc(r.truth[li()])}</p>`);
  const ck=CHK.map((k,i)=>{const v=c[k], known=!(hide||(off&&k!=='cor')), mark=!known?'?':(v===null?'—':(v?'✓':'✗'));
    return `<span class="rgf-pill wait" data-pk="${i}" data-m="${mark}"${known&&v===false?' data-no="1"':''}${known&&v===true?' data-yes="1"':''}${BAD(judge&&CAUSE_CHK[r.kind]===k)} data-tipx="${esc(CKD[k]()+(known&&v===true?passWhy(r.kind,k):''))}"><b>·</b>${esc(L(CKS[k]))}</span>`;}).join('');
  /* the verdict names the stage of the picture, then the kind (round-1 polish P5) */
  const vd=hide?tx('Which kind of failure is it? Pick it in the card.','这是哪一类错？在卡片里选。')
    :off?tx('All you can tell is that the answer is wrong, not which stage failed.','只看得出答错了，看不出错在哪一环。')
    :r.kind==='ok'?tx('Every check passes.','四项检查都通过。')
    :tx(`Failed at <b>${esc(stgTxt(cause))}</b>`,`错在 <b>${esc(stgTxt(cause))}</b>`)+esc(stgWhy(r.kind))+(ear?'':tx(`; kind: <b>${esc(L(KN[r.kind]))}</b>`,`，类别：<b>${esc(L(KN[r.kind]))}</b>`));
  h+=stg('chk',5,tx('Checks','判分'),`<div class="rgf-pills">${ck}</div><p class="rgf-verdict rgf-w${hide||off?' nt':(r.kind==='ok'?' ok':'')}" data-b="verd">${vd}</p>`);
  return h;}
function drawRun(){
  drawRunNote();
  const ri=runNow(), r=RUNS[ri], el=$('#rgf-flow',app), key=[A.lang(),ri,triHidden(),logOff(),early()].join('|');
  if(el._k!==key){el._k=key; el._ph=null; el._fly=null; el.innerHTML=runHTML(r,triHidden(),logOff(),early());}
  runPhase(); drawRunLeg(); drawGo();}
/* show what the replay has reached: class changes only, plus the chips' flight */
function runPhase(){
  const el=$('#rgf-flow',app); if(!el||!el.firstChild) return;
  const r=RUNS[runNow()], a=runAnim(), ph=phase(a?a.p:1,r.ans.length), sig=[ph.n,ph.miss,ph.bl,ph.ctx,ph.ans,ph.truth,ph.nc,ph.verd,ph.pre].join('|');
  if(el._ph!==sig){el._ph=sig; const vis=(e,on)=>e.classList.toggle('rgf-w',!on); el.classList.toggle('rgf-flying',ph.pre);
    $$('.rgf-li',el).forEach(e=>{const i=+e.dataset.i; vis(e,i<ph.n); e.classList.toggle('below',ph.bl&&i>=BUDGET); e.classList.toggle('cutk',ph.bl&&i===BUDGET-1);});
    $$('[data-b]',el).forEach(e=>{const b=e.dataset.b; vis(e,b==='miss'?ph.miss:b==='bl'?ph.bl:b==='n1'?ph.n>0:b==='ctx'?ph.ctx:b==='truth'?ph.truth:ph.verd);});
    $$('[data-cl]',el).forEach(e=>vis(e,+e.dataset.cl<ph.ans));
    $$('.rgf-pill',el).forEach(e=>{const on=+e.dataset.pk<ph.nc; e.classList.toggle('wait',!on); e.classList.toggle('yes',on&&!!e.dataset.yes); e.classList.toggle('no',on&&!!e.dataset.no); e.firstChild.textContent=on?e.dataset.m:'·';});
    $$('[data-bad]',el).forEach(e=>e.classList.toggle('bad',ph.verd));
    if(a) A.dock.render();}
  flyChips(el,ph);}
/* the 3 chunks above the budget line move from their rows into the context (round-1 polish P3) */
function flyChips(el,ph){const chips=$$('.rgf-chip',el); if(!chips.length) return;
  if(ph.fly<=0||ph.fly>=1){el._fly=null; chips.forEach(c=>{c.classList.toggle('rgf-w',!ph.ctx); c.classList.remove('rgf-fly'); c.style.transform='';}); return;}
  if(!el._fly){chips.forEach(c=>{c.style.transform='';});
    el._fly=chips.map(c=>{const s=$(`.rgf-li[data-i="${c.dataset.c}"] .rgf-nm b`,el); if(!s) return [0,0]; const a=s.getBoundingClientRect(), b=c.getBoundingClientRect(); return [a.left-b.left,a.top-b.top];});}
  const f=ph.fly, e=f<.5?2*f*f:1-Math.pow(-2*f+2,2)/2;
  chips.forEach((c,i)=>{const d=el._fly[i]; c.classList.remove('rgf-w'); c.classList.add('rgf-fly'); c.style.transform=`translate(${(d[0]*(1-e)).toFixed(1)}px,${(d[1]*(1-e)).toFixed(1)}px)`;});}
function drawGo(){const b=$('#rgf-go',app); if(b) b.hidden=!(anim&&anim.pending&&runAnim());}
function drawRunLeg(){const el=$('#rgf-r-leg',app), ng=!RUNS[runNow()].gold.length, k=A.lang()+'|'+ng; if(el._k===k) return; el._k=k;
  el.innerHTML=`<span class="lg" data-tipx="${esc(tx('A passage the reference answer needs.','标准答案要用到的段落。'))}"><b class="rgf-x">★</b>${tx('needed passage','要用的段落')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('Chunks below the line do not fit in the context budget and never reach the model.','预算线下面的段落放不进上下文，模型看不到。'))}"><i class="sw rgf-sw-bl"></i>${tx('budget line','预算线')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('Grey chunks were retrieved but not sent.','灰色的段落检索到了，但没放进上下文。'))}"><i class="sw rgf-sw-below"></i>${tx('not sent','没放进去')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('The stage where this run failed, and what went wrong there.','这次运行出错的那一环，和那里错了什么。'))}"><i class="sw rgf-sw-bad"></i>${tx('the stage that failed','出错的那一环')}</span>`+
   (ng?`<span class="lg" data-tipx="${esc(tx('This question has no passage to find, so the two recall checks do not count it.','这个问题没有要找的段落，两项召回率不算它。'))}"><b class="rgf-x">—</b>${tx('not counted','不算')}</span>`:'');}
/* part 4: the two numbers the prediction is about, and nothing that gives the answer away (R3.8; round-1 polish P11) */
function drawSum(){const el=$('#rgf-two',app), k=A.lang(); if(el._k===k) return; el._k=k; const m=compute('base',[]);
  const row=(c,v,sub,cls)=>`<div class="rgf-tw"><div class="rgf-tw-h">${DEF(esc(L(CKN[c])),CKD[c]())}<b class="mono">${pc(v)}</b></div><div class="rgf-tw-t" role="img" aria-label="${esc(L(CKN[c])+' '+pc(v))}"><i class="${cls}" style="width:${(v*100).toFixed(1)}%"></i></div><div class="rgf-tw-s">${esc(sub)}</div></div>`;
  el.innerHTML=row('cor',m.cor,tx(`${m.ok} of the ${m.N} answers are correct`,`${m.N} 个回答里答对 ${m.ok} 个`),'ok')+
    row('ret',m.ret,tx(`for ${m.nRet} of the ${m.ans} questions the documents can answer, every needed passage was retrieved`,`文档里有答案的 ${m.ans} 个问题里，${m.nRet} 个检索到了要用的全部段落`),'rc');}
function drawBars(){
  const m=compute(sys(),fxNow()), mix=MIX[sys()], el=$('#rgf-bars',app), met=mode==='guided'&&STEPS[step].id==='metrics'?S.met:null, mi=met?CHK.indexOf(met):-1;
  const max=Math.max(...KINDS.map(k=>mix[k])), cur=runShown()?RUNS[runNow()].kind:null;
  $('#rgf-b-title',app).textContent=tx(`Where the ${m.wrong} wrong answers fail`,`答错的 ${m.wrong} 个，错在哪里`);   // the stage title already names the bot (round-1 polish P24)
  const okW=mix.ok/m.N*100, fxW=(m.ok-mix.ok)/m.N*100;
  let h=`<div class="rgf-sum" role="img" aria-label="${esc(tx(`${m.ok} of ${m.N} correct`,`${m.N} 个里答对 ${m.ok} 个`))}"><span class="rgf-sum-l">${tx('Correct','答对')} <b>${m.ok}</b> / ${m.N} (${pc(m.cor)})</span><div class="rgf-sum-t"><i class="ok" style="width:${okW}%"></i><i class="fx" style="width:${fxW}%"></i></div></div>`;
  h+=`<div class="rgf-tb${mi>=0?' focus':''}" role="table" aria-label="${esc(tx('Wrong answers by kind, and the checks that catch each kind','答错的按类别，以及能发现每一类的检查'))}"><div class="rgf-tr rgf-th" role="row"><span role="columnheader">${tx('Kind of failure','哪一类')}</span><span role="columnheader">${tx('Questions','问题数')}</span>${CHK.map((k,i)=>`<span role="columnheader" class="rgf-cc${i===mi?' on':''}" data-tipx="${esc(CKD[k]())}">${esc(L(CKS[k]))}</span>`).join('')}</div>`;
  KINDS.forEach(k=>{const n0=mix[k], left=m.left[k], fx=m.fixed[k], on=mi<0||FLAG[k][mi]===1;
    h+=`<div class="rgf-tr${on?'':' dim'}${cur===k?' cur':''}" role="row"><span class="rgf-kn" role="rowheader" data-tipx="${esc(L(KN[k])+tx(': ','：')+L(KD[k]))}">${esc(L(KN[k]))}</span>`+
      `<span class="rgf-bc" role="cell"><span class="rgf-bt"><i class="lf" style="width:${left/max*100}%"></i><i class="fx" style="width:${fx/max*100}%"></i></span><span class="rgf-bn">${fx?tx(`${left} (was ${n0})`,`${left}（原 ${n0}）`):left}</span></span>`+
      CHK.map((c,i)=>{const f=FLAG[k][i]; return `<span role="cell" class="rgf-cc${i===mi?' on':''}">${f===1?'✗':(f===null?'—':'')}</span>`;}).join('')+`</div>`;});
  h+=`</div>`;
  el.innerHTML=h;
  /* the legend names only what the chart shows: “repaired” only with a fix on (round-1 polish P23), and the dash of the recall columns */
  const lg=$('#rgf-b-leg',app), anyFx=fxNow().length>0, lk=A.lang()+'|'+anyFx; if(lg._k!==lk){lg._k=lk;
    lg.innerHTML=`<span class="lg"><i class="sw rgf-sw-ok"></i>${tx('correct','答对')}</span>`+(anyFx?`<span class="lg"><i class="sw rgf-sw-fx"></i>${tx('repaired by the fixes','被改法修好')}</span>`:'')+
     `<span class="lg"><i class="sw rgf-sw-lf"></i>${anyFx?tx('still wrong','仍然答错'):tx('wrong','答错')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('✗ in a check’s column: questions of this kind fail that check, so it catches them.','某项检查那一列里的 ✗：这一类问题过不了这项检查，所以它能发现这一类。'))}"><b class="rgf-x">✗</b>${tx('the check catches it','这项检查能发现')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('These questions have no passage to find, so the two recall checks do not count them.','这一类问题没有要找的段落，两项召回率不算它们。'))}"><b class="rgf-x">—</b>${tx('not counted by recall','召回率不算')}</span>`;}}
const MET=[
 {k:'cor',label:()=>L(CKN.cor),v:m=>pc(m.cor),num:m=>Math.round(m.cor*100),sub:m=>tx(`${m.ok} of ${m.N} questions`,`${m.N} 个里 ${m.ok} 个`),tip:()=>CKD.cor()},
 {k:'ret',label:()=>L(CKN.ret),v:m=>pc(m.ret),num:m=>Math.round(m.ret*100),sub:m=>tx(`${m.nRet} of ${m.ans} answerable`,`${m.ans} 个有答案的里 ${m.nRet} 个`),tip:()=>CKD.ret()},
 {k:'ctx',label:()=>L(CKN.ctx),v:m=>pc(m.ctx),num:m=>Math.round(m.ctx*100),sub:m=>tx(`${m.nCtx} of ${m.ans} answerable`,`${m.ans} 个有答案的里 ${m.nCtx} 个`),tip:()=>CKD.ctx()},
 {k:'grd',label:()=>L(CKN.grd),v:m=>pc(m.grd),num:m=>Math.round(m.grd*100),sub:m=>tx(`${m.nGrd} of ${m.N} answers`,`${m.N} 个回答里 ${m.nGrd} 个`),tip:()=>CKD.grd()}];
let metKey=null;
function drawMetrics(){
  const m=compute(sys(),fxNow()), host=$('#rgf-metrics',app), key=sys()+fxNow().slice().sort().join(','), cfgCh=metKey!=null&&metKey!==key; metKey=key;
  const id=mode==='guided'?STEPS[step].id:'', m0=id==='fixes'?compute('base',[]):null;   // part 6 compares with no fix, as its observation does (round-1 polish P16)
  host.classList.toggle('rsv',mode==='sandbox'||id==='fixes');   // where a change badge can show, its two lines are kept, so the cards keep their height
  if(!host.children.length||host._lang!==A.lang()){host.innerHTML=MET.map(x=>`<div class="metric" data-k="${x.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=A.lang();}
  MET.forEach(x=>{const el=host.querySelector(`[data-k="${x.k}"]`), d=el.querySelector('.d'), num=x.num(m), prev=m0?x.num(m0):shownPrev[x.k];
    el.querySelector('.k').textContent=x.label(); el.querySelector('.v').textContent=x.v(m); el.querySelector('.sub').textContent=x.sub(m); el.dataset.tipx=x.tip();
    el.classList.toggle('on',id==='metrics'&&S.met===x.k);   // the card of the check picked in part 5, as its column (P23)
    let dt=''; if(userAct&&cfgCh&&prev!=null&&prev!==num) dt=A.fmt.delta(prev,num,{pts:true,dp:0,base:m0?{en:'no fix',zh:'不改'}:{en:'your last pick',zh:'你上一个选择'}});
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else {d.classList.remove('on'); d.textContent='';}
    shownPrev[x.k]=num;});}
/* the run's badge: a neutral word while the run plays, the result once the verdict shows (round-1 polish P4, P22) */
function runBadge(){const r=RUNS[runNow()], a=runAnim();
  if(a&&!phase(a.p,r.ans.length).verd) return {cls:'status neutral',text:a.pending?tx('Not played','待播放'):(a.paused?tx('Paused','已暂停'):tx('Playing','播放中')),tip:tx('The run plays stage by stage; the verdict shows at the end.','这次运行一环一环地播放，结论在最后出现。')};
  if(triHidden()) return {cls:'status neutral',text:tx('Your call','你来判断'),tip:tx('Which kind of failure is it? Pick it in the card.','这是哪一类错？在卡片里选。')};
  if(r.kind==='ok') return {cls:'status ok',text:tx('Correct','答对'),tip:tx('This run passes every check.','这次运行四项检查都通过。')};
  if(logOff()||early()) return {cls:'status bad',text:tx('Wrong','答错'),tip:tx('The answer does not agree with the reference answer.','回答和标准答案不一致。')};
  return {cls:'status bad',text:L(KN[r.kind]),tip:tx('The kind of failure of this run: ','这次运行的错误类别：')+L(KD[r.kind])};}
/* a label that changes keeps one width, that of the widest text it can show, so the head and the transport row never move (P4) */
function fitW(el,all){const k=A.lang(); if(!el||el._fit===k||!el.getClientRects().length) return;
  const keep=[el.className,el.textContent]; el.style.minWidth=''; let w=0; all.forEach(t=>{el.textContent=t; w=Math.max(w,el.getBoundingClientRect().width);});
  el.className=keep[0]; el.textContent=keep[1]; if(w>0){el.style.minWidth=Math.ceil(w)+'px'; el._fit=k;}}
const badgeFit=el=>fitW(el,[tx('Not played','待播放'),tx('Paused','已暂停'),tx('Playing','播放中'),tx('Your call','你来判断'),tx('Correct','答对'),tx('Wrong','答错'),tx('100% correct','答对 100%')].concat(KINDS.map(x=>L(KN[x]))));
const PLAY=()=>[tx('▶ Play','▶ 播放'),tx('❚❚ Pause','❚❚ 暂停')];
function drawHead(){
  const st=$('#rgf-st-title',app), el=$('#rgf-status',app), m=compute(sys(),fxNow());
  if(mode==='challenge'){st.textContent=tx('Installer bot (example): 200 graded questions','安装助手（示例）：判过对错的 200 个问题');
    const me=chMe(), done=chDone(); el.hidden=false; el.className='status'+(done?(me.met?' ok':' bad'):''); el.textContent=done?(me.met?tx('Goal met','达标'):tx('Not met','没达标')):tx('Not submitted','还没提交');
    el.dataset.tipx=tx('Whether your two fixes bring at least 75% of the answers right with at most 300 ms more wait per question. It shows once you submit.','你的两种改法是否让至少 75% 的回答答对，并且每个问题最多多等 300 毫秒。提交后才显示。'); return;}
  const pans=ctx(), num=pans.includes('bars')||pans.includes('sum');
  const title=pans.includes('sum')?tx('Help-center bot (example): 200 graded questions','帮助中心机器人（示例）：判过对错的 200 个问题')
    :pans.includes('bars')?tx(`Help-center bot (example): ${fxList(fxNow())}`,`帮助中心机器人（示例）：${fxList(fxNow())}`)
    :tx('One run of the help-center bot (example), stage by stage','帮助中心机器人的一次运行（示例），一环一环看');   // one title while the run plays (P4)
  if(st.textContent!==title) st.textContent=title;
  const b=num?{cls:'status',text:tx(`${pc(m.cor)} correct`,`答对 ${pc(m.cor)}`),tip:CKD.cor()}:runBadge();
  el.hidden=false; if(el.className!==b.cls) el.className=b.cls; if(el.textContent!==b.text) el.textContent=b.text; el.dataset.tipx=b.tip;
  if(num) el.style.minWidth=''; else badgeFit(el);}
function drawTransport(){
  const tEl=$('#rgf-transport',app), k=A.lang();
  if(tEl._k!==k){tEl._k=k;   // drawn once: Play keeps one width whatever it says (P4)
    tEl.innerHTML=`<button type="button" class="btn sm rgf-play" data-act="play" data-tipx="${esc(tx('Play the run stage by stage; press again to pause.','一环一环地播放这次运行；再按一次暂停。'))}">${PLAY()[0]}</button>`+
      A.ui.seg({id:'rgf-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(x=>[x,L(SPEED_N[x])]),value:speed})+
      `<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this run again from the question.','从问题开始，再看一遍这次运行。'))}">${tx('Replay','重放')}</button>`+
      A.ui.range({id:'rgf-tr-p',key:'prog',label:tx('Progress','进度'),min:0,max:100,value:100,out:'100%',inline:true});
    A.ui.initRanges(tEl); tEl._f=null;}
  const pl=$('.rgf-play',tEl); if(pl){const t=PLAY()[playing()?1:0]; if(pl.textContent!==t) pl.textContent=t; fitW(pl,PLAY());}
  const a=runAnim(), i=$('#rgf-tr-p-in',app), p=Math.round((a?a.p:1)*100); if(i&&+i.value!==p){i.value=p; A.ui.setRange(i);} const o=i&&i.closest('.inline-ctl,.ctl'), out=o&&o.querySelector('output'); if(out){if(out.textContent!==p+'%') out.textContent=p+'%'; fitW(out,['100%']);}}   // one width from 0% to 100% (P4)
function panelsShow(){const c=ctx(), sb=mode==='sandbox', open=!sb||runOpen;
  ['sum','bars','run'].forEach(p=>{const el=$('#rgf-pan-'+p,app); if(el) el.hidden=!c.includes(p);});
  $('#rgf-metrics',app).hidden=!c.includes('metrics');
  const tg=$('#rgf-runtog',app); tg.hidden=!sb; tg.setAttribute('aria-expanded',String(open)); $('#rgf-runbody',app).hidden=!open;
  const tk=A.lang()+'|'+open; if(tg._k!==tk){tg._k=tk; tg.textContent=open?tx('Hide the run','收起这次运行'):tx('See one run, stage by stage (example)','看一次运行，一环一环看（示例）');}
  $('#rgf-pan-run',app).classList.toggle('rgf-sbx',sb);
  const pk=$('#rgf-pk',app); pk.hidden=!sb; if(sb){const c=presetNow(); $$('[data-pick]',pk).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.pick===c))); $$('[data-pfx]',pk).forEach(b=>b.setAttribute('aria-pressed',String(S.fx.includes(b.dataset.pfx))));}
  const rs=$('#rgf-rsel',app); rs.disabled=mode==='guided';}
function drawRsel(){const el=$('#rgf-rsel',app), k=A.lang()+'|'+runNow(); if(el._k===k) return; el._k=k;
  el.innerHTML=RUNS.map((r,i)=>`<option value="${i}"${i===runNow()?' selected':''}>${i+1}. ${esc(r.q[li()])}</option>`).join(''); el.value=String(runNow());}
/* Show the math: one short block per check, then the fixes */
function mathHTML(){const m=compute(sys(),fxNow()), Z=LG()==='zh', T=s=>`\\text{${s}}`;
  const blk=(lead,a,b)=>`<div class="rgf-mb"><p class="rgf-mb-lead">${lead}</p>${M(a)}${b?M(b):''}</div>`;
  const nm=(k)=>T(L(CKN[k]));
  let h=blk(tx('Answer correctness: correct answers over all questions.','答案正确率：答对的问题占全部问题的比例。'),`${nm('cor')} = \\frac{n_{\\text{${Z?'答对':'correct'}}}}{N}`,`= \\frac{${m.ok}}{${m.N}} = ${T(pc1(m.cor))}`);
  h+=blk(tx(`Retrieval recall: only the ${m.ans} questions the documents can answer count; those of the kinds “${L(KN.miss)}” and “${L(KN.multi)}” fail it.`,`检索召回率：只算文档里有答案的 ${m.ans} 个问题；属于“${L(KN.miss)}”和“${L(KN.multi)}”这两类的问题，不通过。`),
    `${nm('ret')} = \\frac{n_{\\text{${Z?'有答案':'answerable'}}} - n_{\\text{${Z?'没找到':'miss'}}} - n_{\\text{${Z?'要两段':'two'}}}}{n_{\\text{${Z?'有答案':'answerable'}}}}`,`= \\frac{${m.ans} - ${m.left.miss} - ${m.left.multi}}{${m.ans}} = ${T(pc1(m.ret))}`);
  h+=blk(tx('Context recall: the same questions; those whose passage was cut by the budget fail it too.','上下文召回率：还是这些问题；段落被预算挤掉的，也不通过。'),
    `${nm('ctx')} = \\frac{n_{\\text{${Z?'有答案':'answerable'}}} - n_{\\text{${Z?'没找到':'miss'}}} - n_{\\text{${Z?'要两段':'two'}}} - n_{\\text{${Z?'被挤掉':'cut'}}}}{n_{\\text{${Z?'有答案':'answerable'}}}}`,`= \\frac{${m.ans} - ${m.left.miss} - ${m.left.multi} - ${m.left.cut}}{${m.ans}} = ${T(pc1(m.ctx))}`);
  h+=blk(tx('Grounded: answers whose every sentence has a source in the context, over all questions.','有据可依：每句话都能在上下文里找到依据的回答，占全部问题的比例。'),
    `${nm('grd')} = \\frac{N - n_{\\text{${Z?'没照着答':'unsupported'}}} - n_{\\text{${Z?'没答案':'no answer'}}}}{N}`,`= \\frac{${m.N} - ${m.left.unsup} - ${m.left.noans}}{${m.N}} = ${T(pc1(m.grd))}`);
  const fx=fxNow();
  if(fx.length){const ks=KINDS.filter(k=>fx.some(f=>SHARE[f][k]));
    h+=`<div class="rgf-mb"><p class="rgf-mb-lead">${tx('The fixes: a question stays wrong only if none of the chosen fixes repairs it, so the shares left over multiply (the lab’s assumption), rounded to whole questions. n is how many questions the kind had; s₁, s₂ … are the shares the chosen fixes repair of it, such as 0.6.','改法：一个问题只有在所选的改法都没修好它时才仍然答错，所以剩下的比例相乘（这个 Lab 的假设），再取整到整数个问题。n 是这一类原来有多少个问题；s₁、s₂ …… 是所选的改法各能修好这一类的比例，比如 0.6。')}</p>`+
      M(`n_{\\text{${Z?'剩下':'left'}}} = n \\times (1 - s_1) \\times (1 - s_2) \\times \\cdots`)+
      ks.map(k=>{const fs=fx.filter(f=>SHARE[f][k]); return `<div class="rgf-mk"><div class="small muted">${esc(L(KN[k]))}</div>`+M(`${MIX[sys()][k]} ${fs.map(f=>`\\times (1 - ${SHARE[f][k]})`).join(' ')} ${Math.abs(fs.reduce((a,f)=>a*(1-SHARE[f][k]),MIX[sys()][k])-m.left[k])<1e-9?'=':'\\approx'} ${m.left[k]}`)+`</div>`;}).join('')+`</div>`;}
  return h;}
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key;
    if(k==='speed') A.ui.setPressed(gEl,speed); else if(k.startsWith('fx-')) A.ui.setPressed(gEl,S.fx.includes(k.slice(3))?'1':'0'); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));});
  const lock=mode==='challenge'||mode==='guided', c=$('#rgf-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#rgf-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=mode==='challenge'?tx('In the challenge you pick the two fixes in the task card.','挑战中，两种改法在任务卡里选。'):tx('In the guided parts each part sets the fixes; turn them on freely in Scenarios.','分步引导里，每一部分自己设定改法；到“应用场景”里可以随意打开。');
  $$('#rgf-controls .lockable',app).forEach(e=>{e.hidden=lock;});}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(){
  const rail=$('#rgf-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; box.hidden=false; if(again) A.ui.obsBump(box);}}}
    A.ui.setNext(rail,isNextDisabled());
    $$('[data-nav="showme"]',rail).forEach(sm=>{sm.hidden=!!observed[step];});
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#rgf-insight',rail); if(ins){const html=`<div class="eyebrow rgf-keep">${c>=0?esc(L(PRESETS[c].t)):tx('Your own fixes','你自己的改法')}</div>`+insightHTML(); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const me=chMe(), done=chDone(), nv=$('#rgf-ch-nv',rail); if(!nv) return;
    nv.textContent=String(chFx.length); $('#rgf-ch-wv',rail).textContent=waitTxt(me.wait); $('#rgf-ch-w',rail).classList.toggle('over',me.wait>CH_WAIT);
    $('#rgf-ch-cv',rail).textContent=done?pc(me.cor):tx('shown after you submit','提交后才知道'); $('#rgf-ch-c',rail).classList.toggle('met',done&&me.cor>=CH_GOAL-1e-9);
    $$('[data-chfx]',rail).forEach(b=>{const on=chFx.includes(b.dataset.chfx); b.setAttribute('aria-pressed',String(on)); b.disabled=!on&&chFx.length>=2;});
    const sb=$('[data-act="submit"]',rail); if(sb) sb.disabled=chFx.length!==2; const sw=$('.rgf-subw',rail); if(sw) sw.textContent=chFx.length===2?'':tx('Pick two fixes to submit.','选好两种改法再提交。');
    const sub=$('#rgf-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    const res=$('#rgf-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#rgf-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  panelsShow(); drawRsel();
  if(runShown()){maybePlay(); drawTransport(); drawRun();} else if(anim) stopAnim();
  reqPlay=false;
  drawHead();
  if(!$('#rgf-pan-sum',app).hidden) drawSum();
  if(!$('#rgf-pan-bars',app).hidden) drawBars();
  if(!$('#rgf-metrics',app).hidden) drawMetrics();
  const mb=$('#rgf-mathbody',app); mb.innerHTML=mathHTML();
  syncControls(); updateRail();
  A.hash.write(token());   // outside Scenarios the kit writes the bare lab address, so a reload puts back the saved mode (round 3 #6)
  A.dock.refresh(); announce();}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const m=compute(sys(),fxNow()); A.announce(tx(`${pc(m.cor)} of answers correct, retrieval recall ${pc(m.ret)}.`,`答对 ${pc(m.cor)}，检索召回率 ${pc(m.ret)}。`));},400);}

/* ---------- the run replay (R3.9, R3.20): it plays only for a run not yet watched; a watched one shows at once. On phones the record is taller
   than what is left of the screen, so a new run waits for ▶, which first brings the record into view (round-1 polish P2); in Scenarios the
   record opens only on the reader's request ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} if(anim&&anim.p>0) seenRun.add(anim.key); anim=null; lastTs=0;}
function maybePlay(){const k=RUNS[runNow()].id, ask=reqPlay; reqPlay=false;
  if(anim&&anim.key===k){if(ask&&anim.pending) resumePlay(); return;}   // N6 of the round-1 check
  if(anim) stopAnim(); if(seenRun.has(k)) return;
  if(phone()&&!ask&&!A.reduceMotion()){anim={key:k,p:0,paused:true,pending:true}; return;}
  startPlay(0);}
function startPlay(p0){stopAnim(); const k=RUNS[runNow()].id;
  if(A.reduceMotion()){seenRun.add(k); return;}
  anim={key:k,p:p0||0,paused:false}; lastTs=0; raf=requestAnimationFrame(tick);}
function pausePlay(){if(!anim) return; anim.paused=true; if(raf){cancelAnimationFrame(raf); raf=0;} if(anim.p>0) seenRun.add(anim.key);}
function resumePlay(){if(!anim) return; anim.paused=false; anim.pending=false; lastTs=0; raf=requestAnimationFrame(tick);}
const playing=()=>!!anim&&!anim.paused;
function tick(ts){raf=0; if(!anim||anim.paused||!app) return;
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts;
  anim.p=Math.min(1,anim.p+dt/(DUR[speed]*beats().end/BT.end));
  if(anim.p>=1){const k=anim.key; anim=null; seenRun.add(k); render(); return;}
  runPhase(); drawTransport(); drawHead(); drawGo();
  raf=requestAnimationFrame(tick);}
/* bring the record into view before it plays. Phones: the highest of the stage's head (guided parts), the run's controls and the note on
   the record that still fits on the screen together with the whole record goes under the top bar, so Pause stays in reach (N9 of the
   round-1 check) and the note that the record is the initial setup's is not covered (N7); the record is then the dock's anchor, so the
   dock stays away. Wider screens, in Scenarios: the stage scrolls inside itself to the record, and the page follows when the record's
   end is below the window */
function revealRun(){if(!app) return; const fl=$('#rgf-flow',app), st=$('#rgf-stage',app); if(!fl||!st||!fl.getClientRects().length) return;
  const beh=A.reduceMotion()?'auto':'smooth', bar=document.querySelector('.topbar'), tb=(bar?A.ui.barB():56)+8;
  if(phone()){const fb=fl.getBoundingClientRect().bottom, room=innerHeight-tb-4, nt=$('#rgf-rnote',app);
    const t=[mode==='guided'?$('#rgf-stage .rgf-head',app):null,$('#rgf-pan-run .rgf-ctl',app),nt&&!nt.hidden?nt:null].find(e=>e&&e.getClientRects().length&&fb-e.getBoundingClientRect().top<=room)||fl;
    const d=t.getBoundingClientRect().top-tb; if(Math.abs(d)>4) window.scrollBy({top:d,behavior:beh}); return;}
  if(mode!=='sandbox') return;
  const pan=$('#rgf-pan-run',app), sr=st.getBoundingClientRect(), pr=pan.getBoundingClientRect(), fr=fl.getBoundingClientRect();
  const d=st.scrollHeight>st.clientHeight+1?Math.max(-st.scrollTop,Math.min(pr.top-sr.top-8,st.scrollHeight-st.clientHeight-st.scrollTop)):0;
  if(Math.abs(d)>2) st.scrollTo({top:st.scrollTop+d,behavior:beh});
  const bottom=fr.bottom-d, top=pr.top-d; if(bottom>innerHeight-8){const dy=Math.min(bottom-innerHeight+8,top-tb); if(dy>2) window.scrollBy({top:dy,behavior:beh});}}
/* in Scenarios and Settings a change shows in the breakdown at the top of the stage: if the stage was scrolled down to the record, back up */
function showTop(){if(!app||mode==='guided') return; const st=$('#rgf-stage',app), b=$('#rgf-pan-bars',app);
  if(st&&b&&st.scrollTop>0&&b.getBoundingClientRect().top<st.getBoundingClientRect().top) st.scrollTo({top:0,behavior:A.reduceMotion()?'auto':'smooth'});}

/* ---------- navigation ---------- */
function enterStep(i){
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter){const e=s.enter(); Object.assign(S,e); if(e.fx) S.fx=e.fx.slice();}
  tried.clear(); tried.add(trialKey());
  renderRail();}
function trialKey(){const id=mode==='guided'?STEPS[step].id:''; return id==='metrics'?'m'+S.met:(id==='fixes'?'f'+S.fx.join(','):'');}
function revealStep(){A.ui.reveal($('#rgf-rail .step.current',app)||$('#rgf-rail',app));}
/* scroll a result into view, but never past the option just clicked (keep): it stays on screen (round-1 polish P15) */
function revealBox(box,keep){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  let dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(keep&&keep.getClientRects().length) dy=Math.min(dy,keep.getBoundingClientRect().top-top);
  if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=keepSel=>requestAnimationFrame(()=>{if(!app) return; const keep=keepSel?$(keepSel,app):null;
  if(mode==='guided'){const ob=$('#rgf-rail [data-obs]',app); revealBox(ob&&!ob.hidden?ob:$('#rgf-rail .rgf-verd',app),keep);} else if(mode==='sandbox') revealBox($('#rgf-insight',app),keep); else revealBox($('#rgf-ch-res',app),keep);});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m; A.shell.setMode(app,m,DESCS());
  if(was==='challenge'&&saved){S={...saved,fx:saved.fx.slice()}; saved=null;}
  if(m==='challenge'){saved={...S,fx:S.fx.slice()}; chSubmitted=false; chReveal=false; chFx=[];}
  if(m==='sandbox'&&was==='guided'){S={...DEFAULT,fx:[],run:S.run};}
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){S.fx=PRESETS[i].fx.slice(); byLearner(render); showTop(); revealObs();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
/* phones: keep a tapped button under the finger while what is above it settles (the stage's title taking another line, the legend), as the
   kit's pinCtl does for Settings (R3.2) */
let pinG=0;
function pinEl(el){if(!phone()||!el) return; const y0=el.getBoundingClientRect().top, g=++pinG; let sy=scrollY;
  const fix=()=>{if(g!==pinG||!app||!el.isConnected) return; if(Math.abs(scrollY-sy)>1){pinG++; return;}   // the reader scrolled meanwhile: leave the page where they took it
    const dy=el.getBoundingClientRect().top-y0; if(Math.abs(dy)>=1) window.scrollBy(0,dy); sy=scrollY;};
  requestAnimationFrame(()=>{fix(); requestAnimationFrame(fix);}); setTimeout(fix,150); setTimeout(fix,450);}
/* after a mode switch. Phones, Scenarios and Settings: the steps' list goes right under the top bar, so the breakdown, the cards and the
   buttons under them are on the screen together (P18 of the round-1 check). The kit's reveal would keep room for the dock there, but the
   dock stays away while the breakdown is on the screen; and in Settings it would bring up the drawer, which is under the stage */
function revealMode(){if(!(mode==='sandbox'&&phone())){A.ui.reveal($('#rgf-rail',app)); return;}
  const bn=$('.bench',app), bar=document.querySelector('.topbar'); if(!bn) return; A.tip.hide();
  const d=bn.getBoundingClientRect().top-((bar?A.ui.barB():56)+6); if(Math.abs(d)>4) window.scrollBy({top:d,behavior:A.reduceMotion()?'auto':'smooth'});}
const token=()=>'f'+FIXES.map(f=>S.fx.includes(f)?1:0).join('')+'_r'+S.run;
function parse(tok){const mm=/^f([01]{5})_r(\d)$/.exec(String(tok||'')); if(!mm) return null; const r=+mm[2]; if(r>=RUNS.length) return null;
  return {fx:FIXES.filter((f,i)=>mm[1][i]==='1'),run:r};}
function act(a,btn){
  if(a==='play'||a==='go'){if(anim&&anim.pending){revealRun(); resumePlay();} else if(playing()) pausePlay(); else if(anim) resumePlay(); else {revealRun(); startPlay(0);} render(); return;}
  if(a==='replay'){revealRun(); startPlay(0); render(); return;}
  if(a==='runtog'){runOpen=!runOpen; if(runOpen) reqPlay=true; else stopAnim(); render(); if(runOpen) requestAnimationFrame(revealRun); else showTop(); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{S.fx=[]; render();}); return;}
  if(a==='submit'){if(chFx.length!==2) return; chSubmitted=true; render(); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#rgf-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(p){byLearner(()=>{Object.assign(S,p); if(p.fx){S.fx=p.fx.slice(); p.fx.forEach(f=>lvTried.add(f));} tried.add(trialKey()); render();}); revealObs();}

/* ---------- interview drill (seeded): which stage loses the most answers? ---------- */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const rnd=mulberry(seed*7919+13), pick=(a,b)=>a+Math.floor(rnd()*(b-a+1));
  const N=400, ans=pick(340,380); let miss, cut, unsup;
  for(;;){miss=pick(4,40); cut=pick(4,40); unsup=pick(4,40); const o=[miss,cut,unsup].sort((a,b)=>b-a); if(o[0]-o[1]>=3) break;}   // one stage loses clearly the most
  const ret=(ans-miss)/ans, ctxR=(ans-miss-cut)/ans, grd=(N-unsup-(N-ans))/N;
  const losses={ret:miss,ctx:cut,grd:unsup}; const worst=Object.keys(losses).sort((a,b)=>losses[b]-losses[a])[0];
  return {seed,N,ans,miss,cut,unsup,ret,ctx:ctxR,grd,worst,losses};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#rgf-drill-q',app), a=$('#rgf-drill-a',app);
  const r1=v=>Math.round(v*1000)/1000, sh=v=>A.fmt.num(r1(v)*100,1)+'%', f3=v=>r1(v).toFixed(3);   // the values the question shows, and the same values in the answer
  q.innerHTML=tx(`A RAG bot answered ${d.N} test questions; the documents can answer ${d.ans} of them. Retrieval recall is ${sh(d.ret)}, context recall ${sh(d.ctx)}. Among all ${d.N} answers, ${sh(d.grd)} are grounded, and every unanswerable question got a made-up answer. Assume that when retrieval or the budget loses a passage, the answer still sticks to the passages it got, so it counts as grounded. Which stage loses the most answerable questions, and how many?`,
    `一个 RAG 机器人回答了 ${d.N} 个测试问题，其中文档里有答案的 ${d.ans} 个。检索召回率 ${sh(d.ret)}，上下文召回率 ${sh(d.ctx)}。全部 ${d.N} 个回答里有据可依的占 ${sh(d.grd)}，而且每个没有答案的问题都被编了一个答案。假设检索或预算丢掉了段落的问题，回答都只照着拿到的段落说，都算有依据。哪一环丢掉的有答案的问题最多？丢了多少个？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const nm={ret:tx('retrieval','检索'),ctx:tx('the context budget','上下文预算'),grd:tx('the model (not grounded)','模型（没有依据）')}, eq=(x,n)=>Math.abs(x-n)<1e-6?'=':'\\approx';
  a.innerHTML=tx(`<p><b>${nm[d.worst]}: ${d.losses[d.worst]} questions</b></p>`,`<p><b>${nm[d.worst]}：${d.losses[d.worst]} 个问题</b></p>`)+
    M(`${d.ans} \\times (1 - ${f3(d.ret)}) ${eq(d.ans*(1-r1(d.ret)),d.miss)} ${d.miss}`)+M(`${d.ans} \\times (${f3(d.ret)} - ${f3(d.ctx)}) ${eq(d.ans*(r1(d.ret)-r1(d.ctx)),d.cut)} ${d.cut}`)+M(`${d.N} \\times (1 - ${f3(d.grd)}) - ${d.N-d.ans} ${eq(d.N*(1-r1(d.grd))-(d.N-d.ans),d.unsup)} ${d.unsup}`)+
    tx(`<p>Each check’s loss is the drop from the stage before it. Not-grounded answers include the ${d.N-d.ans} made-up answers to unanswerable questions; subtract them to count the answerable ones the model got wrong.</p>`,`<p>每一环丢掉的，是它比上一环少的那部分。没有依据的回答里，包括给 ${d.N-d.ans} 个没有答案的问题编出来的回答；减掉它们，才是模型在有答案的问题上答错的数。</p>`);}

/* ---------- jumps ---------- */
function jumpAct(k){
  if(k==='flow') return tx('Click to show the record of the run.','点击查看这次运行的记录。');
  if(k==='bars') return tx('Click to show the chart of wrong answers by stage.','点击查看按环节分类的错答图。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  return '';}
function jump(k){
  if(!app) return;
  const pan=id=>{const el=$('#rgf-pan-'+id,app); A.tip.hide(); if(el&&!el.hidden){el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);}};
  if(k==='flow'){pan('run'); return;}
  if(k==='bars'){pan('bars'); return;}
  if(k==='controls'){const c=$('#rgf-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c);}}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    const st=$('#rgf-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#rgf-rail .step.current',app)||$('#rgf-rail .railcard',app)||$('#rgf-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, rc=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=rc.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='prog'){const p=+e.target.value/100, key=RUNS[runNow()].id; stopAnim(); seenRun.add(key); if(p<1) anim={key,p,paused:true}; drawTransport(); drawRun(); drawHead(); updateRail();}});
  el.addEventListener('change',e=>{if(e.target.id==='rgf-rsel'){S.run=+e.target.value; reqPlay=true; byLearner(render); requestAnimationFrame(revealRun);}});
  /* after a click that scrolls the page, the option under the still pointer shows no hover frame until the pointer moves (round-1 polish P15) */
  el.addEventListener('mousemove',e=>{const r=$('#rgf-rail',el); if(r&&r._hx!=null&&Math.abs(e.clientX-r._hx)+Math.abs(e.clientY-r._hy)>3){r.classList.remove('rgf-nohov'); r._hx=null;}});
  /* the same after a click on a part's title, Next or Back: the page scrolls to the new part under the still pointer (待办 4 of the round-1 check) */
  const noHov=e=>{const r=$('#rgf-rail',el); if(r){r.classList.add('rgf-nohov'); r._hx=e.clientX; r._hy=e.clientY;}};
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#rgf-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#rgf-rail');
      if(k==='speed'){speed=d.v; drawTransport(); syncControls(); return;}
      if(k.startsWith('fx-')){const f=k.slice(3); byLearner(()=>{S.fx=FIXES.filter(x=>x===f?d.v==='1':S.fx.includes(x)); render();}); showTop(); return;}
      byLearner(()=>{S[k]=k==='log'?+d.v:d.v; tried.add(trialKey()); if(k==='log'&&phone()) reqPlay=true; render();});
      const a=runAnim(); if(k==='log'&&phone()&&a&&!a.paused) requestAnimationFrame(revealRun);   // the record plays where it can be seen; the observation is right under it after (N6)
      else if(inRail) revealObs(); return;}
    /* phones: the scenario and fix buttons under the cards (P18); the tapped button stays where it was while the stage above it settles */
    if(d.pick!=null){pinEl(b); byLearner(()=>{S.fx=PRESETS[+d.pick].fx.slice(); render();}); return;}
    if(d.pfx){const f=d.pfx; pinEl(b); byLearner(()=>{S.fx=FIXES.filter(x=>x===f?!S.fx.includes(f):S.fx.includes(x)); render();}); return;}
    if(d.lever!=null&&b.closest('#rgf-rail')){noHov(e);
      byLearner(()=>{S.fx=[d.lever]; lvTried.add(d.lever); tried.add(trialKey()); renderRail();}); revealObs(`#rgf-rail [data-lever="${d.lever}"]`); return;}
    if(d.tri){const i=tri.i; if(tri.picks[i]) return; const a=runAnim(); if(a&&a.pending){seenRun.add(a.key); anim=null;}   // picked before ▶ on a phone: the record shows whole, as the verdict does (N6)
      tri.picks[i]=d.tri; renderRail(); revealObs(); return;}
    if(d.trigo!=null){stopAnim(); tri.i=+d.trigo; reqPlay=phone(); renderRail(); if(phone()) requestAnimationFrame(revealRun); else revealStep(); return;}
    if(d.chfx){const f=d.chfx; if(chFx.includes(f)) chFx=chFx.filter(x=>x!==f); else if(chFx.length<2) chFx=chFx.concat(f); else return;
      chSubmitted=false;
      render(); return;}
    if(d.go!=null){noHov(e); enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){noHov(e); enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){noHov(e); enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){showMe(s.showMe); if(s.id==='fixes') renderRail();} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#rgf-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); revealMode(); return;}
    if(d.act) act(d.act,b);});}
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): fixes, run, log and check settings, part, prediction, the triage picks, what was
     tried, the runs watched, the challenge picks and result */
  snap:()=>({v:1,mode,step,S:{...S,fx:S.fx.slice()},pred,observed:{...observed},saved:saved?{...saved,fx:saved.fx.slice()}:null,tri:{i:tri.i,picks:tri.picks.slice()},
    chFx:chFx.slice(),chSubmitted,chReveal,tried:[...tried],lvTried:[...lvTried],seenRun:[...new Set([...seenRun,...(anim&&anim.p>0?[anim.key]:[])])],   // a replay under way counts as watched, as stopAnim() counts it
   drillSeed,drillOpen,speed,runOpen,passed:mode==='challenge'&&chSubmitted&&chMe().met}),
  restore(o){if(!o||o.v!==1) return;
    const ps=s=>{if(!s||typeof s!=='object'||!Array.isArray(s.fx)||!s.fx.every(f=>FIXES.includes(f))||!Number.isInteger(s.run)||s.run<0||s.run>=RUNS.length) return null;
      return {...DEFAULT,run:s.run,fx:FIXES.filter(f=>s.fx.includes(f)),log:s.log===0?0:1,met:CHK.includes(s.met)?s.met:DEFAULT.met};}, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b;
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.includes(o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    const t=o.tri&&typeof o.tri==='object'?o.tri:{}, pk=Array.isArray(t.picks)?t.picks:[];
    tri={i:ints(t.i,0,TRI.length-1)?t.i:0,picks:TRI.map((x,j)=>Object.keys(KN).includes(pk[j])?pk[j]:null)};
    chFx=Array.isArray(o.chFx)?[...new Set(o.chFx.filter(f=>FIXES.includes(f)))].slice(0,2):[]; chSubmitted=bool(o.chSubmitted)&&chFx.length===2; chReveal=chSubmitted&&bool(o.chReveal);
    tried.clear(); if(Array.isArray(o.tried)) o.tried.forEach(k=>{if(typeof k==='string'&&k.length<60&&/^(m[a-z]+|f[a-z,]*)?$/.test(k)) tried.add(k);}); if(!tried.size) tried.add(trialKey());
    lvTried.clear(); if(Array.isArray(o.lvTried)) o.lvTried.forEach(f=>{if(FIXES.includes(f)) lvTried.add(f);});
    seenRun.clear(); if(Array.isArray(o.seenRun)) o.seenRun.forEach(k=>{if(RUNS.some(r=>r.id===k)) seenRun.add(k);});
    drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen); runOpen=bool(o.runOpen); if(Object.keys(DUR).includes(o.speed)) speed=o.speed;},
  mount(el,tok){
    stopAnim(); root=el;
    const p=tok?parse(tok):null; if(p){S={...DEFAULT,fx:p.fx,run:p.run}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#rgf-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); revealMode(); keepStage();},token});
    /* a tap on the dock while the run waits for ▶ plays it, where it can be seen (N6 of the round-1 check); the kit's own handler has
       already scrolled to the stage */
    const dk=document.getElementById('dock'); if(dk&&!dk._rgf){dk._rgf=1; dk.addEventListener('click',()=>{if(!app||!runShown()) return; const a=runAnim(); if(a&&a.pending){resumePlay(); render(); requestAnimationFrame(revealRun);}});}
    bind(app); renderDrill();
    renderRail();},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const dm=$('#rgf-math',app), open=dm&&dm.open; stopAnim(); this.mount(root,null); if(open) $('#rgf-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    /* the picture that answers the reader's last change: part 4's two numbers, the breakdown, or the record of the run. In Scenarios with
       the record open, once the reader is down at the record (the breakdown above the screen) the record is the anchor, so the dock does
       not cover it or its note (N7 of the round-1 check) */
    anchor:()=>{if(!app) return null; const sum=$('#rgf-pan-sum:not([hidden])',app); if(sum) return sum;
      const bars=$('#rgf-pan-bars:not([hidden])',app), fl=$('#rgf-flow',app); if(!bars) return fl;
      if(runShown()&&fl&&fl.getClientRects().length&&bars.getBoundingClientRect().bottom<70&&fl.getBoundingClientRect().top<innerHeight) return fl;
      return bars;}, bench:()=>app&&$('.bench',app), target:()=>app&&$('#rgf-stage',app),
    render({status,vis,read}){if(!app) return; const m=compute(sys(),fxNow()), m0=compute(sys(),[]), ok0=MIX[sys()].ok;
      /* the breakdown's top bar in small: correct, repaired by the fixes (hatched), still wrong (P18) */
      vis.innerHTML=`<div class="rgf-dk"><i class="ok" style="width:${(ok0/m.N*100).toFixed(1)}%"></i><i class="fx" style="width:${((m.ok-ok0)/m.N*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const me=chMe(), done=chDone(); status.className='status'+(done?(me.met?' ok':' bad'):''); status.textContent=done?(me.met?tx('Goal met','已达标'):tx('Not yet','未达标')):tx('Not submitted','未提交');
        read.textContent=done?tx(`${pc(me.cor)} correct, ${waitTxt(me.wait)} more per question`,`答对 ${pc(me.cor)}，每个问题多等 ${waitTxt(me.wait)}`):tx(`${chFx.length} of 2 fixes picked, ${waitTxt(me.wait)} more per question`,`已选 ${chFx.length} / 2 种改法，每个问题多等 ${waitTxt(me.wait)}`); return;}
      const pans=ctx();
      if(pans.includes('bars')||pans.includes('sum')){status.className='status'; status.textContent=tx(`${pc(m.cor)} correct`,`答对 ${pc(m.cor)}`);
        if(mode==='guided'&&STEPS[step].id==='metrics'){const c=S.met, i=CHK.indexOf(c), n=KINDS.filter(k=>FLAG[k][i]===1).reduce((a,k)=>a+m.left[k],0), v={ret:m.ret,ctx:m.ctx,grd:m.grd,cor:m.cor}[c];   // the check picked (round-1 polish P27)
          read.textContent=tx(`${L(CKN[c])} ${pc(v)}: catches ${n===m.wrong?`all ${n} wrong answers`:`${n} of the ${m.wrong} wrong answers`}`,`${L(CKN[c])} ${pc(v)}：${n===m.wrong?`答错的 ${n} 个全都发现了`:`答错的 ${m.wrong} 个里发现 ${n} 个`}`);}
        else if(fxNow().length){const bc=bigChange(m,m0), nm=bc?L(CKN[bc[0]]):'';   // with fixes on: what they add, and the check that moves the most (P18)
          read.textContent=tx(`${m.ok-m0.ok} more correct than with no fix; `,`比不改多答对 ${m.ok-m0.ok} 个；`)+(bc?`${LG()==='zh'?nm:nm.toLowerCase()} ${bc[1]}% → ${bc[2]}%`:tx('the other three checks are unchanged','其余三项检查不变'));}
        else read.textContent=tx(`${m.ok} of 200 answers correct; retrieval recall ${pc(m.ret)}`,`200 个回答里答对 ${m.ok} 个；检索召回率 ${pc(m.ret)}`);
        return;}
      /* the run: what this part asks of the reader (round-1 polish P7) */
      const r=RUNS[runNow()], a=runAnim(), b=runBadge();
      if(logOff()){status.className='status neutral'; status.textContent=tx('Not logged','没有记录'); read.textContent=tx('Only the question and the answer were logged','只记了问题和回答'); return;}
      status.className=b.cls; status.textContent=b.text;
      /* waiting for ▶: a tap on the dock plays it (N6); after the verdict: the stage, and for an outdated page what went wrong there (N1).
         The status names the kind already, so the line does not repeat it */
      read.textContent=a&&a.pending?tx('Tap here to play this run','点这里播放这次运行')
        :a&&!phase(a.p,r.ans.length).verd?tx('The verdict shows at the end of the run.','结论在这次运行的最后出现。')
        :triHidden()?tx('Which kind of failure is it? Pick it in the card.','这是哪一类错？在卡片里选。')
        :r.kind==='ok'?tx('This run passes every check.','这次运行四项检查都通过。')
        :tx(`Failed at ${stgTxt(CAUSE[r.kind])}`,`错在 ${stgTxt(CAUSE[r.kind])}`)+stgWhy(r.kind);}}
});
/* read-only hook for tests/rag-failure_numbers.py and the home tile */
window.__rgf={state:()=>({mode,step,run:runNow(),fx:S.fx.slice(),log:S.log,met:S.met,pred,tri:JSON.parse(JSON.stringify(tri)),chFx:chFx.slice(),chSubmitted,chReveal,lang:LG(),speed}),
  animState:()=>anim?{p:anim.p,key:anim.key,paused:!!anim.paused,pending:!!anim.pending}:null,seen:()=>[...seenRun],runOpen:()=>runOpen,
  data:()=>({CH,RUNS,MIX,SHARE,WAIT,FLAG,CAUSE,KINDS,FIXES,NRET,BUDGET,CH_GOAL,CH_WAIT}),
  compute:(s,fx)=>compute(s,fx),checks:i=>checks(RUNS[i]),chAll:()=>chAll(),drill:makeDrill,drillShow:sd=>{drillSeed=sd; drillOpen=true; renderDrill();},kindName:k=>L(KN[k])};
})();
