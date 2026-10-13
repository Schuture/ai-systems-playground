/* Chunking & rerank lab: why does your retriever miss the right passage? The help center of a fictional smart-home company (示例,
   the same Lumo Home / 小鹿家居 world as the embeddings lab, documents written for this lab) is cut into chunks (fixed size with
   overlap, or by heading), searched three ways (keyword search with BM25, vector search with a small hand-written encoder, hybrid by
   reciprocal rank fusion) and optionally reranked by the lab's stand-in reranker. Everything is computed in the page and is
   deterministic; tests/chunk-rerank_numbers.py recomputes it in Python.
   Names (batch 10 brief S6): 切块、块大小、重叠; 关键词检索（BM25）、混合检索; rerank（重排）; 召回率 recall@k (here: is a chunk that
   holds the whole answer among the k chunks kept). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='chunk-rerank';

/* ---------- engine: pure functions (ENGINE-START) ---------- */
/* the help center (示例, written for this lab): [id, [title en, zh], sections]; a section is [[heading en, zh], [[sentence en, zh], …]] */
const DOCS=[
 ["L",["Smart lock S2: user guide", "智能门锁 S2 使用说明"],[
   [["Batteries", "电池"],[["The lock runs on four AA batteries.", "门锁用四节 5 号电池。"],["A new set lasts about eight months.", "一套新电池大约能用八个月。"],["When the batteries are low, the keypad flashes red and the app sends a reminder.", "电量不足时，键盘会闪红灯，App 也会发提醒。"],["Replace all four batteries at the same time.", "换电池时，四节要一起换。"]]],
   [["Emergency entry", "应急开门"],[["If the batteries run out completely, you can still get in.", "电池彻底没电了，也还能进门。"],["Slide off the cover under the handle and turn the mechanical key.", "把把手下方的盖板推开，插入机械钥匙转动即可。"],["You can also hold a 9-volt battery to the two contacts under the keypad and enter your code within one minute.", "也可以把一节 9 伏电池贴住键盘下方的两个触点，在一分钟内输入密码。"]]],
   [["Adding a fingerprint", "添加指纹"],[["In the app, choose Lock, then Users, then Add fingerprint.", "在 App 里依次点门锁、用户、添加指纹。"],["Place the same finger on the sensor six times.", "把同一根手指在传感器上按六次。"],["Lift it and turn it slightly each time, so the sensor records the whole finger.", "每次抬起后稍微换个角度，让传感器录下整根手指。"]]],
   [["Auto-lock", "自动上锁"],[["Auto-lock locks the door again after it closes.", "自动上锁会在门关上后重新锁门。"],["By default it waits 30 seconds.", "默认等待 30 秒。"],["In Settings you can set any wait from 5 seconds to 5 minutes, or switch auto-lock off.", "在设置里可以改成 5 秒到 5 分钟，也可以关掉自动上锁。"]]]
 ]],
 ["T",["Thermostat T2: user guide", "恒温器 T2 使用说明"],[
   [["Setting a schedule", "设置日程"],[["In the app, choose Thermostat, then Schedule.", "在 App 里依次点恒温器、日程。"],["Add a time and a temperature for each part of the day.", "给一天里的每个时段设一个时间和一个温度。"],["The thermostat follows the schedule every day until you change it.", "恒温器每天按日程运行，直到你修改它。"]]],
   [["Eco mode", "节能模式"],[["Eco mode lowers the temperature by 2 degrees while nobody is home.", "节能模式会在家里没人时把温度调低 2 度。"],["It uses the location of your phone to know when you leave.", "它根据手机的位置判断你什么时候离开。"],["Switch it off in Settings if you work from home.", "如果你在家办公，可以在设置里关掉它。"]]],
   [["Power", "电源"],[["The thermostat is powered by the wires of the heating system.", "恒温器由供暖系统的电线供电。"],["It has no batteries to replace.", "它没有需要更换的电池。"],["If the screen stays dark, check the fuse of the heating system.", "如果屏幕一直不亮，检查供暖系统的保险丝。"]]]
 ]],
 ["V",["Robot vacuum V1: user guide", "扫地机器人 V1 使用说明"],[
   [["Charging", "充电"],[["The robot returns to its dock to charge when its battery is low.", "电量低时，机器人会自己回到底座充电。"],["A full charge takes about four hours and cleans for 120 minutes.", "充满一次大约要四个小时，能清扫 120 分钟。"],["Keep the dock against a wall, with half a metre free on each side.", "底座要靠墙放，两侧各留半米空间。"]]],
   [["Cleaning the brush", "清理主刷"],[["Clean the main brush once a week.", "主刷每周清理一次。"],["Press the two clips, lift the brush out and cut away any hair with the tool in the box.", "按下两个卡扣，取出主刷，用包装盒里的小刀割掉缠绕的头发。"]]],
   [["Emptying the bin", "倒尘盒"],[["Empty the dust bin after every run.", "每次清扫完都要倒掉尘盒里的灰尘。"],["Wash the filter with water once a month and let it dry for a day.", "滤网每月用水冲洗一次，晾干一天再装回。"]]]
 ]],
 ["E",["Troubleshooting: error codes", "故障排查：错误码"],[
   [["Reading a code", "看懂错误码"],[["Every Lumo device shows an error code on its screen or in the app.", "小鹿家居的设备出问题时，会在屏幕上或 App 里显示错误码。"],["A code is the letter E followed by three digits.", "错误码是字母 E 加三位数字。"],["The first digit names the device: 1 for the thermostat, 2 for the lock, 3 for the robot vacuum.", "第一位数字表示设备：1 是恒温器，2 是门锁，3 是扫地机器人。"]]],
   [["E-101", "E-101"],[["E-101 means the thermostat has just been reset.", "E-101 表示恒温器刚被重置过。"],["Set the schedule again.", "请重新设置日程。"]]],
   [["E-102", "E-102"],[["E-102 means the thermostat has lost its Wi-Fi connection.", "E-102 表示恒温器断开了 Wi-Fi。"],["Move the router closer, then wait two minutes.", "把路由器挪近一点，再等两分钟。"]]],
   [["E-103", "E-103"],[["E-103 means the thermostat cannot reach the heating system.", "E-103 表示恒温器连不上供暖系统。"],["Call the company that installed your heating.", "请联系安装供暖系统的师傅。"]]],
   [["E-104", "E-104"],[["E-104 means the thermostat cannot read the room temperature.", "E-104 表示恒温器读不到室温。"],["The sensor cable is loose, or something covers the sensor.", "传感器的线松了，或者被挡住了。"],["Make sure nothing hangs over the thermostat, then restart it.", "确认上方没有遮挡，再重启它。"]]],
   [["E-105", "E-105"],[["E-105 means the room is warming up too slowly.", "E-105 表示房间升温太慢。"],["Close the windows and open the radiator valves.", "关好窗户，打开暖气片的阀门。"]]],
   [["E-201", "E-201"],[["E-201 means the lock motor is blocked.", "E-201 表示门锁的电机卡住了。"],["The bolt cannot slide fully into the door frame.", "锁舌没法完全伸进门框。"],["Close the door all the way and check that the strike plate lines up with the bolt.", "把门完全关好，再检查门框上的锁扣板是否对准锁舌。"]]],
   [["E-202", "E-202"],[["E-202 means the fingerprint sensor is dirty.", "E-202 表示指纹传感器脏了。"],["Wipe it with a dry cloth.", "用干布擦擦。"]]],
   [["E-203", "E-203"],[["E-203 means the keypad is locked after five wrong codes in a row.", "E-203 表示连续输错五次密码，键盘被锁住了。"],["Wait ten minutes, or unlock it in the app.", "等十分钟，或者在 App 里解锁。"]]],
   [["E-204", "E-204"],[["E-204 means the handle is loose.", "E-204 表示把手松了。"],["Tighten the two screws on the handle.", "拧紧把手上的两颗螺丝。"]]],
   [["E-205", "E-205"],[["E-205 means the door is not fully closed.", "E-205 表示门没有关严。"],["Push the door shut until you hear a click.", "把门推紧，听到咔嗒一声就好。"]]],
   [["E-301", "E-301"],[["E-301 means a wheel of the robot vacuum is stuck.", "E-301 表示扫地机器人的轮子卡住了。"],["Check the wheels for thread or a sock, then push each wheel by hand.", "检查轮子上有没有缠着线头或袜子，再用手推一推每个轮子。"]]],
   [["E-302", "E-302"],[["E-302 means the dust bin is missing.", "E-302 表示尘盒没装上。"],["Put the bin back until it clicks.", "把尘盒装回去，听到咔嗒一声就好。"]]],
   [["E-303", "E-303"],[["E-303 means the robot was lifted off the floor.", "E-303 表示机器人被抬离了地面。"],["Put it back on the floor and press start.", "把它放回地面，按开始键。"]]],
   [["E-304", "E-304"],[["E-304 means a cliff sensor is dirty.", "E-304 表示防跌落传感器脏了。"],["Wipe the sensors under the robot.", "擦一擦机器人底部的传感器。"]]],
   [["E-305", "E-305"],[["E-305 means the main brush of the robot vacuum is stuck.", "E-305 表示扫地机器人的主刷卡住了。"],["Usually hair or a cable is wrapped around it.", "通常是头发或电线缠在了上面。"],["Turn the robot over, take out the brush and clean it.", "把机器人翻过来，取下主刷清理干净。"]]]
 ]],
 ["W",["Warranty policy", "保修政策"],[
   [["Coverage", "保修范围"],[["Every Lumo device has a two-year warranty from the date of purchase.", "小鹿家居的每台设备，从购买之日起保修两年。"],["It covers faults in materials and workmanship.", "保修范围是材料和做工上的故障。"],["Batteries are covered for six months only.", "电池只保修六个月。"]]],
   [["Exclusions", "不保修的情况"],[["Damage from water, drops or fire is not covered.", "进水、摔落或火烧造成的损坏不保修。"],["A device that was opened or repaired by anyone other than Lumo is not covered either.", "由小鹿家居以外的人拆开或修理过的设备，也不保修。"]]],
   [["Making a claim", "申请保修"],[["To claim, first find your receipt, which proves the date of purchase.", "申请保修前，先找到购买凭证，它能证明购买日期。"],["Then take a photo of the receipt and of the serial number on the device.", "然后拍下凭证和设备上序列号的照片。"],["Send both photos to support in the app, and we ship a replacement within five working days.", "在 App 里把两张照片发给客服，我们会在五个工作日内寄出换新的设备。"]]]
 ]]
];
/* questions: [en, zh, doc, section, first sentence, last sentence, kind] */
const QS=[
 ["What does E-201 mean?", "E-201 是什么意思？", "E", 6, 0, 0, "code"],
 ["What does E-104 mean?", "E-104 是什么意思？", "E", 4, 0, 0, "code"],
 ["What does E-305 mean?", "E-305 是什么意思？", "E", 15, 0, 0, "code"],
 ["Nothing happens at the door and I am outside. What now?", "门没反应了，我进不去，怎么办？", "L", 1, 0, 1, "para"],
 ["How long does a set of lock batteries last?", "门锁的电池能用多久？", "L", 0, 1, 1, "plain"],
 ["My lock shows E-203. What happened?", "门锁显示 E-203，怎么回事？", "E", 8, 0, 0, "code"],
 ["Can I make the door wait longer before it locks again?", "门关上后，能让它晚一点再自动上锁吗？", "L", 3, 1, 2, "plain"],
 ["Which photos do I send for a warranty claim?", "申请保修要拍哪些照片？", "W", 2, 1, 1, "plain"],
 ["My vacuum fell down the stairs. Will you fix it for free?", "扫地机器人从楼梯上摔下来了，能免费修吗？", "W", 1, 0, 0, "para"],
 ["How long is the warranty?", "保修期多长？", "W", 0, 0, 0, "plain"],
 ["Does the warranty cover batteries?", "电池在保修范围内吗？", "W", 0, 2, 2, "distract"],
 ["The thermostat shows the wrong room temperature. What should I check?", "恒温器显示的室温不对，该检查什么？", "E", 4, 1, 2, "para"],
 ["The robot stopped and its brush will not turn.", "扫地机器人停了，刷子转不动。", "E", 15, 1, 2, "para"],
 ["The bolt will not go all the way into the frame.", "锁舌伸不进门框。", "E", 6, 1, 2, "plain"],
 ["How do I switch off auto-lock?", "怎么关掉自动上锁？", "L", 3, 2, 2, "plain"],
 ["A local shop repaired my lock. Is it still under warranty?", "门锁在外面的店里修过，还能保修吗？", "W", 1, 1, 1, "para"],
 ["What does E-302 mean?", "E-302 是什么意思？", "E", 12, 0, 0, "code"],
 ["Do I need to change the batteries of the thermostat?", "恒温器要换电池吗？", "T", 2, 0, 1, "distract"],
 ["How long does the robot clean on one charge?", "扫地机器人充一次电能扫多久？", "V", 0, 1, 1, "plain"],
 ["How often should I clean the brush?", "主刷多久清理一次？", "V", 1, 0, 0, "distract"]
];
/* the small encoder's word list: [meaning, English forms, Chinese words]; each meaning is one dimension */
const LEX=[
 ["battery", "battery batteries 9-volt aa", "电池 电量"],
 ["power", "dead low out nothing", "没电 不足 没反应"],
 ["replace", "replace replacement", "换 换新"],
 ["key", "key", "钥匙"],
 ["entry", "entry inside outside", "进门 进去 门外 进不去"],
 ["pin", "pin keypad type enter", "密码 键盘 输入"],
 ["lock", "lock locked unlock", "门锁 锁住 解锁"],
 ["door", "door", "门"],
 ["finger", "fingerprint finger", "指纹 手指"],
 ["sensor", "sensor", "传感器"],
 ["app", "app", "app"],
 ["add", "add record records", "添加 录"],
 ["relock", "locks auto-lock", "上锁 锁门"],
 ["wait", "wait waits seconds minutes longer", "等待 秒 分钟 晚一点"],
 ["duration", "long last lasts months years", "多久 多长 能用 个月 两年"],
 ["setting", "settings switch off", "设置 关掉"],
 ["code", "code codes digit digits e-101 e-102 e-103 e-104 e-105 e-201 e-202 e-203 e-204 e-205 e-301 e-302 e-303 e-304 e-305", "错误码 数字 e-101 e-102 e-103 e-104 e-105 e-201 e-202 e-203 e-204 e-205 e-301 e-302 e-303 e-304 e-305"],
 ["mean", "mean means", "表示 意思"],
 ["thermostat", "thermostat", "恒温器"],
 ["temperature", "temperature", "室温 温度"],
 ["check", "check restart", "检查 重启 确认"],
 ["cable", "cable loose wires", "线 松 电线 保险丝"],
 ["stuck", "stuck blocked blocks hangs", "卡住 挡住 遮挡 转不动"],
 ["motor", "motor", "电机"],
 ["bolt", "bolt", "锁舌"],
 ["frame", "frame", "门框"],
 ["plate", "plate strike", "锁扣板"],
 ["vacuum", "vacuum robot", "扫地机器人 机器人"],
 ["brush", "brush", "刷子 主刷"],
 ["hair", "hair wrapped", "头发 缠"],
 ["clean", "clean cleans cleaning", "清理 干净 清扫 扫"],
 ["turn", "turn turning", "转 转动"],
 ["warranty", "warranty covered covers free", "保修 保修期 免费"],
 ["purchase", "purchase bought", "购买"],
 ["fault", "faults fault", "故障"],
 ["damage", "damage drops fell water fire", "损坏 摔落 摔 进水 火烧"],
 ["repair", "repaired opened fix", "修理 修过 拆开 修"],
 ["receipt", "receipt proves", "凭证 证明"],
 ["photo", "photo photos", "照片 拍"],
 ["serial", "serial", "序列号"],
 ["support", "support", "客服"],
 ["claim", "claim", "申请"],
 ["schedule", "schedule", "日程"],
 ["time", "time day", "时间 时段"],
 ["heat", "heating warm warmer", "供暖 暖"],
 ["away", "eco nobody home leave", "节能 没人 离开 在家"],
 ["phone", "phone location", "手机 位置"],
 ["screen", "screen dark", "屏幕 不亮"],
 ["charge", "charge charging dock", "充电 充 底座"],
 ["freq", "often once week month", "每周 每月 多久一次 一次"],
 ["bin", "bin dust filter", "尘盒 灰尘 滤网"],
 ["network", "wi-fi router connection", "wi-fi 路由器 断开"],
 ["wheel", "wheel wheels sock thread", "轮子 袜子 线头"]
];
const K1=1.2, BB=0.75, RRF_K=60, NCAND=5, RR_MS=0.2;   // BM25 as Lucene's defaults; RRF constant 60; the reranker reads up to 5 candidates at 0.2 ms per token (the lab's assumption)
/* English keyword terms leave out Lucene's English stop words (EnglishAnalyzer's default set) */
const STOP=new Set('a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with'.split(' '));
const SIZES=[16,32,64,128,'h'], OVF=[0,0.25,0.5], RETS=['kw','vec','hyb'], KS=[1,3,5], LAT={kw:5,vec:10,hyb:15};
const NQ=QS.length, DIM=LEX.length, DOC_IDS=DOCS.map(d=>d[0]);
const LI=lang=>lang==='zh'?1:0;
/* tokens: English words (codes such as E-104 count as one word); Chinese: every character is a token, runs of Latin letters and digits
   (E-104, App, 30) are one token each. Each token keeps the text after it (spaces, punctuation) for display. */
const EN_RE=/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g, ZH_RE=/[一-鿿]|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*/g;
function pieces(s,lang){const ms=[...s.matchAll(lang==='zh'?ZH_RE:EN_RE)];
  return ms.map((m,i)=>[m[0],s.slice(m.index+m[0].length,i+1<ms.length?ms[i+1].index:s.length),i===ms.length-1]);}
const isCJK=t=>t.charCodeAt(0)>=0x4e00&&t.charCodeAt(0)<=0x9fff;
const EN_MAP=new Map(), ZH_MAP=new Map(); let ZMAX=1;
LEX.forEach((c,i)=>{c[1].split(' ').forEach(w=>EN_MAP.set(w,i)); c[2].split(' ').forEach(w=>{ZH_MAP.set(w.toLowerCase(),i); ZMAX=Math.max(ZMAX,w.length);});});
const builtC={};
/* a document as one stream of tokens: each heading, then its sentences; the start and end of every section and sentence */
function build(lang){if(builtC[lang]) return builtC[lang]; const li=LI(lang);
  const docs=DOCS.map(d=>{const toks=[], secs=[], sents=[];
    d[2].forEach(sec=>{const s0=toks.length; pieces(sec[0][li],lang).forEach(p=>toks.push({t:p[0],tl:p[1],end:p[2],h:true}));
      const row=[]; sec[1].forEach(x=>{const a=toks.length; pieces(x[li],lang).forEach(p=>toks.push({t:p[0],tl:p[1],end:p[2],h:false})); row.push([a,toks.length]);});
      sents.push(row); secs.push([s0,toks.length]);});
    return {id:d[0],toks,secs,sents};});
  return (builtC[lang]=docs);}
/* keyword terms: English words in lower case; Chinese: every two adjacent characters of a run (a run ends at punctuation, at the end of
   a sentence or heading, or at a Latin token), a lone character alone; Latin tokens in lower case */
function kwTerms(toks,lang){
  if(lang!=='zh') return toks.map(x=>x.t.toLowerCase()).filter(w=>!STOP.has(w));
  const out=[]; let run=[];
  const flush=()=>{if(run.length===1) out.push(run[0]); for(let i=0;i+1<run.length;i++) out.push(run[i]+run[i+1]); run=[];};
  toks.forEach(x=>{if(isCJK(x.t)){run.push(x.t); if(x.tl.trim()||x.end) flush();} else {flush(); out.push(x.t.toLowerCase());}});
  flush(); return out;}
/* the small encoder: each word found in the word list adds 1 to its meaning's dimension (Chinese: longest match first, up to ZMAX
   characters, never across punctuation); the vector is then scaled to length 1 */
function concepts(toks,lang){const v=new Array(DIM).fill(0);
  if(lang!=='zh'){toks.forEach(x=>{const i=EN_MAP.get(x.t.toLowerCase()); if(i!=null) v[i]++;}); return v;}
  let s='';
  toks.forEach(x=>{if(isCJK(x.t)) s+=x.t; else {const i=ZH_MAP.get(x.t.toLowerCase()); if(i!=null) v[i]++; s+='|';} if(x.tl.trim()||x.end) s+='|';});
  let i=0;
  while(i<s.length){let hit=0; for(let n=Math.min(ZMAX,s.length-i);n>0;n--){const w=s.slice(i,i+n); if(ZH_MAP.has(w)){v[ZH_MAP.get(w)]++; i+=n; hit=1; break;}} if(!hit) i++;}
  return v;}
const idsOf=toks=>new Set(toks.filter(x=>/^[A-Za-z]-\d+$/.test(x.t)).map(x=>x.t.toLowerCase()));
function unit(v){const n=Math.sqrt(v.reduce((a,x)=>a+x*x,0)); return n?v.map(x=>x/n):v.slice();}
const ovTok=(size,ov)=>size==='h'?0:Math.round(size*OVF[ov]);
/* cut every document: fixed size with overlap (each chunk starts size − overlap tokens after the last; the last one reaches the end),
   or one chunk per section ('h') */
function cut(lang,size,ov){const docs=build(lang), out=[];
  docs.forEach((d,di)=>{const n=d.toks.length;
    if(size==='h'){d.secs.forEach((s,si)=>out.push({d:di,a:s[0],b:s[1],sec:si})); return;}
    const step=size-ovTok(size,ov); for(let s=0;;s+=step){out.push({d:di,a:s,b:Math.min(s+size,n)}); if(s+size>=n) break;}});
  out.forEach((c,j)=>{c.j=j; c.n=c.b-c.a; const tk=docs[c.d].toks.slice(c.a,c.b); c.terms=kwTerms(tk,lang); c.cv=concepts(tk,lang); c.v=unit(c.cv); c.ids=idsOf(docs[c.d].sents.flat().filter(([a,b])=>a>=c.a&&b<=c.b).flatMap(([a,b])=>docs[c.d].toks.slice(a,b)));   // codes in whole sentences only
    c.no=out.filter(x=>x.d===c.d&&x.j<=j).length;});
  return out;}
const qToks=(qi,lang)=>pieces(QS[qi][LI(lang)],lang).map(p=>({t:p[0],tl:p[1],end:p[2],h:false}));
function order(sc){return sc.map((s,j)=>j).sort((a,b)=>sc[b]-sc[a]||a-b);}
/* one chunking: every chunk's scores for every question, the three rankings, the rerank, and where the right chunk lands */
const setC=new Map();
function setup(lang,size,ov){const key=[lang,size,ov].join('|'); let R=setC.get(key); if(R) return R;
  const docs=build(lang), C=cut(lang,size,ov), N=C.length, avg=C.reduce((a,c)=>a+c.terms.length,0)/N, df=new Map();
  C.forEach(c=>new Set(c.terms).forEach(t=>df.set(t,(df.get(t)||0)+1)));
  const idf=t=>Math.log(1+(N-df.get(t)+0.5)/(df.get(t)+0.5));
  const Q=QS.map((q,qi)=>{const qt=qToks(qi,lang), terms=[...new Set(kwTerms(qt,lang))], qc=concepts(qt,lang), qv=unit(qc), qids=idsOf(qt);
    const kw=C.map(c=>{let s=0; const dl=c.terms.length; terms.forEach(t=>{const f=c.terms.filter(x=>x===t).length; if(f) s+=idf(t)*f/(f+K1*(1-BB+BB*dl/avg));}); return s;});
    const vec=C.map(c=>c.v.reduce((a,x,i)=>a+x*qv[i],0));
    const ok=order(kw), ovv=order(vec), rrf=new Array(N).fill(0);
    [[ok,kw],[ovv,vec]].forEach(([lst,sc])=>lst.filter(j=>sc[j]>0).slice(0,NCAND).forEach((j,r)=>{rrf[j]+=1/(RRF_K+r+1);}));   // RRF over each search's first 5
    const ret=(l,sc)=>l.filter(j=>sc[j]>0), lists={kw:ret(ok,kw),vec:ret(ovv,vec),hyb:ret(order(rrf),rrf)}, qset=qc.map((x,i)=>x?i:-1).filter(i=>i>=0), den=Math.max(1,qset.length+qids.size);
    const cov=C.map(c=>(qset.filter(i=>c.cv[i]>0).length+[...qids].filter(x=>c.ids.has(x)).length)/den);
    // a list holds only the chunks a search returns (score above 0). Rerank reads the first 5 of them: more coverage first; equal coverage,
    // the closer vector first; then the first search's order. Below the 5 comes the rest of what was returned, in the first search's order.
    const rr={}, nc={}; RETS.forEach(r=>{const l=lists[r], cand=l.slice(0,NCAND), pos=new Map(cand.map((j,i)=>[j,i]));
      cand.sort((a,b)=>cov[b]-cov[a]||vec[b]-vec[a]||pos.get(a)-pos.get(b)); rr[r]=cand.concat(l.slice(NCAND)); nc[r]=cand.length;});
    const d=DOC_IDS.indexOf(q[2]), sa=docs[d].sents[q[3]][q[4]][0], sb=docs[d].sents[q[3]][q[5]][1];
    const right=C.filter(c=>c.d===d&&c.a<=sa&&c.b>=sb).map(c=>c.j);
    const part=C.filter(c=>c.d===d&&c.a<sb&&c.b>sa).map(c=>c.j);   // chunks holding some of the answer
    const rank=lst=>{const p=right.map(j=>lst.indexOf(j)).filter(x=>x>=0); return p.length?Math.min(...p):-1;};   // -1: not returned, or no chunk holds the whole answer
    const rk={}; RETS.forEach(r=>{rk[r+'0']=rank(lists[r]); rk[r+'1']=rank(rr[r]);});
    return {kw,vec,rrf,cov,lists,rr,nc,right,part,d,sa,sb,rk,terms,qc,qids};});
  const longest=Math.max(...C.map(c=>c.n));
  R={lang,size,ov,C,N,avg,df,Q,longest,splits:Q.filter(q=>!q.right.length).length};
  setC.set(key,R); return R;}
/* the measures of a setting: recall@k over the 20 questions, context tokens (k × the longest chunk), latency (the lab's assumption) */
function measure(lang,s){const R=setup(lang,s.size,s.ov), key=s.ret+(s.rr?1:0);
  const hits=R.Q.filter(q=>q.rk[key]>=0&&q.rk[key]<s.k).length;
  // rerank time: the candidates the first retrieval really returned (up to 5), each padded to the longest chunk; the slowest question counts
  const nc=s.rr?Math.max(...R.Q.map(q=>q.nc[s.ret])):0, rerankMs=nc*R.longest*RR_MS;
  return {R,hits,rec:hits/NQ,ctx:s.k*R.longest,lat:LAT[s.ret]+rerankMs,rerankMs,nc,splits:R.splits};}
const CH_GOAL=0.95, CH_CTX=200, CH_LAT=80, CH_SIZES=[16,32,64,128];
function chAll(lang){const out=[]; CH_SIZES.forEach(size=>[0,1,2].forEach(ov=>RETS.forEach(ret=>[0,1].forEach(rr=>KS.forEach(k=>{const s={size,ov,ret,rr,k}, m=measure(lang,s);
  out.push({...s,rec:m.rec,ctx:m.ctx,lat:m.lat,ok:m.rec>=CH_GOAL-1e-9&&m.ctx<=CH_CTX&&m.lat<=CH_LAT+1e-9});})))));
  const okp=out.filter(p=>p.ok), minC=okp.length?Math.min(...okp.map(p=>p.ctx)):Infinity; out.forEach(p=>{p.best=p.ok&&p.ctx===minC;}); return out;}
/* ENGINE-END */

/* ---------- words and helpers ---------- */
const f=(v,dg)=>A.fmt.num(v,dg), pc=p=>f(p*100,0)+'%';
const LG=()=>A.lang()==='zh'?'zh':'en', li=()=>LI(LG());
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn ckr-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="ckr-math-line">${A.ui.math(t)}</div>`;
const MI=t=>A.ui.math(t);
const KL=()=>MI('k \\times L');
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(!l) return esc(id); if(C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return A.shell.labLink?A.shell.labLink(id):esc(id);};
const quote=s=>`“${esc(s)}”`;
const qText=i=>QS[i][li()];
const DOC_N={L:{en:'Lock guide',zh:'门锁说明'},T:{en:'Thermostat guide',zh:'恒温器说明'},V:{en:'Vacuum guide',zh:'扫地机器人说明'},E:{en:'Error codes',zh:'错误码'},W:{en:'Warranty',zh:'保修政策'}};
const RET_N={kw:{en:'Keyword search (BM25)',zh:'关键词检索（BM25）'},vec:{en:'Vector search',zh:'向量检索'},hyb:{en:'Hybrid search (RRF)',zh:'混合检索（RRF）'}};
const RET_S={kw:{en:'keyword',zh:'关键词'},vec:{en:'vector',zh:'向量'},hyb:{en:'hybrid',zh:'混合'}};
const sizeTxt=s=>s==='h'?tx('by heading','按标题切'):tx(`${s} tokens`,`每块 ${s} 个 token`);
const sizeShort=s=>s==='h'?tx('heading','按标题'):String(s);
const ovTxt=(size,ov)=>size==='h'?tx('no overlap','不重叠'):(ov===0?tx('no overlap','不重叠'):tx(`overlap ${ovTok(size,ov)} tokens`,`重叠 ${ovTok(size,ov)} 个 token`));
const OV_N=[{en:'None',zh:'不重叠'},{en:'¼ of the chunk',zh:'1/4 块'},{en:'½ of the chunk',zh:'半块'}];
const msTxt=v=>tx(`${f(v,0)} ms`,`${f(v,0)} 毫秒`);
const tokTxt=n=>tx(`${A.fmt.int(n)} tokens`,`${A.fmt.int(n)} 个 token`);
/* a rank in words: 1st place is "first"; -1 means the search did not return the chunk */
const rkTxt=r=>r<0?tx('does not return it','不返回它'):tx(`ranks it ${r+1}`,`排第 ${r+1}`);
function setTxt(s){return tx(`${sizeTxt(s.size)}${s.size==='h'?'':', '+ovTxt(s.size,s.ov)}, ${L(RET_S[s.ret])} search${s.rr?' + rerank':''}, keep ${s.k}`,
  `${sizeTxt(s.size)}${s.size==='h'?'':'，'+ovTxt(s.size,s.ov)}，${L(RET_S[s.ret])}检索${s.rr?' + rerank':''}，${s.k===1?'取 1 块':`取前 ${s.k} 块`}`);}
const docRange=()=>{const n=build(LG()).map(d=>d.toks.length); return [Math.min(...n),Math.max(...n)];};
const zhMore=()=>{const sum=l=>build(l).reduce((a,d)=>a+d.toks.length,0); return Math.round((sum('zh')/sum('en')-1)*20)*5;};   // to the nearest 5%
const D_CHUNK=()=>tx('Chunking: a document is cut into pieces, chunks, before it is stored in the index; search returns chunks, not whole documents. Chunk size: how many tokens a chunk holds. Overlap: how many tokens two neighbouring chunks share.','切块：文档先切成一块块，再存进索引；检索返回的是块，不是整篇文档。块大小：一块有多少个 token。重叠：相邻两块共用多少个 token。');
const D_TOK=()=>tx('Here a token is an English word (codes such as E-104 count as one) or one Chinese character: the lab’s own counting rule. Real tokenizers cut differently.','这里的 token，英文是一个词（E-104 这类编号算一个），中文是一个字：这是本 Lab 自己的数法，真实的 tokenizer 切得不一样。');
const D_REC=()=>tx('Recall@k here: of the 20 questions, the share whose whole answer lies inside one of the k chunks kept. The lab counts only a chunk that holds all of the answer, a strict rule: two halves kept in two chunks do not count, because with k = 1 only one of them can be kept.','这里的 recall@k：20 个问题里，完整答案落在取回的某一块里的，占几成。本 Lab 只算装着完整答案的那一块，这是比较严的算法：两半分别在两块里，也不算，因为只取 1 块时最多只能拿回其中一半。');
const D_ENC=()=>tx(`The small encoder of this lab is written by hand, not trained: a word list puts each word under one of ${DIM} meanings (“fell”, “drops” and “damage” are one meaning), and a text’s vector counts how often each meaning occurs, scaled to length 1. Every error code falls under the one meaning “error code”.`,`这个 Lab 的小编码器是手写的，没有训练：一张词表把每个词归到 ${DIM} 个意思之一（“摔”“摔落”“损坏”算同一个意思），一段文字的向量就是每个意思出现了几次，再缩放成长度 1。所有错误码都归到“错误码”这一个意思。`);
const D_CE=()=>tx('A cross-encoder is a Transformer that reads the question and one passage together, in one input, and outputs one relevance score. Because it sees both at once it can check that the passage answers the question, but it must run once per passage.','cross-encoder 是一个 Transformer：把问题和一段文字拼成一个输入一起读，输出一个相关度分数。它同时看到两者，所以能判断这段文字是不是在回答这个问题；代价是每一段文字都要单独跑一次。');
const D_STAND=()=>tx('The lab’s stand-in for a cross-encoder lists the question’s meanings (the small encoder’s meanings) and its codes such as E-201, and scores each candidate by the share of them it contains. A code counts only inside a whole sentence of the chunk, letter for letter. When two candidates score the same, the one whose vector is closer to the question comes first. It cannot read negation or word order, which a real cross-encoder can.','本 Lab 用来代替 cross-encoder 的替身：列出问题里的意思（小编码器的那些意思）和 E-201 这样的编号，看每个候选里有几成，就打几成。编号要在这一块的完整句子里、逐字一样才算。两个候选得分一样时，向量和问题更接近的排前面。它读不懂否定和词序，真实的 cross-encoder 能读懂。');
const D_CTX=()=>tx(`The context: the prompt the model reads before it answers. The chunks kept go into it, so k chunks of up to L tokens take up to ${KL()} tokens of it on every question; L is the length of the longest chunk.`,`上下文：模型回答前读到的那段 prompt。取回的块放在里面，所以 k 块、每块最多 L 个 token，每个问题都要占掉最多 ${KL()} 个 token；L 是最长那一块的 token 数。`);

/* ---------- state ---------- */
const DEFAULT={size:64,ov:0,ret:'vec',rr:0,k:1,q:5};
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0;
const seenCut=new Set(), shownPrev={}, tried=new Set();
let speed='mid', anim=null, raf=0, lastTs=0;
const DUR={slow:4.8,mid:2.4,fast:1.2};   // seconds to cut one document (R3.9: a step-by-step mechanism defaults to 中)
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const cutKey=s=>[LG(),s.size,s.size==='h'?0:s.ov,QS[s.q][2]].join('|');
const sameSet=(a,b)=>['size','ov','ret','rr','k'].every(x=>a[x]===b[x]);

/* ---------- guided parts ---------- */
const PRED=['kw','vec','both','none'];
const predTxt=v=>({kw:tx('Keyword search puts it first; vector search does not','关键词检索把它排第一，向量检索不会'),vec:tx('Vector search puts it first; keyword search does not','向量检索把它排第一，关键词检索不会'),both:tx('Both put it first','两种都把它排第一'),none:tx('Neither puts it first','两种都不会把它排第一')})[v];
const segIn=(key,label,opts,val)=>A.ui.seg({id:'ckr-ic-'+key,key,label,options:opts,value:String(val),inline:true});
const inSize=(sizes)=>segIn('size',tx('Chunk size','块大小'),(sizes||SIZES).map(s=>[String(s),sizeShort(s)]),S.size);
const inOv=()=>segIn('ov',tx('Overlap','重叠'),OV_N.map((o,i)=>[String(i),L(o),S.size==='h']),S.ov);
const inRet=()=>segIn('ret',tx('First retrieval','先用哪种检索'),RETS.map(r=>[r,L(RET_S[r])]),S.ret);
const inRr=()=>segIn('rr',tx('Rerank','rerank'),[['0',tx('Off','不用')],['1',tx(`On: rescore the top ${NCAND}`,`用：给前 ${NCAND} 块重新打分`)]],S.rr);
const inK=()=>segIn('k',tx('Chunks kept, k','取前 k 块'),KS.map(k=>[String(k),String(k)]),S.k);
const P4Q=[0,8,9], P5Q=[17,0];
const qShort=i=>({0:tx('E-201','E-201'),8:tx('Fell down the stairs','摔下楼梯'),9:tx('Warranty length','保修期多长'),17:tx('Thermostat batteries','恒温器的电池')})[i]||String(i);
const inQ=qs=>segIn('q',tx('A question','一个问题'),qs.map(i=>[String(i),qShort(i)]),S.q);
const BM25_TEX=nm=>`\\text{${nm}} = \\sum_{t \\in q} \\mathrm{idf}(t) \\cdot \\frac{f_t}{f_t + k_1 \\, n_d}`;
const NORM_TEX=`n_d = 1 - b + b \\, \\frac{|d|}{\\mathrm{avgdl}}`;
const RRF_TEX=nm=>`\\text{${nm}} = \\frac{1}{c + r_{\\text{kw}}} + \\frac{1}{c + r_{\\text{vec}}}`;
/* questions that rank the answer first, or anywhere in the top NCAND, with a first search alone (64 tokens, half a chunk of overlap) */
const firstN=(R,r)=>R.Q.filter(q=>q.rk[r+'0']===0).length, poolN=(R,r)=>R.Q.filter(q=>q.rk[r+'0']>=0&&q.rk[r+'0']<NCAND).length;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'The model only sees the chunks retrieval returns.',zh:'模型只看得到检索返回的那几块。'},
  panels:['doc','lists'],cols:()=>[S.ret+'0'],enter:()=>({...DEFAULT}),
  body:()=>{const m=measure(LG(),S), low=NQ-m.hits-m.splits;
   return tx(`<p>The help center of Lumo Home, a smart-home company made up for this lab, answers customers with retrieval-augmented generation (RAG): search the documents for the question, put the passages found into the model’s prompt, and let the model answer from them. The model can only use what search puts in front of it.</p><p>The documents are first cut into ${DEF('chunks',D_CHUNK())}. Why cut: a whole document is long, and putting it into every prompt costs tokens; it also covers many topics, so one vector for all of it blurs them. With chunks, search puts only the few passages about the question into the prompt. Here every chunk holds 64 ${DEF('tokens',D_TOK())}, and search keeps only the chunk it ranks first. A customer writes ${quote(qText(DEFAULT.q))} The answer is one sentence on the error-code page, and the cut falls inside it (red, ${J('doc','in the document strip')}): one chunk ends with the code, the next holds what the code means but not the code. Keeping one chunk, the model sees the code or its meaning, never both.</p><p>This lab counts a question as found only when one kept chunk holds the whole answer. With this setup that is ${m.hits} of the 20 questions written for this lab; the recall@1 at the top of the stage is this share.</p><p>Of the others, ${m.splits} are cut apart and ${low} rank below first. The list under the document strip shows the 5 chunks vector search ranks highest for this question, each with its score on the right; only the first goes into the prompt. The parts that follow look at the three steps where an answer can go missing: how the documents are cut, how the chunks are scored, and how many chunks are kept.</p>`,
             `<p>小鹿家居（为这个 Lab 编的一家智能家居公司）的帮助中心用检索增强生成（RAG）回答用户的问题：先拿问题去搜文档，把搜到的段落放进模型的 prompt，再让模型照着回答。模型只能用检索放到它面前的东西。</p><p>文档先要${DEF('切成一块块',D_CHUNK())}。为什么要切：一整篇文档太长，每次都放进 prompt 太贵；一篇里讲很多件事，整篇变成一个向量，意思也被冲淡。切成小块以后，检索只把和问题有关的那几块放进 prompt。这里每块 64 个 ${DEF('token',D_TOK())}，检索先只取排在第一的那一块。用户问${quote(qText(DEFAULT.q))}答案是错误码那页的一句话，一刀正好切在这句里面（${J('doc','文档条')}里标红的地方）：前一块以编号结尾，后一块装着编号的意思，却没有编号。只取一块，模型要么只看到编号，要么只看到意思。</p><p>本 Lab 只在取回的某一块装着完整答案时，才算这个问题找到了。按这个设置，为这个 Lab 写的 20 个问题里找到了 ${m.hits} 个，舞台上方的 recall@1 就是这个比例。</p><p>其余的问题里，${m.splits} 个答案被切开，${low} 个没排到第一。文档条下面的列表，是向量检索给这个问题排出的前 5 块，右边的数是每一块的分数；只有第 1 块会放进 prompt。后面几部分依次看答案会在哪三步丢掉：文档怎么切，块怎么打分，最后取几块。</p>`);}},
 {id:'chunk',title:{en:'How to cut',zh:'怎么切'},lead:{en:'Chunk size and overlap decide whether an answer stays whole.',zh:'块大小和重叠决定答案会不会被切开。'},
  panels:['doc','metrics'],enter:()=>({...DEFAULT}),expect:()=>tried.size>1,showMe:{size:64,ov:2},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another chunk size or overlap to continue.',zh:'换一种块大小或重叠再继续。'},
  body:()=>{const n0=setup(LG(),64,0).N, n2=setup(LG(),64,2).N;
   return tx(`<p>A fixed-size cut ignores sentences: every chunk is the next L tokens, L the chunk size. With overlap, each chunk starts before the previous one ends, so neighbours share some tokens; an answer no longer than the overlap then always lies whole inside some chunk. Overlap has two costs. There are more chunks to store: here ${n0} without overlap and ${n2} with half a chunk shared (close to twice as many for long documents; these are short). And neighbouring chunks are half the same text, so they often rank side by side and put the same passage into the context twice. Cutting by heading keeps each section whole, but sections differ in length.</p><p>The overlap is set as a share of the chunk size: none, a quarter of a chunk, or half a chunk (with 64-token chunks, 16 or 32 tokens shared). Pick a chunk size or an overlap and watch the same answer in ${J('doc','the document strip')}.</p>`,
              `<p>按固定大小切，不管句子：每一块就是接下来的 L 个 token，L 是块大小。有重叠时，每一块在上一块结束之前就开始，相邻两块共用一些 token；这样，长度不超过重叠的答案，一定完整地落在某一块里。重叠有两个代价。块变多了：这里不重叠是 ${n0} 块，重叠半块是 ${n2} 块（文档长时接近翻倍，这里的文档短）。而且相邻两块有一半内容相同，常常一起排进前几名，同一段话在上下文里占了两份。按标题切能让每一节保持完整，但每节长短不一。</p><p>重叠按块大小的比例选：不重叠、1/4 块或半块，也就是相邻两块共用块大小的四分之一或一半（块大小 64 时，共用 16 或 32 个 token）。选一种块大小或重叠，看${J('doc','文档条')}里同一个答案怎么样。</p>`)+inSize()+inOv();},
  obs:()=>obsChunk()},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'An error code: which search finds it?',zh:'一个错误码：哪种检索找得到？'},
  panels:['doc'],enter:()=>({size:64,ov:2,ret:'vec',rr:0,k:1,q:0}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>From here on every chunk holds 64 tokens and neighbours overlap by half a chunk (32 tokens), so no answer here is cut apart. Two ways to score a chunk:</p><ul><li><b>Keyword search</b> counts the words a chunk shares with the question, letter for letter; a word that appears in few chunks counts more. The commonest English words (“the”, “is”, “of” and so on) are left out.</li><li><b>Vector search</b> compares meanings: both texts become ${DEF('vectors',D_ENC())} and the chunk with the most similar direction wins (see ${LL('embeddings')}). In this lab’s small encoder every error code, E-201 as much as E-305, is just “an error code”.</li></ul><p>A customer asks ${quote(qText(0))} Only one section of the help center explains E-201. Which search puts the chunk with that section first?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
             `<p>从这里起，每块 64 个 token，相邻两块重叠半块（32 个 token），所以这里没有答案被切开。给一块打分有两种办法：</p><ul><li><b>关键词检索</b>数这一块和问题有几个词逐字一样（中文没有空格，本 Lab 把相邻的两个字算一个词）；越少的块里出现的词，算得越重。</li><li><b>向量检索</b>比意思：两段文字都变成${DEF('向量',D_ENC())}，方向最接近的那块胜出（见 ${LL('embeddings')}）。在这个 Lab 的小编码器里，每个错误码（不管是 E-201 还是 E-305）都只是“一个错误码”。</li></ul><p>用户问${quote(qText(0))}帮助中心里只有一节讲 E-201。哪种检索会把装着这一节的块排在第一？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'search',title:{en:'Keyword, vector, hybrid',zh:'关键词、向量、混合'},lead:{en:'Each search misses what the other catches; RRF merges them.',zh:'两种检索各有漏掉的；RRF 把排名合起来。'},
  panels:['lists','metrics'],cols:()=>['kw0','vec0','hyb0'],enter:()=>({size:64,ov:2,ret:'hyb',rr:0,k:1,q:0}),expect:()=>tried.size>1,showMe:{q:8},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another question to continue.',zh:'换一个问题再继续。'},
  body:()=>{const R=setup(LG(),64,2), q=R.Q[0], ok=pred==='kw';
   const ans=tx(`The answer: keyword search puts the E-201 chunk first, vector search does not. Vector search ${rkTxt(q.rk.vec0)}, because to the small encoder every error code means the same.`,`正确答案是“关键词检索把它排第一，向量检索不会”：向量检索把讲 E-201 的块${rkTxt(q.rk.vec0)}，因为在小编码器眼里，每个错误码的意思都一样。`);
   const v=pred?tx(`<p class="${ok?'okline':'badline'}"><b>${ok?'Right.':'Not quite.'}</b> ${ok?'':`You predicted “${esc(predTxt(pred))}”. `}${ans}</p>`,`<p class="${ok?'okline':'badline'}"><b>${ok?'对了。':'不对。'}</b>${ok?'':`你预测的是“${esc(predTxt(pred))}”。`}${ans}</p>`):`<p>${ans}</p>`;
   return v+tx(`<p>Keyword search here is BM25: over the question’s words, it adds up how rare each word is (its idf) times how often it occurs in the chunk. It makes two adjustments. Each further occurrence of the same word adds less than the one before; k₁ sets how fast that falls. A chunk longer than average is marked down; b sets by how much. The formula is under “Show the math”.</p><p>The two searches miss different questions: keyword search misses a question in the customer’s own words, vector search misses a code that must match letter for letter. Hybrid search runs both, takes each one’s top ${NCAND} and merges the two lists by reciprocal rank fusion (RRF). It uses only the ranks r, not the scores, with a constant c = 60:</p>${M(RRF_TEX('RRF'))}<p>A chunk on both lists collects two shares, so it beats a chunk that is first on only one list. Pick a question and compare the three lists; the first chunk of each is what keeping 1 would put into the context.</p>`,
                 `<p>这里的关键词检索用 BM25：把问题里每个词的分数加起来，词越少见（idf 越大）、在这一块里出现得越多，分越高。BM25 还做了两处调整。同一个词在一块里出现很多次时，后面每次加的分越来越少，k₁ 决定减得多快。比平均长的块，分数要打折，b 决定打折多少。完整的公式在“展开公式”里。</p><p>两种检索漏掉的问题不一样：关键词检索漏掉用户换了说法的问题，向量检索漏掉要逐字对上的编号。混合检索两种都跑，各取前 ${NCAND} 块，再用倒数排名融合（RRF）把两个列表合起来，只用名次 r，不用分数，常数 c = 60：</p>${M(RRF_TEX('RRF'))}<p>两个列表里都有的块拿两份分，所以会赢过只在一个列表里排第一的块。选一个问题，比一比三个列表；每个列表的第一块，就是只取 1 块时进上下文的那块。</p>`)+inQ(P4Q);},
  obs:()=>obsSearch()},
 {id:'rerank',title:{en:'Rerank',zh:'rerank'},lead:{en:`Read the top ${NCAND} again with a model that sees question and chunk together; keep 1.`,zh:`用一起读问题和块的模型，把前 ${NCAND} 块重读一遍，只留 1 块。`},
  panels:['lists','metrics'],cols:()=>['vec0','vec1'],enter:()=>({size:64,ov:2,ret:'vec',rr:0,k:1,q:17}),expect:()=>tried.has('rr1'),showMe:{rr:1},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Turn rerank on to continue.',zh:'打开 rerank 再继续。'},
  body:()=>{const q=setup(LG(),64,2).Q[17];
   return tx(`<p>Keyword and vector search prepare every chunk before any question arrives (an index of its words, its vector); when a question comes they only make cheap comparisons, which is why they can search millions of chunks (how vector search avoids comparing every vector: ${LL('vector-index')}). The price is that no chunk was ever read together with the question. Rerank adds a second pass: a model that reads the question and one chunk together, a ${DEF('cross-encoder',D_CE())}, scores each candidate again and can tell whether the chunk answers this question. It needs one model run per candidate, so it reads only the first ${NCAND} chunks the first search returns, then keeps the best k.</p><p>No cross-encoder runs in this page; the lab uses a ${DEF('stand-in',D_STAND())}. It first lists the meanings and codes the question mentions, then checks what share of them each candidate chunk contains. That share is the chunk’s new score, the “covers” number on the right of the list. For ${quote(qText(17))} vector search ranks the chunk with the answer ${q.rk.vec0+1}, under two chunks about the lock’s batteries. Keep 1 chunk and turn rerank on.</p>`,
              `<p>关键词检索和向量检索在问题来之前就把每一块处理好了（建好词表、算好向量），问题来了只做很便宜的比较，所以能在 1M 以上的块里搜（向量检索怎么做到不逐个比较，见${LL('vector-index')}）。代价是每一块都是单独处理的，没有和问题一起读过。rerank 再加一道工序：用一个把问题和这一块拼在一起读的模型（${DEF('cross-encoder',D_CE())}）给每个候选重新打分，看得出这一块是不是在回答这个问题。每一对都要现跑一次模型，所以它只读第一轮返回的前 ${NCAND} 块，再留下最好的 k 块。</p><p>这个页面里跑不了 cross-encoder，本 Lab 用一个${DEF('替身',D_STAND())}：先列出问题里讲到的几个意思和编号，再看每个候选块包含了其中几成，这个比例就是它的新分数，也就是列表右边的“覆盖”。${quote(qText(17))}这一问，向量检索把答案${rkTxt(q.rk.vec0)}，前面两块讲的是门锁的电池。只取 1 块，打开 rerank 看看。</p>`)+inRr()+inQ(P5Q);},
  obs:()=>obsRerank()},
 {id:'budget',title:{en:'The context budget',zh:'上下文预算'},lead:{en:'Every chunk kept costs context tokens, on every question.',zh:'每多取一块，每个问题都多占一份上下文 token。'},
  panels:['chart','lists','metrics'],cols:()=>[S.ret+S.rr],enter:()=>({size:64,ov:2,ret:'hyb',rr:0,k:1,q:8}),expect:()=>tried.size>1,showMe:{k:3},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another k or chunk size to continue.',zh:'换一个 k 或块大小再继续。'},
  body:()=>tx(`<p>Every chunk kept goes into the ${DEF('context',D_CTX())}. Keeping k chunks of up to L tokens costs up to ${KL()} tokens on every question, paid in money, in time to the first token and in room taken from the conversation (see ${LL('context-eng')}). When an answer does not rank first, keeping more chunks can still bring it into the context. While some answers are still missing, keeping one more chunk can raise recall; once all of them are in, more chunks only add tokens, plus passages that do not answer the question.</p><p>Here hybrid search runs without rerank. Change how many chunks are kept, or the chunk size, and watch ${J('chart','the chart')}: across it run the chunk sizes, with the context tokens per question written under each; the three colours are the recall of keyword, vector and hybrid search. Splitting by heading has no fixed size, so it is drawn as a separate point.</p>`,
              `<p>取回的每一块都放进${DEF('上下文',D_CTX())}。取 k 块、每块最多 L 个 token，每个问题就要占最多 ${KL()} 个 token：要花钱，要拉长首 token 的等待，还要挤掉对话的空间（见${LL('context-eng')}）。答案没排到第一时，多取几块也能把它带进上下文。只要还有答案没进来，多取一块，召回率就可能升；等答案都进来了，再多取只是多花 token，还会带进一些和问题无关的段落。</p><p>这里用混合检索，不用 rerank。改一改取几块或块大小，看${J('chart','图')}：横着排开的是各种块大小，每个块大小下面一行写着这时每个问题的上下文 token；三种颜色分别是关键词、向量和混合检索的召回率。按标题切没有固定的块大小，所以单独画成一点。</p>`)+inK()+inSize(),
  obs:()=>obsBudget()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Cut, score, keep: three places a passage goes missing.',zh:'切、打分、取几块：段落会丢在三个地方。'},
  panels:['chart','metrics'],enter:()=>({size:64,ov:2,ret:'hyb',rr:0,k:3,q:8}),
  body:()=>{const b=measure(LG(),{...DEFAULT,size:16}), R=setup(LG(),64,2), q0=R.Q[0], big=measure(LG(),{size:128,ov:0,ret:'vec',rr:0,k:5});
   return tx(`<ul><li><b>Check whether the answer survives the cut before tuning search.</b> When the cut falls inside an answer, no chunk holds all of it: keeping 1 chunk, any search returns at most part of it. Overlap of half a chunk keeps every answer up to half a chunk long whole.</li><li><b>Keep keyword search next to vector search.</b> A customer who types an error code needs the letters to match: keyword search puts the E-201 chunk first, vector search ${rkTxt(q0.rk.vec0)}. A customer who describes the problem in other words needs the meaning: the fall down the stairs is found by vector search, not by keywords. Merged by RRF, the two leave the fewest answers outside the first few chunks, the ones rerank reads.</li><li><b>Rerank can make the first chunk more reliable, so fewer chunks need to be kept, but it only reorders what the first search returns.</b> On vector search, rerank moved the thermostat-battery question to first place, but the E-201 chunk, which vector search ranks below the candidates, never reaches the reranker. Each candidate costs reranker time, and when an answer is spread over several places, keep more chunks anyway.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Smaller chunks are more precise.” Cut into 16-token chunks, ${b.splits} answers here no longer fit in any chunk.</p><p>✗ “Vector search can replace keyword search.” Codes, model numbers and product names have to match letter for letter; that is what keyword search does.</p><p>✗ “Retrieving more is always better.” Keeping five 128-token chunks puts up to ${A.fmt.int(big.ctx)} tokens into every question’s context, most of them about something else.</p></div>${A.shell.toScenarios()}`,
             `<ul><li><b>先查答案有没有被切开，再调检索。</b>一刀切在答案中间，就没有哪一块装着完整的答案：只取 1 块时，哪种检索都只能拿回一部分。相邻两块重叠半块，长度不超过半块的答案就都能完整地落在某一块里。</li><li><b>向量检索旁边要留着关键词检索。</b>用户输入错误码时，要逐字对上：关键词检索把讲 E-201 的块排第一，向量检索${rkTxt(q0.rk.vec0)}。用户换一种说法描述问题时，要靠意思：机器人摔下楼梯那一问，向量检索找得到，关键词检索找不到。用 RRF 合起来，落在前几块之外的答案最少，而 rerank 读的正是前几块。</li><li><b>rerank 能让排在第一的那块更可靠，于是可以少取几块；但它只能重排第一轮返回的块。</b>在向量检索上，rerank 把恒温器电池那一问的答案排到了第一；可讲 E-201 的那块，向量检索排在候选之外，根本到不了 rerank 面前。每个候选都要花 rerank 的时间；答案分散在几处时，还是要多取几块。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “块越小越精确。”切成每块 16 个 token，这里有 ${b.splits} 个答案装不进任何一块。</p><p>✗ “向量检索能替代关键词检索。”错误码、型号、产品名要逐字对上，这正是关键词检索做的事。</p><p>✗ “取得越多越好。”取五块、每块 128 个 token，每个问题要往上下文里放最多 ${A.fmt.int(big.ctx)} 个 token，大部分讲的是别的事。</p></div>${A.shell.toScenarios()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
function obsChunk(){const R=setup(LG(),S.size,S.ov), q=R.Q[S.q], whole=q.right.length>0;
  const cap=t=>t.charAt(0).toUpperCase()+t.slice(1);
  let h=whole?tx(`<p><b>${cap(sizeTxt(S.size))}${S.size==='h'?'':', '+ovTxt(S.size,S.ov)}: the answer lies whole inside one chunk.</b></p>`,`<p><b>${sizeTxt(S.size)}${S.size==='h'?'':'，'+ovTxt(S.size,S.ov)}：答案完整地落在一块里。</b></p>`)
   :tx(`<p class="badline"><b>${cap(sizeTxt(S.size))}${S.size==='h'?'':', '+ovTxt(S.size,S.ov)}: the answer is cut apart.</b></p>`,`<p class="badline"><b>${sizeTxt(S.size)}${S.size==='h'?'':'，'+ovTxt(S.size,S.ov)}：答案被切开了。</b></p>`);
  h+=tx(`<ul><li>answers cut apart: ${R.splits} of 20</li><li>chunks in the help center: ${R.N}</li></ul>`,`<ul><li>被切开的答案：20 个里有 ${R.splits} 个</li><li>整个帮助中心：${R.N} 块</li></ul>`);
  if(whole&&(S.size==='h'||S.size>=128)) h+=tx(`<p>Bigger chunks keep answers whole, but every chunk kept takes more of the context, and more of it is about something else.</p>`,`<p>块越大，答案越不容易被切开，但取回的每一块都要占更多上下文，一块里和问题无关的话也更多。</p>`);
  return h;}
/* the chunk holding the answer shares no word with the question, so keyword search does not return it */
const kwZero=q=>q.right.length>0&&q.right.every(j=>q.kw[j]<=0);
const placeTxt=r=>r<0?tx('not returned','不返回'):tx(`${r+1}`,`第 ${r+1} 名`);
function obsSearch(){const R=setup(LG(),64,2), q=R.Q[S.q];
  const z=kwZero(q), rk=r=>q.rk[r+'0'];
  let h=tx(`<p><b>${quote(qText(S.q))} Where the chunk with the answer ranks:</b></p><ul><li>keyword search: ${z?'not returned (no word in common)':placeTxt(rk('kw'))}</li><li>vector search: ${placeTxt(rk('vec'))}</li><li>hybrid search: ${placeTxt(rk('hyb'))}</li></ul>`,
           `<p><b>${quote(qText(S.q))}装着答案的块排在哪：</b></p><ul><li>关键词检索：${z?'不返回（没有共同的词）':placeTxt(rk('kw'))}</li><li>向量检索：${placeTxt(rk('vec'))}</li><li>混合检索：${placeTxt(rk('hyb'))}</li></ul>`);
  if(z) h+=tx(`<p>This question shares no word with its answer, so keyword search does not return that chunk at all. In RRF it gets one share, from vector search; chunks on both lists get two, so it falls to place ${rk('hyb')+1}. In the next part, rerank reads the question together with each chunk’s text and scores the chunks again.</p>`,`<p>这一问和装着答案的那一块没有一个相同的词，关键词检索根本不返回它。RRF 合并时，它只从向量检索拿到一份分数，比不过两个列表里都有的块，所以掉到了第 ${rk('hyb')+1} 名。下一部分的 rerank，会把问题和每一块的内容放在一起重新读一遍，再打一次分。</p>`);
  h+=tx(`<p>Recall@1 over the 20 questions:</p><ul><li>keyword search: ${pc(firstN(R,'kw')/NQ)}</li><li>vector search: ${pc(firstN(R,'vec')/NQ)}</li><li>hybrid search: ${pc(firstN(R,'hyb')/NQ)}</li></ul><p>Hybrid search does not always put the answer first, but it most often has it in its top ${NCAND}, and those top ${NCAND} are what the reranker in the next part reads.</p><p>In hybrid search’s top ${NCAND}: ${poolN(R,'hyb')} of the 20 answers.</p>`,
        `<p>20 个问题的 recall@1：</p><ul><li>关键词检索：${pc(firstN(R,'kw')/NQ)}</li><li>向量检索：${pc(firstN(R,'vec')/NQ)}</li><li>混合检索：${pc(firstN(R,'hyb')/NQ)}</li></ul><p>混合检索不一定把答案排第一，但最常把它排进前 ${NCAND}；下一部分的 rerank，正是从这前 ${NCAND} 块里重新挑。</p><p>排进混合检索前 ${NCAND} 的答案：20 个里有 ${poolN(R,'hyb')} 个。</p>`);
  return h;}
function obsRerank(){const R=setup(LG(),64,2), q=R.Q[S.q], a=measure(LG(),{size:64,ov:2,ret:'vec',rr:0,k:1}), b=measure(LG(),{size:64,ov:2,ret:'vec',rr:1,k:1}), q0=R.Q[0];
  const out=q.rk.vec0<0||q.rk.vec0>=NCAND;
  let h=out?tx(`<p class="badline"><b>${quote(qText(S.q))} Vector search ${rkTxt(q.rk.vec0)}, outside the candidates: the reranker never reads that chunk, so it stays out${S.rr?'':' with rerank too'}.</b></p>`,`<p class="badline"><b>${quote(qText(S.q))}向量检索把答案${rkTxt(q.rk.vec0)}，不在候选里：rerank 根本读不到这一块${S.rr?'，它还是进不来':'，打开 rerank 也进不来'}。</b></p>`)
   :S.rr?tx(`<p><b>With rerank, the answer to ${quote(qText(S.q))} moves from ${q.rk.vec0+1} to ${q.rk.vec1+1}.</b></p>`,`<p><b>用了 rerank，${quote(qText(S.q))}的答案从第 ${q.rk.vec0+1} 名升到第 ${q.rk.vec1+1} 名。</b></p>`)
   :tx(`<p><b>Without rerank, vector search ${rkTxt(q.rk.vec0)}.</b></p>`,`<p><b>不用 rerank，向量检索把答案${rkTxt(q.rk.vec0)}。</b></p>`);
  h+=tx(`<p>Keeping 1 chunk, the questions whose kept chunk holds the answer: with vector search alone, ${Math.round(a.rec*NQ)} of the 20 (${pc(a.rec)}); with vector search followed by rerank, ${Math.round(b.rec*NQ)} (${pc(b.rec)}).</p><p>The price is time: the reranker reads every candidate, about ${msTxt(b.rerankMs)} more per question in this lab (the arithmetic is under “Show the math”).</p>`,
        `<p>只取 1 块时，这一块装着答案的问题：只用向量检索是 20 个里 ${Math.round(a.rec*NQ)} 个（${pc(a.rec)}）；向量检索后再加 rerank，是 ${Math.round(b.rec*NQ)} 个（${pc(b.rec)}）。</p><p>代价是时间：rerank 要把每个候选读一遍，按本 Lab 的假设，每个问题要多等约 ${msTxt(b.rerankMs)}（算法见“展开公式”）。</p>`);
  if(!out) h+=tx(`<p>Rerank only reorders what the first search hands it. Pick “E-201”: vector search ${rkTxt(q0.rk.vec0)}, outside the candidates.</p>`,`<p>rerank 只能重排第一轮交给它的块。选“E-201”看看：向量检索把那一块${rkTxt(q0.rk.vec0)}，不在候选里。</p>`);
  return h;}
/* the observation in part 6 follows the state: what keeping more chunks would bring, or that it brings nothing (R6.10) */
function obsBudget(){const m=measure(LG(),S), at=k=>measure(LG(),{...S,k}), miss=NQ-m.hits;
  let h=tx(`<p><b>Keeping ${S.k}: recall@${S.k} ${pc(m.rec)}.</b></p><p>Context: up to ${tokTxt(m.ctx)} per question.</p>`,`<p><b>取 ${S.k} 块：recall@${S.k} 是 ${pc(m.rec)}。</b></p><p>上下文：每个问题最多 ${tokTxt(m.ctx)}。</p>`);
  const kN=KS[KS.indexOf(S.k)+1], kAll=KS.find(k=>at(k).hits===NQ);
  if(miss&&kN){const u=at(kN);
    h+=tx(`<p>Keeping ${S.k}, ${miss} answer${miss===1?' is':'s are'} still missing.</p>`,`<p>取 ${S.k} 块时，还有 ${miss} 个答案没进来。</p>`);
    h+=u.hits>m.hits?tx(`<ul><li>keeping ${kN}: ${u.hits-m.hits} more answer${u.hits-m.hits===1?'':'s'} in</li><li>context: ${A.fmt.int(m.ctx)} → ${A.fmt.int(u.ctx)} tokens</li></ul>`,`<ul><li>取 ${kN} 块：多进来 ${u.hits-m.hits} 个答案</li><li>上下文：${A.fmt.int(m.ctx)} → ${A.fmt.int(u.ctx)} 个 token</li></ul>`)
      :tx(`<p>Keeping ${kN} brings in none of them: they are cut apart or rank below ${kN}.</p>`,`<p>取 ${kN} 块也一个都带不进来：它们被切开了，或者排在第 ${kN} 名之后。</p>`);}
  else if(miss) h+=tx(`<p>Even keeping ${S.k}, ${miss} answer${miss===1?' is':'s are'} missing.</p>`,`<p>就算取 ${S.k} 块，还有 ${miss} 个答案没进来。</p>`)+(m.splits?tx(`<p>${m.splits} of them are cut apart; the rest rank below ${S.k}.</p>`,`<p>其中 ${m.splits} 个被切开了，其余的排在第 ${S.k} 名之后。</p>`):'');
  else if(kAll<S.k) h+=tx(`<p>Keeping ${kAll} already brings every answer in; keeping ${S.k} only costs more tokens.</p><ul><li>context: ${A.fmt.int(at(kAll).ctx)} → ${A.fmt.int(m.ctx)} tokens</li></ul>`,`<p>取 ${kAll} 块时答案已经全进来了，取 ${S.k} 块只是多花 token。</p><ul><li>上下文：${A.fmt.int(at(kAll).ctx)} → ${A.fmt.int(m.ctx)} 个 token</li></ul>`);
  else h+=tx(`<p>Keeping ${S.k}, every answer is in.</p>`,`<p>取 ${S.k} 块，答案全进来了。</p>`);
  return h;}

/* ---------- scenarios ---------- */
const PRESETS=[
 {k:'init',t:{en:'Initial setup: 64 tokens, vector search, keep 1',zh:'初始设置：64 个 token，向量检索，取 1 块'},d:{en:'A first RAG demo: fixed chunks, no overlap, no rerank.',zh:'第一版 RAG：固定大小、不重叠、不 rerank。'},p:{size:64,ov:0,ret:'vec',rr:0,k:1}},
 {k:'small',t:{en:'Small chunks: 16 tokens',zh:'小块：16 个 token'},d:{en:'Each chunk about one short sentence.',zh:'每块大约一句短话。'},p:{size:16,ov:0,ret:'vec',rr:0,k:1}},
 {k:'head',t:{en:'Cut by heading',zh:'按标题切'},d:{en:'One chunk per section, whatever its length.',zh:'每节一块，不管多长。'},p:{size:'h',ov:0,ret:'vec',rr:0,k:1}},
 {k:'kw',t:{en:'Keyword search only',zh:'只用关键词检索'},d:{en:'64 tokens, overlap of half a chunk, BM25.',zh:'64 个 token，重叠半块，BM25。'},p:{size:64,ov:2,ret:'kw',rr:0,k:1}},
 {k:'hyb',t:{en:'Hybrid search, keep 3',zh:'混合检索，取 3 块'},d:{en:'64 tokens, overlap of half a chunk, BM25 and vectors merged by RRF.',zh:'64 个 token，重叠半块，BM25 和向量检索用 RRF 合并。'},p:{size:64,ov:2,ret:'hyb',rr:0,k:3}},
 {k:'big',t:{en:'Big chunks, keep 5',zh:'大块，取 5 块'},d:{en:'128 tokens per chunk, vector search.',zh:'每块 128 个 token，向量检索。'},p:{size:128,ov:0,ret:'vec',rr:0,k:5}}];
const presetNow=()=>PRESETS.findIndex(p=>sameSet(p.p,S));
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many of the 20 answers reach the context, and what it costs.','选一个场景，看看 20 个答案里有几个能进到上下文，又要花多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="ckr-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(){const m=measure(LG(),S), i=presetNow(), base=measure(LG(),PRESETS[0].p);
  let h=tx(`<p><b>${m.hits} of the 20 answers reach the context.</b></p><ul><li>recall@${S.k}: ${pc(m.rec)}</li><li>context: up to ${tokTxt(m.ctx)} per question</li><li>search${S.rr?' and rerank':''}: ${msTxt(m.lat)} per question</li></ul>`,
           `<p><b>20 个答案里有 ${m.hits} 个进到上下文。</b></p><ul><li>recall@${S.k}：${pc(m.rec)}</li><li>上下文：每个问题最多 ${tokTxt(m.ctx)}</li><li>检索${S.rr?'和 rerank':''}：每个问题 ${msTxt(m.lat)}</li></ul>`);
  if(i<0) return h+tx(`<p class="small">The current settings match none of these scenarios: ${setTxt(S)}.</p>`,`<p class="small">现在的设置和这几个场景都不一样：${setTxt(S)}。</p>`);
  const k=PRESETS[i].k, other=NQ-m.hits-m.splits;
  if(k==='init') h+=tx(`<p>${m.splits} answers are cut apart, so no chunk holds them whole.</p><p>Of the others, ${other} rank below first in vector search.</p>`,`<p>有 ${m.splits} 个答案被切开了，没有哪一块装着完整的答案。</p><p>其余的答案里，有 ${other} 个在向量检索里没排到第一。</p>`);
  else if(k==='small') h+=tx(`<p>${m.splits} of the 20 answers are cut apart.</p><p>Small chunks look precise, but most answers here are one or two sentences, longer than a chunk.</p>`,`<p>20 个答案里有 ${m.splits} 个被切开。</p><p>小块看起来精确，可这里的答案大多有一两句话，比一块还长。</p>`);
  else if(k==='head') h+=tx(`<p>No answer is cut apart, since every answer sits inside one section. The initial setup reaches ${pc(base.rec)}.</p>`,`<p>没有答案被切开，因为每个答案都在一节之内；初始设置只有 ${pc(base.rec)}。</p>`)+
    (m.ctx>base.ctx?tx(`<p>But sections differ in length, and the longest one sets the context for every question: more than the ${base.ctx} tokens of the initial setup.</p>`,`<p>可各节长短不一，最长的那一节决定了每个问题的上下文，比初始设置的 ${base.ctx} 个 token 还多。</p>`)
      :tx(`<p>The sections here are short, so even the longest is under ${base.ctx} tokens. Real sections often run far longer, and the longest one would set the context for every question.</p>`,`<p>这里的每一节都很短，最长的也不到 ${base.ctx} 个 token。真实文档的一节常常长得多，最长的那一节会决定每个问题的上下文。</p>`));
  else if(k==='kw') h+=tx(`<p>Error codes rank first, but questions in the customer’s own words miss: the fall down the stairs shares no word with its answer, so keyword search does not return it.</p>`,`<p>错误码都排第一，可用户换了说法的问题会漏：摔下楼梯那一问和答案没有共同的词，关键词检索不返回它。</p>`);
  else if(k==='hyb') h+=tx(`<p>Codes come from the keyword list, paraphrases from the vector list. RRF uses only ranks, so BM25 scores and cosines need no common scale.</p><p>Both lists weigh the same here; LangChain and the Anthropic cookbook let you weight them.</p>`,`<p>错误码靠关键词列表，换了说法的问题靠向量列表。RRF 只用名次，所以 BM25 和余弦不用换算到同一个尺度。</p><p>本 Lab 两边的权重一样；LangChain 和 Anthropic 的 cookbook 都允许调权重。</p>`);
  else h+=tx(`<p>Recall rises from ${pc(base.rec)} (the initial setup) to ${pc(m.rec)}.</p><p>But each question carries ${f(m.ctx/base.ctx,1)} times as much context.</p>${m.splits?`<p>And ${m.splits} answer${m.splits===1?' is':'s are'} still cut apart.</p>`:''}`,`<p>召回率从初始设置的 ${pc(base.rec)} 升到 ${pc(m.rec)}。</p><p>但每个问题要带的上下文是原来的 ${f(m.ctx/base.ctx,1)} 倍。</p>${m.splits?`<p>而且还有 ${m.splits} 个答案被切开。</p>`:''}`);
  return h;}

/* ---------- challenge ---------- */
const chDone=()=>mode==='challenge'&&chSubmitted;
const chPick=()=>({size:S.size,ov:S.size==='h'?0:S.ov,ret:S.ret,rr:S.rr,k:S.k});
function chMe(){const m=measure(LG(),S); return {...chPick(),rec:m.rec,ctx:m.ctx,lat:m.lat,ok:m.rec>=CH_GOAL-1e-9&&m.ctx<=CH_CTX&&m.lat<=CH_LAT+1e-9};}
const pickTxt=p=>tx(`${sizeTxt(p.size)}, ${ovTxt(p.size,p.ov)}, ${L(RET_S[p.ret])}${p.rr?' + rerank':''}, keep ${p.k}`,`${sizeTxt(p.size)}，${ovTxt(p.size,p.ov)}，${L(RET_S[p.ret])}${p.rr?' + rerank':''}，取 ${p.k} 块`);
const GOAL=()=>pc(CH_GOAL);
function challengeHTML(){
  const sub=`<div id="ckr-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, the 20 questions are searched with your setting; then you see recall.','提交后，用你的设置把 20 个问题都搜一遍，再告诉你召回率。')}</p></div>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Get at least ${GOAL()} of the answers into the context, with as few tokens as possible`,`让至少 ${GOAL()} 的答案进到上下文，token 越少越好`)}</h3>`+
   `<p>${tx('The same help center and 20 questions. An answer counts when one kept chunk holds all of it.','还是这个帮助中心、这 20 个问题。取回的某一块装着完整的答案才算数。')}</p>`+
   `<p>${tx(`Context tokens per question: ${KL()}, where L is the length of the longest chunk.`,`每个问题的上下文 token：${KL()}，L 是最长那一块的 token 数。`)}</p>`+
   `<p>${tx('The time follows this lab’s latency model:','时间按本 Lab 的延迟假设算：')}</p><ul>`+
   `<li>${tx(`search: ${LAT.kw} to ${LAT.hyb} ms, by method`,`检索：${LAT.kw} 到 ${LAT.hyb} 毫秒，看用哪种`)}</li>`+
   `<li>${tx(`rerank: up to ${NCAND} candidates, ${f(RR_MS,1)} ms per token read`,`rerank：最多读 ${NCAND} 块候选，每个 token ${f(RR_MS,1)} 毫秒`)}</li></ul>`+
   `<p>${tx('Only fixed-size chunks this time, so that the chunk size is your choice and not the length of the documents’ sections. Choose:','这次只用固定大小切块，块多大由你定，而不是由文档里每节有多长来定。选：')}</p>`+
   inSize(CH_SIZES)+inOv()+inRet()+inRr()+inK()+
   `<div class="goal mono" id="ckr-ch-a"><span>${tx('Answers in the context, recall@k','进到上下文的答案 recall@k')}</span><span><b id="ckr-ch-av"></b>${tx(` (goal: at least ${GOAL()})`,`（目标：至少 ${GOAL()}）`)}</span></div><div class="progress" id="ckr-ch-ap" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="ckr-ch-c"><span>${tx('Context tokens per question','每个问题的上下文 token')}</span><span><b id="ckr-ch-cv"></b>${tx(` (at most ${CH_CTX})`,`（最多 ${CH_CTX}）`)}</span></div>`+
   `<div class="goal mono" id="ckr-ch-l"><span>${tx('Search and rerank time','检索和 rerank 的时间')}</span><span><b id="ckr-ch-lv"></b>${tx(` (at most ${CH_LAT} ms)`,`（最多 ${CH_LAT} 毫秒）`)}</span></div>`+
   `<p class="small muted">${tx('These three limits are set for this challenge in this lab.','这三个限制是这个 Lab 为这道题定的。')}</p>`+
   sub+`<div id="ckr-ch-res" class="ckr-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chDone()) return '';
  const pts=chAll(LG()), me=chMe(), okp=pts.filter(p=>p.ok), minC=Math.min(...okp.map(p=>p.ctx)), best=okp.filter(p=>p.best);
  const bb=best.slice().sort((a,b)=>b.ov-a.ov||a.lat-b.lat)[0], why=[];
  if(me.rec<CH_GOAL-1e-9) why.push(tx(`recall@${me.k} is ${pc(me.rec)}, under ${GOAL()}`,`recall@${me.k} 是 ${pc(me.rec)}，不到 ${GOAL()}`));
  if(me.ctx>CH_CTX) why.push(tx(`${tokTxt(me.ctx)} of context, over ${CH_CTX}`,`上下文 ${tokTxt(me.ctx)}，超过 ${CH_CTX}`));
  if(me.lat>CH_LAT+1e-9) why.push(tx(`${msTxt(me.lat)}, over ${CH_LAT} ms`,`${msTxt(me.lat)}，超过 ${CH_LAT} 毫秒`));
  let h;
  if(me.ok&&me.ctx===minC) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met with ${tokTxt(me.ctx)} of context: recall@${me.k} ${pc(me.rec)}.</b></p><p class="small">No setting meets the goal with fewer context tokens.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标，上下文 ${tokTxt(me.ctx)}：recall@${me.k} ${pc(me.rec)}。</b></p><p class="small">没有哪种设置能用更少的上下文 token 达标。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: recall@${me.k} ${pc(me.rec)}.</b></p><p class="small">But this setting also meets it, with fewer tokens:</p><ul class="small"><li>${pickTxt(bb)}</li><li>context: ${A.fmt.int(bb.ctx)} tokens instead of ${A.fmt.int(me.ctx)}</li></ul>`,`<p class="okline" style="margin-top:12px"><b>达成目标：recall@${me.k} ${pc(me.rec)}。</b></p><p class="small">不过下面这个设置也能达标，token 更少：</p><ul class="small"><li>${pickTxt(bb)}</li><li>上下文：${A.fmt.int(bb.ctx)} 个 token，而不是 ${A.fmt.int(me.ctx)} 个</li></ul>`);
  else {h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet:</b></p><ul class="small">${why.map(w=>`<li>${w}</li>`).join('')}</ul><p class="small">Hint: keeping fewer chunks saves tokens only if the right chunk comes first. What keeps the answer whole, what puts it first, and what decides whether it is among the chunks that can be put first?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：</b></p><ul class="small">${why.map(w=>`<li>${w}</li>`).join('')}</ul><p class="small">提示：少取几块能省 token，前提是正确的那块排在第一。什么能保证答案不被切开？什么能把它排到第一？又是什么决定它在不在能被排到第一的那几块里？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(me.ok) chReveal=true;
  if(!chReveal) return h;
  const isInit=p=>sameSet(p,DEFAULT), isMe=p=>sameSet(p,me);
  const rows=pts.map(p=>({label:pickTxt(p),value:p.rec*100,text:tx(`${pc(p.rec)}, ${A.fmt.int(p.ctx)} tok, ${f(p.lat,0)} ms`,`${pc(p.rec)}，${A.fmt.int(p.ctx)} token，${f(p.lat,0)} 毫秒`),ok:p.ok,me:isMe(p),best:p.best,
    tags:(isInit(p)?[{text:tx('initial setup','初始设置')}]:[]).concat(isMe(p)?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  rows.sort((a,b)=>(b.best-a.best)||(b.ok-a.ok));
  h+=A.ui.cmpChart({brief:true,title:tx('Answers in the context (recall@k) of the initial setup, your pick and the settings that meet all three limits','初始设置、你的选择，以及三项都达标的设置：各自进到上下文的答案（recall@k）'),
    aria:tx('Recall by setting.','按设置的召回率。'),rows,goal:CH_GOAL*100,goalLabel:tx(`goal: ${GOAL()}`,`目标 ${GOAL()}`),max:100,
    hiddenText:n=>tx(`The other ${n} settings miss at least one limit and are not shown.`,`其余 ${n} 种设置至少有一项不达标，图里没有列出。`),
    okText:tx('meets all three','三项都达标'),noText:tx('misses one','有一项没达标'),goalText:tx(`goal: ${GOAL()}`,`目标：${GOAL()}`)});
  const R=setup(LG(),bb.size,bb.ov), alt=p=>measure(LG(),{...bb,...p}), pl=r=>`<li>${L(RET_N[r])}${tx(': ','：')}${poolN(R,r)}</li>`;
  h+=tx(`<p class="small">Why: the fewest tokens come from keeping a single chunk, so that chunk must be the right one, and rerank is what puts it first.</p><p class="small">But rerank only reorders the first ${NCAND} chunks of the first search, so the first search decides what rerank can find. Only hybrid search has ${poolN(R,'hyb')===NQ?'every':'almost every'} answer in its first ${NCAND}: keyword search misses the questions in other words, vector search misses codes. Overlap keeps the answers whole.</p>`,
        `<p class="small">原因：上下文最省的做法是只取一块，所以这一块必须就是对的那块，而把它排到第一的是 rerank。</p><p class="small">可 rerank 只能重排第一轮检索的前 ${NCAND} 块，所以第一轮决定了 rerank 能找到什么。只有混合检索把${poolN(R,'hyb')===NQ?'每一个':'几乎每一个'}答案都放进了前 ${NCAND}：关键词检索漏掉换了说法的问题，向量检索漏掉编号。重叠让答案保持完整。</p>`);
  h+=`<p class="small">${tx(`With this chunking, how many of the 20 answers are in each search’s first ${NCAND}:`,`这种切法下，20 个答案里有几个在各种检索的前 ${NCAND} 块里：`)}</p><ul class="small">${pl('kw')}${pl('vec')}${pl('hyb')}</ul>`;
  const row=(t,m)=>`<li>${t}${tx(': ','：')}${pc(m.rec)}${m.lat>CH_LAT+1e-9?tx(`, ${msTxt(m.lat)}`,`，${msTxt(m.lat)}`):''}</li>`;
  h+=`<details class="ckr-why small"><summary>${tx(`The best setting with one choice changed`,`最好的设置，只改一处`)}</summary><ul>`+
    row(tx('keyword search + rerank','关键词检索 + rerank'),alt({ret:'kw'}))+row(tx('vector search + rerank','向量检索 + rerank'),alt({ret:'vec'}))+
    row(tx('no rerank','不用 rerank'),alt({rr:0}))+row(tx('no overlap','不重叠'),alt({ov:0}))+row(tx('32-token chunks','每块 32 个 token'),alt({size:32}))+row(tx('128-token chunks','每块 128 个 token'),alt({size:128}))+`</ul></details>`;
  h+=tx(`<p class="small muted">Note: in this lab 20 questions decide recall, so one question is 5 percentage points, and the latency is an assumption, not a measurement. Before choosing, measure recall on your own questions and time the search for real.</p>`,`<p class="small muted">注意：在这个 Lab 里只有 20 个问题，一个问题就是 5 个百分点；延迟是假设，不是实测。真正选设置之前，要用自己的问题集测召回率，并实测延迟。</p>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('find where a RAG pipeline loses the right passage (the cut, the scoring or k), and choose chunk size, overlap, hybrid search, rerank and k within a token budget.','找出 RAG 流程在哪里丢了正确的段落（切块、打分还是 k），并在 token 预算内选好块大小、重叠、混合检索、rerank 和 k。');
const EXITS=()=>({
 explain:[
  {q:tx('A team cuts its manuals into 16-token chunks “to be precise”. Recall drops. Why?','一个团队把说明书切成 16 个 token 的小块，想让检索“更精确”，召回率却降了。为什么？'),
   o:[tx('Most answers are longer than a chunk, so they are cut apart and no single chunk holds the whole answer.','大多数答案比一块还长，被切开了，没有哪一块装着完整的答案。'),
      tx('Small chunks make BM25’s idf smaller, so every score drops.','块小了，BM25 的 idf 会变小，所以每个分数都降了。'),
      tx('The vector of a small chunk is longer, so its cosine is wrong.','小块的向量更长，所以余弦算错了。')],a:0,
   n:[null,tx('idf depends on how many chunks contain a word, not on the chunk size alone; scores are compared within one chunking anyway.','idf 取决于有多少块含有这个词，而不只是块的大小；何况分数只在同一种切法里互相比较。'),tx('Every vector is scaled to length 1 before the cosine; length plays no part.','算余弦之前每个向量都缩放成长度 1，长度不起作用。')],
   why:tx('Recall counts a chunk only when it holds the whole answer; overlap or a larger chunk keeps answers whole.','只有装着完整答案的块才算找到；重叠或更大的块能让答案保持完整。')},
  {q:tx('Why does adding a reranker let a team keep fewer chunks?','为什么加了 rerank，团队就能少取几块？'),
   o:[tx('It reads the question and each candidate together, so the right chunk is more often first; then keeping 1 or 2 chunks is enough.','它把问题和每个候选放在一起读，正确的那块更常排在第一；这样取 1、2 块就够了。'),
      tx('It shortens every chunk, so more chunks fit.','它把每一块都缩短了，所以能放进更多块。'),
      tx('It searches the whole collection again with a bigger model.','它用更大的模型把整个知识库重新搜一遍。')],a:0,
   n:[null,tx('A reranker only reorders the candidates; the chunks keep their length.','rerank 只是给候选重新排序，块的长度不变。'),tx(`It sees only the first-stage candidates (${NCAND} here); scoring every chunk with it would be far too slow.`,`它只看第一轮的候选（这里是 ${NCAND} 块）；用它给每一块打分太慢了。`)],
   why:tx('Fewer chunks kept means fewer context tokens, as long as the answer is in them; rerank makes that more likely at the cost of one model run per candidate.','只要答案在里面，少取几块就少占上下文；rerank 让这件事更有把握，代价是每个候选跑一次模型。')}],
 transfer:[
  {q:tx('A help center keeps five chunks of 256 tokens for every question. It switches to chunks twice as long and still keeps five. How do its context tokens per question change?','一个帮助中心每个问题取五块，每块 256 个 token。换成两倍长的块，还是取五块。每个问题的上下文 token 会怎么变？'),
   o:[tx('They double','翻一倍'),tx('They stay the same','不变'),tx('They halve','减半')],a:0,
   why:tx(`Context tokens are ${KL()}: the same k with chunks twice as long means twice the tokens. A bigger chunk costs more on every question, whether or not it helps recall.`,`上下文 token 是 ${KL()}：k 不变、块长一倍，token 就多一倍。块变大，每个问题都要多花，不管召回率有没有提高。`)},
  {q:tx('A shop’s search uses vectors only. Customers type part numbers like “HX-2207” and get the wrong part. What should it add first?','一家店的搜索只用向量检索。顾客输入“HX-2207”这样的零件号，却搜到了别的零件。最先该加什么？'),
   o:[tx('Keyword search (BM25), merged with the vector results','关键词检索（BM25），和向量检索的结果合并'),tx('Bigger chunks','更大的块'),tx('More overlap between chunks','块之间更多的重叠')],a:0,
   why:tx('Part numbers must match letter for letter, which keyword search does. Vector search often cannot tell HX-2207 from HX-2208; the Anthropic cookbook says semantic search “can miss exact keyword matches”.','零件号要逐字对上，这正是关键词检索做的。向量检索常常分不清 HX-2207 和 HX-2208；Anthropic 的 cookbook 也说，语义检索“可能漏掉完全一致的关键词”（原文 “can miss exact keyword matches”）。')}],
 practice:[{id:'rag-agent-platform',d:tx('Its retrieval part: chunking, hybrid search and reranking over company documents, the pipeline this lab takes apart.','其中检索那部分：在公司文档上做切块、混合检索和 rerank，就是这个 Lab 拆开来看的流程。')},
           {id:'rag-search-ml',d:tx('Search and RAG design spoken out loud: retrieval stages, rerankers and how to measure recall.','口述搜索和 RAG 的设计：检索分几级、rerank、怎么衡量召回率。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Seven short parts. See an answer cut apart, predict which search finds an error code, compare keyword, vector and hybrid search, add rerank, then weigh recall against context tokens.','一共 7 个部分。先看一个答案被切开，预测哪种检索能找到错误码，比较关键词、向量和混合检索，加上 rerank，再权衡召回率和上下文 token。'),
  sandbox:tx('Pick a chunking and a search setup and see how many of the 20 answers reach the context.','选一种切法和检索设置，看 20 个答案里有几个能进到上下文。'),
  challenge:tx(`Choose chunk size, overlap, search, rerank and k so that at least ${GOAL()} of the answers reach the context within ${CH_CTX} tokens and ${CH_LAT} ms, with as few tokens as possible.`,`选块大小、重叠、检索方式、rerank 和 k，让至少 ${GOAL()} 的答案进到上下文，每个问题不超过 ${CH_CTX} 个 token、${CH_LAT} 毫秒，token 越少越好。`)});
const LUPIN='18d5c42edba7d160e77709fd54f680751892a3dc', LCPIN='b060a7466482ed5dac5fa1ef12e3ad21cccce385', OAPIN='ef676dbc199bc09d12a1d051f3b8b2486aa53dc2', STPIN='ff69486771998c07008ebd3c7e05e827f03df46f', FEPIN='fd1a2bdf69488ffebe0327999d4400d8c8058a0b', ACPIN='d7265d6ae994ccd8429db0594b000073b2f9ad43';
const GH=(repo,pin,path)=>`https://github.com/${repo}/blob/${pin}/${path}`;
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const U_LU=GH('apache/lucene',LUPIN,'lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java');
const U_ENS=GH('langchain-ai/langchain',LCPIN,'libs/langchain/langchain_classic/retrievers/ensemble.py');
const U_SPL=GH('langchain-ai/langchain',LCPIN,'libs/text-splitters/langchain_text_splitters/base.py');
const U_OA=GH('openai/openai-python',OAPIN,'src/openai/types/static_file_chunking_strategy_param.py');
const U_RR=GH('UKPLab/sentence-transformers',STPIN,'examples/sentence_transformer/applications/retrieve_rerank/README.md');
const U_CE=GH('UKPLab/sentence-transformers',STPIN,'examples/cross_encoder/applications/README.md');
const U_BGE=GH('FlagOpen/FlagEmbedding',FEPIN,'README.md');
const U_AC=GH('anthropics/anthropic-cookbook',ACPIN,'capabilities/contextual-embeddings/guide.ipynb');
const FIDPOP=()=>tx(`<p><b>Toy run:</b> the page really cuts, scores and ranks the lab’s own documents; the reranker is a stand-in.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Five short documents of a made-up help center and 20 questions with the sentences that answer them (示例, written for this lab).</li><li>Fixed-size chunks with overlap, or one chunk per section. Tokens: one per English word or Chinese character, the lab’s counting rule; the Chinese pages take about ${zhMore()}% more tokens than the English ones, so the same chunk size holds less.</li><li>BM25 as Lucene computes it (k₁ 1.2, b 0.75, its idf), on English words without Lucene’s English stop words, or on Chinese character pairs; RRF over each search’s first ${NCAND}, with the constant 60; recall@k: of the 20 questions, the share whose whole answer is in one kept chunk.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>Vector search uses a ${DIM}-dimensional small encoder written by hand (a word list of meanings), not a trained model.</li><li>The reranker is a stand-in that counts which of the question’s meanings and codes a chunk contains; no cross-encoder runs in the page.</li><li>Latency is an assumption: search ${LAT.kw} ms (keyword), ${LAT.vec} ms (vector), ${LAT.hyb} ms (both); rerank reads the candidates the first search returned, up to ${NCAND}, at ${f(RR_MS,1)} ms per token, each counted as long as the longest chunk. The cookbook below reports 100–200 ms for a hosted reranker.</li><li>Lucene stores chunk lengths in one byte; the lab uses exact lengths.</li></ul></div></div><p class="small"><b>Sources:</b> ${SRC(U_LU,'Lucene BM25Similarity')}, ${SRC(U_ENS,'LangChain EnsembleRetriever')}, ${SRC(U_RR,'sentence-transformers retrieve &amp; re-rank')}, ${SRC(U_AC,'Anthropic cookbook: contextual retrieval')}.</p>`,
  `<p><b>玩具实跑：</b>页面真的在切块、打分、排序这个 Lab 自己的文档；rerank 用的是替身。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>一个虚构帮助中心的 5 篇短文档，和 20 个问题以及回答它们的句子（示例，为这个 Lab 编写）。</li><li>固定大小加重叠切块，或每节一块。token：英文一个词、中文一个字算 1 个，这是本 Lab 的数法；中文页面比英文多出约 ${zhMore()}% 的 token，所以同样的块大小装的内容少一些。</li><li>BM25 按 Lucene 的算法（k₁ 1.2、b 0.75、它的 idf），英文按词、去掉 Lucene 英文的停用词，中文按相邻两个字；RRF 只看每种检索的前 ${NCAND} 块，常数是 60；recall@k：20 个问题里，完整答案落在取回的某一块里的，占几成。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul><li>向量检索用的是一个手写的 ${DIM} 维小编码器（一张按意思分类的词表），不是训练出来的模型。</li><li>rerank 是替身：数一数问题里的意思和编号，块里有哪些；页面里没有跑 cross-encoder。</li><li>延迟是假设：关键词检索 ${LAT.kw} 毫秒、向量检索 ${LAT.vec} 毫秒、两者都跑 ${LAT.hyb} 毫秒；rerank 读第一轮返回的候选，最多 ${NCAND} 块，每个 token ${f(RR_MS,1)} 毫秒，每块按最长那一块算。下面的 cookbook 报告托管的 rerank 服务要 100–200 毫秒。</li><li>Lucene 把块的长度压成一个字节存，本 Lab 用精确长度。</li></ul></div></div><p class="small"><b>来源：</b>${SRC(U_LU,'Lucene BM25Similarity')}，${SRC(U_ENS,'LangChain EnsembleRetriever')}，${SRC(U_RR,'sentence-transformers retrieve &amp; re-rank')}，${SRC(U_AC,'Anthropic cookbook：contextual retrieval')}。</p>`);
const PRIPOP=()=>tx(`<p>An embedding turns a text into a vector; texts with similar meanings get vectors pointing in similar directions, and vector search returns the passages with the highest cosine similarity. The ${LL('embeddings')} lab trains such an encoder. A token is the unit a model reads; ${LL('tokenization-cost')} shows how text is cut into tokens.</p>`,
                    `<p>embedding 把一段文字变成一个向量；意思相近的文字，向量方向相近。向量检索返回余弦相似度最高的段落。${LL('embeddings')} 那个 Lab 里训练过这样一个编码器。token 是模型读文字的单位；文字怎么切成 token，见 ${LL('tokenization-cost')}。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'apps',outcome:esc(OUTCOME()),short:tx('Chunking & rerank','切块与 rerank'),title:tx('Why does your retriever miss the right passage?','检索为什么漏掉了对的那段？'),
  minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: embeddings and tokens','前置知识：embedding 和 token'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
function stageHTML(){
  return `<div class="panel stage ckr-stage" id="ckr-stage">`+
   `<div class="stage-head ckr-head"><div class="ckr-st-title" id="ckr-st-title"></div><div class="stage-actions"><span class="status" id="ckr-status" tabindex="0"></span></div></div>`+
   `<div class="ckr-qrow"><label class="ckr-ql" for="ckr-qsel">${tx('Question','问题')}</label><select id="ckr-qsel" class="ckr-qsel" data-key="q"></select></div>`+
   `<section class="ckr-pan" id="ckr-pan-doc"><div class="ckr-pt" id="ckr-d-title"></div><div class="ckr-transport" id="ckr-transport"></div>`+
     `<div class="ckr-scap mono" id="ckr-d-cap"></div><svg class="ckr-strip" id="ckr-strip" viewBox="0 0 600 76" role="img" aria-label=""></svg><div class="ckr-scap mono" id="ckr-d-cnt"></div><div class="legend-row ckr-leg" id="ckr-d-leg"></div><div class="ckr-text" id="ckr-text"></div></section>`+
   `<section class="ckr-pan" id="ckr-pan-chart"><div class="ckr-pt" id="ckr-c-title"></div><svg class="ckr-chart" id="ckr-chart" viewBox="0 0 440 200" role="img" aria-label=""></svg><div class="legend-row ckr-leg" id="ckr-c-leg"></div></section>`+
   `<section class="ckr-pan" id="ckr-pan-lists"><div class="ckr-pt" id="ckr-l-title"></div><div class="ckr-cols" id="ckr-cols"></div><div class="legend-row ckr-leg" id="ckr-l-leg"></div></section>`+
   `<div class="metrics ckr-metrics" id="ckr-metrics"></div>`+
   `<details class="math" id="ckr-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="ckr-mathbody"></div></details></div>`;}
function controlsHTML(){
  const seg=(key,label,opts,note)=>`<div class="lockable">`+A.ui.seg({id:'ckr-ctl-'+key,key,label,options:opts,value:String(S[key]),note})+`</div>`;
  return `<aside class="panel controls" id="ckr-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Watch the cut in the document strip and the lists of chunks found.','看文档条上怎么切，和找到的块的列表。')}</p><p class="ctl-lock" id="ckr-lock" hidden></p>`+
   seg('size',tx('Chunk size','块大小'),SIZES.map(s=>[String(s),sizeShort(s)]),tx(`Tokens per chunk, or one chunk per section. The lab’s documents are short (${docRange()[0]} to ${docRange()[1]} tokens a page), so its chunks are small too; real systems use hundreds of tokens a chunk (OpenAI’s file search defaults to 800).`,`每块多少个 token，或每节一块。这个 Lab 的文档很短（每页 ${docRange()[0]} 到 ${docRange()[1]} 个 token），所以块也小；真实系统一块几百个 token（OpenAI 的文件搜索默认 800）。`))+
   seg('ov',tx('Overlap','重叠'),OV_N.map((o,i)=>[String(i),L(o)]),tx('How much of a chunk the next one repeats. An answer no longer than the overlap is always whole in some chunk. OpenAI’s file search defaults to half (400 of 800 tokens).','下一块重复上一块的多少。长度不超过重叠的答案，一定完整地落在某一块里。OpenAI 的文件搜索默认重叠一半（800 里的 400）。'))+
   seg('ret',tx('First retrieval','先用哪种检索'),RETS.map(r=>[r,L(RET_S[r])]),tx('Keyword: BM25 on words (Chinese: character pairs). Vector: cosine with the small encoder. Hybrid: both, merged by RRF with the constant 60.','关键词：按词算 BM25（中文按相邻两个字）。向量：小编码器的余弦。混合：两种都跑，用常数 60 的 RRF 合并。'))+
   seg('rr',tx('Rerank','rerank'),[['0',tx('Off','不用')],['1',tx('On','用')]],tx(`On: the first ${NCAND} chunks of the first retrieval are rescored by the lab’s stand-in reranker before k are kept.`,`用：取第一轮的前 ${NCAND} 块，用本 Lab 的 rerank 替身重新打分，再留下 k 块。`))+
   seg('k',tx('Chunks kept, k','取前 k 块'),KS.map(k=>[String(k),String(k)]),tx(`How many chunks go into the context. Context tokens: ${KL()}, L being the length of the longest chunk.`,`放进上下文的块数。上下文 token：${KL()}，L 是最长那一块的 token 数。`))+
   `<details class="adv"><summary>${tx('Assumed in this lab: latency, rerank candidates','这个 Lab 假设的：延迟、rerank 的候选数')}</summary>`+
   `<p class="ctl-note">${tx(`Search time ${LAT.kw} ms (keyword), ${LAT.vec} ms (vector: embed the question and compare), ${LAT.hyb} ms (both). Rerank reads the candidates the first search returned, up to ${NCAND}, at ${f(RR_MS,1)} ms per token, each counted as long as the longest chunk: ${f(NCAND*64*RR_MS,0)} ms for 64-token chunks. These are the lab’s assumptions, chosen so that rerank costs the same order as the 100–200 ms the Anthropic cookbook reports for a hosted reranker; real times depend on the model, the hardware and the batch.`,`检索时间：关键词 ${LAT.kw} 毫秒，向量 ${LAT.vec} 毫秒（把问题变成向量再比），两者都跑 ${LAT.hyb} 毫秒。rerank 读第一轮返回的候选，最多 ${NCAND} 块，每个 token ${f(RR_MS,1)} 毫秒，每块按最长那一块算：64 个 token 的块要 ${f(NCAND*64*RR_MS,0)} 毫秒。这些是本 Lab 的假设，让 rerank 的耗时和 Anthropic cookbook 报告的托管 rerank 服务的 100–200 毫秒在同一个量级；真实的时间取决于模型、硬件和 batch。`)}</p>`+
   `<p class="ctl-note">${tx(`Why only ${NCAND} candidates: real systems rerank about 100 or 150 candidates out of millions of chunks. This help center has only ${setup(LG(),64,0).N} chunks at 64 tokens, so a pool of 100 would hold every chunk and the first search would no longer matter. A pool of ${NCAND} keeps it smaller than the index, as in a real system.`,`为什么只有 ${NCAND} 块候选：真实系统从 1M 以上的块里取大约 100 或 150 块给 rerank。这个帮助中心切成 64 个 token 的块，一共只有 ${setup(LG(),64,0).N} 块，取 100 块就把所有块都放进去了，第一轮检索也就不起作用了。取 ${NCAND} 块，候选池才像真实系统一样比整个索引小。`)}</p></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import math
from sentence_transformers.cross_encoder import CrossEncoder

def chunks(tokens, size=64, overlap=32):
    step = size - overlap
    for s in range(0, len(tokens), step):
        yield tokens[s:s + size]
        if s + size >= len(tokens):
            break

def bm25(q, d, df, n, avgdl, k1=1.2, b=0.75):
    s = 0.0
    for t in set(q):
        f = d.count(t)
        if f:
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f / (f + k1 * (1 - b + b * len(d) / avgdl))
    return s

def rrf(*rankings, c=60):
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0) + 1 / (c + r)
    return sorted(score, key=score.get, reverse=True)

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def search(query, texts, kw_rank, vec_rank, keep=1, cand=5):
    first = rrf(kw_rank[:cand], vec_rank[:cand])[:cand]
    scores = ce.predict([(query, texts[i]) for i in first])
    best = sorted(zip(scores, first), reverse=True)[:keep]
    return [i for _, i in best]`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ckr-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ckr-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Fixed-size chunks with overlap, BM25, reciprocal rank fusion and a cross-encoder rerank, in Python.','用 Python 写的：固定大小加重叠的切块、BM25、倒数排名融合和 cross-encoder rerank。')}</p><pre class="code" id="ckr-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ckr-prod">`+
   tx(`<li>Lucene ${SRC(U_LU,'<code>BM25Similarity.java</code>')}: the defaults <code>k1 = 1.2</code>, <code>b = 0.75</code> and idf <code>log(1 + (docCount − docFreq + 0.5)/(docFreq + 0.5))</code>, the BM25 this lab computes.</li>`+
      `<li>LangChain ${SRC(U_ENS,'<code>EnsembleRetriever</code>')}: merges several retrievers by weighted reciprocal rank fusion with <code>c: int = 60</code>. Its ${SRC(U_SPL,'text splitter')} defaults to <code>chunk_size=4000</code> and <code>chunk_overlap=200</code>, measured in characters.</li>`+
      `<li>OpenAI ${SRC(U_OA,'<code>static_file_<wbr>chunking_<wbr>strategy_<wbr>param.py</code>')}: file search chunks of 800 tokens with 400 tokens of overlap by default; the overlap may not exceed half the chunk.</li>`+
      `<li>sentence-transformers ${SRC(U_RR,'retrieve &amp; re-rank')}: retrieve candidates (“e.g. 100”) with a bi-encoder (question and passage each turned into a vector: vector search) or lexical search, rescore them with a ${SRC(U_CE,'<code>CrossEncoder</code>')}, show the top 5. BGE’s ${SRC(U_BGE,'README')} recommends its cross-encoder rerankers “to re-rank top-k documents returned by embedding models”.</li>`+
      `<li>Anthropic’s ${SRC(U_AC,'contextual retrieval cookbook')}: semantic search “can miss exact keyword matches”, so it adds BM25 (top 150 from each) and then a reranker, which adds about 100–200 ms per query.</li>`,
      `<li>Lucene ${SRC(U_LU,'<code>BM25Similarity.java</code>')}：默认 <code>k1 = 1.2</code>、<code>b = 0.75</code>，idf 是 <code>log(1 + (docCount − docFreq + 0.5)/(docFreq + 0.5))</code>，就是本 Lab 算的 BM25。</li>`+
      `<li>LangChain ${SRC(U_ENS,'<code>EnsembleRetriever</code>')}：用加权的倒数排名融合把几个检索器合起来，<code>c: int = 60</code>。它的${SRC(U_SPL,'文本切分器')}默认 <code>chunk_size=4000</code>、<code>chunk_overlap=200</code>，单位是字符。</li>`+
      `<li>OpenAI ${SRC(U_OA,'<code>static_file_<wbr>chunking_<wbr>strategy_<wbr>param.py</code>')}：文件搜索默认每块 800 个 token、重叠 400 个；重叠不能超过块的一半。</li>`+
      `<li>sentence-transformers ${SRC(U_RR,'retrieve &amp; re-rank')}：先用双塔模型（bi-encoder，问题和段落各自变成向量，也就是向量检索）或关键词检索取一批候选（比如 100 个），再用 ${SRC(U_CE,'<code>CrossEncoder</code>')} 重新打分，给用户看前 5 个。BGE 的 ${SRC(U_BGE,'README')} 建议用它的 cross-encoder rerank 模型“给 embedding 模型返回的前 k 个文档重新排序”（原文 “to re-rank top-k documents returned by embedding models”）。</li>`+
      `<li>Anthropic 的 ${SRC(U_AC,'contextual retrieval cookbook')}：语义检索“可能漏掉完全一致的关键词”（原文 “can miss exact keyword matches”），所以加上 BM25（两边各取前 150），再接一个 rerank，每次查询多花约 100–200 毫秒。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: Lucene 18d5c42, LangChain b060a74, openai-python ef676db, sentence-transformers ff69486, FlagEmbedding fd1a2bd, anthropic-cookbook d7265d6.','固定的 commit：Lucene 18d5c42，LangChain b060a74，openai-python ef676db，sentence-transformers ff69486，FlagEmbedding fd1a2bd，anthropic-cookbook d7265d6。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2005.11401" target="_blank" rel="noopener">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</a><span>${tx('Lewis et al., 2020: RAG','Lewis 等，2020：RAG')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/1901.04085" target="_blank" rel="noopener">Passage Re-ranking with BERT</a><span>${tx('Nogueira and Cho, 2019: a cross-encoder reranker','Nogueira 和 Cho，2019：cross-encoder rerank')}</span></li>`+
   `<li><span>Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods</span><span>${tx('Cormack, Clarke and Büttcher, SIGIR 2009: RRF with k = 60','Cormack、Clarke、Büttcher，SIGIR 2009：RRF，k = 60')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ckr-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ckr-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="ckr-app m2" id="ckr-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="ckr-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels and list columns the stage shows now */
function ctx(){
  if(mode==='guided'){const s=STEPS[step]; return {panels:s.panels,cols:s.cols?s.cols():[]};}
  if(mode==='challenge'&&!chSubmitted) return {panels:['doc'],cols:[]};
  return {panels:['doc','lists','chart','metrics'],cols:S.rr?[S.ret+'0',S.ret+'1']:[S.ret+'0']};}

/* ---------- drawing ---------- */
const docOf=qi=>DOC_IDS.indexOf(QS[qi][2]);
/* Chinese text keeps a space between a Latin run (E-201, App) and a Chinese character, as the page's own copy does */
const zsp=(tk,i)=>!tk[i].tl.trim()&&!(tk[i].h&&tk[i].end)&&i+1<tk.length&&isCJK(tk[i].t)!==isCJK(tk[i+1].t)?' ':'';
/* what follows a token when the text is shown: its punctuation, then a space between words and after a sentence (a heading's colon is added by the caller) */
const after=(tk,i,zh)=>{const x=tk[i]; if(zh) return x.tl.trim()+zsp(tk,i); if(x.h&&x.end) return x.tl.trim(); if(!x.tl) return ' '; return x.end&&!/\s$/.test(x.tl)?x.tl+' ':x.tl;};
const chunkName=(R,j)=>{const c=R.C[j], d=DOC_IDS[c.d]; const sec=c.sec!=null?esc(DOCS[c.d][2][c.sec][0][li()]):''; return c.sec!=null?tx(`${L(DOC_N[d])}, section “${sec}”`,`${L(DOC_N[d])}的“${sec}”一节`):tx(`${L(DOC_N[d])}, chunk ${c.no}`,`${L(DOC_N[d])}第 ${c.no} 块`);};
function snippet(R,j,n){const c=R.C[j], tk=build(LG())[c.d].toks, zh=LG()==='zh', m=n||(zh?14:8);
  let s=''; for(let i=c.a;i<Math.min(c.b,c.a+m);i++) s+=tk[i].t+after(tk,i,zh)+(tk[i].h&&tk[i].end?(zh?'：':': '):''); return s.trim()+(c.b>c.a+m?'…':'');}
/* the progress of the cut: how many of the document's chunks are drawn (all of them unless the cut is playing) */
function cutShown(n){return anim?Math.min(n,Math.floor(anim.p*n+1e-9)):n;}
function drawStrip(){
  const R=setup(LG(),S.size,S.ov), q=R.Q[S.q], d=q.d, doc=build(LG())[d], n=doc.toks.length, mine=R.C.filter(c=>c.d===d), shown=cutShown(mine.length), done=!anim;
  const svg=$('#ckr-strip',app), W=Math.round(Math.max(300,Math.min(600,svg.clientWidth||600)));   // one user unit per pixel up to 600, so the numbers stay legible at 390 (R7.4)
  const x0=8, x1=W-8, X=t=>x0+(x1-x0)*t/n, yD=8, hD=12, yL=[30,50], hL=14;
  const whole=q.right.length>0, cov=new Array(n).fill(0);
  mine.slice(0,shown).forEach(c=>{for(let t=c.a;t<c.b;t++) cov[t]++;});
  let h=`<rect class="ckr-docbar" x="${x0}" y="${yD}" width="${x1-x0}" height="${hD}" rx="2"></rect>`;
  let t=0; while(t<n){if(cov[t]>1){let e=t; while(e<n&&cov[e]>1) e++; h+=`<rect class="ckr-ovl" x="${X(t)}" y="${yD}" width="${X(e)-X(t)}" height="${hD}"></rect>`; t=e;} else t++;}
  doc.secs.forEach(s=>{h+=`<line class="ckr-sec" x1="${X(s[0])}" y1="${yD-3}" x2="${X(s[0])}" y2="${yD+hD+3}"></line>`;});
  const ansCls=done?(whole?'ok':'bad'):'';
  h+=`<rect class="ckr-ans ${ansCls}" x="${X(q.sa)}" y="${yD-5}" width="${Math.max(2,X(q.sb)-X(q.sa))}" height="${yL[1]+hL-yD+10}" rx="2"></rect>`;
  mine.slice(0,shown).forEach((c,i)=>{const y=yL[i%2], w=X(c.b)-X(c.a), isR=done&&q.right.includes(c.j), isP=done&&!whole&&q.part.includes(c.j);
    h+=`<rect class="ckr-ch${isR?' ok':''}${isP?' bad':''}" x="${X(c.a)+0.5}" y="${y}" width="${Math.max(1,w-1)}" height="${hL}" rx="3"></rect>`;
    if(w>=(c.no>9?16:11)) h+=`<text class="ckr-chn" x="${X(c.a)+w/2}" y="${y+10.5}" text-anchor="middle">${c.no}</text>`;});
  const cut=done&&!whole?q.part.slice(1).map(j=>R.C[j].a).filter(a=>a>q.sa&&a<q.sb):[];
  if(done&&!whole) mine.forEach(c=>{if(q.part.includes(c.j)&&c.b>q.sa&&c.b<q.sb) cut.push(c.b);});
  [...new Set(cut)].forEach(a=>{h+=`<line class="ckr-cut" x1="${X(a)}" y1="${yD-6}" x2="${X(a)}" y2="${yL[1]+hL+6}"></line>`;});
  svg.setAttribute('viewBox',`0 0 ${W} 76`); svg.innerHTML=h;
  svg.setAttribute('aria-label',done?(whole?tx('The answer lies whole inside one chunk.','答案完整地落在一块里。'):tx('The answer is cut apart: no chunk holds all of it.','答案被切开了：没有哪一块装着完整的答案。')):tx('The document is being cut into chunks.','文档正在被切成块。'));
  $('#ckr-d-cap',app).textContent=tx(`${L(DOC_N[DOC_IDS[d]])} (${n} tokens)`,`${L(DOC_N[DOC_IDS[d]])}（${n} 个 token）`);
  $('#ckr-d-cnt',app).textContent=anim?tx(`Cutting: ${shown} of ${mine.length} chunks`,`正在切：第 ${shown} 块，共 ${mine.length} 块`):tx(`${mine.length} chunks in this document; ${R.N} in the help center`,`这篇切成 ${mine.length} 块；整个帮助中心 ${R.N} 块`);
  $('#ckr-d-title',app).textContent=S.size==='h'?tx('How the page that answers this question is cut (by heading)','回答这个问题的那一页是怎么切的（按标题切）')
    :tx(`How the page that answers this question is cut (${sizeTxt(S.size)} a chunk, ${ovTxt(S.size,S.ov)})`,`回答这个问题的那一页是怎么切的（${sizeTxt(S.size)}、${ovTxt(S.size,S.ov)}）`);
  drawDocLegend();
  drawText(R,q,done);}
function drawDocLegend(){const el=$('#ckr-d-leg',app), k=A.lang(); if(el._k===k) return; el._k=k;
  el.innerHTML=`<span class="lg" data-tipx="${esc(tx('One bar per chunk, numbered in order; neighbouring chunks are drawn on two rows so that overlaps show.','每条是一块，按顺序编号；相邻的块画在上下两行，重叠的部分才看得出来。'))}"><i class="sw ckr-sw-ch"></i>${tx('chunk','块')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('Tokens that belong to two chunks: they are stored twice.','同时属于两块的 token：存了两次。'))}"><i class="sw ckr-sw-ov"></i>${tx('overlap','重叠')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('A short line on the document bar: a section heading starts here.','文档条上的短竖线：一节的标题从这里开始。'))}"><i class="sw ckr-sw-sec"></i>${tx('heading','标题')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('The sentences that answer the question. Green: one chunk holds all of them. Red: the cut splits them.','回答这个问题的句子。绿色：一块装下了全部。红色：被一刀切开。'))}"><i class="sw ckr-sw-ok"></i><i class="sw ckr-sw-bad"></i>${tx('answer: whole / cut apart','答案：完整 / 被切开')}</span>`;}
function drawText(R,q,done){const el=$('#ckr-text',app), tk=build(LG())[q.d].toks, zh=LG()==='zh';
  if(!done){el.innerHTML=`<p class="small muted">${tx('The chunks that hold the answer show when the cut is done.','切完以后，这里显示装着答案的块。')}</p>`; return;}
  const show=q.right.length?[q.right[0]]:q.part, ctxN=zh?10:6;
  el.innerHTML=show.map(j=>{const c=R.C[j], a=Math.max(c.a,q.sa-ctxN), b=Math.min(c.b,q.sb+ctxN), ok=q.right.length>0;
    let s=a>c.a?(zh?'…':'… '):'';
    for(let i=a;i<b;i++){const inA=i>=q.sa&&i<q.sb, w=esc(tk[i].t)+esc(after(tk,i,zh)); s+=inA?`<mark class="ckr-m${ok?' ok':' bad'}">${w}</mark>`:w; if(tk[i].h&&tk[i].end) s+=zh?'：':': ';}
    if(b<c.b) s+=zh?'…':' …';
    return `<div class="ckr-tx${ok?' ok':' bad'}"><span class="ckr-lab">${chunkName(R,j)}${ok?'':tx(' (part of the answer)','（只有一部分答案）')}</span><p>${s}</p></div>`;}).join('')+
   (q.right.length?'':`<p class="small ckr-bad">${tx('The cut falls inside the answer: each chunk holds only part of it, so no single chunk can give the whole answer.','这一刀落在答案中间：每一块只装着一部分，没有哪一块能单独给出完整的答案。')}</p>`);}
const colTitle=c=>{const r=c.slice(0,-1), rr=c.slice(-1)==='1'; return rr?tx(`${L(RET_S[r])[0].toUpperCase()+L(RET_S[r]).slice(1)} + rerank`,`${L(RET_S[r])}检索 + rerank`):L(RET_N[r]);};
function scoreTxt(q,c,j){const r=c.slice(0,-1);
  if(c.slice(-1)==='1') return tx(`covers ${pc(q.cov[j])}`,`覆盖 ${pc(q.cov[j])}`);
  if(r==='kw') return f(q.kw[j],2); if(r==='vec') return f(q.vec[j],2); return f(q.rrf[j],4);}
function drawLists(){
  const R=setup(LG(),S.size,S.ov), q=R.Q[S.q], cols=ctx().cols, k=S.k, SHOW=NCAND;
  $('#ckr-l-title',app).textContent=tx(`What each search returns for this question (up to ${SHOW} chunks); ${k===1?'only the first goes':`the first ${k} go`} into the context`,`每种检索为这个问题返回的块（最多列 ${SHOW} 块）；${k===1?'只有第 1 块':`前 ${k} 块`}进上下文`);
  const host=$('#ckr-cols',app); host.classList.toggle('dim',!!anim); host.style.setProperty('--n',cols.length); host.dataset.n=cols.length;
  host.innerHTML=cols.map(c=>{const r=c.slice(0,-1), isRr=c.slice(-1)==='1', lst=isRr?q.rr[r]:q.lists[r], rk=q.rk[c], firstRk=q.rk[r+'0'];
    let h=`<div class="ckr-col"><div class="ckr-ch-t">${esc(colTitle(c))}</div><ol class="ckr-ol">`;
    lst.slice(0,SHOW).forEach((j,i)=>{const right=q.right.includes(j), cls=right?(i<k?' ok':' bad'):'';
      h+=`<li class="ckr-li${cls}${i===k-1&&k<Math.min(SHOW,lst.length)?' cutk':''}"><span class="ckr-rk mono">${i+1}</span><span class="ckr-nm"><b>${chunkName(R,j)}</b>${right?(i<k?' ✓':tx(' (not kept)','（没取到）')):''}<span class="muted ckr-sn">${esc(snippet(R,j))}</span></span><span class="ckr-sc mono">${esc(scoreTxt(q,c,j))}</span></li>`;});
    h+='</ol>';
    if(lst.length<SHOW&&!isRr) h+=`<p class="small muted">${tx(`${colTitle(c)} returns only ${lst.length} chunk${lst.length===1?'':'s'}: the others score 0.`,`${L(RET_S[r])}检索只返回了 ${lst.length} 块：其余的块得分是 0。`)}</p>`;
    if(!q.right.length) h+=`<p class="small ckr-bad">${tx('No chunk holds the whole answer.','没有哪一块装着完整的答案。')}</p>`;
    else if(rk<0&&r==='kw'&&kwZero(q)) h+=`<p class="small ckr-bad">${tx('The chunk with the answer shares no word with the question: keyword search does not return it.','装着答案的块和问题没有共同的词：关键词检索不会返回它。')}</p>`;
    else if(rk<0) h+=`<p class="small ckr-bad">${tx('This search does not return the chunk with the answer.','这种检索不返回装着答案的块。')}</p>`;
    else if(isRr&&firstRk>=NCAND) h+=`<p class="small ckr-bad">${tx(`The chunk with the answer ranks ${firstRk+1} in the first search, outside the ${NCAND} candidates, so rerank cannot reach it.`,`装着答案的块在第一轮排第 ${firstRk+1}，不在 ${NCAND} 块候选里，rerank 够不着它。`)}</p>`;
    else if(rk>=SHOW) h+=`<p class="small ckr-bad">${tx(`The chunk with the answer ranks ${rk+1}: not in the context${S.rr?`, and outside the ${NCAND} candidates of rerank`:''}.`,`装着答案的块排第 ${rk+1}：没进上下文${S.rr?`，也不在 rerank 的 ${NCAND} 块候选里`:''}。`)}</p>`;
    return h+'</div>';}).join('');
  const el=$('#ckr-l-leg',app), key=A.lang()+cols.join(); if(el._k!==key){el._k=key;
    el.innerHTML=`<span class="lg" data-tipx="${esc(tx('The chunk that holds the whole answer, among the chunks kept.','装着完整答案的块，而且在取回的块里。'))}"><i class="sw ckr-sw-okb"></i>${tx('right chunk, kept','正确的块，取到了')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('The chunk that holds the answer ranks below k, so it stays out of the context.','装着答案的块排在第 k 名之后，没进上下文。'))}"><i class="sw ckr-sw-badb"></i>${tx('right chunk, not kept','正确的块，没取到')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('The dashed line under rank k: chunks above it go into the context.','第 k 名下面的虚线：线上面的块进上下文。'))}"><i class="sw ckr-sw-k"></i>${tx('cut-off at k','第 k 名的分界')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('The number on the right is the chunk’s score in that list: BM25 for keyword search, cosine for vector search, the RRF sum for hybrid search, and for rerank the share of the question’s meanings and codes the chunk contains.','右边的数是这一块在这个列表里的分数：关键词检索是 BM25，向量检索是余弦，混合检索是 RRF 的和，rerank 是这一块包含了问题里几成的意思和编号。'))}"><i class="sw ckr-sw-sc"></i>${tx('score','分数')}</span>`;}}
const OV_C=[{en:'no overlap',zh:'不重叠'},{en:'overlap ¼ chunk',zh:'重叠 1/4 块'},{en:'overlap ½ chunk',zh:'重叠半块'}];
function drawChart(){
  const svg=$('#ckr-chart',app), W=Math.round(Math.max(300,Math.min(440,svg.clientWidth||440))), H=200, X0=56, Y0=14, Y1=H-44, n=SIZES.length, nf=n-1;   // one unit per pixel at phone width (R7.4)
  const Xh=W-24, Xa=Xh-84, Xi=i=>i<nf?X0+(Xa-X0)*i/(nf-1):Xh, Y=v=>Y1-(Y1-Y0)*v;   // the last x is 'by heading': a different cut, drawn apart and not joined (R5.2); the line labels sit between
  let h='';
  [0,0.25,0.5,0.75,1].forEach(v=>{h+=`<line class="ckr-grid" x1="${X0}" x2="${Xh+8}" y1="${Y(v)}" y2="${Y(v)}"></line><text class="ckr-ax" x="${X0-6}" y="${Y(v)+3.5}" text-anchor="end">${f(v*100,0)}%</text>`;});
  h+=`<line class="ckr-sep" x1="${Xh-16}" x2="${Xh-16}" y1="${Y0}" y2="${Y1+30}"></line>`;
  SIZES.forEach((s,i)=>{const c=measure(LG(),{size:s,ov:S.ov,ret:S.ret,rr:S.rr,k:S.k}).ctx; h+=`<text class="ckr-ax" x="${Xi(i)}" y="${Y1+14}" text-anchor="middle">${esc(sizeShort(s))}</text><text class="ckr-ctx" x="${Xi(i)}" y="${Y1+27}" text-anchor="middle">${A.fmt.int(c)}</text>`;});
  h+=`<text class="ckr-ax" x="2" y="${Y1+14}" text-anchor="start">${esc(tx('size','块'))}</text><text class="ckr-ctx" x="2" y="${Y1+27}" text-anchor="start">${esc(tx('context','上下文'))}</text>`;
  h+=`<text class="ckr-ax" x="${W/2}" y="${H-5}" text-anchor="middle">${esc(tx(`size in tokens; below: context tokens at k = ${S.k}`,`块大小（token）；下一行：k = ${S.k} 时的上下文 token`))}</text>`;
  const pts={}; RETS.forEach(r=>{pts[r]=SIZES.map(s=>measure(LG(),{size:s,ov:S.ov,ret:r,rr:S.rr,k:S.k}).rec);
    h+=`<polyline class="ckr-ln ckr-ln-${r}${r===S.ret?' cur':''}" points="${pts[r].slice(0,nf).map((v,i)=>`${Xi(i)},${Y(v)}`).join(' ')}"></polyline>`;
    pts[r].forEach((v,i)=>{h+=`<circle class="ckr-dot ckr-ln-${r}" cx="${Xi(i)}" cy="${Y(v)}" r="${i===nf?3.4:2.6}"></circle>`;});});
  const lab=RETS.map(r=>({r,y:Y(pts[r][nf-1])})).sort((a,b)=>a.y-b.y); for(let i=1;i<lab.length;i++) if(lab[i].y-lab[i-1].y<12) lab[i].y=lab[i-1].y+12;
  lab.forEach(o=>{h+=`<text class="ckr-cl" x="${Xa+8}" y="${Math.min(Y1+4,o.y+3.5)}">${esc(L(RET_S[o.r]))}</text>`;});
  const i0=SIZES.indexOf(DEFAULT.size), r0=measure(LG(),DEFAULT).rec, x0=Xi(i0), y0=Y(r0);   // the initial setup, marked whatever the chart shows (R5.4)
  h+=`<path class="ckr-init" d="M${x0} ${y0-5}L${x0+5} ${y0}L${x0} ${y0+5}L${x0-5} ${y0}Z"></path>`;
  const ci=SIZES.indexOf(S.size); h+=`<circle class="ckr-me" cx="${Xi(ci)}" cy="${Y(pts[S.ret][ci])}" r="6.5"></circle>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML=h;
  const cond=S.size==='h'?'':L(OV_C[S.ov])+tx(', ','，');
  $('#ckr-c-title',app).textContent=tx(`How large should a chunk be? Answers in the context out of 20, by chunk size (${cond}${S.rr?'with rerank, ':''}keep ${S.k})`,`块切多大，20 个答案里有几成能进上下文？（${cond}${S.rr?'用 rerank，':''}取 ${S.k} 块）`);
  const el=$('#ckr-c-leg',app), key=A.lang(); if(el._k!==key){el._k=key;
    el.innerHTML=RETS.map(r=>`<span class="lg" data-tipx="${esc(L(RET_N[r]))}"><i class="sw ckr-sw-${r}"></i>${esc(L(RET_S[r]))}</span>`).join('')+
     `<span class="lg" data-tipx="${esc(tx('The ring marks your current chunk size and search.','圆圈是你现在的块大小和检索方式。'))}"><i class="sw ckr-sw-me"></i>${tx('your setting','你的设置')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('The initial setup: 64-token chunks, no overlap, vector search, no rerank, keep 1.','初始设置：每块 64 个 token、不重叠、向量检索、不用 rerank、取 1 块。'))}"><i class="sw ckr-sw-init"></i>${tx('initial setup','初始设置')}</span>`+
     `<span class="lg" data-tipx="${esc(tx('Cutting by heading is a different cut, one chunk per section, so its point is not joined to the line.','按标题切是另一种切法，每节一块，所以这个点不和折线连在一起。'))}"><i class="sw ckr-sw-sep"></i>${tx('by heading: apart','按标题切：单独一点')}</span>`;}}
const MET=[
 {k:'rec',label:()=>tx(`Recall@${S.k}`,`recall@${S.k}`),v:m=>pc(m.rec),sub:m=>tx(`${m.hits} of 20 questions`,`20 个问题里 ${m.hits} 个`),tip:()=>D_REC(),pts:true},
 {k:'ctx',label:()=>tx('Context tokens','上下文 token'),v:m=>A.fmt.int(m.ctx),sub:m=>tx(`${S.k} ${S.k===1?'chunk':'chunks'}, each up to ${m.R.longest}`,`取 ${S.k} 块，每块最多 ${m.R.longest}`),tip:()=>D_CTX()},
 {k:'lat',label:()=>tx('Time','耗时'),v:m=>msTxt(m.lat),sub:m=>S.rr?tx(`rerank ${msTxt(m.rerankMs)} of it (${m.nc} candidates)`,`其中 rerank ${msTxt(m.rerankMs)}（${m.nc} 块候选）`):tx('no rerank','不用 rerank'),tip:()=>tx(`This lab’s latency assumption: search ${LAT.kw} ms (keyword), ${LAT.vec} ms (vector), ${LAT.hyb} ms (both). Rerank reads the candidates the first search returned, up to ${NCAND}, at ${f(RR_MS,1)} ms per token, each counted as long as the longest chunk; the question with the most candidates counts.`,`本 Lab 的延迟假设：检索 ${LAT.kw} 毫秒（关键词）、${LAT.vec} 毫秒（向量）、${LAT.hyb} 毫秒（两者）。rerank 读第一轮返回的候选，最多 ${NCAND} 块，每个 token ${f(RR_MS,1)} 毫秒，每块按最长那一块算；按候选最多的那个问题算。`)},
 {k:'spl',label:()=>tx('Answers cut','被切开的答案'),v:m=>tx(`${m.splits} of 20`,`${m.splits} / 20`),sub:m=>tx(`${m.R.N} chunks in all`,`一共 ${m.R.N} 块`),tip:()=>tx('Questions whose answer no single chunk holds. Keeping 1 chunk, the model sees at most part of the answer; this lab counts a question only when one chunk holds the whole answer.','没有哪一块装着完整答案的问题。只取 1 块时，模型最多看到半个答案；本 Lab 只在一块装着完整答案时才算找到。')}];
let metKey=null;
function drawMetrics(){
  const m=measure(LG(),S), host=$('#ckr-metrics',app), key=JSON.stringify(chPick()), cfgCh=metKey!=null&&metKey!==key; metKey=key;
  if(!host.children.length||host._lang!==A.lang()){host.innerHTML=MET.map(x=>`<div class="metric" data-k="${x.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=A.lang();}
  MET.forEach(x=>{const el=host.querySelector(`[data-k="${x.k}"]`), v=x.v(m), d=el.querySelector('.d');
    el.querySelector('.k').textContent=x.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=x.sub(m); el.dataset.tipx=x.tip();
    let dt=''; const num={rec:Math.round(m.rec*100),ctx:m.ctx,lat:m.lat,spl:m.splits}[x.k];
    if(userAct&&cfgCh&&shownPrev[x.k]!=null&&shownPrev[x.k]!==num) dt=x.pts?A.fmt.delta(shownPrev[x.k],num,{pts:true,dp:0}):A.fmt.delta(shownPrev[x.k],num,{dp:0});
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else {d.classList.remove('on'); d.textContent='';}
    shownPrev[x.k]=num;});}
function drawHead(){
  const st=$('#ckr-st-title',app), el=$('#ckr-status',app), m=measure(LG(),S);
  st.textContent=anim?tx(`Cutting the page into chunks: ${setTxt(S)}`,`正在把这一页切成块：${setTxt(S)}`):tx(`Searching the help center (example): ${setTxt(S)}`,`在帮助中心（示例）里检索：${setTxt(S)}`);
  if(mode==='challenge'){const me=chMe(), done=chDone(); el.className='status'+(done?(me.ok?' ok':' bad'):''); el.textContent=done?(me.ok?tx('Goal met','达标'):tx('Not met','没达标')):tx('Not submitted','还没提交');
    el.dataset.tipx=tx(`Whether your setting meets all three limits: at least ${GOAL()} of the answers in the context, at most ${CH_CTX} tokens, at most ${CH_LAT} ms. It shows once you submit.`,`你的设置是否三项都达标：至少 ${GOAL()} 的答案进到上下文、最多 ${CH_CTX} 个 token、最多 ${CH_LAT} 毫秒。提交后才显示。`); return;}
  const showRec=ctx().panels.includes('metrics')||ctx().panels.includes('lists');
  el.hidden=!showRec; el.className='status'; el.textContent=tx(`Recall@${S.k} ${pc(m.rec)}`,`recall@${S.k} ${pc(m.rec)}`); el.dataset.tipx=D_REC();}
function drawTransport(){
  const tEl=$('#ckr-transport',app), want=A.lang()+'|'+(playing()?1:0)+'|'+speed;
  if(tEl._k!==want){tEl._k=want;
    tEl.innerHTML=`<button type="button" class="btn sm" data-act="play" data-tipx="${esc(tx('Cut the page chunk by chunk; press again to pause.','一块一块地切这一页；再按一次暂停。'))}">${playing()?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Cut','▶ 切')}</button>`+
      A.ui.seg({id:'ckr-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(k=>[k,L(SPEED_N[k])]),value:speed})+
      `<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch the cut again from the start.','从头再看一遍怎么切。'))}">${tx('Replay','重放')}</button>`+
      A.ui.range({id:'ckr-tr-p',key:'prog',label:tx('Progress','进度'),min:0,max:100,value:100,out:'100%',inline:true});
    A.ui.initRanges(tEl);}
  const i=$('#ckr-tr-p-in',app), p=Math.round((anim?anim.p:1)*100); if(i&&+i.value!==p){i.value=p; A.ui.setRange(i);} const o=i&&i.closest('.inline-ctl,.ctl'), out=o&&o.querySelector('output'); if(out) out.textContent=p+'%';}
function panelsShow(){const c=ctx(); ['doc','lists','chart'].forEach(p=>{const el=$('#ckr-pan-'+p,app); if(el) el.hidden=!c.panels.includes(p);});
  $('#ckr-metrics',app).hidden=!c.panels.includes('metrics');
  const ch=$('#ckr-pan-chart',app), ls=$('#ckr-pan-lists',app), chFirst=c.panels[0]==='chart';   // the part's main picture first (R7.12)
  if(chFirst&&ch.nextElementSibling!==ls) ls.parentNode.insertBefore(ch,ls); else if(!chFirst&&ls.nextElementSibling!==ch) ls.parentNode.insertBefore(ch,ls.nextSibling);
  const qs=$('#ckr-qsel',app); qs.disabled=mode==='guided'&&['mission','chunk','predict','search','rerank'].includes(STEPS[step].id);}
function drawQsel(){const el=$('#ckr-qsel',app), k=A.lang()+'|'+S.q; if(el._k===k) return; el._k=k;
  el.innerHTML=QS.map((q,i)=>`<option value="${i}"${i===S.q?' selected':''}>${i+1}. ${esc(q[li()])}</option>`).join(''); el.value=String(S.q);}
/* Show the math: four short blocks, each a lead line, the named formula, the numbers of the current state, the result */
function mathHTML(){
  const R=setup(LG(),S.size,S.ov), q=R.Q[S.q], m=measure(LG(),S), zh=LG()==='zh', blk=(lead,tex,sub,sub2)=>`<div class="ckr-mb"><p class="ckr-mb-lead">${lead}</p>${M(tex)}${sub?M(sub):''}${sub2?M(sub2):''}</div>`;
  const j=q.right.length?q.right.slice().sort((a,b)=>q.lists.kw.indexOf(a)-q.lists.kw.indexOf(b))[0]:q.lists.kw[0], c=R.C[j];
  const t=c?q.terms.filter(x=>c.terms.includes(x)).sort((a,b)=>R.df.get(a)-R.df.get(b))[0]:null;
  const words=zh?'词（相邻两个字算一个词）':'words';
  const b0=blk(tx(`The numbers below use the question now on the stage, ${quote(qText(S.q))}. 1. BM25 adds up, over the words t of the question q, how often t occurs in the chunk (f<sub>t</sub>), weighted by idf(t), which is larger for a word found in fewer chunks. |d| is the chunk’s number of keyword terms, avgdl the average over all chunks; k₁ = 1.2 and b = 0.75.`,`下面用舞台上现在这一问（${quote(qText(S.q))}）来算。1. BM25：对问题 q 里的每个词 t，看它在这一块里出现几次（f<sub>t</sub>），再乘上 idf(t)，含有它的块越少，idf 越大。|d| 是这一块有几个${words}，avgdl 是所有块的平均；k₁ = 1.2，b = 0.75。`),BM25_TEX('BM25'),NORM_TEX);
  let b1='';
  if(t){const fq=c.terms.filter(x=>x===t).length, n=R.df.get(t), idf=Math.log(1+(R.N-n+0.5)/(n+0.5)), nd=1-BB+BB*c.terms.length/R.avg, part=idf*fq/(fq+K1*nd);
    b1=blk(tx(`For example, the word “${esc(t)}” in ${chunkName(R,j)}: it is in ${n} of the ${R.N} chunks, ${fq} time${fq===1?'':'s'} in this one, which has ${c.terms.length} words (average ${f(R.avg,1)}).`,`例如词“${esc(t)}”在${chunkName(R,j)}里：${R.N} 块里有 ${n} 块含有它，这一块里出现 ${fq} 次；这一块有 ${c.terms.length} 个词（平均 ${f(R.avg,1)} 个）。`),
      `\\mathrm{idf} = \\log\\left(1 + \\frac{${R.N} - ${n} + 0.5}{${n} + 0.5}\\right) = ${f(idf,3)}`,
      `n_d = 1 - 0.75 + 0.75 \\cdot \\frac{${c.terms.length}}{${f(R.avg,1)}} = ${f(nd,3)}`,
      `\\text{${tx(`what “${t}” adds`,`“${t}”加的分`)}} = ${f(idf,3)} \\cdot \\frac{${fq}}{${fq} + 1.2 \\times ${f(nd,3)}} = ${f(part,3)}`);}
  else b1=`<div class="ckr-mb"><p class="ckr-mb-lead">${tx('This question shares no word with the chunk that holds its answer, so that chunk’s keyword score is 0.','这一问和装着答案的那一块没有共同的词，所以那一块的关键词分数是 0。')}</p></div>`;
  const rkw=c?q.lists.kw.indexOf(j):-1, rv=c?q.lists.vec.indexOf(j):-1, inK=rkw>=0&&rkw<NCAND, inV=rv>=0&&rv<NCAND;
  const where=r=>r<0?tx('does not return it','不返回它'):tx(`ranks it ${r+1}`,`排第 ${r+1}`);
  const parts=[inK?`\\frac{1}{60 + ${rkw+1}}`:'',inV?`\\frac{1}{60 + ${rv+1}}`:''].filter(x=>x);
  const b2=blk(tx(`2. RRF of ${q.right.length?'the chunk that holds the answer':'the chunk keyword search ranks first'}: each search whose first ${NCAND} include the chunk adds ${MI('1/(c + r)')}, r being the chunk’s rank there, c = 60. Here keyword search ${where(rkw)}, vector search ${where(rv)}.`,`2. ${q.right.length?'装着答案的那一块':'关键词检索排第一的那一块'}的 RRF：哪种检索的前 ${NCAND} 块里有它，就加 ${MI('1/(c + r)')}，r 是它在那里的名次，c = 60。这里关键词检索${where(rkw)}，向量检索${where(rv)}。`),
    RRF_TEX('RRF'),`\\text{RRF} = ${parts.length?parts.join(' + '):'0'} = ${f(c?q.rrf[j]:0,4)}`);
  const b3=blk(tx(`3. Context tokens per question: k chunks, each up to L tokens, L being the longest chunk.`,`3. 每个问题的上下文 token：k 块，每块最多 L 个 token，L 是最长那一块的 token 数。`),`\\text{${tx('context','上下文')}} = k \\times L = ${S.k} \\times ${R.longest} = ${m.ctx}`);
  const b4=S.rr?blk(tx(`4. Time: search, plus rerank. Rerank reads the candidates the first search returned, up to ${NCAND} (here ${m.nc}, for the question with the most), each counted as L tokens at ${f(RR_MS,1)} ms per token: this lab’s assumption.`,`4. 时间：检索，加上 rerank。rerank 读第一轮返回的候选，最多 ${NCAND} 块（这里按候选最多的问题算，是 ${m.nc} 块），每块按 L 个 token、每个 token ${f(RR_MS,1)} 毫秒算：这是本 Lab 的假设。`),
      `\\text{${tx('time','时间')}} = \\text{${tx('search','检索')}} + n_{\\text{${tx('cand','候选')}}} \\times L \\times ${f(RR_MS,1)}`,`\\text{${tx('time','时间')}} = ${LAT[S.ret]} + ${m.nc} \\times ${R.longest} \\times ${f(RR_MS,1)} = ${f(m.lat,0)}\\,\\text{ms}`)
    :blk(tx(`4. Time: search only, without rerank (this lab’s assumption).`,`4. 时间：只有检索，不用 rerank（本 Lab 的假设）。`),`\\text{${tx('time','时间')}} = ${LAT[S.ret]}\\,\\text{ms}`);
  return b0+b1+b2+b3+b4;}

function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key;
    if(k==='speed') A.ui.setPressed(gEl,speed); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));
    if(k==='ov') $$('button[data-v]',gEl).forEach(b=>{b.disabled=S.size==='h';});});
  const lock=mode==='challenge', c=$('#ckr-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#ckr-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you choose everything in the task card: chunk size, overlap, search, rerank and k. Cutting by heading is left out.','挑战中所有选择都在任务卡里：块大小、重叠、检索方式、rerank 和 k。这次不能按标题切。');
  $$('#ckr-controls .lockable',app).forEach(e=>{e.hidden=lock;});}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(){
  const rail=$('#ckr-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step])&&!anim;
    if(box){if(!show&&anim&&box._h) {/* keep the last result while the cut plays */} else box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; box.hidden=false; if(again) A.ui.obsBump(box);}}}
    A.ui.setNext(rail,isNextDisabled());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#ckr-insight',rail); if(ins){const html=`<div class="eyebrow ckr-keep">${c>=0?esc(L(PRESETS[c].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const me=chMe(), done=chDone(), av=$('#ckr-ch-av',rail); if(!av) return;
    av.textContent=done?pc(me.rec):tx('shown after you submit','提交后才知道');
    $('#ckr-ch-cv',rail).textContent=A.fmt.int(me.ctx); $('#ckr-ch-lv',rail).textContent=msTxt(me.lat);
    $('#ckr-ch-c',rail).classList.toggle('over',me.ctx>CH_CTX); $('#ckr-ch-l',rail).classList.toggle('over',me.lat>CH_LAT+1e-9);
    $('#ckr-ch-ap i',rail).style.width=(done?Math.min(100,me.rec/CH_GOAL*100):0)+'%'; $('#ckr-ch-a',rail).classList.toggle('met',done&&me.rec>=CH_GOAL-1e-9); $('#ckr-ch-ap',rail).classList.toggle('met',done&&me.rec>=CH_GOAL-1e-9);
    const sub=$('#ckr-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    const res=$('#ckr-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#ckr-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  panelsShow(); drawQsel(); drawHead();
  if(!$('#ckr-pan-doc',app).hidden){maybeCut(); drawTransport(); drawStrip();}
  if(!$('#ckr-pan-lists',app).hidden) drawLists();
  if(!$('#ckr-pan-chart',app).hidden) drawChart();
  if(!$('#ckr-metrics',app).hidden) drawMetrics();
  const mb=$('#ckr-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const m=measure(LG(),S); A.announce(tx(`Recall@${S.k} ${pc(m.rec)}, ${m.ctx} context tokens.`,`recall@${S.k} ${pc(m.rec)}，上下文 ${m.ctx} 个 token。`));},400);}
function setState(p){Object.assign(S,p);
  if(!SIZES.includes(S.size)) S.size=64; if(!(S.ov>=0&&S.ov<=2)) S.ov=0; if(S.size==='h') S.ov=0;
  if(!RETS.includes(S.ret)) S.ret='vec'; S.rr=S.rr?1:0; if(!KS.includes(S.k)) S.k=3; if(!(S.q>=0&&S.q<NQ)) S.q=0;
  render();}

/* ---------- the cut animation (R3.9, R3.20): it plays only for a chunking of this page not yet watched; a watched one shows at once ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} const was=anim; anim=null; lastTs=0; return was;}
function maybeCut(){const k=cutKey(S); if(anim&&anim.key===k) return; if(anim) stopAnim(); if(seenCut.has(k)) return; startCut(0);}
function startCut(p0){stopAnim(); const k=cutKey(S);
  if(A.reduceMotion()){seenCut.add(k); return;}
  anim={key:k,p:p0||0,paused:false}; lastTs=0; raf=requestAnimationFrame(tick);}
function pauseCut(){if(!anim) return; anim.paused=true; if(raf){cancelAnimationFrame(raf); raf=0;} seenCut.add(anim.key);}
function resumeCut(){if(!anim) return; anim.paused=false; lastTs=0; raf=requestAnimationFrame(tick);}
const playing=()=>!!anim&&!anim.paused;
function tick(ts){raf=0; if(!anim||anim.paused||!app) return;
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts; const before=Math.floor(anim.p*1000);
  anim.p=Math.min(1,anim.p+dt/DUR[speed]);
  if(anim.p>=1){const k=anim.key; anim=null; seenCut.add(k); render(); return;}
  if(Math.floor(anim.p*1000)!==before){drawTransport(); drawStrip(); drawHead();}
  raf=requestAnimationFrame(tick);}

/* ---------- navigation ---------- */
function enterStep(i){
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  tried.clear(); tried.add(s.id==='search'?'q'+S.q:(s.id==='rerank'?'rr'+S.rr:JSON.stringify(chPick())));
  const st=$('#ckr-stage',app); if(st) st.scrollTop=0;
  renderRail();}
function revealStep(){A.ui.reveal($('#ckr-rail .step.current',app)||$('#ckr-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  const dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#ckr-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#ckr-insight',app)); else revealBox($('#ckr-ch-res',app));});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...DEFAULT,q:S.q};}
  if(m==='sandbox'&&was==='guided') S={...DEFAULT,q:S.q};
  A.shell.setMode(app,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){Object.assign(S,PRESETS[i].p); byLearner(render); revealObs();}
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=(s=S)=>[s.size,s.ov,s.ret,s.rr,s.k,s.q].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6) return null; const o={size:p[0]==='h'?'h':+p[0],ov:+p[1],ret:p[2],rr:+p[3],k:+p[4],q:+p[5]};
  if(!SIZES.includes(o.size)||![0,1,2].includes(o.ov)||!RETS.includes(o.ret)||![0,1].includes(o.rr)||!KS.includes(o.k)||!Number.isInteger(o.q)||o.q<0||o.q>=NQ) return null; return o;}
function act(a,btn){
  if(a==='play'){if(playing()) pauseCut(); else if(anim) resumeCut(); else startCut(0); render(); return;}
  if(a==='replay'){startCut(0); render(); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{Object.assign(S,DEFAULT); render();}); return;}
  if(a==='submit'){chSubmitted=true; render(); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#ckr-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(p){byLearner(()=>{Object.assign(S,p); noteTried(); render();}); revealObs();}
function noteTried(){const id=mode==='guided'?STEPS[step].id:''; tried.add(id==='search'?'q'+S.q:(id==='rerank'?'rr'+S.rr:JSON.stringify(chPick())));}

/* ---------- interview drill (seeded): reciprocal rank fusion of two rankings ---------- */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const rnd=mulberry(seed*7919+13), pick=()=>1+Math.floor(rnd()*12);
  const ch=['A','B','C'].map(n=>({n,kw:pick(),vec:pick()}));
  ch.forEach(c=>{c.s=1/(60+c.kw)+1/(60+c.vec);}); const best=ch.slice().sort((a,b)=>b.s-a.s||a.n.localeCompare(b.n))[0]; return {seed,ch,best};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#ckr-drill-q',app), a=$('#ckr-drill-a',app);
  const list=d.ch.map(c=>tx(`chunk ${c.n} ranks ${c.kw} in keyword search and ${c.vec} in vector search`,`块 ${c.n} 在关键词检索里排第 ${c.kw}，在向量检索里排第 ${c.vec}`)).join(tx('; ','；'));
  q.innerHTML=tx(`Hybrid search merges two full rankings with RRF and the constant 60. ${esc(list)}. Which chunk comes first, and what is its RRF score?`,`混合检索用常数 60 的 RRF 合并两个完整的排名。${esc(list)}。哪一块排第一？它的 RRF 分数是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=tx(`<p><b>Chunk ${d.best.n}, ${f(d.best.s,4)}</b></p>`,`<p><b>块 ${d.best.n}，${f(d.best.s,4)}</b></p>`)+d.ch.map(c=>M(`\\text{${c.n}}: \\frac{1}{60 + ${c.kw}} + \\frac{1}{60 + ${c.vec}} = ${f(c.s,4)}`)).join('')+
    tx(`<p>Only ranks count, not scores, so BM25 and cosine need no common scale.</p><p>The constant 60 keeps the shares close: rank 1 gets ${MI('1/61')}, rank 10 gets ${MI('1/70')}, about 13% less. So a chunk ranked well in both lists beats one ranked first in only one.</p>`,`<p>只用名次、不用分数，所以 BM25 和余弦不用换算到同一个尺度。</p><p>常数 60 让名次之间的分差很小：第 1 名得 ${MI('1/61')}，第 10 名得 ${MI('1/70')}，只少约 13%。所以两个列表里都排得靠前的块，会赢过只在一个列表里排第一的块。</p>`);}

/* ---------- jumps ---------- */
const CTLN={size:{en:'Chunk size',zh:'块大小'},ov:{en:'Overlap',zh:'重叠'},ret:{en:'First retrieval',zh:'先用哪种检索'},rr:{en:'Rerank',zh:'rerank'},k:{en:'Chunks kept, k',zh:'取前 k 块'}};
function jumpAct(k){
  if(k==='doc') return tx('Click to show the document strip.','点击查看文档条。');
  if(k==='chart') return tx('Click to show the chart.','点击查看曲线图。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k,src){
  if(!app) return;
  const pan=id=>{const el=$('#ckr-pan-'+id,app); if(el&&el.hidden){el.hidden=false; render();} A.tip.hide(); if(el){el.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el.querySelector('svg')||el);}};
  if(k==='doc'){pan('doc'); return;}
  if(k==='chart'){pan('chart'); return;}
  if(k==='controls'){const c=$('#ckr-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(!CTLN[k]) return;
  const el=$('#ckr-ic-'+k,app)||$('#ckr-ctl-'+k,app); if(el) A.ui.jumpToEl(el);}
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    const st=$('#ckr-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ckr-rail .step.current',app)||$('#ckr-rail .railcard',app)||$('#ckr-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, rc=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=rc.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='prog'){const p=+e.target.value/100, key=cutKey(S); stopAnim(); seenCut.add(key); if(p<1) anim={key,p,paused:true}; render();}});
  el.addEventListener('change',e=>{if(e.target.id==='ckr-qsel'){S.q=+e.target.value; byLearner(render);}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#ckr-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#ckr-rail');
      if(k==='speed'){speed=d.v; drawTransport(); syncControls(); return;}
      const v=k==='ret'?d.v:(k==='size'&&d.v==='h'?'h':+d.v);
      if(mode==='challenge'&&chSubmitted&&!chReveal) chSubmitted=false;
      byLearner(()=>{S[k]=v; if(S.size==='h') S.ov=0; noteTried(); setState({});});
      if(inRail) revealObs(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#ckr-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ckr-rail',app)); return;}
    if(d.act) act(d.act,b);});}
A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, what was tried in the part, the cuts watched, the
     challenge result */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,tried:[...tried],seenCut:[...new Set([...seenCut,...(anim&&anim.p>0?[anim.key]:[])])].slice(-100),   // a cut under way counts as watched, as stopAnim() counts it
   
    drillSeed,drillOpen,speed,passed:mode==='challenge'&&chSubmitted&&chMe().ok}),
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(token(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b;
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.includes(o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    tried.clear(); if(Array.isArray(o.tried)) o.tried.forEach(k=>{if(typeof k==='string'&&k.length<200&&/^(q\d+|rr[01]|\{[^<>]*\})$/.test(k)) tried.add(k);}); if(!tried.size) noteTried();
    seenCut.clear(); if(Array.isArray(o.seenCut)) o.seenCut.forEach(k=>{if(typeof k==='string'&&k.length<80&&/^(en|zh)\|[^<>]*$/.test(k)) seenCut.add(k);});
    drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen); if(Object.keys(DUR).includes(o.speed)) speed=o.speed;},
  mount(el,tok){
    stopAnim(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#ckr-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ckr-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    renderRail();},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const dm=$('#ckr-math',app), open=dm&&dm.open; stopAnim(); this.mount(root,null); if(open) $('#ckr-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#ckr-pan-doc:not([hidden])',app)||$('#ckr-pan-chart:not([hidden])',app)||$('#ckr-pan-lists',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#ckr-stage',app),
    render({status,vis,read}){if(!app) return; const m=measure(LG(),S), q=setup(LG(),S.size,S.ov).Q[S.q];
      vis.innerHTML=`<div class="ckr-dk"><i style="width:${(m.rec*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const me=chMe(), done=chDone(); status.className='status'+(done?(me.ok?' ok':' bad'):''); status.textContent=done?(me.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):tx('Not submitted','未提交');
        read.textContent=done?tx(`Recall@${S.k} ${pc(me.rec)}, ${me.ctx} tokens, ${f(me.lat,0)} ms`,`recall@${S.k} ${pc(me.rec)}，${me.ctx} 个 token，${f(me.lat,0)} 毫秒`):tx(`${me.ctx} context tokens, ${f(me.lat,0)} ms per question`,`每个问题 ${me.ctx} 个上下文 token、${f(me.lat,0)} 毫秒`); return;}
      status.className='status'+(q.right.length?'':' bad'); status.textContent=q.right.length?tx('Answer whole','答案完整'):tx('Answer cut','答案被切开');
      read.textContent=tx(`Recall@${S.k} ${pc(m.rec)} with ${m.ctx} context tokens per question`,`recall@${S.k} ${pc(m.rec)}，每个问题 ${m.ctx} 个上下文 token`);}}
});
/* read-only hook for tests/chunk-rerank_numbers.py and the home tile */
window.__ckr={state:()=>({mode,step,...S,pred,chSubmitted,chReveal,lang:LG(),speed}),animState:()=>anim?{p:anim.p,key:anim.key,paused:!!anim.paused}:null,seen:()=>[...seenCut],
  data:()=>({DOCS,QS,LEX}),consts:{K1,BB,RRF_K,NCAND,RR_MS,SIZES,OVF,RETS,KS,LAT,CH_GOAL,CH_CTX,CH_LAT,CH_SIZES,DIM,NQ},
  setup:(lang,size,ov)=>{const R=setup(lang,size,ov); return {N:R.N,avg:R.avg,longest:R.longest,splits:R.splits,C:R.C.map(c=>({d:c.d,a:c.a,b:c.b,no:c.no,n:c.n})),
    Q:R.Q.map(q=>({kw:q.kw,vec:q.vec,rrf:q.rrf,cov:q.cov,lists:q.lists,rr:q.rr,nc:q.nc,right:q.right,part:q.part,rk:q.rk,sa:q.sa,sb:q.sb}))};},
  measure:(lang,s)=>{const m=measure(lang,s); return {hits:m.hits,rec:m.rec,ctx:m.ctx,lat:m.lat,rerankMs:m.rerankMs,nc:m.nc,splits:m.splits};},
  chAll:lang=>chAll(lang),docLen:lang=>build(lang).map(d=>d.toks.length),drill:makeDrill};
})();
