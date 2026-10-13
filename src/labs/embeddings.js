/* Embeddings lab: how does an embedding know two sentences mean the same? A live small encoder (a 16-dimensional table of word
   vectors, averaged and normalised) is trained in the browser with the in-batch-negatives loss (InfoNCE, as sentence-transformers'
   MultipleNegativesRankingLoss) and Adam, on the help center of a fictional smart-home company (示例). Seeded and deterministic;
   tests/embeddings_numbers.py retrains it in Python. Also registers the home-tile teaser's engine hook (window.__emb).
   Names (batch 10 brief S6): embedding（向量）, 相似度 (cosine), 正例 / 负例 / batch 内负例, 召回率 recall@k (here: is the held-out
   question's own passage among the k most similar passages). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='embeddings';

/* ---------- engine: pure functions (ENGINE-START) ---------- */
/* the help center (示例, written for this lab): [topic, short title, passage, two training questions, one held-out question], each [en, zh] */
const DATA=[
 ["t", ["Change the target temperature", "调目标温度"], ["To change the target temperature, turn the dial on the thermostat or drag the slider in the app.", "想改目标温度，转动恒温器上的旋钮，或者在 App 里拖动滑块。"], [["How do I make the room warmer?", "怎么让房间暖和一点？"], ["Set the heat a bit higher", "把暖气开大一点"]], ["How do I make my room a bit warmer?", "房间想再暖和一点怎么办？"]],
 ["t", ["Heating does not start", "供暖没有启动"], ["If the heating does not start, check that the thermostat is in Heat mode and that the boiler has power.", "如果供暖没有启动，检查恒温器是不是在制热模式，锅炉有没有通电。"], [["The house is still cold", "家里还是很冷"], ["Why is the radiator not getting hot?", "暖气片为什么不热？"]], ["Why is my house still so cold?", "为什么家里一直这么冷？"]],
 ["t", ["Night schedules", "夜间定时计划"], ["Schedules let the thermostat lower the temperature at night and raise it before you wake up.", "定时计划可以让恒温器在夜里调低温度，在你起床前再调高。"], [["Can it turn the heat down while I sleep?", "睡觉的时候能把暖气关小吗？"], ["Warm the house before I get up in the morning", "早上起来之前先把家里烧暖"]], ["Turn the heat down while I sleep", "我睡觉时把暖气关小"]],
 ["t", ["Low thermostat battery", "恒温器电池电量低"], ["The thermostat shows a battery icon when its two AA batteries are low; replace them within a week.", "恒温器的两节 AA 电池快没电时，屏幕上会出现电池图标，一周内换掉。"], [["A little battery symbol is on the screen", "屏幕上有个小电池标志"], ["When should I change the batteries?", "电池什么时候该换？"]], ["What does the little battery symbol mean?", "小电池标志是什么意思？"]],
 ["l", ["Add a fingerprint", "添加指纹"], ["To add a fingerprint, open the lock settings in the app, choose Add fingerprint and touch the sensor five times.", "添加指纹：在 App 里打开门锁设置，选“添加指纹”，把手指在传感器上按五次。"], [["How do I register my finger?", "怎么录入我的手指？"], ["Let my kid open the door with a finger", "让孩子用手指开门"]], ["Register a finger for my kid", "给孩子录入手指"]],
 ["l", ["Keypad stays dark", "键盘不亮"], ["If the keypad stays dark, the lock batteries are empty; hold a 9V battery to the contacts under the handle to open it.", "键盘一直不亮，说明门锁电池没电了；把 9V 电池贴在把手下方的触点上，就能开门。"], [["I am locked out and the keypad does not light up", "我被锁在门外，键盘不亮"], ["The door lock is dead and I cannot get in", "门锁没反应，我进不了门"]], ["I cannot get in and the keypad is dead", "键盘没反应，我进不了门"]],
 ["l", ["Auto-lock", "自动上锁"], ["Auto-lock locks the door 30 seconds after it closes; you can change the delay or turn it off in the app.", "自动上锁会在门关上 30 秒后锁门；延迟时间可以在 App 里改，也可以关掉。"], [["Does the door lock itself?", "门会自己锁上吗？"], ["It locks too fast when I carry bags", "我拎着东西进门，它锁得太快"]], ["Can the door lock itself when I leave?", "我出门后门能自己锁上吗？"]],
 ["l", ["Guest codes", "访客密码"], ["Guest codes are temporary PINs that work only during the hours you set, for example for a cleaner.", "访客密码是临时密码，只在你设定的时段内有效，比如给保洁阿姨用。"], [["Give the cleaner a code that expires", "给保洁一个会过期的密码"], ["Temporary access for a visitor", "给来访的客人临时开门"]], ["A code for a visitor that expires", "给客人一个会过期的密码"]],
 ["v", ["Returns to the dock", "回充电座"], ["The robot vacuum returns to its dock when the battery drops below 15% and resumes cleaning after charging.", "扫地机器人电量低于 15% 时会回到充电座，充好电后接着扫。"], [["Why did the robot stop halfway and go home?", "机器人为什么扫到一半就回家了？"], ["It goes back to charge in the middle of cleaning", "扫到一半它跑回去充电"]], ["The robot went home before it finished", "机器人没扫完就回家了"]],
 ["v", ["Clean the brush roll", "清理滚刷"], ["Clean the brush roll every week: hair wrapped around it reduces suction and can stop the brush.", "每周清理一次滚刷：缠在上面的头发会降低吸力，还可能让滚刷停转。"], [["Hair is stuck in the robot", "头发卡在机器人里了"], ["It does not pick up dirt anymore", "它吸不起灰尘了"]], ["The robot does not pick up hair", "机器人吸不起头发"]],
 ["v", ["No-go zones", "禁区"], ["No-go zones drawn on the map keep the robot out of rooms or areas, such as under a desk with cables.", "在地图上画禁区，机器人就不会进这些房间或区域，比如有电线的书桌下面。"], [["Stop the robot from going into the kids room", "别让机器人进儿童房"], ["Keep it away from the cables", "让它离电线远一点"]], ["Keep the robot out of the kids room", "让机器人别进儿童房"]],
 ["v", ["Falls down stairs", "从楼梯掉下去"], ["If the robot keeps falling down stairs, wipe the cliff sensors on its underside with a dry cloth.", "如果机器人总从楼梯上掉下去，用干布擦一擦它底部的防跌落传感器。"], [["My robot fell down the steps", "我的机器人从台阶上摔下去了"], ["It drives off the edge of the stairs", "它从楼梯边上冲下去"]], ["Why does it fall off the steps?", "它为什么会从台阶上摔下去？"]],
 ["a", ["Wi-Fi: 2.4 GHz only", "只连 2.4 GHz Wi-Fi"], ["The app only connects to 2.4 GHz Wi-Fi networks; 5 GHz networks are not supported during setup.", "App 只能连 2.4 GHz 的 Wi-Fi，设置时不支持 5 GHz 网络。"], [["My phone cannot find the device during pairing", "配对时手机找不到设备"], ["Setup fails on my home network", "在家里的网络上一直设置失败"]], ["Pairing fails on my home network", "在家里的网络上配对失败"]],
 ["a", ["Share with family", "和家人共享"], ["To share your home with family members, go to Members in the app and send them an invitation.", "想和家人共享，在 App 的“成员”里给他们发邀请。"], [["Can my wife control the devices too?", "我老婆也能控制这些设备吗？"], ["Give my family access", "让家里人也能用"]], ["Give my wife access to the devices", "让我老婆也能用这些设备"]],
 ["a", ["Notifications", "通知"], ["Notifications are sent when a door is unlocked, the vacuum gets stuck or a device goes offline.", "门被打开、扫地机卡住、设备掉线时，App 都会发通知。"], [["Will I get an alert when someone opens the door?", "有人开门时我会收到提醒吗？"], ["Tell me when the robot is stuck", "机器人卡住了告诉我"]], ["Get an alert when the robot is stuck", "机器人卡住时提醒我"]],
 ["a", ["Two-step verification", "两步验证"], ["Turn on two-step verification in Account Security so that a new phone needs a code sent by SMS.", "在“账号安全”里打开两步验证，新手机登录时要输入短信验证码。"], [["How do I protect my account?", "怎么保护我的账号？"], ["Someone logged in from another phone", "有人用别的手机登录了"]], ["Stop someone from logging in to my account", "别让别人登录我的账号"]],
 ["e", ["E-104: hub connection lost", "E-104：网关断开"], ["E-104 means the thermostat lost its connection to the hub; move the hub closer or restart it.", "E-104 表示恒温器和网关断开了；把网关挪近一点，或者重启网关。"], [["What is E-104?", "E-104 是什么？"], ["The thermostat says it is offline", "恒温器显示离线"]], ["Why is the thermostat offline?", "恒温器为什么离线了？"]],
 ["e", ["E-201: wheel blocked", "E-201：轮子卡住"], ["E-201 means a wheel of the vacuum is blocked; lift the robot and remove anything caught in the wheels.", "E-201 表示扫地机的轮子被卡住；把机器人拿起来，清掉缠在轮子里的东西。"], [["The robot shows E-201", "机器人显示 E-201"], ["A wheel is jammed", "有个轮子卡死了"]], ["What does E-201 mean on the robot?", "机器人上的 E-201 是什么意思？"]],
 ["e", ["E-310: lock motor jammed", "E-310：门锁电机卡住"], ["E-310 appears when the lock motor is jammed, often because the door is not fully closed.", "门锁电机卡住时会出现 E-310，常常是因为门没关严。"], [["The lock shows E-310", "门锁显示 E-310"], ["The bolt will not come out", "锁舌伸不出来"]], ["Lock error E-310 when I close the door", "我关门时门锁报 E-310"]],
 ["e", ["Red light: update failed", "红灯：升级失败"], ["A blinking red light on any device means a firmware update failed; keep it plugged in and retry from the app.", "任何设备的红灯闪烁，都表示固件升级失败；保持通电，在 App 里重试。"], [["The light keeps flashing red", "灯一直在闪红色"], ["The update did not finish", "升级没完成"]], ["Red light flashing after the update", "升级以后红灯一直闪"]],
 ["w", ["Two-year warranty", "保修两年"], ["All Lumo Home devices have a two-year warranty from the date of purchase; keep your receipt.", "小鹿家居的所有设备从购买之日起保修两年，请保留好发票。"], [["How long is the guarantee?", "保修多久？"], ["Is my device still covered after a year?", "用了一年还在保修期内吗？"]], ["How long am I covered?", "我的设备保多久？"]],
 ["w", ["What is not covered", "不保修的情况"], ["Water damage, drops and repairs by unauthorised shops are not covered by the warranty.", "进水、摔坏和非授权店维修，都不在保修范围内。"], [["I dropped my lock, will you fix it for free?", "门锁摔了，能免费修吗？"], ["The robot got wet", "机器人进水了"]], ["Will you fix it for free if it got wet?", "进水了能免费修吗？"]],
 ["w", ["Returns and refunds", "退货和退款"], ["To start a return, contact support within 30 days with your order number; refunds take 5 to 7 days.", "要退货，请在 30 天内带着订单号联系客服；退款需要 5 到 7 天。"], [["I want my money back", "我想把钱要回来"], ["Can I send it back?", "可以寄回去吗？"]], ["How do I get my money back?", "怎么把钱要回来？"]],
 ["w", ["Spare parts", "备用零件"], ["Replacement brushes, filters and batteries can be ordered in the app shop and are not covered by the warranty.", "替换用的刷子、滤网和电池可以在 App 商城里买，它们不在保修范围内。"], [["Where can I buy new filters?", "新滤网在哪里买？"], ["Spare parts for the robot", "机器人的备用零件"]], ["Buy spare filters for the robot", "给机器人买备用滤网"]]];
const TOPICS=['t','l','v','a','e','w'];
const SEED=20261009, MAXS=200, LR=0.05, AB1=0.9, AB2=0.999, AEPS=1e-8, NP=DATA.length;
const BS=[2,4,8,16], TAUS=[1,0.2,0.05,0.01], DS=[4,8,16,32], CH_STEPS=[10,25,50,100];
/* one run is one random start; a claim about a setting is checked on the average of NSEED starts (seeds SEED + 1000k) */
const NSEED=16, SEEDS=Array.from({length:NSEED},(_,k)=>SEED+1000*k);
const DEFAULT={B:8,tau:0.05,d:16,steps:0,q:0};
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* tokens: English words (letters, digits, and codes such as e-104 or 2.4); Chinese: every two adjacent characters (a character pair), a
   lone character by itself; ASCII inside Chinese as words */
const WRE=/[a-z0-9]+(?:[-.][a-z0-9]+)*/g, ZRE=/[一-鿿]+|[a-z0-9]+(?:[-.][a-z0-9]+)*/g;
function tokens(s,lang){
  s=String(s).toLowerCase(); if(lang==='en') return s.match(WRE)||[];
  const out=[]; for(const w of (s.match(ZRE)||[])){ if(/[一-鿿]/.test(w[0])){ if(w.length===1) out.push(w); else for(let i=0;i<w.length-1;i++) out.push(w.slice(i,i+2)); } else out.push(w);}
  return out;}
const LI=lang=>lang==='zh'?1:0;
const builtC={};
/* the vocabulary is every token of the passages and the training questions; a held-out question's unknown tokens are skipped */
function built(lang){
  if(builtC[lang]) return builtC[lang];
  const li=LI(lang), vocab=new Map(), add=s=>tokens(s,lang).map(t=>{if(!vocab.has(t)) vocab.set(t,vocab.size); return vocab.get(t);});
  const Pi=DATA.map(x=>add(x[2][li])), Ti=[]; DATA.forEach(x=>x[3].forEach(q=>Ti.push(add(q[li]))));
  const known=s=>tokens(s,lang).filter(t=>vocab.has(t)).map(t=>vocab.get(t));
  const Hi=DATA.map(x=>known(x[4][li]));
  return (builtC[lang]={V:vocab.size,vocab,Pi,Ti,Hi,known});}
function encode(E,d,ids,out,o){   // mean of the rows, then scaled to length 1; returns the length before scaling
  const n=ids.length; for(let k=0;k<d;k++) out[o+k]=0;
  for(const t of ids) for(let k=0;k<d;k++) out[o+k]+=E[t*d+k];
  let s=0; for(let k=0;k<d;k++){out[o+k]/=n; s+=out[o+k]*out[o+k];} const nr=Math.sqrt(s); for(let k=0;k<d;k++) out[o+k]/=nr; return nr;}
const dot=(a,ao,b,bo,d)=>{let s=0; for(let k=0;k<d;k++) s+=a[ao+k]*b[bo+k]; return s;};
/* one training run: MAXS steps of the in-batch-negatives loss with Adam. State t (0…MAXS) keeps the held-out questions' and the passages'
   unit vectors, recall@1 and @5, and the batch of step t+1 scored by the encoder as it is at t (its similarities, softmax and loss). */
function train(lang,B,tau,d,seed=SEED,maxs=MAXS,light=false){
  const b=built(lang), V=b.V, r=mulberry(seed), E=new Float64Array(V*d);
  for(let i=0;i<V*d;i++) E[i]=r()*2-1;
  const m=new Float64Array(V*d), v=new Float64Array(V*d), G=new Float64Array(V*d), rb=mulberry(seed+1);
  const T={lang,B,tau,d,P:[],H:[],r1:[],r5:[],rank:[],batch:[],S:[],Pr:[],loss:[]};
  const q=new Float64Array(B*d), p=new Float64Array(B*d), nq=new Float64Array(B), npn=new Float64Array(B);
  for(let s=0;s<=maxs;s++){
    /* evaluate state s: every held-out question against every passage */
    const Pv=new Float64Array(NP*d), Hv=new Float64Array(NP*d);
    for(let i=0;i<NP;i++){encode(E,d,b.Pi[i],Pv,i*d); encode(E,d,b.Hi[i],Hv,i*d);}
    const rk=[]; let c1=0, c5=0;
    for(let i=0;i<NP;i++){const own=dot(Hv,i*d,Pv,i*d,d); let k=0; for(let j=0;j<NP;j++) if(j!==i&&dot(Hv,i*d,Pv,j*d,d)>own) k++; rk.push(k); if(k<1) c1++; if(k<5) c5++;}
    if(!light){T.P.push(Pv); T.H.push(Hv); T.rank.push(rk);} T.r1.push(c1/NP); T.r5.push(c5/NP);
    /* the batch of step s+1: B different passages, each with one of its two training questions (no passage twice: no false negatives) */
    const idx=[]; for(let i=0;i<NP;i++) idx.push(i);
    for(let i=0;i<B;i++){const j=i+Math.floor(rb()*(NP-i)); const t=idx[i]; idx[i]=idx[j]; idx[j]=t;}
    const ps=idx.slice(0,B), qs=ps.map(pp=>2*pp+Math.floor(rb()*2));
    for(let i=0;i<B;i++){nq[i]=encode(E,d,b.Ti[qs[i]],q,i*d); npn[i]=encode(E,d,b.Pi[ps[i]],p,i*d);}
    const S=new Float64Array(B*B), Pr=new Float64Array(B*B); let loss=0;
    for(let i=0;i<B;i++){let mx=-Infinity; for(let j=0;j<B;j++){S[i*B+j]=dot(q,i*d,p,j*d,d); mx=Math.max(mx,S[i*B+j]/tau);}
      let z=0; for(let j=0;j<B;j++){Pr[i*B+j]=Math.exp(S[i*B+j]/tau-mx); z+=Pr[i*B+j];} for(let j=0;j<B;j++) Pr[i*B+j]/=z; loss-=Math.log(Pr[i*B+i]);}
    loss/=B; if(!light){T.batch.push({ps,qs}); T.S.push(S); T.Pr.push(Pr);} T.loss.push(loss);
    if(s===maxs) break;
    /* exact gradient: dL/dS = (softmax − 1 on the diagonal) / (τB); through the dot products, the normalisation and the average */
    G.fill(0);
    const dq=new Float64Array(d), dp=new Float64Array(d);
    for(let i=0;i<B;i++){dq.fill(0); for(let j=0;j<B;j++){const g=(Pr[i*B+j]-(i===j?1:0))/(tau*B); for(let k=0;k<d;k++) dq[k]+=g*p[j*d+k];}
      const c=dot(q,i*d,dq,0,d), ids=b.Ti[qs[i]]; for(const t of ids) for(let k=0;k<d;k++) G[t*d+k]+=(dq[k]-q[i*d+k]*c)/nq[i]/ids.length;}
    for(let j=0;j<B;j++){dp.fill(0); for(let i=0;i<B;i++){const g=(Pr[i*B+j]-(i===j?1:0))/(tau*B); for(let k=0;k<d;k++) dp[k]+=g*q[i*d+k];}
      const c=dot(p,j*d,dp,0,d), ids=b.Pi[ps[j]]; for(const t of ids) for(let k=0;k<d;k++) G[t*d+k]+=(dp[k]-p[j*d+k]*c)/npn[j]/ids.length;}
    const st=s+1, bc1=1-Math.pow(AB1,st), bc2=1-Math.pow(AB2,st);
    for(let i=0;i<V*d;i++){m[i]=AB1*m[i]+(1-AB1)*G[i]; v[i]=AB2*v[i]+(1-AB2)*G[i]*G[i]; E[i]-=LR*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+AEPS);}
  }
  T.E=E; if(!light) T.pca=pca(T); return T;}
/* the map: the two directions in which the vectors of the last step (passages and held-out questions) spread most (PCA, Jacobi), used
   for every step so that the points move smoothly */
function pca(T){
  const d=T.d, X=[], P=T.P[MAXS], H=T.H[MAXS]; for(let i=0;i<NP;i++){X.push(Array.from(P.slice(i*d,i*d+d)));} for(let i=0;i<NP;i++){X.push(Array.from(H.slice(i*d,i*d+d)));}
  const n=X.length, mu=new Array(d).fill(0); X.forEach(x=>x.forEach((v,k)=>{mu[k]+=v/n;}));
  const C=[]; for(let a=0;a<d;a++){C.push(new Array(d).fill(0)); for(let c=0;c<d;c++){let s=0; for(const x of X) s+=(x[a]-mu[a])*(x[c]-mu[c]); C[a][c]=s/n;}}
  const {val,vec}=jacobi(C), ord=val.map((v,i)=>i).sort((i,j)=>val[j]-val[i]), ax=[];
  for(const i of ord.slice(0,2)){const e=vec.map(row=>row[i]); let bi=0; for(let k=1;k<d;k++) if(Math.abs(e[k])>Math.abs(e[bi])) bi=k; if(e[bi]<0) for(let k=0;k<d;k++) e[k]=-e[k]; ax.push(e);}
  let R=0; const xy=[];
  for(let s=0;s<=MAXS;s++){const pts=[]; for(const M of [T.P[s],T.H[s]]) for(let i=0;i<NP;i++){const c=ax.map(e=>{let z=0; for(let k=0;k<d;k++) z+=(M[i*d+k]-mu[k])*e[k]; return z;}); pts.push(c); R=Math.max(R,Math.abs(c[0]),Math.abs(c[1]));} xy.push(pts);}
  return {mu,ax,xy,R,share:(val[ord[0]]+val[ord[1]])/val.reduce((a,b)=>a+b,0)};}
function jacobi(M){
  const n=M.length, a=M.map(r=>r.slice()), V=[]; for(let i=0;i<n;i++){V.push(new Array(n).fill(0)); V[i][i]=1;}
  for(let sweep=0;sweep<100;sweep++){let off=0; for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) off+=a[i][j]*a[i][j]; if(off<1e-30) break;
    for(let p=0;p<n;p++) for(let q=p+1;q<n;q++){if(Math.abs(a[p][q])<1e-300) continue;
      const th=(a[q][q]-a[p][p])/(2*a[p][q]), t=Math.sign(th||1)/(Math.abs(th)+Math.sqrt(th*th+1)), c=1/Math.sqrt(t*t+1), s=t*c;
      for(let k=0;k<n;k++){const akp=a[k][p], akq=a[k][q]; a[k][p]=c*akp-s*akq; a[k][q]=s*akp+c*akq;}
      for(let k=0;k<n;k++){const apk=a[p][k], aqk=a[q][k]; a[p][k]=c*apk-s*aqk; a[q][k]=s*apk+c*aqk;}
      for(let k=0;k<n;k++){const vkp=V[k][p], vkq=V[k][q]; V[k][p]=c*vkp-s*vkq; V[k][q]=s*vkp+c*vkq;}}}
  return {val:a.map((r,i)=>r[i]),vec:V};}
const runC=new Map();
function run(lang,B,tau,d){const k=[lang,B,tau,d].join('|'); let T=runC.get(k); if(!T){T=train(lang,B,tau,d); runC.set(k,T); if(runC.size>14) runC.delete(runC.keys().next().value);} return T;}
/* cosine of two sentences under the encoder of a run at step s (unknown tokens skipped) */
function cosOf(T,s1,s2){const b=built(T.lang), d=T.d, a=new Float64Array(d), c=new Float64Array(d), i1=b.known(s1), i2=b.known(s2);
  const E=T.E; encode(E,d,i1,a,0); encode(E,d,i2,c,0); return dot(a,0,c,0,d);}
/* keyword search for the mission: how many distinct words a question shares with each passage */
function shared(lang,i,j){const li=LI(lang), a=new Set(tokens(DATA[i][4][li],lang)), b=new Set(tokens(DATA[j][2][li],lang)); return [...a].filter(t=>b.has(t));}
function kwRank(lang,i){const sc=DATA.map((x,j)=>shared(lang,i,j).length); let k=0; for(let j=0;j<NP;j++) if(j!==i&&sc[j]>=sc[i]) k++; return {rank:k,sc};}
/* recall@1 averaged over the NSEED starts, at every step up to maxs: the count of questions found first, summed, over NSEED × 24.
   It is filled one start at a time, so the page can fill it in the background (warm, below) or finish it at once (avgCurve). */
const avgC=new Map(), akey=(lang,B,tau,d)=>[lang,B,tau,d].join('|');
function avgEntry(lang,B,tau,d,maxs){const k=akey(lang,B,tau,d); let c=avgC.get(k); if(!c||c.maxs<maxs){c={maxs,n:new Array(maxs+1).fill(0),k:0}; avgC.set(k,c);} return c;}
function avgOne(c,lang,B,tau,d){const T=train(lang,B,tau,d,SEEDS[c.k],c.maxs,true); for(let s=0;s<=c.maxs;s++) c.n[s]+=Math.round(T.r1[s]*NP); c.k++;}
function avgCurve(lang,B,tau,d,maxs){const c=avgEntry(lang,B,tau,d,maxs); while(c.k<NSEED) avgOne(c,lang,B,tau,d); return c.n.map(x=>x/(NSEED*NP));}
/* the average curve if all NSEED starts are done, else null */
function avgReady(lang,B,tau,d,maxs){const c=avgC.get(akey(lang,B,tau,d)); return c&&c.maxs>=maxs&&c.k>=NSEED?c.n.map(x=>x/(NSEED*NP)):null;}
const avgR1=(lang,B,steps,tau=0.05,d=16)=>avgCurve(lang,B,tau,d,Math.max(steps,CH_STEPS[CH_STEPS.length-1]))[steps];
const chPoint=(lang,B,steps)=>{const r=avgR1(lang,B,steps); return {B,steps,r1:r,ok:r>=CH_GOAL-1e-9};};
const CH_GOAL=0.55;
/* ENGINE-END */

/* ---------- words and helpers ---------- */
const TOPIC_N={t:{en:'Thermostat',zh:'恒温器'},l:{en:'Smart lock',zh:'智能门锁'},v:{en:'Robot vacuum',zh:'扫地机器人'},a:{en:'App and account',zh:'App 和账号'},e:{en:'Error codes',zh:'错误码'},w:{en:'Warranty and returns',zh:'保修和退货'}};
const TRAPS=[{k:'order',t:{en:'Word order',zh:'词序'},a:['E-104 appears before E-201','E-104 出现在 E-201 之前'],b:['E-201 appears before E-104','E-201 出现在 E-104 之前']},
             {k:'neg',t:{en:'A negation',zh:'否定'},a:['The robot is stuck','如果供暖启动'],b:['The robot is not stuck','如果供暖没有启动']}];
const PICKS=[0,9,17,22];        // part 2: four held-out questions to try
const f=(v,dg)=>A.fmt.num(v,dg), MINUS='−';
const fx=(v,dg)=>{const s=Math.abs(v).toFixed(dg); return (v<0&&+s!==0?MINUS:'')+s;};
function pc(p,dg){return f(p*100,dg||0)+'%';}
const LG=()=>A.lang()==='zh'?'zh':'en', li=()=>LI(LG());
const stepsTxt=n=>tx(n===1?'1 step':`${A.fmt.int(n)} steps`,`${A.fmt.int(n)} 步`);
const tauTxt=t=>String(t);
const DEF=(t,tip)=>A.ui.def(t,tip);
const J=(k,t)=>`<span class="linkbtn emb-j" data-jump="${k}">${t}</span>`;
const M=t=>`<div class="emb-math-line">${A.ui.math(t)}</div>`;
const MI=t=>A.ui.math(t);
const LL=id=>{const C=A.catalog, l=C&&C.LAB&&C.LAB[id];   // a lab that is not live is named in plain text (R6.39)
  if(l&&C.labStatus&&C.labStatus(l)!=='live') return esc(L(l.s||{en:id,zh:id}));
  return A.shell.labLink?A.shell.labLink(id):esc(id);};
const live=id=>{const c=A.catalog, x=c&&c.LAB&&c.LAB[id]; return !!(x&&x.st&&x.st!=='planned');};
const nameOr=(id,en,zh)=>live(id)?LL(id):tx(en,zh);
const qText=i=>DATA[i][4][li()], pText=i=>DATA[i][2][li()], pTitle=i=>DATA[i][1][li()];
const quote=s=>tx(`“${s}”`,`“${s}”`);
const D_EMB=()=>tx('An embedding is the vector a model makes from a piece of text: a list of numbers. Texts with similar meanings get vectors pointing in similar directions.','embedding 是模型把一段文字变成的一串数，也就是一个向量。意思相近的文字，向量的方向相近。');
const D_SIM=()=>tx('Cosine similarity measures how close two vectors point: with both scaled to length 1 it is their dot product. 1 means the same direction, 0 unrelated, −1 opposite.','余弦相似度量两个向量的方向有多近：两个向量都缩放成长度 1 以后，它就是两者的点积。1 是同一个方向，0 是不相关，−1 是相反。');
const D_TOK=()=>tx('Here a token is an English word (codes such as E-104 count as one word). Chinese is cut into pairs of adjacent characters: “暖和一点” gives 暖和, 和一, 一点.','这里的 token，英文是一个词（E-104 这类代码算一个词）；中文按相邻两个字切成字对：“暖和一点”切成 暖和、和一、一点。');
const D_POS=()=>tx('A positive is the text paired with a question in training: here the passage that answers it. A negative is a passage it should not be paired with. In-batch negatives are the other questions’ positives in the same batch, used as negatives at no extra cost.','正例是训练时和一个问题配对的文字，这里是能回答它的那段说明。负例是不该和它配对的段落。batch 内负例就是同一个 batch 里别的问题的正例，直接拿来当负例，不用另外找。');
const D_HELD=()=>tx('24 more questions, one per passage, that training never sees. Each rewords the training questions of its passage, so they check whether what the encoder learns from the training pairs carries over to new wording.','另外写的 24 个问题，每段一个，训练时不给模型看。它们是同一段训练问题的另一种问法，用来检查模型从训练配对学到的东西，能不能用到换了说法的问题上。');
const D_REC=()=>tx('Recall@k here: the share of the 24 held-out questions whose own passage is among the k passages most similar to them (out of 24).','这里的 recall@k：24 个新问题里，正确段落排在相似度前 k 名（共 24 段）的占多少。');
const D_TAU=()=>tx('The temperature τ divides every similarity before the softmax. Small τ makes the softmax sharp: the most similar passage takes almost all the probability. Large τ makes it flat. sentence-transformers calls 1/τ the scale; its default scale 20 is τ = 0.05.','温度 τ：softmax 之前，每个相似度都先除以它。τ 小，softmax 尖，最相近的那段几乎拿走全部概率；τ 大，softmax 平。sentence-transformers 把 1/τ 叫 scale，默认 scale 20，也就是 τ = 0.05。');
const D_ADAM=()=>tx('Adam is the optimizer used here: it keeps a running average of each parameter’s gradient and of its square, and uses their ratio to decide which way to move each parameter and how far; a step usually moves a parameter by no more than about the learning rate. Learning rate 0.05 is this lab’s choice.','Adam 是这里用的优化器：它为每个参数记着梯度的滑动平均和梯度平方的滑动平均，用两者的比值决定往哪挪、挪多少；每一步挪动的幅度通常不超过一个学习率。学习率 0.05 是这个 Lab 自己定的。');
const D_PCA=()=>tx('PCA finds the two directions in which the 48 vectors (24 passages, 24 held-out questions) differ most after the last training step; the map shows each vector’s position along them. The same two directions are used at every step, so the points move smoothly. A flat map cannot keep every distance of 16 dimensions.','PCA 找出训练最后一步时 48 个向量（24 段说明、24 个新问题）差别最大的两个方向，图上画的是每个向量在这两个方向上的位置。每一步都用这两个方向，所以点是连续移动的。16 维的距离，平面图不可能全部保留。');

/* ---------- state ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, saved=null;
let chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, userAct=false, annT=0, view='r1', trap=null;
const seenRun=new Set(), shownPrev={}, triedB=new Set(), triedT=new Set(), triedQ=new Set();
let runSpeed='mid', anim=null, raf=0, lastTs=0;
const RUN_SPEEDS={slow:200/6,mid:200/3,fast:200/1.5};   // training steps per second: 200 steps take 6 s, 3 s or 1.5 s (R3.9)
const SPEED_N={slow:{en:'Slow',zh:'慢'},mid:{en:'Medium',zh:'中'},fast:{en:'Fast',zh:'快'}};
const runKey=s=>[LG(),s.B,s.tau,s.d].join('|');
/* the 16-start averages are filled in the background, one start per slice of about 30 ms, so a click never waits for them;
   the page redraws when one is complete. A job is [lang, B, τ, d, steps]. */
const warmQ=[]; let warmT=0;
function warm(jobs){jobs.forEach(j=>{if(!avgReady(...j)&&!warmQ.some(w=>w.join()===j.join())) warmQ.push(j);}); if(!warmT&&warmQ.length) warmT=setTimeout(warmTick,40);}
function warmTick(){
  warmT=0; const t0=performance.now(); let fin=false;
  while(warmQ.length&&performance.now()-t0<30){const j=warmQ[0], c=avgEntry(...j); if(c.k>=NSEED){warmQ.shift(); fin=true; continue;} avgOne(c,j[0],j[1],j[2],j[3]);}
  if(fin&&app&&!anim) render();
  if(warmQ.length) warmT=setTimeout(warmTick,10);}
function warmStop(){clearTimeout(warmT); warmT=0; warmQ.length=0;}
const cur=()=>run(LG(),S.B,S.tau,S.d);
function compute(s){const T=run(LG(),s.B,s.tau,s.d), t=Math.max(0,Math.min(MAXS,Math.round(s.steps)));
  const B=s.B, Pr=T.Pr[t]; let pd=0; for(let i=0;i<B;i++) pd+=Pr[i*B+i]; pd/=B;
  return {T,t,r1:T.r1[t],r5:T.r5[t],loss:T.loss[t],pd,rank:T.rank[t][s.q]};}

/* ---------- guided parts ---------- */
const PRED=[{v:'both'},{v:'pos'},{v:'all'}];
const predTxt=v=>({both:tx('Raise each question’s similarity with its own passage, and lower it with the other passages of the batch','把每个问题和它自己那段的相似度往上推，和 batch 里其他段落的往下压'),
  pos:tx('Raise only the similarity with its own passage; leave the other cells alone','只把和它自己那段的相似度往上推，其他格子不管'),
  all:tx('Raise every similarity in the batch, since all of them are help-center texts about the same products','因为都是讲同一批产品的帮助说明，所以把 batch 里所有相似度都往上推')})[v];
function inRun(){return `<div class="emb-ib"><button type="button" class="btn sm" data-act="play" data-play>${playLabel()}</button></div>`;}
function inPick(){return A.ui.seg({id:'emb-ic-q',key:'q',label:tx('A held-out question','一个新问题'),options:PICKS.map(i=>[String(i),esc(qText(i))]),value:String(S.q),inline:true});}
function inB(){return A.ui.seg({id:'emb-ic-B',key:'B',label:tx('Pairs per batch, B','每个 batch 的配对数 B'),options:BS.map(b=>[String(b),String(b)]),value:String(S.B),inline:true});}
function inTau(){return A.ui.seg({id:'emb-ic-tau',key:'tau',label:tx('Temperature τ','温度 τ'),options:TAUS.map(t=>[String(t),tauTxt(t)]),value:String(S.tau),inline:true});}
const LOSS_TEX=nm=>`\\text{${nm}}\\,\\,\\ell_i = -\\log \\frac{e^{s_{ii}/\\tau}}{\\sum_{j=1}^{B} e^{s_{ij}/\\tau}}`;
const ENC_TEX=nm=>`\\text{${nm}}\\,\\,e(x) = \\frac{v}{\\lVert v \\rVert}`, AVG_TEX=`v = \\frac{1}{n}\\sum_{k=1}^{n} E[w_k]`;   // two lines: one does not fit the card at 1440 and 900 (R6.37)
const COS_TEX=nm=>`\\text{${nm}}\\,\\,\\cos(a, b) = a \\cdot b = \\sum_{k} a_k b_k`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Search a help center by meaning, not by shared words.',zh:'按意思搜帮助中心，而不是按共同的词。'},
  panels:['pair'],enter:()=>({...DEFAULT}),
  body:()=>{const k=kwRank(LG(),0), ahead=k.sc.filter((v,j)=>j!==0&&v>k.sc[0]).length;
   return tx(`<p>The help center of Lumo Home, a smart-home company made up for this lab, has 24 short passages: the thermostat, the smart lock, the robot vacuum, the app, error codes and the warranty. A customer asks a question; search has to find the passage that answers it.</p><p>A customer asks ${quote(qText(0))} The passage that answers it begins ${quote(pText(0).split(',')[0]+' …')} The two share no word, so a keyword search that ranks passages by shared words puts ${ahead} other passages first (${J('pair','in the chart')}).</p><p>An ${DEF('embedding',D_EMB())} model works differently. It turns every text into a list of numbers, a vector, and a trained model gives texts with similar meanings vectors that point in similar directions. Search then compares the question’s vector with every passage’s and returns the closest. In this lab your browser trains a small embedding model and you watch how it learns that. Next comes the vector index: finding the closest vectors fast among 1M or more.</p>`,
             `<p>小鹿家居是为这个 Lab 编的一家智能家居公司。它的帮助中心有 24 段简短的说明，讲恒温器、智能门锁、扫地机器人、App、错误码和保修。用户来问问题，搜索要找出能回答它的那一段。</p><p>用户问${quote(qText(0))}，能回答它的那一段是${quote(pText(0).split('，')[0]+'……')}。这里的关键词搜索把中文切成相邻的两个字（字对）来比，这两句话没有一个共同的字对。所以按共同字对数排序的关键词搜索，会先给出另外 ${ahead} 段（${J('pair','见图')}）。</p><p>${DEF('embedding',D_EMB())} 模型换一种办法：它把每段文字变成一串数，也就是一个向量。训练好的模型让意思相近的文字，向量的方向也相近。搜索时，拿问题的向量和每一段的向量比，最相近的那段就是答案。这个 Lab 在你的浏览器里训练一个小的 embedding 模型，看它是怎么学会的。之后是向量索引：在 1M 个以上的向量里快速找到最近的那几个。</p>`);}},
 {id:'encoder',title:{en:'A small encoder, untrained',zh:'训练前的小编码器'},lead:{en:'Average the words’ vectors; compare directions.',zh:'把词的向量取平均，再比较方向。'},
  panels:['map','metrics'],enter:()=>({...DEFAULT}),target:'q',expect:()=>triedQ.size>1,showMe:{q:9},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick another question to continue.',zh:'换一个问题再继续。'},
  body:()=>tx(`<p>This lab’s embedding model is a 16-dimensional small encoder. It has an ${DEF('embedding table',A.terms.embedding())} with one row of 16 numbers for every ${DEF('token',D_TOK())} (see ${LL('tokenization-cost')} for how text is cut into tokens). A text’s vector is the average of its tokens’ rows, scaled to length 1:</p>${M(ENC_TEX('text vector'))}${M(AVG_TEX)}<p>${MI('E[w_k]')} is the row of the k-th token, ${MI('n')} the number of tokens. How close two vectors are is their ${DEF('cosine similarity',D_SIM())}; for vectors of length 1 it is the dot product:</p>${M(COS_TEX('similarity'))}<p>Before training, the table holds random numbers, so the directions are random too.</p><p>The map also shows 24 ${DEF('held-out questions',D_HELD())}, one per passage. Training never uses them, but each asks the same thing as the training questions of its passage, in other words, and most of its words appear in those training questions. (The next part shows the training questions.) These questions give two numbers: recall@1 is the share of them whose correct passage ranks first, and recall@5 the share whose correct passage is in the top 5. Both measure whether what the small encoder learns from the training pairs carries over to a reworded question. Pick one and look at its closest passages in the map.</p>`,
              `<p>这个 Lab 的 embedding 模型是一个 16 维的小编码器。它有一张 ${DEF('embedding 表',A.terms.embedding())}，每个 ${DEF('token',D_TOK())} 一行（文字怎么切成 token，见 ${LL('tokenization-cost')}），每行 16 个数值。一段文字的向量，就是它所有 token 的那几行取平均，再缩放成长度 1：</p>${M(ENC_TEX('文字的向量'))}${M(AVG_TEX)}<p>${MI('E[w_k]')} 是第 k 个 token 的那一行，${MI('n')} 是 token 数。两个向量有多近，用${DEF('余弦相似度',D_SIM())}量；长度都是 1 时，它就是点积：</p>${M(COS_TEX('相似度'))}<p>训练前，表里都是随机数，向量的方向也是随机的。</p><p>图里还有 24 个${DEF('新问题',D_HELD())}，每段一个。训练时不会用到它们，每个新问题问的事，和这一段说明的训练问题一样，只是换了一种说法；新问题里的大部分字词，在这一段的训练问题里出现过。（训练问题下一部分会讲。）这些新问题用来算两个数：recall@1 是正确段落排在第一的问题占多少，recall@5 是正确段落排进前 5 的问题占多少。它们量的是小编码器从训练配对学到的联系，能不能用到换了说法的问题上。选一个新问题，看看图里离它最近的是哪几段。</p>`)+inPick(),
  obs:r=>obsEnc(r)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'A batch of 8 pairs: what should training do to the table?',zh:'一个 batch 有 8 对：训练该怎么改这张表？'},
  panels:['heat'],enter:()=>({...DEFAULT}),gate:()=>!!pred,skip:true,nextWhy:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案再继续，或点“跳过”。'},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(predTxt(o.v))}</button>`).join('')}</div>`;
   return tx(`<p>Training needs pairs. Each passage gets two questions it answers: 48 pairs in all, each question with its ${DEF('positive',D_POS())}. One training step takes a batch: 8 different passages, each with one of its questions.</p><p>${J('heat','The table in the chart')} is that batch: row i is question i, column j is passage j, and the darker a cell, the higher their cosine similarity. The true pairs sit on the diagonal. After training, each question’s own passage should be the closest of all. What should one training step do to the table?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
             `<p>训练要用成对的数据。每段说明配两个它能回答的问题，一共 48 对；和问题配对的那段说明叫它的${DEF('正例',D_POS())}。训练的每一步取一个 batch：8 段不同的说明，各配一个问题。</p><p>${J('heat','图里的表格')}就是这个 batch：第 i 行是第 i 个问题，第 j 列是第 j 段说明，格子越深，两者的余弦相似度越高。对角线上是真正配对的格子。训练好以后，每个问题自己那段应该比其他段都近。训练的一步，应该怎么改这张表？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'train',title:{en:'Train with in-batch negatives',zh:'用 batch 内负例训练'},lead:{en:'Pull each pair together, push the rest of the row apart.',zh:'把每一对拉近，把同一行其他的推开。'},
  panels:['heat','map','chart','metrics'],view:'r1',enter:()=>({...DEFAULT,steps:seenRun.has(runKey(DEFAULT))?MAXS:0}),target:'steps',expect:()=>S.steps>=50,showMe:{steps:MAXS},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Train at least 50 steps to continue.',zh:'至少训练 50 步再继续。'},
  body:()=>{const ok=pred==='both', l0=run(LG(),8,0.05,16).loss[0], v=pred?tx(`<p class="${ok?'okline':'badline'}"><b>${ok?'Right.':'Not quite.'}</b> ${ok?'':`You predicted “${esc(predTxt(pred))}”. `}A step raises each true pair and lowers the rest of its row.</p>`,`<p class="${ok?'okline':'badline'}"><b>${ok?'对了。':'不对。'}</b>${ok?'':`你预测的是“${esc(predTxt(pred))}”。`}训练的一步把每个真正的配对往上推，把同一行其余的往下压。</p>`):tx(`<p>The answer: a step raises each true pair and lowers the rest of its row.</p>`,`<p>答案：训练的一步把每个真正的配对往上推，把同一行其余的往下压。</p>`);
   return v+tx(`<p>Raising only the diagonal is not enough. If all vectors point the same way, the diagonal is all 1, but so is every other cell, and a question can no longer tell which passage is its own. So the other cells of each row must go down too. The other passages of a row are the other questions’ positives; used as negatives they are called ${DEF('in-batch negatives',D_POS())} and cost nothing extra. The loss of question i takes two steps: a softmax turns its row of similarities into probabilities (see ${LL('softmax-numerics')}), then the loss is minus the log of its own passage’s probability, the cross-entropy.</p>${M(LOSS_TEX('loss'))}<p>${MI('s_{ij}')} is the cosine of question i and passage j, and log is the natural logarithm. ${MI('\\tau')} = 0.05 is the ${DEF('temperature',D_TAU())}: every similarity is divided by it first, which makes the gaps 20 times larger (a later part shows what that changes). Lowering this loss raises ${MI('s_{ii}')} and lowers the rest of the row. Training like this is called contrastive learning: paired texts are pulled together and unpaired ones pushed apart. This loss is known as InfoNCE; sentence-transformers calls it MultipleNegativesRankingLoss.</p><p>Before training, the loss of the first batch is ${f(l0,2)}, even higher than ln 8 ≈ ${f(Math.log(8),2)}, the loss when every passage gets the same probability. Before training the cosines are random, and dividing by τ makes those random gaps 20 times larger too. So the softmax gives most of a row’s probability to one passage, and that passage is usually not the right one.</p><p>Each step averages the loss of the batch’s questions, then ${DEF('Adam',D_ADAM())} (see ${LL('optimizers')}) uses the gradient of that average loss to update the numbers in the embedding table. Press Train. The bar at the right of each table row is the probability the softmax gives that row’s correct passage; the line chart shows the recall@1 of the 24 held-out questions as training goes on. Watch the diagonal darken and the points in the map move.</p>`,
                 `<p>只把对角线往上推是不够的。所有向量都指向同一个方向，对角线就全是 1，可那时其他格子也全是 1，问题分不出哪段是自己的。所以同一行里的其他格子还要往下压。同一行里的其他段落，是别的问题的正例；拿来当负例用，叫 ${DEF('batch 内负例',D_POS())}，不用另外找。算问题 i 的 loss 分两步：先用 softmax 把这一行的相似度变成概率（见 ${LL('softmax-numerics')}），再取它自己那段概率的负对数，也就是 cross-entropy。</p>${M(LOSS_TEX('loss'))}<p>${MI('s_{ij}')} 是问题 i 和段落 j 的余弦，log 是自然对数。${MI('\\tau')} = 0.05 是${DEF('温度',D_TAU())}：每个相似度都先除以它，差距就放大 20 倍（后面会细讲）。把这个 loss 降下去，就会把 ${MI('s_{ii}')} 往上推，把这一行其余的往下压。这种训练方法叫对比学习（contrastive learning）：让配对的两段文字靠近，不配对的分开。这个 loss 叫 InfoNCE，sentence-transformers 里叫 MultipleNegativesRankingLoss。</p><p>训练前，第一个 batch 的 loss 是 ${f(l0,2)}，比每段概率一样时的 ln 8 ≈ ${f(Math.log(8),2)} 还高。原因是训练前的余弦是随机的，除以 τ 以后，这些随机的差距也放大了 20 倍。于是 softmax 把一行的大部分概率给了某一段，而这一段多半不是正确的那段。</p><p>每一步在 batch 上取平均，再用 ${DEF('Adam',D_ADAM())}（见 ${LL('optimizers')}）更新表里的数值。按“训练”。表格每行右边的条，是 softmax 给这一行正确段落的概率；曲线图画的是训练过程中 24 个新问题的 recall@1。看对角线怎么变深，图里的点怎么移动。</p>`)+inRun();},
  obs:r=>obsRun(r)},
 {id:'batch',title:{en:'Batch size',zh:'batch 大小'},lead:{en:'More pairs per batch, more negatives per question.',zh:'每个 batch 的配对越多，每个问题的负例越多。'},
  panels:['chart','heat','metrics'],view:'r1',enter:()=>({...DEFAULT,steps:MAXS}),target:'B',expect:()=>triedB.size>1,showMe:{B:16},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another batch size to continue.',zh:'换一个 batch 大小再继续。'},
  body:()=>tx(`<p>With B pairs per batch, each question is compared with B − 1 in-batch negatives in every step. With B = 2 it has only one, usually from another topic and easy to tell apart. But a bigger batch also uses more training pairs per step, so comparing by steps alone is unfair to small batches. So also compare the runs after they have used the same number of pairs.</p><p>Pick a batch size and watch ${J('chart','the recall curve')}. Every batch size you try leaves its curve in the chart, with its B written at the end of the line. A run also depends on its random start (the random table and the batches drawn), so the observation below averages 16 starts.</p>`,
              `<p>每个 batch 有 B 对时，每一步里每个问题要和 B − 1 个 batch 内负例比。B = 2 时只有 1 个，而且多半来自别的主题，很容易分开。但 batch 越大，每一步用掉的训练配对也越多，只按步数比，对小 batch 不公平。所以还要比一比：用掉同样多的配对以后，谁的 recall 高。</p><p>选一个 batch 大小，看 ${J('chart','recall 曲线')}。你试过的每个 batch 大小都会在图里留下一条线，线的末端写着它的 B。一次训练的结果还和它的随机种子有关（种子决定随机的表和抽到的 batch），所以下面的“观察到”取 16 个种子的平均。</p>`)+inB(),
  obs:()=>obsBatch()},
 {id:'temp',title:{en:'Temperature',zh:'温度'},lead:{en:'How sharp the softmax is decides which negatives get pushed.',zh:'softmax 有多尖，决定把哪些负例推开。'},
  panels:['chart','heat','metrics'],view:'loss',enter:()=>({...DEFAULT,steps:MAXS}),target:'tau',expect:()=>triedT.size>1,showMe:{tau:0.01},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Try another temperature to continue.',zh:'换一个温度再继续。'},
  body:()=>tx(`<p>A cosine lies between −1 and 1, so before the softmax the loss divides it by τ to spread the scores apart. With τ = 1 the scores differ by at most 2. In a batch of 16 the own passage then gets at most about a third of the probability. With τ = 0.05 every gap is 20 times larger: a slightly higher similarity already takes almost all of it.</p><p>The temperature also decides which negatives get pushed. For a wrong passage j, the gradient of row i’s loss is ${MI('\\partial \\ell_i / \\partial s_{ij} = p_{ij} / \\tau')}: the probability the softmax gives it, divided by τ. A small τ gives nearly all the probability, and so nearly all of the push, to the most similar wrong passage; a large τ spreads it over the whole row. Try a few and watch ${J('chart','the loss')}.</p>`,
              `<p>余弦只在 −1 到 1 之间，所以 softmax 之前先除以 τ，把分数拉开。τ = 1 时，除完以后的分数最多只差 2。batch 有 16 段时，正确那段最多也只拿到约三分之一的概率。τ = 0.05 时，每个差距放大 20 倍，相似度稍高一点，就能拿走几乎全部概率。</p><p>温度还决定推开哪些负例。对一个错段落 j，第 i 行的 loss 对它的梯度是 ${MI('\\partial \\ell_i / \\partial s_{ij} = p_{ij} / \\tau')}：softmax 给它的概率，再除以 τ。τ 小，几乎全部概率都给了最像的那个错段落，推开的幅度也几乎全落在它身上；τ 大，推开的幅度分给整行。换几个温度，看 ${J('chart','loss 曲线')}。</p>`)+inTau(),
  obs:()=>obsTemp()},
 {id:'limits',title:{en:'What an average cannot see',zh:'取平均看不到什么'},lead:{en:'Word order and negation; how real models do it.',zh:'词序和否定；真实的模型怎么做。'},
  panels:['map','metrics'],enter:()=>({...DEFAULT,steps:MAXS}),expect:()=>!!trap,showMe:{trap:'order'},label:{en:'Observed',zh:'观察到'},nextWhy:{en:'Pick a pair of sentences to continue.',zh:'选一对句子再继续。'},
  body:()=>tx(`<p>The small encoder only averages, so it cannot see word order; tokens that never appear in the training data it skips altogether. Pick a pair of sentences and compare them with the encoder trained with the initial setup (batch 8, τ = 0.05, 200 steps).</p>`,
              `<p>这个小编码器只取平均，所以看不到词的顺序；训练数据里没出现过的字对，它直接跳过。选一对句子，用训练好的编码器（batch 8、τ = 0.05、训练 200 步）比一比。</p>`)+leversTrap()+
   tx(`<p>Real embedding models first read the whole text with a Transformer, so every token’s vector carries what comes before and after it (see ${LL('mha')}), and word order and negation change the result; then they average the vectors or take one position’s. The same model encodes short questions and long passages, so some models ask you to mark which one a text is: the three English E5 v2 models (384, 768 and 1,024 dimensions) put <code>query: </code> before a question and <code>passage: </code> before a passage; BGE’s English v1.5 models suggest an instruction before a question, and also work without it.</p>`,
      `<p>真实的 embedding 模型先用 Transformer 读完整段文字，每个 token 的向量都带上了它前后的内容（见 ${LL('mha')}），所以词序和否定都会改变结果；然后再取平均，或取某一个位置的向量。同一个模型既要编码短问题，也要编码长段落，所以有的模型让你标明这段文字是哪一种：E5 v2 的三个英文模型（384、768、1,024 维）在问题前加 <code>query: </code>，在段落前加 <code>passage: </code>；BGE 的英文 v1.5 模型建议在问题前加一句检索指令，不加也能用。</p>`),
  obs:()=>obsTrap()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Pairs, in-batch negatives, temperature; learned links, not only shared words.',zh:'配对、batch 内负例、温度；学到的联系，不只是共同的词。'},
  panels:['map','chart'],view:'r1',enter:()=>({...DEFAULT,steps:MAXS}),
  body:()=>{const T=run(LG(),8,0.05,16), T32=run(LG(),8,0.05,32), k0=Math.round(T.r1[0]*NP), k16=Math.round(T.r1[MAXS]*NP), k32=Math.round(T32.r1[MAXS]*NP), rk=T.rank[MAXS][0], p2=sameTopic(2), p16=sameTopic(16);
   return tx(`<ul><li>An embedding is a text’s vector; search turns the question into a vector too and returns the passages with the highest cosine similarity. ${quote(qText(0))} shares no word with the passage about the dial, and the trained small encoder ${rk===0?'ranks that passage first':`ranks it ${rk+1}`}.</li><li>Training uses pairs and in-batch negatives: each question is pulled toward its own passage and pushed away from the other passages of its batch. The hardest passages to tell apart are those of the same topic, so it matters whether a batch holds any: with B = 2 a question’s one negative is from its own topic ${pc(p2)} of the time; with B = 16 a batch holds at least one passage of its topic ${pc(p16)} of the time.</li><li>The temperature decides how the push is shared: a small τ pushes almost only the most similar wrong passage, a large τ the whole row.</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “High similarity means the same words.” Before training the small encoder does rely on shared words only: ${k0===0?'not one':`only ${k0}`} of the 24 held-out questions ${k0<=1?'finds its':'find their'} own passage first. After training in this lab, ${quote(qText(0))} shares no word with the passage about the dial and ${rk===0?'still ranks it first':`still ranks it ${rk+1}`}: the training pairs taught the encoder which words point to which passage.</p><p>✗ “More dimensions are always better.” More dimensions can hold more distinctions, but every dimension adds 4 bytes to every stored vector and one multiply to every comparison, so storage and search time grow in step with it. In this lab, going from 16 to 32 dimensions finds ${k32-k16} more of the 24 held-out questions. Real models can gain even less: E5-large-v2 stores vectors almost three times as long as E5-small-v2 and ${DEF('scores only a little higher',BEIR_TIP())} on the retrieval benchmark BEIR.</p></div>${A.shell.toScenarios()}`,
             `<ul><li>embedding 是一段文字的向量。搜索时把问题也变成向量，返回余弦相似度最高的段落。${quote(qText(0))}和讲旋钮的那段没有一个共同的字对，训练后的小编码器${rk===0?'把那段排在第一':`把那段排在第 ${rk+1}`}。</li><li>训练用成对的数据和 batch 内负例：把每个问题往它自己那段拉，从同一个 batch 的其他段落推开。最难分开的是同主题的段落，所以 batch 里有没有同主题的段落很重要。B = 2 时，一个问题唯一的负例只有 ${pc(p2)} 的机会和它同主题；B = 16 时，${pc(p16)} 的 batch 里至少有一段和它同主题。</li><li>温度决定推开的幅度怎么分：τ 小，几乎只推最像的那个错段落；τ 大，推整行。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “相似度高就是用词一样。”训练前的小编码器确实只靠共同的字词：24 个新问题里${k0===0?'没有一个':`只有 ${k0} 个`}第一个找到自己的段落。在这个 Lab 里训练以后，${quote(qText(0))}和讲旋钮的那段没有一个共同的字对，却${rk===0?'排在第一':`排在第 ${rk+1}`}：训练配对教会了它哪些词指向哪一段。</p><p>✗ “维度越高越好。”维度高能多存一些区别，但每多一维，每个存下的向量多 4 字节，每次比较多一次乘法，存储和搜索的时间都跟着维数一起涨。在这个 Lab 里，从 16 维加到 32 维，24 个新问题里多找对 ${k32-k16} 个。真实模型里可能涨得更少：E5-large-v2 存下的向量差不多是 E5-small-v2 的三倍长，在检索基准 BEIR 上却${DEF('只高一点',BEIR_TIP())}。</p></div>${A.shell.toScenarios()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
/* the chance that the B − 1 other passages of a batch include one of the 3 other passages of the question's topic (23 to draw from) */
function sameTopic(B){let none=1; for(let k=0;k<B-1;k++) none*=(20-k)/(23-k); return 1-none;}
/* τ = 1: the most the own passage can get in a batch of B, when its cosine is 1 and every other one −1 */
const capP=B=>Math.exp(2)/(Math.exp(2)+B-1);
const CAP_TIP=()=>tx(`With τ = 1 the own passage’s score beats each other one by at most 2, so it gets at most e² / (e² + 7) = ${pc(capP(8))} in a batch of 8, and the loss stays above −ln ${f(capP(8),2)} = ${f(-Math.log(capP(8)),2)}.`,`τ = 1 时，正确那段的分数最多比其他每段高 2，所以 batch 8 时它最多拿到 e² / (e² + 7) = ${pc(capP(8))}，loss 不会低于 −ln ${f(capP(8),2)} = ${f(-Math.log(capP(8)),2)}。`);
function leversTrap(){return A.ui.levers({aria:tx('Pairs of sentences','几对句子'),value:trap,options:TRAPS.map(x=>({v:x.k,title:L(x.t),desc:`${quote(x.a[li()])}${tx(' and ','和')}${quote(x.b[li()])}`,result:trap===x.k?f(cosOf(run(LG(),8,0.05,16),x.a[li()],x.b[li()]),2):''}))});}
function topTxt(T,t,i,k){const d=T.d, H=T.H[t], P=T.P[t], sc=DATA.map((x,j)=>({j,s:dot(H,i*d,P,j*d,d)})); sc.sort((a,b)=>b.s-a.s||a.j-b.j); return sc.slice(0,k);}
function obsEnc(r){
  const T=r.T, i=S.q, rk=T.rank[0][i], top=topTxt(T,0,i,1)[0];
  return tx(`<p><b>Before training, the right passage comes first for ${Math.round(T.r1[0]*NP)} of the 24 held-out questions: recall@1 ${pc(T.r1[0])}, about what picking at random gives.</b> For ${quote(qText(i))} the closest passage is ${quote(pTitle(top.j))}; the right one, ${quote(pTitle(i))}, ranks ${rk+1}.</p><p>${DEF('Recall@5',D_REC())} is already ${pc(T.r5[0])}, above the 21% of picking 5 at random. The reason: two texts that share some words also share those rows, so even when the rows are random, their averages come out a little similar.</p>`,
            `<p><b>训练前，24 个${DEF('新问题',D_HELD())}里${Math.round(T.r1[0]*NP)?`只有 ${Math.round(T.r1[0]*NP)} 个`:'没有一个'}的正确段落排第一：recall@1 是 ${pc(T.r1[0])}，和随便挑一段差不多。</b>${quote(qText(i))}最近的是${quote(pTitle(top.j))}；正确的${quote(pTitle(i))}排第 ${rk+1}。</p><p>${DEF('recall@5',D_REC())} 已经有 ${pc(T.r5[0])}，比随便挑 5 段的 21% 高。原因是两段文字有共同的字对，就会用到表里相同的几行；即使这些行是随机数，取平均以后两段的向量也会有点相近。</p>`);}
/* the training question of passage i that holds the most of held-out question i's known tokens */
function trainLink(i){const lang=LG(), V=built(lang).vocab, hk=[...new Set(tokens(DATA[i][4][LI(lang)],lang).filter(t=>V.has(t)))];
  let best=null; DATA[i][3].forEach(x=>{const s=x[LI(lang)], tk=new Set(tokens(s,lang)), n=hk.filter(t=>tk.has(t)).length; if(!best||n>best.n) best={s,n};}); best.m=hk.length; return best;}
const ownCos=(T,s)=>{const d=T.d; let z=0; for(let i=0;i<NP;i++) z+=dot(T.H[s],i*d,T.P[s],i*d,d); return z/NP;};
function obsRun(r){
  if(r.t===0) return '';
  const T=r.T, t=r.t, q=S.q, rk=T.rank[t][q], done=t>=MAXS, up=r.r1>T.r1[0], c0=ownCos(T,0), c1=ownCos(T,t), nsh=shared(LG(),q,q).length, k=trainLink(q);
  const why=k.n?tx(` Its passage is paired with the training question ${quote(k.s)}. Of the words of the held-out question, ${k.m} appear in the training data and so have their own row in the embedding table; this training question holds ${k.n===k.m?'all of them':`${k.n} of them`}. Training pulled the table rows of those words toward the direction of that passage.`,`原因是这段和训练问题${quote(k.s)}配对，新问题切出的字对里，有 ${k.m} 个在训练数据里出现过、在 embedding 表里有自己的一行，这 ${k.m} 个${k.n===k.m?'全都':`里有 ${k.n} 个`}在这句训练问题里。训练把这些字对在 embedding 表里的那几行，拉向了这段的方向。`):'';
  return tx(`<p><b>After ${stepsTxt(t)}, recall@1 on the 24 held-out questions ${up?'rose':'went'} from ${pc(T.r1[0])} to ${pc(r.r1)}.</b> The diagonal of the table darkened, and the average cosine between a held-out question and its own passage ${c1>c0?'rose':'went'} from ${fx(c0,2)} to ${fx(c1,2)}.</p><p>${quote(qText(q))} ${rk===0?'now finds its passage first':`now ranks its passage ${rk+1}`}${rk===0&&nsh===0?', although the two share no word':''}.${why}${done?'':' Keep training to 200 steps to see the whole run.'}</p>`,
            `<p><b>训练 ${t} 步后，24 个新问题的 recall@1 从 ${pc(T.r1[0])} ${up?'升到':'变到'} ${pc(r.r1)}。</b>表格的对角线变深了；新问题和它自己那段的平均余弦，从 ${fx(c0,2)} ${c1>c0?'升到':'变到'} ${fx(c1,2)}。</p><p>${quote(qText(q))}${rk===0?'现在第一个就找到了它的段落':`现在把它的段落排在第 ${rk+1}`}${rk===0&&nsh===0?'，尽管两者没有一个共同的字对':''}。${why}${done?'':'继续训练到 200 步，看完整次运行。'}</p>`);}
const AVG_WAIT=()=>tx('<p class="muted small">Averaging 16 random starts …</p>','<p class="muted small">正在算 16 个随机种子的平均……</p>');
function obsBatch(){
  const lg=LG(), tau=S.tau, d=S.d, need=[[4,200],[8,100],[16,100],[2,100]].map(([B,m])=>[lg,B,tau,d,m]);
  if(!need.every(j=>avgReady(...j))){warm(need); return tx(`<p><b>A bigger batch raises recall@1 faster per step.</b></p>`,`<p><b>batch 越大，每一步 recall@1 涨得越快。</b></p>`)+AVG_WAIT();}
  const av=(B,s)=>avgReady(lg,B,tau,d,B===4?200:100)[s];
  const lst=[...triedB].sort((a,b)=>a-b).map(B=>tx(`<li>B = ${B}: ${pc(av(B,25))}</li>`,`<li>B = ${B}：${pc(av(B,25))}</li>`)).join('');
  return tx(`<p><b>A bigger batch raises recall@1 faster per step.</b> One run depends on its random start, so these are averages over 16 starts, after 25 steps:</p><ul class="emb-ul">${lst}</ul><p>Counted per pair seen, the batch size makes little difference in this lab: after the same 800 pairs, B = 4, 8 and 16 all average between ${pc(Math.min(av(4,200),av(8,100),av(16,50)))} and ${pc(Math.max(av(4,200),av(8,100),av(16,50)))}. In real training a bigger batch does help: sentence-transformers’ documentation links much higher batch sizes with better performance, since each question meets more negatives per step. This lab, with 24 passages, is too small to show it.</p>`,
            `<p><b>batch 越大，每一步 recall@1 涨得越快。</b>一次训练的结果和它的随机种子有关，所以下面是 16 个不同种子的平均，训练 25 步后：</p><ul class="emb-ul">${lst}</ul><p>按用掉的配对数比，在这个 Lab 里 batch 大小差别不大：同样用掉 800 对，B = 4、8、16 的平均都在 ${pc(Math.min(av(4,200),av(8,100),av(16,50)))} 到 ${pc(Math.max(av(4,200),av(8,100),av(16,50)))} 之间。真实训练里大 batch 确实更好：sentence-transformers 的文档说，batch 大得多时效果也更好，因为每个问题每一步能比过更多负例。这个 Lab 只有 24 段说明，太小，看不出来。</p>`);}
function lossZero(T){for(let s=0;s<=MAXS;s++) if(T.loss[s]<0.01) return s; return -1;}
function obsTemp(){
  const lg=LG(), ts=[...triedT].sort((a,b)=>b-a), rows=ts.map(t=>{const T=run(lg,S.B,t,S.d), z=lossZero(T); return {t,T,z,lz:T.loss[MAXS],r:T.r1[MAXS],rz:z>=0?T.r1[z]:0};});
  const lst=rows.map(x=>tx(`<li>τ = ${x.t}: ${x.z>=0?`the loss first falls under 0.01 at step ${x.z}, when recall@1 is ${pc(x.rz)}`:`the loss is still ${f(x.lz,2)} at step 200`}; recall@1 after 200 steps ${pc(x.r)}.</li>`,`<li>τ = ${x.t}：${x.z>=0?`第 ${x.z} 步 loss 第一次降到 0.01 以下，这时 recall@1 是 ${pc(x.rz)}`:`第 200 步 loss 还有 ${f(x.lz,2)}`}；200 步后 recall@1 是 ${pc(x.r)}。</li>`)).join('');
  const need=ts.map(t=>[lg,S.B,t,S.d,MAXS]), ok=need.every(j=>avgReady(...j)); if(!ok) warm(need);
  const avg=ok?tx(`<p>Training once with each of 16 random seeds and averaging, recall@1 after 200 steps is ${ts.map(t=>`${pc(avgReady(lg,S.B,t,S.d,MAXS)[MAXS])} with τ = ${t}`).join(', ')}.`,`<p>换 16 个随机种子各训练一次，取平均，200 步后的 recall@1：${ts.map(t=>`τ = ${t} 是 ${pc(avgReady(lg,S.B,t,S.d,MAXS)[MAXS])}`).join('，')}。`):'';
  return tx(`<p>The run drawn on the stage (seed 20261009):</p><ul class="emb-ul">${lst}</ul><p>With a small τ, after only a few steps each training question gives almost all of its probability to its own passage, so most batches have a loss near 0. A loss near 0 means a small gradient, so each update gets smaller and recall@1 rises more and more slowly. Now and then a batch that is not yet separated comes up and the loss jumps: the spikes in the chart. With τ = 1 the loss cannot get near 0, so every step keeps pulling pairs together.</p>${ok?avg:AVG_WAIT()+'<p>'} In this lab, with only 48 training pairs, everything is separated within a few dozen steps, so a small τ learns less. Real training data is far bigger (sentence-transformers’ all-* models used more than 1B pairs): there are always batches not yet separated, and the loss does not reach 0 that fast. sentence-transformers’ default is τ = 0.05; its documentation calls scales from 10 to 100 common, that is τ from 0.1 to 0.01.</p>`,
            `<p>舞台上画的这一次（种子 20261009）：</p><ul class="emb-ul">${lst}</ul><p>τ 小时，没训练几步，每个训练问题就把几乎全部概率给了自己那段，大多数 batch 的 loss 接近 0。loss 接近 0，梯度也就很小，所以每一步的更新越来越小，recall@1 涨得越来越慢。偶尔抽到还没分开的 batch，loss 会跳起来，就是图里的尖峰。τ = 1 时，loss 降不到 0 附近，每一步都还在把配对往一起拉。</p>${ok?avg:AVG_WAIT()+'<p>'}在这个 Lab 里只有 48 对训练数据，几十步就全分开了，所以 τ 小反而学得少。真实的训练数据大得多（sentence-transformers 的 all-* 模型用了 1B 对以上），总有还没分开的 batch，loss 不会这么快降到 0。sentence-transformers 默认 τ = 0.05，它的文档说 scale 常在 10 到 100 之间，也就是 τ 在 0.1 到 0.01 之间。</p>`);}
function obsTrap(){
  if(!trap) return ''; const x=TRAPS.find(z=>z.k===trap), T=run(LG(),8,0.05,16), c=cosOf(T,x.a[li()],x.b[li()]);
  if(x.k==='order') return tx(`<p><b>Similarity ${f(c,2)}: the two sentences get exactly the same vector.</b> They contain the same tokens, so their averages are equal, although one says E-104 came first and the other E-201.</p>`,`<p><b>相似度 ${f(c,2)}：两句话的向量完全一样。</b>它们切出来的 token 一模一样，平均也就一样；可一句说 E-104 先出现，另一句说 E-201 先出现。</p>`);
  const lg=LG(), V=built(lg).vocab, ta=tokens(x.a[li()],lg), tb=tokens(x.b[li()],lg), neg=tb.filter(t=>!ta.includes(t)&&V.has(t)), nb=tb.filter(t=>V.has(t)).length, skip=[...new Set(ta.concat(tb).filter(t=>!V.has(t)))];
  const qs=a=>a.map(t=>`“${t}”`), en=a=>a.length<2?a.join(''):a.slice(0,-1).join(', ')+' and '+a[a.length-1];
  return tx(`<p><b>Similarity ${f(c,2)}: the two sentences mean the opposite, yet their vectors stay close.</b> The negation adds ${en(qs(neg))}, ${neg.length===1?'a token':'tokens'} the small encoder knows, but the second sentence’s vector is the average of ${nb} table rows and ${neg.length===1?'it is only 1':`they are only ${neg.length}`} of them, and an average cannot tell what ${neg.length===1?'it negates':'they negate'}.${skip.length?` ${skip.every(t=>!tb.includes(t))?'In the first sentence, ':''}${en(qs(skip))} never ${skip.length===1?'appears':'appear'} in the training data and ${skip.length===1?'is':'are'} skipped.`:''}</p>`,
            `<p><b>相似度 ${f(c,2)}：两句话意思相反，向量却仍然很近。</b>否定多出来的${neg.length>1?`字对${qs(neg).join('')}都`:`${qs(neg).join('')}`}在词表里，但第二句的向量是 embedding 表里 ${nb} 行的平均，${neg.length>1?'它们':'它'}只占其中 ${neg.length} 行；取平均以后，分不出否定的是什么。${skip.length?`${skip.every(t=>!tb.includes(t))?'第一句里的':''}字对${qs(skip).join('')}，训练数据里没出现过，小编码器把${skip.length>1?'它们':'它'}直接跳过了。`:''}</p>`);}

/* ---------- scenarios ---------- */
const PRESETS=[
 {t:{en:'Initial setup: batch 8, τ = 0.05, 16 dimensions',zh:'初始设置：batch 8，τ = 0.05，16 维'},d:{en:'sentence-transformers’ default temperature, a batch of 8 pairs.',zh:'sentence-transformers 的默认温度，每个 batch 8 对。'},p:{B:8,tau:0.05,d:16}},
 {t:{en:'A tiny batch: 2 pairs',zh:'batch 很小：只有 2 对'},d:{en:'Each question has one in-batch negative per step.',zh:'每一步每个问题只有 1 个 batch 内负例。'},p:{B:2,tau:0.05,d:16}},
 {t:{en:'A big batch: 16 pairs',zh:'batch 大：16 对'},d:{en:'Each question has 15 in-batch negatives per step.',zh:'每一步每个问题有 15 个 batch 内负例。'},p:{B:16,tau:0.05,d:16}},
 {t:{en:'A flat softmax: τ = 1',zh:'softmax 很平：τ = 1'},d:{en:'Similarities are not spread apart before the softmax.',zh:'softmax 之前，相似度没有被拉开。'},p:{B:8,tau:1,d:16}},
 {t:{en:'A sharp softmax: τ = 0.01',zh:'softmax 很尖：τ = 0.01'},d:{en:'Almost all the push goes to the most similar wrong passage.',zh:'推开的幅度几乎全落在最像的那个错段落上。'},p:{B:8,tau:0.01,d:16}},
 {t:{en:'Only 4 dimensions',zh:'只有 4 维'},d:{en:'Every text has to fit into 4 numbers.',zh:'每段文字只能用 4 个数值表示。'},p:{B:8,tau:0.05,d:4}}];
const presetNow=()=>PRESETS.findIndex(p=>p.p.B===S.B&&p.p.tau===S.tau&&p.p.d===S.d);
function sandboxHTML(){const c=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how recall@1 on the 24 held-out questions changes as the small encoder trains.','选一个场景，看看小编码器训练时，24 个新问题的 recall@1 怎么变。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===c}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="emb-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function insightHTML(r){
  if(r.t===0) return tx('<p><b>Step 0: nothing trained yet.</b> Press Train in the chart, or drag the training steps.</p>','<p><b>第 0 步：还没训练。</b>按图里的“训练”，或拖动训练步数。</p>');
  const T=r.T, base=run(LG(),8,0.05,16), i=presetNow();
  let h=tx(`<p><b>After ${stepsTxt(r.t)}, recall@1 on the 24 held-out questions went from ${pc(T.r1[0])} to ${pc(r.r1)}.</b></p>`,`<p><b>训练 ${r.t} 步后，24 个新问题的 recall@1 从 ${pc(T.r1[0])} 变到 ${pc(r.r1)}。</b></p>`);
  if(i<0) return h+tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  if(r.t<MAXS) return h;
  const k=['init','b2','b16','t1','t001','d4'][i], z=lossZero(T);
  if(k==='init') h+=tx(`<p>Each question compares itself with 7 in-batch negatives per step. The loss of a batch first falls under 0.01 at step ${z}, when recall@1 is ${pc(T.r1[z])}; after that the updates get smaller, and recall@1 only rises slowly.</p>`,`<p>每一步每个问题和 7 个 batch 内负例比。第 ${z} 步 batch 的 loss 就降到 0.01 以下，这时 recall@1 是 ${pc(T.r1[z])}；之后更新越来越小，recall@1 只是慢慢上升。</p>`);
  else if(k==='b2') h+=tx(`<p>With one negative per question, a step mostly separates two passages of different topics, which is easy. It ends below a batch of 8, which reaches ${pc(base.r1[MAXS])}.</p>`,`<p>每个问题只有 1 个负例，一步里多半只是把两个不同主题的段落分开，这很容易。最后比 batch 8 时的 ${pc(base.r1[MAXS])} 低。</p>`);
  else if(k==='b16') h+=tx(`<p>With 15 negatives per question, a step usually includes passages of the same topic, the hard ones. After 25 steps it already reaches ${pc(T.r1[25])}; a batch of 8 has ${pc(base.r1[25])} then.</p>`,`<p>每个问题有 15 个负例，一步里通常就有同一主题的段落，也就是难分的那些。25 步时就到 ${pc(T.r1[25])}；batch 8 那时是 ${pc(base.r1[25])}。</p>`);
  else if(k==='t1') h+=tx(`<p>The loss is still ${f(T.loss[MAXS],2)} at the end. With τ = 1 the similarities are not magnified, and cosines lie between −1 and 1, so in a batch of 8 the own passage can get ${DEF('only about half',CAP_TIP())} of the probability and the loss cannot get near 0. A loss above 0 means every step still pulls the pairs together, and in this lab it ends higher than with τ = 0.05, which reaches ${pc(base.r1[MAXS])}.</p>`,`<p>最后 loss 还有 ${f(T.loss[MAXS],2)}。τ = 1 时相似度没有被放大，而余弦只在 −1 到 1 之间，所以 batch 8 时正确那段${DEF('最多只拿到约一半',CAP_TIP())}概率，loss 降不到 0 附近。loss 不到 0，每一步就都还在把配对往一起拉；在这个 Lab 里，它最后比 τ = 0.05 时的 ${pc(base.r1[MAXS])} 还高。</p>`);
  else if(k==='t001') h+=tx(`<p>The loss first falls under 0.01 at step ${z}, when recall@1 is ${pc(T.r1[z])}. After that the updates soon get small and recall@1 rises only slowly.</p>`,`<p>第 ${z} 步 loss 就降到 0.01 以下，这时 recall@1 是 ${pc(T.r1[z])}。之后每一步的更新很快变小，recall@1 涨得很慢。</p>`);
  else h+=tx(`<p>Four numbers cannot keep 24 passages apart: passages of different topics share directions. It ends below 16 dimensions, which reach ${pc(base.r1[MAXS])}.</p>`,`<p>4 个数值分不开 24 段说明：不同主题的段落挤在相同的方向上。最后比 16 维时的 ${pc(base.r1[MAXS])} 低。</p>`);
  return h;}

/* ---------- challenge: the fewest steps to an average recall@1 of 55% over the 16 starts ---------- */
let chN=25;
const G55=()=>pc(CH_GOAL);
function challengeHTML(){
  const sub=`<div id="emb-ch-sub"${chSubmitted?' hidden':''}><button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, the small encoder trains with your batch size for your number of steps from 16 random starts; the stage shows the first, the verdict uses the average.','提交后，小编码器按你选的 batch 大小和步数，用 16 个随机种子各训练一次；舞台上画第一次，判定用平均值。')}</p></div>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Get the 16-dimensional small encoder to an average recall@1 of at least ${G55()} on the 24 held-out questions, in as few training steps as possible`,`让 16 维的小编码器在 24 个新问题上的平均 recall@1 至少到 ${G55()}，训练步数越少越好`)}</h3>`+
   `<p>${tx('The same help center and the same 48 training pairs; τ = 0.05, sentence-transformers’ default. One run depends on its random start (the random table and the batches drawn), so the verdict averages 16 runs from 16 different starts. Choose the batch size and the number of steps:','还是这个帮助中心、这 48 对训练数据；τ = 0.05，sentence-transformers 的默认值。一次训练的结果和它的随机种子有关（种子决定随机的表和抽到的 batch），所以按 16 个不同种子的 16 次训练取平均来判。选 batch 大小和训练步数：')}</p>`+
   A.ui.seg({id:'emb-ic-chB',key:'B',label:tx('Pairs per batch, B','每个 batch 的配对数 B'),options:BS.map(b=>[String(b),String(b)]),value:String(S.B),inline:true})+
   A.ui.seg({id:'emb-ic-chN',key:'chN',label:tx('Training steps','训练步数'),options:CH_STEPS.map(n=>[String(n),String(n)]),value:String(chN),inline:true})+
   `<div class="goal mono" id="emb-ch-a"><span>${tx('Average recall@1 over 16 starts','16 个种子的平均 recall@1')}</span><span><b id="emb-ch-av"></b>${tx(` (goal: at least ${G55()})`,`（目标：至少 ${G55()}）`)}</span></div><div class="progress" id="emb-ch-ap" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="emb-ch-g"><span>${tx('Training steps','训练步数')}</span><span><b id="emb-ch-gv"></b></span></div>`+
   `<p class="small muted">${tx(`The ${G55()} goal is set for this challenge in this lab.`,`${G55()} 是这个 Lab 为这道题定的目标。`)}</p>`+
   sub+`<div id="emb-ch-res" class="emb-ch-res" aria-live="polite"></div></div>`;}
const chDone=()=>mode==='challenge'&&chSubmitted&&S.steps>=chN&&!anim;
function chAll(){const out=[]; for(const B of BS) for(const n of CH_STEPS) out.push(chPoint(LG(),B,n)); return out;}
const chWarm=()=>warm(BS.map(B=>[LG(),B,0.05,16,CH_STEPS[CH_STEPS.length-1]]));
function chResultHTML(){
  if(!chDone()) return '';
  const pts=chAll(), me=pts.find(p=>p.B===S.B&&p.steps===chN), okp=pts.filter(p=>p.ok), minS=Math.min(...okp.map(p=>p.steps)), best=okp.filter(p=>p.steps===minS), bb=best[0];
  let h;
  if(me.ok&&me.steps===minS) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: an average recall@1 of ${pc(me.r1)} after ${stepsTxt(me.steps)}.</b> And no choice reaches ${G55()} in fewer steps.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：训练 ${me.steps} 步，平均 recall@1 是 ${pc(me.r1)}。</b>而且没有哪种选择能用更少的步数到 ${G55()}。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: an average recall@1 of ${pc(me.r1)}.</b> But B = ${bb.B} reaches ${G55()} in ${stepsTxt(bb.steps)}, instead of ${me.steps}.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：平均 recall@1 是 ${pc(me.r1)}。</b>不过 B = ${bb.B} 只要 ${bb.steps} 步就能到 ${G55()}，而不是 ${me.steps} 步。</p>`);
  else {h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: an average recall@1 of ${pc(me.r1)}, under ${G55()}.</b></p><p class="small">Hint: in one step, which batch size trains on the most pairs and gives every question the most wrong passages to beat?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：平均 recall@1 是 ${pc(me.r1)}，不到 ${G55()}。</b></p><p class="small">提示：一步里，哪种 batch 大小训练的配对最多，每个问题要比过的错段落也最多？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal) return h;
  const isInit=p=>p.B===2&&p.steps===25;
  const rows=pts.map(p=>({label:tx(`B = ${p.B}, ${stepsTxt(p.steps)}`,`B = ${p.B}，${p.steps} 步`),value:p.r1*100,text:pc(p.r1),ok:p.ok,me:p===me,best:p.ok&&p.steps===minS,
    tags:(isInit(p)?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  h+=A.ui.cmpChart({brief:true,title:tx(`Average recall@1 over 16 starts, τ = 0.05 (goal: at least ${G55()})`,`16 个种子的平均 recall@1，τ = 0.05（目标：至少 ${G55()}）`),
    aria:tx('Average recall@1 by batch size and steps.','按 batch 大小和步数的平均 recall@1。'),rows,goal:CH_GOAL*100,goalLabel:tx(`goal: ${G55()}`,`目标 ${G55()}`),max:100,
    hiddenText:n=>tx(`The other ${n} choices stay under ${G55()} and are not shown.`,`其余 ${n} 种选择到不了 ${G55()}，图里没有列出。`),
    okText:tx(`reaches ${G55()}`,`达到 ${G55()}`),noText:tx('misses it','没达到'),goalText:tx(`goal: ${G55()}`,`目标：${G55()}`)});
  const s0=CH_STEPS[0], pp=16*s0, per=[4,8,16].map(B=>avgR1(LG(),B,pp/B)), lo=Math.min(...per), hi=Math.max(...per);
  h+=tx(`<p class="small">Why: a bigger batch trains on more pairs in each step and gives every question more negatives, so per step it learns fastest. After ${s0} steps, only B = 16 reaches ${G55()}. Each of its steps does more work, though. Counted per pair seen, B = 4, 8 and 16 are about equal in this lab: after the same ${pp} pairs they all average between ${pc(lo)} and ${pc(hi)}.</p>`,
        `<p class="small">原因：batch 越大，一步里训练的配对越多，每个问题的负例也越多，所以按步数算学得最快。训练 ${s0} 步，只有 B = 16 能到 ${G55()}。不过它的每一步工作量也更大。按用掉的配对数比，在这个 Lab 里 B = 4、8、16 差不多：同样用掉 ${pp} 对，平均都在 ${pc(lo)} 到 ${pc(hi)} 之间。</p>`);
  h+=tx(`<p class="small muted">Note: recall is counted on 24 questions, so one question is about 4 percentage points, and one run depends on its random start; that is why the verdict averages 16 starts.</p>`,
        `<p class="small muted">注意：recall 只在 24 个问题上数，一个问题就是约 4 个百分点；一次训练的结果又和随机种子有关。所以判定用 16 个种子的平均。</p>`);
  return h;}

/* ---------- capability layer ---------- */
const OUTCOME=()=>tx('explain how pairs and in-batch negatives train an embedding model, read a batch’s similarity table, and say what batch size and temperature change.','讲清成对的数据和 batch 内负例怎么训练 embedding 模型，看懂一个 batch 的相似度表，并说出 batch 大小和温度各改变了什么。');
const EXITS=()=>({
 explain:[
  {q:tx('After training, a held-out question that shares no word with its passage still finds it first. Why?','训练后，一个和正确段落没有共同字对的新问题，也能第一个找到它。为什么？'),
   o:[tx('Its words appear in training questions paired with that passage, so training pulled their vectors toward that passage’s vector.','它的词在训练问题里出现过，而那些问题和这段配对，所以训练把这些词的向量拉向了这段的向量。'),
      tx('The encoder looks the words up in a dictionary of synonyms built from the passages.','编码器从段落里建了一本同义词词典，查词就知道。'),
      tx('Cosine similarity ignores the words and only compares the lengths of the two texts.','余弦相似度不看词，只比较两段文字的长度。')],a:0,
   n:[null,tx('Nothing in the encoder is a dictionary: it only has a table of vectors, moved by the loss on the training pairs.','编码器里没有词典，只有一张向量表，靠训练配对上的 loss 来改。'),tx('Every vector is scaled to length 1, so the length plays no part; the direction does.','每个向量都缩放成长度 1，长度不起作用，起作用的是方向。')],
   why:tx('The loss moves each word’s row toward the passages its questions are paired with; a new question made of those words lands near the same passage.','loss 把每个词的那一行，拉向用到它的问题所配对的段落；用这些词组成的新问题，就落在同一段附近。')},
  {q:tx('Why do in-batch negatives cost nothing extra?','为什么 batch 内负例不用另外花成本？'),
   o:[tx('The other passages of the batch are encoded anyway, for their own pairs; each question reuses their vectors as negatives.','batch 里的其他段落本来就要为它们自己那一对编码；每个问题直接拿这些向量当负例。'),
      tx('Negatives are random vectors, which cost nothing to make.','负例是随机向量，生成它们不花成本。'),
      tx('The loss leaves the negatives out of the gradient, so they cost no training.','loss 算梯度时不管负例，所以负例不占训练。')],a:0,
   n:[null,tx('The negatives are real passages of the help center, encoded by the same small encoder.','负例是帮助中心里真实的段落，由同一个小编码器编码。'),tx('The gradient does reach the negatives: it lowers each row’s wrong similarities.','梯度确实会传到负例：它把每一行错段落的相似度往下压。')],
   why:tx('A batch of B pairs needs B question vectors and B passage vectors anyway. The B × B table of similarities reuses them, so every question gets B − 1 negatives without encoding anything more.','一个 B 对的 batch，本来就要算 B 个问题向量和 B 个段落向量。B × B 的相似度表直接复用它们，所以每个问题多了 B − 1 个负例，却不用多编码任何文字。')}],
 transfer:[
  {q:tx('A help center stores 1M passages as float32 vectors of 384 dimensions: 1.54 GB. It switches to a model with 1,024 dimensions. About how much storage do the vectors need now?','一个帮助中心把 1M 段存成 384 维的 float32 向量，占 1.54 GB。换成 1,024 维的模型后，向量大约要占多少？'),
   o:[tx('About 4.1 GB','约 4.1 GB'),tx('About 1.5 GB, the same','约 1.5 GB，不变'),tx('About 0.6 GB','约 0.6 GB')],a:0,
   why:tx('Storage grows with the dimension: 1,000,000 × 1,024 × 4 bytes = 4.10 GB, 2.7 times as much.','存储量随维数涨：1,000,000 × 1,024 × 4 字节 = 4.10 GB，是原来的 2.7 倍。')},
  {q:tx('Under the loss card stands a baseline: when every passage gets the same probability, the loss is ln 8 ≈ 2.08 for B = 8. A team raises the batch to 64. What is that baseline now?','loss 卡片下面写着每段概率一样时的值：B = 8 时是 ln 8 ≈ 2.08。一个团队把 batch 加到 64。每段概率一样时，loss 是多少？'),
   o:[tx('About 4.16, since each question now has 63 negatives','约 4.16，因为每个问题现在有 63 个负例'),tx('About 2.08, the same as before','约 2.08，和原来一样'),tx('About 1.04, half as much','约 1.04，减半')],a:0,
   why:tx(`With 64 passages of equal probability the own passage gets 1/64, and the loss is ${MI('-\\ln \\frac{1}{64} = \\ln 64 \\approx 4.16')}: a higher number with a bigger batch does not mean a worse model. It is a baseline, not the untrained loss: with τ = 0.05 a random table gives most of a row’s probability to one wrong passage, so the untrained loss is higher, ${f(run(LG(),8,0.05,16).loss[0],2)} for B = 8 in this lab.`,`64 段概率一样时，正确那段得到 1/64，loss 是 ${MI('-\\ln \\frac{1}{64} = \\ln 64 \\approx 4.16')}：batch 大了数字变大，不代表模型更差。这是基准线，不是训练前的 loss：τ = 0.05 时，随机的表会把一行的大部分概率给某个错段落，所以训练前的 loss 比它高，这个 Lab 里 B = 8 时是 ${f(run(LG(),8,0.05,16).loss[0],2)}。`)}],
 practice:[{id:'rag-search-ml',d:tx('Its first part: the contrastive loss, the temperature, positives, negatives and batch size, the ones this lab trains with.','其中第一部分：对比学习的 loss、温度、正例负例和 batch 大小，就是这个 Lab 里训练用的那些。')},
           {id:'numpy-1nn',d:tx('Nearest neighbour as one matrix multiply, the search step after the vectors are made.','用一次矩阵乘法做最近邻，也就是有了向量以后的搜索那一步。')}]});

/* ---------- header, popovers ---------- */
const DESCS=()=>({
  guided:tx('Eight short parts. See why shared words are not enough, predict what training does to a batch, train the small encoder, then change the batch size and the temperature.','一共 8 个部分。先看为什么只看共同词不够，预测训练对一个 batch 做了什么，训练小编码器，再改 batch 大小和温度。'),
  sandbox:tx('Pick a batch size, a temperature or a dimension and watch the small encoder train.','选一种 batch 大小、温度或维数，看小编码器怎么训练。'),
  challenge:tx('Choose the batch size and the training steps so that recall@1, averaged over 16 random starts, reaches 55% in the fewest steps.','只选 batch 大小和训练步数，让 16 个随机种子的平均 recall@1 到 55%，训练步数越少越好。')});
const STPIN='ff69486771998c07008ebd3c7e05e827f03df46f', UNPIN='31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd', FEPIN='fd1a2bdf69488ffebe0327999d4400d8c8058a0b';
const GH=(repo,pin,path)=>`https://github.com/${repo}/blob/${pin}/${path}`;
const SRC=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
const U_MNRL=GH('UKPLab/sentence-transformers',STPIN,'sentence_transformers/sentence_transformer/losses/multiple_negatives_ranking.py');
const U_SEM=GH('UKPLab/sentence-transformers',STPIN,'examples/sentence_transformer/applications/semantic-search/README.md');
const U_E5=GH('microsoft/unilm',UNPIN,'e5/README.md'), U_E5EV=GH('microsoft/unilm',UNPIN,'e5/mteb_beir_eval.py'), U_BGE=GH('FlagOpen/FlagEmbedding',FEPIN,'README.md');
const BEIR_TIP=()=>tx('BEIR is a set of search tasks used as a retrieval benchmark; higher is better. The E5 README gives E5-large-v2 (1,024 dimensions, 24 layers) 50.6 and E5-small-v2 (384 dimensions, 12 layers) 49.0.','BEIR 是一组搜索任务，用来做检索基准，分数越高越好。E5 的 README 里，E5-large-v2（1,024 维，24 层）是 50.6，E5-small-v2（384 维，12 层）是 49.0。');
const U_PRE=GH('UKPLab/sentence-transformers',STPIN,'docs/sentence_transformer/pretrained_models.md');
const FIDPOP=()=>tx(`<p><b>Toy run:</b> your browser trains a real small encoder with the real in-batch-negatives loss.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>The small encoder: an embedding table of 16 numbers per token (English words; Chinese character pairs), started at random numbers between −1 and 1 (seed ${SEED}). A text’s vector is the average of its tokens’ rows, scaled to length 1. Unknown tokens of a held-out question are skipped.</li><li>Training: contrastive learning with the in-batch-negatives loss (InfoNCE, sentence-transformers’ MultipleNegativesRankingLoss in one direction, question to passage), the exact gradient, and Adam (learning rate 0.05, this lab’s choice). Each batch holds B different passages, as sentence-transformers’ no-duplicates sampler recommends.</li><li>Recall@1 and recall@5 count the 24 held-out questions against all 24 passages. The held-out questions reword the training questions: most of their words appear in their passage’s training questions. The map is a PCA of the vectors after the last step.</li><li>A run depends on its random start. The observations on batch size and temperature, and the challenge, average 16 starts (seeds ${SEED} + 1,000k); the stage shows the first.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>No Transformer: an average cannot see word order or negation, which real models can. 48 training pairs, while sentence-transformers’ all-* models were trained on more than 1B pairs, so a conclusion here holds in this lab; the help center (示例) is written for the lab.</li><li>The English and Chinese help centers are trained separately, so their numbers differ a little.</li></ul><div class="eyebrow">Sources</div><ul><li>${SRC(U_MNRL,'sentence-transformers, multiple_negatives_ranking.py')}: scale 20 = temperature 0.05, cosine similarity, “values between 10 and 100 are common”, “also known as the InfoNCE loss”, “much higher batch sizes (and thus better performance)”.</li><li>${SRC(U_PRE,'sentence-transformers, pretrained_models.md')}: the all-* models, “more than 1 billion training pairs”.</li><li>${SRC(U_E5,'E5 README')} (dimensions, BEIR) and ${SRC(U_E5EV,'mteb_beir_eval.py')} (<code>query: </code>, <code>passage: </code>); ${SRC(U_BGE,'FlagEmbedding README')} (BGE’s instruction, v1.5 also without it).</li></ul></div></div>`,
  `<p><b>玩具实跑：</b>页面在你的浏览器里，用真实的 batch 内负例 loss 训练一个真实的小编码器。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>小编码器：一张 embedding 表，每个 token（英文的词、中文的字对）一行 16 个数值，初始是 −1 到 1 之间的随机数（随机种子 ${SEED}）。一段文字的向量是它各 token 那几行的平均，再缩放成长度 1。新问题里表中没有的 token 直接跳过。</li><li>训练：对比学习，用 batch 内负例 loss（InfoNCE，即 sentence-transformers 的 MultipleNegativesRankingLoss，只算问题到段落一个方向），梯度是精确的，优化器是 Adam（学习率 0.05，这个 Lab 自己定的）。每个 batch 里的 B 段说明互不相同，和 sentence-transformers 推荐的不重复抽样一样。</li><li>recall@1 和 recall@5 拿 24 个新问题和全部 24 段比。新问题是训练问题的另一种问法：大部分字词在同一段的训练问题里出现过。图是最后一步向量的 PCA。</li><li>一次训练的结果和随机种子有关。讲 batch 大小和温度时的“观察到”以及挑战，取 16 个种子的平均（种子 ${SEED} + 1,000k）；舞台上画的是第一次。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul><li>没有 Transformer：取平均看不到词序和否定，真实的模型能看到。训练数据只有 48 对，sentence-transformers 的 all-* 模型用了 1B 对以上，所以这里的结论只在这个 Lab 里成立；帮助中心（示例）是为这个 Lab 写的。</li><li>中文和英文的帮助中心分开训练，所以两种语言的数字略有不同。</li></ul><div class="eyebrow">来源</div><ul><li>${SRC(U_MNRL,'sentence-transformers，multiple_negatives_ranking.py')}：scale 20 就是温度 0.05，用余弦相似度，“values between 10 and 100 are common”，“also known as the InfoNCE loss”，“much higher batch sizes (and thus better performance)”。</li><li>${SRC(U_PRE,'sentence-transformers，pretrained_models.md')}：all-* 模型用了“more than 1 billion training pairs”。</li><li>${SRC(U_E5,'E5 README')}（维数、BEIR 分数）和 ${SRC(U_E5EV,'mteb_beir_eval.py')}（<code>query: </code> 和 <code>passage: </code>）；${SRC(U_BGE,'FlagEmbedding README')}（BGE 的检索指令，v1.5 不加也能用）。</li></ul></div></div>`);
const PRIPOP=()=>tx(`<p>A softmax turns a row of scores into probabilities that add up to 1; dividing the scores by a temperature first makes it sharper or flatter. Cross-entropy is minus the log of the probability given to the right answer. ${LL('softmax-numerics')} covers both. An embedding table holds one vector per token, as in ${LL('architecture')}.</p>`,
                    `<p>softmax 把一行分数变成加起来为 1 的概率；先把分数除以温度，它就变尖或变平。cross-entropy 是正确答案那一项概率的负对数。这两个都在 ${LL('softmax-numerics')} 里讲。embedding 表每个 token 存一个向量，见 ${LL('architecture')}。</p>`);
function headerHTML(){return A.shell.header({id:ID,world:'apps',outcome:esc(OUTCOME()),short:tx('Embeddings','Embedding'),title:tx('How does an embedding know two sentences mean the same?','embedding 怎么知道两句话意思相同？'),
  minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: softmax and cross-entropy','前置知识：softmax 和 cross-entropy'),primerPop:PRIPOP(),descs:DESCS(),mode});}

/* ---------- stage ---------- */
function playLabel(){return anim?tx('❚❚ Pause','❚❚ 暂停'):tx('▶ Train','▶ 训练');}
function stageHTML(){
  return `<div class="panel stage emb-stage" id="emb-stage">`+
   `<div class="stage-head emb-head"><div class="emb-st-title" id="emb-st-title"></div><div class="stage-actions"><span class="status" id="emb-status" tabindex="0"></span></div></div>`+
   `<div class="emb-transport" id="emb-transport"></div>`+
   `<section class="emb-pan" id="emb-pan-pair"><div class="emb-pt" id="emb-pa-title"></div><div class="emb-pair" id="emb-pair"></div></section>`+
   `<div class="emb-row" id="emb-row"><section class="emb-pan" id="emb-pan-map"><div class="emb-pt" id="emb-m-title"></div>`+
     `<svg class="emb-map" id="emb-map" viewBox="0 0 360 300" role="img" aria-label=""></svg><div class="legend-row emb-leg" id="emb-m-leg"></div><div class="emb-side"><label class="emb-ql" for="emb-qsel">${tx('Search with a held-out question','用一个新问题搜索')}</label><select id="emb-qsel" class="emb-qsel" data-key="q"></select><ol class="emb-top" id="emb-top"></ol><p class="small emb-miss" id="emb-miss"></p></div></section>`+
   `<section class="emb-pan" id="emb-pan-heat"><div class="emb-pt" id="emb-h-title"></div><svg class="emb-heat" id="emb-heat" role="img" aria-label=""></svg><div class="legend-row emb-leg" id="emb-h-leg"></div></section></div>`+
   `<section class="emb-pan" id="emb-pan-chart"><div class="emb-chead"><div class="emb-pt" id="emb-c-title"></div><div id="emb-c-view"></div></div><svg class="emb-chart" id="emb-chart" viewBox="0 0 440 178" role="img" aria-label=""></svg><div class="legend-row emb-leg" id="emb-c-leg"></div></section>`+
   `<div class="metrics emb-metrics" id="emb-metrics"></div>`+
   `<details class="math" id="emb-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="emb-mathbody"></div></details></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="emb-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="emb-lock" hidden></p>`+
   `<div class="lockable">`+A.ui.seg({id:'emb-ctl-B',key:'B',label:tx('Pairs per batch, B','每个 batch 的配对数 B'),options:BS.map(b=>[String(b),String(b)]),value:String(S.B),
      note:tx('Each step takes B different passages with one of their questions; each question gets B − 1 in-batch negatives. At most 16 here, since the help center has 24 passages.','每一步取 B 段不同的说明，各配一个问题；每个问题有 B − 1 个 batch 内负例。这里最多 16，因为帮助中心只有 24 段。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'emb-ctl-tau',key:'tau',label:tx('Temperature τ','温度 τ'),options:TAUS.map(t=>[String(t),tauTxt(t)]),value:String(S.tau),
      note:tx('Every similarity is divided by τ before the softmax. sentence-transformers’ default is 0.05 (its scale 20); its documentation calls scales from 10 to 100 common, that is τ from 0.1 to 0.01.','softmax 之前，每个相似度都除以 τ。sentence-transformers 默认 0.05（scale 20），它的文档说 scale 常在 10 到 100 之间，也就是 τ 在 0.1 到 0.01 之间。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'emb-ctl-d',key:'d',label:tx('Dimensions per vector, d','每个向量的维数 d'),options:DS.map(d=>[String(d),String(d)]),value:String(S.d),
      note:tx('The length of every row of the table and of every text’s vector. Stored as float32, each vector takes 4 × d bytes: a 768-dimensional one 3,072 bytes, so 1M passages take 3.07 GB (GB = 10⁹ bytes). Real models use hundreds: the three English E5 v2 models 384, 768 or 1,024.','表里每一行、每段文字的向量有多长。存成 float32 时每个向量占 4 × d 字节：768 维的向量占 3,072 字节，1M 段就是 3.07 GB（GB 是 10⁹ 字节）。真实的模型用几百维：E5 v2 的三个英文模型是 384、768 或 1,024 维。')})+`</div>`+
   `<div class="lockable">`+A.ui.range({id:'emb-ctl-steps',key:'steps',label:tx('Training steps','训练步数'),min:0,max:MAXS,value:S.steps,out:stepsTxt(S.steps),
      note:tx('Steps taken from the random table; each step uses one batch.','从随机的表开始训练了几步；每一步用一个 batch。')})+`</div>`+
   `<details class="adv"><summary>${tx('Fixed in this lab: optimizer, learning rate, seed, data','这个 Lab 里固定的：优化器、学习率、随机种子、数据')}</summary>`+
   `<p class="ctl-note">${tx(`Adam with learning rate ${LR}, β₁ 0.9, β₂ 0.999: this lab’s choice, set so that 200 steps show the whole effect; it does not carry over to a real model. Seed ${SEED} fixes the starting table and the batches: the same settings give the same run. The observations on batch size and temperature, and the challenge, also average 16 starts: seeds ${SEED} + 1,000k for k = 0 to 15. 48 training pairs and 24 held-out questions, written for this lab.`,`Adam，学习率 ${LR}，β₁ 0.9，β₂ 0.999：这个 Lab 自己定的，为了让 200 步就能看到完整的效果，不能照搬到真实模型上。随机种子 ${SEED} 决定初始的表和每一步的 batch：设置一样，运行就一样。讲 batch 大小和温度时的“观察到”以及挑战，还会取 16 个种子的平均，种子是 ${SEED} + 1,000k，k 从 0 到 15。48 对训练数据和 24 个新问题，都是为这个 Lab 写的。`)}</p></details>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import torch
import torch.nn.functional as F

class BagEncoder(torch.nn.Module):
    """The lab's small encoder: average rows, scale to 1."""
    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.table = torch.nn.EmbeddingBag(vocab_size, dim,
                                           mode="mean")

    def forward(self, ids, offsets):
        return F.normalize(self.table(ids, offsets), dim=-1)

def in_batch_loss(q, p, tau=0.05):
    # q, p: (B, d) unit vectors; p[i] answers q[i]
    sim = q @ p.T / tau            # (B, B) cosines / tau
    labels = torch.arange(len(q))  # true pairs: the diagonal
    return F.cross_entropy(sim, labels)

def search(query_vec, passage_vecs, k=5):
    scores = passage_vecs @ query_vec  # all unit length
    return scores.topk(k).indices`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="emb-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="emb-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The small encoder, the in-batch-negatives loss and the search, in PyTorch.','用 PyTorch 写的小编码器、batch 内负例 loss 和搜索。')}</p><pre class="code" id="emb-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="emb-prod">`+
   tx(`<li>sentence-transformers ${SRC(U_MNRL,'<code>multiple_negatives_ranking.py</code>')}: <code>MultipleNegativesRankingLoss</code>, cosine similarity times <code>scale</code> 20.0 (temperature 0.05), cross-entropy over the in-batch candidates; it recommends the no-duplicates batch sampler so that no negative repeats a positive.</li>`+
      `<li>sentence-transformers ${SRC(U_SEM,'<code>semantic-search/README.md</code>')}: normalise the vectors to length 1, then the dot product equals the cosine, and search is one matrix multiply.</li>`+
      `<li>E5 ${SRC(U_E5EV,'<code>e5/mteb_beir_eval.py</code>')}: <code>'query: '</code> before each question, <code>'passage: '</code> before each passage, and <code>F.normalize</code> on the output.</li>`+
      `<li>BGE ${SRC(U_BGE,'<code>FlagEmbedding/README.md</code>')}: the English v1.5 models put <code>Represent this sentence for searching relevant passages: </code> before a query.</li>`,
      `<li>sentence-transformers ${SRC(U_MNRL,'<code>multiple_negatives_ranking.py</code>')}：<code>MultipleNegativesRankingLoss</code>，余弦相似度乘以 <code>scale</code> 20.0（温度 0.05），在 batch 内的候选上算 cross-entropy；它推荐不重复抽样的 batch sampler，免得负例和正例重复。</li>`+
      `<li>sentence-transformers ${SRC(U_SEM,'<code>semantic-search/README.md</code>')}：把向量归一化成长度 1，点积就等于余弦，搜索就是一次矩阵乘法。</li>`+
      `<li>E5 ${SRC(U_E5EV,'<code>e5/mteb_beir_eval.py</code>')}：问题前加 <code>'query: '</code>，段落前加 <code>'passage: '</code>，输出再做 <code>F.normalize</code>。</li>`+
      `<li>BGE ${SRC(U_BGE,'<code>FlagEmbedding/README.md</code>')}：英文 v1.5 模型在问题前加 <code>Represent this sentence for searching relevant passages: </code>。</li>`)+
   `</ul><p class="small muted">${tx('Pinned commits: sentence-transformers ff69486 (6.2.0.dev0), microsoft/unilm 31c5b90, FlagEmbedding fd1a2bd.','固定的 commit：sentence-transformers ff69486（6.2.0.dev0），microsoft/unilm 31c5b90，FlagEmbedding fd1a2bd。')}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/1807.03748" target="_blank" rel="noopener">Representation Learning with Contrastive Predictive Coding</a><span>${tx('van den Oord et al., 2018: the InfoNCE loss','van den Oord 等，2018：InfoNCE loss')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2004.04906" target="_blank" rel="noopener">Dense Passage Retrieval for Open-Domain Question Answering</a><span>${tx('Karpukhin et al., 2020: question–passage pairs with in-batch negatives','Karpukhin 等，2020：问题和段落配对，用 batch 内负例')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/1908.10084" target="_blank" rel="noopener">Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks</a><span>${tx('Reimers and Gurevych, 2019','Reimers 和 Gurevych，2019')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2212.03533" target="_blank" rel="noopener">Text Embeddings by Weakly-Supervised Contrastive Pre-training</a><span>${tx('Wang et al., 2022 (E5)','Wang 等，2022（E5）')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="emb-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="emb-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="emb-app m2" id="emb-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="emb-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* which panels the stage shows now */
function ctx(){
  if(mode==='guided'){const s=STEPS[step]; return {panels:s.panels,run:s.panels.includes('chart')||s.id==='train'};}
  return {panels:['map','heat','chart','metrics'],run:true,noTr:mode==='challenge'&&!chSubmitted};}
const TCOL=t=>`emb-c-${t}`;
function drawPair(){
  const i=0, k=kwRank(LG(),i), top=DATA.map((x,j)=>({j,s:k.sc[j]})).sort((a,b)=>b.s-a.s||a.j-b.j).slice(0,3), ahead=k.sc.filter((v,j)=>j!==i&&v>k.sc[i]).length;
  $('#emb-pa-title',app).textContent=tx('A keyword search for one question: which passages share the most words with it?','用关键词搜一个问题：哪几段和它共同的字对最多？');
  const sw=j=>{const w=shared(LG(),i,j); return w.length?w.map(esc).join(tx(', ','、')):tx('none','没有');};
  $('#emb-pair',app).innerHTML=`<div class="emb-pq"><span class="emb-lab">${tx('The question','问题')}</span><p>${esc(qText(i))}</p></div>`+
   `<div class="emb-pp"><span class="emb-lab">${tx('The passage that answers it','能回答它的那一段')}${tx(': ','：')}${esc(pTitle(i))}</span><p>${esc(pText(i))}</p><p class="emb-bad small">${tx(`Shared words: none. A keyword search returns ${ahead} other passages before it.`,`共同的字对：没有。关键词搜索会先给出另外 ${ahead} 段。`)}</p></div>`+
   `<div class="emb-kw"><span class="emb-lab">${tx('What the keyword search returns first','关键词搜索排在最前面的')}</span><ol>${top.map(x=>`<li><b>${esc(pTitle(x.j))}</b> <span class="muted">${tx(`shares ${x.s} word${x.s===1?'':'s'}`,`共同的字对 ${x.s} 个`)}${tx(': ','：')}${sw(x.j)}</span></li>`).join('')}</ol><p class="small muted">${tx('Only words like these are shared: they say nothing about warmth.','共同的只是这类字对，和“暖和”没有关系。')}</p></div>`;}
function drawMapLegend(){
  const el=$('#emb-m-leg',app), k=A.lang(); if(el._k===k) return; el._k=k;
  el.innerHTML=TOPICS.map(t=>`<span class="lg"><i class="sw emb-sw ${TCOL(t)}"></i>${esc(L(TOPIC_N[t]))}</span>`).join('')+
   `<span class="lg" data-tipx="${esc(tx('A filled dot is one of the 24 passages, coloured by its topic.','实心点是 24 段说明中的一段，颜色是它的主题。'))}"><i class="sw emb-sw-p"></i>${tx('passage','说明段落')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('A hollow circle is a held-out question; a thin line joins it to its own passage.','空心圈是一个新问题，细线连到它自己的段落。'))}"><i class="sw emb-sw-q"></i>${tx('held-out question','新问题')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('The thick ring is the question you search with; dashed lines go to its 3 most similar passages, numbered by rank.','粗圈是你用来搜索的问题；虚线连到和它最相似的 3 段，数字是名次。'))}"><i class="sw emb-sw-me"></i>${tx('your question, its top 3','你的问题和它的前 3 名')}</span>`+
   `<span class="lg" data-tipx="${esc(tx('A green ring: the right passage is the most similar. A red ring: it is not.','绿圈：正确段落是最相似的。红圈：它不是。'))}"><i class="sw emb-sw-ok"></i><i class="sw emb-sw-bad"></i>${tx('right passage: found first / missed','正确段落：排第一 / 没排第一')}</span>`;}
function drawMap(r){
  const T=r.T, t=r.t, xy=T.pca.xy[t], R=T.pca.R*1.06||1, W=360, H=300, cx=W/2, cy=H/2, sc=Math.min(W,H)/2/R*0.94;
  const X=p=>cx+p[0]*sc, Y=p=>cy-p[1]*sc, q=S.q, d=T.d;
  const top=topTxt(T,t,q,3), rk=T.rank[t][q];
  let h=`<rect class="emb-frame" x="0.5" y="0.5" width="${W-1}" height="${H-1}" rx="6"></rect>`;
  for(let i=0;i<NP;i++){const a=xy[i], b=xy[NP+i]; h+=`<line class="emb-pl ${TCOL(DATA[i][0])}" x1="${X(a).toFixed(1)}" y1="${Y(a).toFixed(1)}" x2="${X(b).toFixed(1)}" y2="${Y(b).toFixed(1)}"></line>`;}
  top.forEach(x=>{const a=xy[NP+q], b=xy[x.j]; h+=`<line class="emb-tl" x1="${X(a).toFixed(1)}" y1="${Y(a).toFixed(1)}" x2="${X(b).toFixed(1)}" y2="${Y(b).toFixed(1)}"></line>`;});
  for(let i=0;i<NP;i++){const a=xy[NP+i]; h+=`<circle class="emb-qd ${TCOL(DATA[i][0])}" cx="${X(a).toFixed(1)}" cy="${Y(a).toFixed(1)}" r="3.4" data-tipx="${esc(tx(`Held-out question ${quote(qText(i))}: its own passage ranks ${T.rank[t][i]+1} of 24 at step ${t}.`,`新问题${quote(qText(i))}第 ${t} 步时，它自己的段落排第 ${T.rank[t][i]+1}（共 24 段）。`))}"></circle>`;}
  for(let i=0;i<NP;i++){const a=xy[i]; h+=`<circle class="emb-pd ${TCOL(DATA[i][0])}" cx="${X(a).toFixed(1)}" cy="${Y(a).toFixed(1)}" r="5.2" data-tipx="${esc(tx(`Passage (${L(TOPIC_N[DATA[i][0]])}). ${pText(i)}`,`说明（${L(TOPIC_N[DATA[i][0]])}）。${pText(i)}`))}"></circle>`;}
  const own=xy[q]; h+=`<circle class="emb-own ${rk===0?'ok':'bad'}" cx="${X(own).toFixed(1)}" cy="${Y(own).toFixed(1)}" r="9"></circle>`;
  top.forEach((x,k)=>{const b=xy[x.j]; h+=`<text class="emb-rk" x="${(X(b)+8).toFixed(1)}" y="${(Y(b)-7).toFixed(1)}">${k+1}</text>`;});
  const me=xy[NP+q]; h+=`<circle class="emb-me" cx="${X(me).toFixed(1)}" cy="${Y(me).toFixed(1)}" r="6.5"></circle>`;
  const svg=$('#emb-map',app); svg.innerHTML=h; svg.setAttribute('aria-label',tx(`Map of the 24 passages and 24 held-out questions at step ${t}.`,`第 ${t} 步时 24 段说明和 24 个新问题的分布图。`));
  $('#emb-m-title',app).innerHTML=tx(`Where the vectors point at step ${t}: 24 passages and 24 held-out questions (${DEF(`${d} dimensions projected onto a plane`,D_PCA())})`,`第 ${t} 步时向量的方向：24 段说明和 24 个新问题（${DEF(`把 ${d} 维向量投影到平面上`,D_PCA())}）`);
  const sel=$('#emb-qsel',app); if(sel._k!==A.lang()){sel.innerHTML=DATA.map((x,i)=>`<option value="${i}">${esc(qText(i))}</option>`).join(''); sel._k=A.lang();} if(+sel.value!==q) sel.value=String(q);
  $('#emb-top',app).innerHTML=top.map(x=>`<li class="${x.j===q?'emb-hit':''}"><span>${esc(pTitle(x.j))}</span><span class="mono">${fx(x.s,2)}</span></li>`).join('');
  const miss=$('#emb-miss',app), ownS=dot(T.H[t],q*d,T.P[t],q*d,d);
  miss.className='small emb-miss'+(rk===0?' ok':' bad');
  miss.innerHTML=rk===0?tx(`The right passage, ${quote(pTitle(q))}, is first (cosine ${fx(ownS,2)}).`,`正确段落${quote(pTitle(q))}排第一（余弦 ${fx(ownS,2)}）。`):tx(`The right passage, ${quote(pTitle(q))}, ranks ${rk+1} (cosine ${fx(ownS,2)}).`,`正确段落${quote(pTitle(q))}排第 ${rk+1}（余弦 ${fx(ownS,2)}）。`);}
const showP=()=>!(mode==='guided'&&step<SIX('train'));   // softmax and the loss are defined where training starts: before that, no probabilities (R2.8)
function drawHeat(r){
  const T=r.T, t=r.t, B=S.B, Sx=T.S[t], Pr=T.Pr[t], bt=T.batch[t], sp=showP(), cs=B<=4?40:(B<=8?30:13), lab=18, gp=sp?(B<=2?60:12):0, bw=sp?40:0, pw=sp&&B<=8?34:0, W=Math.max(220,lab+B*cs+gp+bw+pw), H=lab+B*cs+4, txt=B<=8;
  let h=`<text class="emb-ax" x="${Math.max(80,lab+B*cs/2)}" y="11" text-anchor="middle">${tx('passages of the batch','batch 里的说明')}</text>`;
  if(sp) h+=`<text class="emb-ax" x="${lab+B*cs+gp-2+bw+pw}" y="11" text-anchor="end">${tx('own p','正确的 p')}</text>`;
  for(let i=0;i<B;i++){
    h+=`<text class="emb-ax" x="${lab-4}" y="${lab+i*cs+cs/2+3}" text-anchor="end">${i+1}</text>`;
    for(let j=0;j<B;j++){const s=Sx[i*B+j], op=Math.max(0,Math.min(1,s))*0.85+0.05, x=lab+j*cs, y=lab+i*cs;
      const tip=tx(`Row ${i+1}, the question ${quote(DATA[Math.floor(bt.qs[i]/2)][3][bt.qs[i]%2][0])}, against column ${j+1}, the passage ${quote(pTitle(bt.ps[j]))}: cosine ${fx(s,2)}${sp?`; the softmax gives it ${pc(Pr[i*B+j],0)} of the row`:''}.`,`第 ${i+1} 行的问题${quote(DATA[Math.floor(bt.qs[i]/2)][3][bt.qs[i]%2][1])}和第 ${j+1} 列的段落${quote(pTitle(bt.ps[j]))}，余弦是 ${fx(s,2)}${sp?`；softmax 给它这一行的 ${pc(Pr[i*B+j],0)}`:''}。`);
      h+=`<rect class="emb-cell${i===j?' dg':''}" x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" style="fill-opacity:${op.toFixed(3)}" data-tipx="${esc(tip)}"></rect>`;
      if(txt) h+=`<text class="emb-cv${op>0.5?' inv':''}" x="${x+cs/2}" y="${y+cs/2+3.5}" text-anchor="middle">${fx(s,1)}</text>`;}
    if(!sp) continue;
    const pd=Pr[i*B+i], bx=lab+B*cs+gp; h+=`<rect class="emb-pbt" x="${bx}" y="${lab+i*cs+cs*0.2}" width="${bw}" height="${cs*0.6}"></rect><rect class="emb-pb" x="${bx}" y="${lab+i*cs+cs*0.2}" width="${(bw*pd).toFixed(1)}" height="${cs*0.6}"></rect>`;
    if(B<=8) h+=`<text class="emb-ax" x="${bx+bw+3}" y="${lab+i*cs+cs/2+3}">${pc(pd)}</text>`;}
  const svg=$('#emb-heat',app); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.style.maxWidth=Math.round(W*1.1)+'px'; svg.innerHTML=h;
  svg.setAttribute('aria-label',tx(`Similarity table of one more batch drawn after ${t} training steps.`,`训练 ${t} 步后再抽一个 batch，它的相似度表。`));
  $('#emb-h-title',app).textContent=tx(`After ${t} training steps, one more batch drawn to look at: ${B} questions (rows) and their ${B} passages (columns)`,`训练 ${t} 步后，再抽一个 batch 来看：${B} 个问题（行）和它们的 ${B} 段说明（列）`);
  const lg=$('#emb-h-leg',app), k=A.lang()+B+sp; if(lg._k!==k){lg._k=k; lg.innerHTML=`<span class="lg"><i class="sw emb-sw-cell"></i>${tx('darker: higher cosine','越深：余弦越高')}</span><span class="lg"><i class="sw emb-sw-dg"></i>${tx('framed: the true pair','带框：真正的配对')}</span>`+(sp?`<span class="lg" data-tipx="${esc(tx('The bar on the right of each row: the probability the softmax gives the row’s own passage. The loss of that row is minus its log.','每行右边的条：softmax 给这一行正确段落的概率。这一行的 loss 就是它的负对数。'))}"><i class="sw emb-sw-pb"></i>${tx('own passage’s probability','正确段落的概率')}</span>`:'');}}
const VIEWS=[['r1',{en:'Recall',zh:'召回率'}],['loss',{en:'Loss',zh:'loss'}]];
function ghosts(){
  if(mode==='guided'&&STEPS[step].id==='batch') return [...triedB].filter(b=>b!==S.B).map(b=>({T:run(LG(),b,S.tau,S.d),lab:`B = ${b}`}));
  if(mode==='guided'&&STEPS[step].id==='temp') return [...triedT].filter(x=>x!==S.tau).map(x=>({T:run(LG(),S.B,x,S.d),lab:`τ = ${x}`}));
  return [];}
function drawChart(r){
  const T=r.T, t=r.t, W=440, H=178, X0=40, X1=W-64, Y0=12, Y1=H-30, gs=ghosts(), loss=view==='loss';
  const ser=TT=>loss?TT.loss:TT.r1;
  const LO=-3, HI=2, ymax=1;   // the loss view is on a log scale from 0.001 to 100
  const X=s=>X0+(X1-X0)*s/MAXS, Y=v=>loss?Y1-(Y1-Y0)*(Math.max(LO,Math.min(HI,Math.log10(Math.max(v,1e-12))))-LO)/(HI-LO):Y1-(Y1-Y0)*Math.min(1,v/ymax);
  let h='';
  const yt=loss?[0.001,0.01,0.1,1,10,100]:[0,0.25,0.5,0.75,1];
  yt.forEach(v=>{h+=`<line class="emb-grid" x1="${X0}" x2="${X1}" y1="${Y(v)}" y2="${Y(v)}"></line><text class="emb-ax" x="${X0-5}" y="${Y(v)+3}" text-anchor="end">${loss?String(v):pc(v)}</text>`;});
  if(loss) h+=`<text class="emb-ax" x="${X0+4}" y="${Y0+2}">${tx('log scale','对数刻度')}</text>`;
  [0,50,100,150,200].forEach(s=>{h+=`<text class="emb-ax" x="${X(s)}" y="${Y1+13}" text-anchor="middle">${s}</text>`;});
  h+=`<text class="emb-ax" x="${(X0+X1)/2}" y="${H-3}" text-anchor="middle">${tx('training steps','训练步数')}</text>`;
  if(!loss) h+=`<line class="emb-chance" x1="${X0}" x2="${X1}" y1="${Y(1/NP)}" y2="${Y(1/NP)}"></line>`;
  const path=(arr,upto)=>arr.slice(0,upto+1).map((v,s)=>`${s?'L':'M'}${X(s).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  gs.forEach(g=>{const a=ser(g.T); h+=`<path class="emb-ghost" d="${path(a,MAXS)}"></path><text class="emb-gl" x="${W-2}" y="${Y(a[MAXS])+3}" text-anchor="end">${esc(g.lab)}</text>`;});
  if(!loss) h+=`<path class="emb-r5" d="${path(T.r5,t)}"></path>`;
  h+=`<path class="emb-line" d="${path(ser(T),t)}"></path>`;
  h+=`<line class="emb-head-l" x1="${X(t)}" x2="${X(t)}" y1="${Y0}" y2="${Y1}"></line><circle class="emb-dot" cx="${X(t)}" cy="${Y(ser(T)[t])}" r="3.5"></circle>`;
  const lab=loss?fmtLoss(T.loss[t]):pc(T.r1[t]), rt=t<MAXS*0.7; h+=`<text class="emb-cl" x="${rt?X(t)+6:X(t)-6}" y="${Math.max(Y0+10,Y(ser(T)[t])-6)}"${rt?'':' text-anchor="end"'}>${lab}</text>`;
  const svg=$('#emb-chart',app); svg.innerHTML=h;
  svg.setAttribute('aria-label',loss?tx(`Loss of the batch drawn at each step, up to step ${t}.`,`到第 ${t} 步为止，每一步抽到的 batch 的 loss。`):tx(`Recall@1 on the held-out questions up to step ${t}.`,`到第 ${t} 步为止，新问题的 recall@1。`));
  $('#emb-c-title',app).textContent=loss?tx('How does the loss of each step’s batch change as training runs?','训练时，每一步那个 batch 的 loss 怎么变？'):tx('How many held-out questions find their own passage first as training runs?','训练时，有多少新问题第一个就找到自己的段落？');
  const vw=$('#emb-c-view',app), vk=A.lang()+view; if(vw._k!==vk){vw._k=vk; vw.innerHTML=A.ui.seg({id:'emb-view',key:'view',label:tx('Show','显示'),options:VIEWS.map(([k,n])=>[k,L(n)]),value:view});}
  const cur=`B = ${S.B}, τ = ${S.tau}, d = ${S.d}`;
  $('#emb-c-leg',app).innerHTML=(loss?`<span class="lg"><i class="sw emb-sw-line"></i>${tx(`loss (${cur})`,`loss（${cur}）`)}</span>`:`<span class="lg"><i class="sw emb-sw-line"></i>${tx(`recall@1 (${cur})`,`recall@1（${cur}）`)}</span><span class="lg"><i class="sw emb-sw-r5"></i>recall@5</span><span class="lg" data-tipx="${esc(tx('Picking one of the 24 passages at random finds the right one 1 time in 24: 4%.','24 段里随便挑一段，24 次里对 1 次：4%。'))}"><i class="sw emb-sw-ch"></i>${tx('random pick (4%)','随便挑（4%）')}</span>`)+
    (gs.length?`<span class="lg"><i class="sw emb-sw-gh"></i>${tx('other settings you tried (named at the end of each line)','你试过的其他设置（名字写在线的末端）')}</span>`:'')+
    `<span class="lg"><i class="sw emb-sw-head"></i>${tx('current step','当前步')}</span>`;}
const fmtLoss=v=>v>=0.01?f(v,2):(v>=0.001?f(v,3):tx('under 0.001','不到 0.001'));
function niceMax(v){if(v<=0) return 1; const e=Math.pow(10,Math.floor(Math.log10(v))), m=v/e; return (m<=1?1:m<=2?2:m<=5?5:10)*e;}
const MET=[
 {k:'r1',label:()=>tx('Recall@1, held-out','新问题 recall@1'),v:r=>pc(r.r1),sub:r=>tx(`24 questions; ${pc(r.T.r1[0])} untrained`,`24 个问题；训练前 ${pc(r.T.r1[0])}`),tip:()=>tx('The share of the 24 held-out questions whose own passage is the most similar of the 24 passages.','24 个新问题里，正确段落是 24 段中最相似那段的占多少。'),pts:true},
 {k:'r5',label:()=>tx('Recall@5, held-out','新问题 recall@5'),v:r=>pc(r.r5),sub:()=>tx('own passage in the top 5','正确段落在前 5 名'),tip:()=>tx('The share of the 24 held-out questions whose own passage is among their 5 most similar passages.','24 个新问题里，正确段落在相似度前 5 名的占多少。'),pts:true},
 {k:'pd',label:()=>tx('Own passage’s probability','正确段落的概率'),v:r=>pc(r.pd),sub:r=>tx(`one more batch, average of ${S.B} rows`,`再抽的 batch，${S.B} 行平均`),tip:()=>tx('Draw one more batch after the current step: the probability the softmax gives each question’s own passage, averaged over the rows.','当前这一步之后再抽一个 batch：softmax 给每个问题正确段落的概率，按行取平均。'),pts:true},
 {k:'loss',label:()=>tx('Loss, one more batch','再抽一个 batch 的 loss'),v:r=>f(r.loss,2),sub:()=>tx(`all passages equal: ln ${S.B} = ${f(Math.log(S.B),2)}`,`每段概率一样时：ln ${S.B} = ${f(Math.log(S.B),2)}`),tip:()=>tx('Draw one more batch after the current step: its in-batch-negatives loss, averaged over its questions. Giving every passage the same probability scores ln B.','当前这一步之后再抽一个 batch：它的 batch 内负例 loss，按问题取平均。给每段一样的概率时是 ln B。')}];
let metKey=null;
function drawMetrics(r){
  const host=$('#emb-metrics',app), rk=runKey(S), cfgCh=metKey!=null&&metKey!==rk; metKey=rk; const mets=showP()?MET:MET.slice(0,2), hk=A.lang()+mets.length;
  if(!host.children.length||host._lang!==hk){host.innerHTML=mets.map(m=>`<div class="metric" data-k="${m.k}" data-tipx="${esc(m.tip())}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join(''); host._lang=hk;}
  mets.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r); el.dataset.tipx=m.tip();
    let dt=''; if(userAct&&cfgCh&&!anim&&shownPrev[m.k]!=null&&shownPrev[m.k]!==v&&m.pts) dt=A.fmt.delta(parseFloat(shownPrev[m.k])/100,parseFloat(v)/100,{pts:true,dp:0});
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2800);} else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=v;});}
function drawHead(r){
  const st=$('#emb-st-title',app), el=$('#emb-status',app), c=ctx();
  const pair=c.panels[0]==='pair'; el.hidden=pair;
  st.textContent=pair?tx('Keyword search: the help center of Lumo Home (example)','关键词搜索：小鹿家居的帮助中心（示例）'):(anim?tx(`Training the ${S.d}-dimensional small encoder: step ${r.t} of ${mode==='challenge'?chN:MAXS} (${S.B} pairs per batch, τ = ${S.tau})`,`正在训练 ${S.d} 维的小编码器：第 ${r.t} 步，共 ${mode==='challenge'?chN:MAXS} 步（每个 batch ${S.B} 对，τ = ${S.tau}）`):
    (r.t===0?tx(`The ${S.d}-dimensional small encoder before training`,`训练前的 ${S.d} 维小编码器`):tx(`The ${S.d}-dimensional small encoder after ${stepsTxt(r.t)} (${S.B} pairs per batch, τ = ${S.tau})`,`训练 ${r.t} 步后的 ${S.d} 维小编码器（每个 batch ${S.B} 对，τ = ${S.tau}）`)));
  if(mode==='challenge'){const done=chDone(), p=done?chPoint(LG(),S.B,chN):null; el.className='status'+(done?(p.ok?' ok':' bad'):''); el.textContent=done?(p.ok?tx(`Average reaches ${G55()}`,`平均达到 ${G55()}`):tx(`Average under ${G55()}`,`平均不到 ${G55()}`)):(chSubmitted?tx(`Training: step ${S.steps}`,`训练中：第 ${S.steps} 步`):tx('Not trained yet','还没训练')); el.dataset.tipx=tx(`Whether recall@1 on the 24 held-out questions, averaged over 16 random starts, reaches the ${G55()} goal. It shows once you submit and the training has run.`,`24 个新问题的 recall@1，按 16 个随机种子取平均，是否达到 ${G55()}。提交并训练完以后才显示。`); return;}
  el.className='status'; el.textContent=tx(`Recall@1 ${pc(r.r1)}`,`recall@1 ${pc(r.r1)}`); el.dataset.tipx=tx('Recall@1 on the 24 held-out questions now.','现在 24 个新问题的 recall@1。');}
function drawTransport(){
  const tEl=$('#emb-transport',app), c=ctx(), want=A.lang()+'|'+(anim?1:0)+'|'+runSpeed+'|'+mode+'|'+c.run+'|'+(c.noTr?1:0);
  tEl.hidden=!c.run||!!c.noTr;
  if(tEl._k!==want){tEl._k=want;
    tEl.innerHTML=`<button type="button" class="btn sm" data-act="play" data-play data-tipx="${esc(tx('Train the small encoder step by step; press again to pause.','把小编码器一步步训练下去；再按一次暂停。'))}">${playLabel()}</button>`+
      A.ui.seg({id:'emb-speed',key:'speed',label:tx('Speed','速度'),options:['slow','mid','fast'].map(k=>[k,L(SPEED_N[k])]),value:runSpeed})+
      `<button type="button" class="btn ghost sm" data-act="replay" data-tipx="${esc(tx('Watch this run again from step 0.','从第 0 步再看一遍这次运行。'))}">${tx('Replay','重放')}</button>`+
      (mode==='challenge'?'':A.ui.range({id:'emb-tr-steps',key:'steps',label:tx('Step','步数'),min:0,max:MAXS,value:S.steps,out:String(S.steps),inline:true}));
    A.ui.initRanges(tEl);}
  const i=$('#emb-tr-steps-in',app); if(i&&+i.value!==S.steps){i.value=S.steps; A.ui.setRange(i);} const o=i&&i.closest('.inline-ctl,.ctl'), out=o&&o.querySelector('output'); if(out) out.textContent=String(S.steps);}
function panelsShow(){const c=ctx(); ['pair','map','heat','chart'].forEach(p=>{const el=$('#emb-pan-'+p,app); if(el) el.hidden=!c.panels.includes(p);}); const mt=$('#emb-metrics',app); if(mt) mt.hidden=!c.panels.includes('metrics');
  const row=$('#emb-row',app), two=c.panels.includes('map')&&c.panels.includes('heat'); row.classList.toggle('two',two); row.hidden=!c.panels.includes('map')&&!c.panels.includes('heat');
  const ch=$('#emb-pan-chart',app), chFirst=c.panels[0]==='chart';   // the main picture of the part comes first (R7.12)
  if(chFirst&&ch.nextElementSibling!==row) row.parentNode.insertBefore(ch,row); else if(!chFirst&&row.nextElementSibling!==ch) row.parentNode.insertBefore(ch,row.nextSibling);}
/* Show the math: four short blocks, each a lead line, the named formula, the numbers of the current state, the result */
function mathHTML(r){
  const T=r.T, t=r.t, d=T.d, b=built(LG()), q=S.q, n=b.Hi[q].length, B=S.B, Sx=T.S[t], Pr=T.Pr[t], tau=S.tau;
  const ownS=dot(T.H[t],q*d,T.P[t],q*d,d), row=[]; for(let j=0;j<B;j++) row.push(Sx[j]);
  const blk=(lead,tex,sub)=>`<div class="emb-mb"><p class="emb-mb-lead">${lead}</p>${M(tex)}${sub?M(sub):''}</div>`;
  /* row 1 rewritten as log(1 + Σ_{j≠1} e^{(s_1j − s_11)/τ}): the two biggest terms written out, the rest summed as r, so the line can be
     recomputed by hand and fits a phone (kit_math_scroll) */
  const ex=row.map((s,j)=>({j,s,e:Math.exp((s-row[0])/tau)})).slice(1).sort((a,b)=>b.e-a.e), top=ex.slice(0,2), rest=ex.slice(2).reduce((a,c)=>a+c.e,0), l1=Math.log1p(ex.reduce((a,c)=>a+c.e,0));
  const terms=top.map(x=>`e^{(${fx(x.s,3)}-${fx(row[0],3)})/${tau}}`).join(' + ')+(ex.length>2?' + r':''), terms2=top.map(x=>`e^{${fx((x.s-row[0])/tau,2)}}`).join(' + ')+(ex.length>2?` + ${sig2(rest)}`:'');
  const k1=Math.round(T.r1[t]*NP);
  return blk(tx(`1. The vector of your question: the average of the rows of its ${n} tokens that appear in the training data (and so have a row in the table), scaled to length 1 (${d} numbers).`,`1. 你的问题的向量：它有 ${n} 个 token 在训练数据里出现过、在表里有自己的一行，取这 ${n} 行的平均，再缩放成长度 1（${d} 个数值）。`),ENC_TEX(tx('text vector','文字的向量')),AVG_TEX)+
   blk(tx(`2. Its similarity with its own passage at step ${t}: the dot product of two unit vectors.`,`2. 第 ${t} 步时它和自己那段的相似度：两个单位向量的点积。`),COS_TEX(tx('similarity','相似度')),`\\cos(\\text{${tx('question','问题')}}, \\text{${tx('passage','段落')}}) = ${fx(ownS,3)}`)+
   `<div class="emb-mb"><p class="emb-mb-lead">${tx(`3. The loss of row 1 of one more batch drawn now (B = ${B}, τ = ${tau}). The numerator is ${MI('e^{s_{11}/\\tau}')} of its own passage; the denominator adds up all ${B} of the row:`,`3. 现在再抽一个 batch，它第 1 行的 loss（B = ${B}，τ = ${tau}）。分子是它自己那段的 ${MI('e^{s_{11}/\\tau}')}，分母把这一行 ${B} 段的都加起来：`)}</p>${M(LOSS_TEX('loss'))}<p class="emb-mb-lead">${tx(`Divided through by the numerator, only the gaps to the own passage are left${ex.length>2?`; the two biggest terms are written out, r is the other ${ex.length-2}`:''}:`,`分子分母同除以分子，只剩下和自己那段的差${ex.length>2?`；写出最大的两项，r 是其余 ${ex.length-2} 项的和`:''}：`)}</p>${M(`\\ell_1 = \\log\\Big(1 + \\sum_{j \\ne 1} e^{(s_{1j}-s_{11})/\\tau}\\Big)`)}${M(`= \\log(1 + ${terms})`)}${M(`= \\log(1 + ${terms2}) = ${sig2(l1)}`)}</div>`+
   blk(tx(`4. Recall@1: held-out questions whose own passage is the most similar, out of 24.`,`4. recall@1：自己的段落是最相似那段的新问题，除以 24。`),`\\text{recall@1} = \\frac{${k1}}{24} = ${f(T.r1[t]*100,1)}\\,\\text{%}`);}
/* two significant digits (TeX): written out down to 0.001, below that as a power of ten so the line still fits a phone */
function sig2(v){if(!(v>0)) return '0'; if(v>=100) return f(v,0); if(v>=0.1) return f(v,2); const e=Math.floor(Math.log10(v));
  if(e>=-3) return v.toFixed(1-e); let m=+(v/Math.pow(10,e)).toFixed(1), k=e; if(m>=10){m=1; k++;} return `${m.toFixed(1)} \\times 10^{${k}}`;}

function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(gEl=>{const k=gEl.dataset.key;
    if(k==='view') A.ui.setPressed(gEl,view); else if(k==='speed') A.ui.setPressed(gEl,runSpeed); else if(k==='chN') A.ui.setPressed(gEl,String(chN)); else if(S[k]!=null) A.ui.setPressed(gEl,String(S[k]));});
  $$('#emb-controls input[data-key="steps"]',app).forEach(i=>{if(+i.value!==S.steps) i.value=S.steps; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl'), out=o&&o.querySelector('output'); if(out) out.textContent=stepsTxt(S.steps);});
  const lock=mode==='challenge', c=$('#emb-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#emb-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=tx('In the challenge you choose only the batch size and the training steps, in the task card. The rest is fixed: τ = 0.05, 16 dimensions, the same 48 training pairs.','挑战中只能在任务卡里选 batch 大小和训练步数。其余固定：τ = 0.05、16 维、同样的 48 对训练数据。');
  $$('#emb-controls .lockable',app).forEach(e=>{e.hidden=lock;});}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function updateRail(r){
  const rail=$('#emb-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect(r)) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs(r); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again&&!anim) A.ui.obsBump(box);}}}
    A.ui.setNext(rail,isNextDisabled());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-play]',rail).forEach(b=>{b.textContent=playLabel();});
  } else if(mode==='sandbox'){
    const c=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===c)));
    const ins=$('#emb-insight',rail); if(ins){const html=`<div class="eyebrow emb-keep">${c>=0?esc(L(PRESETS[c].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {
    const av=$('#emb-ch-av',rail), done=chDone(); if(!av) return;
    const p=done?chPoint(LG(),S.B,chN):null, ok=done&&p.ok;
    av.textContent=done?pc(p.r1):(chSubmitted?tx(`first start, step ${S.steps}: ${pc(r.r1)}`,`第一个种子，第 ${S.steps} 步：${pc(r.r1)}`):tx('shown after you submit','提交后才知道'));
    $('#emb-ch-gv',rail).textContent=tx(`${chN} (B = ${S.B})`,`${chN} 步（B = ${S.B}）`);
    $('#emb-ch-ap i',rail).style.width=(chSubmitted?Math.min(100,(done?p.r1:r.r1)/CH_GOAL*100):0)+'%'; $('#emb-ch-a',rail).classList.toggle('met',ok); $('#emb-ch-ap',rail).classList.toggle('met',ok);
    const sub=$('#emb-ch-sub',rail); if(sub) sub.hidden=chSubmitted;
    if(done&&p.ok) chReveal=true;
    const res=$('#emb-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:s.nextWhy?L(s.nextWhy):undefined,skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#emb-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  const r=compute(S);
  panelsShow(); drawTransport(); drawHead(r);
  if(!$('#emb-pan-pair',app).hidden) drawPair();
  if(!$('#emb-pan-map',app).hidden){drawMapLegend(); drawMap(r);}
  if(!$('#emb-pan-heat',app).hidden) drawHeat(r);
  if(!$('#emb-pan-chart',app).hidden) drawChart(r);
  if(!$('#emb-metrics',app).hidden) drawMetrics(r);
  const mb=$('#emb-mathbody',app), mh=mathHTML(r); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(r);
  A.hash.write(token()); A.dock.refresh(); announce(r);}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; A.announce(tx(`After ${stepsTxt(r.t)}: recall@1 ${pc(r.r1)}.`,`训练 ${r.t} 步后：recall@1 ${pc(r.r1)}。`));},400);}
function setState(p){
  Object.assign(S,p);
  if(mode==='challenge'){S.tau=0.05; S.d=16; if(!BS.includes(S.B)) S.B=2;}
  if(!BS.includes(S.B)) S.B=8; if(!TAUS.includes(S.tau)) S.tau=0.05; if(!DS.includes(S.d)) S.d=16;
  S.steps=Math.max(0,Math.min(MAXS,Math.round(S.steps))); if(!(S.q>=0&&S.q<NP)) S.q=0;
  render();}
/* ---------- animation (R3.9, R3.20): a run plays only when it is new; one already watched shows its end at once ---------- */
function stopAnim(){if(raf){cancelAnimationFrame(raf); raf=0;} const was=anim; anim=null; lastTs=0; return was;}
const endStep=()=>mode==='challenge'?chN:MAXS;
function finishAnim(){const a=stopAnim(); if(!a) return; if(S.steps>=endStep()) seenRun.add(a.rk+(mode==='challenge'?'|'+chN:'')); if(app) render();}
function runStart(fromZero,sp){
  stopAnim(); if(fromZero) S.steps=0; const end=endStep(); if(S.steps>=end){S.steps=end; render(); return;}
  if(A.reduceMotion()){S.steps=end; seenRun.add(runKey(S)+(mode==='challenge'?'|'+chN:'')); render(); return;}
  anim={rk:runKey(S),acc:0,sp:sp||null}; lastTs=0; render(); raf=requestAnimationFrame(tick);}
function tick(ts){
  raf=0; if(!anim||!app) return;
  const dt=lastTs?Math.min(0.1,(ts-lastTs)/1000):0; lastTs=ts;
  anim.acc+=dt*RUN_SPEEDS[anim.sp||runSpeed]; const n=Math.floor(anim.acc);
  if(n>0){anim.acc-=n; S.steps=Math.min(endStep(),S.steps+n); if(S.steps>=endStep()){finishAnim(); return;} render();}
  raf=requestAnimationFrame(tick);}
/* a new setting: play its run from 0 if it has never been watched, else show its end at once */
function showRun(sp){const k=runKey(S); if(seenRun.has(k)){S.steps=MAXS; render();} else runStart(true,sp);}

/* ---------- navigation ---------- */
function enterStep(i){
  stopAnim(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  if(s.view) view=s.view;
  if(s.id==='batch'){triedB.clear(); triedB.add(S.B); warm([[4,200],[2,100],[8,100],[16,100]].map(([B,m])=>[LG(),B,S.tau,S.d,m]));}
  if(s.id==='temp'){triedT.clear(); triedT.add(S.tau); warm([0.05,0.01,1,0.2].map(t=>[LG(),S.B,t,S.d,MAXS]));}
  if(s.id==='encoder'){triedQ.clear(); triedQ.add(S.q);}
  const st=$('#emb-stage',app); if(st) st.scrollTop=0;
  renderRail();}
/* where the kit's reveal will leave the page (its rule: the card's top below the top bar and the dock, or no scroll when the card is
   already in view), so that keepStage can tell the reveal's own scroll from a later one it must not undo */
function revealStep(){const el=$('#emb-rail .step.current',app)||$('#emb-rail',app); if(!el) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), r=el.getBoundingClientRect();
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12;
  const stay=innerWidth<=760?Math.abs(r.top-top)<8:(r.top>=top&&r.top<=innerHeight-160);
  stageGoal=stay?scrollY:Math.min(Math.max(0,scrollY+r.top-top),document.documentElement.scrollHeight-innerHeight);
  A.ui.reveal(el);}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, rc=box.getBoundingClientRect();
  if(rc.bottom<=innerHeight-8) return;
  const dy=Math.min(rc.bottom-innerHeight+16,rc.top-top); if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
const revealObs=()=>requestAnimationFrame(()=>{if(!app) return; if(mode==='guided') revealBox($('#emb-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#emb-insight',app)); else revealBox($('#emb-ch-res',app));});
function setMode(m){
  if(m===mode) return; stopAnim();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; chN=25; S={...S,B:2,tau:0.05,d:16,steps:0}; view='r1'; chWarm();}
  if(m==='sandbox'&&was==='guided'){S={...DEFAULT,q:S.q,steps:seenRun.has(runKey(DEFAULT))?MAXS:0}; view='r1';}
  A.shell.setMode(app,m,DESCS());
  const st=$('#emb-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){const p=PRESETS[i]; stopAnim(); Object.assign(S,p.p); byLearner(()=>showRun('fast')); revealObs();}   // Scenarios: the end within 1.5 s (R3.9), the learner's speed kept
function byLearner(fn){userAct=true; try{fn();} finally{userAct=false;}}
const token=(s=S)=>[s.B,s.tau,s.d,s.steps,s.q].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null; const o={B:+p[0],tau:+p[1],d:+p[2],steps:+p[3],q:+p[4]};
  if(!BS.includes(o.B)||!TAUS.includes(o.tau)||!DS.includes(o.d)||!Number.isInteger(o.steps)||o.steps<0||o.steps>MAXS||!Number.isInteger(o.q)||o.q<0||o.q>=NP) return null; return o;}
function act(a,btn){
  if(a==='play'){if(anim){stopAnim(); render(); return;} if(S.steps>=endStep()){runStart(true); return;} byLearner(()=>runStart(false)); return;}
  if(a==='replay'){runStart(true); return;}
  if(a==='reset'){stopAnim(); byLearner(()=>{Object.assign(S,DEFAULT); render();}); return;}
  if(a==='submit'){chSubmitted=true; const k=runKey(S)+'|'+chN; if(seenRun.has(k)){S.steps=chN; render();} else runStart(true); revealObs(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#emb-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}}
function flashBtn(b,t,back){b.textContent=t; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
function showMe(p){
  if(p.steps!=null){if(S.steps>=MAXS) render(); else runStart(false); revealObs(); return;}
  if(p.q!=null){S.q=p.q; triedQ.add(p.q); byLearner(render); revealObs(); return;}
  if(p.B!=null){S.B=p.B; triedB.add(p.B); byLearner(showRun); revealObs(); return;}
  if(p.tau!=null){S.tau=p.tau; triedT.add(p.tau); byLearner(showRun); revealObs(); return;}
  if(p.trap){trap=p.trap; renderRail(); revealObs();}}

/* ---------- interview drill (seeded): one row of the loss ---------- */
function makeDrill(seed){const rnd=mulberry(seed*7919+13), tau=[0.1,0.05][Math.floor(rnd()*2)], s=[];
  for(let j=0;j<4;j++) s.push(Math.round((rnd()*1.1-0.2)*100)/100); if(s.slice(1).some(v=>v>=s[0])) s[0]=Math.round((Math.max(...s.slice(1))+0.05+rnd()*0.2)*100)/100;
  const z=s.map(v=>Math.exp(v/tau)), Z=z.reduce((a,b)=>a+b,0); return {seed,tau,s,p:z[0]/Z,loss:-Math.log(z[0]/Z),z,Z};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), q=$('#emb-drill-q',app), a=$('#emb-drill-a',app);
  q.innerHTML=tx(`A batch of 4 pairs. The first question’s cosines with the 4 passages are <b>[${d.s.map(v=>fx(v,2)).join(', ')}]</b>; the first passage is its own. With τ = ${d.tau}, what probability does the softmax give its own passage, and what is that row’s loss?`,
                 `一个 batch 有 4 对。第一个问题和 4 段的余弦是 <b>[${d.s.map(v=>fx(v,2)).join(', ')}]</b>，第一段是它自己的。τ = ${d.tau} 时，softmax 给它自己那段多少概率？这一行的 loss 是多少？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const eqs=`\\frac{e^{${fx(d.s[0],2)}/${d.tau}}}{${d.s.map(v=>`e^{${fx(v,2)}/${d.tau}}`).join(' + ')}}`;
  a.innerHTML=tx(`<p><b>probability ${pc(d.p,1)}, loss ${f(d.loss,3)}</b></p>${M(`p = ${eqs} = ${f(d.p,4)}`)}${M(`\\ell = -\\log ${f(d.p,4)} = ${f(d.loss,3)}`)}<p>Dividing by τ multiplies every gap: a cosine lead of only ${f(d.s[0]-Math.max(...d.s.slice(1)),2)} becomes ${f((d.s[0]-Math.max(...d.s.slice(1)))/d.tau,1)} in the exponent. Training lowers this loss by raising the first cosine and lowering the other three.</p>`,
                 `<p><b>概率 ${pc(d.p,1)}，loss ${f(d.loss,3)}</b></p>${M(`p = ${eqs} = ${f(d.p,4)}`)}${M(`\\ell = -\\log ${f(d.p,4)} = ${f(d.loss,3)}`)}<p>除以 τ 会把每个差距放大：余弦只领先 ${f(d.s[0]-Math.max(...d.s.slice(1)),2)}，到指数里就是 ${f((d.s[0]-Math.max(...d.s.slice(1)))/d.tau,1)}。训练降低这个 loss 的办法，是推高第一个余弦，压低另外三个。</p>`);}

/* ---------- jumps ---------- */
const CTLN={B:{en:'Pairs per batch, B',zh:'每个 batch 的配对数 B'},tau:{en:'Temperature τ',zh:'温度 τ'},d:{en:'Dimensions per vector, d',zh:'每个向量的维数 d'},steps:{en:'Training steps',zh:'训练步数'}};
function jumpAct(k){
  if(k==='pair') return tx('Click to show the keyword search.','点击查看关键词搜索。');
  if(k==='heat') return tx('Click to show the batch’s similarity table.','点击查看 batch 的相似度表。');
  if(k==='chart') return tx('Click to show the chart.','点击查看曲线图。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(!CTLN[k]) return '';
  if(mode==='challenge'&&k!=='B') return tx(`Fixed during the challenge: “${L(CTLN[k])}”.`,`挑战中固定不变：“${L(CTLN[k])}”。`);
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);}
function jump(k,src){
  if(!app) return;
  const pan=(id,mark)=>{const el=$('#emb-pan-'+id,app); if(el&&el.hidden){el.hidden=false; render();} A.tip.hide(); const m=mark&&$(mark,app);
    if(el){(m||el).scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); if(m) A.ui.flash(m);}};
  if(k==='pair'){pan('pair','#emb-pair'); return;}
  if(k==='heat'){pan('heat','#emb-heat'); return;}
  if(k==='chart'){pan('chart','#emb-chart'); return;}
  if(k==='controls'){const c=$('#emb-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(!CTLN[k]) return;
  if(mode==='challenge'&&k!=='B'){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=$('#emb-ic-'+k,app)||$('#emb-ctl-'+k,app); if(el) A.ui.jumpToEl(el);}
let stageT=0, stageGoal=null;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop(); stageGoal=null;
  let lastY=null, waited=0;
  const tickS=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tickS,120); return;}
    if(stageGoal!=null&&Math.abs(scrollY-stageGoal)>4) return;   // the page was scrolled after the part's reveal: leave it where it is
    const st=$('#emb-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#emb-rail .step.current',app)||$('#emb-rail .railcard',app)||$('#emb-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, rc=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=rc.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tickS,A.reduceMotion()?30:150);}

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='steps'){stopAnim(); byLearner(()=>setState({steps:+e.target.value}));}});
  el.addEventListener('change',e=>{if(e.target.id==='emb-qsel'){S.q=+e.target.value; triedQ.add(S.q); byLearner(render);}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#emb-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key, inRail=!!seg.closest('#emb-rail');
      if(k==='view'){view=d.v; render(); return;}
      if(k==='speed'){runSpeed=d.v; render(); return;}
      if(k==='q'){S.q=+d.v; triedQ.add(S.q); byLearner(render); if(inRail) revealObs(); return;}
      if(k==='chN'){stopAnim(); chN=+d.v; if(!chReveal){chSubmitted=false; S.steps=0;} else S.steps=chN; render(); return;}
      stopAnim(); const v=+d.v;
      if(mode==='challenge'){S.B=v; if(!chReveal){chSubmitted=false; S.steps=0;} else S.steps=chN; render(); return;}
      if(k==='B') triedB.add(v); if(k==='tau') triedT.add(v);
      byLearner(()=>{S[k]=v; if(mode==='guided'&&(STEPS[step].id==='batch'||STEPS[step].id==='temp')) showRun(); else setState({});});
      if(inRail) revealObs(); return;}
    if(d.lever!=null&&b.closest('#emb-rail')){trap=d.lever; renderRail(); revealObs(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#emb-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#emb-rail',app)); return;}
    if(d.act) act(d.act,b);});}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12 07:13 UTC): four sections read top to bottom before the guided parts, text
   and formulas only. Why search has to go by meaning (semantic search); what a text embedding model is (one vector for a whole text,
   search = the largest cosine); why it is trained on pairs (contrastive learning, the InfoNCE loss, in-batch negatives; B and τ named
   but not what they change, and not what training does to the batch table, which the part "先预测" asks); and RAG. How strong each
   effect is stays with the guided parts, so no number of theirs is here, and the example is not the lab's help center
   (A.registerPrimer, CONTRACT §17). After the first-time reader's check (2026-10-12) there is no figure: the batch table was the
   one "先预测" shows
   ===================================================================== */
// a paragraph per item; a formula block (a <div>) stands on its own line
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return /^<div/.test(h)?h:`<p>${h}</p>`;}).join('')});
const PFX=t=>`<div class="emb-math-line emb-prfx">${A.ui.math(t)}</div>`;
const prLK=id=>{const a=LL(id); return tx(a,`“${a}”`);};
const Q_=()=>tx('\\text{question}','\\text{问题}'), P_=()=>tx('\\text{passage}','\\text{段落}');
const EX_Q=()=>tx('My laptop screen cracked. Who can repair that?','笔记本屏幕碎了找谁修？');
A.registerPrimer(ID,{min:7,sections:[
 {h:()=>tx('Search has to go by meaning (semantic search)','搜索要按意思找（语义搜索）'),parts:[prP(
   ()=>tx('A company’s internal documents, product manuals and policies add up to thousands of passages of text. A user asks a question, and search has to find the passages that answer it. The most common way is keyword search: count the words the question shares with each passage, and rank a passage higher the more words it shares.',
          '一家公司的内部文档、产品手册、规章制度，加起来有成千上万段文字。用户问一个问题，搜索要从中找出能回答它的那几段。最常见的做法是关键词搜索：数一数问题和每一段有几个共同的词，共同的词越多，排得越靠前。'),
   ()=>tx(`But the person asking does not know how the documents are worded. An employee asks “${EX_Q()}”, and the line of the company policy that answers it is “To report damaged equipment, file a ticket with the IT desk.” The two share no word, so keyword search gives that line a score of 0, lower than any text that mentions “laptop” or “screen”. The other way round, shared words do not mean the same subject: “How do I make my laptop screen brighter?” shares three words with the question, my, laptop and screen, and asks about something else.`,
          `可提问的人并不知道文档是怎么写的。员工问“${EX_Q()}”，公司制度里能回答它的那一句是“设备损坏请提交 IT 工单。”两句话没有一个共同的词，关键词搜索给这一句的分数是 0，比任何一段提到“笔记本”或“屏幕”的文字都低。反过来，共同的词多，也不等于说的是同一件事：“笔记本屏幕怎么调亮？”和这个问题有两个共同的词，笔记本和屏幕，问的却是另一件事。`),
   ()=>tx('A list of synonyms fills some gaps, but one meaning can be put in too many ways to list them all. What is needed is search by meaning: whatever words they use, does the passage talk about the same thing as the question? This is semantic search. It needs a way of representing text in which “close in meaning” becomes a closeness that can be computed.',
          '同义词表能补上一些，可同一个意思的说法太多，列不完。要的是按意思找：不管用了哪些词，只看问题和段落说的是不是同一件事。这叫语义搜索（semantic search）。为此需要一种表示文字的办法，让“意思相近”变成一个能算出来的“接近”。'))]},
 {h:()=>tx('Text embedding models: texts close in meaning get close vectors','文本 embedding 模型：意思近的文字，向量也近'),parts:[prP(
   ()=>tx(`The first thing a language model does with text is to take, for every token (a piece of text from the model’s vocabulary), a vector from its embedding table (see ${prLK('architecture')}). Semantic search compares whole texts, so it needs one vector for a whole text. A text embedding model (below, simply an embedding model) reads a whole text, a question or a passage of a document, and outputs one vector that stands for the meaning of the whole text. However long the text, the vector has the same number of dimensions, usually a few hundred to a few thousand.`,
          `语言模型处理文字的第一步，是给每个 token（模型词表里的一个文字片段）从 embedding 表里取出一个向量（见${prLK('architecture')}）。语义搜索要比的是整段文字，要的是给整段文字一个向量。文本 embedding 模型（下面简称 embedding 模型）读完一整段文字，一个问题或者一段文档，输出一个向量，代表整段的意思。不论文字多长，这个向量的维数都一样，常见的是几百到几千维。`),
   ()=>tx(`Write this model as a function ${MI('f_\\theta')} with parameters ${MI('\\theta')}. We want one property from it: the closer two texts are in meaning, the larger the cosine of their vectors. With that property, “find the passage closest in meaning” becomes “find the vector with the largest cosine”:`,
          `把这个模型记作函数 ${MI('f_\\theta')}，${MI('\\theta')} 是它的参数。我们要它满足一条性质：两段文字意思越近，两个向量的余弦越大。有了这条性质，“找意思最近的段落”就变成了“找余弦最大的向量”：`),
   ()=>PFX(`j^{*} = \\arg\\max_{j}\\ \\cos\\big(f_\\theta(${Q_()}),\\, f_\\theta(${P_()}_j)\\big)`),
   ()=>tx(`In practice search usually returns the few passages with the largest cosines, not just ${MI('j^{*}')}. This also keeps the work at question time small: a text’s vector depends only on that text, so every passage’s vector can be computed ahead of time and stored. When a user asks, the model only computes the question’s vector, which is then compared with the stored ones by cosine.`,
          `实际检索一般返回余弦最大的前几段，而不只 ${MI('j^{*}')} 这一段。这样做，提问时要算的也不多：一段文字的向量只由这段文字决定，所以每段文档的向量可以事先算好存起来。用户提问时，只要用模型算出问题的向量，再拿它和存好的向量逐个比余弦。`),
   ()=>tx(`What is left is how ${MI('f_\\theta')} gets that property. ${MI('\\theta')} is set by training: training defines a loss that measures how wrong the model is, and lowers it by gradient descent (see ${prLK('backprop')}). But what counts as wrong here?`,
          `剩下的问题是 ${MI('f_\\theta')} 怎样才有这条性质。${MI('\\theta')} 要靠训练定下来：训练先定一个 loss，衡量模型错了多少，再用梯度下降把它降下去（见${prLK('backprop')}）。可在这里，什么算错？`))]},
 {h:()=>tx('Contrastive learning (InfoNCE): pairs as the training signal','对比学习（InfoNCE）：用配对当训练信号'),parts:[prP(
   ()=>tx('In many training tasks every input comes with a right answer, and the loss measures how far the output is from it. An embedding model has no such answer. Search uses only the cosines between vectors, and applying the same rotation to every vector leaves every cosine the same, so nobody can say what a vector should be by itself; what means something is only which vectors are close to which. What can be had is pairs: this question is answered by this passage. Such pairs are easy to collect, for example a question and its answer on a Q&A site, an article’s title and its body, or a search query and the result the user opened.',
          '很多训练任务里，每个输入都配一个标准答案，loss 量的是输出离答案有多远。embedding 模型没有这样的答案。搜索只用到向量之间的余弦，而把所有向量做同一个旋转，余弦都不变，所以一个向量本身该是多少，谁也说不出；有意义的只是谁离谁近。能拿到的是配对：这个问题由这一段回答。这样的配对有现成的来源，比如问答网站上的问题和回答、文章的标题和正文、搜索记录里的查询和用户点开的结果。'),
   ()=>tx(`A pair says exactly what search needs: a question’s own passage should be closer to it than the other passages. So training becomes a multiple-choice question: question ${MI('i')} gets ${MI('B')} candidate passages, numbered 1 to ${MI('B')}, of which only passage ${MI('i')} is its pair, and the model has to pick it out by cosine. Write ${MI('s_{ij}')} for the cosine between the vectors of question ${MI('i')} and passage ${MI('j')}. Each of the question’s ${MI('B')} cosines is divided by the temperature ${MI('\\tau')}, a positive number chosen before training; a softmax turns them into probabilities (see ${prLK('architecture')} for the softmax); and the loss is minus the log of passage ${MI('i')}’s probability, the cross-entropy:`,
          `配对说的正好是搜索要的：一个问题自己的那段，要比别的段落离它更近。于是把训练写成一道选择题：给问题 ${MI('i')} 准备 ${MI('B')} 个候选段落，编号 1 到 ${MI('B')}，其中只有段落 ${MI('i')} 和它配对，让模型凭余弦把它挑出来。记问题 ${MI('i')} 和段落 ${MI('j')} 的向量的余弦为 ${MI('s_{ij}')}。先把问题 ${MI('i')} 的这 ${MI('B')} 个余弦都除以温度 ${MI('\\tau')}（训练前选定的一个正数），再用 softmax 变成概率（softmax 见${prLK('architecture')}），loss 取段落 ${MI('i')} 那一项概率的负对数，叫 cross-entropy（交叉熵）：`),
   ()=>PFX('\\ell_i = -\\log \\frac{e^{s_{ii}/\\tau}}{\\sum_{j=1}^{B} e^{s_{ij}/\\tau}}'),
   ()=>tx(`Dividing the numerator and the denominator by ${MI('e^{s_{ii}/\\tau}')} shows that ${MI('\\ell_i')} depends only on how far ${MI('s_{ii}')} is above the question’s other cosines:`,
          `分子分母同除以 ${MI('e^{s_{ii}/\\tau}')}，可以看出 ${MI('\\ell_i')} 只取决于 ${MI('s_{ii}')} 比其余的余弦高出多少：`),
   ()=>PFX('\\ell_i = \\log\\Big(1 + \\sum_{j \\ne i} e^{(s_{ij} - s_{ii})/\\tau}\\Big)'),
   ()=>tx(`The larger each gap ${MI('s_{ii} - s_{ij}')}, the smaller ${MI('\\ell_i')}; ${MI('\\ell_i')} looks only at these gaps, never at what the vectors themselves are. Training with such a loss is contrastive learning, and this loss is called InfoNCE.`,
          `每个差 ${MI('s_{ii} - s_{ij}')} 越大，${MI('\\ell_i')} 越小；${MI('\\ell_i')} 只看这些差，不看向量本身是多少。用这样的 loss 训练，叫对比学习（contrastive learning），这个 loss 名叫 InfoNCE。`),
   ()=>tx(`Where do the candidates come from? Each training step takes a small set of ${MI('B')} pairs, a batch. For question ${MI('i')}, its paired passage ${MI('i')} is called the positive. Each of the other ${MI('B-1')} passages in the batch is paired with some other question, and for question ${MI('i')} they are negatives, its wrong choices; taking the negatives from the batch itself is called in-batch negatives. The cosines of a batch form a ${MI('B \\times B')} table, and ${MI('\\ell_i')} uses its row ${MI('i')}. The loss of a step is the average of these ${MI('B')} losses, and gradient descent updates ${MI('\\theta')} by it:`,
          `候选段落从哪来？训练每一步取一小批配对，共 ${MI('B')} 对，叫一个 batch。对问题 ${MI('i')} 来说，和它配对的段落 ${MI('i')} 叫正例；batch 里其余 ${MI('B-1')} 段各自配着别的问题，对问题 ${MI('i')} 就是负例，也就是错误选项。负例直接从同一个 batch 里取，这种做法叫 batch 内负例。一个 batch 的余弦排成一张 ${MI('B \\times B')} 的表，${MI('\\ell_i')} 用的是它的第 ${MI('i')} 行。一步的 loss 是这 ${MI('B')} 个 ${MI('\\ell_i')} 的平均，梯度下降按它更新 ${MI('\\theta')}：`),
   ()=>PFX('L(\\theta) = \\frac{1}{B} \\sum_{i=1}^{B} \\ell_i'))]},
 {h:()=>tx('Retrieve first, then let the model answer from it (RAG)','先检索，再让模型照着回答（RAG）'),parts:[prP(
   ()=>tx(`Semantic search returns a few passages, but the user wants an answer. Give the question straight to a large model and it will probably get it wrong. The employee asks “${EX_Q()}”, but the model knows only what was in its training data: your company’s policies were not there, and it knows nothing of what changed after its training data was collected. Asked anyway, it often makes up an answer that sounds plausible.`,
          `语义搜索给出的是几段文档，用户要的是一个回答。把问题直接交给大模型，它多半答不对。员工问“${EX_Q()}”，可模型知道的只是训练数据里有的东西：你公司的制度不在其中，训练数据收集以后的变化它也不知道。被问到时，它常常编出一个听上去合理的回答。`),
   ()=>tx(`Putting the whole document store into the prompt (all of the model’s input; see ${prLK('prefill-decode')}) does not work either: a model takes only so many tokens at once, and every token of the prompt has to be computed, so the more goes in, the slower and costlier every answer.`,
          `把整个文档库都放进 prompt（给模型的全部输入，见${prLK('prefill-decode')}）也不行：模型一次能处理的 token 数有上限，而且 prompt 的每个 token 都要计算，放得越多，每次回答越慢、越贵。`),
   ()=>tx('Retrieval-augmented generation (RAG) retrieves first and generates second. Ahead of time, the documents are cut into passages, and the embedding model computes every passage’s vector and stores it. When a question comes, it is turned into a vector and the few passages with the largest cosines are taken; they go into the prompt with the question and an instruction to answer from these passages, and to say so when the answer is not in them. The model does not have to remember your documents, only to write from the material in front of it. When a document changes, only the changed passages’ vectors are computed again; the model is not retrained.',
          '检索增强生成（RAG，retrieval-augmented generation）的做法是先检索，再生成。事先把文档切成段落，用 embedding 模型算出每段的向量存好。提问时，先把问题变成向量，取余弦最大的前几段，把它们和问题一起放进 prompt，并告诉模型根据这些资料回答，资料里没有就直说。模型不必记住你的文档，只要照着眼前的资料写。文档改了，只要重算改动那几段的向量，模型不用重新训练。'),
   ()=>tx('Making this work in a real system leaves three hard parts: with millions of passages, computing the cosine with every one of them is too slow; how the documents are cut into passages and how the retrieved ones are ordered decide what the model gets to see; and even when retrieval and generation both run as designed, the answer can still be wrong.',
          '要在真实系统里用好这条流程，还有三个难题：段落有上百万时，和每一段都算一遍余弦太慢；文档怎样切成段落、取回的几段怎样排序，决定了模型能看到什么；而就算检索和生成都照常运行，回答也仍然可能是错的。'))]}
]});

A.registerLab({
  id:ID,
  /* the learner's state for a reload (round 3 #6): settings, part, prediction, the view, the pair picked, what was tried, the runs
     watched, the challenge picks and result. A run still playing is saved as finished (its end is what the learner is about to see) */
  snap:()=>{const sk=anim?anim.rk+(mode==='challenge'?'|'+chN:''):null, done=!!anim||S.steps>=chN;
    return {v:1,mode,step,S:{...S,steps:anim?endStep():S.steps},pred,observed:{...observed},saved:saved?{...saved}:null,chN,chSubmitted,chReveal,view,trap,
      triedB:[...triedB],triedT:[...triedT],triedQ:[...triedQ],seenRun:[...new Set([...seenRun,...(sk?[sk]:[])])].slice(-100),runSpeed,drillSeed,drillOpen,
      passed:mode==='challenge'&&chSubmitted&&done&&chPoint(LG(),S.B,chN).ok};},
  restore(o){if(!o||o.v!==1) return; const ps=s=>s&&typeof s==='object'?parse(token(s)):null, p=ps(o.S); if(!p) return;
    const bool=v=>v===true, ints=(v,a,b)=>Number.isInteger(v)&&v>=a&&v<=b, arr=(v,ok)=>Array.isArray(v)?v.filter(ok):[];
    S=p; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=ints(o.step,0,STEPS.length-1)?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=ps(o.saved); saved=mode==='challenge'&&sv?sv:null;
    chN=CH_STEPS.includes(o.chN)?o.chN:25; chSubmitted=bool(o.chSubmitted); chReveal=chSubmitted&&bool(o.chReveal);
    view=VIEWS.some(x=>x[0]===o.view)?o.view:'r1'; trap=TRAPS.some(x=>x.k===o.trap)?o.trap:null;
    triedB.clear(); arr(o.triedB,v=>BS.includes(v)).forEach(v=>triedB.add(v)); triedT.clear(); arr(o.triedT,v=>TAUS.includes(v)).forEach(v=>triedT.add(v));
    triedQ.clear(); arr(o.triedQ,v=>ints(v,0,NP-1)).forEach(v=>triedQ.add(v));
    seenRun.clear(); arr(o.seenRun,k=>typeof k==='string'&&k.length<60&&/^(en|zh)(\|[0-9.]+){3,4}$/.test(k)).forEach(k=>seenRun.add(k));
    if(Object.keys(RUN_SPEEDS).includes(o.runSpeed)) runSpeed=o.runSpeed; drillSeed=ints(o.drillSeed,1,1e6)?o.drillSeed:1; drillOpen=bool(o.drillOpen);},
  mount(el,tok){
    stopAnim(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#emb-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#emb-rail',app)); keepStage();},token});
    bind(app); renderDrill(); if(mode==='challenge') chWarm();
    if(mode==='guided'){const s=STEPS[step]; if(s.view) view=s.view;}
    renderRail();},
  leave(){stopAnim(); stageStop(); warmStop(); clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const dm=$('#emb-math',app), open=dm&&dm.open; const a=anim; stopAnim(); this.mount(root,null); if(open) $('#emb-math',app).open=true; if(a) runStart(false);}},
  onResize(){if(app) render();},
  termTip(){return '';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&($('#emb-pan-map:not([hidden])',app)||$('#emb-pan-heat:not([hidden])',app)||$('#emb-pan-chart:not([hidden])',app)||$('#emb-pan-pair',app)), bench:()=>app&&$('.bench',app), target:()=>app&&$('#emb-stage',app),
    render({status,vis,read}){if(!app) return; const r=compute(S), c=ctx();
      vis.innerHTML=`<div class="emb-dk"><i style="width:${(r.r1*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const done=chDone(), p=done?chPoint(LG(),S.B,chN):null; status.className='status'+(done?(p.ok?' ok':' bad'):''); status.textContent=done?(p.ok?tx('Goal met','已达标'):tx('Not yet','未达标')):(chSubmitted?tx('Training','训练中'):tx('Not submitted','未提交'));
        read.textContent=done?tx(`Batch ${S.B} for ${chN} steps: average recall@1 ${pc(p.r1)} over 16 starts`,`batch ${S.B}、训练 ${chN} 步：16 个种子的平均 recall@1 是 ${pc(p.r1)}`):tx(`Batch ${S.B} for ${chN} steps`,`batch ${S.B}、训练 ${chN} 步`); return;}
      if(c.panels[0]==='pair'){status.className='status'; status.textContent=tx('Keyword search','关键词搜索'); read.textContent=tx('The right passage shares no word with the question','正确段落和问题没有共同的字对'); return;}
      status.className='status'; status.textContent=tx(`Step ${r.t}`,`第 ${r.t} 步`);
      read.textContent=tx(`After ${stepsTxt(r.t)}, recall@1 on 24 held-out questions is ${pc(r.r1)}`,`训练 ${r.t} 步后，24 个新问题的 recall@1 是 ${pc(r.r1)}`);}}
});
/* read-only hook for tests/embeddings_numbers.py and the home tile */
window.__emb={state:()=>({mode,step,B:S.B,tau:S.tau,d:S.d,steps:S.steps,q:S.q,pred,view,trap,chN,chSubmitted,chReveal,lang:LG()}),animState:()=>anim?{steps:S.steps}:null,
  seen:()=>[...seenRun],
  run:(lang,B,tau,d)=>{const T=run(lang,B,tau,d); return {r1:T.r1,r5:T.r5,loss:T.loss,rank:T.rank,batch:T.batch,S:T.S.map(a=>Array.from(a)),P:[0,MAXS].map(s=>Array.from(T.P[s])),H:[0,MAXS].map(s=>Array.from(T.H[s])),xy:[0,50,MAXS].map(s=>T.pca.xy[s]),R:T.pca.R};},
  runRaw:(lang,B,tau,d)=>run(lang,B,tau,d),
  tokens,data:()=>DATA,vocab:lang=>[...built(lang).vocab.keys()],cosOf:(lang,a,b)=>cosOf(run(lang,8,0.05,16),a,b),kwRank,chPoint,drill:makeDrill,avg:(lang,B,tau,d,maxs)=>avgCurve(lang,B,tau,d,maxs),
  consts:{SEED,MAXS,LR,BS,TAUS,DS,CH_STEPS,CH_GOAL,NP,TRAPS,NSEED,SEEDS}};
})();
