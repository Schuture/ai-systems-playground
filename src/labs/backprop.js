/* backprop (ML basics, world roots): How does autograd get every gradient in one backward pass?
   Fidelity: Toy run. A scalar autograd engine in the style of micrograd runs in the page, in float64:
   - forward: each operation computes its value from its inputs; backward: ∂L/∂L = 1, then every operation in reverse topological
     order (depth-first from the loss, inputs in their written order, as micrograd's build_topo) adds local derivative × its own
     gradient into each input's gradient. Local derivatives read the values at backward time, so a value changed in place is seen,
     as with PyTorch's saved tensors.
   - bugs as switches: "=" instead of "+=" (a value used twice keeps only the last contribution), tanh′ written as 1 − z², h clipped in
     place to 0.5 between forward and backward (small engine: read silently; PyTorch: the version check stops the backward at the first
     step that reads h), and the challenge's four hidden bugs.
   - gradient check: central difference (L(θ + h) − L(θ − h)) ÷ 2h and the relative error |a − n| ÷ max(|a| + |n|, 1e-12).
   Words (EN / ZH): gradient / 梯度; loss; local derivative / 局部导数; backward pass / 反向传播（一趟反向）; part / 部分; step / 步 (a backward
   step processes one operation). window.__bp is a read-only hook for tests/backprop_numbers.py and tests/backprop_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='backprop';
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);

/* =====================================================================
   engine
   ===================================================================== */
const NETS={
 two:{id:'two',x1:1.5,x2:-0.6,t:0.5,P:{w1:0.6,w2:0.5,b:0.1,v:1.5},params:['w1','w2','b','v']},
 tied:{id:'tied',x1:1.5,x2:-0.6,t:0.5,P:{w:0.6,b:0.1,v:1.5},params:['w','b','v']},
 ch:{id:'ch',x1:2.0,x2:-0.5,t:1.0,P:{w:0.5,b:-0.2,v:1.5,c:0.3},params:['w','b','v','c']}
};
const CLIP=0.5;                       // part 6: h.clamp_(max=0.5) after the forward
const HS=[0.5,0.1,0.001,0.00001];     // the nudges a learner can pick
const PASS=1e-7, SUSPECT=1e-4;        // relative error: below PASS passes, above SUSPECT points to a bug
function mk(id,op,ins,o){return {id,op,ins:ins||[],data:0,grad:0,c:o&&o.c,t:o&&o.t};}
function fval(n){const a=n.ins.map(x=>x.data);
  switch(n.op){case 'mulc': return a[0]*n.c; case 'add': return a.reduce((s,x)=>s+x,0); case 'tanh': return Math.tanh(a[0]);
    case 'sig': return 1/(1+Math.exp(-a[0])); case 'mul': return a[0]*a[1]; case 'aff': return a[0]*a[1]+a[2]; case 'sq': return (a[0]-n.t)**2;}
  return n.data;}
function build(net,P){
  const X=NETS[net], N={}, leaf=id=>{N[id]=mk(id,'param'); N[id].data=P[id]; return N[id];}, op=(id,o,ins,ex)=>{N[id]=mk(id,o,ins,ex); N[id].data=fval(N[id]); return N[id];};
  if(net==='two'){leaf('w1'); leaf('w2'); leaf('b'); leaf('v'); op('m1','mulc',[N.w1],{c:X.x1}); op('m2','mulc',[N.w2],{c:X.x2});}
  else {leaf('w'); leaf('b'); leaf('v'); if(net==='ch') leaf('c'); op('m1','mulc',[N.w],{c:X.x1}); op('m2','mulc',[N.w],{c:X.x2});}
  op('z','add',[N.m1,N.m2,N.b]);
  if(net==='ch'){op('s','sig',[N.z]); op('y','aff',[N.s,N.v,N.c]);}
  else {op('h','tanh',[N.z]); op('y','mul',[N.h,N.v]);}
  op('L','sq',[N.y],{t:X.t});
  return N;
}
function topo(N){const out=[], seen=new Set(); const go=n=>{if(seen.has(n.id)) return; seen.add(n.id); n.ins.forEach(go); out.push(n);}; go(N.L); return out;}
/* local derivatives, read at backward time; f holds the bugs */
function locals(n,f){const a=n.ins;
  switch(n.op){case 'mulc': return [n.c]; case 'add': return a.map(()=>1);
    case 'tanh': return [f.tz?1-a[0].data**2:1-n.data**2];
    case 'sig': return [f.sg?1-n.data**2:n.data*(1-n.data)];
    case 'mul': return f.sw?[a[0].data,a[1].data]:[a[1].data,a[0].data];
    case 'aff': return f.sw?[a[0].data,a[1].data,1]:[a[1].data,a[0].data,1];
    case 'sq': return [f.half?(a[0].data-n.t):2*(a[0].data-n.t)];}
  return [];}
const RC=new Map();
/* the whole run: forward, an optional in-place change, then the backward pass one operation per step */
function run(s){
  const f={over:!!s.over,tz:!!s.tz,sg:!!s.sg,sw:!!s.sw,half:!!s.half}, inpl=s.inpl||'none';
  const key=[s.net,+f.over,+f.tz,+f.sg,+f.sw,+f.half,inpl].join('|'); if(RC.has(key)) return RC.get(key);
  const X=NETS[s.net], P={...X.P}, N=build(s.net,P), ord=topo(N), ops=ord.filter(n=>n.op!=='param').reverse();
  const vals={}; ord.forEach(n=>{vals[n.id]=n.data;});
  let changed=null;
  if(inpl!=='none'&&N.h){changed={id:'h',from:N.h.data,to:Math.min(N.h.data,CLIP)}; N.h.data=changed.to;}
  N.L.grad=1;
  /* a value's gradient is complete once every operation that uses it has run (w waits for both of its uses) */
  const left={}; ops.forEach(n=>n.ins.forEach(x=>{left[x.id]=(left[x.id]||0)+1;}));
  const snap=()=>{const g={}; ord.forEach(n=>{g[n.id]=n.grad;}); return g;}, got=new Set(['L']), done=()=>ord.filter(n=>!left[n.id]&&got.has(n.id)).map(n=>n.id);
  const steps=[{id:null,grads:snap(),got:[...got],done:done()}]; let error=null;
  for(const n of ops){
    if(inpl==='pt'&&changed&&n.ins.some(x=>x.id==='h')){error={at:n.id,step:steps.length}; steps.push({id:n.id,err:true,up:n.grad,contribs:[],grads:snap(),got:[...got],done:done()}); break;}
    const lc=locals(n,f), g=n.grad;
    const contribs=n.ins.map((x,i)=>{const c=lc[i]*g, before=x.grad; x.grad=f.over?c:x.grad+c; got.add(x.id); left[x.id]--; return {to:x.id,local:lc[i],up:g,c,before,after:x.grad};});
    steps.push({id:n.id,up:g,contribs,grads:snap(),got:[...got],done:done()});
  }
  const fin=steps[steps.length-1].grads;
  const r={s:{net:s.net,...f,inpl},X,P,ord:ord.map(n=>n.id),ops:ops.map(n=>n.id),nt:ops.length,L:vals.L,vals,changed,steps,error,n:steps.length-1,grads:fin,params:X.params};
  RC.set(key,r); return r;
}
function lossAt(net,P){return build(net,P).L.data;}
function fdiff(net,p,h){const P=NETS[net].P; return (lossAt(net,{...P,[p]:P[p]+h})-lossAt(net,{...P,[p]:P[p]-h}))/(2*h);}
const relErr=(a,n)=>Math.abs(a-n)/Math.max(Math.abs(a)+Math.abs(n),1e-12);
function check(r,p,h){const gn=fdiff(r.s.net,p,h), ga=r.error?null:r.grads[p], e=ga==null?null:relErr(ga,gn);
  return {p,h,ga,gn,err:e,pass:e!=null&&e<PASS,bad:e!=null&&e>SUSPECT};}
/* the loss as one parameter moves, for the check panel's curve */
function curve(net,p,half,n){const P=NETS[net].P, out=[]; for(let i=0;i<=n;i++){const th=P[p]-half+2*half*i/n; out.push([th,lossAt(net,{...P,[p]:th})]);} return out;}
/* the challenge: four candidate bugs; which parameters fail the check under each */
const CHB=[{id:'A',f:{over:true}},{id:'B',f:{sg:true}},{id:'C',f:{sw:true}},{id:'D',f:{half:true}}];
const chRun=i=>run({net:'ch',...CHB[i].f});
const failsOf=i=>{const r=chRun(i); return NETS.ch.params.filter(p=>!check(r,p,1e-5).pass);};
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* each block of four seeds hides the four bugs in a seeded order, so “Another hidden bug” reaches every bug within four tries */
function hiddenOf(seed){const sd=Math.max(1,Math.floor(seed)), blk=Math.floor((sd-1)/4), rnd=mulberry(blk*7919+17), o=[0,1,2,3];
  for(let i=3;i>0;i--){const j=Math.floor(rnd()*(i+1)); const t=o[i]; o[i]=o[j]; o[j]=t;} return o[(sd-1)%4];}
/* the bugs that agree with every check in the set (each check: does this parameter fail?) */
function consistent(hid,set){const fh=failsOf(hid); return CHB.map((_,j)=>j).filter(j=>{const fj=failsOf(j); return set.every(p=>fj.includes(p)===fh.includes(p));});}
/* the fewest checks that single out the hidden bug, and one such set */
function minChecks(hid){const ps=NETS.ch.params; let best=null;
  for(let m=0;m<(1<<ps.length);m++){const set=ps.filter((_,i)=>m>>i&1); if(consistent(hid,set).length===1&&(!best||set.length<best.length)) best=set;}
  return best;}

/* =====================================================================
   formatting
   ===================================================================== */
const MINUS='−';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sg=(v,d)=>{const s=fixd(Math.abs(v),d); return (v<0&&+s!==0?MINUS:'')+s;};
const f4=v=>sg(v,4);
const f6=v=>{if(v===0) return '0'; const d=Math.max(0,5-Math.floor(Math.log10(Math.abs(v)))); return sg(v,Math.min(d,8));};   // 6 significant digits
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const sup=e=>String(e).split('').map(c=>SUP[c]||c).join('');
/* a relative error: two decimals down to 0.01, then a power of ten */
function fE(x){if(x==null||!isFinite(x)) return '—'; if(x>=0.0995) return fixd(x,2); if(x>=0.0095) return fixd(x,Math.min(4,1-Math.floor(Math.log10(x)))); if(x===0) return '0'; let e=Math.floor(Math.log10(x)), m=x/10**e, ms=fixd(m,1); if(ms==='10.0'){ms='1.0'; e+=1;} return `${ms} × 10${sup(e)}`;}
const fH=h=>h>=0.001?String(h):(Math.abs(h-0.00002)<1e-12?'0.00002':'0.00001');
const SUB={w1:'w₁',w2:'w₂',w:'w',b:'b',v:'v',c:'c',m1:'m₁',m2:'m₂',z:'z',h:'h',s:'s',y:'y',L:'L'};
const nm=id=>SUB[id]||id;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
/* H1 (the owner's decision of 2026-10-09): every formula typeset with the kit's TeX subset, never a plain-text string */
const M=t=>A.ui.math(t);
const dLd=v=>`\\frac{∂L}{∂${v}}`;
const FD_TEX='\\frac{L(\\theta + \\varepsilon) - L(\\theta - \\varepsilon)}{2\\varepsilon}', REL_TEX='\\frac{|a - n|}{|a| + |n|}';
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const big=n=>A.ui.big?A.ui.big(n):A.fmt.big(n);
const pl=(n,one,many)=>n===1?one:many;

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
const BASE={net:'two',over:false,tz:false,inpl:'none',p:'w1',hi:3};
let S={...BASE}, K=6, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let seenEnd=false, hTried=new Set(), fanTried=new Set(['add']), inTried=new Set(['none']), remView='walk';
/* R3.20: the backward passes the reader has stepped or played to the end, by the setup they draw (not by part or mode). Part 3's pass
   watched to the end elsewhere (Scenarios, “What to remember”) opens finished in part 3 and counts for its Next */
const watched=new Set();
const sigOf=s=>[s.net,s.over?1:0,s.tz?1:0,s.inpl,s.p,s.hi].join('_');
let playing=false, speed='med', playT=0, playAt=0;
let chSeed=1, chChecks=[], chPick=null, chSubmitted=false, chReveal=false, saved=null, sbTouched=false;
let drillSeed=1, drillOpen=false, userAct=false, wantObs=false, annT=0;
const shownPrev={};
const R=()=>mode==='challenge'?chRun(hiddenOf(chSeed)):run(S);
const kNow=()=>Math.min(K,R().n);
const SPEEDS={slow:5000,med:3500,fast:1200};   // ms per backward step: a whole pass of 6 steps takes 30, 21 or 7 s (R3.9)

/* =====================================================================
   definitions shown on hover (R2.8)
   ===================================================================== */
const TIP={
 loss:()=>tx(`The loss is one number that says how wrong the output is. Here ${M('L = (y - t)^2')}, the squared gap between the output y and the target t, so it is 0 only when y hits t.`,
             `loss 是一个数值，衡量输出错得有多远。这里 ${M('L = (y - t)^2')}，是输出 y 和目标 t 之差的平方，只有 y 正好等于 t 时它才是 0。`),
 grad:()=>tx(`The gradient of the loss with respect to a parameter, written ${M(dLd('w'))}, is how much the loss changes per unit change of that parameter when only it is nudged by a tiny amount. Training moves each parameter a little against its gradient.`,
             `梯度 ${M(dLd('w'))} 表示：只把这个参数推动一点点时，loss 每单位变化多少。训练就是把每个参数朝梯度的反方向挪一小步。`),
 param:()=>kt('param')+tx(' This small network has 4: the weights w₁ and w₂, the bias b and the output weight v.',' 这个小网络有 4 个参数：权重 w₁、w₂，偏置 b，输出权重 v。'),
 local:()=>tx(`The local derivative of an operation is how fast its output changes when one input changes and the others stay fixed. For ${M('m = w \\cdot x')} it is x. For ${M('h = tanh(z)')} it is ${M('1 - h^2')}. For ${M('L = (y - t)^2')} it is ${M('2(y - t)')}. It needs only the operation’s own inputs and output.`,
              `一个运算的局部导数，是其他输入不变、只改变某一个输入时，它的输出变化得有多快。${M('m = w \\cdot x')} 对 w 的局部导数是 x。${M('h = tanh(z)')} 的局部导数是 ${M('1 - h^2')}。${M('L = (y - t)^2')} 的局部导数是 ${M('2(y - t)')}。它只用到这个运算自己的输入和输出。`),
 pass:()=>tx('A backward pass is a walk through the operations the forward pass recorded, from the loss back to the parameters, computing gradients on the way.',
             '反向传播就是从 loss 出发，沿着前向记录下来的运算一路走回参数，边走边算梯度。'),
 gcheck:()=>tx(`A gradient check compares autograd’s gradient with the finite difference, a slope measured from the loss itself (a central difference):<div class="fx">${M(FD_TEX)}</div>θ is the parameter being checked and ε a small nudge. The check trusts only the forward pass, so it catches mistakes in the backward.`,
               `梯度检查拿 autograd 的梯度和有限差分比较。有限差分是直接从 loss 量出来的斜率（中心差分）：<div class="fx">${M(FD_TEX)}</div>θ 是被检查的参数，ε 是一个很小的推动量。它只信任前向计算，所以能抓出反向里的错。`),
 rel:()=>tx(`The relative error compares the two gradients:<div class="fx">${M(REL_TEX)}</div>a is autograd’s gradient and n the finite difference. This lab passes a check below 10⁻⁷ and fails it above 10⁻⁴, the usual line for suspecting a bug. In between, the check does not pass yet and needs a closer look.`,
            `相对误差比较两个梯度差了多少：<div class="fx">${M(REL_TEX)}</div>a 是 autograd 的梯度，n 是有限差分。这个 Lab 里低于 10⁻⁷ 算通过，高于 10⁻⁴ 算不通过，这是怀疑有 bug 的常用界线。介于两者之间时还不算通过，要再仔细看。`),
 inplace:()=>tx('An in-place operation writes its result into the memory of an existing value instead of making a new one, as PyTorch’s methods ending in an underscore do (add_, clamp_, mul_).',
                '原地（in-place）运算把结果写回已有值的内存，而不是另建一个新值；PyTorch 里名字以下划线结尾的方法（add_、clamp_、mul_）都是这样。'),
 shared:()=>tx('Sharing a parameter means using the same number in several places. A convolution applies one filter at every position, and a tied LM head reuses the embedding table, so their gradients collect a contribution from every use.',
               '共享参数就是同一个参数在几处用到。卷积在每个位置用同一个滤波器，绑定的 LM head 复用 embedding 表，所以它们的梯度要从每一处用到它的地方各收一份贡献。'),
 toy:()=>tx('Every value and gradient on this page is computed by a small autograd engine running here, in 64-bit floating point. The network is tiny on purpose: real models do the same per element of their tensors.',
            '页面上的每个值和梯度，都由这里运行的一个小 autograd 引擎用 64 位浮点数算出。网络故意做得很小：真实模型对张量的每个元素做的是同样的事。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'one',t:{en:'Once, and w₁, w₂ and b share it',zh:'一次，w₁、w₂ 和 b 共用它'}},
  {v:'few',t:{en:'Twice: once for the two weights, once for b',zh:'两次：两个权重一次，b 一次'}},
  {v:'each',t:{en:'3 times: once for each of w₁, w₂ and b',zh:'3 次：w₁、w₂、b 各一次'}}];
const PRED_RIGHT='one';
const predT=v=>L(PRED.find(o=>o.v===v).t);
const isP=id=>mode==='guided'&&STEPS[step]&&STEPS[step].id===id;
const SIX=id=>STEPS.findIndex(s=>s.id===id);
/* the options of parts 5 and 6, each from the same initial setup (R3.6) */
const FAN=[{v:'add',t:{en:'+= : add each contribution (initial setup)',zh:'+=：把每份贡献加进去（初始设置）'},d:{en:'what micrograd (a 100-line autograd engine in Python) and PyTorch do',zh:'micrograd（一个 100 行的 Python autograd 引擎）和 PyTorch 都这样做'}},
  {v:'over',t:{en:'= : keep only the last one',zh:'=：只留下最后一份'},d:{en:'a one-character bug in a hand-written backward',zh:'手写反向时差一个字符的 bug'}}];
const INP=[{v:'none',t:{en:'No in-place change (initial setup)',zh:'没有原地修改（初始设置）'},d:{en:'h keeps the value the forward pass computed',zh:'h 保持前向算出的值'}},
  {v:'eng',t:{en:'Clip h in place, in this lab’s small engine',zh:'原地截断 h，在这个 Lab 的小引擎里'},d:{en:'the backward reads whatever h holds now',zh:'反向读的是 h 现在的值'}},
  {v:'pt',t:{en:'Clip h in place, in PyTorch',zh:'原地截断 h，在 PyTorch 里'},d:{en:'autograd checks each saved value’s version',zh:'autograd 会核对每个保存值的版本'}}];
const fanRes=v=>{const r=run({net:'tied',over:v==='over'}), c=check(r,'w',1e-5); return c.pass?tx('check passes','检查通过'):tx('check fails','检查不通过');};
const inRes=v=>{const r=run({net:'tied',inpl:v}); if(r.error) return tx('backward stops','反向报错'); const n=r.params.filter(p=>!check(r,p,1e-5).pass).length; return n?tx(n===3?'all 3 checks fail':`${n} of 3 checks fail`,n===3?'3 个检查全部不通过':`3 个检查有 ${n} 个不通过`):tx('all 3 checks pass','3 个检查都通过');};
const lvHTML=(opts,val,tried,res,aria)=>`<div class="bp-lv">`+A.ui.levers({aria,value:val,options:opts.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:esc(tried.has(o.v)||o.v===val?res(o.v):'')}))})+`</div>`;
const pCtl=(id,net,inline)=>A.ui.seg({id,key:'p',label:tx('Parameter to check','要检查的参数'),options:NETS[net].params.map(p=>[p,nm(p)]),value:S.p,inline});
const hCtl=(id,inline,note)=>A.ui.seg({id,key:'h',label:tx('Nudge ε','推动量 ε'),options:HS.map((h,i)=>[String(i),fH(h)]),value:String(S.hi),inline,note});
const playerHTML=(id,inline)=>`<div class="bp-player${inline?' bp-inline':''}" id="${id}" data-player>`+
  `<button type="button" class="btn ghost sm bp-pb" data-play="back" aria-label="${esc(tx('One backward step back','后退一步'))}" data-tipx="${esc(tx('Undo the last backward step.','撤回最后一个反向步骤。'))}">◂</button>`+
  `<button type="button" class="btn sm bp-pb" data-play="fwd" aria-label="${esc(tx('Next backward step','下一个反向步骤'))}" data-tipx="${esc(tx('Run the next backward step: one operation passes its gradient to its inputs.','走下一个反向步骤：一个运算把梯度传给它的输入。'))}">▸</button>`+
  `<button type="button" class="btn ghost sm bp-pb bp-playbtn" data-play="play" data-tipx="${esc(tx('Play the rest of the backward pass at the chosen speed; press again to pause. After the last step, the button plays the pass again from the first step.','按选定的速度把剩下的反向步骤放完；再按一次暂停。放完最后一步后，这个按钮从第一个反向步骤重新播放。'))}"></button>`+
  `<span class="bp-scrub"><input type="range" min="0" max="6" step="1" value="0" data-key="k" aria-label="${esc(tx('Backward step','反向步骤'))}"><output class="mono"></output></span>`+
  (inline?'':`<span class="seg-ctl bp-speed" role="group" aria-label="${esc(tx('Speed','速度'))}" data-key="speed">${[['slow',tx('slow','慢')],['med',tx('medium','中')],['fast',tx('fast','快')]].map(([v,t])=>`<button type="button" data-v="${v}" aria-pressed="${v===speed}" data-tipx="${esc(tx(`About ${SPEEDS[v]/1000} seconds per backward step.`,`每个反向步骤约 ${SPEEDS[v]/1000} 秒。`))}">${t}</button>`).join('')}</span>`)+`</div>`;
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},
  lead:()=>tx('A small network, 4 parameters, one loss: how much does each parameter move the loss?','一个小网络，4 个参数，一个 loss：每个参数让 loss 变多少？'),
  enter:()=>{K=0; return {...BASE};},
  body:()=>tx(`<p>To train a network you need to know, for each parameter, which way to move it and by how much. That is its ${DEF('gradient',TIP.grad())}: how much the loss changes when only that parameter is nudged a little.</p><p>This small network has 4 ${DEF('parameters',TIP.param())}. It turns two inputs, x₁ and x₂, into an output y, and a ${DEF('loss',TIP.loss())}, ${M('L = (y - t)^2')}, scores how far y is from the target t. The ${J('graph','computation graph')} shows every operation with the value the forward pass computed.</p><p>In PyTorch one line, <code>loss.backward()</code>, fills in all 4 gradients. How does it do that?</p>`,
              `<p>训练一个网络，要知道每个参数该往哪边调、调多少。这就是参数的${DEF('梯度',TIP.grad())}：只把这一个参数推动一点点，loss 会变多少。</p><p>这里的小网络有 4 个${DEF('参数',TIP.param())}。它把两个输入 x₁、x₂ 变成输出 y，再用 ${DEF('loss',TIP.loss())} ${M('L = (y - t)^2')} 衡量 y 离目标 t 有多远。${J('graph','计算图')}画出了每一步运算，以及前向算出的值。</p><p>在 PyTorch 里，一行 <code>loss.backward()</code> 就把 4 个梯度都算好了。它是怎么做到的？</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},gate:()=>!!pred,skip:true,
  lead:()=>tx('Guess first: the gradients of w₁, w₂ and b all pass through z. How many times is z’s gradient computed?','先猜：w₁、w₂ 和 b 的梯度都要经过 z。z 的梯度要算几次？'),
  enter:()=>{K=0; return {...BASE};},
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>A ${DEF('backward pass',TIP.pass())} starts at the loss and walks the graph back to the parameters. The three parameters w₁, w₂ and b all reach the loss through ${M('z = m_1 + m_2 + b')}. To get all three of their gradients, how many times is z’s gradient computed?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>${DEF('反向',TIP.pass())}从 loss 出发，沿着图一路走回参数。w₁、w₂ 和 b 这三个参数，都要经过 ${M('z = m_1 + m_2 + b')} 才连到 loss。要把这三个参数的梯度都算出来，z 的梯度要算几次？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'walk',title:{en:'Walk the backward pass',zh:'走一遍反向'},
  lead:()=>tx('Press ▸: each step multiplies one operation’s gradient by a local derivative and passes it on.','按 ▸：每一步把一个运算的梯度乘上局部导数，再往回传。'),
  enter:()=>{if(watched.has(sigOf(BASE))) seenEnd=true; K=seenEnd?6:0; return {...BASE};},expect:()=>seenEnd,showMe:()=>setK(R().n),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The backward pass starts at the loss itself: the loss’s gradient with respect to itself is 1, ${M(dLd('L')+' = 1')}. Then it climbs the graph from the bottom, one operation per step. Each step multiplies the gradient of the operation’s output by the operation’s ${DEF('local derivative',TIP.local())}, and adds the product into the gradient of each input.</p><p>Take it one step at a time:</p>`,
              `<p>反向从 loss 本身开始：loss 对它自己的梯度是 1，也就是 ${M(dLd('L')+' = 1')}。之后沿着图从下往上走，每一步处理一个运算。这一步把运算输出的梯度乘上它的${DEF('局部导数',TIP.local())}，再加到它每个输入的梯度上。</p><p>一步一步走：</p>`)+playerHTML('bp-ic-play',true),
  obs:()=>obsWalk()},
 {id:'check',title:{en:'Check a gradient by nudging',zh:'推一下参数，核对梯度'},
  lead:()=>tx('Nudge one parameter up and down by ε, and check autograd’s gradient against the change in the loss.','把一个参数上下推动 ε，用 loss 的变化核对 autograd 的梯度。'),
  enter:()=>{K=6; hTried=new Set([0]); return {...BASE,hi:0};},expect:()=>hTried.has(3),showMe:{hi:3},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>How do you know the backward pass is right? A ${DEF('gradient check',TIP.gcheck())} uses only the forward pass: it moves the checked parameter θ up and down by ${T_('h','ε')} and computes the loss each time. The difference of the two losses divided by 2ε is the finite difference:</p><p class="bp-ml"><span class="bp-mname">finite difference</span> ${M('= '+FD_TEX)}</p><p>It should match autograd’s gradient. In the ${J('check','check panel')}, the solid line’s slope is autograd’s gradient and the dashed line’s slope is the finite difference.</p>`,
              `<p>怎么知道反向算得对不对？${DEF('梯度检查',TIP.gcheck())}不用反向，只用前向：把被检查的参数 θ 往上、往下各推 ${T_('h','ε')}，各算一次 loss。两次 loss 之差除以 2ε，叫有限差分：</p><p class="bp-ml"><span class="bp-mname">有限差分</span> ${M('= '+FD_TEX)}</p><p>它应该和 autograd 的梯度一样。${J('check','检查面板')}里，实线的斜率是 autograd 的梯度，虚线的斜率是有限差分。</p>`)+pCtl('bp-ic-p','two',true)+hCtl('bp-ic-h',true),
  obs:()=>obsCheck()},
 {id:'fan',title:{en:'A weight used twice',zh:'一个权重用了两次'},
  lead:()=>tx('Now one weight w is used twice, so its gradient is the sum of two parts.','现在一个权重 w 用了两次，它的梯度由两份加起来。'),
  enter:()=>{K=6; fanTried=new Set(['add']); return {...BASE,net:'tied',p:'w'};},expect:()=>fanTried.has('over'),showMe:()=>setFan('over'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Now both inputs ${DEF('share',TIP.shared())} one weight w, so two arrows leave w in the graph. In the backward pass w gets one contribution through each use. The engine adds them with <code>+=</code>; writing <code>=</code> instead is a common bug.</p><p>Pick a way to write it and see whether w’s gradient passes its check.</p>`,
              `<p>现在两个输入${DEF('共用',TIP.shared())}同一个权重 w，所以图里从 w 出发有两条箭头。反向时，w 从这两处各收到一份梯度。引擎用 <code>+=</code> 把两份加起来。写成 <code>=</code>，就是一个常见的 bug。</p><p>选一种写法，看看 w 的梯度能不能通过检查。</p>`)+lvHTML(FAN,S.over?'over':'add',fanTried,fanRes,tx('How w’s gradient is written, pick one','w 的梯度怎么写，选一个')),
  obs:()=>obsFan()},
 {id:'inplace',title:{en:'A saved value changed in place',zh:'保存的值被原地改了'},
  lead:()=>tx('The backward pass reads values the forward pass saved: change one, and the gradients go wrong.','反向要读前向存下的值；改了其中一个，梯度就会出错。'),
  enter:()=>{K=6; inTried=new Set(['none']); return {...BASE,net:'tied',p:'v'};},expect:()=>inTried.size>1,showMe:()=>setIn('eng'),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>The backward pass uses values from the forward pass. tanh’s local derivative is ${M('1 - h^2')}, which needs h, and ${M('y = h \\cdot v')}’s local derivative for v is h itself. So the forward pass keeps h for the backward.</p><p>Suppose a line after the forward and before <code>backward()</code> changes h ${DEF('in place',TIP.inplace())}, clipping it to at most ${CLIP}, as <code>h.clamp_(max=${CLIP})</code> does. What happens?</p><p>Pick a case and see whether the 3 gradients still pass their checks.</p>`,
              `<p>反向要用到前向算出的值。tanh 的局部导数是 ${M('1 - h^2')}，要用 h。${M('y = h \\cdot v')} 对 v 的局部导数就是 h。所以前向会把 h 存下来，留给反向用。</p><p>假设前向之后、<code>backward()</code> 之前，有一行代码${DEF('原地',TIP.inplace())}修改了 h，把它截到不超过 ${CLIP}，比如 <code>h.clamp_(max=${CLIP})</code>。会发生什么？</p><p>选一种情况，看看 3 个梯度还能不能通过检查。</p>`)+lvHTML(INP,S.inpl,inTried,inRes,tx('The in-place change, pick one','原地修改，选一个')),
  obs:()=>obsIn()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  lead:()=>tx('One pass, local derivatives multiplied; check with a tiny ε; a value used twice adds both parts; never change a saved value.','一趟反向，局部导数相乘；用很小的 ε 核对；一个值用了两次，梯度就加两份；存下的值不能改。'),
  enter:()=>{K=6; remView='walk'; return {...BASE};},
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”。正确答案是“${rt}”。</p>`);
}
const readyOf=(r,k)=>r.params.filter(p=>r.steps[k].done.includes(p)).length;
const NT=r=>r.nt;   // the backward pass's steps, also when PyTorch stops it early
function obsWalk(){
  const r=R(), k=kNow(), n=r.n, rd=readyOf(r,k);
  if(k<n) return tx(`<p><b>After backward step ${k} of ${n}, ${rd} of the 4 parameters ${pl(rd,'has its gradient','have their gradients')}.</b> Keep pressing ▸ until all four have one.</p>`,
                    `<p><b>走完第 ${k} 步（共 ${n} 步），4 个参数里有 ${rd} 个有了梯度。</b>继续按 ▸，直到 4 个都有。</p>`);
  return tx(`<p><b>One backward pass of ${n} steps filled in all 4 gradients. z’s gradient was computed once, at step 3, and then passed on to w₁, w₂ and b.</b></p>`,`<p><b>一趟反向走了 ${n} 步，4 个梯度全部算好。z 的梯度只在第 3 步算了一次，之后传给了 w₁、w₂ 和 b。</b></p>`)+verdict()+
    tx(`<p>Why: the pass handles the operations in order, each one once. When an operation’s turn comes, every operation that uses its output has already run, so the gradient it holds is complete and can be passed straight on. Every parameter’s gradient falls out on the way.</p><p>Real models are mostly matrix products. For them, one such pass costs about twice the forward pass, however many parameters there are (${J('fid','see the notes at the end of the page')}).</p>`,
       `<p>原因：反向按顺序处理运算，每个运算只处理一次。轮到一个运算时，所有用到它输出的运算都已经处理过，所以它手上的梯度已经完整，可以直接往回传。每个参数的梯度，就在这一路上顺带算出来。</p><p>真实模型的计算主要是矩阵乘法。对它们来说，不管参数有多少，一趟反向的计算量都大约是前向的两倍（${J('fid','见页末的“简化、假设和来源”')}）。</p>`);
}
function obsCheck(){
  const r=R(), c=check(r,S.p,HS[S.hi]), P=nm(S.p), h=fH(HS[S.hi]);
  if(S.hi===3) return tx(`<p><b>With ε = ${h}, the finite difference and autograd agree for ${P}: the relative error is ${fE(c.err)}, far below the 10⁻⁷ that counts as a pass.</b></p><p>Why: the closer the two points θ − ε and θ + ε, the closer the line through them comes to the curve’s tangent. The error shrinks with ε²: each 10 times smaller ε makes it about 100 times smaller.</p><p>Each parameter checked costs two more forward passes. Real models have billions of parameters, so a gradient check is for finding bugs, not a replacement for the backward pass.</p>`,
                         `<p><b>ε = ${h} 时，${P} 的有限差分和 autograd 一致：相对误差是 ${fE(c.err)}，远低于算作通过的 10⁻⁷。</b></p><p>原因：θ − ε 和 θ + ε 两点离得越近，过这两点的直线就越接近曲线的切线。误差和 ε² 成正比：ε 缩小到 1/10，误差约缩小到 1/100。</p><p>每检查一个参数，就要多跑两次前向。真实模型的参数多得多，逐个检查太慢，所以梯度检查只用来查错，不能代替反向。</p>`);
  if(c.pass) return tx(`<p><b>With ε = ${h}, the finite difference for ${P} is ${f4(c.gn)}, the same as autograd’s ${f4(c.ga)}: the check passes even with this large ε.</b> The loss is a parabola in ${P}, and a central difference is exact on a parabola. Pick w₁ to see a curve that bends, then a smaller ε.</p>`,
                       `<p><b>ε = ${h} 时，${P} 的有限差分是 ${f4(c.gn)}，和 autograd 的 ${f4(c.ga)} 一样。ε 这么大，检查也通过了。</b>只看 ${P} 时，loss 是一条抛物线；在抛物线上，中心差分算出的斜率正好等于切线的斜率。选 w₁ 看一条更弯的曲线，再试更小的 ε。</p>`);
  const same=f4(c.gn)===f4(c.ga);
  const head=same?tx(`<p><b>With ε = ${h}, the finite difference for ${P} and autograd’s gradient agree to four decimals (${f4(c.ga)}), but the relative error is ${fE(c.err)}, so the check ${c.bad?'fails':'does not pass yet'} at this ε.</b> `,
                     `<p><b>ε = ${h} 时，${P} 的有限差分和 autograd 的梯度前四位小数相同（${f4(c.ga)}），但相对误差是 ${fE(c.err)}，所以这个 ε 下检查${c.bad?'不通过':'还不算通过'}。</b>`)
    :tx(`<p><b>With ε = ${h}, the finite difference for ${P} is ${f4(c.gn)}, while autograd says ${f4(c.ga)}: the relative error is ${fE(c.err)}, so the check ${c.bad?'fails':'does not pass yet'} at this ε.</b> `,
        `<p><b>ε = ${h} 时，${P} 的有限差分是 ${f4(c.gn)}，autograd 给的是 ${f4(c.ga)}，相对误差 ${fE(c.err)}，所以这个 ε 下检查${c.bad?'不通过':'还不算通过'}。</b>`);
  return head+(S.hi===2?tx('The error is between 10⁻⁷ and 10⁻⁴: the two points are still far enough apart for the bend of the curve to show. Pick a smaller ε.</p>','误差在 10⁻⁷ 和 10⁻⁴ 之间：两点还是离得有点远，曲线的弯曲还看得出来。再选一个更小的 ε。</p>')
    :tx('The cause is the large ε, not the backward pass: autograd’s gradient is right. The two points are far apart, so the line through them misses the bend of the curve. Pick a smaller ε.</p>','问题出在 ε 太大，不在反向，autograd 的梯度是对的。两点离得太远，过这两点的直线跟不上曲线的弯曲。选一个更小的 ε。</p>'));
}
function obsFan(){
  const r=R(), c=check(r,'w',1e-5), st=fanSteps(r);
  if(!S.over) return tx(`<p><b>w’s gradient is the sum of its two contributions, ${f4(st.c2)} through x₂ and ${f4(st.c1)} through x₁: ${f4(st.c2)} + ${f4(st.c1)} = ${f4(st.c2+st.c1)}. The check passes.</b> Pick “=” to see the bug.</p>`,
                        `<p><b>w 的梯度是两份之和：经过 x₂ 的一份是 ${f4(st.c2)}，经过 x₁ 的一份是 ${f4(st.c1)}，${f4(st.c2)} + ${f4(st.c1)} = ${f4(st.c2+st.c1)}。检查通过。</b>选“=”看看这个 bug。</p>`);
  const right=run({net:'tied'}).grads.w;
  return tx(`<p><b>With =, the contribution through x₁ overwrote the one through x₂: w’s gradient is ${f4(r.grads.w)} instead of ${f4(right)}, and the check fails (relative error ${fE(c.err)}).</b></p><p>Why: nudging w moves the loss through both uses at once, so the true slope is the sum over both paths. The other gradients are still right, because only w is used twice: a check of b or v alone would miss this bug.</p>`,
            `<p><b>写成 = 时，经过 x₁ 的一份把经过 x₂ 的一份覆盖了。w 的梯度成了 ${f4(r.grads.w)}，正确值是 ${f4(right)}，检查不通过（相对误差 ${fE(c.err)}）。</b></p><p>原因：推动 w 时，两处用到 w 的地方都会让 loss 变化，所以真正的斜率是两条路径之和。其他梯度都还是对的，因为只有 w 用了两次。如果只检查 b 或 v，就发现不了这个 bug。</p>`);
}
/* the two contributions w receives in the tied network: through x₂ first, then through x₁ */
function fanSteps(r){let c1=0,c2=0; r.steps.forEach(s=>{(s.contribs||[]).forEach(c=>{if(c.to==='w'){if(s.id==='m1') c1=c.c; else if(s.id==='m2') c2=c.c;}});}); return {c1,c2};}
function obsIn(){
  const r=R(), ok=run({net:'tied'});
  if(S.inpl==='none') return tx(`<p><b>No change: h keeps ${f4(ok.vals.h)}, and every gradient matches its check.</b> Pick one of the other two.</p>`,`<p><b>没有修改时，h 还是 ${f4(ok.vals.h)}，每个梯度都通过了检查。</b>再选另外两种情况之一。</p>`);
  if(S.inpl==='pt') return tx(`<p><b>PyTorch stops at y’s backward step with “RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation”.</b></p><p>Why: each tensor carries a version counter, and every in-place operation bumps it. Autograd records the version of each value it saves, and compares it before the backward uses the value.</p>`,
                              `<p><b>PyTorch 走到 y 的反向步骤就停下了，报错“RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation”。</b></p><p>原因：每个张量都带着一个版本号，每做一次原地运算，版本号就加一。autograd 保存一个值时，会记下当时的版本号；反向要用这个值之前，先核对版本号有没有变。</p>`);
  const fails=r.params.filter(p=>!check(r,p,1e-5).pass).map(nm).join(tx(', ','、'));
  return tx(`<p><b>The loss is still ${f4(r.L)}, but the backward read h = ${f4(r.changed.to)} instead of ${f4(r.changed.from)}: the gradients of ${fails} are all wrong, and their checks fail.</b></p><p>Why: the loss was computed before the change, while tanh’s and y’s local derivatives are computed in the backward from the h it finds. Nothing in a small engine notices, so training would quietly follow wrong gradients.</p>`,
            `<p><b>loss 还是 ${f4(r.L)}，但反向读到的 h 是 ${f4(r.changed.to)}，不是 ${f4(r.changed.from)}。${fails} 的梯度都错了，检查全部不通过。</b></p><p>原因：loss 是在修改之前算出来的，而 tanh 和 y 的局部导数是反向时才算的，用的是当时读到的 h。这个小引擎不会发现问题，训练就会悄悄地沿着错的梯度走。</p>`);
}
/* part 7: three lines with an example each, a button puts that part's picture back in the chart (R6.23) */
function remBtn(k){const on=remView===k;
  const tip={walk:tx('Shows the graph after the whole backward pass.','在图里显示走完整趟反向的计算图。'),check:tx('Shows the check of w₁ with ε = 0.5.','在图里显示 ε = 0.5 时对 w₁ 的检查。'),
    fan:tx('Shows the overwrite bug, = in place of +=, and its failing check.','在图里显示把 += 写成 = 的覆盖 bug，和不通过的检查。'),
    inpl:tx('Shows h clipped in place in the small engine, and the failing checks.','在图里显示小引擎里 h 被原地截断后，检查不通过。')}[k];
  return ` <button type="button" class="linkbtn bp-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const two=run({net:'two'}), tied=run({net:'tied'}), over=run({net:'tied',over:true}), big5=check(two,'w1',0.5), st=fanSteps(tied);
  return tx(`<ul class="bp-remlist"><li><b>One backward pass gives every gradient.</b> Each step multiplies an operation’s output gradient by a local derivative, so one walk from the loss back to the parameters fills in all their gradients, however many parameters there are.${remBtn('walk')}</li>`+
    `<li><b>Check a gradient with a central difference and a tiny ε.</b> With a large ε the two points are too far apart for the line to follow the curve, so a correct backward can look wrong.${remBtn('check')}</li>`+
    `<li><b>A value used twice adds both gradients.</b> Writing = instead of += keeps only the last contribution: the gradient is wrong, and nothing reports an error.${remBtn('fan')}</li>`+
    `<li><b>A value the forward pass saved must not change before backward.</b> The backward reads the value as it is now, so changing it in place makes every gradient that uses it wrong. PyTorch checks each saved value’s version and stops with an error instead.${remBtn('inpl')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Backward computes each gradient separately.” It does not: one pass from the loss computes all of them, and a gradient that several parameters share is computed only once.</p><p>✗ “A passing check means the backward is right everywhere.” A check covers only the parameters and the graph you check: when the gradient of a value used twice is written wrong, only that value’s check fails, and the others still pass. It also says nothing about whether training will converge.</p><p>✗ “A gradient flows along one path.” A value used in several places collects the sum of all its paths, one contribution for each use.</p></div>${A.shell.toScenarios()}`,
    `<ul class="bp-remlist"><li><b>一趟反向就能算出所有梯度。</b>每一步都是把梯度乘上一个局部导数，所以从 loss 走回参数一趟，参数再多，梯度也全部算好。${remBtn('walk')}</li>`+
    `<li><b>用中心差分和很小的 ε 核对梯度。</b>ε 太大时，两点离得太远，直线跟不上曲线的弯曲，对的反向也会显得不对。${remBtn('check')}</li>`+
    `<li><b>一个值用了两次，就要把两份梯度加起来。</b>把 += 写成 =，只留下最后一份，梯度就错了，程序却不会报错。${remBtn('fan')}</li>`+
    `<li><b>前向存下的值，在反向之前不能改。</b>反向读的是它现在的值，原地改了它，用到它的梯度就全错了。PyTorch 会核对每个保存值的版本，发现被改过就报错停下。${remBtn('inpl')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “反向要一个一个地算每个梯度。”不是。从 loss 出发走一趟，就全部算出来了，几个参数共用的那段梯度也只算一次。</p><p>✗ “检查通过了，反向就处处都对。”检查只管你查过的那几个参数、这一张图。一个值用了两次、它的梯度却写错了时，只有它自己的检查不通过，别的照样通过。检查通过，也不代表训练一定能收敛。</p><p>✗ “梯度只沿一条路传回来。”一个值在几处用到，它的梯度就是所有路径之和，每用一次加一份。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: real situations
   ===================================================================== */
const SRC={
 micro:'https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py',
 engine:'https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp',
 func:'https://github.com/pytorch/pytorch/blob/main/torch/autograd/function.py',
 saved:'https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/saved_variable.cpp',
 jax:'https://github.com/jax-ml/jax/blob/main/jax/_src/interpreters/ad.py',
 rhw:'https://www.nature.com/articles/323533a0',
 baydin:'https://arxiv.org/abs/1502.05767',
 kaplan:'https://arxiv.org/abs/2001.08361',
 cs231n:'https://cs231n.github.io/neural-networks-3/#gradcheck'
};
const PRESETS=[
 {id:'neuron',t:{en:'Train a single neuron',zh:'训练一个神经元'},d:{en:'The unit every network is built from: two inputs, a weight each, a bias and tanh, plus an output weight v',zh:'所有网络的基本单元：两个输入各乘一个权重，加上偏置，过 tanh，再乘输出权重 v'},s:{net:'two',over:false,tz:false,inpl:'none',p:'w1'}},
 {id:'tied',t:{en:'Reuse one weight in two places',zh:'一个权重在两处复用'},d:{en:'As a convolution reuses its filter, or a tied LM head reuses the embedding table',zh:'就像卷积复用滤波器，或绑定的 LM head 复用 embedding 表'},s:{net:'tied',over:false,tz:false,inpl:'none',p:'w'}},
 {id:'custom',t:{en:'A hand-written backward that overwrites',zh:'手写的反向覆盖了梯度'},d:{en:'From-scratch code, as in micrograd or a NumPy backward, that writes a gradient instead of adding to it',zh:'从零手写的反向（像 micrograd 或用 NumPy 写的反向）把梯度直接写进去，而不是加进去'},s:{net:'tied',over:true,tz:false,inpl:'none',p:'w'}},
 {id:'clip',t:{en:'Clip an activation in place after the forward',zh:'前向之后原地截断一个激活值'},d:{en:'A cleanup line that edits h before backward runs',zh:'一行“整理”代码在反向之前改了 h'},s:{net:'tied',over:false,tz:false,inpl:'eng',p:'v'}}
];
const presetNow=()=>PRESETS.findIndex(p=>p.s.net===S.net&&p.s.over===S.over&&p.s.tz===S.tz&&p.s.inpl===S.inpl&&p.s.p===S.p&&S.hi===3);
function scenHTML(i){
  const p=PRESETS[i], r=run({...p.s}), fails=r.error?[]:r.params.filter(q=>!check(r,q,1e-5).pass);
  const res=r.error?'':(fails.length?tx(fails.length===r.params.length?`All ${r.params.length} gradient checks fail.`:`${fails.length} of ${r.params.length} gradient checks ${pl(fails.length,'fails','fail')}.`,(fails.length===r.params.length?`${r.params.length} 个梯度检查全部不通过。`:`${r.params.length} 个梯度检查里有 ${fails.length} 个不通过。`)):tx(`All ${r.params.length} gradient checks pass.`,`${r.params.length} 个梯度检查全部通过。`));
  if(p.id==='neuron') return tx(`<p><b>A single neuron plus an output weight v: one backward pass of ${r.n} steps gives all 4 gradients. ${res}</b></p><p>Every network, however large, repeats this pattern: a weighted sum, a nonlinearity, then the loss. Press ◂ under the graph and replay the pass to see each local derivative used once.</p>`,
                                 `<p><b>一个神经元，再加一个输出权重 v：一趟反向走 ${r.n} 步，就算出全部 4 个梯度。${res}</b></p><p>再大的网络，也是在重复这个模式：加权求和，经过一个非线性函数，最后算 loss。按图下的 ◂ 退回去再放一遍，可以看到每个局部导数都只用了一次。</p>`);
  if(p.id==='tied'){const st=fanSteps(r); return tx(`<p><b>w is used twice, so its gradient is the sum of two contributions, ${f4(st.c2)} + ${f4(st.c1)} = ${f4(r.grads.w)}. ${res}</b></p><p>A convolution uses each filter weight at every position of the image, and a tied LM head uses the embedding table twice, so frameworks always add into a gradient and never overwrite it.</p>`,
                                                 `<p><b>w 用了两次，所以它的梯度是两份之和：${f4(st.c2)} + ${f4(st.c1)} = ${f4(r.grads.w)}。${res}</b></p><p>这种情况很常见。卷积在图像的每个位置都用同一组滤波器权重；绑定的 LM head 把 embedding 表用了两次。所以框架总是把每一份加进梯度，从不覆盖。</p>`);}
  if(p.id==='custom') return tx(`<p><b>A hand-written backward that writes = instead of += keeps only w’s last contribution: ${f4(r.grads.w)} instead of ${f4(run({net:'tied'}).grads.w)}. ${res}</b></p><p>A backward written from scratch, such as micrograd’s <code>_backward</code> closures or a NumPy backward, must add each use’s contribution into w’s gradient. In PyTorch the engine adds them for you; a <code>torch.autograd.Function</code> can only make this mistake when its own forward uses one input twice. A gradient check on every parameter, including the shared ones, finds it.</p>`,
                                 `<p><b>手写的反向把 += 写成了 =，w 只留下最后一份：${f4(r.grads.w)}，正确值是 ${f4(run({net:'tied'}).grads.w)}。${res}</b></p><p>从零手写反向时（比如 micrograd 的 <code>_backward</code> 闭包，或用 NumPy 写的反向），每一处用到 w 的地方，都要把自己那一份加进 w 的梯度。在 PyTorch 里，引擎会替你相加；只有你自己写的 <code>torch.autograd.Function</code> 在 forward 里把同一个输入用了两次，才可能犯这个错。对每个参数都做一次梯度检查，共享的参数也不例外，就能把它找出来。</p>`);
  return tx(`<p><b>Clipping h in place after the forward leaves the loss unchanged, but the backward reads the clipped h. ${res}</b></p><p>In a small engine the gradients are silently wrong. PyTorch would stop with the in-place error instead (switch it under ${J('controls','Settings')}): fix it by making a new value, <code>h2 = h.clamp(max=0.5)</code>, instead of <code>clamp_</code>.</p>`,
            `<p><b>前向之后原地截断 h：loss 不变，但反向读到的是截断后的 h。${res}</b></p><p>在小引擎里，梯度会悄悄出错。换成 PyTorch，它会直接报原地修改的错（可以在“${J('controls','自由调参')}”里切换）。改法是不用 <code>clamp_</code>，而是生成一个新值：<code>h2 = h.clamp(max=0.5)</code>。</p>`);
}

/* =====================================================================
   challenge
   ===================================================================== */
const CHT=[{en:'w’s two contributions are written with = instead of +=',zh:'w 的两份贡献用 = 写入，而不是 +='},
  {en:'σ’s local derivative is written as 1 − s² instead of s(1 − s)',zh:'σ 的局部导数写成 1 − s²，而不是 s(1 − s)'},
  {en:'In y = v·s + c, v and s get each other’s local derivative',zh:'y = v·s + c 里，v 和 s 的局部导数互换了'},
  {en:'The loss’s local derivative is (y − t), missing the 2',zh:'loss 的局部导数写成 (y − t)，漏了 2'}];
const chHid=()=>hiddenOf(chSeed);
/* until the answer is submitted and shown, the stage keeps the hand-written backward to itself: only the checks speak (R3.8) */
const chHide=()=>mode==='challenge'&&!chReveal;
const chT=i=>L(CHT[i]);
/* the option with its formulas typeset (H1), for HTML: the results and the lever's title (the kit keeps A.ui.math markup there) */
const CHTM=[[],[['1 − s²','1 - s^2'],['s(1 − s)','s(1 - s)']],[['y = v·s + c','y = v \\cdot s + c']],[['(y − t)','(y - t)']]];
const chTH=i=>CHTM[i].reduce((h,[a,b])=>h.replace(a,M(b)),esc(chT(i)));
function chCheckRows(){const r=chRun(chHid());
  return NETS.ch.params.map(p=>{const done=chChecks.includes(p), c=check(r,p,1e-5);
    return `<button type="button" class="bp-chk${done?(c.pass?' bp-ok':' bp-bad'):''}" data-chk="${p}" aria-pressed="${done}"${done?' disabled':''}><b>${tx('Check','检查')} ${nm(p)}</b><span class="mono">${done?(c.pass?tx(`passes, ${fE(c.err)}`,`通过，${fE(c.err)}`):tx(`fails, ${fE(c.err)}`,`不通过，${fE(c.err)}`)):tx('not run yet','还没做')}</span></button>`;}).join('');}
function chResultHTML(){
  if(!chSubmitted) return '';
  const hid=chHid(), cons=consistent(hid,chChecks), mn=minChecks(hid), right=chPick===hid;
  const fl=p=>failsOf(hid).includes(p);
  const said=chChecks.map(p=>`${nm(p)} ${fl(p)?'✗':'✓'}`).join(tx(', ','，'));
  let h;
  if(right&&cons.length===1&&chChecks.length<=mn.length) h=tx(`<p class="okline bp-mt"><b>Right: ${chTH(hid)}. And ${mn.length===1?'one check':`${mn.length} checks`} is the fewest that single it out.</b></p>`,`<p class="okline bp-mt"><b>答对了：${chTH(hid)}。而且你用的 ${mn.length} 个检查，已经是能确定它的最少次数。</b></p>`);
  else if(right&&cons.length===1) h=tx(`<p class="okline bp-mt"><b>Right: ${chTH(hid)}.</b></p><p>But you ran ${chChecks.length} checks, and ${mn.length===1?'one was':`${mn.length} were`} enough: checking ${mn.map(nm).join(' and ')} already singles it out.</p>`,`<p class="okline bp-mt"><b>答对了：${chTH(hid)}。</b></p><p>不过你做了 ${chChecks.length} 个检查，其实 ${mn.length} 个就够：只查 ${mn.map(nm).join(' 和 ')} 就能确定它。</p>`);
  else if(right){const other=cons.filter(j=>j!==hid).map(j=>`“${chTH(j)}”`).join(tx(' or ',' 或 '));
    h=tx(`<p class="okline bp-mt"><b>Right: ${chTH(hid)}.</b></p><p>But your checks (${said||'none'}) also fit ${other}, so they alone do not single it out. Checking ${mn.map(nm).join(' and ')} would tell them apart.</p>`,`<p class="okline bp-mt"><b>答对了：${chTH(hid)}。</b></p><p>不过你的检查（${said||'没有'}）也符合${other}，单凭它们还确定不了。查一下 ${mn.map(nm).join(' 和 ')} 就能分清。</p>`);}
  else h=tx(`<p class="badline bp-mt"><b>Not this one. “${(chPick!=null?chTH(chPick):'')}” would make a different set of checks fail.</b></p><p class="small">Hint: a wrong step passes a wrong gradient up to its inputs, so every gradient computed from it on the way up to the parameters is wrong too, while every gradient between it and the loss stays right. Your checks so far: ${said||'none'}. Which bug fails exactly those? Checking c, the parameter closest to the loss, tells the loss’s own bug apart from the rest.</p>`,
            `<p class="badline bp-mt"><b>不是这个。“${(chPick!=null?chTH(chPick):'')}”会让另一组检查不通过。</b></p><p class="small">提示：某一步错了，它传给输入的梯度就错了；再往上，由它算出来的梯度也都跟着错。这一步和 loss 之间的梯度都不受影响。你目前的检查：${said||'没有'}。哪个 bug 正好让这些检查不通过？先查离 loss 最近的 c，就能把 loss 自己的 bug 和其他 bug 分开。</p>`)
     +(chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  if(right) chReveal=true;
  if(!chReveal) return h+(right?'':'');
  /* the owner's review of batch 8 (2026-10-07): no table of all four candidate bugs. A right answer gets the chart of the hidden bug's
     checks; a wrong one also gets a two-row table, the hidden bug next to the pick */
  return h+chChart(hid)+(right?'':chTable(hid,chPick))+
    tx(`<p class="small">The hand-written backward raises no error, so only a gradient check finds the bug. Every parameter whose gradient has to go through the buggy step fails its check. The parameters closer to the loss, whose gradients do not go through it, pass.</p><p class="small muted">Note: here every failing check has a relative error of 0.14 or more, over a million times the pass line, so ε = 0.00001 separates pass from fail clearly. A bug that is wrong by a tiny amount, or a check at a kink such as ReLU at 0, needs a closer look at the relative error before you trust it.</p>`,
       `<p class="small">手写的反向不会报错，只有梯度检查能发现 bug。梯度要经过写错的那一步才能算出的参数，检查都不通过。离 loss 更近、不经过这一步的参数，检查都通过。</p><p class="small muted">注意：这里每个不通过的检查，相对误差都在 0.14 以上，比通过线 10⁻⁷ 高 6 个数量级以上，所以 ε = 0.00001 能把通过和不通过分得很清楚。错得很少的 bug，或者在 ReLU 的 0 点这种拐点上做检查，要先仔细看相对误差，再下结论。</p>`)+
    `<button type="button" class="btn ghost sm" data-act="another">${tx('Another hidden bug','换一个隐藏的 bug')}</button>`;
}
const logv=e=>Math.max(0,Math.log10(Math.max(e,1e-13))+13);
function chChart(hid){const r=chRun(hid);
  const rows=NETS.ch.params.map(p=>{const c=check(r,p,1e-5); return {label:tx(`${nm(p)}: autograd ${f4(c.ga)}, finite difference ${f4(c.gn)}`,`${nm(p)}：autograd ${f4(c.ga)}，有限差分 ${f4(c.gn)}`),value:logv(c.err),text:fE(c.err),ok:c.pass,me:chChecks.includes(p),tags:chChecks.includes(p)?[{text:tx('you checked','你查过'),me:true}]:[]};});
  return A.ui.cmpChart({title:tx('Relative error of each gradient under the hidden bug, on a log scale (the check passes below 10⁻⁷)','隐藏 bug 下每个梯度的相对误差（对数刻度；低于 10⁻⁷ 算通过）'),
    aria:tx('Relative errors: ','相对误差：')+rows.map(x=>x.label+' '+x.text).join(tx('; ','；')),rows,goal:logv(PASS),goalLabel:tx('pass: below 10⁻⁷','通过：低于 10⁻⁷'),max:13.5,
    okText:tx('✓ the check passes','✓ 检查通过'),noText:tx('the check fails','检查不通过'),goalText:tx('10⁻⁷','10⁻⁷')});}
function chTable(hid,pick){const ps=NETS.ch.params;
  return `<div class="bp-sig"><div class="cmp-title">${tx('Which checks fail under the hidden bug and under your pick','这次藏的 bug 和你选的 bug 各让哪些检查不通过')}</div><table><thead><tr><th>${tx('bug','bug')}</th>${ps.map(p=>`<th>${nm(p)}</th>`).join('')}</tr></thead><tbody>`+
    [hid,pick].map(i=>{const f=failsOf(i); return `<tr${i===hid?' class="bp-on"':''}><td>${chTH(i)}${i===hid?tx(' (hidden here)','（这次藏的）'):tx(' (your pick)','（你选的）')}</td>${ps.map(p=>`<td class="${f.includes(p)?'bp-x':'bp-v'}">${f.includes(p)?'✗':'✓'}</td>`).join('')}</tr>`;}).join('')+`</tbody></table></div>`;}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const DESCS=()=>({
  guided:tx('Seven short parts. Predict first, then walk one backward pass step by step, check a gradient by nudging a parameter, and meet two classic bugs: a weight used twice and a saved value changed in place. Start here if this is new to you.',
            '一共 7 个部分。先预测；再一步一步走一趟反向，推动参数核对梯度；最后看两个经典 bug：一个权重用了两次，一个存下的值被原地改掉。第一次学，从这里开始。'),
  sandbox:tx('Four real situations, from training one neuron to a hand-written backward that overwrites a gradient. Pick one to see its graph, its gradients and their checks.',
             '四个真实情况，从训练一个神经元，到手写反向时覆盖了梯度。选一个，看它的计算图、梯度和检查结果。'),
  challenge:tx('A new network with a sigmoid, whose hand-written backward hides one of four bugs. Run gradient checks on as few parameters as you need, then name the bug.',
               '一个带 sigmoid 的新网络，手写的反向里藏着四个 bug 之一。用尽量少的梯度检查，找出是哪一个。')
});
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> ${esc(TIP.toy())} ${J('math','“Show the math”')} works every gradient out line by line.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>The forward pass records each operation and its inputs. The backward pass seeds ${M(dLd('L')+' = 1')} and visits the operations in reverse topological order (depth-first from the loss, as micrograd does); each adds its local derivative times its own gradient into its inputs’ gradients with +=.</li>`+
 `<li>Local derivatives read the values at backward time, as saved tensors do, so a value changed in place is seen. PyTorch’s version check is modeled as: the backward stops at the first step that reads a changed saved value.</li>`+
 `<li>The gradient check: a central difference with the chosen ε, and the relative error ${M(REL_TEX)}.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>Scalars, not tensors: a matrix multiplication’s backward is two matrix products, but each element follows the same chain rule. Inputs and the target are constants, so they get no gradient.</li>`+
 `<li>“About twice the forward pass” holds for matrix multiplications, which dominate a model’s compute: training costs about 6 operations per parameter per token, 2 forward and 4 backward (Kaplan et al., 2020). In this tiny graph the counts are smaller.</li>`+
 `<li>This lab passes a check below 10⁻⁷ and fails it above 10⁻⁴, the usual line for suspecting a bug; the band in between counts as not a pass yet. These thresholds are a rule of thumb for float64; with float32 or at a kink the errors are larger.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>The engine’s shape: micrograd’s ${cite(SRC.micro,'<code>engine.py</code>')}. The in-place error and the version check: PyTorch ${cite(SRC.saved,'<code>saved_variable.cpp</code>')}.</li>`+
 `<li>Backpropagation: ${cite(SRC.rhw,'Rumelhart, Hinton and Williams, 1986')}. Reverse-mode autodiff: ${cite(SRC.baydin,'Baydin et al., 2018')}. Compute per token: ${cite(SRC.kaplan,'Kaplan et al., 2020')}. The relative error and the thresholds: the practice problem mlp-backprop (Go further, under the challenge); the ${cite(SRC.cs231n,'CS231n notes')} give similar thresholds with max(|a|, |n|) as the denominator.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-10-01. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>${esc(TIP.toy())}${J('math','“展开公式”')}里有每个梯度的逐行算式。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>前向记录每个运算和它的输入。反向先设 ${M(dLd('L')+' = 1')}，再按逆拓扑序（从 loss 深度优先，和 micrograd 一样）经过每个运算；每个运算把局部导数乘以自己的梯度，用 += 加进输入的梯度。</li>`+
 `<li>局部导数在反向时读取当时的值，和保存的张量一样，所以原地修改会被读到。PyTorch 的版本核对按这样建模：反向在第一个读到被改过的保存值的步骤处停下。</li>`+
 `<li>梯度检查：用所选 ε 做中心差分，相对误差为 ${M(REL_TEX)}。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>标量，不是张量：矩阵乘法的反向是两个矩阵乘法，但每个元素遵循同样的链式法则。输入和目标是常数，所以没有梯度。</li>`+
 `<li>“大约是前向的两倍”对矩阵乘法成立，而矩阵乘法占了模型计算的大头：训练时每个参数每个 token 大约 6 次运算，前向 2 次，反向 4 次（Kaplan 等，2020）。这个小图里的次数更少。</li>`+
 `<li>这个 Lab 里低于 10⁻⁷ 算通过，高于 10⁻⁴ 算不通过（这是怀疑有 bug 的常用界线），介于两者之间算还不通过。这些阈值是 float64 下常用的经验值；float32 或在拐点处误差会更大。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>引擎的结构：micrograd 的 ${cite(SRC.micro,'<code>engine.py</code>')}。原地修改的报错和版本核对：PyTorch 的 ${cite(SRC.saved,'<code>saved_variable.cpp</code>')}。</li>`+
 `<li>反向传播：${cite(SRC.rhw,'Rumelhart、Hinton 和 Williams，1986')}。反向模式自动微分：${cite(SRC.baydin,'Baydin 等，2018')}。每个 token 的计算量：${cite(SRC.kaplan,'Kaplan 等，2020')}。相对误差和阈值：练习题 mlp-backprop（挑战题下面的“延伸练习”）；${cite(SRC.cs231n,'CS231n 讲义')}给出相近的阈值，分母用 max(|a|, |n|)。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-10-01。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A derivative says how fast an output changes when its input changes: for ${M('y = x^2')}, ${M('\\frac{\\mathrm{d}y}{\\mathrm{d}x} = 2x')}. The chain rule composes them. If y depends on h and h depends on z, then:</p><div class="fx">${M('\\frac{\\mathrm{d}y}{\\mathrm{d}z} = \\frac{\\mathrm{d}y}{\\mathrm{d}h} \\cdot \\frac{\\mathrm{d}h}{\\mathrm{d}z}')}</div><p>The lab ${labLink('tensors')} comes before this one; here every value is a single number.</p>`,
                    `<p>导数说明输入变化时输出变得有多快：${M('y = x^2')} 时，${M('\\frac{\\mathrm{d}y}{\\mathrm{d}x} = 2x')}。链式法则把导数串起来。y 依赖 h，h 依赖 z，那么：</p><div class="fx">${M('\\frac{\\mathrm{d}y}{\\mathrm{d}z} = \\frac{\\mathrm{d}y}{\\mathrm{d}h} \\cdot \\frac{\\mathrm{d}h}{\\mathrm{d}z}')}</div><p>${labLink('tensors')}这个 Lab 在本 Lab 之前；这里的每个值都只是一个标量。</p>`);
const OUTCOME=()=>tx('trace a backward pass by hand, check a gradient with finite differences, and find the bug in a hand-written backward.','手算一遍反向传播，用有限差分核对梯度，并找出手写反向里的 bug。');
function cubeFD(h){return ((1+h)**3-(1-h)**3)/(2*h);}
const EXITS=()=>({
 core:()=>chSubmitted, explain:[
  {q:tx('Why does one backward pass give the gradient of every parameter?','为什么一趟反向就能给出每个参数的梯度？'),
   o:[tx('The pass starts at the loss and visits each operation once in reverse; each multiplies its output’s gradient by its local derivative, so every parameter’s gradient is complete when the pass reaches it.','这一趟从 loss 出发，倒着把每个运算经过一次；每个运算把自己输出的梯度乘以局部导数，所以走到某个参数时，它的梯度已经完整了。'),
      tx('Autograd runs a separate small backward pass for each parameter at the same time on the GPU, so all of them finish together and look like a single pass.','autograd 在 GPU 上为每个参数同时各跑一趟小的反向，它们一起结束，看起来就像一趟。'),
      tx('The forward pass already stores each parameter’s gradient next to its value, and the backward pass only collects them in reverse order and sums them up.','前向时就把每个参数的梯度存在它的值旁边，反向只是倒着把它们收集起来再加总。')],a:0,
   n:[null,tx('There is only one pass: each operation is visited once, and its result is shared by every parameter whose path to the loss goes through it. So a gradient that several parameters share is computed once, not once for each of them.','只有一趟：每个运算只经过一次，凡是要经过它才连到 loss 的参数，都共用它的结果。所以几个参数共用的梯度只算一次，而不是每个参数各算一次。'),
     tx('The forward pass computes only values. A gradient needs the gradient of the operation’s output, which exists only after the loss is known, so it is computed in the backward pass.','前向只算值。梯度需要运算输出的梯度，而那要等 loss 算出来以后才有，所以梯度是在反向里算的。')],
   why:tx('Each step uses only its own local derivative and its output’s gradient, and the reverse order guarantees that the output’s gradient is complete. For the matrix products of real models the whole pass costs about twice the forward pass.','每一步只用到两样东西：自己的局部导数，和输出的梯度。倒着走的顺序，保证了输出的梯度已经完整。真实模型主要是矩阵乘法，一整趟反向的计算量大约是前向的两倍。')},
  {q:tx('When one weight w is used in two places, why must its gradient add the two contributions?','一个权重 w 在两处用到时，为什么它的梯度要把两份贡献加起来？'),
   o:[tx('Nudging w moves the loss through both of its uses at once, so the total change of the loss is the sum of the change through each path.','推动 w 时，两处用到 w 的地方都会让 loss 变化，所以 loss 的总变化是两条路径上的变化之和。'),
      tx('The second contribution is a corrected version of the first, so only the last one written is accurate and the earlier one should be replaced.','第二份贡献是第一份的修正版，所以最后写进去的才准确，前一份应该被替换掉。'),
      tx('Both uses give the same contribution, so adding them only doubles the gradient, which keeps its direction and only speeds up training.','两处用法给出的贡献相同，所以相加只是把梯度翻倍，方向不变，只是让训练更快。')],a:0,
   n:[null,tx('Neither contribution corrects the other: each measures one path. Overwriting drops one of the paths, and the gradient check fails.','两份贡献谁也不修正谁：每一份衡量一条路径。覆盖掉其中一份，就丢了一条路径，梯度检查也就不通过了。'),
     tx('The two contributions usually differ, even in sign, because each is multiplied by its own input.','两份贡献一般不一样，连正负都可能相反，因为各自乘的是自己的输入。')],
   why:tx(`This is the multivariable chain rule: when L depends on w through several paths, ${M(dLd('w'))} is the sum of the path derivatives. That is why frameworks add with += into every gradient.`,`这就是多元链式法则：L 通过几条路径依赖 w，${M(dLd('w'))} 就等于各条路径的导数之和。所以框架对每个梯度都用 += 累加。`)}],
 transfer:[
  {q:tx('Three later operations use a hidden value h, and each sends a gradient back to h: 0.2, −0.5 and 0.1. What is h’s gradient?','后面有三个运算用到中间值 h，它们各给 h 传回一份梯度：0.2、−0.5 和 0.1。h 的梯度是多少？'),
   o:[tx('−0.2 (the sum)','−0.2（总和）'),tx('0.1 (the last one)','0.1（最后一份）'),tx('About −0.067 (the mean)','约 −0.067（平均）'),tx('0.8 (the sum of their sizes)','0.8（大小之和）')],a:0,
   why:tx(`Contributions add with their signs: 0.2 + (−0.5) + 0.1 = ${sg(0.2-0.5+0.1,1)}. Keeping only the last one is the overwrite bug: = in place of +=.`,`各份贡献带着符号相加：0.2 + (−0.5) + 0.1 = ${sg(0.2-0.5+0.1,1)}。只留最后一份，就是把 += 写成 = 的覆盖 bug。`)},
  {q:tx(`For L = θ³ at θ = 1, the exact slope is 3, and a central difference with ε = 0.1 gives ${fixd(cubeFD(0.1),2)}. With ε = 0.01, about what does it give?`,`L = θ³ 在 θ = 1 处的准确斜率是 3，ε = 0.1 的中心差分给出 ${fixd(cubeFD(0.1),2)}。ε = 0.01 时大约给出多少？`),
   o:[tx('3.01 (no better)','3.01（没有变好）'),tx('3.001 (error 10 times smaller)','3.001（误差缩小到 1/10）'),tx('3.0001 (error 100 times smaller)','3.0001（误差缩小到 1/100）'),tx('3.1 (error 10 times larger)','3.1（误差变成 10 倍）')],a:2,
   why:tx(`A central difference is off by about ε² times a constant. Here it is exactly:</p><p class="bp-ml">${A.ui.math('\\frac{(1 + \\varepsilon)^3 − (1 − \\varepsilon)^3}{2\\varepsilon} = 3 + \\varepsilon^2')}</p><p>So ε = 0.1 gives 3.01 and ε = 0.01 gives ${fixd(cubeFD(0.01),4)}: 10 times smaller ε, 100 times smaller error.`,`中心差分的误差约为 ε² 乘一个常数。这里正好是：</p><p class="bp-ml">${A.ui.math('\\frac{(1 + \\varepsilon)^3 − (1 − \\varepsilon)^3}{2\\varepsilon} = 3 + \\varepsilon^2')}</p><p>所以 ε = 0.1 给出 3.01，ε = 0.01 给出 ${fixd(cubeFD(0.01),4)}。ε 缩小到 1/10，误差就缩小到 1/100。`)}],
 practice:[
  {id:'mlp-backprop',d:tx(`Parts 1 and 2: derive every gradient of a two-layer network in matrix form, from ${M(dLd('S'))} back to ${M('W_1')}, then write grad_check with central differences and the same relative error.`,`第 1、2 部分：从 ${M(dLd('S'))} 开始，用矩阵形式推出两层网络的每个梯度，一直到 ${M('W_1')}。再用中心差分和同样的相对误差写 grad_check。`)},
  {id:'autograd-scan',d:tx('Part 1(b) and Part 3: explain why indexed assignment into P breaks autograd with the in-place error, then write the backward of a chain of matrix products by hand.','第 1(b) 和第 3 部分：解释为什么对 P 做下标赋值会让 autograd 报原地修改的错，再手写一串矩阵连乘的反向。')}]});

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'roots',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Backprop','反向传播'),title:l?L(l.q):tx('How does autograd get every gradient in one backward pass?','autograd 怎么一趟反向算出所有梯度？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: derivatives and the chain rule','前置知识：导数和链式法则'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage bp-stage" id="bp-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="bp-st-title"></div><div class="meter-sub" id="bp-st-sub"></div></div><div class="stage-actions"><span class="status" id="bp-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="bp-legend"></div>`+
   `<div class="stage-graph graph bp-graph" id="bp-graph"><div class="bp-top" id="bp-top"><div class="bp-gwrap" id="bp-gwrap"><svg class="bp-svg" id="bp-svg" role="img" xmlns="http://www.w3.org/2000/svg"></svg><p class="bp-ops" id="bp-ops"></p></div>`+
    `<div class="bp-fx" id="bp-fx" aria-live="polite" hidden></div>`+
    `<div class="bp-side" id="bp-side"><div class="bp-err" id="bp-err" hidden></div>`+
    `<div class="bp-check" id="bp-check" hidden><div class="bp-sub" id="bp-ck-cap"></div><div class="bp-ckrow"><svg class="bp-cksvg" id="bp-cksvg" role="img" xmlns="http://www.w3.org/2000/svg"></svg><div class="bp-cknum" id="bp-cknum"></div></div><div class="bp-chips" id="bp-chips"></div></div></div></div>`+
    `<div class="bp-pwrap" id="bp-pwrap">${playerHTML('bp-st-play',false)}<p class="bp-read" id="bp-read" aria-live="polite"></p></div>`+
   `</div>`+
   `<p class="bp-scope small muted" tabindex="0" data-tipx="${esc(TIP.toy())}">${tx('A real autograd engine computes every value and gradient here.','这里的每个值和梯度都由一个真实的 autograd 引擎算出。')}</p>`+
   `<div class="metrics bp-metrics" id="bp-metrics"></div>`+
   `<details class="math" id="bp-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="bp-mathbody"></div></details></div>`;
}
function controlsHTML(){
  const seg=(id,key,label,opts,val,note)=>`<div class="lockable" id="${id}-wrap">`+A.ui.seg({id,key,label,options:opts,value:val,note})+`</div>`;
  return `<aside class="panel controls" id="bp-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="bp-lock" hidden></p>`+
   seg('bp-ctl-net','net',tx('Network','网络'),[['two',tx('two weights','两个权重')],['tied',tx('one weight used twice','一个权重用两次')]],S.net,tx('With one weight, w multiplies both inputs, so it is used twice.','只有一个权重时，w 同时乘两个输入，所以用了两次。'))+
   seg('bp-ctl-over','over',tx('How a gradient is written','梯度怎么写入'),[['0','+='],['1','=']],S.over?'1':'0',tx('+= adds every contribution; = keeps only the last one. It only matters where a value is used twice.','+= 把每份贡献加起来；= 只留最后一份。只有在某个值被用了两次的地方才有区别。'))+
   seg('bp-ctl-tz','tz',tx('tanh’s local derivative','tanh 的局部导数'),[['0',M('1 - h^2')+tx(' (right)','（对）')],['1',M('1 - z^2')+tx(' (a typo)','（笔误）')]],S.tz?'1':'0',tx(`${M('h = tanh(z)')}, so the right local derivative is ${M('1 - h^2')}. Writing ${M('1 - z^2')} is an easy slip.`,`${M('h = tanh(z)')}，所以正确的局部导数是 ${M('1 - h^2')}。写成 ${M('1 - z^2')} 是很容易犯的笔误。`))+
   seg('bp-ctl-inpl','inpl',tx('In-place change of h after the forward','前向之后原地修改 h'),[['none',tx('none','没有')],['eng',tx('small engine','小引擎')],['pt','PyTorch']],S.inpl,tx(`h.clamp_(max=${CLIP}) before backward. A small engine reads the changed h; PyTorch stops with an error.`,`在反向之前执行 h.clamp_(max=${CLIP})。小引擎会读到改过的 h；PyTorch 会报错停下。`))+
   `<div class="lockable" id="bp-ctl-p-wrap">${pCtl('bp-ctl-p',S.net,false)}</div>`+
   `<div class="lockable" id="bp-ctl-h-wrap">${hCtl('bp-ctl-h',false,tx(`The check moves the parameter up and down by ε. Too large and the finite difference misses the curve’s bend; 0.00001 is a usual choice in float64, small enough for an error near ${M('\\varepsilon^2')}, large enough to stay clear of rounding.`,`检查把参数上下各推 ε。太大，有限差分跟不上曲线的弯曲；float64 下常用 0.00001：小到误差接近 ${M('\\varepsilon^2')}，又大到不受舍入误差影响。`))}</div>`+
   `<div class="bp-grp lockable" id="bp-ctl-vals"><div class="bp-gh">${tx('Assumed values','假设值')}</div><div class="ctl"><span class="lbl">${tx('Inputs, target and starting parameters','输入、目标和参数的初始值')}</span><div class="ctl-note">${tx(`x₁ = 1.5, x₂ = −0.6, t = 0.5; w₁ = 0.6, w₂ = 0.5 (with one weight, w = 0.6), b = 0.1, v = 1.5. They are made up for this lab, small enough to follow by hand and chosen so that no gradient is near 0. A real network starts from random weights; the rule for the gradients is the same.`,`x₁ = 1.5，x₂ = −0.6，t = 0.5；w₁ = 0.6，w₂ = 0.5（只有一个权重时 w = 0.6），b = 0.1，v = 1.5。这些数是为这个 Lab 编的，小到可以手算，并且选得让每个梯度都不接近 0。真实网络从随机权重开始；求梯度的规则是一样的。`)}</div></div></div>`+
   `<div class="lockable" id="bp-ctl-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=`import math
class Value:
    def __init__(self, data, kids=()):
        self.data, self.grad, self.kids = data, 0.0, kids
        self.back = lambda: None
    def __add__(self, o):
        out = Value(self.data + o.data, (self, o))
        def back():                  # += : a value may be used twice
            self.grad += out.grad; o.grad += out.grad
        out.back = back; return out
    def __mul__(self, o):
        out = Value(self.data * o.data, (self, o))
        def back():                  # reads the saved inputs
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out.back = back; return out
    def tanh(self):
        out = Value(math.tanh(self.data), (self,))
        def back(): self.grad += (1 - out.data ** 2) * out.grad
        out.back = back; return out
    def backward(self):              # reverse topological order
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v)); [visit(k) for k in v.kids]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v.back()

def loss(w1, w2, b, v):              # L = (v*tanh(z) - t)**2
    x1, x2, minus_t = Value(1.5), Value(-0.6), Value(-0.5)
    d = v * (w1 * x1 + w2 * x2 + b).tanh() + minus_t
    return d * d                     # d is used twice

ps = [Value(x) for x in (0.6, 0.5, 0.1, 1.5)]
L = loss(*ps); L.backward()
print(round(L.data, 4), [round(p.grad, 4) for p in ps])
h = 1e-5                             # check w1 by nudging it
up, dn = (loss(Value(0.6 + s * h), *ps[1:]).data for s in (1, -1))
print(round((up - dn) / (2 * h), 4))
# 0.1653 [1.1612, -0.4645, 0.7742, 0.4914]
# 1.1612`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="bp-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="bp-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('This lab’s network in 40 lines of Python: forward, one backward pass, and a finite-difference check of w₁. The square is d * d, a value used twice, so += matters.','用 40 行 Python 实现这个 Lab 的网络：前向、一趟反向，以及对 w₁ 的有限差分检查。平方写成 d * d，一个值用了两次，所以 += 很要紧。')}</p><pre class="code" id="bp-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="bp-prod">`+
   tx(`<li>PyTorch’s engine, ${cite(SRC.engine,'<code>torch/csrc/autograd/engine.cpp</code>')}, runs the backward pass: it counts how many uses of each node are still pending and runs a node only when all its incoming gradients have been added up.</li>`+
      `<li>${cite(SRC.func,'<code>torch/autograd/function.py</code>')} defines <code>torch.autograd.Function</code>: you write <code>forward</code>, keep values with <code>ctx.save_for_backward</code>, and write <code>backward</code> yourself. This is where a hand-written backward lives, and <code>torch.autograd.gradcheck</code> checks it with finite differences.</li>`+
      `<li>${cite(SRC.saved,'<code>torch/csrc/autograd/saved_variable.cpp</code>')} raises “one of the variables needed for gradient computation has been modified by an inplace operation” when a saved tensor’s version changed. JAX builds reverse mode from forward-mode rules in ${cite(SRC.jax,'<code>jax/_src/interpreters/ad.py</code>')}.</li>`,
      `<li>PyTorch 的引擎 ${cite(SRC.engine,'<code>torch/csrc/autograd/engine.cpp</code>')} 执行反向：它记着每个节点还有几处用法没传回来，等所有传入的梯度都加完了，才运行这个节点。</li>`+
      `<li>${cite(SRC.func,'<code>torch/autograd/function.py</code>')} 定义了 <code>torch.autograd.Function</code>：你自己写 <code>forward</code>，用 <code>ctx.save_for_backward</code> 保存值，再自己写 <code>backward</code>。手写的反向就在这里，<code>torch.autograd.gradcheck</code> 用有限差分检查它。</li>`+
      `<li>${cite(SRC.saved,'<code>torch/csrc/autograd/saved_variable.cpp</code>')} 在保存的张量版本变了时报错“one of the variables needed for gradient computation has been modified by an inplace operation”。JAX 在 ${cite(SRC.jax,'<code>jax/_src/interpreters/ad.py</code>')} 里用前向模式的规则搭出反向模式。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.rhw,'Learning representations by back-propagating errors')}<span>${tx('Rumelhart, Hinton and Williams, Nature 323, 1986: the gradients of a layered network computed backwards from the output error, layer by layer','Rumelhart、Hinton、Williams，Nature 323，1986：从输出的误差出发，一层一层倒着算出分层网络的梯度')}</span></li>`+
   `<li>${cite(SRC.baydin,'Automatic Differentiation in Machine Learning: a Survey')}<span>${tx('Baydin, Pearlmutter, Radul and Siskind, JMLR 18, 2018: forward and reverse mode, and why reverse mode gives every gradient of one output in one pass','Baydin、Pearlmutter、Radul、Siskind，JMLR 18，2018：前向模式和反向模式，以及为什么反向模式一趟就给出一个输出对所有输入的梯度')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="bp-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="bp-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="bp-app m2" id="bp-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="bp-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see whether its gradients pass their checks.','选一个场景，看看它的梯度能不能通过检查。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="bp-insight" aria-live="polite" hidden></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
/* the challenge network's formulas, one per line (review v6 BP-S2) */
const CHF_TEX=['z = w \\cdot x_1 + w \\cdot x_2 + b','s = \\sigma(z) = \\frac{1}{1 + e^{−z}}','y = v \\cdot s + c','L = (y − t)^2'];
function challengeHTML(){
  const CHF=`<div class="bp-mls">${CHF_TEX.map(t=>`<p class="bp-ml">${A.ui.math(t)}</p>`).join('')}</div>`;
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit"${chPick==null?' disabled':''}>${tx('Submit my answer','提交答案')}</button>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Find the bug in a hand-written backward with as few checks as you can','用尽量少的检查，找出手写反向里的 bug')}</h3>`+
   tx(`<p>A colleague hand-wrote the backward for this new network; the chart shows its forward values:</p>${CHF}<p>σ is the sigmoid function. Exactly one of the four bugs below is in this backward.</p><p>Each gradient check (ε = 0.00001) shows the gradient the hand-written backward gives one parameter, next to the finite difference. Use as few checks as you can. Then name the bug:</p>`,
      `<p>同事给下面这个新网络手写了反向，图里标着前向算出的值：</p>${CHF}<p>σ 是 sigmoid 函数。下面四个 bug 中，正好有一个藏在这段反向里。</p><p>每做一次梯度检查（ε = 0.00001），都能看到手写反向给这个参数的梯度，旁边是有限差分。检查用得越少越好。查完以后，选出是哪个 bug：</p>`)+
   `<div class="bp-chks" id="bp-chks">${chCheckRows()}</div>`+
   `<div class="goal mono bp-goal" id="bp-ch-g"><span>${tx('Checks run','已做的检查')}</span><span><b id="bp-ch-n">${chChecks.length}</b></span></div>`+
   `<div class="bp-lv" id="bp-ch-pick"><p class="small">${tx('The bug, pick one:','是哪个 bug，选一个：')}</p>`+A.ui.levers({aria:tx('The bug, pick one','是哪个 bug，选一个'),value:chPick==null?'':String(chPick),options:CHT.map((t,i)=>({v:String(i),title:chTH(i),desc:'',result:''}))})+`</div>`+
   submit+`<div id="bp-ch-res" class="bp-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function nextWhy(){
  if(mode!=='guided'||!isNextDisabled()) return '';
  const w={predict:{en:'Pick an answer to continue, or Skip.',zh:'选一个答案才能继续，也可以跳过。'},walk:{en:'Press ▸ until the last backward step to continue.',zh:'按 ▸ 走到最后一个反向步骤，才能继续。'},
    check:{en:'Pick ε = 0.00001 to continue.',zh:'选 ε = 0.00001 才能继续。'},fan:{en:'Pick “=” to continue.',zh:'选“=”才能继续。'},inplace:{en:'Pick one of the in-place changes to continue.',zh:'选一种原地修改才能继续。'}}[STEPS[step].id];
  return w?L(w):'';
}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function fillRail(){const rail=$('#bp-rail',app); if(!rail) return null; rail.innerHTML=railFull();
  if(mode==='guided'){const nav=$('.step.current .stepnav',rail), pb=$('.pbar',rail);
    if(nav) nav.insertAdjacentHTML('afterend','<p class="bp-why" hidden></p>'); if(pb) pb.insertAdjacentHTML('beforeend','<p class="bp-why" hidden></p>');}
  A.ui.initRanges(rail); return rail;}
function renderRail(){if(fillRail()) render();}
function updateRail(){
  const rail=$('#bp-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]||(s.id==='walk'&&kNow()>0)||s.id==='check'||s.id==='fan'||s.id==='inplace')&&(!s.gate||s.gate());
    if(isP('fan')) $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===(S.over?'over':'add'))); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=fanTried.has(v)?fanRes(v):'';});
    if(isP('inplace')) $$('[data-lever]',rail).forEach(b=>{const v=b.dataset.lever; b.setAttribute('aria-pressed',String(v===S.inpl)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=inTried.has(v)?inRes(v):'';});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    const why=nextWhy();
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled(); if(why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('.bp-why',rail).forEach(p=>{if(why){p.textContent=why; p.hidden=false; p.style.visibility='';} else if(!p.hidden) p.style.visibility='hidden';});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-rem]',rail).forEach(b=>{const on=b.dataset.rem===remView; b.setAttribute('aria-pressed',String(on)); b.textContent=on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看');});
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#bp-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:'');
      ins.hidden=!html; if(ins._h!==html){const again=ins._h!=null&&ins._h!==''&&html!==''; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const n=$('#bp-ch-n',rail); if(n) n.textContent=String(chChecks.length);
    const cr=$('#bp-chks',rail); if(cr){const h=chCheckRows(); if(cr._h!==h){cr.innerHTML=h; cr._h=h;}}
    $$('#bp-ch-pick [data-lever]',rail).forEach(b=>b.setAttribute('aria-pressed',String(chPick!=null&&b.dataset.lever===String(chPick))));
    const sb=$('[data-act="submit"]',rail); if(sb) sb.disabled=chPick==null;
    const res=$('#bp-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing: the graph, the player, the check panel
   ===================================================================== */
const LAYOUT={
 two:{w1:[0,0],w2:[0,1],m1:[1,0],m2:[1,1],b:[1,2],z:[2,1],h:[3,1],v:[3,2],y:[4,1],L:[5,1]},
 tied:{w:[0,0.5],m1:[1,0],m2:[1,1],b:[1,2],z:[2,1],h:[3,1],v:[3,2],y:[4,1],L:[5,1]},
 ch:{w:[0,0.5],m1:[1,0],m2:[1,1],b:[1,2],z:[2,1],s:[3,1],v:[3,0],c:[3,2],y:[4,1],L:[5,1]}
};
const netNow=()=>mode==='challenge'?'ch':S.net;
/* the label of each box in the SVG: its name and what kind of operation it is. H1 (the owner's decision of 2026-10-09): an SVG text
   cannot hold a typeset formula, so the formulas are in the line under the graph (#bp-ops), in the hovers and in Show the math */
const BOXK={m1:{en:'product',zh:'乘积'},m2:{en:'product',zh:'乘积'},z:{en:'sum',zh:'求和'},h:{en:'tanh',zh:'tanh'},s:{en:'sigmoid',zh:'sigmoid'},y:{en:'output',zh:'输出'},L:{en:'loss',zh:'loss'}};
function boxTitle(id,net){
  if(NETS[net].params.includes(id)) return tx(`${nm(id)} (parameter)`,`${nm(id)}（参数）`);
  const k=BOXK[id]; return k?tx(`${nm(id)} (${k.en})`,`${nm(id)}（${k.zh}）`):id;
}
/* the formula of each operation, in the kit's TeX subset */
function opTex(id,net){const w=net==='two';
  return {m1:`m_1 = ${w?'w_1':'w'} \\cdot x_1`,m2:`m_2 = ${w?'w_2':'w'} \\cdot x_2`,z:'z = m_1 + m_2 + b',h:'h = tanh(z)',s:'s = σ(z)',
    y:net==='ch'?'y = v \\cdot s + c':'y = h \\cdot v',L:'L = (y - t)^2'}[id]||id;}
/* the line under the graph: what each box computes */
function opsHTML(net){const X=NETS[net], ids=Object.keys(LAYOUT[net]).filter(id=>!X.params.includes(id));
  const order=['m1','m2','z','h','s','y','L'].filter(id=>ids.includes(id));
  return `<span class="bp-ops-k">${tx('What each box computes','每个框算什么')}</span>`+order.map(id=>`<span class="bp-op">${M(opTex(id,net))}</span>`).join('');}
/* input tags next to the boxes that use them */
const tagsOf=net=>{const X=NETS[net]; return {m1:`x₁ = ${sg(X.x1,1)}`,m2:`x₂ = ${sg(X.x2,1)}`,L:`t = ${sg(X.t,1)}`};};
let vertical=false;
function geom(net){
  const W=$('#bp-gwrap',app), cw=W?W.clientWidth:640; vertical=true;
  const lay=LAYOUT[net], bw=vertical?100:104, bh=56, pad=10, top=18;
  const pos={}; Object.keys(lay).forEach(id=>{const [c,r]=lay[id]; pos[id]=vertical?{x:pad+r*112,y:top+c*72}:{x:pad+c*(cw>=700?(cw-2*pad-bw)/5:118),y:top+r*80};});
  const xs=Object.values(pos).map(p=>p.x), ys=Object.values(pos).map(p=>p.y);
  return {pos,bw,bh,W:Math.max(...xs)+bw+pad,H:Math.max(...ys)+bh+pad+(vertical?0:46)};
}
const gradTxt=(id,r,k)=>(chHide()&&!chChecks.includes(id))||!r.steps[k].got.includes(id)?'?':f4(r.steps[k].grads[id]);
function drawGraph(){
  const r=R(), net=netNow(), k=kNow(), st=r.steps[k], g=geom(net), svg=$('#bp-svg',app), N=build(net,NETS[net].P);
  const hide=chHide(), cur=st.id&&!hide?st:null, curIns=cur?N[cur.id].ins.map(x=>x.id):[];
  const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
  let edges='', labels='', boxes='';
  const ent=id=>{const p=g.pos[id]; return vertical?[p.x+g.bw/2,p.y]:[p.x,p.y+g.bh/2];}, ext=id=>{const p=g.pos[id]; return vertical?[p.x+g.bw/2,p.y+g.bh]:[p.x+g.bw,p.y+g.bh/2];};
  Object.values(N).forEach(n=>n.ins.forEach((x,i)=>{
    const a=ext(x.id), b=ent(n.id), on=cur&&cur.id===n.id&&!cur.err;
    const d=vertical?`M${a[0]} ${a[1]} C${a[0]} ${a[1]+22} ${b[0]} ${b[1]-22} ${b[0]} ${b[1]}`:`M${a[0]} ${a[1]} C${a[0]+26} ${a[1]} ${b[0]-26} ${b[1]} ${b[0]} ${b[1]}`;
    edges+=`<path d="${d}" class="bp-e${on?' bp-on':''}"${on?'':' marker-end="url(#bp-ar)"'}/>`;
    if(on){const c=cur.contribs[i], m=mid(a,b), txt=r.s.over?'= '+f4(c.c):(c.c<0?`+ (${f4(c.c)})`:'+'+f4(c.c)), w=txt.length*6.6+8;
      labels+=`<g class="bp-cl"><rect x="${(m[0]-w/2).toFixed(1)}" y="${(m[1]-9).toFixed(1)}" width="${w.toFixed(1)}" height="16" rx="3"/><text x="${m[0].toFixed(1)}" y="${(m[1]+3).toFixed(1)}" text-anchor="middle">${esc(txt)}</text></g>`;}
  }));
  const tags=tagsOf(net), isPar=id=>NETS[net].params.includes(id);
  Object.keys(g.pos).forEach(id=>{
    const p=g.pos[id], par=isPar(id), on=cur&&cur.id===id, src=curIns.includes(id)&&cur&&!cur.err, err=cur&&cur.err&&cur.id===id;
    const known=hide?chChecks.includes(id):st.got.includes(id), val=id==='h'&&r.changed?f4(r.changed.to):f4(r.vals[id]);
    const tip=boxTip(id,r,k,net);
    boxes+=`<g class="bp-n${par?' bp-par':''}${on?' bp-on':''}${src?' bp-src':''}${err?' bp-errn':''}${known?' bp-known':''}" data-n="${id}" tabindex="0" data-tipx="${esc(tip)}">`+
      `<rect x="${p.x}" y="${p.y}" width="${g.bw}" height="${g.bh}" rx="6"/>`+
      `<text x="${p.x+7}" y="${p.y+15}" class="bp-t1">${esc(boxTitle(id,net))}</text>`+
      `<text x="${p.x+7}" y="${p.y+32}" class="bp-t2">${esc(tx('value ','值 '))}${esc(val)}</text>`+
      `<text x="${p.x+7}" y="${p.y+48}" class="bp-t3">${esc(tx('grad ','梯度 '))}${esc(gradTxt(id,r,k))}</text></g>`;
    if(id==='h'&&r.changed) tags.h=tx(`clipped, was ${f4(r.changed.from)}`,`截断了，原为 ${f4(r.changed.from)}`);
    if(tags[id]&&!on) boxes+=`<text x="${p.x+g.bw/2+6}" y="${p.y-5}" class="bp-tag">${esc(tags[id])}</text>`;
  });
  /* a value used twice: its two contributions and their sum, next to the box (R6.25) */
  const fan=net!=='two'&&!hide?fanInfo(r,k):null;
  if(fan&&fan.n){const p=g.pos.w, x=vertical?p.x+g.bw+8:p.x, y=vertical?p.y+12:p.y+g.bh+16;
    const lines=fan.n===1?[tx(`${fan.first.lab}: ${f4(fan.first.c)}`,`${fan.first.lab}：${f4(fan.first.c)}`),tx('one use still to come','还有一处没传回')]
      :(r.s.over?[tx(`${f4(fan.first.c)} lost`,`${f4(fan.first.c)} 被覆盖`),tx(`kept only ${f4(fan.second.c)}`,`只留下 ${f4(fan.second.c)}`)]:[`${f4(fan.first.c)} + ${f4(fan.second.c)}`,`= ${f4(fan.first.c+fan.second.c)}`]);
    boxes+=`<g class="bp-fan${r.s.over&&fan.n===2?' bp-bad':''}">`+lines.map((t,i)=>`<text x="${x}" y="${y+i*14}" class="bp-fant">${esc(t)}</text>`).join('')+`</g>`;}
  const defs=`<defs><marker id="bp-ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 z" class="bp-arh"/></marker></defs>`;
  svg.setAttribute('viewBox',`0 0 ${g.W.toFixed(0)} ${g.H.toFixed(0)}`); svg.setAttribute('width',g.W.toFixed(0)); svg.setAttribute('height',g.H.toFixed(0));
  svg.innerHTML=defs+edges+boxes+labels;
  svg.setAttribute('aria-label',hide?tx('The challenge network’s computation graph with its forward values.','挑战题网络的计算图和前向算出的值。'):tx(`The computation graph, after backward step ${k} of ${NT(r)}.`,`计算图，第 ${k} 个反向步骤（共 ${NT(r)} 个）之后。`));
  $('#bp-gwrap',app).classList.toggle('bp-vert',vertical);
  const ops=$('#bp-ops',app), ok=net+A.lang(); if(ops&&ops._k!==ok){ops._k=ok; ops.innerHTML=opsHTML(net);}
  const er=$('#bp-err',app), showErr=!!r.error&&k>=r.error.step;
  er.hidden=!showErr; if(showErr) er.innerHTML=tx(`<b>RuntimeError</b> at y’s backward step: one of the variables needed for gradient computation has been modified by an inplace operation. No gradient reaches the parameters.`,`在 y 的反向步骤处报 <b>RuntimeError</b>：one of the variables needed for gradient computation has been modified by an inplace operation。没有任何梯度传到参数。`);
}
/* where w's contributions stand at walk position k: through x₂ (m₂ runs first), then through x₁ */
function fanInfo(r,k){const out=[]; for(let i=1;i<=k;i++){const s=r.steps[i]; (s.contribs||[]).forEach(c=>{if(c.to==='w') out.push({c:c.c,lab:s.id==='m2'?tx('via x₂','经 x₂'):tx('via x₁','经 x₁')});});}
  return {n:out.length,first:out[0],second:out[1]};}
function boxTip(id,r,k,net){
  const par=NETS[net].params.includes(id), v=f4(r.vals[id]), st=r.steps[k], hide=chHide()&&!chChecks.includes(id);
  const g=st.got.includes(id)&&!hide?f4(st.grads[id]):null, truth=run({net}).grads[id], right=Math.abs(st.grads[id]-truth)<=1e-9*Math.max(1,Math.abs(truth));
  let gl;
  if(hide) gl=tx(' The hand-written backward’s gradient for it stays hidden until the answer is shown; a check of a parameter shows that parameter’s gradient.',' 手写反向给它的梯度在揭晓答案之前不显示；检查一个参数，就能看到这个参数的梯度。');
  else if(g==null) gl=tx(' Its gradient is not known yet at this backward step.',' 走到这一步时，它的梯度还不知道。');
  else if(!st.done.includes(id)) gl=tx(` The backward has added up ${g} for it so far; another use of it has still to pass its contribution back.`,` 反向到这一步为它加起来的是 ${g}；还有一处用到它的地方没把贡献传回来。`);
  else if(mode==='challenge') gl=tx(` The hand-written backward computed ${g} for it.`,` 手写的反向给它算出 ${g}。`);
  else if(!right) gl=tx(` The backward computed ${g}, but a bug is on, and the loss’s real slope for it is ${f4(truth)}.`,` 反向算出 ${g}，但打开了 bug，loss 对它的真实斜率是 ${f4(truth)}。`);
  else gl=tx(` Its gradient is ${g}: the loss changes about ${g} times as much as a tiny nudge of it.`,` 它的梯度是 ${g}：推动它一点点，loss 大约变化推动量的 ${g} 倍。`);
  if(par) return tx(`${nm(id)} is a parameter with value ${v}.`,`${nm(id)} 是参数，值为 ${v}。`)+gl;
  const loc={m1:tx('Its local derivative for its weight is x₁.','它对权重的局部导数是 x₁。'),m2:tx('Its local derivative for its weight is x₂.','它对权重的局部导数是 x₂。'),z:tx('Its local derivative for each input is 1, so it passes its gradient on unchanged.','它对每个输入的局部导数都是 1，所以把梯度原样传给每个输入。'),
    h:tx(`Its local derivative is ${M('1 - h^2')}.`,`它的局部导数是 ${M('1 - h^2')}。`),s:tx(`Its local derivative is ${M('s(1 - s)')}.`,`它的局部导数是 ${M('s(1 - s)')}。`),y:net==='ch'?tx('Its local derivatives are v for s, s for v and 1 for c.','它对 s 的局部导数是 v，对 v 的是 s，对 c 的是 1。'):tx('Its local derivatives are v for h and h for v.','它对 h 的局部导数是 v，对 v 的是 h。'),
    L:tx(`Its local derivative is ${M('2(y - t)')}. Its own gradient, ${M(dLd('L'))}, is 1.`,`它的局部导数是 ${M('2(y - t)')}。它自己的梯度 ${M(dLd('L'))} 是 1。`)}[id]||'';
  return tx(`${M(opTex(id,net))} computed ${v} in the forward pass. ${loc}`,`${M(opTex(id,net))} 在前向时算出 ${v}。${loc}`)+gl;
}
/* the name of each local derivative as the backward that ran used it (with a bug switched on, the buggy one) */
function locName(op,to,r){const f=r.s;
  if(op==='m1') return 'x₁'; if(op==='m2') return 'x₂'; if(op==='z') return '1';
  if(op==='h') return f.tz?'1 − z²':'1 − h²'; if(op==='s') return f.sg?'1 − s²':'s(1 − s)'; if(op==='L') return f.half?'(y − t)':'2(y − t)';
  if(op==='y'){if(to==='c') return '1'; return f.net==='ch'?((to==='s')!==!!f.sw?'v':'s'):((to==='h')!==!!f.sw?'v':'h');}
  return '';}
function readTxt(r,k){
  const nt=NT(r);
  if(k===0) return tx('The backward pass has not started. It starts at the loss, whose gradient with respect to itself is 1. Press ▸ for the first step.','反向还没开始。它会从 loss 出发，loss 对自己的梯度是 1。按 ▸ 走第一步。');
  const s=r.steps[k], net=netNow(), head=tx(`Step ${k} of ${nt}: `,`第 ${k} 步（共 ${nt} 步）：`);
  if(s.err) return head+tx('PyTorch stops here. y’s backward needs the saved h, and h was changed in place after the forward, so no gradient reaches the parameters.','PyTorch 在这里停下。y 的反向要用保存的 h，而 h 在前向之后被原地改过，所以没有梯度传到参数。');
  let t=s.id==='z'?tx('z adds its three inputs, so m₁, m₂ and b each get z’s gradient unchanged.','z 是三项相加，所以 m₁、m₂ 和 b 都原样拿到 z 的梯度。')
    :tx(`${boxTitle(s.id,net)} multiplies its gradient by a local derivative and passes it to ${s.contribs.map(c=>nm(c.to)).join(' and ')}.`,`${boxTitle(s.id,net)}把自己的梯度乘上局部导数，传给 ${s.contribs.map(c=>nm(c.to)).join(' 和 ')}。`);
  s.contribs.filter(c=>c.before!==0).forEach(c=>{const x=nm(c.to);
    t+=r.s.over?tx(` With =, ${x}’s earlier ${f4(c.before)} is overwritten.`,`写成 = 时，${x} 之前那一份 ${f4(c.before)} 被覆盖了。`):tx(` ${x} already had ${f4(c.before)}; this part is added to it.`,`${x} 之前已有一份 ${f4(c.before)}，这一份加上去。`);});
  if(k===r.n) t+=tx(` All ${r.params.length} gradients are in.`,`${r.params.length} 个梯度都有了。`);
  return head+t;
}
/* round 3 (review 2026-10-03): next to the graph, the formula of the operation at the current backward step, forward and backward, with
   its numbers, in the kit's TeX subset. Hidden before part 3 and, in the challenge, until the answer is shown. Every row has one height
   and the box keeps the row count of the step with the most rows, so the stage does not jump while stepping (R7.8) */
const texId=id=>({w1:'w_1',w2:'w_2',m1:'m_1',m2:'m_2'})[id]||id;
const TXLOC={'x₁':'x_1','x₂':'x_2','1 − h²':'(1 − h^2)','1 − z²':'(1 − z^2)','1 − s²':'(1 − s^2)'};
const dL=id=>`\\frac{∂L}{∂${texId(id)}}`;
const nv=v=>f4(v).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');   // 0.5000 → 0.5
const n6=v=>f6(v).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');   // a number that goes into a sum or product: 6 significant digits
const nf=v=>v<0?`({${n6(v)}})`:n6(v);   // a negative number that is not first: in parentheses, grouped so its minus is a sign, not a gap
/* the forward formula of an operation: [the formula, its numbers] */
function fwdTex(id,r,net){const X=NETS[net], v=r.vals, w=net==='two';
  if(id==='m1') return [`m_1 = ${w?'w_1':'w'} \\cdot x_1`,`= ${n6(w?v.w1:v.w)} \\times ${nf(X.x1)} = ${nv(v.m1)}`];
  if(id==='m2') return [`m_2 = ${w?'w_2':'w'} \\cdot x_2`,`= ${n6(w?v.w2:v.w)} \\times ${nf(X.x2)} = ${nv(v.m2)}`];
  if(id==='z') return ['z = m_1 + m_2 + b',`= ${n6(v.m1)} + ${nf(v.m2)} + ${nf(v.b)} = ${nv(v.z)}`];
  if(id==='h') return ['h = tanh(z)',`= tanh(${n6(v.z)}) = ${nv(v.h)}`];
  if(id==='s') return ['s = σ(z)',`= σ(${n6(v.z)}) = ${nv(v.s)}`];
  if(id==='y') return net==='ch'?['y = v \\cdot s + c',`= ${n6(v.v)} \\times ${nf(v.s)} + ${nf(v.c)} = ${nv(v.y)}`]:['y = h \\cdot v',`= ${n6(v.h)} \\times ${nf(v.v)} = ${nv(v.y)}`];
  return ['L = (y − t)^2',`= (${n6(v.y)} − ${nf(X.t)})^2 = ${nv(v.L)}`];}
/* step k as two groups, forward and backward; each item is a formula on one line and its numbers on the next, or a short sentence */
function fxGroups(r,k,net){
  const s=r.steps[k], F={lab:tx('forward','前向'),items:[]}, B={lab:tx('backward','反向'),items:[]};
  if(k===0){const f=fwdTex('L',r,net); F.items.push({m:f[0],n:f[1],f:1}); B.items.push({m:`${dL('L')} = 1`}); return [F,B];}
  const f=fwdTex(s.id,r,net); F.items.push({m:f[0],n:f[1],f:1});
  if(s.id==='h'&&r.changed) F.items.push({t:tx(`then clipped in place to ${nv(r.changed.to)}`,`之后被原地截断成 ${nv(r.changed.to)}`)});
  if(s.err){B.items.push({t:tx('stops: h was changed in place','停下：保存的 h 被原地改过')}); return [F,B];}
  const op=r.s.over?'=':'\\text{+=}';
  if(s.id==='z') B.items.push({m:`${s.contribs.map(c=>dL(c.to)).join(',\\, ')} \\, ${op} \\, ${dL('z')} \\cdot 1`,n:`= ${nv(s.up)}`});
  else s.contribs.forEach(c=>{const ln=locName(s.id,c.to,r);
    B.items.push({m:`${dL(c.to)} \\, ${op} \\, ${dL(s.id)} \\cdot ${TXLOC[ln]||ln}`,n:`= ${n6(c.up)} \\times ${nf(c.local)} = ${nv(c.c)}`});
    if(c.before!==0) B.items.push(r.s.over?{t:tx(`the earlier ${f4(c.before)} is lost`,`之前那一份 ${f4(c.before)} 丢了`)}:{m:`${dL(c.to)} = ${n6(c.before)} + ${nf(c.c)}`,n:`= ${nv(c.after)}`});});
  return [F,B];
}
/* every line has a fixed height, so the box's height follows from its lines alone; the box keeps the height of the tallest step */
const FXH={head:26,lab:18,f:26,m:36,n:24,t:24,pad:20};
const fxHeight=gs=>FXH.head+FXH.pad+gs.reduce((a,g)=>a+FXH.lab+g.items.reduce((b,x)=>b+(x.t!=null?FXH.t:(x.f?FXH.f:FXH.m)+(x.n?FXH.n:0)),0),0);
function drawFx(){
  const box=$('#bp-fx',app); if(!box) return;
  const off=(mode==='guided'&&step<SIX('walk'))||chHide(); box.hidden=off; if(off) return;
  const r=R(), net=netNow(), k=kNow(), nt=NT(r), most=Math.max(...r.steps.map((_,i)=>fxHeight(fxGroups(r,i,net))));
  const h=`<div class="bp-fxh">${k===0?tx('Before the backward pass','反向开始之前'):tx(`Backward step ${k} of ${nt}`,`第 ${k} 个反向步骤（共 ${nt} 个）`)}</div>`+
    fxGroups(r,k,net).map(g=>`<div class="bp-fxl">${esc(g.lab)}</div>`+g.items.map(x=>x.t!=null?`<div class="bp-fxt">${esc(x.t)}</div>`
      :`<div class="bp-fxm${x.f?' bp-fxf':''}">${A.ui.math(x.m)}</div>`+(x.n?`<div class="bp-fxn">${A.ui.math(x.n)}</div>`:'')).join('')).join('');
  box.style.minHeight=most+'px';
  if(box._h!==h){box.innerHTML=h; box._h=h;}
}
function drawPlayer(){
  const r=R(), k=kNow();
  $$('[data-player]',app).forEach(pl=>{const inp=pl.querySelector('input[data-key="k"]'); if(inp){inp.max=r.n; if(+inp.value!==k) inp.value=k; A.ui.setRange(inp); inp.setAttribute('aria-valuetext',tx(`step ${k} of ${NT(r)}`,`第 ${k} / ${NT(r)} 步`));}
    const o=pl.querySelector('output'); if(o) o.textContent=tx(`step ${k} of ${NT(r)}`,`第 ${k} / ${NT(r)} 步`);
    const pb=pl.querySelector('[data-play="play"]'); if(pb){pb.textContent=playing?tx('❚❚ pause','❚❚ 暂停'):(k>=r.n?tx('↺ replay','↺ 重放'):tx('▶ play','▶ 播放')); pb.setAttribute('aria-pressed',String(playing));}
    pl.querySelector('[data-play="back"]').disabled=k<=0; pl.querySelector('[data-play="fwd"]').disabled=k>=r.n;
    $$('.bp-speed button',pl).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===speed)));});
  const rd=$('#bp-read',app); if(rd) rd.textContent=readTxt(r,k);
}
const checkOn=()=>mode==='challenge'?chChecks.length>0:!(mode==='guided'&&step<SIX('check'));
const chkP=()=>mode==='challenge'?chChecks[chChecks.length-1]:S.p;
const chkH=()=>mode==='challenge'?1e-5:HS[S.hi];
function drawCheck(){
  const box=$('#bp-check',app), on=checkOn(); box.hidden=!on;
  const top=$('#bp-top',app); if(top) top.classList.toggle('bp-two',on||!$('#bp-err',app).hidden);
  if(!on) return;
  const r=R(), net=netNow(), p=chkP(), h=chkH(), c=check(r,p,h), P0=NETS[net].P[p];
  $('#bp-ck-cap',app).textContent=tx(`Gradient check of ${nm(p)}: the loss as ${nm(p)} moves, with ε = ${fH(h)}`,`${nm(p)} 的梯度检查：${nm(p)} 变化时的 loss，ε = ${fH(h)}`);
  const W=Math.max(180,Math.min(300,($('#bp-check',app).clientWidth||300)-2)), H=118, half=0.9, pts=curve(net,p,half,48), ys=pts.map(q=>q[1]);
  const y0=Math.min(...ys), y1=Math.max(...ys), pad=8, sx=th=>pad+(th-(P0-half))/(2*half)*(W-2*pad), sy=l=>H-pad-(l-y0)/((y1-y0)||1)*(H-2*pad-6);
  const L0=lossAt(net,NETS[net].P), line=(slope,x0,x1,b=L0)=>`${sx(x0).toFixed(1)},${sy(b+slope*(x0-P0)).toFixed(1)} ${sx(x1).toFixed(1)},${sy(b+slope*(x1-P0)).toFixed(1)}`;
  const span=0.45, poly=pts.map(q=>`${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ');
  const lp=lossAt(net,{...NETS[net].P,[p]:P0+h}), lm=lossAt(net,{...NETS[net].P,[p]:P0-h});
  const svg=$('#bp-cksvg',app); svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width',W); svg.setAttribute('height',H);
  svg.innerHTML=`<rect x="0" y="0" width="${W}" height="${H}" class="bp-ckbg"/><polyline points="${poly}" class="bp-curve"/>`+
    (c.ga!=null?`<polyline points="${line(c.ga,P0-span,P0+span)}" class="bp-tan${c.bad?' bp-bad':''}"/>`:'')+
    `<polyline points="${line(c.gn,P0-Math.max(span,h),P0+Math.max(span,h),(lp+lm)/2)}" class="bp-sec"/>`+
    (h>=0.05?`<circle cx="${sx(P0-h).toFixed(1)}" cy="${sy(lm).toFixed(1)}" r="3.5" class="bp-dot"/><circle cx="${sx(P0+h).toFixed(1)}" cy="${sy(lp).toFixed(1)}" r="3.5" class="bp-dot"/>`:'')+
    `<circle cx="${sx(P0).toFixed(1)}" cy="${sy(L0).toFixed(1)}" r="3" class="bp-pt"/>`;
  svg.setAttribute('aria-label',tx(`The loss as ${nm(p)} moves around ${f4(P0)}, a line with autograd’s gradient as its slope and the finite-difference line.`,`${nm(p)} 在 ${f4(P0)} 附近变化时的 loss，一条斜率是 autograd 梯度的直线，和有限差分那条直线。`));
  svg.dataset.tipx=tx(`The curve is the loss as ${nm(p)} moves from ${f4(P0-half)} to ${f4(P0+half)} with everything else fixed. The solid line’s slope is autograd’s gradient; the dashed line, whose slope is the finite difference, goes through the loss at ${nm(p)} − ε and ${nm(p)} + ε.`,`曲线是其他都不变、${nm(p)} 从 ${f4(P0-half)} 变到 ${f4(P0+half)} 时的 loss。实线的斜率是 autograd 的梯度；虚线的斜率是有限差分，它穿过 ${nm(p)} − ε 和 ${nm(p)} + ε 处的 loss。`);
  svg.setAttribute('tabindex','0');
  const res=c.ga==null?`<span class="bp-bad">${tx('no gradient: the backward stopped','没有梯度：反向报错停下了')}</span>`:(c.pass?`<span class="okc">${tx('✓ passes','✓ 通过')}</span>`:(c.bad?`<span class="bp-bad">${tx('✗ fails','✗ 不通过')}</span>`:`<span>${tx('not a pass yet','还不算通过')}</span>`));
  const full=mode==='challenge'||kNow()>=r.n;
  $('#bp-cknum',app).innerHTML=`<div><span>${full?tx('autograd','autograd'):tx('autograd, full pass','autograd，整趟')}</span><b class="mono">${c.ga==null?'—':f4(c.ga)}</b></div><div><span>${tx('finite difference','有限差分')}</span><b class="mono">${f4(c.gn)}</b></div>`+
    `<div tabindex="0" data-tipx="${esc(TIP.rel())}"><span>${tx('relative error','相对误差')}</span><b class="mono">${fE(c.err)}</b></div><div>${res}</div>`;
  /* every parameter's check at this h; in the challenge only the ones run */
  const chips=NETS[net].params.map(q=>{const cq=check(r,q,h), shown=mode!=='challenge'||chChecks.includes(q);
    const lab=!shown?'?':(cq.ga==null?'—':(cq.pass?'✓':(cq.bad?'✗':'≈')));
    const tip=!shown?tx(`${nm(q)} has not been checked yet.`,`${nm(q)} 还没检查。`):(cq.ga==null?tx(`${nm(q)} has no gradient: the backward stopped.`,`${nm(q)} 没有梯度：反向报错停下了。`):tx(`${nm(q)}: autograd ${f4(cq.ga)}, finite difference ${f4(cq.gn)}, relative error ${fE(cq.err)}, so the check ${cq.pass?'passes':(cq.bad?'fails':'does not pass yet')}.`,`${nm(q)}：autograd ${f4(cq.ga)}，有限差分 ${f4(cq.gn)}，相对误差 ${fE(cq.err)}，所以检查${cq.pass?'通过':(cq.bad?'不通过':'还不算通过')}。`));
    return `<button type="button" class="bp-chip${shown&&cq.ga!=null?(cq.pass?' bp-ok':(cq.bad?' bp-bad':'')):''}${q===p?' bp-on':''}" data-chip="${q}" aria-pressed="${q===p}" data-tipx="${esc(tip)}">${nm(q)} ${lab}</button>`;}).join('');
  const ce=$('#bp-chips',app); ce.innerHTML=`<span class="bp-sub">${tx(`Every parameter at ε = ${fH(h)}:`,`ε = ${fH(h)} 时每个参数的检查：`)}</span>`+chips;
}
/* how many parameters have their whole gradient (review v6 BP-S8) */
const readyTip=(k,rd,np,twice)=>(k===0?tx(`The backward pass has not started: none of the ${np} parameters has a gradient yet.`,`反向还没开始，${np} 个参数都还没有梯度。`)
  :rd===0?tx(`After ${k} backward ${pl(k,'step','steps')}, none of the ${np} parameters has a complete gradient yet.`,`走完 ${k} 个反向步骤，${np} 个参数都还没有拿到完整的梯度。`)
  :rd===np?tx(`After ${k} backward ${pl(k,'step','steps')}, all ${np} parameters have a complete gradient.`,`走完 ${k} 个反向步骤，${np} 个参数全都拿到了完整的梯度。`)
  :tx(`After ${k} backward ${pl(k,'step','steps')}, ${rd} of the ${np} parameters ${pl(rd,'has','have')} a complete gradient.`,`走完 ${k} 个反向步骤，${np} 个参数里已经有 ${rd} 个拿到完整的梯度。`))
  +(twice&&k>0?tx(' A parameter used twice has its complete gradient only after both uses have sent theirs back.','用了两次的参数，要等两处都传回来，梯度才完整。'):'');
function statusOf(){
  const r=R(), k=kNow();
  if(mode==='challenge'){const n=chChecks.length; return {cls:chSubmitted&&chPick===chHid()?'ok':'neutral',txt:chSubmitted?(chPick===chHid()?tx('Bug found','找到 bug'):tx('Not yet','还没找到')):tx(`${n} ${pl(n,'check','checks')} run`,`做了 ${n} 个检查`),short:chSubmitted?(chPick===chHid()?tx('Bug found','找到 bug'):tx('Not yet','还没找到')):tx(`${n} ${pl(n,'check','checks')}`,`${n} 个检查`),
    tip:tx(`You have run ${n} gradient ${pl(n,'check','checks')}.`,`你已经做了 ${n} 个梯度检查。`)};}
  if(r.error&&k>=r.error.step) return {cls:'bad',txt:tx('Backward stopped','反向报错'),short:tx('Error','报错'),tip:tx('PyTorch stopped the backward pass: a value it saved was changed in place.','PyTorch 停止了反向：它保存的一个值被原地改了。')};
  const rd=readyOf(r,k), np=r.params.length;
  if(!checkOn()||k<r.n) return {cls:'neutral',txt:tx(`${rd} of ${np} gradients ready`,`${np} 个梯度已有 ${rd} 个`),short:tx(`${rd} of ${np} ready`,`已有 ${rd} / ${np}`),tip:readyTip(k,rd,np,false)};
  const c=check(r,S.p,HS[S.hi]), e=fH(HS[S.hi]), word=c.pass?tx('passes','通过'):(c.bad?tx('fails','不通过'):tx('does not pass yet','还不算通过'));
  return {cls:c.pass?'ok':(c.bad?'bad':'neutral'),txt:c.pass?tx(`Check of ${nm(S.p)} passes`,`${nm(S.p)} 的检查通过`):(c.bad?tx(`Check of ${nm(S.p)} fails`,`${nm(S.p)} 的检查不通过`):tx(`Check of ${nm(S.p)}: not a pass yet`,`${nm(S.p)} 的检查还不算通过`)),
    short:c.pass?tx('Check passes','检查通过'):(c.bad?tx('Check fails','检查不通过'):tx('Not a pass yet','还不算通过')),
    tip:c.pass?tx(`With ε = ${e}, the relative error of ${nm(S.p)} is ${fE(c.err)}, below 10⁻⁷: the check passes.`,`ε = ${e} 时，${nm(S.p)} 的相对误差是 ${fE(c.err)}，低于 10⁻⁷，检查通过。`)
      :(c.bad?tx(`With ε = ${e}, the relative error of ${nm(S.p)} is ${fE(c.err)}, above 10⁻⁴: the check fails (a pass needs below 10⁻⁷).`,`ε = ${e} 时，${nm(S.p)} 的相对误差是 ${fE(c.err)}，高于 10⁻⁴，检查不通过（低于 10⁻⁷ 才算通过）。`)
      :tx(`With ε = ${e}, the relative error of ${nm(S.p)} is ${fE(c.err)}: not a pass yet (a pass needs below 10⁻⁷).`,`ε = ${e} 时，${nm(S.p)} 的相对误差是 ${fE(c.err)}，还不算通过（低于 10⁻⁷ 才算通过）。`))};
}
function drawHead(){
  const net=netNow(), t=$('#bp-st-title',app), sub=$('#bp-st-sub',app), st=$('#bp-status',app), r=R();
  t.textContent=net==='ch'?tx('The challenge network and its hand-written backward','挑战题的网络和手写的反向'):(net==='tied'?tx('One weight w used for both inputs','一个权重 w 同时用于两个输入'):tx('A small network with two weights','有两个权重的小网络'));
  const bugs=[]; if(mode!=='challenge'){if(S.over) bugs.push(tx('= instead of +=','= 而不是 +=')); if(S.tz) bugs.push(tx('tanh′ written as 1 − z²','tanh′ 写成 1 − z²')); if(S.inpl==='eng') bugs.push(tx('h clipped in place (small engine)','原地截断 h（小引擎）')); if(S.inpl==='pt') bugs.push(tx('h clipped in place (PyTorch)','原地截断 h（PyTorch）'));}
  sub.textContent=tx(`Loss L = ${f4(r.L)}; `,`loss L = ${f4(r.L)}；`)+(mode==='challenge'?tx('one hidden bug in the backward','反向里藏着一个 bug'):(bugs.length?tx('bugs on: ','打开的 bug：')+bugs.join(tx(', ','，')):tx('no bugs','没有 bug')));
  const s=statusOf(); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function legendHTML(){
  const it=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  let h=it('bp-sw-par',tx('box: a parameter','方框：参数'),tx('A box with a thick border is a parameter: training changes it, so it needs a gradient.','粗边框的方框是参数：训练会改它，所以它需要梯度。'))+
    it('bp-sw-op',tx('box: an operation','方框：运算'),tx('Each other box is one operation of the forward pass, with the value it computed and, once the backward pass reaches it, its gradient.','其余每个方框是前向的一个运算，写着它算出的值；反向走到它时，再写上它的梯度。'))+
    it('bp-sw-on',tx('dashed outline: the step being run','虚线框：正在执行的步骤'),tx('The dashed outline marks the operation whose backward step was just run; its thick arrows carry each contribution to its inputs.','虚线框标出刚执行反向步骤的那个运算；它的粗箭头把每份贡献送到输入。'));
  if(checkOn()) h+=it('bp-sw-tan',tx('solid line: slope = autograd’s gradient','实线：斜率 = autograd 的梯度'),tx('In the check panel, the solid line goes through the current point with the slope autograd computed.','检查面板里，实线穿过当前点，斜率是 autograd 算出的梯度。'))+
    it('bp-sw-sec',tx('dashed line: slope = the finite difference','虚线：斜率 = 有限差分'),tx('In the check panel, the dashed line’s slope is the finite difference, measured between θ − ε and θ + ε, where θ is the parameter being checked (the two dots show when ε is large enough to see).','检查面板里，虚线的斜率是有限差分，在 θ − ε 和 θ + ε 之间量出，θ 是被检查的参数（ε 够大时，两个点看得见）。'));
  return h;
}
const numOf=v=>{const m=String(v).replace(/,/g,'').replace(MINUS,'-').match(/^-?\d+(?:\.\d+)?/); return m?parseFloat(m[0]):NaN;};
function metricsDef(){
  const r=R(), k=kNow(), np=r.params.length, rd=readyOf(r,k), nt=NT(r);
  const loss={k:'loss',label:tx('Loss L','loss L'),v:f4(r.L),sub:tx('from the forward pass','前向算出'),tip:tx(`The loss the forward pass computed: ${M(`L = (y - t)^2 = ${f4(r.L)}`)}. The backward pass does not change it.`,`前向算出的 loss：${M(`L = (y - t)^2 = ${f4(r.L)}`)}。反向不会改变它。`)};
  const errM=c=>({k:'err',label:tx('Check: relative error','检查：相对误差'),v:fE(c.err),sub:c.ga==null?tx('no gradient','没有梯度'):(c.pass?tx('passes, below 10⁻⁷','通过，低于 10⁻⁷'):(c.bad?tx('fails, above 10⁻⁴','不通过，高于 10⁻⁴'):tx('not a pass yet, above 10⁻⁷','还不算通过，高于 10⁻⁷'))),tip:TIP.rel()});
  if(chHide()){const n=chChecks.length;
    const chk={k:'chk',label:tx('Checks run','已做的检查'),v:tx(`${n} of ${np}`,`${n} / ${np}`),sub:tx('parameters checked','个参数已检查'),tip:tx(`You have checked ${n} of the ${np} parameters.`,`${np} 个参数里，你检查了 ${n} 个。`)};
    if(!n) return [loss,chk];
    const p=chkP(), c=check(r,p,1e-5);
    return [loss,chk,{k:'grad',subj:'ch|'+p,label:tx(`Gradient of ${nm(p)}`,`${nm(p)} 的梯度`),v:f4(c.ga),sub:tx('from the hand-written backward','手写反向给出'),tip:tx(`The hand-written backward’s gradient of ${nm(p)} is ${f4(c.ga)}. The finite difference measures ${f4(c.gn)}.`,`手写反向给 ${nm(p)} 的梯度是 ${f4(c.ga)}。有限差分量出 ${f4(c.gn)}。`)},errM(c)];}
  const steps={k:'steps',label:tx('Backward steps run','已执行的反向步骤'),v:tx(`${k} of ${nt}`,`${k} / ${nt}`),sub:tx('one per operation','每个运算一步'),tip:tx(`The backward pass has ${nt} steps, one per operation, from the loss back to the parameters; ${k} have run.`,`反向一共 ${nt} 步，每个运算一步，从 loss 走回参数；已经执行了 ${k} 步。`)};
  const ready={k:'ready',label:tx('Gradients ready','已有梯度'),v:tx(`${rd} of ${np}`,`${rd} / ${np}`),sub:tx('parameters','参数'),tip:readyTip(k,rd,np,true)};
  if(!checkOn()) return [loss,steps,ready];
  const p=chkP(), c=check(r,p,chkH()), full=mode==='challenge'||k>=r.n;
  return [loss,ready,{k:'grad',subj:`${netNow()}|${p}`,label:tx(`Gradient of ${nm(p)}`,`${nm(p)} 的梯度`),v:c.ga==null?'—':f4(c.ga),sub:full?tx('from autograd','autograd 给出'):tx('autograd, after the full pass','autograd 跑完整趟后给出'),delta:true,tip:c.ga==null?tx('No gradient: the backward pass stopped with an error.','没有梯度：反向报错停下了。'):tx(`Autograd’s gradient of ${nm(p)} after the full backward pass is ${f4(c.ga)}. The finite difference measures ${f4(c.gn)}.`,`autograd 跑完整趟反向后给 ${nm(p)} 的梯度是 ${f4(c.ga)}。有限差分量出 ${f4(c.gn)}。`)},errM(c)];
}
function drawMetrics(){
  const host=$('#bp-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; defs.forEach(m=>{shownPrev[m.k]=null;}); host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip;
    let dt=''; const prev=shownPrev[m.k];
    if(m.delta&&userAct&&prev&&prev.v!==m.v&&prev.subj===m.subj){const a=numOf(prev.v), b=numOf(m.v); if(isFinite(a)&&isFinite(b)&&a>0&&b>0) dt=A.fmt.delta(a,b,{dp:4})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]={v:m.v,subj:m.subj};});
}
/* Show the math: every backward step from the numbers as shown (6 significant digits in, 4 decimals out), then the check (R6.6) */
/* Show the math as a few short blocks (H2, the owner's decision of 2026-10-09): each block has a lead line in words, the formula with its
   name on the left, then the same formula with the numbers, and the result; a chain breaks at “=” onto lines whose “=” line up (R6.29) */
const EQ=(lhs,rhs)=>`<div class="bp-eq bp-math-line">`+rhs.map((x,i)=>`<span class="bp-eq-l">${i?'':M(lhs)}</span><span class="bp-eq-r">${M('= '+x)}</span>`).join('')+`</div>`;
const FXL=(name,tex)=>`<div class="bp-mfx bp-math-line"><span class="bp-mname">${name}</span>${M(tex)}</div>`;
const BLK=(lead,body)=>`<div class="bp-mblk"><p class="bp-mlead">${lead}</p>${body}</div>`;
const SUPN={'⁻':'-','⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'};
const fET=x=>fE(x).replace(/ × 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/,(m,e)=>` \\times 10^{${e.split('').map(c=>SUPN[c]).join('')}}`);
function mathHTML(){
  const r=R(), net=netNow(), X=NETS[net], out=[], ops=r.ord.filter(id=>!X.params.includes(id));
  out.push(BLK(tx('<b>Forward.</b> Each operation computes its value from its inputs. The backward pass reuses these values.','<b>前向。</b>每个运算用它的输入算出自己的值。反向会再用到这些值。'),
    `<div class="bp-mtab-w"><table class="bp-mtab"><tr><th>${tx('operation','运算')}</th><th>${tx('value','值')}</th></tr>`+
      ops.map(id=>`<tr><td>${M(opTex(id,net))}</td><td>${f4(r.vals[id])}</td></tr>`).join('')+`</table></div>`+
    `<p class="bp-msym">${tx(`The inputs are ${M('x_1')} = ${sg(X.x1,1)} and ${M('x_2')} = ${sg(X.x2,1)}, and the target is ${M('t')} = ${sg(X.t,1)}.`,`输入 ${M('x_1')} = ${sg(X.x1,1)}，${M('x_2')} = ${sg(X.x2,1)}，目标 ${M('t')} = ${sg(X.t,1)}。`)}</p>`));
  if(chHide()) out.push(BLK(tx('<b>Backward.</b> The hand-written backward’s arithmetic appears here once you submit a right answer or click “Show the answer”. Until then, only the checks you run tell you about it.','<b>反向。</b>答对之后，或者点了“直接看答案”之后，这里才列出手写反向的算式。在那之前，只有你做的检查能告诉你它对不对。'),''));
  else {const op=r.s.over?'=':'\\text{+=}';
    const cell=c=>`<div>${M(`${dLd(texId(c.to))} \\, ${op} \\, ${n6(c.up)} \\times ${nf(c.local)} = ${nv(c.c)}`)}${c.before!==0&&!r.s.over?tx(`, total ${f4(c.after)}`,`，合计 ${f4(c.after)}`):''}</div>`;
    out.push(BLK(tx('<b>Backward.</b> The pass starts at the loss, whose gradient with respect to itself is 1. At each step, an operation multiplies its own gradient by its local derivative and adds the product into the gradient of each input.',
                    '<b>反向。</b>反向从 loss 出发，loss 对自己的梯度是 1。每一步，一个运算把自己的梯度乘上局部导数，再加进它每个输入的梯度。'),
      FXL(tx('each step','每一步'),'\\frac{∂L}{∂u} \\, \\text{+=} \\, \\frac{∂L}{∂o} \\cdot \\frac{∂o}{∂u}')+
      `<p class="bp-msym">${tx(`${M('o')} is the operation’s output, ${M('u')} one of its inputs, and ${M('\\frac{∂o}{∂u}')} its local derivative.`,`${M('o')} 是运算的输出，${M('u')} 是它的一个输入，${M('\\frac{∂o}{∂u}')} 是它的局部导数。`)}</p>`+
      `<div class="bp-mtab-w"><table class="bp-mtab bp-mtab-b"><tr><th>${tx('step','步')}</th><th>${tx('operation','运算')}</th><th>${tx('added into the inputs’ gradients','加进输入的梯度')}</th></tr>`+
      r.steps.slice(1).map((st,i)=>`<tr><td>${i+1}</td><td>${M(opTex(st.id,net))}</td><td>${st.err?tx('PyTorch stops: the saved h changed.','PyTorch 停下：保存的 h 被改过。'):st.contribs.map(cell).join('')}</td></tr>`).join('')+`</table></div>`));}
  if(checkOn()){const p=chkP(), h=chkH(), c=check(r,p,h), P0=X.P[p], lp=lossAt(net,{...X.P,[p]:P0+h}), lm=lossAt(net,{...X.P,[p]:P0-h}), th=texId(p);
    const verdict=c.ga==null?'':(c.pass?tx('Below 10⁻⁷: the check passes.','低于 10⁻⁷：检查通过。'):(c.bad?tx('Above 10⁻⁴: the check fails.','高于 10⁻⁴：检查不通过。'):tx('Above 10⁻⁷: not a pass yet.','高于 10⁻⁷：还不算通过。')));
    out.push(BLK(tx(`<b>Gradient check of ${nm(p)}.</b> Move ${nm(p)} up and down by ε and compute the loss each time; the difference divided by 2ε is the finite difference n.`,`<b>${nm(p)} 的梯度检查。</b>把 ${nm(p)} 往上、往下各推 ε，各算一次 loss。两次 loss 之差除以 2ε，就是有限差分 n。`),
      FXL(tx('finite difference','有限差分'),'n = '+FD_TEX)+
      `<p class="bp-msym">${tx(`Here ${M('\\theta')} is ${M(th)} = ${f6(P0)} and ${M('\\varepsilon')} = ${fH(h)}:`,`这里 ${M('\\theta')} 是 ${M(th)} = ${f6(P0)}，${M('\\varepsilon')} = ${fH(h)}：`)}</p>`+
      EQ('n',[`\\frac{L(${f6(P0+h)}) - L(${f6(P0-h)})}{${fH(2*h)}}`,`\\frac{${lp.toFixed(10)} - ${lm.toFixed(10)}}{${fH(2*h)}}`,f6(c.gn)])+
      (c.ga==null?`<p>${tx('The backward gave no gradient, so there is nothing to compare.','反向没有给出梯度，所以没有可比较的。')}</p>`:
        FXL(tx('relative error','相对误差'),REL_TEX)+
        `<p class="bp-msym">${tx(`${M('a')} is autograd’s gradient, ${f6(c.ga)}:`,`${M('a')} 是 autograd 的梯度，${f6(c.ga)}：`)}</p>`+
        EQ(tx('\\text{relative error}','\\text{相对误差}'),[`\\frac{|${f6(c.ga)} - ${f6(c.gn)}|}{|${f6(c.ga)}| + |${f6(c.gn)}|}`,fET(c.err)])+`<p>${verdict}</p>`)));}
  return out.join('');
}
function render(){
  if(!app) return;
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead(); drawGraph(); drawFx(); drawPlayer(); drawCheck(); drawMetrics();
  /* a wide stage with no check on: the metric cards fill the column beside the graph, so the stage fits 1440 × 900 (review A6) */
  const top=$('#bp-top',app), side=$('#bp-side',app), met=$('#bp-metrics',app), chk=!$('#bp-check',app).hidden||!$('#bp-err',app).hidden, inSide=$('#bp-stage',app).clientWidth>=610&&!chk&&$('#bp-fx',app).hidden;
  if(inSide&&met.parentNode!==side) side.appendChild(met); else if(!inSide&&met.parentNode===side) $('.bp-scope',app).after(met);
  /* the formula box: first in the column beside the graph on a wide stage, right under the graph on a narrow one */
  const fx=$('#bp-fx',app), wide=$('#bp-stage',app).clientWidth>=610;
  if(wide&&fx.parentNode!==side) side.prepend(fx); else if(!wide&&fx.parentNode===side) $('#bp-gwrap',app).after(fx);
  top.classList.toggle('bp-two',chk||inSide||(wide&&!fx.hidden));
  const gr=$('#bp-graph',app); gr.dataset.focus=mode==='guided'?({walk:'graph',check:'check',fan:'graph',inplace:'graph'}[STEPS[step].id]||''):'';
  $('#bp-pwrap',app).hidden=(mode==='guided'&&step<SIX('walk'))||chHide();
  const lg=$('#bp-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#bp-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function lockNote(){
  if(mode==='challenge') return tx('In the challenge you run checks and name the bug in the task card. The network, its values and its hand-written backward are fixed.','挑战中只能在任务卡里做检查、选 bug。网络、它的值和手写的反向都固定不变。');
  if(mode!=='guided') return '';
  const id=STEPS[step].id;
  if(id==='check') return tx('In this part you pick the parameter and ε in the card; the network has two weights and no bugs. You can change everything in Scenarios.','这一部分在卡片里选参数和 ε；网络有两个权重，没有 bug。在应用场景里可以改所有设置。');
  if(id==='fan') return tx('In this part you pick how w’s gradient is written in the card; the network uses one weight twice. You can change everything in Scenarios.','这一部分在卡片里选 w 的梯度怎么写入；网络把一个权重用了两次。在应用场景里可以改所有设置。');
  if(id==='inplace') return tx('In this part you pick the in-place change in the card. You can change everything in Scenarios.','这一部分在卡片里选原地修改。在应用场景里可以改所有设置。');
  return tx('Fixed in this part: the network with two weights and no bugs. You can change everything in Scenarios.','这一部分固定：两个权重的网络，没有 bug。在应用场景里可以改所有设置。');
}
function syncControls(){
  const c=$('#bp-controls',app), g=mode==='guided', ch=mode==='challenge';
  $$('.seg-ctl[data-key="net"]',app).forEach(e=>A.ui.setPressed(e,S.net)); $$('.seg-ctl[data-key="over"]',app).forEach(e=>A.ui.setPressed(e,S.over?'1':'0'));
  $$('.seg-ctl[data-key="tz"]',app).forEach(e=>A.ui.setPressed(e,S.tz?'1':'0')); $$('.seg-ctl[data-key="inpl"]',app).forEach(e=>A.ui.setPressed(e,S.inpl));
  $$('.seg-ctl[data-key="h"]',app).forEach(e=>A.ui.setPressed(e,String(S.hi)));
  /* the parameter buttons follow the network */
  const pw=$('#bp-ctl-p-wrap',app); if(pw&&pw._net!==S.net){pw._net=S.net; pw.innerHTML=pCtl('bp-ctl-p',S.net,false);}
  $$('.seg-ctl[data-key="p"]',app).forEach(e=>A.ui.setPressed(e,S.p));
  c.classList.toggle('locked',ch);
  ['#bp-ctl-net-wrap','#bp-ctl-over-wrap','#bp-ctl-tz-wrap','#bp-ctl-inpl-wrap','#bp-ctl-p-wrap','#bp-ctl-h-wrap','#bp-ctl-vals','#bp-ctl-reset-wrap'].forEach(id=>{const e=$(id,app); if(e) e.hidden=g||(ch&&id!=='#bp-ctl-vals');});
  const vals=$('#bp-ctl-vals',app); if(vals) vals.hidden=g||ch;
  const ln=$('#bp-lock',app), msg=lockNote(); ln.hidden=!msg; if(msg&&ln.textContent!==msg) ln.textContent=msg;
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const r=R(), k=kNow(); A.announce(tx(`Backward step ${k} of ${NT(r)}; ${readyOf(r,k)} of ${r.params.length} gradients ready.`,`第 ${k} / ${NT(r)} 个反向步骤；${r.params.length} 个梯度已有 ${readyOf(r,k)} 个。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p,keepK){
  Object.assign(S,p);
  if(!NETS[S.net]||S.net==='ch') S.net='two'; if(!NETS[S.net].params.includes(S.p)) S.p=NETS[S.net].params[0];
  if(!['none','eng','pt'].includes(S.inpl)) S.inpl='none'; if(!(S.hi>=0&&S.hi<HS.length)) S.hi=3;
  if(!keepK) K=R().n;
  stopPlay(); render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function setK(v){K=Math.max(0,Math.min(R().n,v)); if(K>=R().n){if(mode!=='challenge') watched.add(sigOf(S)); if(isP('walk')) seenEnd=true;} render();}
function setFan(v){fanTried.add(v); S.over=v==='over'; K=R().n; render();}
function setIn(v){inTried.add(v); S.inpl=v; K=R().n; render();}
function setRem(k){if(!['walk','check','fan','inpl'].includes(k)) return; remView=k; if(k==='walk') S={...BASE}; else if(k==='check') S={...BASE,hi:0}; else if(k==='inpl') S={...BASE,net:'tied',p:'v',inpl:'eng'}; else S={...BASE,net:'tied',p:'w',over:true}; K=R().n; renderRail(); if(innerWidth<=760) A.ui.reveal($('#bp-stage',app));}
/* the player: time-based, one backward step per interval (R3.9); Show me never changes the learner's speed */
function stopPlay(){playing=false; clearTimeout(playT); playT=0;}
function tickPlay(){playT=0; if(!app||!playing) return; const r=R(); if(K>=r.n){stopPlay(); drawPlayer(); return;} setK(K+1); if(K>=r.n){stopPlay(); drawPlayer(); return;} playT=setTimeout(tickPlay,A.reduceMotion()?400:SPEEDS[speed]);}
function startPlay(){const r=R(); if(K>=r.n) K=0; playing=true; drawPlayer(); playT=setTimeout(tickPlay,A.reduceMotion()?200:Math.min(900,SPEEDS[speed]/2));}
function enterStep(i){
  wantObs=false; stopPlay(); step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  const st=$('#bp-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#bp-rail .step.current',app)||$('#bp-rail',app));}
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
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), as labs/sampling.js does */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#bp-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#bp-rail .step.current',app)||$('#bp-rail .railcard',app)||$('#bp-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function stageShow(){if(!app) return; const st=$('#bp-stage',app); if(!st||innerWidth<=760||st.scrollHeight<=st.clientHeight+2) return;
  const el=$('#bp-err:not([hidden])',app)||$('#bp-svg .bp-n.bp-errn',app)||$('#bp-svg .bp-n.bp-on',app); if(!el) return;
  const sr=st.getBoundingClientRect(), r=el.getBoundingClientRect(); if(r.top>=sr.top+8&&r.bottom<=sr.bottom-8) return;
  st.scrollTo({top:st.scrollTop+(r.bottom>sr.bottom-8?r.bottom-sr.bottom+24:r.top-sr.top-24),behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#bp-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#bp-insight',app)); else revealBox($('#bp-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  stopPlay(); const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chChecks=[]; chPick=null; chSubmitted=false; chReveal=false;}
  if(m==='sandbox'){sbTouched=false; if(was==='guided') S={...BASE};}
  K=R().n;
  A.shell.setMode(app,m,DESCS());
  const st=$('#bp-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[S.net,S.over?1:0,S.tz?1:0,S.inpl,S.p,S.hi].join('_');
function parse(tok){
  const p=String(tok||'').split('_'); if(p.length!==6) return null;
  const [net,ov,tz,inpl,pp,hi]=p;
  if(!['two','tied'].includes(net)||!['0','1'].includes(ov)||!['0','1'].includes(tz)||!['none','eng','pt'].includes(inpl)||!NETS[net].params.includes(pp)||!/^[0-3]$/.test(hi)) return null;
  return {net,over:ov==='1',tz:tz==='1',inpl,p:pp,hi:+hi};
}

/* =====================================================================
   interview drill: one neuron, seeded; every line redoes from the numbers it shows (R6.6)
   ===================================================================== */
function makeDrill(seed){const rnd=mulberry(seed*104729+7), pick=a=>a[Math.floor(rnd()*a.length)];
  const x=pick([0.5,1.5,2.0,-0.5,-1.5])  /* never ±1, so ∂L/∂w ≠ ±∂L/∂b shows the × x (A20) */, w=pick([0.3,0.4,0.5,-0.3,-0.5]), b=pick([0.1,0.2,-0.1]), t=pick([0.0,0.5,1.0]);
  const z=w*x+b, y=Math.tanh(z), dy=2*(y-t), loc=1-y*y, gb=dy*loc, gw=gb*x; return {seed,x,w,b,t,z,y,dy,loc,gb,gw};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), qd=$('#bp-drill-q',app), a=$('#bp-drill-a',app);
  /* H1/H2 (2026-10-09): the formulas typeset, one line each */
  const ML=t=>`<div class="bp-mfx bp-math-line">${M(t)}</div>`;
  qd.innerHTML=tx(`A neuron computes ${M('y = tanh(w \\cdot x + b)')} with x = <b>${sg(d.x,1)}</b>, w = <b>${sg(d.w,1)}</b> and b = <b>${sg(d.b,1)}</b>. The loss is ${M('L = (y - t)^2')} with t = <b>${sg(d.t,1)}</b>. Find ${M(dLd('w'))} and ${M(dLd('b'))} by hand.`,
                  `一个神经元计算 ${M('y = tanh(w \\cdot x + b)')}，x = <b>${sg(d.x,1)}</b>，w = <b>${sg(d.w,1)}</b>，b = <b>${sg(d.b,1)}</b>。loss 是 ${M('L = (y - t)^2')}，t = <b>${sg(d.t,1)}</b>。手算 ${M(dLd('w'))} 和 ${M(dLd('b'))}。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  a.innerHTML=`<p><b>${M(`${dLd('w')} = ${f4(d.gw)}`)}${tx(', ','，')}${M(`${dLd('b')} = ${f4(d.gb)}`)}</b></p>`+
    `<p>${tx('Forward:','前向：')}</p>`+ML(`z = w \\cdot x + b = ${sg(d.w,1)} \\times ${nf(d.x)} + ${nf(d.b)} = ${f6(d.z)}`)+ML(`y = tanh(z) = ${f6(d.y)}`)+
    `<p>${tx('Backward, from the loss to w:','反向，从 loss 走回 w：')}</p>`+ML(`${dLd('y')} = 2(y - t) = ${f6(d.dy)}`)+ML(`\\frac{∂y}{∂z} = 1 - y^2 = ${f6(d.loc)}`)+
    ML(`${dLd('b')} = ${dLd('y')} \\cdot \\frac{∂y}{∂z} = ${f6(d.dy)} \\times ${nf(d.loc)} = ${f4(d.gb)}`)+ML(`${dLd('w')} = ${dLd('b')} \\cdot x = ${f6(d.gb)} \\times ${nf(d.x)} = ${f4(d.gw)}`);
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const TERM={h:()=>`<div>${esc(tx('ε is how far the check moves the parameter θ up and down. The finite difference it measures is:','ε 是检查时把参数 θ 上下推动的距离。量出的有限差分是：'))}</div><div class="fx">${M(FD_TEX)}</div>`};
const termTipHTML=k=>TERM[k]?`${TERM[k]()}<div class="act">${esc(jumpAct(k))}</div>`:'';
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='graph') return tx('Click to scroll to the computation graph.','点击滚到计算图。');
  if(k==='check') return tx('Click to scroll to the check panel.','点击滚到检查面板。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(k==='h'){if(mode==='challenge') return tx('Fixed during the challenge: ε = 0.00001.','挑战中固定不变：ε = 0.00001。'); if(mode==='guided'&&!isP('check')) return tx('Fixed in this part: ε = 0.00001; part 4 lets you change it.','这一部分固定：ε = 0.00001；第 4 部分可以改。'); return tx('Click to jump to “Nudge ε”.','点击跳到“推动量 ε”。');}
  return '';
}
function jump(k,src){
  if(!app) return;
  const sm=A.reduceMotion()?'auto':'smooth';
  if(k==='math'){const d=$('#bp-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(d); return;}
  if(k==='graph'||k==='check'){const el=$(k==='graph'?'#bp-gwrap':'#bp-check',app), g=el&&!el.hidden&&el.getClientRects().length?el:$('#bp-graph',app); A.tip.hide(); g.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(g); return;}
  if(k==='controls'){const c=$('#bp-controls',app); A.tip.hide(); A.ui.jumpToEl(c); return;}
  if(k==='fid'){const b=$(`[data-pop="${ID}-fid"]`,app), p=document.getElementById(`${ID}-fid`); A.tip.hide(); if(p&&p.hidden&&b) b.click(); const el=p||b; if(el){el.scrollIntoView({block:'center',behavior:sm}); A.ui.flash(el);} return;}
  if(k==='h'){const el=isP('check')?$('#bp-ic-h',app):(mode==='sandbox'?$('#bp-ctl-h',app):null); if(!el){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;} A.ui.jumpToEl(el);}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{S={...BASE}; setState({});}); return;}
  if(a==='submit'){if(chPick==null) return; chSubmitted=true; if(chPick===chHid()) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='another'){chSeed++; chChecks=[]; chPick=null; chSubmitted=false; chReveal=false; renderRail(); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#bp-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
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
const bySettings=(el,f)=>el&&el.closest('#bp-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key; if(k!=='k') return; stopPlay(); byLearner(()=>setK(+t.value)); A.tip.quiet&&A.tip.quiet();});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&!d.play&&b.closest('#bp-rail')&&!b.closest('[data-exits]')) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; stopPlay(); byLearner(()=>typeof s.showMe==='function'?s.showMe():(s.showMe.hi!=null?(hTried.add(s.showMe.hi),setState(s.showMe,true)):setState(s.showMe))); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#bp-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.play){if(A.tip.quiet) A.tip.quiet(); const r=R();
      if(d.play==='fwd'){stopPlay(); byLearner(()=>setK(K+1));} else if(d.play==='back'){stopPlay(); byLearner(()=>setK(K-1));} else if(d.play==='play'){if(playing){stopPlay(); drawPlayer();} else startPlay();}
      if(b.closest('#bp-rail')) requestAnimationFrame(stageShow);
      if(K>=r.n&&isP('walk')){wantObs=true; revealSoon(); wantObs=false;} return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&isP('fan')){byLearner(()=>setFan(d.lever)); revealSoon(); requestAnimationFrame(stageShow); return;}
    if(d.lever!=null&&isP('inplace')){byLearner(()=>setIn(d.lever)); revealSoon(); requestAnimationFrame(stageShow); return;}
    if(d.lever!=null&&b.closest('#bp-ch-pick')){const was=chSubmitted; chPick=+d.lever; if(chSubmitted){chSubmitted=false; chReveal=false;} if(was){renderRail(); const f=$(`#bp-ch-pick [data-lever="${chPick}"]`,app); if(f) f.focus({preventScroll:true});} else render(); return;}
    if(d.chk){if(!chChecks.includes(d.chk)) chChecks.push(d.chk); render(); return;}
    if(d.chip){if(mode==='challenge'){if(chChecks.includes(d.chip)){chChecks=chChecks.filter(x=>x!==d.chip).concat([d.chip]); render();} return;}
      if(mode==='guided'&&!isP('check')) {A.tip.show(b,`<div>${esc(tx('Fixed in this part; part 4 lets you pick the parameter.','这一部分固定；第 4 部分可以选参数。'))}</div>`); return;}
      byLearner(()=>setState({p:d.chip},true)); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='speed'){speed=d.v; drawPlayer(); if(playing){clearTimeout(playT); playT=setTimeout(tickPlay,SPEEDS[speed]);} return;}
      if(k==='p') bySettings(b,()=>setState({p:d.v},true));
      else if(k==='h') bySettings(b,()=>{hTried.add(+d.v); setState({hi:+d.v},true); if(isP('check')){wantObs=true; revealSoon(); wantObs=false;}});
      else if(k==='net') bySettings(b,()=>setState({net:d.v}));
      else if(k==='over') bySettings(b,()=>setState({over:d.v==='1'}));
      else if(k==='tz') bySettings(b,()=>setState({tz:d.v==='1'}));
      else if(k==='inpl') bySettings(b,()=>setState({inpl:d.v}));
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({hi:3,...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#bp-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#bp-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   知识梳理 (the owner's outline of 2026-10-12 07:13 UTC): four sections read before the guided parts. Why training needs every
   parameter's gradient, why measuring them one parameter at a time is too slow, how the chain rule walks back along the graph, and
   what autograd records in the forward pass. They state mechanisms only; how strong each effect is stays with the guided parts. The one
   picture is the lab's own graph (drawGraph) after the forward pass and before any backward step, drawn with the primer's own state
   ===================================================================== */
// drawGraph with the primer's own state (the initial network, no backward step), whatever the reader set in the lab, put back after
const PRDEF=()=>({mode:'guided',S:{...BASE},K:0,chChecks:[],chReveal:false});
function withPop(o,f){const sv={mode,S,K,chChecks,chReveal,app};
  ({mode,S,K,chChecks,chReveal}={...PRDEF(),...o});
  try{return f();} finally{({mode,S,K,chChecks,chReveal,app}=sv);}}
// the primer's copy gets its own ids, so the lab's own lookups (#bp-svg, #bp-gwrap, the arrowhead marker) never land in the primer
const prIds=h=>h.replace(/(\b(?:id|for|aria-labelledby|aria-describedby|aria-controls)="|url\(#|href="#)bp-/g,'$1pr-bp-');
function prGraph(){
  const box=document.createElement('div');
  box.innerHTML=`<div class="bp-gwrap" id="bp-gwrap"><svg class="bp-svg" id="bp-svg" role="img" xmlns="http://www.w3.org/2000/svg"></svg><p class="bp-ops" id="bp-ops"></p></div><div class="bp-err" id="bp-err" hidden></div>`;
  withPop({},()=>{app=box; drawGraph();});
  /* before the backward pass no gradient is known, the loss's own included (the backward pass sets it when it starts); the boxes are
     a picture here, not hover targets */
  const gL=box.querySelector('[data-n="L"]'); if(gL){gL.classList.remove('bp-known'); const t=gL.querySelector('.bp-t3'); if(t) t.textContent=tx('grad ','梯度 ')+'?';}
  box.querySelectorAll('[data-tipx]').forEach(e=>{e.removeAttribute('data-tipx'); e.removeAttribute('tabindex');});
  const svg=box.querySelector('svg'); if(svg) svg.setAttribute('aria-label',tx('The small network’s computation graph with the values the forward pass computed; no gradient is known yet.','小网络的计算图，写着前向算出的值；还没有任何梯度。'));
  box.querySelector('#bp-err').remove();
  return prIds(box.innerHTML);
}
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
const PFX=t=>`<div class="bp-prfx">${M(t)}</div>`;   // a formula on its own line
const dd=(a,b)=>`\\frac{∂${a}}{∂${b}}`, ds=(a,b)=>`∂${a}/∂${b}`;   // a derivative on its own line, and in running text
A.registerPrimer(ID,{min:7,sections:[
 {h:()=>tx('Training makes the loss smaller (gradient descent)','训练就是让 loss 变小（梯度下降）'),parts:[
   prP(()=>tx(`The numbers in a neural network that are set by training are called its parameters. Before training they are random, so the network’s outputs are mostly wrong. To improve the parameters you first need to know how wrong the network is: compare its outputs on the training data with the right answers and compute one number, the loss. The more wrong the outputs, the larger it is. Write all the parameters together as ${M('\\theta')}; the loss is a function of them, ${M('L(\\theta)')}, and training looks for a ${M('\\theta')} that makes ${M('L(\\theta)')} as small as possible.`,
         `神经网络里要靠训练定下来的数，叫参数。训练开始时参数是随机的，网络的输出基本都是错的。要把参数调好，先得知道错了多少：拿网络在训练数据上的输出和正确答案比，算出一个数值，叫 loss（损失），错得越多它越大。把全部参数合起来记作 ${M('\\theta')}，loss 就是它的函数 ${M('L(\\theta)')}，训练要找的是让 ${M('L(\\theta)')} 尽量小的 ${M('\\theta')}。`),
       ()=>tx(`There are far too many parameters to try out all their values. What you can do is look only nearby: around the current ${M('\\theta')}, which direction makes the loss fall fastest? The direction opposite to the gradient ${M(ds('L','\\theta'))}. Each step of gradient descent moves all the parameters a little that way:`,
         `参数很多，不可能把各种取值都试一遍。能做的是只看眼前：在当前的 ${M('\\theta')} 附近，往哪个方向动，loss 降得最快？是梯度 ${M(ds('L','\\theta'))} 的反方向。梯度下降每一步把所有参数一起朝这个方向挪一小步：`),
       ()=>PFX(`\\theta \\leftarrow \\theta - \\eta \\, ${dd('L','\\theta')}`),
       ()=>tx(`${M('\\eta')} is the learning rate: how far one step goes.`,`${M('\\eta')} 叫学习率，决定一步挪多远。`),
       ()=>tx('Once the parameters change, so does the gradient, so every step computes it afresh: run the forward pass with the current parameters, from the inputs all the way to the loss, then get the gradients of all the parameters, then update them. Training repeats this step many times, and each time it needs the gradient of every parameter.',
         '参数一变，梯度也跟着变，所以每一步都要重新算：用当前的参数跑一遍前向，也就是从输入一路算到 loss，再求出所有参数的梯度，然后更新参数。训练要把这一步重复很多次，每次都要全部参数的梯度。'),
       ()=>tx(`The gradient has to come out of the way the loss is computed, so first take that computation apart step by step. Take a small network with only 4 parameters: ${M('y = v \\cdot tanh(w_1 x_1 + w_2 x_2 + b)')} and ${M('L = (y - t)^2')}, where x₁ and x₂ are the inputs and t is the target. Split the forward pass into simple operations and draw each result as a box, with arrows from the values an operation uses to its box; this picture is the computation graph. Each box shows the value the forward pass computed, and its last line is kept for the gradient of the loss with respect to that value. Training needs the gradients in the 4 parameter boxes.`,
         `梯度要从 loss 的计算过程里求出来，所以先把这个过程一步步拆开。拿一个只有 4 个参数的小网络：${M('y = v \\cdot tanh(w_1 x_1 + w_2 x_2 + b)')}，${M('L = (y - t)^2')}，其中 x₁、x₂ 是输入，t 是目标。把前向拆成一个个简单运算，每个运算的结果画成一个框，箭头从运算用到的值指向它，这张图叫计算图。框里写着前向算出的值，最后一行留给 loss 对这个值的梯度；训练要的，是 4 个参数框里的梯度。`)),
   {fig:[{html:prGraph,cap:()=>tx('The thick boxes are the 4 parameters and every other box is one operation; x₁, x₂ and t beside the arrows are the fixed inputs and the target, with no gradient. No box’s gradient has been computed yet.','粗框是 4 个参数，其余每个框是一次运算；箭头旁的 x₁、x₂、t 是固定的输入和目标，不求梯度。每个框的梯度都还没有算。'),ms:1e9}],cls:'pr-bp'}]},
 {h:()=>tx('Nudging one parameter at a time is too slow (finite differences)','逐个推一推参数太慢（有限差分）'),parts:[
   prP(()=>tx(`How do you get a gradient? The most direct way is to measure it: keep the other parameters fixed, set ${M('\\theta_i')} to ${M('\\theta_i + \\varepsilon')} and to ${M('\\theta_i - \\varepsilon')}, run the forward pass for each, and take the slope of the line through the two points as an approximation to the loss’s derivative at ${M('\\theta_i')}:`,
         `梯度怎么求？最直接的办法是量出来：其余参数固定，只把 ${M('\\theta_i')} 换成 ${M('\\theta_i + \\varepsilon')} 和 ${M('\\theta_i - \\varepsilon')}，各跑一遍前向，用这两点连线的斜率近似 loss 在 ${M('\\theta_i')} 处的导数：`),
       ()=>PFX(`${dd('L','\\theta_i')} \\approx \\frac{L(\\theta_i + \\varepsilon) - L(\\theta_i - \\varepsilon)}{2\\varepsilon}`),
       ()=>tx('This is the finite difference. It uses only the forward pass, and no derivative formula has to be written.','这叫有限差分。它只用到前向，一条求导公式也不用写。'),
       ()=>tx(`The trouble is the number of parameters. Each parameter is nudged twice on its own, and each nudge costs a full forward pass, so n parameters take ${M('2n')} forward passes for one set of gradients. For a language model with ${M('8 \\times 10^9')} parameters, every training step would take ${M('1.6 \\times 10^{10}')} forward passes with finite differences, and each of them uses all ${M('8 \\times 10^9')} parameters. Put together, the work per step grows with the square of the number of parameters.`,
         `麻烦在参数的数量。每个参数要单独推两次，每次都是一趟完整的前向，所以 n 个参数要 ${M('2n')} 趟前向，才得到一次全部的梯度。一个有 ${M('8 \\times 10^9')} 个参数的语言模型，用有限差分求梯度，训练每走一步就要 ${M('1.6 \\times 10^{10}')} 趟前向，而每一趟前向都要把这 ${M('8 \\times 10^9')} 个参数全部用一遍。合起来，每一步的计算量和参数个数的平方成正比。`),
       ()=>tx(`And ${M('\\varepsilon')} cannot be 0, so what you measure is always an approximation. For one or two parameters, though, a few forward passes are cheap, so finite differences are often used to check gradients computed some other way. To get every gradient, another way is needed: a few passes that compute every parameter’s gradient at once, and the exact derivative rather than an approximation.`,
         `而且 ${M('\\varepsilon')} 不能取 0，量出来的始终是近似值。不过只查一两个参数时，几趟前向并不贵，所以有限差分常用来核对别的方法算出的梯度。要求全部的梯度，就需要另一种办法：只跑很少几趟，就把全部参数的梯度一起算出来，而且得到的是导数的精确值。`))]},
 {h:()=>tx('The chain rule multiplies back along the graph (backpropagation)','链式法则沿计算图往回乘（反向传播）'),parts:[
   prP(()=>tx(`The other way is to compute the gradient by the rules of differentiation instead of measuring it. The forward pass is a chain of simple operations: multiply, add, tanh, square. The derivative of one operation’s output with respect to one of its inputs is its local derivative, and it is easy to write down: for ${M('y = h \\cdot v')} it is v with respect to h, and for ${M('L = (y - t)^2')} it is ${M('2(y - t)')} with respect to y.`,
         `另一种办法是不去量，而是按求导法则直接算。前向是一串简单运算：乘、加、tanh、平方。一个运算的输出对它某个输入的导数，叫局部导数，很容易写出来：${M('y = h \\cdot v')} 对 h 的局部导数是 v，${M('L = (y - t)^2')} 对 y 的是 ${M('2(y - t)')}。`),
       ()=>tx(`On the computation graph, the chain rule reads: walk from a parameter along the arrows to the loss and multiply the local derivatives on the way; if there are several paths from the parameter to the loss, add up the products of all of them. For example, when ${M('\\theta')} reaches L through node p and then node q:`,
         `放到计算图上，链式法则就是：从参数沿箭头走到 loss，把路上的局部导数乘起来；从参数到 loss 有几条路，就把几条路的乘积加起来。比如 ${M('\\theta')} 先经过节点 p、再经过节点 q 到达 L：`),
       ()=>PFX(`${dd('L','\\theta')} = ${dd('L','q')} \\cdot ${dd('q','p')} \\cdot ${dd('p','\\theta')}`),
       ()=>tx(`The paths of different parameters share their part near the loss. In a deep network, every parameter of an early layer reaches the loss through all the layers after it, and the product of the local derivatives along that stretch is the same for all of them. If each parameter multiplied along its own path, that stretch would be computed again and again. Backpropagation turns the direction around and walks back from the loss. At each node u it first gets u’s gradient ${M(ds('L','u'))}, then multiplies it by a local derivative and hands it to u’s input x. When x is used only by u, this is x’s gradient:`,
         `不同参数通往 loss 的路，在靠近 loss 的一段是重合的。在一个很深的网络里，前面一层的每个参数都要穿过后面所有的层才到 loss，这一段的局部导数之积对它们都一样。要是每个参数各乘各的路，这一段就要重算很多遍。反向传播（backpropagation）把方向倒过来，从 loss 出发往回走。每到一个节点 u，先得到 u 的梯度 ${M(ds('L','u'))}，再乘上局部导数，交给 u 的输入 x。x 只被 u 一个运算用到时，这就是 x 的梯度：`),
       ()=>PFX(`${dd('L','x')} = ${dd('L','u')} \\cdot ${dd('u','x')}`),
       ()=>tx('So every node’s gradient is computed once, and when the walk reaches a parameter, that parameter’s gradient is there.','这样每个节点的梯度只算一次，走到参数时，参数的梯度也就有了。'),
       ()=>tx(`Each operation is handled once in the backward pass, with multiplications and additions of the same order as in its forward computation. So one backward pass costs the same order as one forward pass, and the number of passes does not depend on how many parameters there are: for 4 parameters or for ${M('8 \\times 10^9')}, one forward pass and one backward pass give every gradient. It walks backwards because there is one loss and very many parameters. Walking back, each node keeps one value, ${M(ds('L','u'))}, and one walk serves every parameter. Walking forward from one parameter (forward mode), each node would keep ${M(ds('u','\\theta'))}, which serves only that parameter, so n parameters would take n walks.`,
         `每个运算在反向里只处理一次，做的乘法和加法跟它在前向里是同一个量级。所以一趟反向的计算量和一趟前向是同一个量级。要跑的趟数和参数有多少个无关：4 个参数也好，${M('8 \\times 10^9')} 个也好，一趟前向加一趟反向，就得到全部梯度。之所以倒着走，是因为 loss 只有一个，参数却有很多。倒着走，每个节点只记一个值 ${M(ds('L','u'))}，一趟就照顾到所有参数。要是从一个参数往前走（前向模式），每个节点记的 ${M(ds('u','\\theta'))} 只对这一个参数有用，n 个参数就要走 n 趟。`))]},
 {h:()=>tx('How autograd runs the backward pass: the graph, saved values, accumulating gradients','autograd 怎样自动反向：计算图、保存的值、梯度累加'),parts:[
   prP(()=>tx('Backpropagation walks back over the operations the forward pass ran, but the forward pass is ordinary code, and which operations it runs is known only when it runs. So the framework has to record, during the forward pass, what the backward pass will need. In PyTorch this part is autograd (automatic differentiation); calling <code>loss.backward()</code> starts the walk back from the loss. It records two things during the forward pass, and the backward pass has one more job.',
         '反向传播要沿着前向做过的运算倒着走，可前向是一段普通代码，要执行哪些运算，运行时才知道。所以框架得在前向时把反向要用的东西记下来。PyTorch 里负责这件事的部分叫 autograd（自动微分），调用 <code>loss.backward()</code> 就从 loss 开始往回走。它在前向时记下两样东西，反向时还要多做一件事。'),
       ()=>tx('The first is the computation graph: which values each operation used and which value it produced. The backward pass handles the operations in the reverse of their forward order, and which inputs each operation hands its gradient to comes from the graph. PyTorch adds each operation to the graph as the forward pass runs it.',
         '第一样是计算图：每个运算用了哪些值，算出了哪个值。反向按前向的倒序处理运算，每个运算把梯度交给哪些输入，都要从图里查。PyTorch 在前向执行每个运算时，顺手把它记进图里。'),
       ()=>tx(`The second is the values the local derivatives need. Many local derivatives use numbers from the forward pass. The addition ${M('z = m_1 + m_2 + b')} has a local derivative of 1 for every input and needs nothing stored; the product ${M('m_1 = w_1 \\cdot x_1')} has ${M('x_1')} as its local derivative with respect to ${M('w_1')}, and the square ${M('L = (y - t)^2')} has ${M('2(y - t)')} with respect to y, so the backward pass needs the forward values. Since the forward pass already has these values, autograd stores them (saved tensors, in PyTorch), and the backward pass reads them instead of computing them again. They stay in memory until the backward pass has used them, one reason training takes more GPU memory than running the forward pass alone.`,
         `第二样是局部导数要用的值。很多局部导数要用到前向时的数值。加法 ${M('z = m_1 + m_2 + b')} 对每个输入的局部导数都是 1，什么也不用存；乘法 ${M('m_1 = w_1 \\cdot x_1')} 对 ${M('w_1')} 的局部导数是 ${M('x_1')}，平方 ${M('L = (y - t)^2')} 对 y 的是 ${M('2(y - t)')}，反向时都要拿到前向的数值。这些值前向时已经有了，autograd 就把它们存下来（PyTorch 里叫 saved tensors），反向直接读，不用重算。它们要一直留到反向用完，这是训练比只跑前向多占显存的一个原因。`),
       ()=>tx('The backward pass has one more job: accumulation. A value can be used by several operations: one weight multiplied in several places, or one intermediate result used by two operations. The multivariable chain rule says that its gradient is then the sum of the contributions passed back by each operation u that uses it:',
         '反向时还要多做一件事：累加。一个值可能被几个运算用到：同一个权重在几处相乘，或者一个中间结果同时进入两个运算。多元链式法则说，这时它的梯度是各个用到它的运算 u 传回的贡献之和：'),
       ()=>PFX(`${dd('L','x')} = \\sum_{u} ${dd('L','u')} \\cdot ${dd('u','x')}`),
       ()=>tx('The contributions arrive one after another during the backward pass, so autograd starts every gradient at 0 and adds each contribution as it arrives (<code>+=</code>).',
         '这几份贡献在反向时先后到达，所以 autograd 让每个梯度从 0 开始，每到一份就加进去（<code>+=</code>）。'))]}
]});

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S and K as they were (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S},K,pred,observed:{...observed},saved:saved?{...saved}:null,chSeed,chChecks:chChecks.slice(),chPick,chSubmitted,chReveal,sbTouched,
    seenEnd,hTried:[...hTried],fanTried:[...fanTried],inTried:[...inTried],remView,watched:[...watched],speed,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chPick===chHid()}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse(sigOf(x)):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...BASE,...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...BASE,...sv}:null;
    chSeed=Number.isInteger(o.chSeed)&&o.chSeed>=1&&o.chSeed<1e6?o.chSeed:1;
    chChecks=mode==='challenge'&&Array.isArray(o.chChecks)?[...new Set(o.chChecks.filter(x=>NETS.ch.params.includes(x)))]:[];
    chPick=mode==='challenge'&&Number.isInteger(o.chPick)&&o.chPick>=0&&o.chPick<CHT.length?o.chPick:null;
    chSubmitted=chPick!=null&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true;
    seenEnd=o.seenEnd===true; remView=['walk','check','fan','inpl'].includes(o.remView)?o.remView:'walk';
    const arr=(a,ok)=>Array.isArray(a)?a.filter(ok):[];
    hTried=new Set(arr(o.hTried,x=>Number.isInteger(x)&&x>=0&&x<=3)); fanTried=new Set(['add',...arr(o.fanTried,x=>x==='add'||x==='over')]);
    inTried=new Set(['none',...arr(o.inTried,x=>['none','eng','pt'].includes(x))]);
    watched.clear(); arr(o.watched,x=>typeof x==='string'&&!!parse(x)).forEach(x=>watched.add(x));
    speed=SPEEDS[o.speed]&&Object.prototype.hasOwnProperty.call(SPEEDS,o.speed)?o.speed:'med';
    const n=R().n; K=Number.isInteger(o.K)&&o.K>=0&&o.K<=n?o.K:n;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el; stopPlay();
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...BASE,...p}; mode='sandbox'; sbTouched=true; K=run(S).n;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#bp-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#bp-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&step===0&&!fromSnap){const s=STEPS[0]; Object.assign(S,s.enter());}
    fromSnap=false;
    fillRail();
    render();
    requestAnimationFrame(()=>{if(app){drawGraph(); drawCheck();}});
  },
  leave(){clearTimeout(annT); stopPlay(); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#bp-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#bp-math',app).open=true;}},
  onResize(){if(app){drawGraph(); drawCheck();}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    /* the bar shows only once the stage's title row has gone (A5); while the stage is stuck on a desktop the kit keeps the bar away
       itself (kit polish K4), so the anchor no longer switches to the whole stage */
    anchor:()=>app&&$('#bp-stage .stage-head',app),
    bench:()=>app&&$('.bench',app), target:()=>app&&$('#bp-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(), r=R(), k=kNow(), nt=NT(r), hide=chHide(), np=r.params.length;
      status.className='status '+s.cls; status.textContent=s.short;
      const g=r.steps[k].grads, shown=p=>hide?chChecks.includes(p):r.steps[k].got.includes(p), mx=Math.max(1e-9,...r.params.filter(shown).map(p=>Math.abs(g[p]||0)));
      vis.innerHTML=`<div class="bp-dk" tabindex="0" data-tipx="${esc(tx('Each bar’s length is the size of that parameter’s gradient, relative to the largest one shown; the sign is left out. An empty bar has no gradient to show yet.','每根条的长度是这个参数梯度的大小，相对于显示出的最大的那个；不计正负。空条表示还没有梯度可显示。'))}">`+r.params.map(p=>`<span><i style="width:${shown(p)?(Math.abs(g[p])/mx*100).toFixed(1):0}%"></i><em>${nm(p)}</em></span>`).join('')+`</div>`;
      if(r.error&&k>=r.error.step){read.innerHTML=esc(tx(`Stopped at step ${k} of ${nt}; no gradients`,`停在第 ${k} / ${nt} 步，没有梯度`)); return;}
      if(hide){const n=chChecks.length; read.innerHTML=esc(tx(`${n} of ${np} parameters checked`,`${np} 个参数已检查 ${n} 个`)); return;}
      if(checkOn()&&mode!=='challenge'){const c=check(r,S.p,HS[S.hi]); read.innerHTML=esc(tx(`${nm(S.p)}: autograd ${f4(c.ga)}, finite difference ${f4(c.gn)}`,`${nm(S.p)}：autograd ${f4(c.ga)}，有限差分 ${f4(c.gn)}`)); return;}
      read.innerHTML=esc(tx(`Step ${k} of ${nt}: ${readyOf(r,k)} of ${np} gradients ready`,`第 ${k} / ${nt} 步：${np} 个梯度已有 ${readyOf(r,k)} 个`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__bp=Object.freeze({
  run:s=>{const r=run(s); return deep({L:r.L,vals:r.vals,grads:r.grads,n:r.n,ops:r.ops,ord:r.ord,changed:r.changed,error:r.error,steps:r.steps.map(x=>({id:x.id,err:!!x.err,got:x.got,grads:x.grads,contribs:x.contribs||[]}))});},
  check:(s,p,h)=>deep(check(run(s),p,h)), fdiff, fx:(s,k)=>deep(fxGroups(run(s),k,s.net)), lossAt:(net,P)=>lossAt(net,{...NETS[net].P,...(P||{})}),
  fails:i=>failsOf(i), hidden:hiddenOf, consistent:(h,set)=>consistent(h,set), minChecks:h=>minChecks(h), drill:s=>deep(makeDrill(s)),
  exits:()=>{const e=EXITS(); return deep({transferA:e.transfer.map(q=>q.a)});},
  fmt:Object.freeze({f4,f6,fE}),
  consts:Object.freeze({NETS:deep(NETS),HS:HS.slice(),CLIP,PASS,SUSPECT,BASE:{...BASE},STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),SPEEDS:{...SPEEDS}}),
  state:()=>({...S,K,mode,step,pred,observed:{...observed},seenEnd,hTried:[...hTried],fanTried:[...fanTried],inTried:[...inTried],remView,playing,speed,chSeed,chChecks:chChecks.slice(),chPick,chSubmitted,chReveal,stageBusy:stageT!==0,vertical})
});
})();
