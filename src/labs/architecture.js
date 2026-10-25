/* Architecture overview v2 (#architecture): what a decoder-only Transformer is made of, drawn as a data flow (specs/architecture-v2.md,
   review 9, LAB_UX_GUIDE R0.1: the picture explains, the text almost disappears). Five token lanes flow through the tokenizer, the
   embedding table, a stack of layers (RMSNorm → self-attention → ⊕, RMSNorm → MLP → ⊕), the final RMSNorm, the LM head and softmax to
   the next token, which loops back to the input. Only attention mixes lanes; the MLP keeps them parallel. Every component opens a
   pop-up whose diagram animates one computation step at a time. The model switch changes the drawing: lane width ∝ √d, MLP bulge
   ∝ √intermediate size, one plate per layer, one hairline per head. Values inside the pop-ups are illustrative (labelled once); the
   softmax, SiLU and RMSNorm demos compute live. Shapes come from each model's config.json (the next lab, #transformer, counts them).
   Also registers the home tile (A.registerTeaser) and the home hero (A.registerHero, wide: six stages left to right). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='architecture';

/* ---------- models: config.json of each (specs/transformer-facts.md) ---------- */
const MODELS={
 'llama32-1b':{name:'Llama-3.2-1B',V:128256,d:2048,L:16,H:32,KV:8,f:8192,tied:true},
 'llama31-8b':{name:'Llama-3.1-8B',V:128256,d:4096,L:32,H:32,KV:8,f:14336,tied:false},
 'llama31-70b':{name:'Llama-3.1-70B',V:128256,d:8192,L:80,H:64,KV:8,f:28672,tied:false},
 'llama31-405b':{name:'Llama-3.1-405B',V:128256,d:16384,L:126,H:128,KV:8,f:53248,tied:false}
};
const MKEYS=Object.keys(MODELS), DEF_M='llama31-8b';
const short=k=>MODELS[k].name.replace(/^Llama-3\.\d-/,'');
/* the prompt and the next token; IDs from Meta's Llama 3 tokenizer */
const TOKS=[{t:'The',id:791},{t:' capital',id:6864},{t:' of',id:315},{t:' France',id:9822},{t:' is',id:374}];
const NEXT={t:' Paris',id:12366};
const CANDS=[{t:' Paris',id:12366},{t:' a',id:264},{t:' the',id:279},{t:' located',id:7559},{t:' Lyon',id:56301}];
const Z0=[8,6,5.5,5,4];                                        // illustrative logits
const fi=n=>A.fmt.int(n);
const qt=t=>`“${esc(t)}”`;
const DEF=(t,tip)=>A.ui.def(t,tip);
const LK=id=>{const a=A.shell.labLink(id); return tx(a,`“${a}”`);};
const ease=p=>p<.5?2*p*p:1-Math.pow(-2*p+2,2)/2;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const seg01=(p,a,b)=>clamp((p-a)/(b-a),0,1);
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
/* each token's vector, drawn as a strip of 12 cells: fixed illustrative values in (−1, 1) */
const VEC=TOKS.map((t,i)=>{const r=mulberry(11+i*7); return Array.from({length:12},()=>Math.round((r()*2-1)*100)/100);});
/* the hero's MLP: each token's widened vector before SiLU, 42 cells (3.5 × 12, as 14,336 is 3.5 × 4,096): fixed illustrative values */
const MLPW=TOKS.map((t,j)=>{const r=mulberry(101+j*13); return Array.from({length:42},()=>Math.round((r()*2.8-1.4)*100)/100);});
const softmax=z=>{const mx=Math.max(...z), e=z.map(v=>Math.exp(v-mx)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s);};
const sig=x=>1/(1+Math.exp(-x));
const silu=x=>x*sig(x);
/* four heads with recognisably different illustrative score patterns (pre-softmax), row = query token, column = key token */
const HEADS=[
 {k:'prev',en:'previous token',zh:'前一个 token',S:(i,j)=>j===i-1?3.2:(j===i?1.2:0)},
 {k:'fr',en:'“ is” → “ France”',zh:'“ is” → “ France”',S:(i,j)=>(i===4&&j===3?3.6:(i===3&&j===1?2.8:(i===2&&j===1?1.6:(j===i?0.8:0.2*j))))},
 {k:'sink',en:'first token',zh:'第一个 token',S:(i,j)=>j===0?2.8:(j===i?0.6:0)},
 {k:'broad',en:'broad',zh:'均匀',S:(i,j)=>0.3*Math.sin(i*1.7+j*2.3)}
];
const scoreM=h=>TOKS.map((_,i)=>TOKS.map((_,j)=>Math.round(HEADS[h].S(i,j)*10)/10));
const weightM=(h,mask)=>scoreM(h).map((row,i)=>{const z=row.map((v,j)=>(mask&&j>i)?-Infinity:v); return softmax(z);});
/* whole percentages that add up to exactly 100 (largest remainder), for the hero's attention weights */
function pctInt(p){const u=p.map(v=>v*100), fl=u.map(Math.floor); let left=100-fl.reduce((a,b)=>a+b,0);
  const order=p.map((v,i)=>i).sort((a,b)=>(u[b]-fl[b])-(u[a]-fl[a])||a-b);
  for(const i of order){if(left<=0) break; fl[i]++; left--;}
  return fl;}
/* percentages that add up to exactly 100.0 (largest remainder at 0.1%) */
function pcts(p){const u=p.map(v=>v*1000), fl=u.map(Math.floor); let left=1000-fl.reduce((a,b)=>a+b,0);
  const order=p.map((v,i)=>i).sort((a,b)=>(u[b]-fl[b])-(u[a]-fl[a])||a-b);
  for(const i of order){if(left<=0) break; fl[i]++; left--;}
  return fl.map(v=>v/10);}

/* ---------- words: the kit's definition (A.terms), then the formula and where it is used (R0.1: all of it on hover) ---------- */
const TIPS={
 tok:{d:()=>A.terms.token(),f:'\\text{text} \\to [791, 6864, 315, 9822, 374]',u:()=>tx('Byte-pair encoding (BPE) tokenizers: GPT, Llama 3 (128,256 tokens), Mistral, Qwen.','BPE（byte-pair encoding）分词器：GPT、Llama 3（128,256 个 token）、Mistral、Qwen。')},
 emb:{d:()=>A.terms.embedding(),f:'x = E[\\mathrm{id}]',u:()=>tx('Similar tokens end up with similar rows. Every LLM starts this way; some (Llama-3.2-1B) reuse the table as the LM head.','意思相近的 token，行也相近。每个 LLM 都从这里开始；有的模型（Llama-3.2-1B）把这张表同时当 LM head 用。')},
 norm:{d:()=>A.terms.rmsnorm(),f:'\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\text{mean}(x^2) + \\epsilon}} \\odot g',u:()=>tx('Used by Llama, Mistral, Qwen, Gemma; cheaper than LayerNorm (no mean to subtract). Without it, vector sizes drift across layers.','Llama、Mistral、Qwen、Gemma 都用它；比 LayerNorm 省（不用减均值）。没有它，向量的大小会一层层漂移。')},
 attn:{d:()=>A.terms.attention(),f:'\\mathrm{softmax}(\\frac{Q K^T}{\\sqrt{d_{head}}} + \\text{mask}) \\, V',u:()=>tx('Every Transformer LLM. The only component where tokens exchange information.','所有 Transformer LLM 都有。唯一让 token 之间交换信息的部件。')},
 mlp:{d:()=>A.terms.mlp(),f:'W_{down} (\\mathrm{SiLU}(W_{gate} x) \\odot W_{up} x)',u:()=>tx('SwiGLU in Llama, Mistral, Qwen, PaLM. Most of a model’s parameters live here: see the lab Model parameters.','Llama、Mistral、Qwen、PaLM 用的 SwiGLU。模型的大部分参数在这里：见“模型参数”Lab。')},
 res:{d:()=>A.terms.residual(),f:'h = x + f(\\mathrm{RMSNorm}(x))',u:()=>tx('Every deep network since ResNet (2015). It lets 100+ layers train: the signal and the gradient have a straight path.','ResNet（2015）以来的深层网络都用它。有了它，100 多层也能训练：信号和梯度都有一条直路。')},
 layer:{d:()=>A.terms.layer(),f:'h \\leftarrow h + \\mathrm{Attn}(\\mathrm{Norm}(h)) \\quad h \\leftarrow h + \\mathrm{MLP}(\\mathrm{Norm}(h))',u:()=>tx('Llama-3.2-1B has 16 layers, Llama-3.1-405B has 126: the same shape repeated.','Llama-3.2-1B 有 16 层，Llama-3.1-405B 有 126 层：同一个形状重复叠起来。')},
 head:{d:()=>A.terms.lmhead(),f:'\\text{logits} = W_{LM} \\, \\mathrm{RMSNorm}(h_{last})',u:()=>tx('One logit per vocabulary token: 128,256 for Llama 3.','词表里每个 token 一个 logit：Llama 3 是 128,256 个。')},
 prob:{d:()=>A.terms.softmax(),f:'p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}',u:()=>tx('T is the temperature: chat apps sample with T ≈ 0.7; T → 0 is greedy decoding.','T 是温度：聊天应用常用 T ≈ 0.7 采样；T → 0 就是贪心解码。')},
 next:{d:()=>tx('The next token is picked from the probabilities, appended to the text, and the whole model runs again: one token at a time.','下一个 token 从概率里选出，接到文字后面，整个模型再跑一遍：一次一个 token。'),f:'\\text{text} \\leftarrow \\text{text} + \\text{pick}(p)',u:()=>tx('This loop is why generation is sequential; the KV cache saves redoing the earlier tokens (lab Why a KV cache).','这个循环让生成只能一个一个来；KV cache 省掉重算前面的 token（见“为什么要 KV cache”Lab）。')},
 q:{d:()=>tx('Query: what this token is looking for.','query：这个 token 在找什么。'),f:'q = W_q \\, x',u:()=>tx('Compared with every key the token can see.','拿去和它能看到的每个 key 比较。')},
 k:{d:()=>tx('Key: what this token offers to be found by.','key：这个 token 拿什么让别人找到它。'),f:'k = W_k \\, x',u:()=>tx('Stored in the KV cache during generation.','生成时存进 KV cache。')},
 v:{d:()=>tx('Value: what this token hands over to a head that attends to it.','value：一个 head 注意到这个 token 时，它交出去的信息。'),f:'v = W_v \\, x',u:()=>tx('Also stored in the KV cache.','也存进 KV cache。')},
 qkv:{d:()=>A.terms.qkv(),f:'q = W_q x, \\quad k = W_k x, \\quad v = W_v x',u:()=>tx('Llama shares each K and V among several heads (GQA): see the lab GQA and MLA.','Llama 让几个 head 共用一份 K 和 V（GQA）：见“GQA 和 MLA”Lab。')},
 heads:{d:()=>A.terms.mha(),f:'\\mathrm{concat}(\\mathrm{head}_1, \\ldots, \\mathrm{head}_h) \\, W_o',u:()=>tx('Llama-3.1-8B runs 32 heads per layer, Llama-3.1-405B 128.','Llama-3.1-8B 每层 32 个 head，Llama-3.1-405B 128 个。')},
 score:{d:()=>tx('A score says how well a query matches a key: their dot product, divided by the square root of d_head to keep it in a stable range.','得分表示 query 和 key 有多匹配：两者的点积，再除以 d_head 的平方根，让数值保持在稳定的范围。'),f:'s_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d_{head}}}',u:()=>tx('d_head = 128 in Llama 3.','Llama 3 的 d_head = 128。')},
 mask:{d:()=>A.terms.mask(),f:'s_{ij} = -\\infty \\text{ where } j > i',u:()=>tx('Every decoder-only LLM (GPT, Llama). Encoders such as BERT are bidirectional.','所有 decoder-only LLM（GPT、Llama）都用。BERT 这样的 encoder 是双向的。')},
 softmax:{d:()=>A.terms.softmax(),f:'w_j = \\frac{e^{s_j}}{\\sum_k e^{s_k}}',u:()=>tx('Used in every attention head and on the logits.','每个 attention head 里都用，logit 上也用。')},
 aw:{d:()=>A.terms.aweight(),f:'\\mathrm{out} = \\sum_j w_j \\, v_j',u:()=>tx('The values here are illustrative.','这里的数值是示意的。')},
 wo:{d:()=>tx('Wo mixes the heads’ joined outputs back into one vector of d numbers. This is where the heads meet.','Wo 把各个 head 拼起来的输出混回一个 d 维向量。各个 head 在这里才汇合。'),f:'\\mathrm{out} = \\mathrm{concat}(h_1, \\ldots, h_n) \\, W_o',u:()=>tx('One Wo per layer, d × d.','每层一个 Wo，d × d。')},
 silu:{d:()=>tx('SiLU (swish) lets large positive numbers through and nearly blocks large negative ones.','SiLU（swish）让大的正数通过，把大的负数几乎挡掉。'),f:'\\mathrm{SiLU}(x) = x \\cdot \\sigma(x) = \\frac{x}{1 + e^{-x}}',u:()=>tx('The gate of SwiGLU in Llama, Mistral, Qwen.','Llama、Mistral、Qwen 里 SwiGLU 的门。')},
 gate:{d:()=>tx('gate and up each widen the vector to the intermediate size; SiLU(gate) then scales each number of up.','gate 和 up 各把向量放大到中间维度；SiLU(gate) 再给 up 的每一维乘一个系数。'),f:'\\mathrm{SiLU}(W_{gate} x) \\odot W_{up} x',u:()=>tx('Llama-3.1-8B widens 4,096 to 14,336 (3.5×).','Llama-3.1-8B 从 4,096 放大到 14,336（3.5 倍）。')},
 down:{d:()=>tx('down narrows the widened vector back to d, so it can be added to the residual stream.','down 把放大后的向量变回 d 维，才能加回残差流。'),f:'y = W_{down} \\, (\\mathrm{SiLU}(W_{gate} x) \\odot W_{up} x)',u:()=>tx('The third matrix of the MLP block.','MLP 块的第三个矩阵。')},
 temp:{d:()=>tx('Temperature divides the logits before softmax: below 1 it sharpens the probabilities, above 1 it flattens them.','温度先除 logit 再做 softmax：小于 1 让概率更集中，大于 1 让它更平。'),f:'p = \\mathrm{softmax}(z / T)',u:()=>tx('Chat apps often use T ≈ 0.7; code models lower.','聊天应用常用 T ≈ 0.7；写代码的更低。')},
 rope:{d:()=>tx('RoPE (rotary position embedding) turns each query and key, not the value, by an angle set by the token’s position, inside every attention layer. A score then depends on the content and on the relative position; for the same content, only the distance m − n matters.','RoPE（旋转位置编码）在每一层的注意力里，按 token 的位置把 query 和 key 转一个角度（value 不转）。得分同时取决于内容和相对位置；内容不变时，位置的影响只取决于距离 m − n。'),f:"q'_m = R(m \\theta) \\, q_m, \\quad q'_m \\cdot k'_n = q_m \\cdot R((n - m) \\theta) \\, k_n",u:()=>tx('Llama, Mistral, Qwen. It adds no parameters.','Llama、Mistral、Qwen 都用。它没有参数。')}
};
/* formulas are TeX (the kit's subset), shown as MathML in tips; texPlain gives a readable text form for the copy deck and aria */
const TEXP={frac:0,sqrt:0,text:0,mathrm:0,sum:'Σ',cdot:'·',odot:'⊙',oplus:'⊕',times:'×',div:'÷',approx:'≈',to:'→',leftarrow:'←',infty:'∞',ldots:'…',epsilon:'ε',sigma:'σ',theta:'θ',quad:'  '};
function texPlain(src){const s=String(src); let i=0; const SUP={'2':'²','T':'ᵀ'};
  const grp=()=>{if(s[i]==='{'){i++; const r=seq('}'); i++; return r;} return atom();};
  function atom(){const c=s[i]; if(c==='\\'){i++; let n=''; while(i<s.length&&/[A-Za-z]/.test(s[i])) n+=s[i++]; if(!n){i++; return ' ';}
      if(n==='frac'){const x=grp(), y=grp(); return `${/^[\w.]+$/.test(x)?x:'('+x+')'} ÷ ${/^[\w.]+$/.test(y)?y:'('+y+')'}`;}
      if(n==='sqrt') return `√(${grp()})`; if(n==='text'||n==='mathrm') return grp(); const g=TEXP[n]; return typeof g==='string'?g:n;}
    if(c==='{'){i++; const r=seq('}'); i++; return r;} i++; return c;}
  function seq(end){let out=''; while(i<s.length&&s[i]!==end){if(s[i]==='^'||s[i]==='_'){const op=s[i++], g=grp(); out+=op==='^'&&SUP[g]?SUP[g]:op+(op==='^'&&g.length>1?`(${g})`:g);} else out+=atom();} return out;}
  return seq(undefined).replace(/\s+/g,' ').trim();}
const tipHTML=(k,act)=>{const t=TIPS[k]; if(!t) return ''; return `<div>${esc(t.d())}</div><div class="fx">${A.ui.math(t.f)}</div><div class="ar-tu">${esc(t.u())}</div>`+(act?`<div class="act">${esc(act)}</div>`:'');};
const tipText=k=>{const t=TIPS[k]; return t?`${t.d()} ${tx('Formula','公式')}: ${texPlain(t.f)}. ${t.u()}`:'';};
/* a term inside a pop-up or the rail: definition + formula + use on hover (kit .t, handled by termTip) */
const T=(k,txt)=>`<span class="t ar-term" tabindex="0" role="button" data-ctl="term" data-term="${k}">${txt}</span>`;

/* ---------- drawing kit: colours from CSS variables, crisp canvases, lanes and particles ---------- */
let COL=null, colT=0, colK='';
function cols(){const now=performance.now(), k=(document.documentElement.dataset.theme||'')+(window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'d':'l');
  if(COL&&now-colT<800&&k===colK) return COL; colK=k;
  const cs=getComputedStyle(document.documentElement), g=n=>cs.getPropertyValue(n).trim();
  COL={ink:g('--ink'),ink2:g('--ink-2'),ink3:g('--ink-3'),line:g('--line'),line2:g('--line-2'),surf:g('--surface'),surf2:g('--surface-2'),sunk:g('--sunken'),bg:g('--bg'),
    w:g('--c-weights'),q:g('--ar-q')||'#C8367E',k:g('--c-kv'),v:g('--ar-v')||'#9A5B00',att:g('--ar-att')||'#0E8C8C',tok:[0,1,2,3,4].map(i=>g('--ar-t'+i)||'#888'),
    font:g('--font-ui')||'system-ui,sans-serif',mono:g('--font-mono')||'monospace'};
  colT=now; return COL;}
const rgb=h=>{h=String(h).replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join(''); const n=parseInt(h,16)||0; return [n>>16&255,n>>8&255,n&255];};
const rgba=(h,a)=>{const [r,g,b]=rgb(h); return `rgba(${r},${g},${b},${a})`;};
const mix=(h1,h2,t)=>{const a=rgb(h1),b=rgb(h2); return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;};
function fit(cv){const r=cv.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1), w=Math.max(1,Math.round(r.width)), h=Math.max(1,Math.round(r.height));
  if(cv._w!==w||cv._h!==h||cv._d!==dpr){cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr); cv._w=w; cv._h=h; cv._d=dpr;}
  const c=cv.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); return {c,W:w,H:h};}
function rr(c,x,y,w,h,r){c.beginPath(); if(!(w>0&&h>0)) return; r=Math.max(0,Math.min(r,w/2,h/2)); c.moveTo(x+r,y); c.arcTo(x+w,y,x+w,y+h,r); c.arcTo(x+w,y+h,x,y+h,r); c.arcTo(x,y+h,x,y,r); c.arcTo(x,y,x+w,y,r); c.closePath();}
/* a vector: a strip of cells tinted by its token, darker = larger |value| */
function strip(c,x,y,cw,ch,vals,col,alpha,n){n=n||vals.length; alpha=alpha==null?1:alpha;
  for(let i=0;i<n;i++){const v=vals[i%vals.length]; c.fillStyle=rgba(col,alpha*(0.22+0.78*Math.abs(v))); c.fillRect(x+i*cw,y,cw-1,ch);}}
/* paths: sampled polylines, a point at an arc length */
function bez(p0,p1,p2,p3,n,out){out=out||[]; for(let i=out.length?1:0;i<=n;i++){const t=i/n,u=1-t; out.push([u*u*u*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*p3[0],u*u*u*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*p3[1]]);} return out;}
function mkPath(pts){const len=[0]; for(let i=1;i<pts.length;i++) len.push(len[i-1]+Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1])); return {pts,len,L:len[len.length-1]};}
function at(P,s){const l=P.len; let a=0,b=l.length-1; if(s<=0) return P.pts[0]; if(s>=P.L) return P.pts[b];
  while(b-a>1){const m=(a+b)>>1; if(l[m]<s) a=m; else b=m;} const t=(s-l[a])/((l[b]-l[a])||1); return [lerp(P.pts[a][0],P.pts[b][0],t),lerp(P.pts[a][1],P.pts[b][1],t)];}
function strokePts(c,pts){c.beginPath(); c.moveTo(pts[0][0],pts[0][1]); for(let i=1;i<pts.length;i++) c.lineTo(pts[i][0],pts[i][1]); c.stroke();}
function dots(c,P,t,col,o){o=o||{}; const sp=o.sp||56, v=o.v||60, r=o.r||2.4, ph=o.ph||0; if(P.L<4) return;
  const R=c.__rec; const off=((t/1000)*v+ph)%sp; for(let s=off;s<P.L;s+=sp){const p=at(P,s); if(o.hide&&o.hide(p)) continue; if(R&&(!o.alpha||o.alpha(p)>0.1)) R.dots.push([+p[0].toFixed(1),+p[1].toFixed(1),r*1.6]);
    const a=o.alpha?o.alpha(p):1; if(a<=0) continue; c.globalAlpha=a*0.35; c.fillStyle=col; c.beginPath(); c.arc(p[0],p[1],r*2.1,0,7); c.fill();
    c.globalAlpha=a; c.beginPath(); c.arc(p[0],p[1],r,0,7); c.fill();} c.globalAlpha=1;}
function arrowHead(c,x,y,ang,s,col){c.fillStyle=col; c.beginPath(); c.moveTo(x,y); c.lineTo(x-s*Math.cos(ang-0.45),y-s*Math.sin(ang-0.45)); c.lineTo(x-s*Math.cos(ang+0.45),y-s*Math.sin(ang+0.45)); c.closePath(); c.fill();}
function pill(c,x,y,txt,bg,fg,font,padX,h){c.font=font; const w=c.measureText(txt).width+2*(padX||7); rr(c,x,y,w,h||20,(h||20)/2); c.fillStyle=bg; c.fill(); c.fillStyle=fg; c.textBaseline='middle'; c.fillText(txt,x+(padX||7),y+(h||20)/2+0.5); return w;}
const tokTxt=t=>t.replace(/^ /,'␣');
/* a render loop that runs only while its host is on screen, the tab is visible and motion is allowed; otherwise one still frame */
function makeLoop(host,draw){
  const S={on:false,vis:true,raf:0};
  const tick=t=>{S.raf=0; if(!S.on) return; draw(t,false); S.raf=requestAnimationFrame(tick);};
  S.sync=()=>{const run=S.vis&&!document.hidden&&!A.reduceMotion()&&host.isConnected;
    if(run&&!S.on){S.on=true; S.raf=requestAnimationFrame(tick);} else if(!run&&S.on){S.on=false; cancelAnimationFrame(S.raf); S.raf=0;}
    if(!run&&host.isConnected) draw(performance.now(),true);};
  S.io=new IntersectionObserver(es=>{S.vis=es.some(e=>e.isIntersecting); S.sync();},{threshold:0}); S.io.observe(host);
  S.onVis=()=>S.sync(); document.addEventListener('visibilitychange',S.onVis);
  const mq=window.matchMedia?matchMedia('(prefers-reduced-motion: reduce)'):null; S.onMq=()=>S.sync(); if(mq&&mq.addEventListener) mq.addEventListener('change',S.onMq);
  /* a theme change: a still picture is drawn again in the new colours (a running one takes them on its next frame) */
  const md=window.matchMedia?matchMedia('(prefers-color-scheme: dark)'):null; if(md&&md.addEventListener) md.addEventListener('change',S.onMq);
  S.mo=new MutationObserver(S.onMq); S.mo.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  S.kill=()=>{S.on=false; cancelAnimationFrame(S.raf); S.io.disconnect(); S.mo.disconnect(); document.removeEventListener('visibilitychange',S.onVis); if(mq&&mq.removeEventListener) mq.removeEventListener('change',S.onMq); if(md&&md.removeEventListener) md.removeEventListener('change',S.onMq);};
  S.sync(); return S;}

/* ---------- geometry recorder for the layout audit (tests/architecture_layout.py): the same drawing code runs on a context that also
   remembers every text box, stroked path and particle, in canvas coordinates ---------- */
function recorder(W,H){
  const cv=document.createElement('canvas'); cv.width=Math.max(1,W); cv.height=Math.max(1,H); const real=cv.getContext('2d');
  const G={texts:[],lines:[],dots:[],fills:[],_panel:-1,_tag:'',n:0}; let m=[1,0,0,1,0,0], path=[], sub=null, last=[0,0]; const stack=[];
  const T=(x,y)=>[m[0]*x+m[2]*y+m[4],m[1]*x+m[3]*y+m[5]];
  const mul=(a,b)=>[a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3],a[0]*b[4]+a[2]*b[5]+a[4],a[1]*b[4]+a[3]*b[5]+a[5]];
  const pt=(x,y)=>{last=[x,y]; const q=T(x,y); if(!sub){sub=[q]; path.push(sub);} else sub.push(q);};
  const api={
    save(){stack.push({m:m.slice(),p:G._panel}); real.save();}, restore(){const o=stack.pop(); if(o){m=o.m;} real.restore();},
    translate(x,y){m=mul(m,[1,0,0,1,x,y]); real.translate(x,y);}, scale(a,b){m=mul(m,[a,0,0,b,0,0]); real.scale(a,b);},
    rotate(r){const c=Math.cos(r),s=Math.sin(r); m=mul(m,[c,s,-s,c,0,0]); real.rotate(r);},
    setTransform(a,b,c,d,e,f){m=[a,b,c,d,e,f]; real.setTransform(a,b,c,d,e,f);},
    beginPath(){path=[]; sub=null; real.beginPath();},
    moveTo(x,y){sub=null; pt(x,y); real.moveTo(x,y);}, lineTo(x,y){pt(x,y); real.lineTo(x,y);},
    bezierCurveTo(a,b,c2,d,x,y){const p0=last; for(let i=1;i<=12;i++){const t=i/12,u=1-t; pt(u*u*u*p0[0]+3*u*u*t*a+3*u*t*t*c2+t*t*t*x,u*u*u*p0[1]+3*u*u*t*b+3*u*t*t*d+t*t*t*y);} last=[x,y]; real.bezierCurveTo(a,b,c2,d,x,y);},
    arc(x,y,r,a0,a1){for(let i=0;i<=16;i++){const a=a0+(a1-a0)*i/16; pt(x+r*Math.cos(a),y+r*Math.sin(a));} real.arc(x,y,r,a0,a1);},
    ellipse(x,y,rx,ry,ro,a0,a1){for(let i=0;i<=16;i++){const a=a0+(a1-a0)*i/16; pt(x+rx*Math.cos(a),y+ry*Math.sin(a));} real.ellipse(x,y,rx,ry,ro,a0,a1);},
    arcTo(x1,y1,x2,y2,r){pt(x1,y1); real.arcTo(x1,y1,x2,y2,r);},
    rect(x,y,w,h){sub=null; pt(x,y); pt(x+w,y); pt(x+w,y+h); pt(x,y+h); pt(x,y); real.rect(x,y,w,h);},
    closePath(){if(sub&&sub.length) sub.push(sub[0].slice()); real.closePath();},
    stroke(){if(real.globalAlpha>0.05) path.forEach(q=>{if(q.length>1) G.lines.push({pts:q.map(v=>[+v[0].toFixed(1),+v[1].toFixed(1)]),w:real.lineWidth,panel:G._panel,tag:G._tag,seq:++G.n});}); real.stroke();},
    strokeRect(x,y,w,h){const a=[T(x,y),T(x+w,y),T(x+w,y+h),T(x,y+h)]; a.push(a[0]); if(real.globalAlpha>0.05) G.lines.push({pts:a,w:real.lineWidth,panel:G._panel,tag:G._tag,seq:++G.n}); real.strokeRect(x,y,w,h);},
    fillText(t,x,y){const w=real.measureText(t).width, fs=parseFloat((real.font.match(/([\d.]+)px/)||[0,12])[1]), al=real.textAlign, bl=real.textBaseline;
      const x0=al==='center'?x-w/2:(al==='right'||al==='end')?x-w:x, y0=bl==='middle'?y-fs*0.5:bl==='top'?y:bl==='bottom'?y-fs:y-fs*0.78;
      const cs=[T(x0,y0),T(x0+w,y0),T(x0,y0+fs),T(x0+w,y0+fs)], xs=cs.map(v=>v[0]), ys=cs.map(v=>v[1]);
      if(real.globalAlpha>0.25&&String(t).trim()) G.texts.push({t:String(t),x0:Math.min(...xs),y0:Math.min(...ys),x1:Math.max(...xs),y1:Math.max(...ys),panel:G._panel,tag:G._tag,seq:++G.n});
      real.fillText(t,x,y);},
    fill(){if(real.globalAlpha>0.1){const fs=real.fillStyle, fa=typeof fs==='string'?(/^rgba/.test(fs)?parseFloat(fs.split(',')[3]):1):0, op=fa*real.globalAlpha>=0.95;
        path.forEach(q=>{if(q.length<3) return; const xs=q.map(v=>v[0]), ys=q.map(v=>v[1]); G.fills.push({x0:Math.min(...xs),y0:Math.min(...ys),x1:Math.max(...xs),y1:Math.max(...ys),panel:G._panel,tag:G._tag,op,seq:++G.n});});} real.fill();},
    fillRect(x,y,w,h){const a=[T(x,y),T(x+w,y+h)]; if(real.globalAlpha>0.1&&w>0&&h>0) G.fills.push({x0:Math.min(a[0][0],a[1][0]),y0:Math.min(a[0][1],a[1][1]),x1:Math.max(a[0][0],a[1][0]),y1:Math.max(a[0][1],a[1][1]),panel:G._panel,tag:G._tag,r:1,seq:++G.n}); real.fillRect(x,y,w,h);},
    clearRect(){}, measureText(t){return real.measureText(t);}
  };
  return new Proxy(api,{get(o,k){if(k==='__rec') return G; if(k in o) return o[k]; const v=real[k]; return typeof v==='function'?v.bind(real):v;},
    set(o,k,v){if(k==='__panel'){G._panel=v; return true;} if(k==='__tag'){G._tag=v; return true;} real[k]=v; return true;}});
}
/* ======================================================================================================================
   HOME HERO: six stages left to right; one is expanded (its inside animated), the others are previews; particles keep flowing
   ====================================================================================================================== */
/* the stage names use the terms the lab's primer introduces (the owner's review of 2026-10-11 18:46 UTC: titles name the real terms) */
const HST=[{k:'tok',t:{en:'Tokenization',zh:'token化'}},{k:'emb',t:{en:'Embedding',zh:'embedding'}},{k:'attn',t:{en:'Attention',zh:'注意力'}},
  {k:'mlp',t:{en:'MLP',zh:'MLP'}},{k:'prob',t:{en:'Probabilities',zh:'概率'}},{k:'next',t:{en:'Next token',zh:'下一个 token'}}];
const HDESC={tok:{en:'The sentence is cut into tokens.',zh:'句子被切成 token。'},emb:{en:'Each token becomes a vector.',zh:'每个 token 变成一个向量。'},
  attn:{en:'Through attention, each token takes information from itself and the tokens before it.',zh:'每个 token 借助注意力，从它自己和前面的 token 取信息。'},
  mlp:{en:'Each token’s vector goes through the MLP on its own.',zh:'每个 token 的向量单独经过 MLP，再加工一次。'},
  prob:{en:'The last vector becomes the probabilities of the next token.',zh:'最后一个向量变成下一个 token 的概率。'},next:{en:'The chosen token joins the sentence; repeat.',zh:'选出的 token 接到句子后面，再来一遍。'}};
/* the hero plays on its own, with nothing to press (the owner, 2026-10-11 19:28 UTC); HNOTE stays for the tests' record of past wording */
const HNOTE={en:'',zh:''};
const HT=4200, HT_IDLE=10000;                                 // one stage's animation; the pause after the visitor acts
/* self-attention goes token by token and the MLP widens, then narrows each vector, so those two stages run longer (2026-10-02) */
const HTS={attn:8400,mlp:6000}, htOf=i=>HTS[HST[i].k]||HT;
const PROB5=softmax(Z0);
const AW_IS=weightM(1,true)[4];                               // “ is” in the head that looks at “ France”
const hero={cur:0,fx:0,tr:null,t0:0,last:0,k:0,user:-1e9,hover:false,loop:null,el:null,cv:null,btns:[],narrow:false,flyT:-1};
function heroGo(i,user){i=(i+6)%6; const now=performance.now(); if(user) hero.user=now;
  if(i!==hero.cur){hero.tr={a:hero.fx,b:i,t0:now}; hero.cur=i; hero.t0=now; hero.last=now; hero.k=0;}
  heroSync(); if(hero.loop&&!hero.loop.on) hero.loop.sync();}
function heroSync(){if(!hero.el) return; hero.btns.forEach((b,i)=>{b.classList.toggle('on',i===hero.cur);});
  const d=$('.ar-hdesc',hero.el); if(d) d.textContent=L(HDESC[HST[hero.cur].k]);}
function heroLayout(W,H,fx){
  const top=hero.narrow?6:28, y=top, h=H-top-(hero.narrow?6:40);
  if(hero.narrow){const x=14, w=W-28; return {st:HST.map((s,i)=>({x,y,w,h,e:i===hero.cur?1:0})),top};}
  const gaps=[34,44,20,44,34], sumG=gaps.reduce((a,b)=>a+b,0), wp=clamp((W-sumG)*0.093,80,112), we=W-sumG-5*wp;
  let x=0; const st=HST.map((s,i)=>{const e=Math.max(0,1-Math.abs(i-fx)), w=wp+(we-wp)*e, r={x,y,w,h,e}; x+=w+(gaps[i]||0); return r;});
  return {st,top,br:{x:st[2].x-12,y:y-5,w:st[3].x+st[3].w-st[2].x+24,h:h+10}};}
/* the loop back to the input: down from the next-token panel, along the bottom 13 px under the stack's lowest plate, up into
   the tokenization panel (review 12: it must keep ≥ 12 px from every frame of the stack) */
function heroLoop(st,br,H){const a=st[5], z=st[0], ax=a.x+a.w/2, zx=z.x+z.w/2, y0=a.y+a.h+2, y1=z.y+z.h+2, pb=br?br.y+br.h+12:a.y+a.h, yL=Math.min(H-3,Math.max(a.y+a.h+16,pb+13)), rc=10;
  const P=[[ax,y0],[ax,yL-rc]]; bez([ax,yL-rc],[ax,yL],[ax,yL],[ax-rc,yL],6,P); P.push([zx+rc,yL]); bez([zx+rc,yL],[zx,yL],[zx,yL],[zx,yL-rc],6,P); P.push([zx,y1]); return P;}
/* the open next-token panel, centred: the five probabilities, a gap, the picked token (and its ID on wide panels) */
function nextGeo(r){const sm=r.w<520, bw=sm?Math.min(72,r.w*0.25):Math.min(130,r.w*0.24), gap=sm?46:96, tw=sm?64:84, tot=bw+gap+tw+(sm?0:86), x0=r.x+Math.max(16,(r.w-tot)/2),
    top=r.y+HDR+(sm?30:36), bot=r.y+r.h-(sm?34:14); return {sm,bw,x0,cx:x0+bw+gap+tw/2,cy:(top+bot)/2+(sm?0:6)};}
/* where a lane crosses a stage: previews bundle the five lanes, the expanded stage spreads them over its rows */
const HDR=30;
function laneY(r,k){const ye=r.y+HDR+(k+0.5)*(r.h-HDR-10)/5, yp=r.y+r.h*0.55+(k-2)*14; return lerp(yp,ye,r.e);}
function heroDraw(t,still){
  const cv=hero.cv; if(!cv||!cv.isConnected) return;
  const pw=cv.parentNode.getBoundingClientRect().width; if(pw<60) return;
  const nar=pw<860;
  if(nar!==hero.narrow){hero.narrow=nar; const box=$('.ar-hero2',hero.el); if(box) box.classList.toggle('narrow',nar);}
  const {c,W,H}=fit(cv); c.clearRect(0,0,W,H);
  /* the hero moves on by itself only where a pass of the current stage ends, so a pass is never cut off: after at least one pass, 10 s
     after the visitor's last click, and not while the pointer is on it (owner, 2026-10-04: on the first visit, the pointer crossing
     the hero reset the stage clock, and the tokenization pass restarted and was cut after about a second) */
  if(!still){const ht=htOf(hero.cur), k=Math.floor((t-hero.t0)/ht), ended=k>hero.k; hero.k=k;
    if(ended&&!hero.hover&&t>=Math.max(hero.last+ht,hero.user+HT_IDLE)) heroGo(hero.cur+1,false);}
  heroPaint(c,W,H,t,still,false);}
function heroPaint(c,W,H,t,still,rec){
  const C=cols();
  if(hero.tr){const p=Math.min(1,(t-hero.tr.t0)/420); hero.fx=lerp(hero.tr.a,hero.tr.b,ease(p)); if(p>=1||still){hero.fx=hero.tr.b; hero.tr=null;}}
  const lay=heroLayout(W,H,hero.fx), st=lay.st, font=C.font;
  if(!rec){if(!hero.narrow) hero.btns.forEach((b,i)=>{const r=st[i], s=`left:${r.x}px;top:${r.y}px;width:${r.w}px;height:${r.h}px`; if(b._s!==s){b.style.cssText=s; b._s=s;}});
  else hero.btns.forEach(b=>{if(b._s){b.style.cssText=''; b._s='';}});}
  const ht=htOf(hero.cur), p=still?0.999:((t-hero.t0)%ht)/ht;
  /* the stack of layers behind self-attention and MLP */
  if(lay.br){const b=lay.br; c.__tag='stack'; for(let i=3;i>=1;i--){rr(c,b.x+i*5,b.y+i*4,b.w,b.h,12); c.strokeStyle=rgba(C.ink3,0.16+0.05*(3-i)); c.lineWidth=1; c.stroke();}
    rr(c,b.x,b.y,b.w,b.h,12); c.fillStyle=rgba(C.w,0.035); c.fill(); c.setLineDash([4,4]); c.strokeStyle=rgba(C.ink3,0.55); c.stroke(); c.setLineDash([]); c.__tag='';
    c.font=`600 11px ${C.mono}`; c.fillStyle=C.ink2; c.textBaseline='middle'; const lab=tx('repeated 32 times','重复 32 遍'), lw=c.measureText(lab).width;
    rr(c,b.x+b.w/2-lw/2-8,b.y-23,lw+16,18,9); c.fillStyle=C.surf; c.fill(); c.strokeStyle=rgba(C.ink3,0.5); c.stroke(); c.fillStyle=C.ink2; c.fillText(lab,b.x+b.w/2-lw/2,b.y-14);}
  /* the loop: the next token goes back to the input, in straight runs with rounded corners, clear of the layer stack (≥ 12 px) */
  let flyR=null;
  if(!hero.narrow){const a=st[5], z=st[0], lp=heroLoop(st,lay.br,H), zx=z.x+z.w/2;
    if(HST[hero.cur].k==='next'){const fly=ease(seg01(still?0:p,0.4,0.9));
      if(fly>0&&fly<1){const g=nextGeo(a), ax=a.x+a.w/2, yb=a.y+a.h+2, pts=bez([g.cx,g.cy],[g.cx,g.cy+40],[ax,yb-30],[ax,yb],14); lp.slice(1).forEach(v=>pts.push(v));
        bez([zx,z.y+z.h+2],[zx,z.y+z.h-18],[zx,laneY(z,4)+40],[zx,laneY(z,4)+26],10,pts); const P=mkPath(pts), q=at(P,P.L*fly);
        c.font=`600 12px ${C.font}`; const tw=c.measureText(NEXT.t.trim()).width+16; flyR={x:q[0]-tw/2,y:q[1]-10,w:tw,h:20,q};}}
    c.strokeStyle=rgba(C.ink3,0.4); c.setLineDash([3,5]); c.lineWidth=1.2; const runs=[[]];
    lp.forEach(v=>{const inside=flyR&&v[0]>flyR.x-4&&v[0]<flyR.x+flyR.w+4&&v[1]>flyR.y-4&&v[1]<flyR.y+flyR.h+4; if(inside){if(runs[runs.length-1].length) runs.push([]);} else runs[runs.length-1].push(v);});
    c.__tag='loop'; runs.forEach(r=>{if(r.length>1) strokePts(c,r);}); c.__tag=''; c.setLineDash([]); arrowHead(c,zx,z.y+z.h+2,-Math.PI/2,7,rgba(C.ink3,0.65));}
  /* ribbons between stages, five lanes (one after the probabilities) */
  const paths=[[],[],[],[],[]];
  if(!hero.narrow){for(let g=0;g<5;g++){const A_=st[g], B=st[g+1];
      for(let k=0;k<5;k++){if(g===4&&k!==4) continue;
        const y0=g===4?probY(A_,0):laneY(A_,k), y1=g===4?laneY(B,2):laneY(B,k), x0=A_.x+A_.w, x1=B.x, xm=(x0+x1)/2, col=g===4?C.ink:C.tok[k];
        const pts=bez([x0,y0],[xm,y0],[xm,y1],[x1,y1],18); c.strokeStyle=rgba(col,0.18); c.lineWidth=g===4?6:7; strokePts(c,pts); c.strokeStyle=rgba(col,0.7); c.lineWidth=1.3; strokePts(c,pts);
        if(!paths[k][g]) paths[k][g]=pts;}}}
  /* stage panels */
  hero.p=p;
  st.forEach((r,i)=>{const on=i===hero.cur; if(hero.narrow&&!on) return;
    if(r.e>0.02){c.save(); c.shadowColor=rgba(C.ink,0.14*r.e); c.shadowBlur=18*r.e; c.shadowOffsetY=4*r.e; rr(c,r.x,r.y,r.w,r.h,10); c.fillStyle=C.surf; c.fill(); c.restore();}
    rr(c,r.x,r.y,r.w,r.h,10); c.fillStyle=r.e>0.5?C.surf:rgba(C.surf2,0.9); c.fill(); c.lineWidth=on?1.6:1; c.strokeStyle=on?rgba(C.ink,0.55):rgba(C.ink3,0.35); c.stroke();
    c.__panel=i; const pa=1-r.e; if(pa>0.02) heroPreview(c,C,HST[i].k,r,pa,t,still);
    if(r.e>0.02){c.save(); rr(c,r.x+1,r.y+1,r.w-2,r.h-2,9); c.clip(); heroStage(c,C,HST[i].k,r,r.e,p,t,still); c.restore();} c.__panel=-1;});
  if(flyR){const q=flyR.q; c.font=`600 12px ${C.font}`; rr(c,flyR.x,flyR.y,flyR.w,flyR.h,10); c.fillStyle=C.ink; c.fill(); c.fillStyle=C.surf; c.textBaseline='middle'; c.textAlign='center'; c.fillText(NEXT.t.trim(),q[0],q[1]+1); c.textAlign='left';}
  /* particles along the ribbons and through the previews */
  if(!hero.narrow&&!still){
    for(let k=0;k<5;k++){for(let g=0;g<5;g++){const pts=paths[k][g]; if(!pts) continue; const P=mkPath(pts);
        dots(c,P,t,g===4?C.ink:mix(C.tok[k],C.ink,0.25),{sp:26,v:46,r:2,ph:k*7+g*3});}}
    st.forEach((r,i)=>{if(r.e>0.6||i>=4) return; const pa=1-r.e;
      for(let k=0;k<5;k++){const y=laneY(r,k), P=mkPath([[i===0?r.x+r.w/2+10:r.x+2,y],[r.x+r.w-2,y]]);
        dots(c,P,t,mix(C.tok[k],C.ink,0.25),{sp:26,v:46,r:2,ph:k*7+i*5,alpha:()=>pa});}});}
  if(hero.narrow){const r=st[hero.cur], kk=HST[hero.cur].k; c.font=`600 12px ${C.font}`; c.fillStyle=C.ink2; c.textBaseline='middle'; c.fillText(L(HST[hero.cur].t),r.x+12,r.y+15);
    /* lanes enter on the left and leave on the right: none enter the tokenization; one (the next token) leaves the probabilities and
       enters the next-token panel; nothing leaves it */
    const sb=[];
    for(let k=0;k<5;k++){const y=laneY(r,k), col=C.tok[k];
      if(kk!=='tok'&&kk!=='next') sb.push([0,r.x,y,col,k]);
      if(kk!=='prob'&&kk!=='next') sb.push([r.x+r.w,W,y,col,k]);}
    if(kk==='prob') sb.push([r.x+r.w,W,probY(r,0),C.ink,5]);
    if(kk==='next') sb.push([0,r.x,laneY(r,2),C.ink,5]);
    sb.forEach(([x0,x1,y,col,k])=>{c.strokeStyle=rgba(col,0.6); c.lineWidth=2; c.beginPath(); c.moveTo(x0,y); c.lineTo(x1,y); c.stroke();
      if(!still) dots(c,mkPath([[x0,y],[x1,y]]),t,mix(col,C.ink,0.25),{sp:9,v:14,r:1.6,ph:k*3+(x0>0?4:0)});});}
  return lay;
}
function probY(r,j){const ye=r.y+HDR+8+(j+0.5)*(r.h-HDR-24)/5, yp=r.y+r.h*0.55+(j-2)*14; return lerp(yp,ye,r.e);}
/* the compact previews: a glyph, lanes passing through */
function heroPreview(c,C,k,r,a,t,still,SS){SS=SS||hero;
  c.save(); c.globalAlpha=a; const cx=r.x+r.w/2, y0=r.y+HDR, live=SS===tile&&HST[SS.cur].k===k;   // live: the home tile's active panel
  const lanes=(from,to)=>{for(let j=0;j<5;j++){const y=laneY(r,j); c.strokeStyle=rgba(C.tok[j],0.75); c.lineWidth=2; c.beginPath(); c.moveTo(from,y); c.lineTo(to,y); c.stroke();}};
  if(k==='tok'){lanes(cx+10,r.x+r.w); for(let j=0;j<5;j++){const y=laneY(r,j); rr(c,cx-24,y-5,34,10,5); c.fillStyle=C.tok[j]; c.fill();}
    if(HST[SS.cur].k==='next'&&SS.p>0.9){const y=laneY(r,4)+26; rr(c,cx-26,y-8,38,16,8); c.fillStyle=C.ink; c.fill(); c.fillStyle=C.surf; c.font=`600 9px ${C.font}`; c.textBaseline='middle'; c.fillText(NEXT.t.trim(),cx-21,y+0.5);}}
  else if(k==='emb'){lanes(r.x,r.x+r.w);
    /* the table spans every lane: in the small home tile (no header) the panel is short, and a table that starts below the header
       left the first lane above it (product owner, Batch 3 review) */
    const et=Math.min(y0+6,laneY(r,0)-10), eb=Math.max(r.y+r.h-10,laneY(r,4)+10);
    rr(c,cx-13,et,26,eb-et,4); c.fillStyle=rgba(C.w,0.14); c.fill(); c.strokeStyle=rgba(C.w,0.8); c.lineWidth=1.2; c.stroke();
    c.strokeStyle=rgba(C.w,0.25); c.lineWidth=0.6; for(let y=et+4;y<eb-2;y+=4){c.beginPath(); c.moveTo(cx-11,y); c.lineTo(cx+11,y); c.stroke();}}
  else if(k==='attn'){lanes(r.x,r.x+r.w);
    /* causal at a glance: column j carries token j's information down to every later lane, and lane i holds a dot for each token it
       takes from (0…i), a lower triangle. The old crossing curves read as every token seeing every other (product owner, 2026-10-02) */
    const sp=Math.min(9,(r.w-14)/5), xs=j=>cx+(j-2)*sp;
    if(live){
      /* the tile's active panel: one token at a time (0.9 s each), as in the hero. Its row lights up with a dot for itself and each
         earlier token, the columns of those tokens run down to it, and the later tokens' places in its row are empty rings (it cannot
         see them). Reduced motion stops on the middle token, where both show */
      const q=still?2:Math.floor(t/900)%5;
      for(let j=0;j<4;j++){const on=j<q; c.strokeStyle=rgba(C.att,on?0.85:0.18); c.lineWidth=on?1.5:1; c.beginPath(); c.moveTo(xs(j),laneY(r,j)); c.lineTo(xs(j),laneY(r,on?q:4)); c.stroke();}
      for(let i=0;i<5;i++) for(let j=0;j<=i;j++){if(i===q) continue; c.fillStyle=rgba(C.att,0.22); c.beginPath(); c.arc(xs(j),laneY(r,i),1.8,0,7); c.fill();}
      for(let j=0;j<5;j++){const y=laneY(r,q);
        if(j<=q){c.fillStyle=C.att; c.beginPath(); c.arc(xs(j),y,j===q?3:2.4,0,7); c.fill();}
        else{c.strokeStyle=rgba(C.ink3,0.6); c.lineWidth=1; c.beginPath(); c.arc(xs(j),y,2.2,0,7); c.stroke();}}}
    else{
      for(let j=0;j<4;j++){c.strokeStyle=rgba(C.att,0.6); c.lineWidth=1.2; c.beginPath(); c.moveTo(xs(j),laneY(r,j)); c.lineTo(xs(j),laneY(r,4)); c.stroke();}
      for(let i=0;i<5;i++) for(let j=0;j<=i;j++){c.fillStyle=j===i?C.att:rgba(C.att,0.75); c.beginPath(); c.arc(xs(j),laneY(r,i),j===i?2.6:2.1,0,7); c.fill();}}}
  else if(k==='mlp'){lanes(r.x,r.x+r.w);
    if(live){
      /* the tile's active panel: each token's vector, a short bar, widens to 3.5 times its length (14,336 / 4,096) and narrows back,
         over 3.2 s: widen, hold, narrow, hold. Reduced motion shows it widened, with the short length marked inside */
      const ph=still?0.4:(t%3200)/3200, u=ph<0.35?ease(ph/0.35):ph<0.5?1:ph<0.85?1-ease((ph-0.5)/0.35):0;
      const w1=Math.max(24,r.w-12), w0=w1/3.5, w=lerp(w0,w1,u);
      for(let j=0;j<5;j++){const y=laneY(r,j); c.fillStyle=rgba(C.w,0.32); c.strokeStyle=rgba(C.w,0.9); c.lineWidth=1; c.fillRect(cx-w/2,y-3,w,6); c.strokeRect(cx-w/2,y-3,w,6);
        if(still){c.strokeStyle=rgba(C.w,0.9); c.beginPath(); c.moveTo(cx-w0/2,y-3); c.lineTo(cx-w0/2,y+3); c.moveTo(cx+w0/2,y-3); c.lineTo(cx+w0/2,y+3); c.stroke();}}}
    else for(let j=0;j<5;j++){const y=laneY(r,j); c.fillStyle=rgba(C.w,0.3); c.strokeStyle=rgba(C.w,0.85); c.lineWidth=1;
      const hw=Math.min(24,r.w/2-6); c.beginPath(); c.moveTo(cx-hw,y); c.lineTo(cx,y-6); c.lineTo(cx+hw,y); c.lineTo(cx,y+6); c.closePath(); c.fill(); c.stroke();}}
  else if(k==='prob'){c.__tag='bar'; for(let j=0;j<5;j++){const yy=probY(r,j), br=still?1:0.86+0.14*Math.sin(t/380+j*0.9), w=(6+Math.min(44,r.w-30)*PROB5[j]/PROB5[0])*br; c.fillStyle=j===0?C.ink:rgba(C.ink3,0.55); c.fillRect(cx-18,yy-4,w,8);} c.__tag='';}
  else if(k==='next'){c.font=`600 ${r.w<60?10:11}px ${C.font}`; const tw=Math.min(r.w-4,c.measureText(NEXT.t.trim()).width+14), yc=laneY(r,2); rr(c,cx-tw/2,yc-10,tw,20,10); c.fillStyle=C.ink; c.fill();
    c.fillStyle=C.surf; c.textBaseline='middle'; c.textAlign='center'; c.fillText(NEXT.t.trim(),cx,yc+0.5); c.textAlign='left';}
  c.restore();}
/* lane stubs from the panel's edges to its content (xin: where the lanes enter the content; xout: where they leave; xpill: the
   tokenization's pills, whose lanes start at the pill's right end) */
function stubs(c,C,r,xin,xout,xpill){
  for(let j=0;j<5;j++){const y=laneY(r,j); c.strokeStyle=rgba(C.tok[j],0.7); c.lineWidth=2; c.beginPath();
    if(xin!=null){c.moveTo(r.x,y); c.lineTo(xin,y);}
    if(xout!=null){c.moveTo(xout,y); c.lineTo(r.x+r.w,y);}
    if(xpill!=null){c.moveTo(xpill+60,y); c.lineTo(r.x+r.w,y);}
    c.stroke();}}
/* the expanded stage: its inside, animated (p = 0 … 1 over one stage) */
function heroStage(c,C,k,r,a,p,t,still){
  c.save(); c.globalAlpha=a; const X=r.x+16, Wd=r.w-32, Y=r.y+HDR, ry=j=>laneY(r,j), F=C.font, M=C.mono;
  if(k==='tok'){
    /* the text is typed on the bottom row; then, one token at a time, a pill rises from its word to its row and slides to the column;
       its ID and a dashed tie (up from the word, then right) appear when it lands, so nothing crosses the sentence */
    const nar=r.w<520, sent=TOKS.map(x=>x.t).join(''), px=X+Wd-96, sx=X+4, sy=ry(4);
    let fs=nar?13:21; c.font=`500 ${fs}px ${F}`; const room=px-(nar?16:52)-sx, sw0=c.measureText(sent).width;
    if(sw0>room){fs=Math.max(10,Math.floor(fs*room/sw0)); c.font=`500 ${fs}px ${F}`;}
    c.textBaseline='middle';
    const typed=Math.floor(seg01(p,0.02,0.28)*sent.length), fl=j=>{const s0=0.32+0.075*j+(j===4?0.05:0); return seg01(p,s0,s0+0.15);}, fade=seg01(p,0.29,0.34), outE=u=>1-(1-u)*(1-u), spW=c.measureText(' ').width;
    const geo=[]; let off=0;
    TOKS.forEach((tk,j)=>{const x0=sx+c.measureText(sent.slice(0,off)).width, lead=j?spW:0, w=c.measureText(tk.t).width-lead; geo.push({x0,x:x0+lead,w,o:off}); off+=tk.t.length;});
    TOKS.forEach((tk,j)=>{const g=geo[j], n=Math.max(0,Math.min(tk.t.length,typed-g.o)); if(!n) return;
      c.globalAlpha=a*(1-0.55*fade); c.fillStyle=C.ink; c.fillText(tk.t.slice(0,n),g.x0,sy); c.globalAlpha=a;
      const f=fl(j); if(f>0){c.fillStyle=rgba(C.tok[j],0.9*Math.min(1,f*4)); c.fillRect(g.x,sy+fs*0.62,g.w,2.5);}});
    if(fade<=0&&typed<sent.length&&Math.floor(t/400)%2){const w=c.measureText(sent.slice(0,typed)).width; c.fillStyle=C.ink2; c.fillRect(sx+w+1,sy-11,1.5,22);}
    TOKS.forEach((tk,j)=>{const f=fl(j); if(f<=0) return; const g=geo[j], txt=tokTxt(tk.t), yr=ry(j), y0=sy-fs*0.6-14;
      c.font=`600 14px ${F}`; const tw=c.measureText(txt).width+14; let x, y, al;
      if(j<4){const fr=outE(seg01(f,0,0.45)), fsl=outE(seg01(f,0.45,1)); x=lerp(g.x0,px,fsl); y=lerp(y0,yr,fr); al=seg01(f,0,0.3);}
      else{const fsl=outE(f); x=lerp(g.x+g.w+10,px,fsl); y=yr; al=seg01(f,0,0.3);}
      const land=seg01(f,0.93,1);
      if(land>0){c.globalAlpha=a*land;
        let idL=px-6; if(!nar){c.font=`500 10px ${M}`; const idw=c.measureText(String(tk.id)).width; c.fillStyle=C.ink3; c.textAlign='right'; c.fillText(String(tk.id),px-8,yr+0.5); c.textAlign='left'; idL=px-8-idw-4;}
        const xc=g.x+g.w/2; c.strokeStyle=rgba(C.tok[j],0.65); c.lineWidth=1.2; c.setLineDash([3,3]); c.beginPath();
        if(j<4){c.moveTo(xc,sy-fs*0.6-3); c.lineTo(xc,yr+7); c.bezierCurveTo(xc,yr+3,xc+3,yr,xc+7,yr); c.lineTo(idL,yr);}
        else{c.moveTo(g.x+g.w+5,yr); c.lineTo(idL,yr);}
        if(idL>xc+10||j===4) c.stroke(); c.setLineDash([]);
        c.strokeStyle=rgba(C.tok[j],0.7); c.lineWidth=2; c.beginPath(); c.moveTo(px-2+tw,yr); c.lineTo(r.x+r.w,yr); c.stroke(); c.globalAlpha=a;
        if(land>=1&&!still) dots(c,mkPath([[px-2+tw,yr],[r.x+r.w,yr]]),t,mix(C.tok[j],C.ink,0.25),{sp:26,v:46,r:2,ph:j*7});}
      c.globalAlpha=a*al; rr(c,x-2,y-12,tw,24,12); c.fillStyle=rgba(C.tok[j],0.85); c.fill(); c.font=`600 14px ${F}`; c.fillStyle=C.ink; c.fillText(txt,x+5,y+0.5); c.globalAlpha=a;});
    c.font=`500 ${fs}px ${F}`;
  }
  else if(k==='emb'){
    stubs(c,C,r,null,X+Wd-86+84);
    c.font=`600 12px ${F}`; const pws=TOKS.map(tk=>c.measureText(tokTxt(tk.t)).width+12), pmax=Math.max(...pws);
    const tx0=Math.max(X+Wd*0.34,X+pmax+46), tw=Wd*0.2, ty=Y+4, th=r.h-HDR-18, rowAt=[0.14,0.66,0.06,0.8,0.36].map(f=>ty+6+f*(th-12));
    rr(c,tx0,ty,tw,th,6); c.fillStyle=rgba(C.w,0.07); c.fill(); c.strokeStyle=rgba(C.w,0.7); c.lineWidth=1.2; c.stroke();
    c.strokeStyle=rgba(C.w,0.16); c.lineWidth=0.7; for(let y=ty+5;y<ty+th-3;y+=5){c.beginPath(); c.moveTo(tx0+3,y); c.lineTo(tx0+tw-3,y); c.stroke();}
    const m1=ease(seg01(p,0.04,0.4)), m2=ease(seg01(p,0.44,0.76)), sx0=X+Wd-86;
    TOKS.forEach((tk,j)=>{const y0=ry(j), yr=rowAt[j];
      c.font=`600 12px ${F}`; const txt=tokTxt(tk.t), pw=pws[j], xl=X+pmax-pw, xe=X+pmax, gg=Math.min(50,(tx0-xe)/2);
      /* the pill stays (right-aligned, so no curve crosses another pill); a dot travels to its row, the row lights up, the row slides out */
      c.strokeStyle=rgba(C.tok[j],0.25+0.35*m1); c.lineWidth=1.2; c.setLineDash([3,3]); c.beginPath(); c.moveTo(xe,y0); c.bezierCurveTo(xe+gg,y0,tx0-gg,yr,tx0,yr); c.stroke(); c.setLineDash([]);
      rr(c,xl,y0-9,pw,18,9); c.fillStyle=rgba(C.tok[j],0.85); c.fill(); c.fillStyle=C.ink; c.textBaseline='middle'; c.fillText(txt,xl+6,y0+0.5);
      if(m1>0&&m1<1){const P=mkPath(bez([xe,y0],[xe+gg,y0],[tx0-gg,yr],[tx0,yr],16)), q=at(P,P.L*m1); c.fillStyle=C.tok[j]; c.beginPath(); c.arc(q[0],q[1],3.5,0,7); c.fill();}
      if(m1>=1){c.fillStyle=rgba(C.w,0.6); c.fillRect(tx0+2,yr-2.5,tw-4,5);}
      if(m2>0){c.strokeStyle=rgba(C.tok[j],0.35*m2); c.lineWidth=1.2; c.beginPath(); c.moveTo(tx0+tw,yr); c.bezierCurveTo(tx0+tw+40,yr,sx0-40,y0,sx0,y0); c.stroke();
        const sx=lerp(tx0+tw/2-30,sx0,m2), sy=lerp(yr,y0,m2); strip(c,sx,sy-5,7,10,VEC[j],C.tok[j],Math.min(1,m2*1.5));}});
  }
  else if(k==='attn'){
    /* token by token (product owner, 2026-10-02: every line at once did not read as causal). Query i lights up, its lines reach back
       from its output to tokens 0…i (width = attention weight, the weights above the inputs), the tokens after it are hatched out
       (the causal mask), then their colours flow along the lines into its output. One slot of 0.17 per token from 0.04; from 0.89
       the last token's state holds, which is also the still frame (reduced motion) */
    const lx=X+2, rx=X+Wd-86, cw=7, xa=lx+12*cw, xb=rx-2, xm=lerp(xa,xb,0.5), W4=weightM(1,true);
    const S0=0.04, SL=0.17, cur=p<S0?-1:Math.min(4,Math.floor((p-S0)/SL)), u=cur<0?0:(p>=S0+5*SL?1:seg01(p,S0+cur*SL,S0+(cur+1)*SL));
    const reach=ease(seg01(u,0.06,0.4)), pctA=seg01(u,0.36,0.5), fill=seg01(u,0.48,0.9);
    stubs(c,C,r,lx,rx+84);
    /* inputs: the tokens after the current query are faded and hatched (masked) */
    TOKS.forEach((tk,j)=>{const y=ry(j), out=cur>=0&&j>cur;
      c.globalAlpha=a*(out?0.3:1); strip(c,lx,y-5,cw,10,VEC[j],C.tok[j],1); c.globalAlpha=a;
      c.font=`500 10px ${M}`; c.fillStyle=out?rgba(C.ink3,0.55):C.ink3; c.textBaseline='middle'; c.textAlign='left'; c.fillText(tokTxt(tk.t),lx,y-12);
      if(out){c.strokeStyle=rgba(C.ink3,0.75); c.lineWidth=1; c.beginPath();
        for(let s0=lx-10;s0<xa;s0+=5){const xL=Math.max(lx,s0), xR=Math.min(xa-1,s0+10); if(xR>xL){c.moveTo(xL,y+5-(xL-s0)); c.lineTo(xR,y+5-(xR-s0));}}
        c.stroke();}});
    /* lines: earlier queries stay faint; the current one reaches back from its output */
    const curve=(i,j)=>bez([xa,ry(j)],[xm,ry(j)],[xm,ry(i)],[xb,ry(i)],24);
    for(let i=0;i<=cur;i++){const now=i===cur;
      for(let j=0;j<=i;j++){const w=W4[i][j], pts=curve(i,j), q=now?pts.slice(Math.floor((1-reach)*(pts.length-1))):pts; if(q.length<2) continue;
        c.strokeStyle=rgba(C.att,now?0.25+0.6*w:0.06+0.12*w); c.lineWidth=now?Math.max(0.8,w*12):Math.max(0.5,w*2.5); strokePts(c,q);}}
    /* the current query's weights, above the right end of each input it sees; its inputs' colours flowing in */
    if(cur>=0){const pc=pctInt(W4[cur]);
      if(pctA>0){c.globalAlpha=a*pctA; c.font=`600 10px ${M}`; c.fillStyle=C.att; c.textAlign='right'; for(let j=0;j<=cur;j++) c.fillText(pc[j]+'%',xa,ry(j)-12); c.textAlign='left'; c.globalAlpha=a;}
      if(fill>0&&fill<1&&!still) for(let j=0;j<=cur;j++){const w=W4[cur][j]; if(w<0.03) continue; dots(c,mkPath(curve(cur,j)),t,mix(C.tok[j],C.ink,0.2),{sp:24,v:90,r:1.4+1.4*w,ph:j*9});}}
    /* outputs: rows already computed are full, the current one fills cell by cell, the rest wait as empty slots */
    TOKS.forEach((tk,i)=>{const y=ry(i), w=W4[i], n=(i<cur||(cur===4&&u>=1))?12:(i===cur?Math.floor(fill*12+1e-9):0);
      let acc=0; const src=VEC[i].map((v,cI)=>{const uu=(cI+0.5)/12; let s1=0; acc=0; for(let j=0;j<=i;j++){acc+=w[j]; if(uu<=acc){s1=j; break;} s1=j;} return s1;});
      if(n<12){c.strokeStyle=rgba(C.ink3,0.4); c.lineWidth=1; c.setLineDash([2,3]); c.strokeRect(rx+n*cw+0.5,y-4.5,(12-n)*cw-1.5,9); c.setLineDash([]);}
      for(let cI=0;cI<n;cI++){const v=VEC[src[cI]][cI]; c.fillStyle=rgba(C.tok[src[cI]],0.22+0.78*Math.abs(v)); c.fillRect(rx+cI*cw,y-5,cw-1,10);}
      if(i===cur&&u<1){c.strokeStyle=rgba(C.ink,0.7); c.lineWidth=1.4; rr(c,rx-3,y-8,12*cw+5,16,4); c.stroke();}});
    /* the causal mask: a lower-triangular heatmap, filled row by row with the weights; the current query's row outlined */
    const cs=6, hx=r.x+r.w/2-18, hy=r.y+5;
    if(r.w>=520){for(let i=0;i<5;i++) for(let j=0;j<5;j++){const x=hx+j*(cs+1), y=hy+i*(cs+1);
        if(j>i){c.fillStyle=rgba(C.ink3,0.14); c.fillRect(x,y,cs,cs); c.strokeStyle=rgba(C.ink3,0.35); c.lineWidth=0.7; c.beginPath(); c.moveTo(x,y+cs); c.lineTo(x+cs,y); c.stroke();}
        else{const on=i<cur||(i===cur&&pctA>0); c.fillStyle=on?rgba(C.att,(0.15+0.85*W4[i][j])*(i===cur&&u<1?Math.max(0.2,pctA):1)):rgba(C.ink3,0.08); c.fillRect(x,y,cs,cs);}}
      if(cur>=0){c.strokeStyle=rgba(C.ink,0.6); c.lineWidth=1; c.strokeRect(hx-1,hy+cur*(cs+1)-1,5*(cs+1)+1,cs+2);}}
  }
  else if(k==='mlp'){
    /* each token's vector on its own, no lane mixing (product owner, 2026-10-02: show the short vector becoming long, then squeezed
       back). A copy of the input stretches out to the intermediate size (up and gate), SiLU dims part of it, then it is squeezed back
       to d (down) and becomes the output. Phases: widen 0.05–0.34, SiLU 0.36–0.50, narrow 0.52–0.84 (each lane 0.02 after the one
       above), then hold; the long vector stays as a faint trace. The still frame (reduced motion) is that end state */
    const cw=clamp(Math.floor((Wd-52)/66),3,5), NW=42, tot=24*cw+NW*cw+52, x0=X+Math.max(0,(Wd-tot)/2), x1=x0+12*cw+26, x2=x1+NW*cw+26, k12=12*cw/NW;
    stubs(c,C,r,x0,x2+12*cw);
    const hh=Math.min(15,(r.h-HDR)/5-12);
    /* the sizes above the first lane: Llama-3.1-8B's hidden size 4,096 and intermediate size 14,336 (3.5 times as long) */
    const ly=ry(0)-hh/2-11; c.font=`500 10px ${M}`; c.fillStyle=C.ink3; c.textBaseline='middle'; c.textAlign='center';
    /* with the unit when all three fit over their strips (a short strip may lend 20 px on each side) and inside the panel; else bare numbers (phones) */
    const DL=[[x0+6*cw,12*cw+40,'4,096'],[x1+NW*cw/2,NW*cw,'14,336'],[x2+6*cw,12*cw+40,'4,096']], full=n=>tx(`${n} numbers`,`${n} 维`);
    const unit=DL.every(([x,w,n])=>{const m=c.measureText(full(n)).width; return m<=w&&x-m/2>=r.x+8&&x+m/2<=r.x+r.w-8;});
    DL.forEach(([x,w,n])=>c.fillText(unit?full(n):n,x,ly)); c.textAlign='left';
    TOKS.forEach((tk,j)=>{const y=ry(j), lag=0.02*j, eW=ease(seg01(p,0.05+lag,0.34+lag)), eG=seg01(p,0.36+lag,0.5+lag), eN=ease(seg01(p,0.52+lag,0.84+lag)), done=eN>=1;
      const out=VEC[j].map((v,i)=>Math.sin(v*3+i)), val=i=>{const pre=MLPW[j][i]; return lerp(Math.abs(pre)/1.4,Math.min(1,Math.abs(silu(2*pre)/2)*1.1),eG);};
      strip(c,x0,y-hh/2,cw,hh,VEC[j],C.tok[j],1,12);
      /* the two trapezoids (up and gate widen, down narrows) light up while their matrices work */
      const gU=eW>0&&eW<1?0.22:0, gD=eN>0&&eN<1?0.22:0; c.lineWidth=1; c.strokeStyle=rgba(C.w,0.55);
      c.beginPath(); c.moveTo(x0+12*cw+3,y-hh/2); c.lineTo(x1-3,y-hh/2-3); c.lineTo(x1-3,y+hh/2+3); c.lineTo(x0+12*cw+3,y+hh/2); c.closePath(); c.fillStyle=rgba(C.w,0.14+gU); c.fill(); c.stroke();
      c.beginPath(); c.moveTo(x1+NW*cw+3,y-hh/2-3); c.lineTo(x2-3,y-hh/2); c.lineTo(x2-3,y+hh/2); c.lineTo(x1+NW*cw+3,y+hh/2+3); c.closePath(); c.fillStyle=rgba(C.w,0.14+gD); c.fill(); c.stroke();
      /* slots: the long one waits empty, then keeps a faint trace once squeezed; the output waits empty until the copy lands */
      if(eW<=0){c.strokeStyle=rgba(C.ink3,0.4); c.lineWidth=1; c.setLineDash([2,3]); c.strokeRect(x1+0.5,y-hh/2-2.5,NW*cw-1,hh+5); c.setLineDash([]);}
      if(eN>0){c.globalAlpha=a*0.45*Math.min(1,eN*2); for(let i=0;i<NW;i++){c.fillStyle=rgba(C.tok[j],0.2+0.8*val(i)); c.fillRect(x1+i*cw,y-hh/2-3,cw-1,hh+6);} c.globalAlpha=a;}
      if(!done){c.strokeStyle=rgba(C.ink3,0.4); c.lineWidth=1; c.setLineDash([2,3]); c.strokeRect(x2+0.5,y-hh/2+0.5,12*cw-1,hh-1); c.setLineDash([]);}
      /* the moving copy: 42 cells packed inside the input spread out to the long slot, then pack into the output slot */
      if(eW>0&&!done) for(let i=0;i<NW;i++){let x, w, h, al;
        if(eN<=0){x=lerp(x0+i*k12,x1+i*cw,eW); w=lerp(k12,cw,eW); h=lerp(hh,hh+6,eW); al=val(i);}
        else{x=lerp(x1+i*cw,x2+i*k12,eN); w=lerp(cw,k12,eN); h=lerp(hh+6,hh,eN); al=lerp(val(i),Math.abs(out[Math.floor(i*12/NW)]),eN);}
        c.fillStyle=rgba(C.tok[j],0.2+0.8*al); c.fillRect(x,y-h/2,Math.max(0.6,w-(w>2.5?1:0.2)),h);}
      if(done) strip(c,x2,y-hh/2,cw,hh,out,C.tok[j],1,12);});
  }
  else if(k==='prob'){
    const y4=ry(4), hx=X+4, g1=seg01(p,0.05,0.55);
    rr(c,hx+2,Y+6,16,r.h-HDR-22,4); c.fillStyle=rgba(C.w,0.2); c.fill(); c.strokeStyle=rgba(C.w,0.85); c.lineWidth=1.2; c.stroke();
    c.font=`600 9px ${M}`; c.save(); c.translate(hx+10,Y+(r.h-HDR)/2); c.rotate(-Math.PI/2); c.fillStyle=C.w; c.textAlign='center'; c.fillText('LM head',0,3); c.restore();
    const bx=hx+132, bw=Wd-190, pc=pcts(PROB5);
    CANDS.forEach((cd,j)=>{const y=probY(r,j), w=bw*PROB5[j]*ease(g1);
      c.strokeStyle=rgba(C.w,0.3); c.lineWidth=1; c.beginPath(); c.moveTo(hx+18,Y+(r.h-HDR)/2); c.bezierCurveTo(hx+40,Y+(r.h-HDR)/2,hx+40,y,hx+62,y); c.stroke();
      c.font=`500 11px ${M}`; c.fillStyle=j===0?C.ink:C.ink2; c.textBaseline='middle'; c.textAlign='right'; c.fillText(tokTxt(cd.t),bx-8,y); c.textAlign='left';
      rr(c,bx,y-6,bw,12,3); c.fillStyle=rgba(C.ink3,0.09); c.fill();
      c.fillStyle=j===0?C.ink:rgba(C.ink3,0.5); rr(c,bx,y-6,Math.max(2,w),12,3); c.__tag='bar'; c.fill(); c.__tag='';
      c.fillStyle=C.ink2; c.font=`500 10px ${M}`; if(g1>0.6) c.fillText(pc[j].toFixed(1)+'%',bx+Math.max(2,w)+6,y);
      if(j===0&&g1>0.6){c.strokeStyle=rgba(C.ink,0.8*(g1-0.6)/0.4); c.lineWidth=2; c.beginPath(); c.moveTo(bx+w+54,y); c.lineTo(r.x+r.w,y); c.stroke();}});
  }
  else if(k==='next'){
    /* centred: the text so far on top, the five probabilities, the picked token (and its ID); wide: it then flies back along the loop,
       narrow: it flies to the end of the text. Lanes end at the panel's edge. */
    const nar=hero.narrow, sent=TOKS.map(x=>x.t).join(''), pop=ease(seg01(p,0.02,0.2)), fly=ease(seg01(p,0.4,0.9)), txt=NEXT.t.trim();
    const g=nextGeo(r), sm=g.sm, cx=g.cx, cy=g.cy, bx=g.x0, bw=g.bw, sfs=sm?12:15, sx=sm?r.x+16:g.x0, sy=r.y+HDR+(sm?16:20), yb=j=>cy+(j-2)*(sm?13:16);
    c.font=`500 ${sfs}px ${F}`; c.textBaseline='middle'; c.textAlign='left'; c.fillStyle=C.ink2; c.fillText(sent,sx,sy); const sw=c.measureText(sent).width;
    c.__tag='bar'; CANDS.forEach((cd,j)=>{rr(c,bx,yb(j)-4,Math.max(3,bw*PROB5[j]/PROB5[0]),8,3); c.fillStyle=j===0?C.ink:rgba(C.ink3,0.5); c.fill();}); c.__tag='';
    c.strokeStyle=rgba(C.ink,0.8); c.lineWidth=1.5; c.beginPath(); c.moveTo(bx+bw+8,yb(0)); c.bezierCurveTo(bx+bw+(sm?26:52),yb(0),cx-(sm?56:84),cy,cx-(sm?34:44),cy); c.stroke();
    c.font=`600 ${Math.round(12+6*pop)}px ${F}`; const tw=c.measureText(txt).width+22, sc=0.6+0.4*pop+(p<0.24?0.08*Math.sin(p*40):0);
    c.save(); c.translate(cx,cy); c.scale(sc,sc); c.globalAlpha=a*(fly>0&&fly<1?0:1); rr(c,-tw/2,-15,tw,30,15); c.fillStyle=C.ink; c.fill(); c.fillStyle=C.surf; c.textBaseline='middle'; c.textAlign='center'; c.fillText(txt,0,1); c.restore();
    if(!(fly>0&&fly<1)){c.font=`500 11px ${M}`; c.fillStyle=C.ink3; if(sm){c.textAlign='center'; c.fillText(`ID ${NEXT.id}`,cx,cy+28);} else c.fillText(`ID ${NEXT.id}`,cx+tw/2+8,cy+1); c.textAlign='left';}
    if(nar){c.font=`500 11px ${F}`; c.fillStyle=C.ink3; c.textAlign='right'; c.fillText(tx('↻ then the model runs again on the longer text','↻ 然后模型在更长的文字上再跑一遍'),r.x+r.w-14,sm?r.y+r.h-14:sy); c.textAlign='left';}
    c.font=`600 12px ${F}`; const w2=c.measureText(txt).width+14;
    if(!nar){if(pop>=1){c.setLineDash([3,3]); rr(c,sx+sw+8,sy-10,w2,20,10); c.strokeStyle=rgba(C.ink,0.45); c.lineWidth=1; c.stroke(); c.setLineDash([]); c.fillStyle=rgba(C.ink,0.55); c.fillText(txt,sx+sw+15,sy+1);}}
    else if(fly>0){const x=lerp(cx,sx+sw+8+w2/2,ease(seg01(fly,0.3,1))), y=lerp(cy,sy,1-(1-Math.min(1,fly*1.5))**2);
      rr(c,x-w2/2,y-10,w2,20,10); c.fillStyle=rgba(C.ink,0.85); c.fill(); c.fillStyle=C.surf; c.textAlign='center'; c.fillText(txt,x,y+1); c.textAlign='left';}
  }
  c.restore();}
function heroHTML(){
  return `<div class="ar-hero2" data-narrow="0"><div class="ar-hbar"><p class="ar-hcap"><span class="ar-hq">“The capital of France is”</span> <span aria-hidden="true">→</span> <b>“ Paris”</b></p>`+
    `<span class="ar-hdesc small muted" aria-live="polite">${esc(L(HDESC[HST[hero.cur].k]))}</span></div>`+
    `<div class="ar-hstage"><canvas class="ar-hcv" aria-hidden="true"></canvas><div class="ar-hbtns" aria-hidden="true">`+
    HST.map((s,i)=>`<span class="ar-hb${i===hero.cur?' on':''}"><span>${esc(L(s.t))}</span></span>`).join('')+
    `</div></div></div>`;}
function heroMount(el){
  if(hero.loop){hero.loop.kill(); hero.loop=null;}
  hero.el=el; hero.narrow=null; el.innerHTML=heroHTML(); hero.cv=$('.ar-hcv',el); hero.btns=$$('.ar-hb',el); hero.t0=performance.now(); hero.last=hero.t0; hero.k=0;
  hero.loop=makeLoop($('.ar-hstage',el),heroDraw);
}
function heroBind(el){}   // nothing to press: the hero plays through its six stages on its own (2026-10-11 19:28 UTC)

/* ======================================================================================================================
   LAB OVERVIEW: the hero's detailed version, top to bottom, with “The capital of France is” → “ Paris”
   ====================================================================================================================== */
/* what the model switch changes in the drawing (interpolated over 700 ms): lane width ∝ √d, bulge ∝ √intermediate size, plates, hairlines */
const ov={P:null,from:null,to:null,t0:0,cv:null,box:null,hot:null,loop:null,sig:''};
const mP=k=>{const m=MODELS[k]; return {d:m.d,f:m.f,L:m.L,H:m.H};};
function ovTarget(k){const to=mP(k); if(!ov.P){ov.P={...to}; ov.to=to; return;} ov.from={...ov.P}; ov.to=to; ov.t0=performance.now(); if(ov.loop) ov.loop.sync();}
function ovParams(t,still){if(!ov.from) return ov.P; const p=still?1:Math.min(1,(t-ov.t0)/700), e=ease(p);
  ov.P={d:lerp(ov.from.d,ov.to.d,e),f:lerp(ov.from.f,ov.to.f,e),L:lerp(ov.from.L,ov.to.L,e),H:lerp(ov.from.H,ov.to.H,e)}; if(p>=1) ov.from=null; return ov.P;}
const HOT=[['tok','emb'],['emb','emb'],['norm1','layer'],['attn','attn'],['res1','layer'],['norm2','layer'],['mlp','mlp'],['res2','layer'],['stack','layer'],['normF','layer'],['head','out',0],['prob','out',1],['next','out',2]];
const HOT_TIP={tok:'tok',emb:'emb',norm1:'norm',norm2:'norm',normF:'norm',attn:'attn',mlp:'mlp',res1:'res',res2:'res',stack:'layer',head:'head',prob:'prob',next:'next'};
const HOT_NAME={tok:{en:'Tokens',zh:'token'},emb:{en:'Embedding table',zh:'embedding 表'},norm1:{en:'RMSNorm',zh:'RMSNorm'},norm2:{en:'RMSNorm',zh:'RMSNorm'},normF:{en:'Final RMSNorm',zh:'最后的 RMSNorm'},
  attn:{en:'Self-attention',zh:'自注意力'},mlp:{en:'MLP',zh:'MLP'},res1:{en:'Residual add ⊕',zh:'残差相加 ⊕'},res2:{en:'Residual add ⊕',zh:'残差相加 ⊕'},stack:{en:'The stack of layers',zh:'层的堆叠'},
  head:{en:'LM head',zh:'LM head'},prob:{en:'Probabilities',zh:'概率'},next:{en:'Next token',zh:'下一个 token'}};
function ovLayout(W,P,k){k=k||1; const g=v=>Math.round(v*k), bh=v=>Math.round(v*(0.8+0.2*k));
  const pad=12, narrow=W<480, lw=Math.sqrt(P.d)/22, sb=lw+2.4, bw=5*sb, xRes=Math.max(64,W*0.2), G=narrow?16:24;   // G: the loop's own lane on the right
  const o={W,pad,narrow,lw,sb,bw,xRes,k,G};
  o.pillY=8; o.pillH=24; o.tabY=o.pillY+o.pillH+g(16); o.tabH=30; o.stripY=o.tabY+o.tabH+g(8); o.card={x:pad+2,y:o.stripY+g(46),w:W-2*pad-16-G};
  const cy=o.card.y, bxL=xRes+bw/2+(narrow?42:70), bxR=W-pad-26-G;
  o.p1=cy+Math.max(24,g(22)); o.a={x:bxL,y:o.p1+g(26),w:bxR-bxL,h:bh(88)}; o.r1=o.a.y+o.a.h+g(24);
  o.p2=o.r1+g(20); o.m={x:bxL,y:o.p2+g(26),w:bxR-bxL,h:bh(84)}; o.r2=o.m.y+o.m.h+g(24);
  o.rad=bw/2+7; o.card.h=o.r2+Math.max(g(20),o.rad+7)-cy; o.nF=o.card.y+o.card.h+12+g(14); o.head={w:Math.min(300,W*0.5),h:22}; o.head.x=W/2-o.head.w/2+(narrow?0:40); o.head.y=o.nF+Math.max(26,g(22));
  o.barY=o.head.y+o.head.h+g(12); o.barH=14; o.nextY=o.barY+5*o.barH+g(8); o.H=o.nextY+32;
  return o;}
/* the largest spacing (down to 40%) whose drawing fits the sticky stage without scrolling inside it */
function ovFit(cv,W0,P,t){
  const sig=W0+'|'+innerHeight+'|'+Math.round(P.d)+'|'+Math.round(P.L);
  if(ov.ksig===sig&&ov.k) return ov.k;
  const st=cv.closest('.stage'); let k=1;
  if(st){const cs=getComputedStyle(st);
    if(cs.position==='sticky'){const mh=parseFloat(cs.maxHeight), other=st.scrollHeight-cv.offsetHeight, avail=(mh>0?mh:innerHeight-86)-other-2;
      for(const kk of [1,0.9,0.8,0.7,0.6,0.5,0.4]){k=kk; if(ovLayout(W0,P,kk).H<=avail) break;}}}
  ov.k=k; ov.ksig=sig; return k;}
function ovDraw(t,still){
  const cv=ov.cv; if(!cv||!cv.isConnected) return;
  const W0=cv.parentNode.getBoundingClientRect().width; if(W0<60) return;
  const P=ovParams(t,still), k=ovFit(cv,W0,P,t), o=ovLayout(W0,P,k);
  if(cv._hh!==o.H){cv.style.height=o.H+'px'; cv._hh=o.H;}
  const {c}=fit(cv); c.clearRect(0,0,W0,o.H);
  ovPaint(c,W0,P,t,still,false,k);}
function ovPaint(c,W,P,t,still,rec,k){
  const C=cols(), o=ovLayout(W,P,k||ov.k||1);
  const F=C.font, M=C.mono, lw=o.lw, sb=o.sb, bx=k=>o.xRes+(k-2)*sb;
  /* 1. the prompt's pills */
  c.font=`600 ${o.narrow?11:13}px ${F}`; const pw=TOKS.map(tk=>c.measureText(tokTxt(tk.t)).width+14), gap=o.narrow?4:8, tot=pw.reduce((a,b)=>a+b,0)+gap*4;
  c.font=`600 ${o.narrow?11:12}px ${F}`; const gw0=c.measureText(NEXT.t.trim()).width+14; c.font=`600 ${o.narrow?11:13}px ${F}`;
  let x=W/2-(tot+gw0+2)/2; const px=[];
  TOKS.forEach((tk,k)=>{rr(c,x,o.pillY,pw[k],o.pillH,o.pillH/2); c.fillStyle=rgba(C.tok[k],0.85); c.fill(); c.fillStyle=C.ink; c.textBaseline='middle'; c.fillText(tokTxt(tk.t),x+7,o.pillY+o.pillH/2+0.5); px.push(x+pw[k]/2); x+=pw[k]+gap;});
  const pillsR={x:px[0]-pw[0]/2-4,y:o.pillY-4,w:tot+8,h:o.pillH+8};
  /* the loop: “ Paris” is appended and the model runs again */
  c.font=`600 ${o.narrow?11:12}px ${F}`; const gw=c.measureText(NEXT.t.trim()).width+14, gx=x+2;
  rr(c,gx,o.pillY+2,gw,o.pillH-4,(o.pillH-4)/2); c.setLineDash([3,3]); c.strokeStyle=rgba(C.ink,0.45); c.lineWidth=1; c.stroke(); c.setLineDash([]);
  c.fillStyle=rgba(C.ink,0.5); c.fillText(NEXT.t.trim(),gx+7,o.pillY+o.pillH/2+0.5);
  /* 2. the embedding table */
  const tab={x:o.pad+4,y:o.tabY,w:W-2*o.pad-8-o.G,h:o.tabH};
  rr(c,tab.x,tab.y,tab.w,tab.h,6); c.fillStyle=rgba(C.w,0.08); c.fill(); c.strokeStyle=rgba(C.w,0.75); c.lineWidth=1.2; c.stroke();
  c.strokeStyle=rgba(C.w,0.14); c.lineWidth=0.6; for(let yy=tab.y+4;yy<tab.y+tab.h-2;yy+=4){c.beginPath(); c.moveTo(tab.x+3,yy); c.lineTo(tab.x+tab.w-3,yy); c.stroke();}
  if(!o.narrow){c.font=`600 10px ${M}`; c.fillStyle=rgba(C.w,0.95); c.textAlign='right'; c.fillText(tx('embedding table','embedding 表'),tab.x+tab.w-8,tab.y+tab.h-7); c.textAlign='left';}
  const lanes=[];
  TOKS.forEach((tk,k)=>{const x0=px[k], yl=tab.y+8+[0.1,0.55,0.25,0.8,0.4][k]*(tab.h-16);
    c.strokeStyle=rgba(C.tok[k],0.85); c.lineWidth=lw; c.beginPath(); c.moveTo(x0,o.pillY+o.pillH); c.lineTo(x0,tab.y); c.stroke();
    c.fillStyle=rgba(C.w,0.55); c.fillRect(x0-14,yl-2,28,4);
    strip(c,x0-24,o.stripY,4,7,VEC[k],C.tok[k],1);
    c.strokeStyle=rgba(C.tok[k],0.85); c.lineWidth=lw; c.beginPath(); c.moveTo(x0,tab.y+tab.h); c.lineTo(x0,o.stripY); c.stroke();
    /* converge into the residual stream */
    const pts=bez([x0,o.stripY+9],[x0,o.stripY+36],[bx(k),o.card.y-24],[bx(k),o.card.y],24);
    c.strokeStyle=rgba(C.tok[k],0.85); c.__tag='lane'+k; strokePts(c,pts); c.__tag=''; lanes.push([[x0,o.pillY+o.pillH]].concat(pts));});
  /* 3. the layer card, with one plate per other layer behind it */
  const cd=o.card, nPl=Math.max(1,Math.round(P.L)-1), sp=10/Math.max(1,nPl);
  c.lineWidth=0.7; c.__tag='stack'; for(let i=nPl;i>=1;i--){const off=i*sp; rr(c,cd.x+off,cd.y+off,cd.w,cd.h,12); c.strokeStyle=rgba(C.ink3,nPl>60?0.11:0.22); c.stroke();} c.__tag='';
  rr(c,cd.x,cd.y,cd.w,cd.h,12); c.fillStyle=C.surf; c.fill(); c.strokeStyle=rgba(C.ink3,0.55); c.lineWidth=1; c.__tag='stack'; c.stroke(); c.__tag='';
  c.font=`600 11px ${M}`; const tag=tx(`× ${Math.round(P.L)} layers`,`× ${Math.round(P.L)} 层`), tgw=c.measureText(tag).width+16, tagR={x:cd.x+cd.w-tgw-10,y:cd.y-10,w:tgw,h:20};
  rr(c,tagR.x,tagR.y,tagR.w,tagR.h,10); c.fillStyle=C.surf; c.fill(); c.strokeStyle=rgba(C.ink3,0.6); c.stroke(); c.fillStyle=C.ink2; c.textBaseline='middle'; c.fillText(tag,tagR.x+8,tagR.y+10.5);
  /* the residual stream: five lanes straight down through the card */
  const bot=o.r2+22;
  /* the lanes continue from where they reach the card's top edge (the card is drawn over their ends), with no gap */
  for(let k=0;k<5;k++){c.strokeStyle=rgba(C.tok[k],0.9); c.lineWidth=lw; c.__tag='lane'+k; c.beginPath(); c.moveTo(bx(k),cd.y); c.lineTo(bx(k),o.nF); c.stroke(); c.__tag=''; lanes[k]=lanes[k].concat([[bx(k),o.nF]]);}
  c.font=`500 10px ${F}`; c.fillStyle=C.ink3; c.save(); c.translate(o.xRes-o.bw/2-19,(cd.y+8+o.r1-o.bw/2-12)/2); c.rotate(-Math.PI/2); c.textAlign='center'; c.textBaseline='middle'; c.fillText(tx('residual stream','残差流'),0,0); c.restore();
  /* the two side branches */
  const branch=(py,blk,ry,kind)=>{
    const res=o.narrow?22:Math.max(96,blk.w*0.25), lw5=(blk.w-res)/5, ringX=o.xRes+o.bw/2+(o.narrow?14:22), ent=k=>blk.x+res+(k+0.5)*lw5, blw=Math.max(1.2,lw*0.6), paths=[];
    for(let k=0;k<5;k++){
      const a=bez([bx(k),py],[bx(k)+10,py],[ringX-10,py+(k-2)*2.2],[ringX,py+(k-2)*2.2],10);
      const b=bez([ringX,py+(k-2)*2.2],[ent(k)-10,py+(k-2)*2.2],[ent(k),blk.y-20],[ent(k),blk.y],16,a);
      const e=bez([ent(k),blk.y+blk.h],[ent(k),blk.y+blk.h+18],[bx(k)+30,ry],[bx(k)+o.sb*0,ry],18);
      c.strokeStyle=rgba(C.tok[k],0.75); c.lineWidth=blw; strokePts(c,b); strokePts(c,e); paths.push([b,e]);}
    /* the RMSNorm ring the branch passes through */
    c.beginPath(); c.ellipse(ringX,py,5,9,0,0,7); c.fillStyle=rgba(C.surf,0.4); c.fill(); c.lineWidth=2; c.strokeStyle=C.ink2; c.stroke();
    if(kind==='attn'){c.font=`500 9px ${M}`; c.fillStyle=C.ink3; c.textBaseline='alphabetic'; c.fillText('RMSNorm',ringX+9,py-9);}
    /* the block */
    rr(c,blk.x,blk.y,blk.w,blk.h,10); c.fillStyle=kind==='attn'?rgba(C.att,0.06):rgba(C.w,0.05); c.fill(); c.strokeStyle=kind==='attn'?rgba(C.att,0.7):rgba(C.w,0.7); c.lineWidth=1.2; c.stroke();
    const nm=kind==='attn'?tx('Self-attention','自注意力'):'MLP';
    if(o.narrow){c.save(); c.font=`600 11px ${F}`; const fw=c.measureText(nm).width, fz=Math.min(11,Math.floor(11*(blk.h-14)/Math.max(1,fw))); c.font=`600 ${fz}px ${F}`; c.fillStyle=C.ink; c.translate(blk.x+12,blk.y+blk.h/2); c.rotate(-Math.PI/2); c.textAlign='center'; c.textBaseline='middle'; c.fillText(nm,0,0); c.restore();}
    else{c.font=`600 12px ${F}`; c.fillStyle=C.ink; c.textBaseline='alphabetic'; c.fillText(nm,blk.x+10,blk.y+18);}
    for(let k=0;k<5;k++){c.strokeStyle=rgba(C.tok[k],0.75); c.lineWidth=blw; c.beginPath(); c.moveTo(ent(k),blk.y); c.lineTo(ent(k),blk.y+blk.h); c.stroke();}
    if(kind==='attn'){const Wm=weightM(1,true), yA=blk.y+(o.narrow?18:14), yB=blk.y+blk.h-10, pul=still?1:0.8+0.2*Math.sin(t/300);
      for(let i=1;i<5;i++) for(let j=0;j<i;j++){const w=Wm[i][j]; c.strokeStyle=rgba(C.att,0.25+0.65*w); c.lineWidth=Math.max(0.6,w*7*(i===4?pul:1));
        c.beginPath(); c.moveTo(ent(j),yA); c.bezierCurveTo(ent(j),(yA+yB)/2,ent(i),(yA+yB)/2,ent(i),yB); c.stroke();}
      /* one hairline per head */
      const nH=Math.round(P.H), hx0=o.narrow?blk.x+res+6:blk.x+10, hx1=o.narrow?blk.x+blk.w-8:blk.x+res-12, hy=o.narrow?blk.y+4:blk.y+28;
      c.strokeStyle=rgba(C.att,0.7); c.lineWidth=0.6; for(let i=0;i<nH;i++){const xx=lerp(hx0,hx1,nH>1?i/(nH-1):0.5); c.beginPath(); c.moveTo(xx,hy); c.lineTo(xx,hy+(o.narrow?8:14)); c.stroke();}}
    else{const bwid=Math.min(lw5-8,Math.sqrt(P.f)/6.2), yA=blk.y+10, yB=blk.y+blk.h-8;
      for(let k=0;k<5;k++){const xk=ent(k); c.beginPath(); c.moveTo(xk,yA); c.bezierCurveTo(xk+bwid*0.66,yA+10,xk+bwid*0.66,yB-10,xk,yB); c.bezierCurveTo(xk-bwid*0.66,yB-10,xk-bwid*0.66,yA+10,xk,yA);
        c.fillStyle=rgba(C.tok[k],0.3); c.fill(); c.strokeStyle=rgba(C.w,0.75); c.lineWidth=1; c.stroke();
        for(let i=0;i<7;i++){const yy=lerp(yA+8,yB-8,i/6), half=Math.sin(Math.PI*(yy-yA)/(yB-yA))*bwid*0.42, g=still?0.6:0.5+0.5*Math.sin(t/330+i*1.3+k*2.1);
          c.strokeStyle=rgba(C.tok[k],0.25+0.7*g); c.lineWidth=2; c.beginPath(); c.moveTo(xk-half,yy); c.lineTo(xk+half,yy); c.stroke();}}}
    /* ⊕ on the residual stream */
    const rad=o.bw/2+7; c.beginPath(); c.arc(o.xRes,ry,rad,0,7); c.fillStyle=rgba(C.surf,0.75); c.fill(); c.lineWidth=1.6; c.strokeStyle=C.ink2; c.stroke();
    c.beginPath(); c.moveTo(o.xRes-rad*0.5,ry); c.lineTo(o.xRes+rad*0.5,ry); c.moveTo(o.xRes,ry-rad*0.5); c.lineTo(o.xRes,ry+rad*0.5); c.stroke();
    return {paths,ring:{x:ringX-12,y:py-14,w:24,h:28},plus:{x:o.xRes-rad,y:ry-rad,w:2*rad,h:2*rad}};};
  const b1=branch(o.p1,o.a,o.r1,'attn'), b2=branch(o.p2,o.m,o.r2,'mlp');
  /* 4. the final RMSNorm, the LM head (only the last token matters for the next one), probabilities, the next token */
  /* all five lanes go on through the final RMSNorm (drawn over the joint) into the LM head; the last token's lane (the one whose
     logits pick the next token) is the strong one */
  const hd=o.head, ex=k=>hd.x+hd.w/2-(4-k)*Math.min(16,hd.w/12);
  for(let k=0;k<5;k++){const q=bez([bx(k),o.nF],[bx(k),o.nF+(hd.y-o.nF)*0.55],[ex(k),hd.y-(hd.y-o.nF)*0.5],[ex(k),hd.y],18);
    c.strokeStyle=rgba(C.tok[k],k===4?0.9:0.5); c.lineWidth=lw; c.__tag='lane'+k; strokePts(c,q); c.__tag=''; lanes[k]=lanes[k].concat(q.slice(1));}
  const nF={x:o.xRes-o.bw/2-8,y:o.nF-7,w:o.bw+16,h:14}; rr(c,nF.x,nF.y,nF.w,nF.h,7); c.fillStyle=C.surf; c.__tag='norm'; c.fill(); c.__tag=''; c.lineWidth=2; c.strokeStyle=C.ink2; c.stroke();
  /* its name sits left of the pill when there is room: on the right the lanes sweep towards the LM head */
  c.font=`500 10px ${M}`; c.fillStyle=C.ink3; c.textBaseline='middle'; const nlw=c.measureText('RMSNorm').width;
  if(nF.x-8-nlw>=4){c.textAlign='right'; c.fillText('RMSNorm',nF.x-8,o.nF); c.textAlign='left';} else c.fillText('RMSNorm',nF.x+nF.w+6,o.nF);
  rr(c,hd.x,hd.y,hd.w,hd.h,6); c.fillStyle=rgba(C.w,0.14); c.__tag='head'; c.fill(); c.__tag=''; c.strokeStyle=rgba(C.w,0.85); c.lineWidth=1.2; c.stroke();
  c.strokeStyle=rgba(C.w,0.2); c.lineWidth=0.6; for(let xx=hd.x+4;xx<hd.x+hd.w-2;xx+=4){c.beginPath(); c.moveTo(xx,hd.y+3); c.lineTo(xx,hd.y+hd.h-3); c.stroke();}
  c.font=`600 11px ${M}`; c.fillStyle=C.w; c.textAlign='center'; c.fillText('LM head',hd.x+hd.w/2,hd.y+hd.h/2+0.5); c.textAlign='left';
  const pr=softmax(Z0), pc=pcts(pr), bx0=hd.x+hd.w*0.3, bwMax=hd.w*0.62;
  CANDS.forEach((cd2,j)=>{const y=o.barY+j*o.barH+o.barH/2; c.font=`500 11px ${M}`; c.fillStyle=j===0?C.ink:C.ink3; c.textAlign='right'; c.fillText(tokTxt(cd2.t),bx0-8,y); c.textAlign='left';
    const w=Math.max(2,bwMax*pr[j]); rr(c,bx0,y-5,w,10,3); c.fillStyle=j===0?C.ink:rgba(C.ink3,0.45); c.__tag='bar'; c.fill(); c.__tag=''; c.fillStyle=C.ink3; c.font=`500 10px ${M}`; c.fillText(pc[j].toFixed(1)+'%',bx0+w+6,y);});
  const probR={x:hd.x-10,y:o.barY-4,w:hd.w+20,h:5*o.barH+8};
  c.font=`600 13px ${F}`; const nw=c.measureText(NEXT.t.trim()).width+20, nx=hd.x+hd.w/2-nw/2, ny=o.nextY;
  const xe=bx0+Math.max(2,bwMax*pr[0])+50, ye=o.barY+o.barH/2; c.strokeStyle=C.ink; c.lineWidth=1.4; c.beginPath(); c.moveTo(xe,ye); c.bezierCurveTo(xe+22,ye,nx+nw/2+24,ny-20,nx+nw/2,ny); c.stroke();
  rr(c,nx,ny,nw,24,12); c.fillStyle=C.ink; c.fill(); c.fillStyle=C.surf; c.textBaseline='middle'; c.fillText(NEXT.t.trim(),nx+10,ny+12.5);
  /* the loop back to the prompt */
  /* right along the bottom, a rounded corner, straight up the loop's own lane (clear of the layer stack), then into the prompt */
  const lx=W-o.pad-Math.round(o.G*0.35), yb=ny+12, rc=12, loopPts=bez([nx+nw,yb],[nx+nw+20,yb],[lx-rc-20,yb],[lx-rc,yb],8);
  bez([lx-rc,yb],[lx,yb],[lx,yb],[lx,yb-rc],6,loopPts); bez([lx,yb-rc],[lx,yb-rc-40],[lx,o.pillY+o.pillH/2+70],[lx,o.pillY+o.pillH/2+30],16,loopPts);
  bez([lx,o.pillY+o.pillH/2+30],[lx,o.pillY+o.pillH/2],[gx+gw+30,o.pillY+o.pillH/2],[gx+gw+2,o.pillY+o.pillH/2],12,loopPts);
  c.setLineDash([3,4]); c.strokeStyle=rgba(C.ink,0.35); c.lineWidth=1.2; c.__tag='loop'; strokePts(c,loopPts); c.__tag=''; c.setLineDash([]); arrowHead(c,gx+gw+2,o.pillY+o.pillH/2,Math.PI,6,rgba(C.ink,0.5));
  /* particles: slower and subtler than the hero's */
  if(!still){for(let k=0;k<5;k++){const P1=mkPath(lanes[k]); dots(c,P1,t,mix(C.tok[k],C.ink,0.3),{sp:44,v:30,r:1.8,ph:k*9});
      for(const b of [b1,b2]){const [p1,p2]=b.paths[k]; dots(c,mkPath(p1),t,mix(C.tok[k],C.ink,0.3),{sp:40,v:30,r:1.6,ph:k*5}); dots(c,mkPath(p2),t,mix(C.tok[k],C.ink,0.3),{sp:40,v:30,r:1.6,ph:k*5+13});}}
    dots(c,mkPath(loopPts),t,C.ink,{sp:60,v:36,r:1.8});}
  /* the hot spots: each component is a button (hover: definition, formula, use; click: its pop-up) */
  const R={tok:pillsR,emb:tab,norm1:b1.ring,attn:o.a,res1:b1.plus,norm2:b2.ring,mlp:o.m,res2:b2.plus,stack:tagR,normF:nF,head:hd,prob:probR,next:{x:nx,y:ny,w:nw,h:24}};
  const sig=Object.values(R).map(r=>[r.x,r.y,r.w,r.h].map(v=>Math.round(v)).join(',')).join(';');
  const fk=!ov.hk&&ov.focus&&ov.focus.length?ov.focus[Math.floor(t/1100)%ov.focus.length]:null;
  if(!rec&&(ov.hk||fk)&&R[ov.hk||fk]){const hk=ov.hk||fk, r=R[hk]; c.save(); c.strokeStyle=ov.hk?C.ink:rgba(C.ink,0.85); c.lineWidth=ov.hk?2.2:2.6; if(!ov.hk) c.setLineDash([5,3]);
    if(hk==='norm1'||hk==='norm2'){c.beginPath(); c.ellipse(r.x+12,r.y+14,7.5,11.5,0,0,7); c.stroke();}
    else if(hk==='res1'||hk==='res2'){c.beginPath(); c.arc(r.x+r.w/2,r.y+r.h/2,r.w/2+1.5,0,7); c.stroke();}
    else if(hk==='normF'){rr(c,r.x-1,r.y-1,r.w+2,r.h+2,(r.h+2)/2); c.stroke();}
    else{rr(c,r.x-2,r.y-2,r.w+4,r.h+4,(hk==='tok'||hk==='next'||hk==='stack')?(r.h+4)/2:10); c.stroke();}
    c.restore();}
  if(!rec&&sig!==ov.sig&&ov.hot){ov.sig=sig; $$('.ar-hot',ov.hot).forEach(h=>{const r=R[h.dataset.k]; if(!r) return; h.style.cssText=`left:${r.x-3}px;top:${r.y-3}px;width:${r.w+6}px;height:${r.h+6}px`;});}
  return {o,R};
}
function ovHTML(){
  return `<div class="ar-ov" id="ar-ov"><canvas class="ar-ovcv" aria-hidden="true"></canvas><div class="ar-hots">`+
    HOT.map(([k,pop])=>`<span class="t ar-hot" tabindex="0" role="button" data-ctl="${k}" data-k="${k}" data-pop="${pop}" aria-label="${esc(L(HOT_NAME[k]))}"></span>`).join('')+`</div></div>`;}
function ovMount(box){if(ov.loop){ov.loop.kill(); ov.loop=null;} ov.box=box; ov.cv=$('.ar-ovcv',box); ov.hot=$('.ar-hots',box); ov.sig=''; ov.hk=null; ov.kt=-1e9; ov.ksig='';
  const hi=k=>{ov.hk=k; if(ov.loop&&!ov.loop.on) ov.loop.sync();};
  ov.hot.addEventListener('mouseover',e=>{const h=e.target.closest('.ar-hot'); hi(h?h.dataset.k:null);}); ov.hot.addEventListener('mouseout',e=>{if(!e.relatedTarget||!e.relatedTarget.closest||!e.relatedTarget.closest('.ar-hot')) hi(null);});
  ov.hot.addEventListener('focusin',e=>{const h=e.target.closest('.ar-hot'); hi(h?h.dataset.k:null);}); ov.hot.addEventListener('focusout',()=>hi(null));
  ov.loop=makeLoop(box,ovDraw);}

/* ======================================================================================================================
   POP-UPS: one diagram per component, a stepper (◀ ▶, play) that shows one computation step at a time, one caption line per step
   ====================================================================================================================== */
let dlg=null, dst=0, dplay=0, openerSel=null;
/* R3.20: a step's entrance animation plays once per content; a re-render that shows the same step with the same settings (pause, the end of
   a play-through, a click on the option already chosen, a language switch) draws it finished. A new draw animates only the drawn token */
let dlgShown='', flyShown=0;
const dlgSig=()=>[dlg,dst,aHead,aTok,aMask,a1M,noNorm,noRes].join('|');
/* R3.20: the steps whose entrance has played to its end (or that were drawn finished), by what they draw (dlgSig), not by visit: coming back
   to one (◀ ▶, an option switched A → B → A, a pop-up opened again) draws it finished; Play and ⟲ Replay, the reader's choice, animate as before */
const dlgSeen=new Set(); let dlgT=0;
/* when the entrance animations under el end: the longest delay + duration, in ms */
const animEnd=(el,sel)=>{let m=0; el.querySelectorAll(sel).forEach(e=>{const cs=getComputedStyle(e), d=(parseFloat(cs.animationDelay)||0)+(parseFloat(cs.animationDuration)||0); if(d>m) m=d;}); return Math.ceil(m*1000);};
let aHead=1, aTok=4, aMask='causal', a1M=false, siluX=-1.5, Z=Z0.slice(), temp=1, sampled=null, flyN=0, noNorm=false, noRes=false;
/* the draws of step 3 of the output pop-up: the last ~12 (token and random number), counts per token; cleared when a logit or T changes */
let draws=[], dCount=[0,0,0,0,0], dN=0, dU=null;
function drawOne(){const p=probsNow(), u=Math.random(); let acc=0, j=0; for(;j<p.length-1;j++){acc+=p[j]; if(u<acc) break;}
  draws.unshift({j,u}); if(draws.length>12) draws.length=12; dCount[j]++; dN++; dU=u; sampled=j; return j;}
function resetDraws(){draws=[]; dCount=[0,0,0,0,0]; dN=0; dU=null; sampled=null;}
const NEW=(on,d)=>on?` class="ar-new" style="animation-delay:${d||0}ms"`:'';
const tint=k=>`var(--ar-t${k})`;
/* an SVG strip of cells: its colour, and the opacity of each cell from |value| */
function sStrip(x,y,cw,ch,vals,fill,o){o=o||{}; let s=`<g${o.cls?` class="${o.cls}"`:''}${o.style?` style="${o.style}"`:''}>`;
  vals.forEach((v,i)=>{const gx=o.gap&&o.gap.every?Math.floor(i/o.gap.every)*o.gap.w:0; s+=`<rect x="${(x+i*cw+gx).toFixed(1)}" y="${y}" width="${cw-1}" height="${ch}" rx="1.5" fill="${fill}" fill-opacity="${(0.2+0.8*Math.min(1,Math.abs(v))).toFixed(2)}"/>`;});
  return s+'</g>';}
const sT=(x,y,t,o)=>{o=o||{}; return `<text x="${x}" y="${y}"${o.a?` text-anchor="${o.a}"`:''} class="${o.c||'ar-sv'}"${o.st?` style="${o.st}"`:''}>${esc(t)}</text>`;};
const NAR=()=>innerWidth<=760;
const svgOpen=(w,h,lab,fitW)=>`<svg class="ar-psvg${fitW?' fit':''}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(lab)}"><defs><pattern id="ar-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="ar-hbg"/><line x1="0" y1="0" x2="0" y2="6" class="ar-hl"/></pattern></defs>`;
const fxo=(x,y,w,tex,o)=>{o=o||{}; const h=o.h||26, x0=o.a==='middle'?x-w/2:(o.a==='end'?x-w:x);
  return `<foreignObject x="${x0}" y="${y}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="ar-fxo${o.c?' '+o.c:''}" style="text-align:${o.a==='middle'?'center':(o.a==='end'?'right':'left')}">${o.pre?`<span>${esc(o.pre)}</span> `:''}${A.ui.math(tex)}</div></foreignObject>`;};
const tg=(k,inner,lab)=>`<g class="t ar-sg" tabindex="0" role="button" data-ctl="term" data-term="${k}"${lab?` aria-label="${esc(lab)}"`:''}>${inner}</g>`;
const illus=()=>`<span class="ar-illus">${tx('illustrative values','数值是示意的')}</span>`;

/* ---------- 1. tokens and embedding ---------- */
const VOCAB=['␣the','ing','The','␣Paris','Token','␣capital','ization','的','␣of','ed','。','␣France','!','␣a','er','␣is','␣Lyon','s','␣Tokyo','ly','␣and','ation','␣city','？'];
const P_EMB={t:{en:'Tokens and embedding',zh:'token 和 embedding'},steps:[
  {en:'A tokenizer cuts the text into tokens: words, pieces of words, punctuation.',zh:'分词器把文字切成 token：词、词的一部分、标点。'},
  {en:'Each token has an ID: its number in the vocabulary.',zh:'每个 token 有一个 ID：它在词表里的编号。'},
  {en:'The ID picks one row of the embedding table.',zh:'ID 选出 embedding 表里的一行。'},
  {en:'That row is the token’s vector: it says which token, not where it stands.',zh:'这一行就是 token 的向量：它说明是哪个 token，不说明它在哪。'},
  {en:'Without position information (and without a mask), attention is order-blind: shuffle the tokens and every pair keeps its score.',zh:'没有位置信息（也没有 mask）时，attention 分不出顺序：把 token 打乱，每一对的得分都不变。'},
  {en:'RoPE adds the positions inside attention: it turns each query and key (not the value) by its position. For the same content, position acts only through the distance m − n.',zh:'RoPE 在注意力里加入位置：按位置转动 query 和 key（value 不转）。内容不变时，位置的影响只取决于距离 m − n。'}]};
function popEmb(st,bare){   // bare: step 3 without the “5th or 50th token” comparison (the primer's section 3 ends on the vectors)
  if(st===4) return posBlind(NAR());
  /* RoPE happens in attention, not in the embedding (Codex review 1 #7): the step says so and links to where it is drawn */
  if(st===5) return ropeSVG(NAR())+`<div class="ar-pctl ar-pgo"><button type="button" class="btn ghost sm" data-act="rope2attn">${tx('See where it happens: self-attention, step 1 →','看它发生在哪：自注意力第 1 步 →')}</button></div>`;
  if(NAR()) return popEmbN(st,bare);
  const W=960,H=360, xs=[], ws=TOKS.map(tk=>tokTxt(tk.t).length*9.2+22); let x=40;
  ws.forEach(w=>{xs.push(x); x+=w+14;});
  let s=svgOpen(W,H,L(P_EMB.steps[st]));
  s+=sT(40,42,TOKS.map(t=>t.t).join(''),{c:'ar-sv big'});
  const rows=[0.3,0.56,0.08,0.8,0.18].map(f=>30+f*290);
  TOKS.forEach((tk,k)=>{const cx=xs[k]+ws[k]/2;
    s+=`<g${NEW(st===0,k*90)}><path d="M${cx} 52 L${cx} 92" class="ar-sl"/>`+tg('tok',`<rect x="${xs[k]}" y="96" width="${ws[k]}" height="30" rx="15" fill="${tint(k)}"/>`+sT(xs[k]+11,116,tokTxt(tk.t),{c:'ar-sv tokt'}),tokTxt(tk.t))+`</g>`;
    if(st>=1) s+=`<g${NEW(st===1,k*90)}>`+sT(cx,150,String(tk.id),{a:'middle',c:'ar-sv mono'})+`</g>`;
    if(st>=1){const yk=196+(4-k)*7, xt=470+(4-k)*14, ro=rows[k], up=ro<yk, q=up?-6:6;
      s+=`<g${NEW(st===1,300+k*120)}><path d="M${cx} 158 L${cx} ${yk-6} Q${cx} ${yk} ${cx+6} ${yk} L${xt-6} ${yk} Q${xt} ${yk} ${xt} ${yk+q} L${xt} ${ro-q} Q${xt} ${ro} ${xt+6} ${ro} L560 ${ro}" class="ar-sl id" fill="none"/></g>`;}});
  /* steps 1-2: the vocabulary, where the table will be */
  if(st<=1){s+=`<g${NEW(true,200)}><rect x="560" y="14" width="380" height="322" rx="8" class="ar-vcard"/>`+sT(578,322,tx('The vocabulary of Llama 3: 128,256 tokens','Llama 3 的词表：128,256 个 token'),{c:'ar-sv small muted'});
    if(st===0){let vx=580, vy=40; VOCAB.forEach((v,i)=>{const k=TOKS.findIndex(tk=>tokTxt(tk.t)===v), w=v.length*7.4+18; if(vx+w>924){vx=580; vy+=36;}
        s+=`<rect x="${vx}" y="${vy-13}" width="${w}" height="24" rx="12" ${k>=0?`fill="${tint(k)}"`:'class="ar-vpill"'}/>`+sT(vx+9,vy+4,v,{c:'ar-sv small mono'+(k>=0?' tk':'')}); vx+=w+8;});
      s+=sT(580,vy+42,tx('… and about 128,000 more','……还有大约 128,000 个'),{c:'ar-sv small muted'});}
    else{const ord=[2,4,0,1,3]; ord.forEach((k,n)=>{const y=rows[k]; s+=`<rect x="566" y="${y-11}" width="200" height="22" rx="4" fill="${tint(k)}" fill-opacity=".28"/>`+sT(640,y+4,String(TOKS[k].id),{a:'end',c:'ar-sv mono'})+sT(652,y+4,tokTxt(TOKS[k].t),{c:'ar-sv mono'});
        const nx=n<4?rows[ord[n+1]]:rows[k]+46; if(nx-y>44) s+=sT(630,(y+nx)/2+5,'⋮',{a:'middle',c:'ar-sv muted'});});
      s+=sT(786,rows[2]+4,tx('← IDs in order','← 按 ID 排列'),{c:'ar-sv small muted'});}
    s+='</g>';}
  /* the table */
  if(st>=2){s+=tg('emb',`<rect x="560" y="20" width="80" height="310" rx="6" class="ar-wt"/>`+Array.from({length:50},(_,i)=>`<line x1="564" x2="636" y1="${26+i*6}" y2="${26+i*6}" class="ar-wl"/>`).join('')+
      rows.map((y,k)=>`<rect x="562" y="${y-3}" width="76" height="6" class="ar-wlit"/>`).join('')+sT(600,346,tx('embedding table','embedding 表'),{a:'middle',c:'ar-sv small w'}),tx('embedding table','embedding 表'));}
  if(st>=3){TOKS.forEach((tk,k)=>{s+=sStrip(660,rows[k]-7,16,14,VEC[k],tint(k),{cls:'ar-slide',style:`animation-delay:${k*110}ms`});});
    if(!bare) s+=`<g${NEW(true,500)}>`+sT(40,262,tx('“ is” as the 5th token','“ is” 是第 5 个 token'),{c:'ar-sv small muted'})+sStrip(250,250,16,14,VEC[4],tint(4))+
      sT(40,292,tx('“ is” as the 50th token','“ is” 是第 50 个 token'),{c:'ar-sv small muted'})+sStrip(250,280,16,14,VEC[4],tint(4))+sT(452,286,tx('the same row','同一行'),{c:'ar-sv small b'})+`</g>`;}
  if(st===0){const y=250; s+=`<g${NEW(true,600)}>`+sT(40,y-18,tx('A second example: one word, two tokens','再举个例子：一个词，两个 token'),{c:'ar-sv small muted'})+sT(40,y+12,'Tokenization',{c:'ar-sv big'})+`<path d="M200 ${y+6} L240 ${y+6}" class="ar-sl"/><path d="M234 ${y} L242 ${y+6} L234 ${y+12}" class="ar-sl"/>`+
      `<rect x="252" y="${y-10}" width="64" height="28" rx="14" fill="${tint(0)}"/>`+sT(262,y+9,'Token',{c:'ar-sv tokt'})+`<rect x="322" y="${y-10}" width="66" height="28" rx="14" fill="${tint(1)}"/>`+sT(332,y+9,'ization',{c:'ar-sv tokt'})+`</g>`;}
  return s+'</svg>';}

/* step 5: without positions, attention is order-blind. The same head's scores for the text as written and shuffled: every pair keeps
   its score, only its place in the grid moves */
function posBlind(nar){
  const S=scoreM(1), perm=[3,0,4,1,2], cs=nar?30:54, W=nar?360:960, H=nar?584:404;
  /* the demo's conditions (Codex review 1 #7): with positions or a causal mask the order does matter */
  const cond=sT(nar?180:480,H-(nar?14:10),tx('Shown here: one attention layer, no position encoding, no mask','演示条件：一层注意力，没有位置编码，没有 mask'),{a:'middle',c:'ar-sv small muted'});
  const grid=(x0,y0,ord,lab)=>{let g=sT(x0+cs*2.5,y0-(nar?60:34),lab,{a:'middle',c:'ar-sv small b'});
    ord.forEach((ti,c)=>{g+=nar?`<text transform="translate(${x0+c*cs+cs/2+3} ${y0-5}) rotate(-90)" class="ar-sv tokl" style="font-size:9px">${esc(tokTxt(TOKS[ti].t))}</text>`
      :sT(x0+c*cs+cs/2,y0-9,tokTxt(TOKS[ti].t),{a:'middle',c:'ar-sv tokl',st:'font-size:10px'});});
    ord.forEach((qi,r)=>{g+=sT(x0-6,y0+r*cs+cs/2+4,tokTxt(TOKS[qi].t),{a:'end',c:'ar-sv tokl',st:nar?'font-size:9px':''});
      ord.forEach((ki,c)=>{const v=S[qi][ki], on=qi===4&&ki===3; g+=`<rect x="${x0+c*cs+1}" y="${y0+r*cs+1}" width="${cs-2}" height="${cs-2}" rx="3" class="ar-sc${on?' hot':''}" fill-opacity="${(0.08+0.14*Math.abs(v)).toFixed(2)}"/>`+
        sT(x0+c*cs+cs/2,y0+r*cs+cs/2+4,(v<0?'−':'')+Math.abs(v).toFixed(1),{a:'middle',c:'ar-sv mono',st:nar?'font-size:10px':''});});});
    return g;};
  let s=svgOpen(W,H,L(P_EMB.steps[4]),nar);
  const ord0=[0,1,2,3,4];
  if(nar){s+=grid(92,86,ord0,tx('as written','原来的顺序'))+`<path d="M180 250 L180 270" class="ar-sl"/><path d="M174 264 L180 272 L186 264" class="ar-sl"/>`+sT(196,266,tx('shuffle','打乱'),{c:'ar-sv small muted'})+
    grid(92,362,perm,tx('shuffled','打乱以后'))+sT(180,H-38,tx('␣is · ␣France: 3.6 in both','␣is · ␣France：两边都是 3.6'),{a:'middle',c:'ar-sv small b'})+cond;}
  else{s+=grid(130,70,ord0,tx('as written','原来的顺序'))+`<g${NEW(true,200)}><path d="M${130+5*cs+30} 205 L${580-40} 205" class="ar-sl"/><path d="M${580-48} 199 L${580-40} 205 L${580-48} 211" class="ar-sl"/>`+
      sT((130+5*cs+580)/2,195,tx('shuffle','打乱'),{a:'middle',c:'ar-sv small muted'})+`</g>`+grid(580,70,perm,tx('shuffled','打乱以后'))+
      sT(480,H-32,tx('␣is looking at ␣France scores 3.6 in both: without positions, the order cannot matter','␣is 看 ␣France 两边都是 3.6：没有位置，顺序就不起作用'),{a:'middle',c:'ar-sv small b'})+cond;}
  return s+'</svg>';}
/* step 6: RoPE. Each position turns its query and key by its own angle (position × θ); the angle between a query at m and a key at
   n is (m − n) θ, so the same distance gives the same angle, and the same score, anywhere in the text */
function ropeSVG(nar){
  const th=30, W=nar?360:960, H=nar?400:330, rad=d=>d*Math.PI/180;
  const arrow=(cx,cy,Lr,deg,cls)=>{const a=rad(deg), x=cx+Lr*Math.cos(a), y=cy-Lr*Math.sin(a), b1=a+Math.PI*0.84, b2=a-Math.PI*0.84, h=Math.min(8,Lr*0.3);
    return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="${cls}"/><path d="M${x.toFixed(1)} ${y.toFixed(1)} L${(x+h*Math.cos(b1)).toFixed(1)} ${(y-h*Math.sin(b1)).toFixed(1)} L${(x+h*Math.cos(b2)).toFixed(1)} ${(y-h*Math.sin(b2)).toFixed(1)} Z" class="${cls} hd"/>`;};
  const dial=(cx,cy,r,m,n,lab)=>{const am=m*th%360, an=n*th%360, a0=rad(an), a1=rad(am), rr_=r*0.46;
    return `<circle cx="${cx}" cy="${cy}" r="${r}" class="ar-dial"/>`+`<path d="M${(cx+rr_*Math.cos(a0)).toFixed(1)} ${(cy-rr_*Math.sin(a0)).toFixed(1)} A${rr_} ${rr_} 0 0 0 ${(cx+rr_*Math.cos(a1)).toFixed(1)} ${(cy-rr_*Math.sin(a1)).toFixed(1)}" class="ar-arc"/>`+
      arrow(cx,cy,r-6,an,'ar-kar')+arrow(cx,cy,r-6,am,'ar-qar')+sT(cx,cy+r+(nar?18:22),lab,{a:'middle',c:'ar-sv small mono'});};
  let s=svgOpen(W,H,L(P_EMB.steps[5]),nar);
  const n=TOKS.length, sx=nar?36:70, dx=nar?72:108, yd=nar?48:92, rs=nar?14:22;
  TOKS.forEach((tk,i)=>{const x=sx+i*dx, pos=i+1, on=i===3||i===4, cls=i===4?'ar-qar':(i===3?'ar-kar':'ar-gar');
    s+=`<g${NEW(true,i*90)}><circle cx="${x}" cy="${yd}" r="${rs}" class="ar-dial${on?'':' dim'}"/>`+arrow(x,yd,rs-4,pos*th,cls)+
      `<rect x="${x-(nar?30:40)}" y="${yd+rs+8}" width="${nar?60:80}" height="${nar?20:24}" rx="${nar?10:12}" fill="${tint(i)}"/>`+sT(x,yd+rs+(nar?22:25),tokTxt(tk.t),{a:'middle',c:'ar-sv tokt',st:nar?'font-size:10px':''})+
      sT(x,yd+rs+(nar?44:52),`${pos}·θ`,{a:'middle',c:'ar-sv small mono'})+`</g>`;});
  if(nar){s+=`<g${NEW(true,600)}>`+tg('rope',dial(100,236,52,5,4,'m = 5, n = 4')+dial(260,236,52,15,14,'m = 15, n = 14'),'RoPE')+fxo(180,322,340,'\\text{angle} = (m - n)\\,\\theta',{a:'middle',h:30})+
      sT(180,370,tx('same content, same distance → same score','内容相同、距离相同 → 得分相同'),{a:'middle',c:'ar-sv small b'})+`</g>`;}
  else{s+=sT(sx-40,yd+rs+52+26,tx('each position turns its query and its key by the same angle, position × θ (θ = 30° here); the value is not turned','每个位置把自己的 query 和 key 转同一个角度：位置 × θ（这里 θ = 30°）；value 不转'),{c:'ar-sv small muted'});
    s+=`<g${NEW(true,600)}>`+tg('rope',dial(700,112,62,5,4,'m = 5, n = 4')+dial(870,112,62,15,14,'m = 15, n = 14'),'RoPE')+
      `<circle cx="640" cy="236" r="5" class="ar-qdot"/>`+sT(652,240,tx('query at m','位置 m 的 query'),{c:'ar-sv small'})+`<circle cx="780" cy="236" r="5" class="ar-kdot"/>`+sT(792,240,tx('key at n','位置 n 的 key'),{c:'ar-sv small'})+
      fxo(785,254,320,'\\text{angle} = (m - n)\\,\\theta',{a:'middle',h:30})+sT(785,306,tx('same content, same distance → same score','内容相同、距离相同 → 得分相同'),{a:'middle',c:'ar-sv small b'})+`</g>`;}
  return s+'</svg>';}
function popEmbN(st,bare){
  const W=360,H=st===0?350:(st===3&&!bare?390:320), rows=[0.3,0.56,0.08,0.8,0.18].map(f=>62+f*220), ry=k=>62+k*48; let s=svgOpen(W,H,L(P_EMB.steps[st]),1);
  s+=`<text x="14" y="30" class="ar-sv" style="font-size:17px;fill:var(--ink)">${esc(TOKS.map(t=>t.t).join(''))}</text>`;
  TOKS.forEach((tk,k)=>{const pw=tokTxt(tk.t).length*8.4+20, y=ry(k);
    s+=`<g${NEW(st===0,k*90)}>`+tg('tok',`<rect x="14" y="${y-14}" width="${pw}" height="28" rx="14" fill="${tint(k)}"/>`+sT(24,y+5,tokTxt(tk.t),{c:'ar-sv tokt'}),tokTxt(tk.t))+`</g>`;
    if(st>=1) s+=`<g${NEW(st===1,k*90)}>`+sT(14+pw+8,y+4,String(tk.id),{c:'ar-sv mono'})+`</g>`;
    if(st>=2){const x0=14+pw+8+String(tk.id).length*7.3+6; s+=`<g${NEW(st===2,k*120)}><path d="M${x0} ${y} C ${x0+40} ${y}, 210 ${rows[k]}, 240 ${rows[k]}" class="ar-sl id" fill="none"/></g>`;}});
  if(st>=2) s+=tg('emb',`<rect x="240" y="50" width="44" height="244" rx="6" class="ar-wt"/>`+Array.from({length:39},(_,i)=>`<line x1="243" x2="281" y1="${56+i*6}" y2="${56+i*6}" class="ar-wl"/>`).join('')+
      rows.map(y=>`<rect x="242" y="${y-3}" width="40" height="6" class="ar-wlit"/>`).join('')+sT(262,312,tx('embedding table','embedding 表'),{a:'middle',c:'ar-sv small w'}),tx('embedding table','embedding 表'));
  if(st>=3){TOKS.forEach((tk,k)=>{s+=sStrip(292,rows[k]-6,5,12,VEC[k],tint(k),{cls:'ar-slide',style:`animation-delay:${k*110}ms`});});
    if(!bare) s+=`<g${NEW(true,500)}>`+sT(14,344,tx('“ is”, 5th or 50th:','“ is” 不管是第 5 还是第 50 个：'),{c:'ar-sv small muted'})+sStrip(14,354,5,12,VEC[4],tint(4))+sT(82,364,tx('the same row','同一行'),{c:'ar-sv small b'})+`</g>`;}
  if(st<=1){const cx0=206, cw0=140; s+=`<g${NEW(true,200)}><rect x="${cx0}" y="44" width="${cw0}" height="${st===0?232:244}" rx="8" class="ar-vcard"/>`+sT(cx0+10,64,tx('vocabulary:','词表：'),{c:'ar-sv small muted'})+sT(cx0+10,80,tx('128,256 tokens','128,256 个 token'),{c:'ar-sv small muted'});
    if(st===0){let vx=cx0+10, vy=104; VOCAB.slice(0,12).forEach(v=>{const k=TOKS.findIndex(tk=>tokTxt(tk.t)===v), w=v.length*6.6+14; if(vx+w>cx0+cw0-8){vx=cx0+10; vy+=28;}
        s+=`<rect x="${vx}" y="${vy-11}" width="${w}" height="20" rx="10" ${k>=0?`fill="${tint(k)}"`:'class="ar-vpill"'}/>`+sT(vx+7,vy+3,v,{c:'ar-sv small mono'+(k>=0?' tk':''),st:'font-size:10px'}); vx+=w+5;});}
    else{[2,4,0,1,3].forEach((k,n)=>{const y=106+n*36; s+=`<rect x="${cx0+6}" y="${y-12}" width="${cw0-12}" height="22" rx="4" fill="${tint(k)}" fill-opacity=".28"/>`+sT(cx0+56,y+3,String(TOKS[k].id),{a:'end',c:'ar-sv mono',st:'font-size:11px'})+sT(cx0+64,y+3,tokTxt(TOKS[k].t),{c:'ar-sv mono',st:'font-size:11px'});});
      s+=sT(cx0+10,106+5*36-6,tx('in ID order','按 ID 排列'),{c:'ar-sv small muted'});}
    s+='</g>';}
  if(st===0){const y=318; s+=`<g${NEW(true,600)}>`+sT(14,y-24,tx('One word, two tokens:','一个词，两个 token：'),{c:'ar-sv small muted'})+sT(14,y+5,'Tokenization',{c:'ar-sv',st:'font-size:16px;fill:var(--ink)'})+`<path d="M122 ${y} L146 ${y}" class="ar-sl"/><path d="M140 ${y-5} L147 ${y} L140 ${y+5}" class="ar-sl"/>`+
      `<rect x="154" y="${y-13}" width="62" height="26" rx="13" fill="${tint(0)}"/>`+sT(164,y+5,'Token',{c:'ar-sv tokt'})+`<rect x="222" y="${y-13}" width="66" height="26" rx="13" fill="${tint(1)}"/>`+sT(232,y+5,'ization',{c:'ar-sv tokt'})+`</g>`;}
  return s+'</svg>';}

/* ---------- 2. self-attention, multi-head: the centrepiece ---------- */
const P_ATTN={t:{en:'Self-attention (multi-head)',zh:'自注意力（多头）'},steps:[
  {en:'Three blue matrices turn each token’s vector x into a query Q, a key K and a value V; RoPE then turns Q and K by the token’s position (↻), not V.',zh:'三个蓝色矩阵把每个 token 的向量 x 变成 query Q、key K、value V；RoPE 再按 token 的位置转动 Q 和 K（↻），V 不转。'},
  {en:'Q, K and V are cut into slices, one per head. Each head works on its own slices, independently.',zh:'Q、K、V 被切成几份，每个 head 一份。每个 head 只用自己那份，互不相干。'},
  {en:'One head computes a score for every pair of tokens:',zh:'一个 head 给每一对 token 算一个得分：',fx:'s = q \\cdot k \\div \\sqrt{d_{\\text{head}}}'},
  {en:'The causal mask hides later tokens; softmax turns each row into attention weights that add up to 100%.',zh:'因果 mask 挡住后面的 token；softmax 把每一行变成加起来是 100% 的注意力权重。'},
  {en:'Each token mixes the V slices by its attention weights.',zh:'每个 token 按注意力权重混合各个 V 切片。'},
  {en:'The heads meet only here: their outputs are joined, multiplied by Wo and added to the residual stream at ⊕.',zh:'各个 head 只在这里汇合：输出拼起来，乘 Wo，在 ⊕ 处加回残差流。'},
  {en:'Masks on 16 tokens: sparse masks compute fewer pairs, which makes very long contexts affordable.',zh:'16 个 token 上的 mask：稀疏的 mask 算的对更少，超长上下文才算得起。'}]};
const QKV=TOKS.map((t,i)=>{const r=mulberry(101+i*13), g=()=>Math.round((r()*2-1)*100)/100; return {q:Array.from({length:12},g),k:Array.from({length:12},g),v:Array.from({length:12},g)};});
const HCOL=['#6b7280','#6b7280','#6b7280','#6b7280'];
function popAttn(st){
  let s='';
  if(st>=2&&st<=4) s+=`<div class="ar-pctl"><span class="lbl">${tx('Head','head')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Head','head'))}" data-key="ahead">`+
     HEADS.map((h,i)=>`<button type="button" data-v="${i}" aria-pressed="${i===aHead}" data-tipx="${esc(tx(`Head ${i+1} (illustrative pattern): ${h.en}`,`第 ${i+1} 个 head（示意的模式）：${h.zh}`))}">H${i+1}</button>`).join('')+`</div><span class="ar-tag">${tx('Llama-3.1-8B: 32 heads','Llama-3.1-8B：32 个 head')}</span>${illus()}</div>`;
  if(st>=4&&st<=5) s+=`<div class="ar-pctl"><span class="lbl">${tx('Token','token')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Token','token'))}" data-key="atok">`+
     TOKS.map((tk,i)=>`<button type="button" data-v="${i}" aria-pressed="${i===aTok}">${esc(tokTxt(tk.t))}</button>`).join('')+`</div></div>`;
  if(st===6) s+=maskCtl();
  s+=`<div class="ar-pscroll">`+(st<=1?attnQKV(st):(st<=4?attnGrid(st):(st===5?attnMerge():maskSVG())))+`</div>`+(st===6&&innerWidth<=760?`<p class="small">${esc(L(MASKS.find(x=>x.k===aMask).u))}</p>`:'');
  return s;}
/* RoPE turns q and k, not v, by the token's position (Codex review 1 #7): one small dial per token next to Q and K, angle = position × 30° */
const rdial=(cx,cy,r,deg,cls)=>{const a=deg*Math.PI/180, x=cx+(r-1.5)*Math.cos(a), y=cy-(r-1.5)*Math.sin(a);
  return `<circle cx="${cx}" cy="${cy}" r="${r}" class="ar-dial"/><line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="${cls}"/>`;};
const ropeCols=(xs,hy,ry,r,dy)=>tg('rope',xs.map((x,c)=>sT(x,hy,'↻',{a:'middle',c:'ar-sv b'})+TOKS.map((tk,i)=>rdial(x,ry(i)+dy,r,(i+1)*30,c?'ar-kar':'ar-qar')).join('')).join(''),tx('RoPE turns Q and K','RoPE 转动 Q 和 K'));
function attnQKV(st){
  if(NAR()) return attnQKVN(st);
  const W=980,H=372, ry=i=>70+i*62, cw=10, split=st>=1, gp={every:3,w:8};
  let s=svgOpen(W,H,L(P_ATTN.steps[st])); if(!split) s=s.replace('viewBox="0 0 980 372"','viewBox="-60 0 880 372"');
  s+=tg('qkv',['Wq','Wk','Wv'].map((nm,i)=>`<rect x="${230+i*20}" y="40" width="16" height="${ry(4)-ry(0)+50}" rx="3" class="ar-wt"/><text transform="translate(${242+i*20} 70) rotate(-90)" text-anchor="middle" class="ar-sv small w">${nm}</text>`).join(''),'Wq Wk Wv');
  const colX=[310,480,650], lab=[['q','Q'],['k','K'],['v','V']], fills=['var(--ar-q)','var(--c-kv)','var(--ar-v)'];
  lab.forEach(([k,t],c)=>{s+=tg(k,`<rect x="${colX[c]}" y="18" width="${12*cw+(split?24:0)}" height="24" rx="5" class="ar-chd"/>`+sT(colX[c]+(12*cw+(split?24:0))/2,35,t,{a:'middle',c:'ar-sv b'}),t);});
  TOKS.forEach((tk,i)=>{const y=ry(i);
    s+=sT(84,y+12,tokTxt(tk.t),{a:'end',c:'ar-sv tokl'})+sStrip(94,y,cw,16,VEC[i],tint(i));
    s+=`<path d="M216 ${y+8} L230 ${y+8}" class="ar-sl"/><path d="M284 ${y+8} L304 ${y+8}" class="ar-sl"/>`;
    ['q','k','v'].forEach((k,c)=>{s+=sStrip(colX[c],y,cw,16,QKV[i][k],fills[c],split?{gap:gp,cls:st===1?'ar-slide':'',style:`animation-delay:${i*60}ms`}:{cls:st===0?'ar-grow':'',style:`animation-delay:${i*90+c*60}ms`});});});
  {const sw=12*cw+(split?24:0); s+=`<g${NEW(st===0,500)}>`+ropeCols([colX[0]+sw+13,colX[1]+sw+13],35,ry,7,8)+`</g>`;}
  if(split){for(let h=0;h<4;h++){colX.forEach(x=>{const x0=x+h*(3*cw+8)-3; s+=`<g${NEW(true,300+h*80)}><rect x="${x0}" y="${ry(0)-8}" width="${3*cw+6}" height="${ry(4)-ry(0)+32}" rx="5" class="ar-hbox"/></g>`;});
      s+=`<g${NEW(true,300+h*80)}>`+sT(colX[0]+h*(3*cw+8)+13,ry(4)+42,`H${h+1}`,{a:'middle',c:'ar-sv small b'})+`</g>`;}
    s+=`<g${NEW(true,700)}>`+tg('heads',`<rect x="820" y="${ry(0)-8}" width="150" height="${ry(4)-ry(0)+32}" rx="10" class="ar-note"/>`+
      sT(895,ry(1)+4,tx('4 heads drawn','这里画了 4 个 head'),{a:'middle',c:'ar-sv b'})+sT(895,ry(1)+26,tx('Llama-3.1-8B: 32','Llama-3.1-8B：32 个'),{a:'middle',c:'ar-sv small'})+
      sT(895,ry(3)-6,tx('no arrows between','head 之间'),{a:'middle',c:'ar-sv small muted'})+sT(895,ry(3)+12,tx('heads: independent','没有箭头：互不相干'),{a:'middle',c:'ar-sv small muted'}),tx('heads','head'))+`</g>`;}
  return s+'</svg>';}
function attnQKVN(st){
  const W=360, split=st>=1, ry=i=>54+i*50, cw=4, H=split?420:320, gp={every:3,w:3}; let s=svgOpen(W,H,L(P_ATTN.steps[st]),1);
  const colX=[158,222,286], sw=12*cw+(split?9:0), fills=['var(--ar-q)','var(--c-kv)','var(--ar-v)'];
  s+=tg('qkv',['Wq','Wk','Wv'].map((nm,i)=>`<rect x="${110+i*15}" y="${ry(0)-18}" width="12" height="${ry(4)-ry(0)+36}" rx="3" class="ar-wt"/><text transform="translate(${119.5+i*15} ${ry(0)+8}) rotate(-90)" text-anchor="middle" class="ar-sv w" style="font-size:9px">${nm}</text>`).join(''),'Wq Wk Wv');
  [['q','Q'],['k','K'],['v','V']].forEach(([k,t],c)=>{s+=tg(k,`<rect x="${colX[c]}" y="8" width="${sw}" height="20" rx="5" class="ar-chd"/>`+sT(colX[c]+sw/2,23,t,{a:'middle',c:'ar-sv b'}),t);});
  TOKS.forEach((tk,i)=>{const y=ry(i); s+=sT(52,y+10,tokTxt(tk.t),{a:'end',c:'ar-sv tokl',st:'font-size:10px'})+sStrip(56,y,cw,14,VEC[i],tint(i))+`<path d="M104 ${y+7} L108 ${y+7}" class="ar-sl"/><path d="M152 ${y+7} L156 ${y+7}" class="ar-sl"/>`;
    ['q','k','v'].forEach((k,c)=>{s+=sStrip(colX[c],y,cw,14,QKV[i][k],fills[c],split?{gap:gp,cls:st===1?'ar-slide':'',style:`animation-delay:${i*60}ms`}:{cls:st===0?'ar-grow':'',style:`animation-delay:${i*90+c*60}ms`});});});
  if(!split) s+=`<g${NEW(true,500)}>`+ropeCols([colX[0]+sw+8,colX[1]+sw+8],23,ry,5,7)+`</g>`;
  if(split){for(let h=0;h<4;h++){colX.forEach(x=>{const x0=x+h*(3*cw+3)-1.5; s+=`<g${NEW(true,300+h*80)}><rect x="${x0}" y="${ry(0)-5}" width="${3*cw+3}" height="${ry(4)-ry(0)+24}" rx="3" class="ar-hbox"/></g>`;});}
    s+=`<g${NEW(true,500)}>`+[0,1,2,3].map(h=>sT(colX[0]+h*(3*cw+3)+6,ry(4)+34,`H${h+1}`,{a:'middle',c:'ar-sv b',st:'font-size:9px'})).join('')+`</g>`;
    s+=`<g${NEW(true,700)}>`+tg('heads',`<rect x="14" y="${ry(4)+48}" width="332" height="84" rx="10" class="ar-note"/>`+sT(180,ry(4)+72,tx('4 heads drawn; Llama-3.1-8B has 32','这里画了 4 个 head；Llama-3.1-8B 有 32 个'),{a:'middle',c:'ar-sv b'})+
      sT(180,ry(4)+96,tx('no arrows between heads:','head 之间没有箭头：'),{a:'middle',c:'ar-sv small muted'})+sT(180,ry(4)+114,tx('each works on its own slices','每个 head 只用自己那份'),{a:'middle',c:'ar-sv small muted'}),tx('heads','head'))+`</g>`;}
  return s+'</svg>';}
function attnGrid(st){
  if(NAR()) return attnGridN(st);
  const W=980,H=392, cs=54, gx=170, gy=62, S=scoreM(aHead), Wm=weightM(aHead,true), pc=Wm.map(r=>pcts(r));
  let s=svgOpen(W,H,L(P_ATTN.steps[st]));
  s+=tg('k',sT(gx+cs*2.5,20,tx('keys (K), one per token','key（K），每个 token 一个'),{a:'middle',c:'ar-sv small'}),'K');
  s+=tg('q',`<g transform="translate(22 ${gy+cs*2.5}) rotate(-90)">`+sT(0,0,tx('queries (Q)','query（Q）'),{a:'middle',c:'ar-sv small'})+`</g>`,'Q');
  TOKS.forEach((tk,j)=>{s+=sT(gx+j*cs+cs/2,gy-18,tokTxt(tk.t),{a:'middle',c:'ar-sv tokl'})+`<rect x="${gx+j*cs+cs/2-9}" y="${gy-11}" width="18" height="6" rx="2" fill="var(--c-kv)"/>`;
    s+=sT(gx-26,gy+j*cs+cs/2+4,tokTxt(tk.t),{a:'end',c:'ar-sv tokl'})+`<rect x="${gx-20}" y="${gy+j*cs+cs/2-3}" width="12" height="6" rx="2" fill="var(--ar-q)"/>`;});
  for(let i=0;i<5;i++) for(let j=0;j<5;j++){const x=gx+j*cs, y=gy+i*cs, masked=st>=3&&j>i, d=(i+j)*35;
    const cls=`ar-cell${masked?' msk':''}${st>=4&&i===aTok?' sel':''}`, tip=tx(`q of ${tokTxt(TOKS[i].t)} · k of ${tokTxt(TOKS[j].t)}`,`${tokTxt(TOKS[i].t)} 的 q · ${tokTxt(TOKS[j].t)} 的 k`);
    let inner;
    if(st===2) inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" class="ar-sc" fill-opacity="${(0.08+0.14*Math.abs(S[i][j])).toFixed(2)}"/>`+sT(x+cs/2,y+cs/2+4,(S[i][j]<0?'−':'')+Math.abs(S[i][j]).toFixed(1),{a:'middle',c:'ar-sv mono'});
    else if(masked) inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" fill="url(#ar-hatch)"/>`;
    else inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" class="ar-aw" fill-opacity="${(0.1+0.9*Wm[i][j]).toFixed(2)}"/>`+sT(x+cs/2,y+cs/2+4,Math.round(pc[i][j])+'%',{a:'middle',c:`ar-sv mono aw${Wm[i][j]>0.6?' hi':''}`});
    s+=`<g class="${cls}" data-i="${i}" data-j="${j}" data-tipx="${esc(tip)}"${st===3?` style="animation-delay:${d}ms"`:''}>${inner}</g>`;}
  if(st>=3){for(let i=0;i<5;i++) s+=`<g${NEW(st===3,400+i*60)}>`+sT(gx+5*cs+12,gy+i*cs+cs/2+4,'= 100%',{c:'ar-sv small mono'})+`</g>`;
    s+=`<g${NEW(st===3,200)}>`+tg('mask',`<rect x="${gx+3*cs}" y="${gy-4+5*cs+14}" width="14" height="14" fill="url(#ar-hatch)"/>`+sT(gx+3*cs+20,gy+5*cs+21,tx('causal mask','因果 mask'),{c:'ar-sv small'}),tx('causal mask','因果 mask'))+
      tg('softmax',sT(gx,gy+5*cs+21,tx('softmax per row →','每行做 softmax →'),{c:'ar-sv small'}),'softmax')+`</g>`;}
  else s+=tg('score',fxo(gx,gy+5*cs+2,300,'s_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d_{head}}}',{h:50,pre:tx('score','得分')}),tx('score','得分'));
  if(st===4){const w=Wm[aTok], vx=620, sy=j=>gy+j*cs+cs/2, sx=880, syc=gy+cs*2.5, mixv=[0,1,2].map(c=>w.reduce((a,wj,j)=>a+wj*QKV[j].v[aHead*3+c],0));
    s+=`<g${NEW(true,0)}>`+sT(vx+36,gy-10,tx('V slices','V 切片'),{a:'middle',c:'ar-sv small b'});
    TOKS.forEach((tk,j)=>{const on=j<=aTok&&w[j]>0.0005;
      s+=sStrip(vx,sy(j)-9,24,18,QKV[j].v.slice(aHead*3,aHead*3+3),'var(--ar-v)',{style:on?'':'opacity:.25'});
      if(on) s+=`<path d="M${vx+78} ${sy(j)} C ${vx+150} ${sy(j)}, ${sx-80} ${syc}, ${sx-22} ${syc}" class="ar-flow" style="stroke-width:${Math.max(1,w[j]*16).toFixed(1)};animation-delay:${j*120}ms"/>`+sT(vx-8,sy(j)+4,Math.round(pcts(w)[j])+'%',{a:'end',c:'ar-sv small mono'});});
    s+=tg('aw',`<circle cx="${sx}" cy="${syc}" r="18" class="ar-sum"/>`+sT(sx,syc+6,'Σ',{a:'middle',c:'ar-sv b big'}),tx('weighted sum','加权求和'))+sStrip(sx+26,syc-9,24,18,mixv,'var(--ar-v)',{cls:'ar-grow',style:'animation-delay:700ms'})+
      sT(sx+98,syc+36,tx(`output of ${tokTxt(TOKS[aTok].t)}`,`${tokTxt(TOKS[aTok].t)} 的输出`),{a:'end',c:'ar-sv small'})+`</g>`;}
  if(st===2||st===3){const i=aTok, row=S[i], jb=row.reduce((b,v,j)=>j<=i&&v>row[b]?j:b,0), px0=580, pw0=370, py0=gy-8, hd=MODELS[DEF_M].d/MODELS[DEF_M].H;
    s+=`<g${NEW(true,300)}><rect x="${px0}" y="${py0}" width="${pw0}" height="${5*cs+16}" rx="10" class="ar-note"/>`;
    if(st===2){const q=QKV[i].q.slice(aHead*3,aHead*3+3), kk=QKV[jb].k.slice(aHead*3,aHead*3+3), yy=py0+118;
      s+=sT(px0+18,py0+30,tx(`One score: ${tokTxt(TOKS[i].t)} looking at ${tokTxt(TOKS[jb].t)}`,`一个得分：${tokTxt(TOKS[i].t)} 看 ${tokTxt(TOKS[jb].t)}`),{c:'ar-sv b'})+
        sT(px0+18,yy-34,tx(`q of ${tokTxt(TOKS[i].t)}`,`${tokTxt(TOKS[i].t)} 的 q`),{c:'ar-sv small muted'})+sStrip(px0+18,yy-22,26,22,q,'var(--ar-q)')+sT(px0+108,yy-6,'·',{a:'middle',c:'ar-sv b big'})+
        sT(px0+120,yy-34,tx(`k of ${tokTxt(TOKS[jb].t)}`,`${tokTxt(TOKS[jb].t)} 的 k`),{c:'ar-sv small muted'})+sStrip(px0+120,yy-22,26,22,kk,'var(--c-kv)')+
        fxo(px0+206,yy-26,80,`\\div \\sqrt{${hd}} =`,{h:34})+`<text x="${px0+pw0-18}" y="${yy}" text-anchor="end" class="ar-sv huge mono">${(row[jb]<0?'−':'')+Math.abs(row[jb]).toFixed(1)}</text>`+
        fxo(px0+18,yy+30,340,`d_{head} = 4096 \\div 32 = ${hd}`,{pre:'Llama-3.1-8B:',c:'sm'})+
        sT(px0+18,yy+72,tx('a larger score: more attention, after softmax','得分越大，softmax 之后分到的注意力越多'),{c:'ar-sv small muted'})+
        sT(px0+18,yy+94,tx('the scores need not add up to anything yet','这些得分现在还不用加起来等于什么'),{c:'ar-sv small muted'});}
    else{const w=Wm[i], pcI=pcts(w), bx0=px0+150, bwm=pw0-230, ry0=py0+64;
      s+=sT(px0+18,py0+30,tx(`Row ${tokTxt(TOKS[i].t)}: scores → softmax → weights`,`${tokTxt(TOKS[i].t)} 这一行：得分 → softmax → 权重`),{c:'ar-sv b'});
      TOKS.forEach((tk,j)=>{const y=ry0+j*36, m=j>i;
        s+=sT(px0+18,y+4,tokTxt(tk.t),{c:'ar-sv tokl'})+sT(px0+130,y+4,m?tx('masked','挡住'):(row[j]<0?'−':'')+Math.abs(row[j]).toFixed(1),{a:'end',c:'ar-sv small mono'+(m?' muted':'')})+
          (m?`<rect x="${bx0}" y="${y-7}" width="${bwm}" height="14" rx="3" fill="url(#ar-hatch)"/>`:`<rect x="${bx0}" y="${y-7}" width="${Math.max(2,bwm*w[j]).toFixed(1)}" height="14" rx="3" class="ar-aw"/>`+sT(bx0+Math.max(2,bwm*w[j])+6,y+4,Math.round(pcI[j])+'%',{c:'ar-sv small mono'}));});
      s+=sT(px0+18,ry0+5*36-2,tx('the weights add up to 100%','权重加起来是 100%'),{c:'ar-sv small muted'});}
    s+='</g>';}
  return s+'</svg>';}
function attnGridN(st){
  const cs=44, gx=86, gy=58, S=scoreM(aHead), Wm=weightM(aHead,true), pc=Wm.map(r=>pcts(r)), H=st===4?520:340; let s=svgOpen(360,H,L(P_ATTN.steps[st]),1);
  s+=tg('k',sT(gx+cs*2.5,14,tx('keys (K)','key（K）'),{a:'middle',c:'ar-sv small'}),'K');
  TOKS.forEach((tk,j)=>{s+=sT(gx+j*cs+cs/2,gy-16,tokTxt(tk.t),{a:'middle',c:'ar-sv tokl',st:'font-size:9px'})+`<rect x="${gx+j*cs+cs/2-8}" y="${gy-9}" width="16" height="5" rx="2" fill="var(--c-kv)"/>`;
    s+=sT(gx-20,gy+j*cs+cs/2+4,tokTxt(tk.t),{a:'end',c:'ar-sv tokl',st:'font-size:10px'})+`<rect x="${gx-15}" y="${gy+j*cs+cs/2-3}" width="10" height="6" rx="2" fill="var(--ar-q)"/>`;});
  for(let i=0;i<5;i++) for(let j=0;j<5;j++){const x=gx+j*cs, y=gy+i*cs, masked=st>=3&&j>i;
    const cls=`ar-cell${masked?' msk':''}${st>=4&&i===aTok?' sel':''}`, tip=tx(`q of ${tokTxt(TOKS[i].t)} · k of ${tokTxt(TOKS[j].t)}`,`${tokTxt(TOKS[i].t)} 的 q · ${tokTxt(TOKS[j].t)} 的 k`);
    let inner;
    if(st===2) inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" class="ar-sc" fill-opacity="${(0.08+0.14*Math.abs(S[i][j])).toFixed(2)}"/>`+sT(x+cs/2,y+cs/2+4,(S[i][j]<0?'−':'')+Math.abs(S[i][j]).toFixed(1),{a:'middle',c:'ar-sv mono',st:'font-size:11px'});
    else if(masked) inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" fill="url(#ar-hatch)"/>`;
    else inner=`<rect x="${x+1}" y="${y+1}" width="${cs-2}" height="${cs-2}" rx="4" class="ar-aw" fill-opacity="${(0.1+0.9*Wm[i][j]).toFixed(2)}"/>`+sT(x+cs/2,y+cs/2+4,Math.round(pc[i][j])+'%',{a:'middle',c:`ar-sv mono aw${Wm[i][j]>0.6?' hi':''}`,st:'font-size:11px'});
    s+=`<g class="${cls}" data-i="${i}" data-j="${j}" data-tipx="${esc(tip)}">${inner}</g>`;}
  const ly=gy+5*cs+22;
  if(st>=3){s+=`<g${NEW(st===3,200)}>`+tg('softmax',sT(14,ly,tx('softmax per row: each = 100%','每行 softmax：各 = 100%'),{c:'ar-sv small'}),'softmax')+tg('mask',`<rect x="270" y="${ly-11}" width="13" height="13" fill="url(#ar-hatch)"/>`+sT(288,ly,tx('mask','mask'),{c:'ar-sv small'}),tx('causal mask','因果 mask'))+`</g>`;}
  else s+=tg('score',fxo(14,ly-18,300,'s_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d_{head}}}',{h:50,pre:tx('score','得分')}),tx('score','得分'));
  if(st===4){const w=Wm[aTok], y0=ly+26, sy=j=>y0+14+j*34, sx=236, syc=y0+14+2*34, mixv=[0,1,2].map(c=>w.reduce((a,wj,j)=>a+wj*QKV[j].v[aHead*3+c],0));
    s+=`<g${NEW(true,0)}>`+sT(64,y0-2,tx('V slices','V 切片'),{a:'middle',c:'ar-sv small b'});
    TOKS.forEach((tk,j)=>{const on=j<=aTok&&w[j]>0.0005;
      s+=sStrip(40,sy(j)-8,16,16,QKV[j].v.slice(aHead*3,aHead*3+3),'var(--ar-v)',{style:on?'':'opacity:.25'});
      if(on) s+=sT(34,sy(j)+4,Math.round(pcts(w)[j])+'%',{a:'end',c:'ar-sv small mono'})+`<path d="M92 ${sy(j)} C 150 ${sy(j)}, ${sx-60} ${syc}, ${sx-20} ${syc}" class="ar-flow" style="stroke-width:${Math.max(1,w[j]*14).toFixed(1)};animation-delay:${j*120}ms"/>`;});
    s+=tg('aw',`<circle cx="${sx}" cy="${syc}" r="16" class="ar-sum"/>`+sT(sx,syc+6,'Σ',{a:'middle',c:'ar-sv b big'}),tx('weighted sum','加权求和'))+sStrip(sx+24,syc-8,16,16,mixv,'var(--ar-v)',{cls:'ar-grow',style:'animation-delay:700ms'})+
      sT(sx+48,syc+30,tx(`output of ${tokTxt(TOKS[aTok].t)}`,`${tokTxt(TOKS[aTok].t)} 的输出`),{a:'middle',c:'ar-sv small'})+`</g>`;}
  return s+'</svg>';}
/* step 6: the heads meet only here. Each head's output for the chosen token sits side by side, drops straight into its slot of one
   joined strip, and one path goes on through Wo to the output and ⊕: no line crosses another (review 12) */
function attnMergeN(){
  const W=360,H=470; let s=svgOpen(W,H,L(P_ATTN.steps[5]),1);
  const outs=HEADS.map((h,hi)=>{const w=weightM(hi,true)[aTok]; return [0,1,2].map(c=>w.reduce((a,wj,j)=>a+wj*QKV[j].v[hi*3+c],0));});
  s+=sT(14,20,tx(`each head’s output for ${tokTxt(TOKS[aTok].t)}`,`${tokTxt(TOKS[aTok].t)} 在各个 head 的输出`),{c:'ar-sv small muted'});
  const hx=hi=>22+hi*84, sx=hi=>48+hi*66;
  outs.forEach((o,hi)=>{s+=sT(hx(hi)+30,44,`H${hi+1}`,{a:'middle',c:'ar-sv b',st:'font-size:11px'})+sStrip(hx(hi),52,20,18,o,'var(--ar-v)')+
    `<path d="M${hx(hi)+30} 74 L${sx(hi)+30} 118" class="ar-flow" style="animation-delay:${hi*120}ms"/>`;});
  s+=`<g class="ar-new" style="animation-delay:450ms"><rect x="${sx(0)-6}" y="120" width="${4*66+12}" height="30" rx="6" class="ar-hbox"/>`+outs.map((o,hi)=>sStrip(sx(hi),125,20,20,o,'var(--ar-v)')).join('')+
     fxo(180,156,300,'\\mathrm{concat}(H_1, \\ldots, H_4)',{a:'middle',pre:tx('joined:','拼起来：'),c:'sm'})+`</g>`;
  s+=`<g class="ar-new" style="animation-delay:800ms"><path d="M180 186 L180 200" class="ar-sl"/>`+tg('wo',`<rect x="150" y="202" width="60" height="44" rx="6" class="ar-wt"/>`+sT(180,229,'Wo',{a:'middle',c:'ar-sv b w'}),'Wo')+`<path d="M180 248 L180 264" class="ar-sl"/>`+sStrip(84,268,16,18,VEC[aTok].map((v,i)=>Math.sin(v*2+i*0.7)),tint(aTok))+`</g>`;
  s+=`<g class="ar-new" style="animation-delay:1100ms"><path d="M180 290 L180 306" class="ar-sl"/>`+tg('res',`<circle cx="180" cy="324" r="17" class="ar-sum"/>`+sT(180,330,'+',{a:'middle',c:'ar-sv b big'}),'⊕')+sT(206,329,tx('residual','残差流'),{c:'ar-sv small'})+`</g>`;
  s+=`<g class="ar-new" style="animation-delay:1300ms"><rect x="14" y="362" width="332" height="62" rx="10" class="ar-note hot"/>`+sT(180,388,tx('The heads meet here, and only here:','各个 head 在这里、也只在这里汇合：'),{a:'middle',c:'ar-sv b'})+
     sT(180,410,tx('inside a head, no head sees another','在 head 内部，谁也看不到别的 head'),{a:'middle',c:'ar-sv small'})+`</g>`;
  return s+'</svg>';}
function attnMerge(){
  if(NAR()) return attnMergeN();
  const W=980,H=312; let s=svgOpen(W,H,L(P_ATTN.steps[5]));
  const outs=HEADS.map((h,hi)=>{const w=weightM(hi,true)[aTok]; return [0,1,2].map(c=>w.reduce((a,wj,j)=>a+wj*QKV[j].v[hi*3+c],0));});
  const hx=hi=>110+hi*104, cx0=130, sx=hi=>cx0+hi*80, ys=62, yc=150;
  s+=sT(40,28,tx(`each head’s output for ${tokTxt(TOKS[aTok].t)}`,`${tokTxt(TOKS[aTok].t)} 在各个 head 的输出`),{c:'ar-sv small muted'});
  outs.forEach((o,hi)=>{s+=sT(hx(hi)+40,ys-8,`H${hi+1}`,{a:'middle',c:'ar-sv b'})+sStrip(hx(hi)+4,ys,24,22,o,'var(--ar-v)')+
    `<path d="M${hx(hi)+40} ${ys+26} L${sx(hi)+40} ${yc-2}" class="ar-flow" style="animation-delay:${hi*120}ms"/>`;});
  s+=`<g class="ar-new" style="animation-delay:450ms"><rect x="${cx0-6}" y="${yc}" width="${4*80+12}" height="32" rx="6" class="ar-hbox"/>`+outs.map((o,hi)=>sStrip(sx(hi)+4,yc+5,24,22,o,'var(--ar-v)')).join('')+
     fxo(cx0+160,yc+40,340,'\\mathrm{concat}(H_1, \\ldots, H_4)',{a:'middle',pre:tx('joined:','拼起来：'),c:'sm'})+`</g>`;
  const xr=cx0+4*80+6, ym=yc+16;
  s+=`<g class="ar-new" style="animation-delay:800ms"><path d="M${xr} ${ym} L${xr+44} ${ym}" class="ar-sl"/>`+tg('wo',`<rect x="${xr+46}" y="${ym-34}" width="64" height="68" rx="6" class="ar-wt"/>`+sT(xr+78,ym+5,'Wo',{a:'middle',c:'ar-sv b w'}),'Wo')+
     `<path d="M${xr+112} ${ym} L${xr+144} ${ym}" class="ar-sl"/>`+sStrip(xr+146,ym-10,10,20,VEC[aTok].map((v,i)=>Math.sin(v*2+i*0.7)),tint(aTok))+`</g>`;
  s+=`<g class="ar-new" style="animation-delay:1100ms"><path d="M${xr+268} ${ym} L${xr+290} ${ym}" class="ar-sl"/>`+tg('res',`<circle cx="${xr+310}" cy="${ym}" r="18" class="ar-sum"/>`+sT(xr+310,ym+6,'+',{a:'middle',c:'ar-sv b big'}),'⊕')+
     sT(xr+310,ym+44,tx('residual','残差流'),{a:'middle',c:'ar-sv small'})+`</g>`;
  s+=`<g class="ar-new" style="animation-delay:1300ms"><rect x="${xr+40}" y="${yc+86}" width="420" height="56" rx="10" class="ar-note hot"/>`+sT(xr+250,yc+110,tx('The heads meet here, and only here:','各个 head 在这里、也只在这里汇合：'),{a:'middle',c:'ar-sv b'})+
     sT(xr+250,yc+130,tx('inside a head, no head sees another','在 head 内部，谁也看不到别的 head'),{a:'middle',c:'ar-sv small'})+`</g>`;
  return s+'</svg>';}
/* masks on a longer sequence */
const MASKS=[
 {k:'causal',t:{en:'Causal',zh:'因果'},f:(i,j)=>j<=i,u:{en:'Every decoder-only LLM: GPT, Llama.',zh:'所有 decoder-only LLM：GPT、Llama。'}},
 {k:'window',t:{en:'Sliding window',zh:'滑动窗口'},f:(i,j)=>j<=i&&i-j<4,u:{en:'Mistral 7B v0.1: a 4,096-token window; Gemma 2 and 3 alternate local (window) and global layers.',zh:'Mistral 7B v0.1：4,096 个 token 的窗口；Gemma 2、3 让局部（窗口）层和全局层交替。'}},
 {k:'sink',t:{en:'Sink + window',zh:'sink + 窗口'},f:(i,j)=>j<=i&&(j===0||i-j<3),u:{en:'StreamingLLM: keep the first tokens (attention sinks) plus a window, for endless streams.',zh:'StreamingLLM：保留开头几个 token（attention sink）加一个窗口，用于无限长的流。'}},
 {k:'strided',t:{en:'Strided',zh:'跨步'},f:(i,j)=>j<=i&&(i-j<4||(i-j)%4===0),u:{en:'Sparse Transformer, Longformer and BigBird mix local, strided (dilated) and global patterns.',zh:'Sparse Transformer、Longformer、BigBird 混用局部、跨步（dilated）和全局的模式。'}},
 {k:'bi',t:{en:'Bidirectional',zh:'双向'},f:()=>true,u:{en:'Encoders such as BERT: every token sees every token, including later ones.',zh:'BERT 这样的 encoder：每个 token 看得到所有 token，包括后面的。'}}];
/* the pairs each mask computes at 1M tokens (n = 2^20), from its rule (tests/architecture_numbers.py recomputes them):
   causal n(n+1)/2; window 4,096; 4 sinks + a 4,092 window; strided = the local 1,024 plus every 1,024th (√n, Sparse Transformer); n² */
const M1M={causal:549756338176,window:4286580736,sink:4286580736,strided:1609564672,bi:1099511627776,
  lab:{causal:{en:'causal',zh:'因果'},window:{en:'window of 4,096',zh:'4,096 窗口'},sink:{en:'4 sinks + 4,092 window',zh:'4 个 sink + 4,092 窗口'},strided:{en:'local 1,024 + every 1,024th',zh:'局部 1,024 + 每隔 1,024'},bi:{en:'bidirectional n²',zh:'双向 n²'}},
  note:{causal:{en:'Every token looks at itself and every earlier token.',zh:'每个 token 看自己和前面所有的 token。'},window:{en:'Information still travels further, one window per layer.',zh:'信息仍能传得更远：每过一层，多走一个窗口。'},
    sink:{en:'The same count as the window; the first tokens stay in view.',zh:'和窗口一样多；开头几个 token 一直看得到。'},strided:{en:'Two hops (two layers) reach any earlier token.',zh:'两跳（两层）就能到达任何更早的 token。'},
    bi:{en:'Every token also sees later ones, so it cannot generate text.',zh:'每个 token 也看后面的，所以不能用来生成文字。'}}};
function maskCtl(){
  return `<div class="ar-pctl"><span class="lbl">${tx('Mask','mask')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Mask','mask'))}" data-key="amask">`+
    MASKS.map(m=>`<button type="button" data-v="${m.k}" aria-pressed="${m.k===aMask}" data-tipx="${esc(L(m.u))}">${esc(L(m.t))}</button>`).join('')+`</div>`+
    `<button type="button" class="btn ghost sm" data-act="a1m" aria-pressed="${a1M}">${tx('At 1M tokens','放到 1M 个 token')}</button></div>`;}
function maskSVG(){
  const nar=innerWidth<=760, n=16, cs=17, m=MASKS.find(x=>x.k===aMask), W=nar?420:980, H=nar?(a1M?600:480):340, ox=nar?40:360, oy=nar?330:0; let cnt=0, s=nar?svgOpen(W,H,L(m.t)).replace('class="ar-psvg"','class="ar-psvg fit"'):svgOpen(W,H,L(m.t));
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){const on=m.f(i,j); if(on) cnt++; const x=40+j*cs, y=30+i*cs;
    s+=on?`<rect x="${x}" y="${y}" width="${cs-2}" height="${cs-2}" rx="2" class="ar-aw" fill-opacity="${(0.35+0.5*((i*7+j*3)%5)/4).toFixed(2)}" style="animation-delay:${(i+j)*12}ms"/>`:`<rect x="${x}" y="${y}" width="${cs-2}" height="${cs-2}" rx="2" fill="url(#ar-hatch)"/>`;}
  s+=sT(40+n*cs/2,22,tx('keys →','key →'),{a:'middle',c:'ar-sv small muted'})+`<g transform="translate(30 ${30+n*cs/2}) rotate(-90)">`+sT(0,0,tx('queries →','query →'),{a:'middle',c:'ar-sv small muted'})+`</g>`;
  const bw=nar?340:300, nw=nar?360:560;
  s+=sT(ox,oy+70,tx('pairs computed','要算的对'),{c:'ar-sv small muted'})+`<text x="${ox}" y="${oy+112}" class="ar-sv huge mono">${cnt} <tspan class="ar-sv muted">${tx('of','/')} 256</tspan></text>`;
  s+=`<rect x="${ox}" y="${oy+130}" width="${bw}" height="10" rx="5" class="ar-trk"/><rect x="${ox}" y="${oy+130}" width="${(bw*cnt/256).toFixed(1)}" height="10" rx="5" class="ar-aw"/>`;
  if(!nar) s+=sT(ox,oy+176,L(m.u),{c:'ar-sv small'});
  if(a1M){const y0=oy+(nar?150:200), c1=M1M.causal, v=M1M[m.k], r=c1/v, sup=e=>String(e).split('').map(d=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join(''),
      sci=x=>{const e=Math.floor(Math.log10(x)); return `${(x/10**e).toFixed(1)} × 10${sup(e)}`;},
      cmp=m.k==='causal'?tx('the baseline','基准'):(r>=1?tx(`${Math.round(r)}× fewer`,`少 ${Math.round(r)} 倍`):tx(`${Math.round(1/r)}× as many`,`是因果的 ${Math.round(1/r)} 倍`));
    s+=`<g class="ar-new"><rect x="${ox}" y="${y0}" width="${nw}" height="${nar?108:112}" rx="10" class="ar-note"/>`+
      sT(ox+14,y0+24,tx('At 1M tokens (1,048,576), per head and layer:','放到 1M 个 token（1,048,576），每个 head 每层：'),{c:'ar-sv b'})+
      sT(ox+14,y0+48,tx(`full causal ≈ ${sci(c1)} pairs`,`完整因果 ≈ ${sci(c1)} 对`),{c:'ar-sv mono'})+
      (nar?sT(ox+14,y0+70,`${L(M1M.lab[m.k])}:`,{c:'ar-sv mono'})+sT(ox+14,y0+92,`≈ ${sci(v)}: ${cmp}`,{c:'ar-sv mono'})
        :sT(ox+14,y0+70,`${L(M1M.lab[m.k])} ≈ ${sci(v)}${m.k==='causal'?'':tx(' pairs',' 对')}: ${cmp}`,{c:'ar-sv mono'}))+
      (nar?'':sT(ox+14,y0+94,L(M1M.note[m.k]),{c:'ar-sv small muted'}))+`</g>`;}
  return s+'</svg>';}

/* ---------- 3. MLP (SwiGLU) ---------- */
const P_MLP={t:{en:'MLP (SwiGLU)',zh:'MLP（SwiGLU）'},steps:[
  {en:'A classic MLP: widen, a nonlinearity, narrow back.',zh:'经典的 MLP：放宽，过一个非线性，再收窄。'},
  {en:'up proposes features; gate decides how much of each passes (SiLU).',zh:'up 提出候选特征；gate 决定每个特征放行多少（SiLU）。'},
  {en:'gate × up, pair by pair, then down back to d.',zh:'gate × up，一对一对相乘，再由 down 变回 d 维。'},
  {en:'The result is added to the residual stream.',zh:'结果加回残差流。'}]};
const WIDE=TOKS.map((t,i)=>{const r=mulberry(301+i*17), g=()=>Math.round((r()*2-1)*2.4*100)/100; return {g:Array.from({length:42},g),u:Array.from({length:42},()=>Math.round((r()*2-1)*100)/100)};});
/* the SiLU plot; the probe is kept inside the plot; its readout (MathML) sits under it */
function siluPlot(x0,y0,w,h,px){const X=v=>x0+7+(v+5)/10*(w-14), Y=v=>y0+h-7-(v+1)/6*(h-14); let d='';
  for(let i=0;i<=60;i++){const v=-5+i/6; d+=(i?'L':'M')+X(v).toFixed(1)+' '+Y(silu(v)).toFixed(1);}
  const sy=silu(px), f=v=>(v<0?'-':'')+Math.abs(v).toFixed(v===Math.round(v)?0:(Math.abs(v)<1&&String(v).length>4?2:1));
  return `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" rx="6" class="ar-plot"/><line x1="${x0}" x2="${x0+w}" y1="${Y(0)}" y2="${Y(0)}" class="ar-ax"/><line x1="${X(0)}" x2="${X(0)}" y1="${y0}" y2="${y0+h}" class="ar-ax"/>`+
    `<path d="${d}" class="ar-curve"/><circle cx="${X(px).toFixed(1)}" cy="${Y(sy).toFixed(1)}" r="5" class="ar-probe"/>`+
    fxo(x0+w/2,y0+h+6,w+60,`\\mathrm{SiLU}(${px.toFixed(1)}) = ${sy.toFixed(2)}`.replace(/\((-)/,'(−').replace(/= -/,'= −'),{a:'middle',h:28,c:'sm'});}
/* the MLP as a network: 6 neurons stand for d, 16 for the 3.5× wider middle. Step 1 is the classic MLP; from step 2 the middle is
   Llama's two rows: up (candidate features) and gate (switches through SiLU: bright = open, dim = closed); step 3 multiplies each gate
   neuron with its own up neuron and down brings the result back to d; step 4 adds it to the residual stream. The SiLU probe drives one
   gate neuron (ringed in the probe's colour). */
const GK=12;
function mlpGeo(nar){return nar?{W:360,xh:k=>60+k*18.2,xi:i=>92+i*38,yI:34,yU:112,yG:168,yP:222,yO:284,yR:336,r:6,ri:8,lx:4}
  :{W:980,xh:k=>200+k*38,xi:i=>485+(i-2.5)*62,yI:40,yU:140,yG:200,yP:262,yO:330,yR:392,r:7,ri:9,lx:16};}
function mlpVals(){const n=16, g=WIDE[4].g.slice(0,n).map((v,k)=>k===GK?siluX:v), u=WIDE[4].u.slice(0,n), sl=g.map(silu);
  return {g,u,sl,gb:sl.map(v=>clamp(v/2.2,0,1)),pv:sl.map((v,k)=>v*u[k])};}
function popMlp(st){
  const nar=NAR(), G=mlpGeo(nar), V=mlpVals(), n=16, I=6, W=G.W;
  const yH=nar?150:170, yO0=nar?262:292, H=st===0?(nar?300:342):(st===1?(nar?330:250):(st===2?(nar?318:356):(nar?382:420)));
  let s=`<div class="ar-pscroll">`+svgOpen(W,H,L(P_MLP.steps[st]),nar);
  const lab=(y,t1,t2)=>nar?sT(G.lx,y+4,t1,{c:'ar-sv small b',st:'font-size:10px'}):sT(G.lx,y+(t2?-2:4),t1,{c:'ar-sv small b'})+(t2?sT(G.lx,y+14,t2,{c:'ar-sv small muted'}):'');
  const fc=(x0,y0,x1,y1,c)=>`<line x1="${x0.toFixed(1)}" y1="${y0}" x2="${x1.toFixed(1)}" y2="${y1}" class="ar-fc${c?' '+c:''}"/>`;
  const neu=(x,y,r,cls,op,id)=>`<circle${id?` id="${id}"`:''} cx="${x.toFixed(1)}" cy="${y}" r="${r}" class="ar-nn ${cls}" fill-opacity="${op.toFixed(2)}"/>`;
  const inRow=y=>{let g=''; for(let i=0;i<I;i++) g+=neu(G.xi(i),y,G.ri,'in',0.25+0.75*Math.min(1,Math.abs(VEC[4][i*2])),''); return g;};
  if(st===0){let L1='',L2='',N='';
    for(let i=0;i<I;i++) for(let k=0;k<n;k++){L1+=fc(G.xi(i),G.yI+G.ri,G.xh(k),yH-G.r); L2+=fc(G.xh(k),yH+G.r,G.xi(i),yO0-G.ri);}
    for(let k=0;k<n;k++) N+=neu(G.xh(k),yH,G.r,'hid',0.25+0.75*Math.abs(Math.tanh(V.g[k])),'');
    let O=''; for(let i=0;i<I;i++) O+=neu(G.xi(i),yO0,G.ri,'out',0.3+0.6*Math.abs(Math.sin(VEC[4][i]*3+i)),'');
    s+=`<g class="ar-new">${L1}</g><g class="ar-new" style="animation-delay:250ms">${L2}</g>`+inRow(G.yI)+`<g class="ar-new" style="animation-delay:150ms">${N}</g><g class="ar-new" style="animation-delay:400ms">${O}</g>`;
    s+=lab(G.yI,nar?'d':tx('input: d','输入：d'))+lab(yH,nar?'3.5×':tx('hidden: 3.5 × d','中间层：3.5 × d'),nar?'':tx('+ a nonlinearity','加一个非线性'))+lab(yO0,nar?'d':tx('output: d','输出：d'));
    if(!nar) s+=sT(485,yO0+36,tx('Llama-3.1-8B: 4,096 → 14,336 → 4,096','Llama-3.1-8B：4,096 → 14,336 → 4,096'),{a:'middle',c:'ar-sv small muted'});
  }
  else{let LU='',LG='',NU='',NG='';
    for(let i=0;i<I;i++) for(let k=0;k<n;k++){LU+=fc(G.xi(i),G.yI+G.ri,G.xh(k),G.yU-G.r); LG+=fc(G.xi(i),G.yI+G.ri,G.xh(k),G.yG-G.r,'g');}
    for(let k=0;k<n;k++){NU+=neu(G.xh(k),G.yU,G.r,'up',0.2+0.8*Math.min(1,Math.abs(V.u[k])),''); NG+=neu(G.xh(k),G.yG,G.r,'gate'+(k===GK?' probe':''),0.1+0.9*V.gb[k],k===GK?'ar-gk':'');}
    s+=`<g class="${st===1?'ar-new':''}">${LG}</g><g>${LU}</g>`+inRow(G.yI)+`<g>${NU}</g>`+tg('gate',`<g>${NG}</g>`,'gate');
    s+=lab(G.yI,nar?'d':tx('input: d','输入：d'))+lab(G.yU,'up',nar?'':tx('candidate features','候选特征'))+lab(G.yG,'gate',nar?'':tx('SiLU: how much passes','SiLU：放行多少'));
    if(st>=2){let LP='',NP='',LD='',NO='';
      for(let k=0;k<n;k++){const x=G.xh(k), r=G.r;
        LP+=`<path d="M${(x+r*0.7).toFixed(1)} ${G.yU+r*0.7} C ${(x+r+9).toFixed(1)} ${G.yU+24}, ${(x+r+9).toFixed(1)} ${G.yP-26}, ${(x+r*0.7).toFixed(1)} ${G.yP-r*0.8}" class="ar-fc up"/>`+fc(x,G.yG+r,x,G.yP-r,'gp');
        NP+=neu(x,G.yP,G.r,'prod',0.12+0.88*clamp(Math.abs(V.pv[k])/1.4,0,1),k===GK?'ar-pk':'');
        for(let i=0;i<I;i++) LD+=fc(x,G.yP+r,G.xi(i),G.yO-G.ri);}
      for(let i=0;i<I;i++) NO+=neu(G.xi(i),G.yO,G.ri,'out',0.3+0.6*Math.abs(Math.sin(VEC[4][i]*3+i)),'');
      s+=`<g class="${st===2?'ar-new':''}">${LP}${LD}</g><g class="${st===2?'ar-new':''}" style="animation-delay:200ms">${NP}${NO}</g>`;
      s+=lab(G.yP,nar?'×':tx('gate × up','gate × up'))+tg('down',lab(G.yO,nar?'d':tx('down → d','down → d')),'down');}
    if(st>=3){const cx=G.xi(0)+(G.xi(5)-G.xi(0))/2, x0=nar?40:200, x1=nar?340:780;
      let LR=''; for(let i=0;i<I;i++) LR+=`<path d="M${G.xi(i).toFixed(1)} ${G.yO+G.ri} C ${G.xi(i).toFixed(1)} ${G.yO+26}, ${cx} ${G.yR-30}, ${cx} ${G.yR-13}" class="ar-fc res"/>`;
      s+=`<g class="ar-new">`+`<rect x="${x0}" y="${G.yR-9}" width="${x1-x0}" height="18" rx="9" class="ar-road"/>`+LR+tg('res',`<circle cx="${cx}" cy="${G.yR}" r="12" class="ar-sum"/>`+sT(cx,G.yR+5,'+',{a:'middle',c:'ar-sv b'}),'⊕')+
        (nar?'':sT(x1+14,G.yR+4,tx('residual stream','残差流'),{c:'ar-sv small'}))+`</g>`;
      if(nar) s+=sT(x0,G.yR+28,tx('residual stream','残差流'),{c:'ar-sv small'});}
    if(st===1){s+=nar?`<g id="ar-siluG" class="ar-sg t" tabindex="0" role="button" data-ctl="term" data-term="silu" aria-label="SiLU">${siluPlot(120,214,150,76,siluX)}</g>`
      :`<g id="ar-siluG" class="ar-sg t" tabindex="0" role="button" data-ctl="term" data-term="silu" aria-label="SiLU">${siluPlot(800,104,160,100,siluX)}</g>`;}}
  s+='</svg></div>';
  if(st===1) s+=`<div class="ar-pctl">`+A.ui.range({id:'ar-silu',key:'silux',label:tx('Probe x (the ringed gate neuron)','探针 x（带圈的那个 gate 神经元）'),min:-5,max:5,step:0.1,value:siluX,out:siluX.toFixed(1),inline:true})+`</div>`;
  return s;}
/* dragging the probe: the plot, its readout and the ringed gate neuron follow */
function mlpProbe(el){const V=mlpVals(), g=$('#ar-siluG',el), n=$('#ar-gk',el), nar=NAR();
  if(g) g.innerHTML=nar?siluPlot(120,214,150,76,siluX):siluPlot(800,104,160,100,siluX);
  if(n) n.setAttribute('fill-opacity',(0.1+0.9*V.gb[GK]).toFixed(2));}
/* ---------- 4. one layer: the residual stream and RMSNorm, with a live simulation ---------- */
const P_LAYER={t:{en:'One layer: residual stream and RMSNorm',zh:'一层：残差流和 RMSNorm'},steps:[{en:'Switch RMSNorm or the residual connection off and watch 32 layers.',zh:'关掉 RMSNorm 或残差连接，看 32 层会怎样。'}]};
function gauss(r){let u=0,v=0; while(u===0) u=r(); while(v===0) v=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
let SIM=null;
function simWeights(){if(SIM) return SIM; const r=mulberry(2024), d=64, h=172, mat=(a,b,s)=>Array.from({length:a},()=>Array.from({length:b},()=>gauss(r)*s));
  const layers=Array.from({length:64},()=>({g:mat(h,d,1/Math.sqrt(d)),u:mat(h,d,1/Math.sqrt(d)),o:mat(d,h,0.9/Math.sqrt(h))}));
  SIM={d,h,layers,x0:Array.from({length:d},()=>gauss(r))}; return SIM;}
const mv=(M,x)=>M.map(row=>row.reduce((a,w,i)=>a+w*x[i],0));
const rms=x=>Math.sqrt(x.reduce((a,v)=>a+v*v,0)/x.length);
function simulate(norm,res){const S0=simWeights(); let x=S0.x0.slice(); const sz=[rms(x)], sim=[1], n0=Math.hypot(...S0.x0);
  for(const Ly of S0.layers){const xin=norm?x.map(v=>v/(rms(x)+1e-6)):x, a=mv(Ly.g,xin), b=mv(Ly.u,xin), f=mv(Ly.o,a.map((v,i)=>silu(v)*b[i]));
    x=res?x.map((v,i)=>v+f[i]):f; x=x.map(v=>Math.max(-1e30,Math.min(1e30,v))); const r0=rms(x);
    sz.push(r0); const dot=x.reduce((a,v,i)=>a+v*S0.x0[i],0), nx=Math.hypot(...x); sim.push(nx>0&&isFinite(nx)&&isFinite(dot)?clamp(dot/(nx*n0),0,1):0);}
  return {sz,sim};}
const SIMC={};
const simC=(n,r)=>SIMC[n+'|'+r]||(SIMC[n+'|'+r]=simulate(n,r));
function popLayer(){
  const Wd=440, H=300, runs=[{n:true,r:true,k:'both'},{n:false,r:true,k:'nonorm'},{n:true,r:false,k:'nores'}], cur=simC(!noNorm,!noRes);
  let s=`<div class="ar-pctl"><button type="button" class="btn ghost sm" data-act="nonorm" aria-pressed="${noNorm}">${noNorm?tx('RMSNorm: off','RMSNorm：关'):tx('RMSNorm: on','RMSNorm：开')}</button>`+
    `<button type="button" class="btn ghost sm" data-act="nores" aria-pressed="${noRes}">${noRes?tx('Residual: off','残差连接：关'):tx('Residual: on','残差连接：开')}</button><span class="ar-illus">${tx('random illustrative weights, computed in the page','随机的示意参数，在页面里现算')}</span></div>`;
  s+=`<div class="ar-pscroll ar-two">`+svgOpen(Wd,H,L(P_LAYER.t)).replace(`viewBox="0 0 ${Wd} ${H}"`,`viewBox="0 60 ${Wd} 270"`);
  /* the highway: the residual stream, with two side branches */
  const y=190, xa=[80,250];
  s+=`<rect x="10" y="${y-9}" width="${Wd-20}" height="18" rx="9" class="ar-road${noRes?' cut':''}"/>`;
  if(noRes) xa.forEach(x=>{s+=`<rect x="${x+62}" y="${y-12}" width="44" height="24" class="ar-gap"/>`;});
  xa.forEach((x,bi)=>{const nm=bi?'MLP':tx('Attention','Attention');
    s+=`<path d="M${x} ${y-9} C ${x} ${y-60}, ${x+20} ${y-86}, ${x+40} ${y-96}" class="ar-br"/><path d="M${x+128} ${y-96} C ${x+148} ${y-86}, ${x+168} ${y-60}, ${x+168} ${y-9}" class="ar-br"/>`;
    s+=tg('norm',`<ellipse cx="${x+10}" cy="${y-58}" rx="12" ry="7" class="ar-ringb${noNorm?' off':''}"/>`+(noNorm?`<path d="M${x-2} ${y-50} L${x+22} ${y-66}" class="ar-x"/>`:''),'RMSNorm');
    s+=`<rect x="${x+40}" y="${y-116}" width="88" height="40" rx="8" class="${bi?'ar-mlpb':'ar-attb'}"/>`+sT(x+84,y-91,nm,{a:'middle',c:'ar-sv b'});
    s+=tg('res',`<circle cx="${x+168}" cy="${y}" r="13" class="ar-sum"/>`+sT(x+168,y+5,'+',{a:'middle',c:'ar-sv b'}),'⊕');});
  s+=fxo(20,y+24,420,'h = x + f(\\mathrm{RMSNorm}(x))',{pre:tx('residual stream:','残差流：'),c:'sm'});
  s+=sT(20,y+60,tx('× 32 layers','× 32 层'),{c:'ar-sv small muted'});
  s+=`<ellipse cx="32" cy="${y+92}" rx="10" ry="6" class="ar-ringb"/>`+sT(52,y+96,tx('RMSNorm: rescales what goes into each block','RMSNorm：把进入每个块的向量缩放到稳定大小'),{c:'ar-sv small'})+
    `<circle cx="32" cy="${y+120}" r="10" class="ar-sum"/>`+sT(32,y+124,'+',{a:'middle',c:'ar-sv b'})+sT(52,y+124,tx('⊕: adds the block’s change; the input is kept','⊕：加上块的改动；原来的输入保留'),{c:'ar-sv small'})+'</svg>';
  /* the live plot: size of the vector and how much of the input survives, over 32 layers */
  const pw=460, ph=300, X=i=>50+i/64*(pw-70), lg=v=>Math.log10(Math.max(1e-3,Math.min(1e12,v))), Y=v=>28+(12-lg(v))/(12+3)*110, Y2=v=>190+(1-v)*80;
  let g=svgOpen(pw,ph,tx('vector size across layers','向量大小随层数的变化'));
  g+=`<rect x="50" y="28" width="${pw-70}" height="110" class="ar-plot"/>`+sT(50,20,tx('vector size (log scale)','向量大小（对数刻度）'),{c:'ar-sv small b'});
  [0,4,8,12].forEach(e=>{g+=`<line x1="50" x2="${pw-20}" y1="${Y(10**e)}" y2="${Y(10**e)}" class="ar-ax"/>`+sT(44,Y(10**e)+4,e===0?'1':`10${['','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','¹⁰','¹¹','¹²'][e]}`,{a:'end',c:'ar-sv small mono'});});
  const line=(arr,f,cls)=>`<path d="${arr.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+f(v).toFixed(1)).join('')}" class="${cls}"/>`;
  runs.forEach(r=>{const o=simC(r.n,r.r); g+=line(o.sz,Y,'ar-ghost');});
  g+=line(cur.sz,Y,'ar-live');
  const last=cur.sz[cur.sz.length-1];
  g+=sT(pw-26,46,last>1e11?tx('explodes ↑','爆炸 ↑'):(last<0.2?tx('fades ↓','消失 ↓'):tx('stable','稳定')),{a:'end',c:'ar-sv b'});
  g+=`<rect x="50" y="190" width="${pw-70}" height="80" class="ar-plot"/>`+sT(50,182,tx('how much of the input is still there','输入的信息还剩多少'),{c:'ar-sv small b'});
  runs.forEach(r=>{const o=simC(r.n,r.r); g+=line(o.sim,Y2,'ar-ghost');});
  g+=line(cur.sim,Y2,'ar-live2')+sT(50,288,tx('layer 0','第 0 层'),{c:'ar-sv small muted'})+sT(pw-20,288,tx('32 layers (64 blocks)','32 层（64 个块）'),{a:'end',c:'ar-sv small muted'});
  s+=g+'</svg></div>';
  return s;}

/* ---------- 5. output: LM head → probabilities → token (the logits drag; temperature; greedy or sampled; the loop) ---------- */
const P_OUT={t:{en:'Output: LM head → probabilities → next token',zh:'输出：LM head → 概率 → 下一个 token'},steps:[
 {en:'The LM head scores every vocabulary token: the last vector meets one row per token, and each row gives one logit.',zh:'LM head 给词表里每个 token 打分：最后一个向量和每个 token 的那一行相乘，每行得出一个 logit。'},
 {en:'Softmax turns the logits into probabilities. Drag a logit or the temperature.',zh:'softmax 把 logit 变成概率。拖一下某个 logit 或温度试试。'},
 {en:'Pick the next token, greedily or by sampling; it is appended and the model runs again.',zh:'挑出下一个 token（贪心或采样）；它接到后面，模型再跑一遍。'}]};
/* step 1 of the output pop-up: the last vector × the LM head (one row per vocabulary token) = one logit per token */
function popHead(){
  const nar=NAR(), cz=CANDS.length, zMax=12;
  const vlab=tx('the last vector (“ is”)','最后一个向量（“ is”）'), vsub=tx('after the final RMSNorm','经过最后的 RMSNorm'), mlab='LM head';
  const mnote=tx('one row per vocabulary token: 128,256 rows','词表里每个 token 一行：128,256 行'), zlab=tx('logit: one score per token','logit：每个 token 一个得分');
  const others=[2.1,3.4,1.2,2.8,0.7,1.9,3.1,1.5];
  if(nar){/* phones: the vector stands on the left, its fan reaches the rows from the left, labels and logits sit to the right */
    const W=360,H=432, mx=92,mw=46,my=56,mh=330, R=CANDS.map((_,j)=>my+26+j*66), zx=220, zw=100, cw=16, ch=14, vy=R[0]+8;
    let s=svgOpen(W,H,L(P_OUT.steps[0]),1);
    s+=sT(16,16,vlab,{c:'ar-sv small b'})+sT(16,33,vsub,{c:'ar-sv small muted'});
    s+=tg('norm',`<g>`+VEC[4].map((v,i)=>`<rect x="16" y="${vy+i*ch}" width="${cw}" height="${ch-1}" rx="1.5" fill="${tint(4)}" fill-opacity="${(0.2+0.8*Math.min(1,Math.abs(v))).toFixed(2)}"/>`).join('')+`</g>`,vsub);
    const fy=vy+6*ch;
    R.forEach((y,j)=>{s+=`<g${NEW(true,120+j*110)}><path d="M${16+cw+2} ${fy} C ${16+cw+26} ${fy}, ${mx-26} ${y}, ${mx-2} ${y}" class="ar-sl" fill="none"/></g>`;});
    s+=tg('head',`<rect x="${mx}" y="${my}" width="${mw}" height="${mh}" rx="6" class="ar-wt"/>`+Array.from({length:52},(_,i)=>`<line x1="${mx+4}" x2="${mx+mw-4}" y1="${my+5+i*6.3}" y2="${my+5+i*6.3}" class="ar-wl"/>`).join('')+
      R.map(y=>`<rect x="${mx+2}" y="${y-3}" width="${mw-4}" height="6" class="ar-wlit"/>`).join('')+sT(mx+mw/2,my+mh+16,mlab,{a:'middle',c:'ar-sv small w'}),mlab);
    s+=sT(16,my+mh+36,mnote,{c:'ar-sv small muted'})+sT(mx+mw+8,my-8,tx('logit per token','每个 token 的 logit'),{c:'ar-sv small b'});
    R.forEach((y,j)=>{s+=`<g${NEW(true,300+j*110)}>`+sT(mx+mw+8,y+4,tokTxt(CANDS[j].t),{c:'ar-sv small mono'})+`<rect x="${zx}" y="${y-7}" width="${(Z[j]/zMax*zw).toFixed(1)}" height="14" rx="3" class="ar-zbar${j===0?' top':''}"/>`+sT(zx+Z[j]/zMax*zw+5,y+4,A.fmt.num(Z[j],1),{c:'ar-sv small mono'})+`</g>`;
      if(j<cz-1) s+=sT(mx+mw+18,y+37,'⋮',{c:'ar-sv muted'});});
    return s+'</svg>';}
  const W=960,H=340, mx=380,mw=84,my=22,mh=290, R=CANDS.map((_,j)=>my+26+j*58), zx=600, zw=250, vx=40, vy=150;
  let s=svgOpen(W,H,L(P_OUT.steps[0]));
  s+=sT(vx,vy-16,vlab,{c:'ar-sv small b'})+tg('norm',sStrip(vx,vy,16,16,VEC[4],tint(4)),vsub)+sT(vx,vy+36,vsub,{c:'ar-sv small muted'});
  R.forEach((y,j)=>{s+=`<g${NEW(true,120+j*110)}><path d="M${vx+194} ${vy+8} C ${vx+260} ${vy+8}, ${mx-60} ${y}, ${mx-2} ${y}" class="ar-sl" fill="none"/></g>`;});
  s+=tg('head',`<rect x="${mx}" y="${my}" width="${mw}" height="${mh}" rx="6" class="ar-wt"/>`+Array.from({length:46},(_,i)=>`<line x1="${mx+4}" x2="${mx+mw-4}" y1="${my+5+i*6.2}" y2="${my+5+i*6.2}" class="ar-wl"/>`).join('')+
    R.map(y=>`<rect x="${mx+2}" y="${y-3}" width="${mw-4}" height="6" class="ar-wlit"/>`).join('')+sT(mx+mw/2,my+mh+18,mlab,{a:'middle',c:'ar-sv small w'}),mlab);
  s+=sT(mx+mw+16,my+mh+18,mnote,{c:'ar-sv small muted'});
  s+=sT(zx,my+2,zlab,{c:'ar-sv small b'});
  R.forEach((y,j)=>{s+=`<g${NEW(true,300+j*110)}>`+sT(mx+mw+12,y+4,tokTxt(CANDS[j].t),{c:'ar-sv small mono'})+`<path d="M${mx+mw+96} ${y} L${zx-8} ${y}" class="ar-sl dot"/>`+
      `<rect x="${zx}" y="${y-8}" width="${(Z[j]/zMax*zw).toFixed(1)}" height="16" rx="3" class="ar-zbar${j===0?' top':''}"/>`+sT(zx+Z[j]/zMax*zw+8,y+4,A.fmt.num(Z[j],1),{c:'ar-sv small mono'})+`</g>`;
    if(j<cz-1){s+=sT(mx+mw+26,y+34,'⋮',{c:'ar-sv muted'}); s+=`<rect x="${zx}" y="${y+25}" width="${(others[j]/zMax*zw).toFixed(1)}" height="6" rx="2" class="ar-zbar faint"/>`;}});
  return s+'</svg>';}
let prRest=false; const RZ=6.5;   // the primer: the other 128,251 tokens as one lumped logit, so the five shown add up to less than 1
function probsNow(){const z=Z.map(z=>z/temp); return prRest?softmax([...z,RZ/temp]).slice(0,-1):softmax(z);}
function popOut(st){
  if(!st) return `<div class="ar-out">`+popHead()+`<p class="small muted ar-onote">${illus()}</p></div>`;
  const p=probsNow(), rest=prRest?1-p.reduce((a,b)=>a+b,0):0, pc=prRest?pcts([...p,rest]):pcts(p), best=p.indexOf(Math.max(...p)), live=st===1;
  let s=`<div class="ar-out">`;
  if(st===2) s+=`<div class="ar-oprompt"><span>${esc(TOKS.map(t=>t.t).join(''))}</span><span class="ar-ofly${sampled!=null?' on':''}" data-n="${flyN}">${sampled!=null?esc(CANDS[sampled].t.trim()):''}</span></div>`;
  s+=`<div class="ar-ohead">`+tg('prob',`<span class="ar-lmh ar-sm">softmax</span>`,'softmax')+`<span class="small muted">${tx('five of the 128,256 probabilities','128,256 个概率里的 5 个')}</span>${illus()}</div>`;
  s+=`<div class="ar-cands${live?'':' ro'}${st===2?' cnt':''}">`+CANDS.map((cd,j)=>`<div class="ar-cand${j===best?' pick':''}${sampled===j?' smp':''}"><span class="ar-ct mono">${esc(tokTxt(cd.t))}</span>`+
    (live?`<span class="ar-cr"><input type="range" min="0" max="12" step="0.5" value="${Z[j]}" data-z="${j}" aria-label="${esc(tx(`logit of ${tokTxt(cd.t)}`,`${tokTxt(cd.t)} 的 logit`))}"><output class="mono">${A.fmt.num(Z[j],1)}</output></span>`:'')+
    `<span class="ar-cp"><i style="width:${(p[j]*100).toFixed(1)}%"></i>`+(st===2&&dN?`<em style="width:${(dCount[j]/dN*100).toFixed(1)}%"></em>`:'')+`<b class="mono">${pc[j].toFixed(1)}%</b></span>`+
    (st===2?`<span class="ar-cn mono">${dN?`${dCount[j]} / ${dN}`:''}</span>`:'')+`</div>`).join('')+
    (prRest?`<div class="ar-cand ar-crest"><span class="ar-ct">${tx(`the other ${A.fmt.int(MODELS[DEF_M].V-CANDS.length)}`,`其余 ${A.fmt.int(MODELS[DEF_M].V-CANDS.length)} 个`)}</span>`+
      `<span class="ar-cp"><i style="width:${(rest*100).toFixed(1)}%"></i><b class="mono">${pc[CANDS.length].toFixed(1)}%</b></span></div>`:'')+`</div>`;
  if(live) s+=`<div class="ar-pctl">`+A.ui.range({id:'ar-temp',key:'temp',label:tx('Temperature','温度'),min:0.2,max:2,step:0.1,value:temp,out:temp.toFixed(1),inline:true})+
    `<span class="t ar-term ar-info" tabindex="0" role="button" data-ctl="term" data-term="temp" aria-label="${esc(tx('Temperature','温度'))}">i</span></div>`;
  else{
    /* the draw, made visible: the five probabilities laid end to end from 0 to 1, and where the random number landed */
    let acc=0; const seg=p.map((v,j)=>{const x0=acc; acc+=v; return {x0,w:v};}), hit=dU!=null?sampled:-1;
    s+=`<div class="ar-cum" role="img" aria-label="${esc(tx('The probabilities laid end to end from 0 to 1; a random number picks the segment it lands in.','概率首尾相接排成 0 到 1；随机数落在哪一段，就选哪个 token。'))}"><div class="ar-cumw"><div class="ar-cumbar">`+
      seg.map((g,j)=>`<span class="ar-seg s${j}${j===hit?' hit':''}" style="width:${(g.w*100).toFixed(2)}%">${g.w>0.09?esc(tokTxt(CANDS[j].t)):''}</span>`).join('')+(prRest?`<span class="ar-seg ar-segr" style="width:${(rest*100).toFixed(2)}%">${rest>0.09?tx('rest','其余'):''}</span>`:'')+`</div>`+
      (dU!=null?`<i class="ar-mark" data-n="${flyN}" style="left:${(dU*100).toFixed(2)}%"></i>`:'')+`</div>`+
      `<div class="ar-cumax mono"><span>0</span><span>1</span>${dU!=null?`<span class="ar-ulab ${dU<0.12?'al':(dU>0.88?'ar':'')}" style="left:${(dU*100).toFixed(2)}%">u = ${dU.toFixed(2)}</span>`:''}</div></div>`;
    s+=`<div class="ar-pctl"><span class="ar-greedy">${tx('Greedy pick','贪心选')}: <b>${esc(CANDS[best].t)}</b></span><button type="button" class="btn sm" data-act="sample">${tx('Sample one','采样一个')}</button>`+
      `<button type="button" class="btn ghost sm" data-act="sample20">${tx('Sample 20','采样 20 个')}</button>`+(dN?`<button type="button" class="btn ghost sm" data-act="sreset">${tx('Clear','清空')}</button>`:'')+`</div>`;
    if(draws.length) s+=`<div class="ar-hist"><span class="small muted">${tx('last draws','最近几次')}:</span>`+draws.map((d,i)=>`<span class="ar-hp s${d.j}${i===0?' new':''}">${esc(CANDS[d.j].t.trim())}</span>`).join('')+`</div>`;
    s+=`<p class="small muted">${tx(`The picked token is appended and the whole model runs again. The lab ${LK('kv-basics')} shows how it avoids redoing the first tokens.`,`选出的 token 接到后面，整个模型再跑一遍。${LK('kv-basics')}讲它怎样不必重算前面的 token。`)}</p>`;}
  return s+`</div>`;}
const POP={emb:{...P_EMB,body:popEmb},attn:{...P_ATTN,body:popAttn},mlp:{...P_MLP,body:popMlp},layer:{...P_LAYER,body:popLayer},out:{...P_OUT,body:popOut}};
function modalHTML(){
  return `<div class="ar-modal" id="ar-modal" hidden><div class="ar-backdrop" data-close="1"></div>`+
    `<div class="ar-dialog" role="dialog" aria-modal="true" aria-labelledby="ar-dlg-title" tabindex="-1"><div class="ar-dlg-head"><h2 id="ar-dlg-title" tabindex="-1"></h2>`+
    `<div class="ar-stepper" id="ar-stepper"></div><button type="button" class="btn ghost sm ar-close" data-close="1"></button></div><p class="ar-pcap" id="ar-pcap" aria-live="polite"></p><div class="ar-dlg-body" id="ar-dlg-body"></div></div></div>`;}
function fillDlg(keep){
  if(!app||!dlg) return; const P=POP[dlg], n=P.steps.length, body=$('#ar-dlg-body',app), a=document.activeElement, sel=keep&&a&&body.contains(a)&&a.dataset?(a.dataset.v!=null?`[data-v="${a.dataset.v}"]`:(a.dataset.act?`[data-act="${a.dataset.act}"]`:(a.dataset.z!=null?`[data-z="${a.dataset.z}"]`:(a.id?'#'+a.id:null)))):null;
  $('#ar-dlg-title',app).innerHTML=`<span class="ar-deepchip">${tx('Deep dive','深入')}</span>${esc(L(P.t))}`;
  const c=$('.ar-close',app); c.textContent=tx('Close','关闭'); c.setAttribute('aria-label',tx(`Close “${L(P.t)}”`,`关闭“${L(P.t)}”`));
  $('#ar-stepper',app).innerHTML=n>1?`<button type="button" class="btn ghost sm" data-step="-1"${dst===0?' disabled':''} aria-label="${esc(tx('Previous step','上一步'))}">◀</button><span class="mono ar-sn">${dst+1} / ${n}</span>`+
    `<button type="button" class="btn ghost sm" data-step="1"${dst===n-1?' disabled':''} aria-label="${esc(tx('Next step','下一步'))}">▶</button><button type="button" class="btn ghost sm" data-act="play" aria-pressed="${!!dplay}"${!dplay&&dst===n-1?` data-tipx="${esc(tx('Play the steps again from step 1.','从第 1 步起把每一步再播一遍。'))}"`:''}>${dplay?tx('❚❚ Pause','❚❚ 暂停'):(dst===n-1?tx('⟲ Replay','⟲ 重放'):tx('⏵ Play','⏵ 播放'))}</button>`:'';
  {const sp=P.steps[Math.min(dst,n-1)], pc=$('#ar-pcap',app); pc.textContent=L(sp); if(sp.fx) pc.insertAdjacentHTML('beforeend',' '+A.ui.math(sp.fx));}   // H1 (2026-10-09): a formula in a caption is typeset
  const sig=dlgSig(), still=sig===dlgShown||(!dplay&&dlgSeen.has(sig)); body.classList.toggle('ar-still',still); body.classList.toggle('ar-still-fly',flyN===flyShown); dlgShown=sig; flyShown=flyN;
  body.innerHTML=P.body(dst); A.ui.initRanges(body);
  clearTimeout(dlgT); if(still||A.reduceMotion()) dlgSeen.add(sig); else dlgT=setTimeout(()=>{if(dlg&&dlgShown===sig) dlgSeen.add(sig);},animEnd(body,'.ar-new,.ar-grow,.ar-slide,.ar-fadein,.ar-cell,.ar-flow')+50);   // a step left part-way is not watched
  if(sel){const e=$(sel,body)||$(sel,app); if(e) e.focus({preventScroll:true});}
}
function setStep(i,keepPlay){const n=POP[dlg].steps.length; dst=clamp(i,0,n-1); if(!keepPlay) stopPlay(); fillDlg(false); A.tip.hide();}
function stopPlay(){if(dplay){clearInterval(dplay); dplay=0;}}
function togglePlay(){if(dplay){stopPlay(); fillDlg(true); return;} const n=POP[dlg].steps.length; if(dst>=n-1) dst=-1;
  dplay=setInterval(()=>{if(!dlg){stopPlay(); return;} if(dst>=POP[dlg].steps.length-1){stopPlay(); fillDlg(true); return;} setStep(dst+1,true);},A.reduceMotion()?3200:2600); setStep(dst+1,true);}
function openDlg(k,src,step){
  if(!app||!POP[k]) return;
  dlg=k; dst=step||0; stopPlay(); seen[k]=true; sampled=null;
  openerSel=src&&src.dataset&&src.dataset.k?`#ar-ov [data-k="${src.dataset.k}"]`:(src&&src.dataset&&src.dataset.open?`#ar-rail [data-open="${k}"]`:null);
  A.tip.hide(); const md=$('#ar-modal',app); md.hidden=false; md.dataset.k=k;
  fillDlg(false); $('#ar-dlg-body',app).scrollTop=0; $('#ar-dlg-title',app).focus({preventScroll:true}); A.tip.hide();
  if(mode==='guided') updateRail();
}
function closeDlg(){
  if(!app||!dlg) return; stopPlay(); dlgShown='';
  $('#ar-modal',app).hidden=true; dlg=null; A.tip.hide();
  const back=openerSel&&$(openerSel,app); openerSel=null; if(back) back.focus({preventScroll:true}); A.tip.hide();
  if(mode==='guided') updateRail();
}
function dlgKey(e){
  if(!dlg) return;
  if(e.key==='Escape'){e.preventDefault(); closeDlg(); return;}
  if(e.key!=='Tab') return;
  const box=$('.ar-dialog',app), f=$$('button:not([disabled]),input:not([disabled]),a[href],[tabindex="0"]',box).filter(x=>x.getClientRects().length);
  if(!f.length) return; const first=f[0], last=f[f.length-1], a=document.activeElement;
  if(e.shiftKey&&(a===first||!box.contains(a)||a===box||a.id==='ar-dlg-title')){e.preventDefault(); last.focus();}
  else if(!e.shiftKey&&(a===last||!box.contains(a))){e.preventDefault(); first.focus();}
}

/* ======================================================================================================================
   THE CAUSAL SWITCH (guided part 5, curriculum: "a causal switch: remove the causal mask and see the information leak"). In training the
   model reads “The capital of France is Paris” at once and every position predicts the next token. The causal mask keeps each position
   from seeing the tokens after it; switched off, each position also sees the token it must predict (red: the leak) and copies it, so the
   training loss collapses; but when the model generates, the next token does not exist yet and there is nothing to copy. The stage
   swaps the overview for one SVG: horizontal lanes as in the hero (one per position, teal curves from earlier lanes), the deep dive's
   attention grid (hatched = hidden by the mask), the training loss, and the next token when generating.
   Illustrative (labelled; the Fidelity card lists them): one head's scores (mkS), how likely positions 1–4 are to give the right next
   token (MK_P), the copying (+MK_C on the logit of the token a position can see) and the weaker guess of a model trained without the
   mask (the lab's logits ÷ MK_K). Computed here: softmax with and without the mask, the loss (mean of −ln p), the percentages.
   ====================================================================================================================== */
let mkOn=true, mkTried=false;
const MK_T=TOKS.concat([NEXT]);                                  // six positions: “The capital of France is Paris”
const MK_P=[0.04,0.70,0.12,0.55];                                // illustrative: The → capital, capital → of, of → France, France → is
const MK_C=9, MK_K=4;                                            // illustrative: copying adds 9 to a logit; the leak-trained guess is ÷ 4
/* one head's illustrative scores: it takes the next token whenever it can see it, else itself and the token before */
const mkS=(i,j)=>j===i+1?4:(j===i?1.5:(j===i-1?1:(j<i?0.5:0)));
const mkW=on=>MK_T.map((_,i)=>softmax(MK_T.map((_,j)=>on&&j>i?-Infinity:mkS(i,j))));
const MKV=(()=>{const e=Math.exp(MK_C), pOn=MK_P.concat([softmax(Z0)[0]]), pOff=pOn.map(p=>p*e/(p*e+1-p)), nll=a=>a.reduce((s,p)=>s-Math.log(p),0)/a.length;
  return {pOn,pOff,lossOn:nll(pOn),lossOff:nll(pOff),gOn:softmax(Z0),gOff:softmax(Z0.map(z=>z/MK_K)),wOn:mkW(true),wOff:mkW(false),L0:Math.log(128256)};})();
const mkLoss=v=>v<0.01?'< 0.01':v.toFixed(2);
const mkPct=p=>(Math.round(p*1000)/10).toFixed(1)+'%';
const mkGen=on=>pcts(on?MKV.gOn:MKV.gOff);                       // the five next-token percentages when generating, adding up to 100.0
const isMk=()=>mode==='guided'&&STEPS[step]&&STEPS[step].id==='mask';
const qtk=t=>`“${t}”`;
/* one-sentence hovers (review L3–L5): what "training loss" is, and how the constructed numbers were made (the Fidelity card has the rest) */
const mkTipLoss=()=>tx(`Training loss is how unsure the model is of the right next token, averaged over the positions: the mean of ${A.ui.math('-\\ln p')}, in nats, where p is the probability the model gives the right next token. 0 means it was 100% sure and right every time.`,
  `训练 loss 表示模型对正确的下一个 token 有多没把握，在各个位置上取平均：它是 ${A.ui.math('-\\ln p')} 的平均值，单位是 nats，p 是模型给正确的下一个 token 的概率。0 表示它每次都 100% 确定而且都对。`);
const mkTipLossOn=()=>tx(`Illustrative: ${mkLoss(MKV.lossOn)} is the mean of ${A.ui.math('-\\ln p')} over the five positions, where p is the illustrative probability of the right next token on each training bar.`,
  `示意：${mkLoss(MKV.lossOn)} 是 5 个位置的 ${A.ui.math('-\\ln p')} 的平均值，p 是每根训练条上正确的下一个 token 的示意概率。`);
const mkTipLossOff=()=>tx(`Illustrative: below 0.01 is the mean of ${A.ui.math('-\\ln p')} over the five positions when each position copies the next token, which it can now see (+${MK_C} on that token’s logit).`,
  `示意：低于 0.01 是每个位置都照抄它现在看得到的下一个 token（这个 token 的 logit 加 ${MK_C}）时，各个位置的 ${A.ui.math('-\\ln p')} 的平均值。`);
const mkTipOff=()=>tx(`Illustrative: ${mkGen(false)[0].toFixed(1)}% is the chance that a model trained without the mask gives “\u00a0Paris” as the next token; that model learned to copy instead of guessing from the earlier tokens, so its guess is drawn as the lab’s logits divided by ${MK_K}: ${A.ui.math('\\text{logit} \\div '+MK_K)}.`,
  `示意：${mkGen(false)[0].toFixed(1)}% 是去掉 mask 训练出来的模型给 “\u00a0Paris” 作为下一个 token 的概率；这个模型学会的是照抄，没学会从前面的 token 猜，所以把它的猜测画成本 Lab 的 logit 除以 ${MK_K}：${A.ui.math('\\text{logit} \\div '+MK_K)}。`);
const mkTipOn=()=>tx(`Illustrative: ${mkGen(true)[0].toFixed(1)}% is the chance that the model trained with the mask gives “\u00a0Paris” as the next token, from the lab’s logits through softmax, as in the picture in parts 1–4.`,
  `示意：${mkGen(true)[0].toFixed(1)}% 是带 mask 训练的模型给 “\u00a0Paris” 作为下一个 token 的概率，由本 Lab 的 logit 经过 softmax 得到，和第 1–4 部分的图上一样。`);
/* the result, after the learner has tried the switch: result first, with its baseline, then the reason (R3.10, R6.1) */
function mkObs(){const pOn=DEF(mkGen(true)[0].toFixed(1)+'%',mkTipOn()), pOff=DEF(mkGen(false)[0].toFixed(1)+'%',mkTipOff()), lOn=DEF(mkLoss(MKV.lossOn),mkTipLossOn()), TL=DEF(tx('training loss','训练 loss'),mkTipLoss());
  return mkOn?`<p>`+tx(`<b>Mask on: training and generation see the same thing.</b> Each position sees only itself and the tokens before it, so the ${TL} stays at ${lOn} instead of falling to nearly 0.`,
      `<b>mask 开：训练和生成看到的一样。</b>每个位置只看得到自己和前面的 token，所以${TL} 停在 ${lOn}，不会降到接近 0。`)+`</p><p>`+
     tx(`The model has to learn to predict from the earlier tokens, so when it generates, “\u00a0Paris” gets ${pOn}, not ${pOff} as after training without the mask.`,
      `模型只能学着从前面的 token 预测，所以生成时 “\u00a0Paris” 有 ${pOn}，而不是去掉 mask 训练后的 ${pOff}。`)+`</p>`
   :`<p>`+tx(`<b>Mask off: training looks perfect, generating does not.</b> Every position can see the token it must predict, so the model learns to copy it, and the ${TL} falls to nearly 0 (${lOn} with the mask on).`,
      `<b>mask 关：训练看起来完美，生成却不行。</b>每个位置都看得到自己要预测的 token，模型就学会了照抄，${TL} 降到接近 0（开着 mask 时是 ${lOn}）。`)+`</p><p>`+
     tx(`When the model generates, the next token does not exist yet, so there is nothing to copy: “\u00a0Paris” falls from ${pOn} to ${pOff}.`,
      `生成时，下一个 token 还不存在，没有东西可抄：“\u00a0Paris” 从 ${pOn} 降到 ${pOff}。`)+`</p>`;}
const mkCap=()=>mkOn?tx('Causal mask on in training: each position sees only itself and the tokens before it.','训练时因果 mask 开：每个位置只看得到自己和前面的 token。')
  :tx('Causal mask off in training: each position also sees the token it must predict.','训练时因果 mask 关：每个位置还看得到它要预测的 token。');
const mkRead=()=>{const p=mkGen(mkOn)[0].toFixed(1)+'%';
  return mkOn?tx(`Training loss ${mkLoss(MKV.lossOn)}, and “\u00a0Paris” gets ${p} when generating.`,`训练 loss 是 ${mkLoss(MKV.lossOn)}，生成时 “\u00a0Paris” 有 ${p}。`)
    :tx(`Training loss below 0.01, but “\u00a0Paris” gets only ${p} when generating.`,`训练 loss 低于 0.01，但生成时 “\u00a0Paris” 只有 ${p}。`);};
function mkLegend(){const it=(cls,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i>${t}</span>`;
  return it('ar-sw-lane',tx('one lane per token','每个 token 一条线'),tx('Each of the six tokens of the training sentence is one position, with its own lane and colour; “\u00a0Paris” is ink.','训练句子的 6 个 token 各是一个位置，各有自己的线和颜色；“\u00a0Paris” 是黑色。'))+
    it('ar-sw-att',tx('teal: attention','青色：注意力'),tx('Teal curves and cells are attention weights from one head: thicker or darker means more weight. The scores are illustrative; softmax is computed here.','青色曲线和格子是一个 head 的注意力权重：越粗、越深，权重越大。得分是示意的，softmax 是现算的。'))+
    it('ar-sw-hatch',tx('hatched: hidden by the mask','斜纹：被 mask 挡住'),tx('A hatched cell is a later token that the causal mask hides from that position: it gets no attention weight.','斜纹格子是因果 mask 对这个位置挡住的后面的 token：它分不到注意力权重。'))+
    it('ar-sw-leak',tx('red: the leak','红色：泄露'),tx('Red marks the leak: a position sees tokens after it, including the one it must predict, so it can copy the answer. When generating, the dashed red arc points at the next token, the one the model learned to copy; it does not exist yet.','红色标出泄露：一个位置看得到它后面的 token，包括它要预测的那一个，所以可以直接抄答案。生成时，红色虚线指向下一个 token，也就是模型学会去抄的那个；它还不存在。'));}
/* the stage's drawing. Wide (≥ 640 px): the lanes on the left (training above generation), the attention grid (its rows on the lanes'
   rows) and the training loss on the right. Narrow: one column, the grid and the loss side by side between training and generation. */
let MCX=null;
const mW=(s,f,mono)=>{if(!MCX) MCX=document.createElement('canvas').getContext('2d'); MCX.font=`${f} ${mono?cols().mono:cols().font}`; return MCX.measureText(s).width;};
function mkSVG(W){
  const wide=W>=640, on=mkOn, V=MKV, w=on?V.wOn:V.wOff, n=MK_T.length, rh=wide?30:26, fs=wide?12:11, tfs=wide?11.5:10.5, pf=`600 ${fs}px`, tf=`600 ${tfs}px`;
  const tint=i=>i===n-1?'var(--ink)':`var(--ar-t${i})`, tt=t=>tokTxt(t.t), pc=p=>mkPct(p);
  const pw=MK_T.map(t=>mW(tt(t),pf)+14), PR=Math.ceil(Math.max(...pw))+2;
  const tw=MK_T.slice(1).map(t=>mW(tt(t),tf)+12), TW=Math.ceil(Math.max(...tw));
  /* the training bars keep their header and their percentages at every width (review L6): narrow stages get shorter bars and a smaller font */
  const bfs=wide?12:10, BL=Math.ceil(mW('100.0%',`500 ${bfs}px`,true))+7, BW=wide?88:30;
  const LW=wide?W-240:W, BX=LW-BW-BL, TX=BX-8-TW, ax0=PR+10, ax1=TX-16;
  const hy=wide?14:30, ry=i=>(wide?34:46)+i*rh, G0=ry(n-1)+(wide?38:0);   // narrow: the column headers get their own line under the title
  const pill=(i,y,x1)=>{const t=MK_T[i], x=x1-pw[i];
    return `<g class="ar-mk-pill"><rect x="${x.toFixed(1)}" y="${y-11}" width="${pw[i].toFixed(1)}" height="22" rx="11" fill="${tint(i)}"${i===n-1?'':' fill-opacity=".85"'}/>`+
      `<text x="${(x+7).toFixed(1)}" y="${y+4}" class="ar-sv ${i===n-1?'ar-mk-invt':'tokt'}" style="font-size:${fs}px;font-weight:600">${esc(tt(t))}</text></g>`;};
  const arc=(x0,y0,x1,y1)=>`M${x0.toFixed(1)} ${y0} C ${((x0+x1)/2).toFixed(1)} ${y0}, ${((x0+x1)/2).toFixed(1)} ${y1}, ${x1.toFixed(1)} ${y1}`;
  const head=(x,y,cls)=>`<path d="M${x.toFixed(1)} ${y} l-7 -4 l0 8 z" class="${cls}"/>`;
  const lab=(x,y,t,o)=>sT(x,y,t,Object.assign({c:'ar-sv small muted'},o||{}));
  const tipg=(tip,inner,cls)=>`<g data-tipx="${esc(tip)}" tabindex="0" class="ar-mk-tip${cls?' '+cls:''}">${inner}</g>`;
  let s=`<svg class="ar-mksvg" viewBox="0 0 ${W} H_" preserveAspectRatio="xMidYMin meet" role="img" aria-label="${esc(mkCap())}"><defs><pattern id="ar-mkhatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="ar-hbg"/><line x1="0" y1="0" x2="0" y2="6" class="ar-hl"/></pattern></defs>`;
  /* ---- training: six lanes; each ends at the token it must predict and how likely the position is to give it ---- */
  const ph=tx('probability','概率'), phW=mW(ph,'400 12px');
  s+=sT(0,14,wide?tx('Training: the whole sentence at once','训练：整句一次读进'):tx('Training: all at once','训练：整句一次读进'),{c:'ar-sv small b'})+lab(TX+TW,hy,tx('must predict','要预测'),{a:'end'})+
    (BX+phW<=W-2?lab(BX,hy,ph):lab(W-2,hy,ph,{a:'end'}));
  for(let i=0;i<n;i++){const y=ry(i); s+=`<path d="M${PR+1} ${y} L${TX-6} ${y}" class="ar-mk-lane" style="stroke:${tint(i)}"/>`+head(TX-2,y,'ar-mk-hd');}
  for(let i=1;i<n;i++) for(let j=0;j<i;j++){const v=w[i][j]; s+=`<path d="${arc(ax0,ry(j),ax1,ry(i))}" class="ar-mk-att" style="stroke-width:${Math.max(0.6,v*9).toFixed(2)};stroke-opacity:${(0.25+0.6*v).toFixed(2)}"/>`;}
  if(!on) for(let i=0;i<n-1;i++){const v=w[i][i+1];
    s+=`<g class="ar-mk-leak" data-i="${i}"><path d="${arc(ax0,ry(i+1),ax1-6,ry(i))}" class="ar-mkdraw" style="stroke-width:${Math.max(1.5,v*6).toFixed(2)}"/>${head(ax1,ry(i),'ar-mk-lhd')}</g>`;}
  for(let i=0;i<n;i++){const y=ry(i);
    s+=pill(i,y,PR);
    if(i===n-1){s+=tipg(tx('The token after “\u00a0Paris” comes later in the text; this example stops here.','“\u00a0Paris” 后面的 token 在文字更后面；这个例子到这里为止。'),lab(TX+6,y+4,'…',{c:'ar-sv b'})); continue;}
    const p=on?V.pOn[i]:V.pOff[i], nx=MK_T[i+1], a=qtk(tt(MK_T[i])), b=qtk(tt(nx)), last=i===n-2;
    /* one sentence: what the number is, that it is illustrative and how it was made (review L4) */
    const tip=on?(last?tx(`Illustrative: position ${a} gives the right next token, ${b}, a probability of ${pc(p)}, from the lab’s logits through softmax, as in the picture in parts 1–4.`,`示意：位置 ${a} 给正确的下一个 token ${b} 的概率是 ${pc(p)}，由本 Lab 的 logit 经过 softmax 得到，和第 1–4 部分的图上一样。`)
        :tx(`Illustrative, set by hand: position ${a} gives the right next token, ${b}, a probability of ${pc(p)}.`,`示意，手动设定：位置 ${a} 给正确的下一个 token ${b} 的概率是 ${pc(p)}。`))
      :tx(`Illustrative: position ${a} sees ${b}, the token it must predict, and copies it, so its probability rises from ${pc(V.pOn[i])} to ${pc(p)}.`,`示意：位置 ${a} 看得到它要预测的 ${b}，直接照抄，概率从 ${pc(V.pOn[i])} 升到 ${pc(p)}。`);
    s+=`<g class="ar-mk-pill"><rect x="${TX}" y="${y-10}" width="${tw[i].toFixed(1)}" height="20" rx="10" class="ar-mk-tgt" style="stroke:${tint(i+1)}${on?'':`;fill:${tint(i+1)};fill-opacity:.3`}"/>`+sT(TX+6,y+4,tt(nx),{c:'ar-sv',st:`font-size:${tfs}px;font-weight:600;fill:var(--ink)`})+`</g>`;
    s+=tipg(tip,`<rect x="${BX}" y="${y-4}" width="${BW}" height="8" rx="3" class="ar-trk"/><rect x="${BX}" y="${y-4}" width="${Math.max(1.5,BW*p).toFixed(1)}" height="8" rx="3" class="ar-mk-p ar-grow"/>`+
      `<g class="ar-mk-late">${sT(BX+BW+5,y+4,pc(p),{c:'ar-sv small mono',st:`font-size:${bfs}px`})}</g>`,'ar-mk-bar');}
  /* ---- the attention grid: rows = positions, columns = the tokens they see; hatched = hidden by the mask, red = seen after the position ---- */
  const cs=wide?24:16, cg=wide?6:2, gx=wide?LW+44:14, gy=wide?ry(0)-cs/2:ry(n-1)+44, gc=(i,j)=>[gx+j*(cs+cg),gy+i*(cs+cg)];
  for(let k=0;k<n;k++){const [x]=gc(0,k), [,y]=gc(k,0), t=qtk(tt(MK_T[k]));
    s+=tipg(tx(`Column ${t}: the attention weight each position gives ${t}.`,`${t} 这一列：每个位置分给 ${t} 的注意力权重。`),`<rect x="${x+cs/2-4}" y="${gy-12}" width="8" height="8" rx="2" fill="${tint(k)}"/>`,'ar-mk-hdot')+
      tipg(tx(`Row ${t}: the attention weights of position ${t}, adding up to 100%.`,`${t} 这一行：位置 ${t} 的注意力权重，加起来是 100%。`),`<rect x="${gx-12}" y="${y+cs/2-4}" width="8" height="8" rx="2" fill="${tint(k)}"/>`,'ar-mk-hdot');}
  s+=`<g class="ar-mk-grid">`;
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){const [x,y]=gc(i,j), v=w[i][j], hid=on&&j>i, fut=!on&&j>i, a=tt(MK_T[i]), b=tt(MK_T[j]);
    const tip=hid?tx(`The causal mask hides ${qtk(b)} from position ${qtk(a)}: no attention weight.`,`因果 mask 对位置 ${qtk(a)} 挡住了 ${qtk(b)}：注意力权重是 0。`)
      :tx(`Position ${qtk(a)} gives ${qtk(b)} ${pc(v)} of its attention weight${fut&&j===i+1?': the token it must predict':''}.`,`位置 ${qtk(a)} 分给 ${qtk(b)} ${pc(v)} 的注意力权重${fut&&j===i+1?'：正是它要预测的 token':''}。`);
    s+=`<g data-tipx="${esc(tip)}" class="ar-mkc${hid?' hid':''}${fut?' fut':''}"><rect x="${x}" y="${y}" width="${cs}" height="${cs}" rx="3" `+(hid?'fill="url(#ar-mkhatch)"':`class="${fut?'ar-mk-lk':'ar-aw'}" fill-opacity="${(0.1+0.9*v).toFixed(2)}"`)+`/></g>`;}
  s+=`</g>`;
  const gb=gy+n*(cs+cg)-cg;
  s+=lab(gx-12,gb+16,tx('attention weights, one head','注意力权重（一个 head）'),{c:'ar-sv small b'})+lab(gx-12,gb+31,tx('rows: the positions','行：位置'))+lab(gx-12,gb+45,tx('columns: the tokens they see','列：它们看到的 token'));
  /* ---- the training loss over training, from a uniform guess (ln 128,256 ≈ 11.8) to its end value, labelled at the curve's end (review L5) ---- */
  const lx0=wide?gx-10:gx+6*(cs+cg)+44, lx1=W-4, ly0=wide?gb+82:gy+14, ly1=ly0+(wide?62:56), tau=on?0.2:0.07, Le=on?V.lossOn:V.lossOff;
  const Lt=t=>Le+(V.L0-Le)*(Math.exp(-t/tau)-Math.exp(-1/tau))/(1-Math.exp(-1/tau)), Y=v=>ly1-(v/V.L0)*(ly1-ly0);
  let d=''; for(let k=0;k<=40;k++){const t=k/40; d+=(k?'L':'M')+(lx0+t*(lx1-lx0)).toFixed(1)+' '+Y(Lt(t)).toFixed(1);}
  const ltip=on?tx(`Illustrative: the training loss is the mean of ${A.ui.math('-\\ln p')} over the five positions, where p is the probability of the right next token.<div class="fx">${A.ui.math('\\text{training loss} = \\frac{1}{5} \\sum (-\\ln p)')}</div>= (${V.pOn.map(p=>(-Math.log(p)).toFixed(2)).join(' + ')}) ÷ 5 = ${mkLoss(V.lossOn)}`,
      `示意：训练 loss 是 5 个位置的 ${A.ui.math('-\\ln p')} 的平均值，p 是正确的下一个 token 的概率。<div class="fx">${A.ui.math('\\text{训练 loss} = \\frac{1}{5} \\sum (-\\ln p)')}</div>= (${V.pOn.map(p=>(-Math.log(p)).toFixed(2)).join(' + ')}) ÷ 5 = ${mkLoss(V.lossOn)}`):mkTipLossOff();
  const ttl=`<text x="${lx0}" y="${(wide?ly0-10:ly0-24)}" class="ar-sv small b ar-mk-dtxt">${esc(tx('training loss','训练 loss'))}</text>`;
  s+=tipg(mkTipLoss(),ttl,'ar-mk-def')+(wide?lab(lx0+mW(tx('training loss','训练 loss'),'700 12px')+5,ly0-10,tx('(illustrative)','（示意）')):lab(lx0,ly0-10,tx('(illustrative)','（示意）')))+
    `<line x1="${lx0}" y1="${ly0}" x2="${lx0}" y2="${ly1}" class="ar-ax"/><line x1="${lx0}" y1="${ly1}" x2="${lx1}" y2="${ly1}" class="ar-ax"/>`+
    tipg(tx(`A uniform guess over the whole vocabulary has a loss of ln 128,256 ≈ ${V.L0.toFixed(1)}, so training starts near there.`,`在整个词表里均匀地猜，loss 是 ln 128,256 ≈ ${V.L0.toFixed(1)}，训练从这附近开始。`),
      lab(lx0-4,ly0+4,`≈ ${V.L0.toFixed(1)}`,{a:'end',c:'ar-sv small mono muted',st:wide?'':'font-size:10px'}),'ar-mk-y')+
    `<path d="${d}" class="ar-mk-loss${on?'':' off'} ar-mkdraw"/>`+lab(lx0-4,ly1+4,'0',{a:'end',c:'ar-sv small mono muted'})+lab(lx1,ly1+15,tx('training →','训练 →'),{a:'end'})+
    tipg(ltip,`<circle cx="${lx1}" cy="${Y(Le).toFixed(1)}" r="3" class="ar-mk-end${on?'':' off'}"/>`+`<g class="ar-mk-late">${sT(lx1-6,Y(Le)-7,mkLoss(Le),{a:'end',c:`ar-sv b mono${on?'':' ar-mk-red'}`})}</g>`,'ar-mk-lv');
  /* ---- generation: the prompt only; the last position (“\u00a0is”) looks for the next token, which does not exist yet ---- */
  /* the next-token block: wide, under the training targets; narrow, right-aligned, its widths measured (production fonts are wider) */
  const cfs=wide?12:11, cf=`500 ${cfs}px`, clw=Math.ceil(Math.max(...CANDS.map(c=>mW(tt(c),cf,true)))), cpw=Math.ceil(mW('100.0%',cf,true)), cbw=wide?88:36;
  const clx=wide?TX+clw+12:W-4-cpw-5-cbw, CX=clx-clw-12, gx1=wide?ax1:CX-14;
  const g0=wide?G0:Math.max(gb+74,ly1+34), gry=i=>g0+(wide?20:34)+i*rh, ch=wide?18:16, cy=k=>gry(n-2)+(k-2)*ch;
  const gt=wide?tx('Generation: the next token does not exist yet','生成：下一个 token 还不存在'):tx('Generation: no next token yet','生成：下一个 token 还不存在');
  s+=sT(0,g0,gt,{c:'ar-sv small b'});
  /* review L3: with the switch off, this panel shows the model that was trained without the mask */
  if(!on) s+=wide?lab(mW(gt,'700 12px')+10,g0,tx('(the model trained without the mask)','（去掉 mask 训练出来的模型）'),{c:'ar-sv small ar-mk-red'}):lab(0,g0+14,tx('the model trained without the mask','去掉 mask 训练出来的模型'),{c:'ar-sv small ar-mk-red'});
  for(let i=0;i<n-1;i++){const y=gry(i), last=i===n-2; s+=`<path d="M${PR+1} ${y} L${last?CX-8:gx1} ${y}" class="ar-mk-lane${last?'':' dim'}" style="stroke:${tint(i)}"/>`;}
  for(let j=0;j<n-2;j++){const v=V.wOn[n-2][j]; s+=`<path d="${arc(ax0,gry(j),gx1,gry(n-2))}" class="ar-mk-att" style="stroke-width:${Math.max(0.6,v*9).toFixed(2)};stroke-opacity:${(0.25+0.6*v).toFixed(2)}"/>`;}
  if(!on) s+=`<g class="ar-mk-leak ar-mk-gen"><path d="${arc(ax0,gry(n-1),gx1,gry(n-2))}" class="dash" style="stroke-width:2"/></g>`;
  s+=`<path d="M${PR+1} ${gry(n-1)} L${ax0} ${gry(n-1)}" class="ar-mk-lane empty"/>`;
  for(let i=0;i<n-1;i++) s+=pill(i,gry(i),PR);
  const qw=Math.max(34,pw[n-1]*0.6);
  s+=tipg(tx('When the model generates, the token after “\u00a0is” has not been written yet: this is the token it must produce.','模型生成时，“\u00a0is” 后面的 token 还没写出来：这正是它要生成的 token。'),
    `<g class="ar-mk-pill"><rect x="${PR-qw}" y="${gry(n-1)-11}" width="${qw}" height="22" rx="11" class="ar-mk-q${on?'':' off'}"/>`+sT(PR-qw/2,gry(n-1)+4,'?',{a:'middle',c:'ar-sv b'})+`</g>`);
  /* review L7: the dashed arc points at what the model learned to copy from, and there is nothing there */
  if(!on) s+=lab(Math.max(0,PR-qw),gry(n-1)+25,tx('nothing to copy','没有东西可抄'),{c:'ar-sv small ar-mk-red'});
  /* the last position's next token: five of the probabilities, as in the picture of parts 1–4 */
  const g=mkGen(on), pr=on?V.gOn:V.gOff;
  s+=`<path d="M${CX-4} ${cy(0)-6} L${CX-4} ${cy(4)+6}" class="ar-mk-br"/>`+lab(clx,cy(0)-14,tx('next token','下一个 token'),{c:'ar-sv small b'});
  CANDS.forEach((cd,k)=>{const y=cy(k), top=k===0, t=qtk(tt(cd));
    const tip=on?tx(`Illustrative: when generating, the model trained with the mask gives ${t} ${g[k].toFixed(1)}%, from the lab’s logits through softmax, as in the picture in parts 1–4.`,`示意：生成时，带 mask 训练的模型给 ${t} 的概率是 ${g[k].toFixed(1)}%，由本 Lab 的 logit 经过 softmax 得到，和第 1–4 部分的图上一样。`)
      :tx(`Illustrative: a model trained without the mask learned to copy, not to guess from the earlier tokens, so its guess is drawn as the lab’s logits divided by ${MK_K} (${A.ui.math('\\text{logit} \\div '+MK_K)}), which gives ${t} ${g[k].toFixed(1)}%.`,`示意：去掉 mask 训练出来的模型学会的是照抄，没学会从前面的 token 猜，所以把它的猜测画成本 Lab 的 logit 除以 ${MK_K}（${A.ui.math('\\text{logit} \\div '+MK_K)}），${t} 得到 ${g[k].toFixed(1)}%。`);
    s+=tipg(tip,sT(clx-6,y+4,tt(cd),{a:'end',c:'ar-sv small mono',st:`font-size:${cfs}px`+(top?';fill:var(--ink)':'')})+`<rect x="${clx}" y="${y-5}" width="${cbw}" height="10" rx="3" class="ar-trk"/>`+
      `<rect x="${clx}" y="${y-5}" width="${Math.max(2,cbw*pr[k]).toFixed(1)}" height="10" rx="3" class="ar-mk-c${top?(on?' top':' bad'):''} ar-grow"/>`+
      `<g class="ar-mk-late">${sT(clx+cbw+5,y+4,g[k].toFixed(1)+'%',{c:`ar-sv small mono${top&&!on?' ar-mk-red':''}`,st:`font-size:${cfs}px`})}</g>`,'ar-mk-cand');});
  const H=Math.ceil(Math.max(gry(n-1)+30,cy(4)+14,wide?ly1+24:0))+16;
  s+=lab(W,H-4,tx('illustrative values','数值是示意的'),{a:'end'});
  return s.replace('H_',String(H))+'</svg>';}
/* the drawing fits the sticky stage without scrolling inside it (as ovFit does for the overview) */
function mkFit(svg){if(!svg) return; svg.style.maxHeight=''; const st=svg.closest('.stage'); if(!st) return; const cs=getComputedStyle(st); if(cs.position!=='sticky') return;
  const mh=parseFloat(cs.maxHeight), hh=svg.getBoundingClientRect().height, avail=(mh>0?mh:innerHeight-86)-(st.scrollHeight-hh)-2;
  if(avail>160&&hh>avail) svg.style.maxHeight=Math.floor(avail)+'px';}
let mkShown='', mkShownW=0;   // R3.20: the mask state whose chart has drawn itself in, and its width; the same state is drawn finished
/* R3.20: the mask states whose chart has drawn itself in to the end (or was drawn finished); coming back to one (on → off → on, part 5
   again after leaving the lab) draws it finished */
const mkSeen=new Set(); let mkT=0;
/* nothing new (same state, same width, chart on screen): no redraw, so a resize event cannot replay or cut the draw-in; force (the web
   fonts arrived): redraw at once, keeping the box's current animated or finished state */
function mkDraw(force){const box=app&&$('#ar-mk',app); if(!box||box.hidden) return; const W=Math.round(box.clientWidth); if(W<200) return;
  const k=String(mkOn), same=k===mkShown;
  if(same&&!force&&W===mkShownW&&box.firstElementChild){mkFit($('svg',box)); return;}
  if(!force) box.classList.toggle('ar-still',same||mkSeen.has(k)); mkShown=k; mkShownW=W;
  box.innerHTML=mkSVG(W); mkFit($('svg',box));
  if(!force){clearTimeout(mkT); if(box.classList.contains('ar-still')||A.reduceMotion()) mkSeen.add(k); else mkT=setTimeout(()=>{if(app&&mkShown===k) mkSeen.add(k);},animEnd(box,'.ar-mkdraw,.ar-mk-late,.ar-grow')+50);}}
function syncStage(){if(!app) return; const on=isMk(), o=$('#ar-ov',app), m=$('#ar-mk',app), st=$('#ar-stage',app); if(!o||!m) return;
  st.classList.toggle('ar-mkon',on); o.hidden=on; m.hidden=!on; if(on) mkDraw(); else m.innerHTML='';}
function setMask(v){if(v===mkOn) return; const first=!mkTried; mkOn=v; mkTried=true;
  renderRail(); render(); const f=$(`#ar-rail [data-key="mkmask"] [data-v="${mkOn?'on':'off'}"]`,app); if(f) f.focus({preventScroll:true});
  /* the result box: blink on a change; if its top is below the fold, scroll just enough to show its first lines, keeping the switch in view (R3.10) */
  const box=$('#ar-rail .ar-mkobs',app); if(box){if(!first) A.ui.obsBump(box); const r=box.getBoundingClientRect(); if(r.top>innerHeight-110) window.scrollBy({top:r.top-(innerHeight-170),behavior:A.reduceMotion()?'auto':'smooth'});}
  A.announce(mkRead());}

/* ======================================================================================================================
   THE LAB: rail (guided parts, scenarios, challenge), stage, settings, connect; kit integration
   ====================================================================================================================== */
let S={m:DEF_M}, mode='guided', step=0, pred=null, seen={}, chA=Array(9).fill(null), chI=0, chSub=false, iq=0, iqOpen=false;
let root=null, app=null, annT=null;
const cur=()=>MODELS[S.m];
const openBtn=(k,txt)=>`<div class="ar-open"><button type="button" class="btn sm" data-open="${k}" aria-haspopup="dialog">${txt}</button>${seen[k]?`<span class="ar-seen">✓ ${tx('opened','已打开')}</span>`:''}</div>`;
const PRED=[{v:'emb',t:{en:'The embedding table',zh:'embedding 表'}},{v:'attn',t:{en:'Self-attention',zh:'自注意力'}},{v:'mlp',t:{en:'The MLP',zh:'MLP'}},
  {v:'head',t:{en:'The LM head',zh:'LM head'}},{v:'many',t:{en:'More than one of them',zh:'不止一个'}}];
/* the deep dives, offered in the parts that touch them; ✓ once opened */
const DEEP={emb:{en:'Tokens and embedding',zh:'token 和 embedding'},attn:{en:'Self-attention',zh:'自注意力'},mlp:{en:'MLP',zh:'MLP'},layer:{en:'The layer, × 32',zh:'一层，× 32'},out:{en:'Output: LM head → next token',zh:'输出：LM head → 下一个 token'}};
/* an entry "attn:3" opens that deep dive at step 4 (the causal mask), under its own name */
const DEEPX={'attn:3':{en:'Self-attention: causal mask',zh:'自注意力：因果 mask'}};
const deepRow=ks=>`<div class="ar-deep"><span class="lbl">${tx('Go deeper (optional)','深入（可选）')}</span>`+ks.map(x=>{const [k,s]=String(x).split(':');
  return `<button type="button" class="btn ghost sm" data-open="${k}"${s?` data-ostep="${s}"`:''} aria-haspopup="dialog">${esc(L(DEEPX[x]||DEEP[k]))}${seen[k]?' <span class="ar-seen">✓</span>':''}</button>`;}).join('')+`</div>`;
/* the guided tour: three core ideas, each lit up on the picture (focus: the components that light up in turn) */
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Which part lets a token take information from other tokens?',zh:'哪个部件让一个 token 从别的 token 取信息？'},body:()=>tx(`<p>Follow “The capital of France is” through the picture to “ Paris”.</p><p><b>Which component lets a token take information from other tokens?</b></p>`,
     `<p>跟着 “The capital of France is” 走过这张图，一直到 “ Paris”。</p><p><b>哪个部件让一个 token 从别的 token 取信息？</b></p>`)+
   `<div class="ar-pred" role="group" aria-label="${esc(tx('Your answer','你的答案'))}">`+PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')+`</div>`+
   (pred?`<div class="obs ar-fb ${pred==='attn'?'ok':'no'}"><p>${pred==='attn'?tx('<b>Yes: self-attention.</b> Its curves are the only ones that cross from one lane to another.','<b>对：自注意力。</b>只有它的曲线从一条线连到另一条线。')
     :tx('<b>Not quite.</b> Look for the curves that cross from one lane to another: they are in self-attention.','<b>不太对。</b>找一找从一条线连到另一条线的曲线：它们在自注意力里。')}</p></div>`:'')},
 {id:'flow',title:{en:'How data flows',zh:'数据怎么流'},lead:{en:'Watch the parts light up in turn: top to bottom, one lane per token.',zh:'看图上依次亮起的部件：从上往下，每个 token 一条线。'},focus:['tok','emb','stack','normF','head','prob','next'],deep:['emb','layer'],
  body:()=>tx('<p>Text → tokens → one vector per token → 32 layers, each changing every vector a little → the last vector → probabilities for the next token.</p>','<p>文字 → token → 每个 token 一个向量 → 32 层，每层把每个向量改一点 → 最后一个向量 → 下一个 token 的概率。</p>')+deepRow(['emb','layer'])},
 {id:'mix',title:{en:'Only attention mixes tokens',zh:'只有注意力在 token 之间传信息'},lead:{en:'Only the teal curves cross from one lane to another: they are self-attention.',zh:'只有青色曲线从一条线连到另一条线：那是自注意力。'},focus:['attn'],deep:['attn','mlp'],
  body:()=>tx('<p>Each token keeps its own lane. Only the teal curves in self-attention cross from one lane to another; the MLP works on each lane alone.</p>','<p>每个 token 走自己的线。只有自注意力里的青色曲线从一条线连到另一条线；MLP 只在每条线上单独算。</p>')+deepRow(['attn','mlp'])},
 {id:'loop',title:{en:'One token at a time',zh:'一次生成一个 token'},lead:{en:'“ Paris” goes back into the input; the model runs again for the next token.',zh:'“ Paris”接回输入，模型再跑一遍，生成下一个 token。'},focus:['head','prob','next'],deep:['out'],
  body:()=>tx(`<p>Only the last token’s vector is used: it becomes a probability for every token in the vocabulary. One is picked, “ Paris”, appended to the text, and the model runs again for the next one. The earlier tokens’ keys and values are kept (${LK('kv-basics')}).</p>`,
     `<p>只用最后一个 token 的向量：它变成词表里每个 token 的概率。挑出一个（“ Paris”），接到文字后面，模型再跑一遍，生成下一个。前面 token 的 key 和 value 会留着（${LK('kv-basics')}）。</p>`)+deepRow(['out'])},
 /* the causal switch: training reads the whole sentence at once; the mask keeps each position from seeing the token it must predict.
    Next waits until the learner has switched the mask at least once (the result box then states the result with its baseline) */
 {id:'mask',title:{en:'Why the mask: training must not peek',zh:'为什么要 mask：训练时不能偷看'},lead:{en:'In training, every position predicts the next token; the causal mask hides it.',zh:'训练时，每个位置都预测下一个 token；因果 mask 把它挡住。'},deep:['attn:3'],
  body:()=>tx(`<p>In training, the model reads the whole sentence “The capital of France is Paris” in one pass, not token by token as when it generates, and every position must predict the token after it.</p><p>The ${DEF('causal mask',A.terms.mask())} keeps each position from seeing the tokens after it, including the one it must predict. Turn it off:</p>`,
     `<p>训练时，模型一遍读进整句 “The capital of France is Paris”，而不是像生成时那样一个 token 一个 token 地读；每个位置都要预测它后面的那个 token。</p><p>${DEF('因果 mask',A.terms.mask())} 让每个位置看不到它后面的 token，包括它要预测的那一个。把它关掉：</p>`)+
   A.ui.seg({id:'ar-mkseg',key:'mkmask',label:tx('Causal mask during training','训练时的因果 mask'),options:[['on',tx('On','开')],['off',tx('Off','关')]],value:mkOn?'on':'off',inline:true})+
   (mkTried?`<div class="obs ar-fb ar-mkobs ${mkOn?'on':'off'}" aria-live="polite">${mkObs()}</div>`:'')+deepRow(['attn:3'])},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},lead:{en:'Three core ideas; the deep dives are optional.',zh:'三个核心；深入讲解可选。'},deep:['emb','attn','mlp','layer','out'],body:()=>{const li=[
     tx('<b>Data flow:</b> text → tokens → vectors → 32 layers → probabilities → next token.','<b>数据流：</b>文字 → token → 向量 → 32 层 → 概率 → 下一个 token。'),
     tx('<b>Self-attention</b> is the only place tokens exchange information; every other part works on each token alone.','<b>自注意力</b>是 token 之间交换信息的唯一地方；其余部件都只对单个 token 算。'),
     /* the causal switch belongs to this idea: the mask makes training see what generation sees (R6.23: the lab's own case) */
     tx(`<b>One token per run:</b> the picked token is appended and the model runs again. The causal mask makes training see the same: without it, the training loss collapses, but “\u00a0Paris” falls from ${mkGen(true)[0].toFixed(1)}% to ${mkGen(false)[0].toFixed(1)}% when the model generates.`,
        `<b>一次一个 token：</b>挑出的 token 接回输入，模型再跑一遍。因果 mask 让训练时看到的也一样：去掉它，训练 loss 骤降，生成时 “\u00a0Paris” 却从 ${mkGen(true)[0].toFixed(1)}% 降到 ${mkGen(false)[0].toFixed(1)}%。`)];
   /* R07 (round 3): one next step, the scenarios (primary); the other labs stay as text links, after it */
   return `<ul class="ar-sum">${li.map(x=>`<li>${x}</li>`).join('')}</ul>`+
     `<p class="ar-nextstep">${tx('<b>Next step:</b> pick a scenario and see the same components in models of four sizes.','<b>下一步：</b>选一个应用场景，看看同样这些部件在四种大小的模型里是什么样子。')}</p>`+A.shell.toScenarios()+
     deepRow(['emb','attn','mlp','layer','out'])+
     tx(`<p class="small muted ar-later">Later, other labs go further: ${LK('transformer')} counts the parameters, ${LK('mha')} opens the heads, ${LK('kv-basics')} follows generation, ${LK('gqa-mla')} shrinks K and V.</p>`,
        `<p class="small muted ar-later">以后想接着学，别的 Lab 讲得更深：${LK('transformer')}数参数，${LK('mha')}拆开 head，${LK('kv-basics')}跟着生成走，${LK('gqa-mla')}把 K 和 V 变小。</p>`);}}
];
/* scenarios: the same components at four sizes; the drawing changes */
const PRESETS=[
 {k:'llama32-1b',t:{en:'On a phone',zh:'在手机上跑'},d:{en:'Llama-3.2-1B: 16 thin layers, made for small devices.',zh:'Llama-3.2-1B：16 层，很细，为小设备做的。'}},
 {k:'llama31-8b',t:{en:'One GPU',zh:'一张 GPU'},d:{en:'Llama-3.1-8B: 32 layers, the guided tour’s model.',zh:'Llama-3.1-8B：32 层，分步引导用的模型。'}},
 {k:'llama31-70b',t:{en:'One 8-GPU server',zh:'一台 8 卡服务器'},d:{en:'Llama-3.1-70B: 80 layers, twice as wide, 64 heads.',zh:'Llama-3.1-70B：80 层，宽一倍，64 个 head。'}},
 {k:'llama31-405b',t:{en:'Llama 3.1’s largest',zh:'Llama 3.1 最大的'},d:{en:'Llama-3.1-405B: 126 layers, 128 heads, the widest lanes.',zh:'Llama-3.1-405B：126 层，128 个 head，最宽的线。'}}];
/* the challenge: three core questions (the three ideas of the tour), then the deeper ones; structure, never parameter counts */
const CHQ_CORE=[
 {q:{en:'Token 5 (“ is”) needs something from token 4 (“ France”). Which part can carry it across?',zh:'第 5 个 token（“ is”）需要第 4 个 token（“ France”）的信息。哪个部件能把它带过来？'},
  o:[{en:'The MLP of token 5',zh:'第 5 个 token 的 MLP'},{en:'Self-attention',zh:'自注意力'},{en:'The residual add ⊕',zh:'残差相加 ⊕'},{en:'The LM head',zh:'LM head'}],a:1,
  why:{en:'The MLP, ⊕ and the LM head each work on one lane. Only attention’s curves go from one lane to another.',zh:'MLP、⊕ 和 LM head 都只在一条线上算。只有注意力的曲线从一条线连到另一条线。'},show:['attn',4]},
 {q:{en:'“ is” gets the same embedding row whether it is the 5th or the 50th token. Where does the model get the position?',zh:'“ is” 不管是第 5 个还是第 50 个 token，embedding 表给的都是同一行。模型从哪里得到位置？'},
  o:[{en:'The embedding table has one row per position.',zh:'embedding 表每个位置一行。'},{en:'In attention: RoPE turns each query and key by its position.',zh:'在注意力里：RoPE 按位置转动 query 和 key。'},{en:'The MLP counts the tokens.',zh:'MLP 数 token 的个数。'},{en:'Nowhere: word order is lost.',zh:'哪里都没有：词序丢了。'}],a:1,
  why:{en:'Llama adds no position to the embedding. In every attention layer RoPE turns q and k by the token’s position, so scores depend on the relative position.',zh:'Llama 不在 embedding 里加位置。每一层的注意力里，RoPE 按 token 的位置转动 q 和 k，所以得分和相对位置有关。'},show:['attn',0]},
 {q:{en:'After “ Paris” the model writes 3 more tokens. How many times does it run through its 32 layers?',zh:'“ Paris”之后，模型又写了 3 个 token。它要把 32 层走几遍？'},
  o:[{en:'Once: the 3 tokens come out together.',zh:'一遍：3 个 token 一起出来。'},{en:'3 times: one run per new token.',zh:'3 遍：每个新 token 一遍。'},{en:'96 times: 32 layers × 3.',zh:'96 遍：32 层 × 3。'},{en:'Once for every token of the whole text.',zh:'全文有几个 token 就走几遍。'}],a:1,
  why:{en:'A run gives probabilities for one next token; the pick goes back into the input for the next run. The earlier tokens’ keys and values are kept, so each run computes only the new token.',zh:'每跑一遍只给出下一个 token 的概率；挑出的 token 接回输入，再跑下一遍。前面 token 的 key 和 value 会留着，所以每一遍只算新 token。'},show:['out',2]}];
const CHQ=[
 {q:{en:'Do attention heads exchange information while they compute? Where do their results come together?',zh:'attention 的各个 head 在计算时会互相交换信息吗？它们的结果在哪里汇合？'},
  o:[{en:'Yes: every head reads the other heads’ scores.',zh:'会：每个 head 都读别的 head 的得分。'},{en:'No: each head works on its own slices; their outputs are joined and mixed by Wo.',zh:'不会：每个 head 只用自己那份；输出拼起来，由 Wo 混合。'},
     {en:'No, and they never combine: each head writes its own part of the output.',zh:'不会，而且从不合并：每个 head 各写各的一段输出。'},{en:'Yes: the MLP merges the heads while they compute.',zh:'会：MLP 在计算时把各个 head 合并。'}],a:1,
  why:{en:'Inside a head nothing looks at another head. Their outputs meet at concat(…) · Wo, just before ⊕.',zh:'head 内部谁也不看别的 head。输出在 concat(…) · Wo 处汇合，就在 ⊕ 之前。'},show:['attn',5]},
 {q:{en:'Remove every RMSNorm from a 32-layer model. What happens?',zh:'把 32 层模型里的 RMSNorm 全部拿掉，会怎样？'},
  o:[{en:'Nothing: RMSNorm only saves compute.',zh:'没影响：RMSNorm 只是省计算。'},{en:'The vectors’ sizes drift and can explode from layer to layer; training becomes unstable.',zh:'向量的大小一层层漂移，甚至爆炸；训练变得不稳定。'},
     {en:'Tokens can no longer exchange information.',zh:'token 之间不能再交换信息。'},{en:'Every vector shrinks to zero after the first layer.',zh:'每个向量在第一层之后就变成 0。'}],a:1,
  why:{en:'Each block sees whatever size arrives; with SwiGLU the change grows faster than the vector, so sizes run away.',zh:'每个块收到多大就算多大；SwiGLU 的改动比向量本身长得还快，大小就失控了。'},show:['layer','norm']},
 {q:{en:'Why is the MLP only two layers deep (widen, then narrow)?',zh:'MLP 为什么只有两层（先放大，再缩回）？'},
  o:[{en:'Deeper MLPs cannot be trained.',zh:'更深的 MLP 训练不了。'},{en:'One wide hidden layer per block is enough; depth comes from stacking many blocks, each adding a refinement.',zh:'每个块有一个宽的隐藏层就够了；深度来自叠很多块，每块加一点改进。'},
     {en:'Because attention already has two layers.',zh:'因为 attention 已经有两层了。'},{en:'So that the MLP can mix tokens.',zh:'为了让 MLP 能混合 token。'}],a:1,
  why:{en:'32 layers already give 32 MLPs in a row, each adding its change to the residual stream.',zh:'32 层就是连着 32 个 MLP，每个都把改动加回残差流。'},show:['mlp',3]},
 {q:{en:'Remove the residual connections. What goes wrong?',zh:'把残差连接拿掉，会出什么问题？'},
  o:[{en:'Nothing: ⊕ is only there for speed.',zh:'没问题：⊕ 只是为了快。'},{en:'Each block must pass everything through itself; the input’s signal and the training gradient fade through a deep stack.',zh:'每个块都得让所有信息从自己身上过；输入的信号和训练的梯度在很深的堆叠里逐渐消失。'},
     {en:'The model can only see the last token.',zh:'模型只能看到最后一个 token。'},{en:'Attention weights no longer add up to 100%.',zh:'注意力权重加起来不再是 100%。'}],a:1,
  why:{en:'With ⊕ the input keeps a straight path through all layers; without it, a few random layers scramble it.',zh:'有了 ⊕，输入在所有层里都有一条直路；没有它，几层随机变换就把它打乱了。'},show:['layer','res']},
 {q:{en:'A model uses a 4,096-token sliding window on a 1M-token context. Which is true?',zh:'一个模型在 1M 个 token 的上下文上用 4,096 个 token 的滑动窗口。哪个说法对？'},
  o:[{en:'Each token still sees all 1M tokens directly.',zh:'每个 token 仍然直接看到全部 1M 个 token。'},{en:'Its vectors must be 1M numbers wide.',zh:'它的向量必须是 1M 维。'},
     {en:'Compute per token stays constant; a token cannot see beyond the window directly, but information travels across layers.',zh:'每个 token 的计算量不变；token 不能直接看到窗口以外，但信息可以一层层传过去。'},{en:'Sparse masks make the MLP cheaper.',zh:'稀疏 mask 让 MLP 变便宜。'}],a:2,
  why:{en:'Each layer reaches one window further back, so 32 layers can relay information across 32 windows.',zh:'每一层能多往回走一个窗口，所以 32 层可以把信息接力传过 32 个窗口。'},show:['attn',6]},
 {q:{en:'A model is trained without the causal mask. Its training loss drops to nearly 0, far below the usual. What happens when it generates text?',zh:'一个模型训练时没有加因果 mask，训练 loss 降到接近 0，比平常低得多。它生成文字时会怎样？'},
  o:[{en:'It generates better: a lower training loss always means a better model.',zh:'生成得更好：训练 loss 越低，模型就一定越好。'},{en:'It generates poorly: each position learned to copy the next token, which does not exist yet when the model generates.',zh:'生成得很差：每个位置学会了抄下一个 token，而生成时下一个 token 还不存在。'},
     {en:'It generates just as well: the mask only saves compute.',zh:'生成得一样好：mask 只是省计算。'},{en:'It cannot run: without the mask, the attention weights no longer add up to 100%.',zh:'根本跑不起来：没有 mask，注意力权重加起来就不是 100%。'}],a:1,
  why:{en:'The causal mask gives later tokens zero attention weight, so training matches generation, where later tokens do not exist yet. Without it, each position could see the token it had to predict and learned to copy it; when the model generates, there is nothing to copy.',zh:'因果 mask 让后面的 token 的注意力权重为 0，这样训练和生成一致：生成时后面的 token 还不存在。去掉它，每个位置都看得到自己要预测的 token，于是学会了照抄；生成时没有东西可抄。'},show:['attn',3]}];
function railHTML(){
  if(mode==='guided'){const s=STEPS[step], wait=s.id==='mask'&&!mkTried, why=tx('Turn the mask off first to see what it prevents.','先把 mask 关掉，看看它挡住了什么。');
    const h=A.ui.steps({steps:STEPS.map((x,i)=>({title:L(x.title),lead:x.lead?esc(L(x.lead))+(wait&&i===step?`${tx(' ','')}<span class="ar-mkwait-l">${esc(why)}</span>`:''):''})),current:step,body:s.body(),back:step>0,showMe:false,next:step<STEPS.length-1,nextDisabled:wait,skip:false,m2:true});
    return wait?h.replace(/(<div class="stepnav">[\s\S]*?<\/div>)/,`$1<p class="small ar-mkwait" id="ar-mkwait">${esc(why)}</p>`):h;}
  if(mode==='sandbox') return `<div class="panel railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><p><b>${tx('Pick a model size and watch the drawing change.','选一种模型大小，看图怎么变。')}</b></p><div class="ar-presets">`+
    PRESETS.map((p,i)=>`<button type="button" class="choice" data-preset="${i}" aria-pressed="${p.k===S.m}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')+`</div><div class="ar-shape" id="ar-shape">${shapeHTML()}</div>`+A.shell.toChallenge()+`</div>`;
  return chHTML();}
/* one question at a time: core first, then the optional deeper ones; answers are checked on click */
const CHALL=()=>CHQ_CORE.map(q=>({...q,core:true})).concat(CHQ.map(q=>({...q,core:false})));
/* R07 (round 3): once the three core questions are right, the card says the lab's core is done and offers the next lab (the kit's
   A.ui.done): under the answer while a core question is shown, at the top while a deeper one is. The deeper six are marked optional,
   and every forward button names its own set */
const coreDone=()=>CHQ_CORE.every((x,i)=>chA[i]===x.a);
function chHTML(){const Q=CHALL(), q=Q[chI], nC=CHQ_CORE.length, a=chA[chI], ok=a===q.a, done=coreDone(), sum=done&&A.ui.done?A.ui.done(ID):'';
  const dots=(from,to)=>Q.slice(from,to).map((x,j)=>{const i=from+j, v=chA[i]; return `<button type="button" class="pb-dot${i===chI?' on':''}${v==null?'':(v===x.a?' ok':' no')}" data-chi="${i}" aria-label="${esc(tx(`Question ${i+1}`,`第 ${i+1} 题`))}"${i===chI?' aria-current="true"':''}>${v==null?i+1:(v===x.a?'✓':'✗')}</button>`;}).join('');
  const rc=Q.slice(0,nC).filter((x,i)=>chA[i]===x.a).length, ra=Q.slice(nC).filter((x,i)=>chA[nC+i]===x.a).length;
  const fw=chI<nC-1?tx('Next core question','下一道核心题'):(chI===nC-1?tx('Deeper questions (optional)','进阶题（可选）'):tx('Next deeper question','下一道进阶题'));
  return `<div class="panel railcard ar-ch"><div class="eyebrow">${A.t('modeChallenge')}</div>${q.core?'':sum}`+
    `<div class="ar-chdots"><span class="ar-chg"><span class="lbl">${tx('Core','核心')}</span>${dots(0,nC)}</span><span class="ar-chg"><span class="lbl">${tx('Deeper (optional)','进阶（可选）')}</span>${dots(nC,Q.length)}</span></div>`+
    `<p class="ar-chk mono">${q.core?tx(`Core ${chI+1} of ${nC}`,`核心 ${chI+1} / ${nC}`):tx(`Deeper (optional) ${chI-nC+1} of ${Q.length-nC}`,`进阶（可选）${chI-nC+1} / ${Q.length-nC}`)}</p>`+
    `<div class="ar-chq" id="ar-chq-${chI}"><p><b>${esc(L(q.q))}</b></p><div class="ar-cho">`+q.o.map((o,j)=>`<button type="button" class="choice${a!=null&&j===q.a?' right':''}${a===j&&j!==q.a?' wrong':''}" data-cq="${chI}" data-v="${j}" aria-pressed="${a===j}">${esc(L(o))}</button>`).join('')+`</div>`+
    (a!=null?`<div class="ar-chr ${ok?'ok':'no'}" aria-live="polite"><p><b>${ok?tx('✓ Right.','✓ 对。'):tx('✗ Not quite.','✗ 不太对。')}</b> ${esc(L(q.why))}</p><button type="button" class="btn ghost sm" data-show="${chI}">${tx('Show me','看图')}</button></div>`:'')+`</div>`+(q.core?sum:'')+
    `<div class="stepnav">${chI>0?`<button type="button" class="btn ghost sm" data-chi="${chI-1}">${tx('Previous','上一题')}</button>`:''}${chI<Q.length-1?`<button type="button" class="btn${done?' ghost':''} sm" data-chi="${chI+1}">${fw}</button>`:''}</div>`+
    `<p class="ar-chs mono">${tx(`Core: ${rc} of ${nC} right. Deeper: ${ra} of ${Q.length-nC} right.`,`核心 ${nC} 题对了 ${rc} 题；进阶 ${Q.length-nC} 题对了 ${ra} 题。`)}</p></div>`;}
function renderRail(){const rail=$('#ar-rail',app); if(!rail) return; const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID,mode==='guided'?'ar-nlq':''):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); ovFocus();
  if($('#ar-mkwait',rail)) $$('[data-nav="next"]',rail).forEach(b=>b.setAttribute('aria-describedby','ar-mkwait'));}
/* review L1 (R3.2, R3.7): after a jump or Next, the new card's top comes on screen; if the page's scroll then leaves the sticky stage's
   title under the top bar (a short card cannot hold it), scroll back just enough, never pushing the card's top off screen, and stop as
   soon as the learner scrolls (as labs/kv-basics.js does) */
let stageT=0;
function stageStop(){clearTimeout(stageT); stageT=0;}
function keepStage(){stageStop(); let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#ar-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#ar-rail .step.current',app)||$('#ar-rail',app), want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return; dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function updateRail(){ovFocus(); if(mode!=='guided') return; const r=$('#ar-rail .ar-deep',app), ks=STEPS[step].deep; if(!r||!ks) return;
  const a=document.activeElement, f=a&&r.contains(a)?a.dataset.open:null; r.outerHTML=deepRow(ks); if(f){const nb=$(`#ar-rail .ar-deep [data-open="${f}"]`,app); if(nb) nb.focus({preventScroll:true});}}
function ovFocus(){ov.focus=mode==='guided'&&STEPS[step]&&STEPS[step].focus?STEPS[step].focus:null; if(ov.loop&&!ov.loop.on) ov.loop.sync();}
/* ---------- stage and settings ---------- */
const capHTML=()=>{const m=cur(); return tx(`${esc(m.name)}: ${DEF(`${m.L} layers`,`num_hidden_layers in ${m.name}’s config.json: ${m.L}. One plate per layer behind the card.`)}, ${DEF(`${m.H} heads`,`num_attention_heads: ${m.H}. One hairline per head in self-attention.`)}, ${DEF(`${fi(m.d)}-wide vectors`,`hidden_size: ${fi(m.d)}. The lanes are drawn thicker for wider vectors, in proportion to ${A.ui.math('\\sqrt{d}')}. The MLP widens them to ${fi(m.f)}.`)}`,
  `${esc(m.name)}：${DEF(`${m.L} 层`,`${m.name} 的 config.json 里 num_hidden_layers = ${m.L}。卡片后面每层一块板。`)}，${DEF(`${m.H} 个 head`,`num_attention_heads = ${m.H}。自注意力里每个 head 一根细线。`)}，${DEF(`${fi(m.d)} 维的向量`,`hidden_size = ${fi(m.d)}。向量越宽，线画得越粗，粗细和 ${A.ui.math('\\sqrt{d}')} 成正比。MLP 把它放大到 ${fi(m.f)}。`)}`);};
const capText=()=>{const m=cur(); return tx(`${m.name}: ${m.L} layers, ${m.H} heads, ${fi(m.d)}-wide vectors`,`${m.name}：${m.L} 层，${m.H} 个 head，${fi(m.d)} 维的向量`);};
function legendHTML(){const it=(cls,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i>${t}</span>`;
  return it('ar-sw-lane',tx('one lane per token','每个 token 一条线'),tx('Each of the 5 tokens keeps its own lane and colour from the text to the last layer.','5 个 token 各有自己的线和颜色，从文字一直到最后一层。'))+
    it('ar-sw-w',tx('blue: matrices','蓝色：矩阵'),tx('Blue marks what is made of parameters: the embedding table, the blocks’ matrices, the LM head.','蓝色标出由参数构成的部分：embedding 表、各个块的矩阵、LM head。'))+
    it('ar-sw-att',tx('teal: attention','青色：注意力'),tx('Teal curves are attention weights: the only place where lanes exchange information.','青色曲线是注意力权重：各条线交换信息的唯一地方。'))+
    it('ar-sw-plus','⊕ '+tx('residual add','残差相加'),A.terms.residual());}
/* what is real and what is drawn, always in view (Codex review 1 #8); the details are in the Fidelity card */
const fidChip=()=>`<span class="chip ar-fchip" tabindex="0" data-tipx="${esc(tx('Real here: the order of the components (Meta’s Llama 3 code), each model’s shape from its config.json, the pairs each mask computes, and softmax, SiLU, temperature and the RMSNorm demo, computed live. Illustrative: the token vectors, scores, attention weights and logits, and the probabilities and the training loss of the causal-mask switch.','真实的：部件的顺序（Meta 的 Llama 3 代码）、各模型来自 config.json 的形状、每种 mask 要算多少对 token，以及现算的 softmax、SiLU、温度和 RMSNorm 演示。示意的：token 的向量、得分、注意力权重和 logit，以及因果 mask 开关那一步的概率和训练 loss。'))}">${tx('Illustrative structure · parts computed','结构示意 · 局部实算')}</span>`;
const modelSeg=()=>`<div class="seg-ctl ar-mseg" role="group" aria-label="${esc(tx('Model','模型'))}" data-key="model">`+MKEYS.map(k=>`<button type="button" data-v="${k}" aria-pressed="${k===S.m}" data-tipx="${esc(MODELS[k].name)}">${short(k)}</button>`).join('')+`</div>`;
function stageHTML(){
  return `<div class="panel stage ar-stage" id="ar-stage"><div class="stage-head"><div><div class="meter-title">${tx('“The capital of France is” → “ Paris”','“The capital of France is” → “ Paris”')}</div><div class="meter-sub" id="ar-cap"></div></div><div class="ar-shead">${modelSeg()}${fidChip()}</div></div>`+
   `<div class="legend-row ar-legend" id="ar-legend"></div><div class="stage-graph graph ar-graph" id="ar-graph">${ovHTML()}<div class="ar-mk" id="ar-mk" hidden></div></div>`+
   `<p class="small muted ar-hint">${tx('Hover a component for its definition and formula; click it to open.','把鼠标放在部件上看定义和公式；点击打开。')}</p></div>`;}
function controlsHTML(){
  return `<aside class="panel controls" id="ar-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<div class="ctl" id="ar-ctl-model"><label for="ar-model">${tx('Model','模型')}</label><select id="ar-model">${MKEYS.map(k=>`<option value="${k}"${k===S.m?' selected':''}>${MODELS[k].name}</option>`).join('')}</select>`+
   `<div class="ctl-note">${tx('The drawing changes: lanes, layers, heads.','图会跟着变：线的粗细、层数、head 数。')}</div></div>`+
   `<div class="ctl"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div><div class="ctl ar-shape" id="ar-shape"></div></aside>`;}
/* the selected model's shape: its config numbers, and the four models as rectangles (height = layers, width = d) */
const fmtI=v=>Math.round(v).toLocaleString('en-US');
function mParams(m){const hd=m.d/m.H; return m.V*m.d*(m.tied?1:2)+m.L*(2*m.d*m.d+2*m.d*m.KV*hd+3*m.d*m.f+2*m.d)+m.d;}
function shapeHTML(){const m=MODELS[S.m], hd=m.d/m.H;
  const rows=[[tx('Layers','层数'),fmtI(m.L)],[tx('Vector width d','向量宽度 d'),fmtI(m.d)],[tx('Heads','head 数'),fmtI(m.H)],[tx('Width per head','每个 head 的宽度'),fmtI(hd)],
    [tx('MLP width','MLP 宽度'),`${fmtI(m.f)} (${(m.f/m.d).toFixed(1)}×)`],[tx('Vocabulary','词表'),fmtI(m.V)]];   // no parameter count here (review 9): the lab Model parameters counts them
  const xs=[]; let x=6; MKEYS.forEach(k=>{const w=Math.max(6,MODELS[k].d/16384*64); xs.push([x,w]); x+=w+22;});
  const svg=`<svg class="ar-shp-svg" viewBox="0 0 ${x} 118" role="img" aria-label="${esc(tx('The four models: height = layers, width = d','四个模型：高 = 层数，宽 = d'))}">`+
    MKEYS.map((k,i)=>{const h=MODELS[k].L/126*90, [x0,w]=xs[i], on=k===S.m; return `<rect x="${x0}" y="${(96-h).toFixed(1)}" width="${w}" height="${h.toFixed(1)}" rx="2"${on?' class="on"':''}/><text x="${x0+w/2}" y="112" text-anchor="middle"${on?' class="on"':''}>${short(k)}</text>`;}).join('')+`</svg>`;
  return `<div class="eyebrow">${tx('Model shape','模型的形状')}</div><dl class="ar-shp">`+rows.map(([a,b])=>`<div><dt>${a}</dt><dd class="mono">${b}</dd></div>`).join('')+`</dl>`+svg+
    `<p class="small muted ar-shp-note">${tx('Rectangles: height = layers, width = d.','矩形：高 = 层数，宽 = d。')}</p>`;}
/* ---------- header, fidelity, connect ---------- */
const DESCS=()=>({guided:tx('Six short parts: three core ideas on the picture, then a switch that shows why training needs the causal mask; the deep dives are optional.','六个小部分：在图上看三个核心，再用一个开关看训练为什么需要因果 mask；深入讲解可选。'),
  sandbox:tx('Four Llama sizes: pick one and watch the drawing change.','四种大小的 Llama：选一个，看图怎么变。'),
  challenge:tx('Three core questions, then six optional deeper ones. No numbers to compute.','三道核心题，另有六道可选的进阶题。不用算数。')});
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const ML='https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py';
const TFL='https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py';
const FIDPOP=()=>tx(
 `<div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>The order of the components, as in Meta’s Llama 3 code: RMSNorm before each block, the residual add after it, the final RMSNorm, the LM head.</li><li>The model switch: layers, heads, hidden size and intermediate size from each config.json.</li><li>Softmax, SiLU, temperature and the RMSNorm demo compute live.</li><li>The pairs each mask computes, at 16 tokens and at 1M tokens.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul><li>The model is not run: vectors, scores, attention patterns and logits are illustrative.</li><li>Heads are drawn as 4 (Llama-3.1-8B has 32) with full-size K and V; Llama shares K and V among heads (GQA).</li><li>The residual/RMSNorm simulation uses random weights, 64 wide.</li><li>RoPE is drawn as one angle per token (θ = 30°). Real RoPE turns each pair of numbers in a head by its own angle, from fast-turning pairs to slow ones.</li><li>The position vectors in section 4 of the basics are 12 numbers from the first Transformer’s sine and cosine formula, to show a position embedding; Llama has no position embedding and uses RoPE instead.</li><li>The causal switch (part 5) is illustrative: one head’s scores, how likely each position is to give the right next token, the copying (+9 on the logit of a token the position can see) and the weaker guess of a model trained without the mask (the lab’s logits divided by 4, ${A.ui.math('\\text{logit} \\div 4')}). Softmax with and without the mask, the training loss (the mean of ${A.ui.math('-\\ln p')} over the five positions) and the percentages are computed from them. Without the mask, a real model can copy each target in the same way, so the pattern holds (a training loss far below normal, poor generation).</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>Meta, ${cite(ML,'models/llama3/model.py')}; Hugging Face, ${cite(TFL,'modeling_llama.py')}.</li><li>Vaswani et al., 2017 (sine and cosine position encodings); Radford et al., 2019 (GPT-2, a learned position embedding); Su et al., 2021 (RoPE); Zhang and Sennrich, 2019 (RMSNorm); Shazeer, 2020 (SwiGLU); Beltagy et al., 2020 (Longformer); Xiao et al., 2023 (StreamingLLM).</li><li>Visual idea: Transformer Explainer (Georgia Tech Polo Club).</li></ul></div></div>`,
 `<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>部件的顺序和 Meta 的 Llama 3 代码一样：每个块之前一个 RMSNorm，块之后加回残差，最后是 RMSNorm 和 LM head。</li><li>换模型：层数、head 数、隐藏维度、中间维度来自各自的 config.json。</li><li>softmax、SiLU、温度和 RMSNorm 演示都是现算的。</li><li>每种 mask 在 16 个 token 和 1M 个 token 时要算多少对 token。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul><li>不运行模型：向量、得分、注意力模式和 logit 都是示意的。</li><li>head 画了 4 个（Llama-3.1-8B 有 32 个），K、V 画成完整大小；Llama 实际让几个 head 共用 K 和 V（GQA）。</li><li>残差和 RMSNorm 的模拟用随机参数，宽 64。</li><li>RoPE 画成每个 token 一个角度（θ = 30°）。真实的 RoPE 把一个 head 里的每一对数各转一个角度，有转得快的，也有转得慢的。</li><li>知识梳理第 4 段的位置向量是按最早的 Transformer 的正弦、余弦公式算出的 12 个数，用来示意 position embedding；Llama 没有 position embedding，用的是 RoPE。</li><li>因果 mask 开关（第 5 部分）是示意的：一个 head 的得分、每个位置给出正确的下一个 token 的概率、照抄（给这个位置看得到的 token 的 logit 加 9），以及去掉 mask 训练出的模型较弱的猜测（本 Lab 的 logit 除以 4，即 ${A.ui.math('\\text{logit} \\div 4')}）。有和没有 mask 的 softmax、训练 loss（5 个位置的 ${A.ui.math('-\\ln p')} 的平均值）和百分比都是由它们现算的。去掉 mask，真实模型也能这样照抄每个目标，所以规律一样（训练 loss 远低于正常值，生成很差）。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>Meta ${cite(ML,'models/llama3/model.py')}；Hugging Face ${cite(TFL,'modeling_llama.py')}。</li><li>Vaswani 等，2017（正弦、余弦位置编码）；Radford 等，2019（GPT-2，训练学出来的位置表）；Su 等，2021（RoPE）；Zhang 和 Sennrich，2019（RMSNorm）；Shazeer，2020（SwiGLU）；Beltagy 等，2020（Longformer）；Xiao 等，2023（StreamingLLM）。</li><li>可视化思路参考：Transformer Explainer（佐治亚理工 Polo Club）。</li></ul></div></div>`);
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers.
   The challenge asks which part moves information between tokens, where the position comes from and how many runs new tokens take,
   so the exits ask none of these: why the last vector is enough, what a larger model changes, and two counts for a model the lab never
   shows (Qwen2.5-7B, from its config.json; its numbers are data, everything derived from them is computed here) */
const QW={name:'Qwen2.5-7B',V:152064,d:3584,L:28,H:28,KV:4,f:18944};
const nbh=s=>String(s).replace(/-/g,'\u2011');                   // model names with no-break hyphens, as in the other labs' exits
const OUTCOME=()=>tx('follow a prompt through a Transformer to its next token, and tell what a larger model changes.','跟着一段 prompt 走过 Transformer 直到下一个 token，并讲清更大的模型改变了什么。');
const EXITS=()=>{const s=MODELS[DEF_M], b=MODELS['llama31-405b'], sS=short(DEF_M), sB=short('llama31-405b'), q=QW, qn=nbh(QW.name), n=TOKS.length;
 /* the tokens as the challenge quotes them (“ is”), with a no-break space inside the quotes so that a line never ends on “ */
 const tk=t=>`“\u00a0${t.trim()}”`, PR=`“${TOKS.map(x=>x.t).join('')}”`, IS=tk(TOKS[n-1].t), CAP=tk(TOKS[1].t), FR=tk(TOKS[3].t), PA=tk(NEXT.t);
 return {
 selfDone:true, core:()=>CHQ_CORE.every((q,i)=>chA[i]!=null), explain:[
  {q:tx(`To pick the token after ${PR}, the LM head reads only the vector of the last token, ${IS}. Why is that one vector enough?`,`要选出 ${PR} 后面的 token，LM head 只读最后一个 token ${IS} 的向量。为什么这一个向量就够了？`),
   o:[tx(`Inside the layers, ${IS} takes in information from the earlier tokens, so its final vector already holds what the next token depends on.`,`${IS} 在各层里从前面的 token 取来了信息，所以它最后的向量已经包含决定下一个 token 的内容。`),
     tx(`The LM head first adds up the vectors of all ${n} tokens, so the one vector it scores stands for the whole text.`,`LM head 先把 ${n} 个 token 的向量加在一起，所以它打分的这一个向量代表了整段文字。`),
     tx('The next token depends mostly on the last word, so the model can leave out the vectors of the earlier tokens.','下一个 token 主要取决于最后一个词，所以模型可以不管前面那些 token 的向量。')],a:0,
   n:[null,tx(`The LM head does not combine the ${n} final vectors: it scores the vector of ${IS} alone. The other tokens’ information got into that vector earlier, inside the layers.`,`LM head 不会把 ${n} 个最后的向量合起来，它只给 ${IS} 的向量打分。其他 token 的信息，早在各层里就进了这个向量。`),
     tx(`${IS} alone says nothing about ${PA}: the answer comes from ${CAP} and ${FR}, earlier in the text. Their information reaches the vector of ${IS} inside the layers.`,`单看 ${IS} 猜不到 ${PA}，答案来自前面的 ${CAP} 和 ${FR}。它们的信息是在各层里进到 ${IS} 的向量里的。`)],
   why:tx(`Only the last position’s vector goes to the LM head. That is enough: inside the layers, ${IS} has already taken in what it needs from ${CAP} and ${FR}. The other vectors did their part there.`,`只有最后一个位置的向量进 LM head。这就够了，因为 ${IS} 在各层里已经从 ${CAP} 和 ${FR} 取到了需要的信息。其他向量的作用，在那里已经发挥完了。`)},
  {q:tx(`${nbh(b.name)} is Llama 3.1’s largest model, far larger than ${nbh(s.name)}. What makes it larger?`,`${nbh(b.name)} 是 Llama 3.1 最大的模型，比 ${nbh(s.name)} 大得多。它大在哪里？`),
   o:[tx(`It stacks more and wider layers of the same kind as the ${sS}, so every token goes through more of the same computation.`,`它用的是和 ${sS} 同一种层，只是更多、更宽，所以每个 token 要经过更多同样的计算。`),
     tx(`It adds kinds of components that the ${sS} does not have, so every token goes through steps that the smaller model skips.`,`它多了几种 ${sS} 没有的部件，所以每个 token 要经过小模型里没有的步骤。`),
     tx('It has a much larger vocabulary, so its embedding table and its LM head have far more tokens to choose from.','它的词表大得多，所以它的 embedding 表和 LM head 里可选的 token 多得多。')],a:0,
   n:[null,tx('Both have the same components in the same order: the embedding table, the layers, the final RMSNorm and the LM head. Only their sizes differ.','两者的部件和顺序完全一样：embedding 表、各层、最后的 RMSNorm 和 LM head。不同的只是大小。'),
     tx(`Both use the same vocabulary of ${fi(b.V)} tokens, so the embedding table and the LM head have the same number of rows in both models; only the rows of the ${sB} are wider.`,`两者用的是同一个词表，都是 ${fi(b.V)} 个 token，所以两个模型的 embedding 表和 LM head 行数一样；只是 ${sB} 的每一行更宽。`)],
   /* R6.20: the trend and one number, the layer ratio; the shapes themselves are on the stage (Scenarios, the model shape card) */
   why:tx(`Every Llama 3.1 model stacks layers of one shape. The ${sB} has about ${Math.round(b.L/s.L)} times as many layers as the ${sS}, and each layer is wider and has more heads.`,`每个 Llama 3.1 模型都是把同一种形状的层叠起来。${sB} 的层数约是 ${sS} 的 ${Math.round(b.L/s.L)} 倍，每一层也更宽、head 更多。`)}],
 transfer:[
  {q:tx(`${qn}’s embedding table turns each token into a vector of ${fi(q.d)} numbers, and the MLP in each of its ${q.L} layers widens that vector to ${fi(q.f)}. How many numbers does the last token’s vector have when it reaches the LM head?`,`${qn} 的 embedding 表把每个 token 变成一个 ${fi(q.d)} 维的向量，它 ${q.L} 层里的每个 MLP 都把这个向量放大到 ${fi(q.f)} 维。最后一个 token 的向量到达 LM head 时是多少维？`),
   o:[fi(q.d),fi(q.f),fi(q.L*q.d)],a:0,
   /* a worked case (R6.6), one step per sentence (R6.20) */
   why:tx(`${fi(q.d)}. Each block adds a change of ${fi(q.d)} numbers at ⊕. The MLP widens the vector to ${fi(q.f)} inside, but its down matrix narrows it back to ${fi(q.d)} first. ⊕ adds number by number, so the vector never gets longer. It keeps ${fi(q.d)} numbers through all ${q.L} layers, not ${q.L} × ${fi(q.d)} = ${fi(q.L*q.d)}.`,
          `${fi(q.d)} 维。每个块在 ⊕ 处加上的改动都是 ${fi(q.d)} 维。MLP 在里面把向量放大到 ${fi(q.f)} 维，但 down 矩阵会先把它缩回 ${fi(q.d)} 维。⊕ 是逐维相加，所以向量不会变长。它经过全部 ${q.L} 层后仍是 ${fi(q.d)} 维，而不是 ${q.L} × ${fi(q.d)} = ${fi(q.L*q.d)} 维。`)},
  {q:tx(`${qn} has ${q.L} layers with ${q.H} heads each. From the embedding table to the LM head, how many self-attention blocks does a token’s vector pass through?`,`${qn} 有 ${q.L} 层，每层 ${q.H} 个 head。从 embedding 表到 LM head，一个 token 的向量要经过几个自注意力块？`),
   o:['1',fi(q.L),fi(2*q.L),fi(q.L*q.H)],a:1,
   why:tx(`${q.L}, one per layer. Each layer has one self-attention block, with all ${q.H} of its heads inside it, and one MLP block. So the vector passes through ${q.L} self-attention blocks, and 2 × ${q.L} = ${2*q.L} blocks in all.`,
          `${q.L} 个，每层一个。每一层有一个自注意力块（${q.H} 个 head 都在这个块里）和一个 MLP 块。所以这个向量经过 ${q.L} 个自注意力块，算上 MLP 块一共是 2 × ${q.L} = ${2*q.L} 个块。`)}],
 practice:[
  {id:'transformer-bug-hunt',d:tx('Parts 1 and 2: find and fix four bugs in a small decoder-only Transformer, then replace its LM head with a classifier that averages the final vectors.','第 1、2 部分：找出并修好一个小型 decoder-only Transformer 里的 4 个 bug，再把 LM head 换成一个对最后的向量取平均的分类器。')},
  {id:'llm-fundamentals',d:tx('Questions 4 and 6: two derivations, one about rotary position embedding (RoPE), the other comparing pre-norm and post-norm blocks.','第 4、6 题：两道推导题，一道关于旋转位置编码（RoPE），另一道比较 pre-norm 和 post-norm 的块。')}]};};
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Architecture overview','结构总览'),title:l?L(l.q):tx('What is a Transformer made of?','Transformer 由哪些部件组成？'),minutes:l?l.min:12,fid:'Conceptual',fidPop:FIDPOP(),descs:DESCS(),mode});}
const CODE=`import numpy as np

def rms_norm(x, g, eps=1e-5):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g

def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def attention(x, p, n_heads):
    T, d = x.shape; hd = d // n_heads
    split = lambda a: a.reshape(T, n_heads, hd).transpose(1, 0, 2)
    q, k, v = (split(x @ p[w]) for w in ("wq", "wk", "wv"))
    s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)   # (heads, T, T)
    s[:, np.triu(np.ones((T, T), bool), 1)] = -np.inf   # causal
    out = softmax(s) @ v              # each head on its own
    out = out.transpose(1, 0, 2).reshape(T, d)   # heads joined
    return out @ p["wo"]              # ... and mixed by Wo

def mlp(x, p):                        # SwiGLU: widen, gate, narrow
    g = x @ p["gate"]
    return (g / (1 + np.exp(-g)) * (x @ p["up"])) @ p["down"]

def next_token(ids, P, n_heads, temp=1.0):
    h = P["embedding"][ids]           # lookup: (T, d)
    for p in P["layers"]:             # the residual stream
        h = h + attention(rms_norm(h, p["norm1"]), p, n_heads)
        h = h + mlp(rms_norm(h, p["norm2"]), p)
    logits = rms_norm(h, P["norm"])[-1] @ P["lm_head"]
    return int(np.argmax(logits / temp))   # greedy pick`;
const IQ=[
 {q:{en:'Why does a Transformer need both attention and an MLP?',zh:'Transformer 为什么既要 attention 又要 MLP？'},a:{en:'Attention moves information between tokens; the MLP transforms each token’s vector on its own. One mixes, the other computes.',zh:'attention 在 token 之间搬信息；MLP 单独变换每个 token 的向量。一个负责混合，一个负责计算。'}},
 {q:{en:'Where do the heads of multi-head attention interact?',zh:'多头注意力的各个 head 在哪里交互？'},a:{en:'Only after they finish: their outputs are concatenated and multiplied by Wo. Inside, each head is independent.',zh:'只在算完之后：输出拼起来再乘 Wo。在内部，每个 head 各自独立。'}},
 {q:{en:'What does the causal mask do, and why do decoders need it?',zh:'因果 mask 做什么？decoder 为什么需要它？'},a:{en:'It gives later tokens zero attention weight, so training matches generation, where later tokens do not exist yet.',zh:'它让后面的 token 的注意力权重为 0，这样训练和生成一致：生成时后面的 token 还不存在。'}},
 {q:{en:'Why pre-norm (RMSNorm before each block) rather than after?',zh:'为什么用 pre-norm（每个块之前做 RMSNorm），而不是之后？'},a:{en:'The residual stream stays an untouched sum of changes, so gradients flow straight through deep stacks; it trains more stably.',zh:'残差流一直是各个改动的直接相加，梯度可以直通很深的堆叠；训练更稳定。'}}];
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="ar-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="ar-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The whole pipeline in NumPy, in the order of the picture.','整条流水线的 NumPy 版本，顺序和图一样。')}</p><pre class="code" id="ar-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="ar-prod">`+
   tx(`<li>Meta’s ${cite(ML,'<code>model.py</code>')}: each <code>TransformerBlock</code> computes <code>h = x + attention(attention_norm(x))</code>, then <code>h + feed_forward(ffn_norm(h))</code>.</li><li>Hugging Face ${cite(TFL,'<code>modeling_llama.py</code>')}: <code>LlamaDecoderLayer</code> = <code>input_layernorm</code>, <code>self_attn</code>, <code>post_attention_layernorm</code>, <code>mlp</code>.</li><li>Sliding-window attention: Mistral 7B v0.1 (4,096 tokens); Gemma 2 and 3 alternate local and global layers.</li>`,
      `<li>Meta 的 ${cite(ML,'<code>model.py</code>')}：每个 <code>TransformerBlock</code> 先算 <code>h = x + attention(attention_norm(x))</code>，再算 <code>h + feed_forward(ffn_norm(h))</code>。</li><li>Hugging Face 的 ${cite(TFL,'<code>modeling_llama.py</code>')}：<code>LlamaDecoderLayer</code> = <code>input_layernorm</code>、<code>self_attn</code>、<code>post_attention_layernorm</code>、<code>mlp</code>。</li><li>滑动窗口注意力：Mistral 7B v0.1（4,096 个 token）；Gemma 2、3 让局部层和全局层交替。</li>`)+`</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite('https://arxiv.org/abs/1706.03762','Attention Is All You Need')}<span>${tx('Vaswani et al., 2017: the Transformer','Vaswani 等，2017：Transformer')}</span></li>`+
   `<li>${cite('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')}<span>${tx('Cho et al., 2024: an interactive GPT-2 in the browser','Cho 等，2024：浏览器里的交互式 GPT-2')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/1910.07467','Root Mean Square Layer Normalization')}<span>${tx('Zhang and Sennrich, 2019: RMSNorm','Zhang 和 Sennrich，2019：RMSNorm')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/2002.05202','GLU Variants Improve Transformer')}<span>${tx('Shazeer, 2020: SwiGLU','Shazeer，2020：SwiGLU')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/2004.05150','Longformer')}<span>${tx('Beltagy et al., 2020: sliding-window and global attention','Beltagy 等，2020：滑动窗口加全局注意力')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/2309.17453','Efficient Streaming Language Models with Attention Sinks')}<span>${tx('Xiao et al., 2023: StreamingLLM','Xiao 等，2023：StreamingLLM')}</span></li>`+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="ar-iq-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="iq-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="iq-new">${A.t('drillNew')}</button></div><div class="drill-a" id="ar-iq-a" hidden></div></article>`+
   `</div></section>`;}
function renderIQ(){if(!app) return; const k=IQ[iq%IQ.length]; $('#ar-iq-q',app).textContent=L(k.q); const a=$('#ar-iq-a',app); a.hidden=!iqOpen; a.innerHTML=iqOpen?`<p>${esc(L(k.a))}</p>`:'';}
function viewHTML(){return `<div class="ar-app m2" id="ar-app">`+headerHTML()+`<div class="wrap bench no-ctl"><aside class="rail" id="ar-rail" aria-label="${esc(A.t('railAria'))}"></aside>`+stageHTML()+`</div>`+connectHTML()+modalHTML()+`</div>`;}
/* ---------- render and state ---------- */
function render(){
  if(!app) return;
  const mk=isMk(); $('#ar-cap',app).innerHTML=mk?esc(mkCap()):capHTML(); $('#ar-legend',app).innerHTML=mk?mkLegend():legendHTML(); const shp=$('#ar-shape',app); if(shp) shp.innerHTML=shapeHTML();
  syncStage();
  $$('.ar-mseg button',app).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===S.m)));
  if(mode==='sandbox') $$('#ar-rail [data-preset]',app).forEach(b=>b.setAttribute('aria-pressed',String(PRESETS[+b.dataset.preset].k===S.m)));
  A.hash.write(S.m); A.dock.refresh();
  if(!mk){clearTimeout(annT); annT=setTimeout(()=>{if(app) A.announce(capText());},400);}
}
function setModel(k){if(!MODELS[k]) k=DEF_M; if(k===S.m){render(); return;} S.m=k; ovTarget(k); render();}
function enterStep(i){step=clamp(i,0,STEPS.length-1); renderRail(); render();}
function setMode(m){if(m===mode) return; mode=m; A.shell.setMode(app,m,DESCS()); renderRail(); render();}
/* ---------- tooltips and clicks from the kit (.t elements) ---------- */
function termTip(k,el){
  if(k==='term'){const t=el&&el.dataset.term; return tipHTML(t);}
  if(HOT_TIP[k]) return `<div class="ar-tn">${esc(L(HOT_NAME[k]))}</div>`+tipHTML(HOT_TIP[k],tx('Click for the deep dive.','点击打开深入讲解。'));
  return '';}
function jump(k,src){if(k==='term') return; const h=HOT.find(x=>x[0]===k); if(h){A.tip.hide(); openDlg(h[1],src,h[2]||0);}}
function jumpAct(k){return k==='controls'?tx('Click to jump to Settings.','点击跳到“自由调参”。'):'';}
function act(a,btn){
  if(a==='reset'){setModel(DEF_M); return;}
  if(a==='submit'){chSub=true; renderRail(); return;}
  if(a==='play'){togglePlay(); return;}
  if(a==='rope2attn'){openDlg('attn',null,0); return;}
  if(a==='a1m'){a1M=!a1M; fillDlg(true); return;}
  if(a==='nonorm'){noNorm=!noNorm; fillDlg(true); return;}
  if(a==='nores'){noRes=!noRes; fillDlg(true); return;}
  if(a==='sample'){drawOne(); flyN++; fillDlg(true); return;}
  if(a==='sample20'){for(let n=0;n<20;n++) drawOne(); flyN++; fillDlg(true); return;}
  if(a==='sreset'){resetDraws(); fillDlg(false); return;}
  if(a==='iq-new'){iq++; iqOpen=false; renderIQ(); return;}
  if(a==='iq-reveal'){iqOpen=true; renderIQ(); return;}
  if(a==='copy'){const pre=$('#ar-code',app), fl=t=>{btn.textContent=t; clearTimeout(btn._t); btn._t=setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);};
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); fl(tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(()=>fl(tx('Copied','已复制')),sel); else sel();}catch(err){sel();}}
}
function showAnswer(i){const [k,x]=CHALL()[i].show;
  if(k==='layer'){noNorm=x==='norm'; noRes=x==='res'; openDlg('layer');}
  else if(k==='attn'&&x===6){aMask='window'; a1M=true; openDlg('attn',null,6);}
  else openDlg(k,null,x);}
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key;
    if(k==='silux'){siluX=Math.round(+t.value*10)/10; A.ui.setRange(t); const o=t.closest('.inline-ctl').querySelector('output'); if(o) o.textContent=siluX.toFixed(1); mlpProbe(el); return;}
    if(k==='temp'){temp=Math.round(+t.value*10)/10; resetDraws(); A.ui.setRange(t); updOut(el); const o=t.closest('.inline-ctl').querySelector('output'); if(o) o.textContent=temp.toFixed(1); return;}
    if(t.dataset.z!=null){Z[+t.dataset.z]=+t.value; resetDraws(); A.ui.setRange(t); updOut(el);}});
  el.addEventListener('change',e=>{if(e.target.id==='ar-model') setModel(e.target.value);});
  el.addEventListener('click',e=>{
    if(e.target.closest('#ar-modal')&&e.target.closest('[data-close]')){closeDlg(); return;}
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(d.step){setStep(dst+(+d.step)); return;}
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&seg.dataset.key==='model'&&d.v){setModel(d.v); return;}
    if(seg&&seg.dataset.key==='mkmask'&&d.v){setMask(d.v==='on'); return;}
    if(seg&&d.v!=null&&b.closest('#ar-modal')){const k=seg.dataset.key; if(k==='ahead') aHead=+d.v; else if(k==='atok') aTok=+d.v; else if(k==='amask') aMask=d.v; if(dlgSig()!==dlgShown) fillDlg(true); return;}
    if(d.open){openDlg(d.open,b,+(d.ostep||0)); return;}
    if(d.go!=null){enterStep(+d.go); A.ui.reveal($('#ar-rail .step.current',app)); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); A.ui.reveal($('#ar-rail .step.current',app)); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); A.ui.reveal($('#ar-rail .step.current',app)); keepStage(); return;}
    if(d.pred){pred=d.pred; renderRail(); const f=$(`#ar-rail [data-pred="${pred}"]`,app); if(f) f.focus({preventScroll:true}); return;}
    if(d.preset!=null){setModel(PRESETS[+d.preset].k); return;}
    if(d.cq!=null){chA[+d.cq]=+d.v; renderRail(); const f=$(`#ar-rail [data-cq="${d.cq}"][data-v="${d.v}"]`,app); if(f) f.focus({preventScroll:true}); return;}
    if(d.chi!=null){chI=+d.chi; renderRail(); const f=$('#ar-rail .ar-chq .choice',app); if(f) f.focus({preventScroll:true}); return;}
    if(d.show!=null){showAnswer(+d.show); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#ar-rail',app)); return;}
    if(d.act) act(d.act,b);});
  el.addEventListener('keydown',e=>{if(dlg&&e.target.closest&&e.target.closest('#ar-modal')) dlgKey(e);});
  el.addEventListener('wheel',stageStop,{passive:true}); el.addEventListener('touchmove',stageStop,{passive:true});
  const md=$('#ar-modal',el), body=$('#ar-dlg-body',el);
  body.addEventListener('scroll',()=>A.tip.hide(),{passive:true});
  const stop=e=>{if(!e.target.closest('.ar-dialog')) e.preventDefault();};
  md.addEventListener('wheel',stop,{passive:false}); md.addEventListener('touchmove',stop,{passive:false});
  /* hovering a score cell lights up the query row and the key column that made it */
  body.addEventListener('mouseover',e=>{const g=e.target.closest('.ar-cell'); $$('.ar-cell.hl',body).forEach(x=>x.classList.remove('hl')); if(!g) return; const i=g.dataset.i, j=g.dataset.j; $$('.ar-cell',body).forEach(x=>{if(x.dataset.i===i||x.dataset.j===j) x.classList.add('hl');});});
}
function updOut(el){const p=probsNow(), pc=pcts(p), best=p.indexOf(Math.max(...p)); el=$('#ar-dlg-body',el)||el;   // the pop-up's rows, not the primer's pictures
  $$('.ar-cand:not(.ar-crest)',el).forEach((row,j)=>{row.querySelector('.ar-cp i').style.width=(p[j]*100).toFixed(1)+'%'; row.querySelector('.ar-cp b').textContent=pc[j].toFixed(1)+'%'; row.classList.toggle('pick',j===best);
    const o=row.querySelector('output'); if(o) o.textContent=A.fmt.num(Z[j],1);});
  const g=$('.ar-greedy b',el); if(g) g.textContent=CANDS[best].t;}
/* ======================================================================================================================
   知识梳理 (the owner's decision of 2026-10-11 19:28 UTC): ten sections read top to bottom before the guided parts, in the owner's
   titles. Each section says why the step is needed before how it works, and goes one step beyond the section before; the pictures are
   this lab's own deep-dive steps and overview, shown as a display (A.registerPrimer)
   ====================================================================================================================== */
const DM=MODELS[DEF_M].d, FFN=MODELS[DEF_M].f, NL=MODELS[DEF_M].L, NH=MODELS[DEF_M].H, HD=DM/NH, nf=v=>A.fmt.int(v);
// a deep-dive step drawn with the primer's own settings (the lab's defaults, whatever the reader set in the lab), put back after
const PRDEF=()=>({aHead:1,aTok:4,aMask:'causal',noNorm:false,noRes:false,temp:1,sampled:null,dU:null,dN:0,Z:Z0.slice(),prRest:true});
function withPop(o,f){const sv={aHead,aTok,aMask,noNorm,noRes,temp,sampled,dU,dN,Z,prRest};
  ({aHead,aTok,aMask,noNorm,noRes,temp,sampled,dU,dN,Z,prRest}={...sv,...PRDEF(),...o});
  try{return f();} finally{({aHead,aTok,aMask,noNorm,noRes,temp,sampled,dU,dN,Z,prRest}=sv);}}
const PRC='ar-dlg-body pr-ar';
// o.rep: ()=>[[from, to], ...], the primer's own wording for a label inside a step (from: a string or a RegExp)
const prRep=(h,rep)=>rep?rep().reduce((s,[a,b])=>s.replace(a,b),h):h;
// the primer's copy of a step gets its own ids, so the lab's own lookups (#ar-temp-in, #ar-silu, gradients) never land in the primer
const prIds=h=>h.replace(/(\b(?:id|for|aria-labelledby|aria-describedby|aria-controls)="|url\(#|href="#)ar-/g,'$1pr-ar-');
const prFr=(body,cap,o)=>({html:()=>prIds(prRep(withPop(o||{},body),o&&o.rep)),cap,ms:(o&&o.ms)||5200});
const prCap=(P,i)=>()=>{const s=P.steps[i]; return esc(L(s))+(s.fx?' '+A.ui.math(s.fx):'');};
const prP=(...ps)=>({html:()=>ps.map(p=>`<p>${p()}</p>`).join('')});
const M_=A.ui.math, VN=nf(MODELS[DEF_M].V), KVH=8;
const REP_H=()=>[[M_('h = x + f(\\mathrm{RMSNorm}(x))'),()=>M_('h \\leftarrow h + f(\\mathrm{RMSNorm}(h))')]];
const REP_V=()=>[[new RegExp(tx('V slices','V 切片'),'g'),tx('values v','各个 v')]];
const KV=MODELS[DEF_M].KV, PM=3*DM*FFN, PA=2*DM*DM+2*DM*KV*HD, yi=v=>(v/1e8).toFixed(2), mi=v=>Math.round(v/1e6);
/* the overview, drawn by its own loop (the lab's stage keeps its own) */
function prOvMount(fig){const host=fig.querySelector('.pr-ovbox'), cv=host&&host.querySelector('canvas'); if(!cv) return;
  const draw=(t,still)=>{if(!cv.isConnected){loop.kill(); return;} const W0=host.getBoundingClientRect().width; if(W0<60) return;
    const P=mP(DEF_M), sv={hk:ov.hk,focus:ov.focus}; ov.hk=null; ov.focus=null;
    try{const k=ovFit(cv,W0,P,t), o=ovLayout(W0,P,k); if(cv._hh!==o.H){cv.style.height=o.H+'px'; cv._hh=o.H;}
      const {c}=fit(cv); c.clearRect(0,0,W0,o.H); ovPaint(c,W0,P,t,still,false,k);} finally{ov.hk=sv.hk; ov.focus=sv.focus;}};
  const loop=makeLoop(host,draw);}
/* the primer's section 4: a position embedding, drawn for the primer only (the owner, 2026-10-12). f = 0: each position looks up its
   own row of a position table; 1: the position vector is added to the token vector; 2: the same token at positions 5 and 50 gets two
   different input vectors. The position vectors follow the sine and cosine pattern of the first Transformer */
const PEV=pos=>Array.from({length:12},(_,j)=>{const a=pos/Math.pow(10000,2*Math.floor(j/2)/12); return Math.round((j%2?Math.cos(a):Math.sin(a))*100)/100;});
const PSUM=(k,pos)=>VEC[k].map((v,j)=>Math.max(-1,Math.min(1,(v+PEV(pos)[j])/1.4)));
function prPosFig(f){
  const nar=NAR(), W=nar?360:960, cw=nar?6:14, ch=nar?12:14, sw=12*cw, gray='var(--ink-3)';
  const lab=[tx('Each position looks up its own row of a position table','每个位置在位置表里取自己的一行'),tx('token vector + position vector = input vector','token 向量 + 位置向量 = 输入向量'),
    tx('The same token at positions 5 and 50 gets two different input vectors','同一个 token 在第 5 个和第 50 个位置，得到两个不同的输入向量')][f];
  const pill=(x,y,k,t)=>{const w=t.length*(nar?7:8.6)+(nar?14:20); return `<rect x="${x}" y="${y-(nar?11:13)}" width="${w}" height="${nar?22:26}" rx="${nar?11:13}" fill="${tint(k)}"/>`+sT(x+(nar?7:10),y+(nar?4:5),t,{c:'ar-sv tokt',st:nar?'font-size:11px':''});};
  const op=(x,y,c)=>sT(x,y+5,c,{a:'middle',c:'ar-sv b',st:'font-size:16px'});
  const head=(x,y,t)=>sT(x,y,t,{c:'ar-sv small muted',st:nar?'font-size:10.5px':''});
  if(f===2){const H=nar?210:210, y1=nar?96:100, y2=nar?166:168;
    let s=svgOpen(W,H,lab,nar), x0=nar?10:40, xa=nar?96:260, xb=xa+sw+(nar?22:40), xc=xb+sw+(nar?22:40);
    s+=head(xa,nar?44:50,tx('token vector','token 向量'))+head(xb,nar?44:50,tx('position vector','位置向量'))+head(xc,nar?44:50,tx('input vector','输入向量'));
    [[y1,5],[y2,50]].forEach(([y,pos],r)=>{s+=`<g${NEW(true,r*260)}>`+pill(x0,y,4,tokTxt(TOKS[4].t))+sT(x0,y+(nar?24:30),tx(`at position ${pos}`,`第 ${pos} 个位置`),{c:'ar-sv small mono',st:nar?'font-size:10.5px':''})+
      sStrip(xa,y-ch/2,cw,ch,VEC[4],tint(4))+op(xb-(nar?11:20),y,'+')+sStrip(xb,y-ch/2,cw,ch,PEV(pos),gray)+op(xc-(nar?11:20),y,'=')+sStrip(xc,y-ch/2,cw,ch,PSUM(4,pos),tint(4),{cls:'ar-slide',style:`animation-delay:${400+r*260}ms`})+`</g>`;});
    return s+'</svg>';}
  const top=nar?62:76, dy=nar?44:46, H=top+4*dy+(nar?40:44), ys=TOKS.map((_,k)=>top+k*dy);
  let s=svgOpen(W,H,lab,nar); const x0=nar?10:40;
  TOKS.forEach((tk,k)=>{s+=pill(x0,ys[k],k,tokTxt(tk.t));});
  if(f===0){const xp=nar?90:200, xt=nar?150:330, tw=nar?34:64, xv=nar?(W-10-sw):520;
    s+=head(xp,top-(nar?30:34),tx('position','位置'))+head(xt,top-(nar?30:34),tx('position table','位置表'))+head(xv,top-(nar?30:34),tx('position vector','位置向量'));
    s+=`<rect x="${xt}" y="${top-18}" width="${tw}" height="${4*dy+36}" rx="6" class="ar-wt"/>`+Array.from({length:Math.floor((4*dy+30)/6)},(_,i)=>`<line x1="${xt+3}" x2="${xt+tw-3}" y1="${top-14+i*6}" y2="${top-14+i*6}" class="ar-wl"/>`).join('');
    TOKS.forEach((_,k)=>{const y=ys[k];
      s+=`<g${NEW(true,k*140)}>`+sT(xp,y+4,String(k+1),{c:'ar-sv mono'})+`<path d="M${xp+(nar?14:20)} ${y} L${xt-4} ${y}" class="ar-sl id"/>`+`<rect x="${xt+2}" y="${y-3}" width="${tw-4}" height="6" class="ar-wlit"/>`+
        `<path d="M${xt+tw+4} ${y} L${xv-6} ${y}" class="ar-sl"/>`+sStrip(xv,y-ch/2,cw,ch,PEV(k+1),gray,{cls:'ar-slide',style:`animation-delay:${300+k*140}ms`})+`</g>`;});
    s+=sT(xt+tw/2,H-(nar?12:14),tx('row k: position k','第 k 行：第 k 个位置'),{a:'middle',c:'ar-sv small muted',st:nar?'font-size:10px':''});
    return s+'</svg>';}
  const xa=nar?88:200, xb=xa+sw+(nar?18:46), xc=xb+sw+(nar?18:46);
  s+=head(xa,top-(nar?30:34),tx('token vector','token 向量'))+head(xb,top-(nar?30:34),tx('position vector','位置向量'))+head(xc,top-(nar?30:34),tx('input vector','输入向量'));
  TOKS.forEach((_,k)=>{const y=ys[k];
    s+=sStrip(xa,y-ch/2,cw,ch,VEC[k],tint(k))+`<g${NEW(true,200+k*120)}>`+op(xb-(nar?9:23),y,'+')+sStrip(xb,y-ch/2,cw,ch,PEV(k+1),gray)+`</g>`+
      `<g${NEW(true,700+k*120)}>`+op(xc-(nar?9:23),y,'=')+sStrip(xc,y-ch/2,cw,ch,PSUM(k,k+1),tint(k))+`</g>`;});
  return s+'</svg>';}
A.registerPrimer(ID,{min:10,sections:[
 {h:()=>tx('How a model predicts the next word of an unfinished sentence','模型怎样预测一句话里没写完的下一个词'),parts:[
   prP(()=>tx('Give a language model an unfinished sentence, “The capital of France is”. It does one thing: it gives every word it knows a probability of coming next, a probability distribution over the next word. Here the most likely one is “Paris”.',
              '给语言模型一句没写完的话，“The capital of France is”。它只做一件事：给它认识的每个词一个概率，也就是下一个词的概率分布。这句话后面最可能的是“Paris”。'),
       ()=>tx('Writing a whole answer is this step again and again: pick a word from the distribution, append it, predict the next one. So to understand a large language model, it is enough to follow how a sentence becomes this distribution.',
              '写出一整段回答，就是把这一步一遍遍重复：从分布里挑出一个词，接到句子后面，再预测下一个。所以要看懂大模型，只要看清一句话怎样变成这个概率分布。'),
       ()=>tx('That takes four answers: how text becomes numbers, how each word uses the words before it, how the gathered information is processed, and how the result becomes probabilities. All the numbers here are those of Llama 3.1 8B, an open language model Meta released in 2024.',
              '这需要回答四个问题：文字怎样变成数，每个词怎样用上前面的词，取来的信息怎样加工，结果怎样变成概率。文中的数字都取自 Llama 3.1 8B（Meta 2024 年发布的开源语言模型）。')),
   {fig:[prFr(()=>popOut(2),()=>tx('The five most likely next words after the sentence. One is picked and appended, and the model predicts again.','这句话后面概率最高的 5 个词。挑出一个接到句子后面，模型再预测下一个。'),{ms:8000})],cls:PRC}]},
 {h:()=>tx('First, cut the sentence into pieces (tokenization)','先把句子切成小片段（token化）'),parts:[
   prP(()=>tx('A model only computes with numbers, and every input unit needs its own set of trained numbers, so the units must come from a table fixed in advance. The first step cuts the text into pieces from this vocabulary, called tokens.',
              '模型只会对数做运算，而每个输入单位都要对应一组训练好的数，所以这些单位只能来自一张事先定好的表。第一步就是把文字切成这张词表里的片段，这些片段叫 token。'),
       ()=>tx(`Letters would make sequences very long, and a single letter carries almost no meaning; whole words would need an endless vocabulary, and a new or misspelled word would have no entry. Real tokenization sits between the two: a common word is one token, a rare word is split into a few common pieces. The program that does this is the tokenizer; Llama 3’s uses the BPE algorithm and has a vocabulary of ${VN} tokens.`,
              `按字母切，序列太长，单个字母也几乎不带意思；按整词切，词表永远收不全，新词和拼错的词都没有位置。实际的切法在两者之间：常见的词是一个 token，少见的词拆成几个常见的片段。做这件事的叫分词器（tokenizer），Llama 3 用的是 BPE 算法，词表有 ${VN} 个 token。`),
       ()=>tx('“The capital of France is” becomes 5 tokens. The space before a word belongs to its token, written ␣.','“The capital of France is”切成 5 个 token。词前面的空格算在 token 里，写成 ␣。')),
   {fig:[prFr(()=>popEmb(0),prCap(P_EMB,0),{ms:7000})],cls:PRC}]},
 {h:()=>tx('Turn each token into a vector the model can compute with','将token转化为模型可以处理的向量输入'),parts:[
   prP(()=>tx('Each token has a number in the vocabulary, its ID: ␣France is 9822. An ID is only a position in a list. 9822 is no closer in meaning to 9823 than to 5, so matrix arithmetic on IDs would mean nothing.','每个 token 在词表里有一个编号，叫 ID：␣France 是 9822。ID 只是在表里的位置，9822 和 9823 在意思上并不比 9822 和 5 更接近，直接拿 ID 做矩阵运算没有意义。'),
       ()=>tx(`So the model keeps an embedding table with one row per token, ${VN} rows, each a vector of ${nf(DM)} numbers. The ID picks a row, and that row is the token’s input vector. The numbers in the table, like those in every matrix later on, are learned in training; together they are the model’s parameters. Tokens with similar meanings end up with similar rows.`,
              `所以模型有一张 embedding 表，每个 token 一行，共 ${VN} 行，每一行是一个 ${nf(DM)} 维的向量。ID 选出其中一行，这一行就是这个 token 的输入向量。表里的数和后面所有矩阵里的数一样，都是训练学出来的，统称参数；意思相近的 token，向量也相近。`),
       ()=>tx(`From here on the sentence is 5 vectors of ${M_('d')} = ${nf(DM)} numbers, and every later step computes on these 5 vectors.`,`从这里开始，这句话就是 5 个 ${M_('d')} = ${nf(DM)} 维的向量，后面的每一步都在这 5 个向量上计算。`)),
   {fig:[prFr(()=>popEmb(1),prCap(P_EMB,1)),prFr(()=>popEmb(2),prCap(P_EMB,2)),
         prFr(()=>popEmb(3,true),()=>tx(`The row the ID picks is the token’s input vector: 5 tokens, 5 vectors of ${nf(DM)} numbers.`,`ID 选出的那一行，就是这个 token 的输入向量：5 个 token，5 个 ${nf(DM)} 维的向量。`),{ms:6500})],cls:PRC}]},
 {h:()=>tx('The model needs the order of the tokens','模型关心token的先后顺序'),parts:[
   prP(()=>tx('“The dog bit the man” and “The man bit the dog” use the same tokens and mean opposite things. To predict the next word, the model has to know the order of the tokens: which comes first, which later, and how far apart they are.','“The dog bit the man”和“The man bit the dog”用的是同样的 token，意思却相反。要预测下一个词，模型必须知道 token 的先后：谁在前，谁在后，隔多远。'),
       ()=>tx('An embedding only depends on which token it is: the same token at position 5 and at position 50 gets the same row of the table. And the computations that follow apply the same operation to every vector, whatever its place in the line, so the model has to be told the position some other way.',
              'embedding 只取决于 token 是谁：同一个 token 在第 5 个位置和第 50 个位置，取出的是表里同一行。而后面的计算对每个向量做同样的运算，不看它排在第几个，所以位置得用别的办法告诉模型。'),
       ()=>tx('The most direct way is another lookup, just like the embedding: a position table with one row per position, called a position embedding, each row as long as the row the embedding picks (below, the token vector). The token at position k takes row k and adds it to its own token vector. After the addition, the same token at different positions has different input vectors. GPT-2 (a language model OpenAI released in 2019) learns its position table in training; the first Transformer computed each position’s vector with sine and cosine functions. Both add it to the token vector this way.',
              '最直接的做法和 embedding 一样是查表：再准备一张位置表，每个位置一行，叫 position embedding（位置向量），每行和 embedding 取出的那一行（下面叫 token 向量）一样长。第 k 个位置的 token 取出第 k 行，加到自己的 token 向量上。相加以后，同一个 token 放在不同位置，输入向量就不同了。GPT-2（OpenAI 2019 年发布的语言模型）的位置表是训练学出来的；最早的 Transformer 用正弦和余弦函数算出每个位置的向量。两者都是这样加到 token 向量上。'),
       ()=>tx('Llama has no position table: its input vector is just the row the embedding picks. It adds the position inside attention, by turning vectors according to their position (RoPE); the next section shows at which step.',
              'Llama 没有位置表，输入向量就是 embedding 取出的那一行；位置是在注意力里加的：按位置转动向量（RoPE），下一节会看到在哪一步。')),
   {fig:[prFr(()=>prPosFig(0),()=>tx('In GPT-2, the token at position k takes row k of the position table: its position vector.','GPT-2 里，第 k 个位置的 token 取出位置表的第 k 行，就是它的位置向量。'),{ms:6000}),
         prFr(()=>prPosFig(1),()=>tx('In GPT-2, each token’s vector plus its position vector gives the input vector. It now says which token it is and where it stands.','GPT-2 里，每个 token 的向量加上它的位置向量，得到输入向量：既说明是哪个 token，也说明它在第几个。'),{ms:6500}),
         prFr(()=>prPosFig(2),()=>tx('In GPT-2, “␣is” at position 5 and at position 50: the same token vector plus two different position vectors gives two different input vectors.','GPT-2 里，“␣is”在第 5 个和第 50 个位置：token 向量相同，加上的位置向量不同，输入向量也就不同。'),{ms:6500})],cls:PRC}]},
 {h:()=>tx('Tokens look back at earlier tokens through attention','token借助注意力机制回看过去的信息'),parts:[
   prP(()=>tx('So far each vector stands for its own token only. But what decides the word after “is” lies earlier in the sentence: France, capital. Each token has to take information from the tokens before it. This step is attention; since the tokens look at other tokens of the same sentence, it is also called self-attention.',
              '到这里，每个向量只代表它自己那个 token。可是决定“is”后面是什么词的信息在前面：France、capital。每个 token 都要能从前面的 token 取信息，这一步就是注意力（attention）；token 看的是同一句话里的其他 token，所以也叫自注意力（self-attention）。'),
       ()=>tx(`Each vector is multiplied by three learned matrices ${M_('W_q')}, ${M_('W_k')}, ${M_('W_v')} into a query, a key and a value (q, k, v): q says what the token is looking for, k how it can be found, and v the content it hands over. The dot product of token i’s query with token j’s key, divided by ${M_('\\sqrt{d_{head}}')}, is a score: how much i should take from j. ${M_('d_{head}')} is the dimension of q and k; the dot product is a sum of ${M_('d_{head}')} terms, so without the division the scores grow with the dimension and softmax would put almost all the weight on one token.`,
              `每个向量分别乘三个训练学出来的矩阵 ${M_('W_q')}、${M_('W_k')}、${M_('W_v')}，得到 query、key、value 三个向量（q、k、v）：q 表示这个 token 要找什么，k 表示它能被怎样找到，v 是它被取走的内容。token i 的 q 和 token j 的 k 做点积，再除以 ${M_('\\sqrt{d_{head}}')}，得到一个得分：i 该从 j 取多少。${M_('d_{head}')} 是 q 和 k 的维数；点积是 ${M_('d_{head}')} 项的和，不除的话数值随维数变大，softmax 会把几乎全部权重压到一个 token 上。`),
       ()=>tx(`A token sees only itself and the tokens before it (the causal mask): when the next word is predicted, the later words do not exist yet. Softmax turns token i’s scores for all ${M_('j \\le i')} into weights that add up to 1, and the token takes the weighted sum of the values:`,
              `每个 token 只能看它自己和前面的 token（因果 mask）：预测下一个词的时候，后面的词还不存在。softmax 把 token i 对所有 ${M_('j \\le i')} 的得分变成加起来是 1 的权重，再按权重把各个 v 加起来：`),
       ()=>`<span class="pr-fx">${M_('\\mathrm{out}_i=\\sum_{j\\le i}\\mathrm{softmax}_j\\!\\left(\\frac{q_i\\cdot k_j}{\\sqrt{d_{head}}}\\right)v_j')}</span>`,
       ()=>tx('A high weight means much of that token’s information is mixed in.','权重高，说明从那个 token 混进来的信息多。')),
   {fig:[prFr(()=>popAttn(2),()=>tx('Attention computes a score for every pair of tokens:','注意力给每一对 token 算一个得分：')+' '+M_(P_ATTN.steps[2].fx),{rep:()=>[
           [/(<foreignObject[^>]*><div[^>]*>)<span>Llama-3\.1-8B:<\/span>[\s\S]*?(<\/div><\/foreignObject>)/,(m,a,b)=>a+M_('d_{head}')+esc(tx(': the dimension of q and k','：q 和 k 的维数'))+b],
           [tx('the scores need not add up to anything yet','这些得分现在还不用加起来等于什么'),tx('any real number; softmax turns them into weights','得分可以是任意实数，softmax 之后才是权重')]]}),
         prFr(()=>popAttn(3),prCap(P_ATTN,3),{rep:REP_V}),
         prFr(()=>popAttn(4),()=>tx('Each token adds up the values v by its attention weights.','每个 token 按注意力权重把各个 v 加起来。'),{ms:6500,rep:REP_V})],cls:PRC},
   prP(()=>tx('These scores only look at the two vectors, and Llama’s input vectors have no position added: shuffle the tokens and every pair keeps its score. The causal mask only says who may look at whom, not how far apart two tokens are.',
              '这样算出的得分只看两个向量本身，而 Llama 的输入向量里没有加位置：把 token 打乱，每一对的得分都不变。因果 mask 只规定谁能看谁，不告诉模型两个 token 隔多远。'),
       ()=>tx(`Llama adds the position right here, with RoPE (rotary position embedding): the components of q and k are paired up, each pair is treated as a 2-D vector and rotated by an angle set by the token’s position, ${M_('m\\theta')} at position m, with a different ${M_('\\theta')} for each pair; v is not rotated. A rotation keeps lengths, so for a query at position m and a key at position n only the difference of the two angles is left in their dot product:`,
              `Llama 就在这一步加入位置，用的是 RoPE（旋转位置编码）：把 q 和 k 的分量两两配成一对，每一对看成一个二维向量，按 token 的位置旋转，位置 m 的 token 转 ${M_('m\\theta')}，每一对的 ${M_('\\theta')} 各不相同；v 不转。旋转不改变长度，所以位置 m 的 q 和位置 n 的 k 做点积时，只剩两个角度之差：`),
       ()=>`<span class="pr-fx">${M_('(R_{m\\theta}\\,q)\\cdot(R_{n\\theta}\\,k)=q\\cdot R_{(n-m)\\theta}\\,k')}</span>`,
       ()=>tx(`The score depends on q and k themselves and, for position, only on the distance ${M_('m-n')}, not on where the two tokens stand in the sentence.`,`得分除了取决于 q、k 本身，只取决于两个 token 的距离 ${M_('m-n')}，和它们在句子里的绝对位置无关。`)),
   {fig:[prFr(()=>popEmb(4),()=>tx('Without position information, attention is order-blind: shuffle the tokens and every pair keeps its score.','没有位置信息时，注意力分不出顺序：把 token 打乱，每一对的得分都不变。'),{ms:6500}),
         prFr(()=>popEmb(5),()=>tx('RoPE turns q and k by their positions, not v. For the same content, position acts only through the distance m − n.','RoPE 按位置转动 q 和 k，v 不转。内容相同时，位置的影响只取决于距离 m − n。'),{ms:6500}),
         prFr(()=>popAttn(0),()=>tx(`Three blue matrices ${M_('W_q')}, ${M_('W_k')}, ${M_('W_v')} turn each token’s vector into q, k and v; RoPE then turns q and k by the token’s position (↻), not v.`,`三个蓝色矩阵 ${M_('W_q')}、${M_('W_k')}、${M_('W_v')} 把每个 token 的向量变成 q、k、v；RoPE 再按 token 的位置转动 q 和 k（↻），v 不转。`),{ms:6500})],cls:PRC}]},
 {h:()=>tx('Several attention heads look from several angles at once','多种注意力让token同时参考多个角度'),parts:[
   prP(()=>tx('One set of weights adds up to 1, so it cannot give most of its weight to two different tokens at once. Yet the next word often needs several things at once: the word just before, the subject of the sentence, a name further back.','一组权重加起来是 1，没法同时把大部分权重给两个不同的 token。可预测下一个词常常同时需要几种信息：紧挨着的前一个词、句子的主语、更前面出现过的名字。'),
       ()=>tx('So q, k and v are each cut into parts, one per head. Each head computes its own scores, weights and output from its own part, independently of the others; each head has its own set of weights adding up to 1, so one head can attend to the word just before while another attends to the subject.',
              '所以把 q、k、v 各切成几份，每份交给一个 head。每个 head 只用自己那份，算自己的得分、权重和输出，互不相干；每个 head 有自己一组加起来是 1 的权重，所以一个 head 可以盯住前一个词，另一个 head 同时盯住主语。'),
       ()=>tx(`Llama 3.1 8B has ${NH} heads, and ${nf(DM)} ÷ ${NH} = ${HD}: each head’s q and k have ${HD} numbers, the ${M_('d_{head}')} of the formula above. Llama cuts k and v into only ${KV} parts, each shared by ${NH/KV} heads (grouped-query attention, GQA), so fewer keys and values are computed and stored.`,
              `Llama 3.1 8B 有 ${NH} 个 head，${nf(DM)} ÷ ${NH} = ${HD}：每个 head 的 q 和 k 是 ${HD} 维，就是上一节公式里的 ${M_('d_{head}')}。k 和 v 在 Llama 里只切成 ${KV} 份，每 ${NH/KV} 个 head 共用一份（分组查询注意力，GQA），要算、要存的 k 和 v 就少了。`),
       ()=>tx(`The heads’ outputs are joined back into ${nf(DM)} numbers, multiplied by one more matrix, ${M_('W_o')}, and added to the token’s vector: attention only computes what to add, and what was there is kept.`,`各 head 的输出拼回 ${nf(DM)} 维，乘一个矩阵 ${M_('W_o')}，再加到这个 token 原来的向量上：注意力只算出要补充的信息，原来的信息不丢。`)),
   {fig:[prFr(()=>popAttn(1),()=>tx('q, k and v are cut per head; each head uses only its own part.','q、k、v 按 head 切开，每个 head 只用自己那一份。'),{ms:6000}),
         prFr(()=>popAttn(5),()=>tx(`The heads meet only here: their outputs are joined, multiplied by ${M_('W_o')} and added to the vector at ⊕.`,`各个 head 只在这里汇合：输出拼起来，乘 ${M_('W_o')}，在 ⊕ 处加到原来的向量上。`),{ms:6000})],cls:PRC}]},
 {h:()=>tx('What attention gathered needs further processing','经过注意力提取的信息需要二次加工'),parts:[
   prP(()=>tx('Attention moves information between tokens: its output is a weighted average of values, which can only blend existing vectors in proportion. Combining what was gathered into new features, such as getting the idea of Paris from capital and France, takes one more nonlinear transformation of each token’s vector.',
              '注意力负责在 token 之间搬运信息：它的输出是各个 v 的加权平均，只能把已有的向量按比例混在一起。把搬来的信息组合成新的特征，比如从 capital 和 France 得出 Paris 这个意思，需要对每个 token 的向量再做一次非线性变换。'),
       ()=>tx(`So each token’s vector then goes through an MLP (multilayer perceptron) on its own; tokens exchange nothing here. A classic MLP multiplies the vector by a matrix into more dimensions, applies a nonlinear function to each component, and multiplies by another matrix back to the original size: in Llama 3.1 8B from ${nf(DM)} to ${nf(FFN)} numbers and back to ${nf(DM)}.`,
              `所以每个 token 的向量再各自过一个 MLP（多层感知机），token 之间在这里不交换信息。经典的 MLP 先乘一个矩阵把向量升维，逐个分量过一个非线性函数，再乘一个矩阵降回原来的维数：Llama 3.1 8B 从 ${nf(DM)} 维升到 ${nf(FFN)} 维，再降回 ${nf(DM)} 维。`),
       ()=>tx(`Llama uses SwiGLU: the up matrix proposes features, the gate matrix’s result goes through SiLU, ${M_('\\mathrm{SiLU}(x)=x\\,\\sigma(x)')} with ${M_('\\sigma')} the sigmoid, to decide how much of each passes, the two are multiplied element by element, and the down matrix maps back to ${nf(DM)} numbers, which are added to the vector: ${M_('W_{down}\\,(\\mathrm{SiLU}(W_{gate}\\,x)\\odot W_{up}\\,x)')}.`,
              `Llama 用的是 SwiGLU：up 矩阵提出候选特征，gate 矩阵的结果过 SiLU（${M_('\\mathrm{SiLU}(x)=x\\,\\sigma(x)')}，${M_('\\sigma')} 是 sigmoid），决定每个特征放行多少，两者逐元素相乘，再由 down 矩阵变回 ${nf(DM)} 维，加到原来的向量上：${M_('W_{down}\\,(\\mathrm{SiLU}(W_{gate}\\,x)\\odot W_{up}\\,x)')}。`),
       ()=>tx(`Most of the model’s parameters are in the MLP: three ${nf(DM)} × ${nf(FFN)} matrices, about ${mi(PM)} million parameters, more than four times the attention matrices (about ${mi(PA)} million).`,
              `模型的大部分参数在 MLP 里：三个 ${nf(DM)} × ${nf(FFN)} 的矩阵，共约 ${yi(PM)} 亿个参数，是注意力那几个矩阵（约 ${yi(PA)} 亿个）的四倍多。`)),
   {fig:[prFr(()=>popMlp(0),()=>tx('A classic MLP: up to more dimensions, a nonlinear function, back down.','经典的 MLP：升维，过一个非线性函数，再降维。')),prFr(()=>popMlp(1),prCap(P_MLP,1)),prFr(()=>popMlp(2),prCap(P_MLP,2)),prFr(()=>popMlp(3),()=>tx('The result is added to the vector.','结果加到原来的向量上。'))],cls:PRC}]},
 {h:()=>tx('Attention and an MLP form a layer, repeated many times','注意力和MLP组合成一层，并重复多次'),parts:[
   prP(()=>tx(`One attention and one MLP gather once and process once. Getting from “capital of France” to “Paris” takes several such steps, so attention and the MLP form one layer, and the model stacks ${NL} of them, each with its own parameters.`,`一次注意力加一次 MLP，只能取一次信息、加工一次。从“capital of France”到“Paris”要经过好几步这样的组合，所以注意力和 MLP 组成一层，模型叠 ${NL} 层，每层有自己的参数。`),
       ()=>tx('A deep stack has two problems: the vector’s length grows or shrinks layer after layer until it overflows or vanishes, and what the early layers found is overwritten by the later ones.','叠得这么深有两个问题：向量的长度会一层层放大或缩小，最后溢出或者消失；前面几层找到的信息，会被后面的层覆盖掉。'),
       ()=>tx(`Two fixes. First, attention and the MLP (each called a block) apply RMSNorm before they use the vector: it divides the vector by the root mean square of its components and multiplies by learned gains ${M_('g')}, so every block gets an input of steady length: ${M_('\\mathrm{RMSNorm}(x)=\\frac{x}{\\sqrt{\\mathrm{mean}(x^2)}}\\odot g')}.`,
              `两个办法。一是注意力和 MLP（各叫一个块）在用到向量之前先做 RMSNorm：把向量除以它各分量的均方根，再乘一组学出来的系数 ${M_('g')}，让每个块拿到的输入长度稳定：${M_('\\mathrm{RMSNorm}(x)=\\frac{x}{\\sqrt{\\mathrm{mean}(x^2)}}\\odot g')}。`),
       ()=>tx(`Second, a block does not replace the vector; its output is added to it, as in the last two sections. This is the residual connection. With h the token’s current vector: ${M_('h \\leftarrow h + \\mathrm{Attn}(\\mathrm{RMSNorm}(h))')}, then ${M_('h \\leftarrow h + \\mathrm{MLP}(\\mathrm{RMSNorm}(h))')}. This vector, passed from the first layer to the last with every block adding to it, is the residual stream.`,
              `二是块不替换向量，而是把输出加回去，就是前两节里的“加到原来的向量上”，叫残差连接。记这个 token 当前的向量为 h：${M_('h \\leftarrow h + \\mathrm{Attn}(\\mathrm{RMSNorm}(h))')}，再 ${M_('h \\leftarrow h + \\mathrm{MLP}(\\mathrm{RMSNorm}(h))')}。这个从第一层一直传到最后一层、每个块都往上加东西的向量，叫残差流（residual stream）。`)),
   {fig:[prFr(()=>popLayer(),()=>tx(`With RMSNorm and the residual connection: after ${NL} layers the vector neither explodes nor fades, and much of the input is still there.`,`有 RMSNorm 和残差连接：${NL} 层之后，向量既没有爆炸也没有消失，输入的信息还留着不少。`),{noNorm:false,noRes:false,ms:6000,rep:REP_H}),
         prFr(()=>popLayer(),()=>tx('Without RMSNorm: the vector grows layer after layer.','去掉 RMSNorm：向量一层比一层大。'),{noNorm:true,noRes:false,ms:6000,rep:REP_H}),
         prFr(()=>popLayer(),()=>tx('Without the residual connection: little of the input is left after a few layers.','去掉残差连接：过了几层，输入的信息就剩不下多少了。'),{noNorm:false,noRes:true,ms:6000,rep:REP_H})],cls:PRC}]},
 {h:()=>tx('The model’s output is the probability of the next token','模型最终输出是下一个token的概率'),parts:[
   prP(()=>tx(`After ${NL} layers the vector of the last token, ␣is, carries what the sentence says about the next word: under the causal mask only it has seen the whole sentence, so the next word is computed from it alone. One step is left: turn it into a probability distribution like the one at the top.`,
              `${NL} 层之后，最后一个 token ␣is 的向量已经带上了这句话里预测下一个词需要的信息：因果 mask 下只有它看过整句话，所以下一个词只从它来算。只差最后一步：把它变成开头那样的概率分布。`),
       ()=>tx(`The residual stream has only been added to, never rescaled, so a last RMSNorm comes first. Then the LM head: the output layer’s matrix (unrelated to the attention heads), one row per vocabulary token, ${VN} rows. Each row’s dot product with the vector is that token’s score, its logit. Softmax turns the ${VN} logits into probabilities that add up to 1.`,
              `残差流一路上只被加、没被缩放过，所以先做最后一次 RMSNorm，再乘 LM head：输出层的一个矩阵（和注意力里的 head 无关），词表里每个 token 一行，共 ${VN} 行。每一行和这个向量做点积，得到这个 token 的得分，叫 logit。softmax 把 ${VN} 个 logit 变成加起来是 1 的概率。`),
       ()=>tx('A token is picked, the most likely one (greedy) or one drawn at random by its probability (sampling), appended to the sentence, and the whole computation runs again for the token after it.','挑出一个 token（取概率最大的，叫贪心；或者按概率随机抽一个，叫采样），接到句子后面，整个计算再跑一遍，得到再下一个 token。')),
   {fig:[prFr(()=>popOut(0),prCap(P_OUT,0),{ms:6500}),prFr(()=>popOut(1),()=>tx('Softmax turns the logits into probabilities.','softmax 把 logit 变成概率。'),{ms:6000})],cls:PRC}]},
 {h:()=>tx('The whole Transformer at a glance','Transformer流程总览'),parts:[
   prP(()=>tx(`All the steps in one line: tokenization → embedding → ${NL} layers of (RMSNorm → attention → ⊕ → RMSNorm → MLP → ⊕) → a last RMSNorm → LM head → softmax → the next token.`,`把前面的步骤连起来：token化 → embedding → ${NL} 层（RMSNorm → 注意力 → ⊕ → RMSNorm → MLP → ⊕）→ 最后一次 RMSNorm → LM head → softmax → 下一个 token。`),
       ()=>tx('In the picture each token is one coloured lane flowing down through the layers; the lane of the last token, ␣is, reaches the LM head, which gives “Paris”.','图里每个 token 是一条彩色的线，从上往下流过各层；最后一个 token ␣is 的那条线到 LM head 得出“Paris”。')),
   {fig:[{html:()=>`<div class="ar-ov pr-ovbox"><canvas class="ar-ovcv" aria-hidden="true"></canvas></div>`,cap:()=>tx('The capital of France is → Paris, through the whole model.','“The capital of France is” → “Paris”，走过整个模型。'),ms:1e9}],mount:prOvMount}]}
]});

A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload (the kit saves snap() and hands it to restore() before the first mount).
     The pop-ups' own settings are kept too; the open pop-up and the samples drawn are not. passed: the three core questions are right */
  snap:()=>({v:1,mode,step,S:{m:S.m},pred,seen:Object.keys(seen).filter(k=>seen[k]),chA:chA.slice(),chI,chSub,iq,iqOpen,mk:{on:mkOn,tried:mkTried},
    pop:{aHead,aTok,aMask,a1M,siluX,temp,Z:Z.slice(),noNorm,noRes},passed:coreDone()}),
  restore(o){if(!o||o.v!==1) return; const B=v=>v===true, I=(v,n)=>Number.isInteger(v)&&v>=0&&v<n, N=(v,lo,hi)=>typeof v==='number'&&isFinite(v)&&v>=lo&&v<=hi;
    S={m:o.S&&typeof o.S==='object'&&MODELS[o.S.m]?o.S.m:DEF_M}; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=I(o.step,STEPS.length)?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    seen={}; if(Array.isArray(o.seen)) o.seen.forEach(k=>{if(typeof k==='string'&&POP[k]) seen[k]=true;});
    const Q=CHALL(); chA=Q.map((q,i)=>Array.isArray(o.chA)&&I(o.chA[i],q.o.length)?o.chA[i]:null); chI=I(o.chI,Q.length)?o.chI:0; chSub=B(o.chSub);
    iq=Number.isInteger(o.iq)&&o.iq>=0&&o.iq<1e6?o.iq:0; iqOpen=B(o.iqOpen);
    const mk=o.mk&&typeof o.mk==='object'?o.mk:{}; mkTried=B(mk.tried); mkOn=mkTried?mk.on!==false:true;
    const p=o.pop&&typeof o.pop==='object'?o.pop:{}; aHead=I(p.aHead,HEADS.length)?p.aHead:1; aTok=I(p.aTok,TOKS.length)?p.aTok:4;
    aMask=MASKS.some(x=>x.k===p.aMask)?p.aMask:'causal'; a1M=B(p.a1M); siluX=N(p.siluX,-5,5)?p.siluX:-1.5; temp=N(p.temp,0.2,2)?p.temp:1;
    Z=Array.isArray(p.Z)&&p.Z.length===Z0.length&&p.Z.every(v=>N(v,0,12))?p.Z.slice():Z0.slice(); noNorm=B(p.noNorm); noRes=B(p.noRes);},
  mount(el,tok){
    root=el; if(tok&&MODELS[tok]){S.m=tok; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#ar-app',el);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#ar-rail',app));},token:()=>S.m});
    bind(app); renderIQ(); renderRail();
    ov.P=null; ovTarget(S.m); ovMount($('#ar-ov',app));
    if(dlg){const k=dlg, s=dst; dlg=null; openDlg(k,null,s);}
    render();
    /* the causal switch measures its labels: draw it again once the web fonts are in */
    if(document.fonts&&document.fonts.ready) document.fonts.ready.then(()=>{if(app&&isMk()) mkDraw(true);});
  },
  leave(){clearTimeout(annT); stopPlay(); stageStop(); if(ov.loop){ov.loop.kill(); ov.loop=null;} dlg=null; dlgShown=''; mkShown=''; mkShownW=0; root=null; app=null;},
  onLang(){if(root) this.mount(root,null);},
  /* the overview fits its spacing to the window's height (ovFit); with reduced motion, or off screen, its loop draws only a still frame on
     a change of view, so a resize draws that still frame again (精装修 step 2: after a resize to a shorter window it kept the taller
     drawing and the stage scrolled inside); a running loop redraws on its next frame anyway */
  onResize(){if(isMk()) mkDraw(); if(ov.loop&&!ov.loop.on) ov.loop.sync();},
  termTip, jump, jumpAct,
  dock:{
    anchor:()=>app&&$(isMk()?'#ar-mk':'#ar-ov',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#ar-graph',app),
    render({status,vis,read}){if(!app) return; const m=cur();
      if(isMk()){const pp=mkGen(mkOn)[0]; vis.innerHTML=`<div class="ar-dkp${mkOn?'':' bad'}" data-tipx="${esc(tx(`“ Paris”’s chance of coming next when the model generates: ${pp.toFixed(1)}%.`,`生成时 “ Paris” 排在下一个的概率：${pp.toFixed(1)}%。`))}"><i style="width:${pp}%"></i></div>`;
        status.className='status '+(mkOn?'ok':'bad'); status.textContent=mkOn?tx('Mask on','mask 开'):tx('Mask off','mask 关'); read.textContent=mkRead(); return;}
      vis.innerHTML=`<div class="ar-dk" aria-hidden="true">${[0,1,2,3,4].map(k=>`<i style="background:var(--ar-t${k})"></i>`).join('')}</div>`;
      status.className='status neutral'; status.textContent=short(S.m); read.textContent=capText();}
  }
});

/* ---------- home tile: the hero's flow in small; one click moves the highlight to the next stage; one short line (R11.5) ---------- */
const tile={cur:2,p:0,cv:null,loop:null,el:null};
function tileLayout(W,H){const g=10, top=8, h=H-top-24, wp=(W-5*g)/6.4, wc=wp*1.4; let x=0;
  return HST.map((s,i)=>{const w=i===tile.cur?wc:wp, r={x,y:top,w,h,e:0}; x+=w+g; return r;});}
function tilePaint(c,W,H,t,still){
  const C=cols(), st=tileLayout(W,H);
  /* the loop, under the stages */
  const a=st[5], z=st[0], yb=a.y+a.h+2; c.strokeStyle=rgba(C.ink3,0.4); c.setLineDash([3,4]); c.lineWidth=1.1; c.beginPath(); c.moveTo(a.x+a.w/2,yb); c.bezierCurveTo(a.x+a.w/2,H-2,z.x+z.w/2,H-2,z.x+z.w/2,yb+2); c.stroke(); c.setLineDash([]);
  arrowHead(c,z.x+z.w/2,yb+1,-Math.PI/2,6,rgba(C.ink3,0.65));
  const paths=[];
  for(let g=0;g<5;g++){const A_=st[g], B=st[g+1]; for(let k=0;k<5;k++){if(g===4&&k!==4) continue; const y0=g===4?probY(A_,0):laneY(A_,k), y1=g===4?laneY(B,2):laneY(B,k);
    const pts=[[A_.x+A_.w,y0],[B.x,y1]]; c.strokeStyle=rgba(g===4?C.ink:C.tok[k],0.7); c.lineWidth=1.3; strokePts(c,pts); paths.push([pts,g===4?C.ink:C.tok[k]]);}}
  st.forEach((r,i)=>{const on=i===tile.cur; rr(c,r.x,r.y,r.w,r.h,8); c.fillStyle=on?C.surf:rgba(C.surf2,0.9); c.fill(); c.lineWidth=on?1.6:1; c.strokeStyle=on?rgba(C.ink,0.6):rgba(C.ink3,0.35); c.stroke();
    c.__panel=i; heroPreview(c,C,HST[i].k,r,1,t,still,tile); c.__panel=-1;});
  if(!still){paths.forEach(([pts,col],j)=>dots(c,mkPath(pts),t,mix(col,C.ink,0.25),{sp:18,v:30,r:1.7,ph:j*5}));
    // the lanes end at the probabilities: no particles run through that panel (a blue lane used to cross its bars; Batch 3 review)
    st.forEach((r,i)=>{if(i>=4) return; for(let k=0;k<5;k++){const y=laneY(r,k); dots(c,mkPath([[r.x+2,y],[r.x+r.w-2,y]]),t,mix(C.tok[k],C.ink,0.25),{sp:18,v:30,r:1.7,ph:k*7+i*5});}});}
  return st;}
function tileDraw(t,still){const cv=tile.cv; if(!cv||!cv.isConnected) return; const pw=cv.getBoundingClientRect().width; if(pw<60) return; const {c,W,H}=fit(cv); c.clearRect(0,0,W,H); tilePaint(c,W,H,t,still);}
function tileRead(){const s=HST[tile.cur]; return tx(`${L(s.t)}: ${L(HDESC[s.k]).replace(/\.$/,'')}. Click the flow for the next stage.`,`${L(s.t)}：${L(HDESC[s.k]).replace(/。$/,'')}。点这张图看下一步。`);}
A.registerTeaser(ID,{fid:'illus',render(el){
  if(tile.loop){tile.loop.kill(); tile.loop=null;}
  const it=(sw,t,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${sw}"></i>${t}</span>`;
  el.innerHTML=`<div class="tz ar-tzw"><button type="button" class="ar-tzbtn" aria-label="${esc(tx('Show the next stage','看下一个阶段'))}"><canvas class="ar-tzcv" aria-hidden="true"></canvas><span class="ar-tz-next" aria-hidden="true">${esc(tx('Next stage ▸','下一个阶段 ▸'))}</span></button>`+
    `<div class="tz-leg ar-tz-leg">`+it('ar-sw-lane',tx('one lane per token','每个 token 一条线'),tx('The 5 tokens of “The capital of France is”, one lane each.','“The capital of France is” 的 5 个 token，每个一条线。'))+
      it('ar-sw-w',tx('blue: matrices','蓝色：矩阵'),tx('Blue marks what is made of parameters.','蓝色标出由参数构成的部分。'))+it('ar-sw-att',tx('teal: attention','青色：注意力'),tx('Only attention moves information between lanes.','只有注意力在线之间搬信息。'))+`</div>`+
    `<p class="tz-read" aria-live="polite"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  tile.el=el; tile.cv=el.querySelector('.ar-tzcv');
  const rd=el.querySelector('.tz-read'), notes=el.querySelectorAll('.tz-note'), upd=()=>{rd.textContent=tileRead(); if(tile.loop&&!tile.loop.on) tile.loop.sync();};
  const N=[tx('An illustrative flow: “The capital of France is” → “ Paris”; the lab animates what happens inside each stage.','示意的数据流：“The capital of France is” → “ Paris”；Lab 里会把每个阶段内部的计算动起来。'),
           tx('Drawn after Transformer Explainer (Georgia Tech Polo Club).','画法参考 Transformer Explainer（佐治亚理工 Polo Club）。')];
  notes.forEach((n,i)=>{n.textContent=N[i]||'';});
  el.querySelector('.ar-tzbtn').addEventListener('click',()=>{tile.cur=(tile.cur+1)%6; upd();});
  upd(); tile.loop=makeLoop(tile.cv,tileDraw);
}});

/* ---------- home hero ---------- */
A.registerHero({id:ID,wide:true,cta:()=>tx('Learn what is inside a Transformer in 10 minutes','10 分钟学会 Transformer 内部结构'),
  mount(el){heroMount(el); if(!el._arb){el._arb=1; heroBind(el);}},
  render(el){heroMount(el);}});

/* read-only hook for the smoke tests */
/* canvas geometry for the layout audit: text boxes, stroked paths and particles of one frame, in client coordinates */
function geom(kind,o){o=o||{};
  const box=kind==='hero'?hero.cv:(kind==='tile'?(tile.cv):ov.cv); if(!box||!box.isConnected) return null;
  const cr=box.getBoundingClientRect(), W=Math.round(cr.width), H=Math.round(cr.height), R=recorder(W,H), t=o.t||0; let extra={};
  if(kind==='hero'){const sv={cur:hero.cur,fx:hero.fx,tr:hero.tr,t0:hero.t0,p:hero.p};
    hero.cur=o.i; hero.fx=o.i; hero.tr=null; hero.t0=t-(o.p||0)*htOf(o.i); const lay=heroPaint(R,W,H,t,false,true); Object.assign(hero,sv); extra.panels=lay.st.map(r=>({x0:r.x,y0:r.y,x1:r.x+r.w,y1:r.y+r.h,e:r.e}));}
  else if(kind==='tile'){tilePaint(R,W,H,t,false);}
  else{const mk=o.m||S.m, res=ovPaint(R,W,mP(mk),t,false,true,ov.k||1); extra.hots=res.R;}
  const g=R.__rec, sh=(q)=>({...q,x0:q.x0+cr.left,x1:q.x1+cr.left,y0:q.y0+cr.top,y1:q.y1+cr.top});
  return {canvas:{x0:cr.left,y0:cr.top,x1:cr.right,y1:cr.bottom},texts:g.texts.map(sh),lines:g.lines.map(l=>({...l,pts:l.pts.map(v=>[v[0]+cr.left,v[1]+cr.top])})),
    dots:g.dots.map(v=>[v[0]+cr.left,v[1]+cr.top,v[2]]),fills:g.fills.map(sh),panels:(extra.panels||[]).map(sh)};}
/* the causal switch's inputs and results, for tests/architecture_numbers.py (which recomputes them) */
const MK=Object.freeze({P:MK_P,C:MK_C,K:MK_K,Z0,S:MK_T.map((_,i)=>MK_T.map((_,j)=>mkS(i,j))),pOn:MKV.pOn,pOff:MKV.pOff,lossOn:MKV.lossOn,lossOff:MKV.lossOff,
  genOn:mkGen(true),genOff:mkGen(false),wOn:MKV.wOn,wOff:MKV.wOff,shown:{lossOn:mkLoss(MKV.lossOn),lossOff:mkLoss(MKV.lossOff),pOn:MKV.pOn.map(mkPct),pOff:MKV.pOff.map(mkPct)}});
window.__ar=Object.freeze({MODELS,TOKS,CANDS,HEADS,MASKS,M1M,CHQ,CHQ_CORE,MK,STEP_IDS:STEPS.map(s=>s.id),softmax,pcts,weightM,simulate,geom,
  heroGo:i=>heroGo(i,false),   /* tests only: the hero has no clickable stages since 2026-10-11 */
  texts:()=>({hero:HST.map(s=>L(s.t)+': '+L(HDESC[s.k])),heroNote:L(HNOTE),parts:STEPS.map(s=>L(s.title)),pops:Object.fromEntries(Object.entries(POP).map(([k,v])=>[k,{t:L(v.t),steps:v.steps.map(x=>L(x))}])),
    tips:Object.fromEntries(Object.keys(TIPS).map(k=>[k,tipText(k)])),presets:PRESETS.map(x=>L(x.t)+': '+L(x.d)),masks:MASKS.map(x=>L(x.t)+': '+L(x.u)),
    chq:CHALL().map(q=>({q:L(q.q),o:q.o.map(x=>L(x)),a:q.a,why:L(q.why),core:q.core})),leads:STEPS.map(s=>s.lead?L(s.lead):''),iq:IQ.map(q=>({q:L(q.q),a:L(q.a)})),descs:DESCS(),
    exits:(()=>{const e=EXITS(); return {explain:e.explain.map(q=>({q:q.q,o:q.o,a:q.a,n:q.n,why:q.why})),transfer:e.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why}))};})(),
    /* the causal switch's words that appear only after the switch: the result box, the caption, the mini result bar, both states */
    mask:(()=>{const o=mkOn, r={}; for(const v of [true,false]){mkOn=v; const k=v?'on':'off'; r[k]={obs:mkObs(),cap:mkCap(),read:mkRead()};} mkOn=o;
      r.tips=[mkTipLoss(),mkTipLossOn(),mkTipLossOff(),mkTipOn(),mkTipOff()]; r.wait=tx('Turn the mask off first to see what it prevents.','先把 mask 关掉，看看它挡住了什么。'); return r;})()}),
  state:()=>({m:S.m,mode,step,pred,dlg,dst,aHead,aTok,aMask,a1M,noNorm,noRes,temp,chA:chA.slice(),chSub,mk:{on:mkOn,tried:mkTried},hero:{cur:hero.cur,narrow:hero.narrow,on:!!(hero.loop&&hero.loop.on),ms:Math.round(performance.now()-hero.t0),ht:htOf(hero.cur)},ov:{on:!!(ov.loop&&ov.loop.on),focus:ov.focus?ov.focus.slice():null},chI})});
})();
