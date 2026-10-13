/* SFT lab: which tokens of a conversation count in the supervised fine-tuning loss, and why the end-of-turn token must.
   Two engines. (1) Token counts: a conversation in the Llama 3.1 chat format, every position labeled with its token (counted) or
   −100 (ignored, PyTorch's default ignore_index); a part's weight in the mean = its counted tokens ÷ all counted tokens.
   (2) A live toy: a 26-token vocabulary, logits = b + Σ_{j=1..8} W_j[token j back] (5,434 parameters), pretrained on plain FAQ
   text, then fine-tuned on 6 example conversations by full-batch gradient descent, mean cross-entropy over the counted positions.
   No random numbers anywhere: zero start, full batch, greedy decoding, so the same settings always give the same run.
   Generation is a mechanism replay (Medium by default, R3.9); a run watched to the end is not replayed (R3.20); a run plays where it
   can be seen, and outside the guided parts a change's run ends within 1.5 s (G6, the owner's message of 2026-10-08).
   The home-tile teaser (teasers/sft.js) reads the toy through window.__sft. Tests: tests/sft_numbers.py (recomputes the toy in numpy), tests/sft_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='sft';

/* ======================= engine 1: the toy ======================= */
/* the vocabulary: special tokens exactly as Meta's Llama 3.1 tokenizer spells them; ordinary words are toy tokens (one word or
   phrase each), shown in the page's language. The order matters only for ties in argmax; tests/sft_numbers.py uses the same. */
const VOC=[
 {k:'bot',s:'<|begin_of_text|>',sp:1},{k:'sh',s:'<|start_header_id|>',sp:1},{k:'eh',s:'<|end_header_id|>',sp:1},{k:'eot',s:'<|eot_id|>',sp:1},{k:'pad',s:'<|finetune_right_pad_id|>',sp:1},
 {k:'user',s:'user',mono:1},{k:'asst',s:'assistant',mono:1},{k:'nl',s:'\\n\\n',mono:1},
 {k:'what',en:'What is',zh:'什么是'},{k:'qm',en:'?',zh:'？'},{k:'dot',en:'.',zh:'。'},{k:'is',en:'is',zh:'是'},
 {k:'kv',s:'KV cache'},{k:'batch',s:'batch'},{k:'token',s:'token'},{k:'gpu',s:'GPU'},{k:'softmax',s:'softmax'},{k:'logit',s:'logit'},
 {k:'d_kv',en:'the saved keys and values',zh:'存下来的 key 和 value'},{k:'d_batch',en:'requests run together',zh:'一起算的一组请求'},
 {k:'d_token',en:'a small piece of text',zh:'一小段文字'},{k:'d_gpu',en:'the chip that does the math',zh:'做计算的芯片'},
 {k:'d_softmax',en:'the step that turns scores into probabilities',zh:'把得分变成概率的一步'},{k:'d_logit',en:'a score before softmax',zh:'softmax 之前的得分'},
 {k:'U',en:'User:',zh:'用户：'},{k:'As',en:'Assistant:',zh:'助手：'}];
const NV=VOC.length, IX={}; VOC.forEach((v,i)=>{IX[v.k]=i;});
const K=8, LR=1, PRE=300, NPAR=K*NV*NV+NV, MAXG=16, STEPS_OPT=[20,60,150], NPAD=3;
const TERMS=['kv','batch','token','gpu','softmax','logit'];
const q=t=>['what',t,'qm'], a=t=>[t,'is','d_'+t,'dot'], head=r=>['sh',r,'eh','nl'];
function conv(t){return {p:['bot',...head('user'),...q(t),'eot',...head('asst')], r:[...a(t),'eot']};}
const ORDERS=[[0,1,2,3,4,5],[3,0,5,2,1,4],[1,4,0,3,5,2],[5,2,4,1,0,3]];   // four FAQ pages, the same six Q&A in different orders
const PAGES=ORDERS.map(o=>o.flatMap(i=>[...q(TERMS[i]),...a(TERMS[i])]));
const CONVS=TERMS.map(conv);
const P_TMPL=conv('kv').p, P_PLAIN=['what','kv','qm'], P_DIFF=['U','what','kv','qm','As'];
const ids=keys=>keys.map(k=>IX[k]);
function newModel(){return {W:new Float64Array(K*NV*NV),b:new Float64Array(NV)};}
const cloneM=m=>({W:m.W.slice(),b:m.b.slice()});
function logitsAt(m,x,t,z){z.set(m.b); for(let j=1;j<=K&&t-j>=0;j++){const o=((j-1)*NV+x[t-j])*NV; for(let v=0;v<NV;v++) z[v]+=m.W[o+v];} return z;}
function softmax(z,p){let mx=-Infinity; for(let v=0;v<NV;v++) if(z[v]>mx) mx=z[v]; let s=0; for(let v=0;v<NV;v++){p[v]=Math.exp(z[v]-mx); s+=p[v];} for(let v=0;v<NV;v++) p[v]/=s; return p;}
/* data: every counted position {x (token ids), t (position), y (its token)}; one step = mean cross-entropy over them, plain gradient descent */
function train(m,data,steps){
  const N=data.length, z=new Float64Array(NV), p=new Float64Array(NV), gW=new Float64Array(K*NV*NV), gb=new Float64Array(NV);
  for(let s=0;s<steps;s++){gW.fill(0); gb.fill(0);
    for(const d of data){logitsAt(m,d.x,d.t,z); softmax(z,p); p[d.y]-=1;
      for(let v=0;v<NV;v++) gb[v]+=p[v];
      for(let j=1;j<=K&&d.t-j>=0;j++){const o=((j-1)*NV+d.x[d.t-j])*NV; for(let v=0;v<NV;v++) gW[o+v]+=p[v];}}
    for(let i=0;i<gW.length;i++) m.W[i]-=LR*gW[i]/N;
    for(let v=0;v<NV;v++) m.b[v]-=LR*gb[v]/N;}
  return m;
}
function pageData(){const out=[]; PAGES.forEach(pg=>{const x=ids(pg); for(let t=1;t<x.length;t++) out.push({x,t,y:x[t]});}); return out;}
/* which positions count: every position after the first ('all') or the response's ('resp'). mk: '' none masked; 'ans' the answer's
   end token gets −100; 'id' every position holding the end token's id gets −100 (the pad token is the end token) */
function convData(count,mk){const out=[];
  CONVS.forEach(c=>{const keys=[...c.p,...c.r], x=ids(keys);
    for(let t=1;t<x.length;t++){if(count!=='all'&&t<c.p.length) continue; if(keys[t]==='eot'&&(mk==='id'||(mk==='ans'&&t>=c.p.length))) continue; out.push({x,t,y:x[t]});}});
  return out;}
let BASEM=null; const base=()=>BASEM||(BASEM=train(newModel(),pageData(),PRE));
const RUNS=new Map();
/* the model after SFT, and its loss on the response tokens of the six conversations (end token included, whatever the mask) */
function sftRun(count,mk,steps){const key=`${count}|${mk||'-'}|${steps}`; let r=RUNS.get(key);
  if(!r){const d=convData(count,mk), m=train(cloneM(base()),d,steps); r={m,rl:respLoss(m),n:d.length}; RUNS.set(key,r);} return r;}
function respLoss(m){const z=new Float64Array(NV), p=new Float64Array(NV); let s=0, n=0;
  CONVS.forEach(c=>{const x=ids([...c.p,...c.r]); for(let t=c.p.length;t<x.length;t++){logitsAt(m,x,t,z); softmax(z,p); s-=Math.log(p[x[t]]); n++;}}); return s/n;}
/* greedy decoding: each step takes the most likely token; stops at <|eot_id|> or after MAXG tokens */
function generate(m,prompt){const x=ids(prompt), z=new Float64Array(NV), p=new Float64Array(NV), out=[];
  for(let i=0;i<MAXG;i++){logitsAt(m,x,x.length,z); softmax(z,p); let best=0; for(let v=1;v<NV;v++) if(p[v]>p[best]) best=v;
    out.push({k:VOC[best].k,p:p[best]}); x.push(best); if(VOC[best].k==='eot') break;}
  return out;}
/* where the model should have stopped: the first token after the answer's full stop that is not the end token. stop: the end
   token comes right after the answer's full stop (writing a whole user turn first and then an end token is not stopping) */
function genInfo(g){let red=-1; const iDot=g.findIndex(x=>x.k==='dot'); if(iDot>=0&&iDot+1<g.length&&g[iDot+1].k!=='eot') red=iDot+1;
  const stop=iDot>=0&&iDot+1<g.length&&g[iDot+1].k==='eot', ok=stop&&g[0].k==='kv';
  return {red,stop,ok,first:g.length?g[0].k:null,wroteHeader:g.some(x=>x.k==='sh'),wroteQ:g.some((x,i)=>x.k==='what'&&i>0)};}
/* the response's log-probability under a model: each token's logit, probability and log-probability (the bridge to DPO) */
function respLogp(m,t){const c=conv(t||'kv'), x=ids([...c.p,...c.r]), z=new Float64Array(NV), p=new Float64Array(NV), rows=[];
  for(let i=c.p.length;i<x.length;i++){logitsAt(m,x,i,z); softmax(z,p); rows.push({k:VOC[x[i]].k,z:z[x[i]],p:p[x[i]],lp:Math.log(p[x[i]])});}
  return {rows,sum:rows.reduce((s,r)=>s+r.lp,0)};}

/* ======================= engine 2: token counts ======================= */
/* template tokens: <|begin_of_text|> once; each turn adds its header (4 tokens: <|start_header_id|>, the role, <|end_header_id|>, \n\n)
   and <|eot_id|>. The first token is never predicted (nothing comes before it). */
const EXS={
 toy:{turns:null},
 chat:{turns:[[30,199]]},           // a 30-token question, a 199-token answer (+ its end token = 200)
 article:{turns:[[1790,119]]},      // a 1,790-token article (+ template = 1,800), a 119-token summary (+ end token = 120)
 multi:{turns:[[30,150],[20,120],[25,180]]}};
function partsOf(ex){   // [{who:'p'|'r', kind:'tok'|'text', k|n}] in order
  if(ex==='toy'){const c=conv('kv'); return [...c.p.map(k=>({who:'p',k})),...c.r.map(k=>({who:'r',k}))];}
  const out=[{who:'p',k:'bot'}];
  EXS[ex].turns.forEach(([u,r],i)=>{head('user').forEach(k=>out.push({who:'p',k})); out.push({who:'p',text:'u',n:u,i}); out.push({who:'p',k:'eot'});
    head('asst').forEach(k=>out.push({who:'p',k})); out.push({who:'r',text:'r',n:r,i}); out.push({who:'r',k:'eot',end:1});});
  return out;}
/* counts for one example under one rule: prompt tokens (incl. template), response tokens (incl. end tokens), counted ones */
function counts(ex,count,mk){
  const ps=partsOf(ex); let P=0,R=0,cp=0,cr=0,first=true;
  ps.forEach(x=>{const n=x.text?x.n:1; if(x.who==='p') P+=n; else R+=n;
    let c=n; if(first){c=n-1; first=false;}                 // the first token is not predicted
    if(x.who==='p'){if(count==='all'&&!(mk==='id'&&x.k==='eot')) cp+=c;}
    else if(!(mk&&x.k==='eot')) cr+=c;});
  const N=cp+cr; return {P,R,cp,cr,N,share:N?cr/N:0,wAll:P+R-1,ratio:(P+R-1)/R};}

/* ======================= state ======================= */
const DEF={ex:'toy',count:'resp',eotl:'yes',pad:'sep',tmpl:'same',steps:60};
const CFG=['ex','count','eotl','pad','tmpl','steps'];
/* '' : the end token counts; 'ans': its label after the answer is −100; 'id': padding with the end token, every position with its id is −100 */
const masked=s=>s.pad==='eot'?'id':(s.eotl==='no'?'ans':'');
let S={...DEF}, mode='guided', step=0, pred=null, root=null, app=null, saved=null, chSubmitted=false, chReveal=false;
let tried4=new Set(), tried5=new Set(), drillSeed=3, drillOpen=false, annT=null, wantObs=false, relang=false;
let bk=0, brow=0, restored=false;   // part 6: the step of the calculation and the highlighted token's row; restored: restore() ran before this mount
let lastShown={};   // metric values as last shown, for ▲▼ after the learner's own change (R6.6)
let userAct=false, obsPending=false;
/* the generation player: i = tokens shown (fractional while playing); a picture watched to the end is remembered by its signature */
const SPD={slow:0.9,med:0.45,fast:0.15};   // seconds per token; Medium by default (a mechanism replay, R3.9)
/* G6 (the owner's message of 2026-10-08): in Scenarios, Settings and the challenge, where readers switch often, a run that a change
   starts shows its end within QUICK seconds; Replay plays at the chosen speed */
const QUICK=1.5;
const PL={i:0,on:false,raf:0,last:0,spd:'med',sig:null,g:null,watched:new Set(),quick:false};
/* challenge: a colleague's run, 150 steps; four settings, two of them wrong */
const CH_START={ex:'toy',count:'all',eotl:'yes',pad:'eot',tmpl:'same',steps:150}, CH_KEYS=['count','eotl','pad','tmpl'], CH_GOAL=0.9;
const CH_OPTS={count:['all','resp'],eotl:['yes','no'],pad:['eot','sep'],tmpl:['same','diff']};

/* which picture the generation panel shows */
function genView(){
  if(mode==='guided'){const id=STEPS[step].id; if(id==='mission') return 'basePlain'; if(id==='template') return 'baseTmpl'; if(id==='predict'||id==='dpo') return 'none';}
  return 'sft';}
function genFor(s,gv){
  if(gv==='basePlain') return generate(base(),P_PLAIN);
  if(gv==='baseTmpl') return generate(base(),P_TMPL);
  if(gv==='none') return [];
  return generate(sftRun(s.count,masked(s),s.steps).m,s.tmpl==='diff'?P_DIFF:P_TMPL);}
const sigOf=(g,gv)=>gv+'|'+g.map(x=>x.k+':'+Math.round(x.p*100)).join(',');
/* the challenge's verdict for any combination */
function chEval(c){const s={...CH_START,...c}, g=genFor(s,'sft'), gi=genInfo(g), ct=counts('toy',s.count,masked(s));
  const ch=CH_KEYS.filter(k=>s[k]!==CH_START[k]).length;
  return {c:CH_KEYS.reduce((o,k)=>{o[k]=s[k]; return o;},{}),stop:gi.stop,share:ct.share,ok:gi.stop&&ct.share>=CH_GOAL-1e-12,changes:ch,g:g.map(x=>x.k)};}
function chAll(){const out=[]; CH_OPTS.count.forEach(c=>CH_OPTS.eotl.forEach(e=>CH_OPTS.pad.forEach(p=>CH_OPTS.tmpl.forEach(t=>out.push(chEval({count:c,eotl:e,pad:p,tmpl:t}))))));
  return out;}
const chMin=()=>Math.min(...chAll().filter(x=>x.ok).map(x=>x.changes));

/* ======================= formatting ======================= */
const f=(v,d)=>A.fmt.num(v,d), I=n=>A.fmt.int(n);
const MINUS='−';
const sn=(v,d)=>v<0?MINUS+f(-v,d):f(v,d);
function pc(p){const v=p*100; if(v>0&&v<0.5) return '<1%'; if(v<100&&v>=99.5) return '>99%'; return f(v,0)+'%';}
const pc2=p=>f(p*100,2)+'%';
const pcShare=p=>{const v=p*100; return Math.abs(v-Math.round(v))<1e-9?f(v,0)+'%':(v<10?f(v,2)+'%':f(v,1)+'%');};
const tokS=k=>{const v=VOC[IX[k]]; return v.s!=null?v.s:L({en:v.en,zh:v.zh});};
const isSp=k=>!!VOC[IX[k]].sp, isMono=k=>!!(VOC[IX[k]].sp||VOC[IX[k]].mono);
const code=s=>`<code>${esc(s)}</code>`;
const EOT=code('<|eot_id|>'), PADT=code('<|finetune_right_pad_id|>');
const J=(k,t)=>`<button type="button" class="linkbtn" data-jump="${k}">${t}</button>`;
const DEF_=(t,tip)=>A.ui.def(t,tip);
const NATS=()=>A.ui.def('nats',tx('nats: the unit of a loss computed with the natural log, ln.','nats 是用自然对数 ln 算出来的 loss 的单位。'));
const M=t=>`<div class="sft-math-line">${A.ui.math(t)}</div>`;
/* H2 (the owner, 2026-10-09): a formula with its name in words on the left, and its chain broken at “=” onto lines that line up */
const EQ=(name,lines,op='=')=>`<div class="sft-eq">${lines.map((l,i)=>`<div class="sft-eq-n">${i?'':name}</div><div class="sft-eq-o">${op?A.ui.math(op):''}</div><div class="sft-math-line">${A.ui.math(l)}</div>`).join('')}</div>`;
/* H1 (the owner, 2026-10-09): −log p and the loss typeset, never written as a string */
const NLP=()=>A.ui.math('-\\log\\, p'), LOSS_TEX='-\\frac{1}{N} \\sum_{t} \\log\\, p(x_t | x_{<t})';
const XSYM=()=>tx(`${A.ui.math('x_t')} is the correct token at position t, and ${A.ui.math('x_{<t}')} the tokens before it.`,`${A.ui.math('x_t')} 是位置 t 上正确的 token，${A.ui.math('x_{<t}')} 是它前面的 token。`);
const qTxt=()=>tx('“What is KV cache ?”','“什么是 KV cache ？”');
const zsp=s=>/[A-Za-z0-9)>]$/.test(s)?s+' ':s;
const cite=(url,t)=>`<a href="${url}" target="_blank" rel="noopener">${t}</a>`;
const SRC={
 fmt:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/prompt_format.md',
 card:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_1/MODEL_CARD.md',
 chat:'https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/chat_format.py',
 torch:'https://github.com/pytorch/pytorch/blob/e70172a0c89ad35a2322f359227b15c9e9773677/torch/nn/modules/loss.py',
 alpaca:'https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py',
 alpacaR:'https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/README.md',
 trl:'https://github.com/huggingface/trl/blob/3d9261f1fec9f9a8140099c78a65c7da73dce79c/trl/trainer/sft_config.py',
 trlT:'https://github.com/huggingface/trl/blob/3d9261f1fec9f9a8140099c78a65c7da73dce79c/trl/trainer/sft_trainer.py',
 tulu:'https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/scripts/train/tulu3/finetune_8b.sh',
 hf:'https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/data/data_collator.py',
 tt:'https://github.com/meta-pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/datasets/_alpaca.py',
 shi:'https://arxiv.org/abs/2405.14394', igpt:'https://arxiv.org/abs/2203.02155'};

/* ======================= words used in many places ======================= */
const TOY_TIP=()=>tx(`This small model has a vocabulary of ${NV} tokens and ${I(NPAR)} parameters. It looks only at the 8 tokens before the next one. Each of them, by how far back it is, picks a row of ${NV} scores from its own table. Those rows plus a bias are the ${NV} logits of the next token, and softmax turns them into probabilities. When it writes, it takes the most likely token at every step.`,
  `这个小模型的词表只有 ${NV} 个 token，一共 ${I(NPAR)} 个参数。它只看前面 8 个 token。每个 token 按离得多远，从对应的那张表里查出一行 ${NV} 个得分。这几行加起来，再加上一个偏置，就是下一个 token 的 ${NV} 个 logit，softmax 再把它们变成概率。写的时候，它每一步都选概率最高的 token。`);
/* the small model's two formulas, typeset (R6.29): in its hover and in Show the math */
const TOY_ZR='b + \\sum_{j=1}^{8} W_j[x_{t-j}]', TOY_Z='z = '+TOY_ZR, TOY_P='p = softmax(z)';
const TOY_SYM=()=>{const x=A.ui.math('x_{t-j}'), w=A.ui.math('W_j[x_{t-j}]'); return tx(`${x} is the token j places back, ${w} its row in table j, b the bias.`,`${x} 是往前数第 j 个 token，${w} 是第 j 张表里它的那一行，b 是偏置。`);};
/* a formula term: its hover is HTML (a definition, then the typeset formula), a click opens Show the math (CONTRACT: .t, termTip, jump) */
const TT=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const TERM={toy:()=>`<div>${esc(TOY_TIP())}</div><div class="fx">${A.ui.math(TOY_Z)}</div><div class="fx">${A.ui.math(TOY_P)}</div><div>${TOY_SYM()}</div><div class="act">${esc(tx('Click to open “Show the math”.','点击打开“展开公式”。'))}</div>`};
const IGN_TIP=()=>tx('A label of −100 tells PyTorch’s cross-entropy to skip that position: −100 is the default ignore_index of torch.nn.CrossEntropyLoss, and the mean is taken over the positions that remain.',
  'label 是 −100 的位置，PyTorch 的交叉熵会跳过：−100 是 torch.nn.CrossEntropyLoss 默认的 ignore_index，平均只在剩下的位置上取。');
const SP_TIP={
 bot:()=>tx('<|begin_of_text|> marks the start of the whole input. Nothing comes before it, so no position predicts it.','<|begin_of_text|> 标出整段输入的开头。它前面没有别的 token，所以没有哪个位置要预测它。'),
 sh:()=>tx('<|start_header_id|> and <|end_header_id|> enclose the name of who speaks next: user or assistant.','<|start_header_id|> 和 <|end_header_id|> 把接下来说话的人的名字括起来：user 或 assistant。'),
 eh:()=>tx('<|start_header_id|> and <|end_header_id|> enclose the name of who speaks next: user or assistant.','<|start_header_id|> 和 <|end_header_id|> 把接下来说话的人的名字括起来：user 或 assistant。'),
 nl:()=>tx('After each header the template adds two newlines, encoded as their own ordinary token.','每个 header 后面，模板加两个换行，它们单独编码成一个普通 token。'),
 user:()=>tx('The role name inside the header: an ordinary token.','header 里的角色名，是普通 token。'),
 asst:()=>tx('The role name inside the header: an ordinary token.','header 里的角色名，是普通 token。'),
 eot:()=>tx('<|eot_id|> ends a turn. When the model writes it, the inference engine stops: Meta’s documentation says it “signals to the executor that the model has finished generating a response”.','<|eot_id|> 表示这一轮说完了。模型写出它，推理引擎就停下：Meta 的文档说，它告诉执行方模型已经写完了回答。'),
 pad:()=>tx('<|finetune_right_pad_id|> is the token Llama 3.1 reserves for padding sequences to the same length in a batch.','<|finetune_right_pad_id|> 是 Llama 3.1 留给补齐用的 token：一个 batch 里短的序列用它补到一样长。')};

/* ======================= copy ======================= */
const LEV4=[{v:'all',t:{en:'Every token counts',zh:'每个 token 都算'},d:{en:'The question, the template tokens and the answer: every position after the first.',zh:'提问、模板加的 token 和回答，除了第一个 token，每个位置都算。'}},
            {v:'resp',t:{en:'Only the answer counts',zh:'只算回答部分'},d:{en:'The prompt’s labels are −100; the answer and its end token count.',zh:'提问部分的 label 都是 −100；回答和它末尾的结束 token 算。'}}];
const LEV5=[{v:'sep',t:{en:'A separate pad token',zh:'用单独的 pad token'},d:{en:'Padding uses <|finetune_right_pad_id|>; only those positions get −100.',zh:'补齐用 <|finetune_right_pad_id|>，只有这些位置是 −100。'}},
            {v:'eot',t:{en:'The pad token is the end token',zh:'pad token 就是结束 token'},d:{en:'Padding uses <|eot_id|>; every position holding that id gets −100.',zh:'补齐用 <|eot_id|>，凡是这个 id 的位置都设成 −100。'}}];
const PREDS=[{v:'lt10',t:{en:'Under 10%',zh:'不到 10%'}},{v:'10to50',t:{en:'10% to 50%',zh:'10% 到 50%'}},{v:'50to90',t:{en:'50% to 90%',zh:'50% 到 90%'}},{v:'gt90',t:{en:'Over 90%',zh:'超过 90%'}}];
const predOf=sh=>sh<0.1?'lt10':(sh<0.5?'10to50':(sh<0.9?'50to90':'gt90'));
const PRESETS=[
 {t:{en:'Short question, long answer (chat)',zh:'短提问、长回答（聊天）'},d:{en:'The user asks one line; the model answers at length.',zh:'用户问一句，模型答一大段。'},s:{ex:'chat',count:'all',eotl:'yes',pad:'sep',tmpl:'same',steps:60}},
 {t:{en:'A long article, a short summary',zh:'长文章、短摘要'},d:{en:'The user pastes an article; the model writes a few lines.',zh:'用户贴一篇文章，模型写几句摘要。'},s:{ex:'article',count:'all',eotl:'yes',pad:'sep',tmpl:'same',steps:60}},
 {t:{en:'A multi-turn chat: count each assistant turn',zh:'多轮对话：只算每一轮助手的回答'},d:{en:'Three rounds; the user and the assistant take turns.',zh:'三轮对话，用户和助手轮流说。'},s:{ex:'multi',count:'resp',eotl:'yes',pad:'sep',tmpl:'same',steps:60}},
 {t:{en:'The pad token is the end token',zh:'pad token 就是结束 token'},d:{en:'The tokenizer has no pad token, so the script pads with the end token.',zh:'tokenizer 没有 pad token，脚本拿结束 token 补齐。'},s:{ex:'toy',count:'resp',eotl:'yes',pad:'eot',tmpl:'same',steps:60}}];
const presetNow=()=>PRESETS.findIndex(p=>CFG.every(k=>p.s[k]===S[k]));
const EXSHORT={toy:{en:'Small model',zh:'小模型的数据'},chat:{en:'Chat',zh:'聊天'},article:{en:'Article',zh:'文章'},multi:{en:'3 rounds',zh:'三轮'}};
const EXNAME={toy:{en:'The small model’s conversation',zh:'小模型的一段对话'},chat:{en:'Chat',zh:'聊天'},article:{en:'Article and summary',zh:'文章和摘要'},multi:{en:'Three rounds',zh:'三轮对话'}};

/* ======================= guided parts ======================= */
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'A base model only continues the text and never stops.',zh:'基础模型只会接着往下写，不会停。'},
  body:()=>tx(`<p>A model fresh from pretraining, a base model, only continues text. This lab uses a ${TT('toy','small model')} with a vocabulary of ${NV} tokens and ${I(NPAR)} parameters, pretrained on the plain text of a few FAQ pages. Given ${qTxt()} as plain text, it answers once, then writes the user’s next question itself and keeps going (${J('gen','see the chart')}).</p><p>SFT (supervised fine-tuning) teaches it to read a question in the chat format, write only the answer, and stop. It is the first step of post-training: the ${A.shell.labLink('reward-model','reward model')}, ${A.shell.labLink('ppo','PPO')} and ${A.shell.labLink('dpo','DPO')} that come later all start from the model SFT produces.</p>`,
              `<p>预训练出来的模型叫基础模型，它只会接着往下写。这个 Lab 用一个${TT('toy','小模型')}：词表 ${NV} 个 token，${I(NPAR)} 个参数，先在几个问答网页的纯文字上做了预训练。把${qTxt()}当纯文字给它，它答了一句，又替用户写下一个提问，一直写下去（${J('gen','见图')}）。</p><p>SFT（监督微调）要教会它：按聊天的格式读提问，只写回答，写完就停。它是后训练的第一步，后面的${A.shell.labLink('reward-model','奖励模型')}、${A.shell.labLink('ppo','PPO')}、${A.shell.labLink('dpo','DPO')} 都从 SFT 之后的模型开始。</p>`)},
 {id:'template',title:{en:'A conversation becomes tokens',zh:'对话怎么变成 token'},lead:{en:'The chat template marks who speaks and where a turn ends.',zh:'模板用特殊 token 标出谁在说话、在哪里结束。'},
  body:()=>tx(`<p>SFT data is conversations. Before training, each one is laid out with the chat template as one sequence of tokens. ${J('strip','The strip')} shows one example conversation in Llama 3.1’s format:</p><ul class="sft-al"><li>${code('<|begin_of_text|>')}: the start of the input</li><li>${code('<|start_header_id|>')} … ${code('<|end_header_id|>')}: who speaks, user or assistant</li><li>${EOT}: the turn is over; when the model writes it, generation stops</li></ul><p>The special tokens are exactly the ones Llama 3.1 uses. How ordinary words split into tokens is only illustrative here. The base model has never seen these tokens: asked in this format, it rambles on. A model must be used with the template it was trained with.</p>`,
              `<p>SFT 的数据是一段段对话。训练之前，每段对话先按 chat template 拼成一串 token。${J('strip','图里')}是一段示例对话按 Llama 3.1 的格式拼好的样子：</p><ul class="sft-al"><li>${code('<|begin_of_text|>')}：整段输入的开头</li><li>${code('<|start_header_id|>')} … ${code('<|end_header_id|>')}：谁在说话，user 还是 assistant</li><li>${EOT}：这一轮说完了，模型写出它，生成就停下</li></ul><p>特殊 token 和 Llama 3.1 用的一模一样。普通文字怎么切成 token，这里只是示意。基础模型没见过这些特殊 token，按这个格式问它，它照样乱写。模型用的时候，要用和训练时同一个模板。</p>`)},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'A long article, a short summary: how much of the loss is the summary’s?',zh:'长文章、短摘要：摘要占算 loss 的 token 的多少？'},gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PREDS.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>SFT is still next-token prediction. Each counted position adds ${NLP()}, where p is the probability the model gives the correct next token. The loss is the mean over the N counted positions:</p>${EQ('SFT loss',[LOSS_TEX])}<p class="small">${XSYM()}</p><p>One training example is an article and its summary. The prompt part (the article and the template’s tokens) is 1,800 tokens; the response part (the summary and its end token) is 120. If every token counts, what share of the counted tokens is the summary’s?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part.</p>':''}`,
             `<p>SFT 还是下一个 token 预测。每个算 loss 的位置贡献 ${NLP()}，p 是模型给正确的下一个 token 的概率。loss 是这 N 个位置的平均：</p>${EQ('SFT 的 loss',[LOSS_TEX])}<p class="small">${XSYM()}</p><p>一条训练数据是一篇文章加它的摘要。提问部分（文章加上模板的 token）有 1,800 个 token，回答部分（摘要加上结束 token）有 120 个。每个 token 都算 loss 时，算 loss 的 token 里，摘要部分占多少？</p>${ch}${pred?'<p class="muted small">已记下。进入下一部分之前，还可以改。</p>':''}`);},
  obs:()=>obsPredict()},
 {id:'resp',title:{en:'Count the answer only',zh:'只算回答部分'},lead:{en:'Train the same base model twice, counting different tokens.',zh:'同一个基础模型，用两种方式算 loss，各训练一次。'},gate:()=>tried4.has('resp')&&watchedNow(),
  enter:()=>({...DEF,count:'all'}),showMe:()=>setLever4('resp'),
  body:()=>tx(`<p>In the training data every token carries a label, usually the token itself: the position before it must predict it. To count the answer only, set every prompt token’s label to ${DEF_('−100',IGN_TIP())}. PyTorch’s cross-entropy skips those positions and takes the mean over the rest. This happens when the data is prepared, so it costs no extra compute.</p><p>Both ways start from the same base model, train 60 steps on 6 example conversations, then answer ${qTxt()} in the training template. Pick which tokens count, and see how the trained small model answers.</p>${A.ui.levers({aria:tx('Which tokens count','哪些 token 算 loss'),value:S.count,options:LEV4.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:tried4.has(o.v)?esc(lev4Res(o.v)):''}))})}`,
              `<p>训练数据里，每个 token 都配一个 label，通常就是这个 token 本身，意思是前一个位置要预测出它。只算回答部分，就是把提问部分每个 token 的 label 都设成 ${DEF_('−100',IGN_TIP())}。PyTorch 的交叉熵会跳过这些位置，平均只在剩下的位置上取。这一步在准备数据时就做完了，不多花计算。</p><p>两种方式都从同一个基础模型出发，用 6 段示例对话训练 60 步，再按训练时的模板问它${qTxt()}。选一种算 loss 的方式，看看训练出来的小模型怎么回答。</p>${A.ui.levers({aria:tx('Which tokens count','哪些 token 算 loss'),value:S.count,options:LEV4.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:tried4.has(o.v)?esc(lev4Res(o.v)):''}))})}`),
  obs:()=>obsResp()},
 {id:'eot',title:{en:'The end token must count',zh:'结束 token 也要算'},lead:{en:'When the pad token is the end token, the model never learns to stop.',zh:'pad token 就是结束 token 时，模型学不会停。'},gate:()=>tried5.has('eot')&&watchedNow(),
  enter:()=>({...DEF}),showMe:()=>setLever5('eot'),
  body:()=>tx(`<p>Training runs several conversations together as one batch. Their tokens are stacked into one matrix, one conversation per row, so every row must have the same length. Shorter conversations are filled up at the end with a pad token, up to the length of the longest (${J('batch','see the chart')}). The padded positions only hold places, so they must not count in the loss.</p><p>In Hugging Face’s transformers library, the function that puts several conversations into one batch is called a data collator. Its <code>DataCollatorForLanguageModeling</code>, made for language models, sets the label to −100 wherever the token id equals the pad token’s.</p><p>When a tokenizer has no pad token, a training script may pad with the end token instead. The same rule then also sets the real ${EOT} after the answer to −100, so the model never learns to write it. Pick a pad token, and see whether the trained small model stops after its answer.</p>${A.ui.levers({aria:tx('The pad token','pad token'),value:S.pad,options:LEV5.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:tried5.has(o.v)?esc(lev5Res(o.v)):''}))})}`,
              `<p>训练时，几段对话合成一个 batch 一起算。它们的 token 要叠成一个矩阵，每段对话占一行，所以每一行必须一样长。短的对话就在后面补 pad token，补到和最长的一样长（${J('batch','见图')}）。补上的位置只是占位，不该算 loss。</p><p>Hugging Face 的 transformers 库里，把几段对话拼成一个 batch 的函数叫 data collator。给语言模型用的 <code>DataCollatorForLanguageModeling</code> 会找出 token id 等于 pad token 的位置，把这些位置的 label 都设成 −100。</p><p>tokenizer 没有 pad token 时，训练脚本可能拿结束 token 来补。这样一来，同一条规则会把回答末尾真正的 ${EOT} 也设成 −100，模型就学不到要写它。选一种 pad token，看看训练出来的小模型答完停不停。</p>${A.ui.levers({aria:tx('The pad token','pad token'),value:S.pad,options:LEV5.map(o=>({v:o.v,title:L(o.t),desc:L(o.d),result:tried5.has(o.v)?esc(lev5Res(o.v)):''}))})}`),
  obs:()=>obsEot()},
 {id:'dpo',title:{en:'From the SFT loss to DPO',zh:'从 SFT 的 loss 到 DPO'},lead:{en:'The response’s log-probability: DPO uses it too.',zh:'回答的 log 概率：DPO 也用它。'},enter:()=>({...DEF}),
  body:()=>bridgeHTML()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:()=>({...DEF}),
  body:()=>rememberHTML()}];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const isP=id=>mode==='guided'&&STEPS[step].id===id;
const watchedNow=()=>!!PL.sig&&PL.watched.has(PL.sig);

function lev4Res(v){const g=genFor({...DEF,count:v},'sft'), gi=genInfo(g);
  return gi.ok?tx('Right answer, stops','答对了，停了'):(gi.stop?tx(`Answers about ${tokS(gi.first)}, stops`,`答成了 ${tokS(gi.first)}，停了`):tx('Does not stop','没有停'));}
function lev5Res(v){const g=genFor({...DEF,pad:v},'sft'), gi=genInfo(g);
  return gi.stop?tx('Stops','停了'):tx('Does not stop','没有停');}
function obsPredict(){
  if(!pred) return '';
  const c=counts('article','all',false), ok=pred===predOf(c.share), pt=esc(L(PREDS.find(o=>o.v===pred).t)), right=esc(L(PREDS.find(o=>o.v===predOf(c.share)).t));
  return tx(`<p><b>The summary is only ${pcShare(c.share)} of the counted tokens.</b> You predicted “${pt}”${ok?'; that is right.':`; the right answer is “${right}”.`}</p><p>The loss is a mean, so every counted token gets the same weight. Counting the article as well, each summary token’s weight in that mean drops to about 1/${f(c.ratio,0)} of its weight when only the summary counts (${J('math','how that is computed')}). Most counted tokens are the article’s, yet at use time the user pastes the article in, and the model never has to write it.</p>`,
            `<p><b>摘要只占算 loss 的 token 的 ${pcShare(c.share)}。</b>你预测的是“${pt}”${ok?'，预测对了。':`，正确答案是“${right}”。`}</p><p>loss 是平均，每个算 loss 的 token 权重一样。文章也算的话，每个摘要 token 在平均里的权重，降到只算摘要时的约 1/${f(c.ratio,0)}（${J('math','怎么算的')}）。算 loss 的 token 大部分是文章的，可用的时候文章是用户贴进来的，模型不用学着写它。</p>`);}
function obsResp(){
  if(!watchedNow()) return '';
  const g=genFor(S,'sft'), gi=genInfo(g), ct=counts('toy',S.count,masked(S)), other=genInfo(genFor({...S,count:S.count==='all'?'resp':'all'},'sft'));
  if(S.count==='all') return tx(`<p><b>Every token counts: after 60 steps the small model answers about ${tokS(gi.first)}, not KV cache.</b> It does stop after its answer.</p><p>Only ${pcShare(ct.share)} of the counted tokens are the answer’s, so with the same steps and step size the answer gets a small share of every update. ${tried4.has('resp')?'':'Now count the answer only.'}</p>`,
            `<p><b>每个 token 都算：训练 60 步后，小模型答的是 ${tokS(gi.first)}，不是 KV cache。</b>它答完倒是停了。</p><p>算 loss 的 token 里只有 ${pcShare(ct.share)} 是回答的，所以同样的步数和步长下，每次更新里回答只分到一小份。${tried4.has('resp')?'':'再试试只算回答部分。'}</p>`);
  return tx(`<p><b>Only the answer counts: after the same 60 steps the small model answers KV cache and stops at ${EOT}.</b>${other.ok?'':' Counting every token, it still answered about another term.'}</p><p>All the counted tokens are now the answer’s, so every update goes to what the model will have to write. In this lab that is plain gradient descent with the same step size; the effect is the weight in the mean, not a different loss.</p>`,
          `<p><b>只算回答部分：同样训练 60 步，小模型答对了 KV cache，写出 ${EOT} 就停了。</b>${other.ok?'':'每个 token 都算时，它答的还是别的词。'}</p><p>现在算 loss 的 token 全是回答的，每次更新都用在模型要学着写的东西上。在这个 Lab 里用的是普通梯度下降、同样的步长；差别来自平均里的权重，不是换了一种 loss。</p>`);}
function obsEot(){
  if(!watchedNow()) return '';
  const g=genFor(S,'sft'), gi=genInfo(g);
  if(S.pad==='sep') return tx(`<p><b>A separate pad token: the small model answers and stops at ${EOT}.</b> Only the padded positions got −100; the real end token kept its label.</p>`,
                              `<p><b>用单独的 pad token：小模型答完，写出 ${EOT} 就停了。</b>只有补齐的位置是 −100，真正的结束 token 保留了它的 label。</p>`);
  return tx(`<p><b>The pad token is the end token: the small model answers, does not stop, and writes the user’s next question itself.</b> It was still writing after ${MAXG} tokens; the red token in the chart is where it should have stopped.</p><p>Its training never contained “after the answer comes ${EOT}”, so after the full stop it falls back on what pretraining taught it: on an FAQ page a new question follows. Two fixes: pad with a separate token (Llama 3.1 reserves ${PADT}), or set the padded labels to −100 by position, not by token id.</p>`,
          `<p><b>pad token 就是结束 token：小模型答完不停，接着替用户写下一个提问。</b>写满 ${MAXG} 个 token 还没停，图里标红的 token 就是它该停的地方。</p><p>它的训练里从来没有“答完接着写 ${EOT}”这一条，所以写完句号，它就照预训练的习惯来：问答网页上，一个回答后面是下一个提问。修法有两个：用单独的 pad token 补齐（Llama 3.1 留了 ${PADT}）；或者按位置把补齐处的 label 设成 −100，不按 token id。</p>`);}
/* part 6 (Codex R11, round 3): the calculation sits on the stage, not in the narrow rail. The table of the 5 answer tokens and the
   derivation go step by step: probability, log, sum, minus the reference model; each step adds one thing to the table, and the row of
   the current token is highlighted. The rail keeps one short paragraph that says what to look at. */
const BK=[{en:'Probability',zh:'概率'},{en:'Take the log',zh:'取 log'},{en:'Add up',zh:'加起来'},{en:'Minus the reference',zh:'减去参考模型'}];
const tn=(v,d)=>(v<0?'-':'')+f(Math.abs(v),d);   // a signed number inside TeX (the kit draws '-' as a minus sign)
const bridgeData=()=>{const R=respLogp(sftRun('resp',false,60).m), B=respLogp(base()); return {R,B,n:R.rows.length,ratio:R.sum-B.sum};};
function bridgeHTML(){
  return tx(`<p>The chart works out the answer’s log-probability in 4 steps (${J('dpo','see the chart')}). The first is the probability that the small model after SFT gives each answer token. Each click on “Next step” adds one thing to the table: the log of each probability, their sum, then the same sum from the model before SFT, subtracted. The log-ratio at the end is what the loss of ${A.shell.labLink('dpo','DPO')} uses too.</p>`,
            `<p>图里一步一步算回答的 log 概率，一共 4 步（${J('dpo','见图')}）。第一步是 SFT 之后的小模型给回答里每个 token 的概率。之后每点一次“下一步”，表里只多出一样东西：先给每个概率取 log，再把它们加起来，最后减去 SFT 之前的模型算出的同一个和。最后得到的 log 比值，${A.shell.labLink('dpo','DPO')} 的 loss 也要用。</p>`);}
function drawBridge(){
  const box=$('#sft-dpobox',app); if(!box) return;
  const on=isP('dpo'); box.hidden=!on; if(!on){box._h=''; return;}
  const {R,B,n,ratio}=bridgeData(), r=R.rows[brow], rb=B.rows[brow];
  const PI=A.ui.math('\\pi'), PREF=A.ui.math('\\pi_{\\text{ref}}'), LP=A.ui.math('\\log\\, p'), LPI=A.ui.math('\\log\\, \\pi(y|x)');
  const chip=k=>`<span class="sft-dp-tok${isMono(k)?' mono':''}">${esc(tokS(k))}</span>`;
  const cap=tx(`The log-probability of the answer y. x is the question ${qTxt()} in the training template; y is its answer in the training data, ${n} tokens.`,
               `回答 y 的 log 概率。x 是提问${qTxt()}，按训练时的模板；y 是训练数据里它的回答，共 ${n} 个 token。`);
  const sl=tx('Step of the calculation','计算的步骤'), seg=`<div class="sft-dp-steps" id="sft-dp-k"><span class="lbl">${sl}</span><div class="seg-ctl" role="group" aria-label="${esc(sl)}" data-key="bk">${BK.map((o,i)=>`<button type="button" data-v="${i}" aria-pressed="${i===bk}">${i+1} ${esc(L(o))}</button>`).join('')}</div></div>`;
  const nav=`<div class="sft-dp-btns">`+(bk>0?`<button type="button" class="btn ghost sm" data-act="bk-prev">${tx('Previous step','上一步')}</button>`:'')+
    (bk<BK.length-1?`<button type="button" class="btn sm" data-act="bk-next">${tx(`Next step: ${L(BK[bk+1]).toLowerCase()}`,`下一步：${L(BK[bk+1])}`)}</button>`:'')+`</div>`;
  /* the table: a column or a row more at each step; the current token's row highlighted, and at step 3 every log p and the sum */
  const head=`<tr><th>token</th><th>logit</th><th>${tx('probability p','概率 p')}</th>${bk>=1?`<th>${LP}</th>`:''}${bk>=3?`<th>${tx(`${LP} before SFT`,`SFT 之前的 ${LP}`)}</th>`:''}</tr>`;
  const rows=R.rows.map((x,i)=>`<tr class="${i===brow&&bk!==2?'sft-dp-cur':''}"><td><button type="button" class="sft-dp-rb${isMono(x.k)?' mono':''}" data-brow="${i}" aria-pressed="${i===brow}">${esc(tokS(x.k))}</button></td><td class="mono">${f(x.z,2)}</td><td class="mono">${pc2(x.p)}</td>`+
    (bk>=1?`<td class="mono${bk===1&&i===brow||bk===2?' sft-dp-new':''}">${sn(x.lp,3)}</td>`:'')+(bk>=3?`<td class="mono sft-dp-new">${sn(B.rows[i].lp,3)}</td>`:'')+`</tr>`).join('');
  const foot=bk>=2?`<tfoot><tr><td colspan="3">${tx('sum','加起来')}</td><td class="mono${bk===2?' sft-dp-new':''}"><b>${sn(R.sum,3)}</b></td>${bk>=3?`<td class="mono sft-dp-new"><b>${sn(B.sum,2)}</b></td>`:''}</tr></tfoot>`:'';
  const tbl=`<div class="sft-dp-tw"><table class="sft-tbl sft-dp-tbl"><thead>${head}</thead><tbody>${rows}</tbody>${foot}</table></div>`;
  const pick=tx('Click another token in the table to move the highlight to its row.','点表里的另一个 token，高亮就换到那一行。');
  let now;
  if(bk===0) now=tx(`<p><b>Step 1, probability.</b> At every position of the answer the small model gives each of the ${NV} tokens in its vocabulary a logit, and softmax turns the logits into probabilities. p is the probability it gives the answer’s token there, knowing the question x and the answer tokens before it.</p><p>The highlighted row is ${chip(r.k)}: its logit is ${f(r.z,2)}, and after softmax p = ${pc2(r.p)}. ${pick}</p>`,
                    `<p><b>第 1 步，概率。</b>在回答的每个位置，小模型给词表里 ${NV} 个 token 各算一个 logit，softmax 再把这些 logit 变成概率。p 是在提问 x 和前面的回答 token 都已知时，模型给这个位置上回答 token 的概率。</p><p>高亮的这一行是 ${chip(r.k)}：它的 logit 是 ${f(r.z,2)}，softmax 之后 p = ${pc2(r.p)}。${pick}</p>`);
  else if(bk===1) now=tx(`<p><b>Step 2, take the log.</b> Each row’s p goes through the natural log and becomes ${LP}. p is below 1, so ${LP} is negative; the closer p is to 1, the closer ${LP} is to 0.</p><p>The highlighted row is ${chip(r.k)}:</p>${M(`\\log\\, ${f(r.p,4)} = ${tn(r.lp,3)}`)}<p>${pick}</p>`,
                         `<p><b>第 2 步，取 log。</b>每一行的 p 取自然对数，得到 ${LP}。p 小于 1，所以 ${LP} 是负数；p 越接近 1，${LP} 越接近 0。</p><p>高亮的这一行是 ${chip(r.k)}：</p>${M(`\\log\\, ${f(r.p,4)} = ${tn(r.lp,3)}`)}<p>${pick}</p>`);
  else if(bk===2) now=tx(`<p><b>Step 3, add up.</b> The ${LP} of the ${n} answer tokens add up to the log-probability of the whole answer:</p>`,`<p><b>第 3 步，加起来。</b>回答里 ${n} 个 token 的 ${LP} 加起来，就是整段回答的 log 概率：</p>`)+
      EQ(tx('log-probability','回答的 log 概率'),['\\log\\, \\pi(y|x)','\\sum_{t} \\log\\, p(y_t | x, y_{<t})',tn(R.sum,3)])+
      tx(`<p class="small">${PI} is the small model after SFT (only the answer counts, 60 steps); ${A.ui.math('y_t')} is the answer’s token at position t, and ${A.ui.math('y_{<t}')} the answer tokens before it.</p><p>Average the ${LP} of these ${n} tokens and flip the sign: that is the SFT loss on this example, ${f(-R.sum/n,3)}.</p>`,
         `<p class="small">${PI} 是 SFT 之后的小模型（只算回答部分，训练 60 步）；${A.ui.math('y_t')} 是回答在位置 t 的 token，${A.ui.math('y_{<t}')} 是它前面的回答 token。</p><p>把这 ${n} 个 token 的 ${LP} 取平均、再取负，就是这条数据的 SFT loss，等于 ${f(-R.sum/n,3)}。</p>`);
  else now=tx(`<p><b>Step 4, minus the reference model.</b> A reference model is a frozen copy of the model taken when training starts; it is never updated. Here the reference is the small model before SFT, written ${PREF}. The new column is the ${LP} it gives each token of the same answer: for ${chip(r.k)} the reference model gives ${sn(rb.lp,3)}, and the small model after SFT ${sn(r.lp,3)}. The column adds up to ${sn(B.sum,2)}. The difference of the two sums is the log-ratio:</p>`,
              `<p><b>第 4 步，减去参考模型。</b>参考模型是训练开始时模型的一份冻结副本，之后不再更新。这里拿 SFT 之前的小模型当参考模型，记作 ${PREF}。新的一列是它给同一个回答每个 token 的 ${LP}：${chip(r.k)} 这一行，参考模型给的是 ${sn(rb.lp,3)}，SFT 之后的小模型给的是 ${sn(r.lp,3)}。这一列加起来是 ${sn(B.sum,2)}。两个和相减，就是 log 比值：</p>`)+
      EQ(tx('log-ratio','log 比值'),['\\log\\, \\pi(y|x) - \\log\\, \\pi_{\\text{ref}}(y|x)',`${tn(R.sum,3)} - ({${tn(B.sum,2)}})`,tn(ratio,2)])+
      tx(`<p class="small">${PI} is the small model after SFT; ${PREF} is the reference model, the small model before SFT.</p><p>The log-ratio is large: SFT made this answer far more likely. ${A.shell.labLink('dpo','DPO')} uses the same kind of log-ratio in its loss, with two other models. There π is the model being trained by DPO, and the reference model is the frozen model after SFT. DPO takes two answers to one question, a chosen and a rejected one, and raises the chosen answer’s log-ratio above the rejected one’s.</p>`,
         `<p class="small">${PI} 是 SFT 之后的小模型；${PREF} 是参考模型，也就是 SFT 之前的小模型。</p><p>log 比值很大，说明 SFT 让这个回答的概率大了很多。${A.shell.labLink('dpo','DPO')} 的 loss 也用这种 log 比值，只是换了两个模型。在 DPO 里，π 是正在用 DPO 训练的模型，参考模型是冻结的 SFT 之后的模型。DPO 拿同一个提问的两个回答，一个选中、一个落选，让选中回答的 log 比值比落选回答的大。</p>`);
  const h=`<div class="sft-cap">${cap}</div><div class="sft-dp-bar">${seg}${nav}</div>${tbl}<div class="sft-dp-now" aria-live="polite">${now}</div>`;
  if(box._h!==h+A.lang()){box.innerHTML=h; box._h=h+A.lang();}}
function setBk(v){bk=Math.max(0,Math.min(BK.length-1,v|0)); render(); const b=$('#sft-dpobox [data-act="bk-next"]',app)||$('#sft-dpobox [data-act="bk-prev"]',app); if(b&&document.activeElement===document.body) b.focus({preventScroll:true});}
function setBrow(v){brow=Math.max(0,Math.min(4,v|0)); render(); const b=$(`#sft-dpobox [data-brow="${brow}"]`,app); if(b) b.focus({preventScroll:true});}
/* the recap states each rule and what follows from it, not a detail of an earlier part (the owner's review of batch 9, item 3; G2) */
function rememberHTML(){
  return tx(`<ul class="sft-rem"><li><b>SFT is next-token prediction on question-and-answer data, with only the answer counting.</b> If every token counts and the prompt is long, the answer gets only a small share of the loss. Most of the training then goes to a prompt the model never has to write, and it learns its answers more slowly.</li><li><b>The end token after the answer must count.</b> Once its label is −100, the model never learns to stop, and it keeps writing after its answer. So pad with a separate pad token, or set the padded labels to −100 by position.</li><li><b>Train and serve with the same template.</b> The model learned the format it was trained with. Asked in another format, it cannot tell where its answer starts or where to stop.</li></ul><p class="small">Counting the prompt is not always worse: Shi et al. (2024, arXiv 2405.14394) found that counting the prompt helps especially when the prompts are long and the answers short, or when there are few training examples.</p><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “SFT is a new kind of loss.” It is the same next-token cross-entropy; what changes is the data and which tokens count.</p><p>✗ “Counting the prompt too makes the model learn more.” Counting the prompt dilutes the answer’s weight in the mean, and the extra training goes to text the model never has to write at use time. Whether that helps depends on the data, as Shi et al. found above.</p><p>✗ “The end token takes care of itself.” If it is not in the labels, the model never learns to stop.</p></div>${A.shell.toScenarios()}`,
            `<ul class="sft-rem"><li><b>SFT 还是下一个 token 预测，只是数据换成一问一答，并且只算回答部分。</b>如果每个 token 都算 loss，提问一长，回答在 loss 里就只占一小份。训练的力气大多花在模型用不着写的提问上，回答反而学得慢。</li><li><b>回答末尾的结束 token 也要算。</b>它的 label 一旦成了 −100，模型就学不会停，答完还会接着往下写。所以补齐时要用单独的 pad token，或者按位置把补齐处设成 −100。</li><li><b>训练和使用要用同一个模板。</b>模型学会的是训练时的格式。换一种格式问它，它认不出该从哪里开始答、在哪里停。</li></ul><p class="small">提问也算 loss 并不总是更差：Shi 等（2024，arXiv 2405.14394）发现，提问很长、回答很短，或者训练样本很少时，提问也算 loss 尤其有帮助。</p><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “SFT 是一种新的 loss。”不是：还是下一个 token 的交叉熵，换的是数据和算哪些 token。</p><p>✗ “提问也算 loss，模型学得更多。”提问算进来，回答在平均里的权重就被摊薄了，多出来的那部分训练，学的是用的时候不用写的文字。到底有没有好处，要看数据，见上面 Shi 等人的结果。</p><p>✗ “结束 token 不用管。”它不在 label 里，模型就学不会停。</p></div>${A.shell.toScenarios()}`);}

/* ======================= views ======================= */
const DESCS=()=>({
  guided:tx('Seven short parts. See why a base model does not stop, lay a conversation out as tokens, predict what share of the counted tokens a short answer gets, train a small model three ways, and connect the SFT loss to DPO.','一共 7 个部分。先看基础模型为什么停不下来，再把对话拼成 token、预测短回答在算 loss 的 token 里占几成，用三种方式训练小模型，最后接到 DPO。'),
  sandbox:tx('Pick an example and see which tokens count and what share of the counted tokens the answer gets. The small model in the chart always trains on its own 6 example conversations, with the settings you pick.','选一个例子，看哪些 token 算 loss、回答部分占几成。图里的小模型始终用它自己的 6 段示例对话，按你选的设置训练。'),
  challenge:tx('A fine-tuned model does not stop after its answer. Change the fewest of four settings so that it stops and at least 90% of the counted tokens are the answer’s.','一个 SFT 之后的模型答完不停。只改 4 个设置里最少的几个，让它答完就停，并且算 loss 的 token 至少 90% 是回答部分的。')});
const OUTCOME=()=>tx('explain which tokens the SFT loss should count, compute the answer’s share of it, and find why a model does not stop.','讲清 SFT 的 loss 该算哪些 token，算出回答部分的占比，并查出模型答完不停的原因。');
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> your browser trains a small model with real gradient steps.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>A small model with a vocabulary of ${NV} tokens and ${I(NPAR)} parameters. The logits for the next token are a bias plus one learned row per earlier token, for the 8 tokens before it.</li><li>Pretraining: 300 full-batch gradient steps on four plain-text FAQ pages (the same six questions and answers in different orders). SFT: full-batch gradient descent, step size 1, mean cross-entropy over the counted positions, on 6 example conversations in the Llama 3.1 format.</li><li>Writing: the most likely token at every step, until ${EOT} or ${MAXG} tokens. No random numbers anywhere, so the same settings always give the same run.</li><li>Token counts: what share of all counted tokens one part’s tokens are. Nothing comes before the first token, so no position predicts it.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>In this lab, ordinary words and phrases are single tokens; Llama 3’s tokenizer splits text into more tokens. The special tokens are exact. The 1,800 and 120 tokens of the article example and the other counts are examples written for this lab.</li><li>The small model does not pad: padded positions carry no loss either way. It only applies the rule “label −100 wherever the token is the pad token”, so with the end token as pad the real end token loses its label.</li><li>Plain gradient descent at one step size. Real runs use AdamW, which rescales updates, so how much counting the prompt slows the answer differs; the share of the counted tokens does not.</li><li>Real SFT learning rates are small. Stanford Alpaca 7B used 2e-5 for 3 epochs (3 passes over the data). AI2’s Tülu 3 8B SFT script, on Llama‑3.1‑8B, uses 5e-6 for 2 epochs. This small model’s step size, its learning rate, is 1 only so that it learns a difference within a few dozen steps; it is not comparable with real training.</li></ul></div><div><div class="eyebrow">Sources</div><ul><li>Llama 3.1 format and special tokens: ${cite(SRC.fmt,'prompt_format.md')}, ${cite(SRC.chat,'chat_format.py')} (llama-models, pinned).</li><li>ignore_index = −100 and the mean over non-ignored targets: ${cite(SRC.torch,'torch/nn/modules/loss.py')}.</li><li>Padding labels set by token id: ${cite(SRC.hf,'transformers data_collator.py')} (v4.57.6). Set by position: ${cite(SRC.trlT,'trl/trainer/sft_trainer.py')} (TRL v1.13.0).</li><li>Learning rates and epochs: ${cite(SRC.alpacaR,'Stanford Alpaca README')}, ${cite(SRC.tulu,'open-instruct finetune_8b.sh')} (pinned).</li><li>Shi et al. 2024, arXiv 2405.14394, as summarised in its repository’s README.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-10-05. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>页面在你的浏览器里，用真的梯度下降训练一个小模型。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>小模型：词表 ${NV} 个 token，${I(NPAR)} 个参数。下一个 token 的 logit 等于一个偏置，加上前面 8 个 token 各自查到的一行。</li><li>预训练：在四个纯文字的问答网页上（同样六问六答，顺序不同）做 300 步全批量梯度下降。SFT：在 6 段按 Llama 3.1 格式拼好的示例对话上做全批量梯度下降，步长 1，loss 是算 loss 的位置上交叉熵的平均。</li><li>生成：每一步选概率最高的 token，直到写出 ${EOT} 或写满 ${MAXG} 个。整个 Lab 不用随机数，同样的设置总是同样的结果。</li><li>token 计数：看某一部分的 token 占全部算 loss 的 token 的几成。第一个 token 前面没有别的 token，没有哪个位置预测它。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>普通的词和短语在这里各算一个 token；Llama 3 的 tokenizer 会切得更细。特殊 token 是真实的。文章例子的 1,800 和 120 个 token 以及其他计数，是为这个 Lab 写的示例。</li><li>小模型训练时不真的补齐：补齐的位置本来就不算 loss。它只照搬“凡是 pad token 的位置 label 设成 −100”这条规则，所以拿结束 token 当 pad 时，真正的结束 token 也丢了 label。</li><li>用的是固定步长的普通梯度下降。真实训练用 AdamW，它会重新缩放每一步，所以提问也算时回答学得慢多少会不同；回答在算 loss 的 token 里占多少，不会变。</li><li>真实 SFT 的学习率很小。Stanford Alpaca 7B 的学习率是 2e-5，训练 3 个 epoch，也就是把数据过 3 遍。AI2 在 Llama‑3.1‑8B 上做 Tülu 3 8B 的 SFT，脚本里的学习率是 5e-6，训练 2 个 epoch。这个小模型的步长（也就是学习率）是 1，只是为了让它在几十步里学出差别，和真实训练没有可比性。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>Llama 3.1 的格式和特殊 token：${cite(SRC.fmt,'prompt_format.md')}、${cite(SRC.chat,'chat_format.py')}（llama-models，固定版本）。</li><li>ignore_index = −100，平均只在没被忽略的位置上取：${cite(SRC.torch,'torch/nn/modules/loss.py')}。</li><li>按 token id 把补齐处的 label 设成 −100：${cite(SRC.hf,'transformers 的 data_collator.py')}（v4.57.6）。按位置设：${cite(SRC.trlT,'trl/trainer/sft_trainer.py')}（TRL v1.13.0）。</li><li>学习率和 epoch 数：${cite(SRC.alpacaR,'Stanford Alpaca 的 README')}、${cite(SRC.tulu,'open-instruct 的 finetune_8b.sh')}（固定版本）。</li><li>Shi 等 2024，arXiv 2405.14394，依据它代码仓库 README 里的摘要。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-10-05。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A language model reads the tokens so far, gives every token in its vocabulary a logit, and softmax turns the logits into probabilities for the next token (see ${A.shell.labLink('sampling')}). Training raises the probability of the token that actually comes next: each position’s loss is ${A.ui.math('-\\log\\, p')} of it, averaged over the positions. Logs and probabilities: ${A.shell.labLink('softmax-numerics')}.</p>`,
                     `<p>语言模型读进已有的 token，给词表里每个 token 一个 logit，softmax 把它们变成下一个 token 的概率（见 ${A.shell.labLink('sampling')}）。训练就是提高真正的下一个 token 的概率：每个位置的 loss 是它的 ${A.ui.math('-\\log\\, p')}，再对所有位置取平均。概率和 log 见 ${A.shell.labLink('softmax-numerics')}。</p>`);
function headerHTML(){
  return A.shell.header({id:ID,world:'post',outcome:esc(OUTCOME()),short:'SFT',title:tx('Why does SFT count the loss on the answer only?','SFT 为什么只算回答部分的 loss？'),
    minutes:12,level:tx('Core','核心'),fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: next-token prediction','前置知识：下一个 token 预测'),primerPop:PRIPOP(),descs:DESCS(),mode});}
function stageHTML(){
  return `<div class="panel stage sft-stage" id="sft-stage">`+
   `<div class="stage-head"><div class="sft-st-title" id="sft-st-title"></div><div class="stage-actions"><span class="status" id="sft-status" tabindex="0"></span></div></div>`+
   `<div class="stage-graph graph sft-graph" id="sft-graph">`+
    `<div class="sft-box" id="sft-stripbox"><div class="sft-scope" data-scope="cnt" hidden></div><div class="sft-cap" id="sft-strip-cap"></div><div class="sft-strip" id="sft-strip"></div><div class="sft-leg" id="sft-strip-leg"></div><div class="sft-batch" id="sft-batch" hidden></div></div>`+
    `<div class="sft-box" id="sft-genbox"><div class="sft-scope sft-scope-m" data-scope="model" hidden></div><div class="sft-cap" id="sft-gen-cap"></div><div class="sft-player" id="sft-player"></div><div class="sft-gen" id="sft-gen"></div><div class="sft-leg" id="sft-gen-leg"></div></div>`+
    `<div class="sft-box" id="sft-sharebox"><div class="sft-scope" data-scope="share" hidden></div><div class="sft-cap" id="sft-share-cap"></div><div class="sft-share" id="sft-sharebar"></div></div>`+
    `<div class="sft-box sft-dp" id="sft-dpobox" hidden></div>`+
   `</div>`+
   `<div class="metrics sft-metrics" id="sft-metrics"></div>`+
   `<details class="math" id="sft-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="sft-mathbody"></div></details></div>`;}
const segC=(id,key,label,opts,note)=>A.ui.seg({id,key,label,options:opts,value:String(S[key]),note});
function controlsHTML(){
  return `<aside class="panel controls" id="sft-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div><p class="ctl-lock" id="sft-lock" hidden></p>`+
   `<div class="lockable">`+segC('sft-ctl-ex','ex',tx('Example shown in the strip','图里显示的例子'),Object.keys(EXSHORT).map(k=>[k,L(EXSHORT[k])]),tx('The small model’s conversation is its real training data; the other three are token counts written for this lab.','小模型的那段对话就是它的训练数据；另外三个是为这个 Lab 写的 token 计数。'))+`</div>`+
   `<div class="lockable">`+segC('sft-ctl-count','count',tx('Which tokens count','哪些 token 算 loss'),LEV4.map(o=>[o.v,L(o.t)]),tx('Only the answer: every prompt label is −100 and the mean is over the answer’s tokens.','只算回答部分：提问部分的 label 都是 −100，平均只在回答的 token 上取。'))+`</div>`+
   `<div class="lockable">`+segC('sft-ctl-eotl','eotl',tx('The end token after the answer','回答末尾的结束 token'),[['yes',tx('Counts','算 loss')],['no',tx('Label −100','label 设成 −100')]],'')+`</div>`+
   `<div class="lockable">`+segC('sft-ctl-pad','pad',tx('The pad token','pad token'),[['sep',tx('Separate pad token','单独的 pad token')],['eot',tx('The end token','就是结束 token')]],tx('Padded labels are set to −100 by token id, so the end token as pad loses its own label too.','补齐处的 label 按 token id 设成 −100，所以拿结束 token 当 pad，它自己的 label 也丢了。'))+`</div>`+
   `<div class="lockable">`+segC('sft-ctl-tmpl','tmpl',tx('Template when the model is used','使用时的模板'),[['same',tx('Same as training','和训练时一样')],['diff',tx('Plain text, no special tokens','纯文字，没有特殊 token')]],tx('Plain text: each turn starts with “User” or “Assistant” written out, with no special tokens, so there is no end token to stop at either.','纯文字：每一轮前面只写“用户”或“助手”，没有特殊 token，所以也没有可以停下的结束 token。'))+`</div>`+
   `<div class="lockable">`+segC('sft-ctl-steps','steps',tx('SFT training steps','SFT 训练步数'),STEPS_OPT.map(v=>[String(v),String(v)]),tx('Full-batch gradient steps from the pretrained small model. The step size 1 is chosen for this small model. Real SFT uses AdamW with a far smaller learning rate, for example 2e-5 for Alpaca and 5e-6 for Tülu 3 8B.','从预训练好的小模型开始做全批量梯度下降。步长 1 是为这个小模型选的。真实的 SFT 用 AdamW，学习率小得多，比如 Alpaca 用 2e-5，Tülu 3 8B 用 5e-6。'))+`</div>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import torch
import torch.nn.functional as F

IGNORE = -100  # the default ignore_index of cross_entropy

def build(prompt_ids, response_ids):
    # response_ids already ends with <|eot_id|>: keep that label
    ids = prompt_ids + response_ids
    labels = [IGNORE] * len(prompt_ids) + response_ids
    return ids, labels

def pad_batch(batch, pad_id):
    # mask the padding by position, never by token id
    n = max(len(ids) for ids, _ in batch)
    X, Y = [], []
    for ids, labels in batch:
        k = n - len(ids)
        X.append(ids + [pad_id] * k)
        Y.append(labels + [IGNORE] * k)
    return torch.tensor(X), torch.tensor(Y)

def sft_loss(logits, labels):
    # position t predicts token t + 1
    v = logits.size(-1)
    logits = logits[:, :-1].reshape(-1, v)
    labels = labels[:, 1:].reshape(-1)
    # the mean over positions whose label is not -100
    return F.cross_entropy(logits, labels, ignore_index=IGNORE)`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="sft-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="sft-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('Labels for SFT: −100 on the prompt, the answer and its end token kept, padding masked by position.','SFT 的 label：提问部分是 −100，回答和它末尾的结束 token 保留，补齐的位置按位置设成 −100。')}</p><pre class="code" id="sft-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="sft-prod">`+
   tx(`<li>${cite(SRC.trl,'<code>trl/trainer/sft_config.py</code>')} (TRL v1.13.0): <code>assistant_only_loss</code> defaults to <code>False</code> (loss on the whole sequence); <code>True</code> computes it on the assistant responses only, which needs a chat template that marks them. <code>completion_only_loss</code> defaults to <code>None</code>: the completion only, for prompt-completion datasets.</li>`+
      `<li>${cite(SRC.alpaca,'Stanford Alpaca <code>train.py</code>')}: <code>IGNORE_INDEX = -100</code> and <code>label[:source_len] = IGNORE_INDEX</code> mask the prompt; the model was fine-tuned on ${cite(SRC.alpacaR,'52K')} instruction-following examples.</li>`+
      `<li>${cite(SRC.hf,'transformers <code>DataCollatorForLanguageModeling</code>')}: <code>labels[labels == self.tokenizer.pad_token_id] = -100</code>, by token id, so a pad token equal to the end token masks the end token as well.</li>`+
      `<li>${cite(SRC.trlT,'TRL’s <code>trl/trainer/sft_trainer.py</code>')} (v1.13.0): when the tokenizer has no pad token, <code>SFTTrainer</code> also pads with the end token (<code>pad_token = … or self._tokenizer.eos_token</code>). Its own collator, though, pads the labels by position with <code>padding_value=-100</code>, so the end token after the answer keeps its label.</li>`+
      `<li>Meta’s ${cite(SRC.card,'Llama 3.1 model card')}: “The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF)”; its prompt format reserves ${PADT} for padding.</li><li>Not every library masks the prompt by default. ${cite(SRC.tt,'torchtune’s Alpaca dataset builder')} has <code>train_on_input</code> default to <code>True</code>; set it to <code>False</code> and the prompt’s labels become −100.</li>`,
      `<li>${cite(SRC.trl,'<code>trl/trainer/sft_config.py</code>')}（TRL v1.13.0）：<code>assistant_only_loss</code> 默认 <code>False</code>（整段都算 loss）；设成 <code>True</code> 就只算助手的回答，前提是 chat template 标出了回答在哪。<code>completion_only_loss</code> 默认 <code>None</code>：prompt-completion 格式的数据只算 completion。</li>`+
      `<li>${cite(SRC.alpaca,'Stanford Alpaca 的 <code>train.py</code>')}：<code>IGNORE_INDEX = -100</code>，<code>label[:source_len] = IGNORE_INDEX</code> 把提问部分设成 −100；模型在 ${cite(SRC.alpacaR,'52K')} 条指令数据上微调。</li>`+
      `<li>${cite(SRC.hf,'transformers 的 <code>DataCollatorForLanguageModeling</code>')}：<code>labels[labels == self.tokenizer.pad_token_id] = -100</code>，按 token id 找，所以 pad token 和结束 token 是同一个时，结束 token 也被设成 −100。</li>`+
      `<li>${cite(SRC.trlT,'TRL 的 <code>trl/trainer/sft_trainer.py</code>')}（v1.13.0）：tokenizer 没有 pad token 时，<code>SFTTrainer</code> 也拿结束 token 补齐（<code>pad_token = … or self._tokenizer.eos_token</code>）。不过它自己的 collator 按位置补 label，补齐处用 <code>padding_value=-100</code>，回答末尾的结束 token 不受影响。</li>`+
      `<li>Meta 的 ${cite(SRC.card,'Llama 3.1 模型卡')}：微调过的版本用了监督微调（SFT）和基于人类反馈的强化学习（RLHF）；它的提示格式留了 ${PADT} 专门用来补齐。</li><li>不是每个库都默认只算回答。${cite(SRC.tt,'torchtune 的 Alpaca 数据集构建函数')}里，<code>train_on_input</code> 默认是 <code>True</code>；设成 <code>False</code>，提问部分的 label 才变成 −100。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="${SRC.igpt}" target="_blank" rel="noopener">Training language models to follow instructions with human feedback</a><span>${tx('Ouyang et al., 2022 (InstructGPT): SFT on demonstrations, then a reward model and PPO','Ouyang 等，2022（InstructGPT）：先在示范数据上 SFT，再训练奖励模型、用 PPO')}</span></li>`+
   `<li><a href="${SRC.shi}" target="_blank" rel="noopener">Instruction Tuning With Loss Over Instructions</a><span>${tx('Shi et al., 2024: counting the prompt can help with long prompts and short answers, or few examples','Shi 等，2024：提问很长、回答很短，或样本很少时，提问也算 loss 可能更好')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="sft-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="sft-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="sft-app m2" id="sft-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="sft-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ======================= rail: scenarios, challenge ======================= */
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3><p class="small">${tx('The chart shows two kinds of data. The boxes marked “Counts for the current scenario” count the scenario’s example. The small model in the chart is always trained on the same 6 example conversations, so a scenario changes its settings but not its training data.','图里有两种数据。标着“当前场景的计数”的两块，数的是这个场景的例子。图里的小模型始终用同样的 6 段示例对话训练，换场景会换它的设置，不会换它的训练数据。')}</p><p class="small">${tx('Pick a scenario and see what share of the counted tokens the answer gets.','选一个场景，看看回答部分在算 loss 的 token 里占多少。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="sft-insight" aria-live="polite"></div><p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
/* in a sentence: which example the strip shows (the small model in the chart always trains on its own 6 conversations, review S1) */
const EXIN={toy:{en:'In the small model’s conversation',zh:'小模型的这段训练对话里'},chat:{en:'In the chat example',zh:'“聊天”这个例子里'},article:{en:'In the article-and-summary example',zh:'“文章和摘要”这个例子里'},multi:{en:'In the three-round example',zh:'“三轮对话”这个例子里'}};
function insightHTML(){
  const i=presetNow(), ct=counts(S.ex,S.count,masked(S)), g=genFor(S,'sft'), gi=genInfo(g), toy=counts('toy',S.count,masked(S));
  const share=tx(`<p><b>${L(EXIN[S.ex])}, with ${S.count==='all'?'every token counting':'only the answer counting'}, the answer is ${pcShare(ct.share)} of the counted tokens.</b></p>`,
                 `<p><b>${L(EXIN[S.ex])}，${S.count==='all'?'每个 token 都算':'只算回答部分'}时，回答部分占算 loss 的 token 的 ${pcShare(ct.share)}。</b></p>`);
  const res=gi.stop?(gi.ok?tx('answers KV cache and stops','答对了 KV cache，然后停了'):tx(`answers about ${tokS(gi.first)} and stops`,`答成了 ${tokS(gi.first)}，然后停了`))
    :tx(`does not stop after its answer${gi.wroteHeader?': it writes the next user header itself':(gi.wroteQ?': it writes the user’s next question itself':'')}`,`答完没有停${gi.wroteHeader?'，自己写起了下一轮 user 的 header':(gi.wroteQ?'，替用户写起了下一个提问':'')}`);
  const data=S.ex==='toy'?tx('The small model in the chart is trained on 6 example conversations like this one.','图里的小模型就是用这样的 6 段示例对话训练的。')
    :tx(`The small model in the chart always trains on its own 6 example conversations, with the same settings.${S.count==='all'?` There the answer is only ${pcShare(toy.share)} of the counted tokens.`:''}`,`图里的小模型始终用它自己的 6 段示例对话训练，设置和这里相同。${S.count==='all'?`在那 6 段对话里，回答只占算 loss 的 token 的 ${pcShare(toy.share)}。`:''}`);
  const gen=watchedNow()?`<p>${data}${tx(' ','')}${tx(`After ${S.steps} steps it ${res}.`,`训练 ${S.steps} 步后，它${res}。`)}</p>`:'';
  let why='';
  if(i===0) why=tx('<p class="small">In chat data the question is short. Counting it too, most counted tokens are still the answer’s, and the answer loses only a little weight.</p>','<p class="small">聊天数据的提问很短。把提问也算上，算 loss 的 token 大部分仍是回答，回答的权重只少了一点。</p>');
  else if(i===1) why=tx(`<p class="small">The article dominates: each summary token’s weight in the mean drops to about 1/${f(ct.ratio,0)} of its weight when only the summary counts. Yet this long-prompt, short-answer case is where Shi et al. found that counting the prompt can help.</p>`,`<p class="small">文章占了大头。每个摘要 token 在平均里的权重，降到只算摘要时的约 1/${f(ct.ratio,0)}。不过 Shi 等人发现，正是这种提问长、回答短的数据，提问也算 loss 可能更好。</p>`);
  else if(i===2) why=tx('<p class="small">The labels are set turn by turn. Every assistant turn and its end token count; every user turn and header gets −100. Masking “everything before the last answer” would drop the first two answers.</p>','<p class="small">label 要一轮一轮地设。每一轮助手的回答和它的结束 token 都算 loss，每一轮用户的话和 header 都设成 −100。如果只把“最后一个回答之前的全部”设成 −100，前两轮的回答就白白丢了。</p>');
  else if(i===3) why=tx(`<p class="small">With the end token as pad, the real ${EOT} after the answer is −100 as well (red in the strip).</p>`,`<p class="small">拿结束 token 当 pad，回答末尾真正的 ${EOT} 也成了 −100（图里标红）。</p>`);
  else why=tx('<p class="small">The current settings match none of the scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>');
  return share+gen+why;}
const CHNAME={count:{all:{en:'every token counts',zh:'每个 token 都算'},resp:{en:'only the answer counts',zh:'只算回答部分'}},eotl:{yes:{en:'end token counts',zh:'结束 token 算'},no:{en:'end token −100',zh:'结束 token 是 −100'}},
  pad:{eot:{en:'pad = end token',zh:'pad 就是结束 token'},sep:{en:'separate pad',zh:'单独的 pad'}},tmpl:{same:{en:'same template',zh:'同一个模板'},diff:{en:'plain-text template',zh:'纯文字模板'}}};
const chLabel=c=>CH_KEYS.map(k=>L(CHNAME[k][c[k]])).join(tx(', ','，'));
function challengeHTML(){
  const seg=(k,label,opts,note)=>A.ui.seg({id:'sft-ch-'+k,key:k,label,options:opts,value:S[k],inline:true,note});
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('After you submit, you see whether both goals hold, and why.','提交后告诉你两个目标是否都达到，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Make the fine-tuned small model stop after its answer, with at least 90% of the counted tokens from the answer','让 SFT 后的小模型答完就停，并且算 loss 的 token 至少 90% 是回答部分的')}</h3>`+
   `<p>${tx(`A colleague fine-tuned the small model on the 6 example conversations for 150 steps. Asked ${qTxt()}, it answers and then does not stop: it writes the next user header and a question itself. Some of these four settings are wrong. Change the fewest so that the model stops after its answer and at least 90% of the counted tokens are the answer’s.`,`同事用 6 段示例对话把小模型 SFT 了 150 步。问它${qTxt()}，它答完不停，自己写起了下一轮 user 的 header 和提问。下面 4 个设置里有错的。改最少的设置，让模型答完就停，并且算 loss 的 token 至少 90% 是回答部分的。`)}</p>`+
   seg('count',tx('Which tokens count','哪些 token 算 loss'),LEV4.map(o=>[o.v,L(o.t)]))+
   seg('eotl',tx('The end token after the answer','回答末尾的结束 token'),[['yes',tx('Counts','算 loss')],['no',tx('Label −100','label 设成 −100')]])+
   seg('pad',tx('The pad token (padded labels are set to −100 by token id)','pad token（补齐处的 label 按 token id 设成 −100）'),[['eot',tx('The end token','就是结束 token')],['sep',tx('Separate pad token','单独的 pad token')]])+
   seg('tmpl',tx('Template when the model is used','使用时的模板'),[['same',tx('Same as training','和训练时一样')],['diff',tx('Plain text, no special tokens','纯文字，没有特殊 token')]],tx('Plain text: each turn starts with “User” or “Assistant” written out, with no special tokens.','纯文字：每一轮前面只写“用户”或“助手”，没有特殊 token。'))+
   `<div class="goal mono" id="sft-ch-stop"><span>${tx('Stops after its answer','答完就停')}</span><b id="sft-ch-sv"></b></div>`+
   `<div class="goal mono" id="sft-ch-share"><span>${tx('Answer’s share of the counted tokens','回答部分占算 loss 的 token')}</span><span><b id="sft-ch-hv"></b>${tx(' (goal: at least 90%)','（目标：至少 90%）')}</span></div><div class="progress" id="sft-ch-hp" aria-hidden="true"><i></i></div>`+
   submit+`<div id="sft-ch-res" class="sft-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})), best=chMin();
  let h;
  if(me.ok&&me.changes<=best) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: the model stops after its answer, and ${pcShare(me.share)} of the counted tokens are the answer’s.</b> You changed ${me.changes} settings, the fewest that work.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标：模型答完就停，算 loss 的 token 里 ${pcShare(me.share)} 是回答部分的。</b>你改了 ${me.changes} 个设置，已经是最少的。</p>`);
  else if(me.ok) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met.</b> You changed ${me.changes} settings; ${best} would do.</p>`,`<p class="okline" style="margin-top:12px"><b>达成目标。</b>你改了 ${me.changes} 个设置，其实 ${best} 个就够。</p>`);
  else {h=!me.stop?tx(`<p class="badline" style="margin-top:12px"><b>Not yet: the model still does not stop after its answer.</b></p><p class="small">Hint: which label tells the model that ${EOT} comes after the answer, and what could remove it without you asking?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：模型答完还是不停。</b></p><p class="small">提示：哪个 label 告诉模型回答后面是 ${EOT}？什么会在你没注意时把它去掉？</p>`)
      :tx(`<p class="badline" style="margin-top:12px"><b>Not yet: the model stops, but only ${pcShare(me.share)} of the counted tokens are the answer’s.</b></p><p class="small">Hint: which tokens does the loss still count?</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：模型停了，但算 loss 的 token 里只有 ${pcShare(me.share)} 是回答部分的。</b></p><p class="small">提示：loss 现在还算了哪些 token？</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!(chReveal||me.ok)) return h;
  const all=chAll(), rows=all.map(x=>({label:chLabel(x.c),value:x.share*100,text:pcShare(x.share)+(x.stop?tx(', stops','，停了'):tx(', no stop','，不停')),ok:x.ok,me:CH_KEYS.every(k=>x.c[k]===S[k]),
    tags:(CH_KEYS.every(k=>x.c[k]===S[k])?[{text:tx('your pick','你的选择'),me:true}]:[]).concat(x.changes===0?[{text:tx('initial setup','初始设置')}]:[])}));
  /* brief (the owner's review of batch 8, 2026-10-07): only the combinations that meet both goals, your pick and the initial setup;
     the note under the chart counts the rest */
  const shown=rows.filter(r=>r.ok||r.me||r.tags.length);
  h+=A.ui.cmpChart({brief:true,title:tx('Answer’s share of the counted tokens, and whether the small model stops after 150 steps of SFT (goal: stops, and at least 90%)','回答部分占算 loss 的 token 的比例，以及小模型 SFT 150 步后停不停（目标：停，并且至少 90%）'),
    aria:shown.map(r=>r.label+tx(': ','：')+r.text).join(tx('; ','；')),rows,goal:90,goalLabel:tx('goal: 90%','目标 90%'),max:100,okText:tx('stops and at least 90%','停了，并且至少 90%'),noText:tx('misses at least one','至少差一个'),goalText:tx('goal: at least 90%','目标：至少 90%'),
    hiddenText:n=>tx(`The other ${n} combinations miss at least one goal and are not shown.`,`其余 ${n} 种组合至少有一个目标没达到，图里没有列出。`)});
  h+=tx(`<p class="small">Two things were wrong. The initial setup counts every token, so the answer is only ${pcShare(counts('toy','all','id').share)} of the counted tokens. Its pad token is also the end token, so the padding rule set every ${EOT} label to −100 by token id, and the model never learned to stop. Counting only the answer and padding with a separate token fixes both. The end token’s own setting and the template were right; changing either of them breaks stopping again.</p><p class="small muted">Note: this checks one question to the small model. On a real model, check that it stops on held-out prompts, and that the inference engine treats ${EOT} as a stop token.</p>`,
        `<p class="small">原因有两个。初始设置每个 token 都算，回答只占算 loss 的 token 的 ${pcShare(counts('toy','all','id').share)}。它还拿结束 token 当 pad，补齐的规则按 token id 把每个 ${EOT} 的 label 都设成了 −100，所以模型从没学过停。改成只算回答部分、用单独的 pad token，两个问题就都解决了。结束 token 那一项和模板本来就是对的，改动其中任何一个，模型又会停不下来。</p><p class="small muted">注意：这里只问了小模型一个问题。真实模型要在没训练过的提问上检查它停不停，并确认推理引擎把 ${EOT} 当作停止的 token。</p>`);
  return h;}

/* ======================= stage drawing ======================= */
function stripParts(){
  const showLab=!(mode==='guided'&&step<SIX('predict'));
  let ex=S.ex; if(mode==='guided'){const id=STEPS[step].id; ex=id==='predict'?'article':'toy'; if(id==='mission') ex='plain';}
  return {ex,showLab,showPad:ex==='toy'&&!(mode==='guided'&&step<SIX('eot'))};}
function chipLab(x,idx,s){   // the label under a chip: 'none' (not predicted), 'ok' (counted), 'ign' (−100), plus bad (the real end token lost its label)
  if(idx===0) return {cls:'none',t:tx('—','—')};
  const msk=masked(s);
  if(x.who==='p') return s.count==='all'&&!(msk==='id'&&x.k==='eot')?{cls:'ok',t:'✓'}:{cls:'ign',t:'−100'};
  if(x.k==='eot'&&msk) return {cls:'ign bad',t:'−100',why:msk};
  return {cls:'ok',t:'✓'};}
function chipHTML(x,lab,showLab){
  const textN=x.text?x.n:0, isS=x.k&&isSp(x.k);
  const t=x.text?(x.text==='u'?tx(`user text: ${I(textN)} tokens`,`用户的话：${I(textN)} 个 token`):tx(`answer: ${I(textN)} tokens`,`回答：${I(textN)} 个 token`)):esc(tokS(x.k));
  let tip=x.text?(x.who==='p'?tx(`The user’s text in this turn: ${I(textN)} tokens (an example count).`,`这一轮用户的话：${I(textN)} 个 token（示例计数）。`):tx(`The assistant’s answer in this turn: ${I(textN)} tokens (an example count).`,`这一轮助手的回答：${I(textN)} 个 token（示例计数）。`))
    :(SP_TIP[x.k]?SP_TIP[x.k]():(x.who==='p'?tx('Part of the prompt: the user writes it at use time.','提问部分的 token：用的时候由用户给出。'):tx('Part of the answer: the model must learn to write it.','回答部分的 token：模型要学着写出它。')));
  if(showLab&&lab&&!(lab.cls==='none'&&x.k==='bot')) tip+=' '+(lab.cls==='none'?tx('No position predicts it.','没有位置预测它。'):(lab.cls.startsWith('ign')?(/bad/.test(lab.cls)?(lab.why==='id'?tx('Its label is −100 because it has the pad token’s id: the model never learns to write it here.','它的 label 是 −100，因为它和 pad token 是同一个 id：模型学不到要在这里写它。'):tx('Its label was set to −100: the model never learns to write it here.','它的 label 被设成了 −100：模型学不到要在这里写它。')):tx('Label −100: not in the loss.','label 是 −100：不算 loss。')):tx('Counted: its label is the token itself, which the position before must predict.','算 loss：label 就是这个 token 本身，前一个位置要预测出它。')));
  const cls=['sft-tk',x.who==='p'?'sft-tp':'sft-tr',x.text?'sft-tx':'',isS?'sft-sp':'',lab&&/bad/.test(lab.cls)&&showLab?'sft-bad':'',x.pad?'sft-padc':''].filter(Boolean).join(' ');
  return `<span class="${cls}" tabindex="0" data-tipx="${esc(tip)}"><span class="sft-tt${(x.k&&isMono(x.k))||x.pad?' mono':''}">${t}</span>${showLab?`<span class="sft-lb mono ${lab?lab.cls:''}">${lab?lab.t:''}</span>`:''}</span>`;}
/* Codex R10A (round 3): in Scenarios and Settings the strip and the bar count the example picked there, while the small model below is
   always trained on its own 6 example conversations. Each box says on its first line which data it shows, so a reader who skips the
   card still knows that a scenario changes the counts and not the model's training data. */
const SCOPE={cnt:{en:'Counts for the current scenario: its tokens, and which of them count in the loss',zh:'当前场景的计数：这段例子的 token，以及哪些算 loss'},
  share:{en:'Counts for the current scenario',zh:'当前场景的计数'},
  model:{en:'The small model trained on its fixed 6 example conversations: its training data does not change with the scenario',zh:'固定 6 段示例对话训练的小模型：训练数据不随场景变化'}};
function drawScope(){$$('[data-scope]',app).forEach(e=>{const on=mode==='sandbox', t=on?L(SCOPE[e.dataset.scope]):''; e.hidden=!on; if(e.textContent!==t) e.textContent=t;});}
function drawStrip(){
  const sb=$('#sft-stripbox',app); sb.hidden=isP('dpo'); if(sb.hidden) return;
  const {ex,showLab,showPad}=stripParts(), host=$('#sft-strip',app), cap=$('#sft-strip-cap',app), leg=$('#sft-strip-leg',app);
  const s=mode==='guided'&&STEPS[step].id==='predict'?{...S,count:'all',eotl:'yes',pad:'sep'}:S;
  let html='';
  if(ex==='plain'){cap.innerHTML=tx(`The base model’s input: ${qTxt()} as plain text, no template`,`基础模型的输入：纯文字的${qTxt()}，没有模板`);
    html=`<div class="sft-grp"><div class="sft-grp-t">${tx('Plain text','纯文字')}</div><div class="sft-chips">${P_PLAIN.map(k=>chipHTML({who:'p',k},null,false)).join('')}</div></div>`; leg.innerHTML=''; host.innerHTML=html; return;}
  const ps=partsOf(ex);
  const groups=[]; let cur=null;
  ps.forEach((x,i)=>{if(!cur||cur.who!==x.who){cur={who:x.who,items:[]}; groups.push(cur);} cur.items.push({x,lab:chipLab(x,i,s)});});
  if(showPad){const padK=s.pad==='eot'?'eot':'pad'; const g={who:'pad',items:[]}; for(let i=0;i<NPAD;i++) g.items.push({x:{who:'pad',k:padK,pad:1},lab:{cls:'ign',t:'−100'}}); groups.push(g);}
  html=groups.map(g=>`<div class="sft-grp sft-g-${g.who}"><div class="sft-grp-t">${g.who==='p'?tx('Prompt part','提问部分'):(g.who==='r'?tx('Response part','回答部分'):tx('Padding (illustrative)','补齐（示意）'))}</div><div class="sft-chips">${g.items.map(it=>it.x.pad?padChip(it.x,showLab,s):chipHTML(it.x,it.lab,showLab)).join('')}</div></div>`).join('');
  host.innerHTML=html;
  const c=counts(ex,s.count,masked(s));
  cap.innerHTML=(ex==='toy'?tx(`One of the 6 example conversations, in Llama 3.1’s format: ${c.P+c.R} tokens`,`6 段示例对话中的一段，按 Llama 3.1 的格式拼成 ${c.P+c.R} 个 token`)
    :tx(`${L(EXNAME[ex])} in Llama 3.1’s format: ${I(c.P+c.R)} tokens (example counts)`,`${L(EXNAME[ex])}，按 Llama 3.1 的格式拼好：${I(c.P+c.R)} 个 token（示例计数）`));
  leg.innerHTML=`<span data-tipx="${esc(tx('Hatched chips are the prompt part: the user’s words and the tokens the template adds. At use time they are already in front of the model, so it never has to write them.','斜线的方块是提问部分，包括用户写的话和模板加的 token。用的时候，这些都已经摆在模型前面，模型不用写。'))}"><i class="sw sft-sw-p"></i>${tx('prompt part','提问部分')}</span><span data-tipx="${esc(tx('Plain chips are the response part: the answer and its end token, which the model must learn to write.','空白的方块是回答部分：回答和它末尾的结束 token，模型要学着写出它们。'))}"><i class="sw sft-sw-r"></i>${tx('response part','回答部分')}</span>`+
    (showLab?`<span data-tipx="${esc(tx('✓: this position counts. Its label is the token itself, which the position before must predict. Its loss enters the mean.','打 ✓ 的位置算 loss。它的 label 就是这个 token 本身，前一个位置要预测出它。这个位置的 loss 进入平均。'))}"><b class="mono">✓</b> ${tx('counted','算 loss')}</span><span data-tipx="${esc(IGN_TIP())}"><b class="mono">−100</b> ${tx('not counted','不算 loss')}</span>`:'')+
    (showLab&&masked(s)&&ex!=='plain'?`<span data-tipx="${esc(tx('Red: the real end token after the answer lost its label.','红框：回答末尾真正的结束 token 丢了 label。'))}"><i class="sw sft-sw-bad"></i>${tx('end token lost its label','结束 token 丢了 label')}</span>`:'');}
/* part 5: why a batch is padded at all (the owner's review of batch 9, item 4; G4). The conversations of one batch are stacked into
   one matrix, a row each, so the shorter ones are filled up to the longest, and the padded positions get −100. The middle row is the
   conversation in the strip above (13 prompt and 5 response tokens, then the 3 padded positions drawn there); the other two rows are
   illustrative. With the end token as pad, the padding rule also sets every row's real end token to −100 (red, R5.11). */
const BATCH=[{p:15,r:6},{p:13,r:5,me:1},{p:9,r:5}];   // prompt part, response part with its end token; the longest row has 21
function drawBatch(s){
  const box=$('#sft-batch',app); if(!box) return;
  box.hidden=!isP('eot'); if(box.hidden) return;
  const eot=s.pad==='eot', sig=A.lang()+'|'+eot; if(box._sig===sig) return; box._sig=sig;
  const wid=Math.max(...BATCH.map(b=>b.p+b.r));
  const rows=BATCH.map(b=>{const n=b.p+b.r; let c='';
    for(let j=0;j<wid;j++) c+=`<i class="sft-bc ${j<b.p?'sft-bc-p':(j<n-1?'sft-bc-r':(j===n-1?'sft-bc-e'+(eot?' sft-bad':''):'sft-bc-x'))}"></i>`;
    return `<span class="sft-brow-t">${b.me?tx('the one above','上面这段'):tx('another one','另一段')}</span><span class="sft-bcells">${c}</span>`;}).join('');
  const aria=BATCH.map(b=>{const n=b.p+b.r; return b.me?tx(`the one above: ${n} tokens and ${wid-n} padded`,`上面这段：${n} 个 token，补齐 ${wid-n} 个`):(n===wid?tx(`another one: ${n} tokens, the longest`,`另一段：${n} 个 token，最长`):tx(`another one: ${n} tokens and ${wid-n} padded`,`另一段：${n} 个 token，补齐 ${wid-n} 个`));}).join(tx('; ','；'));
  box.innerHTML=`<div class="sft-cap">${tx('One batch: its conversations stacked into one matrix, every row the same length (illustrative)','一个 batch：几段对话叠成一个矩阵，每一行一样长（示意）')}</div>`+
    `<div class="sft-brows" role="img" aria-label="${esc(aria)}">${rows}</div>`+
    `<div class="sft-leg"><span><i class="sw sft-sw-p"></i>${tx('prompt part','提问部分')}</span><span><i class="sw sft-sw-r"></i>${tx('response part','回答部分')}</span>`+
    (eot?`<span data-tipx="${esc(tx('The pad token is the end token, so the rule “label −100 wherever the token is the pad token” hits every real end token too.','pad token 就是结束 token，“凡是 pad token 的位置 label 设成 −100”这条规则，也会改到每一行真正的结束 token。'))}"><i class="sw sft-sw-bad"></i>${tx('end token lost its label','结束 token 丢了 label')}</span>`
        :`<span data-tipx="${esc(tx('The end token closes each answer; its label stays, so the model learns to write it.','结束 token 在每个回答的末尾。它的 label 保留，模型才学得到要写它。'))}"><i class="sw sft-sw-e"></i>${tx('end token','结束 token')}</span>`)+
    `<span data-tipx="${esc(tx('Padded positions only fill the row up to the longest one. Their label is −100, so they are not in the loss.','补齐的位置只是把这一行补到和最长的一样长。它们的 label 是 −100，不算 loss。'))}"><i class="sw sft-sw-x2"></i>${tx('padded, label −100','补齐的位置，label 是 −100')}</span></div>`;}
function padChip(x,showLab,s){const t=tokS(x.k), tip=s.pad==='eot'?tx('Padding with the end token: every position holding its id gets label −100, the real end token included.','拿结束 token 补齐：凡是这个 id 的位置，label 都设成 −100，真正的结束 token 也不例外。'):SP_TIP.pad()+tx(' ','')+tx('Its positions get label −100.','补齐的位置 label 都是 −100。');
  return `<span class="sft-tk sft-tpad sft-sp" tabindex="0" data-tipx="${esc(tip)}"><span class="sft-tt mono">${esc(t)}</span>${showLab?'<span class="sft-lb mono ign">−100</span>':''}</span>`;}

/* the generation panel and its player (R3.9, R3.20) */
function showGen(){
  const gv=genView(); const box=$('#sft-genbox',app); if(box) box.hidden=gv==='none';
  if(gv==='none'){stopAnim(); PL.g=null; PL.sig=null; return;}
  const g=genFor(S,gv), sig=sigOf(g,gv);
  if(sig===PL.sig){PL.g=g; return;}
  const prev=PL.g; PL.g=g; PL.sig=sig;
  if(PL.watched.has(sig)){stopAnim(); PL.i=g.length; return;}
  let from=0; if(prev){while(from<prev.length&&from<g.length&&prev[from].k===g[from].k&&Math.round(prev[from].p*100)===Math.round(g[from].p*100)) from++;}
  PL.i=Math.min(PL.i,from); PL.quick=mode!=='guided'; play(true);
  if(mode==='guided'){const mine=userAct; requestAnimationFrame(()=>revealGen(!mine));}}
function play(keep){if(!PL.g) return; if(!keep&&PL.i>=PL.g.length) PL.i=0; PL.on=true; PL.last=0; if(A.reduceMotion()){finishRun(); return;} if(!PL.raf) PL.raf=requestAnimationFrame(tick); drawPlayer();}
function stopAnim(){PL.on=false; if(PL.raf) cancelAnimationFrame(PL.raf); PL.raf=0;}
function tick(t){PL.raf=0; if(!app||!PL.on||!PL.g) return;
  if(mode==='guided'&&!genInView()){PL.last=0; PL.raf=requestAnimationFrame(tick); return;}   // G6: wait, at the token reached, until it can be seen
  const dt=PL.last?Math.min(0.1,(t-PL.last)/1000):0.016; PL.last=t;
  const per=PL.quick?Math.min(SPD[PL.spd],QUICK/PL.g.length):SPD[PL.spd];
  PL.i=Math.min(PL.g.length,PL.i+dt/per); if(PL.i>=PL.g.length){finishRun(); return;} drawGen(); drawPlayer(); PL.raf=requestAnimationFrame(tick);}
/* G6 (the owner's message of 2026-10-08): when a run the reader started begins, the place where it plays is on screen. Wide screens:
   the sticky stage scrolls inside itself; phones: the page scrolls to it, and the observation comes into view when the run ends.
   stageOnly (a part entered with Next or Back): only the sticky stage scrolls, the page stays on the new card */
function revealGen(stageOnly){
  if(!app||!PL.on) return; const box=$('#sft-genbox',app), st=$('#sft-stage',app); if(!box||box.hidden||!st) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), dh=dk&&!dk.hidden?dk.getBoundingClientRect().height:0;
  const top=(bar?bar.getBoundingClientRect().bottom:56)+dh+8, r=box.getBoundingClientRect(), need=Math.min(r.height,170), beh=A.reduceMotion()?'auto':'smooth';
  if(getComputedStyle(st).position==='sticky'&&st.scrollHeight>st.clientHeight+1){const sr=st.getBoundingClientRect(), v0=Math.max(sr.top,top), v1=Math.min(sr.bottom,innerHeight)-8;
    if(r.top>=v0&&r.top+need<=v1) return; st.scrollBy({top:r.top+need>v1?r.top+need-v1:r.top-v0,behavior:beh}); return;}
  if(stageOnly||(r.top>=top&&r.top+need<=innerHeight-8)) return;
  window.scrollBy({top:r.top-top,behavior:beh});}
/* G6: is the row of written tokens on screen (inside the sticky stage's visible part on wide screens)? A guided run that starts out
   of view (a phone after Next: the card in view, the tokens below the fold) waits at its first token until the reader scrolls to it */
function genInView(){const el=$('#sft-gen',app), box=$('#sft-genbox',app); if(!el||!box||box.hidden) return true; const r=el.getBoundingClientRect(); if(!r.height) return true;
  const bar=document.querySelector('.topbar'); let v0=bar?bar.getBoundingClientRect().bottom:0, v1=innerHeight;
  const st=$('#sft-stage',app); if(st&&getComputedStyle(st).position==='sticky'){const sr=st.getBoundingClientRect(); v0=Math.max(v0,sr.top); v1=Math.min(v1,sr.bottom);}
  return r.top<v1-16&&r.bottom>v0+16;}
/* a run the learner started from the rail ends: its observation appears, and is brought into view once */
function finishRun(){stopAnim(); if(PL.g) PL.i=PL.g.length; if(PL.sig) PL.watched.add(PL.sig); const rv=obsPending; obsPending=false; render(); if(rv){wantObs=true; revealSoon(); wantObs=false;}}
function drawPlayer(){const pl=$('#sft-player',app); if(!pl||!PL.g) return;
  const n=PL.g.length, k=Math.floor(PL.i+1e-9), done=k>=n;
  const btn=PL.on?tx('❚❚ Pause','❚❚ 暂停'):(done?tx('↺ Replay','↺ 重放'):tx('▶ Play','▶ 播放'));
  const cnt=tx(`${k} of ${n} tokens`,`第 ${k} 个 token（共 ${n} 个）`);
  if(pl._lang===A.lang()&&pl._n===n){pl.querySelector('[data-act="play"]').textContent=btn; const sc=pl.querySelector('input[data-key="scrub"]'); sc.value=k; A.ui.setRange(sc); pl.querySelector('.sft-cnt').textContent=cnt; $$('.seg-ctl[data-key="spd"]',pl).forEach(gg=>A.ui.setPressed(gg,PL.spd)); return;}
  pl._lang=A.lang(); pl._n=n;
  pl.innerHTML=`<button type="button" class="btn ghost sm" data-act="play" data-tipx="${esc(tx('Write the answer again token by token at the chosen speed; press again to pause.','按选定的速度一个 token 一个 token 地重新写；再按一次暂停。'))}">${btn}</button>`+
   A.ui.seg({id:'sft-spd',key:'spd',label:tx('Speed','速度'),options:[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]],value:PL.spd})+
   `<div class="sft-scrub"><input type="range" min="0" max="${n}" step="1" value="${k}" data-key="scrub" aria-label="${esc(tx('How many tokens are shown','显示到第几个 token'))}"><span class="sft-cnt mono">${cnt}</span></div>`;
  A.ui.initRanges(pl);}
function drawGen(){
  const host=$('#sft-gen',app), cap=$('#sft-gen-cap',app), leg=$('#sft-gen-leg',app); if(!host) return;
  const gv=genView(); if(gv==='none'||!PL.g) return;
  const g=PL.g, gi=genInfo(g), k=Math.floor(PL.i+1e-9);
  /* the caption says who is asked what, then what it does (review S12: a sentence with a verb, not a noun phrase) */
  const tail=tx(' It takes the most likely token at every step and writes the tokens below.','它每一步都挑概率最高的 token，写出下面这些。');
  const diff=(mode==='challenge'||gv==='sft')&&S.tmpl==='diff';
  const capT=gv==='basePlain'?tx(`The base model is asked ${qTxt()} as plain text.`,`问基础模型纯文字的${qTxt()}，`)
    :gv==='baseTmpl'?tx(`The base model is asked ${qTxt()} in the chat template, ending with the assistant header.`,`按聊天模板问基础模型${qTxt()}，提问末尾接上 assistant 的 header。`)
    :diff?tx(`After ${S.steps} SFT steps, the small model is asked ${qTxt()} in plain text with no special tokens.`,`小模型 SFT ${S.steps} 步之后，用没有特殊 token 的纯文字格式问它${qTxt()}。`)
    :tx(`After ${S.steps} SFT steps, the small model is asked ${qTxt()} in the training template, ending with the assistant header.`,`小模型 SFT ${S.steps} 步之后，按训练时的模板问它${qTxt()}，提问末尾接上 assistant 的 header。`);
  const sig=[PL.sig,k,A.lang(),mode].join('|'); if(host._sig===sig) return; host._sig=sig;
  cap.innerHTML=capT+tail;
  host.innerHTML=g.map((x,i)=>{const show=i<k, red=i===gi.red, extra=gi.red>=0&&i>gi.red, stop=x.k==='eot';
    const cls=['sft-gt',stop?'sft-stop':'',red?'sft-red':'',extra?'sft-extra':''].filter(Boolean).join(' ');
    const tip=(red?tx('It should have written <|eot_id|> here and stopped. ','这里该写 <|eot_id|> 然后停下。'):(stop?tx('The end token: generation stops here. ','结束 token：生成到这里就停了。'):''))+tx(`The model gave this token ${pc(x.p)} probability.`,`模型给这个 token 的概率是 ${pc(x.p)}。`);
    return `<span class="${cls}"${show?'':' style="visibility:hidden"'} tabindex="${show?0:-1}" data-tipx="${esc(tip)}"><span class="sft-tt${isMono(x.k)?' mono':''}">${esc(tokS(x.k))}</span><span class="sft-pr mono">${pc(x.p)}</span>${red&&show?`<span class="sft-redt">${tx('should stop','该停了')}</span>`:''}</span>`;}).join('')+
    (k>=g.length&&g.length&&g[g.length-1].k!=='eot'?`<span class="sft-more muted small">${tx(`… still going after ${MAXG} tokens`,`…写满 ${MAXG} 个 token 还没停`)}</span>`:'');
  leg.innerHTML=`<span data-tipx="${esc(tx('The number under each token is the probability the model gave it.','每个 token 下面的数，是模型给它的概率。'))}"><b class="mono">%</b> ${tx('probability','概率')}</span><span data-tipx="${esc(tx('Green: the model wrote the end token and stopped.','绿框：模型写出结束 token，停了。'))}"><i class="sw sft-sw-stop"></i>${tx('end token, stops','结束 token，停了')}</span><span data-tipx="${esc(tx('Red: where the answer ended and the model should have written <|eot_id|>.','红框：回答已经结束，这里该写 <|eot_id|>。'))}"><i class="sw sft-sw-bad"></i>${tx('should have stopped','该停没停')}</span><span data-tipx="${esc(tx('Dashed: tokens written after the point where it should have stopped.','虚线框：该停之后多写的 token。'))}"><i class="sw sft-sw-x"></i>${tx('written after that','多写的')}</span>`;}
function drawShare(){
  const box=$('#sft-sharebox',app), host=$('#sft-sharebar',app), cap=$('#sft-share-cap',app);
  const {ex}=stripParts(); const pv=mode==='guided'&&STEPS[step].id==='predict';
  if(ex==='plain'||(mode==='guided'&&step<SIX('predict'))||isP('dpo')){box.hidden=true; return;} box.hidden=false;
  const s=pv?{...S,count:'all',eotl:'yes',pad:'sep'}:S, c=counts(ex,s.count,masked(s));
  cap.innerHTML=esc(pv&&!pred?tx('Of the counted tokens in this example, how many are the prompt’s and how many the answer’s','这段例子算 loss 的 token 里，提问部分和回答部分各有多少')
    :tx(`Of the ${I(c.N)} counted tokens in this example, how many are the prompt’s and how many the answer’s`,`这段例子算 loss 的 ${I(c.N)} 个 token 里，提问部分和回答部分各有多少`));
  if(pv&&!pred){host.innerHTML=`<div class="sft-hide muted small">${tx('Shown once you pick an answer.','选好答案后再显示。')}</div>`; return;}
  const a=c.N?c.cp/c.N:0, psh=c.share>0&&c.share<0.1?f(a*100,2)+'%':pcShare(a);   // same decimals as the answer's share, so the two add up to 100%
  host.innerHTML=`<div class="sft-bar" role="img" aria-label="${esc(tx(`Prompt ${I(c.cp)} tokens, answer ${I(c.cr)} tokens`,`提问 ${I(c.cp)} 个 token，回答 ${I(c.cr)} 个 token`))}"><i class="sft-bp" style="width:${(a*100).toFixed(2)}%"></i><i class="sft-br" style="width:${((1-a)*100).toFixed(2)}%"></i></div>`+
   `<div class="sft-bar-l small"><span data-tipx="${esc(tx('Counted prompt tokens, including the template’s.','算 loss 的提问部分 token，包括模板加的。'))}"><i class="sw sft-sw-p"></i>${tx(`prompt ${I(c.cp)} (${psh})`,`提问 ${I(c.cp)} 个（${psh}）`)}</span><span data-tipx="${esc(tx('Counted answer tokens, including the end token when it counts.','算 loss 的回答部分 token，结束 token 算的话也在里面。'))}"><i class="sw sft-sw-r"></i>${tx(`answer ${I(c.cr)} (${pcShare(c.share)})`,`回答 ${I(c.cr)} 个（${pcShare(c.share)}）`)}</span></div>`;}
const MET=[
 {k:'n',label:()=>tx('Counted tokens','算 loss 的 token'),tip:()=>tx('How many positions of this example enter the mean: every position after the first, minus those labeled −100.','这段例子里有多少个位置进入平均：除了第一个 token 以外的所有位置，再去掉 label 是 −100 的。'),v:o=>I(o.c.N),sub:o=>L(EXNAME[o.ex])},
 {k:'sh',label:()=>tx('Answer’s share of them','其中回答部分占'),tip:()=>tx('The share of the counted tokens that belong to the answer. Every counted token has the same weight in the mean, so this is also the answer’s share of that weight.','算 loss 的 token 里，回答部分的 token 占几成。每个算 loss 的 token 在平均里权重一样，所以这也是回答部分分到的权重。'),v:o=>pcShare(o.c.share),pts:true,sub:o=>o.s.count==='all'?tx('every token counts','每个 token 都算'):tx('only the answer counts','只算回答部分')},
 {k:'rl',label:()=>tx('Small model’s loss on answers','小模型在回答上的 loss'),tip:()=>tx('The small model’s loss, averaged over the answer tokens of its 6 conversations, end tokens included whatever the labels say. The loss uses the natural log, so its unit is called nats. Lower is better; it is high when the model never learned to write the end token.','小模型在 6 段对话的回答 token 上的平均 loss。不管 label 怎么设，结束 token 都算在内。loss 用的是自然对数，单位叫 nats。这个数越低越好，模型没学会写结束 token 时，它会偏高。'),v:o=>o.rl==null?'—':f(o.rl,3),sub:()=>tx('in nats, after SFT','单位 nats，SFT 之后')},
 {k:'st',label:()=>tx('SFT steps','SFT 训练步数'),tip:()=>tx('Full-batch gradient steps from the pretrained small model.','从预训练好的小模型开始的全批量梯度下降步数。'),v:o=>o.rl==null?'—':String(S.steps),sub:()=>tx('step size 1','步长 1')}];
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/^[<>]?(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).match(/\.(\d+)/); return m?m[1].length:0;};
function drawMetrics(){
  const host=$('#sft-metrics',app); const {ex}=stripParts(); const pv=mode==='guided'&&STEPS[step].id==='predict';
  host.hidden=mode==='guided'&&(step<SIX('predict')||isP('dpo'));
  const s=pv?{...S,count:'all',eotl:'yes',pad:'sep'}:S, exx=ex==='plain'?'toy':ex, c=counts(exx,s.count,masked(s));
  const sftShown=genView()==='sft';
  const o={c,ex:exx,s,rl:sftShown?sftRun(S.count,masked(S),S.steps).rl:null};   // s: the setting the counts were computed with (part 3: every token, review B1)
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  const hideSh=(pv&&!pred)||(mode==='guided'&&step<SIX('predict'));
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`); el.dataset.tipx=m.tip(); el.querySelector('.k').textContent=m.label();
    let v=m.v(o); if(hideSh&&(m.k==='sh'||m.k==='n')) v='—';
    el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(o);
    const d=el.querySelector('.d'), a=lastShown[m.k], x=numOf(a), y=numOf(v);
    const dt=userAct&&a!=null&&isFinite(x)&&isFinite(y)?(A.fmt.delta(x,y,{pts:!!m.pts,dp:Math.max(decOf(a),decOf(v))})||''):'';
    clearTimeout(d._t); if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2600);} else {d.classList.remove('on'); d.textContent='';}
    lastShown[m.k]=v;});}
function mathHTML(){
  const {ex}=stripParts(), exx=ex==='plain'?'toy':ex, pv=mode==='guided'&&STEPS[step].id==='predict', s=pv?{...S,count:'all',eotl:'yes',pad:'sep'}:S, c=counts(exx,s.count,masked(s));
  const rows=[];
  // H2 (the owner, 2026-10-09): a few short blocks; each a lead line in words, the named formula, the reader's numbers under it, the result
  rows.push(`<p>${tx(`Each counted position adds ${NLP()}, where p is the probability the model gives the correct next token. The SFT loss is the mean over the N counted positions:`,`每个算 loss 的位置贡献 ${NLP()}，p 是模型给正确的下一个 token 的概率。SFT 的 loss 是这 N 个位置的平均：`)}</p>${EQ(tx('SFT loss','SFT 的 loss'),[LOSS_TEX])}<p>${XSYM()}</p>`);
  if(!(pv&&!pred)&&!(mode==='guided'&&step<SIX('predict'))){
    rows.push(`<div class="sft-mh">${tx('This example','这段例子')}${tx(': ','：')}${esc(L(EXNAME[exx]))}</div>`);
    rows.push(`<p>${tx(`The prompt part has ${I(c.P)} tokens and the response part ${I(c.R)}. Nothing comes before the first token, so no position predicts it and it does not count. Every counted token weighs the same in the mean, so the answer’s weight is its share of the counted tokens:`,`提问部分有 ${I(c.P)} 个 token，回答部分有 ${I(c.R)} 个。第一个 token 前面什么都没有，没有位置预测它，所以它不算。每个算 loss 的 token 在平均里权重一样，所以回答部分分到的权重，就是它在算 loss 的 token 里的占比：`)}</p>`);
    rows.push(EQ(tx('answer’s share','回答部分的占比'),[`\\frac{\\text{${tx('counted answer tokens','回答部分算 loss 的 token 数')}}}{\\text{${tx('all counted tokens','全部算 loss 的 token 数')}}}`,`\\frac{${I(c.cr)}}{${I(c.cp)} + ${I(c.cr)}}`,pcShare(c.share)]));
    rows.push(`<p>${tx(`Counting only the answer, an answer token weighs 1/${I(c.R)} in the mean. Counting every token, it weighs 1/${I(c.wAll)}. So its weight shrinks by:`,`只算回答时，一个回答 token 在平均里的权重是 1/${I(c.R)}。每个 token 都算时，它的权重是 1/${I(c.wAll)}。权重缩小的倍数是：`)}</p>`);
    rows.push(EQ(tx('weight shrinks by','缩小的倍数'),[`\\frac{\\text{${tx('positions if all count','每个 token 都算时的位置数')}}}{\\text{${tx('answer tokens','回答部分的 token 数')}}}`,`\\frac{${I(c.wAll)}}{${I(c.R)}}`,f(c.ratio,2)]));}
  rows.push(`<div class="sft-mh">${tx('The small model','小模型')}</div><p>${tx(`A vocabulary of ${NV} tokens and ${I(NPAR)} parameters. The ${NV} logits of the next token are a bias plus one row looked up for each of the 8 tokens before it, and softmax turns them into probabilities:`,`词表 ${NV} 个 token，共 ${I(NPAR)} 个参数。下一个 token 的 ${NV} 个 logit，等于一个偏置加上前面 8 个 token 各自查到的一行。softmax 再把它们变成概率：`)}</p>${EQ(`${tx('logits','logit')} ${A.ui.math('z')}`,[TOY_ZR])}${EQ(`${tx('probabilities','概率')} ${A.ui.math('p')}`,['softmax(z)'])}<p>${TOY_SYM()}</p>`+
    `<p>${tx('One SFT step uses all counted positions of the 6 conversations at once. g is the gradient of the loss with respect to each parameter, and the step size η is 1:','SFT 每一步都用上 6 段对话里所有算 loss 的位置。g 是 loss 对各个参数的梯度，步长 η 是 1：')}</p>${EQ(tx('one step','每一步'),['W \\leftarrow W - \\eta\\, g_W','b \\leftarrow b - \\eta\\, g_b'],'')}`);
  const r=sftRun(S.count,masked(S),S.steps);
  if(genView()==='sft') rows.push(`<p>${tx(`The current run of the small model (${S.count==='all'?'every token counts':'only the answer counts'}, end token ${masked(S)?'label −100':'counts'}, ${S.steps} steps) has ${r.n} counted positions; its loss on the answers is ${f(r.rl,3)} ${NATS()}.`,`当前这次训练（${S.count==='all'?'每个 token 都算':'只算回答部分'}，结束 token ${masked(S)?'label 是 −100':'算 loss'}，${S.steps} 步）有 ${r.n} 个算 loss 的位置，在回答上的 loss 是 ${f(r.rl,3)} ${NATS()}。`)}</p>`);
  return rows.join('');}
function statusOf(){
  if(mode==='challenge'){const e=chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})); return {cls:e.ok?'ok':'bad',t:e.ok?tx('Goal met','已达标'):tx('Not yet','未达标')};}
  if(isP('dpo')) return {cls:'',t:tx(`Step ${bk+1} of ${BK.length}`,`第 ${bk+1} 步，共 ${BK.length} 步`)};
  const gv=genView(); if(gv==='none') return {cls:'',t:tx('Token counts','token 计数')};
  if(!watchedNow()) return {cls:'',t:tx('Writing…','正在写…')};
  const gi=genInfo(PL.g||[]); return gi.stop?{cls:'ok',t:tx('Stops','停了')}:{cls:'bad',t:tx('Does not stop','没有停')};}
function drawHead(){
  const st=$('#sft-st-title',app), el=$('#sft-status',app), s=statusOf();
  st.textContent=isP('dpo')?tx('The answer’s log-probability, step by step','回答的 log 概率，一步一步算'):tx('Which tokens count, and what the trained model writes','哪些 token 算 loss，训练后的模型写出什么');
  el.className='status '+s.cls; el.textContent=s.t;
  el.dataset.tipx=isP('dpo')?tx('Which of the 4 steps of the calculation the chart shows.','图里正在算 4 步里的哪一步。'):mode==='challenge'?tx('Goal: the small model stops after its answer, and at least 90% of the counted tokens are the answer’s.','目标：小模型答完就停，并且算 loss 的 token 至少 90% 是回答部分的。'):tx('Whether the model in the chart wrote the end token and stopped.','图里的模型有没有写出结束 token、停下来。');}

/* ======================= controls, rail sync ======================= */
function syncControls(){
  $$('.seg-ctl[data-key]',app).forEach(g=>{const k=g.dataset.key; if(k==='spd') A.ui.setPressed(g,PL.spd); else if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
  const lock=mode==='challenge'||mode==='guided', c=$('#sft-controls',app); c.classList.toggle('locked',lock);
  const ln=$('#sft-lock',app); ln.hidden=!lock;
  if(mode==='challenge') ln.textContent=tx('In the challenge only the four settings in the task card change. The rest is fixed: the small model’s 6 conversations, 150 SFT steps.','挑战中只能改任务卡里的 4 个设置。其余固定：小模型的 6 段对话，SFT 150 步。');
  else if(mode==='guided') ln.textContent=tx('In the guided parts each part sets what it shows; every setting opens under Scenarios.','分步引导里由每一部分决定显示什么；所有设置在“应用场景”里都能改。');}
function isNextDisabled(){const s=STEPS[step]; return !!s.gate&&!s.gate();}
function nextWhy(){const id=STEPS[step].id; if(id==='predict') return tx('Pick an answer to continue, or Skip.','选一个答案才能继续，或点“跳过”。');
  if(id==='resp') return tx('Try “Only the answer counts” and watch the model write.','试一下“只算回答部分”，看模型写完。'); if(id==='eot') return tx('Try “The pad token is the end token” and watch the model write.','试一下“pad token 就是结束 token”，看模型写完。'); return '';}
function updateRail(){
  const rail=$('#sft-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    const html=s.obs?s.obs():''; if(box){box.hidden=!html; if(html&&box._h!==html){const again=box._h!=null&&box._step===step; box.innerHTML=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}}
    if(s.id==='resp'||s.id==='eot'){$$('[data-lever]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.lever===(s.id==='resp'?S.count:S.pad)));});
      $$('.lever',rail).forEach(lv=>{const b=lv.matches('[data-lever]')?lv:lv.querySelector('[data-lever]'); if(!b) return; const v=b.dataset.lever, r=lv.querySelector('.lv-r'); if(r){const txt=(s.id==='resp'?tried4:tried5).has(v)&&watchedNow()||((s.id==='resp'?tried4:tried5).has(v)&&v!==(s.id==='resp'?S.count:S.pad))?(s.id==='resp'?lev4Res(v):lev5Res(v)):''; if(r.textContent!==txt) r.textContent=txt;}});}
    if(s.id==='predict') $$('[data-pred]',rail).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pred===pred)));
    A.ui.setNext(rail,isNextDisabled(),nextWhy());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!(s.gate&&s.gate());}
  else if(mode==='sandbox'){const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#sft-insight',rail); if(ins){const h=insightHTML(); if(ins._h!==h){const again=ins._h!=null; ins.innerHTML=`<div class="eyebrow">${presetNow()>=0?esc(L(PRESETS[presetNow()].t)):tx('Your own setup','你自己的设置')}</div>`+h; ins._h=h; if(again) A.ui.obsBump(ins);}}}
  else {const e=chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})), sv=$('#sft-ch-sv',rail); if(!sv) return;
    sv.textContent=e.stop?tx('yes','是'):tx('no','否'); $('#sft-ch-stop',rail).classList.toggle('met',e.stop);
    $('#sft-ch-hv',rail).textContent=pcShare(e.share); $('#sft-ch-hp i',rail).style.width=Math.min(100,e.share/CH_GOAL*100)+'%';
    $('#sft-ch-share',rail).classList.toggle('met',e.share>=CH_GOAL); $('#sft-ch-hp',rail).classList.toggle('met',e.share>=CH_GOAL);
    if(chSubmitted&&e.ok) chReveal=true;
    const res=$('#sft-ch-res',rail), h=chResultHTML(); if(res._h!==h){const again=res._h!=null&&res._h!==''&&h!==''; res.innerHTML=h; res._h=h; if(again) A.ui.obsBump(res);}}}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:nextWhy(),skip:!!s.skip&&!pred});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#sft-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}

/* ======================= render ======================= */
function render(){
  if(!app) return;
  showGen();
  drawHead(); drawStrip(); drawBatch(S); drawGen(); drawPlayer(); drawShare(); drawBridge(); drawScope(); drawMetrics();
  const mb=$('#sft-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const s=statusOf(); A.announce(s.t);},400);}
function setState(p){Object.assign(S,p); if(mode==='challenge'){CH_KEYS.forEach(k=>{if(!CH_OPTS[k].includes(S[k])) S[k]=CH_START[k];}); S.ex='toy'; S.steps=CH_START.steps;} render();}
function setLever4(v){tried4.add(v); obsPending=true; setState({count:v});}
function setLever5(v){tried5.add(v); obsPending=true; setState({pad:v});}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter()); else Object.assign(S,DEF);
  if(s.id==='resp') tried4.add(S.count); if(s.id==='eot') tried5.add(S.pad);
  const st=$('#sft-stage',app); if(st) st.scrollTop=0;
  renderRail();}
function revealStep(){A.ui.reveal($('#sft-rail .step.current',app)||$('#sft-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#sft-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#sft-insight',app)); else revealBox($('#sft-ch-res',app));}
const revealSoon=()=>{if(wantObs) requestAnimationFrame(revealObs);};
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
/* after a click in the rail, keep the sticky stage's head below the top bar (R3.2), copied from labs/llm-judge.js (R9.7) */
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#sft-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#sft-rail .step.current',app)||$('#sft-rail .railcard',app)||$('#sft-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on (keys, scrollbar, a link), so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH_START};}
  if(m==='sandbox'&&was==='guided') S={...PRESETS[0].s};
  A.shell.setMode(app,m,DESCS());
  const st=$('#sft-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
const byLearner=fn=>{userAct=true; try{fn();} finally{userAct=false;}};
const token=()=>[S.ex,S.count,S.eotl,S.pad,S.tmpl,S.steps].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==6) return null; const [ex,count,eotl,pad,tmpl,st]=p, o={ex,count,eotl,pad,tmpl,steps:+st};
  if(!EXNAME[ex]||!['all','resp'].includes(count)||!['yes','no'].includes(eotl)||!['sep','eot'].includes(pad)||!['same','diff'].includes(tmpl)||!STEPS_OPT.includes(o.steps)) return null; return o;}

/* ======================= drill and exits ======================= */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){const r=mulberry(seed); const P=10*Math.round(20+r()*380), R=5*Math.round(10+r()*150); const N=P+R-1; return {seed,P,R,N,share:R/N,ratio:N/R};}
function renderDrill(){if(!app) return; const d=makeDrill(drillSeed), q=$('#sft-drill-q',app), a=$('#sft-drill-a',app);
  q.innerHTML=tx(`One SFT example: the prompt part, with the template’s tokens, is ${I(d.P)} tokens, and the response part, with its end token, is ${I(d.R)}. If every token counts, what share of the counted tokens is the answer’s, and what fraction of its answer-only weight does each answer token keep?`,
                 `一条 SFT 数据里，提问部分连同模板加的 token 有 ${I(d.P)} 个，回答部分连同结束 token 有 ${I(d.R)} 个。每个 token 都算 loss 时，回答部分占算 loss 的 token 的多少？每个回答 token 的权重，降到只算回答时的几分之一？`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const fr=fracOf(d.ratio);
  a.innerHTML=tx(`<p><b>${pcShare(d.share)}, and each answer token keeps about ${fr} of its answer-only weight.</b></p><p>Nothing comes before the first token, so ${I(d.P)} + ${I(d.R)} − 1 = ${I(d.N)} positions count. The answer’s share is ${I(d.R)} ÷ ${I(d.N)} = ${pcShare(d.share)}. Counting every token, an answer token weighs 1/${I(d.N)} in the mean; counting only the answer, 1/${I(d.R)}. ${I(d.N)} ÷ ${I(d.R)} = ${f(d.ratio,2)}, so about ${fr}.</p>`,
                 `<p><b>回答部分占 ${pcShare(d.share)}。每个回答 token 的权重，降到只算回答时的约 ${fr}。</b></p><p>第一个 token 前面什么都没有，没有位置预测它，所以算 loss 的位置有 ${I(d.P)} + ${I(d.R)} − 1 = ${I(d.N)} 个。回答部分占 ${I(d.R)} ÷ ${I(d.N)} = ${pcShare(d.share)}。每个 token 都算时，一个回答 token 在平均里的权重是 1/${I(d.N)}，只算回答时是 1/${I(d.R)}。${I(d.N)} ÷ ${I(d.R)} = ${f(d.ratio,2)}，所以约是 ${fr}。</p>`);}
/* a weight ratio as a fraction a reader can say: 1/38, or 1/2.4 below 10 (review S13: not “38 times smaller”) */
function fracOf(r){return '1/'+f(r,r>=10?0:1);}
/* new case 1: a share by ratio, with the page's reference (the article: 1,800 and 120); new case 2: a weight ratio */
function exCases(){const c1={P:200,R:600}, n1=c1.P+c1.R-1, s1=c1.R/n1; const c2={P:3000,R:100}, n2=c2.P+c2.R-1, r2=n2/c2.R; return {c1,n1,s1,c2,n2,r2};}
const EXITS=()=>{const E=exCases(), art=counts('article','all',false);
 const o1=[tx('About 25%','约 25%'),tx('About 50%','约 50%'),tx('About 75%','约 75%'),tx('About 100%','约 100%')], a1=Math.round(E.s1*4)-1;
 const o2=[tx('About 1/3','约 1/3'),tx('About 1/30','约 1/30'),tx('About 1/300','约 1/300')], a2=Math.round(Math.log10(E.r2/3));
 return {core:()=>chSubmitted, explain:[
  {q:tx('Why does SFT usually count the loss on the answer only?','SFT 为什么通常只算回答部分的 loss？'),
   o:[tx('At use time the user writes the prompt, so the model need not learn to write it; counting it lets a long prompt take most of the counted tokens, and the answer gets a small share of the loss.','用的时候提问由用户给出，模型不用学着写它；提问也算的话，长提问会占掉大部分算 loss 的 token，回答在 loss 里只分到一小份。'),
      tx('Counting the prompt makes the model forget what it learned in pretraining, so the prompt’s labels are set to −100 to protect that knowledge.','提问也算 loss 会让模型忘掉预训练学到的知识，所以要把提问部分的 label 设成 −100 来保护它。'),
      tx('Counting only the answer makes every training step much cheaper, because the model runs only on the answer’s tokens.','只算回答部分能让每一步训练便宜很多，因为模型只在回答的 token 上运行。')],a:0,
   n:[null,tx('−100 only keeps those positions out of the mean. The reasons are that the model never has to write the prompt, and that the prompt dilutes the answer’s weight.','−100 只是让这些位置不进平均。真正的原因是用的时候不用写提问，而且提问会摊薄回答的权重。'),
      tx('The forward and backward passes still run over the whole sequence, because the answer’s tokens attend to the prompt. −100 only changes which positions enter the mean.','前向和反向照样要算整段序列，因为回答的 token 要看提问。−100 只改变哪些位置进入平均。')],
   why:tx('The loss is the mean over the counted positions, so each counted token has the same weight. When the prompt counts too, the longer it is compared with the answer, the smaller the answer’s share of the loss.','loss 是算 loss 的位置的平均，每个算 loss 的 token 权重一样。提问也算 loss 时，提问比回答长得越多，回答在 loss 里分到的就越少。')},
  {q:tx('After the pad token was set to the end token, the fine-tuned model no longer stops after its answer. Why?','把 pad token 设成结束 token 以后，SFT 出来的模型答完不停了。为什么？'),
   o:[tx('The padded labels are set to −100 by token id; the real end token has the same id, so its label became −100 too and the model never learned to write it after an answer.','补齐处的 label 是按 token id 设成 −100 的；真正的结束 token 也是这个 id，它的 label 也成了 −100，模型从没学过答完要写它。'),
      tx('So many pad tokens dilute the end token’s weight in the mean that the model writes it too rarely, even though its label is still there.','pad token 太多，把结束 token 在平均里的权重摊薄了，虽然它的 label 还在，模型也很少写它。'),
      tx('At inference the pad tokens are read as part of the prompt, so the model thinks the conversation is not over yet.','推理时 pad token 被当成提问的一部分，模型以为对话还没结束。')],a:0,
   n:[null,tx('Padded positions are −100 and never enter the mean, so they dilute nothing. The end token’s own label is what went missing.','补齐的位置是 −100，根本不进平均，摊薄不了什么。丢掉的是结束 token 自己的 label。'),
      tx('Padding is added only in training batches; a single prompt at inference has none. The model fails to stop because it never learned to write the end token.','补齐只在训练的 batch 里加；推理时单个提问没有补齐。模型不停，是因为它从没学过写结束 token。')],
   why:tx(`transformers’ <code>DataCollatorForLanguageModeling</code> runs <code>labels[labels == pad_token_id] = -100</code>. With the end token as pad, that line also sets the ${EOT} after every answer to −100.`,`transformers 的 <code>DataCollatorForLanguageModeling</code> 会执行 <code>labels[labels == pad_token_id] = -100</code>。拿结束 token 当 pad 时，这一行会把每个回答末尾的 ${EOT} 也设成 −100。`)}],
 transfer:[
  {q:tx(`For example, with a ${I(art.P)}-token prompt part and a ${I(art.R)}-token response part, counting every token leaves the answer ${pcShare(art.share)} of the counted tokens. Now take another example: a ${I(E.c1.P)}-token prompt part and a ${I(E.c1.R)}-token response part. Counting every token, about what share of the counted tokens is the answer’s?`,
        `比如提问部分有 ${I(art.P)} 个 token、回答部分有 ${I(art.R)} 个，每个 token 都算时，回答只占算 loss 的 token 的 ${pcShare(art.share)}。现在换一条数据，提问部分 ${I(E.c1.P)} 个 token，回答部分 ${I(E.c1.R)} 个。每个 token 都算时，回答部分大约占算 loss 的 token 的多少？`),
   o:o1,a:a1,n:o1.map((x,i)=>i===a1?null:tx(`The right answer is “${o1[a1]}”.`,`正确答案是“${o1[a1]}”。`)),
   why:tx(`The answer is ${f(E.c1.R/E.c1.P,0)} times as long as the prompt, so it is about ${A.ui.def('three quarters',`${I(E.c1.P)} + ${I(E.c1.R)} − 1 = ${I(E.n1)} positions count, and ${I(E.c1.R)} ÷ ${I(E.n1)} = ${pcShare(E.s1)}.`)} of the counted tokens.`,
          `回答是提问的 ${f(E.c1.R/E.c1.P,0)} 倍长，所以大约占算 loss 的 token 的${A.ui.def('四分之三',`算 loss 的位置有 ${I(E.c1.P)} + ${I(E.c1.R)} − 1 = ${I(E.n1)} 个，${I(E.c1.R)} ÷ ${I(E.n1)} = ${pcShare(E.s1)}。`)}。`)},
  {q:tx(`For example, with a ${I(art.P)}-token prompt part and a ${I(art.R)}-token response part, counting every token cuts each answer token’s weight in the mean to about 1/${f(art.ratio,0)} of its weight when only the answer counts. Now take another example: a ${I(E.c2.P)}-token prompt and a ${I(E.c2.R)}-token answer. Counting every token, about what fraction of its answer-only weight does each answer token keep?`,
        `比如提问部分有 ${I(art.P)} 个 token、回答部分有 ${I(art.R)} 个，每个 token 都算时，每个回答 token 在平均里的权重，降到只算回答时的约 1/${f(art.ratio,0)}。现在换一条数据，提问 ${I(E.c2.P)} 个 token，回答 ${I(E.c2.R)} 个。每个 token 都算时，每个回答 token 的权重大约降到只算回答时的几分之一？`),
   o:o2,a:a2,n:o2.map((x,i)=>i===a2?null:tx(`The right answer is “${o2[a2]}”.`,`正确答案是“${o2[a2]}”。`)),
   why:tx(`The longer the prompt is relative to the answer, the thinner the answer is spread. Here each answer token keeps ${A.ui.def('about a thirtieth',`${I(E.c2.P)} + ${I(E.c2.R)} − 1 = ${I(E.n2)} positions count. Counting every token, an answer token weighs 1/${I(E.n2)}; counting only the answer, 1/${I(E.c2.R)}. ${I(E.n2)} ÷ ${I(E.c2.R)} ≈ ${f(E.r2,0)}.`)} of its answer-only weight.`,
          `提问比回答长得越多，回答被摊得越薄。这条数据里，每个回答 token 的权重，降到只算回答时的${A.ui.def('约三十分之一',`算 loss 的位置有 ${I(E.c2.P)} + ${I(E.c2.R)} − 1 = ${I(E.n2)} 个。每个 token 都算时，一个回答 token 的权重是 1/${I(E.n2)}，只算回答时是 1/${I(E.c2.R)}。${I(E.n2)} ÷ ${I(E.c2.R)} ≈ ${f(E.r2,0)}。`)}。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 10: state what the pretraining, SFT and preference-optimisation losses optimise, and why the SFT loss is masked to the response tokens.','第 10 题：说出预训练、SFT 和偏好优化的 loss 各自优化什么，并解释 SFT 的 loss 为什么只算回答部分。')},
  {id:'cross-entropy',d:tx('Write cross-entropy from scratch and match PyTorch with ignore_index = −100 and reduction “mean”: ignored positions stay out of the mean.','从零写交叉熵，并和 PyTorch 的 ignore_index = −100、reduction “mean”对上：被忽略的位置不进平均。')}]};};

/* ======================= tooltips, jumps ======================= */
function jumpAct(k){
  if(k==='math'||k==='toy') return tx('Click to open “Show the math”.','点击打开“展开公式”。');
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='gen') return tx('Click to jump to what the model writes.','点击跳到模型写出的 token。');
  if(k==='strip') return tx('Click to jump to the conversation’s tokens.','点击跳到这段对话的 token。');
  if(k==='batch') return tx('Click to jump to the batch in the chart.','点击跳到图里的 batch。');
  if(k==='dpo') return tx('Click to jump to the table in the chart.','点击跳到图里的表。');
  return '';}
function jump(k){
  if(!app) return;
  if(k==='toy') k='math';   // the small model's formula term opens Show the math, where its formulas are
  if(k==='math'){const d=$('#sft-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#sft-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='gen'||k==='strip'||k==='batch'||k==='dpo'){const el=$(k==='gen'?'#sft-genbox':(k==='batch'?'#sft-batch':(k==='dpo'?'#sft-dpobox':'#sft-stripbox')),app); A.tip.hide(); el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}}
function act(a,b){
  if(a==='play'){if(PL.on){stopAnim(); drawPlayer();} else {PL.quick=false; play(false);} return;}
  if(a==='reset'){byLearner(()=>{S={...DEF}; render();}); return;}
  if(a==='submit'){chSubmitted=true; const e=chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})); if(e.ok) chReveal=true; renderRail(); wantObs=true; revealSoon(); wantObs=false; return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='bk-next'||a==='bk-prev'){setBk(bk+(a==='bk-next'?1:-1)); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#sft-code',app); try{navigator.clipboard.writeText(pre.textContent).then(()=>{b.textContent=tx('Copied','已复制'); setTimeout(()=>{b.textContent=A.t('copyCode');},1600);},()=>{});}catch(e){} }}
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k==='scrub'){stopAnim(); PL.i=+e.target.value; if(PL.g&&PL.i>=PL.g.length){finishRun(); return;} drawGen(); drawPlayer(); drawHead();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#sft-rail')&&!b.closest('[data-exits]')) keepStage();
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key; if(k==='spd'){PL.spd=d.v; PL.last=0; drawPlayer(); return;} if(k==='bk'){setBk(+d.v); return;}
      const v=k==='steps'?+d.v:d.v; wantObs=!!b.closest('#sft-rail'); byLearner(()=>setState({[k]:v})); revealSoon(); wantObs=false; return;}
    if(d.brow!=null){setBrow(+d.brow); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) byLearner(()=>s.showMe()); return;}
    if(d.pred){pred=d.pred; renderRail(); wantObs=true; revealSoon(); wantObs=false; const fb=$(`#sft-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.lever!=null){if(isP('resp')) byLearner(()=>setLever4(d.lever)); else if(isP('eot')) byLearner(()=>setLever5(d.lever)); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>{Object.assign(S,p.s); render();}); requestAnimationFrame(()=>{if(app) revealBox($('#sft-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#sft-rail',app)); return;}
    if(d.act) act(d.act,b);});
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));}

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-12 07:13 UTC, under the rule "不要和lab里面要展示的内容直接重复"): four
   text-only sections read before the guided parts. They own base vs chat model, the post-training steps (SFT → preference learning
   → RL, one line each), the chat template and special tokens, and the SFT loss (the next-token loss itself is training-memory's: section 4 restates it on one line and links there; the SFT loss is the mean over the set C, as part 3).
   Both figure ideas of the outline were dropped: one replayed part 1's base-model run, the other part 2's template strip. The
   example conversation, the Llama 3.1 template, which positions count and the end token are left to the guided parts: section 4
   gives a case for each side and only asks the question (first-time reader, 2026-10-12). Symbols: θ parameters and L(θ) the loss as
   in backprop's primer; i a position (t is roofline's time); |x| and |C| counts, as |y| in ppo's primer (N is the parameter count)
   (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=id=>{const a=A.shell.labLink(id); return tx(a,`“${a}”`);};
// a paragraph per item; a formula block (a <div>) stands on its own
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
// a formula on its own line, in pieces that wrap under each other (at “=”) in a narrow column, so nothing scrolls sideways
const prFx=(...xs)=>`<div class="sft-prfx">${xs.map(x=>`<span>${prM(x)}</span>`).join('')}</div>`;
const PR_P='p_\\theta(x_i | x_{<i})', PR_LP='\\log\\, '+PR_P;
A.registerPrimer(ID,{min:7,sections:[
 {h:()=>tx('Pre-training gives a base model: it only continues documents','预训练得到基础模型：它只会续写文档'),parts:[prP(
   ()=>tx(`What we want is an assistant to talk to: the user asks something, it writes an answer, then hands the conversation back to the user. But a language model does only one thing: it reads the tokens so far and gives a probability distribution for the next token, and an answer is that step repeated many times (see ${prLK('architecture')}). What it writes depends entirely on the distribution training has taught it.`,
          `我们想要的是一个能对话的助手：用户提出问题，它写一段回答，然后把话交回给用户。可语言模型只会做一件事：读进前面的 token，给出下一个 token 的概率分布，一段回答就是把这一步重复很多次写出来的（见${prLK('architecture')}）。它会写出什么，全看训练让它学到了什么样的分布。`),
   ()=>tx(`A large model’s training starts with pre-training: on huge amounts of web pages, books and code, the model is trained to predict the next token of that text (see ${prLK('training-memory')}). A model trained this way is called a base model. Most of its language, facts and coding skill come from this stage; what it has learned is what usually follows a piece of text in a document.`,
          `大模型训练的第一个阶段是预训练：在海量的网页、书籍和代码上，训练模型预测这些文字的下一个 token（见${prLK('training-memory')}）。这样训练出来的模型，叫基础模型（base model）。语言、事实和写代码的本事，主要是在这一步学到的；它学会的，是一篇文档里一段文字后面通常接着什么。`),
   ()=>tx('But most of those documents are not conversations between a user and an assistant, and they do not mark who said which part or whose turn comes next. Given a question, a base model takes it as the start of a document and continues it the way such documents most often go. In a document, what follows a question is not always its answer: on an exercise sheet, the next line is the next exercise. So asked “What is the capital of France?”, it may go on with “What is the capital of Germany?”.',
          '可这些文档大多不是用户和助手之间的对话，也没有标出哪段话是谁说的、接下来该谁说。给基础模型一个问题，它把问题当作一篇文档的开头，按这类文档最常见的样子往下续写。而在文档里，一个问题后面接的不一定是它的答案：在一份练习题里，下一行是下一道题。所以问它“法国的首都是哪里？”，它可能接着写“德国的首都是哪里？”。'),
   ()=>tx('The chat assistants people use are chat models (also called instruct models): base models trained further, so that they answer what the user asks. Meta released Llama 3.1 in both forms: Llama-3.1-8B is the base model, and Llama-3.1-8B-Instruct is the chat model trained from it. The training that turns a base model into a chat model is called post-training. It uses far less data than pre-training, and what it sets out to teach is how the model responds to a user in a conversation.',
          '平时用的聊天助手是聊天模型（chat model，也叫指令模型）：在基础模型上接着训练，学会了按用户的要求回答。Meta 发布 Llama 3.1 时两种都发布了：Llama-3.1-8B 是基础模型，Llama-3.1-8B-Instruct 是在它上面训练出来的聊天模型。从基础模型到聊天模型的这段训练，叫后训练（post-training）。后训练用的数据比预训练少得多，它要教的是模型在对话里怎样回应用户。'))]},
 {h:()=>tx('The steps of post-training: SFT, preference learning, reinforcement learning','后训练的几步：SFT、偏好学习、强化学习'),parts:[prP(
   ()=>tx('A chat model has more than one thing to learn. First, to respond in the format of a conversation: understand what the user wants, write an answer, and hand the conversation back. Next, among several answers that all address the question, to write the one people like better: more useful, more accurate, leaving out what should not be said. For maths and coding questions it must also reason step by step to the right answer. These need different training data, namely written demonstrations, comparisons of two answers, and whether an answer is right, so post-training usually comes in several steps, each one training further the model the step before produced.',
          '聊天模型要学的不止一件事。首先是按对话的格式回应：看懂用户要什么，写出回答，写完把话交回给用户。其次，在几个都切题的回答里，写出人更满意的那个：更有用、更准确，不该说的不说。数学题、编程题还要一步步推理出正确答案。这几件事要的训练数据不一样：写好的示范、两个回答的比较、答案的对错，所以后训练通常分几步，每一步都在上一步训练出的模型上接着训练。'),
   ()=>tx('The first step is supervised fine-tuning (SFT). Its training data are demonstrations: a prompt paired with a written answer, written by people or generated by a stronger model. “Supervised” means every example comes with the answer the model should write; “fine-tuning” means training does not start from scratch but continues from the parameters pre-training produced. From the demonstrations the model learns the conversation format and to answer as asked. But it can only imitate them, and an excellent demonstration is hard and expensive to write; many questions also have no single good answer, and a demonstration gives only one of them, saying nothing about which of several answers is better.',
          '第一步是监督微调（supervised fine-tuning，SFT）。训练数据是示范：一个提问，配一个写好的回答，由人写，或者由更强的模型生成。“监督”是说每条数据都带着要模型写出的回答；“微调”是说不从头训练，而是在预训练得到的参数上接着训练。模型从示范里学会对话的格式和按要求作答。可它只能模仿示范，而写一份出色的示范又难又贵；很多问题也没有唯一的好答案，一份示范只给出其中一个，说不出几个回答之间哪个更好。'),
   ()=>tx(`The second step is preference learning. Writing the best answer by hand is hard for people, but judging which of two answers is better is much easier. So the model writes two answers to the same prompt, a person picks the better one, and the model learns from these comparisons. There are two families of methods. One first trains a model that scores answers, called the reward model, then uses the reinforcement learning algorithm PPO to raise the scores of the model’s answers: reinforcement learning here means the model writes answers itself, and its parameters are adjusted by the score each answer gets, called its reward. This combination is called RLHF (reinforcement learning from human feedback). The other family trains no reward model and learns from the paired comparisons directly, DPO for example (see ${prLK('reward-model')}, ${prLK('ppo')} and ${prLK('dpo')}).`,
          `第二步是偏好学习。人要亲手写出最好的回答很难，判断两个回答哪个更好却容易得多。于是让模型对同一个提问写两个回答，由人选出更好的一个，模型从这些比较里学。做法有两类。一类先训练一个给回答打分的模型，叫奖励模型，再用强化学习算法 PPO 提高模型回答的得分：这里的强化学习，是让模型自己写回答，按每个回答得到的分数（叫奖励）调整参数。这一套叫 RLHF（基于人类反馈的强化学习）。另一类不训练奖励模型，直接从成对的比较里学，比如 DPO（见${prLK('reward-model')}${prLK('ppo')}${prLK('dpo')}）。`),
   ()=>tx(`After that, reinforcement learning is also used to teach reasoning: the answers to maths and coding questions can be checked by a program, and a right answer earns a reward; this is called RLVR (reinforcement learning with verifiable rewards, see ${prLK('rlvr')}). Preference learning and reinforcement learning usually both learn from answers the model writes itself: the comparisons are between its answers, and the rewards go to the questions it gets right. A model that cannot yet answer in the format of a conversation gives them no answers to compare or reward. So SFT is usually the first step of post-training.`,
          `再往后，强化学习还用来教模型推理：数学题、编程题的答案能由程序判对错，答对就给奖励，这叫 RLVR（基于可验证奖励的强化学习，见${prLK('rlvr')}）。偏好学习和强化学习通常都在模型自己写出的回答上学：比较的是它写的回答，奖励的是它答对的题。模型要是连按对话的格式作答都不会，就没有回答可比、可奖励。所以 SFT 通常是后训练的第一步。`))]},
 {h:()=>tx('SFT data: demonstration conversations and the chat template','SFT 的数据：示范对话和 chat template'),parts:[prP(
   ()=>tx('The demonstrations SFT imitates come as conversations: the user sends a message and the assistant replies with one; a multi-turn conversation is several such exchanges. A conversation has structure: who wrote each message, and where each message ends. But the model’s input is only one sequence of tokens, with no other place to tell it who wrote which part. So the structure has to be written into that same sequence.',
          'SFT 要模仿的示范是对话的形式：用户发来一条消息，助手回一条；多轮对话就是几个这样的来回。对话有结构：每条消息是谁说的，在哪里说完。可模型的输入只有一串 token，没有别的地方能告诉它“这一段是谁说的”。所以这些结构只能写进同一串 token 里。'),
   ()=>tx('The simplest way is to write “User:” or “Assistant:” before each message. But those words also occur in ordinary text, and a user’s message may itself contain “Assistant:”; the model cannot tell whether the speaker has changed or this is just part of the message. So the vocabulary sets aside special tokens: each is an entry of its own with its own ID, stands for no text, and only marks structure, such as where a message starts and where it ends. The roles are usually user and assistant; many models also have system, for the rules a developer sets for the model.',
          '最简单的写法，是在每条消息前面写上“用户：”或“助手：”。可这几个字在普通文字里也会出现，用户的消息里也可能写着“助手：”，模型分不清这是换了人说话，还是消息本身的内容。所以词表里专门留了一些特殊 token（special token）：每个都是词表里单独的一项，有自己的 ID，不代表任何文字，只用来标出结构，比如一条消息从哪里开始、在哪里结束。角色通常是 user（用户）和 assistant（助手），很多模型还有 system，放开发者给模型定的规则。'),
   ()=>tx(`The rule that lays a conversation out as one sequence of tokens in a fixed format is called the chat template: which special tokens go before and after each message, and where the role name goes. Different models use different templates. Qwen (an open language model released by Alibaba), for example, encloses each message in ${code('<|im_start|>')} and ${code('<|im_end|>')}, with the role name user or assistant written right after ${code('<|im_start|>')}. Written out, ${code('<|im_start|>')} is only the name of one special token: it is a single entry in the vocabulary, not made of these characters. When the model is used, the template lays out the conversation so far and then appends the start of an assistant message: ${code('<|im_start|>assistant')} and a newline. A chat model, too, only continues text: the template puts the place to continue at the start of the assistant’s answer, so what it writes next is the answer.`,
          `把一段对话按固定格式拼成一串 token 的规则，叫 chat template（对话模板）：每条消息前后插入哪些特殊 token，角色名写在哪里。不同的模型用不同的模板。比如 Qwen（阿里巴巴发布的开源语言模型）用 ${code('<|im_start|>')} 和 ${code('<|im_end|>')} 把每条消息括起来，角色名 user 或 assistant 写在 ${code('<|im_start|>')} 后面。写出来的 ${code('<|im_start|>')} 只是一个特殊 token 的名字，它在词表里是单独的一项，不是由这几个字符拼成的。用模型的时候，模板把已有的对话拼好，最后再接上助手消息的开头：${code('<|im_start|>assistant')} 和一个换行。聊天模型也还是只会续写：模板把续写的位置安排在助手的回答处，它往下写出的就是回答。`))]},
 {h:()=>tx('SFT’s objective: the next-token cross-entropy loss','SFT 的训练目标：下一个 token 的交叉熵 loss'),parts:[prP(
   ()=>tx(`Once a conversation is a sequence of tokens, SFT uses pre-training’s loss: at each position, ${prM('-\\log')} of the probability the model gives the correct next token, averaged over the positions; this is the cross-entropy loss (see ${prLK('training-memory')}). For a sequence ${prM('x')} of ${prM('|x|')} tokens, with ${prM('x_i')} its ${prM('i')}-th token, ${prM('x_{<i}')} the tokens before it and ${prM('\\theta')} the model’s parameters:`,
          `对话拼成一串 token 以后，SFT 用的就是预训练的 loss：在每个位置上，对模型给正确的下一个 token 的概率取 ${prM('-\\log')}，再对所有位置取平均，这叫交叉熵 loss（见${prLK('training-memory')}）。记一段序列为 ${prM('x')}，共 ${prM('|x|')} 个 token，第 ${prM('i')} 个 token 为 ${prM('x_i')}，它前面的 token 为 ${prM('x_{<i}')}，模型的参数为 ${prM('\\theta')}：`),
   ()=>prFx('L(\\theta) = -\\frac{1}{|x|} \\sum_{i=1}^{|x|} '+PR_LP)),
  prP(
   ()=>tx(`In pre-training every position of a document is in the sum. In SFT the sequences are demonstration conversations laid out with the chat template, and the tokens of such a sequence come from different places: the user’s messages, the special tokens the template inserts, the assistant’s answers. When the model is used, the user types their own messages and the model writes only the assistant’s answers; yet the user’s messages are ordinary text too, and left in the sum they give the model more positions to learn from. So which positions go into the sum becomes a choice. Write the set of positions in the sum as ${prM('C')}; the mean is then taken over these ${prM('|C|')} positions only:`,
          `预训练时，文档里每个位置都在求和里。SFT 的序列换成了按 chat template 拼好的示范对话，这样一段序列里，token 的来历各不相同：有用户的消息、模板插入的特殊 token、助手的回答。用模型的时候，用户的消息由用户自己输入，模型只写助手的回答；可用户的消息也是正常的文字，留在求和里，模型就多了一些可以学的位置。所以求和取哪些位置成了一个选择。把进入求和的位置记作集合 ${prM('C')}，平均也只在这 ${prM('|C|')} 个位置上取：`),
   ()=>prFx('L_{\\text{SFT}}(\\theta) = -\\frac{1}{|C|} \\sum_{i \\in C} '+PR_LP),
   ()=>tx(`Which positions should ${prM('C')} hold? And the special token at the end of the assistant’s answer that says this message is finished (${code('<|im_end|>')} in Qwen) is inserted by the template: should it be in ${prM('C')}?`,
          `${prM('C')} 该包含哪些位置？助手回答末尾那个表示“这条消息说完了”的特殊 token（Qwen 里是 ${code('<|im_end|>')}）是模板插入的，它该不该在 ${prM('C')} 里？`))]}
]});

A.registerLab({
  id:ID,
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#sft-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#sft-rail',app)); keepStage();},token});
    bind(app); renderDrill();
    if(mode==='guided'&&!p&&!relang&&!restored){const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());}
    restored=false; renderRail();},
  /* round 3 #6 (the lab state contract): a reload continues where the learner was. The snapshot holds what the learner did: the mode, the
     part, the setup, the prediction, the levers tried in parts 4 and 5, the challenge's picks and its submission, part 6's step and row,
     the drill. The watched runs are not kept: a run the reader comes back to plays again. */
  snap:()=>({v:1,mode,step,S:{...S},pred,tried4:[...tried4],tried5:[...tried5],saved:saved?{...saved}:null,chSubmitted,chReveal,bk,brow,drillSeed,drillOpen,
    passed:mode==='challenge'&&chSubmitted&&chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})).ok}),
  restore(o){if(!o||o.v!==1) return;
    const tokOf=x=>x&&typeof x==='object'?[x.ex,x.count,x.eotl,x.pad,x.tmpl,x.steps].join('_'):'', p=parse(tokOf(o.S)); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    if(mode==='guided'){const st=STEPS[step]; S={...DEF,...(st.enter?st.enter():{})}; if(st.id==='resp') S.count=p.count; if(st.id==='eot') S.pad=p.pad;}
    else if(mode==='challenge'){S={...p}; CH_KEYS.forEach(k=>{if(!CH_OPTS[k].includes(S[k])) S[k]=CH_START[k];}); S.ex='toy'; S.steps=CH_START.steps;}
    else S={...p};
    pred=PREDS.some(x=>x.v===o.pred)?o.pred:null;
    tried4=new Set(Array.isArray(o.tried4)?o.tried4.filter(v=>LEV4.some(x=>x.v===v)):[]);
    tried5=new Set(Array.isArray(o.tried5)?o.tried5.filter(v=>LEV5.some(x=>x.v===v)):[]);
    const sv=parse(tokOf(o.saved)); saved=mode==='challenge'&&sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    bk=Number.isInteger(o.bk)&&o.bk>=0&&o.bk<BK.length?o.bk:0; brow=Number.isInteger(o.brow)&&o.brow>=0&&o.brow<5?o.brow:0;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=1&&o.drillSeed<1e6?o.drillSeed:3; drillOpen=o.drillOpen===true;
    restored=true;},
  leave(){stopAnim(); stageStop(); clearTimeout(annT); A.tip.hide(); root=null; app=null;},
  onLang(){if(root){const d=$('#sft-math',app), open=d&&d.open, was=PL.on; stopAnim(); relang=true; try{this.mount(root,null);} finally{relang=false;} if(open) $('#sft-math',app).open=true; if(was&&!watchedNow()) play(true);}},
  onResize(){},
  termTip(k){return TERM[k]?TERM[k]():'';},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#sft-dpobox:not([hidden])',app)||app&&$('#sft-genbox:not([hidden])',app)||app&&$('#sft-stripbox',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#sft-graph',app),
    render({status,vis,read}){if(!app) return; const s=statusOf(); status.className='status '+s.cls; status.textContent=s.t; vis.innerHTML='';
      if(mode==='challenge'){const e=chEval(CH_KEYS.reduce((o,k)=>{o[k]=S[k]; return o;},{})); read.innerHTML=esc(tx(`The small model ${e.stop?'stops':'does not stop'}; the answer is ${pcShare(e.share)} of the counted tokens`,`小模型${e.stop?'停了':'没有停'}；回答占算 loss 的 token 的 ${pcShare(e.share)}`)); return;}
      if(isP('dpo')){read.innerHTML=esc(tx(`Step ${bk+1} of ${BK.length}: ${L(BK[bk]).toLowerCase()}`,`第 ${bk+1} 步，共 ${BK.length} 步：${L(BK[bk])}`)); return;}
      const {ex}=stripParts(); const pv=mode==='guided'&&STEPS[step].id==='predict';
      if(ex==='plain'||(pv&&!pred)||(mode==='guided'&&step<SIX('predict'))){read.innerHTML=esc(tx(`The base model writes ${qTxt()}’s continuation`,`基础模型接着${qTxt()}往下写`)); return;}
      const c=counts(ex==='plain'?'toy':ex,pv?'all':S.count,pv?false:masked(S));
      read.innerHTML=esc(tx(`The answer is ${pcShare(c.share)} of the counted tokens`,`回答占算 loss 的 token 的 ${pcShare(c.share)}`));}}});

/* read-only engine hook: teasers/sft.js draws the home tile from it, and the tests read the page's numbers through it */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__sft=Object.freeze({
  vocab:()=>VOC.map(v=>v.k), tok:k=>tokS(k), mono:k=>isMono(k), pcShare, NPAR, K, LR, PRE, MAXG,
  gen:(o,gv)=>deep(genFor({...DEF,...(o||{})},gv||'sft')), genInfo:(o,gv)=>deep(genInfo(genFor({...DEF,...(o||{})},gv||'sft'))),
  respLoss:o=>{const s={...DEF,...(o||{})}; return sftRun(s.count,masked(s),s.steps).rl;}, nCounted:o=>{const s={...DEF,...(o||{})}; return sftRun(s.count,masked(s),s.steps).n;},
  bridge:()=>{const R=respLogp(sftRun('resp',false,60).m), B=respLogp(base()); return deep({rows:R.rows,sum:R.sum,ref:B.sum,refRows:B.rows});},
  counts:(ex,c,m)=>deep(counts(ex,c,m===true?'id':(m||''))), chAll:()=>deep(chAll()), chMin, drill:s=>deep(makeDrill(s)), exCases:()=>deep(exCases()),
  exits:()=>{const e=EXITS(); return deep({explainA:e.explain.map(q=>q.a),transferA:e.transfer.map(q=>q.a),text:[...e.explain,...e.transfer].map(q=>[q.q,...q.o,...q.n.filter(Boolean),q.why].join(' '))});},
  consts:Object.freeze({DEF:deep(DEF),CH_START:deep(CH_START),STEP_IDS:STEPS.map(s=>s.id),PRESETS:deep(PRESETS.map(p=>p.s)),EXS:deep(EXS),SPD:deep(SPD),CODE}),
  state:()=>deep({...S,mode,step,pred,bk,brow,tried4:[...tried4],tried5:[...tried5],chSubmitted,chReveal,i:PL.i,n:PL.g?PL.g.length:0,playing:PL.on,spd:PL.spd,sig:PL.sig,watched:[...PL.watched],stageBusy:stageT!==0})});
})();
