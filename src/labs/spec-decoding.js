/* Speculative decoding lab (batch 10, round 1): how guessing makes decode faster without changing the output, and when it stops paying.
   Simulated: a seeded simulator of draft-and-check rounds (i.i.d. acceptance α per guess, the papers' simplifying assumption), step times
   from the serving labs' step model (brief S2, as capstone/engine.js prices a step), and an exact demo of the acceptance rule on a
   5-token distribution. tests/spec-decoding_numbers.py recomputes every number independently.
   Names (brief S6): target model 目标模型, draft model draft 模型, draft length k 每轮猜的 token 数 k, acceptance rate α 接受率 α. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='spec-decoding';

/* ---------- engine (pure) ---------- */
/* H100 SXM: 989.5 TFLOP/s dense BF16 (the datasheet's 1,979 is with sparsity), 3.35 TB/s; 50% and 60% of peak are assumptions (S2) */
const PEAK=1979e12/2, BW=3.35e12, MFU=0.5, MBU=0.6, FS=PEAK*MFU, BS=BW*MBU;
const VOCAB=128256, CTX=1024;
/* shapes: meta-llama/llama-models models/sku_list.py (pinned in sources/b10src/spec-decoding); the 3.2 models tie the LM head to the embedding table */
function mdl(id,name,Lr,D,H,KVH,HD,FFN,tied){
  const PL=Lr*(2*D*D+2*D*KVH*HD+3*D*FFN+2*D)+D, PH=VOCAB*D;
  return {id,name,L:Lr,D,H,KVH,HD,FFN,tied,PL,PH,P:PL+PH+(tied?0:PH),W:2*(PL+PH),KV:2*Lr*KVH*HD*2};}
const NB='\u2011';
const MD={
  t8:mdl('t8',`Llama${NB}3.1${NB}8B`,32,4096,32,8,128,14336,false),
  t70:mdl('t70',`Llama${NB}3.1${NB}70B`,80,8192,64,8,128,28672,false),
  d1:mdl('d1',`Llama${NB}3.2${NB}1B`,16,2048,32,8,64,8192,true),
  d3:mdl('d3',`Llama${NB}3.2${NB}3B`,28,3072,24,8,128,8192,true)};
const GPUS={t8:1,t70:4};                       // the 70B on 4 H100s with tensor parallelism (131.4 GiB of BF16 weights do not fit on one)
const DRAFTS=['none','ngram','d1','d3'], KS=[1,2,3,4,5,6,7,8], BATCHES=[1,4,16,32,64,128], CURVE_B=[1,2,4,8,16,32,64,128];
/* α: the lab's own assumptions for each pair (no reachable source gives them); the document-Q&A scenario uses 0.6 for n-gram */
const ALPHA={t8:{ngram:0.3,d1:0.7,d3:0.8},t70:{ngram:0.3,d1:0.65,d3:0.75}};
const DOC_A=0.6;
const attn=(m,a,b)=>2*m.L*m.D*(b*(b+1)-a*(a+1));            // causal attention FLOPs for query positions a..b-1 (query i sees i+1 keys)
/* one step on n GPUs: B requests, each adding ntok tokens at positions ctx..ctx+ntok-1 */
function stepT(m,B,ntok,ctx,n){
  const fl=2*m.PL*B*ntok+2*m.PH*B*ntok+B*attn(m,ctx,ctx+ntok), kvB=m.KV*B*(ctx+ntok), by=m.W+kvB;
  const tc=fl/n/FS, tm=by/n/BS; return {t:Math.max(tc,tm),tc,tm,fl,by,kvB,tw:m.W/n/BS,tk:kvB/n/BS};}
const expTok=(a,k)=>a>=1?k+1:(1-Math.pow(a,k+1))/(1-a);   // 1 + α + α² + … + α^k
function compute(s){
  const T=MD[s.tgt], n=GPUS[s.tgt], plain=stepT(T,s.B,1,CTX,n);
  const r={s,T,n,plain,tPlain:plain.t,tpsPlain:1/plain.t};
  if(s.dr==='none'){Object.assign(r,{E:1,td:0,td1:0,v:plain,tRound:plain.t,speed:1,tps:1/plain.t}); return r;}
  let td=0, td1=0; if(s.dr!=='ngram'){for(let j=0;j<s.k;j++){const d=stepT(MD[s.dr],s.B,1,CTX+j,n); td+=d.t; if(!j) td1=d.t;}}
  const v=stepT(T,s.B,s.k+1,CTX,n), tRound=td+v.t, E=expTok(s.a,s.k);
  Object.assign(r,{E,td,td1,v,tRound,tps:E/tRound,speed:E*plain.t/tRound}); return r;}
/* the example round's cost at batch 1 (parts 3-4: the 8B with the 1B draft): its k draft steps, the check of k + 1 positions, and
   plain decode writing the same tokens one step each. FLOPs too: the round computes more than plain decode, yet takes less time */
function roundCost(R){const T=MD.t8, D=MD.d1, ds=[]; let flD=0;
  for(let j=0;j<R.k;j++){const d=stepT(D,1,1,CTX+j,1); ds.push(d.t); flD+=d.fl;}
  const v=stepT(T,1,R.k+1,CTX,1), p=stepT(T,1,1,CTX,1), n=R.acc+1, td=ds.reduce((a,b)=>a+b,0);
  return {ds,td,v,tv:v.t,tRound:td+v.t,n,tPlain:p.t,plain:n*p.t,flRound:v.fl+flD,flDraft:flD,flCheck:v.fl,flPlain:n*p.fl,dW:D.W};}
/* memory check (GiB per GPU): weights of both models, their KV cache, 2 GiB runtime overhead, against 80 GiB × 0.9 */
function memFit(s){const T=MD[s.tgt], n=GPUS[s.tgt], D=s.dr==='d1'||s.dr==='d3'?MD[s.dr]:null, tok=s.B*(CTX+s.k+1);
  const used=(2*T.P+(D?2*D.P:0)+tok*(T.KV+(D?D.KV:0)))/n+2*A.GiB; return {used,cap:80*A.GiB*0.9,ok:used<=80*A.GiB*0.9};}
/* seeded simulator (S5): mulberry32 as capstone/engine.js; each round draws until a rejection or k acceptances */
function mulberry(seed){let h=seed>>>0; return ()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};}
function rounds(a,k,n,seed){const rnd=mulberry(seed), acc=[]; let tot=0; for(let i=0;i<n;i++){let j=0; while(j<k&&rnd()<a) j++; acc.push(j); tot+=j+1;} return {acc,mean:tot/n};}
const SIM_N=2000, SIM_A=[0.6,0.7,0.8,0.9];
const simSeed=(a,k)=>1000+Math.round(a*100)*10+k;
const simCache=new Map();
function simMean(a,k){const key=a+'|'+k; if(!simCache.has(key)) simCache.set(key,rounds(a,k,SIM_N,simSeed(a,k)).mean); return simCache.get(key);}
/* the timeline: one request's first OUT output tokens, plain decode vs draft-and-check. Round i draws one number u from the golden-ratio
   sequence (u = frac((i + 1) × 0.618…)) and accepts j guesses when α^j > u ≥ α^(j+1): the same i.i.d. model, but spread evenly, so this
   short run sits near the average instead of on one lucky or unlucky seed (it must agree with the speed card beside it, R6.10) */
const OUT=24, PHI=0.6180339887498949;
function timeline(r){const s=r.s, plain=OUT*r.tPlain;
  if(s.dr==='none') return {plain,rounds:[],spec:plain,made:OUT};
  const list=[]; let made=0, t=0, i=0;
  while(made<OUT){i++; const u=(i*PHI)%1; let j=0; while(j<s.k&&Math.pow(s.a,j+1)>u) j++; const t0=t; t+=r.tRound; list.push({acc:j,t0,t1:t,from:made}); made+=j+1;}
  const last=list[list.length-1]; return {plain,rounds:list,spec:last.t1,made};}
/* the acceptance rule, exactly, on one position: p = target, q = draft (illustrative numbers); α here = Σ min(p, q) = 0.80 */
const PQ={p:[0.40,0.25,0.15,0.12,0.08],q:[0.55,0.15,0.20,0.05,0.05]};
function accRule(){const p=PQ.p, q=PQ.q, mn=p.map((x,i)=>Math.min(x,q[i])), al=mn.reduce((u,v)=>u+v,0), rz=p.map((x,i)=>Math.max(0,x-q[i])), rs=rz.reduce((u,v)=>u+v,0);
  const res=rz.map(x=>x/rs), out=mn.map((m,i)=>m+(1-al)*res[i]), acc=p.map((x,i)=>Math.min(1,x/q[i]));
  return {p,q,mn,al,res,out,acc,rej:1-al};}
const ACC_N=10000, ACC_SEED=11;
function accSample(mode){const R=accRule(), rnd=mulberry(ACC_SEED), cnt=[0,0,0,0,0], pick=(w)=>{const u=rnd(); let c=0; for(let i=0;i<w.length;i++){c+=w[i]; if(u<c) return i;} return w.length-1;};
  for(let i=0;i<ACC_N;i++){const x=pick(R.q);
    if(mode==='nocheck'){cnt[x]++; continue;}
    if(mode==='greedy'){cnt[0]++; continue;}          // greedy: the draft's top token is checked against the target's top token; both are token 1 here
    if(rnd()<R.acc[x]) cnt[x]++; else cnt[pick(R.res)]++;}
  return cnt.map(c=>c/ACC_N);}
let accCache={};
const accFreq=m=>accCache[m]||(accCache[m]=accSample(m));

/* ---------- the example round (示例): one word stands for one token ---------- */
const EXW={en:{ctx:'Once upon a time, there',t:['was','a','black','dog','who','lived','near','the','old'],
    d:{2:['was','a'],4:['was','a','little','dog'],8:['was','a','black','dog','who','slept','in','a']},x:{4:['girl','named'],8:['under','the','barn']}},
  zh:{ctx:'很久很久以前，',t:['有','一只','黑','狗','住','在','山','边','。'],
    d:{2:['有','一只'],4:['有','一只','小','狗'],8:['有','一只','黑','狗','住','进','森林','里']},x:{4:['猫','住'],8:['了','深处','的']}}};
// t: what the target model writes; d[k]: the draft's guesses in the k-round; x[k]: the target's own tokens after the first mismatch in that
// round (the same step computes them from the rejected guess, so they are not used)
const ROUNDS={2:{k:2,acc:2},4:{k:4,acc:2},8:{k:8,acc:5}};    // the rounds the replay shows for k = 2, 4, 8 (example rounds, not averages)
const nFrames=R=>R.k+1+Math.min(R.acc+1,R.k)+1;
const SPEEDS={slow:1.6,med:1.0,fast:0.5};                  // seconds per frame; 中 plays the k = 4 round (9 frames) in 9 s

/* ---------- formatting ---------- */
const f=(v,d)=>A.fmt.num(v,d);
const fd=(v,d)=>{const t=f(v,d); if(!d) return t; const i=t.indexOf('.'); return i<0?t+'.'+'0'.repeat(d):t+'0'.repeat(Math.max(0,d-(t.length-i-1)));};   // fixed decimals: 16.0, not 16 (R6.6)
const msT=sec=>{const v=sec*1e3; return (v>=100?f(v,0):v>=10?fd(v,1):fd(v,2))+' ms';};
const msP=(a,b)=>msT(a)===msT(b)&&Math.abs(a-b)>1e-9?fd(a*1e3,3)+' ms':msT(a);   // two times that round alike side by side: one more digit
const tms=sec=>{const v=sec*1e3; return v>=100?f(v,0):v>=10?fd(v,1):fd(v,2);};
const near1=v=>Math.abs(v-1)<0.005&&Math.abs(v-1)>1e-9;
const xT=v=>near1(v)?tx(`${fd(v,3)} times`,`${fd(v,3)} 倍`):tx(`${fd(v,2)} times`,`${fd(v,2)} 倍`);   // 0.998, not a rounded 1.00 that hides the side (R6.6)
const pc=(v,d=0)=>fd(v*100,d)+'%';
const aT=a=>fd(a,2);
const aS=a=>String(+a.toFixed(2));   // in a sentence: 0.7, 0.75
const DEF=(text,tip)=>A.ui.def(text,tip);
const M=s=>A.ui.math(s);
let moreOpen={};   // which folds of the cards are open (the rail re-renders on every lever)
const MORE=(key,sum,html)=>`<details class="spd-more" data-more="${key}"${moreOpen[key]?' open':''}><summary>${sum}</summary><div class="spd-more-b">${html}</div></details>`;
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const live=id=>{const c=A.catalog, l=c&&c.LAB&&c.LAB[id]; return !!l&&(c.labStatus?c.labStatus(l):l.st)==='live';};
const labRef=(id,en,zh)=>live(id)&&A.shell.labLink?A.shell.labLink(id):esc(tx(en,zh));
const cite=(url,txt)=>`<a href="${url}" target="_blank" rel="noopener">${txt}</a>`;
const EQ=(name,lines)=>`<div class="spd-eq"><div class="spd-eq-n">${name}</div>${lines.map((l,i)=>`<div class="spd-fx">${M((i?'= ':'')+l)}</div>`).join('')}</div>`;
const VLLM='https://github.com/vllm-project/vllm/blob/ac8c2130a8a6ecc4940f6c5433cb2ba15e983aa7/';
const SGL='https://github.com/sgl-project/sglang/blob/6d8ddaf1d786c43cc84c6292f930c174c07e4b02/python/sglang/srt/speculative/spec_info.py';
const HFT='https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/generation/utils.py';
const DSV3='https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md';
const EAGLE='https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/README.md';
const SKU='https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py';
const SBOOK='https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/inference.md';

const DNAME={none:{en:'No draft',zh:'不猜'},ngram:{en:'n-gram lookup',zh:'n-gram 查找'},d1:{en:`1B draft`,zh:'1B draft 模型'},d3:{en:`3B draft`,zh:'3B draft 模型'}};
const dName=d=>L(DNAME[d]);
const tName=t=>MD[t].name+(GPUS[t]>1?tx(` on ${GPUS[t]} H100s`,`（${GPUS[t]} 张 H100）`):'');

/* ---------- definitions (R2.8) ---------- */
const D_TGT=()=>tx(`The target model is the large model whose output we want. Here it is ${MD.t8.name}.`,`目标模型是我们要它的输出的那个大模型，这里是 ${MD.t8.name}。`);
const D_DRAFT=()=>tx(`The draft model is a small model that guesses a few tokens ahead for the target model. Here it is ${MD.d1.name}, about a sixth of the 8B’s size. It uses the same tokenizer, a 128,256-token vocabulary, so the target model can check its tokens directly.`,`draft 模型是替目标模型先猜几个 token 的小模型，这里是 ${MD.d1.name}，大小约是 8B 的六分之一。它和目标模型用同一套 tokenizer（128,256 个 token 的词表），所以目标模型能直接检查它猜的 token。`);
const D_K=()=>tx('The draft length k is how many tokens the draft model writes in one round. The target model then runs once to check all k.','每轮猜的 token 数 k：draft 模型一轮先写出几个 token。写完以后，目标模型跑一次，把这 k 个都检查了。');
const D_ALPHA=()=>tx('The acceptance rate α is the chance that the next guess is accepted, given that all the guesses before it were.','接受率 α：前面的猜测都被收下时，下一个猜测被收下的概率。');
const D_MASK=()=>tx('Causal mask: inside the model, the attention at a position may look only at that position and the ones before it. Nothing after it can change its result, so known tokens can all be computed in one step.','因果掩码：模型里的注意力在一个位置上，只能看这个位置和它前面的位置。后面的 token 改变不了它的结果，所以已经给定的 token 可以在一步里一起算。');
const D_MB=()=>tx('Memory-bound: the step’s time goes to reading the weights and the KV cache from GPU memory, and most of the arithmetic units sit idle.','受显存带宽限制（memory-bound）：一步的时间花在从显存读权重和 KV cache 上，大部分算力闲着。');
const D_NGRAM=()=>tx('n-gram lookup (prompt lookup): instead of a model, the guesser looks for the last few tokens earlier in the prompt and proposes what followed them there. It costs almost no GPU time and works when the answer repeats the input, as in document Q&A or code edits.','n-gram 查找（prompt lookup）：不用模型来猜，而是在 prompt 里找到和最后几个 token 相同的片段，把它后面的内容当作猜测。它几乎不占 GPU 时间，回答会照抄输入时（文档问答、改代码）最有用。');
const D_BONUS=()=>tx(`When every guess is accepted, the target model’s step has already computed the token after the last guess, so that one comes free as well: ${M('k+1')} tokens from one step.`,`所有猜测都被收下时，目标模型这一次已经算出了最后一个猜测之后的 token，它也直接收下：跑一次得到 ${M('k+1')} 个 token。`);
const D_STEP=()=>tx('Step model, as in the serving labs: a step takes the longer of its compute time and its memory time. This lab assumes compute at 50% of the H100’s 989.5 TFLOP/s dense BF16 peak and memory reads at 60% of its 3.35 TB/s.','步时模型和 Serving 各 Lab 一样：一步的时间取计算时间和读显存时间中较长的那个。本 Lab 假设计算按 H100 稠密 BF16 峰值 989.5 TFLOP/s 的 50% 算，读显存按 3.35 TB/s 的 60% 算。');

const DESCS=()=>({
  guided:tx('Eight short parts. See why checking several tokens costs about as much as making one, watch one round of guessing and checking, then find out why the output stays the same and when it stops paying.','一共 8 个部分。先看为什么检查几个 token 和生成一个差不多贵，再看一轮“猜、检查”怎么走，最后弄清输出为什么不变、什么时候不再划算。'),
  sandbox:tx('Pick a service and a guesser, and see how many tokens per second each request gets.','选一种服务和一种猜法，看看每个请求每秒能拿到多少 token。'),
  challenge:tx('Llama‑3.1‑70B on 4 H100s serving 128 requests at once: pick the guesser and k that give the most tokens per second, at least 1.5 times plain decode.','4 张 H100 上的 Llama‑3.1‑70B 同时服务 128 个请求。选一种猜法和 k，让每秒的 token 数最多，至少是普通 decode 的 1.5 倍。')});
const OUTCOME=()=>tx('estimate how many tokens one step of the large model yields when a small model guesses ahead, explain why the output is unchanged, and say when guessing stops paying.','估算让小模型先猜时，大模型跑一次能出几个 token，讲清输出为什么不变，并说出什么时候投机解码不再划算。');
const FIDPOP=()=>tx(
 `<p><b>Toy run:</b> a seeded simulator of draft-and-check rounds, with step times from the serving labs’ step model.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Each step costs the longer of its compute time and its memory time. As in the serving labs, this lab assumes compute at 50% of 989.5 TFLOP/s and memory reads at 60% of 3.35 TB/s (H100 SXM, BF16).</li><li>A round: k draft steps of one token per request, then one target step over ${M('k+1')} tokens per request. n-gram lookup costs no GPU time. This lab assumes 1,024 tokens of context per request; its KV cache is read every step. The draft model keeps its own KV cache for every request: each draft step reads it, and it is counted in GPU memory.</li><li>Acceptance: each guess is accepted with probability α if all before it were, independently (the papers’ simplifying assumption). The rounds are drawn with a seeded generator (mulberry32).</li><li>The acceptance rule is computed exactly on a 5-token example.</li></ul></div><div><div class="eyebrow">Simplified</div><ul><li>In the example rounds one word stands for one token.</li><li>The timeline’s rounds are not drawn at random: each possible outcome appears about as often as its chance says, so a short run stays near the average.</li><li>This lab assumes the α of each pair: 8B with n-gram lookup 0.3, with the 1B 0.7, with the 3B 0.8; 70B with the 1B 0.65, with the 3B 0.75. The real α depends on the two models and the text, and is measured on real traffic.</li><li>The draft model’s catch-up on the tokens accepted last round, sampling overhead and kernel launch time are left out; real drafts cost more than this step model says.</li><li>The 70B runs on 4 H100s with tensor parallelism and its draft is split the same way; communication between GPUs is left out, so in this lab the speedup does not depend on the number of GPUs (${labRef('multi-gpu-serving','Multi-GPU serving','多卡推理')} covers the communication between GPUs).</li><li>Every setting fits in GPU memory (checked).</li></ul></div></div><p class="small muted">Sources: model shapes, ${cite(SKU,'llama-models sku_list.py')}; the acceptance rule, ${cite(HFT,'transformers generation/utils.py')} and vLLM’s rejection sampler; long contexts, ${cite(SBOOK,'scaling book, appendix D')}; how production systems use it and why their outputs can differ slightly, ${cite(VLLM+'docs/features/speculative_decoding/README.md','vLLM docs')}; the papers by Leviathan et al. (arXiv 2211.17192) and Chen et al. (arXiv 2302.01318) are cited by title.</p>`,
 `<p><b>玩具实跑：</b>一个带种子的“猜、检查”轮次模拟器，每一步的耗时按 Serving 各 Lab 的步时模型算。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>一步的耗时取计算时间和读显存时间中较长的那个。和 Serving 各 Lab 一样，本 Lab 假设计算按 989.5 TFLOP/s 的 50%、读显存按 3.35 TB/s 的 60%（H100 SXM，BF16）。</li><li>一轮：draft 模型走 k 步，每步给每个请求出 1 个 token；然后目标模型走一步，每个请求算 ${M('k+1')} 个 token。n-gram 查找不占 GPU 时间。本 Lab 假设每个请求有 1,024 个 token 的上下文，每一步都要读它的 KV cache。draft 模型也给每个请求存一份自己的 KV cache：它每走一步都要读，显存里也算上了。</li><li>接受：前面的猜测都被收下时，下一个猜测以概率 α 被收下，各次相互独立（论文用的简化假设）。轮次用带种子的随机数生成（mulberry32）。</li><li>接受规则在一个 5 个 token 的例子上精确计算。</li></ul></div><div><div class="eyebrow">简化了什么</div><ul><li>示例轮次里，一个词当一个 token。</li><li>时间线上的几轮不是随机抽的：每种结果出现的次数和它的概率成比例，所以一小段也接近平均水平。</li><li>本 Lab 假设每一对模型的 α：8B 配 n-gram 查找 0.3、配 1B 0.7、配 3B 0.8；70B 配 1B 0.65、配 3B 0.75。真实的 α 取决于两个模型和文本，要在真实流量上测。</li><li>没算 draft 模型补读上一轮收下的 token、采样的开销和 kernel 启动时间；真实的 draft 模型比这个步时模型算的更贵。</li><li>70B 用 4 张 H100 做张量并行，draft 模型也同样切开；没算卡之间的通信，所以在这个 Lab 里，加速倍数和卡数无关（卡之间的通信见${labRef('multi-gpu-serving','Multi-GPU serving','多卡推理')}）。</li><li>每种设置都放得进显存（已核对）。</li></ul></div></div><p class="small muted">来源：模型形状见 ${cite(SKU,'llama-models sku_list.py')}；接受规则见 ${cite(HFT,'transformers generation/utils.py')} 和 vLLM 的 rejection sampler；长上下文见 ${cite(SBOOK,'scaling book 附录 D')}；生产系统怎么用、输出为什么可能有细微差别，见 ${cite(VLLM+'docs/features/speculative_decoding/README.md','vLLM 文档')}；Leviathan 等（arXiv 2211.17192）和 Chen 等（arXiv 2302.01318）的论文只引题目。</p>`);
const PRIPOP=()=>tx(`<p>Decode makes one token per step, and each step reads all the weights from GPU memory. At a small batch that reading, not the arithmetic, sets the step’s time: see ${labRef('prefill-decode','Prefill and decode','Prefill 和 decode')}. Sampled outputs and temperature are in ${labRef('sampling','Sampling','采样')}.</p>`,
  `<p>decode 每一步只出一个 token，每一步都要从显存把全部权重读一遍。batch 小的时候，决定一步耗时的是读显存，而不是计算，见 ${labRef('prefill-decode','Prefill and decode','Prefill 和 decode')}。按概率采样和 temperature 见 ${labRef('sampling','Sampling','采样')}。</p>`);

/* ---------- state ---------- */
const DEFAULT={tgt:'t8',dr:'d1',k:4,B:1,a:0.7};
let S={...DEFAULT}, mode='guided', step=0, root=null, app=null, pred=null, saved=null;
let tried={}, observed={}, costN=1, accMode='sample', userAct=false, annT=0, stageT=0, shownPrev={};
let chSubmitted=false, chReveal=false, drill=null, drillOpen=false;
let RP={on:false,raf:0,f:0,key:'',last:0,acc:0,quick:false}, rpSpd='med', rpR=ROUNDS[4];
const watched=new Set();
let nPlays=0;
const CH={tgt:'t70',B:128}, CH0={dr:'d3',k:4}, CH_GOAL=1.5;   // a target and batch the guided parts never priced (R3.8)

/* ---------- guided parts ---------- */
const PRED=['a2','a34','a42'];
const predTxt=v=>({a2:tx('Under 2.5 tokens','不到 2.5 个 token'),a34:tx('2.5 to 4 tokens','2.5 到 4 个 token'),a42:tx('4 to 5 tokens','4 到 5 个 token')}[v]);   // the three options cover 1 to 5, the possible range
const COSTN=[1,5,80,640], COSTROWS=[1,5,80,128,640], COSTB={1:[1,1],5:[1,5],80:[16,5],128:[128,1],640:[128,5]};   // tokens in one step → [requests, tokens each]; 128 = plain decode of 128 requests, the baseline of 640
const sid=()=>STEPS[step].id;
const seen=id=>mode!=='guided'||step>=STEPS.findIndex(s=>s.id===id);
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'One token per step: can one step give several?',zh:'一步只出一个 token，能不能一步出好几个？'},
  body:()=>{const r=compute({...DEFAULT,dr:'none'});
   return tx(`<p>A chat service runs ${MD.t8.name} on one H100. In decode it writes the answer one token per step, and every step reads ${fd(MD.t8.W/1e9,1)} GB of weights from GPU memory: all the weights except the embedding table, that is every layer and the LM head. With one user that reading takes almost the whole step, ${DEF('about '+msT(r.tPlain),D_STEP())} (the chart).</p><p>The requirement is that the output does not change. With greedy decoding the answer must match, word for word, what the 8B writes one token at a time on its own; with sampling, every token must have the same probabilities as with the 8B alone. So: can one 8B step give you several tokens?</p>`,
     `<p>一个聊天服务在一张 H100 上跑 ${MD.t8.name}。decode 时它一步只写一个 token，每一步都要从显存读 ${fd(MD.t8.W/1e9,1)} GB 权重，也就是 embedding 表以外的全部权重：每一层和 LM head。只有一个用户时，读权重几乎占满了整步，${DEF('约 '+msT(r.tPlain),D_STEP())}（见图）。</p><p>要求是不改变输出。贪心解码时，回答要和 8B 自己一个一个写出来的一字不差；按概率采样时，每个 token 的概率要和 8B 自己写时一样。那么，8B 跑一步，能不能得到好几个 token？</p>`);}},
 {id:'cost',title:{en:'A few more tokens per step cost almost nothing',zh:'一步多算几个 token，几乎不多花时间'},lead:{en:'Does one step with 5 tokens take longer than one with 1?',zh:'一步算 5 个 token，比算 1 个慢多少？'},expect:()=>!!(tried.cost&&tried.cost.has(5)&&tried.cost.has(640)),showMe:'cost',
  body:()=>tx(`<p>Decode writes one token at a time only because the next token is unknown until the one before it is chosen. When the tokens are known in advance (the prompt in prefill, or tokens a small model guessed), one step computes them all and gives each position’s next token. Each position sees only the tokens before it (the ${DEF('causal mask',D_MASK())}), so computing them together gives the same result as one by one.</p><p>And a few more tokens cost almost no time: a step reads the weights once, however many tokens it holds. With few requests that reading fills almost the whole step, and the extra arithmetic finishes meanwhile: the step is ${DEF('memory-bound',D_MB())}.</p>${costLevers()}`,
    `<p>decode 一次只写一个 token，是因为下一个 token 要等上一个选出来才知道。要是几个 token 事先已经知道（比如 prefill 时的 prompt，或小模型先猜好的 token），模型跑一步就能把它们一起算完，得到每个位置的下一个 token。这是因为每个位置只看得到它前面的 token（这叫${DEF('因果掩码',D_MASK())}），一起算和一个一个算，结果一样。</p><p>多放几个 token 也几乎不多花时间：一步只读一遍权重，不管放了几个 token。请求少的时候，读权重几乎占满整步，多出来的计算在读的同时就做完了，这一步${DEF('受显存带宽限制',D_MB())}。</p>${costLevers()}`),
  obs:()=>obsCost()},
 {id:'round',title:{en:'One round: a small model guesses, the large one checks once',zh:'一轮：小模型猜，大模型一次检查'},lead:{en:'The 1B guesses 4 tokens one by one; the 8B checks all 4 in one step.',zh:'1B 先一个一个猜 4 个，8B 跑一步把 4 个都检查了。'},expect:()=>watched.has(rkey(ROUNDS[4])),showMe:'round',
  body:()=>tx(`<p>With one user, an 8B step is spent almost entirely reading the weights, and a few more positions add almost no time. So let a small model guess the next few tokens and have the 8B check them all in one step: every right guess is one more token from that step. A round has three stages, marked ① ② ③ on the figure:</p><ol class="spd-3s"><li><b>① The draft model guesses.</b> The small model, called the ${DEF('draft model',D_DRAFT())}, writes ${DEF('k',D_K())} = 4 tokens one by one.</li><li><b>② The target model checks, in one step.</b> The 8B is the ${DEF('target model',D_TGT())}: its output is the one we want. Given the context’s last token, “there”, and the 4 guesses, one step computes its own token at each of these 5 positions.</li><li><b>③ Keep.</b> From the left, every guess that matches the 8B is kept; at the first mismatch the 8B’s own token is kept instead, and the guesses after it are thrown away.</li></ol><p>This way to speed up inference is called <b>speculative decoding</b>: the 8B is unchanged, and so is its output. It does not save computation, since the thrown-away guesses and the draft model’s 4 small steps are computed too; what it saves is reads of the 8B’s weights.</p><p class="small muted">This is greedy decoding (the 8B takes its most likely token); how sampled outputs are checked comes in “Why the output does not change”.</p>${rpButtons()}`,
    `<p>只有一个用户时，8B 一步几乎都在读权重，多算几个位置几乎不加时间。所以先让小模型往后猜几个 token，8B 跑一步一起检查，猜对几个就多出几个 token。一轮分三步，对应图上的 ①②③：</p><ol class="spd-3s"><li><b>① draft 模型猜。</b>小模型叫 ${DEF('draft 模型',D_DRAFT())}，一个一个写出 ${DEF('k',D_K())} = 4 个 token。</li><li><b>② 目标模型一次检查。</b>8B 叫${DEF('目标模型',D_TGT())}，我们要的是它的输出。上文末尾的“，”和 4 个猜测已经给定，它跑一步就算出这 5 个位置上自己会写的 token。</li><li><b>③ 留下。</b>从左往右，和 8B 一样的猜测收下；第一个不一样的换成 8B 自己的 token，后面的都丢掉。</li></ol><p>这种推理加速方法叫<b>投机解码</b>，不用改 8B，输出也不变。它并不省计算，被丢掉的猜测和 draft 模型的 4 小步都要算；省下的是 8B 读权重的次数。</p><p class="small muted">这里按贪心解码讲（8B 取概率最高的 token）；采样时怎么检查，见“输出为什么不变”。</p>${rpButtons()}`),
  obs:()=>obsRound()},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Each guess kept with chance 0.8, 4 guesses per round: how many tokens per target step, on average?',zh:'每个猜测被收下的机会是 0.8、每轮猜 4 个：目标模型跑一次平均出几个 token？'},gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(v=>`<button type="button" class="choice" data-pred="${v}" aria-pressed="${pred===v}">${esc(predTxt(v))}</button>`).join('')}</div>`;
   return tx(`<p>The draft model guesses 4 tokens per round. One target step then yields at least 1 token (the first guess is wrong: only the target’s own) and at most 5 (all 4 kept, plus its own). The chance that a guess is kept when all the guesses before it were is the ${DEF('acceptance rate α',D_ALPHA())}. It depends on the draft model and the text; this part and the next take 0.8. A guess after a rejected one is thrown away even if it was right.</p><p>On average, how many tokens does one target step yield?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before the next part shows the result.</p>':''}`,
     `<p>draft 模型每轮猜 4 个 token。目标模型跑一次，最少得到 1 个 token（第一个就猜错，只剩它自己的 1 个），最多 5 个（4 个全被收下，再加它自己的 1 个）。前面的猜测都被收下时，下一个猜测被收下的概率叫${DEF('接受率 α',D_ALPHA())}，它取决于 draft 模型和文本；这一部分和下一部分先按 0.8 算。排在被拒绝的猜测后面的，即使猜对了也会被丢掉。</p><p>平均下来，目标模型跑一次能得到几个 token？</p>${ch}${pred?'<p class="muted small">已记下。下一部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'expect',title:{en:'Tokens per target step',zh:'目标模型跑一次出几个 token'},lead:{en:'At most one more than the guesses; the average adds up each token’s chance.',zh:'最多比猜的多 1 个；平均要把每个 token 出现的机会加起来。'},expect:()=>!!(tried.expect&&tried.expect.has(2)&&tried.expect.has(8)),showMe:'expect',
  body:()=>{const E2=fd(expTok(0.8,2),2), eq=EQ(tx('tokens per target step','目标模型跑一次的 token 数')+' '+M('E'),['1 + \\alpha + \\alpha^2 + \\cdots + \\alpha^k','\\frac{1 - \\alpha^{k+1}}{1 - \\alpha}']);
   return tx(`<p>The guesses per round and the tokens per target step are two different numbers. With 2 guesses per round, one target step yields at least 1 token (its own); if guess 1 is kept (chance 0.8), one more; if guesses 1 and 2 are both kept (chance ${M('0.8 \\times 0.8 = 0.64')}), one more again. On average that is ${M('1 + 0.8 + 0.64 = '+E2)} tokens, at most 3.</p><p>The more you guess, the harder it is for the later guesses to survive, so each extra guess adds less (see the chart).</p>${MORE('expect','The same sum for any k',`<p>With k guesses per round and acceptance rate ${M('\\alpha')}, adding up the chances the same way gives:</p>${eq}`)}<p>${kLead()}</p>${levers('expect')}`,
    `<p>每轮猜几个，和目标模型跑一次得到几个，是两个数。每轮猜 2 个时，目标模型跑一次至少得到 1 个 token（它自己的）；猜测 1 被收下（机会是 0.8），就多 1 个；猜测 1 和 2 都被收下（机会是 ${M('0.8 \\times 0.8 = 0.64')}），再多 1 个。平均是 ${M('1 + 0.8 + 0.64 = '+E2)} 个，最多 3 个。</p><p>猜得越多，越往后的猜测越难留下，多猜的收益越来越小（见图）。</p>${MORE('expect','展开：任意 k 时的公式',`<p>每轮猜 k 个、接受率为 ${M('\\alpha')} 时，同样把机会加起来：</p>${eq}`)}<p>${kLead()}</p>${levers('expect')}`);},
  obs:()=>obsExpect()},
 {id:'same',title:{en:'Why the output does not change',zh:'输出为什么不变'},lead:{en:'Every token kept is one the target model would write itself.',zh:'留下的每个 token，都是目标模型自己会写的。'},expect:()=>!!(tried.same&&tried.same.has('sample')&&tried.same.has('nocheck')),showMe:'same',
  body:()=>tx(`<p>Every token that is kept is one the target model would have written itself. With greedy decoding a guess is kept only if it is the target’s most likely token. With sampling, when the draft model gives a word more probability than the target does, a guess of that word is kept only part of the time and rejected otherwise; after a rejection the target model redraws among the words it likes more than the draft does. Every word then comes out as often as when the target writes alone.</p><p>With the acceptance rule, the chart gives each word two bars. The left one is the draft model’s probability. The right one is how often the word comes out: green for accepted guesses, grey for redraws, and together they reach the thick line, the target model’s probability.</p>${MORE('same',tx('How the acceptance rule works','展开：接受规则怎么算'),`<p>With sampling, checking whether the guess is the top word is not enough; an acceptance rule decides. For a guess ${M('x')}, let ${M('p(x)')} be the target model’s probability and ${M('q(x)')} the draft model’s. If the draft model gives the word no more probability than the target does, the guess is always kept; if it gives more, the guess is kept only with chance ${M('p(x) / q(x)')}. Together: keep it with probability ${M('\\min\\left(1, p(x) / q(x)\\right)')}. If it is rejected, the target model draws a new word itself, only among the words it likes more than the draft model does, each in proportion to the difference ${M('\\max(0, p - q)')} (rescaled to sum to 1). So a word’s final chance has two parts: being kept as a guess, plus being drawn again after some guess was rejected. The two parts add up to exactly the target’s ${M('p')} (Leviathan et al., 2023; Chen et al., 2023).</p>`)}${levers('same')}`,
    `<p>留下的每个 token，都是目标模型自己本来就会写的。贪心解码时，猜测要和目标模型概率最高的 token 一样才收下。按概率采样时，如果 draft 模型给某个词的概率比目标模型高，这个词被猜到时只按一定比例收下，否则就拒绝；拒绝后，由目标模型在它比 draft 模型更看好的词里重抽。这样，每个词出现的机会就和目标模型自己写时一样。</p><p>按接受规则采样时，图里每个词有两根条。左边一根是 draft 模型给它的概率；右边一根是它出现的频率，绿段来自被收下的猜测，灰段来自重抽，两段合起来正好到粗横线，也就是目标模型的概率。</p>${MORE('same',tx('How the acceptance rule works','展开：接受规则怎么算'),`<p>按概率采样时，只看猜测是不是概率最高的词就不够了，要用一条接受规则来判断。对一个猜测 ${M('x')}，记目标模型给它的概率为 ${M('p(x)')}，draft 模型给的为 ${M('q(x)')}。draft 模型给这个词的概率不比目标模型高时，猜测一定收下；比目标模型高时，只按 ${M('p(x) / q(x)')} 的概率收下。合起来，就是以 ${M('\\min\\left(1, p(x) / q(x)\\right)')} 的概率收下。被拒绝时，目标模型自己重新抽一个词：只在目标模型比 draft 模型更看好的那些词里抽，每个词的机会按差额 ${M('\\max(0, p - q)')} 分配（先缩放到总和为 1）。这样，一个词最后出现的概率有两部分：它作为猜测被收下，加上别的猜测被拒绝后它被重新抽到。两部分加起来正好是目标模型的 ${M('p')}（Leviathan 等，2023；Chen 等，2023）。</p>`)}${levers('same')}`),
  obs:()=>obsSame()},
 {id:'batch',title:{en:'When it stops paying',zh:'什么时候不再划算'},lead:{en:'The busier the server, the smaller the speedup.',zh:'服务越忙，加速越小。'},expect:()=>!!(tried.batch&&tried.batch.has(1)&&tried.batch.has(128)),showMe:'batch',
  body:()=>{const g=d3big(); return tx(`<p>With many requests in a step, guessing gets dear: the draft model’s weights are small, but it keeps a KV cache for every request and reads all of it at every step; and checking that many tokens at once fills the 8B’s arithmetic. So the busier the server, the smaller the speedup, until guessing makes it slower.</p><p>The chart compares three guessers: the 1B and 3B draft models, and ${DEF('n-gram lookup',D_NGRAM())}, which uses no model and copies its guess from the prompt. From here on each guesser has its own acceptance rate: 1B ${aS(ALPHA.t8.d1)}, 3B ${aS(ALPHA.t8.d3)}, n-gram ${aS(ALPHA.t8.ngram)}.</p>${MORE('batch',tx('The two costs in numbers, and how to read the timeline','展开：两笔开销各有多大，下面的时间线怎么看'),`<p>Speculative decoding (guess, then check) has two costs, and both grow with the number of requests in a step (the batch). The first is the draft model’s steps. With one request per step, the 1B guesses 4 tokens in about ${pct16()} of an 8B step; at 128 requests the 3B takes ${msT(g.td)} to guess 4, ${fd(g.td/g.tPlain,1)} times a plain 8B step (${msT(g.tPlain)}), because its KV cache per token, ${f(MD.d3.KV/1024,0)} KiB, is close to the 8B’s ${f(MD.t8.KV/1024,0)} KiB. The second is the check: with k = 4 every request brings 5 tokens to it; from about ${bCB()} requests on, the arithmetic units fill up and the 8B’s check step gets longer too.</p><p>The chart shows tokens per second compared with plain decode, for each guesser. The timeline below it shows how long one request’s first ${OUT} output tokens take at the current batch: the top row is plain decode, one target step per token; the bottom row is guess-then-check, with red for draft time spent on guesses that get thrown away.</p>`)}${levers('batch')}`,
    `<p>一步里的请求一多，猜就变贵了：draft 模型权重虽小，也要给每个请求存一份 KV cache，每走一步都全读一遍；8B 一次检查那么多 token，也会把算力占满。所以服务越忙，加速越小，最后反而变慢。</p><p>图里比了三种猜法：1B、3B 两个 draft 模型，和不用模型、直接从 prompt 里抄猜测的 ${DEF('n-gram 查找',D_NGRAM())}。下面换成每种猜法各自的接受率：1B ${aS(ALPHA.t8.d1)}、3B ${aS(ALPHA.t8.d3)}、n-gram ${aS(ALPHA.t8.ngram)}。</p>${MORE('batch',tx('The two costs in numbers, and how to read the timeline','展开：两笔开销各有多大，下面的时间线怎么看'),`<p>投机解码（先猜再检查）有两笔开销，一步里的请求越多（batch 越大），两笔都越大。第一笔是 draft 模型的那几步。一步只有 1 个请求时，1B 猜 4 个约是 8B 走一步的 ${pct16()}；128 个请求时，3B 猜 4 个要 ${msT(g.td)}，是 8B 普通 decode 一步（${msT(g.tPlain)}）的 ${fd(g.td/g.tPlain,1)} 倍，因为它每个 token 的 KV cache 有 ${f(MD.d3.KV/1024,0)} KiB，和 8B 的 ${f(MD.t8.KV/1024,0)} KiB 差不多。第二笔是检查：每轮猜 4 个时，每个请求给检查带来 5 个 token；请求多到约 ${bCB()} 个以后，算力被占满，8B 检查的那一步也开始变长。</p><p>图里画的是每种猜法下每秒的 token 数是普通 decode 的几倍。下面的时间线是在当前请求数下，一个请求的前 ${OUT} 个输出 token 要多久：上一行是普通 decode，目标模型一步出一个；下一行是先猜再检查，红色是花在被丢掉的猜测上的 draft 时间。</p>`)}${levers('batch')}`);},
  obs:()=>obsBatch()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},
  body:()=>{const b1=compute({...DEFAULT}), b128=compute({...DEFAULT,B:128,dr:'d3',a:0.8});
   return tx(`<ul><li><b>One step of the large model can yield several tokens.</b> With one user, an 8B step is spent almost entirely reading the weights, so checking a few guesses adds almost no time. With the 1B guessing 4 tokens per round and each guess kept 7 times in 10, an 8B step yields ${fd(b1.E,2)} tokens on average; each round also spends about ${f(b1.td*1e3,0)} ms on the 1B’s 4 guesses, so the speed is ${xT(b1.speed)} plain decode’s.</li><li><b>The output does not change.</b> Every token kept is one the target model would write itself; a wrong guess only costs time, never quality.</li><li><b>The busier the server, the less it pays.</b> With many requests per step, the draft model’s steps get dearer and the tokens to check fill the GPU’s arithmetic. At 128 requests the 3B draft makes the 8B slower: ${xT(b128.speed)} plain decode’s speed.</li></ul>`+
     MORE('rm-long','Long contexts are different',`<p>This lab assumes 1,024 tokens of context per request. Long contexts differ: every step reads the whole context’s KV cache, and that is most of what it reads. Plain decode reads it once per token; checking ${M('k+1')} tokens in one step also reads it only once, so the more guesses are accepted, the fewer reads.</p>`)+
     MORE('rm-prod','How production systems use it',`<p>vLLM supports draft models, n-gram lookup, EAGLE and MTP, and its docs recommend speculative decoding for memory-bound workloads with medium to low request rates. EAGLE does not run a separate full model: a small draft head on the target model guesses from the target’s own features (its second-to-last layer), so it guesses well and costs little. SGLang’s methods include EAGLE, EAGLE3, a separate draft model and n-gram. DeepSeek‑V3 ships one MTP module, extra layers trained to predict a further token, that can serve as its own draft. In real systems, floating-point error and changing batch sizes can make outputs differ slightly; the algorithm itself does not change the output.</p><p>Other ways to shorten a memory-bound decode step combine with guessing: smaller weights read fewer bytes per step (${labRef('quantization','Quantization','量化')}).</p>`)+
     MORE('rm-mis','Three common misconceptions',`<p>✗ “A wrong guess makes the answer worse.” A wrong guess is rejected and replaced by the target model’s own token; only speed suffers.</p><p>✗ “The larger k, the faster.” Each extra guess adds a draft step to every round, but the later a guess, the harder it is to keep: at an acceptance rate of 0.8, guess 8 survives only about ${pc(Math.pow(0.8,8))} of the time. Past a few guesses the extra time outweighs the extra tokens.</p><p>✗ “It is just as fast at any batch.” At a large batch every draft step reads more KV cache and the check is no longer free, so the speedup shrinks or turns into a slowdown.</p>`)+
     `${A.shell.toScenarios()}`,
     `<ul><li><b>大模型走一步，能出好几个 token。</b>只有一个用户时，8B 走一步的时间几乎都花在读权重上，多检查几个猜测几乎不加时间。1B 每轮猜 4 个、每个猜测有七成机会被收下时，8B 走一步平均出 ${fd(b1.E,2)} 个 token；每轮还要加上 1B 猜 4 个的约 ${f(b1.td*1e3,0)} ms，所以速度是普通 decode 的 ${xT(b1.speed)}。</li><li><b>输出不变。</b>留下的每个 token 都是目标模型自己会写的；猜错只会多花点时间，不会让回答变差。</li><li><b>服务越忙，越不划算。</b>一步里请求很多时，draft 模型的几步变贵，要检查的 token 也会把算力占满。128 个请求时，3B draft 模型反而让 8B 变慢，只有普通 decode 的 ${xT(b128.speed)}。</li></ul>`+
     MORE('rm-long','展开：上下文很长时又不一样',`<p>本 Lab 假设每个请求的上下文是 1,024 个 token。上下文很长时情况不同：每一步都要把整段上下文的 KV cache 读一遍，它占了读取的大头。普通 decode 每出一个 token 读一遍；一次检查 ${M('k+1')} 个 token 也只读一遍，所以猜对的越多，读的次数越少。</p>`)+
     MORE('rm-prod','展开：生产系统里怎么用',`<p>vLLM 支持 draft 模型、n-gram 查找、EAGLE 和 MTP，文档说投机解码适合请求不多、受显存带宽限制的负载。EAGLE 不另跑一个完整的小模型，而是在目标模型上加一个小的 draft 头，用目标模型自己的特征（倒数第二层）来猜，所以猜得准、又便宜。SGLang 支持的方法包括 EAGLE、EAGLE3、单独的 draft 模型和 n-gram。DeepSeek‑V3 的权重里带了 1 个 MTP 模块，是专门训练来多预测一个 token 的额外层，可以当它自己的 draft 模型。真实系统里，浮点误差和 batch 大小的变化可能让输出有细微差别；算法本身不改变输出。</p><p>缩短受显存带宽限制的 decode 一步，还有别的办法，可以和投机解码一起用：权重变小，每一步要读的字节就少（${labRef('quantization','Quantization','量化')}）。</p>`)+
     MORE('rm-mis','展开：三个常见误解',`<p>✗ “猜错会让回答变差。”猜错的会被拒绝，换成目标模型自己的 token，只影响速度。</p><p>✗ “k 越大越快。”每多猜一个，每轮就多走一步 draft；可越往后的猜测越难留下，接受率 0.8 时，第 8 个猜测留下的机会只有约 ${pc(Math.pow(0.8,8))}。猜到一定数量以后，多花的时间比多得的 token 还多。</p><p>✗ “batch 多大都一样快。”batch 大了，draft 模型每一步要读的 KV cache 变多，检查也不再免费，加速会缩小，甚至变成减速。</p>`)+
     `${A.shell.toScenarios()}`);}}
];
const SIX=id=>STEPS.findIndex(s=>s.id===id);
const pct16=()=>{const r=compute({...DEFAULT}); return pc(r.td/r.tPlain);};
const d3big=()=>compute({...DEFAULT,B:128,dr:'d3',a:ALPHA.t8.d3});   // part 7: the 3B's 4 guesses at 128 requests (td) against a plain 8B step
const bCB=()=>{for(let B=1;B<=512;B++){const v=stepT(MD.t8,B,5,CTX,1); if(v.tc>v.tm) return B;} return 512;};   // from this batch on, the 8B's check of 5 tokens per request is compute-bound
const kLead=()=>tx('Pick how many tokens the draft model guesses per round and see how the average changes at α = 0.8. Each choice replays an example round; the draft model draws its words by probability, so a position guessed right in one round can be guessed wrong in another.','选每轮猜几个，看看 α = 0.8 时平均得到的 token 数怎么变。每种选择会重放一个示例轮次；draft 模型按概率抽词，同一个位置这一轮猜对，下一轮可能猜错。');
const costName=n=>{const [b,t]=COSTB[n]; return tx(`${b} request${b>1?'s':''} × ${t} token${t>1?'s':''}`,`${b} 个请求 × ${t} 个 token`);};
const costSub=n=>COSTB[n][1]===1?tx('plain decode','普通 decode'):tx(`${n} tokens in all`,`共 ${n} 个 token`);
function costLevers(){return `<p>${tx('Each option is how many requests one step holds, times how many tokens each. Pick one and see whether the step gets longer; the top of the chart draws which tokens each position sees.','每个选项是一步里放几个请求、每个请求几个 token。选一个，看这一步会不会变长；图上半部分画出每个位置看得到哪些 token。')}</p>`+
  A.ui.levers({aria:tx('Options: pick one','选项：选一个'),value:String(costN),options:COSTN.map(n=>({v:String(n),title:costName(n),
    desc:n===1?tx('Plain decode: one token per step','普通 decode：一步写 1 个'):n===5?tx('One user, 5 positions in one step','一个用户，一步算 5 个位置'):n===80?tx('80 tokens in one step','一步共 80 个'):tx('640 tokens in one step: a busy server','一步共 640 个，很忙的服务'),
    result:tried.cost&&tried.cost.has(n)?esc(msT(costRow(n).t)):''}))}).replace(/data-lever=/g,'data-lkey="cost" data-lever=');}
const costRow=n=>stepT(MD.t8,COSTB[n][0],COSTB[n][1],CTX,1);
/* levers of parts 5–7: each from the same starting point (R3.6) */
const LEV={
 expect:{opts:[2,4,8].map(k=>({v:k,t:{en:`Guess ${k} per round`,zh:`每轮猜 ${k} 个`},d:{en:`At most ${k+1} tokens per target step${k===4?', as in the example round':''}`,zh:`跑一次最多 ${k+1} 个 token${k===4?'，和示例那一轮一样':''}`}})),cur:()=>rpR.k,
   res:v=>tx(`${fd(expTok(0.8,v),2)} tokens per target step on average`,`跑一次平均 ${fd(expTok(0.8,v),2)} 个 token`)},
 same:{opts:[{v:'greedy',t:{en:'Greedy: keep only the top token',zh:'贪心：只收概率最高的'},d:{en:'Temperature 0',zh:'temperature 0'}},{v:'sample',t:{en:'Sampling: the acceptance rule',zh:'采样：按接受规则'},d:{en:'Each guess kept or replaced by the rule',zh:'按规则收下或换掉每个猜测'}},{v:'nocheck',t:{en:'Keep every guess, no check',zh:'不检查，猜什么收什么'},d:{en:'What the rule prevents',zh:'规则要防止的情况'}}],cur:()=>accMode,
   res:v=>{const fr=accFreq(v), R=accRule(), gap=Math.max(...fr.map((x,i)=>Math.abs(x-(v==='greedy'?(i?0:1):R.p[i]))));
     const pp=x=>fd(x*100,1);   // the sampled frequencies, as the chart and the observation show them (no check: 55.4% against 40%)
     return v==='nocheck'?tx(`over 10,000 draws, each word’s frequency is off the target’s probability by up to ${pp(gap)} percentage points`,`抽 10,000 次，每个词出现的频率和目标模型的概率最多差 ${pp(gap)} 个百分点`):gap<1e-9?tx('exactly the target’s output','和目标模型的输出完全一样'):tx(`over 10,000 draws, each word’s frequency is within ${pp(gap)} percentage points of the target’s probability`,`抽 10,000 次，每个词出现的频率和目标模型的概率最多差 ${pp(gap)} 个百分点`);}},
 batch:{opts:[1,32,128].map(b=>({v:b,t:{en:b===1?'1 request per step':`${b} requests per step`,zh:b===1?'一步 1 个请求':`一步 ${b} 个请求`},d:{en:b===1?'One user':b===32?'A moderately busy server':'A busy server',zh:b===1?'只有一个用户':b===32?'不算忙的服务':'很忙的服务'}})),cur:()=>S.B,
   res:v=>{const r=compute({...DEFAULT,B:v}); return tx(`1B draft: ${xT(r.speed)}`,`1B draft 模型：${xT(r.speed)}`);}}};
const LEAD={same:()=>tx('Pick a rule and compare, token by token, how often each comes out with the target model’s probability.','选一种规则，逐个 token 比一比它出现的频率和目标模型的概率。'),
  batch:()=>tx('Pick how many requests share each step and see how the speedup of each guesser changes, k = 4.','选一步里有几个请求，看看每轮猜 4 个时，各种猜法的加速倍数怎么变。'),expect:()=>''};
function levRes(key,v){if(!(tried[key]&&tried[key].has(v))) return ''; if(key==='expect'&&RP.on&&rpR.k===v) return esc(tx('playing…','播放中…')); return esc(LEV[key].res(v));}
function levers(key){const lv=LEV[key], cur=lv.cur();
  return (LEAD[key]()?`<p>${LEAD[key]()}</p>`:'')+A.ui.levers({aria:tx('Options: pick one','选项：选一个'),value:String(cur),
    options:lv.opts.map(o=>({v:String(o.v),title:L(o.t),desc:typeof o.d==='function'?o.d():L(o.d),result:levRes(key,o.v)}))}).replace(/data-lever=/g,`data-lkey="${key}" data-lever=`);}
const rkey=R=>R.k+'-'+R.acc;
const rpButtons=()=>`<div class="spd-tbtns"><button type="button" class="btn sm" data-act="rp-play">${rpBtnTxt()}</button></div>`;
const rpBtnTxt=()=>RP.on?tx('❚❚ Pause','❚❚ 暂停'):(watched.has(rkey(rpR))?tx('▶ Replay the round','▶ 重放这一轮'):(RP.f>0?tx('▶ Continue','▶ 继续'):tx('▶ Play the round','▶ 播放这一轮')));

/* observations */
function obsCost(){const n=costN, r=costRow(n), r1=costRow(1), nm=costName(n), B=COSTB[n][0];
  if(n===1) return tx(`<p><b>${nm}: ${msT(r.t)}.</b> Reading the weights and the KV cache takes ${msT(r.tm)}; the arithmetic, ${msT(r.tc)}.</p>`,`<p><b>${nm}：${msT(r.t)}。</b>读权重和 KV cache 要 ${msT(r.tm)}，计算只要 ${msT(r.tc)}。</p>`);
  if(n===5) return tx(`<p><b>${nm}: ${msT(r.t)}, the same as 1 token.</b> The weights are still read once; the extra arithmetic (${msT(r.tc)} in all) finishes while they are read. So one step computes 5 positions in the time of 1.</p>`,`<p><b>${nm}：${msT(r.t)}，和 1 个 token 一样。</b>权重还是只读一遍，多出来的计算（一共 ${msT(r.tc)}）在读权重的这段时间里就做完了。所以一步算 5 个位置，只花算 1 个的时间。</p>`);
  if(r.tc<=r.tm) return tx(`<p><b>${nm}: ${msT(r.t)}, still set by reading memory.</b> The extra time over ${msT(r1.t)} is reading ${B} requests’ KV cache; the arithmetic, ${msT(r.tc)}, has not caught up with the reading.</p>`,`<p><b>${nm}：${msT(r.t)}，还是由读显存决定。</b>比 ${msT(r1.t)} 多出来的，是读 ${B} 个请求的 KV cache；计算要 ${msT(r.tc)}，还没赶上读显存。</p>`);
  const rp=costRow(128);
  return tx(`<p><b>${nm}: ${msT(r.t)}, longer than plain decode of the same 128 requests (${msT(rp.t)}).</b> With this many requests the arithmetic (${msT(r.tc)}) takes longer than reading memory (about ${f(r.tm*1e3,0)} ms): the step is now compute-bound, and the 4 extra tokens per request are no longer free.</p>`,`<p><b>${nm}：${msT(r.t)}，比同样 128 个请求的普通 decode（${msT(rp.t)}）还长。</b>请求这么多时，计算（${msT(r.tc)}）比读显存（约 ${f(r.tm*1e3,0)} ms）还慢，这一步受算力限制（compute-bound），每个请求多算的 4 个 token 不再免费。</p>`);}
function obsRound(){const R=ROUNDS[4], c=roundCost(R);
  return tx(`<p><b>In this round the 8B ran one step and kept ${c.n} tokens, in ${msT(c.tRound)}; plain decode needs ${msT(c.plain)} for the same ${c.n} tokens.</b></p>`,
    `<p><b>这一轮 8B 只跑了一步，留下 ${c.n} 个 token，用了 ${msT(c.tRound)}；普通 decode 写这 ${c.n} 个 token 要 ${msT(c.plain)}。</b></p>`);}
function obsExpect(){const k=rpR.k, E=expTok(0.8,k), E4=expTok(0.8,4);
  let v='';
  if(pred){const right=pred==='a34'; v=right?tx(`<p class="okline"><b>Your prediction, “${predTxt(pred)}”, is right: ${fd(E4,2)} tokens per target step on average at k = 4.</b></p>`,`<p class="okline"><b>你的预测“${predTxt(pred)}”是对的：每轮猜 4 个时，跑一次平均得到 ${fd(E4,2)} 个 token。</b></p>`)
    :pred==='a42'?tx(`<p class="badline"><b>Your prediction was “${predTxt(pred)}”; the answer is ${fd(E4,2)}.</b> Counting each guess on its own gives ${M('0.8 \\times 4 + 1 = 4.2')}. But guess ${M('j')} survives only if it and all before it were accepted: chance ${M('0.8^j')}.</p>`,`<p class="badline"><b>你的预测是“${predTxt(pred)}”，答案是 ${fd(E4,2)}。</b>把每个猜测单独算，会得到 ${M('0.8 \\times 4 + 1 = 4.2')}。可猜测 ${M('j')} 要它自己和前面的都被收下才留得下，概率是 ${M('0.8^j')}。</p>`)
    :tx(`<p class="badline"><b>Your prediction was “${predTxt(pred)}”; the answer is ${fd(E4,2)}.</b> With α = 0.8 most rounds keep 3 or more tokens: even 4 accepted in a row has a chance of ${M('0.8^4')}, about 41%.</p>`,`<p class="badline"><b>你的预测是“${predTxt(pred)}”，答案是 ${fd(E4,2)}。</b>α = 0.8 时，大多数轮次能留下 3 个以上：连续 4 个都被收下的概率也有 ${M('0.8^4')}，约 41%。</p>`);}
  const tail=k===2?tx(`Few guesses cap the gain: the average, ${fd(E,2)}, can never pass 3.`,`猜得少，收益有上限：平均的 ${fd(E,2)} 个永远到不了 3 个。`):k===8?tx(`Twice the guesses of k = 4 add only ${fd(E-E4,2)} tokens per target step on average. The 4 extra guesses come last and survive only if every guess before them was accepted: guess 8 needs the first 7 and itself all kept, a chance of only about ${pc(Math.pow(0.8,8))}. Yet each extra guess costs the draft model one more step every round.`,`比每轮猜 4 个多猜一倍，跑一次平均却只多 ${fd(E-E4,2)} 个 token。多出来的 4 个猜测排在后面，前面的猜测都被收下，它们才留得下：猜测 8 要前 7 个和它自己都被收下，机会只有约 ${pc(Math.pow(0.8,8))}。可是每多猜一个，draft 模型每轮都要多走一步。`):tx('Each extra guess adds less than the one before.','每多猜一个，多得的都比上一个少。');
  return v+tx(`<p><b>k = ${k}: at most ${k+1} tokens per target step, ${fd(E,2)} on average.</b> ${tail}</p>`,`<p><b>每轮猜 ${k} 个时，目标模型跑一次最多 ${k+1} 个 token，平均 ${fd(E,2)} 个。</b>${tail}</p>`);}
function obsSame(){const R=accRule(), fr=accFreq(accMode);
  if(accMode==='greedy') return tx(`<p><b>Greedy: the output is always “${wd(0)}”, the 8B’s top token.</b> The draft’s top guess is also “${wd(0)}”, so it is accepted; had it been another token, the 8B would have put its own top token there instead.</p>`,`<p><b>贪心：输出总是“${wd(0)}”，就是 8B 概率最高的 token。</b>draft 模型概率最高的也是“${wd(0)}”，所以被收下；换成别的 token，8B 就会换上自己概率最高的那个。</p>`);
  if(accMode==='sample') return tx(`<p><b>The rule reproduces the 8B’s probabilities: in ${A.fmt.int(ACC_N)} draws “${wd(0)}” came out ${pc(fr[0],1)} of the time; the 8B gives it ${pc(R.p[0])}.</b> The draft overrates “${wd(0)}” (${pc(R.q[0])}), so it is sometimes rejected; the resampling hands that probability to the tokens the draft underrates. ${f(R.al*100,0)}% of the guesses are accepted here: this is the acceptance rate of this position.</p>`,`<p><b>按规则做，结果和 8B 的概率一样：抽 ${A.fmt.int(ACC_N)} 次，“${wd(0)}”出现了 ${pc(fr[0],1)}，8B 给它的概率是 ${pc(R.p[0])}。</b>draft 模型高估了“${wd(0)}”（${pc(R.q[0])}），所以它有时会被拒绝；重新抽样把这部分概率交给了 draft 模型低估的 token。这里 ${f(R.al*100,0)}% 的猜测被收下，这就是这个位置的接受率。</p>`);
  return tx(`<p class="badline"><b>Without the check the output follows the draft model: “${wd(0)}” comes out ${pc(fr[0],1)} of the time instead of the 8B’s ${pc(R.p[0])}.</b> The answer would be a different model’s answer: faster, but not the 8B’s (red in the chart).</p>`,`<p class="badline"><b>不检查，输出就跟着 draft 模型走：“${wd(0)}”出现了 ${pc(fr[0],1)}，而 8B 给它的概率是 ${pc(R.p[0])}。</b>回答变成了另一个模型的回答：快了，却不是 8B 的（图里红色）。</p>`);}
function obsBatch(){const B=S.B, rs=['ngram','d1','d3'].map(d=>compute({...DEFAULT,B,dr:d,a:ALPHA.t8[d]})), p=compute({...DEFAULT,B,dr:'none'}), [ng,d1,d3]=rs;
  if(B===1) return tx(`<p><b>One request per step: the 1B draft gives ${xT(d1.speed)} plain decode’s tokens per second, the 3B ${xT(d3.speed)}.</b> The 3B guesses better (8 guesses in 10 kept, against 7), but its 4 steps take ${msT(d3.td)}, longer than one 8B step (${msT(d3.tPlain)}); the 1B’s take ${msT(d1.td)}.</p>`,`<p><b>一步 1 个请求时，1B draft 模型让每秒的 token 数变成普通 decode 的 ${xT(d1.speed)}，3B 是 ${xT(d3.speed)}。</b>3B 猜得更准（10 个猜测收下 8 个，1B 是 7 个），可它走 4 步要 ${msT(d3.td)}，比 8B 走一步（${msT(d3.tPlain)}）还长；1B 走 4 步只要 ${msT(d1.td)}。</p>`);
  if(B===32) return tx(`<p><b>${B} requests: the 1B draft gives ${xT(d1.speed)}, the 3B draft ${xT(d3.speed)}.</b> The check step takes almost exactly as long as a plain decode step (about ${fd(p.tPlain*1e3,1)} ms). The speedup shrinks because every draft step now reads ${B} requests’ KV cache: the 3B’s 4 steps take ${msT(d3.td)}, ${fd(d3.td/p.tPlain,1)} times an 8B step.</p>`,`<p><b>${B} 个请求时，1B draft 模型是 ${xT(d1.speed)}，3B draft 模型是 ${xT(d3.speed)}。</b>检查那一步和普通 decode 一步几乎一样长（约 ${fd(p.tPlain*1e3,1)} ms）。加速变小，是因为 draft 模型每走一步都要读 ${B} 个请求的 KV cache：3B 走 4 步要 ${msT(d3.td)}，是 8B 一步的 ${fd(d3.td/p.tPlain,1)} 倍。</p>`);
  return tx(`<p class="badline"><b>${B} requests: the 3B draft makes the 8B slower than plain decode, ${xT(d3.speed)}</b> (red in the chart); the 1B draft still gives ${xT(d1.speed)}. Mostly the 3B draft itself is too dear: the 3B’s 4 steps take ${msT(d3.td)}, ${fd(d3.td/p.tPlain,1)} times a plain 8B step (${msT(p.tPlain)}). Checking ${B} × 5 tokens also stretches the 8B step to ${msT(d1.v.t)}.</p>`,`<p class="badline"><b>${B} 个请求时，3B draft 模型反而让 8B 比普通 decode 慢，只有 ${xT(d3.speed)}</b>（图里红色）；1B draft 模型还有 ${xT(d1.speed)}。主要是 3B draft 模型自己太贵：3B 走 4 步要 ${msT(d3.td)}，是 8B 普通 decode 一步（${msT(p.tPlain)}）的 ${fd(d3.td/p.tPlain,1)} 倍。8B 检查 ${B} × 5 个 token，这一步也变长到 ${msT(d1.v.t)}。</p>`);}
const WORDS={en:['sunny','cloudy','rainy','cold','warm'],zh:['晴','多云','下雨','冷','热']};
const wd=i=>L(WORDS)[i];

/* ---------- presets (Scenarios) ---------- */
const PRESETS=[
 {t:{en:'One user, 1B draft',zh:'一个用户，1B draft 模型'},d:{en:'8B, batch 1, α 0.7 (assumed)',zh:'8B，batch 1，α 0.7（假设）'},p:{tgt:'t8',dr:'d1',k:4,B:1,a:0.7}},
 {t:{en:'Busy server, 1B draft',zh:'很忙的服务，1B draft 模型'},d:{en:'8B, batch 128, α 0.7 (assumed)',zh:'8B，batch 128，α 0.7（假设）'},p:{tgt:'t8',dr:'d1',k:4,B:128,a:0.7}},
 {t:{en:'One user, 3B draft',zh:'一个用户，3B draft 模型'},d:{en:'8B, batch 1, α 0.8 (assumed)',zh:'8B，batch 1，α 0.8（假设）'},p:{tgt:'t8',dr:'d3',k:4,B:1,a:0.8}},
 {t:{en:'70B target, 3B draft',zh:'70B 目标模型，3B draft 模型'},d:{en:'4 H100s, batch 1, α 0.75 (assumed)',zh:'4 张 H100，batch 1，α 0.75（假设）'},p:{tgt:'t70',dr:'d3',k:4,B:1,a:0.75}},
 {t:{en:'Document Q&A, n-gram lookup',zh:'文档问答，n-gram 查找'},d:{en:'8B, batch 8, α 0.6 (assumed)',zh:'8B，batch 8，α 0.6（假设）'},p:{tgt:'t8',dr:'ngram',k:4,B:8,a:DOC_A}}];
// batch 8 is not one of the Settings batch options, so the n-gram scenario keeps its own value; Settings add it while it is set
const presetNow=()=>PRESETS.findIndex(p=>['tgt','dr','k','B'].every(k=>p.p[k]===S[k])&&Math.abs(p.p.a-S.a)<1e-9);
function insightHTML(r){const i=presetNow(), s=r.s;
  if(s.dr==='none') return tx(`<p><b>No guessing: ${fd(r.tpsPlain,1)} tokens per second for each request.</b> Pick a guesser in Settings to compare.</p>`,`<p><b>不猜：每个请求每秒 ${fd(r.tpsPlain,1)} 个 token。</b>在“自由调参”里选一种猜法来比较。</p>`);
  const head=tx(`<p><b>${fd(r.E,2)} tokens per target step on average; each request gets ${fd(r.tps,1)} tokens per second instead of ${fd(r.tpsPlain,1)}: ${xT(r.speed)} plain decode.</b></p>`,`<p><b>目标模型跑一次平均得到 ${fd(r.E,2)} 个 token；每个请求每秒 ${fd(r.tps,1)} 个 token，普通 decode 是 ${fd(r.tpsPlain,1)} 个，是它的 ${xT(r.speed)}。</b></p>`);
  const why=[
   tx('<p>One user leaves the 8B step memory-bound, so the check is nearly free; the 1B’s 4 steps are the main extra cost.</p>','<p>只有一个用户时，8B 那一步受显存带宽限制，检查几乎免费；主要的额外开销是 1B 走的那 4 步。</p>'),
   tx(`<p>With 128 requests every 1B step also reads 128 requests’ KV cache: its 4 steps take ${msT(r.td)}. Each check carries ${128*5} tokens, so the 8B step is compute-bound and longer: ${msT(r.v.t)} instead of ${msT(r.tPlain)}. The gain shrinks; guessing fewer tokens per round wins back part of the speedup.</p>`,`<p>128 个请求时，1B 每走一步也要读 128 个请求的 KV cache，4 步要 ${msT(r.td)}；每次检查要算 ${128*5} 个 token，8B 那一步受算力限制，变长到 ${msT(r.v.t)}，普通 decode 是 ${msT(r.tPlain)}。所以加速变小了。每轮少猜几个，能把加速找回来一些。</p>`),
   tx(`<p>The 3B guesses better, but its 4 steps take ${msT(r.td)}, longer than one 8B step (${msT(r.tPlain)}), so it gains less than the 1B.</p>`,`<p>3B 猜得更准，可它走 4 步要 ${msT(r.td)}，比 8B 走一步（${msT(r.tPlain)}）还长，所以收益比 1B 小。</p>`),
   tx(`<p>With the α this lab assumes, a larger target makes a bigger draft worth it: the 3B’s 4 steps cost only ${pc(r.td/r.tPlain)} of a 70B step, and each extra token it gets right is worth a whole 70B step.</p>`,`<p>在本 Lab 假设的 α 下，目标模型越大，大一些的 draft 模型越划算：3B 走 4 步只占 70B 一步的 ${pc(r.td/r.tPlain)}，3B 每多猜对一个 token，就省下 70B 的一整步。</p>`),
   tx('<p>The answer copies whole phrases of the document, so looking them up in the prompt guesses well, and the lookup costs no GPU time.</p>','<p>回答会整句照抄文档，在 prompt 里查找就能猜得不错，而且查找不占 GPU 时间。</p>')];
  return head+(i>=0?why[i]:tx('<p class="small">The current settings match none of these scenarios.</p>','<p class="small">现在的设置和这几个场景都不一样。</p>'));}

/* ---------- challenge ---------- */
const chPoint=(dr,k)=>{const a=dr==='none'?0:ALPHA[CH.tgt][dr], r=compute({tgt:CH.tgt,B:CH.B,dr,k:dr==='none'?1:k,a}); return {dr,k:dr==='none'?0:k,a,r,speed:r.speed,ok:r.speed>=CH_GOAL};};
const chAll=()=>{const o=[chPoint('none',1)]; ['ngram','d1','d3'].forEach(d=>KS.forEach(k=>o.push(chPoint(d,k)))); return o;};
const chBest=()=>chAll().reduce((b,p)=>p.speed>b.speed?p:b);
const chName=p=>p.dr==='none'?tx('No draft','不猜'):tx(`${dName(p.dr)}, k = ${p.k}`,`${dName(p.dr)}，每轮猜 ${p.k} 个`);
const chMe=()=>chPoint(S.dr,S.k);

/* ---------- view ---------- */
function headerHTML(){const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'inference',outcome:esc(OUTCOME()),short:tx('Speculative decoding','投机解码'),title:l?L(l.q):tx('How can guessing speed up generation without changing it?','先猜再检查，怎么能更快，而且不改变输出？'),
    minutes:12,fid:'Simulated',fidPop:FIDPOP(),primer:tx('Primer: decode','前置知识：decode'),primerPop:PRIPOP(),descs:DESCS(),mode});}
const spdCtl=()=>`<div class="spd-speed"><span class="spd-speed-l">${tx('Speed','速度')}</span><div class="seg-ctl" role="group" aria-label="${esc(tx('Playback speed','播放速度'))}" data-key="rpSpd">${[['slow',tx('Slow','慢')],['med',tx('Medium','中')],['fast',tx('Fast','快')]].map(([v,x])=>`<button type="button" data-v="${v}" aria-pressed="${rpSpd===v}">${x}</button>`).join('')}</div></div>`;
function stageHTML(){
  return `<div class="panel stage spd-stage" id="spd-stage">`+
   `<div class="stage-head"><div class="spd-st-title" id="spd-st-title"></div><div class="stage-actions"><span class="status" id="spd-status" tabindex="0"></span></div></div>`+
   `<div id="spd-main" class="spd-main">`+
    `<section class="spd-blk" id="spd-cost" hidden></section>`+
    `<section class="spd-blk" id="spd-round" hidden><div class="spd-rows" id="spd-rows"></div><div class="spd-rcost" id="spd-rcost" hidden></div><div class="spd-player" id="spd-player">`+
      `<button type="button" class="btn sm" data-act="rp-play" id="spd-rp-play"></button><button type="button" class="btn ghost sm spd-ib" data-act="rp-back" aria-label="${esc(tx('One frame back','后退一格'))}">◂</button><button type="button" class="btn ghost sm spd-ib" data-act="rp-fwd" aria-label="${esc(tx('One frame forward','前进一格'))}">▸</button><button type="button" class="btn ghost sm" data-act="rp-replay">${tx('Replay','重放')}</button>${spdCtl()}`+
      `<label class="spd-scrub"><span class="spd-sr">${tx('Progress','进度')}</span><input type="range" id="spd-rp-pos" min="0" max="9" step="1" value="0" aria-label="${esc(tx('Progress of the round','这一轮的进度'))}"></label></div></section>`+
    `<section class="spd-blk" id="spd-acc" hidden></section>`+
    `<section class="spd-blk" id="spd-kcurve" hidden></section>`+
    `<section class="spd-blk" id="spd-bcurve" hidden></section>`+
    `<section class="spd-blk" id="spd-time" hidden></section>`+
   `</div>`+
   `<div class="metrics spd-metrics" id="spd-metrics"></div>`+
   `<details class="math" id="spd-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="spd-mathbody"></div></details></div>`;}
const segOpts=(arr,fn)=>arr.map(v=>[String(v),fn(v)]);
function controlsHTML(){
  const bOpts=BATCHES.includes(S.B)?BATCHES:BATCHES.concat([S.B]).sort((a,b)=>a-b);
  return `<aside class="panel controls" id="spd-controls" aria-label="${esc(A.t('ctrlAria')||'Settings')}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Watch the speedup chart and the timeline.','看加速倍数的图和时间线。')}</p><p class="ctl-lock" id="spd-lock" hidden></p><div id="spd-ctls">`+
   A.ui.seg({id:'spd-ctl-tgt',key:'tgt',label:tx('Target model','目标模型'),options:[['t8',MD.t8.name],['t70',MD.t70.name]],value:S.tgt,note:tx('The 70B runs on 4 H100s with tensor parallelism; the 8B on one.','70B 用 4 张 H100 做张量并行，8B 用一张。')})+
   A.ui.seg({id:'spd-ctl-dr',key:'dr',label:tx('Who guesses','谁来猜'),options:DRAFTS.map(d=>[d,dName(d)]),value:S.dr,note:tx(`n-gram lookup looks the guess up in the prompt; the 1B and 3B are ${MD.d1.name} and ${MD.d3.name}. Picking one sets α to this lab’s assumption for the pair.`,`n-gram 查找在 prompt 里找猜测；1B 和 3B 是 ${MD.d1.name} 和 ${MD.d3.name}。选一个，α 就设成这个 Lab 给这一对模型假设的值。`)})+
   A.ui.seg({id:'spd-ctl-k',key:'k',label:tx('Guesses per round k','每轮猜的 token 数 k'),options:segOpts(KS,String),value:String(S.k)})+
   A.ui.seg({id:'spd-ctl-B',key:'B',label:tx('Requests per step (batch)','一步里的请求数（batch）'),options:segOpts(bOpts,String),value:String(S.B),note:tx('This lab assumes 1,024 tokens of context per request; every setting fits in GPU memory.','本 Lab 假设每个请求有 1,024 个 token 的上下文；每种设置都放得进显存。')})+
   A.ui.range({id:'spd-ctl-a',key:'a',label:tx('Acceptance rate α','接受率 α'),min:0.3,max:0.95,step:0.05,value:S.a,out:aT(S.a),note:tx(`This lab assumes these α: 8B with n-gram 0.3 (open chat), 1B 0.7, 3B 0.8; 70B with 1B 0.65, 3B 0.75. α depends on how alike the two models are and on the text: higher for predictable text such as code (scaling book) and for answers that copy the prompt.`,`本 Lab 假设的 α：8B 配 n-gram 0.3（开放聊天）、1B 0.7、3B 0.8；70B 配 1B 0.65、3B 0.75。α 取决于两个模型有多像，也取决于文本：代码这类好预测的文本（scaling book）和照抄 prompt 的回答更高。`)})+
   `</div><div id="spd-reset-wrap"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;}
const CODE=`import numpy as np

def verify(guesses, q, p, rng):
    """One round of speculative sampling.
    guesses: k tokens from the draft model
    q[i]: draft probabilities at guess i (vocab,)
    p[i]: target probabilities at position i,
          from ONE target pass (k + 1 rows)"""
    out = []
    for i, x in enumerate(guesses):
        if rng.random() < min(1.0, p[i][x] / q[i][x]):
            out.append(x)             # accepted
            continue
        rest = np.maximum(p[i] - q[i], 0.0)
        out.append(rng.choice(len(rest), p=rest / rest.sum()))
        return out                    # first rejection ends it
    k = len(guesses)                  # all accepted:
    out.append(rng.choice(len(p[k]), p=p[k]))   # bonus
    return out`;
function connectHTML(){
  const fl=(url,label,x)=>`<a href="${url}" target="_blank" rel="noopener"><code>${label}</code></a>${x}`;
  return `<section class="wrap connect" aria-labelledby="spd-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="spd-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx(`The check of one round: accept a guess with probability ${M('\\min(1, p/q)')}, otherwise draw from ${M('\\max(0, p - q)')} and stop; if all are accepted, draw one more token from the target. Greedy decoding is the case where p and q put all their probability on one token.`,`一轮的检查：以 ${M('\\min(1, p/q)')} 的概率收下猜测，否则从 ${M('\\max(0, p - q)')} 里抽一个并结束；全部收下时，再从目标模型抽一个 token。贪心解码就是 p 和 q 把概率全放在一个 token 上的情况。`)}</p><pre class="code" id="spd-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="spd-prod">`+
   tx(`<li>${fl(VLLM+'vllm/v1/sample/rejection_sampler.py','vllm/v1/sample/rejection_sampler.py',`: vLLM’s RejectionSampler “strictly follows the algorithm described in” arXiv 2211.17192; it names accepted, recovered (drawn from the adjusted distribution) and ${DEF('bonus tokens',D_BONUS())}.`)}</li>`+
      `<li>${fl(VLLM+'docs/features/speculative_decoding/README.md','docs/features/speculative_decoding',': methods EAGLE, MTP, draft model, n-gram and suffix decoding; <code>num_speculative_tokens</code> is k. The docs aim it at “medium-to-low QPS (queries per second), memory-bound workloads”.')}</li>`+
      `<li>${fl(SGL,'sglang/srt/speculative/spec_info.py',': SGLang’s methods include EAGLE, EAGLE3, STANDALONE (a separate draft model) and NGRAM.')}</li>`+
      `<li>${fl(HFT,'transformers generation/utils.py',`: <code>_speculative_sampling</code> applies “algorithm 1” of the paper: keep while a random number is at most ${M('p/q')}, then sample from ${M('\\max(0, p - q)')}.`)}</li>`+
      `<li>${fl(DSV3,'DeepSeek-V3 README_WEIGHTS.md',': the open weights include 1 MTP module; the README says MTP “can also be used for speculative decoding”.')}</li>`+
      `<li>${fl(EAGLE,'EAGLE README',': EAGLE drafts from the target model’s own features; on a 13B model it reports EAGLE-1 at 3 times vanilla decoding’s speed and EAGLE-3 at 5.6 times.')}</li>`,
      `<li>${fl(VLLM+'vllm/v1/sample/rejection_sampler.py','vllm/v1/sample/rejection_sampler.py',`：vLLM 的 RejectionSampler“严格按照” arXiv 2211.17192 的算法实现，代码里分成被收下的 token、从调整后的分布里重新抽的 token（recovered）和 ${DEF('bonus token',D_BONUS())}。`)}</li>`+
      `<li>${fl(VLLM+'docs/features/speculative_decoding/README.md','docs/features/speculative_decoding','：方法有 EAGLE、MTP、draft 模型、n-gram 和 suffix decoding；<code>num_speculative_tokens</code> 就是 k。文档说它适合请求不多、受显存带宽限制的负载（medium-to-low QPS, memory-bound workloads）。')}</li>`+
      `<li>${fl(SGL,'sglang/srt/speculative/spec_info.py','：SGLang 支持的方法包括 EAGLE、EAGLE3、STANDALONE（单独的 draft 模型）和 NGRAM。')}</li>`+
      `<li>${fl(HFT,'transformers generation/utils.py',`：<code>_speculative_sampling</code> 按论文的“algorithm 1”实现，随机数不超过 ${M('p/q')} 就收下，否则从 ${M('\\max(0, p - q)')} 里抽。`)}</li>`+
      `<li>${fl(DSV3,'DeepSeek-V3 README_WEIGHTS.md','：开源权重里带了 1 个 MTP 模块；README 说 MTP“也可以用于投机解码”。')}</li>`+
      `<li>${fl(EAGLE,'EAGLE README','：EAGLE 用目标模型自己的特征来猜；它报告在 13B 模型上，EAGLE-1 的速度是普通 decode 的 3 倍，EAGLE-3 是 5.6 倍。')}</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li><a href="https://arxiv.org/abs/2211.17192" target="_blank" rel="noopener">Fast Inference from Transformers via Speculative Decoding</a><span>${tx('Leviathan, Kalman and Matias, ICML 2023 (arXiv 2211.17192)','Leviathan、Kalman、Matias，ICML 2023（arXiv 2211.17192）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2302.01318" target="_blank" rel="noopener">Accelerating Large Language Model Decoding with Speculative Sampling</a><span>${tx('Chen et al., 2023 (arXiv 2302.01318)','Chen 等，2023（arXiv 2302.01318）')}</span></li>`+
   `<li><a href="https://arxiv.org/abs/2401.15077" target="_blank" rel="noopener">EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty</a><span>${tx('Li et al., 2024 (arXiv 2401.15077)','Li 等，2024（arXiv 2401.15077）')}</span></li>`+
   `<li><a href="${SBOOK}" target="_blank" rel="noopener">How to Scale Your Model: Speculative Sampling</a><span>${tx('JAX scaling book, appendix D: why it is a latency win while decode is not compute-bound','JAX scaling book 附录 D：为什么在 decode 不受算力限制时能降低延迟')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="spd-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="spd-drill-a" hidden></div></article>`+
   `</div></section>`;}
function viewHTML(){return `<div class="spd-app m2" id="spd-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="spd-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- rail ---------- */
function sandboxHTML(){const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3><p>${tx('Pick a scenario and see how many tokens per second each request gets with guessing, compared with plain decode.','选一个场景，看看用了投机解码以后，每个请求每秒的 token 数是普通 decode 的几倍。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios: pick one','应用场景：选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="spd-insight" aria-live="polite"></div><p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;}
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted spd-sub">${tx('After you submit, you see whether you reached 1.5 times and whether a better choice exists.','提交后告诉你是否达到 1.5 倍，以及有没有更好的选择。')}</p>`;
  const r0=chPoint(CH0.dr,CH0.k);
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('Llama‑3.1‑70B on 4 H100s serves 128 requests per step: give each request at least 1.5 times plain decode’s tokens per second','4 张 H100 上的 Llama‑3.1‑70B 同时服务 128 个请求：让每个请求每秒的 token 数至少是普通 decode 的 1.5 倍')}</h3>`+
   `<p>${tx(`This lab assumes 1,024 tokens of context per request. Plain decode gives each request ${fd(r0.r.tpsPlain,1)} tokens per second. The initial setup, the 3B draft guessing 4 tokens per round, gives ${xT(r0.speed)} that. This lab assumes α = 0.3 for n-gram lookup, 0.65 for the 1B and 0.75 for the 3B with the 70B. Pick the guesser and k that give the most tokens per second.`,`本 Lab 假设每个请求有 1,024 个 token 的上下文。普通 decode 时每个请求每秒 ${fd(r0.r.tpsPlain,1)} 个 token。初始设置是 3B draft 模型每轮猜 4 个，速度是普通 decode 的 ${xT(r0.speed)}。本 Lab 假设配 70B 时，n-gram 查找的 α 是 0.3，1B 是 0.65，3B 是 0.75。选出让每秒 token 数最多的猜法和 k。`)}</p>`+
   A.ui.seg({id:'spd-ic-dr',key:'dr',label:tx('Who guesses','谁来猜'),options:DRAFTS.map(d=>[d,dName(d)]),value:S.dr,inline:true})+
   A.ui.seg({id:'spd-ic-k',key:'k',label:tx('Guesses per round k','每轮猜的 token 数 k'),options:segOpts(KS,String),value:String(S.k),inline:true})+
   `<div class="goal mono" id="spd-ch-g"><span>${tx('Speed vs plain decode','速度是普通 decode 的')}</span><span><b id="spd-ch-v"></b>${tx(' (goal: at least 1.5 times)','（目标：至少 1.5 倍）')}</span></div><div class="progress" id="spd-ch-p" aria-hidden="true"><i></i></div>`+
   `<p class="small muted">${tx('The target, the batch, the context and the α of each guesser are fixed in this challenge.','这道题里，目标模型、batch、上下文和每种猜法的 α 都固定不变。')}</p>`+submit+`<div id="spd-ch-res" aria-live="polite"></div></div>`;}
function chResultHTML(){
  if(!chSubmitted) return '';
  const all=chAll(), me=all.find(p=>p.dr===S.dr&&(p.dr==='none'||p.k===S.k)), best=all.reduce((b,p)=>p.speed>b.speed?p:b), sp=p=>fd(p.speed,2);
  let h;
  if(me.ok&&me===best) h=tx(`<p class="okline"><b>Goal met: ${xT(me.speed)} plain decode, and no other choice does better.</b></p>`,`<p class="okline"><b>达成目标：速度是普通 decode 的 ${xT(me.speed)}，没有别的选择更快。</b></p>`);
  else if(me.ok) h=tx(`<p class="okline"><b>Goal met: ${xT(me.speed)} plain decode.</b> ${chName(best)} does a little better: ${xT(best.speed)}.</p>`,`<p class="okline"><b>达成目标：速度是普通 decode 的 ${xT(me.speed)}。</b>${chName(best)}还能再快一点：${xT(best.speed)}。</p>`);
  else {h=tx(`<p class="badline"><b>Not yet: ${xT(me.speed)} plain decode, short of 1.5.</b></p>`,`<p class="badline"><b>还没达到：速度是普通 decode 的 ${xT(me.speed)}，不到 1.5 倍。</b></p>`)+
     (me.dr==='ngram'?tx(`<p class="small">Hint: with α = 0.3 a target step yields few tokens however many you guess, and every extra guess still makes the 70B’s check longer. A guesser that agrees more often is needed.</p>`,`<p class="small">提示：α = 0.3 时，猜多少个，目标模型跑一次得到的 token 都不多，多猜的每一个还让 70B 的检查变长。需要一个更常猜对的猜法。</p>`)
     :me.dr==='none'?tx('<p class="small">Hint: without guessing every request gets one token per 70B step. Pick a guesser.</p>','<p class="small">提示：不猜的话，70B 每走一步，每个请求只得到一个 token。选一种猜法。</p>')
     :tx(`<p class="small">Hint: at 128 requests per step, the 70B’s check of ${M('k+1')} tokens per request takes longer with every extra guess, for all 128 requests, while a late guess is rarely kept. Try fewer guesses per round.</p>`,`<p class="small">提示：一步 128 个请求时，70B 要给每个请求检查 ${M('k+1')} 个 token，每多猜一个，128 个请求的检查都变长，可排在后面的猜测很少留得下。试试每轮少猜几个。</p>`));
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal&&!(me.ok&&me===best)) return h;   // the best pick shows the chart at once
  const init=all.find(p=>p.dr===CH0.dr&&p.k===CH0.k);
  const rows=all.map(p=>({label:chName(p),value:p.speed,text:xT(p.speed),ok:p.ok,me:p===me,best:p===best,
    tags:(p===init?[{text:tx('initial setup','初始设置')}]:[]).concat(p===me?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  h+=A.ui.cmpChart({brief:true,title:tx('70B, 128 requests per step: tokens per second compared with plain decode (goal: at least 1.5 times)','70B 一步 128 个请求时，每秒 token 数是普通 decode 的几倍（目标：至少 1.5 倍）'),
    aria:tx('Speed vs plain decode: ','速度是普通 decode 的倍数：')+all.filter(p=>p.ok||p===me||p===init).map(p=>`${chName(p)}${tx(': ','：')}${sp(p)}`).join(tx('; ','；')),rows,
    hiddenText:n=>tx(`The other ${n} choices stay under 1.5 times and are not shown.`,`其余 ${n} 种不到 1.5 倍，图里没有列出。`),goal:CH_GOAL,goalLabel:tx('1.5 times','1.5 倍'),max:2,okText:tx('at least 1.5 times','至少 1.5 倍'),noText:tx('under 1.5 times','不到 1.5 倍'),goalText:tx('the goal','目标')});
  const c2=chPoint('d1',2).r, c4=chPoint('d1',4).r, d1s=chPoint('d1',1).r.td1, d3s=chPoint('d3',1).r.td1, ngb=all.filter(p=>p.dr==='ngram').reduce((u,p)=>p.speed>u.speed?p:u), b=best, o=all.filter(p=>p.ok&&p!==b&&p.k===b.k)[0];
  h+=tx(`<p class="small">Why: at 128 requests the 70B’s plain decode step, ${msT(c2.tPlain)}, is still memory-bound, but a check of 3 tokens per request is already compute-bound and takes ${msT(c2.v.t)}; with 4 guesses it takes ${msT(c4.v.t)}. So each extra guess costs a lot of check time, and 1 or 2 guesses per round are best. The drafts are cheap next to a 70B step: one 1B step takes ${msT(d1s)} and one 3B step ${msT(d3s)}, mostly reading their own KV cache for 128 requests. With ${b.k} guesses the 1B is a little ahead${o?` of the 3B (${xT(o.speed)})`:''}: the 3B’s higher α does not quite pay for its steps, which cost about three times as much. n-gram lookup costs nothing, but with α = 0.3 it reaches at most ${xT(ngb.speed)}.</p>`,
    `<p class="small">原因：一步 128 个请求时，70B 普通 decode 的一步（${msT(c2.tPlain)}）还受显存带宽限制，可每个请求检查 3 个 token，这一步就受算力限制了，要 ${msT(c2.v.t)}；每轮猜 4 个时要 ${msT(c4.v.t)}。所以每多猜一个，检查就多花不少时间，每轮猜 1 到 2 个最好。和 70B 的一步比，draft 模型很便宜：1B 走一步 ${msT(d1s)}，3B 走一步 ${msT(d3s)}，大部分时间花在读它自己那份 128 个请求的 KV cache 上。每轮猜 ${b.k} 个时，1B 略好${o?`（3B 是 ${xT(o.speed)}）`:''}：3B 的 α 高一些，可它每一步的开销约是 1B 的三倍，补不回来。n-gram 查找不占 GPU 时间，可 α = 0.3 时最多只有 ${xT(ngb.speed)}。</p>`);
  h+=tx('<p class="small muted">This lab assumes the α here. In a real deployment, measure the acceptance on your own traffic (vLLM reports it per request) before choosing the draft and k.</p>','<p class="small muted">这里的 α 是本 Lab 的假设值。真实部署时，先在自己的流量上测接受率（vLLM 可以按请求报告），再选 draft 模型和 k。</p>');
  return h;}

/* ---------- stage drawing ---------- */
let W=640;   // the drawings use the panel's own width as their viewBox, so text keeps its size on a phone
function svgWrap(h,body,aria,w){return `<svg class="spd-svg" viewBox="0 0 ${w||W} ${h}" role="img" aria-label="${esc(aria)}" preserveAspectRatio="xMinYMin meet">${body}</svg>`;}
/* polish round 3 (2026-10-16, R7.12): on a wide stage the k curve, the batch curve and the timeline have their legend in a column on their right
   (--spd-legw, the column's width, set in spec-decoding.css) and are drawn shorter (--spd-ch, the height in viewBox units). The plot
   keeps the scale it has across the whole panel, so its text is the same size on screen: its viewBox is narrower by as much as its
   box. Without the variables (phones, narrower stages) it is W wide and h high, as before. */
function plotBox(el,h){const cs=getComputedStyle(el), lw=parseFloat(cs.getPropertyValue('--spd-legw')), ch=parseFloat(cs.getPropertyValue('--spd-ch'));
  const cw=el.clientWidth-parseFloat(cs.paddingLeft)-parseFloat(cs.paddingRight), gap=parseFloat(cs.columnGap)||0;
  return {w:lw>0&&cw>lw+gap?Math.round((cw-lw-gap)*W/cw):W, h:ch>0?ch:h};}
/* the drawings take their width from the stage (W) and, on a wide stage, the sizes above: the stage widens without a window resize when
   the page puts the step list beside it or leaves the primer, so the stage is drawn again when its width changes */
let stRO=null, stW=0;
function watchStage(){if(stRO){stRO.disconnect(); stRO=null;} const st=app&&$('#spd-stage',app); if(!st||typeof ResizeObserver!=='function') return; stW=st.offsetWidth;
  stRO=new ResizeObserver(()=>requestAnimationFrame(()=>{if(!app||!app.contains(st)) return; const w=st.offsetWidth; if(!w||Math.abs(w-stW)<=2) return; stW=w; render();}));
  stRO.observe(st,{box:'border-box'});}
/* part 2's top diagram: what one step holds (requests × tokens each) and what each position sees. A row is a position: it sees the
   context (read from the KV cache) and the tokens before it, never the ones after (the causal mask), so all rows come out of one step */
const CAUSAL={en:{ctx:'context',in:['there','was','a','black','dog'],out:['was','a','black','dog','who']},
  zh:{ctx:'上文',in:['，','有','一只','黑','狗'],out:['有','一只','黑','狗','住']}};
function stepInHTML(){const [B,n]=COSTB[costN], w=CAUSAL[A.lang()==='zh'?'zh':'en'], nar=W<480;
  const p=6, labW=nar?0:54, ctxW=nar?48:52, cw=nar?32:46, gx=nar?3:4, arW=nar?15:22, outW=nar?38:50, ch=15, rowH=20, top=24, ct=nar?' spd-ctn':'';
  const xC=p+labW, xT=j=>xC+ctxW+gx+j*(cw+gx), xA=xT(n-1)+cw+gx, xO=xA+arW, cx1=xO+outW+p, cardH=p*2+n*rowH-5, cardMax=p*2+5*rowH-5;
  const H=top+cardMax+(nar?22:4);
  let b=(n>1?`<text class="spd-tk" x="${xT(0)}" y="11">${esc(tx('tokens in this step','这一步送进去的 token'))}</text>`:'')+   // one token: the header would run into “next token”; the legend names it
    `<text class="spd-tk" x="${xO}" y="11">${esc(nar?tx('next','下一个'):tx('next token','算出的下一个'))}</text>`;
  if(B>1){[10,5].forEach(o=>{b+=`<rect class="spd-card spd-shadow" x="${o}" y="${top-o}" width="${cx1}" height="${cardH}" rx="5"/>`;});
    const t=tx(`× ${B} requests, one card each`,`× ${B} 个请求，每个一张这样的卡`);
    b+=nar?`<text class="spd-tl" x="0" y="${top+cardH+16}">${esc(t)}</text>`:`<text class="spd-tl" x="${cx1+20}" y="${top+cardH/2+4}">${esc(t)}</text>`;}
  b+=`<rect class="spd-card" x="0" y="${top}" width="${cx1}" height="${cardH}" rx="5"/>`;
  for(let i=0;i<n;i++){const y=top+p+i*rowH;
    if(labW) b+=`<text class="spd-tk" x="${p}" y="${y+11}">${esc(tx(`pos. ${i+1}`,`位置 ${i+1}`))}</text>`;
    b+=`<rect class="spd-cxc" x="${xC}" y="${y}" width="${ctxW}" height="${ch}" rx="3"/><text class="spd-ct${ct}" x="${xC+ctxW/2}" y="${y+11}" text-anchor="middle">${esc(w.ctx)}</text>`;
    for(let j=0;j<n;j++){const x=xT(j);
      b+=j<=i?`<rect class="spd-tin" x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3"/><text class="spd-ct${ct}" x="${x+cw/2}" y="${y+11}" text-anchor="middle">${esc(w.in[j])}</text>`
        :`<rect class="spd-hid" x="${x}" y="${y}" width="${cw}" height="${ch}" rx="3"/>`;}
    b+=`<text class="spd-tk" x="${xA+arW/2}" y="${y+11}" text-anchor="middle">→</text><rect class="spd-tout" x="${xO}" y="${y}" width="${outW}" height="${ch}" rx="3"/>`+
      `<text class="spd-ct spd-cto${ct}" x="${xO+outW/2}" y="${y+11}" text-anchor="middle">${esc(w.out[i])}</text>`;}
  const cap=n===1?tx('Plain decode: the step holds only the newest token, so it computes one position.','普通 decode：这一步只放最新的 1 个 token，只算 1 个位置。')
    :tx(`Each row is one position: it sees only the context and the tokens before it, so all ${n} positions come out of the same step.`,`每一行是一个位置：它只看得到上文和它前面的 token，所以 ${n} 个位置能在同一步里一起算。`);
  const shared=B>1?tx(` The ${B} requests share one read of the weights.`,`${B} 个请求共用一次权重读取。`):'';
  const fig=svgWrap(H,b,tx('The positions one step computes and the tokens each one sees','一步里算的各个位置，和每个位置看得到的 token'))+
    `<div class="legend-row spd-leg">${lg('spd-sw-cxc',tx('context, read from the KV cache','上文，从 KV cache 读'),tx('The tokens before this step. Their keys and values are already in the KV cache, so each position reads them from memory instead of computing them again.','这一步之前的 token。它们的 k、v 已经存在 KV cache 里，每个位置从显存读出来用，不用重算。'))}`+
    `${lg('spd-sw-tin',tx('a token in this step','这一步送进去的 token'),tx('A token already known when the step starts: in plain decode the newest token; when checking guesses, each guess.','这一步开始时已经知道的 token：普通 decode 时是最新的那个 token，检查猜测时是每个猜测。'))}`+
    `${n>1?lg('spd-sw-hid',tx('dashed: hidden from this position','虚线格：这个位置看不到'),D_MASK()):''}`+
    `${lg('spd-sw-tout',tx('the next token it computes','算出的下一个 token'),tx('Each position’s prediction of the token after it. All of them come out of the same step.','每个位置对它后面那个 token 的预测。它们都在同一步里算出来。'))}</div>`+
    '';
  return `<div class="spd-ch-t">${tx(`What one step computes: ${costName(costN)}`,`这一步里算什么：${costName(costN)}`)}</div>`+fig+`<p class="small muted spd-cap">${cap}${shared}</p>`;}
function drawCost(){
  const el=$('#spd-cost',app), g=sid(), rows=g==='mission'?[1]:COSTROWS, mx=Math.max(...COSTROWS.map(n=>costRow(n).t))*1.05, nar=W<480;
  const X0=nar?0:184, X1=W-(nar?58:70), sx=t=>X0+(X1-X0)*t/mx, rowH=nar?60:46, off=nar?18:0;
  let b='';
  rows.forEach((n,i)=>{const r=costRow(n), y0=14+i*rowH, y=y0+off, on=n===costN&&g==='cost', hide=g==='cost'&&!(tried.cost&&tried.cost.has(n===128?640:n));   // part 2: a row shows its step once its lever is tried
    const lab=esc(costName(n)), sub=esc(costSub(n));
    b+=`<g class="${on?'spd-on':''}">`+(on?`<rect class="spd-rowbg" x="-6" y="${nar?y0-6:y-6}" width="${W+12}" height="${nar?rowH:44}" rx="4"/>`:'')+
      (nar?`<text class="spd-tl" x="0" y="${y0+9}">${lab}<tspan class="spd-tsub" dx="8">${sub}</tspan></text>`:`<text class="spd-tl" x="0" y="${y+12}">${lab}</text><text class="spd-tk" x="0" y="${y+27}">${sub}</text>`)+
      (hide?`<rect class="spd-q" x="${X0}" y="${y}" width="${Math.min(40,X1-X0)}" height="32" fill-opacity=".5"/><text class="spd-tv" x="${X0+48}" y="${y+21}">?</text></g>`:`<rect class="spd-w" x="${X0}" y="${y}" width="${(sx(r.tw)-X0).toFixed(1)}" height="14"/><rect class="spd-kv" x="${sx(r.tw).toFixed(1)}" y="${y}" width="${(sx(r.tm)-sx(r.tw)).toFixed(1)}" height="14"/>`+
      `<rect class="spd-cmp" x="${X0}" y="${y+18}" width="${Math.max(0.8,sx(r.tc)-X0).toFixed(1)}" height="14"/>`+
      `<line class="spd-endl" x1="${sx(r.t).toFixed(1)}" x2="${sx(r.t).toFixed(1)}" y1="${y-3}" y2="${y+35}"/><text class="spd-tv" x="${(sx(r.t)+6).toFixed(1)}" y="${y+21}">${esc(msT(r.t))}</text></g>`);});
  const ax=14+rows.length*rowH+(nar?0:4);
  b+=`<line class="spd-ax" x1="${X0}" x2="${X1}" y1="${ax}" y2="${ax}"/>`+[0,5,10,15].filter(v=>v/1e3<=mx).map(v=>`<text class="spd-tk" x="${sx(v/1e3).toFixed(1)}" y="${ax+14}" text-anchor="${!v&&X0<20?'start':'middle'}">${v} ms</text>`).join('');
  el.innerHTML=(g==='cost'?`<div class="spd-stepin">${stepInHTML()}</div>`:'')+
    `<div class="spd-ch-t">${g==='cost'?tx('How long does the step take? Each row: requests × tokens each','这一步要多久？每行是一种放法：几个请求 × 每个几个 token'):tx(`How long is one step of ${MD.t8.name}?`,`${MD.t8.name} 走一步要多久？`)}</div>`+
    svgWrap(ax+22,b,tx('One step’s time: memory reading and compute for each way of filling the step','一步的耗时：每种放法下读显存和计算的时间'))+
    `<div class="legend-row spd-leg">${lg('spd-sw-w',tx('reading weights','读权重'),tx(`Time to read the 8B’s ${fd(MD.t8.W/1e9,1)} GB of weights (layers and LM head) at 60% of 3.35 TB/s. It is the same in every row: the weights are read once per step.`,`按 3.35 TB/s 的 60% 读 8B 的 ${fd(MD.t8.W/1e9,1)} GB 权重（各层和 LM head）要的时间。每一行都一样：一步只读一遍权重。`))}${lg('spd-sw-kv',tx('reading the KV cache','读 KV cache'),tx('Time to read each request’s KV cache: 1,024 tokens of context, 128 KiB per token. It grows with the number of requests.','读每个请求的 KV cache 要的时间：每个请求 1,024 个 token 的上下文，每个 token 128 KiB。请求越多，它越长。'))}${lg('spd-sw-c',tx('compute','计算'),tx('Arithmetic time at 50% of 989.5 TFLOP/s; it runs while the memory is read, and grows with the number of tokens in the step.','按 989.5 TFLOP/s 的 50% 算的计算时间，和读显存同时进行；一步里的 token 越多，它越长。'))}${lg('spd-sw-end',tx('thick line: the step’s time. Each row has two bars, reading memory and computing; the longer one counts','粗竖线：这一步的耗时。每一行里有读显存和计算两根条，取较长的那根'),tx('A step takes the longer of its memory time and its compute time.','一步的耗时取读显存时间和计算时间中较长的那个。'))}</div>`;}
const lg=(cls,txt,tip)=>`<span data-tipx="${esc(tip)}" tabindex="0"><i class="spd-sw ${cls}"></i><span>${txt}</span></span>`;
/* the round replay: what frame f shows. The three stages of the part-3 card, in its order: ① the draft model guesses, one chip a frame
   (frames 1..k); ② the target model's one step gives its own token at every position, all in one frame (k+1); ③ the guesses are compared
   from the left and what is kept fills in (frames k+2..), then the end */
function roundState(R,fr){const k=R.k, nf=nFrames(R), d=Math.min(fr,k), tg=fr>=k+1, c=Math.max(0,Math.min(fr-k-1,Math.min(R.acc+1,k))), done=fr>=nf;
  return {k,nf,d,tg,c,done,stg:done?4:fr<=k?1:fr===k+1?2:3};}
function rowsHTML(R,fr){const w=EXW[A.lang()==='zh'?'zh':'en'], st=roundState(R,fr), k=R.k, rej=R.acc<k, seenRej=rej&&st.c>R.acc;
  const cell=(cls,txt,tip)=>`<span class="spd-chip ${cls}"${tip?` data-tipx="${esc(tip)}"`:''}>${esc(txt)}</span>`;
  const ph='<span class="spd-chip spd-ph"></span>', tgtTok=j=>j<=R.acc?w.t[j]:(w.x[k]||[])[j-R.acc-1]||'·';
  // ① the draft model's guesses
  let dr='';
  for(let j=0;j<k;j++){
    if(j>=st.d){dr+=ph; continue;}
    let cls='spd-g', tip=tx(`Guess ${j+1}, written by the draft model in one small step.`,`猜测 ${j+1}，draft 模型走一小步写出来的。`);
    if(j<st.c){ if(j<R.acc){cls='spd-ok'; tip=tx(`Guess ${j+1}: the 8B writes the same token here, so the guess is kept.`,`猜测 ${j+1}：8B 在这里也写这个 token，收下。`);} else {cls='spd-bad'; tip=tx(`Guess ${j+1}: the 8B would write “${w.t[j]}” here, so this guess is rejected.`,`猜测 ${j+1}：8B 在这里会写“${w.t[j]}”，所以这个猜测被拒绝。`);} }
    else if(seenRej&&j>R.acc){cls='spd-drop'; tip=tx(`Guess ${j+1} is thrown away: it comes after the rejected guess.`,`猜测 ${j+1} 被丢掉：它排在被拒绝的猜测后面。`);}
    dr+=cell(cls,w.d[k][j],tip);}
  dr+='<span class="spd-chip spd-ph spd-ghost"></span>';
  // ② the target model's one step: the token it would write itself at each of the k + 1 positions
  let tr='';
  for(let j=0;j<=k;j++){
    if(!st.tg){tr+=ph; continue;}
    const txt=tgtTok(j);
    let cls='spd-tg', tip=j===0?tx(`Position 1: from the context, the 8B would write “${txt}”.`,`位置 1：看着上文，8B 会写“${txt}”。`):j===1?tx(`Position 2: from the context and guess 1, the 8B would write “${txt}”.`,`位置 2：看着上文和猜测 1，8B 会写“${txt}”。`):tx(`Position ${j+1}: from the context and guesses 1 to ${j}, the 8B would write “${txt}”.`,`位置 ${j+1}：看着上文和猜测 1 到 ${j}，8B 会写“${txt}”。`);
    if(j<R.acc){if(j<st.c) cls='spd-tg spd-agree';}
    else if(j===R.acc){if(rej?seenRej:st.done){cls='spd-own'; tip=rej?tx(`Position ${j+1}: the 8B’s own token “${txt}” takes the place of the rejected guess, and is kept.`,`位置 ${j+1}：8B 自己的“${txt}”换掉被拒绝的猜测，留下。`):tx(`Position ${j+1}, after the last guess: the 8B’s own next token, kept as well because every guess was right.`,`位置 ${j+1}，在最后一个猜测之后：8B 自己的下一个 token；猜测全对，它也留下。`);}}
    else if(seenRej){cls='spd-na'; tip=tx(`Position ${j+1}: computed in the same step, but from the rejected guess, so it is not used.`,`位置 ${j+1}：同一步也算了，可它是接着被拒绝的猜测算的，用不上。`);}
    tr+=cell(cls,txt,tip);}
  // ③ what is kept, filled in as the guesses are compared
  let out='', nk=0;
  for(let j=0;j<Math.min(st.c,R.acc);j++){out+=cell('spd-ok',w.t[j],tx('An accepted guess.','被收下的猜测。')); nk++;}
  if(rej?seenRej:st.done){out+=cell('spd-own',w.t[R.acc],tx('The 8B’s own token.','8B 自己的 token。')); nk++;}
  out+=ph.repeat(k+1-nk);
  const S3=(n,t,x)=>`<div class="spd-stl${st.stg===n?' on':st.stg>n?' done':''}"><span class="spd-stn">${['①','②','③'][n-1]}</span><b>${t}</b>${x?`<span class="spd-stx">${x}</span>`:''}</div>`;
  const row=h=>`<div class="spd-row${k>=8?' spd-wide':''}" style="--n:${k+1}">${h}</div>`;
  return `<div class="spd-ctx"><div class="spd-rl">${tx('Context','上文')}</div><span class="spd-ctxt">${esc(w.ctx)}</span></div>`+
    `<div class="spd-stg">${S3(1,tx('The draft model guesses','draft 模型猜'),tx(`${MD.d1.name} writes ${k} tokens, one small step each`,`${MD.d1.name} 一个一个写出 ${k} 个，每个走一小步`))}${row(dr)}</div>`+
    `<div class="spd-stg">${S3(2,tx('The target model checks, in one step','目标模型一次检查'),tx(`${MD.t8.name}: its own token at every position`,`${MD.t8.name} 算出每个位置它自己会写的 token`))}`+
      `<div class="spd-brk${st.tg?' on':''}"><span>${tx(`one 8B step, all ${k+1} positions at once`,`8B 一步，${k+1} 个位置一起算`)}</span></div>${row(tr)}</div>`+
    `<div class="spd-stg">${S3(3,tx('Keep','留下'),st.done?tx(`${R.acc+1} tokens from one 8B step`,`8B 跑一步，留下 ${R.acc+1} 个 token`):tx('matching guesses from the left, then the 8B’s token at the first mismatch','从左往右，一致的收下，第一个不一致处换成 8B 自己的'))}${row(out)}</div>`+
    `<div class="legend-row spd-leg spd-rleg">${lg('spd-sw-ok',tx('kept guess','收下的猜测'),tx('A guess the 8B agrees with.','8B 也会这样写的猜测。'))}${lg('spd-sw-bad',tx('rejected guess','被拒绝的猜测'),tx('The first guess that differs from the 8B’s own token.','第一个和 8B 自己的 token 不一样的猜测。'))}${lg('spd-sw-own',tx('the 8B’s own token, kept','8B 自己的 token，留下'),tx('At the first mismatch, or after the last guess when all are right.','在第一个不一致处；猜测全对时，在最后一个猜测之后。'))}${lg('spd-sw-drop',tx('dropped or unused','丢掉或用不上'),tx('Guesses after the rejected one, and the 8B’s tokens computed from it.','排在被拒绝的猜测后面的猜测，和 8B 接着它算出的 token。'))}</div>`;}
function roundTitle(R,fr){const st=roundState(R,fr), w=EXW[A.lang()==='zh'?'zh':'en'], pre=tx(`Example round, k = ${R.k}: `,`示例轮次，每轮猜 ${R.k} 个：`);
  if(st.done) return pre+tx(`one 8B step, ${R.acc+1} tokens kept`,`8B 跑一步，留下 ${R.acc+1} 个 token`);
  if(fr===0) return pre+tx('① the draft model is about to guess','① draft 模型准备开始猜');
  if(!st.tg) return tx(`① The draft model guesses: guess ${st.d} of ${R.k} written`,`① draft 模型猜：写好第 ${st.d} / ${R.k} 个`);
  if(st.c===0) return tx(`② The target model runs one step: its own token at all ${R.k+1} positions at once`,`② 目标模型跑一步：同时算出 ${R.k+1} 个位置上它自己会写的 token`);
  const j=st.c; return j<=R.acc?tx(`③ Guess ${j} matches the 8B: kept`,`③ 猜测 ${j} 和 8B 一样：收下`):tx(`③ Guess ${j} differs: the 8B writes “${w.t[j-1]}”, which is kept; the later guesses are dropped`,`③ 猜测 ${j} 不一样：8B 会写“${w.t[j-1]}”，留下它，后面的丢掉`);}
/* parts 3–4: the round's time against plain decode writing the same tokens, to scale, filled in frame by frame with the chips above
   (a draft step per guess in frames 1..k, the 8B's one step in frame k+1); plain decode's row adds a step for every token kept in ③ */
function rcostHTML(R,fr){const st=roundState(R,fr), c=roundCost(R), X1=W-132, mx=Math.max(c.tRound,c.plain), sx=t=>X1*t/mx;
  const tok=n=>tx(`${n} token${n>1?'s':''}`,`${n} 个 token`), GB=fd(MD.t8.W/1e9,1);
  const tipD=tx(`One draft step: the 1B reads its own ${fd(c.dW/1e9,2)} GB of weights and writes one guess, ${msT(c.ds[0])}.`,`draft 模型的一小步：1B 读自己 ${fd(c.dW/1e9,2)} GB 的权重，写出 1 个猜测，${msT(c.ds[0])}。`);
  const tipT=tx(`The 8B’s one step: reading its ${GB} GB of weights takes ${msT(c.v.tw)}; computing all ${R.k+1} positions takes only ${msT(c.v.tc)}, inside the reading time.`,`8B 的一步：读 ${GB} GB 权重要 ${msT(c.v.tw)}；${R.k+1} 个位置的计算只要 ${msT(c.v.tc)}，在读权重的时间里就做完了。`);
  const tipP=tx(`One step of plain decode: the 8B reads all ${GB} GB of weights and makes 1 token.`,`普通 decode 的一步：8B 读一遍 ${GB} GB 权重，出 1 个 token。`);
  let b=`<text class="spd-tl" x="0" y="11">${esc(tx('This round, guessing','投机解码这一轮'))}</text><text class="spd-tl" x="0" y="54">${esc(tx('Plain decode','普通 decode'))}</text>`, t=0;
  for(let j=0;j<st.d;j++){const a=sx(t), z=sx(t+c.ds[j]); b+=`<rect class="spd-drf" x="${a.toFixed(2)}" y="21" width="${Math.max(0.8,z-a-0.8).toFixed(2)}" height="9" data-tipx="${esc(tipD)}"/>`; t+=c.ds[j];}
  if(st.tg){const a=sx(c.td), z=sx(c.tRound); b+=`<rect class="spd-tgt" x="${a.toFixed(2)}" y="17" width="${Math.max(0.8,z-a).toFixed(2)}" height="17" data-tipx="${esc(tipT)}"/>`; t=c.tRound;}
  // the tokens kept so far, as row ③ fills in; plain decode's row grows with them (one step per kept token, at least the one-step reference)
  const nk=Math.min(st.c,R.acc)+((R.acc<R.k?st.c>R.acc:st.done)?1:0), np=Math.max(1,nk);
  if(st.d) b+=`<text class="spd-tv2" x="${(sx(t)+6).toFixed(1)}" y="30">${esc(msT(t)+(nk?` → ${tok(nk)}`:''))}</text>`;
  for(let i=0;i<np;i++){const a=sx(i*c.tPlain), z=sx((i+1)*c.tPlain); b+=`<rect class="spd-tgt" x="${a.toFixed(2)}" y="60" width="${Math.max(0.8,z-a-1.2).toFixed(2)}" height="17" data-tipx="${esc(tipP)}"/>`;}
  b+=`<text class="spd-tv2" x="${(sx(np*c.tPlain)+6).toFixed(1)}" y="73">${esc(msT(np*c.tPlain)+` → ${tok(np)}`)}</text>`;
  const note=tx(`Counting the thrown-away guesses, this round computes ${fd(c.flRound/c.flPlain,1)} times the arithmetic of plain decode for these ${c.n} tokens; what it saves is reading the 8B’s weights, once instead of ${c.n} times.`,`连同被丢掉的猜测，这一轮的计算量是普通 decode 写这 ${c.n} 个 token 的 ${fd(c.flRound/c.flPlain,1)} 倍；省下的是读 8B 权重的次数，1 次，而不是 ${c.n} 次。`);
  return `<div class="spd-ch-t">${tx(`Time: this round against plain decode (${MD.t8.name}, one user)`,`耗时：这一轮和普通 decode 比（${MD.t8.name}，只有一个用户）`)}</div>`+
    svgWrap(85,`<g transform="translate(0,3)">${b}</g>`,tx('The round’s time against plain decode writing the same tokens','这一轮的耗时，和普通 decode 写同样多 token 的耗时'))+
    `<div class="legend-row spd-leg">${lg('spd-sw-drf',tx('a draft step (1B)','draft 模型的一小步（1B）'),tipD)}${lg('spd-sw-tgt',tx('a target step (8B)','目标模型的一步（8B）'),tx(`An 8B step: almost all of it is reading the ${GB} GB of weights, however many positions it computes.`,`8B 的一步：时间几乎全花在读 ${GB} GB 权重上，不管它算几个位置。`))}</div>`+
    `<p class="small muted spd-rnote${st.done?'':' spd-later'}">${note}</p>`;}
function drawRound(){const el=$('#spd-rows',app); if(!el) return; const html=rowsHTML(rpR,RP.f); if(el._h!==html){el.innerHTML=html; el._h=html;}
  const rc=$('#spd-rcost',app), showC=mode==='guided'&&(sid()==='round'||sid()==='predict'); if(rc){rc.hidden=!showC; if(showC){const h=rcostHTML(rpR,RP.f); if(rc._h!==h){rc.innerHTML=h; rc._h=h;}}}
  const nf=nFrames(rpR), pos=$('#spd-rp-pos',app); if(pos){pos.max=String(nf); pos.value=String(RP.f); A.ui.setRange&&A.ui.setRange(pos);}
  const pb=$('#spd-rp-play',app); if(pb) pb.textContent=rpBtnTxt();
  $$('.seg-ctl[data-key="rpSpd"]',app).forEach(g=>A.ui.setPressed(g,rpSpd));
  const ex=$('#spd-rail [data-act="rp-play"]',app); if(ex) ex.textContent=rpBtnTxt();}
function drawAcc(){const el=$('#spd-acc',app), R=accRule(), fr=accFreq(accMode), m=accMode;
  const X0=40, cw=(W-X0)/5, H=200, top=m==='greedy'?1:0.6, sy=v=>H-20-v*(H-40)/top;   // greedy puts 100% on one word: its axis runs to 100%
  let b=`<line class="spd-ax" x1="${X0}" x2="${W}" y1="${sy(0)}" y2="${sy(0)}"/>`;
  (m==='greedy'?[0,0.25,0.5,0.75,1]:[0,0.2,0.4,0.6]).forEach(v=>{b+=`<text class="spd-tk" x="${X0-6}" y="${sy(v)+4}" text-anchor="end">${pc(v)}</text><line class="spd-grid" x1="${X0}" x2="${W}" y1="${sy(v)}" y2="${sy(v)}"/>`;});
  for(let i=0;i<5;i++){const x=X0+i*cw+cw*0.12, bw=cw*0.3, x2=x+bw+cw*0.08;
    b+=`<rect class="spd-q" x="${x.toFixed(1)}" y="${sy(R.q[i]).toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(0)-sy(R.q[i])).toFixed(1)}"/>`;
    if(m==='sample'){b+=`<rect class="spd-okb" x="${x2.toFixed(1)}" y="${sy(R.mn[i]).toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(0)-sy(R.mn[i])).toFixed(1)}"/>`;
      const rr=R.rej*R.res[i]; if(rr>1e-9) b+=`<rect class="spd-rsb" x="${x2.toFixed(1)}" y="${sy(R.mn[i]+rr).toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(R.mn[i])-sy(R.mn[i]+rr)).toFixed(1)}"/>`;}
    else if(m==='greedy'){const v=i?0:1; if(v) b+=`<rect class="spd-okb" x="${x2.toFixed(1)}" y="${sy(1).toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(0)-sy(1)).toFixed(1)}"/><text class="spd-tv2" x="${(x2+bw/2).toFixed(1)}" y="${(sy(1)-6).toFixed(1)}" text-anchor="middle">100%</text>`;}
    else {const bad=Math.abs(R.q[i]-R.p[i])>0.005; b+=`<rect class="${bad?'spd-badb':'spd-okb'}" x="${x2.toFixed(1)}" y="${sy(fr[i]).toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(0)-sy(fr[i])).toFixed(1)}"/>`;}
    const tp=m==='greedy'?(i?0:1):R.p[i];
    if(m!=='greedy') b+=`<line class="spd-pt" x1="${(x2-3).toFixed(1)}" x2="${(x2+bw+3).toFixed(1)}" y1="${sy(R.p[i]).toFixed(1)}" y2="${sy(R.p[i]).toFixed(1)}"/>`;
    b+=`<text class="spd-tl" x="${(X0+i*cw+cw/2).toFixed(1)}" y="${H-4}" text-anchor="middle">${esc(wd(i))}</text>`;
    if(m!=='greedy') b+=`<text class="spd-tv2" x="${(x2+bw/2).toFixed(1)}" y="${(Math.min(sy(fr[i]),sy(R.p[i]))-6).toFixed(1)}" text-anchor="middle">${pc(fr[i],1)}</text>`;
    void tp;}
  const t=m==='greedy'?tx('Greedy: what the output is, for each candidate next word','贪心：每个候选的下一个词，输出的概率'):m==='sample'?tx(`Sampling with the acceptance rule: how often each word comes out, ${A.fmt.int(ACC_N)} draws`,`按接受规则采样：每个词出现的频率，抽 ${A.fmt.int(ACC_N)} 次`):tx(`No check: how often each word comes out, ${A.fmt.int(ACC_N)} draws`,`不检查：每个词出现的频率，抽 ${A.fmt.int(ACC_N)} 次`);
  el.innerHTML=`<div class="spd-ch-t">${t}</div><p class="small muted spd-acc-ctx">${tx('The next word after “The weather today is”: 5 candidates.','“今天天气”后面的下一个词：5 个候选。')}</p>`+svgWrap(H,b,t)+
    `<div class="legend-row spd-leg">${lg('spd-sw-q',tx('draft model’s probability q','draft 模型的概率 q'),tx('The draft model’s probability for each word; its guess is drawn from these.','draft 模型给每个词的概率，猜测按它来抽。'))}${m!=='greedy'?lg('spd-sw-pt',tx('thick line: target model’s probability p','粗横线：目标模型的概率 p'),tx('The target model’s probability for each word: what the output must follow.','目标模型给每个词的概率，输出必须照它来。')):''}${m==='nocheck'&&!R.q.some((x,i)=>Math.abs(x-R.p[i])<=0.005)?'':lg('spd-sw-ok',m==='nocheck'?tx('output, close to the target','输出，和目标模型接近'):tx('accepted guess','被收下的猜测'),m==='sample'?tx(`The share that comes from accepted guesses: ${M('\\min(p, q)')} for each word.`,`来自被收下的猜测的那部分：每个词是 ${M('\\min(p, q)')}。`):tx('How often the word comes out.','这个词出现的频率。'))}${m==='sample'?lg('spd-sw-rs',tx('redrawn after a rejection','被拒绝后重新抽到的'),tx(`The share that comes from redrawing after a rejection: the rejection chance times the leftover ${M('\\max(0, p - q)')}, rescaled to sum to 1.`,`被拒绝后重新抽到的那部分：拒绝的概率乘以剩下的 ${M('\\max(0, p - q)')}（缩放到总和为 1）。`)):''}${m==='nocheck'?lg('spd-sw-bad',tx('output, far from the target','输出，和目标模型差得多'),tx('This word comes out with the draft’s probability, not the target’s.','这个词按 draft 模型的概率出现，而不是目标模型的。')):''}</div>`;}
function drawKCurve(){const el=$('#spd-kcurve',app), pb=plotBox(el,210), X0=44, X1=pb.w-56, H=pb.h, sx=k=>X0+(X1-X0)*(k-1)/7, sy=v=>H-24-(v-1)*(H-44)/8, kk=rpR.k;
  let b=`<line class="spd-ax" x1="${X0}" x2="${X1}" y1="${sy(1)}" y2="${sy(1)}"/>`;
  [1,3,5,7,9].forEach(v=>{b+=`<line class="spd-grid" x1="${X0}" x2="${X1}" y1="${sy(v)}" y2="${sy(v)}"/><text class="spd-tk" x="${X0-6}" y="${sy(v)+4}" text-anchor="end">${v}</text>`;});
  KS.forEach(k=>{b+=`<text class="spd-tk" x="${sx(k)}" y="${H-8}" text-anchor="middle">${k}</text>`;});
  b+=`<path class="spd-up" d="M${sx(1)},${sy(2)} L${sx(8)},${sy(9)}"/>`;
  SIM_A.forEach(a=>{const cur=a===0.8; b+=`<path class="spd-kl${cur?' spd-cur':''}" d="${KS.map((k,i)=>(i?'L':'M')+sx(k).toFixed(1)+','+sy(expTok(a,k)).toFixed(1)).join(' ')}"/>`+
    KS.map(k=>`<circle class="spd-dot${cur?' spd-cur':''}" cx="${sx(k).toFixed(1)}" cy="${sy(simMean(a,k)).toFixed(1)}" r="${cur?3.4:2.6}"/>`).join('')+
    `<text class="spd-tl2" x="${X1+6}" y="${sy(expTok(a,8))+4}">α = ${a}</text>`;});
  b+=`<line class="spd-kmark" x1="${sx(kk)}" x2="${sx(kk)}" y1="${sy(1)}" y2="${sy(9)}"/><circle class="spd-me" cx="${sx(kk)}" cy="${sy(expTok(0.8,kk))}" r="5"/>`;
  el.innerHTML=`<div class="spd-ch-t">${tx('How many tokens does one target step yield on average, for each number of guesses per round k?','每轮猜 k 个时，目标模型跑一次平均得到几个 token？')}</div>`+svgWrap(H,b,tx('Tokens per target step against k for α 0.6 to 0.9','α 从 0.6 到 0.9 时，目标模型跑一次的 token 数随 k 的变化'),pb.w)+
    `<div class="spd-axl small muted">${tx('Across: guesses per round k. Up: tokens per target step.','横轴：每轮猜的 token 数 k。纵轴：目标模型跑一次得到的 token 数。')}</div>`+
    `<div class="legend-row spd-leg">${lg('spd-sw-line',tx('formula, one line per α','公式，每种 α 一条线'),tx(`The expected tokens per target step: ${M('\\frac{1 - \\alpha^{k+1}}{1 - \\alpha}')}.`,`平均每次得到的 token 数：${M('\\frac{1 - \\alpha^{k+1}}{1 - \\alpha}')}。`))}${lg('spd-sw-dot',tx(`dots: ${A.fmt.int(SIM_N)} simulated rounds`,`点：模拟 ${A.fmt.int(SIM_N)} 轮的平均`),tx('Each dot averages 2,000 seeded rounds in which each guess is accepted with chance α if all before it were.','每个点是 2,000 个带种子的模拟轮次的平均：前面都被收下时，下一个猜测以概率 α 被收下。'))}${lg('spd-sw-cur',tx('thick line α = 0.8, ring at the chosen k','粗线是 α = 0.8，圆圈是选中的 k'),tx('The setting of this part.','这一部分的设置。'))}${lg('spd-sw-up',tx('dashed: the most, every guess kept','虚线：最多能得到的，猜测全对'),tx(`The most one step can yield: all ${M('k')} guesses plus the extra token the target model computed after the last one.`,`跑一次最多能得到的：${M('k')} 个猜测全收下，再加目标模型在最后一个猜测之后多算出的 1 个 token。`))}</div>`;}
function drawBCurve(){const el=$('#spd-bcurve',app), s=S, pb=plotBox(el,210), X0=44, X1=pb.w-70, H=pb.h, n=CURVE_B.length, sx=i=>X0+(X1-X0)*i/(n-1);
  const lines=(mode==='challenge'?['ngram','d1','d3'].filter(d=>d===s.dr):['ngram','d1','d3']).map(d=>({d,a:(s.dr===d?s.a:ALPHA[s.tgt][d]),pts:CURVE_B.map(B=>compute({...s,B,dr:d,a:(s.dr===d?s.a:ALPHA[s.tgt][d])}).speed)}));
  const ymax=Math.max(2,Math.ceil(Math.max(...lines.flatMap(l=>l.pts))*2)/2), sy=v=>H-24-v*(H-40)/ymax;
  let b='';
  for(let v=0;v<=ymax+1e-9;v+=0.5) b+=`<line class="spd-grid" x1="${X0}" x2="${X1}" y1="${sy(v)}" y2="${sy(v)}"/><text class="spd-tk" x="${X0-6}" y="${sy(v)+4}" text-anchor="end">${fd(v,1)}</text>`;
  b+=`<rect class="spd-slow" x="${X0}" y="${sy(1)}" width="${X1-X0}" height="${sy(0)-sy(1)}"/><line class="spd-one" x1="${X0}" x2="${X1}" y1="${sy(1)}" y2="${sy(1)}"/>`;
  CURVE_B.forEach((B,i)=>{b+=`<text class="spd-tk" x="${sx(i)}" y="${H-8}" text-anchor="middle">${B}</text>`;});
  lines.forEach(l=>{const cls='spd-bl spd-bl-'+l.d+(s.dr===l.d?' spd-cur':'');
    b+=`<path class="${cls}" d="${l.pts.map((v,i)=>(i?'L':'M')+sx(i).toFixed(1)+','+sy(v).toFixed(1)).join(' ')}"/>`;
    l.pts.forEach((v,i)=>{if(v<1) b+=`<circle class="spd-baddot" cx="${sx(i).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="3"/>`;});
    b+=`<text class="spd-tl2" x="${X1+6}" y="${sy(l.pts[n-1])+4}">${esc(l.d==='ngram'?'n-gram':l.d==='d1'?'1B':'3B')}</text>`;});
  const bi=CURVE_B.indexOf(s.B);
  if(bi>=0){b+=`<line class="spd-kmark" x1="${sx(bi)}" x2="${sx(bi)}" y1="${sy(0)}" y2="${sy(ymax)}"/>`; if(s.dr!=='none'){const r=compute(s); b+=`<circle class="spd-me${r.speed<1?' spd-me-bad':''}" cx="${sx(bi)}" cy="${sy(r.speed)}" r="5"/>`;}}
  el.innerHTML=`<div class="spd-ch-t">${tx(`How many times plain decode’s tokens per second? ${tName(s.tgt)}, k = ${s.k}, for each batch size`,`每秒 token 数是普通 decode 的几倍？${tName(s.tgt)}，每轮猜 ${s.k} 个，各种 batch 大小`)}</div>`+svgWrap(H,b,tx('Speedup against batch size for each guesser','各种猜法的加速倍数随 batch 大小的变化'),pb.w)+
    `<div class="spd-axl small muted">${tx('Across: requests per step (batch). Up: tokens per second as a multiple of plain decode’s.','横轴：一步里的请求数（batch）。纵轴：每秒 token 数是普通 decode 的几倍。')}</div>`+
    `<div class="legend-row spd-leg">${lines.map(l=>lg('spd-sw-'+(l.d==='ngram'?'ng':l.d),tx(`${l.d==='ngram'?'n-gram lookup':l.d==='d1'?'1B draft':'3B draft'}, α ${aT(l.a)}`,`${l.d==='ngram'?'n-gram 查找':l.d==='d1'?'1B draft 模型':'3B draft 模型'}，α ${aT(l.a)}`),l.d==='ngram'?D_NGRAM()+tx(` This lab assumes α ${aT(l.a)}.`,`本 Lab 假设 α 为 ${aT(l.a)}。`):MD[l.d].name+tx(`. This lab assumes α ${aT(l.a)}.`,`。本 Lab 假设 α 为 ${aT(l.a)}。`))).join('')}${lg('spd-sw-one',tx('dashed: plain decode (1 time)','虚线：普通 decode（1 倍）'),tx('Below this line guessing makes it slower.','低于这条线，投机解码反而更慢。'))}${lg('spd-sw-bad',tx('red: slower than plain decode','红色：比普通 decode 慢'),tx('A point below 1 time.','低于 1 倍的点。'))}${lg('spd-sw-me',tx('ring: current setting','圆圈：当前设置'),tx('The guesser and batch chosen now.','现在选的猜法和 batch。'))}</div>`;}
/* when the short run and the average disagree on which is faster, say so (R6.10) */
function tlNote(tl,s0){const s=s0||S, r=compute(s); if(s.dr==='none') return ''; const avgFaster=r.speed>1, runFaster=tl.spec<tl.plain; if(avgFaster===runFaster&&!near1(r.speed)) return '';
  if(avgFaster===runFaster) return tx(` On average the two are about equally fast (${xT(r.speed)}).`,`按平均算，两者几乎一样快（${xT(r.speed)}）。`);
  return tx(` This short run of ${OUT} tokens is a little ${runFaster?'faster':'slower'} than the average; on average guessing gives ${xT(r.speed)} plain decode’s speed.`,`这 ${OUT} 个 token 的短运行比平均略${runFaster?'快':'慢'}；按平均算，速度是普通 decode 的 ${xT(r.speed)}。`);}
function drawTime(){const el=$('#spd-time',app), r=compute(S), tl=timeline(r), pb=plotBox(el,118), X0=0, X1=pb.w-62, mx=Math.max(tl.plain,tl.spec), sx=t=>X0+(X1-X0)*t/mx, H=pb.h;
  let b=`<text class="spd-tl" x="0" y="11">${esc(tx(`Plain decode: ${OUT} target steps`,`普通 decode：目标模型走 ${OUT} 步`))}</text><text class="spd-tl" x="0" y="51">${esc(tx('Guess and check','先猜再检查'))}</text>`;
  for(let i=0;i<OUT;i++){const a=sx(i*r.tPlain), c=sx((i+1)*r.tPlain); b+=`<rect class="spd-tgt" x="${a.toFixed(2)}" y="17" width="${Math.max(0.6,c-a-0.8).toFixed(2)}" height="16"/>`;}
  b+=`<text class="spd-tv" x="${(sx(tl.plain)+5).toFixed(1)}" y="29">${esc(msT(tl.plain))}</text>`;
  if(S.dr==='none') b+=`<text class="spd-tk" x="${X0}" y="80">${esc(tx('no guesser chosen','没有选猜法'))}</text>`;
  else tl.rounds.forEach(rd=>{let t=rd.t0;
    if(S.dr!=='ngram') for(let j=0;j<S.k;j++){const d=r.td/S.k, a=sx(t), c=sx(t+d), wasted=j>=rd.acc; b+=`<rect class="spd-drf${wasted?' spd-waste':''}" x="${a.toFixed(2)}" y="76" width="${Math.max(0.5,c-a-0.4).toFixed(2)}" height="10"/>`; t+=d;}
    const a=sx(t), c=sx(rd.t1); b+=`<rect class="spd-tgt" x="${a.toFixed(2)}" y="70" width="${Math.max(0.6,c-a-0.8).toFixed(2)}" height="22"/>`;
    if(c-a>8) b+=`<text class="spd-cnt" x="${((a+c)/2).toFixed(1)}" y="65" text-anchor="middle">${rd.acc+1}</text>`;});
  if(S.dr!=='none') b+=`<text class="spd-tv" x="${(sx(tl.spec)+5).toFixed(1)}" y="85">${esc(msT(tl.spec))}</text>`;
  b+=`<line class="spd-ax" x1="${X0}" x2="${X1}" y1="${H-14}" y2="${H-14}"/><text class="spd-tk" x="${X0}" y="${H-2}">0</text><text class="spd-tk" x="${X1}" y="${H-2}" text-anchor="end">${esc(msT(mx))}</text>`;
  el.innerHTML=`<div class="spd-ch-t">${tx(`Time for one request’s first ${OUT} output tokens, ${tName(S.tgt)}, batch ${S.B}`,`一个请求的前 ${OUT} 个输出 token 要多久？${tName(S.tgt)}，batch ${S.B}`)}</div>`+svgWrap(H+8,`<g transform="translate(0,4)">${b}</g>`,tx('Timeline: plain decode against guess and check','时间线：普通 decode 和先猜再检查'),pb.w)+
    `<div class="legend-row spd-leg">${lg('spd-sw-tgt',tx('target model step','目标模型的一步'),tx('One step of the target model: one token in plain decode; in a round, the check of all guesses.','目标模型的一步：普通 decode 时出 1 个 token；一轮里是检查全部猜测。'))}${S.dr==='d1'||S.dr==='d3'?lg('spd-sw-drf',tx('draft model step','draft 模型的一步'),tx('One step of the draft model: one guess.','draft 模型的一步：猜 1 个。'))+lg('spd-sw-waste',tx('red: a guess that was thrown away','红色：被丢掉的猜测'),tx('Draft time spent on a guess that was rejected or came after a rejection.','花在被拒绝、或排在被拒绝之后的猜测上的 draft 时间。')):''}${S.dr!=='none'?lg('spd-sw-cnt',tx('number: tokens that round kept','数字：这一轮留下的 token 数'),tx('Accepted guesses plus the target model’s own token.','被收下的猜测，加上目标模型自己的 1 个 token。')):''}</div>`+
    `<p class="small muted spd-tnote">${S.dr==='none'?'':tx(`${dName(S.dr)}, acceptance rate ${aS(S.a)}: these ${tl.rounds.length} rounds keep ${tl.made} tokens in all, close to the average.`,`${S.dr==='ngram'?dName(S.dr):S.dr==='d1'?'1B':'3B'}，接受率 ${aS(S.a)}：这 ${tl.rounds.length} 轮共留下 ${tl.made} 个 token，接近平均水平。`)+tlNote(tl,S)}</p>`;}   // how the rounds are picked: the end notes

/* metric cards */
const MET=[
 {k:'E',label:()=>tx('Average tokens per target step','目标模型跑一次平均出的 token 数'),v:r=>fd(r.E,2),sub:r=>r.s.dr==='none'?tx('no guessing','不猜'):tx(`α ${aT(r.s.a)}, k = ${r.s.k}, at most ${r.s.k+1}`,`α ${aT(r.s.a)}，每轮猜 ${r.s.k} 个，最多 ${r.s.k+1} 个`),tip:()=>tx('The tokens one target step yields on average: the accepted guesses plus the target model’s own token. At most one more than the guesses.','目标模型跑一次平均得到的 token 数：被收下的猜测，加上目标模型自己的 1 个。最多比猜的多 1 个。')},
 {k:'round',label:()=>tx('Time per round','每轮的耗时'),v:r=>msP(r.tRound,r.tPlain),sub:r=>tx(`plain decode step ${msP(r.tPlain,r.tRound)}`,`普通 decode 一步 ${msP(r.tPlain,r.tRound)}`),tip:()=>tx(`${M('k')} draft steps plus one target step that checks ${M('k+1')} tokens per request.`,`draft 模型走 ${M('k')} 步，加上目标模型检查每个请求 ${M('k+1')} 个 token 的那一步。`)},
 {k:'tps',label:()=>tx('Tokens per second, each request','每个请求每秒的 token 数'),v:r=>fd(r.tps,1),sub:r=>tx(`plain decode ${fd(r.tpsPlain,1)}`,`普通 decode ${fd(r.tpsPlain,1)}`),tip:()=>tx('How fast one user sees the answer appear: tokens per target step divided by the time per round.','一个用户看到回答出现的速度：跑一次的 token 数除以每轮的耗时。')},
 {k:'speed',label:()=>tx('Speed vs plain decode','速度是普通 decode 的'),v:r=>xT(r.speed),sub:r=>near1(r.speed)?tx('about the same as plain decode','和普通 decode 几乎一样'):r.speed<1?tx('slower than plain decode','比普通 decode 慢'):tx('tokens per second, with guessing vs without','每秒 token 数，用投机解码和不用相比'),tip:()=>tx('Tokens per second with guessing divided by plain decode’s, for the same requests.','同样的请求，用了投机解码以后每秒的 token 数除以普通 decode 的。')}];
const numOf=s=>parseFloat(String(s).replace(/,/g,''));
function drawMetrics(r){const host=$('#spd-metrics',app);
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  const showAll=mode!=='guided'||step>=SIX('batch'), showE=showAll||(mode==='guided'&&sid()==='expect');
  host.hidden=!showE;
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), v=m.v(r), d=el.querySelector('.d');
    el.hidden=m.k!=='E'&&!showAll;
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=m.sub(r); el.dataset.tipx=m.tip();
    el.classList.toggle('spd-mbad',m.k==='speed'&&r.speed<0.995);
    const a=numOf(shownPrev[m.k]), b=numOf(v), dt=userAct&&isFinite(a)&&isFinite(b)?(A.fmt.delta(a,b,{dp:m.k==='tps'?1:2})||''):'';
    if(dt){d.textContent=dt; d.classList.add('on'); d._at=performance.now(); clearTimeout(d._t); d._t=setTimeout(()=>{d.classList.remove('on'); d.textContent='';},2600);}
    else if(!userAct||performance.now()-(d._at||0)>50){d.classList.remove('on'); d.textContent='';}   // a change that left this number as it was clears an older delta
    shownPrev[m.k]=v;});}
/* the state the metric cards and the speed charts use in guided parts: parts 5–6 use the 8B with the 1B at α 0.8 (part 5's setting) */
function viewState(){if(mode==='guided'&&step<SIX('batch')) return {...DEFAULT,a:0.8,k:rpR.k}; return S;}
function mathHTML(r){
  const rows=[], s=r.s, gid=mode==='guided'?sid():'';
  if(gid==='mission') return '';
  const m=gid==='cost'?MD.t8:r.T, Bq=gid==='cost'?COSTB[costN][0]:s.B, nt=gid==='cost'?COSTB[costN][1]:(s.dr==='none'?1:s.k+1), nG=gid==='cost'?1:r.n;
  const one=stepT(m,Bq,nt,CTX,nG), nT=Bq*nt, G=v=>fd(v/1e9,v>=1e11?1:v>=1e10?2:3);
  const fl1=2*m.PL*nT, fl2=2*m.PH*nT, fl3=one.fl-fl1-fl2, tokR=Bq*(CTX+nt), per=nG>1?`${nG} \\times `:'', U=one.fl>=1e12?1e12:1e9, UN=U>1e9?'TFLOP':'GFLOP', G2=v=>fd(v/U,v/U>=100?1:v/U>=10?2:3);
  rows.push(`<div class="spd-mh">${gid==='cost'?tx(`One 8B step with ${costN} token${costN>1?'s':''} (${Bq} request${Bq>1?'s':''} × ${nt})`,`8B 走一步，处理 ${costN} 个 token（${Bq} 个请求 × 每个 ${nt} 个）`):tx(`The target model’s check step: ${nT} tokens (${Bq} request${Bq>1?'s':''} × ${nt})`,`目标模型检查的那一步：${nT} 个 token（${Bq} 个请求 × 每个 ${nt} 个）`)}${nG>1?tx(`, split over ${nG} GPUs`,`，分在 ${nG} 张卡上`):''}</div>`+
    `<p>${tx(`The arithmetic. Each token is multiplied by the ${G(m.PL)} billion parameters of the layers (${M('P_{layers}')}) and the ${G(m.PH)} billion of the LM head (${M('P_{head}')}), 2 FLOPs per parameter; attention adds the work of each new token looking at the ${A.fmt.int(CTX)} tokens before it:`,`计算量。每个 token 都要和各层的 ${G(m.PL)}B 个参数（${M('P_{layers}')}）、LM head 的 ${G(m.PH)}B 个参数（${M('P_{head}')}）相乘，每个参数 2 次运算；attention 再加上每个新 token 看前面 ${A.fmt.int(CTX)} 个 token 的计算：`)}</p>`+
    EQ(tx('FLOPs of the step','这一步的 FLOPs'),[`2 P_{layers}\\, n + 2 P_{head}\\, n + \\text{attention}`,`${G2(fl1)} + ${G2(fl2)} + ${G2(fl3)}\\ \\text{${UN}}`,`${G2(one.fl)}\\ \\text{${UN}}`])+
    `<p>${tx(`The bytes read. The weights once (${fd(m.W/1e9,2)} GB), plus the KV cache of every token the step looks at: ${A.fmt.int(tokR)} tokens, ${f(m.KV/1024,0)} KiB each:`,`读的字节。权重读一次（${fd(m.W/1e9,2)} GB），加上这一步要看的每个 token 的 KV cache：${A.fmt.int(tokR)} 个 token，每个 ${f(m.KV/1024,0)} KiB：`)}</p>`+
    EQ(tx('bytes of the step','这一步读的字节'),[`W + \\text{KV per token} \\times \\text{tokens}`,`${fd(m.W/1e9,2)} + ${fd(one.kvB/1e9,3)}\\ \\text{GB}`,`${fd(one.by/1e9,2)}\\ \\text{GB}`])+
    `<p>${tx('Compute time. This lab assumes 50% of the dense BF16 peak; the FLOPs divided by that:','计算时间。本 Lab 假设能用到稠密 BF16 峰值的 50%，FLOPs 除以它：')}</p>`+
    EQ(tx('compute time','计算时间')+' '+M('t_c'),[`\\frac{\\text{FLOPs}}{${per}989.5\\ \\text{TFLOP/s} \\times 0.5}`,`\\frac{${G2(one.fl)}\\ \\text{${UN}}}{${per}494.75\\ \\text{TFLOP/s}}`,`${tms(one.tc)}\\ \\text{ms}`])+
    `<p>${tx('Memory time. This lab assumes 60% of the peak memory bandwidth; the bytes divided by that:','读显存时间。本 Lab 假设能用到显存带宽峰值的 60%，字节数除以它：')}</p>`+
    EQ(tx('memory time','读显存时间')+' '+M('t_m'),[`\\frac{\\text{bytes}}{${per}3.35\\ \\text{TB/s} \\times 0.6}`,`\\frac{${fd(one.by/1e9,2)}\\ \\text{GB}}{${per}2.01\\ \\text{TB/s}}`,`${tms(one.tm)}\\ \\text{ms}`])+
    EQ(tx('step time','一步的耗时')+' '+M('t'),[`\\max(t_c, t_m)`,`${tms(one.t)}\\ \\text{ms}`]));
  if(gid==='round'||gid==='predict'){const R=rpR, c=roundCost(R), Gf=v=>fd(v/1e9,v>=1e11?0:v>=1e10?1:2), GW=v=>fd(v/1e9,2);
    rows.push(`<div class="spd-mh">${tx(`The example round against plain decode: ${R.k} guesses, ${R.acc} kept, ${c.n} tokens`,`示例这一轮和普通 decode 比（猜 ${R.k} 个，收下 ${R.acc} 个，得到 ${c.n} 个 token）`)}</div>`+
      `<p>${tx(`Time. The 1B’s ${R.k} draft steps, each priced the same way with its own weights and KV cache, plus the 8B’s check step above. Plain decode writes the same ${c.n} tokens in ${c.n} steps of ${tms(c.tPlain)} ms:`,`时间。1B 的 ${R.k} 小步（每一步也按上面的方法算，用它自己的权重和 KV cache），加上上面 8B 检查的一步。普通 decode 写同样 ${c.n} 个 token，要走 ${c.n} 步，每步 ${tms(c.tPlain)} ms：`)}</p>`+
      EQ(tx('time of the round','这一轮的耗时'),['\\sum_{j=1}^{k} t_{draft,j} + t_{check}',`${tms(c.td)}\\ \\text{ms} + ${tms(c.tv)}\\ \\text{ms}`,`${tms(c.tRound)}\\ \\text{ms}`])+
      EQ(tx('plain decode, same tokens','普通 decode 写同样多 token'),[`n \\times t_{plain}`,`${c.n} \\times ${tms(c.tPlain)}\\ \\text{ms}`,`${tms(c.plain)}\\ \\text{ms}`])+
      `<p>${tx(`Arithmetic. The round computes the draft’s ${R.k} steps and all ${R.k+1} positions of the check, the thrown-away guesses included; plain decode computes ${c.n} positions, one per step (in GFLOP):`,`计算量。这一轮要算 draft 的 ${R.k} 小步，和检查时全部 ${R.k+1} 个位置（包括被丢掉的猜测）；普通 decode 一步算 1 个位置，一共 ${c.n} 个（单位 GFLOP）：`)}</p>`+
      EQ(tx('arithmetic of the round over plain decode','计算量是普通 decode 的几倍'),['\\frac{\\text{FLOPs}_{draft} + \\text{FLOPs}_{check}}{n \\times \\text{FLOPs}_{plain}}',`\\frac{${Gf(c.flDraft)} + ${Gf(c.flCheck)}}{${c.n} \\times ${Gf(c.flPlain/c.n)}}`,fd(c.flRound/c.flPlain,2)])+
      `<p>${tx(`What the round saves is reading weights, and reading them is what sets the time of these steps:`,`这一轮省下的是读权重，而这几步的耗时正是由读权重决定的：`)}</p>`+
      EQ(tx('weights read in the round','这一轮读的权重'),[`W_{8B} + k \\times W_{1B}`,`${GW(MD.t8.W)} + ${R.k} \\times ${GW(c.dW)}\\ \\text{GB}`,`${GW(MD.t8.W+R.k*c.dW)}\\ \\text{GB}`])+
      EQ(tx('weights read by plain decode','普通 decode 读的权重'),[`n \\times W_{8B}`,`${c.n} \\times ${GW(MD.t8.W)}\\ \\text{GB}`,`${GW(c.n*MD.t8.W)}\\ \\text{GB}`]));}
  if(gid==='cost'||gid==='round'||gid==='predict') return rows.join('');
  if(s.dr!=='none'){
    rows.push(`<div class="spd-mh">${tx('Tokens per target step','目标模型跑一次的 token 数')}</div><p>${tx(`Token ${M('j+1')} of a round comes only if the first ${M('j')} guesses were all accepted, with chance ${M('\\alpha^j')}; adding these chances:`,`一轮的第 ${M('j+1')} 个 token，要前 ${M('j')} 个猜测都被收下才有，概率是 ${M('\\alpha^j')}；把这些概率加起来：`)}</p>`+
      EQ(tx('tokens per step','跑一次的 token 数')+' '+M('E'),['\\frac{1 - \\alpha^{k+1}}{1 - \\alpha}',`\\frac{1 - ${aT(s.a)}^{${s.k+1}}}{1 - ${aT(s.a)}}`,fd(r.E,4)]));}
  if(gid==='expect'||gid==='same') return rows.join('');
  if(s.dr!=='none'){const close=tms(r.tPlain)===tms(r.tRound), tq=v=>close?fd(v*1e3,4):tms(v);   // the two times round to the same value: show more digits so the row recomputes (R6.6)
    rows.push(`<div class="spd-mh">${tx('Time per round','每轮的耗时')}</div><p>${s.dr==='ngram'?tx('n-gram lookup costs no GPU time, so a round is one target step:','n-gram 查找不占 GPU 时间，所以一轮就是目标模型的一步：'):tx(`${M('k')} draft steps, each priced the same way as above with the draft model’s weights and its own KV cache (the first takes ${tms(r.td1)} ms), plus the check:`,`draft 模型走 ${M('k')} 步，每一步也按上面同样的方法算，用的是 draft 模型的权重和它自己的 KV cache（第一步 ${tms(r.td1)} ms），再加上检查的那一步：`)}</p>`+
      EQ(tx('time per round','每轮的耗时')+' '+M('t_r'),[s.dr==='ngram'?'t_{check}':'\\sum_{j=1}^{k} t_{draft,j} + t_{check}',s.dr==='ngram'?`${tq(r.v.t)}\\ \\text{ms}`:`${tq(r.td)}\\ \\text{ms} + ${tq(r.v.t)}\\ \\text{ms}`,`${tq(r.tRound)}\\ \\text{ms}`])+
      `<div class="spd-mh">${tx('Speed vs plain decode','速度是普通 decode 的几倍')}</div><p>${tx(`Plain decode makes one token per request in a step of ${tq(r.tPlain)} ms; a round makes ${M('E')} tokens in ${M('t_r')}:`,`普通 decode 每一步给每个请求出 1 个 token，一步 ${tq(r.tPlain)} ms；一轮出 ${M('E')} 个 token，用时 ${M('t_r')}：`)}</p>`+
      EQ(tx('speedup','加速倍数'),['\\frac{E \\times t_{plain}}{t_r}',close?`\\frac{${fd(r.E,4)} \\times ${fd(r.tPlain*1e3,4)}}{${fd(r.tRound*1e3,4)}}`:`\\frac{${fd(r.E,4)} \\times ${tms(r.tPlain)}}{${tms(r.tRound)}}`,close?fd(r.speed,4):fd(r.speed,2)]));}
  return rows.join('');}

/* ---------- render ---------- */
function stageTitle(){const g=mode==='guided'?sid():'';
  if(g==='mission') return tx(`One decode step of ${MD.t8.name} on one H100`,`${MD.t8.name} 在一张 H100 上的一步 decode`);
  if(g==='cost') return tx(`One step of ${MD.t8.name} on one H100: what it holds, and how long it takes`,`${MD.t8.name} 在一张 H100 上的一步：放进几个 token，要多久`);
  if(g==='round'||g==='predict') return roundTitle(rpR,RP.f);
  if(g==='expect') return RP.on?roundTitle(rpR,RP.f):tx('Average tokens per target step for each k','每种 k 下，目标模型跑一次平均得到的 token 数');
  if(g==='same') return tx('The acceptance rule on one position','一个位置上的接受规则');
  return tx(`Guessing with ${S.dr==='none'?'no guesser':dName(S.dr)}: ${tName(S.tgt)}, batch ${S.B}`,`${S.dr==='none'?'不猜':'用 '+dName(S.dr)+'来猜'}：${tName(S.tgt)}，batch ${S.B}`);}
function syncControls(){const c=$('#spd-controls',app); if(!c) return;
  const lock=$('#spd-lock',c), ctls=$('#spd-ctls',c), rst=$('#spd-reset-wrap',c), early=mode==='guided'&&step<SIX('batch');
  const msg=mode==='challenge'?tx('Fixed in this challenge: the 70B on 4 H100s, 128 requests per step, 1,024 tokens of context each, and the α of each guesser. The guesser and k are in the task card.','这道挑战题固定：4 张 H100 上的 70B，一步 128 个请求，每个 1,024 个 token 的上下文，以及每种猜法的 α。猜法和 k 在任务卡里选。')
    :early?tx(`This part uses a fixed example: ${MD.t8.name} on one H100 at batch 1${step>=SIX('round')?`, the ${MD.d1.name} draft`:''}${step>=SIX('expect')?', α = 0.8':''}. Settings take effect from part 7 and in Scenarios.`,`这一部分用固定的例子：一张 H100 上的 ${MD.t8.name}，batch 1${step>=SIX('round')?`，draft 模型是 ${MD.d1.name}`:''}${step>=SIX('expect')?'，α = 0.8':''}。从第 7 部分起，以及在应用场景里，自由调参才起作用。`):'';
  lock.hidden=!msg; lock.textContent=msg; ctls.hidden=!!msg; rst.hidden=!!msg; const hint=$('.ctl-hint',c); if(hint) hint.hidden=!!msg;
  if(msg) return;
  $$('.seg-ctl[data-key]',c).forEach(g=>{const k=g.dataset.key; if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
  rebuildB();
  const kseg=$('.seg-ctl[data-key="k"]',c); if(kseg) kseg.closest('.ctl')&&(kseg.closest('.ctl').hidden=S.dr==='none');
  const ain=$('#spd-ctl-a-in',c); if(ain&&document.activeElement!==ain){ain.value=S.a; A.ui.setRange(ain);} const ao=ain&&ain.closest('.ctl')&&ain.closest('.ctl').querySelector('output'); if(ao) ao.textContent=aT(S.a);
  if(ain&&ain.closest('.ctl')) ain.closest('.ctl').hidden=S.dr==='none';}
function updateRail(r){const rail=$('#spd-rail',app); if(!rail) return;
  if(mode==='guided'){const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()&&!RP.on) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step])&&!RP.on;
    if(box){box.hidden=!show; if(show){const html=`<div class="eyebrow">${tx('Observed','观察到')}</div>`+s.obs(); if(box._h!==html){const again=box._h!=null; box.innerHTML=html; box._h=html; if(again) A.ui.obsBump(box);}}}
    const dis=isNextDisabled(); A.ui.setNext(rail,dis,WHY());
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
    $$('[data-lkey]',rail).forEach(b=>{const key=b.dataset.lkey, cur=key==='cost'?String(costN):String(LEV[key].cur()), on=cur===b.dataset.lever; b.setAttribute('aria-pressed',String(on)); b.classList.toggle('on',on);
      const rr=b.querySelector('.lv-r'); if(rr){const v=key==='cost'||key==='expect'||key==='batch'?+b.dataset.lever:b.dataset.lever, h=key==='cost'?(tried.cost&&tried.cost.has(v)?esc(msT(costRow(v).t)):''):levRes(key,v); if(rr.innerHTML!==h) rr.innerHTML=h;}});
    const pb=$('[data-act="rp-play"]',rail); if(pb) pb.textContent=rpBtnTxt();
  } else if(mode==='sandbox'){const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#spd-insight',rail); if(ins){const html=`<div class="eyebrow">${cur>=0?esc(L(PRESETS[cur].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r); if(ins._h!==html){ins.innerHTML=html; ins._h=html;}}
  } else {const p=chMe(), v=$('#spd-ch-v',rail); if(!v) return; v.textContent=xT(p.speed); $('#spd-ch-g',rail).classList.toggle('met',p.ok);
    $('#spd-ch-p i',rail).style.width=Math.min(100,p.speed/CH_GOAL*100)+'%'; $('#spd-ch-p',rail).classList.toggle('met',p.ok);
    $$('.seg-ctl[data-key]',rail).forEach(g=>{const k=g.dataset.key; if(S[k]!=null) A.ui.setPressed(g,String(S[k]));});
    const kc=$('#spd-ic-k',rail); const kw=kc&&(kc.closest('.inline-ctl')||kc.closest('.ctl')||kc); if(kw) kw.hidden=S.dr==='none';
    const res=$('#spd-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=!!res._h&&!!html; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}}}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
const WHY=()=>({cost:tx('Try 1 request × 5 tokens and 128 requests × 5 tokens to continue, or press Show me.','试过“1 个请求 × 5 个 token”和“128 个请求 × 5 个 token”才能继续，或点“演示给我看”。'),round:tx('Play the round to its end to continue, or press Show me.','把这一轮播完才能继续，或点“演示给我看”。'),predict:tx('Pick an answer to continue, or Skip.','选一个答案才能继续，或点“跳过”。'),expect:tx('Try 2 and 8 guesses per round to continue.','试过每轮猜 2 个和 8 个才能继续。'),same:tx('Try the acceptance rule and “no check” to continue.','试过“按接受规则”和“不检查”才能继续。'),batch:tx('Try 1 and 128 requests per step to continue.','试过一步 1 个和 128 个请求才能继续。')}[STEPS[step].id]||'');
function railHTML(){
  if(mode==='guided'){const s=STEPS[step]; return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),nextWhy:WHY(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();}
function renderRail(){const rail=$('#spd-rail',app);
  const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):'';
  rail.innerHTML=railHTML()+(mode==='challenge'?A.ui.exits(ID,EXITS()):'')+nl; A.ui.initRanges(rail); render();}
function panels(){const g=mode==='guided'?sid():'';
  return {cost:g==='mission'||g==='cost',round:g==='round'||g==='predict'||(g==='expect'),kcurve:g==='expect',acc:g==='same',bcurve:mode!=='guided'||g==='batch'||g==='remember',time:mode!=='guided'||g==='batch'};}
function render(){if(!app) return;
  const vs=viewState(), r=compute(vs), pn=panels(), mw=$('#spd-main',app).clientWidth; if(mw) W=Math.round(Math.max(300,Math.min(680,mw-26)));
  $('#spd-st-title',app).textContent=stageTitle();
  [['cost','#spd-cost'],['round','#spd-round'],['kcurve','#spd-kcurve'],['acc','#spd-acc'],['bcurve','#spd-bcurve'],['time','#spd-time']].forEach(([k,sel])=>{const el=$(sel,app); el.hidden=!pn[k];});
  if(pn.cost) drawCost(); if(pn.round) drawRound(); if(pn.kcurve) drawKCurve(); if(pn.acc) drawAcc(); if(pn.bcurve) drawBCurve(); if(pn.time) drawTime();
  drawMetrics(r);
  const st=$('#spd-status',app), g=mode==='guided'?sid():'';
  let ok=true, txt;
  if(mode==='challenge'){const p=chMe(); ok=p.ok; txt=p.ok?tx('Goal met','已达标'):tx('Not yet','未达标');}
  else if(g==='mission'||g==='cost'){txt=tx('Memory-bound','受显存带宽限制'); if(g==='cost'&&costRow(costN).tc>costRow(costN).tm){txt=tx('Compute-bound','受算力限制'); ok=false;}}
  else if(g==='round'||g==='predict'||g==='expect'){txt=RP.on?tx('Playing','播放中'):tx('Example round','示例轮次');}
  else if(g==='same'){ok=accMode!=='nocheck'; txt=ok?tx('Output unchanged','输出不变'):tx('Output changed','输出变了');}
  else {ok=r.speed>=1; txt=vs.dr==='none'?tx('No guessing','不猜'):ok?tx('Faster','更快'):tx('Slower','更慢');}
  st.className='status '+(ok?'ok':'bad'); st.textContent=txt;
  st.dataset.tipx=mode==='challenge'?tx('Whether your pick reaches 1.5 times plain decode’s tokens per second.','你的选择是否达到普通 decode 每秒 token 数的 1.5 倍。'):tx('What the chart shows right now.','图上现在的状态。');
  const mb=$('#spd-mathbody',app), mh=mathHTML(r), md=$('#spd-math',app); md.hidden=!mh; if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail(r);
  if(mode!=='sandbox'||!RP.on) A.hash.write(token());   // outside Scenarios the address drops the settings, so a reload keeps the mode (round 3 #6)
  A.dock.refresh(); announce(r);}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; A.announce(r.s.dr==='none'?tx('No guessing.','不猜。'):tx(`${fd(r.E,2)} tokens per target step on average, ${fd(r.speed,2)} times plain decode.`,`目标模型跑一次平均 ${fd(r.E,2)} 个 token，速度是普通 decode 的 ${fd(r.speed,2)} 倍。`));},400);}

/* ---------- the round player (R3.9, R3.20) ---------- */
function stopRP(){if(RP.raf) cancelAnimationFrame(RP.raf); RP.on=false; RP.raf=0;}
function playRP(from,quick){stopRP(); const nf=nFrames(rpR); RP.f=from>=nf?0:from; RP.on=true; RP.last=0; RP.acc=0; RP.quick=!!quick; RP.key=rkey(rpR); nPlays++;
  const tick=ts=>{if(!RP.on||!app){RP.raf=0; return;}
    if(RP.last){RP.acc+=Math.min(0.1,(ts-RP.last)/1000); const per=RP.quick?0.25:SPEEDS[rpSpd]; if(RP.acc>=per){RP.acc-=per; RP.f++; if(RP.f>=nf){RP.f=nf; watched.add(RP.key); RP.on=false; RP.raf=0; render(); keepStage(); return;} render();}}
    RP.last=ts; RP.raf=requestAnimationFrame(tick);};
  RP.raf=requestAnimationFrame(tick); render();}
/* show a round: play it only if this round has not been watched (R3.20); a watched round shows its end at once */
function showRound(R,src){stopRP(); rpR=R; const nf=nFrames(R);
  if(watched.has(rkey(R))||A.reduceMotion()||src==='quiet'){RP.f=nf; watched.add(rkey(R)); render(); return;}
  if(src==='auto'){RP.f=0; render(); playRP(0); return;}
  RP.f=0; playRP(0);}

/* ---------- state changes ---------- */
function setState(p,src){
  const before=JSON.stringify(S);
  if(p.dr!=null&&p.a==null){const t=p.tgt||S.tgt; p.a=p.dr==='none'?S.a:ALPHA[t][p.dr];}
  if(p.tgt!=null&&p.dr==null&&p.a==null&&S.dr!=='none') p.a=ALPHA[p.tgt][S.dr];
  Object.assign(S,p);
  if(mode==='challenge'){Object.assign(S,CH); S.a=S.dr==='none'?0.7:ALPHA[CH.tgt][S.dr];}
  if(!KS.includes(S.k)) S.k=4;
  if(before!==JSON.stringify(S)&&mode==='guided'&&sid()==='batch'&&p.B!=null) (tried.batch=tried.batch||new Set()).add(S.B);
  render();}
function enterStep(i){stopRP(); step=Math.max(0,Math.min(STEPS.length-1,i)); const s=STEPS[step];
  if(s.id==='batch'){S={...DEFAULT,B:tried.batch?S.B:1}; if(!tried.batch) tried.batch=new Set([1]);}
  if(s.id==='remember') S={...DEFAULT};
  if(s.id==='cost'&&!tried.cost) tried.cost=new Set([costN]);
  if(s.id==='same'&&!tried.same) tried.same=new Set([accMode]);
  if(s.id==='expect'&&!tried.expect) tried.expect=new Set([rpR.k]);
  const st=$('#spd-stage',app); if(st) st.scrollTop=0;
  renderRail();
  if(s.id==='round') showRound(ROUNDS[4],'auto');
  else if(s.id==='predict') showRound(ROUNDS[4],'quiet');
  else if(s.id==='expect') showRound(ROUNDS[rpR.k]||ROUNDS[4],'auto');}
function revealStep(){A.ui.reveal($('#spd-rail .step.current',app)||$('#spd-rail',app));}
/* keepStage (R9.7), as labs/llm-judge.js */
function stageStop(){if(stageT){clearTimeout(stageT); stageT=0;}}
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#spd-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#spd-rail .step.current',app)||$('#spd-rail .railcard',app)||$('#spd-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;   // the card has left the screen: the reader scrolled on, so stay put
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function setMode(m){
  if(m===mode) return; stopRP();
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...S,...CH,...CH0,a:ALPHA[CH.tgt][CH0.dr]};}
  if(m==='sandbox'&&was==='guided') S={...DEFAULT};
  A.shell.setMode(app,m,DESCS());
  const st=$('#spd-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();}
function applyPreset(i){userAct=true; try{S={...PRESETS[i].p}; rebuildB(); render();}finally{userAct=false;}}
function rebuildB(){const c=$('#spd-controls',app); if(!c) return; const g=$('.seg-ctl[data-key="B"]',c); if(!g) return; const has=$(`button[data-v="${S.B}"]`,g), extra=$$('button',g).some(b=>!BATCHES.includes(+b.dataset.v));
  if(has&&!extra) return; if(has&&extra&&!BATCHES.includes(S.B)) return;
  const host=g.closest('.ctl')||g.parentNode, opts=BATCHES.includes(S.B)?BATCHES:BATCHES.concat([S.B]).sort((a,b)=>a-b), tmp=document.createElement('div');
  tmp.innerHTML=A.ui.seg({id:'spd-ctl-B',key:'B',label:tx('Requests per step (batch)','一步里的请求数（batch）'),options:segOpts(opts,String),value:String(S.B),note:tx('This lab assumes 1,024 tokens of context per request; every setting fits in GPU memory.','本 Lab 假设每个请求有 1,024 个 token 的上下文；每种设置都放得进显存。')}); host.replaceWith(tmp.firstElementChild);}
const token=(s=S)=>[s.tgt,s.dr,s.k,s.B,Math.round(s.a*100)].join('_');
function parse(tok){const p=String(tok||'').split('_'); if(p.length!==5) return null;
  const [tgt,dr,k,B,a]=p; if(!MD[tgt]||!GPUS[tgt]||!DRAFTS.includes(dr)) return null; const kk=+k, bb=+B, aa=+a/100;
  if(!KS.includes(kk)||!(bb>=1&&bb<=128)||!(aa>=0.3&&aa<=0.95)) return null; return {tgt,dr,k:kk,B:bb,a:aa};}
function showMe(id){
  if(id==='cost'){['5','640'].forEach(v=>(tried.cost=tried.cost||new Set()).add(+v)); costN=640; renderRail(); return;}
  if(id==='round'){stopRP(); rpR=ROUNDS[4]; playRP(RP.f>=nFrames(rpR)?0:RP.f,true); return;}
  if(id==='expect'){(tried.expect=tried.expect||new Set()).add(2).add(8); rpR=ROUNDS[8]; renderRail(); showRound(ROUNDS[8],'auto'); return;}
  if(id==='same'){(tried.same=tried.same||new Set()).add('sample').add('nocheck'); accMode='nocheck'; renderRail(); return;}
  if(id==='batch'){(tried.batch=tried.batch||new Set()).add(1).add(128); S.B=128; renderRail(); return;}}
function act(a,btn){
  if(a==='rp-play'){if(RP.on){stopRP(); render(); return;} const nf=nFrames(rpR); playRP(RP.f>=nf?0:RP.f); return;}
  if(a==='rp-replay'){playRP(0); return;}
  if(a==='rp-back'){stopRP(); RP.f=Math.max(0,RP.f-1); A.tip.quiet&&A.tip.quiet(); render(); return;}
  if(a==='rp-fwd'){stopRP(); const nf=nFrames(rpR); RP.f=Math.min(nf,RP.f+1); if(RP.f>=nf) watched.add(rkey(rpR)); A.tip.quiet&&A.tip.quiet(); render(); return;}
  if(a==='reset'){userAct=true; try{S={...DEFAULT}; rebuildB(); render();}finally{userAct=false;} return;}
  if(a==='submit'){chSubmitted=true; {const m=chMe(), b=chBest(); if(m.ok&&m.dr===b.dr&&m.k===b.k) chReveal=true;} renderRail(); keepStage(); return;}
  if(a==='reveal'){chReveal=true; renderRail(); keepStage(); return;}
  if(a==='copy'){const t=CODE; (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).catch(()=>{}); flashBtn(btn,tx('Copied','已复制'),A.t('copyCode')); return;}
  if(a==='drill-new'){drill=makeDrill(((drill&&drill.seed)||0)+1); drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* interview drill: a new case from the engine each time */
function makeDrill(seed){const rnd=mulberry(500+seed), B=[1,8,16][Math.floor(rnd()*3)], a=[0.6,0.7,0.8][Math.floor(rnd()*3)], k=[3,4,5][Math.floor(rnd()*3)], s={tgt:'t8',dr:'d1',k,B,a}, r=compute(s); return {seed,s,r};}
function renderDrill(){if(!app) return; if(!drill) drill=makeDrill(0); const {s,r}=drill, q=$('#spd-drill-q',app), an=$('#spd-drill-a',app);
  q.innerHTML=tx(`${MD.t8.name} serves ${s.B} request${s.B>1?'s':''} per step on one H100; a plain decode step takes ${msT(r.tPlain)}. A ${MD.d1.name} draft guesses ${s.k} tokens per round with α = ${aT(s.a)}; its ${s.k} steps take ${msT(r.td)} together and the 8B’s check step ${msT(r.v.t)}. How many tokens per target step, and how many times plain decode’s speed?`,
    `${MD.t8.name} 在一张 H100 上每步服务 ${s.B} 个请求，普通 decode 一步要 ${msT(r.tPlain)}。${MD.d1.name} 作 draft 模型，每轮猜 ${s.k} 个，α = ${aT(s.a)}；它走 ${s.k} 步一共 ${msT(r.td)}，8B 检查的那一步 ${msT(r.v.t)}。目标模型跑一次平均得到几个 token？速度是普通 decode 的几倍？`);
  an.hidden=!drillOpen;
  an.innerHTML=drillOpen?tx(`<p><b>${fd(r.E,2)} tokens per target step on average, ${xT(r.speed)} plain decode.</b></p>`,`<p><b>跑一次平均 ${fd(r.E,2)} 个 token，速度是普通 decode 的 ${xT(r.speed)}。</b></p>`)+
    EQ('E',['\\frac{1 - \\alpha^{k+1}}{1 - \\alpha}',`\\frac{1 - ${aT(s.a)}^{${s.k+1}}}{1 - ${aT(s.a)}}`,fd(r.E,4)])+
    EQ(tx('speedup','加速倍数'),['\\frac{E \\times t_{plain}}{t_{draft} + t_{check}}',`\\frac{${fd(r.E,4)} \\times ${tms(r.tPlain)}}{${tms(r.td)} + ${tms(r.v.t)}}`,fd(r.speed,2)])+
    `<p class="small">${tx('Times in ms. The check step processes one token more than the guesses per request; while the step is memory-bound it costs about one plain step.','时间单位是 ms。检查的那一步每个请求处理的 token 比猜的多 1 个；只要这一步受显存带宽限制，它就和普通的一步差不多贵。')}</p>`:'';}
/* capability layer */
function EXITS(){const e9=expTok(0.9,4), b1=compute({tgt:'t8',dr:'d1',k:4,B:1,a:0.7}), b70=compute({tgt:'t70',dr:'d1',k:4,B:1,a:0.65});
  return {core:()=>chSubmitted,
   explain:[
    {q:tx('Why does checking 4 guesses in one target step cost about as much as making one token?','为什么目标模型跑一次检查 4 个猜测，和生成 1 个 token 差不多贵？'),
     o:[tx('At a small batch a decode step’s time goes to reading all the weights from memory; the arithmetic for 5 tokens still finishes within that reading time, so the step takes about as long.','batch 小的时候，decode 一步的时间花在从显存读全部权重上；5 个 token 的计算在读权重的这段时间里就能做完，所以这一步差不多一样长。'),
        tx('The target model only reads the layers it needs for the guesses, so checking skips most of the weights.','目标模型检查时只读需要的那几层，所以跳过了大部分权重。'),
        tx('The 4 guesses are checked on 4 other GPUs at the same time, so they take no extra time.','4 个猜测在另外 4 张卡上同时检查，所以不多花时间。')],a:0,
     n:[null,tx('Every layer still runs for every token; the saving is that the weights are read once for all 5 tokens.','每个 token 仍要走完每一层；省下的是：5 个 token 共用一次权重读取。'),tx('One GPU does the whole check in one step; no extra GPUs are involved.','一张卡在一步里就检查完了，不需要别的卡。')],
     why:tx('Decode at a small batch is memory-bound: reading the weights takes far longer than the arithmetic, so more tokens per step are nearly free until the arithmetic catches up.','batch 小时 decode 受显存带宽限制：读权重比计算慢得多，所以一步多算几个 token 几乎不花时间，直到计算追上读显存。')},
    {q:tx('Why is the output the same as the target model writing alone?','为什么输出和目标模型自己写的一样？'),
     o:[tx(`A guess x is kept with probability ${M('\\min(1, p(x)/q(x))')} and otherwise a token is drawn from the leftover ${M('\\max(0, p - q)')}; the two parts add up to exactly the target’s probabilities p.`,`猜测 x 以 ${M('\\min(1, p(x)/q(x))')} 的概率收下，否则从剩下的 ${M('\\max(0, p - q)')} 里抽一个 token；两部分加起来正好是目标模型的概率 p。`),
        tx('The draft model is trained to copy the target model, so its guesses are almost always the same tokens.','draft 模型是照着目标模型训练的，所以它猜的几乎总是同样的 token。'),
        tx('The target model rewrites the whole answer at the end and replaces anything the draft model got wrong.','目标模型最后把整个回答重写一遍，把 draft 模型写错的都换掉。')],a:0,
     n:[null,tx('A poor draft model does not change the output at all, it only lowers α; equality comes from the rule, not from the draft’s quality.','draft 模型猜得差，输出也不会变，只是 α 变低；输出相同靠的是规则，而不是 draft 模型的质量。'),tx('There is no rewrite at the end: each round’s tokens are final once checked.','最后没有重写：每一轮的 token 检查完就定了。')],
     why:tx(`The rule accepts each token with ${M('\\min(p, q)')} and redraws the rest from ${M('\\max(0, p - q)')}, so every position follows p exactly, whatever q is.`,`规则按 ${M('\\min(p, q)')} 接受、把剩下的从 ${M('\\max(0, p - q)')} 里重新抽，所以不管 q 是什么，每个位置都正好服从 p。`)}],
   transfer:[
    {q:tx('A better draft model reaches α = 0.9 with 4 guesses per round. About how many tokens does one target step yield? For reference, α = 0.8 gives 3.36.','一个更好的 draft 模型 α 能到 0.9，每轮猜 4 个。目标模型跑一次大约得到几个 token？参照：α = 0.8 时是 3.36 个。'),
     o:[tx(`About ${fd(e9,1)}`,`约 ${fd(e9,1)} 个`),tx('About 3.6','约 3.6 个'),tx('About 4.6','约 4.6 个')],a:0,
     why:tx(`1 + 0.9 + 0.81 + 0.73 + 0.66 ≈ ${fd(e9,2)}: each later guess counts with a higher power of α, so raising α lifts the later guesses most.`,`1 + 0.9 + 0.81 + 0.73 + 0.66 ≈ ${fd(e9,2)}：越靠后的猜测，α 的次方越高，所以 α 提高时，靠后的猜测收益最大。`)},
    {q:tx(`The same ${MD.d1.name} draft now serves ${MD.t70.name} instead of the 8B, one user, k = 4, with a slightly lower α of 0.65. Does the speedup over plain decode go up or down?`,`同一个 ${MD.d1.name} draft 模型，现在给 ${MD.t70.name} 而不是 8B 当 draft，只有一个用户，每轮猜 4 个，α 略低，是 0.65。比普通 decode 的加速是变大还是变小？`),
     o:[tx('Up: the 1B’s steps are now a much smaller share of a target step','变大：1B 走的那几步只占目标模型一步的很小一部分了'),tx('Down: α is lower, so fewer tokens per target step','变小：α 低了，跑一次得到的 token 少了'),tx('About the same: the draft model did not change','差不多：draft 模型没变')],a:0,
     why:tx(`In this lab it goes from ${xT(b1.speed)} to ${xT(b70.speed)}. The lower α costs a little, but the 1B’s 4 steps fall from ${pc(b1.td/b1.tPlain)} of an 8B step to ${pc(b70.td/b70.tPlain)} of a 70B step.`,`在这个 Lab 里，从 ${xT(b1.speed)} 变成 ${xT(b70.speed)}。α 低了一点，损失不大；而 1B 走 4 步，从占 8B 一步的 ${pc(b1.td/b1.tPlain)} 降到只占 70B 一步的 ${pc(b70.td/b70.tPlain)}。`)}],
   practice:[{id:'llm-fundamentals',d:tx('The speculative decoding item: explain why one target step can check several draft tokens and where generation resumes after a rejection.','投机解码那一条：讲清目标模型为什么跑一次就能检查好几个 draft token，以及被拒绝后从哪里接着生成。')},
     {id:'inference-serving',d:tx('Design a serving system with latency targets; decide where speculative decoding helps the per-token latency and where the batch is too large for it.','设计一个有延迟目标的推理服务；判断投机解码在哪里能降低每个 token 的延迟，在哪里 batch 太大用不上。')}]};}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  const C={k:tx('Guesses per round k','每轮猜的 token 数 k'),a:tx('Acceptance rate α','接受率 α'),B:tx('Requests per step (batch)','一步里的请求数（batch）')};
  if(!C[k]) return '';
  if(mode==='challenge'&&k!=='k') return tx(`Fixed during the challenge: “${C[k]}”.`,`挑战中固定不变：${C[k]}。`);
  if(mode==='guided'&&step<SIX('batch')) return tx(`Fixed in this part: “${C[k]}”.`,`这一部分固定：${C[k]}。`);
  return tx(`Click to jump to “${C[k]}”.`,`点击跳到“${C[k]}”。`);}
function jump(k,src){if(!app) return;
  if(k==='controls'){const c=$('#spd-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  const locked=(mode==='challenge'&&k!=='k')||(mode==='guided'&&step<SIX('batch'));
  if(locked){if(src) A.tip.show(src,`<div>${jumpAct(k)}</div>`); return;}
  const el=mode==='challenge'?$('#spd-ic-'+k,app):$('#spd-ctl-'+k,app)||$('#spd-ctl-'+k+'-in',app); if(el) A.ui.jumpToEl(el);}
const TERM={k:()=>D_K(),a:()=>D_ALPHA(),B:()=>tx('Requests per step: how many requests the server runs together in each step.','一步里的请求数：服务端每一步同时跑多少个请求。')};

/* ---------- mount ---------- */
function bind(el){
  el.addEventListener('input',e=>{const t=e.target; if(t.id==='spd-rp-pos'){stopRP(); RP.f=+t.value; if(RP.f>=nFrames(rpR)) watched.add(rkey(rpR)); render(); return;}
    const k=t.dataset&&t.dataset.key; if(k==='a'){userAct=true; try{S.a=Math.round(+t.value*100)/100; render();}finally{userAct=false;}}});
  el.addEventListener('toggle',e=>{const d=e.target; if(d&&d.matches&&d.matches('details[data-more]')) moreOpen[d.dataset.more]=d.open;},true);
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset, inRail=!!b.closest('#spd-rail');
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){const k=seg.dataset.key;
      if(k==='rpSpd'){rpSpd=d.v; A.ui.setPressed(seg,d.v); return;}
      userAct=true; try{setState({[k]:k==='tgt'||k==='dr'?d.v:+d.v});}finally{userAct=false;} if(inRail) keepStage(); return;}
    if(d.lever!=null&&d.lkey){const key=d.lkey, v=d.lever; userAct=true;
      try{if(key==='cost'){costN=+v; (tried.cost=tried.cost||new Set()).add(costN); render();}
        else if(key==='expect'){const k=+v; (tried.expect=tried.expect||new Set()).add(k); render(); showRound(ROUNDS[k],'lever');}
        else if(key==='same'){accMode=v; (tried.same=tried.same||new Set()).add(v); render();}
        else if(key==='batch'){setState({B:+v});}}finally{userAct=false;}
      renderRail(); keepStage(); return;}
    // a part's title, Next and Back keep the stage's head below the top bar too (R9.7, the owner's B4; as admission and pd-disagg; polish
    // round 3, 2026-10-16): on the last parts little of the rail is left under the card, and the bench's end pushed the stage under the top bar
    if(d.go!=null){enterStep(+d.go); revealStep(); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe) showMe(s.showMe); keepStage(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#spd-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); keepStage(); return;}
    if(d.preset!=null){applyPreset(+d.preset); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#spd-rail',app)); return;}
    if(d.act){act(d.act,b); if(inRail&&!b.closest('[data-exits]')&&d.act!=='copy') keepStage();}
  });}
const sOf=s=>({tgt:s.tgt,dr:s.dr,k:s.k,B:s.B,a:s.a});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>{const tr={}; for(const k in tried) if(tried[k] instanceof Set) tr[k]=[...tried[k]];
    return {v:1,mode,step,S:sOf(S),pred,observed:{...observed},tried:tr,costN,accMode,rpK:rpR.k,rpSpd,watched:[...watched],saved:saved?sOf(saved):null,
      chSubmitted,chReveal,drillSeed:drill?drill.seed:0,drillOpen,passed:mode==='challenge'&&chSubmitted&&chMe().ok};},
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parse(token(o.S)):null; if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided'; step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S=p; if(mode==='challenge'){Object.assign(S,CH); S.a=S.dr==='none'?0.7:ALPHA[CH.tgt][S.dr];}
    pred=PRED.includes(o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const OK={batch:x=>Number.isInteger(x)&&x>=1&&x<=128,cost:x=>COSTN.includes(x),same:x=>['greedy','sample','nocheck'].includes(x),expect:x=>!!ROUNDS[x]};
    tried={}; if(o.tried&&typeof o.tried==='object') for(const k in OK) if(Array.isArray(o.tried[k])) tried[k]=new Set(o.tried[k].filter(OK[k]));
    costN=COSTN.includes(o.costN)?o.costN:1; accMode=OK.same(o.accMode)?o.accMode:'sample'; rpR=ROUNDS[o.rpK]||ROUNDS[4]; if(SPEEDS[o.rpSpd]) rpSpd=o.rpSpd;
    watched.clear(); if(Array.isArray(o.watched)) o.watched.forEach(k=>{if(Object.values(ROUNDS).some(R=>rkey(R)===k)) watched.add(k);});
    const sv=o.saved&&typeof o.saved==='object'?parse(token(o.saved)):null; saved=mode==='challenge'&&sv?sv:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    drill=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?makeDrill(o.drillSeed):null; drillOpen=o.drillOpen===true;
    // the round on the stage, as enterStep() leaves it: whole once watched (always in part 2), otherwise at its start with Play
    stopRP(); const sid=mode==='guided'?STEPS[step].id:''; if(sid==='round'||sid==='predict') rpR=ROUNDS[4];
    if(sid==='predict') watched.add(rkey(rpR)); RP.f=watched.has(rkey(rpR))?nFrames(rpR):0;},
  mount(el,tok){stopRP(); root=el;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#spd-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#spd-rail',app)); keepStage();},token});
    bind(app); renderRail(); renderDrill(); watchStage();},
  leave(){stopRP(); stageStop(); clearTimeout(annT); if(stRO){stRO.disconnect(); stRO=null;} root=null; app=null;},
  onLang(){if(root){const d=$('#spd-math',app), open=d&&d.open; if(RP.on){stopRP(); RP.f=nFrames(rpR); watched.add(rkey(rpR));} this.mount(root,null); if(open) $('#spd-math',app).open=true;}},
  onResize(){if(app) render();},
  termTip(k){const tm=TERM[k]; if(!tm) return ''; return `<div>${tm()}</div><div class="act">${jumpAct(k)}</div>`;},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#spd-main',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#spd-main',app),
    render({status,vis,read}){if(!app) return; const r=compute(viewState()), g=mode==='guided'?sid():'';
      vis.innerHTML=`<div class="spd-dk"><i style="width:${Math.min(100,r.speed/3*100).toFixed(1)}%"></i></div>`;
      if(mode==='challenge'){const p=chMe(); status.className='status '+(p.ok?'ok':'bad'); status.textContent=p.ok?tx('Goal met','已达标'):tx('Not yet','未达标');
        read.textContent=tx(`70B, 128 requests per step, ${chName(p)}: ${xT(p.speed)} plain decode`,`70B 一步 128 个请求，${chName(p)}：普通 decode 的 ${xT(p.speed)}`); return;}
      if(g==='mission'||g==='cost'){const c=costRow(g==='cost'?costN:1); status.className='status '+(c.tc>c.tm?'bad':'ok'); status.textContent=c.tc>c.tm?tx('Compute-bound','受算力限制'):tx('Memory-bound','受显存带宽限制');
        read.textContent=tx(`One 8B step with ${g==='cost'?costN:1} token${(g==='cost'?costN:1)>1?'s':''}: ${msT(c.t)}`,`8B 一步处理 ${g==='cost'?costN:1} 个 token：${msT(c.t)}`); return;}
      if(g==='round'||g==='predict'||(g==='expect'&&RP.on)){status.className='status ok'; status.textContent=RP.on?tx('Playing','播放中'):tx('Example','示例');
        read.textContent=roundTitle(rpR,RP.f); return;}
      if(g==='expect'){status.className='status ok'; status.textContent='k = '+rpR.k; read.textContent=tx(`α = 0.8, k = ${rpR.k}: ${fd(expTok(0.8,rpR.k),2)} tokens per target step on average`,`α = 0.8、每轮猜 ${rpR.k} 个：目标模型跑一次平均 ${fd(expTok(0.8,rpR.k),2)} 个 token`); return;}
      if(g==='same'){const ok=accMode!=='nocheck'; status.className='status '+(ok?'ok':'bad'); status.textContent=ok?tx('Unchanged','不变'):tx('Changed','变了'); read.textContent=obsSameShort(); return;}
      status.className='status '+(r.speed>=1?'ok':'bad'); status.textContent=r.s.dr==='none'?tx('No guessing','不猜'):r.speed>=1?tx('Faster','更快'):tx('Slower','更慢');
      read.textContent=r.s.dr==='none'?tx(`Plain decode: ${fd(r.tpsPlain,1)} tokens per second per request`,`普通 decode：每个请求每秒 ${fd(r.tpsPlain,1)} 个 token`):tx(`Batch ${r.s.B}, ${dName(r.s.dr)}, k = ${r.s.k}: ${fd(r.speed,2)} times plain decode`,`batch ${r.s.B}，${dName(r.s.dr)}，每轮猜 ${r.s.k} 个：普通 decode 的 ${fd(r.speed,2)} 倍`);}
  }
});
const obsSameShort=()=>{const fr=accFreq(accMode), R=accRule(); return accMode==='greedy'?tx(`Greedy: always “${wd(0)}”`,`贪心：总是“${wd(0)}”`):tx(`“${wd(0)}” comes out ${pc(fr[0],1)}; the target gives it ${pc(R.p[0])}`,`“${wd(0)}”出现 ${pc(fr[0],1)}，目标模型给它 ${pc(R.p[0])}`);};

/* =====================================================================
   知识梳理 (the outline the owner approved on 2026-10-14: decode is slow → known tokens take one forward pass → guess, then check →
   why the output stays the same and what the speedup depends on): four sections read before the guided parts. Section 1 only points
   to the prefill-decode primer, which owns memory-bound decode. The two pictures are schematics in the text's own symbols (x₁…x₅ and
   their distributions; the guesses d₁…d₄ against the target's ŷ₁…ŷ₅), with no word, model or number of the guided parts; how strong
   each effect is (step times, tokens per step, the speedup, the batch) stays with the guided parts (A.registerPrimer, CONTRACT §17)
   ===================================================================== */
const prM=t=>A.ui.math(t);
const prLK=(id,en,zh)=>{const a=labRef(id,en,zh); return tx(a,`“${a}”`);};
const prP=(...ps)=>({html:()=>ps.map(p=>{const h=p(); return h.startsWith('<div')?h:`<p>${h}</p>`;}).join('')});
const prFx=f=>()=>`<div class="spd-prfx">${f().map(l=>`<span>${prM(l)}</span>`).join('')}</div>`;   // a formula on its own line, in pieces that wrap before “=” or “+” (made at render time, in the language shown)
const prPic=(html,cap)=>({fig:[{html,cap}],cls:'spd-prf'});   // a still picture: an HTML grid that sizes to the figure, so nothing is redrawn
// section 2: one forward pass over x₁…x₅. Row i is position i: it sees x₁…xᵢ (solid cells; the dashed ones come after it) and gives p(· | x₁, …, xᵢ)
const PR_SEEN=['x_1','x_1, x_2','x_1, x_2, x_3','x_1, \\ldots, x_4','x_1, \\ldots, x_5'];
const prPassHTML=()=>`<div class="spd-prg"><span class="spd-prg-h">${prM('i')}</span><span class="spd-prg-h spd-prg-hc">${tx('the tokens it sees','看得到的 token')}</span><span></span><span class="spd-prg-h">${tx('output','输出')}</span>`+
  [1,2,3,4,5].map(i=>`<span class="spd-prg-i">${i}</span>`+[1,2,3,4,5].map(j=>j<=i?`<span class="spd-prg-c spd-seen">${prM('x_'+j)}</span>`:'<span class="spd-prg-c"></span>').join('')+
    `<span class="spd-prg-a">→</span><span class="spd-prg-o">${prM(`p(\\cdot \\mid ${PR_SEEN[i-1]})`)}</span>`).join('')+'</div>';
// section 3: one round with k = 4. The guesses d₁…d₄ against the target's ŷ₁…ŷ₅: kept, kept, rejected (ŷ₃ goes in), void after it;
// the chips take the lab's own chip colours (spd-ok, spd-bad, spd-tg, spd-own, spd-drop), the marks under them are words
const prRoundHTML=()=>{const c=(k,t)=>`<span class="spd-prc spd-${k}">${t?prM(t):''}</span>`, l=(en,zh)=>`<span class="spd-prr-l">${tx(en,zh)}</span>`,
    m=(k,en,zh)=>`<span class="spd-prm spd-pm-${k}">${tx(en,zh)}</span>`, y=i=>`\\hat{y}_${i}`;
  return `<div class="spd-prr">${l('position','位置')}${[1,2,3,4,5].map(i=>`<span class="spd-prr-n">${i}</span>`).join('')}`+
    l('guesses','猜测')+c('ok','d_1')+c('ok','d_2')+c('bad','d_3')+c('drop','d_4')+'<span></span>'+
    l('target','目标模型')+c('tg',y(1))+c('tg',y(2))+c('own',y(3))+c('drop',y(4))+c('drop',y(5))+
    l('check','对照')+m('ok','keep','收下')+m('ok','keep','收下')+m('bad','reject','拒绝')+m('void','void','作废')+'<span></span>'+
    l('kept','留下')+c('ok','d_1')+c('ok','d_2')+c('own',y(3))+'</div>';};
if(A.registerPrimer) A.registerPrimer(ID,{min:7,sections:[
 {h:()=>tx('Why decode is slow: it is memory-bound','decode 为什么慢：受显存带宽限制'),parts:[prP(
   ()=>tx(`A large model first reads the whole prompt at once (prefill), then writes its answer one token per step (decode), and every decode step reads all the weights from GPU memory into the compute units. When only a few requests are being written at the same time (that one read of the weights serves only them), reading the weights takes much longer than the arithmetic, so most of the GPU’s compute sits idle: decode is memory-bound (derived in the basics of ${prLK('prefill-decode','Prefill and decode','Prefill 和 decode')}). To write an answer faster, the large model has to take fewer steps.`,
          `大模型先一次读完 prompt（prefill），再一步一个 token 地写回答（decode），而 decode 的每一步都要把全部权重从显存读进计算单元一遍。同时在写的请求不多时（这一遍权重只给这几个请求用），读权重比做计算慢得多，GPU 的算力大半闲着，这叫受显存带宽限制（推导见${prLK('prefill-decode','Prefill and decode','Prefill 和 decode')}的知识梳理）。所以，想让回答写得更快，就得让大模型少走几步。`))]},
 {h:()=>tx('Tokens already known: one forward pass computes them all','已知的 token，一次前向算完'),parts:[prP(
   ()=>tx(`Why must generation go one token per step? Token ${prM('i+1')} is drawn from the distribution ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')}, and until it has been drawn it cannot be fed in, so the next step has to wait for this one.`,
          `生成为什么只能一步一个？因为第 ${prM('i+1')} 个 token 要从分布 ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')} 里抽出来，抽定之前，它还不能当作输入，下一步只能等这一步算完。`),
   ()=>tx(`If ${prM('x_1, \\ldots, x_n')} are all known already, things change: feed them through the model together in one forward pass, and position ${prM('i')} outputs ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')}, so all ${prM('n')} distributions come out at once (see the picture). Position ${prM('i')} uses only ${prM('x_1')} to ${prM('x_i')} (attention’s causal mask hides the tokens after it), so feeding the later tokens in alongside changes nothing at any earlier position. Training uses this to learn the next token at every position of a sentence in one forward pass, and prefill uses it to take a whole prompt in one step.`,
          `可要是 ${prM('x_1, \\ldots, x_n')} 全都已经知道，情况就不同了：把它们一起送进模型跑一次前向（forward pass），位置 ${prM('i')} 输出 ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')}，${prM('n')} 个分布一次全部算出（见下图）。位置 ${prM('i')} 只用到 ${prM('x_1')} 到 ${prM('x_i')}（注意力的因果掩码挡住了后面的 token），所以后面的 token 一起送进来，也不会改变前面任何位置的结果。训练时一句话跑一次前向，每个位置都同时在学着预测下一个 token；prefill 一步处理完整个 prompt，也是这个道理。`),
   ()=>tx('That opens a door: if someone proposes the next few tokens, append them to the input and run the large model forward once, and it gives its own distribution at every proposed position, ready to compare with the proposals one by one. The pass computes a few more positions, but it is still the same one read of the weights.',
          '这就给了一个机会：只要有人先提议后面几个 token，把它们接在输入后面，大模型跑一次前向，就能在每个提议的位置给出自己的分布，拿来和提议逐个对照。这一次前向多算了几个位置，读的仍是同一遍权重。')),
   prPic(prPassHTML,()=>tx(`One forward pass computes all 5 positions at once. Row ${prM('i')} is position ${prM('i')}: it uses only ${prM('x_1')} to ${prM('x_i')} (the dashed cells are the tokens after it, which it cannot see) and gives the distribution of the next token, ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')}.`,
     `一次前向，5 个位置同时算：第 ${prM('i')} 行是位置 ${prM('i')}，只用到 ${prM('x_1')} 到 ${prM('x_i')}（虚线格是排在它后面、它看不到的 token），给出下一个 token 的分布 ${prM('p(\\cdot \\mid x_1, \\ldots, x_i)')}。`))]},
 {h:()=>tx('Speculative decoding: guess first, then check','投机解码：先猜，再检查'),parts:[prP(
   ()=>tx(`Where do the proposals come from? Take a much smaller model, the draft model, and let it write ${prM('k')} tokens ${prM('d_1, \\ldots, d_k')} ahead the plain-decode way. It also goes one token per step, but each step reads only its own, much smaller weights, so it is fast. The output that counts is the large model’s, which is therefore called the target model. Feed the last context token and the ${prM('k')} guesses into the target model together and run it for one step: you get the token it would write at each of the ${prM('k+1')} positions, ${prM('\\hat{y}_1, \\ldots, \\hat{y}_{k+1}')} (greedy decoding for now: at each position, the most likely token), where ${prM('\\hat{y}_i')} is computed from the context and ${prM('d_1, \\ldots, d_{i-1}')}.`,
          `候选从哪来？找一个小得多的模型，叫 draft 模型，让它照普通 decode 的办法往后写 ${prM('k')} 个 token ${prM('d_1, \\ldots, d_k')}。它也是一步一个，但每一步只读自己那份小得多的权重，所以快。输出以大模型为准，大模型因此叫目标模型。把上文最后一个 token 和 ${prM('k')} 个猜测一起送进目标模型跑一步，得到 ${prM('k+1')} 个位置上它会写的 token ${prM('\\hat{y}_1, \\ldots, \\hat{y}_{k+1}')}（先按贪心解码说：每个位置取概率最高的），其中 ${prM('\\hat{y}_i')} 是看着上文和 ${prM('d_1, \\ldots, d_{i-1}')} 算出来的。`),
   ()=>tx(`Then compare from the left: if ${prM('d_1 = \\hat{y}_1')}, keep ${prM('d_1')} and compare ${prM('d_2')} with ${prM('\\hat{y}_2')}, and so on. At the first ${prM('d_i \\ne \\hat{y}_i')}, keep the target’s own ${prM('\\hat{y}_i')} and drop ${prM('d_i')} and every guess after it; the positions after it were computed from a wrong guess, so they are void too. If every guess matches, keep the last ${prM('\\hat{y}_{k+1}')} as well.`,
          `接着从左往右对照：${prM('d_1 = \\hat{y}_1')} 就收下 ${prM('d_1')}，再比 ${prM('d_2')} 和 ${prM('\\hat{y}_2')}，依此类推。遇到第一个 ${prM('d_i \\ne \\hat{y}_i')}，收下目标模型自己的 ${prM('\\hat{y}_i')}，丢掉 ${prM('d_i')} 和它后面的猜测；后面那些位置是接着错的猜测算的，也一并作废。猜测全对时，再收下最后的 ${prM('\\hat{y}_{k+1}')}。`),
   ()=>tx('So in one round the target model takes a single step yet may write several tokens, and never fewer than one, its own: no fewer than plain decode. A small model guesses, one step of the large one checks: that is speculative decoding.',
          '这样一轮里目标模型只走一步，却可能写出好几个 token，最少也有它自己的 1 个，不比普通 decode 少。小模型先猜、大模型一步检查，这就是投机解码（speculative decoding）。')),
   prPic(prRoundHTML,()=>tx(`One round, sketched (${prM('k = 4')}): ${prM('d_1')}, ${prM('d_2')} match ${prM('\\hat{y}_1')}, ${prM('\\hat{y}_2')} and are kept; ${prM('d_3 \\ne \\hat{y}_3')}, so it is rejected and the target’s own ${prM('\\hat{y}_3')} takes its place; ${prM('d_4')} comes after a wrong guess, and ${prM('\\hat{y}_4')}, ${prM('\\hat{y}_5')} were computed from wrong guesses, so they are all void.`,
     `一轮的示意（${prM('k = 4')}）：${prM('d_1')}、${prM('d_2')} 和 ${prM('\\hat{y}_1')}、${prM('\\hat{y}_2')} 一致，收下；${prM('d_3 \\ne \\hat{y}_3')}，拒绝，换上目标模型自己的 ${prM('\\hat{y}_3')}；${prM('d_4')} 排在错的猜测后面，${prM('\\hat{y}_4')}、${prM('\\hat{y}_5')} 是接着错的猜测算的，都作废。`))]},
 {h:()=>tx('Why the output does not change, and what sets the speedup','输出为什么不变，能快多少看什么'),parts:[prP(
   ()=>tx('With greedy decoding, every kept guess equals the token the target would write there, and every replacement is its own, so the output is identical to what the target model writes alone.',
          '贪心解码时，收下的猜测都等于目标模型在那里会写的 token，换上的本来就是它自己的，所以输出和目标模型单独写时一字不差。'),
   ()=>tx(`With sampling, the target’s output at each position is random, so the requirement becomes that the output follow the target’s distribution ${prM('p')}. The draft model draws a guess from its distribution ${prM('q')}. If the guess is ${prM('x')}, it is kept with probability ${prM('\\min(1, p(x)/q(x))')}; if it is rejected, a new token is drawn in proportion to ${prM('\\max(0, p - q)')}. So the output can be ${prM('x')} in two ways: the guess is ${prM('x')} and is kept, or the guess is rejected and the redraw gives ${prM('x')}. Adding the two:`,
          `按概率采样时，目标模型每个位置的输出本身是随机的，要求就变成输出服从目标模型的分布 ${prM('p')}。draft 模型按它的分布 ${prM('q')} 抽出一个猜测；猜到 ${prM('x')} 时，以概率 ${prM('\\min(1, p(x)/q(x))')} 收下；被拒绝时，按 ${prM('\\max(0, p - q)')} 的比例重抽一个。输出 ${prM('x')} 有两条路：猜到 ${prM('x')} 并收下，或猜测被拒后重抽到 ${prM('x')}。两条路的概率相加：`),
   prFx(()=>[`P(\\text{${tx('output','输出')}}\\ x)`,'= q(x) \\min(1, p(x)/q(x))','+ r \\cdot \\frac{\\max(0, p(x) - q(x))}{r}','= \\min(p(x), q(x)) + \\max(0, p(x) - q(x))','= p(x)']),
   ()=>tx(`Here ${prM('r = 1 - \\sum_y \\min(p(y), q(y))')} is the chance that a guess is rejected, and it is also the normalizer of the redraw, ${prM('\\sum_y \\max(0, p(y) - q(y))')}, so it cancels.`,
          `其中 ${prM('r = 1 - \\sum_y \\min(p(y), q(y))')} 是一次猜测被拒绝的概率，它正好也是重抽时的归一化常数 ${prM('\\sum_y \\max(0, p(y) - q(y))')}，所以消掉了。`),
   ()=>tx('At every position the output follows exactly the target’s distribution: every token kept has the same probabilities as when the target model writes alone. A poor guesser only gets rejected more often; the distribution of the output stays the same.',
          '每个位置的输出都正好服从目标模型的分布：收下的每个 token，概率都和目标模型自己写时一样。猜得差，只会多被拒绝几次，输出的分布不变。'),
   ()=>tx(`How much faster it gets depends on two things. One is how often the guesses are right, measured by the acceptance rate ${prM('\\alpha = P(d_i\\ \\text{kept} \\mid d_1, \\ldots, d_{i-1}\\ \\text{all kept})')}; the higher it is, the more tokens a round keeps on average. The other is the cost of guessing: the draft model’s ${prM('k')} steps take time, and the target’s check computes a few more positions. More guesses may keep more tokens but always cost more; how that trade works out in different settings is what the guided parts let you try.`,
          `能快多少，取决于两件事。一是猜对的比例，用接受率 ${prM('\\alpha')} 衡量：${prM('\\alpha = P(d_i\\ \\text{被收下} \\mid d_1, \\ldots, d_{i-1}\\ \\text{都被收下})')}，它越高，一轮平均留下的 token 越多。二是猜的成本：draft 模型那 ${prM('k')} 步要花时间，目标模型检查时也多算了几个位置。多猜几个，可能多留下几个，也一定多花一些；这笔账在不同设置下怎么算，就是分步引导里要动手试的。`))]}
]});

/* read-only hook for the golden-number tests */
window.__spd=Object.freeze({compute:s=>{const r=compute({...DEFAULT,...s}); return {E:r.E,tPlain:r.tPlain,td:r.td,td1:r.td1,tv:r.v.t,tRound:r.tRound,tps:r.tps,tpsPlain:r.tpsPlain,speed:r.speed};},
  step:(id,B,n,ctx,g)=>stepT(MD[id],B,n,ctx==null?CTX:ctx,g||1),models:()=>Object.fromEntries(Object.entries(MD).map(([k,m])=>[k,{P:m.P,W:m.W,KV:m.KV,PL:m.PL,PH:m.PH}])),
  expTok,rounds,simMean,timeline:s=>{const t=timeline(compute({...DEFAULT,...s})); return {plain:t.plain,spec:t.spec,made:t.made,acc:t.rounds.map(x=>x.acc)};},tlNote:s=>{const s2={...DEFAULT,...s}; return tlNote(timeline(compute(s2)),s2);},xT:v=>xT(v),bCB:()=>bCB(),roundCost:k=>{const c=roundCost(ROUNDS[k]); return {ds:c.ds,td:c.td,tv:c.tv,tRound:c.tRound,n:c.n,tPlain:c.tPlain,plain:c.plain,flRound:c.flRound,flDraft:c.flDraft,flCheck:c.flCheck,flPlain:c.flPlain,dW:c.dW};},accRule,accFreq,memFit:s=>memFit({...DEFAULT,...s}),
  chAll:()=>chAll().map(p=>({dr:p.dr,k:p.k,speed:p.speed,ok:p.ok})),chBest:()=>{const b=chBest(); return {dr:b.dr,k:b.k,speed:b.speed};},drill:sd=>{const d=makeDrill(sd); return {s:d.s,E:d.r.E,speed:d.r.speed,tPlain:d.r.tPlain,td:d.r.td,tv:d.r.v.t};},
  consts:Object.freeze({FS,BS,CTX,OUT,PHI,SIM_N,SIM_A,ACC_N,ACC_SEED,ALPHA,DOC_A,BATCHES,CURVE_B,KS,GPUS,CH,CH0,CH_GOAL,ROUNDS,STEP_IDS:STEPS.map(s=>s.id),PRESETS:PRESETS.map(p=>p.p)}),
  state:()=>({...S,mode,step,pred,costN,accMode,rpK:rpR.k,rpAcc:rpR.acc,rpF:RP.f,playing:RP.on,plays:nPlays,rpSpd,watched:[...watched],chSubmitted,chReveal})});
})();
