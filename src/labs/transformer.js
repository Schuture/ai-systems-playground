/* Transformer structure path, lab 2 of 6 (after Architecture overview, #architecture): where the parameters of a decoder-only
   transformer live, counted exactly from config.json.
   Engine: compute(cfg) is a pure function of the seven config numbers (+ bytes per weight); every number on the page comes from it.
   Shapes follow Meta's reference code (models/llama3/model.py in meta-llama/llama-models) and transformers' modeling_llama.py:
   no biases, head_dim = hidden size ÷ heads, Wq Wk Wv Wo, gate up down, two RMSNorm vectors per layer, a final RMSNorm,
   an embedding table and an LM head that may reuse it (tie_word_embeddings). Also registers the home-tile teaser.
   Words (EN / ZH, the kit's A.terms, CONTRACT §11): token ID, embedding table / embedding 表, hidden size d / 隐藏维度 d, layer / 层,
   attention block / attention 块, MLP block / MLP 块, intermediate size / 中间维度, residual connection / 残差连接, RMSNorm, LM head,
   logit, parameter / 参数, initial setup / 初始设置. Numbers over 6 digits as 8.03B / 525M with the exact value on hover (R6.18). */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='transformer';

/* ---------- presets: config.json of each model (specs/transformer-facts.md A1–A5) ---------- */
const MODELS={
 'llama32-1b':{name:'Llama-3.2-1B',cfg:{V:128256,d:2048,L:16,H:32,KV:8,f:8192,tied:true}},
 'llama31-8b':{name:'Llama-3.1-8B',cfg:{V:128256,d:4096,L:32,H:32,KV:8,f:14336,tied:false}},
 'mistral-7b':{name:'Mistral-7B-v0.1',cfg:{V:32000,d:4096,L:32,H:32,KV:8,f:14336,tied:false}},
 'llama31-70b':{name:'Llama-3.1-70B',cfg:{V:128256,d:8192,L:80,H:64,KV:8,f:28672,tied:false}},
 'llama31-405b':{name:'Llama-3.1-405B',cfg:{V:128256,d:16384,L:126,H:128,KV:8,f:53248,tied:false}}
};
const MKEYS=Object.keys(MODELS);
const KEYS=['V','d','L','H','KV','f','tied'];
const PREC={bf16:2,fp8:1,fp4:0.5}, PL={bf16:'BF16',fp8:'FP8',fp4:'FP4'};
const DEFAULT={...MODELS['llama31-8b'].cfg,p:'bf16'};
/* slider stops: every preset value is on them; every hidden size divides by every head count (head_dim stays whole) */
const D_STOPS=[1024,1536,2048,2560,3072,3584,4096,5120,6144,7168,8192,12288,16384];
const F_STOPS=[2048,4096,5632,8192,11008,14336,18944,28672,43008,53248];
const V_STOPS=[16000,32000,50257,65536,128256,152064,256000];
const H_OPTS=[8,16,32,64,128], KV_OPTS=[1,2,4,8,16,32,64,128];
const L_MIN=1, L_MAX=128;
/* the challenge: Llama-3.1-8B's shape in BF16, only the layer count moves; 20 GiB of weights on one RTX 4090 (24 GB read as 24 GiB) */
const CH={...DEFAULT}, CH_BUDGET=20, CH_GPU=24, CH_MIN=32, CH_MAX=56, CH_ROWS=[32,36,40,44,48,52];
/* part 3: one token through the stack. Token ids from Meta's Llama 3 tokenizer (models/llama3/tokenizer.model in meta-llama/llama-models):
   "The capital of France is" → 791 6864 315 9822 374; " Paris" → 12366 */
const TOK={id:374,next:12366};
const WALK=['id','emb','attn','mlp','rest','norm','head'];

/* ---------- engine: a pure function of the config ---------- */
function compute(s){
  const d=s.d, hd=d/s.H, qw=s.H*hd, kvw=s.KV*hd;
  const Wq=d*qw, Wk=d*kvw, Wv=d*kvw, Wo=qw*d, attn=Wq+Wk+Wv+Wo, mlp=3*d*s.f, lnorm=2*d, layer=attn+mlp+lnorm;
  const emb=s.V*d, head=s.tied?0:s.V*d, fnorm=d, layers=s.L*layer, total=emb+layers+fnorm+head;
  const parts={mlp:s.L*mlp,attn:s.L*attn,emb,head,norm:s.L*lnorm+fnorm};
  const bytes=total*PREC[s.p];
  const mult=total-(s.tied?0:emb);      // weights a token is multiplied by: all but the embedding table's lookup (a tied table is multiplied as the LM head)
  return {s,hd,qw,kvw,Wq,Wk,Wv,Wo,attn,mlp,lnorm,layer,emb,head,fnorm,layers,total,parts,bytes,gib:bytes/A.GiB,gb:bytes/1e9,mult,flops:2*mult};
}
const presetOf=s=>MKEYS.find(k=>KEYS.every(x=>MODELS[k].cfg[x]===s[x]))||null;
/* the stack's one scale (R6.10, R3.3): a full-length bar is 1,050,673,152 parameters, the longest bar any guided part draws
   (Llama-3.1-70B's embedding table, and the "twice as wide" one). Only a config with a longer bar gets its own scale; the caption says so. */
const SCALE=compute({...MODELS['llama31-70b'].cfg,p:'bf16'}).emb;
const scaleOf=r=>Math.max(SCALE,r.emb,r.layer,r.head||0);
const maxLayersIn=budgetGiB=>{let best=0; for(let l=1;l<=L_MAX;l++){if(compute({...CH,L:l}).gib<=budgetGiB+1e-12) best=l;} return best;};
const CH_BEST=maxLayersIn(CH_BUDGET);

/* ---------- formatting ---------- */
const fmtInt=A.fmt.int;
const g1=v=>v.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});       // 15.0, 756.0: one decimal, always shown
const g3=v=>v.toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3});
const gibS=v=>g1(v)+' GiB', gbS=v=>g1(v)+' GB';
/* R6.18: a count over 6 digits is written 8.03B / 525M / 41.9M in both languages (A.ui.big, no hover since the owner's clarification of 2026-10-07).
   bgT is the plain-text form for places without markup (titles, metric cards, the mini result bar, announcements).
   Exact values stay in Show the math and in tooltips. */
const bgT=n=>A.fmt.big(n);
/* the kit's A.ui.big and its wording (R6.18, common r15 (f)); the sentence around it says what the number counts */
const BG=n=>Math.abs(n)<1e6?fmtInt(n):A.ui.big(n);
/* R6.12: GiB carries its definition where it first appears, and a bytes → GiB result shows GB too: "16.1 GB = 15.0 GiB" */
const GIB=()=>A.ui.def('GiB',A.terms.gib());
const memEq=(r,def)=>`${gbS(r.gb)} = ${g1(r.gib)} ${def?GIB():'GiB'}`;
/* 128,256 is written out (6 digits) with the familiar rounding (R6.18); its K is 1,000, unlike the K of a context length, so it
   says so on hover (R14 term table) */
const K128=()=>DEF(tx('about 128K','约 128K'),tx('Here K is 1,000: 128,000 text tokens plus 256 special tokens. For context length K is 1,024, so 128K tokens are 131,072.',
                                                 '这里的 K 是 1,000：128,000 个文本 token 加 256 个特殊 token。上下文长度的 K 是 1,024，128K 个 token 就是 131,072 个。'));
const vocTxt=V=>V===128256?tx(`128,256 (${K128()})`,`128,256（${K128()}）`):fmtInt(V);
/* shares: three significant digits, at least one decimal, at most three ("70.2%", "6.54%", "0.003%") */
const shareDec=v=>v<=0?1:Math.max(1,Math.min(3,2-Math.floor(Math.log10(v))));
const shareRound=v=>{const d=shareDec(v), k=10**d; return {v:Math.round(v*k+1e-9)/k,d};};
function pctS(frac){const v=frac*100; if(v<=0) return '0%'; if(v<0.0005) return '<0.001%'; const {v:x,d}=shareRound(v); return x.toFixed(d)+'%';}
/* the embedding table and LM head card: the sum of the two shares as shown, so it reproduces from them (R6.6) */
function vocShare(r){const e=r.parts.emb/r.total*100, h=r.parts.head/r.total*100;
  if(!h) return pctS(r.parts.emb/r.total);
  const a=shareRound(e), b=shareRound(h), d=Math.max(a.d,b.d); return (a.v+b.v).toFixed(d)+'%';}
const pct0=frac=>Math.round(frac*100)+'%';
/* R6.20: explanations say the trend and the reason; a share or a ratio the engine computes is said in words there ("about four fifths",
   "four times as big"), and its exact value stays on the stage, the metric cards, the tooltips and Show the math */
const FR=[[1/10,'a tenth','十分之一'],[1/5,'a fifth','五分之一'],[1/4,'a quarter','四分之一'],[1/3,'a third','三分之一'],[2/5,'two fifths','五分之二'],[1/2,'half','一半'],
          [3/5,'three fifths','五分之三'],[2/3,'two thirds','三分之二'],[3/4,'three quarters','四分之三'],[4/5,'four fifths','五分之四'],[9/10,'nine tenths','十分之九']];
const FR_ROUND=FR.filter(f=>[1/10,1/5,1/4,1/3,1/2,2/3,3/4].includes(f[0]));
const fracW=x=>{const f=FR.reduce((a,b)=>Math.abs(b[0]-x)<Math.abs(a[0]-x)?b:a), ab=Math.abs(f[0]-x)>=0.005; return tx((ab?'about ':'')+f[1],(ab?'约':'')+f[2]);};
const moreW=x=>{const f=FR_ROUND.slice().reverse().find(g=>g[0]<x); return f?tx('more than '+f[1],f[2]+'以上'):tx('some','一些');};
const NW=['','once','twice','three times','four times','five times','six times','seven times','eight times','nine times','ten times'], NZ='一两三四五六七八九十';
/* "four times" / "about four times" and "四倍"; ex says whether "about" / 约 is needed */
const timesW=q=>{const n=Math.round(q), ex=Math.abs(q-n)<0.05; return {ex,en:(ex?'':'about ')+(NW[n]||`${n} times`),zh:n>=1&&n<=10?NZ[n-1]+'倍':` ${n} 倍`};};
const bytesTxt=p=>{const b=PREC[p]; return tx(b===1?'1 byte':`${b} bytes`,`${b} 字节`);};
const layersTxt=n=>tx(n===1?'1 layer':`${fmtInt(n)} layers`,`${fmtInt(n)} 层`);
/* model names and "8 × H100" never break across lines (the kit's copy handler gives plain characters back) */
const NB=s=>String(s).replace(/\b(?:Llama-3\.[12]-\d+B|Mistral-7B-v0\.1)\b/g,m=>m.replace(/-/g,'‑')).replace(/(\d) × (?=H100|RTX)/g,'$1 × ').replace(/RTX 4090/g,'RTX 4090');
const DEF=(text,tip)=>A.ui.def(text,tip);
/* "a 4,096 × …" but "an 8,192 × …": the article follows how the number is read aloud */
const an=n=>{const g=String(n).split(',')[0]; return /^8/.test(g)||g==='11'||g==='18'?'an':'a';};
const J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
/* another lab, by its short name with its question on hover (CONTRACT §9); in Chinese the name sits in quotation marks */
const LK=id=>{const a=A.shell.labLink(id); return tx(a,`“${a}”`);};
/* H4 (the owner, 2026-10-09): a lab not built yet is named as planned only through its status, so the sentence becomes a link the day it ships */
const quantRef=()=>{const c=A.catalog, l=c&&c.LAB&&c.LAB.quantization; return l&&c.labStatus(l)==='live'?tx(` (the lab ${A.shell.labLink('quantization')} covers it)`,`（${LK('quantization')}这个 Lab 会讲）`):tx(' (a planned lab on quantization covers it)','（规划中的量化 Lab 会讲）');};
/* one name for the setup everywhere (stage title, mini result bar, scenario result, announcements): a model, part 6's option,
   the challenge's layer count, or a custom config (R6.13) */
function configName(s){
  const k=presetOf(s); if(k) return MODELS[k].name;
  if(mode==='challenge') return tx(`Llama-3.1-8B’s shape with ${layersTxt(s.L)}`,`Llama-3.1-8B 的结构，${layersTxt(s.L)}`);
  if(mode==='guided'&&STEPS[step].id==='config'){const i=lvNow(); if(i>0) return tx(`Llama-3.1-8B ${L(LV[i].t).toLowerCase()}`,`Llama-3.1-8B ${L(LV[i].t)}`);}
  return tx('Custom config','自定义配置');}

/* ---------- module state (survives re-mounts and language switches) ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, root=null, app=null;
let walk=0, revealed=false, saved=null, chSubmitted=false, chReveal=false, drillSeed=5, drillOpen=false, annT=null;
let seenMistral=false; const lvTried=new Set(['start']);
let userAct=false; const shownPrev={};
let tzModel='llama31-8b', wantObs=false;

/* ---------- definitions shown on hover (R2.8): the first place a reader meets each term. The structural terms are the kit's
   (A.terms, the words of the lab Architecture overview, CONTRACT §11), each followed by this lab's own model sentence ---------- */
const TT=(k,en,zh)=>A.terms[k]()+tx(' '+en,zh);
const TIP={
 vocab:()=>A.terms.token()+tx(' The vocabulary is every token the model knows: 128,256 in Llama-3.1-8B, 128,000 text tokens and 256 special ones, so about 128K with K = 1,000.',
                              '词表是模型认识的全部 token：Llama-3.1-8B 有 128,256 个，其中 128,000 个文本 token，256 个特殊 token，所以约 128K（这里 K 是 1,000）。'),
 hidden:()=>TT('hidden','In Llama-3.1-8B, d = 4,096.','Llama-3.1-8B 的 d 是 4,096。'),
 layer:()=>TT('layer','Llama-3.1-8B stacks 32 of them.','Llama-3.1-8B 叠了 32 层。'),
 /* part 1 names the two head counts by the shapes they fix; what heads do is the next lab's subject (R2.8: no variant before the base idea).
    No query/key/value here, and no "a head is a slice of the token": the mha lab lists that as a misconception (R2.8, R6.10) */
 heads:()=>tx('How many heads the attention block runs side by side: 32 in Llama-3.1-8B. Each head works with pieces of 128 numbers (head_dim = 4,096 ÷ 32), computed by the attention block’s matrices Wq, Wk and Wv from all 4,096 numbers of the token, not cut from the token itself. The next number uses head_dim.',
              'attention 块里并排跑几个 head：Llama-3.1-8B 是 32 个。attention 块的矩阵 Wq、Wk、Wv 用 token 的全部 4,096 维算出新的向量，每个 head 分到其中 128 维（head_dim = 4,096 ÷ 32），不是从 token 里切下的一段。下一项会用到 head_dim。'),
 kvheads:()=>tx('How wide two of the attention block’s four matrices, Wk and Wv, are: they give 8 × 128 = 1,024 numbers per token instead of 4,096. So each is a quarter the size of Wq or Wo. A KV head is one key and one value each token stores in every layer, which several heads can share; Llama-3.1-8B lets every 4 heads share one, as the lab GQA and MLA explains.',
                '它决定 attention 块四个矩阵里 Wk 和 Wv 有多宽：每个 token 只得到 8 × 128 = 1,024 维，而不是 4,096 维。所以它们各只有 Wq 或 Wo 的四分之一大。KV head 是每个 token 在每一层存的一组 key 和 value，可以由几个 head 共用；Llama-3.1-8B 让每 4 个 head 共用一组，“GQA 和 MLA”这个 Lab 里讲。'),
 inter:()=>TT('inter','In Llama-3.1-8B it is 14,336, 3.5 times the hidden size.','Llama-3.1-8B 是 14,336 维，是隐藏维度的 3.5 倍。'),
 tie:()=>tx('Whether the two ends of the model share their parameters. At the start, the embedding table turns a token ID into a vector; at the end, the LM head (LM: language model) turns the last vector into one score per vocabulary token. If true, the LM head reuses the embedding table’s parameters. In Llama-3.1-8B they are two separate tables (false).',
            '模型两头是否共享参数。开头的 embedding 表把 token ID 变成向量，末尾的 LM head（LM 即语言模型，language model）把最后的向量变成词表里每个 token 的一个得分。为 true 时，LM head 直接用 embedding 表的参数。Llama-3.1-8B 的两张表是分开的（false）。'),
 layerAny:()=>TT('layer',`This model stacks ${fmtInt(S.L)}.`,`这个模型叠了 ${fmtInt(S.L)} 层。`),
 /* no counts here: part 2 asks for the attention blocks' share */
 attnBlock:()=>TT('attention','Llama-3.1-8B has one in each of its 32 layers.','Llama-3.1-8B 的 32 层里每层一个。'),
 mlpBlock:()=>TT('mlp','In Llama-3.1-8B the intermediate size is 14,336.','Llama-3.1-8B 的中间维度是 14,336。'),
 /* part 3's MLP stop names the width in words; the numbers of the current config are here (R6.20) */
 interNow:()=>TT('inter',`Here it is ${fmtInt(S.f)}, ${A.fmt.num(S.f/S.d,1)} times the hidden size of ${fmtInt(S.d)}.`,`这里是 ${fmtInt(S.f)} 维，是隐藏维度 ${fmtInt(S.d)} 的 ${A.fmt.num(S.f/S.d,1)} 倍。`),
 /* the shapes in these come from the current config; tooltips keep exact counts (R6.18) */
 emb:()=>TT('embedding',`Here it has ${fmtInt(S.V)} rows of ${fmtInt(S.d)} numbers: ${fmtInt(S.V*S.d)} parameters.`,`这里它有 ${fmtInt(S.V)} 行，每行 ${fmtInt(S.d)} 维，共 ${fmtInt(S.V*S.d)} 个参数。`),
 residual:()=>TT('residual',`In this model the vector stays ${fmtInt(S.d)} numbers long.`,`这个模型里向量一直是 ${fmtInt(S.d)} 维。`),
 rms:()=>TT('rmsnorm',`In this model each has ${fmtInt(S.d)} parameters, two per layer and one after the last layer; they are the only parameters outside the matrices.`,
                      `这个模型里每个 RMSNorm 有 ${fmtInt(S.d)} 个参数，每层两个，最后一层之后还有一个；它们是矩阵之外仅有的参数。`),
 head:()=>A.terms.lmhead()+(S.tied?tx(' Here the two are tied: the LM head reuses the embedding table’s parameters, so it adds none.','这里两者共享参数：LM head 直接用 embedding 表的参数，不另加参数。')
                               :tx(` Here it is ${an(fmtInt(S.d))} ${fmtInt(S.d)} × ${fmtInt(S.V)} matrix (hidden size × vocabulary size): ${fmtInt(S.d*S.V)} parameters.`,`这里它是一个 ${fmtInt(S.d)} × ${fmtInt(S.V)} 的矩阵（隐藏维度 × 词表大小），共 ${fmtInt(S.d*S.V)} 个参数。`)),
 logit:()=>A.terms.logit(),
 param:()=>A.terms.param(),
 flops:()=>tx('A FLOP is one floating-point operation. Multiplying by one weight and adding the product to a sum is 2 FLOPs.',
              'FLOP 是一次浮点运算。乘上一个权重再把乘积加进总和，是 2 次 FLOP。'),
 /* the challenge and the RTX 4090 scenario leave memory for these two; both are other labs' subjects, so each gets its size here (R2.7) */
 kvcache:()=>TT('kvcache','For Llama-3.1-8B in BF16, the two vectors that Wk and Wv make take 128 KiB per token (2 × 32 layers × 1,024 numbers × 2 bytes), so each GiB holds 8,192 tokens.',
                          'Llama-3.1-8B 在 BF16 下，Wk 和 Wv 算出的这两个向量每个 token 占 128 KiB（2 × 32 层 × 1,024 个数值 × 2 字节），所以每 GiB 能放 8,192 个 token。'),
 overhead:()=>tx('Runtime overhead is memory the serving engine needs besides the weights and the KV cache, for temporary activations, CUDA graphs and communication buffers. Real engines measure it at startup.',
                 '运行时开销是推理引擎在权重和 KV cache 之外还要的显存，用于临时激活、CUDA graph 和通信缓冲区。真实引擎在启动时实测它。')
};

/* ---------- guided parts ---------- */
const PRED=[{v:'q1',t:{en:'Under 1/4',zh:'不到 1/4'}},{v:'q2',t:{en:'1/4 to 1/2',zh:'1/4 到 1/2'}},{v:'q3',t:{en:'1/2 to 3/4',zh:'1/2 到 3/4'}},{v:'q4',t:{en:'Over 3/4',zh:'3/4 以上'}}];
const predRight=f=>f<0.25?'q1':(f<0.5?'q2':(f<0.75?'q3':'q4'));
const base=()=>({...DEFAULT});
const r8=()=>compute(DEFAULT);
/* part 6: single choice, each from the initial setup (R3.6); "twice as wide" keeps Llama-3.1's intermediate size of 3.5 × d */
const LV=[
 {id:'start',t:{en:'Initial setup',zh:'初始设置'},d:()=>tx('Llama-3.1-8B as released','Llama-3.1-8B 原样'),p:{}},
 {id:'wide',t:{en:'Twice as wide',zh:'宽一倍'},d:()=>tx(`Hidden size ${fmtInt(DEFAULT.d)} → ${fmtInt(2*DEFAULT.d)} and intermediate size ${fmtInt(DEFAULT.f)} → ${fmtInt(2*DEFAULT.f)}, still ${A.fmt.num(DEFAULT.f/DEFAULT.d,1)} times the hidden size`,
   `隐藏维度 ${fmtInt(DEFAULT.d)} → ${fmtInt(2*DEFAULT.d)}，中间维度 ${fmtInt(DEFAULT.f)} → ${fmtInt(2*DEFAULT.f)}，仍是隐藏维度的 ${A.fmt.num(DEFAULT.f/DEFAULT.d,1)} 倍`),p:{d:2*DEFAULT.d,f:2*DEFAULT.f}},
 {id:'deep',t:{en:'Twice as deep',zh:'深一倍'},d:()=>tx(`${DEFAULT.L} → ${2*DEFAULT.L} layers, nothing else`,`${DEFAULT.L} 层 → ${2*DEFAULT.L} 层，其他都不变`),p:{L:2*DEFAULT.L}}
];
const lvCfg=i=>({...DEFAULT,...LV[i].p});
const lvNow=()=>LV.findIndex((l,i)=>{const c=lvCfg(i); return KEYS.every(k=>c[k]===S[k])&&c.p===S.p;});
const MISTRAL=()=>({...MODELS['mistral-7b'].cfg,p:'bf16'});

const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},lead:{en:'Llama‑3.1‑8B has 8.03B parameters: where in the model are they?',zh:'Llama‑3.1‑8B 有 8.03B 个参数：它们都在模型的哪里？'},enter:base,body:()=>{const r=r8();
   const cfg=[['vocab_size',DEFAULT.V,'vocab'],['hidden_size',DEFAULT.d,'hidden'],['num_hidden_layers',DEFAULT.L,'layer'],['num_attention_heads',DEFAULT.H,'heads'],
     ['num_key_value_heads',DEFAULT.KV,'kvheads'],['intermediate_size',DEFAULT.f,'inter'],['tie_word_embeddings',String(DEFAULT.tied),'tie']];
   const js=`<div class="tf-cfg mono" role="list" aria-label="config.json">${cfg.map(([k,v,t],i)=>`<div role="listitem"><span>"${DEF(k,TIP[t]())}": ${v}${i<cfg.length-1?',':''}</span></div>`).join('')}</div>`;
   const more=tx(`<p class="small muted">The lab ${LK('mha')} explains what heads do; the lab ${LK('gqa-mla')} explains why Wk and Wv are narrower.</p>`,
                 `<p class="small muted">head 做什么，${LK('mha')}这个 Lab 会讲；Wk 和 Wv 为什么更窄，见${LK('gqa-mla')}。</p>`);
   /* R6.20: the mission keeps the count, the lab's subject, and names the memory in words; the weights card on the stage shows
      15.0 GiB and 16.1 GB (its tooltip defines both units, R6.12), part 7 says how memory follows from the count */
   return tx(`<p>Llama-3.1-8B has ${BG(r.total)} ${DEF('parameters',TIP.param())}, and each one takes up GPU memory. Where in the model are they?</p><p>Seven numbers from its config.json are enough to count every one of them:</p>${js}${more}`,
             `<p>Llama-3.1-8B 有 ${BG(r.total)} 个${DEF('参数',TIP.param())}，每一个都要占显存。这些参数都在模型的哪里？</p><p>它的 config.json 里有 7 项，就足够把每一个参数都数出来：</p>${js}${more}`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: how much of the 8.03B is in the attention blocks?',zh:'先猜：8.03B 个参数里，attention 块占多少？'},enter:base,gate:()=>!!pred,skip:true,body:()=>{const r=r8();
   const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   const saved_=pred?`<p class="muted small">${tx('Saved. You can still change it before part 4 shows the answer.','已记下。第 4 部分揭晓答案之前，还可以改。')}</p>`:'';
   return tx(`<p>Each of the ${DEFAULT.L} layers has an ${DEF('attention block',TIP.attnBlock())}, where a token takes information from the tokens before it, and an ${DEF('MLP block',TIP.mlpBlock())}, which transforms each token on its own. Of the ${BG(r.total)} parameters, how much is in the attention blocks?</p>${ch}${saved_}`,
             `<p>${DEFAULT.L} 层里，每层都有一个 ${DEF('attention 块',TIP.attnBlock())}，token 在这里从前面的 token 取信息；还有一个 ${DEF('MLP 块',TIP.mlpBlock())}，单独变换每个 token。${BG(r.total)} 个参数里，attention 块一共占多少？</p>${ch}${saved_}`);}},
 {id:'walk',title:{en:'One token, top to bottom',zh:'一个 token 从上走到下'},lead:{en:'Move “ is” down the model and see which parts hold parameters.',zh:'让 “ is” 往下走，看哪些部件有参数。'},enter:base,expect:()=>walk>=WALK.length-1,showMe:true,obsAlways:true,
  label:()=>tx(`“ is” on its way down: ${walk+1} / ${WALK.length}`,`“ is” 往下走：${walk+1} / ${WALK.length}`),
  body:()=>tx(`<p>The lab ${LK('architecture')} follows a token through every component; here the walk only marks which of them hold parameters. Move “ is” (space included), the last token of “The capital of France is”, down the model.</p>`,
              `<p>第一个 Lab${LK('architecture')}已经跟着一个 token 走过每个部件；这里只看哪些部件有参数。让 “The capital of France is”（法国的首都是）的最后一个 token “ is”（连同前面的空格）往下走。</p>`)+walkCtl(),
  obs:r=>walkObs(r)},
 {id:'split',title:{en:'Where the parameters are',zh:'参数在哪'},lead:{en:'The MLP blocks hold 70.2% of the parameters, attention 16.7%.',zh:'MLP 块占 70.2% 的参数，attention 块占 16.7%。'},enter:base,label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Every layer has the same shapes, so one layer’s split holds for all ${DEFAULT.L}.</p>`,`<p>每一层的形状都一样，所以一层怎么分，${DEFAULT.L} 层就怎么分。</p>`),
  obs:r=>splitObs(r)},
 {id:'vocab',title:{en:'What the vocabulary costs',zh:'词表有多贵'},lead:{en:'The same 32 layers with a vocabulary 4 times smaller: compare the two models.',zh:'32 层完全一样，词表小 4 倍：比一比这两个模型。'},enter:base,expect:()=>seenMistral,showMe:'mistral',label:{en:'Observed',zh:'观察到'},
  body:()=>{const m=MODELS['mistral-7b'].cfg;
   return tx(`<p>Mistral-7B-v0.1 has the same ${m.L} layers as Llama-3.1-8B: the same hidden size, intermediate size and heads. Its vocabulary has ${fmtInt(m.V)} tokens instead of ${vocTxt(DEFAULT.V)}.</p>`,
             `<p>Mistral-7B-v0.1 的 ${m.L} 层和 Llama-3.1-8B 完全一样：隐藏维度、中间维度、head 数都相同。只是它的词表有 ${fmtInt(m.V)} 个 token，而不是 ${fmtInt(DEFAULT.V)} 个（${K128()}）。</p>`)+
     A.ui.seg({id:'tf-ic-model',key:'model',label:tx('Model','模型'),options:[['llama31-8b',MODELS['llama31-8b'].name],['mistral-7b',MODELS['mistral-7b'].name]],value:presetOf(S)||'',inline:true});},
  obs:r=>vocabObs(r)},
 {id:'config',title:{en:'Wider or deeper?',zh:'宽一倍还是深一倍？'},lead:{en:'Twice as wide or twice as deep: which adds more parameters?',zh:'宽一倍还是深一倍：哪个多出的参数更多？'},enter:base,expect:()=>lvTried.size>1,showMe:'wide',obsAlways:true,levers:true,label:{en:'Result',zh:'结果'},
  body:()=>{const cur=lvNow();
   const opts=A.ui.levers({aria:tx('Options, pick one','选项，选一个'),value:cur,options:LV.map((l,i)=>({v:i,title:L(l.t),desc:l.d(),result:lvTried.has(l.id)?esc(bgT(compute(lvCfg(i)).total)):''}))});
   return tx(`<p>Which grows more: Llama-3.1-8B twice as wide, or twice as deep?</p><p class="small">Pick a change and see what the parameter count becomes. Each starts from the initial setup.</p>`,
             `<p>Llama-3.1-8B 宽一倍和深一倍，哪个长得多？</p><p class="small">选一种改法，看看参数总数变成多少。每种改法都从初始设置出发。</p>`)+opts;},
  obs:r=>configObs(r)},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},enter:base,body:()=>rememberBody()}
];
const PART={};
STEPS.forEach((s,i)=>{PART[s.id]=i;});

function walkCtl(){
  return `<div class="inline-ctl tf-walkctl" id="tf-ic-walk"><div class="lbl"><span>${tx('Move “ is” down the model','让 “ is” 在模型里往下走')}</span></div>`+
    `<div class="tf-walkbtns"><button type="button" class="btn ghost sm" data-walk="up"${walk<=0?' disabled':''}>${tx('◂ Up','◂ 往上')}</button><button type="button" class="btn sm" data-walk="down"${walk>=WALK.length-1?' disabled':''}>${tx('Down ▸','往下 ▸')}</button></div></div>`;
}
/* one thing per stop, from the current config (R6.13: every number says what it counts). R6.20: each stop says in words whether the
   component holds parameters; the shapes are on the stage's labels and in the hovers of the terms */
function walkObs(r){
  const s=r.s, d=fmtInt(s.d), llama=s.V===DEFAULT.V, st=WALK[walk];
  const none=tx('; no parameters are used yet.','，这里还没有用到参数。');
  if(st==='id') return llama?tx(`<p><b>Token ID ${TOK.id}.</b> The tokenizer turns “ is” into its number in the vocabulary${none}</p>`,
                                `<p><b>token ID ${TOK.id}。</b>tokenizer 把 “ is” 换成它在词表里的编号${none}</p>`)
                           :tx(`<p><b>Token ID.</b> The tokenizer turns “ is” into its number in this model’s vocabulary${none}</p>`,`<p><b>token ID。</b>tokenizer 把 “ is” 换成它在这个模型的词表里的编号${none}</p>`);
  if(st==='emb') return tx(`<p><b>${DEF('Embedding table',TIP.emb())}.</b> The ID picks its row of the table: the token is now a vector of ${d} numbers. Nothing is multiplied.</p>`,
                           `<p><b>${DEF('embedding 表',TIP.emb())}。</b>这个 ID 取出表里对应的一行：token 变成一个 ${d} 维的向量。这里只是查表，不做乘法。</p>`);
  if(st==='attn') return tx(`<p><b>Layer 1, attention block.</b> After an ${DEF('RMSNorm',TIP.rms())}, its four matrices let “ is” take information from the tokens before it. Its output, ${d} numbers, is added back at ⊕, the ${DEF('residual connection',TIP.residual())}, which has no parameters.</p>`,
                            `<p><b>第 1 层，attention 块。</b>先过一个 ${DEF('RMSNorm',TIP.rms())}，块里的四个矩阵再让 “ is” 从前面的 token 取信息。它的输出（${d} 维）在 ⊕ 处加回去，这就是${DEF('残差连接',TIP.residual())}，它没有参数。</p>`);
  if(st==='mlp') return tx(`<p><b>Layer 1, MLP block.</b> After another RMSNorm, three matrices widen the vector to the ${DEF('intermediate size',TIP.interNow())} and narrow it back again. Its output is added back at ⊕ too.</p>`,
                           `<p><b>第 1 层，MLP 块。</b>再过一个 RMSNorm，三个矩阵先把向量放大到${DEF('中间维度',TIP.interNow())}，再缩回原来的维数。它的输出同样在 ⊕ 处加回去。</p>`);
  if(st==='rest') return s.L>1?tx(`<p><b>Layers 2 to ${s.L}.</b> Each layer repeats the same two blocks with its own parameters. The vector keeps the same length the whole way.</p>`,
                                  `<p><b>第 2 到 ${s.L} 层。</b>每一层都重复同样的两个块，但各用自己的参数。向量一路都是同样的维数。</p>`)
                              :tx('<p><b>No more layers.</b> This config has only one layer.</p>','<p><b>没有别的层了。</b>这个配置只有一层。</p>');
  if(st==='norm') return tx(`<p><b>Final RMSNorm.</b> One last rescaling, with ${d} parameters.</p>`,`<p><b>最后的 RMSNorm。</b>最后再缩放一次，有 ${d} 个参数。</p>`);
  const tied=s.tied?tx(' Here it is tied: it reuses the embedding table’s parameters.','这里它和 embedding 表共享参数，不另加参数。'):'';
  return tx(`<p><b>${DEF('LM head',TIP.head())}.</b> A matrix the size of the embedding table gives one ${DEF('logit',TIP.logit())} per vocabulary token; the next token is picked from them.${tied} ${llama?`Training makes the right next token get a large logit: here “ Paris”, token ID ${TOK.next}.`:'Training makes the right next token get a large logit.'}</p>`,
            `<p><b>${DEF('LM head',TIP.head())}。</b>一个和 embedding 表一样大的矩阵，把向量变成词表里每个 token 一个 ${DEF('logit',TIP.logit())}，下一个 token 从中选出。${tied}${llama?`训练的目标是让正确的下一个 token 得到大的 logit，这里就是 “ Paris”（token ID ${TOK.next}）。`:'训练的目标是让正确的下一个 token 得到大的 logit。'}</p>`);
}
function splitObs(r){
  const b=r8(), same=presetOf(S)==='llama31-8b', fa=b.parts.attn/b.total, fm=b.parts.mlp/b.total, inL=b.mlp/b.layer, x=A.fmt.num(DEFAULT.f/DEFAULT.d,1);
  let h=same?'':tx(`<p>You changed something under ${J('controls','Settings')}, so the chart shows another model. The question was about Llama-3.1-8B:</p>`,
                   `<p>你在${J('controls','“自由调参”')}里改了设置，图上现在是另一个模型。题目问的是 Llama-3.1-8B：</p>`);
  /* R6.20: one key number, the share the prediction asked for; the counts are on the metric cards, the shapes in Show the math */
  h+=tx(`<p><b>The attention blocks hold only ${pctS(fa)} of the parameters; the MLP blocks hold ${fm>0.5?'most of them':'far more'}.</b></p>`,
        `<p><b>attention 块只占全部参数的 ${pctS(fa)}，${fm>0.5?'大部分参数在 MLP 块里':'MLP 块占得多得多'}。</b></p>`);
  if(pred){const right=predRight(fa), pt=L(PRED.find(p=>p.v===pred).t), rt=L(PRED.find(p=>p.v===right).t);
    h+=pred===right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
                   :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);}
  h+=tx(`<p>Why: each of the MLP block’s three matrices has the ${DEF('intermediate size',TIP.inter())} on one side, ${x} times the hidden size, while attention’s Wk and Wv are narrow. So ${fracW(inL)} of every layer is MLP.</p>`,
        `<p>原因：${DEF('中间维度',TIP.inter())}是隐藏维度的 ${x} 倍，MLP 块的三个矩阵都有一边是它；attention 块的 Wk 和 Wv 却很窄。所以每层${fracW(inL)}的参数在 MLP 块里。</p>`);
  return h;
}
function vocabObs(r){
  const k=presetOf(S), a=compute({...DEFAULT,p:S.p}), m=compute({...MISTRAL(),p:S.p}), diff=a.total-m.total, mc=MODELS['mistral-7b'].cfg;
  let h;
  if(k==='mistral-7b') h=tx(`<p><b>Mistral-7B-v0.1: ${BG(m.total)} parameters, ${BG(diff)} fewer than Llama-3.1-8B, the initial setup.</b></p>`,
                          `<p><b>Mistral-7B-v0.1：${BG(m.total)} 个参数，比初始设置 Llama-3.1-8B 少 ${BG(diff)} 个。</b></p>`);
  else if(k==='llama31-8b') h=tx(`<p><b>Llama-3.1-8B, the initial setup: ${BG(a.total)} parameters, ${BG(diff)} more than Mistral-7B-v0.1.</b></p>`,
                               `<p><b>初始设置 Llama-3.1-8B：${BG(a.total)} 个参数，比 Mistral-7B-v0.1 多 ${BG(diff)} 个。</b></p>`);
  else return tx(`<p>You changed something under ${J('controls','Settings')}, so the model is neither of the two. Right now it has <b>${BG(r.total)} parameters</b>.</p>`,
                 `<p>你在${J('controls','“自由调参”')}里改了设置，现在的模型两个都不是。它现在有 <b>${BG(r.total)} 个参数</b>。</p>`);
  /* R6.20: the reason in words; the line (128,256 − 32,000) × 4,096 × 2 = 788,529,152 is in Show the math */
  const q=fracW(mc.V/DEFAULT.V);
  h+=tx(`<p>All of the difference is the vocabulary, ${q} the size in Mistral-7B-v0.1: the embedding table and LM head hold one vector per token. The ${mc.L} layers are identical.</p>`,
        `<p>差别全在词表：Mistral-7B-v0.1 的词表只有${q}大，而 embedding 表和 LM head 给每个 token 各存一个向量。${mc.L} 层完全一样。</p>`);
  return h;
}
/* Chinese: no space between a full-width closing bracket and the operator after it, “（4,096）+ …” */
const zop=h=>tx(h,h.replace(/）((?:<\/[a-z]+>)*) ([+=÷×−])/g,'）$1$2'));
/* part 6 (R6.20): the result says the size and the reason in words; the count, line by line, stays in Show the math, one click away (R1.10) */
const mathLink=()=>tx(` ${J('math',A.t('showMath'))} counts it.`,`算式见${J('math','“'+A.t('showMath')+'”')}。`);
function configObs(r){
  const i=lvNow(), b=compute(lvCfg(0)), ratio=v=>g1(v);
  if(i<0) return tx(`<p>You changed something under ${J('controls','Settings')}, so the setup is none of the options above. Right now: <b>${BG(r.total)} parameters</b>.${mathLink()}</p>`,
                    `<p>你在${J('controls','“自由调参”')}里改了设置，现在的配置不属于上面任何一个选项。现在是 <b>${BG(r.total)} 个参数</b>。${mathLink()}</p>`);
  if(i===0) return tx(`<p><b>Initial setup: Llama-3.1-8B, ${BG(b.total)} parameters; one layer holds ${BG(b.layer)}.</b>${mathLink()}</p>`,
                      `<p><b>初始设置 Llama-3.1-8B：${BG(b.total)} 个参数，一层有 ${BG(b.layer)} 个。</b>${mathLink()}</p>`);
  if(LV[i].id==='wide'){const t=timesW(r.layer/b.layer);
    return tx(`<p><b>Twice as wide: ${BG(r.total)} parameters, ${ratio(r.total/b.total)} times the initial setup.</b> Every matrix in a layer doubles on both sides, so each layer becomes ${t.en} as big; the embedding table and LM head only double, since the vocabulary stays the same.${mathLink()}</p>`,
              `<p><b>宽一倍：${BG(r.total)} 个参数，是初始设置的 ${ratio(r.total/b.total)} 倍。</b>层里每个矩阵的两条边都翻了一倍，所以每层${t.ex?'':'约'}变成原来的${t.zh}；embedding 表和 LM head 只翻一倍，因为词表没变。${mathLink()}</p>`);}
  return tx(`<p><b>Twice as deep: ${BG(r.total)} parameters, ${ratio(r.total/b.total)} times the initial setup.</b> The layers double; the embedding table, LM head and final RMSNorm do not, so the total falls just short of twice.${mathLink()}</p>`,
            `<p><b>深一倍：${BG(r.total)} 个参数，是初始设置的 ${ratio(r.total/b.total)} 倍。</b>各层翻一倍，embedding 表、LM head 和最后的 RMSNorm 却不变，所以总数不到两倍。${mathLink()}</p>`);
}
function rememberBody(){
  /* R6.20: each point says the principle, with at most one key number; the shares of the five models are on the stage in Scenarios */
  const r=r8(), fa=r.parts.attn/r.total, t=timesW(r.parts.mlp/r.parts.attn), r16=compute({...DEFAULT,H:DEFAULT.H/2}), mi=compute(MISTRAL()), gap=r.total-mi.total;
  const sm=A.t('modeSandbox'), sc=`<button type="button" class="linkbtn" data-mode-go="sandbox" data-tipx="${esc(tx(`Click to switch to ${sm}, where the five models are.`,`点击切换到“${sm}”，五个模型都在那里。`))}">${esc(sm)}</button>`;
  /* R3.12: the guided tour ends in Scenarios (A.shell.toScenarios); Scenarios end in the challenge */
  return tx(`<ul><li>A model like Llama is a stack: an embedding table, identical layers of an attention block and an MLP block, a final RMSNorm and an LM head. Seven numbers in config.json fix every shape.</li><li>The MLP blocks hold most of the parameters in all five models under ${sc}: their three matrices span the intermediate size, several times the hidden size.</li><li>${MX('\\text{weights in memory} = \\text{parameters} \\times \\text{bytes each}')}: ${g1(r.gib)} ${GIB()} (${gbS(r.gb)}) for Llama-3.1-8B in BF16.</li></ul>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Most parameters are in attention.” In Llama-3.1-8B attention holds ${pctS(fa)}, the MLP blocks ${t.en} as much.</p><p>✗ “More heads, more parameters.” With the same KV heads, halving the heads doubles head_dim and so Wk and Wv (KV heads × head_dim wide), not Wq and Wo: the model grows from ${BG(r.total)} to ${BG(r16.total)} parameters.</p><p>✗ “A 7B model has smaller layers than an 8B one.” Mistral-7B-v0.1’s layers match Llama-3.1-8B’s; all of the ${BG(gap)} difference is in the embedding table and LM head.</p></div>`+
     A.shell.toScenarios(),
     `<ul><li>Llama 这类模型就是一叠：embedding 表，形状相同的若干层（每层一个 attention 块、一个 MLP 块），最后的 RMSNorm，再加 LM head。config.json 里的 7 项决定了所有形状。</li><li>${sc}里的五个模型，大部分参数都在 MLP 块里：它的三个矩阵都有一边是中间维度，而中间维度是隐藏维度的好几倍。</li><li>${MX('\\text{权重占的显存} = \\text{参数量} \\times \\text{每个参数的字节数}')}。Llama-3.1-8B 在 BF16 下是 ${g1(r.gib)} ${GIB()}（${gbS(r.gb)}）。</li></ul>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “大部分参数在 attention 里。”Llama-3.1-8B 的 attention 块只占 ${pctS(fa)}，MLP 块${t.ex?'':'约'}是它的${t.zh}。</p><p>✗ “head 越多，参数越多。”KV head 数不变时，head 减半，head_dim 就翻倍，宽“KV head 数 × head_dim”的 Wk 和 Wv 也跟着翻倍；Wq 和 Wo 大小不变，整个模型从 ${BG(r.total)} 个参数变成 ${BG(r16.total)} 个。</p><p>✗ “7B 模型的层比 8B 模型的小。”Mistral-7B-v0.1 的层和 Llama-3.1-8B 完全一样，${BG(gap)} 个参数的差距全在 embedding 表和 LM head 上。</p></div>`+
     A.shell.toScenarios());
}

/* ---------- scenarios: a real deployment each, a plain name and one plain line (R2.1) ---------- */
const PRESETS=[
 {k:'llama32-1b',t:{en:'On a phone',zh:'在手机上跑'},d:{en:'Llama-3.2-1B, which Meta made for phones and other small devices',zh:'Llama-3.2-1B，Meta 为手机等小设备做的'}},
 {k:'llama31-8b',t:{en:'One RTX 4090 (24 GB)',zh:'一张 RTX 4090（24 GB）'},d:{en:'Llama-3.1-8B, the initial setup',zh:'Llama-3.1-8B，初始设置'}},
 {k:'mistral-7b',t:{en:'The same RTX 4090 with a 7B',zh:'同一张 RTX 4090 换成 7B'},d:{en:'Mistral-7B-v0.1: Llama-3.1-8B’s layers, a quarter of its vocabulary',zh:'Mistral-7B-v0.1，层和 Llama-3.1-8B 一样，词表只有它的四分之一'}},
 {k:'llama31-70b',t:{en:'One 8 × H100 node',zh:'一台 8 × H100'},d:{en:'Llama-3.1-70B, the model of the KV cache memory lab',zh:'Llama-3.1-70B，“KV cache 显存”Lab 里的模型'}},
 {k:'llama31-405b',t:{en:'Serving Llama 3.1’s largest model',zh:'部署 Llama 3.1 最大的模型'},d:{en:'Llama-3.1-405B',zh:'Llama-3.1-405B'}}
];
const presetNow=()=>{const k=presetOf(S); return S.p==='bf16'?PRESETS.findIndex(p=>p.k===k):-1;};
function insightHTML(r){
  const i=presetNow(), s=r.s, b=r8();
  /* the result: the count and the weights' memory, GB beside GiB and the first GiB in this mode defined (R1.9, R6.12). R6.20: the shares
     are on the metric cards and the share bar; the sentences below say what stands out and why, with at most one number each */
  let h=tx(`<p><b>${esc(configName(s))}: ${BG(r.total)} parameters, ${g1(r.gib)} ${GIB()} (${gbS(r.gb)}) in ${PL[s.p]}.</b></p>`,`<p><b>${esc(configName(s))}：${BG(r.total)} 个参数，${PL[s.p]} 下占 ${g1(r.gib)} ${GIB()}（${gbS(r.gb)}）。</b></p>`);
  if(i<0) return h+`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`;
  const k=PRESETS[i].k;
  const kvc=DEF('KV cache',TIP.kvcache()), ovh=DEF(tx('runtime overhead','运行时开销'),TIP.overhead());
  if(k==='llama32-1b'){const q=fracW(PREC.fp4/PREC.bf16);
    h+=tx(`<p>The embedding table alone holds ${vocShare(r)} of them, and the LM head reuses its parameters. Why so large a share: the vocabulary is as big as in the initial setup, but the layers are fewer and much smaller.</p><p>In FP4 the weights take ${q} of that (pick FP4 under ${J('p','Settings')}).</p>`,
          `<p>光 embedding 表就占 ${vocShare(r)}，LM head 和它共享参数。为什么占比这么大：词表和初始设置一样大，层却更少，每层也小得多。</p><p>FP4 下权重只有 BF16 的${q}（在${J('p','“自由调参”')}里选 FP4）。</p>`);}
  if(k==='llama31-8b') h+=tx(`<p>This is the initial setup. In BF16 its weights fill ${moreW(r.gib/CH_GPU)} of the RTX 4090’s memory; the rest is left for the ${kvc} and the ${ovh}.</p>`,
                             `<p>这就是初始设置。BF16 下它的权重占了 RTX 4090 ${moreW(r.gib/CH_GPU)}的显存，剩下的留给 ${kvc} 和${ovh}。</p>`);
  if(k==='mistral-7b'){const q=fracW(s.V/b.s.V);
    h+=tx(`<p>Why it is smaller than the initial setup: the layers are the same, but the vocabulary is ${q} the size, and so are the embedding table and LM head. On the same RTX 4090, its weights take ${g1(+g1(b.gib)-(+g1(r.gib)))} GiB less than Llama-3.1-8B’s.</p>`,
          `<p>为什么比初始设置小：层一样，但词表只有它的${q}，embedding 表和 LM head 也只有原来的${q}。在同一张 RTX 4090 上，它的权重比 Llama-3.1-8B 少占 ${g1(+g1(b.gib)-(+g1(r.gib)))} GiB。</p>`);}
  if(k==='llama31-70b') h+=tx(`<p>The embedding table and LM head are only ${vocShare(r)} of the model: the vocabulary is the same as in the initial setup, but there are more layers, and each is much bigger.</p><p>Split over the 8 GPUs, each holds ${gibS(r.gib/8)} of weights. How many users fit next to them is worked out in ${LK('kv-cache')}.</p>`,
                              `<p>embedding 表和 LM head 只占整个模型的 ${vocShare(r)}：词表和初始设置一样大，但层更多，每层也大得多。</p><p>分到 8 张 GPU 上，每张存 ${gibS(r.gib/8)} 的权重。这些权重之外还能装几个用户，在${LK('kv-cache')}里算。</p>`);
  if(k==='llama31-405b'){const node=tx('One 8 × H100 node has 8 GPUs of 80 GB each, read here as 640 GiB in all.','一台 8 × H100 有 8 张 GPU，每张 80 GB，这里一共按 640 GiB 计。');
    h+=tx(`<p>The embedding table and LM head are only ${vocShare(r)} of the model: the vocabulary is the same as in the initial setup, but there are far more layers, and each is far bigger.</p><p>In BF16 its weights alone are more than a whole ${DEF('8 × H100 node',node)} can hold.</p><p class="small">${bgT(r.total)} is ${A.fmt.num(r.total/1e9,1)}B rounded to 3 significant digits; the name rounds it down to 405B.</p>`,
          `<p>embedding 表和 LM head 只占整个模型的 ${vocShare(r)}：词表和初始设置一样大，但层多得多，每层也大得多。</p><p>BF16 下光权重就比一整台 ${DEF('8 × H100',node)} 的显存还多。</p><p class="small">${bgT(r.total)} 是 ${A.fmt.num(r.total/1e9,1)}B 取 3 位有效数字的结果；模型名字把它舍成了 405B。</p>`);}
  return h;
}

/* ---------- challenge ---------- */
const chPoint=l=>{const r=compute({...CH,L:l}); return {l,gib:r.gib,shown:+g1(r.gib).replace(/,/g,''),fits:r.gib<=CH_BUDGET+1e-12};};
function challengeHTML(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether this is the most layers that fit, and why.','提交后会告诉你这是不是能放下的最多层数，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Fit the most layers in ${CH_BUDGET} GiB of BF16 weights on one RTX 4090`,`在一张 RTX 4090 上，用 ${CH_BUDGET} GiB 的 BF16 权重放下最多的层`)}</h3>`+
    `<p>${tx(`The model is shaped like Llama-3.1-8B, with its weights in BF16. An RTX 4090 has 24 GB, read here as ${CH_GPU} ${GIB()}. The weights may take at most ${CH_BUDGET} GiB; the rest is kept for the ${DEF('KV cache',TIP.kvcache())} and the ${DEF('runtime overhead',TIP.overhead())}. You may change only the number of layers:`,
            `模型结构和 Llama-3.1-8B 一样，权重用 BF16。RTX 4090 有 24 GB，这里按 ${CH_GPU} ${GIB()} 计。权重最多占 ${CH_BUDGET} GiB，其余留给 ${DEF('KV cache',TIP.kvcache())} 和${DEF('运行时开销',TIP.overhead())}。你只能改层数：`)}</p>`+
    A.ui.range({id:'tf-ic-L',key:'L',label:tx('Layers','层数'),min:CH_MIN,max:CH_MAX,step:1,value:S.L,out:layersTxt(S.L),inline:true})+
    `<div class="goal mono" id="tf-ch-goal"><span>${tx('BF16 weights now','现在的 BF16 权重')}</span><span><b id="tf-ch-now"></b> / ${CH_BUDGET} GiB</span></div><div class="progress" id="tf-ch-prog" aria-hidden="true"><i></i></div>`+
    submit+`<div id="tf-ch-res" class="tf-ch-res" aria-live="polite"></div></div>`;
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.L);
  let h;
  /* R6.20: the pick and its size, then the reason in words; the division that gives the answer is in Show the math once the chart shows */
  if(me.l===CH_BEST) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${me.l} layers take ${gibS(me.gib)}, and one more would go over.</b> This is the most that fit.</p>`,
                          `<p class="okline" style="margin-top:12px"><b>达成目标：${me.l} 层占 ${gibS(me.gib)}，再加一层就超了。</b>这已经是能放下的最多层数。</p>`);
  else if(me.fits) h=tx(`<p class="okline" style="margin-top:12px"><b>It fits: ${me.l} layers take ${gibS(me.gib)}.</b> But ${CH_BEST-me.l} more ${CH_BEST-me.l===1?'layer':'layers'} would still fit.</p>`,
                        `<p class="okline" style="margin-top:12px"><b>放得下：${me.l} 层占 ${gibS(me.gib)}。</b>不过还能再加 ${CH_BEST-me.l} 层。</p>`);
  else {const r=compute(CH);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${me.l} layers take ${gibS(me.gib)}, over the ${CH_BUDGET} GiB budget.</b></p><p class="small">Hint: each extra layer adds ${BG(r.layer)} parameters, ${g3(r.layer*2/A.GiB)} GiB in BF16.</p>`,
         `<p class="badline" style="margin-top:12px"><b>还没达到：${me.l} 层要 ${gibS(me.gib)}，超过了 ${CH_BUDGET} GiB。</b></p><p class="small">提示：每加一层多 ${BG(r.layer)} 个参数，BF16 下是 ${g3(r.layer*2/A.GiB)} GiB。</p>`);
    if(!chReveal) h+=`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`;}
  if(!chReveal) return h;
  const ls=[...new Set(CH_ROWS.concat([S.L]))].sort((a,b)=>a-b), pts=ls.map(chPoint), mx=Math.max(28,...pts.map(p=>p.gib))*1.12;   // room for the value after the longest bar (R7.4)
  const rows=pts.map(p=>({label:layersTxt(p.l),value:p.gib,text:gibS(p.gib),ok:p.fits,me:p.l===S.L,
    tags:(p.l===CH.L?[{text:tx('initial setup','初始设置')}]:[]).concat(p.l===S.L?[{text:tx('your pick','你的选择'),me:true}]:[])}));
  /* the owner's review of batch 8 (2026-10-07): the chart keeps the layer counts that fit, the pick and the initial setup (brief);
     a note under it says how many it left out, all of them over the budget */
  const seen=pts.filter(p=>p.fits||p.l===S.L||p.l===CH.L);
  h+=A.ui.cmpChart({title:tx(`BF16 weights of a Llama-3.1-8B-shaped model (budget: ${CH_BUDGET} GiB on one RTX 4090)`,`Llama-3.1-8B 结构下 BF16 权重的大小（一张 RTX 4090 上的预算是 ${CH_BUDGET} GiB）`),
    aria:tx('BF16 weights of a Llama-3.1-8B-shaped model. ','Llama-3.1-8B 结构下 BF16 权重的大小：')+seen.map(p=>`${layersTxt(p.l)}${tx(': ','：')}${gibS(p.gib)}`).join(tx('; ','；')),
    rows,brief:true,hiddenText:n=>tx(n===1?'The other layer count does not fit and is not shown.':`The other ${n} layer counts do not fit and are not shown.`,`其余 ${n} 种层数放不下，图里没有列出。`),
    goal:CH_BUDGET,goalLabel:tx(`budget: ${CH_BUDGET} GiB`,`预算 ${CH_BUDGET} GiB`),max:mx,
    okText:tx(`fits in ${CH_BUDGET} GiB`,`${CH_BUDGET} GiB 以内，放得下`),noText:tx(`over ${CH_BUDGET} GiB`,`超过 ${CH_BUDGET} GiB`),goalText:tx(`budget: ${CH_BUDGET} GiB`,`预算：${CH_BUDGET} GiB`)});
  /* initial setup → effect → reason (R3.10), in words: the sizes are on the chart, the division in Show the math (R6.20) */
  h+=tx(`<p class="small">Initial setup: ${CH.L} layers. The embedding table, LM head and final RMSNorm take the same memory at any depth, and each layer adds the same amount, so ${CH_BEST} layers fit. ${J('math',A.t('showMath'))} works it out.</p>`,
        `<p class="small">初始设置是 ${CH.L} 层。embedding 表、LM head 和最后的 RMSNorm 不管多少层都占同样多的显存，每加一层多占的也都一样，所以放得下 ${CH_BEST} 层。算式在${J('math','“'+A.t('showMath')+'”')}里。</p>`);
  h+=tx(`<p class="small muted">Note: this challenge counts only the weights. More layers mean training a new model, and more layers alone do not make it better. The memory left must also hold the KV cache, which grows with every layer too. The lab ${LK('kv-basics')} explains it, and the lab ${LK('kv-cache')} works out how many users fit.</p>`,
        `<p class="small muted">注意：这道题只算权重。加层意味着重新训练一个模型，层数多本身也不等于模型更好。剩下的显存还要放 KV cache，而它也随层数增加。KV cache 是什么，见${LK('kv-basics')}；能装下几个用户，在${LK('kv-cache')}里算。</p>`);
  return h;
}

/* ---------- stage ---------- */
const MASKED=()=>mode==='guided'&&!revealed&&step<PART.split;
function legendHTML(r){
  const it=(cls,txt,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${cls}"></i><span>${txt}</span></span>`;
  const m=MASKED();
  let h=it('tf-sw-emb',tx('embedding table','embedding 表'),TIP.emb());
  if(m) h+=it('tf-sw-layer',tx('layer (split in part 4)','一层（第 4 部分拆开）'),tx('A whole layer: its attention block and MLP block together. How they split is shown in part 4.','一整层：attention 块和 MLP 块合在一起。两者各占多少，第 4 部分揭晓。'));
  else h+=it('tf-sw-attn',tx('attention block','attention 块'),TIP.attnBlock())+it('tf-sw-mlp',tx('MLP block','MLP 块'),TIP.mlpBlock());
  h+=it('tf-sw-norm','RMSNorm',TIP.rms());
  h+=r.s.tied?it('tf-sw-tied',tx('LM head = embedding table (tied)','LM head = embedding 表（共享参数）'),A.terms.lmhead()+tx(' Here the two are tied, so the LM head is drawn as a dashed outline: its parameters are counted once, in the embedding table.','这里两者共享参数，所以 LM head 画成虚线框：这些参数只在 embedding 表里算一次。'))
             :it('tf-sw-head','LM head',TIP.head());
  h+=it('tf-sw-rs',tx('residual connection','残差连接'),TIP.residual()+tx(' The gray line on the left is the vector on its way down from the embedding table to the LM head; in every layer, each block’s output is added back to it at a ⊕.','左边的灰线就是从 embedding 表往下走到 LM head 的那个向量；每一层里，每个块的输出都在一个 ⊕ 处加回它上面。'));
  if(mode==='guided'&&step===PART.walk) h+=it('tf-sw-tok',tx('the token “ is”','token “ is”'),tx('The dot marks where the token “ is” is now on its way down.','圆点标出 token “ is” 现在走到了哪里。'))+
    it('tf-sw-on',tx('outline: where “ is” is now','粗框：“ is” 现在所在的位置'),tx('A thick outline marks the block or step where the token “ is” is now, the same place as the dot.','粗框标出 token “ is” 现在所在的块或那一步，和圆点是同一个位置。'));
  return h;
}
const W=(n,sc)=>Math.max(0,n/sc*100);
/* every section uses the same three columns (layer number | bar | brace), so all bars share one scale: width = parameters */
function stackHTML(r){
  const s=r.s, m=MASKED(), sc=scaleOf(r), wk=mode==='guided'&&step===PART.walk?WALK[walk]:null, on=k=>wk===k?' tf-on':'';
  const lab=(name,shape,n)=>`<div class="tf-lab"><b>${name}</b>${shape?`<span class="tf-shape"> ${shape} =</span>`:''} <span class="mono tf-n">${n}</span></div>`;   // n: a count, BG() over 6 digits (R6.18)
  const llama=s.V===DEFAULT.V&&wk;
  let h=`<div class="tf-io tf-in${on('id')}" data-stop="id">${llama?`“ is” → token ID ${TOK.id}`:tx(`token ID (one of ${fmtInt(s.V)})`,`token ID（${fmtInt(s.V)} 个之一）`)}</div>`;
  h+=`<div class="tf-sec" data-stop="emb">${lab(DEF(tx('embedding table','embedding 表'),TIP.emb()),`${fmtInt(s.V)} × ${fmtInt(s.d)}`,BG(r.emb))}<div class="tf-bar"><i class="tf-seg tf-pat-emb${on('emb')}" style="width:${W(r.emb,sc).toFixed(3)}%"></i></div></div>`;
  /* the layers: at most six rows drawn (1, 2, 3, ⋮, L−1, L) and a "×L" brace; a row is one layer, its width the layer's parameters */
  const rows=s.L<=6?Array.from({length:s.L},(_,i)=>i+1):[1,2,3,0,s.L-1,s.L];
  const lw=W(r.layer,sc), ai=r.attn/r.layer*100, mi=r.mlp/r.layer*100;
  const seg=l=>m?`<i class="tf-seg tf-pat-layer" style="width:100%"></i>`
                :`<i class="tf-seg tf-pat-attn${l===1&&wk==='attn'?' tf-on':''}" style="width:${ai.toFixed(3)}%"></i><i class="tf-seg tf-pat-mlp${l===1&&wk==='mlp'?' tf-on':''}" style="width:${mi.toFixed(3)}%"></i>`;
  const rowOn=l=>(wk==='rest'&&l!==1)||(m&&l===1&&(wk==='attn'||wk==='mlp'));
  const lrows=rows.map(l=>l===0?`<span class="tf-ln">⋮</span><div class="tf-gap"></div>`
    :`<span class="tf-ln mono">${l}</span><div class="tf-bar tf-lbar" data-l="${l}"><span class="tf-lw${rowOn(l)?' tf-on':''}" style="width:${lw.toFixed(3)}%">${seg(l)}</span></div>`).join('');
  /* part 3: the order inside a layer, not to scale (so it gives nothing away), with the step “ is” is at outlined (R1.9) */
  const fl=(k,txt,tip)=>{const hit=(wk==='attn'&&/1$/.test(k))||(wk==='mlp'&&/2$/.test(k))||wk==='rest';
    return `<span class="tf-fl${hit?' tf-on':''}"${tip?` tabindex="0" data-tipx="${esc(tip)}"`:''}>${txt}</span>`;};
  const plus=tx('⊕, the residual connection, adds the block’s output back to the vector from before the block’s RMSNorm. It has no parameters.','⊕ 是残差连接：把这个块的输出加回到这个块的 RMSNorm 之前的向量上。它没有参数。');
  const each=wk?`${tx('each layer: ','每层：')}${fl('n1','RMSNorm')} → ${fl('a1',tx('attention block','attention 块'))} → ${fl('p1','⊕',plus)} → ${fl('n2','RMSNorm')} → ${fl('m2',tx('MLP block','MLP 块'))} → ${fl('p2','⊕',plus)}`
            :m?tx(`each: attention block + MLP block + 2 RMSNorms`,`每层：attention 块 + MLP 块 + 2 个 RMSNorm`)
              :tx(`each: attention ${BG(r.attn)} + MLP ${BG(r.mlp)} + 2 RMSNorms ${fmtInt(r.lnorm)}`,`每层：attention ${BG(r.attn)} + MLP ${BG(r.mlp)} + 2 个 RMSNorm ${fmtInt(r.lnorm)}`);
  h+=`<div class="tf-sec tf-layers" data-stop="layers"><div class="tf-lab"><b>${DEF(layersTxt(s.L),TIP.layerAny())}</b> <span class="mono tf-n">× ${BG(r.layer)} = ${BG(r.layers)}</span></div><div class="tf-lab tf-each mono${wk?' tf-flow':''}">${each}</div>`+
     lrows+`<div class="tf-brace" style="grid-row:3 / span ${rows.length}" aria-hidden="true"><span class="mono">×${fmtInt(s.L)}</span></div></div>`;
  h+=`<div class="tf-sec tf-normrow" data-stop="norm">${lab(tx(`final ${DEF('RMSNorm',TIP.rms())}`,`最后的 ${DEF('RMSNorm',TIP.rms())}`),'',fmtInt(r.fnorm))}<div class="tf-bar tf-thin"><i class="tf-seg tf-pat-norm${on('norm')}" style="width:${W(r.fnorm,sc).toFixed(4)}%"></i></div></div>`;
  h+=s.tied?`<div class="tf-sec" data-stop="head"><div class="tf-lab"><b>${DEF('LM head',TIP.head())}</b> <span class="tf-n">${tx('tied: the embedding table’s parameters, none added','共享参数：用 embedding 表的参数，不另加')}</span></div><div class="tf-bar"><i class="tf-seg tf-pat-tied${on('head')}" style="width:${W(r.emb,sc).toFixed(3)}%"></i></div></div>`
           :`<div class="tf-sec" data-stop="head">${lab(DEF('LM head',TIP.head()),`${fmtInt(s.d)} × ${fmtInt(s.V)}`,BG(r.head))}<div class="tf-bar"><i class="tf-seg tf-pat-head${on('head')}" style="width:${W(r.head,sc).toFixed(3)}%"></i></div></div>`;
  h+=`<div class="tf-io tf-out" data-stop="out">${wk==='head'&&llama?tx(`${fmtInt(s.V)} logits, one per vocabulary token; “ Paris” is token ID ${TOK.next}`,`${fmtInt(s.V)} 个 logit，词表里每个 token 一个；“ Paris” 的 token ID 是 ${TOK.next}`):tx(`${fmtInt(s.V)} ${DEF('logits',TIP.logit())}, one per vocabulary token`,`${fmtInt(s.V)} 个 ${DEF('logit',TIP.logit())}，词表里每个 token 一个`)}</div>`;
  return h+`<i class="tf-rs" aria-hidden="true"></i><i class="tf-tok" aria-hidden="true" hidden></i>`;
}
/* the residual connection's line runs down the left edge from the embedding table to the LM head; the token marker sits on it in part 3 */
function placeStream(){
  const st=$('#tf-stack',app); if(!st) return;
  const e=$('[data-stop="emb"] .tf-bar',st), hd=$('[data-stop="head"] .tf-bar',st), rs=$('.tf-rs',st), tk=$('.tf-tok',st); if(!e||!hd) return;
  const y0=st.getBoundingClientRect().top, mid=el=>{const b=el.getBoundingClientRect(); return b.top-y0+b.height/2;};
  const top=mid(e), bot=mid(hd), x0=st.getBoundingClientRect().left;
  rs.style.top=top+'px'; rs.style.height=Math.max(0,bot-top)+'px'; rs.style.width=Math.max(8,e.getBoundingClientRect().left-x0-5)+'px';
  const wk=mode==='guided'&&step===PART.walk?WALK[walk]:null;
  if(!wk){tk.hidden=true; return;}
  let el=null;
  const lbars=$$('.tf-lbar[data-l]',st);
  if(wk==='id') el=$('[data-stop="id"]',st); else if(wk==='emb') el=e; else if(wk==='attn'||wk==='mlp') el=lbars[0];
  else if(wk==='rest') el=lbars[lbars.length-1]; else if(wk==='norm') el=$('[data-stop="norm"] .tf-bar',st); else el=hd;
  if(!el){tk.hidden=true; return;}
  tk.hidden=false; tk.style.top=mid(el)+'px';
}
function shareHTML(r){
  if(MASKED()) return `<div class="tf-share-t">${tx(`Share of the ${BG(r.total)} parameters`,`${BG(r.total)} 个参数的占比`)}</div><div class="tf-share-mask">${pred||step>PART.predict?tx('The shares are shown in part 4.','占比在第 4 部分揭晓。'):tx('The shares are shown in part 4, after you predict in part 2.','占比在第 4 部分揭晓，先在第 2 部分预测。')}</div>`;
  const P=r.parts, t=r.total, segs=[['mlp','tf-pat-mlp',tx('MLP blocks','MLP 块')],['attn','tf-pat-attn',tx('attention blocks','attention 块')],['emb','tf-pat-emb',r.s.tied?tx('embedding table (tied to the LM head)','embedding 表（和 LM head 共享参数）'):tx('embedding table','embedding 表')],['head','tf-pat-head','LM head'],['norm','tf-pat-norm',tx('RMSNorms','RMSNorm')]].filter(x=>P[x[0]]>0);
  const bar=segs.map(([k,cls])=>`<i class="tf-seg ${cls}" data-k="${k}" style="width:${(P[k]/t*100).toFixed(4)}%"><span class="mono">${pctS(P[k]/t)}</span></i>`).join('');
  /* the key is a legend too: each entry has the part's definition on hover, so MLP and LM are written out there (common r15 (h)) */
  const ktip={mlp:TIP.mlpBlock,attn:TIP.attnBlock,emb:()=>r.s.tied?TIP.emb()+tx(' ','')+TIP.head():TIP.emb(),head:TIP.head,norm:TIP.rms};
  const key=segs.map(([k,cls,name])=>`<span tabindex="0" data-tipx="${esc(ktip[k]())}"><i class="sw ${cls.replace('tf-pat-','tf-sw-')}"></i>${name} <b class="mono">${pctS(P[k]/t)}</b></span>`).join('');
  return `<div class="tf-share-t">${tx(`Share of the ${BG(t)} parameters`,`${BG(t)} 个参数的占比`)}</div><div class="tf-sbar">${bar}</div><div class="tf-skey">${key}</div>`;
}
/* a label inside a share segment only when it fits (R7.4); the key under the bar always has every value */
function fitShareLabels(){
  $$('#tf-share .tf-sbar .tf-seg',app).forEach(sg=>{const sp=sg.firstElementChild; if(!sp) return; sp.style.visibility='hidden'; sp.style.display='';
    const ok=sp.scrollWidth+8<=sg.clientWidth; sp.style.visibility=''; if(!ok) sp.style.display='none';});
}
const MET=[
 /* R6.18: the cards show 8.03B; the tooltip says what is counted, with no exact count (the owner's clarification, 2026-10-07) */
 {k:'total',tip:r=>TIP.param()+tx(' Counted here: the embedding table, the layers’ matrices, the RMSNorms’ scales and the LM head.','这里数进去的有 embedding 表、各层的矩阵、RMSNorm 的系数和 LM head。'),
  label:()=>tx('Parameters','参数量'),v:r=>bgT(r.total),sub:()=>tx('counted from config.json','按 config.json 数出')},
 {k:'mem',tip:r=>tx(`The weights’ size in GPU memory: ${MX('\\text{parameters} \\times \\text{bytes per parameter}')}, here ${fmtInt(r.bytes)} bytes = ${memEq(r)}. 1 GiB is ${MX('2^{30}')} bytes; 1 GB is ${MX('10^{9}')} bytes.`,`权重在显存里的大小等于${MX('\\text{参数量} \\times \\text{每个参数的字节数}')}，这里是 ${fmtInt(r.bytes)} 字节 = ${memEq(r)}。1 GiB 是 ${MX('2^{30}')} 字节，1 GB 是 ${MX('10^{9}')} 字节。`),
  label:r=>tx(`Weights in ${PL[r.s.p]}`,`${PL[r.s.p]} 权重`),v:r=>gibS(r.gib),sub:r=>tx(`${gbS(r.gb)}, ${bytesTxt(r.s.p)} per parameter`,`${gbS(r.gb)}，每个参数 ${bytesTxt(r.s.p)}`)},
 {k:'mlp',share:true,tip:()=>tx('The share of all parameters in the MLP blocks (MLP: multi-layer perceptron). Each has three matrices: gate, up and down.','所有参数里 MLP 块（MLP 即多层感知机）占的比例。每个 MLP 块有三个矩阵：gate、up、down。'),
  label:()=>tx('MLP blocks','MLP 块'),v:r=>pctS(r.parts.mlp/r.total),sub:r=>tx(`${bgT(r.parts.mlp)} in ${fmtInt(r.s.L)} blocks`,`${fmtInt(r.s.L)} 个块共 ${bgT(r.parts.mlp)}`)},
 {k:'attn',share:true,tip:()=>tx('The share of all parameters in the attention blocks: four matrices per layer (Wq, Wk, Wv and Wo).','所有参数里 attention 块占的比例：每层四个矩阵（Wq、Wk、Wv、Wo）。'),
  label:()=>tx('Attention blocks','attention 块'),v:r=>pctS(r.parts.attn/r.total),sub:r=>tx(`${bgT(r.parts.attn)} in ${fmtInt(r.s.L)} blocks`,`${fmtInt(r.s.L)} 个块共 ${bgT(r.parts.attn)}`)},
 {k:'voc',share:true,tip:r=>r.s.tied?tx('The share in the embedding table. The LM head (LM: language model) is tied to it and reuses its parameters, so they are counted once.','embedding 表占的比例。LM head（LM 即语言模型）和它共享参数，直接用它的参数，所以只算一次。')
                                 :tx('The share in the two tables that hold one vector per vocabulary token, the embedding table at the top and the LM head (LM: language model) at the bottom.','两张给词表里每个 token 各存一个向量的表占的比例：顶部的 embedding 表和底部的 LM head（LM 即语言模型）。'),
  label:r=>r.s.tied?tx('Embedding table (tied to the LM head)','embedding 表（和 LM head 共享参数）'):tx('Embedding table and LM head','embedding 表和 LM head'),v:r=>vocShare(r),
  sub:r=>r.s.tied?tx(`1 table of ${bgT(r.emb)}`,`一张表 ${bgT(r.emb)}`):tx(`2 tables of ${bgT(r.emb)}`,`两张表各 ${bgT(r.emb)}`)}
];
const decOf=v=>{const m=String(v).match(/\.(\d+)/); return m?m[1].length:0;};
/* a share's change in points, from the two values as shown, with the decimals they are shown with ("5.0 pts", R6.6) */
function ptsDelta(a,b,base){
  const dp=Math.max(decOf(a),decOf(b)), k=10**dp, x=Math.round(parseFloat(a)*k)/k, y=Math.round(parseFloat(b)*k)/k, dd=y-x;
  if(!isFinite(dd)||Math.abs(dd)<0.5/k) return '';
  const n=Math.abs(dd).toFixed(dp);
  return (dd>0?'▲ ':'▼ ')+n+tx(n==='1'?' pt':' pts',' 个百分点')+tx(` vs ${base.en}`,`（比${base.zh}）`);
}
function drawMetrics(r){
  const host=$('#tf-metrics',app), hid=MASKED();
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  /* ▲▼ only after the learner's own change (R6.6). Parts 5 and 6 compare with the initial setup, as their cards do (R6.1); elsewhere with the last value.
     The parameter count and the weights change by the same factor when the bytes per parameter stay, so both come from the exact numbers,
     which the card's tooltip then shows; a share's change comes from the shares as shown. */
  const init=mode==='guided'&&(step===PART.vocab||step===PART.config), b0=init?compute(DEFAULT):null;
  const base=init?{en:'the initial setup',zh:'初始设置'}:{en:'last',zh:'上次'}, opt=init?{base}:{};
  const exact={total:x=>x.total,mem:x=>x.bytes};
  MET.forEach(m=>{const el=$(`[data-k="${m.k}"]`,host), h=hid&&m.share, v=h?'?':m.v(r), d=$('.d',el);
    const tip=h?(pred?tx('This share stays hidden until part 4, where it answers your prediction.','这个占比先隐藏到第 4 部分，它就是你那道预测题的答案。')
                   :tx('This share is hidden until you answer part 2; part 4 shows it.','这个占比在第 2 部分作答前先隐藏，第 4 部分揭晓。')):m.tip(r);
    $('.k',el).textContent=m.label(r); $('.v',el).textContent=v; $('.sub',el).textContent=h?tx('shown in part 4','第 4 部分揭晓'):m.sub(r);
    let dt='', how='';
    if(userAct&&!h){
      if(exact[m.k]){const a=init?exact[m.k](b0):shownPrev['x_'+m.k], b=exact[m.k](r);
        if(a!=null&&a!==b){dt=A.fmt.delta(a,b,opt);
          const was=init?tx('in the initial setup','初始设置'):tx('before','之前');
          if(dt) how=m.k==='total'?tx(` “${dt}” compares the exact counts: ${fmtInt(b)} now, ${fmtInt(a)} ${was}.`,`“${dt}”比的是精确的参数量：现在 ${fmtInt(b)} 个，${was} ${fmtInt(a)} 个。`)
                                 :tx(` “${dt}” compares the exact sizes: ${fmtInt(b)} bytes now, ${fmtInt(a)} bytes ${was}.`,`“${dt}”比的是精确大小：现在 ${fmtInt(b)} 字节，${was} ${fmtInt(a)} 字节。`);}}
      else {const a=init?m.v(b0):shownPrev[m.k]; if(a!=null&&a!=='?'&&a!==v) dt=ptsDelta(a,v,base);}}
    el.dataset.tipx=tip+how;
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent=''; el.dataset.tipx=tip;},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=h?null:v; if(exact[m.k]) shownPrev['x_'+m.k]=exact[m.k](r);});
}
/* the owner's decisions of 2026-10-09 (H1, H2): Show the math is a few short blocks. Each says in words what it computes, gives the
   quantity's name and its formula typeset (A.ui.math), then the same line with the current numbers (its terms wired to Settings) and the
   result; a long chain breaks at “=” onto lines that line up (.tf-chain). The number lines stay <p> so the numbers test reads them. */
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
  const piece=x=>`<span class="tf-fp">${MX('\\mathrm{} '+x)}</span>`;
  return `<span class="tf-fwn">${MX(name+' '+rel+' \\mathrm{}')}</span><span class="tf-fwr">${FWcut(rhs,[' + ',' - ']).map(g=>`<span class="tf-fg">${FWcut(g,[' \\times ']).map(piece).join('<wbr>')}</span>`).join('<wbr>')}</span>`;};
const FL=f=>{const h=FW(f); return `<div class="${h.includes('tf-fwn')?'tf-fw ':''}tf-math-line">${h}</div>`;};
const MB=(lead,f,lines)=>`<div class="tf-mb"><div class="tf-mlead">${lead}</div>${FL(f)}${lines}</div>`;
const CHN=(name,rows,more=[])=>`<p class="tf-chain">${[[name,rows[0]],...rows.slice(1).map(x=>['',x]),...more].map(([n,x])=>`<span class="tf-cn">${n}</span><span class="tf-cr">${x}</span>`).join('')}</p>`;
const HD_TEX=()=>tx('\\text{head_dim} = \\text{hidden size} \\div \\text{heads}','\\text{head_dim} = \\text{隐藏维度} \\div \\text{head 数}');
function mathHTML(r){
  const s=r.s, d=fmtInt(s.d), q=fmtInt(r.qw), kv=fmtInt(r.kvw), hd=fmtInt(r.hd), p=[];
  const layerF=tx('\\text{one layer} = \\text{attention block} + \\text{MLP block} + 2 \\times \\text{hidden size}','\\text{一层} = \\text{attention 块} + \\text{MLP 块} + 2 \\times \\text{隐藏维度}');
  const layerL=tx('One layer is these two blocks plus two RMSNorms. Each RMSNorm is one vector of scales, as long as the hidden size.','一层就是这两个块，再加两个 RMSNorm。每个 RMSNorm 只有一个系数向量，长度等于隐藏维度。');
  if(MASKED()){   // the split between the two blocks answers part 2: hidden until part 4 (R3.8)
    p.push(MB(tx('One layer is an attention block, an MLP block and two RMSNorms. Each RMSNorm is one vector of scales, as long as the hidden size.','一层就是一个 attention 块、一个 MLP 块和两个 RMSNorm。每个 RMSNorm 只有一个系数向量，长度等于隐藏维度。'),layerF,
      `<p>${tx('one layer','一层')} = ${tx('attention block + MLP block + 2 RMSNorms','attention 块 + MLP 块 + 2 个 RMSNorm')} = <b>${fmtInt(r.layer)}</b></p><p class="small muted">${tx('How the layer splits between its two blocks is shown in part 4.','这一层在两个块之间怎么分，第 4 部分揭晓。')}</p>`));}
  else {
  p.push(MB(tx('The hidden size is shared out evenly among the heads. Each head’s share of every query, key and value is head_dim.','隐藏维度平均分给所有 head。每个 head 在 query、key、value 里分到的维数，就是 head_dim。'),
    tx('\\text{head_dim} = \\frac{\\text{hidden size}}{\\text{heads}}','\\text{head_dim} = \\frac{\\text{隐藏维度}}{\\text{head 数}}'),
    `<p>head_dim = ${T_('d',tx(`${d} (hidden size)`,`${d}（隐藏维度）`))} ÷ ${T_('H',tx(`${s.H} heads`,`${s.H} 个 head`))} = <b>${hd}</b></p>`));
  p.push(MB(tx('The attention block’s parameters are those of its four matrices, and a matrix holds its rows times its columns. Wk and Wv compute keys and values only for the KV heads, so they are narrower than Wq and Wo.','attention 块的参数，就是它四个矩阵的参数。一个矩阵的参数量等于行数乘以列数。Wk 和 Wv 只为 KV head 算 key 和 value，所以比 Wq 和 Wo 窄。'),
    tx('\\text{attention block} = \\text{Wq} + \\text{Wk} + \\text{Wv} + \\text{Wo}','\\text{attention 块} = \\text{Wq} + \\text{Wk} + \\text{Wv} + \\text{Wo}'),
    `<p>${tx('attention block','attention 块')} = Wq ${d} × ${q} + Wk ${d} × ${kv} + Wv ${d} × ${kv} + Wo ${q} × ${d} = <b>${fmtInt(r.attn)}</b>${tx(` (${kv} = `,`（${kv} = `)}${T_('KV',tx(`${s.KV} KV heads`,`${s.KV} 个 KV head`))} × ${hd}${tx(')','）')}</p>`));
  p.push(MB(tx('The MLP block has three matrices, gate, up and down, each of them hidden size by intermediate size.','MLP 块有 gate、up、down 三个矩阵。每个矩阵的行和列，一边是隐藏维度，一边是中间维度。'),
    tx('\\text{MLP block} = 3 \\times \\text{hidden size} \\times \\text{intermediate size}','\\text{MLP 块} = 3 \\times \\text{隐藏维度} \\times \\text{中间维度}'),
    `<p>${tx('MLP block','MLP 块')} = 3 × ${d} × ${T_('f',fmtInt(s.f))} = <b>${fmtInt(r.mlp)}</b></p>`));
  p.push(MB(layerL,layerF,
    `<p>${tx('2 RMSNorms','2 个 RMSNorm')} = 2 × ${d} = <b>${fmtInt(r.lnorm)}</b></p><p>${tx('one layer','一层')} = ${fmtInt(r.attn)} + ${fmtInt(r.mlp)} + ${fmtInt(r.lnorm)} = <b>${fmtInt(r.layer)}</b></p>`));}
  const head=s.tied?`${T_('tied',tx('0 (LM head tied)','0（LM head 共享参数）'))}`:`${T_('tied',`${d} × ${fmtInt(s.V)}`)}${tx(' (LM head)','（LM head）')}`;
  p.push(MB(tx('The whole model adds up the embedding table, every layer, the final RMSNorm and the LM head.','整个模型的参数，是 embedding 表、所有层、最后的 RMSNorm 和 LM head 加起来。'),
    tx('\\text{model} = \\text{vocabulary} \\times \\text{hidden size} + \\text{layers} \\times \\text{one layer} + \\text{hidden size} + \\text{LM head}','\\text{模型} = \\text{词表大小} \\times \\text{隐藏维度} + \\text{层数} \\times \\text{一层} + \\text{隐藏维度} + \\text{LM head}'),
    CHN(tx('model','模型'),[`= ${T_('V',fmtInt(s.V))} × ${d}${tx(' (embedding table)','（embedding 表）')} + ${T_('L',layersTxt(s.L))} × ${fmtInt(r.layer)} + ${d}${tx(' (final RMSNorm)','（最后的 RMSNorm）')} + ${head}`,`= <b>${fmtInt(r.total)}</b>`])));
  /* part 5: what the vocabulary costs, the line behind the result box's words (R6.20) */
  const pk=presetOf(s);
  if(mode==='guided'&&step===PART.vocab&&(pk==='mistral-7b'||pk==='llama31-8b')){const a=compute({...DEFAULT,p:s.p}), m=compute({...MISTRAL(),p:s.p});
    const vd=(x,name)=>DEF(fmtInt(x.s.V),tx(`${name}’s vocabulary size, vocab_size in its config.json.`,`${name} 的词表大小，即它 config.json 里的 vocab_size。`));
    p.push(MB(tx('The two models differ only in their vocabularies. Each extra vocabulary token adds one vector as long as the hidden size to the embedding table, and one to the LM head.','这两个模型只差在词表上。词表里每多一个 token，embedding 表和 LM head 就各多一个向量，长度等于隐藏维度。'),
      tx('\\text{difference} = (\\text{vocabulary}_{\\text{Llama}} - \\text{vocabulary}_{\\text{Mistral}}) \\times \\text{hidden size} \\times 2','\\text{相差的参数} = (\\text{词表大小}_{\\text{Llama}} - \\text{词表大小}_{\\text{Mistral}}) \\times \\text{隐藏维度} \\times 2'),
      `<p>Llama-3.1-8B − Mistral-7B-v0.1 = (${vd(a,'Llama-3.1-8B')} − ${vd(m,'Mistral-7B-v0.1')}) × ${T_('d',d)} × 2${tx(' (embedding table and LM head)','（embedding 表和 LM head）')} = <b>${fmtInt(a.total-m.total)}</b></p>`));}
  /* R6.12: bytes → GB → GiB, so 15.0 does not read as a slip for 16 */
  p.push(MB(tx('The weights’ memory is the parameter count times the bytes each parameter takes.','权重占的显存，是参数量乘以每个参数占的字节数。'),
    tx('\\text{weights} = \\text{parameters} \\times \\text{bytes per parameter}','\\text{权重} = \\text{参数量} \\times \\text{每个参数的字节数}'),
    CHN(tx('weights','权重'),[`= ${fmtInt(r.total)} × ${T_('p',bytesTxt(s.p))}`,`= ${fmtInt(r.bytes)} ${tx('bytes','字节')}`,`= <b>${memEq(r,true)}</b>`])));
  /* the challenge, once its chart shows (R3.8): the division that the verdict's reason says in words (R6.20); each term's derivation is on
     its hover (R1.4), and the terms are rounded as shown, so the line reproduces from them (R6.6) */
  if(mode==='challenge'&&chReveal){const r0=compute(CH), by=PREC[CH.p], fixN=r0.emb+r0.head+r0.fnorm, fixedG=+g3(fixN*by/A.GiB), perG=+g3(r0.layer*by/A.GiB), qq=(CH_BUDGET-fixedG)/perG;
    const eq=(n,g)=>tx(`${fmtInt(n)} parameters × ${bytesTxt(CH.p)} = ${fmtInt(n*by)} bytes = ${g3(n*by/1e9)} GB = ${g3(g)} GiB.`,`${fmtInt(n)} 个参数 × ${bytesTxt(CH.p)} = ${fmtInt(n*by)} 字节 = ${g3(n*by/1e9)} GB = ${g3(g)} GiB。`);
    const tBud=DEF(String(CH_BUDGET),tx(`The budget: at most ${CH_BUDGET} GiB of BF16 weights on one RTX 4090; the rest of its memory is kept for the KV cache and the runtime overhead.`,`预算：一张 RTX 4090 上最多放 ${CH_BUDGET} GiB 的 BF16 权重，其余显存留给 KV cache 和运行时开销。`));
    const tFix=DEF(g3(fixedG),tx('The embedding table, the LM head and the final RMSNorm in BF16, the same at any depth: ','embedding 表、LM head 和最后的 RMSNorm 在 BF16 下占的显存，不管多少层都一样：')+eq(fixN,fixedG));
    const tPer=DEF(g3(perG),tx('One layer in BF16: ','一层在 BF16 下：')+eq(r0.layer,perG));
    p.push(MB(tx('The budget first pays for what every depth needs: the two tables and the final RMSNorm. What is left, divided by one layer’s memory, is how many layers fit.','预算先要放下不管多少层都有的部分：两张表和最后的 RMSNorm。剩下的显存除以一层占的显存，就是放得下的层数。'),
      tx('\\text{layers in the budget} = \\frac{\\text{budget} - \\text{tables and final RMSNorm}}{\\text{one layer}}','\\text{预算里放得下的层数} = \\frac{\\text{预算} - \\text{两张表和最后的 RMSNorm}}{\\text{一层}}'),
      `<p>${tx('layers in the budget','预算里放得下的层数')} = (${tBud} − ${tFix}) ÷ ${tPer} = <b>${A.fmt.num(Math.floor(qq*10)/10,1)}</b>${tx(`, so ${CH_BEST} layers fit`,`，所以放得下 ${CH_BEST} 层`)}</p>`));}
  p.push(s.tied?MB(tx('A token’s vector is multiplied by each parameter, 2 FLOPs per parameter: a multiply and an add. The shared table is multiplied as the LM head, so every parameter counts.','每个 token 的向量要和参数逐个相乘，每个参数算 2 次 FLOP：一次乘法，一次加法。embedding 表和 LM head 共享参数时，这张表作为 LM head 也要参与乘法，所以全部参数都算。'),
                   tx('\\text{FLOPs per token} \\approx 2 \\times \\text{parameters}','\\text{每个 token 的 FLOP} \\approx 2 \\times \\text{参数量}'),
                   CHN(tx('multiplied per token','每个 token 参与乘法的参数'),[tx(`= all ${fmtInt(r.total)}`,`= 全部 ${fmtInt(r.total)} 个`)],[[tx('FLOPs per token','每个 token 的 FLOP'),`≈ 2 × ${fmtInt(r.total)} ≈ <b>${bgT(r.flops)}</b>`]]))
               :MB(tx('A token’s vector is multiplied by each parameter, 2 FLOPs per parameter: a multiply and an add. The embedding table is only looked up, so it is left out.','每个 token 的向量要和参数逐个相乘，每个参数算 2 次 FLOP：一次乘法，一次加法。embedding 表只是查表，不算在内。'),
                   tx('\\text{FLOPs per token} \\approx 2 \\times (\\text{parameters} - \\text{embedding table})','\\text{每个 token 的 FLOP} \\approx 2 \\times (\\text{参数量} - \\text{embedding 表})'),
                   CHN(tx('multiplied per token','每个 token 参与乘法的参数'),[`= ${fmtInt(r.total)} − ${fmtInt(r.emb)}${tx(' (the embedding table is a lookup)','（embedding 表只是查表）')}`,`= ${fmtInt(r.mult)}`],[[tx('FLOPs per token','每个 token 的 FLOP'),`≈ 2 × ${fmtInt(r.mult)} ≈ <b>${bgT(r.flops)}</b>`]])));
  p.push(`<p class="small muted tf-mnote">${tx('Counted as in Meta’s reference code and transformers’ modeling_llama.py: no biases, and head_dim as in the first formula. The FLOPs leave out attention over the context, which grows with its length (Kaplan et al., 2020, Table 1).','按 Meta 的参考实现和 transformers 的 modeling_llama.py 计算：没有 bias，head_dim 按第一个公式算。FLOP 没算上下文上的 attention，它随上下文长度增长（Kaplan 等，2020，表 1）。')}</p>`);
  return zop(p.join(''));
}
/* the caption says what a full-length bar is, and when this model needs its own scale */
function capTxt(r){
  const sc=scaleOf(r), own=sc>SCALE;
  return own?tx(`Top to bottom in the order a token passes through. Each bar’s length is proportional to its parameter count. This model does not fit the usual scale, so here a full-length bar is ${BG(sc)} parameters instead of ${BG(SCALE)}.`,
                `从上往下就是 token 经过的顺序。每条的长度和它的参数量成正比。这个模型超出了平常的尺子，所以这里满格是 ${BG(sc)} 个参数，而不是 ${BG(SCALE)} 个。`)
            :tx(`Top to bottom in the order a token passes through. Each bar’s length is proportional to its parameter count, on one scale for every model: a full-length bar is ${BG(SCALE)} parameters.`,
                `从上往下就是 token 经过的顺序。每条的长度和它的参数量成正比，所有模型用同一把尺子：满格是 ${BG(SCALE)} 个参数。`);}
function drawStage(r){
  const s=r.s, st=$('#tf-status',app);
  $('#tf-st-title',app).textContent=NB(tx(`${configName(s)}, weights in ${PL[s.p]}`,`${configName(s)}，权重 ${PL[s.p]}`));
  $('#tf-st-sub',app).innerHTML=tx(`Where ${BG(r.total)} parameters live`,`${BG(r.total)} 个参数都在哪`);
  if(mode==='challenge'){const me=chPoint(s.L), sp=+(CH_BUDGET-me.shown).toFixed(1);
    st.hidden=false; st.className='status '+(me.fits?'ok':'bad'); st.dataset.tipx=tx(`Green means the BF16 weights fit in the ${CH_BUDGET} GiB budget on one RTX 4090; red means they do not.`,`绿色表示 BF16 权重在一张 RTX 4090 的 ${CH_BUDGET} GiB 预算以内，红色表示超出。`);
    st.textContent=me.fits?tx(`Fits in ${CH_BUDGET} GiB: ${g1(sp)} GiB to spare`,`不超过 ${CH_BUDGET} GiB：还剩 ${g1(sp)} GiB`):tx(`Over ${CH_BUDGET} GiB by ${g1(-sp)} GiB`,`超出 ${CH_BUDGET} GiB，多了 ${g1(-sp)} GiB`);}
  else {st.hidden=true;}
  $('#tf-cap',app).innerHTML=NB(capTxt(r));
  $('#tf-legend',app).innerHTML=legendHTML(r);
  $('#tf-stack',app).innerHTML=NB(stackHTML(r));
  $('#tf-share',app).innerHTML=shareHTML(r);
  placeStream(); fitShareLabels();
  const wk=$('#tf-now',app);
  if(mode==='guided'&&step===PART.walk){wk.hidden=false; wk.textContent=walkWhere();} else wk.hidden=true;
}
function walkWhere(){
  const st=WALK[walk], n=tx(`“ is” is now at: `,`“ is” 现在在：`);
  return n+({id:tx('its token ID','它的 token ID'),emb:tx('the embedding table','embedding 表'),attn:tx('layer 1, attention block','第 1 层的 attention 块'),mlp:tx('layer 1, MLP block','第 1 层的 MLP 块'),
    rest:S.L>1?tx(`layers 2 to ${S.L}`,`第 2 到 ${S.L} 层`):tx('the end of the layers','层的末尾'),norm:tx('the final RMSNorm','最后的 RMSNorm'),head:tx('the LM head','LM head')}[st]);
}

/* ---------- settings panel ---------- */
function controlsHTML(){
  const opts=MKEYS.map(k=>`<option value="${k}">${esc(MODELS[k].name)}</option>`).join('')+`<option value="custom" disabled>${tx('Custom (changed by hand)','自定义（手动改过）')}</option>`;
  const rg=(key,label,stops,idx,out,note)=>A.ui.range({id:'tf-ctl-'+key,key,label,min:0,max:stops.length-1,step:1,value:idx,out,note});
  const K=k=>tx(`<code>${k}</code> in config.json. `,`config.json 里的 <code>${k}</code>。`);
  return `<aside class="panel controls" id="tf-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-hint">${tx('Below the model: the seven numbers of its config.json and, last, the precision its weights are stored in.','模型下面是它 config.json 里的 7 项，最后一项是权重存储的精度。')}</p><p class="ctl-lock" id="tf-lock" hidden></p>`+
   `<div class="ctl lockable" id="tf-ctl-model"><label for="tf-model">${tx('Model','模型')}</label><select id="tf-model">${opts}</select><div class="ctl-note">${tx('Sets every number below from that model’s config.json.','从这个模型的 config.json 取下面所有的数。')}</div></div>`+
   A.ui.range({id:'tf-ctl-L',key:'L',label:tx('Layers','层数'),min:L_MIN,max:L_MAX,step:1,value:S.L,out:layersTxt(S.L),
     note:K('num_hidden_layers')+tx('How many layers the stack has. Each adds one attention block, one MLP block and two RMSNorms.','这一叠有几层。每层多一个 attention 块、一个 MLP 块和两个 RMSNorm。')})+
   `<div class="lockable">`+rg('d',tx('Hidden size','隐藏维度'),D_STOPS,D_STOPS.indexOf(S.d),fmtInt(S.d),
     K('hidden_size')+tx('The length d of each token’s vector between layers. Nearly every matrix has it on one side.','层与层之间每个 token 的向量的维数 d。几乎每个矩阵的一边都是它。'))+`</div>`+
   `<div class="lockable">`+rg('f',tx('Intermediate size','中间维度'),F_STOPS,F_STOPS.indexOf(S.f),fmtInt(S.f),
     K('intermediate_size')+tx('The length the MLP block widens the vector to. Its gate, up and down matrices are each hidden size × intermediate size.','MLP 块把向量放大到的维数。gate、up、down 三个矩阵都是“隐藏维度 × 中间维度”。'))+`</div>`+
   `<div class="lockable">`+rg('V',tx('Vocabulary size','词表大小'),V_STOPS,V_STOPS.indexOf(S.V),fmtInt(S.V),
     K('vocab_size')+tx('How many tokens the model knows. The embedding table and the LM head each hold one vector per token, as long as the hidden size.','模型认识多少个 token。embedding 表和 LM head 给每个 token 各存一个向量，维数等于隐藏维度。'))+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'tf-ctl-tied',key:'tied',label:tx('Embedding table and LM head','embedding 表和 LM head'),options:[['0',tx('Separate','分开')],['1',tx('Tied','共享参数')]],value:S.tied?'1':'0',
     note:K('tie_word_embeddings')+tx('Tied: the LM head reuses the embedding table’s parameters and adds none. Llama-3.2-1B does this; the other four models here do not.','共享参数：LM head 直接用 embedding 表的参数，不另加参数。Llama-3.2-1B 这样做，这里另外四个模型不这样做。')})+`</div>`+
   `<div class="lockable">`+A.ui.seg({id:'tf-ctl-H',key:'H',label:tx('Heads','head 数'),options:H_OPTS.map(v=>[String(v),String(v)]),value:String(S.H),
     note:K('num_attention_heads')+tx(`How many heads the attention block runs side by side; in this lab each head works with pieces of head_dim numbers cut from what Wq, Wk and Wv compute, with ${MX(HD_TEX())}, as in Meta’s Llama code. More heads leave Wq and Wo at ${MX('\\text{hidden size} \\times \\text{hidden size}')} but make Wk and Wv narrower, because those are ${MX('\\text{KV heads} \\times \\text{head_dim}')} wide.`,
             `attention 块里并排跑几个 head。在这个 Lab 里，每个 head 分到 Wq、Wk、Wv 算出的向量里的 head_dim 维，${MX(HD_TEX())}，和 Meta 的 Llama 代码一样。head 越多，Wq 和 Wo 仍是 ${MX('\\text{隐藏维度} \\times \\text{隐藏维度}')}，Wk 和 Wv 却变窄，因为它们的宽度是 ${MX('\\text{KV head 数} \\times \\text{head_dim}')}。`)})+`</div>`+
   `<div class="lockable" id="tf-kvwrap"></div>`+
   `<div class="lockable">`+A.ui.seg({id:'tf-ctl-p',key:'p',label:tx('Weight precision','权重精度'),options:[['bf16','BF16'],['fp8','FP8'],['fp4','FP4']],value:S.p,
     note:tx('Bytes per weight in memory: 2 in BF16, the format these models are published in, 1 in FP8 and 0.5 in FP4. Whoever serves the model picks it; it is not in config.json, so it changes the memory, not the count. In this lab every weight takes the same bytes. Real FP8 and FP4 checkpoints also store small scale factors, left out here, and lower precision can cost quality '+quantRef()+'.',
             '每个权重在显存里占的字节数：BF16 是 2，这些模型发布时就是 BF16；FP8 是 1；FP4 是 0.5。它由部署模型的人决定，不在 config.json 里，所以只改显存，不改参数量。在这个 Lab 里每个权重的字节数都一样。真实的 FP8 和 FP4 checkpoint 还会存一些小的缩放系数，这里没算。降精度也可能影响质量'+quantRef()+'。')})+`</div>`+
   `<div><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
function kvSegHTML(){
  const ok=KV_OPTS.filter(v=>v<=S.H);
  return A.ui.seg({id:'tf-ctl-KV',key:'KV',label:tx('KV heads','KV head 数'),options:ok.map(v=>[String(v),String(v)]),value:String(S.KV),
    note:tx(`<code>num_key_value_heads</code> in config.json. It sets how wide Wk and Wv are: each is hidden size × (KV heads × head_dim), so each KV head adds 2 × hidden size × head_dim parameters per layer. The lab ${LK('gqa-mla')} explains why models keep fewer KV heads than heads.`,
            `config.json 里的 <code>num_key_value_heads</code>。它决定 Wk 和 Wv 的宽度：两者都是“隐藏维度 ×（KV head 数 × head_dim）”，所以每多一个 KV head，每层多“2 × 隐藏维度 × head_dim”个参数。为什么 KV head 比 head 少，见${LK('gqa-mla')}。`)});
}
function syncControls(r){
  const c=$('#tf-controls',app), lock=mode==='challenge';
  c.classList.toggle('locked',lock);
  const ln=$('#tf-lock',app); ln.hidden=!lock;
  if(lock) ln.textContent=NB(tx(`In the challenge you can change only the number of layers. Everything else is fixed: Llama-3.1-8B’s config, BF16 weights and a ${CH_BUDGET} GiB budget on one RTX 4090.`,
                                `挑战中只能改层数。其余设置固定：Llama-3.1-8B 的 config，权重 BF16，一张 RTX 4090 上最多 ${CH_BUDGET} GiB。`));
  const k=presetOf(S), sel=$('#tf-model',app); sel.value=k||'custom';
  const setR=(key,val,out)=>$$(`input[data-key="${key}"]`,app).forEach(i=>{if(+i.value!==val) i.value=val; A.ui.setRange(i); const o=i.closest('.ctl,.inline-ctl').querySelector('output'); if(o) o.textContent=out;});
  const li=$('#tf-ctl-L-in',app); if(li){li.min=lock?CH_MIN:L_MIN; li.max=lock?CH_MAX:L_MAX;}
  setR('L',S.L,layersTxt(S.L)); setR('d',D_STOPS.indexOf(S.d),fmtInt(S.d)); setR('f',F_STOPS.indexOf(S.f),fmtInt(S.f)); setR('V',V_STOPS.indexOf(S.V),fmtInt(S.V));
  const kw=$('#tf-kvwrap',app); if(kw&&kw.dataset.h!==String(S.H)+A.lang()){kw.innerHTML=kvSegHTML(); kw.dataset.h=String(S.H)+A.lang();}
  $$('.seg-ctl[data-key]',app).forEach(g=>{const key=g.dataset.key;
    if(key==='tied') A.ui.setPressed(g,S.tied?'1':'0'); else if(key==='model') A.ui.setPressed(g,k||''); else if(S[key]!=null) A.ui.setPressed(g,String(S[key]));});
}

/* ---------- rail ---------- */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx('Pick a scenario and see how many parameters the model has and how much memory its weights take.','选一个场景，看看模型有多少参数、权重占多少显存。')}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="tf-insight" aria-live="polite"></div>`+
   `<p class="small">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在${J('controls','“自由调参”')}里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function renderRail(){
  const rail=$('#tf-rail',app); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):'';     // the next lab, below the card (R3.11)
  rail.innerHTML=NB(railHTML())+exitsCard()+nl; A.ui.initRanges(rail); render();
}
function updateRail(r){
  const rail=$('#tf-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect(r)) observed[step]=true;
    const show=!!s.obs&&(s.obsAlways||!s.expect||!!observed[step]);
    if(box){box.hidden=!show;
      if(show){const lab=typeof s.label==='function'?s.label():L(s.label||{en:'Observed',zh:'观察到'}), html=NB(`<div class="eyebrow${/\d/.test(lab)?' keepcase':''}">${esc(lab)}</div>`+s.obs(r));
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(wantObs){wantObs=false; requestAnimationFrame(()=>revealBox(box));}}}
    // part 3's stepper and part 6's options follow the state
    const wc=$('#tf-ic-walk',rail); if(wc){const up=$('[data-walk="up"]',wc), dn=$('[data-walk="down"]',wc); if(up) up.disabled=walk<=0; if(dn) dn.disabled=walk>=WALK.length-1;}
    const cur=lvNow(); $$('[data-lever]',rail).forEach(b=>{const i=+b.dataset.lever; b.setAttribute('aria-pressed',String(i===cur)); const rr=$('.lv-r',b); if(rr) rr.textContent=lvTried.has(LV[i].id)?bgT(compute(lvCfg(i)).total):'';});
    $$('[data-nav="next"]',rail).forEach(nb=>{nb.disabled=isNextDisabled();});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#tf-insight',rail); if(ins){const i=cur, html=NB(`<div class="eyebrow">${i>=0?esc(L(PRESETS[i].t)):tx('Your own setup','你自己的设置')}</div>`+insightHTML(r)); if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const me=chPoint(S.L), nw=$('#tf-ch-now',rail); if(!nw) return;
    nw.textContent=g1(me.gib); const pr=$('#tf-ch-prog',rail), gl=$('#tf-ch-goal',rail);
    $('i',pr).style.width=Math.min(100,me.gib/CH_BUDGET*100)+'%'; pr.classList.toggle('met',me.fits); gl.classList.toggle('met',me.fits);
    pr.classList.toggle('tf-over',!me.fits); gl.classList.toggle('tf-over',!me.fits);
    if(chSubmitted&&me.fits) chReveal=true;
    const res=$('#tf-ch-res',rail), html=NB(chResultHTML()); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}
/* after an action, the result box comes into view (R3.10): scroll only as far as needed to show its end, but never so far that the option
   just clicked, or the box's own top, goes under the top bar; a box taller than the screen then shows its top (the result) first */
let lastClick=null;
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, q=box.getBoundingClientRect(), vh=innerHeight;
  if(q.bottom<=vh-8&&q.top>=top) return;
  let dy=q.bottom-vh+16;
  const c=lastClick&&lastClick.isConnected?lastClick.getBoundingClientRect():null;
  if(c&&c.top>=top) dy=Math.min(dy,c.top-top);
  dy=Math.min(dy,q.top-top);
  if(dy>0) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});
}
/* after Submit in the challenge (R3.10, review r13 M4): scroll only as far as the verdict's first paragraph needs, and keep the layer
   slider below the top bar and the mini result bar, since the verdict may ask for more layers. Of the scrolls that do both, the smallest
   is taken; only on a screen too short for both does the verdict win */
function revealVerdict(){
  const res=$('#tf-ch-res',app), v=res&&res.firstElementChild, sl=$('#tf-ic-L',app); if(!v) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?A.ui.barB():56)+Math.max(dk&&!dk.hidden?dk.offsetHeight:0,50)+8;   // room for the mini result bar even before it shows
  const lo=v.getBoundingClientRect().bottom-(innerHeight-16), hi=sl?sl.getBoundingClientRect().top-top:Infinity;
  const dy=lo>hi?lo:Math.min(Math.max(0,lo),hi);
  if(Math.abs(dy)>=1) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});
}

/* ---------- render ---------- */
function render(){
  if(!app) return;
  const r=compute(S);
  if(mode==='guided'&&STEPS[step].id==='vocab'&&presetOf(S)==='mistral-7b') seenMistral=true;
  if(mode==='guided'){const s=STEPS[step]; if(s.expect&&s.expect(r)) observed[step]=true;}
  if(mode==='challenge'&&chSubmitted&&chPoint(S.L).fits) chReveal=true;     // before Show the math, which adds its budget line once the chart shows
  drawStage(r); drawMetrics(r);
  const mb=$('#tf-mathbody',app), mh=NB(mathHTML(r)); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(r); updateRail(r);
  A.hash.write(token()); A.dock.refresh(); announce(r);
  userAct=false;
}
function announce(r){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return;
  const sh=MASKED()?'':tx(` MLP blocks ${pctS(r.parts.mlp/r.total)}, attention blocks ${pctS(r.parts.attn/r.total)}.`,`MLP 块 ${pctS(r.parts.mlp/r.total)}，attention 块 ${pctS(r.parts.attn/r.total)}。`);
  A.announce(tx(`${configName(r.s)}: ${bgT(r.total)} parameters, ${gibS(r.gib)} in ${PL[r.s.p]}.`,`${configName(r.s)}：${bgT(r.total)} 个参数，${PL[r.s.p]} 下 ${gibS(r.gib)}。`)+sh);},400);}
function clampCfg(){
  if(!H_OPTS.includes(S.H)) S.H=32;
  if(S.KV>S.H) S.KV=S.H;
  if(S.d%S.H) S.H=H_OPTS.filter(h=>S.d%h===0).pop();
  S.L=Math.max(L_MIN,Math.min(L_MAX,Math.round(S.L)));
}
function setState(p){
  Object.assign(S,p); clampCfg();
  if(mode==='challenge'){const l=Math.max(CH_MIN,Math.min(CH_MAX,S.L)); Object.assign(S,CH); S.L=l;}
  render();
}
const learnerSet=p=>{userAct=true; setState(p);};
function applyModel(k){if(!MODELS[k]) return; learnerSet({...MODELS[k].cfg});}
function enterStep(i){
  step=Math.max(0,Math.min(STEPS.length-1,i)); wantObs=false;
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id==='walk') walk=0;
  if(step>=PART.split) revealed=true;
  const st=$('#tf-stage',app); if(st) st.scrollTop=0;
  renderRail();
}
function revealStep(){A.ui.reveal($('#tf-rail .step.current',app)||$('#tf-rail',app));}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH};}
  A.shell.setMode(app,m,DESCS());
  const st=$('#tf-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
const token=()=>[S.V,S.d,S.L,S.H,S.KV,S.f,S.tied?1:0,S.p].join('_');
function parse(tok){
  const a=String(tok||'').split('_'); if(a.length!==8) return null;
  const o={V:+a[0],d:+a[1],L:+a[2],H:+a[3],KV:+a[4],f:+a[5],tied:a[6]==='1',p:a[7]};
  if(!V_STOPS.includes(o.V)||!D_STOPS.includes(o.d)||!F_STOPS.includes(o.f)||!H_OPTS.includes(o.H)||!KV_OPTS.includes(o.KV)||o.KV>o.H||!(a[6]==='0'||a[6]==='1')||!PREC[o.p]) return null;
  if(!Number.isInteger(o.L)||o.L<L_MIN||o.L>L_MAX||o.d%o.H) return null;
  return o;
}

/* ---------- header, popovers, connect ---------- */
const DESCS=()=>({
  guided:tx('Seven short parts. Predict first, follow one token through Llama-3.1-8B, see where its parameters are, then count a wider and a deeper version from config.json. Start here if this is new to you.',
            '一共 7 个部分。先预测，再跟着一个 token 走一遍 Llama-3.1-8B，看它的参数都在哪，再从 config.json 算出宽一倍和深一倍的版本。第一次学从这里开始。'),
  sandbox:tx('Five real models, from Llama-3.2-1B for phones to Llama-3.1-405B. Pick one to see how many parameters it has, how much memory its weights take and where they are.',
             '五个真实模型，从给手机用的 Llama-3.2-1B 到 Llama-3.1-405B。选一个，看它有多少参数、权重占多少显存、参数都在哪。'),
  challenge:tx(`One RTX 4090 may hold at most ${CH_BUDGET} GiB of BF16 weights. Change only the number of layers of a model shaped like Llama-3.1-8B, and fit as many layers as you can.`,
               `一张 RTX 4090 最多放 ${CH_BUDGET} GiB 的 BF16 权重。模型结构和 Llama-3.1-8B 一样，只改层数，放下尽量多的层。`)
});
const cite=(url,txt)=>`<a href="${url}" target="_blank" rel="noopener">${txt}</a>`;
const HF=p=>`https://huggingface.co/${p}/blob/main/config.json`;
const FIDPOP=()=>tx(
 `<div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul><li>Every parameter, counted exactly from the shapes in config.json: the embedding table, each layer’s Wq, Wk, Wv and Wo, its gate, up and down matrices and its two RMSNorms’ scale vectors, the final RMSNorm and the LM head.</li><li>Weight memory: ${MX('\\text{weights} = \\text{parameters} \\times \\text{bytes per parameter}')}.</li><li>For the five models the totals equal the parameter counts Hugging Face reads from their safetensors files.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul><li>No biases: Llama and Mistral have none. Some models add them, for example Qwen2.5 on its q, k and v projections.</li><li>In this lab ${MX(HD_TEX())} always, as in Meta’s Llama code; some models set head_dim on their own.</li><li>FP8 and FP4 leave out the scale factors real checkpoints store.</li><li>Mixture-of-experts models are not covered. RoPE, the position encoding of these models, has no parameters.</li><li>GPT-2 differs: learned position embeddings, LayerNorm with a bias, and an MLP with two matrices and an intermediate size of 4 × hidden size.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul><li>Shapes: config.json of ${cite(HF('unsloth/Llama-3.2-1B'),'Llama-3.2-1B')}, ${cite(HF('unsloth/Meta-Llama-3.1-8B'),'Llama-3.1-8B')} and ${cite(HF('unsloth/Meta-Llama-3.1-70B'),'Llama-3.1-70B')} (public mirrors of Meta’s gated repos) and of ${cite(HF('mistralai/Mistral-7B-v0.1'),'Mistral-7B-v0.1')} (mistralai), and Meta’s ${cite('https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py','sku_list.py')} for Llama-3.1-405B; Llama 3 paper, Table 3.</li><li>Structure: Meta’s ${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py','models/llama3/model.py')} and transformers’ ${cite('https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py','modeling_llama.py')}.</li><li>Token IDs 374 (“ is”) and 12366 (“ Paris”): Meta’s ${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.model','Llama 3 tokenizer')}.</li><li>Llama-3.2-1B for phones: its ${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md','model card')} (“highly constrained environments, such as mobile devices”).</li><li>RTX 4090: 24 GB (NVIDIA), read here as 24 GiB.</li></ul><p class="mono small muted" style="margin-top:8px">Last verified 2026-09-25. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>按 config.json 里的形状精确地数出每一个参数：embedding 表，每层的 Wq、Wk、Wv、Wo，gate、up、down 三个矩阵和两个 RMSNorm 的系数向量，最后的 RMSNorm，以及 LM head。</li><li>权重占的显存：${MX('\\text{权重} = \\text{参数量} \\times \\text{每个参数的字节数}')}。</li><li>五个模型的总数都和 Hugging Face 从它们的 safetensors 文件读出的参数量一致。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul><li>没有 bias：Llama 和 Mistral 都没有。有的模型会加，比如 Qwen2.5 的 q、k、v 投影。</li><li>这个 Lab 里总是 ${MX(HD_TEX())}，和 Meta 的 Llama 代码一样。有的模型单独设 head_dim。</li><li>FP8 和 FP4 没算真实 checkpoint 里存的缩放系数。</li><li>不涉及混合专家（MoE）模型。这些模型的位置编码 RoPE 没有参数。</li><li>GPT-2 不一样：位置编码是学出来的，LayerNorm 带 bias，MLP 只有两个矩阵，中间维度是 4 × 隐藏维度。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul><li>形状：${cite(HF('unsloth/Llama-3.2-1B'),'Llama-3.2-1B')}、${cite(HF('unsloth/Meta-Llama-3.1-8B'),'Llama-3.1-8B')}、${cite(HF('unsloth/Meta-Llama-3.1-70B'),'Llama-3.1-70B')} 的 config.json（Meta 的仓库需要申请，这里用公开镜像），${cite(HF('mistralai/Mistral-7B-v0.1'),'Mistral-7B-v0.1')} 的 config.json（mistralai），Llama-3.1-405B 用 Meta 的 ${cite('https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py','sku_list.py')}；Llama 3 论文表 3。</li><li>结构：Meta 的 ${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py','models/llama3/model.py')} 和 transformers 的 ${cite('https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py','modeling_llama.py')}。</li><li>token ID 374（“ is”）和 12366（“ Paris”）：Meta 的 ${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.model','Llama 3 tokenizer')}。</li><li>Llama-3.2-1B 给手机用：它的${cite('https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md','模型卡')}（“highly constrained environments, such as mobile devices”）。</li><li>RTX 4090：24 GB（NVIDIA），这里按 24 GiB 计。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-09-25。原型，尚未经专家审阅。</p></div></div>`);
const PRIPOP=()=>tx(`<p>A language model reads a sequence of token IDs and gives every token it knows a probability of coming next. Its parameters, also called weights, are the numbers it learned in training, stored as matrices whose shapes the model’s config.json lists.</p><p>The components these matrices make up, from the embedding table to the LM head, are shown one by one in ${LK('architecture')}, the first lab of this path.</p>`,
                    `<p>语言模型读入一串 token ID，给它认识的每个 token 一个“接下来出现”的概率。它的参数（也叫权重）是训练学到的数，存成一个个矩阵，这些矩阵的形状写在模型的 config.json 里。</p><p>这些矩阵组成了哪些部件（从 embedding 表到 LM head），在这条路径的第一个 Lab${LK('architecture')}里逐一展示。</p>`);
/* the capability layer (Codex review 1 #5, Batch 2): what the learner can do after the lab, and three ways to show it transfers.
   Every number comes from compute(). Nothing here gives the challenge away: no layer or model is turned into GiB, and nothing asks how
   many layers fit in a budget. Model names carry the no-break hyphen (U+2011) themselves, since the kit re-renders the card without NB() */
const OUTCOME=()=>tx('count a model’s parameters and weight memory from its config.json, and explain why most of them sit in the MLP blocks.',
                     '从 config.json 算出模型的参数量和权重占的显存，并讲清为什么大部分参数在 MLP 块里。');
/* a new case opens as its scenario, on the stage (as kvGo does in the KV cache lab); the lab sets the state, so no ▲▼ (R6.6) */
function exGo(k){setMode('sandbox'); setState({...MODELS[k].cfg,p:'bf16'}); A.ui.reveal($('#tf-stage',app));}
function EXITS(){
  const c=DEFAULT, r=r8(), d=fmtInt(c.d), hd=fmtInt(r.hd), n1=A.fmt.num, h2=c.H/2;
  const mha=compute({...c,KV:c.H}), r16=compute({...c,H:h2});                         // a key and value for every head; half the heads
  const m=compute(MISTRAL()), mc=m.s;
  /* Mixtral-8x7B: Mistral-7B-v0.1's shapes with 8 experts per MLP block, 2 used per token (config.json: num_local_experts 8,
     num_experts_per_tok 2); total 46,702,792,704 as Hugging Face reports */
  const XE=8, XK=2, xExtra=mc.L*(XE-1)*m.mlp, xRouter=mc.L*mc.d*XE, xTot=m.total+xExtra+xRouter, xAct=m.total+mc.L*(XK-1)*m.mlp+xRouter, xo=[xAct,xTot,XE*m.total];
  const s=compute({...MODELS['llama32-1b'].cfg,p:'bf16'}), sc=s.s, sd=fmtInt(sc.d), ratio=s.emb/s.parts.attn;
  const about=n=>tx(`About ${bgT(n)}`,`约 ${bgT(n)}`);
  /* R6.20: "Explain it" says the principle in words, with at most one number; "A new case" works it out in short steps, one per line */
  const hf=FR_ROUND.find(g=>g[0]>mha.attn/r.mlp)||FR[FR.length-1];                    // an attention block with a KV head per head, against an MLP block
  return {
  core:()=>chSubmitted, explain:[
   {q:tx('In Llama‑3.1‑8B, why do the MLP blocks hold far more parameters than the attention blocks?','Llama‑3.1‑8B 里，MLP 块的参数为什么比 attention 块多得多？'),
    o:[tx(`The MLP’s three matrices each connect the hidden size to an intermediate size ${n1(c.f/c.d,1)} times larger, so together they outweigh attention’s four.`,
          `MLP 块的三个矩阵都是“隐藏维度 × 中间维度”，中间维度又是隐藏维度的 ${n1(c.f/c.d,1)} 倍，所以它们比 attention 块大得多。`),
       tx(`Each head works with a piece of only ${hd} numbers, so attention’s four matrices stay narrow, while the MLP’s three span every number of the token.`,
          `每个 head 只处理 ${hd} 维，所以 attention 块的四个矩阵都很窄，而 MLP 块的三个矩阵要覆盖 token 的每一维。`),
       tx(`Every ${c.H/c.KV} heads share one KV head, and this sharing shrinks attention so far that the MLP’s three matrices end up with most of the layer.`,
          `每 ${c.H/c.KV} 个 head 共用一个 KV head，这种共用把 attention 块缩得很小，于是一层的大部分参数落在 MLP 块的三个矩阵上。`)],a:0,
    n:[null,
       tx('Each head does work with a small piece, but the heads together cover the whole hidden size, so Wq and Wo are square. The MLP is larger because its intermediate size is wider still.',
          '每个 head 确实只处理一小段，但所有 head 合起来覆盖全部隐藏维度，所以 Wq 和 Wo 是方阵。MLP 块更大，是因为它的中间维度还要宽得多。'),
       tx(`Sharing narrows only Wk and Wv. Even with a key and value for every head, an attention block would hold less than ${hf[1]} as many parameters as an MLP block.`,
          `共用只让 Wk 和 Wv 变窄。就算每个 head 都有自己的 key 和 value，一个 attention 块的参数也不到一个 MLP 块的${hf[2]}。`)],
    why:tx(`Wq and Wo are hidden size × hidden size, and Wk and Wv are narrower. Each MLP matrix is hidden size × intermediate size, and the intermediate size is ${n1(c.f/c.d,1)} times the hidden size. So ${fracW(r.mlp/r.layer)} of every layer is MLP.`,
           `Wq 和 Wo 是“隐藏维度 × 隐藏维度”，Wk 和 Wv 更窄。MLP 块的每个矩阵是“隐藏维度 × 中间维度”，而中间维度是隐藏维度的 ${n1(c.f/c.d,1)} 倍。所以每层${fracW(r.mlp/r.layer)}的参数在 MLP 块里。`)},
   /* the lab's own case ("What to remember"): half the heads, the same KV heads */
   {q:tx(`Llama‑3.1‑8B goes from ${c.H} heads to ${h2} and keeps its ${c.KV} KV heads. What happens to its parameter count?`,`Llama‑3.1‑8B 的 head 从 ${c.H} 个减到 ${h2} 个，KV head 还是 ${c.KV} 个。参数量会怎样？`),
    o:[tx('It grows: head_dim doubles, so Wk and Wv, which are KV heads × head_dim wide, double, while Wq and Wo keep their size.','变多：head_dim 翻倍，Wk 和 Wv 宽“KV head 数 × head_dim”，也跟着翻倍；Wq 和 Wo 大小不变。'),
       tx('It shrinks: each head adds its own slice to Wq, Wk, Wv and Wo, so with half the heads the attention blocks hold half as much.','变少：每个 head 在 Wq、Wk、Wv、Wo 里各占自己的一份，head 减半，attention 块的参数也减半。'),
       tx('It stays the same: heads only decide how the vectors from Wq, Wk and Wv are cut into pieces, so no matrix changes size.','不变：head 数只决定 Wq、Wk、Wv 算出的向量怎样切给各个 head，所以四个矩阵的大小都不变。')],a:0,
    n:[null,
       tx('Wq and Wo keep their size at any head count: with half the heads, each head’s slice of them is twice as wide. Only Wk and Wv change, and they grow.',
          '不管几个 head，Wq 和 Wo 的大小都不变：head 减半，每个 head 分到的那份就宽一倍。变的只有 Wk 和 Wv，而且是变大。'),
       tx('That holds when every head has its own key and value. Here the KV heads stay, and each one’s key and value become twice as long, so Wk and Wv double.',
          '每个 head 都有自己的 key 和 value 时是这样。这里 KV head 数不变，而每个 KV head 的 key 和 value 都变长一倍，所以 Wk 和 Wv 翻倍。')],
    why:tx(`Halving the heads doubles head_dim. Wk and Wv are KV heads × head_dim wide, so both double, while Wq and Wo keep their size. The model grows a little, from ${BG(r.total)} to ${BG(r16.total)} parameters.`,
           `head 减半，head_dim 就翻倍。Wk 和 Wv 宽“KV head 数 × head_dim”，所以都翻倍，Wq 和 Wo 大小不变。模型只长了一点，从 ${BG(r.total)} 个参数变成 ${BG(r16.total)} 个。`)}],
  transfer:[
   /* a mixture of experts (this lab's own models have none): only the MLP blocks multiply */
   {q:tx(`Mixtral‑8x7B is Mistral‑7B‑v0.1 with each MLP block replaced by ${XE} MLP blocks of the same size, called experts, and a small router in each layer (${d} × ${XE}) that picks ${XK} of them for every token. How many parameters does Mixtral‑8x7B hold?`,
         `Mixtral‑8x7B 就是 Mistral‑7B‑v0.1 把每层的 MLP 块换成 ${XE} 个同样大小的 MLP 块（叫作专家），每层再加一个小路由器（${d} × ${XE}），为每个 token 选其中 ${XK} 个。Mixtral‑8x7B 一共有多少参数？`),
    o:xo.map(about),a:1,
    n:[tx(`That is what one token uses, with ${XK} experts in each layer. The model holds all ${XE}.`,`这是一个 token 用到的参数：每层 ${XK} 个专家。模型要存全部 ${XE} 个。`),
       null,
       tx(`That copies the whole model ${XE} times. Only the MLP blocks are copied: attention, the embedding table and the LM head stay as in Mistral‑7B‑v0.1.`,`这是把整个模型复制了 ${XE} 份。复制的只有 MLP 块：attention、embedding 表和 LM head 都和 Mistral‑7B‑v0.1 一样。`)],
    why:tx(`About ${bgT(xTot)}: only the MLP blocks are copied.<br>Each layer gains ${XE-1} more MLP blocks: ${mc.L} × ${XE-1} × ${fmtInt(m.mlp)} = ${fmtInt(xExtra)}.<br>The routers add ${mc.L} × ${d} × ${XE} = ${fmtInt(xRouter)}.<br>Total: ${fmtInt(m.total)} (Mistral‑7B‑v0.1) + ${fmtInt(xExtra)} + ${fmtInt(xRouter)} = ${fmtInt(xTot)}.<br>One token uses only ${XK} experts per layer, about ${bgT(xAct)}.`,
           `约 ${bgT(xTot)}：复制的只有 MLP 块。<br>每层多出 ${XE-1} 个 MLP 块：${mc.L} × ${XE-1} × ${fmtInt(m.mlp)} = ${fmtInt(xExtra)}。<br>路由器再加 ${mc.L} × ${d} × ${XE} = ${fmtInt(xRouter)}。<br>一共：${fmtInt(m.total)}（Mistral‑7B‑v0.1）+ ${fmtInt(xExtra)} + ${fmtInt(xRouter)} = ${fmtInt(xTot)}。<br>一个 token 每层只用 ${XK} 个专家，约 ${bgT(xAct)}。`)},
   {q:tx(`Llama‑3.2‑1B has hidden size ${sd} and ${sc.L} layers, each with ${sc.H} heads and ${sc.KV} KV heads, and Llama‑3.1‑8B’s vocabulary of ${fmtInt(sc.V)} tokens. Which holds more parameters: its embedding table, or its ${sc.L} attention blocks together?`,
         `Llama‑3.2‑1B 的隐藏维度是 ${sd}，共 ${sc.L} 层，每层 ${sc.H} 个 head、${sc.KV} 个 KV head，词表和 Llama‑3.1‑8B 一样有 ${fmtInt(sc.V)} 个 token。一边是它的 embedding 表，一边是 ${sc.L} 个 attention 块加起来，哪边的参数更多？`),
    o:[tx('The embedding table','embedding 表'),tx('About the same','两边差不多'),tx(`The ${sc.L} attention blocks`,`${sc.L} 个 attention 块`)],a:Math.abs(ratio-1)<0.1?1:(ratio>1?0:2),
    why:tx(`The embedding table, about ${n1(ratio,1)} times the ${sc.L} attention blocks together. A small model keeps the whole vocabulary, but its layers are few and narrow.<br>Embedding table: ${fmtInt(sc.V)} × ${sd} = ${BG(s.emb)} parameters.<br>head_dim = ${sd} ÷ ${sc.H} = ${fmtInt(s.hd)}, so Wk and Wv are ${sd} × ${fmtInt(s.kvw)} (${sc.KV} KV heads × ${fmtInt(s.hd)}).<br>One attention block: 2 × ${sd} × ${sd} + 2 × ${sd} × ${fmtInt(s.kvw)} = ${BG(s.attn)}; the ${sc.L} blocks hold ${BG(s.parts.attn)}.`,
           `embedding 表更多，约是 ${sc.L} 个 attention 块合计的 ${n1(ratio,1)} 倍。小模型保留了完整的词表，层却又少又窄。<br>embedding 表：${fmtInt(sc.V)} × ${sd} = ${BG(s.emb)} 个参数。<br>head_dim = ${sd} ÷ ${sc.H} = ${fmtInt(s.hd)}，所以 Wk 和 Wv 是 ${sd} × ${fmtInt(s.kvw)}（${sc.KV} 个 KV head × ${fmtInt(s.hd)}）。<br>一个 attention 块：2 × ${sd} × ${sd} + 2 × ${sd} × ${fmtInt(s.kvw)} = ${BG(s.attn)}；${sc.L} 个块合计 ${BG(s.parts.attn)}。`),
    go:()=>exGo('llama32-1b'),goLabel:tx('Open it in Scenarios: Llama‑3.2‑1B','在应用场景里打开：Llama‑3.2‑1B')}],
  practice:[
   {id:'llm-fundamentals',d:tx('Questions 5 and 12: count the parameters of one layer and of a whole model with a tied LM head, then total versus active parameters in a mixture of experts.',
                              '第 5、12 题：数出一层和整个模型（LM head 和 embedding 表共享参数）的参数量，再算混合专家模型的总参数量和每个 token 激活的参数量。')},
   {id:'matmul-roofline',d:tx('Part 2: count the FLOPs and bytes moved by one pass through a 72-layer stack of matmuls, then check whether the tensors it must hold at once fit in 40 GB.',
                             '第 2 部分：算出一次前向传播经过 72 层矩阵乘法的 FLOPs 和搬运的字节数，再判断必须同时留在显存里的张量能不能放进 40 GB。')}]};
}
/* in the challenge only: after the challenge card, before the next lab (R3.11) */
const exitsCard=()=>mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'';
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Model parameters','模型参数'),title:l?L(l.q):tx('Where do an 8B model’s parameters live?','8B 模型的参数都在哪？'),
    minutes:l?l.min:15,fid:'Analytical',fidPop:FIDPOP(),primer:tx('Primer: a model is a set of parameters','前置知识：模型就是一组参数'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
function stageHTML(){
  return `<div class="panel stage tf-stage" id="tf-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="tf-st-title"></div><div class="meter-sub" id="tf-st-sub"></div></div><div class="stage-actions"><span class="status" id="tf-status" tabindex="0" hidden></span></div></div>`+
   `<p class="tf-cap" id="tf-cap"></p>`+
   `<div class="legend-row tf-legend" id="tf-legend"></div>`+
   `<div class="stage-graph graph tf-graph" id="tf-graph"><p class="tf-now mono" id="tf-now" hidden></p><div class="tf-stack" id="tf-stack"></div><div class="tf-share" id="tf-share"></div></div>`+
   `<div class="metrics tf-metrics" id="tf-metrics"></div>`+
   `<details class="math" id="tf-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="tf-mathbody"></div></details></div>`;
}
const CODE=`def count_params(c):
    """Parameters of a Llama-style model, from config.json."""
    d = c["hidden_size"]
    L = c["num_hidden_layers"]
    V = c["vocab_size"]
    H = c["num_attention_heads"]
    KV = c["num_key_value_heads"]
    f = c["intermediate_size"]
    hd = c.get("head_dim", d // H)
    attn = 2 * d * H * hd + 2 * d * KV * hd  # Wq Wo, Wk Wv
    mlp = 3 * d * f                          # gate, up, down
    layer = attn + mlp + 2 * d               # + 2 RMSNorms
    emb = V * d                              # embedding table
    tied = c.get("tie_word_embeddings")
    head = 0 if tied else V * d              # LM head
    return emb + L * layer + d + head        # + final RMSNorm

llama_31_8b = {
    "vocab_size": 128256,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "tie_word_embeddings": False,
}
n = count_params(llama_31_8b)
print(n)              # 8030261248
print(n * 2 / 1e9)    # 16.06... GB in BF16
print(n * 2 / 2**30)  # 14.95... GiB`;
function connectHTML(){
  const TF='https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py', VL='https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py', ML='https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py';
  const one=compute({...MODELS['llama32-1b'].cfg,p:'bf16'});
  return `<section class="wrap connect" aria-labelledby="tf-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="tf-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('The whole count from a config.json, as in this lab.','从 config.json 算出全部参数，和这个 Lab 的算法一样。')}</p><pre class="code" id="tf-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="tf-prod">`+
   tx(`<li>Hugging Face transformers builds Llama from these numbers in ${cite(TF,'<code>models/llama/modeling_llama.py</code>')}: <code>LlamaModel</code> holds the embedding table (<code>embed_tokens</code>), the layers and the final <code>norm</code>, and <code>LlamaForCausalLM</code> adds <code>lm_head</code>. Each <code>LlamaDecoderLayer</code> has a <code>LlamaAttention</code> (<code>q_proj</code>, <code>k_proj</code>, <code>v_proj</code>, <code>o_proj</code>), a <code>LlamaMLP</code> (<code>gate_proj</code>, <code>up_proj</code>, <code>down_proj</code>) and two <code>LlamaRMSNorm</code>s.</li>`+
      `<li>vLLM fuses Wq, Wk and Wv into one <code>QKVParallelLinear</code> and gate and up into one <code>MergedColumnParallelLinear</code> (${cite(VL,'<code>vllm/model_executor/models/llama.py</code>')}); the count is the same.</li>`+
      `<li><code>sum(p.numel() for p in model.parameters())</code> counts a tied table once: ${BG(one.total)} for Llama-3.2-1B.</li>`+
      `<li>Meta’s own reference code: ${cite(ML,'<code>models/llama3/model.py</code>')} in llama-models.</li>`,
      `<li>Hugging Face transformers 在 ${cite(TF,'<code>models/llama/modeling_llama.py</code>')} 里按这些数搭出 Llama：<code>LlamaModel</code> 包含 embedding 表（<code>embed_tokens</code>）、各层和最后的 <code>norm</code>，<code>LlamaForCausalLM</code> 再加上 <code>lm_head</code>。每个 <code>LlamaDecoderLayer</code> 有一个 <code>LlamaAttention</code>（<code>q_proj</code>、<code>k_proj</code>、<code>v_proj</code>、<code>o_proj</code>）、一个 <code>LlamaMLP</code>（<code>gate_proj</code>、<code>up_proj</code>、<code>down_proj</code>）和两个 <code>LlamaRMSNorm</code>。</li>`+
      `<li>vLLM 把 Wq、Wk、Wv 合成一个 <code>QKVParallelLinear</code>，把 gate 和 up 合成一个 <code>MergedColumnParallelLinear</code>（${cite(VL,'<code>vllm/model_executor/models/llama.py</code>')}），参数量不变。</li>`+
      `<li><code>sum(p.numel() for p in model.parameters())</code> 对共享参数的表只算一次：Llama-3.2-1B 是 ${BG(one.total)}。</li>`+
      `<li>Meta 自己的参考实现：llama-models 里的 ${cite(ML,'<code>models/llama3/model.py</code>')}。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite('https://arxiv.org/abs/1706.03762','Attention Is All You Need')}<span>${tx('Vaswani et al., 2017','Vaswani 等，2017')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/2407.21783','The Llama 3 Herd of Models')}<span>${tx('Llama Team, 2024; Table 3 lists the shapes of the 8B, 70B and 405B','Llama 团队，2024；表 3 列出 8B、70B、405B 的形状')}</span></li>`+
   `<li>${cite('https://arxiv.org/abs/2001.08361','Scaling Laws for Neural Language Models')}<span>${tx('Kaplan et al., 2020; Table 1 counts parameters and forward FLOPs per token','Kaplan 等，2020；表 1 数了参数量和每个 token 前向的 FLOP')}</span></li>`+
   `</ul><p class="small tf-see">${tx(`See also: ${cite('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')} (Cho et al., 2024) runs GPT-2 small in your browser on text you type. ${cite('https://bbycroft.net/llm','LLM Visualization')} (Brendan Bycroft) is a 3D walkthrough of a tiny GPT, nano-gpt, next to the shapes of GPT-2 small and XL.`,
        `另见：${cite('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')}（Cho 等，2024）在你的浏览器里对你输入的文字运行 GPT-2 small。${cite('https://bbycroft.net/llm','LLM Visualization')}（Brendan Bycroft）用 3D 一步步演示一个很小的 GPT（nano-gpt），旁边画出 GPT-2 small 和 XL 的形状。`)}</p></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="tf-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="tf-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="tf-app m2" id="tf-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="tf-rail" aria-label="${esc(A.t('railAria'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* ---------- interview drill: a seeded Llama-style config; every answer line redoes from the numbers shown (R6.6) ---------- */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function makeDrill(seed){
  const rnd=mulberry(seed), pick=a=>a[Math.floor(rnd()*a.length)];
  const d=pick([2048,3072,4096,5120,6144,8192]), H=pick([16,32,64].filter(h=>d%h===0&&d/h>=64&&d/h<=256)), KV=pick([1,2,4,8].filter(k=>k<=H));
  const f=Math.round(pick([2.75,3,3.5,4])*d/256)*256, L=pick([16,24,28,32,40,48]), V=pick([32000,50257,128256,152064]), tied=rnd()<0.3;
  const cfg={V,d,L,H,KV,f,tied,p:'bf16'}; return {seed,cfg,r:compute(cfg)};
}
function renderDrill(){
  if(!app) return; const {cfg:c,r}=makeDrill(drillSeed), q=$('#tf-drill-q',app), a=$('#tf-drill-a',app);
  q.innerHTML=tx(`A Llama-style model’s config.json says: vocab_size <b>${c.V}</b>, hidden_size <b>${c.d}</b>, num_hidden_layers <b>${c.L}</b>, num_attention_heads <b>${c.H}</b>, num_key_value_heads <b>${c.KV}</b>, intermediate_size <b>${c.f}</b>, tie_word_embeddings <b>${c.tied}</b>. How many parameters does it have, and how much GPU memory do its weights take in BF16? Assume no biases.`,
                 `一个 Llama 结构的模型，config.json 里写着：vocab_size <b>${c.V}</b>，hidden_size <b>${c.d}</b>，num_hidden_layers <b>${c.L}</b>，num_attention_heads <b>${c.H}</b>，num_key_value_heads <b>${c.KV}</b>，intermediate_size <b>${c.f}</b>，tie_word_embeddings <b>${c.tied}</b>。它有多少参数？权重用 BF16 要占多少显存？假设没有 bias。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const d=fmtInt(c.d), hd=fmtInt(r.hd), kvw=fmtInt(r.kvw);
  a.innerHTML=tx(`<p><b>${BG(r.total)} parameters, ${memEq(r)}</b></p><p>head_dim = ${d} ÷ ${c.H} = ${hd}; Wk and Wv each give ${c.KV} × ${hd} = ${kvw} numbers per token.</p><p>attention = 2 × ${d} × ${d} + 2 × ${d} × ${kvw} = ${fmtInt(r.attn)}</p><p>MLP = 3 × ${d} × ${fmtInt(c.f)} = ${fmtInt(r.mlp)}</p><p>one layer = ${fmtInt(r.attn)} + ${fmtInt(r.mlp)} + 2 × ${d} = ${fmtInt(r.layer)}</p><p>model = ${fmtInt(c.V)} × ${d} + ${c.L} × ${fmtInt(r.layer)} + ${d}${c.tied?' (LM head tied: + 0)':` + ${d} × ${fmtInt(c.V)}`} = ${fmtInt(r.total)}</p><p>BF16: ${fmtInt(r.total)} × 2 bytes = ${fmtInt(r.bytes)} bytes = ${memEq(r)}</p>`,
                 `<p><b>${BG(r.total)} 个参数，${memEq(r)}</b></p><p>head_dim = ${d} ÷ ${c.H} = ${hd}；Wk 和 Wv 每个 token 各得到 ${c.KV} × ${hd} = ${kvw} 维。</p><p>attention = 2 × ${d} × ${d} + 2 × ${d} × ${kvw} = ${fmtInt(r.attn)}</p><p>MLP = 3 × ${d} × ${fmtInt(c.f)} = ${fmtInt(r.mlp)}</p><p>一层 = ${fmtInt(r.attn)} + ${fmtInt(r.mlp)} + 2 × ${d} = ${fmtInt(r.layer)}</p><p>模型 = ${fmtInt(c.V)} × ${d} + ${c.L} × ${fmtInt(r.layer)} + ${d}${c.tied?'（LM head 共享参数：+ 0）':` + ${d} × ${fmtInt(c.V)}`} = ${fmtInt(r.total)}</p><p>BF16：${fmtInt(r.total)} × 2 字节 = ${fmtInt(r.bytes)} 字节 = ${memEq(r)}</p>`);
}

/* ---------- term tooltips and jumps ---------- */
const CTLN={V:{en:'Vocabulary size',zh:'词表大小'},d:{en:'Hidden size',zh:'隐藏维度'},L:{en:'Layers',zh:'层数'},H:{en:'Heads',zh:'head 数'},KV:{en:'KV heads',zh:'KV head 数'},
  f:{en:'Intermediate size',zh:'中间维度'},tied:{en:'Embedding table and LM head',zh:'embedding 表和 LM head'},p:{en:'Weight precision',zh:'权重精度'},model:{en:'Model',zh:'模型'}};
const TERM={
 V:()=>tx(`The vocabulary size, ${fmtInt(S.V)} tokens, comes from vocab_size in config.json. Each token has a row in the embedding table and a column in the LM head.`,`词表大小 ${fmtInt(S.V)} 来自 config.json 的 vocab_size。每个 token 在 embedding 表里有一行，在 LM head 里有一列。`),
 d:()=>tx(`The hidden size d, ${fmtInt(S.d)}, comes from hidden_size in config.json: the length of each token’s vector between layers.`,`隐藏维度 d = ${fmtInt(S.d)}，来自 config.json 的 hidden_size：层与层之间每个 token 的向量有这么多维。`),
 L:()=>tx(`The layer count, ${S.L}, comes from num_hidden_layers in config.json.`,`层数 ${S.L} 来自 config.json 的 num_hidden_layers。`),
 H:()=>tx(`The number of heads, ${S.H}, comes from num_attention_heads in config.json. Each head works with pieces of head_dim numbers cut from what Wq, Wk and Wv compute from the whole token: ${MX(HD_TEX()+' = '+fmtInt(S.d/S.H))}.`,`head 数 ${S.H} 来自 config.json 的 num_attention_heads。每个 head 分到 Wq、Wk、Wv 从整个 token 算出的向量里的 head_dim 维，${MX(HD_TEX()+' = '+fmtInt(S.d/S.H))}。`),
 KV:()=>tx(`The KV heads, ${S.KV}, come from num_key_value_heads in config.json. They set how wide Wk and Wv are: ${S.KV} × ${fmtInt(S.d/S.H)} = ${fmtInt(S.KV*S.d/S.H)}.`,`KV head 数 ${S.KV} 来自 config.json 的 num_key_value_heads，它决定 Wk 和 Wv 的宽度：${S.KV} × ${fmtInt(S.d/S.H)} = ${fmtInt(S.KV*S.d/S.H)}。`),
 f:()=>tx(`The intermediate size, ${fmtInt(S.f)}, comes from intermediate_size in config.json.`,`中间维度 ${fmtInt(S.f)} 来自 config.json 的 intermediate_size。`),
 tied:()=>S.tied?tx('tie_word_embeddings is true, so the LM head reuses the embedding table’s parameters and adds none.','tie_word_embeddings 是 true，所以 LM head 和 embedding 表共享参数，不另加参数。'):tx('tie_word_embeddings is false, so the LM head is its own hidden size × vocabulary matrix.','tie_word_embeddings 是 false，所以 LM head 是单独的一个“隐藏维度 × 词表大小”矩阵。'),
 p:()=>tx(`Each weight takes ${bytesTxt(S.p)} in ${PL[S.p]}: the precision the model is served in, not a config.json number.`,`${PL[S.p]} 下每个权重占 ${bytesTxt(S.p)}：这是部署时选的精度，不是 config.json 里的数。`)
};
const locked=k=>mode==='challenge'&&!!CTLN[k]&&k!=='L';
function ctlValue(k){return {V:fmtInt(S.V),d:fmtInt(S.d),L:layersTxt(S.L),H:String(S.H),KV:String(S.KV),f:fmtInt(S.f),tied:S.tied?tx('one table (tied)','同一张表（共享参数）'):tx('two separate tables','分开的两张表'),p:PL[S.p],model:configName(S)}[k];}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='graph') return tx('Click to jump to the chart.','点击跳到图表。');
  if(!CTLN[k]) return '';
  if(locked(k)){const v=ctlValue(k); return tx(`Fixed during the challenge: “${L(CTLN[k])}” ${k==='tied'?'are':'is'} ${v}.`,`挑战中固定不变：“${L(CTLN[k])}”是${/^[A-Za-z0-9]/.test(v)?' ':''}${v}。`);}
  return tx(`Click to jump to “${L(CTLN[k])}”.`,`点击跳到“${L(CTLN[k])}”。`);
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#tf-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='controls'){const c=$('#tf-controls',app); A.tip.hide(); A.ui.reveal(c); A.ui.flash(c); return;}
  if(k==='graph'){const g=$('#tf-graph',app); A.tip.hide(); g.scrollIntoView({block:'start',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(g); return;}
  if(!CTLN[k]) return;
  if(locked(k)){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  const el=(k==='L'&&$('#tf-ic-L',app))||$('#tf-ctl-'+k,app); if(el) A.ui.jumpToEl(el);
}

/* ---------- events ---------- */
function act(a,btn){
  if(a==='submit'){chSubmitted=true; if(chPoint(S.L).fits) chReveal=true; renderRail(); revealVerdict(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='reset'){if(mode==='challenge') learnerSet({L:CH.L}); else learnerSet({...DEFAULT}); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#tf-code',app), fl=(t)=>{btn.textContent=t; clearTimeout(btn._t); btn._t=setTimeout(()=>{btn.textContent=A.t('copyCode');},1600);};
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); fl(tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(()=>fl(tx('Copied','已复制')),sel); else sel();}catch(err){sel();}}
}
function showMe(){const s=STEPS[step];
  wantObs=true;
  if(s.id==='walk'){walk=WALK.length-1; userAct=true; render(); return;}
  if(s.id==='vocab'){learnerSet(MISTRAL()); return;}
  if(s.id==='config'){const i=LV.findIndex(l=>l.id==='wide'); lvTried.add('wide'); learnerSet(lvCfg(i)); return;}
}
function bind(el){
  el.addEventListener('input',e=>{const t=e.target, k=t.dataset&&t.dataset.key; if(!k) return; const v=+t.value;
    if(k==='L') learnerSet({L:v}); else if(k==='d') learnerSet({d:D_STOPS[v]}); else if(k==='f') learnerSet({f:F_STOPS[v]}); else if(k==='V') learnerSet({V:V_STOPS[v]});});
  el.addEventListener('change',e=>{if(e.target.id==='tf-model') applyModel(e.target.value);});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset; lastClick=b;
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key;
      if(k==='model'){wantObs=true; applyModel(d.v); return;}
      if(k==='tied'){learnerSet({tied:d.v==='1'}); return;}
      if(k==='p'){learnerSet({p:d.v}); return;}
      learnerSet({[k]:+d.v}); return;}
    if(d.walk){walk=Math.max(0,Math.min(WALK.length-1,walk+(d.walk==='down'?1:-1))); wantObs=walk===WALK.length-1; userAct=false; render(); return;}
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){showMe(); return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#tf-rail [data-pred="${pred}"]`,app); if(fb) fb.focus(); return;}
    if(d.lever!=null){const i=+d.lever; lvTried.add(LV[i].id); wantObs=true; learnerSet(lvCfg(i)); return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; learnerSet({...MODELS[p.k].cfg,p:'bf16'}); requestAnimationFrame(()=>{if(app) revealBox($('#tf-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#tf-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
}
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,walk,revealed,seenMistral,
    lvTried:[...lvTried],drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&S.L===CH_BEST&&chPoint(S.L).fits}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=s=>s&&typeof s==='object'?parse([s.V,s.d,s.L,s.H,s.KV,s.f,s.tied?1:0,s.p].join('_')):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    S={...p}; if(mode==='challenge') S={...CH,L:Math.max(CH_MIN,Math.min(CH_MAX,p.L))};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    walk=Number.isInteger(o.walk)&&o.walk>=0&&o.walk<WALK.length?o.walk:0; revealed=o.revealed===true||step>=PART.split; seenMistral=o.seenMistral===true;
    lvTried.clear(); lvTried.add('start'); if(Array.isArray(o.lvTried)) o.lvTried.forEach(x=>{if(LV.some(l=>l.id===x)) lvTried.add(x);});
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:5; drillOpen=o.drillOpen===true;},
  mount(el,tok){
    root=el; wantObs=false;
    const p=tok?parse(tok):null; if(p){S={...p}; mode='sandbox';}
    el.innerHTML=viewHTML(); app=$('#tf-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#tf-rail',app));},token});
    bind(app);
    renderDrill();
    const rail=$('#tf-rail',app); rail.innerHTML=NB(railHTML())+exitsCard()+((mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):''); A.ui.initRanges(rail);   // a language switch re-mounts: the card stays (R8.3)
    for(const k in shownPrev) delete shownPrev[k];
    render();
  },
  leave(){clearTimeout(annT); root=null; app=null;},
  onLang(){if(root){const d=$('#tf-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#tf-math',app).open=true;}},
  onResize(){if(app){placeStream(); fitShareLabels();}},
  termTip(k){const t=TERM[k]; if(!t) return ''; return NB(`<div>${t()}</div><div class="act">${esc(jumpAct(k))}</div>`);},   // TERM is HTML (no markup but A.ui.math)
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#tf-stack',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#tf-graph',app),
    render({status,vis,read}){if(!app) return; const r=compute(S), mem=x=>`<span class="dk-mem">${x}</span>`;
      if(mode==='challenge'){const me=chPoint(S.L); status.className='status '+(me.fits?'ok':'bad'); status.textContent=me.fits?tx(`Fits in ${CH_BUDGET} GiB`,`不超过 ${CH_BUDGET} GiB`):tx(`Over ${CH_BUDGET} GiB`,`超出 ${CH_BUDGET} GiB`);
        vis.innerHTML=`<div class="tf-dkbar"><i class="${me.fits?'ok':'bad'}" style="width:${Math.min(100,me.gib/CH_GPU*100).toFixed(2)}%"></i><b style="left:${(CH_BUDGET/CH_GPU*100).toFixed(2)}%"></b></div>`;
        read.innerHTML=NB(esc(tx(`${S.L} layers take ${gibS(me.gib)} of BF16 weights`,`${S.L} 层的 BF16 权重占 ${gibS(me.gib)}`))+mem(esc(tx(` (budget: ${CH_BUDGET} GiB on one RTX 4090)`,`（一张 RTX 4090 上预算 ${CH_BUDGET} GiB）`)))); return;}
      status.className='status neutral';
      if(mode==='guided'&&step===PART.walk){status.textContent=`${walk+1} / ${WALK.length}`;
        vis.innerHTML=`<div class="tf-dkwalk">${WALK.map((_,i)=>`<i class="${i===walk?'on':(i<walk?'done':'')}"></i>`).join('')}</div>`;
        read.innerHTML=NB(esc(walkWhere())); return;}
      /* phones (≤560px): the unit and the share bar hide, since the sentence repeats both, so the sentence keeps a readable width (R3.2) */
      status.innerHTML=`<span>${bgT(r.total)}<span class="tf-dk-unit">${tx(' parameters',' 个参数')}</span></span>`;   // one flex item: no gap before the unit
      vis.innerHTML=MASKED()?`<div class="tf-dkshare"><i class="tf-pat-layer" style="width:100%"></i></div>`
        :`<div class="tf-dkshare">${[['mlp','tf-pat-mlp'],['attn','tf-pat-attn'],['emb','tf-pat-emb'],['head','tf-pat-head']].filter(x=>r.parts[x[0]]>0).map(([k,c])=>`<i class="${c}" style="width:${(r.parts[k]/r.total*100).toFixed(3)}%"></i>`).join('')}</div>`;
      const gib=mem(esc(tx(`; ${gibS(r.gib)} in ${PL[r.s.p]}`,`；${PL[r.s.p]} 下 ${gibS(r.gib)}`)));
      read.innerHTML=MASKED()?NB(esc(tx(`${configName(r.s)}: ${bgT(r.total)} parameters`,`${configName(r.s)}：${bgT(r.total)} 个参数`))+mem(esc(tx(` (${gibS(r.gib)} in ${PL[r.s.p]})`,`（${PL[r.s.p]} 下 ${gibS(r.gib)}）`)))+esc(tx('; the shares are shown in part 4','，占比在第 4 部分揭晓')))
        :NB(esc(tx(`${configName(r.s)}: MLP blocks ${pctS(r.parts.mlp/r.total)}, attention blocks ${pctS(r.parts.attn/r.total)} of ${bgT(r.total)} parameters`,`${configName(r.s)}：${bgT(r.total)} 个参数里，MLP 块 ${pctS(r.parts.mlp/r.total)}，attention 块 ${pctS(r.parts.attn/r.total)}`))+gib);}
  }
});

/* ---------- home tile teaser (R11.5): one control, a picture of where the parameters live with its legend, one sentence.
   The picture is drawn to scale by area. Its frame is all of the model's parameters; the embedding table and the LM head are columns at
   its two ends, in the order a token meets them (one shared block when they are tied, as in Llama-3.2-1B); the layers fill the middle
   as L equal cells, one per layer, column by column, since every layer has the same shapes. The final RMSNorm, under 0.001%, is too
   small to draw (note 2). Nothing shows how a layer splits between its attention and MLP blocks, part 2's question (R3.8): every cell
   has the one dotted pattern of parts 1–3, and no tooltip counts the blocks. Switching the model re-tiles the cells one after another
   (under 1 s, time-based) while the two tables resize; the first time the picture scrolls into view it builds up once in the same
   order. Reduced motion draws the end state. Keyboard: the picture is one tab stop, and the arrow keys move between its blocks. ---------- */
const TZ=['llama32-1b','llama31-8b','llama31-70b'];
/* rows of cells: the divisor of L that makes a cell closest to square in a 300 × 90 px frame (1B: 2 rows, 8B: 4, 70B: 5) */
function tzRows(r){let best=1, bd=1/0;
  for(let R=1;R<=r.s.L;R++){if(r.s.L%R) continue; const d=Math.abs(Math.log((300*r.layers/r.total)/(r.s.L/R)/(90/R))); if(d<bd){bd=d; best=R;}}
  return best;}
/* timing in ms, time-based (CSS animations): the old cells fade for TZ_LEAD, then cell i starts at TZ_LEAD + i·TZ_SPAN/L and takes TZ_CELL;
   the LM head of the build-up fades in at 660 ms (transformer.css). All of it ends by 0.96 s */
const TZ_LEAD=100, TZ_SPAN=620, TZ_CELL=220;
let tzIntro=false, tzIO=null, tzAt=0;   // the build-up has played; the observer that waits for it; the block that holds the picture's tab stop
A.registerTeaser(ID,{fid:'formula',render(el){
  if(tzIO){tzIO.disconnect(); tzIO=null;}
  const btn=k=>[k,MODELS[k].name.replace(/^Llama-3\.\d-/,'')];
  el.innerHTML=`<div class="tz tf-tz">`+A.ui.seg({key:'tz',label:tx('Model','模型'),options:TZ.map(btn),value:tzModel})+
    `<div class="tz-vis tf-tz-vis"><div class="tf-tz-cap"></div><div class="tf-tz-frame" role="group"><div class="tf-tz-in">`+
    `<span class="tf-tz-emb tf-pat-emb" role="img" data-k="emb"></span><div class="tf-tz-lay"></div><span class="tf-tz-head tf-pat-head" role="img" data-k="head"></span></div></div></div>`+
    `<div class="tz-leg tf-tz-leg"></div><p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const root=el.querySelector('.tf-tz'), q=s=>root.querySelector(s), seg=q('.seg-ctl'), cap=q('.tf-tz-cap'), frame=q('.tf-tz-frame'), emb=q('.tf-tz-emb'),
        lay=q('.tf-tz-lay'), head=q('.tf-tz-head'), leg=q('.tf-tz-leg'), rd=q('.tz-read'), notes=root.querySelectorAll('.tz-note');
  /* the picture's blocks in reading order: embedding table, layer 1 … L, LM head; one of them is in the tab order (roving tab stop) */
  const items=()=>[...frame.querySelectorAll('.tf-tz-emb, .tf-tz-grid:not(.tf-tz-old) .tf-tz-cell, .tf-tz-head[data-tipx]')];
  const roam=(i,focus)=>{const it=items(); if(!it.length) return; tzAt=Math.max(0,Math.min(it.length-1,i));
    it.forEach((x,j)=>{x.tabIndex=j===tzAt?0:-1;}); if(focus) it[tzAt].focus();};
  let geo={L:0,R:1,C:1,tied:false};
  /* how: 'still' draws the end state; 'go' re-tiles after the learner switched the model; 'wait' leaves the picture empty until it scrolls into view */
  const upd=how=>{const m=MODELS[tzModel], r=compute({...m.cfg,p:'bf16'}), P=r.parts, t=r.total, L_=r.s.L, tied=r.s.tied, R=tzRows(r), C=Math.ceil(L_/R);
    geo={L:L_,R,C,tied};
    A.ui.setPressed(seg,tzModel);
    const big=bgT(t), sh=n=>pctS(n/t), V=fmtInt(r.s.V), d=fmtInt(r.s.d);
    cap.innerHTML=tx(`All ${A.ui.big(t)} parameters, drawn to scale by area`,`全部 ${A.ui.big(t)} 个参数，按面积等比例画`);
    frame.setAttribute('aria-label',tied?tx(`Where ${m.name}’s ${big} parameters are, to scale by area: the embedding table, which the LM head reuses, and ${L_} layers of ${bgT(r.layer)} each.`,
                                            `${m.name} 的 ${big} 个参数在哪，按面积画：embedding 表（LM head 和它共享参数）和 ${L_} 层，每层 ${bgT(r.layer)} 个。`)
                                        :tx(`Where ${m.name}’s ${big} parameters are, to scale by area: the embedding table, ${L_} layers of ${bgT(r.layer)} each, and the LM head.`,
                                            `${m.name} 的 ${big} 个参数在哪，按面积画：embedding 表、${L_} 层（每层 ${bgT(r.layer)} 个）和 LM head。`));
    /* the two tables are columns of their share of the frame's width; the layers take the rest, 2 px from each table */
    const e=P.emb/t*100, h=P.head/t*100;
    emb.style.width=e.toFixed(4)+'%'; head.style.width=h.toFixed(4)+'%';
    lay.style.left=`calc(${e.toFixed(4)}% + 2px)`; lay.style.right=tied?'calc(0% + 0px)':`calc(${h.toFixed(4)}% + 2px)`;
    const embTip=tx(`The embedding table, ${an(V)} ${V} × ${d} matrix (vocabulary size × hidden size), holds ${bgT(P.emb)} parameters, ${sh(P.emb)} of the model’s ${big}.`,
                    `embedding 表是一个 ${V} × ${d} 的矩阵（词表大小 × 隐藏维度），有 ${bgT(P.emb)} 个参数，占模型 ${big} 个参数的 ${sh(P.emb)}。`)
                 +(tied?tx(' The LM head reuses the embedding table’s parameters, so it adds none.','LM head 直接用 embedding 表的参数，不另加参数。'):'');
    emb.dataset.tipx=embTip; emb.setAttribute('aria-label',embTip);
    if(tied){delete head.dataset.tipx; head.removeAttribute('aria-label'); head.setAttribute('aria-hidden','true'); head.tabIndex=-1;}
    else{const hTip=tx(`The LM head, ${an(d)} ${d} × ${V} matrix (hidden size × vocabulary size), holds ${bgT(P.head)} parameters, ${sh(P.head)} of the model’s ${big}.`,
                       `LM head 是一个 ${d} × ${V} 的矩阵（隐藏维度 × 词表大小），有 ${bgT(P.head)} 个参数，占模型 ${big} 个参数的 ${sh(P.head)}。`);
      head.dataset.tipx=hTip; head.setAttribute('aria-label',hTip); head.removeAttribute('aria-hidden');}
    /* the layers: one cell each, numbered column by column from the top left, so they follow the token from the embedding table to the LM head */
    const cellTip=i=>tx(`Layer ${i} of ${L_} holds ${bgT(r.layer)} parameters, ${sh(r.layer)} of the model’s ${big}. All ${L_} layers have the same shapes, so each holds the same number.`,
                        `第 ${i} 层（共 ${L_} 层）有 ${bgT(r.layer)} 个参数，占模型 ${big} 个参数的 ${sh(r.layer)}。${L_} 层的形状都一样，所以每层的参数一样多。`);
    const anim=how!=='still', old=[...lay.querySelectorAll('.tf-tz-grid:not(.tf-tz-old)')];
    const g=document.createElement('div'); g.className='tf-tz-grid'+(how==='go'?' tf-tz-go':(how==='wait'?' tf-tz-wait':''));
    g.style.gridTemplateColumns=`repeat(${C},minmax(0,1fr))`; g.style.gridTemplateRows=`repeat(${R},minmax(0,1fr))`;
    g.innerHTML=Array.from({length:L_},(_,i)=>{const tip=esc(cellTip(i+1));
      return `<span class="tf-tz-cell tf-pat-layer" role="img" tabindex="-1" data-tipx="${tip}" aria-label="${tip}"${anim?` style="animation-delay:${Math.round(TZ_LEAD+i*TZ_SPAN/L_)}ms"`:''}></span>`;}).join('');
    old.forEach(o=>{if(how==='go'){o.classList.add('tf-tz-old'); o.setAttribute('aria-hidden','true'); setTimeout(()=>o.remove(),TZ_LEAD+40);} else o.remove();});
    lay.appendChild(g);
    frame.classList.toggle('tf-tz-pend',how==='wait');
    roam(tzAt,false);
    /* the legend, in the picture's order. Each item's tooltip says what that part of the model does (home review v4 A #7: these
       definitions were the first note); the blocks' own tooltips say what each one holds. Hovering or focusing an item also lightens the
       rest of the picture. The attention block in the home hero's words, "from itself and the tokens before it" (A #20, B-19). Nothing
       here says how a layer's parameters divide between its two blocks: the lab's part 2 asks (R3.8) */
    const li=(k,sw,text,tip)=>`<span data-k="${k}" tabindex="0" data-tipx="${esc(tip)}"><i class="sw ${sw}"></i>${esc(text)}</span>`;
    const embDef=tx('The embedding table turns each token ID into a vector: it has one row for every token in the vocabulary, and the model looks up the token’s row.',
                    'embedding 表把每个 token ID 变成一个向量：词表里每个 token 占一行，模型查出这个 token 的那一行。');
    const headDef=tx('The LM head (LM: language model) turns the final vector into one score for every token in the vocabulary.',
                     'LM head（LM 即语言模型）把最后的向量变成词表里每个 token 的一个得分。');
    const layDef=tx('Each layer has an attention block, where each token takes information from itself and the tokens before it, and an MLP block (multi-layer perceptron), which transforms each token on its own.',
                    '每层有一个 attention 块（每个 token 在这里从它自己和前面的 token 取信息）和一个 MLP 块（多层感知机，单独变换每个 token）。');
    leg.innerHTML=li('emb','tf-sw-emb',tied?tx('embedding table (tied to the LM head)','embedding 表（和 LM head 共享参数）'):tx('embedding table','embedding 表'),
                     tied?tx(`The embedding table turns each token ID into a vector, and the LM head (LM: language model) turns the final vector into one score for every token in the vocabulary. In ${m.name} the LM head reuses the embedding table’s parameters.`,
                             `embedding 表把每个 token ID 变成一个向量，LM head（LM 即语言模型）把最后的向量变成词表里每个 token 的一个得分。${m.name} 的 LM head 直接用 embedding 表的参数。`):embDef)
                 +li('lay','tf-sw-layer',tx(`${L_} layers, one per cell`,`${L_} 层，每格一层`),layDef)+(tied?'':li('head','tf-sw-head','LM head',headDef));
    /* R6.20: one key number, the vocabulary's share; the layers hold the rest (the final RMSNorm, under 0.001%, is left out, note 2) */
    const v0=pct0((P.emb+P.head)/t);
    rd.innerHTML=NB(tied?tx(`${m.name}: the embedding table, whose parameters the LM head reuses, holds <b>${v0}</b> of its parameters; its ${L_} layers hold the rest.`,
                            `${m.name} 的参数里，<b>${v0}</b> 在 embedding 表（LM head 和它共享参数），其余都在 ${L_} 层里。`)
                        :tx(`${m.name}: the embedding table and LM head hold <b>${v0}</b> of its parameters; its ${L_} layers hold the rest.`,
                            `${m.name} 的参数里，<b>${v0}</b> 在 embedding 表和 LM head，其余都在 ${L_} 层里。`));
    /* "Assumptions & sources" (R11.5, home review v4 A #7): two notes, one question each. Where do the counts come from? (config.json;
       review B-04: it gives the sizes, and Llama's design turns them into matrix shapes) What does the picture leave out? */
    const N=[tx(`Every count is exact: the sizes in each model’s config.json on Hugging Face fix the shape of every matrix in Llama’s design, so ${m.name} has exactly ${fmtInt(t)} parameters.`,
                `每个参数量都是精确的：按 Llama 的结构，各模型在 Hugging Face 上的 config.json 里的维数决定每个矩阵的形状，所以 ${m.name} 正好有 ${fmtInt(t)} 个参数。`),
             tx('The picture leaves out the final RMSNorm, under 0.001% of the parameters, and how each layer’s parameters divide between its attention and MLP blocks, which the lab asks you to predict.',
                '图里没画最后的 RMSNorm（不到参数的 0.001%），也没画每层的参数在 attention 块和 MLP 块之间怎么分：这是 Lab 让你先预测的问题。')];
    notes.forEach((n,i)=>{n.textContent=N[i]||'';});};
  /* the build-up, once per page: the embedding table, then the layers one after another, then the LM head (transformer.css) */
  const play=()=>{tzIntro=true; if(tzIO){tzIO.disconnect(); tzIO=null;} const g=lay.querySelector('.tf-tz-grid:not(.tf-tz-old)');
    frame.classList.remove('tf-tz-pend'); frame.classList.add('tf-tz-intro'); if(g){g.classList.remove('tf-tz-wait'); g.classList.add('tf-tz-go');}
    setTimeout(()=>frame.classList.remove('tf-tz-intro'),TZ_LEAD+TZ_SPAN+TZ_CELL+400);};
  root.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button[data-v]'); if(!b||!seg.contains(b)||b.dataset.v===tzModel) return;
    tzModel=b.dataset.v; if(!tzIntro){tzIntro=true; if(tzIO){tzIO.disconnect(); tzIO=null;}}
    upd(A.reduceMotion()?'still':'go'); A.announce(rd.textContent);});
  frame.addEventListener('keydown',e=>{const it=items(), i=it.indexOf(document.activeElement); if(i<0) return;
    const {L:n,R,C,tied}=geo, c=i-1; let j;   // item 0 is the embedding table, items 1…n the layers, item n+1 the LM head
    switch(e.key){
      case 'ArrowDown': j=i+1; break;
      case 'ArrowUp': j=i-1; break;
      case 'ArrowRight': j=i===0?1:(i<=n?(c+R<n?i+R:(tied?i:n+1)):i); break;
      case 'ArrowLeft': j=i===0?0:(i<=n?(c-R>=0?i-R:0):1+(C-1)*R); break;
      case 'Home': j=0; break;
      case 'End': j=it.length-1; break;
      default: return;}
    e.preventDefault(); roam(j,true);});
  frame.addEventListener('focusin',e=>{const i=items().indexOf(e.target); if(i>=0&&i!==tzAt) roam(i,false);});
  const hl=k=>{if(k) frame.dataset.hl=k; else delete frame.dataset.hl;};
  leg.addEventListener('mouseover',e=>{const s=e.target.closest('[data-k]'); hl(s&&s.dataset.k);});
  leg.addEventListener('mouseleave',()=>hl(null));
  leg.addEventListener('focusin',e=>{const s=e.target.closest('[data-k]'); hl(s&&s.dataset.k);});
  leg.addEventListener('focusout',()=>hl(null));
  if(!tzIntro&&!A.reduceMotion()&&'IntersectionObserver' in window){
    upd('wait');
    tzIO=new IntersectionObserver(es=>{if(es.some(x=>x.isIntersecting&&x.intersectionRatio>=0.6)) play();},{threshold:[0.6]});
    tzIO.observe(frame);}
  else {tzIntro=true; upd('still');}
}});

/* read-only hook for the golden-number tests */
window.__tf=Object.freeze({compute:s=>compute({...DEFAULT,...s}),MODELS,DEFAULT,CH:Object.freeze({...CH,budget:CH_BUDGET,gpu:CH_GPU,best:CH_BEST,rows:CH_ROWS,min:CH_MIN,max:CH_MAX}),
  LV:LV.map(l=>({id:l.id,p:{...l.p}})),STEP_IDS:STEPS.map(s=>s.id),WALK,TOK,stops:{D_STOPS,F_STOPS,V_STOPS,H_OPTS,KV_OPTS},
  fmt:{pctS,vocShare,g1,g3},makeDrill,chPoint,state:()=>({...S,mode,step,pred,walk,revealed,masked:MASKED(),seenMistral,lvTried:[...lvTried],chSubmitted,chReveal})});
})();
