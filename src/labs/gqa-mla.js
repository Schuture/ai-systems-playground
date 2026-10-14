/* GQA and MLA (the fifth lab of the Transformer path: architecture, transformer, mha, kv-basics, gqa-mla): how sharing keys and values between heads shrinks the KV cache,
   what it costs, and what tensor parallelism across 8 GPUs does to each design. Analytical: every number is a formula. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='gqa-mla';
const KiB=A.KiB, MiB=A.MiB, GiB=A.GiB;

/* ================= engine: pure functions of the state ================= */
/* model shapes from each config.json (specs/transformer-facts.md A2, A4, A5, A8); the parameter counts are the Hugging Face totals,
   the same as the KV cache memory lab's (A.kv.MODELS), so both labs give the same weights and the same users */
const MODELS={
 'llama31-8b':{name:'Llama-3.1-8B',params:8030261248,hidden:4096,layers:32,heads:32,kvHeads:8,headDim:128,native:'gqa8'},
 'llama31-70b':{name:'Llama-3.1-70B',params:70553706496,hidden:8192,layers:80,heads:64,kvHeads:8,headDim:128,native:'gqa8'},
 'llama31-405b':{name:'Llama-3.1-405B',params:405853388800,hidden:16384,layers:126,heads:128,kvHeads:8,headDim:128,native:'gqa8'},
 'deepseek-v3':{name:'DeepSeek-V3',params:671e9,active:37e9,hidden:7168,layers:61,heads:128,headDim:128,kvLora:512,ropeDim:64,native:'mla'}
};
/* MLA's sizes in DeepSeek-V2 and V3 (DeepSeek-V2 §3.1.2; V3 config.json): used for the what-if MLA in a Llama model */
const MLA_C=512, MLA_R=64;
/* Falcon-7B, the real MQA model named in the MQA definition (config.json and model card, facts A7) */
const FALCON={layers:32,heads:71,kv:1,headDim:64};
/* FP8 here is 1 byte per parameter. Real FP8 checkpoints are larger (review B r12 N9). Meta's own Llama-3.1-405B-Instruct-FP8 keeps
   every attention matrix, the MLPs of the first and last layers, the embedding and the output layer in BF16 (modules_to_not_convert
   in its config.json); its safetensors hold 81,314,922,496 BF16, 324,538,466,304 FP8 and 15,237,120 FP32 numbers (Hugging Face API,
   checked 2026-09-26). DeepSeek-V3's embedding and output layer are BF16 and every 128 x 128 block has an FP32 scale: about
   0.24 GiB more per GPU on 8 GPUs (review B r12 N9). */
const META405_FP8={bf16:81314922496,f8:324538466304,f32:15237120}, DSV3_EXTRA=0.24;
const DESIGNS=['mha','gqa16','gqa8','gqa4','mqa','mla'];
const kvHeadsOf=(m,a)=>({mha:m.heads,gqa16:16,gqa8:8,gqa4:4,mqa:1}[a]||0);
const GPUS={h100:{name:'H100 80GB',mem:80},h200:{name:'H200 141GB',mem:141}};
const PREC={bf16:2,fp8:1}, PL={bf16:'BF16',fp8:'FP8'};
/* fixed in this lab: the KV cache memory lab's initial setup (memory utilization, runtime overhead per GPU, KV cache in BF16) */
const UTIL=0.9, OH=2, KVB=2;
const CTXS=[1024,2048,4096,8192,16384,32768,65536,131072];
const NS=[1,2,4,8];
/* the initial setup starts from multi-head attention, which the reader knows from the earlier labs (review 8: not from GQA-8) */
const INIT={model:'llama31-70b',attn:'mha',gpu:'h100',n:8,wp:'bf16',ctx:131072};
const SKEYS=['model','attn','gpu','n','wp','ctx'];
const CH_BASE={model:'llama31-70b',gpu:'h100',n:8,wp:'bf16',ctx:131072}, GOAL=10;
/* the GQA paper, Table 1 (Ainslie et al., 2023; facts G): average dev score over 7 tasks and inference time per sample, T5-XXL only,
   the three designs this lab compares (review 8: no other model's row, no other paper detail) */
const GQA_T1=[{d:'mha',score:47.2,time:1.51},{d:'gqa8',score:47.1,time:0.28},{d:'mqa',score:46.6,time:0.24}];
/* "under a fifth": the largest k with both GQA-8's and MQA's time under 1/k of MHA's */
const fracK=()=>{const [mh,g8,mq]=GQA_T1; return Math.floor(Math.min(mh.time/g8.time,mh.time/mq.time));};
const fracW=k=>tx({2:'half',3:'a third',4:'a quarter',5:'a fifth',6:'a sixth'}[k]||`1/${k}`,{2:'二分之一',3:'三分之一',4:'四分之一',5:'五分之一',6:'六分之一'}[k]||`${k} 分之一`);

function compute(s){
  const m=MODELS[s.model], a=s.attn, g=GPUS[s.gpu];
  const usable=g.mem*GiB*UTIL, oh=OH*GiB, wTotal=m.params*PREC[s.wp], wPerGpu=wTotal/s.n, budget=usable-wPerGpu-oh;
  const base={s,m,g,usable,oh,wTotal,wPerGpu,budget,weightsFit:budget>0};
  if(!a) return {...base,none:true,users:0};
  const mla=a==='mla', G=mla?0:kvHeadsOf(m,a);
  const dc=mla?(m.kvLora||MLA_C):0, dr=mla?(m.ropeDim||MLA_R):0;
  const elems=mla?dc+dr:2*G*m.headDim;                        // numbers each token stores per layer
  const perTok=mla?m.layers*(dc+dr)*KVB:2*m.layers*G*m.headDim*KVB;   // bytes per token, all layers (the KV cache memory lab's order)
  const shard=mla?1:Math.min(s.n,G);                         // KV heads split across GPUs only down to one per GPU; MLA not split
  const perGpuTok=perTok/shard, kvPerReq=perTok*s.ctx, perUserGpu=kvPerReq/shard;
  const users=budget>0?Math.floor(budget*shard/kvPerReq+1e-9):0;     // the KV cache memory lab's formula, term by term
  const copies=mla?s.n:Math.max(1,s.n/G), headsPerGpu=mla?0:Math.max(1,G/s.n);
  return {...base,a,mla,G,dc,dr,elems,perTok,shard,perGpuTok,kvPerReq,perUserGpu,users,copies,headsPerGpu,
    native:a===m.native,hypo:mla&&!m.kvLora,group:G?m.heads/G:0};
}
const designsOf=m=>m.native==='mla'?['mla']:DESIGNS;

/* ================= formatting: every number as shown, so formula lines redo (R6.6) ================= */
const int=n=>A.fmt.int(n);
const fmtD=(v,d)=>(+v).toLocaleString('en-US',{maximumFractionDigits:d});
/* exactly d decimals: weights and room in Show the math (15.0 GiB, 55.0 GiB) */
const fmtF=(v,d)=>(+v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
/* more than 6 digits: 3 significant digits with M or B (8.03B, 70.6B, 406B, 671B; R6.18) */
const big=n=>A.fmt.big(n);
/* an exact count over 6 digits, where HTML is allowed: the kit's A.ui.big ("Exactly N; shown rounded to 3 significant digits.", R6.18,
   R2.9). Not for DeepSeek-V3's 671B and 37B, which are the paper's round figures. */
const bigH=n=>A.ui.big(n);
const r=(v,d)=>{const k=10**d; return Math.round(v*k+1e-9)/k;};
const kib=b=>fmtD(b/KiB,3)+' KiB';
const mibIf=b=>b>=MiB?tx(` (${fmtD(b/MiB,3)} MiB)`,`（${fmtD(b/MiB,3)} MiB）`):'';
const gibS=(v,d)=>fmtD(v,d==null?1:d)+' GiB';
const ctxK=c=>A.fmt.tok(c);
const users=n=>A.fmt.users(n);
/* "1 user fits", "10 users fit" (review A r12 N10) */
const fitV=n=>n===1?'fits':'fit';
const gpus=n=>A.fmt.gpus(n);
/* model names and "8 × H100" stay on one line (the KV cache memory lab's rule); copying from the page gives plain characters again */
const NB=s=>String(s).replace(/\b(?:Llama-3\.1-\d+B|DeepSeek-V\d|Falcon-7B|GQA-\d+|T5-XXL)\b/g,m=>m.replace(/-/g,'‑')).replace(/(\d) × (?=H100|H200)/g,'$1 × ');
const DNAME={mha:'MHA',gqa16:'GQA-16',gqa8:'GQA-8',gqa4:'GQA-4',mqa:'MQA',mla:'MLA'};
/* review 8: each abbreviation with its full name where it first appears and in the hover of every design option, legend and chart label */
const FULL=a=>({mha:tx('multi-head attention','多头注意力'),mqa:'multi-query attention',mla:'multi-head latent attention'}[a]||'grouped-query attention');
const cap1=t=>t.charAt(0).toUpperCase()+t.slice(1);
/* a design's name. The real model's own design is named as such where it matters ("the design Llama-3.1-70B uses"), never "as designed"
   or 原设计: the lab starts from MHA, which the reader knows, not from the variant a real model happens to use (review 8, R2.8) */
const dName=(a,m)=>{m=m||MODELS[S.model]; if(a==='mla'&&m.native!=='mla') return tx(`MLA (${MLA_C} + ${MLA_R})`,`MLA（${MLA_C} + ${MLA_R}）`); return DNAME[a];};
const usesTxt=m=>tx(`the design ${m.name} uses`,`${m.name} 用的设计`);
/* "8 KV heads (GQA-8, the design Llama-3.1-70B uses)" for results; the challenge does not say which design Llama uses */
const kvLead=r1=>{const a=r1.a, G=r1.G, n=DNAME[a], nat=r1.native&&mode!=='challenge';
  return tx(`${G} KV ${G===1?'head':'heads'} (${n}${nat?`, ${usesTxt(r1.m)}`:''})`,`${G} 个 KV head（${n}${nat?`，${usesTxt(r1.m)}`:''}）`);};
/* the hover of a design option: its full name and what it stores, with a model's numbers (without them when m is null) */
function designTip(a,m){
  const H=m?m.heads:0;
  if(a==='mha') return m?tx(`MHA, multi-head attention: every query head has its own KV head, ${H} per layer.`,`MHA，多头注意力：每个 query head 都有自己的 KV head，每层 ${H} 个。`)
                        :tx('MHA, multi-head attention: every head has its own key and value.','MHA，多头注意力：每个 head 都有自己的 key 和 value。');
  if(a==='mqa') return m?tx(`MQA, multi-query attention: one KV head per layer, shared by all ${H} query heads.`,`MQA，multi-query attention：每层只有一个 KV head，${H} 个 query head 共用。`)
                        :tx('MQA, multi-query attention: all heads share one key and one value.','MQA，multi-query attention：所有 head 共用一个 key 和一个 value。');
  if(a==='mla'){const dc=(m&&m.kvLora)||MLA_C, dr=(m&&m.ropeDim)||MLA_R;
    return tx(`MLA, multi-head latent attention: one compressed vector and one RoPE key per token and layer (${dc} + ${dr} numbers) instead of a key and a value per head.`,`MLA，multi-head latent attention：每个 token 每层存一个压缩向量和一个 RoPE key（${dc} + ${dr} 维），代替各 head 的 key 和 value。`);}
  const g={gqa16:16,gqa8:8,gqa4:4}[a];
  return m?tx(`${DNAME[a]}, grouped-query attention with ${g} groups: ${g} KV heads per layer, each shared by ${H/g} query heads.`,`${DNAME[a]}，grouped-query attention，分 ${g} 组：每层 ${g} 个 KV head，每个由 ${H/g} 个 query head 共用。`)
          :tx(`${DNAME[a]}, grouped-query attention with ${g} groups: the heads of a group share one key and one value.`,`${DNAME[a]}，grouped-query attention，分 ${g} 组：同一组的 head 共用一个 key 和一个 value。`);
}
const times=(a,b)=>{const q=a/b; return Math.abs(q-Math.round(q))<1e-9?String(Math.round(q)):fmtD(q,1);};
const FRW={2:['half','一半'],4:['a quarter','四分之一'],8:['one eighth','八分之一'],16:['one sixteenth','十六分之一'],64:['one sixty-fourth','六十四分之一']};
/* "8 times", "twice", "half", "one eighth" against a baseline value (R6.6: who is compared with whom) */
function rel(v,base,what){
  if(Math.abs(v-base)<1e-9) return tx(`the same as ${what}`,`和${what}一样`);
  if(v>base){const k=times(v,base); return k==='2'?tx(`twice ${what}`,`是${what}的 2 倍`):tx(`${k} times ${what}`,`是${what}的 ${k} 倍`);}
  const k=times(base,v), fr=FRW[k];
  if(fr) return tx(`${fr[0]} of ${what}`,`是${what}的${fr[1]}`);
  return tx(`${what} is ${k} times as much`,`${what}是它的 ${k} 倍`);
}

/* ================= state (module scope: survives re-mounts and language switches) ================= */
let S={...INIT}, mode='guided', step=0, pred=null, observed={}, seen=false, userAct=false;
const tried5=new Set(), chTried=new Set();
let chPicked=false, chSubmitted=false, chReveal=false, savedS=null, drillK=0, drillOpen=false, mounted=false, tzD='mha', skipped=false;
/* the designs picked in each guided part: a part shows its result once anything in it was picked, the starting option included (R3.6) */
const triedAt={};
let root=null, host=null, prevShown={}, roW=null;

const atInit=()=>SKEYS.every(k=>k==='attn'||S[k]===INIT[k]);
/* parts 1–4 hide everything about GPUs and users: the prediction in part 2 is about them, and part 5 checks it (R3.8) */
const gpuHidden=()=>mode==='guided'&&step<=3;
const hideNote=()=>tx('shown in part 5','第 5 部分显示');

/* ================= definitions at first use (R2.8): full sentences with the numbers ================= */
const DEF=(text,tip)=>A.ui.def(text,tip);
const m70=()=>MODELS['llama31-70b'];
const T={
 /* the site's shared definitions (A.terms, R2.8: one term, one wording; CONTRACT §11), each with this lab's model sentence after it */
 mha:()=>{const m=m70(); return A.terms.mha()+tx(` In ${m.name}’s shape each layer has ${m.heads} heads, and each head’s query, key and value have ${m.headDim} numbers each.`,`${m.name} 的结构每层有 ${m.heads} 个 head，每个 head 的 query、key、value 各是 ${m.headDim} 维。`);},
 qh:()=>A.terms.qhead()+tx(` ${m70().name} has ${m70().heads} query heads per layer.`,`${m70().name} 每层有 ${m70().heads} 个 query head。`),
 kvh:()=>A.terms.kvhead()+tx(' In Llama-3.1-70B each key and each value has 128 numbers.','Llama-3.1-70B 里每个 key 和每个 value 都是 128 维。'),
 kvc:()=>A.terms.kvcache(),
 kib:()=>A.terms.kib()+tx(' Here 2,560 KiB is 2,621,440 bytes (2.5 MiB).','这里 2,560 KiB 就是 2,621,440 字节（2.5 MiB）。'),
 mqa:()=>{const f=2*FALCON.layers*FALCON.kv*FALCON.headDim*KVB;
   return A.terms.mqa()+tx(` Falcon-7B is a real example: its ${FALCON.heads} query heads share one KV head, ${kib(f)} per token in BF16.`,`Falcon-7B 就是这样：${FALCON.heads} 个 query head 共用一个 KV head，BF16 下每个 token ${kib(f)}。`);},
 /* the site's text, then this lab's model; neither says which keys and values each GPU keeps: part 4 asks the reader to predict it */
 tp:()=>A.terms.tp()+tx(` On ${INIT.n} GPUs, each computes ${m70().heads/INIT.n} of ${m70().name}’s ${m70().heads} query heads.`,`${INIT.n} 张 GPU 时，每张算 ${m70().name} ${m70().heads} 个 query head 里的 ${m70().heads/INIT.n} 个。`),
 gqa:()=>{const m=m70(); return A.terms.gqa()+tx(` ${m.name} has ${m.heads} query heads in ${m.kvHeads} groups of ${m.heads/m.kvHeads}, so each token stores ${m.kvHeads} keys and ${m.kvHeads} values per layer instead of ${m.heads} of each. Part 3 shows more of the paper’s numbers.`,
                                                  `${m.name} 的 ${m.heads} 个 query head 分 ${m.kvHeads} 组、每组 ${m.heads/m.kvHeads} 个，所以每个 token 每层只存 ${m.kvHeads} 个 key 和 ${m.kvHeads} 个 value，而不是各 ${m.heads} 个。第 3 部分给出论文的更多数字。`);},
 paper:()=>tx('Ainslie et al., 2023, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”, EMNLP 2023. It introduced GQA and a way to turn a trained MHA model into a GQA or MQA model.',
              'Ainslie 等，2023，“GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”，EMNLP 2023。它提出了 GQA，以及把训好的 MHA 模型改成 GQA 或 MQA 的办法。'),
 score:()=>tx('The paper’s average of seven dev-set scores (five summarization sets, WMT translation and TriviaQA question answering), each on a 0 to 100 scale.',
              '论文里 7 个开发集得分的平均（五个摘要数据集、WMT 翻译和 TriviaQA 问答），每项满分 100。'),
 time:()=>tx('The paper’s measured inference time for one input example, in seconds, in its own setup.','论文在自己的测试环境里测的、处理一个样本的推理用时（秒）。'),
 copy:()=>tx('With fewer KV heads than GPUs, one KV head serves query heads on several GPUs. Each GPU computes attention from its own memory, so each of those GPUs stores that KV head’s keys and values itself instead of fetching them from another GPU at every decode step. vLLM does this: with more GPUs than KV heads, it gives every GPU one KV head and copies each KV head as often as needed.',
             'KV head 比 GPU 少时，一个 KV head 要服务好几张 GPU 上的 query head。每张 GPU 都在自己的显存里算 attention，所以这几张 GPU 各自存一份它的 key 和 value，而不是每个 decode 步都去别的 GPU 上取。vLLM 就是这样做的：GPU 比 KV head 多时，每张 GPU 分一个 KV head，每个 KV head 按需要复制几份。'),
 mla:()=>A.terms.mla()+tx(' DeepSeek-V2 reports that, in its own comparisons, MLA scored better than MHA (§2.1.4).','DeepSeek-V2 报告说，在它自己的对比实验里，MLA 比 MHA 效果更好（§2.1.4）。'),
 latent:()=>tx(`The compressed vector (latent) is computed from the token’s vector (hidden size d) by one down-projection matrix. It has ${MLA_C} numbers in DeepSeek-V2 and V3, as many as the keys and values of two heads at 128 numbers per vector, and every head of the layer rebuilds its key and value from it.`,
               `压缩向量（latent）由 token 的向量（隐藏维度 d）乘一个下投影矩阵得到。DeepSeek-V2 和 V3 里它是 ${MLA_C} 维，相当于两个 head 的 key 和 value（每个 128 维）；这一层所有 head 的 key 和 value 都从它还原。`),
 rope:()=>tx(`The RoPE key is a ${MLA_R}-number key that carries the token’s position (RoPE, rotary position embedding, marks the position by rotating a key). MLA stores it next to the compressed vector and shares it across all heads; DeepSeek-V2 §2.1.3 explains why it cannot go inside the compressed vector.`,
             `RoPE key 是一个 ${MLA_R} 维、带着 token 位置的 key（RoPE 即旋转位置编码，靠旋转 key 来标记位置）。MLA 把它和压缩向量分开存，所有 head 共用；它为什么不能放进压缩向量，见 DeepSeek-V2 §2.1.3。`),
 /* what is stored does not depend on the folding; vLLM folds only at decode steps (review B r12 N3, review A r12 N8) */
 up:()=>tx('Up-projection matrices turn the compressed vector back into each head’s key and value, 128 numbers each. Only the compressed vector and the RoPE key are stored. At decode steps, engines such as vLLM fold these matrices into Wq and Wo and compare queries with the compressed vector directly (DeepSeek-V2 §2.1.2); the prefill, which runs the prompt through once, may still rebuild full keys and values, without storing them.',
           '上投影矩阵把压缩向量还原成每个 head 的 key 和 value，各 128 维。存下来的只有压缩向量和 RoPE key。decode 步里，vLLM 这类引擎把这些上投影矩阵并进 Wq 和 Wo，直接拿 query 和压缩向量比较（DeepSeek-V2 §2.1.2）；prefill（把 prompt 过一遍的那一步）时仍可能临时还原出完整的 key 和 value，但不存。'),
 /* why this lab's MLA (Llama's 80 layers on 8 × H100) differs from the KV cache memory lab's (DeepSeek-V3 itself on 8 × H200) */
 sizes:()=>{const d=MODELS['deepseek-v3'], dv=compute({model:'deepseek-v3',attn:'mla',gpu:'h200',n:8,wp:'fp8',ctx:INIT.ctx}), ml=compute({...INIT,attn:'mla'});
   return tx(`DeepSeek-V3’s sizes: a ${MLA_C}-number compressed vector and a ${MLA_R}-number RoPE key per layer. DeepSeek-V3 itself has ${d.layers} layers, so it stores ${kib(dv.perTok)} per token; the “KV cache memory” lab uses it as its MLA option, on 8 × H200 with FP8 weights. Here the same sizes go into Llama-3.1-70B’s ${m70().layers} layers: ${kib(ml.perTok)} per token.`,
             `DeepSeek-V3 的大小：每层一个 ${MLA_C} 维的压缩向量和一个 ${MLA_R} 维的 RoPE key。DeepSeek-V3 本身只有 ${d.layers} 层，每个 token 存 ${kib(dv.perTok)}，“KV cache 显存”Lab 里的 MLA 选项用的就是它（8 × H200，FP8 权重）。这里把同样的大小放进 Llama-3.1-70B 的 ${m70().layers} 层，每个 token 就是 ${kib(ml.perTok)}。`);},
 /* "plain tensor parallelism", the KV cache memory lab's phrase: the site's definition (model-free), this lab's model sentence
    (review A r13 M2), then what "plain" means and what the alternative is (M3) */
 plain:()=>{const m=m70(); return A.terms.tpmla()+tx(` For example, the ${DNAME[m.native]} that ${m.name} really uses has ${m.kvHeads} KV heads per layer, one per GPU on ${INIT.n} GPUs, 1/${INIT.n} of the total; this lab’s initial setup, MHA, has ${m.heads}, ${m.heads/INIT.n} per GPU. “Plain” means every layer is split across the GPUs, attention by head; the alternative keeps the other layers split but runs attention data-parallel.`,
                                                     `比如 ${m.name} 真实用的 ${DNAME[m.native]} 每层有 ${m.kvHeads} 个 KV head，${INIT.n} 张 GPU 正好每张存一个，即总量的 1/${INIT.n}；这个 Lab 的初始设置 MHA 有 ${m.heads} 个，每张存 ${m.heads/INIT.n} 个。“纯”指每一层都拆到各张 GPU 上，attention 按 head 拆；另一种做法是其余各层照样拆，attention 改做数据并行。`);},
 dp:()=>tx('Data-parallel attention gives each GPU the whole attention of a different set of users, so each user’s compressed vector sits on one GPU only. SGLang v0.4 added it for DeepSeek models as an option (--enable-dp-attention), off by default; its release notes say that tensor parallelism on 8 GPUs would duplicate their KV cache.',
           '数据并行 attention 让每张 GPU 负责一批不同用户的全部 attention，每个用户的压缩向量只存在一张 GPU 上。SGLang v0.4 为 DeepSeek 模型加了这个可选的做法（--enable-dp-attention，默认不开）；它的发布说明指出，在 8 张 GPU 上做张量并行会让它们的 KV cache 重复存。')
};

/* ================= the guided parts =================
   Review 8 (R2.8, R3.13): start from multi-head attention (MHA), which the earlier labs built; then grouped-query attention (GQA) as the
   improvement and multi-query attention (MQA) as the extreme, what sharing costs, a prediction about 8 GPUs and its check, then
   multi-head latent attention (MLA) as a different approach. GQA-8 is "the design Llama-3.1-70B uses", never the original design. */
const PRED=[{v:'same',t:{en:'1, the same as MHA',zh:'1 个，和 MHA 一样'}},{v:'few',t:{en:'2 to 20',zh:'2 到 20 个'}},{v:'mid',t:{en:'21 to 63',zh:'21 到 63 个'}},{v:'many',t:{en:'64 or more',zh:'64 个或更多'}}];
const predOf=u=>u<=1?'same':(u<=20?'few':(u<=63?'mid':'many'));
const KV_OPTS=()=>{const m=MODELS[INIT.model]; return [['mha',String(m.heads)],['gqa16','16'],['gqa8','8'],['gqa4','4'],['mqa','1']];};
const segKV=id=>A.ui.seg({id,key:'attn',label:tx(`KV heads per layer (${m70().heads} query heads)`,`每层 KV head 数（${m70().heads} 个 query head）`),options:KV_OPTS(),value:S.model===INIT.model?S.attn:'',inline:true});
const segMLA=id=>A.ui.seg({id,key:'attn',label:tx('Attention design','注意力设计'),options:[['mha','MHA'],['gqa8','GQA-8'],['mla',tx(`MLA (${MLA_C} + ${MLA_R})`,`MLA（${MLA_C} + ${MLA_R}）`)]],value:S.model===INIT.model?S.attn:'',inline:true});
const J=(key,txt)=>`<button type="button" class="linkbtn" data-jump="${key}">${txt}</button>`;
const changedTxt=()=>tx(`<p>You changed the setup under ${J('controls','Settings')}, so this is no longer the initial setup. Pick an option in this part to return to it.</p>`,
                        `<p>你在“${J('controls','自由调参')}”里改了设置，现在不是初始设置了。点这一部分里的任意选项就能回去。</p>`);
const I=()=>compute({...INIT});
const Ia=a=>compute({...INIT,attn:a});
const tried=k=>!!(triedAt[k]&&triedAt[k].size);
/* "2,560 KiB" with KiB defined where it first appears (R6.12) */
const kibU=b=>`${fmtD(b/KiB,3)} ${DEF('KiB',T.kib())}`;

const STEPS=[
 {title:{en:'Your mission',zh:'你的任务'},lead:{en:'With MHA, 1 user fits at 128K on 8 H100s. How far can KV heads be shared?',zh:'用 MHA，8 张 H100 每人 128K 只装得下 1 个用户。KV head 能共用到什么程度？'},enter:()=>({...INIT}),
  body:()=>{const mha=I();
   return tx(`<p>The ${A.shell.labLink('mha')} lab built ${DEF('multi-head attention (MHA)',T.mha())}: each head has its own query, key and value. This lab calls a head’s query side a ${DEF('query head',T.qh())} and its key and value a ${DEF('KV head',T.kvh())}, because the designs below let query heads share KV heads.</p>`+
     `<p>With MHA, each token keeps a key and a value for every head in every layer: in Llama-3.1-70B’s shape, ${kibU(mha.perTok)} of ${DEF('KV cache',T.kvc())} per token. On one 8 × H100 node, with weights and KV cache in BF16 and ${ctxK(INIT.ctx)} tokens of context per user, that leaves room for only ${users(mha.users)}.</p>`+
     `<p>Real models let query heads share KV heads. How far can the sharing go, what does it cost, and does it still save memory once the model is split across the ${INIT.n} GPUs?</p>`,
     `<p>“${A.shell.labLink('mha')}”Lab 搭的就是${DEF('多头注意力（MHA）',T.mha())}：每个 head 都有自己的 query、key 和 value。这个 Lab 把 head 的 query 这一边叫 ${DEF('query head',T.qh())}，把它的 key 和 value 叫 ${DEF('KV head',T.kvh())}，因为下面的设计会让几个 query head 共用一个 KV head。</p>`+
     `<p>用 MHA 时，每个 token 在每一层都要为每个 head 存一个 key 和一个 value：在 Llama-3.1-70B 的结构下，每个 token 要 ${kibU(mha.perTok)} 的 ${DEF('KV cache',T.kvc())}。在一台 8 × H100 上，权重和 KV cache 都是 BF16，每个用户 ${ctxK(INIT.ctx)} 个 token 的上下文，这样只装得下 ${mha.users} 个用户。</p>`+
     `<p>真实的模型会让几个 query head 共用一个 KV head。这种共用能走多远？代价是什么？模型分到 ${INIT.n} 张 GPU 上以后，它还省显存吗？</p>`);}},
 {title:{en:'Share keys and values',zh:'共用 key 和 value'},lead:{en:'GQA: a group of query heads shares one KV head; MQA: all share one.',zh:'GQA：一组 query head 共用一个 KV head；MQA：全部共用一个。'},enter:()=>({...INIT}),target:'gm-ic-kv',label:{en:'Observed',zh:'观察到'},
  expect:()=>!!observed[1],showMe:{...INIT,attn:'gqa8'},showObs:()=>tried(1)||!atInit(),
  body:()=>{const m=m70();
   return tx(`<p>In ${DEF('grouped-query attention (GQA)',T.gqa())}, the query heads are split into groups, and each group shares one KV head: Llama-3.1-70B uses ${m.kvHeads} groups of ${m.heads/m.kvHeads} (GQA-${m.kvHeads}). The extreme is a single group, where all the query heads share one KV head: ${DEF('multi-query attention (MQA)',T.mqa())}.</p><p>Start from MHA and give each layer fewer KV heads:</p>`,
             `<p>${DEF('grouped-query attention（GQA）',T.gqa())}把 query head 分成几组，同一组共用一个 KV head。Llama-3.1-70B 分 ${m.kvHeads} 组，每组 ${m.heads/m.kvHeads} 个（GQA-${m.kvHeads}）。最极端的是只分一组，所有 query head 共用一个 KV head，这就是 ${DEF('multi-query attention（MQA）',T.mqa())}。</p><p>从 MHA 出发，减少每层的 KV head：</p>`)+segKV('gm-ic-kv');},
  obs:r1=>{if(!atInit()||r1.mla) return changedTxt();
   /* R6.20: one size per result (on the chart and the metric card), its fraction of MHA's in words, and who shares */
   const b=I(), a=r1.a, lead=kvLead(r1), per=kib(r1.perTok);
   if(a===INIT.attn) return tx(`<p><b>${lead}: ${per} per token</b>, the initial setup. Every query head has its own key and value.</p>`,`<p><b>${lead}：每个 token ${per}</b>，这是初始设置。每个 query head 都有自己的 key 和 value。</p>`);
   const cmp=rel(r1.perTok,b.perTok,tx('MHA’s',' MHA '));
   const who=a==='mqa'?tx('All the query heads share it.','所有 query head 共用它。'):tx(`Every ${r1.group} query heads share one KV head.`,`每 ${r1.group} 个 query head 共用一个 KV head。`);
   return tx(`<p><b>${lead}: ${per} per token</b>, ${cmp}. ${who}</p>`,`<p><b>${lead}：每个 token ${per}</b>，${cmp}。${who}</p>`)+
     tx('<p>Only keys and values are shared: every query head still compares its own query with them.</p>','<p>共用的只是 key 和 value：每个 query head 还是拿自己的 query 去和它们比较。</p>');}},
 {title:{en:'What sharing costs',zh:'共用的代价'},lead:{en:'Sharing changes the model: the GQA paper’s quality and speed.',zh:'共用会改变模型：看 GQA 论文里的质量和速度。'},
  /* R6.20: the scores are in the table; the sentence says the trend in words (the same thresholds as part 7) */
  body:()=>{const [mh,g8,mq]=GQA_T1, close=mh.score-g8.score<0.3, little=mh.score-mq.score<1, fr=fracW(fracK());
   const sc=v=>fmtF(v,1), tm=v=>tx(`${fmtF(v,2)} s`,`${fmtF(v,2)} 秒`);
   const row=x=>`<tr><th scope="row">${DEF(DNAME[x.d],designTip(x.d,null))}</th><td class="mono">${sc(x.score)}</td><td class="mono">${tm(x.time)}</td></tr>`;
   const tb=`<table class="gm-t1"><caption>${tx('The GQA paper, Table 1 (T5-XXL)','GQA 论文表 1（T5-XXL）')}</caption><thead><tr><th scope="col">${tx('Design','设计')}</th><th scope="col">${DEF(tx('Average score (higher is better)','平均分（越高越好）'),T.score())}</th><th scope="col">${DEF(tx('Time per sample','每个样本用时'),T.time())}</th></tr></thead><tbody>`+GQA_T1.map(row).join('')+`</tbody></table>`;
   return tx(`<p>Sharing changes the model, so it has to be trained that way. The ${DEF('GQA paper',T.paper())} converted T5-XXL, a model trained with MHA, to GQA-8 and to MQA and trained each a little further; it did not test decoder-only models such as Llama:</p>`+tb+
     `<p>GQA-8 scored ${close?'almost as well as MHA':'a little below MHA'}, and MQA ${little?'a little lower':'lower'}. Both took under ${fr} of MHA’s time per sample: each decode step reads every earlier token’s keys and values from memory, and fewer KV heads leave less to read. Llama 3 uses GQA-8 in all three sizes.</p>`,
     `<p>共用 key 和 value 改变了模型，所以模型得按这种结构训练。${DEF('GQA 论文',T.paper())}把用 MHA 训好的 T5-XXL 分别改成 GQA-8 和 MQA，再各自稍加训练；论文没有测 Llama 这类 decoder-only 模型：</p>`+tb+
     `<p>GQA-8 的平均分${close?'几乎和 MHA 一样':'比 MHA 略低'}，MQA ${little?'略低一点':'更低'}。两者每个样本的用时都不到 MHA 的${fr}：每个 decode 步都要从显存读出之前所有 token 的 key 和 value，KV head 越少，要读的越少。Llama 3 的三个尺寸都用 GQA-8。</p>`);}},
 {title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: with MQA on the same 8 GPUs, how many users fit?',zh:'先猜：同样 8 张 GPU 换成 MQA，能装下几个用户？'},enter:()=>({...INIT}),gate:()=>!!pred,skip:true,
  body:()=>{const q=Ia('mqa'), mha=I(), k=Math.round(mha.perTok/q.perTok), ft=FRW[k]||[`1/${k}`,`${k} 分之一`];
   const ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}"${seen?' disabled':''}>${esc(L(o.t))}</button>`).join('')}</div>`;
   const note=seen?(pred?tx('Saved. Part 5 has checked it, so it can no longer be changed.','已记下。第 5 部分已经验证过，不能再改了。'):tx('Part 5 has already shown the answer.','第 5 部分已经给出答案。'))
     :(pred?tx('Saved. You can still change it before part 5 checks it.','已记下。第 5 部分验证之前，还可以改。'):'');
   return tx(`<p>Per token, MQA stores ${ft[0]} of what MHA stores. With MHA, only ${users(mha.users)} ${fitV(mha.users)} at ${ctxK(INIT.ctx)} each on the 8 × H100, whose GPUs split every layer between them (${DEF('tensor parallelism',T.tp())}).</p><p>With MQA on the same 8 GPUs, how many users fit?</p>`,
             `<p>按每个 token 算，MQA 只存 MHA 的${ft[1]}。用 MHA 时，8 × H100 每人 ${ctxK(INIT.ctx)} 上下文只装得下 ${mha.users} 个用户；这些 GPU 把每一层分开来算（${DEF('张量并行',T.tp())}）。</p><p>换成 MQA，还是这 8 张 GPU，能装下多少用户？</p>`)+ch+(note?`<p class="muted small">${note}</p>`:'');}},
 {title:{en:'Across 8 GPUs',zh:'分到 8 张 GPU 上'},lead:{en:'Share down to fewer KV heads than GPUs and watch each GPU’s share.',zh:'把 KV head 减到比 GPU 还少，看每张 GPU 存多少。'},enter:()=>({...INIT}),target:'gm-ic-kv5',label:{en:'Result',zh:'结果'},
  expect:()=>!!observed[4],showMe:{...INIT,attn:'mqa'},showObs:()=>tried(4)||!atInit(),
  body:()=>{const mha=I(), fq=FRW[INIT.n]||[`1/${INIT.n}`,`${INIT.n} 分之一`];
   return tx(`<p>With ${DEF('tensor parallelism',T.tp())}, the ${INIT.n} GPUs split attention by head: each computes ${fq[0]} of the query heads and stores the keys and values those heads read. With MHA, each GPU stores ${fq[0]} of the KV cache, ${kib(mha.perGpuTok)} per token.</p><p>Now share more, down to fewer KV heads than GPUs:</p>`,
             `<p>${DEF('张量并行',T.tp())}按 head 把 attention 分到 ${INIT.n} 张 GPU 上：每张算${fq[1]}的 query head，并存下这些 head 要读的 key 和 value。用 MHA 时，每张 GPU 存${fq[1]}的 KV cache，每个 token ${kib(mha.perGpuTok)}。</p><p>现在多共用一些，一直减到比 GPU 还少的 KV head：</p>`)+segKV('gm-ic-kv5');},
  obs:r1=>{if(!atInit()||r1.mla) return changedTxt();
   /* R6.20: the result is the users, with one comparison; what each GPU stores is said in words (the sizes are on the chart and the
      metric cards, the division in Show the math) */
   const b=I(), g8=Ia('gqa8'), a=r1.a, lead=kvLead(r1), U=r1.users;
   const vsG8=U===g8.users?tx('the same as with GQA-8','和 GQA-8 一样'):tx(`against ${g8.users} with GQA-8`,`GQA-8 是 ${g8.users} 个`);
   const kg=Math.round(b.perGpuTok/r1.perGpuTok), fg=FRW[kg]||[`1/${kg}`,`${kg} 分之一`], fq=FRW[INIT.n]||[`1/${INIT.n}`,`${INIT.n} 分之一`];
   if(a==='mqa'){let pf='';
     if(pred){const right=predOf(U), pt=L(PRED.find(x=>x.v===pred).t), rt=L(PRED.find(x=>x.v===right).t);
       pf=pred===right?tx(`<p class="okline">Your prediction, “${esc(pt)}”, was right.</p>`,`<p class="okline">你的预测“${esc(pt)}”是对的。</p>`)
                      :tx(`<p>You predicted “${esc(pt)}”; the right answer is “${esc(rt)}”.</p>`,`<p>你预测的是“${esc(pt)}”，正确答案是“${esc(rt)}”。</p>`);}
     const kt=Math.round(b.perTok/r1.perTok), ft=FRW[kt]||[`1/${kt}`,`${kt} 分之一`];
     return tx(`<p><b>${lead}: at most ${users(U)} ${fitV(U)}, ${vsG8}.</b></p>`,
               `<p><b>${lead}：最多装下 ${U} 个用户，${vsG8}。</b></p>`)+pf+
       tx(`<p>Why: each GPU computes attention for its own query heads from its own memory at every decode step, and all of them read that one KV head, so ${DEF('each GPU keeps a copy',T.copy())}. Per token MQA stores ${ft[0]} of what MHA stores, but on each GPU it is still ${fg[0]}.</p>`,
          `<p>原因：每张 GPU 在每个 decode 步都用自己显存里的 key 和 value，给自己的 query head 算 attention；这些 query head 都要读这一个 KV head，所以${DEF('每张 GPU 各存一份',T.copy())}。按每个 token 算，MQA 只存 MHA 的${ft[1]}；可在每张 GPU 上，仍然是 MHA 的${fg[1]}。</p>`);}
   if(r1.G<INIT.n){const nudge=observed[4]?'':(pred?tx('<p>Now pick 1 KV head (MQA) to check your prediction.</p>','<p>再选 1 个 KV head（MQA），验证你的预测。</p>')
                                                  :tx('<p>Now pick 1 KV head (MQA) and see how many users fit on the same 8 GPUs.</p>','<p>再选 1 个 KV head（MQA），看看同样 8 张 GPU 能装下多少个用户。</p>'));
     return tx(`<p><b>${lead}: at most ${users(U)} ${fitV(U)}, ${vsG8}.</b> Each KV head now serves query heads on ${r1.copies} GPUs, so each of them keeps a copy, and every GPU still stores a whole KV head.</p>`,
               `<p><b>${lead}：最多装下 ${U} 个用户，${vsG8}。</b>现在每个 KV head 要服务 ${r1.copies} 张 GPU 上的 query head，这 ${r1.copies} 张 GPU 各存一份，每张 GPU 仍然存一整个 KV head。</p>`)+nudge;}
   if(a===INIT.attn) return tx(`<p><b>${lead}: each GPU stores ${fq[0]} of them, and at most ${users(U)} ${fitV(U)}.</b> This is the initial setup.</p>`,
                               `<p><b>${lead}：每张 GPU 存其中的${fq[1]}，最多装下 ${U} 个用户。</b>这是初始设置。</p>`);
   return tx(`<p><b>${lead}: each GPU stores ${fg[0]} as much as with MHA, so at most ${users(U)} ${fitV(U)}, against ${users(b.users)} with MHA.</b></p>`,
             `<p><b>${lead}：每张 GPU 存的只有 MHA 的${fg[1]}，所以最多装下 ${U} 个用户，MHA 只装下 ${b.users} 个。</b></p>`);}},
 {title:{en:'MLA keeps one compressed vector',zh:'MLA：存一个压缩向量'},lead:{en:'MLA stores one compressed vector per token and layer, not a key and a value per head.',zh:'MLA 每个 token 每层只存一个压缩向量，不为每个 head 存 key 和 value。'},enter:()=>({...INIT}),target:'gm-ic-mla',label:{en:'Result',zh:'结果'},
  expect:()=>!!observed[5],showMe:{...INIT,attn:'mla'},showObs:()=>tried(5)||!atInit(),
  body:()=>{const hd=m70().headDim, eq=fmtD((MLA_C+MLA_R)/(2*hd),2), n=MLA_C+MLA_R;
   /* R6.20: the one number in the sentence is the comparison; how it is worked out is on hover */
   const eqTip=tx(`The compressed vector’s ${MLA_C} numbers and the RoPE key’s ${MLA_R} make ${n} numbers per token and layer. A KV head stores a key and a value of ${hd} numbers each, ${2*hd} in all, so ${n} numbers are as much as ${n} ÷ ${2*hd} = ${eq} KV heads.`,
                  `压缩向量的 ${MLA_C} 维加上 RoPE key 的 ${MLA_R} 维，每个 token 每层一共 ${n} 维。一个 KV head 存 ${hd} 维的 key 和 ${hd} 维的 value，共 ${2*hd} 维，所以 ${n} 维相当于 ${n} ÷ ${2*hd} = ${eq} 个 KV head。`);
   return tx(`<p>${DEF('Multi-head latent attention (MLA)',T.mla())}, used in DeepSeek-V2 and V3, takes another approach: it stores no key and value per head. Per token and layer it stores one ${DEF('compressed vector (latent)',T.latent())} of ${MLA_C} numbers and a ${MLA_R}-number ${DEF('RoPE key',T.rope())}, and every head’s key and value are ${DEF('rebuilt',T.up())} from them.</p>`+
     `<p>Together that is as much as ${DEF(`${eq} KV heads`,eqTip)} store. What if Llama-3.1-70B’s shape used MLA with ${DEF('these sizes',T.sizes())}?</p>`,
     `<p>DeepSeek-V2 和 V3 用的 ${DEF('multi-head latent attention（MLA）',T.mla())}换了一种做法：它不为每个 head 存 key 和 value。每个 token 每层只存一个 ${MLA_C} 维的${DEF('压缩向量（latent）',T.latent())}和一个 ${MLA_R} 维的 ${DEF('RoPE key',T.rope())}，各个 head 的 key 和 value 都从它们${DEF('还原',T.up())}出来。</p>`+
     `<p>两者加起来，相当于 ${DEF(`${eq} 个 KV head`,eqTip)} 存的量。如果 Llama-3.1-70B 的结构用${DEF('同样大小',T.sizes())}的 MLA 呢？</p>`)+segMLA('gm-ic-mla');},
  obs:r1=>{if(!atInit()||!['mha','gqa8','mla'].includes(r1.a)) return changedTxt();
   /* R6.20: the users are the one number of each result; what each GPU keeps is said in words (the sizes are on the chart and the metric
      cards); the reason in words */
   const b=I(), g8=Ia('gqa8'), fr=x=>{const k=Math.round(x.perTok/x.perGpuTok); return FRW[k]||[`1/${k}`,`${k} 分之一`];};
   if(r1.a==='mha') return tx(`<p><b>MHA, the initial setup: each GPU stores ${fr(b)[0]} of its KV cache, and at most ${users(b.users)} ${fitV(b.users)}.</b></p>`,
                              `<p><b>MHA（初始设置）：每张 GPU 存它 KV cache 的${fr(b)[1]}，最多装下 ${b.users} 个用户。</b></p>`);
   if(r1.a==='gqa8') return tx(`<p><b>GQA-8, ${usesTxt(r1.m)}: each GPU stores ${fr(g8)[0]} of its KV cache, and at most ${users(g8.users)} ${fitV(g8.users)}.</b></p>`,
                               `<p><b>GQA-8（${usesTxt(r1.m)}）：每张 GPU 存它 KV cache 的${fr(g8)[1]}，最多装下 ${g8.users} 个用户。</b></p>`);
   const less=r1.perTok<g8.perTok, fewer=r1.users<g8.users;
   return tx(`<p><b>MLA: ${less?'much less KV cache per token than GQA-8':'per token about as much as GQA-8'}, but each GPU keeps all of it, so at most ${users(r1.users)} ${fitV(r1.users)}, ${fewer?`fewer than GQA-8’s ${g8.users}`:`against ${g8.users} with GQA-8`}.</b></p>`+
     `<p>Why: every head rebuilds its key and value from the whole compressed vector, so no GPU can work from a slice of it. Like the single KV head of MQA, the compressed vector is copied to all ${INIT.n} GPUs under ${DEF('plain tensor parallelism',T.plain())}, while GQA-8’s KV cache splits evenly across them.</p>`,
     `<p><b>MLA：每个 token 的 KV cache ${less?'比 GQA-8 少得多':'和 GQA-8 差不多'}，但每张 GPU 都要存全部，所以最多装下 ${r1.users} 个用户，${fewer?`比 GQA-8 的 ${g8.users} 个少`:`GQA-8 是 ${g8.users} 个`}。</b></p>`+
     `<p>原因：每个 head 都要用整个压缩向量来还原自己的 key 和 value，所以哪张 GPU 都不能只存其中一段。压缩向量和 MQA 唯一的 KV head 一样，${DEF('纯张量并行',T.plain())}时要复制到全部 ${INIT.n} 张 GPU 上。GQA-8 的 KV cache 却能平均分到各张 GPU 上。</p>`);}},
 {title:{en:'What to remember',zh:'要记住的'},enter:()=>({...INIT}),
  /* R6.20: the principles in words; the per-design sizes are on the chart and in Show the math, the paper's scores in part 3's table */
  body:()=>{const b=I(), g8=Ia('gqa8'), q=Ia('mqa'), g4=Ia('gqa4'), ml=Ia('mla'), [t1,t2,t3]=GQA_T1, fr=fracW(fracK());
   const close=t1.score-t2.score<0.3, little=t1.score-t3.score<1;
   const same=q.perGpuTok===g8.perGpuTok&&g4.perGpuTok===g8.perGpuTok, sameU=q.users===g8.users;
   return tx(`<ul><li>KV cache per token = 2 (a key and a value) × layers × KV heads × head_dim × bytes per number. MHA keeps a KV head for every query head; ${DEF('GQA',T.gqa())} lets a group share one, and ${DEF('MQA',T.mqa())} shares one for all, so the cache per token shrinks in proportion to the KV heads.</li>`+
     `<li>In the GQA paper’s T5-XXL tests, GQA-8 scored ${close?'almost as well as MHA':'a little below MHA'} on average and MQA ${little?'a little lower':'lower'}, and both took under ${fr} of MHA’s time per sample.</li>`+
     `<li>${DEF('Tensor parallelism',T.tp())} gives every GPU at least one whole KV head, so with fewer KV heads than GPUs they are copied. ${DEF('MLA',T.mla())} stores far less per token than MHA or GQA-8, yet every GPU keeps all of it: every head needs the whole compressed vector.</li></ul>`+
     `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Fewer KV heads always save memory.” Not below one KV head per GPU: GQA-4 and MQA store ${same?'as much on each GPU as GQA-8':`${kib(q.perGpuTok)} per token on each GPU, against GQA-8’s ${kib(g8.perGpuTok)}`}, so ${sameU?'no more users fit':`${q.users} users fit instead of ${g8.users}`}.</p>`+
     `<p>✗ “MLA always uses less memory than GQA-8.” Per token it does, but under ${DEF('plain tensor parallelism',T.plain())} each GPU keeps all of it, while GQA-8 splits across the GPUs. So in this lab ${ml.users} users fit instead of ${g8.users}.</p></div>`+A.shell.toScenarios(),
     `<ul><li>每个 token 的 KV cache = 2（key 和 value）× 层数 × KV head 数 × head_dim × 每个数值的字节数。MHA 给每个 query head 一个 KV head；${DEF('GQA',T.gqa())} 让一组共用一个，${DEF('MQA',T.mqa())} 全部共用一个，所以每个 token 的 KV cache 随 KV head 数成比例缩小。</li>`+
     `<li>在 GQA 论文的 T5-XXL 实验里，GQA-8 的平均分${close?'几乎和 MHA 一样':'比 MHA 略低'}，MQA ${little?'略低一点':'更低'}，两者每个样本的用时都不到 MHA 的${fr}。</li>`+
     `<li>${DEF('张量并行',T.tp())}至少给每张 GPU 一整个 KV head，所以 KV head 比 GPU 少时就要复制。${DEF('MLA',T.mla())} 每个 token 比 MHA 和 GQA-8 都少存很多，但每个 head 都要用整个压缩向量，所以每张 GPU 都要存全部。</li></ul>`+
     `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “KV head 越少越省显存。”KV head 比 GPU 少以后就不是这样：GQA-4 和 MQA 每张 GPU 存的${same?'和 GQA-8 一样多':`还是 ${kib(q.perGpuTok)}，GQA-8 是 ${kib(g8.perGpuTok)}`}，${sameU?'装下的用户也不会更多':`能装下 ${q.users} 个用户，GQA-8 是 ${g8.users} 个`}。</p>`+
     `<p>✗ “MLA 在哪都比 GQA-8 省显存。”每个 token 确实更少，但${DEF('纯张量并行',T.plain())}时每张 GPU 都要存全部，GQA-8 却能分到各张 GPU 上。所以在这个 Lab 的假设里只能装下 ${ml.users} 个用户，而不是 ${g8.users} 个。</p></div>`+A.shell.toScenarios());}}
];
const DESCS=()=>({
 guided:tx('Seven short parts. Start from multi-head attention (MHA), let heads share keys and values (grouped-query and multi-query attention, GQA and MQA) and see what that costs, predict and check what happens on 8 GPUs, then meet multi-head latent attention (MLA). Start here if this is new to you.',
           '一共 7 个部分。从多头注意力（MHA）出发，让 head 共用 key 和 value（grouped-query attention 和 multi-query attention，即 GQA 和 MQA），看看代价，再预测并验证分到 8 张 GPU 上会怎样，最后认识 multi-head latent attention（MLA）。第一次学从这里开始。'),
 sandbox:tx('Five scenarios, from one H100 to 8 × H200: Llama-3.1-70B’s shape with multi-head attention (MHA), three Llama models with the grouped-query attention (GQA) they use, and DeepSeek-V3 with multi-head latent attention (MLA). Pick one to see what its attention stores per token, what each GPU keeps, and how many users fit.',
            '五个场景，从一张 H100 到 8 × H200，包括用多头注意力（MHA）的 Llama-3.1-70B 结构、用 grouped-query attention（GQA）的三个 Llama 模型，以及用 multi-head latent attention（MLA）的 DeepSeek-V3。选一个，看它的 attention 每个 token 存多少、每张 GPU 存多少、能装下几个用户。'),
 challenge:tx(`A model with Llama-3.1-70B’s shape serves users with ${ctxK(CH_BASE.ctx)} of context each on 8 × H100. Choose only its attention design: at least ${GOAL} users must fit, with as many KV heads as possible.`,
              `一个 Llama-3.1-70B 结构的模型，在 8 × H100 上服务每人 ${ctxK(CH_BASE.ctx)} 上下文的用户。只选它的注意力设计：至少同时装下 ${GOAL} 个用户，KV head 越多越好。`)
});

/* ================= scenarios: real deployments, 128K of context per user ================= */
/* the initial setup (MHA) first, then Llama-3.1-70B with the GQA-8 it uses, then the other real models (review 8) */
const PRESETS=[
 {t:{en:'Llama-3.1-70B’s shape with MHA',zh:'Llama-3.1-70B 的结构用 MHA'},d:{en:'Multi-head attention on 8 × H100, one KV head per query head: the initial setup',zh:'多头注意力，8 × H100，每个 query head 一个 KV head：初始设置'},p:{...INIT}},
 {t:{en:'Llama-3.1-70B on 8 × H100',zh:'8 × H100 跑 Llama-3.1-70B'},d:{en:'Grouped-query attention, the design it uses: 8 KV heads (GQA-8)',zh:'grouped-query attention，它用的设计：8 个 KV head（GQA-8）'},p:{...INIT,attn:'gqa8'}},
 {t:{en:'Llama-3.1-8B on one H100',zh:'一张 H100 跑 Llama-3.1-8B'},d:{en:'GQA-8 as well, with everything on one GPU',zh:'也是 GQA-8，全部放在一张卡上'},p:{...INIT,model:'llama31-8b',attn:'gqa8',n:1}},
 {t:{en:'Llama-3.1-405B on 8 × H100',zh:'8 × H100 跑 Llama-3.1-405B'},d:{en:'GQA-8, with FP8 weights, since BF16 ones do not fit',zh:'GQA-8，权重用 FP8，因为 BF16 放不下'},p:{...INIT,model:'llama31-405b',attn:'gqa8',wp:'fp8'}},
 {t:{en:'DeepSeek-V3 on 8 × H200',zh:'8 × H200 跑 DeepSeek-V3'},d:{en:'Multi-head latent attention (MLA), with FP8 weights, which need the 141 GB GPUs',zh:'multi-head latent attention（MLA），权重 FP8，要 141 GB 的卡才放得下'},p:{...INIT,model:'deepseek-v3',attn:'mla',gpu:'h200',wp:'fp8'}}
];
const presetNow=()=>PRESETS.findIndex(p=>SKEYS.every(k=>p.p[k]===S[k]));
/* the hover of a scenario button: its design's full name and what it stores (R2.8: every option that shows an abbreviation) */
const presetTip=p=>{const m=MODELS[p.p.model]; return NB(designTip(p.p.attn,m)+(p.p.attn===m.native?tx(` ${m.name} uses it.`,`${m.name} 用的就是它。`):tx(' This is the initial setup.','这是初始设置。')));};
/* the 405B and DeepSeek-V3 scenarios with their real FP8 checkpoints (same memory model; META405_FP8, DSV3_EXTRA) */
const usersWith=(x,w)=>{const budget=x.usable-w-x.oh; return budget>0?Math.floor(budget*x.shard/x.kvPerReq+1e-9):0;};
function fp8Real(){const p4=compute(PRESETS.find(p=>p.p.model==='llama31-405b').p), ds=compute(PRESETS.find(p=>p.p.model==='deepseek-v3').p), M=META405_FP8;
  const w4=(M.bf16*2+M.f8+M.f32*4)/p4.s.n, wd=ds.wPerGpu+DSV3_EXTRA*GiB;
  return {p4,w4,u4:usersWith(p4,w4),bf16:M.bf16,ds,uds:usersWith(ds,wd)};}
/* the line under the scenarios: what the click picked, its numbers, and one comparison with the initial setup (R1.9, R6.1) */
function scenarioLine(r1){
  const cur=presetNow(), b=I(), g8=Ia('gqa8');
  if(cur<0) return tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。');
  const P=PRESETS[cur], sel=tx(`Selected: ${L(P.t)}.`,`已选：${L(P.t)}。`), ctx=ctxK(r1.s.ctx);
  if(cur===0) return sel+tx(` At most ${users(r1.users)} ${fitV(r1.users)} at ${ctx} each. This is the initial setup: every query head has its own KV head.`,
                            `每人 ${ctx} 上下文时最多装下 ${r1.users} 个用户。这是初始设置：每个 query head 都有自己的 KV head。`);
  const res=r1.users>0?tx(` At most ${users(r1.users)} ${fitV(r1.users)} at ${ctx} each (initial setup: ${users(b.users)}).`,`每人 ${ctx} 上下文时最多装下 ${r1.users} 个用户（初始设置是 ${b.users} 个用户）。`)
                      :tx(` Not even one user fits at ${ctx} each (initial setup: ${users(b.users)}).`,`每人 ${ctx} 上下文时一个用户也装不下（初始设置是 ${b.users} 个用户）。`);
  /* R6.20: after the result (users, one pair with the initial setup), the reason in words; the sizes are on the chart and the metric
     cards, the weights in Show the math, the FP8 checkpoints' numbers in the Fidelity card */
  const wMost=r1.wPerGpu/r1.usable>0.5;
  let why='';
  if(cur===1){const k=Math.round(b.perGpuTok/r1.perGpuTok), f=FRW[k]||[`1/${k}`,`${k} 分之一`];
    why=tx(` Every ${r1.group} query heads share one KV head, so each GPU stores ${f[0]} as much as with MHA.`,`每 ${r1.group} 个 query head 共用一个 KV head，所以每张 GPU 存的只有 MHA 的${f[1]}。`);}
  if(cur===2) why=r1.perTok<g8.perTok?tx(' Per token it stores less than Llama-3.1-70B, but with no tensor parallelism its one GPU keeps all of it, while Llama-3.1-70B splits its KV cache across its GPUs.',
                                        '它每个 token 存得比 Llama-3.1-70B 少，但不做张量并行，这一张 GPU 要存全部；Llama-3.1-70B 却能把 KV cache 分到各张 GPU 上。')
                                      :tx(' With no tensor parallelism, its one GPU keeps all of its KV cache.','不做张量并行，这一张 GPU 要存全部 KV cache。');
  if(cur===3){const f=fp8Real(), fr=fracW(Math.round(f.p4.m.params/f.bf16)), big=r1.perTok>g8.perTok;
    // review A r13 M4: the BF16 part is more than the attention matrices, so its size is named (a fifth of the parameters)
    why=tx(` Even in FP8, its weights take ${wMost?'more than half of':'much of'} the memory each GPU can use${big?', and its extra layers make each token’s KV cache bigger than Llama-3.1-70B’s':''}. That counts one byte per parameter: Meta’s own FP8 release keeps ${fr} of the parameters in BF16, every attention matrix among them, which leaves room for only ${users(f.u4)}.`,
           `即使用 FP8，权重也要占每张 GPU 可用显存的${wMost?'一半以上':'很大一部分'}${big?'；它的层数更多，每个 token 的 KV cache 也比 Llama-3.1-70B 的大':''}。这是按每个参数一个字节算的；Meta 官方的 FP8 版本有${fr}的参数留在 BF16，所有 attention 矩阵都在其中，只装得下 ${f.u4} 个用户。`);}
  if(cur===4) why=tx(` It stores the least per token of these five scenarios, but every GPU keeps a full copy, and its FP8 weights take ${wMost?'more than half of':'much of'} the memory each GPU can use.`,
                     `在这五个场景里它每个 token 存得最少，但每张 GPU 都要存一整份；FP8 的权重还要占每张 GPU 可用显存的${wMost?'一半以上':'很大一部分'}。`);
  return sel+res+why;
}

/* ================= challenge ================= */
const chEval=a=>{const r1=compute({...CH_BASE,attn:a}); return {a,r:r1,u:r1.users,met:r1.users>=GOAL,kv:r1.G};};
const chAll=()=>DESIGNS.map(chEval);
/* the best pick: it fits the goal and keeps the most KV heads (MLA keeps none) */
const chBest=()=>chAll().filter(x=>x.met).sort((x,y)=>y.kv-x.kv)[0];
/* what a design stores in Llama-3.1-70B's shape; the options add the full name (review 8) */
const chShape=a=>{const m=m70();
  return {mha:tx(`${m.heads} KV heads, one per query head`,`${m.heads} 个 KV head，每个 query head 一个`),gqa16:tx(`16 KV heads, ${m.heads/16} query heads each`,`16 个 KV head，每个给 ${m.heads/16} 个 query head 用`),gqa8:tx(`8 KV heads, ${m.heads/8} query heads each`,`8 个 KV head，每个给 ${m.heads/8} 个 query head 用`),
    gqa4:tx(`4 KV heads, ${m.heads/4} query heads each`,`4 个 KV head，每个给 ${m.heads/4} 个 query head 用`),mqa:tx(`1 KV head shared by all ${m.heads} query heads`,`1 个 KV head，${m.heads} 个 query head 共用`),mla:tx(`a compressed vector and a RoPE key, ${MLA_C} + ${MLA_R} numbers`,`一个压缩向量加一个 RoPE key，${MLA_C} + ${MLA_R} 维`)}[a];};
const chDesc=a=>tx(`${cap1(FULL(a))}: ${chShape(a)}`,`${FULL(a)}：${chShape(a)}`);
function chResultHTML(){
  if(!chSubmitted||!chPicked) return '';
  const me=chEval(S.attn), best=chBest(), all=chAll(), g8=all.find(x=>x.a==='gqa8');
  let h;
  if(me.met&&me.a===best.a) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${me.u} users fit, with ${me.kv} KV heads.</b> No design with more KV heads fits as many, and it is the design Llama-3.1-70B uses.</p>`,
                                 `<p class="okline" style="margin-top:12px"><b>达成目标：装下 ${me.u} 个用户，有 ${me.kv} 个 KV head。</b>KV head 更多的设计都达不到目标，这也正是 Llama-3.1-70B 的设计。</p>`);
  else if(me.met){const eqU=best.u===me.u, eqG=best.r.perGpuTok===me.r.perGpuTok;
    h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${me.u} users fit.</b> But ${DNAME[best.a]} fits ${eqU?'as many':best.u} with ${best.kv} KV heads instead of ${me.kv}. ${eqG?'Each GPU stores the same either way, so the extra sharing saves no memory and only risks quality.':'Its extra KV heads keep it closer to MHA in quality.'}</p>`,
         `<p class="okline" style="margin-top:12px"><b>达成目标：装下 ${me.u} 个用户。</b>但 ${DNAME[best.a]} 也能装下${eqU?'这么多':` ${best.u} 个`}，而且有 ${best.kv} 个 KV head，不是 ${me.kv} 个。${eqG?'两者每张 GPU 存的一样多，多共用并不省显存，只会多担质量风险。':'KV head 多，质量更接近 MHA。'}</p>`);}
  else {
    const hint=me.a==='mla'?tx('Hint: MLA stores much less per token than GQA-8, but how much of it does each GPU keep?','提示：MLA 每个 token 比 GQA-8 少存很多，可每张 GPU 要存多少？')
      :tx(`Hint: with ${me.kv} KV heads on ${CH_BASE.n} GPUs, how many KV heads does each GPU store, and how much does one user take on it at ${ctxK(CH_BASE.ctx)}? The numbers are in ${J('math','Show the math')}.`,
          `提示：${me.kv} 个 KV head 分到 ${CH_BASE.n} 张 GPU 上，每张存几个？每人 ${ctxK(CH_BASE.ctx)} 上下文时，一个用户在每张 GPU 上占多少？算式在“${J('math','展开公式')}”里。`);
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: only ${users(me.u)} ${me.u===1?'fits':'fit'}.</b></p><p class="small">${hint}</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：只能装下 ${me.u} 个用户。</b></p><p class="small">${hint}</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  const rows=all.map(x=>{const tags=[]; if(x.a==='mha') tags.push({text:tx('initial setup','初始设置')}); if(x.a==='gqa8') tags.push({text:tx('Llama-3.1-70B uses it','Llama-3.1-70B 用的')}); if(x.a===me.a) tags.push({text:tx('your pick','你的选择'),me:true});
    return {label:tx(`${DNAME[x.a]} (${FULL(x.a)}): ${chShape(x.a)}`,`${DNAME[x.a]}（${FULL(x.a)}）：${chShape(x.a)}`),value:x.u,text:String(x.u),ok:x.met,me:x.a===me.a,tags};});
  h+=`<div class="gm-cmpwrap">`+A.ui.cmpChart({title:tx(`Most users that fit on 8 × H100 at ${ctxK(CH_BASE.ctx)} of context each (Llama-3.1-70B’s shape)`,`8 × H100、每人 ${ctxK(CH_BASE.ctx)} 上下文时最多能装下的用户数（Llama-3.1-70B 的结构）`),
    brief:true,hiddenText:n=>tx(n===1?`The other design fits fewer than ${GOAL} users and is not shown.`:`The other ${n} designs fit fewer than ${GOAL} users and are not shown.`,`其余 ${n} 种设计装不下 ${GOAL} 个用户，图里没有列出。`),
    aria:tx('Most users that fit: ','最多能装下的用户数：')+all.filter(x=>x.met||x.a===me.a||x.a==='mha').map(x=>`${DNAME[x.a]} ${x.u}`).join(tx(', ','，')),
    rows,goal:GOAL,goalLabel:tx(`goal: ${GOAL}`,`目标 ${GOAL}`),max:Math.max(GOAL*1.5,...all.map(x=>x.u))*1.1,
    okText:tx(`${GOAL} or more users fit`,`装得下 ${GOAL} 个或更多`),noText:tx(`fewer than ${GOAL}`,`不到 ${GOAL} 个`),goalText:tx(`goal: ${GOAL} users`,`目标：${GOAL} 个用户`)})+`</div>`;
  const u=a=>all.find(x=>x.a===a).u;
  /* baseline → effect → reason (R3.10), in words with one pair of numbers (R6.20); MHA, GQA-8, GQA-4 and MQA are on the brief chart, MLA is named in words */
  const g4same=u('gqa4')===u('gqa8')&&u('mqa')===u('gqa8'), mlaFewer=u('mla')<u('gqa8');
  h+=tx(`<p class="small">MHA, the initial setup, fits only ${users(u('mha'))}; GQA-8 fits ${u('gqa8')}, and ${g4same?'GQA-4 and MQA no more':`GQA-4 and MQA ${u('gqa4')} and ${u('mqa')}`}.</p>`+
        `<p class="small">Why: fewer KV heads shrink each GPU’s share only down to one per GPU; below that, they are copied. MLA stores little per token, but every GPU keeps all of it, so ${mlaFewer?'fewer users fit than with GQA-8':'no more users fit than with GQA-8'}.</p>`+
        `<p class="small muted">Note: this challenge counts memory only. The rule “more KV heads, closer to MHA” extends the GQA paper’s T5 results, which compare quality only for MHA, GQA-8 and MQA; before choosing a design for a real model, compare the candidates on your own evals. Serving stacks such as SGLang can run ${DEF('data-parallel attention',T.dp())} instead, so that MLA’s copies go away; this lab does not model that.</p>`,
        `<p class="small">初始设置 MHA 只装下 ${u('mha')} 个用户；GQA-8 装下 ${u('gqa8')} 个，${g4same?'GQA-4 和 MQA 也不会更多':`GQA-4 和 MQA 分别是 ${u('gqa4')} 个和 ${u('mqa')} 个`}。</p>`+
        `<p class="small">原因：KV head 减少，每张 GPU 分到的就少，但只到每张 GPU 一个为止；再少，KV head 就只能复制。MLA 每个 token 存得少，但每张 GPU 都要存全部，所以${mlaFewer?'装下的用户比 GQA-8 少':'装下的用户不比 GQA-8 多'}。</p>`+
        `<p class="small muted">注意：这个挑战只算显存。“KV head 越多越接近 MHA”是从 GQA 论文的 T5 实验外推的，论文只比较了 MHA、GQA-8 和 MQA 的质量；给真实模型选设计之前，要在自己的评测上比一比候选方案。SGLang 等框架可以改用${DEF('数据并行 attention',T.dp())}，MLA 的副本就没有了；这个 Lab 没有建模这种做法。</p>`);
  return h;
}

/* ================= the capability layer (Codex review 1 #5, Batch 2) =================
   Under the title, what the learner can do after the lab; in the challenge only, after its task card, three ways to show it transfers.
   The challenge's own finding stays out of it: no design's user count, which design wins, or what fewer KV heads than GPUs do (parts
   5 and 6 teach that, and the challenge checks it). The new cases are real shapes that are not in MODELS (each from its config.json),
   sized with compute()'s per-token formula, term by term. */
const OUTCOME=()=>tx('explain how GQA and MLA shrink the KV cache and what each costs, and size the cache each GPU keeps.',
                     '讲清 GQA 和 MLA 怎样让 KV cache 变小、各有什么代价，并算出每张 GPU 要存多少 cache。');
const QWEN={name:'Qwen2.5‑7B',layers:28,heads:28,kvHeads:4,headDim:128}, MISTRAL={name:'Mistral‑7B‑v0.1',layers:32,heads:32,kvHeads:8,headDim:128};
/* one token's KV cache with G KV heads per layer: compute()'s perTok for MHA, GQA and MQA */
const kvTok=(m,G)=>2*m.layers*G*m.headDim*KVB;
/* a new case's options in natural order (smallest first), each wrong one with its note */
const exOpts=(xs,right)=>{const s=xs.slice().sort((x,y)=>x.b-y.b); return {o:s.map(x=>kib(x.b)),a:s.findIndex(x=>x.b===right),n:s.map(x=>x.b===right?null:x.n)};};
function EXITS(){
  const [mh,g8]=GQA_T1, n8=kvHeadsOf(m70(),g8.d), sc=v=>fmtF(v,1);
  const dv=compute({model:'deepseek-v3',attn:'mla',gpu:'h200',n:8,wp:'fp8',ctx:INIT.ctx}), dm=dv.m, eq=fmtD((dv.dc+dv.dr)/(2*dm.headDim),2);
  const q=QWEN, qT=kvTok(q,q.kvHeads), qM=kvTok(q,q.heads), w=MISTRAL, wT=kvTok(w,w.kvHeads), wM=kvTok(w,w.heads), wg=w.heads/w.kvHeads;
  const two=tx('That leaves out the 2: each KV head stores a key and a value.','漏了那个 2：每个 KV head 存一个 key 和一个 value。');
  const t1=exOpts([{b:qT/2,n:two},{b:qT},{b:qT*2,n:tx(`That counts ${2*KVB} bytes per number; BF16 takes ${KVB}.`,`这是按每个数值 ${2*KVB} 字节算的，BF16 每个数值只占 ${KVB} 字节。`)},
    {b:qM,n:tx(`That counts all ${q.heads} query heads. Only the ${q.kvHeads} KV heads store keys and values.`,`这是按 ${q.heads} 个 query head 算的。存 key 和 value 的只有 ${q.kvHeads} 个 KV head。`)}],qT);
  const t2=exOpts([{b:wT,n:tx(`That is what it keeps with its ${w.kvHeads} shared KV heads. With MHA each of the ${w.heads} query heads has its own KV head.`,`这是它用 ${w.kvHeads} 个共用的 KV head 时存的量。用 MHA 时，${w.heads} 个 query head 各有自己的 KV head。`)},
    {b:wM/2,n:two},{b:wM},
    {b:wT*w.heads,n:tx(`That multiplies the whole ${kib(wT)} by the ${w.heads} query heads. MHA has ${w.heads} KV heads instead of ${w.kvHeads}: ${times(w.heads,w.kvHeads)} times as many.`,`这是把整个 ${kib(wT)} 乘上了 ${w.heads} 个 query head。MHA 有 ${w.heads} 个 KV head，是 ${w.kvHeads} 个的 ${times(w.heads,w.kvHeads)} 倍。`)}],wM);
  return {
 core:()=>chSubmitted, explain:[
  {q:tx('Grouped-query attention (GQA) lets a group of query heads share one KV head. Why does that shrink the KV cache, yet cost little quality?',
        'grouped-query attention（GQA）让一组 query head 共用一个 KV head。为什么这样能让 KV cache 变小，质量却损失不大？'),
   o:[tx('Only keys and values are cached, and a group stores one of each instead of one per head; little is lost because every head still compares its own query with the keys.',
         '只有 key 和 value 进 cache，一组只存一份，不再每个 head 各存一份；损失不大，是因为每个 head 还是拿自己的 query 去和 key 比较。'),
     tx('The heads of a group share one query as well, so the group needs one key and one value; little is lost because the heads of a group looked for similar tokens anyway.',
        '同一组的 head 连 query 也共用一个，所以一组只需要一个 key 和一个 value；损失不大，是因为同组的 head 本来就在找相似的 token。'),
     tx('The shared KV head keeps the keys and values of only the tokens that got the most attention, so there is less to store; little is lost because the dropped tokens mattered little.',
        '共用的 KV head 只保留得到注意力最多的那些 token 的 key 和 value，所以要存的更少；损失不大，是因为丢掉的 token 本来就不重要。')],a:0,
   n:[null,tx('It is the other way round: every head keeps its own query, and only the key and the value are shared. Different queries give different scores, so heads that share a KV head can still attend to different tokens.',
              '正好反过来：每个 head 都有自己的 query，共用的只是 key 和 value。query 不同，得分就不同，所以共用一个 KV head 的几个 head 仍能关注不同的 token。'),
     tx('GQA drops no token: each token still stores a key and a value for every KV head in every layer. The cache shrinks because a group of query heads shares one KV head.',
        'GQA 不丢 token：每个 token 在每一层仍为每个 KV head 存一个 key 和一个 value。KV cache 变小，是因为一组 query head 共用一个 KV head。')],
   why:tx(`Each token’s KV cache is 2 (a key and a value) × layers × KV heads × head_dim × bytes per number. Queries are not stored, so when the query heads of a group share one KV head, each token’s KV cache is divided by the number of heads in the group; each head still compares its own query with the keys. In the GQA paper, T5‑XXL, converted from multi-head attention (MHA) to ${n8} groups and trained a little further, scored ${sc(g8.score)} on average, against ${sc(mh.score)} before (the paper did not test decoder-only models).`,
          `每个 token 的 KV cache = 2（key 和 value）× 层数 × KV head 数 × head_dim × 每个数值的字节数。query 不存进 cache，所以一组 query head 共用一个 KV head 时，组里有几个 query head，每个 token 的 KV cache 就缩小到几分之一；每个 head 仍拿自己的 query 和 key 比较。GQA 论文把 T5‑XXL 从多头注意力（MHA）改成 ${n8} 组、再稍加训练，平均分是 ${sc(g8.score)}，改之前是 ${sc(mh.score)}（论文没有测 decoder-only 模型）。`)},
  {q:tx('How does multi-head latent attention (MLA) keep each token’s KV cache small?','multi-head latent attention（MLA）靠什么让每个 token 的 KV cache 变小？'),
   o:[tx('Each token stores one compressed vector per layer, and every head rebuilds its own key and value from it, so the cache no longer grows with the number of heads.',
         '每个 token 在每一层存一个压缩向量，每个 head 都从它还原出自己的 key 和 value，所以 cache 不再随 head 数增长。'),
     tx('Each token stores one compressed vector per head, holding that head’s key and value in fewer numbers, so the cache is smaller but still grows with the number of heads.',
        '每个 token 为每个 head 各存一个压缩向量，用更少的维装下这个 head 的 key 和 value，所以 cache 变小了，但仍随 head 数增长。'),
     tx('Each token stores compressed vectors for only a few heads, and the other heads reuse those, so it works like grouped-query attention with fewer groups.',
        '每个 token 只为少数几个 head 存压缩向量，其余的 head 复用它们，所以它就像组数更少的 grouped-query attention。')],a:0,
   n:[null,tx('MLA stores nothing per head: each token keeps one compressed vector and one small key for its position (the RoPE key) per layer, however many heads the layer has.',
              'MLA 不为每个 head 存任何东西：每个 token 每层存一个压缩向量和一个带位置信息的小 key（RoPE key），和这一层有多少个 head 无关。'),
     tx('No head reuses another head’s key and value: each rebuilds its own from the compressed vector, so every head can have a different key and value. In grouped-query attention, the heads of a group read the very same ones.',
        '没有哪个 head 复用别的 head 的 key 和 value：每个 head 都从压缩向量还原出自己的一份，所以各 head 的 key 和 value 可以都不一样。grouped-query attention 里，同一组的 head 读的才是同一份。')],
   why:tx(`Per token and layer, MLA stores one compressed vector, computed from the token’s vector, and a small RoPE key that carries its position: together as much as ${eq} KV heads store. Every head’s key and value are rebuilt from them, so the cache grows with the number of layers, not with the number of heads. ${NB(dm.name)} has ${dm.heads} heads per layer, yet stores only ${kib(dv.perTok)} per token.`,
          `MLA 每个 token 每层存两样东西：一个由 token 的向量算出的压缩向量，和一个带着 token 位置的小 RoPE key，两者加起来相当于 ${eq} 个 KV head 存的量。各个 head 的 key 和 value 都从它们还原，所以 KV cache 只随层数增长，不随 head 数增长。${NB(dm.name)} 每层有 ${dm.heads} 个 head，每个 token 也只存 ${kib(dv.perTok)}。`)}],
 transfer:[
  {q:tx(`${q.name} has ${q.layers} layers, each with ${q.heads} query heads that share ${q.kvHeads} KV heads, and head_dim ${q.headDim}. How much KV cache does it keep per token in BF16?`,
        `${q.name} 有 ${q.layers} 层，每层 ${q.heads} 个 query head 共用 ${q.kvHeads} 个 KV head，head_dim 是 ${q.headDim}。BF16 下每个 token 要存多少 KV cache？`),
   o:t1.o,a:t1.a,n:t1.n,
   why:tx(`2 (a key and a value) × ${q.layers} layers × ${q.kvHeads} KV heads × ${q.headDim} numbers × ${KVB} bytes = ${int(qT)} bytes = ${kib(qT)}. The query heads are not in the formula: only keys and values are stored.`,
          `2（key 和 value）× ${q.layers} 层 × ${q.kvHeads} 个 KV head × ${q.headDim} 维 × ${KVB} 字节 = ${int(qT)} 字节 = ${kib(qT)}。公式里没有 query head：存下来的只有 key 和 value。`)},
  {q:tx(`${w.name} has ${w.layers} layers, each with ${w.heads} query heads that share ${w.kvHeads} KV heads, and head_dim ${w.headDim}. With multi-head attention (MHA) instead, where every query head has its own KV head, how much KV cache would it keep per token in BF16?`,
        `${w.name} 有 ${w.layers} 层，每层 ${w.heads} 个 query head 共用 ${w.kvHeads} 个 KV head，head_dim 是 ${w.headDim}。如果改用多头注意力（MHA），让每个 query head 都有自己的 KV head，BF16 下每个 token 要存多少 KV cache？`),
   o:t2.o,a:t2.a,n:t2.n,
   why:tx(`With MHA it has ${w.heads} KV heads, one per query head. 2 (a key and a value) × ${w.layers} layers × ${w.heads} KV heads × ${w.headDim} numbers × ${KVB} bytes = ${int(wM)} bytes = ${kib(wM)}. That is ${times(wM,wT)} times the ${kib(wT)} it keeps with ${w.kvHeads} KV heads, because every ${wg} query heads share one KV head (${w.heads} ÷ ${w.kvHeads} = ${wg}).`,
          `用 MHA 时它有 ${w.heads} 个 KV head，每个 query head 一个。2（key 和 value）× ${w.layers} 层 × ${w.heads} 个 KV head × ${w.headDim} 维 × ${KVB} 字节 = ${int(wM)} 字节 = ${kib(wM)}。这是它用 ${w.kvHeads} 个 KV head 时 ${kib(wT)} 的 ${times(wM,wT)} 倍，因为每 ${wg} 个 query head 共用一个 KV head（${w.heads} ÷ ${w.kvHeads} = ${wg}）。`)}],
 practice:[
  {id:'attention-kv-cache',d:tx('Part 2: count a KV cache’s bytes from the layers, KV heads, head_dim and sequence length, and in a follow-up work out how much grouped-query attention shrinks it.',
                                '第 2 部分：按层数、KV head 数、head_dim 和序列长度算出 KV cache 的字节数，并在追问里算出 grouped-query attention 能让它小多少。')},
  {id:'inference-batching',d:tx('Item 1 (the scale estimate) and deep dive (c): work out the KV cache bytes per token for a stated model configuration, and how many sequences fit in the GPU memory the weights leave.',
                                '第 1 项（规模估算）和深入话题 (c)：给定一个模型配置，算出每个 token 的 KV cache 字节数，以及权重之外的显存能放下多少条序列。')}]};
}

/* ================= views ================= */
const SRC=(repo,p)=>`<a href="https://github.com/${repo}/blob/main/${p}" target="_blank" rel="noopener"><code>${p}</code></a>`;
const HF=repo=>`<a href="https://huggingface.co/${repo}" target="_blank" rel="noopener"><code>${repo}</code></a>`;
function fidPop(){const dW=mhaExtraParams(), mhX=mhaBig();
 const li=a=>a.map(x=>`<li>${x}</li>`).join('');
 /* FP8 at 1 byte per parameter (review B r12 N9), with what the real checkpoints change at 128K (numbers from fp8Real) */
 const f=fp8Real(), c4=ctxK(f.p4.s.ctx), cd=ctxK(f.ds.s.ctx), w4=gibS(r(f.w4/GiB,1)), w40=gibS(r(f.p4.wPerGpu/GiB,1)), b4=bigH(f.bf16);
 const fp8E=`FP8 weights count 1 byte per parameter. Real FP8 checkpoints keep some tensors in BF16 and store scale factors. For DeepSeek-V3 that adds about ${DSV3_EXTRA} GiB per GPU, and `+
   (f.uds===f.ds.users?`its ${f.ds.users} users at ${cd} each still fit.`:`${users(f.uds)} ${fitV(f.uds)} at ${cd} each instead of ${f.ds.users}.`)+
   ` Meta’s own FP8 release of Llama-3.1-405B keeps ${b4} parameters in BF16, every attention matrix among them: ${w4} per GPU instead of ${w40}, so at ${c4} each only ${users(f.u4)} ${fitV(f.u4)} instead of ${f.p4.users}.`;
 const fp8Z=`FP8 权重按每个参数 1 字节算。真实的 FP8 checkpoint 有些张量是 BF16，还要存缩放系数。DeepSeek-V3 因此每张 GPU 约多 ${DSV3_EXTRA} GiB，`+
   (f.uds===f.ds.users?`每人 ${cd} 上下文时还是装下 ${f.ds.users} 个用户。`:`每人 ${cd} 上下文时装下 ${f.uds} 个用户，而不是 ${f.ds.users} 个。`)+
   `Meta 官方的 Llama-3.1-405B FP8 版本有 ${b4} 个参数是 BF16，所有 attention 矩阵都在其中：每张 GPU ${w4}，而不是 ${w40}，每人 ${c4} 上下文时只装得下 ${f.u4} 个用户，而不是 ${f.p4.users} 个。`;
 return tx(`<div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>${li([
   'KV cache per token and layer, for each design: multi-head attention (MHA) and grouped-query attention (GQA) store a key and a value of head_dim numbers for each KV head; multi-query attention (MQA) one pair in all; multi-head latent attention (MLA) one compressed vector and one RoPE key (512 + 64 numbers in DeepSeek-V2 and V3). Model shapes come from each model’s config.json.',
   'Tensor parallelism over n GPUs: each GPU computes 1/n of the query heads and stores the KV heads they read, as vLLM does. With fewer KV heads than GPUs, every GPU whose query heads read a KV head keeps its own copy of it, and MLA’s compressed vector is copied to every GPU.',
   `Most users that fit: the memory model of the ${A.shell.labLink('kv-cache')} lab. Each GPU uses ${fmtD(UTIL*100,0)}% of its memory; the weights are split evenly; ${OH} GiB per GPU goes to runtime overhead; the rest holds KV cache, and every user fills the whole context.`])}</ul></div>`+
   `<div><div class="eyebrow">Simplified and assumed</div><ul>${li([
   `Every attention design keeps the model’s parameters, so only the KV cache changes. The initial setup, Llama-3.1-70B’s shape with MHA, would really have larger key and value matrices, about ${bigH(dW)} more parameters; counted in, it still fits ${users(mhX.users)} at ${ctxK(INIT.ctx)}.`,
   'MLA in a Llama model is a what-if: Llama’s layers with DeepSeek-V3’s compressed vector and RoPE key (512 + 64 numbers).',
   `Memory utilization ${UTIL}, ${OH} GiB of runtime overhead per GPU and a BF16 KV cache are fixed here (${J('fixed','see why under Settings')}).`,
   'Plain tensor parallelism only: every layer is split across the GPUs, attention by head. Serving stacks such as SGLang can run attention data-parallel instead, which this lab does not model.',
   fp8E,
   'Quality is not modeled. The quality scores in this lab are the GQA paper’s results for T5-XXL.'])}</ul></div>`+
   `<div><div class="eyebrow">Sources</div><ul>${li([
   'Model shapes: config.json on Hugging Face (the Llama-3.1 repos are gated, so through public mirrors; DeepSeek-V3; Falcon-7B). Llama 3 paper, Table 3: 8 KV heads in all three sizes.',
   '<a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noopener">Shazeer, 2019</a>: MQA, and why loading keys and values slows decoding. <a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noopener">Ainslie et al., 2023</a>: GQA, Table 1 (T5-XXL scores and times), §2.2 (copies under sharding). Pope et al., 2022, §3.3: multi-query keys and values are copied to every chip.',
   '<a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noopener">DeepSeek-V2</a>: §2.1.2 (the compressed vector; folding the up-projections), §2.1.3 (the RoPE key), §2.1.4 and Table 1 (equal to GQA with 2.25 groups).',
   `vLLM: ${SRC('vllm-project/vllm','vllm/model_executor/layers/linear.py')} (QKVParallelLinear), ${SRC('vllm-project/vllm','vllm/model_executor/models/llama.py')} and ${SRC('vllm-project/vllm','vllm/config/model.py')} (ModelConfig.get_num_kv_heads returns 1 for MLA). SGLang v0.4 release notes (LMSYS blog, 2024-12-04): data-parallel attention for MLA, an option (--enable-dp-attention).`,
   `FP8 checkpoints on Hugging Face: ${HF('meta-llama/Llama-3.1-405B-Instruct-FP8')} (modules_to_not_convert in config.json; parameters by type) and ${HF('deepseek-ai/DeepSeek-V3')} (its quantization_config sets one scale per 128 × 128 block).`])}</ul>`+
   `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-26. Prototype, not yet expert-reviewed.</p></div></div>`,
  `<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>${li([
   '每种设计每个 token 每层存多少：多头注意力（MHA）和 grouped-query attention（GQA）为每个 KV head 存一个 key 和一个 value，各 head_dim 维；multi-query attention（MQA）一共只存一对；multi-head latent attention（MLA）存一个压缩向量和一个 RoPE key（DeepSeek-V2 和 V3 里是 512 + 64 维）。模型结构来自各自的 config.json。',
   '在 n 张 GPU 上张量并行：每张 GPU 算 1/n 的 query head，并存下它们要读的 KV head，和 vLLM 的做法一样。KV head 比 GPU 少时，读同一个 KV head 的几张 GPU 各存它一份。MLA 的压缩向量则每张 GPU 都要存一份。',
   `最多用户数沿用“${A.shell.labLink('kv-cache')}”Lab 的显存模型。每张 GPU 用 ${fmtD(UTIL*100,0)}% 的显存；权重平均分到各张 GPU；每张 GPU 留 ${OH} GiB 给运行时开销；剩下的放 KV cache，每个用户都占满自己的上下文。`])}</ul></div>`+
   `<div><div class="eyebrow">简化和假设</div><ul>${li([
   `每种注意力设计都保留模型原来的参数，所以变的只有 KV cache。初始设置是用 MHA 的 Llama-3.1-70B 结构，真这样做的话 key 和 value 矩阵更大，大约多 ${bigH(dW)} 个参数；算上它们，每人 ${ctxK(INIT.ctx)} 时也还是只能装下 ${mhX.users} 个用户。`,
   'Llama 模型里的 MLA 是假设的：Llama 的层，配上 DeepSeek-V3 的压缩向量和 RoPE key（512 + 64 维）。',
   `显存利用率 ${UTIL}、每张 GPU ${OH} GiB 运行时开销和 BF16 的 KV cache 在这里是固定的（${J('fixed','原因见“自由调参”')}）。`,
   '只算纯张量并行：每一层都拆到各张 GPU 上，attention 按 head 拆。SGLang 等框架可以改让 attention 做数据并行，这个 Lab 没有建模这种做法。',
   fp8Z,
   '不建模质量。这个 Lab 里的质量分数取自 GQA 论文里 T5-XXL 的结果。'])}</ul></div>`+
   `<div><div class="eyebrow">来源</div><ul>${li([
   '模型结构：Hugging Face 上的 config.json（Llama-3.1 的仓库需要授权，用的是公开镜像；DeepSeek-V3；Falcon-7B）。Llama 3 论文表 3：三个尺寸都是 8 个 KV head。',
   '<a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noopener">Shazeer，2019</a>：MQA，以及为什么读 key 和 value 会拖慢解码。<a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noopener">Ainslie 等，2023</a>：GQA，表 1（T5-XXL 的分数和用时），§2.2（切分时的副本）。Pope 等，2022，§3.3：multi-query 的 key 和 value 会复制到每块芯片上。',
   '<a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noopener">DeepSeek-V2</a>：§2.1.2（压缩向量；把上投影矩阵并进去），§2.1.3（RoPE key），§2.1.4 和表 1（和只有 2.25 组的 GQA 一样多）。',
   `vLLM：${SRC('vllm-project/vllm','vllm/model_executor/layers/linear.py')}（QKVParallelLinear）、${SRC('vllm-project/vllm','vllm/model_executor/models/llama.py')} 和 ${SRC('vllm-project/vllm','vllm/config/model.py')}（MLA 时 ModelConfig.get_num_kv_heads 返回 1）。SGLang v0.4 发布说明（LMSYS 博客，2024-12-04）：为 MLA 做的数据并行 attention，是一个可选项（--enable-dp-attention）。`,
   `Hugging Face 上的 FP8 checkpoint：${HF('meta-llama/Llama-3.1-405B-Instruct-FP8')}（config.json 里的 modules_to_not_convert；各类型的参数数）和 ${HF('deepseek-ai/DeepSeek-V3')}（quantization_config 规定每个 128 × 128 块一个缩放系数）。`])}</ul>`+
   `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-26。原型，尚未经专家审阅。</p></div></div>`);}
/* MHA's larger key and value matrices: 2 × d × (64 − 8) × head_dim per layer (config.json shapes; facts A4, B) */
function mhaExtraParams(){const m=m70(); return 2*m.hidden*(m.heads-m.kvHeads)*m.headDim*m.layers;}
/* the users an MHA Llama-3.1-70B would fit with those extra weights counted in (same memory model) */
function mhaBig(){const x=compute({...INIT,attn:'mha'}), w=(x.m.params+mhaExtraParams())*PREC[INIT.wp]/INIT.n, budget=x.usable-w-x.oh;
  return {users:budget>0?Math.floor(budget*x.shard/x.kvPerReq+1e-9):0,wPerGpu:w};}
/* the kit's primer ends by linking "Why a KV cache", so "That lab" right after it needs no second link */
const priPop=()=>A.t('priPop')+tx(`<p>That lab ends with the size of one token’s KV cache: 2 (a key and a value) × layers × KV heads × head_dim × bytes per number. This lab first changes the number of KV heads in the formula, then shows how multi-head latent attention (MLA) replaces the per-head keys and values with one compressed vector.</p>`,
                                  `<p>那个 Lab 最后算出一个 token 的 KV cache = 2（key 和 value）× 层数 × KV head 数 × head_dim × 每个数值的字节数。这个 Lab 先改公式里的 KV head 数，最后看 multi-head latent attention（MLA）怎么用一个压缩向量代替各 head 的 key 和 value。</p>`);
function controlsHTML(){
  const opt=(o,v)=>Object.entries(o).map(([k,x])=>`<option value="${k}"${k===v?' selected':''}>${esc(x.name)}</option>`).join('');
  return `<aside class="panel controls" id="gm-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div><p class="ctl-lock" id="gm-lock" hidden></p>`+
    `<div class="ctl lockable" id="gm-ctl-model"><label for="gm-model">${tx('Model','模型')}</label><select id="gm-model">${opt(MODELS,S.model)}</select><div class="ctl-note" id="gm-model-note"></div></div>`+
    `<div class="gm-seg6">${A.ui.seg({id:'gm-ctl-attn',key:'attn',label:tx('Attention design','注意力设计'),options:DESIGNS.map(a=>[a,DNAME[a]]),value:S.attn||'',note:'<span id="gm-attn-note"></span>'})}</div>`+
    `<div class="ctl lockable" id="gm-ctl-gpu"><label for="gm-gpu">GPU</label><select id="gm-gpu">${opt(GPUS,S.gpu)}</select><div class="ctl-note" id="gm-gpu-note"></div></div>`+
    `<div class="lockable">${A.ui.seg({id:'gm-ctl-n',key:'n',label:tx('GPUs, tensor parallel','GPU 数（张量并行）'),options:NS.map(n=>[String(n),String(n)]),value:String(S.n),note:'<span id="gm-n-note"></span>'})}</div>`+
    `<div class="lockable">${A.ui.seg({id:'gm-ctl-wp',key:'wp',label:tx('Weight precision','权重精度'),options:[['bf16','BF16'],['fp8','FP8']],value:S.wp,note:tx('Weights take the parameter count times 2 bytes in BF16, 1 byte in FP8.','权重占的显存是参数量乘以每个参数的字节数：BF16 是 2，FP8 是 1。')})}</div>`+
    `<div class="lockable">${A.ui.range({id:'gm-ctl-ctx',key:'ctx',label:tx('Context per user','每个用户的上下文'),min:0,max:CTXS.length-1,step:1,value:CTXS.indexOf(S.ctx),out:''})}</div>`+
    `<details class="adv lockable" id="gm-fixed"><summary>${tx('Fixed in this lab: memory utilization, runtime overhead and KV cache precision','这个 Lab 里固定的：显存利用率、运行时开销和 KV cache 精度')}</summary>`+
      `<div class="ctl-note gm-fx" id="gm-fx-util">${tx(`<b>Memory utilization: ${UTIL}.</b> The server uses at most this share of each GPU’s memory and keeps the rest as a safety margin, so memory the engine does not account for cannot push the GPU out of memory. It is set by hand: vLLM used 0.9 by default up to v0.19 and 0.92 since v0.20.0. The ${A.shell.labLink('kv-cache')} lab lets you change it.`,
        `<b>显存利用率：${UTIL}。</b>服务端最多只用每张卡显存的这个比例，剩下的留作安全余量，免得引擎没算到的占用把显存撑爆。它是人为设定的。vLLM 在 v0.19 及以前默认 0.9，v0.20.0 起默认 0.92。“${A.shell.labLink('kv-cache')}”Lab 里可以改它。`)}</div>`+
      `<div class="ctl-note gm-fx" id="gm-fx-oh">${tx(`<b>Runtime overhead: ${OH} GiB per GPU.</b> This is memory the serving engine needs besides the weights and the KV cache, such as one step’s temporary activations and CUDA graphs. It is an estimate: engines such as vLLM measure it at startup, and it grows with the tokens one step may process. It is usually a GiB or two, because inference frees each layer’s activations right away.`,
        `<b>运行时开销：每张 GPU ${OH} GiB。</b>这是推理引擎在权重和 KV cache 之外要占的显存，比如一步前向的临时激活和 CUDA graph。它是估计值：vLLM 这类引擎在启动时实测它，一步最多处理的 token 越多，它就越大。它通常只有一两个 GiB，因为推理时每层的激活用完就释放。`)}</div>`+
      `<div class="ctl-note gm-fx" id="gm-fx-kvb">${tx(`<b>KV cache precision: BF16, ${KVB} bytes per number.</b> It is fixed so that only the attention design changes the KV cache here. Serving engines can also keep the KV cache in FP8, 1 byte per number, which halves every size in this lab at some risk to quality; the ${A.shell.labLink('kv-cache')} lab shows its effect.`,
        `<b>KV cache 精度：BF16，每个数值占 ${KVB} 字节。</b>固定它，是为了让这里只有注意力设计在改变 KV cache。推理引擎也可以把 KV cache 存成 FP8，每个数值占 1 字节，这个 Lab 里的每个大小都会减半，但质量有一定风险；“${A.shell.labLink('kv-cache')}”Lab 里能看到它的效果。`)}</div></details>`+
    `<div><button class="btn ghost sm" type="button" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
function connectHTML(){
  const code=`KiB = 1024

def kv_bytes_per_token(layers, kv_heads=None, head_dim=128,
                       latent=None, rope=None, nbytes=2):
    """One token's KV cache, all layers."""
    if latent is not None:
        # MLA: one compressed vector + one RoPE key
        per_layer = latent + rope
    else:
        # MHA, GQA, MQA: a key and a value per KV head
        per_layer = 2 * kv_heads * head_dim
    return per_layer * layers * nbytes

def per_gpu(total, gpus, kv_heads=None):
    """Tensor parallelism splits KV heads across the
    GPUs, never below one per GPU."""
    if kv_heads is None:
        return total  # MLA: every GPU keeps it all
    # fewer KV heads than GPUs: copies
    return total / min(gpus, kv_heads)

# Llama-3.1-70B's shape: 80 layers, 64 query heads
designs = {"MHA": 64, "GQA-16": 16, "GQA-8": 8,
           "GQA-4": 4, "MQA": 1}
for name, kv in designs.items():
    t = kv_bytes_per_token(80, kv_heads=kv)
    print(name, t / KiB, per_gpu(t, 8, kv) / KiB)
t = kv_bytes_per_token(80, latent=512, rope=64)
print("MLA", t / KiB, per_gpu(t, 8) / KiB)
# KiB per token: MHA 2560.0, GQA-16 640.0,
# GQA-8 320.0, GQA-4 160.0, MQA 40.0, MLA 90.0;
# the second column is each GPU's share`;
  const HF='https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py';
  return `<section class="wrap connect" aria-labelledby="gm-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="gm-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
    `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('One token’s KV cache for each design, and what each GPU stores under tensor parallelism:','每种设计下一个 token 的 KV cache，以及张量并行时每张 GPU 存多少：')}</p><pre class="code" id="gm-code">${esc(code)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
    `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="gm-prod">`+
      `<li><a href="${HF}" target="_blank" rel="noopener"><code>transformers/models/llama/modeling_llama.py</code></a>${tx(': <code>LlamaAttention</code> builds <code>k_proj</code> and <code>v_proj</code> with <code>num_key_value_heads</code> × head_dim outputs, so the cache holds only the KV heads; <code>repeat_kv</code> expands them to the query heads only when attention is computed.','：<code>LlamaAttention</code> 的 <code>k_proj</code> 和 <code>v_proj</code> 只输出 <code>num_key_value_heads</code> × head_dim 维，所以 cache 里只有 KV head；算 attention 时 <code>repeat_kv</code> 才把它们展开到每个 query head。')}</li>`+
      `<li>${SRC('vllm-project/vllm','vllm/model_executor/layers/linear.py')}${tx(': <code>QKVParallelLinear</code> splits the query heads across the GPUs; with more GPUs than KV heads it gives every GPU one KV head (<code>num_kv_head_replicas</code> copies of each).','：<code>QKVParallelLinear</code> 把 query head 分到各张 GPU 上；GPU 比 KV head 多时，每张 GPU 分一个 KV head（每个复制 <code>num_kv_head_replicas</code> 份）。')}</li>`+
      `<li>${SRC('vllm-project/vllm','vllm/model_executor/models/llama.py')}${tx(`: <code>LlamaAttention</code> keeps ${A.ui.math('\\max\\left(1,\\ \\frac{\\text{KV heads}}{\\text{GPUs}}\\right)')} KV heads on each GPU and replicates them when there are fewer KV heads than GPUs.`,`：<code>LlamaAttention</code> 在每张 GPU 上放 ${A.ui.math('\\max\\left(1,\\ \\frac{\\text{KV head 数}}{\\text{GPU 数}}\\right)')} 个 KV head，KV head 比 GPU 少时就复制。`)}</li>`+
      `<li>${SRC('vllm-project/vllm','vllm/model_executor/models/deepseek_v2.py')}${tx(': <code>DeepseekV2MLAAttention</code>, the MLA of DeepSeek-V2 and V3.','：<code>DeepseekV2MLAAttention</code>，DeepSeek-V2 和 V3 的 MLA。')}</li>`+
      `<li><a href="https://lmsys.org/blog/2024-12-04-sglang-v0-4/" target="_blank" rel="noopener">${tx('SGLang v0.4 release notes','SGLang v0.4 发布说明')}</a>${tx(': data-parallel attention for DeepSeek’s MLA, an option (<code class="gm-nw">--enable-dp-attention</code>) that keeps its KV cache from being duplicated on every GPU.','：为 DeepSeek 的 MLA 做的数据并行 attention，是一个可选项（<code class="gm-nw">--enable-dp-attention</code>），打开后 KV cache 不再在每张 GPU 上重复存。')}</li></ul></article>`+
    `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
      `<li><a href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noopener">Fast Transformer Decoding: One Write-Head is All You Need</a><span>${tx('Shazeer, 2019 (MQA)','Shazeer，2019（MQA）')}</span></li>`+
      `<li><a href="https://arxiv.org/abs/2305.13245" target="_blank" rel="noopener">GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</a><span>${tx('Ainslie et al., EMNLP 2023','Ainslie 等，EMNLP 2023')}</span></li>`+
      `<li><a href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noopener">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</a><span>${tx('DeepSeek-AI, 2024 (MLA, §2.1)','DeepSeek-AI，2024（MLA，§2.1）')}</span></li></ul>`+
      `<p class="small gm-see">${tx('See also: <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a> (Cho et al., 2024) runs GPT-2 small in the browser and shows its 12 heads per layer. <a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a> (Brendan Bycroft) walks through a tiny GPT, nano-gpt, in 3D, next to the shapes of GPT-2 small and XL.',
        '另见：<a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a>（Cho 等，2024）在浏览器里运行 GPT-2 small，能看到它每层的 12 个 head。<a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a>（Brendan Bycroft）用 3D 走一遍一个很小的 GPT（nano-gpt），旁边画出 GPT-2 small 和 XL 的结构。')}</p></article>`+
    `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="gm-drillq"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-show">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="gm-drilla" hidden></div></article>`+
    `</div></section>`;
}
function view(){
  const lab=A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:tx('GQA and MLA','GQA 和 MLA'),title:L(lab?lab.q:{en:'How do GQA and MLA shrink the KV cache?',zh:'GQA 和 MLA 怎么让 KV cache 变小？'}),minutes:lab?lab.min:15,
      fid:'Analytical',fidPop:fidPop(),primer:A.t('priBtn').replace(/\s*▾$/,''),primerPop:priPop(),descs:DESCS(),mode})+
    `<div class="wrap bench"><aside class="rail" id="gm-rail" aria-label="${esc(A.t('railAria'))}"></aside>`+
    `<div class="panel stage" id="gm-stage"><div class="stage-head"><div><div class="meter-title" id="gm-title"></div><div class="meter-sub" id="gm-sub"></div></div>`+
    `<div class="stage-actions"><span class="status" id="gm-status" tabindex="0"></span></div></div>`+
    `<div class="legend-row gm-legend" id="gm-legend"></div>`+
    `<div class="stage-graph graph" id="gm-graph"><div class="bar-label" id="gm-cap"></div><div class="gm-diag" id="gm-diag"></div><p class="stage-note" id="gm-gpuline"></p><div class="gm-chart" id="gm-chart"></div></div>`+
    `<div class="metrics gm-metrics" id="gm-metrics">${MET.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('')}</div>`+
    `<details class="math" id="gm-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="gm-mathbody"></div></details></div>`+
    controlsHTML()+`</div>`+connectHTML();
}

/* ================= rail ================= */
function railGuided(){
  const s=STEPS[step];
  return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:x.lead?esc(L(x.lead)):''})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe,next:step<STEPS.length-1,
    nextDisabled:(!!s.expect&&!s.expect())||(!!s.gate&&!s.gate()),skip:!!s.skip&&!pred&&!seen});
}
function railSandbox(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
    `<p class="small">${tx('Pick a scenario and see how many users fit at once.','选一个场景，看看最多能同时装下多少个用户。')}</p>`+
    `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}" data-tipx="${esc(presetTip(p))}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
    `<p class="small gm-sbnote" id="gm-sbnote" aria-live="polite"></p>`+
    `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function railChallenge(){
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit"${chPicked?'':' disabled'}>${tx('Submit my answer','提交答案')}</button><p class="small muted" id="gm-subnote" style="margin:8px 0 0"></p>`;
  const lv=A.ui.levers({aria:tx('Attention designs, pick one','注意力设计，选一种'),value:chPicked?S.attn:'',options:DESIGNS.map(a=>({v:a,title:DNAME[a],desc:chDesc(a),result:chTried.has(a)?lvRes(a):''}))});
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx(`Fit at least ${GOAL} users on 8 × H100 at ${ctxK(CH_BASE.ctx)} of context each, keeping as many KV heads as possible`,`让 8 张 H100 在每人 ${ctxK(CH_BASE.ctx)} 上下文下至少装下 ${GOAL} 个用户，KV head 越多越好`)}</h3>`+
    `<p>${tx(`The model has Llama-3.1-70B’s shape (${m70().layers} layers, ${m70().heads} query heads of ${m70().headDim} numbers), with weights and KV cache in BF16. You choose only its attention design. Among the designs that fit ${GOAL} users, pick the one with the most KV heads: in the paper on grouped-query attention (GQA), GQA-8 (8 KV heads) scored closer to multi-head attention (MHA) than multi-query attention (MQA, 1 KV head) did; the paper compared quality only for these two and MHA.`,
             `模型是 Llama-3.1-70B 的结构（${m70().layers} 层，${m70().heads} 个 query head，每个 ${m70().headDim} 维），权重和 KV cache 都是 BF16。你只能选注意力设计。能装下 ${GOAL} 个用户的设计里，选 KV head 最多的：在 grouped-query attention（GQA）的论文里，GQA-8（8 个 KV head）的平均分比 multi-query attention（MQA，1 个 KV head）更接近多头注意力（MHA）；论文只比较了这两种和 MHA 的质量。`)}</p>`+
    `<p class="small gm-pick">${tx('Pick an attention design and see how many users fit.','选一种注意力设计，看看能同时装下多少个用户。')}</p><div id="gm-ch-lv">${lv}</div>`+
    `<div class="goal mono" id="gm-goal"><span>${tx('Users that fit now','现在能同时装下')}</span><span><b id="gm-goalv"></b> / ${GOAL} ${tx('users','个用户')}</span></div><div class="progress" id="gm-prog" aria-hidden="true"><i></i></div>`+
    submit+`<div id="gm-chres" aria-live="polite"></div></div>`;
}
const lvRes=a=>{const u=chEval(a).u; return tx(`${u} ${u===1?'user':'users'}`,`${u} 个用户`);};
function renderRail(){
  const rail=$('#gm-rail',host); if(!rail) return;
  const nl=(mode!=='guided'||step===STEPS.length-1)?A.shell.nextLab(ID):'';   // the next lab below the card (R3.11)
  // the capability layer (Batch 2): in the challenge only, after its task card and before the next lab
  const ex=mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'';
  rail.innerHTML=NB(mode==='guided'?railGuided():(mode==='sandbox'?railSandbox():railChallenge()))+ex+nl;
  A.ui.initRanges(rail);
  updateRail(compute(S)); optTips();
}
/* review 8: the hover of every design option gives its full name and what it stores; the challenge does not say which one Llama uses */
function optTips(){if(!host) return;
  const set=(b,t)=>{if(b.getAttribute('data-tipx')!==t) b.setAttribute('data-tipx',t);};
  $$('.seg-ctl[data-key="attn"] button[data-v]',host).forEach(b=>{const inRail=!!b.closest('#gm-rail'), m=inRail?m70():MODELS[S.model], a=b.dataset.v;
    set(b,designTip(a,m)+(a===m.native&&mode!=='challenge'?tx(` ${m.name} uses it.`,`${m.name} 用的就是它。`):''));});
  $$('#gm-rail [data-lever]',host).forEach(b=>set(b,designTip(b.dataset.lever,m70())));}
function cue(){$$('.cue',host).forEach(el=>el.classList.remove('cue')); if(mode!=='guided') return;
  const id=STEPS[step].target, el=id&&document.getElementById(id); if(el){void el.offsetWidth; el.classList.add('cue');}}
function updateRail(r1){
  if(!host) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('#gm-rail [data-obs]',host);
    if(box){const show=!!s.obs&&(s.showObs?s.showObs():(!s.expect||s.expect()));
      box.hidden=!show;
      if(show){const lab=`<div class="eyebrow">${esc(L(s.label||{en:'Observed',zh:'观察到'}))}</div>`, html=NB(lab+s.obs(r1));
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}}}
    $$('#gm-rail [data-nav="next"]',host).forEach(nb=>{nb.disabled=(!!s.expect&&!s.expect())||(!!s.gate&&!s.gate());});
    const sm=$('#gm-rail [data-nav="showme"]',host); if(sm) sm.hidden=!!(s.expect&&s.expect());
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('#gm-rail [data-preset]',host).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const n=$('#gm-sbnote',host); if(n){const h=NB(scenarioLine(r1)); if(n.innerHTML!==h) n.innerHTML=h;}
  } else {
    $$('#gm-rail [data-lever]',host).forEach(b=>{const a=b.dataset.lever; b.setAttribute('aria-pressed',String(chPicked&&a===S.attn)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=chTried.has(a)?lvRes(a):'';});
    const v=$('#gm-goalv',host); if(!v) return;
    const u=chPicked?r1.users:0, met=chPicked&&u>=GOAL;
    v.textContent=chPicked?String(u):'—'; $('#gm-prog i',host).style.width=Math.min(100,u/GOAL*100)+'%';
    $('#gm-goal',host).classList.toggle('met',met); $('#gm-prog',host).classList.toggle('met',met);
    const sb=$('#gm-rail [data-act="submit"]',host); if(sb) sb.disabled=!chPicked;
    const sn=$('#gm-subnote',host); if(sn) sn.textContent=chPicked?tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。'):tx('Pick a design first. Submitting then tells you whether you met the goal, and why.','先选一种设计。提交后会告诉你是否达标，以及原因。');
    if(chSubmitted&&met) chReveal=true;
    const res=$('#gm-chres',host), h=NB(chResultHTML()); if(res&&res._h!==h){res.innerHTML=h; res._h=h;}
  }
  segSync();
}
function segSync(){
  $$('.seg-ctl[data-key]',host).forEach(g=>{const k=g.dataset.key; if(k==='attn'){const v=(mode==='challenge'&&!chPicked)?'':(g.closest('#gm-rail')&&mode==='guided'&&S.model!==INIT.model?'':S.attn); A.ui.setPressed(g,v);} else if(k==='n'||k==='wp') A.ui.setPressed(g,String(S[k]));});
}

/* ================= stage ================= */
const MET=[
 {k:'kv',label:()=>tx('KV heads per layer','每层 KV head 数'),tip:()=>tx('This is the number of KV heads in each layer. Each token stores one key and one value per KV head in every layer.','这是每层的 KV head 数。每个 token 在每一层、每个 KV head 上存一个 key 和一个 value。')},
 {k:'tok',label:()=>tx('KV cache per token','每个 token 的 KV cache'),tip:()=>tx('One token of one user’s context needs this much KV cache, all layers together.','一个用户上下文里的一个 token，所有层加起来要存这么多 KV cache。')},
 {k:'gpu',label:()=>tx('Per token, on each GPU','每个 token 在每张 GPU 上'),tip:()=>tx('Each GPU stores this much of one token’s KV cache: its share when the KV heads are split across the GPUs, or a whole copy when they cannot be.','每张 GPU 为一个 token 存这么多 KV cache：KV head 能分开时是它分到的那一份，分不开时是一整份副本。')},
 {k:'users',label:()=>tx('Most users that fit','最多用户数'),tip:()=>tx('This many users fit at the current context length, in the memory each GPU has left after the weights and the runtime overhead.','每张 GPU 的可用显存扣掉权重和运行时开销后，按当前的上下文长度，最多能装下这么多用户。')}
];
function metricVals(r1){
  const hid=gpuHidden(), ctx=ctxK(r1.s.ctx), n=r1.s.n;
  if(r1.none) return {kv:['—',tx('pick a design in the task card','先在任务卡里选一种设计')],tok:['—',''],gpu:['—',''],users:['—','']};
  const kvSub=r1.mla?tx('one compressed vector instead','改存一个压缩向量'):(r1.a==='mha'?tx(`one for each of the ${r1.m.heads} query heads`,`${r1.m.heads} 个 query head 各有一个`):(r1.G===1?tx(`shared by all ${r1.m.heads} query heads`,`${r1.m.heads} 个 query head 共用`):tx(`${r1.m.heads} query heads, ${r1.group} per KV head`,`${r1.m.heads} 个 query head，每 ${r1.group} 个共用一个 KV head`)));
  const tokSub=r1.mla?tx(`${r1.m.layers} layers, one compressed vector in each`,`${r1.m.layers} 层，每层一个压缩向量`):tx(`${r1.m.layers} layers, ${r1.G} KV ${r1.G===1?'head':'heads'} in each`,`${r1.m.layers} 层，每层 ${r1.G} 个 KV head`);
  let gpuSub;
  if(n===1) gpuSub=tx('one GPU holds all of it','一张 GPU 存全部');
  else if(r1.mla) gpuSub=tx('a full copy of the compressed vector','压缩向量的一整份副本');
  else if(r1.G>=n) gpuSub=tx(`${r1.headsPerGpu} of the ${r1.G} KV heads`,`${r1.G} 个 KV head 里的 ${r1.headsPerGpu} 个`);
  else gpuSub=r1.G===1?tx('a copy of the only KV head','唯一那个 KV head 的副本'):tx(`a copy of 1 of the ${r1.G} KV heads`,`${r1.G} 个 KV head 之一的副本`);
  const uSub=!r1.weightsFit?tx('the weights alone do not fit','光权重就放不下'):tx(`at ${ctx} of context each`,`每人 ${ctx} 上下文时`);
  // parts 1-4: the initial setup's count (MHA) is in part 1's text, so it shows; every other design waits for part 5 (R3.8)
  const hidU=hid&&!(atInit()&&r1.a===INIT.attn);
  return {kv:[r1.mla?'—':String(r1.G),kvSub],tok:[kib(r1.perTok),tokSub],gpu:hid?['?',hideNote()]:[kib(r1.perGpuTok),gpuSub],
    users:hidU?['?',tx('other designs: shown in part 5','其他设计第 5 部分显示')]:[int(r1.users),hid?tx(`the initial setup, at ${ctx} of context each`,`初始设置，每人 ${ctx} 上下文时`):uSub]};
}
const shownNum=s=>{const x=parseFloat(String(s).replace(/,/g,'')); return isFinite(x)?x:NaN;};
const decs=s=>{const m=String(s).match(/\.(\d+)/); return m?m[1].length:0;};
function drawMetrics(r1){
  const hm=$('#gm-metrics',host); if(!hm) return;
  const V=metricVals(r1);
  MET.forEach(m=>{const el=hm.querySelector(`[data-k="${m.k}"]`), [v,sub]=V[m.k], d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').textContent=sub; el.setAttribute('data-tipx',m.tip());
    const prev=prevShown[m.k];
    const dt=userAct&&prev!=null&&v!=='?'&&prev!=='?'?A.fmt.delta(shownNum(prev),shownNum(v),{dp:Math.max(decs(prev),decs(v))}):'';
    clearTimeout(d._t); d.textContent=dt; d.classList.toggle('on',!!dt); if(dt) d._t=setTimeout(()=>d.classList.remove('on'),2600);
    prevShown[m.k]=v;});
  fitMetrics();
}
/* long values (68.625 KiB) shrink together until each fits its card, never below 11 px (R7.2, R7.4) */
function fitMetrics(){const vs=$$('#gm-metrics .metric .v',host); if(!vs.length||!vs[0].clientWidth) return;
  vs.forEach(e=>{e.style.fontSize='';}); let k=1; vs.forEach(e=>{if(e.scrollWidth>e.clientWidth+0.5) k=Math.min(k,e.clientWidth/e.scrollWidth);});
  if(k<1){const f=parseFloat(getComputedStyle(vs[0]).fontSize); vs.forEach(e=>{e.style.fontSize=Math.max(11,Math.floor(f*k*10)/10)+'px';});}}
function clearBadges(){if(!host) return; $$('#gm-metrics .metric .d',host).forEach(d=>{clearTimeout(d._t); d.textContent=''; d.classList.remove('on');});}

function status(r1){
  if(mode==='challenge'){
    if(!chPicked) return {cls:'neutral',text:tx('Pick a design','先选一种设计'),short:tx('Pick a design','先选一种设计'),tip:tx(`The goal: at least ${GOAL} users fit at ${ctxK(CH_BASE.ctx)} each. Pick a design in the task card.`,`目标：每人 ${ctxK(CH_BASE.ctx)} 上下文时至少装下 ${GOAL} 个用户。先在任务卡里选一种设计。`)};
    const met=r1.users>=GOAL;
    return {cls:met?'ok':'bad',text:met?tx(`Goal met: ${r1.users} users fit`,`达成目标：装下 ${r1.users} 个用户`):tx(`Not yet: ${users(r1.users)} ${r1.users===1?'fits':'fit'}`,`还没达到：只装下 ${r1.users} 个用户`),short:met?tx('Goal met','达成目标'):tx('Not yet','还没达到'),
      tip:tx(`The goal: at least ${GOAL} users fit at ${ctxK(CH_BASE.ctx)} each, with as many KV heads as possible.`,`目标：每人 ${ctxK(CH_BASE.ctx)} 上下文时至少装下 ${GOAL} 个用户，KV head 越多越好。`)};}
  // parts 1-4: the initial setup's count (MHA) is known from part 1, so its badge shows it like the metric does
  const ask=gpuHidden()&&!pred&&!skipped&&!seen&&step===3;
  if(ask||(gpuHidden()&&!(atInit()&&r1.a===INIT.attn))){const t=ask?tx('Predict first','先预测'):tx('Users: part 5','用户数：第 5 部分');
    return {cls:'neutral',text:t,short:t,
      tip:pred?tx('How many users fit with designs other than MHA stays hidden until part 5, where you check your prediction.','除了 MHA，其他设计能装下多少用户先隐藏，到第 5 部分验证预测时再显示。')
         :(skipped||seen?tx('How many users fit with designs other than MHA stays hidden until part 5.','除了 MHA，其他设计能装下多少用户先隐藏，到第 5 部分再显示。')
              :tx('How many users fit with designs other than MHA stays hidden until part 5, which checks the prediction you make in part 4.','除了 MHA，其他设计能装下多少用户先隐藏：你在第 4 部分预测，到第 5 部分验证时再显示。'))};}
  const tip=tx('This badge shows the most users that fit at the current context length: the memory each GPU has left after the weights and the runtime overhead, divided by what one user’s KV cache takes on that GPU.','这个标记显示按当前上下文长度最多能装下几个用户：每张 GPU 扣掉权重和运行时开销后剩下的显存，除以一个用户的 KV cache 在这张 GPU 上占的量。');
  if(!r1.weightsFit) return {cls:'bad',text:tx('No user fits: the weights alone do not fit','一个用户也装不下：光权重就放不下'),short:tx('Weights too big','权重放不下'),tip};
  if(r1.users<1) return {cls:'bad',text:tx(`No user fits at ${ctxK(r1.s.ctx)} each`,`每人 ${ctxK(r1.s.ctx)} 上下文时一个用户也装不下`),short:tx('No user fits','一个也装不下'),tip};
  return {cls:'ok',text:tx(`At most ${users(r1.users)} ${r1.users===1?'fits':'fit'}`,`最多装下 ${r1.users} 个用户`),short:tx(`${users(r1.users)}`,`${r1.users} 个用户`),tip};
}
function drawStatus(r1){const st=status(r1), el=$('#gm-status',host); if(!el) return; el.className='status '+st.cls; el.textContent=st.text; el.setAttribute('data-tipx',st.tip);}

/* ----- the main visual: one layer's heads, the keys and values they read, and (from part 5) the GPUs ----- */
const tpView=r1=>!gpuHidden()&&r1.s.n>1;
/* the KV heads carry their number when there is room for it (16 or fewer blocks across): copies then show which KV head they are */
const kvNums=(r1,W)=>!r1.none&&!r1.mla&&W/Math.max(r1.G,tpView(r1)?r1.s.n:1)>=12&&r1.G<=16;
function legendHTML(r1){
  const it=(sw,txt,tip)=>`<span tabindex="0" data-tipx="${esc(tip)}"><i class="gm-sw ${sw}"></i><span>${txt}</span></span>`, m=r1.m, items=[];
  const el=host&&$('#gm-diag',host), nums=kvNums(r1,Math.max(200,Math.floor((el&&el.clientWidth)||560)));
  items.push(it('gm-sw-q',tx('query head','query head'),tx(`Each gray square is one query head. ${m.name} has ${m.heads} in every layer; each compares the new token’s query with the keys it reads.`,`每个灰色小方块是一个 query head。${m.name} 每层有 ${m.heads} 个，每个都拿新 token 的 query 去和它读的 key 比较。`)));
  if(!r1.none&&!r1.mla) items.push(it('gm-sw-k',tx('KV head: a key and a value','KV head：一个 key 和一个 value'),tx(`Each amber block is one KV head’s key and value for one token in one layer, 2 × ${m.headDim} numbers${nums?'; the number under it says which KV head it is':''}. Amber widths are to scale. ${designTip(r1.a,m)}`,`每个琥珀色方块是一个 token 在一层里、一个 KV head 的 key 和 value，共 2 × ${m.headDim} 维${nums?'；下面的数字表示它是第几个 KV head':''}。琥珀色的宽度按比例画。${designTip(r1.a,m)}`)));
  if(r1.mla) items.push(it('gm-sw-l',tx('compressed vector and RoPE key','压缩向量和 RoPE key'),tx(`The long amber block is the compressed vector of MLA, multi-head latent attention (${r1.dc} numbers); the short one next to it is the RoPE key (${r1.dr} numbers). Widths are to scale: one KV head’s key and value would be ${2*r1.m.headDim} numbers wide.`,`长的琥珀色块是 MLA（multi-head latent attention）的压缩向量（${r1.dc} 维），旁边短的是 RoPE key（${r1.dr} 维）。宽度按比例画，一个 KV head 的 key 和 value 合起来有 ${2*r1.m.headDim} 维那么宽。`)));
  if(!r1.none) items.push(r1.mla?it('gm-sw-dash',tx('dashed line: rebuilt from it','虚线：从它还原'),tx('A dashed line means the head’s key and value are rebuilt from the compressed vector; at decode steps, engines skip the rebuilding and compute attention on the compressed vector directly.','虚线表示这个 head 的 key 和 value 从压缩向量还原；decode 步里引擎不做这次还原，直接在压缩向量上算 attention。'))
                                :it('gm-sw-line',tx('line: the KV head it reads','细线：它读的 KV head'),tx('A thin line joins a query head to the KV head whose keys and values it reads.','细线把 query head 和它要读 key、value 的那个 KV head 连起来。')));
  if(!r1.none&&tpView(r1)){items.push(it('gm-sw-gpu',tx('dashed box: one GPU','虚线框：一张 GPU'),tx('A dashed box is one GPU: the query heads it computes and the keys and values it stores.','每个虚线框是一张 GPU：它算的 query head，和它存的 key、value。')));
    if(r1.copies>1){
      items.push(it('gm-sw-copy',tx('hatched: a copy','斜纹：副本'),r1.mla?tx('Hatched amber is a copy: every GPU stores the same compressed vector.','带斜纹的琥珀色是副本：每张 GPU 都存同一个压缩向量。')
        :tx(`Hatched amber is a copy: several GPUs store the same KV head${nums?', and the number under a block says which one':''}.`,`带斜纹的琥珀色是副本：几张 GPU 存着同一个 KV head${nums?'，方块下面的数字表示是哪一个':''}。`)));
      items.push(it('gm-sw-br',r1.mla?tx('line below: GPUs that hold the same compressed vector','下方横线：存同一个压缩向量的 GPU'):tx('line below: GPUs that hold the same KV head','下方横线：存同一个 KV head 的 GPU'),
        tx('The line under the blocks joins the GPUs that store the same keys and values.','方块下面的横线把存着同一份 key 和 value 的 GPU 连起来。')));}}
  return items.join('');
}
function diagramSVG(r1,W){
  const m=r1.m, H=m.heads, p=W/H, s=Math.max(1.5,Math.min(9,p*0.72)), tp=!r1.none&&tpView(r1), n=tp?r1.s.n:1;
  const yq=tp?18:6, yk=yq+s+46, hk=11, wn=(p*0.92)/(2*m.headDim), kw=2*m.headDim*wn, nums=kvNums(r1,W), yl=yk+hk+(nums?12:0);
  let g='';
  const qx=i=>i*p+p/2;
  // GPU boxes and labels
  if(tp){for(let b=0;b<n;b++){const x0=b*W/n, x1=(b+1)*W/n; g+=`<rect class="gm-gpu" x="${(x0+1.5).toFixed(1)}" y="${yq-5}" width="${(x1-x0-3).toFixed(1)}" height="${(yl+6-(yq-5)).toFixed(1)}" rx="3"/>`+
    `<text class="gm-gl" x="${((x0+x1)/2).toFixed(1)}" y="${yq-8}" text-anchor="middle">${W/n>=38?`GPU ${b+1}`:b+1}</text>`;}}
  // what is stored, and which query heads read it
  const blocks=[];            // {x, w, kind:'k'|'l'|'r', copy, reads:[query head indexes]}
  if(!r1.none){
    if(r1.mla){const cw=r1.dc*wn, rw=r1.dr*wn, gap=Math.max(1,p*0.15);
      for(let b=0;b<n;b++){const cx=(b+0.5)*W/n, x0=cx-(cw+gap+rw)/2, rd=[]; for(let i=b*H/n;i<(b+1)*H/n;i++) rd.push(i);
        blocks.push({x:x0,w:cw,kind:'l',copy:n>1,reads:rd,fan:true}); blocks.push({x:x0+cw+gap,w:rw,kind:'r',copy:n>1,reads:[]});}}
    else {const G=r1.G, k=H/G;
      if(G>=n){for(let j=0;j<G;j++){const cx=(j*k+k/2)*p, rd=[]; for(let i=j*k;i<(j+1)*k;i++) rd.push(i); blocks.push({x:cx-kw/2,w:kw,kind:'k',copy:false,reads:rd,head:j});}}
      else {for(let b=0;b<n;b++){const cx=(b+0.5)*W/n, rd=[]; for(let i=b*H/n;i<(b+1)*H/n;i++) rd.push(i); blocks.push({x:cx-kw/2,w:kw,kind:'k',copy:true,reads:rd,head:Math.floor(b*G/n)});}}}
  }
  // wires first, so squares sit on top
  // on a phone the dashed fan closes into a gray cone: sparser dashes keep it readable as dashed lines (review A r11 #16)
  const fanC=p<6?'gm-fan gm-fan-n':'gm-fan';
  blocks.forEach(bk=>{const cx=bk.x+bk.w/2; bk.reads.forEach(i=>{g+=`<line class="${bk.fan?fanC:'gm-wire'}" x1="${qx(i).toFixed(1)}" y1="${(yq+s).toFixed(1)}" x2="${cx.toFixed(1)}" y2="${yk}"/>`;});});
  for(let i=0;i<H;i++) g+=`<rect class="gm-q" x="${(i*p+(p-s)/2).toFixed(2)}" y="${yq}" width="${s.toFixed(2)}" height="${s.toFixed(2)}" rx="${s>4?1:0}"/>`;
  blocks.forEach(bk=>{g+=`<rect class="gm-k${bk.kind==='r'?' gm-rope':''}" x="${bk.x.toFixed(2)}" y="${yk}" width="${Math.max(1,bk.w).toFixed(2)}" height="${hk}" rx="1"/>`;
    if(bk.copy) g+=`<rect class="gm-hatch" x="${bk.x.toFixed(2)}" y="${yk}" width="${Math.max(1,bk.w).toFixed(2)}" height="${hk}" rx="1"/>`;});
  // which KV head each block is (copies carry the same number)
  if(nums) blocks.forEach(bk=>{if(bk.kind==='k') g+=`<text class="gm-kl" x="${(bk.x+bk.w/2).toFixed(1)}" y="${yk+hk+10}" text-anchor="middle">${bk.head+1}</text>`;});
  // lines joining copies of the same keys and values
  let yb=yl+(tp?12:8);
  if(tp&&!r1.none&&r1.copies>1){const groups=r1.mla?[[...Array(n).keys()]]:[...Array(r1.G).keys()].map(j=>[...Array(n).keys()].filter(b=>Math.floor(b*r1.G/n)===j));
    groups.forEach(gr=>{const xs=gr.map(b=>(b+0.5)*W/n), x0=xs[0], x1=xs[xs.length-1];
      g+=`<path class="gm-br" d="M${x0.toFixed(1)} ${yb-5} V${yb} H${x1.toFixed(1)} V${yb-5}"/>`; xs.slice(1,-1).forEach(x=>{g+=`<path class="gm-br" d="M${x.toFixed(1)} ${yb-5} V${yb}"/>`;});});}
  const Hh=Math.ceil(yb+(tp?4:2));
  const aria=r1.none?tx(`One layer of ${m.name}: ${H} query heads; no design picked yet.`,`${m.name} 的一层：${H} 个 query head；还没选设计。`):diagCaption(r1,true);
  return `<svg class="gm-svg" viewBox="0 0 ${W} ${Hh}" width="${W}" height="${Hh}" role="img" aria-label="${esc(aria)}"><defs><pattern id="gm-hp" width="3" height="3" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="1.2" height="3" class="gm-hp-l"/></pattern></defs>${g}</svg>`;
}
function diagCaption(r1,long){
  const m=r1.m;
  if(r1.none) return tx(`One layer of ${m.name}: ${m.heads} query heads. Pick a design in the task card.`,`${m.name} 的一层：${m.heads} 个 query head。先在任务卡里选一种设计。`);
  const head=r1.mla?tx(`One layer of ${m.name}: ${m.heads} query heads and the compressed vector their keys and values are rebuilt from`,`${m.name} 的一层：${m.heads} 个 query head，和还原 key、value 用的压缩向量`)
                   :tx(`One layer of ${m.name}: ${m.heads} query heads and the ${r1.G} KV ${r1.G===1?'head':'heads'} they read`,`${m.name} 的一层：${m.heads} 个 query head 和它们读的 ${r1.G} 个 KV head`);
  const cp=tpView(r1)&&r1.copies>1?(r1.mla?tx(', drawn once on every GPU','（每张 GPU 上各画一份）'):(r1.G===1?tx(', drawn once on every GPU that stores it','（存它的每张 GPU 上各画一份）'):tx(', each drawn once on every GPU that stores it','（每个 KV head 在存它的每张 GPU 上各画一份）'))):'';
  return long?head+cp+tx('. ','。')+gpuLine(r1):head+cp;
}
function gpuLine(r1){
  const n=r1.s.n, per=kib(r1.perGpuTok), L_=r1.m.layers;
  if(r1.none||!tpView(r1)) return n===1&&!r1.none&&!gpuHidden()?(r1.mla?tx(`One GPU stores the whole compressed vector: ${per} per token, all ${L_} layers.`,`一张 GPU 存整个压缩向量：每个 token ${per}（${L_} 层合计）。`)
                                                                  :tx(`One GPU stores every KV head: ${per} per token, all ${L_} layers.`,`一张 GPU 存全部 KV head：每个 token ${per}（${L_} 层合计）。`)):'';
  if(r1.mla) return tx(`Each GPU stores a full copy of the compressed vector: ${per} per token, all ${L_} layers.`,`每张 GPU 都存一整份压缩向量：每个 token ${per}（${L_} 层合计）。`);
  if(r1.G>=n) return tx(`Each GPU stores ${r1.headsPerGpu} of the ${r1.G} KV heads: ${per} per token, all ${L_} layers.`,`每张 GPU 存 ${r1.G} 个 KV head 里的 ${r1.headsPerGpu} 个：每个 token ${per}（${L_} 层合计）。`);
  if(r1.G===1) return tx(`Each GPU stores a copy of the only KV head: ${per} per token, all ${L_} layers.`,`每张 GPU 都存唯一那个 KV head 的一份副本：每个 token ${per}（${L_} 层合计）。`);
  return tx(`Each GPU stores 1 KV head, and each KV head is on ${r1.copies} GPUs: ${per} per token, all ${L_} layers.`,`每张 GPU 存 1 个 KV head，每个 KV head 在 ${r1.copies} 张 GPU 上各有一份：每个 token ${per}（${L_} 层合计）。`);
}
function drawDiagram(r1){
  const el=$('#gm-diag',host); if(!el) return;
  const W=Math.max(200,Math.floor(el.clientWidth||560));
  el.innerHTML=diagramSVG(r1,W); el._w=W;
  $('#gm-cap',host).textContent=NB(diagCaption(r1,false));
  const gl=$('#gm-gpuline',host), t=NB(gpuLine(r1)); gl.textContent=t; gl.hidden=!t;
  $('#gm-legend',host).innerHTML=NB(legendHTML(r1));
}

/* ----- the chart: one token's KV cache for each design, in total and on each GPU ----- */
function chartRows(){
  const m=MODELS[S.model];
  if(mode==='guided'){const base=step===0?['mha']:(step<=4?['mha','gqa16','gqa8','gqa4','mqa']:DESIGNS); return m===m70()?base:designsOf(m);}
  return designsOf(m);
}
function chartHTML(r1){
  if(mode==='challenge'||r1.none) return '';
  const rows=chartRows(), n=S.n, rs=rows.map(a=>compute({...S,attn:a}));
  const hideAll=gpuHidden(), p5=mode==='guided'&&step===4&&!observed[4], p6=mode==='guided'&&step===5&&!observed[5];
  const showGpu=a=>!hideAll&&(!p5||a===INIT.attn||tried5.has(a))&&!(p6&&a==='mla');
  const mt=Math.max(...rs.map(x=>x.perTok)), mg=Math.max(...rs.filter(x=>showGpu(x.a)).map(x=>x.perGpuTok),1);
  const colG=n===1?tx('on the one GPU','这一张 GPU'):tx(`on each of the ${n} GPUs`,`${n} 张 GPU 每张`);
  const cell=(v,max,shown)=>shown?`<span class="gm-cc"><span class="gm-ct"><i style="width:${(v/max*100).toFixed(2)}%"></i></span><b class="mono">${kib(v)}</b></span>`:`<span class="gm-cc gm-masked"><span class="gm-ct"></span><b class="mono">?</b></span>`;
  const tipOf=x=>{const m=x.m;
    if(x.mla) return designTip('mla',m)+(x.hypo?tx(` For ${m.name} this is a what-if with DeepSeek-V3’s sizes.`,`对 ${m.name} 来说，这是用 DeepSeek-V3 的大小做的假设。`):tx(` ${m.name} uses it.`,`${m.name} 用的就是它。`));
    return designTip(x.a,m)+(x.native?tx(` ${m.name} uses it.`,`${m.name} 用的就是它。`):'');};
  const uses=x=>x.native&&!x.mla?`<small class="gm-uses">${tx(`${x.m.name} uses it`,`${x.m.name} 用的`)}</small>`:'';
  const note=hideAll?tx(' The per-GPU column is shown in part 5.','每张 GPU 的那一列在第 5 部分显示。'):(p5||p6?tx(' A ? marks a design not yet picked in this part.','“?”表示这一部分里还没选过的设计。'):'');
  return `<div class="gm-ch-title">${tx(`One token’s KV cache, all ${r1.m.layers} layers, by attention design`,`一个 token 的 KV cache（${r1.m.layers} 层合计），按注意力设计`)}</div>`+
    `<div class="gm-ch-grid" role="table" aria-label="${esc(tx('One token’s KV cache for each design','每种设计下一个 token 的 KV cache'))}"><div class="gm-ch-h" role="row"><span role="columnheader">${tx('Design','设计')}</span><span role="columnheader">${tx('in total','合计')}</span><span role="columnheader">${colG}</span></div>`+
    rs.map(x=>`<div class="gm-ch-r${x.a===r1.a?' cur':''}" role="row"><span class="gm-ch-l" role="rowheader"><span class="def" tabindex="0" data-tipx="${esc(tipOf(x))}">${esc(dName(x.a,x.m))}</span>${uses(x)}${x.a===r1.a?`<em>${tx('current','当前')}</em>`:''}</span>${cell(x.perTok,mt,true)}${cell(x.perGpuTok,mg,showGpu(x.a))}</div>`).join('')+
    `</div><p class="gm-ch-note">${tx('Bars in each column are to scale within that column; the outlined row is the current design.','每一列里的条按这一列的比例画；带框的一行是当前的设计。')}${note}</p>`;
}
function drawChart(r1){const el=$('#gm-chart',host); if(!el) return; const h=NB(chartHTML(r1)); if(el._h!==h){el.innerHTML=h; el._h=h;} el.hidden=!h;}

/* ----- Show the math: every line redoes from its own shown operands (R6.6) ----- */
/* Weights (total t and per GPU w) and the room, all with the fewest decimals d that make two redos work (R6.6): t ÷ n rounds
   to w (review B r12 N4: 378.0 ÷ 8 = 47.25 did not give the 47.2 shown), and ⌊room ÷ per⌋ is the engine's answer (not needed
   when nothing fits). null when no d up to 4 works with this per-user value. */
function roomFit(r1,per){
  const use=r(r1.usable/GiB,1), oh=r1.oh/GiB, W=r1.wTotal/GiB, n=r1.s.n;
  for(let d=1;d<=4;d++){const t=r(W,d), w=r(W/n,d), room=r(use-w-oh,d);
    const redo=Math.abs(r(t/n,d)-w)<1e-9, fits=r1.budget<=0||Math.floor(room/per+1e-9)===r1.users;
    if(redo&&fits) return {use,t,w,oh,room,d};}
  return null;
}
/* one user's KV cache on each GPU, in GiB with 3 decimals (the KV cache memory lab's 8.578 GiB), more when the room and the floor
   need them, chosen together with the room's decimals */
function perUserShown(r1){const g=r1.perUserGpu/GiB;
  for(let d=3;d<=6;d++){const v=r(g,d); if(Math.abs(v-g)<1e-12||roomFit(r1,v)) return {v,d};}
  return {v:r(g,6),d:6};}
function roomShown(r1,per){
  const R=roomFit(r1,per); if(R) return R;
  const use=r(r1.usable/GiB,1), oh=r1.oh/GiB, d=4, w=r(r1.wTotal/GiB/r1.s.n,d);
  return {use,t:r(r1.wTotal/GiB,d),w,oh,room:r(use-w-oh,d),d};
}
/* the weights' total as Show the math prints it, for the weight-precision tip next to it */
const wTotalShown=r1=>{if(r1.none) return fmtF(r(r1.wTotal/GiB,1),1); const R=roomShown(r1,perUserShown(r1).v); return fmtF(R.t,R.d);};
/* a quotient with 2 decimals that never rounds up past the whole number the floor gives (9.999 shows 9.99, not 10) */
const q2=q=>{let v=r(q,2); if(Math.floor(v+1e-9)>Math.floor(q+1e-9)) v=Math.floor(q*100+1e-7)/100; return v;};
const sgnD=(v,d)=>(v<0?'−':'')+fmtF(Math.abs(v),d);
function mathLines(r1){
  if(r1.none) return [tx('Pick a design in the task card to see its numbers.','先在任务卡里选一种设计，这里才有它的算式。')];
  const s=r1.s, m=r1.m, T_=(k,x,term)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}"${term?` data-term="${term}"`:''}>${x}</span>`;
  const bytes=T_('kvb',tx(`${KVB} bytes`,`${KVB} 字节`)), layers=T_('model',tx(`${m.layers} layers`,`${m.layers} 层`),'layers');
  const l1=r1.mla?`${tx('KV cache per token','每个 token 的 KV cache')} = ${layers} × (${T_('model',`${r1.dc} latent`,'latent')} + ${T_('model',`${r1.dr} RoPE`,'rope')}) × ${bytes} = ${tx(`${int(r1.perTok)} bytes`,`${int(r1.perTok)} 字节`)} = <b>${kib(r1.perTok)}</b>`
    :`${tx('KV cache per token','每个 token 的 KV cache')} = ${tx('2 (a key and a value)','2（key 和 value）')} × ${layers} × ${T_('attn',tx(`${r1.G} KV ${r1.G===1?'head':'heads'}`,`${r1.G} 个 KV head`))} × ${T_('model',`${m.headDim} head_dim`,'headdim')} × ${bytes} = ${tx(`${int(r1.perTok)} bytes`,`${int(r1.perTok)} 字节`)} = <b>${kib(r1.perTok)}</b>`;
  if(gpuHidden()) return [l1,`<span class="muted">${tx('The lines for each GPU and for users appear in part 5.','每张 GPU 和用户数的算式在第 5 部分显示。')}</span>`];
  const n=s.n, nG=T_('n',gpus(n));
  const pa=x=>tx(` (${x})`,`（${x}）`);
  let l2;
  if(r1.mla) l2=`${tx('On each GPU','每张 GPU 上')} = <b>${kib(r1.perGpuTok)}</b>${n===1?tx(` per token (not divided: ${nG} keeps the whole compressed vector)`,`（每个 token；不拆分，${nG} 存一整份压缩向量）`)
      :tx(` per token (not divided: each of the ${nG} keeps the whole compressed vector)`,`（每个 token；压缩向量分不开，${nG} 每张都存一整份）`)}`;
  else if(n===1) l2=`${tx('On each GPU','每张 GPU 上')} = ${kib(r1.perTok)} ÷ 1${pa(tx(`${nG}: it stores every KV head`,`${nG}，存全部 KV head`))} = <b>${kib(r1.perGpuTok)}</b>`;
  else if(r1.G>=n) l2=`${tx('On each GPU','每张 GPU 上')} = ${kib(r1.perTok)} ÷ ${n}${pa(tx(`${r1.G} KV heads over ${nG}, ${r1.headsPerGpu} on each`,`${r1.G} 个 KV head 分到 ${nG} 上，每张 ${r1.headsPerGpu} 个`))} = <b>${kib(r1.perGpuTok)}</b>`;
  else l2=`${tx('On each GPU','每张 GPU 上')} = ${kib(r1.perTok)} ÷ ${r1.G}${pa(r1.G===1?tx(`1 KV head for ${nG}: a copy on each`,`1 个 KV head 对 ${nG}，每张存一份副本`)
      :tx(`${r1.G} KV heads for ${nG}: one on each, ${r1.copies} copies of each`,`${r1.G} 个 KV head 对 ${nG}，每张一个，每个 ${r1.copies} 份`))} = <b>${kib(r1.perGpuTok)}</b>`;
  const PU=perUserShown(r1), perG=PU.v, per=fmtD(perG,PU.d)+' GiB';
  const l3=`${tx('One user on each GPU','一个用户在每张 GPU 上')} = ${kib(r1.perGpuTok)} × ${T_('ctx',tx(`${int(s.ctx)} tokens`,`${int(s.ctx)} 个 token`))} = <b>${fmtD(perG,PU.d)}</b> ${DEF('GiB',A.terms.gib())}`;
  const R=roomShown(r1,perG), ws=fmtF(R.w,R.d);
  const l4=`${tx('Room for KV cache on each GPU','每张 GPU 留给 KV cache 的显存')} = ${T_('gpu',`${r1.g.mem} GiB`)} × ${T_('util',fmtD(UTIL,2))} − ${T_('wp',tx(`${ws} GiB of weights`,`权重 ${ws} GiB`))} − ${T_('oh',tx(`${OH} GiB of overhead`,`开销 ${OH} GiB`))} = ${fmtD(R.use,1)} − ${ws} − ${OH} = <b>${sgnD(R.room,R.d)} GiB</b>`;
  // the weights from bytes to GiB, with GB next to it (R6.12, review 8: "16.1 GB = 15.0 GiB"); exact counts belong here (R6.18)
  const wt=fmtF(R.t,R.d), gb=fmtF(r(r1.wTotal/1e9,1),1), bw=PREC[s.wp];
  const wline=`<span class="muted">${tx(`(${PL[s.wp]} weights: ${pCount(m)} × ${bw} ${bw===1?'byte':'bytes'} = ${gb} GB = ${wt} GiB, ÷ ${gpus(n)})`,`（${PL[s.wp]} 权重：${pCount(m)} × ${bw} 字节 = ${gb} GB = ${wt} GiB，÷ ${gpus(n)}）`)}</span>`;
  const q=R.room/perG, qs=fmtF(q2(q),2);
  const l5=R.room<=0?`${tx('Most users that fit','最多用户数')} = <b>0</b> ${tx('(no room left after the weights)','（放完权重就没有空间了）')}`:`${tx('Most users that fit','最多用户数')} = ⌊${fmtF(R.room,R.d)} GiB ÷ ${per}⌋ = ⌊${qs}⌋ = <b>${int(r1.users)}</b>`;
  return [l1,l2,l3,l4+tx(' ','')+wline,l5];
}
/* Show the math as a few short blocks (H2, the owner on 2026-10-09: “展开公式……没有人类会喜欢看的”): each line of mathLines becomes a
   block with a lead line in words, the quantity's name and its formula typeset, then the reader's numbers under it; a chain breaks at
   “=” onto rows that line up (R6.29). mathLines itself (and window.__gm.mathText, which the numbers test redoes) is unchanged */
const gmSplitEq=h=>{const out=[]; let d=0, i=0, last=0;
  while(i<h.length){if(h[i]==='<'){const close=h[i+1]==='/', j=h.indexOf('>',i); const self=h[j-1]==='/'||/^<(br|img|input|mspace)\b/i.test(h.slice(i,j)); if(close) d--; else if(!self) d++; i=j+1; continue;}
    if(d===0&&h.startsWith(' = ',i)){out.push(h.slice(last,i)); i+=3; last=i; continue;} i++;}
  out.push(h.slice(last)); return out;};
/* the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): a long formula is cut before each
   operator outside braces and brackets into pieces that wrap onto the next line, indented under its first term, as in the kit's
   .kv-fx (a piece that starts with an operator gets a medium space on each side of it: Chromium drops the space around an mrow's
   leading operator) */
const gmFx=tex=>{const out=[]; let d=0, last=0;
  for(let i=0;i<tex.length;i++){const c=tex[i];
    if(c==='{'||c==='(') d++; else if(c==='}'||c===')') d--;
    else if(tex.startsWith('\\left',i)){d++; i+=4;} else if(tex.startsWith('\\right',i)){d--; i+=5;}
    else if(d===0&&i>last&&(tex.startsWith(' \\times ',i)||tex.startsWith(' + ',i)||tex.startsWith(' - ',i))){out.push(tex.slice(last,i)); last=i+1;}}
  out.push(tex.slice(last));
  return out.length<2?A.ui.math(tex):out.map((x,k)=>`<span class="gm-fp">${A.ui.math(k?x.replace(/^(\\times|[+-]) /,'\\:$1\\: '):x)}</span>`).join('<wbr>');};
function gmBlock(line,lead,fx){const p=gmSplitEq(line); if(p.length<2) return `<div class="gm-mb">${lead?`<p class="gm-ml">${lead}</p>`:''}<p>${line}</p></div>`;
  const rows=[`<span class="gm-mn">${p[0]}</span><span class="gm-me"> = </span><span class="gm-mr${fx?' gm-fxc':''}">${fx?gmFx(fx):p[1]}</span>`].concat((fx?p.slice(1):p.slice(2)).map(x=>`<span></span><span class="gm-me"> = </span><span class="gm-mr">${x}</span>`));
  return `<div class="gm-mb">${lead?`<p class="gm-ml">${lead}</p>`:''}<div class="gm-mg">${rows.join('')}</div></div>`;}
function drawMath(r1){const el=$('#gm-mathbody',host); if(!el) return; const L=mathLines(r1), X=x=>'\\text{'+x+'}';
  if(r1.none||L.length<3){el.innerHTML=NB(L.map((x,i)=>i===0&&!r1.none?gmBlock(x,r1.mla?tx('MLA stores, for each token and layer, one compressed vector and one RoPE key:','MLA 每个 token 每层只存一个压缩向量和一个 RoPE key：'):tx('Every token stores a key and a value for each KV head, in every layer:','每个 token 在每一层、每个 KV head 里，都存一个 key 和一个 value：'),
      r1.mla?tx(`${X('layers')} \\times (${X('latent')} + ${X('RoPE')}) \\times ${X('bytes')}`,`${X('层数')} \\times (${X('latent')} + ${X('RoPE')}) \\times ${X('每个数值的字节数')}`)
        :tx(`2 \\times ${X('layers')} \\times ${X('KV heads')} \\times ${X('head_dim')} \\times ${X('bytes')}`,`2 \\times ${X('层数')} \\times ${X('KV head 数')} \\times ${X('head_dim')} \\times ${X('每个数值的字节数')}`)):`<div class="gm-mb"><p>${x}</p></div>`).join('')); return;}
  const leads=[r1.mla?tx('MLA stores, for each token and layer, one compressed vector and one RoPE key:','MLA 每个 token 每层只存一个压缩向量和一个 RoPE key：'):tx('Every token stores a key and a value for each KV head, in every layer:','每个 token 在每一层、每个 KV head 里，都存一个 key 和一个 value：'),
    r1.mla?tx('The compressed vector cannot be split, so every GPU keeps a whole copy:','压缩向量拆不开，所以每张 GPU 都存一整份：'):tx('The KV heads are split over the GPUs; each GPU keeps only its own heads, or a copy when there are fewer heads than GPUs:','KV head 分到各张 GPU 上，每张只存自己那几个；KV head 比 GPU 少时，每张存一份副本：'),
    tx('One user’s context needs that much for every token it holds:','一个用户的上下文有多少个 token，就要存多少份：'),
    tx('Each GPU’s memory, cut by the utilization limit, minus the weights and the overhead, is what is left for the KV cache:','每张 GPU 的显存先乘上利用率，再减去权重和开销，剩下的才留给 KV cache：'),
    tx('The users that fit are that room divided by one user’s KV cache, rounded down:','能装下的用户数，是这块空间除以一个用户的 KV cache，再向下取整：')];
  const fx=[r1.mla?tx(`${X('layers')} \\times (${X('latent')} + ${X('RoPE')}) \\times ${X('bytes')}`,`${X('层数')} \\times (${X('latent')} + ${X('RoPE')}) \\times ${X('每个数值的字节数')}`)
      :tx(`2 \\times ${X('layers')} \\times ${X('KV heads')} \\times ${X('head_dim')} \\times ${X('bytes')}`,`2 \\times ${X('层数')} \\times ${X('KV head 数')} \\times ${X('head_dim')} \\times ${X('每个数值的字节数')}`),
    r1.mla?null:tx(`\\frac{${X('KV cache per token')}}{${X('GPUs that share it')}}`,`\\frac{${X('每个 token 的 KV cache')}}{${X('分摊它的 GPU 数')}}`),
    tx(`${X('KV cache per token on each GPU')} \\times ${X('tokens')}`,`${X('每张 GPU 上每个 token 的 KV cache')} \\times ${X('token 数')}`),
    tx(`${X('GPU memory')} \\times ${X('utilization')} - ${X('weights')} - ${X('overhead')}`,`${X('显存')} \\times ${X('利用率')} - ${X('权重')} - ${X('开销')}`),
    tx(`\\left⌊\\frac{${X('room for KV cache')}}{${X('one user’s KV cache')}}\\right⌋`,`\\left⌊\\frac{${X('留给 KV cache 的显存')}}{${X('一个用户的 KV cache')}}\\right⌋`)];
  el.innerHTML=NB(L.map((x,i)=>gmBlock(x,leads[i],fx[i])).join(''));}


function drawStage(r1){
  const m=r1.m, t=$('#gm-title',host);
  t.textContent=NB(tx(`${m.name} on ${r1.s.n} × ${r1.g.name}`,`${r1.s.n} × ${r1.g.name} 跑 ${m.name}`));
  // "GQA-8 (grouped-query attention, the design Llama-3.1-70B uses)", "MLA (multi-head latent attention, 512 + 64)": one pair of brackets
  const nat=r1.native&&mode!=='challenge', xtra=r1.hypo?`, ${r1.dc} + ${r1.dr}`:'';
  const nm=r1.none?'':tx(`${DNAME[r1.a]} (${FULL(r1.a)}${xtra}${nat?`, ${usesTxt(m)}`:''})`,`${DNAME[r1.a]}（${FULL(r1.a)}${xtra?`，${r1.dc} + ${r1.dr}`:''}${nat?`，${usesTxt(m)}`:''}）`);
  const sub=r1.none?tx(`Pick a design; ${ctxK(r1.s.ctx)} of context per user, weights in ${PL[r1.s.wp]}`,`先选一种设计；每人 ${ctxK(r1.s.ctx)} 上下文，权重 ${PL[r1.s.wp]}`)
    :(r1.mla?tx(`${nm}: one compressed vector per token and layer; ${ctxK(r1.s.ctx)} of context per user, weights in ${PL[r1.s.wp]}`,`${nm}：每个 token 每层一个压缩向量；每人 ${ctxK(r1.s.ctx)} 上下文，权重 ${PL[r1.s.wp]}`)
      :(r1.a==='mha'?tx(`${nm}: each of the ${m.heads} query heads has its own KV head; ${ctxK(r1.s.ctx)} of context per user, weights in ${PL[r1.s.wp]}`,`${nm}：${m.heads} 个 query head 各有自己的 KV head；每人 ${ctxK(r1.s.ctx)} 上下文，权重 ${PL[r1.s.wp]}`)
      :tx(`${nm}: ${m.heads} query heads share ${r1.G} KV ${r1.G===1?'head':'heads'}; ${ctxK(r1.s.ctx)} of context per user, weights in ${PL[r1.s.wp]}`,`${nm}：${m.heads} 个 query head 共用 ${r1.G} 个 KV head；每人 ${ctxK(r1.s.ctx)} 上下文，权重 ${PL[r1.s.wp]}`)));
  $('#gm-sub',host).textContent=NB(sub);
  drawStatus(r1); drawDiagram(r1); drawChart(r1); drawMetrics(r1); drawMath(r1);
}

/* ================= settings panel sync ================= */
function notes(r1){
  const set=(id,h)=>{const el=$('#'+id,host); if(el&&el._h!==h){el.innerHTML=h; el._h=h;}}, m=r1.m;
  set('gm-model-note',NB(m.native==='mla'?tx(`A mixture of experts with multi-head latent attention (MLA): ${m.layers} layers, ${m.heads} query heads, a compressed vector and a RoPE key of ${m.kvLora} + ${m.ropeDim} numbers per layer, ${big(m.params)} parameters. Each token uses only ${big(m.active)} of them, but all ${big(m.params)} must sit in GPU memory.`,`用 multi-head latent attention（MLA）的混合专家（MoE）模型：${m.layers} 层，${m.heads} 个 query head，每层一个压缩向量和一个 RoPE key，共 ${m.kvLora} + ${m.ropeDim} 维，${big(m.params)} 个参数。每个 token 只用到其中 ${big(m.active)} 个，但 ${big(m.params)} 个全都要放进显存。`)
    :tx(`${m.layers} layers, ${m.heads} query heads, ${bigH(m.params)} parameters; the real model uses ${m.kvHeads} KV heads of ${m.headDim} numbers (${DNAME[m.native]}). Picking a model also picks the attention design it uses.`,`${m.layers} 层，${m.heads} 个 query head，${bigH(m.params)} 个参数；真实模型用 ${m.kvHeads} 个 KV head（${DNAME[m.native]}，每个 ${m.headDim} 维）。换模型时，注意力设计也换成这个模型用的设计。`)));
  // the full name first (review 8); the challenge does not say which design Llama uses (no answer from Settings)
  const ch=mode==='challenge';
  let an='';
  if(r1.none) an=tx('No design picked yet.','还没选设计。');
  else if(m.native==='mla') an=tx('DeepSeek-V3 is built with multi-head latent attention (MLA); the other designs would make it a different model.','DeepSeek-V3 就是用 multi-head latent attention（MLA）设计的；换成别的设计就是另一个模型了。');
  else if(r1.mla) an=tx(`Multi-head latent attention (MLA), a what-if: DeepSeek-V3’s compressed vector and RoPE key (${r1.dc} + ${r1.dr} numbers) in this model’s ${m.layers} layers.`,`multi-head latent attention（MLA），假设的：在这个模型的 ${m.layers} 层里用 DeepSeek-V3 的压缩向量和 RoPE key（${r1.dc} + ${r1.dr} 维）。`);
  else {const f=tx(`${cap1(FULL(r1.a))} (${DNAME[r1.a]})`,`${FULL(r1.a)}（${DNAME[r1.a]}）`);
    const shape=r1.a==='mha'?tx(`${r1.G} KV heads, one per query head`,`${r1.G} 个 KV head，每个 query head 一个`):(r1.G===1?tx(`one KV head shared by all ${m.heads} query heads`,`${m.heads} 个 query head 共用一个 KV head`):tx(`${r1.G} KV heads, each shared by ${r1.group} query heads`,`${r1.G} 个 KV head，每个由 ${r1.group} 个 query head 共用`));
    const tail=ch?'':(r1.native?tx(`, ${usesTxt(m)}`,`，${usesTxt(m)}`):(r1.a===INIT.attn&&S.model===INIT.model?tx(', the initial setup (a what-if; the real model uses GQA-8)','，初始设置（这是假设，真实模型用 GQA-8）'):tx(', a what-if for this model','，对这个模型是假设的')));
    an=tx(`${f}: ${shape}${tail}.`,`${f}：${shape}${tail}。`);}
  set('gm-attn-note',NB(an));
  // review R14 2-6: the vendor's figure read as GiB, and why (the kit's GiB text says the same); GiB is defined here, its first use
  const mem=r1.g.mem, gibT=DEF('GiB',A.terms.gib());
  set('gm-gpu-note',tx(`The vendor’s ${mem} GB is read as ${mem} ${gibT}, as in the lab “${A.shell.labLink('kv-cache')}”: a GPU’s usable memory is close to that many GiB.`,
                       `厂商标的 ${mem} GB 按 ${mem} ${gibT} 计，和“${A.shell.labLink('kv-cache')}”Lab 一样：显卡实际能用的显存接近同样多的 GiB。`));
  set('gm-n-note',r1.s.n===1?tx(`One GPU computes all ${m.heads} query heads and holds all the weights.`,`一张 GPU 算全部 ${m.heads} 个 query head，存全部权重。`)
    :tx(`Each GPU computes 1/${r1.s.n} of the ${m.heads} query heads and holds 1/${r1.s.n} of the weights.`,`每张 GPU 算 ${m.heads} 个 query head 里的 1/${r1.s.n}，存 1/${r1.s.n} 的权重。`));
}
function syncControls(r1){
  const ms=$('#gm-model',host); if(ms&&ms.value!==S.model) ms.value=S.model;
  const gs=$('#gm-gpu',host); if(gs&&gs.value!==S.gpu) gs.value=S.gpu;
  $$('#gm-ctl-attn button[data-v]',host).forEach(b=>{b.disabled=S.model==='deepseek-v3'&&b.dataset.v!=='mla';});
  const ci=$('#gm-ctl-ctx-in',host); if(ci){const i=CTXS.indexOf(S.ctx); if(+ci.value!==i) ci.value=i; A.ui.setRange(ci); const o=ci.closest('.ctl').querySelector('output'); if(o) o.textContent=tx(`${ctxK(S.ctx)} tokens`,`${ctxK(S.ctx)} 个 token`);}
  const lock=mode==='challenge', c=$('#gm-controls',host); c.classList.toggle('locked',lock);
  const ln=$('#gm-lock',host); ln.hidden=!lock;
  if(lock) ln.textContent=NB(tx(`In the challenge you can change only the attention design. Everything else is fixed: Llama-3.1-70B’s shape on 8 × H100 80GB, weights and KV cache in BF16, ${ctxK(CH_BASE.ctx)} of context per user.`,
                               `挑战中只能改注意力设计。其余设置固定：8 × H100 80GB 跑 Llama-3.1-70B 结构的模型，权重和 KV cache 都是 BF16，每个用户 ${ctxK(CH_BASE.ctx)} 上下文。`));
  notes(r1); segSync(); optTips();
}

/* ================= update: everything that depends on the state ================= */
let annT=0;
function update(){
  if(!host) return;
  const r1=compute(S);
  if(mode==='guided'){
    if(step===1&&atInit()&&S.attn!==INIT.attn&&S.attn!=='mla') observed[1]=true;
    if(step===4&&atInit()&&S.attn==='mqa'){observed[4]=true; seen=true;}
    if(step===5&&atInit()&&S.attn==='mla') observed[5]=true;
  }
  drawStage(r1); syncControls(r1); updateRail(r1);
  A.hash.write(token());
  if(userAct&&!r1.none){clearTimeout(annT); const hid=gpuHidden();
    A.announce(hid?tx(`${dName(r1.a,r1.m)}: ${kib(r1.perTok)} of KV cache per token.`,`${dName(r1.a,r1.m)}：每个 token ${kib(r1.perTok)} 的 KV cache。`)
               :tx(`${dName(r1.a,r1.m)}: ${kib(r1.perTok)} per token, ${kib(r1.perGpuTok)} on each GPU; ${r1.users>0?`at most ${users(r1.users)} ${fitV(r1.users)}`:'not even one user fits'} at ${ctxK(r1.s.ctx)} each.`,
                   `${dName(r1.a,r1.m)}：每个 token ${kib(r1.perTok)}，每张 GPU ${kib(r1.perGpuTok)}；每人 ${ctxK(r1.s.ctx)} 上下文时${r1.users>0?`最多装下 ${r1.users} 个用户`:'一个用户也装不下'}。`));}
  A.dock.refresh();
}
function setS(p){
  S={...S,...p};
  if(MODELS[S.model].native==='mla'&&S.attn&&S.attn!=='mla') S.attn='mla';
  if(mode==='challenge') Object.assign(S,CH_BASE);
  update();
}
const learner=f=>{userAct=true; try{f();} finally{userAct=false;}};
/* R3.2 (review 8): a change under Settings never moves the page. Nothing scrolls on purpose; on a narrow screen the card and the
   stage sit above Settings and change height with the design, and the browser's scroll anchoring, anchored in the stage, let the
   control drift by a line or more. So anchoring is paused for the update and the control just used is put back where it was. */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const setAct=(el,f)=>el&&el.closest('#gm-controls')?holdPlace(el,()=>learner(f)):learner(f);
/* a design picked inside a guided card: a controlled experiment from the initial setup (R3.6) */
function pickGuided(a){
  (triedAt[step]=triedAt[step]||new Set()).add(a); if(step===4) tried5.add(a);
  S={...INIT,attn:a}; update(); showResult();
}
/* bring the result into view when it starts below the fold (R3.10); a shorter card can also let the sticky stage slide
   under the top bar again (review A r12 N5: MHA or GQA-16 in part 5), so put the stage back without losing the result */
function showResult(){const box=$('#gm-rail [data-obs]',host); if(box&&!box.hidden) keepInView(box); keepStage(true);}
/* R3.10: a result box already on screen stays put; one cut off at the bottom scrolls just enough to show its end,
   never pushing its top under the top bar. Room is kept for the mini result bar at every width, since it appears
   as soon as the stage scrolls away (at 1440 x 1000 in Scenarios it covered the first line of the note). */
function keepInView(box){const q=box.getBoundingClientRect(), vh=window.innerHeight, bar=$('.topbar'), dk=$('#dock'),
    hdr=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:50)+12, dy=q.top<hdr?q.top-hdr:(q.bottom>vh-8?Math.min(q.bottom-vh+12,q.top-hdr):0);
  if(dy) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* After Next or Back, or a pick that shortens the card, the rail can end above the sticky stage's bottom; the stage then slides up
   under the top bar with the bench's bottom edge (review A r11 #15, r12 N5). Once every scroll has stopped, scroll up just enough for
   the stage to stick below the top bar again, keeping the card's top on screen and, after a pick, the result box's end (R3.10).
   Review A r13 M7: a fixed 450 ms delay could fire during the smooth scroll after Next and cut it short, so this waits for scrollY to
   stop changing (up to 1.5 s), as kv-basics does. Wide screens only, where the stage is sticky. */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(withResult){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!host) return;
    if(window.scrollY!==lastY){if(waited>=1500) return; lastY=window.scrollY; waited+=120; stageT=setTimeout(tick,120); return;}   // a reveal still scrolling
    const st=$('#gm-stage',host); if(!st||window.innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#gm-rail .step.current',host)||$('#gm-rail .railcard',host)||$('#gm-rail',host); if(!card) return;
    const vh=window.innerHeight, want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return; dy=Math.max(dy,c.top-(vh-160));
    const box=withResult&&$('#gm-rail [data-obs]',host); if(box&&!box.hidden){const q=box.getBoundingClientRect(); dy=q.bottom<=vh?Math.max(dy,q.bottom-(vh-8)):0;}
    if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
/* the learner's own scrolling wins: a wheel, a swipe or a scrolling key cancels a pending correction */
window.addEventListener('wheel',stageStop,{passive:true}); window.addEventListener('touchmove',stageStop,{passive:true});
window.addEventListener('keydown',e=>{if(/^(PageUp|PageDown|ArrowUp|ArrowDown|Home|End| )$/.test(e.key)&&!(e.target.closest&&e.target.closest('input,select,textarea,button,[role="button"]'))) stageStop();});
function enterStep(i){
  step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) S={...s.enter()};
  renderRail(); update(); clearBadges(); cue();
}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&savedS){S={...savedS}; savedS=null;}
  if(m==='challenge'){savedS={...S}; chSubmitted=false; chReveal=false; chPicked=false; chTried.clear(); S={...CH_BASE,attn:null};}
  A.shell.setMode(host,m,DESCS());
  if(m==='guided'){enterStep(step); return;}
  renderRail(); update(); clearBadges(); cue();
}
function chPick(a,fromCard){chPicked=true; chTried.add(a); S={...CH_BASE,attn:a}; update();
  // after a submit, a pick in the task card brings its new feedback into view (review A r12 N6, R3.10); a pick under Settings never
  // scrolls the page (review 8, R3.2: it used to jump to the challenge's answer)
  if(chSubmitted&&fromCard){const res=$('#gm-chres',host); if(res) keepInView(res);}}
function submitChallenge(){if(!chPicked) return; chSubmitted=true; if(compute(S).users>=GOAL) chReveal=true; renderRail(); update();
  const res=$('#gm-chres',host); if(res) res.scrollIntoView({block:'nearest',behavior:A.reduceMotion()?'auto':'smooth'});}

/* ================= jumps and formula-term tooltips ================= */
const CTL={model:'gm-ctl-model',attn:'gm-ctl-attn',gpu:'gm-ctl-gpu',n:'gm-ctl-n',wp:'gm-ctl-wp',ctx:'gm-ctl-ctx'};
const CTLNAME={model:{en:'Model',zh:'模型'},attn:{en:'Attention design',zh:'注意力设计'},gpu:{en:'GPU',zh:'GPU'},n:{en:'GPUs, tensor parallel',zh:'GPU 数（张量并行）'},wp:{en:'Weight precision',zh:'权重精度'},ctx:{en:'Context per user',zh:'每个用户的上下文'}};
const FIXED={util:'gm-fx-util',oh:'gm-fx-oh',kvb:'gm-fx-kvb'};
const locked=k=>mode==='challenge'&&k!=='attn'&&!['math','controls','fid','fixed','stage','chart'].includes(k);
function fixedVal(k){const r1=compute(S);
  return {model:tx(`the model is Llama-3.1-70B’s shape`,`模型是 Llama-3.1-70B 的结构`),gpu:tx('the GPU is the H100 80GB','GPU 是 H100 80GB'),n:tx('there are 8 GPUs','一共 8 张 GPU'),wp:tx('the weights are in BF16','权重是 BF16'),
    ctx:tx(`each user has ${ctxK(CH_BASE.ctx)} of context`,`每人 ${ctxK(CH_BASE.ctx)} 上下文`),util:tx(`memory utilization is ${UTIL}`,`显存利用率是 ${UTIL}`),oh:tx(`runtime overhead is ${OH} GiB per GPU`,`每张 GPU 的运行时开销是 ${OH} GiB`),kvb:tx('the KV cache is in BF16','KV cache 是 BF16')}[k]||'';}
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='fid') return tx('Click to jump to the notes at the end of the page.','点击跳到页末的“简化、假设和来源”。');
  if(k==='fixed') return mode==='challenge'?tx('Fixed during the challenge. The reasons are under Settings, in “Fixed in this lab”.','挑战中固定不变。原因在“自由调参”的“这个 Lab 里固定的”里。'):tx('Click to open “Fixed in this lab” under Settings.','点击打开“自由调参”里的“这个 Lab 里固定的”。');
  if(FIXED[k]) return mode==='challenge'?tx(`Fixed during the challenge: ${fixedVal(k)}.`,`挑战中固定不变：${fixedVal(k)}。`):tx('Fixed in this lab. Click to see why under Settings.','这个 Lab 里固定不变。点击到“自由调参”里看原因。');
  if(locked(k)) return tx(`Fixed during the challenge: ${fixedVal(k)}.`,`挑战中固定不变：${fixedVal(k)}。`);
  if(k==='attn'&&mode==='challenge') return tx('Click to jump to the designs in the task card.','点击跳到任务卡里的设计选项。');
  if(k==='attn'&&mode==='guided'&&inlineAttn()) return tx('Click to jump to the choice in this part’s card.','点击跳到这一部分卡片里的选项。');
  if(CTLNAME[k]) return tx(`Click to jump to “${CTLNAME[k].en}”.`,`点击跳到“${CTLNAME[k].zh}”。`);
  return '';
}
const inlineAttn=()=>host&&($('#gm-ic-kv',host)||$('#gm-ic-kv5',host)||$('#gm-ic-mla',host));
/* the Llama counts are exact Hugging Face totals; DeepSeek-V3's 671B is the paper's round figure */
const pCount=m=>m.native==='mla'?big(m.params):int(m.params);
function termText(k,term){const r1=compute(S), m=r1.m;
  if(term==='layers') return tx(`${m.name} has ${m.layers} layers, and each keeps its own keys and values; the count comes from its config.json.`,`${m.name} 有 ${m.layers} 层，每层存自己的 key 和 value；层数来自它的 config.json。`);
  if(term==='headdim') return tx(`Each key or value vector holds ${m.headDim} numbers (head_dim), from the model’s config.json.`,`每个 key 或 value 向量是 ${m.headDim} 维（head_dim），来自模型的 config.json。`);
  if(term==='latent') return tx(`In each layer, MLA (multi-head latent attention) stores one compressed vector (latent) of ${r1.dc} numbers per token instead of a key and a value per head.`,`MLA（multi-head latent attention）在每一层为每个 token 存一个 ${r1.dc} 维的压缩向量（latent），代替各个 head 的 key 和 value。`);
  if(term==='rope') return tx(`It also stores a ${r1.dr}-number key that carries the token’s position (RoPE, rotary position embedding).`,`另外还存一个 ${r1.dr} 维的 key，它带着 token 的位置信息（RoPE，旋转位置编码）。`);
  const o={attn:r1.none?tx('The attention design sets the number of KV heads per layer.','KV head 的数量由注意力设计决定。')
      :(r1.a==='mha'?tx(`The attention design sets the KV heads per layer: ${r1.G} here with MHA (multi-head attention), one for each query head.`,`KV head 的数量由注意力设计决定：这里用 MHA（多头注意力），每层 ${r1.G} 个，每个 query head 各有一个。`)
      :(r1.G===1?tx(`The attention design sets the KV heads per layer: 1 here with MQA (multi-query attention), shared by all ${m.heads} query heads.`,`KV head 的数量由注意力设计决定：这里用 MQA（multi-query attention），每层 1 个，${m.heads} 个 query head 共用。`)
      :tx(`The attention design sets the KV heads per layer: ${r1.G} here with ${DNAME[r1.a]} (grouped-query attention), each shared by ${r1.group} query heads.`,`KV head 的数量由注意力设计决定：这里用 ${DNAME[r1.a]}（grouped-query attention），每层 ${r1.G} 个，每个由 ${r1.group} 个 query head 共用。`))),
    kvb:tx(`Each number in the KV cache takes ${KVB} bytes in BF16.`,`KV cache 里每个数值在 BF16 下占 ${KVB} 字节。`),
    n:tx(`Tensor parallelism splits every layer across these ${gpus(r1.s.n)}, attention by head.`,`张量并行把每一层拆到这 ${gpus(r1.s.n)} 上，attention 按 head 拆。`),
    ctx:tx('Each user keeps this many tokens of context, and every one of them is stored.','每个用户保留这么多个 token 的上下文，每个都要存。'),
    gpu:tx(`The ${r1.g.name} has ${r1.g.mem} GB of memory, read here as ${r1.g.mem} GiB.`,`${r1.g.name} 有 ${r1.g.mem} GB 显存，这里按 ${r1.g.mem} GiB 计。`),
    util:tx(`The server uses at most ${fmtD(UTIL*100,0)}% of each GPU’s memory and keeps the rest as a safety margin.`,`服务端最多只用每张卡显存的 ${fmtD(UTIL*100,0)}%，剩下的留作安全余量。`),
    wp:tx(`The weights take the parameter count times the bytes per parameter: ${pCount(m)} × ${PREC[r1.s.wp]} ${PREC[r1.s.wp]===1?'byte':'bytes'} = ${fmtF(r(r1.wTotal/1e9,1),1)} GB = ${wTotalShown(r1)} GiB, split over ${gpus(r1.s.n)}. A GiB is 2^30 bytes, about 7% more than a GB.`,`权重占的显存是参数量乘以每个参数的字节数：${pCount(m)} × ${PREC[r1.s.wp]} 字节 = ${fmtF(r(r1.wTotal/1e9,1),1)} GB = ${wTotalShown(r1)} GiB，分到 ${gpus(r1.s.n)} 上。1 GiB = 2^30 字节，比 1 GB 大约 7%。`),
    oh:tx(`Runtime overhead is memory the engine needs besides the weights and the KV cache, such as temporary activations and CUDA graphs: an estimate of ${OH} GiB per GPU.`,`运行时开销是引擎在权重和 KV cache 之外要占的显存，比如临时激活和 CUDA graph，这里估计每张 GPU ${OH} GiB。`),
    model:tx(`This number comes from ${m.name}’s config.json.`,`这个值来自 ${m.name} 的 config.json。`)};
  return o[k]||'';
}
function termTip(k,el){const term=el&&el.dataset?el.dataset.term:null; return `<div>${esc(NB(termText(k,term)))}</div><div class="act">${esc(jumpAct(k))}</div>`;}
function jump(k,src){
  if(!host) return;
  if(k==='math'){const d=$('#gm-math',host); d.open=true; A.ui.jumpToEl(d); return;}
  if(k==='controls'){A.ui.jumpToEl($('#gm-controls',host)); return;}
  if(k==='fid'){const pop=document.getElementById(ID+'-fid'), b=$(`[data-pop="${ID}-fid"]`,host); if(pop&&pop.hidden&&b) b.click(); if(pop) A.ui.jumpToEl(pop); return;}
  if(k==='stage'||k==='chart'){const el=$(k==='stage'?'#gm-graph':'#gm-chart',host); A.tip.hide(); el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el); return;}
  if(mode==='challenge'&&(k==='fixed'||FIXED[k]||locked(k))){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  if(k==='fixed'||FIXED[k]){const d=$('#gm-fixed',host); d.open=true; const el=FIXED[k]?$('#'+FIXED[k],host):d; A.ui.jumpToEl(el); return;}
  if(k==='attn'){const inl=mode==='challenge'?$('#gm-ch-lv',host):(mode==='guided'?inlineAttn():null); A.ui.jumpToEl(inl||$('#'+CTL.attn,host)); return;}
  const el=CTL[k]&&$('#'+CTL[k],host); if(el) A.ui.jumpToEl(el);
}

/* ================= hash token ================= */
const token=(s=S)=>[s.model,s.attn||'none',s.gpu,s.n,s.wp,s.ctx].join('.');
function parse(tok){
  const p=String(tok||'').split('.'); if(p.length!==6) return null;
  const o={model:p[0],attn:p[1],gpu:p[2],n:+p[3],wp:p[4],ctx:+p[5]};
  if(!MODELS[o.model]||!DESIGNS.includes(o.attn)||!GPUS[o.gpu]||!NS.includes(o.n)||!PREC[o.wp]||!CTXS.includes(o.ctx)) return null;
  if(!designsOf(MODELS[o.model]).includes(o.attn)) return null;
  return o;
}

/* ================= interview drill (seeded by question number) ================= */
function rng(seed){let a=seed>>>0; return ()=>{a=(a+0x6D2B79F5)>>>0; let t=a; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296;};}
function drill(k){const R=rng(4040+k*7919), pick=a=>a[Math.floor(R()*a.length)];
  if(k%3===2){const Ly=pick([27,40,48,61,64,80]), n=pick([4,8]); return {mla:true,L:Ly,n,dc:MLA_C,dr:MLA_R,per:Ly*(MLA_C+MLA_R)*KVB};}
  for(;;){const Ly=pick([32,40,48,60,64,80,96,126]), H=pick([32,48,64,96,128]), G=pick([1,2,4,8,16]), hd=pick([64,128]), n=pick([2,4,8]);
    if(H%G||H%n||G>=H) continue; const per=2*Ly*G*hd*KVB, sh=Math.min(n,G); return {mla:false,L:Ly,H,G,hd,n,per,sh,pg:per/sh};}}
function drillQ(){const d=drill(drillK);
  if(d.mla) return tx(`A model with multi-head latent attention (MLA) stores a compressed vector of ${d.dc} numbers and a ${d.dr}-number RoPE key per token in each of its <b>${d.L}</b> layers, in BF16. How much KV cache does one token take, and how much of it does each GPU store with tensor parallelism across <b>${d.n}</b> GPUs?`,
                      `一个用 multi-head latent attention（MLA）的模型，<b>${d.L}</b> 层里每层为每个 token 存一个 ${d.dc} 维的压缩向量和一个 ${d.dr} 维的 RoPE key，BF16。一个 token 的 KV cache 有多大？在 <b>${d.n}</b> 张 GPU 上做张量并行时，每张 GPU 存多少？`);
  return tx(`A model has <b>${d.L}</b> layers with <b>${d.H}</b> query heads and <b>${d.G}</b> KV ${d.G===1?'head':'heads'} of ${d.hd} numbers, in BF16. How much KV cache does one token take, and how much of it does each GPU store with tensor parallelism across <b>${d.n}</b> GPUs?`,
            `一个模型有 <b>${d.L}</b> 层，每层 <b>${d.H}</b> 个 query head、<b>${d.G}</b> 个 KV head，每个 head ${d.hd} 维，BF16。一个 token 的 KV cache 有多大？在 <b>${d.n}</b> 张 GPU 上做张量并行时，每张 GPU 存多少？`);}
const bytesIf=(b,zh)=>b<1e6?(zh?`${int(b)} 字节 = `:`${int(b)} bytes = `):'';
function drillA(){const d=drill(drillK);
  if(d.mla) return tx(`<p><b>${kib(d.per)} per token; each GPU stores all ${kib(d.per)}</b></p><p>(${d.dc} + ${d.dr}) × ${d.L} layers × 2 bytes = ${bytesIf(d.per)}${kib(d.per)}.</p><p>Every head needs the whole compressed vector, so tensor parallelism cannot split it: each of the ${d.n} GPUs keeps a full copy.</p>`,
                      `<p><b>每个 token ${kib(d.per)}；每张 GPU 都存全部 ${kib(d.per)}</b></p><p>(${d.dc} + ${d.dr}) × ${d.L} 层 × 2 字节 = ${bytesIf(d.per,true)}${kib(d.per)}。</p><p>每个 head 都要用整个压缩向量，张量并行没法把它拆开，${d.n} 张 GPU 每张都存一整份。</p>`);
  const split=d.G>=d.n;
  return tx(`<p><b>${kib(d.per)} per token; ${kib(d.pg)} on each GPU</b></p><p>2 (a key and a value) × ${d.L} layers × ${d.G} KV ${d.G===1?'head':'heads'} × ${d.hd} × 2 bytes = ${bytesIf(d.per)}${kib(d.per)}.</p>`+
            (split?`<p>${d.G} KV heads over ${d.n} GPUs, ${d.G/d.n} on each: ${kib(d.per)} ÷ ${d.n} = ${kib(d.pg)}.</p>`:`<p>${d.G} KV ${d.G===1?'head':'heads'} for ${d.n} GPUs: each GPU keeps a copy of one, so ${kib(d.per)} ÷ ${d.G} = ${kib(d.pg)}; each KV head sits on ${d.n/d.G} GPUs.</p>`),
            `<p><b>每个 token ${kib(d.per)}；每张 GPU ${kib(d.pg)}</b></p><p>2（key 和 value）× ${d.L} 层 × ${d.G} 个 KV head × ${d.hd} × 2 字节 = ${bytesIf(d.per,true)}${kib(d.per)}。</p>`+
            (split?`<p>${d.G} 个 KV head 分到 ${d.n} 张 GPU 上，每张 ${d.G/d.n} 个：${kib(d.per)} ÷ ${d.n} = ${kib(d.pg)}。</p>`:`<p>${d.G} 个 KV head 对 ${d.n} 张 GPU，每张存其中一个的副本，所以 ${kib(d.per)} ÷ ${d.G} = ${kib(d.pg)}；每个 KV head 在 ${d.n/d.G} 张 GPU 上各有一份。</p>`));}
function renderDrill(){const q=$('#gm-drillq',host), a=$('#gm-drilla',host); if(!q) return; q.innerHTML=drillQ(); a.hidden=!drillOpen; if(drillOpen) a.innerHTML=drillA();}

/* ================= mount ================= */
function mount(el,tok){
  root=el; const p=tok?parse(tok):null;
  if(p){S={...p}; mode='sandbox'; savedS=null;}
  host=document.createElement('div'); host.className='gm-host m2'; el.replaceChildren(host);
  host.innerHTML=view(); A.ui.initRanges(host);
  A.shell.bindHeader(host,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#gm-rail',host));},token});
  host.addEventListener('click',ev=>{
    const b=ev.target.closest('button'); if(!b||!host.contains(b)) return; const d=b.dataset;
    const seg=b.closest('.seg-ctl[data-key]');
    if(seg&&d.v!=null){if(b.disabled) return; const k=seg.dataset.key;
      if(k==='attn'){if(mode==='challenge'){setAct(b,()=>chPick(d.v,!!seg.closest('#gm-rail'))); return;}
        if(mode==='guided'&&seg.closest('#gm-rail')){learner(()=>pickGuided(d.v)); return;}
        setAct(b,()=>setS({attn:d.v})); return;}
      if(k==='n'){setAct(b,()=>setS({n:+d.v})); return;}
      if(k==='wp'){setAct(b,()=>setS({wp:d.v})); return;}
      return;}
    if(d.go!=null){enterStep(+d.go); A.ui.reveal($('#gm-rail .step.current',host)); keepStage(); return;}
    if(d.nav==='next'||d.nav==='skip'){if(d.nav==='skip') skipped=true; enterStep(step+1); A.ui.reveal($('#gm-rail .step.current',host)); keepStage(); return;}
    if(d.nav==='back'){enterStep(step-1); A.ui.reveal($('#gm-rail .step.current',host)); keepStage(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){(triedAt[step]=triedAt[step]||new Set()).add(s.showMe.attn); if(step===4) tried5.add(s.showMe.attn); S={...s.showMe}; update(); showResult();} return;}
    if(d.pred){if(seen) return; pred=d.pred; renderRail(); update(); const f=$(`#gm-rail [data-pred="${pred}"]`,host); if(f) f.focus(); return;}
    if(d.lever!=null){learner(()=>chPick(d.lever,true)); return;}
    if(d.preset!=null){learner(()=>setS({...PRESETS[+d.preset].p})); const n=$('#gm-sbnote',host); if(n) keepInView(n); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#gm-rail',host)); return;}
    if(d.act==='submit'){submitChallenge(); return;}
    if(d.act==='reveal'){chReveal=true; updateRail(compute(S)); return;}
    if(d.act==='reset'){setAct(b,()=>{if(mode==='challenge'){chPicked=false; S={...CH_BASE,attn:null}; update();} else setS({...INIT});}); return;}
    if(d.act==='drill-new'){drillK++; drillOpen=false; renderDrill(); return;}
    if(d.act==='drill-show'){drillOpen=true; renderDrill(); return;}
    if(d.act==='copy'){const pre=$('#gm-code',host), txt=pre.textContent, done=w=>{b.textContent=w; setTimeout(()=>{b.textContent=A.t('copyCode');},1600);},
      sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); done(tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'));};
      try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(()=>done(tx('Copied','已复制')),sel); else sel();}catch(e){sel();} return;}
  });
  host.addEventListener('change',ev=>{const t=ev.target;
    if(t.id==='gm-model'){const m=MODELS[t.value]; setAct(t,()=>setS({model:t.value,attn:m.native,wp:m.native==='mla'&&S.wp==='bf16'?'fp8':S.wp}));}
    if(t.id==='gm-gpu') setAct(t,()=>setS({gpu:t.value}));});
  host.addEventListener('input',ev=>{const t=ev.target; if(t.dataset&&t.dataset.key==='ctx'){setAct(t,()=>setS({ctx:CTXS[+t.value]}));}});
  // Enter / Space on focusable non-button items that jump
  host.addEventListener('keydown',ev=>{if(ev.key!=='Enter'&&ev.key!==' ') return; const el=ev.target; if(!el.dataset||!el.dataset.jump||el.tagName==='BUTTON'||el.classList.contains('t')) return; ev.preventDefault(); jump(el.dataset.jump,el);});
  if(window.ResizeObserver){roW=new ResizeObserver(()=>{const d=$('#gm-diag',host); if(d&&Math.abs((d.clientWidth||0)-(d._w||0))>2) drawDiagram(compute(S));}); roW.observe($('#gm-graph',host));}
  const c=$('#gm-controls',host); c.classList.toggle('locked',mode==='challenge');
  if(!mounted&&!p&&mode==='guided'){mounted=true; enterStep(step);} else {mounted=true; renderRail(); update(); cue();}
  renderDrill();
}
A.addStrings({en:{},zh:{}});
const sOf=s=>({model:s.model,attn:s.attn,gpu:s.gpu,n:s.n,wp:s.wp,ctx:s.ctx});
A.registerLab({
  id:ID,
  // round 3 #6: what the learner did here, saved by the kit and put back once per page load before the first mount (CONTRACT.md)
  snap:()=>{const ta={}; for(const k in triedAt) ta[k]=[...triedAt[k]];
    return {v:1,mode,step,S:sOf(S),pred,seen,skipped,observed:{...observed},triedAt:ta,tried5:[...tried5],savedS:savedS?sOf(savedS):null,chPicked,chTried:[...chTried],
      chSubmitted,chReveal,drillK,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPicked&&compute(S).users>=GOAL};},
  restore(o){if(!o||o.v!==1||!o.S||typeof o.S!=='object') return;
    const md=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided', chOk=a=>DESIGNS.includes(a)&&designsOf(MODELS[CH_BASE.model]).includes(a);
    // the challenge starts with no design picked (attn null), which the shared settings cannot name: its pick is checked on its own
    const p=md==='challenge'?null:parse(token(o.S)); if(md!=='challenge'&&!p) return;
    mode=md; S=md==='challenge'?sOf({...CH_BASE,attn:chOk(o.S.attn)?o.S.attn:null}):sOf(p);
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; pred=PRED.some(x=>x.v===o.pred)?o.pred:null;
    seen=o.seen===true; skipped=o.skipped===true; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    for(const k in triedAt) delete triedAt[k];
    if(o.triedAt&&typeof o.triedAt==='object') for(const k in o.triedAt) if(/^\d+$/.test(k)&&+k<STEPS.length&&Array.isArray(o.triedAt[k])) triedAt[k]=new Set(o.triedAt[k].filter(a=>DESIGNS.includes(a)));
    tried5.clear(); if(Array.isArray(o.tried5)) o.tried5.forEach(a=>{if(DESIGNS.includes(a)) tried5.add(a);});
    const sv=o.savedS&&typeof o.savedS==='object'?parse(token(o.savedS)):null; savedS=mode==='challenge'&&sv?sOf(sv):null;
    chPicked=mode==='challenge'&&o.chPicked===true&&S.attn!=null; chTried.clear(); if(mode==='challenge'&&Array.isArray(o.chTried)) o.chTried.forEach(a=>{if(chOk(a)) chTried.add(a);});
    chSubmitted=chPicked&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;
    if(Number.isInteger(o.drillK)&&o.drillK>=0&&o.drillK<1e6) drillK=o.drillK; drillOpen=o.drillOpen===true;
    mounted=true;},   // the first mount then draws this part as a return in the same session does (enterStep() would reset the settings)
  mount,
  leave(){if(roW){roW.disconnect(); roW=null;} clearTimeout(stageT); A.tip.hide(); host=null; root=null;},
  onLang(){if(root) mount(root,null);},
  onResize(){if(host){drawDiagram(compute(S)); fitMetrics();}},
  termTip, jump, jumpAct,
  dock:{
    anchor:()=>host&&$('#gm-diag',host), bench:()=>host&&$('.bench',host), target:()=>host&&$('#gm-graph',host),
    render({status:st,vis,read}){if(!host) return; const r1=compute(S), s=status(r1), hid=gpuHidden();
      st.className='status '+s.cls; st.textContent=s.short||s.text;
      if(r1.none){vis.innerHTML=''; read.textContent=NB(tx(`Llama-3.1-70B’s shape on 8 × H100: pick a design (goal: ${GOAL} users)`,`8 × H100 跑 Llama-3.1-70B 结构：先选一种设计（目标 ${GOAL} 个用户）`)); return;}
      const n=r1.s.n, cells=hid?1:n, max=hid?compute({...S,attn:'mha'}).perTok:compute({...S,attn:designsOf(r1.m)[0]}).perGpuTok, v=hid?r1.perTok:r1.perGpuTok;
      const cl=hid?tx('per token','每个 token'):(n===1?tx('1 GPU','1 张 GPU'):tx(`${n} GPUs`,`${n} 张 GPU`));
      vis.innerHTML=`<div class="gm-dkw"><span class="gm-dk-l">${cl}</span><div class="gm-dk">${Array.from({length:cells},()=>`<span><i class="${!hid&&r1.copies>1?'cp':''}" style="width:${Math.min(100,v/max*100).toFixed(1)}%"></i></span>`).join('')}</div></div>`;
      const hw=`${n} × ${r1.g.name.split(' ')[0]}`;
      const who=NB(tx(`${r1.m.name} on ${hw}, ${dName(r1.a,r1.m)}`,`${hw} 跑 ${r1.m.name}，${dName(r1.a,r1.m)}`));
      let lead;
      if(hid) lead=tx(`${who}: ${kib(r1.perTok)} of KV cache per token`,`${who}：每个 token ${kib(r1.perTok)} 的 KV cache`);
      else if(mode==='challenge') lead=tx(`${who}: at most ${users(r1.users)} ${fitV(r1.users)} at ${ctxK(r1.s.ctx)} each (goal: ${GOAL})`,`${who}：每人 ${ctxK(r1.s.ctx)} 上下文时最多装下 ${r1.users} 个用户（目标 ${GOAL} 个）`);
      else lead=r1.users>0?tx(`${who}: at most ${users(r1.users)} ${fitV(r1.users)} at ${ctxK(r1.s.ctx)} each`,`${who}：每人 ${ctxK(r1.s.ctx)} 上下文时最多装下 ${r1.users} 个用户`):tx(`${who}: not even one user fits at ${ctxK(r1.s.ctx)} each`,`${who}：每人 ${ctxK(r1.s.ctx)} 上下文时一个用户也装不下`);
      const mem=hid?'':tx(`; each GPU stores ${kib(r1.perGpuTok)} per token`,`；每张 GPU 每个 token 存 ${kib(r1.perGpuTok)}`);
      read.innerHTML=`${esc(lead)}<span class="dk-mem">${esc(mem)}</span>`;}
  }
});

/* ================= home tile: one token's KV cache for four designs, and each GPU's part ================= */
const TZ_D=['mha','gqa8','mqa','mla'];
function tzData(a){const x=compute({...INIT,attn:a}), mh=compute({...INIT,attn:'mha'}); return {a,perTok:x.perTok,perGpu:x.perGpuTok,copies:x.copies,n:x.s.n,maxTok:mh.perTok,maxGpu:mh.perGpuTok};}
A.registerTeaser(ID,{fid:'formula',render(el){
  const m=m70();
  el.innerHTML=`<div class="tz">${A.ui.seg({id:'tz-gm-d',key:'tzgm',label:tx('Attention design (Llama-3.1-70B’s shape)','注意力设计（Llama-3.1-70B 的结构）'),options:TZ_D.map(a=>[a,DNAME[a]]),value:tzD})}`+
    `<div class="tz-vis tz-gm-vis" role="img"><div class="tz-gm-row"><span class="tz-gm-l">${tx('per token','每个 token')}</span><span class="tz-gm-t"><i class="tz-gm-f"></i></span></div>`+
    `<div class="tz-gm-row"><span class="tz-gm-l">${tx(`each of ${INIT.n} GPUs`,`${INIT.n} 张 GPU 每张`)}</span><span class="tz-gm-cells">${Array.from({length:INIT.n},()=>`<span class="tz-gm-t"><i class="tz-gm-f"></i></span>`).join('')}</span></div></div>`+
    `<div class="tz-leg"><span><i class="sw kv"></i>KV cache</span><span class="tz-gm-lgc"><i class="sw tz-gm-swc"></i>${tx('hatched: the same keys and values, copied','斜纹：同样的 key 和 value，复制了几份')}</span></div>`+
    `<p class="tz-read"></p>`+
    `<p class="tz-note">${esc(tx(`MHA (multi-head attention) gives each of the ${m.heads} query heads its own KV head, GQA-8 (grouped-query attention) lets every ${m.heads/8} query heads share one, and MQA (multi-query attention) lets all ${m.heads} share one; MLA (multi-head latent attention) stores one compressed vector per token and layer instead.`,
      `MHA（多头注意力）让 ${m.heads} 个 query head 各有一个 KV head，GQA-8（grouped-query attention）让每 ${m.heads/8} 个共用一个，MQA（multi-query attention）让 ${m.heads} 个共用一个；MLA（multi-head latent attention）改为每个 token 每层存一个压缩向量。`))}</p>`+
    `<p class="tz-note">${esc(tx(`KV cache per token = 2 (a key and a value) × ${m.layers} layers × KV heads × ${m.headDim} numbers × ${KVB} bytes (BF16); MLA stores ${MLA_C} + ${MLA_R} numbers per layer instead, DeepSeek-V3’s sizes, so that option is a what-if for this model.`,
      `每个 token 的 KV cache = 2（key 和 value）× ${m.layers} 层 × KV head 数 × ${m.headDim} 维 × ${KVB} 字节（BF16）；MLA 改为每层存 ${MLA_C} + ${MLA_R} 维，这是 DeepSeek-V3 的大小，所以对这个模型来说是假设。`))}</p>`+
    `<p class="tz-note">${esc(tx(`On ${INIT.n} GPUs, tensor parallelism splits the KV heads among them but never below one per GPU, so MQA’s single KV head and MLA’s compressed vector are copied to every GPU.`,
      `在 ${INIT.n} 张 GPU 上，张量并行把 KV head 分到各张 GPU，但每张最少一个，所以 MQA 唯一的 KV head 和 MLA 的压缩向量每张 GPU 都要存一份。`))}</p>`+
    `<p class="tz-note">${esc(tx('Shapes from Llama-3.1-70B’s config.json. The copies follow vLLM: QKVParallelLinear copies KV heads, and for MLA every GPU keeps one whole compressed vector (ModelConfig.get_num_kv_heads returns 1).','结构来自 Llama-3.1-70B 的 config.json。复制的规则和 vLLM 一致：KV head 由 QKVParallelLinear 复制；MLA 每张 GPU 存一整份压缩向量（ModelConfig.get_num_kv_heads 返回 1）。'))}</p></div>`;
  const seg=el.querySelector('.seg-ctl'), rd=el.querySelector('.tz-read'), vis=el.querySelector('.tz-gm-vis'), fills=$$('.tz-gm-f',el), lgc=el.querySelector('.tz-gm-lgc');
  $$('button[data-v]',seg).forEach(b=>b.setAttribute('data-tipx',designTip(b.dataset.v,m)+(b.dataset.v===m.native?tx(` ${m.name} uses it.`,`${m.name} 用的就是它。`):'')));
  const upd=()=>{A.ui.setPressed(seg,tzD); const d=tzData(tzD), cp=d.copies>1;
    fills[0].style.width=(d.perTok/d.maxTok*100).toFixed(2)+'%';
    fills.slice(1).forEach(f=>{f.style.width=(d.perGpu/d.maxGpu*100).toFixed(2)+'%'; f.classList.toggle('cp',cp);});
    lgc.hidden=!cp;
    const sz=b=>kib(b)+mibIf(b);
    vis.setAttribute('aria-label',tx(`${DNAME[tzD]}: ${sz(d.perTok)} per token; each of the ${d.n} GPUs stores ${kib(d.perGpu)}.`,`${DNAME[tzD]}：每个 token ${sz(d.perTok)}；${d.n} 张 GPU 每张存 ${kib(d.perGpu)}。`));
    /* R6.20: per token, then what each GPU keeps (the bold number), with the principle (split or copied) and one comparison in words */
    const g=tzData('gqa8'), vsG=d.perGpu===g.perGpu?tx('as much as with GQA-8','和 GQA-8 一样多'):(d.perGpu>g.perGpu?tx('more than with GQA-8','比 GQA-8 还多'):tx('less than with GQA-8','比 GQA-8 少'));
    rd.innerHTML=NB({mha:tx(`Llama-3.1-70B’s shape with multi-head attention (MHA): ${kib(d.perTok)} of KV cache per token, split evenly across the GPUs, <b>${kib(d.perGpu)}</b> on each.`,`Llama-3.1-70B 的结构用多头注意力（MHA）：每个 token ${kib(d.perTok)} 的 KV cache，平均分到各张 GPU 上，每张 <b>${kib(d.perGpu)}</b>。`),
      gqa8:tx(`GQA-8, the grouped-query attention Llama-3.1-70B uses: ${kib(d.perTok)} per token, split evenly across the GPUs, <b>${kib(d.perGpu)}</b> on each.`,`GQA-8，也就是 Llama-3.1-70B 用的 grouped-query attention：每个 token ${kib(d.perTok)}，平均分到各张 GPU 上，每张 <b>${kib(d.perGpu)}</b>。`),
      mqa:tx(`Llama-3.1-70B’s shape with multi-query attention (MQA): ${kib(d.perTok)} per token. But every GPU keeps a copy of all <b>${kib(d.perGpu)}</b>, ${vsG}.`,`Llama-3.1-70B 的结构用 multi-query attention（MQA）：每个 token ${kib(d.perTok)}。但每张 GPU 都要把这 <b>${kib(d.perGpu)}</b> 整个存一份，${vsG}。`),
      mla:tx(`Llama-3.1-70B’s shape with multi-head latent attention (MLA): ${kib(d.perTok)} per token. But every GPU keeps a copy of all <b>${kib(d.perGpu)}</b>, ${vsG}.`,`Llama-3.1-70B 的结构用 multi-head latent attention（MLA）：每个 token ${kib(d.perTok)}。但每张 GPU 都要把这 <b>${kib(d.perGpu)}</b> 整个存一份，${vsG}。`)}[tzD]);};
  seg.addEventListener('click',ev=>{const b=ev.target.closest('button[data-v]'); if(!b) return; tzD=b.dataset.v; upd();});
  upd();
}});

/* read-only hook for tests */
Object.defineProperty(window,'__gm',{configurable:true,value:Object.freeze({
  MODELS:JSON.parse(JSON.stringify(MODELS)),DESIGNS:DESIGNS.slice(),INIT:{...INIT},CH_BASE:{...CH_BASE},GOAL,UTIL,OH,KVB,CTXS:CTXS.slice(),GPUS:JSON.parse(JSON.stringify(GPUS)),GQA_T1:JSON.parse(JSON.stringify(GQA_T1)),fracK:fracK(),MLA_C,MLA_R,FALCON:{...FALCON},
  compute:s=>{const x=compute({...INIT,...s}); const o={}; ['perTok','perGpuTok','shard','kvPerReq','perUserGpu','users','copies','headsPerGpu','G','elems','usable','wPerGpu','budget','wTotal','weightsFit','mla','hypo','native','group','none'].forEach(k=>{o[k]=x[k];}); return o;},
  presets:()=>PRESETS.map(p=>({...p.p})),
  challenge:()=>chAll().map(x=>({a:x.a,u:x.u,met:x.met,kv:x.kv})), best:()=>chBest().a,
  mhaExtraParams, mathText:s=>{const saveS=S; S={...INIT,...s}; const out=mathLines(compute(S)).map(x=>{const d=document.createElement('div'); d.innerHTML=x; return d.textContent.replace(/‑/g,'-').replace(/ /g,' ');}); S=saveS; return out;},
  drill:k=>drill(k), teaser:a=>tzData(a), outcome:()=>OUTCOME(), exits:()=>EXITS(),
  FP8_REAL:{...META405_FP8,dsExtra:DSV3_EXTRA}, fp8Real:()=>{const f=fp8Real(); return {w4:f.w4,u4:f.u4,uds:f.uds,u4base:f.p4.users,udsBase:f.ds.users};},
  wpTip:s=>{const saveS=S; S={...INIT,...s}; const t=termText('wp'); S=saveS; return t.replace(/‑/g,'-').replace(/ /g,' ');},
  state:()=>({S:{...S},mode,step,pred,seen,skipped,observed:{...observed},chPicked,chSubmitted,chReveal,tried5:[...tried5]})
})});
})();
