(function(){
'use strict';
const $=(s,el)=>(el||document).querySelector(s);
const $$=(s,el)=>Array.from((el||document).querySelectorAll(s));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
/* R6.29 (2026-10-09): text that may hold typeset formulas (hovers in data-tipx, quiz options, seg and lever labels): a formula made by
   A.ui.math (<math class="tex">…</math>) and the simple tags around it (b, i, em, strong, br, sub, sup, small, code, div, span, p, with at
   most a class) stay markup; anything else is escaped and shows as text (a token written <s> stays “<s>”) */
const KEEPTAG=/^<\/?(b|i|em|strong|br|sub|sup|small|code|div|span|p)(\s+class="[\w\s-]*")?\s*\/?>$/;
const escMath=s=>String(s).split(/(<math class="tex"[\s\S]*?<\/math>|<\/?[a-z]+(?:\s+[a-z-]+="[^"]*")*\s*\/?>)/).map((p,i)=>i%2&&(p.startsWith('<math class="tex"')||KEEPTAG.test(p))?p:esc(p)).join('');

/* ---------- language: English is the source text, Chinese follows ---------- */
let lang='en';
(function(){let saved=null; try{saved=localStorage.getItem('asp-lang');}catch(e){}
  if(saved==='en'||saved==='zh') lang=saved; else if(/^zh/i.test(navigator.language||'')) lang='zh';})();
const tx=(en,zh)=>lang==='zh'?zh:en;
const L=o=>o?(o[lang]!=null?o[lang]:o.en):'';
const STR={
 en:{
  brandAria:'AI Systems Playground, home', navAria:'Primary', navMap:'Tree', 
  langAria:'Language', themeAria:'Switch between light and dark',
  heroTitle:'Learn AI systems by operating them.',
  heroCta:'Why does a 70B model run out of memory? <span aria-hidden="true">→</span>',
  heroMapCta:'Explore the tree', heroSub:'Start with the first lesson: follow one sentence and see how a large language model writes its next word. No AI background needed.', hmSub:'Llama-3.1-70B · 100 concurrent users', ctxLabel:'Context per user',
  mapTitle:'Start with the path that suits you best',
  pathsAria:'Learning paths', chipAll:'No path, by branch',
  foot2:'Two more visual explainers of what happens inside a transformer: <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a> (GPT-2) and <a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a>.',
  ftBlurb:'Hands-on labs on how modern AI systems work: how a model writes its next word, then training, serving, post-training, evaluation, RAG and agents. Everything runs in your browser.',
  ftPaths:'Learning paths', ftTree:'Knowledge tree', ftProject:'Project', ftSource:'Source code on GitHub', ftIssue:'Report a problem',
  ftLegal:'The code is under the Apache 2.0 license; the fonts are under the SIL Open Font License.',
  crumbMap:'← Tree', crumbWorld:'How LLMs work', crumbKV:'KV cache memory', labTitle:'Why does a 70B model run out of memory at long context?',
  tagMin:'15 min', tagLevel:'Core', fidBtn:'Fidelity: Formula ▾', priBtn:'Primer: what the KV cache is ▾',
  flowAria:'The steps of this lab', modesAria:'Lab mode', modeGuided:'Guided', modeSandbox:'Scenarios', modeChallenge:'Challenge', share:'Copy setup link',
  descGuided:'Seven short parts. Predict first, find where memory runs out, see why the KV cache grows so large, then try ways to fit more users. Start here if this is new to you.',
  descSandbox:'Six real deployments, from a chat service to one desktop GPU. Pick one to see whether it fits and how many users it can serve.',
  descChallenge:'One 8 × H100 node runs Llama-3.1-70B with 128K of context per user. Change only the two precision settings so that 20 users fit at once.',
  fidPop:"<div class=\"pop-grid\"><div><div class=\"eyebrow\">Modeled</div><ul><li>Weight memory: the parameter count times the bytes per parameter.</li><li>KV cache per token, from each model’s config.json: the number of KV heads for GQA models, the size of MLA’s compressed vector (latent) for DeepSeek-V3.</li><li>Tensor parallelism splits the weights and the KV heads evenly across the GPUs; MLA’s compressed vector is copied in full to every GPU.</li><li>The server uses only a share of each GPU’s memory (vLLM’s gpu_memory_utilization).</li></ul></div><div><div class=\"eyebrow\">Simplified</div><ul><li>Temporary activations, CUDA graphs and the like are one fixed runtime overhead per GPU. It is an estimate; real engines measure it at startup (<button type=\"button\" class=\"linkbtn\" data-jump=\"oh\">change it in Settings</button>).</li><li>Every user fills their whole context, the worst case.</li><li>Plain tensor parallelism only. Serving stacks such as SGLang can run attention data-parallel so that MLA’s compressed vector is not copied to every GPU; this lab does not model that.</li><li>No fragmentation (the PagedAttention lab covers it).</li><li>FP8 weights count 1 byte per parameter, with no scale factors. Real FP8 checkpoints keep some tensors in BF16 and store scales. Meta’s FP8 release of Llama-3.1-405B keeps 81.3B parameters in BF16, every attention matrix among them: 56.7 GiB per GPU on 8 GPUs instead of 47.2 GiB. The 16 users of the scenario “405B on one node” still fit, but at 8K each at most 53 users fit, not 92. DeepSeek-V3’s real checkpoint adds about 0.24 GiB per GPU, which changes none of its user counts in this lab.</li></ul></div><div><div class=\"eyebrow\">Sources</div><ul><li>Model shapes: each model’s config.json on Hugging Face.</li><li>GPU memory: the vendor’s nominal capacity, read as GiB; drivers report slightly less.</li><li>vLLM’s default gpu_memory_utilization (vllm/config/cache.py): 0.9 up to v0.19, 0.92 since v0.20.0; the initial setup here uses 0.9.</li><li>SGLang’s data-parallel attention for MLA: SGLang v0.4 release notes (LMSYS blog, 2024-12-04).</li><li>FP8 checkpoints: the tensor types of meta-llama/Llama-3.1-405B-Instruct-FP8 and deepseek-ai/DeepSeek-V3 on Hugging Face.</li></ul><p class=\"mono small muted\" style=\"margin-top:8px\">Last verified 2026-09-26. Prototype, not yet expert-reviewed.</p></div></div>",
  priPop:"<p>A model generates text one token at a time. For each new token, every layer compares the token’s query vector with the key vectors of all earlier tokens and its own, turns those scores into attention weights that add up to 100%, and mixes those tokens’ value vectors by those attention weights. Each layer does this in many heads side by side; each head has its own query, and in many current models, such as Llama 3 and Mistral, several heads share one set of keys and values. A token looks only at itself and earlier tokens (the causal mask), so earlier tokens’ keys and values never change, and a server keeps them in GPU memory instead of recomputing them. That store is the KV cache.</p><p>The “Transformer structure” learning path explains queries, keys, values and heads from scratch; why generation keeps every token’s keys and values is covered in its lab “<a href=\"#kv-basics\" data-tipx=\"Lab: Why does generation keep every token’s keys and values?\">Why a KV cache</a>”.</p>",
  shareLabel:'This link opens this lab with the same setup. It does not carry your predictions or your progress:', shareLabelFrag:'Add this to the end of the page’s link: that link then opens this lab with the same setup. It does not carry your predictions or your progress:', railAria:'Lab parts',
  kvOutcome:'<b>You will be able to: </b>estimate how many users fit on a GPU node at a given context, and choose a fix when they do not.', nextK:'Next lab', prevK:'Previous lab', nextLast:'This is the last lab you can open so far.', nextMap:'Back to the tree to see the planned labs →', chipTf:'Transformer structure',
  lgW:'weights', lgKV:'KV cache', lgOH:'runtime overhead', lgFree:'free', lgRes:'held back', lgOver:'over capacity',
  unpin:'Unpin', showMath:'Show the math', ctrlAria:'Settings', ctrlEyebrow:'Settings',
  cModel:'Model', cGpu:'GPU', cN:'GPUs, tensor parallel', cUsers:'Concurrent users', cWp:'Weight precision', cKp:'KV cache precision', cAttn:'Attention',
  attnNative:'GQA', attnMha:'MHA', adv:'Memory utilization and overhead', cUtil:'Memory utilization', cOh:'Runtime overhead per GPU', reset:'Reset',
  connEyebrow:'Connect', connTitle:'Take it further', cMin:'Minimal implementation', copyCode:'Copy code', cProd:'In production',
  cProd1:'At startup, vLLM loads the weights and runs one forward pass to measure peak activation memory. Then it gives the KV cache all the memory left under <code>gpu_memory_utilization</code> (0.92 by default in current releases), cut into fixed-size blocks. The PagedAttention lab shows why the blocks matter.',
  cProd2:'Source code: <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener">vllm-project/vllm</a>.',
  cPapers:'Papers', paper3:'DeepSeek-V2, which introduces multi-head latent attention (MLA)',
  paperBy1:'Ainslie et al., 2023', paperBy2:'Kwon et al., SOSP 2023', paperBy3:'DeepSeek-AI, 2024',
  cDrill:'Interview drill (optional)', drillReveal:'Reveal answer', drillNew:'New question',
  dockGo:'Back to the chart ↑',
  tipPin:'Save the current setup as A. As you change things, A stays on the chart for comparison.',
  tipStatus:'Fits means each GPU needs no more memory than it can use: its capacity times the memory utilization. Otherwise the setup is out of memory.',
  tipLgW:'The weights are the model’s parameters. Tensor parallelism splits them evenly across the GPUs.',
  tipLgKV:'The KV cache holds a key and a value vector for every token of every user’s context, in every layer. Thin lines split the amber block into one slice per user; they are left out when the slices get too narrow.',
  tipLgOH:'Runtime overhead is memory the serving engine needs besides the weights and the KV cache, such as one step’s temporary activations and CUDA graphs. It is an estimate, usually a GiB or two per GPU, because inference frees each layer’s activations right away.',
  ohNote:'Memory the serving engine needs besides the weights and the KV cache, such as one step’s temporary activations and CUDA graphs. It is an estimate: engines such as vLLM measure it at startup, and it grows with the tokens one step may process and the CUDA graphs captured. It is usually a GiB or two, because inference frees each layer’s activations right away.',
  utilNote:'The server uses at most this share of each GPU’s memory and keeps the rest as a safety margin, so memory the engine does not account for (the CUDA runtime, fragmentation) cannot push the GPU out of memory. It is set by hand: a smaller margin leaves more for the KV cache but makes running out of memory mid-run more likely. vLLM defaulted to 0.9 in earlier releases and uses 0.92 in current ones.',
  tipLgKVt:'The KV cache holds a key and a value vector for every token of every user’s context, in every layer.',
  tipLgOver0:'This is the memory still missing to hold every user at once. Here not even one user fits, so a real server such as vLLM would refuse to start: at startup it checks that the weights and one user’s full context fit.',
  lgWas:'dashed outline: before the change', tipLgWas:'The dashed outline marks how much memory each GPU used before your last change.',
  lgLim:'dashed line: usable limit', tipLgLim:'The dashed line is the usable limit: each GPU uses memory only up to here and keeps the rest as a safety margin.',
  tipLgFree:'Free memory is usable memory that nothing occupies yet.',
  tipLgRes:'The server never allocates this memory. It uses only a share of each GPU’s memory (the memory utilization) and keeps the rest as a safety margin, so memory the engine does not account for cannot push the GPU out of memory.',
  tipLgOver:'This is the memory still missing to hold every user at once. A real server such as vLLM would not crash; it would make the users who do not fit wait in a queue.',
  mTipTok:'One token of one user’s context needs this much KV cache, all layers together.',
  mTipReq:'One user’s whole context needs this much KV cache: the amount per token times the context length.',
  mTipAll:'This is all users’ KV cache added up, before it is split across the GPUs.',
  mTipUsers:'This many users fit at the current context length, in the memory each GPU has left after the weights and the runtime overhead.',
  mTipCtx:'With the current number of users, memory can hold at most this many tokens of context for each user. The model’s own limit can be lower.'
 },
 zh:{
  brandAria:'AI Systems Playground 首页', navAria:'主导航', navMap:'知识树', 
  langAria:'语言', themeAria:'切换明暗主题',
  heroTitle:'亲手操作，看懂现代 AI 系统。',
  heroCta:'为什么 70B 模型会爆显存？<span aria-hidden="true">→</span>',
  heroMapCta:'看知识树', heroSub:'从第一课开始：跟着一句话，看大模型怎样写出下一个词。不需要 AI 背景。', hmSub:'Llama-3.1-70B · 100 个并发用户', ctxLabel:'每个用户的上下文',
  mapTitle:'从最适合你的路线开始',
  pathsAria:'学习路线', chipAll:'不选路线，按枝看',
  foot2:'另外两个讲 Transformer 内部的可视化：<a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a>（GPT-2）和 <a href="https://bbycroft.net/llm" target="_blank" rel="noopener">LLM Visualization</a>。',
  ftBlurb:'亲手操作，看懂现代 AI 系统：从模型怎样写出下一个词，到训练、推理服务、后训练、评测、RAG 和 Agent。全部在你的浏览器里运行。',
  ftPaths:'学习路线', ftTree:'知识树', ftProject:'项目', ftSource:'源码（GitHub）', ftIssue:'报告问题',
  ftLegal:'代码采用 Apache 2.0 许可，字体采用 SIL Open Font License。',
  crumbMap:'← 知识树', crumbWorld:'LLM 怎么工作', crumbKV:'KV cache 显存', labTitle:'为什么 70B 模型一到长上下文就爆显存？',
  tagMin:'15 分钟', tagLevel:'核心', fidBtn:'保真度：公式计算 ▾', priBtn:'前置知识：KV cache 是什么 ▾',
  flowAria:'这个 Lab 的流程', modesAria:'Lab 模式', modeGuided:'分步引导', modeSandbox:'应用场景', modeChallenge:'挑战题', share:'复制配置链接',
  descGuided:'一共 7 个部分。先预测，再找到显存不足的临界点，看 KV cache 为什么这么大，最后试几种多装用户的办法。第一次学从这里开始。',
  descSandbox:'六个真实部署场景，从聊天服务到一张桌面显卡。选一个，看显存装不装得下、最多能服务几个用户。',
  descChallenge:'一台 8 × H100 跑 Llama-3.1-70B，每个用户 128K 上下文。只改两个精度参数，让它同时装下 20 个用户。',
  fidPop:'<div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul><li>权重的显存：参数量乘以每个参数的字节数。</li><li>每个 token 的 KV cache 由各模型的 config.json 算出：GQA 模型按 KV head 数，DeepSeek-V3 的 MLA 按压缩向量（latent）的大小。</li><li>张量并行把权重和 KV head 平均分到各张 GPU；MLA 的压缩向量在每张 GPU 上各存一整份。</li><li>服务端只用每张 GPU 显存的一定比例（vLLM 的 gpu_memory_utilization）。</li></ul></div><div><div class="eyebrow">简化和假设</div><ul><li>临时激活、CUDA graph 等合并成每张卡一个固定的运行时开销。这是估计值，真实引擎在启动时实测（<button type="button" class="linkbtn" data-jump="oh">在“自由调参”里调整</button>）。</li><li>每个用户都占满自己的上下文长度，这是最坏情况。</li><li>只算纯张量并行。SGLang 等框架可以改让 attention 做数据并行，避免把 MLA 的压缩向量复制到每张 GPU；这个 Lab 没有建模这种做法。</li><li>不计显存碎片（PagedAttention Lab 会讲）。</li><li>FP8 权重按每个参数 1 字节算，不计缩放系数。真实的 FP8 checkpoint 有一部分张量是 BF16，还要存缩放系数。Meta 发布的 Llama-3.1-405B FP8 版有 81.3B 个参数是 BF16，所有 attention 矩阵都在其中：8 张 GPU 时每张 56.7 GiB，而不是 47.2 GiB。场景“单机跑 405B”的 16 个用户仍装得下，但每人 8K 上下文时最多只能装下 53 个用户，不是 92 个。DeepSeek-V3 的真实 checkpoint 每张 GPU 约多 0.24 GiB，这个 Lab 里它的用户数都不变。</li></ul></div><div><div class="eyebrow">来源</div><ul><li>模型结构：各模型在 Hugging Face 上的 config.json。</li><li>显存：厂商标称容量，按 GiB 计；驱动报告的会略少。</li><li>vLLM 的 gpu_memory_utilization 默认值（vllm/config/cache.py）：v0.19 及以前是 0.9，v0.20.0 起是 0.92；这里的初始设置用 0.9。</li><li>SGLang 为 MLA 做的数据并行 attention：SGLang v0.4 发布说明（LMSYS 博客，2024-12-04）。</li><li>FP8 checkpoint：Hugging Face 上 meta-llama/Llama-3.1-405B-Instruct-FP8 和 deepseek-ai/DeepSeek-V3 各张量的数据类型。</li></ul><p class="mono small muted" style="margin-top:8px">最近核验 2026-09-26。原型，尚未经专家审阅。</p></div></div>',
  priPop:'<p>模型一次生成一个 token。每生成一个新 token，每一层都要拿它的 query 向量和之前所有 token 以及它自己的 key 向量比较，把得分换成加起来是 100% 的注意力权重，再按这些注意力权重把这些 token 的 value 向量混合起来。每一层同时有很多个 head 在做这件事：每个 head 有自己的 query，而 Llama 3、Mistral 等不少现在的模型让几个 head 共用一组 key 和 value。每个 token 只看它自己和之前的 token（因果 mask），所以之前 token 的 key 和 value 算出来就不会再变，服务端把它们存在显存里，不再重算。这块存储就是 KV cache。</p><p>学习路线“Transformer 结构”从头讲 query、key、value 和 head；生成时为什么要把 K 和 V 存起来，在其中的“<a href="#kv-basics" data-tipx="Lab：生成时为什么要把 K 和 V 存起来？">为什么要 KV cache</a>”这个 Lab 里讲。</p>',
  shareLabel:'这个链接会打开这个 Lab，并恢复同样的配置；链接里不包括你的预测和学习进度：', shareLabelFrag:'把这段加到页面链接的末尾，那个链接就会打开这个 Lab，并恢复同样的配置；它不包括你的预测和学习进度：', railAria:'Lab 各部分',
  kvOutcome:'<b>学完你能：</b>估算一台 GPU 服务器在给定上下文下能装下多少用户，装不下时选对办法。', nextK:'下一个 Lab', prevK:'上一个 Lab', nextLast:'这是目前最后一个能打开的 Lab。', nextMap:'回到知识树，看看规划中的 Lab →', chipTf:'Transformer 结构',
  lgW:'权重', lgKV:'KV cache', lgOH:'运行时开销', lgFree:'空闲', lgRes:'留作余量', lgOver:'超出容量',
  unpin:'取消固定', showMath:'展开公式', ctrlAria:'自由调参', ctrlEyebrow:'自由调参',
  cModel:'模型', cGpu:'GPU', cN:'GPU 数（张量并行）', cUsers:'并发用户数', cWp:'权重精度', cKp:'KV cache 精度', cAttn:'注意力',
  attnNative:'GQA', attnMha:'MHA', adv:'显存利用率和运行时开销', cUtil:'显存利用率', cOh:'每卡运行时开销', reset:'重置',
  connEyebrow:'延伸', connTitle:'继续深入', cMin:'最小实现', copyCode:'复制代码', cProd:'生产中怎么做',
  cProd1:'vLLM 启动时先加载权重，再跑一次前向，测出激活最多要占多少显存。然后在 <code>gpu_memory_utilization</code>（当前版本默认 0.92）允许的范围内，把剩下的显存全给 KV cache，切成固定大小的 block。为什么要切成 block，PagedAttention 这个 Lab 会讲。',
  cProd2:'源码：<a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener">vllm-project/vllm</a>。',
  cPapers:'论文', paper3:'DeepSeek-V2（提出 multi-head latent attention，即 MLA）',
  paperBy1:'Ainslie 等，2023', paperBy2:'Kwon 等，SOSP 2023', paperBy3:'DeepSeek-AI，2024',
  cDrill:'面试题（可选）', drillReveal:'看答案', drillNew:'换一题',
  dockGo:'回到图表 ↑',
  tipPin:'把当前配置存为 A。之后改参数时，A 会留在图上作对比。',
  tipStatus:'“装得下”是说每张 GPU 需要的显存不超过它的可用显存，即显存容量乘以显存利用率。超过了就是显存不足。',
  tipLgW:'权重是模型的参数。张量并行把它们平均分到各张 GPU 上。',
  tipLgKV:'KV cache 存着每个用户上下文里每个 token 在每一层的 key 和 value 向量。琥珀色里的细竖线把它隔成每个用户一段；每段太窄时不画。',
  tipLgOH:'运行时开销是推理引擎在权重和 KV cache 之外要占的显存，比如一步前向的临时激活和 CUDA graph。它是估计值，通常每卡一两个 GiB，因为推理时每层的激活用完就释放。',
  ohNote:'推理引擎在权重和 KV cache 之外要占的显存，比如一步前向的临时激活和 CUDA graph。这是估计值：vLLM 这类引擎在启动时实测它，一步最多处理的 token 越多、录制的 CUDA graph 越多，它就越大。它通常只有一两个 GiB，因为推理时每层的激活用完就释放。',
  utilNote:'服务端最多只用每张卡显存的这个比例，剩下的留作安全余量，免得引擎没算到的占用（如 CUDA 运行时、显存碎片）把显存撑爆。它是人为设定的：余量留得越少，KV cache 能用的越多，但越容易在运行中显存不足。vLLM 早期版本默认 0.9，当前版本默认 0.92。',
  tipLgKVt:'KV cache 存着每个用户上下文里每个 token 在每一层的 key 和 value 向量。',
  tipLgOver0:'这一段是同时装下所有用户还缺的显存。这里一个用户都装不下，vLLM 这类真实服务会直接拒绝启动：它在启动时检查权重和一个用户的完整上下文能不能放下。',
  lgWas:'虚线框：改之前', tipLgWas:'虚线框标出上一次改设置之前，每张 GPU 用了多少显存。',
  lgLim:'虚线：可用上限', tipLgLim:'虚线是可用上限：每张卡的显存只用到这里，右边的留作安全余量。',
  tipLgFree:'空闲的是可用显存里还没被占用的部分。',
  tipLgRes:'这部分显存服务端从不分配。服务端只用每张卡显存的一部分（显存利用率），剩下的留作安全余量，免得引擎没算到的占用把显存撑爆。',
  tipLgOver:'这一段是同时装下所有用户还缺的显存。vLLM 这类真实服务不会因此崩溃，而是让装不下的用户排队等待。',
  mTipTok:'一个用户上下文里的一个 token，所有层加起来要存这么多 KV cache。',
  mTipReq:'一个用户的整段上下文要存这么多 KV cache：每个 token 的量乘以上下文长度。',
  mTipAll:'这是所有用户的 KV cache 加起来的总量，还没分到各张 GPU 上。',
  mTipUsers:'每张 GPU 的可用显存扣掉权重和运行时开销后，按当前的上下文长度，最多能装下这么多用户。',
  mTipCtx:'按当前的用户数，显存最多能给每个用户放这么多个 token 的上下文。模型本身的上限可能更低。'
 }
};
/* the KV lab's static texts that name a model keep the name on one line: a no-break hyphen, as NB() does for the rest */
['en','zh'].forEach(l=>['fidPop','paper3'].forEach(k=>{STR[l][k]=STR[l][k].replace(/\bDeepSeek-V(\d)\b/g,'DeepSeek\u2011V$1');}));
const t=k=>{const v=STR[lang][k]; return v!=null?v:(STR.en[k]!=null?STR.en[k]:k);};
function applyStatic(){
  document.documentElement.lang=lang==='zh'?'zh-CN':'en';
  $$('[data-i18n]').forEach(el=>{el.textContent=t(el.dataset.i18n);});
  $$('[data-i18n-html]').forEach(el=>{el.innerHTML=t(el.dataset.i18nHtml);});
  $$('[data-i18n-aria]').forEach(el=>{el.setAttribute('aria-label',t(el.dataset.i18nAria));});
  $$('.langsw button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lang===lang)));
  const cw=$('#kvCrumbWorld'); if(cw) cw.dataset.tipx=crumbWorldTip('models');   // on the trunk since 2026-10-12 (owner)
  const en=$('#fidPop'); if(en) en.innerHTML=endnotesHead('kv')+`<div class="en-body">${endnotesInner(t('fidPop'))}</div>`;
  const hk=$('#heroKicker span'); if(hk){const n=LABS.filter(l=>labStatus(l)==='live').length; hk.textContent=tx(`One knowledge tree, ${n} hands-on labs`,`一棵知识树，${n} 个可以亲手操作的实验`);}
  kvCrumbs(); renderFoot();
}
/* the page's footer: what the site is, the learning paths and the tree's branches (the same links as the home page's), the project */
const REPO='https://github.com/Schuture/ai-systems-playground';
function renderFoot(){const g=$('#ftGrid'); if(!g) return; const mark=($('.brand svg')||{}).outerHTML||'';
  const col=(h,items)=>`<div class="ft-col"><h2 class="ft-h">${esc(h)}</h2><ul>${items.map(([href,txt,ext])=>`<li><a href="${esc(href)}"${ext?' target="_blank" rel="noopener"':''}>${esc(txt)}</a></li>`).join('')}</ul></div>`;
  g.innerHTML=`<div class="ft-about"><a class="ft-brand" href="#top">${mark}<span>AI Systems Playground</span></a><p>${esc(t('ftBlurb'))}</p></div>`+
    col(t('ftPaths'),Object.entries(PATHS).map(([k,p])=>['#map~route:'+k,L(p.name)]))+
    col(t('ftTree'),Object.entries(WORLDS).map(([k,w])=>['#map~'+k,L(w.name)]))+
    col(t('ftProject'),[[REPO,t('ftSource'),1],[REPO+'/issues',t('ftIssue'),1]]);}
// the KV lab's breadcrumb row (body.html): the previous and next lab of the learner's path, filled again on every visit and language switch
function kvCrumbs(){const nc=$('#kvCrumbNav'); if(nc) nc.innerHTML=crumbPairHTML('kv-cache');}

/* ---------- catalog: model shapes from Hugging Face config.json, HBM from vendor specs ---------- */
const KiB=1024, MiB=KiB*1024, GiB=MiB*1024, TiB=GiB*1024;
const MODELS={
  'llama31-8b':{name:'Llama-3.1-8B',params:8030261248,layers:32,heads:32,kvHeads:8,headDim:128,attn:'gqa',maxCtx:131072,note:{en:'32 layers, 32 query heads, 8 KV heads, 8.03B parameters.',zh:'32 层，32 个 query head，8 个 KV head，8.03B 个参数。'}},
  'llama31-70b':{name:'Llama-3.1-70B',params:70553706496,layers:80,heads:64,kvHeads:8,headDim:128,attn:'gqa',maxCtx:131072,note:{en:'80 layers, 64 query heads, 8 KV heads, 70.6B parameters.',zh:'80 层，64 个 query head，8 个 KV head，70.6B 个参数。'}},
  'llama31-405b':{name:'Llama-3.1-405B',params:405853388800,layers:126,heads:128,kvHeads:8,headDim:128,attn:'gqa',maxCtx:131072,note:{en:'126 layers, 128 query heads, 8 KV heads, 406B parameters (the name rounds them down to 405B).',zh:'126 层，128 个 query head，8 个 KV head，406B 个参数（名字里往下取整成 405B）。'}},
  'mixtral-8x7b':{name:'Mixtral-8x7B',params:46.7e9,layers:32,heads:32,kvHeads:8,headDim:128,attn:'gqa',maxCtx:32768,note:{en:'A mixture of experts: 32 layers, 8 KV heads, 46.7B parameters. Each token uses only 12.9B of them, but all 46.7B must sit in GPU memory.',zh:'混合专家（MoE）模型：32 层，8 个 KV head，46.7B 个参数。每个 token 只用到其中 12.9B 个，但这 46.7B 个参数全都要放进显存。'}},
  'deepseek-v3':{name:'DeepSeek-V3',params:671e9,layers:61,heads:128,attn:'mla',kvLora:512,ropeDim:64,maxCtx:131072,defaultWp:'fp8',note:{en:'A mixture of experts with MLA: 61 layers, 671B parameters. Each token uses only 37B of them, but all 671B must sit in GPU memory.',zh:'用 MLA 的混合专家（MoE）模型：61 层，671B 个参数。每个 token 只用到其中 37B 个，但这 671B 个参数全都要放进显存。'}}
};
const GPUS={
  'rtx4090':{name:'RTX 4090',mem:24},
  'h100':{name:'H100 80GB',mem:80},
  'h200':{name:'H200 141GB',mem:141},
  'mi300x':{name:'MI300X 192GB',mem:192}
};
const PREC={bf16:2,fp8:1,fp4:0.5};
const PL={bf16:'BF16',fp8:'FP8',fp4:'FP4'};
const DEFAULT={model:'llama31-70b',gpu:'h100',n:8,ctx:8192,users:100,wp:'bf16',kp:'bf16',attn:'gqa',util:0.9,overhead:2};
const CTX=[1024,2048,4096,8192,12288,16384,24576,32768,49152,65536,98304,131072,262144,524288,1048576];
const USERS=[1,2,4,8,10,12,16,24,32,48,64,100,128,192,256,384,512];

/* ---------- engine: a pure function of the parameters ---------- */
function compute(s){
  const m=MODELS[s.model], g=GPUS[s.gpu];
  const mla=m.attn==='mla';
  const kvHeads=mla?0:(s.attn==='mha'?m.heads:m.kvHeads);
  const bKV=PREC[s.kp], bW=PREC[s.wp];
  const kvPerToken=mla? m.layers*(m.kvLora+m.ropeDim)*bKV : 2*m.layers*kvHeads*m.headDim*bKV;
  const kvPerReq=kvPerToken*s.ctx;
  const kvTotal=kvPerReq*s.users;
  const shard=mla?1:Math.min(s.n,kvHeads);          // MLA latent is replicated under tensor parallelism
  const kvPerGpu=kvTotal/shard;
  const wTotal=m.params*bW, wPerGpu=wTotal/s.n;
  const cap=g.mem*GiB, usable=cap*s.util, oh=s.overhead*GiB;
  const need=wPerGpu+kvPerGpu+oh;
  const budget=usable-wPerGpu-oh;
  const maxUsers=budget>0?Math.floor(budget*shard/kvPerReq+1e-9):0;
  const maxCtx=budget>0?Math.floor(budget*shard/(s.users*kvPerToken)+1e-9):0;
  return {s,m,g,n:s.n,mla,kvHeads,kvPerToken,kvPerReq,kvTotal,shard,kvPerGpu,wTotal,wPerGpu,cap,usable,oh,need,budget,maxUsers,maxCtx,
    fits:need<=usable,weightsFit:budget>0,overCtx:s.ctx>m.maxCtx,ratio:need/usable};
}


/* ---------- formatting ---------- */
const trim0=s=>s.replace(/\.0(?=\s|$)/,'');
function fmtB(b){const a=Math.abs(b);let v,u;
  if(a>=TiB){v=b/TiB;u='TiB'}else if(a>=GiB){v=b/GiB;u='GiB'}else if(a>=MiB){v=b/MiB;u='MiB'}else if(a>=KiB){v=b/KiB;u='KiB'}else return Math.round(b)+' B';
  return trim0(v.toFixed(Math.abs(v)>=100?0:1))+' '+u;}
const fmtNum=(v,d)=>trim0(v.toFixed(d));
function fmtTok(t){if(t>=1048576&&t%1048576===0)return (t/1048576)+'M';if(t%1024===0)return (t/1024)+'K';return trim0((t/1024).toFixed(1))+'K';}
const fmtInt=n=>Math.round(n).toLocaleString('en-US');
/* R6.18: more than 6 digits → 3 significant digits with M or B (8.03B, 41.9M, 30.0B) (A.fmt.big, A.ui.big); no hover with the exact value */
const fmtBig=n=>{const a=Math.abs(n); if(a<1e6) return fmtInt(n);
  let d=a>=1e9?1e9:1e6, u=d===1e9?'B':'M', v=n/d, dp=Math.max(0,2-Math.floor(Math.log10(Math.abs(v)))), t=v.toFixed(dp);
  if(u==='M'&&Math.abs(+t)>=1000){d=1e9; u='B'; v=n/d; dp=Math.max(0,2-Math.floor(Math.log10(Math.abs(v)))); t=v.toFixed(dp);}
  return t+u;};
/* GiB only, never switching to TiB, so values side by side share a unit */
const fmtG=b=>{const g=b/GiB; return (Math.abs(g)>=100?fmtInt(g):fmtNum(g,1))+' GiB';};
/* the KV lab's sizes: KiB and MiB below 1 GiB, otherwise GiB (never TiB, so they compare with the 72 GiB a GPU can use) */
const fmtKV=b=>Math.abs(b)>=GiB?(b/GiB).toLocaleString('en-US',{maximumFractionDigits:1})+' GiB':fmtB(b);
/* per-GPU totals with one decimal, and the margin computed from those shown values, so every formula line redoes (R6.6) */
const r1=v=>Math.round(v*10)/10, gib1=v=>v.toLocaleString('en-US',{maximumFractionDigits:1})+' GiB';
/* weights and totals always with one decimal (15.0 GiB), as in the other labs (R6.6) */
const gibW=v=>v.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1})+' GiB';
const dfix=(v,d)=>{const k=10**d; return Math.round(v*k+1e-7)/k;}, fmtD=(v,d)=>v.toLocaleString('en-US',{maximumFractionDigits:d});   // +1e-7: 22.35 rounds to 22.4 despite binary 22.3499…
/* Show the math as shown: every result is computed from the operands as displayed, so each line redoes (R6.6).
   Per token: exact (KiB or MiB, up to 4 decimals). Per user: enough decimals that times the users it rounds to the true total.
   GiB sums, usable memory and the margin: one decimal. The status badge, the bars and the mini bar use these same numbers. */
function chain(r){
  const s=r.s, tokU=r.kvPerToken>=MiB?MiB:KiB, tokS=fmtD(r.kvPerToken/tokU,4)+(tokU===MiB?' MiB':' KiB');
  const pu=r.kvPerToken*s.ctx, puU=pu>=GiB?GiB:(pu>=MiB?MiB:KiB), puN=puU===GiB?'GiB':(puU===MiB?'MiB':'KiB'), v0=pu/puU;
  // the total keeps about as many digits as the per-user value, so one change shows one rate on both cards: one decimal
  // (34.3 GiB, 549 GiB), two below 10 GiB (6.43 GiB) or when exactly two hold it (31.25 GiB, not 31.3), three below 1 GiB
  const gT=r.kvTotal/GiB, has=k=>Math.abs(gT*k-Math.round(gT*k))<1e-6, totDec=gT<1?3:((gT<10||(!has(10)&&has(100)))?2:1), trueTot=dfix(gT,totDec);
  let d=v0>=100?1:(v0>=10?2:3), puV, tot;
  for(;;){puV=dfix(v0,d); const g=puV*puU*s.users/GiB; tot=dfix(g,totDec); if(tot===trueTot||d>=6) break; d++;}
  const W=dfix(r.wTotal/GiB,1), oh=s.overhead, kvG=r.mla?tot:tot/r.shard;
  const need=dfix(W/s.n+kvG+oh,1), use=dfix(r.g.mem*s.util,1);
  return {tokS,puS:fmtD(puV,d)+' '+puN,tot,totS:fmtD(tot,totDec)+' GiB',W,oh,kvG:dfix(kvG,kvG>=1?1:3),need,use,margin:dfix(use-need,1)};
}
/* the margin in words: "22.3 GiB free" / "446.4 GiB short", never "0 GiB short" */
const marginAmt=c=>Math.abs(c.margin)>=0.1?gib1(Math.abs(c.margin)):tx('less than 0.1 GiB','不到 0.1 GiB');
const minsTxt=n=>tx(`${n} min`,`${n} 分钟`);
const usersTxt=n=>tx(n===1?'1 user':`${fmtInt(n)} users`,`${fmtInt(n)} 个用户`);
const labsTxt=n=>tx(`${n} lab${n===1?'':'s'}`,`${n} 个 Lab`);
const gpusTxt=n=>tx(n===1?'1 GPU':`${n} GPUs`,`${n} 张 GPU`);
const tokTxt=s=>tx(`${s} tokens`,`${s} 个 token`);
/* model names and "8 × H100" never break across lines: no-break hyphens (U+2011) inside the names, no-break spaces around ×.
   Only for what the KV lab and its tile show; copying from the page gives plain hyphens and spaces again (see the copy handler) */
/* a term the reader may not know yet: dotted underline, question-mark cursor, its definition on hover and focus (R2.8) */
const DEF=(text,tip)=>`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const NB=s=>String(s).replace(/\b(?:Llama-3\.1-\d+B|Mixtral-8x7B|DeepSeek-V\d)\b/g,m=>m.replace(/-/g,'\u2011')).replace(/(\d) × (?=H100|H200|RTX|MI300X|A100)/g,'$1\u00a0×\u00a0');

/* ---------- resource bars ---------- */
const perGpuBar=r=>({cap:r.cap,usable:r.usable,w:r.wPerGpu,oh:r.oh,kv:r.kvPerGpu,need:r.need});
const aggBar=r=>({cap:r.cap*r.n,usable:r.usable*r.n,w:r.wTotal,oh:r.oh*r.n,kv:r.kvPerGpu*r.n,need:r.need*r.n});
function makeBar(host){
  host.innerHTML='<div class="cap"></div><div class="seg w"><span></span></div><div class="seg oh"></div><div class="seg kv"><span></span></div><div class="seg free"></div><div class="seg res"></div><div class="seg over" hidden><span></span></div><div class="usable"><span></span></div><div class="bar-ghost"></div>';
  const q=c=>host.querySelector('.'+c);
  host._e={cap:q('cap'),w:q('w'),oh:q('oh'),kv:q('kv'),free:q('free'),res:q('res'),over:q('over'),usable:q('usable'),ghost:q('bar-ghost')};
}
/* estimated width in px of the red segment's label (11 px mono): 7 px per Latin character, 12 per CJK or full-width one;
   a two-line label counts its wider line */
const textW=t=>Math.max(...t.split('\n').map(l=>{let n=0; for(const ch of l) n+=ch.codePointAt(0)>=0x2e80?12:7; return n;}));
function drawBar(host,b,o){
  o=o||{}; if(!host._e) makeBar(host);
  const E=host._e, W=o.capw||80, SC=o.scaleCap||b.cap;
  const X=v=>Math.max(0,v/SC*W);
  const put=(el,a,z)=>{el.style.left=X(a)+'%';el.style.width=Math.max(0,X(z)-X(a))+'%';};
  put(E.cap,0,b.cap);
  const u=b.usable; let x=0;
  const wIn=Math.min(b.w,u); put(E.w,0,wIn); x=wIn;
  const ohIn=Math.min(b.oh,Math.max(0,u-x)); put(E.oh,x,x+ohIn); x+=ohIn;
  const kvIn=Math.min(b.kv,Math.max(0,u-x)); put(E.kv,x,x+kvIn); x+=kvIn;
  put(E.free,x,u); put(E.res,u,b.cap);
  let end=X(x);
  const over=b.need-u;
  if(over>1){
    const Lx=X(u), full=X(b.need)-Lx, room=100-Lx-0.5, w=Math.min(full,room);
    E.over.hidden=false; E.over.style.left=Lx+'%'; E.over.style.width=w+'%';
    E.over.classList.toggle('clipped',full>room);
    // the longest label that fits: "(cut off)" when the segment is clipped, then the plain one, then none (R7.4)
    const px=host.clientWidth*w/100, cands=[full>room&&o.overTextCut, o.overText||('+'+fmtB(over))].filter(Boolean);
    E.over.firstChild.textContent=o.noText?'':(cands.find(t=>px>=textW(t)+22)||'');
    end=Lx+w;
  } else E.over.hidden=true;
  E.usable.style.left=X(u)+'%';
  E.usable.firstChild.textContent=tx('usable ','可用 ')+(o.useFmt?o.useFmt(u):gibW(u/GiB));
  const lab=(el,long,short,wp)=>{const px=host.clientWidth*wp/100;
    el.firstChild.textContent=o.noText?'':(px>=long.length*7+14?long:(px>=short.length*7+12?short:''));};
  const F=o.fmt||fmtB;
  const FW=o.wfmt||F; lab(E.w,tx('weights ','权重 ')+FW(b.w),FW(b.w),X(wIn));
  lab(E.kv,'KV '+F(b.kv),F(b.kv),X(kvIn));
  // host._ticks: the thin per-user lines are drawn and at least two slices show, so a caption may name them
  if(o.perUser){const px=host.clientWidth*X(o.perUser)/100; host._ticks=px>=3.5&&host.clientWidth*X(kvIn)/100>=2*px;
    if(px>=3.5){E.kv.style.backgroundImage='linear-gradient(90deg,transparent calc(100% - 1px),var(--kv-tick) 0)';E.kv.style.backgroundSize=px+'px 100%';}
    else E.kv.style.backgroundImage='none';}
  else {E.kv.style.backgroundImage='none'; host._ticks=false;}
  if(o.ghost&&host._end!=null&&Math.abs(host._end-end)>0.4&&!reduceMotion()){
    const g=E.ghost; g.style.transition='none'; g.style.width=host._end+'%'; g.style.opacity='1';
    void g.offsetWidth; g.style.transition='opacity 1.4s ease-out'; g.style.opacity='0';
  }
  host._end=end;
}
function drawAxis(host,cap,o){o=o||{};const W=o.capw||80,SC=o.scaleCap||cap;
  host.innerHTML=[0,.25,.5,.75,1].map((f,i)=>`<span style="left:${f*cap/SC*W}%">${fmtNum(f*cap/GiB,0)}${i===4?' GiB':''}</span>`).join('');}
const reduceMotion=()=>window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
function setFill(el){const lo=el.min===''?0:+el.min, hi=el.max===''?100:+el.max, p=hi>lo?(el.value-lo)/(hi-lo)*100:0; el.style.setProperty('--fill',p+'%');}
/* the fill follows the value however code sets it (kit polish K5): the track's dark part is --fill, which only the reader's own input
   events (the document listener in start()) and A.ui.setRange wrote, so a slider moved in code (input.value = …, valueAsNumber, a new
   min or max, a slider rendered without initRanges) kept its old fill until the reader dragged it (PR #1: pd-disagg's progress slider
   ran ahead of its dark line; that lab now calls setRange every frame). Writing value or valueAsNumber of a range input now repaints
   it; kitObs repaints a range input whose min, max, step or value attribute changes and every range input added to the page */
(function(){const P=window.HTMLInputElement&&HTMLInputElement.prototype; if(!P) return;
  ['value','valueAsNumber'].forEach(k=>{const d=Object.getOwnPropertyDescriptor(P,k); if(!d||!d.set||!d.configurable) return;
    Object.defineProperty(P,k,{configurable:true,enumerable:d.enumerable,get:d.get,set(v){d.set.call(this,v); if(this.type==='range') setFill(this);}});});})();
const fillAll=n=>{if(!n||n.nodeType!==1) return; if(n.matches('input[type=range]')) setFill(n); else if(n.firstElementChild) n.querySelectorAll('input[type=range]').forEach(setFill);};

/* ---------- hero meter ---------- */
const HCTX=[4096,8192,12288,16384,32768,65536,131072];
function heroUpdate(){
  const inp=$('#hmCtx'); setFill(inp);
  const ctx=HCTX[+inp.value], r=compute({...DEFAULT,ctx}), b=aggBar(r), pc=Math.round(r.s.util*100);
  const gv=x=>Math.round(x/GiB*10)/10, gs=v=>fmtD(v,1)+' GiB';
  const need=dfix(gv(b.w)+gv(b.oh)+gv(b.kv),1), use=gv(b.usable);
  drawBar($('#hmBar'),b,{ghost:true,fmt:x=>gs(gv(x)),useFmt:u=>fmtNum(u/GiB,0)+' GiB',perUser:b.kv/r.s.users,overText:tx('needs ','需要 ')+gs(need)});
  drawAxis($('#hmAxis'),b.cap);
  $('#hmCtxOut').textContent=fmtTok(ctx)+tx(' tokens',' token');
  const st=$('#hmStatus'); st.className='status '+(r.fits?'ok':'bad'); st.textContent=r.fits?tx('Fits','装得下'):tx('Out of memory','显存不足');
  const parts=tx(`weights ${gs(gv(b.w))} + overhead ${gs(gv(b.oh))} + KV ${gs(gv(b.kv))}`,`权重 ${gs(gv(b.w))} + 开销 ${gs(gv(b.oh))} + KV ${gs(gv(b.kv))}`);
  $('#hmRead').innerHTML=tx(
    `KV cache = 100 users × ${fmtInt(ctx)} tokens × ${fmtB(r.kvPerToken)} = <b>${fmtG(r.kvTotal)}</b><br>`+
      (r.fits?`Needs ${gs(need)} (${parts}) of the ${gs(use)} the 8 GPUs can use: ${pc}% of 8 × 80 GiB (the 80 GB spec read as GiB).`:`Needs ${gs(need)} (${parts}); the 8 GPUs can use ${gs(use)}, so ${gs(dfix(need-use,1))} short.`),
    `KV cache = 100 个用户 × ${fmtInt(ctx)} token × ${fmtB(r.kvPerToken)} = <b>${fmtG(r.kvTotal)}</b><br>`+
      (r.fits?`需要 ${gs(need)}（${parts}），8 张卡共可用 ${gs(use)}：8 × 80 GiB 的 ${pc}%（标称 80 GB 按 GiB 计）。`:`需要 ${gs(need)}（${parts}），8 张卡共可用 ${gs(use)}，差 ${gs(dfix(need-use,1))}。`));
  $('#hmHint').textContent=r.fits
    ?(ctx>=12288?tx('Close to the edge. Try one more notch.','快到边界了，再往右拖一格试试。'):tx('Drag the slider right. The amber block is the KV cache.','向右拖动滑块，琥珀色那块就是 KV cache。'))
    :tx(`It breaks near ${fmtTok(r.maxCtx)} tokens per user, far below 128K. The lab shows why.`,`每个用户约 ${fmtTok(r.maxCtx)} token 就装不下了，远低于 128K。Lab 里会讲为什么。`);
}

/* ---------- map ---------- */
/* the lab header uses the same three words as the home tiles (R2.9): Formula / Toy run / Illustrative */
const FID={Analytical:{en:'Formula',zh:'公式计算'},Simulated:{en:'Toy run',zh:'玩具实跑'},Replay:{en:'Replay',zh:'回放'},Live:{en:'Toy run',zh:'玩具实跑'},Conceptual:{en:'Illustrative',zh:'示意'}};
const LABS=[
 {id:'paged-attention',w:'inference',min:12,st:'wip',fid:'Simulated',s:{en:'PagedAttention',zh:'PagedAttention'},q:{en:'Why is most KV cache memory wasted?',zh:'为什么 KV cache 显存有一大半被浪费？'}},
 {id:'continuous-batching',w:'inference',min:15,st:'wip',fid:'Simulated',s:{en:'Continuous batching',zh:'Continuous batching'},q:{en:'Why does continuous batching raise throughput?',zh:'continuous batching 为什么能提高吞吐？'}},
 {id:'prefill-decode',w:'inference',min:12,st:'planned',fid:'Analytical',s:{en:'Prefill vs decode',zh:'Prefill 和 decode'},q:{en:'Why is the first token slow and the rest fast?',zh:'为什么首 token 要等很久，之后却一个接一个很快？'}},
 {id:'chunked-prefill',w:'inference',min:12,st:'planned',fid:'Simulated',s:{en:'Chunked prefill',zh:'Chunked prefill'},q:{en:'Why does one huge prompt slow everyone down?',zh:'一个超长 prompt 为什么会拖慢所有人？'}},
 {id:'prefix-caching',w:'inference',min:10,st:'planned',fid:'Simulated',s:{en:'Prefix caching',zh:'Prefix caching'},q:{en:'How can a 10,000-token system prompt be nearly free?',zh:'10,000 个 token 的 system prompt 怎么做到几乎免费？'}},
 {id:'serving-capstone',w:'inference',min:20,st:'planned',fid:'Simulated',s:{en:'Serving capstone',zh:'Serving 综合挑战'},q:{en:'How do you rescue a congested chat service?',zh:'怎么救活一个拥堵的聊天服务？'}},
 {id:'quantization',w:'inference',min:15,st:'planned',fid:'Live',s:{en:'Quantization',zh:'量化'},q:{en:'BF16 → FP8 → FP4: what do you actually lose?',zh:'BF16 → FP8 → FP4，到底丢了什么？'}},
 {id:'spec-decoding',w:'inference',min:12,st:'planned',fid:'Simulated',s:{en:'Speculative decoding',zh:'投机解码'},q:{en:'How can guessing speed up generation without changing it?',zh:'靠“猜”怎么能更快，而且不改变输出？'}},
 {id:'pd-disagg',w:'inference',min:12,st:'planned',fid:'Simulated',s:{en:'P/D disaggregation',zh:'P/D 分离'},q:{en:'Why run prefill and decode on different GPUs?',zh:'为什么把 prefill 和 decode 放到不同的 GPU 上？'}},
 {id:'multi-gpu-serving',w:'inference',min:12,st:'planned',fid:'Analytical',s:{en:'Multi-GPU serving',zh:'多卡推理'},q:{en:'How is one model served across 8 GPUs?',zh:'一个模型怎么在 8 张卡上服务？'}},
 {id:'admission',w:'inference',min:20,st:'planned',fid:'Simulated',s:{en:'Capacity capstone',zh:'容量规划综合挑战'},q:{en:'How many GPUs does your traffic need, and who should wait?',zh:'你的流量要几张卡，谁该排队？'}},
 {id:'training-memory',w:'training',min:12,st:'planned',fid:'Analytical',s:{en:'Training memory',zh:'训练显存'},q:{en:'An 8B model’s weights take 16 GB: why does training it not fit on an 80 GB GPU?',zh:'8B 模型的权重只有 16 GB，为什么训练它，80 GB 的卡都放不下？'}},
 {id:'parallelism',w:'training',min:15,st:'planned',fid:'Analytical',s:{en:'ZeRO and FSDP',zh:'ZeRO 与 FSDP'},q:{en:'How do you train a model that does not fit on one GPU?',zh:'一张卡放不下的模型，怎么训练？'}},
 {id:'model-parallel',w:'training',min:15,st:'planned',fid:'Analytical',s:{en:'Pipeline & tensor',zh:'流水线与张量并行'},q:{en:'How do pipeline and tensor parallelism cut one model apart?',zh:'流水线并行和张量并行怎么切开一个模型？'}},
 {id:'scaling-law',w:'training',min:12,st:'planned',fid:'Analytical',s:{en:'Scaling laws',zh:'Scaling law'},q:{en:'How big a model should a compute budget buy?',zh:'一笔算力预算，该训多大的模型？'}},
 {id:'data-pipeline',w:'training',min:12,st:'planned',fid:'Simulated',s:{en:'Data pipeline',zh:'数据管线'},q:{en:'What gets thrown away before pretraining?',zh:'预训练之前，数据要扔掉哪些？'}},
 {id:'checkpointing',w:'training',min:10,st:'planned',fid:'Simulated',s:{en:'Checkpoints',zh:'检查点与容错'},q:{en:'Your run crashed at step 90,000: how do you resume it?',zh:'训练到第 90,000 步崩了，怎么接着训？'}},
 {id:'train-capstone',w:'training',min:20,st:'planned',fid:'Analytical',s:{en:'Training capstone',zh:'训练综合挑战'},q:{en:'How do you plan the memory, parallelism and time of a 70B run?',zh:'训练一个 70B 模型，怎么规划显存、并行和时间？'}},
 {id:'sft',w:'post',min:12,st:'planned',fid:'Live',s:{en:'SFT',zh:'SFT'},q:{en:'Why does SFT count the loss on the answer only?',zh:'SFT 为什么只算回答部分的 loss？'}},
 {id:'lora',w:'post',min:12,st:'planned',fid:'Analytical',s:{en:'LoRA',zh:'LoRA'},q:{en:'How does LoRA fine-tune an 8B model on one GPU?',zh:'LoRA 怎么在一张卡上微调 8B 模型？'}},
 {id:'reward-model',w:'post',min:12,st:'planned',fid:'Live',s:{en:'Reward model',zh:'奖励模型'},q:{en:'How does a reward model turn preferences into scores?',zh:'奖励模型怎么把偏好变成分数？'}},
 {id:'ppo',w:'post',min:15,st:'planned',fid:'Live',s:{en:'PPO',zh:'PPO'},q:{en:'Why does PPO need a value model and a KL penalty?',zh:'PPO 为什么需要价值模型和 KL 惩罚？'}},
 {id:'dpo',w:'post',min:15,st:'wip',fid:'Live',s:{en:'DPO',zh:'DPO'},q:{en:'How does DPO pull a model toward preferences without a reward model?',zh:'不训练奖励模型，DPO 怎么把模型拉向偏好？'}},
 {id:'grpo',w:'post',min:12,st:'planned',fid:'Live',s:{en:'GRPO',zh:'GRPO'},q:{en:'Why can GRPO drop the value model?',zh:'GRPO 为什么能去掉价值模型？'}},
 {id:'rlvr',w:'post',min:15,st:'planned',fid:'Simulated',s:{en:'RLVR',zh:'RLVR'},q:{en:'How does a model learn to reason from right or wrong alone?',zh:'只给“对 / 错”，模型怎么学会推理？'}},
 {id:'post-capstone',w:'post',min:20,st:'planned',fid:'Analytical',s:{en:'Post-training capstone',zh:'后训练综合挑战'},q:{en:'SFT, DPO or GRPO: which one fits your task?',zh:'SFT、DPO 还是 GRPO：你的任务该选哪个？'}},
 {id:'architecture',w:'models',min:12,st:'wip',fid:'Conceptual',s:{en:'Architecture overview',zh:'结构总览'},q:{en:'What is a Transformer made of?',zh:'Transformer 由哪些部件组成？'}},
 {id:'transformer',w:'models',min:15,st:'wip',fid:'Analytical',s:{en:'Model parameters',zh:'模型参数'},q:{en:'Where do an 8B model’s parameters live?',zh:'8B 模型的参数都在哪？'}},
 {id:'mha',w:'models',min:15,st:'wip',fid:'Live',s:{en:'Multi-head attention',zh:'多头注意力'},q:{en:'Why does attention use many heads?',zh:'attention 为什么要分成多个 head？'}},
 {id:'kv-basics',w:'models',min:12,st:'wip',fid:'Live',s:{en:'Why a KV cache',zh:'为什么要 KV cache'},q:{en:'Why does generation keep every token’s keys and values?',zh:'生成时为什么要把 K 和 V 存起来？'}},
 {id:'kv-cache',w:'models',min:15,st:'live',fid:'Analytical',s:{en:'KV cache memory',zh:'KV cache 显存'},q:{en:'Why does a 70B model run out of memory at long context?',zh:'为什么 70B 模型一到长上下文就爆显存？'}},
 {id:'gqa-mla',w:'models',min:15,st:'wip',fid:'Analytical',s:{en:'GQA and MLA',zh:'GQA 和 MLA'},q:{en:'How do GQA and MLA shrink the KV cache?',zh:'GQA 和 MLA 怎么让 KV cache 变小？'}},
 {id:'sampling',w:'models',min:12,st:'planned',fid:'Live',s:{en:'Sampling',zh:'采样'},q:{en:'What do temperature and top-p actually do?',zh:'Temperature 和 top-p 到底做了什么？'}},
 {id:'tokenization-cost',w:'models',min:12,st:'planned',fid:'Live',s:{en:'Token cost',zh:'token 成本'},q:{en:'Chinese or English: which costs more tokens?',zh:'同样的意思，中文和英文谁更费 token？'}},
 {id:'rope',w:'models',min:12,st:'planned',fid:'Live',s:{en:'RoPE',zh:'RoPE'},q:{en:'How does a model know where each token is?',zh:'模型怎么知道每个 token 在哪个位置？'}},
 {id:'moe',w:'models',min:12,st:'planned',fid:'Analytical',s:{en:'MoE',zh:'MoE'},q:{en:'How can a 47B model run like a 13B one?',zh:'47B 的模型怎么跑得像 13B 一样快？'}},
 {id:'flash-attention',w:'models',min:15,st:'planned',fid:'Live',s:{en:'FlashAttention',zh:'FlashAttention'},q:{en:'How does FlashAttention compute exact attention with less memory traffic?',zh:'FlashAttention 怎么在结果不变的情况下少读写显存？'}},
 {id:'rag-failure',w:'apps',min:15,st:'planned',fid:'Replay',s:{en:'Why RAG fails',zh:'RAG 为什么答错'},q:{en:'Why does your RAG system answer wrong?',zh:'你的 RAG 为什么答错？'}},
 {id:'embeddings',w:'apps',min:15,st:'planned',fid:'Live',s:{en:'Embeddings',zh:'Embedding'},q:{en:'How does an embedding know two sentences mean the same?',zh:'embedding 怎么知道两句话意思相同？'}},
 {id:'vector-index',w:'apps',min:12,st:'planned',fid:'Simulated',s:{en:'Vector index',zh:'向量索引'},q:{en:'How do you find the nearest of 1B vectors in milliseconds?',zh:'怎么在几毫秒内从 10 亿个向量里找到最近的？'}},
 {id:'chunk-rerank',w:'apps',min:12,st:'planned',fid:'Simulated',s:{en:'Chunking & rerank',zh:'切块与 rerank'},q:{en:'Why does your retriever miss the right passage?',zh:'检索为什么漏掉了对的那段？'}},
 {id:'agent-reliability',w:'agents',min:10,st:'next',fid:'Simulated',s:{en:'Agent reliability',zh:'Agent 可靠性'},q:{en:'95% reliable per step, 20 steps: how often does the agent succeed?',zh:'每步 95% 可靠，20 步的 agent 还剩多少成功率？'}},
 {id:'agent-anatomy',w:'agents',min:12,st:'planned',fid:'Conceptual',s:{en:'Inside an agent task',zh:'Agent 任务拆解'},q:{en:'What actually happens inside one agent task?',zh:'Agent 一次任务里到底发生了什么？'}},
 {id:'prompt-injection',w:'agents',min:12,st:'planned',fid:'Simulated',s:{en:'Prompt injection',zh:'提示注入'},q:{en:'How can a web page hijack your agent?',zh:'一个网页怎么劫持你的 agent？'}},
 {id:'context-eng',w:'agents',min:12,st:'planned',fid:'Simulated',s:{en:'Context engineering',zh:'上下文工程'},q:{en:'What goes into an agent’s context, and what gets cut?',zh:'agent 的上下文里该放什么、删什么？'}},
 {id:'tool-scheduling',w:'agents',min:10,st:'planned',fid:'Simulated',s:{en:'Tool scheduling',zh:'工具调用调度'},q:{en:'Which tool calls can run in parallel?',zh:'哪些工具调用可以并行？'}},
 {id:'agent-capstone',w:'agents',min:20,st:'planned',fid:'Simulated',s:{en:'Agent capstone',zh:'Agent 综合挑战'},q:{en:'How do you build an agent loop that holds up, and measure it?',zh:'怎么搭一个靠得住的 agent 循环，并评测它？'}},
 {id:'eval-ci',w:'eval',min:10,st:'planned',fid:'Analytical',s:{en:'Benchmark noise',zh:'评测的噪声'},q:{en:'Is a 2-point benchmark gain real?',zh:'Benchmark 上高 2 分是真的吗？'}},
 {id:'pass-k',w:'eval',min:10,st:'planned',fid:'Simulated',s:{en:'pass@k',zh:'pass@k'},q:{en:'Can an agent with 60% pass@1 ship?',zh:'pass@1 = 60% 的 agent 能上线吗？'}},
 {id:'llm-judge',w:'eval',min:12,st:'planned',fid:'Simulated',s:{en:'LLM as judge',zh:'LLM 当裁判'},q:{en:'Can you trust an LLM as the judge?',zh:'LLM 当裁判靠得住吗？'}},
 {id:'contamination',w:'eval',min:10,st:'planned',fid:'Simulated',s:{en:'Contamination',zh:'数据污染'},q:{en:'Did the model see the test set?',zh:'模型是不是见过考题？'}},
 {id:'eval-capstone',w:'eval',min:20,st:'planned',fid:'Analytical',s:{en:'Eval capstone',zh:'评测综合挑战'},q:{en:'How do you evaluate an LLM assistant before and after release?',zh:'怎么给一个 LLM 助手设计上线前后的评测？'}},
 {id:'product-design',w:'product',min:25,st:'planned',fid:'Conceptual',s:{en:'Design a ChatGPT',zh:'设计 ChatGPT 类应用'},q:{en:'How do you design a ChatGPT-like app end to end?',zh:'怎么从头设计一个 ChatGPT 类应用？'}},
 {id:'tensors',w:'roots',min:10,st:'planned',fid:'Live',s:{en:'Tensors & shapes',zh:'张量与形状'},q:{en:'How do you turn a Python loop into one vector operation?',zh:'怎么把 Python 循环变成一次向量运算？'}},
 {id:'backprop',w:'roots',min:12,st:'planned',fid:'Live',s:{en:'Backprop',zh:'反向传播'},q:{en:'How does autograd get every gradient in one backward pass?',zh:'autograd 怎么一趟反向算出所有梯度？'}},
 {id:'softmax-numerics',w:'roots',min:10,st:'planned',fid:'Live',s:{en:'Softmax & numerics',zh:'softmax 与数值'},q:{en:'Why does a naive softmax return NaN?',zh:'朴素的 softmax 为什么会算出 NaN？'}},
 {id:'number-formats',w:'roots',min:10,st:'planned',fid:'Analytical',s:{en:'Number formats',zh:'数值格式'},q:{en:'Why does training use BF16 and not FP16?',zh:'训练为什么用 BF16，不用 FP16？'}},
 {id:'optimizers',w:'roots',min:12,st:'planned',fid:'Live',s:{en:'Optimizers',zh:'优化器'},q:{en:'What does Adam do that SGD does not?',zh:'Adam 比 SGD 多做了什么？'}},
 {id:'train-debug',w:'roots',min:15,st:'planned',fid:'Live',s:{en:'Training diagnosis',zh:'训练诊断'},q:{en:'Your loss will not go down: where is the bug?',zh:'loss 降不下去，bug 在哪？'}},
 {id:'roofline',w:'roots',min:12,st:'planned',fid:'Analytical',s:{en:'GPU & roofline',zh:'GPU 与 roofline'},q:{en:'Is your kernel limited by compute or by memory?',zh:'你的 kernel 卡在算力还是显存带宽？'}}
];
const LAB=Object.fromEntries(LABS.map(l=>[l.id,l]));
/* four learning paths, by what the learner wants to do (the owner, 2026-10-11: "按照真正的需求划分成四个"); the ids are kept from the six
   earlier paths (transformer, serving, research, apps) so saved routes and links still work; post and student were folded in */
const PATHS={
 transformer:{name:{en:'AI systems for beginners',zh:'AI 系统小白入门'},labs:['architecture','tokenization-cost','transformer','mha','sampling','kv-basics','kv-cache','agent-anatomy','eval-ci'],
   out:{en:'Explain how a model reads your question and writes its answer word by word, and why longer chats need more memory and cost more.',zh:'讲清大模型怎样读懂你的问题、一个字一个字写出回答，以及为什么对话越长越占内存、越贵。'}},
 serving:{name:{en:'Infra engineer',zh:'Infra 岗'},labs:['kv-basics','kv-cache','roofline','prefill-decode','paged-attention','continuous-batching','chunked-prefill','prefix-caching','serving-capstone','quantization','spec-decoding','pd-disagg','multi-gpu-serving','admission','training-memory','parallelism','model-parallel','checkpointing','train-capstone'],
   out:{en:'Estimate how many GPUs a model needs, how many people it can serve at once and how long each one waits, and what a large training run takes.',zh:'估算部署一个模型要几张显卡、能同时服务多少人、每人要等多久，以及训练一个大模型要多少卡和时间。'}},
 research:{name:{en:'Research (RS / RE)',zh:'研究岗（RS / RE）'},labs:['tensors','backprop','number-formats','softmax-numerics','optimizers','train-debug','roofline','architecture','tokenization-cost','mha','rope','transformer','kv-basics','sampling','gqa-mla','moe','flash-attention','training-memory','parallelism','model-parallel','scaling-law','data-pipeline','checkpointing','train-capstone','sft','lora','reward-model','ppo','dpo','grpo','rlvr','post-capstone','eval-ci','llm-judge','contamination','eval-capstone'],
   out:{en:'Work out from first principles what training a model costs, find the cause when training goes wrong, and compare the ways to make a model follow instructions.',zh:'从最基本的原理推算训练一个模型要多少算力和时间，训练出问题时自己找到原因，并比较让模型更听话的几种办法。'}},
 apps:{name:{en:'Agent and app engineer',zh:'Agent 应用岗'},labs:['architecture','kv-basics','sampling','eval-ci','pass-k','llm-judge','agent-anatomy','agent-reliability','context-eng','tool-scheduling','prompt-injection','agent-capstone','embeddings','vector-index','chunk-rerank','rag-failure','product-design'],
   out:{en:'Build AI assistants that look things up, use tools and cannot be steered by a web page, show how reliable they are, and design a ChatGPT-like product end to end.',zh:'做出能查资料、会用工具、不被网页带偏的 AI 助手，说清它有多可靠，并从头设计一个像 ChatGPT 那样的产品。'}}
};
/* who each path is for, with its abbreviation spelled out (R2.8): the chips' tooltips, the strip's options, the tiles' header */
const PATH_WHO={
 transformer:{en:'For anyone new to AI systems, students and career changers included: what happens inside a model between your question and its answer.',zh:'给刚接触 AI 系统的人，包括学生和转行的人：从你提问到模型回答，中间发生了什么。'},
 serving:{en:'Infra: AI infrastructure. For engineers who deploy models and keep them fast and affordable, and who run training on many GPUs.',zh:'Infra：AI 基础设施（infrastructure）。给负责部署模型、让服务又快又省，以及在很多张卡上跑训练的工程师。'},
 research:{en:'RS / RE: research scientist / research engineer. For researchers who build up from first principles: ML basics, the Transformer, pretraining, post-training and evaluation.',zh:'RS / RE：研究科学家 / 研究工程师（research scientist / research engineer）。给从第一性原理学起的研究岗：ML 基本功、Transformer、预训练、后训练和评测。'},
 apps:{en:'For engineers who build products on LLMs: agents, RAG (retrieval-augmented generation) and how to evaluate them.',zh:'给用大模型做产品的工程师：agent、RAG（检索增强生成，retrieval-augmented generation），以及怎么评测它们。'}
};
const pathWho=pid=>PATH_WHO[pid]?L(PATH_WHO[pid]):'';
/* read-only hooks for tests/map_geometry.py: each lab's branch, and the labs of a path */
window.__treeW=Object.fromEntries(LABS.map(l=>[l.id,l.w])); window.__treePath=pid=>PATHS[pid]?PATHS[pid].labs.slice():null;
/* the knowledge tree (batch 5, replacing the two loops): roots = ML basics, the trunk = how LLMs work, evaluation at the fork, then
   two limbs, build (pretraining and scaling, post-training) and use (inference serving, RAG, agents, product design). Each lab has one
   home (its branch, w); a path is a view that numbers the labs it walks through. */
const WORLDS={
 roots:{name:{en:'ML basics',zh:'ML 基本功'},limb:'root',desc:{en:'Tensors, gradients, numerics, optimizers and the GPU: what every model rests on.',zh:'张量、梯度、数值、优化器和 GPU：所有模型的地基。'}},
 models:{name:{en:'How LLMs work',zh:'LLM 怎么工作'},limb:'trunk',desc:{en:'The trunk every path passes through: the Transformer, tokens, attention, positions, generation and its variants.',zh:'所有路线都经过的主干：Transformer、分词、注意力、位置、生成和各种变体。'}},
 eval:{name:{en:'Evaluation',zh:'评测'},limb:'eval',desc:{en:'At the fork, shared by building and using models: it gates each release and grades the model in production.',zh:'在分叉处，两枝共用：为每次发布把关，给线上模型打分。'}},
 training:{name:{en:'Pretraining and scaling',zh:'预训练与扩展'},tn:{en:'Pretraining & scaling',zh:'预训练与扩展'},limb:'build',desc:{en:'The memory, parallelism, scaling laws, data and checkpoints of a training run.',zh:'一次训练的显存、并行、scaling law、数据和检查点。'}},
 post:{name:{en:'Post-training',zh:'后训练'},limb:'build',desc:{en:'SFT, LoRA, reward models, PPO, DPO, GRPO and RLVR after pretraining.',zh:'预训练之后的 SFT、LoRA、奖励模型、PPO、DPO、GRPO 和 RLVR。'}},
 inference:{name:{en:'Inference serving',zh:'推理服务'},limb:'use',desc:{en:'Serving: memory, batching, caching, decoding and capacity.',zh:'推理服务：显存、批处理、缓存、解码和容量。'}},
 apps:{name:{en:'RAG and retrieval',zh:'RAG 与检索'},limb:'use',desc:{en:'Embeddings, vector indexes, chunking, and why RAG answers wrong.',zh:'embedding、向量索引、切块，以及 RAG 为什么答错。'}},
 agents:{name:{en:'Agents',zh:'Agent'},limb:'use',desc:{en:'Tool loops, context, reliability and security.',zh:'工具循环、上下文、可靠性与安全。'}},
 product:{name:{en:'Product design',zh:'产品系统设计'},limb:'use',desc:{en:'The top of the “using models” side: design a ChatGPT-like app end to end.',zh:'“用模型”一枝的顶端：从头设计一个 ChatGPT 类应用。'}}
};
/* line icons drawn for this site: 24 × 24, stroke = currentColor */
const ICONS={
 roots:'<path d="M12 3v8"/><path d="M12 11c0 4-3.5 5.2-7 8.5M12 11c0 4 3.5 5.2 7 8.5M12 13v7.5"/><path d="M4 11h16"/>',
 product:'<rect x="3.5" y="4.5" width="17" height="12" rx="2.5"/><path d="M8.5 16.5l-2.5 4 5.2-4"/><path d="M7.5 9h9M7.5 12h5.5"/>',
 training:'<path d="M4 4v16h16"/><path d="M7 7.5c1.6 5 4.6 8.4 11 9.3"/>',
 post:'<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/>',
 models:'<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
 apps:'<path d="M4 5h11M4 9h7M4 13h4"/><circle cx="14.5" cy="14.5" r="4"/><path d="M17.5 17.5L21 21"/>',
 inference:'<path d="M13 2.5L5.5 13h6l-1 8.5 8-11h-6l.5-8z"/>',
 agents:'<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4.6V8"/><circle cx="12" cy="3.6" r="1.1"/><path d="M9.5 16.2h5"/><circle cx="9.6" cy="12.4" r="1.1" fill="currentColor" stroke="none"/><circle cx="14.4" cy="12.4" r="1.1" fill="currentColor" stroke="none"/>',
 eval:'<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4.2-5"/><path d="M6.6 13.6l1.2.7M12 9v1.4M17.4 13.6l-1.2.7"/>',
 path:'<path d="M5 19c5 0 3-7 7-7s2-7 7-7"/><circle cx="5" cy="19" r="1.7"/><circle cx="19" cy="5" r="1.7"/>'
};
const iconSVG=(k,px)=>`<svg class="wicon" viewBox="0 0 24 24" width="${px}" height="${px}" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[k]}</svg>`;
const DYN={}, TEASERS={};
/* ---------- what this learner did, kept on this device (the owner, round 3 #6: a reload or a later visit continues where they left off) ----------
   One localStorage entry (try/catch everywhere: storage can be blocked, full or empty, and the page must work without it).
   labs[id] = {t: last change (ms), mode, step, nSteps, maxStep, guided: reached the last guided part once, passed: met the challenge goal
   once (sticky), ans: per guided part, true once its question was answered or it had none, tok: the settings token when saved,
   snap: the lab's own snapshot (registerLab({snap, restore}), see CONTRACT.md), ex: the practice questions' state};
   last: the lab id changed most recently; route: the path the learner chose themselves (null: they browse by branch). */
const PROG=(()=>{const KEY='asp-progress-v1'; let db=null;
  // automated browsers (the test suite) start clean and keep nothing, so every test sees a first visit; tests/kit_resume.py opts in
  const off=()=>{try{return !!navigator.webdriver&&!window.__aspResume;}catch(e){return false;}};
  const fresh=()=>({v:1,labs:{},last:null,route:null});
  const load=()=>{if(db) return db; if(off()){db=fresh(); return db;} try{db=JSON.parse(localStorage.getItem(KEY)||'null');}catch(e){db=null;}
    if(!db||typeof db!=='object'||db.v!==1||!db.labs||typeof db.labs!=='object') db=fresh(); return db;};
  const save=()=>{if(off()) return; try{localStorage.setItem(KEY,JSON.stringify(db));}catch(e){}};
  return {
    get:id=>{const r=load().labs[id]; return r&&typeof r==='object'?r:null;},
    last:()=>{const d=load(); return d.last&&d.labs[d.last]&&LAB[d.last]?{id:d.last,...d.labs[d.last]}:null;},
    put:(id,o)=>{const d=load(); d.labs[id]={...(d.labs[id]||{}),...o,t:Date.now()}; d.last=id; save();},
    drop:id=>{const d=load(); delete d.labs[id]; if(d.last===id) d.last=null; save();},
    route:()=>{const r=load().route; return r&&PATHS[r]?r:null;},
    setRoute:pid=>{const d=load(); d.route=pid&&PATHS[pid]?pid:null; save();},
    any:()=>Object.keys(load().labs).length>0,
    clear:()=>{db=fresh(); save();}
  };})();
const TOKFN={};   // each lab's settings token, captured from SHELL.bindHeader (the kit stores it with the snapshot)
const RESTORED=new Set();   // labs whose saved state was put back in this page load (it is put back once, before the first mount)
// a lab is live when its code is here: the KV lab in this file, the others as registered modules
const labStatus=l=>(l.id==='kv-cache'||DYN[l.id])?'live':(l.st==='live'?'wip':l.st);
/* "next lab" and "previous lab" (round 3, the owner's items 4 and 5). A lab follows a learning path only when the learner chose that
   path: a path chip or the path menu on the home page, a route's start button (data-route-start), a link that names its path
   (data-path: the links along a chosen path, the start links on the home page) or the join button at the end of a branch (data-join).
   PROG keeps the choice (PROG.route()), so a reload keeps it; picking a branch on the map, or no path, clears it.
   - On the chosen path, when it has the lab, as before (Codex review 1 #2): the path's next open lab with its stop number, labs not
     built yet skipped and named; at the path's end, where it pauses, or its last stop and the path PATH_NEXT continues into.
   - Otherwise the lab follows its own branch, in the order of the branch's tiles: the branch's next and previous lab, with no path and
     no stop numbers. At the branch's last lab: a suggested next step, the next lab of the lab's home path (PATH_HOME, else the first
     path in PATH_PRI that has it), with no stop number, and a button that joins that path.
   No path is ever called finished, since nothing here proves it: at a path's last stop the footer says the learner has reached it and
   how many of its labs were opened on this device (PROG). */
// no path continues into another (PATH_NEXT, kept for the rule below): the beginners' path ends by pointing to the three job paths
const PATH_PRI=['transformer','serving','apps','research'], PATH_NEXT={};
// a lab's home path (its suggested next step) is the first path in PATH_PRI that has it, unless it is listed here: training-memory
// opens the training branch, so its next lab is ZeRO and FSDP (research path), not DPO (post-training path)
const PATH_HOME={'training-memory':'research'};
// false while the map draws itself the first time (it selects a branch then, which is not the learner's choice); start() sets it
let routeReady=false;
const isLive=id=>!!LAB[id]&&labStatus(LAB[id])==='live';
// the path the learner chose, when it has this lab; null: the lab follows its own branch
function pathOf(id){const r=PROG.route(); return r&&PATHS[r].labs.includes(id)?r:null;}
function homePath(id){const h=PATH_HOME[id]; if(h&&PATHS[h]&&PATHS[h].labs.includes(id)) return h;
  return PATH_PRI.find(p=>PATHS[p].labs.includes(id))||null;}
// the next open lab of path pid after position i (-1: from its start), and the labs not built yet passed on the way
function nextOnPath(pid,i,self){const labs=PATHS[pid].labs, skip=[];
  for(let k=i+1;k<labs.length;k++){if(labs[k]===self) continue; if(isLive(labs[k])) return {l:LAB[labs[k]],n:k+1,of:labs.length,skip}; skip.push(LAB[labs[k]]);}
  return {l:null,skip};}
// what lab id's next-lab links show on path pid
function nextFrom(pid,id){const a=nextOnPath(pid,PATHS[pid].labs.indexOf(id),id);
  if(a.l) return {pid,to:pid,...a};
  if(a.skip.length) return {pid,pause:a.skip};
  const p2=PATH_NEXT[pid]; if(p2){const b=nextOnPath(p2,PATHS[p2].labs.indexOf(id),id); if(b.l) return {pid,to:p2,done:true,...b};}
  return {pid,done:true};}
// the labs of branch w in the order of its tiles
const branchLabs=w=>LABS.filter(l=>l.w===w).map(l=>l.id);
// on its own branch: the next open lab; at the branch's end, the suggested next step (rec: the home path's next lab and that path)
function branchNext(id){const w=LAB[id].w, labs=branchLabs(w), skip=[];
  for(let k=labs.indexOf(id)+1;k<labs.length;k++){if(isLive(labs[k])) return {kind:'branch',w,l:LAB[labs[k]],n:k+1,of:labs.length,skip}; skip.push(LAB[labs[k]]);}
  const hp=homePath(id), x=hp?nextFrom(hp,id):null;
  return {kind:'branch',w,l:null,end:true,skip,rec:x&&x.l?{pid:x.to,l:x.l}:null};}
function branchPrev(id){const w=LAB[id].w, labs=branchLabs(w), skip=[];
  for(let k=labs.indexOf(id)-1;k>=0;k--){if(isLive(labs[k])) return {kind:'branch',w,l:LAB[labs[k]],n:k+1,of:labs.length,skip}; skip.push(LAB[labs[k]]);}
  return {kind:'branch',w,l:null,skip};}
function nextInfo(id){if(!LAB[id]) return null; const pid=pathOf(id); return pid?{kind:'route',...nextFrom(pid,id)}:branchNext(id);}
/* "previous lab", the mirror of the next lab (the owner, 2026-10-09: “除了下一个lab，也要带上‘上一个lab’的按钮”), on the same path
   (pathOf): the open lab before this one, skipping labs not built yet. At the path's first open lab, the lab a reader arrives from:
   the last open lab of a path that PATH_NEXT continues into this one, when its next link leads here (on that path); otherwise none.
   With no chosen path that has the lab: the open lab before it on its branch. */
function prevOnPath(pid,i,self){const labs=PATHS[pid].labs, skip=[];
  for(let k=i-1;k>=0;k--){if(labs[k]===self) continue; if(isLive(labs[k])) return {l:LAB[labs[k]],n:k+1,of:labs.length,skip}; skip.push(LAB[labs[k]]);}
  return {l:null,skip};}
function prevInfo(id){if(!LAB[id]) return null; const pid=pathOf(id); if(!pid) return branchPrev(id); const a=prevOnPath(pid,PATHS[pid].labs.indexOf(id),id);
  if(a.l) return {kind:'route',pid,to:pid,...a};
  for(const p1 of Object.keys(PATH_NEXT)){if(PATH_NEXT[p1]!==pid||!PATHS[p1]) continue;
    const labs=PATHS[p1].labs, e=labs.slice().reverse().find(isLive), x=e&&e!==id?nextFrom(p1,e):null;
    if(x&&x.l&&x.l.id===id&&x.to===pid) return {kind:'route',pid,to:p1,l:LAB[e],n:labs.indexOf(e)+1,of:labs.length,skip:a.skip,cont:true};}
  return {kind:'route',pid,l:null,skip:a.skip};}
const pName=pid=>L(PATHS[pid].name);
const labShort=l=>L(l.s||l.q);
const worldName=w=>WORLDS[w]?L(WORLDS[w].name):'';
// how many of path pid's labs were opened on this device (PROG keeps a record per lab opened; the lab on screen counts), or null when
// PROG has no record at all (a first visit, or storage blocked)
function openedOn(pid,id){if(!PROG.any()) return null; return PATHS[pid].labs.filter(x=>x===id||PROG.get(x)).length;}
const openedHTML=(pid,id)=>{const k=openedOn(pid,id), N=PATHS[pid].labs.length;
  return k==null?'':`<span class="nl-note">${tx(`On this device you have opened ${k} of this path’s ${N} labs.`,`这条路线的 ${N} 个 Lab，你在这台设备上打开过 ${k} 个。`)}</span>`;};
const lastStopHTML=pid=>`<span class="nl-done">${tx(`You have reached the last stop of the path “${pName(pid)}”.`,`你已到达“${pName(pid)}”路线的最后一站。`)}</span>`;
// the next path starts at stop n because its earlier stops are all on path p too: say so
const pickUp=(p2,n,p)=>n>1&&PATHS[p2].labs.slice(0,n-1).every(x=>PATHS[p].labs.includes(x));
// below a lab's rail card: the whole question as a link, or where to go when no later lab is open yet. The nav names its lab (data-nl),
// so that joining a path redraws it in place (navRefresh)
function nextLabHTML(id,cls){const x=nextInfo(id), c=`next-lab${cls?' '+cls:''}`, al=esc(t('nextK')); if(!x) return '';
  const nav=h=>`<nav class="${c}" aria-label="${al}" data-nl="${id}"${cls?` data-nlc="${esc(cls)}"`:''}>${h}</nav>`;
  const go=(l,pid)=>`<a class="nl-go" href="#${l.id}"${pid?` data-path="${pid}"`:''}>${esc(L(l.q))} <span aria-hidden="true">→</span></a>`;
  const skipH=sk=>sk.length?`<span class="nl-skip">${tx(`Skipping ${sk.map(labShort).join(', ')}: not built yet`,`跳过还没上线的：${sk.map(labShort).join('、')}`)}</span>`:'';
  const zl=sk=>{const s=sk.map(labShort).join('、'); return s+(/[一-鿿]$/.test(s)?'':' ');};   // no space before 还 when the list ends in a Chinese word
  if(x.kind==='branch'){const bn=worldName(x.w);
    // labs not in this build (only in partial test builds: all 60 are live) are passed over without a word (owner, round 3 #3: no build status on the page)
    if(x.l) return nav(`<span class="nl-k">${tx(`Next lab on this branch (“${bn}”, lab ${x.n} of ${x.of})`,`这一枝的下一个 Lab（“${bn}”第 ${x.n}/${x.of} 个）`)}</span>${go(x.l)}`);
    let h=`<span class="nl-done">${tx(`This is the last lab of the branch “${bn}”.`,`这是“${bn}”这一枝的最后一个 Lab。`)}</span>`;
    if(x.rec){const pn=pName(x.rec.pid);
      h+=`<span class="nl-k">${tx('Suggested next lab','推荐下一步')}</span>${go(x.rec.l)}`+
        `<span class="nl-note">${tx(`On the path “${pn}”, this lab comes right after the one you are in. If you join that path, the previous and next lab follow its order and say which stop each lab is. On the home page you can switch to another path, or to no path, at any time.`,
          `“${pn}”路线在你现在这个 Lab 之后安排的就是它。加入这条路线后，上一个、下一个 Lab 都按它的顺序走，并写明是第几站。在首页可以随时换一条路线，或者不选路线。`)}</span>`+
        `<button type="button" class="btn ghost sm nl-join" data-join="${x.rec.pid}">${tx(`Follow the path “${pn}”`,`按“${pn}”路线学`)}</button>`;}
    else h+=`<a class="nl-go" href="#map">${tx('Back to the tree, to pick another branch','回到知识树，选下一枝')} <span aria-hidden="true">→</span></a>`;
    return nav(h);}
  const route=pid=>`<a class="nl-go" href="#map~route:${pid}" data-path="${pid}" data-tipx="${esc(tx(`Back to the tree, with the ${PATHS[pid].labs.length} labs of the path “${pName(pid)}” numbered in order.`,`回到知识树，按顺序给“${pName(pid)}”路线的 ${PATHS[pid].labs.length} 个 Lab 编号。`))}">${tx('See this path on the tree','在知识树上看这条路线')} <span aria-hidden="true">→</span></a>`;
  if(x.pause) return nav(`<span class="nl-last">${tx(`The path “${pName(x.pid)}” pauses here: ${x.pause.map(labShort).join(', ')} ${x.pause.length>1?'are':'is'} not built yet.`,`“${pName(x.pid)}”路线暂时到此：${zl(x.pause)}还没上线。`)}</span>${route(x.pid)}`);
  // the end of the beginners' path: the three job paths are the next step (the owner, 2026-10-11: four paths, by what the learner wants to do)
  const jobs=pid=>pid==='transformer'?`<a class="nl-go" href="#map">${tx('Next, pick a job path: infra, research, or agents and apps','接下来选一条岗位路线：Infra 岗、研究岗或 Agent 应用岗')} <span aria-hidden="true">→</span></a>`:'';
  if(!x.l) return nav(lastStopHTML(x.pid)+openedHTML(x.pid,id)+jobs(x.pid)+route(x.pid));
  if(x.done) return nav(lastStopHTML(x.pid)+openedHTML(x.pid,id)+
    `<span class="nl-k">${tx(`You can continue with the path “${pName(x.to)}”, stop ${x.n} of ${x.of}`,`接下来可以继续：“${pName(x.to)}”路线第 ${x.n}/${x.of} 站`)}${pickUp(x.to,x.n,x.pid)?tx(` (its first ${x.n-1} stops are on the path “${pName(x.pid)}” too)`,`（它的前 ${x.n-1} 站也在“${pName(x.pid)}”路线上）`):''}</span>${go(x.l,x.to)}${skipH(x.skip)}`);
  return nav(`<span class="nl-k">${tx(`Next lab (stop ${x.n} of ${x.of} on the path “${pName(x.to)}”)`,`下一个 Lab（“${pName(x.to)}”路线第 ${x.n}/${x.of} 站）`)}</span>${go(x.l,x.to)}${skipH(x.skip)}`);}
// the branch in the breadcrumb goes back to the tree with that branch's tiles in view; on a chosen path, that leaves the path
const crumbWorldTip=id=>{const n=WORLDS[id]?L(WORLDS[id].name):'', r=PROG.route();
  return tx(`Back to the tree, showing the labs in ${n}.`,`回到知识树，看“${n}”这一枝的 Lab。`)+(r?tx(` This leaves the path “${pName(r)}”: the previous and next lab then follow each lab’s branch.`,`这会退出“${pName(r)}”路线：之后的上一个、下一个 Lab 按各自那一枝的顺序走。`):'');};
// on phones (≤ 760 px) the rail footer would sit between the task card and the chart, so a second copy goes after the bench
// (card, chart, settings) and CSS shows one or the other (site review R10 #17). Re-placed after every mount and language switch.
function placeEndFooter(){const root=view==='lab'?$('#view-lab'):(view==='dyn'?$('#view-dyn'):null); if(!root) return;
  $$('.next-lab-end',root).forEach(e=>e.remove());
  const id=view==='lab'?'kv-cache':(dyn&&dyn.id), bn=$('.bench',root); if(!id||!bn) return;
  bn.insertAdjacentHTML('afterend',`<div class="wrap next-lab-end">${nextLabHTML(id)}</div>`); drawerSync();}
// a link to another lab by its short name, with its whole question on hover
const qFull=l=>L(l.q)+(l.id==='gqa-mla'?tx(' (GQA: grouped-query attention; MLA: multi-head latent attention)','（GQA 即 grouped-query attention，MLA 即 multi-head latent attention）'):'');
function labLink(id,text){const l=LAB[id]; if(!l) return esc(text||id); return `<a href="#${id}" data-tipx="${esc(tx(`Lab: ${qFull(l)}`,`Lab：${qFull(l)}`))}">${esc(text||L(l.s||{en:id,zh:id}))}</a>`;}
// in the breadcrumb row: the short name only, the whole question and where the lab stands on hover. On a chosen path the links carry it
// (data-path); on a branch they carry no path, so following them chooses none
function nextCrumbHTML(id){const x=nextInfo(id); if(!x) return '';
  const a=(l,pid,k,tip)=>`<a class="crumb-next" href="#${l.id}"${pid?` data-path="${pid}"`:''} data-tipx="${esc(tip)}">${k}${tx(': ','：')}${esc(labShort(l))} <span aria-hidden="true">→</span></a>`;
  if(x.kind==='branch'){const bn=worldName(x.w);
    if(x.l) return a(x.l,null,t('nextK'),tx(`Next lab on this branch (“${bn}”, lab ${x.n} of ${x.of}): ${L(x.l.q)}`,`这一枝的下一个 Lab（“${bn}”第 ${x.n}/${x.of} 个）：${L(x.l.q)}`));
    if(x.rec) return a(x.rec.l,null,tx('Suggested next lab','推荐下一步'),tx(`This is the last lab of the branch “${bn}”. Suggested next lab: ${L(x.rec.l.q)} On the path “${pName(x.rec.pid)}”, it comes right after this lab.`,
      `这是“${bn}”这一枝的最后一个 Lab。推荐下一步：${L(x.rec.l.q)}“${pName(x.rec.pid)}”路线在这个 Lab 之后安排的就是它。`));
    return '';}
  if(!x.l) return '';
  if(x.done) return a(x.l,x.to,tx(`Continue with the path “${pName(x.to)}”`,`接下来可以继续“${pName(x.to)}”路线`),
    tx(`You have reached the last stop of the path “${pName(x.pid)}”. You can continue with the path “${pName(x.to)}”, stop ${x.n} of ${x.of}: ${L(x.l.q)}`,`你已到达“${pName(x.pid)}”路线的最后一站。接下来可以继续“${pName(x.to)}”路线第 ${x.n}/${x.of} 站：${L(x.l.q)}`));
  return a(x.l,x.to,t('nextK'),tx(`Next lab, stop ${x.n} of ${x.of} of the path “${pName(x.to)}”: ${L(x.l.q)}`,`下一个 Lab，“${pName(x.to)}”路线第 ${x.n}/${x.of} 站：${L(x.l.q)}`));}
// the previous lab, the same way; the path of a lab on the path before is named
function prevCrumbHTML(id){const x=prevInfo(id); if(!x||!x.l) return '';
  const a=(pid,k,tip)=>`<a class="crumb-prev" href="#${x.l.id}"${pid?` data-path="${pid}"`:''} data-tipx="${esc(tip)}"><span aria-hidden="true">←</span> ${k}${tx(': ','：')}${esc(labShort(x.l))}</a>`;
  if(x.kind==='branch'){const bn=worldName(x.w); return a(null,t('prevK'),tx(`Previous lab on this branch (“${bn}”, lab ${x.n} of ${x.of}): ${L(x.l.q)}`,`这一枝的上一个 Lab（“${bn}”第 ${x.n}/${x.of} 个）：${L(x.l.q)}`));}
  const k=x.cont?tx(`Previous lab (${pName(x.to)})`,`上一个 Lab（${pName(x.to)}）`):t('prevK');
  const tip=tx(`Previous lab, stop ${x.n} of ${x.of} of the path “${pName(x.to)}”: ${L(x.l.q)}`,`上一个 Lab，“${pName(x.to)}”路线第 ${x.n}/${x.of} 站：${L(x.l.q)}`)+(x.cont?tx(` The path “${pName(x.pid)}” continues from there.`,`“${pName(x.pid)}”路线从那里接着走。`):'');
  return a(x.to,k,tip);}
// the pair at the right end of the breadcrumb row (CSS: one row on wide screens, its own line under the crumbs on phones)
const crumbPairHTML=id=>prevCrumbHTML(id)+nextCrumbHTML(id);
const crumbNavHTML=id=>{const h=crumbPairHTML(id); return h?`<span class="crumb-pn">${h}</span>`:'';};
// joining a path at a branch's end redraws the open lab's links in place: its footers, the breadcrumb pair and the branch link's hover;
// focus goes to the footer's new next link
function navRefresh(from){const id=view==='lab'?'kv-cache':(view==='dyn'&&dyn?dyn.id:null); if(!id) return;
  const end=!!(from&&from.closest('.next-lab-end'));
  $$('nav.next-lab[data-nl]').forEach(n=>{n.outerHTML=nextLabHTML(n.dataset.nl,n.dataset.nlc||'');});
  if(view==='lab') kvCrumbs();
  else {const c=$('#view-dyn .labhead .crumbs'); if(c){const o=$('.crumb-pn',c); if(o) o.remove(); c.insertAdjacentHTML('beforeend',crumbNavHTML(id));}}
  $$('.labhead .crumbs a[href^="#map~"]').forEach(a=>{const w=a.getAttribute('href').slice(5); if(WORLDS[w]) a.dataset.tipx=crumbWorldTip(w);});
  const nv=$$('nav.next-lab[data-nl]').find(n=>!!n.closest('.next-lab-end')===end&&n.offsetParent), g=nv&&$('.nl-go',nv); if(g) g.focus({preventScroll:true});}
// a click that chooses a path, seen before the link is followed or a lab's own handler runs (capture phase): the join button at a
// branch's end, a route's start button (data-route-start="<path>"), a link that names its path (data-path)
function routeClick(e){const el=e.target&&e.target.closest?e.target:null; if(!el) return;
  const j=el.closest('[data-join]'); if(j&&PATHS[j.dataset.join]){e.stopPropagation(); PROG.setRoute(j.dataset.join); navRefresh(j); return;}
  const s=el.closest('[data-route-start]'); if(s&&PATHS[s.dataset.routeStart]){PROG.setRoute(s.dataset.routeStart); return;}
  const a=el.closest('a[data-path]'); if(a&&PATHS[a.dataset.path]) PROG.setRoute(a.dataset.path);}
/* the setup link (Codex D02): on a site of its own (the page is the top window) the whole address with the setup's fragment, which opens
   as it is; inside the Claude Artifact (an iframe, whose address the reader cannot open) the fragment alone, to add to the page's link.
   The label says which, and that the link carries the setup only, not the predictions or the progress (those stay on this device). */
const topWin=()=>{try{return window.top===window.self;}catch(e){return false;}};
const shareText=frag=>topWin()?location.href.split('#')[0]+frag:frag;
const shareKey=()=>topWin()?'shareLabel':'shareLabelFrag';
let sel={type:'world',id:'inference'}, lastWorld='inference';
/* where a lab sits on the knowledge tree: its branch, then the name on its box (owner round 3 #2) */
const treeAt=l=>`${L(WORLDS[l.w].name)} › ${labShort(l)}`;
/* a box's tooltip: the lab's minutes, and what this browser remembers of it */
const labMeta=l=>{const m=labMark(l.id), p=PRIMERS[l.id]; return minsTxt(l.min)+(p?' · '+tx(`primer first, ${p.min} min`,`有知识梳理，${p.min} 分钟`):'')+(m?' · '+m.t:'');};
/* a small open book marks a lab that has a primer (知识梳理), on its tree box and on its tile (the owner, 2026-10-13) */
const BOOK_PATH='M0,-3.6C-1.6,-4.7 -3.7,-4.8 -5.4,-4.1V3.9C-3.7,3.2 -1.6,3.3 0,4.4C1.6,3.3 3.7,3.2 5.4,3.9V-4.1C3.7,-4.8 1.6,-4.7 0,-3.6ZM0,-3.6V4.4';
const BOOK_ICON=`<svg class="bk-ico" viewBox="-6.5 -6 13 12" width="14" height="13" aria-hidden="true"><path d="${BOOK_PATH}" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>`;

/* ---------- the knowledge tree ----------
   Drawn in a 1200 × 700 box, so that at 1440 × 900 the whole tree is on screen at full size. The trunk's labs sit on both sides of the trunk, the roots' in two rows under the ground; each branch of
   the limbs is a column of labs (its lane), from its first lab (bottom) to its last; product design crowns the use limb. Every lab's
   box is filled in its limb's colour. */
const TREE={W:1200,H:700,pad:24,ground:592,fork:404,x:600,bw:128,bh:22,step:28};   // pad: room left of x = 0 for the axis's names
/* the crown (the owner, 2026-10-11: "知识树观感不太平衡"): the tallest branch, inference serving, stands next to the fork and the shorter ones
   step down towards both edges, so the crown rises in the middle instead of at the right edge; each limb is centred on its side */
const LANE={training:{x:206,y0:358},post:{x:400,y0:340},eval:{x:600,y0:262},inference:{x:754,y0:388},agents:{x:908,y0:354},apps:{x:1062,y0:342}};
/* KV cache memory sits on the trunk right after “Why a KV cache” (the owner, 2026-10-12: it applies what that lab teaches, and the serving
   branch now starts with a lab that has a primer); the trunk's labs step up by TREE.step so its 11 boxes stay below the fork */
const LANE_ORDER={
 roots:['tensors','backprop','softmax-numerics','number-formats','optimizers','train-debug','roofline'],
 models:['architecture','tokenization-cost','mha','rope','transformer','kv-basics','kv-cache','sampling','gqa-mla','moe','flash-attention'],
 eval:['eval-ci','pass-k','llm-judge','contamination','eval-capstone'],
 training:['training-memory','parallelism','model-parallel','scaling-law','data-pipeline','checkpointing','train-capstone'],
 post:['sft','lora','reward-model','ppo','dpo','grpo','rlvr','post-capstone'],
 inference:['prefill-decode','paged-attention','continuous-batching','chunked-prefill','prefix-caching','serving-capstone','quantization','spec-decoding','pd-disagg','multi-gpu-serving','admission'],
 apps:['embeddings','vector-index','chunk-rerank','rag-failure'],
 agents:['agent-anatomy','agent-reliability','context-eng','tool-scheduling','prompt-injection','agent-capstone'],
 product:['product-design']};
const ROOT_POS=[[290,624],[470,624],[380,674],[600,674],[730,624],[910,624],[820,674]];
const PRODUCT_POS=[985,124];
const LIMBNAME={build:{en:'BUILD',zh:'造模型'},use:{en:'USE',zh:'用模型'}};
const LIMBSUB={build:{en:'how a model is made',zh:'模型是怎么做出来的'},use:{en:'how a model is put to work',zh:'模型怎么被用起来'}};
/* where each lab sits, and where each branch's name goes (its top lab's y) */
function treeLayout(){const T=TREE, top={};
  Object.entries(LANE_ORDER).forEach(([w,ids])=>ids.forEach((id,i)=>{const l=LAB[id]; if(!l) return;
    if(w==='roots') l.pos=ROOT_POS[i].slice();
    else if(w==='models'){const row=i>>1, left=i%2===0; l.pos=[left?T.x-100:T.x+100,left?576-row*T.step:562-row*T.step];}
    else if(w==='product') l.pos=PRODUCT_POS.slice();
    else {const ln=LANE[w]; l.pos=[ln.x+(i%2?5:-5),ln.y0-i*T.step]; top[w]=l.pos[1];}}));
  return top;}
/* the trunk's half width at height y: 44 at the ground, 22 at the fork */
let treeGrown=false;
const trunkHW=y=>22+22*(y-TREE.fork)/(TREE.ground-TREE.fork);
function drawMap(){
  const T=TREE, top=treeLayout();
  let h=`<g class="tr-axis" aria-hidden="true"><line x1="40" y1="30" x2="40" y2="690"/>`+
    [[230,tx('Branches','枝')],[T.fork,tx('Fork','分叉')],[498,tx('Trunk','主干')],[646,tx('Roots','根')]].map(([y,t])=>`<line class="tk" x1="36" y1="${y}" x2="44" y2="${y}"/><text x="32" y="${y+4}" text-anchor="end">${esc(t)}</text>`).join('')+`</g>`;
  h+=`<line class="tr-ground" x1="56" y1="${T.ground}" x2="985" y2="${T.ground}"/>`;
  /* roots: from the trunk's base to each root lab; the second row passes between the first row's boxes */
  h+=`<g class="tr-roots" aria-hidden="true">`+LANE_ORDER.roots.map((id,i)=>{const [x,y]=LAB[id].pos, sx=T.x+(x<T.x?-16:(x>T.x?16:0)), top_=y-T.bh/2, low=y>650;
    return low?`<path pathLength="1" d="M${sx},${T.ground} C${sx+(x-sx)*.35},${T.ground+5} ${x},${T.ground+6} ${x},${T.ground+20} L${x},${top_}"/>`
             :`<path pathLength="1" d="M${sx},${T.ground} C${sx+(x-sx)*.45},${T.ground+3} ${x},${T.ground+2} ${x},${top_}"/>`;}).join('')+`</g>`;
  /* the trunk, then its twigs to the trunk's labs */
  h+=`<path class="tr-trunk" d="M556,${T.ground} C566,540 574,470 578,${T.fork} L622,${T.fork} C626,470 634,540 644,${T.ground} Z"/>`;
  h+=`<g class="tr-twigs" aria-hidden="true">`+LANE_ORDER.models.map(id=>{const [x,y]=LAB[id].pos, s=x<T.x?-1:1; return `<line x1="${T.x+s*(trunkHW(y)-2)}" y1="${y}" x2="${x-s*T.bw/2}" y2="${y}"/>`;}).join('')+`</g>`;
  /* the limbs, from the fork to each lane, then up the lanes; product design hangs from the tops of RAG and Agents (their stems stop at
     the branch's name and go on above it, so no line crosses a name: threadStems() places them once the names are measured) */
  const base=w=>LANE[w].y0+T.bh/2+1, stem=w=>`M${LANE[w].x},${w==='eval'?T.fork:base(w)} L${LANE[w].x},${top[w]}`, P=PRODUCT_POS;
  /* a pair of small leaves where a lane ends */
  /* a limb: from the fork out to its first lane (the one nearest the trunk), then on to each further lane; every lane's stem starts
     where the limb reaches it */
  const limb=(k,fx,fy,ws)=>{const J=[LANE[ws[0]].x,390];
    return `<path class="lb ${k} main" pathLength="1" d="M${fx},${fy} C${fx+(J[0]-fx)*.3},${fy-10} ${fx+(J[0]-fx)*.7},${J[1]} ${J[0]},${J[1]}"/>`+
      ws.map((w,i)=>{const x=LANE[w].x, b=base(w); if(!i) return `<path class="lb ${k}" pathLength="1" d="M${J[0]},${J[1]} L${x},${b}"/>`;
        const x0=LANE[ws[i-1]].x; return `<path class="lb ${k}" pathLength="1" d="M${x0},${J[1]} C${x0+(x-x0)*.6},${J[1]} ${x},${J[1]-4} ${x},${b}"/>`;}).join('');};
  const leaves=(x,y,k)=>`<path class="lf ${k}" d="M${x},${y} c-9,-3 -14,-10 -13,-17 c8,1 13,8 13,17z"/><path class="lf ${k}" d="M${x},${y} c9,-3 14,-10 13,-17 c-8,1 -13,8 -13,17z"/>`;
  h+=`<g class="tr-limbs" aria-hidden="true">`+
    limb('build',582,408,['post','training'])+limb('use',616,406,['inference','agents','apps'])+
    ['training','post'].map(w=>`<path class="st build" pathLength="1" d="${stem(w)}"/>`).join('')+`<path class="st eval" pathLength="1" d="${stem('eval')}"/>`+['apps','agents','inference'].map(w=>`<path class="st use" pathLength="1" d="${stem(w)}"/>`).join('')+
    ['apps','agents'].map(w=>`<path class="st use" data-thread="${w}" data-part="stub" pathLength="1" d=""/><path class="st use" data-thread="${w}" data-part="arc" pathLength="1" d=""/>`).join('')+
    ['training','post'].map(w=>leaves(LANE[w].x,top[w]-T.bh/2-1,'build')).join('')+leaves(600,top.eval-T.bh/2-1,'eval')+leaves(P[0],P[1]-T.bh/2-1,'use')+leaves(LANE.inference.x,top.inference-T.bh/2-1,'use')+`</g>`;
  /* the two limbs' names, large and faint */
  [['build',300],['use',900]].forEach(([k,x])=>{h+=`<text class="cap-big tr-cap ${k}" x="${x}" y="452" text-anchor="middle">${esc(L(LIMBNAME[k]))}</text><text class="cap-sub" x="${x}" y="474" text-anchor="middle">${esc(L(LIMBSUB[k]))}</text>`;});
  /* each branch's name: above its lane (clear of the leaves), left of the trunk, left of the roots */
  const up=w=>top[w]-48;
  const NAMEPOS={roots:[214,666,'end'],models:[410,510,'end'],eval:[600,up('eval'),'middle'],training:[LANE.training.x,up('training'),'middle'],post:[LANE.post.x,up('post'),'middle'],
    inference:[LANE.inference.x,up('inference'),'middle'],apps:[LANE.apps.x,up('apps'),'middle'],agents:[LANE.agents.x,up('agents'),'middle'],product:[P[0],P[1]-48,'middle']};
  Object.entries(WORLDS).forEach(([id,w])=>{const [x,y,an]=NAMEPOS[id], sub=labsTxt(worldCounts(id).n);
    h+=`<g class="world tw" data-world="${id}" data-limb="${w.limb}" tabindex="0" role="button" aria-label="${esc(L(w.name))}, ${esc(sub)}" style="--d:${id==='roots'?0.2:(id==='models'?0.6:1.2)}s"><rect class="tw-bg" rx="6"/>`+
       `<text class="wname" x="${x}" y="${y}" text-anchor="${an}">${esc(L(w.tn||w.name))}</text><text class="wsub" x="${x}" y="${y+14}" text-anchor="${an}">${esc(sub)}</text></g>`;});
  /* the labs */
  /* when the tree grows: the roots' labs first, then the trunk's, then each lane from the bottom up */
  const delay=l=>{const i=(LANE_ORDER[l.w]||[]).indexOf(l.id); return l.w==='roots'?0.05*i:(l.w==='models'?0.3+0.05*i:0.85+0.04*i);};
  LABS.forEach(l=>{if(!l.pos) return; const [x,y]=l.pos, st=labStatus(l), limb=WORLDS[l.w].limb, nm=L(l.s||l.q);
    const pr=PRIMERS[l.id];
    h+=`<g class="bead tnode ${st}${pr?' haspr':''}" data-lab="${l.id}" data-limb="${limb}" tabindex="0" role="button" aria-label="${esc(L(l.q))}, ${esc(minsTxt(l.min))}${pr?esc(tx(', with a primer','，有知识梳理')):''}" transform="translate(${x} ${y})" style="--d:${delay(l).toFixed(2)}s">`+
       `<rect class="tbx" x="${-T.bw/2}" y="${-T.bh/2}" width="${T.bw}" height="${T.bh}" rx="6"/>${pr?`<path class="tbk" d="${BOOK_PATH}" transform="translate(${-T.bw/2+11} 0.4)"/>`:''}<text class="tlb"${pr?' x="7"':''} y="4" text-anchor="middle">${esc(nm)}</text></g>`;});
  h+=barChartSVG();
  h+='<g id="routeLayer"></g>';
  const svg=$('#mapSvg'); svg.setAttribute('viewBox',`${-T.pad} 0 ${T.W+T.pad} ${T.H}`); svg.innerHTML=h; fitTreeText(svg); threadStems(svg,top); alignBars(svg);
  /* the tree grows once, when it first comes into view (it sits below the hero); reduced motion or no observer: drawn at once */
  if(!treeGrown){treeGrown=true;
    if(!reduceMotion()&&'IntersectionObserver' in window){svg.classList.add('grow');
      const io=new IntersectionObserver(es=>{if(!es.some(e=>e.isIntersecting)) return; io.disconnect(); svg.classList.add('go'); setTimeout(()=>svg.classList.remove('grow','go'),3000);},{threshold:LIST_MQ.matches?0.1:0.25});
      io.observe(svg);}}
  renderMapLegend();
  svg.setAttribute('aria-label',tx('Knowledge tree. Roots: ML basics. Trunk: how LLMs work. At the fork: evaluation. Building models: pretraining and scaling, post-training. Using models: inference serving, RAG and retrieval, agents, product design.',
    '知识树。根：ML 基本功。主干：LLM 怎么工作。分叉处：评测。“造模型”一枝：预训练与扩展、后训练。“用模型”一枝：推理服务、RAG 与检索、Agent、产品系统设计。'));
}
/* a long name is set smaller, down to 9.5 px, then squeezed to fit its box; each branch's name gets a background that shows it is chosen */
function fitTreeText(svg){const max0=TREE.bw-12;
  $$('.tlb',svg).forEach(t=>{const max=t.closest('.haspr')?max0-14:max0; let fs=11; while(t.getComputedTextLength()>max&&fs>9.5){fs-=0.5; t.style.fontSize=fs+'px';}
    if(t.getComputedTextLength()>max){t.setAttribute('textLength',max); t.setAttribute('lengthAdjust','spacingAndGlyphs');}});
  $$('.world',svg).forEach(g=>{const bb=[...g.querySelectorAll('text')].map(t=>t.getBBox()), r=g.querySelector('.tw-bg');
    const x0=Math.min(...bb.map(b=>b.x)), x1=Math.max(...bb.map(b=>b.x+b.width)), y0=Math.min(...bb.map(b=>b.y)), y1=Math.max(...bb.map(b=>b.y+b.height));
    r.setAttribute('x',x0-6); r.setAttribute('y',y0-3); r.setAttribute('width',x1-x0+12); r.setAttribute('height',y1-y0+6);});}
/* names measured while the home was hidden (a language switch inside a lab) or before the web fonts arrived: measure again */
function refitTree(){const svg=$('#mapSvg'); if(!svg||!svg.querySelector('.world')||!svg.getBoundingClientRect().width) return; fitTreeText(svg); threadStems(svg,treeLayout()); alignBars(svg); if(!treePanned) panTreeHome();}
/* narrow screens (the owner, 2026-10-13: the branch list was too simple): the whole tree at full size in a box the reader drags to look
   around, like a map. It starts on the trunk, with the roots at its bottom edge; it is placed once, and again after the window becomes narrow */
let treePanned=false;
function panTreeHome(){const sc=$('#map .map-scroll'), svg=$('#mapSvg'); if(!sc||!svg||!LIST_MQ.matches) return; const w=svg.getBoundingClientRect().width; if(!w) return;
  sc.scrollLeft=Math.max(0,(TREE.x+TREE.pad)*w/(TREE.W+TREE.pad)-sc.clientWidth/2); sc.scrollTop=sc.scrollHeight; treePanned=true;}
/* the bar chart's title starts where its longest row name starts */
function alignBars(svg){const t=svg.querySelector('.tr-bars .tb-t'), xs=[...svg.querySelectorAll('.tr-bars .tb-n')].map(e=>e.getBBox()).filter(b=>b.width>0).map(b=>b.x);
  if(t&&xs.length){t.setAttribute('x',Math.min(...xs).toFixed(1)); t.setAttribute('text-anchor','start');}}
/* product design's two stems: from the top lab of RAG and of Agents up to just under the branch's name, then from just above the
   name into the side of the product design box. A line never crosses a name (review tree-causal-A H2). */
function threadStems(svg,top){const T=TREE, P=PRODUCT_POS, gap=4;
  ['apps','agents'].forEach(w=>{const g=svg.querySelector(`.world[data-world="${w}"] .tw-bg`); if(!g) return;
    const x=LANE[w].x, y0=top[w]-T.bh/2, nb=+g.getAttribute('y'), nt=nb+(+g.getAttribute('height'));
    const stub=svg.querySelector(`[data-thread="${w}"][data-part="stub"]`), arc=svg.querySelector(`[data-thread="${w}"][data-part="arc"]`);
    if(stub){if(y0-(nt+gap)>2) stub.setAttribute('d',`M${x},${y0} L${x},${(nt+gap).toFixed(1)}`); else stub.remove();}
    /* into the box's near side, at mid-height: left side for RAG, right side for Agents */
    const ys=nb-gap, ex=P[0]+(x<P[0]?-1:1)*T.bw/2, ey=P[1], up=Math.max(0,ys-ey);
    if(arc) arc.setAttribute('d',`M${x},${ys.toFixed(1)} C${x},${(ys-up*0.62).toFixed(1)} ${(x+(ex-x)*0.15).toFixed(1)},${ey} ${ex},${ey}`);});}
/* bottom right: how many labs each branch has */
const BARS=['inference','agents','apps','product','eval','post','training','models','roots'];
/* each row is named as the branch is on the tree (review H6) */
function barChartSVG(){const x0=1100, u=5, y0=586, rh=12.5;
  let h=`<g class="tr-bars" aria-hidden="true"><text class="tb-t" x="${x0-6}" y="574" text-anchor="end">${esc(tx('Labs per branch','每一枝的 Lab 数'))}</text>`;
  BARS.forEach((w,k)=>{const n=LABS.filter(l=>l.w===w).length, y=y0+k*rh;
    h+=`<g data-limb="${WORLDS[w].limb}"><text class="tb-n" x="${x0-6}" y="${y+4}" text-anchor="end">${esc(L(WORLDS[w].tn||WORLDS[w].name))}</text>`+
       `<rect class="tb-bar" x="${x0}" y="${y-4.5}" width="${n*u}" height="9" rx="1.5"/><text class="tb-c" x="${x0+n*u+5}" y="${y+4}">${n}</text></g>`;});
  return h+`</g>`;}
function markMap(){
  const path=sel.type==='path'?PATHS[sel.id]:null, inPath=path?new Set(path.labs):null;
  $$('#mapSvg .world').forEach(g=>{g.classList.toggle('sel',!path&&g.dataset.world===sel.id);
    g.classList.toggle('dim',!!path&&!path.labs.some(id=>LAB[id].w===g.dataset.world));});
  $$('#mapSvg .bead').forEach(g=>g.classList.toggle('dim',!!inPath&&!inPath.has(g.dataset.lab)));
  const layer=$('#routeLayer'); if(!path){layer.innerHTML='';return;}
  layer.innerHTML=path.labs.map((id,i)=>{const p=LAB[id].pos; return `<g class="tnum" transform="translate(${(p[0]-TREE.bw/2-5).toFixed(1)} ${p[1].toFixed(1)})"><circle class="bnum-bg" r="8.5"/><text class="bnum" text-anchor="middle" y="3.6">${i+1}</text></g>`;}).join('');
}
/* what a tile's mini experiment is: how its numbers are produced (not the planned lab's own fidelity) */
const TZF={
 formula:{en:'Formula',zh:'公式计算',d:{en:'This mini experiment’s numbers come from closed-form formulas with published specs or stated assumptions; nothing is simulated.',zh:'这个小实验的数由闭式公式算出，代入公开规格或写明的假设，不做模拟。'}},
 toy:{en:'Toy run',zh:'玩具实跑',d:{en:'In this mini experiment your browser really runs the algorithm, on a small made-up input.',zh:'这个小实验里，浏览器真的在运行这个算法，只是输入是小规模的虚构数据。'}},
 illus:{en:'Illustrative',zh:'示意',d:{en:'This mini experiment is a hand-made example that shows the mechanism.',zh:'这个小实验是为演示机制手工构造的例子。'}}
};
const TZKIND={'kv-cache':'formula','paged-attention':'formula','continuous-batching':'toy','prefill-decode':'formula','chunked-prefill':'formula','prefix-caching':'formula','serving-capstone':'toy',
 'quantization':'toy','spec-decoding':'formula','training-memory':'formula','parallelism':'formula','sft':'toy','lora':'formula','reward-model':'toy','ppo':'toy','dpo':'toy','grpo':'toy','rlvr':'toy','post-capstone':'formula','sampling':'toy',
 'tokenization-cost':'toy','rag-failure':'illus','agent-reliability':'formula','agent-anatomy':'illus','prompt-injection':'illus','eval-ci':'formula','pass-k':'formula'};
/* a tile's notes live in the teaser's .tz-note elements (hidden on the tile); the footer button shows their current text */
function noteText(n){const c=n.cloneNode(true); $$('[data-tipx]',c).forEach(x=>x.replaceWith(document.createTextNode(' '+x.getAttribute('data-tipx').replace(/<[^>]+>/g,'')+' ')));
  let s=c.textContent.replace(/\s+/g,' ').trim(); const own=n.getAttribute('data-tipx'); if(own) s+=(s?' ':'')+own; return s;}
const TIPFN={crumbTree:()=>`<p>${esc(heroEntry&&HERO&&lastTile===HERO.id?tx('Back to the top of the home page, where you opened this lab.','回到首页顶部，也就是你打开这个 Lab 的地方。'):tx('Back to the tree, to this lab’s tile.','回到知识树上这个 Lab 的方块。'))}</p>`,
 };
function worldCounts(id){return {n:LABS.filter(l=>l.w===id).length};}
/* a branch's lab count in HTML: it never breaks inside itself (H13) */
const worldSubHTML=id=>`<span class="nw">${esc(labsTxt(worldCounts(id).n))}</span>`;
/* the same count as a sentence, for tooltips (R6.15) */
const worldTip=id=>{const n=L(WORLDS[id].name); return tx(`${n}: ${labsTxt(worldCounts(id).n)}.`,`${n}：${labsTxt(worldCounts(id).n)}。`);};
const worldClickTip=id=>{const c=worldCounts(id), n=L(WORLDS[id].name);
  return c.n===1?tx(`Click to see the one lab of ${n} below.`,`点击在下方看“${n}”这一枝的 1 个 Lab。`):tx(`Click to see the ${c.n} labs of ${n} below.`,`点击在下方看“${n}”这一枝的 ${c.n} 个 Lab。`);};
/* what inside a tile keeps its own click: the mini experiment's controls and the hints (a tap on a hint shows it on a phone) */
const TILE_OWN='a[href],button,input,select,textarea,label,summary,[role=button],[role=slider],[role=radio],[role=radiogroup],[role=tab],[role=switch],[role=checkbox],[role=option],.seg-ctl,[tabindex="0"][data-tipx],[tabindex="0"][data-tip]';
function tileHTML(l,num){
  const st=labStatus(l), f=TZF[(TEASERS[l.id]&&TEASERS[l.id].fid)||TZKIND[l.id]];
  return `<article class="tile${st==='live'?' is-live':''}" id="tile-${l.id}" data-lab="${l.id}" data-limb="${WORLDS[l.w].limb}">`+
    `<div class="tile-meta">${num?`<span class="tile-num mono">${num}</span>`:''}${PRIMERS[l.id]?`<span class="tile-pr" tabindex="0" role="img" aria-label="${esc(tx('Has a primer','有知识梳理'))}" data-tipx="${esc(tx(`This lab starts with a primer (${PRIMERS[l.id].min} min): the ideas and formulas it rests on, before the hands-on parts.`,`这个 Lab 前有知识梳理（${PRIMERS[l.id].min} 分钟）：先讲它用到的概念和公式，再动手。`))}">${BOOK_ICON}</span>`:''}<span class="tile-at">${esc(tx('Tree: ','知识树：')+treeAt(l))}</span><span class="mono muted">${minsTxt(l.min)}</span>`+
    (f?`<span class="chip tz-kind" tabindex="0" data-tipx="${esc(L(f.d))}">${esc(L(f))}</span>`:'')+`<span class="tile-mark" data-mark="${l.id}">${markHTML(l.id)}</span></div>`+
    `<h3 class="tile-q">${st==='live'?`<a href="#${l.id}">${esc(L(l.q))}</a>`:esc(L(l.q))}</h3>`+
    `<div class="tile-tz" data-teaser="${l.id}"></div></article>`;
}
function renderTiles(){
  const head=$('#tilesHead'), grid=$('#tiles'); if(!head) return;
  let ids, num=false;
  if(sel.type==='path'){const p=PATHS[sel.id], labs=p.labs.map(id=>LAB[id]), mins=labs.reduce((a,l)=>a+l.min,0), first=labs[0]; ids=p.labs; num=true;
    // the path's start, as on its entry above the tree (owner round 3 #1): an explicit choice of this path
    const go=`<p class="th-go"><a class="btn sm" href="#${first.id}" data-path="${sel.id}" data-route-start="${sel.id}">${esc(tx(`Stop 1: ${labShort(first)}`,`进入第一站：${labShort(first)}`))} <span aria-hidden="true">→</span></a></p>`;
    delete head.dataset.limb;
    head.innerHTML=`<div class="th-icon">${iconSVG('path',26)}</div><div class="th-text"><div class="th-meta">${tx('Learning path','学习路线')} · ${tx(`${labs.length} labs, about ${mins} min`,`${labs.length} 个 Lab，约 ${mins} 分钟`)}</div><h3 class="th-name">${esc(L(p.name))}</h3><p class="th-who">${esc(pathWho(sel.id))}</p><p class="th-desc">${tx('Afterwards you can: ','学完你能：')}${esc(L((PATH_INFO[sel.id]||p).can||p.out))}</p>${go}</div>`;}
  else {const w=WORLDS[sel.id]; ids=LABS.filter(l=>l.w===sel.id).map(l=>l.id); head.dataset.limb=w.limb;
    head.innerHTML=`<div class="th-icon">${iconSVG(sel.id,26)}</div><div class="th-text"><div class="th-meta">${tx('Branch','枝')} · ${worldSubHTML(sel.id)}</div><h3 class="th-name">${esc(L(w.name))}</h3><p class="th-desc">${esc(L(w.desc))}</p></div>`;}
  grid.innerHTML=ids.map((id,i)=>tileHTML(LAB[id],num?i+1:0)).join('');
  $$('[data-teaser]',grid).forEach(el=>{const tz=TEASERS[el.dataset.teaser];
    if(!tz){el.classList.add('tz-none'); el.innerHTML=`<p class="tz-read muted">${tx('The mini experiment for this lab is coming.','这个 Lab 的小实验还在做。')}</p>`;return;}
    try{tz.render(el);}catch(err){console.error('teaser '+el.dataset.teaser,err); el.innerHTML='';}});
}
/* the map legend: what a box's colour means */
function renderMapLegend(){const el=$('#mapLegend'); if(!el) return;
  const CL=[['trunk',tx('ML basics and how LLMs work','基础和 LLM 怎么工作')],['build',tx('building models','造模型')],['use',tx('using models','用模型')],['eval',tx('evaluation','评测')]];
  el.innerHTML=`<div class="lg-row"><span class="lg-lead">${tx('Each box is one lab. Its colour shows the part of the tree:','每个方块是一个 Lab，颜色表示它在树的哪一部分：')}</span>`+CL.map(([k,x])=>`<span class="lg-c"><i class="tsw c ${k}"></i>${esc(x)}</span>`).join('')+`<span class="lg-c lg-bk">${BOOK_ICON}${esc(tx('has a primer','有知识梳理'))}</span></div>`;}
/* phones: the branches as a list, from the top of the tree down to its roots, in place of the drawn tree */
function renderDomGrid(){const el=$('#domGrid'); if(!el) return;
  const G=[[tx('Using models: how a model is put to work','用模型：模型怎么被用起来'),['inference','apps','agents','product']],[tx('Building models: how a model is made','造模型：模型是怎么做出来的'),['training','post']],
    [tx('Evaluation: where the two meet','评测：两枝交汇的地方'),['eval']],[tx('Trunk: how LLMs work','主干：LLM 怎么工作'),['models']],[tx('Roots: ML basics','根：ML 基本功'),['roots']]];
  el.innerHTML=`<p class="dg-lead">${tx('The screen is too narrow to draw the whole knowledge tree, so its branches are listed here from the top of the tree down: the two big branches, “Using models” and “Building models”, then evaluation, the trunk and the roots.','屏幕太窄，画不下整棵知识树，这里按树的样子从上往下列出它的各枝：最上面是“用模型”和“造模型”两大枝，往下是评测、主干和根。')}</p>`+G.map(([name,ids])=>`<div class="dg-group" data-limb="${WORLDS[ids[0]].limb}"><div class="dg-h">${esc(name)}</div><div class="dg-items">${ids.map(id=>`<button type="button" class="dg-btn" data-world="${id}" data-limb="${WORLDS[id].limb}" aria-pressed="false"><span class="dg-ico">${iconSVG(id,20)}</span><span class="dg-t"><b>${esc(L(WORLDS[id].name))}</b><span>${worldSubHTML(id)}</span></span></button>`).join('')}</div></div>`).join('');}
/* above the tiles: every branch one click away while you read the tiles; it sticks under the top bar */
function renderStrip(){const el=$('#domStrip'); if(!el) return;
  el.innerHTML=`<div class="ds-worlds" role="group" aria-label="${esc(tx('Branches','各枝'))}">${Object.keys(WORLDS).map(id=>`<button type="button" class="ds-btn" data-world="${id}" data-limb="${WORLDS[id].limb}" aria-pressed="false" aria-label="${esc(worldTip(id))}" data-tipx="${esc(worldTip(id))}">${iconSVG(id,18)}<span>${esc(L(WORLDS[id].name))}</span></button>`).join('')}</div>`+
   `<label class="ds-path"><span>${tx('Path','路线')}</span><select id="pathSel"><option value="" title="${esc(tx('No path: the labs of the branch you pick.','不选路线：列出你选的那一枝的 Lab。'))}">${tx('No path','不选路线')}</option>${Object.entries(PATHS).map(([k,p])=>`<option value="${k}" title="${esc(pathWho(k))}">${esc(L(p.name))}</option>`).join('')}</select></label>`;
  fitStrip();}
/* the strip shows every branch's name when they all fit in one row; otherwise icons with tooltips, and the chosen one keeps its name
   (review H4: at 1440 in English three names hid off the edge, the chosen one among them) */
function fitStrip(){const el=$('#domStrip'), w=el&&$('.ds-worlds',el); if(!w||!el.offsetParent) return;
  el.classList.remove('ds-icons'); if(w.scrollWidth>w.clientWidth+1) el.classList.add('ds-icons'); keepStripSel();}
function keepStripSel(){const w=$('#domStrip .ds-worlds'), b=w&&$('.ds-btn[aria-pressed="true"]',w); if(!b||!w.offsetParent) return;
  const a=w.getBoundingClientRect(), r=b.getBoundingClientRect();
  if(r.left<a.left) w.scrollLeft-=a.left-r.left+6; else if(r.right>a.right) w.scrollLeft+=r.right-a.right+6;}
function syncSel(){
  const pw=sel.type==='path'?new Set(PATHS[sel.id].labs.map(id=>LAB[id].w)):null;
  $$('.dg-btn,.ds-btn').forEach(b=>{b.setAttribute('aria-pressed',String(sel.type==='world'&&b.dataset.world===sel.id)); b.classList.toggle('dim',!!pw&&!pw.has(b.dataset.world));});
  const ps=$('#pathSel'); if(ps){ps.value=sel.type==='path'?sel.id:''; ps.title=sel.type==='path'?pathWho(sel.id):'';}
  keepStripSel();
  $$('.pathchip').forEach(c=>c.setAttribute('aria-pressed',String(c.dataset.path===(sel.type==='path'?sel.id:''))));
  $$('.pentry').forEach(e=>e.classList.toggle('sel',sel.type==='path'&&e.dataset.pid===sel.id));
  const pa=$('#mapBar .pc-all'); if(pa) pa.hidden=sel.type!=='path';   // the way back to no path, once a path is chosen
  // the line right above the tree says what the tree shows now: a path's numbered order, or the branches
  const mn=$('#mapNow'); if(mn){const ph=LIST_MQ.matches, p=sel.type==='path'?PATHS[sel.id]:null;
    mn.textContent=p?tx(`The numbers on the tree are the order of the path “${L(p.name)}”, ${labsTxt(p.labs.length)}.`,`树上的编号是“${L(p.name)}”路线的顺序，一共 ${labsTxt(p.labs.length)}。`)+(ph?tx(' Drag the tree to see the rest.','拖动知识树可以看其余部分。'):'')
      :(ph?tx('Drag the tree to see the rest of it. Tap a branch’s name to list its labs below; tap a box to see that lab.','拖动知识树可以看其余部分。点枝的名字，下方列出这一枝的 Lab；点一个方块，看这个 Lab 的介绍。'):tx('No path is marked on the tree. Click a branch’s name to list its labs below; click a box to see that lab.','树上没有标路线。点枝的名字，下方列出这一枝的 Lab；点一个方块，看这个 Lab 的介绍。'));}
  const ph=$('#pathHint'); if(ph){ph.hidden=sel.type!=='path'; if(sel.type==='path'){const n=PATHS[sel.id].labs.length, phone=LIST_MQ.matches; ph.textContent=phone?tx(`↓ Its ${n} labs, in order, below`,`↓ 下方按顺序列出它的 ${n} 个 Lab`):tx(`↓ Its ${n} labs, in order, below the tree`,`↓ 知识树下方按顺序列出它的 ${n} 个 Lab`);}}
}
function revealTiles(){const h=$('#tilesHead'), s=$('#domStrip'), bar=$('.topbar'); if(!h) return;
  const top=(bar?bar.offsetHeight:58)+(s&&s.offsetParent?s.offsetHeight:0)+8, r=h.getBoundingClientRect();
  if(r.top>=top-2&&r.top<=window.innerHeight-200) return;
  quietTips(); window.scrollTo({top:Math.max(0,window.scrollY+r.top-top),behavior:reduceMotion()?'auto':'smooth'});}
/* the tree is drawn at 900 px or wider; up to 1004 px of window (a 15 px scroll bar included) it would not fit its panel, so there, as on
   phones, it is drawn at full size in a box the reader drags (CSS uses the same width; the branch list it replaced is no longer shown) */
const LIST_MQ=window.matchMedia?matchMedia('(max-width:1004px)'):{matches:false};
/* wide screens: the line above the tree right under the top bar and the whole tree below it, so a path's numbered route is in view from its
   first stop (review tree-causal-A H1, R11.6, R11.10). Nothing moves when both are already on screen. Phones draw no tree: false. */
function revealMap(smooth){const pc=$('#mapBar'), cv=$('#map .map-canvas'), svg=$('#mapSvg'); if(!pc||!cv||!svg||!cv.offsetParent||LIST_MQ.matches) return false;
  const bar=$('.topbar'), top=(bar?bar.offsetHeight:58)+6, a=pc.getBoundingClientRect(), z=svg.getBoundingClientRect();
  if(a.top>=top-1&&z.bottom<=window.innerHeight) return true;
  quietTips(); window.scrollTo({top:Math.max(0,window.scrollY+a.top-top),behavior:smooth&&!reduceMotion()?'smooth':'auto'}); return true;}
function selectWorld(id,o){sel={type:'world',id}; lastWorld=id; if(routeReady) PROG.setRoute(null); markMap(); renderTiles(); syncSel(); if(o&&o.scroll) revealTiles();}
function selectPath(pid,o){if(!pid){selectWorld(lastWorld,o);return;} sel={type:'path',id:pid}; if(routeReady) PROG.setRoute(pid); markMap(); renderTiles(); syncSel(); if(o&&o.scroll) revealTiles();}
function focusTile(id){const el=$('#tile-'+id); if(!el) return; quietTips(); el.scrollIntoView({block:'center',behavior:reduceMotion()?'auto':'smooth'}); flashEl(el);}
/* the six learning paths above the tree (owner round 3 #1; they replace the "Start with a path" cards and the path chips). Each entry
   says who the path suits, what you can do after it, what to know first, its labs and minutes, and has a start button (an explicit
   choice of the path: data-route-start). A click anywhere else on the entry (one button stretched over it, 2026-10-12) numbers the path's
   labs on the tree and lists them below in that order (phones, where the tree sits in a box to drag: lists them below). */
const PATH_INFO={
 transformer:{who:{en:'Anyone new to AI systems, including students and people changing careers.',zh:'刚接触 AI 系统的人，包括学生和想转行的人。'},
   pre:{en:'No AI background. Knowing how matrix multiplication works is enough.',zh:'不需要 AI 背景，知道矩阵乘法怎么算就够了。'}},
 serving:{who:{en:'Engineers who deploy models or run training clusters, and software engineers moving into AI infrastructure.',zh:'做模型部署或训练集群的工程师，以及想转做 AI 基础设施的软件工程师。'},
   pre:{en:'You can write code and know roughly what a model is made of. If not, take “AI systems for beginners” first.',zh:'会写程序，大致知道模型由哪些部件组成。不熟的话，先学“AI 系统小白入门”。'}},
 research:{who:{en:'Research scientists and research engineers (RS / RE) who build up from first principles.',zh:'从第一性原理学起的研究科学家和研究工程师（RS / RE）。'},
   pre:{en:'You can write Python and know basic linear algebra and calculus.',zh:'会写 Python，学过线性代数和微积分的基础。'}},
 apps:{who:{en:'Software engineers who build products on large language models.',zh:'用大模型做产品的软件工程师。'},
   pre:{en:'You can write code. You need no ML background and no experience training models.',zh:'会写程序就行。不需要 ML 背景，也不需要训练过模型。'}}
};
/* "Afterwards you can" is the path's out: plain words, no technical terms (the owner, 2026-10-11: no config, MHA, DPO) */
Object.keys(PATH_INFO).forEach(k=>{PATH_INFO[k].can=PATHS[k].out;});
const pathMins=pid=>PATHS[pid].labs.reduce((a,id)=>a+LAB[id].min,0);
function pathEntryHTML(pid){const p=PATHS[pid], I=PATH_INFO[pid]||{}, n=p.labs.length, first=LAB[p.labs[0]], on=sel.type==='path'&&sel.id===pid;
  const f=k=>I[k]?`<p class="pe-f pe-${k}"><b class="pe-k">${PE_K[k]()}</b>${esc(L(I[k]))}</p>`:'';
  // the whole entry is the button that shows the path on the tree (the owner, 2026-10-12: no separate “See the learning path” button);
  // it is stretched over the entry, under the start link
  const hit=tx(`Show the path “${L(p.name)}” on the tree: its ${n} labs numbered in order`,`在知识树上查看“${L(p.name)}”：给它的 ${n} 个 Lab 按顺序编号`);
  return `<li class="pentry${on?' sel':''}" data-pid="${pid}"><button class="pathchip pe-hit" type="button" data-path="${pid}" aria-pressed="${on}" aria-label="${esc(hit)}"></button><div class="pe-head"><h3 class="pe-name">${esc(L(p.name))}</h3><span class="pe-size">${esc(tx(`${n} labs · about ${pathMins(pid)} min`,`${n} 个 Lab · 约 ${pathMins(pid)} 分钟`))}</span><span class="pe-prog" data-pprog="${pid}" hidden></span></div>`+
    f('who')+f('can')+f('pre')+
    `<div class="pe-go"><a class="btn sm" href="#${first.id}" data-path="${pid}" data-route-start="${pid}" data-tipx="${esc(tx(`Open stop 1 of this path: ${L(first.q)}`,`打开这条路线的第 1 站：${L(first.q)}`))}">${esc(tx(`Stop 1: ${labShort(first)}`,`进入第一站：${labShort(first)}`))} <span aria-hidden="true">→</span></a></div></li>`;}
/* each field's name: a column heading on wide screens (where the per-entry names are read only by screen readers), on each entry on narrow ones */
const PE_K={who:()=>tx('For: ','适合：'),can:()=>tx('Afterwards you can: ','学完你能：'),pre:()=>tx('Before you start: ','前置基础：')};
function renderPaths(){const el=$('#pathList'), pa=$('#mapBar .pc-all'); if(!el) return;
  if(pa) pa.dataset.tipx=tx('Takes the numbers off the tree. Click a branch’s name on the tree to list its labs below.','去掉树上的路线编号。点树上一枝的名字，下方列出这一枝的 Lab。');
  el.innerHTML=`<p class="pl-lead">${tx('Each path puts its labs in order: start at stop 1. Click a path to see its order on the tree. A long path can take several sittings, and this browser on this device remembers where you were.','每条路线都把 Lab 排好了顺序，从第 1 站开始。点一条路线，知识树上会按顺序标出它的 Lab。长的路线可以分几次学，这台设备上的浏览器会记住你学到了哪里。')}</p>`+
    `<div class="pl-cols" aria-hidden="true"><span>${tx('Path','路线')}</span><span>${tx('Who it suits','适合谁')}</span><span>${tx('Afterwards you can','学完你能')}</span><span>${tx('Before you start','前置基础')}</span><span></span></div>`+
    `<ol class="plist">${Object.keys(PATHS).map(pathEntryHTML).join('')}</ol>`+
    `<p class="pl-rec" id="progRec" hidden><span>${tx('Your learning record is kept only in this browser on this device.','你的学习记录只保存在这台设备上的浏览器里。')}</span> <button class="linkbtn" type="button" id="progClear">${tx('Clear the learning record on this device','清除这台设备上的学习记录')}</button></p>`;}
/* what this browser on this device remembers of a lab (PROG; owner round 3 #6): the strongest of challenge goal met, guided parts all
   seen, opened */
function labMark(id){const r=PROG.get(id); if(!r) return null;
  if(r.mode==='lesson'){   // a lesson: the end check all right, read to the end, or opened (reading to the end is not called learning it)
    if(r.passed) return {k:'passed',t:tx('✓ Check all right','✓ 检查题全对'),d:tx('In this browser on this device you answered all of this lesson’s check questions right.','你在这台设备上的浏览器里，把这一课的检查题全部答对了。')};
    if(r.guided) return {k:'guided',t:tx('Read to the end','读到了末尾'),d:tx('In this browser on this device you read this lesson to its end.','你在这台设备上的浏览器里，把这一课读到了末尾。')};
    return {k:'opened',t:tx('Opened','打开过'),d:tx('You opened this lesson in this browser on this device and have not read it to the end yet.','你在这台设备上的浏览器里打开过这一课，还没读到末尾。')};}
  if(r.passed) return {k:'passed',t:tx('✓ Challenge goal met','✓ 挑战已达标'),d:tx('In this browser on this device you submitted this lab’s challenge, and it met the goal.','你在这台设备上的浏览器里交过这个 Lab 的挑战题，达到了目标。')};
  if(r.guided) return {k:'guided',t:tx('Guided parts done','引导看完了'),d:tx('In this browser on this device you reached the last part of this lab’s guided mode.','你在这台设备上的浏览器里走到了这个 Lab 分步引导的最后一部分。')};
  return {k:'opened',t:tx('Opened','打开过'),d:tx('You opened this lab in this browser on this device. You have not reached the end of its guided parts yet.','你在这台设备上的浏览器里打开过这个 Lab，还没走到分步引导的最后一部分。')};}
const markHTML=id=>{const m=labMark(id); return m?`<span class="chip tm tm-${m.k}" data-tipx="${esc(m.d)}">${esc(m.t)}</span>`:'';};
/* where the learner was in a lab: guided part k of n, the scenarios, or the challenge */
function resumeWhere(r){if(r.mode==='lesson'){const a=r.snap&&typeof r.snap.at==='string'?r.snap.at:'', n=r.nSteps|0;
    if(/^s\d+$/.test(a)) return tx(`Section ${a.slice(1)} of ${n}`,`读到第 ${a.slice(1)}/${n} 段`);
    if(a==='quiz') return tx('The check at the end','末尾的检查题'); if(a&&a!=='head') return tx('The end of the lesson','本课末尾'); return tx('The beginning','开头');}
  if(r.mode==='challenge') return r.passed?tx('Challenge, goal met','挑战题，已达标'):t('modeChallenge');
  if(r.mode==='sandbox') return t('modeSandbox');
  const n=r.nSteps|0, k=Math.min((r.step|0)+1,n); return n>0?tx(`Guided, part ${k} of ${n}`,`分步引导，第 ${k}/${n} 部分`):t('modeGuided');}
/* the learner's progress on the home page: "Continue" in the hero, a mark on each tile, how many labs of each path were opened, and the
   way to clear it all. It runs with the home page and each time the home page shows again (route) */
let heroCtaEl=null;   // kept here: the lab's redraw replaces the card's content, footer and button included
const heroCtaNode=()=>heroCtaEl||(heroCtaEl=$('#heroCta'));
function renderProgress(){const last=PROG.last(), a=$('#heroResume'), cta=heroCtaNode();
  if(a){if(last){const l=LAB[last.id], w=resumeWhere(last);
      a.dataset.resume=l.id; a.href='#'+l.id; a.hidden=false;
      a.innerHTML=`<span class="rs-t">${esc(tx(`Continue: ${labShort(l)}`,`继续上次：${labShort(l)}`))} <span aria-hidden="true">→</span></span><span class="rs-at">${esc(w)}</span>`;
      a.dataset.tipx=tx(`Back to the lab you opened last: ${L(l.q)} You were at: ${w}.`,`回到上次打开的 Lab：${L(l.q)} 上次停在：${w}。`);}
    else {a.hidden=true; a.removeAttribute('data-resume'); a.removeAttribute('data-tipx'); a.innerHTML=''; a.href='#top';}}
  // a returning learner's main button is "Continue"; the hero's own lab then becomes a second button
  if(cta) cta.classList.toggle('ghost',!!(a&&!a.hidden));
  $$('[data-pprog]').forEach(e=>{const k=PATHS[e.dataset.pprog].labs.filter(id=>PROG.get(id)).length; e.textContent=k?tx(`You opened ${k} of them`,`你打开过其中 ${k} 个`):''; e.hidden=!k;});
  $$('#tiles [data-mark]').forEach(e=>{e.innerHTML=markHTML(e.dataset.mark);});
  const rec=$('#progRec'); if(rec) rec.hidden=!PROG.any();}
function clearProgress(){if(!window.confirm(tx('Clear the learning record kept in this browser on this device? “Continue” and the marks on the labs will disappear.','清除这台设备上的浏览器记住的学习记录吗？“继续上次”和各个 Lab 上的标记都会消失。'))) return;
  PROG.clear(); if(sel.type==='path') selectWorld(lastWorld); renderProgress();}   // the chosen route is part of the record
function renderHome(){drawMap(); renderDomGrid(); renderStrip(); renderPaths(); markMap(); renderTiles(); syncSel(); renderProgress();}
let lastTile=null;
/* one tooltip element for the whole page: above the target, or below it when the top bar is in the way */
const tip=$('#tip'); let tipFor=null;
let tipHTML='';
function showTipHTML(el,html,wide){
  tipHTML=html; tip.innerHTML=html; tip.style.left='0px'; tip.style.top='0px'; tip.classList.remove('below'); tip.classList.toggle('wide',!!wide); tip.hidden=false;
  const r=el.getBoundingClientRect(), half=tip.offsetWidth/2, below=r.top-tip.offsetHeight-14<64;
  tip.style.left=Math.max(half+8,Math.min(window.innerWidth-half-8,r.left+r.width/2))+'px';
  tip.classList.toggle('below',below); tip.style.top=(below?r.bottom:r.top)+'px';
  if(tipDesc!==el){tipUndesc(); if(el.setAttribute&&!el.hasAttribute('aria-describedby')){el.setAttribute('aria-describedby','tip'); tipDesc=el;}}}
function showTip(el,l){const act=TEASERS[l.id]?tx('Click to see its mini experiment below.','点击在下方看它的小实验。'):tx('Click to see its tile below. Its mini experiment is coming.','点击在下方看它的方块，小实验还在做。');
  showTipHTML(el,`<div>${esc(L(l.q))}</div><div class="m">${labMeta(l)}</div><div class="m">${act}</div>`);}
function hideTip(){tip.hidden=true; tipFor=null; tipUndesc();}
/* kit fixes (round K): while the tooltip shows, its trigger points at it with aria-describedby (only a trigger that had none of its own) */
let tipDesc=null;
function tipUndesc(){if(tipDesc){tipDesc.removeAttribute('aria-describedby'); tipDesc=null;}}
/* kit fixes (round K): every hover-tip trigger can be reached by keyboard (R1.6; home review #24). A trigger ([data-tip], [data-tipx],
   [data-tipfn], .t, [data-jump]) that is not a native control gets tabindex="0" (marked data-ktab) and, when it has no text of its own
   (a swatch, a bar, an SVG shape), an aria-label with its tip (marked data-klabel, kept in step with data-tipx); a kit-made jump link also
   gets role="button" and follows Enter and Space. Focus shows the tip, blur and Escape hide it; mouse hover is unchanged. Left alone: the
   map (its boxes are focusable already), anything inside an aria-hidden or inert box (a focusable element there would be a trap for
   screen readers), and a trigger inside another control or focusable element (that one is the keyboard stop). A MutationObserver
   (kitObserve) runs this on everything a lab, a teaser or the home page renders. */
const TIPSEL='[data-tip],[data-tipx],[data-tipfn],.t,[data-jump]', NATIVE='a[href],button,input,select,textarea,summary,iframe,[contenteditable]';
function tipPlain(el){const x=document.createElement('template'); x.innerHTML=el.dataset.tipx!=null?escMath(el.dataset.tipx):(el.dataset.tip?t(el.dataset.tip):'');
  return x.content.textContent.replace(/\s+/g,' ').trim();}
function tipReach(el){if(!el||el.nodeType!==1||!el.matches(TIPSEL)) return;
  if(el.hasAttribute('data-klabel')){const s=tipPlain(el); if(s) el.setAttribute('aria-label',s);}
  if(el.hasAttribute('tabindex')||el.matches(NATIVE)||el.closest('#mapSvg,#tip,[aria-hidden="true"],[inert]')) return;
  const up=el.parentElement&&el.parentElement.closest('a[href],button,summary,label,[role="button"],[role="link"],[tabindex]:not([data-ktab])'); if(up) return;
  el.setAttribute('tabindex','0'); el.setAttribute('data-ktab','');
  if(el.dataset.jump&&!el.hasAttribute('role')) el.setAttribute('role','button');
  if(!el.hasAttribute('aria-label')&&!el.hasAttribute('aria-labelledby')&&!el.textContent.trim()){const s=tipPlain(el); if(s){el.setAttribute('aria-label',s); el.setAttribute('data-klabel','');}}}
function tipReachAll(root){if(!root||root.nodeType!==1) return; tipReach(root); root.querySelectorAll(TIPSEL).forEach(tipReach);}
/* Escape hides the tooltip; the trigger it came from does not bring it back under a resting pointer until the pointer leaves it or
   it is focused again */
let tipEsc=null;
/* tooltips follow the user's pointer: after any scroll they stay quiet until the pointer really moves or taps */
let tipQuiet=false, ptrX=-1, ptrY=-1;
function quietTips(){tipQuiet=true; hideTip();}
function mapInit(){
  renderHome();
  const svg=$('#mapSvg');
  svg.addEventListener('click',e=>{const b=e.target.closest('.bead'); if(b){const l=LAB[b.dataset.lab]; if(sel.type!=='path'||!PATHS[sel.id].labs.includes(l.id)) selectWorld(l.w); focusTile(l.id); return;}
    const w=e.target.closest('.world'); if(w) selectWorld(w.dataset.world,{scroll:true});});
  svg.addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return; const g=e.target.closest('.bead,.world'); if(!g)return; e.preventDefault(); g.dispatchEvent(new MouseEvent('click',{bubbles:true}));});
  // a lab's box: its question, status and what a click does (H9); a branch's name: what a click shows (H10)
  let svgTipFor=null;
  const svgTip=g=>{svgTipFor=g; if(g.classList.contains('bead')) showTip(g,LAB[g.dataset.lab]); else showTipHTML(g,`<div>${esc(worldClickTip(g.dataset.world))}</div>`);};
  svg.addEventListener('mouseover',e=>{if(tipQuiet) return; const g=e.target.closest('.bead,.world'); if(g&&g!==tipEsc) svgTip(g);});
  // after a scroll the tooltips are quiet until the pointer moves; a pointer that moved onto a box and moves on inside it gets its tooltip
  svg.addEventListener('mousemove',e=>{if(e.clientX!==ptrX||e.clientY!==ptrY){ptrX=e.clientX; ptrY=e.clientY; tipQuiet=false;}
    if(tipQuiet) return; const g=e.target.closest('.bead,.world'); if(g&&g!==tipEsc&&(tip.hidden||svgTipFor!==g)) svgTip(g);},{passive:true});
  svg.addEventListener('mouseout',e=>{const g=e.target.closest('.bead,.world'); if(g&&!g.contains(e.relatedTarget)) hideTip();});
  svg.addEventListener('focusin',e=>{const g=e.target.closest('.bead,.world'); if(g) svgTip(g);});
  svg.addEventListener('focusout',hideTip);
  window.addEventListener('scroll',()=>{tipQuiet=true;
    // a focused element keeps its tooltip only while it is on screen below the top bar (site review R10 #23)
    const on=tipFor&&tipFor===document.activeElement&&tipHTML&&(()=>{const r=tipFor.getBoundingClientRect(), bar=$('.topbar'); return r.bottom>(bar?bar.offsetHeight:58)&&r.top<window.innerHeight;})();
    if(on) showTipHTML(tipFor,tipHTML); else hideTip();},{passive:true});
  // a path's "Show its order" and "No path, by branch": on wide screens the line above the tree and the whole tree come into view with
  // the numbered route (H1); phones, where only part of the tree is in view, go to the numbered tiles
  // the path entries are drawn again on a language switch, so their clicks are caught on the section
  $('#map').addEventListener('click',e=>{const c=e.target.closest('.pathchip'); if(c){selectPath(c.dataset.path); if(!revealMap(true)) revealTiles(); return;}
    if(e.target.closest('#progClear')) clearProgress();});
  // "Explore the tree" or the Tree link while the address already ends in #map: no hashchange, so place the tree here
  document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a[href="#map"]'); if(!a||view!=='home'||location.hash!=='#map') return; if(revealMap(true)) e.preventDefault();});
  if(document.fonts&&document.fonts.ready) document.fonts.ready.then(()=>{if(view==='home'){refitTree(); fitStrip();}});
  $('#pathHint').addEventListener('click',revealTiles);
  // the tree's box on narrow screens: a finger scrolls it natively; a mouse drags it too, and a drag is not a click on the box it ends on
  {const sc=$('#map .map-scroll'); let d=null;
   if(sc){sc.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'||e.button!==0||(sc.scrollWidth<=sc.clientWidth+1&&sc.scrollHeight<=sc.clientHeight+1)) return; d={x:e.clientX,y:e.clientY,l:sc.scrollLeft,t:sc.scrollTop,moved:false};});
     window.addEventListener('pointermove',e=>{if(!d) return; const dx=e.clientX-d.x, dy=e.clientY-d.y; if(!d.moved&&Math.hypot(dx,dy)<5) return;
       if(!d.moved){d.moved=true; sc.classList.add('dragging'); hideTip();} sc.scrollLeft=d.l-dx; sc.scrollTop=d.t-dy;});
     window.addEventListener('pointerup',()=>{if(!d) return; const m=d.moved; d=null; sc.classList.remove('dragging'); if(!m) return;
       const stop=ev=>{ev.stopPropagation(); ev.preventDefault();}; sc.addEventListener('click',stop,{capture:true,once:true}); setTimeout(()=>sc.removeEventListener('click',stop,{capture:true}),0);});
     if(LIST_MQ.addEventListener) LIST_MQ.addEventListener('change',()=>{treePanned=false;});}}
  $('#domGrid').addEventListener('click',e=>{const b=e.target.closest('.dg-btn'); if(b) selectWorld(b.dataset.world,{scroll:true});});
  $('#domStrip').addEventListener('click',e=>{const b=e.target.closest('.ds-btn'); if(b) selectWorld(b.dataset.world,{scroll:true});});
  $('#domStrip').addEventListener('change',e=>{if(e.target.id==='pathSel') selectPath(e.target.value,{scroll:true});});
  // remember which tile opened a lab, so coming back lands on it
  $('#tiles').addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]'), t=e.target.closest('.tile'); if(a&&t) lastTile=t.dataset.lab;});
  /* the whole tile is one button that opens its lab (the owner, 2026-10-13: “卡片本身就是一个按钮”), like the path entries above the tree.
     The mini experiment's own controls and the hints keep their clicks; selecting text opens nothing. The title stays a link for the
     keyboard and screen readers, and a click anywhere else on the tile is passed to it. */
  $('#tiles').addEventListener('click',e=>{const t=e.target.closest('.tile.is-live'); if(!t||e.defaultPrevented||e.target.closest(TILE_OWN)) return;
    if(String(getSelection&&getSelection()||'').trim()) return; const a=$('.tile-q a',t); if(a) a.click();});
  selectWorld('inference');
}

/* ---------- lab state ---------- */
let S={...DEFAULT}, mode='guided', step=0, pred=null, observed={}, prevR=null, pinned=null, saved=null, chSubmitted=false, chReveal=false, drill=null;
let view='home', labInit=false;
const CH={model:'llama31-70b',gpu:'h100',n:8,ctx:131072,attn:'gqa',util:0.9,overhead:2};
const PRED=[{v:'lt16',t:{en:'Under 16K',zh:'不到 16K'}},{v:'m16',t:{en:'16K to 64K',zh:'16K 到 64K'}},{v:'m64',t:{en:'64K to 128K',zh:'64K 到 128K'}},{v:'all',t:{en:'128K still fits',zh:'128K 也装得下'}}];
/* which prediction option holds the break point: the longest context per user that still fits */
const predRight=maxCtx=>maxCtx<16384?'lt16':(maxCtx<65536?'m16':(maxCtx<131072?'m64':'all'));
/* step 6 levers: single choice, each one starts from the same point and changes only what its label says */
const LV_BASE={...DEFAULT,ctx:131072};
const LEVERS=[
 {id:'start',t:{en:'Initial setup',zh:'初始设置'},d:{en:'Llama-3.1-70B on 8 × H100, all in BF16, 100 users with 128K of context each',zh:'8 × H100 跑 Llama-3.1-70B，全部 BF16，100 个用户，每人 128K 上下文'},p:{}},
 {id:'kv8',t:{en:'KV cache in FP8',zh:'KV cache 改成 FP8'},d:{en:'only the KV cache changes: 1 byte per number instead of 2',zh:'只改 KV cache：每个数值从 2 字节变成 1 字节'},p:{kp:'fp8'}},
 {id:'h200',t:{en:'Bigger GPUs (8 × H200)',zh:'换更大的 GPU（8 × H200）'},d:{en:'141 GB per GPU instead of 80 GB',zh:'每张 141 GB，原来是 80 GB'},p:{gpu:'h200'}},
 {id:'mla',t:{en:'MLA (DeepSeek-V3)',zh:'换成 MLA（DeepSeek-V3）'},d:{en:'MLA (multi-head latent attention) stores one compressed vector per token and layer. The model does not fit on 8 × H100, so this option also uses 8 × H200 and FP8 weights.',zh:'MLA（multi-head latent attention）每个 token 每层只存一个压缩向量。这个模型 8 × H100 放不下，所以同时换成 8 × H200、权重 FP8。'},p:{model:'deepseek-v3',gpu:'h200',wp:'fp8'}},
 {id:'u10',t:{en:'Serve only 10 users',zh:'只服务 10 个用户'},d:{en:'only the number of users changes',zh:'只改用户数'},p:{users:10}}
];
const lvCfg=i=>({...LV_BASE,...LEVERS[i].p});
const SKEYS=['model','gpu','n','ctx','users','wp','kp','attn','util','overhead'];
const leverNow=()=>LEVERS.findIndex((l,i)=>{const c=lvCfg(i); return SKEYS.every(k=>c[k]===S[k]);});
const lvTried=new Set(['start']);
const presetNow=()=>PRESETS.findIndex(p=>SKEYS.every(k=>p.p[k]===S[k]));
const lvResult=i=>tx(`at most ${compute(lvCfg(i)).maxUsers} users`,`最多 ${compute(lvCfg(i)).maxUsers} 个用户`);
/* scenarios: a real deployment or workload each, a plain name and one plain phrase */
const PRESETS=[
 {t:{en:'Chat service',zh:'聊天服务'},d:{en:'70B on 8 × H100, 100 users with 8K of context each',zh:'8 × H100 跑 70B，100 个用户，每人 8K 上下文'},p:{...DEFAULT}},
 {t:{en:'Long-document QA',zh:'长文档问答'},d:{en:'70B on 8 × H100, 8 users with 128K of context each',zh:'8 × H100 跑 70B，8 个用户，每人 128K 上下文'},p:{...DEFAULT,ctx:131072,users:8}},
 {t:{en:'Coding agent',zh:'编程 agent'},d:{en:'70B on 8 × H100 with an FP8 KV cache, 32 users with 32K of context each',zh:'8 × H100 跑 70B，KV cache 用 FP8，32 个用户，每人 32K 上下文'},p:{...DEFAULT,ctx:32768,users:32,kp:'fp8'}},
 {t:{en:'405B on one node',zh:'单机跑 405B'},d:{en:'405B on 8 × H100 with FP8 weights and KV cache, 16 users with 8K of context each',zh:'8 × H100 跑 405B，权重和 KV cache 都用 FP8，16 个用户，每人 8K 上下文'},p:{...DEFAULT,model:'llama31-405b',users:16,wp:'fp8',kp:'fp8'}},
 {t:{en:'DeepSeek-V3 on one node',zh:'单机跑 DeepSeek-V3'},d:{en:'DeepSeek-V3 (MLA, FP8 weights) on 8 × H200, 16 users with 32K of context each',zh:'8 × H200 跑 DeepSeek-V3（MLA，权重 FP8），16 个用户，每人 32K 上下文'},p:{...DEFAULT,model:'deepseek-v3',gpu:'h200',ctx:32768,users:16,wp:'fp8'}},
 {t:{en:'8B on a desktop GPU',zh:'本地跑 8B'},d:{en:'8B on one RTX 4090, 4 users with 8K of context each',zh:'一张 RTX 4090 跑 8B，4 个用户，每人 8K 上下文'},p:{...DEFAULT,model:'llama31-8b',gpu:'rtx4090',n:1,users:4}}
];
/* inline controls: the one setting a step or task asks for sits inside that card, next to its instructions */
const inlineRange=(k,label)=>`<div class="inline-ctl" id="ic-${k}"><label for="i-${k}"><span>${label}</span><output data-out="${k}"></output></label><input type="range" id="i-${k}" data-bind="${k}" min="0" max="${(k==='ctx'?CTX:USERS).length-1}" step="1"></div>`;
const inlineSeg=(k,label,opts)=>`<div class="inline-ctl" id="ic-${k}"><span class="lbl">${label}</span><div class="seg-ctl" role="group" aria-label="${esc(label)}" data-bind-seg="${k}">${opts.map(([v,txt,tip])=>`<button type="button" data-v="${v}"${tip?` data-tipx="${esc(tip)}"`:''}>${txt}</button>`).join('')}</div></div>`;
/* part 4's conclusion in words, from the numbers Show the math displays */
function kvSplitTxt(r){const c=chain(r), n=r.s.n, T=c.totS, P=gib1(c.kvG), U=gib1(c.use), over=c.kvG>c.use;
  const each=n===1?tx('the GPU','这张卡'):tx('each GPU','每张');
  const rel=over?tx(`, and ${each} can use only ${U}.`,`，而${each}只能用 ${U}。`)
    :(c.margin<0?tx(`; with the weights and the runtime overhead, that is more than the ${U} ${each} can use.`,`，加上权重和运行时开销，就超过了${each}能用的 ${U}。`)
               :tx(`; with the weights and the runtime overhead, it still fits in the ${U} ${each} can use.`,`，加上权重和运行时开销，也还在${each}能用的 ${U} 之内。`));
  if(n===1) return tx(`All ${T} of it sits on the one GPU`,`这 ${T} 全都放在这一张 GPU 上`)+rel;
  if(r.mla) return tx(`Every GPU stores a full copy of the ${T}`,`每张 GPU 都要存一整份 ${T}`)+rel;
  return tx(`Split over the ${n} GPUs, that is ${over?'still ':''}${P} on each GPU`,`${n} 张 GPU 平分这些 KV cache，每张${over?'也要':''}存 ${P}`)+rel;}
/* the query-head counts of the Llama-3.1 models in the catalog: "32, 64 or 128" */
const llamaHeads=()=>{const h=[...new Set(Object.values(MODELS).filter(m=>/^Llama-3\.1/.test(m.name)).map(m=>m.heads))].sort((a,b)=>a-b);
  return h.length>1?tx(h.slice(0,-1).join(', ')+' or '+h[h.length-1],h.slice(0,-1).join('、')+' 或 '+h[h.length-1]):String(h[0]);};
const STEPS=[
 {title:{en:'Your mission',zh:'你的任务'},lead:{en:'8 H100s, 100 users: can each one get 128K of context?',zh:'8 张 H100、100 个用户：能不能给每人 128K 上下文？'},enter:()=>({...DEFAULT}),
  body:()=>tx(
   '<p>You serve Llama-3.1-70B to 100 users at once on one 8 × H100 node. Each conversation keeps 8K (8,192) tokens of context; every number here assumes each user fills it.</p><p>The product team wants 128K of context per user. Before anyone promises it, find out whether the 8 GPUs have enough memory for 100 users at 128K each.</p>',
   '<p>你在一台 8 × H100 的机器上，用 Llama-3.1-70B 同时服务 100 个用户。每段对话保留 8K（8,192）个 token 的上下文，下面都按每人用满来算。</p><p>产品团队想把每个用户的上下文提到 128K。答应之前，先弄清楚：每人 128K 时，这 8 张 GPU 的显存还装不装得下 100 个用户。</p>')},
 {title:{en:'Predict first',zh:'先预测'},lead:{en:'Guess first: at what context does it stop fitting?',zh:'先猜：上下文加到多长就装不下了？'},gate:()=>!!pred,skip:true,
  body:()=>{const ch=`<div class="choices" role="group" aria-label="${tx('Your prediction','你的预测')}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${L(o.t)}</button>`).join('')}</div>`;
    return tx(`<p>With 100 users at once, how long can each user’s context get before the 8 × H100 node runs out of GPU memory?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before you test it in the next part.</p>':''}`,
              `<p>同时服务 100 个用户时，每个用户的上下文加到多长，这 8 张 H100 就会显存不足？</p>${ch}${pred?'<p class="muted small">已记下。进入下一部分验证之前，还可以改。</p>':''}`);}},
 {title:{en:'Find the break point',zh:'找到临界点'},lead:{en:'Drag the context below and watch the bar turn red.',zh:'拖下面的上下文长度，看显存条什么时候变红。'},target:['ic-ctx'],enter:()=>({ctx:8192}),expect:r=>!r.fits&&S.ctx>8192,showMe:{ctx:16384},label:{en:'Observed',zh:'观察到'},
  body:()=>tx('<p>Raise each user’s context from 8K until the node runs out of memory.</p>','<p>把每个用户的上下文从 8K 往上加，直到显存不足。</p>')+inlineRange('ctx',t('ctxLabel')),
  obs:r=>{const r128=compute({...S,ctx:131072}), who=S.users===1?tx('the one user no longer fits','这 1 个用户就装不下了'):tx(`the ${usersTxt(S.users)} no longer fit`,`这 ${usersTxt(S.users)}就装不下了`);
    /* result → verdict → reason (R3.10); the reason in words for everyone, with one key number (R6.20): the weights sit on the bar */
    let pf='';
    if(pred){const right=predRight(r.maxCtx), pt=L(PRED.find(p=>p.v===pred).t), rt=L(PRED.find(p=>p.v===right).t);
      pf=pred===right?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
                     :tx(`<p>You predicted “${pt}”; the right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);}
    const far=r.maxCtx<131072;
    const why=tx(`<p>The weights stayed the same as the context grew; only the KV cache got bigger.${r128.fits?'':` At 128K, each GPU would need about ${fmtNum(r128.ratio,1)} times the memory it can use.`}</p>`,
                 `<p>上下文变长时，权重一直没变，变大的只有 KV cache。${r128.fits?'':`到 128K 时，每张 GPU 需要的显存约是它可用显存的 ${fmtNum(r128.ratio,1)} 倍。`}</p>`);
    return tx(`<p><b>Past ${fmtInt(r.maxCtx)} tokens of context each, ${who}.</b>${far?' That is far short of 128K.':''}</p>`,`<p><b>每人的上下文超过 ${fmtInt(r.maxCtx)} 个 token，${who}</b>${far?'，离 128K 还远':''}。</p>`)+pf+why;}},
 {title:{en:'Why the KV cache explodes',zh:'KV cache 为什么会爆'},lead:{en:'Every token of every user keeps its keys and values.',zh:'每个用户的每个 token 都要存 key 和 value。'},enter:()=>({ctx:131072}),
  label:()=>tx(`With the current settings: ${fmtTok(S.ctx)} (${fmtInt(S.ctx)} tokens) of context per user`,`代入当前设置：每人 ${fmtTok(S.ctx)}（${fmtInt(S.ctx)} 个 token）上下文`),
  /* the mechanism in words; the formula lines in the result box plug in Llama-3.1-70B's shape (R6.20) */
  body:()=>tx(`<p>For every token of context, each layer stores a key vector and a value vector for each ${DEF('KV head',KVH_TIP())}, and each vector holds head_dim numbers. The formulas below plug in Llama-3.1-70B’s layers, KV heads and head_dim, in BF16.</p><p>The weights are stored once and shared by all users. Each user has their own KV cache, and it grows with every token of their context.</p>`,
              `<p>上下文里的每个 token，在每一层的每个 ${DEF('KV head',KVH_TIP())} 上都要存一个 key 向量和一个 value 向量，每个向量有 head_dim 维。下面的算式代入了 Llama-3.1-70B 的层数、KV head 数和 head_dim，按 BF16 算。</p><p>权重只存一份，所有用户共用。KV cache 每个用户各有一份，上下文越长，它越大。</p>`),
  obs:r=>mathLines(r).slice(0,3).map(x=>`<p class="mono small">${x}</p>`).join('')+`<p>${kvSplitTxt(r)}</p>`},
 {title:{en:'What if it used MHA?',zh:'如果用 MHA 呢？'},lead:{en:'With MHA the KV cache would be 8 times larger.',zh:'换成 MHA，KV cache 会变成 8 倍。'},target:['ic-attn'],enter:()=>({ctx:8192,attn:'gqa'}),expect:()=>S.attn==='mha',showMe:{attn:'mha'},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>Llama-3.1-70B has 64 ${DEF('query heads',QHm(MODELS['llama31-70b']))}, and every 8 of them share one ${DEF('KV head',KVH_TIP())}, so it stores keys and values for only 8 heads. That is ${DEF('grouped-query attention (GQA)',GQAm(MODELS['llama31-70b']))}.</p><p>Context is back to 8K per user. Switch <b>Attention</b> below to <b>MHA</b> (multi-head attention) to give each query head its own KV head.</p><p class="small muted">Heads are explained from scratch in the lab ${labLink('mha')}, and GQA in the lab ${labLink('gqa-mla')}; both are in the “Transformer structure” learning path.</p>`,
              `<p>Llama-3.1-70B 有 64 个 ${DEF('query head',QHm(MODELS['llama31-70b']))}，每 8 个共用一个 ${DEF('KV head',KVH_TIP())}，所以只为 8 个 head 存 key 和 value。这叫 ${DEF('grouped-query attention（GQA）',GQAm(MODELS['llama31-70b']))}。</p><p>上下文回到每人 8K。把下面的<b>注意力</b>切到 <b>MHA</b>（多头注意力），让每个 query head 都有自己的 KV head。</p><p class="small muted">head 在${labLink('mha')}那个 Lab 里从头讲，GQA 在 ${labLink('gqa-mla')} 那个 Lab 里讲，两个都在学习路线“Transformer 结构”里。</p>`)
            +inlineSeg('attn',t('cAttn'),[['gqa',t('attnNative'),attnOptTip('gqa',MODELS['llama31-70b'])],['mha',t('attnMha'),attnOptTip('mha',MODELS['llama31-70b'])]]),
  /* the trend and the reason in words, one key number (R6.20); the sizes are on the metric cards, with a ▲ badge */
  obs:()=>{const a=compute({...S,attn:'gqa'}),b=compute({...S,attn:'mha'}),k=fmtNum(b.kvPerToken/a.kvPerToken,0);
    const one=S.users===1, who=one?tx('the one user','这 1 个用户'):tx(`the ${usersTxt(S.users)}`,`这 ${usersTxt(S.users)}`);
    const fit=b.fits?tx(`At ${fmtTok(S.ctx)} of context each, ${who} still ${one?'fits':'fit'}.`,`每人 ${fmtTok(S.ctx)} 上下文时，${who}还装得下。`)
      :tx(`Even at ${fmtTok(S.ctx)} of context each, ${who} no longer ${one?'fits':'fit'}.`,`即使每人只有 ${fmtTok(S.ctx)} 上下文，${who}也装不下了。`);
    return tx(`<p><b>KV cache per token becomes ${k} times as large</b>, since every query head now keeps its own key and value. ${fit}</p><p>That is why every Llama-3.1 model in this lab uses GQA: however many query heads it has, they share just 8 KV heads.</p>`,
              `<p><b>每个 token 的 KV cache 变成原来的 ${k} 倍</b>，因为现在每个 query head 都存自己的 key 和 value。${fit}</p><p>所以这个 Lab 里的 Llama-3.1 模型都用 GQA：不管有多少个 query head，都只共用 8 个 KV head。</p>`);}},
 {title:{en:'When it does not fit',zh:'装不下怎么办'},lead:{en:'Four ways to win memory back: pick one, compare.',zh:'四种省显存的办法：选一个，比一比。'},enter:()=>({...LV_BASE}),levers:true,label:{en:'Result',zh:'结果'},
  body:()=>{const b=compute(LV_BASE), cur=leverNow();
    const opts=`<div class="levers" role="group" aria-label="${tx('Four ways to win memory back, pick one','四种省显存的办法，选一种')}">${LEVERS.map((l,i)=>`<button type="button" class="choice lever" data-lever="${i}" aria-pressed="${i===cur}"><span class="lv-t"><b>${L(l.t)}</b><span class="lv-d">${L(l.d)}</span></span><span class="lv-r mono">${lvTried.has(l.id)?lvResult(i):''}</span></button>`).join('')}</div>`;
    return tx(`<p>Each user now has 128K of context, and 100 users do not fit on 8 × H100: at most ${b.maxUsers} do.</p><p>Below are four ways to win memory back. Pick one and see how many users fit. Each starts from the initial setup and changes only the settings it names.</p>${opts}`,
              `<p>现在每个用户有 128K 上下文，8 × H100 装不下 100 个用户，最多只能装下 ${b.maxUsers} 个。</p><p>下面有四种省显存的办法。选一种，看看能装下多少个用户。每种都从初始设置出发，只改它写明的设置。</p>${opts}`);},
  obs:r=>{const i=leverNow(), b=compute(LV_BASE), J=`<button type="button" class="linkbtn" data-jump="controls">${tx('Settings','自由调参')}</button>`;
    if(i<0) return tx(`<p>You changed something under ${J}, so the setup is no longer one of the options above.</p><p>Right now <b>at most ${usersTxt(r.maxUsers)} fit</b> at ${fmtTok(S.ctx)} of context each.</p>`,
                      `<p>你在“${J}”里改了设置，现在的配置不属于上面任何一个选项。</p><p>当前每人 ${fmtTok(S.ctx)} 上下文时，<b>最多装下 ${fmtInt(r.maxUsers)} 个用户</b>。</p>`);
    if(i===0) return tx(`<p><b>Initial setup: at most ${b.maxUsers} users fit</b> at 128K of context each, far short of the 100 you serve.</p>`,
                        `<p><b>初始设置：每人 128K 上下文时，最多装下 ${b.maxUsers} 个用户</b>，离要服务的 100 个还差很远。</p>`);
    const l=LEVERS[i], d=r.maxUsers-b.maxUsers;
    const dTxt=d>0?tx(`${d} more`,`多 ${d} 个`):(d<0?tx(`${-d} fewer`,`少 ${-d} 个`):tx('no change','没有变化'));
    let h=tx(`<p><b>${L(l.t)}: at most ${usersTxt(r.maxUsers)} fit</b> (initial setup: ${b.maxUsers}, ${dTxt}).</p>`,
             `<p><b>${L(l.t)}：最多装下 ${fmtInt(r.maxUsers)} 个用户</b>（初始设置是 ${b.maxUsers} 个，${dTxt}）。</p>`);
    /* why, in words: the sizes are on the metric cards and in Show the math (R6.20) */
    if(l.id==='kv8') h+=tx(`<p>Each token’s KV cache is now half as big, so the same memory holds about twice as many users. 100 users still do not fit.</p>`,
                          `<p>每个 token 的 KV cache 只有原来的一半，同样的显存能装下的用户大约翻一倍。100 个用户还是装不下。</p>`);
    if(l.id==='h200'){const kv=compute(lvCfg(1)).maxUsers, same=kv===r.maxUsers;
      h+=tx(`<p>Usable memory per GPU grows from ${gib1(chain(b).use)} to ${gib1(chain(r).use)}. The weights stay the same, so all of the extra memory goes to the KV cache. ${same?'That fits as many users as “KV cache in FP8”':'That fits about as many users as “KV cache in FP8”'}, but it means buying new GPUs.</p>`,
            `<p>每张卡的可用显存从 ${gib1(chain(b).use)} 变成 ${gib1(chain(r).use)}。权重没变，多出来的显存全给了 KV cache。${same?'能装下的用户和“KV cache 改成 FP8”一样多':'能装下的用户和“KV cache 改成 FP8”差不多'}，但要花钱换卡。</p>`);}
    if(l.id==='mla'){const k=fmtNum(b.kvPerToken/r.kvPerToken,1), half=r.wPerGpu>r.usable/2;
      /* the same MLA in Llama-3.1-70B, as the lab gqa-mla computes it: 80 layers × (512 + 64) numbers, 2 bytes, a full copy on every GPU */
      const m70=MODELS['llama31-70b'], ds=MODELS['deepseek-v3'], mlaReq=m70.layers*(ds.kvLora+ds.ropeDim)*2*LV_BASE.ctx, u70=b.budget>0?Math.floor(b.budget/mlaReq+1e-9):0;
      h+=tx(`<p>${DEF('MLA',MLA_TIP())} stores far less per token: Llama-3.1-70B stores ${k} times as much. But even in FP8, DeepSeek-V3’s weights take ${half?'more than half':'a large part'} of what each GPU can use.</p><p>And the compressed vector is not split by head, so ${DEF('tensor parallelism cannot divide it',TP70())} across the ${r.s.n} GPUs: every GPU keeps a full copy for every user, so fewer users fit, not more.</p><p class="small muted">The lab ${labLink('gqa-mla')} puts the same MLA into Llama-3.1-70B, whose weights are much smaller. Even there only ${u70} users fit at 128K each on 8 × H100, fewer than with GQA.</p>`,
            `<p>${DEF('MLA',MLA_TIP())} 每个 token 存得少得多：Llama-3.1-70B 要存它的 ${k} 倍。可是 DeepSeek-V3 的权重即使用 FP8，也要占每张卡可用显存的${half?'一半以上':'一大块'}。</p><p>而且这个压缩向量不按 head 分，${DEF('张量并行没法把它拆到',TP70())} ${r.s.n} 张卡上：每张卡都要存每个用户的一整份，所以能装下的用户反而更少。</p><p class="small muted">${labLink('gqa-mla')} 那个 Lab 把同样的 MLA 放进权重小得多的 Llama-3.1-70B。即使这样，在 8 × H100 上每人 128K 时也只能装下 ${u70} 个用户，比 GQA 少。</p>`);}
    if(l.id==='u10') h+=tx('<p>No memory was saved. Demand just dropped from 100 users to 10, which now fits. The cost is turning all the other users away.</p>',
                          '<p>一点显存都没省下。只是需求从 100 个用户降到 10 个，刚好装得下，代价是其余的用户都得不到服务。</p>');
    return h;}},
 {title:{en:'What to remember',zh:'要记住的'},
  /* one key number per point, the rest in words (R6.20): the sizes are on the bars, the metric cards and Show the math */
  body:()=>{const r=compute({...DEFAULT,ctx:131072}), r8=compute({...DEFAULT}), k=fmtNum(r.kvTotal/r.wTotal,0), k8=fmtNum(r8.kvPerGpu/r8.wPerGpu,1);
    /* the rule as a typeset formula (H1, batch 9), in the words of kv-basics' Show the math; pieces that wrap before an operator (.kv-fx) */
    // a piece that starts with × gets a thin space after it (\\,): Chromium drops the space after an mrow's leading ×
    const fx=(...p)=>`<div class="fx kv-fx">${p.map(x=>texMath(x.replace(/^\\times /,'\\times\\, '))).join('<wbr>')}</div>`;
    return tx(`<ul><li>Every token keeps a key and a value for each KV head in every layer:${fx('\\text{KV cache per token}','= 2 \\times \\text{layers}','\\times \\text{KV heads}','\\times \\text{head_dim}','\\times \\text{bytes per number}')}Multiply it by each user’s tokens and by the number of users to get the KV cache of all users.</li><li>At long context the KV cache, not the weights, decides how many users fit: at 128K each, the 100 users’ KV cache is about ${k} times the size of the weights.</li><li>${DEF('GQA',GQAm(MODELS['llama31-70b']))} and an FP8 KV cache shrink it. ${DEF('MLA',MLA_TIP())} stores less per token, but ${DEF('with plain tensor parallelism every GPU keeps a full copy',TPPLAIN())} (the lab ${labLink('gqa-mla')} covers both).</li></ul><div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “Context length only costs compute.” With Llama-3.1-70B in BF16, every token of every user’s context holds ${fmtB(r.kvPerToken)} of GPU memory.</p><p>✗ “The weights are what fill GPU memory.” Even at 8K of context each, the 100 users’ KV cache takes more of each GPU than the weights do: about ${k8} times as much.</p></div>${toScenariosHTML()}`,
              `<ul><li>每个 token 在每一层、每个 KV head 上都存一个 key 和一个 value：${fx('\\text{每个 token 的 KV cache}','= 2 \\times \\text{层数}','\\times \\text{KV head 数}','\\times \\text{head_dim}','\\times \\text{每个数值的字节数}')}再乘以每个用户的 token 数和用户数，就是全部用户的 KV cache。</li><li>长上下文时，决定能服务多少用户的是 KV cache，而不是权重：每人 128K 时，100 个用户的 KV cache 约是权重的 ${k} 倍。</li><li>${DEF('GQA',GQAm(MODELS['llama31-70b']))} 和 FP8 KV cache 能把它变小。${DEF('MLA',MLA_TIP())} 每个 token 存得更少，但${DEF('纯张量并行时每张卡要各存一整份',TPPLAIN())}（${labLink('gqa-mla')} 那个 Lab 专门讲这两种）。</li></ul><div class="obs"><div class="eyebrow">常见误解</div><p>✗ “上下文长度只影响计算。”在 Llama-3.1-70B（BF16）上，每个用户上下文里的每个 token 都要占 ${fmtB(r.kvPerToken)} 显存。</p><p>✗ “占满显存的是权重。”即使每人只有 8K 上下文，100 个用户的 KV cache 在每张卡上占的显存也比权重多，约是权重的 ${k8} 倍。</p></div>${toScenariosHTML()}`);}}
];

/* shared definitions (A.terms): no model in them; each lab adds its own model sentence after the kit text (CONTRACT §9) */
const QH_TIP=()=>tx('In each layer, each token’s query is cut into equal parts, one per query head. A query head computes its part with its own rows of the query matrix Wq (the matrix that turns a token into its query), then compares that part with the keys of the current token and every earlier token to decide how much information to take from each. Because the parts differ, different query heads can attend to different tokens.',
  '每一层里，每个 token 的 query 被切成相等的几份，每份属于一个 query head。query head 用 query 矩阵 Wq（把 token 变成 query 的矩阵）里属于自己的那几行，算出自己那一份，再拿它和当前 token 及之前每个 token 的 key 比较，决定从每个 token 取多少信息。各份 query 不同，所以不同的 query head 可以关注不同的 token。');
const KVH_TIP=()=>tx('A KV head is one key vector and one value vector that each token stores in every layer, used by one head or shared by several. What is stored is the KV cache, so fewer KV heads mean a smaller KV cache in total.',
  'KV head 是每个 token 在每一层存的一个 key 向量和一个 value 向量，可以给一个 head 用，也可以由几个 head 共用。存下来的就是 KV cache，所以 KV head 越少，KV cache 的总量越小。');
const GQA_TIP=()=>tx('Grouped-query attention (GQA) splits the query heads into groups, and each group shares one KV head, so each token stores one key and one value per group instead of one per query head. The GQA paper (Ainslie et al., 2023) took T5-XXL with multi-head attention (MHA, where every head has its own key and value), converted it to 8 groups and trained it a little further; it scored almost as well as before (47.1 against 47.2 on average). The paper did not test decoder-only models such as Llama.',
  'grouped-query attention（GQA）把 query head 分成几组，同一组共用一个 KV head，所以每个 token 每组只存一个 key 和一个 value，而不是每个 query head 各存一份。GQA 论文（Ainslie 等，2023）把用多头注意力（MHA，每个 head 各有 key 和 value）的 T5-XXL 改成 8 组、再稍加训练，平均分几乎和改之前一样（47.1 对 47.2）。论文没有测 Llama 这类 decoder-only 模型。');
const MLA_TIP=()=>tx('MLA (multi-head latent attention, introduced with DeepSeek-V2) does not store a key and a value per head. In each layer it stores one compressed vector per token (512 dimensions in DeepSeek-V3) plus a 64-dimension key that carries the token’s position (RoPE); every head’s keys and values are derived from them. At decode steps, engines usually fold the matrices that derive each head’s key and value into Wq and Wo, and compute attention on the compressed vector directly.',
  'MLA（multi-head latent attention，随 DeepSeek-V2 提出）不再为每个 head 存 key 和 value。它在每一层给每个 token 存一个压缩向量（DeepSeek-V3 是 512 维），再加一个 64 维、带位置信息（RoPE）的 key；每个 head 的 key 和 value 都由它们算出。decode 步里，引擎通常把用来算出各 head 的 key 和 value 的矩阵并进 Wq 和 Wo，直接在压缩向量上算 attention。');
const TPMLA_TIP=()=>tx('Tensor parallelism splits attention across the GPUs by head: each GPU computes some of the query heads and keeps the keys and values those heads read, at least one KV head. MLA’s compressed vector is read by every head, so it cannot be split that way, and every GPU keeps a full copy for every user. Serving stacks such as SGLang can run attention data-parallel instead to avoid this.',
  '张量并行按 head 把 attention 拆到几张 GPU 上：每张 GPU 算其中几个 query head，并存下这些 head 要读的 key 和 value，最少一个 KV head。MLA 的压缩向量每个 head 都要读，没法这样拆，所以每张 GPU 都要为每个用户存一整份。SGLang 等框架可以改让 attention 做数据并行来避免这一点。');
/* the site's structural terms (review 8): the “Transformer structure” lab introduces them and every lab uses the same words.
   Like the definitions above they name no model; a lab adds its own model sentence after them (CONTRACT §11) */
const TERMS={
 token:()=>tx('A token is the unit of text a model reads and writes: a word, part of a word, or a punctuation mark. The tokenizer cuts text into tokens and gives each one an ID, which is its row in the embedding table.',
              'token 是模型读写文字的单位：一个词、词的一部分或一个标点。tokenizer 把文字切成 token，并给每个 token 一个编号（token ID），也就是它在 embedding 表里的行号。'),
 embedding:()=>tx('The embedding table has one row for every token in the vocabulary, and each row is a vector of d numbers (d is the hidden size). The first thing the model does is look up the row of each input token: a lookup, not a computation.',
                  'embedding 表的每一行对应词表里的一个 token，每行是一个 d 维向量（d 是隐藏维度）。模型做的第一件事，就是查出每个输入 token 的那一行：这是查表，不是计算。'),
 hidden:()=>tx('The hidden size d is the length of the vector that stands for each token between the layers. Every layer takes in a vector of d numbers per token and passes on one of the same length.',
               '隐藏维度 d 是层与层之间代表每个 token 的向量的长度。每一层给每个 token 读入一个 d 维向量，也传出一个 d 维向量。'),
 layer:()=>tx('A layer is an attention block followed by an MLP block. Before each block an RMSNorm rescales the vector the block reads, and the block’s output is added back to the vector from before that RMSNorm (the residual connection). A model stacks many layers of the same shape, each with its own parameters.',
              '一层就是一个 attention 块加一个 MLP 块。每个块读入的向量先经过一个 RMSNorm 调整大小，块的输出再加回到这个 RMSNorm 之前的向量上（残差连接）。模型把很多层叠起来，形状都一样，参数各不相同。'),
 attention:()=>tx('The attention block is where a token takes information from other tokens: from itself and the tokens before it. It has four parameter matrices, Wq, Wk, Wv and Wo, and runs several heads side by side (multi-head attention).',
                  'attention 块是一个 token 从别的 token（它自己和它前面的 token）取信息的地方。它有四个参数矩阵 Wq、Wk、Wv、Wo，并排跑好几个 head（多头注意力）。'),
 mha:()=>tx('Multi-head attention (MHA): Wq, Wk and Wv turn each token’s vector into a query, a key and a value; each is cut into equal parts, one per head. Each head compares its part of the query with its part of every key it can see, turns the scores into attention weights that add up to 100%, and mixes its part of the values by those attention weights. The heads’ outputs are joined and multiplied by Wo. In MHA every head keeps its own key and value.',
            '多头注意力（multi-head attention，MHA）：Wq、Wk、Wv 把每个 token 的向量变成一个 query、一个 key 和一个 value，每个都切成相等的几份，每份属于一个 head。每个 head 拿自己那份 query 和它能看到的每个 key 的同一份比较，把得分换成加起来是 100% 的注意力权重，再按这些注意力权重混合 value 的那一份。各 head 的输出拼起来，再乘 Wo。MHA 里每个 head 都有自己的 key 和 value。'),
 head:()=>tx('A head is one attention computation inside the attention block: it gives every token it can see an attention weight, these attention weights add up to 100%, and it takes that mix of their values.',
             'head 是 attention 块里的一次注意力计算：它给能看到的每个 token 一个注意力权重，这些注意力权重加起来是 100%，它就按这个比例取这些 token 的 value。'),
 aweight:()=>tx('An attention weight is how much of one token’s value a head takes, as a share. A head’s attention weights over all the tokens it can see add up to 100%.',
                '注意力权重是一个 head 从某个 token 的 value 里取多少，是一个份额。一个 head 在它能看到的所有 token 上的注意力权重加起来是 100%。'),
 mlp:()=>tx('The MLP block (MLP: multi-layer perceptron) transforms each token’s vector on its own, without looking at other tokens. In Llama models it widens the vector from d to the intermediate size with two matrices (gate and up), applies SiLU to the gate side and multiplies the two element by element, then narrows the result back to d with a third matrix (down).',
            'MLP 块（MLP 即多层感知机，multi-layer perceptron）单独变换每个 token 的向量，不看别的 token。Llama 这类模型用两个矩阵（gate 和 up）把向量从 d 维放大到中间维度，对 gate 那一路做 SiLU，再和 up 那一路逐维相乘，最后用第三个矩阵（down）缩回 d 维。'),
 residual:()=>tx('A residual connection adds a block’s output back to the vector as it was before that block’s RMSNorm, so each block adds a change instead of replacing the vector; the vector keeps its length d.',
                 '残差连接把一个块的输出加回到这个块的 RMSNorm 之前的那个向量上，所以每个块是加上一个改动，而不是换掉整个向量；向量始终是 d 维。'),
 rmsnorm:()=>tx('RMSNorm divides a vector by the root mean square of its numbers, then multiplies each dimension by a learned scale (these d scales are its parameters), so every block sees vectors of a similar size.',
                'RMSNorm 把向量除以它各维的均方根，再逐维乘上一个学到的系数（这 d 个系数就是它的参数），让每个块看到的向量大小相近。'),
 lmhead:()=>tx('The LM head (LM: language model) is the last matrix: it turns the final vector of d numbers into one score, a logit, for every token in the vocabulary. Some models tie the two: the LM head reuses the embedding table’s parameters.',
               'LM head（LM 即语言模型，language model）是最后一个矩阵：它把最后的 d 维向量变成词表里每个 token 一个得分（logit）。有的模型让两者共享参数：LM head 直接用 embedding 表的参数。'),
 logit:()=>tx('A logit is the LM head’s score for one token. Softmax turns all the logits into probabilities that add up to 100%, and the next token is picked from them.',
              'logit 是 LM head 给某个 token 的得分。softmax 把所有 logit 变成加起来是 100% 的概率，下一个 token 就从中选出。'),
 param:()=>tx('A parameter is one learned number in the model’s matrices and norms. A model’s size, such as 8B, counts its parameters; in BF16 each takes 2 bytes.',
              '参数是模型的矩阵和 norm 里一个学到的数。模型的大小（如 8B）数的就是参数；BF16 下每个参数占 2 字节。'),
 kvcache:()=>tx('The KV cache keeps the key and value vectors of every token the model has already processed, in every layer, so that when the model generates the next token it reads them from memory instead of computing them again.',
                'KV cache 存着模型已经处理过的每个 token 在每一层的 key 和 value 向量，模型生成下一个 token 时直接从显存里读，不再重算。'),
 gib:()=>tx('GiB (gibibyte) is 2³⁰ = 1,073,741,824 bytes, about 7% more than a GB (10⁹ bytes). This site gives memory in GiB. File sizes and byte counts are often given in GB, so 16.1 GB is 15.0 GiB. GPU memory is the exception: a card’s GB figure is close to that many GiB (CUDA reports 39.6 GiB for the A100 40GB), so this site reads the vendor’s GB figure as GiB.',
            'GiB = 2³⁰ = 1,073,741,824 字节，比 GB（10⁹ 字节）大约 7%。本站的显存都用 GiB。文件大小和按字节算出的量常用 GB，所以 16.1 GB 就是 15.0 GiB。显卡的显存是例外：显卡标的 GB 实际接近同样多的 GiB（标 40 GB 的 A100，CUDA 报告 39.6 GiB），所以本站把厂商标的 GB 按 GiB 计。'),
 /* round 15: the words the structure labs share, from the lab Architecture overview (A.terms.*; each lab may add its own model sentence) */
 mask:()=>tx('The causal mask gives every later token no attention weight: a token takes information only from itself and the tokens before it. When a model writes text, the later tokens do not exist yet.',
             '因果 mask 让后面的 token 分不到注意力权重：一个 token 只从它自己和它前面的 token 取信息。模型写文字时，后面的 token 本来就还没写出来。'),
 /* R6.29 (2026-10-03): the effect only; a lab that needs the formula shows it in full with A.ui.math (architecture's term tip, the softmax lab) */
 softmax:()=>tx('Softmax turns a list of scores into shares, such as probabilities or attention weights, that add up to exactly 100%. A higher score always gets a larger share.',
                'softmax 把一串得分变成一组份额，这些份额加起来正好是 100%。得分越高，分到的份额越大。下一个 token 的概率、注意力权重，都是这样算出来的。'),
 qkv:()=>tx('Wq, Wk and Wv turn each token’s vector into three vectors of the same length: a query (what the token looks for), a key (what the other tokens’ queries are compared with) and a value (what the token hands over to a head that takes from it).',
            'Wq、Wk、Wv 把每个 token 的向量变成三个同样长的向量：query（这个 token 要找什么）、key（拿来和别的 token 的 query 比较的）和 value（被某个 head 取用时交出去的信息）。'),
 inter:()=>tx('The intermediate size is the length the MLP block widens each vector to before narrowing it back to d; config.json calls it intermediate_size.',
              '中间维度是 MLP 块先把每个向量放大到的长度，之后再缩回 d 维；config.json 里叫 intermediate_size。'),
 mqa:()=>tx('Multi-query attention (MQA): all the query heads share a single KV head, so each token stores one key and one value per layer.',
            'multi-query attention（MQA）：所有 query head 共用一个 KV head，所以每个 token 每层只存一个 key 和一个 value。'),
 tp:()=>tx('Tensor parallelism splits every layer across several GPUs: attention by head, so each GPU computes some of the query heads, and the rest of the layer so that each GPU holds a share of every parameter matrix. The GPUs exchange results after each attention block and MLP block.',
           '张量并行把每一层拆到几张 GPU 上：attention 按 head 拆，每张 GPU 算其中几个 query head；层里其余的矩阵也拆开，每张 GPU 存每个参数矩阵的一块。每个 attention 块和 MLP 块之后，GPU 之间交换结果。'),
 kib:()=>tx('A KiB (kibibyte) is 2¹⁰ = 1,024 bytes, about 2% more than a KB (1,000 bytes), and a MiB is 2²⁰ bytes (1,024 KiB). Both count in powers of 2: 131,072 bytes are 128 KiB, or 131 KB.',
            'KiB = 2¹⁰ = 1,024 字节，比 KB（1,000 字节）大约 2%；MiB = 2²⁰ 字节（1,024 KiB）。它们都按 2 的幂计：131,072 字节就是 128 KiB，也就是 131 KB。')
};
/* R6.12: the first GiB a lab shows carries the definition (and what a GB is) */
const gibD=t=>String(t).replace(/ GiB$/,' '+DEF('GiB',TERMS.gib()));
/* a small TeX subset to MathML, for formulas in hover tips and cards (review 12: they looked like LaTeX source).
   Supports: x_i, x^2, x_{..}^{..}, \frac{a}{b}, \sqrt{x}, \hat{x}, \text{..}, \left( … \right), \big(, Greek (\sigma, \Sigma, \theta …),
   \sum, \cdot, \odot, \oplus, \otimes, \times, \div, \approx, \to, \leftarrow, \max, \min, \log, \exp; multi-letter words stay
   upright (softmax, RMSNorm, SiLU). Batch 9 (the site-wide formula sweep) adds the operators \nabla ∇, \pm ±, \mp ∓, \le ≤, \ge ≥,
   \ne ≠, \sim ∼, \propto ∝, \mid ∣, \prime ′, \star ⋆, \ast ∗ (each an <mo>), the fences \langle ⟨, \rangle ⟩, \lVert ‖, \rVert ‖
   (fixed size; growing after \left / \right), the letters \ell ℓ and \partial ∂ (<mi>: as an <mo>, ∂ got a gap before it), and
   \mathbb{E} 𝔼 (also R ℝ, N ℕ, P ℙ, any letter or digit) as <mi mathvariant="normal">. Those characters typed as themselves render
   the same way. An unknown command still shows as its name: add it here. */
const TEXSYM={alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',varepsilon:'ε',eta:'η',theta:'θ',lambda:'λ',mu:'μ',pi:'π',sigma:'σ',tau:'τ',phi:'φ',rho:'ρ',omega:'ω',Sigma:'Σ',zeta:'ζ',iota:'ι',kappa:'κ',nu:'ν',xi:'ξ',upsilon:'υ',chi:'χ',psi:'ψ',varphi:'φ',vartheta:'ϑ',Gamma:'Γ',Delta:'Δ',Theta:'Θ',Lambda:'Λ',Xi:'Ξ',Pi:'Π',Phi:'Φ',Psi:'Ψ',Omega:'Ω',
  sum:'Σ',cdot:'·',odot:'⊙',oplus:'⊕',otimes:'⊗',times:'×',div:'÷',approx:'≈',leq:'≤',geq:'≥',neq:'≠',to:'→',rightarrow:'→',leftarrow:'←',infty:'∞',ldots:'…',cdots:'⋯',in:'∈',quad:' ',
  nabla:'∇',partial:'∂',pm:'±',mp:'∓',le:'≤',ge:'≥',ne:'≠',sim:'∼',propto:'∝',ell:'ℓ',mid:'∣',prime:'′',star:'⋆',ast:'∗',lVert:'‖',rVert:'‖',langle:'⟨',rangle:'⟩',
  max:'max',min:'min',log:'log',exp:'exp',mean:'mean'};
const TEXLETTER=/[ℓ∂]/, TEXFENCE=/[⟨⟩‖]/, TEXBB={C:'ℂ',H:'ℍ',N:'ℕ',P:'ℙ',Q:'ℚ',R:'ℝ',Z:'ℤ'};
// \mathbb: capitals and digits from the double-struck block (ℂ ℍ ℕ ℙ ℚ ℝ ℤ live in Letterlike Symbols), small letters too
const texBB=ch=>TEXBB[ch]||(/[A-Z]/.test(ch)?String.fromCodePoint(0x1D538+ch.charCodeAt(0)-65):(/[a-z]/.test(ch)?String.fromCodePoint(0x1D552+ch.charCodeAt(0)-97):(/[0-9]/.test(ch)?String.fromCodePoint(0x1D7D8+ch.charCodeAt(0)-48):ch)));
const texFence=g=>`<mo stretchy="false" lspace="0.04em" rspace="0.04em">${g}</mo>`;
function texMath(src){const s=String(src); let i=0;
  const sp=()=>{while(i<s.length&&s[i]===' ') i++;};
  const row=x=>`<mrow>${x}</mrow>`;
  function atom(){sp(); const c=s[i]; if(c===undefined) return '';
    if(c==='\\'){i++; let n=''; while(i<s.length&&/[A-Za-z]/.test(s[i])) n+=s[i++];
      if(!n){const ch=s[i++]||''; if(ch==='!') return ''; const w={',':'0.2em',' ':'0.2em',':':'0.22em',';':'0.28em'}[ch]; return w?`<mspace width="${w}"/>`:`<mo>${esc(ch)}</mo>`;}
      if(n==='frac'){const a=grp(), b=grp(); return `<mfrac>${row(a)}${row(b)}</mfrac>`;}
      if(n==='sqrt') return `<msqrt>${row(grp())}</msqrt>`;
      if(n==='hat') return `<mover accent="true">${row(grp())}<mo>^</mo></mover>`;
      // \left( … \right) is one group whose delimiters grow with what they enclose; \big( and its kin are one fixed size larger
      // (batch 9: they showed as the words “left”, “big”)
      if(n==='left'){const o=delim(), r=seq('\\right'); if(s.startsWith('\\right',i)) i+=6; return row(o+r+delim());}
      if(n==='right') return delim();
      const bg=/^(big|Big|bigg|Bigg)[lr]?$/.test(n)?{big:1.2,Big:1.6,bigg:2,Bigg:2.4}[n.replace(/[lr]$/,'')]:0;
      if(bg){sp(); return atom().replace(/^<mo(?: stretchy="false")?/,`<mo stretchy="true" symmetric="true" minsize="${bg}em" maxsize="${bg}em"`);}
      if(n==='text'||n==='mathrm'||n==='operatorname'){sp(); let t=''; if(s[i]==='{'){i++; while(i<s.length&&s[i]!=='}') t+=s[i++]; i++;} return n==='text'?`<mtext>${esc(t)}</mtext>`:`<mi mathvariant="normal">${esc(t)}</mi>`;}
      if(n==='mathbb'){sp(); let t=''; if(s[i]==='{'){i++; while(i<s.length&&s[i]!=='}') t+=s[i++]; i++;} else t=s[i++]||'';
        return [...t.replace(/\s+/g,'')].map(ch=>`<mi mathvariant="normal">${esc(texBB(ch))}</mi>`).join('');}
      const g=TEXSYM[n]; if(g==null) return `<mi>${esc(n)}</mi>`;
      if(g.length>1) return g===' '?'<mspace width="1em"/>':`<mi>${g}</mi>`;
      if(TEXLETTER.test(g)) return `<mi>${g}</mi>`;
      if(TEXFENCE.test(g)) return texFence(g);
      return /[α-ωϑ∞]/.test(g)?`<mi>${g}</mi>`:(g==='Σ'&&n==='sum'?`<mo largeop="false">Σ</mo>`:(/[Α-Ω]/.test(g)?`<mi mathvariant="normal">${g}</mi>`:`<mo>${g}</mo>`));}
    if(c==='{'){i++; const r=seq('}'); i++; return row(r);}
    if(/[0-9]/.test(c)){let n=''; while(i<s.length&&(/[0-9]/.test(s[i])||(/[.,]/.test(s[i])&&/[0-9]/.test(s[i+1]||'')))) n+=s[i++]; return `<mn>${n}</mn>`;}
    if(/[A-Za-z]/.test(c)){let w=''; while(i<s.length&&/[A-Za-z]/.test(s[i])) w+=s[i++]; return `<mi>${w}</mi>`;}
    // a character outside the basic plane (𝔼, 𝟙, 𝜃 typed as themselves) is one letter, not two halves
    if(/[\uD800-\uDBFF]/.test(c)){const ch=String.fromCodePoint(s.codePointAt(i)); i+=ch.length; return `<mi mathvariant="normal">${ch}</mi>`;}
    i++; if(/[α-ωϑ]/.test(c)||TEXLETTER.test(c)) return `<mi>${c}</mi>`; if(/[Α-Ω]/.test(c)) return c==='Σ'?`<mo largeop="false">Σ</mo>`:`<mi mathvariant="normal">${c}</mi>`;
    if(/[ℂℍℕℙℚℝℤ]/.test(c)) return `<mi mathvariant="normal">${c}</mi>`;
    if(TEXFENCE.test(c)) return texFence(c);
    if('()[]|'.includes(c)) return `<mo stretchy="false" lspace="0.04em" rspace="0.04em">${esc(c)}</mo>`;
    if(c===',') return '<mo lspace="0" rspace="0.3em">,</mo>';
    return c==='−'||c==='-'?'<mo>−</mo>':`<mo>${esc(c)}</mo>`;}
  function grp(){sp(); if(s[i]==='{'){i++; const r=seq('}'); i++; return r;} return atom();}
  function delim(){sp(); if(s[i]==='.'){i++; return '';} return atom().replace(/^<mo(?: stretchy="false")?/,'<mo stretchy="true" symmetric="true"');}
  function seq(end){let out=''; for(;;){sp(); if(i>=s.length||(end&&s.startsWith(end,i)&&!/[A-Za-z]/.test(end.length>1?s[i+end.length]||'':''))) break; let b=atom(); const isFn=/^<mi>(max|min|log|exp|mean)<\/mi>$/.test(b);
      for(;;){sp(); if(s[i]==='_'){i++; const lo=grp(); sp(); if(s[i]==='^'){i++; const hi=grp(); b=`<msubsup>${row(b)}${row(lo)}${row(hi)}</msubsup>`;} else b=`<msub>${row(b)}${row(lo)}</msub>`;}
        else if(s[i]==='^'){i++; const hi=grp(); sp(); if(s[i]==='_'){i++; const lo=grp(); b=`<msubsup>${row(b)}${row(lo)}${row(hi)}</msubsup>`;} else b=`<msup>${row(b)}${row(hi)}</msup>`;}
        else break;}
      // a thin space between a function name and its argument, as TeX sets it (\log p, \log \sigma(x), \max_i x_i); none before ( or [
      if(isFn){sp(); const c=s[i]; if(c!==undefined&&(/[A-Za-z0-9α-ωΑ-Ω]/.test(c)||(c==='\\'&&!/^\\(left|right|big|Big|bigg|Bigg|[,;:! ]|cdot|times|div|quad)/.test(s.slice(i))))) b+='<mspace width="0.167em"/>';}
      out+=b;}
    return out;}
  return `<math class="tex" display="inline" displaystyle="true">${row(seq(undefined))}</math>`;}
/* R6.18: a big number in K / M / B, with no hover that expands it to the exact value (the owner, 2026-10-07: “不需要有鼠标悬浮，把那个
   精确的数字展开给读者”); a lab that needs the exact value writes it out in its “Show the math” */
const BIG=n=>`<span class="num-big">${fmtBig(n)}</span>`;
/* R3.12: the guided tour ends in Scenarios, Scenarios ends in the challenge */
const toScenariosHTML=()=>`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="sandbox">${tx('See the scenarios','去看应用场景')}</button></div>`;
const toChallengeHTML=()=>`<div class="stepnav"><button type="button" class="btn sm" data-mode-go="challenge">${tx('Take the challenge','去做挑战')}</button></div>`;
/* the home hero: a lab can put its mini experiment there; the KV cache meter is the fallback */
let HERO=null;
let heroEntry=false;
/* the hero lab's button sits at the bottom right of its card (the owner, 2026-10-11); the card's content is redrawn by the lab, so the
   button is put back in its own footer row each time */
function heroCtaUpdate(){const a=heroCtaNode(), el=$('#heroLab'); if(!HERO||!a) return; a.removeAttribute('data-i18n-html'); a.href='#'+HERO.id; a.innerHTML=esc(HERO.cta())+' <span aria-hidden="true">→</span>';
  if(el&&!el.hidden){let f=el.querySelector(':scope>.hero-foot'); if(!f){f=document.createElement('div'); f.className='hero-foot'; el.appendChild(f);} if(a.parentNode!==f) f.appendChild(a);}}
function heroRefresh(){if(!HERO){heroUpdate(); return;} const el=$('#heroLab'); if(!el) return;
  // HERO.wide: the card spans the page width under the title (review 9: a left-to-right animated flow)
  const hs=el.closest('.hero'); if(hs) hs.classList.toggle('hero-wide',!!HERO.wide);
  if(el.hidden){$('#heroKV').hidden=true; el.hidden=false; HERO.mount(el);} else if(HERO.render) HERO.render(el);
  heroCtaUpdate();}
/* the KV cache memory lab's own model sentences after the kit text */
const QHm=m=>QH_TIP()+tx(` ${m.name} has ${m.heads} query heads per layer.`,`${m.name} 每层有 ${m.heads} 个 query head。`);
const GQAm=m=>GQA_TIP()+tx(` ${m.name} has ${m.heads} query heads in ${m.kvHeads} groups of ${m.heads/m.kvHeads}, so each token stores ${m.kvHeads} keys and ${m.kvHeads} values per layer instead of ${m.heads} of each.`,
                           `${m.name} 的 ${m.heads} 个 query head 分 ${m.kvHeads} 组、每组 ${m.heads/m.kvHeads} 个，所以每个 token 每层只存 ${m.kvHeads} 个 key 和 ${m.kvHeads} 个 value，而不是各 ${m.heads} 个。`);
const TP70=()=>TPMLA_TIP()+tx(' Here Llama-3.1-70B’s 8 KV heads on 8 GPUs give each GPU one, 1/8 of the total.','这里 Llama-3.1-70B 有 8 个 KV head，8 张 GPU 正好每张存一个，即总量的 1/8。');
const TPPLAIN=()=>TP70()+tx(' “Plain” means every layer is split across the GPUs, attention by head as above; the alternative keeps the other layers split but runs attention data-parallel.','“纯”指每一层都拆到各张 GPU 上，attention 按上面说的按 head 拆；另一种做法是其余各层照样拆，attention 改做数据并行。');
/* a model's note with its first "query head(s)", "KV head(s)" and "MLA" defined (R2.8) */
const attnOptTip=(v,m)=>v==='mla'?MLA_TIP():v==='mha'
  ?tx(`Multi-head attention (MHA): every query head has its own KV head, ${m.heads} here. ${m.name} does not use it; it is a what-if.`,`多头注意力（multi-head attention，MHA）：每个 query head 各有自己的 KV head，这里是 ${m.heads} 个。${m.name} 没有这样做，这是假设的设计。`)
  :tx(`Grouped-query attention (GQA): the ${m.heads} query heads form ${m.kvHeads} groups, and each group shares one KV head. ${m.name} uses it.`,`grouped-query attention（GQA）：${m.heads} 个 query head 分成 ${m.kvHeads} 组，每组共用一个 KV head。${m.name} 用的就是它。`);
const noteHTML=m=>{const seen=new Set(), tip={q:()=>QHm(m),k:()=>KVH_TIP(),m:()=>MLA_TIP()};
  const exact=Number.isInteger(m.params)&&m.params%1e6!==0;   // Llama counts are exact; the MoE totals are rounded figures
  return (exact?esc(L(m.note)).replace(fmtBig(m.params),BIG(m.params)):esc(L(m.note))).replace(/query heads?|KV heads?|\bMLA\b/g,x=>{const t=x[0]==='q'?'q':(x[0]==='K'?'k':'m'); if(seen.has(t)) return x; seen.add(t); return DEF(x,tip[t]());});};
const SHOWME_TIP=()=>tx('Does this part for you: sets what the card asks for, so you can see the result.','替你完成这一部分：直接设好卡片要求的设置，让你看到结果。');
function mathLines(r){
  const s=r.s,m=r.m,c=chain(r), T=(k,x,term)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}"${term?` data-term="${term}"`:''}>${x}</span>`;
  const b=PREC[s.kp], bytes=T('kp',tx(`${b} byte${b===1?'':'s'}`,`${b} 字节`)), byteSum=tx(`${fmtInt(r.kvPerToken)} bytes`,`${fmtInt(r.kvPerToken)} 字节`);
  const layers=T('model',tx(m.layers+' layers',m.layers+' 层'),'layers'), two=tx('2 (a key and a value)','2（key 和 value）');
  const l1=r.mla
    ?`${tx('KV cache per token','每个 token 的 KV cache')} = ${layers} × (${T('model',m.kvLora+' latent','latent')} + ${T('model',m.ropeDim+' RoPE','rope')}) × ${bytes} = ${byteSum} = <b>${c.tokS}</b>`
    :`${tx('KV cache per token','每个 token 的 KV cache')} = ${two}${tx(' ','')}× ${layers} × ${T('attn',tx(r.kvHeads+' KV heads',r.kvHeads+' 个 KV head'))} × ${T('model',m.headDim+' head_dim','headdim')} × ${bytes} = ${byteSum} = <b>${c.tokS}</b>`;
  const l2=`${tx('KV cache per user','每个用户的 KV cache')} = ${c.tokS} × ${T('ctx',tokTxt(fmtInt(s.ctx)))} = <b>${c.puS}</b>`;
  const l3=`${tx('KV cache, all users','全部用户的 KV cache')} = ${c.puS} × ${T('users',usersTxt(s.users))} = <b>${c.totS}</b>`;
  const kvPart=r.mla?(s.n>1?tx(`KV cache ${c.totS} (a full copy on each GPU)`,`KV cache ${c.totS}（每张卡一整份）`):`KV cache ${c.totS}`):(r.shard>1?`KV cache ${c.totS} ÷ ${r.shard}`:`KV cache ${c.totS}`);
  const l4=`${tx('Needed per GPU','每张 GPU 需要')} = ${T('wp',tx('weights ','权重 ')+gibW(c.W))} ÷ ${T('n',gpusTxt(s.n))} + ${kvPart} + ${T('oh',tx(gib1(c.oh)+' overhead','开销 '+gib1(c.oh)))} = <b>${gibW(c.need)}</b>`;
  const l5=`${tx('Usable per GPU','每张 GPU 可用')} = ${T('gpu',r.g.mem+' GiB')} × ${T('util',s.util.toFixed(2))} = <b>${gib1(c.use)}</b> → `+
    (r.fits?`<span class="okc">${tx(`fits, ${marginAmt(c)} free`,`装得下，还剩 ${marginAmt(c)}`)}</span>`:`<span class="badc">${tx(`short by ${marginAmt(c)}`,`还差 ${marginAmt(c)}`)}</span>`);
  return [l1,l2,l3,l4,l5];
}

/* ---------- controls ---------- */
const nearest=(arr,v)=>{let bi=0,bd=Infinity;arr.forEach((a,i)=>{const d=Math.abs(Math.log(a)-Math.log(v));if(d<bd){bd=d;bi=i;}});return bi;};
function setRange(el,v){el.value=v;setFill(el);}
function controlsInit(){
  $('#model').innerHTML=Object.entries(MODELS).map(([k,m])=>`<option value="${k}">${m.name}</option>`).join('');
  $('#gpu').innerHTML=Object.entries(GPUS).map(([k,g])=>`<option value="${k}">${g.name}</option>`).join('');
  $('#model').addEventListener('change',e=>{const m=MODELS[e.target.value],p={model:e.target.value}; if(m.defaultWp&&S.wp==='bf16') p.wp=m.defaultWp; learnerSet(p);});
  $('#gpu').addEventListener('change',e=>learnerSet({gpu:e.target.value}));
  $('#ctx').addEventListener('input',e=>learnerSet({ctx:CTX[+e.target.value]}));
  $('#users').addEventListener('input',e=>learnerSet({users:USERS[+e.target.value]}));
  $('#util').addEventListener('input',e=>learnerSet({util:+e.target.value/100}));
  $('#oh').addEventListener('input',e=>learnerSet({overhead:+e.target.value/2}));
  $$('.seg-ctl[data-key]').forEach(seg=>seg.addEventListener('click',e=>{const b=e.target.closest('button'); if(!b||b.disabled) return;
    const k=seg.dataset.key; learnerSet({[k]:k==='n'?+b.dataset.v:b.dataset.v});}));
  $('#resetBtn').addEventListener('click',()=>{ if(mode==='challenge') learnerSet({wp:'bf16',kp:'bf16'}); else learnerSet({...DEFAULT}); });
}
function syncControls(r){
  $('#model').value=S.model; $('#gpu').value=S.gpu;
  setRange($('#ctx'),nearest(CTX,S.ctx)); setRange($('#users'),nearest(USERS,S.users));
  setRange($('#util'),Math.round(S.util*100)); setRange($('#oh'),Math.round(S.overhead*2));
  $('#ctxOut').textContent=tokTxt(fmtTok(S.ctx));
  $('#usersOut').textContent=usersTxt(S.users);
  $('#utilOut').textContent=S.util.toFixed(2);
  $('#ohOut').textContent=fmtNum(S.overhead,1)+' GiB';
  $$('.seg-ctl[data-key]').forEach(seg=>{const k=seg.dataset.key; $$('button',seg).forEach(b=>{const v=k==='n'?+b.dataset.v:b.dataset.v; b.setAttribute('aria-pressed',String(S[k]===v));});});
  // challenge: controls that cannot change are hidden, not grayed out; a note says what is fixed
  const lock=mode==='challenge';
  $('#controls').classList.toggle('locked',lock);
  const ln=$('#lockNote'); ln.hidden=!lock;
  if(lock) ln.textContent=NB(tx('In the challenge you can change only weight precision and KV cache precision. Everything else is fixed: Llama-3.1-70B on 8 × H100 80GB, with 128K of context per user.',
                             '挑战中只能改权重精度和 KV cache 精度。其余设置固定：8 × H100 80GB 跑 Llama-3.1-70B，每个用户 128K 上下文。'));
  ['#model','#gpu','#ctx','#users','#util','#oh'].forEach(s=>{$(s).disabled=lock;});
  $$('#ctl-n button').forEach(b=>{b.disabled=lock;});
  $$('#ctl-attn button').forEach(b=>{b.disabled=lock||(b.dataset.v==='mha'&&r.mla); const v=b.dataset.v==='gqa'&&r.mla?'mla':b.dataset.v; b.dataset.tipx=attnOptTip(v,r.m); if(b.dataset.v==='gqa') b.textContent=r.mla?'MLA':t('attnNative');});
  // the words query head, KV head, GQA and MLA appear here from part 1 on, so they carry their definitions (R2.8)
  $('#attnNote').innerHTML=r.mla
    ?tx(`Per token and layer, ${DEF('MLA',MLA_TIP())} stores one compressed vector (latent) of ${r.m.kvLora} numbers plus a ${r.m.ropeDim}-number key that carries the position (RoPE), instead of a key and a value per head.`,`${DEF('MLA',MLA_TIP())} 每个 token、每层只存一个 ${r.m.kvLora} 维的压缩向量（latent），加一个 ${r.m.ropeDim} 维、带位置信息（RoPE）的 key，不再为每个 head 存 key 和 value。`)
    :(S.attn==='mha'?tx(`${DEF('Multi-head attention (MHA)',TERMS.mha())}, a what-if: ${r.m.heads} ${DEF('KV heads',KVH_TIP())}, one per ${DEF('query head',QHm(r.m))}.`,`${DEF('多头注意力（MHA）',TERMS.mha())}，这是假设的设计：${r.m.heads} 个 ${DEF('KV head',KVH_TIP())}，每个 ${DEF('query head',QHm(r.m))} 一个。`)
                   :tx(`${r.m.heads} ${DEF('query heads',QHm(r.m))} share ${r.m.kvHeads} ${DEF('KV heads',KVH_TIP())}; this is ${DEF('GQA',GQAm(r.m))} (the lab ${labLink('gqa-mla')} explains it from scratch).`,`${r.m.heads} 个 ${DEF('query head',QHm(r.m))} 共用 ${r.m.kvHeads} 个 ${DEF('KV head',KVH_TIP())}，这叫 ${DEF('GQA',GQAm(r.m))}（${labLink('gqa-mla')} 那个 Lab 从头讲）。`));
  $('#modelNote').innerHTML=noteHTML(r.m);
  const w=$('#ctxWarn'); w.hidden=!r.overCtx;
  if(r.overCtx) w.textContent=NB(tx(`This is beyond ${r.m.name}’s context window of ${fmtTok(r.m.maxCtx)} tokens, so it is hypothetical.`,`这超出了 ${r.m.name} 最长 ${fmtTok(r.m.maxCtx)} 个 token 的上下文窗口，只是假设。`));
}

/* ---------- stage ---------- */
const rigTxt=s=>NB(tx(`${MODELS[s.model].name} on ${s.n} × ${GPUS[s.gpu].name}`,`${s.n} × ${GPUS[s.gpu].name} 跑 ${MODELS[s.model].name}`));
const setupTxt=(s,most)=>{const mha=s.attn==='mha'&&MODELS[s.model].attn!=='mla', u=usersTxt(s.users)+(most?tx(' (the most that fit)','（能装下的最多人数）'):'');
  return tx(`${u} with ${fmtTok(s.ctx)} of context each, weights in ${PL[s.wp]}, KV cache in ${PL[s.kp]}${mha?', MHA':''}`,
            `${u}，每人 ${fmtTok(s.ctx)} 上下文，权重 ${PL[s.wp]}，KV cache ${PL[s.kp]}${mha?'，MHA':''}`);};
const pinLabel=s=>tx(`A: ${rigTxt(s)}, ${setupTxt(s)}`,`A：${rigTxt(s)}，${setupTxt(s)}`);
function drawStage(r){
  const c=chain(r), amt=marginAmt(c), n=S.n;
  $('#stTitle').textContent=rigTxt(S);
  $('#stSub').textContent=setupTxt(S,mode==='challenge');
  const st=$('#stStatus'); st.className='status '+(r.fits?'ok':'bad');
  st.textContent=r.fits?tx(`Fits: ${amt} free per GPU`,`装得下：每卡还剩 ${amt}`)
    :(!r.weightsFit?tx('Out of memory: the weights alone do not fit','显存不足：光权重就放不下'):tx(`Out of memory: ${amt} short per GPU`,`显存不足：每卡还差 ${amt}`));
  // the "over capacity" tooltip: when not even one user fits, a real server refuses to start instead of queueing (R6.10)
  const ov=$('#view-lab .legend-row [data-tip="tipLgOver"]'); if(ov) ov.dataset.tipx=t(r.maxUsers>0?'tipLgOver':'tipLgOver0');
  const scale=pinned?Math.max(r.cap,pinned.r.cap):r.cap;
  drawBar($('#pgBar'),perGpuBar(r),{perUser:r.kvPerGpu/S.users,ghost:true,scaleCap:scale,fmt:fmtKV,wfmt:x=>x>=GiB?gibW(r1(x/GiB)):fmtKV(x),useFmt:()=>gibW(c.use),
    overText:tx(`${amt} over`,`超出 ${amt}`),overTextCut:tx(`${amt} over\n(cut off)`,`超出 ${amt}\n（没画完）`)});
  $('#pgLabel').innerHTML=(n>1?tx(`Memory of each GPU (${n} in total, in ${DEF('GiB',TERMS.gib())})`,`每张 GPU 的显存（共 ${n} 张，单位 ${DEF('GiB',TERMS.gib())}）`):tx(`Memory of the GPU (in ${DEF('GiB',TERMS.gib())})`,`这张 GPU 的显存（单位 ${DEF('GiB',TERMS.gib())}）`))
    +($('#pgBar')._ticks?tx('. Each amber slice is one user.','。琥珀色里每一段是一个用户。'):'');
  drawAxis($('#pgAxis'),r.cap,{scaleCap:scale});
  $('#pinWrap').hidden=!pinned;
  if(pinned){$('#pinLabel').textContent=pinLabel(pinned.s); drawBar($('#pinBar'),perGpuBar(pinned.r),{noText:true,scaleCap:scale});}
  $('#pinBtn').textContent=pinned?tx('Re-pin A','重新固定 A'):tx('Pin as A','固定为 A');
  // the weights, all users' KV cache (a full copy on every GPU with MLA) and n × overhead, as Show the math shows them (R6.6)
  const ag=aggBar(r), needAll=gibW(dfix(c.W+(r.mla?c.tot*n:c.tot)+c.oh*n,1)), useAll=gibW(dfix(c.use*n,1));
  $('#aggLabel').textContent=n>1?tx(`All ${n} GPUs together: ${needAll} needed, ${useAll} usable`,`${n} 张 GPU 合计：需要 ${needAll}，可用 ${useAll}`)
                                :tx(`In total: ${needAll} needed, ${useAll} usable`,`合计：需要 ${needAll}，可用 ${useAll}`);
  drawBar($('#aggBar'),ag,{noText:true});
  let note;
  if(n===1) note=tx('With one GPU, all the weights and every user’s KV cache must go on it.','只有一张 GPU：全部权重和每个用户的 KV cache 都要放在这一张上。');
  else if(r.mla) note=tx(`Tensor parallelism: each GPU holds 1/${n} of the weights. MLA’s compressed vector is not split by head, so every GPU stores a full copy for every user, ${n} copies in all.`,`张量并行：每张 GPU 存 1/${n} 的权重。MLA 的压缩向量不按 head 分，每张 GPU 都要存每个用户的一整份，一共存了 ${n} 份。`);
  else note=tx(`Tensor parallelism: the ${n} GPUs split the weights and every user’s KV cache evenly, 1/${n} on each.`,`张量并行：${n} 张 GPU 平分权重和每个用户的 KV cache，每张存 1/${n}。`);
  if(r.overCtx) note+=tx(` ${r.m.name} supports at most ${fmtTok(r.m.maxCtx)} tokens of context, so longer contexts here are hypothetical.`,`${r.m.name} 最多支持 ${fmtTok(r.m.maxCtx)} 个 token 的上下文，更长的上下文在这里只是假设。`);
  $('#stNote').textContent=NB(note);
}
const MET=[
 {k:'kvPerToken',tip:'mTipTok',label:()=>tx('KV cache per token','每个 token 的 KV cache'),v:(r,c)=>c.tokS,
  sub:r=>r.mla?tx(`${r.m.layers} layers, one compressed vector in each`,`${r.m.layers} 层，每层一个压缩向量`):tx(`${r.m.layers} layers, ${r.kvHeads} ${DEF('KV heads',KVH_TIP())} in each`,`${r.m.layers} 层，每层 ${r.kvHeads} 个 ${DEF('KV head',KVH_TIP())}`)},
 {k:'kvPerReq',tip:'mTipReq',label:()=>tx('KV cache per user','每个用户的 KV cache'),v:(r,c)=>c.puS,sub:r=>tx(`with ${fmtTok(r.s.ctx)} of context`,`每人 ${fmtTok(r.s.ctx)} 上下文时`)},
 {k:'kvTotal',tip:'mTipAll',label:()=>tx('KV cache, all users','全部用户的 KV cache'),v:(r,c)=>c.totS,
  sub:r=>tx(`${usersTxt(r.s.users)} together`,`${usersTxt(r.s.users)}合计`)+(r.mla&&r.n>1?tx('; every GPU stores a full copy','，每张 GPU 各存一整份'):'')},
 {k:'maxUsers',tip:'mTipUsers',label:()=>tx('Most users that fit','最多用户数'),v:r=>fmtInt(r.maxUsers),sub:r=>tx(`at ${fmtTok(r.s.ctx)} of context each`,`每人 ${fmtTok(r.s.ctx)} 上下文时`)},
 {k:'maxCtx',tip:'mTipCtx',label:()=>tx('Longest context per user','每人最长上下文'),v:r=>fmtInt(r.maxCtx),
  // beyond the model's own window the number is a memory limit only (R6.10)
  sub:r=>{const over=r.maxCtx>r.m.maxCtx; return tx(`tokens, with ${usersTxt(r.s.users)}${over?` (memory limit; the model stops at ${fmtInt(r.m.maxCtx)})`:''}`,
                                                   `个 token（${usersTxt(r.s.users)}时${over?`；这是显存上限，模型最多支持 ${fmtInt(r.m.maxCtx)} 个`:''}）`);}}
];
/* the prediction in part 2 asks where memory runs out, so the longest context waits until part 3 finds it (R3.8) */
const hideAnswer=k=>k==='maxCtx'&&view==='lab'&&mode==='guided'&&(step<2||(step===2&&!observed[2]));
const hideNote=()=>step===2?tx('shown once you find the break point','找到临界点后显示'):(pred&&step===1?tx('shown in the next part','下一部分显示'):tx('shown after you predict','预测后显示'));
/* ▲▼ badges: only after the learner's own change (a setting, an option, a scenario), computed from the numbers as shown,
   and cleared when that number did not change; entering a part, a mode switch or a re-render never shows one (R6.6, R6.10) */
let userAct=false; const shownPrev={};
const learnerSet=p=>{userAct=true; setState(p);};
const shownNum=v=>{const m=String(v).match(/^([\d,]+(?:\.\d+)?)\s*(KiB|MiB|GiB|B)?/); return m?parseFloat(m[1].replace(/,/g,''))*({B:1,KiB,MiB,GiB}[m[2]]||1):NaN;};
function drawMetrics(r){
  const host=$('#metrics');
  if(!host.children.length) host.innerHTML=MET.map(m=>`<div class="metric" data-k="${m.k}" data-tip="${m.tip}"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');
  const c=chain(r);
  MET.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), hid=hideAnswer(m.k), v=hid?'?':m.v(r,c), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label(); el.querySelector('.v').textContent=v; el.querySelector('.sub').innerHTML=hid?esc(hideNote()):m.sub(r);
    const dt=userAct&&!hid&&shownPrev[m.k]!=null?deltaVs(shownNum(shownPrev[m.k]),shownNum(v)):'';
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=hid?null:v;});
  fitMetrics();
}
/* long values (68.625 KiB, 20,480 GiB) shrink together until each fits its card, never below 11 px (R7.2, R7.4) */
function fitMetrics(){
  const vs=$$('#metrics .metric .v'); if(!vs.length||!vs[0].clientWidth) return;
  vs.forEach(e=>{e.style.fontSize='';});
  let k=1; vs.forEach(e=>{if(e.scrollWidth>e.clientWidth+0.5) k=Math.min(k,e.clientWidth/e.scrollWidth);});
  if(k<1){const f=parseFloat(getComputedStyle(vs[0]).fontSize); vs.forEach(e=>{e.style.fontSize=Math.max(11,Math.floor(f*k*10)/10)+'px';});}
}
let fitRaf=0; window.addEventListener('resize',()=>{if(!fitRaf) fitRaf=requestAnimationFrame(()=>{fitRaf=0; fitMetrics();});});

/* ---------- rail: guided, sandbox, challenge ---------- */
function railGuided(){
  // R08: ✓ only for parts answered or without a question; Next says "Skip question" while this part's question is open and Next is on
  const s0=STEPS[step], body=s0.body(), q=partQ(body), a=partLog('kv-cache',STEPS.length,step,q),
    nd=(!!s0.expect&&!observed[step])||(!!s0.gate&&!s0.gate()), sq=step<STEPS.length-1&&q.open&&!nd;
  const items=STEPS.map((s,i)=>{
    const cls=stepCls(a,i,step);
    const head=stepHeadHTML(a,i,step,L(s.title));
    if(i!==step) return `<li class="step ${cls}">${head}</li>`;
    const nav=`<div class="stepnav">${step>0?`<button type="button" class="btn ghost sm" data-nav="back">${tx('Back','上一部分')}</button>`:''}${s.showMe?`<button type="button" class="btn ghost sm" data-showme="1" data-tipx="${esc(SHOWME_TIP())}">${tx('Show me','演示给我看')}</button>`:''}${step<STEPS.length-1?`<button type="button" class="btn${sq?' ghost':''} sm" id="nextBtn" data-nav="next">${nextTxt(sq)}</button>`:''}${s.skip&&!sq?`<button type="button" class="linkbtn" data-nav="skip">${tx('Skip','跳过')}</button>`:''}</div>`;
    return `<li class="step ${cls}">${head}<div class="step-b">${body}<div class="obs" data-obs hidden></div>${nav}</div></li>`;
  }).join('');
  const pb=pbarHTML({steps:STEPS.map(s=>({title:L(s.title),lead:s.lead?L(s.lead):''})),current:step,back:step>0,next:step<STEPS.length-1,nextDisabled:nd,ans:a,q});
  return `<div class="rail-top"><span class="eyebrow">${t('modeGuided')}${tx(`: part ${step+1} of ${STEPS.length}`,`：第 ${step+1} / ${STEPS.length} 部分`)}</span></div><ol class="steps">${items}</ol>${pb}`+(step===STEPS.length-1?nextLabHTML('kv-cache'):'');
}
function railSandbox(){
  const cur=presetNow(), J=(k,txt)=>`<button type="button" class="linkbtn" data-jump="${k}">${txt}</button>`;
  return `<div class="railcard"><div class="eyebrow">${t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
    `<p class="small">${tx('Pick a scenario and see how each GPU’s memory is split and how many users fit.','选一个场景，看看每张卡的显存怎么分，能装下多少个用户。')}</p>`+
    `<div class="presets" role="group" aria-label="${tx('Scenarios, pick one','应用场景，选一个')}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${L(p.t)}</b><span>${L(p.d)}</span></button>`).join('')}</div>`+
    `<p class="small muted" id="sbNone"${cur<0?'':' hidden'}>${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`+
    `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${toChallengeHTML()}</div>`+nextLabHTML('kv-cache');
}
/* the capability layer (Codex review 1 #5, Batch 2): three ways to show the lab transfers, after the challenge's own task */
function kvGo(p){setMode('sandbox'); Object.assign(S,{...DEFAULT,...p}); renderRail(); render(); revealTop($('#stage'));}
function KV_EXITS(){const r8=compute({...DEFAULT,model:'llama31-8b',n:1,ctx:32768,users:1}), c8=chain(r8), rm=compute({...DEFAULT,model:'mixtral-8x7b',ctx:32768,users:100}),
  r70=compute({...DEFAULT,ctx:131072,users:1}), kvReq=fmtKV(r70.kvPerReq), kvGpu=fmtKV(r70.kvPerReq/r70.shard), W=gibW(r70.wTotal/GiB),
  /* the new case's steps, from the values as shown (R6.6): 72 − 15 − 2 = 55 GiB left for the KV cache */
  w8=dfix(r8.wPerGpu/GiB,1), left8=gib1(dfix(c8.use-w8-r8.s.overhead,1)), mx=MODELS['mixtral-8x7b'];
 return {
 core:()=>chSubmitted, explain:[
  {q:tx('Llama‑3.1‑70B fits on 8 × H100 at short context, but at 128K only a few users fit. Why?','Llama‑3.1‑70B 在 8 × H100 上，短上下文放得下，到 128K 就只装得下几个用户。为什么？'),
   o:[tx('Every token keeps a key and a value in every layer, so the cache grows with each user’s context: at 128K it, not the weights, fills the GPUs.','每个 token 在每一层都要存一份 key 和 value，cache 随每个用户的上下文增长：到了 128K，占满显存的是它，不是权重。'),
     tx('The weights grow with the context, since a longer prompt makes the model load more of its parameters.','权重随上下文变大：prompt 越长，模型要载入的参数越多。'),
     tx('The attention scores of all earlier tokens are kept until the request ends, and at 128K they fill the GPUs.','前面所有 token 的注意力得分要一直存到请求结束，到了 128K，是它们占满了显存。')],a:0,
   n:[null,tx(`The weights take the same ${W} at any context. What grows with every token is the KV cache.`,`权重在任何上下文下都是 ${W}。随 token 数增长的是 KV cache。`),
     tx('Each step computes the scores and drops them. What stays for the rest of the request is each token’s key and value.','每一步算完注意力得分就丢掉，一直留到请求结束的是每个 token 的 key 和 value。')],
   why:tx(`Every token keeps a key and a value in every layer, so each user’s cache grows with their context, and every user has their own. At 128K one user’s KV cache is already ${kvReq}, while the weights, shared by all users, stay the same size.`,`每个 token 在每一层都要存一份 key 和 value，所以每个用户的 cache 随上下文增长，而且每个用户各有一份。128K 时，一个用户的 KV cache 就有 ${kvReq}，而所有用户共用的权重大小不变。`)},
  {q:tx('Llama‑3.1‑70B runs on 8 × H100 with tensor parallelism. Where is each user’s KV cache kept?','Llama‑3.1‑70B 用张量并行跑在 8 × H100 上。每个用户的 KV cache 存在哪里？'),
   o:[tx('Split by KV head: with 8 KV heads, each GPU keeps one head’s keys and values in every layer.','按 KV head 分：8 个 KV head，每张卡存其中一个 head 在每一层的 key 和 value。'),
     tx('A full copy on each GPU, so that whichever GPU runs the next step has all of it.','每张卡各存一整份，这样哪张卡算下一步，手上都有全部的 cache。'),
     tx('Split by layer: GPU 1 keeps layers 1 to 10, GPU 2 layers 11 to 20, and so on.','按层分：第 1 张卡存第 1 到 10 层，第 2 张卡存第 11 到 20 层，以此类推。')],a:0,
   n:[null,tx('With tensor parallelism each GPU computes only its own heads, so it needs only their keys and values. Each GPU keeps a full copy only when there are fewer KV heads than GPUs, or for MLA’s shared latent vector.','张量并行时每张卡只算自己的那几个 head，所以只需要这几个 head 的 key 和 value。只有 KV head 比 GPU 少，或者存的是 MLA 共用的压缩向量时，才每张卡各存一整份。'),
     tx('That is pipeline parallelism. With tensor parallelism every GPU holds a slice of every layer.','那是流水线并行。张量并行时，每张卡都存每一层的一部分。')],
   why:tx(`Each GPU computes only its own heads, so it keeps only those heads’ keys and values: an equal share of every user’s KV cache, ${kvGpu} per user at 128K.`,`每张卡只算自己的那几个 head，所以只存这些 head 的 key 和 value，也就是每个用户的 KV cache 里相等的一份：128K 时每人 ${kvGpu}。`)}],
 transfer:[
  {q:tx('Llama‑3.1‑8B on one H100 80GB, everything in BF16, 32K of context per user. How many users fit?','一张 H100 80GB 跑 Llama‑3.1‑8B，全部 BF16，每人 32K 上下文。能同时装下多少用户？'),
   o:[tx('1 to 5','1 到 5 个'),tx('6 to 20','6 到 20 个'),tx('21 to 50','21 到 50 个'),tx('More than 50','50 个以上')],a:r8.maxUsers<=5?0:(r8.maxUsers<=20?1:(r8.maxUsers<=50?2:3)),
   why:tx(`${r8.maxUsers} users. The GPU can use ${gib1(c8.use)}. The weights take ${fmtKV(r8.wPerGpu)} and the runtime overhead ${fmtKV(r8.oh)}, which leaves ${left8}. Each user’s 32K of context needs ${fmtKV(r8.kvPerReq)} of KV cache.`,
          `${r8.maxUsers} 个。这张 GPU 可用 ${gib1(c8.use)}。权重占 ${fmtKV(r8.wPerGpu)}，运行时开销 ${fmtKV(r8.oh)}，剩下 ${left8}。每个用户 32K 上下文要 ${fmtKV(r8.kvPerReq)} 的 KV cache。`),
   go:()=>kvGo({model:'llama31-8b',n:1,ctx:32768,users:r8.maxUsers}),goLabel:tx(`Open it in Scenarios: ${r8.maxUsers} users`,`在应用场景里打开：${r8.maxUsers} 个用户`)},
  {q:tx('Mixtral‑8x7B has 46.7B parameters, but the same attention shape as Llama‑3.1‑8B: 32 layers, 8 KV heads, head_dim 128. How much KV cache does it keep per token in BF16?','Mixtral‑8x7B 有 46.7B 个参数，但注意力的形状和 Llama‑3.1‑8B 一样：32 层、8 个 KV head、head_dim 128。BF16 下每个 token 要存多少 KV cache？'),
   o:[tx('128 KiB, as for Llama‑3.1‑8B','128 KiB，和 Llama‑3.1‑8B 一样'),tx('320 KiB, as for Llama‑3.1‑70B','320 KiB，和 Llama‑3.1‑70B 一样'),tx('About 750 KiB: 5.8 times the parameters','约 750 KiB：参数是 5.8 倍'),'2.5 MiB'],a:0,
   why:tx(`The KV cache depends only on the attention shape, and Mixtral‑8x7B’s is the same as Llama‑3.1‑8B’s. So it keeps 2 × ${mx.layers} × ${mx.kvHeads} × ${mx.headDim} × 2 bytes = ${fmtB(rm.kvPerToken)} per token. The mixture of experts adds its extra parameters to the MLP blocks, which the KV cache does not store.`,
          `KV cache 只看注意力的形状，而 Mixtral‑8x7B 的形状和 Llama‑3.1‑8B 一样。所以每个 token 是 2 × ${mx.layers} × ${mx.kvHeads} × ${mx.headDim} × 2 字节 = ${fmtB(rm.kvPerToken)}。混合专家多出来的参数都在 MLP 块里，KV cache 不存它们。`),
   go:()=>kvGo({model:'mixtral-8x7b',ctx:32768,users:100}),goLabel:tx('Open it in Scenarios: 100 users','在应用场景里打开：100 个用户')}],
 practice:[
  {id:'inference-batching',d:tx('Size a batched serving fleet: replicas at peak, KV cache bytes per token, a 500 ms p95 latency budget.','为一个批处理推理服务做容量规划：峰值要多少副本、每个 token 的 KV cache 字节数、500 ms 的 p95 延迟预算。')},
  {id:'inference-serving',d:tx('Design the serving system: GPU scheduling, batching, streaming and rate limits.','设计推理服务：GPU 调度、批处理、流式输出和限流。')}]};}
function railChallenge(){
  const segs=inlineSeg('wp',t('cWp'),[['bf16','BF16'],['fp8','FP8'],['fp4','FP4']])+inlineSeg('kp',t('cKp'),[['bf16','BF16'],['fp8','FP8']]);
  const submit=chSubmitted?'':`<button type="button" class="btn" id="chSubmit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${t('modeChallenge')}</div><h3 class="rh">${tx('Fit 20 users on 8 × H100 at 128K context each','让 8 张 H100 在 128K 上下文下同时装下 20 个用户')}</h3>`+
    `<p>${tx('The model is Llama-3.1-70B, and every user needs the full 128K context. You may change only these two settings; the number of users is always the most that fit:','模型是 Llama-3.1-70B，每个用户都要完整的 128K 上下文。你只能改下面这两项，用户数总是取能装下的最多人数：')}</p>${segs}`+
    `<div class="goal mono" id="chGoal"><span>${tx('Users that fit now','现在能同时装下')}</span><span><b id="chNow"></b> / 20 ${tx('users','个用户')}</span></div><div class="progress" id="chProg" aria-hidden="true"><i id="chBar"></i></div>${submit}<div id="chRes" aria-live="polite"></div></div>`+exitsHTML('kv-cache',KV_EXITS())+nextLabHTML('kv-cache');
}
function renderRail(){
  $('#rail').innerHTML=NB(mode==='guided'?railGuided():(mode==='sandbox'?railSandbox():railChallenge()));
  updateRail(compute(S));
}
function updateRail(r){
  // inline controls mirror the state, same as the Settings panel
  $$('#rail [data-bind]').forEach(el=>{const k=el.dataset.bind; setRange(el,nearest(k==='ctx'?CTX:USERS,S[k]));});
  $$('#rail [data-out]').forEach(el=>{el.textContent=el.dataset.out==='ctx'?tokTxt(fmtTok(S.ctx)):usersTxt(S.users);});
  $$('#rail [data-bind-seg]').forEach(g=>{const k=g.dataset.bindSeg;
    $$('button',g).forEach(b=>{b.setAttribute('aria-pressed',String(S[k]===b.dataset.v)); b.disabled=k==='attn'&&b.dataset.v==='mha'&&r.mla;});});
  if(mode==='guided'){
    const s=STEPS[step], box=$('#rail [data-obs]'); if(!box) return;
    // single-choice options: the pressed one follows the state, and tried ones keep their result
    const cur=leverNow();
    $$('#rail [data-lever]').forEach(bt=>{const i=+bt.dataset.lever; bt.setAttribute('aria-pressed',String(i===cur)); bt.querySelector('.lv-r').textContent=lvTried.has(LEVERS[i].id)?lvResult(i):'';});
    if(s.expect&&s.expect(r)) observed[step]=true;
    const show=!!s.obs&&(!s.expect||!!observed[step]);
    box.hidden=!show;
    if(show){const lab=typeof s.label==='function'?s.label():L(s.label||{en:'Observed',zh:'观察到'}), html=NB(`<div class="eyebrow${/\d/.test(lab)?' keepcase':''}">${lab}</div>`+s.obs(r));
      if(box._html!==html){const again=box._step===step&&box._html!=null; box.innerHTML=html; box._html=html; box._step=step;
        // a click that changes the result makes the result box blink once, so the change is noticed
        if(again&&s.levers&&!reduceMotion()){box.classList.remove('bump'); void box.offsetWidth; box.classList.add('bump');}}}
    const nb=$('#nextBtn'); if(nb) nb.disabled=(!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());
    $$('#rail .pbar [data-nav="next"]').forEach(b=>{b.disabled=!!(nb&&nb.disabled);});
    const sm=$('#rail [data-showme]'); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('#rail [data-preset]').forEach(bt=>bt.setAttribute('aria-pressed',String(+bt.dataset.preset===cur)));
    const nn=$('#sbNone'); if(nn) nn.hidden=cur>=0;
  } else if(mode==='challenge'){
    const n=$('#chNow'), met=S.users>=20; if(!n) return;
    n.textContent=S.users; $('#chBar').style.width=Math.min(100,S.users/20*100)+'%';
    $('#chGoal').classList.toggle('met',met); $('#chProg').classList.toggle('met',met);
    if(chSubmitted&&met) chReveal=true;
    $('#chRes').innerHTML=NB(chResultHTML());
  }
}
function clearCues(){$$('.cue').forEach(el=>el.classList.remove('cue'));}
function cueTargets(){clearCues(); if(mode!=='guided') return;
  (STEPS[step].target||[]).forEach(id=>{const el=document.getElementById(id); if(el){void el.offsetWidth; el.classList.add('cue');}});}
function enterStep(i){
  step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step];
  if(s.enter) Object.assign(S,s.enter());
  if(s.openMath) $('#math').open=true;
  renderRail(); render(); cueTargets();
}
/* challenge result: plain comparison bars, the initial setup first, then one change, then two */
const COMBOS=[['bf16','bf16'],['bf16','fp8'],['fp8','bf16'],['fp4','bf16'],['fp8','fp8'],['fp4','fp8']];
const nChanged=(wp,kp)=>(wp!=='bf16'?1:0)+(kp!=='bf16'?1:0);
const cuts=(wp,kp)=>(wp==='bf16'?0:(wp==='fp8'?1:2))+(kp==='fp8'?1:0);
const chPoints=()=>COMBOS.map(([wp,kp])=>({wp,kp,n:nChanged(wp,kp),c:cuts(wp,kp),u:compute({...CH,users:1,wp,kp}).maxUsers}));
const comboTxt=p=>tx(`weights ${PL[p.wp]} + KV ${PL[p.kp]}`,`权重 ${PL[p.wp]} + KV ${PL[p.kp]}`);
function submitChallenge(){
  chSubmitted=true; if(S.users>=20) chReveal=true;
  renderRail();
  const res=$('#chRes'); if(res) res.scrollIntoView({block:'nearest',behavior:reduceMotion()?'auto':'smooth'});
}
function chResultHTML(){
  if(!chSubmitted) return '';
  const pts=chPoints(), me=pts.find(p=>p.wp===S.wp&&p.kp===S.kp);
  const best=pts.filter(p=>p.u>=20).sort((a,b)=>a.n-b.n||a.c-b.c||b.u-a.u)[0];
  let h;
  if(me.u>=20&&me===best) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${me.u} users fit.</b> And you changed as little as possible.</p>`,
                              `<p class="okline" style="margin-top:12px"><b>达成目标：能装下 ${me.u} 个用户。</b>而且你的改动已经是最少的。</p>`);
  else if(me.u>=20) h=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: ${me.u} users fit.</b> You could change less: ${comboTxt(best)} already meets the goal.</p>`,
                         `<p class="okline" style="margin-top:12px"><b>达成目标：能装下 ${me.u} 个用户。</b>其实改得更少也行：${comboTxt(best)} 就已经达标。</p>`);
  else h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: only ${me.u} users fit.</b></p><p class="small">Hint: at 128K, which is bigger on each GPU, the weights or the KV cache? Compare them in the memory chart, then shrink the bigger one first.</p>`,
            `<p class="badline" style="margin-top:12px"><b>还没达到：只能装下 ${me.u} 个用户。</b></p><p class="small">提示：128K 时，每张卡上权重和 KV cache 哪个更大？在显存图里比一比，先压缩更大的那一块。</p>`)
          +(chReveal?'':`<button type="button" class="linkbtn" data-ch-reveal="1">${tx('Show the answer','直接看答案')}</button>`);
  if(!chReveal) return h;
  const at=(wp,kp)=>pts.find(p=>p.wp===wp&&p.kp===kp).u, base=at('bf16','bf16'), kv8=at('bf16','fp8'), w8=at('fp8','bf16');
  const rb=compute({...CH,users:base,wp:'bf16',kp:'bf16'}), perUser=rb.kvPerReq/rb.shard, kw=fmtNum(perUser*base/rb.wPerGpu,0);
  h+=cmpChart(pts,me);
  /* baseline → effect → reason (R6.1, R6.2) in words, one key number each; the chart above has every combination's number (R6.20) */
  const dbl=kv8/base>=1.8&&kv8/base<2.3, few=w8-base<=3;
  h+=tx(`<p class="small">Initial setup, everything in BF16: ${base} users fit. Putting only the KV cache in FP8 ${dbl?'about doubles that':'raises it a lot'}; putting only the weights in FP8 ${few?'adds just a few':'adds some'}.</p>`,
        `<p class="small">初始设置是全部 BF16：能装下 ${base} 个用户。只把 KV cache 改成 FP8，能装下的用户${dbl?'约翻一倍':'多了很多'}；只把权重改成 FP8，${few?'只多几个':'多了一些'}。</p>`);
  h+=tx(`<p class="small">Why the gap: at 128K the KV cache is much bigger than the weights, about ${kw} times as big on each GPU. Halving the bigger block frees more memory.</p>`,
        `<p class="small">差这么多的原因：128K 时，KV cache 比权重大得多，每张卡上约是权重的 ${kw} 倍。把更大的那一块减半，省下的显存才多。</p>`);
  h+=tx(`<p class="small muted">Note: this challenge only checks what fits in memory. It does not measure how lower precision affects answer quality. Before a real deployment, confirm on an eval set that quality has not dropped noticeably. The ${isLive('quantization')?labLink('quantization','Quantization'):'Quantization'} lab shows where the loss comes from.</p>`,
        `<p class="small muted">注意：这个挑战只算显存装不装得下，没有考虑降低精度对回答质量的影响。真实部署前，要在评测集上确认降精度后质量没有明显下降。${isLive('quantization')?labLink('quantization','量化'):'量化'} Lab 讲了损失从哪里来。</p>`);
  return h;
}
/* only the initial setup, the pick and the combinations that fit 20 users (the owner's review of batch 8, 2026-10-07; as A.ui.cmpChart brief) */
function cmpChart(all,me){
  const pts=all.filter(p=>p.n===0||p===me||p.u>=20), nOut=all.length-pts.length;
  const maxU=Math.max(30,Math.ceil(Math.max(...pts.map(p=>p.u))/5)*5), pct=u=>(u/maxU*100).toFixed(2)+'%';
  const rows=pts.map(p=>{
    const tags=(p.n===0?`<span class="cmp-tag">${tx('initial setup','初始设置')}</span>`:'')+(p===me?`<span class="cmp-you">${tx('your pick','你的选择')}</span>`:'');
    return `<div class="cmp-row${p.u>=20?' ok':''}${p===me?' me':''}"><div class="cmp-lab"><span>${comboTxt(p)}</span>${tags}</div>`+
      `<div class="cmp-track cmp-cq"><i style="width:${pct(p.u)}"></i><b class="g" style="left:${pct(20)}"></b><span class="cmp-val mono cmp-fit" style="left:${pct(p.u)};--v:${(p.u/maxU*100).toFixed(2)}">${p.u}</span></div></div>`;}).join('');
  return `<div class="cmp" role="img" aria-label="${esc(tx('Users that fit on 8 × H100 at 128K context each. ','8 × H100、每人 128K 上下文时能装下的用户数：')+pts.map(p=>comboTxt(p)+tx(': ','：')+p.u).join(tx('; ','；')))}">`+
    `<div class="cmp-title">${tx('Users that fit on 8 × H100 at 128K context each','8 × H100、每人 128K 上下文时，能装下的用户数')}</div>`+
    `<div class="cmp-bars" aria-hidden="true">${rows}<div class="cmp-axis cmp-gx"><span style="left:${pct(20)};--g:${(20/maxU*100).toFixed(2)}">${tx('goal: 20','目标 20')}</span></div></div>`+
    cmpLegend({goal:20,okText:tx('20 or more users fit','装得下 20 个或更多'),noText:tx('fewer than 20','不到 20 个'),goalText:tx('goal: 20 users','目标：20 个用户')})+`</div>`+
    (nOut?`<p class="small muted cmp-note">${tx(`The other ${nOut} combinations fit fewer than 20 users and are not shown.`,`其余 ${nOut} 种组合装不下 20 个用户，图里没有列出。`)}</p>`:'');
}
const MODENAME={guided:'modeGuided',sandbox:'modeSandbox',challenge:'modeChallenge'}, MODEDESC={guided:'descGuided',sandbox:'descSandbox',challenge:'descChallenge'};
function renderModes(){   // the KV lab's flow list (2026-10-11): drawn again on each mode change and language switch
  const nav=$('#kvFlow'); if(!nav) return;
  nav.innerHTML=flowListHTML('kv-cache',mode,{guided:t('descGuided'),sandbox:t('descSandbox'),challenge:t('descChallenge')});
  flowSync($('#view-lab'),mode); if(FLOW['kv-cache']!==undefined) flowDecorate($('#view-lab'));
}
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved};saved=null;}
  if(m==='challenge'){saved={...S};chSubmitted=false;chReveal=false;S={...S,...CH,wp:'bf16',kp:'bf16'};S.users=Math.max(1,compute({...S,users:1}).maxUsers);}
  renderModes();
  if(m==='guided'){enterStep(step);return;}
  clearCues(); renderRail(); render();
}

/* ---------- state, render, url token ---------- */
function setState(p){
  Object.assign(S,p);
  if(MODELS[S.model].attn==='mla') S.attn='gqa';
  if(mode==='challenge'){Object.assign(S,CH); S.users=Math.max(1,compute({...S,users:1}).maxUsers);}
  render();
}
function render(){
  const r=compute(S);
  if(view==='lab'&&mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect(r)) observed[step]=true;}
  $$('.linked').forEach(el=>el.classList.remove('linked'));
  syncControls(r); drawStage(r); drawMetrics(r);
  $('#mathBody').innerHTML=mathLines(r).map(x=>`<p>${x}</p>`).join('');
  updateRail(r); writeHash(); announce(r);
  if(!$('#dock').hidden) renderDock(r,true);
  dockSoon();
  prevR=r; userAct=false;
}

/* ---------- mini chart: when the memory bar scrolls out of view, a compact copy sticks under the top bar ---------- */
let dockRaf=0;
// which element must stay visible (anchor), which area counts as "working on this lab" (bench), where a click on the dock goes (target)
function dockCtx(){
  if(view==='lab') return {anchor:$('#pgBar'),bench:$('#view-lab .bench'),target:$('#stageGraph')};
  if(view==='dyn'&&dyn&&dyn.dock){const a=dyn.dock.anchor(); return {anchor:a,bench:dyn.dock.bench(),target:dyn.dock.target?dyn.dock.target():a};}
  return null;}
function barInView(a,dy){dy=dy||0; const b=a.getBoundingClientRect(); return b.height>0&&b.top-dy>=60&&b.bottom-dy<=window.innerHeight-8;}
// the mini result bar waits until the lab's title has scrolled under the top bar, so it never covers the breadcrumb or the title,
// at any width and whatever pushed the chart down (an open Fidelity popover did; site review R10 #1, #2). On phones, Next, Back and a
// mode switch scroll the new card up to the top bar (R3.7), so the title is gone and the bar shows while the learner works.
function benchInView(bn,dy){dy=dy||0; const b=bn.getBoundingClientRect(), bar=$('.topbar'), tl=$(view==='lab'?'#view-lab .labtitle':'#view-dyn .labtitle');
  if(tl&&tl.getBoundingClientRect().bottom-dy>(bar?bar.offsetHeight:58)-2) return false;
  return b.top-dy<window.innerHeight-120&&b.bottom-dy>200;}
/* the stage's title row (kit polish K4): on a desktop the stage sticks at 70 px, right where the bar sits (59 to about 108 px), so a bar
   shown while the stage was on the screen covered the stage's title and its status badge (product-design P15, at 1440x900 in part 8,
   Scenarios, Settings; backprop and product-design had made their anchor the whole stage while it was stuck to avoid it). The bar now
   stays away while any of the stage's title row is in the bar's band: the stage stuck or about to stick, or pushed up at the bench's end
   with its title row not yet under the top bar. It comes once that row has gone: under the top bar, or scrolled away inside a stuck stage
   whose result the reader has scrolled off (dpo's math at 900 and 1280 px), as before. On phones (no sticky stage) the same holds for a
   stage scrolled up to the top. The stage stays put (no shift of the sticky top, no smaller stage), so nothing on it moves */
const DOCK_H=50;   // the bar's height (min-height 48 px and its border), also when it is hidden
function dockStageTop(c,dy){dy=dy||0; const st=(c.anchor.closest&&c.anchor.closest('.stage'))||c.bench.querySelector(':scope>.stage'); if(!st||!st.getClientRects().length) return false;
  const hd=st.querySelector('.stage-head')||st, bar=$('.topbar'), bb=bar?bar.offsetHeight:58, h=hd.getBoundingClientRect();
  return h.height>0&&h.bottom-dy>bb+1&&h.top-dy<bb+DOCK_H;}   // some of the title row is in the bar's band, from the top bar's bottom to 50 px below it
// whether the bar shows, or would show once the page is scrolled by dy with nothing else moving (revealTop asks that on phones, K3)
function dockShows(c,dy){return !!(c&&c.anchor&&c.bench)&&c.bench.dataset.flow!=='primer'&&benchInView(c.bench,dy)&&!barInView(c.anchor,dy)&&!dockStageTop(c,dy);}   // no dock over the primer (2026-10-11)
function pulseDock(){const d=$('#dock'); if(reduceMotion()) return; d.classList.remove('pulse'); void d.offsetWidth; d.classList.add('pulse');}
function renderDockAny(pulse){
  if(view==='lab'){renderDock(compute(S),pulse); return;}
  if(view==='dyn'&&dyn&&dyn.dock){dyn.dock.render({status:$('#dkStatus'),vis:$('#dkVis'),read:$('#dkRead')}); if(pulse) pulseDock();}}
function renderDock(r,pulse){
  // in the challenge the users are always the most that fit, so "fits" says nothing: the chip shows the goal instead (R6.10)
  const ch=view==='lab'&&mode==='challenge', met=r.maxUsers>=20, st=$('#dkStatus');
  st.className='status '+((ch?met:r.fits)?'ok':'bad');
  st.textContent=ch?(met?tx('Goal met','达成目标'):tx('Not yet','还没达到')):(r.fits?tx('Fits','装得下'):tx('Out of memory','显存不足'));
  if(!$('#dkBar')) $('#dkVis').innerHTML='<div class="bar thin" id="dkBar"></div>';
  drawBar($('#dkBar'),perGpuBar(r),{noText:true});
  const c=chain(r), ctx=fmtTok(r.s.ctx), mem=tx(`<span class="dk-mem">; with ${usersTxt(r.s.users)}, each GPU needs ${gibW(c.need)} and can use ${gib1(c.use)}</span>`,`<span class="dk-mem">；${fmtInt(r.s.users)} 个用户时，每卡需要 ${gibW(c.need)}，可用 ${gib1(c.use)}</span>`);
  $('#dkRead').innerHTML=ch?tx(`At ${ctx} of context each, at most ${usersTxt(r.maxUsers)} ${r.maxUsers===1?'fits':'fit'} (goal: 20)`,`每人 ${ctx} 上下文时最多 ${fmtInt(r.maxUsers)} 个用户（目标 20 个）`)+mem
                              :(r.maxUsers>0?tx(`At ${ctx} of context each, at most ${usersTxt(r.maxUsers)} ${r.maxUsers===1?'fits':'fit'}`,`每人 ${ctx} 上下文时，最多 ${fmtInt(r.maxUsers)} 个用户`)
                                       :tx(`At ${ctx} of context each, not even one user fits`,`每人 ${ctx} 上下文时，一个用户也装不下`))+mem;
  if(pulse) pulseDock();
}
function updateDock(){
  const d=$('#dock'), show=dockShows(dockCtx());
  if(d.hidden!==show) return;
  d.hidden=!show; if(show) renderDockAny(false);
}
function dockSoon(){if(!dockRaf) dockRaf=requestAnimationFrame(()=>{dockRaf=0; updateDock();});}
let annT=null;
function announce(r){ if(view!=='lab') return; clearTimeout(annT);
  annT=setTimeout(()=>{$('#live').textContent=r.fits
    ?tx(`Fits. Each GPU needs ${fmtKV(r.need)} and can use ${fmtB(r.usable)}. At this context, at most ${r.maxUsers} users fit.`,`装得下。每张 GPU 需要 ${fmtKV(r.need)}，可用 ${fmtB(r.usable)}。这个上下文长度下最多 ${r.maxUsers} 个用户。`)
    :tx(`Out of memory. Each GPU needs ${fmtKV(r.need)}, ${fmtNum(r.ratio,1)} times the ${fmtB(r.usable)} it can use.`,`显存不足。每张 GPU 需要 ${fmtKV(r.need)}，是可用的 ${fmtB(r.usable)} 的 ${fmtNum(r.ratio,1)} 倍。`);},600);}
const token=()=>'kv-cache~'+[S.model,S.gpu,S.n,S.ctx,S.users,S.wp,S.kp,S.attn,Math.round(S.util*100),Math.round(S.overhead*10)].join('.');
/* the setup goes into the URL only in Scenarios, as A.hash.write does for the other labs (CONTRACT §9) */
function writeHash(){ if(view!=='lab') return; try{history.replaceState(null,'','#'+(mode==='sandbox'?token():'kv-cache'));}catch(e){} }
function parseToken(tok){
  const p=tok.split('.'); if(p.length!==10) return null;
  const [model,gpu,n,ctx,users,wp,kp,attn,util,oh]=p;
  const o={model,gpu,n:+n,ctx:+ctx,users:+users,wp,kp,attn,util:+util/100,overhead:+oh/10};
  if(!MODELS[model]||!GPUS[gpu]||!PREC[wp]||!(kp==='bf16'||kp==='fp8')||!(attn==='gqa'||attn==='mha')) return null;
  if(![1,2,4,8].includes(o.n)||!(o.ctx>=1&&o.ctx<=1048576)||!(o.users>=1&&o.users<=512)||!(o.util>=0.5&&o.util<=1)||!(o.overhead>=0&&o.overhead<=16)) return null;
  return o;
}

/* ---------- lab chrome ---------- */
const CTL={model:'ctl-model',gpu:'ctl-gpu',n:'ctl-n',ctx:'ctl-ctx',users:'ctl-users',wp:'ctl-wp',kp:'ctl-kp',attn:'ctl-attn',util:'ctl-util',oh:'ctl-oh'};
const CTLNAME={model:'cModel',gpu:'cGpu',n:'cN',ctx:'ctxLabel',users:'cUsers',wp:'cWp',kp:'cKp',attn:'cAttn',util:'cUtil',oh:'cOh'};
const TERM={
 model:{en:'The model sets this number: its layer count, head size and parameter count come from its config.json.',zh:'这个数由模型决定：层数、head 维度和参数量都来自模型的 config.json。'},
 attn:{en:'The attention design sets the number of KV heads. With GQA, several query heads share one KV head.',zh:'KV head 的数量由注意力设计决定。GQA 让几个 query head 共用一个 KV head。'},
 kp:{en:'Each number in the KV cache takes this many bytes: 2 in BF16, 1 in FP8.',zh:'KV cache 里每个数值占这么多字节：BF16 是 2，FP8 是 1。'},
 ctx:{en:'Each user keeps this many tokens of context.',zh:'每个用户保留这么多个 token 的上下文。'},
 users:{en:'The server serves this many users at the same time, and each one has their own KV cache.',zh:'服务端同时服务这么多用户，每个用户都有自己的一份 KV cache。'},
 wp:{en:'The weights take the parameter count times the bytes per parameter: 2 in BF16, 1 in FP8, 0.5 in FP4.',zh:'权重占的显存是参数量乘以每个参数的字节数：BF16 是 2，FP8 是 1，FP4 是 0.5。'},
 n:{en:'Tensor parallelism splits the weights and the KV heads evenly across these GPUs.',zh:'张量并行把权重和 KV head 平均分到这些 GPU 上。'},
 oh:{en:'The serving engine needs this memory besides the weights and the KV cache, for temporary activations, CUDA graphs and communication buffers. Real engines measure it at startup; here it is an estimate.',zh:'推理引擎在权重和 KV cache 之外还要这么多显存，用于临时激活、CUDA graph 和通信缓冲区。真实引擎在启动时实测它，这里用估计值。'},
 gpu:{en:'This is the memory of the chosen GPU; the vendor’s GB figure is read as GiB.',zh:'这是所选 GPU 的显存容量，厂商标的 GB 按 GiB 计。'},
 util:{en:'The server uses at most this share of each GPU’s memory and keeps the rest as a safety margin.',zh:'服务端最多只用每张卡显存的这个比例，剩下的留作安全余量。'}
};
const lockedCtl=c=>mode==='challenge'&&c!=='wp'&&c!=='kp';
// a control asked for inside the current card wins over the one in the Settings panel
const ctlEl=c=>document.getElementById('ic-'+c)||document.getElementById(CTL[c]);
function ctlValue(c){const r=compute(S);
  return {model:r.m.name,gpu:r.g.name,n:gpusTxt(S.n),ctx:tokTxt(fmtTok(S.ctx)),users:usersTxt(S.users),wp:PL[S.wp],kp:PL[S.kp],
          attn:r.mla?'MLA':t(S.attn==='mha'?'attnMha':'attnNative'),util:S.util.toFixed(2),oh:fmtNum(S.overhead,1)+' GiB'}[c];}
// what a click on something tied to control c will do; a fixed control says its value instead of doing nothing
function jumpAct(c){
  if(c==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(c==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(!lockedCtl(c)) return tx(`Click to jump to “${t(CTLNAME[c])}”.`,`点击跳到“${t(CTLNAME[c])}”。`);
  if(c==='users') return tx('In the challenge, the number of users is always the most that fit.','挑战中，用户数自动取能装下的最大值。');
  const name=t(CTLNAME[c]), val=ctlValue(c), sp=x=>/[A-Za-z0-9]/.test(x)?' ':'';
  const q=c==='attn'?tx(`“${val}”`,`“${val}”`):val;
  return tx(`Fixed during the challenge: “${name}” is ${q}.`,`挑战中固定不变：“${name}”是${c==='attn'?'':sp(val[0])}${q}。`);
}
function termText(c,term){const r=compute(S), m=r.m;
  if(term==='layers') return tx(`${m.name} has ${m.layers} layers; the count comes from its config.json.`,`${m.name} 有 ${m.layers} 层，层数来自它的 config.json。`);
  if(term==='headdim') return tx(`Each key or value vector holds ${m.headDim} numbers (head_dim), from the model’s config.json.`,`每个 key 或 value 向量是 ${m.headDim} 维（head_dim），来自模型的 config.json。`);
  if(term==='latent') return tx(`In each layer, MLA stores one compressed vector (latent) of ${m.kvLora} numbers per token instead of a key and a value per head.`,`MLA 在每一层为每个 token 存一个 ${m.kvLora} 维的压缩向量（latent），代替各个 head 的 key 和 value。`);
  if(term==='rope') return tx(`It also stores a ${m.ropeDim}-number key that carries the token’s position (RoPE, rotary position embedding).`,`另外还存一个 ${m.ropeDim} 维的 key，它带着 token 的位置信息（RoPE，旋转位置编码）。`);
  if(c==='n'&&r.mla) return tx('Tensor parallelism splits the weights evenly across these GPUs; MLA’s compressed vector is copied in full to every GPU.','张量并行把权重平均分到这些 GPU 上；MLA 的压缩向量每张 GPU 各存一整份。');
  return L(TERM[c]);}
function termTip(c,term){return `<div>${esc(termText(c,term))}</div><div class="act">${jumpAct(c)}</div>`;}
const AREA={math:'#math',controls:'#controls'};
function jumpTo(c,src){
  if(AREA[c]){const el=$(AREA[c]); if(c==='math') el.open=true; hideTip();
    el.scrollIntoView({block:'nearest',behavior:reduceMotion()?'auto':'smooth'});
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); clearTimeout(el._ft); el._ft=setTimeout(()=>el.classList.remove('flash'),1700); return;}
  if(lockedCtl(c)){if(src){tipFor=src; showTipHTML(src,`<div>${jumpAct(c)}</div>`);} return;}
  const el=ctlEl(c); if(!el) return;
  if(el.closest('#adv')) $('#adv').open=true;
  hideTip();
  el.scrollIntoView({block:'center',behavior:reduceMotion()?'auto':'smooth'});
  const f=el.querySelector('input,select,button[aria-pressed="true"]')||el.querySelector('button'); if(f) f.focus({preventScroll:true});
  el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
  clearTimeout(el._ft); el._ft=setTimeout(()=>el.classList.remove('flash'),1700);
}
// every jump link shows where it goes (R1.1): [data-jump] alone is enough for a tooltip
function tipTarget(n){if(!n||!n.closest||n.closest('#mapSvg')) return null; return n.closest('[data-tip],[data-tipx],[data-tipfn],.t,[data-jump]');}
function tipShow(el){
  tipFor=el;
  if(el.dataset.tipfn){const f=TIPFN[el.dataset.tipfn], h=f?f(el):''; if(h) showTipHTML(el,h,true); else hideTip(); return;}
  if(el.classList.contains('t')){
    if(view==='dyn'){const h=dyn&&dyn.termTip?dyn.termTip(el.dataset.ctl,el):null; if(h) showTipHTML(el,h); else hideTip(); return;}
    showTipHTML(el,NB(termTip(el.dataset.ctl,el.dataset.term))); const c=ctlEl(el.dataset.ctl); if(c) c.classList.add('linked'); return;}
  const body=el.dataset.tipx!=null?escMath(el.dataset.tipx):(el.dataset.tip?t(el.dataset.tip):'');
  let act=''; if(el.dataset.jump) act=view==='dyn'?(dyn&&dyn.jumpAct?dyn.jumpAct(el.dataset.jump)||'':''):jumpAct(el.dataset.jump);
  if(!body&&!act){hideTip(); return;}
  const h=body?`<div>${body}</div>`+(act?`<div class="act">${act}</div>`:''):`<div>${act}</div>`; showTipHTML(el,view==='lab'?NB(h):h);
}
function tipDone(el){
  if(tipFor===el) hideTip();
  if(el.classList.contains('t')&&view==='lab'){const c=ctlEl(el.dataset.ctl); if(c) c.classList.remove('linked');}
}
function doJump(key,src){if(view==='dyn'){if(dyn&&dyn.jump) dyn.jump(key,src);} else jumpTo(key,src);}
function flash(btn,txt,key){btn.textContent=txt; clearTimeout(btn._t); btn._t=setTimeout(()=>{btn.textContent=t(key);},1600);}
function labUI(){
  $('#kvFlow').addEventListener('click',e=>{const b=e.target.closest('button[data-mode]'); if(b){setMode(b.dataset.mode); revealTop(b.dataset.flow==='free'?$('#controls'):$('#rail'));}});
  $('#rail').addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.bind; if(k) learnerSet({[k]:(k==='ctx'?CTX:USERS)[+e.target.value]});});
  $('#rail').addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b) return; const d=b.dataset;
    const seg=b.closest('[data-bind-seg]'); if(seg){if(!b.disabled) learnerSet({[seg.dataset.bindSeg]:d.v}); return;}
    if(d.chReveal){chReveal=true; updateRail(compute(S)); return;}
    if(d.go!==undefined){enterStep(+d.go); revealTop($('#rail .step.current')); return;}
    if(d.pred){pred=d.pred; renderRail(); drawMetrics(compute(S)); const f=$(`#rail [data-pred="${pred}"]`); if(f) f.focus(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealTop($('#rail .step.current')); return;}
    if(d.nav==='back'){enterStep(step-1); revealTop($('#rail .step.current')); return;}
    if(d.showme){const s=STEPS[step]; if(s.showMe) setState(s.showMe); return;}
    if(d.lever!==undefined){const i=+d.lever; lvTried.add(LEVERS[i].id); learnerSet(lvCfg(i));
      // the result box comes into view: from below the fold its top goes under the header (the kit's reveal);
      // when only its end is cut off, a short scroll shows the end and keeps its first line on screen
      const box=$('#rail [data-obs]'); if(box&&!box.hidden){const q=box.getBoundingClientRect(), vh=window.innerHeight, bar=$('.topbar'), dk=$('#dock'),
          hdr=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(window.innerWidth<=760?50:0))+12;
        if(q.top>vh-160) revealTop(box);
        else if(q.bottom>vh){quietTips(); window.scrollBy({top:Math.max(0,Math.min(q.bottom-vh+12,q.top-hdr)),behavior:reduceMotion()?'auto':'smooth'});}}
      return;}
    if(d.modeGo){setMode(d.modeGo); revealTop($('#rail')); return;}
    if(d.preset!==undefined){learnerSet({...PRESETS[+d.preset].p});return;}
    if(b.id==='chSubmit'){submitChallenge();return;}
  });
  if(false) $('#shareBtn').addEventListener('click',()=>{   // the setup link button is gone (2026-10-11); kept for a later share feature
    const frag=shareText('#'+token()), box=$('#shareBox'), inp=$('#shareInput'), b=$('#shareBtn'), lb=box.querySelector('label');
    if(lb){lb.dataset.i18n=shareKey(); lb.textContent=t(shareKey());}   // a whole link on a site of its own, a fragment inside the Artifact
    inp.value=frag; box.hidden=false;
    const ok=()=>flash(b,tx('Copied','已复制'),'share');
    const fb=()=>{inp.focus(); inp.select();};
    try{ if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(frag).then(ok,fb); else fb(); }catch(err){fb();}
  });
  $('#pinBtn').addEventListener('click',()=>{pinned={s:{...S},r:compute(S)}; render();});
  $('#unpinBtn').addEventListener('click',()=>{pinned=null; render();});
  $('#copyCode').addEventListener('click',()=>{
    const pre=$('#codeBlock'), b=$('#copyCode');
    const ok=()=>flash(b,tx('Copied','已复制'),'copyCode');
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flash(b,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),'copyCode');};
    try{ if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(ok,sel); else sel(); }catch(err){sel();}
  });
  $('#drillNew').addEventListener('click',newDrill);
  $('#drillReveal').addEventListener('click',revealDrill);
  // hover or focus shows what a term, legend item, metric or mode means; clicking a formula term jumps to its control
  document.addEventListener('mouseover',e=>{if(tipQuiet) return; const el=tipTarget(e.target); if(el&&el!==tipFor&&el!==tipEsc) tipShow(el);});
  // a scroll hides the tooltip; moving the mouse again over the same item brings it back
  document.addEventListener('mousemove',e=>{
    if(e.clientX!==ptrX||e.clientY!==ptrY){ptrX=e.clientX; ptrY=e.clientY; tipQuiet=false;}
    if(tipQuiet) return; const el=tipTarget(e.target); if(el&&el!==tipEsc&&(el!==tipFor||tip.hidden)) tipShow(el);},{passive:true});
  document.addEventListener('pointerdown',()=>{tipQuiet=false;},{passive:true});
  // names keep no-break hyphens and spaces on screen (NB); what the learner copies gets plain ones, so it pastes and searches
  document.addEventListener('copy',e=>{const sel=String(getSelection()); if(!e.clipboardData||!/[\u2011\u00a0]/.test(sel)) return;
    e.clipboardData.setData('text/plain',sel.replace(/\u2011/g,'-').replace(/\u00a0/g,' ')); e.preventDefault();});
  // an action inside a lab re-checks the mini result bar; it shows only once the lab's title is under the top bar (R3.2)
  const arm=e=>{if(e.target&&e.target.closest&&e.target.closest('.bench')) dockSoon();};
  document.addEventListener('input',arm,true); document.addEventListener('click',arm,true);
  // every slider's fill follows its thumb, whatever the lab's own handler does (the owner's review of batch 8: the scaling-law sliders'
  // fill stayed behind; the lab skipped the slider being dragged). Bubble phase, after the labs' handlers, so a value a lab snaps is shown
  document.addEventListener('input',e=>{const t=e.target; if(t&&t.matches&&t.matches('input[type=range]')&&+t.max>+t.min) setFill(t);});
  document.addEventListener('mouseout',e=>{
    if(tipFor&&tipFor.contains(e.target)&&!tipFor.contains(e.relatedTarget)) hideTip();
    if(tipEsc&&tipEsc.contains(e.target)&&!tipEsc.contains(e.relatedTarget)) tipEsc=null;
    const el=tipTarget(e.target); if(el&&!el.contains(e.relatedTarget)) tipDone(el);});
  document.addEventListener('focusin',e=>{tipEsc=null; const el=tipTarget(e.target); if(el&&el===e.target) tipShow(el);});
  document.addEventListener('keydown',e=>{if(e.key!=='Escape'||tip.hidden) return; tipEsc=tipFor; hideTip();});
  document.addEventListener('focusout',e=>{const el=tipTarget(e.target); if(el&&el===e.target) tipDone(el);});
  // a click in a rail or on a mode button re-renders the card under a still pointer: no tooltip until the pointer really
  // moves (R1.7). Capture phase: the button is still in the page, before the rail handler replaces it
  document.addEventListener('click',e=>{if(e.target.closest&&e.target.closest('.rail button, .modes button, .crumbs a, .next-lab a')) quietTips();},true);
  document.addEventListener('click',e=>{if(!e.target.closest) return;
    if(tipFor&&!tipFor.contains(e.target)) hideTip();
    const el=e.target.closest('.t'); if(el){doJump(el.dataset.ctl,el); return;}
    const j=e.target.closest('[data-jump]'); if(j) doJump(j.dataset.jump,j);});
  document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList&&e.target.classList.contains('t')){e.preventDefault(); doJump(e.target.dataset.ctl,e.target);}
    else if((e.key==='Enter'||e.key===' ')&&e.target.matches&&e.target.matches('[data-jump][data-ktab]')){e.preventDefault(); doJump(e.target.dataset.jump,e.target);}});
  $('#dock').addEventListener('click',()=>{const c=dockCtx(); if(c&&c.target) c.target.scrollIntoView({block:'start',behavior:reduceMotion()?'auto':'smooth'});});
  window.addEventListener('scroll',dockSoon,{passive:true});
  // a stuck stage scrolled inside itself (a wheel over the stage) moves its title row and the lab's anchor without a window scroll: the bar
  // checks again on any scroll, so it neither stays over a title row scrolled back into its band nor stays away once both have gone (kit polish K4)
  document.addEventListener('scroll',e=>{if(e.target!==document) dockSoon();},{capture:true,passive:true});
  $('#stage').addEventListener('scroll',()=>{hideTip(); dockSoon();},{passive:true});
}

/* ---------- interview drill: generated from the catalog ---------- */
function newDrill(){
  const ms=['llama31-8b','llama31-70b','mixtral-8x7b','llama31-405b'], gs=['rtx4090','h100','h200','mi300x'];
  const pick=a=>a[Math.floor(Math.random()*a.length)];
  for(let i=0;i<400;i++){
    const s={model:pick(ms),gpu:pick(gs),n:pick([1,2,4,8]),ctx:pick([8192,32768,131072]),users:1,wp:pick(['bf16','fp8']),kp:pick(['bf16','fp8']),attn:'gqa',util:0.9,overhead:2};
    if(s.ctx>MODELS[s.model].maxCtx) continue;
    const r=compute(s); if(r.maxUsers<1||r.maxUsers>300) continue;
    if(drill&&drill.s.model===s.model&&drill.s.gpu===s.gpu&&drill.s.n===s.n) continue;
    const w=drillWork(r); if(Math.floor(w.room/w.per+1e-9)!==r.maxUsers) continue;   // the shown numbers must divide to the answer (R6.6)
    drill={s,r}; break;
  }
  renderDrillQ(); $('#drillA').hidden=true;
}
/* the answer's numbers as shown, rounded once; each next line is computed from these, so a reader can redo it (R6.6) */
const rd=(v,d)=>Math.round(v*10**d)/10**d;
const gib=v=>v.toLocaleString('en-US',{maximumFractionDigits:3})+' GiB';
function drillWork(r){const per=rd(r.kvPerReq/r.shard/GiB,3), use=rd(r.usable/GiB,1), w=rd(r.wPerGpu/GiB,1), oh=r.oh/GiB;
  return {per,use,w,oh,room:rd(use-w-oh,1)};}
function renderDrillQ(){
  const {s,r}=drill;
  $('#drillQ').innerHTML=NB(tx(
    `How many users with <b>${fmtTok(s.ctx)}</b> of context each can <b>${r.m.name}</b> serve at once on <b>${s.n} × ${r.g.name}</b>? Assume ${PL[s.wp]} weights, a ${PL[s.kp]} KV cache, 90% memory utilization, 2 GiB of runtime overhead per GPU, and tensor parallelism across all the GPUs.`,
    `<b>${s.n} × ${r.g.name}</b> 跑 <b>${r.m.name}</b>，每个用户 <b>${fmtTok(s.ctx)}</b> 上下文，最多能同时服务几个用户？假设权重 ${PL[s.wp]}、KV cache ${PL[s.kp]}、显存利用率 90%、每卡运行时开销 2 GiB，并在所有 GPU 上做张量并行。`));
}
function revealDrill(){
  const {s,r}=drill, w=drillWork(r), split=r.shard>1, b=PREC[s.kp], raw=w.room/w.per;
  // the quotient is rounded to 3 decimals, but never up to the next whole number, so "rounded down" stays true
  let q=rd(raw,3); if(Math.floor(q)>Math.floor(raw+1e-9)) q=Math.floor(raw*1000)/1000;
  const qs=q.toLocaleString('en-US',{maximumFractionDigits:3}), pu=gib(rd(r.kvPerReq/GiB,3));
  $('#drillA').innerHTML=tx(
    `<p><b>${usersTxt(r.maxUsers)}</b></p><p>KV cache per token = 2 (a key and a value) × ${r.m.layers} layers × ${r.kvHeads} KV heads × ${r.m.headDim} (head_dim) × ${b} byte${b===1?'':'s'} = ${fmtB(r.kvPerToken)}.</p>`+
      `<p>One user’s KV cache = ${fmtB(r.kvPerToken)} × ${tokTxt(fmtInt(s.ctx))} = ${pu}${split?`; split over ${gpusTxt(r.shard)}, ${gib(w.per)} on each`:''}.</p>`+
      `<p>Room for KV cache on each GPU = ${gib(w.use)} usable − ${gib(w.w)} of weights − ${gib(w.oh)} of overhead = ${gib(w.room)}.</p><p>${gib(w.room)} ÷ ${gib(w.per)} = ${qs}, rounded down: ${usersTxt(r.maxUsers)}.</p>`,
    `<p><b>${usersTxt(r.maxUsers)}</b></p><p>每个 token 的 KV cache = 2（key 和 value）× ${r.m.layers} 层 × ${r.kvHeads} 个 KV head × ${r.m.headDim}（head_dim）× ${b} 字节 = ${fmtB(r.kvPerToken)}。</p>`+
      `<p>每个用户的 KV cache = ${fmtB(r.kvPerToken)} × ${tokTxt(fmtInt(s.ctx))} = ${pu}${split?`；分到 ${gpusTxt(r.shard)} 上，每张 ${gib(w.per)}`:''}。</p>`+
      `<p>每张卡留给 KV cache 的显存 = 可用 ${gib(w.use)} − 权重 ${gib(w.w)} − 开销 ${gib(w.oh)} = ${gib(w.room)}。</p><p>${gib(w.room)} ÷ ${gib(w.per)} = ${qs}，向下取整：${usersTxt(r.maxUsers)}。</p>`);
  $('#drillA').hidden=false;
}

/* ---------- language switch ---------- */
function setLang(l){
  if(l===lang) return;
  lang=l; try{localStorage.setItem('asp-lang',l);}catch(e){}
  hideTip(); applyStatic(); renderModes();
  renderHome();
  heroRefresh();
  if(view==='dyn'&&dyn&&dyn.onLang){dyn.onLang(); flowDecorate($('#view-dyn'));}
  placeEndFooter(); pageMeta();
  renderDrillQ(); if(!$('#drillA').hidden) revealDrill();
  if(labInit){renderRail(); render();}
}

/* ---------- router: home, the KV lab (static markup), or a registered lab mounted into #view-dyn ---------- */
/* ---------- PROG: save and put back a lab's state (round 3 #6) ----------
   The KV lab's snapshot lives here; the other labs give theirs through registerLab({snap, restore}) (CONTRACT.md). The kit saves a
   lab's state a moment after each click, input or change in it, when it is left, and when the page is hidden or closed. It puts the
   state back once per page load, before the lab's first mount, unless the address carries other settings than the saved ones (a link
   someone shared: its settings win). */
const kvTok=s=>[s.model,s.gpu,s.n,s.ctx,s.users,s.wp,s.kp,s.attn,Math.round(s.util*100),Math.round(s.overhead*10)].join('.');
const KVLAB={id:'kv-cache',
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,passed:mode==='challenge'&&chSubmitted&&compute(S).maxUsers>=20}),
  restore(o){if(!o||o.v!==1) return; const p=o.S&&typeof o.S==='object'?parseToken(kvTok(o.S)):null; if(!p) return;
    S={...DEFAULT,...p}; mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0;
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&o.observed[k]===true) observed[k]=true;
    const sv=o.saved&&typeof o.saved==='object'?parseToken(kvTok(o.saved)):null; saved=mode==='challenge'&&sv?{...DEFAULT,...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true;}};
// the practice questions' state (EXS) as saved: {tab, q:[i, j], ans:[{j: v}, {j: v}]}, anything else dropped
const exClean=x=>{if(!x||typeof x!=='object') return null; const n=v=>Number.isInteger(v)&&v>=0&&v<50, a=o=>{const r={}; if(o&&typeof o==='object') for(const k in o) if(/^\d+$/.test(k)&&n(o[k])) r[k]=o[k]; return r;};
  return {tab:[0,1,2].includes(x.tab)?x.tab:0,q:Array.isArray(x.q)?[n(x.q[0])?x.q[0]:0,n(x.q[1])?x.q[1]:0]:[0,0],ans:Array.isArray(x.ans)?[a(x.ans[0]),a(x.ans[1])]:[{},{}]};};
function progSave(){const id=view==='lab'?'kv-cache':(view==='dyn'&&dyn?dyn.id:null); if(!id||!LAB[id]) return;
  const lab=id==='kv-cache'?KVLAB:dyn, root=id==='kv-cache'?$('#view-lab'):$('#view-dyn'); let snap=null, tok=null;
  try{snap=lab.snap?lab.snap():null; if(snap&&typeof snap!=='object') snap=null;}catch(e){snap=null;}
  const mb=root&&root.querySelector('.modes button[aria-pressed="true"]'), m=(snap&&snap.mode)||(mb&&mb.dataset.mode)||'guided';
  try{tok=m!=='sandbox'?null:(id==='kv-cache'?kvTok(S):(TOKFN[id]?String(TOKFN[id]()):null));}catch(e){tok=null;}
  const items=root?[...root.querySelectorAll('.steps>.step')]:[], cur=items.findIndex(x=>x.classList.contains('current')), prev=PROG.get(id)||{};
  // a lesson page (snap().mode 'lesson', 2026-10-11) has no list of parts: its snapshot gives the section, the number of sections and done
  const nSteps=items.length||(snap&&Number.isInteger(snap.nSteps)?snap.nSteps:0)||prev.nSteps||0, st=m==='guided'&&cur>=0?cur:(snap&&Number.isInteger(snap.step)?snap.step:(prev.step||0));
  PROG.put(id,{mode:m,step:st,nSteps,maxStep:Math.max(prev.maxStep||0,st),guided:!!(prev.guided||(m==='guided'&&nSteps>0&&st===nSteps-1)||(m==='lesson'&&snap&&snap.done===true)),
    passed:!!(prev.passed||(snap&&snap.passed===true)),tok,snap,ex:EXS[id]?exClean(EXS[id]):(prev.ex||null),flow:FLOW[id]||prev.flow||null,pat:PRAT[id]||prev.pat||null});}
let progT=0; const progSoon=()=>{clearTimeout(progT); progT=setTimeout(progSave,400);};
['click','input','change'].forEach(k=>document.addEventListener(k,()=>{if(view!=='home') progSoon();},true));
window.addEventListener('pagehide',()=>{clearTimeout(progT); progSave();});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'){clearTimeout(progT); progSave();}});
// once per page load, before the lab's first mount: its saved state, unless the address names other settings (returns true when put back)
function progRestore(id,lab,tok){if(RESTORED.has(id)) return false; RESTORED.add(id); const r=PROG.get(id);
  if(!r||!r.snap||!lab.restore||(tok&&tok!==r.tok)) return false;
  try{lab.restore(r.snap);}catch(e){return false;} const ex=exClean(r.ex); if(ex) EXS[id]=ex; return true;}
// back in the primer: the section the reader was reading (once per page load, like the lab's own state)
const FRES=new Set();
function flowResume(id){if(FRES.has(id)) return; FRES.add(id); const r=PROG.get(id); if(!r||r.flow!=='primer'||FLOW[id]!=='primer'||!(r.pat>1)) return;
  const sec=$(`#view-dyn .primer [data-psec="${r.pat|0}"].pr-sec`); if(!sec) return; const bar=$('.topbar'); window.scrollTo(0,Math.max(0,window.scrollY+sec.getBoundingClientRect().top-(bar?bar.offsetHeight:56)-12));}
window.__prog={snap:()=>{const lab=view==='lab'?KVLAB:(view==='dyn'?dyn:null); try{return lab&&lab.snap?lab.snap():null;}catch(e){return null;}},get:id=>PROG.get(id),last:()=>PROG.last(),save:progSave,clear:()=>PROG.clear()};
let dyn=null;
function showView(v){view=v; $('#view-home').hidden=v!=='home'; $('#view-lab').hidden=v!=='lab'; $('#view-dyn').hidden=v!=='dyn'; hideTip(); $('#dock').hidden=true; dockSoon();}
/* the tab's title names the lab, and the page wears the colour of the lab's part of the tree (html[data-limb], styles.css: --branch) */
function pageMeta(){const id=view==='lab'?'kv-cache':(view==='dyn'&&dyn?dyn.id:null), l=id&&LAB[id], root=document.documentElement;
  document.title=l?`${L(l.s||l.q)} · AI Systems Playground`:'AI Systems Playground';
  if(l&&WORLDS[l.w]) root.dataset.limb=WORLDS[l.w].limb; else delete root.dataset.limb;}
function leaveDyn(){if(dyn&&dyn.leave) dyn.leave(); dyn=null; $('#view-dyn').innerHTML='';}
let crumbJump=false;   // the address changed because a lab's branch crumb was clicked (not Back)
document.addEventListener('click',e=>{if(e.target.closest&&e.target.closest('.crumbs a[href^="#map~"]')) crumbJump=true;},true);
function route(){
  const cj=crumbJump; crumbJump=false;
  if(view!=='home'){clearTimeout(progT); progSave();}   // the lab being left keeps its last state
  let h=location.hash.slice(1); try{h=decodeURIComponent(h);}catch(e){}
  const k=h.indexOf('~'), id=k>=0?h.slice(0,k):h, tok=k>=0?h.slice(k+1):null;
  if(id==='kv-cache'){
    lastTile='kv-cache';                  // back to the map returns to the tile of the lab the learner left (R11.7), also after lab-to-lab links
    quietTips();                          // the new lab renders under a pointer that has not moved: no tooltip until it moves (R1.7)
    leaveDyn(); const fresh=!labInit; showView('lab');
    if(fresh){
      labInit=true;
      const p=tok?parseToken(tok):null;
      if(progRestore('kv-cache',KVLAB,tok)){renderModes(); renderRail(); render(); cueTargets();}
      else if(p){Object.assign(S,p); mode='sandbox'; renderModes(); renderRail(); render();}
      else {mode='guided'; renderModes(); enterStep(0);}
    } else {renderRail(); render(); cueTargets();}
    kvCrumbs(); flowDecorate($('#view-lab')); placeEndFooter(); pageMeta(); window.scrollTo(0,0); progSoon();
  } else if(DYN[id]){
    lastTile=id; quietTips(); if(!(HERO&&id===HERO.id)) heroEntry=false;
    const lab=DYN[id]; if(dyn!==lab) leaveDyn(); else $('#view-dyn').innerHTML='';
    dyn=lab; showView('dyn'); const holder=document.createElement('div'); holder.className='dyn-root'; $('#view-dyn').appendChild(holder);
    const back=progRestore(id,lab,tok); if(tok&&FLOW[id]===undefined) FLOW[id]='free';   // a link with settings opens on 自由调参
    try{lab.mount(holder,tok);}catch(e){if(back){PROG.drop(id); location.reload(); return;} throw e;}   // a saved state the lab can no longer show: forget it
    flowDecorate(holder); placeEndFooter(); pageMeta(); window.scrollTo(0,0); flowResume(id); dockSoon(); progSoon();
  } else {
    leaveDyn(); const wasLab=view!=='home'; showView('home'); pageMeta(); heroRefresh(); renderProgress();
    // #map~<domain> (the domain in a lab's breadcrumb) or #map~route:<path> (the end of a path): select it and put its tiles in view
    const pk=tok&&tok.indexOf('route:')===0?tok.slice(6):null;
    if(id==='map'&&tok&&(WORLDS[tok]||(pk&&PATHS[pk]))){const lt=wasLab&&!cj?lastTile:null; lastTile=null; quietTips(); if(pk) selectPath(pk); else selectWorld(tok); refitTree(); fitStrip();
      // a path lands on the tree with its route numbered, as a chip does (review H11); a branch, and a path on phones, on its tiles
      if(pk&&revealMap(false)) return;
      // Back from a lab whose tile is among these: land on that tile and flash it, as #map does (R11.7; re-review O8)
      const tl=lt&&LAB[lt]&&(pk?PATHS[pk].labs.includes(lt):LAB[lt].w===tok)?$('#tile-'+lt):null;
      if(tl){tl.scrollIntoView({block:'center'}); flashEl(tl); return;}
      const th=$('#tilesHead'), bar=$('.topbar'), ds=$('#domStrip'); if(th){const top=(bar?bar.offsetHeight:58)+(ds&&ds.offsetParent?ds.offsetHeight:0)+8;
        window.scrollTo(0,Math.max(0,window.scrollY+th.getBoundingClientRect().top-top));}
      return;}
    // the map shows the path the learner chose (PROG.route(); a path link, a start button or a join button in a lab chooses one), or a branch
    {const rc=PROG.route(); if(rc){if(!(sel.type==='path'&&sel.id===rc)) selectPath(rc);} else if(sel.type==='path') selectWorld(lastWorld);}
    // the tile of the lab just left lives in that lab's domain: show that domain unless the chosen path already has the lab
    if(wasLab&&lastTile&&(!h||h==='map')&&LAB[lastTile]&&!(sel.type==='path'&&PATHS[sel.id].labs.includes(lastTile))&&!(sel.type==='world'&&sel.id===LAB[lastTile].w)) selectWorld(LAB[lastTile].w);
    const el=h?document.getElementById(h):null, fromHero=heroEntry&&HERO&&lastTile===HERO.id, back=wasLab&&lastTile&&(!h||h==='map')&&!fromHero?$('#tile-'+lastTile):null;
    lastTile=null; heroEntry=false; quietTips(); refitTree(); fitStrip();
    if(fromHero&&wasLab) window.scrollTo(0,0);
    else if(back){back.scrollIntoView({block:'center'}); flashEl(back);}
    else if(el&&h==='map'){if(!revealMap(false)) el.scrollIntoView({block:'start'});}   // "Explore the tree", the Tree link: chips and the whole tree
    else if(el&&h==='top') el.scrollIntoView({block:'start'}); else if(wasLab) window.scrollTo(0,0);
  }
}

/* ---------- KV teaser for the home tiles ---------- */
/* One GPU's memory against each user's context: Llama-3.1-70B in BF16 on 8 × H100, 90% of 80 GiB usable, 2 GiB runtime overhead
   (the lab's DEFAULT), but with 64 users, a number no part, prediction, scenario or challenge of the lab uses. So the tile neither
   answers part 2's prediction (100 users: under 16K) nor shows part 3's break point, and its own break point (the total passes the
   usable limit near 21.4K) is not labelled either (R3.8). With 64 users each 1K of context per user adds exactly 2.5 GiB to every
   GPU, so every size on the tile redoes from the notes. The weights and the runtime overhead are flat bands drawn to scale (the
   overhead is about 2 px thick; the legend names it), and the KV cache is one wedge that grows in a straight line: it passes the
   weights near 6.6K and, on its own, the usable limit near 28.8K; red stripes mark the part above the usable limit (R5.6). Both
   axes are linear and hold the whole line (0–32K tokens; 0–100 GiB, and 32K needs 98.4), so nothing is clipped. The slider moves
   in 2K steps (K = 1,024 tokens) and starts at 16K, where the KV cache is already larger than the weights and everything still
   fits. The shapes use compute()'s exact sizes; the words use chain(), the numbers as the lab shows them, with the KV cache always
   to one decimal (40.0 GiB, R6.6). */
const KVTZ={users:64,min:2,max:32,step:2,k0:16,xTicks:[0,8,16,24,32]};   // the tile's own users; context per user, in K
const kvTzCfg=k=>({...DEFAULT,users:KVTZ.users,ctx:k*1024});
let kvTzK=KVTZ.k0, kvTzRO=null;   // the learner's context (module scope: a re-render keeps it); the tile's ResizeObserver
/* the axis caption, in HTML over the chart's top margin: its GiB carries the site's definition (R6.12) */
const kvTzCap=()=>{const g=DEF('GiB',TERMS.gib()); return tx(`Memory per GPU (${g})`,`每张 GPU 的显存（${g}）`);};
const kvTzTW=s=>{let n=0; for(const ch of s) n+=ch.codePointAt(0)>=0x2e80?10:6.1; return n;};   // width of 10 px mono text; CJK about 1 em
/* the chart in real pixels for a tile W px wide, with the marker at k (K tokens per user) */
function kvTzSVG(W,k){
  const z=compute(kvTzCfg(1)), wG=z.wPerGpu/GiB, base=(z.wPerGpu+z.oh)/GiB, per=z.kvPerGpu/GiB, use=z.usable/GiB;   // GiB per GPU; per: 1K of context
  const xM=KVTZ.max, need=c=>base+per*c, yM=Math.ceil(need(xM)/50)*50, kc=(use-base)/per;   // kc: where the line passes the usable limit
  const H=126, L=24, R=10, T=16, PH=94, PW=W-L-R, B0=T+PH;
  const X=c=>L+c/xM*PW, Y=g=>T+PH*(1-g/yM), f=v=>Math.round(v*10)/10;
  const P=(cls,pts)=>`<path class="${cls}" d="${pts.map(([x,y],i)=>(i?'L':'M')+f(x)+' '+f(y)).join('')}Z"/>`;   // points in px
  const Ln=(cls,x1,y1,x2,y2)=>`<line class="${cls}" x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}"/>`;
  const Tx=(cls,x,y,s,a)=>`<text${cls?` class="${cls}"`:''} x="${f(x)}" y="${f(y)}"${a?` text-anchor="${a}"`:''}>${esc(s)}</text>`;
  // every part to scale (B-15): the weights' 16.4 GiB and the overhead's 2 GiB as they are, so the bands compare with the wedge
  const x0=X(0), x1=X(xM), yB=Y(base), yW=Y(wG), yU=Y(use), yE=Y(need(xM));
  let h=`<defs><pattern id="tz-kvc-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="tz-kvc-hs" width="2" height="6"/></pattern></defs>`;
  h+=Ln('tz-kvc-grid',L,T,L+PW,T);
  // bottom to top: weights, runtime overhead and the whole KV cache, then red stripes over the part above the usable limit
  h+=P('tz-kvc-w',[[x0,B0],[x1,B0],[x1,yW],[x0,yW]])+P('tz-kvc-oh',[[x0,yW],[x1,yW],[x1,yB],[x0,yB]])+P('tz-kvc-kv',[[x0,yB],[x1,yE],[x1,yB]]);
  if(kc<xM) h+=P('tz-kvc-ov',[[X(kc),yU],[x1,yE],[x1,yU]])+Ln('tz-kvc-ovl',X(kc),yU,x1,yE);
  h+=Ln('tz-kvc-lim',L,yU,L+PW,yU);
  h+=[use,yM].map(g=>Tx(g===use?'tz-kvc-ylim':'',L-4,Y(g)+3.5,fmtNum(g,0),'end')).join('');
  h+=KVTZ.xTicks.map(c=>Ln('tz-kvc-tick',X(c),B0,X(c),B0+3)+Tx('',X(c),B0+13,c?c+'K':'0','middle')).join('');
  // the chosen context: a line up to the top of the stack and a dot there; the KV cache's size inside the amber, just above the
  // overhead, beside the line (on its left near the right end). Where the wedge is too thin for the text near the line (2K,
  // and 4K on the widest tiles) there is no label: the sentence under the chart gives the number.
  const c=chain(compute(kvTzCfg(k))), xd=X(k), yd=Y(need(k)), tag='KV '+gibW(c.kvG), tw=kvTzTW(tag);
  const yb=yB-3, xFit=X(12.5/PH*yM/per);   // text with its baseline 3 px above the overhead needs the wedge 12.5 px tall at its left end
  let xt=Math.max(xd+5,xFit); if(xt+tw>x1-2) xt=xd-5-tw;
  const gap=xt>xd?xt-xd:xd-xt-tw;
  h+=Ln('tz-kvc-mkh',xd,B0,xd,yd)+Ln('tz-kvc-mk',xd,B0,xd,yd)+`<circle class="tz-kvc-dot" cx="${f(xd)}" cy="${f(yd)}" r="4"/>`;
  if(xt>=xFit-0.05&&gap<=14) h+=Tx('tz-kvc-tag',xt,yb,tag);
  return {h,H};
}
/* the tile's "Assumptions & sources": four short sentences, each the answer to one question a reader would ask (R11.5): what the
   chart holds, why the weights are 16.4 GiB (BF16, with GB and GiB, R6.12), how fast the KV cache grows, and why 72 GiB */
function kvTileNotes(){const r=compute(kvTzCfg(1)), m=r.m, n=r.n, bw=PREC[r.s.wp], bk=PREC[r.s.kp], pc=Math.round(r.s.util*100), use=gib1(chain(r).use),
    par=fmtNum(m.params/1e9,2)+'B', wGB=fmtNum(r.wPerGpu/1e9,1)+' GB', wG=gibW(r.wPerGpu/GiB), per=fmtD(r.kvPerGpu/GiB,2)+' GiB', u=r.s.users, gm=r.g.mem;
  return [tx(`The chart shows one of the ${n} GPUs: 1/${n} of the weights, 1/${n} of every user’s KV cache, and an estimated ${fmtKV(r.oh)} of runtime overhead.`,
             `图里画的是 ${n} 张 GPU 里的一张：1/${n} 的权重、每个用户 1/${n} 的 KV cache，以及估计 ${fmtKV(r.oh)} 的运行时开销。`),
          tx(`The weights are in ${PL[r.s.wp]}, ${bw} bytes per parameter: each GPU holds ${par} parameters × ${bw} bytes ÷ ${n} = ${wGB} = ${wG}.`,
             `权重用 ${PL[r.s.wp]}，每个参数 ${bw} 字节：每张卡存 ${par} 个参数 × ${bw} 字节 ÷ ${n} = ${wGB} = ${wG}。`),
          tx(`Each token needs ${fmtB(r.kvPerToken)} (2 × ${m.layers} layers × ${m.kvHeads} KV heads × ${m.headDim} numbers × ${bk} bytes), so 1K more context for each of the ${u} users adds ${per} to every GPU.`,
             `每个 token 要 ${fmtB(r.kvPerToken)}（2 × ${m.layers} 层 × ${m.kvHeads} 个 KV head × ${m.headDim} 维 × ${bk} 字节），所以 ${u} 个用户的上下文每人多 1K，每张卡就多 ${per}。`),
          tx(`An H100 has ${gm} GB, read here as ${gm} GiB, and the server uses only ${pc}% of it, so each GPU can use ${use}.`,
             `H100 有 ${gm} GB（这里按 ${gm} GiB 计），服务端只用其中 ${pc}%，所以每张卡可用 ${use}。`)]
    .map(x=>`<p class="tz-note">${x}</p>`).join('');}
TEASERS['kv-cache']={render(el){
  const lg=(c,k,tip)=>`<span data-tip="${tip}"><i class="sw ${c}"></i>${t(k)}</span>`, lgx=(c,s,tip)=>`<span data-tipx="${esc(tip)}"><i class="${c}"></i>${esc(s)}</span>`, gm=GPUS[DEFAULT.gpu].mem;
  el.innerHTML=`<div class="tz"><label class="tz-lab"><span>${t('ctxLabel')}</span><output class="mono"></output></label>`+
    `<input type="range" min="${KVTZ.min}" max="${KVTZ.max}" step="${KVTZ.step}" value="${kvTzK}" aria-label="${esc(t('ctxLabel'))}">`+
    `<div class="tz-kvc-box"><div class="tz-kvc-cap">${kvTzCap()}</div><svg class="tz-vis tz-kvc-vis" role="img" xmlns="http://www.w3.org/2000/svg"></svg></div>`+
    `<div class="tz-leg" aria-hidden="true">${lg('w','lgW','tipLgW')}${lg('oh','lgOH','tipLgOH')}${lg('kv','lgKV','tipLgKVt')}${lg('tz-kvc-swov','lgOver','tipLgOver')}`+
    lgx('sw lim',t('lgLim'),tx(`The dashed line is the usable limit: each GPU uses its memory only up to here and keeps the rest of its ${gm} GiB as a safety margin.`,`虚线是可用上限：每张卡的显存只用到这里，${gm} GiB 里剩下的留作安全余量。`))+
    lgx('tz-kvc-swmk',tx('vertical line: your context','竖线：你选的上下文'),tx('The vertical line marks the context per user set with the slider. The dot on it is all the memory one GPU needs at that context.','竖线标出滑块选的每个用户的上下文。线上的圆点是这时一张 GPU 需要的全部显存。'))+`</div>`+
    `<p class="tz-read"></p>`+kvTileNotes()+`</div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), vis=el.querySelector('.tz-kvc-vis'), rd=el.querySelector('.tz-read');
  const draw=()=>{const W=Math.max(200,Math.round(el.clientWidth>40?el.clientWidth:340)), s=kvTzSVG(W,kvTzK);
    vis.setAttribute('viewBox',`0 0 ${W} ${s.H}`); vis.setAttribute('width',W); vis.setAttribute('height',s.H); vis.innerHTML=s.h;};
  const upd=()=>{kvTzK=+inp.value; setFill(inp);
    const r=compute(kvTzCfg(kvTzK)), c=chain(r), k=fmtTok(r.s.ctx), kv=gibW(c.kvG), use=gib1(c.use), more=r.kvPerGpu>r.wPerGpu;
    out.textContent=tokTxt(k); inp.setAttribute('aria-valuetext',tokTxt(k)); draw();
    vis.setAttribute('aria-label',tx(`Memory of one GPU against context per user: the weights and the runtime overhead stay flat, the KV cache grows in a straight line, and red stripes mark the part above the usable limit. The vertical line is at ${k}.`,
                                     `一张 GPU 的显存随每个用户的上下文变化：权重和运行时开销不变，KV cache 沿直线增长，超出可用上限的部分画着红色斜纹。竖线在 ${k}。`));
    // the condition in full, then one key number, the KV cache's size, and the trend in words (R6.20); the 16.4 GiB of weights and
    // the 2.5 GiB per 1K are in the notes, the total is the dot on the chart
    rd.innerHTML=NB(tx(`Llama-3.1-70B on 8 × H100, ${usersTxt(r.s.users)} with ${k} of context each: `,`8 × H100 跑 Llama-3.1-70B，${usersTxt(r.s.users)}，每人 ${k} 上下文：`)+(r.fits
      ?tx(`the KV cache takes <b>${kv}</b> per GPU, ${more?'already more':'still less'} than the weights, ${more?'and':'but'} grows linearly with context.`,
          `每张卡的 KV cache 占 <b>${kv}</b>，${more?'已经比权重多，而且':'还比权重少，但'}随上下文线性增长。`)
      :c.kvG<=c.use
      ?tx(`the KV cache needs <b>${kv}</b> per GPU and pushes the total past the usable ${use}.`,
          `每张卡的 KV cache 要 <b>${kv}</b>，总量因此超过了可用的 ${use}。`)
      :tx(`the KV cache alone needs <b>${kv}</b> per GPU, more than the usable ${use}.`,
          `每张卡光 KV cache 就要 <b>${kv}</b>，超过了可用的 ${use}。`))).replace(/(\d) (?=[KG]iB\b)/g,'$1 ');};   // a size never breaks from its unit
  inp.addEventListener('input',upd); upd();
  // redraw in real pixels when the tile changes width (text never scales); the observer goes with the tile
  if(kvTzRO){kvTzRO.disconnect(); kvTzRO=null;}
  if(typeof ResizeObserver==='function'){let last=el.clientWidth; const ro=new ResizeObserver(()=>{if(!el.isConnected){ro.disconnect(); if(kvTzRO===ro) kvTzRO=null; return;}
    const w=el.clientWidth; if(w>40&&Math.abs(w-last)>=1){last=w; draw();}}); ro.observe(el); kvTzRO=ro;}
}};

/* ---------- kit API: what lab and teaser modules may use ---------- */
function flashEl(el){if(!el) return; el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); clearTimeout(el._ft); el._ft=setTimeout(()=>el.classList.remove('flash'),1700);}
function jumpToEl(el){
  if(!el) return; flowReveal(el); const d=el.closest('details'); if(d) d.open=true; hideTip();
  const dr=el.closest('.m2 .controls'); if(dr&&!dr.classList.contains('open')&&PHONE.matches) drawerToggle(dr,true);
  el.scrollIntoView({block:'center',behavior:reduceMotion()?'auto':'smooth'});
  const f=el.matches('input,select,button')?el:(el.querySelector('input,select,button[aria-pressed="true"]')||el.querySelector('button,[tabindex]:not([data-ktab])'));
  if(f) f.focus({preventScroll:true}); flashEl(el);}
function obsBump(box){if(!box||reduceMotion()) return; box.classList.remove('bump'); void box.offsetWidth; box.classList.add('bump');}
/* after Next, Back or a mode switch: bring the new card's top edge into view, below the top bar and the mini result bar */
let structUntil=0;   // a click that changes the part or the mode was just made (see revealTop)
let pinGen=0;   // pinCtl's pending corrections; a click that changes the part or the mode, or a reveal, cancels them
document.addEventListener('click',e=>{if(e.target.closest&&e.target.closest('[data-nav],[data-go],.modes [data-mode],[data-mode-go]')){structUntil=performance.now()+700; pinGen++;}},true);
function revealTop(el){
  if(!el) return; flowReveal(el);
  // a part the lab's current step hides (the rail in 自由调参): the step's own part comes into view instead
  if(!el.getClientRects().length){const b=el.closest('.bench.flowed'); if(b){const c=b.querySelector(':scope>.controls'); el=b.dataset.flow==='free'&&c?c:b;}}
  // phones, m2 labs: the parts' bar sits above the stage and the rail itself has no box, so a new part or mode brings the bar (guided)
  // or the stage (other modes) to the top: the picture first, the card right below it
  // (only right after a click that changes the part or the mode: a lab's own reveal of its card, e.g. "the conclusion is in part 3", goes to the card)
  if(PHONE.matches&&performance.now()<structUntil){const m=el.closest('.m2'); if(m&&(el.matches('.rail')||el.closest('.rail .steps'))){const pb=m.querySelector('.bench .pbar'), st=m.querySelector('.bench>.stage');
    el=pb&&pb.getBoundingClientRect().height>0?pb:(st||el);}}
  const bar=$('.topbar'), dk=$('#dock'), r=el.getBoundingClientRect(), bh=bar?bar.offsetHeight:56;
  let top=bh+(dk&&!dk.hidden?dk.offsetHeight:0)+12;
  // phones: room for the mini result bar only when it will show where the scroll ends (kit polish K3: the 50 px kept for a bar that did not
  // come pushed rag-failure's metric cards off the screen in Scenarios and Settings). Asked for the place without that room: if the bar
  // would show there, the card goes under it, as before; if not, the card goes right under the top bar
  if(window.innerWidth<=760){const t0=bh+12, de=document.documentElement, to=Math.max(0,Math.min(de.scrollHeight-window.innerHeight,window.scrollY+r.top-t0));
    top=dockShows(dockCtx(),to-window.scrollY)?bh+(dk&&!dk.hidden?dk.offsetHeight:DOCK_H)+12:t0;}
  if(window.innerWidth<=760?Math.abs(r.top-top)<8:(r.top>=top&&r.top<=window.innerHeight-160)) return;
  pinGen++; quietTips(); window.scrollTo({top:Math.max(0,window.scrollY+r.top-top),behavior:reduceMotion()?'auto':'smooth'});}
/* a change the learner just made, always saying what it is compared with (R6.6).
   dp: the decimals the card shows; both values are rounded to it first, so the change reproduces from the shown values.
   pts: the values are percentages, so the change is in percentage points. base: what is compared with (default: the last value).
   base:false (kit polish K2) writes no comparison at all, for a card group that names it once above the cards (“Compared with
   ‘Just-launched chat app’” / “和‘刚上线的聊天应用’比”): “▲ 41 pts” / “▲ 41 个百分点”, “▲ 2.1×” / “▲ 2.1 倍”, “▼ −26%”, “▲ up from 0” /
   “▲ 从 0 升上来”; a narrow card then keeps the change on one line (product-design N13, chunk-rerank P4). Without it, nothing changes.
   An empty string means no change: the caller clears its badge. Callers show a badge only after the learner's own change,
   never when a part, a mode or the lab itself sets the state. */
function deltaVs(a,b,o){o=o||{};
  if(!isFinite(a)||!isFinite(b)) return '';
  if(o.dp!=null){const k=Math.pow(10,o.dp); a=Math.round(a*k)/k; b=Math.round(b*k)/k;}
  if(a===b) return '';
  const bare=o.base===false, base=(!bare&&o.base)||{en:'last',zh:'上次'}, vs=bare?'':tx(` vs ${base.en}`,`（比${base.zh}）`);
  if(o.pts){const dp=o.dp!=null?o.dp:1, d=b-a; if(Math.abs(d)<0.5*Math.pow(10,-dp)) return ''; const n=fmtNum(Math.abs(d),dp);
    return (d>0?'▲ ':'▼ ')+n+tx(n==='1'?' pt':' pts',' 个百分点')+vs;}
  // from or to 0 there is no ratio: say so, without "vs last"
  if(a===0) return bare?(b>0?tx('▲ up from 0','▲ 从 0 升上来'):tx('▼ down from 0','▼ 从 0 降下来')):(b>0?tx('▲ up from 0',`▲ ${base.zh}是 0`):tx('▼ down from 0',`▼ ${base.zh}是 0`));
  if(b===0) return a>0?tx('▼ down to 0','▼ 降到 0'):tx('▲ up to 0','▲ 升到 0');
  const q=b/a;
  if(q>=1.995){const n=q>=9.95?fmtInt(q):fmtNum(q,1); return bare?tx(`▲ ${n}×`,`▲ ${n} 倍`):tx(`▲ ${n} times ${o.base?base.en:'the last value'}`,`▲ 是${base.zh}的 ${n} 倍`);}
  // half away from zero, as by hand (160 → 252 is +58%); a value that is not 0 never shows −100%: 1,371 → 1 is −99.9%
  const r=(q-1)*100, p=Math.sign(r)*Math.round(Math.abs(r)+1e-9); if(p===0) return '';
  if(p<=-100){let d=1, v=100; for(;d<=4;d++){v=Math.round(Math.abs(r)*10**d+1e-9)/10**d; if(v<100) break;} return '▼ −'+fmtD(v,d)+'%'+vs;}
  return (p>0?'▲ +':'▼ −')+Math.abs(p)+'%'+vs;}
let annT2=null; function announceText(s){clearTimeout(annT2); annT2=setTimeout(()=>{$('#live').textContent=s;},500);}
/* what the chart's colours mean: green meets the goal, gray does not, the dashed line is the goal */
const cmpLegend=o=>o.legend===false?'':`<div class="cmp-leg" aria-hidden="true"><span><i class="ok"></i>${esc(o.okText||tx('meets the goal','达标'))}</span><span><i></i>${esc(o.noText||tx('misses it','未达标'))}</span>${o.goal!=null?`<span><i class="g"></i>${esc(o.goalText||tx('goal','目标'))}</span>`:''}</div>`;
const modeDescHTML=(m,descs)=>`<b>${t(MODENAME[m])}${tx('.','：')}</b>${tx(' ','')}${esc(descs[m]||'')}`;
/* phones, in labs that opt in with the class m2 (Codex review 1 #3): the guided parts as a bar above the stage, so the figure and an
   action are on the first screen: dots to jump, Back and Next, the part's title and its one line (steps[i].lead, HTML) */
/* round 3, R08: a guided part's question is its group of prediction buttons (data-pred, data-xpred, data-otpred, data-tri) in the part's
   body. It is answered once one of them is pressed, or once the lab has replaced them by the answer. Kept per lab and part in this page
   ('a' answered, 'n' seen and it has no question, 'q' seen with its question open) and in PROG as ans (true: answered or no question,
   false: question open, null: not seen). A part left behind shows ✓ only when 'a' or 'n'; otherwise its number, marked as skipped.
   While the question is open and Next can be pressed, Next says "Skip question"; on phones the bar's main action goes to the question */
const PQ_RE=/<button\b[^>]*\bdata-(?:pred|xpred|otpred|tri)="[^"]*"[^>]*>/g, PQ_SEL='[data-pred],[data-xpred],[data-otpred],[data-tri]',
  PC_SEL='input,select,textarea,.seg-ctl button,[data-lever],[data-preset]';
function partQ(body){const h=String(body||''), bs=h.match(PQ_RE)||[];
  return {has:bs.length>0,open:bs.length>0&&!bs.some(b=>/\baria-pressed="true"/.test(b)),ctl:/<(?:input|select|textarea)\b|class="seg-ctl|\bdata-lever="|\bdata-preset="/.test(h)};}
const PANS={};
const labNow=()=>view==='lab'?'kv-cache':(view==='dyn'&&dyn?dyn.id:null);
function partLog(id,n,cur,q){if(!id) return null; let a=PANS[id];
  if(!a){a=PANS[id]=[]; const r=PROG.get(id); if(r&&Array.isArray(r.ans)) r.ans.forEach((v,i)=>{if(i<n&&(v===true||v===false)) a[i]=v?'a':'q';});}
  if(q&&cur>=0&&cur<n){const w=a[cur], v=q.has?(q.open&&w!=='a'?'q':'a'):(w==='q'||w==='a'?'a':'n');
    if(v!==w){a[cur]=v; if(LAB[id]) PROG.put(id,{ans:Array.from({length:n},(_,i)=>a[i]==='a'||a[i]==='n'?true:(a[i]==='q'?false:null))});}}
  return a;}
// part i, the current part being cur: 'cur', 'done' (✓), 'skip' (left behind without its answer) or 'todo'
const partSt=(a,i,cur)=>i===cur?'cur':(i>cur?'todo':(!a||a[i]==='a'||a[i]==='n'?'done':'skip'));
const skipLabel=(i,title)=>tx(`Part ${i+1}: skipped. ${title}`,`第 ${i+1} 部分：跳过了。${title}`);
const stepCls=(a,i,cur)=>({cur:'current',done:'done',skip:'skipped',todo:'todo'})[partSt(a,i,cur)];
const stepHeadHTML=(a,i,cur,title)=>{const st=partSt(a,i,cur);
  return `<button type="button" class="step-h" data-go="${i}"${st==='cur'?' aria-current="step"':''}${st==='skip'?` aria-label="${esc(skipLabel(i,title))}"`:''}><span class="num">${st==='done'?'✓':i+1}</span><span>${esc(title)}</span>${st==='skip'?`<span class="sk-tag" aria-hidden="true">${tx('skipped','跳过了')}</span>`:''}</button>`;};
const nextTxt=sq=>sq?tx('Skip question','跳过本题'):tx('Next','下一部分');
const pbarHTML=o=>{const c=o.steps[o.current]||{}, a=o.ans||null, q=o.q||{}, sq=!!(o.next&&q.open&&!o.nextDisabled);
  const go=q.open?`<button type="button" class="btn sm" data-pbgo="q">${tx('Answer below','去作答')} <span aria-hidden="true">↓</span></button>`
    :(q.ctl?`<button type="button" class="btn${o.next&&!o.nextDisabled?' ghost':''} sm" data-pbgo="c">${tx('Try it below','去操作')} <span aria-hidden="true">↓</span></button>`:'');
  return `<div class="pbar" role="group" aria-label="${esc(tx('Guided tour','分步引导'))}"><div class="pb-row"><span class="pb-dots">`+
    o.steps.map((s,i)=>{const st=partSt(a,i,o.current);
      return `<button type="button" class="pb-dot${st==='cur'?' on':(st==='done'?' done':(st==='skip'?' skip':''))}" data-go="${i}" aria-label="${esc(st==='skip'?skipLabel(i,s.title):tx(`Part ${i+1}: ${s.title}`,`第 ${i+1} 部分：${s.title}`))}"${st==='cur'?' aria-current="step"':''}>${st==='done'?'✓':i+1}</button>`;}).join('')+
    `</span><span class="pb-nav">${o.back?`<button type="button" class="btn ghost sm" data-nav="back" aria-label="${esc(tx('Back','上一部分'))}">◀</button>`:''}${o.next?`<button type="button" class="btn${sq?' ghost':''} sm" data-nav="next"${o.nextDisabled?' disabled':''}>${nextTxt(sq)}<span class="pb-arr" aria-hidden="true"> ▶</span></button>`:''}</span></div>`+
    nextWhyHTML(o)+`<div class="pb-t"><b>${esc(c.title||'')}</b>${c.lead?`<span class="pb-lead">${c.lead}</span>`:''}${go?` <span class="pb-go">${go}</span>`:''}</div></div>`;};
/* the bar's "Answer below ↓" / "Try it below ↓": the card's question (its line and its options) or its first control comes up under the
   top bar, focused and flashed (R08) */
document.addEventListener('click',e=>{const b=e.target.closest&&e.target.closest('.pbar [data-pbgo]'); if(!b) return; const host=b.closest('.pbar').parentElement,
  sb=host&&host.querySelector(':scope>.steps>.step.current>.step-b'); if(!sb) return; const isQ=b.dataset.pbgo==='q', f=sb.querySelector(isQ?PQ_SEL:PC_SEL);
  if(!f){revealTop(sb); return;} const g=f.closest('[role="group"],.choices,.levers,.inline-ctl,.ctl')||f, pv=g.previousElementSibling;
  // a question comes up from the top of its stem (the paragraphs right above the options, owner 2026-10-14: the question line alone
  // left the setup under the top bar), as far back as still leaves the options on the screen (a long list: at least its first 45%)
  let to=isQ&&pv&&pv.matches('p')?pv:g;
  if(to!==g){const bar=$('.topbar'), dk=$('#dock'), av=window.innerHeight-(bar?bar.offsetHeight:56)-(dk&&!dk.hidden?dk.offsetHeight:50)-12,
    room=av-Math.min(g.offsetHeight,av*0.45), gt=g.getBoundingClientRect().top;
    for(let e=to.previousElementSibling; e&&e.matches('p:not(.nx-why),ul,ol')&&e.getClientRects().length; e=e.previousElementSibling){if(gt-e.getBoundingClientRect().top>room) break; to=e;}}
  revealTop(to); f.focus({preventScroll:true}); flashEl(g);});
/* why Next is disabled (kit fixes, round K): steps({nextWhy}) puts this line under the card's buttons and under the part bar's buttons.
   It shows while Next is disabled; once Next is enabled, in place or by a re-render, it keeps its height but goes blank (CSS, visibility),
   so nothing below it moves. Rendered while Next is already enabled, it takes no room until Next is disabled in place (kitObserve) */
const nextWhyHTML=o=>o.next&&o.nextWhy?`<p class="nx-why" data-next-why${o.nextDisabled?'':' hidden'}>${o.nextWhy}</p>`:'';
/* phones, m2 labs: Settings folds into a drawer that opens from its heading; a jump into it opens it */
const PHONE=window.matchMedia?matchMedia('(max-width:760px)'):{matches:false,addEventListener(){}};
function drawerSync(){$$('.m2 .controls').forEach(c=>{const h=c.querySelector(':scope>.eyebrow'); if(!h) return;
  if(PHONE.matches){h.setAttribute('role','button'); h.setAttribute('tabindex','0'); h.setAttribute('aria-expanded',String(c.classList.contains('open')));}
  else{h.removeAttribute('role'); h.removeAttribute('tabindex'); h.removeAttribute('aria-expanded');}});}
function drawerToggle(c,on){const h=c.querySelector(':scope>.eyebrow'); c.classList.toggle('open',on); if(h&&PHONE.matches) h.setAttribute('aria-expanded',String(on));}
document.addEventListener('click',e=>{const h=e.target.closest&&e.target.closest('.m2 .controls>.eyebrow'); if(h&&PHONE.matches) drawerToggle(h.parentNode,!h.parentNode.classList.contains('open'));});
document.addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ') return; const h=e.target.closest&&e.target.closest('.m2 .controls>.eyebrow'); if(h&&PHONE.matches){e.preventDefault(); drawerToggle(h.parentNode,!h.parentNode.classList.contains('open'));}});
if(PHONE.addEventListener) PHONE.addEventListener('change',drawerSync);
/* phones, m2: the card sits between the stage and the Settings drawer, so a change in Settings that re-renders the card would push the
   control away from the finger; keep the changed control where it was (R3.2) */
function pinCtl(e){if(!PHONE.matches) return; const t=e.target; if(!t||!t.closest) return; const c=t.closest('.m2 .controls'); if(!c) return;
  const ctl=t.closest('.ctl,.inline-ctl,details,.eyebrow')||t, y0=ctl.getBoundingClientRect().top, g=++pinGen;
  // a later click that changes the part or the mode (or a reveal) cancels the pending corrections: an instant scrollBy would stop its smooth scroll
  const fix=()=>{if(g!==pinGen||!ctl.isConnected) return; const dy=ctl.getBoundingClientRect().top-y0; if(Math.abs(dy)>=1) window.scrollBy(0,dy);};
  requestAnimationFrame(()=>{fix(); requestAnimationFrame(fix);}); setTimeout(fix,150); setTimeout(fix,450);}
['input','change','click'].forEach(k=>document.addEventListener(k,pinCtl,true));
/* phones, m2: Settings (and in the challenge the exercises) sit below the card, so when a tap in the stage or the card changes the card's
   height, the browser's scroll anchoring could keep what is below in place and move everything above it instead: after Submit the new
   result pushed the page down by its own height (MHA 961 px, with the switches just used off screen; 100 to 150 px in other labs).
   Anchoring pauses from the tap until the change is laid out, so the page only moves when a lab reveals something on purpose (R3.2).
   Settings has pinCtl instead. */
let anchorRaf=0;
function pauseAnchor(){const de=document.documentElement; de.style.overflowAnchor='none'; cancelAnimationFrame(anchorRaf);
  anchorRaf=requestAnimationFrame(()=>{anchorRaf=requestAnimationFrame(()=>{anchorRaf=0; de.style.overflowAnchor='';});});}
function holdAnchor(e){if(!PHONE.matches) return; const t=e.target; if(!t||!t.closest||!t.closest('.m2 .bench')||t.closest('.controls')) return; pauseAnchor();}
['input','change','click'].forEach(k=>document.addEventListener(k,holdAnchor,true));
/* ---------- the capability layer (Codex review 1 #5, Batch 2) ----------
   Under the title: what the learner can do after the lab. In the challenge, after the lab's own task: three ways to show it transfers,
   one question at a time, each answered with a click (the product owner: choices are friendlier than free text, for beginners above all).
   1 Explain it: pick the explanation that is right; a wrong pick says what it gets wrong. 2 A new case: predict a case the lab never
   showed, then check it (on the stage, where the lab can). 3 Go further: examples of real frontier-lab interview questions from the
   product owner's notes, with no company names on the page. The state lives here, keyed by lab id, so a lab may re-render its rail freely.
   data: {explain:[Q], transfer:[Q], practice:[{id, d}]}; Q = {q, o:[text], a, why:html, n?:[html, one per option, for a wrong pick], go?:fn, goLabel?} */
const outcomeHTML=h=>`<p class="lab-outcome"><b>${tx('You will be able to: ','学完你能：')}</b>${h}</p>`;
/* the practice problems (the product owner's interview notes on GitHub): repo, folder, title EN, title ZH; the page links README.md or README.zh.md */
const PRACTICE={
  'transformer-bug-hunt':['OpenAI-Interview-Notes','coding/transformer-bug-hunt','Transformer Bug Hunt','Transformer 找 bug'],
  'llm-fundamentals':['Google-DeepMind-Interview-Notes','quiz/llm-fundamentals','LLM Fundamentals: Attention, Transformers and the Training Pipeline','大模型基础：注意力、Transformer 与训练流程'],
  'distributed-training':['Google-DeepMind-Interview-Notes','system-design/distributed-training','Training a Model That Does Not Fit on One Accelerator','为单卡放不下的模型设计训练方案'],
  'ckpt-code-review':['Google-DeepMind-Interview-Notes','coding/code-review-round','Code Review: Ranking the Defects in a Checkpointing Change','代码评审：给一次 checkpoint 改动中的缺陷排序'],
  'data-batcher-ckpt':['Anthropic-Interview-Notes','coding/weighted-data-batcher','Weighted Data Batcher with Checkpointing','带断点续跑的加权数据批采样器'],
  'file-dedup':['Anthropic-Interview-Notes','coding/file-deduplication','File Deduplication','重复文件查找'],
  'grpo-debug':['Anthropic-Interview-Notes','coding/grpo-debug','Debugging a GRPO Training Loop','调试 GRPO 训练循环'],
  'matmul-roofline':['Anthropic-Interview-Notes','coding/matmul-performance-modeling','Performance Modelling: Matmuls, Roofline and Sharding','性能建模：矩阵乘法、Roofline 与切分'],
  'qkv-einsum':['Anthropic-Interview-Notes','coding/qkv-attention-einsum','Multi-Head Attention with einsum','用 einsum 实现多头注意力'],
  'attention-kv-cache':['Google-DeepMind-Interview-Notes','coding/attention-kv-cache','Attention with a KV Cache and an Online Softmax','带 KV cache 与在线 softmax 的注意力'],
  'inference-batching':['Anthropic-Interview-Notes','system-design/inference-api-batching','LLM Inference API with Batched GPU Serving','带批处理的 LLM 推理 API'],
  'inference-serving':['OpenAI-Interview-Notes','system-design/llm-inference-serving','LLM Inference Serving','大模型推理服务'],
  'inference-engine':['Anthropic-Interview-Notes','coding/oa-inference-engine','Completing a Small LLM Inference Engine','补全一个小型 LLM 推理引擎'],
  'memory-allocator':['OpenAI-Interview-Notes','coding/memory-allocator','Memory Allocator','内存分配器'],
  'rl-for-llms':['Google-DeepMind-Interview-Notes','quiz/rl-for-llms','RL for Language Models: Policy Gradients, PPO, GRPO and DPO','大模型强化学习：策略梯度、PPO、GRPO 与 DPO'],
  'cross-entropy':['OpenAI-Interview-Notes','coding/cross-entropy-loss','Cross-Entropy Loss from Scratch','从零实现交叉熵损失'],
  'streaming-entropy':['OpenAI-Interview-Notes','coding/streaming-entropy','Streaming Entropy','流式熵计算'],
  'numpy-1nn':['OpenAI-Interview-Notes','coding/numpy-1nn-affine','NumPy 1-NN as an Affine Layer','NumPy 1-NN 写成一层 Wx+b'],
  'autograd-scan':['OpenAI-Interview-Notes','coding/autograd-hillis-steele','Prefix Matmul: Autograd, Manual Backward, Hillis–Steele Scan','前缀矩阵连乘：Autograd、手写反向、Hillis–Steele 扫描'],
  'sharded-matmul-backprop':['OpenAI-Interview-Notes','coding/sharded-matmul-backprop','Sharded Matmul: Forward, Backward, Bug Hunt','分片矩阵乘法：前向、反向与找 bug'],
  'mlp-backprop':['Google-DeepMind-Interview-Notes','coding/mlp-backprop-from-scratch','Backpropagation from Scratch','从零实现反向传播'],
  'ml-fundamentals':['Google-DeepMind-Interview-Notes','quiz/ml-fundamentals','ML Fundamentals: Metrics, Losses, Optimisers and Regularisation','机器学习基础：评估指标、损失函数、优化器与正则化'],
  'training-loop-debug':['Google-DeepMind-Interview-Notes','coding/training-loop-debug','Debugging a Classifier That Does Not Learn','调试一个学不会的分类器'],
  'cs-fundamentals':['Google-DeepMind-Interview-Notes','quiz/cs-fundamentals','CS Fundamentals Quiz: Memory, Concurrency and Floating Point','计算机基础问答：内存、并发与浮点数'],
  'llm-eval-system':['Google-DeepMind-Interview-Notes','system-design/llm-evaluation-system','Design an Evaluation System for an LLM Assistant','为大模型助手设计评测系统'],
  'stats-tests':['Google-DeepMind-Interview-Notes','quiz/maths-stats-information-theory','Maths Quiz: Matrix Computation, Statistical Tests and Information Theory','数学问答：矩阵计算、统计检验与信息论'],
  'agent-tool-loop':['Anthropic-Interview-Notes','coding/llm-agent-tool-loop','Building an Agent Loop with Tool Use','用工具调用搭建智能体循环'],
  'agent-harness-eval':['OpenAI-Interview-Notes','system-design/agent-harness-eval','Agent Harness and Evaluation Framework','Agent 执行框架与评测体系'],
  'chat-app-context':['Google-DeepMind-Interview-Notes','system-design/gemini-app','Design the Gemini App','设计 Gemini App'],
  'prompt-playground':['Anthropic-Interview-Notes','system-design/prompt-playground','Prompt Playground (Product and Data Design)','提示词 Playground（产品与数据设计）'],
  'ai-chatbot-frontend':['OpenAI-Interview-Notes','system-design/ai-chatbot-frontend','AI Chatbot Front End','AI 聊天机器人前端'],
  'design-doc-review':['Anthropic-Interview-Notes','system-design/design-doc-review','Design Doc Review: A Batched Inference Service','设计文档评审：批处理推理服务'],
  'kd-tree':['Google-DeepMind-Interview-Notes','coding/kd-tree-nearest-neighbour','k-d Tree: Nearest Neighbours and Range Queries','k-d 树：最近邻与范围查询'],
  'rag-search-ml':['OpenAI-Interview-Notes','system-design/rag-search-ml-design','Search & RAG ML Design (spoken)','搜索与 RAG 的 ML 设计（口述）'],
  'rag-agent-platform':['Google-DeepMind-Interview-Notes','system-design/rag-agent-platform','Design a Retrieval-Augmented Agent over Company Documents','设计基于企业文档的检索增强智能体']};
const practiceURL=id=>{const p=PRACTICE[id]; return p?`https://github.com/Schuture/${p[0]}/blob/main/${p[1]}/${tx('README.md','README.zh.md')}`:'';};
const EXS={}, EXD={};
const exState=id=>EXS[id]||(EXS[id]={tab:0,q:[0,0],ans:[{},{}]});
/* one tab's questions, one at a time: dots to jump (✓ / ✗ once answered), the question, the options (radio dots), then the verdict.
   Explanations are written with the right one first; they show in a fixed order shuffled by the question's text, so the right one is
   not always on top (a new case keeps its options in the written order, which is usually numeric) */
const exOrder=(q,t)=>{const n=q.o.length, ix=[...Array(n).keys()]; if(t!==0) return ix;
  let h=2166136261; for(const ch of q.q) h=Math.imul(h^ch.codePointAt(0),16777619)>>>0;
  const rnd=()=>{h=(h+0x6D2B79F5)>>>0; let x=Math.imul(h^h>>>15,1|h); x=x+Math.imul(x^x>>>7,61|x)^x; return ((x^x>>>14)>>>0)/4294967296;};
  for(let i=n-1;i>0;i--){const k=Math.floor(rnd()*(i+1)); [ix[i],ix[k]]=[ix[k],ix[i]];} return ix;};
function exQuizHTML(Q,t,s){const j=Math.min(s.q[t]||0,Q.length-1), q=Q[j], A_=s.ans[t], a=A_[j], ok=a===q.a;
  return (Q.length>1?`<div class="ex-qs">${Q.map((x,i)=>{const v=A_[i]; return `<button type="button" class="pb-dot${i===j?' on':''}${v==null?'':(v===x.a?' ok':' no')}" data-ex="q" data-t="${t}" data-j="${i}" aria-label="${esc(tx(`Question ${i+1}`,`第 ${i+1} 题`))}"${i===j?' aria-current="true"':''}>${v==null?i+1:(v===x.a?'✓':'✗')}</button>`;}).join('')}</div>`:'')+
    `<p><b>${q.q}</b></p><p class="small muted">${t===0?tx('Pick the explanation that is right.','选出讲得对的一句。'):tx('Predict first, then check.','先预测，再核对。')}</p>`+
    `<div class="choices ex-opts">`+exOrder(q,t).map(v=>`<button type="button" class="choice${a!=null&&v===q.a?' right':''}${a===v&&v!==q.a?' wrong':''}" data-ex="ans" data-t="${t}" data-j="${j}" data-v="${v}" aria-pressed="${a===v}">${escMath(q.o[v])}</button>`).join('')+`</div>`+
    (a!=null?`<div class="ex-res ${ok?'ok':'no'}" aria-live="polite">`+(ok?`<p><b>${tx('✓ Right.','✓ 对。')}</b>${tx(' ','')}${q.why}</p>`:`<p><b>${tx('✗ Not quite.','✗ 不太对。')}</b>${q.n&&q.n[a]?tx(' ','')+q.n[a]:''}</p><p>${q.why}</p>`)+
      (q.go?`<button type="button" class="btn ghost sm" data-ex="go" data-t="${t}" data-j="${j}">${q.goLabel||tx('See it on the stage','在图上看')} →</button>`:'')+`</div>`:'')+
    (Q.length>1&&j<Q.length-1?`<div class="stepnav"><button type="button" class="btn sm" data-ex="q" data-t="${t}" data-j="${j+1}">${tx('Next question in this set','这组下一题')}</button></div>`:'');}
/* the lab's own interview drill (in Connect, at the bottom of the page), if it has one */
const drillCard=()=>{const h=[...document.querySelectorAll('.c-card>h3')].find(x=>x.offsetParent!==null&&x.textContent.trim()===t('cDrill')); return h?h.closest('.c-card'):null;};
/* round 3, R07: the card is optional practice. Once the lab's challenge is passed (its snap().passed now, or PROG's record that it was
   passed once), the card opens by saying so, with the next lab as the main action (nextInfo(id).l only; none at the end of a path).
   A lab that shows this summary in its own challenge card (A.ui.done) passes selfDone:true in the card's data */
function labPassed(id){let p=false; try{const lab=id==='kv-cache'?KVLAB:DYN[id], o=lab&&lab.snap?lab.snap():null; p=!!(o&&o.passed===true);}catch(e){p=false;}
  if(!p){const r=PROG.get(id); p=!!(r&&r.passed===true);} return p;}
// the main button names the same next step as the footer (nextLabHTML): a branch's next lab, a suggested next step, the next stop of the
// chosen route, or the route to go on with after its last stop; at a route's last stop with nothing after it, no button
function doneHTML(id){let x=null; try{x=nextInfo(id);}catch(e){x=null;}
  const btn=(l,pid,k)=>`<a class="btn sm ex-nextlab" href="#${l.id}"${pid?` data-path="${pid}"`:''}>${k}${esc(labShort(l))} <span aria-hidden="true">→</span></a>`;
  let a='';
  if(x&&x.kind==='branch'){const l=x.l||(x.rec&&x.rec.l); if(l&&l.id) a=btn(l,null,x.l?tx('Next lab on this branch: ','这一枝的下一个 Lab：'):tx('Suggested next lab: ','推荐下一步：'));}
  else if(x&&x.l&&x.l.id) a=btn(x.l,x.to,x.done?tx(`Go on with the path “${pName(x.to)}”: `,`接下来可以继续“${pName(x.to)}”路线：`):tx('Next stop: ','下一站：'));
  return `<div class="ex-passed"><p><b>${tx('✓ You have finished this lab’s core: the challenge meets the goal.','✓ 这个 Lab 的核心学习完成了：挑战已达标。')}</b>${tx(' The practice below can wait for later.','下面的练习可以以后再做。')}</p>`+a+`</div>`;}
function exitsHTML(id,d){EXD[id]=d; const s=exState(id), T=[tx('Explain it','讲清楚'),tx('A new case','新情况'),tx('Go further','延伸练习')];
  const b=s.tab===0?exQuizHTML(d.explain,0,s):(s.tab===1?exQuizHTML(d.transfer,1,s):
    `<p class="small muted">${tx('Examples of frontier-lab interview questions, reconstructed from second-hand accounts. They take this lab further: write the code, or the design.','Frontier lab 真题示例（据面经整理），比这个 Lab 更进一步：要写出代码，或做出设计。')}</p><ul class="ex-links">`+
      d.practice.map(x=>{const p=PRACTICE[x.id]; if(!p) return ''; return `<li><a href="${practiceURL(x.id)}" target="_blank" rel="noopener">${esc(tx(p[2],p[3]))} <span aria-hidden="true">↗</span></a><span>${x.d}</span></li>`;}).join('')+`</ul>`+
      (drillCard()?`<p class="small muted ex-drill">${tx('This lab’s own interview drill, a new question each time, is at the bottom of the page.','这个 Lab 自己的面试题（每次换一道）在页面底部。')}${tx(' ','')}<button type="button" class="linkbtn" data-ex="drill">${tx('Go to it ↓','去看 ↓')}</button></p>`:''));
  return `<div class="panel railcard exits" data-exits="${id}">${!d.selfDone&&labPassed(id)?doneHTML(id):''}<div class="eyebrow">${tx('Going further (optional): can you use it on a new case?','可选进阶：能用到新情况上吗？')}</div>`+
    `<div class="seg-ctl ex-tabs" role="group" aria-label="${esc(tx('Three ways to check','三种检验'))}">${T.map((x,i)=>`<button type="button" data-ex="tab" data-i="${i}" aria-pressed="${i===s.tab}">${i+1}. ${x}</button>`).join('')}</div><div class="ex-body">${b}</div></div>`;}
document.addEventListener('click',e=>{const b=e.target.closest&&e.target.closest('[data-ex]'); if(!b) return; const box=b.closest('[data-exits]'); if(!box) return;
  const id=box.dataset.exits, s=exState(id), d=EXD[id], k=b.dataset.ex, t=+b.dataset.t||0; if(!d) return;
  if(k==='go'){const q=(t===0?d.explain:d.transfer)[+b.dataset.j]; if(q&&q.go) q.go(); return;}
  if(k==='drill'){const c=drillCard(); if(c) jumpToEl(c); return;}
  if(k==='tab') s.tab=+b.dataset.i; else if(k==='ans') s.ans[t][+b.dataset.j]=+b.dataset.v; else if(k==='q') s.q[t]=+b.dataset.j; else return;
  const sel=`[data-ex="${k}"]`+(b.dataset.i!=null?`[data-i="${b.dataset.i}"]`:'')+(k==='ans'?`[data-v="${b.dataset.v}"]`:'');
  // at every width: the card grows or shrinks in place, so the browser's scroll anchoring must not keep what is below it (the next lab)
  // still and push the picked option up under the top bar (Batch 3 regression: gqa-mla, dpo and prefill-decode at 1440)
  pauseAnchor();
  box.outerHTML=exitsHTML(id,d); const nb=document.querySelector(`[data-exits="${id}"]`); if(!nb) return;
  const f=nb.querySelector(k==='q'?'.ex-opts .choice':sel)||nb.querySelector('[data-ex="tab"][aria-pressed="true"]'); if(f) f.focus({preventScroll:true});
  // a pick near the bottom of the screen: bring the verdict up, as far as needed and never so far that the options leave the screen (R3.10)
  const res=k==='ans'&&nb.querySelector('.ex-res'); if(!res) return;
  const bar=$('.topbar'), dk=$('#dock'), safe=(bar?bar.getBoundingClientRect().bottom:56)+(dk&&!dk.hidden?dk.offsetHeight:0)+8;
  const r=res.getBoundingClientRect(), o=nb.querySelector('.ex-opts').getBoundingClientRect();
  if(r.bottom>innerHeight-8){const dy=Math.min(r.bottom-innerHeight+16,o.top-safe); if(dy>=2) window.scrollBy({top:dy,behavior:reduceMotion()?'auto':'smooth'});}});
function exPassSync(){$$('[data-exits]').forEach(box=>{const id=box.dataset.exits, d=EXD[id]; if(!d||d.selfDone||!box.isConnected) return;
  const on=labPassed(id), h=box.querySelector(':scope>.ex-passed'); if(on&&!h) box.insertAdjacentHTML('afterbegin',doneHTML(id)); else if(!on&&h) h.remove();});}
['click','change'].forEach(k=>document.addEventListener(k,()=>{if(view==='home') return; setTimeout(exPassSync,0); setTimeout(exPassSync,900);}));
/* kit fixes (round K): one observer for what any view renders: hover-tip triggers become keyboard-reachable (tipReach), and a Next that a
   lab disables in place shows its why line again (a line rendered while Next was enabled is hidden until then) */
function whyShow(b){const pb=b.closest('.pbar'), host=pb?pb.parentElement:(b.closest('.steps')||{}).parentElement; if(!host) return;
  host.querySelectorAll(':scope>.steps>.step.current [data-next-why], :scope>.pbar>[data-next-why]').forEach(p=>{p.hidden=false;});}
/* comparison charts (kit polish K1): a value label never has a reference line (the dashed goal or budget line, .g) running through it,
   stays in its own row and on one line. The CSS alone put it right of its bar's end (or inside the end when the row had no room), so a
   bar that ended near the line had the line through its number (tool-scheduling 12.7 / 13.4 / 13.7 s against 14 s; chunk-rerank; the
   product-design budget; rag-failure moved its labels itself). Now the kit measures the label and the track and picks the first place
   that keeps clear of every line by CMP_CLR px: right of the bar's end; inside the bar's end; else the place nearest the bar's end just
   past or just before a line (beside the bar or inside it, never across its end); else, when nothing on the bar's row fits, under the
   bar's end. --cx carries the shift for the CSS (styles.css 1c). It runs when a chart is added and again when its track or its label
   changes size (a resize, “See all settings that pass” opened, a font that arrives late). A label a lab placed itself (an inline
   right, no .cmp-fit, or a lab rule that moves it elsewhere) is left where the lab put it. */
const CMP_CLR=4;
const cmpRO=typeof ResizeObserver==='function'?new ResizeObserver(es=>{const s=new Set();
  es.forEach(e=>{const t=e.target.classList.contains('cmp-track')?e.target:e.target.parentElement; if(!t||!t.isConnected){cmpRO.unobserve(e.target); return;} s.add(t);});
  cmpPlaceAll([...s]);}):null;
function cmpSeeAll(n){if(!n||n.nodeType!==1) return; const ts=n.matches('.cmp-track')?[n]:(n.firstElementChild?[...n.querySelectorAll('.cmp-track')]:[]); if(!ts.length) return;
  const fresh=ts.filter(t=>!t._cmpSeen); fresh.forEach(t=>{t._cmpSeen=1; if(cmpRO){cmpRO.observe(t); const v=t.querySelector(':scope>.cmp-val'); if(v) cmpRO.observe(v);}});
  cmpPlaceAll(fresh);}
const cmpOwn=v=>!v||v._cmpOff||!v.classList.contains('cmp-fit')||!/%$/.test(v.style.left)||!!v.style.right;
function cmpPlaceAll(ts){
  // reads first, then writes, then one check (a single layout per batch, however many rows)
  const jobs=[];
  for(const tr of ts){const v=tr.querySelector(':scope>.cmp-val'); if(cmpOwn(v)) continue; const tb=tr.getBoundingClientRect(), W=tb.width; if(!W) continue;   // hidden: the observer comes back when it shows
    const M=parseFloat(getComputedStyle(tr).marginRight)||0, ml=parseFloat(getComputedStyle(v).marginLeft)||0, tr0=tb.left;
    const w=v.getBoundingClientRect().width, xb=Math.max(0,Math.min(W,parseFloat(v.style.left)/100*W)), hi=W+M-2;
    const lines=[...tr.querySelectorAll(':scope>.g')].map(g=>{const r=g.getBoundingClientRect(); return [r.left-tr0,r.left-tr0+Math.max(1.5,r.width)];}).filter(([a,b])=>b>=-1&&a<=W+M+1);
    const clear=x0=>lines.every(([a,b])=>x0+w+CMP_CLR<=a||x0>=b+CMP_CLR);
    const beside=x0=>x0>=xb+ml-0.5, inside=x0=>x0>=2&&x0+w<=xb-2;   // right of the bar's end, or wholly inside the bar
    const fits=x0=>x0>=0&&x0+w<=hi&&clear(x0)&&(beside(x0)||inside(x0));
    let x0=null, lo=false;
    if(fits(xb+ml)) x0=xb+ml; else if(fits(xb-ml-w)) x0=xb-ml-w;
    else {let best=null, bd=Infinity; lines.forEach(([a,b])=>[b+CMP_CLR,a-CMP_CLR-w].forEach(c=>{if(!fits(c)) return; const d=c>=xb?c-xb:xb-(c+w); if(d<bd){bd=d; best=c;}})); x0=best;}
    if(x0==null){lo=true; x0=Math.max(0,Math.min(xb-w,hi-w));}   // nothing on the row fits: under the bar's end (the lines stop 3 px below the track)
    const par=tr.parentElement, blk=!!par&&(par.classList.contains('cmp-lo-row')||getComputedStyle(par).display==='block');   // a block row holds the room under the bar itself (.cmp-lo-row); a grid or flex row already does
    jobs.push({tr,v,x0,lo,xb,ml,tr0,h:v.offsetHeight,par,blk});}
  for(const j of jobs){j.tr.classList.toggle('cmp-lo',j.lo); j.v.classList.toggle('cmp-lo',j.lo); if(j.par) j.par.classList.toggle('cmp-lo-row',j.lo&&j.blk); if(j.lo) j.tr.style.setProperty('--lo',(j.h+7)+'px'); else j.tr.style.removeProperty('--lo');
    j.v.style.setProperty('--cx',(j.x0-j.xb-j.ml).toFixed(2)+'px');}
  // a lab rule that places the label its own way wins over --cx: then the kit leaves that label alone
  for(const j of jobs){const r=j.v.getBoundingClientRect(); if(Math.abs(r.left-(j.tr.getBoundingClientRect().left+j.x0))>1.5){j.v._cmpOff=1; j.v.style.removeProperty('--cx'); j.v.classList.remove('cmp-lo'); j.tr.classList.remove('cmp-lo'); if(j.par) j.par.classList.remove('cmp-lo-row'); j.tr.style.removeProperty('--lo');}}}
if(document.fonts&&document.fonts.addEventListener) document.fonts.addEventListener('loadingdone',()=>cmpPlaceAll([...document.querySelectorAll('.cmp-track')].filter(t=>t._cmpSeen)));
const kitObs=window.MutationObserver?new MutationObserver(recs=>{for(const r of recs){
  if(r.type==='childList'){r.addedNodes.forEach(n=>{if(n.nodeType===1){tipReachAll(n); fillAll(n); cmpSeeAll(n);}});}
  else if(r.attributeName==='disabled'){const b=r.target; if(b.disabled&&b.matches('[data-nav="next"]')) whyShow(b);}
  else if(r.attributeName==='min'||r.attributeName==='max'||r.attributeName==='step'||r.attributeName==='value'){if(r.target.type==='range') setFill(r.target);}   // K5
  else tipReach(r.target);}}):null;
function kitObserve(){tipReachAll(document.body); fillAll(document.body); cmpSeeAll(document.body); if(kitObs) kitObs.observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['data-tip','data-tipx','data-tipfn','data-jump','disabled','min','max','step','value']});}
const UI={
  flash:flashEl, jumpToEl, obsBump, setRange:setFill, reveal:revealTop, done:id=>doneHTML(id),
  /* set every Next in root (the card's and the part bar's) and, if given, the text of their why lines (steps({nextWhy})) */
  setNext:(root,disabled,why)=>{$$('[data-nav="next"]',root).forEach(b=>{b.disabled=!!disabled;}); $$('[data-next-why]',root).forEach(p=>{if(why!=null) p.innerHTML=why; if(disabled) p.hidden=false;});},
  initRanges:root=>$$('input[type=range]',root).forEach(setFill),
  setPressed:(group,v)=>$$('button[data-v]',group).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.v===String(v)))),
  status:(ok,text)=>`<span class="status ${ok?'ok':'bad'}">${esc(text)}</span>`,
  range:o=>`<div class="${o.inline?'inline-ctl':'ctl'}"${o.id?` id="${o.id}"`:''}><label for="${o.id}-in"><span>${esc(o.label)}</span> <output>${o.out||''}</output></label>`+
    `<input type="range" id="${o.id}-in" data-key="${esc(o.key)}" min="${o.min}" max="${o.max}" step="${o.step||1}" value="${o.value}">${o.note?`<div class="ctl-note">${o.note}</div>`:''}</div>`,
  seg:o=>`<div class="${o.inline?'inline-ctl':'ctl'}"${o.id?` id="${o.id}"`:''}><span class="lbl">${esc(o.label)}</span>`+
    `<div class="seg-ctl" role="group" aria-label="${esc(o.label)}" data-key="${esc(o.key)}">${o.options.map(([v,txt,dis])=>`<button type="button" data-v="${esc(v)}" aria-pressed="${String(v)===String(o.value)}"${dis?' disabled':''}>${escMath(txt)}</button>`).join('')}</div>`+
    `${o.note?`<div class="ctl-note">${o.note}</div>`:''}</div>`,
  levers:o=>`<div class="levers" role="group" aria-label="${esc(o.aria||'')}">${o.options.map(op=>`<button type="button" class="choice lever" data-lever="${esc(op.v)}" aria-pressed="${String(op.v)===String(o.value)}"><span class="lv-t"><b>${escMath(op.title)}</b>${op.desc?`<span class="lv-d">${escMath(op.desc)}</span>`:''}</span><span class="lv-r mono">${op.result||''}</span></button>`).join('')}</div>`,
  /* kit fixes (round K): a passing row is marked on its track (green ring, any value, 0% too); a value label stays on one line, right of
     its bar, or inside the bar's end when the chart has no room for it (.cmp-fit, CSS); the goal label is centred on the goal line and
     kept inside the chart near either end (.cmp-gx, CSS). --v and --g carry the positions as numbers for those clamps */
  cmpChart:o=>{
    /* brief:true (the owner's review of batch 8, 2026-10-07: “只需要给读者看……选错了的设置……和所有可行设置即可，无需一一列出”): the chart keeps
       only the rows a reader needs after Submit, the settings that meet the goal, the reader's own pick and the rows the lab tagged (the
       initial setup), and says below how many it left out. When more than briefMax (8) rows would stay and the lab marks its best rows
       (best:true), the chart shows those with the pick and the tagged rows, and the rest of the passing rows wait behind a closed
       “See all settings that pass (N)” */
    const keep=r=>r.ok||r.me||(r.tags&&r.tags.length), vis=o.brief?o.rows.filter(keep):o.rows, pass=o.rows.filter(r=>r.ok);
    const fold=o.brief&&vis.length>(o.briefMax||8)&&pass.some(r=>r.best), shown=fold?o.rows.filter(r=>r.me||r.best||(r.tags&&r.tags.length)):vis;
    const mx=o.max||Math.max(1e-9,...vis.map(r=>r.value))*1.15, pc=v=>Math.max(0,v)/mx*100, pct=v=>pc(v).toFixed(2)+'%';
    const row=r=>{const v=pc(r.value);
      return `<div class="cmp-row${r.ok?' ok':''}${r.me?' me':''}"><div class="cmp-lab"><span>${esc(r.label)}</span>${(r.tags||[]).map(tg=>`<span class="${tg.me?'cmp-you':'cmp-tag'}">${esc(tg.text)}</span>`).join('')}</div>`+
      `<div class="cmp-track cmp-cq"><i style="width:${pct(r.value)}"></i>${o.goal!=null?`<b class="g" style="left:${pct(o.goal)}"></b>`:''}<span class="cmp-val mono${v<=100?' cmp-fit':''}" style="left:${pct(r.value)};--v:${v.toFixed(2)}">${esc(r.text!=null?r.text:String(r.value))}</span></div></div>`;};
    const axis=o.goal!=null?`<div class="cmp-axis cmp-gx"><span style="left:${pct(o.goal)};--g:${Math.min(100,pc(o.goal)).toFixed(2)}">${esc(o.goalLabel||'')}</span></div>`:'';
    const nOut=o.rows.length-vis.length, rest=fold?pass.filter(r=>!shown.includes(r)):[];
    const note=o.brief&&nOut>0?`<p class="small muted cmp-note">${esc(typeof o.hiddenText==='function'?o.hiddenText(nOut):tx(nOut===1?'The other option misses the goal and is not shown.':`The other ${nOut} options miss the goal and are not shown.`,`其余 ${nOut} 种没达到目标，图里没有列出。`))}</p>`:'';
    const more=rest.length?`<details class="cmp-more"><summary>${tx(`See all settings that pass (${pass.length})`,`看全部可行的设置（${pass.length} 种）`)}</summary>`+
      `<div class="cmp" role="img" aria-label="${esc(tx('Every setting that meets the goal: ','全部达到目标的设置：')+pass.map(r=>`${r.label}${tx(': ','：')}${r.text!=null?r.text:r.value}`).join(tx('; ','；')))}"><div class="cmp-bars" aria-hidden="true">${pass.map(row).join('')}${axis}</div></div></details>`:'';
    return `<div class="cmp" role="img" aria-label="${esc(o.aria||o.title)}"><div class="cmp-title">${esc(o.title)}</div><div class="cmp-bars" aria-hidden="true">${shown.map(row).join('')}${axis}</div>`+
      cmpLegend(o)+`</div>`+note+more;},
  steps:o=>{const q=partQ(o.body), a=partLog(labNow(),o.steps.length,o.current,q), sq=!!(o.next&&q.open&&!o.nextDisabled);
    const items=o.steps.map((s,i)=>{const cls=stepCls(a,i,o.current), head=stepHeadHTML(a,i,o.current,s.title);
      if(i!==o.current) return `<li class="step ${cls}">${head}</li>`;
      const nav=`<div class="stepnav">${o.back?`<button type="button" class="btn ghost sm" data-nav="back">${tx('Back','上一部分')}</button>`:''}${o.showMe?`<button type="button" class="btn ghost sm" data-nav="showme" data-tipx="${esc(SHOWME_TIP())}">${tx('Show me','演示给我看')}</button>`:''}${o.next?`<button type="button" class="btn${sq?' ghost':''} sm" data-nav="next"${o.nextDisabled?' disabled':''}>${nextTxt(sq)}</button>`:''}${o.skip&&!sq?`<button type="button" class="linkbtn" data-nav="skip">${tx('Skip','跳过')}</button>`:''}</div>`;
      return `<li class="step ${cls}">${head}<div class="step-b">${o.body}<div class="obs" data-obs hidden></div>${nav}${nextWhyHTML(o)}</div></li>`;}).join('');
    return `<div class="rail-top"><span class="eyebrow">${t('modeGuided')}${tx(`: part ${o.current+1} of ${o.steps.length}`,`：第 ${o.current+1} / ${o.steps.length} 部分`)}</span></div><ol class="steps">${items}</ol>${o.m2?pbarHTML({...o,ans:a,q}):''}`;}   // the bar after the list in the DOM: first match for [data-nav] stays the card's own button; CSS puts the bar first on phones
};
/* ---------- the lab's flow (owner's decision, 2026-10-11 19:28 UTC) ----------
   Every lab: on the left a contents list of its flow (知识梳理 when the lab has a primer, 分步引导, 应用场景, 自由调参, 挑战题), on the
   right the stage or the primer. The mode buttons, the fidelity and primer buttons and "copy setup link" are gone from the top; the
   fidelity notes, without "what is modeled", close the page like a paper's references. The labs keep their own modes: 应用场景 and
   自由调参 are both the lab's sandbox mode, the first with the rail's scenarios and the stage, the second with the settings panel and the
   stage. The kit moves the list into the bench (flowDecorate) after each mount; FLOW keeps the step per lab and PROG saves it. */
const F2M={guided:'guided',scen:'sandbox',free:'sandbox',chal:'challenge'}, M2F={guided:'guided',sandbox:'scen',challenge:'chal'};
const FLOW={}, LASTMODE={}, PRIMERS={};
const flowName=f=>f==='primer'?tx('Basics','知识梳理'):f==='free'?t('ctrlEyebrow'):t(MODENAME[F2M[f]]);
const FLOW_FREE_TIP=()=>tx('All of the lab’s settings, to try your own setups.','把 Lab 的设置全部打开，自己调。');
const flowNavHTML=(id,mode,descs)=>`<nav class="flow modes" data-flow-nav data-lab="${esc(id)}" aria-label="${esc(tx('The steps of this lab','这个 Lab 的流程'))}">${flowListHTML(id,mode,descs)}</nav>`;
function flowListHTML(id,mode,descs){descs=descs||{}; const fl=[...(PRIMERS[id]?['primer']:[]),'guided','scen','free','chal'], now=M2F[mode]||'guided';
  return `<ol class="fl-list">`+
    fl.map((f,i)=>{const m=F2M[f], tip=f==='primer'?'':(f==='free'?FLOW_FREE_TIP():(descs[m]||''));
      return `<li class="fl-i" data-fl="${f}"><button type="button" data-flow="${f}"${m?` data-mode="${m}"`:''} aria-pressed="${f===now}"${tip?` data-tipx="${esc(tip)}"`:''}>`+
        `<span class="fl-n">${i+1}</span><span class="fl-t">${esc(flowName(f))}</span>${f==='primer'?`<span class="fl-m">${esc(minsTxt(PRIMERS[id].min||8))}</span>`:''}</button>`+
        (f==='primer'?primerTocHTML(id):'')+`</li>`;}).join('')+`</ol>`+primerSeeHTML(id);}
/* a lab without its own primer points to the primer of the lab that starts its series (the owner agreed to the plan, 2026-10-12 07:13 UTC);
   the line shows only once that primer exists, and its link opens that lab on the primer */
const PRIMER_SEE={'tokenization-cost':'architecture',mha:'architecture',rope:'architecture',transformer:'architecture','kv-basics':'architecture',sampling:'architecture',
  'gqa-mla':'architecture',moe:'architecture','kv-cache':'architecture','flash-attention':'roofline','softmax-numerics':'number-formats',optimizers:'backprop','train-debug':'backprop',
  'pass-k':'eval-ci','llm-judge':'eval-ci',contamination:'eval-ci','eval-capstone':'eval-ci',
  'paged-attention':'prefill-decode','continuous-batching':'prefill-decode','chunked-prefill':'prefill-decode','prefix-caching':'prefill-decode','serving-capstone':'prefill-decode',
  quantization:'prefill-decode','pd-disagg':'prefill-decode','multi-gpu-serving':'prefill-decode',admission:'prefill-decode',
  'model-parallel':'parallelism','scaling-law':'training-memory','data-pipeline':'training-memory',checkpointing:'training-memory','train-capstone':'training-memory',
  lora:'sft',grpo:'ppo',rlvr:'ppo','post-capstone':'sft','vector-index':'embeddings','chunk-rerank':'embeddings','rag-failure':'embeddings',
  'agent-reliability':'agent-anatomy','context-eng':'agent-anatomy','tool-scheduling':'agent-anatomy','prompt-injection':'agent-anatomy','agent-capstone':'agent-anatomy'};
function primerSeeHTML(id){const p=!PRIMERS[id]&&PRIMER_SEE[id]; if(!p||!PRIMERS[p]||!LAB[p]) return ''; const n=L(LAB[p].s);
  return `<p class="fl-see">${tx('Read first: ','建议先读：')}<a href="#${esc(p)}" data-see-primer="${esc(p)}">${esc(tx(`the basics of “${n}”`,`“${n}”的知识梳理`))}</a></p>`;}
// that link, and a primer's link to another lab that has a primer, open that lab on its primer
document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a[data-see-primer], .primer a[href^="#"]'); if(!a) return;
  const id=a.dataset.seePrimer||a.getAttribute('href').slice(1); if(PRIMERS[id]) FLOW[id]='primer';},true);
// the fidelity notes at the page's end: "what is modeled" dropped, the other blocks (simplified, assumed, left out, sources...) kept with
// their own headings; a note without headings (a few labs) is kept whole, as its paragraphs mix both
const MODELED=/^(Modeled|What .*model|建模了什么|模型里算了什么)/i;
function endnotesInner(html){if(!html) return ''; const tp=document.createElement('template'); tp.innerHTML=html;
  const kids=[...tp.content.children], g=kids.findIndex(e=>e.classList.contains('pop-grid'));
  if(g<0) return `<div class="en-block">${html}</div>`;
  const blocks=[...kids[g].children].map(d=>{const eb=d.querySelector(':scope>.eyebrow'), h=eb?eb.textContent.trim():''; if(eb) eb.remove(); return {h,html:d.innerHTML};}).filter(b=>!MODELED.test(b.h));
  // what follows the grid (a sources line, or a "Sources" eyebrow and its list) is one more block
  const rest=kids.slice(g+1); let tail='';
  if(rest.length){const eb=rest[0].classList.contains('eyebrow')?rest.shift():null; tail=`<div class="en-block">${eb?`<h3>${esc(eb.textContent.trim())}</h3>`:''}${rest.map(e=>e.outerHTML).join('')}</div>`;}
  return blocks.map(b=>`<div class="en-block">${b.h?`<h3>${esc(b.h)}</h3>`:''}${b.html}</div>`).join('')+tail;}
const endnotesHead=id=>`<h2 id="${esc(id)}-enH">${tx('Simplifications, assumptions and sources','简化、假设和来源')}</h2>`;
function endnotesHTML(id,html){const inner=endnotesInner(html); return inner?`<section class="wrap endnotes" id="${esc(id)}-fid" data-endnotes aria-labelledby="${esc(id)}-enH">${endnotesHead(id)}<div class="en-body">${inner}</div></section>`:'';}
const navMode=nav=>{const b=nav.querySelector('button[data-mode][aria-pressed="true"]'); return b?b.dataset.mode:null;};
function flowInit(id,nav){if(FLOW[id]!==undefined) return; const r=PROG.get(id), f=r&&typeof r.flow==='string'?r.flow:null;
  FLOW[id]=f&&nav.querySelector(`[data-flow="${f}"]`)?f:(PRIMERS[id]&&!flowTestOff()?'primer':(M2F[navMode(nav)]||'guided'));}
// automated browsers (the older tests, written for the lab's own steps) open a lab on its own mode, as PROG starts them clean; a test of
// the primer opts in with window.__aspPrimer
const flowTestOff=()=>{try{return !!navigator.webdriver&&!window.__aspPrimer;}catch(e){return false;}};
function flowDecorate(scope){(scope?[...scope.querySelectorAll('[data-flow-nav]')]:$$('[data-flow-nav]')).forEach(nav=>{
  const id=nav.dataset.lab, head=nav.closest('.labhead'), app=head?head.parentElement:nav.closest('.bench')&&nav.closest('.bench').parentElement; if(!app) return;
  const bench=app.querySelector('.bench'); if(!bench) return;
  const m=navMode(nav);
  if(nav.parentElement!==bench) bench.prepend(nav);
  bench.classList.add('flowed');
  if(!bench.querySelector(':scope>.controls')){const li=nav.querySelector('[data-fl="free"]'); if(li) li.remove();}
  $$('.fl-n',nav).forEach((n,i)=>{n.textContent=String(i+1);});
  flowInit(id,nav);
  // the lab's mode when this list was drawn: a list drawn again (the lab's redraw, a language switch) keeps the step, unless the mode moved
  if(m){if(LASTMODE[id]!==undefined&&m!==LASTMODE[id]&&F2M[FLOW[id]]!==m) FLOW[id]=M2F[m]||FLOW[id]; LASTMODE[id]=m;
    if(FLOW[id]!=='primer'&&F2M[FLOW[id]]!==m) FLOW[id]=M2F[m];}
  if(!nav.querySelector(`[data-flow="${FLOW[id]}"]`)) FLOW[id]=M2F[m]||'guided';
  if(PRIMERS[id]&&!bench.querySelector(':scope>.primer')){nav.insertAdjacentHTML('afterend',primerHTML(id)); primerStart(bench.querySelector(':scope>.primer'),id);}
  const en=app.querySelector(':scope>[data-endnotes]'); if(en&&en.nextElementSibling) app.appendChild(en);
  flowApply(nav);});}
function flowApply(nav){const id=nav.dataset.lab, bench=nav.closest('.bench'); if(!bench) return; flowInit(id,nav); const f=FLOW[id];
  if(bench.dataset.flow!==f){bench.dataset.flow=f; bench.classList.remove('peek'); if(f==='primer') primerKick(bench); dockSoon();}
  const c=bench.querySelector(':scope>.controls'); if(c&&f==='free') c.classList.add('open');   // phones: the settings drawer is the step itself
  if(flowTestOff()) bench.classList.add('compat');   // automation without __aspPrimer: the older per-lab tests drive the settings panel in every step
  $$('button[data-flow]',nav).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.flow===f)));
  if(f==='primer') primerTocSync(bench);}
// the lab changed its mode (its own setMode, through SHELL.setMode or the KV lab's): the step follows, unless it already shows that mode
function flowSync(scope,m){const nav=scope&&scope.querySelector('[data-flow-nav]'); if(!nav) return; const id=nav.dataset.lab;
  if(LASTMODE[id]!==undefined&&m!==LASTMODE[id]&&F2M[FLOW[id]]!==m) FLOW[id]=M2F[m]||FLOW[id];
  LASTMODE[id]=m; if(nav.closest('.bench.flowed')) flowApply(nav); else $$('button[data-flow]',nav).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.flow===M2F[m])));}
function flowSet(nav,f){const id=nav.dataset.lab; if(!nav.querySelector(`[data-flow="${f}"]`)) return; FLOW[id]=f; flowApply(nav);}
function flowTop(bench){const bar=$('.topbar'), top=(bar?bar.offsetHeight:56)+12, r=bench.getBoundingClientRect();
  if(r.top<top||r.top>window.innerHeight*0.6) window.scrollTo({top:Math.max(0,window.scrollY+r.top-top),behavior:'auto'});}
// a click on the list sets the step before the lab's own handler (bindHeader, or the KV lab's) switches its mode
document.addEventListener('click',e=>{const b=e.target.closest&&e.target.closest('[data-flow-nav] button[data-flow], [data-flow-go]'); if(!b) return;
  const nav=b.closest('[data-flow-nav]')||(b.closest('.bench')&&b.closest('.bench').querySelector('[data-flow-nav]')); if(!nav) return;
  if(b.dataset.flowGo){const t=nav.querySelector(`button[data-flow="${b.dataset.flowGo}"]`); e.preventDefault(); if(t) t.click(); return;}
  flowSet(nav,b.dataset.flow); const bench=nav.closest('.bench'); if(bench&&b.dataset.flow==='primer') flowTop(bench);},true);
// a lab's jump to a part the current step hides (e.g. "jump to Settings" from a guided part): the settings panel shows below the card
// (.peek), or in the sandbox the step becomes 自由调参; any part in the primer step: back to the lab's own step
function flowReveal(el){if(!el||!el.closest) return; const bench=el.closest('.bench.flowed'); if(!bench) return; const part=[...bench.children].find(c=>c.contains(el)); if(!part) return;
  const nav=bench.querySelector(':scope>[data-flow-nav]'), f=bench.dataset.flow; if(!nav||part===nav||part.classList.contains('primer')) return;
  if(f==='primer'||bench.classList.contains('compat')) return;   // compat: the settings panel already shows in every step
  if(part.classList.contains('controls')&&f!=='free'){if(navMode(nav)==='sandbox') flowSet(nav,'free'); else bench.classList.add('peek');}}
{const siv=Element.prototype.scrollIntoView; Element.prototype.scrollIntoView=function(...a){try{flowReveal(this);}catch(e){} return siv.apply(this,a);};}
let flowObs=null;
function flowWatch(){if(flowObs||typeof MutationObserver!=='function') return; flowObs=new MutationObserver(()=>{if(view==='home') return;
    if($$('[data-flow-nav]').some(n=>!n.parentElement||!n.parentElement.classList.contains('bench')||!n.parentElement.classList.contains('flowed')||(PRIMERS[n.dataset.lab]&&!n.parentElement.querySelector(':scope>.primer')))) flowDecorate();});
  ['#view-dyn','#view-lab'].forEach(s=>{const el=$(s); if(el) flowObs.observe(el,{childList:true,subtree:true});});}

/* ---------- primers (知识梳理): a lab's optional basics, read top to bottom before its interactive steps ----------
   A.registerPrimer(id, {min, sections:[{h:()=>title, parts:[{html:()=>string} | {fig:[{html:()=>string, cap:()=>string, ms}], cls}]}]}).
   A figure shows the lab's own pictures as a display, not as controls (inert): its frames play in turn while it is on screen, each
   frame drawn afresh so its animation starts again; with reduced motion the frames stand one under another. A figure of several
   frames has ‹ › buttons under it and takes a sideways swipe on touch screens; once the reader moves a figure by hand, no figure of
   this primer plays on by itself any more, so nothing jumps away before it has been read (the owner, 2026-10-12). */
const primerTocHTML=id=>{const p=PRIMERS[id]; return `<ol class="fl-sub">${p.sections.map((s,i)=>`<li><a href="#${esc(id)}-pr-${i+1}" data-psec="${i+1}">${esc(s.h())}</a></li>`).join('')}</ol>`;};
function primerHTML(id){const p=PRIMERS[id], n=p.sections.length;
  return `<section class="primer" data-primer="${esc(id)}" aria-label="${esc(tx('Basics','知识梳理'))}">`+p.sections.map((s,i)=>
    `<article class="pr-sec" id="${esc(id)}-pr-${i+1}" data-psec="${i+1}"><div class="pr-k">${i+1}/${n}</div><h2 class="pr-h">${esc(s.h())}</h2>`+
      s.parts.map((q,j)=>q.fig?`<figure class="pr-fig${q.fig.length>1?' pr-multi':''}" data-fig="${i}.${j}"><div class="pr-frames" inert></div><figcaption class="pr-cap"></figcaption>${q.fig.length>1?
        `<div class="pr-nav"><button type="button" class="pr-prev" aria-label="${esc(tx('Previous picture','上一张'))}"><span aria-hidden="true">‹</span></button>`+
        `<div class="pr-dots" aria-hidden="true">${q.fig.map(()=>'<i></i>').join('')}</div><span class="pr-num" aria-live="polite"></span>`+
        `<button type="button" class="pr-next" aria-label="${esc(tx('Next picture','下一张'))}"><span aria-hidden="true">›</span></button></div>`:''}</figure>`:`<div class="pr-text">${q.html()}</div>`).join('')+`</article>`).join('')+
    `<div class="pr-end"><button type="button" class="btn" data-flow-go="guided">${tx('Next: the guided parts','下一步：分步引导')} <span aria-hidden="true">→</span></button></div></section>`;}
const PRF=new WeakMap();   // figure element → {frames, k, t, on}
function primerStart(sec,id){if(!sec) return; const p=PRIMERS[id];
  $$('.pr-fig',sec).forEach(fig=>{const [i,j]=fig.dataset.fig.split('.').map(Number), q=p.sections[i].parts[j], fr=q.fig, box=$('.pr-frames',fig);
    box.innerHTML=fr.map((f,k)=>`<div class="pr-fr${q.cls?' '+esc(q.cls):''}" data-k="${k}">${f.html()}</div>`).join('');
    $$('.pr-fr',box).forEach(x=>{x.querySelectorAll('input,button,select,a,[tabindex]').forEach(c=>c.setAttribute('tabindex','-1'));});
    const st={frames:fr,k:0,t:0,on:false}; PRF.set(fig,st); prShow(fig,0,false); if(q.mount) q.mount(fig);
    if(fr.length>1){let x0=null, y0=0;   // a sideways swipe on the figure (its frames are inert, so the touch lands on the figure)
      fig.addEventListener('touchstart',e=>{const t=e.touches[0]; x0=e.touches.length===1?t.clientX:null; y0=t.clientY;},{passive:true});
      fig.addEventListener('touchend',e=>{if(x0===null) return; const t=e.changedTouches[0], dx=t.clientX-x0, dy=t.clientY-y0; x0=null;
        if(Math.abs(dx)>40&&Math.abs(dx)>1.5*Math.abs(dy)) prStep(fig,dx<0?1:-1);},{passive:true});}});
  sec.addEventListener('click',e=>{const b=e.target.closest('.pr-prev,.pr-next'); if(b){const f=b.closest('.pr-fig'); if(f) prStep(f,b.classList.contains('pr-next')?1:-1);}});
  if(typeof IntersectionObserver==='function'){const io=new IntersectionObserver(es=>es.forEach(en=>{const st=PRF.get(en.target); if(!st) return; st.on=en.isIntersecting&&en.intersectionRatio>=0.35;
      if(st.on) prRun(en.target); else {clearTimeout(st.t); st.t=0;}}),{threshold:[0,0.35,0.7]});
    $$('.pr-fig',sec).forEach(f=>io.observe(f)); sec._io=io;}
  sec.addEventListener('click',e=>{const a=e.target.closest('a[data-psec]'); if(a){e.preventDefault(); prGo(sec,+a.dataset.psec);}});
  primerTocSync(sec.closest('.bench'));}
function prShow(fig,k,fresh){const st=PRF.get(fig); if(!st) return; st.k=k; const fr=$$('.pr-fr',fig);
  fr.forEach((x,i)=>{const on=i===k; x.classList.toggle('on',on); if(on&&fresh){x.innerHTML=st.frames[i].html(); x.querySelectorAll('input,button,select,a,[tabindex]').forEach(c=>c.setAttribute('tabindex','-1'));}});
  const c=st.frames[k].cap; $('.pr-cap',fig).innerHTML=c?c():''; $$('.pr-dots i',fig).forEach((d,i)=>d.classList.toggle('on',i===k));
  const n=st.frames.length, pv=$('.pr-prev',fig), nx=$('.pr-next',fig), nm=$('.pr-num',fig);
  if(pv) pv.disabled=k===0; if(nx) nx.disabled=k===n-1; if(nm) nm.textContent=`${k+1}/${n}`;}
// a hand step: this figure moves one frame (no wrap-around), and no figure of this primer plays on by itself after that
function prStep(fig,d){const st=PRF.get(fig); if(!st) return; const k=st.k+d; if(k<0||k>=st.frames.length) return;
  const sec=fig.closest('.primer'); if(sec){sec._manual=true; $$('.pr-fig',sec).forEach(f=>{const s=PRF.get(f); if(s){clearTimeout(s.t); s.t=0;}});}
  prShow(fig,k,true);}
function prRun(fig){const st=PRF.get(fig); if(!st||st.t||!fig.isConnected) return; if(st.frames.length<2||reduceMotion()) return;
  const sec=fig.closest('.primer'); if(sec&&sec._manual) return;
  const tick=()=>{st.t=0; if(!fig.isConnected||!st.on) return; prShow(fig,(st.k+1)%st.frames.length,true); st.t=setTimeout(tick,st.frames[st.k].ms||4800);};
  st.t=setTimeout(tick,st.frames[st.k].ms||4800);}
function primerKick(bench){const sec=bench.querySelector(':scope>.primer'); if(!sec) return; $$('.pr-fig',sec).forEach(f=>{const st=PRF.get(f); if(st&&st.on) prRun(f);});}
function prGo(sec,k){const a=sec.querySelector(`[data-psec="${k}"].pr-sec`); if(!a) return; const bar=$('.topbar'), top=(bar?bar.offsetHeight:56)+12;
  window.scrollTo({top:Math.max(0,window.scrollY+a.getBoundingClientRect().top-top),behavior:reduceMotion()?'auto':'smooth'});}
// the section being read: the list under 知识梳理 marks it, and PROG keeps it with the lab's record
const PRAT={};
function primerTocSync(bench){if(!bench) return; const sec=bench.querySelector(':scope>.primer'), nav=bench.querySelector(':scope>[data-flow-nav]'); if(!sec||!nav) return;
  const bar=$('.topbar'), line=(bar?bar.offsetHeight:56)+innerHeight*0.25; let k=1;
  $$('.pr-sec',sec).forEach(a=>{if(a.getBoundingClientRect().top<=line) k=+a.dataset.psec;});
  PRAT[nav.dataset.lab]=k; $$('.fl-sub a',nav).forEach(a=>{const on=+a.dataset.psec===k; a.classList.toggle('on',on); if(on) a.setAttribute('aria-current','true'); else a.removeAttribute('aria-current');});}
{let rq=0; window.addEventListener('scroll',()=>{if(rq||view==='home') return; rq=requestAnimationFrame(()=>{rq=0; const b=$('.bench.flowed[data-flow="primer"]'); if(b&&b.offsetParent) primerTocSync(b);});},{passive:true});}
// the nav's list under 知识梳理 jumps to a section
document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('[data-flow-nav] .fl-sub a[data-psec]'); if(!a) return; e.preventDefault();
  const bench=a.closest('.bench'), sec=bench&&bench.querySelector(':scope>.primer'); if(!sec) return; const nav=bench.querySelector(':scope>[data-flow-nav]'); flowSet(nav,'primer'); prGo(sec,+a.dataset.psec);});

const SHELL={
  header:o=>{const w=WORLDS[o.world]||{name:{en:'',zh:''}}, P=o.id;
    return `<div class="wrap labhead"><div class="crumbs"><a href="#map" data-tipfn="crumbTree">${t('crumbMap')}</a><span>/</span><a href="#map~${esc(o.world)}" data-tipx="${esc(crumbWorldTip(o.world))}">${esc(L(w.name))}</a><span>/</span><span>${esc(o.short)}</span>${crumbNavHTML(P)}</div>`+
      `<h1 class="labtitle">${esc(o.title)}</h1><div class="lab-sub">${o.outcome?outcomeHTML(o.outcome):''}<div class="labmeta"><span class="tag">${minsTxt(o.minutes)}</span></div></div>`+
      flowNavHTML(P,o.mode,o.descs)+`</div>`+endnotesHTML(P,o.fidPop);},   // o.primer, o.primerPop: no longer shown (the lab's primer replaces them)
  bindHeader:(root,o)=>{if(o.token) TOKFN[o.id]=o.token; root.addEventListener('click',e=>{const b=e.target.closest('button'); if(!b||!root.contains(b)) return;
      if(b.dataset.pop){const pop=document.getElementById(b.dataset.pop), open=pop.hidden; pop.hidden=!open; b.setAttribute('aria-expanded',String(open)); return;}
      if(b.dataset.mode&&b.closest('.modes')){o.onMode(b.dataset.mode); return;}
      if(b.hasAttribute('data-share')){const frag=shareText('#'+o.id+'~'+o.token()), box=root.querySelector('[data-sharebox]'), inp=box.querySelector('input');
        inp.value=frag; box.hidden=false; const done=()=>flash(b,tx('Copied','已复制'),'share'), fb=()=>{inp.focus(); inp.select();};
        try{ if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(frag).then(done,fb); else fb(); }catch(err){fb();}}});},
  setMode:(root,m)=>{flowSync(root,m);}
};
window.__kv={compute,MODELS,GPUS};
window.ASP={
  lang:()=>lang, t, tx, L, addStrings:o=>{Object.assign(STR.en,o.en||{}); Object.assign(STR.zh,o.zh||{});},
  $, $$, esc, reduceMotion, KiB, MiB, GiB, TiB,
  fmt:{B:fmtB,G:fmtG,num:fmtNum,int:fmtInt,big:fmtBig,tok:fmtTok,users:usersTxt,gpus:gpusTxt,mins:minsTxt,delta:deltaVs},
  tip:{show:(el,html)=>{tipFor=el; showTipHTML(el,html);},hide:hideTip,quiet:quietTips},
  ui:{...UI,def:DEF,big:BIG,math:texMath,exits:exitsHTML}, shell:{...SHELL,nextLab:nextLabHTML,labLink,toScenarios:toScenariosHTML,toChallenge:toChallengeHTML,crumbNav:crumbNavHTML,crumbWorldTip},   // crumbNav, crumbWorldTip: a lesson page's own breadcrumb row (2026-10-11)
  terms:{...TERMS,qhead:QH_TIP,kvhead:KVH_TIP,gqa:GQA_TIP,mla:MLA_TIP,tpmla:TPMLA_TIP},
  // a placeholder teaser (def.placeholder, e.g. set C's tiles for labs not built yet) yields to the lab's own teaser, whichever file loads first
  registerLab:def=>{DYN[def.id]=def;}, registerPrimer:(id,def)=>{PRIMERS[id]=def;}, flow:{set:(id,f)=>{const n=$(`[data-flow-nav][data-lab="${id}"]`); if(n) flowSet(n,f);},now:id=>FLOW[id]||null}, registerTeaser:(id,def)=>{if(def&&def.placeholder&&TEASERS[id]) return; TEASERS[id]=def;}, registerHero:def=>{HERO=def;},
  hash:{write:tok=>{if(view!=='dyn'||!dyn) return; const m=$('#view-dyn .modes button[aria-pressed="true"]'), sb=m&&m.dataset.mode==='sandbox';
    try{history.replaceState(null,'','#'+dyn.id+(tok&&sb?'~'+tok:''));}catch(e){}}},
  announce:announceText,
  dock:{refresh:()=>{if(!$('#dock').hidden) renderDockAny(true); dockSoon();}, render:()=>{if(!$('#dock').hidden) renderDockAny(false); dockSoon();}, update:dockSoon},
  /* kind 'route': on the chosen path, the fields as before; kind 'branch': no path chosen that has the lab, branch: its branch, next and
     n / of along the branch; at its end (end: true) next is the suggested next step (rec: true, no stop number) and join the path it is from */
  catalog:{LABS,LAB,WORLDS,PATHS,labStatus,route:()=>PROG.route(),nextInfo:id=>{const x=nextInfo(id); if(!x) return null;
      if(x.kind==='branch') return {kind:'branch',branch:x.w,pid:null,to:null,next:x.l?x.l.id:(x.rec?x.rec.l.id:null),n:x.l?x.n:null,of:x.l?x.of:null,done:false,pause:null,skip:(x.skip||[]).map(l=>l.id),end:!!x.end,rec:!!x.rec,join:x.rec?x.rec.pid:null};
      return {kind:'route',pid:x.pid,to:x.to||null,next:x.l?x.l.id:null,n:x.n||null,of:x.of||null,done:!!x.done,pause:x.pause?x.pause.map(l=>l.id):null,skip:(x.skip||[]).map(l=>l.id)};},
    prevInfo:id=>{const x=prevInfo(id); if(!x) return null; const b=x.kind==='branch';
      return {kind:x.kind,branch:b?x.w:null,pid:b?null:x.pid,to:b?null:(x.to||null),prev:x.l?x.l.id:null,n:x.n||null,of:x.of||null,cont:!!x.cont,skip:(x.skip||[]).map(l=>l.id)};}},
  kv:{compute,MODELS,GPUS,PREC,PL,DEFAULT,perGpuBar,aggBar,drawBar,drawAxis},
  start(){
    $('#themeBtn').addEventListener('click',()=>{const root=document.documentElement;
      const cur=root.getAttribute('data-theme')||(window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
      root.setAttribute('data-theme',cur==='dark'?'light':'dark');});
    $$('.langsw button').forEach(b=>b.addEventListener('click',()=>setLang(b.dataset.lang)));
    $('#hmCtx').addEventListener('input',heroUpdate);
    applyStatic();
    mapInit(); controlsInit(); labUI(); newDrill(); kitObserve(); flowWatch();
    window.addEventListener('hashchange',route);
    routeReady=true; document.addEventListener('click',routeClick,true);   // from here on the map's choice is the learner's path
    let rt=null; window.addEventListener('resize',()=>{clearTimeout(rt); rt=setTimeout(()=>{
      if(view==='lab') render(); else if(view==='dyn'){if(dyn&&dyn.onResize) dyn.onResize();} else {heroRefresh(); refitTree(); fitStrip();} dockSoon();},150);});
    route(); landMap();
  }
};
/* a load or reload at #map, or a shared link to it: the browser's own jump to the #map anchor comes after the router placed the tree
   (re-review O1), so place it again once the page has loaded, unless the reader has scrolled, clicked or typed meanwhile */
function landMap(){if(location.hash!=='#map') return; let user=false; const stop=()=>{user=true;};
  ['wheel','touchstart','keydown','pointerdown'].forEach(k=>window.addEventListener(k,stop,{once:true,passive:true}));
  const go=()=>{if(!user&&view==='home'&&location.hash==='#map') revealMap(false);};
  const after=()=>{requestAnimationFrame(go); setTimeout(go,150); setTimeout(go,600);};
  if(document.readyState==='complete') after(); else window.addEventListener('load',after,{once:true});}
})();
