/* Lesson: how a model continues a sentence (the owner, 2026-10-11 08:23 UTC: “现在开始改首页和Transformer课”).
   Built to the UX design of 2026-10-11 (docs/UX-DESIGN.md; Codex's version in reviews/CODEX-ux-design-20261011.md), written after a
   senior AI engineer could not read the architecture lab (reviews/OWNER-realuser-20261011.md):
   - one page read top to bottom in a fixed order: ten short sections, one new idea and one figure each. A term is named only after the
     thing it names has been shown, in bold where it is named, and never earlier in a heading, a button, a label or a caption;
   - few interactions, each saying what to press and what to look at: a step-through (section 1), one choice that redraws one figure
     (5), a guess before the answer (8), random draws (9), the end check; optional reading folded at the page end;
   - what is real and what is drawn by hand is said next to each figure; the model's name and the sources are at the page end;
   - the page keeps the reader's place (section and offset, every answer, draw and open fold: snap/restore) and puts the reader back
     there with a note after a reload, Back, or a return from another page.
   It takes over the id 'architecture' (catalog entry, routes, tree point and tile stay); labs/architecture.js keeps the home hero and
   the tile and no longer registers a lab page. */
(()=>{
'use strict';
const A=window.ASP; if(!A) return;
const {tx,L,esc,$,$$}=A, ID='architecture';

/* ---------- data ----------
   Real: the five tokens, their numbers and the next token (Meta's Llama 3 tokenizer, specs/architecture.md) and the sizes (Llama 3.1
   8B's published configuration, specs/transformer-facts.md). Drawn by hand: every value inside a vector, every attention share, every
   probability (the captions say so). */
const TOKS=[{t:'The',id:791},{t:' capital',id:6864},{t:' of',id:315},{t:' France',id:9822},{t:' is',id:374}];
const DM=4096, NLAY=32, NHEAD=32, FFN=14336, VOCAB=128256;
const N=n=>n.toLocaleString('en-US');
// section 5: each token's shares over itself and the tokens before it (rows add up to 1)
const AW=[[1],[.45,.55],[.30,.45,.25],[.25,.20,.15,.40],[.14,.26,.06,.44,.10]];
// section 6: three groups for “ is”
const HEADS=[{w:[.05,.10,.05,.75,.05],d:{en:'Mostly looks at ␣France',zh:'主要看 ␣France'}},
  {w:[.05,.05,.70,.10,.10],d:{en:'Mostly looks at the token just before, ␣of',zh:'主要看前一个 token ␣of'}},
  {w:[.20,.20,.20,.20,.20],d:{en:'Looks at every token a little',zh:'每个 token 都看一点'}}];
// section 9: the candidates and the rest (adds up to 1)
const CANDS=[{t:' Paris',p:.62},{t:' a',p:.08},{t:' the',p:.07},{t:' located',p:.04},{t:' Lyon',p:.03}], REST=.16;
// made-up vector values, the same on every visit
const rnd=s=>{const x=Math.sin(s*12.9898+78.233)*43758.5453; return x-Math.floor(x);};
const vec=(k,n)=>Array.from({length:n||8},(_,i)=>Math.round((rnd(k*31+i*7+3)*1.8-.9)*100)/100);
const cellBg=v=>`background:color-mix(in srgb,${v>=0?'var(--c-weights)':'var(--c-kv)'} ${Math.round(18+Math.abs(v)*70)}%,var(--surface))`;

/* ---------- small pieces ---------- */
// a token as it is written in the text: the leading space shown as ␣
const tk=t=>`<span class="ls-tk">${t[0]===' '?`<i aria-hidden="true">␣</i><span class="sr-only">${tx('space','空格')} </span>`:''}${esc(t.trim())}</span>`;
const tkp=t=>(t[0]===' '?'␣':'')+t.trim();   // plain text form
const pct=p=>Math.round(p*100)+'%';
const b=s=>`<b class="ls-term">${s}</b>`;
// a token mentioned in running text: “␣France” → the chip
const T=s=>s.replace(/␣([A-Za-z]+)/g,(m,w)=>tk(' '+w));

/* ---------- state ---------- */
const BLOCKS=['head','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','sum','quiz','next','deep','src'];
const DEEP=['pos','norm','mask','sizes'];
const S0=()=>({at:'head',off:0,s1:0,pick:4,p8:null,draws:[],quiz:[null,null,null],open:[],done:false});
let S=S0(), root=null, restored=false, holdScroll=0, noteT=0;
const QUIZ=()=>[
  {q:tx('In one round, starting from “The capital of France is”, how many tokens does the model write?','从 “The capital of France is” 往后写，模型一轮写出几个 token？'),
   o:[tx('One','一个'),tx('A whole sentence','一整句'),tx('It depends on how long the sentence is','看句子长短，不一定')],a:0,
   why:tx('One round writes one token. To write a sentence, the model repeats the round many times (sections 1 and 10).','一轮只写一个 token。要写一整句，就把这一轮重复很多次（第 1 段、第 10 段）。')},
  {q:T(tx('In attention, which tokens can ␣France look at?','做注意力的时候，␣France 能参考哪些 token？')),
   o:[tx('Only itself','只有它自己'),T(tx('Itself and the tokens before it: The, ␣capital, ␣of','它自己，和它前面的 The、␣capital、␣of')),tx('All 5 tokens of the sentence','句子里全部 5 个 token')],a:1,
   why:T(tx('Attention only looks back, itself included; the model was trained that way, because looking ahead would mean seeing the answer. ␣is comes after ␣France, so ␣France cannot look at it, even though it is already in the sentence (section 5).','注意力只往前看，也看自己，这是训练时就定下的规矩：往后看就等于偷看答案。␣is 在 ␣France 后面，所以 ␣France 参考不了它，哪怕它已经在句子里（第 5 段）。'))},
  {q:tx('Asked the same question twice, a chatbot may answer differently. What is the main reason?','同一个问题问两遍，聊天机器人的回答可能不一样。主要原因是什么？'),
   o:[tx('The model’s structure is different each time','模型每次的结构不一样'),tx('The last step draws a token at random by the probabilities','最后一步按概率随机抽一个 token'),tx('The sentence is cut into different tokens each time','每次切出来的 token 不一样')],a:1,
   why:tx('The structure and the cutting are the same every time. What changes is the last step: a token is drawn by its probability, so a less likely token can come up too (section 9).','结构和切分每次都一样。不一样的是最后一步：按概率抽，概率低的 token 也可能被抽到（第 9 段）。')}];
const quizPassed=()=>S.quiz.every((v,i)=>v===[0,1,1][i]);

/* ---------- the sections ----------
   The titles are also the contents list at the top of the page, so they use only plain words (“piece”, not “token”) */
const SEC=[
 {h:{en:'The sentence is not finished',zh:'一句话还没写完'},body:()=>
   `<p>${tx('We give a model an unfinished English sentence: <span class="ls-en">The capital of France is</span> …','先给模型一句没写完的英文：<span class="ls-en">The capital of France is</span>（法国的首都是……）。')}</p>`+
   `<p>${tx('The model does one thing at a time: from the text so far, it guesses the next word. Here it adds <span class="ls-en">Paris</span>.','模型每次只做一件事：根据前面的文字，猜下一个词。这里它会接上 <span class="ls-en">Paris</span>（巴黎）。')}</p>`+
   `<p>${tx('A chatbot writes a whole answer by doing this over and over: write one word, add it to the end, guess the next one.','聊天机器人能写出一整段回答，靠的就是把这件事重复很多次：写出一个词，接到句子后面，再猜下一个。')}</p>`+
   `<figure class="ls-fig" id="ls-f1"></figure>`+
   `<p>${tx('This lesson follows only the first round: how the model gets from this sentence to <span class="ls-en">Paris</span>.','这一课只跟着第一轮走，看模型怎样从这句话算出 <span class="ls-en">Paris</span>。')}</p>`},
 {h:{en:'First, the sentence is cut into pieces',zh:'先把句子切成小片段'},body:()=>
   `<p>${tx('The model does not read letter by letter, nor does it take the sentence as one single whole. It first cuts the sentence into small pieces, following a fixed set of rules:','模型并不是一个字母一个字母地读，也不是把整句当成一个整体来读。它先按一套固定的规则，把句子切成一个个小片段：')}</p>`+
   `<figure class="ls-fig"><div class="ls-toks" aria-label="${esc(tx('The sentence in 5 pieces','这句话切成 5 段'))}">${TOKS.map(x=>tk(x.t)).join('')}</div>`+
   `<figcaption>${tx('This is how the lesson’s example model really cuts the sentence. ␣ marks a space.','这是这一课用作例子的那个模型真实的切法。␣ 表示空格。')}</figcaption></figure>`+
   `<p>${tx('In this sentence each word happens to be one piece. The space before a word belongs to that word’s piece (␣ in the picture). A long or uncommon word is cut into several pieces.','这句英文正好一个词一段。词前面的空格也算在这一段里（图里用 ␣ 表示）。长的或不常见的词，会被切成好几段。')}</p>`+
   `<p>${tx(`Each small piece is called a ${b('token')}. So, to be exact, the model guesses the next token each time. This sentence is 5 tokens.`,`这样切出来的一个小片段，叫一个 ${b('token')}。所以更准确地说，模型每次猜的是下一个 token。这句话一共是 5 个 token。`)}</p>`},
 {h:{en:'Each piece becomes a list of numbers',zh:'每个片段换成一列数字'},body:()=>
   `<p>${tx('A model can only calculate with numbers, so every token has to become numbers. This takes two steps.','模型只会做数字计算，所以每个 token 都要换成数字。这分两步。')}</p>`+
   `<p>${T(tx(`First, look up its number. The model knows ${N(VOCAB)} tokens, and each has a fixed number: ␣France, for example, is number 9822. The number is only used to look things up; its size means nothing.`,
     `第一步是查编号。模型认识的 token 一共有 ${N(VOCAB)} 个，每个都有一个固定的编号，比如 ␣France 是 9822 号。编号只是用来查表的，大小没有意义。`))}</p>`+
   `<p>${tx(`Second, use that number to take out the list of numbers that belongs to the token, from another table. The list holds ${N(DM)} numbers, which the model learned beforehand.`,
     `第二步是用这个编号，去另一张表里取出属于这个 token 的一列数字。这一列有 ${N(DM)} 个数，是模型事先学好的。`)}</p>`+
   `<figure class="ls-fig"><div class="ls-emb">${TOKS.map((x,i)=>`<div class="ls-emb-c">${tk(x.t)}<span class="ls-ar" aria-hidden="true">↓</span><span class="ls-id">${tx('No. ','')}${x.id}${tx('',' 号')}</span><span class="ls-ar" aria-hidden="true">↓</span>`+
     `<span class="ls-col">${vec(i+1).map(v=>`<span style="${cellBg(v)}">${v.toFixed(2)}</span>`).join('')}<span class="ls-more">⋮</span></span></div>`).join('')}</div>`+
   `<p class="ls-fignote">${tx(`Each list: ${N(DM)} numbers`,`每列 ${N(DM)} 个数`)}</p>`+
   `<figcaption>${tx(`The token numbers are real. Each list shows only 8 of its ${N(DM)} numbers, and these 8 are made up for the explanation.`,`编号是真实的。每列只画了 8 个数（真实有 ${N(DM)} 个），这 8 个数是为讲解编的。`)}</figcaption></figure>`+
   `<p>${tx(`Such a list of numbers is called the token’s ${b('vector')}. The step that takes out a token’s vector by its number is called ${b('embedding')}.`,`这样一列数字，叫这个 token 的${b('向量')}。按编号取出向量的这一步，叫 ${b('embedding')}（嵌入）。`)}</p>`},
 {h:{en:'The order has to be kept too',zh:'顺序也要记下来'},body:()=>
   `<p>${tx('Every token is now a vector. But the vector only says which token it is, not where it stands. The same token gets the same vector wherever it appears in the sentence.','现在每个 token 都换成了一个向量。可是向量只说明“这是哪个 token”，不说明“它排第几”。同一个 token 不管在句子的哪个位置，取出来的都是同一个向量。')}</p>`+
   `<p>${tx('Yet order changes meaning. These two sentences use exactly the same tokens, in a different order:','顺序却会改变意思。下面两句用的 token 完全一样，只是顺序不同：')}</p>`+
   `<figure class="ls-fig" id="ls-f4"></figure>`+
   `<p>${tx('So the model also needs each token’s position: 1st, 2nd and so on. The model in this lesson does not write the position into the vector. It uses the position in the step of section 5, looking back at earlier pieces; how it does that is explained at the end of the page.','所以模型还需要知道每个 token 的位置：第 1 个、第 2 个……这一课的模型并不把位置写进向量，而是在第 5 段要讲的“参考前面的片段”这一步里用上它。具体怎么用，放在页末。')}</p>`},
 {h:{en:'Looking back at earlier pieces',zh:'参考前面的片段'},body:()=>
   `<p>${T(tx('From ␣is alone you cannot guess the next word. To get to Paris, the model has to use ␣France and ␣capital, which come earlier.','只看 ␣is 这一个 token，猜不出下一个词。要猜到 Paris，得用上前面的 ␣France 和 ␣capital。'))}</p>`+
   `<p>${tx('So every token looks back: at each token before it, and at itself. It gives each one a share, how much to take from it, and the shares add up to 100%. Then it mixes their vectors in those shares and uses the mix to update its own vector.','所以每个 token 都会回头看：看它前面的每个 token，也看它自己，给每个定一个“参考多少”的比例，这些比例加起来是 100%。然后按这些比例把它们的向量混合起来，用来更新自己的向量。')}</p>`+
   `<p>${tx('How big a share is depends on how well the two tokens’ vectors match, and also on how far apart the two tokens are. This is where the position from the last section is used.','比例的大小，取决于两个 token 的向量有多相配，也和这两个 token 相隔多远有关。上一段说的位置，就是在这里用上的。')}</p>`+
   `<p>${T(tx('Each token only looks back, never ahead, even when later tokens are already in the sentence: ␣France cannot look at ␣is. The model was trained this way. In training the whole sentence is there and the model practises guessing what comes next; looking ahead would mean seeing the answer first. (More under “A bit deeper” at the end of the page.)','每个 token 只往前看，不往后看，哪怕后面的 token 已经在句子里：␣France 看不到 ␣is。这是模型训练时就定下的规矩：训练时整句话都在，模型练的是根据前面猜后面，如果能往后看，就等于先看到了答案（页末“深入一点”有细节）。'))}</p>`+
   `<figure class="ls-fig" id="ls-f5"></figure>`+
   `<p>${tx(`This step is called ${b('attention')}.`,`这一步叫${b('注意力')}（attention）。`)}</p>`},
 {h:{en:'Several ways of looking back at once',zh:'同时从几个角度参考'},body:()=>
   `<p>${T(tx('A token often needs several kinds of information at once. For ␣is, it helps to know which country the sentence is about, and also what the token just before it is.','一个 token 往往同时需要好几种信息。对 ␣is 来说，“句子在说哪个国家”有用，“前一个 token 是什么”也有用。'))}</p>`+
   `<p>${tx('So attention does not work out just one set of shares. It splits into several groups that each work out their own shares at the same time, and the groups’ mixes are then put together.','所以注意力不只算一套比例，而是分成好几组，同时各算各的，最后把各组混合出的结果合在一起。')}</p>`+
   `<figure class="ls-fig" id="ls-f6"></figure>`+
   `<p>${tx(`One such group is called an ${b('attention head')}. The model in this lesson has ${NHEAD} heads each time it does attention.`,`这样的一组叫一个${b('注意力头')}（head）。这一课的模型每次做注意力，都有 ${NHEAD} 个头。`)}</p>`},
 {h:{en:'Each piece is then processed on its own',zh:'每个片段再单独加工一次'},body:()=>
   `<p>${tx('Attention let every token take information from the tokens before it. Next, each token’s vector is processed once more, on its own. This step does not look at other tokens: the 5 tokens are each processed separately, by the same calculation.','注意力让每个 token 从前面的 token 那里拿到了信息。接下来，每个 token 的向量还要各自加工一次。这一步不看别的 token：5 个 token 各算各的，用的是同一套计算。')}</p>`+
   `<p>${tx(`It works in three moves: the ${N(DM)} numbers are spread out into ${N(FFN)}; then they are filtered, some kept and some pushed close to 0; finally they are squeezed back into ${N(DM)}.`,`加工分三步：先把 ${N(DM)} 个数展开成 ${N(FFN)} 个；再筛一遍，有的数留下，有的被压到接近 0；最后压回 ${N(DM)} 个。`)}</p>`+
   `<figure class="ls-fig" id="ls-f7"></figure>`+
   `<p>${tx(`This step is called the ${b('feed-forward network')}; many texts call it the ${b('MLP')}; here that is just another name for the same step. Research has found that many facts a model remembers, such as “the capital of France is Paris”, have a lot to do with this step (sources at the end of the page).`,
     `这一步叫${b('前馈网络')}（feed-forward network），很多资料也叫它 ${b('MLP')}，在这里只是同一步的另一个名字。有研究发现，模型记住的很多事实，比如“法国的首都是巴黎”，和这一步关系很大（出处在页末）。`)}</p>`},
 {h:{en:'These two steps are repeated many times',zh:'把这两步重复很多遍'},body:()=>
   `<p>${tx(`Attention, then the feed-forward network: together these two steps are called a ${b('layer')}.`,`注意力，加上前馈网络，这两步合起来叫一${b('层')}（layer）。`)}</p>`+
   `<div class="ls-q" id="ls-p8"></div>`+
   `<p>${tx(`The model works through layer after layer like this. The model in this lesson has ${NLAY} layers. Every layer has the same structure, but the numbers it learned are its own, so each layer processes differently.`,`模型就这样一层接一层地算下去。这一课的模型有 ${NLAY} 层。每层的结构完全一样，但里面学到的数各不相同，所以每层的加工也不一样。`)}</p>`+
   `<figure class="ls-fig" id="ls-f8"></figure>`+
   `<p>${tx('Each layer also has two small steps that keep the calculation steady. They are not drawn here; they are under “A bit deeper” at the end of the page.','每层里其实还有两个让计算更稳的小步骤，这里没有画，放在页末的“深入一点”里。')}</p>`},
 {h:{en:'Score every possible next piece, then pick one',zh:'给下一个片段打分，再选一个'},body:()=>
   `<p>${T(tx(`After the ${NLAY} layers, the word to guess is the one after ␣is, so only the vector of the last token, ␣is, is used.`,`${NLAY} 层算完，要猜的是 ␣is 后面的词，所以只用最后一个 token ␣is 的向量。`))}</p>`+
   `<p>${tx(`The model compares this vector with every one of the ${N(VOCAB)} tokens it knows and gives each a score. The higher the score, the better that token fits next.`,`模型拿这个向量，和它认识的全部 ${N(VOCAB)} 个 token 逐个比较，给每个打一个分。分越高，越适合接在后面。`)}</p>`+
   `<p>${tx(`It then turns the scores into shares that add up to 100%, called ${b('probabilities')}. A higher score gets a higher probability.`,`然后把这些分换算成加起来等于 100% 的比例，叫${b('概率')}。分高的，概率也高。`)}</p>`+
   `<p>${tx('Finally one token is picked. The model could always take the most likely one; chatbots usually draw one at random by these probabilities, which is why the same question can get a different answer each time.','最后从中选一个。可以每次都选概率最高的；聊天机器人通常按概率随机抽一个，所以同一个问题，每次的回答可能不一样。')}</p>`+
   `<figure class="ls-fig" id="ls-f9"></figure>`},
 {h:{en:'Putting it all together',zh:'把整个过程连起来'},body:()=>
   `<p>${T(tx('The chosen ␣Paris is added to the sentence, which becomes <span class="ls-en">The capital of France is Paris</span>. Then the whole process runs again from the start to guess the next token. Token after token, until the model writes a special token that means “finished”, or reaches a length limit.',
     '选出的 ␣Paris 接到句子后面，句子变成 <span class="ls-en">The capital of France is Paris</span>。然后整个过程从头再走一遍，猜再下一个 token。就这样一个接一个，直到模型写出一个表示“写完了”的特殊 token，或者写到长度上限。'))}</p>`+
   `<figure class="ls-fig" id="ls-f10"></figure>`+
   `<p>${tx(`The whole structure you have just walked through is called a ${b('Transformer')}. Most of today’s chat models are built on it.`,`你刚才从头走到尾的这一整套结构，叫 ${b('Transformer')}。今天的大多数聊天模型，都是在它的基础上搭起来的。`)}</p>`}
];

/* ---------- figures that change ---------- */
function f1(){const r=S.s1, sent=['The capital of France is'].concat(r>=1?['Paris']:[]).concat(r>=2?['.']:[]);
  const words=sent.map((w,i)=>`<span class="${i===sent.length-1&&i>0?'ls-new':''}">${i>0&&w!=='.'?' ':''}${esc(w)}</span>`).join('');
  const see=[tx('The box is the next word the model has to guess.','方框是模型要猜的下一个词。'),
    tx('Round 1: Paris is added. The sentence is longer, and the model now guesses the word after it.','第 1 轮：接上了 Paris。句子变长了，模型接着猜再下一个词。'),
    tx('Round 2: a full stop is added (written in for the demonstration). An answer is written like this, one round at a time. The demonstration ends here; press “Start over” to see it again.','第 2 轮：接上了句号（这个句号是为了演示写上的）。回答就是这样一轮一轮写出来的。演示到这里结束，可以点“回到开头”再看一遍。')][r];
  return `<p class="ls-do">${tx('Press “Add a word” and watch the sentence grow one round at a time.','点“接上一个词”，看句子怎样一轮一轮变长。')}</p>`+
    `<div class="ls-sent"><span class="ls-en">${words}</span> <span class="ls-blank" aria-label="${esc(tx('next word','下一个词'))}">?</span></div>`+
    `<div class="ls-ctl"><button type="button" class="btn sm" data-act="s1-next"${r>=2?' disabled':''}>${tx('Add a word','接上一个词')}</button>`+
    (r>0?`<button type="button" class="btn ghost sm" data-act="s1-reset">${tx('Start over','回到开头')}</button>`:'')+`</div>`+
    `<p class="ls-see" aria-live="polite">${see}</p>`+
    `<figcaption>${tx('The lesson uses one real, openly released model as its example (named at the end of the page): its rules for cutting text and its sizes are real, but this page does not run it. Paris is the usual way to finish this sentence, so we use it as the example.','这一课拿一个公开发布的真实模型做例子（页末写了是哪个）：切分规则和尺寸都是它的，但这一页不运行它。Paris 是这句话最常见的接法，我们拿它当例子。')}</figcaption>`;}
function f4(){const A1=['The',' dog',' bites',' the',' man'], A2=['The',' man',' bites',' the',' dog'], key={'The':11,' dog':12,' bites':13,' the':14,' man':15};
  const row=(arr,gl)=>`<div class="ls-ord"><div class="ls-ord-s"><span class="ls-en">${esc(arr.join(''))}</span>${gl?`<span class="ls-gl">${gl}</span>`:''}</div><div class="ls-ord-t">`+
    arr.map((x,i)=>`<div class="ls-ord-c${x===' dog'||x===' man'?' hl':''}"><span class="ls-pos">${tx(`#${i+1}`,`第 ${i+1} 个`)}</span>${tk(x)}<span class="ls-strip" aria-hidden="true">${vec(key[x],6).map(v=>`<i style="${cellBg(v)}"></i>`).join('')}</span></div>`).join('')+`</div></div>`;
  return row(A1,tx('','狗咬了人'))+row(A2,tx('','人咬了狗'))+
    `<figcaption>${T(tx('Each coloured cell stands for one number (made up). ␣dog gets the same vector in both sentences, and so does ␣man. From the vectors alone you cannot tell who bit whom.','每个色块代表一个数（示意）。两句里的 ␣dog 取出的是同一个向量，␣man 也是。只看向量，分不出谁咬了谁。'))}</figcaption>`;}
// the bars under the tokens: shares of token i (its row of AW), later tokens greyed
function barsHTML(w,i,btn){return `<div class="ls-att-row" style="--n:${TOKS.length}">`+TOKS.map((x,j)=>{const on=j<=i, p=on?w[j]:0;
    const chip=btn?`<button type="button" class="ls-pick${j===i?' on':''}" data-pick="${j}" aria-pressed="${j===i}">${tk(x.t)}</button>`:`<span class="ls-pk${j===i?' on':''}">${tk(x.t)}</span>`;
    return `<div class="ls-att-c${on?'':' off'}">${chip}<span class="ls-bar"><i style="height:${Math.round(p*100)}%"></i></span><span class="ls-pc">${on?pct(p):tx('can’t see','看不到')}</span></div>`;}).join('')+`</div>`;}
function arcsSVG(w,i){const n=TOKS.length, cx=j=>(j+.5)*100;
  return `<svg class="ls-arcs" viewBox="0 0 ${n*100} 56" preserveAspectRatio="none" aria-hidden="true">`+
    w.map((p,j)=>j===i?'':`<path d="M${cx(i)} 56 C ${cx(i)} ${56-46*Math.min(1,(i-j)/2.2)}, ${cx(j)} ${56-46*Math.min(1,(i-j)/2.2)}, ${cx(j)} 56" style="stroke-width:${(1+p*12).toFixed(1)}px;opacity:${(.35+p*.65).toFixed(2)}"/>`).join('')+`</svg>`;}
function f5(){const i=S.pick, w=AW[i], x=TOKS[i];
  let see;
  if(i===0) see=tx('The is the first token. Nothing comes before it, so it looks only at itself (100%).','The 是第一个 token，前面没有别的 token，所以它只参考自己（100%）。');
  else {const m=w.indexOf(Math.max(...w)), later=TOKS.length-1-i;
    see=tx(`${tk(x.t)} looks at itself and the ${i} token${i>1?'s':''} before it; the largest share goes to ${tk(TOKS[m].t)} (${pct(w[m])}).`,`${tk(x.t)} 参考它自己和前面 ${i} 个 token，比例最高的是 ${tk(TOKS[m].t)}（${pct(w[m])}）。`)+
      (later?tx(` It cannot see the ${later} token${later>1?'s':''} after it.`,`后面的 ${later} 个 token 它看不到。`):'');}
  return `<p class="ls-do">${tx('Press a token to see which tokens it looks at, and how much of each it takes.','点一个 token，看它参考哪些 token、各参考多少。')}</p>`+
    `<div class="ls-att">${arcsSVG(w,i)}${barsHTML(w,i,true)}</div>`+
    `<p class="ls-see" aria-live="polite">${see}</p>`+
    `<figcaption>${tx('The shares are made up for the explanation. In a real model this step is done many times, with different shares each time. A large share only means more information is mixed in from that token; it is not the model’s “reason”. To keep it simple we say the vectors are mixed; what is really mixed is another list of numbers worked out from each vector.','比例是为讲解编的。真实模型里这一步要做很多次，每次的比例都不一样。比例高只说明从这个 token 混进来的信息多，不等于模型的“理由”。为了好懂，这里说“混合向量”；实际混合的，是从每个向量换算出来的另一列数。')}</figcaption>`;}
function f6(){return `<div class="ls-heads">`+HEADS.map((h,k)=>`<div class="ls-grp"><div class="ls-grp-t">${tx(`Group ${k+1}`,`第 ${k+1} 组`)}</div><div class="ls-hrows">`+
    TOKS.map((x,j)=>`<span class="ls-hk">${tk(x.t)}</span><span class="ls-hb"><i style="width:${Math.round(h.w[j]*100)}%"></i></span><span class="ls-hv">${pct(h.w[j])}</span>`).join('')+
    `</div><p class="ls-grp-d">${T(L(h.d))}</p></div>`).join('')+`</div>`+
    `<div class="ls-merge"><span aria-hidden="true">↓</span> ${T(tx('The 3 groups’ results are put together and update the vector of ␣is','3 组的结果合在一起，更新 ␣is 的向量'))}</div>`+
    `<figcaption>${T(tx('Made up: only 3 groups are drawn, all for ␣is, with invented shares. In a real model it is often hard to say what one group looks for; the picture only shows that each group works on its own.','示意：只画了 3 组，都是 ␣is 的，比例是编的。真实模型里，一组到底在看什么往往说不清；这里只为说明各组各算各的。'))}</figcaption>`;}
function f7(){const row=(n,dim,lbl)=>`<div class="ls-ff-r"><span class="ls-ff-cells">${Array.from({length:n},(_,i)=>`<i class="${dim&&rnd(i*5+2)<.45?'dim':''}"></i>`).join('')}</span><span class="ls-ff-l">${lbl}</span></div>`;
  return `<div class="ls-ff">${row(8,false,tx(`${N(DM)} numbers in`,`进来：${N(DM)} 个数`))}<div class="ls-ff-a">↓ ${tx('① spread out','① 展开')}</div>`+
    row(28,false,tx(`${N(FFN)} numbers`,`${N(FFN)} 个数`))+`<div class="ls-ff-a">↓ ${tx('② filter: some kept, some pushed close to 0','② 筛一遍：有的留下，有的压到接近 0')}</div>`+
    row(28,true,tx('after filtering','筛过以后'))+`<div class="ls-ff-a">↓ ${tx('③ squeeze back','③ 压回')}</div>`+row(8,false,tx(`${N(DM)} numbers out`,`出去：${N(DM)} 个数`))+`</div>`+
    `<p class="ls-fignote">${tx('Each of the 5 tokens goes through this on its own.','5 个 token 各自走一遍，互不相看。')}</p>`+
    `<figcaption>${tx(`Only a sketch: really ${N(DM)} → ${N(FFN)} → ${N(DM)} numbers, drawn as 8 → 28 → 8 cells. Which cells fade out is drawn at random.`,`只是示意：真实是 ${N(DM)} → ${N(FFN)} → ${N(DM)} 个数，这里画成 8 → 28 → 8 格。哪些格子变淡是随手画的。`)}</figcaption>`;}
function p8(){const o=[tx(`Still ${N(DM)}`,`还是 ${N(DM)} 个`),tx(`${N(FFN)}`,`${N(FFN)} 个`),tx(`A few more than ${N(DM)}`,`比 ${N(DM)} 多一点`)], a=S.p8;
  return `<p class="ls-q-t">${tx('Guess first: after one layer, how many numbers are in each token’s vector? (Once you answer, the question mark in the picture below shows the answer.)','先猜一猜：一层做完，每个 token 的向量有多少个数？（答完以后，下面图里的问号会换成答案。）')}</p><div class="ls-opts" role="group">`+
    o.map((x,i)=>`<button type="button" class="ls-opt${a===i?(i===0?' ok':' no'):''}" data-p8="${i}" aria-pressed="${a===i}">${x}</button>`).join('')+`</div>`+
    (a===null?'':`<p class="ls-fb" aria-live="polite"><b>${a===0?tx('Right: still ','对，还是 '):tx('Not quite: it is still ','不对，还是 ')}${N(DM)}${tx('.',' 个。')}</b> ${tx(`The feed-forward network spreads out to ${N(FFN)} in the middle but squeezes back to ${N(DM)} at the end. What goes in and what comes out have the same length, so one layer’s result can go straight into the next layer.`,`前馈网络中间展开到 ${N(FFN)} 个，最后又压回了 ${N(DM)} 个。进出一样长，所以一层的结果可以直接送进下一层。`)}</p>`);}
function f8(){const k=S.p8===null?'?':N(DM);
  return `<div class="ls-stack"><div class="ls-st-io">${tx('In: 5 vectors from the embedding','进来：embedding 取出的 5 个向量')}</div><div class="ls-st-ar">↓</div>`+
    `<div class="ls-layer"><div class="ls-layer-t">${tx('Layer 1','第 1 层')}</div><div class="ls-layer-b"><span>${tx('Attention: each token looks back at itself and the tokens before it','注意力：每个 token 参考自己和前面的 token')}</span><span class="ls-st-ar">↓</span><span>${tx('Feed-forward network: each token is processed on its own','前馈网络：每个 token 单独加工')}</span></div></div>`+
    `<div class="ls-st-ar">↓ <span class="ls-st-k">${tx(`each vector: ${k} numbers`,`每个向量 ${k} 个数`)}</span></div>`+
    `<div class="ls-layer sm"><div class="ls-layer-t">${tx('Layer 2','第 2 层')}</div><span>${tx('the same two steps','同样的两步')}</span></div><div class="ls-st-ar">↓</div>`+
    `<div class="ls-layer sm ghost"><div class="ls-layer-t">${tx('Layers 3 to 31','第 3 到 31 层')}</div><span>⋮</span></div><div class="ls-st-ar">↓</div>`+
    `<div class="ls-layer sm"><div class="ls-layer-t">${tx(`Layer ${NLAY}`,`第 ${NLAY} 层`)}</div><span>${tx('the same two steps','同样的两步')}</span></div><div class="ls-st-ar">↓</div>`+
    `<div class="ls-st-io">${tx('Out: 5 updated vectors, one per token','出去：更新过的 5 个向量，每个 token 一个')}</div></div>`+
    `<figcaption>${tx(`The ${NLAY} layers are real. Layer 1 is drawn with its two steps; layers 2 to ${NLAY} have the same structure.`,`${NLAY} 层是真实的层数。第 1 层画出了里面的两步；第 2 到第 ${NLAY} 层结构相同。`)}</figcaption>`;}
function f9(){const n=S.draws.length, cnt=Array(CANDS.length+1).fill(0); S.draws.forEach(d=>cnt[d]++);
  const nm=d=>d<CANDS.length?tk(CANDS[d].t):tx('one of the other tokens','其余的某个 token');
  const rows=CANDS.map((c,i)=>`<div class="ls-pr"><span class="ls-pr-k">${tk(c.t)}</span><span class="ls-pr-b"><i style="width:${(c.p*100).toFixed(1)}%"></i></span><span class="ls-pr-v">${pct(c.p)}</span></div>`).join('')+
    `<div class="ls-pr rest"><span class="ls-pr-k">${tx(`The other ${N(VOCAB-CANDS.length)}`,`其余 ${N(VOCAB-CANDS.length)} 个`)}</span><span class="ls-pr-b"><i style="width:${(REST*100).toFixed(1)}%"></i></span><span class="ls-pr-v">${pct(REST)}</span></div>`;
  let see=tx('Not drawn yet.','还没抽。');
  if(n){const last=S.draws[n-1], parts=cnt.map((c,d)=>c?`${nm(d)} ${c}${tx(c>1?' times':' time',' 次')}`:'').filter(Boolean);
    see=tx(`This time: ${nm(last)}. In ${n} draw${n>1?'s':''} so far: `,`这次抽到 ${nm(last)}。一共抽了 ${n} 次：`)+parts.join(tx(', ','，'))+tx('.','。');}
  return `<div class="ls-probs">${rows}</div>`+
    `<p class="ls-do">${tx('Press “Draw once” a few times: is it the same token every time?','多按几次“按概率抽一次”：每次抽到的都一样吗？')}</p>`+
    `<div class="ls-ctl"><button type="button" class="btn sm" data-act="draw">${tx('Draw once','按概率抽一次')}</button>${n?`<button type="button" class="btn ghost sm" data-act="draw-reset">${tx('Clear','清空')}</button>`:''}</div>`+
    `<p class="ls-see" aria-live="polite">${see}</p>`+
    `<figcaption>${tx('These probabilities are set by hand for the explanation; no model was run. The button really does draw at random with them.','这些概率是为讲解设定的，没有运行模型。按钮是真的按它们随机抽。')}</figcaption>`;}
function f10(){const box=(n,t,d,s)=>`<div class="ls-rc${s?' sm':''}"><div class="ls-rc-h"><span class="ls-rc-n">${n}</span><span class="ls-rc-t">${t}</span><button type="button" class="ls-back" data-go="${s}">${tx(`Section ${s} ↑`,`回看第 ${s} 段`)}</button></div><p>${d}</p></div>`;
  return `<div class="ls-recap">`+
    box('1',tx('An unfinished sentence','一句没写完的话'),`<span class="ls-en">The capital of France is</span>`,1)+`<div class="ls-st-ar">↓</div>`+
    box('2',tx('Cut into tokens','切成 token'),tx('5 tokens','5 个 token'),2)+`<div class="ls-st-ar">↓</div>`+
    box('3',tx('Each token becomes a vector: embedding','每个 token 换成向量：embedding'),tx(`${N(DM)} numbers per token`,`每个 token ${N(DM)} 个数`),3)+`<div class="ls-st-ar">↓</div>`+
    `<div class="ls-rl"><div class="ls-rl-h"><span>${tx(`One layer, repeated ${NLAY} times`,`一层，重复 ${NLAY} 次`)}</span><button type="button" class="ls-back" data-go="8">${tx('Section 8 ↑','回看第 8 段')}</button></div>`+
      box('4',tx('Attention','注意力'),tx(`Each token looks back at itself and the tokens before it, using their positions; ${NHEAD} heads at once`,`每个 token 按比例参考自己和前面的 token，这里用上了位置；${NHEAD} 个头同时做`),5)+`<div class="ls-st-ar">↓</div>`+
      box('5',tx('Feed-forward network','前馈网络'),tx(`Each token on its own: ${N(DM)} → ${N(FFN)} → ${N(DM)}`,`每个 token 单独加工：${N(DM)} → ${N(FFN)} → ${N(DM)}`),7)+`</div><div class="ls-st-ar">↓</div>`+
    box('6',tx('Score, probabilities, pick one','打分、算概率、选一个'),T(tx(`From the vector of ␣is, a score for each of the ${N(VOCAB)} tokens`,`用 ␣is 的向量，给 ${N(VOCAB)} 个 token 各打一个分`)),9)+`<div class="ls-st-ar">↓</div>`+
    box('7',tx('Add it to the sentence','接回句子'),T(tx('␣Paris is added; then the next round starts from the top','接上 ␣Paris，然后从头开始下一轮')),1)+
    `<div class="ls-loop" aria-hidden="true">↺ ${tx('next round','下一轮')}</div></div>`+
    `<figcaption>${tx('Each “Section ↑” button takes you back up to the section that explains that step.','每个“回看”按钮会带你回到讲这一步的那一段。')}</figcaption>`;}

/* ---------- the page ---------- */
const H=()=>({sum:tx('What this lesson covered','这一课讲了什么'),quiz:tx('Check yourself','检查一下'),next:tx('Next','下一步'),deep:tx('A bit deeper (optional)','深入一点（可选）'),src:tx('Where the pictures and numbers come from','这些图和数字从哪里来')});
function headHTML(){const l=A.catalog.LAB[ID];
  const w=l?l.w:'models', W=A.catalog.WORLDS[w];
  // the site's breadcrumb row: the tree, the branch, this lesson, and the previous and next lab (the owner, 2026-10-09: both at the top of every lab)
  return `<div class="ls-head" data-ls="head"><div class="labhead ls-labhead"><div class="crumbs"><a href="#map" data-tipfn="crumbTree">${A.t('crumbMap')}</a><span>/</span>`+
    `<a href="#map~${w}" data-tipx="${esc(A.shell.crumbWorldTip(w))}">${esc(L(W.name))}</a><span>/</span><span>${esc(l?L(l.s):'')}</span>${A.shell.crumbNav(ID)}</div></div>`+
    `<h1>${tx('How a model continues a sentence','模型怎样接着写一句话')}</h1>`+
    `<p class="ls-lead">${tx('Follow one unfinished sentence and see, step by step, how a large language model writes the next word.','跟着一句没写完的话，一步一步看大模型怎样写出下一个词。')}</p>`+
    `<p class="ls-can"><b>${tx('After this lesson you can','学完你能')}</b>${tx(': explain in your own words how a chatbot writes its answer, one word at a time.','：用自己的话讲清楚，聊天机器人是怎样一个词一个词写出回答的。')}</p>`+
    `<p class="ls-how">${tx(`About ${l?l.min:10} minutes · 10 short sections. Just scroll down to read it all. Where there is a button, the text next to it says what to press and where to look.`,`约 ${l?l.min:10} 分钟 · 10 小段。一直往下滚动就能读完。有按钮的地方，旁边会写明按什么、看哪里。`)}</p>`+
    `<details class="ls-toc-n"><summary>${tx('Contents','本课目录')}</summary>${tocList()}</details></div>`;}
function tocList(){const h=H();
  return `<ol class="ls-toc-l">${SEC.map((s,i)=>`<li><button type="button" data-go="${i+1}"><span class="ls-toc-k">${i+1}</span>${esc(L(s.h))}</button></li>`).join('')}`+
    ['sum','quiz','deep','src'].map(k=>`<li class="x"><button type="button" data-go="${k}"><span class="ls-toc-k">·</span>${esc(h[k])}</button></li>`).join('')+`</ol>`;}
function secHTML(s,i){return `<section class="ls-sec" id="ls-s${i+1}" data-ls="s${i+1}" aria-labelledby="ls-h${i+1}"><p class="ls-k">${tx(`Section ${i+1} of 10`,`第 ${i+1} 段，共 10 段`)}</p><h2 id="ls-h${i+1}">${esc(L(s.h))}</h2>${s.body()}</section>`;}
function sumHTML(){const it=[tx('A model writes one token per round, adds it to the end, and starts the next round.','模型每轮只写一个 token，写完接到句子后面，再开始下一轮。'),
    tx(`The text is first cut into tokens; each token becomes a vector of ${N(DM)} numbers (embedding).`,`文字先切成 token，每个 token 换成一个 ${N(DM)} 个数的向量（embedding）。`),
    tx(`Attention: each token takes information from itself and the tokens before it, in shares, with ${NHEAD} heads at once.`,`注意力：每个 token 按比例参考它自己和前面的 token，${NHEAD} 个头同时做。`),
    tx(`Feed-forward network: each token’s vector is processed on its own. Attention plus the feed-forward network is one layer; there are ${NLAY}.`,`前馈网络：每个 token 的向量单独加工。注意力加前馈网络是一层，一共 ${NLAY} 层。`),
    tx(`Last, every one of the ${N(VOCAB)} tokens gets a score, the scores become probabilities, and one token is picked.`,`最后给 ${N(VOCAB)} 个 token 各打一个分，换算成概率，再选出一个。`)];
  return `<section class="ls-sec ls-sum" data-ls="sum" id="ls-sum"><h2>${H().sum}</h2><ul>${it.map(x=>`<li>${x}</li>`).join('')}</ul></section>`;}
function quizHTML(){return `<section class="ls-sec ls-quiz" data-ls="quiz" id="ls-quiz"><h2>${H().quiz}</h2><p>${tx('3 questions. You see right away whether your answer is right, and you can change it.','3 道题。选完马上告诉你对不对，也可以改答案。')}</p><div id="ls-qz"></div></section>`;}
function qzHTML(){const Q=QUIZ(), n=S.quiz.filter(v=>v!==null).length, ok=S.quiz.filter((v,i)=>v===Q[i].a).length;
  return Q.map((q,k)=>{const a=S.quiz[k];
    return `<div class="ls-q"><p class="ls-q-t"><span class="ls-q-n">${k+1}</span>${q.q}</p><div class="ls-opts" role="group">`+
      q.o.map((x,i)=>`<button type="button" class="ls-opt${a===i?(i===q.a?' ok':' no'):''}" data-qz="${k}" data-o="${i}" aria-pressed="${a===i}">${x}</button>`).join('')+`</div>`+
      (a===null?'':`<p class="ls-fb" aria-live="polite"><b>${a===q.a?tx('Right.','对。'):tx('Not quite.','不对。')}</b> ${q.why}</p>`)+`</div>`;}).join('')+
    (n===3?`<p class="ls-score${ok===3?' ok':''}">${ok===3?tx('All 3 right.','3 题全对。'):tx(`${ok} of 3 right. The section number after each answer says where to look again; you can change your answers.`,`答对 ${ok} / 3 题。每题解释后面写了该回看哪一段，看完可以改答案。`)}</p>`:'');}
function nextHTML(){const x=A.catalog.nextInfo(ID), l=x&&x.next?A.catalog.LAB[x.next]:null; if(!l) return '';
  const path=x.kind==='route'&&x.to?` data-path="${esc(x.to)}"`:'';
  return `<section class="ls-sec ls-next" data-ls="next"><h2>${H().next}</h2><a class="btn" href="#${l.id}"${path}>${tx('Next lesson: ','下一课：')}${esc(L(l.q))} <span aria-hidden="true">→</span></a>`+
    (l.id==='transformer'?`<p class="ls-alt">${tx('It looks at where this lesson’s model keeps its roughly 8 billion learned numbers (called parameters; “8B” means 8 billion).','下一课看这一课的模型里，那约 80 亿个学到的数（叫参数，“8B”就是 80 亿）都放在哪一步。')}</p>`:'')+`<p class="ls-alt">${tx('Or go back to the <a href="#top">home page</a> and pick another path.','或者回到<a href="#top">首页</a>，选别的路线。')}</p></section>`;}
function deepHTML(){const D=[
  {k:'pos',t:tx('How the position is used','位置是怎样用上的'),b:tx(`<p>The model in this lesson uses a method called rotary position embedding (RoPE). Just before attention works out its shares, the numbers each token uses for matching are turned by an angle that depends on the token’s position. The angle between two tokens then depends on how far apart they are, so the shares change with distance. The position is used only inside attention; it is never added to the token’s vector.</p><p>The original Transformer of 2017 did it differently: it added a list of numbers that stands for the position to each token’s vector at the start. Lab: ${A.shell.labLink('rope')}.</p>`,
     `<p>这一课的模型用的方法叫旋转位置编码（RoPE）。注意力在算比例之前，会把每个 token 用来“相配”的那些数，按它的位置转一个角度。两个 token 之间的角度差，取决于它们隔多远，所以算出来的比例会随距离变化。位置只在注意力里起作用，从来不加进 token 的向量。</p><p>2017 年最早的 Transformer 做法不同：一开始就把一列代表位置的数，直接加到每个 token 的向量上。相关 Lab：${A.shell.labLink('rope')}。</p>`)},
  {k:'norm',t:tx('The two small steps in each layer','每层里的两个小步骤'),b:tx(`<p><b>Adding back.</b> What attention and the feed-forward network work out does not replace the token’s vector; it is added to it. The earlier information is kept, and a model with many layers is easier to train this way. This is called a residual connection.</p><p><b>Rescaling.</b> Before attention and before the feed-forward network, the vector is scaled as a whole to a similar size, so the numbers do not keep growing or shrinking from layer to layer. This is called normalization; the model in this lesson uses a kind called RMSNorm. After the last layer, before the scores, it is rescaled once more.</p>`,
     `<p><b>加回去。</b>注意力和前馈网络算出的结果，不是替换掉 token 原来的向量，而是加到它上面。这样原来的信息不会丢，很多层叠起来也更好训练。这叫残差连接。</p><p><b>调大小。</b>在注意力和前馈网络之前，先把向量整体缩放到差不多的大小，免得一层层算下来，数越来越大或越来越小。这叫归一化；这一课的模型用的是一种叫 RMSNorm 的做法。最后一层之后、打分之前，还会再做一次。</p>`)},
  {k:'mask',t:tx('Why attention only looks back','为什么注意力只往前看'),b:tx(`<p>When the model writes, the later words do not exist yet, so there is nothing ahead to look at. During training the whole sentence is there, but the model practises guessing every next token at once: the 1st token guesses the 2nd, the first 2 guess the 3rd, and so on. Letting a token look ahead would show it the answer. So the model looks only back, in training and in writing.</p><p>By the way, every position has a vector after the last layer. In training each of them is used to guess that position’s next token; in writing only the last one is used (section 9).</p>`,
     `<p>模型写的时候，后面的词还没写出来，本来就没有可看的。训练的时候整句话都在，但模型要同时练习猜每一个“下一个 token”：第 1 个 token 猜第 2 个，前 2 个猜第 3 个……如果让它往后看，就等于直接看到了答案。所以训练和写的时候，都只往前看。</p><p>顺带一提：最后一层之后，每个位置都有一个向量。训练时每个位置的向量都用来猜它自己的下一个 token；写的时候只用最后一个（第 9 段）。</p>`)},
  {k:'sizes',t:tx('What changes in a bigger model','大模型和小模型差在哪'),b:tx('<p>The same family comes in three sizes. The structure is the same; these numbers grow:</p>','<p>同一个系列有三种大小。结构一样，变大的是这几个数：</p>')+
     `<div class="ls-tw"><table class="ls-tbl"><thead><tr><th></th><th>${tx('This lesson','本课')}</th><th>${tx('Larger','更大')}</th><th>${tx('Largest','最大')}</th></tr></thead><tbody>`+
     `<tr><th>${tx('Numbers per vector','每个向量的数')}</th><td>4,096</td><td>8,192</td><td>16,384</td></tr><tr><th>${tx('Layers','层数')}</th><td>32</td><td>80</td><td>126</td></tr>`+
     `<tr><th>${tx('Heads per attention','每次注意力的头数')}</th><td>32</td><td>64</td><td>128</td></tr><tr><th>${tx('Feed-forward middle','前馈网络中间')}</th><td>14,336</td><td>28,672</td><td>53,248</td></tr>`+
     `<tr><th>${tx('Tokens it knows','认识的 token')}</th><td>128,256</td><td>128,256</td><td>128,256</td></tr><tr><th>${tx('Learned numbers in total','学到的数，总共')}</th><td>${tx('about 8 billion','约 80 亿')}</td><td>${tx('about 70 billion','约 700 亿')}</td><td>${tx('about 405 billion','约 4,050 亿')}</td></tr></tbody></table></div>`+
     tx('<p>“8B”, “70B” and “405B” in model names count these learned numbers, called parameters.</p>','<p>模型名字里的“8B”“70B”“405B”，数的就是这些学到的数，叫参数（B 是十亿）。</p>')}];
  return `<section class="ls-sec ls-deep" data-ls="deep" id="ls-deep"><h2>${H().deep}</h2><p>${tx('You can skip these; the lesson makes sense without them.','不读也不影响理解这一课。')}</p>`+
    D.map(d=>`<details class="ls-dd" data-deep="${d.k}"${S.open.includes(d.k)?' open':''}><summary>${d.t}</summary><div class="ls-dd-b">${d.b}</div></details>`).join('')+`</section>`;}
function srcHTML(){const a=(u,t)=>`<a href="${u}" target="_blank" rel="noopener">${t}</a>`;
  return `<section class="ls-sec ls-src" data-ls="src" id="ls-src"><h2>${H().src}</h2>`+
    `<p>${tx('<b>The model.</b> The sizes in this lesson are those of Llama 3.1 8B, a language model Meta released openly in 2024 (Llama is the name of Meta’s model family; 8B means about 8 billion learned numbers). We use it because its structure and sizes are fully published; many newer open models keep much the same structure.',
      '<b>用的是哪个模型。</b>这一课的尺寸来自 Llama 3.1 8B，是 Meta 在 2024 年公开发布的一个语言模型（Llama 是 Meta 的模型系列名，8B 指约 80 亿个学到的数）。选它是因为它的结构和尺寸都完整公开；很多更新的公开模型，结构也和它大致相同。')}</p>`+
    `<p>${tx(`<b>What is real.</b> How the sentence is cut and the token numbers (Meta’s Llama 3 tokenizer); the sizes ${N(DM)}, ${NLAY} layers, ${NHEAD} heads, ${N(FFN)} and ${N(VOCAB)} (Meta’s published configuration; ${N(VOCAB)} includes 256 tokens kept for special uses).`,
      `<b>哪些是真的。</b>句子的切法和 token 编号（Meta 的 Llama 3 分词规则）；${N(DM)}、${NLAY} 层、${NHEAD} 个头、${N(FFN)}、${N(VOCAB)} 这些尺寸（Meta 公布的模型配置；${N(VOCAB)} 里有 256 个是留作特殊用途的 token）。`)}</p>`+
    `<p>${tx('<b>What is drawn by hand.</b> The numbers inside the vectors, the attention shares and the probabilities. This page does not run the model; these values are made up so the pictures are easy to read. That Paris comes next is the common-sense example, not something this page checked with the model. The draws in section 9 really are random, using the probabilities shown.',
      '<b>哪些是画出来的。</b>向量里的数、注意力的比例、各个 token 的概率。这一页没有运行模型，这些数是为了让图好读编的。“下一个是 Paris”是常识性的例子，没有用模型核对过。第 9 段的抽取是真的按图上的概率随机抽。')}</p>`+
    `<p>${tx('<b>What was left out to keep it simple.</b> In attention, what is mixed is not the vectors themselves but another list of numbers worked out from each vector. The 32 heads are in 8 groups, and the heads of a group share the numbers being looked at, to save memory (later lessons call this GQA). Each layer also adds its result back and rescales (see “A bit deeper”).','<b>为了好懂，省略了什么。</b>注意力里混合的不是向量本身，而是从每个向量换算出来的另一列数。32 个头分成 8 组，同一组的头共用“被参考”的那部分数，用来省内存（后面的课会讲，叫 GQA）。每层还会把结果加回去、调整大小（见“深入一点”）。')}</p>`+`<ul class="ls-refs"><li>${a('https://arxiv.org/abs/1706.03762','Attention Is All You Need')} · ${tx('Vaswani et al., 2017: the Transformer','Vaswani 等，2017：Transformer')}</li>`+
    `<li>${a('https://arxiv.org/abs/2407.21783','The Llama 3 Herd of Models')} · ${tx('Llama Team, Meta, 2024: the model and its sizes','Meta Llama 团队，2024：模型和尺寸')}</li>`+
    `<li>${a('https://github.com/meta-llama/llama-models','meta-llama/llama-models')} · ${tx('Meta: the model code and tokenizer','Meta：模型代码和分词器')}</li>`+
    `<li>${a('https://arxiv.org/abs/2104.09864','RoFormer')} · ${tx('Su et al., 2021: rotary position embedding','Su 等，2021：旋转位置编码')}</li>`+
    `<li>${a('https://arxiv.org/abs/2012.14913','Transformer Feed-Forward Layers Are Key-Value Memories')} · ${tx('Geva et al., 2021','Geva 等，2021')}</li>`+
    `<li>${a('https://arxiv.org/abs/2202.05262','Locating and Editing Factual Associations in GPT')} · ${tx('Meng et al., 2022: facts and the feed-forward network','Meng 等，2022：事实和前馈网络')}</li>`+
    `<li>${a('https://poloclub.github.io/transformer-explainer/','Transformer Explainer')} · ${tx('Cho et al., 2024: an interactive walk through GPT-2','Cho 等，2024：交互式讲解 GPT-2')}</li></ul></section>`;}
function pageHTML(){return `<div class="ls" id="ls"><div class="ls-grid"><nav class="ls-toc" aria-label="${esc(tx('Lesson contents','本课目录'))}"><p class="ls-toc-h">${tx('Contents','本课目录')}</p>${tocList()}</nav>`+
  `<article class="ls-main">${headHTML()}${SEC.map(secHTML).join('')}${sumHTML()}${quizHTML()}${nextHTML()}${deepHTML()}${srcHTML()}</article></div>`+
  `<div class="ls-note" id="ls-note" role="status" hidden></div></div>`;}
function fill(){if(!root) return; const set=(id,h)=>{const e=$('#'+id,root); if(e) e.innerHTML=h;};
  set('ls-f1',f1()); set('ls-f4',f4()); set('ls-f5',f5()); set('ls-f6',f6()); set('ls-f7',f7()); set('ls-p8',p8()); set('ls-f8',f8()); set('ls-f9',f9()); set('ls-f10',f10()); set('ls-qz',qzHTML());}

/* ---------- where the reader is ---------- */
const topLine=()=>{const bar=$('.topbar'); return (bar?bar.getBoundingClientRect().bottom:58)+12;};
function blockEl(k){return root?root.querySelector(`[data-ls="${k}"]`):null;}
function readPos(){if(!root||!root.isConnected) return; const y=topLine(); let at='head', off=0;
  for(const k of BLOCKS){const e=blockEl(k); if(!e) continue; const r=e.getBoundingClientRect(); if(r.top<=y){at=k; off=Math.round(y-r.top);} else break;}
  if(window.scrollY<4){at='head'; off=0;}
  S.at=at; S.off=Math.max(0,Math.min(off,20000));
  const i=BLOCKS.indexOf(at); if(i>=BLOCKS.indexOf('sum')) S.done=true;
  tocMark();}
function tocMark(){if(!root) return; const k=S.at, n=/^s\d+$/.test(k)?k.slice(1):(['sum','quiz','deep','src'].includes(k)?k:(k==='next'?'quiz':''));
  $$('.ls-toc button[data-go]',root).forEach(b=>{const on=b.dataset.go===n; b.classList.toggle('on',on); if(on) b.setAttribute('aria-current','true'); else b.removeAttribute('aria-current');});}
function goPos(k,off,smooth){const e=blockEl(k); if(!e) return; const y=window.scrollY+e.getBoundingClientRect().top-topLine()+(off||0);
  holdScroll=Date.now()+(smooth?900:500); window.scrollTo({top:Math.max(0,y),behavior:smooth&&!A.reduceMotion()?'smooth':'auto'});}
const secName=k=>/^s\d+$/.test(k)?tx(`section ${k.slice(1)}`,`第 ${k.slice(1)} 段`):({sum:H().sum,quiz:H().quiz,next:H().next,deep:H().deep,src:H().src}[k]||'');
function note(k){const n=$('#ls-note',root); if(!n) return; clearTimeout(noteT);
  n.innerHTML=`<span>${tx(`Back where you left off: ${secName(k)}.`,`已回到你上次读到的地方：${secName(k)}。`)}</span><button type="button" class="linkbtn" data-act="from-top">${tx('Start from the top','从头开始读')}</button><button type="button" class="linkbtn" data-act="note-x">${tx('OK','知道了')}</button>`;
  n.hidden=false; noteT=setTimeout(()=>{n.hidden=true;},9000);}
// after the kit's scrollTo(0, 0): the place the reader left, unless that is the top
function placeReader(){if(S.at==='head'&&!S.off) return; const k=S.at, off=S.off;
  requestAnimationFrame(()=>{goPos(k,off,false); note(k); setTimeout(()=>{if(Date.now()<holdScroll+400) goPos(k,off,false);},260);});}
let saveT=0;
function onScroll(){if(!root||!root.isConnected) return; if(Date.now()<holdScroll) return; readPos();
  clearTimeout(saveT); saveT=setTimeout(()=>{try{window.__prog&&window.__prog.save();}catch(e){}},700);}
let ticking=false; const scrollH=()=>{if(ticking) return; ticking=true; requestAnimationFrame(()=>{ticking=false; onScroll();});};

/* ---------- clicks ---------- */
function bind(el){el.addEventListener('click',e=>{const t=e.target.closest('button'); if(!t||!el.contains(t)) return;
  const act=t.dataset.act;
  if(act==='s1-next'){S.s1=Math.min(2,S.s1+1); fill(); return;}
  if(act==='s1-reset'){S.s1=0; fill(); return;}
  if(t.dataset.pick!==undefined){S.pick=+t.dataset.pick; $('#ls-f5',root).innerHTML=f5(); const nb=$(`#ls-f5 [data-pick="${S.pick}"]`,root); if(nb) nb.focus({preventScroll:true}); return;}
  if(t.dataset.p8!==undefined){S.p8=+t.dataset.p8; $('#ls-p8',root).innerHTML=p8(); $('#ls-f8',root).innerHTML=f8(); return;}
  if(act==='draw'){const r=Math.random(); let c=0, d=CANDS.length; for(let i=0;i<CANDS.length;i++){c+=CANDS[i].p; if(r<c){d=i; break;}}
    S.draws.push(d); if(S.draws.length>500) S.draws.shift(); $('#ls-f9',root).innerHTML=f9(); const nb=$('#ls-f9 [data-act="draw"]',root); if(nb) nb.focus({preventScroll:true}); return;}
  if(act==='draw-reset'){S.draws=[]; $('#ls-f9',root).innerHTML=f9(); const nb=$('#ls-f9 [data-act="draw"]',root); if(nb) nb.focus({preventScroll:true}); return;}
  if(t.dataset.qz!==undefined){S.quiz[+t.dataset.qz]=+t.dataset.o; $('#ls-qz',root).innerHTML=qzHTML(); const nb=$(`#ls-qz [data-qz="${t.dataset.qz}"][data-o="${t.dataset.o}"]`,root); if(nb) nb.focus({preventScroll:true}); return;}
  if(t.dataset.go){const g=t.dataset.go, k=/^\d+$/.test(g)?'s'+g:g; const d=t.closest('details.ls-toc-n'); if(d) d.open=false; goPos(k,0,true); S.at=k; S.off=0; tocMark();
    const h=blockEl(k)&&blockEl(k).querySelector('h2'); if(h){h.setAttribute('tabindex','-1'); setTimeout(()=>h.focus({preventScroll:true}),A.reduceMotion()?0:500);} return;}
  if(act==='from-top'){$('#ls-note',root).hidden=true; S.at='head'; S.off=0; holdScroll=Date.now()+500; window.scrollTo(0,0); tocMark(); return;}
  if(act==='note-x'){$('#ls-note',root).hidden=true; return;}});
  el.addEventListener('toggle',e=>{const d=e.target; if(!d.matches||!d.matches('details.ls-dd')) return; const k=d.dataset.deep;
    S.open=S.open.filter(x=>x!==k); if(d.open) S.open.push(k);},true);}

/* ---------- the kit ---------- */
A.registerLab({id:ID,lesson:true,
  mount(holder){root=holder; holder.innerHTML=pageHTML(); fill(); bind(holder); tocMark();
    window.removeEventListener('scroll',scrollH); window.addEventListener('scroll',scrollH,{passive:true});
    placeReader(); restored=false;},
  leave(){window.removeEventListener('scroll',scrollH); clearTimeout(noteT); clearTimeout(saveT); root=null;},
  onLang(){if(!root) return; const k=S.at, off=S.off, open=S.open.slice(); root.innerHTML=pageHTML(); S.open=open; fill(); tocMark(); if(!(k==='head'&&!off)) goPos(k,off,false);},
  onResize(){},
  snap:()=>({v:1,mode:'lesson',step:Math.max(0,Math.min(9,(/^s\d+$/.test(S.at)?+S.at.slice(1):(S.at==='head'?1:10))-1)),nSteps:10,at:S.at,off:S.off,
    s1:S.s1,pick:S.pick,p8:S.p8,draws:S.draws.slice(),quiz:S.quiz.slice(),open:S.open.slice(),done:S.done,passed:quizPassed()}),
  restore(o){if(!o||o.v!==1) return; const n=S0(), int=(v,a,z)=>Number.isInteger(v)&&v>=a&&v<=z;
    n.at=BLOCKS.includes(o.at)?o.at:'head'; n.off=int(o.off,0,20000)?o.off:0;
    n.s1=int(o.s1,0,2)?o.s1:0; n.pick=int(o.pick,0,4)?o.pick:4; n.p8=int(o.p8,0,2)?o.p8:null;
    n.draws=Array.isArray(o.draws)?o.draws.filter(d=>int(d,0,CANDS.length)).slice(-500):[];
    n.quiz=Array.isArray(o.quiz)&&o.quiz.length===3?o.quiz.map(v=>int(v,0,2)?v:null):[null,null,null];
    n.open=Array.isArray(o.open)?DEEP.filter(k=>o.open.includes(k)):[]; n.done=o.done===true;
    S=n; restored=true;}});

/* read-only hook for the tests */
window.__ls=Object.freeze({state:()=>JSON.parse(JSON.stringify(S)),blocks:BLOCKS,AW,HEADS,CANDS,REST,TOKS,sizes:{DM,NLAY,NHEAD,FFN,VOCAB},
  texts:()=>root?root.innerText:''});
})();
