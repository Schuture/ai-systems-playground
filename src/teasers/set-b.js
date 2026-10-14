/* Set B home-tile teasers: sampling and tokenization-cost (Models). The grpo and rlvr tiles that were here gave way to
   those labs' own tiles in batch 9 (teasers/grpo.js, teasers/rlvr.js), as the training-memory and parallelism tiles
   did in batch 8 (teasers/training-memory.js, teasers/parallelism.js).
   Each teaser: one control, one small visual with a one-line legend,
   one or two sentences (about 35 English words at most, guide R11.5) that restate the setup and the result, and
   .tz-note paragraphs (plain text) with 2–4 short sentences of assumptions and sources, each answering one question
   a reader would ask; the tile hides the notes behind its "Assumptions & sources" button.
   Engines are pure functions of the teaser state; every number on the tiles comes from them.
   window.__tzB exposes them read-only for tests/set-b_numbers.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const tx=A.tx, esc=A.esc, GiB=A.GiB||1073741824, GB=1e9, MINUS='−';

/* ---------- shared helpers ---------- */
// fixed decimals, thousands separators, a real minus sign; JS toFixed rounds ties away from zero
function nf(v,d){d=d||0; const s=Math.abs(v).toFixed(d), neg=v<0&&+s!==0;
  const [i,f]=s.split('.'); return (neg?MINUS:'')+i.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(f!==undefined?'.'+f:'');}
// percentages: whole numbers in the middle of the range, one decimal near 0% and 100%
const pct=x=>{const v=x*100; return nf(v,(v<10||v>90)&&Math.abs(v-Math.round(v))>1e-9?1:0)+'%';};
// memory: GiB on both training tiles (one decimal below 1,000 GiB, two below 1 GiB); GB = 10^9 bytes where a title uses GB
const gib=(b,d)=>{const g=b/GiB; return nf(g,d!=null?d:(g>=1000?0:(g<1?2:1)))+' GiB';};
const gb=(b,d)=>nf(b/GB,d||0)+' GB';
function bump(el){if(!el||A.reduceMotion()) return; el.classList.remove('tz-setb-bump'); void el.offsetWidth; el.classList.add('tz-setb-bump');}
// write the sentence; flash it when it changed because of a user action (guide R1.9)
function setRead(el,html,user){if(el._h===html) return; el.innerHTML=html; el._h=html; if(user){bump(el); A.announce(el.textContent);}}
const q=(sel,el)=>el.querySelector(sel);
function stopSlider(n,i,label){return `<input type="range" class="tz-setb-range" min="0" max="${n-1}" step="1" value="${i}" aria-label="${esc(label)}">`;}
const legItem=(sw,text,extra)=>`<span${extra||''}><i class="sw ${sw}"></i>${text}</span>`;
const notes=n=>Array.from({length:n},()=>'<p class="tz-note"></p>').join('');
function setNotes(el,list){el.querySelectorAll('.tz-note').forEach((p,i)=>{const t=list[i]||''; if(p.textContent!==t) p.textContent=t;});}

/* ======================================================================================
   sampling: temperature on a toy next-word distribution, with top-p fixed at 0.9
   ====================================================================================== */
// p = softmax(logits / T); then top-p keeps the smallest set of most likely words whose probabilities add up
// to at least 0.9 and rescales them (Holtzman et al., 2020). Hugging Face applies temperature before top-p.
const SP={logits:[2.0,1.5,1.0,0.3,-0.5,-1.0], topP:0.9, tMin:1, tMax:20,
  words:{en:['noodles','rice','dumplings','hot pot','salad','a burger'],zh:['面条','米饭','饺子','火锅','沙拉','汉堡']}};
function softmax(z){const m=Math.max(...z), e=z.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s);}
function spCompute(T,topP){
  const p=softmax(SP.logits.map(l=>l/T)), order=p.map((_,i)=>i).sort((a,b)=>p[b]-p[a]||a-b), keep=p.map(()=>false);
  let cum=0; for(const i of order){if(cum<topP){keep[i]=true; cum+=p[i];}}
  const kept=p.reduce((a,v,i)=>a+(keep[i]?v:0),0), fin=p.map((v,i)=>keep[i]?v/kept:0);
  return {T,topP,p,keep,fin,nKept:keep.filter(Boolean).length,top:order[0],raw:softmax(SP.logits)};
}
let spI=10;                                   // temperature = spI / 10
const tStr=i=>nf(i/10,1);
function spRead(r){
  const w=SP.words[A.lang()]||SP.words.en, top=w[r.top], T=nf(r.T,1), tp=nf(r.topP,1);
  if(r.nKept===1) return tx(`At temperature ${T} with top-p ${tp}, only “${top}” stays (it alone has ${pct(r.p[r.top])} of the probability), so it is picked <b>every time</b> (${pct(r.raw[r.top])} at temperature 1 without top-p).`,
    `temperature ${T}、top-p ${tp} 时只剩“${top}”（它一个就占了 ${pct(r.p[r.top])} 的概率），<b>每次都选它</b>（模型原本的概率是 ${pct(r.raw[r.top])}）。`);
  return tx(`At temperature ${T} with top-p ${tp}, ${r.nKept} of ${r.p.length} words stay and “${top}” is picked <b>${pct(r.fin[r.top])}</b> of the time (${pct(r.raw[r.top])} at temperature 1 without top-p).`,
    `temperature ${T}、top-p ${tp} 时，${r.p.length} 个词留下 ${r.nKept} 个，“${top}”被选中的概率是 <b>${pct(r.fin[r.top])}</b>（模型原本的概率是 ${pct(r.raw[r.top])}）。`);
}
// notes: 3 sentences (guide R11.5); no sentence about why the tile keeps top-p fixed (R3.4)
function spNotes(){
  return [tx(`Every score is divided by the temperature before the softmax; top-p then keeps the likeliest words until they add up to ${nf(SP.topP,1)} and rescales them (Holtzman et al., 2020). Hugging Face applies temperature first.`,
      `每个分数先除以 temperature 再做 softmax；top-p 再从最可能的词开始保留，直到概率加起来到 ${nf(SP.topP,1)}，并把它们重新放大（Holtzman 等，2020）。Hugging Face 也是先做 temperature。`),
    tx(`The ${SP.logits.length} scores are made up; a real model scores its whole vocabulary.`,
      `这 ${SP.logits.length} 个分数是编的；真实模型会给整个词表打分。`)];
}
A.registerTeaser('sampling',{render(el){
  const w=SP.words[A.lang()]||SP.words.en, lab=tx('Temperature','Temperature');
  const rows=w.map((x,i)=>`<div class="tz-samp-row"><span class="tz-samp-w">${esc(x)}</span><span class="tz-samp-trk"><i class="tz-samp-pre"></i><i class="tz-samp-fin"></i></span><span class="tz-samp-v mono"></span></div>`).join('');
  el.innerHTML=`<div class="tz tz-setb tz-samp"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+
    `<input type="range" class="tz-setb-range" min="${SP.tMin}" max="${SP.tMax}" step="1" value="${spI}" aria-label="${esc(lab)}">`+
    `<div class="tz-samp-vis" role="img"><div class="tz-samp-title">${tx('Next word after “For lunch today I want”','“我今天午饭想吃”之后的下一个词')}</div><div class="tz-samp-rows">${rows}</div></div>`+
    `<div class="tz-leg">${legItem('tz-samp-swfin',tx('chance of being picked','被选中的概率'))}${legItem('tz-samp-swcut',tx('cut by top-p (its share before the cut)','被 top-p 去掉（虚线是去掉前的份额）'))}</div>`+
    `<p class="tz-read tz-setb-read"></p>`+notes(2)+`</div>`;
  const inp=q('input',el), out=q('output',el), rowsEl=Array.from(el.querySelectorAll('.tz-samp-row')), vis=q('.tz-samp-vis',el), rd=q('.tz-read',el);
  setNotes(el,spNotes());
  const upd=user=>{spI=+inp.value; A.ui.setRange(inp); const r=spCompute(spI/10,SP.topP);
    out.textContent=tStr(spI);
    rowsEl.forEach((row,i)=>{const cut=!r.keep[i]; row.classList.toggle('cut',cut);
      q('.tz-samp-pre',row).style.width=(cut?r.p[i]*100:0)+'%'; q('.tz-samp-fin',row).style.width=(r.fin[i]*100)+'%';
      q('.tz-samp-v',row).textContent=cut?tx('cut','去掉'):pct(r.fin[i]);});
    vis.setAttribute('aria-label',tx(`${r.nKept} of 6 words kept; the likeliest is picked ${pct(r.fin[r.top])} of the time.`,`6 个词留下 ${r.nKept} 个；最可能的词被选中的概率是 ${pct(r.fin[r.top])}。`));
    setRead(rd,spRead(r),user);};
  inp.addEventListener('input',()=>upd(true)); upd(false);
}});

/* ======================================================================================
   tokenization-cost: a byte-level BPE trained live on a small, mostly English text
   ====================================================================================== */
// Training text written for this demo: mostly English, a little Chinese. Pre-tokenized GPT-2 style (a leading
// space joins the next word; letters, digits and other symbols form separate runs), then BPE from the 256 byte
// values: repeatedly merge the most frequent adjacent pair (ties: the smaller pair of ids).
const TK_EN=[
'Every morning I make a cup of tea and sit by the window. The street is quiet at that hour, and the light is soft. I like to read before the day begins, even if it is only a few pages. Some days I read the news, but most days I read books. A good book makes the morning feel longer, in a good way.',
'My sister likes to run in the morning. She says the park is at its best before eight, when the grass is still wet and the air is cool. I have tried to join her, but I am a slow runner, and I would rather walk. We often meet for breakfast after her run. She eats eggs and toast; I eat rice and fruit.',
'In the afternoon I work at a small library near the river. People come in to read, to study, or just to sit somewhere warm. Children like the picture books; students like the long tables by the windows. The library has more than ten thousand books, and every week we add new ones. My job is to put the books back on the shelves, to help people find what they need, and to keep the rooms quiet.',
'The weather in our town changes quickly. In the spring it rains almost every day, and in the summer it is hot and bright. The best time of the year is the autumn, when the trees turn red and gold and the mornings are cool again. In the winter the river freezes near the bridge, and people walk on the ice.',
'On weekends my friends and I like to cook together. We choose a recipe, buy what we need at the market, and spend the whole afternoon in the kitchen. Last week we made noodles by hand. It took three hours, and the noodles were not very even, but they were the best we have ever eaten. Next time we want to try dumplings.',
'Learning a new skill takes time. When I started to learn the piano, I could only play one short song, and I played it slowly. I practiced for twenty minutes every evening. After a year I could read music and play simple pieces with both hands. The most important thing was not talent but the habit of practicing a little every day.',
'My grandmother taught me that habit. She was a teacher for forty years, and she still reads every morning with a pot of tea at her side. She keeps a notebook of the books she has read, with a short note about each one. The notebook is now full, and she has started a second one. When I visit her, we talk about the books we are reading and trade the ones we like.',
'Travel is another way to learn. Two years ago I took a train across the country. The trip took four days, and I spent most of the time looking out of the window, reading, and talking with other people on the train. I met a farmer who grew apples, a student who studied the history of old bridges, and a nurse who was going home to see her family. Each of them had a story, and I wrote many of those stories down in my notebook.',
'At night the city is different. The shops close, the lights come on, and the streets fill with people who want to eat, to walk, or to meet friends. I like to walk home along the river, where it is dark and calm. When I get home I make another cup of tea, read a few more pages, and go to sleep early, so that I can read again in the morning.',
'Our neighbor has a small garden behind his house. He grows tomatoes, beans, and herbs, and in the summer he gives us more than we can eat. He says that a garden teaches patience, because nothing grows faster when you watch it. In the evening he sits on a bench under the old apple tree and listens to the radio.',
'The bus to the city center leaves every fifteen minutes. In the morning it is full of people going to work, and most of them read something on the way: a book, a paper, or the news on their phones. I usually stand near the back and read a chapter before my stop. The ride takes about half an hour, which is just long enough for one chapter.',
'There is a small bakery on the corner of our street. The baker starts work at four in the morning, and by six the whole street smells of fresh bread. On Saturdays there is a line outside the door before it opens. I like the bread with seeds on top, and my sister likes the sweet rolls with cinnamon.',
'When it rains, the city becomes quieter. People stay inside, the cafes are full, and the streets shine under the lights. I like to sit in a cafe on a rainy afternoon with a warm drink and a book. The sound of the rain on the window makes it easy to read for hours.',
'My brother is learning to drive. He practices with our father on Sunday mornings, when the roads are empty. At first he was nervous and drove very slowly, but now he is more careful and more confident. He wants to take the test next month, and he has been reading the rules of the road every night.',
'Last summer we spent a week by the sea. We stayed in a small house near the beach, and every morning we walked along the water before breakfast. The children built castles in the sand, and in the evening we watched the sun go down over the waves. It was the first real holiday we had taken in years.',
'Writing letters is a habit that few people keep today. My aunt still writes to her friends by hand, and she keeps every letter she receives in a wooden box. She says that a letter is slower than a message, but it shows that someone spent time thinking of you. When I was a child, she wrote to me every month.',
'The school near our house has a large field where children play football after class. On warm evenings the field is full of noise and running, and parents stand at the edge talking with each other. When the sun goes down, the lights come on, and the older students play until late.',
'A good meal does not need to be complicated. Some of the best dinners I remember were very simple: fresh bread, a bowl of soup, and good company. What makes a meal special is often the people around the table and the conversation that goes on long after the plates are empty.',
'Every city has its own rhythm. Some wake up early and go to bed early, and others come alive only at night. Our town is an early town. The markets open at dawn, the parks are busy by seven, and by ten in the evening most windows are dark. I have learned to like this rhythm, and I now wake up before my alarm.'];
const TK_ZH='我每天早上都会泡一杯茶，坐在窗边看几页书。这个时候街上很安静，阳光也很柔和。我妹妹喜欢在公园跑步，她说八点以前空气最好。下午我在河边的图书馆工作，那里有一万多本书，每周都会来新书。周末我们常常一起做饭，上个星期自己做了面条，虽然粗细不匀，但是非常好吃。';
const TK={stops:[0,10,25,50,100,200,300], en:'I like to read books in the morning.', zh:'我喜欢在早上看书。'};
const TK_RE=/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;
let TKM=null, tkI=4;
const utf8=s=>Array.from(new TextEncoder().encode(s));
function mergeSeq(s,a,b,id){const out=[]; for(let j=0;j<s.length;j++){if(j+1<s.length&&s[j]===a&&s[j+1]===b){out.push(id); j++;} else out.push(s[j]);} return out;}
function tokModel(){
  if(TKM) return TKM;
  const kmax=TK.stops[TK.stops.length-1], text=TK_EN.join(' ')+' '+TK_ZH, freq=new Map();
  for(const m of text.matchAll(TK_RE)) freq.set(m[0],(freq.get(m[0])||0)+1);
  const words=[]; for(const [w,f] of freq) words.push({s:utf8(w),f});
  const merges=[], len=[]; for(let i=0;i<256;i++) len[i]=1;
  // pair counts over all words (weighted by word frequency), kept up to date: after a merge only the words
  // that contained the merged pair are recounted
  const cnt=new Map(), count=(s,f)=>{for(let j=0;j+1<s.length;j++){const key=s[j]*4096+s[j+1], v=(cnt.get(key)||0)+f; if(v) cnt.set(key,v); else cnt.delete(key);}};
  const has=(s,a,b)=>{for(let j=0;j+1<s.length;j++) if(s[j]===a&&s[j+1]===b) return true; return false;};
  for(const w of words) count(w.s,w.f);
  for(let i=0;i<kmax;i++){
    let best=-1, bc=0; for(const [key,c] of cnt){if(c>bc||(c===bc&&key<best)){bc=c; best=key;}}   // most frequent; ties: smaller pair
    if(bc<2) break;
    const a=Math.floor(best/4096), b=best%4096, id=256+i; merges.push([a,b,bc]); len[id]=len[a]+len[b];
    for(const w of words) if(has(w.s,a,b)){count(w.s,-w.f); w.s=mergeSeq(w.s,a,b,id); count(w.s,w.f);}
  }
  // tokens of the two sentences after every merge count: each token as the number of bytes it covers
  const enc=s=>Array.from(s.matchAll(TK_RE),m=>utf8(m[0]));
  const seqs={en:enc(TK.en),zh:enc(TK.zh)}, toks={en:[],zh:[]};
  for(let k=0;k<=merges.length;k++){
    for(const L of ['en','zh']) toks[L].push(seqs[L].flatMap(s=>s.map(t=>len[t])));
    if(k<merges.length){const [a,b]=merges[k]; for(const L of ['en','zh']) seqs[L]=seqs[L].map(s=>mergeSeq(s,a,b,256+k));}
  }
  const enB=utf8(TK_EN.join(' ')).length, zhB=utf8(TK_ZH).length;
  TKM={merges,toks,bytes:{en:utf8(TK.en).length,zh:utf8(TK.zh).length},corpus:{enBytes:enB,zhBytes:zhB,enShare:enB/(enB+zhB)}};
  return TKM;
}
function tokCompute(k){const m=tokModel(), kk=Math.min(k,m.merges.length);
  return {k:kk,en:m.toks.en[kk].length,zh:m.toks.zh[kk].length,enTok:m.toks.en[kk].slice(),zhTok:m.toks.zh[kk].slice(),
    enBytes:m.bytes.en,zhBytes:m.bytes.zh,ratio:m.toks.zh[kk].length/m.toks.en[kk].length,share:m.corpus.enShare};}
// the sentence says what each count is (tokens of which sentence) in words, without arrows (guide R6.13, R6.14)
function tokRead(r){
  if(r.k===0) return tx(`With no merges, every UTF-8 byte is a token: English has <b>${r.en}</b> tokens, Chinese <b>${r.zh}</b>, since each Chinese character is 3 bytes.`,
    `还没有合并时，每个 UTF-8 字节就是一个 token：英文 <b>${r.en}</b> 个，中文 <b>${r.zh}</b> 个，因为一个汉字占 3 个字节。`);
  const share=nf(r.share*100,0), dEn=r.enBytes-r.en, dZh=r.zhBytes-r.zh;
  const head=tx(`After ${r.k} merges from text that is ${share}% English: English drops from ${r.enBytes} tokens to <b>${r.en}</b>`,`从 ${share}% 是英文的文本学了 ${r.k} 次合并后：英文从 ${r.enBytes} 个 token 减到 <b>${r.en}</b> 个`);
  if(dZh===0) return head+tx(`; Chinese stays at <b>${r.zh}</b>, as no merge fits its bytes.`,`；中文还是 <b>${r.zh}</b> 个，没有一次合并用得上它的字节。`);
  return head+tx(`, Chinese ${dZh<dEn?'only ':''}from ${r.zhBytes} to <b>${r.zh}</b>`,`，中文${dZh<dEn?'只':''}从 ${r.zhBytes} 个减到 <b>${r.zh}</b> 个`)+
    (r.zh>r.en?tx(`, ${nf(r.ratio,1)} times the English count.`,`，是英文的 ${nf(r.ratio,1)} 倍。`):tx('.','。'));
}
// notes: 4 sentences (guide R11.5); the sentence at 0 merges already says a Chinese character is 3 bytes
function tokNotes(m){
  return [tx(`A small byte-level BPE trained on ${nf(m.corpus.enBytes)} bytes of English and ${nf(m.corpus.zhBytes)} of Chinese written for this demo (${nf(m.corpus.enShare*100,0)}% English). Like GPT-2's, it starts from the 256 byte values, merges the most frequent adjacent pair each time, and never merges across words or punctuation.`,
      `一个小的 byte-level BPE，训练文本是为演示写的 ${nf(m.corpus.enBytes)} 字节英文和 ${nf(m.corpus.zhBytes)} 字节中文（${nf(m.corpus.enShare*100,0)}% 是英文）。和 GPT-2 一样，它从 256 个字节值开始，每次合并最常见的相邻一对，不跨词或标点合并。`),
    tx(`These counts come from this small BPE.`,`这里的数来自这个小 BPE。`),
    tx(`On real tokenizers, the same text in different languages can differ up to 15-fold in tokens (Petrov et al., NeurIPS 2023).`,`在真实 tokenizer 上，同一段文字换成不同语言，token 数最多能差 15 倍（Petrov 等，NeurIPS 2023）。`)];
}
function tokStrip(text,toks,n){
  let boxes='', k=0; toks.forEach(len=>{boxes+=`<i class="${k%2?'b':'a'}" style="grid-column:span ${len}"></i>`; k++;});
  const glyphs=Array.from(text).map(ch=>{const b=utf8(ch).length; return `<span style="grid-column:span ${b}">${ch===' '?'<em>·</em>':esc(ch)}</span>`;}).join('');
  return `<div class="tz-tokn-strip" style="--n:${n}"><div class="tz-tokn-glyphs" aria-hidden="true">${glyphs}</div><div class="tz-tokn-boxes">${boxes}</div></div>`;
}
A.registerTeaser('tokenization-cost',{render(el){
  const m=tokModel(), lab=tx('Merges learned','已学的合并次数');
  el.innerHTML=`<div class="tz tz-setb tz-tokn"><label class="tz-lab"><span>${lab}</span><output class="mono"></output></label>`+stopSlider(TK.stops.length,tkI,lab)+
    `<div class="tz-tokn-vis" role="img">`+
    `<div class="tz-tokn-cap"><span>${tx('English','英文')}</span><b></b></div><div class="tz-tokn-en"></div>`+
    `<div class="tz-tokn-cap"><span>${tx('Chinese','中文')}</span><b></b></div><div class="tz-tokn-zh"></div></div>`+
    `<div class="tz-leg">${legItem('tz-tokn-sw',tx('one token, as wide as its UTF-8 bytes','一个 token，宽度 = 它的 UTF-8 字节数'))}<span><b class="tz-tokn-dot">·</b>${tx('a space','空格')}</span></div>`+
    `<p class="tz-read tz-setb-read"></p>`+notes(3)+`</div>`;
  setNotes(el,tokNotes(m));
  const inp=q('input',el), out=q('output',el), caps=el.querySelectorAll('.tz-tokn-cap b'), enEl=q('.tz-tokn-en',el), zhEl=q('.tz-tokn-zh',el), vis=q('.tz-tokn-vis',el), rd=q('.tz-read',el);
  const n=Math.max(tokCompute(0).enBytes,tokCompute(0).zhBytes);
  const upd=user=>{tkI=+inp.value; A.ui.setRange(inp); const r=tokCompute(TK.stops[tkI]);
    out.textContent=String(r.k);
    caps[0].textContent=tx(`${r.en} tokens`,`${r.en} 个 token`); caps[1].textContent=tx(`${r.zh} tokens`,`${r.zh} 个 token`);
    enEl.innerHTML=tokStrip(TK.en,r.enTok,n); zhEl.innerHTML=tokStrip(TK.zh,r.zhTok,n);
    vis.setAttribute('aria-label',tx(`English sentence: ${r.en} tokens. Chinese sentence: ${r.zh} tokens.`,`英文句子 ${r.en} 个 token，中文句子 ${r.zh} 个 token。`));
    setRead(rd,tokRead(r),user);};
  inp.addEventListener('input',()=>upd(true)); upd(false);
}});

/* ---------- read-only hook for tests ---------- */
window.__tzB=Object.freeze({
  fmt:{nf,pct,gib},
  samp:Object.freeze({compute:spCompute,LOGITS:SP.logits.slice(),TOP_P:SP.topP,T_RANGE:[SP.tMin/10,SP.tMax/10]}),
  tok:Object.freeze({compute:tokCompute,STOPS:TK.stops.slice(),SENTENCES:Object.freeze({en:TK.en,zh:TK.zh}),
    corpus:()=>({en:TK_EN.join(' '),zh:TK_ZH,text:TK_EN.join(' ')+' '+TK_ZH}),merges:()=>tokModel().merges.map(x=>x.slice())}),
  state:()=>({spT:spI/10,tokK:TK.stops[tkI]})
});
})();
