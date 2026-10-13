/* tokenization-cost (the trunk "How LLMs work"): Chinese or English: which costs more tokens?
   Fidelity: Live (toy run). The home tile's byte-level BPE (teasers/set-b.js, tokenization-cost): the same training text, the same
   pre-tokenizer regex, the same tie rule (most frequent pair; ties: the smaller pair of ids; stop when the best pair occurs fewer than
   2 times), trained in the browser. The lab extends it with four training mixes built from the same texts (the English paragraphs, and
   the one Chinese paragraph counted 0, 1, 16 or 48 times) and up to 400 merges. The mix "tile" is exactly the tile's text, so its
   merges and counts equal the tile's.
   Words (EN / ZH): tokenizer; byte-level BPE; merge / 合并; training text / 训练文本; vocabulary / 词表; byte / 字节; part / 部分.
   window.__tk is a read-only hook for tests/tokenization-cost_numbers.py and tests/tokenization-cost_flow.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {esc,$,$$}=A, ID='tokenization-cost';
/* Chinese keeps a space between Chinese characters and Latin letters or digits, as in the other labs */
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const L=o=>A.lang()==='zh'?zfix(A.L(o)):A.L(o);
const fint=A.fmt.int, NS=' ';
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});

/* =====================================================================
   engine: the home tile's byte-level BPE (teasers/set-b.js), extended
   ===================================================================== */
/* the training text, copied from teasers/set-b.js (TK_EN, TK_ZH) */
const TK_EN=[
"Every morning I make a cup of tea and sit by the window. The street is quiet at that hour, and the light is soft. I like to read before the day begins, even if it is only a few pages. Some days I read the news, but most days I read books. A good book makes the morning feel longer, in a good way.",
"My sister likes to run in the morning. She says the park is at its best before eight, when the grass is still wet and the air is cool. I have tried to join her, but I am a slow runner, and I would rather walk. We often meet for breakfast after her run. She eats eggs and toast; I eat rice and fruit.",
"In the afternoon I work at a small library near the river. People come in to read, to study, or just to sit somewhere warm. Children like the picture books; students like the long tables by the windows. The library has more than ten thousand books, and every week we add new ones. My job is to put the books back on the shelves, to help people find what they need, and to keep the rooms quiet.",
"The weather in our town changes quickly. In the spring it rains almost every day, and in the summer it is hot and bright. The best time of the year is the autumn, when the trees turn red and gold and the mornings are cool again. In the winter the river freezes near the bridge, and people walk on the ice.",
"On weekends my friends and I like to cook together. We choose a recipe, buy what we need at the market, and spend the whole afternoon in the kitchen. Last week we made noodles by hand. It took three hours, and the noodles were not very even, but they were the best we have ever eaten. Next time we want to try dumplings.",
"Learning a new skill takes time. When I started to learn the piano, I could only play one short song, and I played it slowly. I practiced for twenty minutes every evening. After a year I could read music and play simple pieces with both hands. The most important thing was not talent but the habit of practicing a little every day.",
"My grandmother taught me that habit. She was a teacher for forty years, and she still reads every morning with a pot of tea at her side. She keeps a notebook of the books she has read, with a short note about each one. The notebook is now full, and she has started a second one. When I visit her, we talk about the books we are reading and trade the ones we like.",
"Travel is another way to learn. Two years ago I took a train across the country. The trip took four days, and I spent most of the time looking out of the window, reading, and talking with other people on the train. I met a farmer who grew apples, a student who studied the history of old bridges, and a nurse who was going home to see her family. Each of them had a story, and I wrote many of those stories down in my notebook.",
"At night the city is different. The shops close, the lights come on, and the streets fill with people who want to eat, to walk, or to meet friends. I like to walk home along the river, where it is dark and calm. When I get home I make another cup of tea, read a few more pages, and go to sleep early, so that I can read again in the morning.",
"Our neighbor has a small garden behind his house. He grows tomatoes, beans, and herbs, and in the summer he gives us more than we can eat. He says that a garden teaches patience, because nothing grows faster when you watch it. In the evening he sits on a bench under the old apple tree and listens to the radio.",
"The bus to the city center leaves every fifteen minutes. In the morning it is full of people going to work, and most of them read something on the way: a book, a paper, or the news on their phones. I usually stand near the back and read a chapter before my stop. The ride takes about half an hour, which is just long enough for one chapter.",
"There is a small bakery on the corner of our street. The baker starts work at four in the morning, and by six the whole street smells of fresh bread. On Saturdays there is a line outside the door before it opens. I like the bread with seeds on top, and my sister likes the sweet rolls with cinnamon.",
"When it rains, the city becomes quieter. People stay inside, the cafes are full, and the streets shine under the lights. I like to sit in a cafe on a rainy afternoon with a warm drink and a book. The sound of the rain on the window makes it easy to read for hours.",
"My brother is learning to drive. He practices with our father on Sunday mornings, when the roads are empty. At first he was nervous and drove very slowly, but now he is more careful and more confident. He wants to take the test next month, and he has been reading the rules of the road every night.",
"Last summer we spent a week by the sea. We stayed in a small house near the beach, and every morning we walked along the water before breakfast. The children built castles in the sand, and in the evening we watched the sun go down over the waves. It was the first real holiday we had taken in years.",
"Writing letters is a habit that few people keep today. My aunt still writes to her friends by hand, and she keeps every letter she receives in a wooden box. She says that a letter is slower than a message, but it shows that someone spent time thinking of you. When I was a child, she wrote to me every month.",
"The school near our house has a large field where children play football after class. On warm evenings the field is full of noise and running, and parents stand at the edge talking with each other. When the sun goes down, the lights come on, and the older students play until late.",
"A good meal does not need to be complicated. Some of the best dinners I remember were very simple: fresh bread, a bowl of soup, and good company. What makes a meal special is often the people around the table and the conversation that goes on long after the plates are empty.",
"Every city has its own rhythm. Some wake up early and go to bed early, and others come alive only at night. Our town is an early town. The markets open at dawn, the parks are busy by seven, and by ten in the evening most windows are dark. I have learned to like this rhythm, and I now wake up before my alarm."];
const TK_ZH="我每天早上都会泡一杯茶，坐在窗边看几页书。这个时候街上很安静，阳光也很柔和。我妹妹喜欢在公园跑步，她说八点以前空气最好。下午我在河边的图书馆工作，那里有一万多本书，每周都会来新书。周末我们常常一起做饭，上个星期自己做了面条，虽然粗细不匀，但是非常好吃。";
const EN_TEXT=TK_EN.join(' ');
const TK_RE=/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;   // GPT-2 style pieces, as the tile
const K_MAX=400, CTX=131072, SCEN_K=300;
const BASE={mix:'tile',k:100};
/* the training mixes: the English paragraphs once (or not), the Chinese paragraph w times (a weight on its pieces' counts) */
const MIXES=[
 {id:'en',en:1,zh:0,t:{en:'English only',zh:'只有英文'},p:{en:'English-only text',zh:'只有英文的文本'},d:{en:'the lab’s English paragraphs, no Chinese',zh:'只有 Lab 的英文段落，没有中文'}},
 {id:'tile',en:1,zh:1,t:{en:'Mostly English',zh:'英文为主'},p:{en:'mostly English text',zh:'以英文为主的文本'},d:{en:'the English paragraphs and one Chinese paragraph',zh:'英文段落加一段中文'}},
 {id:'bal',en:1,zh:16,t:{en:'Half and half',zh:'中英各半'},p:{en:'half-and-half text',zh:'中英各半的文本'},d:{en:'the same Chinese paragraph counted 16 times',zh:'同一段中文按 16 份计'}},
 {id:'zh',en:1,zh:48,t:{en:'Mostly Chinese',zh:'中文为主'},p:{en:'mostly Chinese text',zh:'以中文为主的文本'},d:{en:'the same Chinese paragraph counted 48 times',zh:'同一段中文按 48 份计'}}];
const MIX=Object.fromEntries(MIXES.map(m=>[m.id,m]));
/* the sentence pairs: the tile's (every mode but the challenge) and the challenge's, which no other part shows (R3.8) */
const PAIRS={main:{en:'I like to read books in the morning.',zh:'我喜欢在早上看书。'},ch:{en:'My sister runs in the park every morning.',zh:'我妹妹每天早上在公园跑步。'}};
const TE=new TextEncoder(), TD=new TextDecoder('utf-8',{fatal:true});
const utf8=s=>Array.from(TE.encode(s));
function mergeSeq(s,a,b,id){const out=[]; for(let j=0;j<s.length;j++){if(j+1<s.length&&s[j]===a&&s[j+1]===b){out.push(id); j++;} else out.push(s[j]);} return out;}
/* train a mix: pair counts over all pieces (weighted by piece frequency), kept up to date as in the tile */
const MODEL={};
function model(id){
  if(MODEL[id]) return MODEL[id];
  const mx=MIX[id]||MIX.tile, freq=new Map();
  const add=(text,w)=>{if(!w) return; for(const m of text.matchAll(TK_RE)) freq.set(m[0],(freq.get(m[0])||0)+w);};
  add(EN_TEXT,mx.en); add(' '+TK_ZH,mx.zh);
  const words=[]; for(const [w,f] of freq) words.push({s:utf8(w),f});
  const merges=[], bytes=[]; for(let i=0;i<256;i++) bytes[i]=[i];
  const cnt=new Map(), count=(s,f)=>{for(let j=0;j+1<s.length;j++){const key=s[j]*4096+s[j+1], v=(cnt.get(key)||0)+f; if(v) cnt.set(key,v); else cnt.delete(key);}};
  const has=(s,a,b)=>{for(let j=0;j+1<s.length;j++) if(s[j]===a&&s[j+1]===b) return true; return false;};
  for(const w of words) count(w.s,w.f);
  for(let i=0;i<K_MAX;i++){
    let best=-1, bc=0; for(const [key,c] of cnt){if(c>bc||(c===bc&&key<best)){bc=c; best=key;}}
    if(bc<2) break;
    const a=Math.floor(best/4096), b=best%4096, nid=256+i; merges.push([a,b,bc]); bytes[nid]=bytes[a].concat(bytes[b]);
    for(const w of words) if(has(w.s,a,b)){count(w.s,-w.f); w.s=mergeSeq(w.s,a,b,nid); count(w.s,w.f);}
  }
  const enB=utf8(EN_TEXT).length*mx.en, zhB=utf8(TK_ZH).length*mx.zh;
  MODEL[id]={id,merges,bytes,zhPiece:merges.map((_,i)=>bytes[256+i].some(x=>x>=128)),enBytes:enB,zhBytes:zhB,share:enB/(enB+zhB)};
  return MODEL[id];
}
/* a text's tokens after every number of merges: the merges applied in the order they were learned (as the tile) */
const ENC={};
function enc(id,text){
  const key=id+'|'+text; if(ENC[key]) return ENC[key];
  const m=model(id); let seqs=Array.from(text.matchAll(TK_RE),x=>utf8(x[0]));
  const at=[seqs.flat()], used=[];
  for(let k=0;k<m.merges.length;k++){const [a,b]=m.merges[k]; let ch=false;
    seqs=seqs.map(s=>{for(let j=0;j+1<s.length;j++) if(s[j]===a&&s[j+1]===b){ch=true; return mergeSeq(s,a,b,256+k);} return s;});
    used.push(ch); at.push(ch?seqs.flat():at[at.length-1]);}
  /* the characters and the bytes each one takes */
  const chars=[]; let b0=0; for(const ch of Array.from(text)){const n=utf8(ch).length; chars.push({ch,b0,n}); b0+=n;}
  return ENC[key]={text,bytes:b0,at,used,chars};
}
function compute(mix,k,pair){
  const P=PAIRS[pair||'main'], m=model(mix), kk=Math.max(0,Math.min(Math.round(k)||0,m.merges.length)), e=enc(mix,P.en), z=enc(mix,P.zh);
  const en=e.at[kk].length, zh=z.at[kk].length;
  return {mix,k:kk,pair:pair||'main',en,zh,enBytes:e.bytes,zhBytes:z.bytes,enChars:e.chars.length,zhChars:z.chars.length,ratio:zh/en,vocab:256+kk,
    ctxEn:Math.floor(CTX/en),ctxZh:Math.floor(CTX/zh),zhMerges:m.zhPiece.slice(0,kk).filter(Boolean).length,share:m.share};
}
/* the challenge: the new pair; three training texts × four merge counts; the English sentence at most 25 tokens and the Chinese one at
   most 1.5 times the English one's (integers: 2 × zh ≤ 3 × en), with the fewest merges */
const CH={mixes:['tile','bal','zh'],ks:[50,100,200,400],enMax:25,start:{mix:'tile',k:100}};
function chPoint(mix,k){const r=compute(mix,k,'ch'); const okEn=r.en<=CH.enMax, okR=2*r.zh<=3*r.en;
  return {mix,k,en:r.en,zh:r.zh,ratio:r.zh/r.en,okEn,okR,ok:okEn&&okR};}
const chAll=()=>CH.mixes.flatMap(m=>CH.ks.map(k=>chPoint(m,k)));
const chMinK=()=>{const ok=chAll().filter(p=>p.ok); return ok.length?Math.min(...ok.map(p=>p.k)):null;};
const chBest=()=>chAll().filter(p=>p.ok&&p.k===chMinK());
/* B1: what more merges do on the mostly English text, from the challenge's initial setup to the most merges offered */
const chTrend=()=>{const a=chPoint(CH.start.mix,CH.start.k), b=chPoint(CH.start.mix,CH.ks[CH.ks.length-1]); return {a,b,enDrop:a.en-b.en,zhDrop:a.zh-b.zh,zhMore:(a.zh-b.zh)>(a.en-b.en)};};
/* A32: after the most merges offered, two texts can have learned the same merges in a different order */
const sameMerges=(x,y,k)=>{const s=id=>new Set(model(id).merges.slice(0,k).map((_,i)=>model(id).bytes[256+i].join('.'))); const A_=s(x), B_=s(y); return A_.size===B_.size&&[...A_].every(v=>B_.has(v));};
/* the first number of merges after which a character is a whole token of a sentence (part 7's misconception 1) */
function wholeAt(mix,text,ch){const e=enc(mix,text), m=model(mix), want=utf8(ch).join('.'); for(let k=0;k<e.at.length;k++) if(e.at[k].some(id=>m.bytes[id].join('.')===want)) return k; return -1;}
const ZH_B=utf8(TK_ZH), ZH_CH=(()=>{const a=[]; let b0=0; for(const ch of Array.from(TK_ZH)){const n=utf8(ch).length; a.push({ch,b0,n}); b0+=n;} return a;})();
/* where some bytes sit in the Chinese paragraph: “part of 上”, or “end of 看 + 几页 + start of 书” (review A4). A merge's two parts are placed
   inside the new token's own place (at), so a lone byte such as 99 is named from the character it came from in that merge */
const findZh=bs=>{outer: for(let i=0;i+bs.length<=ZH_B.length;i++){for(let j=0;j<bs.length;j++) if(ZH_B[i+j]!==bs[j]) continue outer; return i;} return -1;};
function zhSpan(s,n){
  const e=s+n-1, cs=ZH_CH.filter(c=>c.b0+c.n>s&&c.b0<=e), f=cs[0], l=cs[cs.length-1];
  if(cs.length===1) return {en:`part of “${f.ch}”`,zh:`“${f.ch}”的一部分`};
  const ms=s>f.b0, me=e<l.b0+l.n-1, inner=cs.slice(ms?1:0,me?cs.length-1:cs.length).map(c=>c.ch).join('');
  return {en:[ms?`end of “${f.ch}”`:'',inner?`“${inner}”`:'',me?`start of “${l.ch}”`:''].filter(Boolean).join(' + '),
          zh:[ms?`“${f.ch}”的结尾`:'',inner?`“${inner}”`:'',me?`“${l.ch}”的开头`:''].filter(Boolean).join(' + ')};}
/* the characters of the paragraph that hold these bytes inside them, most frequent first, and where in them (review N3) */
function zhShared(bs){const cnt=new Map(), pos=new Set(), key=bs.join('.');
  for(const c of ZH_CH){if(c.n<=bs.length) continue; const cb=ZH_B.slice(c.b0,c.b0+c.n);
    for(let o=0;o+bs.length<=c.n;o++) if(cb.slice(o,o+bs.length).join('.')===key){cnt.set(c.ch,(cnt.get(c.ch)||0)+1); pos.add(o===0?'s':(o+bs.length===c.n?'e':'m')); break;}}
  return {chars:[...cnt.entries()].sort((a,b)=>b[1]-a[1]).map(x=>x[0]),pos:pos.size===1?[...pos][0]:'m'};}
const POSW={s:{en:'start',zh:'开头'},e:{en:'end',zh:'结尾'},m:{en:'middle',zh:'中间'}};
/* a token on its own (the first merges, a merge's new token): when its bytes sit inside several different characters, it names how many */
function hexWhere(bs,at){
  if(at!=null) return at<0?null:zhSpan(at,bs.length);   // a part placed inside a known place: that place's characters
  const i=findZh(bs); if(i<0) return null;
  const sp=zhSpan(i,bs.length), cs=ZH_CH.filter(c=>c.b0+c.n>i&&c.b0<i+bs.length);
  if(cs.length===1&&cs[0].n>bs.length){const sh=zhShared(bs); if(sh.chars.length>1){const n=sh.chars.length, ex=sh.chars, w=POSW[sh.pos];
    return {en:`${sh.pos==='m'?'inside':w.en+' of'} ${n} characters, such as “${ex[0]}” and “${ex[1]}”`,
            zh:`${n} 个字符的${w.zh}，如“${ex[0]}”和“${ex[1]}”`, shared:ex, pos:sh.pos};}}
  return sp;}
const shList=w=>andList(w.shared.map(c=>`“${c}”`));
const sharedChip=(t,w)=>tx(`Bytes ${t} are the ${POSW[w.pos].en} of ${w.shared.length} different characters in the training text’s Chinese paragraph: ${shList(w)}.`,
  `字节 ${t} 是训练文本那段中文里 ${w.shared.length} 个不同字符的${POSW[w.pos].zh}：${shList(w)}。`);
const sharedTip=w=>w&&w.shared?tx(` The new token’s bytes are the ${POSW[w.pos].en} of ${w.shared.length} different characters in the Chinese paragraph (${shList(w)}), which is why the pair is frequent.`,
  `新 token 的这些字节是那段中文里 ${w.shared.length} 个不同字符的${POSW[w.pos].zh}（${shList(w)}），所以这一对出现得多。`):'';

/* =====================================================================
   formatting
   ===================================================================== */
const pct0=x=>fint(Math.round(x*100))+'%';
const r2=x=>fixd(x,2);
const r1=x=>fixd(x,1);
/* a quotient rounded to one decimal (R6.6, review N1); more decimals only when one would read as a whole number (x.0) */
const q1=x=>{for(let d=1;d<=4;d++){const s=x.toFixed(d); if(!/\.0+$/.test(s)) return fixd(+s,d);} return fixd(x,4);};
const tokN=n=>tx(n===1?'1 token':`${fint(n)} tokens`,`${fint(n)} 个 token`);
const mergeN=n=>tx(n===1?'1 merge':`${fint(n)} merges`,`${fint(n)} 次合并`);
const hex=b=>b.toString(16).padStart(2,'0');
/* a token as text: the string it decodes to (a space shown as “·”), or its bytes in hex when it is part of a character */
function piece(bytes){try{return {t:TD.decode(new Uint8Array(bytes)).replace(/ /g,'·'),ok:true};}catch(e){return {t:bytes.map(hex).join(' '),ok:false};}}
const pieceHTML=bs=>{const p=piece(bs); return p.ok?`<span class="tk-pc">${esc(p.t)}</span>`:`<span class="tk-pc tk-hx">${esc(p.t)}</span>`;};
const q=s=>tx(`“${s}”`,`“${s}”`);
const mixName=id=>L(MIX[id].t);
const mixPh=id=>L(MIX[id].p);
const shareS=id=>pct0(model(id).share);
const cite=(url,t,tip)=>`<a href="${url}" target="_blank" rel="noopener" data-tipx="${esc(tip||tx('Opens the source in a new tab.','在新标签页打开出处。'))}">${t}</a>`;
const J=(k,t,tip)=>`<button type="button" class="linkbtn" data-jump="${k}"${tip?` data-tipx="${esc(tip)}"`:''}>${t}</button>`;
const T_=(k,x)=>`<span class="t" tabindex="0" role="button" data-ctl="${k}">${x}</span>`;
const DEF=(text,tip)=>A.ui.def?A.ui.def(text,tip):`<span class="def" tabindex="0" data-tipx="${esc(tip)}">${text}</span>`;
const kt=k=>(A.terms&&A.terms[k]?A.terms[k]():'');
const labLink=id=>A.shell.labLink?A.shell.labLink(id):'';
const big=n=>A.ui.big?A.ui.big(n):fint(n);

/* =====================================================================
   module state (survives re-mounts and language switches)
   ===================================================================== */
let S={...BASE}, mode='guided', step=0, pred=null, observed={}, root=null, app=null, hexOn=false;
let saved=null, chSubmitted=false, chReveal=false, drillSeed=1, drillOpen=false, annT=null, userAct=false, wantObs=false, sbTouched=false;
let lvTried=new Set(), remView=null;
const shownPrev={};
const pairId=()=>mode==='challenge'?'ch':'main';
const hexShown=()=>hexOn&&mode==='guided'&&STEPS[step]&&STEPS[step].id==='bytes';
const R=()=>compute(S.mix,S.k,pairId());
const R0=()=>compute('tile',100,'main');      // the mission: the initial setup every part compares with

/* =====================================================================
   definitions shown on hover (R2.8): a term is defined where a reader first meets it
   ===================================================================== */
const TIP={
 tokenizer:()=>tx('A tokenizer cuts text into tokens and gives each token an ID, its row in the embedding table. It is trained once, on its own training text, before the model is; the model only ever sees the IDs.',
                  'tokenizer 把文字切成 token，并给每个 token 一个编号，也就是它在 embedding 表里的行号。它在训练模型之前，用自己的训练文本单独训练一次；模型只看得到这些编号。'),
 bpe:()=>tx('Byte-level BPE (byte-pair encoding) starts with the 256 possible byte values as its tokens. It counts every pair of neighbouring tokens in its training text, joins the most frequent pair into one new token, and repeats. GPT-2 introduced this byte-level version.',
            'byte-level BPE（byte-pair encoding，字节对编码）一开始只有 256 个 token，也就是 256 种可能的字节值。它统计训练文本里每一对相邻 token 出现的次数，把出现最多的一对合并成一个新 token，然后重复。GPT-2 首先用了这种按字节的版本。'),
 merge:()=>tx('A merge is one step of BPE training: the most frequent pair of neighbouring tokens, such as “t” and “he”, becomes one new token, “the”. A new text is encoded by applying the learned merges in the order they were learned.',
              '一次合并就是 BPE 训练的一步：出现最多的一对相邻 token（比如“t”和“he”）变成一个新 token“the”。给一段新文字编码时，按学到的顺序依次套用这些合并。'),
 vocab:()=>tx('The vocabulary is every token a tokenizer can produce: the 256 byte values plus one new token per merge. Each of them needs its own row in the embedding table and in the LM head.',
              '词表是 tokenizer 能产出的全部 token：256 个字节值，加上每次合并新造的一个 token。每个 token 在 embedding 表和 LM head 里都要占自己的一行。'),
 utf8:()=>tx('UTF-8 is the standard way to store text as bytes: 1 byte for an English letter, a digit, a space or a full stop; 2 bytes for most accented Latin, Greek or Cyrillic letters; 3 bytes for a common Chinese character or a Chinese full stop; 4 bytes for most emoji.',
             'UTF-8 是把文字存成字节的标准方式：英文字母、数字、空格和英文句号各 1 个字节；大多数带重音的拉丁字母、希腊字母和西里尔字母各 2 个字节；一个常用汉字或中文句号 3 个字节；大多数 emoji 4 个字节。'),
 byte:()=>tx('A byte is a whole number from 0 to 255, written here in hexadecimal, from 00 to ff. Every text is stored as bytes, so a byte-level tokenizer can encode any text, even one unlike anything it was trained on.',
             '字节是 0 到 255 之间的整数，这里用十六进制写，从 00 到 ff。任何文字都存成字节，所以按字节的 tokenizer 能给任何文字编码，哪怕它和训练文本完全不像。'),
 space:()=>tx('“·” marks a space. The space before a word belongs to that word’s token, as in GPT-2: “·like” is a space followed by “like”.',
              '“·”表示一个空格。和 GPT-2 一样，一个词前面的空格属于这个词的 token：“·like”就是一个空格加“like”。'),
 ctx:()=>tx('A context window is the most tokens a model can take in at once, prompt and answer together. Llama-3.1-8B’s is 131,072 tokens (128K), the max_position_embeddings in its config.json.',
            '上下文窗口是模型一次能读入的最多 token 数，prompt 和回答合在一起算。Llama-3.1-8B 的上下文窗口是 131,072 个 token（128K），即它 config.json 里的 max_position_embeddings。'),
 dstep:()=>tx('A decode step is one pass of the model that writes one new token of the answer, so every token of an answer takes its own step.',
              'decode 步是把模型跑一遍、给回答多写一个 token 的过程，所以回答的每个 token 都要一个自己的 decode 步。'),
 mixes:()=>tx('The training text is the text a tokenizer learns its merges from, before the model is trained. This lab has one short Chinese paragraph; to give Chinese a larger share, it counts that paragraph 16 or 48 times, as if the text held that many copies. Real tokenizers train on far more, and more varied, text.',
              '训练文本是 tokenizer 学合并用的文本，在训练模型之前就用完了。这个 Lab 只有一小段中文；为了让中文占更大的比例，Lab 把这一段按 16 份或 48 份计，就像文本里有这么多份拷贝。真实的 tokenizer 用的训练文本多得多，也丰富得多。')
};

/* =====================================================================
   guided parts
   ===================================================================== */
const PRED=[{v:'9',t:{en:'9: one per character',zh:'9 个：每个字符一个'}},
  {v:'18',t:{en:'18: two per character',zh:'18 个：每个字符两个'}},
  {v:'27',t:{en:'27: three per character',zh:'27 个：每个字符三个'}},
  {v:'36',t:{en:'36: as many as the English one',zh:'36 个：和英文一样多'}}];
const PRED_RIGHT='27';
const predT=v=>L(PRED.find(o=>o.v===v).t);
/* parts 1 and 2 (R3.8, review A1): the Chinese sentence's bytes and its tokens with no merges are hidden until part 3 checks the
   prediction, even after a pick (it can still change); Settings stay locked there */
const preLock=()=>mode==='guided'&&step<=1;
const predHidden=()=>mode==='guided'&&step<=1;
const zhUnknown=()=>mode==='guided'&&step===1;     // part 2 asks for the Chinese count with no merges: it shows as “?”
/* the merges slider, with one-merge steps (R1.11: pressed again and again, they neither scroll nor pop a tooltip) */
function kCtl(id,inline){
  return `<div class="${inline?'inline-ctl':'ctl'} tk-kctl" id="${id}"><label for="${id}-in"><span>${esc(tx('Merges learned','已学的合并次数'))}</span> <output></output></label>`+
    `<div class="tk-krow"><button type="button" class="btn ghost sm tk-kstep" data-act="kminus" aria-label="${esc(tx('One merge fewer','少学一次合并'))}">◂</button>`+
    `<input type="range" id="${id}-in" data-key="k" min="0" max="${K_MAX}" step="1" value="${S.k}">`+
    `<button type="button" class="btn ghost sm tk-kstep" data-act="kplus" aria-label="${esc(tx('One more merge','多学一次合并'))}">▸</button></div>`+
    `<div class="ctl-note tk-knote"></div></div>`;
}
const mixSeg=(id,inline,note)=>A.ui.seg({id,key:'mix',label:tx('Training text','训练文本'),options:MIXES.map(m=>[m.id,mixName(m.id)]),value:S.mix,inline,note});
const hexSeg=id=>A.ui.seg({id,key:'hex',label:tx('Byte values under the cells','格子下面的字节值'),options:[['0',tx('hidden','不显示')],['1',tx('shown','显示')]],value:hexOn?'1':'0',inline:true});
/* part 5: the four training texts, each trained from scratch with the same 100 merges (R3.6: the initial setup first; an option's result
   shows once it has been tried, so the options fill in as a comparison table, review A23) */
const LV_ORDER=['tile','en','bal','zh'];
const lvRes=id=>(id==='tile'||lvTried.has(id))?r2(compute(id,100,'main').ratio)+'×':'?';
const lvDesc=id=>tx(`${shareS(id)} English: ${L(MIX[id].d)}`,`${shareS(id)} 是英文：${L(MIX[id].d)}`)+(id==='tile'?tx(' (initial setup)','（初始设置）'):'');
const mixLevers=()=>`<div class="tk-lv"><div class="tk-lv-h" aria-hidden="true"><span>${tx('training text','训练文本')}</span><span>${tx('Chinese ÷ English','中文 ÷ 英文')}</span></div>`+
  A.ui.levers({aria:tx('The training text, pick one','训练文本，选一个'),value:S.mix,options:LV_ORDER.map(id=>({v:id,title:mixName(id),desc:lvDesc(id),result:esc(lvRes(id))}))})+`</div>`;
/* the first merges of a mix, as text (“·t”, “he”, “·a”), from the engine */
const firstPieces=(id,n)=>{const m=model(id); return m.merges.slice(0,n).map((_,i)=>q(piece(m.bytes[256+i]).t));};
const andList=a=>a.length<2?a.join(''):tx(a.slice(0,-1).join(', ')+' and '+a[a.length-1],a.join('、'));
/* a change of a token count in words (R6.20): the counts themselves are on the strips and the metric cards */
function chgW(a,b){
  if(b===a) return {en:'stays the same',zh:'没有变'};
  if(b<a){const x=b/a;
    if(x>=0.9) return {en:'barely shrinks',zh:'几乎没变短'};
    if(x>=0.8) return {en:'shrinks a little',zh:'短了一点'};
    if(x>=0.71) return {en:'shrinks by about a quarter',zh:'短了约四分之一'};
    if(x>=0.6) return {en:'shrinks by about a third',zh:'短了约三分之一'};
    if(x>=0.55) return {en:'shrinks by more than a third',zh:'短了三分之一以上'};
    if(x>=0.4) return {en:'shrinks by about half',zh:'短了约一半'};
    if(x>=0.3) return {en:'shrinks to about a third',zh:'缩到约三分之一'};
    if(x>=0.2) return {en:'shrinks to under a third',zh:'缩到不到三分之一'};
    return {en:'shrinks to under a fifth',zh:'缩到不到五分之一'};}
  const y=b/a;
  if(y<1.15) return {en:'grows a little',zh:'长了一点'};
  if(y<1.4) return {en:'grows by about a quarter',zh:'长了约四分之一'};
  if(y<1.7) return {en:'grows by about half',zh:'长了约一半'};
  if(y<2.3) return {en:'nearly doubles',zh:'长到约两倍'};
  if(y<3.3) return {en:'nearly triples',zh:'长到约三倍'};
  return {en:'more than triples',zh:'长到三倍以上'};}
const chg=(a,b)=>L(chgW(a,b));
const STEPS=[
 {id:'mission',title:{en:'Your mission',zh:'你的任务'},focus:'strips',
  lead:()=>{const b=R0(); return tx(`Same meaning: English ${b.en} tokens, Chinese ${b.zh}. Why?`,`意思一样：英文 ${b.en} 个 token，中文 ${b.zh} 个。为什么？`);},
  enter:()=>({...BASE}),
  body:()=>{const b=R0();
   return tx(`<p>A ${DEF('tokenizer',TIP.tokenizer())} cuts text into ${DEF('tokens',kt('token'))} before the model reads it. This one is a small ${DEF('byte-level BPE',TIP.bpe())}, trained in your browser on mostly English text. After ${b.k} ${DEF('merges',TIP.merge())}, it turns “${PAIRS.main.en}” into ${b.en} tokens, the boxes in ${J('strips','the chart')}.</p><p>“${PAIRS.main.zh}” means the same, yet it takes ${b.zh} tokens. The model reads every token, keeps it in its context window and writes it, so the Chinese message costs more than twice as much. Why?</p>`,
             `<p>${DEF('tokenizer',TIP.tokenizer())} 在模型读文字之前，先把文字切成 ${DEF('token',kt('token'))}。这里是一个小的 ${DEF('byte-level BPE',TIP.bpe())}，就在你的浏览器里、用以英文为主的文本训练。学了 ${b.k} 次${DEF('合并',TIP.merge())}之后，它把“${PAIRS.main.en}”切成 ${b.en} 个 token，也就是${J('strips','图里')}的方框。</p><p>“${PAIRS.main.zh}”意思一样，却要 ${b.zh} 个 token。模型要读入每一个 token，把它留在上下文窗口里，还要把它写出来，所以同一句话用中文要花两倍多的成本。为什么？</p>`);}},
 {id:'predict',title:{en:'Predict first',zh:'先预测'},focus:'strips',gate:()=>!!pred,skip:true,
  lead:()=>tx('Before any merge: how many tokens does the Chinese sentence take?','还没有任何合并时，中文句子是多少个 token？'),
  enter:()=>({mix:'tile',k:0}),
  body:()=>{const b=R0(), ch=`<div class="choices" role="group" aria-label="${esc(tx('Your prediction','你的预测'))}">${PRED.map(o=>`<button type="button" class="choice" data-pred="${o.v}" aria-pressed="${pred===o.v}">${esc(L(o.t))}</button>`).join('')}</div>`;
   return tx(`<p>Go back to the very beginning. Before the tokenizer learns any merge, the English sentence takes ${b.enBytes} tokens: one for each letter, space and full stop.</p><p>The Chinese sentence has ${b.zhChars} characters, its full stop included. With no merges, how many tokens does it take?</p>${ch}${pred?'<p class="muted small">Saved. You can still change it before part 3 shows the result.</p>':''}`,
             `<p>先退回到最开始。tokenizer 还没学任何合并时，英文句子是 ${b.enBytes} 个 token：每个字母、空格和句号各一个。</p><p>中文句子有 ${b.zhChars} 个字符（含句号）。没有任何合并时，它是多少个 token？</p>${ch}${pred?'<p class="muted small">已记下。第 3 部分给出结果之前，还可以改。</p>':''}`);}},
 {id:'bytes',title:{en:'Bytes: where every tokenizer starts',zh:'字节：tokenizer 的起步'},focus:'strips',
  lead:()=>tx('With no merges, every byte is a token.','没有合并时，每个字节就是一个 token。'),
  enter:()=>({mix:'tile',k:0}),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Before any merge, the tokenizer sees only the text’s ${DEF('UTF-8',TIP.utf8())} ${DEF('bytes',TIP.byte())}, and each byte is one token. The chart draws one cell per byte, on one scale for both sentences, and compares this with starting from whole characters.</p>`,
              `<p>还没有合并时，tokenizer 只看得到文字的 ${DEF('UTF-8',TIP.utf8())} ${DEF('字节',TIP.byte())}，每个字节就是一个 token。图里每个字节画成一格，两句话用同一个比例尺，下面还和“从整个字符起步”做了对比。</p>`)+hexSeg('tk-ic-hex'),
  obs:()=>obsBytes()},
 {id:'merges',title:{en:'Merges: frequent pairs become tokens',zh:'合并：常见的一对变成一个 token'},focus:'merges',always:true,
  lead:()=>tx('Each merge joins the pair seen most often in the training text.','每次合并，都把训练文本里出现最多的一对合成一个。'),
  enter:()=>({mix:'tile',k:0}),expect:()=>S.k>=20,showMe:{k:100},label:{en:'Observed',zh:'观察到'},
  body:()=>tx(`<p>BPE learns from its training text: it counts every pair of neighbouring tokens, merges the most frequent pair into one new token, and repeats. Each merge adds one token to the ${DEF('vocabulary',TIP.vocab())}.</p><p>Drag to at least 20 merges and watch which strip shrinks:</p>`,
              `<p>BPE 从训练文本里学：统计每一对相邻 token 出现的次数，把出现最多的一对合并成一个新 token，然后重复。每次合并都给${DEF('词表',TIP.vocab())}添一个 token。</p><p>至少拖到 20 次合并，看哪一条变短：</p>`)+kCtl('tk-ic-k',true),
  obs:()=>obsMerges()},
 {id:'mix',title:{en:'The training text decides',zh:'训练文本说了算'},focus:'merges',
  lead:()=>tx('Put Chinese in the training text: the Chinese sentence shrinks.','训练文本里多放中文，中文句子就变短。'),
  enter:()=>({mix:'tile',k:100}),expect:()=>S.mix!=='tile',always:true,showMe:{mix:'bal'},label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Each option trains a new tokenizer, with the same 100 merges, on a different ${DEF('training text',TIP.mixes())} built from the same paragraphs.</p><p>Pick a training text and see how the Chinese sentence’s token count compares with the English sentence’s.</p>`,
              `<p>每个选项都用同样的 100 次合并，在不同的${DEF('训练文本',TIP.mixes())}上重新训练一个 tokenizer，文本都来自同样的段落。</p><p>选一种训练文本，看看中文句子的 token 数是英文的几倍。</p>`)+mixLevers(),
  obs:()=>obsMix()},
 {id:'cost',title:{en:'What the extra tokens cost',zh:'多出来的 token 花在哪'},focus:'cost',
  lead:()=>tx('The more tokens a message takes, the more it costs, the more of the context window it fills, and the more decode steps it takes to write.','一条消息的 token 越多，花的钱越多，占的上下文窗口越多，写出它要的 decode 步数也越多。'),
  enter:()=>({mix:'tile',k:100}),label:{en:'Result',zh:'结果'},
  body:()=>tx(`<p>Everything is counted in tokens: an API bills per token, a ${DEF('context window',TIP.ctx())} holds a fixed number of them, and the model writes one token per ${DEF('decode step',TIP.dstep())} (${labLink('prefill-decode')}). So a message with more tokens costs more of all three.</p><p>Real tokenizers show the same gap. ${cite(SRC.petrov,'Petrov et al. (NeurIPS 2023)',tx('Opens the paper on arXiv in a new tab: Table 1 for GPT-2, Table 4 for XLM-R.',''))} measured Chinese at about 3.2 times the English tokens on GPT-2’s tokenizer, an English-centric one, and about as many as English on multilingual tokenizers such as XLM-R’s.</p><p>The largest gap they found, 15 times, is for Shan, a language of Myanmar. Compare the training texts:</p>`,
              `<p>一切都按 token 计：API 按 token 收费，${DEF('上下文窗口',TIP.ctx())}只装得下固定数量的 token，模型每个 ${DEF('decode 步',TIP.dstep())}写一个 token（${labLink('prefill-decode')}）。所以 token 多的消息，这三样都花得多。</p><p>真实的 tokenizer 也有这个差距。${cite(SRC.petrov,'Petrov 等（NeurIPS 2023）','在新标签页打开 arXiv 上的论文：GPT-2 见表 1，XLM-R 见表 4。')}测得，在 GPT-2 这种以英文为主的 tokenizer 上，中文要用约 3.2 倍于英文的 token；在 XLM-R 这类多语言 tokenizer 上，则和英文差不多。</p><p>他们发现的最大差距是 15 倍，出现在缅甸的掸语（Shan）上。比较几种训练文本：</p>`)+mixSeg('tk-ic-mix',true),
  obs:()=>obsCost()},
 {id:'remember',title:{en:'What to remember',zh:'要记住的'},focus:'',
  lead:()=>tx('Count bytes first, check the training text, and count tokens on real messages.','先数字节，再看训练文本，最后用真实消息数 token。'),
  enter:()=>({mix:'tile',k:0}),
  body:()=>remember()}
];
const leadOf=s=>typeof s.lead==='function'?s.lead():L(s.lead);

/* the verdict on the prediction (R3.8: the right answer too, when the pick was wrong) */
function verdict(){
  if(!pred) return '';
  const pt=esc(predT(pred)), rt=esc(predT(PRED_RIGHT));
  return pred===PRED_RIGHT?tx(`<p><span class="okc">Your prediction, “${pt}”, was right.</span></p>`,`<p><span class="okc">你的预测“${pt}”是对的。</span></p>`)
    :tx(`<p>You predicted “${pt}”. The right answer is “${rt}”.</p>`,`<p>你预测的是“${pt}”，正确答案是“${rt}”。</p>`);
}
/* part 3: result, verdict, why (R3.10); one key number per paragraph (R6.20) */
function obsBytes(){
  const r=compute(S.mix,0,'main');
  return tx(`<p><b>No merges: the English sentence takes ${r.en} tokens and the Chinese one ${r.zh}, fewer than the English.</b></p>`,
            `<p><b>没有合并时：英文句子 ${r.en} 个 token，中文句子 ${r.zh} 个，比英文还少。</b></p>`)+verdict()+
    tx(`<p>Why: UTF-8 stores an English letter, a space or a full stop in one byte, and a Chinese character in three.</p><p>The same meaning takes far fewer characters in Chinese than in English, so even at three bytes each, the Chinese sentence is the shorter one.</p>`,
       `<p>原因：UTF-8 用 1 个字节存一个英文字母、空格或句号，用 3 个字节存一个汉字。</p><p>同样的意思，中文用的字符比英文少得多，所以即使每个汉字占三个字节，中文句子还是更短。</p>`);
}
/* part 4: the learner's own number of merges; the trend in words, the counts on the strips (R6.20) */
function obsMerges(){
  const r=R(), m=model(S.mix);
  const more=r.k<20?tx('<p class="small">Learn at least 20 merges to go on.</p>','<p class="small">至少学 20 次合并，才能进入下一部分。</p>'):'';
  if(r.k===0) return tx(`<p><b>No merges yet: every byte is a token.</b></p>`,`<p><b>还没有合并：每个字节都是一个 token。</b></p>`)+more;
  const head=tx(`<p><b>The English sentence shrinks from ${r.enBytes} tokens to ${r.en}, while the Chinese one ${r.zh===r.zhBytes?'does not shrink at all':chg(r.zhBytes,r.zh)}.</b></p>`,
                `<p><b>英文句子从 ${r.enBytes} 个 token 缩到 ${r.en} 个，中文句子却${r.zh===r.zhBytes?'一点没变':chg(r.zhBytes,r.zh)}。</b></p>`);
  const english=m.share>0.9, fp=andList(firstPieces(S.mix,3)), few=r.zhMerges*20<=r.k;
  const why=english?tx(`<p>Why: the merges come from the training text, which is almost all English. The first three merges, listed under the ticks, join ${fp}, parts of English words. Chinese bytes are rare there, so ${r.zhMerges===0?'none of the merges joins them':(few?'almost none of the merges join them (the few tall ticks)':'only some of the merges join them (the tall ticks)')}.</p>`,
                       `<p>原因：合并来自训练文本，而它几乎全是英文。前三次合并（列在刻度下面）拼出的是${fp}，都是英文单词的一部分。中文字节在那里很少见，所以${r.zhMerges===0?'没有一次合并用到它们':(few?'几乎没有合并用到它们（图里零星的高刻度）':'只有一部分合并用到它们（图里的高刻度）')}。</p>`)
    :tx(`<p>Why: this training text has much more Chinese, so many of the merges join Chinese bytes (the tall ticks), and the Chinese sentence shrinks as well.</p>`,
        `<p>原因：这份训练文本里的中文多得多，很多合并都用在中文字节上（图里的高刻度），所以中文句子也变短了。</p>`);
  const now=r.zh>r.en&&english?tx(`<p>Now the Chinese sentence takes ${r2(r.ratio)} times the English tokens.</p>`,`<p>现在中文句子的 token 数是英文的 ${r2(r.ratio)} 倍。</p>`):'';
  return head+why+now+more;
}
/* part 5: each training text against the initial setup, in words (R3.6, R6.20) */
function obsMix(){
  const r=R(), b=compute('tile',r.k,'main');
  if(S.mix==='tile') return tx(`<p><b>Mostly English, the initial setup: the Chinese sentence takes ${r2(r.ratio)} times the English tokens.</b> Pick another training text.</p>`,
                               `<p><b>英文为主，也就是初始设置：中文句子的 token 数是英文的 ${r2(r.ratio)} 倍。</b>选另一种训练文本试试。</p>`);
  if(S.mix==='en') return tx(`<p><b>English only: the Chinese sentence goes back to one token per byte, and the English one ${chg(b.en,r.en)}.</b></p><p>Why: this text has no Chinese at all, so no merge joins a Chinese byte, however many merges the tokenizer learns.</p>`,
                             `<p><b>只有英文：中文句子又回到一个字节一个 token，英文句子${chg(b.en,r.en)}。</b></p><p>原因：这份文本里完全没有中文，学多少次合并，都不会有一次用到中文字节。</p>`);
  return tx(`<p><b>${mixName(S.mix)}: the Chinese sentence ${chg(b.zh,r.zh)}, but the English one ${chg(b.en,r.en)}.</b></p><p>Why: BPE merges whatever is most frequent in its training text. With more Chinese, ${fint(r.zhMerges)} of the ${fint(r.k)} merges join Chinese bytes (the tall ticks), so fewer are left for English, and the English sentence grows.</p><p>The merges are one budget that every language shares.</p>`,
            `<p><b>${mixName(S.mix)}：中文句子${chg(b.zh,r.zh)}，英文句子却${chg(b.en,r.en)}。</b></p><p>原因：BPE 合并的是训练文本里出现最多的那一对。中文多了，${fint(r.k)} 次合并里有 ${fint(r.zhMerges)} 次用在中文字节上（图里的高刻度），留给英文的就少了，英文句子因此变长。</p><p>所有语言共用同一份合并预算。</p>`);
}
/* part 6: the cost of the extra tokens; one number, the rest in the chart (R6.20) */
function obsCost(){
  const r=R();
  const head=tx(`<p><b>${mergeN(r.k)} on ${mixPh(S.mix)}: the Chinese message takes ${r2(r.ratio)} times the English tokens.</b></p>`,
                `<p><b>在${mixPh(S.mix)}上学 ${fint(r.k)} 次合并：中文消息的 token 数是英文的 ${r2(r.ratio)} 倍。</b></p>`);
  const why=r.ratio>1.05?tx(`<p>So, next to the English message, the Chinese one costs that many times as much: in money, in room in the context window, and in decode steps to write it.</p>`,`<p>所以和英文消息相比，中文消息的费用、占用的上下文窗口、写出它要的 decode 步数，都是这么多倍。</p>`)
    :(r.ratio<0.95?tx(`<p>Here the Chinese message is the cheaper one: with this training text, the tokenizer compresses Chinese better than English.</p>`,`<p>这里中文消息反而更便宜：用这份训练文本，tokenizer 压缩中文比压缩英文更好。</p>`)
      :tx(`<p>Here the two messages cost about the same.</p>`,`<p>这里两条消息的成本差不多。</p>`));
  return head+why;
}
/* part 7: each point says what to do, with this lab's example and a button that puts that part's picture in the chart (R6.23, review A8) */
const REM={bytes:{s:{mix:'tile',k:0},focus:'strips'},mix:{s:{mix:'bal',k:100},focus:'merges'},cost:{s:{mix:'tile',k:100},focus:'cost'}};
function remBtn(k){const on=remView===k;
  const tip={bytes:tx('Puts the strips with no merges back in the chart: one token per byte.','在图里重新显示没有合并时的长条：一个字节一个 token。'),
    mix:tx('Shows the half-and-half training text with 100 merges: the tall ticks are the merges of Chinese bytes.','显示中英各半的训练文本学 100 次合并的结果：高刻度就是合并中文字节的那些。'),
    cost:tx('Shows the cost rows for the initial setup: the same message in both languages.','显示初始设置下的成本条：同一句话的两种语言。')}[k];
  return ` <button type="button" class="linkbtn tk-rem" data-rem="${k}" aria-pressed="${on}" data-tipx="${esc(tip)}">${on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看')}</button>`;}
function remember(){
  const b=R0(), bal=compute('bal',100,'main'), e=enc('tile',PAIRS.main.en), m=model('tile');
  const toks=e.at[100].map(id=>piece(m.bytes[id]).t), bookI=toks.findIndex(t=>/book/.test(t)), bookS=bookI>=0?toks.slice(bookI,bookI+2):[];
  const balChars=(()=>{const z=enc('bal',PAIRS.main.zh), mm=model('bal'); return z.at[100].map(id=>piece(mm.bytes[id])).filter(p=>p.ok&&Array.from(p.t).length===1&&/[一-鿿。]/.test(p.t)).map(p=>p.t);})();
  const kWo=Math.round(wholeAt('tile',PAIRS.main.zh,'我')/100)*100, H=4096, V3=128256, emb=2*V3*H, total=8030261248;
  const embD=DEF(tx(`about ${A.fmt.big(emb)} parameters`,`约 ${A.fmt.big(emb)} 个参数`),tx(`2 × 128,256 × 4,096 = ${fint(emb)}: 128,256 rows of 4,096 numbers in each of the two tables.`,`2 × 128,256 × 4,096 = ${fint(emb)}：两张表各有 128,256 行，每行 4,096 维。`));
  const cq=balChars.map(c=>`“${c}”`);
  return tx(`<ul class="tk-remlist"><li><b>Count the bytes first.</b> Each Chinese character is three bytes in UTF-8 and each English letter one. With no merges every byte is a token, so a sentence’s token count starts from its byte count.${remBtn('bytes')}</li>`+
    `<li><b>Check what the tokenizer was trained on.</b> Merges only build tokens for pairs that are frequent in the training text. So a language that is rare there stays close to one token per byte, and giving one language more of the text leaves fewer merges for the others.${remBtn('mix')}</li>`+
    `<li><b>Count tokens on your users’ own messages before you choose a tokenizer.</b> The same message can take several times as many tokens in one language as in another, and then it costs that many times the bill, the room in the context window and the decode steps.${remBtn('cost')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">Common misconceptions</div><p>✗ “One Chinese character is one token.” With no merges, each character is three tokens, one per byte. It becomes one token only when the training text has enough Chinese for its bytes to be merged. On mostly English text, even “我” needs about ${kWo} merges.</p><p>✗ “A token is a word.” Here “${bookS.join('” and “')}” are two tokens of one word, and a Chinese character can be split into three byte tokens.</p><p>✗ “A bigger vocabulary is free.” Every token needs a row of d numbers (d is the ${DEF('hidden size',kt('hidden'))}) in the ${DEF('embedding table',kt('embedding'))} and another in the ${DEF('LM head',kt('lmhead'))}. Llama 3’s vocabulary takes ${embD} in the two tables, ${pct0(emb/total)} of Llama-3.1-8B (${labLink('transformer')}).</p></div>${A.shell.toScenarios()}`,
    `<ul class="tk-remlist"><li><b>先数字节。</b>在 UTF-8 里，每个汉字是三个字节，每个英文字母是一个。没有合并时，一个字节就是一个 token，所以一句话的 token 数从它的字节数算起。${remBtn('bytes')}</li>`+
    `<li><b>再看 tokenizer 是用什么文本训练的。</b>只有在训练文本里常见的一对，才会被合并成 token。所以训练文本里少见的语言，几乎还是一个字节一个 token。给一种语言多放文本，留给其他语言的合并就少了。${remBtn('mix')}</li>`+
    `<li><b>选 tokenizer 之前，先用你用户自己的消息数一数 token。</b>同一句话换一种语言，token 数可能差好几倍，账单、占用的上下文窗口和 decode 步数也跟着差这么多倍。${remBtn('cost')}</li></ul>`+
    `<div class="obs"><div class="eyebrow">常见误解</div><p>✗ “一个汉字就是一个 token。”没有合并时，每个汉字是三个 token，一个字节一个。只有训练文本里中文够多，它的字节被合并起来，一个汉字才会变成一个 token。在以英文为主的文本上，连“我”都要学约 ${kWo} 次合并才会变成一个 token。</p><p>✗ “一个 token 就是一个词。”这里“${bookS.join('”和“')}”是同一个词的两个 token，一个汉字也可能被拆成三个字节 token。</p><p>✗ “词表大一点没有代价。”每个 token 在 ${DEF('embedding 表',kt('embedding'))}里要占一行 d 维的向量（d 是${DEF('隐藏维度',kt('hidden'))}），在 ${DEF('LM head',kt('lmhead'))} 里还要再占一行。Llama 3 的词表在这两张表里要占${embD}，是 Llama-3.1-8B 的 ${pct0(emb/total)}（${labLink('transformer')}）。</p></div>${A.shell.toScenarios()}`);
}

/* =====================================================================
   scenarios: four kinds of products, one plain line each, all with 300 merges (R2.1)
   ===================================================================== */
const PRESETS=[
 {id:'en',t:{en:'English-only assistant',zh:'只服务英文的助手'},d:{en:'Its tokenizer never saw Chinese.',zh:'它的 tokenizer 从没见过中文。'},s:{mix:'en',k:SCEN_K}},
 {id:'tile',t:{en:'English first, a little Chinese',zh:'英文为主，带一点中文'},d:{en:'Mostly English text with one Chinese paragraph.',zh:'以英文为主的文本，加一段中文。'},s:{mix:'tile',k:SCEN_K}},
 {id:'bal',t:{en:'Bilingual assistant',zh:'中英双语助手'},d:{en:'Half English, half Chinese.',zh:'一半英文，一半中文。'},s:{mix:'bal',k:SCEN_K}},
 {id:'zh',t:{en:'Chinese-first assistant',zh:'以中文为主的助手'},d:{en:'Mostly Chinese, some English.',zh:'以中文为主，带一些英文。'},s:{mix:'zh',k:SCEN_K}}];
const presetNow=()=>PRESETS.findIndex(p=>p.s.mix===S.mix&&p.s.k===S.k);
function scenHTML(i){
  const p=PRESETS[i], r=compute(p.s.mix,p.s.k,'main'), g=compute(p.s.mix,100,'main');
  const head=tx(`<p><b>${mergeN(r.k)}: the Chinese sentence takes ${r2(r.ratio)} times the English tokens.</b></p>`,
                `<p><b>${fint(r.k)} 次合并：中文句子的 token 数是英文的 ${r2(r.ratio)} 倍。</b></p>`);
  const why={
   en:tx('<p>Its tokenizer never saw Chinese, so no merge joins Chinese bytes: every Chinese character stays three byte tokens, however many merges it learns.</p>','<p>它的 tokenizer 从没见过中文，没有一次合并会用到中文字节：每个汉字始终是三个字节 token，学多少合并都一样。</p>'),
   tile:tx(`<p>One Chinese paragraph is enough for a few Chinese merges: once the English words are whole tokens, common characters such as “我” become single tokens too.</p><p>At the guided parts’ 100 merges, this same text gave ${r2(g.ratio)} times.</p>`,`<p>一段中文就够学到几次中文的合并：等英文单词都成了完整的 token，“我”这样的常用字也会变成单个 token。</p><p>在分步引导的 100 次合并下，同样的文本是 ${r2(g.ratio)} 倍。</p>`),
   bal:tx('<p>Half the training text is Chinese, so many merges join Chinese bytes into characters and pairs of characters. The English sentence pays for it: fewer merges are left for English words.</p>','<p>训练文本一半是中文，很多合并把中文字节拼成汉字，甚至拼成两个字的词。代价落在英文句子上：留给英文单词的合并少了。</p>'),
   zh:tx('<p>Most merges go to Chinese: the Chinese sentence is short, and the English one needs many more merges before its words become whole tokens.</p>','<p>大多数合并都给了中文：中文句子很短，英文句子则要学更多合并，单词才会变成完整的 token。</p>')}[p.id];
  return head+why;
}

/* =====================================================================
   challenge
   ===================================================================== */
const chLabel=p=>tx(`${mixName(p.mix)}, ${mergeN(p.k)}: English ${p.en}, Chinese ${p.zh}`,`${mixName(p.mix)}，${fint(p.k)} 次合并：英文 ${p.en}，中文 ${p.zh}`);
function chResultHTML(){
  if(!chSubmitted) return '';
  const me=chPoint(S.mix,S.k), mk=chMinK(), best=chBest(), b0=best[0];
  let h;
  const met=tx(`<p class="okline" style="margin-top:12px"><b>Goal met: the English sentence takes ${me.en} tokens and the Chinese one ${r2(me.ratio)} times as many.</b></p>`,
               `<p class="okline" style="margin-top:12px"><b>达成目标：英文句子 ${me.en} 个 token，中文句子是它的 ${r2(me.ratio)} 倍。</b></p>`);
  if(me.ok&&me.k===mk) h=met+tx(`<p class="okline">And this is the fewest merges: at ${mergeN(CH.ks[0])}, no training text meets both targets.</p>`,`<p class="okline">而且这已经是最少的合并次数：只学 ${fint(CH.ks[0])} 次合并时，没有哪种训练文本能同时达到两个目标。</p>`);
  else if(me.ok) h=met+tx(`<p class="okline">But a smaller vocabulary also works: ${mixPh(b0.mix)} with ${mergeN(b0.k)} meets both targets.</p>`,`<p class="okline">不过词表还可以更小：${mixPh(b0.mix)}只学 ${fint(b0.k)} 次合并就能同时达到两个目标。</p>`);
  else {
    const fail=!me.okEn&&!me.okR?tx(`the English sentence takes ${me.en} tokens, and the Chinese one ${r2(me.ratio)} times as many`,`英文句子 ${me.en} 个 token，中文是英文的 ${r2(me.ratio)} 倍`)
      :(!me.okEn?tx(`the English sentence takes ${me.en} tokens, over 25`,`英文句子 ${me.en} 个 token，超过了 25`):tx(`the Chinese sentence takes ${r2(me.ratio)} times the English tokens, over 1.5`,`中文句子的 token 数是英文的 ${r2(me.ratio)} 倍，超过了 1.5`));
    const hint=!me.okR?tx('Hint: the Chinese sentence barely shrinks. A merge joins pairs that are frequent in the training text: what must that text contain for Chinese bytes to be merged?','提示：中文句子几乎没有变短。合并的是训练文本里常见的那一对：训练文本里要有什么，中文字节才会被合并？')
      :tx('Hint: the English sentence has too few merges of its own. The more Chinese in the training text, the more of the merges go to Chinese bytes. What else could leave English enough merges?','提示：英文句子分到的合并太少。训练文本里中文越多，越多的合并给了中文字节。还有什么办法能给英文留下足够的合并？');
    h=tx(`<p class="badline" style="margin-top:12px"><b>Not yet: ${fail}.</b></p><p class="small">${hint}</p>`,`<p class="badline" style="margin-top:12px"><b>还没达到：${fail}。</b></p><p class="small">${hint}</p>`)+
      (chReveal?'':`<button type="button" class="linkbtn" data-act="reveal">${tx('Show the answer','直接看答案')}</button>`);
  }
  if(!chReveal) return h;
  const pts=chAll(), mx=Math.max(2.5,...pts.map(p=>p.ratio))*1.05;
  /* the owner's review of batch 8 (2026-10-07): brief, so the chart keeps the options that meet both targets, the pick and the initial
     setup, and a note counts the rest. “English over 25” is a reason, not a tag that keeps a row, so only the pick and the initial
     setup carry it */
  const isMe=p=>p.mix===me.mix&&p.k===me.k, isStart=p=>p.mix===CH.start.mix&&p.k===CH.start.k;
  h+=A.ui.cmpChart({title:tx(`How many tokens does the Chinese sentence take for each English token? (Goal: at most 1.5, with the English sentence at most 25 tokens.)`,`英文每个 token 对应中文几个 token？（目标：最多 1.5 个，而且英文句子最多 25 个 token。）`),
    aria:tx('Chinese tokens ÷ English tokens: ','中文 token 数 ÷ 英文 token 数：')+pts.filter(p=>p.ok||isMe(p)||isStart(p)).map(p=>`${mixName(p.mix)} ${p.k}${tx(': ','：')}${r2(p.ratio)}`).join(tx('; ','；')),
    rows:pts.map(p=>({label:chLabel(p),value:p.ratio,text:r2(p.ratio),ok:p.ok,me:isMe(p),
      tags:(isStart(p)?[{text:tx('initial setup','初始设置')}]:[]).concat(!p.okEn&&(isMe(p)||isStart(p))?[{text:tx('English over 25','英文超过 25')}]:[]).concat(isMe(p)?[{text:tx('your pick','你的选择'),me:true}]:[])})),
    brief:true,goal:1.5,goalLabel:tx('goal: 1.5','目标 1.5'),max:mx,okText:tx('meets both targets','两个目标都达到'),noText:tx('misses at least one','至少差一个'),goalText:tx('goal: at most 1.5 times','目标：最多 1.5 倍')});
  /* initial setup → effect → reason (R5.5), in words (R6.20); the direction of the change comes from the counts (review B1) */
  const tr=chTrend();
  h+=tr.zhMore?tx(`<p class="small tk-trend" data-trend="zh">On mostly English text, the first merges build English words. Once the training text’s common English words are whole tokens, later merges reach the Chinese paragraph’s pairs: from the initial setup to the most merges offered, the Chinese sentence ${chg(tr.a.zh,tr.b.zh)}, more than the English one. Yet even then it takes more than 1.5 times the English tokens.</p>`,
                          `<p class="small tk-trend" data-trend="zh">在以英文为主的文本上，最先学到的合并都用来拼英文单词。等训练文本里常见的英文单词都成了完整的 token，后面的合并才轮到那段中文里的字节对：从初始设置学到最多的合并，中文句子${chg(tr.a.zh,tr.b.zh)}，比英文句子缩短得还多。但即使这样，它仍超过英文的 1.5 倍。</p>`)
    :tx(`<p class="small tk-trend" data-trend="en">On mostly English text, more merges keep building English words. From the initial setup to the most merges offered, the English sentence ${chg(tr.a.en,tr.b.en)}, more than the Chinese one, which still takes more than 1.5 times the English tokens.</p>`,
        `<p class="small tk-trend" data-trend="en">在以英文为主的文本上，更多的合并还是在拼英文单词。从初始设置学到最多的合并，英文句子${chg(tr.a.en,tr.b.en)}，比中文句子缩短得还多；中文句子仍超过英文的 1.5 倍。</p>`);
  h+=tx(`<p class="small">Half-and-half text meets both targets with the fewest merges, because Chinese gets merges of its own from the start. Mostly Chinese text gets there only with the most merges offered: before that, it leaves English too few merges.</p>`,
        `<p class="small">中英各半的文本用最少的合并就能达到两个目标，因为中文从一开始就分到了自己的合并。以中文为主的文本要学到最多的合并才行。在那之前，留给英文的合并太少。</p>`);
  if(sameMerges('bal','zh',CH.ks[CH.ks.length-1])) h+=tx(`<p class="small">At the most merges offered, half-and-half and mostly Chinese text end with the same counts: both have learned the same merges, only in a different order.</p>`,
    `<p class="small">学到最多的合并时，中英各半和以中文为主的文本结果一样：两者学到的合并完全相同，只是顺序不同。</p>`);
  h+=tx(`<p class="small">Why: a merge only helps text that looks like the training text, and the merges are one budget shared by every language.</p>`,
        `<p class="small">原因：合并只帮得上和训练文本相像的文字，而所有语言共用同一份合并预算。</p>`);
  h+=tx(`<p class="small muted">Note: this lab counts one Chinese paragraph many times to make its training texts, so its Chinese merges fit that paragraph’s words closely. A real tokenizer trains on far more varied text; before you choose one, count the tokens of your own users’ messages in each language.</p>`,
        `<p class="small muted">注意：这个 Lab 把同一段中文按很多份计来构造训练文本，所以学到的中文合并和这段话的用词贴得很紧。真实的 tokenizer 用的训练文本丰富得多；选 tokenizer 之前，先用你自己用户的各语言消息数一数 token。</p>`);
  return h;
}

/* =====================================================================
   copy that the kit shows: header, fidelity, primer, capability layer
   ===================================================================== */
const SRC={
 gpt2:'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
 bpe:'https://arxiv.org/abs/1508.07909',
 petrov:'https://arxiv.org/abs/2305.15425',
 llama2:'https://arxiv.org/abs/2307.09288',
 rfc:'https://www.rfc-editor.org/rfc/rfc3629',
 l31m:'https://huggingface.co/meta-llama/Llama-3.1-8B', l31:'https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json',
 l3m:'https://huggingface.co/meta-llama/Meta-Llama-3-8B', l3:'https://huggingface.co/unsloth/llama-3-8b/blob/main/config.json',
 l2m:'https://huggingface.co/meta-llama/Llama-2-7b-hf', l2:'https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json',
 gpt2enc:'https://github.com/openai/gpt-2/blob/master/src/encoder.py',
 tkLib:'https://github.com/openai/tiktoken/blob/main/src/lib.rs',
 tkEdu:'https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py',
 hfTrain:'https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-train/src/trainers/bpe/mod.rs',
 hfModel:'https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-encode/src/models/bpe/model.rs',
 hfByte:'https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-encode/src/pre_tokenizers/byte_level.rs',
 spTrain:'https://github.com/google/sentencepiece/blob/master/src/bpe_model_trainer.cc',
 spModel:'https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc'
};
const DESCS=()=>({
  guided:tx(`${STEPS.length===7?'Seven':fint(STEPS.length)} short parts. Predict first, then see where tokens start (bytes), how merges learned from the training text shrink one language and not the other, and what the extra tokens cost. Start here if this is new to you.`,
            `一共 ${STEPS.length} 个部分。先预测，再看 token 从哪里开始（字节），从训练文本学到的合并为什么只让一种语言变短，以及多出来的 token 花在哪里。第一次学从这里开始。`),
  sandbox:tx(`Four kinds of products, from an English-only assistant to a Chinese-first one, each with its own tokenizer of ${SCEN_K} merges. Pick one to see what the same message costs in each language.`,
             `四种产品，从只服务英文的助手到以中文为主的助手，各有一个学了 ${SCEN_K} 次合并的 tokenizer。选一个，看同一句话在两种语言里各要多少 token。`),
  challenge:tx('A sentence pair you have not seen. Choose the training text and the number of merges so that the English sentence takes at most 25 tokens and the Chinese one at most 1.5 times as many, with as few merges as possible.',
               '一对你没见过的句子。选训练文本和合并次数，让英文句子最多 25 个 token、中文句子最多是英文的 1.5 倍，而且合并次数尽量少。')
});
const FIDPOP=()=>{const zb=utf8(TK_ZH).length, eb=utf8(EN_TEXT).length, tile=`<a href="#map~models" data-tipx="${esc(tx('Opens the tree at “How LLMs work”, where this lab’s tile is.','打开知识树的“LLM 怎么工作”，这个 Lab 的方块就在那里。'))}">${tx('this lab’s tile on the tree','知识树上这个 Lab 的方块')}</a>`;
 const cfg=(m,c,n)=>tx(`${cite(m,n,tx('Opens Meta’s repository on Hugging Face in a new tab.',''))} (${cite(c,'readable copy of its config.json','')})`,`${cite(m,n,'在新标签页打开 Meta 在 Hugging Face 上的仓库。')}（${cite(c,'config.json 的可读副本','在新标签页打开这份 config.json 的副本。')}）`);
 return tx(
 `<p><b>Toy run:</b> every count on this page comes from a small byte-level BPE that this page trains in your browser, the same one ${tile} runs; ${J('math','“Show the math”')} works each count out.</p><div class="pop-grid"><div><div class="eyebrow">Modeled</div><ul>`+
 `<li>Byte-level BPE as in GPT-2: the 256 byte values as the first tokens; the text cut into pieces first (a word with its leading space, digits, other symbols); then, again and again, the most frequent pair of neighbouring tokens becomes one new token. Ties go to the smaller pair of token IDs, and training stops when the best pair occurs fewer than 2 times.</li>`+
 `<li>Encoding applies the learned merges in the order they were learned, which gives the same tokens as merging the earliest-learned pair first.</li>`+
 `<li>The training text: ${fint(TK_EN.length)} English paragraphs (${fint(eb)} bytes) and one Chinese paragraph (${fint(zb)} bytes), written for this demo.</li></ul></div>`+
 `<div><div class="eyebrow">Simplified</div><ul>`+
 `<li>${DEF('The training texts',TIP.mixes())} count the one Chinese paragraph 0, 1, 16 or 48 times. So the Chinese merges fit that paragraph’s words closely, and some join the bytes of two neighbouring characters.</li>`+
 `<li>The pieces follow a simplified form of ${cite(SRC.gpt2enc,'GPT-2’s rule')}: GPT-2 also splits off endings such as ’s and ’ll and treats runs of spaces differently. This lab’s texts have no apostrophes and no double spaces, so every count is the same either way.</li>`+
 `<li>Real tokenizers train on far more text and learn far more merges (Llama 3’s vocabulary has 128,256 tokens); this one learns at most ${K_MAX}. The counts show the mechanism, not the numbers of any real model.</li>`+
 `<li>Cost is counted in tokens only: the price per token, the context window and the decode steps all scale with the token count.</li></ul></div>`+
 `<div><div class="eyebrow">Sources</div><ul>`+
 `<li>${cite(SRC.gpt2,'Radford et al. (2019)')}, sections 2.2 and 2.3: byte-level BPE with a base vocabulary of 256 and no merges across character categories (2.2), a vocabulary of 50,257 tokens (2.3); a base of Unicode characters would need over 130,000 (2.2).</li>`+
 `<li>${cite(SRC.bpe,'Sennrich, Haddow and Birch (ACL 2016)')}: BPE for subword units.</li>`+
 `<li>${cite(SRC.llama2,'Touvron et al. (2023)')}, section 2.2: Llama 2’s tokenizer is BPE in SentencePiece, which uses bytes only for characters it does not know; 32,000 tokens.</li>`+
 `<li>${cite(SRC.rfc,'RFC 3629')}: UTF-8 takes 1 byte for U+0000 to U+007F, 2 up to U+07FF, 3 up to U+FFFF (the common Chinese characters) and 4 beyond.</li>`+
 `<li>${cite(SRC.petrov,'Petrov et al. (NeurIPS 2023)')}: the same text in different languages can differ up to 15 times in tokens (Shan, a language of Myanmar, on the tokenizer of ChatGPT and GPT-4). Chinese (Simplified) takes 3.21 times the English tokens on GPT-2’s tokenizer (Table 1) and 0.97 times on XLM-R’s (Table 4).</li>`+
 `<li>Vocabulary sizes and hidden sizes from config.json: ${cfg(SRC.l2m,SRC.l2,'Llama 2 7B')} 32,000 and 4,096; ${cfg(SRC.l3m,SRC.l3,'Llama 3 8B')} 128,256 and 4,096; ${cfg(SRC.l31m,SRC.l31,'Llama-3.1-8B')} 128,256, with a context window of 131,072 tokens.</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">Last verified 2026-09-30. Prototype, not yet expert-reviewed.</p></div></div>`,
 `<p><b>玩具实跑：</b>这一页上的每个 token 数，都来自在你的浏览器里训练的一个小 byte-level BPE，和${tile}用的是同一个；${J('math','“展开公式”')}里逐项算出这些数。</p><div class="pop-grid"><div><div class="eyebrow">建模了什么</div><ul>`+
 `<li>和 GPT-2 一样的 byte-level BPE：一开始的 token 是 256 个字节值；先把文字切成小段（一个词连同它前面的空格、一串数字、一串其他符号）；然后一遍遍把出现最多的一对相邻 token 合成一个新 token。次数相同时取 token 编号较小的那一对；出现最多的一对不到 2 次时停止训练。</li>`+
 `<li>编码时按学到的顺序依次套用合并，结果和“先合并最早学到的那一对”相同。</li>`+
 `<li>训练文本：为演示写的 ${fint(TK_EN.length)} 段英文（${fint(eb)} 字节）和一段中文（${fint(zb)} 字节）。</li></ul></div>`+
 `<div><div class="eyebrow">简化和假设</div><ul>`+
 `<li>${DEF('几种训练文本',TIP.mixes())}把同一段中文按 0、1、16 或 48 份计。所以学到的中文合并和这段话的用词贴得很紧，有的还会把相邻两个汉字的字节拼在一起。</li>`+
 `<li>切段用的是 ${cite(SRC.gpt2enc,'GPT-2 规则')}的简化版：GPT-2 还会单独切出 ’s、’ll 这类词尾，对连续空格的处理也不同。这个 Lab 的文本里没有撇号，也没有连续空格，所以两种切法的结果完全一样。</li>`+
 `<li>真实的 tokenizer 训练文本多得多，学到的合并也多得多（Llama 3 的词表有 128,256 个 token）；这里最多学 ${K_MAX} 次。这些数展示的是机制，不是任何真实模型的数字。</li>`+
 `<li>成本只按 token 数算：每个 token 的价格、上下文窗口和 decode 步数都随 token 数成比例变化。</li></ul></div>`+
 `<div><div class="eyebrow">来源</div><ul>`+
 `<li>${cite(SRC.gpt2,'Radford 等（2019）')}第 2.2 和 2.3 节：byte-level BPE，初始词表 256 个，不跨字符类别合并（2.2）；词表共 50,257 个 token（2.3）；如果从 Unicode 字符起步，初始词表要超过 130,000 个（2.2）。</li>`+
 `<li>${cite(SRC.bpe,'Sennrich、Haddow、Birch（ACL 2016）')}：用 BPE 构造子词单元。</li>`+
 `<li>${cite(SRC.llama2,'Touvron 等（2023）')}第 2.2 节：Llama 2 的 tokenizer 是 SentencePiece 里的 BPE，只在遇到不认识的字符时才拆成字节；词表 32,000 个 token。</li>`+
 `<li>${cite(SRC.rfc,'RFC 3629')}：UTF-8 对 U+0000 到 U+007F 用 1 个字节，到 U+07FF 用 2 个，到 U+FFFF（常用汉字在这里）用 3 个，再往后用 4 个。</li>`+
 `<li>${cite(SRC.petrov,'Petrov 等（NeurIPS 2023）')}：同一段文字换成不同语言，token 数最多能差 15 倍（缅甸的掸语，在 ChatGPT 和 GPT-4 的 tokenizer 上）。简体中文在 GPT-2 的 tokenizer 上是英文 token 数的 3.21 倍（表 1），在 XLM-R 的 tokenizer 上是 0.97 倍（表 4）。</li>`+
 `<li>词表大小和隐藏维度来自 config.json：${cfg(SRC.l2m,SRC.l2,'Llama 2 7B')}是 32,000 和 4,096；${cfg(SRC.l3m,SRC.l3,'Llama 3 8B')}是 128,256 和 4,096；${cfg(SRC.l31m,SRC.l31,'Llama-3.1-8B')}是 128,256，上下文窗口 131,072 个 token。</li></ul>`+
 `<p class="mono small muted" style="margin-top:8px">最近核验 2026-09-30。原型，尚未经专家审阅。</p></div></div>`);};
/* the primer: this lab's prerequisite is the lab Architecture overview (R2.9: the popover starts with its content) */
const PRIPOP=()=>`<p>${esc(kt('token'))}</p><p>${esc(kt('embedding'))}</p>`+tx(`<p>The lab ${labLink('architecture')} shows where the tokens go next.</p>`,`<p>token 接下来去了哪里，在${labLink('architecture')}这个 Lab 里讲。</p>`);
const OUTCOME=()=>tx('predict a sentence’s token count from its UTF-8 bytes and the tokenizer’s training text.','从 UTF-8 字节数和训练文本，预测一句话要多少 token。');
const EXITS=()=>{
  const b=R0(), zc=compute('zh',100,'main'), mb=model('bal');
  const balChars=enc('bal',PAIRS.main.zh).at[100].map(id=>piece(mb.bytes[id])).filter(p=>p.ok&&/[一-鿿]/.test(p.t)).map(p=>p.t);
  const ru='привет', ruB=utf8(ru).length, ruT=enc('tile',ru).at[100].length;
  const H=4096, V2=32000, V3=128256, emb=2*V3*H, both=2*(V3-V2)*H;
  const RNG=[[0,0.4e9],[0.4e9,0.6e9],[0.6e9,0.9e9],[0.9e9,Infinity]], aV=RNG.findIndex(([lo,hi])=>both>=lo&&both<hi);
  const gainD=DEF(tx('roughly 0.8B','约 0.8B'),tx(`2 × (128,256 − 32,000) × 4,096 = ${fint(both)} parameters: 96,256 more rows of 4,096 numbers in each of the two tables.`,`2 ×（128,256 − 32,000）× 4,096 = ${fint(both)} 个参数：两张表各多出 96,256 行，每行 4,096 维。`));
  const ruA=tx(`The answer is ${ruT}, two per letter.`,`答案是 ${ruT} 个，每个字母两个。`);
  const gA=tx('The answer is 0.6B to 0.9B.','答案是 0.6B 到 0.9B。');
  return {
 core:()=>chSubmitted, explain:[
  {q:tx(`On the initial tokenizer, ${b.k} merges shrink the English sentence to under a third of its bytes, but the Chinese one barely shrinks. Why?`,`在初始设置的 tokenizer 上，${b.k} 次合并把英文句子缩到不到原来字节数的三分之一，中文句子却几乎没变短。为什么？`),
   o:[tx('BPE merges the pairs most frequent in its training text, which is mostly English here, so nearly every merge joins English letters and almost none joins Chinese bytes.','BPE 合并的是训练文本里出现最多的那些对，而这里的训练文本几乎全是英文，所以几乎每次合并都在拼英文字母，几乎没有一次用到中文字节。'),
      tx('A Chinese character takes 3 bytes, and BPE only joins pairs of single bytes, so it can never make the 3 bytes of a character into one token.','一个汉字占 3 个字节，而 BPE 只能合并两个单独的字节，所以永远没法把一个汉字的 3 个字节合成一个 token。'),
      tx('BPE learns its merges from the sentence it encodes, and the English sentence repeats pairs such as “in” and “th”, while the Chinese one repeats none.','BPE 从要编码的这句话里学合并，英文句子里“in”“th”这样的字母对反复出现，中文句子里却没有一个字重复。')],a:0,
   n:[null,tx(`BPE joins tokens, not only bytes: a merge can join a byte to a token made by an earlier merge. On the half-and-half text, “${balChars.slice(0,2).join('” and “')}” become single tokens.`,`BPE 合并的是 token，不只是字节：一次合并可以把一个字节和之前合并出的 token 拼起来。在中英各半的文本上，“${balChars.slice(0,2).join('”和“')}”都变成了单个 token。`),
     tx('The merges are learned once, from the training text, before any sentence is encoded; encoding only applies them in order, so the sentence’s own repeats do not matter.','合并是在给任何句子编码之前，从训练文本里一次学好的；编码时只是按顺序套用它们，所以句子自己重不重复没有关系。')],
   why:tx(`Each merge joins the pair that occurs most often in the training text. That text is almost all English, so the early merges build parts of English words such as ${andList(firstPieces('tile',3))}, and almost none of them joins Chinese bytes.`,
          `每次合并都选训练文本里出现最多的那一对。这份文本几乎全是英文，所以前面的合并拼出的都是${andList(firstPieces('tile',3))}这样的英文单词的一部分，几乎没有一次用到中文字节。`)},
  {q:tx(`With the same ${zc.k} merges, the mostly Chinese training text makes the Chinese sentence ${chg(b.zh,zc.zh).replace(/^shrinks/,'shrink')}, but the English one ${chg(b.en,zc.en).replace(/^nearly triples$/,'nearly triple').replace(/s$/,'')}. Why does the English one grow?`,`同样是 ${zc.k} 次合并，以中文为主的训练文本让中文句子${chg(b.zh,zc.zh)}，英文句子却${chg(b.en,zc.en)}。英文为什么变长了？`),
   o:[tx('The merges are one budget that every language shares: with most of the text in Chinese, most of the frequent pairs are Chinese bytes, so they take merges that went to English words before.','所有语言共用同一份合并预算：训练文本大部分是中文时，出现最多的那些对大多是中文字节，它们占掉了原本给英文单词的合并。'),
      tx('Adding Chinese text lowers the counts of the English pairs, since the same pairs are spread over a longer text, so they fall below the count a pair needs before it is merged.','加入中文后，英文字母对的次数变少了，因为同样的字母对摊到了更长的文本上，于是低于能被合并所需的次数。'),
      tx('The Chinese merges cut English words apart: once a Chinese byte pair becomes a token, the text is also cut into pieces at those bytes, English words included.','中文的合并把英文单词切碎了：一对中文字节变成 token 后，文字也会在这些字节处被切成小段，英文单词也不例外。')],a:0,
   n:[null,tx('Counts are not shares: the English pairs occur exactly as often as before. They still get merged, only later, because the Chinese pairs now occur more often and are merged first.','次数不是比例：英文字母对出现的次数和原来完全一样。它们照样会被合并，只是更晚，因为中文的字节对现在出现得更多，先被合并。'),
     tx('A Chinese merge never touches an English word: English words have no Chinese bytes, and the text is cut into pieces the same way whatever the merges are.','中文的合并碰不到英文单词：英文单词里没有中文字节，而且不管学到什么合并，切段的方式都一样。')],
   why:tx(`BPE always merges the most frequent pair first. The English pairs keep their counts, but with the Chinese paragraph counted many times over, Chinese byte pairs outnumber most of them and take ${fint(zc.zhMerges)} of the ${zc.k} merges. English gets few merges, so its words stay split into many small tokens.`,
          `BPE 总是先合并出现最多的那一对。英文字母对的次数没变，但中文段落按很多份计以后，大多数中文字节对都比它们出现得更多，占掉了 ${zc.k} 次合并里的 ${fint(zc.zhMerges)} 次。英文分到的合并很少，单词就被切成很多个小 token。`)}],
 transfer:[
  {q:tx(`The Russian word “${ru}” (hello) has 6 letters, and this lab’s training text has no Russian at all. On the initial tokenizer, how many tokens does it take?`,`俄语单词“${ru}”（你好）有 6 个字母，而这个 Lab 的训练文本里完全没有俄语。在初始设置的 tokenizer 上，它要多少个 token？`),
   o:[tx('6: one per letter','6 个：每个字母一个'),tx('12: two per letter','12 个：每个字母两个'),tx('18: three per letter','18 个：每个字母三个'),tx('24: four per letter','24 个：每个字母四个')],a:[6,12,18,24].indexOf(ruT),
   n:[tx(`${ruA} “6” would hold only if every letter were one token, but a Cyrillic letter takes 2 bytes and no merge joins them.`,`${ruA}“6 个”要求每个字母都是一个 token，但一个西里尔字母占 2 个字节，而且没有合并能把它们拼起来。`),null,
      tx(`${ruA} “18” would mean 3 bytes per letter, as for a Chinese character; a Cyrillic letter takes 2.`,`${ruA}“18 个”意味着每个字母 3 个字节，那是汉字的情况；西里尔字母只占 2 个。`),
      tx(`${ruA} “24” would mean 4 bytes per letter, as for most emoji; a Cyrillic letter takes 2.`,`${ruA}“24 个”意味着每个字母 4 个字节，那是大多数 emoji 的情况；西里尔字母只占 2 个。`)],
   why:tx(`UTF-8 stores a Cyrillic letter in 2 bytes, so “${ru}” is ${ruB} bytes. The training text has no Cyrillic, so none of the merges joins these bytes, and every byte stays a token. A Chinese character, by comparison, is 3 bytes.`,
          `UTF-8 用 2 个字节存一个西里尔字母，所以“${ru}”是 ${ruB} 个字节。训练文本里没有西里尔字母，没有一次合并用得上这些字节，每个字节仍是一个 token。相比之下，一个汉字是 3 个字节。`)},
  {q:tx(`Llama 3’s embedding table and LM head hold about ${A.fmt.big(emb)} parameters for its vocabulary. Llama 2 has the same hidden size and a vocabulary about a quarter as large. About how many more parameters do Llama 3’s two tables hold?`,`Llama 3 的词表在 embedding 表和 LM head 里约占 ${A.fmt.big(emb)} 个参数。Llama 2 的隐藏维度相同，词表约是它的四分之一。Llama 3 的这两张表比 Llama 2 多出大约多少参数？`),
   o:[tx('Less than 0.4B','不到 0.4B'),tx('0.4B to 0.6B','0.4B 到 0.6B'),tx('0.6B to 0.9B','0.6B 到 0.9B'),tx('More than 0.9B','超过 0.9B')],a:aV,
   n:[tx(`${gA} Less than 0.4B would mean Llama 2 kept well over half of Llama 3’s rows; it has about a quarter.`,`${gA}不到 0.4B，意味着 Llama 2 的行数超过 Llama 3 的一半；它其实只有约四分之一。`),
      tx(`${gA} About half of ${A.fmt.big(emb)} would mean Llama 2 had half of Llama 3’s rows; it has about a quarter.`,`${gA}约 ${A.fmt.big(emb)} 的一半，意味着 Llama 2 有 Llama 3 一半的行；它其实只有约四分之一。`),null,
      tx(`${gA} More than 0.9B would leave Llama 2 almost no rows; its rows are about a quarter of Llama 3’s.`,`${gA}超过 0.9B，Llama 2 就几乎没有行了；它的行数约是 Llama 3 的四分之一。`)],
   why:tx(`The two tables grow in proportion to the vocabulary: one row per token in each. Llama 2 has about a quarter of Llama 3’s rows, so Llama 3’s tables hold about three quarters of ${A.fmt.big(emb)} more, ${gainD}. Sizes from config.json: ${cite(SRC.l2,'Llama 2 7B')}, ${cite(SRC.l3,'Llama 3 8B')}.`,
          `这两张表随词表大小成比例增长：每个 token 在每张表里占一行。Llama 2 的行数约是 Llama 3 的四分之一，所以 Llama 3 的两张表多出约 ${A.fmt.big(emb)} 的四分之三，${gainD}。尺寸来自 config.json：${cite(SRC.l2,'Llama 2 7B')}、${cite(SRC.l3,'Llama 3 8B')}。`)}],
 practice:[
  {id:'llm-fundamentals',d:tx('Question 8: run the first three BPE merges by hand (ties go to the smaller pair, as in this lab), then explain how the number of merges trades vocabulary size against sequence length.','第 8 题：手动做 BPE 的前三次合并（次数相同时取较小的一对，和这个 Lab 一样），再说明合并次数怎样在词表大小和序列长度之间取舍。')}]};
};

/* =====================================================================
   views
   ===================================================================== */
function headerHTML(){
  const l=A.catalog&&A.catalog.LAB&&A.catalog.LAB[ID];
  return A.shell.header({id:ID,world:'models',outcome:esc(OUTCOME()),short:l&&l.s?L(l.s):tx('Token cost','token 成本'),title:l?L(l.q):tx('Chinese or English: which costs more tokens?','同样的意思，中文和英文谁更费 token？'),
    minutes:12,fid:'Live',fidPop:FIDPOP(),primer:tx('Primer: tokens','前置知识：token'),primerPop:PRIPOP(),descs:DESCS(),mode});
}
/* part 3 (review A19, R6.25): starting from bytes, as this lab does, against starting from whole characters, drawn for “我” */
function vsHTML(){
  const b=utf8('我').map(hex), gp=cite(SRC.gpt2,tx('GPT-2 paper','GPT-2 论文'),tx('Opens the GPT-2 paper (section 2.2) in a new tab.','在新标签页打开 GPT-2 论文（第 2.2 节）。')), l2=cite(SRC.llama2,tx('Llama 2 paper','Llama 2 论文'),tx('Opens the Llama 2 paper (section 2.2) in a new tab.','在新标签页打开 Llama 2 论文（第 2.2 节）。'));
  return `<div class="tk-cap">${tx('Why start from bytes and not from whole characters?','为什么从字节起步，而不从整个字符起步？')}</div>`+
    `<div class="tk-vsrow"><span class="tk-vsl">${tx('From bytes (this lab, GPT-2)','从字节起步（本 Lab、GPT-2）')}</span><span class="tk-vsfig" aria-hidden="true"><i class="tk-vc">我</i><span class="tk-op">→</span>${b.map(x=>`<i class="tk-vb tk-one"><b>${x}</b></i>`).join('')}</span>`+
    `<span class="tk-vsn">${tx('3 tokens. The base holds only the 256 byte values, and any text can be encoded.','3 个 token。起步的词表只有 256 个字节值，任何文字都能编码。')}</span></div>`+
    `<div class="tk-vsrow"><span class="tk-vsl">${tx('From characters','从字符起步')}</span><span class="tk-vsfig" aria-hidden="true"><i class="tk-vc">我</i><span class="tk-op">→</span><i class="tk-vb"><b>我</b></i></span>`+
    `<span class="tk-vsn">${tx(`1 token. But the base needs a token for every character: over 130,000 to cover all of Unicode (${gp}), and a character it never saw has no token.`,`1 个 token。但起步的词表要给每个字符一个 token：要覆盖整个 Unicode，得超过 130,000 个（${gp}），没见过的字符还是没有 token。`)}</span></div>`+
    `<p class="tk-vsnote small">${tx(`Llama 2’s tokenizer starts from characters and uses bytes only for a character it does not know (${l2}).`,`Llama 2 的 tokenizer 从字符起步，只在遇到不认识的字符时才拆成字节（${l2}）。`)}</p>`;
}
function stageHTML(){
  const strip=(l,name)=>`<div class="tk-strip" data-l="${l}"><div class="tk-sh"><b>${name}</b><span class="tk-sn mono"></span></div>`+
    `<div class="tk-track"><div class="tk-glyphs"></div><div class="tk-boxes"></div><div class="tk-hex mono" aria-hidden="true"></div></div></div>`;
  const crow=(l,name)=>`<div class="tk-crow" data-l="${l}"><span class="tk-cl">${name}</span><div class="tk-cbar" tabindex="0"><span class="tk-cghost"></span><i class="tk-cfill"></i></div><span class="tk-cv mono"></span></div>`;
  return `<div class="panel stage tk-stage" id="tk-stage">`+
   `<div class="stage-head"><div><div class="meter-title" id="tk-st-title"></div><div class="meter-sub" id="tk-st-sub"></div></div>`+
   `<div class="stage-actions"><span class="status" id="tk-status" tabindex="0"></span></div></div>`+
   `<div class="legend-row" id="tk-legend"></div>`+
   `<div class="stage-graph graph tk-graph" id="tk-graph">`+
    `<div class="tk-strips" id="tk-strips"><div class="tk-cap" id="tk-strips-cap"></div>${strip('en',tx('English','英文'))}${strip('zh',tx('Chinese','中文'))}<p class="tk-hexcap small" id="tk-hexcap" hidden></p><div class="tk-vs" id="tk-vs" hidden></div></div>`+
    `<div class="tk-merges" id="tk-merges"><div class="tk-cap" id="tk-mg-cap"></div><div class="tk-ticks" id="tk-ticks" tabindex="0"></div><div class="tk-mg-sum mono" id="tk-mg-sum"></div><div class="tk-mg-first" id="tk-mg-first"></div><ol class="tk-mg-list" id="tk-mg-list"></ol><p class="tk-mg-hexnote small" id="tk-mg-hexnote" hidden></p></div>`+
    `<div class="tk-cost" id="tk-cost"><div class="tk-cap" id="tk-cost-cap"></div>${crow('en',tx('English','英文'))}${crow('zh',tx('Chinese','中文'))}<div class="tk-ctx small" id="tk-ctx"></div></div>`+
   `</div>`+
   `<div class="metrics tk-metrics" id="tk-metrics"></div>`+
   `<details class="math" id="tk-math"><summary>${A.t('showMath')}</summary><div class="mathbody" id="tk-mathbody"></div></details></div>`;
}
function controlsHTML(){
  const note=(t,n)=>`<div class="ctl"><span class="lbl">${t}</span><div class="ctl-note">${n}</div></div>`;
  return `<aside class="panel controls" id="tk-controls" aria-label="${esc(A.t('ctrlAria'))}"><div class="eyebrow">${A.t('ctrlEyebrow')}</div>`+
   `<p class="ctl-lock" id="tk-lock" hidden></p>`+
   `<div class="lockable" id="tk-ctl-mix-wrap">${mixSeg('tk-ctl-mix',false,`<span class="tk-mixnote"></span>`)}</div>`+
   `<div class="lockable" id="tk-ctl-k-wrap">${kCtl('tk-ctl-k',false)}</div>`+
   `<div class="tk-grp lockable" id="tk-ctl-assume"><div class="tk-gh">${tx('Assumed values','假设值')}</div>`+
    note(tx('Pieces before BPE: GPT-2’s rule, simplified','BPE 之前的切段：简化的 GPT-2 规则'),tx('The text is cut into a word with the space before it, a run of digits, or a run of other symbols, and no merge crosses two pieces. It is assumed because this lab copies GPT-2’s choice, without GPT-2’s extra rules for endings such as ’s and for runs of spaces; this lab’s texts have neither, so the counts are the same. Other tokenizers cut differently: SentencePiece treats the space as part of the text. The rule decides which pairs can be merged at all.','文字被切成“一个词连同它前面的空格”“一串数字”或“一串其他符号”，合并不会跨越两段。这是沿用 GPT-2 的做法，但去掉了 GPT-2 对 ’s 这类词尾和连续空格的额外规则；这个 Lab 的文本里两者都没有，所以结果一样。别的 tokenizer 切法不同：SentencePiece 把空格当成文字的一部分。这条规则决定了哪些对有机会被合并。'))+
    note(tx('Ties: the smaller pair of IDs','次数相同：取编号较小的一对'),tx('When two pairs occur equally often, the one with the smaller token IDs is merged first. Any fixed rule works; this one keeps every run the same.','两对出现次数相同时，先合并 token 编号较小的那一对。任何固定的规则都行；这条规则让每次运行结果一样。'))+
    note(tx(`At most ${K_MAX} merges`,`最多 ${K_MAX} 次合并`),tx(`Training stops after ${K_MAX} merges, or earlier when the most frequent pair occurs fewer than 2 times. Real tokenizers learn far more merges from far more text (Llama 3’s vocabulary has 128,256 tokens); ${K_MAX} is enough to show the mechanism on this lab’s few paragraphs.`,`学满 ${K_MAX} 次合并就停；如果出现最多的一对已经不到 2 次，也会提前停。真实的 tokenizer 从多得多的文本里学多得多的合并（Llama 3 的词表有 128,256 个 token）；对这个 Lab 的几段文字来说，${K_MAX} 次足够看清机制。`))+
    note(tx('Context window: 131,072 tokens','上下文窗口：131,072 个 token'),tx('Llama-3.1-8B’s context window (its config.json). It only turns token counts into “how many messages fit”; the comparison between the languages is the same for any window.','这是 Llama-3.1-8B 的上下文窗口（来自它的 config.json）。它只用来把 token 数换算成“能装下几条消息”；两种语言之间的对比，换成任何窗口都一样。'))+`</div>`+
   `<div class="lockable"><button type="button" class="btn ghost sm" data-act="reset">${A.t('reset')}</button></div></aside>`;
}
const CODE=String.raw`# Byte-level BPE (as in GPT-2): learn merges, then encode with them
import re
from collections import Counter
# pieces: a word with its leading space, digits, other symbols, spaces
PAT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")
def merge(seq, a, b, new):  # replace every (a, b) with new
    out, i = [], 0
    while i < len(seq):
        hit = seq[i:i + 2] == (a, b)
        out.append(new if hit else seq[i]); i += 2 if hit else 1
    return tuple(out)
def train(text, n_merges):  # merges never cross two pieces
    words = Counter(tuple(w.encode()) for w in PAT.findall(text))
    merges = []
    for k in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += f
        # the most frequent pair; ties: the smaller pair of ids
        best = min(pairs, key=lambda p: (-pairs[p], p), default=None)
        if best is None or pairs[best] < 2:
            break
        merges.append(best)
        words = Counter({merge(w, *best, 256 + k): f
                         for w, f in words.items()})
    return merges
def encode(text, merges):  # the bytes, then every merge in order
    out = []
    for w in PAT.findall(text):
        seq = tuple(w.encode())
        for k, (a, b) in enumerate(merges):
            seq = merge(seq, a, b, 256 + k)
        out += seq
    return out
m = train("the cat sat on the mat and the hat. " * 9 + "猫在垫子上。", 20)
for s in ("the hat sat", "猫在垫子上"):
    print(s, len(s.encode()), "bytes ->", len(encode(s, m)), "tokens")
# the hat sat 11 bytes -> 3 tokens
# 猫在垫子上 15 bytes -> 13 tokens`;
function connectHTML(){
  return `<section class="wrap connect" aria-labelledby="tk-connH"><div class="eyebrow">${A.t('connEyebrow')}</div><h2 id="tk-connH">${A.t('connTitle')}</h2><div class="connect-grid">`+
   `<article class="panel c-card"><h3>${A.t('cMin')}</h3><p>${tx('A byte-level BPE with this lab’s rules: learn merges from a text, then encode new text by applying them in order.','按这个 Lab 的规则写的 byte-level BPE：从一段文本学合并，再按顺序套用它们给新文字编码。')}</p><pre class="code" id="tk-code">${esc(CODE)}</pre><button class="btn ghost sm" type="button" data-act="copy">${A.t('copyCode')}</button></article>`+
   `<article class="panel c-card"><h3>${A.t('cProd')}</h3><ul class="tk-prod">`+
   tx(`<li>OpenAI’s tiktoken encodes with a trained merge table in ${cite(SRC.tkLib,'<code>src/lib.rs</code>')} (<code>byte_pair_encode</code>), and ${cite(SRC.tkEdu,'<code>tiktoken/_educational.py</code>')} has a short trainer, <code>bpe_train</code>, much like the one above.</li>`+
      `<li>Hugging Face tokenizers trains BPE in ${cite(SRC.hfTrain,'<code>tokenizers/tk-train/src/trainers/bpe/mod.rs</code>')} (<code>BpeTrainer</code>), applies the merges in ${cite(SRC.hfModel,'<code>tokenizers/tk-encode/src/models/bpe/model.rs</code>')}, and maps bytes the GPT-2 way in ${cite(SRC.hfByte,'<code>…/pre_tokenizers/byte_level.rs</code>')}.</li>`+
      `<li>Google’s SentencePiece trains BPE in ${cite(SRC.spTrain,'<code>src/bpe_model_trainer.cc</code>')} and encodes in ${cite(SRC.spModel,'<code>src/bpe_model.cc</code>')}.</li>`,
      `<li>OpenAI 的 tiktoken 在 ${cite(SRC.tkLib,'<code>src/lib.rs</code>')} 里用训练好的合并表编码（<code>byte_pair_encode</code>）；${cite(SRC.tkEdu,'<code>tiktoken/_educational.py</code>')} 里有一个很短的训练函数 <code>bpe_train</code>，和上面的代码很像。</li>`+
      `<li>Hugging Face 的 tokenizers 在 ${cite(SRC.hfTrain,'<code>tokenizers/tk-train/src/trainers/bpe/mod.rs</code>')} 里训练 BPE（<code>BpeTrainer</code>），在 ${cite(SRC.hfModel,'<code>tokenizers/tk-encode/src/models/bpe/model.rs</code>')} 里套用合并，在 ${cite(SRC.hfByte,'<code>…/pre_tokenizers/byte_level.rs</code>')} 里按 GPT-2 的方式处理字节。</li>`+
      `<li>Google 的 SentencePiece 在 ${cite(SRC.spTrain,'<code>src/bpe_model_trainer.cc</code>')} 里训练 BPE，在 ${cite(SRC.spModel,'<code>src/bpe_model.cc</code>')} 里编码。</li>`)+
   `</ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cPapers')}</h3><ul class="refs">`+
   `<li>${cite(SRC.bpe,'Neural Machine Translation of Rare Words with Subword Units')}<span>${tx('Sennrich, Haddow and Birch, ACL 2016: BPE builds a vocabulary of subword units by merging frequent pairs','Sennrich、Haddow、Birch，ACL 2016：BPE 反复合并常见的一对，构造子词词表')}</span></li>`+
   `<li>${cite(SRC.gpt2,'Language Models are Unsupervised Multitask Learners')}<span>${tx('Radford et al., 2019 (GPT-2), section 2.2: BPE on bytes, with a base vocabulary of 256','Radford 等，2019（GPT-2）第 2.2 节：在字节上做 BPE，初始词表 256 个')}</span></li>`+
   `<li>${cite(SRC.petrov,'Language Model Tokenizers Introduce Unfairness Between Languages')}<span>${tx('Petrov, La Malfa, Torr and Bibi, NeurIPS 2023: the same text can take up to 15 times as many tokens in another language (Shan); Chinese takes about 3.2 times on GPT-2’s tokenizer','Petrov、La Malfa、Torr、Bibi，NeurIPS 2023：同一段文字换一种语言，token 数最多能差 15 倍（掸语）；中文在 GPT-2 的 tokenizer 上约是英文的 3.2 倍')}</span></li></ul></article>`+
   `<article class="panel c-card"><h3>${A.t('cDrill')}</h3><p id="tk-drill-q"></p><div class="cta-row"><button class="btn sm" type="button" data-act="drill-reveal">${A.t('drillReveal')}</button><button class="btn ghost sm" type="button" data-act="drill-new">${A.t('drillNew')}</button></div><div class="drill-a" id="tk-drill-a" hidden></div></article>`+
   `</div></section>`;
}
function viewHTML(){return `<div class="tk-app m2" id="tk-app">`+headerHTML()+`<div class="wrap bench"><aside class="rail" id="tk-rail" aria-label="${esc(tx('Lab parts','Lab 各部分'))}"></aside>`+stageHTML()+controlsHTML()+`</div>`+connectHTML()+`</div>`;}

/* =====================================================================
   rail
   ===================================================================== */
function sandboxHTML(){
  const cur=presetNow();
  return `<div class="railcard"><div class="eyebrow">${A.t('modeSandbox')}</div><h3 class="rh">${tx('Pick a scenario','选一个应用场景')}</h3>`+
   `<p class="small">${tx(`Pick a scenario and see how many tokens the same message takes in each language. Each tokenizer learns ${SCEN_K} merges, three times the guided parts’ 100, so a text can give a different ratio here.`,`选一个场景，看看同一条消息用中文和英文各要多少 token。每个 tokenizer 都学 ${SCEN_K} 次合并，是分步引导里 100 次的三倍，所以同一份文本在这里的比例可能不同。`)}</p>`+
   `<div class="presets" role="group" aria-label="${esc(tx('Scenarios, pick one','应用场景，选一个'))}">${PRESETS.map((p,i)=>`<button type="button" class="preset" data-preset="${i}" aria-pressed="${i===cur}"><b>${esc(L(p.t))}</b><span>${esc(L(p.d))}</span></button>`).join('')}</div>`+
   `<div class="obs" id="tk-insight" aria-live="polite"></div>`+
   `<p class="small muted">${tx(`Change anything else under ${J('controls','Settings')}.`,`想改别的，在“${J('controls','自由调参')}”里改。`)}</p>${A.shell.toChallenge()}</div>`;
}
function challengeHTML(){
  const P=PAIRS.ch, e=enc('tile',P.en);
  const submit=chSubmitted?'':`<button type="button" class="btn" data-act="submit">${tx('Submit my answer','提交答案')}</button><p class="small muted" style="margin:8px 0 0">${tx('Submitting tells you whether you met the goal, and why.','提交后会告诉你是否达标，以及原因。')}</p>`;
  return `<div class="railcard"><div class="eyebrow">${A.t('modeChallenge')}</div><h3 class="rh">${tx('English at most 25 tokens, Chinese at most 1.5 times that, with the fewest merges','英文最多 25 个 token，中文最多是英文的 1.5 倍，合并次数尽量少')}</h3>`+
   tx(`<p>A bilingual app sends a new sentence pair: “${P.en}” and “${P.zh}”. Train its tokenizer on one of this lab’s training texts (English only is left out, since it can never shrink the Chinese sentence), so that:</p><ul class="tk-goals"><li>the English sentence takes at most 25 tokens (it has ${e.bytes} bytes);</li><li>the Chinese sentence takes at most 1.5 times the English one’s tokens;</li><li>the vocabulary is as small as possible, that is, as few merges as possible: each merge adds a row to the embedding table and another to the LM head, ${fint(2*4096)} parameters per token for Llama-3.1-8B.</li></ul>`,
      `<p>一个双语应用要发送一对新句子：“${P.en}”和“${P.zh}”。用这个 Lab 的一种训练文本训练它的 tokenizer（不含“只有英文”，因为它永远不会让中文句子变短），做到：</p><ul class="tk-goals"><li>英文句子最多 25 个 token（它有 ${e.bytes} 个字节）；</li><li>中文句子的 token 数最多是英文的 1.5 倍；</li><li>词表尽量小，也就是合并次数尽量少：每次合并都给 embedding 表和 LM head 各添一行，对 Llama-3.1-8B 来说每个 token 是 ${fint(2*4096)} 个参数。</li></ul>`)+
   A.ui.seg({id:'tk-ic-chmix',key:'chmix',label:tx('Training text','训练文本'),options:CH.mixes.map(id=>[id,mixName(id)]),value:S.mix,inline:true})+
   A.ui.seg({id:'tk-ic-chk',key:'chk',label:tx('Merges learned','已学的合并次数'),options:CH.ks.map(k=>[String(k),fint(k)]),value:String(S.k),inline:true})+
   `<div class="goal mono" id="tk-ch-g1"><span>${tx('English now','现在的英文')}</span><span><b id="tk-ch-en"></b>${tx(' / at most 25 tokens',' / 最多 25 个 token')}</span></div><div class="progress" id="tk-ch-p1" aria-hidden="true"><i></i></div>`+
   `<div class="goal mono" id="tk-ch-g2"><span>${tx('Chinese ÷ English now','现在的中文 ÷ 英文')}</span><span><b id="tk-ch-r"></b>${tx(' / at most 1.5',' / 最多 1.5')}</span></div><div class="progress" id="tk-ch-p2" aria-hidden="true"><i></i></div>`+
   submit+`<div id="tk-ch-res" class="tk-ch-res" aria-live="polite"></div></div>`;
}
function railHTML(){
  if(mode==='guided'){const s=STEPS[step];
    return A.ui.steps({steps:STEPS.map(x=>({title:L(x.title),lead:esc(leadOf(x))})),m2:true,current:step,body:s.body(),back:step>0,showMe:!!s.showMe&&(!s.gate||s.gate()),next:step<STEPS.length-1,nextDisabled:isNextDisabled(),skip:!!s.skip});}
  return mode==='sandbox'?sandboxHTML():challengeHTML();
}
function isNextDisabled(){const s=STEPS[step]; return (!!s.expect&&!observed[step])||(!!s.gate&&!s.gate());}
function railFull(){return railHTML()+(mode==='challenge'&&A.ui.exits?A.ui.exits(ID,EXITS()):'')+((mode!=='guided'||step===STEPS.length-1)&&A.shell.nextLab?A.shell.nextLab(ID):'');}
function renderRail(){const rail=$('#tk-rail',app); if(!rail) return; rail.innerHTML=railFull(); A.ui.initRanges(rail); render();}
function updateRail(){
  const rail=$('#tk-rail',app); if(!rail) return;
  if(mode==='guided'){
    const s=STEPS[step], box=$('[data-obs]',rail);
    if(s.expect&&s.expect()) observed[step]=true;
    const show=!!s.obs&&(!!s.always||!s.expect||!!observed[step])&&(!s.gate||s.gate());
    $$('[data-lever]',rail).forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.lever===S.mix)); const rr=b.querySelector('.lv-r'); if(rr) rr.textContent=lvRes(b.dataset.lever);});
    const ms=$('.seg-ctl[data-key="mix"]',rail); if(ms) A.ui.setPressed(ms,S.mix);
    const hs=$('.seg-ctl[data-key="hex"]',rail); if(hs) A.ui.setPressed(hs,hexOn?'1':'0');
    $$('[data-rem]',rail).forEach(b=>{const on=b.dataset.rem===remView; b.setAttribute('aria-pressed',String(on)); b.textContent=on?tx('In the chart now','图里正显示'):tx('Show in the chart','在图里看');});
    if(box){box.hidden=!show;
      if(show){const html=`<div class="eyebrow">${L(s.label||{en:'Observed',zh:'观察到'})}</div>`+s.obs();
        if(box._h!==html){const again=box._step===step&&box._h!=null; box.innerHTML=html; box._h=html; box._step=step; if(again) A.ui.obsBump(box);}
        if(!box._seen){box._seen=true; if(wantObs) revealSoon();}}}
    /* a disabled Next says why (review A6) */
    const why=s.id==='merges'?tx('Learn at least 20 merges first.','先学至少 20 次合并。'):(s.id==='predict'?tx('Pick an answer first, or skip the prediction.','先选一个答案，或者跳过预测。'):(s.id==='mix'?tx('Try another training text first.','先试另一种训练文本。'):''));
    $$('[data-nav="next"]',rail).forEach(nb=>{const dis=isNextDisabled(); nb.disabled=dis; if(dis&&why) nb.dataset.tipx=why; else delete nb.dataset.tipx;});
    $$('[data-nav="skip"]',rail).forEach(b=>{b.dataset.tipx=tx('Skips the prediction; part 3 shows the answer.','跳过预测；第 3 部分会给出答案。');});
    const sm=$('[data-nav="showme"]',rail); if(sm) sm.hidden=!!observed[step];
  } else if(mode==='sandbox'){
    const cur=presetNow(); $$('[data-preset]',rail).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.preset===cur)));
    const ins=$('#tk-insight',rail); if(ins){const html=cur>=0?`<div class="eyebrow">${esc(L(PRESETS[cur].t))}</div>`+scenHTML(cur):(sbTouched?`<p class="small">${tx('The current settings match none of these scenarios.','现在的设置和这几个场景都不一样。')}</p>`:`<p class="small">${tx('Pick one of the four scenarios above.','从上面四个场景里选一个。')}</p>`);
      if(ins._h!==html){const again=ins._h!=null; ins.innerHTML=html; ins._h=html; if(again) A.ui.obsBump(ins);}}
  } else {
    const p=chPoint(S.mix,S.k), en=$('#tk-ch-en',rail); if(!en) return;
    A.ui.setPressed($('.seg-ctl[data-key="chmix"]',rail),S.mix); A.ui.setPressed($('.seg-ctl[data-key="chk"]',rail),String(S.k));
    en.textContent=tokN(p.en); $('#tk-ch-r',rail).textContent=r2(p.ratio);
    const bar=(g,pr,ok,frac)=>{$(g,rail).classList.toggle('met',ok); $(g,rail).classList.toggle('tk-over',!ok); $(pr,rail).classList.toggle('met',ok); $(pr,rail).classList.toggle('tk-over',!ok); $(pr+' i',rail).style.width=Math.min(100,frac*100).toFixed(1)+'%';};
    bar('#tk-ch-g1','#tk-ch-p1',p.okEn,p.en/CH.enMax); bar('#tk-ch-g2','#tk-ch-p2',p.okR,p.ratio/1.5);
    if(chSubmitted&&p.ok) chReveal=true;
    const res=$('#tk-ch-res',rail), html=chResultHTML(); if(res._h!==html){const again=res._h!=null&&res._h!==''&&html!==''; res.innerHTML=html; res._h=html; if(again) A.ui.obsBump(res);}
  }
}

/* =====================================================================
   stage drawing
   ===================================================================== */
function statusOf(){
  const r=R();
  if(mode==='challenge'){const p=chPoint(S.mix,S.k);
    return {cls:p.ok?'ok':'bad',txt:p.ok?tx('Meets both targets','两个目标都达到'):tx('Misses a target','没达到目标'),short:p.ok?tx('Both met','都达到'):tx('Not yet','还没达到'),
      tip:tx(`This badge turns green when the English sentence takes at most 25 tokens and the Chinese one at most 1.5 times as many. Now the English sentence takes ${p.en} tokens and the Chinese one ${p.zh}, ${r2(p.ratio)} times as many.`,
             `英文句子最多 25 个 token、中文最多是英文的 1.5 倍时，这个徽章变成绿色。现在英文句子是 ${p.en} 个 token，中文句子是 ${p.zh} 个，是英文的 ${r2(p.ratio)} 倍。`)};}
  if(zhUnknown()) return pred?{cls:'neutral',txt:tx('Prediction saved','预测已记下'),short:tx('Saved','已记下'),
    tip:tx('Your prediction is saved. Part 3 shows the Chinese sentence’s tokens with no merges.','你的预测已记下。第 3 部分会显示没有合并时中文句子的 token。')}
    :{cls:'neutral',txt:tx('Predict first','先预测'),short:tx('Predict first','先预测'),
    tip:tx('The Chinese sentence’s tokens with no merges stay hidden until you answer the question in part 2; part 3 shows them.','没有合并时中文句子的 token，在你回答第 2 部分的问题之前先隐藏；第 3 部分会显示。')};
  return {cls:'neutral',txt:tx(`Chinese: ${r2(r.ratio)} times the English tokens`,`中文是英文 token 数的 ${r2(r.ratio)} 倍`),short:tx(`Chinese ${r2(r.ratio)}×`,`中文 ${r2(r.ratio)} 倍`),
    tip:tx(`The Chinese sentence takes ${r2(r.ratio)} times as many tokens as the English one: ${r.zh} ÷ ${r.en} = ${r2(r.ratio)}.`,`中文句子的 token 数是英文句子的 ${r2(r.ratio)} 倍：${r.zh} ÷ ${r.en} = ${r2(r.ratio)}。`)};
}
function drawHead(){
  const r=R(), s=statusOf();
  $('#tk-st-title',app).textContent=mode==='challenge'?tx('How many tokens does the new sentence pair take on the tokenizer you train?','新的一对句子，在你训练的 tokenizer 上各要多少 token？'):tx('How many tokens does the same sentence take in English and in Chinese?','同一句话，英文和中文各要多少 token？');
  $('#tk-st-sub',app).textContent=tx(`Trained on ${mixPh(S.mix)} (${shareS(S.mix)} English), ${mergeN(r.k)} learned`,`训练文本：${mixPh(S.mix)}（${shareS(S.mix)} 是英文），已学 ${fint(r.k)} 次合并`);
  const st=$('#tk-status',app); st.className='status '+s.cls; st.textContent=s.txt; st.dataset.tipx=s.tip;
}
function legendHTML(){
  const it=(cls,txt,tip,jump)=>`<span data-tipx="${esc(tip)}" tabindex="0"${jump?` data-jump="${jump}"`:''}><i class="sw ${cls}"></i><span>${esc(txt)}</span></span>`;
  return `<span data-tipx="${esc(TIP.space())}" tabindex="0" data-jump="strips"><b class="tk-sw-dot">·</b><span>${esc(tx('a space','空格'))}</span></span>`+
    it('tk-sw-one',tx('hatched: one byte, no merge yet','斜纹：单个字节，还没被合并'),tx('A hatched box is a token that is still a single byte: no merge learned so far covers it.','斜纹的方框是仍然只有一个字节的 token：目前学到的合并都用不上它。'),'strips')+
    it('tk-sw-tok',tx('solid: a merged token','实心：合并出的 token'),tx('A solid box is a token that merges made from two or more bytes. Its width is its number of bytes.','实心的方框是合并出来的 token，由两个或更多字节组成，宽度就是它的字节数。'),'strips')+
    it('tk-sw-tall',tx('tall tick: a merge of Chinese bytes','高刻度：合并中文字节'),tx('In the row of ticks, each tick is one merge in the order it was learned. A tall tick made a token with bytes of Chinese characters.','刻度行里，每条刻度是一次合并，按学到的顺序排列。高刻度造出的 token 含有汉字的字节。'),'merges')+
    it('tk-sw-short',tx('short tick: a merge of English or other ASCII','矮刻度：合并英文或其他 ASCII 字符'),tx('A short tick made a token of English letters, spaces or punctuation (1-byte characters).','矮刻度造出的 token 由英文字母、空格或标点组成（每个 1 个字节的字符）。'),'merges')+
    (predHidden()?'':it('tk-sw-ghost',tx('dashed: the bytes, with no merges','虚线框：字节数，也就是没有合并时'),tx('A dashed outline in the cost rows is the sentence’s number of bytes: its tokens with no merges.','成本条里的虚线框是这句话的字节数，也就是没有合并时的 token 数。'),'cost'));
}
/* the strips: one cell per byte on one scale for both sentences; a token is a box over its bytes. Boxes are keyed by their first byte, so
   when a merge joins two tokens the left box grows over the right one, which fades (R0.1); reduced motion jumps to the end state.
   Parts 1 and 2 draw the Chinese sentence as plain text over a dashed box, without its byte cells (R3.8, review A1). */
let cwLast=0;
const glyphTip=c=>c.ch===' '?TIP.space():tx(`“${c.ch}”: ${c.n} byte${c.n>1?'s':''} in UTF-8, ${utf8(c.ch).map(hex).join(' ')}.`,`“${c.ch}”：UTF-8 里 ${c.n} 个字节，${utf8(c.ch).map(hex).join(' ')}。`);
function drawStrips(anim){
  const P=PAIRS[pairId()], m=model(S.mix), k=R().k, st=$('#tk-strips',app); if(!st) return;
  const n=Math.max(enc(S.mix,P.en).bytes,enc(S.mix,P.zh).bytes);
  const W=($('.tk-track',st).parentNode.clientWidth||600)-2, cw=Math.max(6,Math.min(22,Math.floor(W/n))); cwLast=cw;
  const noAnim=!anim||A.reduceMotion(), masked=predHidden();
  ['en','zh'].forEach(l=>{
    const e=enc(S.mix,P[l]), ids=e.at[k], box=$(`.tk-strip[data-l="${l}"]`,st), trk=$('.tk-track',box), gl=$('.tk-glyphs',trk), bx=$('.tk-boxes',trk), hx=$('.tk-hex',trk);
    const hideZh=l==='zh'&&masked;
    trk.style.width=((hideZh?n:e.bytes)*cw)+'px'; trk.style.setProperty('--cw',cw+'px'); trk.classList.toggle('tk-masked',hideZh);
    const gk=P[l]+'|'+cw+'|'+(hideZh?'m':'')+'|'+A.lang();
    if(gl._k!==gk){gl._k=gk; gl.innerHTML=hideZh?`<span class="tk-gplain">${esc(P[l])}</span>`
      :e.chars.map(c=>`<span class="tk-g" style="left:${c.b0*cw}px;width:${c.n*cw}px${c.n===1&&cw<9?`;font-size:${Math.max(8,cw+1)}px`:''}" tabindex="-1" data-tipx="${esc(glyphTip(c))}">${c.ch===' '?'<em>·</em>':esc(c.ch)}</span>`).join('');}
    const hon=hexShown()&&!hideZh, hk=gk+(hon?'|1':'|0'); if(hx._k!==hk){hx._k=hk; hx.hidden=!hon; if(hon){const bs=utf8(P[l]); hx.innerHTML=bs.map((v,i)=>`<span style="left:${i*cw}px;width:${cw}px"><b>${hex(v)[0]}</b><b>${hex(v)[1]}</b></span>`).join('');} else hx.innerHTML='';}
    const map=bx._m||(bx._m=new Map());
    if(hideZh){for(const [s,el] of [...map]){map.delete(s); el.remove();}
      let mk=$('.tk-mask',bx); if(!mk){mk=document.createElement('span'); mk.className='tk-mask'; mk.tabIndex=0; bx.appendChild(mk);}
      mk.textContent=zhUnknown()?(pred?tx('? tokens: shown in part 3','? 个 token：第 3 部分显示'):tx('? tokens: answer the question first','? 个 token：先回答问题')):tx('Its tokens are drawn from part 3 on','第 3 部分起画出它的 token');
      mk.dataset.tipx=tx('The Chinese sentence’s tokens are drawn from part 3 on, after your prediction in part 2.','中文句子的 token 从第 3 部分起才画出来，在你回答第 2 部分的预测之后。');
      $('.tk-sn',box).textContent=zhUnknown()?tx('? tokens','? 个 token'):tokN(ids.length);
      trk.setAttribute('aria-label',tx('Chinese sentence: its tokens are shown in part 3.','中文句子：它的 token 在第 3 部分显示。'));
      return;}
    const mk0=$('.tk-mask',bx); if(mk0) mk0.remove();
    const alive=new Set(); let s0=0;
    const charOf=b=>e.chars.find(c=>b>=c.b0&&b<c.b0+c.n);
    ids.forEach((id,i)=>{const bs=m.bytes[id], len=bs.length; let el=map.get(s0), isNew=false;
      if(!el||el.classList.contains('tk-gone')){el=document.createElement('i'); el.className='tk-tok'; el.tabIndex=0; bx.appendChild(el); map.set(s0,el); isNew=true;}
      el.style.left=(s0*cw+1)+'px'; el.style.width=Math.max(2,len*cw-2)+'px'; el.classList.toggle('tk-one',len===1);
      const p=piece(bs), c0=charOf(s0), c1=charOf(s0+len-1);
      const what=p.ok?tx(`“${p.t}”`,`“${p.t}”`):(c0===c1?tx(`byte${len>1?'s':''} ${p.t} of “${c0.ch}”`,`“${c0.ch}”的字节 ${p.t}`):tx(`bytes ${p.t}, from “${c0.ch}” and “${c1.ch}”`,`字节 ${p.t}，横跨“${c0.ch}”和“${c1.ch}”`));
      el.dataset.tipx=len===1?tx(`Token ${i+1} of ${ids.length}: ${what}. It is a single byte, since no merge learned so far covers it.`,`第 ${i+1} 个 token（共 ${ids.length} 个）：${what}。它还是单个字节，目前学到的合并都用不上它。`)
        :tx(`Token ${i+1} of ${ids.length}: ${what}, ${len} bytes, made by merge ${id-255} of ${fint(k)}.`,`第 ${i+1} 个 token（共 ${ids.length} 个）：${what}，${len} 个字节，由第 ${id-255} 次合并造出（共 ${fint(k)} 次）。`);
      if(isNew&&!noAnim){el.classList.add('tk-new'); requestAnimationFrame(()=>requestAnimationFrame(()=>el.classList.remove('tk-new')));}
      alive.add(s0); s0+=len;});
    for(const [s,el] of [...map]){if(alive.has(s)) continue; map.delete(s);
      if(noAnim) el.remove(); else {el.classList.add('tk-gone'); setTimeout(()=>el.remove(),320);}}
    $('.tk-sn',box).textContent=tokN(ids.length);
    trk.setAttribute('aria-label',tx(`${l==='en'?'English':'Chinese'} sentence: ${ids.length} tokens.`,`${l==='en'?'英文':'中文'}句子：${ids.length} 个 token。`));
  });
  $('#tk-strips-cap',app).textContent=mode==='challenge'?tx('Each cell is one byte, and each box one token: the new pair','每一格是一个字节，每个方框是一个 token：新的一对句子')
    :tx('Each cell is one byte, and each box one token','每一格是一个字节，每个方框是一个 token');
  const hc=$('#tk-hexcap',app), hon=hexShown(); hc.hidden=!hon;
  if(hon) hc.textContent=tx('Each byte in hexadecimal, read top to bottom: 4 over 9 is 49, the byte of “I”.','每个字节用十六进制表示，从上往下读：4 在 9 上面就是 49，也就是“I”的字节。');
  const vs=$('#tk-vs',app), von=mode==='guided'&&STEPS[step].id==='bytes'; vs.hidden=!von; if(von&&vs._l!==A.lang()){vs._l=A.lang(); vs.innerHTML=vsHTML();}
}
/* the merges: one tick per merge in the order learned (tall: Chinese bytes, short: ASCII), the first three, then the last three with
   their counts; bytes that are not whole characters say which characters they come from (review A4) */
function mergeRowPiece(bs,at){const p=piece(bs); if(p.ok) return `<span class="tk-pc">${esc(p.t)}</span>`;
  const w=hexWhere(bs,at), tip=w?(w.shared?sharedChip(p.t,w):tx(`Bytes ${p.t}: ${w.en}, in the training text’s Chinese paragraph.`,`字节 ${p.t}：${w.zh}，来自训练文本里的那段中文。`)):'';
  return `<span class="tk-pc tk-hx"${w?` tabindex="0" data-tipx="${esc(tip)}"`:''}>${esc(p.t)}</span>`+(w?`<span class="tk-where">(${esc(L(w))})</span>`:'');}
function drawMerges(){
  const m=model(S.mix), k=R().k, P=PAIRS[pairId()], e=enc(S.mix,P.en), z=enc(S.mix,P.zh), zc=m.zhPiece.slice(0,k).filter(Boolean).length, w=MIX[S.mix].zh;
  $('#tk-mg-cap',app).textContent=tx(`Merges learned from the training text: ${fint(k)} (this lab stops at ${K_MAX})`,`从训练文本学到的合并：${fint(k)} 次（最多 ${K_MAX} 次）`);
  let h=''; for(let i=0;i<K_MAX;i++){const on=i<k, zh=m.zhPiece[i]; h+=on?`<rect x="${i+0.12}" y="${zh?1:13}" width="0.76" height="${zh?22:10}" class="${zh?'tk-tz':'tk-te'}"/>`:`<rect x="${i+0.12}" y="21.6" width="0.76" height="1.4" class="tk-tu"/>`;}
  const tk=$('#tk-ticks',app); tk.innerHTML=`<svg viewBox="0 0 ${K_MAX} 24" preserveAspectRatio="none" aria-hidden="true">${h}</svg>`;
  tk.dataset.tipx=tx(`Each tick is one merge, in the order it was learned. Of the ${mergeN(k)} learned, ${fint(zc)} built tokens with Chinese bytes (tall) and ${fint(k-zc)} built tokens of English or other 1-byte characters (short).`,`每条刻度是一次合并，按学到的顺序排列。已学的 ${fint(k)} 次合并里，${fint(zc)} 次造出含中文字节的 token（高），${fint(k-zc)} 次造出英文或其他单字节字符的 token（矮）。`);
  tk.setAttribute('aria-label',tk.dataset.tipx);
  $('#tk-mg-sum',app).hidden=!k; $('#tk-mg-sum',app).textContent=tx(`Merges that built English or ASCII tokens: ${fint(k-zc)} of ${fint(k)}. Merges that built Chinese tokens: ${fint(zc)}.`,`造出英文或 ASCII token 的合并：${fint(k-zc)} 次（共 ${fint(k)} 次）。造出中文 token 的合并：${fint(zc)} 次。`);
  const fst=$('#tk-mg-first',app);
  fst.innerHTML=k>=3?`<span class="tk-mg-fl">${tx('The first merges learned:','最先学到的合并：')}</span>`+[0,1,2].map(i=>`<span class="tk-mg-n mono">#${i+1}</span>${mergeRowPiece(m.bytes[256+i])}`).join(' '):'';
  const list=$('#tk-mg-list',app), hn=$('#tk-mg-hexnote',app);
  if(!k){list.innerHTML=`<li class="tk-mg-none">${tx('No merges yet: every byte is its own token.','还没有合并：每个字节都是一个 token。')}</li>`; hn.hidden=true; return;}
  const rows=[]; let anyHex=false;
  for(let i=k-1;i>=Math.max(0,k-3);i--){const [a,b,c]=m.merges[i], nb=m.bytes[256+i], chn=nb.some(x=>x>=128), rw=chn&&!piece(nb).ok?hexWhere(nb):null, at=chn&&!(rw&&rw.shared)?findZh(nb):-1;
    if(!piece(nb).ok||!piece(m.bytes[a]).ok||!piece(m.bytes[b]).ok) anyHex=true;
    const chips=(e.used[i]?`<span class="tk-chip">${tx('in the English sentence','英文句子用到')}</span>`:'')+(z.used[i]?`<span class="tk-chip">${tx('in the Chinese sentence','中文句子用到')}</span>`:'');
    const cnt=chn&&w>1&&c%w===0?tx(` It occurs ${c/w===1?'once':fint(c/w)+' times'} in the Chinese paragraph, which this training text counts ${w} times: ${fint(c)} in all.`,`这一对在那段中文里出现 ${fint(c/w)} 次，这份训练文本把那段中文按 ${w} 份计，一共 ${fint(c)} 次。`):'';
    const use=e.used[i]||z.used[i]?(e.used[i]&&z.used[i]?tx(' Both sentences use it.','两个句子都用到了它。'):(e.used[i]?tx(' The English sentence uses it.','英文句子用到了它。'):tx(' The Chinese sentence uses it.','中文句子用到了它。'))):tx(' Neither sentence uses it.','两个句子都没用到它。');
    rows.push(`<li data-tipx="${esc(tx(`Merge ${i+1}: the pair occurred ${fint(c)} times in the training text, more than any other pair left, so it became one new token.`,`第 ${i+1} 次合并：这一对在训练文本里出现了 ${fint(c)} 次，比剩下的任何一对都多，所以合成了一个新 token。`)+cnt+sharedTip(rw)+use)}" tabindex="0"><span class="tk-mg-n mono">#${i+1}</span>${mergeRowPiece(m.bytes[a],at)}<span class="tk-op">+</span>${mergeRowPiece(m.bytes[b],at<0?-1:at+m.bytes[a].length)}<span class="tk-op">→</span>${mergeRowPiece(nb)}<span class="tk-mg-c mono">${tx(`${fint(c)} times`,`${fint(c)} 次`)}</span>${chips}</li>`);}
  list.innerHTML=`<li class="tk-mg-h">${tx('The last merges learned: pair → new token, and how often the pair occurred','最近学到的合并：一对 → 新 token，以及这一对出现的次数')}</li>`+rows.join('');
  hn.hidden=!anyHex; if(anyHex) hn.textContent=tx('Hex digits are bytes that do not form a whole character; the note in brackets says which characters they come from.','十六进制数字是凑不成完整字符的字节；括号里写的是它们来自哪些字。');
}
/* the cost: the tokens of each sentence on one scale (the bytes, dashed, from part 3 on), then the context window */
function drawCost(){
  const r=R(), hid=predHidden(), unk=zhUnknown(), n=Math.max(r.enBytes,r.zhBytes);
  $('#tk-cost-cap',app).textContent=tx('What the same message costs on this tokenizer','同一句话在这个 tokenizer 上的成本');
  [['en',r.en,r.enBytes],['zh',r.zh,r.zhBytes]].forEach(([l,t,b])=>{const row=$(`.tk-crow[data-l="${l}"]`,app), hide=l==='zh'&&unk;
    $('.tk-cfill',row).style.width=hide?'0%':(t/n*100).toFixed(2)+'%'; const g=$('.tk-cghost',row); g.hidden=hid; g.style.width=(b/n*100).toFixed(2)+'%';
    $('.tk-cv',row).textContent=hide?tx('? tokens','? 个 token'):tokN(t);
    $('.tk-cbar',row).dataset.tipx=hide?(pred?tx('The Chinese sentence’s tokens with no merges are shown in part 3.','没有合并时中文句子的 token 数在第 3 部分显示。'):tx('The Chinese sentence’s tokens with no merges are hidden until you answer the question in part 2.','没有合并时中文句子的 token 数，在你回答第 2 部分的问题之前先隐藏。'))
      :(hid?tx(`The ${l==='en'?'English':'Chinese'} sentence takes ${t} tokens on this tokenizer.`,`${l==='en'?'英文':'中文'}句子在这个 tokenizer 上是 ${t} 个 token。`)
      :tx(`The ${l==='en'?'English':'Chinese'} sentence takes ${t} tokens on this tokenizer; its ${b} bytes (the dashed outline) are what it takes with no merges.`,`${l==='en'?'英文':'中文'}句子在这个 tokenizer 上是 ${t} 个 token；虚线框是它的 ${b} 个字节，也就是没有合并时的 token 数。`));});
  const fewer=r.ctxZh<r.ctxEn;
  $('#tk-ctx',app).innerHTML=unk?tx(`<p>Llama-3.1-8B’s ${DEF('context window',TIP.ctx())} holds ${fint(r.ctxEn)} copies of the English sentence at this point; the Chinese count ${pred?'is shown in part 3':'waits for your prediction'}.</p>`,`<p>此时 Llama-3.1-8B 的${DEF('上下文窗口',TIP.ctx())}能装下 ${fint(r.ctxEn)} 句英文；中文的数字${pred?'在第 3 部分显示':'等你预测之后再显示'}。</p>`)
    :tx(`<p>A bill priced per token is ${r2(r.ratio)} times as high for the Chinese message.</p><p>Llama-3.1-8B’s ${DEF('context window',TIP.ctx())} holds ${fint(r.ctxEn)} copies of the English sentence ${fewer?'but only':'and'} ${fint(r.ctxZh)} of the Chinese one.</p>`,
      `<p>中文消息按 token 计费，是英文的 ${r2(r.ratio)} 倍。</p><p>Llama-3.1-8B 的${DEF('上下文窗口',TIP.ctx())}能装下 ${fint(r.ctxEn)} 句英文，${fewer?'却只能装下':'也能装下'} ${fint(r.ctxZh)} 句中文。</p>`);
}
/* metric cards: numbers only in the value (units in the label and the line under it, R7.8 review A18); ▲▼ only after the learner's own
   change, from the numbers as shown (R6.6) */
const numOf=v=>{const m=String(v).replace(/,/g,'').match(/(\d+(?:\.\d+)?)/); return m?parseFloat(m[1]):NaN;};
const decOf=v=>{const m=String(v).replace(/,/g,'').match(/\d+\.(\d+)/); return m?m[1].length:0;};
function metricsDef(){
  const r=R(), hid=predHidden(), unk=zhUnknown(), d=4096, par=2*r.vocab*d;
  return [
   {k:'en',label:tx('English tokens','英文 token 数'),v:fint(r.en),sub:tx(`from ${r.enBytes} bytes`,`来自 ${r.enBytes} 个字节`),
    tip:tx(`The English sentence has ${r.enBytes} bytes: ${r.enChars} characters of 1 byte each. After ${mergeN(r.k)} it takes ${r.en} tokens.`,`英文句子有 ${r.enBytes} 个字节：${r.enChars} 个字符，每个 1 个字节。学了 ${fint(r.k)} 次合并后，它是 ${r.en} 个 token。`)},
   {k:'zh',label:tx('Chinese tokens','中文 token 数'),v:unk?'?':fint(r.zh),sub:hid?tx(`${r.zhChars} characters`,`${r.zhChars} 个字符`):tx(`from ${r.zhBytes} bytes`,`来自 ${r.zhBytes} 个字节`),hid:unk,
    tip:unk?(pred?tx('Shown in part 3, where your prediction is checked.','在第 3 部分显示，那里会核对你的预测。'):tx('Hidden until you answer the question in part 2; part 3 shows it.','在你回答第 2 部分的问题之前先隐藏；第 3 部分会显示。'))
      :(hid?tx(`The Chinese sentence has ${r.zhChars} characters, its full stop included. After ${mergeN(r.k)} it takes ${r.zh} tokens.`,`中文句子有 ${r.zhChars} 个字符（含句号）。学了 ${fint(r.k)} 次合并后，它是 ${r.zh} 个 token。`)
      :tx(`The Chinese sentence has ${r.zhBytes} bytes: ${r.zhChars} characters of 3 bytes each. After ${mergeN(r.k)} it takes ${r.zh} tokens.`,`中文句子有 ${r.zhBytes} 个字节：${r.zhChars} 个字符，每个 3 个字节。学了 ${fint(r.k)} 次合并后，它是 ${r.zh} 个 token。`))},
   {k:'ratio',label:tx('Chinese ÷ English','中文 ÷ 英文'),v:unk?'?':r2(r.ratio),sub:tx('times as many tokens','倍的 token'),hid:unk,
    tip:unk?(pred?tx('Shown in part 3, where your prediction is checked.','在第 3 部分显示，那里会核对你的预测。'):tx('Hidden until you answer the question in part 2; part 3 shows it.','在你回答第 2 部分的问题之前先隐藏；第 3 部分会显示。'))
      :tx(`The Chinese sentence takes ${r2(r.ratio)} times as many tokens as the English one: ${r.zh} ÷ ${r.en} = ${r2(r.ratio)}. A bill priced per token and the room taken in a context window grow by the same factor.`,`中文句子的 token 数是英文的 ${r2(r.ratio)} 倍：${r.zh} ÷ ${r.en} = ${r2(r.ratio)}。按 token 计的费用和占用的上下文窗口，也是这么多倍。`)},
   {k:'vocab',label:tx('Vocabulary','词表'),v:fint(r.vocab),sub:tx(`tokens: 256 bytes + ${fint(r.k)} merges`,`个 token：256 个字节 + ${fint(r.k)} 次合并`),
    tip:tx(`The vocabulary is every token this tokenizer can produce: the 256 byte values plus one per merge, 256 + ${fint(r.k)} = ${fint(r.vocab)}. In a model with hidden size 4,096, such as Llama-3.1-8B, each token takes a row of 4,096 parameters in the embedding table and another in the LM head: 2 × ${fint(r.vocab)} × 4,096 = ${A.fmt.big(par)} parameters here.`,
           `词表是这个 tokenizer 能产出的全部 token：256 个字节值，加上每次合并一个，256 + ${fint(r.k)} = ${fint(r.vocab)}。在隐藏维度 4,096 的模型里（如 Llama-3.1-8B），每个 token 在 embedding 表里占一行 4,096 个参数，在 LM head 里还有一行：这里是 2 × ${fint(r.vocab)} × 4,096 = ${A.fmt.big(par)} 个参数。`)}];
}
function drawMetrics(){
  const host=$('#tk-metrics',app), defs=metricsDef(), sig=defs.map(m=>m.k).join();
  if(host._sig!==sig){host._sig=sig; host.dataset.n=defs.length; host.innerHTML=defs.map(m=>`<div class="metric" data-k="${m.k}" tabindex="0"><div class="k"></div><div class="v mono"></div><div class="sub"></div><div class="d mono"></div></div>`).join('');}
  defs.forEach(m=>{const el=host.querySelector(`[data-k="${m.k}"]`), d=el.querySelector('.d');
    el.querySelector('.k').textContent=m.label; el.querySelector('.v').textContent=m.v; el.querySelector('.sub').textContent=m.sub; el.dataset.tipx=m.tip; el.classList.toggle('tk-hidm',!!m.hid);
    let dt=''; const prev=shownPrev[m.k];
    if(userAct&&!m.hid&&prev!=null&&prev!==m.v){const a=numOf(prev), b=numOf(m.v); if(isFinite(a)&&isFinite(b)) dt=A.fmt.delta(a,b,{dp:Math.max(decOf(prev),decOf(m.v))})||'';}
    clearTimeout(d._t); clearTimeout(d._t2);
    if(dt){d.textContent=dt; d.classList.add('on'); d._t=setTimeout(()=>{d.classList.remove('on'); d._t2=setTimeout(()=>{d.textContent='';},400);},2600);}
    else {d.classList.remove('on'); d.textContent='';}
    shownPrev[m.k]=m.hid?null:m.v;});
}
/* Show the math: every line redoes from the numbers it shows (R6.6, review S2: the copies are whole, so the quotient is shown before
   rounding down); every term has its own tooltip (R1.4) */
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
const FW=tex=>{const [name,...rest]=FWcut(tex,[' = ',' \\approx ']); if(!rest.length) return A.ui.math(tex);
  const rel=rest[0].startsWith('=')?'=':'\\approx', rhs=rest.join(' ').slice(rel.length+1);
  const piece=x=>`<span class="tk-fp">${A.ui.math('\\mathrm{} '+x)}</span>`;
  return `<span class="tk-fwn">${A.ui.math(name+' '+rel+' \\mathrm{}')}</span><span class="tk-fwr">${FWcut(rhs,[' + ',' - ']).map(g=>`<span class="tk-fg">${FWcut(g,[' \\times ']).map(piece).join('<wbr>')}</span>`).join('<wbr>')}</span>`;};
function mathHTML(){
  const r=R(), hid=predHidden(), unk=zhUnknown(), t=T_, kS=t('k',fint(r.k));
  const cp=nt=>{const x=CTX/nt, f=Math.floor(x); return Number.isInteger(x)?`${t('ctx','131,072')} ÷ ${nt} = <b>${fint(f)}</b>`:tx(`${t('ctx','131,072')} ÷ ${nt} = ${q1(x)}, so <b>${fint(f)}</b> whole copies`,`${t('ctx','131,072')} ÷ ${nt} = ${q1(x)}，也就是 <b>${fint(f)}</b> 句完整的`);};
  /* the owner's decisions of 2026-10-09 (H1, H2): a few short blocks, each a lead line in words, the formula typeset (A.ui.math), then the
     numbers of the current setting; no run-on “…; …” lines */
  const FL=f=>{const h=FW(f); return `<div class="${h.includes('tk-fwn')?'tk-fw ':''}tk-math-line">${h}</div>`;};
  const MB=(lead,f,body)=>`<div class="tk-mb"><div class="tk-mlead">${lead}</div>${[].concat(f||[]).map(FL).join('')}${body}</div>`;
  const rows=[];
  rows.push(MB(tx('The tokenizer starts from bytes. UTF-8 stores an English letter in 1 byte and a Chinese character in 3.','tokenizer 从字节开始。UTF-8 里一个英文字母占 1 个字节，一个汉字占 3 个。'),
    tx('\\text{bytes} = \\text{characters} \\times \\text{bytes per character}','\\text{字节数} = \\text{字符数} \\times \\text{每个字符的字节数}'),
    `<p>${tx('English sentence','英文句子')} = ${fint(r.enChars)} ${tx('characters','个字符')} × 1 = <b>${r.enBytes}</b></p>`+
    (hid?`<p class="muted">${tx('The Chinese sentence’s bytes appear in part 3, where they answer your prediction.','中文句子的字节数在第 3 部分出现，那里揭晓你的预测。')}</p>`
      :`<p>${tx('Chinese sentence','中文句子')} = ${fint(r.zhChars)} ${tx('characters','个字符')} × 3 = <b>${r.zhBytes}</b></p>`)));
  rows.push(MB(`${tx('Each learned merge joins two neighbouring tokens into one, in the order the merges were learned. After','学到的每一次合并，都把相邻的两个 token 合成一个，按学到的顺序套用。学了')} ${kS} ${tx('merges:','次合并之后：')}`,null,
    `<p>${tx('English','英文')}: ${hid?'':`${r.enBytes} → `}<b>${r.en}</b> ${tx('tokens','个 token')}</p><p>${tx('Chinese','中文')}: ${unk?tx('shown in part 3','第 3 部分显示'):`${hid?'':`${r.zhBytes} → `}<b>${r.zh}</b> ${tx('tokens','个 token')}`}</p>`));
  if(!unk) rows.push(MB(tx('How many times as many tokens the Chinese sentence takes as the English one.','中文句子的 token 数是英文句子的几倍。'),
    tx('\\text{Chinese} \\div \\text{English} = \\frac{\\text{tokens of the Chinese sentence}}{\\text{tokens of the English sentence}}','\\text{中文} \\div \\text{英文} = \\frac{\\text{中文句子的 token 数}}{\\text{英文句子的 token 数}}'),
    `<p>${tx('Chinese ÷ English','中文 ÷ 英文')} = ${r.zh} ÷ ${r.en} = <b>${r2(r.ratio)}</b></p>`));
  rows.push(MB(tx('A context window of 131,072 tokens holds only whole copies of a sentence.','131,072 个 token 的上下文窗口能装下这句话几遍，只算完整的。'),
    tx('\\text{copies} = \\frac{\\text{context window}}{\\text{tokens per sentence}}','\\text{句数} = \\frac{\\text{上下文窗口}}{\\text{每句的 token 数}}'),
    `<p>${tx('English','英文')}: ${cp(r.en)}</p>${unk?'':`<p>${tx('Chinese','中文')}: ${cp(r.zh)}</p>`}`));
  rows.push(MB(tx('The vocabulary is the 256 single bytes plus one token per merge. The embedding table and the LM head each store one vector of d numbers per vocabulary token.','词表是 256 个单字节，再加上每次合并得到的一个 token。embedding 表和 LM head 给词表里每个 token 各存一个 d 维的向量。'),
    [tx('\\text{vocabulary} = 256 + \\text{merges}','\\text{词表} = 256 + \\text{合并次数}'),tx('\\text{parameters} = 2 \\times \\text{vocabulary} \\times d','\\text{参数} = 2 \\times \\text{词表} \\times d')],
    `<p>${tx('Vocabulary','词表')} = 256 + ${kS} = <b>${fint(r.vocab)}</b></p><p>${tx('embedding table + LM head','embedding 表 + LM head')} = 2 × ${fint(r.vocab)} × ${t('d','4,096')} = <b>${fint(2*r.vocab*4096)}</b> ${tx('parameters','个参数')}${tx(' (d = 4,096)','（d = 4,096）')}</p>`));
  return rows.join('');
}
function render(){
  if(!app) return;
  if(mode==='challenge'){if(!CH.mixes.includes(S.mix)) S.mix=CH.start.mix; if(!CH.ks.includes(S.k)) S.k=CH.start.k;}
  if(preLock()){S.mix=BASE.mix; S.k=step===1?0:BASE.k;}
  if(mode==='guided'){const s=STEPS[step]; if(s&&s.expect&&s.expect()) observed[step]=true;}
  drawHead();
  const gr=$('#tk-graph',app); gr.dataset.focus=mode==='guided'?(STEPS[step].id==='remember'?(remView?REM[remView].focus:''):(STEPS[step].focus||'')):'';
  drawStrips(!!app._drawn); app._drawn=true; drawMerges(); drawCost(); drawMetrics();
  const lg=$('#tk-legend',app), lh=legendHTML(); if(lg._h!==lh){lg.innerHTML=lh; lg._h=lh;}
  const mb=$('#tk-mathbody',app), mh=mathHTML(); if(mb._h!==mh){mb.innerHTML=mh; mb._h=mh;}
  syncControls(); updateRail();
  A.hash.write(token()); A.dock.refresh(); announce();
}
function syncControls(){
  const r=R();
  $$('input[data-key="k"]',app).forEach(i=>{if(+i.value!==S.k) i.value=S.k; A.ui.setRange(i); const w=i.closest('.tk-kctl'), o=w.querySelector('output'); o.textContent=fint(S.k); i.setAttribute('aria-valuetext',mergeN(S.k));
    w.querySelector('[data-act="kminus"]').disabled=S.k<=0; w.querySelector('[data-act="kplus"]').disabled=S.k>=K_MAX;
    w.querySelector('.tk-knote').textContent=tx(`Vocabulary: 256 bytes + ${fint(S.k)} merges = ${fint(r.vocab)} tokens.`,`词表：256 个字节 + ${fint(S.k)} 次合并 = ${fint(r.vocab)} 个 token。`);});
  $$('.seg-ctl[data-key="mix"]',app).forEach(s=>A.ui.setPressed(s,S.mix));
  $$('.seg-ctl[data-key="mix"] button',app).forEach(b=>{b.dataset.tipx=tx(`${mixName(b.dataset.v)}: ${L(MIX[b.dataset.v].d)}, ${shareS(b.dataset.v)} English by bytes.`,`${mixName(b.dataset.v)}：${L(MIX[b.dataset.v].d)}，按字节算 ${shareS(b.dataset.v)} 是英文。`);});
  const mn=$('.tk-mixnote',app); if(mn) mn.textContent=tx(`${mixName(S.mix)}: ${L(MIX[S.mix].d)}, ${shareS(S.mix)} English by bytes.`,`${mixName(S.mix)}：${L(MIX[S.mix].d)}，按字节算 ${shareS(S.mix)} 是英文。`);
  const lock=mode==='challenge', pl=preLock(), c=$('#tk-controls',app); c.classList.toggle('locked',lock||pl);
  const ln=$('#tk-lock',app); ln.hidden=!(lock||pl);
  if(lock) ln.textContent=tx('In the challenge you change only the training text and the merges, in the task card. Everything else is fixed: the new sentence pair and this lab’s BPE rules.','挑战中只能在任务卡里改训练文本和合并次数。其余都固定：新的一对句子，以及这个 Lab 的 BPE 规则。');
  else if(pl) ln.textContent=tx('The training text and the merges can change from part 3 on, where your prediction is checked.','从第 3 部分起才能改训练文本和合并次数，那里会核对你的预测。');
  ['#tk-ctl-mix-wrap','#tk-ctl-k-wrap'].forEach(s=>{const el=$(s,app); if(el) el.hidden=lock||pl;});
  const sh=$('[data-share]',app); if(sh) sh.dataset.tipx=tx('Copies a link that opens this lab with your current settings.','复制一个链接，打开这个 Lab 时带上你现在的设置。');
}
function announce(){clearTimeout(annT); annT=setTimeout(()=>{if(!app) return; const r=R();
  A.announce(zhUnknown()?tx(`No merges: English ${r.en} tokens; the Chinese count ${pred?'is shown in part 3':'waits for your prediction'}.`,`没有合并：英文 ${r.en} 个 token；中文的数字${pred?'在第 3 部分显示':'等你预测之后再显示'}。`)
    :tx(`${mergeN(r.k)} on ${mixPh(S.mix)}: English ${r.en} tokens, Chinese ${r.zh}.`,`在${mixPh(S.mix)}上学了 ${fint(r.k)} 次合并：英文 ${r.en} 个 token，中文 ${r.zh} 个。`));},400);}

/* =====================================================================
   state changes, parts, modes, hash
   ===================================================================== */
function setState(p){
  if(preLock()) return render();
  Object.assign(S,p);
  if(!MIX[S.mix]) S.mix=BASE.mix; S.k=Math.max(0,Math.min(K_MAX,Math.round(S.k)||0));
  if(remView&&(S.mix!==REM[remView].s.mix||S.k!==REM[remView].s.k)) remView=null;   // part 7: a change in Settings is no longer that point's example
  render();
}
function byLearner(fn){userAct=true; if(mode==='sandbox') sbTouched=true; try{fn();} finally{userAct=false;}}
function enterStep(i){
  wantObs=false; step=Math.max(0,Math.min(STEPS.length-1,i));
  const s=STEPS[step]; if(s.enter) Object.assign(S,s.enter());
  if(s.id==='bytes') hexOn=false;
  if(s.id==='mix') lvTried=new Set();
  remView=s.id==='remember'?'bytes':null;
  const st=$('#tk-stage',app); if(st) st.scrollTop=0;
  renderRail();
  const tg={bytes:'#tk-ic-hex',merges:'#tk-ic-k',mix:'.tk-lv',cost:'#tk-ic-mix'}[s.id];
  if(tg){const el=$(tg,app); if(el){el.classList.remove('cue'); void el.offsetWidth; el.classList.add('cue');}}
}
function revealStep(){A.ui.reveal($('#tk-rail .step.current',app)||$('#tk-rail',app));}
function revealBox(box){
  if(!box||box.hidden) return;
  const bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
  const top=(bar?bar.offsetHeight:56)+(dk&&!dk.hidden?dk.offsetHeight:(innerWidth<=760?50:0))+12, r=box.getBoundingClientRect();
  if(r.height>innerHeight-top-16){if(r.top>=top&&r.top<=innerHeight-200) return; A.ui.reveal(box); return;}
  if(r.top>=top&&r.bottom<=innerHeight-8) return;
  const dy=r.bottom>innerHeight-8?r.bottom-innerHeight+16:r.top-top;
  window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});}
/* after a click in the rail, keep the sticky stage's head below the top bar (as labs/prefill-decode.js does) */
let stageT=0;
const stageStop=()=>{clearTimeout(stageT); stageT=0;};
function keepStage(){stageStop();
  let lastY=null, waited=0;
  const tick=()=>{stageT=0; if(!app) return;
    if(scrollY!==lastY){if(waited>=1500) return; lastY=scrollY; waited+=120; stageT=setTimeout(tick,120); return;}
    const st=$('#tk-stage',app); if(!st||innerWidth<=760||getComputedStyle(st).position!=='sticky') return;
    const card=$('#tk-rail .step.current',app)||$('#tk-rail .railcard',app)||$('#tk-rail',app);
    const want=parseFloat(getComputedStyle(st).top)||70, r=st.getBoundingClientRect(), c=card.getBoundingClientRect();
    let dy=r.top-want; if(dy>=-2||c.bottom<want) return;
    dy=Math.max(dy,c.top-(innerHeight-160)); if(dy<-2) window.scrollBy({top:dy,behavior:A.reduceMotion()?'auto':'smooth'});};
  stageT=setTimeout(tick,A.reduceMotion()?30:150);}
function revealObs(){if(!app) return; if(mode==='guided') revealBox($('#tk-rail [data-obs]',app)); else if(mode==='sandbox') revealBox($('#tk-insight',app)); else revealBox($('#tk-ch-res',app));}
const revealSoon=()=>requestAnimationFrame(revealObs);
function setMode(m){
  if(m===mode) return;
  const was=mode; mode=m;
  if(was==='challenge'&&saved){S={...saved}; saved=null;}
  if(m==='challenge'){saved={...S}; chSubmitted=false; chReveal=false; S={...CH.start};}
  if(m==='sandbox') sbTouched=false;
  A.shell.setMode(app,m,DESCS());
  const st=$('#tk-stage',app); if(st) st.scrollTop=0;
  if(m==='guided'){enterStep(step); return;}
  renderRail();
}
/* part 7: a “Show in the chart” button puts its part's picture on the stage (review A8) */
function setRem(k){if(!REM[k]) return; remView=k; S={...S,...REM[k].s}; render(); const el=$({strips:'#tk-strips',merges:'#tk-merges',cost:'#tk-cost'}[REM[k].focus],app);
  if(el){if(innerWidth>760) A.ui.flash(el); else A.ui.reveal(el);}}   // on a phone the chart is above the card: bring it into view, as Sampling does
const token=()=>`${S.mix}_${S.k}`;
function parse(tok){
  const m=String(tok||'').match(/^(en|tile|bal|zh)_(\d{1,3})$/); if(!m) return null;
  const k=+m[2]; if(k<0||k>K_MAX) return null;
  return {mix:m[1],k};
}

/* =====================================================================
   interview drill: seeded, from the engine. A word of the lab's English sentence and the merges among the first N whose tokens appear
   inside it, listed in the question, so it can be solved from the page (review A11)
   ===================================================================== */
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const DRILL_W=[' morning',' books',' read',' like'], DRILL_K=[40,60,100];
function makeDrill(seed){const rnd=mulberry(seed*7919+17), pick=a=>a[Math.floor(rnd()*a.length)], w=pick(DRILL_W), k=pick(DRILL_K);
  const e=enc('tile',w), m=model('tile'), wb=utf8(w).join('.')+'.', steps=[], cand=[];
  for(let i=0;i<k;i++){if(e.used[i]) steps.push(i); if(('.'+wb).includes('.'+m.bytes[256+i].join('.')+'.')) cand.push(i);}
  return {seed,w,k,bytes:e.bytes,steps,cand,final:e.at[k].map(id=>piece(m.bytes[id]).t),snaps:steps.map(i=>e.at[i+1].map(id=>piece(m.bytes[id]).t))};}
function renderDrill(){
  if(!app) return; const d=makeDrill(drillSeed), m=model('tile'), qd=$('#tk-drill-q',app), a=$('#tk-drill-a',app), w=d.w.replace(/ /g,'·');
  const mg=i=>{const [x,y]=m.merges[i]; return `<code>#${i+1} ${esc(piece(m.bytes[x]).t)} + ${esc(piece(m.bytes[y]).t)} → ${esc(piece(m.bytes[256+i]).t)}</code>`;};
  qd.innerHTML=tx(`The initial tokenizer learned its merges from mostly English text. Among its first <b>${d.k}</b> merges, these build pieces of “<b>${esc(w)}</b>” (${d.bytes} bytes; its leading space is shown as ·): ${d.cand.map(mg).join('; ')}. Apply the first ${d.k} merges in order to the word’s bytes: which of these change it, and which tokens are left? You can check the answer in Settings: set the merges and watch the English strip.`,
                  `初始设置的 tokenizer 从以英文为主的文本里学合并。它的前 <b>${d.k}</b> 次合并里，下面这些能拼出“<b>${esc(w)}</b>”的一部分（${d.bytes} 个字节，前面的空格显示为 ·）：${d.cand.map(mg).join('；')}。把前 ${d.k} 次合并按顺序套用到这个词的字节上：其中哪几次改变了它？最后剩下哪些 token？可以在“自由调参”里设好合并次数，看英文那一条来核对。`);
  a.hidden=!drillOpen; if(!drillOpen) return;
  const tk=arr=>arr.map(x=>`“${esc(x)}”`).join(' ');
  const lines=d.steps.map((i,j)=>`<li>${mg(i)}${tx(', leaving ','，剩下 ')}${tk(d.snaps[j])}</li>`).join('');
  const skipped=d.cand.filter(i=>!d.steps.includes(i));
  a.innerHTML=tx(`<p><b>${d.final.length} token${d.final.length===1?'':'s'}: ${tk(d.final)}.</b></p><p>Start from the ${d.bytes} bytes, one token each, and apply the merges in the order learned:</p><ol>${lines}</ol>`+(skipped.length?`<p>${skipped.map(i=>`#${i+1}`).join(', ')} ${skipped.length===1?'does':'do'} not apply: by ${skipped.length===1?'its':'their'} turn, the letters ${skipped.length===1?'it needs are':'they need are'} already inside other tokens.</p>`:''),
                 `<p><b>${d.final.length} 个 token：${tk(d.final)}。</b></p><p>从 ${d.bytes} 个字节开始，每个字节一个 token，按学到的顺序套用合并：</p><ol>${lines}</ol>`+(skipped.length?`<p>${skipped.map(i=>`#${i+1}`).join('、')}用不上：轮到它的时候，它要的那几个字母已经在别的 token 里了。</p>`:''));
}

/* =====================================================================
   term tooltips and jumps (every data-jump key is handled by both jumpAct and jump)
   ===================================================================== */
const TERM={
 k:()=>tx(`${fint(S.k)} is the number of merges learned from the training text; each one adds a token to the vocabulary.`,`${fint(S.k)} 是从训练文本学到的合并次数；每次合并都给词表添一个 token。`),
 ctx:()=>tx('131,072 tokens is Llama-3.1-8B’s context window, the max_position_embeddings in its config.json.','131,072 个 token 是 Llama-3.1-8B 的上下文窗口，即它 config.json 里的 max_position_embeddings。'),
 d:()=>tx('d is the model’s hidden size: every row of the embedding table and of the LM head has d numbers.','d 是模型的隐藏维度：embedding 表和 LM head 的每一行都是 d 维。')
};
const termTipHTML=k=>TERM[k]?`<div>${esc(TERM[k]())}</div><div class="act">${esc(jumpAct(k))}</div>`:'';
function jumpAct(k){
  if(k==='controls') return tx('Click to jump to Settings.','点击跳到“自由调参”。');
  if(k==='math') return tx('Click to open “Show the math”.','点击展开“展开公式”。');
  if(k==='strips') return tx('Click to scroll to the two strips in the chart.','点击滚到图里的两条长条。');
  if(k==='merges') return tx('Click to scroll to the merges in the chart.','点击滚到图里的合并。');
  if(k==='cost') return tx('Click to scroll to the cost rows in the chart.','点击滚到图里的成本条。');
  if(k==='ctx') return tx('Fixed in this lab: it only converts token counts into copies, and every comparison between the languages is the same for any window.','这个 Lab 里固定不变：它只用来把 token 数换算成句数，两种语言之间的比较换成任何窗口都一样。');
  if(k==='d') return tx('Fixed in this lab at 4,096 (Llama-3.1-8B). It only turns the vocabulary size into parameters of the embedding table and LM head; no token count depends on it.','这个 Lab 里固定为 4,096（Llama-3.1-8B）。它只用来把词表大小换算成 embedding 表和 LM head 的参数，任何 token 数都和它无关。');
  if(k==='k'){
    if(preLock()) return tx('The merges can change from part 3 on, where your prediction is checked.','从第 3 部分起才能改合并次数，那里会核对你的预测。');
    if(mode==='challenge') return tx('Click to jump to “Merges learned” in the task card.','点击跳到任务卡里的“已学的合并次数”。');
    return tx('Click to jump to “Merges learned”.','点击跳到“已学的合并次数”。');}
  return '';
}
function jump(k,src){
  if(!app) return;
  if(k==='math'){const d=$('#tk-math',app); d.open=true; A.tip.hide(); d.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(d); return;}
  if(k==='strips'||k==='merges'||k==='cost'){const el=$({strips:'#tk-strips',merges:'#tk-merges',cost:'#tk-cost'}[k],app); A.tip.hide(); if(el){el.scrollIntoView({block:'center',behavior:A.reduceMotion()?'auto':'smooth'}); A.ui.flash(el);} return;}
  if(k==='controls'){A.tip.hide(); A.ui.jumpToEl($('#tk-controls',app)); return;}
  if(k==='ctx'||k==='d'){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
  if(k==='k'){
    if(preLock()){if(src) A.tip.show(src,`<div>${esc(jumpAct(k))}</div>`); return;}
    const el=mode==='challenge'?$('#tk-ic-chk',app):($('#tk-ic-k',app)||$('#tk-ctl-k',app)); if(el) A.ui.jumpToEl(el); return;}
}

/* =====================================================================
   events
   ===================================================================== */
function act(a,btn){
  if(a==='reset'){bySettings(btn,()=>{if(mode==='challenge') setState({...CH.start}); else setState({...BASE});}); return;}
  if(a==='submit'){chSubmitted=true; if(chPoint(S.mix,S.k).ok) chReveal=true; renderRail(); revealSoon(); return;}
  if(a==='reveal'){chReveal=true; render(); return;}
  if(a==='kminus'||a==='kplus'){A.tip.quiet(); const d=a==='kplus'?1:-1, inRail=!!btn.closest('#tk-rail');
    const f=()=>setState({k:S.k+d}); if(inRail){wantObs=true; byLearner(f); wantObs=false;} else bySettings(btn,f); return;}
  if(a==='drill-new'){drillSeed++; drillOpen=false; renderDrill(); return;}
  if(a==='drill-reveal'){drillOpen=true; renderDrill(); return;}
  if(a==='copy'){const pre=$('#tk-code',app), done=()=>flashBtn(btn,tx('Copied','已复制'),A.t('copyCode'));
    const sel=()=>{const rg=document.createRange(); rg.selectNodeContents(pre); const s=getSelection(); s.removeAllRanges(); s.addRange(rg); flashBtn(btn,tx('Selected. Press Ctrl+C','已选中，按 Ctrl+C 复制'),A.t('copyCode'));};
    try{if(navigator.clipboard&&navigator.clipboard.writeText) navigator.clipboard.writeText(pre.textContent).then(done,sel); else sel();}catch(err){sel();}}
}
function flashBtn(b,txt,back){b.textContent=txt; clearTimeout(b._t); b._t=setTimeout(()=>{b.textContent=back;},1600);}
/* a change under Settings never moves the page (R3.2): scroll anchoring is paused and the control is put back where it was */
function holdPlace(el,f){
  if(!el||!el.isConnected){f(); return;}
  const de=document.documentElement, was=de.style.overflowAnchor, y0=el.getBoundingClientRect().top;
  de.style.overflowAnchor='none';
  try{f();}
  finally{const dy=el.isConnected?el.getBoundingClientRect().top-y0:0; if(Math.abs(dy)>=0.5) window.scrollBy(0,dy); de.style.overflowAnchor=was;}
}
const bySettings=(el,f)=>el&&el.closest('#tk-controls')?holdPlace(el,()=>byLearner(f)):byLearner(f);
function bind(el){
  el.addEventListener('input',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k!=='k') return; bySettings(e.target,()=>setState({k:+e.target.value}));});
  el.addEventListener('change',e=>{const k=e.target.dataset&&e.target.dataset.key; if(k&&e.target.closest('#tk-rail')){wantObs=true; revealSoon(); wantObs=false; keepStage();}});
  el.addEventListener('click',e=>{
    const b=e.target.closest('button'); if(!b||!el.contains(b)) return; const d=b.dataset;
    if(!d.jump&&b.closest('#tk-rail')&&!b.closest('[data-exits]')&&!(d.act&&d.act.startsWith('k'))) keepStage();
    if(d.go!=null){enterStep(+d.go); revealStep(); return;}
    if(d.nav==='next'||d.nav==='skip'){enterStep(step+1); revealStep(); return;}
    if(d.nav==='back'){enterStep(step-1); revealStep(); return;}
    if(d.nav==='showme'){const s=STEPS[step]; if(s.showMe){wantObs=true; if(s.id==='mix') lvTried.add(s.showMe.mix); byLearner(()=>setState(s.showMe)); revealSoon(); wantObs=false;} return;}
    if(d.pred){pred=d.pred; renderRail(); const fb=$(`#tk-rail [data-pred="${pred}"]`,app); if(fb) fb.focus({preventScroll:true}); return;}
    if(d.rem){setRem(d.rem); return;}
    if(d.lever!=null&&MIX[d.lever]){lvTried.add(d.lever); byLearner(()=>setState({mix:d.lever})); revealSoon(); return;}
    const seg=b.closest('.seg-ctl');
    if(seg&&d.v!=null){const key=seg.dataset.key;
      if(key==='hex'){hexOn=d.v==='1'; if(mode==='guided'&&STEPS[step].id==='bytes') observed[step]=true; render(); return;}
      if(key==='mix'){bySettings(b,()=>setState({mix:d.v})); if(b.closest('#tk-rail')) revealSoon(); return;}
      if(key==='chmix'){byLearner(()=>setState({mix:d.v})); return;}
      if(key==='chk'){byLearner(()=>setState({k:+d.v})); return;}
      return;}
    if(d.preset!=null){const p=PRESETS[+d.preset]; byLearner(()=>setState({...p.s})); requestAnimationFrame(()=>{if(app) revealBox($('#tk-insight',app));}); return;}
    if(d.modeGo){setMode(d.modeGo); A.ui.reveal($('#tk-rail',app)); return;}
    if(d.act) act(d.act,b);
  });
  ['wheel','touchmove'].forEach(k=>el.addEventListener(k,stageStop,{passive:true}));
}

/* =====================================================================
   registration
   ===================================================================== */
let fromSnap=false;   // the state was just put back from a saved snapshot: mount keeps S as it was (round 3 #6)
A.registerLab({
  id:ID,
  /* round 3 #6: what the learner did here, kept across a reload. The kit saves snap() and hands it to restore() before the first mount */
  snap:()=>({v:1,mode,step,S:{...S},pred,observed:{...observed},saved:saved?{...saved}:null,chSubmitted,chReveal,sbTouched,hexOn,lvTried:[...lvTried],remView,
    drillSeed,drillOpen,passed:mode==='challenge'&&chSubmitted&&chPoint(S.mix,S.k).ok}),
  restore(o){if(!o||o.v!==1) return;
    const cfg=x=>x&&typeof x==='object'?parse(`${x.mix}_${x.k}`):null, p=cfg(o.S); if(!p) return;
    mode=['guided','sandbox','challenge'].includes(o.mode)?o.mode:'guided';
    step=Number.isInteger(o.step)&&o.step>=0&&o.step<STEPS.length?o.step:0; S={...p};
    pred=PRED.some(x=>x.v===o.pred)?o.pred:null; observed={};
    if(o.observed&&typeof o.observed==='object') for(const k in o.observed) if(/^\d+$/.test(k)&&+k<STEPS.length&&o.observed[k]===true) observed[k]=true;
    const sv=mode==='challenge'?cfg(o.saved):null; saved=sv?{...sv}:null;
    chSubmitted=mode==='challenge'&&o.chSubmitted===true; chReveal=chSubmitted&&o.chReveal===true; sbTouched=o.sbTouched===true; hexOn=o.hexOn===true;
    lvTried=new Set(Array.isArray(o.lvTried)?o.lvTried.filter(x=>typeof x==='string'&&Object.prototype.hasOwnProperty.call(MIX,x)):[]);
    remView=typeof o.remView==='string'&&Object.prototype.hasOwnProperty.call(REM,o.remView)?o.remView:null;
    drillSeed=Number.isInteger(o.drillSeed)&&o.drillSeed>=0&&o.drillSeed<1e6?o.drillSeed:1; drillOpen=o.drillOpen===true; fromSnap=true;},
  mount(el,tok){
    wantObs=false; root=el;
    const p=tok&&!fromSnap?parse(tok):null; if(p){S={...p}; mode='sandbox'; sbTouched=true;}
    if(mode==='challenge'&&!saved) saved={...BASE};
    el.innerHTML=viewHTML(); app=$('#tk-app',el);
    A.ui.initRanges(app);
    A.shell.bindHeader(app,{id:ID,onMode:m=>{setMode(m); A.ui.reveal($('#tk-rail',app)); keepStage();},token});
    bind(app);
    renderDrill();
    if(mode==='guided'&&!p&&!fromSnap){const s=STEPS[step]; if(s.enter&&step===0) Object.assign(S,s.enter());}
    fromSnap=false;
    const rail=$('#tk-rail',app); rail.innerHTML=railFull(); A.ui.initRanges(rail);
    render();
  },
  leave(){clearTimeout(annT); stageStop(); root=null; app=null;},
  onLang(){if(root){const d=$('#tk-math',app), open=d&&d.open; this.mount(root,null); if(open) $('#tk-math',app).open=true;}},
  onResize(){if(app){drawStrips(false);}},
  termTip(k){return termTipHTML(k);},
  jump, jumpAct,
  dock:{
    anchor:()=>app&&$('#tk-strips',app), bench:()=>app&&$('.bench',app), target:()=>app&&$('#tk-graph',app),
    render({status,vis,read}){if(!app) return; const r=R(), s=statusOf(), n=Math.max(r.enBytes,r.zhBytes), unk=zhUnknown();
      status.className='status '+s.cls; status.textContent=s.short;
      vis.innerHTML=`<div class="tk-dk"><i class="tk-dk-en" style="width:${(r.en/n*100).toFixed(1)}%"></i><i class="tk-dk-zh" style="width:${unk?0:(r.zh/n*100).toFixed(1)}%"></i></div>`;
      /* parts 4 and 5 on a phone: the ticks are above the card, so the bar also says how many merges built Chinese tokens (review A28) */
      const zcS=mode==='guided'&&['merges','mix'].includes(STEPS[step].id);
      read.innerHTML=esc(unk?(pred?tx(`No merges: English ${r.en} tokens; Chinese: shown in part 3`,`没有合并：英文 ${r.en} 个 token；中文：第 3 部分显示`):tx(`No merges: English ${r.en} tokens; Chinese: predict first`,`没有合并：英文 ${r.en} 个 token；中文：先预测`))
        :mode==='challenge'?tx(`New pair, ${mergeN(r.k)} on ${mixPh(S.mix)}: English ${r.en} tokens, Chinese ${r.zh}`,`新的一对句子，在${mixPh(S.mix)}上学 ${fint(r.k)} 次合并：英文 ${r.en} 个 token，中文 ${r.zh} 个`)
        :zcS?tx(`${mixName(S.mix)}, ${mergeN(r.k)} (${r.zhMerges===1?'1 of them built a Chinese token':`${fint(r.zhMerges)} of them built Chinese tokens`}): English ${r.en} tokens, Chinese ${r.zh}`,`${mixName(S.mix)}，${fint(r.k)} 次合并（${fint(r.zhMerges)} 次造出中文 token）：英文 ${r.en} 个 token，中文 ${r.zh} 个`)
        :tx(`${mergeN(r.k)} on ${mixPh(S.mix)}: English ${r.en} tokens, Chinese ${r.zh}`,`在${mixPh(S.mix)}上学 ${fint(r.k)} 次合并：英文 ${r.en} 个 token，中文 ${r.zh} 个`));}
  }
});

/* =====================================================================
   read-only hook for tests
   ===================================================================== */
const deep=o=>JSON.parse(JSON.stringify(o));
window.__tk=Object.freeze({
  compute:(mix,k,pair)=>deep(compute(mix,k,pair)), tokens:(mix,text,k)=>{const e=enc(mix,text), m=model(mix); return e.at[Math.min(k,e.at.length-1)].map(id=>m.bytes[id].slice());},
  merges:mix=>model(mix).merges.map(x=>x.slice()), zhPiece:mix=>model(mix).zhPiece.slice(),
  chPoint:(mix,k)=>deep(chPoint(mix,k)), chAll:()=>deep(chAll()), chMinK, chBest:()=>deep(chBest()), chTrend:()=>deep(chTrend()), sameMerges, wholeAt, hexWhere, q1,
  drill:s=>deep(makeDrill(s)), exits:()=>{const x=EXITS(); return deep({explain:x.explain.map(q=>({q:q.q,o:q.o,a:q.a})),transfer:x.transfer.map(q=>({q:q.q,o:q.o,a:q.a,why:q.why}))});},
  corpus:()=>({en:EN_TEXT,zh:TK_ZH}),
  consts:Object.freeze({K_MAX,CTX,SCEN_K,BASE:{...BASE},MIXES:MIXES.map(m=>({id:m.id,en:m.en,zh:m.zh})),PAIRS:deep(PAIRS),CH:deep(CH),PRESETS:deep(PRESETS.map(p=>({id:p.id,...p.s}))),STEP_IDS:STEPS.map(s=>s.id),PRED_RIGHT,CODE}),
  state:()=>({...S,mode,step,pred,observed:{...observed},chSubmitted,chReveal,hexOn,lvTried:[...lvTried],remView,stageBusy:stageT!==0,cw:cwLast})
});
})();
