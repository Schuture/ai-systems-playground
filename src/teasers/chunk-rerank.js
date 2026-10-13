/* Home-tile teaser for the chunk-rerank lab: one control (add the fixes one after another: half-chunk overlap, then rerank on vector
   search), the lab's 20 questions as squares (green: the one chunk kept holds the whole answer; red: the cut splits the answer, so no
   chunk holds it; outlined: the answer is whole but ranked below first). Keep 1 chunk of 64 tokens throughout. The tile stops at
   rerank on vector search: the misses that remain are answers vector search does not hand to the reranker, which is the lab's
   lesson; the challenge's answer (hybrid search + rerank) is left to the lab. The numbers come from the lab's own engine
   (window.__ckr), so tile and lab agree. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const OPTS=[{ov:0,ret:'vec',rr:0},{ov:2,ret:'vec',rr:0},{ov:2,ret:'vec',rr:1}];
const LAB=[{en:'Initial',zh:'初始'},{en:'+ overlap',zh:'+ 重叠'},{en:'+ rerank',zh:'+ rerank'}];
const COND=[{en:'Vector search, 64-token chunks, no overlap, keep 1.',zh:'向量检索，每块 64 个 token，不重叠，取 1 块。'},
  {en:'Vector search, 64-token chunks overlapping by half, keep 1.',zh:'向量检索，每块 64 个 token，重叠半块，取 1 块。'},
  {en:'Vector search plus rerank, 64-token chunks overlapping by half, keep 1.',zh:'向量检索加 rerank，每块 64 个 token，重叠半块，取 1 块。'}];
let si=0;
const T={render(el){
  const E=window.__ckr; if(!E){el.innerHTML=''; return;}
  const lang=A.lang()==='zh'?'zh':'en', o=OPTS[si], R=E.setup(lang,64,o.ov), key=o.ret+(o.rr?1:0), NC=E.consts.NCAND;
  const st=R.Q.map(q=>!q.right.length?'cut':(q.rk[key]===0?'ok':'low')), ok=st.filter(x=>x==='ok').length, cut=st.filter(x=>x==='cut').length, low=st.length-ok-cut;
  const outPool=R.Q.filter(q=>q.right.length&&q.rk[key]!==0&&!(q.rk[o.ret+'0']>=0&&q.rk[o.ret+'0']<NC)).length;
  const rms=E.measure(lang,{size:64,ov:2,ret:'vec',rr:1,k:1}).rerankMs;
  const W=300, H=64, cw=28, gap=2;
  let h=''; st.forEach((x,i)=>{const r=Math.floor(i/10), c=i%10; h+=`<rect class="tz-ckr-q ${x}" x="${(c*(cw+gap)+1).toFixed(1)}" y="${(r*(cw+gap)+2).toFixed(1)}" width="${cw}" height="${cw}" rx="4"/>`;});
  const read=tx(`${ok} of the 20 answers reach the model.`,`20 个答案里有 ${ok} 个送到模型面前。`);
  const why=tx(`Cut apart: ${cut}. Ranked too low: ${low}.`,`被切开的：${cut} 个。排名太低的：${low} 个。`), more=
    (o.rr&&low?(outPool===low?tx(`Rerank cannot lift these: vector search does not put them among the ${NC} chunks it hands to rerank.`,`rerank 也救不回来：向量检索没把它们放进交给 rerank 的前 ${NC} 块。`)
      :tx(`Of these, ${outPool} are not among the ${NC} chunks vector search hands to rerank.`,`其中 ${outPool} 个不在向量检索交给 rerank 的前 ${NC} 块里。`)):'');
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-ckr-s',key:'tzs',label:tx('Add a fix','加上一项改进'),options:LAB.map((l,i)=>[String(i),A.L(l)]),value:String(si)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(tx(`One square per question: ${ok} green, ${cut} red, ${low} outlined.`,`每个问题一个方块：绿色 ${ok} 个，红色 ${cut} 个，空心 ${low} 个。`))}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-ckr-swok"></i>${tx('answer reaches the model','答案送到模型面前')}</span><span><i class="sw tz-ckr-swcut"></i>${tx('cut apart','被切开')}</span><span><i class="sw tz-ckr-swlow"></i>${tx('ranked too low','排名太低')}</span></div>`+
    `<p class="tz-read">${esc(A.L(COND[si]))}<br><b>${esc(read)}</b></p><p class="tz-read">${esc(why)}</p>`+(more?`<p class="tz-read">${esc(more)}</p>`:'')+
    `<p class="tz-note">${esc(tx('Toy run: 20 questions about the help center of a made-up smart-home company, its pages cut into 64-token chunks (a word in English, a character in Chinese).','玩具实跑：一家虚构智能家居公司的帮助中心，20 个问题；每页切成 64 个 token 一块（英文一个词、中文一个字算一个 token）。'))}</p>`+
    `<p class="tz-note">${esc(tx('Each square is one question, and one chunk is kept for it. Vector search uses a small hand-written encoder. Overlap of half a chunk keeps every answer of up to 32 tokens whole.','每个方块是一个问题，每个问题只取一块。向量检索用一个手写的小编码器。重叠半块时，32 个 token 以内的答案都不会被切开。'))}</p>`+
    `<p class="tz-note">${esc(tx(`Rerank is a stand-in for a cross-encoder: it rescores the first ${NC} chunks vector search returns, about ${Math.round(rms)} ms per question (the lab’s own assumption).`,`rerank 是 cross-encoder 的替身：给向量检索返回的前 ${NC} 块重新打分，每个问题约 ${Math.round(rms)} 毫秒（本 Lab 的假设）。`))}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzs"] button').forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('chunk-rerank',T);
})();
