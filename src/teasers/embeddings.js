/* Home-tile teaser for the Embeddings lab: one control (training steps of the lab's initial setup: batch 8, τ = 0.05, 16 dimensions),
   the 24 held-out questions as squares (green when the question's own passage is the most similar of the 24), and the lab's recall@1
   curve with the current step marked. The numbers come from the lab's own seeded engine (window.__emb), so tile and lab agree. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const STEPS=[0,10,25,50,100,200];
let si=3;
const T={render(el){
  const E=window.__emb; if(!E){el.innerHTML=''; return;}
  const lang=A.lang()==='zh'?'zh':'en', R=E.runRaw(lang,8,0.05,16), s=STEPS[si], rk=R.rank[s], k=rk.filter(v=>v===0).length, k0=R.rank[0].filter(v=>v===0).length;
  const W=300, H=104, cw=(W-8)/24, X=t=>8+(W-16)*t/200, Y=v=>98-50*v;
  let h=''; for(let i=0;i<24;i++){const x=4+cw*i; h+=`<rect class="tz-emb-q${rk[i]===0?' ok':''}" x="${(x+1).toFixed(1)}" y="4" width="${Math.max(2,cw-2).toFixed(1)}" height="16" rx="3"/>`;}
  h+=`<path class="tz-emb-line" d="${R.r1.map((v,t)=>`${t?'L':'M'}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join('')}"/>`;
  h+=`<line class="tz-emb-now" x1="${X(s)}" x2="${X(s)}" y1="${Y(1)}" y2="${Y(0)}"/><circle class="tz-emb-dot" cx="${X(s)}" cy="${Y(R.r1[s])}" r="3.5"/>`;
  h+=`<text class="tz-emb-ax" x="${W-4}" y="${Y(0)-3}" text-anchor="end">${esc(tx('steps →','步数 →'))}</text>`;
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-emb-s',key:'tzs',label:tx('Training steps','训练步数'),options:STEPS.map((v,i)=>[String(i),String(v)]),value:String(si)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(tx(`After ${s} steps, ${k} of 24 held-out questions find their own passage first.`,`训练 ${s} 步后，24 个新问题里有 ${k} 个第一个就找到自己的段落。`))}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-emb-swok"></i>${tx('finds its passage first','第一个就找到自己的段落')}</span><span><i class="sw tz-emb-swno"></i>${tx('does not','没找到')}</span><span><i class="sw tz-emb-swline"></i>${tx('share found, over steps','找到的比例随步数')}</span></div>`+
    `<p class="tz-read">${s?tx(`After ${s} training steps, ${k} of 24 questions the encoder never saw find their own help passage first (untrained: ${k0}).`,`训练 ${s} 步后，24 个没见过的问题里有 ${k} 个第一个就找到自己的说明段落（训练前 ${k0} 个）。`):tx(`Before training, ${k} of 24 questions the encoder never saw find their own help passage first: the vectors are still random.`,`训练前，24 个没见过的问题里有 ${k} 个第一个就找到自己的说明段落：向量还是随机的。`)}</p>`+
    `<p class="tz-note">${tx('Toy run: a 16-dimensional small encoder (word vectors, averaged and scaled to length 1) trained in your browser with the in-batch-negatives loss, 8 pairs per batch and temperature 0.05, on 48 question–passage pairs of a made-up smart-home help center. Each square is one held-out question, checked against all 24 passages by cosine similarity.','玩具实跑：一个 16 维的小编码器（词向量取平均，再缩放成长度 1），在你的浏览器里用 batch 内负例 loss 训练，每个 batch 8 对，温度 0.05，数据是为这个 Lab 编的智能家居帮助中心的 48 对问题和段落。每个方块是一个训练时没见过的问题，用余弦相似度和全部 24 段比。')}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzs"] button').forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('embeddings',T);
})();
