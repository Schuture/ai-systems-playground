/* Home-tile teaser for the rag-failure lab: one control (the stage to look at: retrieval, the context budget, the documents, the
   model), the lab's 200 graded questions as squares (green: correct; red: wrong at the stage picked; grey: wrong at another stage).
   The numbers come from the lab's own engine (window.__rgf), so tile and lab agree. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const STG=[{k:'ret',kinds:['miss','multi'],l:{en:'Retrieval',zh:'检索'},w:{en:'did not retrieve all the passages they need',zh:'要用的段落没检索全'}},
  {k:'ctx',kinds:['cut'],l:{en:'Budget',zh:'预算'},w:{en:'had the passage retrieved but cut by the context budget',zh:'要用的段落检索到了，却被上下文预算挤掉'}},
  {k:'doc',kinds:['stale'],l:{en:'Old page',zh:'旧文档'},w:{en:'followed a replaced, outdated page',zh:'照着被替换的旧版文档答'}},
  {k:'ans',kinds:['unsup','noans'],l:{en:'Model',zh:'模型'},w:{en:'said what no passage says',zh:'回答说了段落里没有的内容'}}];
let si=0;
const T={render(el){
  const E=window.__rgf; if(!E){el.innerHTML=''; return;}
  const D=E.data(), mix=D.MIX.base, st=STG[si], order=['ok'].concat(st.kinds, D.KINDS.filter(k=>!st.kinds.includes(k)));
  const cells=[]; order.forEach(k=>{for(let i=0;i<mix[k];i++) cells.push(k==='ok'?'ok':(st.kinds.includes(k)?'me':'other'));});
  const N=cells.length, wrong=N-mix.ok, n=st.kinds.reduce((a,k)=>a+mix[k],0);
  const cols=25, cw=10, gap=2, rows=Math.ceil(N/cols), W=cols*(cw+gap), H=rows*(cw+gap);
  let h=''; cells.forEach((x,i)=>{const r=Math.floor(i/cols), c=i%cols; h+=`<rect class="tz-rgf-q ${x}" x="${c*(cw+gap)}" y="${r*(cw+gap)}" width="${cw}" height="${cw}" rx="2"/>`;});
  const read=tx(`${st.l.en}: ${n} of the ${wrong} wrong answers ${st.w.en}.`,`${st.l.zh}：${wrong} 个错答里有 ${n} 个，${st.w.zh}。`);
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-rgf-s',key:'tzs',label:tx('Where it fails','错在哪一环'),options:STG.map((s,i)=>[String(i),A.L(s.l)]),value:String(si)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(read)}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-rgf-swok"></i>${tx('correct','答对')}</span><span><i class="sw tz-rgf-swme"></i>${tx('wrong at this stage','错在这一环')}</span><span><i class="sw tz-rgf-swot"></i>${tx('wrong elsewhere','错在别处')}</span></div>`+
    `<p class="tz-read">${esc(read)}</p>`+
    `<p class="tz-note">${esc(tx(`Illustrative: a made-up help-center bot answered 200 graded questions; ${mix.ok} are correct. The mix of failures is the lab’s stated example, not a measurement. Each square is one question.`,
      `示意：一个虚构的帮助中心机器人回答了 200 个问题，每个都判过对错，答对 ${mix.ok} 个。各类错的数量是 Lab 写定的例子，不是实测。每个方块是一个问题。`))}</p>`+
    `<p class="tz-note">${esc(tx('Retrieval recall can be high while many answers are still wrong: most failures here happen after retrieval, at the budget, in old pages or in the model.','检索召回率很高，答错的也可能很多。这里大多数错发生在检索之后，在预算、旧文档或模型那几环。'))}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzs"] button').forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('rag-failure',T);
})();
