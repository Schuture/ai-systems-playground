/* Teaser for the sft lab's home tile: one segmented control picks which tokens the SFT loss counts (every token / the answer only /
   the answer only with the end token's label −100), and the tile shows what the lab's toy model writes after 60 SFT steps when asked
   “What is KV cache ?” in the training template. The toy itself lives in labs/sft.js (window.__sft, read-only): the same engine,
   the same runs, no second copy to drift. No random numbers: the same choice always gives the same tokens. The tile does not answer
   part 3's prediction (the article's share): its share is the toy conversation's. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const SHOW=7;   // tokens drawn before “…”, so the tile stays under 130 px at 280 px wide
let v='resp';
const setup=k=>({count:k==='all'?'all':'resp',eotl:'yes',pad:k==='eot'?'eot':'sep',tmpl:'same',steps:60});
function readHTML(T,k,g,gi){
  const c=T.counts('toy',setup(k).count,k==='eot'?'id':'');
  if(k==='all') return tx(`Every token counts: the answer is ${T.pcShare(c.share)} of the counted tokens, and the model answers about ${esc(T.tok(gi.first))}.`,
                          `每个 token 都算：回答只占算 loss 的 token 的 ${T.pcShare(c.share)}，模型答成了 ${esc(T.tok(gi.first))}。`);
  if(k==='resp') return tx('Only the answer and its end token count: the model answers KV cache and <b>stops</b>.','只算回答和它的结束 token：模型答对了 KV cache，然后<b>停了</b>。');
  return tx('The end token’s label is −100 too: the model answers, then <b>keeps writing new questions</b>.','结束 token 的 label 也是 −100：模型答完<b>接着写新的提问</b>，不停。');}
function notes(T){
  const all=T.counts('toy','all',''), resp=T.counts('toy','resp',''), nv=T.vocab().length;
  return [tx(`Toy run: a small model with a ${nv}-token vocabulary and ${A.fmt.int(T.NPAR)} parameters. Your browser fine-tunes it for 60 steps on 6 example conversations in the Llama 3.1 format, then asks it “What is KV cache ?”. It takes the most likely token at every step.`,
             `玩具实跑：小模型的词表有 ${nv} 个 token，共 ${A.fmt.int(T.NPAR)} 个参数。浏览器用 6 段按 Llama 3.1 格式拼好的示例对话，把它 SFT 60 步，然后问它“什么是 KV cache ？”。它每一步都选概率最高的 token。`),
    tx(`The share counts one of those conversations: ${all.N} counted positions when every token counts, ${resp.N} when only the answer and its end token count.`,
       `占比按其中一段对话计：每个 token 都算时有 ${all.N} 个位置算 loss，只算回答和结束 token 时有 ${resp.N} 个。`),
    tx('Plain gradient descent with step size 1, chosen for this small model; the step counts do not carry over to a real model.','普通梯度下降，步长 1，是为这个小模型选的；步数不能搬到真实模型上。')];}
A.registerTeaser('sft',{fid:'toy',render(el){
  const T=window.__sft; if(!T){el.innerHTML=''; return;}
  const OPTS=[['all',tx('Every token','每个 token 都算')],['resp',tx('Answer only','只算回答')],['eot',tx('Answer, end token −100','只算回答，结束 token −100')]];
  el.innerHTML=`<div class="tz tz-sft">`+A.ui.seg({id:'tz-sft-seg',key:'tzsft',label:tx('Which tokens the SFT loss counts','SFT 的 loss 算哪些 token'),options:OPTS,value:v})+
    `<div class="tz-vis tz-sft-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-sft-swstop"></i>${esc(tx('end token, stops','结束 token，停了'))}</span><span><i class="sw tz-sft-swbad"></i>${esc(tx('should have stopped','该停没停'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-sft-vis'), rd=el.querySelector('.tz-read'), seg=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes(T)[i]||'';});
  const upd=user=>{
    const o=setup(v), g=T.gen(o), gi=T.genInfo(o);
    A.ui.setPressed(seg,v);
    vis.innerHTML=g.slice(0,SHOW).map((x,i)=>`<span class="tz-sft-gt${x.k==='eot'?' stop':''}${i===gi.red?' red':''}${gi.red>=0&&i>gi.red?' extra':''}"><span class="${T.mono(x.k)?'mono':''}">${esc(T.tok(x.k))}</span></span>`).join('')+(g.length>SHOW?'<span class="tz-sft-more">…</span>':'');
    const help=tx(`What the small model writes: ${g.map(x=>T.tok(x.k)).join(' ')}${gi.stop?'':' …'}`,`小模型写出的 token：${g.map(x=>T.tok(x.k)).join(' ')}${gi.stop?'':' …'}`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(T,v,g,gi); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||b.dataset.v===v) return; v=b.dataset.v; upd(true);});
  upd(false);
}});
})();
