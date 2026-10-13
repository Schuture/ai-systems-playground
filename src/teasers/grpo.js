/* Home-tile teaser for the GRPO lab: one control (how many times one question is answered, G), the G answers as chips, and the group
   mean against the question's true accuracy. The toy is the lab's: a softmax over 10 candidate answers, near misses at −0.35 × |offset|,
   the right answer's logit set so the policy answers right 40% of the time. The answers are one seeded sequence of 32 draws (mulberry32,
   seed 20261010) and a group of G takes its first G, so a larger group only adds answers to a smaller one (review B2). With this seed
   the group mean gets closer to 40% at every larger G: 2 of 2, 3 of 4, 4 of 8, 7 of 16, 13 of 32 right. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const {tx,esc}=A;
const P=0.4, Q=[27,18], NC=10, RIGHT=4, OFFS=[-4,-3,-2,-1,0,1,2,3,4,5], SEED=20261010, GMAX=32;
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
function probs(){const z=OFFS.map(o=>-0.35*Math.abs(o)); let s=0; for(let j=0;j<NC;j++) if(j!==RIGHT) s+=Math.exp(z[j]); z[RIGHT]=Math.log(P/(1-P)*s);
  let m=-Infinity; for(const v of z) if(v>m) m=v; const e=z.map(v=>Math.exp(v-m)); let t=0; for(const v of e) t+=v; return e.map(v=>v/t);}
/* the 32 answers, drawn once; reward 1 for the right one */
const ALL=(()=>{const pi=probs(), rnd=mulberry(SEED), r=[];
  for(let i=0;i<GMAX;i++){const u=rnd(); let acc=0, k=NC-1; for(let j=0;j<NC;j++){acc+=pi[j]; if(u<acc){k=j; break;}} r.push(k===RIGHT?1:0);}
  return r;})();
let G=8;
const T={render(el){
  const r=ALL.slice(0,G), k=r.filter(v=>v).length, mean=k/G, W=300, H=92, cw=(W-8)/G, y=v=>86-56*v, LX=212, m2=A.fmt.num(mean,2);
  let h=''; for(let i=0;i<G;i++){const x=4+cw*i; h+=`<rect class="tz-grpo-c${r[i]?' ok':''}" x="${(x+1).toFixed(1)}" y="4" width="${Math.max(2,cw-2).toFixed(1)}" height="16" rx="3"/>`;}
  const ym=y(mean), yt=y(P), close=Math.abs(ym-yt)<12, up=ym<=yt;
  /* each line's value at its right end; when the two lines are close, the higher label goes above and the lower one below */
  const lm=close?(up?Math.min(ym,yt)-3:Math.max(ym,yt)+10):ym+3, lt=close?(up?Math.max(ym,yt)+10:Math.min(ym,yt)-3):yt+3;
  h+=`<line class="tz-grpo-true" x1="14" x2="${LX}" y1="${yt}" y2="${yt}"/><line class="tz-grpo-mean" x1="14" x2="${LX}" y1="${ym}" y2="${ym}"/>`;
  h+=`<text class="tz-grpo-lab" x="${LX+6}" y="${lm.toFixed(1)}">${esc(tx(`mean ${m2}`,`平均分 ${m2}`))}</text><text class="tz-grpo-lab tz-grpo-labt" x="${LX+6}" y="${lt.toFixed(1)}">${esc(tx('true 40%','真实 40%'))}</text>`;
  h+=`<text class="tz-grpo-ax" x="4" y="${y(1)+3}">1</text><text class="tz-grpo-ax" x="4" y="${y(0)}">0</text>`;
  el.innerHTML=`<div class="tz">`+A.ui.seg({id:'tz-grpo-g',key:'tzg',label:tx('Answers to one question, G','同一道题答几次 G'),options:[2,4,8,16,32].map(g=>[String(g),String(g)]),value:String(G)})+
    `<div class="tz-vis"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(tx(`${k} of ${G} answers right; group mean ${m2}, true accuracy 40%`,`${G} 个回答里 ${k} 个对，组内平均分 ${m2}，真实答对率 40%`))}">${h}</svg></div>`+
    `<div class="tz-leg"><span><i class="sw tz-grpo-swok"></i>${tx('right answer','答对的回答')}</span><span><i class="sw tz-grpo-swmean"></i>${tx('group mean','组内平均分')}</span><span><i class="sw tz-grpo-swtrue"></i>${tx('true accuracy','真实答对率')}</span></div>`+
    `<p class="tz-read">${tx(`${G} answers to one question, ${k} right: their mean score, ${m2}, is the baseline, with no value model. The true accuracy is 40%; larger groups usually land closer to it.`,
      `同一道题答 ${G} 次，对了 ${k} 次，平均分 ${m2} 就是基准，用不着价值模型。这道题真实的答对率是 40%，组越大，平均分一般离它越近。`)}</p>`+
    `<p class="tz-note">${tx(`Toy run: a small model with 10 parameters, one per candidate answer, answers “${Q[0]} + ${Q[1]}” right 40% of the time. The 32 answers are drawn once with a fixed seed, and a group of G takes the first G. For Llama‑3.1‑8B a value model would take 144.5 GB to train (18 bytes per parameter). A larger group usually estimates the baseline better but costs more generation.`,
      `玩具实跑：一个只有 10 个参数的小模型（每个候选答案一个参数）答对“${Q[0]} + ${Q[1]}”的概率是 40%。32 个回答是用固定的随机种子一次抽好的，一组 G 个就取前 G 个。训练 Llama‑3.1‑8B 时，价值模型要 144.5 GB（每个参数 18 字节）。组越大，基准一般估得越准，生成的成本也越高。`)}</p></div>`;
  el.querySelectorAll('.seg-ctl[data-key="tzg"] button').forEach(b=>b.addEventListener('click',()=>{G=+b.dataset.v; T.render(el);}));}};
A.registerTeaser('grpo',T);
/* read-only hook for tests/grpo_numbers.py: the tile's 32 rewards */
window.__grpoTile={rewards:()=>ALL.slice(),seed:SEED,p:P};
})();
