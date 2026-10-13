/* Teaser for the reward-model lab's home tile: the lab's toy reward model (3 parameters: content, length, list) trained on the lab's
   default 1,000 simulated preference pairs (seed 13, mulberry32, Box–Muller; the same engine as labs/reward-model.js, re-implemented
   here so the tile does not depend on the lab). One slider for the training steps; the visual shows one illustrative pair's two
   scores on a score axis and σ(r_A − r_B) as a bar. The pair is not one of the lab's examples, and its gap is not the lab's
   prediction question (R3.8). One sentence in .tz-read, a one-line legend, plain-text notes. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
function mulberry(a){return function(){a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296;};}
const gauss=r=>{const u=r(), v=r(); return Math.sqrt(-2*Math.log(1-u))*Math.cos(2*Math.PI*v);};
const sig=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const QM=6, QS=1.5, SE=1.0, L0=150, LS=0.4, ETA=0.15, T=200, NP=1000, SEED=13, REL=0.35;
const clampq=q=>Math.max(0.5,Math.min(9.5,q));
function resp(r){const q=clampq(QM+QS*gauss(r)), e=gauss(r), zl=gauss(r), b=r()<0.5?1:0; return {q,c:q+SE*e,L:Math.exp(Math.log(L0)+REL*(q-QM)/QS+0.25*zl),b};}
const phi=x=>[(x.c-QM)/QS,Math.log(x.L/L0)/LS,x.b-0.5];
let W=null;
/* the lab's default run: full-batch gradient descent on mean −log σ(r_chosen − r_rejected), from w = 0 */
function weights(){
  if(W) return W;
  const r=mulberry(SEED), D=[];
  for(let i=0;i<NP;i++){const X=resp(r), Y=resp(r), u=r(), aw=u<sig(X.q-Y.q), fw=phi(aw?X:Y), fl=phi(aw?Y:X); D.push([fw[0]-fl[0],fw[1]-fl[1],fw[2]-fl[2]]);}
  let w=[0,0,0]; W=[w.slice()];
  for(let t=0;t<T;t++){const g=[0,0,0]; for(const d of D){const s=sig(-(w[0]*d[0]+w[1]*d[1]+w[2]*d[2])); g[0]-=s*d[0]; g[1]-=s*d[1]; g[2]-=s*d[2];}
    w=[w[0]-ETA*g[0]/NP,w[1]-ETA*g[1]/NP,w[2]-ETA*g[2]/NP]; W.push(w.slice());}
  return W;
}
/* the tile's pair (illustrative features): A more complete and a little longer, chosen; B rejected */
const PA={c:7.0,L:150,b:0}, PB={c:6.0,L:120,b:0};
const STEPS=[0,5,10,20,50,100,200];
let si=STEPS.length-1;
function at(t){const w=weights()[t], fa=phi(PA), fb=phi(PB), ra=w[0]*fa[0]+w[1]*fa[1]+w[2]*fa[2], rb=w[0]*fb[0]+w[1]*fb[1]+w[2]*fb[2]; return {t,ra,rb,p:sig(ra-rb)};}
const MIN='−', s2=v=>{const a=Math.abs(v).toFixed(2); return (v<0&&+a!==0?MIN:'')+a;};
const pc=v=>Math.round(v*100)+'%';
function readHTML(){
  const x=at(STEPS[si]);
  if(x.t===0) return tx('Before training: both responses score 0, so the reward model says either is picked 50% of the time.',
                        '训练之前：两个回答都是 0 分，奖励模型算出两个被选中的概率各是 50%。');
  return tx(`After ${x.t} steps, A (chosen) scores ${s2(x.ra)} and B (rejected) ${s2(x.rb)}. The reward model says people pick A <b>${pc(x.p)}</b> of the time.`,
            `训练 ${x.t} 步后，选中的回答 ${s2(x.ra)} 分，落选的 ${s2(x.rb)} 分。奖励模型算出，人选 A 的概率是 <b>${pc(x.p)}</b>。`);
}
function notes(){
  return [tx('Toy run: the lab’s small reward model, 3 parameters (weights on content, length and list), trained on the lab’s 1,000 simulated preference pairs (seed 13). The pair’s features are illustrative.',
             '玩具实跑：Lab 里只有 3 个参数的小奖励模型（内容、长度、列表各一个权重），在 Lab 的 1,000 个模拟偏好对上训练（种子 13）。这一对回答的特征是示意的。'),
    tx('P(person picks A) = σ(r_A − r_B). Scores have no fixed zero: only the gap between two scores means something.',
       'P(人选 A) = σ(r_A − r_B)。分数没有固定的零点，只有两个分数的差有意义。')];
}
function visHTML(){
  const x=at(STEPS[si]), X=v=>Math.max(2,Math.min(98,50+v/1.2*48)), near=Math.abs(X(x.ra)-X(x.rb))<7;
  return `<div class="tz-rm-ax"><b class="tz-rm-0"></b>`+
    `<span class="tz-rm-m tz-rm-b" style="left:${X(x.rb).toFixed(1)}%"><i></i><em>${near?'':'B'}</em></span>`+
    `<span class="tz-rm-m tz-rm-a" style="left:${X(x.ra).toFixed(1)}%"><i></i><em>${near?tx('A, B','A、B'):'A'}</em></span></div>`+
    `<div class="tz-rm-tk mono"><span>${MIN}1.2</span><span>${tx('score 0','分数 0')}</span><span>+1.2</span></div>`+
    `<div class="tz-rm-p"><span class="tz-rm-pl">${tx('P(A)','P(选 A)')}</span><span class="tz-rm-pt"><i style="width:${(x.p*100).toFixed(1)}%"></i><b></b></span><span class="tz-rm-pv mono">${pc(x.p)}</span></div>`;
}
A.registerTeaser('reward-model',{fid:'toy',render(el){
  const lab=tx('Training steps on the 1,000 pairs','在 1,000 个偏好对上训练的步数');
  el.innerHTML=`<div class="tz tz-rm">`+
    `<label class="tz-lab" for="tz-rm-in"><span>${esc(lab)}</span><output id="tz-rm-out">${STEPS[si]}</output></label>`+
    `<input type="range" id="tz-rm-in" min="0" max="${STEPS.length-1}" step="1" value="${si}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-rm-vis" role="img"></div>`+
    `<div class="tz-leg"><span><i class="tz-rm-sw tz-rm-swa" aria-hidden="true"></i>${esc(tx('A, chosen','A，选中的回答'))}</span><span><i class="tz-rm-sw tz-rm-swb" aria-hidden="true"></i>${esc(tx('B, rejected','B，落选的回答'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('.tz-rm-vis'), rd=el.querySelector('.tz-read'), inp=el.querySelector('#tz-rm-in'), out=el.querySelector('#tz-rm-out');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const x=at(STEPS[si]); vis.innerHTML=visHTML();
    out.textContent=String(STEPS[si]); inp.value=si; inp.setAttribute('aria-valuetext',tx(`${x.t} steps`,`${x.t} 步`)); if(A.ui.setRange) A.ui.setRange(inp);
    const lab=tx(`Step ${x.t}: A ${s2(x.ra)}, B ${s2(x.rb)}, P(A) ${pc(x.p)}.`,`第 ${x.t} 步：A ${s2(x.ra)} 分，B ${s2(x.rb)} 分，P(选 A) ${pc(x.p)}。`); vis.setAttribute('aria-label',lab);
    const h=readHTML(); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('input',e=>{if(e.target!==inp) return; const v=Math.max(0,Math.min(STEPS.length-1,+inp.value|0)); if(v!==si){si=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/reward-model_numbers.py */
window.__tzRm=Object.freeze({at:t=>({...at(t)}),steps:STEPS.slice(),w:t=>weights()[t].slice(),read:i=>{const k=si; if(i!=null) si=i; const h=readHTML(); si=k; return h;},notes:()=>notes()});
})();
