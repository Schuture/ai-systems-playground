/* Teaser for the scaling-law lab's home tile: one slider sets the compute budget (10²⁰ → 10²⁴ FLOPs). For that budget one iso-FLOP
   curve is drawn: across, the model size N (100M to 1T, log scale); up, the fitted loss when the budget buys D = C ÷ 6N tokens. Its
   lowest point is marked with a ring, and the sentence names that size and its tokens. The lab's mission uses another budget
   (5.9 × 10²³ FLOPs), so the tile does not answer part 1's prediction (R3.8).
   Formula only: L(N, D) = E + A/N^α + B/D^β with Epoch AI's refit of the Chinchilla runs (epoch-research/analyzing-chinchilla:
   A 482.00572, B 2085.43420, E 1.81686, α 0.34781, β 0.36585); lowest point N = G·(C/6)^a, a = β/(α+β), G = (αA/(βB))^(1/(α+β)).
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const F={E:1.81686,A:482.00572,B:2085.43420,al:0.34781,be:0.36585};
const BUD=[1e20,3e20,1e21,3e21,1e22,3e22,1e23,3e23,1e24];
let idx=4;
const W=320, H=120, ML=34, MR=8, MT=8, MB=20, X0=8, X1=12, Y0=1.9, Y1=3.2;
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const SUP='⁰¹²³⁴⁵⁶⁷⁸⁹', sup=n=>String(n).split('').map(c=>SUP[+c]).join('');
const fC=c=>{const e=Math.floor(Math.log10(c)+1e-9), m=Math.round(c/Math.pow(10,e)); return (m===1?'':m+'×')+'10'+sup(e);};
const trim0=t=>String(t).replace(/(\.\d*?)0+([A-Z]?)$/,'$1$2').replace(/\.([A-Z]?)$/,'$1');
const fP=n=>trim0(A.fmt&&A.fmt.big?A.fmt.big(n):fixd(n/1e9,2)+'B');
function fTok(d){const u=d>=1e12?[1e12,'T']:[1e9,'B'], v=d/u[0], dp=Math.max(0,2-Math.floor(Math.log10(v)+1e-9)); return trim0(fixd(v,dp))+u[1];}
const loss=(N,D)=>F.E+F.A/Math.pow(N,F.al)+F.B/Math.pow(D,F.be);
function model(i){const C=BUD[i], a=F.be/(F.al+F.be), G=Math.pow(F.al*F.A/(F.be*F.B),1/(F.al+F.be)), N=G*Math.pow(C/6,a), D=C/6/N;
  return {C,N,D,L:loss(N,D),r:D/N};}
const X=n=>ML+(Math.log10(n)-X0)/(X1-X0)*(W-ML-MR), Y=l=>H-MB-(Math.min(Y1,Math.max(Y0,l))-Y0)/(Y1-Y0)*(H-MT-MB);
function visSVG(i){const m=model(i); let pts=[];
  for(let e=X0;e<=X1+1e-9;e+=0.02){const N=Math.pow(10,e), l=loss(N,m.C/6/N); if(l<=Y1+0.3) pts.push([X(N),Y(l)]);}
  let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">`;
  [8,9,10,11,12].forEach(e=>{const x=X(Math.pow(10,e)); s+=`<line class="tz-sl-gl" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${MT}" y2="${H-MB}"/><text class="tz-sl-tk" x="${x.toFixed(1)}" y="${H-6}" text-anchor="${e===8?'start':(e===12?'end':'middle')}">${['100M','1B','10B','100B','1T'][e-8]}</text>`;});
  [2.0,2.5,3.0].forEach(v=>{const y=Y(v); s+=`<line class="tz-sl-gl" x1="${ML}" x2="${W-MR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/><text class="tz-sl-tk" x="${ML-4}" y="${(y+3).toFixed(1)}" text-anchor="end">${fixd(v,1)}</text>`;});
  s+=`<clipPath id="tz-sl-clip"><rect x="${ML}" y="${MT-2}" width="${W-ML-MR}" height="${H-MT-MB+2}"/></clipPath>`;
  s+=`<path class="tz-sl-cur" clip-path="url(#tz-sl-clip)" d="M${pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join('L')}"/>`;
  const x=X(m.N), y=Y(m.L);
  s+=`<line class="tz-sl-drop" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${(y+5).toFixed(1)}" y2="${H-MB}"/><circle class="tz-sl-min" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"/>`;
  return s+'</svg>';}
function readHTML(i){const m=model(i);
  return tx(`${fC(m.C)} FLOPs: best to train about <b>${fP(m.N)} parameters on about ${fTok(m.D)} tokens</b>.`,
            `${fC(m.C)} FLOP 的预算，最好训一个<b>约 ${fP(m.N)} 个参数的模型，喂它约 ${fTok(m.D)} 个 token</b>。`);}
function notes(){const rs=BUD.map((_,i)=>model(i).r), lo=Math.floor(Math.min(...rs)), hi=Math.ceil(Math.max(...rs));
  return [tx('Formula: the curve is the fitted loss L = E + A ÷ Nᵅ + B ÷ Dᵝ. N, the parameters, and D, the training tokens, are each in a denominator. The budget C = 6ND is fixed, so a bigger model gets fewer tokens. The constants are Epoch AI’s refit of the Chinchilla training runs.',
             '公式：曲线是拟合的 loss，L = E + A ÷ Nᵅ + B ÷ Dᵝ。参数量 N 和训练 token 数 D 各在一个分母上。算力固定为 C = 6ND，所以模型越大，分到的 token 越少。常数用的是 Epoch AI 对 Chinchilla 训练数据的重新拟合。'),
    tx(`From 10²⁰ to 10²⁴ FLOPs the lowest point stays between ${lo} and ${hi} tokens per parameter, and the best size grows about as the square root of the budget.`,
       `从 10²⁰ 到 10²⁴ FLOP，最低点都在每个参数 ${lo} 到 ${hi} 个 token 之间，最好的模型大小大约随预算的平方根增长。`),
    tx('The loss comes from one fit of one family of training runs: use it only to compare sizes.',
       '这个 loss 来自对一组训练的拟合，只能用来比较不同的大小。')];}
A.registerTeaser('scaling-law',{fid:'formula',render(el){
  const lab=tx('Compute budget','算力预算');
  el.innerHTML=`<div class="tz tz-sl">`+
    `<label class="tz-lab" for="tz-sl-in"><span>${esc(lab)}</span><output id="tz-sl-out"></output></label>`+
    `<input type="range" id="tz-sl-in" min="0" max="${BUD.length-1}" step="1" value="${idx}" aria-label="${esc(lab)}">`+
    `<div class="tz-vis tz-sl-vis" role="img" tabindex="0"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-sl-swc" aria-hidden="true"></i>${esc(tx('fitted loss of each model size at this budget','这笔预算下，每种模型大小的拟合 loss'))}</span><span><i class="sw tz-sl-swm" aria-hidden="true"></i>${esc(tx('the lowest point','最低点'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('#tz-sl-in'), out=el.querySelector('#tz-sl-out'), vis=el.querySelector('.tz-sl-vis'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{const m=model(idx), c=fC(m.C); out.textContent=c+' FLOP'; inp.value=idx; inp.setAttribute('aria-valuetext',c+' FLOP'); if(A.ui&&A.ui.setRange) A.ui.setRange(inp);
    vis.innerHTML=visSVG(idx);
    const help=tx(`The curve is the fitted loss of each model size, from 100M to 1T parameters, when ${c} FLOPs buy its tokens. The ring is the lowest point: ${fP(m.N)} parameters on ${fTok(m.D)} tokens, loss ${fixd(m.L,3)}.`,
                  `曲线是 ${c} FLOP 的预算下，从 100M 到 1T 个参数，每种模型大小的拟合 loss。圆圈是最低点：${fP(m.N)} 个参数、${fTok(m.D)} 个 token，loss ${fixd(m.L,3)}。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    const h=readHTML(idx); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>{const v=Math.max(0,Math.min(BUD.length-1,Math.round(+inp.value))); if(v!==idx){idx=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/scaling-law_numbers.py */
window.__tzSl=Object.freeze({BUD:BUD.slice(),F:{...F},model:i=>({...model(i)}),read:i=>readHTML(i),notes:()=>notes()});
})();
