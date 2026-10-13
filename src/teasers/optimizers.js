/* The optimizers lab's home tile answers the lab's question, “What does Adam do that SGD does not?”, on the lab's own valley
   ½(x² + 25 y²) from (−9, 1.5). One control picks SGD or Adam; the picture draws that optimizer's path solid and the other one's
   dashed, so the comparison is always on screen; the sentence says what the picked one does, with its steps to the bottom.
   The runs are the lab's part 1 runs, made by the lab's own run() (labs/optimizers.js, SGD05 and ADAM05): the lab keeps its engine
   inside its module and exports it read-only as window.__op, and build.py always bundles labs/optimizers.js with this file (one stem).
   SGD at learning rate 0.05 reaches the bottom (a loss below 0.001) in 104 steps, Adam at 0.5 in 74, as the lab's part 1 says.
   R3.8: no SGD at 0.1 and no diverging run (the lab's part 2 asks what doubling SGD's learning rate to 0.10 does: it never reaches the
   bottom), and nothing from the challenge (its valley ½(x² + 100 y²) and its optimizer × learning-rate grid).
   Adam is not always faster (SGD at 0.075 needs 69 steps): the sentence never says it is, and a note says it is not.
   The view holds the whole valley (VIEW, the v4 fix: a view that stopped at x = 1.5 cut the ellipses at the frame's edge), and both
   paths stay inside it (x from −9 to 1.36, y from −0.76 to 1.5). Grayscale only (R5.6): a path is not a weight, a KV cache or a failure.
   State lives in module scope, so a re-render (branch or language switch) keeps the pick.
   window.__tzOp is a read-only hook for tests/optimizers_numbers.py, tests/optimizers_flow.py and tests/optimizers_copy_deck.py. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const VIEW=[-10,10,-2,2], H=110, START=[-9,1.5], B=25, LEVELS=[0.5,2,8,32];
const CFG={sgd:{opt:'sgd',lr:0.05},adam:{opt:'adam',lr:0.5}};   // the lab's SGD05 and ADAM05 (part 1)
const OPTS=['sgd','adam'], NAME={sgd:'SGD',adam:'Adam'};
const other=o=>o==='sgd'?'adam':'sgd';
let tzO='adam';   // the tile opens on the answer: Adam's path solid, SGD's dashed beside it
/* a run as the lab draws it: from the start to the step that reaches the bottom */
function runOf(o){const r=window.__op.run(CFG[o]), end=r.hit!=null?r.hit:r.n; return {pts:r.pts.slice(0,end+1),hit:r.hit,div:r.div};}
const fLR=v=>String(+v.toFixed(3));
const NB='\u00a0';   // a no-break space: a step count never breaks from its unit
/* one sentence: the condition (optimizer, learning rate) and what its path does, with the steps (R6.8, R11.5) */
function readHTML(o){const s=runOf('sgd'), a=runOf('adam'), ls=fLR(CFG.sgd.lr), la=fLR(CFG.adam.lr);
  if(o==='sgd') return tx(`SGD’s step is the learning rate times the gradient, so at learning rate ${ls} it zig-zags across the steep valley, then crawls along the flat floor to the bottom in <b>${s.hit}${NB}steps</b>.`,
    `SGD 每一步走学习率乘以梯度，所以学习率为 ${ls} 时，它先在陡峭的山谷里来回横跳，再沿着平坦的山谷中线慢慢爬，<b>${s.hit}${NB}步</b>才到谷底。`);
  return tx(`Adam gives each parameter its own step size, so at learning rate ${la} it neither zig-zags across the valley nor crawls along it: it reaches the bottom in <b>${a.hit}${NB}steps</b>, SGD at ${ls} in ${s.hit}.`,
    `Adam 给每个参数各自的步长，所以学习率为 ${la} 时，它既不在山谷里来回横跳，也不沿着山谷中线慢慢爬：<b>${a.hit}${NB}步</b>到谷底，学习率 ${ls} 的 SGD 要 ${s.hit}${NB}步。`);}
/* the notes behind “Assumptions & sources”: four short sentences, each answering one question (R11.5) */
function notes(){const q=window.__op.run({opt:'sgd',lr:0.075}).hit;
  return [tx('Both runs start at (−9, 1.5) on the loss ½(x² + 25 y²) of two parameters, x and y.',
             '两次运行都从 (−9, 1.5) 出发，山谷是两个参数 x、y 的 loss，即 ½(x² + 25 y²)。'),
    tx('Toy run with the settings of the lab’s part 1; the bottom is the first step with a loss below 0.001.',
       '玩具实跑，用的是 Lab 第 1 部分的设置；loss 第一次低于 0.001 的那一步算到谷底。'),
    tx('Adam’s learning rate is about the largest step it takes, while SGD’s multiplies the gradient: the two are not comparable.',
       'Adam 的学习率大约是它每步最多走的距离，SGD 的学习率却是梯度的倍数，两者不能直接比。'),
    tx(`Adam is not always faster: SGD at learning rate 0.075 needs ${q} steps; the lab shows when Adam pays off.`,
       `Adam 并不总是更快：学习率 0.075 的 SGD ${q} 步就到谷底；Adam 什么时候才划算，Lab 里会讲。`)];}
/* the legend names the two line styles for the current pick (R5.7) */
const sw=cls=>`<svg class="tz-op-sw" width="18" height="6" viewBox="0 0 18 6" aria-hidden="true"><line x1="1" y1="3" x2="17" y2="3" class="tz-op-path ${cls}"/></svg>`;
function legendHTML(o){const q=other(o);
  return `<span>${sw('p')}${esc(tx(`solid: ${NAME[o]}’s path`,`实线：${NAME[o]} 的路径`))}</span>`+
    `<span>${sw('q')}${esc(tx(`dashed: ${NAME[q]}’s path`,`虚线：${NAME[q]} 的路径`))}</span>`+
    `<span><i class="tz-op-mk s" aria-hidden="true"></i>${esc(tx('ring: start','圆圈：出发点'))}</span>`+
    `<span><i class="tz-op-mk m" aria-hidden="true">×</i>${esc(tx('bottom','谷底'))}</span>`;}
const helpText=o=>tx(`The valley seen from above: each ellipse is one loss level, steep across (up and down), flat along it. The solid line is ${NAME[o]}’s path from the ring to the ×; the dashed line is ${NAME[other(o)]}’s.`,
  `从上往下看山谷：每个椭圆是一个 loss 高度，横穿方向（上下）陡，沿山谷方向平。实线是 ${NAME[o]} 从圆圈走到 × 的路径，虚线是 ${NAME[other(o)]} 的。`);
const f1=v=>(Math.round(v*10)/10).toFixed(1);
function svgOf(W,o){
  const [x0,x1,y0,y1]=VIEW, X=x=>(x-x0)/(x1-x0)*W, Y=y=>H-(y-y0)/(y1-y0)*H, line=pts=>pts.map(p=>`${f1(X(p[0]))},${f1(Y(p[1]))}`).join(' ');
  let h=`<rect x="0.5" y="0.5" width="${W-1}" height="${H-1}" class="tz-op-bg"/>`;
  /* contour ellipses: ½(x² + 25 y²) = c has semi-axes √(2c) along x and √(2c ÷ 25) across; the largest, c = 32, is 8 by 1.6, inside the view */
  LEVELS.forEach(c=>{h+=`<ellipse cx="${f1(X(0))}" cy="${f1(Y(0))}" rx="${f1(Math.sqrt(2*c)/(x1-x0)*W)}" ry="${f1(Math.sqrt(2*c/B)/(y1-y0)*H)}" class="tz-op-ct"/>`;});
  h+=`<polyline points="${line(runOf(other(o)).pts)}" class="tz-op-path q"/>`;
  h+=`<polyline points="${line(runOf(o).pts)}" class="tz-op-path p"/>`;
  h+=`<circle cx="${f1(X(START[0]))}" cy="${f1(Y(START[1]))}" r="4" class="tz-op-start"/>`;
  h+=`<path d="M${f1(X(0)-4)} ${f1(Y(0)-4)}l8 8M${f1(X(0)-4)} ${f1(Y(0)+4)}l8 -8" class="tz-op-min"/>`;
  return h;
}
A.registerTeaser('optimizers',{fid:'toy',render(el){
  const lab=tx('Optimizer','优化器');
  el.innerHTML=`<div class="tz tz-op">${A.ui.seg({key:'tz-op',label:lab,options:OPTS.map(o=>[o,NAME[o]]),value:tzO,inline:false})}`+
    `<svg class="tz-vis tz-op-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"></div>`+
    /* the other pick's sentence sits invisibly in the same grid cell, so the tile keeps its height when the pick changes */
    `<div class="tz-op-rd"><p class="tz-read"></p><p class="tz-op-twin" aria-hidden="true"></p></div>`+
    `${notes().map(()=>'<p class="tz-note"></p>').join('')}</div>`;
  const q=s=>el.querySelector(s), seg=q('.seg-ctl'), vis=q('svg.tz-op-vis'), rd=q('.tz-read'), twin=q('.tz-op-twin'), leg=q('.tz-leg');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{A.ui.setPressed(seg,tzO);
    const W=Math.max(260,Math.round(el.clientWidth||320)), help=helpText(tzO);
    vis.setAttribute('viewBox',`0 0 ${W} ${H}`); vis.setAttribute('width',W); vis.setAttribute('height',H); vis.innerHTML=svgOf(W,tzO);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
    leg.innerHTML=legendHTML(tzO); twin.innerHTML=readHTML(other(tzO));
    const h=readHTML(tzO); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  seg.addEventListener('click',e=>{const b=e.target.closest('button[data-v]'); if(!b||!seg.contains(b)||!CFG[b.dataset.v]) return; tzO=b.dataset.v; upd(true);});
  upd(false);
}});
window.__tzOp=Object.freeze({opts:OPTS.slice(),cfg:o=>({...CFG[o]}),view:VIEW.slice(),H,
  run:o=>{const r=runOf(o); return {hit:r.hit,div:r.div,n:r.pts.length-1,pts:r.pts.map(p=>p.slice())};},
  read:o=>readHTML(o),legend:o=>legendHTML(o),help:o=>helpText(o),notes:()=>notes()});
})();
