/* Teaser for the backprop lab's home tile: move the input x of one neuron, y = tanh(w·x) with w = 0.8, and watch the forward
   values (top lane, left to right) and the backward pass (bottom lane, right to left): it starts at dy/dy = 1, multiplies by
   tanh's local derivative 1 − y², then by w·x's local derivative for w, which is x, and so gets dy/dw = (1 − y²)·x.
   One control (a slider), one small visual (two lanes of three boxes, 118 px tall), one sentence in .tz-read, a legend, plain notes.
   The rule is the lab's own (part 3): each backward step multiplies by one local derivative. State lives in module scope. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const MINUS='−', W0=0.8, XMIN=-20, XMAX=20;
let xi=15;   // x = xi / 10, starts at 1.5 (the lab's x₁)
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sg=(v,d)=>(v<0&&Math.abs(v)>=0.5*Math.pow(10,-d)?MINUS:'')+fixd(Math.abs(v),d);
const f1=v=>Math.round(v*10)/10;
function model(i){const x=i/10, z=W0*x, y=Math.tanh(z), loc=1-y*y; return {x,w:W0,z,y,loc,dz:loc,dw:loc*x};}
function readHTML(i){const m=model(i);
  return tx(`At x = ${sg(m.x,1)}, y = ${sg(m.y,4)}. One backward pass multiplies the two local derivatives: dy/dw = 1 × ${sg(m.loc,4)} × ${m.x<0?`(${sg(m.x,1)})`:sg(m.x,1)} = <b>${sg(m.dw,4)}</b>.`,
            `x = ${sg(m.x,1)} 时，y = ${sg(m.y,4)}。一趟反向把两个局部导数乘起来：dy/dw = 1 × ${sg(m.loc,4)} × ${m.x<0?`(${sg(m.x,1)})`:sg(m.x,1)} = <b>${sg(m.dw,4)}</b>。`);}
function notes(){
  return [tx('Toy run: one weight w = 0.8 and one tanh, computed in your browser as you move x.',
             '玩具实跑：一个权重 w = 0.8 和一个 tanh，拖动 x 时在浏览器里现算。'),
    tx('Each backward step multiplies by one local derivative: 1 − y² for y = tanh(z), and x for z = w·x when the gradient goes to w.',
       '反向的每一步乘一个局部导数：y = tanh(z) 的是 1 − y²；z = w·x 把梯度传给 w 时乘的是 x。'),
    tx('The lab runs the same rule on a network with four parameters and checks every gradient by nudging it.',
       '这个 Lab 在一个有四个参数的网络上用同样的规则，并把每个梯度推一推来核对。')];}
function svg(W,i){
  const m=model(i), pad=2, gap=Math.max(44,Math.min(64,W*0.17)), bw=(W-2*pad-2*gap)/3, bh=32, y1=14, y2=78, H=118;
  const bx=k=>pad+k*(bw+gap);
  const box=(k,y,cap,val,cls)=>`<rect x="${f1(bx(k))}" y="${y}" width="${f1(bw)}" height="${bh}" rx="5" class="tz-bp-box${cls?' '+cls:''}"/>`+
    `<text x="${f1(bx(k)+bw/2)}" y="${y+13}" text-anchor="middle" class="tz-bp-cap">${esc(cap)}</text>`+
    `<text x="${f1(bx(k)+bw/2)}" y="${y+26}" text-anchor="middle" class="tz-bp-val">${esc(val)}</text>`;
  const arr=(k,y,lab,back)=>{const xa=bx(k)+bw+3, xb=bx(k+1)-3, ym=y+bh/2;
    const line=back?`<line x1="${f1(xb)}" y1="${ym}" x2="${f1(xa+6)}" y2="${ym}" class="tz-bp-bk"/><path d="M${f1(xa)} ${ym} L${f1(xa+7)} ${ym-4} L${f1(xa+7)} ${ym+4} Z" class="tz-bp-bkh"/>`
                   :`<line x1="${f1(xa)}" y1="${ym}" x2="${f1(xb-6)}" y2="${ym}" class="tz-bp-fw"/><path d="M${f1(xb)} ${ym} L${f1(xb-7)} ${ym-4} L${f1(xb-7)} ${ym+4} Z" class="tz-bp-fwh"/>`;
    return line+`<text x="${f1((xa+xb)/2)}" y="${ym-6}" text-anchor="middle" class="tz-bp-op">${esc(lab)}</text>`;};
  let h=`<text x="${pad}" y="9" class="tz-bp-lane">${esc(tx('forward: values, with w = 0.8','前向：值，w = 0.8'))}</text>`+
    box(0,y1,'x',sg(m.x,1))+arr(0,y1,'× w')+box(1,y1,'z',sg(m.z,4))+arr(1,y1,'tanh')+box(2,y1,'y',sg(m.y,4))+
    `<text x="${pad}" y="${y2-5}" class="tz-bp-lane">${esc(tx('backward: gradients, right to left','反向：梯度，从右往左'))}</text>`+
    box(0,y2,'dy/dw',sg(m.dw,4),'tz-bp-on')+arr(0,y2,`× ${sg(m.x,1)}`,true)+box(1,y2,'dy/dz',sg(m.dz,4))+arr(1,y2,`× ${sg(m.loc,4)}`,true)+box(2,y2,'dy/dy','1');
  return {H,h};
}
A.registerTeaser('backprop',{fid:'toy',render(el){
  const lab=tx('Input x','输入 x');
  el.innerHTML=`<div class="tz tz-bp"><label class="tz-lab"><span>${esc(lab)}</span><output class="mono"></output></label>`+
    `<input type="range" class="tz-bp-range" min="${XMIN}" max="${XMAX}" step="1" value="${xi}" aria-label="${esc(lab)}">`+
    `<svg class="tz-vis tz-bp-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span><i class="tz-bp-sw tz-bp-swfw" aria-hidden="true"></i>${esc(tx('solid: forward','实线：前向'))}</span><span><i class="tz-bp-sw tz-bp-swbk" aria-hidden="true"></i>${esc(tx('dashed: backward, × a local derivative','虚线：反向，乘一个局部导数'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), vis=el.querySelector('svg'), rd=el.querySelector('.tz-read');
  const help=tx('Top lane: the forward values x, z = w·x and y = tanh(z). Bottom lane: the backward pass from dy/dy = 1, multiplied by 1 − y², then by x, which gives dy/dw.','上面一行：前向的值 x、z = w·x 和 y = tanh(z)。下面一行：反向从 dy/dy = 1 开始，乘 1 − y²，再乘 x，得到 dy/dw。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const draw=()=>{const W=Math.max(260,Math.round(el.clientWidth||320)), r=svg(W,xi); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{xi=+inp.value; if(A.ui.setRange) A.ui.setRange(inp); out.textContent=sg(xi/10,1); draw(); const h=readHTML(xi); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>upd(true));
  upd(false);
}});
/* read-only hook for tests/backprop_numbers.py: the tile's numbers */
window.__tzBp=Object.freeze({model:i=>JSON.parse(JSON.stringify(model(i))),read:i=>readHTML(i),W:W0,range:[XMIN,XMAX]});
})();
