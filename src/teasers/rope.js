/* Teaser for the rope lab's home tile: switch RoPE off or on and watch pair 0 of the query of "bites" and of the key of "dog" in
   "dog bites man" and "man bites dog". Off: the same arrows and the same score in both orders. On: each arrow turns by its position,
   so the angle between them, and the score, differ. One control, one small visual (two dials, 112 px tall), one sentence in .tz-read,
   a one-line legend, and plain-text notes. The numbers are the lab's own (labs/rope.js, part 3, pair 0): the query of "bites"
   (length 1.5 at −47°) at position 2, the key of "dog" (1.3 at 30°) at position 1 or 3, turned by the fastest pair of every Llama
   head, 1 radian per position. It shows part 3's result, not part 4's (R3.8: the tile does not answer part 4's prediction).
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const R2D=180/Math.PI, D2R=Math.PI/180, TH0=1, MINUS='−';
const Q=[1.5,-47], K=[1.3,30], M=2, N={a:1,b:3};
let tzOn=false;   // opens at "off", so the click makes the change, as in the lab's part 3
const fixd=(v,d)=>v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const f1=v=>Math.round(v*10)/10;
const wrap=d=>{let b=((d+180)%360+360)%360-180; if(b===-180) b=180; return b;};
function pair(ord,on){const qa=Q[1]+(on?M*TH0*R2D:0), ka=K[1]+(on?N[ord]*TH0*R2D:0), between=wrap(qa-ka);
  return {ord,on,m:M,n:N[ord],qa,ka,between,score:Q[0]*K[0]*Math.cos(between*D2R)};}
function model(on){return {on,a:pair('a',on),b:pair('b',on)};}
const sc2=v=>(v<0?MINUS:'')+fixd(Math.abs(v),2);
const deg=v=>fixd(Math.abs(v),1)+'°';
function readHTML(on){const m=model(on);
  if(!on) return tx(`Without RoPE, pair 0 of bites → dog gives <b>${sc2(m.a.score)}</b> in both “dog bites man” and “man bites dog”: the order is invisible.`,
                    `没有 RoPE 时，“咬 → 狗”的第 0 对在“狗咬人”和“人咬狗”里都给 <b>${sc2(m.a.score)}</b>：看不出顺序。`);
  return tx(`With RoPE, each arrow turns by its position: ${deg(m.a.between)} apart in “dog bites man”, ${deg(m.b.between)} in “man bites dog”, so pair 0 scores <b>${sc2(m.a.score)}</b> against <b>${sc2(m.b.score)}</b> and the order shows.`,
            `有 RoPE 时，每个箭头按位置转动，“狗咬人”里相差 ${deg(m.a.between)}，“人咬狗”里相差 ${deg(m.b.between)}；所以第 0 对得分 <b>${sc2(m.a.score)}</b> 对 <b>${sc2(m.b.score)}</b>，看得出顺序了。`);}
function notes(){
  return [tx('Toy run: pair 0 of a made-up head, turned the way the fastest pair of every Llama head turns, 1 radian (57.3°) per position. “bites” is at position 2, “dog” at 1 in “dog bites man” and at 3 in “man bites dog”.',
             '玩具实跑：一个虚构 head 的第 0 对，按每个 Llama head 里最快的那一对的速度转动，每个位置 1 弧度（57.3°）。“咬”在位置 2，“狗”在“狗咬人”里是位置 1，在“人咬狗”里是位置 3。'),
    tx('The score of this pair is q·k after the turn: 1.5 × 1.3 × cos(the angle between the arrows).',
       '这一对的得分是转动之后的 q·k：1.5 × 1.3 × cos(两个箭头的夹角)。'),
    tx('RoPE (rotary position embedding): Su et al., RoFormer, 2021. Llama models turn every query and key this way inside each attention layer; the value is not turned.',
       'RoPE（旋转位置编码）：Su 等，RoFormer，2021。Llama 模型在每一层的注意力里都这样转动 query 和 key；value 不转。')];}
const PT=(cx,cy,r,a)=>[cx+r*Math.cos(a),cy-r*Math.sin(a)];
function arrow(cx,cy,r,a,cls){const [x,y]=PT(cx,cy,r,a), h=6, b1=a+Math.PI*0.86, b2=a-Math.PI*0.86;
  return `<line x1="${f1(cx)}" y1="${f1(cy)}" x2="${f1(x)}" y2="${f1(y)}" class="${cls}"/><path d="M${f1(x)} ${f1(y)} L${f1(x+h*Math.cos(b1))} ${f1(y-h*Math.sin(b1))} L${f1(x+h*Math.cos(b2))} ${f1(y-h*Math.sin(b2))} Z" class="${cls} tz-rope-hd"/>`;}
function arc(cx,cy,r,a0,a1){const d=a1-a0; if(Math.abs(d)<1e-4) return ''; const [x0,y0]=PT(cx,cy,r,a0), [x1,y1]=PT(cx,cy,r,a1);
  return `<path d="M${f1(x0)} ${f1(y0)} A${r} ${r} 0 ${Math.abs(d)>Math.PI?1:0} ${d>0?0:1} ${f1(x1)} ${f1(y1)}" class="tz-rope-arc"/>`;}
function svg(W,on){
  const H=112, R=36, cy=18+R+4, xs=[W*0.25,W*0.75], m=model(on);
  let h='';
  [[m.a,tx('“dog bites man”','“狗咬人”')],[m.b,tx('“man bites dog”','“人咬狗”')]].forEach(([p,cap],i)=>{const cx=xs[i];
    h+=`<text x="${f1(cx)}" y="11" text-anchor="middle" class="tz-rope-cap">${esc(cap)}</text>`+
      `<circle cx="${f1(cx)}" cy="${f1(cy)}" r="${R}" class="tz-rope-dial"/>`+
      arc(cx,cy,R*0.4,p.ka*D2R,p.ka*D2R+p.between*D2R)+arrow(cx,cy,R*0.8,p.ka*D2R,'tz-rope-k')+arrow(cx,cy,R*0.92,p.qa*D2R,'tz-rope-q')+
      `<text x="${f1(cx)}" y="${f1(cy+R+15)}" text-anchor="middle" class="tz-rope-lab">${esc(tx(`${deg(p.between)} apart: ${sc2(p.score)}`,`相差 ${deg(p.between)}：${sc2(p.score)}`))}</text>`;});
  return {H,h};
}
A.registerTeaser('rope',{fid:'toy',render(el){
  el.innerHTML=`<div class="tz tz-rope">`+A.ui.seg({id:'tz-rope-seg',key:'tzrope',label:tx('RoPE','RoPE'),options:[['off',tx('off','关')],['on',tx('on','开')]],value:tzOn?'on':'off',inline:false})+
    `<svg class="tz-vis tz-rope-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span><i class="tz-rope-sw q" aria-hidden="true"></i>${esc(tx('solid: query of “bites”','实线：“咬”的 query'))}</span><span><i class="tz-rope-sw k" aria-hidden="true"></i>${esc(tx('dashed: key of “dog”','虚线：“狗”的 key'))}</span><span><i class="tz-rope-sw a" aria-hidden="true"></i>${esc(tx('arc: the angle between','弧线：夹角'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const vis=el.querySelector('svg'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  const help=tx('Two dials, one per sentence: pair 0 of the query of “bites” (solid) and of the key of “dog” (dashed); below each, the angle between them and this pair’s score.','两个表盘，每句话一个：“咬”的 query（实线）和“狗”的 key（虚线）的第 0 对；下面是它们的夹角和这一对的得分。');
  vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help);
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const draw=()=>{const W=Math.max(260,Math.round(el.clientWidth||320)), r=svg(W,tzOn); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzOn?'on':'off'); draw(); const h=readHTML(tzOn); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const on=b.dataset.v==='on'; if(on!==tzOn){tzOn=on; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/rope_numbers.py: the tile's numbers */
window.__tzRope=Object.freeze({model:on=>JSON.parse(JSON.stringify(model(!!on))),read:on=>readHTML(!!on)});
})();
