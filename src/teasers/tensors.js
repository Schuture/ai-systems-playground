/* Teaser for the tensors lab's home tile: pick a pair of shapes and see them lined up from the right, axis by axis. A stretched axis
   (size 1, or missing) is a dashed box "1 → 4"; a mismatch is outlined in red with "3 ≠ 4". One control, one small visual (three rows
   of axis boxes, under 130 px), one sentence in .tz-read, a one-line legend, and plain-text notes. The rule is NumPy's (and PyTorch's):
   compare sizes from the last axis; equal, or one of them 1, else an error. The tile never shows (3, 1) + (1, 4), the lab's
   prediction in part 2 (R3.8): every pair here stretches at most one operand.
   State lives in module scope, so a re-render (domain or language switch) keeps the learner's pick. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const PAIRS=[{id:'p1',a:[4,3],b:[3]},{id:'p2',a:[4,3],b:[4]},{id:'p3',a:[4,3],b:[4,1]},{id:'p4',a:[2,4,3],b:[4,3]}];
let tzPick='p1';
const shp=a=>'('+a.join(', ')+(a.length===1?',':'')+')';
/* one entry per axis of the result, left to right; k counts from the right */
function model(id){const p=PAIRS.find(x=>x.id===id)||PAIRS[0], a=p.a, b=p.b, n=Math.max(a.length,b.length), ax=[];
  for(let q=0;q<n;q++){const k=n-1-q, i=a.length-1-k, j=b.length-1-k, x=i>=0?a[i]:null, y=j>=0?b[j]:null, xa=x==null?1:x, ya=y==null?1:y;
    let out=null, bad=false; if(xa===ya) out=xa; else if(xa===1) out=ya; else if(ya===1) out=xa; else bad=true;
    ax.push({k,x,y,out,bad,sa:!bad&&xa===1&&out>1,sb:!bad&&ya===1&&out>1});}
  const bads=ax.filter(c=>c.bad);
  return {id:p.id,a:a.slice(),b:b.slice(),n,ax,ok:!bads.length,err:bads.length?bads[bads.length-1]:null,out:bads.length?null:ax.map(c=>c.out)};}
function readHTML(id){const m=model(id), a=shp(m.a), b=shp(m.b);
  if(id==='p1') return tx(`<b>${a} + ${b} gives ${shp(m.out)}</b>: shapes line up from the right, and b’s missing first axis stretches to 4.`,
                          `<b>${a} + ${b} 得到 ${shp(m.out)}</b>：形状从右边对齐，b 缺的第一个轴拉伸成 4。`);
  if(id==='p2') return tx(`<b>${a} + ${b} is an error</b>: lined up from the right, the last axes are 3 and 4, and neither is 1.`,
                          `<b>${a} + ${b} 报错</b>：从右边对齐后，最后一个轴是 3 和 4，两个都不是 1。`);
  if(id==='p3') return tx(`<b>${a} + ${b} gives ${shp(m.out)}</b>: b’s last axis has size 1, so its one column is read 3 times, not copied.`,
                          `<b>${a} + ${b} 得到 ${shp(m.out)}</b>：b 的最后一个轴大小是 1，那一列被读 3 次，不复制。`);
  return tx(`<b>${a} + ${b} gives ${shp(m.out)}</b>: the last two axes match, and b’s missing first axis stretches to 2.`,
            `<b>${a} + ${b} 得到 ${shp(m.out)}</b>：最后两个轴一样大，b 缺的第一个轴拉伸成 2。`);}
function notes(){
  return [tx('Shapes only: NumPy’s broadcasting rule for every elementwise operation; PyTorch follows the same rule.','只看形状：这是 NumPy 对所有逐元素运算的广播规则，PyTorch 用同样的规则。'),
    tx('A stretched axis is read again with a stride of 0, so nothing is copied; only the result is a new array.','被拉伸的轴用步长 0 重复读取，不复制任何数据；只有结果是新数组。')];}
function cell(v,cls,tip){return `<span class="tz-tensors-ax${cls?' '+cls:''}">${esc(String(v))}</span>`;}
function visHTML(id){const m=model(id), cols=`grid-template-columns:minmax(0,auto) repeat(${m.n},minmax(0,1fr))`;
  const row=(lab,sh,who,get,st)=>`<span class="tz-tensors-rl">${esc(lab)} <span>${esc(shp(sh))}</span></span>`+m.ax.map(c=>{const v=get(c), s=st(c);
    if(v==null&&s) return cell(`(1) → ${c.out}`,'tz-tensors-gh',tx(`${who} has no axis here: it counts as 1 and stretches to ${c.out}.`,`${who} 在这里没有轴：算作 1，拉伸成 ${c.out}。`));
    if(v==null) return cell('–','tz-tensors-none');
    if(s) return cell(`1 → ${c.out}`,'tz-tensors-gh',tx(`${who} has size 1 here: read ${c.out} times, not copied.`,`${who} 在这里大小为 1：读 ${c.out} 次，不复制。`));
    return cell(v,c.bad?'tz-tensors-bad':'');}).join('');
  let h=`<div class="tz-tensors-vis" role="img" tabindex="0" style="${cols}">`;
  h+=row('a',m.a,'a',c=>c.x,c=>c.sa)+row('b',m.b,'b',c=>c.y,c=>c.sb);
  h+=`<span class="tz-tensors-rl tz-tensors-rout">a + b</span>`+m.ax.map(c=>c.bad?cell(`${c.x} ≠ ${c.y}`,'tz-tensors-bad tz-tensors-x'):cell(c.out,'tz-tensors-out')).join('');
  return h+`</div>`;}
A.registerTeaser('tensors',{fid:'formula',render(el){
  el.innerHTML=`<div class="tz tz-tensors">`+A.ui.seg({id:'tz-tensors-seg',key:'tztensors',label:tx('Shapes','形状'),options:PAIRS.map(p=>[p.id,`${shp(p.a)} + ${shp(p.b)}`]),value:tzPick,inline:false})+
    `<div class="tz-tensors-host"></div>`+
    `<div class="tz-leg"><span><i class="sw tz-tensors-sw" aria-hidden="true"></i>${esc(tx('box: an axis and its size','方框：一个轴和它的大小'))}</span><span><i class="sw tz-tensors-sw tz-tensors-sw-gh" aria-hidden="true"></i>${esc(tx('dashed: stretched','虚线：被拉伸'))}</span><span><i class="sw tz-tensors-sw tz-tensors-sw-bad" aria-hidden="true"></i>${esc(tx('red: sizes do not match','红色：大小对不上'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const host=el.querySelector('.tz-tensors-host'), rd=el.querySelector('.tz-read'), grp=el.querySelector('.seg-ctl');
  el.querySelectorAll('.tz-note').forEach((p,i)=>{p.textContent=notes()[i]||'';});
  const upd=user=>{if(grp&&A.ui.setPressed) A.ui.setPressed(grp,tzPick); const m=model(tzPick);
    host.innerHTML=visHTML(tzPick); const v=host.firstChild, help=tx(`a ${shp(m.a)} and b ${shp(m.b)}, one box per axis, lined up from the right; the last row is a + b.`,`a ${shp(m.a)} 和 b ${shp(m.b)}，每个轴一个方框，从右边对齐；最后一行是 a + b。`);
    v.setAttribute('aria-label',help+' '+rd.textContent); v.setAttribute('data-tipx',help);
    const h=readHTML(tzPick); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);} v.setAttribute('aria-label',help+' '+rd.textContent);};
  el.addEventListener('click',e=>{const b=e.target.closest('.seg-ctl button'); if(!b||!el.contains(b)) return; const v=b.dataset.v; if(v&&v!==tzPick&&PAIRS.some(p=>p.id===v)){tzPick=v; upd(true);}});
  upd(false);
}});
/* read-only hook for tests/tensors_numbers.py: the tile's shapes */
window.__tzTs=Object.freeze({pairs:()=>JSON.parse(JSON.stringify(PAIRS)),model:id=>JSON.parse(JSON.stringify(model(id))),read:id=>readHTML(id)});
})();
