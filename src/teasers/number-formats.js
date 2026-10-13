/* Teaser for the number-formats lab's home tile: one value slider (1-2-5 steps, 1e-10 to 5e5) and the range of FP32, BF16, FP16
   and FP8 E4M3 on a log axis, with the value's line. One control (CONTRACT §5): the design's segmented format choice is replaced by
   showing all four strips at once. The rounding rule is the lab's own (labs/number-formats.js, quant): round to nearest, ties to
   even, subnormals, overflow above the largest number. It opens at 0.02, a typical weight, so it does not answer the lab's part 2
   prediction (R3.8). State lives in module scope, so a re-render (domain or language switch) keeps the learner's setting. */
(function(){
'use strict';
const A=window.ASP; if(!A) return;
const esc=A.esc;
const zfix=s=>String(s).replace(/([一-鿿])([A-Za-z0-9])/g,'$1 $2').replace(/([A-Za-z0-9%])([一-鿿])/g,'$1 $2');
const tx=(en,zh)=>A.lang()==='zh'?zfix(zh):en;
const NBH='‑', MINUS='−';
const FM={fp32:{n:'FP32',s:'FP32',e:8,m:23,b:127,k:'ieee'},bf16:{n:'BF16',s:'BF16',e:8,m:7,b:127,k:'ieee'},fp16:{n:'FP16',s:'FP16',e:5,m:10,b:15,k:'ieee'},
  e4m3:{n:'FP8 E4M3',s:'E4M3',e:4,m:3,b:7,k:'fn'}};
const IDS=['fp32','bf16','fp16','e4m3'];
Object.values(FM).forEach(f=>{f.emin=1-f.b; f.emax=(f.k==='ieee'?2**f.e-2:2**f.e-1)-f.b; f.top=f.k==='fn'?2-2*2**-f.m:2-2**-f.m;
  f.max=f.top*2**f.emax; f.minN=2**f.emin; f.minS=2**(f.emin-f.m);});
const fl2=a=>{let e=Math.floor(Math.log2(a)); if(2**e>a) e--; if(2**(e+1)<=a) e++; return e;};
function quant(x,id){const f=FM[id], a=Math.abs(x); if(a===0) return {v:0,kind:'zero'};
  let ex=fl2(a); if(ex<f.emin) ex=f.emin;
  const q=2**(ex-f.m), n=a/q, lo=Math.floor(n), d=n-lo, k=d>0.5?lo+1:(d<0.5?lo:(lo%2===0?lo:lo+1)), v=k*q;
  if(v>f.max) return {v:f.k==='ieee'?Infinity:NaN,kind:'over'};
  if(v===0) return {v:0,kind:'lost'};
  return {v:x<0?-v:v,kind:v<f.minN?'sub':'norm'};}
const XS=[]; for(let d=-10;d<=5;d++) [1,2,5].forEach(m=>XS.push(parseFloat(`${m}e${d}`)));
let tzI=XS.indexOf(0.02);   // opens at a typical weight, 0.02
const fint=v=>Math.round(v).toLocaleString('en-US');
const sci=(v,sig)=>{const [m,e]=Math.abs(v).toExponential(sig-1).split('e'); return (v<0?MINUS:'')+m+'e'+(+e<0?NBH+(-e):(+e));};
const fin=x=>{if(x>=1e-3&&x<1e6) return x>=1000?fint(x):String(+x.toPrecision(6)); return sci(x,1);};
function fv(v){if(v===0) return '0'; if(v===Infinity) return 'Inf'; if(Number.isNaN(v)) return 'NaN';
  const a=Math.abs(v); if(a>=1e-3&&a<1e6){const s=a.toPrecision(8); const t=s.indexOf('.')>=0?s.replace(/0+$/,'').replace(/\.$/,''):s; const [i,dd]=t.split('.'); return (v<0?MINUS:'')+i.replace(/\B(?=(\d{3})+(?!\d))/g,',')+(dd?'.'+dd:'');}
  return sci(v,4);}   // the lab's own display: 8 significant digits, or 4 in scientific notation
const pct=e=>{const p=e*100; if(p===0) return '0%'; if(p<0.01) return '<0.01%'; return (p<0.1?p.toPrecision(2):(p<10?p.toFixed(2):p.toFixed(1)))+'%';};
const list=(a,zh)=>zh?a.join('、'):(a.length<3?a.join(' and '):a.slice(0,-1).join(', ')+' and '+a[a.length-1]);
function model(i){const x=XS[i];
  const rows=IDS.map(id=>{const r=quant(x,id); return {id,n:FM[id].n,v:r.v,kind:r.kind,bad:r.kind==='lost'||r.kind==='over',err:r.kind==='lost'||r.kind==='over'?null:Math.abs(r.v-x)/x};});
  return {x,i,rows};}
function readHTML(i){const m=model(i), X=fin(m.x), bad=m.rows.filter(r=>r.bad), ok=m.rows.filter(r=>!r.bad), zh=A.lang()==='zh';
  if(!bad.length){const w=ok.reduce((a,b)=>b.err>a.err?b:a);
    if(w.err===0) return tx(`${X} is stored exactly in all four formats.`,`${X} 在四种格式里都能精确存下。`);
    if(w.err>0.1) return tx(`${X} fits all four formats, but ${w.n} keeps only a rough <b>${fv(w.v)}</b> (${pct(w.err)} off).`,`${X} 四种格式都存得下，但 ${w.n} 只存下一个粗略的 <b>${fv(w.v)}</b>（误差 ${pct(w.err)}）。`);
    return tx(`${X} fits all four formats; ${w.n} rounds it most, to <b>${fv(w.v)}</b> (${pct(w.err)} off).`,`${X} 四种格式都存得下；${w.n} 舍入得最多，存成 <b>${fv(w.v)}</b>（误差 ${pct(w.err)}）。`);}
  const bn=list(bad.map(r=>r.n),zh), kn=list(ok.map(r=>r.n),zh), bf=ok.find(r=>r.id==='bf16'), lost=bad[0].kind==='lost';
  const bfs=bf?tx(` (BF16 as <b>${fv(bf.v)}</b>)`,`（BF16 存成 <b>${fv(bf.v)}</b>）`):'';
  if(lost) return tx(`${X} becomes <b>0</b> in ${bn}; ${kn} keep it${bfs}.`,`${X} 在 ${bn} 里变成 <b>0</b>；${kn} 留得住${bfs}。`);
  return tx(`${X} overflows ${bn}; ${kn} hold it${bfs}.`,`${X} 超出了 ${bn} 的范围；${kn} 存得下${bfs}。`);}
function notes(){
  return [tx('Formula: each format rounds the value to its nearest storable number, ties to even. Bits: FP32 has 8 exponent and 23 mantissa bits, BF16 8 and 7, FP16 5 and 10, FP8 E4M3 4 and 3.',
             '公式计算：每种格式把这个数舍入到最近的可存的数，平局取偶。位数：FP32 有 8 个指数位、23 个尾数位，BF16 是 8 和 7，FP16 是 5 和 10，FP8 E4M3 是 4 和 3。'),
    tx('The axis is a log scale from 1e‑12 to 1e7. The hatched part holds subnormal numbers: numbers below the format’s smallest normal number, still storable but with fewer and fewer significant digits. FP32 and BF16 reach past the chart: FP32 down to 1.4e‑45, BF16 down to 9.2e‑41, both up to about 3.4e38.',
       '横轴是对数刻度，从 1e‑12 到 1e7。斜纹部分是次正规数：比格式最小的正规数还小、精度越来越差但还存得下的数。FP32 和 BF16 的范围比图还宽：FP32 最小到 1.4e‑45，BF16 最小到 9.2e‑41，两者最大都约 3.4e38。'),
    tx('A value below half the smallest number becomes 0; one above the largest overflows. FP16’s largest is 65,504; FP8 E4M3’s is 448, and it has no Inf.',
       '比最小的数的一半还小的数会变成 0；比最大的数还大就溢出。FP16 最大 65,504；FP8 E4M3 最大 448，而且没有 Inf。')];}
const AX0=-12, AX1=7, pos=v=>Math.max(0,Math.min(1,(Math.log10(v)-AX0)/(AX1-AX0)));
const f1=v=>Math.round(v*10)/10;
function svg(W,i){
  const m=model(i), SL=46, x0=SL, x1=W-4, PX=p=>x0+(x1-x0)*p, top=4, sh=12, gap=6, H=top+4*sh+3*gap+20;
  let h='<defs><pattern id="tz-nf-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="1.5" height="4" class="tz-nf-hl"/></pattern></defs>';
  m.rows.forEach((r,k)=>{const f=FM[r.id], y=top+k*(sh+gap), a=PX(pos(f.minS)), b=PX(pos(f.minN)), c=PX(pos(f.max));
    h+=`<text x="0" y="${f1(y+sh-2)}" class="tz-nf-lab">${esc(f.s)}</text><rect x="${x0}" y="${y}" width="${f1(x1-x0)}" height="${sh}" class="tz-nf-bg"/>`+
      `<rect x="${f1(a)}" y="${y}" width="${f1(Math.max(0,b-a))}" height="${sh}" fill="url(#tz-nf-hatch)"/><rect x="${f1(b)}" y="${y}" width="${f1(Math.max(0,c-b))}" height="${sh}" class="tz-nf-norm"/>`;
    if(r.bad) h+=`<circle cx="${f1(PX(pos(m.x)))}" cy="${f1(y+sh/2)}" r="4.5" class="tz-nf-bad"/>`;});
  const vx=f1(PX(pos(m.x))), yb=top+4*sh+3*gap;
  h+=`<line x1="${vx}" y1="0" x2="${vx}" y2="${yb+2}" class="tz-nf-v"/>`;
  [-12,-8,-4,0,4].forEach((e,j)=>{const xx=f1(PX((e-AX0)/(AX1-AX0))); h+=`<text x="${xx}" y="${yb+15}" text-anchor="${j===0?'start':'middle'}">${e===0?'1':'1e'+(e<0?NBH+(-e):e)}</text>`;});
  return {H,h};
}
A.registerTeaser('number-formats',{fid:'formula',render(el){
  const lab=tx('Value to store','要存的数');
  el.innerHTML=`<div class="tz tz-nf"><label class="tz-lab"><span>${esc(lab)}</span><output class="mono"></output></label>`+
    `<input type="range" class="tz-nf-in" min="0" max="${XS.length-1}" step="1" value="${tzI}" aria-label="${esc(lab)}">`+
    `<svg class="tz-vis tz-nf-vis" role="img" tabindex="0" xmlns="http://www.w3.org/2000/svg"></svg>`+
    `<div class="tz-leg"><span tabindex="0" data-tipx="${esc(tx('Normal numbers: almost all numbers a format can store. Each keeps the same number of significant digits.','正规数：格式能存的数里绝大多数是正规数，每个都有一样多的有效数字。'))}"><i class="sw tz-nf-sw-n" aria-hidden="true"></i>${esc(tx('normal numbers','正规数'))}</span><span tabindex="0" data-tipx="${esc(tx('Subnormal numbers: below the format’s smallest normal number, still storable, but the closer to 0, the fewer significant digits.','次正规数：比格式最小的正规数还小、精度越来越差但还存得下的数。'))}"><i class="sw tz-nf-sw-s" aria-hidden="true"></i>${esc(tx('subnormal','次正规数'))}</span><span><i class="sw tz-nf-sw-v" aria-hidden="true"></i>${esc(tx('the value','这个数'))}</span><span><i class="sw tz-nf-sw-b" aria-hidden="true"></i>${esc(tx('lost or overflow','丢失或溢出'))}</span></div>`+
    `<p class="tz-read"></p><p class="tz-note"></p><p class="tz-note"></p><p class="tz-note"></p></div>`;
  const inp=el.querySelector('input'), out=el.querySelector('output'), vis=el.querySelector('svg'), rd=el.querySelector('.tz-read');
  el.querySelectorAll('.tz-note').forEach((p,k)=>{p.textContent=notes()[k]||'';});
  const draw=()=>{const W=Math.max(260,Math.round(el.clientWidth||320)), r=svg(W,tzI); vis.setAttribute('viewBox',`0 0 ${W} ${r.H}`); vis.setAttribute('width',W); vis.setAttribute('height',r.H); vis.innerHTML=r.h;};
  const upd=user=>{tzI=+inp.value; if(A.ui.setRange) A.ui.setRange(inp); const X=fin(XS[tzI]); out.textContent=X; inp.setAttribute('aria-valuetext',X);
    const help=tx(`The range of FP32, BF16, FP16 and FP8 E4M3 on a log axis, with a line at ${X}.`,`FP32、BF16、FP16 和 FP8 E4M3 在对数轴上的范围，竖线在 ${X}。`);
    vis.setAttribute('aria-label',help); vis.setAttribute('data-tipx',help); draw();
    const h=readHTML(tzI); if(rd._h!==h){rd._h=h; rd.innerHTML=h; if(user) A.announce(rd.textContent);}};
  inp.addEventListener('input',()=>upd(true)); upd(false);
}});
/* read-only hook for tests/number-formats_numbers.py: the tile's numbers */
window.__tzNf=Object.freeze({xs:XS.slice(),start:XS.indexOf(0.02),model:i=>JSON.parse(JSON.stringify(model(i),(k,v)=>typeof v==='number'&&!isFinite(v)?String(v):v)),read:i=>readHTML(i)});
})();
