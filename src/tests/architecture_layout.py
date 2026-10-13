"""Layout audit of the Architecture overview v2: the home hero and tile, the lab overview, every pop-up step, the rail.

python3 tests/architecture_layout.py [--quick] [--no-hover]
Builds out/architecture-layout-build.html with this lab only (--only architecture). For each viewport (1440×900, 1280×800, 900×900, 761×900, 390×844), language (EN, 中文) and theme
(light, dark) it checks:
  - text over text: every visible text line box (HTML text nodes, SVG <text>, the canvas texts) against the others;
  - lines and particles over text: SVG paths, lines and shape outlines sampled with getPointAtLength; the canvas paths and particles
    from window.__ar.geom(), which runs the same drawing code on a recording context (hero: each stage at several animation frames;
    particles over 2 s of animation). A line may end at its own label: samples within 10 px of an open line's ends do not count;
  - spill: canvas texts outside the canvas or outside their stage's panel; SVG texts outside their SVG; anything outside the dialog
    or the hero card; sideways scroll of the page or of a box inside it; the sticky stage scrolling inside itself;
  - hover (1440 and 390, both languages, light): every element with data-tipx, data-tip, title or the kit's .t, and every element with
    a pointer or help cursor; a tooltip with no text, or an element with no text that grows a box, outline or background on hover;
  - (added after the visual QA of v2) a button or chip whose short label breaks onto two lines (“滑动窗 / 口”); text cut off by an
    ancestor that hides overflow; canvas lines and particles over a pill (a filled shape holding a text, e.g. a flying token) and
    lines over probability bars; the hero checked at 40 frames per stage, so tokens in mid-flight are seen; empty regions: the largest
    empty rectangle of each pop-up diagram and of the hero's open stage (flagged above 30% of the area and a third of each side),
    a phone pop-up whose lower part is empty, and a settings column much shorter than the stage beside it; the pop-up controls
    (every token of steps 5 and 6, the SiLU probe and the temperature at both ends) are set before the checks.
Fonts: the page asks Google Fonts for Archivo and IBM Plex Mono; set AR_FONTS to a local fonts.css (default:
../qa-v11/fonts/fonts.css next to the project, if present) so the widths match production. Without it the fallback fonts are used.
Prints the hits (grouped) and their count; exits 1 if there are any."""
import os, pathlib, subprocess, sys, json
from collections import defaultdict
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
QUICK = '--quick' in sys.argv
HOVER = '--no-hover' not in sys.argv
subprocess.run([sys.executable, str(P / 'build.py'), 'out/architecture-layout-build.html', '--only', 'architecture'], cwd=P, check=True, capture_output=True)
PAGE = P / 'out' / f'architecture-layout-{os.getpid()}.html'
PAGE.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                + (P / 'out' / 'architecture-layout-build.html').read_text() + '</body></html>')
URL = 'file://' + str(PAGE)
FONTS = pathlib.Path(os.environ.get('AR_FONTS', str(P.parent / 'qa-v11' / 'fonts' / 'fonts.css')))
FONT_CSS = FONTS.read_text() if FONTS.exists() else None
print('fonts:', FONTS if FONT_CSS else 'fallback (set AR_FONTS for production widths)', flush=True)
VIEWPORTS = [(1440, 900), (390, 844)] if QUICK else [(1440, 900), (1280, 800), (900, 900), (761, 900), (390, 844)]
THEMES = ['light'] if QUICK else ['light', 'dark']
for a in sys.argv[1:]:
    if a.startswith('--vp='):        # e.g. --vp=1280x800,900x900
        VIEWPORTS = [tuple(int(v) for v in x.split('x')) for x in a[5:].split(',')]
    if a.startswith('--theme='):     # light or dark
        THEMES = a[8:].split(',')
hits = []

JS = r"""
(()=>{
const ids=new WeakMap(); let nid=0; const uid=e=>{if(!ids.has(e)) ids.set(e,++nid); return ids.get(e);};
const vis=e=>{if(!e||!e.getClientRects().length) return false; for(let a=e;a&&a.nodeType===1;a=a.parentElement){const cs=getComputedStyle(a); if(cs.visibility==='hidden'||cs.display==='none'||+cs.opacity<0.05) return false; if(a.hidden) return false;} return true;};
function texts(root){const out=[], R=document.createRange(), tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const s=n.textContent; if(!s.trim()) continue; const e=n.parentElement; if(!e||e.closest('svg')||e.closest('#tip')||!vis(e)) continue;
    R.selectNodeContents(n); for(const r of R.getClientRects()){if(r.width<1.5||r.height<1.5) continue; out.push({t:s.trim().slice(0,40),x0:r.left,y0:r.top,x1:r.right,y1:r.bottom,el:uid(e)});}}
  root.querySelectorAll('svg text').forEach(t=>{if(!vis(t)||!t.textContent.trim()) return; const r=t.getBoundingClientRect(); if(r.width<1) return; out.push({t:t.textContent.trim().slice(0,40),x0:r.left,y0:r.top,x1:r.right,y1:r.bottom,el:uid(t),svg:1});});
  return out;}
function overlaps(bs,tol){tol=tol==null?1.5:tol; const out=[]; for(let i=0;i<bs.length;i++) for(let j=i+1;j<bs.length;j++){const a=bs[i], b=bs[j]; if(a.el&&a.el===b.el) continue;
  const ix=Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0), iy=Math.min(a.y1,b.y1)-Math.max(a.y0,b.y0); if(ix>tol&&iy>tol) out.push(`“${a.t}” × “${b.t}”`);} return out;}
function svgLines(root){const out=[]; root.querySelectorAll('svg').forEach(svg=>{if(!vis(svg)) return; svg.querySelectorAll('path,line,polyline,rect,circle,ellipse').forEach(el=>{
  if(el.closest('defs,pattern,clipPath,mask')) return; const cs=getComputedStyle(el); if(!cs.stroke||cs.stroke==='none'||parseFloat(cs.strokeWidth)===0||parseFloat(cs.strokeOpacity)<0.05) return; if(!vis(el)) return;
  let Ln=0; try{Ln=el.getTotalLength();}catch(e){return;} if(!(Ln>0)) return; const M=el.getScreenCTM(); if(!M) return; const n=Math.max(2,Math.ceil(Ln/3)), pts=[];
  for(let i=0;i<=n;i++){const p=el.getPointAtLength(Ln*i/n); pts.push([M.a*p.x+M.c*p.y+M.e,M.b*p.x+M.d*p.y+M.f]);}
  const tag=el.tagName.toLowerCase(), open=(tag==='line'||tag==='polyline'||(tag==='path'&&!/z\s*$/i.test(el.getAttribute('d')||'')));
  out.push({pts,open,w:parseFloat(cs.strokeWidth)||1,name:tag+(el.getAttribute('class')?'.'+el.getAttribute('class'):'')});});}); return out;}
function lineHits(lines,tb,endTol,hidden){endTol=endTol==null?10:endTol; const out=new Set();
  for(const l of lines){const p=l.pts; let acc=[0]; for(let i=1;i<p.length;i++) acc.push(acc[i-1]+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1])); const L=acc[acc.length-1];
    for(let i=0;i<p.length;i++){if(l.open&&(acc[i]<endTol||L-acc[i]<endTol)) continue; const x=p[i][0], y=p[i][1];
      for(const t of tb){if(x>t.x0+1&&x<t.x1-1&&y>t.y0+1&&y<t.y1-1&&!(hidden&&hidden(l,t))){out.add(`${l.name||'line'} over “${t.t}”`);}}}} return [...out];}
function dotHits(dots,tb){const out=new Set(); for(const d of dots) for(const t of tb){const r=d[2]*0.5; if(d[0]>t.x0-r&&d[0]<t.x1+r&&d[1]>t.y0-r&&d[1]<t.y1+r) out.add(`particle over “${t.t}”`);} return [...out];}
function spill(root,boxSel){const out=[]; const B=boxSel?document.querySelector(boxSel):root; if(!B) return out; const b=B.getBoundingClientRect();
  root.querySelectorAll('svg').forEach(svg=>{if(!vis(svg)) return; const s=svg.getBoundingClientRect(); svg.querySelectorAll('text').forEach(t=>{if(!vis(t)) return; const r=t.getBoundingClientRect(); if(r.width<1) return;
    if(r.left<s.left-1||r.right>s.right+1||r.top<s.top-1||r.bottom>s.bottom+1) out.push(`svg text “${t.textContent.trim().slice(0,30)}” outside its svg`);});});
  root.querySelectorAll('*').forEach(e=>{if(e.closest('svg')&&e.tagName.toLowerCase()!=='svg') return; if(!vis(e)) return; const r=e.getBoundingClientRect(); if(r.width<1||r.height<1) return;
    if(r.left<b.left-1||r.right>b.right+1) out.push(`${e.tagName.toLowerCase()}${e.id?'#'+e.id:''}${e.className&&typeof e.className==='string'?'.'+e.className.split(' ')[0]:''} spills sideways out of ${boxSel||'its box'}`);});
  root.querySelectorAll('*').forEach(e=>{if(!vis(e)) return; const cs=getComputedStyle(e); if((cs.overflowX==='auto'||cs.overflowX==='scroll')&&e.scrollWidth>e.clientWidth+1) out.push(`${e.className||e.tagName} scrolls sideways (${e.scrollWidth} > ${e.clientWidth})`);});
  return out;}
function canvasAudit(g,domTexts,opts){opts=opts||{}; const out=[]; if(!g) return out; const tb=g.texts.concat(domTexts||[]);
  overlaps(tb).forEach(x=>out.push('text over text: '+x));
  /* a line drawn before the opaque pill a text sits on is hidden under the pill */
  const pl=pillsOf(g), hid=(l,t)=>t.seq!=null&&l.seq!=null&&pl.some(f=>f.seq>l.seq&&t.x0>=f.x0-3&&t.x1<=f.x1+3&&t.y0>=f.y0-4&&t.y1<=f.y1+4);
  lineHits(g.lines,tb,10,hid).forEach(x=>out.push(x));
  dotHits(g.dots,tb).forEach(x=>out.push(x));
  const c=g.canvas; g.texts.forEach(t=>{if(t.x0<c.x0-1||t.x1>c.x1+1||t.y0<c.y0-1||t.y1>c.y1+1) out.push(`canvas text “${t.t}” outside the canvas`);
    if(opts.panels&&t.panel>=0){const p=g.panels[t.panel]; if(p&&(t.x0<p.x0-1||t.x1>p.x1+1||t.y0<p.y0-1||t.y1>p.y1+1)) out.push(`canvas text “${t.t}” outside its stage`);}});
  return out;}
function wraps(root){const out=[]; root.querySelectorAll('button,[role=radio],[role=tab],.chip,.ar-tag,.btn').forEach(b=>{if(!vis(b)) return; const t=(b.innerText||'').trim();
  if(!t||t.length>22||/\n/.test(t)) return; const R=document.createRange(); R.selectNodeContents(b); const tops=[...R.getClientRects()].filter(r=>r.width>=1&&r.height>=1).map(r=>r.top).sort((a,b)=>a-b);
  let lines=tops.length?1:0; for(let i=1;i<tops.length;i++) if(tops[i]-tops[i-1]>6) lines++; if(lines>1) out.push(`“${t}” breaks onto ${lines} lines inside its button`);}); return out;}
function clipped(root){const out=new Set(), cl=[...root.querySelectorAll('*')].filter(e=>{if(e.tagName.toLowerCase()==='svg'||e.closest('svg')) return false; const cs=getComputedStyle(e);
    if(!/hidden|clip/.test(cs.overflowX+' '+cs.overflowY)||!vis(e)) return false; const r=e.getBoundingClientRect(); return r.width>4&&r.height>4;});
  if(!cl.length) return []; const R=document.createRange();
  const boxes=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())){const e=n.parentElement; if(!n.textContent.trim()||!e||e.closest('svg')||!vis(e)) continue; R.selectNodeContents(n); for(const r of R.getClientRects()) if(r.width>=1.5) boxes.push({e,t:n.textContent.trim().slice(0,30),r});}
  root.querySelectorAll('svg text').forEach(t=>{if(vis(t)&&t.textContent.trim()) boxes.push({e:t,t:t.textContent.trim().slice(0,30),r:t.getBoundingClientRect()});});
  for(const b of boxes) for(const a of cl){if(!a.contains(b.e)) continue; const r=a.getBoundingClientRect(), cs=getComputedStyle(a), hx=/hidden|clip/.test(cs.overflowX), hy=/hidden|clip/.test(cs.overflowY);
    if((hx&&(b.r.left<r.left-1||b.r.right>r.right+1))||(hy&&(b.r.top<r.top-1||b.r.bottom>r.bottom+1))) out.add(`text “${b.t}” is cut off by ${typeof a.className==='string'&&a.className?'.'+a.className.split(' ')[0]:a.tagName.toLowerCase()}`);}
  return [...out];}
function emptiest(area,boxes,nx,ny){const W=area.x1-area.x0, H=area.y1-area.y0; if(W<20||H<20) return {frac:0,fw:0,fh:0}; const cw=W/nx, ch=H/ny, occ=Array.from({length:ny},()=>new Array(nx).fill(0));
  for(const b of boxes){if(b.x1<area.x0||b.x0>area.x1||b.y1<area.y0||b.y0>area.y1) continue; const i0=Math.max(0,Math.floor((b.x0-area.x0)/cw)), i1=Math.min(nx-1,Math.floor((b.x1-area.x0-0.01)/cw)), j0=Math.max(0,Math.floor((b.y0-area.y0)/ch)), j1=Math.min(ny-1,Math.floor((b.y1-area.y0-0.01)/ch));
    for(let j=j0;j<=j1;j++) for(let i=i0;i<=i1;i++) occ[j][i]=1;}
  const h=new Array(nx).fill(0); let best={a:0,w:0,h:0};
  for(let j=0;j<ny;j++){for(let i=0;i<nx;i++) h[i]=occ[j][i]?0:h[i]+1;
    for(let i=0;i<nx;i++){let mh=h[i]; for(let k=i;k<nx&&mh>0;k++){mh=Math.min(mh,h[k]); const a=mh*(k-i+1); if(a>best.a) best={a,w:k-i+1,h:mh};}}}
  return {frac:best.a/(nx*ny),fw:best.w/nx,fh:best.h/ny};}
const tooEmpty=e=>e.frac>0.3&&e.fw>=0.34&&e.fh>=0.34;
/* review 12: in the hero, the loop back to the tokenization keeps at least 12 px from every frame of the “× 32 layers” stack */
function loopClear(g,min){min=min||12; const st=g.lines.filter(l=>l.tag==='stack'), lp=g.lines.filter(l=>l.tag==='loop').flatMap(l=>l.pts); if(!st.length||!lp.length) return [];
  let best=1e9; const segD=(p,a,b)=>{const dx=b[0]-a[0], dy=b[1]-a[1], L2=dx*dx+dy*dy, t=L2?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/L2)):0; return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);};
  for(const q of lp) for(const l of st) for(let i=1;i<l.pts.length;i++) best=Math.min(best,segD(q,l.pts[i-1],l.pts[i]));
  return best<min?[`the loop comes within ${best.toFixed(1)} px of the layer stack (needs ${min})`]:[];}
/* review 12: each lane runs without a gap from the embedding into the layer card, down the residual stream, through the final RMSNorm
   into the LM head: its pieces join end to end, nothing opaque drawn later hides part of it (the RMSNorm pill excepted), it ends at
   the LM head's top edge, and no piece sticks out on its own */
function laneCont(g){const out=[], head=(g.fills||[]).find(f=>f.tag==='head');
  for(let k=0;k<5;k++){const ls=g.lines.filter(l=>l.tag==='lane'+k); if(!ls.length){out.push(`lane ${k+1} is not drawn`); continue;}
    const P=ls.map(l=>({a:l.pts[0],b:l.pts[l.pts.length-1],seq:l.seq,pts:l.pts})).sort((p,q)=>p.a[1]-q.a[1]);
    for(let i=1;i<P.length;i++){const d=Math.hypot(P[i].a[0]-P[i-1].b[0],P[i].a[1]-P[i-1].b[1]); if(d>2.5) out.push(`lane ${k+1} has a gap of ${Math.round(d)} px`);}
    for(const p of P) for(const f of g.fills||[]){if(!f.op||f.seq<=p.seq||f.tag==='norm') continue; const hid=p.pts.filter(q=>q[0]>f.x0+1&&q[0]<f.x1-1&&q[1]>f.y0+1&&q[1]<f.y1-1).length; if(hid>1){out.push(`lane ${k+1} is hidden under a shape`); break;}}
    const end=P[P.length-1].b; if(head&&(Math.abs(end[1]-head.y0)>3||end[0]<head.x0||end[0]>head.x1)) out.push(`lane ${k+1} does not end at the LM head`);}
  return [...new Set(out)];}
/* review 11: the loop from the next token back to the prompt runs in its own lane, clear of the layer stack */
function loopLane(g){const box=ls=>{const p=ls.flatMap(l=>l.pts); if(!p.length) return null; return {x0:Math.min(...p.map(v=>v[0])),y0:Math.min(...p.map(v=>v[1])),x1:Math.max(...p.map(v=>v[0])),y1:Math.max(...p.map(v=>v[1]))};};
  const st=box(g.lines.filter(l=>l.tag==='stack')), lp=g.lines.filter(l=>l.tag==='loop').flatMap(l=>l.pts), hits=[]; if(!st||!lp.length) return hits;
  const inS=p=>p[0]>st.x0-8&&p[0]<st.x1+8&&p[1]>st.y0&&p[1]<st.y1; if(lp.some(inS)) hits.push('the loop back to the prompt runs over the layer stack'); return hits;}
function svgEmpty(root){const out=[]; root.querySelectorAll('svg.ar-psvg').forEach(svg=>{if(!vis(svg)) return; const a=svg.getBoundingClientRect(), A={x0:a.left,y0:a.top,x1:a.right,y1:a.bottom}, area=(A.x1-A.x0)*(A.y1-A.y0);
    const boxes=[...svg.querySelectorAll('text,rect,circle,ellipse,line,path,polyline')].filter(e=>!e.closest('defs,pattern')&&vis(e)).map(e=>{const r=e.getBoundingClientRect(); return {x0:r.left,y0:r.top,x1:r.right,y1:r.bottom};}).filter(b=>(b.x1-b.x0)*(b.y1-b.y0)<0.5*area);
    const E=emptiest(A,boxes,20,10); if(tooEmpty(E)) out.push(`diagram “${(svg.getAttribute('aria-label')||'').slice(0,40)}” is ${Math.round(E.frac*100)}% empty in one block (${Math.round(E.fw*100)}% × ${Math.round(E.fh*100)}%)`);}); return out;}
function dialogEmpty(){const d=document.querySelector('#ar-modal .ar-dialog'), b=d&&d.querySelector('.ar-dlg-body'); if(!b||!vis(d)) return []; const dr=d.getBoundingClientRect();
  let bot=0; for(const c of b.children){if(!vis(c)) continue; bot=Math.max(bot,c.getBoundingClientRect().bottom);} const empty=dr.bottom-bot;
  return empty>Math.max(150,0.3*dr.height)?[`the dialog is empty below its content (${Math.round(empty)} of ${Math.round(dr.height)} px)`]:[];}
function pillsOf(g){const out=[]; for(const f of g.fills||[]){const w=f.x1-f.x0, h=f.y1-f.y0; if(w<8||h<8||w>260||h>44) continue;
  const t=g.texts.find(t=>t.x0>=f.x0-3&&t.x1<=f.x1+3&&t.y0>=f.y0-4&&t.y1<=f.y1+4); if(t) out.push({...f,t:t.t});} return out;}
function boxHits(lines,dots,boxes,what){const out=new Set();
  for(const l of lines){const p=l.pts; if(p.length<2) continue; const open=Math.hypot(p[0][0]-p[p.length-1][0],p[0][1]-p[p.length-1][1])>1; if(!open) continue;
    const acc=[0]; for(let i=1;i<p.length;i++) acc.push(acc[i-1]+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1])); const L=acc[acc.length-1];
    for(const b of boxes){const near=q=>q[0]>b.x0-10&&q[0]<b.x1+10&&q[1]>b.y0-10&&q[1]<b.y1+10; if(near(p[0])||near(p[p.length-1])) continue;
      for(let i=0;i<p.length;i++){if(acc[i]<10||L-acc[i]<10) continue; const x=p[i][0], y=p[i][1]; if(x>b.x0+1.5&&x<b.x1-1.5&&y>b.y0+1.5&&y<b.y1-1.5){out.add(`line over ${what} “${b.t||''}”`); break;}}}}
  for(const d of dots||[]) for(const b of boxes){if(d[0]>b.x0+1.5&&d[0]<b.x1-1.5&&d[1]>b.y0+1.5&&d[1]<b.y1-1.5) out.add(`particle over ${what} “${b.t||''}”`);}
  return [...out];}
function canvasBoxes(g){const bars=(g.fills||[]).filter(f=>f.tag==='bar').map(f=>({...f,t:'bar'}));
  return boxHits(g.lines,[],pillsOf(g),'pill').concat(boxHits(g.lines,[],bars,'probability'));}
function heroEmpty(g,i){const P=g.panels[i]; if(!P) return []; const area=(P.x1-P.x0)*(P.y1-P.y0), boxes=[];
  g.texts.forEach(t=>boxes.push(t)); g.dots.forEach(d=>boxes.push({x0:d[0]-2,y0:d[1]-2,x1:d[0]+2,y1:d[1]+2}));
  (g.fills||[]).forEach(f=>{if((f.x1-f.x0)*(f.y1-f.y0)<0.5*area) boxes.push(f);});
  g.lines.forEach(l=>{for(let k=1;k<l.pts.length;k++){const a=l.pts[k-1], b=l.pts[k]; boxes.push({x0:Math.min(a[0],b[0])-1,y0:Math.min(a[1],b[1])-1,x1:Math.max(a[0],b[0])+1,y1:Math.max(a[1],b[1])+1});}});
  const in4={x0:P.x0+4,y0:P.y0+4,x1:P.x1-4,y1:P.y1-4}; const E=emptiest(in4,boxes.filter(b=>!(b.x0<=in4.x0&&b.x1>=in4.x1&&b.y0<=in4.y0&&b.y1>=in4.y1)),16,8);
  return tooEmpty(E)?[`the open stage is ${Math.round(E.frac*100)}% empty in one block (${Math.round(E.fw*100)}% × ${Math.round(E.fh*100)}%)`]:[];}
window.__lay={uid,vis,texts,overlaps,svgLines,lineHits,dotHits,spill,canvasAudit,wraps,clipped,svgEmpty,dialogEmpty,canvasBoxes,heroEmpty,
  dom(sel,boxSel){const root=document.querySelector(sel); if(!root||!vis(root)) return []; const tb=texts(root), out=[];
    overlaps(tb).forEach(x=>out.push('text over text: '+x)); lineHits(svgLines(root),tb.filter(t=>t.svg)).forEach(x=>out.push(x)); spill(root,boxSel).forEach(x=>out.push(x));
    wraps(root).forEach(x=>out.push(x)); clipped(root).forEach(x=>out.push(x)); svgEmpty(root).forEach(x=>out.push(x)); return out;},
  hero(i,frames){const out=new Set(); const cv=document.querySelector('#heroLab .ar-hcv'); if(!cv) return ['no hero canvas'];
    const dom=texts(document.querySelector('#heroLab .ar-hstage')), dense=Array.from({length:39},(_,k)=>(k+1)/40);
    for(const p of dense){const g=window.__ar.geom('hero',{i,p,t:1000+p*4200}); canvasAudit(g,dom,{panels:1}).concat(canvasBoxes(g)).forEach(x=>out.add(`stage ${i+1}: ${x}`));}
    for(const p of frames){for(let t=0;t<=2000;t+=100){const g2=window.__ar.geom('hero',{i,p,t:t+p*4200}); dotHits(g2.dots,g2.texts.concat(dom)).concat(boxHits([],g2.dots,pillsOf(g2),'pill')).forEach(x=>out.add(`stage ${i+1}: ${x}`));}}
    const gE=window.__ar.geom('hero',{i,p:0.97,t:1000+0.97*4200}); heroEmpty(gE,i).concat(loopClear(gE)).forEach(x=>out.add(`stage ${i+1}: ${x}`));
    return [...out];},
  ov(){const out=new Set(); for(const m of Object.keys(window.__ar.MODELS)) for(let t=0;t<=2000;t+=100){const g=window.__ar.geom('ov',{m,t}); canvasAudit(g,[]).concat(canvasBoxes(g)).concat(loopLane(g)).concat(t===0?laneCont(g):[]).forEach(x=>out.add(`${m}: ${x}`));} return [...out];},
  tile(){const out=new Set(); for(let t=0;t<=2000;t+=100){const g=window.__ar.geom('tile',{t}); if(!g) return ['no tile canvas']; canvasAudit(g,[],{panels:0}).concat(canvasBoxes(g)).forEach(x=>out.add(x));} return [...out];},
  hoverables(sel){const root=document.querySelector(sel); if(!root) return []; return [...root.querySelectorAll('*')].filter(e=>{if(!vis(e)) return false; const cs=getComputedStyle(e);
      return e.hasAttribute('data-tipx')||e.hasAttribute('data-tip')||e.hasAttribute('title')||e.classList.contains('t')||cs.cursor==='pointer'||cs.cursor==='help';})
    .map(e=>({id:uid(e),name:(e.getAttribute('aria-label')||e.innerText||e.textContent||e.tagName).trim().slice(0,30),empty:!(e instanceof SVGElement)&&!(e.innerText||e.textContent||'').trim()&&!e.querySelector('svg text,canvas,img')}));},
  hoverState(id){const e=[...document.querySelectorAll('*')].find(x=>ids.get(x)===id); if(!e) return null; const cs=getComputedStyle(e), tip=document.getElementById('tip');
    return {sty:[cs.boxShadow,cs.outlineStyle,cs.outlineWidth,cs.backgroundColor,cs.borderTopColor].join('|'),tipVis:!!tip&&!tip.hidden&&tip.getBoundingClientRect().width>0,tip:tip?tip.innerText.trim():''};},
  point(id){const e=[...document.querySelectorAll('*')].find(x=>ids.get(x)===id); if(!e) return null; e.scrollIntoView({block:'center',inline:'nearest'}); const r=e.getClientRects()[0]||e.getBoundingClientRect();
    return {x:r.left+Math.min(r.width/2,24),y:r.top+Math.min(r.height/2,12),w:r.width,h:r.height};}
};})();
"""
KILL = "*,*::before,*::after{animation-duration:0s!important;animation-delay:0s!important;transition-duration:0s!important}"


def add(cfg, state, items):
    for x in items:
        hits.append((cfg, state, x))


def page(b, w, h, lang, theme):
    ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme='dark' if theme == 'dark' else 'light')
    if FONT_CSS:
        ctx.route('https://fonts.googleapis.com/**', lambda r: r.fulfill(status=200, content_type='text/css', body=FONT_CSS, headers={'Access-Control-Allow-Origin': '*'}))
        ctx.route('https://fonts.gstatic.com/**', lambda r: r.abort())
    pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(400)
    pg.evaluate("()=>document.fonts&&document.fonts.ready")
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    pg.add_style_tag(content=KILL)
    pg.evaluate(JS)
    return pg, errs


def js(pg, s, wait=120):
    pg.evaluate(s); pg.wait_for_timeout(wait)


def hover_pass(pg, cfg, state, sel):
    for hv in pg.evaluate(f"()=>__lay.hoverables({json.dumps(sel)})"):
        pt = pg.evaluate(f"()=>__lay.point({hv['id']})")
        if not pt or pt['w'] < 1:
            continue
        pg.mouse.move(2, 2); pg.wait_for_timeout(40)
        before = pg.evaluate(f"()=>__lay.hoverState({hv['id']})")
        pg.mouse.move(pt['x'], pt['y']); pg.wait_for_timeout(170)
        after = pg.evaluate(f"()=>__lay.hoverState({hv['id']})")
        if not after:
            continue
        if after['tipVis'] and not after['tip']:
            hits.append((cfg, state, f'hover “{hv["name"]}”: a tooltip with no text'))
        if hv['empty'] and before and after['sty'] != before['sty']:
            hits.append((cfg, state, f'hover “{hv["name"]}”: an empty box appears ({before["sty"]} → {after["sty"]})'))
    pg.mouse.move(2, 2); pg.wait_for_timeout(40)


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in VIEWPORTS:
        for lang in ['en', 'zh']:
            for theme in THEMES:
                cfg = f'{w}x{h} {lang} {theme}'
                pg, errs = page(b, w, h, lang, theme)
                hov = HOVER and theme == 'light' and w in (1440, 390)
                add(cfg, 'page', ['sideways scroll'] if pg.evaluate("()=>document.documentElement.scrollWidth>innerWidth+1") else [])
                # the home hero and tile
                for i in range(6):
                    js(pg, f"()=>window.__ar.heroGo({i})", 520)   # the hero plays on its own since 2026-10-11; no stage to click
                    add(cfg, 'hero', pg.evaluate(f"()=>__lay.hero({i},[0.1,0.3,0.5,0.7,0.85,0.97])"))
                add(cfg, 'hero card', pg.evaluate("()=>__lay.dom('#heroLab','#heroLab')"))
                if hov:
                    hover_pass(pg, cfg, 'hero', '#heroLab')
                js(pg, "()=>{const b=document.querySelector('#domStrip .ds-btn[data-world=\"models\"]')||document.querySelector('#domGrid .dg-btn[data-world=\"models\"]'); if(b) b.click();}", 400)
                js(pg, "()=>{const t=document.querySelector('#tile-architecture'); if(t) t.scrollIntoView({block:'center'})}", 300)
                add(cfg, 'tile', pg.evaluate("()=>__lay.tile()"))
                add(cfg, 'tile card', pg.evaluate("()=>__lay.dom('#tile-architecture','#tile-architecture')"))
                if hov:
                    hover_pass(pg, cfg, 'tile', '#tile-architecture')
                # the lab
                js(pg, "()=>{location.hash='architecture'}", 700)
                pg.add_style_tag(content=KILL)
                add(cfg, 'page (lab)', ['sideways scroll'] if pg.evaluate("()=>document.documentElement.scrollWidth>innerWidth+1") else [])
                add(cfg, 'overview', pg.evaluate("()=>__lay.ov()"))
                js(pg, "()=>window.scrollTo(0,document.querySelector('#ar-stage').getBoundingClientRect().top+scrollY-70)", 200)
                inner = pg.evaluate("()=>{const s=document.querySelector('#ar-stage'); return getComputedStyle(s).position==='sticky'&&s.scrollHeight>s.clientHeight+2?`the stage scrolls inside (${s.scrollHeight} > ${s.clientHeight})`:''}")
                add(cfg, 'stage', [inner] if inner else [])
                add(cfg, 'stage', pg.evaluate("()=>__lay.dom('#ar-stage','#ar-stage')"))
                add(cfg, 'stage head (model switch, fidelity chip)', pg.evaluate("()=>__lay.dom('#ar-stage .stage-head','#ar-stage')"))
                add(cfg, 'settings', ['a Settings panel is back (Batch 1 put the model switch on the stage)'] if pg.query_selector('#ar-controls') else [])
                for k in range(6):
                    js(pg, f"()=>document.querySelector('#ar-rail .steps [data-go=\"{k}\"]').click()", 120)
                    if k == 0:
                        js(pg, "()=>document.querySelector('#ar-rail [data-pred=\"mlp\"]').click()", 120)
                    add(cfg, f'guided part {k + 1}', pg.evaluate("()=>__lay.dom('#ar-rail','#ar-rail')"))
                    if k == 4:   # the causal switch: the card and the stage's own drawing, with the mask on, off and on again
                        for v in ('on', 'off', 'on'):
                            js(pg, f"()=>document.querySelector('#ar-rail [data-key=\"mkmask\"] [data-v=\"{v}\"]').click()", 200)
                            add(cfg, f'guided part 5, mask {v} (card)', pg.evaluate("()=>__lay.dom('#ar-rail','#ar-rail')"))
                            inner = pg.evaluate("()=>{const s=document.querySelector('#ar-stage'); return getComputedStyle(s).position==='sticky'&&s.scrollHeight>s.clientHeight+2?`the stage scrolls inside (${s.scrollHeight} > ${s.clientHeight})`:''}")
                            add(cfg, f'guided part 5, mask {v} (stage)', pg.evaluate("()=>__lay.dom('#ar-stage','#ar-stage')") + ([inner] if inner else []))
                if hov:
                    js(pg, "()=>document.querySelector('#ar-rail .steps [data-go=\"0\"]').click()", 120)
                    hover_pass(pg, cfg, 'lab', '#ar-app .bench')
                js(pg, "()=>document.querySelector('#view-dyn .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()", 200)
                for i in range(4):
                    js(pg, f"()=>document.querySelector('#ar-rail [data-preset=\"{i}\"]').click()", 120)
                    add(cfg, f'scenario {i + 1}', pg.evaluate("()=>__lay.dom('#ar-rail','#ar-rail').concat(__lay.dom('#ar-stage .stage-head','#ar-stage'))"))
                js(pg, "()=>document.querySelector('#ar-rail [data-preset=\"1\"]').click()", 120)
                js(pg, "()=>document.querySelector('#view-dyn .modes button[data-mode=\"challenge\"]').click()", 200)
                add(cfg, 'challenge', pg.evaluate("()=>__lay.dom('#ar-rail','#ar-rail')"))
                for i in range(9):   # one question at a time: every other answer wrong, so both feedback boxes are laid out
                    js(pg, f"()=>{{document.querySelector('#ar-rail [data-chi=\"{i}\"]')&&document.querySelector('#ar-rail [data-chi=\"{i}\"]').click(); const q=window.__ar.CHQ_CORE.concat(window.__ar.CHQ)[{i}]; document.querySelector(`#ar-rail [data-cq=\"{i}\"][data-v=\"${{{i}%2?q.a:(q.a+1)%4}}\"]`).click()}}", 120)
                    add(cfg, f'challenge question {i + 1} answered', pg.evaluate("()=>__lay.dom('#ar-rail','#ar-rail')"))
                js(pg, "()=>document.querySelector('#view-dyn .modes button[data-mode=\"guided\"]').click()", 200)
                # every step of every pop-up
                for k, n in [('emb', 6), ('attn', 7), ('mlp', 4), ('layer', 1), ('out', 3)]:
                    js(pg, f"()=>document.querySelector('#ar-ov [data-pop=\"{k}\"]').click()", 250)
                    for s in range(n):
                        if s:
                            js(pg, "()=>document.querySelector('#ar-stepper [data-step=\"1\"]').click()", 200)
                        variants = [None]
                        if k == 'attn' and s in (2, 3):
                            variants = ['0', '3']
                        if k == 'attn' and s in (4, 5):
                            variants = ['0', '3', 't0', 't1', 't2', 't3', 't4']
                        if k == 'mlp' and s == 1:
                            variants = ['p-5', 'p5', 'p-1.5']
                        if k == 'out' and s == 1:
                            variants = ['T0.2', 'T2', 'T1']
                        if k == 'out' and s == 2:
                            variants = [None, 'S1', 'S20', 'S20']
                        if k == 'attn' and s == 6:
                            variants = ['causal', 'window', 'sink', 'strided', 'bi', '1m', 'm:causal', 'm:window', 'm:sink', 'm:strided', 'm:bi']
                        if k == 'attn' and s == 6:
                            variants = ['causal', 'window', 'sink', 'strided', 'bi', '1m']
                        if k == 'layer':
                            variants = [None, 'nonorm', 'nores']
                        for v in variants:
                            if v in ('0', '3'):
                                if s <= 4:
                                    js(pg, f"()=>document.querySelector('#ar-modal [data-key=\"ahead\"] [data-v=\"{v}\"]').click()", 150)
                                else:
                                    continue
                            elif v and v[0] == 't':
                                js(pg, f"()=>document.querySelector('#ar-modal [data-key=\"atok\"] [data-v=\"{v[1]}\"]').click()", 150)
                            elif v and v[0] in 'pT':
                                js(pg, f"()=>{{const r=document.querySelector('#ar-modal #{'ar-silu-in' if v[0] == 'p' else 'ar-temp-in'}'); r.value={v[1:]}; r.dispatchEvent(new Event('input',{{bubbles:true}}))}}", 150)
                            elif v and v.startswith('m:'):
                                js(pg, f"()=>document.querySelector('#ar-modal [data-key=\"amask\"] [data-v=\"{v[2:]}\"]').click()", 150)
                            elif v in ('S1', 'S20'):
                                js(pg, f"()=>document.querySelector('#ar-modal [data-act=\"{'sample' if v == 'S1' else 'sample20'}\"]').click()", 200)
                            elif v == '1m':
                                js(pg, "()=>document.querySelector('#ar-modal [data-act=\"a1m\"]').click()", 150)
                            elif v in ('nonorm', 'nores'):
                                js(pg, "()=>{const n=document.querySelector('#ar-modal [data-act=\"nonorm\"]'), r=document.querySelector('#ar-modal [data-act=\"nores\"]'); if(n.getAttribute('aria-pressed')==='true') n.click(); if(r.getAttribute('aria-pressed')==='true') r.click();}", 120)
                                js(pg, f"()=>document.querySelector('#ar-modal [data-act=\"{v}\"]').click()", 150)
                            elif v:
                                js(pg, f"()=>document.querySelector('#ar-modal [data-key=\"amask\"] [data-v=\"{v}\"]').click()", 150)
                            add(cfg, f'pop-up {k} step {s + 1}' + (f' ({v})' if v else ''), pg.evaluate("()=>__lay.dom('#ar-modal .ar-dialog','#ar-modal .ar-dialog').concat(__lay.dialogEmpty())"))
                        if hov and (s == 0 or s == n - 1 or (k == 'attn' and s == 3)):
                            hover_pass(pg, cfg, f'pop-up {k} step {s + 1}', '#ar-modal .ar-dialog')
                    if k == 'attn':
                        js(pg, "()=>{const b=document.querySelector('#ar-modal [data-act=\"a1m\"]'); if(b&&b.getAttribute('aria-pressed')==='true') b.click()}", 100)
                        js(pg, "()=>{const b=document.querySelector('#ar-modal [data-key=\"amask\"] [data-v=\"causal\"]'); if(b) b.click()}", 100)
                    pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
                add(cfg, 'errors', [f'page error: {e}' for e in errs])
                pg.context.close()
                print(cfg, 'done', len(hits), flush=True)
    b.close()

PAGE.unlink(missing_ok=True)
groups = defaultdict(list)
for cfg, state, x in hits:
    groups[(state, x)].append(cfg)
for (state, x), cfgs in sorted(groups.items()):
    print(f'- [{state}] {x}  ({len(cfgs)}: {", ".join(cfgs[:4])}{" …" if len(cfgs) > 4 else ""})')
print(f'{len(hits)} hits ({len(groups)} distinct)')
sys.exit(1 if hits else 0)
