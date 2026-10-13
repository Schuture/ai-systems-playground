"""Flow test for the data-pipeline lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal page scroll, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: the prediction gate with the kit's reason under a greyed-out Next; before the pick the run does not start, no count is
  shown and the answer (the count left) appears nowhere on the lab, in text or tooltips (R3.8); the pick starts the run, Fast by
  default (about 10 s, R3.9); Next waits for the end; then the count, the verdict, the choices locked;
- R3.20 and the 2026-10-04 addition: a run watched to the end shows its end at once when it comes back (the next part, Back, a mode
  switch and back, a language switch, a setting moved back to a picture already watched); a new picture replays, from the first stage
  that changed (part 2's threshold: from stage 2); “Replay” replays from stage 1 on demand; Pause pauses; the scrubber jumps;
- parts 2–5: Next waits for the action (threshold moved, a rule off, exact only, another mix); Show me does it; the result appears;
  part 4's MinHash sliders only with MinHash; part 6: three points with “Show in the chart”, three misconceptions;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked; the result after Submit as in train-capstone (owner's review b8, R1): the verdict first, a miss in
  red with its main gap, “Show the cheapest setting”, then your setting next to the cheapest (shared settings once, differing ones
  as chips), the cheapest curve on the chart, only the settings that pass behind a closed disclosure, never every setting; the
  capability card;
- part 4's example pages (owner's review b8): at 1440, 1280, 900, 720 and 390 every page's text stays inside its own box, the boxes
  do not overlap, and no grey (a background or a box-shadow) painted after a word covers it;
- R3: every link in the lab opens a formatted page (github.com/…/blob/…, huggingface.co/…/blob/…, arXiv), never a raw file;
- no hover (outside Show the math) only to give a shown number's exact value: none says “精确值是” or “Exactly” (the owner's
  clarification of 23:37 UTC);
- the writing rules on every visible text node and tooltip (banned words, Chinese spacing and punctuation, two colons, R6.18),
  colours (red only for wrong calls and missed goals, R5.6).
Screenshots: shots/data-pipeline/.
Run: python3 tests/data-pipeline_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time, re
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/data-pipeline.html', '--only', 'data-pipeline'], check=True, cwd=P, stdout=subprocess.DEVNULL)
URL = 'file://' + str(P / 'out' / 'data-pipeline.html')
SH = P / 'shots' / 'data-pipeline'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
fails, n = [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
PUNCT = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][:;,!?]|[:;,!?] ?[\\u4e00-\\u9fff]/g;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(bad); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); const m=t.match(bad); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
 return [...out];}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #dl-mathbody, .num-big, #dl-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|dl-over|dl-dr|dl-gbad|dl-sw-red|dl-sw-dr|dl-mol|over|dl-ttr|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#dl-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#dl-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# the owner's clarification of 23:37 UTC (review b8): no hover only to expand a shown number to its exact value; Show the math may
# write an exact value in its formula text, never as a hover
EXACTHOVER = """()=>{const out=[]; document.querySelectorAll('#dl-app [data-tipx], #dl-app [data-tip], #dl-app [title], #dock [data-tipx]').forEach(e=>{
  if(e.closest('#dl-mathbody, details.math, .mathbody')) return;
  const t=['data-tipx','data-tip','title'].map(a=>e.getAttribute(a)||'').join(' '); if(/精确值是|Exactly/.test(t)) out.push(t.trim().slice(0,90));});
  return out.slice(0,6);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#dl-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""

# review codex-b9v3 应修 4: the on-screen size of every text in the curve's SVG = its font size × the SVG's scale (getScreenCTM), each
# text inside the SVG's box; the axis titles are HTML outside the SVG; the example's text (应修 3) is HTML of at least 11 px
FONTS = """()=>{const svg=document.querySelector('#dl-curve svg'); if(!svg||!svg.getClientRects().length) return null; const R=svg.getBoundingClientRect(), out={svg:[],out:[],html:[]};
 svg.querySelectorAll('text').forEach(t=>{const m=t.getScreenCTM(); out.svg.push(+(parseFloat(getComputedStyle(t).fontSize)*Math.hypot(m.a,m.b)).toFixed(2)); const r=t.getBoundingClientRect(); if(r.left<R.left-2||r.right>R.right+2) out.out.push(t.textContent);});
 document.querySelectorAll('#dl-c-yt, #dl-c-xt, #dl-xbox .dl-xt, #dl-xbox .dl-xc, #dl-xbox .dl-xm, #dl-xbox .dl-xv').forEach(e=>{if(e.getClientRects().length) out.html.push(parseFloat(getComputedStyle(e).fontSize));});
 out.titlesInSvg=[...svg.querySelectorAll('text')].some(t=>/similarity of|chance to be|相似度|被抓住的机会/.test(t.textContent)); out.titles=[document.querySelector('#dl-c-yt').textContent,document.querySelector('#dl-c-xt').textContent]; return out;}"""
EXAMPLE = """()=>{const x=document.querySelector('#dl-xbox'), c=document.querySelector('#dl-cbox'); if(!x||x.hidden||!x.getClientRects().length) return null; const pg=[...x.querySelectorAll('.dl-xpg')];
 return {above:x.getBoundingClientRect().bottom<=c.getBoundingClientRect().top+1, n:pg.length, shB:pg[1].querySelectorAll('.dl-xw.on').length, wB:pg[1].querySelectorAll('.dl-xw').length, shC:pg[2].querySelectorAll('.dl-xw.on').length,
   hashes:pg.map(p=>p.querySelector('.dl-xhash').textContent), badB:pg[1].classList.contains('bad'), redB:!!pg[1].querySelector('.dl-xv.dl-xbad'), hit:[...x.querySelectorAll('.dl-xg.hit')].length,
   sameB:pg.length&&x.querySelectorAll('.dl-xrow:nth-child(2) .dl-xc.same').length, sameC:x.querySelectorAll('.dl-xrow:nth-child(3) .dl-xc.same').length,
   fits:[...x.querySelectorAll('.dl-xg,.dl-xpg')].every(e=>e.getBoundingClientRect().right<=x.getBoundingClientRect().right+0.5)}}"""

# owner's review b8: the example pages' text inside its own box, boxes apart, no grey painted over a word (the 2 px box-shadow of the
# next word used to cover the last 2 px of each Chinese word)
XTEXT = r"""()=>{const x=document.querySelector('#dl-xbox'); if(!x||x.hidden||!x.getClientRects().length) return null;
 const out={n:0,outside:[],covered:[],cardsOverlap:[],scroll:[]};
 const pgs=[...x.querySelectorAll('.dl-xpg')]; out.n=pgs.length;
 const inter=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>0.5&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>0.5;
 pgs.forEach(p=>{const R=p.getBoundingClientRect(), cs=getComputedStyle(p), L=R.left+parseFloat(cs.borderLeftWidth), Rr=R.right-parseFloat(cs.borderRightWidth);
   if(p.scrollWidth>p.clientWidth+1) out.scroll.push(p.dataset.p);
   const tw=document.createTreeWalker(p,NodeFilter.SHOW_TEXT); let n; const gl=[];
   while((n=tw.nextNode())){if(!n.textContent.trim()||!n.parentElement.getClientRects().length) continue; const rg=document.createRange(); rg.selectNodeContents(n);
     for(const q of rg.getClientRects()){if(q.width<0.5) continue; const h=q.bottom-q.top; gl.push({n,q:{left:q.left,right:q.right,top:q.top+0.15*h,bottom:q.bottom-0.15*h}});
       if(q.left<L-0.5||q.right>Rr+0.5) out.outside.push(p.dataset.p+': '+n.textContent.slice(0,14)+' ['+q.left.toFixed(1)+','+q.right.toFixed(1)+'] vs ['+L.toFixed(1)+','+Rr.toFixed(1)+']');}}
   /* a background or box-shadow painted after a glyph (later in the tree, not its own ancestor) must not cover it */
   p.querySelectorAll('*').forEach(d=>{const s=getComputedStyle(d), bg=s.backgroundColor, has=(bg&&bg!=='rgba(0, 0, 0, 0)'&&bg!=='transparent')||s.boxShadow!=='none'; if(!has||d.classList.contains('dl-xpg')) return;
     const m=s.boxShadow==='none'?null:/(-?[\d.]+)px\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+(-?[\d.]+)px/.exec(s.boxShadow), sp=m?Math.max(0,+m[4]):0;
     const rs=[...d.getClientRects()].map(q=>({left:q.left-sp,right:q.right+sp,top:q.top-sp,bottom:q.bottom+sp}));
     gl.forEach(({n,q})=>{if(d.contains(n)) return; if(!(d.compareDocumentPosition(n)&Node.DOCUMENT_POSITION_PRECEDING)) return; if(rs.some(r=>inter(r,q))) out.covered.push(p.dataset.p+': “'+n.textContent+'” under .'+(d.className||d.tagName));});});});
 for(let i=0;i<pgs.length;i++) for(let j=i+1;j<pgs.length;j++){const a=pgs[i].getBoundingClientRect(), b=pgs[j].getBoundingClientRect(); if(inter(a,b)) out.cardsOverlap.push(pgs[i].dataset.p+pgs[j].dataset.p);}
 out.covered=out.covered.slice(0,6); out.outside=out.outside.slice(0,6); return out;}"""
LINKS = """()=>[...document.querySelectorAll('#dl-app a[href]')].map(a=>a.href).filter(h=>/^https?:/.test(h))"""
LINK_OK = re.compile(r'^https://(github\.com/[^/]+/[^/]+/blob/[^/]+/.+|huggingface\.co/[^/]+/[^/]+/blob/[^/]+/.+|arxiv\.org/abs/[\d.]+)$')
links_seen = set()
CHRES = """()=>{const r=document.querySelector('#dl-ch-res'), d=r.querySelector('details.dl-all');
 return {bad:!!r.querySelector('.badline'), ok:!!r.querySelector('.okline'), gap:!!r.querySelector('.dl-gap'), reveal:!!r.querySelector('[data-act="reveal"]'), sum:!!r.querySelector('.dl-sum'),
  rows:[...r.querySelectorAll('.dl-c3r')].map(e=>e.dataset.role), chips:r.querySelectorAll('.dl-chip').length, chipsOn:r.querySelectorAll('.dl-chip.on').length, redCells:r.querySelectorAll('.dl-c3c.dl-over').length,
  same:(r.querySelector('.dl-same')||{}).textContent||'', next:(r.querySelector('.dl-next')||{}).textContent||'', all:!!d, allOpen:!!(d&&d.open), allRows:d?d.querySelectorAll('tbody tr').length:0,
  allSum:d?d.querySelector('summary').textContent:'', cmp:r.querySelectorAll('.cmp-row, .cmp-chart').length, bestCurve:!!document.querySelector('#dl-curve .dl-bestc')}}"""


def page(b, w, h, lang, hash_='data-pipeline'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    pg.set_default_timeout(10000)
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    # wait for the hook, the language and the mounted lab, not for a fixed time: under load a fresh page was read before the lab
    # existed (window.__dl undefined) once; the sleeps below only let the first frame settle
    pg.goto(URL); pg.wait_for_function('()=>!!window.__dl&&!!window.ASP', timeout=30000); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]')
    pg.wait_for_function(f"()=>!!document.querySelector('.langsw button[data-lang=\"{lang}\"][aria-pressed=\"true\"]')", timeout=30000); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}")
    pg.wait_for_function("()=>{const a=document.querySelector('#view-dyn #dl-app'); return !!a&&!!a.querySelector('#dl-rail')&&!!window.__dl.state()}", timeout=30000); pg.wait_for_timeout(600)
    return pg, errs


def st(pg):
    return pg.evaluate('()=>window.__dl.state()')


def click(pg, sel):
    return pg.evaluate(f"""()=>{{const e=[...document.querySelectorAll('{sel}')].find(x=>x.offsetParent!==null)||document.querySelector('{sel}'); if(!e) return 0; e.click(); return 1;}}""")


def nxt(pg):
    click(pg, '#dl-rail [data-nav="next"]'); pg.wait_for_timeout(350)


def mode(pg, m):
    pg.evaluate(f"()=>{{const b=[...document.querySelectorAll('.modes button[data-mode=\"{m}\"]')].find(e=>e.offsetParent!==null); b.click();}}"); pg.wait_for_timeout(400)


def set_range(pg, key, v, where='#dl-rail'):
    pg.evaluate(f"()=>{{const i=[...document.querySelectorAll('{where} input[data-key={key}]')].find(x=>x.offsetParent!==null); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"); pg.wait_for_timeout(250)


# H1 (the owner, 2026-10-09: “全站帮我找到这些旧的，没有latex渲染的公式，并修好”): every formula is typeset with A.ui.math; no TeX command
# shows as a word inside the MathML, and no plain-text formula (x_i, x^y, √, <sup>/<sub>) is left in the lab's visible text or Show the math
FORMULA_H1 = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return {n:0,bad:[],txt:[]};
  const CMD=/^(frac|left|right|text|sqrt|times|cdot|hat|approx|leq|geq|quad|phantom|mathbf|mathrm|operatorname|big|Big|alpha|beta|gamma|delta|eta|theta|lambda|mu|pi|rho|sigma|tau|phi|omega|epsilon|infty|ldots|cdots|leftarrow|rightarrow|div|neq)$/;
  const bad=[]; let n=0;
  root.querySelectorAll('math').forEach(m=>{n++; m.querySelectorAll('mi').forEach(x=>{const t=x.textContent.trim(); if(CMD.test(t)) bad.push(t);}); if(m.textContent.includes('\\')) bad.push(m.textContent.slice(0,40));});
  const PAT=/√|\b[A-Za-z]{1,2}_\{?[A-Za-z0-9]{1,3}\b|\b[A-Za-z]\^[A-Za-z0-9({α-ω]/;
  const txt=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let nd;
  while((nd=tw.nextNode())){const e=nd.parentElement; if(!e||e.closest('math,pre,code,svg,script,style')) continue; if(PAT.test(nd.nodeValue)) txt.push(nd.nodeValue.trim().slice(0,80));}
  root.querySelectorAll('sup,sub').forEach(e=>{if(!e.closest('math,pre,code')) txt.push('<'+e.tagName.toLowerCase()+'>'+e.textContent);});
  return {n,bad:bad.slice(0,5),txt:txt.slice(0,5)};}"""


def formulas_h1(pg, where):
    fm = pg.evaluate(FORMULA_H1, '#dl-app')
    check(f'{where}: formulas typeset as MathML, no TeX word shown, no plain-text formula left (H1, R6.29)', not fm['bad'] and not fm['txt'], fm)


def rules(pg, tag, lang):
    formulas_h1(pg, tag)
    for name, js in (('rules', RULES), ('spacing', SPACING), ('punct', PUNCT), ('colons', COLONS), ('bignum', BIGNUM)):
        if name in ('spacing', 'punct', 'colons') and lang != 'zh': continue
        hits = pg.evaluate("s=>(" + js + ")(s)", '#dl-app')
        check(f'{tag}: writing rules ({name})', not hits, hits[:3])
    check(f'{tag}: no horizontal page scroll', pg.evaluate('()=>document.documentElement.scrollWidth<=innerWidth+1'), pg.evaluate('()=>[document.documentElement.scrollWidth,innerWidth]'))
    over = pg.evaluate(OVER); check(f'{tag}: no text wider than its box', not over, over)
    hp = pg.evaluate(HELP); check(f'{tag}: every help cursor has a tooltip', not hp, hp)
    ex = pg.evaluate(EXACTHOVER); check(f'{tag}: no hover only to give a number’s exact value (the owner, 23:37 UTC)', not ex, ex)
    ln = pg.evaluate(LINKS); links_seen.update(ln); bad = [x for x in ln if not LINK_OK.match(x)]
    check(f'{tag}: every link opens a formatted page, never a raw file (R3)', not bad, bad[:3])


def shot(pg, name):
    pg.evaluate("()=>{const s=document.querySelector('#dl-stage'); if(s) s.scrollIntoView({block:'start'});}"); pg.wait_for_timeout(200)
    pg.screenshot(path=str(SH / f'{name}.png'))


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            tag = f'{lang} {w}'
            pg, errs = page(b, w, h, lang)
            Z = lambda en, zh: zh if lang == 'zh' else en
            kept = pg.evaluate('()=>window.__dl.fates({}).kept')
            # ---------- part 1: the guess first (R3.8)
            s = st(pg)
            check(f'{tag} part 1: the run waits for the guess', s['g'] == 0 and not s['playing'], s)
            check(f'{tag} part 1: Next greyed out with the kit’s reason', pg.evaluate("()=>{const b=document.querySelector('#dl-rail [data-nav=\"next\"]'); return !!b&&b.disabled}"))
            lab_txt = pg.evaluate("()=>{const a=document.querySelector('#dl-app'); let t=a.innerText; a.querySelectorAll('[data-tipx]').forEach(e=>t+=' '+e.getAttribute('data-tipx')); return t}")
            check(f'{tag} part 1: the answer appears nowhere before the guess', not re.search(rf'(?<![\d,.]){kept}(?![\d,.%])', lab_txt), re.findall(rf'.{{20}}{kept}.{{20}}', lab_txt)[:2])
            check(f'{tag} part 1: the player is hidden before the guess', pg.evaluate("()=>document.querySelector('#dl-player').hidden"))
            shot(pg, f'{lang}-{w}-1-mission')
            t0 = time.time()
            click(pg, '#dl-rail [data-pred="tenth"]'); pg.wait_for_timeout(300)
            s = st(pg)
            check(f'{tag} part 1: the guess starts the run, at Fast', s['playing'] and 0 < s['g'] < 4 and s['spd'] == 'fast', s)
            check(f'{tag} part 1: Next waits for the end of the run', pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            pg.wait_for_function('()=>window.__dl.state().g>=4', timeout=20000)
            dt = time.time() - t0
            check(f'{tag} part 1: Fast takes about 10 s (R3.9)', 8 <= dt <= 14, dt)
            pg.wait_for_timeout(300)
            s = st(pg)
            check(f'{tag} part 1: the run watched to the end is remembered', not s['playing'] and s['sig'] in s['watched'], s['playing'])
            obs = pg.evaluate("()=>document.querySelector('#dl-rail [data-obs]').innerText")
            check(f'{tag} part 1: the count and the verdict', str(kept) in obs and (Z('was right', '是对的') in obs), obs[:160])
            check(f'{tag} part 1: the choices lock once the run has ended', pg.evaluate("()=>[...document.querySelectorAll('#dl-rail [data-pred]')].every(b=>b.disabled)"))
            check(f'{tag} part 1: the button now says Replay', pg.evaluate("()=>document.querySelector('#dl-player [data-act=\"play\"]').textContent") == Z('↺ Replay', '↺ 重放'))
            rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}-{w}-1-done')
            # ---------- R3.20: the next part shows the same picture: no replay
            nxt(pg); s = st(pg)
            check(f'{tag} R3.20: part 2 shows the watched run at its end, no replay', s['step'] == 1 and s['g'] >= 4 and not s['playing'], s)
            # a new picture replays from the first stage that changed
            set_range(pg, 'thr', 90); pg.wait_for_timeout(120); s = st(pg)
            check(f'{tag} R3.20: a new threshold replays, from stage 2', s['playing'] and 1 <= s['g'] < 4, s['g'])
            # pause, then the scrubber to the end
            click(pg, '#dl-player [data-act="play"]'); pg.wait_for_timeout(150); s1 = st(pg); pg.wait_for_timeout(400); s2 = st(pg)
            check(f'{tag} R3.20: Pause pauses', not s1['playing'] and abs(s1['g'] - s2['g']) < 1e-9, (s1['g'], s2['g']))
            set_range(pg, 'scrub', 400, '#dl-player'); s = st(pg)
            check(f'{tag} R3.20: the scrubber to the end finishes the run', s['g'] >= 4 and s['sig'] in s['watched'], s['g'])
            # back to a picture already watched: its end at once
            set_range(pg, 'thr', 65); s = st(pg)
            check(f'{tag} R3.20 (2026-10-04): back to 0.65, already watched: its end at once', s['g'] >= 4 and not s['playing'], s['g'])
            set_range(pg, 'thr', 90); s = st(pg)
            check(f'{tag} R3.20 (2026-10-04): 0.90 again, already watched: no replay', s['g'] >= 4 and not s['playing'], s['g'])
            # Replay on demand
            click(pg, '#dl-player [data-act="play"]'); pg.wait_for_timeout(150); s = st(pg)
            check(f'{tag} R3.20: “Replay” replays from stage 1', s['playing'] and s['g'] < 1, s['g'])
            set_range(pg, 'scrub', 400, '#dl-player')
            obs = pg.evaluate("()=>document.querySelector('#dl-rail [data-obs]').innerText")
            check(f'{tag} part 2: the result once the threshold moved', bool(obs.strip()) and not pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"), obs[:100])
            rules(pg, f'{tag} part 2', lang); shot(pg, f'{lang}-{w}-2-lang')
            # Back to part 1: the watched run, at once
            click(pg, '#dl-rail [data-nav="back"]'); pg.wait_for_timeout(300); s = st(pg)
            check(f'{tag} R3.20: Back to part 1 shows its end at once', s['step'] == 0 and s['g'] >= 4 and not s['playing'], s)
            # a mode switch and back
            mode(pg, 'challenge'); mode(pg, 'guided'); s = st(pg)
            check(f'{tag} R3.20: the challenge and back: no replay', s['g'] >= 4 and not s['playing'], s)
            # a language switch
            other = 'zh' if lang == 'en' else 'en'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); s = st(pg)
            check(f'{tag} R3.20: a language switch: no replay', s['g'] >= 4 and not s['playing'], s)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
            nxt(pg)
            # ---------- part 3
            nxt(pg); s = st(pg)
            check(f'{tag} part 3: entered at its end (same picture), Next waits for a rule off', s['step'] == 2 and s['g'] >= 4 and pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"), s)
            check(f'{tag} part 3: five switches, each with its count', pg.evaluate("()=>[...document.querySelectorAll('#dl-rail .dl-sw em')].filter(e=>/^\\d+$/.test(e.textContent)).length") == 5)
            click(pg, '#dl-rail .dl-sw[data-rule="stop"]'); pg.wait_for_timeout(200); s = st(pg)
            check(f'{tag} part 3: a rule off replays from stage 3', s['playing'] and 2 <= s['g'] < 4, s['g'])
            set_range(pg, 'scrub', 400, '#dl-player')
            check(f'{tag} part 3: the result and Next', bool(pg.evaluate("()=>document.querySelector('#dl-rail [data-obs]').innerText").strip()) and not pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            rules(pg, f'{tag} part 3', lang); shot(pg, f'{lang}-{w}-3-rules')
            # ---------- part 4
            nxt(pg)
            check(f'{tag} part 4: Next waits for “Only exact copies”', pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            # review codex-b9v3 应修 3: three example pages above the curve; B shares most 5-grams, C none; the hashes differ; MinHash boxes one group of B
            ex = pg.evaluate(EXAMPLE)
            check(f'{tag} part 4: the example (A, B, C) comes before the curve; B mostly shaded, C not; three different hashes; one group of B boxed', bool(ex) and ex['above'] and ex['n'] == 3 and ex['shB'] >= 0.8 * ex['wB'] and ex['shC'] == 0
                  and len(set(ex['hashes'])) == 3 and ex['hit'] == 1 and ex['sameB'] == 10 and ex['sameC'] == 0 and not ex['badB'] and ex['fits'], ex)
            # scroll first and let the page settle: a scroll hides the tooltip, so hovering while the page still scrolls shows nothing
            xj = pg.locator('#dl-xbox .t[data-ctl="xjB"]'); xj.scroll_into_view_if_needed(); pg.wait_for_timeout(250); xj.hover(); pg.wait_for_timeout(300)
            tip = pg.evaluate("()=>{const t=document.querySelector('#tip'); return t&&!t.hidden?{math:!!t.querySelector('math'),txt:t.innerText,fits:t.querySelector('math').getBoundingClientRect().right<=t.getBoundingClientRect().right}:null}")
            check(f'{tag} part 4: the similarity’s hover gives the idea, then the formula on its own line, inside the tip (28 of 34)', bool(tip) and tip['math'] and tip['fits'] and '28' in tip['txt'] and '34' in tip['txt'], tip)
            pg.mouse.move(0, 0); pg.wait_for_timeout(150)
            fz = pg.evaluate(FONTS)
            check(f'{tag} part 4: the chart’s text is at least 10.5 px on screen, inside the chart; axis titles in HTML; the example’s text at least 11 px (应修 4)', bool(fz) and min(fz['svg']) >= 10.5 and not fz['out'] and not fz['titlesInSvg'] and all(fz['titles']) and min(fz['html']) >= 11, fz)
            check(f'{tag} part 4: MinHash’s sliders shown with MinHash', pg.evaluate("()=>!document.querySelector('#dl-ic-br').hidden"))
            # owner's review b8 (part 4, page B): at 1440, 1280, 900, 720 and 390, every page's text inside its own box, nothing under a grey
            for ww in ((1440, 1280, 900, 720, 390) if w == 1440 else (w,)):
                if ww != w: pg.set_viewport_size({'width': ww, 'height': h}); pg.wait_for_timeout(350)
                xt = pg.evaluate(XTEXT)
                check(f'{tag} part 4 at {ww}: each example page’s text stays inside its box, the boxes do not overlap, no grey covers a word', bool(xt) and xt['n'] == 3 and not xt['outside'] and not xt['covered'] and not xt['cardsOverlap'] and not xt['scroll'], xt)
            if pg.viewport_size['width'] != w: pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(400)
            click(pg, '#dl-rail [data-lever="exact"]'); pg.wait_for_timeout(300)
            ex = pg.evaluate(EXAMPLE)
            check(f'{tag} part 4: exact only: B stays, its card and verdict red, no group boxed (the choice changes the example)', bool(ex) and ex['badB'] and ex['redB'] and ex['hit'] == 0 and ex['sameB'] == 0, ex)
            xt = pg.evaluate(XTEXT)
            check(f'{tag} part 4, exact only: the example pages’ text still inside the boxes, no grey covers a word', bool(xt) and not xt['outside'] and not xt['covered'] and not xt['cardsOverlap'], xt)
            shot(pg, f'{lang}-{w}-4-dedup-exact')
            check(f'{tag} part 4: exact only hides the sliders, shows red dots, opens Next', pg.evaluate("()=>document.querySelector('#dl-ic-br').hidden") and pg.evaluate("()=>document.querySelectorAll('#dl-curve .dl-dr').length") > 20 and not pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            click(pg, '#dl-rail [data-lever="both"]'); pg.wait_for_timeout(300)
            set_range(pg, 'r', 4); red4 = pg.evaluate("()=>document.querySelectorAll('#dl-curve .dl-dr').length")
            check(f'{tag} part 4: a loose MinHash (14 × 4) throws away different articles (red under the curve)', red4 > 0, red4)
            set_range(pg, 'r', 8)
            rules(pg, f'{tag} part 4', lang); shot(pg, f'{lang}-{w}-4-dedup')
            # ---------- part 5
            nxt(pg)
            check(f'{tag} part 5: Next waits for another mix', pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            click(pg, '#dl-rail [data-lever="math40"]'); pg.wait_for_timeout(300)
            check(f'{tag} part 5: math 40% in red, Next open', pg.evaluate("()=>document.querySelectorAll('#dl-mix em.over').length") >= 1 and not pg.evaluate("()=>document.querySelector('#dl-rail [data-nav=\"next\"]').disabled"))
            rules(pg, f'{tag} part 5', lang); shot(pg, f'{lang}-{w}-5-mix')
            # ---------- part 6
            nxt(pg)
            check(f'{tag} part 6: three points with “Show in the chart”, three misconceptions', pg.evaluate("()=>document.querySelectorAll('#dl-rail .dl-remlist li').length") == 3 and pg.evaluate("()=>document.querySelectorAll('#dl-rail [data-rem]').length") == 3
                  and pg.evaluate("()=>[...document.querySelectorAll('#dl-rail .obs p')].filter(p=>p.textContent.startsWith('✗')).length") == 3)
            click(pg, '#dl-rail [data-rem="dup"]'); pg.wait_for_timeout(300)
            check(f'{tag} part 6: “Show in the chart” shows the copies left', pg.evaluate("()=>document.querySelectorAll('#dl-curve .dl-dr').length") > 20)
            click(pg, '#dl-rail [data-rem="run"]'); pg.wait_for_timeout(300); s = st(pg)
            check(f'{tag} part 6: part 1’s run comes back at its end (R3.20)', s['g'] >= 4 and not s['playing'], s)
            rules(pg, f'{tag} part 6', lang); shot(pg, f'{lang}-{w}-6-remember')
            # ---------- Scenarios
            mode(pg, 'sandbox')
            pr = pg.evaluate("()=>[...document.querySelectorAll('#dl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
            lead = pg.inner_text('#dl-rail .railcard > p.small')
            check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z('Pick a scenario and see how much of the data each stage throws away.', '选一个场景，看看每个环节扔掉了多少数据。'), lead)
            check(f'{tag} scenarios: four, the first pressed on entry', pr == ['true', 'false', 'false', 'false'], pr)
            for i in range(4):
                click(pg, f'#dl-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
                set_range(pg, 'scrub', 400, '#dl-player')
                pr = pg.evaluate("()=>[...document.querySelectorAll('#dl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
                check(f'{tag} scenario {i + 1}: single choice, insight shown', pr.count('true') == 1 and pr[i] == 'true' and bool(pg.evaluate("()=>document.querySelector('#dl-insight').innerText").strip()), pr)
                rules(pg, f'{tag} scenario {i + 1}', lang); shot(pg, f'{lang}-{w}-sb{i + 1}')
            hsh = pg.evaluate('()=>location.hash')
            pg2, e2 = page(b, w, h, lang, hsh.lstrip('#'))
            s2 = st(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup from the URL', s2['mode'] == 'sandbox' and not s2['mh'] and s2['corpus'] == 'web', (hsh, s2['mode'], s2['mh']))
            pg2.close()
            # ---------- the challenge
            mode(pg, 'challenge')
            check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.querySelector('#dl-controls').classList.contains('locked')"))
            # owner's review b8, R1 (as in train-capstone): the verdict first; a miss in red with its main gap and, on demand, your setting next to
            # the cheapest; only the settings that pass, behind a closed disclosure; never every setting
            def ch_set(b_, r_):
                click(pg, f'#dl-rail .seg-ctl[data-key="chb"] button[data-v="{b_}"]'); click(pg, f'#dl-rail .seg-ctl[data-key="chr"] button[data-v="{r_}"]'); pg.wait_for_timeout(200)
                click(pg, '#dl-rail [data-act="submit"]'); pg.wait_for_timeout(250)
                return pg.evaluate(CHRES)
            allsum = Z('See all settings that pass (3)', '看全部可行的设置（3 种）')
            c = ch_set(14, 8)
            check(f'{tag} challenge 14 × 8: the verdict in red, its main gap and “Show the cheapest setting”; no comparison yet, no list', c['bad'] and not c['ok'] and c['gap'] and c['reveal'] and not c['sum'] and not c['all'] and c['cmp'] == 0 and not c['bestCurve'], c)
            click(pg, '#dl-rail [data-act="reveal"]'); pg.wait_for_timeout(250); c = pg.evaluate(CHRES)
            check(f'{tag} challenge 14 × 8 shown: yours next to the cheapest, both settings differ (two chips a row, yours highlighted), your two misses in red, the one change, the cheapest curve on the chart', c['rows'] == ['me', 'best'] and c['chips'] == 4 and c['chipsOn'] == 2 and c['redCells'] == 2 and not c['same'] and c['next'] and c['bestCurve'], c)
            check(f'{tag} challenge 14 × 8 shown: only the 3 settings that pass, closed by default, never every setting', c['all'] and not c['allOpen'] and c['allRows'] == 3 and c['allSum'].strip() == allsum and c['cmp'] == 0, c)
            c = ch_set(40, 11)
            check(f'{tag} challenge 40 × 11: goal met, the cheaper setting next to it, the shared setting written once, one chip a row', c['ok'] and not c['bad'] and c['rows'] == ['me', 'best'] and c['chips'] == 2 and c['chipsOn'] == 1 and c['redCells'] == 0 and Z('Both use 11 hashes per group.', '两种设置每组都是 11 个哈希。') in c['same'] and c['bestCurve'], c)
            c = ch_set(34, 11)
            check(f'{tag} challenge 34 × 11: goal met with the fewest hashes, a single row, no other curve; the list still closed', c['ok'] and c['rows'] == ['mebest'] and c['redCells'] == 0 and not c['bestCurve'] and c['all'] and not c['allOpen'] and c['allRows'] == 3, c)
            check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#dl-rail [data-exits]')"))
            fz = pg.evaluate(FONTS)
            check(f'{tag} challenge: the chart’s text is at least 10.5 px on screen and inside the chart (应修 4)', bool(fz) and min(fz['svg']) >= 10.5 and not fz['out'] and not fz['titlesInSvg'], fz)
            rules(pg, f'{tag} challenge', lang); shot(pg, f'{lang}-{w}-ch')
            cols = pg.evaluate(COLORS); check(f'{tag}: colours', not cols, cols)
            check(f'{tag}: no page errors', not errs, errs[:2])
            pg.close()
    b.close()
check('R3: the lab’s links were seen and all open formatted pages', len(links_seen) >= 10 and all(LINK_OK.match(x) for x in links_seen), sorted(links_seen)[:4])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'data-pipeline.js').read_text() for d in ('labs', 'teasers') if (P / d / 'data-pipeline.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["go above 1. Pick a mix and see how many epochs each kind of data needs.", "就超过 1。选一种配比，看看每类数据要用几遍。", "Turn off any of the rules, several at once if you like, and see where the pages they caught go.", "关掉你想关的规则，可以同时关几条，看看它们原来抓住的网页去了哪里。", "选一个场景，看看每个环节扔掉了多少数据。", "Shows where the 1,000 pages go at the initial setup.", "在图里显示初始设置下，1,000 个网页各去了哪里。", "Expect most of a crawl to go.", "So to get the tokens a run needs, start from several times as much raw web text.", "Remove near copies, not only exact ones.", "A repost often changes only a line, and its hash is then completely different. Comparing hashes keeps such reposts; MinHash finds them.", "A small domain can be raised, a little.", "Set its share too high and the same text is read many times over, and each extra pass teaches the model less.", "✗ “More data is always better, so keep everything.” Most crawled pages are menus, other languages, junk or copies.", "✗ “Dedup means deleting identical files.” Most copies differ by a line: comparing hashes misses them, and only MinHash finds them.", "✗ “The stricter the filter, the better.” Set the language threshold too high and good English pages go out along with the foreign ones.", "抓来的网页，大部分都留不下。", "所以训练要多少 token，就得准备好几倍的原始网页。", "去重要去几乎一样的，不只是完全一样的。", "转载常常只改一行字，哈希就完全变了。只比哈希，这些转载都会留下，要靠 MinHash 才找得出来。", "数据少的类别可以多放，但别放太多。", "比例定得太高，同一批文字就要读很多遍，而每多读一遍，模型学到的就更少。", "✗ “数据越多越好，不用筛。”抓来的网页，大部分是菜单、别的语言、垃圾或副本。", "✗ “去重就是删掉完全一样的文件。”大多数副本只差一行字，只比哈希会漏掉，只有 MinHash 找得出来。", "✗ “过滤越严越好。”语言的门槛定得太高，好的英文网页也会跟着别的语言一起被扔掉。", "The tokens on hand for the mix (12T, 4T, 1.5T, 0.3T) are assumed.", "配比用到的各类数据手里有多少 token（12T、4T、1.5T、0.3T），是假设的。", "The three example pages of the dedup part (A, B and C) are made up", "讲去重时的三个示例网页（A、B、C）是编的。", "With math at 15% of a 15T-token run and only 0.3T on hand, the same math text is used", "训练 15T 个 token、数学占 15%，而手里只有 0.3T 时，同一批数学文字要用", "The exact-copy half of dedup, as code.", "把一棵目录树里内容完全相同的文件分组：先比哈希，再逐字节确认。也就是去重里“去完全一样的副本”那一半，写成代码。", "MinHash with groups: cut each page", "带分组的 MinHash：把每篇网页切成 5-gram"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["就超过 1。一次选一种配比：", "The switches stack. Turn one off and see where the pages it caught go.", "这些开关可以叠加。关掉一条，看它原来抓住的网页去了哪里。", "Shows part 1’s run at the initial setup.", "在图里显示第 1 部分的初始设置。", "In part 4, comparing hashes alone left", "In part 5, math at 40% meant reading the same math text", "✗ “More data is always better, so keep everything.” In part 1 most pages were menus, other languages, junk or copies.", "✗ “Dedup means deleting identical files.” In part 4 most copies differed by a line, and only MinHash found them.", "✗ “The stricter the filter, the better.” In part 2 a threshold of 0.9 threw away good English pages along with the foreign ones.", "第 1 部分里，1,000 个网页只剩", "第 4 部分里，只比哈希时，留下了", "第 5 部分里，数学占 40% 时，同一批数学文字要读", "✗ “数据越多越好，不用筛。”第 1 部分里，大部分网页是菜单、别的语言、垃圾或副本。", "✗ “去重就是删掉完全一样的文件。”第 4 部分里，大多数副本只差一行字，只有 MinHash 找得出来。", "✗ “过滤越严越好。”第 2 部分里，门槛定到 0.9，好的英文网页也跟着别的语言一起被扔掉了。", "Tokens on hand in part 5 (12T, 4T, 1.5T, 0.3T) are assumed.", "第 5 部分手里各类数据的 token 数（12T、4T、1.5T、0.3T）是假设。", "Part 4’s three example pages are made up", "第 4 部分的三个示例网页是编的。", "In part 5, with math at 15% of the run and only 0.3T on hand, the same math text is used", "第 5 部分里，数学占 15%、手里只有 0.3T 时，同一批数学文字要用", "The exact-copy half of part 4, as code.", "把一棵目录树里内容完全相同的文件分组：先比哈希，再逐字节确认。就是第 4 部分“去完全一样的副本”那一半，写成代码。", "MinHash with groups, as in part 4: cut each page", "第 4 部分的 MinHash 分组：把每篇网页切成 5-gram"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:40] if fails else 'none')
sys.exit(1 if fails else 0)
