"""Flow test for the rag-failure lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor, the writing rules;
- part 1: the stairs run plays on first open at 中 (R3.9); on a phone it waits for the ▶ over the record, which brings the whole record
  into view (round-1 polish P2) with the highest of the stage head and the run's controls that fits with it (N9 of the round-1
  check); the transport row keeps its place and widths through the replay (P4); the chunks under the budget line fade while the sent
  ones fly (N10); the record fits the screen at the end; the retrieval stage is red at the end (R5.11); Next enabled;
- part 2: the log starts with question and answer only (no checks but “correct” known), and that replay ends in about 5 s at 中 (N8);
  Next waits for “every stage” (the kit's why line); the observation names the warranty passage's rank; the new run played once;
  coming back replays nothing (R3.20);
- part 3: three runs one at a time; no red and no verdict before a pick (R3.8); a wrong pick shows the right kind; each new run
  plays once; going back to a watched run replays nothing; 重放 replays; the progress scrubber pauses; Next waits for all three; the
  outdated-page verdict says what went wrong at retrieval and its ✓ says why retrieval recall passes, and the dock at 390 names the
  kind once (N1);
- part 4: Next waits for a prediction; the breakdown and cards are hidden before it (R3.8); the stage shows the two numbers the
  question is about (70% and 95%);
- part 5: the verdict first, in one sentence with its arithmetic, and the part's point second (P12); the chart before the cards
  (R7.12); Next waits for another check;
- part 6: Next waits for a fix; the observation and the card show the engine's numbers; badges only after the learner's click;
- part 7: the misconceptions, the scenarios button, the next-lab footer;
- Scenarios: the breakdown and the cards at the top of the stage, the run record closed under them until the reader opens it (R7.12,
  round-1 polish P18); a new scenario brings the breakdown back into view; five scenarios, each with its insight; the hash round trip
  (a run not yet watched plays once the record is opened); the language switch keeps the setup; at 390 (P18 of the round-1 check):
  the breakdown and the cards on the screen as Scenarios opens, the scenario buttons under them (a phone's only), each tap keeping the
  breakdown, the cards and the button on the screen; the opened record and its note not under the dock (N7); the dock's small bar
  and its readout of what the fixes add; Settings opening on the same picture, with the fixes as buttons under the cards;
- at 390 on a fresh page (N6): a run waiting for ▶ plays when the reader taps the dock or picks “every stage”, where it can be seen;
  a pick made before ▶ shows the whole record with its verdict;
- the challenge: no result before Submit; Submit needs two fixes; a miss names what is missed, a hint and “Show the answer”; the best
  pair meets the goal; the brief chart with the pick; near the goal (70%–73%) no value label is crossed by the goal line (P21); the
  capability card.
Screenshots: shots/rag-failure/.
Run: python3 tests/rag-failure_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'rag-failure').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rag-failure.html', '--only', 'rag-failure'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'rag-failure' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rag-failure.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'rag-failure' / 'flow-wrapped.html')
SH = P / 'shots' / 'rag-failure'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
pct = lambda v: f'{round(v * 100)}%'


def page(b, w, h, lang, hash_='rag-failure'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__rgf.state()")
anim = lambda pg: pg.evaluate("()=>window.__rgf.animState()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rgf-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#rgf-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#rgf-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click: None means nothing replays
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__rgf.animState();}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__rgf.animState()", timeout=timeout); pg.wait_for_timeout(150)


def start_if_waiting(pg, tag, where):
    """On a phone a run not yet watched waits, still, for the ▶ over the record (round-1 polish P2); ▶ plays it."""
    a = anim(pg)
    if not (a and a.get('pending')): return False
    check(f'{tag} {where}: the new run waits at its start for the ▶ over the record', a['p'] == 0 and a['paused'] and vis(pg, '#rgf-go'), a)
    pg.evaluate("()=>document.querySelector('#rgf-go').click()"); pg.wait_for_timeout(150)
    a = anim(pg)
    check(f'{tag} {where}: ▶ plays the run', a is not None and not a['paused'] and not a.get('pending') and not vis(pg, '#rgf-go'), a)
    return True


# the record (question to verdict) is inside the window below the top bar
FITS = """()=>{const f=document.querySelector('#rgf-flow'), bar=document.querySelector('.topbar'); if(!f||!f.getClientRects().length) return null;
  const r=f.getBoundingClientRect(), tb=bar?bar.getBoundingClientRect().bottom:0; return {top:Math.round(r.top),bottom:Math.round(r.bottom),tb:Math.round(tb),h:innerHeight,ok:r.top>=tb-1&&r.bottom<=innerHeight+1};}"""


# phones: the record whole under the top bar, with the highest of the stage head and the run's controls that fits with it (N9)
REVEAL = """()=>{const q=s=>document.querySelector(s), R=e=>e&&e.getClientRects().length?e.getBoundingClientRect():null, bar=q('.topbar'), tb=bar?bar.getBoundingClientRect().bottom:0;
  const fl=R(q('#rgf-flow')); if(!fl) return {ok:false};
  const head=R(q('#rgf-stage .rgf-head')), ctl=R(q('#rgf-pan-run .rgf-ctl')), room=innerHeight-(tb+8)-4, want=[head,ctl].find(r=>r&&fl.bottom-r.top<=room);
  const d=q('#dock'), dr=d&&!d.hidden&&d.getBoundingClientRect().height>0?d.getBoundingClientRect():null, top=dr?Math.max(tb,dr.bottom):tb;
  return {tb:Math.round(tb),dock:dr&&[Math.round(dr.top),Math.round(dr.bottom)],flow:[Math.round(fl.top),Math.round(fl.bottom)],head:head&&[Math.round(head.top),Math.round(head.bottom)],ctl:ctl&&[Math.round(ctl.top),Math.round(ctl.bottom)],
    ok:fl.top>=top-1&&fl.bottom<=innerHeight+1&&(!want||want.top>=top-1)};}"""
# Scenarios on a phone, the record opened: the record, and its note when both fit, below the top bar and the dock (N7)
N7 = """()=>{const q=s=>document.querySelector(s), R=e=>e&&!e.hidden&&e.getClientRects().length?e.getBoundingClientRect():null, bar=q('.topbar'), tb=bar?bar.getBoundingClientRect().bottom:0;
  const fl=R(q('#rgf-flow')), nt=R(q('#rgf-rnote')), d=q('#dock'), dr=d&&!d.hidden&&d.getBoundingClientRect().height>0?d.getBoundingClientRect():null; if(!fl) return {ok:false};
  const top=dr?Math.max(tb,dr.bottom):tb, noteFits=!!nt&&fl.bottom-nt.top<=innerHeight-(tb+8)-4;
  return {tb:Math.round(tb),dock:dr&&[Math.round(dr.top),Math.round(dr.bottom)],flow:[Math.round(fl.top),Math.round(fl.bottom)],note:nt&&[Math.round(nt.top),Math.round(nt.bottom)],noteFits,
    ok:fl.top>=top-1&&fl.bottom<=innerHeight+1&&(!noteFits||nt.top>=top-1)};}"""
# while a run plays: the transport row's places and widths (P4), and the chunks under the budget line while the sent ones fly (N10)
SAMPLE = """async ()=>{const q=s=>document.querySelector(s);
  const pos=()=>['#rgf-transport [data-act="play"]','#rgf-transport .seg-ctl[data-key="speed"]','#rgf-transport [data-act="replay"]','#rgf-tr-p-in','#rgf-rsel'].map(s=>{const e=q(s); if(!e||!e.getClientRects().length) return '-'; const r=e.getBoundingClientRect(); return Math.round(r.left)+'/'+Math.round(r.width);}).join(' ');
  const seen=new Set(); let fly=false, minOp=1, n=0;
  while(window.__rgf.animState()&&n<3000){seen.add(pos()); const f=q('#rgf-flow');
    if(f&&f.classList.contains('rgf-flying')){fly=true; f.querySelectorAll('.rgf-li.below').forEach(e=>{minOp=Math.min(minOp,+getComputedStyle(e).opacity);});}
    n++; await new Promise(r=>requestAnimationFrame(r));}
  return {pos:[...seen],fly,minOp,n};}"""
# Scenarios and Settings on a phone (P18): the breakdown, the cards, the strip of buttons under them and its two groups
SB = """(sel)=>{const q=s=>document.querySelector(s), R=s=>{const e=q(s); if(!e||!e.getClientRects().length) return null; const r=e.getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom)]}, bar=q('.topbar');
  return {tb:bar?Math.round(bar.getBoundingClientRect().bottom):0,flow:q('#rgf-app .bench').dataset.flow,bars:R('#rgf-pan-bars'),met:R('#rgf-metrics'),pk:R('#rgf-pk'),sc:R('#rgf-pk .rgf-pk-sc'),fxg:R('#rgf-pk .rgf-pk-fx'),
    btn:sel?R(sel):null,pressed:sel&&q(sel)?q(sel).getAttribute('aria-pressed'):null};}"""
# the challenge chart: a value label the goal line runs through, or one that leaves its track (P21)
HIT = """()=>[...document.querySelectorAll('#rgf-ch-res .cmp-track')].map(tr=>{const g=tr.querySelector('b.g'), v=tr.querySelector('.cmp-val'); if(!g||!v) return null;
  const gr=g.getBoundingClientRect(), r=v.getBoundingClientRect(), t=tr.getBoundingClientRect(), x=gr.left+gr.width/2;
  return {txt:v.textContent, hit:r.left-1<=x&&x<=r.right+1, inTrack:r.left>=t.left-0.5&&r.right<=t.right+0.5};}).filter(Boolean)"""
DOCK = "()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText.replace(/\\u00a0/g,' '):''}"


def on_screen(g, k, h):
    return bool(g[k]) and g[k][0] >= g['tb'] - 1 and g[k][1] <= h + 1


def tap_kept(pg, sel):
    """a reader's tap on a button under the cards, scrolled onto the screen first as a reader would: the button stays where it was"""
    h = pg.viewport_size['height']; r = pg.evaluate(SB, sel)
    if r['btn'] and r['btn'][1] > h - 4: pg.evaluate(f"()=>window.scrollBy(0,{r['btn'][1] - h + 12})"); settle(pg)
    g0 = pg.evaluate(SB, sel); pg.click(sel); pg.wait_for_timeout(800); settle(pg); g1 = pg.evaluate(SB, sel)
    return g0, g1, all(on_screen(g1, k, h) for k in ('bars', 'met', 'btn')) and abs(g1['btn'][0] - g0['btn'][0]) <= 2


def dock_below(pg):
    """scroll the stage just past the top of the window, into the card under it on a phone, and read the dock"""
    pg.evaluate("()=>{const a=document.querySelector('#rgf-stage'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY-20)}")
    try: pg.wait_for_function("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0}", timeout=3000)
    except Exception: pass
    pg.wait_for_timeout(400)
    return pg.evaluate(DOCK)


def mode(pg, m):
    click(pg, f'#rgf-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b(?![?？])|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/吃|喂|扛|吞/,'figurative verb'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里|\\bin part [0-9]\\b|part [0-9]’s/,'recall of a part or a one-at-a-time note'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part)/i,'colour named by lightness (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]|[）】][ ][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip,.rgf-tx,.rgf-sn,option,select')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline') return;
   const t=e.textContent; if(!t) return; const prev=e.previousSibling, next=e.nextSibling;
   const pc=prev&&prev.nodeType===3?prev.textContent.slice(-1):'', nc=next&&next.nodeType===3?next.textContent.slice(0,1):'';
   const pair=(a,b)=>a&&b&&bad.test(a+b);
   if(pair(pc,t[0])) out.add(pc+t[0]+' before: '+t.slice(0,40)); if(pair(t[t.length-1],nc)) out.add(t[t.length-1]+nc+' after: '+t.slice(-40));});
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>][\\u4e00-\\u9fff]/g); if(m) out.add(m.join('|')+' in tooltip: '+t.slice(0,80));});
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody,.rgf-tx,.rgf-sn')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; const c=e.cloneNode(true); c.querySelectorAll('code').forEach(x=>x.remove()); texts.push(c.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# red only for a miss (an answer cut apart, the right chunk not kept, a limit not met, the failed verdict); blue, amber and violet are reserved
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/\\bbad\\b|rgf-bad|rgf-sw-bad|rgf-verdict|rgf-miss|rgf-pill|badline|status bad|\\bover\\b|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#rgf-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#rgf-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.rgf-math-line')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(cs.textOverflow==='ellipsis'&&e.closest('.rgf-sn,option,select')) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#rgf-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#rgf-stage svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  const ts=[...s.querySelectorAll('text')].filter(t=>t.getClientRects().length&&t.textContent.trim());
  for(const t of ts){const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push((s.id||'svg')+' outside: '+t.textContent.trim().slice(0,30));}}
  return out.slice(0,8);}"""
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','sum','cdot','quad','mathrm','times']);
  const out=[]; for(const m of document.querySelectorAll('#rgf-app math, #dock math')) for(const e of m.querySelectorAll('mi,mo')){const t=e.textContent.trim(); if(W.has(t)) out.push(t);}
  const pat=[/\\\\frac|\\\\sum|\\\\text|k_1 \\(|1\\/\\(60/];
  const tw=document.createTreeWalker(document.querySelector('#rgf-app'),NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; for(const r of pat) if(r.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  return out;}"""
ALIGN = """()=>{const els=[...document.querySelectorAll('#rgf-transport .btn, #rgf-transport .seg-ctl')].filter(e=>e.getClientRects().length);
  const rows=[]; for(const e of els){const r=e.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<8); if(!row){row={top:r.top,items:[]}; rows.push(row);}
    const b=e.matches('.seg-ctl')?e.querySelector('button'):e, rg=document.createRange(); rg.selectNodeContents(b); const t=rg.getBoundingClientRect();
    row.items.push({t:(e.textContent||'').trim().slice(0,12),top:r.top,h:r.height,base:t.bottom});}
  const bad=[]; for(const row of rows){const it=row.items; for(const x of it){if(Math.abs(x.h-it[0].h)>0.6||Math.abs(x.top-it[0].top)>0.6||Math.abs(x.base-it[0].base)>1) bad.push(x.t+' vs '+it[0].t+' h '+x.h+'/'+it[0].h+' base '+x.base.toFixed(1)+'/'+it[0].base.toFixed(1));}}
  return {n:els.length,bad};}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#rgf-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(TEXWORDS): rule_hits.append(f'{where}: a formula not typeset or a TeX word shown: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    for h in pg.evaluate(SVGIN): rule_hits.append(f'{where}: chart label outside its picture: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


NEXT = '#rgf-rail .step.current [data-nav="next"]'
BACK = '#rgf-rail .step.current [data-nav="back"]'
SEG = lambda k, v: f'#rgf-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'
RED = "()=>[...document.querySelectorAll('#rgf-flow .bad')].filter(e=>e.getClientRects().length).length"


def go_next(pg):
    sel = NEXT if pg.query_selector(NEXT) else '#rgf-rail [data-nav="next"]'
    click(pg, sel, 300)


def timed_run(pg, sel, timeout=20000):
    t0 = pg.evaluate('()=>performance.now()'); a = pg.evaluate(CLICK_ANIM, sel)
    pg.wait_for_function("()=>!window.__rgf.animState()", timeout=timeout)
    return a, (pg.evaluate('()=>performance.now()') - t0) / 1000


src = (P / 'labs' / 'rag-failure.js').read_text()
check('source: no “one at a time” note and no sentence that recalls a part by number', not re.search(r'一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里', src))

with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg); a0 = anim(pg)
            D = pg.evaluate("()=>window.__rgf.data()"); base = pg.evaluate("()=>window.__rgf.compute('base',[])")
            check(f'{tag}: opens on part 1 of the guided tour, the stairs run, no fix', st['mode'] == 'guided' and st['step'] == 0 and st['run'] == 1 and st['fx'] == [], st)
            # ---- part 1
            check(f'{tag} part 1: the run is set to play on first open, at 中 (R3.9)', a0 is not None and a0['key'] == 'stairs' and speed(pg) == 'mid', (a0, speed(pg)))
            check(f'{tag} part 1: nothing red while the run plays (R3.8)', pg.evaluate(RED) == 0)
            waited = start_if_waiting(pg, tag, 'part 1')
            check(f'{tag} part 1: on a phone the run waits for ▶; on a wide screen it plays at once', waited == phone, (waited, a0))
            if phone:
                pg.wait_for_timeout(700); settle(pg); rv = pg.evaluate(REVEAL)
                check(f'{tag} part 1: ▶ brings the record whole onto the screen, with the highest of the stage head and the run’s controls that fits with it (N9)', rv['ok'], rv)
            tr = pg.evaluate(SAMPLE)
            check(f'{tag} part 1: the transport row keeps its places and widths through the replay (P4)', len(tr['pos']) == 1 and tr['n'] > 30, tr)
            check(f'{tag} part 1: the chunks under the budget line fade while the sent ones fly (N10)', tr['fly'] and tr['minOp'] <= 0.35, tr)
            wait_anim(pg)
            fit = pg.evaluate(FITS)
            check(f'{tag} part 1: the whole record, question to verdict, is on the screen at the end (B4)', fit and fit['ok'], fit)
            cur = text(pg, '#rgf-rail .step.current')
            check(f'{tag} part 1: the card names {base["ok"]} correct of 200', str(base['ok']) in cur and '70%' in cur, cur[:300])
            check(f'{tag} part 1: retrieval is the red stage (R5.11)', pg.evaluate("()=>!!document.querySelector('#rgf-flow .rgf-stg.bad[data-s=\"ret\"]')") and pg.evaluate(RED) >= 2)
            check(f'{tag} part 1: Next enabled', not any(nexts(pg)))
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 1: Play, the speed buttons and Replay share one height and one baseline', al['n'] == 3 and not al['bad'], al)
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2
            t0 = pg.evaluate('()=>performance.now()'); a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 2: the E-104 run plays once (a new run)', a is not None and a['key'] == 'e104' and st['step'] == 1 and st['log'] == 0, (a, st))
            waited = start_if_waiting(pg, tag, 'part 2'); wait_anim(pg)
            secs = (pg.evaluate('()=>performance.now()') - t0) / 1000
            if not waited:
                check(f'{tag} part 2: with only the question and the answer logged, the replay ends in about 5 s at 中, not 10 (N8)', 4.0 <= secs <= 7.0, secs)
            check(f'{tag} part 2: question and answer only: retrieval not logged, three checks unknown', text(pg, '#rgf-flow').count(Z(lang, 'Not logged', '没有记录')) == 2 and pg.evaluate("()=>[...document.querySelectorAll('#rgf-flow .rgf-pill b')].filter(b=>b.textContent==='?').length") == 3)
            check(f'{tag} part 2: Next waits, with the kit’s why line', all(nexts(pg)) and any(Z(lang, 'Log every stage', '每一环都记') in x for x in why(pg)), (nexts(pg), why(pg)))
            a = click_anim(pg, SEG('log', '1'))
            check(f'{tag} part 2: logging every stage replays nothing (R3.20)', a is None, a)
            obs = text(pg, '#rgf-rail [data-obs]')
            check(f'{tag} part 2: the observation names the warranty rank 9', '9' in obs and not any(nexts(pg)), obs[:200])
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            a = click_anim(pg, BACK)
            check(f'{tag} part 2 → 1: coming back replays nothing (R3.20)', a is None and state(pg)['step'] == 0, a)
            click(pg, NEXT, 300)
            # ---- part 3: triage
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 3: the first triage run plays', a is not None and a['key'] == 'autolock' and st['step'] == 2, (a, st))
            start_if_waiting(pg, tag, 'part 3'); wait_anim(pg)
            check(f'{tag} part 3: no red, no verdict and no checks before a pick (R3.8)', pg.evaluate(RED) == 0 and not text(pg, '#rgf-rail .rgf-verd') and pg.evaluate("()=>[...document.querySelectorAll('#rgf-flow .rgf-pill b')].every(b=>b.textContent==='?')"))
            check(f'{tag} part 3: Next waits (why line)', all(nexts(pg)) and why(pg))
            # a wrong pick on run 1
            click(pg, '#rgf-rail [data-tri="unsup"]')
            v = text(pg, '#rgf-rail .rgf-verd')
            check(f'{tag} part 3: a wrong pick names the right kind', Z(lang, 'Not quite', '不对') in v and pg.evaluate("()=>window.__rgf.kindName('cut')") in v and pg.evaluate(RED) >= 2, v[:200])
            # replay and scrubber on a watched run
            a = click_anim(pg, '#rgf-transport [data-act="replay"]')
            check(f'{tag} part 3: 重放 replays from the start', a is not None and a['p'] < 0.2, a)
            pg.evaluate("()=>{const i=document.querySelector('#rgf-tr-p-in'); i.value=40; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
            sa = anim(pg)
            check(f'{tag} part 3: the scrubber pauses at the point picked', sa is not None and sa['paused'] and abs(sa['p'] - 0.4) < 0.02, sa)
            click(pg, '#rgf-transport [data-act="play"]'); wait_anim(pg)
            for i, (key, kind) in enumerate([('thermo', 'stale'), ('filter', 'unsup')], start=1):
                a = click_anim(pg, f'#rgf-rail [data-trigo="{i}"]')
                check(f'{tag} part 3: run {i + 1} plays once, at once (the reader asked for it)', a is not None and a['key'] == key and not a.get('pending'), a)
                wait_anim(pg)
                click(pg, f'#rgf-rail [data-tri="{kind}"]')
                v = text(pg, '#rgf-rail .rgf-verd')
                check(f'{tag} part 3: the right pick on run {i + 1}', Z(lang, 'Right', '对') in v and pg.evaluate(RED) >= 1, v[:120])
                if kind == 'stale':
                    # N1 of the round-1 check: the verdict names retrieval while the retrieval check on the same line shows ✓
                    rec = pg.evaluate("()=>{const p=document.querySelector('#rgf-flow .rgf-pill[data-pk=\"0\"]'); return {verd:document.querySelector('#rgf-flow .rgf-verdict').innerText, m:p?p.dataset.m:'', tip:p?p.getAttribute('data-tipx'):''}}")
                    check(f'{tag} part 3: the outdated-page verdict says what went wrong at retrieval, and its ✓ says why retrieval recall passes (N1)',
                          Z(lang, 'it returned the replaced edition too', '取回了已被替换的旧版') in rec['verd'] and rec['m'] == '✓' and Z(lang, 'cannot tell that the replaced edition came back too', '查不出旧版也被取回来了') in rec['tip'], rec)
                    if phone:
                        y = pg.evaluate('()=>scrollY'); dk = dock_below(pg); kn = pg.evaluate("()=>window.__rgf.kindName('stale')")
                        check(f'{tag} part 3: the dock names the kind once and says what went wrong at retrieval (N1)', Z(lang, 'it returned the replaced edition too', '取回了已被替换的旧版') in dk and dk.count(kn) == 1, dk)
                        pg.evaluate(f'()=>window.scrollTo(0,{y})'); pg.wait_for_timeout(200)
            a = click_anim(pg, '#rgf-rail [data-trigo="0"]')
            check(f'{tag} part 3: back to a watched run replays nothing (R3.20)', a is None, a)
            check(f'{tag} part 3: Next enabled after three picks', not any(nexts(pg)))
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4: predict
            a = click_anim(pg, NEXT); wait_anim(pg)
            check(f'{tag} part 4: the breakdown and the cards stay hidden before the prediction (R3.8)', not vis(pg, '#rgf-pan-bars') and not vis(pg, '#rgf-metrics'))
            two = text(pg, '#rgf-pan-sum')
            check(f'{tag} part 4: the stage shows the two numbers of the question, 70% correct and 95% retrieval recall', vis(pg, '#rgf-pan-sum') and '70%' in two and '95%' in two and not vis(pg, '#rgf-pan-run'), two[:200])
            check(f'{tag} part 4: Next waits for a prediction', all(nexts(pg)) and why(pg))
            cur = text(pg, '#rgf-rail .step.current')
            check(f'{tag} part 4: the question gives 190, 95% and 70%', '190' in cur and '95%' in cur and '70%' in cur, cur[:300])
            click(pg, '#rgf-rail [data-pred="ret"]')
            check(f'{tag} part 4: a saved prediction enables Next', not any(nexts(pg)))
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            # ---- part 5
            go_next(pg)
            cur = text(pg, '#rgf-rail .step.current')
            check(f'{tag} part 5: the verdict comes first and gives the answer', cur.find(Z(lang, 'Not quite', '不对')) >= 0 and cur.find(Z(lang, 'Not quite', '不对')) < 80 and '50' in cur, cur[:200])
            sents = [x for x in re.split(r'(?<=[。！？])|(?<=[.!?])\s+', cur[cur.find(Z(lang, 'Not quite', '不对')):]) if x.strip()]
            check(f'{tag} part 5: after a wrong prediction the verdict and its arithmetic are one sentence, and the part’s point is the second (P12)',
                  len(sents) > 1 and '10' in sents[0] and '50' in sents[0] and Z(lang, 'Answer correctness alone', '只看答案正确率') in sents[1], sents[:2])
            order = pg.evaluate("()=>{const a=document.querySelector('#rgf-pan-bars').getBoundingClientRect().top, b=document.querySelector('#rgf-metrics').getBoundingClientRect().top; return a<b}")
            check(f'{tag} part 5: the chart before the metric cards (R7.12)', order)
            check(f'{tag} part 5: no badge on entering (R6.6)', not text(pg, '#rgf-metrics .d.on'))
            check(f'{tag} part 5: Next waits for another check', all(nexts(pg)))
            click(pg, SEG('met', 'ret'))
            obs = text(pg, '#rgf-rail [data-obs]')
            check(f'{tag} part 5: retrieval recall catches 10', '95%' in obs and '10' in obs and not any(nexts(pg)), obs[:200])
            check(f'{tag} part 5: the rows retrieval recall misses are dimmed', pg.evaluate("()=>document.querySelectorAll('#rgf-bars .rgf-tr.dim').length") == 4)
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            # ---- part 6
            go_next(pg)
            check(f'{tag} part 6: Next waits for a fix', all(nexts(pg)))
            click(pg, '#rgf-rail [data-lever="ver"]')
            m = pg.evaluate("()=>window.__rgf.compute('base',['ver'])")
            obs = text(pg, '#rgf-rail [data-obs]')
            check(f'{tag} part 6: the version filter: {m["ok"] - 140} more correct, {round(m["cor"] * 100)}%', str(m['ok'] - 140) in obs and f'{round(m["cor"] * 100)}%' in obs and not any(nexts(pg)), obs[:200])
            d = text(pg, '#rgf-metrics [data-k="cor"] .d')
            check(f'{tag} part 6: a badge after the learner’s click, in points', Z(lang, 'pts', '个百分点') in d, d)
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7
            go_next(pg)
            cur = text(pg, '#rgf-rail .step.current')
            check(f'{tag} part 7: three misconceptions and the scenarios button', cur.count('✗') == 3 and pg.query_selector('#rgf-rail [data-mode-go="sandbox"]') is not None)
            check(f'{tag} part 7: the next-lab footer', pg.evaluate("()=>[...document.querySelectorAll('#rgf-app .next-lab, .next-lab-end')].some(e=>e.getBoundingClientRect().height>0)"))
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- scenarios
            click(pg, '#rgf-rail [data-mode-go="sandbox"]', 400)
            st = state(pg)
            check(f'{tag} scenarios: open at the initial setup', st['mode'] == 'sandbox' and st['fx'] == [], st)
            if phone:
                # P18 of the round-1 check: what a change moves is on the screen as Scenarios opens, with the scenario buttons under it
                g = pg.evaluate(SB, None)
                check(f'{tag} scenarios: on a phone the breakdown and the cards are on the screen as Scenarios opens, the scenario buttons under them (P18)',
                      on_screen(g, 'bars', h) and on_screen(g, 'met', h) and g['sc'] is not None and g['fxg'] is None and g['met'][1] <= g['sc'][0], g)
                for i in range(5):
                    sel = f'#rgf-pk [data-pick="{i}"]'; g0, g1, kept = tap_kept(pg, sel)
                    rail = pg.evaluate(f"()=>document.querySelector('#rgf-rail [data-preset=\"{i}\"]').getAttribute('aria-pressed')")
                    check(f'{tag} scenarios: a tap on scenario {i + 1} under the cards sets it and keeps the breakdown, the cards and the button on the screen (P18)', kept and g1['pressed'] == 'true' and rail == 'true', (g0, g1, rail))
                rules(pg, f'{tag} scenarios strip', lang)
            else:
                check(f'{tag} scenarios: the buttons under the cards are a phone’s only', not vis(pg, '#rgf-pk'))
            for i in range(5):
                click(pg, f'#rgf-rail [data-preset="{i}"]')
                ins = text(pg, '#rgf-insight')
                check(f'{tag} scenarios: scenario {i + 1} has its insight and is pressed', '%' in ins and len(ins) > 60 and pg.evaluate(f"()=>document.querySelector('#rgf-rail [data-preset=\"{i}\"]').getAttribute('aria-pressed')") == 'true', ins[:160])
            click(pg, '#rgf-rail [data-preset="2"]')
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the fixes', hsh.startswith('#rag-failure~f00110_r'), hsh)
            top = pg.evaluate("()=>['#rgf-pan-bars','#rgf-metrics','#rgf-runtog'].map(s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?Math.round(e.getBoundingClientRect().top):null})")
            check(f'{tag} scenarios: the breakdown and the cards at the top of the stage, the run record under them (R7.12, P18)', None not in top and top[0] < top[1] < top[2], top)
            check(f'{tag} scenarios: the run record waits closed until the reader opens it', not vis(pg, '#rgf-flow') and pg.evaluate("()=>document.querySelector('#rgf-runtog').getAttribute('aria-expanded')") == 'false' and anim(pg) is None, anim(pg))
            a = click_anim(pg, '#rgf-runtog')
            check(f'{tag} scenarios: opening the record shows the run, the chart and the cards stay', vis(pg, '#rgf-flow') and vis(pg, '#rgf-pan-bars') and vis(pg, '#rgf-metrics') and pg.evaluate("()=>document.querySelector('#rgf-runtog').getAttribute('aria-expanded')") == 'true')
            if phone:
                pg.wait_for_timeout(500); settle(pg); n7 = pg.evaluate(N7)
                check(f'{tag} scenarios: the opened record, and its note when both fit on the screen, are below the top bar and the dock (N7)', n7['ok'], n7)
            wait_anim(pg)
            pg.evaluate("()=>{const s=document.querySelector('#rgf-rsel'); s.value='3'; s.dispatchEvent(new Event('change',{bubbles:true}));}"); pg.wait_for_timeout(300); wait_anim(pg)
            note = text(pg, '#rgf-rnote') if vis(pg, '#rgf-rnote') else ''
            check(f'{tag} scenarios: with fixes on, the outdated-page run says the record is the initial setup and the filter repairs 90% of its kind', '90%' in note and Z(lang, 'initial setup', '初始设置') in note, note)
            shot(pg, f'{tag}_scen', full=phone); rules(pg, f'{tag} scenarios', lang)
            if not phone:
                click(pg, '#rgf-rail [data-preset="4"]', 700)
                bt = pg.evaluate("()=>{const s=document.querySelector('#rgf-stage').getBoundingClientRect(), b=document.querySelector('#rgf-pan-bars').getBoundingClientRect(); return [Math.round(s.top),Math.round(b.top),Math.round(b.bottom),innerHeight]}")
                check(f'{tag} scenarios: a new scenario brings the breakdown back to the top of the stage (P18)', bt[1] >= bt[0] - 1 and bt[2] <= bt[3], bt)
                click(pg, '#rgf-rail [data-preset="2"]', 400)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag} scenarios: the language switch keeps the setup', all(s0[k] == s1[k] for k in ('mode', 'fx', 'run')), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            if phone:
                dk = dock_below(pg)
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk) and '%' in dk, dk)
                # P18 of the round-1 check: the dock's small bar is the breakdown's top bar, and its readout what the fixes add and the check
                # that moves the most, from the engine
                fx = state(pg)['fx']; m = pg.evaluate("f=>window.__rgf.compute('base',f)", fx); hu = lambda v: int(v * 100 + 0.5 + 1e-9)
                mv = sorted((abs(hu(m[k]) - hu(base[k])), k) for k in ('ret', 'ctx', 'grd') if hu(m[k]) != hu(base[k]))[-1][1]
                want = [str(m['ok'] - base['ok']), f'{hu(base[mv])}% → {hu(m[mv])}%']
                bar_w = pg.evaluate("()=>[...document.querySelectorAll('#dock .rgf-dk i')].map(e=>parseFloat(e.style.width))")
                check(f'{tag}: with fixes on, the dock says what they add ({want[0]} more correct) and the check that moves the most ({want[1]}), over a bar of correct and repaired answers (P18)',
                      all(x in dk for x in want) and len(bar_w) == 2 and abs(bar_w[0] - base['ok'] / 2) < 0.1 and abs(bar_w[1] - (m['ok'] - base['ok']) / 2) < 0.1, (dk, want, bar_w))
                shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
                # Settings on a phone opens on the same picture, the fixes as buttons under the cards (P18)
                click(pg, '#rgf-app .modes button[data-flow="free"]', 400)
                g = pg.evaluate(SB, None)
                check(f'{tag} settings: on a phone Settings opens on the breakdown and the cards, the fixes as buttons under them (P18)',
                      state(pg)['mode'] == 'sandbox' and g['flow'] == 'free' and on_screen(g, 'bars', h) and on_screen(g, 'met', h) and g['fxg'] is not None and g['sc'] is None, g)
                f0 = state(pg)['fx']; g0, g1, kept = tap_kept(pg, '#rgf-pk [data-pfx="ver"]'); f1 = state(pg)['fx']
                seg = pg.evaluate("()=>{const b=document.querySelector('#rgf-controls .seg-ctl[data-key=\"fx-ver\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")
                check(f'{tag} settings: a tap on a fix under the cards switches it, as the switch in the settings does, and keeps the breakdown, the cards and the button on the screen (P18)',
                      kept and set(f0) ^ set(f1) == {'ver'} and seg == ('1' if 'ver' in f1 else '0') and g1['pressed'] == ('true' if 'ver' in f1 else 'false'), (f0, f1, seg, g0, g1))
                shot(pg, f'{tag}_settings'); rules(pg, f'{tag} settings', lang)
            pg.close()
            pg = page(b, w, h, lang, hash_='rag-failure~f10001_r5'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and st['fx'] == ['hyb', 'dec'] and st['run'] == 5, st)
            check(f'{tag}: the record waits closed under the chart, nothing plays on arrival (P18)', anim(pg) is None and not vis(pg, '#rgf-flow') and vis(pg, '#rgf-pan-bars'), anim(pg))
            a = click_anim(pg, '#rgf-runtog')
            check(f'{tag}: opening the record plays a run not yet watched, at once', a is not None and a['key'] == 'e104' and not a.get('pending'), a)
            # ---- challenge
            mode(pg, 'challenge')
            res = text(pg, '#rgf-ch-res')
            check(f'{tag} challenge: no result before Submit, Submit disabled with no fix', not res and pg.evaluate("()=>document.querySelector('#rgf-rail [data-act=\"submit\"]').disabled"))
            check(f'{tag} challenge: correctness hidden before Submit', Z(lang, 'after you submit', '提交后') in text(pg, '#rgf-ch-c'))
            click(pg, '#rgf-rail [data-chfx="ver"]'); click(pg, '#rgf-rail [data-chfx="dec"]')
            check(f'{tag} challenge: a third fix is disabled once two are picked', pg.evaluate("()=>document.querySelector('#rgf-rail [data-chfx=\"hyb\"]').disabled"))
            click(pg, '#rgf-rail [data-act="submit"]')
            res = text(pg, '#rgf-ch-res')
            check(f'{tag} challenge: version + split misses the wait, with a hint and “Show the answer”', Z(lang, 'Not yet', '还没达标') in res and '600' in res and Z(lang, 'Hint', '提示') in res and pg.query_selector('#rgf-rail [data-act="reveal"]') is not None, res[:200])
            click(pg, '#rgf-rail [data-act="reveal"]'); res = text(pg, '#rgf-ch-res')
            rows = pg.evaluate("()=>document.querySelectorAll('#rgf-ch-res .cmp-row').length")
            check(f'{tag} challenge: Show the answer charts the 3 passing pairs, the failed pick and the initial setup, under a title that fits', rows == 5 and Z(lang, 'your pick', '你的选择') in res and Z(lang, 'the initial setup, your pick', '初始设置、你的选择') in res, (rows, res[:200]))
            shot(pg, f'{tag}_ch_miss', full=phone); rules(pg, f'{tag} challenge miss', lang)
            click(pg, '#rgf-rail [data-chfx="dec"]')
            check(f'{tag} challenge: a new choice needs a new Submit', not text(pg, '#rgf-ch-res'))
            click(pg, '#rgf-rail [data-chfx="ctx"]'); click(pg, '#rgf-rail [data-act="submit"]')
            res = text(pg, '#rgf-ch-res')
            check(f'{tag} challenge: version + budget meets the goal, a better pair exists', Z(lang, 'Goal met', '达标') in res and '79%' in res, res[:200])
            click(pg, '#rgf-rail [data-chfx="ctx"]'); click(pg, '#rgf-rail [data-chfx="hyb"]'); click(pg, '#rgf-rail [data-act="submit"]')
            res = text(pg, '#rgf-ch-res')
            check(f'{tag} challenge: hybrid + version is the best pair', Z(lang, 'the best pair', '最好的一对') in res and '79%' in res, res[:200])
            rows = pg.evaluate("()=>document.querySelectorAll('#rgf-ch-res .cmp-row').length")
            check(f'{tag} challenge: the brief chart lists the 3 pairs that pass and the initial setup (59%, tagged)', rows == 4 and Z(lang, 'initial setup', '初始设置') in res and '59%' in res, rows)
            check(f'{tag} challenge: the pick is tagged', Z(lang, 'your pick', '你的选择') in res)
            check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#rgf-app [data-exits], #rgf-rail .exits')"))
            shot(pg, f'{tag}_ch_best', full=phone); rules(pg, f'{tag} challenge', lang)
            # P21 of the round-1 check: a pick just under the goal (70%–73%) keeps its label clear of the goal line, with the answer shown
            bad = []
            for pr in (('hyb', 'ctx'), ('ctx', 'cite'), ('ctx', 'dec'), ('hyb', 'cite'), ('hyb', 'dec'), ('cite', 'dec')):
                for f in ('hyb', 'ctx', 'ver', 'cite', 'dec'):
                    if f not in pr and pg.evaluate(f"()=>document.querySelector('#rgf-rail [data-chfx=\"{f}\"]').getAttribute('aria-pressed')") == 'true': click(pg, f'#rgf-rail [data-chfx="{f}"]', 80)
                for f in pr:
                    if pg.evaluate(f"()=>document.querySelector('#rgf-rail [data-chfx=\"{f}\"]').getAttribute('aria-pressed')") != 'true': click(pg, f'#rgf-rail [data-chfx="{f}"]', 80)
                click(pg, '#rgf-rail [data-act="submit"]', 150)
                if pg.query_selector('#rgf-rail [data-act="reveal"]'): click(pg, '#rgf-rail [data-act="reveal"]', 150)
                rows = pg.evaluate(HIT)
                if len(rows) < 4: bad.append((pr, 'no chart', len(rows)))
                bad += [(pr, r['txt']) for r in rows if r['hit'] or not r['inTrack']]
            check(f'{tag} challenge: near the goal (70%–73%) no value label is crossed by the goal line or leaves its track (P21)', not bad, bad[:4])
            pg.close()
            if phone:
                # N6 of the round-1 check, on a fresh page: a run waiting for ▶ plays when the reader taps the dock or picks “every stage”,
                # where it can be seen; a pick made before ▶ shows the whole record with its verdict
                pg = page(b, w, h, lang); a = anim(pg)
                check(f'{tag} N6: the first run waits for ▶', bool(a) and a.get('pending'), a)
                dk = dock_below(pg)
                check(f'{tag} N6: while the run waits, the dock says a tap plays it', Z(lang, 'Tap here to play this run', '点这里播放这次运行') in dk, dk)
                pg.click('#dock'); pg.wait_for_timeout(900); settle(pg); a = anim(pg); rv = pg.evaluate(REVEAL)
                check(f'{tag} N6: a tap on the dock plays the run where it can be seen', a is not None and not a['paused'] and not a.get('pending') and rv['ok'], (a, rv))
                wait_anim(pg); click(pg, NEXT, 300); a = anim(pg)
                check(f'{tag} N6: part 2’s run waits for ▶', bool(a) and a.get('pending') and a['key'] == 'e104', a)
                pg.evaluate(f"()=>document.querySelector('{SEG('log', '1')}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(300)
                pg.click(SEG('log', '1')); pg.wait_for_timeout(900); settle(pg); a = anim(pg); rv = pg.evaluate(REVEAL)
                check(f'{tag} N6: picking “every stage” plays the run, scrolled to where it can be seen', a is not None and a['key'] == 'e104' and not a['paused'] and not a.get('pending') and rv['ok'], (a, rv))
                shot(pg, f'{tag}_n6_log'); wait_anim(pg)
                obs = text(pg, '#rgf-rail [data-obs]')
                check(f'{tag} N6: the observation follows the run it describes', '9' in obs and not any(nexts(pg)), obs[:200])
                click(pg, NEXT, 300); a = anim(pg)
                check(f'{tag} N6: part 3’s first run waits for ▶', bool(a) and a.get('pending'), a)
                click(pg, '#rgf-rail [data-tri="unsup"]')
                vd = text(pg, '#rgf-flow .rgf-verdict')
                check(f'{tag} N6: a pick made before ▶ shows the whole record with its verdict', anim(pg) is None and Z(lang, 'Failed at', '错在') in vd and pg.evaluate(RED) >= 2, vd)
                rules(pg, f'{tag} N6', lang)
                pg.close()
    b.close()

for x in rule_hits: print('RULE', x)
check('writing, colour and layout rules', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:5])
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
