"""Flow test for the vector-index lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, no text wider than its box, a tooltip for every help cursor;
- part 1: the exact search plays in the picture on entry; Play, Step, the speed buttons and 重放 share one height and one baseline;
  重放 plays the search again at 中 in about 3 s (R3.9); Next enabled;
- part 2: the same picture, so entering replays nothing (R3.20); Next waits for another number of vectors, with the kit's why line;
  picking one replays nothing; Show me picks 1B and the observation names 3,072 GB and 30.7 s;
- part 3: the k-d tree picture and bars, no transport; Next waits for another number of dimensions; the observation names the
  share the engine gives; Show me picks 32 dimensions;
- part 4: Next waits for a prediction, with its why line; no result before it: no rings in the picture, no metric cards, no
  verdict (R3.8); after a pick the card says it is saved;
- part 5: the verdict on the prediction; the recall formula typeset (R6.29); the IVF search plays on entry; the missed true
  neighbours are ringed red (R5.11); the picture comes before the chart and the chart before the metric cards (R7.12); picking the
  nprobe already shown, or one already watched, replays nothing; a new nprobe plays; 重放 replays from the start; Step shows one
  step; the scrubber goes to the end (R3.9, R3.20); the observation names the engine's recall and comparisons; Back and Next
  replay nothing;
- part 6: the HNSW walk plays on entry; ef 64 plays, ef 10 again does not (R3.20); the observation names ef 64's numbers;
- part 7: the PQ grid, no transport; 16 bytes names its recall and 72 GB;
- part 8: the chart first, no picture, the bullets name the engine's numbers, the scenarios button, the next-lab footer;
- Scenarios: six; one already watched does not replay, a new one plays; each has its insight; the hash carries the setup;
  the language switch keeps the setup; the hash round trip opens the same setup;
- the challenge: starts at exact search with no result before Submit, also after another index is chosen; exact search misses
  with a hint and “Show the answer”; a new choice after a miss needs a new Submit; M = 32, ef = 32 is met but names M = 16,
  ef = 64, which is met with the fewest bytes; the chart of the passing settings tags the initial setup and the pick; the capability card;
- the dock at 390 once the stage scrolls away; every chart label inside its picture;
- the writing rules on every visible text and tooltip (no “toy” outside the Toy run label, Chinese spacing and punctuation, two colons in one sentence,
  no 万, 亿 only in “10 亿” as the design asks), the colours (red only for a miss and a failed verdict; the reserved blue, amber
  and violet not used), every formula typeset; and, read from the source, no “one at a time” note and no sentence that recalls a
  part by number.
Screenshots: shots/vector-index/.
Run: python3 tests/vector-index_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'vector-index').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/vector-index.html', '--only', 'vector-index'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'vector-index' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'vector-index.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'vector-index' / 'flow-wrapped.html')
SH = P / 'shots' / 'vector-index'; SH.mkdir(parents=True, exist_ok=True)
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


def pc1(x):
    t = f'{x * 100:.1f}'
    return (t[:-2] if t.endswith('.0') else t) + '%'


def page(b, w, h, lang, hash_='vector-index'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__vix.state()")
stat = lambda pg, ix, a, b_: pg.evaluate(f"()=>window.__vix.stat('{ix}',{a},{b_})")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#vix-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#vix-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#vix-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")
rings = lambda pg, cls='': pg.evaluate(f"()=>document.querySelectorAll('#vix-pic .vix-ring{cls}').length")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


# the animation state two frames after a click: False means nothing replays
CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__vix.state().anim;}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__vix.state().anim", timeout=timeout); pg.wait_for_timeout(150)


def mode(pg, m):
    click(pg, f'#vix-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b(?![?？])|undefined|\\[object|Infinity/,'broken value'],
   [/(?<!10 )亿|万/,'亿 (other than 10 亿) or 万'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft|Amazon|Apple\\b/,'company name'],
   [/吃|喂|扛|吞/,'figurative verb'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里|\\bin part [0-9]\\b|part [0-9]’s/,'recall of a part or a one-at-a-time note'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|dot|ring)/i,'colour named by lightness (R5.7)']];
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
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; const c=e.cloneNode(true); c.querySelectorAll('code').forEach(x=>x.remove()); texts.push(c.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# red only for a miss (the missed true neighbours ringed red, the failed verdict and status); blue, amber and violet are reserved
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/vix-ring bad|vix-sw-bad|badline|status bad|\\bbad\\b|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#vix-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#vix-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.vix-math-line')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#vix-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
SVGIN = """()=>{const out=[]; for(const s of document.querySelectorAll('#vix-stage svg, #vix-rail svg')){if(!s.getClientRects().length) continue; const b=s.getBoundingClientRect();
  const ts=[...s.querySelectorAll('text')].filter(t=>t.getClientRects().length&&t.textContent.trim());
  for(const t of ts){const r=t.getBoundingClientRect(); if(r.left<b.left-1||r.right>b.right+1||r.top<b.top-1||r.bottom>b.bottom+1) out.push((s.id||'svg')+' outside: '+t.textContent.trim().slice(0,30));}}
  return out.slice(0,8);}"""
# no TeX command shown as a word; no plain-text formula left in the rendered text (outside math, code and pre)
TEXWORDS = """()=>{const W=new Set(['frac','left','right','sqrt','text','sum','cdot','quad','lVert','rVert','mathrm','times','approx','cap']);
  const out=[]; for(const m of document.querySelectorAll('#vix-app math, #dock math')) for(const e of m.querySelectorAll('mi,mo')){const t=e.textContent.trim(); if(W.has(t)) out.push(t);}
  const pat=[/\\\\frac|\\\\text|\\\\lVert|\\\\sum|\\\\cap/,/\\|\\|q ?- ?x\\|\\|/];
  const tw=document.createTreeWalker(document.querySelector('#vix-app'),NodeFilter.SHOW_TEXT); let n;
  while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('math,pre,code')||!e.getClientRects().length) continue; for(const r of pat) if(r.test(n.textContent)) out.push('text: '+n.textContent.trim().slice(0,60));}
  for(const e of document.querySelectorAll('#vix-app [data-tipx]')){const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,''); for(const r of pat) if(r.test(t)) out.push('tip: '+t.slice(0,60));}
  return out;}"""
ALIGN = """()=>{const els=[...document.querySelectorAll('#vix-transport .btn, #vix-transport .seg-ctl')].filter(e=>e.getClientRects().length);
  const rows=[]; for(const e of els){const r=e.getBoundingClientRect(); let row=rows.find(x=>Math.abs(x.top-r.top)<8); if(!row){row={top:r.top,items:[]}; rows.push(row);}
    const b=e.matches('.seg-ctl')?e.querySelector('button'):e, rg=document.createRange(); rg.selectNodeContents(b); const t=rg.getBoundingClientRect();
    row.items.push({t:(e.textContent||'').trim().slice(0,12),top:r.top,h:r.height,base:t.bottom});}
  const bad=[]; for(const row of rows){const it=row.items; for(const x of it){if(Math.abs(x.h-it[0].h)>0.6||Math.abs(x.top-it[0].top)>0.6||Math.abs(x.base-it[0].base)>1) bad.push(x.t+' vs '+it[0].t+' h '+x.h+'/'+it[0].h+' base '+x.base.toFixed(1)+'/'+it[0].base.toFixed(1));}}
  return {n:els.length,bad};}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#vix-app', '#dock'):
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


NEXT = '#vix-rail .step.current [data-nav="next"]'
BACK = '#vix-rail .step.current [data-nav="back"]'


def go_next(pg):
    sel = NEXT if pg.query_selector(NEXT) else '#vix-rail [data-nav="next"]'
    click(pg, sel, 300)


def timed_run(pg, sel, timeout=20000):
    t0 = pg.evaluate('()=>performance.now()'); a = pg.evaluate(CLICK_ANIM, sel)
    pg.wait_for_function("()=>!window.__vix.state().anim", timeout=timeout)
    return a, (pg.evaluate('()=>performance.now()') - t0) / 1000


def tops(pg):
    return pg.evaluate("()=>{const q=s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?Math.round(e.getBoundingClientRect().top):null}; return {pic:q('#vix-pan-pic'), chart:q('#vix-pan-chart'), metrics:q('#vix-metrics')}}")


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            # ---- part 1: the exact search plays on entry
            check(f'{tag} part 1: the search plays in the picture on entry, Next enabled', st['anim'] and vis(pg, '#vix-pic') and vis(pg, '#vix-transport') and not any(nexts(pg)), (st, nexts(pg)))
            check(f'{tag} part 1: the speed defaults to 中 (R3.9)', speed(pg) == 'mid', speed(pg))
            al = pg.evaluate(ALIGN)
            check(f'{tag} part 1: Play, Step, the speed buttons and 重放 share one height and one baseline', al['n'] == 4 and not al['bad'], al)
            wait_anim(pg)
            a, dt = timed_run(pg, '#vix-transport [data-act="replay"]')
            check(f'{tag} part 1: 重放 plays the search again in about 3 s (R3.9, R3.20)', a and 2.4 <= dt <= 4.5, (a, dt))
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            # ---- part 2: exact search at real scale
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 2: the same picture, so nothing replays on entry (R3.20)', st['step'] == 1 and not a, (a, st))
            check(f'{tag} part 2: Next waits for another number of vectors, with the why line', all(nexts(pg)) and any(Z(lang, 'Pick another number of vectors', '换一个向量数') in x for x in why(pg)), why(pg))
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="rn"] button[data-v="1e8"]')
            check(f'{tag} part 2: another number of vectors replays nothing (the picture is the same) and enables Next', not a and not any(nexts(pg)) and state(pg)['realN'] == 1e8, (a, state(pg)))
            click(pg, '#vix-rail .step.current [data-nav="showme"]')
            ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 2: Show me picks 1B; the observation names 3,072 GB and 30.7 s', state(pg)['realN'] == 1e9 and '3,072 GB' in ob and Z(lang, '30.7 s', '30.7 秒') in ob, ob[:300])
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3: k-d tree
            go_next(pg); st = state(pg)
            check(f'{tag} part 3: the tree picture and bars, no transport, Next waits for another number of dimensions', st['step'] == 2 and vis(pg, '#vix-pic') and vis(pg, '#vix-pan-pic .vix-kdbars') and not vis(pg, '#vix-transport') and all(nexts(pg)) and any(Z(lang, 'another number of dimensions', '换一个维数') in x for x in why(pg)), (st, why(pg)))
            kd = pg.evaluate("()=>window.__vix.kd()")
            click(pg, '#vix-rail .seg-ctl[data-key="kd"] button[data-v="8"]')
            ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 3: 8 dimensions names {pc1(kd["8"])}; Next enabled', pc1(kd['8']) in ob and not any(nexts(pg)), ob[:200])
            click(pg, '#vix-rail .step.current [data-nav="showme"]')
            ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 3: Show me picks 32 dimensions, over 99.9% compared ({kd["32"]:.6f})', state(pg)['kdD'] == 32 and 0.999 < kd['32'] < 1 and Z(lang, 'over 99.9%', '99.9% 以上') in ob, ob[:200])
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4: predict
            go_next(pg); st = state(pg)
            check(f'{tag} part 4: Next waits for a prediction, with its why line', st['step'] == 3 and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), why(pg))
            cur = text(pg, '#vix-rail .step.current')
            check(f'{tag} part 4: no result before the prediction (R3.8): no rings, no metric cards, no verdict', rings(pg) == 0 and not vis(pg, '#vix-metrics') and not vis(pg, '#vix-transport') and Z(lang, 'Right.', '对了。') not in cur and '%' not in text(pg, '#dock'), (rings(pg), cur[:200]))
            click(pg, '#vix-rail [data-pred="all"]')
            check(f'{tag} part 4: after a pick, Next is enabled and the card says it is saved', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#vix-rail .step.current'))
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            # ---- part 5: IVF
            a = click_anim(pg, NEXT); st = state(pg)
            cur = text(pg, '#vix-rail .step.current')
            check(f'{tag} part 5: the IVF search plays on entry at nprobe 1 (a new picture)', a and st['ix'] == 'ivf' and st['np'] == 1, (a, st))
            check(f'{tag} part 5: the verdict on the prediction', Z(lang, 'Not quite.', '不对。') in cur and Z(lang, 'You predicted', '你预测的是') in cur, cur[:300])
            check(f'{tag} part 5: recall@10 is typeset with the kit (R6.29)', vis(pg, '#vix-rail .step.current .vix-math-line math'))
            wait_anim(pg)
            check(f'{tag} part 5: the true neighbours the search missed are ringed red (R5.11)', rings(pg, '.bad') > 0 and rings(pg, '.ok') > 0, (rings(pg, '.bad'), rings(pg, '.ok')))
            tp = tops(pg)
            check(f'{tag} part 5: the picture first, then the chart, then the metric cards (R7.12)', None not in tp.values() and tp['pic'] < tp['chart'] < tp['metrics'], tp)
            check(f'{tag} part 5: Next waits for another nprobe', all(nexts(pg)) and any(Z(lang, 'another nprobe', '换一个 nprobe') in x for x in why(pg)), why(pg))
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="np"] button[data-v="1"]')
            check(f'{tag} part 5: picking the nprobe already shown replays nothing (R3.20)', not a, a)
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="np"] button[data-v="4"]')
            check(f'{tag} part 5: a new nprobe plays its search (R3.20)', a and state(pg)['np'] == 4, (a, state(pg)))
            wait_anim(pg)
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="np"] button[data-v="1"]')
            check(f'{tag} part 5: back to a watched nprobe replays nothing (R3.20)', not a and state(pg)['picF'] == -1, (a, state(pg)))
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="np"] button[data-v="4"]')
            check(f'{tag} part 5: and nprobe 4 again, watched too, replays nothing (R3.20)', not a, a)
            a = click_anim(pg, '#vix-transport [data-act="replay"]')
            check(f'{tag} part 5: 重放 replays the search from the start (R3.20)', a and state(pg)['picF'] >= 0, (a, state(pg)))
            wait_anim(pg)
            a = click_anim(pg, '#vix-transport [data-act="stepf"]')
            check(f'{tag} part 5: Step shows the first step without playing (R3.9)', not a and state(pg)['picF'] == 0, (a, state(pg)))
            click(pg, '#vix-transport [data-act="stepf"]')
            check(f'{tag} part 5: Step again shows the next step', state(pg)['picF'] == 1, state(pg))
            pg.evaluate("()=>{const r=document.querySelector('#vix-scrub'); r.value=r.max; r.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(150)
            check(f'{tag} part 5: the scrubber goes to the end of the search', state(pg)['picF'] == -1 and not state(pg)['anim'], state(pg))
            s4 = stat(pg, 'ivf', 4, 0)
            ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 5: the observation names nprobe 4: {pc1(s4["rec"])} and {round(s4["comps"])} compared; Next enabled', pc1(s4['rec']) in ob and f'{round(s4["comps"]):,}' in ob and not any(nexts(pg)), ob[:300])
            click(pg, '#vix-rail .step.current [data-nav="showme"]'); wait_anim(pg)
            check(f'{tag} part 5: Show me picks nprobe 16', state(pg)['np'] == 16, state(pg))
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            click(pg, BACK)
            a = click_anim(pg, NEXT)
            check(f'{tag} part 5 again: Back and Next replay nothing (R3.20)', not a and state(pg)['step'] == 4, (a, state(pg)))
            # ---- part 6: HNSW
            a = click_anim(pg, NEXT); st = state(pg)
            check(f'{tag} part 6: the HNSW walk plays on entry (M = 16, ef = 10)', a and (st['ix'], st['M'], st['ef']) == ('hnsw', 16, 10), (a, st))
            wait_anim(pg)
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="64"]')
            check(f'{tag} part 6: ef 64 plays its walk (R3.20)', a, a)
            wait_anim(pg)
            a = click_anim(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="10"]')
            check(f'{tag} part 6: back to ef 10, watched, replays nothing (R3.20)', not a, a)
            click(pg, '#vix-rail .step.current [data-nav="showme"]')
            check(f'{tag} part 6: Show me picks ef 64, already watched, so nothing replays', state(pg)['ef'] == 64 and not state(pg)['anim'], state(pg))
            s = stat(pg, 'hnsw', 16, 64); ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 6: the observation names ef 64: {pc1(s["rec"])} and {round(s["comps"])} compared', pc1(s['rec']) in ob and f'{round(s["comps"]):,}' in ob and not any(nexts(pg)), ob[:300])
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- part 7: PQ
            go_next(pg); st = state(pg)
            check(f'{tag} part 7: the PQ grid, no transport, Next waits for another number of bytes', st['ix'] == 'pq' and pg.evaluate("()=>document.querySelectorAll('#vix-pic .vix-grid').length") > 4 and not vis(pg, '#vix-transport') and all(nexts(pg)), (st, nexts(pg)))
            click(pg, '#vix-rail .seg-ctl[data-key="m"] button[data-v="16"]')
            s = stat(pg, 'pq', 16, 16); ob = text(pg, '#vix-rail [data-obs]')
            check(f'{tag} part 7: 16 bytes names {pc1(s["rec"])} and 72 GB; Next enabled', pc1(s['rec']) in ob and '72 GB' in ob and not any(nexts(pg)), ob[:300])
            shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang)
            # ---- part 8
            go_next(pg); cur = text(pg, '#vix-rail .step.current'); hn = stat(pg, 'hnsw', 16, 64); tp = tops(pg)
            check(f'{tag} part 8: the chart, no picture, the metric cards below', tp['pic'] is None and tp['chart'] is not None and tp['chart'] < tp['metrics'], tp)
            check(f'{tag} part 8: the bullets name {pc1(hn["rec"])} and the scenarios button', state(pg)['step'] == 7 and pc1(hn['rec']) in cur and vis(pg, '#vix-rail .step.current [data-mode-go="sandbox"]'), cur[:300])
            check(f'{tag} part 8: the next-lab footer', pg.evaluate("()=>[...document.querySelectorAll('#vix-app .next-lab')].some(e=>e.getBoundingClientRect().height>0)"))
            shot(pg, f'{tag}_p8', full=phone); rules(pg, f'{tag} part 8', lang)
            # ---- scenarios
            mode(pg, 'sandbox')
            check(f'{tag} scenarios: six scenarios', pg.evaluate("()=>document.querySelectorAll('#vix-rail [data-preset]').length") == 6)
            a = click_anim(pg, '#vix-rail [data-preset="0"]')
            check(f'{tag} scenarios: exact search was watched in the tour, so it does not replay (R3.20)', not a and state(pg)['ix'] == 'flat', (a, state(pg)))
            a = click_anim(pg, '#vix-rail [data-preset="2"]')
            check(f'{tag} scenarios: IVF nprobe 16 was watched in the tour, so it does not replay (R3.20)', not a and state(pg)['np'] == 16, (a, state(pg)))
            ins = text(pg, '#vix-insight'); s = stat(pg, 'ivf', 16, 0)
            check(f'{tag} scenarios: the insight names {pc1(s["rec"])}', pc1(s['rec']) in ins, ins[:200])
            a, dt = timed_run(pg, '#vix-rail [data-preset="4"]')
            # the first M = 32 graph is built on this click (about 0.5–1 s), then the search plays at 中 (3 s)
            check(f'{tag} scenarios: a search not yet watched plays (FAISS defaults, ef 16), building M = 32 first', a and dt <= 5.5, (a, dt))
            for i in (1, 3, 5):
                click(pg, f'#vix-rail [data-preset="{i}"]'); wait_anim(pg)
                ins = text(pg, '#vix-insight')
                check(f'{tag} scenarios: scenario {i + 1} has its insight', '%' in ins and len(ins) > 80, ins[:200])
            click(pg, '#vix-rail [data-preset="4"]')
            hsh = pg.evaluate("()=>location.hash")
            check(f'{tag} scenarios: the hash carries the setup', hsh.startswith('#vector-index~hnsw_') and hsh.endswith('_32_16_8'), hsh)
            shot(pg, f'{tag}_scen', full=phone); rules(pg, f'{tag} scenarios', lang)
            other = 'zh' if lang == 'en' else 'en'
            s0 = state(pg); click(pg, f'.langsw button[data-lang="{other}"]', 300); s1 = state(pg)
            check(f'{tag} scenarios: the language switch keeps the setup', all(s0[k] == s1[k] for k in ('mode', 'ix', 'np', 'M', 'ef', 'm')), (s0, s1))
            click(pg, f'.langsw button[data-lang="{lang}"]', 300)
            if phone:
                pg.evaluate("()=>{const a=document.querySelector('#vix-pan-pic:not([hidden])')||document.querySelector('#vix-pan-chart'); window.scrollTo(0,a.getBoundingClientRect().bottom+scrollY+400)}"); pg.wait_for_timeout(800)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getBoundingClientRect().height>0?d.innerText:''}")
                check(f'{tag}: the dock shows once the stage scrolls away', bool(dk) and '%' in dk, dk)
                shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            pg.close()
            pg = page(b, w, h, lang, hash_='vector-index~pq_16_16_64_16'); st = state(pg)
            check(f'{tag}: the hash round trip opens the same setup in scenarios', st['mode'] == 'sandbox' and (st['ix'], st['np'], st['M'], st['ef'], st['m']) == ('pq', 16, 16, 64, 16), st)
            # ---- challenge
            mode(pg, 'challenge'); st = state(pg)
            nores = lambda: '%' not in text(pg, '#vix-ch-av') and not text(pg, '#vix-ch-res').strip() and not state(pg)['chSubmitted'] and '%' not in text(pg, '#vix-metrics')
            check(f'{tag} challenge: starts at exact search, no result before Submit', st['mode'] == 'challenge' and st['ix'] == 'flat' and nores(), (st, text(pg, '#vix-ch-av')))
            check(f'{tag} challenge: the bytes per vector are shown before Submit (128)', '128' in text(pg, '#vix-ch-gv'), text(pg, '#vix-ch-gv'))
            click(pg, '#vix-rail .seg-ctl[data-key="ix"] button[data-v="hnsw"]')
            check(f'{tag} challenge: another index before Submit still shows no result', nores() and '256' in text(pg, '#vix-ch-gv'), (text(pg, '#vix-ch-av'), state(pg)))
            click(pg, '#vix-rail .seg-ctl[data-key="ix"] button[data-v="flat"]')
            click(pg, '#vix-rail [data-act="submit"]')
            res = text(pg, '#vix-ch-res')
            check(f'{tag} challenge: exact search misses (5,000 compared), with a hint and “{Z(lang, "Show the answer", "直接看答案")}”', Z(lang, 'Not yet', '还没达到') in res and '5,000' in res and vis(pg, '#vix-rail [data-act="reveal"]') and Z(lang, 'Misses a bar', '没过线') in text(pg, '#vix-status'), (res, text(pg, '#vix-status')))
            click(pg, '#vix-rail .seg-ctl[data-key="ix"] button[data-v="hnsw"]')
            check(f'{tag} challenge: a new index after a miss needs a new Submit', not state(pg)['chSubmitted'] and not text(pg, '#vix-ch-res').strip() and vis(pg, '#vix-rail [data-act="submit"]'), state(pg))
            click(pg, '#vix-rail .seg-ctl[data-key="M"] button[data-v="32"]'); click(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="32"]')
            click(pg, '#vix-rail [data-act="submit"]')
            res = text(pg, '#vix-ch-res')
            check(f'{tag} challenge: M = 32, ef = 32 is met, but M = 16, ef = 64 does it with 256 bytes', Z(lang, 'Goal met', '达成目标') in res and '256' in res and '384' in res, res[:300])
            click(pg, '#vix-rail .seg-ctl[data-key="M"] button[data-v="16"]'); click(pg, '#vix-rail .seg-ctl[data-key="ef"] button[data-v="64"]'); click(pg, '#vix-rail [data-act="submit"]')
            res = text(pg, '#vix-ch-res')
            check(f'{tag} challenge: M = 16, ef = 64 is met with the fewest bytes; the chart appears', Z(lang, 'Among the ef values this task offers, no setting meets both bars with fewer bytes', '在这道题能选的 ef 里，没有哪种设置能用更少的字节同时满足两条线') in res and vis(pg, '#vix-ch-res .cmp'), res[:300])
            rows = pg.evaluate("()=>[...document.querySelectorAll('#vix-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:(r.querySelector('.cmp-tag,.cmp-you')||{}).textContent||'',lab:r.querySelector('.cmp-lab').textContent}))")
            check(f'{tag} challenge: the chart tags the initial setup and the pick; every other row meets both bars', sum(r['me'] for r in rows) == 1 and any(Z(lang, 'initial setup', '初始设置') in r['tag'] for r in rows) and all(r['ok'] or r['me'] or r['tag'] for r in rows), rows)
            check(f'{tag} challenge: the capability card', vis(pg, '#vix-rail [data-exits]'))
            shot(pg, f'{tag}_challenge', full=phone); rules(pg, f'{tag} challenge', lang)
            check(f'{tag}: no horizontal overflow at the end', overflow(pg) <= 0, overflow(pg))
            pg.close()
    # ---- 900 wide: chart text inside, no overflow
    if not [x for x in ARGS if x in ('1440', '390')]:
        for lang in LANGS:
            tag = f'{lang}-900'; pg = page(b, 900, 900, lang, hash_='vector-index~hnsw_16_16_64_8'); wait_anim(pg)
            out_ = pg.evaluate(SVGIN); check(f'{tag} scenarios: every chart label inside its picture', not out_, out_)
            check(f'{tag}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            shot(pg, f'{tag}_scen'); pg.close()
    b.close()

for h in sorted(set(rule_hits)): print('RULE', h)
check('writing rules, colours, overflow and tooltips', not rule_hits, len(rule_hits))
check('no page errors', not errs, errs[:6])
# read from the lab's own source, so both languages are covered
_SRC = ''.join((P / d / 'vector-index.js').read_text() for d in ('labs', 'teasers'))
_G1 = re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一|第 [0-9一二三四五六七八] 部分里|[Ii]n part [0-9]|part [0-9]’s|第 [0-9] 部分的|parts? [0-9] (?:to|and) [0-9]|第 [0-9] 到第 [0-9] 部分).{0,30}', _SRC)
check('no “one at a time” note and no sentence recalling a part by number (EN and ZH, R6.34)', not _G1, _G1[:3])
_H4 = re.findall(r'.{0,30}(?:规划中|还没上线|not built|later lab|以后的 Lab|将来|[Pp]lanned lab).{0,30}', _SRC)
check('no lab called planned or not yet live', not _H4, _H4[:3])
_TOY = re.findall(r'.{0,20}(?:\btoy\b(?! run)|玩具(?!实跑)).{0,20}', re.sub(r'Toy run|玩具实跑', '', _SRC), flags=re.I)
check('“toy” only in the Toy run label', not _TOY, _TOY[:3])
_YI = re.findall(r'.{0,12}(?:(?<!10 )(?<!\} )亿|万).{0,12}', _SRC)
check('亿 only in “10 亿” (the design asks for it), no 万', not _YI, _YI[:3])
print(f'{n - len(fails)} / {n} passed')
sys.exit(1 if fails else 0)
