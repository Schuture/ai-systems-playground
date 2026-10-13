"""Flow test for the optimizers lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step;
- part 1: the three runs (104 / 78 / 74 steps) on the stage and in the lead;
- the prediction gate (R3.8): Next disabled until a prediction; before it nothing on the page, in text, popovers or tooltips, gives the
  answer (SGD's limit 0.08, 2 ÷ 25, η < 2/λ, −1.5, “0.10 … diverges”); “Saved …” after a pick (R2.5);
- part 3: Next waits until the learning rate passes 0.08; at 0.10 the result, the verdict with the right answer, and why (R3.10);
  ▲▼ badges only after the learner's own change between two step counts (R6.6);
- the player (R3.9, owner check 2026-10-03): the paths play on their own when the valley is in view and stop at the last step; ▶ Play /
  ❚❚ Pause / ▶ Replay and three speeds; the step slider pauses; with reduced motion the end state at once, and Play still plays;
- part 4: β 0.6 gives 30 steps; part 5: starts at SGD's learning rate 0.05 (Adam 0.05 a step, SGD 0.45 and 1.88), Adam's first step is
  lr in x and in y; part 6: single choice, the result on the lever, 3.16 × lr without bias correction; part 7 (AdamW): two levers, its
  own chart, 0.010% and 1.0% against 0.10% for both, Llama 2's settings; part 8: four points, three misconceptions, “See the
  scenarios”, the next-lab footer (R3.11, R3.12);
- Scenarios: four, single choice, pressed state follows the settings (R1.8), every insight cites its source, Settings unlocked and
  the URL carries the setup (a fresh page restores it);
- the challenge: only the two settings in the card, Settings hidden with a note (R4.2), live progress, the three feedback tiers (R4.4),
  “Show the answer”, a brief chart (the 5 that pass, the initial setup and the pick, of 18; the owner, 2026-10-07), the capability card; a language switch keeps the answer;
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万, no “toy”, no 起点 /
  starting point, no 控件 / Controls), colours only for their data (R5.6), no clipped text (R7.8), a tooltip for every help cursor (R1.1).
- the home tile (review home-v4-A #1: it must answer its title, “What does Adam do that SGD does not?”): SGD / Adam, each pick drawn
  solid with the other one dashed on the same valley, at 1440 and 390, EN and ZH, light and dark, and in 280 and 440 px tiles: the
  sentence is the hook's and changes, at most about 35 words (R11.5), the picture at most 130 px tall with every ellipse and path inside
  its frame, nothing outside the tile and no horizontal scroll, the legend matching the line styles (R5.7), grayscale only (R5.6), a
  steady tile height, four notes behind “Assumptions & sources”, nothing of part 2's answer or of the challenge (R3.8), the writing
  rules, the language switch keeping the pick; screenshots of every pick in out/v4/shots-opt/.
Screenshots: shots/optimizers/ (the lab) and out/v4/shots-opt/ (the tile).
Run: python3 tests/optimizers_flow.py [en|zh] [1440|390] [tile]  (filters for a partial run; “tile” runs only the home tile)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/optimizers-flow.html', '--only', 'optimizers'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'optimizers-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'optimizers-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'optimizers-flow-wrapped.html')
SH = P / 'shots' / 'optimizers'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
TILE_ONLY = 'tile' in ARGS
ST = P / 'out' / 'v4' / 'shots-opt'; ST.mkdir(parents=True, exist_ok=True)
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, hash_='optimizers', dark=False):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.clock.install()   # the playback runs on the page clock, so the test can fast-forward a 10–40 s replay (played)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__op.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#op-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(200); settle(pg)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(200); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(200); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#op-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(300); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(60)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #op-mathbody, .num-big, #op-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|op-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#op-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""

# R7.8: no text wider than its box (clipped or ellipsised), in the lab and the mini result bar
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#op-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# R1.1: every element with a help cursor explains itself on hover (itself or an ancestor carries the tooltip)
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#op-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#op-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""
CHROWS = "()=>[...document.querySelectorAll('#op-ch-res .cmp-row')].map(r=>[r.classList.contains('ok'),r.classList.contains('me'),!!r.querySelector('.cmp-tag')])"


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#op-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    ov = overflow(pg)
    check(f'{where}: no horizontal overflow', ov <= 0, ov)


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#op-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#op-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


NAMES_JS = """()=>{const R=e=>{const b=e.getBBox(); return {x0:b.x,x1:b.x+b.width,y0:b.y,y1:b.y+b.height,t:e.textContent}}, sv=document.querySelector('#op-loss'), mp=document.querySelector('#op-map');
 const L=[...sv.querySelectorAll('.op-ll,.op-gl')].map(R), gy=+sv.querySelector('.op-goal').getAttribute('y1'), bad=[], vb=mp.viewBox.baseVal;
 L.forEach((a,i)=>{if(a.x0<58||a.x1>584||a.y0<8||a.y1>132) bad.push('outside the plot: '+a.t); if(a.y0<gy&&a.y1>gy) bad.push('on the goal line: '+a.t);
   L.slice(i+1).forEach(b=>{if(a.x0<b.x1&&b.x0<a.x1&&a.y0<b.y1&&b.y0<a.y1) bad.push('overlap: '+a.t+' / '+b.t);});});
 const M=[...mp.querySelectorAll('.op-pl,.op-tl')].map(R);
 M.forEach((a,i)=>{if(a.x0<0||a.x1>vb.width||a.y0<0||a.y1>vb.height) bad.push('outside the map: '+a.t); M.slice(i+1).forEach(b=>{if(a.x0<b.x1&&b.x0<a.x1&&a.y0<b.y1&&b.y0<a.y1) bad.push('map overlap: '+a.t+' / '+b.t);});});
 return {map:[...mp.querySelectorAll('.op-pl')].map(e=>e.textContent), loss:[...sv.querySelectorAll('.op-ll')].map(e=>e.textContent), goal:[...sv.querySelectorAll('.op-gl')].map(e=>e.textContent).join(), bad, legend:!document.querySelector('#op-legend').hidden}}"""
def names(pg): return pg.evaluate(NAMES_JS)   # R5.9: the names on the charts, and any overlap among them or with the goal line and the axes
def obs_all(pg): return pg.evaluate("()=>{const o=document.querySelector('#op-rail [data-obs]'); return o&&!o.hidden?o.textContent.replace(/\\u00a0/g,' '):''}")   # with the folded “In numbers” box


def played(pg, ms=60000):   # fast-forwards the page clock until the playback stops (R3.9: a whole replay takes 10 to 40 s)
    t = 0
    while t < ms and state(pg)['playing']: pg.clock.run_for(1000); t += 1000
    return not state(pg)['playing']


# everything a learner can read before the prediction, outside Connect: the header and its popovers (even closed), the rail, the stage,
# Settings, the dock, their tooltips, and every term tooltip in the rail (R3.8)
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#op-app .labhead','#op-rail','#op-stage','#op-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#op-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#op-rail .t, #op-rail .def').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#op-rail .t, #op-rail .def')[{i}]; e.focus({{preventScroll:true}}); e.dispatchEvent(new MouseEvent('mouseover',{{bubbles:true}}))}}"); pg.wait_for_timeout(60)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out)


LEAK = [r'0\.08', r'2 ÷ 25', r'2/λ', r'η < 2', r'−1\.5', r'0\.10?\b[^\n.。]{0,40}(diverg|发散)', r'(diverg|发散)[^\n.。]{0,40}0\.10?\b']


def run_guided(b, w, h, lang):
    tag = f'{w} {lang}'; phone = w <= 760
    pg = page(b, w, h, lang)
    check(f'{tag}: the lab mounts in Guided, part 1', state(pg)['mode'] == 'guided' and state(pg)['step'] == 0)
    # part 1
    t1 = text(pg, '#op-rail') + text(pg, '#op-metrics')
    check(f'{tag} part 1: 104 / 78 / 74 steps in the lead and on the stage', all(x in t1 for x in ('104', '78', '74')), t1[:300])
    check(f'{tag} part 1: three paths on the valley', pg.evaluate("()=>document.querySelectorAll('#op-map .op-path').length") == 3)
    # the player (R3.9): the valley is in view at 1440, so the paths play on their own and stop at the last step (104)
    if not phone:
        pg.wait_for_timeout(300)
        st = state(pg)
        check(f'{tag} player: the paths play on their own when the valley is in view', st['playing'] or st['K'] == 104, st['K'])
    else:
        pg.evaluate("()=>document.getElementById('op-map').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400)
    check(f'{tag} player: the playback stops at the last step of the slowest run (104)', played(pg) and state(pg)['K'] == 104 == pg.evaluate("()=>window.__op.kMax()"), state(pg)['K'])
    pt = text(pg, '#op-player [data-act="play"]')
    check(f'{tag} player: ▶ Replay at the end, three speeds, the step slider and ◂ ▸', pt == Z(lang, '▶ Replay', '▶ 重放') and pg.evaluate("()=>document.querySelectorAll('#op-player .seg-ctl[data-key=\"spd\"] button').length") == 3
          and vis(pg, '#op-k') and vis(pg, '#op-player [data-act=\"k-\"]'), pt)
    click(pg, '#op-player [data-act="play"]'); st = state(pg)
    check(f'{tag} player: Replay plays from the start, the button says Pause', st['playing'] and st['K'] < 60 and text(pg, '#op-player [data-act="play"]') == Z(lang, '❚❚ Pause', '❚❚ 暂停'), (st['K'], text(pg, '#op-player [data-act="play"]')))
    click(pg, '#op-player [data-act="play"]'); k1 = state(pg)['K']; pg.wait_for_timeout(300)
    check(f'{tag} player: Pause stops it where it is', not state(pg)['playing'] and state(pg)['K'] == k1 and text(pg, '#op-player [data-act="play"]') == Z(lang, '▶ Play', '▶ 播放'), (k1, state(pg)['K']))
    click(pg, '#op-player .seg-ctl[data-key="spd"] button[data-v="fast"]'); click(pg, '#op-player [data-act="play"]')
    check(f'{tag} player: a speed is kept, and Play goes on from the step shown', state(pg)['spd'] == 'fast' and (state(pg)['playing'] or state(pg)['K'] == 104) and state(pg)['K'] >= k1)
    slide(pg, '#op-k', 30)
    check(f'{tag} player: the step slider pauses the playback and shows that step', not state(pg)['playing'] and state(pg)['K'] == 30 and Z(lang, 'step 30', '第 30 步') in text(pg, '#op-k-out'), state(pg)['K'])
    click(pg, '#op-player .seg-ctl[data-key="spd"] button[data-v="med"]')
    nm = names(pg)
    check(f'{tag} part 1: every path is named on the map and every loss line at its end, the goal on its line; no overlap, no legend row (R5.9)', all(x in nm['map'] and x in nm['loss'] for x in ['SGD', 'Momentum', 'Adam']) and nm['goal'] == Z(lang, 'goal 0.001', '目标 0.001') and not nm['bad'] and not nm['legend'], nm)
    click(pg, '#op-player [data-act="play"]'); played(pg); nm = names(pg)
    check(f'{tag} part 1, at the end: each loss line named at its own end, nothing overlapping (R5.9)', sorted(nm['loss']) == ['Adam', 'Momentum', 'SGD'] and not nm['bad'], nm)
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang)
    # part 2: the prediction gate
    click(pg, '#op-rail [data-nav="next"]')
    check(f'{tag} part 2: Next disabled until a prediction', all(nexts(pg)) and state(pg)['step'] == 1)
    st = state(pg)
    check(f'{tag} part 2: the SGD path was already on the valley, so it shows at its end, no replay (R3.20)', not st['playing'] and st['K'] == pg.evaluate("()=>window.__op.kMax()"), (st['playing'], st['K']))
    mode(pg, 'sandbox'); st = state(pg)
    check(f'{tag} R3.20: Scenarios starts with the same SGD path, so it shows at its end, no replay', st['mode'] == 'sandbox' and not st['playing'] and st['K'] == pg.evaluate("()=>window.__op.kMax()"), (st['mode'], st['playing'], st['K']))
    mode(pg, 'guided'); st = state(pg)
    check(f'{tag} R3.20: back in part 2, still no replay', st['mode'] == 'guided' and st['step'] == 1 and not st['playing'], (st['step'], st['playing'], st['K']))
    check(f'{tag} part 2: “why Next is greyed out” shown', Z(lang, 'Pick an answer', '选一个答案') in text(pg, '#op-rail'))
    ba = before_answer(pg)
    leaks = [p for p in LEAK if re.search(p, ba)]
    check(f'{tag} part 2: nothing before the answer gives it away (R3.8)', not leaks, [(p, re.search(p, ba).group(0)) for p in leaks])
    shot(pg, f'{tag}_p2', full=phone)
    rules(pg, f'{tag} part 2', lang)
    click(pg, '#op-rail [data-pred="fewer"]')
    check(f'{tag} part 2: saved, Next enabled', Z(lang, 'Saved', '已记下') in text(pg, '#op-rail') and not any(nexts(pg)) and state(pg)['pred'] == 'fewer')
    # part 3
    click(pg, '#op-rail [data-nav="next"]')
    check(f'{tag} part 3: Next waits, the slider at 0.05, no badge yet', all(nexts(pg)) and state(pg)['lrP'] == 0.05 and not badges(pg), badges(pg))
    check(f'{tag} part 3: why Next is greyed out', Z(lang, 'above 0.08', '0.08 以上') in text(pg, '#op-rail'))
    slide(pg, '#op-ic-lr-in', 11)   # 0.06
    if not phone:
        st = state(pg)
        if not st['playing']:   # a new path waits until the valley is in view, then plays
            pg.evaluate("()=>document.getElementById('op-map').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400); st = state(pg)
        check(f'{tag} part 3, 0.06: a new learning rate plays the new path from step 1 (R3.20)', st['playing'] and st['K'] < 60, (st['playing'], st['K']))
    o = obs(pg); bd = badges(pg)
    check(f'{tag} part 3, 0.06: 86 steps, still waiting, a badge for the learner’s change', '86' in o and all(nexts(pg)) and len(bd) >= 1, (o[:200], bd))
    slide(pg, '#op-ic-lr-in', 15)   # 0.08
    o = obs(pg)
    check(f'{tag} part 3, 0.08: bounces forever', Z(lang, 'exactly 0.08', '正好 0.08') in o, o[:200])
    slide(pg, '#op-ic-lr-in', 19)   # 0.10
    o = obs(pg); bd = badges(pg)
    check(f'{tag} part 3, 0.10: never reaches the bottom, the verdict with the right answer, and why', Z(lang, 'never reaches the bottom', '永远到不了谷底') in o and Z(lang, 'The right answer is', '正确答案是') in o and Z(lang, 'Why:', '原因：') in o and not any(nexts(pg)), o[:400])
    check(f'{tag} part 3, 0.10: the arithmetic (−1.5) is folded in “In numbers”, after the words (R6.27)', Z(lang, 'In numbers', '用数字看') in o and '−1.5' not in o and '−1.5' in obs_all(pg), o[-200:])
    check(f'{tag} part 3, 0.10: no badge between a step count and “diverges”', not any('▲' in x or '▼' in x for x in bd), bd)
    check(f'{tag} part 3: the status says diverges', text(pg, '#op-status') == Z(lang, 'diverges', '发散'))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang)
    # part 4
    click(pg, '#op-rail [data-nav="next"]')
    check(f'{tag} part 4: β starts at 0, Next waits', state(pg)['betaP'] == 0 and all(nexts(pg)))
    slide(pg, '#op-ic-beta-in', 12)   # 0.6
    o = obs(pg)
    check(f'{tag} part 4, β 0.6: 30 steps against 104', '30' in o and '104' in o and not any(nexts(pg)), o[:200])
    slide(pg, '#op-ic-beta-in', 18)   # 0.9
    o = obs(pg)
    check(f'{tag} part 4, β 0.9: 78 steps, overshoots', '78' in o and Z(lang, 'overshoots', '冲过了谷底') in o, o[:300])
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang)
    # part 5
    click(pg, '#op-rail [data-nav="next"]')
    check(f'{tag} part 5: Next waits, Adam starts at SGD’s learning rate 0.05', all(nexts(pg)) and state(pg)['lrA'] == 0.05)
    pg.evaluate("()=>document.getElementById('op-map').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 5: Adam at 0.05 is a new path, so it plays (R3.20)', st['playing'], (st['playing'], st['K']))
    check(f'{tag} part 5: the new paths play to the end', played(pg) and state(pg)['K'] == pg.evaluate("()=>window.__op.kMax()"), state(pg)['K'])
    check(f'{tag} part 5: the step bars show x and y for SGD and Adam', pg.evaluate("()=>document.querySelectorAll('#op-bars .op-brow').length") == 2)
    slide(pg, '#op-ic-lrA-in', 5)   # 1.0
    o = obs(pg)
    check(f'{tag} part 5, lr 1.0: the first step moves x and y by 1.00', Z(lang, 'moves x and y by 1.00 each', '在 x 和 y 方向各走 1.00') in o and not any(nexts(pg)), o[:300])
    slide(pg, '#op-ic-lrA-in', 0)   # back to 0.05
    pg.evaluate("()=>document.getElementById('op-map').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 5: back to 0.05 after 1.0 (A → B → A), the paths already watched show at their end, no replay (R3.20)', not st['playing'] and st['K'] == pg.evaluate("()=>window.__op.kMax()"), (st['playing'], st['K']))
    o = obs(pg)
    check(f'{tag} part 5, lr 0.05: both first steps, then why Adam’s learning rates look small for LLMs (owner check, point 2)', all(k in o for k in ('0.05', '0.45', '1.88', '0.5', '0.0003')) and Z(lang, 'LLM', '大模型') in o, o[:500])
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang)
    # part 6
    click(pg, '#op-rail [data-nav="next"]')
    pg.evaluate("()=>document.getElementById('op-map').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 6: Adam at 0.5 raced to the end in part 1, so it shows at its end, no replay (R3.20)', not st['playing'] and st['K'] == pg.evaluate("()=>window.__op.kMax()"), (st['playing'], st['K']))
    check(f'{tag} part 6: two levers, single choice, Next waits', pg.evaluate("()=>document.querySelectorAll('#op-rail [data-lever]').length") == 2 and all(nexts(pg)))
    click(pg, '#op-rail [data-lever="off"]')
    o = obs(pg); lv = pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-lever]')].map(b=>[b.getAttribute('aria-pressed'),b.querySelector('.lv-r')?b.querySelector('.lv-r').textContent:''])")
    check(f'{tag} part 6: off pressed, 3.16 × lr in the result and on the lever', '3.16' in o and lv[1][0] == 'true' and lv[0][0] == 'false' and '3.16' in lv[1][1] and '1.00' in lv[0][1] and not any(nexts(pg)), (o[:200], lv))
    nm = names(pg)
    check(f'{tag} part 6: the dash-dot line for Adam without bias correction, named on both charts (R5.9)', pg.evaluate("()=>document.querySelectorAll('#op-map .op-path.op-s3').length") == 1 and Z(lang, 'no correction', '无校正') in nm['loss'] and not nm['bad'], nm)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang)
    # part 7: AdamW
    click(pg, '#op-rail [data-nav="next"]')
    check(f'{tag} part 7: two levers, Adam + L2 first, Next waits', pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-lever]')].map(b=>b.dataset.lever).join()") == 'l2,adamw' and all(nexts(pg)) and state(pg)['wdP'] == 'l2')
    check(f'{tag} part 7: its own chart, the valley hidden', vis(pg, '#op-wd') and not vis(pg, '#op-map') and not vis(pg, '#op-player') and pg.evaluate("()=>document.querySelectorAll('#op-wd .op-wrow').length") == 2)
    o = obs(pg)
    check(f'{tag} part 7, Adam + L2: 0.010% against 1.0%, about 100 times', all(k in o for k in ('0.010%', '1.0%', '100')), o[:300])
    click(pg, '#op-rail [data-lever="adamw"]')
    o = obs(pg); lv = pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-lever]')].map(b=>[b.getAttribute('aria-pressed'),b.querySelector('.lv-r')?b.querySelector('.lv-r').textContent:''])")
    check(f'{tag} part 7, AdamW: 0.10% for both, Llama 2’s settings, pressed alone, results on both levers, Next enabled', '0.10%' in o and 'Llama 2' in o and Z(lang, 'scenario 1', '应用场景 1') in o and lv[1][0] == 'true' and lv[0][0] == 'false' and '0.10%' in lv[1][1] and '0.010%' in lv[0][1] and not any(nexts(pg)), (o[:300], lv))
    check(f'{tag} part 7: four bars once AdamW is picked', pg.evaluate("()=>document.querySelectorAll('#op-wd .op-wrow').length") == 4)
    shot(pg, f'{tag}_p7', full=phone)
    rules(pg, f'{tag} part 7', lang)
    # part 8
    click(pg, '#op-rail [data-nav="next"]')
    t7 = text(pg, '#op-rail .step.current') or text(pg, '#op-rail')
    check(f'{tag} part 8: four points, three misconceptions, “See the scenarios” (R3.12)', pg.evaluate("()=>document.querySelectorAll('#op-rail .op-remlist li').length") == 4 and t7.count('✗') == 3 and vis(pg, '#op-rail [data-mode-go="sandbox"]'), t7[:200])
    _rl = text(pg, '#op-rail .step.current .op-remlist') + ' ' + text(pg, '#op-rail .step.current .obs')
    check(f'{tag} part 8 (G2, 2026-10-08): no point or misconception leans on a detail of an earlier part', not re.search(r'[Pp]art \d|第 \d 部分', _rl), _rl[:300])
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#op-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e)||shown(r) : shown(r)}""")
    check(f'{tag} part 8: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p8', full=phone)
    rules(pg, f'{tag} part 8', lang)
    click(pg, '#op-rail [data-mode-go="sandbox"]')
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    # the math, opened, in each mode
    pg.evaluate("()=>{document.getElementById('op-math').open=true}"); pg.wait_for_timeout(150)
    mt = pg.evaluate("()=>document.getElementById('op-mathbody').innerHTML")
    check(f'{tag} the math renders as MathML with no raw TeX', '<math' in mt and not re.search(r'tfrac|\\\\|hat<|<mi>iff', mt), mt[:200])
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w} {lang}'; phone = w <= 760
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    check(f'{tag} Scenarios: four, none pressed at the initial setup, Settings unlocked', pg.evaluate("()=>document.querySelectorAll('#op-rail [data-preset]').length") == 4
          and not pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-preset]')].some(b=>b.getAttribute('aria-pressed')==='true')") and vis(pg, '#op-ctl-lr-in') and not vis(pg, '#op-lock'))
    want = {0: ('Llama 2', 'Llama 2'), 1: ('He et al.', 'He 等'), 2: ('Kingma and Ba', 'Kingma 和 Ba'), 3: ('Rosenbrock', 'Rosenbrock')}
    for i in range(4):
        click(pg, f'#op-rail [data-preset="{i}"]')
        pr = pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#op-insight')
        check(f'{tag} scenario {i + 1}: pressed alone, the insight with its source', pr == ['true' if j == i else 'false' for j in range(4)] and Z(lang, *want[i]) in ins, (pr, ins[:200]))
        _st = state(pg)
        check(f'{tag} scenario {i + 1} (G6, 2026-10-08): the paths show at their end at once, no replay to wait for', not _st['playing'] and _st['K'] == pg.evaluate("()=>window.__op.kMax()"), (_st['playing'], _st['K']))
        shot(pg, f'{tag}_scen{i + 1}', full=False)
        rules(pg, f'{tag} scenario {i + 1}', lang)
    click(pg, '#op-rail [data-preset="0"]')
    check(f'{tag} scenario 1: AdamW’s settings visible: β2, bias correction, weight decay, warmup', all(vis(pg, f'#op-ctl-{k}-wrap') for k in ('b2', 'bc', 'wd', 'warm')) and not vis(pg, '#op-ctl-beta-wrap'))
    click(pg, '#op-ctl-opt [data-v="mom"]') if pg.query_selector('#op-ctl-opt [data-v="mom"]') else click(pg, '#op-controls .seg-ctl[data-key="opt"] button[data-v="mom"]')
    pr = pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} a change under Settings unpresses the scenario and says so (R1.8)', 'true' not in pr and text(pg, '#op-insight') != '', pr)
    click(pg, '#op-rail [data-preset="0"]')
    hsh = pg.evaluate('()=>location.hash')
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    st = state(pg2)
    check(f'{tag} the URL carries the setup: a fresh page opens Scenarios with Llama 2’s settings', st['mode'] == 'sandbox' and (st['opt'], st['lr'], st['b2'], st['wd'], st['warm']) == ('adamw', 0.5, 0.95, 0.1, 20), (hsh, st))
    pg2.close()
    # the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: SGD 0.01 to start, two settings in the card, Settings hidden with a note (R4.2)', (st['opt'], st['lr']) == ('sgd', 0.01) and pg.evaluate("()=>document.querySelectorAll('#op-rail .seg-ctl[data-key]').length") == 2
          and vis(pg, '#op-lock') and not any(vis(pg, f'#op-ctl-{k}-wrap') for k in ('opt', 'lr', 'beta', 'b2', 'bc', 'wd', 'warm', 'surf')), st)
    g = text(pg, '#op-ch-g')
    check(f'{tag} challenge: live progress for the initial setup', Z(lang, 'not in 200 steps', '200 步内到不了') in g, g)
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang)
    click(pg, '#op-rail [data-act="submit"]')
    res = text(pg, '#op-ch-res')
    check(f'{tag} challenge tier 3 (missed): not yet, a hint, “Show the answer”', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#op-rail [data-act="reveal"]')
          and text(pg, '#op-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:200])
    # the owner's review of batch 8 (2026-10-07): the answer keeps the 5 settings that pass and the initial setup (the pick, SGD at 0.01)
    click(pg, '#op-ch-res [data-act="reveal"]')
    rows = pg.evaluate(CHROWS)
    check(f'{tag} challenge “Show the answer”: 6 rows of 18, the 5 that pass and the initial setup (your pick); the note counts 12',
          len(rows) == 6 and sum(1 for r in rows if r[0]) == 5 and [r for r in rows if not r[0]] == [[False, True, True]]
          and text(pg, '#op-ch-res .cmp-note') == Z(lang, 'The other 12 options miss the goal and are not shown.', '其余 12 种没达到目标，图里没有列出。'), rows)
    check(f'{tag} challenge: the chart title names the steps and the goal, not “each optimizer and learning rate”', text(pg, '#op-ch-res .cmp-title') == Z(lang, 'Steps until the loss is below 0.001 on the steeper valley (goal: 100 or fewer)', '在更陡的山谷上，把 loss 降到 0.001 以下要几步（目标：不超过 100 步）'), text(pg, '#op-ch-res .cmp-title'))
    click(pg, '#op-ic-chO [data-v="adam"]'); click(pg, '#op-ic-chL [data-v="0.3"]')
    res = text(pg, '#op-ch-res')
    check(f'{tag} challenge tier 2 (two changes): met, but one change is enough', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'one change is enough', '改一处就够了') in res and '91' in res, res[:300])
    click(pg, '#op-ic-chO [data-v="mom"]'); click(pg, '#op-ic-chL [data-v="0.01"]')
    res = text(pg, '#op-ch-res')
    check(f'{tag} challenge tier 1: momentum 0.9 at 0.01, the smallest change', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'smallest change', '改动最少') in res and text(pg, '#op-status') == Z(lang, 'Goal met', '达成目标'), res[:300])
    rows = pg.evaluate(CHROWS)
    check(f'{tag} challenge: the 5 options that pass (one is the pick, momentum at 0.01) and the initial setup, 6 of 18; the note counts 12 (R5.4)',
          len(rows) == 6 and sum(1 for r in rows if r[0]) == 5 and sum(1 for r in rows if r[1]) == 1 and any(r[0] and r[1] for r in rows) and [r for r in rows if not r[0]] == [[False, False, True]]
          and text(pg, '#op-ch-res .cmp-note') == Z(lang, 'The other 12 options miss the goal and are not shown.', '其余 12 种没达到目标，图里没有列出。'), rows)
    check(f'{tag} challenge: baseline, reason and caveat (R5.5)', all(x in res for x in [Z(lang, 'initial setup', '初始设置'), Z(lang, 'Why momentum works', 'momentum 为什么行'), Z(lang, 'Note:', '注意：')]), res[-300:])
    check(f'{tag} challenge: the capability card', vis(pg, '#op-rail [data-exits]'))
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#op-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(120)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#op-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#op-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(120)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the result', (st['opt'], st['lr']) == ('mom', 0.01) and st['chSubmitted'] and vis(pg, '#op-ch-res .cmp-row'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('op-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent||'', read:(document.getElementById('dkRead')||{}).textContent||''}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, a short status and the three runs', dk['shown'] and len(dk['status']) <= 16 and '104' in dk['read'] and '74' in dk['read'] and 'Momentum' in dk['read'], dk)
        gap = pg.evaluate("()=>{const s=document.querySelector('#dkVis svg'), r=document.getElementById('dkRead'); if(!s||!r) return null; return s.getBoundingClientRect().right-r.getBoundingClientRect().left}")
        check(f'{tag} mini result bar: the small chart stays clear of the sentence', gap is not None and gap <= 1, gap)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()


TILE = '#tile-optimizers'
# the teaser area may use no data colour at all: a path is not a weight, a KV cache, an overhead, a failure or a success (R5.6)
TILE_COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const bad=['--c-weights','--c-kv','--c-oom','--c-overhead','--c-ok'].map(v=>toRGB(get(v))), out=[];
 for(const e of document.querySelectorAll('#tile-optimizers .tile-tz *')){if(!e.getClientRects().length) continue; const s=getComputedStyle(e);
   for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){if(p==='borderTopColor'&&s.borderTopWidth==='0px') continue; if(p==='color'&&!e.textContent.trim()) continue;
     if(bad.includes(s[p])) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}
 return out.slice(0,10);}"""
TILE_STATE = """()=>{const t=document.querySelector('#tile-optimizers'), R=t.getBoundingClientRect(), s=t.querySelector('svg.tz-op-vis'), sr=s.getBoundingClientRect(), vb=s.viewBox.baseVal;
 const tok=v=>{const d=document.createElement('i'); d.style.color=getComputedStyle(document.documentElement).getPropertyValue(v).trim(); document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const lh=e=>parseFloat(getComputedStyle(e).lineHeight)||parseFloat(getComputedStyle(e).fontSize)*1.4;
 const drawn=[...s.querySelectorAll('ellipse,polyline,circle,path')].map(e=>{const b=e.getBBox(), w=(parseFloat(getComputedStyle(e).strokeWidth)||0)/2;
   return {tag:e.tagName, cls:e.getAttribute('class'), ok:b.x-w>=-0.01&&b.y-w>=-0.01&&b.x+b.width+w<=vb.width+0.01&&b.y+b.height+w<=vb.height+0.01, box:[b.x-w,b.y-w,b.x+b.width+w,b.y+b.height+w].map(v=>Math.round(v*10)/10)};});
 const line=c=>{const e=s.querySelector('polyline.tz-op-path.'+c), cs=getComputedStyle(e); return {stroke:cs.stroke, dash:cs.strokeDasharray, n:e.getAttribute('points').trim().split(/\\s+/).length};};
 const swatch=c=>{const e=t.querySelector('.tz-leg line.tz-op-path.'+c), cs=getComputedStyle(e); return {stroke:cs.stroke, dash:cs.strokeDasharray};};
 const o=(t.querySelector('.seg-ctl button[aria-pressed="true"]')||{dataset:{}}).dataset.v;
 return {o, pressed:[...t.querySelectorAll('.seg-ctl button')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.v),
  read:t.querySelector('.tz-read').innerText.replace(/\\u00a0/g,' '), sent:o?window.__tzOp.read(o):'', help:o?window.__tzOp.help(o):'', tip:s.getAttribute('data-tipx'), aria:s.getAttribute('aria-label'),
  legend:[...t.querySelectorAll('.tz-leg > span')].map(x=>x.innerText.replace(/\\s+/g,' ').trim()), visH:sr.height, drawn, solid:line('p'), dashed:line('q'), swP:swatch('p'), swQ:swatch('q'),
  nSel:o?window.__tzOp.run(o).n+1:0, nOther:o?window.__tzOp.run(o==='sgd'?'adam':'sgd').n+1:0, ink:tok('--ink'), ink3:tok('--ink-3'),
  out:[...t.querySelectorAll('*')].filter(e=>e.getClientRects().length&&!e.closest('#tip')).filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&(r.left<R.left-1||r.right>R.right+1||r.top<R.top-1||r.bottom>R.bottom+1);}).map(e=>e.tagName+'.'+(e.getAttribute('class')||'')).slice(0,5),
  svgIn:sr.left>=R.left-0.5&&sr.right<=R.right+0.5, rdH:t.querySelector('.tz-op-rd').getBoundingClientRect().height, tileH:R.height, tileW:R.width,
  broken:[...t.querySelectorAll('.tz-leg > span, .tz-read b, .seg-ctl button')].filter(e=>{const tops=[], tw=document.createTreeWalker(e,NodeFilter.SHOW_TEXT); let n;
    while((n=tw.nextNode())){if(!n.textContent.trim()) continue; const r=document.createRange(); r.selectNodeContents(n); for(const x of r.getClientRects()) if(x.width>0.5) tops.push(x.top);}
    tops.sort((a,b)=>a-b); let lines=tops.length?1:0; for(let i=1;i<tops.length;i++) if(tops[i]-tops[i-1]>6) lines++; return lines>1;}).map(e=>e.textContent)}}"""
plain = lambda x: re.sub(r'<[^>]+>', '', x or '').replace('\u00a0', ' ').strip()
zh_units = lambda t: len(re.findall(r'[一-鿿]', t)) + len(re.findall(r'[A-Za-z0-9.%]+', t))   # CJK characters + Latin words and numbers
TILE_LEAK = r'0\.08|2 ÷ 25|0\.10?\b|−1\.5|diverg|发散|never|永远|100 y²|\(−5, 0\.8\)|β'


def zh_rules(t):
    """the Chinese writing rules on hidden text (notes, tooltips): one colon per sentence, spaces between Chinese and Latin, full-width punctuation"""
    bad = [f'two colons: {x[:60]}' for x in re.split(r'(?<=[。？！])', t) if x.count('：') >= 2]
    if re.search(r'[一-鿿][A-Za-z0-9]|[A-Za-z0-9%][一-鿿]', t): bad.append('spacing: ' + t[:60])
    if re.search(r'[一-鿿][:;,!?]|[:;,!?] ?[一-鿿]', t): bad.append('half-width punctuation: ' + t[:60])
    return bad


def tile_checks(b, w, h, lang, dark=False, tile_w=None):
    """the home tile: does it answer “What does Adam do that SGD does not?” for both picks, readably and inside its frame?"""
    theme = 'dark' if dark else 'light'
    tag = f'{w} {lang} {theme}{f" tile{tile_w}" if tile_w else ""} tile'
    pg = page(b, w, h, lang, hash_='map~roots', dark=dark)
    if not pg.query_selector(TILE):
        check(f'{tag}: the tile is on the home page under ML basics', False); pg.close(); return
    if tile_w:   # the narrowest and widest tile the contract allows (280–440 px)
        pg.evaluate(f"()=>{{const g=document.getElementById('tiles'); if(g) g.style.gridTemplateColumns='{tile_w}px'; const t=document.querySelector('{TILE}'); t.style.width='{tile_w}px'; t.style.maxWidth='{tile_w}px'; t.style.minWidth='{tile_w}px'}}"); pg.wait_for_timeout(150)
    pg.evaluate(f"()=>document.querySelector('{TILE}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(200)
    chip = pg.evaluate(f"()=>{{const c=document.querySelector('{TILE} .tz-kind'); return c?c.textContent:''}}")
    check(f'{tag}: the chip says Toy run, as the notes do', chip == Z(lang, 'Toy run', '玩具实跑') and Z(lang, 'Toy run', '玩具实跑') in ' '.join(pg.evaluate("()=>window.__tzOp.notes()")), chip)
    first = pg.evaluate(TILE_STATE)
    check(f'{tag}: the tile opens on Adam, its path solid and SGD’s dashed beside it (the comparison at a glance, R11.1)', first['pressed'] == ['adam'], first['pressed'])
    prev, heights, states = None, {}, {}
    for o in ('sgd', 'adam'):
        pg.click(f'{TILE} .seg-ctl button[data-v="{o}"]'); pg.wait_for_timeout(300)
        st = pg.evaluate(TILE_STATE); states[o] = st; rd = plain(st['read']); oth = 'adam' if o == 'sgd' else 'sgd'; name = {'sgd': 'SGD', 'adam': 'Adam'}
        check(f'{tag} {o}: the button is pressed, the sentence is the hook’s and changed', st['pressed'] == [o] and rd == plain(st['sent']) and rd != prev, (st['pressed'], rd, prev))
        prev = rd
        if lang == 'en':
            check(f'{tag} {o}: one sentence of at most 35 words (R11.5)', len(rd.split()) <= 35 and rd.endswith('.') and '. ' not in rd, (len(rd.split()), rd))
        else:
            check(f'{tag} {o}: one sentence, about as short as the English', zh_units(rd) <= 63 and rd.count('。') == 1 and rd.endswith('。'), (zh_units(rd), rd))
        want = {'sgd': Z(lang, ['learning rate 0.05', 'zig-zags', 'crawls', '104 steps'], ['学习率为 0.05', '来回横跳', '慢慢爬', '104 步']),
                'adam': Z(lang, ['its own step size', 'learning rate 0.5', 'neither zig-zags', 'nor crawls', '74 steps', 'SGD at 0.05 in 104'], ['各自的步长', '学习率为 0.5', '既不', '也不', '74 步', '0.05 的 SGD 要 104 步'])}[o]
        check(f'{tag} {o}: the sentence says what the path does, with the steps', all(k in rd for k in want), rd)
        check(f'{tag} {o}: the picture is at most 130 px tall', st['visH'] <= 130, st['visH'])
        bad = [d for d in st['drawn'] if not d['ok']]
        check(f'{tag} {o}: every ellipse, both paths, the ring and the × inside the picture’s frame', not bad and len(st['drawn']) == 4 + 2 + 2, bad[:3])
        check(f'{tag} {o}: nothing sticks out of the tile, no horizontal scroll, no legend item, button or bold number broken over two lines', not st['out'] and st['svgIn'] and overflow(pg) <= 0 and not st['broken'], (st['out'], st['svgIn'], overflow(pg), st['broken']))
        check(f'{tag} {o}: the solid path is {name[o]}’s whole run, the dashed one {name[oth]}’s', st['solid']['n'] == st['nSel'] and st['dashed']['n'] == st['nOther'] and st['nSel'] != st['nOther'], (st['solid']['n'], st['nSel'], st['dashed']['n'], st['nOther']))
        check(f'{tag} {o}: the legend names both line styles for this pick, the ring and the × (R5.7)', st['legend'] == Z(lang, [f'solid: {name[o]}’s path', f'dashed: {name[oth]}’s path', 'ring: start', '× bottom'], [f'实线：{name[o]} 的路径', f'虚线：{name[oth]} 的路径', '圆圈：出发点', '× 谷底']), st['legend'])
        check(f'{tag} {o}: solid is --ink with no dashes, dashed is --ink-3 with dashes, and the legend draws them the same way', st['solid']['stroke'] == st['ink'] and st['solid']['dash'] in ('none', '') and st['dashed']['stroke'] == st['ink3'] and st['dashed']['dash'] not in ('none', '') and st['swP'] == {'stroke': st['solid']['stroke'], 'dash': st['solid']['dash']} and st['swQ'] == {'stroke': st['dashed']['stroke'], 'dash': st['dashed']['dash']} and st['ink'] != st['ink3'], (st['solid'], st['dashed'], st['swP'], st['swQ']))
        check(f'{tag} {o}: the picture explains itself on hover and to a screen reader', st['tip'] == st['aria'] == st['help'] and name[o] in st['help'], st['help'])
        heights[o] = (round(st['rdH'], 1), round(st['tileH'], 1))
        pg.mouse.move(2, 2); pg.wait_for_timeout(250)
        nm = f'{"tile" + str(tile_w) if tile_w else w}_{lang}_{theme}_{o}.png'
        pg.locator(TILE).screenshot(path=str(ST / nm))
    check(f'{tag}: the tile keeps its height when the pick changes (review home-v4-A #23)', heights['sgd'] == heights['adam'], heights)
    hits = pg.evaluate(TILE_COLORS)
    check(f'{tag}: grayscale only in the tile’s mini experiment (R5.6)', not hits, hits)
    if tile_w:
        pg.close(); return
    notes = pg.evaluate(f"()=>[...document.querySelectorAll('{TILE} .tz-note')].map(e=>e.textContent)")
    check(f'{tag}: four notes, one short sentence each, in the page language (R11.5)', len(notes) == 4 and all(len(x) > 20 and (any('一' <= ch <= '鿿' for ch in x)) == (lang == 'zh') and len(re.findall(r'[.。](?:\s|$)', x)) == 1 for x in notes) and (lang == 'zh' or sum(len(x.split()) for x in notes) <= 80), notes)
    alltext = '\n'.join([pg.inner_text(TILE)] + notes + [states[o]['help'] for o in states] + [plain(states[o]['sent']) for o in states])
    leak = re.findall(TILE_LEAK, alltext)
    check(f'{tag}: no SGD at 0.1, no divergence, no limit, nothing of the challenge or of momentum’s β (R3.8)', not leak, leak)
    check(f'{tag}: no sentence says Adam is faster in general', not re.search(r'Adam (?:is|was) (?:always )?faster|Adam 更快|Adam 总是更快', alltext) and Z(lang, 'not always faster', '并不总是更快') in alltext, re.findall(r'.{30}faster.{10}', alltext))
    if lang == 'zh':
        bad = [x for t_ in notes + [states[o]['help'] for o in states] for x in zh_rules(t_)]
        check(f'{tag}: notes and tooltips follow the Chinese writing rules', not bad, bad)
    for h_ in pg.evaluate(RULES, TILE): rule_hits.append(f'{tag}: {h_}')
    if lang == 'zh':
        for h_ in pg.evaluate(COLONS, TILE): rule_hits.append(f'{tag}: two colons in one sentence: {h_}')
        for h_ in pg.evaluate(SPACING, TILE): rule_hits.append(f'{tag}: Chinese spacing: {h_}')
        for h_ in pg.evaluate(PUNCT, TILE): rule_hits.append(f'{tag}: half-width punctuation in Chinese: {h_}')
    # hover: the picture
    pg.hover(f'{TILE} svg.tz-op-vis'); pg.wait_for_timeout(350)
    tipt = pg.evaluate("()=>{const t=document.getElementById('tip'); return t&&!t.hidden?t.innerText:''}")
    check(f'{tag}: hovering the picture explains it', Z(lang, 'each ellipse is one loss level', '每个椭圆是一个 loss 高度') in tipt, tipt)
    check(f'{tag}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', pg.evaluate(f"()=>!document.querySelector('{TILE} .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('{TILE}').innerText)"))
    box = pg.evaluate(f"""()=>{{const a=document.querySelector('{TILE}').getBoundingClientRect(), t=document.getElementById('tip'), b=t&&!t.hidden?t.getBoundingClientRect():a;
      const x=Math.max(0,Math.min(a.left,b.left)-4), y=Math.max(0,Math.min(a.top,b.top)-4); return {{x, y, width:Math.min(innerWidth,Math.max(a.right,b.right)+4)-x, height:Math.min(innerHeight,Math.max(a.bottom,b.bottom)+4)-y}}}}""")
    pg.screenshot(path=str(ST / f'{w}_{lang}_{theme}_notes.png'), clip=box)   # the tile and the notes' tooltip together
    pg.mouse.move(2, 2); pg.wait_for_timeout(200)
    # the language switch keeps the pick
    pg.click(f'{TILE} .seg-ctl button[data-v="sgd"]'); pg.wait_for_timeout(250)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(500)
    st2 = pg.evaluate(TILE_STATE) if pg.query_selector(TILE) else None
    check(f'{tag}: the language switch keeps SGD picked', bool(st2) and st2['pressed'] == ['sgd'] and '104' in st2['read'] and (('步' in st2['read']) == (other == 'zh')), st2 and st2['read'])
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    pg.click(f'{TILE} .seg-ctl button[data-v="adam"]'); pg.wait_for_timeout(200)
    pg.close()


def run_reduced(b, w, h, lang):
    """With reduced motion the chart shows the end state at once, and Play still plays it (R3.9)."""
    tag = f'{w} {lang} reduced motion'
    pg = b.new_page(viewport={'width': w, 'height': h}, reduced_motion='reduce')
    pg.on('pageerror', lambda e: errs.append(f'{tag}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='optimizers'}"); pg.wait_for_timeout(600)
    st = state(pg)
    check(f'{tag}: no playback on its own, the end state (step 104) at once', not st['playing'] and st['K'] == 104 and text(pg, '#op-player [data-act="play"]') == Z(lang, '▶ Replay', '▶ 重放'), st)
    click(pg, '#op-player [data-act="play"]')
    check(f'{tag}: Play still plays', state(pg)['playing'] or state(pg)['K'] < 104, state(pg))
    check(f'{tag}: … and stops at the end', played(pg) and state(pg)['K'] == 104)
    pg.close()



# H1 and H2, the owner's site-wide decisions of 2026-10-09: the formulas in the hovers, the parts and Show the math are typeset as
# MathML, with no plain-text formula left in a hover.
def h_page(b, lang):
    pg = page(b, 1440, 900, lang)
    PLAIN = "()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>/θ ← θ − lr|m ÷ √v|½\\(x² \\+ 25|½ × wd × θ²/.test(e.getAttribute('data-tipx'))).length"
    check(f'H1 {lang} guided: no plain-text formula left in a hover', pg.evaluate(PLAIN) == 0)
    mode(pg, 'sandbox')
    pg.evaluate("()=>{const d=document.getElementById('op-math'); if(d&&!d.open) d.querySelector('summary').click()}"); pg.wait_for_timeout(300)
    hv = pg.evaluate("()=>{const mb=document.getElementById('op-mathbody'); return {math:mb.querySelectorAll('math').length, tex:/\\\\(frac|text|cdot|sqrt|hat)/.test(mb.textContent)}}")
    check(f'H2 {lang}: Show the math keeps its typeset formulas and shows no TeX source', hv['math'] >= 3 and not hv['tex'], hv)
    check(f'H1 {lang} sandbox: no plain-text formula left in a hover', pg.evaluate(PLAIN) == 0)
    pg.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    if not TILE_ONLY: run_reduced(b, SIZES[0][0], SIZES[0][1], LANGS[0])
    for (w, h) in SIZES:
        for lang in LANGS:
            for dark in (False, True):
                tile_checks(b, w, h, lang, dark=dark)
            if w > 760:
                for tw in (280, 440):
                    tile_checks(b, w, h, lang, tile_w=tw)
            if TILE_ONLY: continue
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    if not TILE_ONLY:
        for lang in LANGS:
            h_page(b, lang)
    b.close()
seen = set()
for hit in rule_hits + fit_hits:
    k = re.sub(r'^\d+ \w+ [^:]*: ', '', hit)
    if k in seen: continue
    seen.add(k); check(hit, False)
check('no page errors', not errs, errs[:5])
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'optimizers.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick with or without bias correction and see how many times lr the first step moves.', '选有或没有偏差校正，看看第 1 步走了 lr 的几倍。', 'Pick a scenario and see how the optimizer’s path reaches the bottom of the valley.', '选一个场景，看看优化器沿什么路径走到谷底。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('In part 3, above 0.08 the path bounced', 'Too large a rate makes the path bounce higher up the steep walls at every step until it diverges.'), ('第 3 部分里，学习率超过 0.08', '学习率太大，路径会在陡的谷壁之间越弹越高，最后发散。'), ('In part 4, β 0.6 cut', 'so β has to be tuned as well.'), ('第 4 部分里，β 0.6 把 SGD', '所以 β 也要调。'), ('In part 5, its first step moved x and y', 'so at the start a steep direction and a flat one move by the same amount, one learning rate.'), ('第 5 部分里，它第一步在 x 和 y 方向', '所以一开始，陡的方向和平的方向走得一样远，都是一个学习率。'), ('In part 7, with Adam + L2', 'AdamW shrinks every weight by the same fraction.'), ('第 7 部分里，用 Adam + L2 时', 'AdamW 让每个权重缩小同样的比例。'), ('the L2 of part 7', 'adds the decay to the gradient as L2, so Adam divides it too.'), ('也就是第 7 部分的 L2', '是把衰减当作 L2 加进梯度，它也会跟着被 Adam 除。'), ('like the too-large learning rate in this lab’s part 3', 'as if the learning rate were set far too high.'), ('就像这个 Lab 第 3 部分里过大的学习率', '相当于学习率设得太大。')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# H1, the owner's decision of 2026-10-09: no plain-text formula left in the lab's source (EN and ZH); they are typeset with A.ui.math.
for _bad in ['θ ← θ − lr × g', 'b ← β b + g', 'θ ← θ − lr × m ÷', 'The loss here is f(x, y) = ½(x² + 25 y²)', '里的 loss 是 f(x, y) = ½(x² + 25 y²)', 'adds ½ × wd × θ²', '（m ÷ √v）', 'm ÷ √v is then 10% ÷', 'm ÷ √v = 10% ÷', 'The limit is 2 ÷ (the steepest curvature)', '上限是 2 ÷（最陡的曲率）', 'The valley ½(x² + 25 y²), seen from above', '从上往下看山谷 ½(x² + 25 y²)', 'The narrow valley is ½(x² + 25 y²)', '窄山谷是 ½(x² + 25 y²)', 'A new valley, ½(x² + 100 y²)', '一条新山谷 ½(x² + 100 y²)', 'The loss is ½(x² + ${d.b} y²)', 'f = (1 − x)² + 10 (y − x²)², with its bottom']:
    check(f'H1: the plain-text formula “{_bad}” is gone (typeset instead)', _bad not in _G_SRC)
for _new in [r'\\theta \\leftarrow \\theta - \\mathrm{lr} \\cdot g', 'op-fx']:
    check(f'H1: “{_new[:40]}” is in the lab', _new in _G_SRC)
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
