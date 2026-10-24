"""Flow test for the eval-ci lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: the prediction gate with the kit's reason under a greyed-out Next (nextWhy), “Saved” after a pick; before the pick no error
  bar is drawn and the answer (±6.1) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the re-runs auto-play on entering (R3.9), pause stops them, Fast finishes in a few seconds, Next waits for the end; then the
  result with the count inside the bar, the verdict with the right answer, the formula on its own line;
- part 3: Next waits until A's bar is within ±1 (6,200 questions), then “about 6,100”; the three benchmark ticks under the slider; the
  status badge follows B − A against 0, not the overlap: “Cannot tell” at 164, “Gap beyond noise” at 5,000 while the bars overlap;
- part 4: Settings fixed, the two ways single choice with their result on the option, Next waits for “question by question”, then
  2.0 ± 1.7 from 0.3 to 3.7; the 2 × 2 sheet with its legend beside it;
- part 5: Next waits for 16 answers; the floor; part 6: four points with “show in the chart”, three misconceptions, See the scenarios;
- Scenarios: four, single choice, all question by question, the first pressed on entry, pressed state follows the settings, the URL
  carries the setup and a fresh page restores it;
- the challenge: Settings locked, the three feedback tiers with the power note on tiers 1 and 2 and its jump to “In production”,
  “Show the answer”; the brief chart (the owner's review of batch 8): only the 6 passing plans, the initial setup and the pick, both
  tagged, no other row, a note that counts the plans left out (9, or 8 when the pick is another plan that misses), a title without
  “each plan”; tier 2 says in its first line that a cheaper plan exists; the capability card; a language switch keeps the plan and
  the result;
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万), colours (R5.6).
Screenshots: shots/eval-ci/.
Run: python3 tests/eval-ci_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-ci.html', '--only', 'eval-ci'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'eval-ci'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'eval-ci.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'eval-ci'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
FULL = not ARGS
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));
   if(!e.closest('.mathbody, details.math')&&/精确值是|Exactly/.test(t)) out.push('a hover that only expands a number to its exact value (the owner, 2026-10-07 23:37): '+t.slice(0,90));});
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #evc-mathbody, .num-big, #evc-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|evc-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#evc-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#evc-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#evc-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, dark=False, hash_='eval-ci'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('evc-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 6 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} options miss the goal', f'其余 {hidden} 种没达到目标') in r['note']
            and Z(lang, 'each plan', '每种方案') not in r['title'])
state = lambda pg: pg.evaluate("()=>window.__evc.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#evc-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__evc||!window.__evc.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#evc-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#evc-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


def every(pg, where, lang, full_shot=None):
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#evc-app .labhead','#evc-rail','#evc-stage','#evc-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#evc-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#evc-rail [data-nav="{what}"]', 300)


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#evc-app .labtitle') == Z(lang, 'Is a 2-point benchmark gain real?', 'Benchmark 上高 2 分是真的吗？'), text(pg, '#evc-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#evc-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#evc-app .labmeta'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#evc-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#evc-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#evc-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: mission and prediction
    st = state(pg)
    check(f'{tag} part 1: 164 questions, A 80%, B 82%, guided part 1', (st['n'], st['pA'], st['pB'], st['mode'], st['step']) == (164, 0.8, 0.82, 'guided', 0), st)
    check(f'{tag} part 1: Next disabled, with the kit’s reason under it', nexts(pg) and all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。')}, (nexts(pg), why(pg)))
    check(f'{tag} part 1: no error bar drawn before the prediction (R3.8)', pg.evaluate("()=>document.querySelectorAll('#evc-stage .evc-bar').length") == 0)
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer (±6.1) appears nowhere before the prediction (R3.8)', '6.1' not in ba and '6.12' not in ba, [l for l in ba.split('\n') if '6.1' in l][:3])
    check(f'{tag} part 1: three options covering the range', pg.evaluate("()=>document.querySelectorAll('#evc-rail [data-pred]').length") == 3)
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#evc-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), ax=document.getElementById('evc-ax');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(ax); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, ax:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['ax'] is not None and fs['ax'] < fs['vh'] - 60, fs)
        shot(pg, f'{tag}_first_screen')
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#evc-rail [data-pred="one"]')
    check(f'{tag} part 1: Saved after a pick, Next enabled, the reason gone', Z(lang, 'Saved.', '已记下。') in text(pg, '#evc-rail') and not any(nexts(pg)) and why(pg) == [], (nexts(pg), why(pg)))
    click(pg, '#evc-rail [data-pred="three"]')
    check(f'{tag} part 1: the prediction can still change', state(pg)['pred'] == 'three')
    # ---- part 2: the re-runs auto-play
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: the re-runs start on their own (R3.9)', st['step'] == 1 and st['playing'], st)
    check(f'{tag} part 2: 200 re-runs are a whole run, so the default speed is Fast (R3.9)', st['spd'] == 'fast', st['spd'])
    check(f'{tag} part 2: Next waits for the re-runs, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Let the re-runs finish to continue (“Fast” takes a few seconds).', '等重考播完才能继续（选“快”只要几秒）。')}, why(pg))
    check(f'{tag} part 2: still no error bar while the pile grows (R3.8)', pg.evaluate("()=>document.querySelectorAll('#evc-ax .evc-bar').length") == 0)
    pg.wait_for_timeout(500)
    click(pg, '#evc-player [data-act="play"]', 100)
    m1 = state(pg)['pileM']; pg.wait_for_timeout(500); st = state(pg)
    check(f'{tag} part 2: Pause stops the re-runs', not st['playing'] and st['pileM'] == m1 and 0 < m1 < 200, (m1, st['pileM']))
    if w == 1440: shot(pg, f'{tag}_p2_paused')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 2: a language switch keeps the paused re-runs paused (R3.20)', not st['playing'] and st['pileM'] == m1 and st['step'] == 1, (st['playing'], st['pileM'], m1))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    click(pg, '#evc-player .seg-ctl[data-key="spd"] button[data-v="fast"]', 50)
    click(pg, '#evc-player [data-act="play"]', 50)
    t0 = time.time()
    try: pg.wait_for_function("()=>window.__evc.state().pileDone", timeout=8000)
    except Exception: pass
    dt = time.time() - t0
    check(f'{tag} part 2: Fast finishes the rest within 5 s', state(pg)['pileDone'] and dt < 5, dt)
    pg.wait_for_timeout(400); settle(pg)
    o = obs(pg)
    check(f'{tag} part 2: the result, the verdict with the right answer, and the formula', Z(lang, 'of the 200 re-runs land between 73.9% and 86.1%', '次落在 73.9% 到 86.1% 之间') in o
          and Z(lang, 'The right answer is “More than 3 points either way”', '正确答案是“上下 3 分以上”') in o and pg.evaluate("()=>document.querySelectorAll('#evc-rail [data-obs] .evc-fx math').length") == 2, o[:300])
    check(f'{tag} part 2: the bars are drawn now, labels at their ends', pg.evaluate("()=>document.querySelectorAll('#evc-ax .evc-bar').length") == 2 and '± 6.1' in text(pg, '#evc-ax'), text(pg, '#evc-ax'))
    check(f'{tag} part 2: the pile shows the count inside the bar', Z(lang, 'of 200 inside 80.0% ± 6.1', '次落在 80.0% ± 6.1 里') in text(pg, '#evc-pile'), text(pg, '#evc-pile'))
    check(f'{tag} part 2: Next enabled', not any(nexts(pg)))
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: more questions
    nav(pg, 'next')
    check(f'{tag} part 3: Next waits until ±1, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Drag until A’s error bar is within ±1 point to continue.', '把 A 的误差线拖到 ±1 分以内，才能继续。')}, why(pg))
    badge = lambda: pg.evaluate("()=>{const s=document.getElementById('evc-status'); return [s.textContent, s.dataset.tipx||'']}")
    bd = badge()
    check(f'{tag} part 3: at 164 the badge says “Cannot tell”, from B − A against 0 (B-1)', bd[0] == Z(lang, 'Cannot tell', '分不出') and Z(lang, 'crosses 0', '跨过了 0') in bd[1], bd)
    ticks = pg.evaluate("()=>[...document.querySelectorAll('#evc-ic-n .evc-ticks span')].map(s=>s.innerText.replace(/\\n/g,' '))")
    check(f'{tag} part 3: the slider’s ticks name three benchmarks', len(ticks) == 3 and 'HumanEval 164' in ticks[0] and 'GSM8K 1,319' in ticks[1] and 'MMLU' in ticks[2], ticks)
    NS = pg.evaluate("()=>window.__evc.consts.NS")
    slide(pg, '#evc-ic-n-in', NS.index(5000))
    check(f'{tag} part 3: 5,000 questions: ±1.1, Next still disabled', all(nexts(pg)) and '±1.1' in obs(pg), obs(pg)[:200])
    bd = badge()
    check(f'{tag} part 3: at 5,000 the bars overlap, yet the badge says “Gap beyond noise” and its tooltip says why (B-1)', bd[0] == Z(lang, 'Gap beyond noise', '超出了噪声')
          and Z(lang, 'still overlap', '还有重叠') in bd[1] and Z(lang, 'stays clear of 0', '不碰 0') in bd[1], bd)
    if w == 1440: shot(pg, f'{tag}_p3_n5000')
    slide(pg, '#evc-ic-n-in', NS.index(6200))
    o = obs(pg)
    check(f'{tag} part 3: 6,200 questions: within ±1, about 6,100, Next enabled', not any(nexts(pg)) and Z(lang, 'about 6,100 questions', '大约要 6,100 道题') in o and '±1.0' in o, o[:300])
    if phone: check(f'{tag} part 3: the phone’s slider sits next to the chart', vis(pg, '#evc-ph-n'))
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4: question by question
    nav(pg, 'next')
    check(f'{tag} part 4: Next waits for question by question, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick “Question by question” to continue.', '选“逐题对比”才能继续。')}, why(pg))
    check(f'{tag} part 4: Settings fixed, with a note', vis(pg, '#evc-lock') and not vis(pg, '#evc-ctl-n-wrap'), text(pg, '#evc-lock'))
    check(f'{tag} part 4: the answer sheet and its legend', pg.evaluate("()=>document.querySelectorAll('#evc-grid .evc-cell').length") == 4 and vis(pg, '#evc-gleg'))
    o = obs(pg)
    check(f'{tag} part 4: each on its own crosses 0', Z(lang, 'B − A = 2.0 ± 3.0 points, from −1.0 to 5.0', 'B − A = 2.0 ± 3.0 分，从 −1.0 到 5.0') in o, o[:200])
    click(pg, '#evc-rail [data-lever="pair"]')
    o = obs(pg)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    res = pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 4: single choice, both results on their options', pressed == ['false', 'true'] and all(res), (pressed, res))
    check(f'{tag} part 4: question by question 2.0 ± 1.7, from 0.3 to 3.7, Next enabled', Z(lang, 'B − A = 2.0 ± 1.7 points, from 0.3 to 3.7', 'B − A = 2.0 ± 1.7 分，从 0.3 到 3.7') in o and not any(nexts(pg)), o[:200])
    check(f'{tag} part 4: the difference axis marks 0 and draws both ways', pg.evaluate("()=>document.querySelectorAll('#evc-df .evc-drow').length") == 2 and vis(pg, '#evc-df .evc-zero'))
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('evc-df'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 4: the mini result bar with a short status and the result', dk['shown'] and dk['status'] == Z(lang, 'Beyond noise', '超出噪声') and '2.0 ± 1.7' in dk['read'], dk)
        shot(pg, f'{tag}_p4_dock')
        rules(pg, f'{tag} part 4 dock', lang)
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: answers per question
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for 16 answers, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Set 16 answers per question to continue.', '把回答次数调到 16，才能继续。')}, why(pg))
    click(pg, '#evc-ic-k .seg-ctl button[data-v="4"]')
    check(f'{tag} part 5: 4 answers: ±5.4, Next still disabled', '±5.4' in obs(pg) and all(nexts(pg)), obs(pg)[:200])
    click(pg, '#evc-ic-k .seg-ctl button[data-v="16"]')
    o = obs(pg)
    check(f'{tag} part 5: 16 answers: ±5.1, never below the floor, Next enabled', '±5.1' in o and Z(lang, 'never goes below ±5.1 points', '也低不过 ±5.1 分') in o and not any(nexts(pg)), o[:300])
    check(f'{tag} part 5: the chart has five columns and the floor', pg.evaluate("()=>document.querySelectorAll('#evc-kc .evc-kcol').length") == 5 and vis(pg, '#evc-kc .evc-floor'))
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: what to remember
    nav(pg, 'next')
    rt = text(pg, '#evc-rail')
    check(f'{tag} part 6: four points, three misconceptions, See the scenarios, the next-lab footer', pg.evaluate("()=>document.querySelectorAll('#evc-rail .evc-remlist li').length") == 4
          and rt.count('✗') == 3 and pg.evaluate("()=>!!document.querySelector('#evc-rail [data-mode-go=\"sandbox\"]')") and pg.evaluate("()=>!!document.querySelector('#evc-rail .nextlab, #evc-rail [class*=next]')"), rt[-300:])
    for k in ('size', 'pair', 'k', 'pile'):
        click(pg, f'#evc-rail [data-rem="{k}"]')
        st = state(pg)
        check(f'{tag} part 6: “show in the chart” {k}', st['remView'] == k and pg.evaluate(f"()=>document.querySelector('#evc-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true', st)
    check(f'{tag} part 6: the pile is complete in the chart', pg.evaluate("()=>document.querySelectorAll('#evc-pile .evc-dots i').length") == 200)
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # skip on part 1 from a fresh page
    pg.close()
    pg = page(b, w, h, lang, dark)
    nav(pg, 'skip')
    check(f'{tag} part 1: Skip goes on without a prediction', state(pg)['step'] == 1 and state(pg)['pred'] is None)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    phone = w <= 760
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#evc-rail')
    _lead = Z(lang, 'Pick a scenario and see whether the error bar of B − A crosses 0.', '选一个场景，看看 B − A 的误差线会不会跨过 0。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    check(f'{tag} scenarios: four, the first pressed on entry, question by question (B-3)', pg.evaluate("()=>document.querySelectorAll('#evc-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#evc-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true' and state(pg)['paired'] == 1, state(pg))
    for i in range(4):
        click(pg, f'#evc-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#evc-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result, question by question', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80 and state(pg)['paired'] == 1, (pressed, ins[:120]))
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (1, 3) else None)
    ins = text(pg, '#evc-insight')
    check(f'{tag} scenario temperature: 4 answers ±1.9, 1 answer ±2.2, floor ±1.8, question by question 2.0 ± 1.1', all(x in ins for x in ('±1.9', '±2.2', '±1.8', '2.0 ± 1.1')), ins[:240])
    check(f'{tag} scenario temperature: the badge follows question by question (B-3)', pg.evaluate("()=>document.getElementById('evc-status').textContent") == Z(lang, 'Gap beyond noise', '超出了噪声'))
    click(pg, '#evc-rail [data-preset="1"]')
    ins = text(pg, '#evc-insight')
    check(f'{tag} scenario exam: 0.8 ± 0.5 question by question; ± 0.9 each on its own', '0.8 ± 0.5' in ins and '0.8 ± 0.9' in ins, ins[:240])
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#eval-ci~'), hsh)
    # a change under Settings un-presses the scenario
    slide(pg, '#evc-ctl-pB-in', 830)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#evc-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change in Settings un-presses the scenario', pressed.count('true') == 0 and Z(lang, 'match none of these scenarios', '和这几个场景都不一样') in text(pg, '#evc-insight'), pressed)
    every(pg, f'{tag} sandbox settings', lang)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the exam pressed', st['mode'] == 'sandbox' and (st['n'], st['pB'], st['paired']) == (14000, 0.808, 1)
          and pg2.evaluate("()=>document.querySelector('#evc-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at 200 questions, each on its own, 1 answer', (st['n'], st['paired'], st['k']) == (200, 0, 1), st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('evc-controls').classList.contains('locked')") and vis(pg, '#evc-lock'))
    check(f'{tag} challenge: plain statement and the assumptions', Z(lang, 'whether the new model B can replace A', '新模型 B 能不能替换线上的 A') in text(pg, '#evc-rail') and pg.evaluate("()=>document.querySelectorAll('#evc-rail .evc-asm li').length") == 3)
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#evc-rail [data-act="submit"]')
    r = text(pg, '#evc-ch-res')
    check(f'{tag} challenge tier 3: not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in r and Z(lang, 'Hint', '提示') in r and vis(pg, '#evc-ch-res [data-act="reveal"]')
          and text(pg, '#evc-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), r[:200])
    click(pg, '#evc-ch-res [data-act="reveal"]')
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 7 rows, the 6 passing plans and the initial setup (the pick), and counts the 9 left out', brief_ok(lang, bf, 7, 9), bf)
    click(pg, '#evc-rail .seg-ctl[data-key="chn"] button[data-v="4000"]')
    r = text(pg, '#evc-ch-res')
    check(f'{tag} challenge tier 2 (4,000, each on its own, 1 answer): goal met, but a cheaper plan', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'a cheaper plan also works', '还有更省的方案') in r and '4,000' in r, r[:300])
    check(f'{tag} challenge tier 2: the power note (about 63%)', Z(lang, 'only about 63% of the time', '只有约 63%') in r, r[:600])
    check(f'{tag} challenge tier 2: the first line says that a cheaper plan exists', Z(lang, 'a cheaper plan also works', '还有更省的方案') in text(pg, '#evc-ch-res > p:first-child'), text(pg, '#evc-ch-res > p:first-child'))
    bf = brief(pg)
    check(f'{tag} challenge: when the goal is met, the chart holds the 6 passing plans (the pick among them) and the initial setup, 9 left out', brief_ok(lang, bf, 7, 9), bf)
    click(pg, '#evc-rail .seg-ctl[data-key="chn"] button[data-v="1000"]')
    click(pg, '#evc-rail .seg-ctl[data-key="chm"] button[data-v="1"]')
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (1,000, question by question, 1 answer) adds its own row, 8 rows, 8 left out', brief_ok(lang, bf, 8, 8), bf)
    click(pg, '#evc-rail .seg-ctl[data-key="chk"] button[data-v="4"]')
    r = text(pg, '#evc-ch-res')
    check(f'{tag} challenge tier 2 (1,000, question by question, 4 answers): a cheaper plan', Z(lang, 'a cheaper plan also works', '还有更省的方案') in r, r[:300])
    click(pg, '#evc-rail .seg-ctl[data-key="chn"] button[data-v="500"]')
    r = text(pg, '#evc-ch-res')
    check(f'{tag} challenge tier 1 (500, question by question, 4 answers): goal met, and the fewest', Z(lang, 'no plan that meets the goal needs fewer', '这已经是最少的') in r and '1.9' in r, r[:300])
    check(f'{tag} challenge tier 1: the power note (about 56%, ±1.4 for 80%) with a jump to “In production” (S-1)', Z(lang, 'only about 56% of the time', '只有约 56%') in r and '±1.4' in r
          and pg.evaluate("()=>!!document.querySelector('#evc-ch-res [data-jump=\"prod\"]')"), r[:600])
    rows = pg.evaluate("()=>{const c=document.querySelector('#evc-ch-res .cmp'); if(!c) return null; const lab=[...c.querySelectorAll('*')].filter(e=>/^✓ /.test(e.textContent)&&e.children.length===0); return {ok:lab.length, all:c.innerText.split('\\n').filter(l=>/(questions|道题)/.test(l)).length}}")
    bf = brief(pg)
    check(f'{tag} challenge: 7 rows, the 6 passing plans with a ✓ and the initial setup, no other plan', rows and rows['ok'] == 6 and rows['all'] == 7 and brief_ok(lang, bf, 7, 9), (rows, bf))
    r = text(pg, '#evc-ch-res')
    check(f'{tag} challenge: the two single changes of the cheapest plan are named in words, their error bars in a hover', Z(lang, 'changing only one of the other two choices misses the goal', '比较方法和回答次数只改一项，都达不到目标') in r
          and pg.evaluate("()=>[...document.querySelectorAll('#evc-ch-res .def')].map(e=>e.getAttribute('data-tipx')).filter(t=>/±4\\.2|±3\\.0/.test(t)).length") == 2, r[-700:])
    tags = text(pg, '#evc-ch-res .cmp')
    check(f'{tag} challenge: the initial setup and the pick tagged', Z(lang, 'initial setup', '初始设置') in tags and Z(lang, 'your pick', '你的选择') in tags)
    check(f'{tag} challenge: the capability card', vis(pg, '#evc-rail [data-exits]'))
    click(pg, '#evc-ch-res [data-jump="prod"]', 700)
    check(f'{tag} challenge: the jump reaches “In production”', pg.evaluate("()=>{const r=document.getElementById('evc-prod-card').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>0}"))
    pg.evaluate("()=>document.getElementById('evc-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['n'], st['paired'], st['k']) == (500, 1, 4) and st['chSubmitted'] and vis(pg, '#evc-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # back to guided keeps the learner's place
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    if FULL:
        run_guided(b, 1440, 900, 'en', dark=True)
    b.close()
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'eval-ci', 'evc'
_H1_PLAIN = ['√', 'p(1 − p)', '÷']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['p<sub>i</sub>', '√(0.8 × 0.2 ÷', 'SE = √((', '= 3.8416 × 0.16 ÷ 0.01²', 'p ± 1.96 × √(p(1 − p) ÷ n)', '除以 √n']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['class="evc-mb"', 'evc-eqs', '公式在 Lab 里']       # sentences or markup written on 2026-10-09
_H1_PROBE = """([pfx,plain])=>{const app=document.querySelector('#view-dyn:not([hidden])')||document; const mb=document.getElementById(pfx+'-mathbody');
  const out={nmath:0,words:[],plain:[],blocks:0,sep:true,clip:[],page:[],wide:[]};
  const TEXW=/^(frac|sqrt|left|right|text|hat|cdot|times|sum|quad|ldots|approx|mathrm|big|Big)$/;
  const scanText=(root,arr)=>{const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n; while((n=tw.nextNode())){const e=n.parentElement;
    if(!e||e.closest('math,code,pre,svg,script,style,#tip')||!e.getClientRects().length) continue; for(const p of plain) if(n.textContent.includes(p)) arr.push(p+' in “'+n.textContent.trim().slice(0,70)+'”');}};
  if(mb){const d=mb.closest('details'); if(d) d.open=true;
    out.nmath=mb.querySelectorAll('math').length;
    mb.querySelectorAll('mi,mtext').forEach(x=>{if(x.tagName.toLowerCase()==='mi'&&TEXW.test(x.textContent.trim())) out.words.push(x.textContent);});
    scanText(mb,out.plain);
    const bl=[...mb.querySelectorAll('.'+pfx+'-mb')]; out.blocks=bl.length;
    bl.slice(1).forEach(b=>{const cs=getComputedStyle(b); if(!(parseFloat(cs.borderTopWidth)>=1&&cs.borderTopStyle==='solid')) out.sep=false;});
    mb.querySelectorAll('.'+pfx+'-eqs').forEach(q=>{if(q.scrollHeight>q.clientHeight+1) out.clip.push(q.textContent.slice(0,40)); if(q.scrollWidth>q.clientWidth+1) out.wide.push(q.scrollWidth+' > '+q.clientWidth+': '+q.textContent.slice(0,30));
      const k=[...q.children]; for(let i=1;i<k.length;i+=3){const a=k[i].getBoundingClientRect(), b=k[i+3]&&k[i+3].getBoundingClientRect(); if(b&&Math.abs(a.left-b.left)>1) out.clip.push('“=” not in one column: '+q.textContent.slice(0,40));}});}
  scanText(app,out.page);
  app.querySelectorAll('mi').forEach(x=>{if(TEXW.test(x.textContent.trim())) out.words.push(x.textContent);});
  return out;}"""
# H5 (the owner, 2026-10-09): the stage's blocks, each with a 1px hairline (var(--line)) on top when another of them sits above it, none on the first or a lone one
_H5_SEP = ['#evc-axbox', '#evc-pairbox', '#evc-kbox', '#evc-dfbox', '#evc-ax', '#evc-pilebox', '#evc-mix', '#evc-kc-cap']
_H5_PROBE = """(sels)=>{const els=[...new Set(sels.flatMap(s=>[...document.querySelectorAll('#view-dyn '+s)]))].filter(e=>e.getClientRects().length&&e.offsetHeight>0);
  const out={bad:[],lines:0}; for(const e of els){const r=e.getBoundingClientRect(), above=els.some(o=>o!==e&&o.parentElement===e.parentElement&&o.getBoundingClientRect().bottom<=r.top+1);
    const cs=getComputedStyle(e), line=parseFloat(cs.borderTopWidth)>=1&&cs.borderTopStyle!=='none'; if(line) out.lines++;
    if(line!==above) out.bad.push((e.id||e.className)+(above?': no hairline under the block above':': a hairline with no block above'));} return out;}"""
def _h1_states(b, w, h, lang):
    pg = page(b, w, h, lang)
    rail = '#view-dyn aside.rail'
    nparts = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", rail)
    got = []
    for k in range(nparts):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", [rail, k]); pg.wait_for_timeout(2600)
        try: pg.evaluate("()=>{for(const b of document.querySelectorAll('#view-dyn [data-speed=\"fast\"], #view-dyn .seg-ctl[data-key=\"speed\"] button[data-v=\"fast\"]')) b.click();}")
        except Exception: pass
        pg.wait_for_timeout(1400)
        got.append((f'part {k + 1}', {**pg.evaluate(_H1_PROBE, [_H1_PFX, _H1_PLAIN]), 'h5': pg.evaluate(_H5_PROBE, _H5_SEP)}))
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(3500)
        got.append((m, {**pg.evaluate(_H1_PROBE, [_H1_PFX, _H1_PLAIN]), 'h5': pg.evaluate(_H5_PROBE, _H5_SEP)}))
    dr = pg.evaluate("p=>{const b=document.querySelector('#view-dyn [data-act=\"drill-reveal\"]'); if(!b) return null; b.click(); return true}", _H1_PFX); pg.wait_for_timeout(300)
    drill = pg.evaluate("p=>{const a=document.getElementById(p+'-drill-a'); if(!a||a.hidden) return null; return {n:a.querySelectorAll('math').length, t:[...a.querySelectorAll('p')].map(x=>{const c=x.cloneNode(true); c.querySelectorAll('math').forEach(m=>m.remove()); return c.textContent}).join(' ')}}", _H1_PFX) if dr else None
    pg.close()
    return got, drill
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (390, 844, 'zh')]:
        if str(_w) not in str(SIZES) or _lang not in LANGS: continue
        _got, _drill = _h1_states(_b, _w, _h, _lang)
        _with = [(s, o) for s, o in _got if o['nmath']]
        check(f'H1 {_w} {_lang}: Show the math has typeset formulas (MathML) in {len(_with)} of {len(_got)} states', len(_with) >= 2, [s for s, o in _got])
        for s, o in _got:
            for x in o.get('wide', []): print(f'INFO H2 {_w} {_lang} {s}: a chain wider than its box, it scrolls sideways: {x}')
            check(f'H1 {_w} {_lang} {s}: no TeX command shown as a word', not o['words'], o['words'][:5])
            check(f'H1 {_w} {_lang} {s}: no plain-text formula left in Show the math', not o['plain'], o['plain'][:4])
            check(f'H1 {_w} {_lang} {s}: no plain-text formula left on the page', not o['page'], o['page'][:4])
            check(f'H2 {_w} {_lang} {s}: a hairline between the blocks of Show the math ({o["blocks"]} blocks)', o['sep'] and (o['blocks'] >= 1 or not o['nmath']), o)
            check(f'H2/R6.33 {_w} {_lang} {s}: no formula box clipped, every “=” of a chain in one column', not o['clip'], o['clip'][:3])
            check(f'H5 {_w} {_lang} {s}: a hairline between the stage’s blocks, none above the first ({o["h5"]["lines"]} hairlines)', not o['h5']['bad'], o['h5']['bad'][:3])
        check(f'H5 {_w} {_lang}: the stage separates its blocks with a hairline in {sum(1 for s, o in _got if o["h5"]["lines"])} of {len(_got)} states', any(o['h5']['lines'] for s, o in _got), [(s, o['h5']) for s, o in _got])
        if _drill is not None:
            check(f'H1 {_w} {_lang}: the drill answer typesets its working', _drill['n'] >= 1 and not any(x in _drill['t'] for x in _H1_PLAIN), _drill)
    _b.close()
import re as _re
_h1src = _re.sub(r'/\*.*?\*/', '', (P / 'labs' / f'{_H1_LAB}.js').read_text() + (P / 'teasers' / f'{_H1_LAB}.js').read_text(), flags=_re.S)
for _x in _H1_SRC_GONE:
    check(f'H1 src: the plain-text formula “{_x}” is gone', _x not in _h1src)
check('H5 src: the stage hairline rule is in the lab’s CSS', 'H5 (the owner, 2026-10-09)' in (P / 'labs' / f'{_H1_LAB}.css').read_text() and '1px solid var(--line)' in (P / 'labs' / f'{_H1_LAB}.css').read_text())
for _x in _H1_SRC_NEW:
    check(f'H1/H2 src: present: {_x[:70]}', _x in _h1src)
# end H1 H2
# polish round 3 (2026-10-16, B4 / R7.12, the owner: "the panel was pushed down, I did not see there was more below"): at 1440 × 900,
# entering each listed guided part the way a reader does (Next), the stage needs no scrolling inside itself and nothing in it
# scrolls either; in Chinese it could still grow by 30 px (its max-height minus the height its content and borders need), in
# English it does not scroll; its main picture lies inside it and on the screen. The real layout: window.__aspPrimer, as the probe.
_FIT_PARTS = [4]
_FIT_V = "(()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); return [...v.querySelectorAll('.bench')].find(x=>x.offsetParent)||v;})()"
# tb: the top bar's bottom where it is on the screen now (on a desktop lab page the bar scrolls away, refit step 2; on a phone it stays on top).
_FIT_M = ("(()=>{const b=" + _FIT_V + ", s=b.querySelector(':scope>.stage'), cur=b.querySelector('.steps>.step.current'), r=s.getBoundingClientRect(), tb=(t=>t?Math.max(0,Math.round(t.getBoundingClientRect().bottom)):0)(document.querySelector('.topbar')); let inner=0;"
          "s.querySelectorAll('*').forEach(e=>{const c=getComputedStyle(e); if(/(auto|scroll)/.test(c.overflowY)&&e.getClientRects().length) inner=Math.max(inner,e.scrollHeight-e.clientHeight);});"
          "const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((x,y)=>y.width*y.height-x.width*x.height)[0];"
          "return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, inner,"
          "fig:!f||(f.top>=r.top-1&&f.bottom<=r.bottom+1&&f.top>=tb-2&&f.bottom<=innerHeight+2)};})()")
if '1440' in str(SIZES):
    with sync_playwright() as _fp:
        _fb = _fp.chromium.launch()
        for _lang in LANGS:
            _pg = _fb.new_page(viewport={'width': 1440, 'height': 900}); _pg.add_init_script('window.__aspPrimer=true')
            _pg.on('pageerror', lambda e, _l=_lang: errs.append(f'fit 1440 {_l}: {e}'))
            _pg.goto(URL); _pg.wait_for_timeout(300); _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(80)
            _pg.evaluate("()=>{location.hash='eval-ci'}"); _pg.wait_for_timeout(500)
            _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
            if _g.count(): _g.first.click(); _pg.wait_for_timeout(700)
            for _i in range(1, max(_FIT_PARTS) + 1):
                if _i > 1:
                    _nb = _pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
                    if _nb.count(): _nb.first.click()
                    else: _pg.evaluate(f"{_FIT_V}.querySelector('.steps>.step:nth-child({_i}) .step-h').click()")
                    _pg.wait_for_timeout(900)
                if _i not in _FIT_PARTS: continue
                _f = _pg.evaluate(_FIT_M)
                if _lang == 'zh':
                    check(f'B4 1440 zh part {_i}: the stage fits without scrolling inside, nothing in it scrolls, 30 px to spare, its picture inside it and on screen (R7.12)',
                          _f['part'] == _i and _f['scroll'] <= 0 and _f['inner'] <= 4 and _f['spare'] >= 30 and _f['fig'], _f)
                else:
                    check(f'B4 1440 en part {_i}: the stage fits without scrolling inside, its picture inside it and on screen (R7.12)',
                          _f['part'] == _i and _f['scroll'] <= 0 and _f['spare'] >= 0 and _f['fig'], _f)
            _pg.close()
        _fb.close()
# end B4
check('no page errors', not errs, errs[:5])
# the owner, 2026-10-08 (G1, G2) src: every option group leads in with what to pick and what to watch (G1); no sentence leans on an earlier part's detail (G2)
_src = (P / 'labs' / 'eval-ci.js').read_text() + (P / 'teasers' / 'eval-ci.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['There are two ways to work out the error bar of B − A. Pick one and see whether the error bar crosses 0.', 'B − A 的误差线有两种算法。选一种，看看误差线会不会跨过 0。', 'With a few hundred questions, the error bar spans several points, so a lead of a few points can be pure luck.', '只有一两百道题时，误差线有好几分长，领先几分很可能只是运气。换模型之前，先算出误差线。', '想看清的差距每小一半，题目就得多 4 倍。所以先定下必须看清的最小差距，再按它准备题目。', '两个都对、两个都错的题，相减时抵消了。所以 B − A 的误差线短得多，同一批题能看清更小的差距。', '多答几次只消掉每次回答的运气。抽到哪些题带来的摆动还在，所以想让误差线明显变短，只能加题。', '各算各的两条误差线，没有用上两个模型做的是同一批题。', '取平均只消掉每次回答的运气。抽到哪些题这件事没有变，所以误差线只短一点。', '取平均不会增加摆动：每道题答的次数越多，误差线越短，只是越来越接近一个下限。', '它会变，只是变得少：多答几次能让误差线短一点，可抽到哪些题带来的摆动还在。', '题数只翻一倍的话，误差线只短三成左右。', '在图里显示 164 道题的重考和 A 的误差线。', '在图里显示两个模型做 1,319 道题的答卷，和 B − A 的两种误差线。', '在图里显示每道题答 16 次时 A 的误差线和它的下限。', '和这个 Lab 算一个分数误差线的公式几乎一样。', '再把误差线的公式倒过来，解出题数。', '所以公式可以化简成下面这样。']:  # 2026-10-09 (H2): the sentence now leads a Show-the-math block; still no part number
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['第 2 部分里，164 道题时', '第 3 部分里，要把 A 的误差线', '第 4 部分里，同样差 2 分', '第 5 部分里，每道题答 16 次', '第 4 部分里，A 和 B 的误差线是重叠的', '第 5 部分里，从答 1 次到答 16 次', '第 3 部分说的就是这件事', '和第 2 部分的公式几乎一样', '也就是第 4 部分里那 10%', '再像第 3 部分那样解出题数', '在图里显示第 2 部分', '在图里显示第 4 部分', '在图里显示第 5 部分', 'In part 2, with 164 problems', 'In part 3, bringing', 'In part 4, for the same 2 points', 'In part 5, 16 answers', 'In part 4, A’s and B’s', 'in part 5 the error bar shrank', 'That is what part 3 showed', 'the formula of part 2', 'the 10% of part 4', 'as part 3 does', 'Shows part 2’s', 'Shows part 4’s', 'Shows part 5 at', '第 5 部分的公式可以化简成', 'the formula of part 5 simplifies']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
