"""Flow test for the tensors lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run and three more widths:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the chart's bars;
- part 1: the three ways at the mission's sizes (2 KiB and 500M runs, 954 GiB in red, 1.86 GiB), GiB defined at its first use;
- the prediction gate (R3.8): Next disabled until a prediction; before it the result reads "?", and "gives (3, 4)" / "Result (3, 4)"
  appear nowhere, in text or tooltips; "Saved …" after a pick (R2.5);
- part 3: the result, the verdict with the right answer, and why (R3.10); ghost copies; Next waits for a pair that does not broadcast,
  and the error names the axis; part 4: einsum single choice, the matrix example's numbers, Next waits for the attention scores (d
  summed, 17.2B multiply-adds); part 5: Next waits for a change of D, ▲ only after it (R6.6), the matmul bar stays 1.86 GiB; part 6:
  three misconceptions with "Show in the chart", "See the scenarios", the next-lab footer (R3.11, R3.12);
- Settings: only the part's group in the guided parts, locked in the challenge (R4.2), a change under Settings does not move the page;
- Scenarios: four, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page restores it;
- the challenge: live progress, the three feedback tiers (R4.4), "Show the answer", a brief chart (the owner's review of batch 8): the
  2 passing ways, the initial setup and the pick, nothing else, and a note that counts the ways left out, with the goal (R5.4), the
  capability card; the language switch keeps the answer;
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
- the mini result bar at 390 once the chart scrolls away, with a short status;
- B4 (polish round 5): entering part 5 at 1440 × 900, the stage needs no scrolling inside (Chinese with 30 px to spare);
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万, no "toy", no 控件 /
  Controls, no hover instructions), colours only for their data (R5.6), no text wider than its box (R7.8), a tooltip for every help
  cursor (R1.1).
Screenshots: shots/tensors/.
Run: python3 tests/tensors_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tensors-flow.html', '--only', 'tensors'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'tensors-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tensors-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tensors-flow-wrapped.html')
SH = P / 'shots' / 'tensors'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='tensors'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ts.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ts-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__ts||!window.__ts.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#ts-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#ts-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#ts-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#ts-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #ts-mathbody, .num-big, #ts-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for an error, an array that does not
# fit, or a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|ts-over|warn|wrong|ex-res|\\bbad\\b|ts-err|\\bover\\b|ts-sw-bad|ts-sw-over/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ts-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


def rules(pg, where, lang):
    for sel in ('#ts-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#ts-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#ts-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")


# round 1: everything a learner can read before the prediction, outside Connect: the header, its popovers (even closed), the card,
# the stage, Settings, their tooltips, and every term tooltip in the card (A3, R3.8)
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#ts-app .labhead','#ts-rail','#ts-stage','#ts-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#ts-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#ts-rail .t').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#ts-rail .t')[{i}]; e.focus({{preventScroll:true}})}}"); pg.wait_for_timeout(60)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out)


# R7.8: no text wider than its box (clipped or ellipsised), in the lab and the mini result bar
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ts-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# R1.1: every element with a help cursor explains itself on hover (itself or an ancestor carries the tooltip)
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ts-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
fit_hits = []


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    if pg.evaluate("()=>/\\b1 runs\\b|\\b1 axes\\b|\\b1 numbers\\b/.test(document.getElementById('ts-app').innerText+[...document.querySelectorAll('#ts-app [data-tipx]')].map(e=>e.dataset.tipx).join(' '))"):
        fit_hits.append(f'{where}: singular and plural')


cells = lambda pg, sel: pg.evaluate(f"()=>[...document.querySelectorAll('{sel}')].filter(e=>e.getClientRects().length).map(e=>e.textContent)")
count = lambda pg, sel: pg.evaluate(f"()=>[...document.querySelectorAll('{sel}')].filter(e=>e.getClientRects().length).length")
shown_groups = lambda pg: [g for g in ('bc', 'es', 'nn') if vis(pg, f'#ts-g-{g}')]


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#ts-app .labtitle') == Z(lang, 'How do you turn a Python loop into one vector operation?', '怎么把 Python 循环变成一次向量运算？'), text(pg, '#ts-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#ts-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#ts-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#ts-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')), text(pg, '#ts-app .lab-outcome'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ts-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#ts-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#ts-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1: the mission, three ways at N 10,000, M 50,000, D 512
    st = state(pg)
    check(f'{tag} part 1: the mission sizes, the 1-NN view', (st['view'], st['N'], st['M'], st['D'], st['step']) == ('nn', 10000, 50000, 512, 0), st)
    bv = cells(pg, '#ts-nn .ts-bv')
    check(f'{tag} part 1: three bars, 2 KiB, 954 GiB (does not fit), 1.86 GiB', len(bv) == 3 and bv[0] == '2 KiB' and bv[1].startswith('954 GiB') and bv[2] == '1.86 GiB'
          and pg.evaluate("()=>[...document.querySelectorAll('#ts-nn .ts-nnr')].map(r=>r.classList.contains('ts-over'))") == [False, True, False], bv)
    card = text(pg, '#ts-rail .step.current')
    jt = pg.evaluate("()=>{const e=document.querySelector('#ts-rail .step.current [data-jump=\"nn\"]'); return e?e.dataset.tipx||'':''}")
    check(f'{tag} part 1: the card gives 500M runs, about a terabyte and under 2 GiB; the link’s tooltip the exact sizes; GiB defined', all(x in card for x in ('500M', Z(lang, 'about a terabyte', '大约 1 TB'), Z(lang, 'under 2 GiB', '不到 2 GiB'))) and all(x in jt for x in ('954 GiB', '1.02 TB', '1.86 GiB', '2.00 GB'))
          and pg.evaluate("()=>[...document.querySelectorAll('#ts-rail .step.current .def')].some(d=>d.textContent==='GiB'&&/1,073,741,824|2\\^30|2³⁰/.test(d.dataset.tipx||''))"), card[:300])
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#ts-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), rows=document.querySelector('#ts-nn .ts-nnrows');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(rows); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, rows:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart’s bars start on screen (R7.3)', fs['bar'] and fs['next'] and fs['rows'] is not None and fs['rows'] < fs['vh'] - 80, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: Settings fixed with a note', vis(pg, '#ts-lock') and not shown_groups(pg), (text(pg, '#ts-lock'), shown_groups(pg)))
    answers = [Z(lang, x, y) for x, y in (('gives (3, 4)', '得到 (3, 4)'), ('Result (3, 4)', '结果 (3, 4)'), ('Result: (3, 4)', '结果：(3, 4)'), ('broadcast to (3, 4)', '广播成 (3, 4)'), ('→ (3, 4)', '→ (3, 4)'))]
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer to part 2 appears nowhere (R3.8)', not any(a in ba for a in answers), [a for a in answers if a in ba])
    # part 2: the gate
    click(pg, '#ts-rail [data-nav="next"]')
    check(f'{tag} part 2 reached, every Next disabled before a prediction', state(pg)['step'] == 1 and all(nexts(pg)), nexts(pg))
    check(f'{tag} part 2: the result reads “?” (boxes and metric), the shapes are shown', count(pg, '#ts-bc .ts-ax.ts-q') == 2 and vis(pg, '#ts-bc .ts-qbox') and text(pg, '#ts-metrics [data-k="out"] .v') == '?'
          and count(pg, '#ts-bc .ts-ax.ts-ghost') == 0 and count(pg, '#ts-bc .ts-c.ts-ghost') == 0, (count(pg, '#ts-bc .ts-ax.ts-q'), text(pg, '#ts-metrics [data-k="out"] .v')))
    c2 = text(pg, '#ts-rail .step.current')
    check(f'{tag} part 2: the question does not say the operation is allowed (A1)', Z(lang, 'What does a + b give?', 'a + b 会得到什么？') in c2 and not any(x in c2 for x in ('anyway', '照样')), c2[:300])
    ba = before_answer(pg)
    check(f'{tag} part 2: the answer appears nowhere (text, popovers, term tooltips; R3.8)', not any(a in ba for a in answers), [a for a in answers if a in ba])
    why = Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。')
    check(f'{tag} part 2: why Next is greyed out, next to the Next that is shown (A16)', pg.evaluate("(w)=>[...document.querySelectorAll('#ts-rail .ts-why')].some(p=>!p.hidden&&p.getClientRects().length&&p.textContent===w)", why))
    if phone:
        check(f'{tag} part 2: on a phone, a way from the chart to the question (A14)', vis(pg, '#ts-phone [data-act="to-q"]'))
    fit(pg, f'{tag} part 2'); rules(pg, f'{tag} part 2', lang)
    click(pg, '#ts-rail [data-pred="31"]')
    check(f'{tag} part 2: “Saved … you can still change it”, pressed', Z(lang, 'Saved. You can still change it before part 3 shows the result.', '已记下。第 3 部分给出结果之前，还可以改。') in text(pg, '#ts-rail .step.current')
          and pg.evaluate("()=>document.querySelector('#ts-rail [data-pred=\"31\"]').getAttribute('aria-pressed')==='true'"))
    check(f'{tag} part 2: Next enabled after a prediction, the reason line gone', not any(nexts(pg)) and not pg.evaluate("()=>[...document.querySelectorAll('#ts-rail .ts-why')].some(p=>!p.hidden&&p.style.visibility!=='hidden')"), nexts(pg))
    shot(pg, f'{tag}_p2')
    # part 3: broadcasting
    click(pg, '#ts-rail [data-nav="next"]')
    o = obs(pg)
    check(f'{tag} part 3: no ▲▼ badge on entering; Next waits for a pair that does not broadcast', not badges(pg) and all(nexts(pg)), (badges(pg), nexts(pg)))
    check(f'{tag} part 3: result, verdict with the right answer, why (R3.10)', all(x in o for x in (Z(lang, '(3, 1) + (1, 4) gives (3, 4).', '(3, 1) + (1, 4) 得到 (3, 4)。'),
          Z(lang, 'You predicted “(3, 1)”. The right answer is “(3, 4)”.', '你预测的是“(3, 1)”。正确答案是“(3, 4)”。'), Z(lang, 'reads it 4 times', '读 4 遍')))
          and o.index(Z(lang, 'gives (3, 4)', '得到 (3, 4)')) < o.index(Z(lang, 'You predicted', '你预测的是')) < o.index(Z(lang, 'Why:', '原因：')), o[:400])
    check(f'{tag} part 3: two stretched axes and 17 ghost numbers, a 3 × 4 result', count(pg, '#ts-bc .ts-ax.ts-ghost') == 2 and count(pg, '#ts-bc .ts-c.ts-ghost') == 17 and cells(pg, '#ts-bc .ts-ax.ts-out') == ['3', '4'],
          (count(pg, '#ts-bc .ts-ax.ts-ghost'), count(pg, '#ts-bc .ts-c.ts-ghost'), cells(pg, '#ts-bc .ts-ax.ts-out')))
    check(f'{tag} part 3: Settings show only the shapes', shown_groups(pg) == ['bc'], shown_groups(pg))
    if phone:
        check(f'{tag} part 3: on a phone the shape chips sit in the chart (A14)', vis(pg, '#ts-phone .seg-ctl'))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    click(pg, '#ts-ic-a button[data-v="4x3"]'); click(pg, '#ts-ic-b button[data-v="4"]')
    o = obs(pg)
    check(f'{tag} part 3: (4, 3) + (4,) fails on the last axis, outlined in red; Next enabled', Z(lang, '(4, 3) + (4,): an error.', '(4, 3) + (4,)：报错。') in o and Z(lang, 'On the last axis, a has 3 and b has 4', '在最后一个轴上，a 是 3，b 是 4') in o
          and count(pg, '#ts-bc .ts-ax.ts-bad') >= 2 and vis(pg, '#ts-bc .ts-err') and not any(nexts(pg)) and text(pg, '#ts-status') == Z(lang, 'Error', '报错'), o[:300])
    check(f'{tag} part 3: why names the fix, shape (4, 1)', Z(lang, 'b needs shape (4, 1)', 'b 的形状要是 (4, 1)') in o, o[-300:])
    shot(pg, f'{tag}_p3_error', full=phone)
    rules(pg, f'{tag} part 3 error', lang); fit(pg, f'{tag} part 3 error')
    click(pg, '#ts-ic-b button[data-v="3"]')
    check(f'{tag} part 3: (4, 3) + (3,) gives (4, 3), the missing axis dashed', Z(lang, '(4, 3) + (3,) gives (4, 3).', '(4, 3) + (3,) 得到 (4, 3)。') in obs(pg) and '(1) → 4' in cells(pg, '#ts-bc .ts-ax.ts-ghost'), (obs(pg)[:200], cells(pg, '#ts-bc .ts-ax.ts-ghost')))
    # part 4: einsum
    click(pg, '#ts-rail [data-nav="next"]')
    o = obs(pg)
    check(f'{tag} part 4: the matrix example, 24 multiply-adds, the top-left sum written out; Next waits', Z(lang, '24 multiply-adds', '一共 24 次乘加') in o and '13 = 0 × 1 + 1 × 3 + 2 × 5' in o.replace(' is ', ' = ')
          and cells(pg, '#ts-es .ts-c.ts-out') == ['13', '16', '22', '28', '31', '40', '40', '52'] and all(nexts(pg)), (o[:300], cells(pg, '#ts-es .ts-c.ts-out')))
    check(f'{tag} part 4: Settings show only the string', shown_groups(pg) == ['es'], shown_groups(pg))
    if phone:
        check(f'{tag} part 4: on a phone, a way from the chart to the strings (A14)', vis(pg, '#ts-phone [data-act="to-es"]'))
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4')
    click(pg, '#ts-ic-es [data-lever="sc"]')
    o = obs(pg)
    check(f'{tag} part 4: the attention scores sum d, 17.2B multiply-adds; Next enabled; the result on the option', Z(lang, 'the output has shape (1, 32, 2048, 2048), and d is summed: 17.2B', '的输出形状是 (1, 32, 2048, 2048)，对 d 求和，一共 17.2B') in o
          and not any(nexts(pg)) and Z(lang, 'sums d', '对 d 求和') in text(pg, '#ts-ic-es [data-lever="sc"]'), o[:300])
    # review v6 TS-B1: part 4 is about the letters only: no query, key, value, heads' work or attention weights in the options or the result
    r4 = pg.evaluate("()=>document.querySelector('#ts-rail .step.current').innerText")
    check(f'{tag} part 4: the options and the result talk about letters and shapes only, no attention background (review v6 TS-B1)',
          not any(x in r4 for x in ('query', 'value', Z(lang, 'attention weights', '注意力权重'), Z(lang, 'a score for every', '算一个得分'))) and Z(lang, 'b, h, q and k are all written after the arrow', 'b、h、q、k 都写在箭头后面') in r4, r4[-400:])
    # review 2026-10-03 (R5.10): the summed letter's output cell is grey with a slash and says so in words; no strike-through on a one-letter cell
    sx = pg.evaluate("()=>[...document.querySelectorAll('#ts-es .ts-lt td.ts-x')].map(td=>({l:td.dataset.l,t:td.textContent.trim(),s:td.querySelectorAll('s').length,bg:getComputedStyle(td).backgroundImage}))")
    check(f'{tag} part 4: the summed letter d: a grey cell with a slash that says “summed away”, no strike-through (review 2026-10-03)', len(sx) == 1 and sx[0]['l'] == 'd' and sx[0]['t'] == Z(lang, 'summed away', '求和后去掉') and sx[0]['s'] == 0 and 'linear-gradient' in sx[0]['bg']
          and Z(lang, 'grey with a slash: summed away', '灰格加斜线：求和后去掉') in text(pg, '#ts-legend'), (sx, text(pg, '#ts-legend')))
    ln = pg.evaluate("()=>[...document.querySelectorAll('#ts-es .ts-lt thead th[data-tipx]')].map(t=>[t.querySelector('.ts-lchar').textContent,t.querySelector('.ts-lname').textContent,(t.dataset.tipx||'').length])")
    check(f'{tag} part 4: each letter column has a name and a definition on hover (A2)', [x[0] for x in ln] == ['b', 'h', 'q', 'k', 'd'] and [x[1] for x in ln] == ['batch', 'head', 'query', 'key', Z(lang, 'dims', '维度')] and all(x[2] > 30 for x in ln), ln)
    # review 2026-10-03: no attention background in the card, one plain sentence on where the two strings come from; the single-choice
    # note sits right above the strings (R2.5), not in the prose
    c4 = pg.evaluate("()=>{const c=document.querySelector('#ts-rail .step.current'), x=c.cloneNode(true); x.querySelectorAll('.ts-lv').forEach(e=>e.remove()); return x.innerText}")
    one = pg.evaluate("()=>{const o=document.querySelector('#ts-rail .step.current .ts-one'), l=document.getElementById('ts-ic-es'); return !!o&&!!l&&o.nextElementSibling===l&&o.getClientRects().length>0?o.textContent:''}")
    check(f'{tag} part 4: no attention background in the card, one sentence on where the strings come from, the lead-in above the strings (review 2026-10-03; owner G1, 2026-10-08)',
          not any(x in c4 for x in (Z(lang, 'a query, a key and a value', 'query、key 和 value'), Z(lang, 'attention weights', '注意力权重'), Z(lang, 'split into 32 heads', '分到 32 个 head'))) and Z(lang, 'here only their letters matter', '这里只看它们的字母') in c4
          and one == Z(lang, 'Pick an einsum string and see which letters it sums over.', '选一个 einsum 字符串，看看它对哪些字母求和。'), (c4[-300:], one))
    check(f'{tag} part 4: single choice', pg.evaluate("()=>[...document.querySelectorAll('#ts-ic-es [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1)
    shot(pg, f'{tag}_p4_scores', full=phone)
    rules(pg, f'{tag} part 4 scores', lang); fit(pg, f'{tag} part 4 scores')
    # part 5: the 1-NN rewrite
    click(pg, '#ts-rail [data-nav="next"]')
    check(f'{tag} part 5: no badge on entering, Next waits for D, the identity as MathML', not badges(pg) and all(nexts(pg)) and count(pg, '#ts-rail .ts-fx math') == 2, (badges(pg), nexts(pg)))
    # review 2026-10-03: each way of writing the distance, term by term, with what it computes and its memory, next to the formula;
    # the bars of ‖x‖² grouped so both stand the same height
    tm = pg.evaluate("()=>[...document.querySelectorAll('#ts-terms [data-term]')].map(r=>[r.dataset.term,r.querySelector('.ts-tb').textContent,r.classList.contains('ts-over'),r.querySelector('.ts-ts').textContent,r.querySelector('.ts-tt').textContent.length])")
    check(f'{tag} part 5: the terms of both ways with shape and memory: 954 GiB (red) and 1.86 GiB; 39.1 KiB, 1.86 GiB and 195 KiB (review 2026-10-03)',
          [(x[0], x[1], x[2]) for x in tm] == [('bcd', '954 GiB', True), ('bcs', '1.86 GiB', False), ('xx', '39.1 KiB', False), ('xy', '1.86 GiB', False), ('yy', '195 KiB', False)]
          and [x[3].split(' ', 1)[1] if ' ' in x[3] else x[3] for x in tm] == ['(N, M, D)', '(N, M)', '(N, 1)', '(N, M)', '(1, M)'] and all(x[4] > 12 for x in tm), tm)
    bars = pg.evaluate("()=>[...document.querySelectorAll('#ts-terms msup')].map(m=>(m.firstElementChild.textContent.match(/‖/g)||[]).length)")
    # 6 squared norms (the distance's definition, review v6 TS-S8, the identity's four and the two term rows) and one (x_d − y_d)²
    check(f'{tag} part 5: every squared norm has both of its bars inside the base of the square (review 2026-10-03)', len([b for b in bars if b]) == 6 and all(b == 2 for b in bars if b) and bars.count(0) == 1, bars)
    check(f'{tag} part 5: Settings show only N, M, D', shown_groups(pg) == ['nn'], shown_groups(pg))
    if phone:
        check(f'{tag} part 5: on a phone the D slider sits in the chart (A14)', vis(pg, '#ts-ph-D-in'))
    slide(pg, '#ts-ic-D-in', 4)
    o = obs(pg)
    check(f'{tag} part 5 at D 1,024: the broadcast grows to 1,907 GiB, the matmul stays 1.86 GiB, ▲ on the broadcast card', Z(lang, 'the broadcast needs 1,907 GiB', '广播写法要 1,907 GiB') in o and Z(lang, 'the expansion needs 1.86 GiB', '展开写法要 1.86 GiB') in o
          and any('▲' in x for x in badges(pg)) and not any(nexts(pg)), (o[:300], badges(pg)))
    check(f'{tag} part 5: X and Y named, the inserted axes drawn as boxes (A3)', vis(pg, '#ts-nn .ts-nnshape') and count(pg, '#ts-nn .ts-nnshape .ts-ax.ts-ghost') == 2
          and Z(lang, 'X holds the 10,000 test vectors, shape (N, D)', 'X 是 10,000 个测试向量，形状 (N, D)') in text(pg, '#ts-nn') and Z(lang, 'The price is precision', '代价是精度') in o, text(pg, '#ts-nn')[:300])
    tm = pg.evaluate("()=>[...document.querySelectorAll('#ts-terms [data-term]')].map(r=>[r.dataset.term,r.querySelector('.ts-tb').textContent])")
    check(f'{tag} part 5 at D 1,024: only the broadcast difference follows D in the terms', tm == [['bcd', '1,907 GiB'], ['bcs', '1.86 GiB'], ['xx', '39.1 KiB'], ['xy', '1.86 GiB'], ['yy', '195 KiB']], tm)
    # review 2026-10-03: why None goes in the middle, in the picture: each column named by what it indexes, and the pairs x_i − y_j
    # (the owner, 2026-10-09, H1: each pair is typeset, so its text is x1−y1 with no spaces, and each of the 9 cells holds MathML)
    ph = pg.evaluate("()=>({h:[...document.querySelectorAll('#ts-nn .ts-nnshape .ts-axh')].map(e=>e.textContent), cells:[...document.querySelectorAll('#ts-nn .ts-pgrid .ts-pc:not(.ts-pe)')].map(e=>e.textContent), mcells:document.querySelectorAll('#ts-nn .ts-pgrid .ts-pc:not(.ts-pe) math.tex').length, cap:(document.querySelector('#ts-nn .ts-pcap')||{}).textContent||''})")
    check(f'{tag} part 5: the picture names each axis and shows the pairs x_i − y_j (review 2026-10-03)', ph['h'] == [Z(lang, 'which test vector', '哪个测试向量'), Z(lang, 'which training vector', '哪个训练向量'), Z(lang, 'which dimension', '第几维')]
          and len(ph['cells']) == 9 and ph['cells'][0] == 'x1−y1' and ph['cells'][5] == 'x2−y3' and ph['mcells'] == 9 and '1,024' in ph['cap'], ph)
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    # part 6: what to remember
    click(pg, '#ts-rail [data-nav="next"]')
    card = text(pg, '#ts-rail .step.current')
    check(f'{tag} part 6: three misconceptions, See the scenarios', card.count('✗') == 3 and vis(pg, '#ts-rail [data-mode-go="sandbox"]'), card[-300:])
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; return shown(document.querySelector('#ts-rail .next-lab'))||shown(document.querySelector('#view-dyn .next-lab-end .next-lab'))}""")
    check(f'{tag} part 6: the next-lab footer (R3.11)', nl)
    click(pg, '#ts-rail [data-rem="es"]')
    check(f'{tag} part 6: “Show in the chart” puts the attention scores in the chart', state(pg)['remView'] == 'es' and vis(pg, '#ts-es') and Z(lang, 'In the chart now', '图里正显示') in text(pg, '#ts-rail [data-rem="es"]'))
    click(pg, '#ts-rail [data-rem="nn"]')
    check(f'{tag} part 6: and the three bars', state(pg)['remView'] == 'nn' and vis(pg, '#ts-nn') and cells(pg, '#ts-nn .ts-bv')[2] == '1.86 GiB')
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6')
    check(f'{tag} guided: no overflow', overflow(pg) <= 0, overflow(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>document.querySelectorAll('#ts-rail [data-preset]').length")
    check(f'{tag} scenarios: four, single choice, with a lead-in (owner G1, 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see how its shapes work together.', '选一个场景，看看那里的形状怎么一起运算。') in text(pg, '#ts-rail .railcard'))
    want = [Z(lang, 'the result is (32, 4096)', '结果是 (32, 4096)'), Z(lang, 'divides (32, 4096) by (32, 1)', '用 (32, 1) 去除 (32, 4096)'), '17.2B', Z(lang, 'the broadcast needs 763 MiB', '广播写法要 763 MiB')]
    for i in range(presets):
        click(pg, f'#ts-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#ts-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#ts-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and want[i] in ins, (pressed, ins[:160]))
        if i < 2:
            check(f'{tag} scenario {i}: shapes too large to print get no sentence about the page (review 2026-10-03)', not any(x in text(pg, '#ts-bc') for x in ('too large to print', '打印不出数', 'carry the rule', '已经说明了规则')), text(pg, '#ts-bc')[-200:])
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        shot(pg, f'{tag}_scenario_{i}', full=phone)
        rules(pg, f'{tag} scenario {i}', lang); fit(pg, f'{tag} scenario {i}')
    check(f'{tag} scenario 4: 15.3 MiB for the matmul, 50 times', '15.3 MiB' in text(pg, '#ts-insight') and '50' in text(pg, '#ts-insight'))
    hsh = pg.evaluate("()=>location.hash")
    st0 = state(pg)
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#tensors~nn_'), hsh)
    p2 = b.new_page(viewport={'width': w, 'height': h}); open_drawers(p2)
    p2.on('pageerror', lambda e: errs.append(f'{w} {lang} reload: {e}'))
    p2.goto(URL + hsh); p2.wait_for_timeout(600)
    st1 = p2.evaluate("()=>window.__ts.state()")
    check(f'{tag} scenarios: a fresh page restores the setup', all(st1[k] == st0[k] for k in ('view', 'a', 'b', 'es', 'N', 'M', 'D')) and st1['mode'] == 'sandbox', (st0, st1))
    p2.close()
    # a change under Settings: the chart follows its group, the page does not move (R3.2), the preset releases
    pg.evaluate("()=>document.getElementById('ts-ctl-a').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
    y0 = pg.evaluate("()=>document.getElementById('ts-ctl-a').getBoundingClientRect().top")
    click(pg, '#ts-ctl-a button[data-v="2x3"]')
    y1 = pg.evaluate("()=>document.getElementById('ts-ctl-a').getBoundingClientRect().top")
    check(f'{tag} Settings: a shape chip switches the chart to broadcasting and does not move the page', state(pg)['view'] == 'bc' and abs(y1 - y0) < 2, (state(pg)['view'], y0, y1))
    check(f'{tag} Settings: no preset pressed, and the card says so', pg.evaluate("()=>[...document.querySelectorAll('#ts-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')")
          and Z(lang, 'match none of these scenarios', '和这几个场景都不一样') in text(pg, '#ts-insight'), text(pg, '#ts-insight'))
    slide(pg, '#ts-ctl-D-in', 5)
    check(f'{tag} Settings: D switches the chart to 1-NN; a ▲ after the change', state(pg)['view'] == 'nn' and state(pg)['D'] == 2048, state(pg))
    click(pg, '#ts-view button[data-view="es"]')
    check(f'{tag} scenarios: the view switch', state(pg)['view'] == 'es' and vis(pg, '#ts-es'))
    rules(pg, f'{tag} scenarios settings', lang); fit(pg, f'{tag} scenarios settings')
    click(pg, '#ts-controls [data-act="reset"]')
    check(f'{tag} Settings: reset keeps the view, restores the sizes', state(pg)['D'] == 512 and state(pg)['view'] == 'es', state(pg))
    # the challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: Settings locked with a note, every group hidden', pg.evaluate("()=>document.getElementById('ts-controls').classList.contains('locked')") and vis(pg, '#ts-lock') and not shown_groups(pg))
    check(f'{tag} challenge: starts at the double loop, 3 KiB and 2.00B runs; live progress', state(pg)['chPick'] == 'loop' and text(pg, '#ts-ch-b') == '3 KiB' and text(pg, '#ts-ch-i') == '2.00B'
          and pg.evaluate("()=>document.getElementById('ts-ch-g1').classList.contains('met')&&document.getElementById('ts-ch-g2').classList.contains('ts-over')"), (text(pg, '#ts-ch-b'), text(pg, '#ts-ch-i')))
    check(f'{tag} challenge: the stage shows only the initial setup before an answer', count(pg, '#ts-nn .ts-nnr') == 1, count(pg, '#ts-nn .ts-nnr'))
    shot(pg, f'{tag}_challenge_start', full=phone)
    rules(pg, f'{tag} challenge start', lang); fit(pg, f'{tag} challenge start')
    click(pg, '#ts-rail [data-act="submit"]')
    res = text(pg, '#ts-ch-res')
    check(f'{tag} challenge tier 3: not yet, the loop runs too often, a hint, “Show the answer”', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'more than 1,000', '超过 1,000 次') in res and vis(pg, '#ts-ch-res [data-act="reveal"]')
          and text(pg, '#ts-status') == Z(lang, 'Not yet', '还没达到') and text(pg, '#ts-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:240])
    click(pg, '#ts-ch-res [data-act="reveal"]')
    cr = pg.evaluate("""()=>{const rows=[...document.querySelectorAll('#ts-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
      return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
        stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#ts-ch-res .cmp-note')||{}).textContent||'',
        title:(document.querySelector('#ts-ch-res .cmp-title')||{}).textContent||''}}""")
    check(f'{tag} challenge, the answer for the initial setup: the 2 passing ways and the double loop (my pick), nothing else', cr['n'] == 3 and cr['ok'] == 2 and cr['me'] == 1 and cr['stray'] == 0, cr)
    check(f'{tag} challenge: the note counts the 3 ways left out; the title no longer says “each way”', cr['note'] == Z(lang, 'The other 3 ways miss at least one target and are not shown.', '其余 3 种写法至少差一个目标，图里没有列出。')
          and not re.search(r'each way|每种写法', cr['title']), (cr['note'], cr['title']))
    check(f'{tag} challenge: the broadcast, not in the chart, is named in words with its size in a hover', Z(lang, 'far larger than one H100', '远超一张 H100 的显存') in text(pg, '#ts-ch-res')
          and '5,722 GiB' in pg.evaluate("()=>[...document.querySelectorAll('#ts-ch-res [data-tipx]')].map(e=>e.dataset.tipx).join(' ')"))
    click(pg, '#ts-ic-ch [data-lever="c256"]')
    res = text(pg, '#ts-ch-res')
    check(f'{tag} challenge: chunks of 256 do not fit, 146 GiB in red', res.startswith(Z(lang, 'Not yet: the largest array is 146 GiB', '还没达到：最大的数组是 146 GiB')) and pg.evaluate("()=>document.getElementById('ts-ch-g1').classList.contains('ts-over')"), res[:200])
    click(pg, '#ts-ic-ch [data-lever="c32"]')
    res = text(pg, '#ts-ch-res')
    check(f'{tag} challenge tier 2: chunks of 32 meet the goal, but the expansion is smaller; 768 times', res.startswith(Z(lang, 'Goal met: 18.3 GiB at the largest and 313 runs of the loop.', '达成目标：最大的数组 18.3 GiB，循环 313 次。'))
          and '7.45 GiB' in res and '768' in res, res[:300])
    rows = pg.evaluate("()=>[...document.querySelectorAll('#ts-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me')}))")
    check(f'{tag} challenge: the brief chart, the initial setup and the two passing ways, the pick marked (R5.4)', len(rows) == 3 and [r['ok'] for r in rows] == [False, True, True] and [r['me'] for r in rows] == [False, True, False], rows)
    click(pg, '#ts-ic-ch [data-lever="mm"]')
    res = text(pg, '#ts-ch-res')
    check(f'{tag} challenge tier 1: the matmul expansion, the smallest array', res.startswith(Z(lang, 'Goal met, with the smallest array: 7.45 GiB and no Python loop.', '达成目标，而且数组最小：7.45 GiB，没有 Python 循环。'))
          and text(pg, '#ts-status') == Z(lang, 'Both targets met', '两个目标都达到') and Z(lang, 'Note:', '注意：') in res and Z(lang, 'initial setup', '初始设置') in res, res[:240])
    vh = pg.evaluate("()=>[...document.querySelectorAll('#ts-ch-res .cmp-val')].map(v=>Math.round(v.getBoundingClientRect().height))")
    check(f'{tag} challenge: every value label of the chart on one line (A8)', len(vh) == 3 and max(vh) <= 18, vh)
    # the owner, 2026-10-09 (H1): N × M is typeset, so the sub's text reads N×M (MathML, no spaces) and the sub holds a math element
    wsub = pg.evaluate("()=>{const e=document.querySelector('#ts-metrics [data-k=\"wr\"] .sub'); return e?{t:e.textContent, m:e.querySelectorAll('math.tex').length}:{t:'',m:0}}")
    check(f'{tag} challenge: the loop limit explained, the third metric named (A6, A7)', vis(pg, '#ts-rail .ts-goalwhy') and wsub['t'] == Z(lang, 'only the N×M dot products', '只有 N×M 个点积') and wsub['m'] == 1, wsub)
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang); fit(pg, f'{tag} challenge met')
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#ts-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#ts-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#ts-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    # review 2026-10-03: the second new case says it plainly: the work grows with the square of the context, in attention, q against every k
    pg.evaluate("()=>document.querySelector('#ts-rail [data-exits] [data-ex=\"tab\"][data-i=\"1\"]').click()"); pg.wait_for_timeout(150)
    pg.evaluate("()=>document.querySelector('#ts-rail [data-exits] .pb-dot[data-j=\"1\"]').click()"); pg.wait_for_timeout(150)
    pg.evaluate("()=>document.querySelector('#ts-rail [data-exits] .ex-opts .choice[data-v=\"1\"]').click()"); pg.wait_for_timeout(150)
    ex = text(pg, '#ts-rail [data-exits] .ex-res')
    check(f'{tag} new case 2: about 16 times, because the scores grow with the square of the context length, every q against every k, in attention (review 2026-10-03)',
          all(x in ex for x in (Z(lang, 'About 16 times', '约 16 倍'), Z(lang, 'grows with the square of the context length', '随上下文长度的平方增长'), Z(lang, 'every token’s q is scored against every token’s k', '每个 token 的 q 都要和每个 token 的 k 算一个得分'), '17.2B', '275B', 'attention')), ex)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the submitted result', st['chPick'] == 'mm' and st['chSubmitted'] and vis(pg, '#ts-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('ts-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, with a short status and both sizes', dk['shown'] and len(dk['status']) <= 14 and '954 GiB' in dk['read'] and '1.86 GiB' in dk['read'], dk)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see how its shapes work together.', '选一个场景，看看那里的形状怎么一起运算。'), 'chal': None, 'parts': [(3, [('Pick an einsum string and see which letters it sums over.', '选一个 einsum 字符串，看看它对哪些字母求和。')]), (5, [('So a row of shape (D,) is added to every row of an (N, D) array', '所以形状为 (D,) 的一行，可以直接加到 (N, D) 数组的每一行上'), ('So the letters after the arrow name the result’s axes', '所以箭头后面的字母就是结果的轴'), ('One vectorized line can build an array far bigger', '一行向量化的代码，中途可能建出比输入和结果都大得多的数组'), ('once for each row it meets', '每遇到一行就读一次'), ('the double loop holds one (D,) difference at a time', '两层循环每次只存一个 (D,) 的差值')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'tensors.js').read_text() + ((P / 'teasers' / 'tensors.js').read_text() if (P / 'teasers' / 'tensors.js').exists() else '')
        left = [s for s in OWNER_B9_NOTES if s in src]
        check('owner b9 G1: no “one at a time / do not stack” note left in the lab or its tile, EN or ZH', not left, left)
    k = 1 if lang == 'zh' else 0
    pg = page(b, w, h, lang)
    norm = "()=>{const c=[...document.querySelectorAll('.railcard')].find(x=>x.offsetParent)||document.querySelector('.step.current'); return c?c.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}"
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"sandbox\"]:not([data-flow=free])')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    t = pg.evaluate(norm)
    check(f'{w} {lang} owner b9 G1: the Scenarios card leads in with what to pick and what to watch', OWNER_B9['scen'][k] in t, t[:240])
    if OWNER_B9['chal']:
        pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"challenge\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
        t = pg.evaluate(norm)
        check(f'{w} {lang} owner b9 G1: the challenge leads in to its options', OWNER_B9['chal'][k] in t, t[:400])
    pg.evaluate("()=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"guided\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}"); pg.wait_for_timeout(600)
    for part, wants in OWNER_B9['parts']:
        pg.evaluate(f"()=>{{const b=document.querySelector('.step-h[data-go=\"{part}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(700)
        t = pg.evaluate("()=>{const c=document.querySelector('.step.current'); return c?c.textContent.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
        for s in wants:
            check(f'{w} {lang} owner b9: part {part + 1} has “{s[k][:40]}”', s[k] in t, t[:300])
        back = OWNER_B9_BACK.findall(t)
        check(f'{w} {lang} owner b9 G2: part {part + 1} leans on no detail of an earlier part', not back, back)
    pg.close()



# ---- the owner's decisions of 2026-10-09 (sweep of the kernels group, reviews/sweep-kernels-owner-b9v6.md): H1, every formula is
# typeset with the kit (MathML, no TeX command shown as a word) and none of the converted plain-text formulas is left in the source or
# on screen (text and hovers, EN and ZH); H2, Show the math comes in short blocks split by a hairline. Every guided part, Scenarios
# and the challenge, at each width and language this test runs.
B9V6_PROBE = r"""([mb,blk,xsep])=>{const root=document.querySelector('#view-dyn:not([hidden])'); if(!root) return {n:0,bad:[],txt:'',sep:[],blocks:0,body:'',inBody:0,xsep:[]};
  const box=document.querySelector(mb); const det=box&&box.closest('details'); if(det) det.open=true;
  const CMD=/^(left|right|frac|sqrt|text|times|cdot|div|approx|Big|big|hat|leftarrow|mathrm|operatorname|lceil|rceil|lfloor|rfloor|quad|infty)$/;
  const maths=[...root.querySelectorAll('math')], bad=[];
  for(const m of maths) for(const e of m.querySelectorAll('mi,mtext,mo')){const t=e.textContent.trim(); if(CMD.test(t)||t.includes('\\')) bad.push(t);}
  const tips=[...root.querySelectorAll('[data-tipx]')].map(e=>e.getAttribute('data-tipx')||'');
  for(const t of tips){const m=t.match(/<m(?:i|text)[^>]*>(left|right|frac|sqrt|text|times|cdot|Big|hat|lceil|rceil)<\/m(?:i|text)>/g); if(m) bad.push(...m);}
  const blocks=box?[...box.querySelectorAll(blk)]:[];
  const sep=blocks.filter(x=>x.previousElementSibling&&x.previousElementSibling.matches(blk)).map(x=>{const cs=getComputedStyle(x); return cs.borderTopWidth+' '+cs.borderTopStyle;});
  const parts=[]; const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let nd;
  while((nd=tw.nextNode())){const e=nd.parentElement; if(!e||e.closest('[data-endnotes],pre,code')||!e.getClientRects().length) continue; parts.push(nd.textContent);}
  const tipTxt=tips.map(t=>{const d=document.createElement('div'); d.innerHTML=t; d.querySelectorAll('pre,code').forEach(x=>x.remove()); return d.textContent;}).join(' ');
  const xs=xsep?[...root.querySelectorAll(xsep)].filter(e=>e.getClientRects().length).map(e=>{const cs=getComputedStyle(e); return cs.borderTopWidth+' '+cs.borderTopStyle;}):[];
  return {n:maths.length, bad, blocks:blocks.length, sep, body:box?box.textContent.trim():'', inBody:box?box.querySelectorAll('math').length:0, txt:parts.join(' ')+' '+tipTxt, xsep:xs};}"""


# the owner's review of 2026-10-09 (R6.37, reviews/math-wrap-kernels-owner-b9v7.md): a row of “Show the math” the reader has to scroll
# sideways is not read. With Show the math open, no box in it scrolls sideways and it is no wider than its own box.
MSCROLL_PROBE = r"""(mb)=>{const box=document.querySelector(mb); const d=box&&box.closest('details'); if(!d||!d.open||!d.getClientRects().length) return {seen:0,bad:[]};
  const out=[];
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.getClientRects().length) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push((typeof e.className==='string'&&e.className?e.className.split(' ')[0]:e.tagName.toLowerCase())+' by '+o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}
  if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px');
  return {seen:1,bad:out};}"""


def owner_b9v6(b, w, h, lang, cfg):
    tag = f'{w} {lang} owner b9v6'
    if not cfg.get('_src_done'):
        cfg['_src_done'] = 1
        src = (P / 'labs' / f"{cfg['lab']}.js").read_text()
        left = [s for s in cfg['src_gone'] if s in src]
        check(f"owner b9v6 H1: no plain-text formula of the converted kinds left in labs/{cfg['lab']}.js", not left, left)
    pg = b.new_page(viewport={'width': w, 'height': h})
    here = []; pg.on('pageerror', lambda e: here.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{cfg['lab']}'}}"); pg.wait_for_timeout(600)
    nparts = pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
    states = [('guided', k) for k in range(nparts)] + [('sandbox', None), ('challenge', None)]
    bad, gone, nomath, seps, nblocks, mathn = [], set(), [], [], 0, 0
    nscr, scr = 0, []
    xseen, xbad = 0, []
    for mode, k in states:
        if k is None or k == 0:
            pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", mode); pg.wait_for_timeout(400)
        if k is not None:
            pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k)
        pg.wait_for_timeout(900)
        r = pg.evaluate(B9V6_PROBE, [cfg['math'], cfg['block'], cfg.get('sep_extra', '')])
        sx = pg.evaluate(MSCROLL_PROBE, cfg['math']); nscr += sx['seen']
        scr += [mode + ('' if k is None else f' part {k + 1}') + ': ' + x for x in sx['bad']]
        xseen += len(r['xsep']); xbad += [f'{where}: {x}' for x in r['xsep'] if not x.startswith('1px solid')]
        where = mode + ('' if k is None else f' part {k + 1}')
        bad += [f'{where}: {x}' for x in r['bad']]
        txt = r['txt'].replace('‑', '-').replace(' ', ' ').replace(' ', ' ')
        for s in cfg['text_gone']:
            if s in txt: gone.add(f'{where}: {s}')
        if r['body'] and not r['inBody'] and not re.search(cfg.get('nomath_ok', r'^$'), r['body']): nomath.append(where)
        if r['blocks'] >= 2: nblocks += 1
        seps += [f'{where}: {x}' for x in r['sep'] if not x.startswith('1px solid')]
        mathn += r['n']
    pg.close()
    check(f'{tag} H1: the formulas render as MathML ({mathn} on the visited screens), no TeX command shown as a word', mathn > 0 and not bad, bad[:5])
    check(f'{tag} H1: no plain-text formula of the converted kinds in the rendered text or the hovers', not gone, sorted(gone)[:6])
    check(f'{tag} H1/H2: Show the math typesets its formulas on every screen that has them', not nomath, nomath[:5])
    check(f'{tag} H2: Show the math comes in short blocks split by a 1px hairline ({nblocks} screens with two or more blocks)', nblocks > 0 and not seps, seps[:5])
    if cfg.get('sep_extra'):
        check(f"{tag} H5: {cfg['sep_extra']} starts its block under a 1px hairline ({xseen} seen)", xseen > 0 and not xbad, xbad[:5])
    check(f'{tag} owner 2026-10-09 (R6.37): no “Show the math” box scrolls sideways, every row fits its box ({nscr} screens with it open)', nscr > 0 and not scr, scr[:5])
    check(f'{tag}: no page errors', not here, here[:3])

B9V6 = {'lab': 'tensors', 'math': '#ts-mathbody', 'block': '.ts-mb', 'sep_extra': '.ts-nums', 'nomath_ok': 'Compare the axes|从右边逐个比较轴|The rule is shown in part 3|你预测之后', 'src_gone': ['gives N × M × D numbers', 'holds all N × M × D', 'write the whole N × M × D', '全部 N × M × D 个差值', "tx('only the N × M dot products'", "sub:'N × M'", '<i>${c}</i><sub>${i}</sub>', 'x<sub>i</sub> − y<sub>j</sub>', 'distance ‖x − y‖² (its formula', '欧氏距离的平方 ‖x − y‖²', 'close, ‖x‖² + ‖y‖² and 2 x·y', '很接近时，‖x‖² + ‖y‖² 和 2 x·y', 'C × M × D × 4 = ${fint(e.bytes)}'], 'text_gone': ['‖x − y‖²', '‖x‖² + ‖y‖²', 'C × M × D × 4 =', '⌈N ÷ C⌉ =']}


# B4 (polish round 5, reviews/r5-fit-roots.md): entering part 5 at 1440 × 900 as the polish probe stage.py does (Next when it
# is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare, English with none
# (147 px of scrolling in both languages before this round: the shapes table and the loop count under one another). Spare = the
# stage's max-height minus what its content and borders need.
# Its top below the top bar: below what is left of the bar on the screen (on a desktop lab page it scrolls away since refit step 2).
FIT_ENTRY = """()=>{const s=document.getElementById('ts-stage'), r=s.getBoundingClientRect(), tb=document.querySelector('.topbar'); return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb?Math.max(0,Math.round(tb.getBoundingClientRect().bottom)):0}}"""


def fit_page(b, lang):
    """the page as the polish probe stage.py opens it: window.__aspPrimer set (without it, automation gets the older tests' compat layout,
    whose step list also holds the settings panel), the guided parts picked in the step list, the real clock"""
    pg = b.new_page(viewport={'width': 1440, 'height': 900}); pg.add_init_script('window.__aspPrimer=true')
    pg.on('pageerror', lambda e: errs.append(f'fit 1440 {lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='tensors'}"); pg.wait_for_timeout(500)
    g = pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if g.count(): g.first.click(); pg.wait_for_timeout(700)
    return pg


def fit_entry(b, lang, parts=(5,)):
    pg = fit_page(b, lang)
    for k in range(2, max(parts) + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#ts-rail .steps>.step:nth-child({k}) .step-h').click()")
        pg.wait_for_timeout(900); settle(pg)
        if k not in parts: continue
        f = pg.evaluate(FIT_ENTRY); f['step'] = state(pg)['step'] + 1
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}, its top below the top bar (B4, R7.12)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= min(59, f['bar']), f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: fit_entry(b, lang)
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
    if FULL:
        run_guided(b, 1440, 900, 'en', dark=True)
        for (w, h) in [(1280, 800), (900, 900), (720, 900)]:
            pg = page(b, w, h, 'en')
            for m in ('guided', 'sandbox', 'challenge'):
                mode(pg, m)
                check(f'{w} {m}: no overflow', overflow(pg) <= 0, overflow(pg))
            shot(pg, f'{w}_en_challenge')
            pg.close()
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
