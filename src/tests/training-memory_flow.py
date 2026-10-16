"""Flow test for the training-memory lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, Chinese spacing and punctuation, no 亿/万), colours (R5.6);
- part 1: the prediction gate with the kit's reason under a greyed-out Next; before the pick the training column is hidden and the
  answer (168.7 GB) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the step auto-plays on entering at Medium (R3.9); Next waits for the end; then 168.7 GB, the verdict with the right answer,
  the 18-byte formula; the red overflow in the column and the out-of-memory marker on the curve (R5.11);
- R3.20: no replay when nothing changed (the optimizer lever, re-entering part 2), a replay when the shape changed (recomputation,
  micro-batches, scenarios), and the replay control (play, pause, replay, scrubber) works;
- part 3: Next waits for a change of the tokens at once; part 4: Next waits for recomputation, 147.7 GB; part 5: Next waits for all
  three tricks, 99.5 GB; part 6: three points with “show in the chart”, three misconceptions, See the scenarios;
- Scenarios: four, single choice, the first pressed on entry, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, the three feedback tiers, “Show all 16 plans”, 16 rows with 2 passing, the initial setup and the pick
  tagged; the capability card; a language switch keeps the plan and the result;
- the dock at 390 once the chart scrolls away.
Screenshots: shots/training-memory/.
Run: python3 tests/training-memory_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/training-memory.html', '--only', 'training-memory'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'training-memory'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'training-memory.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'training-memory'; SH.mkdir(parents=True, exist_ok=True)
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
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #tmem-mathbody, .num-big, #tmem-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/tmem-b-w|tz-tmem-seg w|sw w/,[amber]:/^$/,[red]:/status bad|badline|badc|tmem-over|tmem-oom|tmem-miss|tmem-cdot|tmem-pk|tmem-overarea|tmem-ov|\\bbad\\b|warn|wrong|ex-res|sw over|tmem-sw-over/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#tmem-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#tmem-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#tmem-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, hash_='training-memory'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__tmem.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tmem-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tmem-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#tmem-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__tmem||!window.__tmem.state().stageBusy", timeout=4000)
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
    pg.evaluate(f"()=>document.querySelector('#tmem-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#tmem-app', '#dock'):
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


NOEXACT = """()=>{const bad=[]; document.querySelectorAll('#tmem-app [data-tipx], #tmem-app [data-tip], #tmem-app [title], #dock [data-tipx]').forEach(e=>{
    if(e.closest('#tmem-math')) return; const t=(e.getAttribute('data-tipx')||'')+(e.getAttribute('data-tip')||'')+(e.getAttribute('title')||'');
    if(/精确值是|Exactly/.test(t)) bad.push(t.slice(0,90));});
  document.querySelectorAll('#tmem-app .num-big').forEach(e=>{if(e.classList.contains('def')||e.hasAttribute('tabindex')||e.hasAttribute('data-tipx')) bad.push('num-big with a hover: '+e.textContent);});
  return bad;}"""


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
    fm = pg.evaluate(FORMULA_H1, '#tmem-app')
    check(f'{where}: formulas typeset as MathML, no TeX word shown, no plain-text formula left (H1, R6.29)', not fm['bad'] and not fm['txt'], fm)


# the owner's review of 2026-10-09 (R6.37, R6.33): with Show the math open, no box in it scrolls sideways (a typeset formula never wraps
# by itself, so a row wider than its box hid its end until the reader scrolled), and Show the math is no wider than its own box
MSCROLL = r"""(r)=>{const v=document.querySelector(r); if(!v) return []; const out=[], ds=[...v.querySelectorAll('details.math')], was=ds.map(d=>d.open);
  ds.forEach(d=>{d.open=true;});
  for(const d of ds){if(!d.getClientRects().length) continue;
    for(const e of d.querySelectorAll('*')){if(!e.getClientRects().length||!/auto|scroll/.test(getComputedStyle(e).overflowX)) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(by+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}
    const by=d.scrollWidth-d.clientWidth; if(by>1) out.push('the whole Show the math, '+by+' px');}
  ds.forEach((d,i)=>{d.open=was[i];}); return out;}"""


def math_scroll(pg, where):
    ms = pg.evaluate(MSCROLL, '#tmem-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    math_scroll(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    ex = pg.evaluate(NOEXACT)
    check(f'{where}: no hover that expands a number to its exact value (the owner, 2026-10-07 23:37 UTC)', not ex, ex[:3])
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#tmem-app .labhead','#tmem-rail','#tmem-stage','#tmem-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#tmem-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#tmem-rail [data-nav="{what}"]', 300)


def wait_end(pg, timeout=15000):
    try: pg.wait_for_function("()=>window.__tmem.state().playT>=1&&!window.__tmem.state().playing", timeout=timeout); return True
    except Exception: return False


def scrub_end(pg):
    slide(pg, '#tmem-player input[data-key="scrub"]', 1000)


def run_guided(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#tmem-app .labtitle') == Z(lang, 'An 8B model’s weights take 16 GB: why does training it not fit on an 80 GB GPU?', '8B 模型的权重只有 16 GB，为什么训练它，80 GB 的卡都放不下？'), text(pg, '#tmem-app .labtitle'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#tmem-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#tmem-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#tmem-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1
    st = state(pg)
    check(f'{tag} part 1: guided, step 0, nothing playing', st['mode'] == 'guided' and st['step'] == 0 and not st['playing'], st)
    check(f'{tag} part 1: Next disabled with its reason', nexts(pg) and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer appears nowhere before the pick (R3.8)', '168.7' not in ba and '144.5' not in ba, [l for l in ba.split('\n') if '168.7' in l or '144.5' in l][:3])
    check(f'{tag} part 1: the training column is hidden', pg.evaluate("()=>document.querySelectorAll('#tmem-col rect[class^=\"tmem-b-\"]').length") == 1, pg.evaluate("()=>document.querySelectorAll('#tmem-col rect').length"))
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#tmem-rail [data-pred="80-100"]')
    check(f'{tag} part 1: Saved, Next enabled', Z(lang, 'Saved', '已记下') in text(pg, '#tmem-rail') and not any(nexts(pg)))
    # ---- part 2: auto-plays at Medium
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: the step plays on entering, at Medium (R3.9)', st['step'] == 1 and st['playing'] and st['spd'] == 'mid' and st['playT'] < 0.5, st)
    check(f'{tag} part 2: Next waits for the end', all(nexts(pg)) and any(Z(lang, 'finish playing', '播完') in x for x in why(pg)), why(pg))
    dur = pg.evaluate("()=>window.__tmem.consts.DUR")
    check(f'{tag} part 2: Medium plays in 8 s, Fast in 4 s, Slow in 16 s (B6: within a few seconds)', dur == {'slow': 16000, 'mid': 8000, 'fast': 4000}, dur)
    if phone:
        # B2 (one clock): while it plays, the mini result bar shows the same high-water mark as the card, not the start value
        pg.evaluate("()=>{const t=document.getElementById('tmem-vis'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        lv = pg.evaluate("()=>({playing:window.__tmem.state().playing, shown:!document.getElementById('dock').hidden, read:(document.getElementById('dkRead')||{}).textContent||'', pk:document.querySelector('#tmem-metrics [data-k=\"pk\"] .v').textContent})")
        check(f'{tag} part 2: while it plays, the mini result bar shows the card’s value (B2)', lv['shown'] and lv['playing'] and lv['pk'] in lv['read'] and lv['pk'] != '144.5 GB', lv)
    click(pg, '#tmem-player .seg-ctl[data-key="spd"] button[data-v="fast"]')
    if phone:
        click(pg, '#tmem-rail [data-nav="showme"]', 400)
    else:
        check(f'{tag} part 2: Fast ends within about 4 s', wait_end(pg, 5200), state(pg))
    pg.wait_for_timeout(300)
    st = state(pg)
    o = obs(pg)
    check(f'{tag} part 2: done, watched, Next enabled', st['playT'] >= 1 and not st['playing'] and '8b|0|1' in st['watched'] and not any(nexts(pg)), st)
    check(f'{tag} part 2: 168.7 GB, 88.7 GB over, the right answer', '168.7 GB' in o and '88.7 GB' in o and Z(lang, '130 to 200 GB', '130 到 200 GB') in o, o[:300])
    check(f'{tag} part 2: 18 bytes per parameter, 9 times', '2 + 4 + 4 + 4 + 4 = 18' in text(pg, '#tmem-rail').replace('\n', ' ') or pg.evaluate("()=>[...document.querySelectorAll('#tmem-rail .tmem-fx')].some(e=>/2\\s*\\+\\s*4\\s*\\+\\s*4\\s*\\+\\s*4\\s*\\+\\s*4\\s*=\\s*18/.test(e.textContent))"), text(pg, '#tmem-rail')[:200])
    check(f'{tag} part 2: 9 times the weights', Z(lang, '9 times the weights', '9 倍') in o, o[:400])
    check(f'{tag} part 2: red overflow in the column and the out-of-memory marker (R5.11)', vis(pg, '#tmem-col rect.tmem-over') and vis(pg, '#tmem-curve .tmem-oomdot'))
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('tmem-vis'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!!d&&!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent, read:(document.getElementById('dkRead')||{}).textContent}}")
        check(f'{tag} part 2: the dock with a short status and the peak', dk['shown'] and dk['status'] == Z(lang, 'Out of memory', '放不下') and '168.7 GB' in (dk['read'] or ''), dk)
        shot(pg, f'{tag}_p2_dock')
        rules(pg, f'{tag} part 2 dock', lang)
    # R3.20: the optimizer changes the picture's numbers, not its shape: no replay
    click(pg, '#tmem-rail [data-lever="opt:adam8"]')
    st = state(pg)
    check(f'{tag} R3.20: the optimizer lever updates in place, no replay', st['opt'] == 'adam8' and not st['playing'] and st['playT'] >= 1, st)
    check(f'{tag} part 2: 8-bit AdamW 120.5 GB, still does not fit', '120.5 GB' in obs(pg) and '96.4 GB' in obs(pg), obs(pg)[:300])
    check(f'{tag} part 2: with 8-bit AdamW the verdict says part 1 asked about AdamW, at 168.7 GB (A8)', Z(lang, 'Part 1 asked about AdamW, which peaks at 168.7 GB', '第 1 部分问的是 AdamW，它的峰值是 168.7 GB') in obs(pg), obs(pg)[:400])
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#tmem-rail [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} part 2: the optimizer is single choice', pressed == ['false', 'true', 'false'], pressed)
    # the replay control works
    click(pg, '#tmem-player [data-act="play"]', 400)
    st = state(pg)
    check(f'{tag} R3.20: the replay button plays again from the start', st['playing'] and st['playT'] < 0.3, st)
    click(pg, '#tmem-player [data-act="play"]', 200)
    st = state(pg)
    check(f'{tag} R3.20: pressing again pauses', not st['playing'] and 0 < st['playT'] < 1, st)
    scrub_end(pg)
    check(f'{tag} R3.20: the scrubber reaches the end', state(pg)['playT'] >= 1)
    # re-entering part 2 does not replay what was watched
    nav(pg, 'back'); nav(pg, 'next')
    st = state(pg)
    check(f'{tag} R3.20: re-entering part 2 does not replay', st['step'] == 1 and not st['playing'] and st['playT'] >= 1, st)
    # ---- part 3
    nav(pg, 'next')
    check(f'{tag} part 3: Next waits for a change', all(nexts(pg)) and any(Z(lang, 'Change the length', '改一下序列长度') in x for x in why(pg)), why(pg))
    slide(pg, '#tmem-ic-seq-in', 4)
    st = state(pg); o = obs(pg)
    check(f'{tag} part 3: 16,384 tokens: 88.0 GB of activations; no replay; Next enabled', st['seq'] == 16384 and not st['playing'] and '88.0 GB' in o and not any(nexts(pg)), (st, o[:300]))
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4
    nav(pg, 'next')
    check(f'{tag} part 4: Next waits for recomputation', all(nexts(pg)), why(pg))
    click(pg, '#tmem-rail [data-lever="rc:1"]')
    st = state(pg); o = obs(pg)
    check(f'{tag} R3.20: recomputation changes the shape, so it replays', st['rc'] == 1 and st['playing'], st)
    check(f'{tag} part 4: 1.1 GB of activations, 147.7 GB, still does not fit', '147.7 GB' in o and '1.1 GB' in o and Z(lang, 'still does not fit', '还是放不下') in o, o[:300])
    scrub_end(pg)
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for all three tricks', all(nexts(pg)), why(pg))
    click(pg, '#tmem-rail [data-lever="ac:a4x2"]')
    st = state(pg)
    check(f'{tag} R3.20: two micro-batches change the shape, so it replays', (st['mb'], st['acc']) == (4, 2) and st['playing'], st)
    scrub_end(pg)
    click(pg, '#tmem-rail [data-lever="ac:all"]')
    o = obs(pg)
    check(f'{tag} part 5: all three tricks 99.5 GB, the states alone 96.4 GB, Next enabled', '99.5 GB' in o and '96.4 GB' in o and not any(nexts(pg)), o[:400])
    scrub_end(pg)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6
    nav(pg, 'next')
    rt = text(pg, '#tmem-rail')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#tmem-rail .tmem-remlist li').length") == 3
          and rt.count('✗') == 3 and pg.evaluate("()=>!!document.querySelector('#tmem-rail [data-mode-go=\"sandbox\"]')"), rt[-300:])
    for k in ('states', 'rc', 'all'):
        click(pg, f'#tmem-rail [data-rem="{k}"]')
        st = state(pg)
        check(f'{tag} part 6: “show in the chart” {k}', st['remView'] == k and pg.evaluate(f"()=>document.querySelector('#tmem-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true', st)
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    pg.close()
    pg = page(b, w, h, lang)
    nav(pg, 'skip')
    check(f'{tag} part 1: Skip goes on without a prediction', state(pg)['step'] == 1 and state(pg)['pred'] is None)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    lead = pg.inner_text('#tmem-rail .railcard > p.small')
    check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how much memory each GPU needs.', '选一个场景，看看每张卡要放多少显存。'), lead)
    check(f'{tag} scenarios: four, the first pressed on entry', pg.evaluate("()=>document.querySelectorAll('#tmem-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#tmem-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true', state(pg))
    want = {0: ['168.7 GB'], 1: ['99.5 GB', '96.4 GB'], 2: ['176.1 GB', '8.6 GB'], 3: ['1,270.0 GB', '16']}
    for i in range(4):
        click(pg, f'#tmem-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#tmem-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#tmem-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and all(x in ins for x in want[i]), (pressed, ins[:300]))
        if i == 1:
            check(f'{tag} R3.20: a scenario with a new shape replays', state(pg)['playing'], state(pg))
        scrub_end(pg)
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (1, 3) else None)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#training-memory~'), hsh)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the 70B pressed', st['mode'] == 'sandbox' and st['m'] == '70b'
          and pg2.evaluate("()=>document.querySelector('#tmem-rail [data-preset=\"3\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at AdamW, no recomputation, 1 at a time', (st['m'], st['opt'], st['rc'], st['mb'], st['acc']) == ('1b', 'adamw', 0, 1, 32), st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('tmem-controls').classList.contains('locked')") and vis(pg, '#tmem-lock'))
    check(f'{tag} challenge: plain statement and the assumptions', Z(lang, 'Your team has one RTX 4090', '你们只有一张 24 GB 的 RTX 4090') in text(pg, '#tmem-rail') and pg.evaluate("()=>document.querySelectorAll('#tmem-rail .tmem-asm li').length") == 2)
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#tmem-rail [data-act="submit"]')
    r = text(pg, '#tmem-ch-res')
    check(f'{tag} challenge tier 3: not yet first, a hint, Show the fastest plan; no summary yet', r.startswith(Z(lang, 'Not yet', '还没达到目标')) and Z(lang, 'Hint', '提示') in r and vis(pg, '#tmem-ch-res [data-act="reveal"]') and not vis(pg, '#tmem-ch-res .tmem-sum'), r[:200])
    click(pg, '#tmem-ch-res [data-act="reveal"]')
    SUM = """()=>{const r=document.querySelector('#tmem-ch-res'); const rows=[...r.querySelectorAll('.tmem-c3r')];
      return {rows:rows.map(x=>x.dataset.role), chips:rows.map(x=>[...x.querySelectorAll('.tmem-chip')].map(c=>c.textContent+(c.classList.contains('on')?'*':''))),
        bad:rows.map(x=>[...x.querySelectorAll('.tmem-c3m')].map(m=>m.dataset.goal+':'+(m.querySelector('.tmem-c3v').classList.contains('bad')?'red':'ok'))),
        redbar:rows.map(x=>x.querySelectorAll('.tmem-c3o.bad').length), cmp:!!r.querySelector('.cmp'), det:(()=>{const d=r.querySelector('details.tmem-all'); return d?{open:d.open, n:d.querySelectorAll('li').length, sum:d.querySelector('summary').textContent}:null})(),
        same:(r.querySelector('.tmem-same')||{}).textContent||'', gap:(r.querySelector('.tmem-gap')||{}).textContent||'', next:(r.querySelector('.tmem-next')||{}).textContent||'',
        tips:[...r.querySelectorAll('.tmem-c3trk')].every(t=>!t.hasAttribute('data-tipx')&&!t.hasAttribute('tabindex'))}}"""
    sm = pg.evaluate(SUM)
    check(f'{tag} R1 tier 3: your plan beside the fastest, never all 16 plans', sm['rows'] == ['me', 'best'] and not sm['cmp'], sm)
    check(f'{tag} R1 tier 3: only the differing settings, bold on yours; what both share written once', sm['chips'] == [['AdamW*', Z(lang, '1 sequence at once*', '每次 1 条*')], [Z(lang, '8-bit AdamW', '8 位 AdamW'), Z(lang, '2 sequences at once', '每次 2 条')]] and sm['same'] == Z(lang, 'Neither plan recomputes.', '两种方案都不重算。'), sm)
    check(f'{tag} R1 tier 3: what fails is red (memory and time over), the fastest plan has no red', sm['bad'] == [['mem:red', 'time:red'], ['mem:ok', 'time:ok']] and sm['redbar'] == [2, 0], sm)
    check(f'{tag} R1 tier 3: the main gap and the changes, computed', '22.2 GB' in sm['gap'] and Z(lang, 'No single change meets the goal', '只改一项还达不到目标') in sm['next'] and Z(lang, '8-bit AdamW', '8 位 AdamW') in sm['next'], (sm['gap'], sm['next']))
    check(f'{tag} R1: the plans that pass sit behind a closed disclosure (2); no hover on the bars (the owner, 23:37 UTC)', sm['det'] and not sm['det']['open'] and sm['det']['n'] == 2 and sm['det']['sum'] == Z(lang, 'See all settings that pass (2)', '看全部可行的设置（2 种）') and sm['tips'], sm['det'])
    pg.evaluate("()=>document.getElementById('tmem-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(250)
    every(pg, f'{tag} challenge tier 3 revealed', lang, f'{tag}_challenge_t3')
    click(pg, '#tmem-rail .seg-ctl[data-key="cho"] button[data-v="adam8"]')
    click(pg, '#tmem-rail .seg-ctl[data-key="chr"] button[data-v="1"]')
    click(pg, '#tmem-rail .seg-ctl[data-key="chm"] button[data-v="4"]')
    r = text(pg, '#tmem-ch-res'); sm = pg.evaluate(SUM)
    check(f'{tag} challenge tier 2 (8-bit, recompute, 4 at a time): goal met first, but a faster plan', r.startswith(Z(lang, 'Goal met', '达标了')) and Z(lang, 'But a faster plan also meets it', '不过还有更快的方案') in r and '0.91' in r, r[:300])
    check(f'{tag} R1 tier 2: no red, the differing settings bold, the gap names recomputation, no single change', sm['rows'] == ['me', 'best'] and sm['redbar'] == [0, 0] and all(x == 'ok' for row in sm['bad'] for x in [v.split(':')[1] for v in row])
          and sm['chips'][0] == [Z(lang, 'recompute every layer*', '每一层都重算*'), Z(lang, '4 sequences at once*', '每次 4 条*')] and Z(lang, 'Recomputation', '重算') in sm['gap'] and Z(lang, 'No single change makes it faster', '只改一项快不起来') in sm['next'], sm)
    click(pg, '#tmem-rail .seg-ctl[data-key="chr"] button[data-v="0"]')
    click(pg, '#tmem-rail .seg-ctl[data-key="chm"] button[data-v="2"]')
    r = text(pg, '#tmem-ch-res'); sm = pg.evaluate(SUM)
    check(f'{tag} challenge tier 1 (8-bit, no recomputation, 2 at a time): goal met, and the fastest', r.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'this is the fastest', '这是最快的') in r and '0.79' in r and '23.0 GB' in r, r[:300])
    check(f'{tag} R1 tier 1: one row, your plan is the fastest; the list still closed', sm['rows'] == ['mebest'] and len(sm['chips'][0]) == 3 and sm['redbar'] == [0] and sm['det'] and not sm['det']['open'] and not sm['next'], sm)
    pg.evaluate("()=>{const d=document.querySelector('#tmem-ch-res details.tmem-all'); d.open=true}"); pg.wait_for_timeout(150)
    li = pg.evaluate("()=>[...document.querySelectorAll('#tmem-ch-res details.tmem-all li')].map(l=>l.textContent)")
    check(f'{tag} R1: the open list holds the 2 settings that pass, the fastest first and marked as yours', len(li) == 2 and Z(lang, '(fastest)', '（最快）') in li[0] and Z(lang, '(your plan)', '（你的方案）') in li[0] and '0.91' in li[1], li)
    check(f'{tag} challenge: the capability card', vis(pg, '#tmem-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('tmem-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['opt'], st['rc'], st['mb']) == ('adam8', 0, 2) and st['chSubmitted'] and vis(pg, '#tmem-ch-res .tmem-sum'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


def run_tile(b, w, h, lang):
    """item 1 of the owner's review of batch 8: the home tile opens at the 8B of its title; every stop fits its box"""
    tag = f'{w}_{lang}'
    pg = b.new_page(viewport={'width': w, 'height': h})
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang} tile: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(120)
    pg.click('#domStrip .ds-btn[data-world="training"]'); pg.wait_for_timeout(350)
    t = pg.locator('#tile-training-memory'); t.scroll_into_view_if_needed(); pg.wait_for_timeout(200)
    TS = """()=>{const t=document.querySelector('#tile-training-memory'), i=t.querySelector('input[type=range]'), r=t.getBoundingClientRect();
      const over=[...t.querySelectorAll('.tz-tmem-lab,.tz-tmem-capl,.tz-read,.tz-lab')].filter(e=>{const b=e.getBoundingClientRect(); return b.right>r.right+1||b.left<r.left-1||e.scrollWidth>e.clientWidth+1;}).map(e=>e.className);
      return {title:t.querySelector('h3').textContent, out:t.querySelector('#tz-tmem-out').textContent, read:t.querySelector('.tz-read').textContent, max:+i.max, val:+i.value, over}}"""
    st = pg.evaluate(TS)
    title8 = Z(lang, 'An 8B model’s weights take 16 GB', '8B 模型的权重只有 16 GB')
    check(f'{tag} tile (item 1): the title, the slider and the sentence all say 8B', title8 in st['title'] and st['out'] == '8B' and st['read'].startswith(Z(lang, 'An 8B model', '8B 模型')) and '16.1 GB' in st['read'] and '145 GB' in st['read'] and Z(lang, 'about', '约') not in st['read'] and st['max'] == 3 and st['val'] == 2, st)
    hv = pg.evaluate("()=>document.querySelector('#tile-training-memory .tz-tmem-vis').getAttribute('data-tipx')||''")
    check(f'{tag} tile: the chart’s hover says what it shows, with the tile’s numbers and no exact value', Z(lang, '8.03B parameters', '8.03B 个参数') in hv and '16.1 GB' in hv and not re.search(r'\d,\d{3},\d{3}|\d{2,}\.\d\d GB|精确|Exactly', hv), hv)
    t.screenshot(path=str(SH / f'{tag}_tile_8b.png'))
    seen = []
    for v in range(4):
        pg.evaluate("v=>{const i=document.querySelector('#tile-training-memory input[type=range]'); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true}))}", v); pg.wait_for_timeout(120)
        st = pg.evaluate(TS); seen.append(st['out'])
        check(f'{tag} tile stop {st["out"]}: nothing leaves the tile; no overflow', not st['over'] and overflow(pg) <= 0, st)
    check(f'{tag} tile: the stops are 1B, 3B, 8B, 70B', seen == ['1B', '3B', '8B', '70B'], seen)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_tile(b, w, h, lang)
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'training-memory.js').read_text() for d in ('labs', 'teasers') if (P / d / 'training-memory.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["How m and v are stored depends on the optimizer. Pick an optimizer and see how much memory the model states take.", "m 和 v 怎么存，取决于优化器。选一种优化器，看看模型状态要占多少显存。", "Pick one and see how the peak memory changes.", "选一种做法，看看峰值显存怎么变。", "it just takes more passes. Pick a way to split it and see how the peak memory changes.", "只是多跑了几趟。选一种分法，看看峰值显存怎么变。", "Shows the 8B’s step with AdamW and one sequence of 4,096 tokens.", "在图里显示 8B 用 AdamW、一条 4,096 个 token 的序列时的那一步。", "Shows the 8B’s step with every layer recomputed.", "在图里显示 8B 每一层都重算时的那一步。", "Shows the 8B’s step with all three tricks together.", "在图里显示 8B 三个办法一起用时的那一步。", "Training holds four more things per parameter than inference.", "With AdamW the model states are 9 times the BF16 weights. So to tell whether a model can be trained, count 18 bytes per parameter, not 2.", "Each memory trick shrinks one item only.", "Recomputation shrinks only the activations, and 8-bit AdamW only Adam’s m and v. So before picking a trick, look at which item takes the most memory.", "Even with all three tricks on, the 8B’s model states do not fit in one 80 GB GPU. How to split them", "✗ “Halve the batch, halve the memory.” Only the activations and logits halve; the model states stay as they are. So when the model states take most of the memory, halving the batch barely lowers the peak.", "训练时，每个参数比推理多带四样东西。", "用 AdamW 训练时，模型状态是 BF16 权重的 9 倍。所以估算一个模型能不能训，要按每个参数 18 字节算，不能按 2 字节算。", "每种省显存的办法只压其中一项。", "重算只压激活，8 位 AdamW 只压 Adam 的 m 和 v，别的都不动。所以选办法之前，先看清哪一项占的显存最多。", "三个办法全用上，8B 的模型状态在一张 80 GB 的卡上还是放不下。怎么分到多张卡上，见", "✗ “batch 减半，显存就减半。”减半的只有激活和 logits，模型状态一点不变。所以模型状态占大头时，batch 减半，峰值也降不了多少。"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["How m and v are stored depends on the optimizer. Try another one", "m 和 v 怎么存，取决于优化器。换一个优化器看看：", "Pick one of the two", "Shows part 2’s step: AdamW, one sequence of 4,096 tokens.", "在图里显示第 2 部分的那一步：AdamW，一条 4,096 个 token 的序列。", "Shows part 4’s step with every layer recomputed.", "在图里显示第 4 部分每一层都重算的那一步。", "Shows part 5 with all three tricks together.", "在图里显示第 5 部分三个办法一起用的情况。", "In part 4, recomputation took the activations from", "In part 5, all three tricks together still needed", "✗ “Halve the batch, halve the memory.” Only the activations and logits halve. In part 5, going from 8 sequences at once to 4 took the peak only from", "第 4 部分里，重算把激活从", "第 5 部分里，三个办法一起用，还要", "✗ “batch 减半，显存就减半。”减半的只有激活和 logits。第 5 部分里，一次 8 条减到一次 4 条，峰值只从"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
