"""Flow test for the train-capstone lab (CONTRACT §6).

Every guided part and every mode at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1), the writing rules
  on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万), colours (R5.6);
- part 1: the prediction gate with its reason under a greyed-out Next; before the pick the answer (181 days, 4.46M GPU hours, “half a
  year”) appears nowhere, in text, tooltips or the header's popovers (R3.8);
- part 2: the verdict, 181 days and 4.46M GPU hours; Next waits for a second MFU; 30% gives 242 days;
- part 3: no split 1,356 GB, ZeRO-3 alone 90.7 GB (too big), TP 8 × PP 8 + ZeRO-1 18.2 GB (fits); Next waits for a split that fits;
- part 4: MFU 37.5% → recomputation 27.7% → TP 8 + ZeRO-3 20.7% (51% of the step waiting); Next waits for three layouts; the step
  bar's widths follow the plan;
- part 5: 14.8% → 3.5% wasted; Next waits for “every 15 minutes, async”; part 6: the real runs 13.9%, 25.5%, 33.2%, and the cards show
  the plan's own numbers (201 days, 4.93M GPU hours), not part 2's estimate (review B2); part 7: three points, three misconceptions;
- R3.20 for a lab with no replay: picking the option that is already pressed changes nothing (no bump, same bar widths), and Back
  shows the part as it was left, with nothing moving;
- Scenarios: five, single choice, the 70B one pressed on entry; each shows its own numbers; the GPU-hours change from 70B to 8B and
  back is computed from 4.93M and 27.2K, not from 4.93 (review B1); the URL carries the setup and a fresh page restores it; a change
  in Settings releases the scenario buttons; on a phone a one-line task with a link down to the card sits above the stage (S21);
  entering the challenge shows no change badge from the scenarios (S1);
- the challenge: Settings locked, the start does not fit, three tiers (not yet, with the main gap named; met but not the cheapest; met
  and cheapest), the capability card, a language switch keeps the plan; the Connect drill opens;
- the result after Submit (Codex review b9v3, 应修 5), with the review's plan (TP 8 × PP 4, ZeRO-1, every 15 minutes async): a summary
  first, your plan / the recommended plan / one counterexample as bars of days with the 60-day goal line; what they share written once;
  the settings that differ as chips, highlighted where they differ from the recommended plan; days, “18% fewer than yours”, GPU hours
  rounded (610K), with no hover that gives the exact value (the owner's clarification, 2026-10-07); the main gap, the best single change, why the counterexample fails; at 390 px the
  summary fits in one screen; the full list (77 layouts, exact GPU hours) behind “See all plans”, closed by default, your row in view
  when opened; TP 4 instead of 8 updates your row, the stage and the best single change;
- B4 (polish round 5): entering parts 5 and 7 at 1440 × 900, the stage needs no scrolling inside (Chinese with 30 px to spare).
Screenshots: shots/train-capstone/.
Run: python3 tests/train-capstone_flow.py [en|zh] [1440|390] [guided|modes]  (filters for a partial run)"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-capstone.html', '--only', 'train-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'train-capstone'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'train-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'train-capstone'; SH.mkdir(parents=True, exist_ok=True)
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
# the full list behind “See all plans” carries the exact GPU hours on purpose (Codex review b9v3, 应修 5)
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #tcap-mathbody, .num-big, #tcap-drill-a, .tcap-tbl')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/tcap-mb w|tcap-sw w/,[amber]:/^$/,[red]:/status bad|badline|badc|tcap-over|warn|wrong|ex-res|red|over|bad|tcap-sg rd|tcap-sg rs|tcap-sw rd|tcap-sw rs/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#tcap-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())&&!(p==='stroke'&&s.strokeWidth==='0px')) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#tcap-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#tcap-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, dark=False, hash_='train-capstone'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__tcap.state()")
plan = lambda pg, c: pg.evaluate("c=>window.__tcap.plan(c)", c)
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tcap-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tcap-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#tcap-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
widths = lambda pg, sel: pg.evaluate("s=>[...document.querySelectorAll(s+' .tcap-sg')].map(e=>e.style.width)", sel)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__tcap||!window.__tcap.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#tcap-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#tcap-app', '#dock'):
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
    fm = pg.evaluate(FORMULA_H1, '#tcap-app')
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
    ms = pg.evaluate(MSCROLL, '#tcap-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    math_scroll(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#tcap-app .labhead','#tcap-rail','#tcap-stage','#tcap-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#tcap-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#tcap-rail [data-nav="{what}"]', 300)


def lever(pg, box, v, wait=300):
    click(pg, f'#{box} [data-lever="{v}"]', wait)


def same_pick_changes_nothing(pg, tag, box, v, bar):
    """R3.20: picking the option that is already pressed changes nothing: no bump on the result, the same bar widths."""
    pg.evaluate("()=>{const o=document.querySelector('#tcap-rail [data-obs]'); if(o) o.classList.remove('bump');}")
    w0, o0 = widths(pg, bar), obs(pg)
    lever(pg, box, v, 200)
    bump = pg.evaluate("()=>{const o=document.querySelector('#tcap-rail [data-obs]'); return !!o&&o.classList.contains('bump')}")
    check(f'{tag}: picking the pressed option again changes nothing (R3.20)', not bump and widths(pg, bar) == w0 and obs(pg) == o0, (bump, w0, widths(pg, bar)))


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#tcap-app .labtitle') == Z(lang, 'How do you plan the memory, parallelism and time of a 70B run?', '训练一个 70B 模型，怎么规划显存、并行和时间？'), text(pg, '#tcap-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#tcap-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#tcap-app .labmeta'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#tcap-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#tcap-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#tcap-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: the prediction gate; no answer shows (R3.8)
    st = state(pg)
    check(f'{tag} part 1: gate, Next disabled with its reason', st['step'] == 0 and all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
    ba = before_answer(pg)
    leak = [x for x in ('181', '4.46M', 'half a year', '差不多半年', '242', '145') if x in ba]
    check(f'{tag} part 1: the answer appears nowhere before the pick (R3.8)', not leak, leak)
    check(f'{tag} part 1: the machines are drawn, 128 of them', pg.evaluate("()=>document.querySelectorAll('#tcap-cl .tcap-mach').length") == 128)
    check(f'{tag} part 1: 6ND on its own line', vis(pg, '#tcap-rail .step.current .tcap-fx math, #tcap-rail .tcap-fx'))
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#tcap-rail [data-pred="1to3"]')
    st = state(pg)
    check(f'{tag} part 1: the pick is saved, Next enabled', st['pred'] == '1to3' and not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#tcap-rail'), st['pred'])
    # ---- part 2: the total
    nav(pg, 'next')
    o = obs(pg)
    check(f'{tag} part 2: the verdict, 181 days and 4.46M GPU hours', st['pred'] and Z(lang, '“3 to 12 months”', '“3 到 12 个月”') in o and '181' in o and '4.46M' in o, o[:300])
    check(f'{tag} part 2: Next waits for another MFU, with its reason', all(nexts(pg)) and any(Z(lang, 'another MFU', '换一个 MFU') in x for x in why(pg)), why(pg))
    same_pick_changes_nothing(pg, f'{tag} part 2', 'tcap-ic-lv2', '0.4', '#tcap-run')
    lever(pg, 'tcap-ic-lv2', '0.3')
    o = obs(pg)
    check(f'{tag} part 2: 30% gives 242 days; Next enabled', '242' in o and not any(nexts(pg)), o[:300])
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: memory
    nav(pg, 'next')
    mt = text(pg, '#tcap-metrics'); o = obs(pg)
    check(f'{tag} part 3: no split, 1,356 GB, out of memory', '1,356 GB' in mt and 'bad' in pg.evaluate("()=>document.getElementById('tcap-status').className") and all(nexts(pg)), (mt, o[:120]))
    check(f'{tag} part 3: the memory column shows the over-80 GB part', pg.evaluate("()=>!!document.querySelector('#tcap-mem .tcap-mb.over')"))
    lever(pg, 'tcap-ic-lv3', 'z3')
    o = obs(pg)
    check(f'{tag} part 3: ZeRO-3 alone, 90.7 GB, still too big', '90.7 GB' in o and all(nexts(pg)), o[:200])
    lever(pg, 'tcap-ic-lv3', 'tp8pp8z1')
    o = obs(pg); res = pg.evaluate("()=>[...document.querySelectorAll('#tcap-ic-lv3 [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 3: TP 8 × PP 8 + ZeRO-1, 18.2 GB, fits; Next enabled', '18.2 GB' in o and not any(nexts(pg)) and 'ok' in pg.evaluate("()=>document.getElementById('tcap-status').className"), o[:200])
    check(f'{tag} part 3: results stay on the options tried', '1,356 GB' in res[0] and '90.7 GB' in res[1] and res[2] == '' and '18.2 GB' in res[3], res)
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4: where the time goes
    nav(pg, 'next')
    o = obs(pg)
    check(f'{tag} part 4: the start, MFU 37.5%', '37.5%' in o and '37.5%' in text(pg, '#tcap-metrics') and all(nexts(pg)), o[:200])
    w_start = widths(pg, '#tcap-step')
    same_pick_changes_nothing(pg, f'{tag} part 4', 'tcap-ic-lv4', 'start', '#tcap-step')
    lever(pg, 'tcap-ic-lv4', 'rc')
    o = obs(pg)
    check(f'{tag} part 4: recomputation, 37.5% → 27.7%', '27.7%' in o and widths(pg, '#tcap-step') != w_start, o[:200])
    lever(pg, 'tcap-ic-lv4', 'tp8z3')
    o = obs(pg)
    check(f'{tag} part 4: TP 8 + ZeRO-3, 20.7%, 51% of the step waiting for the network; Next enabled', '20.7%' in o and '51%' in o and not any(nexts(pg)), o[:300])
    lever(pg, 'tcap-ic-lv4', 'tp8z1')
    o = obs(pg)
    check(f'{tag} part 4: TP 8 + ZeRO-1, 38.7%, 64.5 GB', '38.7%' in o and '64.5 GB' in o, o[:300])
    lever(pg, 'tcap-ic-lv4', 'start')
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: failures
    nav(pg, 'next')
    o = obs(pg); r3 = plan(pg, {'ivl': 3, 'asy': 0}); r15 = plan(pg, {'ivl': 0.25, 'asy': 1})
    w3 = f"{r3['waste'] * 100:.1f}%"; w15 = f"{r15['waste'] * 100:.1f}%"
    check(f'{tag} part 5: every 3 hours wastes {w3}; Next waits for async', w3 in o and all(nexts(pg)) and any(Z(lang, 'async', '异步') in x for x in why(pg)), o[:200])
    lever(pg, 'tcap-ic-lv5', '15s')
    lever(pg, 'tcap-ic-lv5', '15a')
    o = obs(pg)
    check(f'{tag} part 5: every 15 minutes async, {w3} → {w15}; Next enabled', w3 in o and w15 in o and not any(nexts(pg)), o[:200])
    check(f'{tag} part 5: the run bar has four parts, the failure parts red', pg.evaluate("()=>document.querySelectorAll('#tcap-run .tcap-sg').length") == 4 and pg.evaluate("()=>!!document.querySelector('#tcap-run .tcap-sg.rd')"))
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('tcap-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 5: the mini result bar shows the plan', dk['shown'] and 'MFU' in dk['read'], dk)
        shot(pg, f'{tag}_p5_dock')
        rules(pg, f'{tag} part 5 dock', lang)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: real runs
    nav(pg, 'next')
    rt = text(pg, '#tcap-real')
    check(f'{tag} part 6: real runs 13.9%, 25.5%, 33.2% and the plan', all(x in rt for x in ('13.9%', '25.5%', '33.2%')) and pg.evaluate("()=>document.querySelectorAll('#tcap-real li').length") == 4, rt)
    check(f'{tag} part 6: published layouts listed', pg.evaluate("()=>document.querySelectorAll('#tcap-rail .tcap-al li').length") == 3)
    mt = text(pg, '#tcap-metrics'); stc = pg.evaluate("()=>document.getElementById('tcap-status').textContent")
    check(f'{tag} part 6: the cards show the plan (201 days, 4.93M GPU hours, MFU 37.5%), not the first estimate (B2)', '201' in mt and '4.93M' in mt and '37.5%' in mt and '145' not in mt and stc != Z(lang, 'Estimate', '估算'), (mt, stc))
    mh = pg.evaluate("()=>{const d=document.getElementById('tcap-math'), was=d.open; d.open=true; const t=document.getElementById('tcap-mathbody').innerText; d.open=was; return t}")
    check(f'{tag} part 6: Show the math gives the share formula of this part', '25.5%' in mh and '36.2%' in mh, mh[:200])
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    # ---- part 7: what to remember
    nav(pg, 'next')
    rt = text(pg, '#tcap-rail')
    check(f'{tag} part 7: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#tcap-rail .step.current .tcap-ul > li, #tcap-rail .tcap-ul > li').length") >= 3 and rt.count('✗') == 3
          and pg.evaluate("()=>!!document.querySelector('#tcap-rail [data-mode-go=\"sandbox\"]')"), rt[-400:])
    every(pg, f'{tag} part 7', lang, f'{tag}_p7')
    # ---- the language switch keeps the place; Back shows part 5 as it was left, nothing moving (R3.20)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    check(f'{tag} language switch keeps the part', state(pg)['step'] == 6)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    nav(pg, 'back'); nav(pg, 'back')
    st = state(pg)
    res = pg.evaluate("()=>[...document.querySelectorAll('#tcap-ic-lv5 [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} back in part 5: shown as it was left (async every 15 min, the results of the options tried), Next enabled, nothing moving (R3.20)', st['step'] == 4 and st['lv5'] == '15a' and st['ivl'] == 0.25 and st['asy'] == 1
          and [bool(x) for x in res] == [v in st['tried5'] for v in ('3s', '1s', '15s', '15a')] and not any(nexts(pg)) and not pg.evaluate("()=>{const o=document.querySelector('#tcap-rail [data-obs]'); return !!o&&o.classList.contains('bump')}"), (st, res))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    lead = pg.inner_text('#tcap-rail .railcard > p.small')
    check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how many days the plan takes.', '选一个场景，看看这种计划要训多少天。'), lead)
    pressed = lambda: pg.evaluate("()=>[...document.querySelectorAll('#tcap-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: five, the 70B plan pressed on entry', len(pressed()) == 5 and pressed() == ['false', 'true', 'false', 'false', 'false'], pressed())
    presets = pg.evaluate("()=>window.__tcap.consts.PRESETS")
    for i in range(5):
        click(pg, f'#tcap-rail [data-preset="{i}"]', 400)
        r = plan(pg, {k: v for k, v in presets[i].items() if k != 'id'})
        ins = text(pg, '#tcap-insight'); mt = text(pg, '#tcap-metrics')
        mfu = f"{r['mfu'] * 100:.1f}%"; mem = f"{r['mem'] / 1e9:.1f} GB"
        check(f'{tag} scenario {i + 1}: pressed alone, MFU {mfu}, {mem}', pressed() == ['true' if j == i else 'false' for j in range(5)] and mfu in mt and mem in mt and ins.strip() != '', (pressed(), mt, ins[:160]))
        every(pg, f'{tag} scenario {i + 1}', lang, f'{tag}_scenario{i + 1}')
    # B1: the GPU-hours badge from 70B (4.93M) to 8B (27.2K) and back, computed from the numbers, not from 4.93
    click(pg, '#tcap-rail [data-preset="1"]', 300)
    pg.evaluate("()=>document.querySelector('#tcap-rail [data-preset=\"0\"]').click()"); pg.wait_for_timeout(250)
    d1 = pg.evaluate("()=>document.querySelector('#tcap-metrics [data-k=\"gpuh\"] .d').textContent")
    pg.evaluate("()=>document.querySelector('#tcap-rail [data-preset=\"1\"]').click()"); pg.wait_for_timeout(250)
    d2 = pg.evaluate("()=>document.querySelector('#tcap-metrics [data-k=\"gpuh\"] .d').textContent")
    check(f'{tag} scenarios: GPU hours 4.93M → 27.2K is ▼ −99%, and back is 181 times (B1)', '▼' in d1 and '99%' in d1 and '▲' in d2 and '181' in d2, (d1, d2))
    if w <= 760:
        check(f'{tag} scenarios: a one-line task and a link down to the scenarios above the stage (S21)', vis(pg, '#tcap-phone [data-act="to-task"]') and Z(lang, 'scenario', '场景') in text(pg, '#tcap-phone'), text(pg, '#tcap-phone'))
    click(pg, '#tcap-rail [data-preset="2"]', 300)
    url = pg.url; st = state(pg)
    pg2 = b.new_page(viewport={'width': w, 'height': h}); open_drawers(pg2); pg2.goto(URL); pg2.wait_for_timeout(250)
    pg2.click(f'.langsw button[data-lang="{lang}"]'); pg2.wait_for_timeout(80)
    pg2.evaluate(f"()=>{{location.hash={url.split('#', 1)[1]!r}}}"); pg2.wait_for_timeout(500)
    st2 = pg2.evaluate("()=>window.__tcap.state()")
    check(f'{tag} scenarios: the URL carries the setup, a fresh page restores it', all(st2[k] == st[k] for k in ('model', 'D', 'G', 't', 'p', 'z', 'rc', 'b', 'B', 'net', 'ivl', 'asy', 'node')) and st2['mode'] == 'sandbox', (url, st2))
    pg2.close()
    click(pg, '#tcap-rail [data-preset="1"]', 300)
    click(pg, '#tcap-ctl-ivl button[data-v="1"]', 300)
    check(f'{tag} scenarios: after a change in Settings no scenario is pressed', pressed() == ['false'] * 5 and state(pg)['ivl'] == 1, (pressed(), state(pg)['ivl']))
    click(pg, '#tcap-ctl-G button[data-v="64"]', 300)
    st = state(pg)
    check(f'{tag} settings: 64 GPUs keeps a valid layout', 64 % (st['t'] * st['p']) == 0 and plan(pg, {k: st[k] for k in ('model', 'D', 'G', 't', 'p', 'z', 'rc', 'b', 'B', 'net', 'ivl', 'asy', 'node')}) is not None, st)
    every(pg, f'{tag} sandbox changed', lang, f'{tag}_sandbox_changed')
    # ---- the challenge; a change badge from the scenarios does not carry over (S1)
    pg.evaluate("()=>document.querySelector('#tcap-rail [data-preset=\"2\"]').click()"); pg.wait_for_timeout(150)
    mode(pg, 'challenge')
    stale = pg.evaluate("()=>[...document.querySelectorAll('#tcap-metrics .d')].map(e=>e.textContent).filter(Boolean)")
    check(f'{tag} challenge: no change badge left from the scenarios (S1)', not stale, stale)
    if w <= 760:
        check(f'{tag} challenge: a one-line task and a link down to the task card above the stage (S21)', vis(pg, '#tcap-phone [data-act="to-task"]') and '512' in text(pg, '#tcap-phone'), text(pg, '#tcap-phone'))
    st = state(pg)
    check(f'{tag} challenge: starts with no split; Settings locked', (st['t'], st['p'], st['z'], st['G']) == (1, 1, 0, 512) and not vis(pg, '#tcap-ctl-t-wrap'), st)
    check(f'{tag} challenge: the start does not fit', 'tcap-over' in pg.evaluate("()=>document.getElementById('tcap-ch-g1').className"), text(pg, '#tcap-ch-mem'))
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#tcap-rail [data-act="submit"]', 400)
    r = text(pg, '#tcap-ch-res')
    check(f'{tag} challenge tier 1 (does not fit): red, a hint, no summary yet', Z(lang, 'Not yet', '还没达到') in r and pg.evaluate("()=>!!document.querySelector('#tcap-ch-res .badline')") and not vis(pg, '#tcap-ch-res .tcap-sum'), r[:200])
    check(f'{tag} challenge tier 1: the main gap is named, the largest block on the memory column (应修 5)', Z(lang, 'The largest part is the optimizer state', '最大的一块是优化器状态') in r and vis(pg, '#tcap-rail [data-act="reveal"]'), r[:300])
    allp = pg.evaluate("()=>window.__tcap.chAll()")
    mid = sorted([e for e in allp if e['ok'] and not e['min'] and e['t'] == 8 and e['p'] == 1 and e['rc'] == 0], key=lambda e: e['gpuh'])
    best = sorted([e for e in allp if e['ok']], key=lambda e: e['gpuh'])[0]

    def pick(e):
        for k in ('t', 'p', 'z', 'rc', 'ivl', 'asy'):
            v = e[k]; v = ('0.25' if v == 0.25 else str(int(v)))
            click(pg, f'#tcap-ic-ch{k} button[data-v="{v}"]', 200)
    if mid:
        pick(mid[0]); st = state(pg); r = text(pg, '#tcap-ch-res')
        check(f'{tag} challenge tier 2 (met, not the cheapest)', Z(lang, 'a cheaper plan exists', '还有更省的方案') in r and vis(pg, '#tcap-ch-res .tcap-sum'), (st['ch'], r[:200]))
    # ---- 应修 5 (Codex review b9v3): the review's plan, TP 8 × PP 4, ZeRO-1, no recomputation, every 15 minutes async: a summary first
    pick({'t': 8, 'p': 4, 'z': 1, 'rc': 0, 'ivl': 0.25, 'asy': 1}); r = text(pg, '#tcap-ch-res')
    sm = pg.evaluate("""()=>{const res=document.getElementById('tcap-ch-res'), q=s=>res.querySelector(s), rows=[...res.querySelectorAll('.tcap-c3r')];
      const v=q('.okline,.badline'), d=q('details.tcap-all'), su=d&&d.querySelector('summary');
      return {roles:rows.map(x=>x.dataset.role), on:rows.map(x=>[...x.querySelectorAll('.tcap-chip')].map(c=>(c.classList.contains('on')?'*':'')+c.textContent.trim())),
        days:rows.map(x=>{const b=x.querySelector('.tcap-c3d'); return b?[b.textContent.trim(),b.classList.contains('bad')]:null}),
        subs:rows.map(x=>(x.querySelector('.tcap-c3s')||{}).textContent||''), same:(q('.tcap-same')||{}).textContent||'', gap:(q('.tcap-gap')||{}).textContent||'', next:(q('.tcap-next')||{}).textContent||'',
        why:(q('.tcap-why')||{}).textContent||'', goal:rows.map(x=>!!x.querySelector('.tcap-c3g')), segs:rows.map(x=>x.querySelectorAll('.tcap-c3fill .tcap-sg').length),
        tips:[...res.querySelectorAll('.tcap-c3s .def, .okline .def')].map(e=>e.getAttribute('data-tipx')),
        exactTips:[...res.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('details')).map(e=>e.getAttribute('data-tipx')||'').filter(t=>/精确值是|Exactly|exactly/.test(t)),
        open:d?d.open:null, tbl:d&&su?Math.round(d.getBoundingClientRect().height-su.getBoundingClientRect().height):null,
        h:v&&su?Math.round(su.getBoundingClientRect().bottom-v.getBoundingClientRect().top):null,
        six:[...res.querySelectorAll('.okline, .tcap-sum')].map(e=>e.innerText).join(' ').match(/\\d{3},\\d{3}/g)}}""")
    # the owner's clarification (2026-10-07 23:37 UTC): no hover that expands a shown number to its exact value
    check(f'{tag} 应修 5: the verdict first, GPU hours rounded the way a person says them (610K), no hover with the exact value',
          Z(lang, 'Goal met', '达标了') in r and Z(lang, 'a cheaper plan exists', '还有更省的方案') in r and '610K' in r and not any('612,819' in (t or '') for t in sm['tips']) and not sm['exactTips'] and not sm['six'], (r[:200], sm['tips'], sm['exactTips'], sm['six']))
    check(f'{tag} 应修 5: your plan, the recommended plan and one counterexample, each a bar of days with the 60-day goal line',
          sm['roles'] == ['me', 'best', 'ctr'] and all(sm['goal']) and all(n >= 2 for n in sm['segs']), sm)
    check(f'{tag} 应修 5: what the three share is written once; the settings that differ are chips, highlighted where they differ from the recommended plan',
          Z(lang, 'no recomputation', '都不重算') in sm['same'] and Z(lang, 'every 15 min, async', '都每 15 分钟异步存一次档') in sm['same']
          and sm['on'][0] == ['*' + Z(lang, 'TP 8', '张量并行 8 份'), '*' + Z(lang, 'PP 4', '流水线 4 段'), '*ZeRO\u20111'] and not any(c.startswith('*') for c in sm['on'][1])
          and sm['on'][2] == ['*' + Z(lang, 'TP 8', '张量并行 8 份'), Z(lang, 'no pipeline', '不分段'), 'ZeRO\u20113'], (sm['same'], sm['on']))
    check(f'{tag} 应修 5: days, the saving as a ratio, and the counterexample in red past the goal',
          sm['days'][0] == [Z(lang, '49.9 days', '49.9 天'), False] and sm['days'][1] == [Z(lang, '40.7 days', '40.7 天'), False] and sm['days'][2] == [Z(lang, '95.7 days', '95.7 天'), True]
          and Z(lang, '18% fewer than yours', '比你的方案省 18%') in sm['subs'][1] and '1.2M' in sm['subs'][2], (sm['days'], sm['subs']))
    check(f'{tag} 应修 5: the main gap, the best single change and why the counterexample fails, one sentence each',
          Z(lang, '9.2 more days', '多训 9.2 天') in sm['gap'] and Z(lang, 'tensor-parallel all-reduces', '张量并行的 all-reduce') in sm['gap']
          and Z(lang, 'cutting tensor parallelism 4 ways instead of 8', '把张量并行从 8 份改成 4 份') in sm['next'] and '6%' in sm['next']
          and '51%' in sm['why'] and Z(lang, 'data-parallel sync', '数据并行同步') in sm['why'], (sm['gap'], sm['next'], sm['why']))
    check(f'{tag} 应修 5: the full list is behind “See all plans”, closed by default', sm['open'] is False and sm['tbl'] is not None and sm['tbl'] < 24 and Z(lang, 'See all plans (77 layouts that fit)', '看全部方案（77 种放得下的切法）') in r, (sm['open'], sm['tbl']))
    if w <= 760:
        check(f'{tag} 应修 5: at {w} px the summary, from the verdict to “See all plans”, fits in one screen ({h} px)', sm['h'] is not None and sm['h'] <= h, sm['h'])
    every(pg, f'{tag} challenge summary', lang, f'{tag}_challenge_summary')
    pg.evaluate("()=>{const d=document.querySelector('#tcap-ch-res details.tcap-all'); d.open=true;}"); pg.wait_for_timeout(300)
    al = pg.evaluate("""()=>{const d=document.querySelector('#tcap-ch-res details.tcap-all'), w=d.querySelector('.tcap-tblw'), tr=d.querySelector('tr[data-me]'), a=w.getBoundingClientRect(), b=tr?tr.getBoundingClientRect():null;
      return {n:d.querySelectorAll('tbody tr').length, me:tr?tr.innerText.replace(/\\s+/g,' ').trim():'', seen:!!b&&b.top>=a.top-1&&b.bottom<=a.bottom+1, first:d.querySelector('tbody tr').innerText.replace(/\\s+/g,' ').trim(), wide:w.scrollWidth>w.clientWidth+1}}""")
    check(f'{tag} 应修 5: opened, the list has every layout that fits (77), the cheapest first, with exact GPU hours; your row is bold and scrolled into view; no sideways scrolling',
          al['n'] == 77 and al['first'].startswith('2 1 3') and '499,925' in al['first'] and '612,819' in al['me'] and al['seen'] and not al['wide'], al)
    every(pg, f'{tag} challenge all plans', lang, f'{tag}_challenge_all')
    # the reviewer then tried TP 4: the summary's first row and the stage follow the choice; the open list stays open
    click(pg, '#tcap-ic-cht button[data-v="4"]', 300)
    t4 = pg.evaluate("""()=>{const r=document.querySelector('#tcap-ch-res .tcap-c3r[data-role="me"]'); return {chips:[...r.querySelectorAll('.tcap-chip')].map(c=>c.textContent.trim()), d:r.querySelector('.tcap-c3d').textContent.trim(),
      open:document.querySelector('#tcap-ch-res details.tcap-all').open, sub:document.getElementById('tcap-st-sub').textContent, next:document.querySelector('#tcap-ch-res .tcap-next').textContent}}""")
    check(f'{tag} 应修 5: TP 4 instead of 8 updates your row (47.0 days), the stage and the best single change; the full list stays open',
          Z(lang, 'TP 4', '张量并行 4 份') in t4['chips'] and t4['d'] == Z(lang, '47.0 days', '47.0 天') and Z(lang, 'TP 4', '张量并行 4 份') in t4['sub'] and t4['open']
          and Z(lang, 'using 2 pipeline stages instead of 4', '把流水线从 4 段改成 2 段') in t4['next'], t4)
    pg.evaluate("()=>{const d=document.querySelector('#tcap-ch-res details.tcap-all'); d.open=false;}"); pg.wait_for_timeout(150)
    pick(best); st = state(pg); r = text(pg, '#tcap-ch-res')
    check(f'{tag} challenge tier 3 (goal met, cheapest): TP {best["t"]} × PP {best["p"]}', Z(lang, 'Goal met', '达成目标') in r and f"{best['days']:.1f}" in r and '500K' in r, (st['ch'], r[:200]))
    roles = pg.evaluate("()=>[...document.querySelectorAll('#tcap-ch-res .tcap-c3r')].map(x=>x.dataset.role+': '+x.querySelector('.tcap-c3t').textContent.trim())")
    check(f'{tag} challenge: the summary shows your plan as the recommended one, and the counterexample', roles == ['mebest: ' + Z(lang, 'Your plan, the recommended one', '你的方案，就是推荐方案'), 'ctr: ' + Z(lang, 'Counterexample', '反例')], roles)
    check(f'{tag} challenge: the capability card', vis(pg, '#tcap-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('tcap-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['ch']['t'], st['ch']['p']) == (best['t'], best['p']) and st['chSubmitted'] and vis(pg, '#tcap-ch-res .tcap-sum'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    pg.evaluate("()=>{const b=document.querySelector('[data-act=\"drill-reveal\"]'); if(b) b.click();}"); pg.wait_for_timeout(250)
    check(f'{tag} connect: the drill answer opens', vis(pg, '#tcap-drill-a'))
    every(pg, f'{tag} connect', lang)
    # 应修 5, as the reviewer did it: a fresh challenge, the review's plan, then Submit; the verdict, the main gap and the next step are on
    # the screen at once, under the top bar and the mini result bar (the targeted check: in English the summary is taller than the space)
    mode(pg, 'guided'); mode(pg, 'challenge')
    pick({'t': 8, 'p': 4, 'z': 1, 'rc': 0, 'ivl': 0.25, 'asy': 1})
    click(pg, '#tcap-rail [data-act="submit"]', 600)
    one = pg.evaluate("""()=>{const res=document.getElementById('tcap-ch-res'), bar=document.querySelector('.topbar'), dk=document.getElementById('dock'), R=s=>res.querySelector(s).getBoundingClientRect();
      const top=bar.getBoundingClientRect().bottom+(dk&&!dk.hidden&&dk.getClientRects().length?dk.getBoundingClientRect().height:0);
      return {top:Math.round(top), verdict:Math.round(R('.okline').top), gap:Math.round(R('.tcap-gap').bottom), next:Math.round(R('.tcap-next').bottom), h:innerHeight}}""")
    check(f'{tag} 应修 5: right after Submit, the verdict, the main gap and the next step share one screen', one['verdict'] >= one['top'] - 1 and one['next'] <= one['h'], one)
    mode(pg, 'guided')
    check(f'{tag} back to Guided', state(pg)['mode'] == 'guided')
    every(pg, f'{tag} guided again', lang)
    pg.close()


# B4 (polish round 5, reviews/r5-fit-roots.md): entering parts 5 and 7 at 1440 × 900 as the polish probe stage.py does (Next when it
# is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare, English with none
# (13 px in English before this round). Spare = the stage's max-height minus what its content and borders need.
# Its top below the top bar: below what is left of the bar on the screen (on a desktop lab page it scrolls away since refit step 2).
FIT_ENTRY = """()=>{const s=document.getElementById('tcap-stage'), r=s.getBoundingClientRect(), tb=document.querySelector('.topbar'); return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb?Math.max(0,Math.round(tb.getBoundingClientRect().bottom)):0}}"""


def fit_page(b, lang):
    """the page as the polish probe stage.py opens it: window.__aspPrimer set (without it, automation gets the older tests' compat layout,
    whose step list also holds the settings panel), the guided parts picked in the step list, the real clock"""
    pg = b.new_page(viewport={'width': 1440, 'height': 900}); pg.add_init_script('window.__aspPrimer=true')
    pg.on('pageerror', lambda e: errs.append(f'fit 1440 {lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='train-capstone'}"); pg.wait_for_timeout(500)
    g = pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if g.count(): g.first.click(); pg.wait_for_timeout(700)
    return pg


def fit_entry(b, lang, parts=(5, 7,)):
    pg = fit_page(b, lang)
    for k in range(2, max(parts) + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#tcap-rail .steps>.step:nth-child({k}) .step-h').click()")
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
            if 'modes' not in ARGS: run_guided(b, w, h, lang)
            if 'guided' not in ARGS: run_modes(b, w, h, lang)
    if FULL:
        run_guided(b, 1440, 900, 'zh', dark=True)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'train-capstone.js').read_text() for d in ('labs', 'teasers') if (P / d / 'train-capstone.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["each H100’s peak is 989.5 TFLOP/s. Pick an MFU and see how many days the run takes.", "每张 H100 的峰值是 989.5 TFLOP/s。选一个 MFU，看看要训多少天。", "Pick a layout and see how much memory each GPU needs.", "选一种切法，看看每张卡要放多少显存。", "it costs 8N operations per token instead of 6N. Pick a layout and see the MFU of one step.", "代价是每个 token 要做 8N 次运算，而不是 6N。选一种切法，看看一步的 MFU 是多少。", "R the restart time. Pick a way to save and see how much of the time is wasted.", "R 是重启要多久。选一种存档方式，看看白费了多少时间。", "A layout that fits, without recomputation", "The training time is inversely proportional to the MFU: if the MFU drops from 40% to 30%, the run takes a third longer.", "Fit first, then compare speed.", "ZeRO splits only the model states, not the activations: those need tensor parallelism or recomputation. Usually several layouts fit, and their speeds can differ a lot.", "More machines, more failures.", "The job’s time between failures is one machine’s divided by the number of machines. So the more machines, the more often to save, and asynchronously, so that the saves do not hold up training.", "✗ “If it fits in memory, the layout is fine.” Several layouts can all fit while the best reaches about", "times the MFU of the worst: waiting for transfers and idling in the pipeline pull the MFU down.", "训练时间和 MFU 成反比。MFU 从四成降到三成，天数就多三分之一。", "ZeRO 只分模型状态，分不了激活。激活要靠张量并行或重算来压。放得下的切法常常不止一种，它们的快慢可以差很多。", "整个任务多久出一次故障，等于一台机器的故障间隔除以机器数。所以机器越多，越要勤存档，而且要用异步存档，存档才不拖慢训练。", "✗ “显存放得下就行。”几种切法都放得下时，MFU 最高的可以是最低的约", "倍。等通信、流水线空转，都会把 MFU 拉低。", "This is the guided plan, TP 8 × PP 8 with ZeRO-1, saving asynchronously every 15 minutes.", "这是分步引导里的计划：张量并行 8 份 × 流水线 8 段 + ZeRO-1，每 15 分钟异步存一次档。", "but the layouts without it stay far below 100% too.", "可不重算的切法，MFU 也远到不了 100%", "Fixed in this part: TP 8 × PP 8 with ZeRO-1, and a machine failure every 64 days. Pick how to save in the card.", "这一部分固定：张量并行 8 份 × 流水线 8 段 + ZeRO-1，每台机器 64 天坏一次。在卡片里选怎么存档。"]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["One at a time; they do not stack", "一次只能选一个，不会叠加：", "each H100’s peak is 989.5 TFLOP/s. Try another MFU.", "每张 H100 的峰值是 989.5 TFLOP/s。换一个 MFU 试试，", "Start from no split and try a few.", "从“不切”出发，试几种切法，", "it costs 8N operations per token instead of 6N. Start from part 3’s layout.", "代价是每个 token 要做 8N 次运算，而不是 6N。从第 3 部分的切法出发，", "The layout from part 3", "In part 2, 70B on 15T tokens with 1,024 GPUs at 40% takes about", "In part 3, ZeRO-3 alone did not fit: the model states were split, but one micro-batch’s activations were still", ". Several layouts fit, and in part 4 their MFU ranged from", "In part 5, 128 machines lose one every 12 hours. Saving every 15 minutes asynchronously cut the wasted time from", "✗ “If it fits in memory, the layout is fine.” In part 4, several layouts all fit, yet the best reached about", "第 2 部分里，70B 喂 15T 个 token，1,024 张卡按四成算，要训约", "第 3 部分里，只用 ZeRO-3 放不下：模型状态分掉了，一个 micro-batch 的激活还有", "。放得下的切法有好几种，第 4 部分里它们的 MFU 从", "第 5 部分里，128 台机器平均 12 小时坏一台。改成每 15 分钟异步存一次，白费的时间从", "✗ “显存放得下就行。”第 4 部分里，几种切法都放得下，MFU 最高的却是最低的约", "This is the guided plan with the best saving from part 5.", "这是分步引导里的计划，存档用第 5 部分里最好的办法。", "but the plans in part 4 without it stay far below 100% too.", "可第 4 部分里不重算的切法，MFU 也远到不了 100%", "Fixed in this part: the plan of part 3, a machine failure every 64 days. Pick how to save in the card.", "这一部分固定：第 3 部分的计划，每台机器 64 天坏一次。在卡片里选怎么存档。"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
