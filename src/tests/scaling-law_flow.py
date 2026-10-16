"""Flow test for the scaling-law lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (plus one dark run when run in full):
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- part 1: the prediction gate with the kit's reason under a greyed-out Next; the three plans' losses appear nowhere before the
  prediction (R3.8); “Saved” after a pick; Skip goes on without one;
- part 2: Next waits for a budget move; 10²⁴ FLOPs = about 29,000 H100 GPU-days = 1,024 H100s for about 29 days;
- part 3: the loss-terms bar instead of the chart; Next waits for all three plans; 70B lowest (1.973), the verdict with the right answer;
- part 4: Next waits for the lowest point; 10²² FLOPs: 9.06B parameters, 184B tokens, about 20 per parameter; 17 to 22 over 10²¹–10²⁵;
- part 5: Next waits for 100T served tokens; the cheapest size moves down; part 6: three points with “show in the chart”, three
  misconceptions, See the scenarios;
- R3.20: a budget change draws one short transition; the same budget again, a part change or a language switch draw none; with
  reduced motion the curve jumps. The lab has no animation to replay, so it has no replay control;
- Scenarios: four, single choice, each with its result; the URL carries the setup and a fresh page restores it; a change in Settings
  un-presses the scenario;
- the challenge (the owner's review b8, R1): Settings locked, the three feedback tiers; the result is the verdict, then your plan beside
  the cheapest plan and one counterexample (the training-only best), shared conditions written once, only the differing settings as
  chips, a failing loss in red, the one change that helps most; no list of every plan, only the 13 that pass behind a disclosure
  closed by default; the capability card with new-situation question 2 stating that compute is unlimited; a language switch keeps
  the plan and the result;
- the owner's review b8: every slider (parts 2, 3, 4 and Settings) dragged with the real mouse: the fill follows the thumb, the number
  and the picture change; one slider moved with the keyboard; part 3 names each term before its symbols (N and D in the denominators,
  α and β explained); Show the math puts each term on its own line, the symbols left and the same shape in numbers right;
- the mini result bar at 390 once the chart scrolls away; the writing rules on every visible text node and tooltip, colours (R5.6);
- the owner's clarification (2026-10-07 23:37 UTC): no hover outside Show the math holds “精确值是” / “Exactly”, and no .num-big has a hover.
Screenshots: shots/scaling-law/.
Run: python3 tests/scaling-law_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/scaling-law.html', '--only', 'scaling-law'], check=True, cwd=P, stdout=subprocess.DEVNULL)
OUT = P / 'out' / 'scaling-law'; OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'wrapped-flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'scaling-law.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-flow.html')
SH = P / 'shots' / 'scaling-law'; SH.mkdir(parents=True, exist_ok=True)
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
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #sl-mathbody, .num-big, #sl-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|sl-over|sl-me bad|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#sl-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
EXACT = """()=>[...document.querySelectorAll('[data-tipx],[data-tip]')].filter(e=>!e.closest('#sl-mathbody')).map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||'').filter(t=>/精确值是|Exactly/.test(t)).map(t=>t.slice(0,90)).concat([...document.querySelectorAll('.num-big')].filter(e=>e.matches('[data-tipx],[tabindex],.def')).map(e=>'num-big with a hover: '+e.textContent))"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#sl-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#sl-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


def page(b, w, h, lang, dark=False, hash_='scaling-law', reduce=False):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduce else 'no-preference')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__sl.state()")
hook = lambda pg, js: pg.evaluate(f"()=>{{const H=window.__sl; return {js}}}")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-next-why]')].filter(e=>!e.hidden&&e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#sl-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__sl||(!window.__sl.state().stageBusy&&!window.__sl.state().tweening)", timeout=4000)
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


SLID = """s=>{const i=document.querySelector(s), w=i.closest('.ctl,.inline-ctl'); return {v:+i.value, p:(i.value-i.min)/(i.max-i.min)*100, fill:parseFloat(i.style.getPropertyValue('--fill')), out:w.querySelector('output').textContent}}"""
OUTF = {'lc': "H.fmt.fF(Math.pow(10,20+v*0.25))+' FLOP'", 'ln': "H.fmt.fP(Math.pow(10,8+v*0.05))", 'ld': "H.fmt.fTok(Math.pow(10,10+v*0.05))", 'u': "v+'%'"}
PIC = {'chart': "()=>document.getElementById('sl-chart').innerHTML", 'terms': "()=>document.getElementById('sl-terms').innerText", 'metrics': "()=>document.getElementById('sl-metrics').innerText"}


def drag(pg, sel, frac):
    """a real mouse drag (the owner's review b8): press on the thumb, move along the track to frac, release"""
    v0 = pg.evaluate(SLID, sel)['p'] / 100
    for _ in range(4):   # wait until the page stops scrolling and the thumb is the element under the pointer
        pg.evaluate("s=>document.querySelector(s).scrollIntoView({block:'center'})", sel); pg.wait_for_timeout(250); settle(pg, 1500)
        bb = pg.locator(sel).bounding_box(); y = bb['y'] + bb['height'] / 2; x0 = bb['x'] + 8 + (bb['width'] - 16) * v0
        if pg.evaluate("([s,x,y])=>document.elementFromPoint(x,y)===document.querySelector(s)", [sel, x0, y]): break
    pg.mouse.move(x0, y); pg.mouse.down()
    pg.mouse.move(bb['x'] + 8 + (bb['width'] - 16) * frac, y, steps=10); pg.mouse.up(); pg.wait_for_timeout(300); settle(pg)


def follows(pg, tag, sel, key, frac, pic):
    b0, p0 = pg.evaluate(SLID, sel), pg.evaluate(PIC[pic])
    drag(pg, sel, frac)
    b1, p1 = pg.evaluate(SLID, sel), pg.evaluate(PIC[pic])
    want = pg.evaluate(f"v=>{{const H=window.__sl; return {OUTF[key]}}}", b1['v'])
    check(f'{tag}: a mouse drag on {sel} moves the fill with the thumb and changes the number and the {pic}', b1['v'] != b0['v'] and abs(b1['fill'] - b1['p']) < 0.01 and b1['out'] == want and b1['out'] != b0['out'] and p1 != p0, (b0, b1, want, p1 == p0))


def keyed(pg, tag, sel, key, pic):
    b0, p0 = pg.evaluate(SLID, sel), pg.evaluate(PIC[pic])
    pg.focus(sel); pg.keyboard.press('ArrowLeft' if b0['p'] > 50 else 'ArrowRight'); pg.wait_for_timeout(250); settle(pg)
    b1, p1 = pg.evaluate(SLID, sel), pg.evaluate(PIC[pic])
    want = pg.evaluate(f"v=>{{const H=window.__sl; return {OUTF[key]}}}", b1['v'])
    check(f'{tag}: the keyboard on {sel} moves the fill and changes the number and the {pic}', abs(b1['v'] - b0['v']) == 1 and abs(b1['fill'] - b1['p']) < 0.01 and b1['out'] == want and p1 != p0, (b0, b1, want))


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#sl-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def rules(pg, where, lang):
    for sel in ('#sl-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(EXACT): rule_hits.append(f'{where}: exact-value hover (the owner, 2026-10-07 23:37 UTC): {h}')


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
    fm = pg.evaluate(FORMULA_H1, '#sl-app')
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
    ms = pg.evaluate(MSCROLL, '#sl-app')
    check(f'{where}: no row of Show the math scrolls sideways at {pg.viewport_size["width"]} px (the owner, 2026-10-09; R6.37)', not ms, ms[:3])


def every(pg, where, lang, full_shot=None):
    formulas_h1(pg, where)
    math_scroll(pg, where)
    check(f'{where}: no overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, where, lang); fit(pg, where)
    if full_shot: shot(pg, full_shot, full=pg.viewport_size['width'] <= 760)


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#sl-app .labhead','#sl-rail','#sl-stage','#sl-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#sl-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def nav(pg, what):
    click(pg, f'#sl-rail [data-nav="{what}"]', 300)


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#sl-app .labtitle') == Z(lang, 'How big a model should a compute budget buy?', '一笔算力预算，该训多大的模型？'), text(pg, '#sl-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#sl-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#sl-app .labmeta'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#sl-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#sl-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#sl-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # ---- part 1: mission and prediction
    st = state(pg)
    check(f'{tag} part 1: guided part 1, the plans’ budget', st['mode'] == 'guided' and st['step'] == 0 and abs(st['C'] / 5.88e23 - 1) < 1e-9, st)
    check(f'{tag} part 1: Next disabled, with the kit’s reason under it', nexts(pg) and all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。')}, (nexts(pg), why(pg)))
    ba = before_answer(pg)
    check(f'{tag} part 1: the plans’ losses appear nowhere before the prediction (R3.8)', not any(x in ba for x in ('1.973', '1.992', '1.994')), [l for l in ba.split('\n') if '1.97' in l or '1.99' in l][:3])
    check(f'{tag} part 1: the three plans in the text and as options', pg.evaluate("()=>document.querySelectorAll('#sl-rail [data-pred]').length") == 3 and pg.evaluate("()=>document.querySelectorAll('#sl-rail .sl-plans li').length") == 3
          and all(x in text(pg, '#sl-rail') for x in ('280B', '350B', '70B', '1.4T', '17.5B', '5.6T')), text(pg, '#sl-rail')[:400])
    check(f'{tag} part 1: the three plans drawn on one diagonal', pg.evaluate("()=>document.querySelectorAll('#sl-chart .sl-plan').length") == 3)
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#sl-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), ax=document.getElementById('sl-chart');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(ax); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, ax:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['ax'] is not None and fs['ax'] < fs['vh'] - 60, fs)
        shot(pg, f'{tag}_first_screen')
    every(pg, f'{tag} part 1', lang, f'{tag}_p1')
    click(pg, '#sl-rail [data-pred="big"]')
    check(f'{tag} part 1: Saved after a pick, Next enabled, the reason gone', Z(lang, 'Saved.', '已记下。') in text(pg, '#sl-rail') and not any(nexts(pg)) and why(pg) == [], (nexts(pg), why(pg)))
    click(pg, '#sl-rail [data-pred="small"]')
    check(f'{tag} part 1: the prediction can still change', state(pg)['pred'] == 'small')
    # ---- part 2: 6ND
    t0 = state(pg)['tweens']
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 2: Next waits for a budget move, with the reason', st['step'] == 1 and all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Move the budget to continue.', '拖动一次预算，才能继续。')}, why(pg))
    check(f'{tag} part 2 (R3.20): entering a part draws no transition', st['tweens'] == t0, (st['tweens'], t0))
    check(f'{tag} part 2: the formula on its own line', pg.evaluate("()=>document.querySelectorAll('#sl-rail .sl-fx math').length") >= 1)
    slide(pg, '#sl-ic-lc-in', 16)
    o = obs(pg); st = state(pg)
    check(f'{tag} part 2: 10²⁴ FLOPs = about 29,000 H100 GPU-days = 1,024 H100s for about 29 days', Z(lang, '10²⁴ FLOPs take about 29,000 H100 GPU-days: 1,024 H100s for about 29 days', '10²⁴ FLOP 在 H100 上约要 29,000 个GPU 天，相当于 1,024 张 H100 跑约 29 天').replace('个GPU', '个 GPU') in o.replace('个GPU', '个 GPU'), o[:300])
    check(f'{tag} part 2: the plans sit on one diagonal of 5.9×10²³ FLOPs', '5.9×10²³' in o and not any(nexts(pg)), o[-200:])
    check(f'{tag} part 2 (R3.20): a budget change draws one short transition', st['tweens'] == t0 + 1, (st['tweens'], t0))
    ba2 = pg.evaluate("""()=>{const t=[]; for(const s of ['#sl-rail','#sl-stage','#sl-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText); e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}} return t.join('\\n')}""")
    check(f'{tag} part 2: no dot, no best size, no ratio, no loss before part 3 (R3.8)', not vis(pg, '#sl-chart .sl-me') and not any(x in ba2 for x in ('1.973', '1.992', '1.994', '20.3', 'Tokens per parameter', '每个参数的 token'))
          and Z(lang, 'Losses in part 3', 'loss 在第 3 部分揭晓') == text(pg, '#sl-status') and 'MFU' in o, [l for l in ba2.split('\n') if 'per parameter' in l or '每个参数' in l][:3])
    follows(pg, f'{tag} part 2', '#sl-ic-lc-in', 'lc', 0.3, 'chart')
    every(pg, f'{tag} part 2', lang, f'{tag}_p2')
    # ---- part 3: the terms
    nav(pg, 'next')
    check(f'{tag} part 3: the loss-terms bar replaces the chart', vis(pg, '#sl-terms .sl-trows') and not vis(pg, '#sl-chart'))
    check(f'{tag} part 3: Next waits for the three plans, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Put all three plans into the formula to continue.', '三种方案都代进公式，才能继续。')}, why(pg))
    click(pg, '#sl-rail [data-lever="big"]')
    o = obs(pg)
    check(f'{tag} part 3: 280B alone: loss 1.992, pick the other two', Z(lang, '280B: loss 1.992', '280B 的 loss 是 1.992') in o and all(nexts(pg)), o[:200])
    click(pg, '#sl-rail [data-lever="small"]'); click(pg, '#sl-rail [data-lever="mid"]')
    o = obs(pg)
    check(f'{tag} part 3: 70B lowest at 1.973; 280B 1.992; 17.5B 1.994; Next enabled', Z(lang, 'The 70B plan has the lowest loss, 1.973', '70B 的方案 loss 最低，是 1.973') in o and '1.992' in o and '1.994' in o and not any(nexts(pg)), o[:300])
    check(f'{tag} part 3: the verdict with the right answer', Z(lang, 'The right answer is “70B parameters, 1.4T tokens”', '正确答案是“70B 个参数，1.4T 个 token”') in o, o[:400])
    res = pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} part 3: single choice, the results on their options', pressed == ['false', 'true', 'false'] and res == ['loss 1.992', 'loss 1.973', 'loss 1.994'], (pressed, res))
    tsum = text(pg, '#sl-terms .sl-tsum')
    check(f'{tag} part 3: the bar adds up: 1.817 + 0.082 + 0.075 = 1.973', '1.817 + 0.081 + 0.075 = 1.973' in tsum, tsum)
    rt3 = text(pg, '#sl-rail')
    check(f'{tag} part 3: each term named before its symbols; N and D in the denominators; α and β explained (owner b8, item 2)',
          all(x in rt3 for x in Z(lang, ['What no model removes is the first term, E', 'The cost of a small model is the second term, ', 'N, the number of parameters, is in its denominator',
                                            'The cost of little data is the third term, ', 'D, the number of training tokens, is in its denominator', 'The exponents α and β set how fast the two costs fall', 'α is about 0.35 and β about 0.37'],
                                     ['谁也去不掉的部分是第一项 E', '模型小的代价是第二项 ', 'N 是参数量，在分母上', '数据少的代价是第三项 ', 'D 是训练用的 token 数，也在分母上', 'α 和 β 是两个指数，管这两项降得多快', 'α 约是 0.35，β 约是 0.37'])), rt3[:900])
    fr = pg.evaluate("()=>[...document.querySelectorAll('#sl-rail .sl-terml math')].map(m=>[!!m.querySelector('mfrac'),!!m.querySelector('msup'),m.textContent.replace(/\\s+/g,'')])")
    check(f'{tag} part 3: the two cost terms are typeset fractions A/N^α and B/D^β, not text (H1, the owner 2026-10-09)', fr == [[True, True, 'ANα'], [True, True, 'BDβ']], fr)
    pg.evaluate("()=>{document.getElementById('sl-math').open=true}"); pg.wait_for_timeout(150)
    ml = pg.evaluate("()=>[...document.querySelectorAll('#sl-mathbody .sl-fx')].map(e=>e.textContent.replace(/\\s+/g,''))")
    check(f'{tag} part 3: Show the math, one line per term: symbols left, the same shape in numbers right (owner b8, item 7)',
          'E=1.81686' in ml and 'ANα=482.00572(7×1010)0.34781=0.081' in ml and 'BDβ=2085.4342(1.4×1012)0.36585=0.075' in ml and 'L=1.817+0.081+0.075=1.973' in ml, ml)
    pg.evaluate("()=>{document.getElementById('sl-math').open=false}")
    follows(pg, f'{tag} part 3', '#sl-st-ln-in', 'ln', 0.25, 'terms')
    follows(pg, f'{tag} part 3', '#sl-st-ld-in', 'ld', 0.75, 'terms')
    every(pg, f'{tag} part 3', lang, f'{tag}_p3')
    # ---- part 4: the iso-FLOP curve
    nav(pg, 'next')
    st = state(pg)
    check(f'{tag} part 4: Next waits for the lowest point, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Find the lowest point of the curve to continue.', '找到曲线的最低点，才能继续。')} and st['view'] == 'iso', why(pg))
    t1 = st['tweens']
    follows(pg, f'{tag} part 4', '#sl-ic-ln-in', 'ln', 0.8, 'chart')
    keyed(pg, f'{tag} part 4', '#sl-ic-ln-in', 'ln', 'chart')
    slide(pg, '#sl-ic-ln-in', 39)
    o = obs(pg)
    check(f'{tag} part 4: 10²²: 9.06B parameters on 184B tokens, about 20 per parameter; 17 to 22 over the budgets', Z(lang, 'the best model has about 9.06B parameters and 184B tokens: about 20 tokens per parameter', '最好的模型约 9.06B 个参数、184B 个 token：每个参数约 20 个 token') in o
          and Z(lang, 'between 17 and 22 tokens per parameter', '每个参数 17 到 22 个 token') in o and not any(nexts(pg)), o[:400])
    check(f'{tag} part 4 (R3.20): a size change draws no transition', state(pg)['tweens'] == t1)
    click(pg, '#sl-rail #sl-ic-bud button[data-v="23"]', 50)
    st = state(pg)
    check(f'{tag} part 4 (R3.20): a new budget draws one transition, about 350 ms', st['tweens'] == t1 + 1, (st['tweens'], t1))
    click(pg, '#sl-rail #sl-ic-bud button[data-v="23"]', 50)
    check(f'{tag} part 4 (R3.20): the same budget again draws nothing', state(pg)['tweens'] == t1 + 1)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400); st = state(pg)
    check(f'{tag} part 4 (R3.20): a language switch keeps the place and draws nothing', st['step'] == 3 and abs(st['C'] / 1e23 - 1) < 1e-9 and st['tweens'] == t1 + 1, st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    check(f'{tag} part 4: the curves are labelled at their lowest points and the reader’s dot is drawn', pg.evaluate("()=>document.querySelectorAll('#sl-chart .sl-isol').length") >= 5 and vis(pg, '#sl-chart .sl-me'))
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('sl-metrics'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+80)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 4: the mini result bar with a short status and the result', dk['shown'] and dk['status'] and 'loss' in dk['read'], dk)
        shot(pg, f'{tag}_p4_dock'); rules(pg, f'{tag} part 4 dock', lang)
    every(pg, f'{tag} part 4', lang, f'{tag}_p4')
    # ---- part 5: serving
    nav(pg, 'next')
    check(f'{tag} part 5: Next waits for 100T, with the reason', all(nexts(pg)) and set(why(pg)) == {Z(lang, 'Pick 100T served tokens to continue.', '选服务 100T 个 token，才能继续。')}, why(pg))
    o = obs(pg); b0 = hook(pg, "H.fmt.fP(H.totBest(H.L5(),0).N)")
    check(f'{tag} part 5: training only: the cheapest size, and Llama 3 8B takes about 5.5 times the compute', b0 in o and '5.5' in o, o[:300])
    click(pg, '#sl-rail #sl-ic-T button[data-v="3"]')
    o = obs(pg); b100 = hook(pg, "H.fmt.fP(H.totBest(H.L5(),1e14).N)")
    check(f'{tag} part 5: serving 100T tokens: the cheapest size moves down to {b100}; SmolLM’s reason; Next enabled', Z(lang, f'the cheapest model for this loss has about {b100} parameters', f'做到这个 loss 最省的模型约 {b100} 个参数') in o and 'SmolLM 135M' in o and not any(nexts(pg)), o[:400])
    check(f'{tag} part 5: training only and training + serving drawn, both cheapest sizes marked', vis(pg, '#sl-chart .sl-trn') and vis(pg, '#sl-chart .sl-tot') and pg.evaluate("()=>document.querySelectorAll('#sl-chart .sl-min').length") == 2)
    every(pg, f'{tag} part 5', lang, f'{tag}_p5')
    # ---- part 6: what to remember
    nav(pg, 'next')
    rt = text(pg, '#sl-rail')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#sl-rail .sl-remlist li').length") == 3
          and rt.count('✗') == 3 and pg.evaluate("()=>!!document.querySelector('#sl-rail [data-mode-go=\"sandbox\"]')"), rt[-300:])
    for k in ('iso', 'tot', 'nd'):
        click(pg, f'#sl-rail [data-rem="{k}"]')
        st = state(pg)
        check(f'{tag} part 6: “show in the chart” {k}', st['remView'] == k and st['view'] == k and pg.evaluate(f"()=>document.querySelector('#sl-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true', st)
    check(f'{tag} part 6: published models on the parameters-and-tokens view', pg.evaluate("()=>document.querySelectorAll('#sl-chart .sl-real').length") == 5 and 'Llama 3 8B' in text(pg, '#sl-chart'))
    every(pg, f'{tag} part 6', lang, f'{tag}_p6')
    pg.close()
    pg = page(b, w, h, lang, dark)
    nav(pg, 'skip')
    check(f'{tag} part 1: Skip goes on without a prediction', state(pg)['step'] == 1 and state(pg)['pred'] is None)
    pg.close()


def run_reduced(b, lang):
    pg = page(b, 1440, 900, lang, reduce=True)
    nav(pg, 'skip'); t0 = state(pg)['tweens']
    slide(pg, '#sl-ic-lc-in', 12)
    check(f'reduced motion {lang}: the budget jumps, no transition', state(pg)['tweens'] == t0 and abs(state(pg)['C'] / 1e23 - 1) < 1e-9, state(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    lead = pg.inner_text('#sl-rail .railcard > p.small')
    check(f'{tag} G1: the scenarios card leads in with what to pick and what to watch, not “one at a time”', lead == Z(lang, 'Pick a scenario and see how the best model size changes.', '选一个场景，看看最好的模型大小怎么变。'), lead)
    check(f'{tag} scenarios: four, the first pressed on entry', pg.evaluate("()=>document.querySelectorAll('#sl-rail [data-preset]').length") == 4
          and pg.evaluate("()=>document.querySelector('#sl-rail [data-preset=\"0\"]').getAttribute('aria-pressed')") == 'true', state(pg))
    want = [Z(lang, '10²² FLOPs buy a model of about 9.06B parameters trained on 184B tokens', '10²² FLOP 最好训一个约 9.06B 个参数的模型，训 184B 个 token'),
            Z(lang, 'To serve 100T tokens, the cheapest way to that loss is not the 29.5B model', '要服务 100T 个 token 时，做到这个 loss 最省的不是每个参数 20 个 token 的 29.5B 模型'),
            Z(lang, 'With only 100B tokens, 10²² FLOPs are best spent on a 16.7B model', '只有 100B 个 token 时，10²² FLOP 最好用来训一个 16.7B 的模型'),
            Z(lang, 'At 10²¹ FLOPs the best model has about 2.78B parameters', '10²¹ FLOP 时，最好的模型约 2.78B 个参数')]
    for i in range(4):
        click(pg, f'#sl-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#sl-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and want[i] in ins, (pressed, ins[:200]))
        dsh = pg.evaluate("()=>[...document.querySelectorAll('#sl-metrics .d')].map(d=>d.textContent).join('')")
        check(f'{tag} scenario {i}: no ▲▼ against the previous scenario (R6.6)', dsh == '', dsh)
        if i == 2:
            check(f'{tag} scenario data: the badge matches the card (S3), the dotted part named one way (S4)', text(pg, '#sl-status') == Z(lang, 'All the data used: best size under the limit', '数据用完了：有上限时最好的大小')
                  and Z(lang, '(the dotted part)', '（点线那段）') in ins and Z(lang, 'dotted: needs over 100B tokens', '点线：需要超过 100B 个 token') in text(pg, '#sl-leg') and vis(pg, '#sl-insight a[href="#data-pipeline"]'), (text(pg, '#sl-status'), text(pg, '#sl-leg')))
        every(pg, f'{tag} scenario {i}', lang, f'{tag}_scenario_{i}' if i in (1, 2) else None)
    check(f'{tag} scenario data: the dotted part and its label', vis(pg, '#sl-chart .sl-dotted') or state(pg)['cap'] == 0)
    click(pg, '#sl-rail [data-preset="1"]')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh.startswith('#scaling-law~'), hsh)
    for v in ('nd', 'iso', 'tot'):
        click(pg, f'#sl-vsw button[data-v="{v}"]')
        check(f'{tag} scenarios: the chart switch shows {v}', state(pg)['view'] == v)
    slide(pg, '#sl-ctl-ln-in', 30)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#sl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change in Settings un-presses the scenario', pressed.count('true') == 0 and Z(lang, 'match none of these scenarios', '和这几个场景都不一样') in text(pg, '#sl-insight'), pressed)
    follows(pg, f'{tag} Settings', '#sl-ctl-lc-in', 'lc', 0.7, 'chart')
    follows(pg, f'{tag} Settings', '#sl-ctl-ln-in', 'ln', 0.35, 'chart')
    follows(pg, f'{tag} Settings', '#sl-ctl-u-in', 'u', 0.9, 'metrics')
    every(pg, f'{tag} sandbox settings', lang)
    pg2 = page(b, w, h, lang, hash_=hsh[1:])
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, the product scenario pressed', st['mode'] == 'sandbox' and st['T'] == 1e14
          and pg2.evaluate("()=>document.querySelector('#sl-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # ---- the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: starts at 8B on 100B tokens', (st['chN'], st['chD']) == (2, 0), st)
    check(f'{tag} challenge: Settings locked', pg.evaluate("()=>document.getElementById('sl-controls').classList.contains('locked')") and vis(pg, '#sl-lock'))
    check(f'{tag} challenge: plain statement and the assumptions', Z(lang, 'Pick one model size and one number of training tokens below', '从下面选一个模型大小，再选训练用的 token 数') in text(pg, '#sl-rail') and pg.evaluate("()=>document.querySelectorAll('#sl-rail .sl-asm li').length") == 2)
    check(f'{tag} challenge: the 2.10 line on the chart', vis(pg, '#sl-chart .sl-goalc'))
    every(pg, f'{tag} challenge start', lang, f'{tag}_challenge_start')
    click(pg, '#sl-rail [data-act="submit"]')
    r = text(pg, '#sl-ch-res')
    check(f'{tag} challenge tier 3: not yet (2.187), the main gap from the two terms, Show the cheapest plan; the failure in red', Z(lang, 'Not yet: the loss is 2.187', '还没达到：loss 是 2.187') in r
          and Z(lang, 'Your plan falls short on the cost of little data', '你的方案差在数据少的代价') in r and vis(pg, '#sl-ch-res [data-act="reveal"]')
          and pg.evaluate("()=>document.getElementById('sl-status').classList.contains('bad')") and not vis(pg, '#sl-ch-res .sl-sum'), r[:300])
    if w == 1440: shot(pg, f'{tag}_challenge_tier3')
    click(pg, '#sl-ch-res [data-act="reveal"]')
    r = text(pg, '#sl-ch-res')
    roles = pg.evaluate("()=>[...document.querySelectorAll('#sl-ch-res .sl-c3r')].map(e=>e.dataset.role)")
    chips = pg.evaluate("()=>[...document.querySelectorAll('#sl-ch-res .sl-c3r')].map(e=>[...e.querySelectorAll('.sl-chip')].map(c=>c.textContent+(c.classList.contains('on')?'*':'')).join('|'))")
    check(f'{tag} challenge R1: your plan, the cheapest and the counterexample; shared conditions once; only differing settings as chips, yours highlighted',
          roles == ['me', 'best', 'ctr'] and Z(lang, 'What the three plans share: the loss must be 2.10 or lower, then the model serves 2T tokens', '三种方案都要把 loss 做到不高于 2.10，训完都要服务 2T 个 token') in r
          and chips == Z(lang, ['8B parameters|100B tokens*', '8B parameters|1T tokens', '20B parameters*|300B tokens*'], ['8B 个参数|训 100B 个 token*', '8B 个参数|训 1T 个 token', '20B 个参数*|训 300B 个 token*']), (roles, chips, r[:300]))
    check(f'{tag} challenge R1: the failing loss in red; the one change that helps most', vis(pg, '#sl-ch-res .sl-c3r.me .badc') and Z(lang, 'loss 2.187, above 2.10: misses the goal', 'loss 2.187，高于 2.10，没达标') in r
          and Z(lang, 'Of the single changes that meet the goal, 1T training tokens instead of 100B takes the least compute. That is the cheapest plan.', '只改一项就能达标的改法里，把训练 token 从 100B 换成 1T 最省。这就是最省的方案。') in r, r[-900:])
    check(f'{tag} challenge R1: no list of every plan; the passing ones behind a disclosure closed by default', pg.evaluate("()=>!document.querySelector('#sl-ch-res .cmp')") and len(roles) <= 3
          and pg.evaluate("()=>{const d=document.querySelector('#sl-ch-res details.sl-all'); return !!d&&!d.open}") and Z(lang, 'See all settings that pass (13)', '看全部可行的设置（13 种）') in r, r[-400:])
    click(pg, '#sl-rail .seg-ctl[data-key="chn"] button[data-v="3"]'); click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="1"]')
    r = text(pg, '#sl-ch-res')
    roles = pg.evaluate("()=>[...document.querySelectorAll('#sl-ch-res .sl-c3r')].map(e=>e.dataset.role)")
    check(f'{tag} challenge tier 2 (20B on 300B, the training-only best): goal met, but a cheaper plan; serving is most of the gap', Z(lang, 'Goal met, but a cheaper plan exists', '达成目标，不过还有更省的方案') in r and roles == ['me', 'best']
          and Z(lang, 'Your plan is the cheapest one when only training counts, 15 tokens per parameter, close to 20. The cheapest plan takes about 31% less compute in all. Most of the difference is serving', '你的方案是只算训练时最省的，每个参数 15 个 token，接近 20。最省的方案总算力比你的少约 31%。你多花的算力，大部分在服务上') in r
          and Z(lang, 'No single change saves compute', '只改一项省不下来') in r, r[:600])
    click(pg, '#sl-rail .seg-ctl[data-key="chn"] button[data-v="2"]'); click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="2"]')
    r = text(pg, '#sl-ch-res')
    roles = pg.evaluate("()=>[...document.querySelectorAll('#sl-ch-res .sl-c3r')].map(e=>e.dataset.role)")
    check(f'{tag} challenge tier 1 (8B on 1T): goal met with the least compute; your plan beside the counterexample', Z(lang, 'Goal met, with the least compute', '达成目标，而且是最省的方案') in r and '8.0×10²²' in r and roles == ['mebest', 'ctr'], (roles, r[:300]))
    check(f'{tag} challenge: the training-only best (20B on 300B) explained', Z(lang, 'The counterexample is the cheapest plan when only training counts: 20B on 300B tokens, 15 tokens per parameter, close to 20', '反例是只算训练时最省的方案：20B 的模型训 300B 个 token，每个参数 15 个，接近 20') in r
          and Z(lang, 'takes about 31% less compute in all', '总算力少约 31%') in r, r[-500:])
    pg.evaluate("()=>{const d=document.querySelector('#sl-ch-res details.sl-all'); d.open=true}"); pg.wait_for_timeout(200)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#sl-ch-res details.sl-all tbody tr')].map(t=>t.className)")
    check(f'{tag} challenge: the disclosure lists the 13 plans that pass, the cheapest first and marked, yours in bold', len(rows) == 13 and rows[0].split() == ['ok', 'me'], rows)
    click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="3"]')
    check(f'{tag} challenge: the disclosure stays open after a change of plan', pg.evaluate("()=>{const d=document.querySelector('#sl-ch-res details.sl-all'); return !!d&&d.open}"))
    click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="2"]')
    ex = hook(pg, "H.exits()")
    check(f'{tag} new situation 2 says the compute is unlimited, so only E is left (owner b8, item 6)', Z(lang, 'Suppose the compute is unlimited', '假设算力不受限制') in ex['transferQ'][1]
          and Z(lang, 'Only E is left, 1.82 in this fit', '最后只剩 E，这个拟合里是 1.82') in ex['transferWhy'][1] and ex['transferA'][1] == 1, ex['transferQ'][1])
    check(f'{tag} challenge: the capability card', vis(pg, '#sl-rail [data-exits]'))
    pg.evaluate("()=>document.getElementById('sl-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(300)
    every(pg, f'{tag} challenge solved', lang, f'{tag}_challenge_solved')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the plan and the result', (st['chN'], st['chD']) == (2, 2) and st['chSubmitted'] and vis(pg, '#sl-ch-res .sl-sum'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
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
    for lang in LANGS:
        run_reduced(b, lang)
    if FULL:
        run_guided(b, 1440, 900, 'en', dark=True)
    b.close()
for hit in rule_hits[:60]:
    check('writing rules: ' + hit, False)
for hit in fit_hits[:60]:
    check(hit, False)
check('no page errors', not errs, errs[:5])
# ---- G1/G2 (the owner's message of 2026-10-08): option lead-ins instead of “one at a time”, and no sentence that leans on a detail
# of an earlier part. Read from the lab's own source, so both languages are covered.
import re as _re
_SRC = ''.join((P / d / 'scaling-law.js').read_text() for d in ('labs', 'teasers') if (P / d / 'scaling-law.js').exists())
_G1 = _re.findall(r'.{0,30}(?:一次只能选|不会叠加|do not stack|[Oo]ne at a time|一次选一).{0,30}', _SRC)
check('G1: no “one at a time / do not stack” note left in the lab or its tile (EN and ZH)', not _G1, _G1[:3])
_NEW = ["The three plans below use the same compute. Pick one, put it into the formula and see its loss.", "下面三种方案用的算力一样多。选一种代进公式，看看它的 loss 是多少。", "The dashed curve counts training only. Pick how many tokens the model will serve: a solid curve that adds serving appears; see where its lowest point moves.", "虚线只算训练。选一个服务量，图里会多出一条加上服务的实线，看看它的最低点往哪边移。", "The three rings in the chart are three training plans of different sizes. Each takes", "FLOPs, so they sit on one diagonal.", "图里的三个圆圈是三种训练方案，模型大小各不相同。每种都是", "FLOP，所以它们在同一条斜线上。", "Shows the iso-FLOP curve at a budget of 10²³ FLOPs.", "在图里显示预算 10²³ FLOP 的 iso-FLOP 曲线。", "Shows training plus serving compute for a model that will serve 100T tokens.", "在图里显示要服务 100T 个 token 时，训练加服务的算力。", "Training compute is about 6 × parameters × tokens.", "Divide it by what the GPUs really deliver, and you know how long the training takes.", "As the budget grows, grow parameters and tokens together.", "Whatever the budget, the best plan sits at about", "tokens per parameter. So ten times the budget means about three times the parameters and three times the tokens.", "A model that will serve many tokens is worth training smaller and longer.", "Every served token costs compute in proportion to the parameters. The more tokens the model serves, the more a smaller model saves, and the extra training soon pays for itself.", "✗ “The bigger the model, the better.” For a fixed budget, too big a model gets too few tokens. The cost of little data then outweighs what the size gains, and the loss ends higher.", "✗ “20 tokens per parameter is a rule.” 20 is only this fit’s best ratio when training alone counts. Count serving too, and the more tokens the model will serve, the higher the cheapest ratio, up to thousands of tokens per parameter.", "训练算力约等于 6 × 参数量 × token 数。", "再除以 GPU 每秒真正做到的运算，就能估出要训多久。", "预算变大时，参数量和 token 数要一起长。", "不管预算多大，最好的方案都在每个参数约", "个 token。所以预算变成 10 倍，参数量和 token 数各长约 3 倍。", "要服务很多 token 的模型，值得训得小一点、久一点。", "每服务一个 token，花的算力都和参数量成正比。要服务的 token 越多，小模型省下的越多，多训的那些 token 很快就赚回来了。", "✗ “模型越大越好。”算力固定时，太大的模型分到的 token 太少。这时数据少的代价超过了模型大的好处，loss 反而更高。", "✗ “每个参数 20 个 token 是规定。”20 倍只是这个拟合在只算训练时给出的最好比例。把服务的算力也算进来，要服务的 token 越多，最省的比例越高，可以到每个参数上千个 token。", "A model too big for 10²¹ FLOPs", "10²¹ FLOP 下，模型太大", "The budget changes, the lesson does not: too big a model for its compute is starved of data.", "预算变了，道理不变。对这笔算力来说太大的模型，分到的数据会不够。", "and a model too big for a smaller budget. Pick one to see how the best size changes.", "以及更小的预算下一个太大的模型。", "The three plans with the same compute (280B, 70B and 17.5B parameters) are this lab’s own example.", "算力相同的三种方案（280B、70B 和 17.5B 个参数）是本 Lab 自己举的例子。", "The best tokens grow with the budget: from 10²¹ to 10²⁵ FLOPs they go from about 60B to about 5T.", "最好的 token 数随预算一起增长：预算从 10²¹ 到 10²⁵ FLOP，最好的 token 数从约 60B 涨到约 5T。", "For Llama 3 8B’s loss, for example, the cheapest model counting training only has", "parameters, far more than 8B.", "正好相反。比训练最优点小的模型，要做到同样的 loss，训练算力更多。比如要做到 Llama 3 8B 的 loss，只算训练时最省的是", "的模型，比 8B 大得多。", "One of three plans with the same compute", "parameters on", "算力相同的三种方案之一：", "ring: a plan with the same compute", "Each ring is one of three training plans, at its parameters and training tokens. All three sit on one diagonal: the same compute.", "每个圆圈是一种训练方案，画在它的参数量和训练 token 数上。三个圆圈在同一条斜线上：算力一样。", "loss 在第 3 部分揭晓，那里会把三种方案代进公式。", "The losses appear in part 3, which puts the three plans into the formula."]
check(f'G1/G2: the new lead-ins and the rewritten sentences are in the lab ({len(_NEW)})', all(x in _SRC for x in _NEW), [x for x in _NEW if x not in _SRC][:3])
_OLD = ["Put part 1’s three plans into the formula. One at a time; they do not stack", "把第 1 部分的三种方案代进公式比一比。一次只能选一个，不会叠加：", "The dashed curve counts training only; the solid one adds serving. Pick how many tokens the model will serve, one at a time", "虚线只算训练，实线再加上服务。选一个服务量，一次只能选一个：", "The three plans of part 1 sit on one diagonal: each takes", "第 1 部分的三种方案在同一条斜线上，每种都是", "Shows the iso-FLOP curves of part 4 at 10²³ FLOPs.", "在图里显示第 4 部分的 iso-FLOP 曲线，预算 10²³ FLOP。", "Shows part 5 at 100T served tokens.", "在图里显示第 5 部分服务 100T 个 token 的情况。", "Counted as in part 2, 10²³ FLOPs take about", "In part 4, from 10²¹ to 10²⁵ FLOPs, every budget’s lowest point sat at about", "In part 5, serving 100T tokens moved the cheapest size from about", "✗ “The bigger the model, the better.” For a fixed budget, too big a model gets too few tokens. In part 3, the 280B plan had four times the parameters of the 70B plan and still ended with a higher loss.", "✗ “20 tokens per parameter is a rule.” 20 is only this fit’s best ratio when training alone counts. In part 5, with 100T tokens to serve, the cheapest ratio became over a thousand tokens per parameter.", "照第 2 部分的算法，10²³ FLOP 在 H100 上约要", "第 4 部分里，从 10²¹ 到 10²⁵ FLOP，每笔预算的最低点都在每个参数约", "第 5 部分里，服务 100T 个 token 时，最省的模型从约", "✗ “模型越大越好。”算力固定时，太大的模型分到的 token 太少。第 3 部分里，280B 的方案参数是 70B 方案的 4 倍，loss 反而更高。", "✗ “每个参数 20 个 token 是规定。”20 倍只是这个拟合在只算训练时给出的最好比例。第 5 部分里，要服务 100T 个 token 时，最省的比例变成每个参数一千多个 token。", "Part 1’s comparison, redone at 10²¹ FLOPs", "在 10²¹ FLOP 下重做第 1 部分的对比", "Part 1’s lesson holds at any budget: too big a model for its compute is starved of data.", "第 1 部分的结论在任何预算下都成立：对这笔算力来说太大的模型，数据会不够。", "and part 1’s comparison at a smaller budget. Pick one to see how the best size changes.", "以及在更小的预算下重做第 1 部分的对比。", "Part 1’s three plans are this lab’s own example at one budget, not published models.", "第 1 部分的三种方案是这个 Lab 自己在同一笔预算下举的例子，不是公开的模型。", "In part 4 the best tokens grew with the budget: from 10²¹ to 10²⁵ FLOPs they went from about 60B to about 5T.", "第 4 部分里，最好的 token 数随预算增长：从 10²¹ 到 10²⁵ FLOP，从约 60B 涨到约 5T。", "In part 5, counting training only, the cheapest was a", "正好相反。比训练最优点小的模型，要做到同样的 loss，训练算力更多。第 5 部分里只算训练时，最省的是", "Plan from part 1", "ring: a plan from part 1", "圆圈：第 1 部分的方案", "Each ring is one of part 1’s three plans, at its parameters and training tokens. All three sit on one diagonal: the same compute.", "每个圆圈是第 1 部分的一种方案，画在它的参数量和训练 token 数上。三个圆圈在同一条斜线上：算力一样。", "那里会把第 1 部分的三种方案代进公式", "which puts part 1’s three plans into the formula"]
check(f'G2: none of the {len(_OLD)} sentences that leaned on an earlier part (or said “one at a time”) is left', not any(x in _SRC for x in _OLD), [x for x in _OLD if x in _SRC][:3])
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
