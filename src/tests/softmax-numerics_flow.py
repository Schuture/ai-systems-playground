"""Flow test for the softmax-numerics lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run and three more widths:
- no page errors and no horizontal overflow at any step; the phone's first screen (m2 template, R7.3);
- the header: title, “Toy run”, the outcome line, the popovers;
- the prediction gate (R3.8): before a prediction the answer (every probability stays the same) appears nowhere: not in the card, the
  stage, the metric cards, Settings, the popovers or tooltips, and Show the math has no identity; Next waits for a pick; “Saved …”;
- part 3: Next waits for a largest logit where the naive softmax works; then the result, the verdict with the right answer, and why;
  below −90 the naive result drifts, from −104 it is NaN everywhere; ▲▼ badges only after the learner's own change (R6.6);
- part 4: Next waits for FP16 (NaN, the window ends at 11.09); BF16 gives a result; part 5: Next waits for a finite naive loss;
  part 6: Next waits for all four blocks; part 7: three points with “Show in the chart”, three misconceptions, “See the scenarios”;
- Settings: only the part's control, with a note saying what is fixed (R4.2); why Next is greyed out;
- Scenarios: four, single choice, pressed state follows the settings, a change under Settings does not move the page (R3.2), the
  padding row with −1e9 and −Inf, the URL carries the setup and a fresh page restores it;
- the challenge: Settings locked, live progress, the three feedback tiers (R4.4), “Show the answer”, a brief chart (the owner's review
  of batch 8): the 3 fixes that meet the goal, the initial setup and the pick, nothing else, and a note that counts the 2 left out, with
  the goal (R5.4), the capability card; the language switch keeps the answer;
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (Chinese spacing and punctuation, no two colons in one sentence, no 亿/万,
  no “toy” but “Toy run”, no 起点 / 控件), and colours only for their data (R5.6: red only for overflow, NaN and a missed goal).
NaN and Inf are this lab's subject, so they are not flagged as broken values; “undefined”, “[object” and “Infinity” are.
Screenshots: shots/softmax-numerics/.
Run: python3 tests/softmax-numerics_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/softmax-numerics-flow.html', '--only', 'softmax-numerics'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'softmax-numerics-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'softmax-numerics-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'softmax-numerics-flow-wrapped.html')
SH = P / 'shots' / 'softmax-numerics'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='softmax-numerics'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__sm.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#sm-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__sm||!window.__sm.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#sm-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#sm-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/undefined|\\[object|Infinity|NaN%|null\\b/,'broken value'],
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #sm-mathbody, #sm-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for overflow, NaN and a missed goal
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|sm-over|\\bbad\\b|\\bnan\\b|warn|wrong|ex-res|sm-sw-bad|sm-sw-nan|sm-sw-inf/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#sm-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#sm-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#sm-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


def rules(pg, where, lang):
    for sel in ('#sm-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#sm-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#sm-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")


# everything a learner can read before the prediction: the header and its popovers, the card, the stage, Settings, the dock, tooltips
def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#sm-app .labhead','#sm-rail','#sm-stage','#sm-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#sm-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);});
      t.push(document.getElementById('sm-mathbody').textContent); return t.join('\\n')}""")


ANSWER = {'en': ['no probability changed', 'changes no probability', 'cancels', 'for free', 'zi−m'], 'zh': ['没有一个概率变了', '不改变任何概率', '约掉', '不花任何代价', 'zi−m']}


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#sm-app .labtitle') == Z(lang, 'Why does a naive softmax return NaN?', '朴素的 softmax 为什么会算出 NaN？'), text(pg, '#sm-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#sm-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#sm-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#sm-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#sm-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#sm-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#sm-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1: the mission
    st = state(pg)
    check(f'{tag} part 1: FP32, the sampler, largest score 92', (st['cs'], st['fmt'], st['top'], st['step']) == ('lm', 'fp32', 92, 0), st)
    check(f'{tag} part 1: the status says the naive softmax gives NaN', text(pg, '#sm-status') == Z(lang, 'Naive: NaN', '朴素：NaN'), text(pg, '#sm-status'))
    cells = pg.evaluate("()=>[...document.querySelectorAll('#sm-tab .sm-tr:not(.sm-th)')].map(r=>[...r.querySelectorAll('.sm-c b')].map(x=>x.textContent))")
    check(f'{tag} part 1: NaN for “Paris” and “the”, 0 for the other four; 94.4% with the max subtracted', [c[0] for c in cells] == ['NaN', 'NaN', '0%', '0%', '0%', '0%'] and cells[0][1] == '94.4%', cells)
    check(f'{tag} part 1: Next enabled (no gate)', nexts(pg) and not any(nexts(pg)))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#sm-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), g=document.getElementById('sm-graph');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(g); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, g:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['g'] is not None and fs['g'] < fs['vh'] - 80, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: the lock note links Scenarios; no self-describing hint (A15, A25)', vis(pg, '#sm-lock [data-mode-go="sandbox"]') and not vis(pg, '#sm-hint'))
    check(f'{tag} part 1: Settings fixed with a note, no slider (R4.2)', vis(pg, '#sm-lock') and not vis(pg, '#sm-ctl-top-wrap') and not vis(pg, '#sm-ctl-fmt-wrap'), text(pg, '#sm-lock'))
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer to part 2 appears nowhere (R3.8)', not any(a in ba for a in ANSWER[lang]), [a for a in ANSWER[lang] if a in ba])
    # part 2: predict
    go_part(pg, 1)
    check(f'{tag} part 2: Next disabled until a prediction, with the reason shown', all(nexts(pg)) and Z(lang, 'Pick an answer to continue', '选一个答案才能继续') in pg.evaluate("()=>[...document.querySelectorAll('#sm-rail .sm-why')].map(p=>p.textContent).join(' ')"))
    ba = before_answer(pg)
    check(f'{tag} part 2: the answer appears nowhere before the pick (R3.8)', not any(a in ba for a in ANSWER[lang]), [a for a in ANSWER[lang] if a in ba])
    check(f'{tag} part 2: no error against the exact probabilities before the pick', pg.evaluate("()=>!document.querySelector('#sm-metrics [data-k=\"es\"]')"))
    rules(pg, f'{tag} part 2', lang); fit(pg, f'{tag} part 2')
    click(pg, '#sm-rail [data-pred="flat"]')
    check(f'{tag} part 2: after a pick, Next enabled and “Saved”', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#sm-rail .step.current'), nexts(pg))
    shot(pg, f'{tag}_p2', full=phone)
    # part 3: move every logit
    go_part(pg, 2)
    check(f'{tag} part 3: Next waits for a working naive softmax', all(nexts(pg)) and Z(lang, 'between −94 and 88.5', '−94 到 88.5 之间') in pg.evaluate("()=>[...document.querySelectorAll('#sm-rail .sm-why')].map(p=>p.textContent).join(' ')"))
    check(f'{tag} part 3: only the largest logit under Settings, with a note', vis(pg, '#sm-ctl-top-wrap') and not vis(pg, '#sm-ctl-fmt-wrap') and vis(pg, '#sm-lock'))
    check(f'{tag} part 3: no ▲▼ before the learner moves anything (R6.6)', not badges(pg), badges(pg))
    slide(pg, '#sm-ic-top-in', 178)
    o = obs(pg)
    check(f'{tag} part 3: at 89 still NaN, the reason given, Next still disabled', 'Inf ÷ Inf' in o and all(nexts(pg)) and text(pg, '#sm-status') == Z(lang, 'Naive: NaN', '朴素：NaN'), o[:200])
    slide(pg, '#sm-ic-top-in', 176)
    o = obs(pg)
    check(f'{tag} part 3: at 88 the naive softmax works and Next is enabled', not any(nexts(pg)) and text(pg, '#sm-status') == Z(lang, 'Naive works', '朴素能算'), (nexts(pg), text(pg, '#sm-status')))
    check(f'{tag} part 3: the result, the verdict with the right answer, and why (R3.10)', Z(lang, 'no probability changed', '没有一个概率变了') in o and Z(lang, 'The right answer is “Nothing: every probability stays the same”', '正确答案是“不变：每个概率都和原来一样”') in o and Z(lang, 'Why:', '原因：') in o, o[:400])
    check(f'{tag} part 3: ▲▼ after the learner’s own change (R6.6)', any(Z(lang, 'vs last', '比上次') in x for x in badges(pg)), badges(pg))
    check(f'{tag} part 3: the identity is now in Show the math', 'zi−m' in pg.evaluate("()=>document.getElementById('sm-mathbody').textContent"))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    slide(pg, '#sm-ic-top-in', -200)
    o = obs(pg)
    check(f'{tag} part 3: at −100 the naive one is off by about 2 points', Z(lang, 'off by up to 2.0 pts', '最多差了 2.0 个百分点') in o and '96.4%' in o, o[:200])
    lg = text(pg, '#sm-legend')
    check(f'{tag} part 3: at −100 the status says so and the legend drops “harmless” (A2)', text(pg, '#sm-status') == Z(lang, 'Naive: 2.0 pts off', '朴素：差 2.0 个百分点') and Z(lang, 'harmless', '无害') not in lg and Z(lang, 'rounds to 0', '取整成 0') in lg, (text(pg, '#sm-status'), lg))
    shot(pg, f'{tag}_p3_m100', full=phone)
    slide(pg, '#sm-ic-top-in', -208)
    o = obs(pg)
    check(f'{tag} part 3: at −104 every e^z is 0, 0 ÷ 0 is NaN', '0 ÷ 0' in o and text(pg, '#sm-status') == Z(lang, 'Naive: NaN', '朴素：NaN'), o[:200])
    st = state(pg)
    check(f'{tag} part 3: the hint to go below −103.97 is gone after it', st['seenLow'] and Z(lang, 'Now drag it below', '再把它拖到') not in o)
    rules(pg, f'{tag} part 3 low', lang)
    # part 4: formats
    go_part(pg, 3)
    st = state(pg)
    check(f'{tag} part 4: attention scores up to 13.5 in FP32; Next waits for FP16', (st['cs'], st['fmt'], st['top']) == ('attn', 'fp32', 13.5) and all(nexts(pg)), st)
    check(f'{tag} part 4: only the format under Settings', vis(pg, '#sm-ctl-fmt-wrap') and not vis(pg, '#sm-ctl-top-wrap'))
    click(pg, '#sm-rail .seg-ctl[data-key="fmt"] button[data-v="fp16"]')
    o = obs(pg)
    check(f'{tag} part 4: FP16 gives NaN, the window ends at 11.09; Next enabled', '11.09' in o and not any(nexts(pg)) and text(pg, '#sm-status') == Z(lang, 'Naive: NaN', '朴素：NaN'), o[:300])
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4')
    click(pg, '#sm-rail .seg-ctl[data-key="fmt"] button[data-v="bf16"]')
    o = obs(pg)
    check(f'{tag} part 4: BF16 has no overflow but fewer digits', Z(lang, 'nothing overflows in BF16', '所以不会溢出') in o and text(pg, '#sm-status') == Z(lang, 'Naive works', '朴素能算'), o[:200])
    rules(pg, f'{tag} part 4 BF16', lang)
    # part 5: cross-entropy
    go_part(pg, 4)
    check(f'{tag} part 5: the right token 120 below, naive loss Inf, Next disabled', state(pg)['gap'] == 120 and text(pg, '#sm-status') == Z(lang, 'Naive loss Inf', '朴素 loss 为 Inf') and all(nexts(pg)))
    lb = pg.evaluate("()=>[...document.querySelectorAll('#sm-ce .sm-lb b')].map(x=>x.textContent)")
    check(f'{tag} part 5: the three losses Inf, Inf, 120.09', lb == ['Inf', 'Inf', '120.09'], lb)
    slide(pg, '#sm-ic-gap-in', 103)
    o = obs(pg)
    check(f'{tag} part 5: at 103 the naive loss is finite (103.28, exact 103.09); Next enabled', '103.28' in o and not any(nexts(pg)), o[:300])
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    # part 6: online softmax
    go_part(pg, 5)
    check(f'{tag} part 6: Next waits for the four blocks', all(nexts(pg)))
    for k in range(2):
        click(pg, '#sm-rail [data-on="1"]')
    o = obs(pg)
    check(f'{tag} part 6: block 2 raises the max from 4.5 to 7.5 and rescales the sum by 0.0498', '4.5' in o and '7.5' in o and '0.0498' in o, o[:300])
    for k in range(3):
        click(pg, '#sm-rail [data-on="1"]')
    o = obs(pg)
    check(f'{tag} part 6: all four blocks read, same result as two-pass, Next enabled', Z(lang, 'All four blocks read', '四块都读完了') in o and not any(nexts(pg)) and text(pg, '#sm-metrics [data-k="diff"] .v') == '0', o[:200])
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6')
    # R3.20 (product owner, 2026-10-04): back to part 5 and on to part 6 again: the four blocks read before stay read, no new stepping
    go_part(pg, 4); go_part(pg, 5); o = obs(pg)
    check(f'{tag} part 6 again: the blocks read before stay read (no restart), Next open', state(pg)['onStep'] == 5 and Z(lang, 'All four blocks read', '四块都读完了') in o and not any(nexts(pg)), (state(pg)['onStep'], o[:120]))
    # part 7: what to remember
    go_part(pg, 6)
    body = text(pg, '#sm-rail .step.current')
    check(f'{tag} part 7: three points, three misconceptions, “See the scenarios”', len(pg.evaluate("()=>[...document.querySelectorAll('#sm-rail .sm-remlist li')]")) == 3 and body.count('✗') == 3 and vis(pg, '#sm-rail [data-mode-go="sandbox"]'), body[:200])
    click(pg, '#sm-rail [data-rem="ce"]')
    check(f'{tag} part 7: “Show in the chart” puts part 5’s losses back', pg.evaluate("()=>document.getElementById('sm-graph').dataset.view") == 'ce' and vis(pg, '#sm-ce'))
    click(pg, '#sm-rail [data-rem="on"]')
    check(f'{tag} part 7: … and part 6’s blocks', pg.evaluate("()=>document.getElementById('sm-graph').dataset.view") == 'on' and vis(pg, '#sm-on'))
    click(pg, '#sm-rail [data-rem="shift"]')
    st = state(pg)
    check(f'{tag} part 7: point 1 shows the mission: largest score 92, naive NaN (A6)', st['top'] == 92 and st['cs'] == 'lm' and text(pg, '#sm-status') == Z(lang, 'Naive: NaN', '朴素：NaN'), (st['top'], text(pg, '#sm-status')))
    shot(pg, f'{tag}_p7', full=phone)
    rules(pg, f'{tag} part 7', lang); fit(pg, f'{tag} part 7')
    check(f'{tag} guided: no overflow at the end', overflow(pg) <= 0, overflow(pg))
    pg.close()


def run_tile(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang, hash_='')
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="roots"], #domStrip [data-world="roots"]')].find(e=>e.offsetParent); if(g) g.click();}"""); pg.wait_for_timeout(400)
    pg.evaluate("()=>{const t=document.getElementById('tile-softmax-numerics'); if(t) t.scrollIntoView({block:'center'})}"); pg.wait_for_timeout(300)
    rows = lambda: pg.evaluate("()=>[...document.querySelectorAll('.tz-sm-row')].map(r=>getComputedStyle(r).visibility)")
    check(f'{tag} tile: at temperature 0.10 the naive bar is NaN and the max-subtracted bar shows', rows() == ['visible', 'visible'] and 'NaN' in text(pg, '.tz-sm .tz-read'), (rows(), text(pg, '.tz-sm .tz-read')))
    shot(pg, f'{tag}_tile_t010')
    slide(pg, '#tz-sm-in', 11)
    rd = text(pg, '.tz-sm .tz-read')
    check(f'{tag} tile: at 0.11 only the naive bar shows, and the sentence names no subtraction (R3.8)', rows() == ['visible', 'hidden'] and Z(lang, 'subtract', '减') not in rd and '%' in rd, (rows(), rd))
    shot(pg, f'{tag}_tile_t011')
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    pr = pg.evaluate("()=>[...document.querySelectorAll('#sm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: four, the sampler pressed at the start', pr == ['true', 'false', 'false', 'false'], pr)
    check(f'{tag} scenarios: the Settings hint says where to act (A15)', Z(lang, 'Pick a scenario in the card', '在卡片里选一个应用场景') in text(pg, '#sm-hint'), text(pg, '#sm-hint'))
    want = [('Naive: NaN', '朴素：NaN'), ('Naive: NaN', '朴素：NaN'), ('Naive loss Inf', '朴素 loss 为 Inf'), ('Naive: NaN', '朴素：NaN')]
    for i in range(4):
        click(pg, f'#sm-rail [data-preset="{i}"]')
        pr = pg.evaluate("()=>[...document.querySelectorAll('#sm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i + 1}: single choice, pressed, its insight shown, status {want[i][0]}', pr.count('true') == 1 and pr[i] == 'true' and vis(pg, '#sm-insight') and text(pg, '#sm-status') == Z(lang, *want[i]), (pr, text(pg, '#sm-status')))
        check(f'{tag} scenario {i + 1}: no overflow', overflow(pg) <= 0, overflow(pg))
        shot(pg, f'{tag}_scenario{i + 1}', full=phone)
        rules(pg, f'{tag} scenario {i + 1}', lang); fit(pg, f'{tag} scenario {i + 1}')
    # the padding row: −1e9 gives 1/6 each with the max subtracted, −Inf gives NaN there too
    col = pg.evaluate("()=>[...document.querySelectorAll('#sm-tab .sm-tr:not(.sm-th)')].map(r=>[...r.querySelectorAll('.sm-c b')].map(x=>x.textContent))")
    check(f'{tag} padding row −1e9: naive NaN (0 ÷ 0), max subtracted 16.7% each', all(c == ['NaN', '16.7%'] for c in col), col)
    click(pg, '#sm-controls .seg-ctl[data-key="mv"] button[data-v="inf"]')
    col = pg.evaluate("()=>[...document.querySelectorAll('#sm-tab .sm-tr:not(.sm-th)')].map(r=>[...r.querySelectorAll('.sm-c b')].map(x=>x.textContent))")
    check(f'{tag} padding row −Inf: NaN even with the max subtracted', all(c == ['NaN', 'NaN'] for c in col) and '−Inf − (−Inf)' in text(pg, '#sm-insight'), col)
    rules(pg, f'{tag} scenario padding −Inf', lang)
    # a change under Settings: no scenario pressed, the card says so, the page does not move
    click(pg, '#sm-rail [data-preset="0"]')
    pg.evaluate("()=>document.querySelector('#sm-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    if not phone:
        pg.evaluate("()=>document.getElementById('sm-ctl-top-in').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
    y0 = pg.evaluate("()=>scrollY")
    slide(pg, '#sm-ctl-top-in', 120, change=False)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page (R3.2)', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    pg.evaluate("()=>document.querySelector('#sm-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    pr = pg.evaluate("()=>[...document.querySelectorAll('#sm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed, and the card says so', 'true' not in pr and Z(lang, 'match none', '都不一样') in text(pg, '#sm-insight'), pr)
    check(f'{tag} the URL carries the setup', pg.evaluate("()=>location.hash").endswith('0_0_120_120_e9'), pg.evaluate("()=>location.hash"))
    pg2 = page(b, w, h, lang, hash_='softmax-numerics~1_2_27_120_e9')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, attention scores, FP16, 13.5, its scenario pressed', st['mode'] == 'sandbox' and (st['cs'], st['fmt'], st['top']) == ('attn', 'fp16', 13.5)
          and pg2.evaluate("()=>document.querySelector('#sm-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    # the challenge
    click(pg, '#sm-rail [data-mode-go="challenge"]')
    st = state(pg)
    check(f'{tag} challenge: starts from the naive softmax in FP16', st['mode'] == 'challenge' and st['chPick'] == 'start', st)
    check(f'{tag} challenge: Settings locked with a note, no controls (R4.2)', vis(pg, '#sm-lock') and not any(vis(pg, x) for x in ('#sm-ctl-top-wrap', '#sm-ctl-fmt-wrap', '#sm-ctl-gap-wrap', '#sm-ctl-z')))
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#sm-rail [data-exits]'))
    check(f'{tag} challenge: live progress red at the start, NaN (R4.3)', pg.evaluate("()=>document.getElementById('sm-ch-g1').classList.contains('sm-over')") and text(pg, '#sm-ch-err') == 'NaN')
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang); fit(pg, f'{tag} challenge')
    click(pg, '#sm-rail [data-lever="bf16"]')
    check(f'{tag} challenge: BF16 live progress 0.18 pts, still red', text(pg, '#sm-ch-err').startswith('0.18') and pg.evaluate("()=>document.getElementById('sm-ch-g1').classList.contains('sm-over')"), text(pg, '#sm-ch-err'))
    click(pg, '#sm-rail [data-act="submit"]')
    res = text(pg, '#sm-ch-res')
    check(f'{tag} challenge tier 3: not met, why, a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#sm-ch-res [data-act="reveal"]') and not vis(pg, '#sm-ch-res .cmp')
          and text(pg, '#sm-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res[:200])
    click(pg, '#sm-ch-res [data-act="reveal"]')
    rows = pg.evaluate("()=>[...document.querySelectorAll('#sm-ch-res > .cmp .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:!!r.querySelector('.cmp-tag,.cmp-you')}))")
    check(f'{tag} challenge brief chart: 5 rows (the 3 fixes that meet the goal, my pick BF16, the initial setup), nothing else', len(rows) == 5 and sum(r['ok'] for r in rows) == 3 and sum(r['me'] for r in rows) == 1
          and all(r['ok'] or r['me'] or r['tag'] for r in rows) and Z(lang, 'initial setup', '初始设置') in text(pg, '#sm-ch-res .cmp'), rows)
    cn = text(pg, '#sm-ch-res .cmp-note'); ct = text(pg, '#sm-ch-res .cmp-title')
    check(f'{tag} challenge: the note counts the 2 fixes left out; the title no longer says “each fix”', cn == Z(lang, 'The other 2 options miss the goal and are not shown.', '其余 2 种没达到目标，图里没有列出。')
          and not re.search(r'each fix|每种办法', ct), (cn, ct))
    fit(pg, f'{tag} challenge chart')
    lab = pg.evaluate("""()=>{const c=document.querySelector('#sm-ch-res .cmp'), cr=c.getBoundingClientRect(); return [...c.querySelectorAll('.cmp-row')].map(r=>{const v=r.querySelector('.cmp-val'), q=v.getBoundingClientRect(), rr=r.getBoundingClientRect(), fs=parseFloat(getComputedStyle(v).fontSize);
      return {t:v.textContent, oneLine:q.height<=fs*1.9, inside:q.right<=cr.right+0.5&&q.left>=cr.left-0.5, inRow:q.top>=rr.top-0.5&&q.bottom<=rr.bottom+0.5, nan:r.classList.contains('sm-cmp-nan')};});}""")
    check(f'{tag} challenge chart: every value label on one line, inside the chart and its own row (A1)', all(x['oneLine'] and x['inside'] and x['inRow'] for x in lab), lab)
    check(f'{tag} challenge chart: the initial setup is a red hatched NaN row, in the legend (A19)', lab[0]['nan'] and lab[0]['t'] == 'NaN' and not any(x['nan'] for x in lab[1:]) and 'NaN' in text(pg, '#sm-ch-res .cmp-leg'), lab[0])
    shot(pg, f'{tag}_ch_chart', full=phone)
    click(pg, '#sm-rail [data-lever="clamp"]')
    res = text(pg, '#sm-ch-res')
    check(f'{tag} challenge: clamping explains the sum overflow', res.startswith(Z(lang, 'Not yet', '还没达到')) and '65,504' in res, res[:300])
    click(pg, '#sm-rail [data-lever="fp32"]')
    res = text(pg, '#sm-ch-res')
    check(f'{tag} challenge tier 2: FP32 meets the goal, but a smaller change exists', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'a smaller change exists', '还有改动更少的做法') in res, res[:220])
    click(pg, '#sm-rail [data-lever="both"]')
    res = text(pg, '#sm-ch-res')
    check(f'{tag} challenge: FP32 arithmetic on FP16 scores is 2 bytes per stored score, what real GPU code does, one change too many here (A13)', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'what real GPU code does', '实际的 GPU 代码就是这样做的') in res and text(pg, '#sm-metrics [data-k="bytes"] .v') == '2' and text(pg, '#sm-metrics [data-k="bytes"] .sub') == 'FP16', res[:300])
    click(pg, '#sm-rail [data-lever="max"]')
    res = text(pg, '#sm-ch-res')
    check(f'{tag} challenge tier 1: subtract the max in FP16, the smallest change; status green', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'that is the smallest change', '改动最少') in res and text(pg, '#sm-status') == Z(lang, 'Goal met', '达成目标'), res[:220])
    check(f'{tag} challenge: baseline, reason and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'initial setup', '初始设置'), '2,048']), res[-300:])
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    for t in (0, 1):
        pg.evaluate(f"()=>{{const b=document.querySelector('#sm-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#sm-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#sm-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the submitted result', st['chPick'] == 'max' and st['chSubmitted'] and vis(pg, '#sm-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('sm-tab'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, with a short status and the result', dk['shown'] and len(dk['status']) <= 14 and 'NaN' in dk['read'], dk)
        ov = pg.evaluate("""()=>{const ids=['dkStatus','dkVis','dkRead'], R=ids.map(i=>{const e=document.getElementById(i); const r=e.getBoundingClientRect(); return r.width&&r.height?r:null;}).filter(Boolean), kids=[...document.querySelectorAll('#dkVis *')].map(e=>e.getBoundingClientRect()).filter(r=>r.width&&r.height), out=[];
          const hit=(a,b)=>a.left<b.right-0.5&&b.left<a.right-0.5&&a.top<b.bottom-0.5&&b.top<a.bottom-0.5;
          for(let i=0;i<R.length;i++) for(let j=i+1;j<R.length;j++) if(hit(R[i],R[j])) out.push([i,j]);
          const rd=document.getElementById('dkRead').getBoundingClientRect(), st=document.getElementById('dkStatus').getBoundingClientRect(); kids.forEach(k=>{if(hit(k,rd)||hit(k,st)) out.push('bar over text');});
          return out;}""")
        check(f'{tag} mini result bar: its status, bar and text do not overlap (A12)', not ov, ov)
        go_part(pg, 2); slide(pg, '#sm-ic-top-in', 176)
        pg.evaluate("()=>{const t=document.getElementById('sm-tab'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(600)
        rd = pg.evaluate("()=>document.getElementById('dkRead').textContent")
        check(f'{tag} mini result bar: part 3 at 88 in one short sentence (A12)', len(rd) <= 60 and '94.4%' in rd, rd)
        shot(pg, f'{tag}_dock_p3')
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see whether the naive softmax breaks there.', '选一个场景，看看朴素 softmax 在那里会不会出错。'), 'chal': ('Try a fix and see how far the largest error drops.', '选一个办法试试，看看最大误差降到多少。'), 'parts': [(6, [('so no exp can overflow, however large the scores get', '所以得分再大，exp 也不会溢出'), ('Log-sum-exp never computes that probability', 'log-sum-exp 不先算这个概率'), ('So each block is read only once', '这样一整行只要读一遍'), ('Softmax depends only on the gaps between scores', 'softmax 只看得分之间的差'), ('The arithmetic alone can give NaN', '只靠计算本身也会出 NaN'), ('It computes the same sum as two passes', '它和读两遍的算法算的是同一个总和')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'softmax-numerics.js').read_text() + ((P / 'teasers' / 'softmax-numerics.js').read_text() if (P / 'teasers' / 'softmax-numerics.js').exists() else '')
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

B9V6 = {'lab': 'softmax-numerics', 'math': '#sm-mathbody', 'block': '.sm-mb', 'sep_extra': '#sm-tab, #sm-ce', 'src_gone': ['const eX=x=>`e<sup>', 'exp(z − m) equals', 'exp(z − old max)', 'exp(z − 旧最大值)', 'exp(z − m) 等于', '−log of the naive probability', '对朴素概率取 −log', "sub:tx('of exp(z − m)'", 'and −log 0 = Inf', '而 −log 0 = Inf', 'is Inf above z = ln of it'], 'text_gone': ['exp(z', 'exp(−m)', '−log of', '取 −log', '−log 0', 'ln of it', 'e^']}


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
            run_tile(b, w, h, lang)
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
