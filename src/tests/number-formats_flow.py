"""Flow test for the number-formats lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (plus one dark run and three more widths on a full run):
- no page errors and no horizontal overflow at any step; no text wider than its box (R7.8); every help cursor has a tooltip (R1.1);
- the header: title, Fidelity: Formula, the outcome line, the popovers open;
- part 1: the weight 0.02 in FP32, BF16 and FP16 with bit boxes; the range strips hidden; Settings fixed with a note;
- part 2 (R3.8): Next disabled until a prediction; before it the value is fixed at 0.02, no range strip, and the answer (1e‑8 is 0 in
  FP16) appears nowhere in text or tooltips; “Saved …” after a pick (R2.5);
- part 3: Next waits for 1e‑8; then the result (FP16 0, BF16 1.001e‑8), the verdict with the right answer, and why (R3.10); 65,504
  and 100,000 show overflow in red for FP16 only;
- part 4: Next waits for an update BF16 loses and FP16 keeps; 3e‑4 is lost in both, 1e‑3 only in BF16;
- part 5: the four fixes single choice, the result on each tried option’s row (R3.6), “+ both” saves both; Settings locked with a note;
- part 6: three points with their pictures, three misconceptions, “See the scenarios”, the next-lab footer (R3.11, R3.12);
- Scenarios: four, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page restores it,
  every control works (value, update, format, loss scaling, master copy, block scale), the three views;
- the challenge: only the recipe in the card, Settings locked (R4.2), live checks, the three feedback tiers (R4.4), “Show the
  answer”, a brief chart (the owner's review of batch 8): the 2 passing recipes, the initial setup and the pick, nothing else, and a note
  that counts the recipes left out (R5.4), the capability card; the language switch keeps the answer;
- no hover that only expands a number to its exact value, outside Show the math (the owner, 2026-10-07 23:37 UTC);
- the mini result bar at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip (as in tests/sampling_flow.py; NaN and Inf are this lab's real values),
  and colours only for their data (R5.6).
Screenshots: shots/number-formats/.
Run: python3 tests/number-formats_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/number-formats-flow.html', '--only', 'number-formats'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'number-formats-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'number-formats-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'number-formats-flow-wrapped.html')
SH = P / 'shots' / 'number-formats'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='number-formats'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__nf.state()")
C = None
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#nf-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__nf||!window.__nf.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#nf-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#nf-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def dock(pg, tag, where, status=None):
    """scroll the chart away (phones), then: the bar shows, its status, and no chip over its one sentence of at most 2 lines (R7.8, review A3)"""
    pg.evaluate("()=>{const t=document.getElementById('nf-graph'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
    dk = pg.evaluate("""()=>{const d=document.getElementById('dock'), v=document.getElementById('dkVis'), r=document.getElementById('dkRead'), s=document.getElementById('dkStatus');
      if(!d||!r) return null; const b=r.getBoundingClientRect(), lh=parseFloat(getComputedStyle(r).lineHeight)||15;
      const hit=[...(v?v.querySelectorAll('*'):[])].concat(s?[s]:[]).filter(e=>e.getClientRects().length).some(c=>{const x=c.getBoundingClientRect(); return !(x.right<=b.left+0.5||x.left>=b.right-0.5||x.bottom<=b.top+0.5||x.top>=b.bottom-0.5)});
      return {shown:!d.hidden, status:s?s.textContent:'', read:r.textContent, hit, lines:Math.round(b.height/lh), chips:v?v.children.length:0}}""")
    check(f'{tag} {where}: the mini result bar once the chart scrolls away', dk and dk['shown'] and (status is None or dk['status'] == status), dk)
    check(f'{tag} {where}: the mini result bar has no chip over its sentence, at most 2 lines (R7.8)', dk and not dk['hit'] and dk['chips'] == 0 and dk['lines'] <= 2, dk)
    shot(pg, f'{tag}_dock_{where.replace(" ", "_")}')
    rules(pg, f'{tag} dock {where}', 'zh' if '_zh' in tag else 'en')
    return dk


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#nf-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/\\btoy\\b|玩具/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/undefined|\\[object|NaN%|Infinity%|\\de[+]\\d/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两每一单\\d ])个数(?![值量])/,'“个数” as “count” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],[/NVIDIA|Nvidia|\\bMeta\\b|Google|Microsoft|OpenAI|Anthropic/,'company name']];
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
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #nf-mathbody, #nf-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for a lost or overflowing value (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|bad|nf-over|nf-nocode|nf-sw-bad|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#nf-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#nf-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.nf-bits')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  const bits=[...document.querySelectorAll('#nf-rows .nf-bits')].filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+1).map(e=>'bit boxes cut off: '+e.scrollWidth+' > '+e.clientWidth);
  return out.concat(bits).slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#nf-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []
EXACT = r"""(sel)=>{const root=document.querySelector(sel); if(!root) return [];
  return [...root.querySelectorAll('[data-tipx],[data-tip],[title]')].filter(e=>!e.closest('details.math,[id$="mathbody"]'))
    .map(e=>e.getAttribute('data-tipx')||e.getAttribute('data-tip')||e.getAttribute('title')||'').filter(t=>/精确值是|Exactly|shown rounded to 3|取 3 位有效数字/.test(t)).map(t=>t.slice(0,90));}"""


CHART = """()=>{const rows=[...document.querySelectorAll('#nf-ch-res > .cmp .cmp-row')], tg=r=>!!r.querySelector('.cmp-tag,.cmp-you');
  return {n:rows.length, ok:rows.filter(r=>r.classList.contains('ok')).length, me:rows.filter(r=>r.classList.contains('me')).length,
    stray:rows.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!tg(r)).length, note:(document.querySelector('#nf-ch-res .cmp-note')||{}).textContent||'',
    title:(document.querySelector('#nf-ch-res .cmp-title')||{}).textContent||''}}"""


def rules(pg, where, lang):
    for sel in ('#nf-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
        for h in pg.evaluate(EXACT, sel): rule_hits.append(f'{where}: a hover that only gives the exact value (the owner, 23:37 UTC): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')


# everything a learner can read before the prediction, outside Connect: header, popovers, card, stage, Settings, tooltips (R3.8)
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#nf-app .labhead','#nf-rail','#nf-stage','#nf-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#nf-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    return '\n'.join(out).replace('‑', '-')


def rowv(pg, f):
    return pg.evaluate(f"()=>{{const r=document.querySelector('#nf-rows .nf-row[data-f=\"{f}\"]'); return r?{{v:r.querySelector('.nf-val').textContent.replace(/\\u2011/g,'-'),e:r.querySelector('.nf-err').textContent,bad:r.classList.contains('nf-bad')}}:null}}")


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    C = pg.evaluate("()=>window.__nf.consts")
    check(f'{tag} title', text(pg, '#nf-app .labtitle') == Z(lang, 'Why does training use BF16 and not FP16?', '训练为什么用 BF16，不用 FP16？'), text(pg, '#nf-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#nf-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#nf-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#nf-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')), text(pg, '#nf-app .lab-outcome'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#nf-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#nf-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#nf-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1: the weight 0.02 in three formats
    st = state(pg)
    check(f'{tag} part 1: value 0.02, view a', st['xi'] == C['X_W'] and st['view'] == 'a', st)
    rows = pg.evaluate("()=>[...document.querySelectorAll('#nf-rows .nf-row')].map(r=>r.dataset.f)")
    check(f'{tag} part 1: three rows FP32, BF16, FP16 with bit boxes', rows == ['fp32', 'bf16', 'fp16'] and pg.evaluate("()=>[...document.querySelectorAll('#nf-rows .nf-row')].map(r=>r.querySelectorAll('.nf-bit').length).join()") == '32,16,16', rows)
    check(f'{tag} part 1: stored values 0.02 / 0.020019531 / 0.020004272', [rowv(pg, f)['v'] for f in rows] == ['0.02', '0.020019531', '0.020004272'], [rowv(pg, f) for f in rows])
    check(f'{tag} part 1: the range strips hidden', not vis(pg, '#nf-range'))
    check(f'{tag} part 1: Settings fixed with a note', vis(pg, '#nf-lock') and '0.02' in text(pg, '#nf-lock') and not vis(pg, '#nf-ctl-x-wrap'), text(pg, '#nf-lock'))
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#nf-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), rows=document.getElementById('nf-rows');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(rows); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, rows:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['rows'] is not None and fs['rows'] < fs['vh'] - 80, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    # part 2: the prediction gate
    click(pg, '#nf-rail [data-nav="next"]')
    check(f'{tag} part 2: Next disabled before a prediction', all(nexts(pg)) and len(nexts(pg)) >= 1, nexts(pg))
    check(f'{tag} part 2: “why Next is greyed out”', Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。') in text(pg, '#nf-rail'))
    ba = before_answer(pg)
    leak = re.search(r'1e-8[^.。\n]{0,40}(?:becomes|is|rounds to|stored as|变成|是|舍入成|存成) ?0(?![.\d])|FP16 (?:stores|存成) 0(?![.\d])|lost to zero in FP16', ba)
    check(f'{tag} part 2: the answer appears nowhere before the prediction (R3.8)', not leak and not vis(pg, '#nf-range') and state(pg)['xi'] == C['X_W'], leak.group(0) if leak else '')
    shot(pg, f'{tag}_p2', full=phone)
    click(pg, '#nf-rail [data-pred="kept"]')
    check(f'{tag} part 2: “Saved” after a pick, Next enabled', Z(lang, 'Saved.', '已记下。') in text(pg, '#nf-rail') and not any(nexts(pg)), nexts(pg))
    rules(pg, f'{tag} part 2', lang); fit(pg, f'{tag} part 2')
    # part 3: range
    click(pg, '#nf-rail [data-nav="next"]')
    check(f'{tag} part 3: the range strips, Next waits for 1e-8', vis(pg, '#nf-range') and all(nexts(pg)) and pg.evaluate("()=>document.querySelectorAll('#nf-range .nf-strip').length") == 3, nexts(pg))
    slide(pg, '#nf-ic-x-in', C['XS'].index(65504) if 65504 in C['XS'] else C['XS'].index(1e5))
    r16 = rowv(pg, 'fp16')
    check(f'{tag} part 3: 100,000 overflows FP16 (red), BF16 stores 99,840', r16['bad'] and r16['v'] == 'Inf' and rowv(pg, 'bf16')['v'] == '99,840' and not rowv(pg, 'bf16')['bad'], [r16, rowv(pg, 'bf16')])
    check(f'{tag} part 3: still waiting for 1e-8', all(nexts(pg)))
    shot(pg, f'{tag}_p3_overflow', full=phone)
    slide(pg, '#nf-ic-x-in', C['X_G'])
    o = obs(pg)
    check(f'{tag} part 3: 1e-8 is 0 in FP16, BF16 1.001e-8; Next enabled', rowv(pg, 'fp16')['v'] == '0' and rowv(pg, 'fp16')['bad'] and rowv(pg, 'bf16')['v'] == '1.001e-8' and not any(nexts(pg)), [rowv(pg, 'fp16'), rowv(pg, 'bf16')])
    check(f'{tag} part 3: result, verdict with the right answer, why (R3.10, R3.8)', all(x in o for x in [Z(lang, '1e-8 in FP16 is 0', '1e-8 存成 FP16 是 0'), Z(lang, 'The right answer is “Exactly 0: the gradient is lost”', '正确答案是“正好是 0：梯度丢了”'), Z(lang, 'Why:', '原因：')]), o[:400])
    check(f'{tag} part 3: the value line and a red dot on FP16', pg.evaluate("()=>!!document.querySelector('#nf-range .nf-vline')&&document.querySelector('#nf-range .nf-strip[data-f=\"fp16\"] .nf-dot.nf-bad')!==null&&document.querySelector('#nf-range .nf-strip[data-f=\"bf16\"] .nf-dot.nf-bad')===null"))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    if phone: dock(pg, tag, 'part 3')
    # part 4: precision
    click(pg, '#nf-rail [data-nav="next"]')
    check(f'{tag} part 4: the number lines, Next waits', vis(pg, '#nf-zoom') and all(nexts(pg)) and pg.evaluate("()=>document.querySelectorAll('#nf-zoom .nf-zr').length") == 3)
    o = obs(pg)
    check(f'{tag} part 4: 3e-4 is lost in both BF16 and FP16', pg.evaluate("()=>[...document.querySelectorAll('#nf-zoom .nf-zr')].map(r=>r.classList.contains('nf-bad')).join()") == 'false,true,true', pg.evaluate("()=>[...document.querySelectorAll('#nf-zoom .nf-zr')].map(r=>r.innerText).join(' | ')"))
    slide(pg, '#nf-ic-u-in', C['US'].index(1e-3))
    o = obs(pg)
    check(f'{tag} part 4: 1e-3 lost in BF16, kept in FP16 as 1.0009766; Next enabled', pg.evaluate("()=>[...document.querySelectorAll('#nf-zoom .nf-zr')].map(r=>r.classList.contains('nf-bad')).join()") == 'false,true,false'
          and '1.0009766' in o and not any(nexts(pg)), o[:300])
    check(f'{tag} part 4: why, with both half-gaps', all(x in o for x in ['0.0039', '0.00049']), o[-300:])
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4')
    # part 5: the fixes
    click(pg, '#nf-rail [data-nav="next"]')
    check(f'{tag} part 5: the recipe view, four fixes, Next waits', vis(pg, '#nf-paths') and pg.evaluate("()=>document.querySelectorAll('#nf-rail [data-lever]').length") == 4 and all(nexts(pg)))
    check(f'{tag} part 5: Settings locked with a note', vis(pg, '#nf-lock') and not vis(pg, '#nf-ctl-x-wrap') and not vis(pg, '#nf-ctl-u-wrap'))
    click(pg, '#nf-rail [data-lever="fp16ls"]')
    o = obs(pg)
    check(f'{tag} part 5: loss scaling saves the gradient, not the update; result on the option row (R3.6)', Z(lang, 'Loss scaling saves the gradient', 'loss scaling 救回了梯度') in o and text(pg, '#nf-rail [data-lever="fp16ls"] .lv-r') == Z(lang, 'gradient ✓, update ✗', '梯度 ✓，更新 ✗'), o[:200])
    click(pg, '#nf-rail [data-lever="fp16m"]')
    check(f'{tag} part 5: the master copy saves the update, not the gradient; earlier result kept', text(pg, '#nf-rail [data-lever="fp16m"] .lv-r') == Z(lang, 'gradient ✗, update ✓', '梯度 ✗，更新 ✓') and text(pg, '#nf-rail [data-lever="fp16ls"] .lv-r') != '' and all(nexts(pg)))
    click(pg, '#nf-rail [data-lever="fp16lsm"]')
    o = obs(pg)
    check(f'{tag} part 5: both fixes save both; single choice; Next enabled', text(pg, '#nf-rail [data-lever="fp16lsm"] .lv-r') == Z(lang, 'gradient ✓, update ✓', '梯度 ✓，更新 ✓')
          and pg.evaluate("()=>[...document.querySelectorAll('#nf-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1 and not any(nexts(pg)) and text(pg, '#nf-status') == Z(lang, 'All 3 checks pass', '3 项检查都通过'), o[:200])
    check(f'{tag} part 5: the stage draws all three checks: smallest gradient, largest gradient, update (review A6)', pg.evaluate("()=>document.querySelectorAll('#nf-paths .nf-pline').length") == 3)
    check(f'{tag} part 5: the chart has the master copy and its FP16 copy', pg.evaluate("()=>!!document.querySelector('#nf-chart .nf-lm')&&!!document.querySelector('#nf-chart .nf-lc')"))
    if phone: dock(pg, tag, 'part 5', Z(lang, 'All 3 checks pass', '3 项检查都通过'))
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    # part 6: what to remember
    click(pg, '#nf-rail [data-nav="next"]')
    r6 = text(pg, '#nf-rail')
    check(f'{tag} part 6: three points and three misconceptions', pg.evaluate("()=>document.querySelectorAll('#nf-rail .nf-remlist li').length") == 3 and r6.count('✗ “') == 3, r6[:300])
    for k, sel in (('b', '#nf-zoom'), ('c', '#nf-paths'), ('a', '#nf-range')):
        click(pg, f'#nf-rail [data-rem="{k}"]')
        st6 = state(pg)
        check(f'{tag} part 6, “{k}”: its picture in the chart, the button pressed', st6['remView'] == k and pg.evaluate(f"()=>document.querySelector('#nf-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true' and vis(pg, sel), st6)
        fit(pg, f'{tag} part 6 {k}')
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#nf-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e)||shown(r) : shown(r)}""")
    check(f'{tag} part 6: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang)
    click(pg, '#nf-rail [data-mode-go="sandbox"]')
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    check(f'{tag} no overflow at the end of the tour', overflow(pg) <= 0, overflow(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    C = pg.evaluate("()=>window.__nf.consts")
    mode(pg, 'sandbox')
    check(f'{tag} Scenarios: four presets, Settings open with every control', pg.evaluate("()=>document.querySelectorAll('#nf-rail [data-preset]').length") == 4
          and all(vis(pg, s) for s in ('#nf-ctl-x-wrap', '#nf-ctl-u-wrap', '#nf-ctl-f16-wrap', '#nf-ctl-ls-wrap', '#nf-ctl-ms-wrap', '#nf-ctl-blk-wrap')) and vis(pg, '#nf-view'))
    check(f'{tag} Scenarios: six rows', pg.evaluate("()=>document.querySelectorAll('#nf-rows .nf-row').length") == 6)
    for i, pz in enumerate(C['PRESETS']):
        click(pg, f'#nf-rail [data-preset="{i}"]')
        st = state(pg)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#nf-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed')).join()")
        ins = text(pg, '#nf-insight')
        check(f'{tag} scenario {pz["id"]}: settings applied, single choice, insight shown (R1.8)', all(st[k] == v for k, v in pz.items() if k != 'id') and pressed.split(',').count('true') == 1 and pressed.split(',')[i] == 'true' and len(ins) > 80, (st, pressed, ins[:120]))
        check(f'{tag} scenario {pz["id"]}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {pz["id"]}', lang); fit(pg, f'{tag} scenario {pz["id"]}')
        shot(pg, f'{tag}_scen_{pz["id"]}', full=phone)
        if pz['id'] in ('fp8', 'inf'):
            labs_ = pg.evaluate("()=>[...document.querySelectorAll('#nf-metrics .metric .k')].map(e=>e.textContent).join(' | ')")
            check(f'{tag} scenario {pz["id"]}: the cards show the scaled FP8 and FP4 rows (review A4)', labs_.count('FP8 E4M3') == 1 and 'FP4' in labs_ and 'BF16' not in labs_, labs_)
            if phone: dock(pg, tag, f'scenario {pz["id"]}')
    ins = text(pg, '#nf-insight')
    check(f'{tag} scenario inf: FP4 with a scale, OCP MX blocks of 32', '32' in ins and 'FP4' in ins, ins[:300])
    # a change under Settings: the preset unpresses, the URL carries the setup, a fresh page restores it
    click(pg, '#nf-ctl-blk [data-v="3"]') if vis(pg, '#nf-ctl-blk [data-v="3"]') else click(pg, '#nf-ctl-blk-wrap [data-v="3"]')
    st = state(pg)
    check(f'{tag} Settings: block scale 1,000× applied, no preset pressed', st['blk'] == 3 and pg.evaluate("()=>[...document.querySelectorAll('#nf-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"), st)
    hsh = pg.evaluate("()=>location.hash")
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    st2 = pg2.evaluate("()=>window.__nf.state()")
    check(f'{tag} the URL restores the setup in a fresh page', all(st2[k] == st[k] for k in ('xi', 'ui', 'view', 'f16', 'ls', 'ms', 'blk')) and st2['mode'] == 'sandbox', (hsh, st2))
    pg2.close()
    # the three views and each control
    slide(pg, '#nf-ctl-x-in', C['XS'].index(500))
    check(f'{tag} Settings: 500 overflows FP8 E4M3 (no Inf: NaN), E5M2 stores 512', state(pg)['view'] == 'a' and rowv(pg, 'e4m3')['bad'] is not None, [rowv(pg, 'e4m3'), rowv(pg, 'e5m2')])
    click(pg, '#nf-ctl-blk-wrap [data-v="0"]')
    check(f'{tag} Settings: unscaled, 500 is NaN in E4M3 and 512 in E5M2, FP4 clamps to 6 (OCP MX)', rowv(pg, 'e4m3')['v'] == 'NaN' and rowv(pg, 'e4m3')['bad'] and rowv(pg, 'e5m2')['v'] == '512' and rowv(pg, 'fp4')['bad'] and rowv(pg, 'fp4')['v'] == '6', [rowv(pg, f) for f in ('e4m3', 'e5m2', 'fp4')])
    check(f'{tag} Settings: the badge says overflow, not lost, at 500 (review A5)', text(pg, '#nf-status') == Z(lang, 'Overflows in 2 of 6', '6 种里 2 种溢出'), text(pg, '#nf-status'))
    shot(pg, f'{tag}_sb_overflow', full=phone)
    rules(pg, f'{tag} sandbox overflow', lang); fit(pg, f'{tag} sandbox overflow')
    slide(pg, '#nf-ctl-u-in', C['US'].index(1e-2))
    check(f'{tag} Settings: the update moves to the update view; 0.01 kept in all', state(pg)['view'] == 'b' and text(pg, '#nf-status') == Z(lang, 'Kept in all', '都留住了'), text(pg, '#nf-status'))
    click(pg, '#nf-ctl-f16-wrap [data-v="bf16"]')
    check(f'{tag} Settings: the format moves to the step view', state(pg)['view'] == 'c' and state(pg)['f16'] == 'bf16')
    click(pg, '#nf-ctl-ms-wrap [data-v="0"]'); click(pg, '#nf-ctl-ls-wrap [data-v="0"]')
    slide(pg, '#nf-ctl-u-in', C['US'].index(3e-4))
    check(f'{tag} Settings: BF16 alone loses the update of 3e-4 on 1.0', text(pg, '#nf-status') == Z(lang, '1 of 3 checks fail', '3 项检查有 1 项失败'), text(pg, '#nf-status'))
    click(pg, '#nf-view [data-view="a"]')
    check(f'{tag} the view buttons switch the chart', vis(pg, '#nf-rows') and not vis(pg, '#nf-zoom'))
    pg.evaluate("()=>document.getElementById('nf-math').open=true"); pg.wait_for_timeout(150)
    check(f'{tag} Show the math decodes every row', pg.evaluate("()=>document.querySelectorAll('#nf-mathbody p').length") >= 7)
    rules(pg, f'{tag} sandbox', lang); fit(pg, f'{tag} sandbox')
    # the challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: five recipes in the card, Settings locked (R4.2)', pg.evaluate("()=>document.querySelectorAll('#nf-rail #nf-ic-ch [data-lever]').length") == 5 and pg.evaluate("()=>document.getElementById('nf-controls').classList.contains('locked')") and vis(pg, '#nf-lock'))
    check(f'{tag} challenge: starts at the initial setup, all FP32, not yet', state(pg)['chR'] == 'fp32' and text(pg, '#nf-status') == Z(lang, 'Not yet', '还没达到'), text(pg, '#nf-status'))
    check(f'{tag} challenge: live checks', pg.evaluate("()=>document.querySelectorAll('#nf-ch-g .goal').length") == 6)
    check(f'{tag} challenge: the stage draws both updates, on 0.02 and on a gain of 1.0', pg.evaluate("()=>document.querySelectorAll('#nf-paths .nf-pline').length") == 4)
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#nf-rail [data-exits]'))
    click(pg, '#nf-rail #nf-ic-ch [data-lever="fp16"]')
    click(pg, '#nf-rail [data-act="submit"]')
    res = text(pg, '#nf-ch-res')
    check(f'{tag} challenge tier 3: all FP16 misses, with what fails and a hint', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#nf-ch-res [data-act="reveal"]'), res[:300])
    check(f'{tag} challenge: the reveal is “Show the answer”', text(pg, '#nf-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), text(pg, '#nf-ch-res [data-act="reveal"]'))
    click(pg, '#nf-ch-res [data-act="reveal"]')
    cr = pg.evaluate(CHART)
    check(f'{tag} challenge, the answer for all FP16: the 2 passing recipes, my pick and the initial setup, nothing else', cr['n'] == 4 and cr['ok'] == 2 and cr['me'] == 1 and cr['stray'] == 0, cr)
    check(f'{tag} challenge: the note counts the 1 recipe left out; the title no longer says “for each recipe”', cr['note'] == Z(lang, 'The other recipe misses the goal and is not shown.', '其余 1 种做法没达到目标，图里没有列出。')
          and not re.search(r'for each|每种做法', cr['title']), (cr['note'], cr['title']))
    shot(pg, f'{tag}_challenge_miss', full=phone)
    rules(pg, f'{tag} challenge miss', lang)
    click(pg, '#nf-rail #nf-ic-ch [data-lever="fp16lsm"]')
    res = text(pg, '#nf-ch-res')
    check(f'{tag} challenge tier 2: FP16 + both meets the goal but is not the fewest', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'one extra part is enough', '只加一个部件就够了') in res and vis(pg, '#nf-ch-res .cmp'), res[:300])
    click(pg, '#nf-rail #nf-ic-ch [data-lever="bf16m"]')
    res = text(pg, '#nf-ch-res')
    check(f'{tag} challenge tier 1: BF16 + master copy, the fewest', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'that is the fewest', '这已经是最少的') in res and text(pg, '#nf-status') == Z(lang, 'Goal met', '达成目标'), res[:300])
    cmp = pg.evaluate("()=>[...document.querySelectorAll('#nf-ch-res .cmp-row')].map(r=>r.innerText.replace(/\\n/g,' '))")
    check(f'{tag} challenge: the brief chart, the 2 passing recipes (one my pick) and the initial setup (R5.4)', len(cmp) == 3 and any(Z(lang, 'initial setup', '初始设置') in r for r in cmp) and any(Z(lang, 'your pick', '你的选择') in r for r in cmp), cmp)
    cr = pg.evaluate(CHART)
    check(f'{tag} challenge: nothing else in the chart, and the note counts the 2 recipes left out', cr['n'] == 3 and cr['ok'] == 2 and cr['me'] == 1 and cr['stray'] == 0
          and cr['note'] == Z(lang, 'The other 2 recipes miss the goal and are not shown.', '其余 2 种做法没达到目标，图里没有列出。'), cr)
    check(f'{tag} challenge: baseline, effect, and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'initial setup', '初始设置'), Z(lang, 'Note:', '注意：'), 'GradScaler']), res[-400:])
    shot(pg, f'{tag}_challenge_met', full=phone)
    if phone: dock(pg, tag, 'challenge', Z(lang, 'Goal met', '达成目标'))
    cl = pg.evaluate("()=>[...document.querySelectorAll('#nf-ch-res .cmp-val')].map(e=>{const r=e.getBoundingClientRect(), lh=parseFloat(getComputedStyle(e).lineHeight)||12; return Math.round(r.height/Math.max(lh,11))})")
    check(f'{tag} challenge: each bar’s value on one line (review A12)', cl and max(cl) <= 1, cl)
    check(f'{tag} challenge: each row names its extra parts (review A11)', all(Z(lang, 'extra part', '个部件') in r for r in cmp), cmp)
    rules(pg, f'{tag} challenge met', lang); fit(pg, f'{tag} challenge met')
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#nf-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#nf-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#nf-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the submitted result', st['chR'] == 'bf16m' and st['chSubmitted'] and vis(pg, '#nf-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; return shown(document.querySelector('#nf-rail .next-lab'))||shown(document.querySelector('#view-dyn .next-lab-end .next-lab'))}""")
    check(f'{tag} challenge: the next lab (R3.11)', nl)
    # the drill and the code
    click(pg, '#nf-app [data-act="drill-reveal"]')
    check(f'{tag} drill: the answer shows', vis(pg, '#nf-drill-a') and len(text(pg, '#nf-drill-a')) > 40, text(pg, '#nf-drill-a')[:200])
    rules(pg, f'{tag} drill', lang)
    pg.close()


def run_teaser(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = b.new_page(viewport={'width': w, 'height': h}); open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'teaser {w} {lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    pg.evaluate("()=>{const b=document.querySelector('.ds-btn[data-world=\"roots\"]')||document.querySelector('.dg-btn[data-world=\"roots\"]'); if(b) b.click()}"); pg.wait_for_timeout(400)
    tz = pg.evaluate("()=>{const t=document.querySelector('.tz-nf'); if(!t) return null; const r=t.querySelector('.tz-read'), v=t.querySelector('svg'); return {read:r.innerText, words:r.innerText.split(/\\s+/).length, h:v.getBoundingClientRect().height, notes:[...t.querySelectorAll('.tz-note')].map(p=>p.textContent)}}")
    check(f'{tag} teaser: shown, one sentence ≤ 25 words, visual ≤ 130 px', tz is not None and (lang == 'zh' or tz['words'] <= 25) and tz['h'] <= 130, tz)
    if tz:
        pg.evaluate("()=>{const i=document.querySelector('.tz-nf input'); i.scrollIntoView({block:'center'}); i.value=window.__tzNf.xs.indexOf(1e-8); i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
        rd = pg.evaluate("()=>document.querySelector('.tz-nf .tz-read').innerText.replace(/\\u2011/g,'-')")
        check(f'{tag} teaser: 1e-8 becomes 0 in FP16 and E4M3, BF16 keeps it', '1e-8' in rd and 'FP16' in rd and ('1.001e-8' in rd), rd)
        check(f'{tag} teaser: no overflow', overflow(pg) <= 0, overflow(pg))
        pg.evaluate("()=>document.querySelector('.tz-nf').closest('.tile, section, div').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        shot(pg, f'{tag}_teaser')
    pg.close()



# ----- the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): G1, a lead-in that says what to pick and what to watch
# instead of “one at a time, they do not stack”; G2, no sentence leans on a detail of an earlier part, the recap states the pattern
OWNER_B9 = {'scen': ('Pick a scenario and see which formats can store its numbers.', '选一个场景，看看哪些格式存得下那里的数。'), 'chal': None, 'parts': [(4, [('Pick a fix and see whether the gradient and the update survive.', '选一种修法，看看梯度和更新能不能保住。'), ('Take one training step with everything stored in FP16', '看一个全用 FP16 存的训练步。')]), (5, [('Anything smaller rounds to 0', '比它还小的数会被舍入成 0'), ('Fewer mantissa bits mean a wider gap', '尾数位越少，间隔越宽'), ('Mixed-precision training in FP16 uses both.', '用 FP16 做混合精度训练时，两个修法都要用。'), ('so it loses small updates that FP16 keeps', '所以 BF16 会丢掉 FP16 留得住的小更新'), ('so very small gradients round to 0', '所以很小的梯度会被舍入成 0')])]}
OWNER_B9_NOTES = ('一次只能选', '一次选一个', '不会叠加', '每次只试', '下面的选项是单选', 'do not stack', 'One at a time', 'Pick one at a time', 'one at a time;')
OWNER_B9_BACK = re.compile(r'In part [0-9]|in part [0-9]|As in part|as in part|第 [0-9] 部分里|在第 [0-9] 部分|和第 [0-9] 部分一样|还是第 [0-9]')
owner_b9_once = []


def owner_b9(b, w, h, lang):
    if not owner_b9_once:
        owner_b9_once.append(1)
        src = (P / 'labs' / 'number-formats.js').read_text() + ((P / 'teasers' / 'number-formats.js').read_text() if (P / 'teasers' / 'number-formats.js').exists() else '')
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
    xseen, xbad = 0, []
    for mode, k in states:
        if k is None or k == 0:
            pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", mode); pg.wait_for_timeout(400)
        if k is not None:
            pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", k)
        pg.wait_for_timeout(900)
        r = pg.evaluate(B9V6_PROBE, [cfg['math'], cfg['block'], cfg.get('sep_extra', '')])
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
    check(f'{tag}: no page errors', not here, here[:3])

B9V6 = {'lab': 'number-formats', 'math': '#nf-mathbody', 'block': '.nf-mb', 'sep_extra': '#nf-chart-cap', 'src_gone': ['× 2^${sg(', '/ 2^${f.m}', '2<sup>${sg(ex)}', 'is 2⁻ᵐ for m mantissa bits', '是 2⁻ᵐ（m 是尾数位数）', 'is 2<sup>−m</sup>', '是 2<sup>−m</sup>', 'const pow2=e=>`2<sup>'], 'text_gone': ['2^', '2⁻ᵐ']}


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
            owner_b9(b, w, h, lang)
            owner_b9v6(b, w, h, lang, B9V6)
            run_teaser(b, w, h, lang)
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
