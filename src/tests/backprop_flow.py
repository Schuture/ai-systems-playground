"""Flow test for the backprop lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run and three more widths:
- no page errors and no horizontal overflow at any step; the phone first screen (m2): the parts' bar with Next, and the graph;
- the header: title, Toy run, the outcome line, both popovers;
- the prediction gate (R3.8): Next disabled with the reason until a prediction; the answer (z's gradient computed once, at step 3)
  appears nowhere before it, in text or tooltips; “Saved …” after a pick (R2.5);
- part 3: Next waits for the last backward step; the player (◂ ▸, play / pause, slow / medium / fast, the scrubber); the graph's
  height stays the same while stepping (R7); the formula box beside the graph (under it on a phone): hidden in parts 1 and 2, then
  the forward and backward formulas of the current step with their numbers, one title per step, one height, nothing cut off; then the result, the verdict with the right answer, and why (R3.10); a language switch
  keeps the part, the step and the prediction;
- part 4: starts at h = 0.5 (check fails), waits for h = 0.00001 (passes); part 5: waits for “=”, the result on the option's
  title row, w's two contributions and their sum on the graph; part 6: waits for an in-place change, PyTorch's error banner, the small
  engine's three failing checks; part 7: three lines with a picture each, three misconceptions, “See the scenarios”, the next lab;
- Settings hidden with a note saying what they are fixed to, in every guided part and in the challenge (R4.2);
- Scenarios: four, single choice, pressed state follows the settings, the URL carries the setup and a fresh page restores it, a
  change under Settings does not move the page (R3.2) and unpresses the scenario;
- the challenge: checks count, the three feedback tiers (R4.4), “show all options”, the chart and the table, the capability card,
  a language switch keeps the answer, “Another hidden bug”;
- the mini result bar at 390 once the graph scrolls away, with a short status;
- B4 (polish round 5): entering each part at 1440 × 900, no part needs scrolling inside the stage (Chinese with 30 px to spare),
  with the graph and the column beside it (the formula box and the check) on the screen; the stage's top never under the top bar;
- the writing rules on every visible text node and tooltip (R6), colours only for their data (R5.6), no text wider than its box and
  no help cursor without a tooltip (R7.8, R1.1), and no SVG box title wider than its box.
Screenshots: shots/backprop/.
Run: python3 tests/backprop_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/backprop-flow.html', '--only', 'backprop'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'backprop-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'backprop-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'backprop-flow-wrapped.html')
SH = P / 'shots' / 'backprop'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='backprop'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__bp.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
# the formula box beside the graph (review round 3): its title, its text, how many formulas, its height and where it sits
FX = lambda pg: pg.evaluate("""()=>{const b=document.getElementById('bp-fx'), g=document.getElementById('bp-gwrap'); if(!b) return null; const r=b.getBoundingClientRect(), q=g.getBoundingClientRect();
  return {vis:!b.hidden&&r.height>0, head:(b.querySelector('.bp-fxh')||{}).textContent||'', txt:b.textContent.replace(/\\u00a0/g,' '), math:b.querySelectorAll('math').length, h:Math.round(r.height),
    beside:r.left>=q.right-1, below:r.top>=q.bottom-1, wide:b.scrollWidth<=b.clientWidth+1}}""")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__bp||!window.__bp.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def nxt(pg): click(pg, '#bp-rail [data-nav="next"]:not([disabled])')


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#bp-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#bp-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)'],
   [/from above|below it|run down to|right to left|从右往左|上面传来|它下面|往下传|传下去/i,'a direction word that contradicts the graph, drawn top to bottom with the loss at the bottom (review A2, B6)'],
   [/\\bh = 0\\.(?:5|1|001|00001)(?!\\d)|Nudge h|推动量 h|nudging [−\\d]/,'the nudge is ε, not h (review A3, A10)'],[/partly a guess|有一半是猜的/,'“partly a guess” (review A1)']];
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #bp-mathbody, .num-big, #bp-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for a missed goal (R5.6)
# red is for failing checks, so it is allowed on the ✗ cells of the challenge table (td.bp-x), whose .bp-sig sits four levels up
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|bad|bp-err|bp-chk|bp-chip|bp-fan|bp-sig|bp-tan|err|warn|wrong|ex-res|(^| )x( |$)|(^| )bp-x( |$)/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#bp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const h=e.closest('math')?e.closest('math'):e;   /* a typeset formula (H1, 2026-10-09): its MathML takes the class of the line that holds the formula */
     const cls=(h.getAttribute('class')||'')+' '+(h.parentElement?h.parentElement.getAttribute('class')||'':'')+' '+(h.parentElement&&h.parentElement.parentElement?h.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
# R7.8: no text wider than its box (clipped or ellipsised), in the lab and the mini result bar
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#bp-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# R1.1: every element with a help cursor explains itself on hover (itself or an ancestor carries the tooltip)
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#bp-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#bp-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#bp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


# every text in a graph box fits inside its box, and no input tag sits on a contribution label
SVGFIT = """()=>{const out=[]; document.querySelectorAll('#bp-svg .bp-n').forEach(g=>{const r=g.querySelector('rect').getBBox();
   g.querySelectorAll('text').forEach(t=>{const b=t.getBBox(); if(b.x+b.width>r.x+r.width+0.5) out.push(g.dataset.n+': '+t.textContent+' ('+(b.x+b.width-r.x-r.width).toFixed(1)+' px too wide)');});});
  const hit=(a,b)=>a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height;
  const tags=[...document.querySelectorAll('#bp-svg .bp-tag')].map(t=>t.getBBox()), labs=[...document.querySelectorAll('#bp-svg .bp-cl rect')].map(t=>t.getBBox());
  tags.forEach(a=>labs.forEach(b=>{if(hit(a,b)) out.push('an input tag under a contribution label');}));
  return out;}"""


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    for h in pg.evaluate(SVGFIT): fit_hits.append(f'{where}: graph (R7.8): {h}')
    if pg.evaluate("()=>/\\b1 (checks|gradients|steps|parameters)\\b/.test(document.getElementById('bp-app').innerText)"):
        fit_hits.append(f'{where}: singular and plural')


# everything a learner can read before the prediction, outside Connect and the choices themselves (R3.8)
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#bp-app .labhead','#bp-rail','#bp-stage','#bp-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){
        const c=e.cloneNode(true); c.querySelectorAll('.choices').forEach(x=>x.remove()); t.push(c.innerText||c.textContent);
        e.querySelectorAll('[data-tipx]').forEach(x=>{if(!x.closest('.choices')) t.push(x.getAttribute('data-tipx'));});}}
      document.querySelectorAll('#bp-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#bp-rail .t, #bp-rail .def').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#bp-rail .t, #bp-rail .def')[{i}]; e.focus({{preventScroll:true}}); e.dispatchEvent(new MouseEvent('mouseover',{{bubbles:true}}))}}"); pg.wait_for_timeout(80)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out)


def lock_ok(pg, lang, words):
    hidden = not any(vis(pg, f'#bp-ctl-{k}-wrap') for k in ('net', 'over', 'tz', 'inpl', 'p', 'h'))
    return vis(pg, '#bp-lock') and hidden and all(Z(lang, *w) in text(pg, '#bp-lock') for w in words)


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#bp-app .labtitle') == Z(lang, 'How does autograd get every gradient in one backward pass?', 'autograd 怎么一趟反向算出所有梯度？'), text(pg, '#bp-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#bp-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#bp-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#bp-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#bp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#bp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#bp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    st = state(pg)
    check(f'{tag} part 1: two weights, no bugs, step 0', (st['net'], st['over'], st['tz'], st['inpl'], st['K'], st['step']) == ('two', False, False, 'none', 0, 0), st)
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    check(f'{tag} part 1: no formula box before part 3', not FX(pg)['vis'], FX(pg))
    check(f'{tag} part 1: 10 boxes in the graph, 4 of them parameters', pg.evaluate("()=>[document.querySelectorAll('#bp-svg .bp-n').length,document.querySelectorAll('#bp-svg .bp-n.bp-par').length]") == [10, 4])
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#bp-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), g=document.getElementById('bp-svg');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(g); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, graph:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the graph starts on screen (R7.3)', fs['bar'] and fs['next'] and fs['graph'] is not None and fs['graph'] < fs['vh'] - 120, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: Settings hidden with a note saying what they are fixed to (R4.2)', lock_ok(pg, lang, [('two weights', '两个权重'), ('no bugs', '没有 bug')]), text(pg, '#bp-lock'))
    ans = [Z(lang, 'computed once', '只在第 3 步算了一次'), Z(lang, 'share it', '共用它')]
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer to part 2 appears nowhere, not in popovers or term tooltips (R3.8)', not any(a in ba for a in ans), [x for x in ba.split('\n') if any(a in x for a in ans)][:2])
    # part 2: the gate
    nxt(pg)
    check(f'{tag} part 2 reached, Next disabled before a prediction, with the reason', state(pg)['step'] == 1 and all(nexts(pg)) and pg.evaluate("()=>[...document.querySelectorAll('#bp-rail .bp-why')].some(p=>!p.hidden&&p.textContent.length>5)"), nexts(pg))
    ba = before_answer(pg)
    check(f'{tag} part 2: the answer appears nowhere before the prediction (R3.8)', not any(a in ba for a in ans), [x for x in ba.split('\n') if any(a in x for a in ans)][:2])
    check(f'{tag} part 2: no formula box before part 3', not FX(pg)['vis'], FX(pg))
    check(f'{tag} part 2: three choices, none pressed', pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-pred]')].map(b=>b.getAttribute('aria-pressed'))") == ['false'] * 3)
    shot(pg, f'{tag}_p2', full=phone)
    rules(pg, f'{tag} part 2', lang); fit(pg, f'{tag} part 2')
    click(pg, '#bp-rail [data-pred="each"]')
    check(f'{tag} part 2: “Saved …” after a pick (R2.5), Next enabled', Z(lang, 'Saved. You can still change it', '已记下。') in text(pg, '#bp-rail') and not any(nexts(pg)) and state(pg)['pred'] == 'each')
    # part 3: walk
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 3: starts before the backward pass, Next waits', st['step'] == 2 and st['K'] == 0 and all(nexts(pg)) and not obs(pg), (st['K'], nexts(pg)))
    check(f'{tag} part 3: the player: back, step, play, the scrubber, three speeds', pg.evaluate("()=>{const p=document.getElementById('bp-st-play'); return !!p&&p.querySelectorAll('[data-play]').length===3&&!!p.querySelector('input[type=range]')&&p.querySelectorAll('.bp-speed button').length===3}"))
    ba = before_answer(pg)
    check(f'{tag} part 3: no result before the last step (R3.8)', not any(a in ba for a in ans))
    fx = FX(pg)
    check(f'{tag} part 3: the formula box shows the loss and ∂L/∂L = 1 before the first step, {"under" if phone else "beside"} the graph, nothing cut off (review round 3)',
          fx['vis'] and fx['head'] == Z(lang, 'Before the backward pass', '反向开始之前') and fx['math'] >= 3 and '∂L∂L=1' in fx['txt'].replace(' ', '') and (fx['below'] if phone else fx['beside']) and fx['wide'], fx)
    hg = [pg.evaluate("()=>Math.round(document.getElementById('bp-graph').getBoundingClientRect().height)")]
    fh, heads = [fx['h']], []
    for k in range(6):
        click(pg, '#bp-ic-play [data-play="fwd"]')
        hg.append(pg.evaluate("()=>Math.round(document.getElementById('bp-graph').getBoundingClientRect().height)"))
        fx = FX(pg); fh.append(fx['h']); heads.append(fx['head'])
        if not fx['wide']: check(f'{tag} part 3 step {k + 1}: the formula box is not cut off', False, fx)
        if k == 2:
            check(f'{tag} part 3 step 3: the formula box shows h = tanh(z) with its numbers and the gradient it passes to z', 'tanh' in fx['txt'] and '∂L∂z' in fx['txt'].replace(' ', '') and '0.6044' in fx['txt'], fx['txt'][:200])
        if k == 2:
            check(f'{tag} part 3: after 3 steps, 0 of 4 gradients and the outlined box is h', state(pg)['K'] == 3 and pg.evaluate("()=>document.querySelector('#bp-svg .bp-n.bp-on').dataset.n") == 'h' and not obs(pg)[:3].strip() == '')
            shot(pg, f'{tag}_p3_step3', full=phone)
            fit(pg, f'{tag} part 3 step 3')
    check(f'{tag} part 3: the graph keeps its height while stepping (R7)', max(hg) - min(hg) <= 1, hg)
    check(f'{tag} part 3: the formula box names each step and keeps its height (R7.8)', max(fh) - min(fh) <= 1 and heads == [Z(lang, f'Backward step {k} of 6', f'第 {k} 个反向步骤（共 6 个）') for k in range(1, 7)], (fh, heads))
    o = obs(pg)
    check(f'{tag} part 3: after the last step the result, the verdict with the right answer, and why (R3.10)', state(pg)['K'] == 6 and not any(nexts(pg)) and Z(lang, 'computed once, at step 3', '只在第 3 步算了一次') in o
          and Z(lang, 'The right answer is “Once, and w₁, w₂ and b share it”', '正确答案是“一次，w₁、w₂ 和 b 共用它”') in o and Z(lang, 'Why:', '原因：') in o, o[:300])
    click(pg, '#bp-st-play [data-play="back"]')
    check(f'{tag} part 3: one step back', state(pg)['K'] == 5)
    click(pg, '#bp-st-play .bp-speed button[data-v="fast"]')
    click(pg, '#bp-st-play [data-play="play"]')
    check(f'{tag} part 3: play shows pause while playing', state(pg)['playing'] and text(pg, '#bp-st-play [data-play="play"]').startswith('❚❚'))
    pg.wait_for_timeout(1600)
    st = state(pg)
    check(f'{tag} part 3: play runs to the end and stops, fast = 1.2 s per step', st['K'] == 6 and not st['playing'] and st['speed'] == 'fast', st)
    check(f'{tag} part 3: at the end the play button says replay (R3.20)', text(pg, '#bp-st-play [data-play="play"]') == Z(lang, '↺ replay', '↺ 重放'), text(pg, '#bp-st-play [data-play="play"]'))
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} part 3: a language switch keeps the part, the step and the prediction', st['step'] == 2 and st['K'] == 6 and st['pred'] == 'each' and bool(obs(pg)), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # R3.20: back in part 3 after the whole pass, it shows finished, no reset to step 0
    pg.evaluate("()=>document.querySelector('#bp-rail [data-go=\"1\"]').click()"); pg.wait_for_timeout(300); settle(pg)
    pg.evaluate("()=>document.querySelector('#bp-rail [data-go=\"2\"]').click()"); pg.wait_for_timeout(300); settle(pg)
    st = state(pg)
    check(f'{tag} part 3, entered again: the pass shows finished, no replay (R3.20)', st['step'] == 2 and st['K'] == 6 and not st['playing'] and bool(obs(pg)), (st['step'], st['K']))
    # part 4: check
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 4: starts at h = 0.5, the check of w₁ fails, Next waits', st['step'] == 3 and st['hi'] == 0 and st['p'] == 'w1' and all(nexts(pg)) and vis(pg, '#bp-check') and 'bad' in pg.evaluate("()=>document.getElementById('bp-status').className"), st)
    check(f'{tag} part 4: the check panel draws the curve, both slopes and the two points', pg.evaluate("()=>['.bp-curve','.bp-tan','.bp-sec','.bp-dot'].every(s=>document.querySelector('#bp-cksvg '+s))"))
    shot(pg, f'{tag}_p4_big_h', full=phone)
    rules(pg, f'{tag} part 4 h 0.5', lang); fit(pg, f'{tag} part 4 h 0.5')
    for v in ('1', '2', '3'):
        click(pg, f'#bp-ic-h button[data-v="{v}"]')
    o = obs(pg)
    check(f'{tag} part 4: at h = 0.00001 the check passes, the result and why', state(pg)['hi'] == 3 and not any(nexts(pg)) and 'ok' in pg.evaluate("()=>document.getElementById('bp-status').className") and Z(lang, 'far below', '远低于') in o and Z(lang, 'billions of parameters', '参数多得多，逐个检查太慢') in o and '16.1B' not in o, o[:300])
    click(pg, '#bp-ic-p button[data-v="v"]')
    check(f'{tag} part 4: another parameter, the check follows', state(pg)['p'] == 'v' and Z(lang, 'Gradient check of v', 'v 的梯度检查') in text(pg, '#bp-ck-cap'))
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4')
    # review v6 BP-S3: part 4's finite difference is a formula on its own line, θ named before it
    ml4 = pg.evaluate("()=>[...document.querySelectorAll('#bp-rail .step.current .bp-ml math')].length")
    check(f'{tag} part 4: the finite difference on its own line (review v6 BP-S3)', ml4 == 1 and 'θ' in text(pg, '#bp-rail .step.current'), ml4)
    # part 5: fan-out
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 5: one weight used twice, += first, Next waits for =', st['step'] == 4 and st['net'] == 'tied' and not st['over'] and all(nexts(pg)), st)
    check(f'{tag} part 5: the graph shows w’s two contributions and their sum', '−0.4257 + 1.0643' in pg.evaluate("()=>[...document.querySelectorAll('#bp-svg .bp-fant')].map(t=>t.textContent).join(' ')"))
    click(pg, '#bp-rail [data-lever="over"]')
    o = obs(pg)
    lv = pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 5: “=” fails w’s check, both results on the options’ title rows (R3.6)', state(pg)['over'] and not any(nexts(pg)) and lv == [Z(lang, 'check passes', '检查通过'), Z(lang, 'check fails', '检查不通过')] and '1.0643' in o, (lv, o[:200]))
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    # part 6: in place
    nxt(pg)
    st = state(pg)
    check(f'{tag} part 6: no in-place change first, Next waits', st['step'] == 5 and st['inpl'] == 'none' and all(nexts(pg)), st)
    click(pg, '#bp-rail [data-lever="pt"]')
    check(f'{tag} part 6: PyTorch stops with the in-place error', vis(pg, '#bp-err') and 'RuntimeError' in text(pg, '#bp-err') and 'modified by an inplace operation' in text(pg, '#bp-err') and not any(nexts(pg)))
    shot(pg, f'{tag}_p6_pt', full=phone)
    rules(pg, f'{tag} part 6 PyTorch', lang); fit(pg, f'{tag} part 6 PyTorch')
    click(pg, '#bp-rail [data-lever="eng"]')
    chips = pg.evaluate("()=>[...document.querySelectorAll('#bp-chips .bp-chip')].map(c=>c.className.includes('bad'))")
    check(f'{tag} part 6: the small engine reads h = 0.5, all three checks fail', not vis(pg, '#bp-err') and chips == [True, True, True] and Z(lang, 'clipped, was 0.5649', '截断了，原为 0.5649') in pg.evaluate("()=>[...document.querySelectorAll('#bp-svg .bp-tag')].map(t=>t.textContent).join(' ')").replace('‑', '-') or (state(pg)['K'] == 3), chips)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6')
    # part 7: remember
    nxt(pg)
    t7 = text(pg, '#bp-rail')
    check(f'{tag} part 7: four lines with a picture each (review v6 BP-S11), three misconceptions, the button to the scenarios (R3.12)', pg.evaluate("()=>document.querySelectorAll('#bp-rail [data-rem]').length") == 4 and t7.count('✗') == 3 and vis(pg, '#bp-rail [data-mode-go="sandbox"]'), t7[:200])
    click(pg, '#bp-rail [data-rem="check"]')
    st = state(pg)
    check(f'{tag} part 7: “Show in the chart” puts part 4’s check (h = 0.5) back in the chart', st['remView'] == 'check' and vis(pg, '#bp-check') and '0.5' in text(pg, '#bp-ck-cap'), (st['remView'], text(pg, '#bp-ck-cap')))
    y0 = pg.evaluate("()=>scrollY"); click(pg, '#bp-rail [data-rem="check"]')
    check(f'{tag} part 7: pressing the same button again does not scroll (R1.11)', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    click(pg, '#bp-rail [data-rem="fan"]')
    check(f'{tag} part 7: the overwrite bug’s picture', state(pg)['remView'] == 'fan' and state(pg)['over'])
    click(pg, '#bp-rail [data-rem="inpl"]')
    st = state(pg)
    check(f'{tag} part 7: the in-place line has its own picture, h clipped in the small engine (review v6 BP-S11)', st['remView'] == 'inpl' and st['inpl'] == 'eng' and not st['over'] and st['net'] == 'tied', st)
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#bp-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e) : shown(r)}""")
    check(f'{tag} part 7: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p7', full=phone)
    rules(pg, f'{tag} part 7', lang); fit(pg, f'{tag} part 7')
    check(f'{tag} guided: no overflow at the end', overflow(pg) <= 0, overflow(pg))
    if phone:
        go = lambda i: (pg.evaluate(f"()=>document.querySelector('#bp-rail [data-go=\"{i}\"]').click()"), pg.wait_for_timeout(300), settle(pg))
        go(2)
        pg.evaluate("()=>{const t=document.getElementById('bp-svg'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden&&d.getBoundingClientRect().height>0, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the graph scrolls away, a short status and the step', dk['shown'] and len(dk['status']) <= 16 and Z(lang, 'Step', '步') in dk['read'], dk)
        check(f'{tag} mini result bar: no overflow', overflow(pg) <= 0, overflow(pg))
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang); fit(pg, f'{tag} dock')
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    # R3.20: part 3's pass (the initial setup) watched to the end in Scenarios opens finished in part 3 and counts for its Next
    click(pg, '#bp-st-play .bp-speed button[data-v="fast"]')
    click(pg, '#bp-st-play [data-play="play"]')
    st0 = state(pg)
    try: pg.wait_for_function("()=>{const s=window.__bp.state(); return s.K>=6&&!s.playing}", timeout=20000)
    except Exception: pass
    st = state(pg)
    check(f'{tag} R3.20: in Scenarios, ↺ replay plays the initial setup’s pass to the end', st0['playing'] and st0['K'] < 6 and st['K'] == 6 and not st['playing'], (st0['K'], st['K'], st['playing']))
    mode(pg, 'guided'); nxt(pg); click(pg, '#bp-rail [data-pred="each"]'); nxt(pg)
    st = state(pg)
    check(f'{tag} R3.20: part 3 after the same pass was watched in Scenarios: finished at once, no reset to step 0, Next open', st['step'] == 2 and st['K'] == 6 and not st['playing'] and not any(nexts(pg)) and bool(obs(pg)), (st['step'], st['K'], nexts(pg)))
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>document.querySelectorAll('#bp-rail [data-preset]').length")
    check(f'{tag} scenarios: four, single choice (R2.1), the lead-in (G1, 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see whether its gradients pass their checks.', '选一个场景，看看它的梯度能不能通过检查。') in text(pg, '#bp-rail') and Z(lang, 'Pick a scenario', '选一个应用场景') in text(pg, '#bp-rail'))
    check(f'{tag} scenarios: Settings shown', vis(pg, '#bp-ctl-net-wrap') and vis(pg, '#bp-ctl-h-wrap') and not vis(pg, '#bp-lock'))
    want = [('two', False, 'none', 'w1'), ('tied', False, 'none', 'w'), ('tied', True, 'none', 'w'), ('tied', False, 'eng', 'v')]
    for i in range(presets):
        click(pg, f'#bp-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        st = state(pg); ins = text(pg, '#bp-insight')
        check(f'{tag} scenario {i}: pressed alone, its setup, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and (st['net'], st['over'], st['inpl'], st['p']) == want[i] and len(ins) > 80, (pressed, st, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        if i in (1, 3):
            d = pg.evaluate("()=>{const e=document.querySelector('#bp-metrics [data-k=\"grad\"] .d'); return e?e.textContent:''}")
            check(f'{tag} scenario {i}: no “vs last” between two different parameters or networks (review A4, B1)', d == '', d)
        shot(pg, f'{tag}_scenario{i}', full=phone)
        rules(pg, f'{tag} scenario {i}', lang); fit(pg, f'{tag} scenario {i}')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#backprop~tied_0_0_eng_v_3', hsh)
    pg2 = page(b, w, h, lang, hash_='backprop~tied_1_0_none_w_2')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, one weight, =, h = 0.001, no scenario pressed', st['mode'] == 'sandbox' and (st['net'], st['over'], st['inpl'], st['p'], st['hi']) == ('tied', True, 'none', 'w', 2) and 'true' not in pg2.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))"), st)
    pg2.close()
    pg.evaluate("()=>document.querySelector('#bp-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    y0 = pg.evaluate("()=>scrollY")
    click(pg, '#bp-ctl-h button[data-v="1"]')
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed', 'true' not in pressed and state(pg)['hi'] == 1, pressed)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page (R3.2)', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    pg.evaluate("()=>document.querySelector('#bp-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    click(pg, '#bp-ctl-tz button[data-v="1"]')
    check(f'{tag} scenarios: the tanh′ typo makes checks fail', state(pg)['tz'] and 'true' in pg.evaluate("()=>[...document.querySelectorAll('#bp-chips .bp-chip')].map(c=>String(c.className.includes('bad')))"))
    rules(pg, f'{tag} settings', lang); fit(pg, f'{tag} settings')
    shot(pg, f'{tag}_settings', full=phone)
    # the challenge
    mode(pg, 'challenge')
    st = state(pg)
    hid = pg.evaluate("s=>window.__bp.hidden(s)", st['chSeed'])
    chf = pg.evaluate("()=>[...document.querySelectorAll('#bp-rail .bp-mls .bp-ml math')].length")
    check(f'{tag} challenge: the network’s four formulas on their own lines, σ defined, no “every check counts” (review v6 BP-S1, S2)', chf == 4 and Z(lang, 'σ is the sigmoid function', 'σ 是 sigmoid 函数') in text(pg, '#bp-rail') and not any(x in text(pg, '#bp-rail') for x in ('counts', '计数', '算数')), chf)
    check(f'{tag} challenge: no checks yet, Settings hidden with a note (R4.2)', st['mode'] == 'challenge' and st['chChecks'] == [] and vis(pg, '#bp-lock') and not vis(pg, '#bp-ctl-net-wrap'), st)
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#bp-rail [data-exits]'))
    HID = """()=>({math:document.getElementById('bp-mathbody').textContent, player:!document.getElementById('bp-pwrap').hidden, fan:document.querySelectorAll('#bp-svg .bp-fant').length, lab:document.querySelectorAll('#bp-svg .bp-cl').length,
      grads:[...document.querySelectorAll('#bp-svg .bp-n')].map(g=>g.dataset.n+':'+g.querySelector('.bp-t3').textContent), on:document.querySelectorAll('#bp-svg .bp-n.bp-on').length, fx:!document.getElementById('bp-fx').hidden})"""
    for seed in range(4):
        hv = pg.evaluate(HID)
        check(f'{tag} challenge (hidden bug {pg.evaluate("s=>window.__bp.hidden(s)", state(pg)["chSeed"])}): before the answer nothing shows the hand-written backward: no arithmetic, no player, no fan-out label, no gradient (review A1, R3.8)',
              '∂L∂' not in hv['math'].replace(' ', '') and not hv['player'] and not hv['fx'] and hv['fan'] == 0 and hv['lab'] == 0 and hv['on'] == 0 and all(x.endswith('?') for x in hv['grads']), hv)
        pg.evaluate("()=>{window.__bp; document.querySelector('#bp-rail [data-act=\"another\"]')}")
        pg.evaluate("()=>{const p=document.querySelector('#bp-ch-pick [data-lever=\"0\"]'); p.click(); document.querySelector('#bp-rail [data-act=\"submit\"]').click()}"); pg.wait_for_timeout(200)
        if pg.evaluate("()=>!!document.querySelector('#bp-ch-res [data-act=\"reveal\"]')"): pg.evaluate("()=>document.querySelector('#bp-ch-res [data-act=\"reveal\"]').click()"); pg.wait_for_timeout(200)
        hv = pg.evaluate(HID)
        check(f'{tag} challenge: once the answer is shown, the hand-written arithmetic, the player and the gradients are back', '∂L∂' in hv['math'].replace(' ', '') and hv['player'] and hv['fx'] and not any(x.endswith('?') for x in hv['grads']), hv)
        pg.evaluate("()=>document.querySelector('#bp-rail [data-act=\"another\"]').click()"); pg.wait_for_timeout(250)
    st = state(pg); hid = pg.evaluate("s=>window.__bp.hidden(s)", st['chSeed'])
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang); fit(pg, f'{tag} challenge')
    fails_of = lambda i: pg.evaluate("i=>window.__bp.fails(i)", i)
    mn = pg.evaluate("h=>window.__bp.minChecks(h)", hid)
    wrong = (hid + 1) % 4
    # tier 3: wrong
    click(pg, '#bp-chks [data-chk="c"]')
    check(f'{tag} challenge: a check counts and shows pass or fail', state(pg)['chChecks'] == ['c'] and text(pg, '#bp-ch-n') == '1')
    click(pg, f'#bp-ch-pick [data-lever="{wrong}"]'); click(pg, '#bp-rail [data-act="submit"]')
    res = text(pg, '#bp-ch-res')
    check(f'{tag} challenge tier 3: wrong, a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not this one', '不是这个')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#bp-ch-res [data-act="reveal"]')
          and text(pg, '#bp-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案') and not vis(pg, '#bp-ch-res .cmp'), res[:200])
    click(pg, '#bp-ch-res [data-act="reveal"]')
    # the owner's review of batch 8 (2026-10-07): the answer is the chart of the hidden bug's checks and a table of two rows, the hidden
    # bug next to the pick, not all four candidates
    tb = pg.evaluate("()=>[...document.querySelectorAll('#bp-ch-res .bp-sig tbody tr')].map(r=>({on:r.classList.contains('bp-on'),t:r.cells[0].textContent,x:[...r.cells].slice(1).map(c=>c.classList.contains('bp-x'))}))")
    check(f'{tag} challenge: “Show the answer” gives the chart and a table of the hidden bug and the pick only, each with its own failing checks',
          vis(pg, '#bp-ch-res .cmp') and len(tb) == 2 and tb[0]['on'] and Z(lang, '(hidden here)', '（这次藏的）') in tb[0]['t'] and Z(lang, '(your pick)', '（你选的）') in tb[1]['t']
          and [p for p, x in zip(['w', 'b', 'v', 'c'], tb[0]['x']) if x] == sorted(fails_of(hid), key=['w', 'b', 'v', 'c'].index) and [p for p, x in zip(['w', 'b', 'v', 'c'], tb[1]['x']) if x] == sorted(fails_of(wrong), key=['w', 'b', 'v', 'c'].index), tb)
    check(f'{tag} challenge: the table title names the two rows, not every candidate bug', text(pg, '#bp-ch-res .bp-sig .cmp-title') == Z(lang, 'Which checks fail under the hidden bug and under your pick', '这次藏的 bug 和你选的 bug 各让哪些检查不通过'), text(pg, '#bp-ch-res .bp-sig .cmp-title'))
    shot(pg, f'{tag}_challenge_wrong', full=phone)
    rules(pg, f'{tag} challenge wrong', lang); fit(pg, f'{tag} challenge wrong')
    # tier 2: right with more checks than needed
    for p in ('w', 'b', 'v'):
        if pg.evaluate(f"()=>!document.querySelector('#bp-chks [data-chk=\"{p}\"]').disabled"): click(pg, f'#bp-chks [data-chk="{p}"]')
    click(pg, f'#bp-ch-pick [data-lever="{hid}"]'); click(pg, '#bp-rail [data-act="submit"]')
    res = text(pg, '#bp-ch-res')
    check(f'{tag} challenge tier 2: right, but fewer checks were enough', res.startswith(Z(lang, 'Right', '答对了')) and Z(lang, 'enough', '就够') in res and state(pg)['chSubmitted'], res[:220])
    # tier 1: a new game with the fewest checks
    click(pg, '#bp-rail [data-act="another"]')
    st = state(pg); hid = pg.evaluate("s=>window.__bp.hidden(s)", st['chSeed']); mn = pg.evaluate("h=>window.__bp.minChecks(h)", hid)
    check(f'{tag} challenge: another hidden bug resets the checks', st['chChecks'] == [] and not st['chSubmitted'], st)
    for p in mn: click(pg, f'#bp-chks [data-chk="{p}"]')
    click(pg, f'#bp-ch-pick [data-lever="{hid}"]'); click(pg, '#bp-rail [data-act="submit"]')
    res = text(pg, '#bp-ch-res')
    check(f'{tag} challenge tier 1: right with the fewest checks', res.startswith(Z(lang, 'Right', '答对了')) and Z(lang, 'fewest', '最少') in res and 'ok' in pg.evaluate("()=>document.getElementById('bp-status').className"), res[:220])
    rows = pg.evaluate("()=>[...document.querySelectorAll('#bp-ch-res .cmp-row')].map(r=>r.classList.contains('ok'))")
    check(f'{tag} challenge chart: one row per parameter, passing rows marked', len(rows) == 4 and sum(rows) == 4 - len(fails_of(hid)), rows)
    check(f'{tag} challenge right: no table of the candidate bugs (the chart already shows the hidden bug’s checks)', not vis(pg, '#bp-ch-res .bp-sig'))
    shot(pg, f'{tag}_challenge_right', full=phone)
    rules(pg, f'{tag} challenge right', lang); fit(pg, f'{tag} challenge right')
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#bp-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#bp-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#bp-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the result', st['chSubmitted'] and st['chPick'] == hid and sorted(st['chChecks']) == sorted(mn) and vis(pg, '#bp-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    # Connect: the drill
    click(pg, '[data-act="drill-reveal"]')
    check(f'{tag} connect: the drill’s answer, its formulas typeset (H1, 2026-10-09)', vis(pg, '#bp-drill-a') and '∂L∂w' in pg.evaluate("()=>document.getElementById('bp-drill-a').textContent.replace(/\\s/g,'')") and pg.evaluate("()=>document.querySelectorAll('#bp-drill-a math').length") >= 6)
    rules(pg, f'{tag} connect', lang)
    pg.close()



# H1, H2 and H5, the owner's site-wide decisions of 2026-10-09: formulas typeset as MathML (the graph's operations in the line under
# the graph, the box titles in the SVG only short names), Show the math as short blocks, and a hairline above the operations line.
def h_page(b, lang):
    pg = page(b, 1440, 900, lang); mode(pg, 'sandbox')
    pg.evaluate("()=>{const d=document.getElementById('bp-math'); if(d&&!d.open) d.querySelector('summary').click()}"); pg.wait_for_timeout(300)
    hv = pg.evaluate("""()=>{const mb=document.getElementById('bp-mathbody'), ops=document.getElementById('bp-ops');
      return {blk:mb.querySelectorAll('.bp-mblk').length, math:mb.querySelectorAll('math').length, tex:/\\\\(frac|text|cdot|sqrt)/.test(mb.textContent),
        ops:ops?ops.querySelectorAll('math').length:0, opsLine:ops?getComputedStyle(ops).borderTopWidth:'', titles:[...document.querySelectorAll('#bp-svg .bp-t1')].map(e=>e.textContent),
        tips:[...document.querySelectorAll('[data-tipx]')].filter(e=>/∂L\/∂|\(y − t\)²|L\(θ \+ ε\)/.test(e.getAttribute('data-tipx'))).length}}""")
    check(f'H1 {lang}: the operations under the graph are typeset (MathML)', hv['ops'] >= 4, hv)
    check(f'H1 {lang}: the SVG box titles keep short names, no formula', hv['titles'] and not any('=' in t for t in hv['titles']), hv['titles'])
    check(f'H1 {lang}: no plain-text formula left in a hover', hv['tips'] == 0, hv)
    check(f'H2 {lang}: Show the math is short blocks with typeset formulas and no TeX source shown', hv['blk'] >= 2 and hv['math'] >= 6 and not hv['tex'], hv)
    check(f'H5 {lang}: a hairline above the operations line', hv['opsLine'] == '1px', hv['opsLine'])
    pg.close()

# the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): in every guided part, Scenarios and the
# challenge, with each Show the math open, no box in it scrolls sideways and it is no wider than its own box (as tests/kit_math_scroll.py)
MSCROLL = """()=>{const out=[]; for(const d of document.querySelectorAll('details.math')){if(!d.getClientRects().length) continue; const was=d.open; d.open=true;
    for(const e of d.querySelectorAll('*')){if(!/auto|scroll/.test(getComputedStyle(e).overflowX)||!e.getClientRects().length) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(e.className+' by '+by+' px: '+(e.textContent||'').trim().replace(/\\s+/g,' ').slice(0,40));}
    if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px'); d.open=was;}
  return out;}"""


def math_scroll(b, w, h, lang):
    pg = page(b, w, h, lang); bad = []
    n = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", '#bp-rail')
    for k in range(n):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", ['#bp-rail', k]); pg.wait_for_timeout(500)
        bad += [f'part {k + 1}: {x}' for x in pg.evaluate(MSCROLL)]
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(600)
        bad += [f'{m}: {x}' for x in pg.evaluate(MSCROLL)]
    check(f'{w} {lang}: no Show the math box scrolls sideways (all {n} guided parts, Scenarios, the challenge)', n >= 5 and not bad, (n, bad[:4]))
    pg.close()


# B4 (polish round 5, reviews/r5-fit-roots.md): entering each part at 1440 × 900 as the polish probe stage.py does (Next when it is
# enabled, else the part's heading), the stage needs no scrolling inside, Chinese with at least 30 px to spare, English with none
# (spare = the stage's max-height minus what its content and borders need; before this round parts 1, 3 and 4–7 scrolled 224, 153
# and 200 px in Chinese, parts 4–7 223 px in English), the graph and the column beside it (the metric cards, the formula box, the
# check or the error) are whole on the screen (part 1 used to put the metric cards above the graph, its lower half off the screen),
# and the stage's top is never under the top bar (since 精装修 step 2 the bar scrolls away with a lab page on a desktop: what of it is still
# on the screen, its bottom edge, not its height).
FIT = """()=>{const s=document.getElementById('bp-stage'), r=s.getBoundingClientRect(), tb=document.querySelector('.topbar'), bar=tb?Math.max(0,tb.getBoundingClientRect().bottom):0, top=Math.max(bar,r.top), bot=Math.min(innerHeight,r.bottom);
 const on=id=>{const e=document.getElementById(id); if(!e||!e.getClientRects().length) return null; const q=e.getBoundingClientRect(); return q.top>=top-1&&q.bottom<=bot+1;};
 return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:Math.round(bar), graph:on('bp-svg'), side:on('bp-side')}}"""


def fit_page(b, lang):
    """the page as the polish probe stage.py opens it: window.__aspPrimer set (without it, automation gets the older tests' compat layout,
    whose step list also holds the settings panel), the guided parts picked in the step list, the real clock"""
    pg = b.new_page(viewport={'width': 1440, 'height': 900}); pg.add_init_script('window.__aspPrimer=true')
    pg.on('pageerror', lambda e: errs.append(f'fit 1440 {lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='backprop'}"); pg.wait_for_timeout(500)
    g = pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if g.count(): g.first.click(); pg.wait_for_timeout(700)
    return pg


def fit_entry(b, lang):
    pg = fit_page(b, lang)
    for k in range(1, 8):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#bp-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900); settle(pg)
        f = pg.evaluate(FIT); f['step'] = state(pg)['step'] + 1
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}, the graph and the column beside it on the screen, the top below the top bar (B4, R7.12)',
              f['step'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['graph'] and f['side'] is not False and f['top'] >= f['bar'] - 1, f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: fit_entry(b, lang)
    for w, h, lang in ((1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'zh'), (390, 844, 'en')):
        math_scroll(b, w, h, lang)
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
    for lang in LANGS:
        h_page(b, lang)
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
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'backprop.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time', 'bp-one']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a way to write it and see whether w’s gradient passes its check.', '选一种写法，看看 w 的梯度能不能通过检查。', 'Pick a case and see whether the 3 gradients still pass their checks.', '选一种情况，看看 3 个梯度还能不能通过检查。', 'Pick a scenario and see whether its gradients pass their checks.', '选一个场景，看看它的梯度能不能通过检查。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('which part 3 showed is right', 'The cause is the large ε, not the backward pass: autograd’s gradient is right.'), ('不在反向（第 3 部分的反向是对的）', '问题出在 ε 太大，不在反向，autograd 的梯度是对的。'), ('In part 3, ${two.n} steps', 'so one walk from the loss back to the parameters fills in all their gradients'), ('第 3 部分里，从 loss 走回参数一共', '所以从 loss 走回参数一趟，参数再多，梯度也全部算好。'), ('In part 4, ε = 0.00001 matched', 'With a large ε the two points are too far apart for the line to follow the curve'), ('第 4 部分里，ε = 0.00001 时', 'ε 太大时，两点离得太远，直线跟不上曲线的弯曲，对的反向也会显得不对。'), ('In part 5, = instead of += turned', 'Writing = instead of += keeps only the last contribution'), ('第 5 部分里，把 += 写成 =，w 的梯度从', '把 += 写成 =，只留下最后一份，梯度就错了，程序却不会报错。'), ('In part 6, clipping h after the forward', 'The backward reads the value as it is now'), ('第 6 部分里，前向之后截断 h', '反向读的是它现在的值，原地改了它，用到它的梯度就全错了。'), ('In part 3, the gradient of z was computed once', 'and a gradient that several parameters share is computed only once.'), ('第 3 部分里，z 的梯度只算了一次', '几个参数共用的那段梯度也只算一次。'), ('in part 5 the overwrite bug left b and v passing', 'only that value’s check fails, and the others still pass.'), ('第 5 部分的覆盖 bug 里', '只有它自己的检查不通过，别的照样通过。'), ('w’s gradient in part 5 is', 'one contribution for each use.'), ('第 5 部分里，w 的梯度是', '每用一次加一份。'), ('In part 5, why must w’s gradient', 'When one weight w is used in two places, why must its gradient add the two contributions?'), ('第 5 部分里，为什么 w 的梯度', '一个权重 w 在两处用到时，为什么它的梯度要把两份贡献加起来？'), ('through x₁ positive in part 5', 'The two contributions usually differ, even in sign'), ('第 5 部分里经过 x₂ 的是负的', '两份贡献一般不一样，连正负都可能相反'), ('overwrite bug of part 5', 'Keeping only the last one is the overwrite bug: = in place of +=.'), ('就是第 5 部分的覆盖 bug', '只留最后一份，就是把 += 写成 = 的覆盖 bug。'), ('Shows part 3’s graph', 'Shows the graph after the whole backward pass.'), ('在图里显示第 3 部分走完整趟反向的计算图', '在图里显示走完整趟反向的计算图。'), ('Shows part 4’s check', 'Shows the check of w₁ with ε = 0.5.'), ('Shows part 5’s overwrite bug', 'Shows the overwrite bug, = in place of +=, and its failing check.'), ('Shows part 6’s h clipped', 'Shows h clipped in place in the small engine, and the failing checks.'), ('在图里显示第 6 部分', '在图里显示小引擎里 h 被原地截断后，检查不通过。')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# H1, the owner's decision of 2026-10-09: no plain-text formula left in the lab's source (EN and ZH); the formulas are typeset with A.ui.math.
for _bad in ['L = (y − t)²', 'z = m₁ + m₂ + b', '(L(θ + ε) − L(θ − ε)) ÷ 2ε', 'The relative error is |a − n|', 'dy/dx', '有限差分是直接从 loss 量出来的斜率：(L(θ', 'parameter, written ∂L/∂w', '梯度 ∂L/∂w 表示', 'at the loss itself: ∂L/∂L = 1', 'loss 本身开始：∂L/∂L = 1', 'ough several paths, ∂L/∂w is the sum']:
    check(f'H1: the plain-text formula “{_bad}” is gone (typeset instead)', _bad not in _G_SRC)
for _new in ["A.ui.math(t)", "FD_TEX='\\\\frac{L(\\\\theta + \\\\varepsilon) - L(\\\\theta - \\\\varepsilon)}{2\\\\varepsilon}'", 'bp-mblk', 'bp-ops']:
    check(f'H1/H2: “{_new[:40]}” is in the lab', _new in _G_SRC)
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
