"""Flow test for the train-debug lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run and three more widths:
- no page errors and no horizontal overflow at any step; the phone template (m2): the parts' bar, the stage, the card;
- the prediction gate (R3.8): Next disabled until a pick (Skip allowed); before it the stage, the cards, Settings, the dock and the
  math show no check result and no learning rate, and nothing names the shuffled labels; “Saved …” after a pick (R2.5);
- part 3 waits for step 300; part 4 for both other learning rates (inf, then NaN at learning rate 10; 0.523 at 0.003), the result on
  the option's title row (R3.6); part 5 for all three bugs, then the verdict with the right answer (R3.10); part 6's table with the
  picked check outlined; part 7: three misconceptions, “See the scenarios”, the next-lab footer;
- ▲▼ badges only after the learner's own change (R6.6);
- Scenarios: six, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page restores it,
  a change under Settings does not move the page (R3.2);
- the challenge: Settings locked (R4.2), checks open one at a time and count; the curve is free to read, so a right answer from the
  curve alone is tier 1 and any opened check is tier 2; the wrong-answer hint names the curve and the check that tells the two
  bugs apart (R4.4, review A1/A12); the five candidates' table after a right answer, the capability card; the language switch
  keeps the answer; the next mystery run brings the task into view (R3.7);
- the chart's blow-up notes sit in the band above the plot (R7.8); ▲▼ name their baseline (the healthy run, percentage points);
- Show me (filter `showme`): each click picks the next untried option until the part's gate is met (R1.1);
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万, no “toy”, no 起点 /
  starting point, no 控件 / Controls; NaN and inf are this lab's subject, so only undefined and Infinity count as broken values),
  colours only for their data (R5.6), no text wider than its box (R7.8), a tooltip for every help cursor (R1.1).
Screenshots: shots/train-debug/.
Run: python3 tests/train-debug_flow.py [en|zh] [1440|390] [showme]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-debug-flow.html', '--only', 'train-debug'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'train-debug-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'train-debug-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'train-debug-flow-wrapped.html')
SH = P / 'shots' / 'train-debug'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='train-debug'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__td.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
has = lambda pg, sel: pg.evaluate("s=>!!document.querySelector(s)", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#td-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2500):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#td-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#td-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception, R6.11)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/undefined|\\[object|Infinity/,'broken value'],
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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #td-mathbody, #td-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for a failed check or a blow-up,
# green only for a passed check or a right answer (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead')), green=toRGB(get('--c-ok'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|td-bad|td-nan|td-mk|td-nant|td-nanbg/,[violet]:/^$/,[green]:/okc|okline|td-ok|status ok|goal met/};
 const out=[]; for(const e of document.querySelectorAll('#td-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet,green]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':'');
   if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')) out.push(p+' '+e.tagName+'.'+(e.getAttribute('class')||'')+' '+(e.textContent||'').trim().slice(0,30));}}}}
 return out.slice(0,12);}"""
# R7.8: no text wider than its box (clipped or ellipsised), in the lab and the mini result bar
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#td-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# R1.1: every element with a help cursor explains itself on hover (itself or an ancestor carries the tooltip)
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#td-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits, fit_hits = [], []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#td-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#td-app', '#dock'):
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


# review 2026-10-03: no legend above the chart; each line named beside its end, inside the plot, clear of the other chart text
LABELS = """()=>{const t=[...document.querySelectorAll('#td-curve text.td-cl, #td-curve text.td-lnl, #td-curve text.td-mk')].map(e=>({e,b:e.getBBox()}));
 const out=[]; for(let i=0;i<t.length;i++){const a=t[i].b; if(t[i].e.classList.contains('td-cl')&&(a.x<35||a.x+a.width>392||a.y<30||a.y+a.height>204)) out.push('outside the plot: '+t[i].e.textContent);
  for(let j=i+1;j<t.length;j++){const c=t[j].b; if(a.x<c.x+c.width&&c.x<a.x+a.width&&a.y<c.y+c.height&&c.y<a.y+a.height) out.push(t[i].e.textContent+' / '+t[j].e.textContent);}}
 return out;}"""


def names(pg): return pg.evaluate("()=>[...document.querySelectorAll('#td-curve text.td-cl')].map(t=>t.textContent)")


def labels_ok(pg, where, want):
    got = names(pg)
    check(f'{where}: no legend above the chart; the lines are named beside them ({", ".join(want)})', not has(pg, '#td-legend') and sorted(got) == sorted(want) and pg.evaluate(LABELS) == [], (got, pg.evaluate(LABELS)))


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#td-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#td-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")


def outside_card(pg):
    """everything a learner can read before the prediction outside the card: the stage (with tooltips and the math), Settings, the dock"""
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#td-stage','#td-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      const m=document.getElementById('td-mathbody'); if(m) t.push(m.textContent); return t.join('\\n')}""")


def lever_rows(pg, box):
    return pg.evaluate(f"()=>[...document.querySelectorAll('#{box} .lever')].map(b=>{{const r=b.querySelector('.lv-r'), t=b.querySelector('.lv-t b'); return {{v:b.dataset.lever, res:r?r.textContent:'', top:r&&t?Math.abs(r.getBoundingClientRect().top-t.getBoundingClientRect().top)<8:false, under:r&&t?(r.getBoundingClientRect().top>=t.getBoundingClientRect().bottom-2&&r.getBoundingClientRect().top-t.getBoundingClientRect().bottom<10):false}}}})")


def run_guided(b, w, h, lang, dark=False):
    tag = f'{w}_{lang}' + ('_dark' if dark else '')
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    # part 1: the mission
    check(f'{tag} part 1: the curve and the regions are drawn', vis(pg, '#td-curve svg') and vis(pg, '#td-reg svg'))
    labels_ok(pg, f'{tag} part 1', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss')])
    check(f'{tag} part 1: the key sits right under the region map, one shape per class', vis(pg, '#td-reg .td-regkey') and pg.evaluate("()=>document.querySelectorAll('#td-reg .td-rk').length") == 3
          and pg.evaluate("()=>{const s=document.querySelector('#td-reg svg.td-regsvg').getBoundingClientRect(), k=document.querySelector('#td-reg .td-regkey').getBoundingClientRect(); return k.top>=s.bottom-1&&k.top-s.bottom<14}"), text(pg, '#td-reg .td-regkey'))
    check(f'{tag} part 1: only the training loss and the accuracy, no check yet (R3.8)', pg.evaluate("()=>document.querySelectorAll('#td-metrics .metric').length") == 2)
    if phone:
        check(f'{tag} part 1: the phone template (parts bar, then the stage, then the card)', pg.evaluate("""()=>{const a=document.querySelector('#td-rail .pbar'), s=document.getElementById('td-stage'), c=document.querySelector('#td-rail .step.current');
          return !!a&&!!s&&!!c&&a.getBoundingClientRect().top<s.getBoundingClientRect().top&&s.getBoundingClientRect().top<c.getBoundingClientRect().top}"""))
    o = outside_card(pg)
    check(f'{tag} part 1: nothing outside the card names the bug, a check result or the learning rate (R3.8)', not re.search(r'shuffl|打乱|✓|✗|η = 0\.3|learning rate 0\.3|学习率 0\.3', o), re.findall(r'.{30}(?:shuffl|打乱|✓|✗|η = 0\.3).{30}', o)[:3])
    check(f'{tag} part 1: the loss sits at ln 3', Z(lang, '1.102', '1.102') in text(pg, '#td-rail') and 'ln 3 = 1.099' in text(pg, '#td-rail'))
    shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    # part 2: predict
    click(pg, '#td-rail [data-nav="next"]')
    check(f'{tag} part 2: Next waits for a prediction, Skip offered', all(nexts(pg)) and pg.evaluate("()=>!!document.querySelector('#td-rail [data-nav=\"skip\"]')"), nexts(pg))
    o = outside_card(pg)
    check(f'{tag} part 2: nothing outside the card gives the answer away (R3.8)', not re.search(r'shuffl|打乱|✓|✗|η = 0\.3', o) and pg.evaluate("()=>document.querySelectorAll('#td-metrics .metric').length") == 2)
    click(pg, '#td-rail [data-pred="high"]')
    check(f'{tag} part 2: “Saved …” after a pick, Next enabled (R2.5)', Z(lang, 'Saved. You can still change it', '已记下') in text(pg, '#td-rail') and not any(nexts(pg)) and state(pg)['pred'] == 'high')
    check(f'{tag} part 2: single choice, the pick pressed', pg.evaluate("()=>[...document.querySelectorAll('#td-rail [data-pred]')].map(b=>b.getAttribute('aria-pressed'))") == ['false', 'true', 'false'])
    shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
    # part 3: a healthy run
    click(pg, '#td-rail [data-nav="next"]')
    check(f'{tag} part 3: Next waits for step 300', all(nexts(pg)))
    check(f'{tag} part 3: no ▲▼ on entering (R6.6)', badges(pg) == [], badges(pg))
    for s in ('30', '100'):
        click(pg, f'#td-rail .seg-ctl[data-key="steps"] button[data-v="{s}"]')
        check(f'{tag} part 3: step {s}, Next still waits, the result follows the step', all(nexts(pg)) and (f'{s} 步' if lang == 'zh' else f'{s} steps') in obs(pg), obs(pg)[:160])
    check(f'{tag} part 3: a ▲▼ badge after the learner moves through the run (R6.6)', any('▼' in x for x in badges(pg)), badges(pg))
    click(pg, '#td-rail .seg-ctl[data-key="steps"] button[data-v="300"]')
    ob = obs(pg)
    check(f'{tag} part 3: step 300 gives the result (0.135, 89.3%) and Next', not any(nexts(pg)) and '0.135' in ob and '89.3%' in ob, ob[:200])
    check(f'{tag} part 3: the three checks appear at step 300', pg.evaluate("()=>document.querySelectorAll('#td-metrics .metric').length") == 5)
    labels_ok(pg, f'{tag} part 3, step 300', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss')])
    check(f'{tag} part 3: check 2 says its number is a loss (review: “× 0.654” read as a ratio)', pg.evaluate("()=>document.querySelector('#td-metrics [data-k=\"c2\"] .v').textContent").startswith('✓ loss '), pg.evaluate("()=>document.querySelector('#td-metrics [data-k=\"c2\"]').innerText"))
    shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3')
    # part 4: the learning rate
    click(pg, '#td-rail [data-nav="next"]')
    check(f'{tag} part 4: Next waits; three rates, the initial setup pressed', all(nexts(pg)) and [r['v'] for r in lever_rows(pg, 'td-ic-lr')] == ['0.003', '0.3', '10'] and state(pg)['lrPick'] == '0.3')
    click(pg, '#td-ic-lr [data-lever="10"]')
    ob = obs(pg); rows = lever_rows(pg, 'td-ic-lr')
    more = pg.evaluate("()=>{const d=document.querySelector('#td-rail [data-obs] details.td-more'); return d?[d.open,d.textContent]:[null,'']}")
    check(f'{tag} part 4: learning rate 10 turns NaN at step 7; inf at step 3 and 61.955 at step 5 wait in a closed detail (R6.28)', Z(lang, 'turns NaN at step 7', '第 7 步变成 NaN') in ob and more[0] is False
          and all(x in more[1] for x in (Z(lang, 'inf at step 3', '第 3 步'), '61.955', Z(lang, 'NaN at step 7', '第 7 步'))) and more[1].find('61.955') < more[1].find(Z(lang, 'NaN at step 7', '第 7 步，')), (ob[:200], more))
    labels_ok(pg, f'{tag} part 4, rate 10', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss'), Z(lang, 'healthy run', '健康的运行')])
    check(f'{tag} part 4: the chart notes sit in the band above the plot, inside the chart (R7.8)', pg.evaluate("()=>[...document.querySelectorAll('#td-curve .td-mk')].every(t=>{const b=t.getBBox(); return b.y+b.height<=33&&b.x>=33&&b.x+b.width<=393})"), pg.evaluate("()=>[...document.querySelectorAll('#td-curve .td-mk')].map(t=>{const b=t.getBBox(); return [t.textContent,Math.round(b.x),Math.round(b.y),Math.round(b.width),Math.round(b.height)]})"))
    # kit round K (CONTRACT §14): the result stays on the title row while the title fits beside it, else it sits right under the title
    check(f'{tag} part 4: the result on the option’s title row, or right under the title (R3.6)', rows[2]['res'] == Z(lang, 'NaN at step 7', '第 7 步变成 NaN') and (rows[2]['top'] or rows[2]['under']), rows)
    check(f'{tag} part 4: the stage marks the blow-up and the healthy run as a thin line', has(pg, '#td-curve .td-nan') and has(pg, '#td-curve .td-cmp') and text(pg, '#td-status') == Z(lang, 'Blows up', '爆掉了'))
    check(f'{tag} part 4: one rate tried, Next still waits', all(nexts(pg)))
    shot(pg, f'{tag}_p4_lr10', full=phone); rules(pg, f'{tag} part 4, rate 10', lang); fit(pg, f'{tag} part 4, rate 10')
    click(pg, '#td-ic-lr [data-lever="0.003"]')
    ob = obs(pg)
    check(f'{tag} part 4: 0.003 crawls to 0.523; Next enabled', '0.523' in ob and not any(nexts(pg)), ob[:200])
    labels_ok(pg, f'{tag} part 4, rate 0.003', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss'), Z(lang, 'healthy run', '健康的运行')])
    check(f'{tag} part 4: the ▲ badge compares with the healthy run, in percentage points for the accuracy (R6.6, R6.13)', any('▲' in x and Z(lang, 'vs the healthy run', '比健康的运行') in x for x in badges(pg)) and not any('last' in x or '上次' in x or 'pts' in x for x in badges(pg)), badges(pg))
    click(pg, '#td-ic-lr [data-lever="0.3"]')
    rules(pg, f'{tag} part 4', lang)
    # part 5: three bugs
    click(pg, '#td-rail [data-nav="next"]')
    check(f'{tag} part 5: Next waits; no verdict yet', all(nexts(pg)) and Z(lang, 'right answer', '正确答案') not in text(pg, '#td-rail'))
    click(pg, '#td-ic-bug [data-lever="nozero"]')
    check(f'{tag} part 5: zero_grad() missing blows up at step 257', Z(lang, 'NaN at step 257', '第 257 步变成 NaN') in obs(pg), obs(pg)[:200])
    labels_ok(pg, f'{tag} part 5, zero_grad() missing', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss'), Z(lang, 'healthy run', '健康的运行')])
    click(pg, '#td-ic-bug [data-lever="twice"]')
    check(f'{tag} part 5: softmax twice stalls at 0.606 above its floor 0.551', '0.606' in obs(pg) and '0.551' in obs(pg), obs(pg)[:240])
    labels_ok(pg, f'{tag} part 5, softmax twice', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss'), Z(lang, 'healthy run', '健康的运行')])
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    st = state(pg)
    check(f'{tag} part 5: a language switch keeps the part, the prediction and the bugs tried', st['step'] == 4 and st['pred'] == 'high' and set(st['bugTried']) >= {'nozero', 'twice'} and st['bugPick'] == 'twice', st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    click(pg, '#td-ic-bug [data-lever="shuffle"]')
    ob = obs(pg)
    check(f'{tag} part 5: the shuffled labels stay at ln 3, then the verdict with the right answer and why (R3.10)', not any(nexts(pg)) and Z(lang, 'The right answer is “The labels are shuffled apart from the inputs”', '正确答案是“标签和输入被分开打乱了”') in ob and Z(lang, 'Why:', '原因：') in ob, ob[:300])
    check(f'{tag} part 5: the result comes before the verdict (R3.10)', ob.find('1.102') < ob.find(Z(lang, 'right answer', '正确答案')), ob[:200])
    shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5')
    # part 6: the checks
    click(pg, '#td-rail [data-nav="next"]')
    check(f'{tag} part 6: the table of six runs replaces the chart; Next waits', vis(pg, '#td-tbl table') and not vis(pg, '#td-curve svg') and all(nexts(pg)) and pg.evaluate("()=>document.querySelectorAll('#td-tbl tbody tr').length") == 6)
    click(pg, '#td-ic-chk [data-lever="c1"]')
    check(f'{tag} part 6: check 1 flags only the shuffled labels, its column outlined', Z(lang, 'flags only the shuffled labels', '只标出了打乱的标签') in obs(pg) and pg.evaluate("()=>document.querySelectorAll('#td-tbl .td-hl').length") == 7, obs(pg)[:160])
    gaps = pg.evaluate("()=>[...document.querySelectorAll('#td-tbl .td-hl')].filter(c=>c.getClientRects().length).map(c=>{const r=document.createRange(); r.selectNodeContents(c); const t=r.getBoundingClientRect(), b=c.getBoundingClientRect(); return [Math.round(t.left-b.left), Math.round(b.right-t.right), Math.round(t.top-b.top)]})")
    check(f'{tag} part 6: the picked column’s frame keeps clear of its text (review 2026-10-03, R5.10)', len(gaps) >= 6 and all(g[0] >= 6 and g[1] >= 2 and g[2] >= 3 for g in gaps), gaps)
    check(f'{tag} part 6: check 2 is named in plain words and says its number is a loss', Z(lang, 'Train on just 9 examples', '只用 9 个样本训练') in text(pg, '#td-rail') and Z(lang, 'learned: loss 0.016', '学会了：loss 0.016') in text(pg, '#td-tbl'), text(pg, '#td-tbl')[:300])
    click(pg, '#td-ic-chk [data-lever="c3"]')
    check(f'{tag} part 6: two checks, Next enabled', not any(nexts(pg)))
    shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6')
    check(f'{tag} part 6: no overflow', overflow(pg) <= 0, overflow(pg))
    # part 7: what to remember
    click(pg, '#td-rail [data-nav="next"]')
    rt = text(pg, '#td-rail')
    check(f'{tag} part 7: three points, three misconceptions, See the scenarios, the next-lab footer', pg.evaluate("()=>document.querySelectorAll('#td-rail .td-remlist li').length") == 3 and rt.count('✗') == 3
          and Z(lang, 'See the scenarios', '去看应用场景') in rt and pg.evaluate("()=>!!document.querySelector('#td-rail .next-lab')"), rt[-300:])
    check(f'{tag} part 7: the misconceptions state the pattern (G2, 2026-10-08: no NaN step of an earlier part) and keep the overfitting scenario’s numbers (0.304 → 0.476)', all(x in rt for x in ('0.304', '0.476')) and Z(lang, 'both end in NaN', '却都会让 loss 变成 NaN') in rt and '257' not in rt, rt[-400:])
    _rl = text(pg, '#td-rail .step.current .td-remlist') + ' ' + text(pg, '#td-rail .step.current .obs')
    check(f'{tag} part 7 (G2): no point or misconception leans on a detail of an earlier part', not re.search(r'[Pp]art \d|第 \d 部分', _rl), _rl[:300])
    shot(pg, f'{tag}_p7', full=phone); rules(pg, f'{tag} part 7', lang); fit(pg, f'{tag} part 7')
    if phone:
        go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('td-stage'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, a short status and the loss', dk['shown'] and len(dk['status']) <= 14 and '1.102' in dk['read'], dk)
        shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang)
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>document.querySelectorAll('#td-rail [data-preset]').length")
    rc = text(pg, '#td-rail .railcard')
    check(f'{tag} scenarios: six, single choice, the lead-in (G1, 2026-10-08), the card’s fixed lines', presets == 6 and Z(lang, 'Pick a scenario and see how its loss curve moves.', '选一个场景，看看它的 loss 曲线怎么走。') in rc and Z(lang, 'Change anything else under Settings.', '想改别的，在“自由调参”里改。') in rc and Z(lang, 'Take the challenge', '去做挑战') in rc, rc[-200:])
    for i in range(presets):
        click(pg, f'#td-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#td-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#td-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80, (pressed, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang); fit(pg, f'{tag} scenario {i}')
        if i in (1, 3, 5): shot(pg, f'{tag}_scenario_{i}', full=phone)
    check(f'{tag} scenario 6 (15 examples): overfitting, not a bug', Z(lang, 'overfitting', '过拟合') in text(pg, '#td-insight') and '0.476' in text(pg, '#td-insight'), text(pg, '#td-insight')[:200])
    click(pg, '#td-rail [data-preset="1"]')
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#train-debug~2_0_0', hsh)
    pg2 = page(b, w, h, lang, hash_='train-debug~2_0_0')
    st2 = state(pg2)
    check(f'{tag} scenarios: a fresh page restores the setup', st2['mode'] == 'sandbox' and st2['lr'] == 10 and st2['bug'] == 'none' and pg2.evaluate("()=>document.querySelector('#td-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true', st2)
    pg2.close()
    # a change under Settings keeps the page still (R3.2) and unpresses the scenario when it matches none
    if phone: pg.evaluate("()=>{const c=document.querySelector('#td-controls'); if(c&&!c.classList.contains('open')) c.querySelector('.eyebrow').click()}"); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#td-ctl-bug').scrollIntoView({block:'center'})"); pg.wait_for_timeout(400)
    y0 = pg.evaluate("()=>document.querySelector('#td-ctl-bug').getBoundingClientRect().top")
    pg.evaluate("()=>document.querySelector('#td-ctl-bug button[data-v=\"twice\"]').click()"); pg.wait_for_timeout(400)
    y1 = pg.evaluate("()=>document.querySelector('#td-ctl-bug').getBoundingClientRect().top")
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#td-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} Settings: a change does not move the page (R3.2)', abs(y1 - y0) < 2, (y0, y1))
    check(f'{tag} Settings: learning rate 10 with softmax twice matches no scenario (R1.8)', pressed.count('true') == 0 and Z(lang, 'match none of these scenarios', '都不一样') in text(pg, '#td-insight'), pressed)
    rules(pg, f'{tag} settings', lang); fit(pg, f'{tag} settings')
    # the challenge
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} challenge: Settings locked with a note (R4.2), the first mystery hides softmax twice', st['myst'] == 'twice' and st['chMin'] == [] and not vis(pg, '#td-ctl-bug') and vis(pg, '#td-lock') if not phone else st['myst'] == 'twice', st)
    check(f'{tag} challenge: the closed checks wait to be run', pg.evaluate("()=>document.querySelectorAll('#td-metrics .td-closed').length") == 3 and not has(pg, '#td-curve .td-ln'))
    rc = text(pg, '#td-rail .railcard')
    check(f'{tag} challenge: the task in plain words (review 2026-10-03, R6.32): no “mystery run”, no “reading the chart is free”', not re.search(r'神秘|看图不算数|[Mm]ystery|free to read', rc + text(pg, '#td-stage'))
          and Z(lang, 'Each check you open counts as one used', '每打开一个检查，就算用了一次') in rc and Z(lang, 'script 1 of 5', '第 1 个脚本，共 5 个') in rc, rc[:400])
    labels_ok(pg, f'{tag} challenge', [Z(lang, 'training loss', '训练 loss'), Z(lang, 'validation loss', '验证集 loss')])
    click(pg, '#td-rail [data-ans="low"]'); click(pg, '#td-rail [data-act="submit"]')
    res = text(pg, '#td-ch-res')
    check(f'{tag} challenge tier 3: wrong, with a hint from the curve and the check that tells the answer from the hidden bug', res.startswith(Z(lang, 'Not “Learning rate too low”', '不是“学习率太低”')) and Z(lang, 'Hint: with “Learning rate too low” the loss falls slowly', '提示：如果是“学习率太低”，loss 会慢慢下降') in res and Z(lang, 'Check 3 would also tell your answer apart from the real bug', '检查 3 也能把你选的 bug 和脚本里真正的 bug 分开') in res, res[:300])
    shot(pg, f'{tag}_challenge_wrong', full=phone); rules(pg, f'{tag} challenge wrong', lang)
    # the owner's review of batch 8 (2026-10-07): “Show the answer” names the bug in this run and puts its curve next to the pick's,
    # instead of listing all five problems
    check(f'{tag} challenge: the reveal button is “Show the answer”', text(pg, '#td-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), text(pg, '#td-ch-res [data-act="reveal"]'))
    click(pg, '#td-ch-res [data-act="reveal"]')
    li = pg.evaluate("()=>[...document.querySelectorAll('#td-ch-res .td-cands li')].map(l=>({me:l.classList.contains('td-me'),t:l.textContent}))")
    res = text(pg, '#td-ch-res')
    check(f'{tag} challenge “Show the answer”: the answer named, two lines (the bug in this run, then the pick), and why',
          len(li) == 2 and li[0]['me'] and Z(lang, '← this run', '← 就是这次') in li[0]['t'] and Z(lang, '← your answer', '← 你选的') in li[1]['t'] and Z(lang, 'Learning rate too low', '学习率太低') in li[1]['t']
          and Z(lang, 'The answer is “Softmax applied twice”', '答案是“softmax 用了两次”') in res and Z(lang, 'Why:', '原因：') in res and not has(pg, '#td-ch-res [data-act="reveal"]'), (li, res[-400:]))
    click(pg, '#td-rail [data-ans="twice"]')
    res = text(pg, '#td-ch-res')
    check(f'{tag} challenge tier 1: right from the curve alone, no check needed, check 3 would confirm (A1)', res.startswith(Z(lang, 'Right', '对')) and Z(lang, 'The curve already tells it apart: the loss drops fast, then stalls far above 0', '光看曲线就能把它分出来：loss 先快速下降，再停在离 0 很远的地方') in res and Z(lang, 'Check 3 would confirm it', '检查 3 可以确认它') in res and 'same curve' not in res, res[:300])
    click(pg, '#td-metrics [data-ck="c3"]')
    res = text(pg, '#td-ch-res')
    check(f'{tag} challenge tier 2: right with check 3, which the curve did not need', res.startswith(Z(lang, 'Right', '对')) and Z(lang, 'But fewer checks would have done: the curve alone tells it apart', '不过用不了这么多检查：光看曲线就能把它分出来') in res and state(pg)['chOpen'] == ['c3'], res[:300])
    check(f'{tag} challenge: after a right answer, no list of the five problems and no table (the owner, 2026-10-07); the reason follows the verdict', not has(pg, '#td-ch-res .td-cands') and not has(pg, '#td-ch-res table') and Z(lang, 'Why:', '原因：') in text(pg, '#td-ch-res'))
    shot(pg, f'{tag}_challenge_t1', full=phone); rules(pg, f'{tag} challenge tier 1', lang); fit(pg, f'{tag} challenge tier 1')
    click(pg, '#td-rail [data-ck="c1"]')
    res = text(pg, '#td-ch-res')
    check(f'{tag} challenge tier 2: right after more checks than needed', Z(lang, 'But fewer checks would have done', '不过用不了这么多检查') in res and has(pg, '#td-curve .td-ln'), res[:220])
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#td-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#td-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#td-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer, the checks and the result', st['chAns'] == 'twice' and st['chOpen'] == ['c3', 'c1'] and st['chSubmitted'] and vis(pg, '#td-ch-res .okline'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    pg.evaluate("()=>document.querySelector('#td-connH').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
    click(pg, '#td-rail [data-act="next-myst"]'); pg.wait_for_timeout(500); settle(pg)
    st = state(pg)
    check(f'{tag} challenge: the next mystery starts clean', st['myst'] == 'nozero' and st['chOpen'] == [] and not st['chSubmitted'], st)
    check(f'{tag} challenge: the next mystery brings the task (the stage on a phone) into view (R3.7)', pg.evaluate("()=>{const e=document.querySelector(innerWidth<=760?'#td-stage':'#td-rail .railcard'); const r=e.getBoundingClientRect(); return r.top>=0&&r.top<innerHeight*0.5}"), pg.evaluate("()=>[scrollY, document.querySelector('#td-rail .railcard').getBoundingClientRect().top]"))
    click(pg, '#td-rail [data-ans="nozero"]'); click(pg, '#td-rail [data-ck="c3"]'); click(pg, '#td-rail [data-act="submit"]')
    check(f'{tag} challenge, mystery 2: zero_grad() missing, read from the curve (it falls first, then blows up)', Z(lang, 'But fewer checks would have done', '不过用不了这么多检查') in text(pg, '#td-ch-res') and Z(lang, 'falls first, then blows up', '先降一阵，再爆掉') in text(pg, '#td-ch-res'), text(pg, '#td-ch-res')[:300])
    shot(pg, f'{tag}_challenge_m2', full=phone); rules(pg, f'{tag} challenge mystery 2', lang)
    pg.close()


def run_showme(b):
    # A3: each Show me picks the next untried option until the part's gate is met
    pg = page(b, 1440, 900, 'en')
    for part, clicks in ((3, 2), (4, 3), (5, 2)):
        go_part(pg, part)
        for _ in range(clicks):
            if not any(nexts(pg)): break
            pg.evaluate("()=>document.querySelector('#td-rail [data-nav=\"showme\"]').click()"); pg.wait_for_timeout(250)
        check(f'Show me in part {part + 1}: {clicks} clicks meet the gate (R1.1)', not any(nexts(pg)), (state(pg)['lrTried'], state(pg)['bugTried'], state(pg)['chkTried']))
    pg.close()



# H1, H2 and H10, the owner's site-wide decisions of 2026-10-09: Show the math as short blocks with typeset formulas.
def h_page(b, lang):
    pg = page(b, 1440, 900, lang); mode(pg, 'sandbox')
    pg.evaluate("()=>{const d=document.getElementById('td-math'); if(d&&!d.open) d.querySelector('summary').click()}"); pg.wait_for_timeout(300)
    hv = pg.evaluate("()=>{const mb=document.getElementById('td-mathbody'); return {blk:mb.querySelectorAll('.td-mblk').length, math:mb.querySelectorAll('math').length, tex:/\\\\(frac|text|ln|sum|exp)/.test(mb.textContent), plain:/exp\\(z\\) ÷|−ln\\(1\\//.test(mb.textContent)}}")
    check(f'H2 {lang}: Show the math is short blocks with typeset formulas, no TeX source and no plain-text formula', hv['blk'] >= 3 and hv['math'] >= 4 and not hv['tex'] and not hv['plain'], hv)
    pg.close()

with sync_playwright() as p:
    b = p.chromium.launch()
    if FULL or 'showme' in ARGS: run_showme(b)
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
_G_SRC = (P / 'labs' / 'train-debug.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time', '一次只有一个', '只选一个', '(one at a time)', '（一次选一个）']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a learning rate and see how its loss curve differs from the thin line.', '选一个学习率，看看 loss 曲线和细线差在哪。', 'Pick a way it can be wrong and see how its loss curve differs from the thin line.', '选一种写错的方式，看看 loss 曲线和细线差在哪。', 'Pick a bug and see how the loss curve changes.', '选一个 bug，看看 loss 曲线怎么变。', 'Pick a scenario and see how its loss curve moves.', '选一个场景，看看它的 loss 曲线怎么走。', 'Which problem do you think it is? Pick one as your answer.', '你认为是哪一种问题？选一个作为答案。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('the five you just broke', 'two with a bad learning rate and three with a wrong line in the script.'), ('加上刚才弄坏的五次', '一次健康的，两次学习率不对的，三次脚本里有一行写错的。'), ('图里的细线是刚才那次健康的运行', '图里的细线是学习率 0.3 时健康的运行，用来对照。'), ('The healthy run of part 3.', 'The healthy run, drawn as the thin line in the chart.'), ('就是第 3 部分那次健康的运行。', '健康的运行，就是图里的细线。'), ('In part 1 the loss stayed near ln 3', 'A loss that stays at ln C means the model is no better than guessing 1/C for each class'), ('第 1 部分里，loss ${STEPS} 步都停在', 'loss 一直停在 ln C，说明模型没比“每类都猜 1/C”好'), ('In part 6, when the problem was in the model', 'When it learns them, the problem is in how the data reaches it.'), ('第 6 部分里，问题出在模型', '学得会，问题就出在数据送进模型的路上。'), ('Learning rate 0.003 (part 4) and softmax twice (part 5)', 'A loss that ends high can mean opposite things.'), ('学习率 0.003（第 4 部分）和 softmax 用两次（第 5 部分）', 'loss 最后还很高，可能说明两种正好相反的情况。'), ('In part 4, at learning rate 0.003 the loss kept falling', 'Too low a learning rate makes the loss fall slowly, but it keeps falling.'), ('第 4 部分里，学习率为 0.003 时 loss', '学习率太低时，loss 降得慢，但一直在降。'), ('Learning rate 10 (part 4) turned NaN', 'yet a learning rate that is too high and a missing zero_grad() both end in NaN'), ('学习率 10（第 4 部分）在第', '学习率太高和漏了 zero_grad() 却都会让 loss 变成 NaN'), ('In part 6 check 2 failed', 'Check 2 fails for four of the five problems'), ('第 6 部分里，五种问题中有四种', '五种问题中有四种都让检查 2 不通过')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# H1 and H10, the owner's decisions of 2026-10-09: no plain-text formula left in the lab's source (EN and ZH), and plain words.
for _bad in ['softmax is written as exp(z) ÷ Σ exp(z)', '脚本里的 softmax 写成 exp(z) ÷ Σ exp(z)', 'costs −ln(1/C) = ln C on every example', '代价都是 −ln(1/C) = ln C', 'gives a loss of −ln(1/10) = ln 10', 'loss 就是 −ln(1/10) = ln 10', 'feeds the loss chances', '喂给 loss']:
    check(f'H1/H10: “{_bad}” is gone', _bad not in _G_SRC)
for _new in ['With softmax twice, the loss gets chances instead of logits.', 'softmax 用两次：loss 拿到的是概率，而不是 logit。', 'td-mblk']:
    check(f'H1/H10: “{_new[:40]}” is in the lab', _new in _G_SRC)
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
