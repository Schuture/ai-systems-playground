"""Flow test for the sampling lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH, light, plus one dark run and three more widths:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the chart's bars;
- the prediction gate (R3.8): Next disabled until a prediction; before it the temperature is fixed at 1 and the answer (temperature 0:
  every run writes the same word) appears nowhere, in text or tooltips; “Saved …” after a pick (R2.5); part 3 waits for temperature 0,
  then gives the result, the verdict with the right answer, and why (R3.10);
- part 4: Settings locked with a note, the three rules single choice, the result on the option's title row after a pick, 4 and 1 kept
  with top-p 0.9, 3 and 3 with top-k 3; part 5: Next waits until the first random number lands on another word, the whole sentence
  changes, the tree outlines the run; part 6: three misconceptions, “See the scenarios”, the next-lab footer (R3.11, R3.12);
- ▲▼ badges only after the learner's own change (R6.6);
- Scenarios: four, single choice, pressed state follows the settings (R1.8), the URL carries the setup and a fresh page restores it,
  a change under Settings does not move the page (R3.2); the sentence tree view;
- the challenge: only the two settings in the card, Settings locked (R4.2), live progress, the three feedback tiers (R4.4), “Show the
  answer”, the brief chart (the 5 passing pairs, the initial setup and the pick; the owner, 2026-10-07) with the goal (R5.4), the capability card; the language switch
  keeps the answer;
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (R6.14, R6.7, Chinese spacing and punctuation, no 亿/万, no “toy”, no 起点 /
  starting point, no 控件 / Controls), and colours only for their data (R5.6);
- round 1 (reviews A and B): the challenge chart marks every passing row visibly and its goal label sits inside the plot (A1); Settings
  fixed in the guided parts except part 3's temperature (A2); no “greedy” before the prediction, in text, popovers and term tooltips (A3);
  the running totals under part 4's lists (A4); the ÷ T column (A5); good words only in the challenge's count (A7); the random numbers'
  labels have tooltips, no “table” (A10); no tag over a label in the tree (A11); the phone's mini result bar in part 4 (A15); why Next
  is greyed out (A16); no text wider than its box (R7.8) and a tooltip for every help cursor (R1.1) in every state; singular and plural
  (A21); the phone's slider next to the bars in part 3 (A14);
- round 1b (re-review N1, N2, N6): no random-number line, number label or tag over a block label in the tree at seven positions of the
  first random number; on a phone the slider stays put when Next becomes enabled (parts 3 and 5); part 4's jump to the rules.
Screenshots: shots/sampling/.
Run: python3 tests/sampling_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sampling-flow.html', '--only', 'sampling,set-b'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'sampling-flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'sampling-flow.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'sampling-flow-wrapped.html')
SH = P / 'shots' / 'sampling'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='sampling'):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__sp.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#sp-rail [data-nav=\"next\"]')].map(b=>b.disabled)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__sp||!window.__sp.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#sp-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#sp-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#sp-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#sp-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #sp-mathbody, .num-big, #sp-drill-a')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
# blue (weights), amber (KV cache) and violet (runtime overhead) mean nothing in this lab; red only for a missed goal (R5.6)
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|sp-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#sp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#sp-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#sp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#sp-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#sp-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")


# round 1: everything a learner can read before the prediction, outside Connect: the header, its popovers (even closed), the card,
# the stage, Settings, their tooltips, and every term tooltip in the card (A3, R3.8)
def before_answer(pg):
    out = [pg.evaluate("""()=>{const t=[]; for(const s of ['#sp-app .labhead','#sp-rail','#sp-stage','#sp-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#sp-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")]
    for i in range(pg.evaluate("()=>document.querySelectorAll('#sp-rail .t').length")):
        pg.evaluate(f"()=>{{const e=document.querySelectorAll('#sp-rail .t')[{i}]; e.focus({{preventScroll:true}})}}"); pg.wait_for_timeout(60)
        out.append(pg.evaluate("()=>{const t=document.getElementById('tip'); return t?t.innerText:''}"))
    pg.evaluate("()=>document.activeElement&&document.activeElement.blur()")
    return '\n'.join(out)


# R7.8: no text wider than its box (clipped or ellipsised), in the lab and the mini result bar
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#sp-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
# R1.1: every element with a help cursor explains itself on hover (itself or an ancestor carries the tooltip)
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#sp-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
fit_hits = []
# re-review N1: in each tree column, the random number's line (a 4 px strip), its label and the run's tag against every visible block label
TREE_OV = """()=>{const R=e=>e.getBoundingClientRect(), out=[], txt=e=>{const r=document.createRange(); r.selectNodeContents(e); return r.getBoundingClientRect();};
  const hit=(a,b)=>a.left<b.right-0.5&&b.left<a.right-0.5&&a.top<b.bottom-0.5&&b.top<a.bottom-0.5;
  document.querySelectorAll('#sp-tree .sp-colb').forEach((col,l)=>{const spans=[...col.querySelectorAll('.sp-blk>span')].filter(s=>!s.hidden&&s.getClientRects().length).map(s=>({t:s.textContent,r:txt(s)}));
    col.querySelectorAll('.sp-u').forEach(u=>{const lr=R(u), strip={left:lr.left,right:lr.right,top:lr.top-1,bottom:lr.top+3}, em=R(u.querySelector('em'));
      spans.forEach(s=>{if(hit(strip,s.r)) out.push('col '+(l+1)+': line through '+s.t); if(hit(em,s.r)) out.push('col '+(l+1)+': number over '+s.t);});});
    col.querySelectorAll('.sp-tag').forEach(t=>{const tr=R(t); spans.forEach(s=>{if(hit(tr,s.r)) out.push('col '+(l+1)+': tag over '+s.t);});});});
  return out;}"""


def still(pg, sel, v):
    """re-review N2: move a phone slider (input only, as during a drag) and report how far it moved on the page, and the Next buttons"""
    pg.evaluate(f"()=>document.querySelector('{sel}').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(300)
    y0 = pg.evaluate(f"()=>document.querySelector('{sel}').getBoundingClientRect().top+scrollY")
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(350)
    y1 = pg.evaluate(f"()=>document.querySelector('{sel}').getBoundingClientRect().top+scrollY")
    return {'moved': abs(y1 - y0), 'y0': y0, 'y1': y1, 'next': nexts(pg)}


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    if pg.evaluate("()=>/\\b1 words\\b|\\b1 different first words\\b|\\b1 good words\\b/.test(document.getElementById('sp-app').innerText+[...document.querySelectorAll('#sp-app [data-tipx]')].map(e=>e.dataset.tipx).join(' '))"):
        fit_hits.append(f'{where}: “1 words” (A21)')


def run_guided(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'
    pg = page(b, w, h, lang, dark)
    phone = w <= 760
    check(f'{tag} title', text(pg, '#sp-app .labtitle') == Z(lang, 'What do temperature and top-p actually do?', 'Temperature 和 top-p 到底做了什么？'), text(pg, '#sp-app .labtitle'))
    check(f'{tag} no fidelity tag on top (owner, 2026-10-11 19:28 UTC)', not any(w in text(pg, '#sp-app .labmeta') for w in ('Fidelity', '保真度')), text(pg, '#sp-app .labmeta'))
    check(f'{tag} outcome line under the title', text(pg, '#sp-app .lab-outcome').startswith(Z(lang, 'You will be able to:', '学完你能：')))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#sp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        pg.evaluate(f"()=>document.querySelector('#sp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(200)
        check(f'{tag} popover {pid} opens', pg.evaluate(f"()=>{{const p=document.getElementById('{pid}'); return !!p&&p.getClientRects().length>0}}"))
        rules(pg, f'{tag} popover {pid}', lang)
        pg.evaluate(f"()=>document.querySelector('#sp-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
    # part 1: the mission, at temperature 1 with no cut
    st = state(pg)
    check(f'{tag} part 1: temperature 1, no cut, seed 1', (st['T'], st['P'], st['K'], st['seed']) == (1, 1, 0, 1), st)
    check(f'{tag} part 1: the 20 runs show 5 different first words', len(pg.evaluate("()=>[...new Set([...document.querySelectorAll('#sp-chips .sp-chip')].map(c=>c.textContent))]")) == 5 and text(pg, '#sp-metrics [data-k="runs"] .v') == '5')
    check(f'{tag} part 1: no overflow', overflow(pg) <= 0, overflow(pg))
    if phone:
        fs = pg.evaluate("""()=>{const pb=document.querySelector('#sp-rail .pbar'), nx=pb&&pb.querySelector('[data-nav="next"]'), rows=document.getElementById('sp-rows');
          const r=e=>e?e.getBoundingClientRect():null, a=r(pb), b=r(nx), c=r(rows); return {bar:!!a&&a.top<innerHeight&&a.height>0, next:!!b&&b.bottom<=innerHeight&&b.height>0, rows:c?Math.round(c.top):null, vh:innerHeight}}""")
        check(f'{tag} phone first screen: the parts’ bar with Next, and the chart’s bars start on screen (R7.3)', fs['bar'] and fs['next'] and fs['rows'] is not None and fs['rows'] < fs['vh'] - 80, fs)
        shot(pg, f'{tag}_first_screen')
    shot(pg, f'{tag}_p1', full=phone)
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    check(f'{tag} part 1: Settings fixed with a note saying to what (A2)', vis(pg, '#sp-lock') and Z(lang, 'top-p off, top-k off, seed 1', '不用 top-p，不用 top-k，种子 1') in text(pg, '#sp-lock')
          and not any(vis(pg, x) for x in ('#sp-ctl-T-wrap', '#sp-ctl-P-wrap', '#sp-ctl-K-wrap', '#sp-ctl-seed')), text(pg, '#sp-lock'))
    greedy = Z(lang, 'greedy', '贪心')
    ba = before_answer(pg)
    check(f'{tag} part 1: the answer to part 2 appears nowhere, not in popovers or term tooltips (A3)', greedy not in ba.lower() and Z(lang, 'all 20 runs write', '20 次运行都写') not in ba, [x for x in ba.split('\n') if greedy in x.lower()][:2])
    ans = Z(lang, 'all 20 runs write', '20 次运行都写')
    # part 2: the gate; the temperature is fixed before the prediction
    click(pg, '#sp-rail [data-nav="next"]')
    check(f'{tag} part 2 reached, every Next disabled before a prediction', state(pg)['step'] == 1 and all(nexts(pg)), nexts(pg))
    check(f'{tag} part 2: the temperature is fixed before the prediction (note shown, control hidden)', vis(pg, '#sp-lock') and not vis(pg, '#sp-ctl-T-wrap') and not vis(pg, '#sp-ctl-P-wrap'))
    ba = before_answer(pg)
    check(f'{tag} part 2: the answer appears nowhere (text, popovers, term tooltips; A3)', ans not in ba and greedy not in ba.lower() and text(pg, '#sp-metrics [data-k="runs"] .v') == '5', [x for x in ba.split('\n') if greedy in x.lower()][:2])
    why = Z(lang, 'Pick an answer to continue, or Skip.', '选一个答案才能继续，也可以跳过。')
    check(f'{tag} part 2: why Next is greyed out, next to the Next that is shown (A16)', pg.evaluate("(w)=>[...document.querySelectorAll('#sp-rail .sp-why')].some(p=>!p.hidden&&p.getClientRects().length&&p.textContent===w)", why))
    if phone:
        check(f'{tag} part 2: on a phone, a way from the chart to the question (A14)', vis(pg, '#sp-phone [data-act="to-q"]'))
    fit(pg, f'{tag} part 2')
    click(pg, '#sp-rail [data-pred="many"]')
    check(f'{tag} part 2: “Saved … you can still change it”, pressed with a radio dot', Z(lang, 'Saved. You can still change it before part 3 shows the result.', '已记下。第 3 部分给出结果之前，还可以改。') in text(pg, '#sp-rail .step.current')
          and pg.evaluate("()=>{const b=document.querySelector('#sp-rail [data-pred=\"many\"]'); return b.getAttribute('aria-pressed')==='true'&&getComputedStyle(b,'::before').content!=='none'}"))
    check(f'{tag} part 2: Next enabled after a prediction, and the reason line gone', not any(nexts(pg)) and not pg.evaluate("()=>[...document.querySelectorAll('#sp-rail .sp-why')].some(p=>!p.hidden)"), nexts(pg))
    shot(pg, f'{tag}_p2')
    rules(pg, f'{tag} part 2', lang)
    # part 3: temperature; Next waits for 0
    click(pg, '#sp-rail [data-nav="next"]')
    check(f'{tag} part 3: no ▲▼ badge on entering a part; Next waits for temperature 0', not badges(pg) and all(nexts(pg)), (badges(pg), nexts(pg)))
    slide(pg, '#sp-ic-T-in', 5)
    o = obs(pg)
    check(f'{tag} part 3 at 0.5: the top word gains, about 65% instead of about 44%, and a ▲ badge', Z(lang, '“noodles” gets about 65% instead of about 44% at 1', '“面条”约占 65%，temperature 为 1 时约占 44%') in o
          and any('▲' in x for x in badges(pg)) and all(nexts(pg)), (o[:200], badges(pg)))
    zs = pg.evaluate("()=>[...document.querySelectorAll('#sp-rows .sp-zs')].map(e=>e.textContent)")
    check(f'{tag} part 3 at 0.5: the ÷ T column shows the logits doubled (A5)', zs == ['4.0', '3.0', '2.0', '0.6', '-1.0', '-2.0'] and vis(pg, '#sp-rows .sp-zh'), zs)
    check(f'{tag} part 3: Settings show only the temperature (A2)', vis(pg, '#sp-ctl-T-wrap') and not any(vis(pg, x) for x in ('#sp-ctl-P-wrap', '#sp-ctl-K-wrap', '#sp-ctl-seed')))
    if phone:
        pp = pg.evaluate("()=>{const s=document.getElementById('sp-ph-T-in'), r=document.getElementById('sp-rows'); if(!s||!r) return null; const a=s.getBoundingClientRect(), b=r.getBoundingClientRect(); return {s:a.height>0&&a.bottom<innerHeight, rows:b.top<innerHeight-60, below:b.top>a.bottom}}")
        check(f'{tag} part 3: on a phone the temperature slider sits in the chart, above the bars, on the first screen (A14)', bool(pp) and all(pp.values()), pp)
    fit(pg, f'{tag} part 3')
    if w == 1440 and not dark:
        pg.set_viewport_size({'width': 1440, 'height': 1000}); pg.wait_for_timeout(200)
        pg.evaluate("()=>window.scrollTo(0,0)"); slide(pg, '#sp-ic-T-in', 15)
        oc = pg.evaluate("""()=>{const b=document.querySelector('#sp-rail [data-obs]'); const r=b.getBoundingClientRect(), top=document.querySelector('.topbar').getBoundingClientRect().bottom; return {top:Math.round(r.top), ok:r.top>=top-2&&r.top<innerHeight-40}}""")
        check(f'{tag} part 3: after the drag the result is on screen at 1440×1000 (R3.10)', oc['ok'], oc)
        check(f'{tag} part 3 at 1.5: the bars flatten (about 11% for the two least likely words, about 5.8% at 1)', Z(lang, 'about 11% instead of about 5.8% at 1', '约占 11%，temperature 为 1 时约占 5.8%') in obs(pg), obs(pg)[:200])
        pg.set_viewport_size({'width': w, 'height': h}); pg.wait_for_timeout(200)
    if phone:
        jy = still(pg, '#sp-ph-T-in', 0)
        check(f'{tag} part 3: on a phone the slider stays put when Next becomes enabled at 0 (re-review N2, O3)', jy['moved'] < 1 and not any(jy['next']), jy)
    slide(pg, '#sp-ic-T-in', 0)
    o = obs(pg)
    i_res, i_ver, i_why = o.find(ans), o.find(Z(lang, 'the right answer is', '正确答案是')), o.find(Z(lang, 'Why:', '原因：'))
    chips = set(pg.evaluate("()=>[...document.querySelectorAll('#sp-chips .sp-chip')].map(c=>c.textContent)"))
    check(f'{tag} part 3 at 0: the result, then the verdict with the right answer, then why; 20 identical chips; Next enabled', -1 < i_res < i_ver < i_why and len(chips) == 1 and not any(nexts(pg)), (o[:300], chips))
    check(f'{tag} part 3 at 0: one word can be picked, the others get no chance (not “cut”)', text(pg, '#sp-metrics [data-k="kept"] .v') == Z(lang, '1 of 6', '1 / 6') and pg.evaluate("()=>document.querySelectorAll('#sp-rows .sp-row.cut').length") == 0)
    check(f'{tag} part 3 at 0: the ÷ T column is empty (A5)', set(pg.evaluate("()=>[...document.querySelectorAll('#sp-rows .sp-zs')].map(e=>e.textContent)")) == {'—'})
    fit(pg, f'{tag} part 3 at 0')
    shot(pg, f'{tag}_p3', full=phone)
    rules(pg, f'{tag} part 3', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(350)
    st = state(pg)
    check(f'{tag} part 3: a language switch keeps the part, the temperature and the prediction', st['step'] == 2 and st['T'] == 0 and st['pred'] == 'many' and bool(obs(pg)), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(350)
    # part 4: top-p against top-k, Settings locked
    click(pg, '#sp-rail [data-nav="next"]')
    check(f'{tag} part 4: Settings locked with a note (R4.2), three single-choice rules, the initial setup pressed, Next waits', vis(pg, '#sp-lock') and not vis(pg, '#sp-ctl-P')
          and pg.evaluate("()=>[...document.querySelectorAll('#sp-rail [data-lever]')].map(b=>b.dataset.lever+':'+b.getAttribute('aria-pressed')).join()") == 'none:true,k3:false,p9:false' and all(nexts(pg)))
    if phone:
        check(f'{tag} part 4: on a phone, a way from the chart to the rules (re-review N6)', vis(pg, '#sp-phone [data-act="to-rules"]'))
    click(pg, '#sp-rail [data-lever="k3"]')
    check(f'{tag} part 4, top-k 3: 3 and 3 kept, its result on its title row', text(pg, '#sp-metrics [data-k="kf"] .v') == Z(lang, '3 of 6', '3 / 6') and text(pg, '#sp-metrics [data-k="kp"] .v') == Z(lang, '3 of 4', '3 / 4')
          and text(pg, '#sp-rail [data-lever="k3"] .lv-r') == Z(lang, 'keeps 3 of 6, 3 of 4', '平的留 3 个，尖的留 3 个') and not any(nexts(pg)), text(pg, '#sp-rail [data-lever="k3"] .lv-r'))
    kb = pg.evaluate("()=>[...document.querySelectorAll('#sp-duo .sp-kbr em')].map(e=>e.textContent)")
    check(f'{tag} part 4, top-k 3: a bracket over 3 words on each list, 86% and 99% of the chance (A4)', kb == [Z(lang, '3 words: 86%', '3 个词：86%'), Z(lang, '3 words: 99%', '3 个词：99%')], kb)
    fit(pg, f'{tag} part 4, top-k 3')
    click(pg, '#sp-rail [data-lever="p9"]')
    o = obs(pg)
    check(f'{tag} part 4, top-p 0.9: 4 on the flat list, 1 after “a burger”, the earlier result kept on its row (R3.6)', Z(lang, 'Top-p 0.9 keeps 4 words on the flat list but only 1 after “a burger”.', 'top-p 0.9 在平的分布里留下 4 个词，在“汉堡”之后只留下 1 个。') in o
          and text(pg, '#sp-rail [data-lever="k3"] .lv-r') != '', o[:200])
    mk = pg.evaluate("()=>['#sp-duo-f','#sp-duo-k'].map(s=>[...document.querySelectorAll(s+' .sp-mk')].filter(e=>!e.hidden).map(e=>e.textContent))")
    check(f'{tag} part 4, top-p 0.9: the running totals under each list stop at 94% ✓ (A4)', mk[0][-1:] == ['94% ✓'] and '86%' in mk[0] and mk[1] == ['94% ✓'], mk)
    lv = pg.evaluate("()=>[...document.querySelectorAll('#sp-rail .lever')].map(b=>{const t=b.querySelector('.lv-t b'); return t.scrollWidth<=t.clientWidth+1&&t.getClientRects().length<=1&&t.getBoundingClientRect().height<30})")
    check(f'{tag} part 4: every rule’s title stays on one line (R7.8)', all(lv), lv)
    fit(pg, f'{tag} part 4, top-p 0.9')
    if phone:
        pg.evaluate("()=>{const t=document.getElementById('sp-duo-k'); window.scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} part 4: the mini result bar, a 2–3 word status and a sentence without the rule name (A15)', dk['shown'] and dk['status'] == Z(lang, '4 + 1 kept', '留下 4 + 1') and dk['read'].startswith(Z(lang, 'First word:', '第一个词')), dk)
    shot(pg, f'{tag}_p4', full=phone)
    rules(pg, f'{tag} part 4', lang)
    # part 5: the tree; the first random number
    click(pg, '#sp-rail [data-nav="next"]')
    check(f'{tag} part 5: the tree, three columns, Next waits', vis(pg, '#sp-tree') and pg.evaluate("()=>document.querySelectorAll('#sp-tree .sp-col').length") == 3 and all(nexts(pg)))
    slide(pg, '#sp-ic-u1-in', 30)
    check(f'{tag} part 5: 0.30 still lands on noodles, Next still waits', all(nexts(pg)) and Z(lang, 'Move it until it lands on another first word.', '移动它，直到落在另一个第一个词上。') in obs(pg), obs(pg)[:200])
    if phone:
        jy = still(pg, '#sp-ph-u1-in', 45)
        check(f'{tag} part 5: on a phone the slider stays put when Next becomes enabled mid-drag (re-review N2)', jy['moved'] < 1 and not any(jy['next']), jy)
    slide(pg, '#sp-ic-u1-in', 97)
    o = obs(pg)
    check(f'{tag} part 5 at 0.97: salad, and the whole sentence changes; Next enabled; three blocks outlined', Z(lang, '“salad with chicken” instead of “noodles with beef”', '“沙拉加鸡肉”，而不是“面条加牛肉”') in o and not any(nexts(pg))
          and pg.evaluate("()=>document.querySelectorAll('#sp-tree .sp-blk.on').length") == 3, o[:250])
    check(f'{tag} part 5: a label for the thin outlined blocks, and three random-number lines', pg.evaluate("()=>document.querySelectorAll('#sp-tree .sp-tag').length") >= 1 and pg.evaluate("()=>document.querySelectorAll('#sp-tree .sp-u').length") == 3)
    ut = pg.evaluate("()=>[...document.querySelectorAll('#sp-tree .sp-u em')].map(e=>e.dataset.tipx||'')")
    check(f'{tag} part 5: each random number’s label says what it is and where it lands (A10)', len(ut) == 3 and all(Z(lang, 'random number of your run', '随机数') in t for t in ut), ut[:1])
    check(f'{tag} part 5: no “table” anywhere (A10)', Z(lang, 'in the table', '表里 38') not in all_text(pg))
    ov = pg.evaluate("""()=>{const R=e=>e.getBoundingClientRect(), hit=(a,b)=>a.left<b.right-1&&b.left<a.right-1&&a.top<b.bottom-1&&b.top<a.bottom-1, out=[];
      document.querySelectorAll('#sp-tree .sp-tag').forEach(t=>{const col=t.closest('.sp-colb'); col.querySelectorAll('.sp-blk>span:not([hidden]), .sp-u em').forEach(x=>{if(x.getClientRects().length&&hit(R(t),R(x))) out.push(t.textContent+' / '+x.textContent);});}); return out}""")
    check(f'{tag} part 5 at 0.97: no tag covers a block label or a random number (A11, R7.8)', not ov, ov)
    tree_hits = {}
    for u in (20, 45, 50, 80, 90, 99, 97):
        slide(pg, '#sp-ic-u1-in', u, change=False)
        h = pg.evaluate(TREE_OV)
        if h: tree_hits[u] = h
    check(f'{tag} part 5: no line, number label or tag runs through a block label at 0.20, 0.45, 0.5, 0.8, 0.9, 0.99, 0.97 (re-review N1)', not tree_hits, tree_hits)
    fit(pg, f'{tag} part 5')
    shot(pg, f'{tag}_p5', full=phone)
    rules(pg, f'{tag} part 5', lang)
    # part 6
    click(pg, '#sp-rail [data-nav="next"]')
    t6 = text(pg, '#sp-rail .step.current')
    check(f'{tag} part 6: three misconceptions, and the button to the scenarios (R3.12)', t6.count('✗') == 3 and vis(pg, '#sp-rail [data-mode-go="sandbox"]'), t6[:200])
    check(f'{tag} part 6: four lines, each a rule and what follows from it (G2, 2026-10-08), with a button that shows its picture (A6)', pg.evaluate("()=>document.querySelectorAll('#sp-rail .sp-remlist li [data-rem]').length") == 4
          and Z(lang, 'a random draw picks the word', '随机抽') in t6 and Z(lang, 'one different early pick can change the whole sentence', '前面抽到一个不同的词，整句话都可能变样') in t6, t6[:300])
    _rl = text(pg, '#sp-rail .step.current .sp-remlist') + ' ' + text(pg, '#sp-rail .step.current .obs')
    check(f'{tag} part 6 (G2): no point or misconception leans on a detail of an earlier part', not re.search(r'[Pp]art \d|第 \d 部分', _rl), _rl[:300])
    for k, v in (('cut', 'duo'), ('tree', 'b'), ('temp', 'a')):
        click(pg, f'#sp-rail [data-rem="{k}"]')
        st6 = state(pg)
        check(f'{tag} part 6, “{k}”: its picture in the chart, the button pressed', st6['remView'] == k and pg.evaluate(f"()=>document.querySelector('#sp-rail [data-rem=\"{k}\"]').getAttribute('aria-pressed')") == 'true'
              and vis(pg, {'duo': '#sp-duo', 'b': '#sp-tree', 'a': '#sp-rows'}[v]), st6)
        fit(pg, f'{tag} part 6 {k}')
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; const r=document.querySelector('#sp-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end .next-lab');
      return innerWidth<=760 ? shown(e) : shown(r)}""")
    check(f'{tag} part 6: the next-lab footer (R3.11)', nl)
    shot(pg, f'{tag}_p6', full=phone)
    rules(pg, f'{tag} part 6', lang)
    click(pg, '#sp-rail [data-mode-go="sandbox"]')
    check(f'{tag} “See the scenarios” opens Scenarios', state(pg)['mode'] == 'sandbox')
    check(f'{tag} no overflow at the end of the tour', overflow(pg) <= 0, overflow(pg))
    pg.close()


def run_modes(b, w, h, lang):
    tag = f'{w}_{lang}'
    pg = page(b, w, h, lang)
    phone = w <= 760
    mode(pg, 'sandbox')
    presets = pg.evaluate("()=>document.querySelectorAll('#sp-rail [data-preset]').length")
    check(f'{tag} scenarios: four, single choice, the lead-in (G1, 2026-10-08)', presets == 4 and Z(lang, 'Pick a scenario and see how its usual settings change the words that get drawn.', '选一个场景，看看它常用的设置怎样改变抽到的词。') in text(pg, '#sp-rail .railcard'))
    for i in range(presets):
        click(pg, f'#sp-rail [data-preset="{i}"]')
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#sp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        ins = text(pg, '#sp-insight')
        check(f'{tag} scenario {i}: pressed alone, with its result', pressed.count('true') == 1 and pressed[i] == 'true' and len(ins) > 80, (pressed, ins[:120]))
        check(f'{tag} scenario {i}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i}', lang); fit(pg, f'{tag} scenario {i}')
    check(f'{tag} scenario 4 (brainstorming): a full opening sentence, and 5 words that can be picked (A21)', Z(lang, 'Brainstorming often uses a higher temperature', '头脑风暴常用较高的 temperature') in text(pg, '#sp-insight') and Z(lang, '5 words that can be picked', '还剩 5 个词可能被抽中') in text(pg, '#sp-insight'), text(pg, '#sp-insight')[:200])
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} scenarios: the URL carries the setup', hsh == '#sampling~12_19_0_1_a', hsh)
    shot(pg, f'{tag}_scenario_brain', full=phone)
    # the tree view
    click(pg, '#sp-view button[data-view="b"]')
    check(f'{tag} scenarios: the sentence tree view, with the likeliest sentence outlined', vis(pg, '#sp-tree') and pg.evaluate("()=>document.querySelectorAll('#sp-tree .sp-blk.on').length") == 3 and state(pg)['view'] == 'b')
    shot(pg, f'{tag}_scenario_tree', full=phone)
    rules(pg, f'{tag} scenario tree', lang); fit(pg, f'{tag} scenario tree')
    check(f'{tag} scenario tree: the legend says what ×N means (A18)', Z(lang, 'of the 20 runs (seed 1) wrote this sentence', '20 次运行（种子 1）里有 N 次写成这句') in text(pg, '#sp-legend'), text(pg, '#sp-legend'))
    pg2 = page(b, w, h, lang, hash_='sampling~12_19_0_1_b')
    st = state(pg2)
    check(f'{tag} hash round-trip: Scenarios, temperature 1.2, top-p 0.95, the tree view, “Brainstorm names” pressed', st['mode'] == 'sandbox' and (st['T'], st['P'], st['K'], st['view']) == (1.2, 0.95, 0, 'b')
          and pg2.evaluate("()=>document.querySelector('#sp-rail [data-preset=\"3\"]').getAttribute('aria-pressed')") == 'true', st)
    pg2.close()
    click(pg, '#sp-view button[data-view="a"]')
    pg.evaluate("()=>document.querySelector('#sp-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
    y0 = pg.evaluate("()=>scrollY")
    slide(pg, '#sp-ctl-P-in', 10, change=False)
    if not phone:
        check(f'{tag} a change under Settings does not scroll the page', abs(pg.evaluate("()=>scrollY") - y0) < 2)
    pg.evaluate("()=>document.querySelector('#sp-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#sp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: after a change under Settings no scenario is pressed, and the card says so', 'true' not in pressed and Z(lang, 'match none', '都不一样') in text(pg, '#sp-insight'), pressed)
    # the challenge
    click(pg, '#sp-rail [data-mode-go="challenge"]')
    st = state(pg)
    check(f'{tag} challenge: temperature 1.0 and no top-p to start', st['mode'] == 'challenge' and (st['T'], st['P'], st['K']) == (1, 1, 0), st)
    check(f'{tag} challenge: Settings locked with a note, no grayed controls (R4.2)', vis(pg, '#sp-lock') and not vis(pg, '#sp-ctl-P') and not vis(pg, '#sp-ctl-z'))
    check(f'{tag} challenge: the capability card after the task', vis(pg, '#sp-rail [data-exits]'))
    check(f'{tag} challenge: live progress red for the wrong words at the start (R4.3)', pg.evaluate("()=>document.getElementById('sp-ch-g1').classList.contains('sp-over')") and text(pg, '#sp-ch-w') == '7.5%')
    check(f'{tag} challenge: no overflow', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_challenge', full=phone)
    rules(pg, f'{tag} challenge', lang); fit(pg, f'{tag} challenge')
    cm = pg.evaluate("()=>[...document.querySelectorAll('#sp-cum-marks .sp-mk')].filter(e=>!e.hidden).map(e=>e.textContent)")
    check(f'{tag} challenge: the running totals end where the good words end, then the wrong words (A9)', cm[-2:] == ['92%', '✗✗✗'] and len(pg.evaluate("()=>[...document.querySelectorAll('#sp-cumbar .sp-otick')]")) == 3, cm)
    seg = lambda k, v: click(pg, f'#sp-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]')
    seg('chT', '0.5')
    click(pg, '#sp-rail [data-act="submit"]')
    res = text(pg, '#sp-ch-res')
    check(f'{tag} challenge tier 3: not met (only 2 words at 5%), a hint and “Show the answer” (R4.4)', res.startswith(Z(lang, 'Not yet', '还没达到')) and Z(lang, 'Hint', '提示') in res and vis(pg, '#sp-ch-res [data-act="reveal"]')
          and text(pg, '#sp-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案') and not vis(pg, '#sp-ch-res .cmp'), res[:200])
    click(pg, '#sp-ch-res [data-act="reveal"]')
    # the owner's review of batch 8 (2026-10-07): the answer keeps the 5 settings that pass, the pick (T 0.5, no top-p) and the initial
    # setup; the note counts the other 9 of the 16
    check(f'{tag} challenge chart: the title names the chance and the goal, not “each temperature and top-p”', text(pg, '#sp-ch-res .cmp-title') == Z(lang, 'Chance of picking one of the three wrong words after “Another word for ‘happy’ is” (goal: below 1%, with at least 3 good words at 5% or more)', '“‘开心’的近义词是”之后抽中三个错词之一的概率（目标：低于 1%，且至少 3 个好词各有 5% 以上）'), text(pg, '#sp-ch-res .cmp-title'))
    a1 = pg.evaluate("""()=>{const ok=getComputedStyle(document.documentElement).getPropertyValue('--c-ok').trim(), d=document.createElement('i'); d.style.color=ok; document.body.appendChild(d); const okc=getComputedStyle(d).color; d.remove();
      const rows=[...document.querySelectorAll('#sp-ch-res .cmp-row.ok')].map(r=>{const t=r.querySelector('.cmp-track'), l=r.querySelector('.cmp-lab>span'); return {w:t.getBoundingClientRect().width, ring:getComputedStyle(t).boxShadow.includes(okc.replace(/\\)$/,'')), lab:getComputedStyle(l).color===okc, tick:l.textContent.startsWith('✓')}});
      const ax=document.querySelector('#sp-ch-res .cmp-axis span'), tr=document.querySelector('#sp-ch-res .cmp-track'), a=ax.getBoundingClientRect(), b=tr.getBoundingClientRect(), card=document.getElementById('sp-rail').getBoundingClientRect();
      return {rows, goal:{inside:a.left>=b.left-0.5&&a.right<=b.right+0.5&&a.left>=card.left, fits:ax.scrollWidth<=ax.clientWidth+1}}}""")
    check(f'{tag} challenge chart: every passing row is marked green at 0% (✓, green label, green ring), and the goal label sits inside the plot (A1)', len(a1['rows']) == 5 and all(r['w'] > 0 and r['ring'] and r['lab'] and r['tick'] for r in a1['rows']) and a1['goal']['inside'] and a1['goal']['fits'], a1)
    seg('chT', '1.3'); seg('chP', '0.9')
    check(f'{tag} challenge at T 1.3, top-p 0.90: the counter counts good words only, 4 (A7)', text(pg, '#sp-ch-n') == '4' and text(pg, '#sp-ch-g2').startswith(Z(lang, 'Good words at 5% or more', '概率 5% 以上的好词')), text(pg, '#sp-ch-g2'))
    fit(pg, f'{tag} challenge chart')
    seg('chT', '0.5'); seg('chP', '1')
    rows = pg.evaluate("()=>[...document.querySelectorAll('#sp-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),init:!!r.querySelector('.cmp-tag')}))")
    check(f'{tag} challenge chart: 7 rows of 16, the 5 green ones, the pick (T 0.5, no top-p) and the initial setup; no other row; the note counts 9',
          len(rows) == 7 and sum(r['ok'] for r in rows) == 5 and sum(r['me'] for r in rows) == 1 and sum(r['init'] for r in rows) == 1 and all(r['ok'] or r['me'] or r['init'] for r in rows)
          and not any(r['ok'] for r in rows if r['me'] or r['init']) and text(pg, '#sp-ch-res .cmp-note') == Z(lang, 'The other 9 options miss the goal and are not shown.', '其余 9 种没达到目标，图里没有列出。'), rows)
    seg('chT', '0.7'); seg('chP', '0.9')
    res = text(pg, '#sp-ch-res')
    check(f'{tag} challenge tier 2 (two changes): met, but one change is enough', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'one change is enough', '改一处就够了') in res, res[:220])
    seg('chT', '1')
    seg('chP', '0.8')
    res = text(pg, '#sp-ch-res')
    check(f'{tag} challenge tier 2 (one change, cuts too much): met, but top-p 0.90 keeps “pleased” too', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'you cut more than needed', '截得比需要的多') in res, res[:220])
    seg('chP', '0.9')
    res = text(pg, '#sp-ch-res')
    check(f'{tag} challenge tier 1: temperature 1.0 and top-p 0.90, the smallest change', res.startswith(Z(lang, 'Goal met', '达成目标')) and Z(lang, 'that is the smallest change', '改动最少') in res and text(pg, '#sp-status') == Z(lang, 'Both targets met', '两个目标都达到'), res[:220])
    check(f'{tag} challenge: baseline → effect → reason, and a specific caveat (R5.5, R6.5)', all(x in res for x in [Z(lang, 'initial setup', '初始设置'), Z(lang, 'Why top-p 0.90 works', '为什么行'), Z(lang, 'Note:', '注意：')]), res[-400:])
    shot(pg, f'{tag}_challenge_met', full=phone)
    rules(pg, f'{tag} challenge met', lang)
    for t in (0, 1):
        pg.evaluate(f"()=>document.querySelector('#sp-rail [data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        for v in pg.evaluate("()=>[...document.querySelectorAll('#sp-rail [data-exits] .ex-opts .choice')].map(b=>+b.dataset.v)"):
            pg.evaluate(f"()=>document.querySelector('#sp-rail [data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
            rules(pg, f'{tag} capability card tab {t + 1}, option {v}', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    st = state(pg)
    check(f'{tag} challenge: a language switch keeps the answer and the submitted result', (st['T'], st['P']) == (1, 0.9) and st['chSubmitted'] and vis(pg, '#sp-ch-res .cmp'), st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if phone:
        mode(pg, 'guided'); go_part(pg, 0)
        pg.evaluate("()=>{const t=document.getElementById('sp-strip'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
        check(f'{tag} mini result bar: shown once the chart scrolls away, with a short status and the runs', dk['shown'] and len(dk['status']) <= 14 and ('5' in dk['read']), dk)
        shot(pg, f'{tag}_dock')
        rules(pg, f'{tag} dock', lang)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            run_guided(b, w, h, lang)
            run_modes(b, w, h, lang)
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
_G_SRC = (P / 'labs' / 'sampling.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a rule and see how many words it keeps on the flat list and on the peaked list.', '选一个规则，看看平的和尖的分布上各留下几个词。', 'Pick a scenario and see how its usual settings change the words that get drawn.', '选一个场景，看看它常用的设置怎样改变抽到的词。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('In part 1 the same logits gave', 'So the same prompt can give a different answer on every run'), ('在第 1 部分，同样的 logit', '所以同一个 prompt 每次运行都可能写出不同的回答'), ('In part 3, temperature 0.5 raised', 'Below 1 the likeliest word gains, and at 0 every run writes the same word.'), ('在第 3 部分，temperature 0.5', '等于 0 时，每次运行都写同一个词。'), ('In part 4, top-p 0.9 kept four', 'So top-p keeps many words when the chances are flat and few when they are peaked.'), ('在第 4 部分，top-p 0.9 在午饭词里', '所以概率平的时候，top-p 留下很多词，尖的时候只留几个。'), ('In part 5, moving only the first random number', 'so one different early pick can change the whole sentence.'), ('在第 5 部分，只移动第一个随机数', '所以前面抽到一个不同的词，整句话都可能变样。'), ('in part 3 every run wrote', 'temperature 0 is greedy: every run writes the likeliest word.'), ('第 3 部分里每次运行都写了', 'temperature 0 就是贪心，每次运行都写最可能的那个词。'), ('in part 4 that took four of the six first words', 'On a flat list that can be most of the words, on a peaked list a single word.'), ('在第 4 部分，第一个词的六个候选里', '分布平的时候可能要留下大部分词，尖的时候可能只留一个。'), ('drag the temperature to 1.5 in part 3', 'so odd words turn up more often.'), ('在第 3 部分把 temperature 拖到 1.5', '怪词就更常出现。'), ('Puts part 1’s 20 runs back', 'Puts the 20 runs at temperature 1 back in the chart.'), ('在图里重新显示第 1 部分的 20 次运行', '在图里重新显示 temperature 1 时的 20 次运行。'), ('Shows part 4’s two lists', 'Shows the flat list and the peaked list with top-p 0.9'), ('在图里显示第 4 部分的两个分布', '在图里显示平的和尖的两个分布'), ('Shows part 5’s tree', 'Shows the tree of sentences with your run.'), ('在图里显示第 5 部分的句子树', '在图里显示句子树和你这次运行。')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# H1, H2, H5: the owner's site-wide decisions of 2026-10-09 (sweep basics-b): every formula typeset with A.ui.math, never a plain-text
# string; Show the math in short blocks (a lead line, the formula with its name, the numbers, the result); the chart's blocks separated.
# From the source (both languages), then on the page.
_H_SRC = (P / 'labs' / 'sampling.js').read_text()
for _bad in ['e<sup>', 'e¹ ÷', 'seed × 7919', '种子 × 7919', '“÷ T”', '>÷ T<', 'Softmax of the logits divided by T', '对除以 T 之后的 logit',
             'e to the power of its logit divided by T', 'e 的（logit ÷ T）次方', 'each chance = its value ÷ the sum', '每个概率 = 自己的值 ÷ 总和']:
    check(f'H1: no plain-text formula “{_bad}” left in the lab (EN and ZH)', _bad not in _H_SRC)
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for lang in LANGS:
        pg = page(_b, 1440, 900, lang)
        go_part(pg, 1); pg.evaluate("()=>document.querySelector('#sp-rail [data-pred=\"few\"]').click()"); pg.wait_for_timeout(200)
        go_part(pg, 2)
        pg.evaluate("()=>{document.getElementById('sp-math').open=true}"); pg.wait_for_timeout(250)
        h = pg.evaluate("""()=>{const m=document.getElementById('sp-mathbody'); return {math:m.querySelectorAll('math').length, blocks:m.querySelectorAll('.sp-mblk').length,
          eq:m.querySelectorAll('.sp-eq').length, tab:m.querySelectorAll('.sp-mtab tr').length, sup:m.querySelectorAll('sup').length,
          words:/(^|[^a-z])(frac|left|right|sqrt|cdot|times)([^a-z]|$)/.test(m.textContent), card:document.querySelectorAll('#sp-rail .step.current math').length,
          col:!!document.querySelector('#sp-rows .sp-zh math'), rule:getComputedStyle(document.getElementById('sp-cum')).borderTopWidth,
          rule2:getComputedStyle(document.getElementById('sp-strip')).borderTopWidth}}""")
        check(f'{lang} H1/H2: Show the math is short blocks of typeset formulas (a lead line, the formula, a table, a chain lined up at “=”), no HTML superscripts, no TeX command shown as a word', h['math'] >= 6 and h['blocks'] >= 2 and h['eq'] >= 1 and h['tab'] >= 7 and h['sup'] == 0 and not h['words'], h)
        check(f'{lang} H1: part 3’s card shows softmax with temperature typeset, and the ÷ T column’s header is typeset', h['card'] >= 2 and h['col'], h)
        check(f'{lang} H5: the chart’s blocks (the bar laid end to end, the 20 runs) are separated by a hairline', h['rule'] == '1px' and h['rule2'] == '1px', h)
        pg.close()
    _b.close()
print(f'{n} checks, {len(fails)} failed')
print('FAILED:', fails[:60] if fails else 'none')
sys.exit(1 if fails else 0)
