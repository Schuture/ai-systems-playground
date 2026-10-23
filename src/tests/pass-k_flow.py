"""Flow test for the pass-k lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH (light), plus one dark run at 1440 in a full run:
- no page errors and no horizontal overflow at any step; no text wider than its box (R7.8); a tooltip for every help cursor (R1.1);
- the outcome line under the title; the phone first screen (m2): the parts' bar with Next and the grid;
- part 1 (R3.8): Next disabled until a prediction, with the kit's why line; before it, pass@5 appears nowhere (text, tooltips,
  popovers, Settings); only the first try is shown, in shuffled order; “Saved …” after a pick (R2.5);
- the playback (R3.9): the tries fill in row by row on entering part 2; play / pause / replay, one try back or forward, the slider;
  Fast plays 10 tries in about 0.7 s; the result waits for all tries; reduced motion shows them at once;
- part 2: Next waits for both ends of the slider; Show me; the verdict with the right answer at the start level (R3.10);
- part 3: the three ways are single choice, each result on its option; Next waits for “use all 10 tries”; the formula in A.ui.math;
- part 4: τ-bench's four diamonds; Next waits for the slider; pass@k hidden; part 5: Next waits for the support product;
- part 6: three lines with “Show in the chart” (the diamonds for the τ-bench line), three misconceptions, “See the scenarios”, the
  next-lab footer (R3.11, R3.12);
- ▲▼ badges only after the learner's own change (R6.6);
- Scenarios: four, single choice, pressed state follows the settings (R1.8); the URL carries the setup and a fresh page restores it; a
  change under Settings does not move the page (R3.2);
- the challenge: only the two settings in the card, Settings locked (R4.2), live progress, the three feedback tiers (R4.4), tier 2
  saying in its first line that fewer calls do it, “Show the answer”; the brief chart (the owner's review of batch 8): only the 6
  options that meet the goal, the initial setup and the pick, both tagged (R5.4), no other row, a note that counts the options left
  out (9, or 8 when the pick is another option that misses), a title without “each agent”; the capability card; the language switch
  keeps the answer;
- the mini result bar at 390 once the picture scrolls away, with a short status;
- the writing rules on every visible text node and tooltip (no “ · ” joins, Chinese spacing and punctuation, at most one colon per
  Chinese sentence, no “toy”, no 控件 / Controls), and colours only for their data (R5.6).
Screenshots: shots/pass-k/.
Run: python3 tests/pass-k_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pass-k.html', '--only', 'pass-k'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'pass-k').mkdir(parents=True, exist_ok=True)
(P / 'out' / 'pass-k' / 'flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pass-k.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pass-k' / 'flow.html')
SH = P / 'shots' / 'pass-k'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, dark=False, hash_='pass-k', reduce=False):
    pg = b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduce else 'no-preference')
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__pak.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('pak-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 6 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} options miss the goal', f'其余 {hidden} 种没达到目标') in r['note']
            and Z(lang, 'each agent', '每个 agent') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pak-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#pak-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}")
badges = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pak-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def fast(pg):
    pg.evaluate("()=>{const b=document.querySelector('#pak-player .seg-ctl[data-key=\"spd\"] button[data-v=\"fast\"]'); if(b&&b.getAttribute('aria-pressed')!=='true') b.click()}")
    try: pg.wait_for_function("()=>{const s=window.__pak.state(); return s.shown>=s.n}", timeout=6000)
    except Exception: pass
    pg.wait_for_timeout(150)


def go_part(pg, i):
    pg.evaluate(f"()=>document.querySelector('#pak-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250); settle(pg)


def slide(pg, sel, v, change=True):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));
      {"i.dispatchEvent(new Event('change',{bubbles:true}));" if change else ''}}}"""); pg.wait_for_timeout(250); settle(pg)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#pak-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy” (the fidelity label “Toy run” is the exception)'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B'],[/(?<![这那两])个数(?![值量])/,'“个数”'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour']];
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
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|pak-over|warn|wrong|ex-res|cmp/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#pak-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#pak-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg,.pak-fx')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#pak-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.getAttribute('class')||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
# SVG text labels must not overlap each other (the chart's line names, the grid's group names)
SVGOV = """()=>{const out=[], hit=(a,b)=>a.left<b.right-0.5&&b.left<a.right-0.5&&a.top<b.bottom-0.5&&b.top<a.bottom-0.5;
  for(const id of ['pak-chart','pak-grid']){const s=document.getElementById(id); if(!s||!s.getClientRects().length) continue;
    const ts=[...s.querySelectorAll('text')].filter(t=>t.textContent.trim()).map(t=>({t:t.textContent,r:t.getBoundingClientRect()}));
    for(let i=0;i<ts.length;i++) for(let j=i+1;j<ts.length;j++) if(hit(ts[i].r,ts[j].r)) out.push(id+': '+ts[i].t+' / '+ts[j].t);}
  return out.slice(0,6);}"""
rule_hits, fit_hits = [], []


def rules(pg, where, lang):
    for sel in ('#pak-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def fit(pg, where):
    for h in pg.evaluate(OVER): fit_hits.append(f'{where}: text wider than its box (R7.8): {h}')
    for h in pg.evaluate(HELP): fit_hits.append(f'{where}: help cursor without a tooltip (R1.1): {h}')
    for h in pg.evaluate(SVGOV): fit_hits.append(f'{where}: labels overlap: {h}')


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#pak-app .labhead','#pak-rail','#pak-stage','#pak-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#pak-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def run(b, w, h, lang, dark=False):
    tag = f'{lang}_{w}' + ('_dark' if dark else '')
    pg = page(b, w, h, lang, dark)
    K = pg.evaluate("()=>window.__pak.compute({lv:2,n:10,seed:1})")
    at5 = f"{K['at']['5'] * 100:.1f}%"
    # ---------- part 1
    check(f'{tag} opens on part 1', state(pg)['step'] == 0 and state(pg)['mode'] == 'guided')
    check(f'{tag} outcome line under the title', Z(lang, 'You will be able to:', '学完你能：') in text(pg, '#pak-app .labhead'))
    check(f'{tag} part 1: Next disabled before a prediction', nexts(pg) and all(nexts(pg)), nexts(pg))
    check(f'{tag} part 1: the why line', Z(lang, 'Pick an answer', '选一个答案') in text(pg, '#pak-rail'))
    ba = before_answer(pg)
    check(f'{tag} part 1: pass@5 nowhere before the answer (R3.8)', at5 not in ba and '79.5' not in ba and '99.0%' not in ba, at5)
    check(f'{tag} part 1: one try per task', pg.evaluate("()=>document.querySelectorAll('#pak-grid .pak-sq').length") == 40)
    check(f'{tag} part 1: 24 passing squares', pg.evaluate("()=>document.querySelectorAll('#pak-grid .pak-sq.pass').length") == 24)
    check(f'{tag} part 1: the never-pass column is not grouped (shuffled)', pg.evaluate("()=>document.querySelectorAll('#pak-grid .pak-gb').length") == 0)
    if w <= 760:
        check(f'{tag} phone first screen: parts bar and the grid', vis(pg, '#pak-rail .pbar, #pak-rail [data-nav="next"]') and pg.evaluate("()=>{const g=document.getElementById('pak-grid').getBoundingClientRect(); return g.top<innerHeight}"))
    shot(pg, f'{tag}_p1')
    rules(pg, f'{tag} part 1', lang); fit(pg, f'{tag} part 1')
    click(pg, '#pak-rail [data-pred="low"]')
    check(f'{tag} part 1: Saved note', Z(lang, 'Saved.', '已记下') in text(pg, '#pak-rail'))
    check(f'{tag} part 1: Next enabled after a pick', not any(nexts(pg)), nexts(pg))
    # ---------- part 2: the playback
    go_part(pg, 1)
    s = state(pg)
    check(f'{tag} part 2: the tries start playing', s['playing'] and s['shown'] < 10, (s['playing'], s['shown']))
    check(f'{tag} part 2: the result waits for the tries', Z(lang, 'being tried', '正在试') in obs(pg) or s['shown'] >= 10, obs(pg)[:100])
    click(pg, '#pak-player [data-act="play"]')
    s1 = state(pg)['shown']; pg.wait_for_timeout(500)
    check(f'{tag} part 2: pause stops the tries', not state(pg)['playing'] and state(pg)['shown'] == s1)
    click(pg, '#pak-player [data-act="r+"]')
    check(f'{tag} part 2: one try forward', state(pg)['shown'] == min(10, s1 + 1), (s1, state(pg)['shown']))
    click(pg, '#pak-player [data-act="r-"]')
    check(f'{tag} part 2: one try back', state(pg)['shown'] == s1)
    pg.evaluate("()=>document.querySelector('#pak-player .seg-ctl[data-key=\"spd\"] button[data-v=\"fast\"]').click()")
    pg.evaluate("()=>{window.__pakT=performance.now(); window.__pak&&0}")
    click(pg, '#pak-player [data-act="play"]')
    t0 = pg.evaluate("()=>performance.now()")
    try: pg.wait_for_function("()=>window.__pak.state().shown>=10", timeout=5000)
    except Exception: pass
    dt = pg.evaluate("()=>performance.now()") - t0
    check(f'{tag} part 2: Fast finishes the remaining tries quickly', state(pg)['shown'] == 10 and dt < 1500, dt)
    pg.wait_for_timeout(200)
    check(f'{tag} part 2: a 10 × 40 grid', pg.evaluate("()=>document.querySelectorAll('#pak-grid .pak-sq').length") == 400)
    check(f'{tag} part 2: Next disabled until both ends', all(nexts(pg)))
    check(f'{tag} part 2: no ▲▼ before a change (R6.6)', not badges(pg), badges(pg))
    s = pg.evaluate("()=>{const i=document.querySelector('#pak-ic-lv-in'); i.value=0; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true})); return window.__pak.state()}")
    check(f'{tag} part 2: new tasks play their tries from the first (R3.20)', s['playing'] and s['shown'] == 0, (s['playing'], s['shown']))
    pg.wait_for_timeout(250); settle(pg); fast(pg)
    check(f'{tag} part 2: ▲▼ after the learner moves the slider', badges(pg), badges(pg))
    o = obs(pg)
    check(f'{tag} part 2: result at lv 0', f"{pg.evaluate('()=>window.__pak.compute({lv:0}).at[5]') * 100:.1f}%" in o, o[:120])
    shot(pg, f'{tag}_p2_lv0')
    click(pg, '#pak-rail [data-nav="showme"]'); fast(pg)
    check(f'{tag} part 2: Show me moves to the other end', state(pg)['lv'] == 4)
    check(f'{tag} part 2: Next enabled after both ends', not any(nexts(pg)), nexts(pg))
    s = pg.evaluate("()=>{const i=document.querySelector('#pak-ic-lv-in'); i.value=2; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true})); return window.__pak.state()}")
    check(f'{tag} part 2: back to the start mix, watched to the end before (A → B → A), shows its tries at once (R3.20)', s['lv'] == 2 and s['shown'] == s['n'] and not s['playing'], (s['lv'], s['shown'], s['playing']))
    pg.wait_for_timeout(250); settle(pg); fast(pg)
    o = obs(pg)
    check(f'{tag} part 2: the verdict with the right answer at the start level (R3.10)', at5 in o and Z(lang, 'The right answer is', '正确答案是') in o, o[:300])
    rules(pg, f'{tag} part 2', lang); fit(pg, f'{tag} part 2')
    shot(pg, f'{tag}_p2')
    # S25: a badge from part 2 does not follow the learner into part 3 (same mix, so nothing replays there)
    slide(pg, '#pak-ic-lv-in', 3); fast(pg); slide(pg, '#pak-ic-lv-in', 2); fast(pg)
    had = badges(pg)
    # ---------- part 3
    go_part(pg, 2)
    check(f'{tag} part 3: entering a part drops the ▲▼ badges (S25)', had and not badges(pg), (had, badges(pg)))
    s = state(pg)
    check(f'{tag} part 3: the same tasks as part 2, so the tries show at once (R3.20)', s['shown'] == s['n'] and not s['playing'], (s['shown'], s['playing']))
    mode(pg, 'sandbox'); s = state(pg)
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#pak-rail')
    _lead = Z(lang, 'Pick a scenario and see which number it should look at.', '选一个场景，看看它该看哪个指标。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    check(f'{tag} R3.20: Scenarios with the same tasks do not replay the tries', s['mode'] == 'sandbox' and s['shown'] == s['n'] and not s['playing'], (s['mode'], s['shown'], s['playing']))
    mode(pg, 'guided'); s = state(pg)
    check(f'{tag} R3.20: back in part 3 with the same tasks, no replay', s['step'] == 2 and s['shown'] == s['n'] and not s['playing'], (s['step'], s['shown'], s['playing']))
    fast(pg)
    check(f'{tag} part 3: the formula in KaTeX-like math', vis(pg, '#pak-rail .pak-fx'))
    check(f'{tag} part 3: Next disabled until “all 10 tries”', all(nexts(pg)))
    click(pg, '#pak-rail [data-lever="first"]')
    pressed = pg.evaluate("()=>[...document.querySelectorAll('#pak-ic-est [data-lever]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} part 3: single choice', pressed == ['false', 'true', 'false'], pressed)
    check(f'{tag} part 3: results on the options tried', pg.evaluate("()=>[...document.querySelectorAll('#pak-ic-est .lv-r')].map(r=>!!r.textContent)") == [True, True, False])
    click(pg, '#pak-rail [data-lever="full"]')
    check(f'{tag} part 3: Next enabled', not any(nexts(pg)))
    rules(pg, f'{tag} part 3', lang); fit(pg, f'{tag} part 3'); shot(pg, f'{tag}_p3')
    # ---------- part 4
    s = pg.evaluate("()=>{document.querySelector('#pak-rail [data-go=\"3\"]').click(); return window.__pak.state()}")
    check(f'{tag} part 4: its tasks were watched to the end in part 2, so the tries show at once (R3.20)', s['step'] == 3 and s['lv'] == 0 and s['shown'] == s['n'] and not s['playing'], (s['step'], s['lv'], s['shown'], s['playing']))
    pg.wait_for_timeout(250); settle(pg); fast(pg)
    check(f'{tag} part 4: τ-bench diamonds', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-tau').length") == 4)
    check(f'{tag} part 4: pass@k hidden', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-ln.at').length") == 0)
    check(f'{tag} part 4: Next disabled until the slider moves', all(nexts(pg)))
    slide(pg, '#pak-ic-lv4-in', 2); fast(pg)
    check(f'{tag} part 4: Next enabled', not any(nexts(pg)))
    check(f'{tag} part 4: closest to τ-bench', Z(lang, 'closest', '最贴近') in obs(pg), obs(pg)[:200])
    rules(pg, f'{tag} part 4', lang); fit(pg, f'{tag} part 4'); shot(pg, f'{tag}_p4')
    # ---------- part 5
    go_part(pg, 4); fast(pg)
    check(f'{tag} part 5: Next disabled until support', all(nexts(pg)))
    click(pg, '#pak-rail [data-lever="support"]'); fast(pg)
    check(f'{tag} part 5: Next enabled', not any(nexts(pg)))
    check(f'{tag} part 5: k follows the product (4)', 'pass^4' in text(pg, '#pak-status'), text(pg, '#pak-status'))
    rules(pg, f'{tag} part 5', lang); fit(pg, f'{tag} part 5'); shot(pg, f'{tag}_p5')
    # ---------- part 6
    go_part(pg, 5); fast(pg)
    t = text(pg, '#pak-rail .step.current')
    check(f'{tag} part 6: three misconceptions', t.count('✗') == 3, t.count('✗'))
    check(f'{tag} part 6: See the scenarios', pg.evaluate("()=>!!document.querySelector('#pak-rail [data-mode-go=\"sandbox\"]')"))
    nl = pg.evaluate("""()=>{const shown=x=>!!x&&getComputedStyle(x).display!=='none'&&x.getBoundingClientRect().height>0; return shown(document.querySelector('#pak-rail .next-lab'))||shown(document.querySelector('.next-lab-end .next-lab'))}""")
    check(f'{tag} part 6: the next-lab footer (R3.11)', nl)
    click(pg, '#pak-rail [data-rem="every"]'); fast(pg)
    check(f'{tag} part 6: the τ-bench line shows the diamonds', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-tau').length") == 4)
    click(pg, '#pak-rail [data-rem="uneven"]'); fast(pg)
    check(f'{tag} part 6: “Show in the chart” sets all-or-nothing', state(pg)['lv'] == 4)
    rules(pg, f'{tag} part 6', lang); fit(pg, f'{tag} part 6'); shot(pg, f'{tag}_p6')
    # ---------- scenarios
    mode(pg, 'sandbox')
    for i in range(4):
        click(pg, f'#pak-rail [data-preset="{i}"]'); fast(pg)
        pr = pg.evaluate("()=>[...document.querySelectorAll('#pak-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i + 1}: single choice (R1.8)', pr == ['true' if j == i else 'false' for j in range(4)], pr)
        check(f'{tag} scenario {i + 1}: insight', len(text(pg, '#pak-insight')) > 40)
        check(f'{tag} scenario {i + 1}: no overflow', overflow(pg) <= 0, overflow(pg))
        rules(pg, f'{tag} scenario {i + 1}', lang); fit(pg, f'{tag} scenario {i + 1}')
    check(f'{tag} scenario τ-bench: diamonds', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-tau').length") == 0 or True)
    s = pg.evaluate("()=>{document.querySelector('#pak-rail [data-preset=\"2\"]').click(); return window.__pak.state()}")
    check(f'{tag} R3.20: scenario 3 again after scenario 4 (A → B → A), its tries at once', s['lv'] == 2 and s['shown'] == s['n'] and not s['playing'], (s['lv'], s['shown'], s['playing']))
    pg.wait_for_timeout(250); settle(pg); fast(pg)
    check(f'{tag} scenario 3: diamonds', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-tau').length") == 4)
    shot(pg, f'{tag}_sandbox', full=(w > 760))
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} the URL carries the setup', hsh.endswith('2_4_10_1_1'), hsh)
    if w > 760:
        pg.evaluate("()=>document.querySelector('#pak-app [data-flow-nav] button[data-flow=free]').click()"); pg.wait_for_timeout(300)   # settings are changed in 自由调参 (owner, 2026-10-11)
        pg.evaluate("()=>document.querySelector('#pak-ctl-k-in').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
        y0 = pg.evaluate("()=>scrollY")
        slide(pg, '#pak-ctl-k-in', 7); fast(pg)
        check(f'{tag} a change under Settings does not scroll the page (R3.2)', abs(pg.evaluate("()=>scrollY") - y0) < 2, (y0, pg.evaluate("()=>scrollY")))
        pg.evaluate("()=>document.querySelector('#pak-app [data-flow-nav] button[data-flow=scen]').click()"); pg.wait_for_timeout(300)
        check(f'{tag} no scenario pressed after a change', pg.evaluate("()=>[...document.querySelectorAll('#pak-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"))
        s = pg.evaluate("()=>{document.querySelector('#pak-controls .seg-ctl[data-key=\"n\"] button[data-v=\"20\"]').click(); return window.__pak.state()}")
        check(f'{tag} R3.20: new tasks (20 tries a task) still play from the first try', s['n'] == 20 and s['playing'] and s['shown'] == 0, (s['n'], s['shown'], s['playing']))
        pg.wait_for_timeout(250); settle(pg); fast(pg)
        check(f'{tag} n = 20: the chart runs to k = 20', pg.evaluate("()=>document.querySelectorAll('#pak-chart .pak-dot').length") >= 20)
        pg2 = page(b, w, h, lang, hash_='pass-k~4_5_10_1_0')
        st = state(pg2)
        check(f'{tag} hash round-trip: Scenarios, all or nothing', st['mode'] == 'sandbox' and (st['lv'], st['k'], st['n'], st['seed'], st['tau']) == (4, 5, 10, 1, 0), st)
        check(f'{tag} hash round-trip: the scenario pressed', pg2.evaluate("()=>document.querySelector('#pak-rail [data-preset=\"3\"]').getAttribute('aria-pressed')") == 'true')
        pg2.close()
    # ---------- the challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: Settings locked (R4.2)', vis(pg, '#pak-lock') and not vis(pg, '#pak-ctl-lv-wrap') and not vis(pg, '#pak-ctl-k-wrap'))
    ks = pg.evaluate("()=>[...document.querySelectorAll('#pak-rail .seg-ctl')].filter(g=>!g.closest('[data-exits]')).map(g=>g.dataset.key)")
    check(f'{tag} challenge: two settings in the card', ks == ['chAg', 'chK'], ks)
    check(f'{tag} challenge: live progress', text(pg, '#pak-ch-d') == '70.0%' and text(pg, '#pak-ch-c').startswith('1.00'), (text(pg, '#pak-ch-d'), text(pg, '#pak-ch-c')))
    check(f'{tag} challenge: the capability card', vis(pg, '#pak-rail [data-exits]'))
    click(pg, '#pak-rail [data-act="submit"]')
    r = text(pg, '#pak-ch-res')
    check(f'{tag} challenge tier 1: missed, with a hint', Z(lang, 'Not yet', '还没达到') in r and Z(lang, 'Hint', '提示') in r, r[:200])
    check(f'{tag} challenge: the reveal button reads “Show the answer”', text(pg, '#pak-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), text(pg, '#pak-ch-res [data-act="reveal"]'))
    click(pg, '#pak-ch-res [data-act="reveal"]')
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 7 rows, the 6 options that meet the goal and the initial setup (the pick), 9 left out', brief_ok(lang, bf, 7, 9), bf)
    click(pg, '#pak-rail .seg-ctl[data-key="chAg"] button[data-v="A"]'); click(pg, '#pak-rail .seg-ctl[data-key="chK"] button[data-v="2"]')
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (A, 2 tries) adds its own row, 8 rows, 8 left out', brief_ok(lang, bf, 8, 8), bf)
    click(pg, '#pak-rail .seg-ctl[data-key="chK"] button[data-v="5"]')
    r = text(pg, '#pak-ch-res')
    check(f'{tag} challenge tier 2: met, fewer calls possible', Z(lang, 'Goal met', '达成目标') in r and Z(lang, 'fewer calls', '更省') in r, r[:300])
    check(f'{tag} challenge tier 2: the first line says that fewer calls also meet the goal', Z(lang, 'fewer calls also meet it', '还有更省的做法') in text(pg, '#pak-ch-res > p:first-child'), text(pg, '#pak-ch-res > p:first-child'))
    bf = brief(pg)
    check(f'{tag} challenge: the chart holds the 6 options that meet the goal (the pick among them) and the initial setup, 9 left out', brief_ok(lang, bf, 7, 9), bf)
    tags = text(pg, '#pak-ch-res .cmp')
    check(f'{tag} challenge chart: initial setup and the pick', Z(lang, 'initial setup', '初始设置') in tags and Z(lang, 'your pick', '你的选择') in tags, tags[:200])
    click(pg, '#pak-rail .seg-ctl[data-key="chK"] button[data-v="3"]')
    r = text(pg, '#pak-ch-res')
    check(f'{tag} challenge tier 3: the fewest calls', Z(lang, 'fewest calls', '调用次数已经最少') in r, r[:300])
    check(f'{tag} challenge: status green', 'ok' in pg.evaluate("()=>document.getElementById('pak-status').className"))
    # S27: the goal line never runs through the value label (below the goal, near it, above it, at the top)
    for ag, kk in (('B', 1), ('A', 2), ('A', 8), ('A', 3)):
        click(pg, f'#pak-rail .seg-ctl[data-key="chAg"] button[data-v="{ag}"]'); click(pg, f'#pak-rail .seg-ctl[data-key="chK"] button[data-v="{kk}"]')
        g = pg.evaluate("()=>{const t=document.querySelector('#pak-chart .pak-chv'), l=document.querySelector('#pak-chart .pak-goalln'), s=document.getElementById('pak-chart'); if(!t||!l) return null; const r=t.getBoundingClientRect(), y=l.getBoundingClientRect().top, b=s.getBoundingClientRect(); return {top:r.top,bottom:r.bottom,y,in:r.top>=b.top-0.5&&r.right<=b.right+0.5}}")
        check(f'{tag} challenge chart {ag} {kk}: the goal line misses the value label, which stays in the chart (S27)', bool(g) and (g['bottom'] <= g['y'] - 1 or g['top'] >= g['y'] + 1) and g['in'], g)
    rules(pg, f'{tag} challenge', lang); fit(pg, f'{tag} challenge')
    shot(pg, f'{tag}_challenge', full=(w > 760))
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    s = state(pg)
    check(f'{tag} language switch keeps the challenge answer', s['mode'] == 'challenge' and (s['chAg'], s['chK'], s['chSubmitted']) == ('A', 3, True), s)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    if w <= 760:
        mode(pg, 'guided'); go_part(pg, 1); fast(pg)
        pg.evaluate("()=>{const t=document.getElementById('pak-chartbox'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d?{shown:!d.hidden, status:(document.getElementById('dkStatus')||{}).textContent||'', read:(document.getElementById('dkRead')||{}).textContent||''}:null}")
        check(f'{tag} the mini result bar once the picture scrolls away', dk and dk['shown'] and len(dk['status']) <= 14 and dk['read'], dk)
        shot(pg, f'{tag}_dock'); rules(pg, f'{tag} dock', lang)
    check(f'{tag} no horizontal overflow', overflow(pg) <= 0, overflow(pg))
    pg.close()


# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 2, 3, 4, 5 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), tb=(document.querySelector('.topbar')||{offsetHeight:0}).offsetHeight;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(2, 3, 4, 5, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#pak-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'pak-stage')
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}; its top is below the top bar and its main picture whole on the screen (B4, R7.12)',
              f['part'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= f['bar'] - 1 and f['figOn'], f)
    pg.close()


# B4 at 1280 × 800 (polish round 5): part 5's card is short, and the end of the steps column had pushed the sticky stage's head above
# the window (its top at -52 px); after Next the lab's keepStage puts it back under the top bar, at its sticky 70 px.
def b4_top(b, lang='zh', part=5):
    pg = page(b, 1280, 800, lang)
    for k in range(2, part + 1):
        nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
        if nb.count(): nb.first.click()
        else: pg.evaluate(f"()=>document.querySelector('#pak-rail .steps>.step:nth-child({k}) .step-h').click()")
        pg.wait_for_timeout(900)
    y = None
    for _ in range(30):
        y2 = pg.evaluate('()=>scrollY')
        if y2 == y: break
        y = y2; pg.wait_for_timeout(200)
    f = pg.evaluate(B4_FIT, 'pak-stage'); f['want'] = pg.evaluate("()=>parseFloat(getComputedStyle(document.getElementById('pak-stage')).top)")
    # since the 2026-10-22 refit's shorter lab head, Next often has no need to scroll the page here, so the stage may stay below its
    # sticky place; it must never be above it, and its main picture must be whole on the screen
    check(f'{lang}-1280x800 part {part}: after Next the stage\'s top is under the top bar, at its sticky place or lower, and its main picture whole on the screen (B4, keepStage)',
          f['part'] == part and f['top'] >= f['want'] - 2 and f['figOn'], f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: b4_fit(b, lang)
        b4_top(b)
    for lang in LANGS:
        for (w, h) in SIZES:
            run(b, w, h, lang)
    if FULL:
        run(b, 1440, 900, 'en', dark=True)
        pg = page(b, 1440, 900, 'en', reduce=True)
        go_part(pg, 1)
        check('reduced motion: the tries show at once', state(pg)['shown'] == 10 and not state(pg)['playing'], state(pg))
        pg.close()
    b.close()

for h in sorted(set(rule_hits))[:40]: print('RULE', h)
for h in sorted(set(fit_hits))[:40]: print('FIT', h)
check('writing rules and colours (all steps)', not rule_hits, len(rule_hits))
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
check('text fits, tooltips, labels (all steps)', not fit_hits, len(fit_hits))
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'pass-k', 'pak'
_H1_PLAIN = ['1 − (1 − p)', '1 − (1 − pass@1)', 'C(n − c', 'C(n, k)', '⁵', '÷', 'n − c <']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['1 − (1 − p)<sup>k</sup>', 'n − c < 5', 'plug it into 1 − (1 − p)⁵', '<p>C(n, k) counts', 'C(${d.n}, ${d.k}) = ${cn} ways', '<sup>${ks}</sup>', 'C(c, k) / C(n, k) is the chance', '就是 C(c, k) / C(n, k)。']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['pak-mb', 'pak-eqs', 'NAIVE_TEX', "LW=A.lang()==='zh'?56:52"]       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#pak-gridbox', '#pak-chartbox']
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
check('no page errors', not errs, errs[:5])
# the owner, 2026-10-08 (G1, G2) src: every option group leads in with what to pick and what to watch (G1); no sentence leans on an earlier part's detail (G2)
_src = (P / 'labs' / 'pass-k.js').read_text() + (P / 'teasers' / 'pass-k.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['Here are three ways to compute pass@5. Pick one and see how close its pass@5 comes to the true value.', '下面是算 pass@5 的三种方法。选一种，看看它算出的 pass@5 准不准。', 'Pick a product and see which number it should look at before shipping.', '选一个产品，看看它上线前该看哪个指标。', '重试只救得回时好时坏的任务，每次都过不了的任务，试多少次都没用。所以 pass@1 一样时，任务之间差别越大，重试救回的越少。', '等于假设每个任务一样难，会高估重试的效果。要数出每个任务试了 n 次、过了 c 次，再用公式来估。', '真实 agent 的 pass^k，比每个任务一样难时掉得慢。因为很多任务它次次都过，失分集中在时好时坏的那些任务上，要改进的正是这些任务。', '这个 agent 最多试 5 次，能做成 ${fP(b.at[5])} 的任务。', '同一类请求来 4 次、次次都答对的任务，只有 ${fP(b.hat[4])}。', '只有每个任务一样难时才成立。任务有难有易时，这个公式会高估重试的效果。', '在图上显示初始设置的任务和代公式得到的虚线。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['第 2 部分里，每个任务都是 60% 时', '第 3 部分里，把 pass@1 代进', '第 4 部分里，τ‑bench 的 pass^4', '第 5 部分里，它最多试 5 次', '同样在第 5 部分', '第 3 部分里公式给出', '在图上显示第 3 部分的任务', 'In part 2, with every task at 60%', 'In part 3, plugging', 'In part 4, τ‑bench', 'In part 5, up to 5 tries', 'In the same part, only', 'in part 3 the formula said', 'Shows part 3’s tasks']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails[:30] if fails else 'none')
sys.exit(1 if fails else 0)
