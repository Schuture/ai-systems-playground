"""Flow test for the agent-capstone lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the grid of runs;
- the prediction gate (R3.8): Next disabled with its reason (nextWhy) until a prediction AND the end of the runs; before the prediction
  the runs do not start and no result shows (text, tooltips, the header's popovers); “Saved” after a pick; then the 150 runs auto-play
  (R3.9) with pause, one round, to the end, three speeds (fast, the default for a whole evaluation, really is fast) and a scrubber; the
  stage keeps its height while they play (R7.9); at the end the result, the verdict with the right answer, and why (R3.10); the play
  button then reads Replay and replays from the start (R3.20);
- replays (R3.20): a part whose setup is already on screen (the next part starts where the last one ended; fewer runs per task; the
  language switch; Reset or a challenge choice that changes nothing) shows the finished result at once, and its gate counts as passed;
  a setup that differs replays from the first cell drawn differently (5 runs: only the new rows); a setup watched to the end before
  (an option chosen again, Back to part 2, leaving the challenge) shows its end at once;
- parts 2 to 5: Next waits for another option AND the end of its runs (with its reason); the result names the change; an option's
  result waits for its runs; Settings shows only the part's own controls (R4.2); ▲▼ only after the learner's own change (R6.6);
  part 5: 1, 3 and 5 runs per task, the initial setup drawn beside; part 6: three points with their buttons, three misconceptions,
  “See the scenarios”, the next-lab footer;
- Scenarios: four, single choice, the pressed state follows the settings (R1.8); the URL carries the setup and a fresh page restores
  it; a change under Settings does not move the page (R3.2);
- the challenge: only the five choices, Settings locked; live progress; the three tiers (R4.4), tier 2 saying in its first line that a
  cheaper combination passes; “Show the answer”; the brief chart (the owner's review of batch 8): only the 2 passing combinations
  (the cheapest tagged), the initial setup and the pick, both tagged, no other row (the combinations tried on the way are gone), a note
  in the lab's words that counts the combinations left out (105, or 104 when the pick is another combination that misses), a title
  without “the combinations you tried”; the capability card; the language switch keeps the answer;
- the mini result bar (dock) at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip, Chinese spacing and punctuation, and colours only for their data (R5.6).
Screenshots: shots/agent-capstone/.
Run: python3 tests/agent-capstone_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-capstone'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-capstone.html', '--only', 'agent-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'flow.html')
SH = P / 'shots' / 'agent-capstone'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='agent-capstone'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__acp.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('acp-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>[...r.querySelectorAll('.cmp-tag')].some(t=>/initial setup|初始设置/.test(t.textContent));
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!r.querySelector('.cmp-tag,.cmp-you')).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 2 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} combinations miss at least one condition', f'其余 {hidden} 种组合至少差一条') in r['note']
            and Z(lang, 'you tried', '你试过的') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#acp-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#acp-rail .nx-why')].filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent).join(' | ')")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__acp||!window.__acp.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=200):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


def end(pg): click(pg, '#acp-player [data-act="end"]', 150)
def go_part(pg, i): click(pg, f'#acp-rail [data-go="{i}"]', 250)
def mode(pg, m): click(pg, f'#acp-app .modes button[data-mode="{m}"]', 350)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def click_now(pg, sel):
    """click and read the player in the same task, before any animation frame: did a replay start, and from which cell (R3.20)"""
    r = pg.evaluate("""s=>{document.querySelector(s).click(); const st=window.__acp.state();
      return {pv:st.pv, playing:st.playing, K:st.K, next:[...document.querySelectorAll('#acp-rail [data-nav="next"]')].map(b=>b.disabled),
        lever:[...document.querySelectorAll('#acp-rail [data-lever] .lv-r')].map(e=>e.textContent),
        why:[...document.querySelectorAll('#acp-rail .nx-why')].filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent).join(' | ')}}""", sel)
    pg.wait_for_timeout(200); settle(pg)
    return r


# the first cell (in play order: run 1 of every task, then run 2, ...) drawn differently under two harnesses of the support suite
FIRST_DIFF = """([a,b,K])=>{const A=window.__acp, n=A.tasks('shop').length, cls=r=>r.res==='ok'?1:(r.res==='unsafe'?2:0);
  for(let c=0;c<n*K;c++){const i=c%n, k=Math.floor(c/n); if(cls(A.run('shop',i,k,a))!==cls(A.run('shop',i,k,b))) return c;} return n*K;}"""
INIT = {'model': 'cheap', 'R': 0, 'ck': 0, 'ctx': 'all', 'gate': 'none', 'K': 3}


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#acp-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")
def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#acp-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
def live_ctl(pg): return pg.evaluate("()=>['model','R','ck','ctx','gate','K','as','seed','reset'].filter(k=>{const e=document.getElementById('acp-w-'+k); return e&&e.getClientRects().length>0})")
def lever_res(pg): return pg.evaluate("()=>[...document.querySelectorAll('#acp-rail [data-lever] .lv-r')].map(e=>e.textContent)")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness (R5.7)']];
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
BIGNUM = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #acp-mathbody')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|acp-miss|acp-cz-u|acp-sw-un|cmp-|ex-res|wrong/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#acp-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#acp-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#acp-app .labhead','#acp-rail','#acp-stage','#acp-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#acp-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


STRIP = """()=>{const svg=document.querySelector('#acp-score svg'); if(!svg) return ['no strip'];
 const out=[], T=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent,b:t.getBBox()}));
 for(const l of svg.querySelectorAll('line')){const x1=+l.getAttribute('x1'), x2=+l.getAttribute('x2'), y1=+l.getAttribute('y1'), y2=+l.getAttribute('y2');
   for(const {t,b} of T){const lx=Math.min(x1,x2), hx=Math.max(x1,x2), ly=Math.min(y1,y2), hy=Math.max(y1,y2);
     if(hx>=b.x&&lx<=b.x+b.width&&hy>=b.y&&ly<=b.y+b.height) out.push(t+' × '+(l.getAttribute('class')||''));}}
 return out;}"""


CMP = """()=>{const out=[]; document.querySelectorAll('#acp-ch-res .cmp-row').forEach(r=>{const g=r.querySelector('.cmp-track .g'), v=r.querySelector('.cmp-val'); if(!g||!v) return;
  const a=g.getBoundingClientRect(), b=v.getBoundingClientRect(), cs=getComputedStyle(v);
  if(a.right<b.left||a.left>b.right) return;
  const covered=b.top<=a.top+0.5&&b.bottom>=a.bottom-0.5&&cs.backgroundColor!=='rgba(0, 0, 0, 0)'&&(+cs.zIndex||0)>(+getComputedStyle(g).zIndex||0);
  if(!covered) out.push(v.textContent);}); return out;}"""


def strip_ok(pg, where):
    hits = pg.evaluate(STRIP)
    check(f'{where}: no line of the score strip crosses a label (S9, R5.10)', not hits, hits[:4])


def heights(pg): return pg.evaluate("()=>[document.querySelector('#acp-stage').offsetHeight,document.querySelector('#acp-graph').offsetHeight,document.querySelector('#acp-tally').offsetHeight,document.querySelector('#acp-score').offsetHeight]")


def run(b, w, h, lang):
    tag = f'{w}-{lang}'
    pg = page(b, w, h, lang)
    ov = lambda where: check(f'{tag} no horizontal overflow: {where}', overflow(pg) <= 0, overflow(pg))
    st = state(pg)
    check(f'{tag} opens on part 1, no prediction, not playing', st['step'] == 0 and st['pred'] is None and not st['playing'] and st['pv'] == 0, st)
    ov('first screen')
    fv = pg.evaluate("()=>{const g=document.querySelector('#acp-grid').getBoundingClientRect(), m=document.querySelector('#acp-metrics').getBoundingClientRect(); return {top:Math.round(g.top), grid:g.width>0&&g.top>=0&&g.bottom<=innerHeight, below:m.top>=g.bottom}}")
    check(f'{tag} R7.12: the grid of runs is whole on the first screen, and the metric cards sit below it', fv['grid'] and fv['below'], fv)
    if w < 700:
        fs = pg.evaluate("()=>{const nx=document.querySelector('#acp-rail .pbar [data-nav=\"next\"]'), g=document.querySelector('#acp-grid'); const a=nx?nx.getBoundingClientRect():null, c=g.getBoundingClientRect(); return {next:!!a&&a.bottom<=innerHeight&&a.width>0, grid:c.top<innerHeight-40}}")
        check(f'{tag} phone first screen: the parts bar with Next and the grid (R7.3)', fs['next'] and fs['grid'], fs)
    shot(pg, f'{tag}-01-first')
    # ---- part 1: the gate ----
    check(f'{tag} part 1: Next disabled before a prediction', all(nexts(pg)), nexts(pg))
    check(f'{tag} part 1: why Next is disabled', Z(lang, 'Pick an answer', '选一个答案') in why(pg), why(pg))
    ba = before_answer(pg)
    for leak in ('62.7', '78.7', '75.3', 'right answer', '正确答案', 'All four met', '四条都满足'):
        check(f'{tag} part 1: nothing gives the answer before the prediction ({leak})', leak not in ba)
    check(f'{tag} part 1: the play button is off before the prediction', pg.evaluate("()=>document.querySelector('#acp-play').disabled"))
    check(f'{tag} R3.9: the grid plays a whole evaluation, so its default speed is fast', st['speed'] == 'f' and pg.evaluate("()=>document.querySelector('#acp-player .acp-speed button[aria-pressed=\"true\"]').dataset.v") == 'f', st['speed'])
    rules(pg, f'{tag} part 1 before', lang)
    # medium for the checks made while the runs play (fast, the default, ends in 1.5 s); fast is timed below
    click(pg, '#acp-player .acp-speed button[data-v="m"]', 20)
    click(pg, '#acp-rail [data-pred="d"]', 150)
    st = state(pg)
    check(f'{tag} part 1: a pick starts the runs (auto-play)', st['pred'] == 'd' and st['playing'], st)
    check(f'{tag} part 1: “Saved” after the pick', Z(lang, 'Saved', '已记下') in text(pg, '#acp-rail .step.current'))
    check(f'{tag} part 1: Next still waits for the runs, with its reason', all(nexts(pg)) and Z(lang, 'Wait for the 150 runs', '等 150 次运行') in why(pg), (nexts(pg), why(pg)))
    check(f'{tag} part 1: no result while the runs play', '62.7' not in before_answer(pg))
    hs = []
    for _ in range(5): hs.append(heights(pg)); pg.wait_for_timeout(200)
    check(f'{tag} R7.9: the stage keeps its height while the runs play', all(x == hs[0] for x in hs), hs)
    shot(pg, f'{tag}-02-running')
    # speeds: fast, the default, is fast (150 runs in 1.5 s); medium about 20 s for the whole evaluation (R3.9)
    p0 = state(pg)['pv']; pg.wait_for_timeout(1000); p1 = state(pg)['pv']
    check(f'{tag} medium: about 8 runs a second, 150 in about 20 s', 4 <= p1 - p0 <= 12, (p0, p1))
    click(pg, '#acp-play', 50); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} pause stops the replay', not s1['playing'] and s1['pv'] == s2['pv'], (s1['pv'], s2['pv']))
    click(pg, '#acp-player [data-act="stepf"]', 50)
    check(f'{tag} one round forward: every task gets its next run', state(pg)['pv'] == (s2['pv'] // 50 + 1) * 50, (s2['pv'], state(pg)['pv']))
    click(pg, '#acp-player .acp-speed button[data-v="f"]', 20)
    click(pg, '#acp-play', 20); t0 = time.time()
    pg.wait_for_function("()=>window.__acp.state().pv>=150", timeout=6000); dt = time.time() - t0
    check(f'{tag} fast is fast: the rest of the runs in under 1.5 s', dt < 1.5, dt)
    st = state(pg)
    check(f'{tag} part 1: the end shows all 150 runs and enables Next', st['runsSeen'] and st['pv'] == 150 and not any(nexts(pg)), (st, nexts(pg)))
    o = obs(pg)
    check(f'{tag} part 1: the result, the verdict with the right answer, and why', '62.7%' in o and Z(lang, 'The right answer is “Handle errors', '正确答案是“处理出错') in o and Z(lang, 'Why', '原因') in o and '78.7%' in o, o)
    check(f'{tag} part 1: the metrics show after the runs', text(pg, '#acp-metrics').count('%') >= 3)
    strip_ok(pg, f'{tag} part 1')
    check(f'{tag} part 1: the note no longer offers a change once the runs are done', Z(lang, 'You can still change it', '还可以改') not in text(pg, '#acp-rail .step.current'))
    check(f'{tag} part 1: Settings shows only the assumed values', live_ctl(pg) == ['as'], live_ctl(pg))
    rules(pg, f'{tag} part 1 result', lang); ov('part 1 result')
    shot(pg, f'{tag}-03-p1-result', full=(w > 700))
    # R3.20: the replay control. After the end the play button reads Replay, its tooltip says it starts again, and it does
    pb = pg.evaluate("()=>{const b=document.querySelector('#acp-play'); return [b.textContent, b.dataset.tipx||'']}")
    check(f'{tag} R3.20: after the end the play button reads Replay, and its tooltip says it replays from the start', pb[0] == Z(lang, '▶ Replay', '▶ 重放') and Z(lang, 'from the start', '从头重放') in pb[1], pb)
    r = click_now(pg, '#acp-play'); t0 = time.time()
    check(f'{tag} R3.20: Replay plays the runs again from the start', r['playing'] and r['pv'] == 0, r)
    pg.wait_for_function("()=>window.__acp.state().pv>=150", timeout=12000); dt = time.time() - t0
    check(f'{tag} R3.9: at fast, the whole evaluation of 150 runs ends in about ten seconds or less ({dt:.1f} s)', dt <= 10, dt)
    settle(pg)
    # ---- part 2 ----
    r = click_now(pg, '#acp-rail [data-go="1"]')
    check(f'{tag} R3.20 part 2: it starts with part 1’s setup, so the finished runs show at once, no replay', r['pv'] == 150 and not r['playing'], r)
    check(f'{tag} part 2: Next disabled until another option, with its reason', all(nexts(pg)) and Z(lang, 'Pick a way to handle errors', '选一种处理错误的办法') in why(pg), why(pg))
    end(pg)
    check(f'{tag} part 2: no ▲▼ before the learner acts', not badges(pg), badges(pg))
    check(f'{tag} part 2: Settings shows retries and the check', live_ctl(pg) == ['R', 'ck', 'as', 'reset'], live_ctl(pg))
    fd = pg.evaluate(FIRST_DIFF, [INIT, dict(INIT, R=2), 3])
    r = click_now(pg, '#acp-rail [data-lever="r2"]')
    check(f'{tag} part 2: the option’s result waits for its runs', r['lever'][1] == '' and all(r['next']) and Z(lang, 'Wait for the runs', '等运行跑完') in r['why'], r)
    check(f'{tag} R3.20 part 2: a new option replays, from the first cell drawn differently (cell {fd})', r['playing'] and r['pv'] == fd and 0 <= fd < 150, (r, fd))
    end(pg)
    check(f'{tag} part 2: 2 retries give 72.0%', '72.0%' in obs(pg) and lever_res(pg)[1] == '72.0%' and not any(nexts(pg)), obs(pg))
    check(f'{tag} part 2: ▲▼ after the learner acts', bool(badges(pg)), badges(pg))
    click(pg, '#acp-rail [data-lever="r2c"]'); end(pg)
    check(f'{tag} part 2: retries and a check give 78.7%', '78.7%' in obs(pg), obs(pg))
    check(f'{tag} part 2: single choice', pg.evaluate("()=>[...document.querySelectorAll('#acp-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1)
    check(f'{tag} part 2: links agent-reliability', pg.evaluate("()=>!!document.querySelector('#acp-rail .step.current [data-lab=\"agent-reliability\"], #acp-rail .step.current a[href*=\"agent-reliability\"]')"))
    rules(pg, f'{tag} part 2', lang); ov('part 2'); shot(pg, f'{tag}-04-p2', full=(w > 700))
    # R3.20 (2026-10-04 addition): an option already watched to the end is not new: it shows its end at once
    for lv in ('r2', 'init', 'r2c'):
        r = click_now(pg, f'#acp-rail [data-lever="{lv}"]')
        check(f'{tag} R3.20 part 2: “{lv}” was watched to the end before, so it shows its end at once, no replay', r['pv'] == 150 and not r['playing'], r)
    # ---- part 3 ----
    r = click_now(pg, '#acp-rail [data-go="2"]')
    check(f'{tag} R3.20 part 3: it starts with part 2’s last option, already on screen: no replay', r['pv'] == 150 and not r['playing'], r)
    end(pg)
    check(f'{tag} part 3: Next disabled until another option', all(nexts(pg)), nexts(pg))
    click(pg, '#acp-rail [data-lever="last"]'); end(pg)
    check(f'{tag} part 3: the last 8 steps fall to 62.0%', '62.0%' in obs(pg) and not any(nexts(pg)), obs(pg))
    click(pg, '#acp-rail [data-lever="notes"]'); end(pg)
    check(f'{tag} part 3: trim + notes give 84.7% at 13.4 cents', '84.7%' in obs(pg) and Z(lang, '13.4 cents', '13.4 美分') in obs(pg), obs(pg))
    check(f'{tag} part 3: Settings shows the context only', live_ctl(pg) == ['ctx', 'as', 'reset'], live_ctl(pg))
    check(f'{tag} part 3: links context-eng', pg.evaluate("()=>!!document.querySelector('#acp-rail .step.current [data-lab=\"context-eng\"], #acp-rail .step.current a[href*=\"context-eng\"]')"))
    rules(pg, f'{tag} part 3', lang); ov('part 3'); shot(pg, f'{tag}-05-p3', full=(w > 700))
    # R3.20 (2026-10-04 addition): back to part 2, whose initial setup was watched to the end in part 1: its end at once
    r = click_now(pg, '#acp-rail [data-nav="back"]')
    check(f'{tag} R3.20 back to part 2: its setup was watched to the end before, so it shows its end at once, no replay', r['pv'] == 150 and not r['playing'] and state(pg)['step'] == 1, r)
    # ---- part 4 ----
    r = click_now(pg, '#acp-rail [data-go="3"]')
    check(f'{tag} R3.20 part 4: it starts with part 3’s last option, watched to the end there: no replay', r['pv'] == 150 and not r['playing'], r)
    end(pg)
    click(pg, '#acp-rail [data-lever="detect"]'); end(pg)
    check(f'{tag} part 4: the detector leaves 3 unsafe actions', Z(lang, 'from 8 to 3', '从 8 次降到 3 次') in obs(pg) and not any(nexts(pg)), obs(pg))
    click(pg, '#acp-rail [data-lever="confirm"]'); end(pg)
    check(f'{tag} part 4: a person confirms: 0 unsafe, 90.0%', '90.0%' in obs(pg) and Z(lang, 'from 8 to 0', '从 8 次降到 0') in obs(pg), obs(pg))
    check(f'{tag} part 4: Settings shows the gate only', live_ctl(pg) == ['gate', 'as', 'reset'], live_ctl(pg))
    check(f'{tag} part 4: the status says all four met', Z(lang, 'All four met', '四条都满足') in text(pg, '#acp-status'), text(pg, '#acp-status'))
    rules(pg, f'{tag} part 4', lang); ov('part 4'); shot(pg, f'{tag}-06-p4', full=(w > 700))
    # ---- part 5 ----
    r = click_now(pg, '#acp-rail [data-go="4"]')
    check(f'{tag} R3.20 part 5: it starts with part 4’s last option, already on screen: no replay', r['pv'] == 150 and not r['playing'], r)
    end(pg)
    check(f'{tag} part 5: the initial setup drawn beside the new harness', pg.evaluate("()=>!!document.querySelector('#acp-score .acp-b0-pt')"))
    r = click_now(pg, '#acp-rail .seg-ctl[data-key="K"] button[data-v="1"]')
    check(f'{tag} R3.20 part 5: 1 run per task is the first row already on screen: no replay, and Next is enabled at once (the gate counts as passed)', r['pv'] == 50 and not r['playing'] and not any(r['next']), r)
    end(pg)
    check(f'{tag} part 5: 1 run: 86.0%, lower end 76.3%', '86.0%' in obs(pg) and '76.3%' in obs(pg) and not any(nexts(pg)), obs(pg))
    check(f'{tag} part 5: 1 run: pass^3 needs 3 runs', Z(lang, 'needs 3 runs', '要跑 3 次') in text(pg, '#acp-metrics'), text(pg, '#acp-metrics'))
    r = click_now(pg, '#acp-rail .seg-ctl[data-key="K"] button[data-v="5"]')
    check(f'{tag} R3.20 part 5: 5 runs per task replay only the runs not on screen yet (from cell 50)', r['playing'] and r['pv'] == 50 and r['K'] == 5, r)
    end(pg)
    check(f'{tag} part 5: 5 runs: 86.7% to 94.9%, about a fifth narrower', '86.7%' in obs(pg) and '94.9%' in obs(pg) and Z(lang, 'about a fifth', '约五分之一') in obs(pg), obs(pg))
    strip_ok(pg, f'{tag} part 5')
    check(f'{tag} part 5: the paired gain with its interval', Z(lang, 'task by task', '逐个任务') in obs(pg) and '+30.0' in obs(pg), obs(pg))
    check(f'{tag} part 5: links eval-ci and pass-k', pg.evaluate("()=>['eval-ci','pass-k'].every(id=>document.querySelector(`#acp-rail .step.current [data-lab=\"${id}\"], #acp-rail .step.current a[href*=\"${id}\"]`))"))
    check(f'{tag} part 5: Settings shows the runs per task only', live_ctl(pg) == ['K', 'as', 'reset'], live_ctl(pg))
    rules(pg, f'{tag} part 5', lang); ov('part 5'); shot(pg, f'{tag}-07-p5', full=(w > 700))
    # ---- part 6 ----
    r = click_now(pg, '#acp-rail [data-go="5"]')
    check(f'{tag} R3.20 part 6: 3 runs per task are the first rows already on screen: no replay', r['pv'] == 150 and not r['playing'] and r['K'] == 3, r)
    cur = text(pg, '#acp-rail .step.current')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', pg.evaluate("()=>document.querySelectorAll('#acp-rail .acp-rem > li').length") == 3 and pg.evaluate("()=>document.querySelectorAll('#acp-rail .acp-mis > li').length") == 3 and pg.evaluate("()=>document.querySelectorAll('#acp-rail [data-rem]').length") == 3 and pg.evaluate("()=>!!document.querySelector('#acp-rail [data-mode-go=\"sandbox\"]')"), cur[:200])
    check(f'{tag} part 6: the next-lab footer', pg.evaluate("()=>!!document.querySelector('#acp-app .next-lab')"))
    click(pg, '#acp-rail [data-rem="det"]'); end(pg)
    st = state(pg)
    check(f'{tag} part 6: “Show the detector” sets its example', st['gate'] == 'detect' and st['remView'] == 'det' and st['ctx'] == 'notes', st)
    check(f'{tag} part 6: its button is pressed', pg.evaluate("()=>document.querySelector('#acp-rail [data-rem=\"det\"]').getAttribute('aria-pressed')") == 'true')
    rules(pg, f'{tag} part 6', lang); ov('part 6'); shot(pg, f'{tag}-08-p6', full=(w > 700))
    # ---- scenarios ----
    mode(pg, 'sandbox')
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#acp-rail')
    _lead = Z(lang, 'Pick a scenario and see which launch conditions it misses.', '选一个场景，看看它达不到哪几条上线标准。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#acp-rail [data-preset]').length") == 4)
    click(pg, '#acp-rail [data-preset="0"]'); end(pg)
    pr = pg.evaluate("()=>[...document.querySelectorAll('#acp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: single choice, pressed follows the settings', pr == ['true', 'false', 'false', 'false'], pr)
    check(f'{tag} scenarios: the stronger model alone, 75.3%', '75.3%' in text(pg, '#acp-insight'), text(pg, '#acp-insight'))
    check(f'{tag} scenarios: every Settings control is live', live_ctl(pg) == ['model', 'R', 'ck', 'ctx', 'gate', 'K', 'as', 'seed', 'reset'], live_ctl(pg))
    click(pg, '#acp-rail [data-preset="3"]'); end(pg)
    check(f'{tag} scenarios: no check saves money, silent errors stay', '81.3%' in text(pg, '#acp-insight'), text(pg, '#acp-insight'))
    # R3.2: a change under Settings does not move the page
    pg.evaluate("()=>document.querySelector('#acp-ctl-K').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    y0 = pg.evaluate("()=>document.querySelector('#acp-ctl-K').getBoundingClientRect().top")
    click(pg, '#acp-controls .seg-ctl[data-key="K"] button[data-v="5"]', 250)
    y1 = pg.evaluate("()=>document.querySelector('#acp-ctl-K').getBoundingClientRect().top")
    check(f'{tag} R3.2: a Settings change keeps the control in place', abs(y1 - y0) < 2, (y0, y1))
    pr = pg.evaluate("()=>[...document.querySelectorAll('#acp-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change under Settings unpresses the scenario', pr == ['false'] * 4, pr)
    click(pg, '#acp-app [data-act="reseed"]', 250); end(pg)
    hsh = pg.evaluate("()=>location.hash")
    check(f'{tag} the URL carries the setup', hsh.endswith('cheap_2_0_notes_confirm_5_2'), hsh)
    rules(pg, f'{tag} scenarios', lang); ov('scenarios'); shot(pg, f'{tag}-09-scenarios', full=(w > 700))
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    s2 = state(pg2)
    check(f'{tag} the URL restores the setup', s2['mode'] == 'sandbox' and s2['ck'] == 0 and s2['K'] == 5 and s2['draw'] == 2 and s2['ctx'] == 'notes', (hsh, s2))
    pg2.close()
    click(pg, '#acp-app [data-act="reset"]', 250); end(pg)
    r = click_now(pg, '#acp-app [data-act="reset"]')
    check(f'{tag} R3.20 scenarios: Reset when nothing would change: no replay', r['pv'] == 150 and not r['playing'], r)
    # ---- challenge ----
    mode(pg, 'challenge'); end(pg)
    check(f'{tag} challenge: Settings locked with a note', live_ctl(pg) == [] and text(pg, '#acp-lock') != '', live_ctl(pg))
    g = text(pg, '#acp-ch-live')
    check(f'{tag} challenge: live progress (38.7%, 29.0 cents)', '38.7%' in g and '29.0' in g, g)
    r = click_now(pg, '#acp-rail .seg-ctl[data-key="chR"] button[data-v="0"]')
    check(f'{tag} R3.20 challenge: picking the choice already set: no replay', r['pv'] == 150 and not r['playing'], r)
    click(pg, '#acp-rail [data-act="submit"]', 250)
    res = text(pg, '#acp-ch-res')
    check(f'{tag} challenge tier 1: not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res and pg.evaluate("()=>!!document.querySelector('#acp-ch-res [data-act=\"reveal\"]')")
          and text(pg, '#acp-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res)
    click(pg, '#acp-ch-res [data-act="reveal"]', 200)
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 3 rows, the 2 passing combinations and the initial setup (the pick), 105 left out', brief_ok(lang, bf, 3, 105), bf)
    click(pg, '#acp-rail .seg-ctl[data-key="chR"] button[data-v="2"]', 120)
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (2 retries, the rest as at the start) adds its own row, 4 rows, 104 left out; the combination tried before is gone', brief_ok(lang, bf, 4, 104), bf)
    for k, v in (('chX', 'notes'), ('chG', 'confirm')): click(pg, f'#acp-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 120)
    end(pg)
    res = text(pg, '#acp-ch-res')
    check(f'{tag} challenge tier 2: met, but a cheaper one passes (10.4 cents)', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'cheaper combination', '更省钱的组合') in res and '10.4' in res, res)
    check(f'{tag} challenge tier 2: the first line says that a cheaper combination passes', Z(lang, 'But a cheaper combination passes too', '不过还有更省钱的组合也能达标') in text(pg, '#acp-ch-res > p:first-child'), text(pg, '#acp-ch-res > p:first-child'))
    bf = brief(pg)
    check(f'{tag} challenge tier 2: the chart holds the 2 passing combinations (the pick among them) and the initial setup, 105 left out', brief_ok(lang, bf, 3, 105), bf)
    click(pg, '#acp-rail .seg-ctl[data-key="chR"] button[data-v="1"]', 150); end(pg)
    res = text(pg, '#acp-ch-res')
    check(f'{tag} challenge tier 3: the minimal one', Z(lang, 'no passing combination costs less', '没有哪个达标的组合更省钱') in res and '91.3%' in res, res)
    strip_ok(pg, f'{tag} challenge')
    rows = pg.evaluate("()=>document.querySelectorAll('#acp-ch-res .cmp-row').length")
    bf = brief(pg)
    check(f'{tag} challenge: the chart holds only the 2 passing combinations (the pick among them) and the initial setup, 105 left out', rows == 3 and brief_ok(lang, bf, 3, 105), (rows, bf))
    check(f'{tag} challenge: the guided parts’ harness is named in words, its numbers in a hover', Z(lang, 'goes over the budget', '超出了预算') in res
          and pg.evaluate("()=>[...document.querySelectorAll('#acp-ch-res .def')].some(e=>/%/.test(e.getAttribute('data-tipx'))&&/(cents|美分|¢)/.test(e.getAttribute('data-tipx')))"), res[-500:])
    hits = pg.evaluate(CMP)
    check(f'{tag} challenge: the goal line never runs through a value label of the chart (S9, R5.10)', not hits, hits[:4])
    check(f'{tag} challenge: the chart tags the initial setup, the pick and the cheapest', all(x in res for x in Z(lang, ('initial setup', 'your pick', 'cheapest that passes'), ('初始设置', '你的选择', '最省钱的达标组合'))), res[:300])
    check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#acp-rail [data-exits]')"))
    rules(pg, f'{tag} challenge', lang); ov('challenge'); shot(pg, f'{tag}-10-challenge', full=(w > 700))
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    s3 = state(pg)
    check(f'{tag} R3.20: the language switch keeps the finished runs, no replay', s3['pv'] == 150 and not s3['playing'], (s3['pv'], s3['playing']))
    check(f'{tag} the language switch keeps the answer', s3['mode'] == 'challenge' and s3['chO'] == {'model': 'cheap', 'R': 1, 'ck': 0, 'ctx': 'notes', 'gate': 'confirm'} and s3['chSubmitted'], s3)
    # ---- the dock at 390 ----
    if w < 700:
        end(pg)
        pg.evaluate("()=>{const t=document.querySelector('#acp-rail [data-exits]')||document.querySelector('#acp-ch-res'); scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getClientRects().length?d.innerText:''}")
        check(f'{tag} the dock shows once the chart scrolls away', bool(dk) and '%' in dk, dk)
        rules(pg, f'{tag} dock', lang)
        shot(pg, f'{tag}-11-dock')
    # R3.20 (2026-10-04 addition): leaving the challenge brings back the scenarios' setup, watched to the end before: its end at once
    r = click_now(pg, '#acp-app .modes button[data-mode="sandbox"]:not([data-flow=free])')
    check(f'{tag} R3.20: leaving the challenge shows the setup watched before at once, no replay', r['pv'] == 150 and not r['playing'] and state(pg)['mode'] == 'sandbox', r)
    pg.close()


# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 4, 5 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), tb=(document.querySelector('.topbar')||{offsetHeight:0}).offsetHeight;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(4, 5, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#acp-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'acp-stage')
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}; its top is below the top bar and its main picture whole on the screen (B4, R7.12)',
              f['part'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= f['bar'] - 1 and f['figOn'], f)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: b4_fit(b, lang)
    for lang in LANGS:
        for (w, h) in SIZES:
            t0 = time.time(); run(b, w, h, lang); print(f'{w} {lang}: {time.time() - t0:.0f}s')
    b.close()

for x in sorted(set(rule_hits)): print('RULE', x)
check('writing and colour rules', not rule_hits, len(set(rule_hits)))
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'agent-capstone', 'acp'
_H1_PLAIN = ['√', '÷']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['÷ √${n}', 'over √n)', '除以 √n）', '1.96 × ${fixd(d.sd,4)} ÷ √10']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['acp-mb', 'acp-eqs', 'tipMath', 'text-anchor="${an}"']       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#acp-grid-cap', '#acp-score-cap', '#acp-cause-cap']
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
_src = (P / 'labs' / 'agent-capstone.js').read_text() + (P / 'teasers' / 'agent-capstone.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['<p>选一种处理错误的办法，看看成功率能涨到多少。</p>', '<p>选一种管上下文的办法，看看成功率怎么变。</p>', '<p>选一种把关的办法，看看不安全的操作还剩几次。</p>', '<p>每个设置选一项，组成一个 harness。达标的 harness 里，花钱最少的最好。</p>', '全部保留（重试和检查已加上）', '不把关（重试、检查、管上下文都已加上）', '在图里看初始设置', '对准失败主因的改动，收效最大。失败的运行大多停在报出来的错上时，重试加检查比换一个贵得多的强模型更管用。', '可每次检查都要多调用一次模型，花费会涨，要靠管好上下文降回来。把关管安全。检测模型只能让不安全的操作变少，只有人确认才能降到 0。', '每个任务只跑 1 次，区间太宽，平均值过了线，下沿也可能掉到线下面。', '更强的模型让每一步更稳，可照样撑满上下文，照样听信注入。这两件事只有 harness 能管。', '从初始设置出发，重试加检查让成功率', '用改好的 harness，每个任务从跑 3 次加到 5 次', '所以管上下文的结论不变', '比较起来噪声比这个 Lab 里的更大', '用每个任务成功的次数，算出成功率、区间下沿和 pass^3。', '固定：重试 2 次加检查，不把关。', '固定：重试 2 次、检查，以及裁剪 + 摘要 + 笔记。', '固定：改好的 harness，也就是重试、检查、裁剪 + 摘要 + 笔记，再加上人确认。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['第 2 部分的 harness', '第 3 部分的 harness', '在图里看第 1 部分', '第 1 部分里，失败的运行', '第 4 部分里，检测模型', '第 5 部分里，每个任务只跑 1 次', '第 1 部分里，更强的模型', '第 1 部分里，重试加检查', '第 5 部分里，每个任务从跑 3 次', '所以第 3 部分的结论', '噪声比第 5 部分', '用第 5 部分的成功次数', '第 2 部分的重试 2 次', '第 2、3 部分的重试', '第 2 到 4 部分改好的', 'part 2’s harness', 'part 3’s harness', 'Show part 1’s', 'In part 1 most failed', 'in part 4 the detector', 'In part 5, one run', 'In part 1 the stronger', 'In part 1, retries', 'In part 5, going', 'part 3’s conclusion', 'the one in part 5', 'part 5’s counts', 'part 2’s 2 retries', 'parts 2 and 3’s', 'parts 2 to 4']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
