"""Flow test for the agent-reliability lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the grid of tasks;
- the prediction gate (R3.8): Next disabled with its reason (nextWhy) until a prediction AND the end of the runs; before the prediction
  the runs do not start and no result shows anywhere (text, tooltips, the header's popovers); “Saved” after a pick; then the 30 tasks
  of the illustration auto-play at the default speed, fast (R3.9), with pause, one step, to the end, three speeds and a scrubber; the stage keeps its height while
  they run (R7.9); at the end the result, the verdict with the right answer, and why (R3.10);
- parts 2 to 5: Next waits for the part's action; the result names the change; Settings shows only the part's own controls (R4.2);
  ▲▼ badges only after the learner's own change (R6.6); part 5 links the pass-k lab; part 6: three points with their “Show in the
  chart” buttons, three misconceptions, “See the scenarios”, the next-lab footer;
- Scenarios: four, single choice, the pressed state follows the settings (R1.8); the URL carries the setup and a fresh page restores it;
  a change under Settings does not move the page (R3.2);
- the challenge: only the four choices, Settings locked; live progress; the three feedback tiers (R4.4), tier 2 saying in its first
  line that a cheaper combination passes; “Show the answer”; the brief chart (the owner's review of batch 8): only the 3 passing
  combinations, the initial setup and the pick, both tagged, no other row, a note in the lab's words that counts the combinations left
  out (28, or 27 when the pick is another combination that misses), a title without “for each combination”; the failing alternatives
  the text names, in words with their numbers in a hover; the capability card; the language switch keeps the answer;
- replays (R3.20, owner check b7): the next part does not replay the run just watched; a new picture replays (only the changed rows
  when the steps stay the same); the same picture is not replayed; a run already watched to the end is not replayed when shown again
  (Back to part 2, a scenario watched before, leaving the challenge); the play button replays the whole run once it has finished; a
  language switch does not replay;
- the mini result bar (dock) at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip, Chinese spacing and punctuation, and colours only for their data (R5.6).
Screenshots: shots/agent-reliability/.
Run: python3 tests/agent-reliability_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time, re
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-reliability'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-reliability.html', '--only', 'agent-reliability'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-reliability.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'flow.html')
SH = P / 'shots' / 'agent-reliability'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='agent-reliability'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__arl.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('arl-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 3 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} combinations miss at least one target', f'其余 {hidden} 种组合至少差一个目标') in r['note']
            and Z(lang, 'for each combination', '按组合列出') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#arl-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#arl-rail .nx-why')].filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent).join(' | ')")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__arl||!window.__arl.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=200):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def go_part(pg, i): click(pg, f'#arl-rail [data-go="{i}"]', 250)
def mode(pg, m): click(pg, f'#arl-app .modes button[data-mode="{m}"]', 350)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#arl-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")
def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#arl-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
def live_ctl(pg): return pg.evaluate("()=>['n','p','r','R','ck','kind','c','pb','seed','reset'].filter(k=>{const e=document.getElementById('arl-w-'+k); return e&&e.getClientRects().length>0})")


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #arl-mathbody')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|arl-over|arl-badm|arl-sw-st|arl-sw-si|cmp-|ex-res|wrong/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#arl-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
# R5.11 (owner check b7): for the first cell of each kind on show, is its top edge drawn red? (the outline of a failure cell runs there)
REDCELLS = """()=>{const s=window.__arl.state(), simp=window.__arl.simple(), host=document.getElementById('arl-grid'), cv=document.getElementById('arl-cv'), g=cv.getContext('2d');
 const W=host.clientWidth, cw=(W-20)/s.n, dpr=cv.width/W, out={};
 const red=(x,y)=>{const d=g.getImageData(Math.round(x*dpr),Math.round(y*dpr),1,1).data; return d[0]-Math.max(d[1],d[2])>60;};
 for(let i=0;i<30;i++){const r=window.__arl.simRun({n:s.n,p:s.p,r:s.r,R:s.R,ck:s.ck,kind:s.kind,c:s.c,pb:s.pb,seed:s.seed},i);
  r.cells.forEach((c,j)=>{if(j>=s.pv) return; const k=simp?(r.out!=='ok'&&j===r.fx?'failed':(j<r.fx||r.out==='ok'?'ok':'')):c.st;
   if(k&&!(k in out)) out[k]=red(j*cw+cw/2-0.5,i*9+0.3);});}
 return out;}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#arl-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#arl-app .labhead','#arl-rail','#arl-stage','#arl-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#arl-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def heights(pg): return pg.evaluate("()=>[document.querySelector('#arl-stage').offsetHeight,document.querySelector('#arl-graph').offsetHeight,document.querySelector('#arl-tally').offsetHeight,document.querySelector('#arl-batch').offsetHeight]")


def run(b, w, h, lang):
    tag = f'{w}-{lang}'
    pg = page(b, w, h, lang)
    ov = lambda where: check(f'{tag} no horizontal overflow: {where}', overflow(pg) <= 0, overflow(pg))
    st = state(pg)
    check(f'{tag} opens on part 1, no prediction, not playing', st['step'] == 0 and st['pred'] is None and not st['playing'] and st['pv'] == 0, st)
    ov('first screen')
    if w < 700:
        fs = pg.evaluate("()=>{const nx=document.querySelector('#arl-rail .pbar [data-nav=\"next\"]'), g=document.querySelector('#arl-grid'); const a=nx?nx.getBoundingClientRect():null, c=g.getBoundingClientRect(); return {next:!!a&&a.bottom<=innerHeight&&a.width>0, grid:c.top<innerHeight-40}}")
        check(f'{tag} phone first screen: the parts bar with Next and the grid (R7.3)', fs['next'] and fs['grid'], fs)
    shot(pg, f'{tag}-01-first')
    # ---- part 1: the gate ----
    check(f'{tag} part 1: Next disabled before a prediction', all(nexts(pg)), nexts(pg))
    check(f'{tag} part 1: why Next is disabled', Z(lang, 'Pick an answer', '选一个答案') in why(pg), why(pg))
    ba = before_answer(pg)
    for leak in ('35.8', '36 in 100', 'a third', '三分之一', '大约 36', 'right answer', '正确答案'):
        check(f'{tag} part 1: nothing gives the answer before the prediction ({leak})', leak not in ba)
    check(f'{tag} part 1: the play button is off before the prediction', pg.evaluate("()=>document.querySelector('#arl-play').disabled"))
    rules(pg, f'{tag} part 1 before', lang)
    click(pg, '#arl-rail [data-pred="b"]', 150)
    st = state(pg)
    check(f'{tag} part 1: a pick starts the runs (auto-play)', st['pred'] == 'b' and st['playing'], st)
    check(f'{tag} part 1: “Saved” after the pick', Z(lang, 'Saved', '已记下') in text(pg, '#arl-rail .step.current'))
    check(f'{tag} part 1: Next still waits for the runs', all(nexts(pg)) and Z(lang, 'Wait for the 30 tasks', '等 30 个任务') in why(pg), (nexts(pg), why(pg)))
    check(f'{tag} R3.9: a whole run defaults to fast', state(pg)['speed'] == 'f', state(pg)['speed'])
    check(f'{tag} the caption says the 30 tasks are an illustration', Z(lang, '30 tasks, as an illustration', '30 个任务，作为示意') in text(pg, '#arl-batch'), text(pg, '#arl-batch'))
    check(f'{tag} part 1: no result while the runs play', '35.8' not in before_answer(pg))
    hs = []
    for _ in range(6): hs.append(heights(pg)); pg.wait_for_timeout(260)
    check(f'{tag} R7.9: the stage keeps its height while the tasks run', all(x == hs[0] for x in hs), hs)
    shot(pg, f'{tag}-02-running')
    # speeds: a whole run of 20 steps takes about 12 s at medium and about 6 s at fast, the default (R3.9)
    click(pg, '#arl-player .arl-speed button[data-v="m"]', 50)
    p0 = state(pg)['pv']; pg.wait_for_timeout(1500); p1 = state(pg)['pv']
    check(f'{tag} medium: about 2.5 columns in 1.5 s (12 s a run)', 1 <= p1 - p0 <= 4, (p0, p1))
    click(pg, '#arl-player .arl-speed button[data-v="f"]', 50)
    p0 = state(pg)['pv']; pg.wait_for_timeout(1500); p1 = state(pg)['pv']
    check(f'{tag} fast: about 5 columns in 1.5 s (6 s a run)', 3 <= p1 - p0 <= 8, (p0, p1))
    click(pg, '#arl-play', 50); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} pause stops the replay', not s1['playing'] and s1['pv'] == s2['pv'], (s1['pv'], s2['pv']))
    click(pg, '#arl-player [data-act="stepf"]', 50)
    check(f'{tag} one step forward', state(pg)['pv'] == s2['pv'] + 1)
    click(pg, '#arl-player [data-act="end"]', 200)
    st = state(pg)
    check(f'{tag} part 1: to the end shows all 30 and enables Next', st['runsSeen'] and st['pv'] == 20 and not any(nexts(pg)), (st, nexts(pg)))
    check(f'{tag} R3.20: at the end the play button replays the run, and its tooltip says so', Z(lang, 'Replay', '重放') in text(pg, '#arl-play') and Z(lang, 'from the start', '从头再放一遍') in pg.evaluate("()=>document.querySelector('#arl-play').dataset.tipx"), text(pg, '#arl-play'))
    o = obs(pg)
    check(f'{tag} part 1: the result, the verdict with the right answer, and why', '35.8%' in o and Z(lang, 'The right answer is “30 to 59”', '正确答案是“30 到 59 个”') in o and Z(lang, 'Why', '原因') in o, o)
    check(f'{tag} part 1: the metrics show after the runs (success and failure only)', text(pg, '#arl-metrics').count('%') == 2, text(pg, '#arl-metrics'))
    # S1: parts 1 and 2 do not use the loud / silent split or the stop before part 3 teaches them
    p1 = text(pg, '#arl-st-sub') + ' ' + text(pg, '#arl-legend') + ' ' + text(pg, '#arl-tally') + ' ' + text(pg, '#arl-metrics')
    for w_ in Z(lang, ('loud', 'silent', 'stopped', 'retries used up', 'wrong at the end'), ('报出来', '悄悄', '停下', '重试用完', '做完但错了')):
        check(f'{tag} part 1: no “{w_}” on the stage before part 3', w_ not in p1, p1)
    check(f'{tag} part 1: the note no longer offers a change once the runs are done', Z(lang, 'You can still change it', '还可以改') not in text(pg, '#arl-rail .step.current'))
    rc = pg.evaluate(REDCELLS)
    check(f'{tag} R5.11: part 1 outlines the step where a task failed in red, and a right step stays gray', rc.get('failed') is True and rc.get('ok') is False, rc)
    rules(pg, f'{tag} part 1 result', lang); ov('part 1 result')
    shot(pg, f'{tag}-03-p1-result', full=(w > 700))
    # ---- part 2 ----
    go_part(pg, 1)
    st = state(pg)
    check(f'{tag} R3.20: part 2 does not replay the run part 1 just showed', not st['playing'] and st['pv'] == 20, (st['playing'], st['pv']))
    check(f'{tag} part 2: Next disabled until a choice', all(nexts(pg)) and why(pg), why(pg))
    check(f'{tag} part 2: no ▲▼ before the learner acts', not badges(pg), badges(pg))
    check(f'{tag} part 2: Settings shows n and p only', live_ctl(pg) == ['n', 'p', 'reset'], live_ctl(pg))
    click(pg, '#arl-rail [data-lever="half"]', 50)
    st = state(pg)
    check(f'{tag} R3.20: half the steps is a new picture, so the run replays from step 1', st['playing'] and st['fix'] is None and st['pv'] < 10, (st['playing'], st['fix'], st['pv']))
    check(f'{tag} part 2: half the steps gives 59.9%', '59.9%' in obs(pg) and not any(nexts(pg)), obs(pg))
    check(f'{tag} part 2: ▲▼ after the learner acts', bool(badges(pg)), badges(pg))
    click(pg, '#arl-rail [data-lever="steady"]')
    check(f'{tag} part 2: steadier steps give 81.8%', '81.8%' in obs(pg), obs(pg))
    check(f'{tag} part 2: single choice', pg.evaluate("()=>[...document.querySelectorAll('#arl-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1)
    check(f'{tag} part 2: the formula in full (A.ui.math)', pg.evaluate("()=>!!document.querySelector('#arl-rail .arl-fx').children.length"))
    rules(pg, f'{tag} part 2', lang); ov('part 2'); shot(pg, f'{tag}-04-p2', full=(w > 700))
    # ---- part 3 ----
    go_part(pg, 2)
    check(f'{tag} part 3: Next disabled until a retry', all(nexts(pg)), nexts(pg))
    click(pg, '#arl-player [data-act="end"]', 100)
    click(pg, '#arl-rail .seg-ctl[data-key="R"] button[data-v="2"]', 50)
    st = state(pg)
    check(f'{tag} R3.20: 2 retries replay only the rows that change, each from its first changed step', st['playing'] and st['fix'] is not None and any(x >= 20 for x in st['fix']) and any(x < 20 for x in st['fix']) and min(x for x in st['fix'] if x < 20) <= st['pv'] <= min(x for x in st['fix'] if x < 20) + 4, (st['playing'], st['fix'], st['pv']))
    check(f'{tag} part 3: 2 retries take 35.8% to 73.0%', '35.8%' in obs(pg) and '73.0%' in obs(pg) and not any(nexts(pg)), obs(pg))
    click(pg, '#arl-player [data-act="end"]', 100)
    click(pg, '#arl-rail .seg-ctl[data-key="R"] button[data-v="2"]', 50)
    st = state(pg)
    check(f'{tag} R3.20: the same setting again does not replay', not st['playing'] and st['pv'] == 20, (st['playing'], st['pv']))
    click(pg, '#arl-play', 50)
    st = state(pg)
    check(f'{tag} R3.20: the play button replays the whole run from step 1', st['playing'] and st['fix'] is None and st['pv'] < 4, (st['playing'], st['fix'], st['pv']))
    click(pg, '#arl-player [data-act="end"]', 100)
    rc = pg.evaluate(REDCELLS)
    check(f'{tag} R5.11: part 3 outlines a silent error and a stop in red; a retried or right step stays gray', rc.get('silent') is True and rc.get('stop', True) is True and rc.get('retry', False) is False and rc.get('ok') is False and rc.get('pale', False) is False, rc)
    slide(pg, '#arl-rail input[data-key="r"]', 3)
    check(f'{tag} part 3: fewer loud failures, fewer rescued', '73.0%' in obs(pg) and state(pg)['r'] == 0.3, obs(pg))
    check(f'{tag} part 3: Settings shows retries and the loud share', live_ctl(pg) == ['r', 'R', 'reset'], live_ctl(pg))
    rules(pg, f'{tag} part 3', lang); ov('part 3'); shot(pg, f'{tag}-05-p3', full=(w > 700))
    # ---- part 4 ----
    # R3.20 (addition of 2026-10-04): a run already watched to the end is not replayed when it is shown again
    go_part(pg, 1)
    st = state(pg)
    check(f'{tag} R3.20: Back to part 2 shows the run already watched at its end, with no replay', not st['playing'] and st['pv'] == 20 and st['step'] == 1, (st['playing'], st['pv'], st['step']))
    go_part(pg, 3)
    check(f'{tag} part 4: Next disabled until a choice', all(nexts(pg)), nexts(pg))
    p4 = text(pg, '#arl-rail .step.current')
    d1, d2, lv = p4.find(Z(lang, 'A check is one more model call', '检查就是一步做完以后，再调用一次模型')), p4.find(Z(lang, 'A retry redoes a step that went wrong', '重试就是把出错的这一步重做')), p4.find(Z(lang, 'No check, up to 2 retries', '不检查，最多重试 2 次'))
    check(f'{tag} R2.10: part 4 defines the check and the retry before its options', 0 <= d1 < lv and 0 <= d2 < lv, (d1, d2, lv))
    check(f'{tag} R6.6: part 4 says whose 80% (of the silent failures)', Z(lang, 'the check finds 80% of those', '出的错里有 30% 是悄悄出的，检查能发现其中的 80%') in p4, p4[:500])
    check(f'{tag} R2.10: each part 4 option says what is redone', all(x in p4 for x in Z(lang, ('Only the errors a tool reports are redone', 'The errors the check finds are redone too', 'An error the check finds is not redone, so the task fails. But the failure is seen.'), ('只有报出来的错会重做', '检查发现的错也会重做', '检查发现了错也不重做，任务就失败了，但失败被看见了'))), p4[-400:])
    click(pg, '#arl-rail [data-lever="ck"]')
    check(f'{tag} part 4: a check with 2 retries, 93.7% and 41.9 calls', '93.7%' in obs(pg) and '41.9' in obs(pg), obs(pg))
    click(pg, '#arl-rail [data-lever="ck0"]')
    check(f'{tag} part 4: a check without retries, 35.8%; wrong 13.2% to 2.3%; calls 14.6 to 26.3', all(x in obs(pg) for x in ('35.8%', '13.2%', '2.3%', '14.6', '26.3')) and '40.0' not in obs(pg), obs(pg))
    check(f'{tag} part 4: Settings shows the check and what it catches', live_ctl(pg) == ['ck', 'c', 'reset'], live_ctl(pg))
    rules(pg, f'{tag} part 4', lang); ov('part 4'); shot(pg, f'{tag}-06-p4', full=(w > 700))
    # ---- part 5 ----
    go_part(pg, 4)
    click(pg, '#arl-rail [data-lever="blind"]')
    check(f'{tag} part 5: the same call again, 58.6%', '58.6%' in obs(pg) and not any(nexts(pg)), obs(pg))
    check(f'{tag} part 5: links the pass-k lab', pg.evaluate("()=>!!document.querySelector('#arl-rail [data-obs] a, #arl-rail [data-obs] [data-lab], #arl-rail [data-obs] button')"))
    check(f'{tag} part 5: Settings shows how to retry and the same-call rate', live_ctl(pg) == ['kind', 'pb', 'reset'], live_ctl(pg))
    rules(pg, f'{tag} part 5', lang); ov('part 5'); shot(pg, f'{tag}-07-p5', full=(w > 700))
    # ---- part 6 ----
    go_part(pg, 5)
    cur = text(pg, '#arl-rail .step.current')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', cur.count('✗') == 3 and pg.evaluate("()=>document.querySelectorAll('#arl-rail [data-rem]').length") == 3 and pg.evaluate("()=>!!document.querySelector('#arl-rail [data-mode-go=\"sandbox\"]')"), cur[:200])
    check(f'{tag} part 6: the next-lab footer', pg.evaluate("()=>!!document.querySelector('#arl-app .next-lab')"))
    click(pg, '#arl-rail [data-rem="chain"]')
    check(f'{tag} part 6: “Show in the chart” sets its example', state(pg)['p'] == 0.99 and state(pg)['remView'] == 'chain', state(pg))
    rules(pg, f'{tag} part 6', lang); ov('part 6'); shot(pg, f'{tag}-08-p6', full=(w > 700))
    # ---- scenarios ----
    mode(pg, 'sandbox')
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#arl-rail')
    _lead = Z(lang, 'Pick a scenario and see how often its tasks succeed.', '选一个场景，看看它的任务能成功几成。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#arl-rail [data-preset]').length") == 4)
    click(pg, '#arl-rail [data-preset="1"]')
    pr = pg.evaluate("()=>[...document.querySelectorAll('#arl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: single choice, pressed follows the settings', pr == ['false', 'true', 'false', 'false'], pr)
    check(f'{tag} scenarios: the research agent’s result', '%' in text(pg, '#arl-insight'), text(pg, '#arl-insight'))
    check(f'{tag} scenarios: every Settings control is live', live_ctl(pg) == ['n', 'p', 'r', 'R', 'ck', 'kind', 'c', 'pb', 'seed', 'reset'], live_ctl(pg))
    # S8 (R5.10): 50 steps: narrow cells; a retried step is a mid-gray cell, distinct from a done one, and the legend says so
    click(pg, '#arl-rail [data-preset="2"]'); click(pg, '#arl-player [data-act="end"]', 250)
    px = pg.evaluate("""()=>{const s=window.__arl.state(), host=document.getElementById('arl-grid'), cv=document.getElementById('arl-cv'), g=cv.getContext('2d');
      const W=host.clientWidth, cw=(W-20)/s.n, dpr=cv.width/W; let ok=null, rt=null;
      const at=(i,j)=>{const d=g.getImageData(Math.round((j*cw+(cw-1)/2)*dpr),Math.round((i*9+3.5)*dpr),1,1).data; return 0.299*d[0]+0.587*d[1]+0.114*d[2];};
      for(let i=0;i<30;i++){const r=window.__arl.simRun({n:s.n,p:s.p,r:s.r,R:s.R,ck:s.ck,kind:s.kind,c:s.c,pb:s.pb,seed:s.seed},i);
        r.cells.forEach((c,j)=>{if(c.st==='ok'&&ok==null) ok=at(i,j); if(c.st==='retry'&&rt==null) rt=at(i,j);});}
      return {tight:window.__arl.tight(), cw, ok, rt, sw:!!document.querySelector('#arl-legend .arl-sw-rt.arl-tight')};}""")
    # since v26 the stage is wider on a desktop (the settings panel left the right side): 50 steps fit as wide cells, with segments
    check(f'{tag} 50 steps: narrow cells drawn without segments when the cells are narrow, and the legend swatch matches the drawing', px['tight'] == px['sw'] and (px['tight'] or px['cw'] >= 12), px)
    if px['tight']:
        check(f'{tag} 50 steps: a retried step stands apart from a done step (R5.10)', px['ok'] is not None and px['rt'] is not None and abs(px['rt'] - px['ok']) > 25 and px['rt'] < 200, px)
    shot(pg, f'{tag}-09b-grid50')
    click(pg, '#arl-rail [data-preset="1"]')
    click(pg, '#arl-player [data-act="end"]', 100)
    click(pg, '#arl-rail [data-preset="2"]')
    st = state(pg)
    check(f'{tag} R3.20: a scenario already watched shows its end at once', not st['playing'] and st['pv'] == 50, (st['playing'], st['pv']))
    click(pg, '#arl-rail [data-preset="1"]')
    st = state(pg)
    check(f'{tag} R3.20: back to the other scenario already watched, no replay either', not st['playing'] and st['pv'] == st['n'], (st['playing'], st['pv'], st['n']))
    slide(pg, '#arl-ctl-n-in', 16)
    pr = pg.evaluate("()=>[...document.querySelectorAll('#arl-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change under Settings unpresses the scenario', pr == ['false'] * 4, pr)
    # R3.2: a change under Settings does not move the page
    pg.evaluate("()=>document.querySelector('#arl-ctl-p-in').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    y0 = pg.evaluate("()=>document.querySelector('#arl-ctl-p-in').getBoundingClientRect().top")
    slide(pg, '#arl-ctl-p-in', 970)
    y1 = pg.evaluate("()=>document.querySelector('#arl-ctl-p-in').getBoundingClientRect().top")
    check(f'{tag} R3.2: a Settings change keeps the slider in place', abs(y1 - y0) < 2, (y0, y1))
    hsh = pg.evaluate("()=>location.hash")
    rules(pg, f'{tag} scenarios', lang); ov('scenarios'); shot(pg, f'{tag}-09-scenarios', full=(w > 700))
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    s2 = state(pg2)
    check(f'{tag} the URL restores the setup', s2['mode'] == 'sandbox' and s2['n'] == 16 and s2['p'] == 0.97 and s2['r'] == 0.2, (hsh, s2))
    pg2.close()
    # ---- challenge ----
    click(pg, '#arl-player [data-act="end"]', 100)   # the Settings run, watched to the end before the challenge
    mode(pg, 'challenge')
    check(f'{tag} challenge: Settings locked with a note', live_ctl(pg) == [] and text(pg, '#arl-lock') != '', live_ctl(pg))
    card = text(pg, '#arl-rail .railcard')
    check(f'{tag} R6.32: the challenge says what the cost counts, in plain words', Z(lang, 'A call to the model that is right 95% of the time per step counts 1', '每步成功率 95% 的模型调用一次，记 1 次') in card and Z(lang, 'each of its calls counts 2', '每步成功率 98% 的模型贵一倍，调用一次记 2 次') in card, card[:600])
    nm = re.compile(r'\b9[58]%[- ]model|9[58]% 模型|(?<!每步成功率 )9[58]% 的模型')
    check(f'{tag} R6.32: no “95% model” that reads like a name', not nm.search(card + text(pg, '#arl-lock') + text(pg, '#arl-app .labhead')), nm.findall(card))
    g = text(pg, '#arl-ch-g1') + text(pg, '#arl-ch-g2')
    check(f'{tag} challenge: live progress (35.8%, 15.9 calls)', '35.8%' in g and '15.9' in g, g)
    click(pg, '#arl-rail [data-act="submit"]', 250)
    res = text(pg, '#arl-ch-res')
    check(f'{tag} challenge tier 1: not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res and pg.evaluate("()=>!!document.querySelector('#arl-ch-res [data-act=\"reveal\"]')")
          and text(pg, '#arl-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res)
    check(f'{tag} challenge tier 1: with no retries the hint points at retries, not at silent failures (B1)', Z(lang, 'Allow retries first', '先允许重试') in res and Z(lang, 'silent', '悄悄') not in res, res)
    click(pg, '#arl-ch-res [data-act="reveal"]', 200)
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 4 rows, the 3 passing combinations and the initial setup (the pick), 28 left out', brief_ok(lang, bf, 4, 28), bf)
    res = text(pg, '#arl-ch-res')
    tips = pg.evaluate("()=>[...document.querySelectorAll('#arl-ch-res .def')].map(e=>e.getAttribute('data-tipx')).join(' | ')")
    check(f'{tag} challenge: retries alone, a check at 20 steps and the 98%-per-step model at 12 steps are named in words, their numbers in hovers',
          all(x in res for x in Z(lang, ('Retries alone stay below 90%', 'goes over the budget', 'keep it below 90%'), ('只靠重试，成功率还是不到 90%', '调用成本超出了预算', '成功率达不到 90%')))
          and '88.5%' in tips and pg.evaluate("()=>document.querySelectorAll('#arl-ch-res .def').length") == 3, (res[-700:], tips))
    click(pg, '#arl-rail .seg-ctl[data-key="chC"] button[data-v="1"]', 120)
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (a check, no retries, 20 steps) adds its own row, 5 rows, 27 left out', brief_ok(lang, bf, 5, 27), bf)
    for k, v in (('chR', 2), ('chN', 12)): click(pg, f'#arl-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 120)
    res = text(pg, '#arl-ch-res')
    check(f'{tag} challenge tier 2: met, but a cheaper one passes (24.8)', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'cheaper combination', '更省的组合') in res and '24.8' in res, res)
    check(f'{tag} challenge tier 2: the first line says that a cheaper combination passes', Z(lang, 'But a cheaper combination passes too', '不过还有更省的组合也能达标') in text(pg, '#arl-ch-res > p:first-child'), text(pg, '#arl-ch-res > p:first-child'))
    bf = brief(pg)
    check(f'{tag} challenge tier 2: the chart holds the 3 passing combinations (the pick among them) and the initial setup, 28 left out', brief_ok(lang, bf, 4, 28), bf)
    click(pg, '#arl-rail .seg-ctl[data-key="chR"] button[data-v="1"]', 150)
    res = text(pg, '#arl-ch-res')
    check(f'{tag} challenge tier 3: the minimal one', Z(lang, 'no passing combination costs less', '没有哪个达标的组合成本更低') in res and '91.6%' in res, res)
    rows = pg.evaluate("()=>document.querySelectorAll('#arl-ch-res .cmp-row').length")
    bf = brief(pg)
    check(f'{tag} challenge: the chart holds only the 3 passing combinations (the pick among them) and the initial setup, 28 left out', rows == 4 and brief_ok(lang, bf, 4, 28), (rows, bf))
    check(f'{tag} challenge: the chart tags the initial setup and the pick', all(x in res for x in Z(lang, ('initial setup', 'your pick'), ('初始设置', '你的选择'))), res[:300])
    check(f'{tag} R6.32: the chart and its notes name the models by their per-step rate', Z(lang, 'the 95%-per-step model', '每步成功率 95% 的模型') in res and Z(lang, 'the 98%-per-step model', '每步成功率 98% 的模型') in res and not nm.search(res), nm.findall(res))
    check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#arl-rail [data-exits]')"))
    rules(pg, f'{tag} challenge', lang); ov('challenge'); shot(pg, f'{tag}-10-challenge', full=(w > 700))
    click(pg, '#arl-player [data-act="end"]', 100)
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    s3 = state(pg)
    check(f'{tag} R3.20: a language switch does not replay the run', not s3['playing'] and s3['pv'] == 12, (s3['playing'], s3['pv']))
    check(f'{tag} the language switch keeps the answer', s3['mode'] == 'challenge' and s3['chO'] == {'strong': 0, 'ck': 1, 'R': 1, 'n': 12} and s3['chSubmitted'], s3)
    # ---- the dock at 390 ----
    if w < 700:
        pg.evaluate("()=>{const t=document.querySelector('#arl-rail [data-exits]')||document.querySelector('#arl-ch-res'); scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getClientRects().length?d.innerText:''}")
        check(f'{tag} the dock shows once the chart scrolls away', bool(dk) and '%' in dk, dk)
        rules(pg, f'{tag} dock', lang)
        shot(pg, f'{tag}-11-dock')
    mode(pg, 'sandbox')
    st = state(pg)
    check(f'{tag} R3.20: leaving the challenge shows the run watched before it at its end, with no replay', st['mode'] == 'sandbox' and not st['playing'] and st['pv'] == st['n'] and st['n'] == 16, (st['mode'], st['playing'], st['pv'], st['n']))
    pg.close()


# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 1, 2, 3, 4, 5 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
# The top bar's bottom is where it is on the screen now: on a desktop lab page the bar scrolls away (refit step 2), on a phone it
# stays on top and this is its height.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), b=document.querySelector('.topbar'), tb=b?Math.max(0,Math.round(b.getBoundingClientRect().bottom)):0;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(1, 2, 3, 4, 5, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#arl-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'arl-stage')
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
_H1_LAB, _H1_PFX = 'agent-reliability', 'arl'
_H1_PLAIN = ['ⁿ', '÷', '√']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['pⁿ', 'r<sub>c</sub>', 'p<sub>2</sub>', '<sup>${S.n}</sup>', '成功率 = 0.9ⁿ']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['arl-mb', 'arl-eqs', "tx('= no retries','= 不重试')"]       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#arl-grid', '#arl-curve']
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
_src = (P / 'labs' / 'agent-reliability.js').read_text() + (P / 'teasers' / 'agent-reliability.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['<p>有两个直接的办法能把它提上去。选一个，看看任务的成功率能涨到多少。</p>', '<p><b>重试</b>就是把出错的这一步重做，前提是错被看见了。', '<p>选一种设置，看看多花的调用能换回多少成功率。</p>', '<p>选一种重试的办法，看看任务的成功率怎么变。</p>', '步数一多，每步差一点，整个任务就差很多。所以长任务要求每一步都很稳，每步提高一点，成功率就能涨一大截。', '重试的前提是有人看见了错。重试次数再多，悄悄出的错也还在，成功率就卡在它们留下的上限。', '检查让悄悄出的错被看见，重试才能去改。代价是多花一些模型调用，换回的是悄悄出的错拿走的那部分成功率。', '一个 agent 的错有 70% 会报出来。给它重试 2 次，成功率从', '在图里显示重试 2 次、不检查的 agent。', '在图里显示每步检查、重试 2 次的 agent。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['和第 3 部分一样', '在图里显示第 3 部分', '在图里显示第 4 部分', '在第 2 部分，每步 95%', '在第 3 部分，重试 2 次', '在第 4 部分，每步检查一次', '第 3 部分的 agent 从', 'works as in part 3', 'Shows part 3’s agent', 'Shows part 4’s agent', 'In part 2, a 20-step', 'In part 3, 2 retries', 'In part 4, a check', 'part 3’s agent rose']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
