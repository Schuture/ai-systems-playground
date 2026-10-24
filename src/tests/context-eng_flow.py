"""Flow test for the context-eng lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the chart of calls;
- the prediction gate (R3.8): Next disabled with its reason (nextWhy) until a prediction AND the end of the run; before the prediction
  the run does not start and no result shows anywhere (text, tooltips, the header's popovers); “Saved” after a pick; then the run
  auto-plays (R3.9) with pause, one call, to the end, three speeds (fast really is fast) and a scrubber; the stage keeps its height while
  it plays (R7.9); at the end the result, the verdict with the right answer, and why (R3.10);
- parts 2 to 5: Next waits for the part's action, with its reason; the result names the change; Settings shows only the part's own
  controls (R4.2); ▲▼ badges only after the learner's own change (R6.6); part 6: three points with their “Show in the chart” buttons,
  three misconceptions, “See the scenarios”, the next-lab footer;
- Scenarios: four, single choice, the pressed state follows the settings (R1.8); the URL carries the setup and a fresh page restores it;
  a change under Settings does not move the page (R3.2);
- the challenge: only the four choices, Settings locked; live progress; the three feedback tiers (R4.4), tier 2 saying in its first
  line that a cheaper combination passes; “Show the answer”; the brief chart (the owner's review of batch 8): only the 4 passing
  combinations, the initial setup and the pick, both tagged, no other row, a note in the lab's words that counts the combinations left
  out (51, or 50 when the pick is another combination that misses), a title without “for each way of cutting”; the capability card;
  the language switch keeps the answer;
- the mini result bar (dock) at 390 once the chart scrolls away;
- the writing rules on every visible text node and tooltip, Chinese spacing and punctuation, and colours only for their data (R5.6).
Screenshots: shots/context-eng/.
Run: python3 tests/context-eng_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'context-eng'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/context-eng.html', '--only', 'context-eng'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'context-eng.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'flow.html')
SH = P / 'shots' / 'context-eng'; SH.mkdir(parents=True, exist_ok=True)
ARGS = sys.argv[1:]
LANGS = [x for x in ('en', 'zh') if x in ARGS] or ['en', 'zh']
SIZES = [s for s in [(1440, 900), (390, 844)] if str(s[0]) in ARGS] or [(1440, 900), (390, 844)]
errs, fails, n = [], [], 0


# polish round 3 (2026-10-16): part 6's per-claim source links moved from the card to the page-end sources (“Sources” / “来源”);
# every one of them must be there: the four Claude Code docs pages, OpenAI's Codex post, and the three Codex files at the pinned commit
PART6_SOURCES = ['https://code.claude.com/docs/en/' + p for p in ('prompt-caching', 'how-claude-code-works', 'context-window', 'sub-agents')] + \
    ['https://openai.com/index/unrolling-the-codex-agent-loop/'] + \
    ['https://github.com/openai/codex/blob/c2f7fe89d87ce853900d0b5cb1f5dc4863e44d73/codex-rs/' + p for p in ('core/src/compact.rs', 'protocol/src/openai_models.rs', 'core/src/client.rs')]

def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, hash_='context-eng'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__cte.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('cte-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 4 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} combinations fail at least one target', f'其余 {hidden} 种组合至少差一项') in r['note']
            and Z(lang, 'for each way of cutting', '按删法列出') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#cte-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#cte-rail .nx-why')].filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent).join(' | ')")
H = lambda pg, js: pg.evaluate(js)


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__cte||!window.__cte.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=200):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait); settle(pg)


def go_part(pg, i): click(pg, f'#cte-rail [data-go="{i}"]', 250)
def mode(pg, m): click(pg, f'#cte-app .modes button[data-mode="{m}"]', 350)
def to_end(pg): click(pg, '#cte-player [data-act="end"]', 200)


def slide(pg, sel, v):
    pg.evaluate(f"""()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}));}}"""); pg.wait_for_timeout(250); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#cte-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")
def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#cte-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
def live_ctl(pg): return pg.evaluate("()=>['task','W','hist','N','M','trim','notes','sub','p','cap','reset'].filter(k=>{const e=document.getElementById('cte-w-'+k); return e&&e.getClientRects().length>0})")


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #cte-mathbody')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|cte-over|cte-badm|cte-sw-o|cmp-|ex-res|wrong/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#cte-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#cte-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#cte-app .labhead','#cte-rail','#cte-stage','#cte-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#cte-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def heights(pg): return pg.evaluate("()=>[document.querySelector('#cte-stage').offsetHeight,document.querySelector('#cte-graph').offsetHeight,document.querySelector('#cte-tally').offsetHeight,document.querySelector('#cte-cap').offsetHeight]")


def run(b, w, h, lang):
    tag = f'{w}-{lang}'
    pg = page(b, w, h, lang)
    ov = lambda where: check(f'{tag} no horizontal overflow: {where}', overflow(pg) <= 0, overflow(pg))
    big = lambda x: H(pg, f"()=>window.ASP.fmt.big({x})")
    R = lambda s: H(pg, f"()=>window.__cte.run({s})")
    st = state(pg)
    check(f'{tag} opens on part 1, no prediction, not playing', st['step'] == 0 and st['pred'] is None and not st['playing'] and st['pv'] == 0, st)
    check(f'{tag} R3.9: a whole run of many calls defaults to fast', st['speed'] == 'f', st['speed'])
    ov('first screen')
    if w < 700:
        fs = pg.evaluate("()=>{const nx=document.querySelector('#cte-rail .pbar [data-nav=\"next\"]'), g=document.querySelector('#cte-chart'); const a=nx?nx.getBoundingClientRect():null, c=g.getBoundingClientRect(); return {next:!!a&&a.bottom<=innerHeight&&a.width>0, chart:c.top<innerHeight-40}}")
        check(f'{tag} phone first screen: the parts bar with Next and the chart (R7.3)', fs['next'] and fs['chart'], fs)
    shot(pg, f'{tag}-01-first')
    # ---- part 1: the gate ----
    base = R('{}')
    check(f'{tag} part 1: the engine says the window is full at step 38', base['over'] == 38 and base['peak'] == 127352, (base['over'], base['peak']))
    check(f'{tag} part 1: Next disabled before a prediction', all(nexts(pg)), nexts(pg))
    check(f'{tag} part 1: why Next is disabled (nextWhy)', Z(lang, 'Pick an answer', '选一个答案') in why(pg), why(pg))
    ba = before_answer(pg)
    for leak in ('step 38', '第 38 步', '127,352', 'right answer', '正确答案', 'Full at', '38 步塞满'):
        check(f'{tag} part 1: nothing gives the answer before the prediction ({leak})', leak not in ba, leak)
    check(f'{tag} part 1: the play button is off before the prediction', pg.evaluate("()=>document.querySelector('#cte-play').disabled"))
    rules(pg, f'{tag} part 1 before', lang)
    click(pg, '#cte-rail [data-pred="b"]', 150)
    st = state(pg)
    check(f'{tag} part 1: a pick starts the run (auto-play)', st['pred'] == 'b' and st['playing'], st)
    check(f'{tag} part 1: “Saved” after the pick', Z(lang, 'Saved', '已记下') in text(pg, '#cte-rail .step.current'))
    check(f'{tag} part 1: Next still waits for the run, with its reason', all(nexts(pg)) and Z(lang, 'Wait for the run', '等这次运行结束') in why(pg), (nexts(pg), why(pg)))
    check(f'{tag} part 1: no result while the run plays', '127,352' not in before_answer(pg) and Z(lang, 'right answer', '正确答案') not in before_answer(pg))
    hs = []
    for _ in range(6): hs.append(heights(pg)); pg.wait_for_timeout(260)
    check(f'{tag} R7.9: the stage keeps its height while the run plays', all(x == hs[0] for x in hs), hs)
    shot(pg, f'{tag}-02-running')
    # speeds: a whole run takes 5 s fast, 15 s medium (38 calls here)
    click(pg, '#cte-player .cte-speed button[data-v="f"]', 50)
    p0 = state(pg)['pv']; pg.wait_for_timeout(1000); p1 = state(pg)['pv']
    check(f'{tag} fast: about 7.6 calls a second', 5 <= p1 - p0 <= 11 or (p1 == 38 and p1 > p0), (p0, p1))
    if p1 >= 37:
        click(pg, '#cte-play', 50)   # replay from the start, to measure medium
    click(pg, '#cte-player .cte-speed button[data-v="m"]', 50)
    if not state(pg)['playing']: click(pg, '#cte-play', 50)
    p0 = state(pg)['pv']; pg.wait_for_timeout(1000); p1 = state(pg)['pv']
    check(f'{tag} medium: about 2.5 calls a second', 1 <= p1 - p0 <= 4, (p0, p1))
    click(pg, '#cte-play', 50); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} pause stops the replay', not s1['playing'] and s1['pv'] == s2['pv'], (s1['pv'], s2['pv']))
    click(pg, '#cte-player [data-act="stepf"]', 50)
    check(f'{tag} one call forward', state(pg)['pv'] == s2['pv'] + 1, (s2['pv'], state(pg)['pv']))
    to_end(pg)
    st = state(pg)
    check(f'{tag} part 1: to the end shows all 38 calls and enables Next', st['runsSeen'] and st['pv'] == 38 and not any(nexts(pg)), (st, nexts(pg)))
    o = obs(pg)
    check(f'{tag} part 1: the result, the verdict with the right answer, and why',
          Z(lang, 'At step 38', '到第 38 步') in o and '131,072' in o and Z(lang, 'The right answer is “Between step 30 and 49”', '正确答案是“第 30 到 49 步之间”') in o and Z(lang, 'About 43 steps fit', '大约够放 43 步') in o and Z(lang, 'steps 22 to 30', '第 22 到 30 步') in o, o)
    check(f'{tag} part 1: the metrics show after the run', Z(lang, 'window full at step 38', '第 38 步塞满') in text(pg, '#cte-metrics') and big(base['read']) in text(pg, '#cte-metrics'), text(pg, '#cte-metrics'))
    check(f'{tag} part 1: the note no longer offers a change once the run is done', Z(lang, 'You can still change it', '还可以改') not in text(pg, '#cte-rail .step.current'))
    check(f'{tag} part 1: no ▲▼ for the first run', not badges(pg), badges(pg))
    rules(pg, f'{tag} part 1 result', lang); ov('part 1 result')
    shot(pg, f'{tag}-03-p1-result', full=(w > 700))
    # ---- part 2: what fills the context ----
    go_part(pg, 1)
    st = state(pg)
    check(f'{tag} R3.20: part 2 shows the run part 1 just played, so it does not replay it', not st['playing'] and st['pv'] == 38, st)
    check(f'{tag} part 2: Next disabled until a choice, with its reason', all(nexts(pg)) and Z(lang, 'Pick a kind', '选一类') in why(pg), why(pg))
    check(f'{tag} part 2: Settings shows nothing to change, with a note', live_ctl(pg) == [] and text(pg, '#cte-lock') != '', live_ctl(pg))
    click(pg, '#cte-rail [data-lever="conv"]')
    check(f'{tag} part 2: the conversation is a small share', '%' in obs(pg) and not any(nexts(pg)), obs(pg))
    click(pg, '#cte-rail [data-lever="tool"]')
    sh = H(pg, "()=>{const r=window.__cte.run({}); const c=r.calls.filter(x=>x.k!=='over').pop(); return Math.round(c.seg[3]/c.ctx*100)+'%'}")
    check(f'{tag} part 2: tool outputs are most of it ({sh})', sh in obs(pg) and state(pg)['hl'] == 'tool', (sh, obs(pg)))
    check(f'{tag} part 2: single choice', pg.evaluate("()=>[...document.querySelectorAll('#cte-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1)
    rules(pg, f'{tag} part 2', lang); ov('part 2'); shot(pg, f'{tag}-04-p2', full=(w > 700))
    # ---- part 3: trim and slide ----
    go_part(pg, 2)
    st = state(pg)
    check(f'{tag} R3.20: part 3 starts from the same run, no replay', not st['playing'] and st['pv'] == 38, st)
    check(f'{tag} part 3: Next disabled until a cut, with its reason', all(nexts(pg)) and Z(lang, 'Pick a cut', '选一种删法') in why(pg), why(pg))
    check(f'{tag} part 3: no ▲▼ before the learner acts', not badges(pg), badges(pg))
    check(f'{tag} part 3: Settings shows the history, its length, the trim and its cap', live_ctl(pg) == ['hist', 'N', 'trim', 'cap', 'reset'], live_ctl(pg))
    click(pg, '#cte-rail [data-lever="trim"]')
    st = state(pg); nt = len(R('{trim:1}')['calls'])
    check(f'{tag} R3.20: a cut that changes the run replays it', st['playing'] and 0 < st['pv'] < nt, (st['pv'], nt))
    to_end(pg)
    check(f'{tag} part 3: trimming fits, 82,361 at most, all 3 facts', '82,361' in obs(pg) and not any(nexts(pg)), obs(pg))
    tip = pg.evaluate("()=>document.querySelector('#cte-play').dataset.tipx||''")
    check(f'{tag} R3.20: the finished run has a replay control that says it starts over', Z(lang, 'again from the start', '从头再播') in tip and Z(lang, 'Replay', '重放') in text(pg, '#cte-play'), (tip, text(pg, '#cte-play')))
    click(pg, '#cte-play', 50); st = state(pg)
    check(f'{tag} R3.20: the replay control plays the run again from the start', st['playing'] and st['pv'] < nt // 2, st['pv'])
    to_end(pg)
    click(pg, '#cte-rail [data-lever="trim"]'); st = state(pg)
    check(f'{tag} R3.20: the same choice again does not replay', not st['playing'] and st['pv'] == nt, (st['pv'], nt))
    check(f'{tag} part 3: ▲▼ after the learner acts', bool(badges(pg)), badges(pg))
    click(pg, '#cte-rail [data-lever="slide"]'); to_end(pg)
    t, s = R("{trim:1}"), R("{trim:1,hist:'last'}")
    o = obs(pg)
    fm = lambda x: big(round(x)) if x >= 1e6 else f'{x / 1e6:.2f}M'
    check(f'{tag} part 3: the sliding window, 20,726 at most, about 2.5 times the cost in words, no fact left', '20,726' in o and Z(lang, 'about 2.5 times as much as trimming alone', '只修剪时的约 2.5 倍') in o and fm(t['cost']) not in o and fm(s['cost']) not in o and Z(lang, 'none of the 3 facts', '一个都不在') in o, o)
    check(f'{tag} part 3: the formula in full (A.ui.math)', pg.evaluate("()=>!!document.querySelector('#cte-mathbody math, #cte-mathbody .cte-fx')"))
    rules(pg, f'{tag} part 3', lang); ov('part 3'); shot(pg, f'{tag}-05-p3', full=(w > 700))
    # ---- part 4: summaries and notes ----
    go_part(pg, 3)
    st = state(pg)
    check(f'{tag} R3.20: part 4 starts with the sliding window part 3 just showed, no replay', not st['playing'] and st['pv'] > 0, st)
    check(f'{tag} part 4: Next disabled until a choice, with its reason', all(nexts(pg)) and why(pg), why(pg))
    check(f'{tag} part 4: Settings shows the history, the summary interval, notes and p', live_ctl(pg) == ['hist', 'M', 'notes', 'p', 'reset'], live_ctl(pg))
    click(pg, '#cte-rail [data-lever="sum"]'); to_end(pg)
    o = obs(pg)
    check(f'{tag} part 4: a summary every 20 steps, 64% and 33% for all three', '64%' in o and '33%' in o and not any(nexts(pg)), o)
    click(pg, '#cte-rail [data-lever="notes"]'); to_end(pg)
    b2 = R("{trim:1,hist:'sum',M:20,notes:1}")
    o = obs(pg)
    check(f'{tag} part 4: notes keep all 3 for sure, the cost barely moves (no exact value)', Z(lang, 'barely moves', '几乎没变') in o and fm(b2['cost']) not in o and Z(lang, 'for sure', '一定在') in o, o)
    nc = len(R("{trim:1,hist:'sum',M:20}")['calls'])
    click(pg, '#cte-rail [data-lever="sum"]'); st = state(pg)
    check(f'{tag} R3.20: a lever that brings back a run already watched shows its end at once', st['notes'] == 0 and not st['playing'] and st['pv'] == nc, (st['pv'], nc, st['playing']))
    click(pg, '#cte-rail [data-lever="notes"]'); st = state(pg)
    check(f'{tag} R3.20: and back to the notes run, also watched, no replay', st['notes'] == 1 and not st['playing'] and st['pv'] == len(b2['calls']), (st['pv'], st['playing']))
    slide(pg, '#cte-ctl-p-in', 9)
    check(f'{tag} part 4: the p slider changes p', abs(state(pg)['p'] - 0.9) < 1e-9, state(pg)['p'])
    rules(pg, f'{tag} part 4', lang); ov('part 4'); shot(pg, f'{tag}-06-p4', full=(w > 700))
    # ---- part 5: the sub-agent ----
    go_part(pg, 4)
    check(f'{tag} part 5: Next disabled until the sub-agent, with its reason', all(nexts(pg)) and Z(lang, 'sub-agent', '子 agent') in why(pg), why(pg))
    check(f'{tag} part 5: Settings shows the sub-agent only', live_ctl(pg) == ['sub', 'reset'], live_ctl(pg))
    click(pg, '#cte-rail [data-lever="sub"]'); to_end(pg)
    o = obs(pg)
    check(f'{tag} part 5: a sub-agent fits, 114,927 at most, 88% of the window', '114,927' in o and '88%' in o and not any(nexts(pg)), o)
    rules(pg, f'{tag} part 5', lang); ov('part 5'); shot(pg, f'{tag}-07-p5', full=(w > 700))
    click(pg, '#cte-rail [data-nav="back"]', 250); st = state(pg); ns = len(R("{trim:1,hist:'last'}")['calls'])
    check(f'{tag} R3.20: Back from part 5 to part 4 shows the sliding window already watched, its end at once', st['step'] == 3 and st['hist'] == 'last' and not st['playing'] and st['pv'] == ns, (st['step'], st['pv'], ns, st['playing']))
    # ---- part 6: how Codex and Claude Code do it (owner point 4) ----
    go_part(pg, 5)
    st = state(pg)
    check(f'{tag} R3.20: part 6 shows the run part 4 already played to the end, so it shows its end at once', not st['playing'] and st['pv'] == len(b2['calls']), (st['pv'], st['playing']))
    cur = text(pg, '#cte-rail .step.current')
    links = pg.evaluate("()=>[...document.querySelectorAll('[data-endnotes] a')].map(a=>a.href)")
    check(f'{tag} part 6: Codex and Claude Code, five points, Next open; the card points to the page-end sources, which hold every source of the five points', all(x in cur for x in ('Codex', 'Claude Code', Z(lang, '90% of the window', '窗口的 90%'), 'Σ', Z(lang, 'sources at the end of the page', '出处在页末'))) and cur.count(Z(lang, 'Show in the chart', '在图里看')) == 4 and all(u in links for u in PART6_SOURCES) and not any(nexts(pg)), (cur[:300], [u for u in PART6_SOURCES if u not in links]))
    # polish round 3, the first-time reader (2026-10-16): the first point's button showed trimming alone, which costs more than the run
    # on entry, the opposite of the point; it now shows the sliding window the point names, the run whose cost rises
    click(pg, '#cte-rail [data-rem="slide"]'); st = state(pg)
    check(f'{tag} part 6: the first point’s “Show in the chart” shows the sliding window it names (trim, the last 10 steps); part 3 already played it, so its end shows at once', st['hist'] == 'last' and st['N'] == 10 and st['trim'] == 1 and st['notes'] == 0 and st['remView'] == 'slide' and not st['playing'] and st['pv'] == ns, st)
    to_end(pg); click(pg, '#cte-rail [data-rem="comp"]'); st = state(pg)
    check(f'{tag} part 6: “Show in the chart” for compaction sets trim + a summary every 20 steps; part 4 already played it, so its end shows at once', st['hist'] == 'sum' and st['trim'] == 1 and st['notes'] == 0 and not st['playing'] and st['pv'] == nc, st)
    click(pg, '#cte-rail [data-rem="comp"]'); st = state(pg)
    check(f'{tag} R3.20: the same “Show in the chart” again does not replay', not st['playing'], st)
    click(pg, '#cte-rail [data-rem="subx"]'); st = state(pg)
    check(f'{tag} R3.20: a run not watched yet still plays', st['sub'] == 1 and st['playing'], st)
    to_end(pg); click(pg, '#cte-rail [data-rem="comp"]'); st = state(pg)
    check(f'{tag} R3.20: back to a watched run, its end at once', not st['playing'] and st['pv'] == nc, (st['pv'], nc, st['playing']))
    click(pg, '#cte-rail [data-rem="sum"]'); to_end(pg)
    rules(pg, f'{tag} part 6', lang); ov('part 6'); shot(pg, f'{tag}-08-p6-prod', full=(w > 700))
    # ---- part 7: what to remember ----
    go_part(pg, 6)
    st = state(pg)
    check(f'{tag} R3.20: part 7 starts with the run part 6 just showed, no replay', not st['playing'] and st['pv'] > 0, st)
    cur = text(pg, '#cte-rail .step.current')
    check(f'{tag} part 7: three points, three misconceptions, See the scenarios', cur.count('✗') == 3 and pg.evaluate("()=>document.querySelectorAll('#cte-rail [data-rem]').length") == 3 and pg.evaluate("()=>!!document.querySelector('#cte-rail [data-mode-go=\"sandbox\"]')"), cur[:200])
    check(f'{tag} part 7: the misconceptions carry computed numbers (more than 2 times; 80%, 64%, 51%)', all(x in cur for x in ('80%', '64%', '51%')) and Z(lang, 'more than 2 times', '2 倍多') in cur, cur[-600:])
    check(f'{tag} part 7: the next-lab footer', pg.evaluate("()=>!!document.querySelector('#cte-app .next-lab')"))
    click(pg, '#cte-rail [data-rem="sum"]')
    st = state(pg)
    check(f'{tag} part 7: “Show in the chart” sets its example, the same run, so no replay', st['hist'] == 'sum' and st['M'] == 20 and st['trim'] == 1 and st['remView'] == 'sum' and not st['playing'], st)
    rules(pg, f'{tag} part 7', lang); ov('part 7'); shot(pg, f'{tag}-08b-p7', full=(w > 700))
    # ---- scenarios ----
    mode(pg, 'sandbox')
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#cte-rail')
    _lead = Z(lang, 'Pick a scenario and see how its context grows.', '选一个场景，看看它的上下文怎么涨。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    st = state(pg)
    check(f'{tag} R3.20: the coding scenario is the run part 7 showed, so switching mode does not replay it', not st['playing'] and st['pv'] > 0, st)
    check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#cte-rail [data-preset]').length") == 4)
    click(pg, '#cte-rail [data-preset="1"]')
    pr = pg.evaluate("()=>[...document.querySelectorAll('#cte-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: single choice, pressed follows the settings', pr == ['false', 'true', 'false', 'false'], pr)
    check(f'{tag} scenarios: the research agent’s result (84,001 at most)', '84,001' in text(pg, '#cte-insight'), text(pg, '#cte-insight'))
    check(f'{tag} scenarios: every Settings control is live', live_ctl(pg) == ['task', 'W', 'hist', 'N', 'M', 'trim', 'notes', 'sub', 'p', 'cap', 'reset'], live_ctl(pg))
    click(pg, '#cte-rail [data-preset="2"]')
    check(f'{tag} scenarios: the chat loses its early request', Z(lang, 'is gone', '已经不在') in text(pg, '#cte-insight'), text(pg, '#cte-insight'))
    click(pg, '#cte-rail [data-preset="1"]')
    # R3.2: a change under Settings does not move the page
    pg.evaluate("()=>document.querySelector('#cte-ctl-p-in').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    y0 = pg.evaluate("()=>document.querySelector('#cte-ctl-p-in').getBoundingClientRect().top")
    slide(pg, '#cte-ctl-p-in', 7)
    y1 = pg.evaluate("()=>document.querySelector('#cte-ctl-p-in').getBoundingClientRect().top")
    check(f'{tag} R3.2: a Settings change keeps the slider in place', abs(y1 - y0) < 2, (y0, y1))
    pr = pg.evaluate("()=>[...document.querySelectorAll('#cte-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change under Settings unpresses the scenario', pr == ['false'] * 4, pr)
    hsh = pg.evaluate("()=>location.hash")
    rules(pg, f'{tag} scenarios', lang); ov('scenarios'); shot(pg, f'{tag}-09-scenarios', full=(w > 700))
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    s2 = state(pg2)
    check(f'{tag} the URL restores the setup', s2['mode'] == 'sandbox' and s2['task'] == 'research' and s2['hist'] == 'sum' and s2['sub'] == 1 and abs(s2['p'] - 0.7) < 1e-9, (hsh, s2))
    pg2.close()
    to_end(pg)
    # ---- challenge ----
    mode(pg, 'challenge')
    check(f'{tag} challenge: Settings locked with a note', live_ctl(pg) == [] and text(pg, '#cte-lock') != '', live_ctl(pg))
    g = text(pg, '#cte-ch-g1') + text(pg, '#cte-ch-g2')
    check(f'{tag} challenge: live progress (full at step 45)', Z(lang, 'full at step 45', '第 45 步塞满') in g, g)
    click(pg, '#cte-rail [data-act="submit"]', 250)
    res = text(pg, '#cte-ch-res')
    check(f'{tag} challenge tier 1: not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res and pg.evaluate("()=>!!document.querySelector('#cte-ch-res [data-act=\"reveal\"]')")
          and text(pg, '#cte-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res)
    click(pg, '#cte-ch-res [data-act="reveal"]', 200)
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 5 rows, the 4 passing combinations and the initial setup (the pick), 51 left out', brief_ok(lang, bf, 5, 51), bf)
    click(pg, '#cte-rail .seg-ctl[data-key="chH"] button[data-v="4"]', 120)
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (last 30 steps, no notes) adds its own row, 6 rows, 50 left out', brief_ok(lang, bf, 6, 50), bf)
    click(pg, '#cte-rail .seg-ctl[data-key="chN"] button[data-v="1"]', 120)
    best = H(pg, "()=>window.__cte.chBest()")
    res = text(pg, '#cte-ch-res')
    fd = lambda x: (lambda d: '$' + (str(round(d)) if d >= 10 else (f'{d:.1f}' if d >= 1 else f'{d:.2f}')))(x * 3 / 1e6)   # dollars at $3 per million input tokens
    check(f'{tag} challenge tier 2: met, but a cheaper one passes ({fd(best["cost"])})', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'cheaper combination', '更省的组合') in res and fd(best['cost']) in res and fm(best['cost']) not in res, res)
    check(f'{tag} challenge tier 2: the first line says that a cheaper combination passes', Z(lang, 'But a cheaper combination passes too', '不过还有更省的组合也能达标') in text(pg, '#cte-ch-res > p:first-child'), text(pg, '#cte-ch-res > p:first-child'))
    for k, v in (('chH', 3), ('chS', 1)): click(pg, f'#cte-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 120)
    res = text(pg, '#cte-ch-res')
    check(f'{tag} challenge tier 3: the cheapest one', Z(lang, 'no passing combination costs less', '没有哪个达标的组合花得更少') in res and '63,631' in res and '$8.3' in res, res)
    mt = pg.evaluate("()=>{const m=document.querySelector('#cte-metrics .metric[data-k=\"read\"]'); return m?[m.querySelector('.v').textContent, m.getAttribute('data-tipx')||'']:null}")
    check(f'{tag} the tokens-read card shows M and its hover says what the number is, without spelling it out exactly (the owner, 2026-10-07 23:37)',
          bool(mt) and mt[0].endswith('M') and Z(lang, 'sum over all calls', '所有调用加起来') in mt[1] and not re.search(r'\d{1,3}(,\d{3})+', mt[1]), mt)
    to_end(pg); click(pg, '#cte-rail .seg-ctl[data-key="chS"] button[data-v="1"]', 120); st = state(pg)
    check(f'{tag} R3.20: the same challenge choice again does not replay', not st['playing'] and st['pv'] > 0, st)
    rows = pg.evaluate("()=>document.querySelectorAll('#cte-ch-res .cmp-row').length")
    bf = brief(pg)
    check(f'{tag} challenge: the chart holds only the 4 passing combinations (the pick among them) and the initial setup, 51 left out', rows == 5 and brief_ok(lang, bf, 5, 51), (rows, bf))
    check(f'{tag} challenge: the chart tags the initial setup and the pick', all(x in res for x in Z(lang, ('initial setup', 'your pick'), ('初始设置', '你的选择'))), res[:300])
    check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#cte-rail [data-exits]')"))
    rules(pg, f'{tag} challenge', lang); ov('challenge'); shot(pg, f'{tag}-10-challenge', full=(w > 700))
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    s3 = state(pg)
    check(f'{tag} the language switch keeps the answer', s3['mode'] == 'challenge' and s3['chO'] == {'h': 3, 'trim': 0, 'notes': 1, 'sub': 1} and s3['chSubmitted'], s3)
    # ---- the dock at 390 ----
    if w < 700:
        pg.evaluate("()=>{const t=document.querySelector('#cte-rail [data-exits]')||document.querySelector('#cte-ch-res'); scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getClientRects().length?d.innerText:''}")
        check(f'{tag} the dock shows once the chart scrolls away', bool(dk) and '%' in dk, dk)
        rules(pg, f'{tag} dock', lang)
        shot(pg, f'{tag}-11-dock')
    # leaving the challenge brings back the research scenario, already watched to the end: no replay (R3.20)
    mode(pg, 'sandbox'); st = state(pg); nr = len(R("{task:'research',hist:'sum',M:20,notes:1,sub:1}")['calls'])
    check(f'{tag} R3.20: leaving the challenge shows the scenario run already watched, its end at once', st['mode'] == 'sandbox' and st['task'] == 'research' and not st['playing'] and st['pv'] == nr, (st['task'], st['pv'], nr, st['playing']))
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            t0 = time.time(); run(b, w, h, lang); print(f'{w} {lang}: {time.time() - t0:.0f}s')
    b.close()

for x in sorted(set(rule_hits)): print('RULE', x)
check('writing and colour rules', not rule_hits, len(set(rule_hits)))
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'context-eng', 'cte'
_H1_PLAIN = ['÷', 'm_j', 'o_j', 'C_hit', 'Chit']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['C<sub>hit</sub>', 'C<sub>缓存</sub>', 'm_j is step', 'm_j 是第 j 步', '÷ 2 = ${d.n} ×', '÷ 1,000,000 ≈']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['cte-mb', 'cte-eqs']       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#cte-chart', '#cte-lanes-h']
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
# polish round 3 (2026-10-16, B4 / R7.12, the owner: "the panel was pushed down, I did not see there was more below"): at 1440 × 900,
# entering each listed guided part the way a reader does (Next), the stage needs no scrolling inside itself and nothing in it
# scrolls either; in Chinese it could still grow by 30 px (its max-height minus the height its content and borders need), in
# English it does not scroll; its main picture lies inside it and on the screen. The real layout: window.__aspPrimer, as the probe.
_FIT_PARTS = [1, 2, 3, 4, 5, 6, 7]
_FIT_V = "(()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); return [...v.querySelectorAll('.bench')].find(x=>x.offsetParent)||v;})()"
# tb: the top bar's bottom where it is on the screen now (on a desktop lab page the bar scrolls away, refit step 2; on a phone it stays on top).
_FIT_M = ("(()=>{const b=" + _FIT_V + ", s=b.querySelector(':scope>.stage'), cur=b.querySelector('.steps>.step.current'), r=s.getBoundingClientRect(), tb=(t=>t?Math.max(0,Math.round(t.getBoundingClientRect().bottom)):0)(document.querySelector('.topbar')); let inner=0;"
          "s.querySelectorAll('*').forEach(e=>{const c=getComputedStyle(e); if(/(auto|scroll)/.test(c.overflowY)&&e.getClientRects().length) inner=Math.max(inner,e.scrollHeight-e.clientHeight);});"
          "const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((x,y)=>y.width*y.height-x.width*x.height)[0];"
          "return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, inner,"
          "fig:!f||(f.top>=r.top-1&&f.bottom<=r.bottom+1&&f.top>=tb-2&&f.bottom<=innerHeight+2)};})()")
if '1440' in str(SIZES):
    with sync_playwright() as _fp:
        _fb = _fp.chromium.launch()
        for _lang in LANGS:
            _pg = _fb.new_page(viewport={'width': 1440, 'height': 900}); _pg.add_init_script('window.__aspPrimer=true')
            _pg.on('pageerror', lambda e, _l=_lang: errs.append(f'fit 1440 {_l}: {e}'))
            _pg.goto(URL); _pg.wait_for_timeout(300); _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(80)
            _pg.evaluate("()=>{location.hash='context-eng'}"); _pg.wait_for_timeout(500)
            _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
            if _g.count(): _g.first.click(); _pg.wait_for_timeout(700)
            for _i in range(1, max(_FIT_PARTS) + 1):
                if _i > 1:
                    _nb = _pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
                    if _nb.count(): _nb.first.click()
                    else: _pg.evaluate(f"{_FIT_V}.querySelector('.steps>.step:nth-child({_i}) .step-h').click()")
                    _pg.wait_for_timeout(900)
                if _i not in _FIT_PARTS: continue
                _f = _pg.evaluate(_FIT_M)
                if _lang == 'zh':
                    check(f'B4 1440 zh part {_i}: the stage fits without scrolling inside, nothing in it scrolls, 30 px to spare, its picture inside it and on screen (R7.12)',
                          _f['part'] == _i and _f['scroll'] <= 0 and _f['inner'] <= 4 and _f['spare'] >= 30 and _f['fig'], _f)
                else:
                    check(f'B4 1440 en part {_i}: the stage fits without scrolling inside, its picture inside it and on screen (R7.12)',
                          _f['part'] == _i and _f['scroll'] <= 0 and _f['spare'] >= 0 and _f['fig'], _f)
            _pg.close()
        _fb.close()
# end B4
check('no page errors', not errs, errs[:5])
# the owner, 2026-10-08 (G1, G2) src: every option group leads in with what to pick and what to watch (G1); no sentence leans on an earlier part's detail (G2)
# polish round 3 (2026-10-16): part 6's three G2 sentences were shortened; the list below holds their new wording (still no part numbers)
_src = (P / 'labs' / 'context-eng.js').read_text() + (P / 'teasers' / 'context-eng.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['<p>选一类内容，看看它占了上下文的多少。图里会把它标出来。比例按第 ${lastFit().step} 步算，这是最后一次还放得下的调用。</p>', '<p>这个任务里有三个事实，要隔很久才用到。图下面的三行显示它们在哪里。</p><p>选一种删法，看看三个事实还剩几个。</p>', '为了好比较，这里固定每 20 步压缩一次。</p><p>选一种办法，看看三个事实能不能都留住。</p>', '去做。选一种做法，看看上下文最多占窗口的多少。</p>', '工具的输出原样留着，上下文一步比一步长。', '早先的步骤被删掉，三个事实也跟着没了。', 'agent 自己读文档，网页都留在它的上下文里。', '用的正是这个 Lab 的办法。', '滑动窗口每一步都改开头，用不上缓存，反而更贵。', '这和记笔记是同一个思路。', 'agent 的上下文里，大头是工具的输出。修剪它们最省事，常常单靠这一招，长任务就放得下了。', '每一步都改开头的删法（比如滑动窗口）用不上缓存。所以它读的 token 少了，花的钱反而可能更多。', '每压缩一次，都可能丢掉一个事实，所以只靠摘要，事实越往后越不保险。写进笔记就一定留得住，花的钱也只多一点。', '每 20 步压缩一次时，第 3 步用户提的要求只有 64% 的把握还在。为什么？', '这个 60 步的 coding agent 一共读进去', '如果 agent 同时修剪、每 20 步压缩、再记笔记，再加上子 agent', '每一步都改开头的删法（比如滑动窗口）会让缓存失效。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['第 1 部分的 agent', '第 3 部分丢了事实', '前面几部分的办法', '第 3 部分的滑动窗口', '第 4 部分的笔记', '这就是第 5 部分的做法', '第 2 部分里，工具的输出', '第 3 部分修剪之后', '第 4 部分里，只靠摘要', '第 4 部分每 20 步压缩一次', 'Part 1’s agent', 'Part 3’s agent', 'the parts before', 'Part 3’s sliding window', 'part 4’s notes', 'part 5’s sub-agent', 'In part 2, tool outputs', 'In part 3, the sliding window', 'In part 4, with summaries', 'In part 4, a summary', 'part 1’s agent reads', '第 4 部分最后的做法', '第 3 部分的滑动窗口就是这样', 'Part 4’s last setup', 'as part 3 showed']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
