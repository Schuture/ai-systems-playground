"""Flow test for the prompt-injection lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors and no horizontal overflow at any step;
- the phone first screen (m2 template, R7.3): the parts' bar with Next, and the replay;
- the prediction gate (R3.8): Next disabled with its reason (nextWhy) until a prediction AND the end of the replay; before the
  prediction the replay shows only the setup and no result shows anywhere (text, tooltips, the header's popovers); “Saved” after a
  pick; then the replay auto-plays (R3.9) with pause, one step, to the end, three speeds (fast really is fast) and a scrubber; the stage
  keeps its height while it plays (R7.9); at the end the result, the verdict with the right answer, and why (R3.10);
- parts 2 to 4: Next waits for the part's action; the result names the change; Settings shows only the part's own controls (R4.2);
  ▲▼ badges only after the learner's own change (R6.6); part 2 links agent-anatomy and context-eng; part 5: five practices, the
  inbox assistant introduced with its replayed task, CaMeL in the papers list; a new replay (a part, ▶ Replay) shows only the events
  played so far from the first frame (round 3); part 6: three points, three misconceptions, “See the scenarios”, the next-lab footer;
- Scenarios: four, single choice, the pressed state follows the settings (R1.8); the URL carries the setup and a fresh page restores
  it; a change under Settings does not move the page (R3.2);
- the challenge: only the four choices, Settings locked; live progress; the three feedback tiers (R4.4), tier 2 saying in its first
  line that a combination that asks less passes; “Show the answer”; the brief chart (the owner's review of batch 8): only the 4
  passing combinations, the initial setup and the pick, both tagged, no other row, a note in the lab's words that counts the
  combinations left out (27, or 26 when the pick is another combination that misses), a title without “for each combination”; the
  capability card; the language switch keeps the answer;
- the mini result bar (dock) at 390 once the chart scrolls away;
- owner check of batch 7 (2026-10-04): the replay comes first and the metric cards sit under the picture (R7.12); the injected
  instruction and the harmful action turn red, with ⚠ / ✗ and a word, when the replay reaches each (R5.11); a new part, mode,
  choice, scenario or setting replays only when the replayed task goes differently, and only from the first event that differs; the
  ▶ Replay button replays on demand (R3.20); a run already seen does not play again when it comes back (email back on in part 3,
  part 5 → part 6, leaving the challenge), and ▶ Replay still plays it;
- the writing rules on every visible text node and tooltip, Chinese spacing and punctuation, and colours only for their data (R5.6).
Screenshots: shots/prompt-injection/.
Run: python3 tests/prompt-injection_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'prompt-injection'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prompt-injection.html', '--only', 'prompt-injection'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(OUT / 'flow.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prompt-injection.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'flow.html')
SH = P / 'shots' / 'prompt-injection'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='prompt-injection'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__pin.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('pin-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 4 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} combinations miss at least one goal', f'其余 {hidden} 种组合至少差一个目标') in r['note']
            and Z(lang, 'for each combination', '每种组合') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pin-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pin-rail .nx-why')].filter(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden').map(e=>e.textContent).join(' | ')")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__pin||!window.__pin.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=200, settle_=True):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait)
    if settle_: settle(pg)


def go_part(pg, i): click(pg, f'#pin-rail [data-go="{i}"]', 250)
# polish round 3, the first-time reader (2026-10-16): entering a part with a new replay showed the whole new task for one event, then
# shrank it and replayed. Clicks here and reads the replay in the same task as the click, before any frame can redraw it.
def click_now(pg, sel): return pg.evaluate("""s=>{document.querySelector(s).click(); const st=window.__pin.state();
  return {pos:st.pos, len:st.len, playing:st.playing, blk:document.querySelectorAll('#pin-ctx .pin-blk').length, li:document.querySelectorAll('#pin-log .pin-li:not(.none)').length};}""", sel)
def mode(pg, m): click(pg, f'#pin-app .modes button[data-mode="{m}"]', 350)


def slide(pg, sel, v):
    pg.evaluate("([s,v])=>{const i=document.querySelector(s); i.value=v; i.dispatchEvent(new Event('input',{bubbles:true})); i.dispatchEvent(new Event('change',{bubbles:true}));}", [sel, v]); pg.wait_for_timeout(250); settle(pg)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def obs(pg): return pg.evaluate("()=>{const o=document.querySelector('#pin-rail [data-obs]'); return o&&!o.hidden?o.innerText.replace(/\\u00a0/g,' '):''}")
def badges(pg): return pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric .d')].map(d=>d.textContent).filter(Boolean)")
def live_ctl(pg): return pg.evaluate("()=>['inp','files','send','det','prm','scope','rule','f','d','fa','reset'].filter(k=>{const e=document.getElementById('pin-w-'+k); return e&&e.getClientRects().length>0})")


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
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, #pin-mathbody')||!e.getClientRects().length) continue;
   const m=n.textContent.match(/\\d{1,3}(?:,\\d{3}){2,}/); if(m) out.push(m[0]+' in: '+n.textContent.trim().slice(0,90));}
 return out;}"""
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|pin-over|pin-bad|harm|cmp-|ex-res|wrong|pin-hid|pin-sw-h|pin-hx|pin-tag/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#pin-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#pin-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
        for h in pg.evaluate(BIGNUM, sel): rule_hits.append(f'{where}: more than 6 digits (R6.18): {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')


def before_answer(pg):
    return pg.evaluate("""()=>{const t=[]; for(const s of ['#pin-app .labhead','#pin-rail','#pin-stage','#pin-controls','#dock']){const e=document.querySelector(s); if(e&&e.getClientRects().length){t.push(e.innerText);
        e.querySelectorAll('[data-tipx]').forEach(x=>t.push(x.getAttribute('data-tipx')));}}
      document.querySelectorAll('#pin-app .labhead [data-pop]').forEach(b=>{const p=document.getElementById(b.getAttribute('data-pop')); if(p) t.push(p.textContent);}); return t.join('\\n')}""")


def first_diff(pg, a, b):
    """The first event of the replayed task that differs between two setups (the system prompt only lists the tools: skipped)."""
    ea, eb = pg.evaluate("([a,b])=>[window.__pin.run(a).sampleEv, window.__pin.run(b).sampleEv]", [a, b])
    for i in range(1, min(len(ea), len(eb))):
        if ea[i] != eb[i]: return i
    return -1 if len(ea) == len(eb) else min(len(ea), len(eb))


def red(pg): return pg.evaluate("""()=>({hid:document.querySelectorAll('#pin-ctx mark.pin-hid').length, tag:[...document.querySelectorAll('#pin-ctx .pin-tag')].map(e=>e.textContent),
    blk:document.querySelectorAll('#pin-ctx .pin-blk.harm').length, li:document.querySelectorAll('#pin-log .pin-li.harm').length,
    col:(()=>{const m=document.querySelector('#pin-ctx mark.pin-hid'); return m?getComputedStyle(m).color:''})(), oom:(()=>{const d=document.createElement('i'); d.style.color='var(--c-oom)'; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c})(),
    now:document.querySelectorAll('#pin-app .pin-now').length})""")


def heights(pg): return pg.evaluate("()=>['#pin-stage','#pin-graph','#pin-ctx','#pin-log','#pin-grid'].map(s=>document.querySelector(s).offsetHeight)")


LEAKS = ('10 times', '10 harmful', 'acted on it', 'right answer', '照做了 10', '有害 10', '有害操作 10', '正确答案', 'files left', '文件漏了出去', 'call:harm')


def run(b, w, h, lang):
    tag = f'{w}-{lang}'
    pg = page(b, w, h, lang)
    ov = lambda where: check(f'{tag} no horizontal overflow: {where}', overflow(pg) <= 0, overflow(pg))
    st = state(pg)
    check(f'{tag} opens on part 1, no prediction, not playing, the setup shown', st['step'] == 0 and st['pred'] is None and not st['playing'] and st['pos'] == 4, st)
    ov('first screen')
    if w < 700:
        fs = pg.evaluate("()=>{const nx=document.querySelector('#pin-rail .pbar [data-nav=\"next\"]'), g=document.querySelector('#pin-graph'); const a=nx?nx.getBoundingClientRect():null, c=g.getBoundingClientRect(); return {next:!!a&&a.bottom<=innerHeight&&a.width>0, rp:c.top<innerHeight-120}}")
        check(f'{tag} phone first screen: the parts bar with Next and the picture (R7.3)', fs['next'] and fs['rp'], fs)
    shot(pg, f'{tag}-01-first')
    # ---- part 1: the gate ----
    check(f'{tag} part 1: Next disabled before a prediction', all(nexts(pg)), nexts(pg))
    check(f'{tag} part 1: why Next is disabled', Z(lang, 'Pick an answer', '选一个答案') in why(pg), why(pg))
    ba = before_answer(pg)
    for leak in LEAKS:
        check(f'{tag} part 1: nothing gives the answer before the prediction ({leak})', leak not in ba)
    check(f'{tag} part 1: the replay stops after the setup (no action after the read)', pg.evaluate("()=>document.querySelectorAll('#pin-log .pin-li').length") == 1 and pg.evaluate("()=>document.querySelectorAll('#pin-log .pin-li.harm,#pin-log .pin-li.inj').length") == 0)
    check(f'{tag} part 1: the play button is off before the prediction', pg.evaluate("()=>document.querySelector('#pin-play').disabled"))
    rd = red(pg)
    check(f'{tag} R5.11: the setup shows the injected instruction in red with its word, and no harmful action yet', rd['hid'] == 1 and rd['col'] == rd['oom'] and any(Z(lang, '⚠ injected instruction', '⚠ 注入的指令') in t for t in rd['tag']) and rd['blk'] == 0 and rd['li'] == 0, rd)
    rules(pg, f'{tag} part 1 before', lang)
    click(pg, '#pin-rail [data-pred="b"]', 150, settle_=False)
    st = state(pg)
    check(f'{tag} part 1: a pick starts the replay (auto-play)', st['pred'] == 'b' and st['playing'], st)
    check(f'{tag} part 1: “Saved” after the pick', Z(lang, 'Saved', '已记下') in text(pg, '#pin-rail .step.current'))
    check(f'{tag} part 1: Next still waits for the replay', all(nexts(pg)) and Z(lang, 'Wait for the replay', '等演示放完') in why(pg), (nexts(pg), why(pg)))
    hs = []
    for _ in range(5): hs.append(heights(pg)); pg.wait_for_timeout(220)
    check(f'{tag} R7.9: the stage keeps its height while the replay plays', all(x == hs[0] for x in hs), hs)
    check(f'{tag} part 1: no result while the replay plays', all(x not in before_answer(pg) for x in ('right answer', '正确答案', '10 harmful', '有害 10')))
    shot(pg, f'{tag}-02-running')
    # speeds: fast is 0.5 s an event, medium 1.8 s, slow 3 s
    click(pg, '#pin-player [data-act="play"]', 30, settle_=False)
    slide(pg, '#pin-scrub', 4)
    check(f'{tag} the scrubber moves the replay', state(pg)['pos'] == 4 and not state(pg)['playing'], state(pg))
    click(pg, '#pin-player .pin-speed button[data-v="f"]', 30, settle_=False)
    click(pg, '#pin-play', 30, settle_=False)
    p0 = state(pg)['pos']; pg.wait_for_timeout(1250); p1 = state(pg)['pos']
    check(f'{tag} fast: about 2 events a second', 2 <= p1 - p0 <= 3, (p0, p1))
    click(pg, '#pin-play', 30, settle_=False); slide(pg, '#pin-scrub', 4)
    click(pg, '#pin-player .pin-speed button[data-v="s"]', 30, settle_=False)
    click(pg, '#pin-play', 30, settle_=False)
    p0 = state(pg)['pos']; pg.wait_for_timeout(1250); p1 = state(pg)['pos']
    check(f'{tag} slow: no event in the first 1.25 s', p1 - p0 == 0, (p0, p1))
    click(pg, '#pin-player .pin-speed button[data-v="m"]', 30, settle_=False)
    click(pg, '#pin-play', 30, settle_=False); s1 = state(pg); pg.wait_for_timeout(400); s2 = state(pg)
    check(f'{tag} pause stops the replay', not s1['playing'] and s1['pos'] == s2['pos'], (s1['pos'], s2['pos']))
    click(pg, '#pin-player [data-act="stepf"]', 50, settle_=False)
    check(f'{tag} one step forward', state(pg)['pos'] == s2['pos'] + 1)
    hix = pg.evaluate("()=>window.__pin.run({}).sampleEv.indexOf('call:harm')")
    slide(pg, '#pin-scrub', hix)
    rd = red(pg)
    check(f'{tag} R5.11: one event before the harmful send, nothing harmful is red yet', rd['blk'] == 0 and rd['li'] == 0, (hix, rd))
    click(pg, '#pin-player [data-act="stepf"]', 50, settle_=False)
    rd = red(pg)
    check(f'{tag} R5.11: the step that reaches it marks the harmful send red in both boxes, with ✗ and a word, and pulses once', rd['blk'] == 1 and rd['li'] == 1 and any(Z(lang, '✗ harmful action', '✗ 有害操作') in t for t in rd['tag']) and rd['now'] >= 1, rd)
    click(pg, '#pin-player [data-act="end"]', 250)
    st = state(pg)
    check(f'{tag} part 1: to the end shows the whole task and enables Next', st['runsSeen'] and st['pos'] == st['len'] and not any(nexts(pg)), (st, nexts(pg)))
    o = obs(pg)
    check(f'{tag} part 1: the result, the verdict with the right answer, and why', Z(lang, 'acted on it 10 times', '照做了 10 次') in o and Z(lang, 'The right answer is “In at least some', '正确答案是“至少有一部分任务里') in o and Z(lang, 'Why', '原因') in o, o)
    check(f'{tag} part 1: the harmful send in the log, in red', pg.evaluate("()=>document.querySelectorAll('#pin-log .pin-li.harm').length") == 1)
    rd = red(pg)
    check(f'{tag} R5.11: the finished replay keeps both red marks', rd['hid'] == 1 and rd['blk'] == 1 and rd['li'] == 1, rd)
    pb = pg.evaluate("()=>{const b=document.querySelector('#pin-play'); return [b.textContent, b.dataset.tipx||'']}")
    check(f'{tag} R3.20: at the end the play button is the replay control, and says so', Z(lang, 'Replay', '重放') in pb[0] and Z(lang, 'again from the start', '从头再放一遍') in pb[1], pb)
    sc = pg.evaluate("()=>['#pin-ctx','#pin-log'].map(s=>{const e=document.querySelector(s); return e.scrollHeight-e.clientHeight-e.scrollTop})")
    check(f'{tag} S17: at the end of the replay both boxes show their newest block', all(x <= 1 for x in sc), sc)
    check(f'{tag} part 1: the grid shows 10 harmful cells', pg.evaluate("()=>document.querySelectorAll('#pin-grid .pin-c.harm').length") == 10)
    check(f'{tag} part 1: the note no longer offers a change once the replay is done', Z(lang, 'You can still change it', '还可以改') not in text(pg, '#pin-rail .step.current'))
    rules(pg, f'{tag} part 1 result', lang); ov('part 1 result')
    shot(pg, f'{tag}-03-p1-result', full=(w > 700))
    # ---- part 2 ----
    s_end = state(pg)
    go_part(pg, 1)
    st = state(pg)
    check(f'{tag} R3.20: entering part 2 does not replay the run just watched', not st['playing'] and st['pos'] == st['len'] and st['from'] == s_end['from'], st)
    click(pg, '#pin-rail [data-nav="back"]', 250)
    st = state(pg)
    check(f'{tag} R3.20: back in part 1 the replay shows its end at once and Next stays open', st['step'] == 0 and not st['playing'] and st['pos'] == st['len'] and not any(nexts(pg)), (st, nexts(pg)))
    pg.evaluate("()=>document.querySelector('#pin-play').click()"); pg.wait_for_timeout(120)
    st = state(pg)
    check(f'{tag} R3.20: the replay control plays the task again from the start', st['playing'] and st['from'] == 4, st)
    click(pg, '#pin-player [data-act="end"]', 120)
    go_part(pg, 1)
    check(f'{tag} part 2: Next disabled until the model view', all(nexts(pg)) and Z(lang, 'Switch to the model', '切到模型读到的样子') in why(pg), why(pg))
    check(f'{tag} part 2: Settings shows nothing to change', live_ctl(pg) == [], live_ctl(pg))
    click(pg, '#pin-rail .seg-ctl[data-key="view"] button[data-v="text"]')
    o = obs(pg)
    check(f'{tag} part 2: one text, Next enabled', Z(lang, 'one text', '一整段文字') in o and not any(nexts(pg)) and pg.evaluate("()=>!!document.querySelector('#pin-ctx.pin-one .pin-stream')"), o)
    check(f'{tag} part 2: links agent-anatomy', pg.evaluate("()=>[...document.querySelector('#pin-rail [data-obs]').querySelectorAll('a')].some(a=>(a.getAttribute('href')||'').includes('agent-anatomy'))"))
    rules(pg, f'{tag} part 2', lang); ov('part 2'); shot(pg, f'{tag}-04-p2', full=(w > 700))
    # ---- part 3 ----
    go_part(pg, 2)
    check(f'{tag} part 3: Next disabled until a tool goes off', all(nexts(pg)) and Z(lang, 'Turn one tool off', '关掉一个工具') in why(pg), why(pg))
    check(f'{tag} part 3: no ▲▼ before the learner acts', not badges(pg), badges(pg))
    check(f'{tag} part 3: Settings shows the three tools', live_ctl(pg) == ['inp', 'files', 'send', 'reset'], live_ctl(pg))
    st = state(pg)
    check(f'{tag} R3.20: entering part 3 (same task as part 2) does not replay', not st['playing'] and st['pos'] == st['len'], st)
    click(pg, '#pin-rail .seg-ctl[data-key="send"] button[data-v="0"]', 150, settle_=False)
    st = state(pg); k = first_diff(pg, {}, {'send': 0})
    check(f'{tag} R3.20: email off changes the task from the blocked send on: it replays from there only', st['playing'] and k > 2 and st['from'] == k and st['pos'] >= k, (st, k))
    settle(pg)
    o = obs(pg)
    check(f'{tag} part 3: email off: 10 to 0, 25 tasks lost', Z(lang, 'from 10 to 0', '从 10 次降到 0 次') in o and '25' in o and not any(nexts(pg)), o)
    check(f'{tag} part 3: ▲▼ after the learner acts', bool(badges(pg)), badges(pg))
    check(f'{tag} part 3: the chip turns off', pg.evaluate("()=>document.querySelector('#pin-tools .pin-chip[data-k=\"send\"]').classList.contains('off')"))
    if w > 700:   # R7.12 moved the cards under the picture; the result in view is the status chip and the replay (R3.10)
        mr = pg.evaluate("()=>{const r=s=>document.querySelector(s).getBoundingClientRect(); const st=r('#pin-status'), rp=r('#pin-rp'), m=r('#pin-metrics'); return [st.top, st.bottom, rp.top, rp.bottom, m.top, innerHeight]}")
        check(f'{tag} after the change in part 3 the status chip and the replay are on screen; the cards sit under the picture', mr[0] >= 56 and mr[3] <= mr[5] and mr[4] > mr[3], mr)
    rules(pg, f'{tag} part 3', lang); ov('part 3'); shot(pg, f'{tag}-05-p3', full=(w > 700))
    click(pg, '#pin-rail .seg-ctl[data-key="send"] button[data-v="1"]', 150, settle_=False)
    st = state(pg)
    check(f'{tag} R3.20: email back on brings back the run already watched: no replay, its end at once', not st['playing'] and st['pos'] == st['len'], st)
    # ---- part 4 ----
    go_part(pg, 3)
    check(f'{tag} part 4: Next disabled until a choice', all(nexts(pg)) and Z(lang, 'Pick a defense', '选一种防护') in why(pg), why(pg))
    check(f'{tag} part 4: Settings shows the defenses', live_ctl(pg) == ['det', 'prm', 'scope', 'rule', 'reset'], live_ctl(pg))
    click(pg, '#pin-player [data-act="end"]', 120)
    s0 = state(pg)
    click(pg, '#pin-rail [data-lever="det"]', 150, settle_=False)
    st = state(pg)
    check(f'{tag} R3.20: the detector lets the replayed page through, so the task is the same: no replay', first_diff(pg, {}, {'det': 1}) == -1 and not st['playing'] and st['pos'] == st['len'] and st['from'] == s0['from'], st)
    settle(pg)
    check(f'{tag} part 4: the detector: 10 to 2, 82 done', Z(lang, 'from 10 to 2', '从 10 次降到 2 次') in obs(pg) and '82' in obs(pg) and not any(nexts(pg)), obs(pg))
    click(pg, '#pin-rail [data-lever="scope"]', 150, settle_=False)
    st = state(pg); k = first_diff(pg, {'det': 1}, {'scope': 1})
    check(f'{tag} R3.20: least privilege blocks the injected read: it replays from that event', st['playing'] and k > 2 and st['from'] == k, (st, k))
    settle(pg)
    check(f'{tag} part 4: least privilege: 0, all done', Z(lang, 'from 10 to 0, and all 100 tasks', '从 10 次降到 0 次，100 个任务照样全做成') in obs(pg), obs(pg))
    check(f'{tag} part 4: single choice', pg.evaluate("()=>[...document.querySelectorAll('#pin-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').length") == 1)
    lr = pg.evaluate("()=>[...document.querySelectorAll('#pin-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag} part 4: results of the tried options', lr[0] and lr[1] and lr[3] and not lr[2] and not lr[4], lr)
    click(pg, '#pin-rail [data-lever="ask"]', 150, settle_=False)
    bd = pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric')].filter(m=>m.querySelector('.d').textContent).map(m=>m.dataset.k)")
    check(f'{tag} S18: right after another change, only the card whose value changed has a ▲▼ badge', bd == ['ask'], bd)
    settle(pg)
    rules(pg, f'{tag} part 4', lang); ov('part 4'); shot(pg, f'{tag}-06-p4', full=(w > 700))
    # ---- part 5 ----
    r5 = click_now(pg, '#pin-rail [data-go="4"]'); pg.wait_for_timeout(250); settle(pg)
    check(f'{tag} part 5: the inbox replay starts over and its box holds only the events played so far (no end shown first)', r5['playing'] and r5['pos'] < r5['len'] and r5['blk'] == r5['pos'] and r5['li'] == 0, r5)
    cur = text(pg, '#pin-rail .step.current'); cap = text(pg, '#pin-rp-cap')
    papers = pg.evaluate("()=>[...document.querySelectorAll('ul.refs li')].map(l=>l.textContent).join(' | ')")
    # round 3: the card introduces the inbox assistant and names the replayed task as its caption does; CaMeL moved to the papers list
    task = Z(lang, 'Reply to a colleague', '回复同事')
    check(f'{tag} part 5: five practices; the card introduces the inbox assistant and its replayed task; CaMeL is in the papers list; Next enabled', pg.evaluate("()=>document.querySelectorAll('#pin-rail .pin-teams li').length") == 5 and Z(lang, 'The chart now shows an inbox assistant', '图里换成了邮件助手') in cur and task in cur and task in cap and 'read_files' in cur and 'CaMeL' in papers and not any(nexts(pg)), (cur[:300], cap, papers[-300:]))
    rules(pg, f'{tag} part 5', lang); ov('part 5'); shot(pg, f'{tag}-07-p5', full=(w > 700))
    # ---- part 6 ----
    go_part(pg, 5)
    st = state(pg)
    check(f'{tag} R3.20: part 6 brings back the page agent’s run already watched: no replay, its end at once', st['wl'] == 'page' and not st['playing'] and st['pos'] == st['len'], st)
    r6 = click_now(pg, '#pin-play'); pg.wait_for_timeout(120)
    st = state(pg)
    check(f'{tag} R3.20: ▶ Replay still plays a run already seen, from the request', st['playing'] and st['from'] == 2, st)
    check(f'{tag} ▶ Replay: the box goes back to the events played so far at once (round 3)', r6['playing'] and r6['pos'] < r6['len'] and r6['blk'] == r6['pos'], r6)
    click(pg, '#pin-player [data-act="end"]', 120)
    cur = text(pg, '#pin-rail .step.current')
    check(f'{tag} part 6: three points, three misconceptions, See the scenarios', cur.count('✗') == 3 and pg.evaluate("()=>document.querySelectorAll('#pin-rail .pin-remlist li').length") == 3 and pg.evaluate("()=>!!document.querySelector('#pin-rail [data-mode-go=\"sandbox\"]')"), cur[:200])
    check(f'{tag} part 6: the next-lab footer', pg.evaluate("()=>!!document.querySelector('#pin-app .next-lab')"))
    rules(pg, f'{tag} part 6', lang); ov('part 6'); shot(pg, f'{tag}-08-p6', full=(w > 700))
    # ---- scenarios ----
    click(pg, '#pin-player [data-act="end"]', 120)
    s0 = state(pg)
    mode(pg, 'sandbox')
    # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
    _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#pin-rail')
    _lead = Z(lang, 'Pick a scenario and see what a hidden line can make its agent do.', '选一个场景，看看藏着的话能让它的 agent 做出什么事。')
    check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
    check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
    # end G1 G2 scen
    st = state(pg)
    check(f'{tag} R3.20: Scenarios open on the same task as part 6: no replay', not st['playing'] and st['pos'] == st['len'] and st['from'] == s0['from'], st)
    check(f'{tag} scenarios: four', pg.evaluate("()=>document.querySelectorAll('#pin-rail [data-preset]').length") == 4)
    click(pg, '#pin-rail [data-preset="1"]', 150, settle_=False)
    st = state(pg)
    check(f'{tag} R3.20: another agent is another task: it replays', st['playing'] and st['wl'] == 'mail' and st['from'] == 2, st)
    settle(pg); click(pg, '#pin-player [data-act="end"]', 120)
    pr = pg.evaluate("()=>[...document.querySelectorAll('#pin-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: single choice, pressed follows the settings', pr == ['false', 'true', 'false', 'false'], pr)
    check(f'{tag} scenarios: the inbox assistant’s result', Z(lang, '10 harmful', '有害操作 10 次') in text(pg, '#pin-insight'), text(pg, '#pin-insight'))
    check(f'{tag} scenarios: every Settings control is live', live_ctl(pg) == ['inp', 'files', 'send', 'det', 'prm', 'scope', 'rule', 'f', 'd', 'fa', 'reset'], live_ctl(pg))
    s0 = state(pg)
    click(pg, '#pin-ctl-det button[data-v="1"]')
    st = state(pg)
    check(f'{tag} R3.20: a detector that misses the replayed email leaves the task as it was: no replay', not st['playing'] and st['pos'] == st['len'] and st['from'] == s0['from'], st)
    pr = pg.evaluate("()=>[...document.querySelectorAll('#pin-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
    check(f'{tag} scenarios: a change under Settings unpresses the scenario', pr == ['false'] * 4, pr)
    pg.evaluate("()=>document.querySelector('#pin-ctl-f-in').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    y0 = pg.evaluate("()=>document.querySelector('#pin-ctl-f-in').getBoundingClientRect().top")
    slide(pg, '#pin-ctl-f-in', 14)
    y1 = pg.evaluate("()=>document.querySelector('#pin-ctl-f-in').getBoundingClientRect().top")
    check(f'{tag} R3.2: a Settings change keeps the slider in place', abs(y1 - y0) < 2, (y0, y1))
    hsh = pg.evaluate("()=>location.hash")
    rules(pg, f'{tag} scenarios', lang); ov('scenarios'); shot(pg, f'{tag}-09-scenarios', full=(w > 700))
    pg2 = page(b, w, h, lang, hash_=hsh.lstrip('#'))
    s2 = state(pg2)
    check(f'{tag} the URL restores the setup', s2['mode'] == 'sandbox' and s2['wl'] == 'mail' and s2['det'] == 1 and s2['f'] == 0.7, (hsh, s2))
    pg2.close()
    # ---- challenge ----
    s0 = state(pg)
    mode(pg, 'challenge')
    st = state(pg)
    check(f'{tag} R3.20: the challenge starts on the task already on screen: no replay', not st['playing'] and st['from'] == s0['from'], st)
    check(f'{tag} challenge: Settings locked with a note', live_ctl(pg) == [] and text(pg, '#pin-lock') != '', live_ctl(pg))
    g = text(pg, '#pin-ch-g1') + text(pg, '#pin-ch-g2')
    check(f'{tag} challenge: live progress (10 harmful, 100 done)', Z(lang, '10 times', '10 次') in g and '100' in g, g)
    click(pg, '#pin-rail [data-act="submit"]', 250)
    res = text(pg, '#pin-ch-res')
    check(f'{tag} challenge tier 1: not yet, a hint, Show the answer', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res and pg.evaluate("()=>!!document.querySelector('#pin-ch-res [data-act=\"reveal\"]')")
          and text(pg, '#pin-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), res)
    click(pg, '#pin-ch-res [data-act="reveal"]', 200)
    bf = brief(pg)
    check(f'{tag} challenge: Show the answer draws 5 rows, the 4 passing combinations and the initial setup (the pick), 27 left out', brief_ok(lang, bf, 5, 27), bf)
    click(pg, '#pin-rail .seg-ctl[data-key="chR"] button[data-v="askAll"]', 150)
    st = state(pg)
    check(f'{tag} R3.20: asking before every send changes the task: it replays from the change', st['playing'] and st['from'] > 2, st)
    res = text(pg, '#pin-ch-res')
    check(f'{tag} challenge tier 2: met, but one that asks less passes (17)', Z(lang, 'Goal met', '达成目标') in res and Z(lang, 'asks less passes too', '更省事的组合') in res and '17' in res, res)
    check(f'{tag} challenge tier 2: the first line says that a combination that asks less passes', Z(lang, 'But a combination that asks less passes too', '不过还有更省事的组合也能达标') in text(pg, '#pin-ch-res > p:first-child'), text(pg, '#pin-ch-res > p:first-child'))
    bf = brief(pg)
    check(f'{tag} challenge tier 2: the chart holds the 4 passing combinations (the pick among them) and the initial setup, 27 left out', brief_ok(lang, bf, 5, 27), bf)
    click(pg, '#pin-rail .seg-ctl[data-key="chD"] button[data-v="1"]', 120)
    click(pg, '#pin-rail .seg-ctl[data-key="chR"] button[data-v="allowAsk"]', 150)
    res = text(pg, '#pin-ch-res')
    check(f'{tag} challenge tier 3: the minimal one', Z(lang, 'no passing combination asks the user less often', '没有哪个达标的组合问用户的次数更少') in res and '17' in res, res)
    rows = pg.evaluate("()=>document.querySelectorAll('#pin-ch-res .cmp-row').length")
    bf = brief(pg)
    check(f'{tag} challenge: the chart holds only the 4 passing combinations (the pick among them) and the initial setup, 27 left out', rows == 5 and brief_ok(lang, bf, 5, 27), (rows, bf))
    check(f'{tag} challenge: the chart tags the initial setup and the pick', all(x in res for x in Z(lang, ('initial setup', 'your pick'), ('初始设置', '你的选择'))), res[:300])
    click(pg, '#pin-rail .seg-ctl[data-key="chR"] button[data-v="free"]', 150)
    bf = brief(pg)
    check(f'{tag} challenge: a pick that misses (a detector, sending freely) adds its own row, 6 rows, 26 left out', brief_ok(lang, bf, 6, 26), bf)
    click(pg, '#pin-rail .seg-ctl[data-key="chR"] button[data-v="allowAsk"]', 150)
    res = text(pg, '#pin-ch-res')
    check(f'{tag} challenge: the capability card', pg.evaluate("()=>!!document.querySelector('#pin-rail [data-exits]')"))
    rules(pg, f'{tag} challenge', lang); ov('challenge'); shot(pg, f'{tag}-10-challenge', full=(w > 700))
    other = 'en' if lang == 'zh' else 'zh'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    s3 = state(pg)
    check(f'{tag} the language switch keeps the answer', s3['mode'] == 'challenge' and s3['chO'] == {'files': 1, 'send': 1, 'det': 1, 'rule': 'allowAsk'} and s3['chSubmitted'], s3)
    # ---- the dock at 390 ----
    if w < 700:
        pg.evaluate("()=>{const t=document.querySelector('#pin-rail [data-exits]')||document.querySelector('#pin-ch-res'); scrollTo(0, t.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(700)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d&&!d.hidden&&d.getClientRects().length?d.innerText:''}")
        check(f'{tag} the dock shows once the chart scrolls away', bool(dk) and Z(lang, 'harmful', '有害') in dk, dk)
        rules(pg, f'{tag} dock', lang)
        shot(pg, f'{tag}-11-dock')
    mode(pg, 'guided')
    st = state(pg)
    check(f'{tag} R3.20: leaving the challenge brings back the run already watched: no replay', st['mode'] == 'guided' and st['wl'] == 'page' and not st['playing'] and st['pos'] == st['len'], st)
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
_H1_LAB, _H1_PFX = 'prompt-injection', 'pin'
_H1_PLAIN = ['g_files', 'g_send', 'g_rule', 'N_send', 'N_tried', '× (1 −']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ["tx('g_files (the task can read the files)'", "'N_send (ordinary sends that wait for a yes)'", 'On average: ${expr}', '${expr} = ${fN(ex)}', ': ${nS} + ${d.res.att} = ${d.res.ask}', '的个数']     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['pin-mb', 'pin-eqs', 'symHTML']       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#pin-tools', '#pin-rp-cap', '.pin-gridw']
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
_FIT_PARTS = [1, 2, 3, 4, 5, 6]
_FIT_V = "(()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); return [...v.querySelectorAll('.bench')].find(x=>x.offsetParent)||v;})()"
_FIT_M = ("(()=>{const b=" + _FIT_V + ", s=b.querySelector(':scope>.stage'), cur=b.querySelector('.steps>.step.current'), r=s.getBoundingClientRect(), tb=(document.querySelector('.topbar')||{offsetHeight:0}).offsetHeight; let inner=0;"
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
            _pg.evaluate("()=>{location.hash='prompt-injection'}"); _pg.wait_for_timeout(500)
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
_src = (P / 'labs' / 'prompt-injection.js').read_text() + (P / 'teasers' / 'prompt-injection.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['<p>选一种防护，看看有害操作还剩几次。</p>', '<p>Pick a defense and see how many harmful actions are left.</p>', '这段文字里，没有什么能可靠地标出哪句是命令。所以读不可信内容的 agent，总有一些时候会照做。', '数据要漏出去，同一个任务里三样缺一不可。去掉任何一样，agent 照样可能上当，但东西出不去。', '检测器和提示词里的叮嘱，只能让 agent 少照做几次。没有哪个任务既要读文件、又要发邮件时，最小权限就能让照做了也不出事。真有这样的任务，就要靠用户点头来把关。', '这样的叮嘱只能让有害操作少一半左右。', '检测器会漏掉一些藏着的话，还会扣下一些用户想看的网页。', '20 个藏了话的任务里，agent 照着藏着的话做了 ${b.att} 次。它为什么会听网页里的话？', '为什么关掉读文件或发邮件中的任何一个，有害操作就降到 0 次？', '这就是靠设计来防，就算模型照做了，危险的那一步也走不通。']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['（第 5 部分的图里能看到）', '就像第 4 部分那样', '第 1 部分里，20 个藏了话的任务', '第 3 部分里，关掉读文件', '第 4 部分里，用了检测器', '第 4 部分里，这样的叮嘱', '第 1 部分里，20 个任务中', '这正是这个 Lab 的第 4 部分', '(the chart in part 5 shows it)', 'as in part 4', 'In part 1, the agent acted', 'In part 3, turning off', 'In part 4, a detector', 'In part 4 such a warning', 'In part 1 the agent acted', 'In part 3, why did', 'this lab’s part 4']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} passed')
sys.exit(1 if fails else 0)
