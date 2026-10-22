"""Flow test for the serving capstone (CONTRACT §6, the spec's test list).
- Every guided part, Scenarios and the challenge at 1440×900, 900×900 and 390×844, in EN and ZH, light, plus one dark run: no page
  errors, no horizontal overflow, the writing rules on every visible text and tooltip (no “·” between clauses, at most one “：” per Chinese
  sentence, a space between Chinese and Latin letters or digits, full-width punctuation, no “toy” but the fidelity name, no 起点 / 控件 /
  “Locked in”, no broken values).
- The header: the catalog's question, “Fidelity: Toy run”, the outcome line; the next-lab footer on the last part, in Scenarios and the
  challenge.
- The prediction (R3.8): Next is disabled until a pick; the answer (“the whole prompt went into one step”) appears nowhere before the pick;
  after it the verdict, and Show me puts the playhead in the step behind the stall (the strip shows the document's block).
- Playback: Play advances, Pause stops, ▸ lands on a step's end, ◂ on a step's start, Replay starts from 0; 1× plays a run in about 15 s
  and 2× twice as fast; the speed survives Show me. Results wait for the end of the run (R3.8): the table's “Yours” column and the lever's
  result.
- Reduced motion: no autoplay (a changed option shows the whole run at once), Play shows the whole run, ▸ and the slider still move the
  playhead.
- Rapid changes never leave an old animation drawing: after a burst of option clicks the playhead advances at the 1× rate (one loop), a
  language switch and leaving the lab stop it, and the run token changes with every state change.
- Tracing: a chip, a timeline row and a memory block select a request; the trace names it; the chip and the block are outlined.
- Phones (m2): the first screen shows the strip, the timeline and a control; a lever tap keeps the lever in place; the mini result bar
  appears when the strip scrolls away, with a short status.
- The challenge: the load and goals before the settings; tiers 1 (fewest changes), 2 (fewer changes pass) and 3 (hint, “Show the answer”
  → a brief chart: the named settings that pass, the pick and the initial setup, with a note counting the rest); the plan's Copy button
  puts the plan on the clipboard; the capability card.
- Scenarios: three presets, pressed state follows the load; a Settings change does not move the page (R3.2); the URL carries the setup
  and a fresh page opens it (hash round-trip). A language switch keeps the part, the prediction and the settings.
- The stage fits a 1440 × 900 window in every guided part but part 7 (the owner, 2026-10-22): the whole-run table is folded under a
  button in Show the math's row; the button and the cost sentence's link open it whole on the screen; the choice holds across Next, a
  language switch and a reload; phones have neither the button nor the table.
Screenshots: out/sc_shots/ (the ones the task lists) and shots/serving-capstone/.
Run: python3 tests/serving-capstone_flow.py"""
import pathlib, re, subprocess, sys, json, time
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/serving-capstone.html', '--only', 'serving-capstone'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-serving-capstone.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'serving-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-serving-capstone.html')
SH = P / 'shots' / 'serving-capstone'; SH.mkdir(parents=True, exist_ok=True)
SC = P / 'out' / 'sc_shots'; SC.mkdir(parents=True, exist_ok=True)
errs, fails, n = [], [], 0


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(name); print('FAIL', name, '::', str(info)[:400])


NB = lambda s: s.replace(' ', ' ').replace('‑', '-').replace('\xa0', ' ')
Z = lambda lang, en, zh: zh if lang == 'zh' else en


def page(b, w, h, lang, dark=False, reduced=False, drawers=True, hash_='serving-capstone', perms=False):
    ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference')
    if perms: ctx.grant_permissions(['clipboard-read', 'clipboard-write'])
    pg = ctx.new_page()
    if drawers: open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


st = lambda pg: pg.evaluate("()=>window.__sc.state")
click = lambda pg, sel: pg.evaluate("s=>document.querySelector(s).click()", sel)
overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__sc.state.stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def act(pg, sel, wait=250):
    click(pg, sel); pg.wait_for_timeout(wait); settle(pg)


def end(pg):
    pg.evaluate("()=>{const b=document.querySelector('[data-act=toend]'); if(b){b.click(); return;} const i=document.querySelector('#sc-scrub'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)


def go(pg, k): act(pg, f'#sc-rail [data-go="{k}"]', 350)


def mode(pg, m): act(pg, f'#sc-app .modes button[data-mode="{m}"]', 450)


def shot(pg, path, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(path), full_page=full)


def all_text(pg):
    return NB(pg.evaluate("""()=>{const out=[]; for(const s of ['#sc-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#sc-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}"""))


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'“Locked in”'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万 instead of K, M, B (R6.18)'],[/(?<![这那两])个数(?![值量])/,'“个数” (R6.17)'],[/\\ba 8/,'“a 8…” (an)'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b/,'a company name'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), bad=/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/;
 const shown=e=>e&&!e.closest('[data-endnotes],pre,code,svg,.mathbody,#tip')&&e.getClientRects().length;
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){if(!shown(n.parentElement)) continue; const t=n.textContent.replace(/\\s+/g,' '); const m=t.match(new RegExp(bad.source,'g')); if(m) out.add(m.join('|')+' in: '+t.trim().slice(0,80));}
 root.querySelectorAll('a,.def,b,.t,em,code,button.linkbtn').forEach(e=>{if(!shown(e)||getComputedStyle(e).display!=='inline'&&!e.matches('button.linkbtn')) return;
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
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=new Set();
 const texts=[]; root.querySelectorAll('p,.lv-d,.lv-t>b,td,th,.sc-fb li,.sc-tr li,.sc-onote,.sc-clock,.sc-goalcard li,.choice,.preset span,.cmp-lab span,.sc-m2l,.ex-res p,.mode-desc,.pb-t').forEach(e=>{if(e.getClientRects().length&&!e.closest('li.step:not(.current)')) texts.push(e.innerText);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 texts.forEach(t=>{for(const s of t.replace(/\\s+/g,' ').split(/(?<=[。？！])/)){if(/[\\u4e00-\\u9fff]/.test(s)&&(s.match(/：/g)||[]).length>=2) out.add(s.trim().slice(0,120));}});
 return [...out];}"""


def rules(pg, tag, lang):
    r = pg.evaluate(RULES, '#sc-app'); check(f'{tag}: writing rules', not r, r[:4])
    if lang == 'zh':
        s = pg.evaluate(SPACING, '#sc-app'); check(f'{tag}: a space between Chinese and Latin letters or digits', not s, s[:4])
        pu = pg.evaluate(PUNCT, '#sc-app'); check(f'{tag}: full-width punctuation in Chinese', not pu, pu[:4])
        co = pg.evaluate(COLONS, '#sc-app'); check(f'{tag}: at most one “：” per Chinese sentence', not co, co[:4])


def visible_in(pg, sel, h):
    return pg.evaluate("([s,h])=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&r.top>=0&&r.top<h-8}", [sel, h])


WL_ORDER = ['h', 'def', 'bud', 'work', 'dl', 'di', 'da', 'pl', 'pi', 'pn', 'pa', 'al', 'ac', 'ab', 'lk', 'mc', 'mb', 'rf', 'rd', 'ad', 'an', 'aa', 'tm', 'fl', 'fr', 'un', 'ev', 'rt']


def panel(pg):
    return pg.evaluate("()=>({head:document.getElementById('sc-wct').textContent, hidden:document.getElementById('sc-wc').hidden, rows:[...document.querySelectorAll('#sc-wcb .sc-wl')].map(l=>({k:l.dataset.k, st:([...l.classList].filter(c=>!['sc-wl','key','sc-wlf'].includes(c))[0]||'plain'), v:l.querySelector('.sc-wv').textContent, tip:l.querySelector('.sc-wv').getAttribute('data-tipx')||''})), gpu:document.getElementById('sc-gh').textContent, code:window.__sc.codeStep()})")


KEYLN = {'usual': 'an', 'budget': 'ac', 'cache': 'lk', 'evict': 'ev', 'cap': 'ac'}
GEO7 = """()=>{const r=e=>{const b=document.querySelector(e).getBoundingClientRect(); return [b.top,b.bottom]}, bar=document.querySelector('.topbar').getBoundingClientRect().bottom, st=document.getElementById('sc-stage'), sticky=getComputedStyle(st).position==='sticky'&&innerWidth>760, sr=st.getBoundingClientRect();
 return {wc:r('#sc-wc'), strip:r('#sc-strip'), top:sticky?Math.max(sr.top,bar):bar, bot:sticky?Math.min(sr.bottom,innerHeight):innerHeight, dock:!document.getElementById('dock').hidden}}"""


def gpu_says(pn, n, done, lang):
    """the GPU box names step n (1-based), done or not: the same step as the panel (review A, finding 1)"""
    d = ('done' in pn['gpu']) if lang == 'en' else ('已结束' in pn['gpu'])
    return NB(f'{n:,}') in NB(pn['gpu']) and d == done


def walkthrough(pg, tag, w, h, lang):
    """Part 7 (product owner, 29 Sep): the scheduler's code on the stage, synced with the playhead; five moments found by rule.
    Review A (29 Sep): the GPU box shows the panel's step at its end too; the panel comes into view after a moment; values fit"""
    go(pg, 6)
    s0 = st(pg); p0 = panel(pg)
    if s0.get('speedPicked') is None:   # R3.9 (2026-10-04): the code, read step by step, plays at 1× unless the learner picked a speed
        check(f'{tag} part 7: the default speed is 1× (0.8 s a step)', s0['speed'] == 1 and pg.evaluate("()=>document.querySelector('.sc-speed button[aria-pressed=true]').dataset.v") == '1', s0['speed'])
    entry = (('Pick a moment on the card below' if w <= 760 else 'No step picked yet') if lang == 'en' else ('在下面的卡片里选一个时刻' if w <= 760 else '还没选中哪一步'))
    check(f'{tag} part 7: the part is “code”, the panel shows with no step picked yet, and Next waits for a moment', s0['partId'] == 'code' and not p0['hidden'] and p0['code'] is None
          and entry in p0['head'] and all(r['st'] in ('skip', 'plain') for r in p0['rows'])
          and pg.evaluate("()=>document.querySelector('#sc-rail .step.current [data-nav=next]').disabled"), (s0['partId'], p0['head']))
    if w > 760:
        pg.wait_for_timeout(500); g = pg.evaluate(GEO7)
        check(f'{tag} part 7: on entry the whole panel, its last line included, is on screen (review B, B1)', g['wc'][0] >= g['top'] - 1 and g['wc'][1] <= g['bot'] + 1, g)
    check(f'{tag} part 7: 28 lines, the Connect code’s statements line for line', len(p0['rows']) == 28 and [r['k'] for r in p0['rows']] == WL_ORDER
          and pg.evaluate("()=>{const a=window.__sc.wcode, c=window.__sc.code; return a.length===c.length&&a.every((x,i)=>i===0||x[1].split('#')[0].trim()===c[i][0].split('#')[0].trim())}"))
    # ▸ from the whole run: the end of step 1, in the panel and the GPU box (review A, finding 9)
    act(pg, '#sc-play [data-act=fwd]', 300); pn = panel(pg)
    check(f'{tag} part 7: ▸ from the whole run shows step 1 ended, in the panel and the GPU box', pn['code'] == {'s': 0, 'phase': 'end'} and gpu_says(pn, 1, True, lang), (pn['code'], pn['head'], pn['gpu']))
    moms = pg.evaluate("()=>window.__sc.moments()")
    check(f'{tag} part 7: five moments, each found in its run', [m['id'] for m in moms] == ['usual', 'budget', 'cache', 'evict', 'cap'] and all(m['s'] >= 0 for m in moms), moms)
    for m in moms:
        if w <= 760: pg.evaluate(f"()=>document.querySelector('#sc-rail [data-moment=\"{m['id']}\"]').scrollIntoView({{block:'center'}})"); pg.wait_for_timeout(150)
        act(pg, f'#sc-rail [data-moment="{m["id"]}"]', 450); pg.wait_for_timeout(500)
        pn = panel(pg); ss = st(pg)
        exp = pg.evaluate(f"()=>window.__sc.codeLines('{m['cfg']['load']}',{json.dumps(m['cfg'])},{m['s']},'{m['at']}')")
        rows = {r['k']: r for r in pn['rows']}
        if m['at'] == 'end':   # what the eviction moment shows happens at the step's end: the playhead is there, and the whole stage shows that step, done
            check(f'{tag} part 7 “{m["id"]}”: the setting and the playhead at the step’s end', ss['load'] == m['cfg']['load'] and ss['playhead'] is not None and abs(ss['playhead'] - m['t1']) < 1e-9 and not ss['playing'], (ss['load'], ss['playhead'], m))
            check(f'{tag} part 7 “{m["id"]}”: the panel and the GPU box show that step, ended', pn['code'] == {'s': m['s'], 'phase': 'end'} and NB(f'{m["s"] + 1:,}') in NB(pn['head']) and gpu_says(pn, m['s'] + 1, True, lang), (pn['code'], pn['head'], pn['gpu']))
        else:
            check(f'{tag} part 7 “{m["id"]}”: the setting and the playhead inside the step', ss['load'] == m['cfg']['load'] and ss['playhead'] is not None and m['t0'] < ss['playhead'] < m['t1'] and not ss['playing'], (ss['load'], ss['playhead'], m))
            check(f'{tag} part 7 “{m["id"]}”: the panel shows that step, running, the same step as the GPU box', pn['code'] == {'s': m['s'], 'phase': 'run'} and NB(f'{m["s"] + 1:,}') in NB(pn['head']) and gpu_says(pn, m['s'] + 1, False, lang), (pn['code'], pn['head'], pn['gpu']))
        bad = [k for k in WL_ORDER if rows[k]['st'] != exp[k]['st'] or NB(rows[k]['v']) != NB(exp[k]['v'])]
        check(f'{tag} part 7 “{m["id"]}”: every line’s state and value is the step’s (decided at its start ran; step_time now and the end lines dashed, or all of it at the end)', not bad and (rows['tm']['st'] == ('ran' if m['at'] == 'end' else 'now')) and all(rows[k]['st'] == ('ran' if m['at'] == 'end' else 'next') for k in ('fl', 'ev', 'rt')), [(k, rows[k], exp[k]) for k in bad][:3])
        stops = [k for k in WL_ORDER if rows[k]['st'] == 'stop']
        want = {'budget': ['ac', 'ab'], 'cap': ['ac', 'ab']}.get(m['id'], [])
        check(f'{tag} part 7 “{m["id"]}”: red lines only where admission stops ({want or "none"})', stops == want, stops)
        keys = pg.evaluate("()=>[...document.querySelectorAll('#sc-wcb .sc-wl.key')].map(l=>l.dataset.k)")
        check(f'{tag} part 7 “{m["id"]}”: the deciding line is drawn heavier ({KEYLN[m["id"]]})', keys == [KEYLN[m['id']]], keys)
        tips = [k for k in WL_ORDER if rows[k]['v'] and not rows[k]['tip']]
        check(f'{tag} part 7 “{m["id"]}”: every value has a sentence in its tooltip (R6.21)', not tips, tips)
        empt = [k for k in WL_ORDER if not rows[k]['v'] and rows[k]['tip']]
        check(f'{tag} part 7 “{m["id"]}”: no tooltip on an empty value cell', not empt, empt)
        clip = pg.evaluate("()=>[...document.querySelectorAll('#sc-wcb .sc-wl')].flatMap(l=>[...l.querySelectorAll('.sc-wx,.sc-wv')].filter(c=>c.scrollWidth>c.clientWidth+1).map(c=>l.dataset.i+':'+c.textContent.trim().slice(0,24)))")
        check(f'{tag} part 7 “{m["id"]}”: no code line and no value is cut off (R7.4, R7.8)', not clip, clip)
        g = pg.evaluate(GEO7)
        check(f'{tag} part 7 “{m["id"]}”: the whole panel is on screen, below the top bars, after the moment (R3.10)', g['wc'][0] >= g['top'] - 1 and g['wc'][1] <= g['bot'] + 1, g)
        if h >= 844: check(f'{tag} part 7 “{m["id"]}”: the GPU box is on screen with it', g['strip'][0] >= g['top'] - 1 and g['strip'][1] <= g['bot'] + 1 and not g['dock'], g)
        ob = NB(pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
        key = {'usual': ('the cap, the budget and memory', '上限、预算和显存三项检查'), 'budget': ('budget <= 0', 'budget <= 0'), 'cache': ('cache.lookup', 'cache.lookup'), 'evict': ('cache.evict_lru', 'cache.evict_lru'), 'cap': ('len(running) >= CAP', 'len(running) >= CAP')}[m['id']]
        check(f'{tag} part 7 “{m["id"]}”: the card says what happened, pointing at the deciding line', ob.lower().startswith('observed' if lang == 'en' else '观察到') and key[0 if lang == 'en' else 1] in ob and NB(f'{m["s"] + 1:,}') in ob, ob[:300])
        g2 = {'budget': ('This is how a new chat waits for the budget while a document prefills.', '文档在 prefill 时，新来的对话就是这样在等预算。'),
              'evict': ('which is what a cache budget that is too small costs.', '这就是缓存预算太小的代价。'),
              'cap': ('This is how a queue builds up when the cap is too low.', '上限设得太低时，排队就是这样来的。')}.get(m['id'])
        if g2: check(f'{tag} part 7 “{m["id"]}”: the card states the situation itself, not a part to recall (G2, the owner 2026-10-08)', NB(g2[0 if lang == 'en' else 1]) in ob and not re.search(r'[Pp]art \d|第 \d 部分', ob), ob[-240:])
        paras = pg.evaluate("()=>[...document.querySelectorAll('#sc-rail [data-obs] p')].filter(p=>!p.classList.contains('small')).map(p=>p.innerText)")
        nsent = sum(len([x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9“])|(?<=[。！？])', p_) if x and x.strip()]) for p_ in paras)
        check(f'{tag} part 7 “{m["id"]}”: the observation is a bold sentence and at most two more (review A, finding 7)', len(paras) == 2 and nsent <= 3, (nsent, paras))
        if m['id'] in ('budget', 'evict'): shot(pg, SC / f'p7_{m["id"]}_{w}_{lang}.png')
        rules(pg, f'{tag} part 7 {m["id"]}', lang)
    check(f'{tag} part 7: Next opens after a moment; every moment marked seen', not pg.evaluate("()=>document.querySelector('#sc-rail .step.current [data-nav=next]').disabled")
          and pg.evaluate("()=>[...document.querySelectorAll('#sc-rail [data-moment]')].every(b=>b.textContent.includes('✓'))"))
    # ◂ into the eviction's step: the eviction at its end has not happened yet, so the memory bar draws the document once, cached (review B, B2)
    act(pg, '#sc-rail [data-moment="evict"]', 450); act(pg, '#sc-play [data-act=back]', 300)
    ml = pg.evaluate("()=>({ev:[...document.querySelectorAll('#sc-mlab .sc-mb-ev')].map(e=>e.textContent), all:[...document.querySelectorAll('#sc-mlab .sc-mb:not(.sc-mb-ev)')].map(e=>e.textContent)})")
    evd = [t_.split(' ')[0] for t_ in ml['ev']]
    check(f'{tag} part 7: inside the eviction’s step the memory bar does not yet show the eviction', not evd and panel(pg)['code']['phase'] == 'run', ml)
    # GPT-6 review of v9, #1: the card follows the phase too; while the step runs it says what will happen at its end, with no “moved on” note
    obr = NB(pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
    ev_s = next(x['s'] for x in moms if x['id'] == 'evict')
    check(f'{tag} part 7: ◂ into the eviction’s step: the card says the eviction is still to come, at this step’s end', Z(lang, 'when this step ends', '这一步结束时') in obr and Z(lang, 'will evict', '随后') in obr
          and Z(lang, 'evicts', '淘汰了') not in obr.replace('will evict', '') and Z(lang, 'moved on', '离开了') not in obr and NB(f'{ev_s + 1:,}') in obr, obr[:300])
    paras = pg.evaluate("()=>[...document.querySelectorAll('#sc-rail [data-obs] p')].filter(p=>!p.classList.contains('small')).map(p=>p.innerText)")
    nsent = sum(len([x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9“])|(?<=[。！？])', p_) if x and x.strip()]) for p_ in paras)
    check(f'{tag} part 7: ◂ into the eviction’s step: still a bold sentence and at most two more', len(paras) == 2 and nsent <= 3, (nsent, paras))
    act(pg, '#sc-play [data-act=fwd]', 300)
    obf = NB(pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
    check(f'{tag} part 7: ▸ to the step’s end: the card says the eviction happened', panel(pg)['code'] == {'s': ev_s, 'phase': 'end'} and Z(lang, 'finishes, and', '淘汰了') in obf and Z(lang, 'when this step ends', '这一步结束时') not in obf, obf[:300])
    act(pg, '#sc-play [data-act=back]', 300)
    # review of v10, S4: one more ◂ puts the stage before the moment's step: the note says so first, and the card speaks of what will happen
    act(pg, '#sc-play [data-act=back]', 300)
    obb = NB(pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
    note_b = NB(pg.evaluate("()=>{const e=document.querySelector('#sc-rail [data-obs] .sc-obnote'); return e?e.innerText:''}"))
    check(f'{tag} part 7: two ◂ from the eviction: the note says the stage is before this step, and the card still says what will happen',
          Z(lang, 'before this step', '还没到这一步') in note_b and Z(lang, 'when this step ends', '这一步结束时') in obb and Z(lang, 'now', '现在') not in obb.replace(note_b, '')
          and Z(lang, f'Press ▸ until step {ev_s + 1:,} ends', f'按 ▸ 直到第 {ev_s + 1:,} 步结束') in NB(obb), (note_b, obb[:300]))
    act(pg, '#sc-play [data-act=fwd]', 300)
    # review of v10, S3: during play the card turns to the past tense as soon as the stage shows the step ended
    act(pg, '#sc-rail [data-moment="evict"]', 450); act(pg, '#sc-play [data-act=back]', 300)
    pg.evaluate("()=>{window.__obsN=0; const box=document.querySelector('#sc-rail [data-obs]'); window.__obsMO=new MutationObserver(()=>{window.__obsN++;}); window.__obsMO.observe(box,{childList:true});}")
    pg.evaluate("()=>document.querySelector('#sc-play [data-act=play]').click()")
    seen_ = []
    for _ in range(40):
        pg.wait_for_timeout(60)
        v = pg.evaluate("()=>[document.querySelector('#sc-wct').textContent, document.querySelector('#sc-rail [data-obs]').innerText]")
        seen_.append(v)
    ended_future = [p_ for p_, t_ in seen_ if ((('ended at' in p_) if lang == 'en' else ('结束' in p_)) and NB(f'{ev_s + 1:,}') in NB(p_) and Z(lang, 'when this step ends', '这一步结束时') in t_)]
    check(f'{tag} part 7: during play, the eviction step’s end never shows the future-tense card', not ended_future, ended_future[:2])
    check(f'{tag} part 7: during play the card showed the eviction step both running and ended', any(Z(lang, 'when this step ends', '这一步结束时') in t_ for _, t_ in seen_) and any(Z(lang, 'finishes, and', '淘汰了') in t_ for _, t_ in seen_), [p_[:40] for p_, _ in seen_][:6])
    nobs = pg.evaluate("()=>{window.__obsMO.disconnect(); return window.__obsN}")
    check(f'{tag} part 7: during 2.4 s of play the card changes only when its message does (running, ended, moved on), not at every step', nobs <= 3, nobs)
    pg.evaluate("()=>{if(window.__sc.state.playing) document.querySelector('#sc-play [data-act=play]').click();}"); pg.wait_for_timeout(150)
    # the speed control says what 1× means in this part (review B, B6)
    sp_ = pg.evaluate("()=>document.querySelector('.sc-speed').dataset.tipx")
    check(f'{tag} part 7: the speed tooltip says 1× shows one step every 0.8 s', '0.8' in sp_, sp_)
    if w <= 760:   # a phone's line under the code leads to the card's observation (review B, B10)
        act(pg, '#sc-rail [data-moment="budget"]', 450); pg.wait_for_timeout(500)
        vis = pg.evaluate("()=>{const e=document.getElementById('sc-wobs'); return !!e&&!e.hidden&&e.getClientRects().length>0}")
        pg.evaluate("()=>document.querySelector('#sc-wobs [data-jump=obs]').click()"); pg.wait_for_timeout(900)
        ob_r = pg.evaluate("()=>{const r=document.querySelector('#sc-rail [data-obs]').getBoundingClientRect(); return [r.top,r.bottom,innerHeight]}")
        check(f'{tag} part 7: on a phone, a line under the code leads to the card’s observation', vis and 56 <= ob_r[0] < ob_r[2] - 40, (vis, ob_r))
    # ▸ ends the step: its release and eviction lines run, and the GPU box shows it done; ▸ again: the next step, ended
    m = moms[1]; act(pg, f'#sc-rail [data-moment="{m["id"]}"]', 450)
    act(pg, '#sc-play [data-act=fwd]', 300); pn = panel(pg); rows = {r['k']: r for r in pn['rows']}
    exp = pg.evaluate(f"()=>window.__sc.codeLines('{m['cfg']['load']}',{json.dumps(m['cfg'])},{m['s']},'end')")
    check(f'{tag} part 7: ▸ ends the step: the panel shows it ended, and its end lines ran with its values', pn['code'] == {'s': m['s'], 'phase': 'end'}
          and (('ended at' in pn['head']) if lang == 'en' else ('结束' in pn['head'])) and all(rows[k]['st'] == exp[k]['st'] and NB(rows[k]['v']) == NB(exp[k]['v']) for k in WL_ORDER)
          and rows['ev']['st'] == 'ran' and rows['fl']['st'] == 'ran' and rows['tm']['st'] == 'ran', (pn['code'], pn['head']))
    check(f'{tag} part 7: ▸ ends the step: the GPU box shows the same step, done', gpu_says(pn, m['s'] + 1, True, lang), pn['gpu'])
    qn = pg.evaluate("()=>+document.getElementById('sc-qn').textContent"); ci2 = pg.evaluate(f"()=>window.__sc.codeInfo('{m['cfg']['load']}',{json.dumps(m['cfg'])},{m['s'] + 1})")
    check(f'{tag} part 7: at a step’s end the queue holds what the next step’s admission loop sees', qn == len(ci2['adm']) + len(ci2['wait']), (qn, ci2['adm'], ci2['wait']))
    act(pg, '#sc-play [data-act=fwd]', 300); pn = panel(pg)
    check(f'{tag} part 7: ▸ again: the next step, ended, in the panel and the GPU box', pn['code'] == {'s': m['s'] + 1, 'phase': 'end'} and gpu_says(pn, m['s'] + 2, True, lang), (pn['code'], pn['gpu']))
    act(pg, '#sc-play [data-act=back]', 300); pn = panel(pg)
    check(f'{tag} part 7: ◂ goes back to the start of that step: the panel shows it running, the GPU box starting it', pn['code'] is not None and pn['code']['s'] == m['s'] + 1 and pn['code']['phase'] == 'run' and gpu_says(pn, m['s'] + 2, False, lang), (pn['code'], pn['gpu']))
    # the GPU box counts the requests that ran in the step it shows done (review B, B11); its meters are not cut off (B13)
    act(pg, '#sc-play [data-act=fwd]', 300)
    if w > 760:
        mt = pg.evaluate("()=>{const m=document.getElementById('sc-meters'); return {t:m.innerText, clip:m.scrollWidth>m.clientWidth+1, lab:[...m.querySelectorAll('.sc-meter')].map(x=>x.dataset.tipx)}}")
        check(f'{tag} part 7: at a step’s end the GPU box’s meter counts the requests that ran in it, and no meter is cut off', not mt['clip'] and any(('ran in this step' if lang == 'en' else '跑过') in t_ for t_ in mt['lab']), mt)
    # the last step's end shows before the whole run (review B, B4)
    pg.evaluate("()=>{const i=document.querySelector('#sc-scrub'); i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
    act(pg, '#sc-play [data-act=back]', 200); act(pg, '#sc-play [data-act=fwd]', 200); last = panel(pg)
    nst = pg.evaluate("()=>window.__sc.view(window.__sc.state.load,window.__sc.state.settings).nSteps")
    check(f'{tag} part 7: ▸ stops on the last step’s end before the whole run (review B, B4)', last['code'] == {'s': nst - 1, 'phase': 'end'} and gpu_says(last, nst, True, lang), (last['code'], nst))
    act(pg, '#sc-play [data-act=fwd]', 200); pn = panel(pg)
    check(f'{tag} part 7: after a moment, the whole run’s header says the run is done (review B, B4)', pn['code'] is None and ((('whole run is done' in pn['head']) or ('Run done' in pn['head'])) if lang == 'en' else ('运行已结束' in pn['head'])), pn['head'])
    # play, one step at a time: the panel follows the GPU box, and nothing moves (R7.9; review A, finding 10)
    act(pg, '#sc-rail [data-moment="budget"]', 450); pg.wait_for_timeout(400)
    geo0 = pg.evaluate("()=>[document.getElementById('sc-wc').getBoundingClientRect().height, document.getElementById('sc-strip').getBoundingClientRect().height]")
    click(pg, '#sc-play [data-act=play]'); pg.wait_for_timeout(100)
    same, sync, n_, steps_, phases = True, True, 0, set(), set()
    t0_ = time.time()
    for _ in range(14):
        pg.wait_for_timeout(110)
        g = pg.evaluate("()=>({h:[document.getElementById('sc-wc').getBoundingClientRect().height, document.getElementById('sc-strip').getBoundingClientRect().height], c:window.__sc.codeStep(), gh:document.getElementById('sc-gh').textContent, wt:document.getElementById('sc-wct').textContent, playing:window.__sc.state.playing})")
        if not g['playing']: break
        n_ += 1
        if any(abs(a - b) > 0.5 for a, b in zip(g['h'], geo0)): same = False
        if g['c']:
            steps_.add(g['c']['s']); phases.add(g['c']['phase'])
            num = NB(f"{g['c']['s'] + 1:,}"); done = ('done' in g['gh']) if lang == 'en' else ('已结束' in g['gh'])
            if num not in NB(g['gh']) or num not in NB(g['wt']) or done != (g['c']['phase'] == 'end'): sync = False
    el_ = time.time() - t0_
    click(pg, '#sc-play [data-act=play]'); pg.wait_for_timeout(150)
    check(f'{tag} part 7: while it plays the panel shows the GPU box’s step, running or done', n_ >= 5 and sync, n_)
    check(f'{tag} part 7: it plays one step at a time, slowly enough to read (about 0.8 s per step at 1×), showing each step’s end', 1 <= len(steps_) <= 2 + el_ / 0.8 and 'end' in phases, (sorted(steps_), phases, round(el_, 2)))
    check(f'{tag} part 7: while it plays the panel and the strip keep their heights (R7.9)', n_ >= 5 and same, (geo0, n_))
    # the Connect card's button opens this part
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(500)
    s2 = st(pg); p2 = panel(pg)
    check(f'{tag} part 7: a language switch keeps the part, the moment and the step', s2['partId'] == 'code' and pg.evaluate("()=>window.__sc.codeM") == 'budget' and p2['code'] is not None, (s2['partId'], p2['code']))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(500)
    check(f'{tag} part 7: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
    rules(pg, f'{tag} part 7', lang)
    if w > 760:   # ▸ pressed again and again under a resting pointer: its tooltip stays away (review A, finding 11)
        pg.hover('#sc-play [data-act=fwd]'); pg.wait_for_timeout(400)
        pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(350); t1_ = pg.evaluate("()=>document.getElementById('tip').hidden")
        pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(350); t2_ = pg.evaluate("()=>document.getElementById('tip').hidden")
        check(f'{tag} part 7: ▸’s tooltip stays hidden while it is pressed again and again', t1_ and t2_, (t1_, t2_))


# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'serving-capstone.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a budget and see how many requests meet their goal. Each option starts from the initial setup.", "选一个预算，看看有多少个请求达标。每个选项都从初始设置出发。", "Caching off or on? Pick one and see how many requests meet their goal.", "关闭还是打开缓存？选一种，看看有多少个请求达标。", "Pick a cache budget and see how many requests meet their goal.", "选一个缓存预算，看看有多少个请求达标。", "Pick a cap on running requests and see how many requests meet their goal.", "选一个运行请求上限，看看有多少个请求达标。", "Pick a scenario and see how many requests meet their goal.", "选一个场景，看看有多少个请求达标。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["The stalls, the queues and the cache hits all come from one scheduler loop.", "卡顿、排队和缓存命中，都来自同一个调度循环。", "the stall on the timeline", "时间线上的那次卡顿", "Each waiting request’s trace says what it waits for", "每个排队的请求都记着它在等什么", "With no token budget, a ${tokK(", "不设 token 预算时，一份 ${tokK(", "In a support app, ${s6.R.length} chats", "一个客服应用里，${s6.R.length} 个对话", "This is what prefix caching saves.", "前缀缓存省下的就是这些。", "Initial setup, ${nm.charAt(0).toLowerCase()+nm.slice(1)}", "的初始设置`):tx('Initial setup','初始设置')", "Three workloads on one ${GPUN}", "三个负载，都是一张 ${GPUN}",
           "This is how a new chat waits for the budget while a document prefills.", "文档在 prefill 时，新来的对话就是这样在等预算。",
           "prefills the whole document again, which is what a cache budget that is too small costs.", "就得把整份文档重新 prefill，这就是缓存预算太小的代价。",
           "This is how a queue builds up when the cap is too low.", "上限设得太低时，排队就是这样来的。"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["Everything you saw comes from one loop", "前面看到的一切都来自同一个循环", "the stall you saw", "你看到的那次卡顿", "In part 3, the chats", "比如第 3 部分里", "In part one the", "第一部分里", "In part four, with caching on", "第四部分里", "the table there shows", "Part 3: a chat waits", "第 3 部分：对话在等预算", "Part 4: a cache hit", "Part 5: an eviction", "Part 6: the cap is full", "This is the saving of part 4.", "第 4 部分省下的就是这些", "Part ${SIX(rp)+1}’s initial setup", "第 ${SIX(rp)+1} 部分的初始设置", "The three workloads of the guided parts", "分步引导里的三个负载",
           "This is the wait you saw in part 3.", "第 3 部分里对话等的就是这个。", "the cost of part 5.", "这就是第 5 部分里的代价。", "This is the queue of part 6.", "第 6 部分里的排队就是这样来的。"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

with sync_playwright() as p:
    b = p.chromium.launch()
    # =================== every part and mode, three widths, two languages ===================
    for (w, h) in ((1440, 900), (900, 900), (390, 844)):
        for lang in ('en', 'zh'):
            tag = f'{w} {lang}'
            pg = page(b, w, h, lang)
            head = NB(pg.evaluate("()=>document.querySelector('#sc-app .labhead').innerText"))
            check(f'{tag}: the question as the title', ('How do you rescue a congested chat service?' if lang == 'en' else '怎么救活一个拥堵的聊天服务？') in head, head[:120])
            check(f'{tag}: no Fidelity button on top (owner, 2026-10-11 19:28 UTC)', not any(x in head for x in ('Fidelity', '保真度')), head[:200])
            check(f'{tag}: the outcome line', ('You will be able to:' if lang == 'en' else '学完你能：') in head, head[:200])
            if w == 390:
                check(f'{tag}: first screen shows the strip', visible_in(pg, '#sc-strip', h))
                check(f'{tag}: first screen shows the timeline', visible_in(pg, '#sc-tl', h))
                check(f'{tag}: first screen shows a control (Play)', visible_in(pg, '#sc-play [data-act=play]', h))
                if lang == 'zh': shot(pg, SC / 'p1_first_390_zh.png')
            # part 1: play, look at it mid-run, then end
            click(pg, '#sc-play [data-act=play]'); pg.wait_for_timeout(300)
            s1 = st(pg); pg.wait_for_timeout(700); s2 = st(pg)
            check(f'{tag}: Play advances the playhead', s1['playing'] and s2['playhead'] is not None and s2['playhead'] > (s1['playhead'] or 0), (s1['playhead'], s2['playhead']))
            yours = NB(pg.evaluate("()=>{const t=document.querySelector('#sc-mt tr[data-k=ok] td:nth-child(3)'); return t?t.textContent:''}"))
            check(f'{tag}: whole-run values wait for the end (R3.8)', yours in ('after the run', '跑完后显示'), yours)
            if w in (1440, 900) or lang == 'zh':
                pg.evaluate("()=>{const i=document.querySelector('#sc-scrub'); i.value=Math.round(3.3/12*1000); i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
                if w == 390: pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
                shot(pg, SC / f'p1_playing_{w}_{lang}.png')
            end(pg)
            check(f'{tag}: part 1 result after the run', pg.evaluate("()=>!document.querySelector('#sc-rail [data-obs]').hidden"))
            rules(pg, f'{tag} part 1', lang)
            check(f'{tag} part 1: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            # part 2: gating and the answer hidden
            act(pg, '#sc-rail .step.current [data-nav=next]', 350)
            check(f'{tag}: part 2 Next disabled before a prediction', pg.evaluate("()=>[...document.querySelectorAll('#sc-rail [data-nav=next]')].every(b=>b.disabled)"))
            txt = all_text(pg)
            hid = ('whole prompt went into one step' not in txt) if lang == 'en' else ('整个 prompt 进了一步' not in txt)
            check(f'{tag}: the answer appears nowhere before the pick', hid)
            act(pg, '#sc-rail [data-pred="c"]')
            check(f'{tag}: part 2 Next enabled after a pick', pg.evaluate("()=>!document.querySelector('#sc-rail .step.current [data-nav=next]').disabled"))
            ob = NB(pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
            check(f'{tag}: a wrong pick shows the right answer', ('The right answer is' in ob) if lang == 'en' else ('正确答案是' in ob), ob[:160])
            speed0 = st(pg)['speed']
            act(pg, '#sc-rail .step.current [data-nav=showme]', 400)
            s = st(pg)
            blk = NB(pg.evaluate("()=>[...document.querySelectorAll('#sc-gbar .sc-pb em')].map(e=>e.textContent).join('|')"))
            check(f'{tag}: Show me puts the playhead in the stall step (the document block)', s['playhead'] is not None and '32,768' in blk, (s['playhead'], blk))
            check(f'{tag}: Show me keeps the speed', s['speed'] == speed0)
            pg.evaluate("()=>{const d=document.querySelector('#sc-dots .sc-dot'); if(d) d.click();}"); pg.wait_for_timeout(250)
            ring = pg.evaluate("()=>{const d=document.querySelector('#sc-dots .sc-dot.sel'), b=document.querySelector('#sc-dots'); if(!d) return null; const r=d.getBoundingClientRect(), c=b.getBoundingClientRect(), o=d.classList.contains('f')?3.5:3; return [r.top-o>=c.top-0.5, r.bottom+o<=c.bottom+0.5, r.left-o>=c.left-0.5, r.right+o<=c.right+0.5]}")
            check(f'{tag}: the selected token’s ring is inside the tokens-out box (review r3, finding 2)', ring is not None and all(ring), ring)
            pg.evaluate("()=>{const x=document.querySelector('#sc-tcard [data-act=unsel], #sc-trace [data-act=unsel]'); if(x) x.click();}"); pg.wait_for_timeout(150)
            if lang == 'en' and w == 1440 or lang == 'zh':
                if w == 390: pg.evaluate("()=>document.querySelector('#sc-strip').scrollIntoView({block:'start'})"); pg.wait_for_timeout(150)
                shot(pg, SC / f'p2_showme_{w}_{lang}.png')
            rules(pg, f'{tag} part 2', lang)
            # parts 3–6: each option; the result appears after the run
            for k, opts in ((2, ['2048', '128']), (3, ['true']), (4, ['3072', '6144', 'null']), (5, ['8', '64'])):
                go(pg, k); end(pg)
                for v in opts:
                    lv = f'#sc-rail [data-lever="{v}"]'
                    y0 = pg.evaluate(f"()=>{{const e=document.querySelector('{lv}'); e.scrollIntoView({{block:'center'}}); return e.getBoundingClientRect().top}}"); pg.wait_for_timeout(120)
                    y0 = pg.evaluate(f"()=>document.querySelector('{lv}').getBoundingClientRect().top")
                    pg.locator(lv).click(); pg.wait_for_timeout(250)
                    if w == 390:
                        y1 = pg.evaluate(f"()=>document.querySelector('{lv}').getBoundingClientRect().top")
                        check(f'{tag} part {k + 1}: the lever stays in place after a tap', abs(y1 - y0) < 40 and 0 <= y1 < h, (y0, y1))
                    ss = st(pg)
                    check(f'{tag} part {k + 1} option {v}: autoplay from 0', ss['playing'] and ss['playhead'] is not None, ss['playhead'])
                    res = pg.evaluate(f"()=>document.querySelector('{lv} .lv-r').textContent")
                    check(f'{tag} part {k + 1} option {v}: no result before the run ends', res == '', res)
                    end(pg)
                    geo = pg.evaluate(f"""()=>{{const L=document.querySelector('{lv}'), t=L.querySelector('b'), d=L.querySelector('.lv-d'), r=L.querySelector('.lv-r'), R=e=>e?e.getBoundingClientRect():null, lh=parseFloat(getComputedStyle(t).lineHeight)||20, rh=parseFloat(getComputedStyle(r).lineHeight)||16, rt=R(t), rd=R(d), rr=R(r);
                      const hit=(a,b)=>!!(a&&b&&a.width&&b.width&&!(a.right<=b.left+0.5||a.left>=b.right-0.5||a.bottom<=b.top+0.5||a.top>=b.bottom-0.5));
                      return {{overlap:hit(rt,rd)||hit(rt,rr)||hit(rd,rr), titleLines:rt.height/lh, titleW:rt.width, resLines:rr.height/rh, text:r.textContent}}}}""")
                    check(f'{tag} part {k + 1} option {v}: title, description and result do not overlap (review r3, finding 1)', not geo['overlap'], geo)
                    check(f'{tag} part {k + 1} option {v}: the title stays on ≤ 2 lines and never goes vertical (review r3, finding 5)', geo['titleLines'] <= 2.3 and geo['titleW'] >= 30, geo)
                    check(f'{tag} part {k + 1} option {v}: the result is only the goal count, on one line (review r3, finding 7; review r2, N5)', re.fullmatch(r'\d+ / \d+ (meet the goal|达标)', NB(geo['text']).strip()) is not None and geo['resLines'] <= 1.3, geo)
                    if k == 2 and v == '2048':   # the document-prefill label at every width (review B4, N1; R5.8, R7.8)
                        pf = pg.evaluate("()=>window.__sc.pfGeo()")
                        both = bool(pf) and (('document prefill' in pf['text'] and 'new requests wait' in pf['text']) if lang == 'en' else ('文档 prefill' in pf['text'] and '新请求要等' in pf['text']))
                        check(f'{tag} part 3: the document-prefill label shows both halves (review B4, N1)', bool(pf) and not pf['hidden'] and both, pf)
                        place = bool(pf) and (pf['lab'][3] <= pf['rowsTop'] + 0.5 or (pf['lab'][1] >= pf['docBottom'] - 0.5 and pf['lab'][2] <= pf['xa'] + 0.5))
                        check(f'{tag} part 3: the label covers no row: above every row, or below the document’s row and left of its arrival (review B4, N1)', place, pf)
                    if w == 1440:
                        pg.wait_for_timeout(700)
                        ob = pg.evaluate(f"()=>{{const o=document.querySelector('#sc-rail [data-obs]'), l=document.querySelector('{lv}'); if(!o||o.hidden) return null; const r=o.getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom), innerHeight, Math.round(l.getBoundingClientRect().top)]}}")
                        check(f'{tag} part {k + 1} option {v}: the result box is in view after the run (review r1, S1)', ob is not None and 0 <= ob[0] <= ob[2] - 120, ob)
                        check(f'{tag} part {k + 1} option {v}: the option picked stays in view', ob is not None and 0 <= ob[3] < ob[2], ob)
                        if k == 2 and v == '2048':
                            act(pg, '#sc-rail [data-obs] [data-sel]', 350)
                            rc = pg.evaluate("()=>{const c=document.querySelector('#sc-tcard'), r=c.getBoundingClientRect(); return [c.hidden,Math.round(r.top),Math.round(r.bottom),innerHeight]}")
                            check(f'{tag} part 3: “See why” shows the trace card in the viewport (review r1, A1)', not rc[0] and rc[1] >= 0 and rc[2] <= rc[3], rc)
                            ov = pg.evaluate("()=>{const s=window.__sc.state.selected, b=window.__sc.rowBox(s), c=document.querySelector('#sc-tcard').getBoundingClientRect(); return [b, [c.left,c.top,c.right,c.bottom], !(c.right<=b.left||c.left>=b.right||c.bottom<=b.top||c.top>=b.bottom), document.querySelector('#sc-tcard').scrollHeight-document.querySelector('#sc-tcard').clientHeight]}")
                            check(f'{tag} part 3: the trace card does not cover the row it explains (review r2, N4)', not ov[2], ov)
                            check(f'{tag} part 3: the trace card shows all its lines (review r2, N4)', ov[3] <= 1, ov[3])
                            act(pg, '#sc-tcard [data-act=unsel]')
                    res = pg.evaluate(f"()=>document.querySelector('{lv} .lv-r').textContent")
                    check(f'{tag} part {k + 1} option {v}: result after the run', bool(res), res)
                check(f'{tag} part {k + 1}: Next enabled after an option', pg.evaluate("()=>!document.querySelector('#sc-rail .step.current [data-nav=next]').disabled"))
                # every option card of the part after the runs, the initial option included (it shows the result of the run seen first):
                # no overlap, the title horizontal on ≤ 2 lines, the result only the goal count (review r3, findings 1, 5 and 7; R7.8)
                cards = pg.evaluate("""()=>[...document.querySelectorAll('#sc-rail .step.current .lever')].map(L=>{const t=L.querySelector('b'), d=L.querySelector('.lv-d'), r=L.querySelector('.lv-r'), R=e=>e?e.getBoundingClientRect():null, lh=parseFloat(getComputedStyle(t).lineHeight)||20, rh=parseFloat(getComputedStyle(r).lineHeight)||16, rt=R(t), rd=R(d), rr=R(r);
                  const hit=(a,b)=>!!(a&&b&&a.width&&b.width&&!(a.right<=b.left+0.5||a.left>=b.right-0.5||a.bottom<=b.top+0.5||a.top>=b.bottom-0.5));
                  return {v:L.getAttribute('data-lever'), overlap:hit(rt,rd)||hit(rt,rr)||hit(rd,rr), titleLines:rt.height/lh, titleW:rt.width, resLines:rr.height?rr.height/rh:0, text:r.textContent}})""")
                check(f'{tag} part {k + 1}: every option card, the initial one included, has no overlapping text (review r3, finding 1)', cards and not any(c['overlap'] for c in cards), [c for c in cards if c['overlap']][:2])
                check(f'{tag} part {k + 1}: every option title stays on ≤ 2 lines and never goes vertical (review r3, finding 5)', cards and all(c['titleLines'] <= 2.3 and c['titleW'] >= 30 for c in cards), [c for c in cards if not (c['titleLines'] <= 2.3 and c['titleW'] >= 30)][:2])
                check(f'{tag} part {k + 1}: every result shown is only the goal count, on one line (review r3, finding 7)', cards and all(c['text'] == '' or (re.fullmatch(r'\d+ / \d+ (meet the goal|达标)', NB(c['text']).strip()) and c['resLines'] <= 1.3) for c in cards), [c for c in cards if c['text']][:2])
                check(f'{tag} part {k + 1}: the initial option shows its result after its run (review r3, finding 5)', any(c['text'] for c in cards if c['v'] not in opts), [(c['v'], c['text']) for c in cards])
                check(f'{tag} part {k + 1}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
                rules(pg, f'{tag} part {k + 1}', lang)
                if k == 3 and (w, lang) in ((1440, 'en'), (1440, 'zh'), (900, 'zh')): shot(pg, SC / f'p4_cache_on_{w}_{lang}.png')
                if k == 4 and (w, lang) in ((1440, 'en'), (1440, 'zh'), (900, 'zh')):
                    pg.evaluate("()=>document.querySelector('#sc-rail .levers').scrollIntoView({block:'center'})"); pg.wait_for_timeout(150)   # part 5's chart: the bars on the option rows (review r1, S13)
                    shot(pg, SC / f'p5_chart_{w}_{lang}.png')
            if w != 390:
                act(pg, '#sc-ic-pol button[data-v="static"]'); end(pg)
                check(f'{tag} part 6: static batching runs', st(pg)['cfg']['policy'] == 'static')
            walkthrough(pg, tag, w, h, lang)
            go(pg, 7)
            txt = NB(pg.evaluate("()=>document.querySelector('#sc-rail .step.current').innerText"))
            check(f'{tag} part 8: three misconceptions', txt.count('✗') == 3, txt.count('✗'))
            check(f'{tag} part 8: the button to the scenarios (R3.12)', pg.evaluate("()=>!!document.querySelector('#sc-rail [data-mode-go=sandbox]')"))
            nl = pg.evaluate("()=>{const r=document.querySelector('#sc-rail .next-lab'), e=document.querySelector('.next-lab-end .next-lab'); return !!(r&&(r.offsetParent||e&&e.offsetParent))}")
            check(f'{tag} part 8: the next-lab footer (R3.11)', nl)
            rules(pg, f'{tag} part 8', lang)
            # Scenarios
            act(pg, '#sc-rail [data-mode-go=sandbox]', 450); end(pg)
            check(f'{tag}: Scenarios mode', st(pg)['mode'] == 'sandbox')
            check(f'{tag}: three scenarios', pg.evaluate("()=>document.querySelectorAll('#sc-rail [data-preset]').length") == 3)
            act(pg, '#sc-rail [data-preset="1"]', 300); end(pg)
            check(f'{tag}: the preset follows the load', pg.evaluate("()=>document.querySelector('#sc-rail [data-preset=\"1\"]').getAttribute('aria-pressed')") == 'true')
            if w != 390:
                y0 = pg.evaluate("()=>{const e=document.querySelector('#sc-ctl-cap'); e.scrollIntoView({block:'center'}); return e.getBoundingClientRect().top}"); pg.wait_for_timeout(150)
                y0 = pg.evaluate("()=>document.querySelector('#sc-ctl-cap').getBoundingClientRect().top")
                pg.locator('#sc-ctl-cap button[data-v="16"]').click(); pg.wait_for_timeout(300)
                y1 = pg.evaluate("()=>document.querySelector('#sc-ctl-cap').getBoundingClientRect().top")
                check(f'{tag}: a Settings change does not move the page (R3.2)', abs(y1 - y0) < 2, (y0, y1))
            else:
                pg.locator('#sc-ctl-cap button[data-v="16"]').click(); pg.wait_for_timeout(300)
            end(pg)
            check(f'{tag}: Settings set the cap', st(pg)['cfg']['cap'] == 16)
            rules(pg, f'{tag} Scenarios', lang)
            check(f'{tag} Scenarios: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            # the challenge
            act(pg, '#sc-rail [data-mode-go=challenge]', 450)
            if w == 390 and lang == 'zh':
                pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100); shot(pg, SC / 'ch_first_390_zh.png')
            order = pg.evaluate("()=>{const c=document.querySelector('#sc-rail .sc-chcard'); const a=c.querySelector('.sc-load'), g=c.querySelector('.sc-goalcard'), s=c.querySelector('.sc-chset'); return a.compareDocumentPosition(g)&4 && g.compareDocumentPosition(s)&4}")
            check(f'{tag}: the challenge shows the load and the goals before the settings', bool(order))
            for k_, v_ in (('cap', '16'), ('budget', '2048'), ('cache', 'true')):
                pg.evaluate(f"()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-{k_}\"] button[data-v=\"{v_}\"]').click()"); pg.wait_for_timeout(120)
            seg = pg.evaluate("()=>{const g=document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-cb\"]'); if(!g) return null; const c=g.getBoundingClientRect(); return [...g.querySelectorAll('button')].map(b=>{const r=b.getBoundingClientRect(); return r.width>8&&r.left>=c.left-0.5&&r.right<=c.right+0.5&&r.top>=c.top-0.5&&r.bottom<=c.bottom+0.5;})}")
            check(f'{tag}: every option of the challenge’s cache budget is visible (review r3, finding 8)', seg is not None and len(seg) == 5 and all(seg), seg)
            act(pg, '#sc-rail [data-act=run]', 300); end(pg)
            pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000); pg.wait_for_timeout(200)
            check(f'{tag}: cap 16 misses (tier 3)', st(pg)['chTier']['tier'] == 3)
            check(f'{tag}: tier 3 offers “Show the answer”', pg.evaluate("()=>{const b=document.querySelector('#sc-ch-res [data-act=reveal]'); return b?b.textContent:''}") == Z(lang, 'Show the answer', '直接看答案'))
            pl = pg.evaluate("()=>window.__sc.plan()") or ''; plo = pg.evaluate("()=>document.querySelector('#sc-plan').textContent")
            cf = ('undo it', 'What each change does', '撤回后', '每处改动的作用')
            check(f'{tag}: a failing plan holds no counterfactual (review r1, A2)', pl and not any(x in pl for x in cf) and not any(x in plo for x in cf), pl[-300:])
            act(pg, '#sc-ch-res [data-act=reveal]')
            pl2 = pg.evaluate("()=>window.__sc.plan()") or ''
            check(f'{tag}: after “Show the answer” the plan says what each change does', any(x in pl2 for x in cf), pl2[-200:])
            check(f'{tag}: “Show the answer” shows the chart', pg.evaluate("()=>!!document.querySelector('#sc-ch-res .cmp')"))
            # the owner's review of batch 8: the chart keeps the named settings that pass, the pick (cap 16, budget 2,048, caching on) and the
            # initial setup; the note counts the other named settings (9 in all, the pick among them)
            cr = pg.evaluate("()=>{const rs=[...document.querySelectorAll('#sc-ch-res .cmp-row')]; return {n:rs.length, ok:rs.filter(r=>r.classList.contains('ok')).length, me:rs.filter(r=>r.classList.contains('me')).length, stray:rs.filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!r.querySelector('.cmp-tag')).length, note:(document.querySelector('#sc-ch-res .cmp-note')||{}).textContent||'', title:(document.querySelector('#sc-ch-res .cmp-title')||{}).textContent||''}}")
            nh = 9 - cr['n']
            check(f'{tag}: the brief chart holds only the passing settings, the pick and the initial setup; the note counts the other {nh}', cr['stray'] == 0 and cr['me'] == 1 and cr['ok'] >= 1 and cr['n'] == cr['ok'] + 2 and nh >= 1
                  and cr['note'] == Z(lang, f'The other {nh} settings compared here get under 90% of the requests to their goal and are not shown.' if nh > 1 else 'The other setting compared here gets under 90% of the requests to their goal and is not shown.', f'这里比较的另外 {nh} 种设置达标的请求不到 90%，图里没有列出。')
                  and Z(lang, 'per setting', '不同设置下') not in cr['title'], cr)
            xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
            check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
            pg.evaluate("()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-cap\"] button[data-v=\"32\"]').click()"); pg.wait_for_timeout(150)
            act(pg, '#sc-rail [data-act=run]', 300); end(pg)
            pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000); pg.wait_for_timeout(200)
            check(f'{tag}: cap 32, budget 2,048, caching on: met with the fewest changes (tier 1)', st(pg)['chTier']['tier'] == 1, st(pg)['chTier'])
            res = NB(pg.evaluate("()=>document.querySelector('#sc-ch-res').innerText"))
            check(f'{tag}: the result has the plan and the two practice links', pg.evaluate("()=>!!document.querySelector('#sc-plan')&&document.querySelectorAll('#sc-ch-res .sc-jobs a').length===2"))
            check(f'{tag}: the capability card in the challenge', pg.evaluate("()=>!!document.querySelector('#sc-rail [data-exits]')"))
            if (w, lang) in ((1440, 'en'), (1440, 'zh'), (900, 'zh')):
                pg.evaluate("()=>document.querySelector('#sc-ch-res').scrollIntoView({block:'start'})"); pg.wait_for_timeout(200); shot(pg, SC / f'ch_result_{w}_{lang}.png')
            rules(pg, f'{tag} challenge', lang)
            check(f'{tag} challenge: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
            pg.context.close()
    # =================== one-off checks at 1440 EN ===================
    pg = page(b, 1440, 900, 'en', perms=True)
    E = "window.__sc.engine"
    # playback: step ends, starts, replay, the 1× rate and 2×
    act(pg, '#sc-play [data-act=fwd]'); t1 = st(pg)['playhead']
    act(pg, '#sc-play [data-act=fwd]'); t2 = st(pg)['playhead']
    ends = pg.evaluate(f"()=>{{const r={E}.simulate({E}.LOADS.stall(),{E}.BASELINE); return r.steps.map(s=>[s.t0,s.t1]);}}")
    check('▸ lands on a step’s end', any(abs(t2 - e) < 1e-9 for _, e in ends) and t2 > t1, (t1, t2))
    act(pg, '#sc-play [data-act=back]'); t3 = st(pg)['playhead']
    check('◂ lands on a step’s start', any(abs(t3 - s0) < 1e-9 for s0, _ in ends) and t3 < t2, (t2, t3))
    s = st(pg)   # R3.9 (product owner, 2026-10-04): a whole run of many requests plays at 2× by default, about 7.5 s
    check('the default speed for a whole run is 2×, until the learner picks one', s['speed'] == 2 and s['speedPicked'] is None and pg.evaluate("()=>document.querySelector('.sc-speed button[aria-pressed=true]').dataset.v") == '2', s['speed'])
    act(pg, '#sc-play .sc-speed button[data-v="1"]', 100)
    act(pg, '#sc-play [data-act=replay]', 100); s = st(pg)
    check('Replay starts from 0 and plays', s['playing'] and s['playhead'] < 0.5, s['playhead'])
    pg.wait_for_timeout(300); a = st(pg)['playhead']; t0 = time.time(); pg.wait_for_timeout(2000); b_ = st(pg)['playhead']; dt = time.time() - t0
    span = pg.evaluate("()=>{const r=window.__sc; return 12}")
    rate = (b_ - a) / dt
    check('1× plays the run in about 15 s (12 s of run at 0.8 s per s)', 0.55 <= rate <= 1.0, rate)
    act(pg, '#sc-play .sc-speed button[data-v="2"]', 100); a = st(pg)['playhead']; t0 = time.time(); pg.wait_for_timeout(1500); b2 = st(pg)['playhead']; rate2 = (b2 - a) / (time.time() - t0)
    check('2× is about twice as fast', 1.6 <= rate2 / rate <= 2.4, (rate, rate2))
    act(pg, '#sc-play [data-act=play]', 100); s = st(pg)
    check('Pause stops the playhead', not s['playing'])
    pg.wait_for_timeout(400); check('paused means paused', abs(st(pg)['playhead'] - s['playhead']) < 1e-9)
    act(pg, '#sc-play .sc-speed button[data-v="1"]', 100)
    # rapid option changes: one loop, at the 1× rate
    go(pg, 1); act(pg, '#sc-rail [data-pred="b"]'); go(pg, 2); end(pg)
    tok0 = st(pg)['runTok']
    for v in ('8192', '2048', '512', '128', '2048'):
        pg.evaluate(f"()=>document.querySelector('#sc-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(20)
    s = st(pg)
    check('each option change bumps the run token', s['runTok'] >= tok0 + 5, (tok0, s['runTok']))
    check('the last option wins', s['cfg']['budget'] == 2048)
    pg.wait_for_timeout(300); a = st(pg)['playhead']; t0 = time.time(); pg.wait_for_timeout(1500); b_ = st(pg)['playhead']; rate3 = (b_ - a) / (time.time() - t0)
    check('after a burst of changes a single loop plays (the 1× rate)', 0.55 <= rate3 <= 1.0, rate3)
    # R3.20 (product owner, 2026-10-04): the option already picked leaves the run on screen as it is; a different one plays from 0
    end(pg); tok1 = st(pg)['runTok']
    pg.evaluate("()=>document.querySelector('#sc-rail [data-lever=\"2048\"]').click()"); pg.wait_for_timeout(300); s = st(pg)
    check('R3.20: the option already picked does not play the run again (the whole run stays shown)', not s['playing'] and s['playhead'] is None and s['runTok'] == tok1, (s['playing'], s['playhead'], tok1, s['runTok']))
    pg.evaluate("()=>document.querySelector('#sc-rail [data-lever=\"512\"]').click()"); pg.wait_for_timeout(300); s = st(pg)
    check('R3.20: a different option plays its run from 0', s['playing'] and s['playhead'] is not None and s['playhead'] < 2, (s['playing'], s['playhead']))
    pg.evaluate("()=>document.querySelector('#sc-rail [data-lever=\"512\"]').click()"); pg.wait_for_timeout(300); s2 = st(pg)
    check('R3.20: the same option while its run plays: the run goes on, not from 0', s2['playing'] and s2['playhead'] is not None and s2['playhead'] > s['playhead'], (s['playhead'], s2['playhead']))
    pg.evaluate("()=>document.querySelector('#sc-rail [data-lever=\"2048\"]').click()"); pg.wait_for_timeout(100); s = st(pg)
    check('R3.20: back to 2048 after 512 (A → B → A): its run was watched to the end, so it shows whole at once, no replay', not s['playing'] and s['playhead'] is None and s['cfg']['budget'] == 2048, (s['playing'], s['playhead']))
    act(pg, '#sc-play [data-act=replay]', 100)   # ↺ Replay plays it again (the language switch below happens while it plays)
    # a language switch while playing keeps the part, the prediction and the settings
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(400)
    s = st(pg)
    check('a language switch keeps the part, the prediction and the settings', s['partId'] == 'budget' and s['pred'] == 'b' and s['cfg']['budget'] == 2048, s)
    pg.wait_for_timeout(300); a = st(pg)['playhead']; t0 = time.time(); pg.wait_for_timeout(1000); b_ = st(pg)['playhead']
    r4 = (b_ - a) / (time.time() - t0) if a is not None and b_ is not None else 0
    check('after a language switch one loop plays (not two)', r4 <= 1.0, r4)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(300)
    # leaving the lab stops the animation
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(500)
    check('leaving the lab stops the replay', not st(pg)['playing'])
    raf = pg.evaluate("()=>new Promise(res=>{let k=0; const o=window.requestAnimationFrame; let c=0; window.requestAnimationFrame=f=>{c++; return o(f);}; setTimeout(()=>{window.requestAnimationFrame=o; res(c);},400);})")
    check('no frame is requested after leaving (the old loop is gone)', raf == 0, raf)
    pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(600)
    check('back in the lab: the same part', st(pg)['partId'] == 'budget')
    # tracing: a chip, a row, a memory block
    go(pg, 5); end(pg); act(pg, '#sc-rail [data-lever="8"]')
    pg.evaluate("()=>{const i=document.querySelector('#sc-scrub'); i.value=450; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(200)
    chip = pg.evaluate("()=>{const c=document.querySelector('#sc-chips .sc-chip'); return c?+c.dataset.sel:null}")
    check('a queue at the cap has chips', chip is not None)
    if chip is not None:
        click(pg, f'#sc-chips .sc-chip[data-sel="{chip}"]'); pg.wait_for_timeout(200)
        check('a chip selects its request', st(pg)['selected'] == chip)
        tr = NB(pg.evaluate("()=>document.querySelector('#sc-tcard').innerText"))
        check('the trace names the request and its wait at the cap', f'#{chip + 1}' in tr and 'cap' in tr, tr[:200])
        rc = pg.evaluate("()=>{const r=document.querySelector('#sc-tcard').getBoundingClientRect(); return [r.top,r.bottom,r.height,innerHeight]}")
        check('1440 × 900: the trace card is in the viewport after a chip click (review r1, A1)', rc[2] > 40 and rc[0] >= 0 and rc[1] <= rc[3], rc)
        check('the chip is outlined', pg.evaluate(f"()=>document.querySelector('#sc-chips .sc-chip[data-sel=\"{chip}\"]').classList.contains('sel')"))
    box = pg.evaluate("()=>{const r=document.querySelector('#sc-tl').getBoundingClientRect(); return [r.x,r.y,r.width,r.height]}")
    pg.mouse.click(box[0] + box[2] * 0.3, box[1] + 5 * 6 + 3); pg.wait_for_timeout(200)
    check('a timeline row selects a request', st(pg)['selected'] is not None and st(pg)['selected'] != chip)
    rc = pg.evaluate("()=>{const c=document.querySelector('#sc-tcard'), r=c.getBoundingClientRect(); return [c.hidden,r.top,r.bottom,innerHeight,c.innerText.slice(0,60)]}")
    rid = st(pg)['selected']
    check('1440 × 900: after a row click the trace card is in the viewport and names the request (review r1, A1)', not rc[0] and rc[1] >= 0 and rc[2] <= rc[3] and f'#{rid + 1}' in rc[4], rc)
    ov = pg.evaluate("()=>{const s=window.__sc.state.selected, b=window.__sc.rowBox(s), c=document.querySelector('#sc-tcard').getBoundingClientRect(); return [!(c.right<=b.left||c.left>=b.right||c.bottom<=b.top||c.top>=b.bottom), b, [c.left,c.top,c.right,c.bottom]]}")
    check('1440 × 900: the trace card does not cover the selected row (review r2, N4)', not ov[0], ov)
    pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
    check('Escape closes the trace card', st(pg)['selected'] is None and pg.evaluate("()=>document.querySelector('#sc-tcard').hidden"))
    pg.mouse.click(box[0] + box[2] * 0.3, box[1] + 5 * 6 + 3); pg.wait_for_timeout(200)
    own = pg.evaluate("()=>{const e=document.querySelector('#sc-mbar .sc-ms-own'); return e?+e.dataset.sel:null}")
    if own is not None:
        click(pg, f'#sc-mbar .sc-ms-own[data-sel="{own}"]'); pg.wait_for_timeout(200)
        check('a memory block selects its request, outlined', st(pg)['selected'] == own and pg.evaluate(f"()=>document.querySelector('#sc-mbar .sc-ms-own[data-sel=\"{own}\"]').classList.contains('sel')"))
    pg.focus('#sc-tl'); pg.keyboard.press('ArrowDown'); pg.keyboard.press('ArrowDown'); pg.keyboard.press('Enter'); pg.wait_for_timeout(150)
    check('keyboard: arrows and Enter select a row', st(pg)['selected'] is not None)
    # the plan's Copy button
    mode(pg, 'challenge')
    for k_, v_ in (('budget', '2048'), ('cache', 'true')):
        pg.evaluate(f"()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-{k_}\"] button[data-v=\"{v_}\"]').click()"); pg.wait_for_timeout(120)
    act(pg, '#sc-rail [data-act=run]', 300); end(pg); pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000); pg.wait_for_timeout(200)
    pg.evaluate("()=>document.querySelector('#sc-ch-res [data-act=copyplan]').click()"); pg.wait_for_timeout(300)
    clip = pg.evaluate("()=>navigator.clipboard.readText()")
    check('Copy puts the plan on the clipboard', clip == pg.evaluate("()=>window.__sc.plan()") and 'Settings:' in clip, clip[:120])
    pg.evaluate("()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-cap\"] button[data-v=\"64\"]').click()"); pg.wait_for_timeout(120)
    act(pg, '#sc-rail [data-act=run]', 300); end(pg); pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000)
    check('cap 64 as well: met, but fewer changes pass (tier 2)', st(pg)['chTier']['tier'] == 2, st(pg)['chTier'])
    # R3.20: the same plan submitted again is not played again; its result shows at once
    tok1 = st(pg)['runTok']; act(pg, '#sc-rail [data-act=run]', 120); s = st(pg)
    check('R3.20: the same plan submitted again does not replay', not s['playing'] and s['playhead'] is None, (s['playing'], s['playhead']))
    pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000)
    check('R3.20: … and its result shows at once (tier 2 again)', st(pg)['chTier']['tier'] == 2, st(pg)['chTier'])
    pg.evaluate("()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-cap\"] button[data-v=\"32\"]').click()"); pg.wait_for_timeout(120)
    act(pg, '#sc-rail [data-act=run]', 120); s = st(pg)
    check('R3.20: the first plan run again after another (A → B → A): watched to the end, so it shows whole at once, no replay', not s['playing'] and s['playhead'] is None, (s['playing'], s['playhead']))
    pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000)
    check('R3.20: … and its verdict shows at once (tier 1 again)', st(pg)['chTier']['tier'] == 1, st(pg)['chTier'])
    pg.evaluate("()=>document.querySelector('#sc-rail .seg-ctl[data-key=\"ch-budget\"] button[data-v=\"512\"]').click()"); pg.wait_for_timeout(120)
    act(pg, '#sc-rail [data-act=run]', 120); s = st(pg)
    check('R3.20: a changed plan not run before plays its run', s['playing'] and s['playhead'] is not None, (s['playing'], s['playhead']))
    end(pg); pg.wait_for_function("()=>window.__sc.state.chTier!==null", timeout=20000)
    # hash round-trip
    mode(pg, 'sandbox'); act(pg, '#sc-rail [data-preset="2"]'); pg.locator('#sc-ctl-budget button[data-v="512"]').click(); pg.wait_for_timeout(300)
    hsh = pg.evaluate("()=>location.hash")
    check('the URL carries the setup in Scenarios (a scenario sets the load; the settings stay the learner’s)', 'apps_c_32_512_0_n' in hsh, hsh)
    pg.context.close()
    pg = page(b, 1440, 900, 'en', hash_=hsh.lstrip('#'))
    s = st(pg)
    check('a fresh page opens the shared setup', s['mode'] == 'sandbox' and s['load'] == 'apps' and s['cfg']['budget'] == 512 and not s['cfg']['cache'], s)
    pg.context.close()
    # =================== reduced motion ===================
    pg = page(b, 1440, 900, 'en', reduced=True)
    s = st(pg); check('reduced motion: no autoplay on entry', not s['playing'])
    act(pg, '#sc-play [data-act=play]'); s = st(pg)
    check('reduced motion: Play shows the whole run', not s['playing'] and s['playhead'] is None, s['playhead'])
    go(pg, 1); act(pg, '#sc-rail [data-pred="b"]'); go(pg, 2)
    act(pg, '#sc-rail [data-lever="2048"]'); s = st(pg)
    check('reduced motion: an option shows the whole run at once', not s['playing'] and s['playhead'] is None, s)
    check('reduced motion: the result is there at once', bool(pg.evaluate("()=>document.querySelector('#sc-rail [data-lever=\"2048\"] .lv-r').textContent")))
    act(pg, '#sc-play [data-act=fwd]'); check('reduced motion: ▸ moves the playhead', st(pg)['playhead'] is not None)
    pg.evaluate("()=>{const i=document.querySelector('#sc-scrub'); i.value=300; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(150)
    check('reduced motion: the slider moves the playhead', abs((st(pg)['playhead'] or 0) - 0.3 * 12) < 0.6, st(pg)['playhead'])
    pg.context.close()
    # =================== dark, once ===================
    pg = page(b, 1440, 900, 'en', dark=True)
    go(pg, 3); act(pg, '#sc-rail [data-lever="true"]'); end(pg)
    shot(pg, SH / 'dark_part4_1440_en.png')
    rules(pg, 'dark part 4', 'en')
    pg.context.close()
    # =================== phone: the mini result bar ===================
    pg = page(b, 390, 844, 'zh')
    pg.evaluate("()=>{const r=document.querySelector('#sc-mem'); r.scrollIntoView({block:'start'}); window.scrollBy(0,500); document.querySelector('#sc-app .bench').click()}"); pg.wait_for_timeout(500)
    dk = pg.evaluate("()=>{const d=document.querySelector('#dock'); return d&&!d.hidden?[d.innerText,document.querySelector('#dkStatus').textContent]:null}")
    check('390: the mini result bar appears when the strip scrolls away', dk is not None, dk)
    if dk: check('390: its status is short', len(dk[1]) <= 12, dk[1])
    pg.context.close()
    b.close()
# The owner's review of 2026-10-09 (R6.37, R6.33): no row of “Show the math” scrolls sideways at the sizes this test visits, in every
# guided part, Scenarios and the challenge. On a box too narrow for the one-line table, the lab sets the name on its own line and breaks
# a long formula before an operator, the numbers still under the terms they replace.
MSCROLL = r"""r=>{const out=[]; for(const d of document.querySelectorAll(r+' details.math')){if(!d.offsetParent) continue;
  for(const e of d.querySelectorAll('*')){const cs=getComputedStyle(e); if(!/auto|scroll/.test(cs.overflowX)||!e.offsetParent) continue;
    const o=e.scrollWidth-e.clientWidth; if(o>1) out.push(o+' px: '+(e.textContent||'').trim().replace(/\s+/g,' ').slice(0,40));}}
  const pg=document.documentElement.scrollWidth-innerWidth; if(pg>1) out.push('the page scrolls sideways by '+pg+' px'); return out;}"""
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for (_w, _h, _lang) in [(1440, 900, 'en'), (1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'en'), (390, 844, 'zh')]:
        _pg = _b.new_page(viewport={'width': _w, 'height': _h}); _pg.goto(URL); _pg.wait_for_timeout(250)
        _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(60)
        _pg.evaluate("()=>{location.hash='serving-capstone'}"); _pg.wait_for_timeout(500)
        _n = _pg.evaluate("()=>new Set([...document.querySelectorAll('#view-dyn aside.rail [data-go]')].map(e=>e.dataset.go)).size")
        _bad = []
        for _mode, _k in [('guided', k) for k in range(_n)] + [('sandbox', None), ('challenge', None)]:
            if _k is None or _k == 0:
                _pg.evaluate("m=>{const b=[...document.querySelectorAll('#view-dyn .modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", _mode); _pg.wait_for_timeout(300)
            if _k is not None:
                _pg.evaluate("k=>{const e=document.querySelector('#view-dyn aside.rail [data-go=\"'+k+'\"]'); if(e) e.click();}", _k)
            _pg.wait_for_timeout(700)
            _pg.evaluate("()=>document.querySelectorAll('#view-dyn details.math').forEach(d=>{d.open=true})"); _pg.wait_for_timeout(400)
            _bad += [f"{_mode}{'' if _k is None else ' part ' + str(_k + 1)}: {x}" for x in _pg.evaluate(MSCROLL, '#view-dyn')]
        check(f'{_w} {_lang}: no row of Show the math scrolls sideways, in any guided part, Scenarios or the challenge (the owner, 2026-10-09; R6.37)', not _bad, _bad[:6])
        _pg.close()
    _b.close()


# polish round 5 (2026-10-17, R3.2, R9.7, R7.9, R7.12, the owner's B4) and the table's fold (the owner, 2026-10-22, on a decision card:
# "收起来"): at 1440 × 900 the stage of every guided part but part 7 fits the window. Round 5 tightened the spacing and drew the timeline
# shorter, but the strip, the timeline and the KV pool, the whole-run table and Show the math still needed about 1,160 px (the stage has
# 814), so the stage scrolled 350–390 px inside (830–850 at part 7) and only the block each part's card talks about was brought whole on
# the screen. The whole-run table (about 350 px) now opens from a button in Show the math's row, the scope note sits in that row too, and
# on a wide stage the trace hint moves into the head; the timeline's rows are 5 px for the 37 requests and 4 px for the 48 (6 and 5 in
# round 5). Part 7's code panel still needs the stage to scroll inside, under a ceiling (this page's numbers plus about 20 px). The walk is
# on the whole site (built into out/ and wrapped with a doctype; the one-lab page lacks the “Read first” line, which moves every card
# 28 px), in the lab's real layout (window.__aspPrimer, as tests/kit_flow.py), entering each part as the polish probe stage.py does: Next
# when it is on, else the part's title; Chinese with reduced motion, English with motion (its scrolls are smooth). A phone keeps its rows.
subprocess.run([sys.executable, str(P / 'build.py'), 'out/site-serving-capstone.html'], check=True, cwd=P, capture_output=True)
(P / 'out' / 'wrapped-site-serving-capstone.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                                                          + (P / 'out' / 'site-serving-capstone.html').read_text() + '</body></html>')
SITE = 'file://' + str(P / 'out' / 'wrapped-site-serving-capstone.html')
FOCUS = {'mission': '#sc-tlw', 'predict': '#sc-tlw', 'budget': '#sc-tlw', 'cache': '#sc-mem', 'cbudget': '#sc-mem', 'cap': '#sc-strip', 'code': '#sc-wc', 'remember': None}
CEIL7 = {'zh': 420, 'en': 440}
WALK5 = """q=>{const s=document.getElementById('sc-stage'), r=s.getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect().bottom,
  lo=Math.max(r.top+s.clientTop,bar), hi=Math.min(r.bottom-s.clientTop,innerHeight), h=s.querySelector('.stage-head').getBoundingClientRect(), R=e=>{const b=e.getBoundingClientRect(); return [Math.round(b.top),Math.round(b.bottom)]};
  const st=window.__sc.state, f=q[st.partId], fe=f&&document.querySelector(f), cv=document.getElementById('sc-tl'), b0=window.__sc.rowBox(0);
  return {part:st.part+1, id:st.partId, top:Math.round(r.top), head:Math.round(h.top), bar:Math.round(bar), lo:Math.round(lo), hi:Math.round(hi), scroll:s.scrollHeight-s.clientHeight,
    focus:fe?R(fe):null, tl:R(cv), rh:b0?Math.round(b0.bottom-b0.top):null, drawn:Math.round(cv.width/Math.min(2,devicePixelRatio||1)), boxW:document.getElementById('sc-tlbox').clientWidth}}"""
# the fold: what the stage's foot shows, where the table is, and whether it is whole on the screen (inside the stage's visible part)
FOLD = """()=>{const s=document.getElementById('sc-stage'), w=document.getElementById('sc-mtw'), b=document.getElementById('sc-rtbtn'), m=document.querySelector('#sc-math>summary'), sc=document.getElementById('sc-scope'),
  ft=document.querySelector('#sc-stage>.sc-foot'), hn=document.querySelector('#sc-stage .sc-hint'), bar=document.querySelector('.topbar').getBoundingClientRect().bottom, sr=s.getBoundingClientRect(), r=w.getBoundingClientRect(),
  shown=e=>!!e&&e.getClientRects().length>0, mid=e=>{const q=e.getBoundingClientRect(); return Math.round((q.top+q.bottom)/2)}, card=document.querySelector('#sc-rail .step.current');
  const lo=Math.max(sr.top+s.clientTop,bar), hi=Math.min(sr.bottom-s.clientTop,innerHeight);
  return {open:window.__sc.state.runOpen, hidden:w.hidden, table:shown(w), btn:shown(b), exp:b.getAttribute('aria-expanded'), mark:getComputedStyle(b,'::before').content, btnText:b.textContent.trim(),
    whole:shown(w)&&r.top>=lo-1&&r.bottom<=hi+1, gap:shown(w)?Math.round(r.top-ft.getBoundingClientRect().bottom):null, row:shown(b)?[mid(m),mid(b),mid(sc)]:null,
    top:Math.round(sr.top), bar:Math.round(bar), hint:shown(hn)&&!!hn.closest('.stage-head'), low:shown(document.querySelector('#sc-stage>.sc-low')),
    xs:shown(b)?[m,b,sc].map(e=>Math.round(e.getBoundingClientRect().left)):null, scopeAbove:sc.getBoundingClientRect().bottom<=m.getBoundingClientRect().top+1,
    card:card?Math.round(card.getBoundingClientRect().top):null, part:window.__sc.state.part+1}}"""
TIP = "()=>{const t=document.querySelector('#tip'); return t&&!t.hidden?t.innerText.trim():null}"
LINKS = "()=>[...document.querySelectorAll('#sc-rail [data-jump=table]')].filter(e=>e.offsetParent).length"


def enter(pg, i, rm):
    """Next when it is on, else the part's title (as the polish probe stage.py); i is the 0-based part"""
    nb = pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
    if nb.count(): nb.first.click()
    else: pg.evaluate(f"()=>document.querySelector('#sc-rail .steps>.step:nth-child({i + 1}) .step-h').click()")
    pg.wait_for_timeout(1500 if rm == 'reduce' else 2500)


def site(b, w, h, lang, rm='reduce', resume=False):
    pg = b.new_page(viewport={'width': w, 'height': h}, reduced_motion=rm); pg.add_init_script('window.__aspPrimer=true' + ('; window.__aspResume=true' if resume else ''))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang} site: {e}'))
    pg.goto(SITE); pg.wait_for_timeout(600); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate("()=>{location.hash='serving-capstone'}"); pg.wait_for_timeout(500)
    g = pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if g.count(): g.first.click(); pg.wait_for_timeout(700)
    return pg


with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for _lang, _rm in (('zh', 'reduce'), ('en', 'no-preference')):
        _pg = site(_b, 1440, 900, _lang, _rm)
        _n = _pg.evaluate("()=>document.querySelectorAll('#sc-rail .steps>.step').length"); _rows = [_pg.evaluate(WALK5, FOCUS)]
        if _rm != 'reduce':   # R7.9: playing the run leaves every block's height alone (the stage's content keeps its height)
            _pg.evaluate("()=>{window.__H5=[]; const s=document.getElementById('sc-stage'), f=()=>{window.__H5.push(s.scrollHeight); if(window.__H5.length<240) requestAnimationFrame(f);}; document.querySelector('#sc-play [data-act=play]').click(); f();}")
            _pg.wait_for_timeout(4500); _hs = _pg.evaluate("()=>window.__H5"); _pg.evaluate("()=>{const b=document.querySelector('#sc-play [data-act=play]'); if(window.__sc.state.playing) b.click();}")
            check(f'1440 {_lang} part 1: playing the run leaves the stage’s content height alone (R7.9)', len(_hs) > 30 and max(_hs) - min(_hs) <= 1, (len(_hs), min(_hs or [0]), max(_hs or [0])))
        for _i in range(1, _n):
            enter(_pg, _i, _rm); _rows.append(_pg.evaluate(WALK5, FOCUS))
        check(f'1440 {_lang} walk: entering every part, the stage’s top at its sticky line (70 px) or lower and its head below the top bar (R3.2, R9.7, B4)',
              len(_rows) == _n == 8 and all(r['part'] == i + 1 and r['top'] >= max(68, r['bar'] - 1) and r['head'] >= r['bar'] - 1 for i, r in enumerate(_rows)), [(r['part'], r['top']) for r in _rows])
        for _r in _rows:
            if _r['id'] == 'code':   # part 7: the code panel stands above the timeline (DOM order), the panel is the part's figure
                check(f'1440 {_lang} walk part 7: what the stage cannot hold scrolls inside it, at most {CEIL7[_lang]} px (B4; 830–850 px before the fold)', _r['scroll'] <= CEIL7[_lang], _r)
            else:
                check(f'1440 {_lang} walk part {_r["part"]}: the whole stage fits the window, nothing scrolls inside it (the owner, 2026-10-22; 350–390 px before the fold)', _r['scroll'] <= 0, _r)
                check(f'1440 {_lang} walk part {_r["part"]}: the timeline (the main figure) is whole on the screen when the part is entered (B4, R7.12)',
                      _r['tl'][0] >= _r['lo'] - 1 and _r['tl'][1] <= _r['hi'] + 1, _r)
            if _r['focus']:
                check(f'1440 {_lang} walk part {_r["part"]}: the block the card talks about ({FOCUS[_r["id"]]}) is whole on the screen when the part is entered (B4, R7.12)',
                      _r['focus'][0] >= _r['lo'] - 1 and _r['focus'][1] <= _r['hi'] + 1, _r)
            check(f'1440 {_lang} walk part {_r["part"]}: the timeline is drawn for its width, its rows {4 if _r["part"] in (4, 5, 6) else 5} px (37 or 48 requests)',
                  _r['drawn'] == _r['boxW'] and _r['rh'] == (4 if _r['part'] in (4, 5, 6) else 5), _r)
        _pg.close()

    # the fold (the owner, 2026-10-22): the table hides until its button is pressed; the button, or the card's sentence that names the
    # table, opens it whole on the screen with the stage at its sticky line; the choice holds across Next, a language switch and a reload
    for _lang in ('zh', 'en'):
        T = f'1440 {_lang} fold'
        _pg = site(_b, 1440, 900, _lang, resume=True)
        _f = _pg.evaluate(FOLD)
        check(f'{T}: the table is folded at first, its button reads “{Z(_lang, "Show the whole-run table", "展开整次运行的表格")}” with a + (as Show the math)',
              not _f['open'] and _f['hidden'] and not _f['table'] and _f['btn'] and _f['exp'] == 'false' and _f['mark'] == '"+"' and _f['btnText'] == Z(_lang, 'Show the whole-run table', '展开整次运行的表格'), _f)
        check(f'{T}: Show the math, the table’s button and the scope note share one row under the stage, in that order', _f['row'] and max(_f['row']) - min(_f['row']) <= 6 and _f['xs'][0] < _f['xs'][1] < _f['xs'][2], _f)
        check(f'{T}: on a wide stage the trace hint sits in the head and the row under the stage that held it is gone', _f['hint'] and not _f['low'], _f)
        _pg.evaluate("()=>document.getElementById('sc-rtbtn').click()"); _pg.wait_for_timeout(700); _f = _pg.evaluate(FOLD)
        check(f'{T}: pressed at the page’s top, the button opens the table (−) right under its row, whole on the screen, the stage at its sticky line',
              _f['open'] and _f['table'] and _f['exp'] == 'true' and _f['mark'] == '"−"' and 0 <= _f['gap'] <= 12 and _f['whole'] and _f['top'] >= 68, _f)
        rules(_pg, f'{T} open table', _lang)
        enter(_pg, 1, 'reduce'); _f = _pg.evaluate(FOLD)
        check(f'{T}: Next keeps the table open', _f['part'] == 2 and _f['open'] and _f['table'], _f)
        _o = 'en' if _lang == 'zh' else 'zh'
        _pg.click(f'.langsw button[data-lang="{_o}"]'); _pg.wait_for_timeout(700); _f = _pg.evaluate(FOLD)
        check(f'{T}: a language switch keeps it open', _f['open'] and _f['table'] and _f['btnText'] == Z(_o, 'Show the whole-run table', '展开整次运行的表格'), _f)
        _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(700)
        _pg.reload(); _pg.wait_for_timeout(1500); _f = _pg.evaluate(FOLD)
        check(f'{T}: a reload keeps it open (the lab’s saved state)', _f['part'] == 2 and _f['open'] and _f['table'] and _f['exp'] == 'true', _f)
        _pg.locator('#sc-rtbtn').click(); _pg.wait_for_timeout(500); _f = _pg.evaluate(FOLD)
        check(f'{T}: pressed again, the button folds the table (+)', not _f['open'] and not _f['table'] and _f['exp'] == 'false' and _f['mark'] == '"+"', _f)
        # part 3: a budget's cost sentence names the table, as a link while it is folded on a desktop
        enter(_pg, 2, 'reduce'); _pg.evaluate("()=>document.querySelectorAll('#sc-rail .step.current .levers button')[1].click()"); _pg.wait_for_timeout(500); end(_pg); _pg.wait_for_timeout(400)
        check(f'{T}: part 3, a budget’s cost sentence names the whole-run table as a link', _pg.evaluate(LINKS) == 1, _pg.evaluate("()=>document.querySelector('#sc-rail [data-obs]').innerText"))
        if _pg.evaluate(LINKS):
            _l = _pg.locator('#sc-rail [data-jump=table]:visible').first; _l.hover(); _pg.wait_for_timeout(500); _tp = _pg.evaluate(TIP)
            check(f'{T}: the link’s tooltip says what a click does (R1.1)', _tp is not None and Z(_lang, 'Click to open the whole-run table.', '点击展开整次运行的表格。') in _tp, _tp)
            _c0 = _pg.evaluate(FOLD)['card']; _l.click(); _pg.wait_for_timeout(900); _f = _pg.evaluate(FOLD)
            check(f'{T}: the link opens the table whole on the screen, the stage at its sticky line, the card’s head never pushed under the top bar', _f['open'] and _f['whole'] and _f['top'] >= 68 and _f['card'] >= min(_c0, _f['bar'] + 8) - 1, (_c0, _f))
        _pg.close()
    # a 900 px window: the stage is narrower, so the trace hint keeps its row under the stage; the button still opens the table in view
    _pg = site(_b, 900, 900, 'en'); _f = _pg.evaluate(FOLD)
    check('900 en fold: the trace hint keeps its row under the stage; the table is folded', not _f['hint'] and _f['low'] and not _f['table'] and _f['btn'], _f)
    _pg.evaluate("()=>document.getElementById('sc-rtbtn').click()"); _pg.wait_for_timeout(700); _f = _pg.evaluate(FOLD)
    check('900 en fold: the button opens the table whole on the screen, the stage at its sticky line', _f['open'] and _f['whole'] and _f['top'] >= 68, _f)
    _pg.close()
    # a phone keeps its layout: no button and no table (the scope note above Show the math, as before), its timeline rows 4 px for 37 requests
    for _lang in ('zh', 'en'):
        _pg = site(_b, 390, 844, _lang); _r = _pg.evaluate(WALK5, FOCUS); _f = _pg.evaluate(FOLD)
        if _lang == 'zh': check('390 part 1: a phone keeps its timeline rows (4 px for 37 requests)', _r['rh'] == 4 and _r['drawn'] == _r['boxW'], _r)
        _pg.evaluate("()=>document.getElementById('sc-rtbtn').click()"); _pg.wait_for_timeout(400); _g = _pg.evaluate(FOLD)
        check(f'390 {_lang} fold: a phone shows no table and no button, even with the table open, and the scope note stands above Show the math',
              not _f['btn'] and not _f['table'] and _f['scopeAbove'] and _g['open'] and not _g['table'], (_f, _g))
        _pg.close()
    _b.close()


check('no page errors', not errs, errs[:5])
print(f'{n} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
