"""Flow test for the contamination lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- the phone first screen (m2 template, R7.3): the parts' bar with Next, then the stage;
- part 1: three metric cards, the score chart with the gain on all questions only (no clean set, no real-gain line yet: R3.8);
- part 2: the six sample questions, Next disabled with the kit's why line (steps({nextWhy})) until a second sample; each sample's
  verdict (caught / missed / false alarm / right) and the leak tag; the reworded sample marks nothing at n = 8;
- part 3: the overlap formula; Next waits for a prediction; the prediction is remembered;
- part 4: the verdict on the prediction (right or wrong, with the numbers 924 and 124); Next waits for a second setting; the grid's
  legend counts match the engine; “Show me” sets n = 13;
- part 5: the clean-set row appears with the real-gain line; Next waits for a second n;
- part 6: three levers, each with its own result line once tried; the measured-only observation shows on entry (review b10r1 S4) and
  changes with the lever; Next waits for the
  clean-set rule; 32 / 31, 98 / 92, 95 / 1 in the result lines;
- part 7: three misconceptions, “See the scenarios”, the next-lab footer in plain text (eval-capstone is not live);
- Scenarios: six, single choice, the insight for each; the URL carries the setup (only in Scenarios) and a fresh page restores it; a
  change under Settings does not move the page (R3.2); the language switch keeps the setup;
- the challenge: the three choices in the card, Settings locked (R4.2), the flagged count but no right-call count before Submit (R3.8),
  the tiers (not met with each failing candidate, met but a setting that keeps more exists, best), the brief chart (the 3 passing
  setups, the initial setup and the pick, tagged), “Show the answer”, the capability card;
- R3.20: the lab has no animation or player; a part change, a click, a repeated click, a scenario or a submit redraws the stage at
  once, and nothing in the stage animates two frames after the click (the kit's cue outline excepted);
- the dock (A.dock) at 390 once the stage scrolls away, with a short status;
- the writing rules on every visible text and tooltip (Chinese spacing and punctuation, two colons, no “toy”, no 控件 / Controls, no
  sentence about the page itself, no company names, the brief's phrases), colours only for failures and the challenge's goal (R5.6).
Screenshots: shots/contamination/.
Run: python3 tests/contamination_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/contamination.html', '--only', 'contamination'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-contamination.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'contamination.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-contamination.html')
SH = P / 'shots' / 'contamination'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='contamination'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ctm.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ctm-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ctm-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('ctm-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||res.innerText};}""")


def settle(pg, ms=2500):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(220); settle(pg)


def seg(pg, key, v, root='#ctm-rail'):
    click(pg, f'{root} .seg-ctl[data-key="{key}"] button[data-v="{v}"]')


def mode(pg, m):
    click(pg, f'#ctm-app .modes button[data-mode="{m}"]')


# R3.20: no animation in this lab. Two frames after a click nothing in the stage may be animating (the kit's cue outline and the
# metric cards' change label, which fades in, excepted).
STILL = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  return document.querySelector('#ctm-stage').getAnimations({subtree:true}).filter(a=>a.animationName!=='cue'&&!(a.effect&&a.effect.target&&a.effect.target.closest('.metrics'))).map(a=>a.animationName||a.transitionProperty||'?');}"""


def still(pg, sel, tag, what):
    an = pg.evaluate(STILL, sel); pg.wait_for_timeout(220); settle(pg)
    check(f'{tag} {what}: the stage is redrawn at once, nothing replays (R3.20)', not an, an)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里/,'phrase banned by the brief'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/OpenAI|Anthropic|DeepMind|Microsoft|\\bMeta\\b/,'company name'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part|square)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, [lang="en"]')||!e.getClientRects().length) continue; const t=n.textContent;
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
COLORS = """()=>{const cs=getComputedStyle(document.documentElement), get=v=>cs.getPropertyValue(v).trim();
 const toRGB=h=>{const d=document.createElement('i'); d.style.color=h; document.body.appendChild(d); const c=getComputedStyle(d).color; d.remove(); return c;};
 const blue=toRGB(get('--c-weights')), amber=toRGB(get('--c-kv')), red=toRGB(get('--c-oom')), violet=toRGB(get('--c-overhead'));
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|bad|miss|wrong|\\bwr\\b|ctm-sw-wr|ctm-leak|ex-res|ctm-pv|ctm-dot|ctm-bar|fail/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ctm-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':'');
   if(!ok[c].test(cls)) out.push(p+' '+c+' on '+e.tagName+'.'+cls.trim().slice(0,60));}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ctm-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ctm-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#ctm-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: Chinese spacing: {h}')
            for h in pg.evaluate(PUNCT, sel): rule_hits.append(f'{where}: half-width punctuation in Chinese: {h}')
    for h in pg.evaluate(COLORS): rule_hits.append(f'{where}: colour used for something else: {h}')
    for h in pg.evaluate(OVER): rule_hits.append(f'{where}: text wider than its box: {h}')
    for h in pg.evaluate(HELP): rule_hits.append(f'{where}: help cursor without a tooltip: {h}')
    o = overflow(pg)
    if o > 0: rule_hits.append(f'{where}: horizontal overflow {o}px')


def next_part(pg):
    click(pg, '#ctm-rail .step.current [data-nav="next"]')


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for lang in LANGS:
        for (w, h) in SIZES:
            tag = f'{lang} {w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag} opens in guided part 1', st['mode'] == 'guided' and st['step'] == 0, st)
            # ---- part 1
            check(f'{tag} part 1: three metric cards', pg.evaluate("()=>document.querySelectorAll('#ctm-metrics .metric').length") == 3)
            s1 = text(pg, '#ctm-pan-score')
            check(f'{tag} part 1: the score chart has all questions, not the clean set or the real gain yet (R3.8)',
                  Z(lang, 'All 1,000 questions', '全部 1,000 道题') in pg.evaluate("()=>document.querySelector('#ctm-score').textContent") and Z(lang, 'Clean set', '干净子集') not in pg.evaluate("()=>document.querySelector('#ctm-score').textContent")
                  and not pg.evaluate("()=>!!document.querySelector('#ctm-score .ctm-true')"), s1)
            check(f'{tag} part 1: 10.9 points in the card (no sign with “higher”, review b10r1 O12)', Z(lang, '10.9 points higher', '高了 10.9 分') in text(pg, '#ctm-rail .step.current') and '+10.9' not in text(pg, '#ctm-rail .step.current'))
            if w == 390:
                fs = pg.evaluate("()=>{const r=document.querySelector('#ctm-rail').getBoundingClientRect(), s=document.querySelector('#ctm-stage').getBoundingClientRect(); return {rail:Math.round(r.top), stage:Math.round(s.top)}}")
                check(f'{tag} phone: the parts come first, then the stage (m2, R7.3)', fs['rail'] < fs['stage'], fs)
            rules(pg, f'{tag} part 1', lang); shot(pg, f'{lang}-{w}-p1')
            still(pg, '#ctm-rail .step.current [data-nav="next"]', tag, 'part 1 → 2')
            # ---- part 2: samples
            st = state(pg); check(f'{tag} part 2 open', st['step'] == 1, st)
            check(f'{tag} part 2: Next waits for a second sample, with the why line', nexts(pg)[-1] is True and why(pg), why(pg))
            check(f'{tag} part 2: the pair panel first, the six samples in the card', vis(pg, '#ctm-pan-pair') and pg.evaluate("()=>document.querySelectorAll('#ctm-rail .seg-ctl[data-key=\"s\"] button').length") == 6)
            verdicts = {}
            for v in range(6):
                still(pg, f'#ctm-rail .seg-ctl[data-key="s"] button[data-v="{v}"]', tag, f'sample {v + 1}')
                verdicts[v] = (text(pg, '#ctm-pair .ctm-pv'), pg.evaluate("()=>document.querySelectorAll('#ctm-pair mark').length"), pg.evaluate("()=>document.querySelector('#ctm-pair .ctm-pv').classList.contains('bad')"))
            check(f'{tag} part 2: copied samples caught', all(Z(lang, 'the leak is caught', '泄漏被抓到了') in verdicts[i][0] for i in (0, 1, 2)), verdicts)
            check(f'{tag} part 2: the reworded sample marks nothing and is a missed leak, in red', verdicts[3][1] == 0 and Z(lang, 'a leak is missed', '漏掉了一道泄漏题') in verdicts[3][0] and verdicts[3][2], verdicts[3])
            check(f'{tag} part 2: sample 5 is right, sample 6 a false alarm in red', Z(lang, 'right, it is clean', '判对了') in verdicts[4][0] and Z(lang, 'false alarm', '误报') in verdicts[5][0] and verdicts[5][2] and not verdicts[4][2], (verdicts[4], verdicts[5]))
            check(f'{tag} part 2: Next enabled, observation shown', nexts(pg)[-1] is False and vis(pg, '#ctm-rail [data-obs]'))
            rules(pg, f'{tag} part 2', lang)
            next_part(pg)
            # ---- part 3: prediction
            check(f'{tag} part 3: the overlap formula', pg.evaluate("()=>!!document.querySelector('#ctm-rail .step.current math, #ctm-rail .step.current .ctm-math-line')"))
            check(f'{tag} part 3: Next waits for a prediction', nexts(pg)[-1] is True and why(pg), why(pg))
            pred = 'long' if lang == 'en' else 'more'
            click(pg, f'#ctm-rail [data-pred="{pred}"]')
            check(f'{tag} part 3: the prediction is remembered, Next enabled', state(pg)['pred'] == pred and nexts(pg)[-1] is False, state(pg))
            rules(pg, f'{tag} part 3', lang)
            next_part(pg)
            # ---- part 4: the verdict, the grid
            c4 = text(pg, '#ctm-rail .step.current')
            check(f'{tag} part 4: the verdict on the prediction with 924 and 124', ('924' in c4 and '124' in c4) and ((Z(lang, 'Right', '对了') in c4) if pred == 'more' else (Z(lang, 'Not quite', '不太对') in c4 or Z(lang, 'Not', '不') in c4)), c4[:200])
            check(f'{tag} part 4: the grid is shown, Next waits for a second setting', vis(pg, '#ctm-pan-grid') and nexts(pg)[-1] is True)
            leg = text(pg, '#ctm-g-leg'); d3 = pg.evaluate("()=>window.__ctm.detect('main',3,0)")
            check(f'{tag} part 4: legend counts = the engine (n = 3)', f"{d3['caught']}" in leg and f"{d3['fa']}" in leg, leg)
            for v in ('8', '13'):
                still(pg, f'#ctm-rail .seg-ctl[data-key="n"] button[data-v="{v}"]', tag, f'part 4 n = {v}')
            still(pg, '#ctm-rail .seg-ctl[data-key="n"] button[data-v="13"]', tag, 'part 4 the same n again')
            check(f'{tag} part 4: Next enabled after a second setting; the metric deltas say points or counts', nexts(pg)[-1] is False and '%' not in text(pg, '#ctm-metrics .d.on') if pg.query_selector('#ctm-metrics .d.on') else nexts(pg)[-1] is False)
            rules(pg, f'{tag} part 4', lang); shot(pg, f'{lang}-{w}-p4')
            next_part(pg)
            # ---- part 5: clean set
            check(f'{tag} part 5: the clean-set row and the real-gain line', Z(lang, 'Clean set', '干净子集') in pg.evaluate("()=>document.querySelector('#ctm-score').textContent") and pg.evaluate("()=>!!document.querySelector('#ctm-score .ctm-true')"))
            check(f'{tag} part 5: Next waits for a second n', nexts(pg)[-1] is True)
            still(pg, '#ctm-rail .seg-ctl[data-key="n"] button[data-v="8"]', tag, 'part 5 n = 8')
            check(f'{tag} part 5: +6.5 ± 2.9 at n = 8', '+6.5 ± 2.9' in pg.evaluate("()=>document.querySelector('#ctm-score').textContent") and nexts(pg)[-1] is False)
            rules(pg, f'{tag} part 5', lang)
            next_part(pg)
            # ---- part 6: the gate
            check(f'{tag} part 6: three levers, the gate panel', pg.evaluate("()=>document.querySelectorAll('#ctm-rail [data-lever]').length") == 3 and vis(pg, '#ctm-pan-gate'))
            o6 = text(pg, '#ctm-rail [data-obs]')
            check(f'{tag} part 6: the measured-only observation shows on entry with 32 / 31, Next waits for the clean rule',
                  vis(pg, '#ctm-rail [data-obs]') and Z(lang, 'about half', '大约一半的时候') in o6 and '32' in o6 and '31' in o6 and nexts(pg)[-1] is True, o6[:200])
            still(pg, '#ctm-rail [data-lever="all"]', tag, 'part 6 all questions')
            o6 = text(pg, '#ctm-rail [data-obs]')
            check(f'{tag} part 6: the observation follows the lever', vis(pg, '#ctm-rail [data-obs]') and Z(lang, 'in most runs', '大多数时候') in o6 and nexts(pg)[-1] is True, o6[:200])
            still(pg, '#ctm-rail [data-lever="clean"]', tag, 'part 6 clean set')
            lv = text(pg, '#ctm-rail .levers')
            check(f'{tag} part 6: result lines 32 / 31, 98 / 92, 95 / 1', all(re.search(x, lv) for x in (r'(new model|新模型) 32\b', r'(memorizer|只背题的模型) 31\b', r'(new model|新模型) 98\b', r'(memorizer|只背题的模型) 92\b', r'(new model|新模型) 95\b', r'(memorizer|只背题的模型) 1\b')), lv)
            check(f'{tag} part 6: Next enabled', nexts(pg)[-1] is False)
            rules(pg, f'{tag} part 6', lang); shot(pg, f'{lang}-{w}-p6')
            next_part(pg)
            # ---- part 7
            c7 = text(pg, '#ctm-rail .step.current')
            check(f'{tag} part 7: three misconceptions and “See the scenarios”', c7.count('✗') == 3 and pg.query_selector('#ctm-rail [data-mode-go="sandbox"]') is not None, c7[-200:])
            nl = pg.evaluate("()=>{const e=document.querySelector('.next-lab'); return e?{t:e.innerText, a:[...e.querySelectorAll('a,[data-go]')].map(x=>x.textContent)}:null}")
            check(f'{tag} part 7: next-lab footer names eval-capstone as not built yet', nl is not None and Z(lang, 'not built yet', '还没上线') in nl['t'], nl)
            rules(pg, f'{tag} part 7', lang)
            # ---- Scenarios
            mode(pg, 'sandbox')
            check(f'{tag} scenarios: six, single choice', pg.evaluate("()=>document.querySelectorAll('#ctm-rail [data-preset]').length") == 6)
            ins = []
            for i in range(6):
                still(pg, f'#ctm-rail [data-preset="{i}"]', tag, f'scenario {i + 1}')
                ins.append(text(pg, '#ctm-insight'))
                check(f'{tag} scenario {i + 1}: pressed alone', pg.evaluate("i=>[...document.querySelectorAll('#ctm-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed')==='true')", i) == [j == i for j in range(6)])
            check(f'{tag} scenarios: six different insights', len(set(ins)) == 6, ins)
            click(pg, '#ctm-rail [data-preset="1"]')
            hsh = pg.evaluate("()=>location.hash"); check(f'{tag} scenarios: the URL carries the setup', 'contamination' in hsh and '~' in hsh, hsh)
            y0 = pg.evaluate('()=>scrollY')
            if w == 1440:
                seg(pg, 'm', '5', '#ctm-controls'); check(f'{tag} settings: a change does not move the page (R3.2)', abs(pg.evaluate('()=>scrollY') - y0) < 4)
                hsh = pg.evaluate("()=>location.hash")
            rules(pg, f'{tag} scenarios', lang); shot(pg, f'{lang}-{w}-sb')
            st_sb = state(pg)
            pg2 = page(b, w, h, lang, hsh.lstrip('#')); s2 = state(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup', s2['mode'] == 'sandbox' and all(s2[k] == st_sb[k] for k in ('n', 't', 'bench', 'rule', 'm')), (s2, st_sb))
            pg2.close()
            other = 'zh' if lang == 'en' else 'en'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
            s3 = state(pg); check(f'{tag} language switch keeps the setup', s3['mode'] == 'sandbox' and all(s3[k] == st_sb[k] for k in ('n', 't', 'bench', 'rule', 'm')), s3)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
            if w == 390:
                pg.evaluate("()=>window.scrollTo(0,document.querySelector('#ctm-controls').getBoundingClientRect().top+scrollY)"); pg.wait_for_timeout(400)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return d?{shown:!!d.getClientRects().length&&!d.hidden&&getComputedStyle(d).display!=='none', t:d.innerText}:null}")
                check(f'{tag} dock: shown once the stage scrolls away, short', dk is not None and dk['shown'] and 0 < len(dk['t']) < 160, dk)
                rules(pg, f'{tag} dock', lang)
                pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
            # ---- challenge
            mode(pg, 'challenge')
            st = state(pg); check(f'{tag} challenge opens at the initial setup', (st['n'], st['t'], st['m']) == (13, 0, 3), st)
            check(f'{tag} challenge: Settings locked (R4.2)', vis(pg, '#ctm-lock') and pg.evaluate("()=>[...document.querySelectorAll('#ctm-controls .lockable')].every(e=>e.classList.contains('locked')||e.closest('.locked')||e.querySelector('button[disabled]'))"))
            card = text(pg, '#ctm-rail .railcard')
            check(f'{tag} challenge: three candidates and the three choices in the card', all(pg.query_selector(f'#ctm-rail .seg-ctl[data-key="{k}"]') for k in ('n', 't', 'm')) and card.count(Z(lang, 'should', '应该')) + card.count(Z(lang, 'should not', '不该')) >= 3, card[:300])
            check(f'{tag} challenge: before Submit the flagged count, no right calls (R3.8)', '149' in card and not vis(pg, '#ctm-strips') and Z(lang, 'right call', '结论正确') not in text(pg, '#ctm-stage'), card[-300:])
            rules(pg, f'{tag} challenge before', lang)
            still(pg, '#ctm-rail [data-act="submit"]', tag, 'submit the initial setup')
            r0 = text(pg, '#ctm-ch-res')
            check(f'{tag} challenge: not met names the failing candidates', Z(lang, 'Not yet', '还没达到') in r0 and '76' in r0 and '84' in r0, r0[:200])
            check(f'{tag} challenge: Show the answer offered after a miss', pg.query_selector('#ctm-rail [data-act="reveal"]') is not None)
            seg(pg, 'n', '3'); seg(pg, 't', '20'); seg(pg, 'm', '5'); click(pg, '#ctm-rail [data-act="submit"]')
            r1 = text(pg, '#ctm-ch-res')
            check(f'{tag} challenge: met, but a setting keeping more exists', Z(lang, 'Goal met', '达成目标') in r1 and Z(lang, 'among the settings you can pick', '在能选的设置里') in r1 and '17' in r1, r1[:240])
            bf = brief(pg)
            check(f'{tag} challenge: brief chart = 3 passing + the initial setup, pick tagged', bf is not None and bf['rows'] == 4 and bf['ok'] == 3 and bf['me'] == 1 and bf['init'] == 1 and bf['stray'] == 0 and '44' in bf['note'], bf)
            seg(pg, 'n', '8'); seg(pg, 't', '0'); click(pg, '#ctm-rail [data-act="submit"]')
            r2 = text(pg, '#ctm-ch-res')
            check(f'{tag} challenge: the best pick says so', Z(lang, 'Goal met', '达成目标') in r2 and Z(lang, 'Among the settings you can pick here, no passing setting keeps more', '在这道题能选的设置里，没有哪种能达标的设置') in r2 and '786' in r2 and '789' in r2 and Z(lang, 'instead of', '而不是') not in r2, r2[:240])
            check(f'{tag} challenge: after Submit the strips show right calls', vis(pg, '#ctm-strips'))
            rules(pg, f'{tag} challenge after', lang); shot(pg, f'{lang}-{w}-ch')
            check(f'{tag} capability card', pg.query_selector('#ctm-rail [data-exits]') is not None and Z(lang, 'Explain it', '讲清楚') in text(pg, '#ctm-rail [data-exits]'))
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
            s4 = state(pg); check(f'{tag} challenge: the language switch keeps the answer', s4['chSubmitted'] and (s4['n'], s4['t'], s4['m']) == (8, 0, 5), s4)
            pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
            pg.close()
            pg = page(b, w, h, lang); mode(pg, 'challenge'); click(pg, '#ctm-rail [data-act="submit"]'); click(pg, '#ctm-rail [data-act="reveal"]')
            st = state(pg); r3 = text(pg, '#ctm-ch-res')
            check(f'{tag} challenge: Show the answer shows the best setting', st['chReveal'] and '8' in r3 and brief(pg) is not None, (st, r3[:200]))
            rules(pg, f'{tag} challenge answer', lang)
            pg.close()
    b.close()

# polish round 3 (2026-10-16, B4 / R7.12, the owner: "the panel was pushed down, I did not see there was more below"): at 1440 × 900,
# entering each listed guided part the way a reader does (Next), the stage needs no scrolling inside itself and nothing in it
# scrolls either; in Chinese it could still grow by 30 px (its max-height minus the height its content and borders need), in
# English it does not scroll; its main picture lies inside it and on the screen. The real layout: window.__aspPrimer, as the probe.
_FIT_PARTS = [7]
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
            _pg.evaluate("()=>{location.hash='contamination'}"); _pg.wait_for_timeout(500)
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
for e in errs: print('ERROR', e)
check('no page errors', not errs, errs[:3])
for r in rule_hits[:40]: print('RULE', r)
check('writing and layout rules', not rule_hits, len(rule_hits))
print(f'{n - len(fails)} of {n} checks passed')
sys.exit(1 if fails else 0)
