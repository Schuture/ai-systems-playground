"""Flow test for the llm-judge lab (CONTRACT §6).

Every mode and every guided part at 1440×900 and 390×844, in EN and ZH:
- no page errors, no horizontal overflow, no text wider than its box (R7.8), a tooltip for every help cursor (R1.1);
- the phone first screen (m2 template, R7.3): the parts' bar with Next, then the chart;
- part 1: Next disabled with the kit's why line (steps({nextWhy})) until a prediction; people's win rate appears nowhere (R3.8);
- part 2: the formula, the counts and the interval; “Next pair” moves the pair card and the outlined mark;
- part 3: Next waits for both orders; random order is a real option; the result on each option's title row; the inconsistent pairs
  named beside the marks;
- part 4: Next waits for length taken out; then people's line, the verdict on the prediction with the right answer, and why;
- part 5: the agreement view (two 10 × 10 blocks, the subtitle names the judge checked); Next waits for a second judge; the second
  person counted on the judge's pairs; the MT-Bench reference and the three practices;
- part 6: three misconceptions, “See the scenarios”, the next-lab footer; each “Show in the chart” puts its picture back;
- Scenarios: four, single choice, the insight for each; the URL carries the setup and a fresh page restores it; a change under
  Settings does not move the page (R3.2);
- the challenge: only the three choices in the card, Settings locked (R4.2), only the cost before Submit (no gap, no people's line, no
  win rate: R3.8), the three feedback tiers (R4.4), tier 2 saying in its first line that a cheaper setup exists, “Show the answer”;
  the brief chart (the owner's review of batch 8): only the 4 passing setups, the initial setup and the pick, both tagged (R5.4), no
  other row, a note that counts the setups left out (7, or 6 when the pick is another setup that misses), a title without “each
  setup”; the random order alone named in words with its gap in a hover; the capability card;
  the language switch keeps the answer; the stage shows only the learner's own setup;
- the mini result bar at 390 once the chart scrolls away, with a short status;
- the writing rules on every visible text and tooltip (Chinese spacing and punctuation, two colons, no “toy”, no 控件 / Controls, no
  sentence about the page itself), no company names, and colours only for the challenge's goal (R5.6).
Screenshots: shots/llm-judge/.
Run: python3 tests/llm-judge_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'llm-judge').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/llm-judge.html', '--only', 'llm-judge'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'llm-judge' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'llm-judge.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'llm-judge' / 'flow-wrapped.html')
SH = P / 'shots' / 'llm-judge'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='llm-judge'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__ljg.state()")
vis = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0}", sel)
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-'):''}", sel)
# the challenge's brief chart (A.ui.cmpChart({brief:true})): its rows by kind and the note under it
brief = lambda pg: pg.evaluate("""()=>{const res=document.getElementById('ljg-ch-res'), c=res&&res.querySelector(':scope .cmp:not(details .cmp)'); if(!c) return null;
  const rows=[...c.querySelectorAll('.cmp-row')], has=(r,k)=>r.classList.contains(k), init=r=>!!r.querySelector('.cmp-tag');
  return {rows:rows.length, ok:rows.filter(r=>has(r,'ok')).length, me:rows.filter(r=>has(r,'me')).length, init:rows.filter(init).length,
    stray:rows.filter(r=>!has(r,'ok')&&!has(r,'me')&&!init(r)).length, note:(res.querySelector('.cmp-note')||{}).textContent||'',
    title:(c.querySelector('.cmp-title')||{}).textContent||'', more:res.querySelectorAll('details').length};}""")
def brief_ok(lang, r, rows, hidden):
    return (r is not None and r['rows'] == rows and r['ok'] == 4 and r['me'] == 1 and r['init'] == 1 and r['stray'] == 0 and r['more'] == 0
            and Z(lang, f'The other {hidden} options miss the goal', f'其余 {hidden} 种没达到目标') in r['note']
            and Z(lang, 'each setup', '每种方案') not in r['title'])
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ljg-rail [data-nav=\"next\"]')].map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#ljg-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")


def settle(pg, ms=2500):
    try: pg.wait_for_function("()=>!window.__ljg||!window.__ljg.state().stageBusy", timeout=4000)
    except Exception: pass
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(250); settle(pg)


def mode(pg, m):
    pg.evaluate(f"()=>document.querySelector('#ljg-app .modes button[data-mode=\"{m}\"]').click()"); pg.wait_for_timeout(350); settle(pg)


# R3.20: this lab has no animation; a part change, a click, a repeated click, a scenario or a submit redraws the chart at once.
# Any animation found in the chart two frames after the click (other than the kit's cue outline on the part's control) is a replay.
STILL = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  return document.querySelector('#ljg-graph').getAnimations({subtree:true}).filter(a=>a.animationName!=='cue').map(a=>a.animationName||a.transitionProperty||'?');}"""


def still(pg, sel, tag, what):
    an = pg.evaluate(STILL, sel); pg.wait_for_timeout(250); settle(pg)
    check(f'{tag} {what}: the chart is redrawn at once, nothing replays (R3.20)', not an, an)


# R7.12: the main picture first (the pair card, then the 300 marks), the metric cards below it and compact: the marks end within one
# screen of the stage's top, and at 1440 x 900 they are on the first screen as the page opens
FIRST = """()=>{const q=s=>document.querySelector(s).getBoundingClientRect(), st=q('#ljg-stage'), wf=q('#ljg-wf-box'), bars=q('#ljg-bars'), m=q('#ljg-metrics');
  return {stage:Math.round(st.top), marks:Math.round(wf.top), marksEnd:Math.round(wf.bottom), bars:Math.round(bars.bottom), metrics:Math.round(m.top), metricsH:Math.round(m.height), ih:innerHeight};}"""


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#ljg-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#ljg-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n')}""")


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],[/\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],
   [/深色|浅色|\\b(?:dark|light)(?:er)? (?:bar|line|block|part)/i,'dark / light named as a colour (R5.7)']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag, a[href*="github.com/Schuture"]')||!e.getClientRects().length) continue; const t=n.textContent;
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
 const ok={[blue]:/^$/,[amber]:/^$/,[red]:/status bad|badline|badc|ljg-over|warn|wrong|ex-res/,[violet]:/^$/};
 const out=[]; for(const e of document.querySelectorAll('#ljg-app *')){const s=getComputedStyle(e); if(s.display==='none'||!e.getClientRects().length) continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke']){const v=s[p]; for(const c of [blue,amber,red,violet]){ if(v===c){const cls=(e.getAttribute('class')||'')+' '+(e.parentElement?e.parentElement.getAttribute('class')||'':'')+' '+(e.parentElement&&e.parentElement.parentElement?e.parentElement.parentElement.getAttribute('class')||'':''); if(!ok[c].test(cls)&&!(p==='borderTopColor'&&s.borderTopWidth==='0px')&&!(p==='color'&&!e.textContent.trim())) out.push(e.tagName+'.'+(e.getAttribute('class')||'')+' '+p);}}}}
 return out.slice(0,12);}"""
OVER = """()=>{const out=[]; for(const e of document.querySelectorAll('#ljg-app *, #dock *')){if(e.closest('[data-endnotes],pre,code,.mathbody,svg')||!e.getClientRects().length) continue;
  const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden') continue;
  const clip=/hidden|clip|auto|scroll/.test(cs.overflowX)||cs.textOverflow==='ellipsis'; if(!clip||e.clientWidth===0) continue;
  if(e.scrollWidth>e.clientWidth+1) out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40)+' ('+e.scrollWidth+' > '+e.clientWidth+')');}
  return out.slice(0,8);}"""
HELP = """()=>{const out=[]; for(const e of document.querySelectorAll('#ljg-app *, #dock *')){if(!e.getClientRects().length) continue; if(getComputedStyle(e).cursor!=='help') continue;
  if(e.closest('[data-tipx],[data-tip],.t,.def')) continue; out.push((e.id||e.className||e.tagName)+': '+(e.textContent||'').trim().slice(0,40));}
  return out.slice(0,8);}"""
rule_hits = []


def rules(pg, where, lang):
    for sel in ('#ljg-app', '#dock'):
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


# B4 (polish round 5, reviews/r5-fit-agents-eval.md): entering parts 1 and 6 at 1440 × 900 the way the polish probe stage.py does
# (Next when it is enabled, else the part's heading), the stage needs no scrolling inside: Chinese with at least 30 px to spare,
# English with none (spare = the stage's max-height minus what its content and borders need). Its top is not under the top bar,
# and the main picture (the stage's largest svg or canvas, if it has one) is whole on the screen.
B4_FIT = """(id)=>{const s=document.getElementById(id), r=s.getBoundingClientRect(), tb=(document.querySelector('.topbar')||{offsetHeight:0}).offsetHeight;
  const cur=document.querySelector('#'+id.replace(/-stage$/,'-rail')+' .steps>.step.current');
  const f=[...s.querySelectorAll('svg,canvas')].filter(e=>e.getClientRects().length&&!e.closest('button,.btn')).map(e=>e.getBoundingClientRect()).sort((a,c)=>c.width*c.height-a.width*a.height)[0];
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)),
    scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), bar:tb, fig:f?[Math.round(f.top),Math.round(f.bottom)]:null, figOn:!f||(f.top>=tb-2&&f.bottom<=innerHeight+2)};}"""


def b4_fit(b, lang, parts=(1, 6)):
    pg = page(b, 1440, 900, lang)
    for k in range(1, max(parts) + 1):
        if k > 1:
            nb = pg.locator('.pbar [data-nav="next"]:visible:not([disabled])')
            if nb.count(): nb.first.click()
            else: pg.evaluate(f"()=>document.querySelector('#ljg-rail .steps>.step:nth-child({k}) .step-h').click()")
            pg.wait_for_timeout(900)
        if k not in parts: continue
        y = None
        for _ in range(20):   # the page's scroll has settled (the rail's reveal, then the lab's keepStage)
            y2 = pg.evaluate('()=>scrollY')
            if y2 == y: break
            y = y2; pg.wait_for_timeout(150)
        f = pg.evaluate(B4_FIT, 'ljg-stage')
        check(f'{lang}-1440 part {k}: entering it, the stage fits without scrolling inside{", 30 px to spare" if lang == "zh" else ""}; its top is below the top bar and its main picture whole on the screen (B4, R7.12)',
              f['part'] == k and f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['top'] >= f['bar'] - 1 and f['figOn'], f)
    pg.close()


with sync_playwright() as pw:
    b = pw.chromium.launch()
    if (1440, 900) in SIZES:
        for lang in LANGS: b4_fit(b, lang)
    for lang in LANGS:
        for (w, h) in SIZES:
            phone = w <= 760; tag = f'{lang}-{w}'
            pg = page(b, w, h, lang)
            st = state(pg)
            check(f'{tag}: opens on part 1 of the guided tour', st['mode'] == 'guided' and st['step'] == 0, st)
            fv = pg.evaluate(FIRST)
            check(f'{tag}: the main picture leads the stage; the metric cards sit below the bars and are compact (R7.12)',
                  fv['metrics'] >= fv['bars'] and fv['metricsH'] <= 140 and fv['marksEnd'] - fv['stage'] <= fv['ih'] and (phone or fv['marksEnd'] <= fv['ih']), fv)
            if phone:
                fs = pg.evaluate("()=>{const p=document.querySelector('#ljg-rail .pbar'), g=document.querySelector('#ljg-bars'); return {bar:!!p&&p.getBoundingClientRect().top<innerHeight, chart:!!g&&g.getBoundingClientRect().top<innerHeight*1.6}}")
                check(f'{tag}: the phone first screen has the parts bar, the chart follows it', fs['bar'] and fs['chart'], fs)
            # ---- part 1
            check(f'{tag} part 1: Next disabled until a prediction, with the kit’s why line', all(nexts(pg)) and any(Z(lang, 'Pick an answer', '选一个答案') in x for x in why(pg)), (nexts(pg), why(pg)))
            t1 = all_text(pg)
            check(f'{tag} part 1: people’s win rate appears nowhere (R3.8)', not st['people'] and not vis(pg, '.ljg-hl') and '49%' not in t1 and '49.3' not in t1)
            check(f'{tag} part 1: the judge’s 63% is on the card and the chart', t1.count('63%') >= 2)
            shot(pg, f'{tag}_p1', full=phone); rules(pg, f'{tag} part 1', lang)
            click(pg, '#ljg-rail [data-pred="60"]')
            check(f'{tag} part 1: after a pick, Next is enabled and the card says it is saved', not any(nexts(pg)) and Z(lang, 'Saved', '已记下') in text(pg, '#ljg-rail .step.current'))
            # ---- part 2
            click(pg, '#ljg-rail [data-nav="next"]')
            st = state(pg); p0 = st['pairAt']
            check(f'{tag} part 2: the formula, the counts and the 95% interval', vis(pg, '#ljg-rail .step.current .fx') and '95%' in text(pg, '#ljg-rail [data-obs]') and not any(nexts(pg)))
            click(pg, '#ljg-rail [data-act="pair-next"]')
            check(f'{tag} part 2: “Next pair” moves to the next pair and its outlined mark', state(pg)['pairAt'] == (p0 + 1) % 300 and pg.evaluate("()=>document.querySelectorAll('#ljg-wf-box .ljg-mon').length") == 1)
            shot(pg, f'{tag}_p2', full=phone); rules(pg, f'{tag} part 2', lang)
            # ---- part 3
            still(pg, '#ljg-rail [data-nav="next"]', tag, 'part 3, Next')
            check(f'{tag} part 3: Next waits for both orders, with its why line', all(nexts(pg)) and any(Z(lang, 'both orders', '两种顺序') in x for x in why(pg)), why(pg))
            still(pg, '#ljg-rail [data-lever="A"]', tag, 'part 3, A first')
            still(pg, '#ljg-rail [data-lever="A"]', tag, 'part 3, A first again (nothing changed)')
            ob = text(pg, '#ljg-rail [data-obs]')
            check(f'{tag} part 3: A first, B falls from 63% to 47%; Next still waits', '63%' in ob and '47%' in ob and all(nexts(pg)), ob)
            click(pg, '#ljg-rail [data-lever="R"]')
            ob = text(pg, '#ljg-rail [data-obs]')
            check(f'{tag} part 3: random order, B 55%, 150 pairs each way, cost unchanged; Next still waits', '55%' in ob and '150' in ob and Z(lang, 'on average', '平均下来') in ob and all(nexts(pg)) and Z(lang, 'Random order', '顺序随机') in text(pg, '#ljg-pair'), ob)
            click(pg, '#ljg-rail [data-lever="both"]')
            ob = text(pg, '#ljg-rail [data-obs]'); res = pg.evaluate("()=>[...document.querySelectorAll('#ljg-rail .lever .lv-r')].map(e=>e.textContent)")
            check(f'{tag} part 3: both orders, 26% inconsistent, B 55%, twice the cost; Next enabled', '26%' in ob and '55%' in ob and not any(nexts(pg)) and Z(lang, 'twice', '费用翻倍') in ob, ob)
            check(f'{tag} part 3: each tried option shows its result on its title row (R3.6)', all(res) and len(res) == 4, res)
            leg = text(pg, '#ljg-wf-leg')
            check(f'{tag} part 3: the inconsistent pairs are named beside the marks with their count (R5.9)', Z(lang, 'inconsistent', '不一致') in leg and '78' in leg, leg)
            lv = pg.evaluate("()=>[...document.querySelectorAll('#ljg-rail .lever')].map(b=>{const t=b.querySelector('.lv-t b'); return t.getBoundingClientRect().height<30})")
            check(f'{tag} part 3: every option title stays on one line (R3.17)', all(lv), lv)
            shot(pg, f'{tag}_p3', full=phone); rules(pg, f'{tag} part 3', lang)
            # ---- part 4
            click(pg, '#ljg-rail [data-nav="next"]')
            check(f'{tag} part 4: Next waits for length taken out; people still hidden', all(nexts(pg)) and not vis(pg, '.ljg-hl') and any(Z(lang, 'Take length out', '去掉长度') in x for x in why(pg)), why(pg))
            still(pg, '#ljg-rail .seg-ctl[data-key="lc"] button[data-v="1"]', tag, 'part 4, length taken out')
            ob = text(pg, '#ljg-rail [data-obs]')
            check(f'{tag} part 4: length out gives 50%, people 49%, the verdict names the right answer, and why', '50%' in ob and '49%' in ob and Z(lang, 'You predicted', '你预测的是') in ob and Z(lang, 'About 50%', '50% 左右') in ob and Z(lang, 'Why', '原因') in ob, ob)
            check(f'{tag} part 4: people’s line is on the chart and Next is enabled', vis(pg, '.ljg-hl') and not any(nexts(pg)))
            if phone:
                pg.evaluate("()=>{const t=document.getElementById('ljg-bars'); window.scrollTo(0, t.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(700)
                dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {shown:!d.hidden, status:document.getElementById('dkStatus').textContent, read:document.getElementById('dkRead').textContent}}")
                check(f'{tag} part 4: the mini result bar with a short status and both numbers', dk['shown'] and len(dk['status']) <= 12 and '50%' in dk['read'] and '49%' in dk['read'], dk)
            shot(pg, f'{tag}_p4', full=phone); rules(pg, f'{tag} part 4', lang)
            # ---- part 5
            still(pg, '#ljg-rail [data-nav="back"]', tag, 'part 4, Back to part 3')
            still(pg, '#ljg-rail [data-nav="next"]', tag, 'part 3, Next to part 4 again')
            click(pg, '#ljg-rail [data-nav="next"]')
            check(f'{tag} part 5: the agreement view replaces the pair; Next waits', not vis(pg, '#ljg-pair') and all(nexts(pg)) and '63%' in text(pg, '#ljg-bars') and '64%' in text(pg, '#ljg-bars'))
            blk = pg.evaluate("()=>[...document.querySelectorAll('#ljg-wf-box .ljg-agb svg')].map(s=>({w:s.getBoundingClientRect().width, n:s.children.length}))")
            check(f'{tag} part 5: two 10 × 10 blocks of 100 marks, marks at least 8 px', len(blk) == 2 and all(b['n'] == 100 and b['w'] >= 80 for b in blk), blk)
            click(pg, '#ljg-rail [data-lever="big"]')
            check(f'{tag} part 5: the stage subtitle names the large judge (B1)', text(pg, '#ljg-st-sub') == Z(lang, 'Large judge, both orders', '大模型，两种顺序都判'), text(pg, '#ljg-st-sub'))
            click(pg, '#ljg-rail [data-lever="both"]')
            ob = text(pg, '#ljg-rail [data-obs]')
            check(f'{tag} part 5: the subtitle names the small judge in both orders', text(pg, '#ljg-st-sub') == Z(lang, 'Small judge, both orders', '小模型，两种顺序都判'), text(pg, '#ljg-st-sub'))
            check(f'{tag} part 5: 72% against 69% on the same 67 pairs, the MT-Bench reference, and three practices', '72%' in ob and '69%' in ob and '67' in ob and '80%' in ob and pg.evaluate("()=>document.querySelectorAll('#ljg-rail .ljg-prac li').length") == 3 and not any(nexts(pg)), ob)
            shot(pg, f'{tag}_p5', full=phone); rules(pg, f'{tag} part 5', lang)
            # ---- part 6
            click(pg, '#ljg-rail [data-nav="next"]')
            cur = text(pg, '#ljg-rail .step.current')
            check(f'{tag} part 6: three misconceptions, See the scenarios, the next lab', cur.count('✗') == 3 and vis(pg, '#ljg-rail [data-mode-go="sandbox"]') and pg.evaluate("()=>!!document.querySelector('#ljg-rail .next-lab')"), cur[:200])
            click(pg, '#ljg-rail [data-rem="check"]')
            check(f'{tag} part 6: “Show in the chart” for part 5 shows the agreement view', not vis(pg, '#ljg-pair') and pg.evaluate("()=>document.querySelector('#ljg-rail [data-rem=\"check\"]').getAttribute('aria-pressed')") == 'true')
            click(pg, '#ljg-rail [data-rem="length"]')
            check(f'{tag} part 6: “Show in the chart” for part 4 shows people’s line', vis(pg, '.ljg-hl') and vis(pg, '#ljg-pair'))
            shot(pg, f'{tag}_p6', full=phone); rules(pg, f'{tag} part 6', lang)
            # ---- Scenarios
            mode(pg, 'sandbox')
            # the owner, 2026-10-08 (G1, G2) scen: the Scenarios card leads in with what to pick and what to watch; no “one at a time” note is left
            _rt = pg.evaluate("s=>document.querySelector(s).innerText", '#ljg-rail')
            _lead = Z(lang, 'Pick a scenario and see how far the judge’s win rate is from people’s.', '选一个场景，看看裁判给的胜率和人评差多少。')
            check(f'{tag} G1: the Scenarios card leads in “{_lead[:30]}…”', _lead in _rt, _rt[:300])
            check(f'{tag} G1: no “one at a time / 一次只能选 / 不会叠加” note in the Scenarios card', not any(x in _rt for x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time')), _rt[:300])
            # end G1 G2 scen
            pres = pg.evaluate("()=>document.querySelectorAll('#ljg-rail [data-preset]').length")
            check(f'{tag} scenarios: four', pres == 4, pres)
            for k, sc in enumerate(['chatty', 'better', 'self', 'math']):
                still(pg, f'#ljg-rail [data-preset="{k}"]', tag, f'scenario {sc}')
                pr = pg.evaluate("()=>[...document.querySelectorAll('#ljg-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
                check(f'{tag} scenario {sc}: single choice, its insight shown', pr.count('true') == 1 and pr[k] == 'true' and vis(pg, '#ljg-insight') and state(pg)['sc'] == sc, pr)
                shot(pg, f'{tag}_sb_{sc}', full=phone); rules(pg, f'{tag} scenario {sc}', lang)
            ins = text(pg, '#ljg-insight')
            check(f'{tag} scenario math: the checker gives 50%', '50%' in ins, ins)
            if not phone:
                pg.evaluate("()=>document.querySelector('#ljg-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(300)
                y0 = pg.evaluate("()=>document.querySelector('#ljg-ctl-judge').getBoundingClientRect().top")
                pg.evaluate("()=>document.querySelector('#ljg-ctl-judge button[data-v=\"big\"]').click()"); pg.wait_for_timeout(300)
                y1 = pg.evaluate("()=>document.querySelector('#ljg-ctl-judge').getBoundingClientRect().top")
                check(f'{tag} scenarios: a change under Settings does not move the page (R3.2)', abs(y1 - y0) < 1.5 and state(pg)['judge'] == 'big', (y0, y1))
            hs = pg.evaluate('()=>location.hash')
            check(f'{tag} scenarios: the URL carries the setup', hs.startswith('#llm-judge~math_'), hs)
            pg2 = page(b, w, h, lang, hash_=hs[1:])
            s2 = state(pg2)
            check(f'{tag} scenarios: a fresh page restores the setup', s2['mode'] == 'sandbox' and s2['sc'] == 'math' and s2['judge'] == state(pg)['judge'], s2)
            pg2.close()
            # ---- the challenge
            mode(pg, 'challenge')
            st = state(pg)
            check(f'{tag} challenge: starts from the small judge, once, as judged, on new data', (st['sc'], st['judge'], st['order'], st['lc']) == ('ch', 'small', 'B', False), st)
            check(f'{tag} challenge: Settings locked; three choices in the card', pg.evaluate("()=>document.querySelector('#ljg-controls').classList.contains('locked')") and pg.evaluate("()=>document.querySelectorAll('#ljg-rail .seg-ctl[data-key^=\"ch\"]').length") == 3)
            check(f'{tag} challenge: before Submit only the cost shows: no gap, no people’s line, no win rate (R3.8)', not vis(pg, '#ljg-ch-g1') and vis(pg, '#ljg-ch-g2') and '300' in text(pg, '#ljg-ch-g2') and not vis(pg, '.ljg-hl') and not vis(pg, '#ljg-bars .ljg-brow') and '%' not in text(pg, '#ljg-status') and '%' not in text(pg, '#ljg-metrics'), (text(pg, '#ljg-status'), text(pg, '#ljg-metrics')))
            still(pg, '#ljg-rail [data-act="submit"]', tag, 'challenge, Submit')
            gp = text(pg, '#ljg-ch-gap')
            check(f'{tag} challenge: after Submit, the gap (14.2 points at the start) and people’s line', vis(pg, '#ljg-ch-g1') and '14.2' in gp and vis(pg, '.ljg-hl'), gp)
            check(f'{tag} challenge: the stage shows only the learner’s own setup', pg.evaluate("()=>document.querySelectorAll('#ljg-bars .ljg-brow').length") == 1)
            r = text(pg, '#ljg-ch-res')
            check(f'{tag} challenge: not met, with a hint and “Show the answer” (R4.4)', vis(pg, '#ljg-ch-res .badline') and Z(lang, 'Hint', '提示') in r and vis(pg, '#ljg-ch-res [data-act="reveal"]')
                  and text(pg, '#ljg-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), r[:200])
            click(pg, '#ljg-ch-res [data-act="reveal"]')
            rows = pg.evaluate("()=>[...document.querySelectorAll('#ljg-ch-res .cmp-row')].map(e=>({ok:e.classList.contains('ok'),me:e.classList.contains('me'),tags:e.querySelectorAll('.cmp-tag,.cmp-you').length}))")
            bf = brief(pg)
            check(f'{tag} challenge: Show the answer draws 5 rows, the 4 passing setups and the initial setup (the pick), tagged (R5.4), 7 left out', len(rows) == 5 and sum(x['tags'] for x in rows) >= 2 and brief_ok(lang, bf, 5, 7), (rows, bf))
            r = text(pg, '#ljg-ch-res')
            check(f'{tag} challenge: the random order alone is named in words, its gap (8.2) in a hover', Z(lang, 'A random order alone still leaves the gap above 3 points', '只改成顺序随机，差距还是超过 3 个百分点') in r
                  and pg.evaluate("()=>[...document.querySelectorAll('#ljg-ch-res .def')].some(e=>/8\\.2/.test(e.getAttribute('data-tipx')))"), r[-600:])
            click(pg, '#ljg-rail .seg-ctl[data-key="chJ"] button[data-v="big"]'); click(pg, '#ljg-rail .seg-ctl[data-key="chO"] button[data-v="both"]')
            bf = brief(pg)
            check(f'{tag} challenge: a pick that misses (large judge, both orders, length left in) adds its own row, 6 rows, 6 left out', brief_ok(lang, bf, 6, 6), bf)
            click(pg, '#ljg-rail .seg-ctl[data-key="chL"] button[data-v="1"]')
            r = text(pg, '#ljg-ch-res')
            check(f'{tag} challenge: met, but a cheaper setup exists (300)', vis(pg, '#ljg-ch-res .okline') and '300' in r and Z(lang, 'cheaper', '更便宜') in r, r[:240])
            check(f'{tag} challenge: the first line says that a cheaper setup exists', Z(lang, 'But a cheaper setup also meets it', '不过有更便宜的方案也能达标') in text(pg, '#ljg-ch-res > p:first-child'), text(pg, '#ljg-ch-res > p:first-child'))
            bf = brief(pg)
            check(f'{tag} challenge: the chart holds the 4 passing setups (the pick among them) and the initial setup, 7 left out', brief_ok(lang, bf, 5, 7), bf)
            click(pg, '#ljg-rail .seg-ctl[data-key="chJ"] button[data-v="small"]')
            r = text(pg, '#ljg-ch-res')
            check(f'{tag} challenge: small judge in both orders: met, but random order is cheaper', vis(pg, '#ljg-ch-res .okline') and '300' in r and '600' in r and Z(lang, 'random order', '随机安排顺序') in r, r[:300])
            click(pg, '#ljg-rail .seg-ctl[data-key="chO"] button[data-v="R"]')
            r = text(pg, '#ljg-ch-res')
            check(f'{tag} challenge: met and the cheapest of the 12 setups, with its price', vis(pg, '#ljg-ch-res .okline') and Z(lang, '4 meet the goal, and this one is the cheapest', '这 12 种选法里有 4 种能达标，它最便宜') in r and Z(lang, 'twice the cost', '费用翻倍') in r and vis(pg, '#ljg-status.ok'), r[:300])
            check(f'{tag} challenge: the capability card follows', vis(pg, '#ljg-rail [data-exits]'))
            shot(pg, f'{tag}_ch', full=True); rules(pg, f'{tag} challenge', lang)
            other = 'en' if lang == 'zh' else 'zh'
            pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
            st = state(pg)
            check(f'{tag} challenge: the language switch keeps the answer', st['mode'] == 'challenge' and (st['judge'], st['order'], st['lc']) == ('small', 'R', True) and st['chSubmitted'] and vis(pg, '#ljg-ch-res .okline'), st)
            pg.close()
    b.close()
# the owner, 2026-10-09 (H1, H2): every formula typeset as MathML (A.ui.math), never a plain-text string; Show the math in short blocks,
# a hairline (1px, the kit's line colour) between them, each chain's “=” in one column, nothing clipped (R6.33)
_H1_LAB, _H1_PFX = 'llm-judge', 'ljg'
_H1_PLAIN = ['√', '/2) /', 'ln(B', '/ √']          # plain-text formula pieces that must not be left in the rendered text (outside <math>, <code>, <pre>, <svg>)
_H1_SRC_GONE = ['L<sub>A</sub>', 'ε<sub>1</sub>', 'r<sub>A</sub>', '× ${f3(r.sd)} / √', '(${d.bb} + ${d.cc}/2) / ${d.n} =', 'Win rate = (wins + half the ties) ÷ 100', "${TD('sigma','σ')}(g)"]     # plain-text formulas converted on 2026-10-09: gone from the lab's source (comments aside)
_H1_SRC_NEW = ['ljg-mb', 'ljg-eqs']       # sentences or markup written on 2026-10-09
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
_H5_SEP = ['#ljg-wf', '#ljg-bars']
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
for x in rule_hits[:40]: print('RULE', x)
check('writing, colour and layout rules', not rule_hits, len(rule_hits))
check('no hover outside Show the math only expands a number to its exact value (“精确值是” / “Exactly”; the owner, 2026-10-07 23:37)', not [h for h in rule_hits if 'exact value' in h], [h for h in rule_hits if 'exact value' in h][:3])
# the owner, 2026-10-08 (G1, G2) src: every option group leads in with what to pick and what to watch (G1); no sentence leans on an earlier part's detail (G2)
_src = (P / 'labs' / 'llm-judge.js').read_text() + (P / 'teasers' / 'llm-judge.js').read_text()
for _x in ('一次只能选', '不会叠加', '一次选一个', 'do not stack', 'one at a time', 'One at a time'):
    check(f'G1: no “{_x}” left in the lab or its tile', _x not in _src)
for _x in ['回答一个字没改，只换先后顺序，胜率就会跟着变。', '一个版本的回答更长，裁判给它的胜率就会比人评高。去掉长度以后，胜率就接近人评了。', '请人标一小批，看裁判和人一致多少，再看两个人之间一致多少。裁判和人的一致率跟多请一个人差不多，就算合格。', '一致率只数裁判选对了几对。一个裁判可以常和人一致，却还是把胜率往一边推高好几个百分点。', '一换顺序结论就变的回答对，正是位置偏好说了算的那些对。把它们算作平局，它们就不再把胜率推向排在前面的回答。', '所以一个裁判和人的一致率可以比第二个人还高，给出的胜率却仍然比人评高好几个百分点。', '在图里显示两种顺序都判的结论。', '在图里显示去掉长度后的胜率，和人评的胜率。', '在图里显示裁判和人的一致率。', '再用和这个 Lab 一样的胜率公式报告', 'Swap which answer comes first, and the win rate moves, though not a word of the answers has changed.']:
    check(f'G1/G2: the new sentence is there: {_x[:60]}', _x in _src)
for _x in ['在第 3 部分，把 B 在前换成 A 在前', '在第 4 部分，去掉长度后', '在第 5 部分，两种顺序都判的裁判', '第 5 部分里判一次的裁判', '所以要先用它检验裁判（第 5 部分）', '在第 3 部分，${fR(t.inc/N)}', '和第 3 部分一样', '和任务里一样', '任务里的做法', '在图里显示第 3 部分', '在图里显示第 4 部分', '在图里显示第 5 部分', '再用第 2 部分的公式', 'In part 3, showing A first', 'In part 4, taking length out', 'In part 5, the judge in both orders', 'In part 5 the judge judged once', '(part 5).', 'In part 3, ${fR(t.inc/N)}', 'as in part 3', 'as in the mission', 'what the mission did', 'Shows part 3’s', 'Shows part 4’s', 'Shows part 5’s', 'the formula of part 2']:
    check(f'G2: the old sentence is gone: {_x[:60]}', _x not in _src)
# end G1 G2 src
print(f'{n - len(fails)}/{n} checks passed')
sys.exit(1 if fails else 0)
