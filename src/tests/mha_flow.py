"""Flow test for the MHA lab (labs/mha.js): every mode and every guided part at 1440 x 900 and 390 x 844 in EN and ZH,
dark mode once, and the layout widths 1280 / 900 / 720. Checks: no page errors, no horizontal overflow, the next-part gating,
the prediction's answer hidden until part 4 shows it (R3.8), the challenge feedback tiers and the chart, the language switch
keeping state, the hash round-trip, the mini result bar at 390, the result on screen after each action at 1440 x 900 (R3.10),
the first word click in part 3 keeping the picker, H1's row and the result in view (N1), part 4 after "2 heads" first and after a
skipped prediction (N8, N9), the challenge's mini result bar on a phone (N6), the chart link (N7), the fidelity note (N10),
the colours (R5.6), and the writing rules on every visible string and tooltip: no "·" between clauses, no two colons in one
Chinese sentence, no "in this toy" / "在这个玩具里", no 起点 / starting point, no 控件 / Controls, no sentence about the
mini result bar or hovering. Screenshots go to shots/mha/.

Run: python3 tests/mha_flow.py
"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/mha.html', '--only', 'mha'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'wrapped-mha.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'mha.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-mha.html')
SH = P / 'shots' / 'mha'; SH.mkdir(parents=True, exist_ok=True)
errs, notes, fails = [], [], []
IDS = ['mission', 'predict', 'look', 'one', 'two', 'four', 'real', 'remember']


MATHBLOCKS = """(a)=>{const [sel,pre]=a; const body=document.querySelector(sel); if(!body) return null;
  const bs=[...body.querySelectorAll('.'+pre+'-mb')];
  const words=[...body.querySelectorAll('math mi, math mtext')].map(e=>e.textContent.trim()).filter(w=>/^(left|right|frac|sqrt|mathrm|hat|cdot|times|div|approx|top|big|quad|ldots|cdots|alpha|beta|theta|phi|pi|sigma|varphi)$/.test(w));
  const ruled=bs.filter(b=>{const cs=getComputedStyle(b); return cs.borderTopStyle==='solid'&&parseFloat(cs.borderTopWidth)>=1}).length;
  const clip=[...body.querySelectorAll('[class$="math-line"]')].filter(x=>{const m=x.querySelector('math'); if(!m) return true; const r=m.getBoundingClientRect(), b=x.getBoundingClientRect(); return r.height>0&&(r.top<b.top-1||r.bottom>b.bottom+1)}).length;
  return {n:bs.length, lead:bs.filter(b=>b.querySelector('.'+pre+'-mlead')).length, fx:bs.filter(b=>b.querySelector('math')).length, ruled, words, clip, math:body.querySelectorAll('math').length}}"""

def check(name, ok, info=''):
    notes.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{str(info)[:300]}]' if info else ''))
    if not ok: fails.append(f'{name} [{str(info)[:300]}]')


def page(b, w, h, lang, hash_='mha', dark=False, reduced=False):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference'))
    pg.set_default_timeout(8000)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(400)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
Z = lambda lang, en, zh: zh if lang == 'zh' else en
state = lambda pg: pg.evaluate("()=>{const t=location.hash; return t}")


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#mha-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(250)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(120)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def settle(pg, ms=2500):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def rect(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); if(!e||!e.offsetParent) return null; const r=e.getBoundingClientRect(); return {top:Math.round(r.top),bottom:Math.round(r.bottom),h:Math.round(r.height)}}", sel)


# the writing rules, on every visible text node of the lab (R6.14, R6.17, R2.1, R3.4, R6.11)
RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · /,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/in this toy|在这个玩具里|这个玩具/i,'“in this toy”'],[/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/updates as you change|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/Locked in|已锁定/,'locked in']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||!e.offsetParent&&getComputedStyle(e).position!=='fixed') continue;
   if(e.closest('svg, pre, code, #tip')) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push('tooltip '+name+': '+t.slice(0,90));});
 return out;}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span.lbl,.status,.dk-read,.cmp-lab,h1,h2,h3,.tz-read,.tz-note,.legend-row > span,.mha-rlab,.metric .sub').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,table')) return;
   if(e.matches('.presets,.choices,.legend-row,.seg-ctl,.modes,.levers,.mha-words,.mha-onbtns')||e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){for(const x of t.replace(/\\s+/g,' ').split(/(?<=[。？！])/)){if(!/[\\u4e00-\\u9fff]/.test(x)) continue; const n=(x.match(/：/g)||[]).length; if(n>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
# a typeset formula in a hover (H1, 2026-10-09) counts as one Latin token: its MathML tags around a Chinese \text{} are not text
SPACING = """(sel)=>{const root=document.querySelector(sel); if(!root) return []; const out=new Set();
 const tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT); let n;
 while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre,code,svg,table,.mha-tok,.mha-words,.mha-rlab')) continue; const t=n.textContent;
   const m=t.match(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/); if(m) out.add(t.trim().slice(0,80));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx').replace(/<math[\\s\\S]*?<\\/math>/g,'M'); if(/[\\u4e00-\\u9fff][A-Za-z0-9<]|[A-Za-z0-9>%][\\u4e00-\\u9fff]/.test(t)) out.add('tooltip: '+t.slice(0,80));});
 return [...out];}"""
rule_hits = []


# the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value (Show the math excepted)
EXACT_HOVERS = """()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('#mha-mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||''))
  .map(e=>e.textContent.trim().slice(0,30)+' → '+e.getAttribute('data-tipx').slice(0,90))"""


def rules(pg, where, lang):
    ex = pg.evaluate(EXACT_HOVERS)
    check(f'{where}: no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
    for sel in ('#mha-app', '#dock'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        if lang == 'zh':
            for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons in one sentence: {h}')
            for h in pg.evaluate(SPACING, sel): rule_hits.append(f'{where}: no space between Chinese and Latin: {h}')


# R5.6: blue only for the weight matrices; no amber or violet anywhere; red and green only in the status, the goal lines and the chart
COLORS = """()=>{const blue=['rgb(40, 85, 197)','rgb(90, 138, 255)'], other=['rgb(211, 135, 0)','rgb(113, 83, 230)','rgb(239, 174, 54)','rgb(155, 133, 255)'];
 const out=[]; for(const e of document.querySelectorAll('#mha-app *')){const cs=getComputedStyle(e); if(cs.display==='none') continue;
  for(const p of ['color','backgroundColor','borderTopColor','fill','stroke','backgroundImage']){const v=cs[p]; if(!v) continue;
    if(other.some(c=>v.includes(c))) out.push('amber/violet '+(e.getAttribute('class')||e.tagName)+' '+p);
    if(blue.some(c=>v.includes(c))&&!e.closest('.mha-mats,.mha-mleg,.sw.mha-sw-w,a,.def')) out.push('blue outside the matrices '+(e.getAttribute('class')||e.tagName)+' '+p);}}
 return out;}"""


def run_flow(b, w, h, lang, dark=False):
    tag = f'{"dark_" if dark else ""}{w}_{lang}'; pg = page(b, w, h, lang, dark=dark)
    title = pg.inner_text('#mha-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'Why does attention use many heads?', 'attention 为什么要分成多个 head？'), title)
    check(f'{tag} starts guided, part 1 of 8', Z(lang, 'part 1 of 8', '第 1 / 8 部分') in pg.inner_text('#mha-rail .rail-top').lower())
    check(f'{tag} no horizontal overflow at start', overflow(pg) <= 0, overflow(pg))
    shot(pg, f'{tag}_p1')
    # parts 1-2: the weights, the two weight metrics and the head count stay hidden until part 3 (R3.8)
    for k in (0, 1):
        go(pg, k)
        vals = pg.evaluate("()=>[...document.querySelectorAll('#mha-grid .mha-pv')].map(e=>e.textContent).filter(Boolean)")
        mets = pg.evaluate("()=>[...document.querySelectorAll('#mha-metrics .metric .v')].map(e=>e.textContent)")
        check(f'{tag} part {k + 1}: no weight shown before the prediction', not vals and mets[1] == '?' and mets[2] == '?', (vals, mets))
        check(f'{tag} part {k + 1}: the number of heads is hidden in Settings', not pg.is_visible('#mha-ctl-H'))
        body = pg.evaluate("()=>document.getElementById('mha-app').innerText+' '+[...document.querySelectorAll('#mha-app [data-tipx]')].map(e=>e.dataset.tipx).join(' ')")
        one = pg.evaluate(f"()=>{{const r=__mha.compute({{lang:'{lang}',q:8,H:1}}); return [Math.round(r.heads[0].w[7]*100)+'%',Math.round(r.heads[0].w[r.cat_i]*100)+'%']}}")
        check(f'{tag} part {k + 1}: the one-head answer ({one}) appears nowhere', not any(x in body for x in one), one)
    # prediction gating
    go(pg, 1)
    check(f'{tag} part 2: Next disabled before a prediction', pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
    check(f'{tag} part 2: two options with radio dots, none pressed', pg.evaluate("()=>{const c=[...document.querySelectorAll('#mha-rail .choice')]; return c.length===2&&c.every(b=>b.getAttribute('aria-pressed')==='false')}"))
    pg.click('#mha-rail [data-pred="both"]'); pg.wait_for_timeout(150)
    check(f'{tag} part 2: Next enabled after a prediction', not pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
    check(f'{tag} part 2: “saved, you can still change it”', Z(lang, 'You can still change it', '还可以改') in pg.inner_text('#mha-rail .step.current'))
    check(f'{tag} part 2: the badge says the weights come from part 3 on', pg.inner_text('#mha-status') == Z(lang, 'Shown from part 3 on', '从第 3 部分起显示'), pg.inner_text('#mha-status'))
    shot(pg, f'{tag}_p2')
    # part 3: one head at a word other than “it” (R3.13: one head first); the picker is in the card; Next waits for a pick
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 3: Next disabled until a word is picked', pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
    rows = pg.evaluate("()=>[...document.querySelectorAll('#mha-grid .mha-row')].map(r=>r.classList.contains('dim'))")
    check(f'{tag} part 3: one head, drawn in full', rows == [False], rows)
    check(f'{tag} part 3: the word picker is inside the card', pg.query_selector('#mha-rail .step.current #mha-ic-q') is not None)
    check(f'{tag} part 3: “it” is kept for part 4 (dimmed, aria-disabled)', pg.evaluate("()=>{const b=document.querySelector('#mha-ic-q button[data-q=\"8\"]'); return b.classList.contains('mha-later')&&b.getAttribute('aria-disabled')==='true'}"))
    # on a phone, Next scrolls the parts' bar to the top (smooth); a scroll hides tips, so wait for it to end first
    settle(pg); pg.evaluate("()=>document.querySelector('#mha-ic-q').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
    pg.click('#mha-ic-q button[data-q="8"]', force=True); pg.wait_for_timeout(250)
    check(f'{tag} part 3: a click on “it” keeps the current word and says why', pg.evaluate("()=>document.querySelector('#mha-grid .mha-tok.q').dataset.q") == '3' and pg.is_visible('#tip') and Z(lang, 'part 4', '第 4 部分') in pg.inner_text('#tip'), pg.inner_text('#tip') if pg.is_visible('#tip') else '')
    pg.mouse.move(1, 1)
    pg.click('#mha-ic-q button[data-q="3"]'); pg.wait_for_timeout(350)
    obs = pg.inner_text('#mha-rail [data-obs]')
    w3 = pg.evaluate(f"()=>{{const r=__mha.compute({{lang:'{lang}',q:3,H:1}}); const o=r.heads[0]; let j=0; o.w.forEach((v,i)=>{{if(v>o.w[j]) j=i;}}); return Math.round(o.w[j]*100)}}")
    check(f'{tag} part 3: the result names the one head’s largest weight ({w3}%)', f'{w3}%' in obs and Z(lang, 'the one head', '唯一的 head') in obs, obs[:200])
    check(f'{tag} part 3: Next enabled after picking a word', not pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
    masked = pg.evaluate("()=>[...document.querySelectorAll('#mha-grid .mha-row:first-of-type .mha-cell')].slice(4).every(c=>c.classList.contains('masked'))")
    check(f'{tag} part 3: the 5 words after word 3 are hatched (causal mask)', masked)
    shot(pg, f'{tag}_p3')
    rules(pg, f'{tag} part 3', lang)
    # part 4: the one head at “it”; the verdict quotes the prediction and the right answer
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 4: no result before “it” is picked', not pg.is_visible('#mha-rail [data-obs]'))
    pg.click('#mha-ic-q button[data-q="8"]'); pg.wait_for_timeout(350)
    obs = pg.inner_text('#mha-rail [data-obs]')
    r1 = pg.evaluate(f"()=>{{const r=__mha.compute({{lang:'{lang}',q:8,H:1}}); return [Math.round(r.heads[0].w[7]*100),Math.round(r.heads[0].w[r.cat_i]*100)]}}")
    check(f'{tag} part 4: the result states the blend {r1}', f'{r1[0]}%' in obs and f'{r1[1]}%' in obs, obs[:200])
    check(f'{tag} part 4: the verdict quotes both options and says why', Z(lang, 'You predicted “Both can get 80% or more”; the right answer is “At most one can get 80% or more”: one head’s attention weights add up to 100%', '你预测的是“两个都能在 80% 以上”，正确答案是“最多一个能在 80% 以上”：一个 head 的注意力权重加起来是 100%') in obs, obs[:300])
    check(f'{tag} part 4: the in-lab outcome says “in this lab”', Z(lang, 'in this lab', '在这个 Lab 里') in obs, obs[:400])
    order = pg.evaluate("()=>{const t=document.querySelector('#mha-rail [data-obs]').innerText; return [t.indexOf('%'),t.search(/predicted|预测的是/),t.search(/Why|原因|为什么/)]}")
    check(f'{tag} part 4: result, then verdict, then reason (R3.10)', order[0] < order[1] < order[2], order)
    check(f'{tag} part 4: badge red', 'bad' in pg.get_attribute('#mha-status', 'class'))
    badges = pg.evaluate("()=>[...document.querySelectorAll('#mha-metrics .metric .d')].map(d=>d.textContent)")
    check(f'{tag} part 4: ▲▼ in points after the pick', any(('pts' in x or '百分点' in x) for x in badges), badges)
    shot(pg, f'{tag}_p4')
    rules(pg, f'{tag} part 4', lang)
    # part 5: two heads, each word in its own head
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 5: enters with one head', pg.evaluate("()=>document.querySelectorAll('#mha-grid .mha-row').length") == 1)
    badges0 = pg.evaluate("()=>[...document.querySelectorAll('#mha-metrics .metric .d')].map(d=>d.textContent).join('')")
    check(f'{tag} part 5: no ▲▼ on entering a part', badges0 == '', badges0)
    opts = pg.evaluate("()=>[...document.querySelectorAll('#mha-ic-H button[data-v]')].map(b=>b.dataset.v)")
    check(f'{tag} part 5: the card’s control offers 1 and 2 heads only (N15, R3.13)', opts == ['1', '2'], opts)
    sm = pg.query_selector('#mha-rail [data-nav="showme"]')
    check(f'{tag} part 5: Show me offered', sm is not None and sm.is_visible())
    sm.click(); pg.wait_for_timeout(350)
    obs = pg.inner_text('#mha-rail [data-obs]')
    r2 = pg.evaluate(f"()=>{{const r=__mha.compute({{lang:'{lang}',q:8,H:2}}); return [Math.round(r.heads[0].w[7]*100),Math.round(r.heads[1].w[r.cat_i]*100)]}}")
    check(f'{tag} part 5: head A on “because” and head B on “cat” ({r2}), nothing added', 'head A' in obs and 'head B' in obs and f'{r2[0]}%' in obs and f'{r2[1]}%' in obs and '256' in obs, obs[:300])
    check(f'{tag} part 5: badge green', 'ok' in pg.get_attribute('#mha-status', 'class'))
    shot(pg, f'{tag}_p5')
    rules(pg, f'{tag} part 5', lang)
    # part 6: four heads, one pattern each
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 6: enters with two heads', pg.evaluate("()=>document.querySelectorAll('#mha-grid .mha-row').length") == 2)
    opts = pg.evaluate("()=>[...document.querySelectorAll('#mha-ic-H button[data-v]')].map(b=>b.dataset.v)")
    check(f'{tag} part 6: the card’s control offers 2 and 4 heads only (N15)', opts == ['2', '4'], opts)
    pg.click('#mha-ic-H button[data-v="4"]'); pg.wait_for_timeout(350)
    obs = pg.inner_text('#mha-rail [data-obs]')
    check(f'{tag} part 6: the result names H1 to H4 and 256', all(x in obs for x in ('H1', 'H2', 'H3', 'H4', '256')), obs[:300])
    check(f'{tag} part 6: badge green', 'ok' in pg.get_attribute('#mha-status', 'class'))
    shot(pg, f'{tag}_p6')
    rules(pg, f'{tag} part 6', lang)
    # part 7: the real model, without KV cache arithmetic (R6.16)
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    card = pg.inner_text('#mha-rail .step.current')
    check(f'{tag} part 7: 32 heads of 128, 8 KV heads in one sentence, no byte counts (N9)', Z(lang, 'cut into 32 heads of 128', '切成 32 个 head，每个 128 维') in card and Z(lang, 'config.json lists 8 KV heads', 'config.json 里写着 8 个 KV head') in card and '131,072' not in card and not pg.is_visible('#mha-rail [data-obs]'), card[-300:])
    shot(pg, f'{tag}_p7')
    rules(pg, f'{tag} part 7', lang)
    # part 8: what to remember; the tour goes on to Scenarios (R3.12)
    pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300)
    check(f'{tag} part 8: next-lab footer under the card', pg.query_selector('#mha-rail .next-lab') is not None)
    if w <= 760:
        check(f'{tag} part 8: on phones the next-lab footer shows after the bench', pg.is_visible('.next-lab-end') and not pg.is_visible('#mha-rail .next-lab'))
    else:
        check(f'{tag} part 8: on desktops the footer is under the card', pg.is_visible('#mha-rail .next-lab') and not pg.is_visible('.next-lab-end'))
    check(f'{tag} part 8: three misconceptions', pg.evaluate("()=>[...document.querySelectorAll('#mha-rail .step.current .obs p')].filter(p=>p.textContent.startsWith('✗')).length") == 3)
    check(f'{tag} part 8: ends with “See the scenarios”, no challenge button (R3.12)', pg.query_selector('#mha-rail .step.current [data-mode-go="sandbox"]') is not None and pg.query_selector('#mha-rail .step.current [data-mode-go="challenge"]') is None)
    shot(pg, f'{tag}_p8')
    rules(pg, f'{tag} part 8', lang)
    check(f'{tag} no overflow in guided mode', overflow(pg) <= 0, overflow(pg))
    pg.click('#mha-rail .step.current [data-mode-go="sandbox"]'); pg.wait_for_timeout(400)
    check(f'{tag} R3.12: the guided tour’s last button opens Scenarios', pg.evaluate("()=>document.querySelector('#mha-app .modes button[aria-pressed=true]').dataset.mode") == 'sandbox')
    # scenarios
    check(f'{tag} scenarios: the lead-in (G1, 2026-10-08)', Z(lang, 'Pick a scenario and see which word each head puts its attention on.', '选一个场景，看看每个 head 把注意力放在哪个词上。') in pg.inner_text('#mha-rail'))
    for i in range(5):
        pg.click(f'#mha-rail [data-preset="{i}"]'); pg.wait_for_timeout(250)
        pressed = pg.evaluate("()=>[...document.querySelectorAll('#mha-rail [data-preset]')].map(b=>b.getAttribute('aria-pressed'))")
        check(f'{tag} scenario {i + 1}: only it is pressed', pressed == ['true' if j == i else 'false' for j in range(5)], pressed)
        ins = rect(pg, '#mha-insight')
        check(f'{tag} scenario {i + 1}: the result box is on screen', ins is not None and ins['top'] < h, ins)
        if i in (1, 3):
            shot(pg, f'{tag}_scenario{i + 1}')
    pg.click('#mha-ctl-q button[data-q="5"]'); pg.wait_for_timeout(250)
    check(f'{tag} scenarios: a change in Settings clears the pressed scenario', pg.evaluate("()=>[...document.querySelectorAll('#mha-rail [data-preset]')].every(b=>b.getAttribute('aria-pressed')==='false')"))
    rules(pg, f'{tag} scenarios', lang)
    check(f'{tag} scenarios: a next-lab footer is shown', pg.is_visible('.next-lab-end') if w <= 760 else pg.is_visible('#mha-rail .next-lab'))
    # challenge, reached from the Scenarios card's last button (R3.12)
    check(f'{tag} R3.12: the Scenarios card ends with “Take the challenge”', pg.query_selector('#mha-rail .railcard [data-mode-go="challenge"]') is not None)
    pg.click('#mha-rail .railcard [data-mode-go="challenge"]'); pg.wait_for_timeout(350)
    check(f'{tag} R3.12: it opens the challenge', pg.evaluate("()=>document.querySelector('#mha-app .modes button[aria-pressed=true]').dataset.mode") == 'challenge')
    check(f'{tag} challenge: Settings locked with a note', pg.is_visible('#mha-lock') and not pg.is_visible('#mha-ctl-q') and not pg.is_visible('#mha-ctl-H-wrap'))
    check(f'{tag} challenge: the switches are in the card', pg.query_selector('#mha-rail #mha-ic-on [data-on]') is not None)
    shot(pg, f'{tag}_challenge')
    pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(300)
    res = pg.inner_text('#mha-ch-res')
    check(f'{tag} challenge tier 2 (met, more can go): names H2 and H4, and is not “Goal met” (M2)', res.startswith(Z(lang, 'Both still at 80% or more:', '两个词仍都在 80% 以上：')) and Z(lang, 'But you could turn off more', '不过还能再关') in res and 'H2' in res and 'H4' in res and Z(lang, 'Goal met', '达成目标') not in res, res[:200])
    pg.click('#mha-ic-on button[data-on="2"]'); pg.wait_for_timeout(300)
    res = pg.inner_text('#mha-ch-res')
    check(f'{tag} challenge tier 3 (missed): hint, no answer', Z(lang, 'Not yet', '还没达到') in res and Z(lang, 'Hint', '提示') in res, res[:200])
    check(f'{tag} challenge tier 3: the chart is revealed (after tier 2 met it)', pg.query_selector('#mha-ch-res .mha-cmp') is not None)
    check(f'{tag} challenge: badge red when H3 is off', 'bad' in pg.get_attribute('#mha-status', 'class'))
    for hh in ('1', '2', '3'):
        pg.click(f'#mha-ic-on button[data-on="{hh}"]'); pg.wait_for_timeout(200)
    res = pg.inner_text('#mha-ch-res')
    check(f'{tag} challenge tier 1 (met with the fewest heads)', Z(lang, 'Goal met: with only H1 and H3 on', '达成目标：只开着 H1 和 H3') in res, res[:200])
    rows = pg.evaluate("()=>[...document.querySelectorAll('#mha-ch-res .mha-cr')].map(r=>[r.classList.contains('ok'),r.classList.contains('me')])")
    check(f'{tag} challenge chart: 6 rows, 4 green, your pick marked', len(rows) == 6 and sum(1 for r in rows if r[0]) == 4 and sum(1 for r in rows if r[1]) == 1, rows)
    shot(pg, f'{tag}_challenge_met')
    rules(pg, f'{tag} challenge', lang)
    check(f'{tag} no overflow in the challenge', overflow(pg) <= 0, overflow(pg))
    check(f'{tag} challenge: a next-lab footer is shown', pg.is_visible('.next-lab-end') if w <= 760 else pg.is_visible('#mha-rail .next-lab'))
    if dark:
        cols = pg.evaluate(COLORS)
        check(f'{tag} colours: blue only in the matrices, no amber or violet', not cols, cols[:5])
    pg.close()


# the owner's review of 2026-10-09 (R6.37: a row the reader has to scroll sideways is not read): in every guided part, Scenarios and the
# challenge, with each Show the math open, no box in it scrolls sideways and it is no wider than its own box (as tests/kit_math_scroll.py)
MSCROLL = """()=>{const out=[]; for(const d of document.querySelectorAll('details.math')){if(!d.getClientRects().length) continue; const was=d.open; d.open=true;
    for(const e of d.querySelectorAll('*')){if(!/auto|scroll/.test(getComputedStyle(e).overflowX)||!e.getClientRects().length) continue;
      const by=e.scrollWidth-e.clientWidth; if(by>1) out.push(e.className+' by '+by+' px: '+(e.textContent||'').trim().replace(/\\s+/g,' ').slice(0,40));}
    if(d.scrollWidth-d.clientWidth>1) out.push('the whole Show the math by '+(d.scrollWidth-d.clientWidth)+' px'); d.open=was;}
  return out;}"""


def math_scroll(b, w, h, lang):
    pg = page(b, w, h, lang); bad = []
    n = pg.evaluate("r=>new Set([...document.querySelectorAll(r+' [data-go]')].map(e=>e.dataset.go)).size", '#mha-rail')
    for k in range(n):
        pg.evaluate("([r,k])=>{const e=document.querySelector(r+' [data-go=\"'+k+'\"]'); if(e) e.click();}", ['#mha-rail', k]); pg.wait_for_timeout(500)
        bad += [f'part {k + 1}: {x}' for x in pg.evaluate(MSCROLL)]
    for m in ('sandbox', 'challenge'):
        pg.evaluate("m=>{const b=[...document.querySelectorAll('.modes button[data-mode=\"'+m+'\"]')].filter(x=>x.offsetParent).pop(); if(b) b.click();}", m); pg.wait_for_timeout(600)
        bad += [f'{m}: {x}' for x in pg.evaluate(MSCROLL)]
    check(f'{w} {lang}: no Show the math box scrolls sideways (all {n} guided parts, Scenarios, the challenge)', n >= 5 and not bad, (n, bad[:4]))
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h, lang in ((1440, 900, 'zh'), (900, 900, 'en'), (390, 844, 'zh'), (390, 844, 'en')):
        math_scroll(b, w, h, lang)
    for w, h in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            run_flow(b, w, h, lang)
    run_flow(b, 1440, 900, 'en', dark=True)
    run_flow(b, 390, 844, 'zh', dark=True)
    # colours in light mode too
    pg = page(b, 1440, 900, 'en', 'mha~8_4_1111')
    cols = pg.evaluate(COLORS)
    check('light colours: blue only in the matrices, no amber or violet', not cols, cols[:5])
    # hash round-trip: a setup link opens Scenarios with that setup
    pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(200)
    pg.evaluate("()=>{location.hash='mha~3_2_1111'}"); pg.wait_for_timeout(400)
    st = pg.evaluate("()=>({mode:document.querySelector('#mha-app .modes button[aria-pressed=true]').dataset.mode, rows:document.querySelectorAll('#mha-grid .mha-row').length, q:document.querySelector('#mha-grid .mha-tok.q').textContent, hash:location.hash})")
    check('hash round-trip: #mha~3_2_1111 opens Scenarios, word 3, 2 heads', st['mode'] == 'sandbox' and st['rows'] == 2 and st['q'] == 'sat' and st['hash'] == '#mha~3_2_1111', st)
    pg.click('#mha-ctl-H button[data-v="4"]'); pg.wait_for_timeout(200); pg.click('#mha-ctl-on button[data-on="2"]'); pg.wait_for_timeout(200)
    check('hash follows the state', pg.evaluate("()=>location.hash") == '#mha~3_4_1101', pg.evaluate("()=>location.hash"))
    check('no share button (the owner removed “复制配置链接” on 2026-10-11 19:28 UTC); the address still carries the setup', pg.query_selector('#mha-app [data-share]') is None)
    pg.evaluate("()=>{location.hash='mha~bad_token'}"); pg.wait_for_timeout(300)
    check('a bad token does not break the page', pg.query_selector('#mha-grid .mha-row') is not None)
    pg.close()
    # language switch keeps state: mid-guided and mid-challenge
    pg = page(b, 1440, 900, 'en')
    go(pg, 1); pg.click('#mha-rail [data-pred="one"]'); pg.wait_for_timeout(100); go(pg, 3); pg.click('#mha-ic-q button[data-q="8"]'); pg.wait_for_timeout(200)
    pg.click('.langsw button[data-lang="zh"]'); pg.wait_for_timeout(400)
    s2 = pg.evaluate("()=>({part:document.querySelector('#mha-rail .rail-top').innerText, rows:document.querySelectorAll('#mha-grid .mha-row').length, obs:document.querySelector('#mha-rail [data-obs]').innerText})")
    check('language switch keeps the guided part, the head count and the verdict', '第 4 / 8 部分' in s2['part'] and s2['rows'] == 1 and '你的预测“最多一个能在 80% 以上”是对的' in s2['obs'], s2)
    pg.click('#mha-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
    pg.click('#mha-ic-on button[data-on="3"]'); pg.wait_for_timeout(150); pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(200)
    pg.click('.langsw button[data-lang="en"]'); pg.wait_for_timeout(400)
    s3 = pg.evaluate("()=>({on:[...document.querySelectorAll('#mha-ic-on [data-on]')].map(b=>b.getAttribute('aria-pressed')), res:document.getElementById('mha-ch-res').innerText})")
    check('language switch keeps the challenge switches and the submitted result', s3['on'] == ['true', 'true', 'true', 'false'] and 'Both still at 80% or more' in s3['res'], s3)
    pg.close()
    # R3.10: after each action at 1440 x 900 the result is on screen; report each card's height (the wrong prediction, whose verdict is longest)
    heights = []
    VH = 900
    for lang in ('en', 'zh'):
        pg = page(b, 1440, VH, lang)
        go(pg, 1); pg.click('#mha-rail [data-pred="both"]'); pg.wait_for_timeout(150)
        for k, act in ((2, "#mha-ic-q button[data-q='5']"), (3, "#mha-ic-q button[data-q='8']"), (4, "#mha-ic-H button[data-v='2']"), (5, "#mha-ic-H button[data-v='4']")):
            go(pg, k); pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
            pg.evaluate("()=>{const s=document.querySelector('#mha-rail .step.current'); s.scrollIntoView({block:'start'}); window.scrollBy(0,-80)}"); pg.wait_for_timeout(150)
            pg.click(act); pg.wait_for_timeout(200); settle(pg)
            box = rect(pg, '#mha-rail [data-obs]'); card = rect(pg, '#mha-rail .step.current')
            heights.append((lang, IDS[k], card['h'] if card else None))
            check(f'R3.10 {lang} part {k + 1}: the result’s first line is on screen after the action', box is not None and 56 <= box['top'] <= VH - 60, box)
            # the verdict (part 4) must be on screen too
            if IDS[k] == 'one':
                v = pg.evaluate("()=>{const p=[...document.querySelectorAll('#mha-rail [data-obs] p')].find(p=>/predicted|prediction|预测/.test(p.textContent)); if(!p) return null; const r=p.getBoundingClientRect(); return Math.round(r.bottom)}")
                check(f'R3.10 {lang} part 4: the verdict is on screen', v is not None and v <= VH, v)
        for k in (0, 1, 6, 7):
            go(pg, k); pg.wait_for_timeout(150)
            heights.append((lang, IDS[k], rect(pg, '#mha-rail .step.current')['h']))
        pg.click('#mha-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300)
        for i in range(5):
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(80)
            pg.click(f'#mha-rail [data-preset="{i}"]'); pg.wait_for_timeout(250); settle(pg)
            ins = rect(pg, '#mha-insight')
            check(f'R3.10 {lang} scenario {i + 1}: its result box starts on screen', ins is not None and ins['top'] <= VH - 60, ins)
        pg.click('#mha-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
        pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(300); settle(pg)
        res = rect(pg, '#mha-ch-res')
        check(f'R3.10 {lang} challenge: the feedback starts on screen after submitting', res is not None and res['top'] <= VH - 40, res)
        pg.close()
    notes.append('card heights at 1440 x 900 (841 px below the top bar): ' + ', '.join(f'{l} {i} {hh}px' for l, i, hh in heights))
    # N1: entering part 3 with Next and clicking a word in the card: the picker, the chart's row and the result's first line stay in
    # view, below the top bar and the mini result bar (at 1440 x 900 the whole result fits under the picker)
    POS = """()=>{const q=e=>{if(!e||e.hidden) return null; const r=e.getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom)]};
      const dk=document.getElementById('dock'), bar=document.querySelector('.topbar'), box=document.querySelector('#mha-rail [data-obs]');
      return {y:Math.round(scrollY), safe:Math.round(bar.getBoundingClientRect().bottom+(dk&&!dk.hidden?dk.getBoundingClientRect().height:0)),
        picker:q(document.getElementById('mha-ic-q')), box:q(box), first:q(box&&box.querySelector('p')), h1:q(document.querySelector('#mha-grid .mha-row'))}}"""
    jumps = []
    for w, h, langs in ((1440, 900, ('en', 'zh')), (1280, 900, ('en', 'zh')), (1280, 800, ('en', 'zh')), (900, 900, ('en', 'zh'))):
        for lang in langs:
            pg = page(b, w, h, lang)
            go(pg, 1); pg.click('#mha-rail [data-pred="one"]'); pg.wait_for_timeout(150)
            pg.click('#mha-rail [data-nav="next"]'); pg.wait_for_timeout(300); settle(pg)
            pg.hover('#mha-ic-q button[data-q="5"]'); pg.wait_for_timeout(100); y0 = pg.evaluate('()=>Math.round(scrollY)')   # hovering scrolls the button into view, as a reader would
            pg.click('#mha-ic-q button[data-q="5"]'); pg.wait_for_timeout(250); settle(pg); pg.wait_for_timeout(150)
            d = pg.evaluate(POS); inview = lambda r: r is not None and r[0] >= d['safe'] and r[1] <= h
            jumps.append(f"{w}x{h} {lang} {d['y'] - y0}px")
            check(f'N1 {w} x {h} {lang}: after the first word click the picker is in view', inview(d['picker']), d)
            check(f'N1 {w} x {h} {lang}: the result’s first line is in view', inview(d['first']), d)
            check(f'N1 {w} x {h} {lang}: the chart’s row is in view', inview(d['h1']), d)
            if w == 1440:
                check(f'N1 {w} x {h} {lang}: the whole result is in view', inview(d['box']), d)
                shot(pg, f'n1_{w}_{lang}_p3_first_click')
            pg.close()
    notes.append('N1: scroll after the first word click in part 3: ' + ', '.join(jumps))
    # the same for a scenario: on smaller screens its result starts below the fold; after the click the pressed scenario and the
    # result's first line are in view (R3.10)
    SPOS = """()=>{const q=e=>{const r=e.getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom)]}, dk=document.getElementById('dock');
      return {safe:Math.round(document.querySelector('.topbar').getBoundingClientRect().bottom+(dk&&!dk.hidden?dk.getBoundingClientRect().height:0)),
        pre:q(document.querySelector('#mha-rail .preset[aria-pressed="true"]')), first:q(document.querySelector('#mha-insight p'))}}"""
    for w, h in ((1280, 800), (1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            pg = page(b, w, h, lang)
            pg.click('#mha-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(300)
            bad = []
            for i in range(5):
                pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
                pg.click(f'#mha-rail [data-preset="{i}"]'); pg.wait_for_timeout(250); settle(pg)
                d = pg.evaluate(SPOS)
                if not all(r[0] >= d['safe'] and r[1] <= h for r in (d['pre'], d['first'])): bad.append((i + 1, d))
            check(f'R3.10 {w} x {h} {lang}: after each scenario click, the pressed scenario and the result’s first line are in view', not bad, bad[:2])
            pg.close()
    # R3.2: a change in Settings never scrolls the page (guided part 5 and Scenarios)
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang)
        go(pg, 1); pg.click('#mha-rail [data-pred="both"]'); pg.wait_for_timeout(120); go(pg, 4)
        pg.evaluate("()=>{document.getElementById('mha-ctl-H').scrollIntoView({block:'center'})}"); pg.wait_for_timeout(300); settle(pg)
        # since 2026-10-11 Settings sit under the card when every part is shown (tests); the card above grows, so the page holds the button in place instead of holding scrollY
        y0 = pg.evaluate("()=>Math.round(document.querySelector('#mha-ctl-H button[data-v=\"2\"]').getBoundingClientRect().top)")
        pg.click('#mha-ctl-H button[data-v="2"]'); pg.wait_for_timeout(400); settle(pg)
        y1 = pg.evaluate("()=>Math.round(document.querySelector('#mha-ctl-H button[data-v=\"2\"]').getBoundingClientRect().top)")
        check(f'R3.2 {lang}: 2 heads picked in Settings in part 5 does not move the page under the pointer', abs(y1 - y0) <= 1, (y0, y1))
        check(f'R3.2 {lang}: …and it still counts as the part’s action', not pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
        pg.click('#mha-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(350)
        pg.click('#mha-app [data-flow-nav] button[data-flow=free]'); pg.wait_for_timeout(350)   # settings are changed in 自由调参 (owner, 2026-10-11)
        pg.evaluate("()=>{document.getElementById('mha-ctl-q').scrollIntoView({block:'center'})}"); pg.wait_for_timeout(300); settle(pg)
        y0 = pg.evaluate("()=>Math.round(document.querySelector('#mha-ctl-q button[data-q=\"4\"]').getBoundingClientRect().top)")
        pg.click('#mha-ctl-q button[data-q="4"]'); pg.wait_for_timeout(400); settle(pg)
        y1 = pg.evaluate("()=>Math.round(document.querySelector('#mha-ctl-q button[data-q=\"4\"]').getBoundingClientRect().top)")
        check(f'R3.2 {lang}: a word picked in Settings in Scenarios does not move the page under the pointer', abs(y1 - y0) <= 1, (y0, y1))
        pg.close()
    # N8 (now in part 4): another word before “it” gets its own result and points to “it”, with no verdict; N9: a skipped prediction
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang)
        go(pg, 1); pg.click('#mha-rail [data-pred="both"]'); pg.wait_for_timeout(120); go(pg, 3)
        pg.click('#mha-ic-q button[data-q="5"]'); pg.wait_for_timeout(300)
        ob = pg.inner_text('#mha-rail [data-obs]') if pg.is_visible('#mha-rail [data-obs]') else ''
        check(f'N8 {lang}: another word first shows its result and points to “it”', Z(lang, 'Pick “it” to see whether your prediction was right.', '选“它”，看你的预测对不对。') in ob and not re.search(Z(lang, 'You predicted|right answer', '你预测的是|正确答案'), ob), ob[:300])
        check(f'N8 {lang}: Next still waits for “it”', pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"))
        pg.click('#mha-ic-q button[data-q="8"]'); pg.wait_for_timeout(300)
        ob = pg.inner_text('#mha-rail [data-obs]')
        check(f'N8 {lang}: then “it” gives the verdict', Z(lang, 'You predicted “Both can get 80% or more”', '你预测的是“两个都能在 80% 以上”') in ob, ob[:300])
        if lang == 'zh':
            check('M8 zh: the comma token is written 逗号“，” in part 4', '剩下的给了逗号“，”和“它”' in ob, ob[-200:])   # R6.20 wording: where the rest of the weight goes
        pg.close()
        pg = page(b, 1440, 900, lang)
        go(pg, 1); pg.click('#mha-rail [data-nav="skip"]'); pg.wait_for_timeout(200); go(pg, 3)
        pg.click('#mha-ic-q button[data-q="8"]'); pg.wait_for_timeout(300)
        ob = pg.inner_text('#mha-rail [data-obs]')
        check(f'N9 {lang}: after skipping the prediction, part 4 still says why at most one word can get 80%', Z(lang, 'One head’s attention weights add up to 100%, so at most one word can get 80% or more.', '一个 head 的注意力权重加起来是 100%，所以最多只有一个词能在 80% 以上。') in ob, ob[:300])
        pg.close()
    # M3 (now in part 5): 4 heads before 2 gives the state and points to 2 heads; 2 heads then gives the result
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang)
        go(pg, 1); pg.click('#mha-rail [data-pred="both"]'); pg.wait_for_timeout(120); go(pg, 4)
        pg.click('#mha-ctl-H button[data-v="4"]'); pg.wait_for_timeout(300)
        ob = pg.inner_text('#mha-rail [data-obs]') if pg.is_visible('#mha-rail [data-obs]') else ''
        check(f'M3 {lang}: 4 heads picked in Settings in part 5 shows that state and points to 2 heads', Z(lang, 'Switch to 2 heads.', '切换到 2 个 head。') in ob and pg.evaluate("()=>document.querySelector('#mha-rail [data-nav=next]').disabled"), ob[:300])
        pg.click('#mha-ic-H button[data-v="2"]'); pg.wait_for_timeout(300)
        ob = pg.inner_text('#mha-rail [data-obs]')
        check(f'M3 {lang}: then 2 heads gives the part’s result', Z(lang, 'Two heads: at “it”', '两个 head：当前的词是“它”时') in ob and Z(lang, 'Switch to 2 heads.', '切换到 2 个 head。') not in ob, ob[:300])
        go(pg, 6)
        order = pg.evaluate("()=>[...document.querySelectorAll('#mha-rail .step.current .def')].map(d=>d.textContent.trim())")
        card7 = pg.inner_text('#mha-rail .step.current')
        check(f'N9 {lang}: part 7 introduces only KV heads and GQA (with its full name), no MHA, query head or KV cache', order == [Z(lang, 'KV heads', 'KV head'), 'GQA'] and Z(lang, 'grouped-query attention (GQA)', 'grouped-query attention（GQA）') in card7 and all(x not in card7 for x in ('MHA', 'query head', 'KV cache')), (order, card7[-300:]))
        # the 前置知识 popover went with the header buttons (owner, 2026-10-11 19:28 UTC)
        # 3-3 / ARCH #16: the causal mask, softmax and query/key/value hovers are the kit texts, then one sentence of this lab
        go(pg, 2)
        tips = pg.evaluate("()=>{const o={}; document.querySelectorAll('#mha-rail .step.current .def').forEach(d=>{o[d.textContent.trim()]=d.getAttribute('data-tipx')||''}); return o}")
        kit = pg.evaluate("()=>({mask:ASP.terms.mask(), softmax:ASP.terms.softmax(), qkv:ASP.terms.qkv()})")
        qkv = tips.get(Z(lang, 'query, key and value', 'query、key、value'), '')
        check(f'3-3 {lang}: query, key and value: the kit text, then this lab’s 8-number sentence', qkv.startswith(kit['qkv']) and Z(lang, 'each is 8 numbers', '三个向量都是 8 维') in qkv, qkv[-200:])
        check(f'3-3 {lang}: softmax: the kit text, then one sentence', tips.get('softmax', '').startswith(kit['softmax']) and Z(lang, 'attention weights', '注意力权重') in tips.get('softmax', '')[len(kit['softmax']):], tips.get('softmax', '')[-160:])
        pg.click('#mha-ic-q button[data-q="5"]'); pg.wait_for_timeout(300)
        mk = pg.evaluate("()=>{const d=[...document.querySelectorAll('#mha-rail [data-obs] .def')].find(e=>/causal mask|因果 mask/.test(e.textContent)); return d?d.getAttribute('data-tipx'):''}")
        check(f'3-3 {lang}: causal mask: the kit text, then “In this lab each token is a word.”', mk == kit['mask'] + Z(lang, ' In this lab each token is a word.', '在这个 Lab 里，每个 token 就是一个词。'), mk)
        pg.close()
    # N6 and M2: the challenge's mini result bar on a phone: the constraint before and after submitting; “Goal met” only for H1 and H3
    for lang in ('en', 'zh'):
        pg = page(b, 390, 844, lang)
        pg.click('#mha-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
        dockup = "()=>{document.querySelector('#mha-controls').scrollIntoView({block:'start'}); window.dispatchEvent(new Event('scroll'))}"
        pg.evaluate(dockup); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>{const d=document.getElementById('dock'); return {hidden:d.hidden, st:(d.querySelector('.status')||{}).textContent||'', t:d.innerText.replace(/\\s+/g,' ')}}")
        r4 = pg.evaluate(f"()=>{{const r=__mha.compute({{lang:'{lang}',q:8,H:4}}); return [Math.round(r.bec.v*100),Math.round(r.catw.v*100)]}}")
        check(f'N6 390 {lang}: before submitting, the mini result bar states the constraint', not dk['hidden'] and dk['st'] == Z(lang, 'Both ≥ 80%', '两个都 ≥ 80%'), dk)
        check(f'N6 390 {lang}: the sentence on the phone includes the heads turned off', Z(lang, f'At “it”, {r4[0]}% on “because” and {r4[1]}% on “cat”, with 0 of 4 heads off', f'当前的词是“它”：“因为”{r4[0]}%，“小猫”{r4[1]}%，关掉了 0 / 4 个 head') in dk['t'], dk)
        shot(pg, f'n6_390_{lang}_challenge_dock')
        pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(150)
        pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        pg.evaluate(dockup); pg.wait_for_timeout(400)
        st = pg.evaluate("()=>(document.querySelector('#dock .status')||{}).textContent||''")
        check(f'M2 390 {lang}: after submitting with all 4 heads on, still the constraint, not “Goal met”', st == Z(lang, 'Both ≥ 80%', '两个都 ≥ 80%'), st)
        for hh in ('1', '3'):
            pg.click(f'#mha-ic-on button[data-on="{hh}"]'); pg.wait_for_timeout(200)
        pg.evaluate(dockup); pg.wait_for_timeout(400)
        st = pg.evaluate("()=>(document.querySelector('#dock .status')||{}).textContent||''")
        check(f'M2 390 {lang}: with only H1 and H3 on, “Goal met”', st == Z(lang, 'Goal met', '达成目标'), st)
        pg.close()
    # M4: after Submit the head switches stay in view below the top bar and the mini result bar, with the result's first line on screen
    for w, h in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            pg = page(b, w, h, lang)
            pg.click('#mha-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
            pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(250); settle(pg); pg.wait_for_timeout(150)
            d = pg.evaluate("""()=>{const q=e=>{const r=e.getBoundingClientRect(); return [Math.round(r.top),Math.round(r.bottom)]}, dk=document.getElementById('dock');
              return {safe:Math.round(document.querySelector('.topbar').getBoundingClientRect().bottom+(dk&&!dk.hidden?dk.getBoundingClientRect().height:0)),
                sw:q(document.getElementById('mha-ic-on')), first:q(document.querySelector('#mha-ch-res p'))}}""")
            check(f'M4 {w} x {h} {lang}: after Submit the switches and the result’s first line are in view', all(r[0] >= d['safe'] and r[1] <= h for r in (d['sw'], d['first'])), d)
            if w == 390: shot(pg, f'm4_{w}_{lang}_after_submit')
            pg.close()
    # N7: the link that opens the chart names how many choices it compares
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang)
        pg.click('#mha-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(300)
        pg.click('#mha-ic-on button[data-on="2"]'); pg.wait_for_timeout(150); pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        lk = pg.inner_text('#mha-ch-res [data-act="reveal"]') if pg.query_selector('#mha-ch-res [data-act="reveal"]') else ''
        check(f'N7 {lang}: the reveal link says what it shows', lk == Z(lang, 'Compare 6 ways to turn heads off', '看 6 种开关组合的对比'), lk)
        pg.click('#mha-ch-res [data-act="reveal"]'); pg.wait_for_timeout(300)
        n = pg.evaluate("()=>document.querySelectorAll('#mha-ch-res .mha-cr').length")
        check(f'N7 {lang}: the chart then shows those 6 choices', n == 6, n)
        pg.close()
    # the fidelity card: the four patterns in the heads' order, the KV heads in part 7, no language note (the numbers are the same)
    pg = page(b, 1440, 900, 'en')
    fid = pg.evaluate("()=>{const p=document.querySelector('#mha-app [id$=\"-fid\"]'); return p?p.textContent:''}")
    check('fidelity: the four patterns in the heads’ order, the KV heads in part 7', '(the previous word, the first token, the word “it” refers to, the nouns)' in fid and 'KV heads (part 7)' in fid and 'English and Chinese' not in fid, fid[:300])
    # R7.6: the code block reads without sideways scrolling at 1440 (Connect is two columns)
    cw = pg.evaluate("()=>{const e=document.getElementById('mha-code'); return [e.scrollWidth,e.clientWidth]}")
    check('R7.6: the PyTorch code fits its card at 1440 (no sideways scroll)', cw[0] <= cw[1], cw)
    pg.close()
    # the mini result bar at 390: appears once the chart is off screen, with a status and a full sentence
    pg = page(b, 390, 844, 'en')
    go(pg, 1); pg.click('#mha-rail [data-pred="one"]'); go(pg, 2); pg.click('#mha-ic-q button[data-q="3"]'); pg.wait_for_timeout(300)
    pg.evaluate("()=>{document.querySelector('#mha-controls').scrollIntoView({block:'start'}); window.dispatchEvent(new Event('scroll'))}"); pg.wait_for_timeout(400)
    dk = pg.evaluate("()=>({hidden:document.getElementById('dock').hidden, t:document.getElementById('dock').innerText})")
    w3 = pg.evaluate("()=>{const r=__mha.compute({lang:'en',q:3,H:1}); return Math.round(r.heads[0].w[2]*100)}")
    check('390: the mini result bar shows when the chart is off screen', not dk['hidden'] and f'{w3}%' in dk['t'], dk)
    shot(pg, '390_en_dock')
    pg.close()
    # other widths: no overflow, the rail comes first on phones, the stage beside the cards on desktops
    for w in (1280, 900, 720):
        pg = page(b, w, 900, 'zh', 'mha~8_4_1111')
        check(f'{w}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
        pg.click('#mha-app .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
        go(pg, 1); pg.click('#mha-rail [data-pred="both"]'); go(pg, 3); pg.click('#mha-ic-q button[data-q="8"]'); pg.wait_for_timeout(300)
        check(f'{w}: no horizontal overflow in part 4', overflow(pg) <= 0, overflow(pg))
        shot(pg, f'{w}_zh_p4')
        pg.close()
    # keyboard: a token button and the head switches work with Enter
    pg = page(b, 1440, 900, 'en', 'mha~8_4_1111')
    pg.focus('#mha-grid .mha-tok[data-q="5"]'); pg.keyboard.press('Enter'); pg.wait_for_timeout(200)
    check('keyboard: Enter on a word makes it the current word', pg.evaluate("()=>document.querySelector('#mha-grid .mha-tok.q').dataset.q") == '5')
    pg.close()
    # the home tile: bars drawn for 1 and 2 heads, both languages, desktop and phone, and after the lab was used
    for w, h in ((1440, 900), (390, 844)):
        for lang in ('en', 'zh'):
            pg = page(b, w, h, lang, 'mha~3_4_1111')          # open the lab first, so the tile is drawn after a lab state exists
            pg.evaluate("()=>{location.hash=''}"); pg.wait_for_timeout(400)
            pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
              if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
            for v in ('1', '2'):
                pg.click(f'#tile-mha .seg-ctl button[data-v="{v}"]'); pg.wait_for_timeout(150)
                bars = pg.evaluate("()=>[...document.querySelectorAll('#tile-mha .mha-bar')].filter(b=>b.getBoundingClientRect().height>0).length")
                rows = pg.evaluate("()=>document.querySelectorAll('#tile-mha .mha-row').length")
                check(f'tile {w} {lang}, {v} head(s): {rows} row(s) with visible bars', rows == (1 if v == '1' else 2) and bars >= (4 if v == '1' else 4), (rows, bars))   # 1 head: 43, 41, 8, 8; 2 heads: head A 83, 8, 8 and head B 100
                check(f'tile {w} {lang}: no horizontal overflow', overflow(pg) <= 0, overflow(pg))
                pg.evaluate("()=>document.querySelector('#tile-mha').scrollIntoView({block:'center'})"); pg.wait_for_timeout(150)
                el = pg.query_selector('#tile-mha'); pg.mouse.move(1, 1); el.screenshot(path=str(SH / f'tile_{w}_{lang}_{v}.png'))
            if lang == 'zh':
                cap = pg.inner_text('#tile-mha')
                check(f'tile {w} zh: the 2-head caption reads “在每个 head 里，“它”把多少注意力放在每个词上” (N16 c)', '竖条：在每个 head 里，“它”把多少注意力放在每个词上。虚线框：“它”需要的两个词。' in cap, cap[-200:])
            for hit in pg.evaluate(RULES, '#tile-mha'): rule_hits.append(f'tile {w} {lang}: {hit}')
            pg.close()
    # the owner's decisions of 2026-10-09 (H1, H2): Show the math is a few blocks, each a lead line in words and a typeset formula (MathML),
    # with a hairline between blocks; no TeX command shown as a word; nothing clipped
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang); go(pg, 3)
        pg.evaluate("()=>{document.getElementById('mha-math').open=true}"); pg.wait_for_timeout(200)
        mb = pg.evaluate(MATHBLOCKS, ['#mha-mathbody', 'mha'])
        check(f'{lang} H2: Show the math in blocks (lead line + typeset formula, hairline between), no TeX word, nothing clipped',
              bool(mb) and mb['n'] >= 4 and mb['lead'] == mb['n'] and mb['fx'] == mb['n'] and mb['ruled'] == mb['n'] and not mb['words'] and mb['clip'] == 0, mb)
        t = pg.evaluate("s=>{const e=document.querySelector(s).cloneNode(true); e.querySelectorAll('math').forEach(m=>m.remove()); return e.textContent}", '#mha-mathbody')   # the text outside the typeset formulas
        check(f'{lang} H1: no plain-text formula left in Show the math (e^(, Σ attention, Wq·x, √ as text)', not any(x in t for x in ['e^(', 'Σ attention', 'Σ 注意力', 'Wq·x', 'Wo·(']), t[:300])
        go(pg, 2)
        check(f'{lang} H1: part 2 shows the score formula on its own line, typeset', pg.evaluate("()=>!!document.querySelector('#mha-rail .step.current .mha-math-line math msqrt')"))
        pg.close()
    b.close()

for h in rule_hits[:40]: fails.append('rule: ' + h)
for h in sorted(set(x.split(': ',1)[1] for x in rule_hits))[:60]: print('RULE', h)
print('\n'.join(n for n in notes if n.startswith('FAIL')) or 'all checks passed')
print('\n'.join(n for n in notes if n.startswith('card heights') or n.startswith('N1:')))
print('page errors:', errs[:5])
if errs: fails.append('page errors')
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'mha.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a scenario and see which word each head puts its attention on.', '选一个场景，看看每个 head 把注意力放在哪个词上。', 'Turn off any heads you like, several at once, and see which words the block still takes from.', '想关哪个 head 就关哪个，可以同时关几个，看看这个块还能从哪些词取信息。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('${oneEn}</li>', 'one head has to split its attention between them, and neither gets most of it.'), ('${oneZh}</li>', '一个 head 只能把注意力分给两边，哪一边都拿不到大头。'), ('at “${esc(wd(3))}”, every word after it gets ${masked3}', 'so the model cannot peek at words it has not written yet.'), ('它后面的每个词都是 ${masked3}', '模型看不到还没写出来的词。')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# the owner's decisions of 2026-10-09: H1, no plain-text formula of the kinds converted left in the lab's source (outside comments and
# the plain number lines); H4, the grouped-query attention lab linked as live, not “two labs on”
_H_CODE = __import__('re').sub(r'/\*.*?\*/', '', _G_SRC, flags=__import__('re').S)
for _bad in ['q·k ÷ √head_dim', '√head_dim', '√d_head', '√d_h', 'e^(', 'Σ attention weight', 'Σ 注意力权重', 'Wq·x', 'q·kᵀ', 'head_dim^−0.5', "<th>q·k</th>", '8 ÷ heads', '8 ÷ head 数', 'two labs on', '再往后两个 Lab']:
    check(f'H1/H4: “{_bad}” is gone from the lab', _bad not in _H_CODE)
check('H1: formulas use A.ui.math', _H_CODE.count('A.ui.math(') + _H_CODE.count('MX(') >= 15, _H_CODE.count('MX('))
check('H4: the GQA lab is linked with A.shell.labLink', "A.shell.labLink('gqa-mla')" in _H_CODE)
print(f'{len(notes)} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
