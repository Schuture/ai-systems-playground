"""Flow test for the pd-disagg lab, in English and Chinese, at 1440 and 390 wide. Filters: en|zh, 1440|390 (python3 tests/pd-disagg_flow.py zh 390).
- the guided parts in order; Next waits with the kit's why line where a part asks for a choice (parts 3-7); the prediction gate (part 4);
- R3.20: entering part 2 plays its run; Back and Next again replays nothing (the run was watched); in part 3 a new budget plays, the
  same budget again and a budget already watched replay nothing; Replay plays from the start; Play pauses and resumes; the speed
  control changes the speed;
- parts 6 and 7 show their observations (27 ms -> 104 ms; the sweep), part 8 links to the scenarios;
- Scenarios: five, each one shows its insight; a new one plays, one already watched does not; the hash round trip;
- the challenge: starts at 2 prefill + 6 decode; a pick that misses says why and offers Show the answer, which shows the chart of all
  18 setups; a split that passes is told a colocated setup has more room; a budget of 2,048 is told it has the highest goodput (tied with 512); the exits;
- the mini result bar (dock) at 390 when the stage is off screen;
- the home tile: the switch changes the sentence and the bars, chip Toy run;
- no horizontal page scroll, no page errors, the text rules (no “toy”, no company names as sources of practice problems, no two colons in
  one sentence, no “一次只能选 / 不会叠加 / 第 N 部分里”).
Run: python3 tests/pd-disagg_flow.py [en|zh] [1440|390]"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'pd-disagg').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/pd-disagg.html', '--only', 'pd-disagg'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'pd-disagg' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'pd-disagg.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pd-disagg' / 'flow-wrapped.html')
SH = P / 'shots' / 'pd-disagg'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='pd-disagg'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__pdd.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
cur = lambda pg: text(pg, '#pdd-rail .step.current')
next_dis = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#pdd-rail .step.current [data-nav=\"next\"]'); return b?b.disabled:null}")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#pdd-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#pdd-run .seg-ctl[data-key=\"spd\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait)


def finish(pg):
    """scrub the run to its end, as a reader can with the progress bar"""
    pg.evaluate("()=>{const r=document.querySelector('#pdd-prog'); r.value='1000'; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true}));}")
    pg.wait_for_timeout(150)


def nav(pg, which):
    click(pg, f'#pdd-rail .step.current [data-nav="{which}"]', 350)


def mode(pg, m):
    click(pg, f'#pdd-app .modes button[data-mode="{m}"]', 400)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里/,'banned phrase'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数”'],[/OpenAI|Anthropic|DeepMind|Google|Meta\\b|Microsoft/,'company name'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness']];
 let n; while((n=tw.nextNode())){const e=n.parentElement; if(!e||e.closest('[data-endnotes],pre, code, #tip, .fid-tag')||!e.getClientRects().length) continue; const t=n.textContent;
   for(const [re,name] of bad) if(re.test(t)) out.push(name+': '+t.trim().slice(0,90));}
 root.querySelectorAll('[data-tipx]').forEach(e=>{const t=e.getAttribute('data-tipx'); for(const [re,name] of bad) if(re.test(t)) out.push(name+' (tooltip): '+t.slice(0,90));});
 return out;}"""
COLONS = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=new Set(), texts=[];
 root.querySelectorAll('p,li,button,label,div,span,h1,h2,h3,summary').forEach(e=>{if(e.closest('[data-endnotes],pre,code,svg,.mathbody')||!e.getClientRects().length) return; if(e.querySelector('p,li,div,ul,ol,h1,h2,h3,table')) return;
   if(![...e.childNodes].some(c=>c.nodeType===3&&c.textContent.trim())) return; texts.push(e.textContent);});
 root.querySelectorAll('[data-tipx]').forEach(e=>texts.push(e.getAttribute('data-tipx')));
 for(const t of texts){const sents=t.replace(/\\s+/g,' ').split(/(?<=[。？！])|(?<=[.?!])\\s+/);
   for(const x of sents){const k=(x.match(/：|: /g)||[]).length; if(k>=2) out.add(x.trim().slice(0,160));}}
 return [...out];}"""
rule_hits = []


def rules(pg, where):
    for sel in ('#pdd-app', '#tile-pd-disagg'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons: {h}')
    ov = overflow(pg); check(f'{where}: no horizontal page scroll', ov <= 0, ov)


def guided(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    title = text(pg, '#pdd-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'Why run prefill and decode on different GPUs?', '为什么把 prefill 和 decode 放到不同的 GPU 上？'), title)
    s = state(pg); check(f'{tag} part 1: guided, nothing plays on the first part', s['mode'] == 'guided' and s['step'] == 0 and s['plays'] == 0 and not s['playing'], s)
    rules(pg, f'{tag} part 1')
    # part 2: a new run plays; Back and Next again replays nothing (R3.20)
    nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 2: entering plays the stall run at fast speed (about 10 s, R3.9)', s['step'] == 1 and s['playing'] and s['plays'] == 1 and speed(pg) == 'fast', (s, speed(pg)))
    click(pg, '#pdd-play', 200); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} part 2: Play pauses the run where it is', not s1['playing'] and s1['ph'] == s2['ph'] and s1['ph'] > 0, (s1['ph'], s2['ph']))
    click(pg, '#pdd-play', 300); s3 = state(pg)
    check(f'{tag} part 2: Play resumes from there', s3['playing'] and s3['ph'] >= s1['ph'] and s3['plays'] == 2, s3)
    click(pg, '#pdd-run .seg-ctl[data-key="spd"] button[data-v="slow"]', 150)
    check(f'{tag} part 2: the speed control switches to slow', speed(pg) == 'slow' and state(pg)['spd'] == 'slow', speed(pg))
    click(pg, '#pdd-run .seg-ctl[data-key="spd"] button[data-v="fast"]', 150)
    finish(pg); t2 = cur(pg)
    check(f'{tag} part 2: after the run, the observation quotes the 1.49 s step', Z(lang, 'That step takes 1.49 s', '这一步要 1.49 秒') in t2, t2[-400:])
    rules(pg, f'{tag} part 2')
    p0 = state(pg)['plays']; nav(pg, 'back'); nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 2 again: Back and Next replay nothing (R3.20)', s['step'] == 1 and not s['playing'] and s['plays'] == p0, s)
    click(pg, '#pdd-replay', 200); s = state(pg)
    check(f'{tag} part 2: Replay plays the watched run from the start', s['playing'] and s['plays'] == p0 + 1 and s['ph'] < 0.5, s)
    finish(pg)
    # part 3: Next waits for a budget; a new budget plays, the same one or a watched one does not
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 3: Next waits for a budget, with the why line', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick a token budget to continue.', '选一个 token 预算才能继续。')}, (next_dis(pg), why(pg)))
    p0 = state(pg)['plays']; click(pg, '#pdd-rail [data-lever="512"]'); s = state(pg)
    check(f'{tag} part 3: picking 512 plays the new run (R3.20)', s['playing'] and s['plays'] == p0 + 1 and s['p3b'] == 512, s)
    finish(pg); click(pg, '#pdd-rail [data-lever="512"]'); s = state(pg)
    check(f'{tag} part 3: picking 512 again replays nothing (R3.20)', not s['playing'] and s['plays'] == p0 + 1, s)
    click(pg, '#pdd-rail [data-lever="2048"]'); finish(pg); click(pg, '#pdd-rail [data-lever="512"]'); s = state(pg)
    check(f'{tag} part 3: back to the watched 512 replays nothing (R3.20)', not s['playing'] and s['plays'] == p0 + 2 and s['p3b'] == 512, s)
    t3 = cur(pg); check(f'{tag} part 3: the observation quotes 32 ms and Next is enabled', Z(lang, 'falls to 32 ms', '降到 32 毫秒') in t3 and next_dis(pg) is False, t3[-300:])
    rules(pg, f'{tag} part 3')
    # part 4: the prediction gate
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 4: Next waits for an answer', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick an answer to continue.', '选一个答案才能继续。')}, why(pg))
    click(pg, '#pdd-rail [data-pred="both"]'); t4 = cur(pg)
    check(f'{tag} part 4: an answer is saved and Next opens', next_dis(pg) is False and Z(lang, 'Saved.', '已记下') in t4, t4[-400:])
    nav(pg, 'next'); finish(pg); v5 = text(pg, '#pdd-verdict')
    check(f'{tag} part 5: the verdict on a wrong answer names what really happens', bool(v5.strip()) and Z(lang, 'documents', '文档') in v5, v5[:300])
    nav(pg, 'back'); click(pg, '#pdd-rail [data-pred="gapT"]')
    # part 5: another split
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 5: Next waits for another split', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try another split to continue.', '换一种分法才能继续。')}, why(pg))
    p0 = state(pg)['plays']; click(pg, '#pdd-rail [data-lever="2"]'); s = state(pg); finish(pg)
    check(f'{tag} part 5: 2 + 2 plays its run and opens Next', s['plays'] == p0 + 1 and next_dis(pg) is False and '98%' in cur(pg), (s, cur(pg)[-300:]))
    rules(pg, f'{tag} part 5')
    # part 6: the link
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 6: Next waits for the link across nodes', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try the link across nodes to continue.', '试一下跨机器传才能继续。')}, why(pg))
    click(pg, '#pdd-rail .seg-ctl[data-key="p6l"] button[data-v="ib"]'); finish(pg); t6 = cur(pg)
    check(f'{tag} part 6: the observation quotes 27 ms -> 104 ms and 98%', Z(lang, 'from 27 ms to 104 ms', '从 27 毫秒涨到 104 毫秒') in t6 and Z(lang, 'With either link, 98% of requests', '两种连接下，都是 98% 的请求达标') in t6 and next_dis(pg) is False, t6[-400:])
    rules(pg, f'{tag} part 6')
    # part 7: the sweep
    nav(pg, 'next'); pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=90000); pg.wait_for_timeout(300); finish(pg)
    check(f'{tag} part 7: Next waits for a setup', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick a setup to show to continue.', '选一种方案显示才能继续。')}, why(pg))
    chart = text(pg, '#pdd-app .stage')
    check(f'{tag} part 7: the chart shows goodput per GPU for the setups', '6.5' in chart and '4' in chart, chart[:300])
    click(pg, '#pdd-rail .seg-ctl[data-key="p7"] button[data-v="0"]'); finish(pg); t7 = cur(pg)
    check(f'{tag} part 7: the observation names the best split and the chunked colocated run', Z(lang, 'The best split gives 6 GPUs to prefill and the rest to decode', '最好的分法是 6 张 prefill、2 张 decode') in t7 and '6.5' in t7 and next_dis(pg) is False, t7[-500:])
    pg.screenshot(path=str(SH / f'flow-{tag}-p7.png'))
    rules(pg, f'{tag} part 7')
    # part 8
    nav(pg, 'next'); finish(pg); t8 = pg.evaluate("()=>document.querySelector('#pdd-rail .step.current').textContent")
    check(f'{tag} part 8: what to remember, misconceptions, real systems', all(x in t8 for x in (Z(lang, 'What to remember', '要记住的'), Z(lang, 'Common misconceptions', '常见误解'), Z(lang, 'What real systems do', '真实系统怎么做'), 'DistServe', 'Mooncake', 'Dynamo'))
          and pg.evaluate("()=>!!document.querySelector('#pdd-rail .step.current [data-mode-go=\"sandbox\"], #pdd-rail .step.current [data-modego=\"sandbox\"], #pdd-rail .step.current a[href*=\"scen\"], #pdd-rail .step.current button')"), t8[:200])
    rules(pg, f'{tag} part 8')
    pg.screenshot(path=str(SH / f'flow-{tag}-p8.png'))
    pg.close()


def sandbox(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'sandbox'); finish(pg)
    pres = pg.evaluate("()=>[...document.querySelectorAll('#pdd-app [data-preset]')].map(e=>e.textContent.trim().slice(0,40))")
    check(f'{tag} scenarios: five', len(pres) == 5, pres)
    for i in range(5):
        p0 = state(pg)['plays']; click(pg, f'#pdd-app [data-preset="{i}"]', 300); s = state(pg)
        ins = text(pg, '#pdd-insight')
        # scenario 0 is the sandbox's own start, already shown: nothing changes, nothing replays (R3.20)
        check(f'{tag} scenario {i}: {"the run on screen does not replay" if i == 0 else "a new run plays"} and its insight shows', s['plays'] == p0 + (0 if i == 0 else 1) and '%' in ins, (s['plays'], p0, ins[:120]))
        finish(pg)
    p0 = state(pg)['plays']; click(pg, '#pdd-app [data-preset="1"]', 300); s = state(pg)
    check(f'{tag} scenarios: one already watched replays nothing (R3.20)', s['plays'] == p0 and not s['playing'], s)
    hs = pg.evaluate('()=>location.hash')
    check(f'{tag} sandbox writes its settings into the link', hs == '#pd-disagg~A_8_co_6_512_nvl_3', hs)
    rules(pg, f'{tag} sandbox')
    if w == 390:
        pg.evaluate("()=>{const r=document.querySelector('#pdd-rail'); window.scrollTo(0, r.getBoundingClientRect().top+scrollY+r.offsetHeight-200)}"); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>{const d=document.querySelector('#dock'); if(!d) return null; const r=d.getBoundingClientRect(); return {vis:getComputedStyle(d).display!=='none'&&r.height>0&&r.bottom<=innerHeight+1, t:d.innerText}}")
        check(f'{tag}: the mini result bar shows the share meeting goals when the stage is off screen', dk and dk['vis'] and '%' in dk['t'], dk)
    pg.close()
    pg = page(b, w, h, lang, hash_='pd-disagg~A_8_split_5_0_ib_2.5'); s = state(pg)
    check(f'{tag} hash round trip: a link opens the sandbox with its settings', s['mode'] == 'sandbox' and s['kind'] == 'A' and s['n'] == 8 and s['p'] == 5 and s['link'] == 'ib' and s['rate'] == 2.5 and s['mode'] == 'sandbox', s)
    pg.close()


def challenge(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'challenge'); finish(pg); s = state(pg)
    check(f'{tag} challenge: starts at 2 prefill + 6 decode in one node', s['mode'] == 'challenge' and s['p'] == 2 and s['link'] == 'nvl', s)
    click(pg, '#pdd-app .seg-ctl[data-key="chp"] button[data-v="3"]'); finish(pg)
    click(pg, '#pdd-app [data-act="submit"]', 350); r = text(pg, '#pdd-ch-res')
    check(f'{tag} challenge: 3 + 5 misses, says why, offers the answer', Z(lang, 'Not met: 24% of requests meet their goals.', '没达标：24% 的请求达标。') in r and Z(lang, 'Show the answer', '直接看答案') in r and not pg.query_selector('#pdd-ch-res .cmp'), r)
    click(pg, '#pdd-ch-res [data-act="reveal"]', 350); r = text(pg, '#pdd-ch-res')
    rows = pg.evaluate("()=>document.querySelectorAll('#pdd-ch-res .cmp-row, #pdd-ch-res [class*=cmp] li, #pdd-ch-res tr').length")
    tags = text(pg, '#pdd-ch-res')
    check(f'{tag} challenge: Show the answer shows the chart (passing setups, your pick, the initial setup) and why', rows >= 4 and Z(lang, 'your pick', '你的选择') in tags and Z(lang, 'initial setup', '初始设置') in tags and Z(lang, 'The initial setup, 2 prefill + 6 decode, meets only 7%.', '初始设置是 2 张 prefill、6 张 decode，只有 7% 达标。') in r and Z(lang, 'goodput 3 per GPU', '达标吞吐：每张卡每秒 3 个') in tags, (rows, r[:300]))
    click(pg, '#pdd-app .seg-ctl[data-key="chp"] button[data-v="5"]'); finish(pg)
    pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=90000); pg.wait_for_timeout(400); r = text(pg, '#pdd-ch-res')
    check(f'{tag} challenge: 5 + 3 passes but a colocated setup has a higher goodput per GPU', Z(lang, 'Met: 100% of requests meet their goals.', '达标：100% 的请求达标。') in r and Z(lang, 'But “colocated, budget of 2,048 tokens per step” has a higher goodput per GPU.', '不过“混在一起跑，每步的 token 预算 2,048”的达标吞吐更高。') in r and Z(lang, 'It still passes at 3 requests per second per GPU; yours at 1.5, so a little more traffic and yours misses the target.', '它在每张卡每秒 3 个请求时仍能达标，你的是 1.5 个，流量再多一点就达不到目标了。') in r, r[:400])
    click(pg, '#pdd-app .seg-ctl[data-key="chm"] button[data-v="co"]'); click(pg, '#pdd-app .seg-ctl[data-key="chb"] button[data-v="2048"]'); finish(pg)
    pg.wait_for_function('()=>window.__pdd.state().gpQueue===0', timeout=90000); pg.wait_for_timeout(300); r = text(pg, '#pdd-ch-res')
    check(f'{tag} challenge: a budget of 2,048 meets the target with the highest goodput, tied with 512', Z(lang, 'this setup has the highest goodput per GPU (tied with “colocated, budget of 512 tokens per step”).', '而且每张卡的达标吞吐最高（和“混在一起跑，每步的 token 预算 512”并列）。') in r and Z(lang, 'It still passes when traffic grows to 3 requests per second per GPU; today it is 1.5.', '流量涨到每张卡每秒 3 个请求时，它仍能达标；现在是 1.5 个。') in r, r[:300])
    ex = pg.evaluate("()=>[...document.querySelectorAll('#pdd-app .exits a, #pdd-app .exits button, #pdd-app [data-exit]')].length")
    check(f'{tag} challenge: the exits follow the result', ex >= 3, ex)
    pg.screenshot(path=str(SH / f'flow-{tag}-challenge.png'), full_page=False)
    rules(pg, f'{tag} challenge')
    pg.close()


def tile(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang, hash_='map')
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="inference"], #domStrip [data-world="inference"]')].find(e=>e.offsetParent);
      if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="inference"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
    t = pg.query_selector('#tile-pd-disagg')
    check(f'{tag} tile: the pd-disagg tile is on the inference branch', t is not None)
    if not t: pg.close(); return
    t.scroll_into_view_if_needed(); pg.wait_for_timeout(150)
    r0 = text(pg, '#tile-pd-disagg .tz-read'); bars0 = pg.evaluate("()=>document.querySelectorAll('#tile-pd-disagg .tz-pdd-row').length")
    click(pg, '#tile-pd-disagg .seg-ctl button[data-v="split"]', 200)
    r1 = text(pg, '#tile-pd-disagg .tz-read'); bars1 = pg.evaluate("()=>document.querySelectorAll('#tile-pd-disagg .tz-pdd-row').length")
    check(f'{tag} tile: one GPU stalls 1.49 s; split, 14 ms; bars for 1 then 2 GPUs', Z(lang, '1.49 s', '1.49 秒') in r0 and Z(lang, '14 ms', '14 毫秒') in r1 and bars0 == 1 and bars1 == 2, (r0, r1, bars0, bars1))
    chip = text(pg, '#tile-pd-disagg .tz-kind')
    check(f'{tag} tile: the chip says Toy run, as the note does', chip == Z(lang, 'Toy run', '玩具实跑'), chip)
    hgt = pg.evaluate("()=>document.querySelector('#tile-pd-disagg .tz-pdd-vis').getBoundingClientRect().height")
    check(f'{tag} tile: the picture is at most 130 px tall', hgt <= 130, hgt)
    rules(pg, f'{tag} tile')
    pg.close()


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for (w, h) in SIZES:
        for lang in LANGS:
            guided(b, w, h, lang)
            sandbox(b, w, h, lang)
            challenge(b, w, h, lang)
            tile(b, w, h, lang)
    b.close()

check('no page errors', not errs, errs[:4])
check('text rules', not rule_hits, rule_hits[:8])
print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
