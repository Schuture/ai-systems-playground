"""Flow test for the multi-gpu-serving lab, in English and Chinese, at 1440 and 390 wide. Filters: en|zh, 1440|390
(python3 tests/multi-gpu-serving_flow.py zh 390).
- the guided parts in order; part 1 plays nothing; Next waits with the kit's why line where a part asks for a choice (parts 3, 5, 6, 7)
  and on the prediction (part 4);
- R3.20: entering part 2 plays its layer at medium speed; Play pauses and resumes; the speed control works; Back and Next again replay
  nothing (watched); Replay plays from the start; in part 3 a new split plays, the same split again and a split already watched replay
  nothing; parts 4 to 7 open on a watched picture and replay nothing; a new layout in parts 5 and 6 plays;
- the observations (TP 8 14.5 ms and 6.2 times faster than one GPU, TP 16 17.1 ms, the pipeline and replicas against one node,
  the MoE imbalance drawn on the stage with the 70B pictures hidden), part 8 links to the scenarios;
- every metric value fits its box (scrollWidth <= clientWidth);
- at 1440 the sticky stage fits the screen in every part (no scrolling inside it);
- Scenarios: six, each shows its insight; a new one plays, one already watched does not; the setup goes into the link; the hash round trip;
- the challenge (128 users, 8 or 16 GPUs): starts at TP 16 on 16 GPUs; a miss says why and offers Show the answer, which shows the
  brief chart; TP 8 x 2 stages on two nodes passes and is told another setup makes more tokens per GPU; one node of TP 8 is told it
  is the best; the exits, whose first question does not give the answer away;
- the mini result bar (dock) at 390 when the stage is off screen; the home tile (switch, sentence, bars, chip Formula, height);
- no horizontal page scroll, no page errors, the text rules (no “toy”, no company names, no two colons in one sentence, no
  “一次只能选 / 不会叠加 / 第 N 部分里”).
Run: python3 tests/multi-gpu-serving_flow.py [en|zh] [1440|390]"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'multi-gpu-serving').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/multi-gpu-serving.html', '--only', 'multi-gpu-serving'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'multi-gpu-serving' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'multi-gpu-serving.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'multi-gpu-serving' / 'flow-wrapped.html')
SH = P / 'shots' / 'multi-gpu-serving'; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_='multi-gpu-serving'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__mgs.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
cur = lambda pg: text(pg, '#mgs-rail .step.current')
next_dis = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#mgs-rail .step.current [data-nav=\"next\"]'); return b?b.disabled:null}")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#mgs-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#mgs-run .seg-ctl[data-key=\"spd\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")
stage_fits = lambda pg: pg.evaluate("()=>{const s=document.querySelector('#mgs-stage'); if(!s) return true; return getComputedStyle(s).position!=='sticky'||s.scrollHeight-s.clientHeight<=1}")


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait)


def finish(pg):
    """scrub the layer to its end, as a reader can with the progress bar"""
    pg.evaluate("()=>{const r=document.querySelector('#mgs-prog'); r.value='1000'; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true}));}")
    pg.wait_for_timeout(150)


def nav(pg, which):
    click(pg, f'#mgs-rail .step.current [data-nav="{which}"]', 350)


def mode(pg, m):
    click(pg, f'#mgs-app .modes button[data-mode="{m}"]', 400)


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
    for sel in ('#mgs-app', '#tile-multi-gpu-serving'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons: {h}')
    ov = overflow(pg); check(f'{where}: no horizontal page scroll', ov <= 0, ov)
    mv = pg.evaluate("()=>[...document.querySelectorAll('#mgs-metrics .metric .v')].filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+0.5).map(e=>e.textContent+' '+e.scrollWidth+'>'+e.clientWidth)")
    check(f'{where}: every metric value fits its box (S22)', not mv, mv)
    check(f'{where}: the sticky stage fits the screen', stage_fits(pg))


def guided(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    title = text(pg, '#mgs-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'How is one model served across 8 GPUs?', '一个模型怎么在 8 张卡上服务？'), title)
    s = state(pg); check(f'{tag} part 1: guided, nothing plays on the first part', s['mode'] == 'guided' and s['step'] == 0 and s['plays'] == 0 and not s['playing'], s)
    t1 = cur(pg); check(f'{tag} part 1: the weights do not fit one GPU (131.4 GiB), 69.2 ms per read', '131.4 GiB' in t1 and Z(lang, '69.2 ms', '69.2 毫秒') in t1, t1[:300])
    rules(pg, f'{tag} part 1')
    # part 2: a new picture plays at medium speed; pause, resume, speed; Back and Next replay nothing; Replay
    nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 2: entering plays one layer at medium speed (R3.9)', s['step'] == 1 and s['playing'] and s['plays'] == 1 and speed(pg) == 'med', (s, speed(pg)))
    check(f'{tag} part 2: the observation waits for the layer to play', not pg.evaluate("()=>{const o=document.querySelector('#mgs-rail [data-obs]'); return !!o&&!o.hidden}"))
    click(pg, '#mgs-play', 200); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} part 2: Play pauses the layer where it is', not s1['playing'] and s1['ph'] == s2['ph'] and s1['ph'] > 0, (s1['ph'], s2['ph']))
    click(pg, '#mgs-play', 300); s3 = state(pg)
    check(f'{tag} part 2: Play resumes from there', s3['playing'] and s3['ph'] >= s1['ph'] and s3['plays'] == 2, s3)
    click(pg, '#mgs-run .seg-ctl[data-key="spd"] button[data-v="slow"]', 150)
    check(f'{tag} part 2: the speed control switches to slow', speed(pg) == 'slow' and state(pg)['spd'] == 'slow', speed(pg))
    click(pg, '#mgs-run .seg-ctl[data-key="spd"] button[data-v="med"]', 150)
    finish(pg); t2 = cur(pg)
    check(f'{tag} part 2: the observation quotes 14.5 ms and 6.2 times against one GPU (S1)', Z(lang, 'A token takes 14.5 ms here, about 6.2 times faster than one GPU.', '这里每个 token 要 14.5 毫秒，比一张卡快约 6.2 倍。') in t2 and Z(lang, '90.5 ms', '90.5 毫秒') in t2, t2[-500:])
    rules(pg, f'{tag} part 2')
    p0 = state(pg)['plays']; nav(pg, 'back'); nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 2 again: Back and Next replay nothing (R3.20)', s['step'] == 1 and not s['playing'] and s['plays'] == p0, s)
    click(pg, '#mgs-replay', 200); s = state(pg)
    check(f'{tag} part 2: Replay plays the watched layer from the start', s['playing'] and s['plays'] == p0 + 1 and s['ph'] < 0.5, s)
    finish(pg)
    # part 3: Next waits for two splits; new plays, same or watched does not
    nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 3: opens on the watched TP 8 picture, nothing plays', not s['playing'], s)
    check(f'{tag} part 3: Next waits, with the why line', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try two ways to split to continue.', '试两种分法才能继续。')}, (next_dis(pg), why(pg)))
    p0 = state(pg)['plays']; click(pg, '#mgs-rail [data-lever="4"]'); s = state(pg)
    check(f'{tag} part 3: two TP 4 replicas play the new picture (R3.20)', s['playing'] and s['plays'] == p0 + 1 and s['p3'] == 4, s)
    finish(pg); click(pg, '#mgs-rail [data-lever="4"]'); s = state(pg)
    check(f'{tag} part 3: the same split again replays nothing (R3.20)', not s['playing'] and s['plays'] == p0 + 1, s)
    click(pg, '#mgs-rail [data-lever="8"]'); s = state(pg)
    check(f'{tag} part 3: TP 8, already watched, replays nothing (R3.20)', not s['playing'] and s['plays'] == p0 + 1 and next_dis(pg) is False, s)
    t3 = cur(pg); check(f'{tag} part 3: the observation compares 14.5 ms with 23.0 ms', Z(lang, 'One group of eight: 14.5 ms per output token, against 23.0 ms for two replicas of four GPUs.', '一个八张卡的组：每个输出 token 14.5 毫秒；两个四张卡的副本是 23.0 毫秒。') in t3, t3[-500:])
    click(pg, '#mgs-rail [data-lever="2"]'); t3 = cur(pg)
    check(f'{tag} part 3: four TP 2 replicas do not fit, in red', Z(lang, 'Does not fit', '放不下') in t3 and pg.evaluate("()=>!!document.querySelector('#mgs-rail [data-obs] .badline')"), t3[-300:])
    rules(pg, f'{tag} part 3')
    # part 4: the prediction
    nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 4: nothing plays (TP 8 watched)', not s['playing'], s)
    check(f'{tag} part 4: Next waits for an answer', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick an answer to continue.', '选一个答案才能继续。')}, why(pg))
    click(pg, '#mgs-rail [data-pred="2x"]'); t4 = cur(pg)
    check(f'{tag} part 4: an answer is saved and Next opens', next_dis(pg) is False and Z(lang, 'Saved.', '已记下') in t4, t4[-300:])
    rules(pg, f'{tag} part 4')
    # part 5: TP 16
    nav(pg, 'next'); s = state(pg); t5 = cur(pg)
    check(f'{tag} part 5: the verdict on a wrong answer, nothing plays', not s['playing'] and Z(lang, 'The answer: TP 16 is slower than TP 8.', '答案是：TP 16 反而比 TP 8 慢。') in t5, t5[:300])
    check(f'{tag} part 5: Next waits for TP 16', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try TP 16 across two nodes to continue.', '试一下跨两台机器的 TP 16 才能继续。')}, why(pg))
    p0 = state(pg)['plays']; click(pg, '#mgs-rail [data-lever="16"]'); s = state(pg); finish(pg); t5 = cur(pg)
    check(f'{tag} part 5: TP 16 plays and is slower: 17.1 ms, not halved 11.3 -> 7.0', s['plays'] == p0 + 1 and Z(lang, 'Across two nodes a token takes 17.1 ms, slower than 14.5 ms in one node.', '跨两台机器，每个 token 要 17.1 毫秒，比一台机器的 14.5 毫秒还慢。') in t5 and Z(lang, 'from 11.3 ms to 7.0 ms, not to half', '从 11.3 毫秒降到 7.0 毫秒，没能减半') in t5 and next_dis(pg) is False, t5[-600:])
    red = pg.evaluate("()=>document.querySelectorAll('#mgs-layer .mgs-seg.ar.bad').length")
    check(f'{tag} part 5: the merges are red in the layer', red == 2, red)
    nav(pg, 'back'); click(pg, '#mgs-rail [data-pred="slow"]'); nav(pg, 'next'); t5 = cur(pg)
    check(f'{tag} part 5: the right answer is told so', Z(lang, 'Your prediction was right', '你猜对了') in t5, t5[:200])
    pg.screenshot(path=str(SH / f'flow-{tag}-p5.png'))
    rules(pg, f'{tag} part 5')
    # part 6: pipeline and replicas
    nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 6: opens on the watched TP 16, nothing plays', not s['playing'], s)
    check(f'{tag} part 6: Next waits for both', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try the pipeline and the two replicas to continue.', '试一下流水线和两个副本才能继续。')}, why(pg))
    p0 = state(pg)['plays']; click(pg, '#mgs-rail [data-lever="pp"]'); s = state(pg); finish(pg)
    check(f'{tag} part 6: the pipeline plays', s['plays'] == p0 + 1, s)
    click(pg, '#mgs-rail [data-lever="rep"]'); finish(pg); t6 = cur(pg)
    check(f'{tag} part 6: pipeline 13.1 ms, replicas 13.0 ms, both a little faster than one node (S12)', Z(lang, 'The pipeline (13.1 ms) and the two replicas (13.0 ms) are about equally fast, and both beat one group of sixteen.', '流水线（13.1 毫秒）和两个副本（13.0 毫秒）差不多快，都比一个十六张卡的组快。') in t6 and Z(lang, 'Both are also a little faster than one node (14.5 ms)', '两种也都比一台机器（14.5 毫秒）快一点') in t6 and next_dis(pg) is False, t6[-500:])
    gl = pg.evaluate("()=>[...document.querySelectorAll('#mgs-gpus .mgs-bt')].map(e=>e.textContent)")
    check(f'{tag} part 6: the GPU picture names the two replicas', any(Z(lang, 'replica 2', '副本 2') in x or Z(lang, 'rep. 2', '副本 2') in x for x in gl), gl)
    rules(pg, f'{tag} part 6')
    # part 7: MoE
    p0 = state(pg)['plays']; nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 7: entering plays the MoE layer once, a new picture (R3.20)', s['playing'] and s['plays'] == p0 + 1, s)
    steps = pg.evaluate("()=>[...document.querySelectorAll('#mgs-ep-steps .mgs-st4')].map(e=>e.textContent)")
    check(f'{tag} part 7: the stage names the four steps of the layer', steps == [Z(lang, '① Route', '① 路由'), Z(lang, '② Dispatch (all-to-all)', '② 发出去（all-to-all）'), Z(lang, '③ Experts compute', '③ 专家计算'), Z(lang, '④ Combine (all-to-all)', '④ 收回来（all-to-all）')], steps)
    check(f'{tag} part 7: Next waits for another number of tokens', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try another number of tokens to continue.', '换一个 token 数才能继续。')}, why(pg))
    finish(pg); bars = pg.evaluate("()=>document.querySelectorAll('#mgs-stage .mgs-ep-col').length")
    check(f'{tag} part 7: the stage draws 8 GPUs of experts, the busiest in red (S13)', bars == 8 and pg.evaluate("()=>document.querySelectorAll('#mgs-stage .mgs-ep-pile.bad').length") == 1, bars)
    p0 = state(pg)['plays']; nav(pg, 'back'); nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 7 again: Back and Next replay nothing (R3.20)', not s['playing'] and s['plays'] == p0, s)
    st7 = text(pg, '#mgs-st-title'); hid = pg.evaluate("()=>['#mgs-gb','#mgs-lb','#mgs-metrics','#mgs-math'].map(s=>document.querySelector(s).hidden)")
    check(f'{tag} part 7: the stage title names the MoE layer; the 70B pictures, cards and math are hidden', 'DeepSeek' in st7 and 'MoE' in st7 and hid == [True]*4, (st7, hid))
    leg = text(pg, '#mgs-stage .mgs-leg')
    check(f'{tag} part 7: the legend says token–expert pairs, and pairs computed for the done part of a bar (S15)', Z(lang, '2 token–expert pairs on their way', '每个是 2 个 token–专家对') in leg and Z(lang, 'pairs computed', '已算完的对') in leg and '对数' not in leg, leg)
    pg.screenshot(path=str(SH / f'flow-{tag}-p7.png'))
    p0 = state(pg)['plays']; click(pg, '#mgs-rail .seg-ctl[data-key="p7"] button[data-v="256"]'); t7 = cur(pg); s = state(pg)
    check(f'{tag} part 7: 256 tokens, 1.07 times', Z(lang, 'the busiest GPU has 1.07 times the average work', '最忙的卡要做平均的 1.07 倍') in t7 and next_dis(pg) is False, t7[-500:])
    check(f'{tag} part 7: 256 tokens play their own picture (R3.20)', s['playing'] and s['plays'] == p0 + 1, s)
    finish(pg); p0 = state(pg)['plays']; click(pg, '#mgs-rail .seg-ctl[data-key="p7"] button[data-v="32"]'); s = state(pg)
    check(f'{tag} part 7: back to the watched 32 tokens, nothing replays (R3.20)', not s['playing'] and s['plays'] == p0, s)
    click(pg, '#mgs-rail .seg-ctl[data-key="p7"] button[data-v="256"]')
    rules(pg, f'{tag} part 7')
    # part 8
    nav(pg, 'next'); t8 = pg.evaluate("()=>document.querySelector('#mgs-rail .step.current').textContent")
    check(f'{tag} part 8: what to remember, misconceptions, real systems, scenarios', all(x in t8 for x in (Z(lang, 'Common misconceptions', '常见误解'), Z(lang, 'What real systems do', '真实系统怎么做'), 'vLLM', 'DeepEP', '--enable-expert-parallel'))
          and pg.evaluate("()=>!!document.querySelector('#mgs-rail .step.current [data-mode-go=\"sandbox\"]')"), t8[:200])
    rules(pg, f'{tag} part 8')
    pg.screenshot(path=str(SH / f'flow-{tag}-p8.png'))
    pg.close()


def sandbox(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'sandbox')
    pres = pg.evaluate("()=>[...document.querySelectorAll('#mgs-app [data-preset]')].map(e=>e.textContent.trim().slice(0,40))")
    check(f'{tag} scenarios: six', len(pres) == 6, pres)
    for i in range(6):
        p0 = state(pg)['plays']; click(pg, f'#mgs-app [data-preset="{i}"]', 300); s = state(pg)
        ins = text(pg, '#mgs-insight')
        # scenario 0 is the sandbox's own start, already on screen: nothing changes, nothing replays (R3.20)
        check(f'{tag} scenario {i}: {"the picture on screen does not replay" if i == 0 else "a new picture plays"} and its insight shows', s['plays'] == p0 + (0 if i == 0 else 1) and Z(lang, 'ms', '毫秒') in ins, (s['plays'], p0, ins[:120]))
        finish(pg)
        rules(pg, f'{tag} scenario {i}')
    p0 = state(pg)['plays']; click(pg, '#mgs-app [data-preset="2"]', 300); s = state(pg)
    check(f'{tag} scenarios: one already watched replays nothing (R3.20)', s['plays'] == p0 and not s['playing'], s)
    sw = pg.evaluate("()=>document.querySelectorAll('#mgs-sweep .mgs-sw-row').length")
    check(f'{tag} sandbox: the TP sweep shows 4 sizes', sw == 4, sw)
    hs = pg.evaluate('()=>location.hash')
    check(f'{tag} sandbox writes its settings into the link', hs == '#multi-gpu-serving~70b_16_16_1_32_4096', hs)
    click(pg, '#mgs-controls .seg-ctl[data-key="g"] button[data-v="8"]', 300); s = state(pg)
    check(f'{tag} settings: 8 GPUs bring TP 16 back to TP 8', s['g'] == 8 and s['tp'] == 8 and s['pp'] == 1, s)
    click(pg, '#mgs-controls .seg-ctl[data-key="tp"] button[data-v="16"]', 300); s = state(pg)
    check(f'{tag} settings: TP 16 takes 16 GPUs', s['g'] == 16 and s['tp'] == 16, s)
    if w == 390:
        pg.evaluate("()=>{const r=document.querySelector('#mgs-rail'); window.scrollTo(0, r.getBoundingClientRect().top+scrollY+r.offsetHeight-200)}"); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>{const d=document.querySelector('#dock'); if(!d) return null; const r=d.getBoundingClientRect(); return {vis:getComputedStyle(d).display!=='none'&&r.height>0&&r.bottom<=innerHeight+1, t:d.innerText}}")
        check(f'{tag}: the mini result bar shows the time per token when the stage is off screen', dk and dk['vis'] and Z(lang, 'ms', '毫秒') in dk['t'], dk)
    pg.close()
    pg = page(b, w, h, lang, hash_='multi-gpu-serving~70b8_16_8_2_128_1024'); s = state(pg)
    check(f'{tag} hash round trip: a link opens the sandbox with its settings', s['mode'] == 'sandbox' and s['m'] == '70b8' and s['g'] == 16 and s['tp'] == 8 and s['pp'] == 2 and s['u'] == 128 and s['ctx'] == 1024, s)
    rules(pg, f'{tag} hash')
    pg.close()


def challenge(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'challenge'); finish(pg); s = state(pg)
    check(f'{tag} challenge: starts at one TP 16 group on 16 GPUs for 128 users of 70B', s['mode'] == 'challenge' and s['tp'] == 16 and s['pp'] == 1 and s['g'] == 16 and s['u'] == 128, s)
    dis = pg.evaluate("()=>[...document.querySelectorAll('#mgs-rail .seg-ctl[data-key=\"chpp\"] button')].filter(b=>b.disabled).map(b=>b.dataset.v)")
    check(f'{tag} challenge: with TP 16 only 1 stage is possible', dis == ['2', '4', '8', '16'], dis)
    opts = {(o['g'], o['tp'], o['pp']): o for o in pg.evaluate("()=>window.__mgs.chOpts()")}
    tok = lambda k: f"{round(opts[k]['tps']):,}"
    mz = lambda k: Z(lang, f"{opts[k]['tpot']*1e3:.1f} ms", f"{opts[k]['tpot']*1e3:.1f} 毫秒")
    click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="4"]'); finish(pg)
    click(pg, '#mgs-app [data-act="submit"]', 350); r = text(pg, '#mgs-ch-res')
    check(f'{tag} challenge: 4 replicas of TP 4 on 16 GPUs miss, say why, offer the answer', Z(lang, f'Not met: {mz((16, 4, 1))} per output token.', f'没达标：每个输出 token {mz((16, 4, 1))}。') in r and Z(lang, 'Show the answer', '直接看答案') in r and not pg.query_selector('#mgs-ch-res .cmp'), r)
    click(pg, '#mgs-ch-res [data-act="reveal"]', 350); r = text(pg, '#mgs-ch-res')
    rows = pg.evaluate("()=>document.querySelectorAll('#mgs-ch-res .cmp-row, #mgs-ch-res [class*=cmp] li, #mgs-ch-res tr').length")
    check(f'{tag} challenge: Show the answer shows the chart (your pick, the initial setup) and why', rows >= 4 and Z(lang, 'your pick', '你的选择') in r and Z(lang, 'initial setup', '初始设置') in r and Z(lang, f'One node is enough: one group of eight meets the goal ({mz((8, 8, 1))})', f'一台机器就够了：一个八张卡的组就能达标（{mz((8, 8, 1))}）') in r and Z(lang, 'miss the goal or do not fit', '没达到目标或放不下') in r, (rows, r[:400]))
    click(pg, '#mgs-app .seg-ctl[data-key="chtp"] button[data-v="8"]'); click(pg, '#mgs-app .seg-ctl[data-key="chpp"] button[data-v="2"]'); finish(pg); r = text(pg, '#mgs-ch-res')
    check(f'{tag} challenge: TP 8 x 2 stages on two nodes passes; one node makes more per GPU', Z(lang, f'Met: {mz((16, 8, 2))} per output token.', f'达标：每个输出 token {mz((16, 8, 2))}。') in r and Z(lang, f'But another setup gets more tokens per second from each GPU (the chart): {tok((8, 8, 1))} against your {tok((16, 8, 2))}.', f'不过另一种分法每张卡每秒出的 token 更多（见图）：{tok((8, 8, 1))} 个，你的是 {tok((16, 8, 2))} 个。') in r, r[:400])
    click(pg, '#mgs-app .seg-ctl[data-key="chg"] button[data-v="8"]'); finish(pg); s = state(pg); r = text(pg, '#mgs-ch-res')
    check(f'{tag} challenge: 8 GPUs keep TP 8 and drop to one stage', (s['g'], s['tp'], s['pp']) == (8, 8, 1), s)
    dis = pg.evaluate("()=>[...document.querySelectorAll('#mgs-rail .seg-ctl[data-key=\"chtp\"] button')].filter(b=>b.disabled).map(b=>b.dataset.v)")
    check(f'{tag} challenge: on 8 GPUs TP 16 is not offered', dis == ['16'], dis)
    check(f'{tag} challenge: one node of TP 8 is the best', Z(lang, f'Met: {mz((8, 8, 1))} per output token, and the most tokens per second per GPU ({tok((8, 8, 1))}).', f'达标：每个输出 token {mz((8, 8, 1))}，而且每张卡每秒出的 token 最多（{tok((8, 8, 1))} 个）。') in r, r[:300])
    ex = pg.evaluate("()=>[...document.querySelectorAll('#mgs-app .exits a, #mgs-app .exits button')].length")
    check(f'{tag} challenge: the exits follow the result', ex >= 3, ex)
    exq = pg.evaluate("()=>document.querySelector('#mgs-rail [data-exits]').textContent")
    check(f'{tag} challenge: the first exit question does not hand out the answer (S25)', Z(lang, 'Why do they not halve the time per output token?', '为什么每个输出 token 的耗时没有减半？') in exq and Z(lang, 'usually kept inside one node', '一般不跨机器') not in exq, exq[:300])
    pg.screenshot(path=str(SH / f'flow-{tag}-challenge.png'), full_page=False)
    rules(pg, f'{tag} challenge')
    pg.close()


def tile(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang, hash_='map')
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="inference"], #domStrip [data-world="inference"]')].find(e=>e.offsetParent);
      if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="inference"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
    t = pg.query_selector('#tile-multi-gpu-serving')
    check(f'{tag} tile: the multi-gpu-serving tile is on the inference branch', t is not None)
    if not t: pg.close(); return
    t.scroll_into_view_if_needed(); pg.wait_for_timeout(150)
    r0 = text(pg, '#tile-multi-gpu-serving .tz-read')
    click(pg, '#tile-multi-gpu-serving .seg-ctl button[data-v="4"]', 200); r1 = text(pg, '#tile-multi-gpu-serving .tz-read')
    click(pg, '#tile-multi-gpu-serving .seg-ctl button[data-v="2"]', 200); r2 = text(pg, '#tile-multi-gpu-serving .tz-read')
    boxes = pg.evaluate("()=>[document.querySelectorAll('#tile-multi-gpu-serving .tz-mgs-box').length, document.querySelectorAll('#tile-multi-gpu-serving .tz-mgs-box.bad').length]")
    check(f'{tag} tile: TP 8 14.5 ms; two TP 4 23.0 ms; four TP 2 do not fit, 8 GPUs in red', Z(lang, '14.5 ms', '14.5 毫秒') in r0 and Z(lang, '23.0 ms', '23.0 毫秒') in r1 and Z(lang, 'does not fit', '放不下') in r2 and boxes == [8, 8], (r0, r1, r2, boxes))
    words = [len(x.split()) for x in (r0, r1, r2)] if lang == 'en' else [0]
    check(f'{tag} tile: the sentence is at most 25 words', max(words) <= 25, words)
    chip = text(pg, '#tile-multi-gpu-serving .tz-kind')
    check(f'{tag} tile: the chip says Formula, as the note does', chip == Z(lang, 'Formula', '公式计算'), chip)
    hgt = pg.evaluate("()=>document.querySelector('#tile-multi-gpu-serving .tz-mgs-vis').getBoundingClientRect().height")
    check(f'{tag} tile: the picture is at most 130 px tall', hgt <= 130, hgt)
    t.screenshot(path=str(SH / f'tile-{tag}.png'))
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
