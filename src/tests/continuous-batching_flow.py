"""Flow test for the continuous batching lab (#continuous-batching).

Walks every mode and every guided step at 1440×900 and 390×844 in English and Chinese and checks: no page errors,
no horizontal overflow, next-step gating, single-choice levers and presets that follow the settings, formula-term
tooltips and jumps (and the fixed-value message in the challenge), the three challenge feedback tiers and the
comparison chart, language switch keeping state, the hash round trip, keyboard access to the schedule chart,
the mini result bar at 390, reduced motion, dark mode, the other layout widths (1280, 900, 720) and the home tile.
Copy pass v2 (specs/copy-pass-v2.md) adds, over every string a learner can meet in the lab and on its tile (visible text,
tooltips, aria labels, popovers), in both languages: no "·" joining clauses, no 起点 / "starting point" as the baseline
name, no 控件 / "Controls" as the panel name, no sentence about the mini result bar or about the page's own behavior
(hover, tap, "takes about 6 seconds", ...), a space between Chinese and Latin text; and the playback speed control
(slow / medium / fast, time-based, medium by default, "Show me" plays fast).
Round 7 adds: the ▲▼ badges only after the learner's own change and computed from the shown values, the challenge's
dashed line hidden until the first submit, "Show me" leaving the learner's speed setting alone, the stage badge
("Running" / "Paused", empty before a run), and the new wording (CB-A r7, R7-B, SITE-A r7).
Screenshots go to shots/continuous-batching/. Run: python3 tests/continuous-batching_flow.py"""
import pathlib, subprocess, sys, json, re
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
subprocess.run([sys.executable, str(P / 'build.py'), 'out/continuous-batching.html', '--only', 'continuous-batching'], check=True, capture_output=True)
(P / 'out/wrapped-continuous-batching.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                                                      + (P / 'out/continuous-batching.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out/wrapped-continuous-batching.html')
SH = P / 'shots/continuous-batching'; SH.mkdir(parents=True, exist_ok=True)
fails, log, errs = [], [], []

def check(name, ok, info=''):
    log.append(('OK   ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

def page(b, w, h, lang='en', dark=False, reduced=False, hash_='continuous-batching'):
    ctx = b.new_context(viewport={'width': w, 'height': h}, color_scheme='dark' if dark else 'light', reduced_motion='reduce' if reduced else 'no-preference')
    pg = open_drawers(ctx.new_page()); tag = f'{w}/{lang}'
    pg.on('pageerror', lambda e: errs.append(f'{tag}: {e}'))
    pg.on('console', lambda m: errs.append(f'{tag} console: {m.text}') if m.type == 'error' and 'ERR_TUNNEL' not in m.text and 'fonts.g' not in m.text and 'net::' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash={json.dumps(hash_)}}}"); pg.wait_for_timeout(350)
    return pg

ev = lambda pg, js, arg=None: pg.evaluate(js, arg)
over = lambda pg: ev(pg, "()=>document.documentElement.scrollWidth-innerWidth")
nextdis = lambda pg: ev(pg, "()=>{const b=document.querySelector('#cb-rail [data-nav=next]'); return b?b.disabled:null}")
obs = lambda pg: ev(pg, "()=>{const o=document.querySelector('#cb-rail [data-obs]'); return o&&!o.hidden?o.innerText:''}")
tipt = lambda pg: ev(pg, "()=>{const t=document.getElementById('tip'); return t.hidden?'':t.innerText}")
state = lambda pg: ev(pg, "()=>__cb.state()")
def go(pg, i): ev(pg, f"()=>document.querySelector('#cb-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(160)
def nxt(pg): ev(pg, "()=>document.querySelector('#cb-rail [data-nav=next]').click()"); pg.wait_for_timeout(160)
def setr(pg, id_, v): ev(pg, "([i,v])=>{const e=document.getElementById(i); e.value=v; e.dispatchEvent(new Event('input',{bubbles:true}))}", [id_, v]); pg.wait_for_timeout(150)
def click(pg, sel): pg.locator(sel).first.scroll_into_view_if_needed(); pg.click(sel); pg.wait_for_timeout(150)
def shot(pg, name, full=False): pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)
def visible(pg, sel): return ev(pg, "s=>{const e=document.querySelector(s); if(!e) return false; const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&getComputedStyle(e).display!=='none'}", sel)

Z = lambda lang, en, zh: zh if lang == 'zh' else en
# the dashed line's label must stay inside the chart (review CB-A #13)
ghost_in = lambda pg: ev(pg, "()=>{const p=document.getElementById('cb-plot').getBoundingClientRect(), s=document.querySelector('#cb-plot .cb-ghost span'); if(!s) return true; const q=s.getBoundingClientRect(); return q.left>=p.left-0.5&&q.right<=p.right+0.5}")
# the stage's state: chart cut, status badge and its tooltip, whole-run cards, the dashed line and its legend item
stage = lambda pg: ev(pg, """()=>({cut:getComputedStyle(document.getElementById('cb-plot')).getPropertyValue('--cut').trim(), status:document.getElementById('cb-status').textContent,
  tip:document.getElementById('cb-status').dataset.tipx||'', stHidden:document.getElementById('cb-status').hidden, pending:document.getElementById('cb-metrics').classList.contains('cb-pending'),
  values:[...document.querySelectorAll('#cb-metrics .metric .v')].map(e=>e.textContent),
  mhead:document.querySelector('#view-dyn .cb-mhead .eyebrow').textContent, ghost:__cb.ghost(), lgGhost:!document.getElementById('cb-lg-ghost').hidden,
  lgGhostText:document.querySelector('#cb-lg-ghost span').textContent, lgHead:!document.getElementById('cb-lg-head').hidden, clock:document.getElementById('cb-clock').textContent})""")

# the badge while a run is shown only in part says only whether it plays (the clock line has the time and the counts)
RUNNING = {'en': 'Running', 'zh': '运行中'}; PAUSED = {'en': 'Paused', 'zh': '已暂停'}
deltas = lambda pg: ev(pg, "()=>[...document.querySelectorAll('#cb-metrics .metric .d')].map(e=>e.textContent)")
# a badge by what it says (direction, number, unit), not by the kit's exact wording, which is being reworded
def dparse(x):
    if not x: return None
    m = re.search(r'([\d,]+(?:\.\d+)?)', x)
    return (x[0], float(m.group(1).replace(',', '')), 'pts' if ('pt' in x or '百分点' in x) else ('%' if '%' in x else 'x'))
dsays = lambda pg: [dparse(x) for x in deltas(pg)]
PENDING = {'en': 'shown after the run', 'zh': '跑完后显示'}

def walk(b, w, h, lang):
    tag = f'{lang} {w}'
    pg = page(b, w, h, lang)
    maxo = [over(pg)]
    title = pg.inner_text('#view-dyn .labtitle')
    check(f'{tag}: title', title == Z(lang, 'Why does continuous batching raise throughput?', 'continuous batching 为什么能提高吞吐？'), title)
    check(f'{tag}: mode description (guided cards are “parts”; “step” is the engine step)', Z(lang, 'Seven short parts. Predict first, see where static batching leaves the GPU waiting', '一共 7 个部分。先预测，再看 static batching 让 GPU 在哪里空等') in pg.evaluate("()=>{const b=document.querySelector('#view-dyn [data-flow-nav] button[aria-pressed=true]'); return b?(b.dataset.tipx||''):''}"))
    # nothing shows the finished run before it is played; no comparison line before both policies ran
    st0 = stage(pg)
    check(f'{tag}: part 1 starts with an empty chart, no badge before the run (the clock says 0 s), the whole-run cards wait, no deltas',
          st0['cut'] != '0%' and st0['status'] == '' and st0['stHidden'] and st0['pending'] and st0['tip'] == '' and deltas(pg) == [''] * 4
          and st0['values'] == [PENDING[lang]] * 4 and st0['mhead'] == Z(lang, 'Whole run (all 48 requests)', '整次运行（48 个请求）') and not st0['ghost'] and not st0['lgGhost'], str(st0))
    c1 = pg.inner_text('#cb-rail .step.current')
    check(f'{tag}: part 1 has no chips row (it used “slot” before the paragraph defines it)', not ev(pg, "()=>!!document.querySelector('#cb-rail .step.current .chips')"))
    check(f'{tag}: part 1 says memory caps the slots at 22 and the server starts with 8, defines the slot and the step, and names the question', all(x in c1 for x in Z(lang, ['memory caps how many requests can run at once: 22 here. The server starts with a limit of 8', 'each running request takes one slot', 'One pass of the model is a step', 'how soon does its slot go to a waiting request?'],
          ['显存决定了最多能同时跑几个请求：这里最多 22 个。服务端先设成最多同时跑 8 个', '每个在跑的请求占一个 slot', '模型每跑一次前向叫一步', '它的 slot 什么时候交给排队的请求？'])), c1[:160])
    check(f'{tag}: part 1 gives the answers’ spread in words, no derived range or mean (R6.20)', Z(lang, 'Each has a 512-token prompt, and the answers range from a few tokens to a few hundred.', '每个请求的 prompt 是 512 个 token；回答从几个 token 到几百个 token 不等。') in c1
          and '455' not in c1 and '128.9' not in c1, c1[:200])
    check(f'{tag}: the chart caption names the rows and the bars', pg.inner_text('#cb-graph .cb-glabel') == Z(lang, 'Each row is a slot and each bar a request, from its start to its last token.', '每一行是一个 slot，每一条是一个请求：从它开始，到它的最后一个 token。'))
    shot(pg, f'{lang}_{w}_1_mission')
    # step 2: predict gates Next
    nxt(pg); check(f'{tag}: predict gates Next', nextdis(pg) is True)
    click(pg, '#cb-rail [data-pred="mid"]')
    card = pg.inner_text('#cb-rail .step.current')
    check(f'{tag}: prediction is saved (still changeable) and enables Next', nextdis(pg) is False and ev(pg, "()=>document.querySelector('#cb-rail [data-pred=mid]').getAttribute('aria-pressed')") == 'true'
          and Z(lang, 'Saved. You can still change it until part 4 shows the result.', '已记下。第 4 部分给出结果之前，还可以改。') in card and 'Locked' not in card and '锁定' not in card)
    check(f'{tag}: the two policies in one sentence each; options cover 1.5–3 and more than 3', Z(lang, 'the next 8 start only when the longest answer of the 8 is done', '下一批 8 个才开始') in card
          and Z(lang, 'More than 3 times', '3 倍以上') in card and Z(lang, 'How many times static batching’s tokens per second does continuous batching make?', 'continuous batching 每秒生成的 token 数，是 static batching 的多少倍？') in card)
    st1 = stage(pg)
    check(f'{tag}: while predicting, no result and no comparison anywhere (badge, tooltip, legend)', st1['pending'] and not st1['ghost'] and not st1['lgGhost'] and st1['tip'] == '', st1['tip'][:80])
    shot(pg, f'{lang}_{w}_2_predict'); maxo.append(over(pg))
    # step 3: run static batching
    nxt(pg)
    check(f'{tag}: step 3 starts with an empty chart and Next disabled', nextdis(pg) is True and ev(pg, "()=>getComputedStyle(document.getElementById('cb-plot')).getPropertyValue('--cut').trim()") != '0%')
    click(pg, '#cb-graph [data-act="next"]')
    clk = pg.inner_text('#cb-clock')
    check(f'{tag}: “Next finish” moves the playhead to the first finished request; the clock is a sentence', clk == Z(lang, 'At 1.7 s: 7 requests running, 40 waiting, 1 done.', '1.7 s：7 个请求在跑，40 个在排队，1 个已完成。'), clk)
    tp1 = ev(pg, "()=>__cb.reqTip(2)")
    check(f'{tag}: a running request’s tooltip says its final length and nothing after the moment shown', Z(lang, 'its answer will have', '回答最终会有') in tp1 and Z(lang, 'finished at', '写完') not in tp1
          and Z(lang, 'by 1.7 s it has generated', '到 1.7 s 已生成') in tp1, tp1)
    setr(pg, 'cb-scrub', 0)
    if lang == 'en' or w == 390:
        click(pg, '#cb-rail [data-act="play"]'); pg.wait_for_timeout(2200)
        clk = pg.inner_text('#cb-clock'); sm = stage(pg)
        check(f'{tag}: playback runs (playhead and clock), at medium speed by default', visible(pg, '#cb-head') and Z(lang, 'running', '在跑') in clk and not obs(pg) and ev(pg, "()=>__cb.play().speed") == 'med', clk)
        check(f'{tag}: during the run the badge says it runs, the clock line has the time and counts, the whole-run cards wait', sm['status'] == RUNNING[lang] and sm['values'] == [PENDING[lang]] * 4
              and sm['pending'] and sm['lgHead'] and sm['tip'] == '' and Z(lang, 'done.', '已完成。') in sm['clock'], f"{sm['status']} | {sm['clock']}")
        if w == 390:
            dk = ev(pg, "()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText})")
            check(f'{tag}: mini result bar during the run says how many are done, not the result', (not dk['shown']) or (Z(lang, 'of 48 requests done', '48 个请求完成了') in dk['text'] and Z(lang, 'tokens/s', '每秒') not in dk['text']), dk['text'].replace('\n', ' '))
        shot(pg, f'{lang}_{w}_3_playing')
        ev(pg, "()=>document.querySelector('#cb-rail .seg-ctl[data-key=speed] button[data-v=fast]').click()")   # the rest of the run at fast speed
        for _ in range(80):
            if obs(pg): break
            pg.wait_for_timeout(200)
        pg.wait_for_timeout(250)
        if w == 390:
            dn = ev(pg, "()=>{const d=document.getElementById('cb-done'); return d.hidden?'':d.textContent}")
            check(f'{tag}: at 390 the finished run offers a way back to the step card', dn.replace('\xa0', ' ') == Z(lang, 'Run finished: the conclusion is in part 3 ↓', '跑完了，结论在第 3 部分 ↓'), dn)   # phones (m2): the card is under the stage
            shot(pg, f'{lang}_{w}_3_done_line')
            click(pg, '#cb-done'); pg.wait_for_timeout(900)
            top = ev(pg, "()=>{const r=document.querySelector('#cb-rail [data-obs]').getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom), innerHeight]}")
            check(f'{tag}: clicking it brings the result back into view', top[0] >= 50 and top[0] < top[2] - 100 and ev(pg, "()=>document.getElementById('cb-done').hidden"), str(top))
    else:
        setr(pg, 'cb-scrub', 1000)
    o = obs(pg); se = stage(pg)
    check(f'{tag}: part 3 observation after the run, Next enabled: one key number (46%), the idle slot in words; the finish is on the badge (R6.20)', bool(o) and nextdis(pg) is False and '46%' in o and '38.7 s' not in o
          and Z(lang, 'the shortest answer was done long before the longest one', '最短的回答比最长的早很久就写完了') in o, o[:90].replace('\n', ' '))
    check(f'{tag}: after the run the badge gives the finish, the cards show the run, still no comparison line or badge tooltip', se['status'] == Z(lang, 'All 48 requests done at 38.7 s', '48 个请求在 38.7 s 全部完成') and not se['pending']
          and not se['ghost'] and se['tip'] == '' and se['values'][0].startswith('160'), str(se))
    notes = ev(pg, "()=>[...document.querySelectorAll('#cb-metrics .metric .sub')].map(e=>e.textContent)")
    check(f'{tag}: the metric notes are sentences with their referents; the throughput note’s time reproduces the throughput (6,189 ÷ 38.73)', notes == Z(lang, ['The 48 answers, 6,189 tokens in all, took 38.73 s.', 'Slots sat idle while requests were waiting for 48% of all slot time.', '90% of the requests are done within 35.6 s of arriving.', 'That is 55 tokens per second for each request.'],
          ['48 个回答共 6,189 个 token，用了 38.73 s。', 'slot 空着、同时还有请求在排队的时间，占全部 slot 时间的 48%。', '90% 的请求在到达后 35.6 s 内写完。', '相当于每个请求每秒收到 55 个 token。']), str(notes))
    check(f'{tag}: one name for the busy metric, and its tooltip says what it is a share of', pg.inner_text('#cb-metrics [data-k=util] .k') == Z(lang, 'Slots busy', 'slot 忙碌比例')
          and ev(pg, "()=>document.querySelector('#cb-metrics [data-k=util]').dataset.tipx") == Z(lang, 'Slots busy: of all the time of the 8 slots over the whole run, the share spent on a request (prefill or decode).', 'slot 忙碌比例：整次运行里，8 个 slot 的全部时间中，花在处理请求（prefill 或 decode）上的比例。'))
    check(f'{tag}: the run finishing (not a change of setup) shows no ▲▼', deltas(pg) == [''] * 4, str(deltas(pg)))
    shot(pg, f'{lang}_{w}_3_static'); maxo.append(over(pg))
    # step 4: levers
    nxt(pg)
    pressed = lambda: ev(pg, "()=>[...document.querySelectorAll('#cb-rail [data-lever]')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.lever)")
    check(f'{tag}: part 4 starts at static, Next disabled, no comparison line yet', pressed() == ['static'] and nextdis(pg) is True and not stage(pg)['ghost'])
    c4 = pg.inner_text('#cb-rail .step.current')
    check(f'{tag}: part 4 says what changes, not how the buttons work; the static option is the initial setup', Z(lang, 'only the batching policy changes', '只换调度方式') in c4
          and Z(lang, 'at a time', '一次选') not in c4 and Z(lang, 'initial setup', '初始设置') in c4, c4[:160].replace('\n', ' | '))
    click(pg, '#cb-rail [data-lever="cont"]')
    o = obs(pg)
    check(f'{tag}: continuous lever pressed, the result states the ratio to the baseline (the two runs are on the options)', pressed() == ['cont'] and Z(lang, '1.9 times as many tokens per second as the initial setup', '是初始设置的 1.9 倍') in o and nextdis(pg) is False, o[:100].replace('\n', ' '))
    dl = deltas(pg)
    check(f'{tag}: the learner’s change shows ▲▼ computed from the shown values (160 → 297, 46% → 87%, 19.7 → 10.5 s, 18.2 → 22.6 ms)', [dparse(x) for x in dl] == [('▲', 86, '%'), ('▲', 41, 'pts'), ('▼', 47, '%'), ('▲', 24, '%')], str(dl))
    check(f'{tag}: the cost paragraph says why the join step is long, in words', Z(lang, 'its whole prompt runs through the model (its prefill) in the same step as the others’ next tokens, so that step takes over four times as long', '除了其他请求的下一个 token，还要把它整个 prompt 过一遍模型（prefill），所以这一步的耗时是没有新请求加入时的四倍多') in o, o[-160:])
    check(f'{tag}: prediction feedback', Z(lang, 'was right', '是对的') in o)
    sg = stage(pg)
    check(f'{tag}: after continuous ran, the dashed line and its legend item show static, and the badge tooltip compares', sg['ghost'] and sg['ghost']['pol'] == 'static' and sg['lgGhost'] and sg['lgGhostText'] == Z(lang, 'when static batching finishes', 'static batching 做完的时刻')
          and sg['tip'] == Z(lang, 'With static batching and the same 8 slots, the same 48 requests are done at 38.7 s (the dashed line).', '同样这 48 个请求，用同样 8 个 slot 的 static batching 在 38.7 s 做完（图中的虚线）。'), str(sg))
    click(pg, '#cb-rail [data-lever="static"]'); g1 = ghost_in(pg); click(pg, '#cb-rail [data-lever="cont"]')
    check(f'{tag}: the dashed line’s label stays inside the chart (continuous line mid-chart, static line at the end)', g1 and ghost_in(pg))
    click(pg, '#cb-rail [data-lever="static"]')
    o2 = obs(pg); bump = ev(pg, "()=>document.querySelector('#cb-rail [data-obs]').classList.contains('bump')")
    check(f'{tag}: switching back changes the text and blinks', o2 != o and pressed() == ['static'] and bump, o2[:80].replace('\n', ' '))
    res = ev(pg, "()=>[...document.querySelectorAll('#cb-rail [data-lever] .lv-r')].map(e=>e.textContent)")
    check(f'{tag}: tried options keep their result, with the unit', res == Z(lang, ['160 tokens/s', '297 tokens/s'], ['160 token/s', '297 token/s']), str(res))
    setr(pg, 'cb-ctl-B-in', 9)
    o3 = obs(pg)
    check(f'{tag}: a manual change leaves no option pressed and says so', pressed() == [] and Z(lang, 'matches neither option', '不属于上面任何一个选项') in o3)
    click(pg, '#cb-rail [data-lever="cont"]')
    shot(pg, f'{lang}_{w}_4_continuous'); maxo.append(over(pg))
    # step 5: explain, term tooltip and jump
    nxt(pg)
    c5 = pg.inner_text('#cb-rail .step.current')
    check(f'{tag}: part 5 is two paragraphs and the step chart: no formula box, no third paragraph, the ≈100 threshold stays in Show the math', obs(pg) == '' and deltas(pg) == [''] * 4
          and all(x in c5 for x in Z(lang, ['In every step the GPU reads all of OPT-13B’s weights from memory once, and that read takes most of the step.', 'yet makes 8 times as many tokens', '19.2 ms per step, 8 tokens', 'How long is a step with more requests in it?'],
                                     ['每一步，GPU 都要把 OPT-13B 的全部权重从显存读一遍，这占了一步的大部分时间。', '产出的 token 却是它的 8 倍', '一步 19.2 ms，产出 8 个 token', '一步里的请求越多，一步要多久？']))
          and '16.53' not in c5
          and Z(lang, 'the mean the answer lengths are drawn with', '抽取回答长度时设定的均值') not in c5
          and '416' not in c5 and '0.1648' not in c5 and Z(lang, 'where these numbers come from', '这些数怎么来的') not in c5.lower(), c5[:160].replace('\n', ' | '))
    check(f'{tag}: step-time chart in the card', visible(pg, '#cb-rail .cb-stc'))
    ev(pg, "()=>document.getElementById('cb-math').open=true"); pg.wait_for_timeout(100)
    t = pg.locator('#cb-mathbody .t[data-ctl="B"]').first
    t.scroll_into_view_if_needed(); t.hover(); pg.wait_for_timeout(150)
    tp = tipt(pg)
    check(f'{tag}: term tooltip is a sentence with a subject and ends with where a click lands', Z(lang, 'The slots are the most requests the GPU runs in one step.', 'slot 数是 GPU 每一步最多同时跑的请求数。') in tp and Z(lang, 'Click to jump to “Slots” under Settings.', '点击跳到“自由调参”里的“slot 数”。') in tp, tp.replace('\n', ' | '))
    t.click(); pg.wait_for_timeout(700)
    check(f'{tag}: click jumps to the slots control', ev(pg, "()=>document.activeElement&&document.activeElement.id") == 'cb-ctl-B-in')
    shot(pg, f'{lang}_{w}_5_explain'); maxo.append(over(pg))
    # step 6: grow the batch
    go(pg, 5)
    check(f'{tag}: part 6 Next disabled until the slider moves', nextdis(pg) is True)
    pg.locator('#cb-ic-B').scroll_into_view_if_needed(); pg.wait_for_timeout(200)
    setr(pg, 'cb-ic-B-in', 16)
    o = obs(pg)
    check(f'{tag}: 16 slots observed against 8 as a ratio (386 and 297 on the cards and the dashed line), panel synced', Z(lang, '1.3 times as many tokens per second as with 8', '是 8 个 slot 时的 1.3 倍') in o and nextdis(pg) is False
          and pg.inner_text('#cb-metrics [data-k=thr] .v').startswith('386') and ev(pg, "()=>document.getElementById('cb-ctl-B-in').value") == '16', o[:100].replace('\n', ' '))
    check(f'{tag}: the time per output token in words from the displayed values, and the two reasons as “First … Second …”; no throughput sentence, no cap paragraph below 22 slots', Z(lang, 'but each output token comes a little slower', '但每个输出 token 慢了一点') in o
          and all(x in o for x in Z(lang, ['Each output token gets slower for two reasons. First, every step reads the KV cache of more requests. Second, other requests join more often: each joining prompt’s prefill makes its step several times longer, and every running request waits through it.'],
                                          ['每个输出 token 变慢有两个原因。一是每一步要读更多请求的 KV cache。二是别的请求加入得更频繁：每个新 prompt 的 prefill 都让那一步长好几倍，在跑的请求都得等它。']))
          and Z(lang, 'Throughput grows more slowly', '吞吐涨得比 slot 数慢') not in o and Z(lang, 'is the limit', '是上限') not in o, o[:400].replace('\n', ' '))
    dl = deltas(pg)
    check(f'{tag}: 8 → 16 slots: ▲▼ from the shown values (297 → 386, 87% → 73%, 10.5 → 7.8 s, 22.6 → 27.4 ms)', [dparse(x) for x in dl] == [('▲', 30, '%'), ('▼', 14, 'pts'), ('▼', 26, '%'), ('▲', 21, '%')], str(dl))
    setr(pg, 'cb-ic-B-in', 9); o9 = obs(pg)
    check(f'{tag}: 9 slots: 1.1 times, and 23.0 ms (on the card, one decimal) is about as fast as 22.6 ms', Z(lang, 'With 9 slots, continuous batching makes 1.1 times as many tokens per second as with 8, and each output token comes about as fast.', '9 个 slot 时，continuous batching 每秒生成的 token 是 8 个 slot 时的 1.1 倍，而且每个输出 token 几乎一样快。') in o9
          and pg.inner_text('#cb-metrics [data-k=tpot] .v').startswith('23.0'), o9[:300].replace('\n', ' '))
    setr(pg, 'cb-ic-B-in', 22); o22 = obs(pg)
    check(f'{tag}: at 22 slots the cap paragraph says why the slider stops', Z(lang, '22 slots is the limit: memory holds the KV cache of only 22 requests (how that is computed).', '22 个 slot 是上限：显存只放得下 22 个请求的 KV cache（怎么算的）。') in o22, o22[-200:].replace('\n', ' '))
    setr(pg, 'cb-ic-B-in', 16)
    sg = stage(pg)
    check(f'{tag}: the dashed line now marks the 8-slot continuous run the text compares with', sg['ghost'] and sg['ghost']['kind'] == 'b8' and sg['lgGhostText'] == Z(lang, 'when 8 slots finish', '8 个 slot 做完的时刻'), str(sg))
    if w == 390:
        pg.wait_for_timeout(250)
        dock = ev(pg, "()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText})")
        check(f'{tag}: mini result bar visible while the slider is in view, in words', dock['shown'] and Z(lang, '386 tokens/s', '每秒 386 个 token') in dock['text'] and Z(lang, 'slots busy 73% of the time', 'slot 有 73% 的时间在干活') in dock['text'], dock['text'].replace('\n', ' '))
    shot(pg, f'{lang}_{w}_6_grow'); maxo.append(over(pg))
    # step 7
    nxt(pg)
    t7 = pg.inner_text('#cb-rail .step.current')
    check(f'{tag}: what to remember: short bullets that state the pattern, not the 1.9 times of part 4 (G2, the owner 2026-10-08), two common misconceptions in words (the latency says its setup); no 16.8 ms, no Anyscale', t7.count('✗') == 2 and '57.1' not in t7 and '16.8' not in t7 and 'Anyscale' not in t7 and '54%' not in t7
          and Z(lang, 'its mean latency with continuous batching on 8 slots is under a fifth of that with one request at a time', '用 8 个 slot 的 continuous batching，平均延迟不到一次只跑一个请求时的五分之一') in t7
          and Z(lang, 'so the same slots make more tokens per second. The more the answer lengths differ, the bigger the gain.', '所以同样多的 slot，每秒能生成更多的 token。回答的长短差得越多，多出来的越多。') in t7
          and Z(lang, '1.9 times', '1.9 倍') not in t7 and Z(lang, 'part 4', '第 4 部分') not in t7 and Z(lang, 'common misconceptions', '常见误解') in t7.lower() and deltas(pg) == [''] * 4, t7[:200].replace('\n', ' | '))
    check(f'{tag}: no related-labs list between the forward button and Back (R14-site 1-2)', ev(pg, "()=>document.querySelectorAll('#cb-rail .cb-rel').length") == 0 and '(规划中)' not in t7)
    shot(pg, f'{lang}_{w}_7_remember'); maxo.append(over(pg))
    # challenge
    # R3.12: guided ends in Scenarios, whose card ends with the way to the challenge
    click(pg, '#cb-rail [data-mode-go="sandbox"]'); pg.wait_for_timeout(200)
    check(f'{tag}: guided ends in Scenarios, and the scenarios card ends with “Take the challenge”', pg.get_attribute('#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])', 'aria-pressed') == 'true' and pg.query_selector('#cb-rail [data-mode-go="challenge"]') is not None)
    click(pg, '#cb-rail [data-mode-go="challenge"]'); pg.wait_for_timeout(200)
    h3 = pg.inner_text('#cb-rail .rh')
    check(f'{tag}: challenge goal written in full', h3.replace('\xa0', ' ') == Z(lang, 'Reach 330 tokens per second on one A100 40GB, with at most 25 ms per output token on average', '让一张 A100 40GB 每秒生成 330 个 token，平均每个输出 token 不超过 25 ms'), h3)
    rl = pg.inner_text('#cb-rail')
    check(f'{tag}: the goals are explained; no sentence about what the submit button will do', Z(lang, 'Why these goals', '为什么是这两个目标') in rl and Z(lang, 'Submitting tells you', '提交后会告诉你') not in rl)
    check(f'{tag}: only the allowed controls, the rest hidden with a note',
          visible(pg, '#cb-ic-pol') and visible(pg, '#cb-ic-B') and not visible(pg, '#cb-ctl-cap') and not visible(pg, '#cb-adv') and not visible(pg, '#cb-ctl-setup') and visible(pg, '#cb-lock'))
    check(f'{tag}: live progress as sentences', pg.inner_text('#cb-g1v') == '160' and '18.2 ms' in pg.inner_text('#cb-g2v') and pg.inner_text('#cb-g1').replace('\xa0', ' ') == Z(lang, 'Throughput is now 160 tokens per second; the goal is at least 330.', '吞吐现在是每秒 160 个 token，目标是至少 330 个。'), pg.inner_text('#cb-g1'))
    check(f'{tag}: the challenge badge says which goal is missing', pg.inner_text('#cb-status') == Z(lang, 'Under 330 tokens per second', '吞吐还不到每秒 330 个 token'), pg.inner_text('#cb-status'))
    sc0 = stage(pg)
    check(f'{tag}: before the first submit the challenge shows no dashed line, no label, no legend item, and no ▲▼ from the previous mode', not sc0['ghost'] and not sc0['lgGhost'] and not visible(pg, '#cb-plot .cb-ghost') and deltas(pg) == [''] * 4, str(sc0)[:120])
    ev(pg, "()=>document.getElementById('cb-math').open=true"); pg.wait_for_timeout(100)
    t = pg.locator('#cb-mathbody .t[data-ctl="eff"]').first
    t.scroll_into_view_if_needed(); t.hover(); pg.wait_for_timeout(150)
    tp = tipt(pg)
    check(f'{tag}: a fixed term says its value during the challenge', Z(lang, 'Fixed during the challenge', '挑战中固定不变') in tp and '50%' in tp, tp.replace('\n', ' | '))
    t.click(); pg.wait_for_timeout(300)
    check(f'{tag}: clicking it explains instead of jumping', Z(lang, 'Fixed during the challenge', '挑战中固定不变') in tipt(pg) and ev(pg, "()=>document.activeElement&&document.activeElement.id") != 'cb-ctl-eff-in')
    click(pg, '#cb-rail [data-act="submit"]')
    r = pg.inner_text('#cb-ch-res')
    check(f'{tag}: tier 3 (not met) says which goal is missing and gives a hint, no answer', Z(lang, 'Not there yet: throughput is under 330 tokens per second.', '还没达到目标：吞吐还不到每秒 330 个 token。') in r and Z(lang, 'Hint', '提示') in r and not visible(pg, '#cb-ch-res .cmp'), r[:80].replace('\n', ' '))
    sg = stage(pg)
    check(f'{tag}: a wrong static submit draws no continuous-batching line (it would answer the hint)', not sg['ghost'] and not sg['lgGhost'] and not visible(pg, '#cb-plot .cb-ghost'), str(sg)[:120])
    check(f'{tag}: after submitting, no note about the page’s behavior', pg.inner_text('#cb-rail .cb-subwrap').strip() == '')
    check(f'{tag}: the reveal button says “Show the answer”', pg.inner_text('#cb-ch-res [data-act="reveal"]') == Z(lang, 'Show the answer', '直接看答案'), pg.inner_text('#cb-ch-res [data-act="reveal"]'))
    click(pg, '#cb-rail [data-act="reveal"]')
    sg = stage(pg)
    check(f'{tag}: after “Show the answer” the other policy’s line and legend item appear', sg['ghost'] and sg['ghost']['pol'] == 'cont' and sg['lgGhost'], str(sg)[:120])
    tags = ev(pg, "()=>[...document.querySelectorAll('#cb-ch-res .cmp-tag,#cb-ch-res .cmp-you')].map(e=>e.textContent)")
    check(f'{tag}: Show the answer draws the chart with the initial setup and your pick', visible(pg, '#cb-ch-res .cmp') and Z(lang, 'initial setup', '初始设置') in tags and Z(lang, 'your pick', '你的选择') in tags, str(tags[:4]))
    # the owner's review of batch 8: a brief chart, the 2 settings that meet both goals and the initial setup (= the pick); the note counts the other 41 of 44
    rws = ev(pg, "()=>[...document.querySelectorAll('#cb-ch-res .cmp-row')].map(r=>({ok:r.classList.contains('ok'),me:r.classList.contains('me'),tag:!!r.querySelector('.cmp-tag'),t:r.querySelector('.cmp-lab span').textContent}))")
    note = pg.inner_text('#cb-ch-res .cmp-note') if visible(pg, '#cb-ch-res .cmp-note') else ''
    check(f'{tag}: the brief chart holds only the 2 passing settings and the initial setup, and the note counts the other 41', len(rws) == 3 and all(r['ok'] or r['me'] or r['tag'] for r in rws) and sum(r['ok'] for r in rws) == 2
          and note == Z(lang, 'The other 41 settings miss at least one goal and are not shown.', '其余 41 种设置至少有一个目标没达到，图里没有列出。') and Z(lang, 'over 25 ms', '超过 25 ms') not in str(tags), (rws, note))
    ttl = pg.inner_text('#cb-ch-res .cmp-title')
    check(f'{tag}: the chart title names the quantity, the conditions and the goal, not “by batching policy and slots”', 'A100 40GB' in ttl and '330' in ttl and Z(lang, 'by batching policy', '各调度方式') not in ttl, ttl)
    xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
    check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
    leg = pg.inner_text('#cb-ch-res .cmp-leg')
    check(f'{tag}: the chart legend says what green means', Z(lang, 'meets both goals', '两项都达标') in leg and Z(lang, 'misses at least one', '至少一项未达标') in leg and ev(pg, "()=>document.querySelectorAll('#cb-ch-res .cmp-row.ok').length") == 2, leg.replace('\n', ' '))
    click(pg, '#cb-rail .seg-ctl[data-key="pol"] button[data-v="cont"]'); setr(pg, 'cb-ic-B-in', 11)
    r = pg.inner_text('#cb-ch-res')
    check(f'{tag}: tier 2 (met, a smaller change exists)', Z(lang, 'Both goals met', '两个目标都达到了') in r and Z(lang, 'Fewer slots would do', '其实 slot 少一些也够') in r and '10' in r)
    setr(pg, 'cb-ic-B-in', 10)
    r = pg.inner_text('#cb-ch-res')
    check(f'{tag}: tier 1 (met and minimal)', Z(lang, 'changed as little as possible: in this simulation, with fewer slots continuous batching stays under the throughput goal', '你的改动已经是最少的：在这个模拟里，slot 再少，continuous batching 的吞吐就达不到目标') in r, r[:90].replace('\n', ' '))
    check(f'{tag}: the chart paragraph says which options pass and why the others fail, in words (the chart and Show the math hold the numbers); the sensitivity moved to Fidelity',
          all(x in r for x in [Z(lang, 'Only continuous batching with 10 or 11 slots meets both goals: with fewer slots the throughput stays under the goal, and with more each output token takes too long.', '只有 continuous batching 用 10 或 11 个 slot 时两个目标都达到：slot 再少，吞吐达不到目标；再多，每个输出 token 又太慢。'),
                               Z(lang, 'Static batching falls short even with every slot that memory allows.', 'static batching 就算用满显存放得下的所有 slot 也达不到目标。'), Z(lang, 'every step reads more KV cache', '每一步要读的 KV cache 越多')])
          and '88.9 ms' not in r and Z(lang, '90% of the datasheet', '90% 读取') not in r, r[-500:].replace('\n', ' '))
    if w == 390:
        ev(pg, "()=>document.getElementById('cb-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(350)
        dk = ev(pg, "()=>({shown:!document.getElementById('dock').hidden, text:document.getElementById('dock').innerText})")
        fits = ev(pg, "()=>{const r=document.getElementById('dkRead'); return r.scrollWidth<=r.clientWidth+1 && r.getBoundingClientRect().right<=innerWidth}")
        check(f'{tag}: challenge mini result bar reads both values with their units, and fits', dk['shown'] and Z(lang, '338 tokens/s, 23.7 ms per output token', '每秒 338 个 token，每个输出 token 23.7 ms') in dk['text'] and fits, dk['text'].replace('\n', ' '))
        shot(pg, f'{lang}_{w}_8b_challenge_dock')
    check(f'{tag}: goal status turns green and says so', ev(pg, "()=>document.getElementById('cb-status').className") == 'status ok' and pg.inner_text('#cb-status') == Z(lang, 'Both goals met', '两个目标都达到了'))
    shot(pg, f'{lang}_{w}_8_challenge_met', full=(w == 390))
    setr(pg, 'cb-ic-B-in', 16)
    r = pg.inner_text('#cb-ch-res').replace('\xa0', ' ')
    check(f'{tag}: over the 25 ms limit: not met, try fewer slots, with the reason in words; the badge names the missed goal', Z(lang, 'Not there yet: each output token takes over 25 ms.', '还没达到目标：每个输出 token 超过了 25 ms。') in r and Z(lang, 'Hint: the throughput is enough. With more slots', '提示：吞吐够了。slot 越多') in r and Z(lang, 'Try fewer slots', '试试少几个 slot') in r and Z(lang, 'so each output token gets slower', '所以每个输出 token 越来越慢') in r
          and pg.inner_text('#cb-status') == Z(lang, 'Over 25 ms per output token', '每个输出 token 超过 25 ms'))
    maxo.append(over(pg))
    # scenarios: presets follow the settings
    click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
    ev(pg, "()=>window.scrollTo(0,0)")
    click(pg, '#cb-rail [data-preset="1"]')
    pp = lambda: ev(pg, "()=>[...document.querySelectorAll('#cb-rail [data-preset]')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.dataset.preset)")
    rs = pg.inner_text('#cb-sb-res')
    check(f'{tag}: preset pressed and named in the result, which says why the gain is small', pp() == ['1'] and Z(lang, 'In the “Classification and tagging” scenario', '在“分类和打标签”这个场景里') in rs and Z(lang, 'The answers are all short', '回答都很短') in rs, rs[:80].replace('\n', ' '))
    sc = pg.inner_text('#cb-rail .railcard')
    check(f'{tag}: the scenario card is “Pick a scenario”, opens with a lead-in that says what to pick and what to watch (G1, the owner 2026-10-08), plain one-line descriptions with the cap as a cap', pg.inner_text('#cb-rail .rh') == Z(lang, 'Pick a scenario', '选一个应用场景')
          and Z(lang, 'Pick a scenario and see how many more tokens per second continuous batching makes than static batching. A scenario sets the answer lengths', '选一个场景，看看 continuous batching 每秒生成的 token 是 static batching 的几倍。每个场景定下回答长度') in sc and Z(lang, 'do not stack', '不会叠加') not in sc and all('·' not in x for x in ev(pg, "()=>[...document.querySelectorAll('#cb-rail [data-preset] span')].map(e=>e.textContent)"))
          and Z(lang, 'capped at 1,536', '上限 1,536 个') in sc and '最长 1,536' not in sc)
    onscreen = []
    for k in range(4):
        click(pg, f'#cb-rail [data-preset="{k}"]'); pg.wait_for_timeout(250)
        onscreen.append(ev(pg, "()=>{const r=document.getElementById('cb-sb-res').getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom), innerHeight]}"))
    check(f'{tag}: after a scenario click the result box is on screen (it sits above the list)', all(55 <= t_ and t_ < h_ - 100 for t_, b_, h_ in onscreen), str(onscreen))
    click(pg, '#cb-rail [data-preset="1"]')
    click(pg, '#cb-rail [data-preset="2"]')
    rs2 = pg.inner_text('#cb-sb-res')
    check(f'{tag}: another preset replaces it and the text changes; live traffic says who keeps up', pp() == ['2'] and rs2 != rs and Z(lang, 'neither policy can finish before the last one arrives', '两种方式都不可能在最后一个请求到达之前做完') in rs2 and Z(lang, 'keeps up', '跟得上') in rs2, rs2[:300].replace('\n', ' '))
    if lang == 'zh': check(f'{tag}: no stray space before 不到', '平均等不到 0.1 s 就开始' in rs2 and '等 不到' not in rs2, rs2[:200].replace('\n', ' '))
    setr(pg, 'cb-ctl-B-in', 8); click(pg, '#cb-rail [data-preset="3"]')
    rs3 = pg.inner_text('#cb-sb-res')
    gh3, st3 = ev(pg, "()=>__cb.ghost()"), pg.inner_text('#cb-status')
    check(f'{tag}: 4 per second with 8 slots: continuous cannot keep up, no “follows the arrival rate”; each paragraph’s one number is the mean wait (1.9 s, 12.0 s); the finishes on the badge (21.7 s) and the dashed line (41.6 s)', Z(lang, 'cannot keep up', '跟不上') in rs3 and Z(lang, 'follows the arrival rate', '取决于请求到达的速度') not in rs3
          and all(x in rs3 for x in ['1.9 s', '12.0 s']) and not any(x in rs3 for x in ['21.7 s', '41.6 s', '8.2 s', '28.1 s'])
          and st3 == Z(lang, 'All 48 requests done at 21.7 s', '48 个请求在 21.7 s 全部完成') and gh3 and gh3['pol'] == 'static' and f"{gh3['mk']:.1f}" == '41.6', rs3[:300].replace('\n', ' ') + ' | ' + st3)
    click(pg, '#cb-rail [data-preset="2"]')
    click(pg, '#cb-ctl-cap button[data-v="512"]')
    check(f'{tag}: changing a control un-presses the preset and says so', pp() == [] and Z(lang, 'match none of the scenarios', '不属于任何一个场景') in pg.inner_text('#cb-sb-res'))
    setr(pg, 'cb-ctl-B-in', 12)
    shot(pg, f'{lang}_{w}_9_explore'); maxo.append(over(pg))
    # language switch keeps state; hash round trip
    before = state(pg); h0 = ev(pg, "()=>location.hash")
    other = 'en' if lang == 'zh' else 'zh'
    click(pg, f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(250)
    after = state(pg)
    check(f'{tag}: language switch keeps mode and settings', before == after and ev(pg, "()=>location.hash") == h0, f'{before} -> {after}')
    check(f'{tag}: the notes close the page after a re-render (owner, 2026-10-11 19:28 UTC)', visible(pg, '#continuous-batching-fid') and not pg.query_selector('#view-dyn [data-pop="continuous-batching-fid"]'))
    click(pg, f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    pg2 = page(b, w, h, lang, hash_=h0[1:])
    s2 = state(pg2)
    check(f'{tag}: hash round trip restores the setup in the scenarios mode', s2['mode'] == 'sandbox' and all(s2[k] == before[k] for k in ('pol', 'B', 'cap', 'rate', 'set', 'eff')), f'{h0} -> {s2}')
    pg2.context.close()
    # keyboard on the schedule chart
    ev(pg, "()=>document.querySelector('.cb-req[tabindex=\"0\"]').focus()"); pg.wait_for_timeout(120)
    pg.keyboard.press('ArrowRight'); pg.wait_for_timeout(150)
    k = ev(pg, "()=>document.activeElement.dataset.i")
    check(f'{tag}: arrow keys move between requests and show their tooltip', k == '1' and Z(lang, 'Request 2', '请求 2') in tipt(pg), f'{k} {tipt(pg)[:40]}')
    # dock at 390: scroll to the controls, then click it back
    if w == 390:
        ev(pg, "()=>document.getElementById('cb-controls').scrollIntoView({block:'center'})"); pg.wait_for_timeout(350)
        shown = ev(pg, "()=>!document.getElementById('dock').hidden")
        shot(pg, f'{lang}_{w}_10_dock')
        pg.click('#dock'); pg.wait_for_timeout(900)
        check(f'{tag}: dock appears off-chart and brings the chart back', shown and ev(pg, "()=>{const r=document.getElementById('cb-graph').getBoundingClientRect(); return r.top<innerHeight&&r.bottom>60}"))
    # interview drill: reveal shows a worked answer, a new question hides it again
    q0 = pg.inner_text('#cb-drill-q'); click(pg, '#view-dyn [data-act="drill-reveal"]')
    a0 = ev(pg, "()=>{const a=document.getElementById('cb-drill-a'); return a.hidden?'':a.innerText}")
    # the worked answer's H_B is a typeset formula since 2026-10-09 (H1): its letters are MathML, read from the formula itself
    h0 = ev(pg, "()=>[...document.querySelectorAll('#cb-drill-a math.tex mi')].some(m=>m.textContent==='H')")
    click(pg, '#view-dyn [data-act="drill-new"]')
    check(f'{tag}: interview drill reveals a worked answer and changes the question', h0 and '36.8%' in a0 and '39.0%' in a0 and pg.inner_text('#cb-drill-q') != q0 and ev(pg, "()=>document.getElementById('cb-drill-a').hidden"), a0[:80])
    check(f'{tag}: the drill says why its 39.0% (steps) differs from the 46% Slots busy (time)', Z(lang, 'The same run’s Slots busy is 46% because it counts time: in the 6 long prefill steps (about 0.7 s each) all 8 slots are busy.', '同一次运行的 slot 忙碌比例是 46%：它按时间算，而 6 个较长的 prefill 步（每步约 0.7 s）里 8 个 slot 都在忙。') in a0, a0[-160:])
    click(pg, '#view-dyn [data-act="drill-new"]'); click(pg, '#view-dyn [data-act="drill-reveal"]')
    a2 = ev(pg, "()=>document.getElementById('cb-drill-a').innerText")
    h2 = ev(pg, "()=>[...document.querySelectorAll('#cb-drill-a math.tex')].filter(m=>m.querySelector('mfrac')).map(m=>m.textContent.replace(/\\s/g,'')).join(' | ')")
    check(f'{tag}: drill at 2 slots writes H₂ = 1 + 1/2 (typeset)', 'H2=1+12' in h2, h2[:120])
    maxo.append(over(pg))
    check(f'{tag}: no horizontal overflow on any screen', max(maxo) <= 0, str(maxo))
    pg.context.close()

# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'continuous-batching.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick a scenario and see how many more tokens per second continuous batching makes than static batching.", "选一个场景，看看 continuous batching 每秒生成的 token 是 static batching 的几倍。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["That is a little under ${ms(r1.t)}, a step for one request with ${AVG_CTX} tokens in its KV cache", "这比 KV cache 里有 ${AVG_CTX} 个 token 时 1 个请求的一步（${ms(r1.t)}）略短", "so the same slots make more tokens per second. The more the answer lengths differ, the bigger the gain.", "所以同样多的 slot，每秒能生成更多的 token。回答的长短差得越多，多出来的越多。"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["step in part 5", "比第 5 部分里 1 个请求的一步", "so the same 8 slots make ${times(k)}", "所以同样 8 个 slot，每秒生成的 token 是 static batching 的"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

with sync_playwright() as p:
    b = p.chromium.launch()
    for (w, h) in [(1440, 900), (390, 844)]:
        for lang in ['en', 'zh']:
            walk(b, w, h, lang)
    # other widths: every mode, both languages, no overflow
    for w in (1280, 900, 720):
        for lang in ('en', 'zh'):
            pg = page(b, w, 900, lang); o = [over(pg)]
            for i in range(7): go(pg, i); o.append(over(pg))
            for m in ('sandbox', 'challenge'):
                ev(pg, "()=>window.scrollTo(0,0)"); click(pg, f'#view-dyn .modes button[data-mode="{m}"]'); o.append(over(pg))
            click(pg, '#cb-rail [data-act="submit"]'); click(pg, '#cb-rail [data-act="reveal"]'); o.append(over(pg))
            check(f'{lang} {w}: no horizontal overflow in any mode', max(o) <= 0, str(o))
            if lang == 'zh': shot(pg, f'zh_{w}_challenge')
            pg.context.close()
    # ---------- copy pass v2: rules over every string a learner can meet, in both languages ----------
    # visible text of the lab (every part before and after its action, mid-run, the three modes, the settings panel with
    # its sections open, both popovers, the challenge before and after submitting, connect with the drill answer) and of
    # the home tile, plus every tooltip, aria label and alt text found there
    GRAB = """(sel)=>{const r=document.querySelector(sel); if(!r) return ''; const out=[r.innerText];
      r.querySelectorAll('[data-tipx],[aria-label],[title]').forEach(e=>{['data-tipx','aria-label','title'].forEach(a=>{const v=e.getAttribute(a); if(v) out.push(v);});});
      r.querySelectorAll('.pop').forEach(e=>out.push(e.textContent));
      return out.join('\\n');}"""
    TERMS = "()=>__cb.terms()"                                   # every formula-term tooltip with its click action
    BANNED = {'en': ['starting point', 'Controls', 'mini result bar', 'mini bar', 'hover', 'Hover', ' tap ', 'Tap ', 'takes about 6 seconds', 'Feedback now updates', 'Submitting tells you', 'Pick one policy at a time', 'grey until', 'greyed',
                     'Misconceptions checked', 'multiply throughput', 'Guide and tasks', 'planned for', 'in this model,', 'for each user', 'Settings panel'],
              'zh': ['起点', '控件', '迷你结果条', '结果条', '悬停', '点按', '鼠标', '大约 6 秒', '反馈会随时更新', '提交后会告诉你', '一次选一种', '显示为灰色',
                     '已破除的误解', '拉高几倍', '引导和任务', 'KV cache 规划', '里规划', 'token 规划', '持有', '在这个模型里', '每个用户', '“自由调参”面板', '最长 1,536']}
    for lang in ('en', 'zh'):
        pg = page(b, 1440, 900, lang); texts = []
        grab = lambda: texts.append(ev(pg, GRAB, '#view-dyn'))
        grab()
        for i in range(7):
            go(pg, i); grab()
            if i == 1: click(pg, '#cb-rail [data-pred="mid"]'); grab()
            if i == 2:
                setr(pg, 'cb-scrub', 300); grab(); ev(pg, "()=>document.querySelectorAll('.cb-req').forEach(e=>e.dispatchEvent(new MouseEvent('mouseover',{bubbles:true})))"); grab()
                setr(pg, 'cb-scrub', 1000); grab()
            if i == 3: click(pg, '#cb-rail [data-lever="cont"]'); grab()
            if i == 4: ev(pg, "()=>document.getElementById('cb-math').open=true"); grab()
            if i == 5: setr(pg, 'cb-ic-B-in', 16); grab(); setr(pg, 'cb-ic-B-in', 4); grab()
        ev(pg, "()=>window.scrollTo(0,0)"); click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])'); grab()
        for k in range(4): click(pg, f'#cb-rail [data-preset="{k}"]'); grab()
        setr(pg, 'cb-ctl-B-in', 8); click(pg, '#cb-rail [data-preset="3"]'); grab()
        click(pg, '#cb-ctl-cap button[data-v="512"]'); grab()
        ev(pg, "()=>document.querySelectorAll('#cb-controls details').forEach(d=>d.open=true)"); grab()
        ev(pg, "()=>window.scrollTo(0,0)"); click(pg, '#view-dyn .modes button[data-mode="challenge"]'); grab()
        click(pg, '#cb-rail [data-act="submit"]'); grab()
        click(pg, '#cb-rail .seg-ctl[data-key="pol"] button[data-v="cont"]'); grab()
        setr(pg, 'cb-ic-B-in', 12); grab(); click(pg, '#cb-rail [data-act="reveal"]'); grab()
        for v in (11, 10, 16): setr(pg, 'cb-ic-B-in', v); grab()
        click(pg, '#view-dyn [data-act="drill-reveal"]'); grab()
        ev(pg, "()=>document.getElementById('cb-math').open=true"); texts.append(ev(pg, TERMS))
        ev(pg, "()=>{location.hash='map'}"); pg.wait_for_timeout(350)
        texts.append(ev(pg, GRAB, '#tile-continuous-batching'))
        click(pg, '#tile-continuous-batching .seg-ctl button[data-v="cont"]'); texts.append(ev(pg, GRAB, '#tile-continuous-batching'))
        allt = '\n'.join(texts)
        dots = sorted({l.strip() for l in allt.split('\n') if '·' in l})
        check(f'copy {lang}: no “·” joining clauses anywhere in the lab or on its tile (R6.14)', not dots, str(dots[:3]))
        hits = [x for x in BANNED[lang] if x in allt]
        check(f'copy {lang}: no 起点 / “starting point”, no 控件 / “Controls”, no sentence about the mini result bar or the page’s own behavior (R6.17, R2.1, R3.4)', not hits, str(hits))
        check(f'copy {lang}: the baseline is called {Z(lang, "“initial setup”", "“初始设置”")} and the panel {Z(lang, "“Settings”", "“自由调参”")}', Z(lang, 'initial setup', '初始设置') in allt and Z(lang, 'Settings', '“自由调参”') in allt)
        texts2 = []
        # sentences with two or more colons, in the lab's own text (the code sample is left out)
        COLON = """(sel)=>{const r=document.querySelector(sel); if(!r) return ''; const out=[];
          r.querySelectorAll('p,li,.ctl-note,.meter-sub,.cb-sub2,.bar-label,.cb-clock,.status,.goal,.tz-read,.tz-note,.tz-cb-cap,h3,.lv-d,.preset span,.cb-stc-title').forEach(e=>{if(!e.closest('[data-endnotes],pre')) out.push(e.innerText);});
          r.querySelectorAll('[data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx')));
          return out.join('\\n');}"""
        pg2 = page(b, 1440, 900, lang)
        for i in range(7):
            go(pg2, i)
            if i == 3: click(pg2, '#cb-rail [data-lever="cont"]')
            if i == 5: setr(pg2, 'cb-ic-B-in', 16)
            texts2.append(ev(pg2, COLON, '#view-dyn'))
        ev(pg2, "()=>window.scrollTo(0,0)"); click(pg2, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
        for k in range(4): click(pg2, f'#cb-rail [data-preset="{k}"]'); texts2.append(ev(pg2, COLON, '#view-dyn'))
        ev(pg2, "()=>window.scrollTo(0,0)"); click(pg2, '#view-dyn .modes button[data-mode="challenge"]'); click(pg2, '#cb-rail [data-act="submit"]'); click(pg2, '#cb-rail [data-act="reveal"]')
        for v in (12, 10): setr(pg2, 'cb-ic-B-in', v); texts2.append(ev(pg2, COLON, '#view-dyn'))
        click(pg2, '#view-dyn [data-act="drill-reveal"]'); texts2.append(ev(pg2, COLON, '#view-dyn')); texts2.append(ev(pg2, TERMS))
        pg2.context.close()
        two = set()
        for block in texts2:
            for line in block.split('\n'):
                for sent in re.split(r'(?<=[。？！])|(?<=[.?!])\s', line):
                    if sent.count('：') + len(re.findall(r':\s', sent)) >= 2: two.add(sent.strip()[:90])
        check(f'copy {lang}: no sentence with two colons (R6.7)', not two, str(sorted(two)[:3]))
        if lang == 'zh':
            glued = sorted({allt[max(0, m.start() - 8):m.end() + 8].replace('\n', ' ') for m in re.finditer(r'[A-Za-z0-9%][\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z0-9]', allt)})
            check('copy zh: a space between Chinese and Latin text or numbers everywhere', not glued, str(glued[:4]))
        n_chars = len(allt)
        log.append(f'     copy {lang}: {n_chars:,} characters checked')
        pg.context.close()

    # ---------- playback speed (R3.9): slow / medium / fast, time-based, medium by default, the same control in the card and the stage ----------
    pg = page(b, 1440, 900, 'en')
    go(pg, 2)
    segs = ev(pg, "()=>[...document.querySelectorAll('.seg-ctl[data-key=speed]')].map(g=>[...g.querySelectorAll('button')].map(x=>x.textContent+(x.getAttribute('aria-pressed')==='true'?'*':'')).join('/'))")
    check('speed: a Slow / Medium / Fast control next to Play in part 3 and in the stage, Medium pressed', segs == ['Slow/Medium*/Fast', 'Slow/Medium*/Fast'], str(segs))
    # R3.20 (owner, 2026-10-04): nothing replays the run on its own; Play is its replay control, and its tooltip says so
    pt = ev(pg, "()=>[...document.querySelectorAll('[data-act=play]')].map(x=>x.getAttribute('data-tipx')||'')")
    check('Play: in the card and in the stage, its tooltip says it starts the run again from 0 s once the whole run is drawn (R3.20)', len(pt) == 2 and all(x == 'Plays the run from the moment shown. When the whole run is drawn, Play starts it again from 0 s.' for x in pt), str(pt))
    def rate(v):
        ev(pg, f"()=>document.querySelector('#cb-graph .seg-ctl[data-key=speed] button[data-v={v}]').click()")
        setr(pg, 'cb-scrub', 0); ev(pg, "()=>document.querySelector('#cb-graph [data-act=play]').click()"); pg.wait_for_timeout(400)
        a = ev(pg, "()=>[__cb.play().t, performance.now()]"); pg.wait_for_timeout(1500); z = ev(pg, "()=>[__cb.play().t, performance.now()]")
        ev(pg, "()=>document.querySelector('#cb-graph [data-act=play]').click()"); pg.wait_for_timeout(100)
        return (z[0] - a[0]) / ((z[1] - a[1]) / 1000)
    rs = {v: rate(v) for v in ('slow', 'med', 'fast')}
    check('speed: seconds of the run per real second ≈ 0.5 / 1.5 / 5 (time-based)', abs(rs['slow'] - 0.5) < 0.12 and abs(rs['med'] - 1.5) < 0.3 and abs(rs['fast'] - 5) < 1.0, str({k: round(v, 2) for k, v in rs.items()}))
    same = ev(pg, "()=>[...document.querySelectorAll('.seg-ctl[data-key=speed] button[aria-pressed=true]')].map(x=>x.dataset.v)")
    check('speed: the card’s and the stage’s controls stay in step', same == ['fast', 'fast'], str(same))
    ev(pg, "()=>document.querySelector('#cb-rail .seg-ctl[data-key=speed] button[data-v=slow]').click()"); setr(pg, 'cb-scrub', 0)
    ev(pg, "()=>document.querySelector('#cb-rail [data-nav=showme]').click()"); pg.wait_for_timeout(200)
    pl = ev(pg, "()=>__cb.play()")
    check('speed: “Show me” plays fast once, and the learner’s setting stays Slow', pl['playing'] and pl['rate'] == 5 and pl['speed'] == 'slow'
          and ev(pg, "()=>[...document.querySelectorAll('.seg-ctl[data-key=speed] button[aria-pressed=true]')].map(x=>x.dataset.v)") == ['slow', 'slow'], str(pl))
    for _ in range(60):
        if not ev(pg, "()=>__cb.play().playing"): break
        pg.wait_for_timeout(200)
    go(pg, 3); click(pg, '#cb-rail [data-lever="cont"]'); ev(pg, "()=>document.querySelector('#cb-graph [data-act=play]').click()"); pg.wait_for_timeout(300)
    pl = ev(pg, "()=>__cb.play()")
    check('speed: after “Show me”, the next Play runs at the learner’s speed', pl['playing'] and pl['rate'] == 0 and pl['speed'] == 'slow', str(pl))
    tipsp = ev(pg, "()=>document.querySelector('#cb-graph .cb-speed').dataset.tipx")
    check('speed: its tooltip is a sentence with the three rates, naming both kinds of second', tipsp == 'Slow plays 0.5 s of the run in each real second, Medium 1.5 s and Fast 5 s.', tipsp)
    pg.context.close()

    # reduced motion: play jumps to the end
    pg = page(b, 1440, 900, 'en', reduced=True)
    go(pg, 2); click(pg, '#cb-rail [data-act="play"]'); pg.wait_for_timeout(150)
    check('reduced motion: Play shows the end state at once', bool(obs(pg)) and nextdis(pg) is False)
    pg.context.close()
    # R3.20 (owner, 2026-10-04): a run already watched to the end elsewhere (here in Scenarios) is drawn whole when part 3 shows it
    # again, and part 3's gate counts it; a run not watched yet still starts at 0 s and waits for Play
    def watch_in_scenarios(pg):
        ev(pg, "()=>document.querySelector('#cb-graph .seg-ctl[data-key=speed] button[data-v=fast]').click()")
        ev(pg, "()=>document.querySelector('#cb-graph [data-act=play]').click()"); pg.wait_for_timeout(300)
        for _ in range(80):
            if not ev(pg, "()=>__cb.play().playing"): break
            pg.wait_for_timeout(200)
    pg = page(b, 1440, 900, 'en')
    ev(pg, "()=>{location.hash='continuous-batching~cont.12.1536.0.1.50'}"); pg.wait_for_timeout(350)
    watch_in_scenarios(pg)
    click(pg, '#view-dyn .modes button[data-mode="guided"]'); go(pg, 2)
    pl = ev(pg, "()=>__cb.play()")
    check('R3.20 part 3: after another run was watched in Scenarios, part 3’s run still starts at 0 s and Next waits for it', pl['t'] == 0 and not pl['playing'] and nextdis(pg) is True and not obs(pg), str(pl))
    click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
    st = state(pg)
    check('R3.20 Scenarios: part 3’s run (static, 8 slots) drawn whole, nothing playing', (st['pol'], st['B'], st['mode']) == ('static', 8, 'sandbox') and ev(pg, "()=>__cb.play().t") is None, str(st))
    watch_in_scenarios(pg)
    click(pg, '#view-dyn .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
    pl = ev(pg, "()=>__cb.play()")
    check('R3.20 part 3: the same run already watched to the end in Scenarios is drawn whole at once, nothing plays, and Next is open', state(pg)['step'] == 2 and pl['t'] is None and not pl['playing'] and nextdis(pg) is False and bool(obs(pg)), str(pl))
    pg.context.close()
    # a shared setup link opened while the guided chart is still empty shows the whole run
    pg = page(b, 1440, 900, 'en')
    ev(pg, "()=>{location.hash='continuous-batching~cont.12.1536.0.1.50'}"); pg.wait_for_timeout(350)
    sl = stage(pg)
    check('a setup link opened mid-guide shows the finished run in the scenarios mode', not sl['pending'] and sl['status'].startswith('All 48 requests done') and state(pg)['mode'] == 'sandbox', str(sl)[:100])
    pg.context.close()
    # dark mode, both languages and both widths: mid-run (grey cards, playhead in the legend), the 8-slot comparison line, the challenge chart
    for w, h, lang in [(1440, 900, 'en'), (1440, 900, 'zh'), (390, 844, 'en'), (390, 844, 'zh')]:
        pg = page(b, w, h, lang, dark=True); o = [over(pg)]
        go(pg, 2); setr(pg, 'cb-scrub', 400); shot(pg, f'{lang}_{w}_dark_step3_mid'); o.append(over(pg))
        sm = stage(pg)
        check(f'dark {lang} {w}: mid-run state (badge “Paused”, whole-run cards waiting, playhead in the legend)', sm['pending'] and sm['lgHead'] and sm['status'] == PAUSED[lang], str(sm)[:120])
        go(pg, 3); click(pg, '#cb-rail [data-lever="cont"]'); shot(pg, f'{lang}_{w}_dark_step4'); o.append(over(pg))
        go(pg, 5); setr(pg, 'cb-ic-B-in', 16)
        ev(pg, "()=>document.getElementById('cb-stage').scrollIntoView({block:'start'})"); pg.wait_for_timeout(200)
        shot(pg, f'{lang}_{w}_dark_step6'); o.append(over(pg))
        ev(pg, "()=>window.scrollTo(0,0)"); click(pg, '#view-dyn .modes button[data-mode="challenge"]')
        click(pg, '#cb-rail [data-act="submit"]'); click(pg, '#cb-rail [data-act="reveal"]')
        pg.locator('#cb-ch-res .cmp').scroll_into_view_if_needed(); pg.wait_for_timeout(150)
        shot(pg, f'{lang}_{w}_dark_challenge'); o.append(over(pg))
        check(f'dark {lang} {w}: no horizontal overflow', max(o) <= 0, str(o))
        pg.context.close()
    # home tile teaser
    for w, lang in [(1440, 'en'), (390, 'zh')]:
        pg = page(b, w, 900, lang, hash_='map')
        tile = '#tile-continuous-batching'
        pg.locator(tile).scroll_into_view_if_needed()
        t0 = pg.inner_text(f'{tile} .tz-read')
        pg.click(f'{tile} .seg-ctl button[data-v="cont"]'); pg.wait_for_timeout(150)
        t1 = pg.inner_text(f'{tile} .tz-read')
        check(f'teaser {lang} {w}: the sentence changes and states the result of each policy with its referents: one key number each (47%, 78%); the runs’ lengths on the axis', t0 != t1 and '47%' in t0 and '78%' in t1 and '28' not in t0 and '17' not in t1
              and Z(lang, 'these requests a batch at a time', '让这些请求一批一批地开始') in t0 and Z(lang, 'so these requests finish sooner', '这些请求也就更早做完') in t1, t1)
        if lang == 'en':
            check(f'teaser {lang} {w}: one sentence of at most 35 words (R11.5), in both states', max(len(t0.split()), len(t1.split())) <= 35 and t0.count('. ') == 0 and t1.count('. ') == 0 and 'the slots are busy 78% of the time' in t1, f'{len(t0.split())}/{len(t1.split())} words')
        else:
            units = lambda t: len(re.findall(r'[\u4e00-\u9fff]', t)) + len(re.findall(r'[A-Za-z0-9%]+', t))   # CJK characters + Latin words and numbers
            check(f'teaser {lang} {w}: one sentence in both states, about as long as the English', max(units(t0), units(t1)) <= 60 and t0.count('。') == 1 and t1.count('。') == 1, f'{units(t0)}/{units(t1)} units')
        check(f'teaser {lang} {w}: visual at most 130 px tall', ev(pg, f"()=>document.querySelector('{tile} .tz-cb-g').getBoundingClientRect().height") <= 130)
        ax = ev(pg, f"()=>[...document.querySelectorAll('{tile} .tz-cb-ax span')].map(e=>e.textContent)")
        check(f'teaser {lang} {w}: the axis marks the run’s own end', ax == ['0', Z(lang, '17 steps', '17 步')], str(ax))
        leg = pg.inner_text(f'{tile} .tz-leg')
        check(f'teaser {lang} {w}: the caption says what a slot is and names the rows and bars; legend for the numbers and the hatching, in the lab’s words', pg.inner_text(f'{tile} .tz-cb-cap') == Z(lang, 'The GPU runs at most 4 requests at once, one per slot. Each row is a slot, each bar a request.', 'GPU 一次最多跑 4 个请求，每个在跑的请求占一个 slot。每一行是一个 slot，每一条是一个请求。')
              and Z(lang, 'number: tokens in that answer', '数字：这个回答的 token 数') in leg and Z(lang, 'idle while requests wait', '有请求在等时空着') in leg, leg.replace('\n', ' '))
        notes = ev(pg, f"()=>[...document.querySelectorAll('{tile} .tz-note')].map(n=>n.textContent)")
        check(f'teaser {lang} {w}: 2 to 4 notes as sentences, one question each (step and bar length, hand-picked lengths, the 47% / 78%, the lab); no nested tooltips', 2 <= len(notes) <= 4 and Z(lang, 'so a bar is as many steps long as its answer has tokens', '所以一条有几步长，这个回答就有几个 token') in notes[0]
              and Z(lang, 'hand-picked', '手工挑') in notes[1] and Z(lang, '53 ÷ (4 × 28) = 47%', '53 ÷（4 × 28）= 47%') in notes[2] and Z(lang, '53 ÷ (4 × 17) = 78%', '53 ÷（4 × 17）= 78%') in notes[2]
              and ev(pg, f"()=>document.querySelectorAll('{tile} .tz [data-tipx]').length") == 0, str([n[:30] for n in notes]))
        hat = ev(pg, f"()=>document.querySelectorAll('{tile} .tz-cb-idle').length")
        check(f'teaser {lang} {w}: continuous batching: the idle time at the end, with nobody waiting, is blank, as in the lab', hat == 0, str(hat))
        chip = ev(pg, f"()=>{{const c=document.querySelector('{tile} .tz-kind'); return c?c.textContent:''}}")
        check(f'teaser {lang} {w}: the tile says it is a toy run', chip == Z(lang, 'Toy run', '玩具实跑'), chip)
        check(f'teaser {lang} {w}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', ev(pg, f"()=>!document.querySelector('{tile} .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('{tile}').innerText)"))
        check(f'teaser {lang} {w}: no overflow', over(pg) <= 0)
        pg.mouse.move(2, 2); pg.wait_for_timeout(100)
        pg.locator(tile).screenshot(path=str(SH / f'teaser_{lang}_{w}.png'))
        pg.click(f'{tile} .seg-ctl button[data-v="static"]'); pg.wait_for_timeout(120)
        ax0 = ev(pg, f"()=>[...document.querySelectorAll('{tile} .tz-cb-ax span')].map(e=>e.textContent)")
        check(f'teaser {lang} {w}: static batching: the axis marks its own end', ax0 == ['0', Z(lang, '28 steps', '28 步')], str(ax0))
        hat = ev(pg, f"()=>document.querySelectorAll('{tile} .tz-cb-idle').length")
        check(f'teaser {lang} {w}: static batching: hatched idle only in the two batches others wait behind (6 marks), the last batch blank', hat == 6, str(hat))
        pg.locator(tile).screenshot(path=str(SH / f'teaser_{lang}_{w}_static.png'))
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
        _pg.evaluate("()=>{location.hash='continuous-batching'}"); _pg.wait_for_timeout(500)
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

# polish round 5 (2026-10-17, R3.2, R9.7, R7.12, the owner's B4; as tests/moe_flow.py): the reader's walk on the whole site (built into
# out/, wrapped with a doctype; the one-lab page lacks the “Read first” line, which moves every card 28 px), in the lab's real layout (the
# list of steps beside the rail, window.__aspPrimer, as tests/kit_flow.py), entering each part as the polish probe stage.py does: Next
# when it is on (a real click, which scrolls it into view first), else the part's title. Entering part 6 in English, the kit's reveal put
# the card's top under the top bar, little of the bench was left under it, and the bench's end pushed the sticky stage up to −285 px
# (the whole chart above the screen); part 7 in Chinese to 61 px. keepStage() pulls it back once the page stops scrolling. At every
# part the stage's top stays at or below the top bar's bottom; at parts 6 and 7 it is not pushed above its sticky line (70 px; it sits
# there in English at both and in Chinese at part 7, lower when the page did not scroll), needs no scrolling inside, and has 30 px to
# spare in Chinese
subprocess.run([sys.executable, str(P / 'build.py'), 'out/site-continuous-batching.html'], check=True, capture_output=True)
(P / 'out/wrapped-site-continuous-batching.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                                                           + (P / 'out/site-continuous-batching.html').read_text() + '</body></html>')
SITE = 'file://' + str(P / 'out/wrapped-site-continuous-batching.html')
WALK = """()=>{const s=document.querySelector('#cb-stage'), r=s.getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect().bottom,
  cur=document.querySelector('#cb-rail .steps>.step.current'), h=s.querySelector('.stage-head').getBoundingClientRect(), g=(document.querySelector('#cb-gantt')||s).getBoundingClientRect();
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, top:Math.round(r.top), head:Math.round(h.top), bar:Math.round(bar), scroll:s.scrollHeight-s.clientHeight,
    chart:[Math.round(g.top),Math.round(g.bottom)], vh:innerHeight,
    spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight))};}"""
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for _lang in ('en', 'zh'):
        _pg = _b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); _pg.add_init_script('window.__aspPrimer=true')
        _pg.on('pageerror', lambda e: errs.append(f'1440 walk: {e}'))
        _pg.goto(SITE); _pg.wait_for_timeout(600); _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(80)
        _pg.evaluate("()=>{location.hash='continuous-batching'}"); _pg.wait_for_timeout(500)
        _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
        if _g.count(): _g.first.click(); _pg.wait_for_timeout(600)
        _n = _pg.evaluate("()=>document.querySelectorAll('#cb-rail .steps>.step').length"); _rows = [_pg.evaluate(WALK)]
        for _i in range(1, _n):
            _nb = _pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
            if _nb.count(): _nb.first.click()
            else: _pg.evaluate(f"()=>document.querySelector('#cb-rail .steps>.step:nth-child({_i + 1}) .step-h').click()")
            _pg.wait_for_timeout(1500); _rows.append(_pg.evaluate(WALK))
        check(f'1440 {_lang} walk: entering every part, the stage’s top and its head stay below the top bar’s bottom, the top at its sticky line (70 px) or lower (R3.2, R9.7, B4)',
              len(_rows) == _n and all(r['part'] == i + 1 and r['top'] >= max(68, r['bar'] - 1) and r['head'] >= r['bar'] - 1 for i, r in enumerate(_rows)), [(r['part'], r['top']) for r in _rows])
        for _k in (6, 7):
            _r = _rows[_k - 1] if len(_rows) >= _k else {}
            check(f'1440 {_lang} walk part {_k}: the stage is not pushed above its sticky line (its top at 70 px or lower), needs no scrolling inside' + (', 30 px to spare' if _lang == 'zh' else '') + ' (B4, R7.12)',
                  bool(_r) and _r['top'] >= 68 and _r['scroll'] <= 0 and _r['spare'] >= (30 if _lang == 'zh' else 0), _r)
        # the parts whose stage the bench's end pushed up before (or that sit there): keepStage brings it back, the Gantt chart whole on
        # screen. Since the 2026-10-22 refit's shorter lab head the reveal often has no need to scroll here, so the stage may stay lower
        # than its sticky line; the chart must still be whole on screen and the stage never above its sticky line
        for _k in {'en': (6, 7), 'zh': (7,)}[_lang]:
            _r = _rows[_k - 1] if len(_rows) >= _k else {}
            check(f'1440 {_lang} walk part {_k}: the stage’s top at its sticky line (70 px) or lower and the Gantt chart whole on screen (the stage was at ' + {('en', 6): '−285', ('en', 7): '70', ('zh', 7): '61'}[(_lang, _k)] + ' px before keepStage)',
                  bool(_r) and _r['top'] >= 68 and _r['chart'][0] >= _r['bar'] - 1 and _r['chart'][1] <= _r['vh'], _r)
        _pg.close()
    _b.close()


print('\n'.join(log))
print('page errors:', errs or 'none')
print('FAILED:', fails or 'none')
sys.exit(1 if fails or errs else 0)
