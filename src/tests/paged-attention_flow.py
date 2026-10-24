"""Flow test for the PagedAttention lab: every mode and step at 1440x900 and 390x844, in EN and ZH, in light and dark:
page errors, horizontal overflow (also at 1280, 900, 720), next-step gating, the hidden answer until step 3 (stage, status,
its tooltip, mini result bar), the whole run shown only at the last step, scrolling the new step card into view after
Next / Back / a step header / "Take the challenge" / a mode switch, challenge feedback tiers and status reasons, language
switch keeping state, hash round-trip, the mini result bar at 390, tooltips and jumps, keyboard on the pool, reduced motion,
playback to the last step, the playback speed control (three speeds, pressed state, a change while playing, Show me at Fast),
no "·" and no "起点" in any visible Chinese text of the lab and the tile, and the home-tile teaser with the spec's sentences.
Round 8: the part-3 verdict names the right option for a wrong prediction; change badges reproduce from the shown values and are
cleared on every part entry and mode switch, even when the run is unchanged; the scenario note sits under the scenario list and is
on screen after a scenario click; the challenge's mini result bar is not cut on a phone. Exact kit badge wordings for edge cases
(multiples, from or to 0, thousands separators, 1 point) are not asserted.
Screenshots go to shots/paged-attention/.
Run: python3 tests/paged-attention_flow.py
"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
SH = P / 'shots' / 'paged-attention'
SH.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/paged-attention.html', '--only', 'paged-attention'], check=True, cwd=P, capture_output=True)
(P / 'out' / 'wrapped-paged-attention.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'paged-attention.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-paged-attention.html')
out, fails, errs = [], [], []


def check(name, cond, info=''):
    out.append(('OK   ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not cond:
        fails.append(name)


def open_page(b, w, h, lang, hash_='paged-attention', **kw):
    pg = open_drawers(b.new_page(viewport={'width': w, 'height': h}, **kw))
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.goto(URL)
    pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]')
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}")
    pg.wait_for_timeout(450)
    return pg


def click(pg, sel, wait=160):
    pg.evaluate("s=>{const e=document.querySelector(s); if(e) e.scrollIntoView({block:'center'})}", sel)
    pg.wait_for_timeout(60)
    try:
        pg.click(sel, timeout=2500)
    except Exception:
        pg.evaluate("s=>document.querySelector(s).click()", sel)
    pg.wait_for_timeout(wait)


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-window.innerWidth")
next_disabled = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#view-dyn [data-nav=next]'); return b?b.disabled:null}")
obs = lambda pg: pg.evaluate("()=>{const o=document.querySelector('#view-dyn [data-obs]'); return o&&!o.hidden?o.innerText:''}")
state = lambda pg: pg.evaluate("()=>window.__pa.state()")
tip = lambda pg: pg.evaluate("()=>document.getElementById('tip').hidden?'':document.getElementById('tip').innerText")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText:''}", sel)
top_of = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?Math.round(e.getBoundingClientRect().top):null}", sel)
dock = lambda pg: pg.evaluate("()=>({hidden:document.getElementById('dock').hidden,text:document.getElementById('dock').innerText})")


def in_view(pg, sel, name, h):
    """after a reveal: the element's top edge is on screen, below the top bar (and the mini bar), not far down. On a desktop lab page the
    top bar scrolls away (refit step 2): there the bound is what is left of the bar on the screen, wherever that is less than 40 px"""
    pg.wait_for_timeout(900)
    y = top_of(pg, sel)
    bar = pg.evaluate("()=>{const b=document.querySelector('.topbar'); return b?Math.max(0,Math.round(b.getBoundingClientRect().bottom)):0}")
    check(name, y is not None and min(40, bar) <= y <= h * 0.6, f'{y} (top bar {bar})')


def shot(pg, name, full=False):
    """save a screenshot and check that no broken value (NaN, undefined) or 'Locked in' is on screen"""
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)
    txt = pg.evaluate("()=>{const v=document.getElementById('view-dyn'); return (v.hidden?'':v.innerText)+' '+document.getElementById('dock').innerText+' '+document.getElementById('tip').innerText}")
    bad = [w for w in ('NaN', 'undefined', 'Infinity', '[object', 'Locked in', '已锁定', 'in step 3', 'answer step 2', 'Step 3 has', 'next step.', '第 3 步显示', '第 3 步看', '第 2 步作答', '结果在第 3 步') if w in txt]
    check(f'{name}: no broken values, "Locked in" or guided "step" on screen', not bad, str(bad))
    copy_rules(pg, name, txt)


def copy_rules(pg, name, txt):
    """the owner's copy rules on everything visible: no '·' joining clauses, no '起点' / 'starting point', room is never counted in
    '位置' / 'slots', the settings panel is never called '控件' / 'Controls', and no sentence describes the page itself (the mini result bar, hovering)"""
    zh = pg.evaluate("()=>document.documentElement.lang").startswith('zh')
    bad = []
    if zh:
        bad += [w for w in ('·', '起点', '控件', '迷你', '悬停', '重画') if w in txt]
        bad += re.findall(r'\d 个位置', txt)
    else:
        low = txt.lower()
        bad += re.findall(r'\d[\d,]* (?:token )?slots', txt) + [w for w in ('starting point', 'real tokens', 'mini result bar', 'mini bar', 'hover', 'redraws') if w in low]
        bad += re.findall(r'\bcontrols\b', low) + re.findall(r' · ', txt)
    check(f'{name}: copy rules ({"no ·, 起点, 控件 or page-behavior sentences" if zh else "no ·, slots, starting point, Controls or page-behavior sentences"})', not bad, str(bad))


def hover(pg, sel):
    """scroll first and let the scroll event pass (a scroll quiets tooltips on purpose), then move the mouse onto the element"""
    pg.evaluate("s=>document.querySelector(s).scrollIntoView({block:'center'})", sel)
    pg.wait_for_timeout(250)
    pg.hover(sel)
    pg.wait_for_timeout(150)


badges = "()=>[...document.querySelectorAll('#pa-metrics .metric .d')].map(x=>x.textContent)"
ghost = lambda pg: pg.evaluate("()=>getComputedStyle(document.getElementById('pa-ghost')).opacity")
mvals = "()=>['util','run','steps','unused','table'].map(k=>document.querySelector(`#pa-metrics [data-k=${k}] .v`).innerText)"


def shown_num(v):
    v = re.sub(r'[^0-9.\-]', '', v)
    return float(v) if v not in ('', '.', '-') else None


def badge_rule(a, b_, pts):
    """what a badge must start with, from the two shown values (the kit's rule: points for shares, otherwise a percentage rounded
    half away from zero); multiples and changes from or to 0 are checked for direction only, since their wording is being reworded"""
    fa, fb = shown_num(a), shown_num(b_)
    if fa is None or fb is None or fa == fb:
        return ''
    up = fb > fa
    if pts:
        return ('▲ ' if up else '▼ ') + f'{abs(fb - fa):.1f}'.rstrip('0').rstrip('.')
    if fa == 0 or fb == 0 or fb / fa >= 1.995:
        return '▲' if up else '▼'
    r = (fb / fa - 1) * 100
    p = int(abs(r) + 0.5 + 1e-9) * (1 if r > 0 else -1)
    if p == 0:
        return ''
    if p <= -100:
        return '▼ −'
    return ('▲ +' if p > 0 else '▼ −') + f'{abs(p)}%'


def badges_match(before, after, got):
    """every metric card's badge against the rule; returns the mismatches"""
    bad = []
    for k, (a, b_, g) in enumerate(zip(before, after, got)):
        want = badge_rule(a, b_, k == 0)
        if (want == '' and g != '') or (want != '' and not g.startswith(want)):
            bad.append(f'{a} -> {b_}: {g!r}, want {want!r}')
    return bad


def note_on_screen(pg):
    """the scenario note is fully on screen: below the top bar and the mini result bar, above the bottom edge"""
    pg.wait_for_timeout(750)
    return pg.evaluate("""()=>{const n=document.getElementById('pa-sbnote'), r=n.getBoundingClientRect(), bar=document.querySelector('.topbar'), dk=document.getElementById('dock');
      const lo=Math.max(bar?bar.getBoundingClientRect().bottom:0, dk&&!dk.hidden?dk.getBoundingClientRect().bottom:0);
      return {ok:r.height>0&&r.top>=lo-1&&r.bottom<=innerHeight+1, top:Math.round(r.top), bottom:Math.round(r.bottom), lo:Math.round(lo), vh:innerHeight}}""")


def set_time(pg, v):
    pg.evaluate("v=>{const i=document.getElementById('pa-ic-time-in'); i.value=(v==='max'?i.max:v); i.dispatchEvent(new Event('input',{bubbles:true}))}", v)
    pg.wait_for_timeout(200)


# G1 and G2, the owner's message of 2026-10-08 (reviews/owner-b9-2026-10-08.md): a lead-in above each group of options instead of
# “one at a time; they do not stack”, and no sentence that leans on a detail of an earlier part. Checked on the lab's source, both languages.
_G_SRC = (P / 'labs' / 'paged-attention.js').read_text()
for _b in ["一次只能选", "一次选一个", "一次选一种", "一次只选", "不会叠加", "do not stack", "One at a time", "pick one at a time", "Pick one at a time", "Try one change at a time", "一次试一种"]:
    check(f'G1: no “{_b}” left in the lab (EN and ZH)', _b not in _G_SRC)
for _s in ["Pick one and see how much of the pool it takes.", "选一种，看看它占了显存池的多少。", "Pick an allocation strategy and see how much of the pool stores tokens while requests wait.", "选一种分配方式，看看有请求排队时，显存池有多少存着 token。", "Pick a scenario and see how much of the pool stores tokens while requests wait.", "选一个场景，看看有请求排队时，显存池有多少存着 token。"]:
    check(f'G1: the lead-in “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["To return to the initial setup, press Back and then Next.", "想回到初始设置，就先点“上一部分”，再点“下一部分”。"]:
    check(f'G2: the rewritten sentence “{_s[:48]}” is in the lab', _s in _G_SRC)
for _s in ["Go Back and then Next to return", "点“上一部分”再点“下一部分”，就能回到初始设置。"]:
    check(f'G2: the sentence that leaned on an earlier part, “{_s[:48]}”, is gone', _s not in _G_SRC)

with sync_playwright() as pw:
    b = pw.chromium.launch()
    for scheme in ['light', 'dark']:
        for (w, h) in [(1440, 900), (390, 844)]:
            for lang in ['en', 'zh']:
                tag = f'{w}_{lang}_{scheme}'
                zh = lang == 'zh'
                Z = lambda en, zh_: zh_ if zh else en
                pg = open_page(b, w, h, lang, color_scheme=scheme)
                check(f'{tag} title', Z('Why is most KV cache memory wasted?', '一大半被浪费') in pg.inner_text('#view-dyn .labtitle'))
                check(f'{tag} no Fidelity button on top; the notes close the page (owner, 2026-10-11 19:28 UTC)', not pg.query_selector('#view-dyn [data-pop="paged-attention-fid"]') and pg.evaluate("()=>{const e=document.getElementById('paged-attention-fid'); return !!e&&!e.hidden&&!e.nextElementSibling}"))
                check(f'{tag} no overflow at load', overflow(pg) <= 0, str(overflow(pg)))
                if w == 390:
                    check(f'{tag} step cards come before the stage', top_of(pg, '#pa-rail') < top_of(pg, '#pa-stage'))
                # --- step 1: the answer is hidden everywhere, including the status tooltip and the mini result bar
                check(f'{tag} step1 next enabled', next_disabled(pg) is False)
                hid = pg.evaluate("()=>({st:document.getElementById('pa-status').innerText,tipx:document.getElementById('pa-status').dataset.tipx||'',key:document.getElementById('pa-status').dataset.tip||'',v:document.querySelector('#pa-metrics [data-k=util] .v').innerText,bar:getComputedStyle(document.getElementById('pa-baravg')).visibility,now:getComputedStyle(document.getElementById('pa-barnow')).visibility,nowlab:document.getElementById('pa-nowlab').innerText})")
                check(f'{tag} the answer stays hidden until the prediction', hid == {'st': Z('Predict first', '先预测'), 'tipx': Z('The whole-run result stays hidden until you answer part 2.', '整次运行的结果在第 2 部分作答前先隐藏。'), 'key': '', 'v': '?', 'bar': 'hidden',
                                                                                'now': 'hidden', 'nowlab': Z('Now: shown after you predict', '此刻：预测后显示')}, str(hid))
                rd1 = text(pg, '#pa-st-time .pa-tread')
                check(f'{tag} before the prediction the live line gives no share', rd1 == Z('Decode step 1 of 2,809: 6 requests running, 34 waiting, 0 finished.', 'decode 第 1 步（共 2,809 步）：6 个请求在跑，34 个在排队，0 个已完成。'), rd1)
                check(f'{tag} metrics titled as the whole run', text(pg, '#pa-mhead').lower() == Z('whole run (all 40 requests)', '整次运行（全部 40 个请求）'), text(pg, '#pa-mhead'))
                check(f'{tag} before the prediction no whole-run number shows', pg.evaluate(mvals) == ['?', '?', '?', '?', Z('none', '无')], str(pg.evaluate(mvals)))
                check(f'{tag} part 1 gives no average length to compute the answer from', '640' not in text(pg, '#view-dyn .step.current .step-b'))
                p1 = pg.evaluate("()=>({chips:document.querySelectorAll('#view-dyn .step.current .step-b .chips, #view-dyn .step.current .step-b .tag').length,paras:document.querySelectorAll('#view-dyn .step.current .step-b p').length})")
                check(f'{tag} part 1: three paragraphs, no chips and no RTX 4090 paragraph', p1 == {'chips': 0, 'paras': 3} and 'RTX' not in text(pg, '#view-dyn .step.current .step-b'), str(p1))
                if w == 390:
                    pg.evaluate("()=>{const c=document.querySelector('#pa-rail .step.current'); window.scrollTo(0,c.getBoundingClientRect().top+scrollY-120)}")
                    pg.wait_for_timeout(450)
                    d = dock(pg)
                    check(f'{tag} step1 mini result bar hides the answer', not d['hidden'] and Z('Predict first', '先预测') in d['text'] and '%' not in d['text'], d['text'].replace('\n', ' '))
                shot(pg, f'{tag}_step1')
                # --- step 2: predict; the prediction can change until step 3 shows the run
                click(pg, '#view-dyn [data-nav="next"]')
                check(f'{tag} step2 gated before a prediction', next_disabled(pg) is True)
                check(f'{tag} step2 choices show radio dots', pg.evaluate("()=>getComputedStyle(document.querySelector('#view-dyn .choices .choice'),'::before').content!=='none'"))
                opts2 = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-pred]')].map(x=>x.innerText)")
                check(f'{tag} step2 four prediction options without gaps', opts2 == Z(['Under 30%', '30% to 60%', '60% to 80%', 'Over 80%'], ['不到 30%', '30% 到 60%', '60% 到 80%', '80% 以上']), str(opts2))
                mine = '30to60' if scheme == 'light' else 'lt30'
                click(pg, '#view-dyn [data-pred="gt80"]')
                click(pg, f'#view-dyn [data-pred="{mine}"]')
                body2 = text(pg, '#view-dyn .step.current .step-b')
                check(f'{tag} step2 prediction saved and still changeable', next_disabled(pg) is False and pg.get_attribute(f'#view-dyn [data-pred="{mine}"]', 'aria-pressed') == 'true'
                      and Z('Saved. You can still change it before the next part.', '已记下。进入下一部分之前，还可以改。') in body2, body2[-90:])
                hid = pg.evaluate("()=>({st:document.getElementById('pa-status').innerText,tipx:document.getElementById('pa-status').dataset.tipx||'',v:document.querySelector('#pa-metrics [data-k=util] .v').innerText})")
                check(f'{tag} step2 answer still hidden after predicting', hid == {'st': Z('Result in part 3', '结果在第 3 部分'), 'tipx': Z('The whole-run result stays hidden until you watch the run in part 3.', '整次运行的结果先隐藏，到第 3 部分看运行时再显示。'), 'v': '?'}, str(hid))
                shot(pg, f'{tag}_step2')
                # --- step 3: watch; the whole-run result appears only at the last step
                click(pg, '#view-dyn [data-nav="next"]')
                if w == 1440:
                    in_view(pg, '#pa-rail .step.current', f'{tag} Next brings the new step card into view', h)
                check(f'{tag} step3 gated before the run', next_disabled(pg) is True and obs(pg) == '')
                live3 = pg.evaluate("()=>({st:document.getElementById('pa-status').innerText,bar:getComputedStyle(document.getElementById('pa-baravg')).visibility,lab:document.getElementById('pa-avglab').innerText,v:document.querySelector('#pa-metrics [data-k=util] .v').innerText})")
                check(f'{tag} step3 the badge says running, no whole-run value before the run ends', live3 == {'st': Z('Running', '运行中'), 'bar': 'hidden', 'lab': Z('While requests wait: shown after the run', '排队时平均：跑完后显示'), 'v': '?'}, str(live3))
                check(f'{tag} step3 running and decode-step totals also wait for the end of the run', pg.evaluate(mvals)[:4] == ['?', '?', '?', '?'], str(pg.evaluate(mvals)))
                check(f'{tag} step3 the running badge explains itself', pg.get_attribute('#pa-status', 'data-tipx') == Z('The whole-run averages appear after the last decode step.', '整次运行的平均值在最后一个 decode 步之后显示。'))
                check(f'{tag} step3 speed control next to Play in the card', pg.evaluate("()=>[...document.querySelectorAll('#pa-ic-time .seg-ctl[data-key=speed] button')].map(x=>x.textContent)") == Z(['Slow', 'Medium', 'Fast'], ['慢', '中', '快']))
                check(f'{tag} step3 whole-run numbers are titled and greyed', pg.evaluate("()=>getComputedStyle(document.querySelector('#pa-metrics .metric')).backgroundColor")
                      == pg.evaluate("()=>{const d=document.createElement('div'); d.style.background='var(--surface-2)'; document.body.appendChild(d); const c=getComputedStyle(d).backgroundColor; d.remove(); return c}"))
                click(pg, '#view-dyn #pa-ic-time [data-pa="step"]')
                live = text(pg, '#view-dyn .pa-live')
                check(f'{tag} step3 +1 decode step shows the live step, not the result', obs(pg) == '' and next_disabled(pg) is True and live == Z('Decode step 2 of 2,809: 6 requests running, 34 waiting, 0 finished. Right now 16.3% of the pool stores tokens.', 'decode 第 2 步（共 2,809 步）：6 个请求在跑，34 个在排队，0 个已完成。此刻显存池有 16.3% 存着 token。') and pg.inner_text('#pa-status') == Z('Running', '运行中'), live)
                check(f'{tag} step3 the stage line leaves the share to the bar', text(pg, '#pa-st-time .pa-tread') == Z('Decode step 2 of 2,809: 6 requests running, 34 waiting, 0 finished.', 'decode 第 2 步（共 2,809 步）：6 个请求在跑，34 个在排队，0 个已完成。')
                      and text(pg, '#pa-nowlab') == Z('Now: 16.3% stores tokens', '此刻：16.3% 存着 token'), text(pg, '#pa-st-time .pa-tread'))
                check(f'{tag} step3 both time controls in sync', pg.evaluate("()=>[document.getElementById('pa-ic-time-in').value,document.getElementById('pa-st-time-in').value]") == ['1', '1'])
                if w == 390:
                    # the stage is below the step cards: the mini result bar keeps the result in view
                    pg.evaluate("()=>document.getElementById('pa-ic-time').scrollIntoView({block:'center'})")
                    pg.wait_for_timeout(350)
                    d = dock(pg)
                    check(f'{tag} dock during the run: running, the decode step and a number marked "now"', not d['hidden'] and Z('Running', '运行中') in d['text'] and Z('Decode step 2: 16.3% of the pool stores tokens now', 'decode 第 2 步：此刻 16.3% 存着 token') in d['text'] and Z('while requests wait', '排队时') not in d['text'].lower(), d['text'][:80])
                    shot(pg, f'{tag}_step3_dock')
                    w0 = pg.evaluate("()=>document.querySelector('#dkVis .pa-sg').style.width")
                    set_time(pg, 900)
                    check(f'{tag} dock mini bar follows the step', pg.evaluate("()=>document.querySelector('#dkVis .pa-sg').style.width") != w0 and Z('Decode step 901', 'decode 第 901 步') in dock(pg)['text'], dock(pg)['text'][:80])
                set_time(pg, 'max')
                o3 = obs(pg)
                check(f'{tag} step3 result at the last step: the rounded share in the text, the exact run on the cards', next_disabled(pg) is False and Z('about 25%', '约 25%') in o3 and '2,809' not in o3, o3[:90])
                if mine == 'lt30':
                    check(f'{tag} step3 a right prediction is confirmed', Z('Your prediction, “Under 30%”, was right.', '你预测的“不到 30%”是对的。') in o3, o3[-120:])
                else:
                    check(f'{tag} step3 a wrong prediction gets the verdict with the right option', Z('You predicted “30% to 60%”; the right answer is “Under 30%”.', '你预测的是“30% 到 60%”，正确答案是“不到 30%”。') in o3, o3[-160:])
                check(f'{tag} step3 the reason follows, in words, right or wrong', Z('Why: most of each region is empty, either waiting for later tokens or never used.', '原因：每一块大多空着，有的在等后面的 token，有的到结束也用不上。') in o3, o3[-160:])
                check(f'{tag} step3 the observation names who reserves (no dangling “Reserving the maximum,”)', 'Reserving the maximum,' not in o3 and (Z('When each request reserves the maximum, only about 25% of the pool stores tokens while requests wait, so only a few requests run at once.',
                      '按最大长度预留时，有请求排队的那些 decode 步里，显存池平均只有约 25% 存着 token，所以同时只能跑几个请求。') in o3), o3[:160])
                check(f'{tag} step3 last-step readout', text(pg, '#view-dyn .pa-live') == Z('Decode step 2,809 (the last): the final request finishes.', 'decode 第 2,809 步（最后一步）：最后一个请求在这一步完成。'), text(pg, '#view-dyn .pa-live'))
                end3 = pg.evaluate("()=>({st:document.getElementById('pa-status').innerText,bar:getComputedStyle(document.getElementById('pa-baravg')).visibility,v:document.querySelector('#pa-metrics [data-k=util] .v').innerText})")
                check(f'{tag} step3 the waiting-step average shows once the run ends', end3 == {'st': Z('While requests wait: 24.8% stores tokens on average', '排队时平均 24.8% 存着 token'), 'bar': 'visible', 'v': '24.8%'}, str(end3))
                check(f'{tag} step3 the metrics name the waiting-step averages', [text(pg, f'#pa-metrics [data-k={k}] .k') for k in ('util', 'run')] == Z(['Stores tokens while requests wait', 'Running at once while requests wait'], ['排队时存着 token', '排队时同时在跑'])
                      and text(pg, '#pa-avglab') == Z('While requests wait: 24.8% on average', '排队时平均：24.8%') and pg.evaluate(mvals) == ['24.8%', '5', '2,809', '1,232.5', Z('none', '无')], str(pg.evaluate(mvals)))
                if w == 390:
                    pg.evaluate("()=>document.getElementById('pa-ic-time').scrollIntoView({block:'center'})")
                    pg.wait_for_timeout(350)
                    d = dock(pg)
                    check(f'{tag} dock after the run: the waiting-step average, labelled', not d['hidden'] and Z('While requests wait', '排队时平均') in d['text'] and Z('24.8% stores tokens on average', '24.8% 存着 token') in d['text'], d['text'][:80])
                click(pg, '#view-dyn #pa-ic-time [data-pa="step"]')
                check(f'{tag} +1 decode step at the end explains it', Z('the final request finishes in it', '最后一个请求在这一步完成') in tip(pg), tip(pg)[:120])
                shot(pg, f'{tag}_step3')
                # back to step 2: the prediction is now fixed
                click(pg, '#view-dyn [data-nav="back"]')
                check(f'{tag} after the run the prediction is closed', pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-pred]')].every(x=>x.disabled)")
                      and Z('Part 3 has shown the answer', '第 3 部分已经给出答案') in text(pg, '#view-dyn .step.current .step-b'))
                click(pg, '#view-dyn [data-nav="next"]')
                check(f'{tag} step3 revisited keeps its result', next_disabled(pg) is False and Z('about 25%', '约 25%') in obs(pg))
                # --- step 4: kinds of waste (single choice levers, results kept)
                click(pg, '#view-dyn [data-nav="next"]')
                check(f'{tag} step4 gated', next_disabled(pg) is True)
                before = obs(pg)
                click(pg, '#view-dyn [data-lever="hl-resv"]')
                o4 = obs(pg)
                check(f'{tag} step4 lever changes the result text (a rounded share; the lever keeps the exact one)', o4 != before and Z('about 11%', '约占显存池的 11%') in o4 and pg.inner_text('#view-dyn [data-lever="hl-resv"] .lv-r').strip() == Z('11.4% of the pool while requests wait', '排队时占显存池 11.4%'), o4[:80])
                check(f'{tag} step4 single choice pressed', pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-lever^=hl-]')].filter(x=>x.getAttribute('aria-pressed')==='true').map(x=>x.dataset.lever)") == ['hl-resv'])
                click(pg, '#view-dyn [data-lever="hl-intf"]')
                check(f'{tag} step4 internal fragmentation, stopped at decode step 1,189 and why', Z('about 41%', '约占显存池的 41%') in obs(pg) and Z('The pool stops at decode step 1,189: its mix of room is the closest to the average while requests wait.', '显存池停在 decode 第 1,189 步：这一步各种空间的比例最接近排队时的平均。') in text(pg, '#view-dyn .step.current .step-b'))
                check(f'{tag} step4 the note names the two kinds a server cannot tell apart', Z('cannot tell “reserved for later tokens” from “internal fragmentation”', '分不清“为后面的 token 预留”和“内部碎片”') in text(pg, '#view-dyn .step.current .step-b'))
                check(f'{tag} step4 lever result names the pool and the average', pg.inner_text('#view-dyn [data-lever="hl-intf"] .lv-r').strip() == Z('40.5% of the pool while requests wait', '排队时占显存池 40.5%'))
                click(pg, '#view-dyn [data-lever="hl-ext"]')
                o4e = obs(pg)
                check(f'{tag} step4 external fragmentation: gaps and the end of the pool, too small for the next request', Z('about 23%', '约占显存池的 23%') in o4e and Z('the whole region the next request needs', '下一个请求要的整块') in o4e and Z('the end of the pool', '显存池末尾') in o4e
                      and Z('the end of the pool included', '包括显存池末尾') in text(pg, '#view-dyn [data-lever="hl-ext"] .lv-d'))
                check(f'{tag} step4 tried options keep their result', pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-lever^=hl-] .lv-r')].every(x=>x.textContent.trim().length>0)"))
                check(f'{tag} step4 legend shows the highlight', pg.evaluate("()=>document.querySelector('#pa-legend [data-jump=hl-ext]').getAttribute('aria-pressed')") == 'true')
                check(f'{tag} step4 region-start line is in the legend', pg.evaluate("()=>!!document.getElementById('pa-lg-start').offsetParent"))
                shot(pg, f'{tag}_step4')
                # --- step 5: switch to paging (controlled comparison)
                click(pg, '#view-dyn [data-nav="next"]')
                check(f'{tag} step5 gated', next_disabled(pg) is True)
                click(pg, '#view-dyn [data-lever="st-paged"]', wait=500)
                o5 = obs(pg)
                check(f'{tag} step5 paged result against the initial setup (one rounded pair; the running requests and steps are on the cards)', all(v in o5 for v in ('96%', '25%', Z('initial setup', '初始设置'))) and not any(v in o5 for v in ('96.3%', '25.5', '976', '2,809')) and next_disabled(pg) is False, o5[:100])
                check(f'{tag} step5 side facts moved to Fidelity', not any(v in o5 for v in ('188', '2.14', 'Orca', Z('ran out of', '暂停'))), o5[:200])
                check(f'{tag} step5 stage follows', pg.inner_text('#pa-sub') == Z('Paging, 16-token blocks', '分页，16 个 token 一个 block'))
                check(f'{tag} step5 tried levers keep their result', pg.inner_text('#view-dyn [data-lever="st-paged"] .lv-r').strip() == Z('96.3% stores tokens while requests wait', '排队时 96.3% 存着 token'))
                check(f'{tag} step5 says why the decode steps drop less than the running requests rise', Z('So over five times as many requests run at once as in the initial setup. The decode steps do not fall to a fifth, though: once every waiting request has started, fewer and fewer are still running.',
                      '所以同时跑的请求是初始设置的五倍多。decode 步数却没有降到五分之一：排队的请求都开始以后，还在跑的请求越来越少。') in o5, o5[-240:])
                check(f'{tag} step5 the page-table analogy explains the name', Z('memory pages and page tables', '内存页和页表') in text(pg, '#view-dyn .step.current .step-b'))
                check(f'{tag} step5 no region-start legend for paging', not pg.evaluate("()=>!!document.getElementById('pa-lg-start').offsetParent"))
                shot(pg, f'{tag}_step5_paged')
                click(pg, '#view-dyn [data-lever="st-exact"]')
                o5x = obs(pg)
                check(f'{tag} step5 exact length: knowing the lengths is not enough', '53%' in o5x and '53.2%' not in o5x and Z('Knowing the lengths is not enough', '即使事先知道长度也不够') in o5x and o5x != o5)
                shot(pg, f'{tag}_step5_exact')
                # language switch keeps state (step, lever, observation)
                other = 'en' if zh else 'zh'
                pg.click(f'.langsw button[data-lang="{other}"]')
                pg.wait_for_timeout(300)
                st = state(pg)
                check(f'{tag} language switch keeps step and setting', st['step'] == 4 and st['S']['strat'] == 'exact' and '53%' in obs(pg), str(st['S']))
                pg.click(f'.langsw button[data-lang="{lang}"]')
                pg.wait_for_timeout(300)
                click(pg, '#view-dyn [data-lever="st-start"]')
                o5s = obs(pg)
                check(f'{tag} step5 the initial setup ends with what its share means, in words', o5s.strip().endswith(Z('so few requests run at once and the run takes many decode steps.', '所以同时只能跑几个请求，整次运行要很多个 decode 步。')) and not any(x in o5s for x in ('Pick one', '比一比')), o5s[-90:])
                # the learner's own change shows badges; Next into part 6 (the same run: paging, 16) clears them and the dashed line
                v0 = pg.evaluate(mvals)
                click(pg, '#view-dyn [data-lever="st-paged"]', wait=60)
                dl, v1 = pg.evaluate(badges), pg.evaluate(mvals)
                check(f'{tag} step5 badges reproduce from the shown values', any(dl) and not badges_match(v0, v1, dl), str(badges_match(v0, v1, dl) or dl))
                # --- step 6: block size; Next from the bottom of a long card brings the new card into view
                click(pg, '#view-dyn [data-nav="next"]', wait=60)
                check(f'{tag} entering a part clears the badges and the dashed line, even when the run is unchanged', pg.evaluate(badges) == [''] * 5 and ghost(pg) == '0' and state(pg)['S']['strat'] == 'paged', f'{pg.evaluate(badges)} {ghost(pg)}')
                in_view(pg, '#pa-rail .pbar' if w <= 760 else '#pa-rail .step.current', f'{tag} Next from the bottom of step 5 shows step 6' + (' (phones: its part bar)' if w <= 760 else ''), h)
                check(f'{tag} step6 gated at the default size', next_disabled(pg) is True)
                check(f'{tag} entering a part shows no change badge', pg.evaluate(badges) == [''] * 5, str(pg.evaluate(badges)))
                v0 = pg.evaluate(mvals)
                click(pg, '#view-dyn #pa-ic-B button[data-v="256"]', wait=60)
                dl, v1 = pg.evaluate(badges), pg.evaluate(mvals)
                check(f'{tag} the learner\'s own change shows badges against the shown values', dl[0].startswith('▼ 19.1 ') and Z('pts', '个百分点') in dl[0] and dl[4].startswith('▼ −93%') and not badges_match(v0, v1, dl), str(badges_match(v0, v1, dl) or dl))
                o6 = obs(pg)
                check(f'{tag} step6 256-token blocks vs 16: the share as a rounded pair, the tables in words (the badges hold the exact difference)', '77%' in o6 and '96%' in o6 and '19.1' not in o6 and Z('block tables get much shorter', 'block table 短了很多') in o6 and next_disabled(pg) is False, o6[:100])
                check(f'{tag} step6 memory-only note', Z('does not time attention', '不计算 attention 的时间') in text(pg, '#view-dyn .step.current .step-b'))
                check(f'{tag} step6 tried sizes keep their results', pg.evaluate("()=>[...document.querySelectorAll('#view-dyn .pa-btried span')].filter(x=>x.textContent.trim()).length") == 2)
                check(f'{tag} step6 inline and panel controls in sync', pg.evaluate("()=>document.querySelector('#pa-ctl-B button[data-v=\"256\"]').getAttribute('aria-pressed')") == 'true')
                shot(pg, f'{tag}_step6')
                click(pg, '#view-dyn #pa-ic-B button[data-v="1"]')
                o61 = obs(pg)
                check(f'{tag} step6 1-token blocks against 16, in words (the cards hold 97.7% and 639.6)', Z('1-token blocks: against the default size, the share of the pool storing tokens barely changes, but block tables get about 16 times as long. Why: no block is half written, but every token needs its own entry.',
                      '1 个 token 一个 block：和默认大小相比，存着 token 的比例几乎不变，但 block table 变成约 16 倍长。原因：没有写了一半的 block，但每个 token 都要一条 block table 记录。') in o61
                      and pg.evaluate(mvals)[0] == '97.7%' and pg.evaluate(mvals)[4] == '639.6', o61[:200])
                check(f'{tag} step6 1-token blocks: the unused-room note', text(pg, '#pa-metrics [data-k="unused"] .sub') == Z('tokens of room: a block holds 1 token, so none is left unfilled', '个 token 的空间：每个 block 只放 1 个 token，不会有没写满的部分'), text(pg, '#pa-metrics [data-k="unused"] .sub'))
                shot(pg, f'{tag}_step6_b1')
                # a step header also reveals its card
                pg.evaluate("()=>window.scrollTo(0,document.body.scrollHeight)")
                pg.wait_for_timeout(300)
                click(pg, '#view-dyn [data-go="6"]')
                in_view(pg, '#pa-rail .pbar' if w <= 760 else '#pa-rail .step.current', f'{tag} a step header brings its card into view' + (' (phones: its part bar)' if w <= 760 else ''), h)
                check(f'{tag} a part set by the lab clears the change badges', pg.evaluate(badges) == [''] * 5, str(pg.evaluate(badges)))
                # --- step 7: what to remember
                body7 = pg.inner_text('#view-dyn .step.current .step-b')
                nxq = pg.evaluate("()=>window.ASP.catalog.LAB['continuous-batching'].q")
                nl7 = pg.inner_text('#pa-rail .next-lab') if pg.query_selector('#pa-rail .next-lab') else ''
                cb_live = pg.evaluate("()=>window.ASP.catalog.labStatus(window.ASP.catalog.LAB['continuous-batching'])==='live'")
                nl_ok = (Z('Next lab', '下一个 Lab') in nl7 and nxq[lang] in nl7) if cb_live else (Z('pauses here', '暂时到此') in nl7 or Z('last lab of the branch', '这一枝的最后一个 Lab') in nl7)   # a build with only this lab: the serving path pauses here (Codex review 1 #2), or, with no path chosen (round 3), its branch's other labs are not in this build
                if w <= 760:
                    nl_ok = pg.evaluate("()=>{const r=document.querySelector('#pa-rail .next-lab'), e=document.querySelector('#view-dyn .next-lab-end'); return !!e&&getComputedStyle(e).display!=='none'&&(!r||getComputedStyle(r).display==='none')}")
                check(f'{tag} step7 common misconceptions, and the next lab below the card', body7.count('✗') == 3 and Z('common misconceptions', '常见误解') in body7.lower() and nl_ok, nl7)
                shot(pg, f'{tag}_step7')
                check(f'{tag} no overflow in guided', overflow(pg) <= 0, str(overflow(pg)))
                # --- challenge
                # R3.12: guided ends in Scenarios, whose card ends with the way to the challenge
                click(pg, '#view-dyn [data-mode-go="sandbox"]')
                check(f'{tag} guided ends in Scenarios, and the scenarios card ends with "Take the challenge"', pg.get_attribute('#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])', 'aria-pressed') == 'true' and pg.query_selector('#pa-rail [data-mode-go="challenge"]') is not None)
                click(pg, '#view-dyn [data-mode-go="challenge"]')
                in_view(pg, '#pa-stage' if w <= 760 else '#pa-rail', f'{tag} "Take the challenge" brings the task card into view' + (' (phones: the stage, the card under it)' if w <= 760 else ''), h)
                st = state(pg)
                check(f'{tag} challenge starts at reserve-the-maximum', st['mode'] == 'challenge' and st['S'] == {'strat': 'max', 'B': 16, 'mu': 256, 'lim': 1536}, str(st['S']))
                vis = pg.evaluate("()=>['pa-ctl-mu','pa-ctl-lim','pa-adv'].map(id=>{const e=document.getElementById(id); return !!(e&&e.offsetParent)})")
                check(f'{tag} challenge hides the fixed controls', vis == [False, False, False], str(vis))
                check(f'{tag} challenge lock note', not pg.evaluate("()=>document.getElementById('pa-lock').hidden"))
                check(f'{tag} challenge card has only the two allowed controls', pg.evaluate("()=>[...document.querySelectorAll('#pa-rail .seg-ctl:not(.ex-tabs)')].map(x=>x.dataset.key)") == ['strat', 'B'])
                check(f'{tag} the table goal\'s tooltip: met trivially without a block table', pg.get_attribute('#pa-g2', 'data-tipx') == Z('This is the average number of block-table entries per request when it finishes. Reserving by length has no block table, so this goal is met trivially.',
                      '这一项是请求结束时 block table 的平均条数。按长度预留时没有 block table，这一项自然达标。'), pg.get_attribute('#pa-g2', 'data-tipx'))
                check(f'{tag} goal says average', Z('averaging 64 entries or fewer', '平均不超过 64 条') in text(pg, '#pa-rail .rh') and text(pg, '#pa-g2') == Z('Block table, average: none (goal: at most 64 entries)', 'block table 平均：无（目标：最多 64 条）'), text(pg, '#pa-g2'))
                check(f'{tag} live progress', text(pg, '#pa-g1') == Z('Stores tokens while requests wait: 24.8% (goal: at least 90%)', '排队时存着 token：24.8%（目标：至少 90%）'), text(pg, '#pa-g1'))
                goal_fit = "()=>{const r=document.querySelector('#pa-rail .railcard').getBoundingClientRect().right; return [...document.querySelectorAll('#pa-rail .goal>span')].every(e=>e.getBoundingClientRect().right<=r+1)}"
                check(f'{tag} goal rows stay inside the task card', pg.evaluate(goal_fit))
                check(f'{tag} status names the failing goal', pg.inner_text('#pa-status') == Z('Not yet: while requests wait, only 24.8% of the pool stores tokens, under 90%', '未达标：排队时显存池只有 24.8% 存着 token，不到 90%'), pg.inner_text('#pa-status'))
                shot(pg, f'{tag}_ch_start')
                click(pg, '#view-dyn [data-act="submit"]')
                r = pg.inner_text('#pa-chres')
                check(f'{tag} tier 3: not met, hint, no answer, a “Show the answer” button', Z('Not yet', '还没达到') in r and Z('Hint', '提示') in r and not pg.query_selector('#pa-chres .cmp')
                      and text(pg, '#pa-chres [data-act="reveal"]') == Z('Show the answer', '直接看答案'), r[:90])
                shot(pg, f'{tag}_ch_notmet')
                click(pg, '#pa-chres [data-act="reveal"]')
                rows = pg.evaluate("()=>document.querySelectorAll('#pa-chres .cmp-row').length")
                # the owner's review of batch 8: a brief chart, the 3 choices that pass and the initial setup (= the pick); the note counts the other 5
                stray = pg.evaluate("()=>[...document.querySelectorAll('#pa-chres .cmp-row')].filter(r=>!r.classList.contains('ok')&&!r.classList.contains('me')&&!r.querySelector('.cmp-tag')).length")
                check(f'{tag} the answer’s chart holds only the 3 choices that pass and the initial setup; the note counts the other 5', rows == 4 and stray == 0
                      and text(pg, '#pa-chres .cmp-note') == Z('The other 5 choices miss at least one goal and are not shown.', '其余 5 种选择至少差一个目标，图里没有列出。'), (rows, stray, text(pg, '#pa-chres .cmp-note')))
                ttl = text(pg, '#pa-chres .cmp-title')
                check(f'{tag} chart title names the quantity, the conditions and the goal, not “for each choice”', 'Llama' in ttl and '90%' in ttl and Z('for each choice', '每种选择') not in ttl, ttl)
                xt = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-tipx]')].filter(e=>!e.closest('[id$=\"-mathbody\"],[id$=\"-math\"]')).map(e=>e.getAttribute('data-tipx')).filter(t=>t.includes('精确值是')||t.includes('Exactly'))")
                check(f'{tag} no hover outside Show the math gives a number’s exact value (the owner’s clarification of 2026-10-07)', not xt, xt[:2])
                leg = text(pg, '#pa-chres .cmp-leg')
                check(f'{tag} chart legend explains the colours', Z('meets both goals', '两个目标都达到') in leg and Z('misses at least one', '至少差一个') in leg and pg.evaluate("()=>document.querySelectorAll('#pa-chres .cmp-row.ok').length") == 3, leg.replace('\n', ' '))
                tags = pg.evaluate("()=>[...document.querySelectorAll('#pa-chres .cmp-tag')].map(x=>x.textContent)")
                check(f'{tag} chart tags the initial setup', Z('initial setup', '初始设置') in tags, str(tags))
                click(pg, '#pa-ic-B button[data-v="8"]')
                click(pg, '#pa-ic-strat button[data-v="paged"]')
                r = pg.inner_text('#pa-chres')
                check(f'{tag} 8-token blocks fail on table length', ('80.4' in r) and Z('64 entries or fewer', '不超过 64 条') in r, r[:120])
                tg8 = pg.evaluate("()=>[...document.querySelectorAll('#pa-chres .cmp-row')].map(r=>[r.classList.contains('me'),[...r.querySelectorAll('.cmp-tag')].map(t=>t.textContent).join()])")
                check(f'{tag} with 8-token blocks picked the chart adds that row (5 rows) and says its table is over 64', len(tg8) == 5 and any(m and '80.4' in t for m, t in tg8), str(tg8))
                check(f'{tag} status says the table is too long', pg.inner_text('#pa-status') == Z('Not yet: block tables average 80.4 entries per request, over 64', '未达标：每个请求的 block table 平均 80.4 条，超过 64 条'), pg.inner_text('#pa-status'))
                dockfit = "()=>{const d=document.getElementById('dock'), i=d.querySelector('.dock-in'), r=d.querySelector('.pa-dk'); return {hidden:d.hidden,fits:i.scrollWidth<=i.clientWidth+1,uncut:!!r&&r.scrollHeight<=r.clientHeight+1&&r.scrollWidth<=r.clientWidth+1,text:r?r.innerText:''}}"
                if w == 390:
                    d = pg.evaluate(dockfit)
                    check(f'{tag} mini result bar gives the reason, fits and is not cut', not d['hidden'] and d['fits'] and d['uncut'] and d['text'].startswith(Z('block tables average 80.4 entries, over 64', 'block table 平均 80.4 条，超过 64 条')), str(d))
                click(pg, '#pa-ic-B button[data-v="64"]')
                r = pg.inner_text('#pa-chres')
                check(f'{tag} tier 2: met, a smaller change exists', ('达成目标' in r and '改得更少' in r) if zh else ('Goal met' in r and 'could change less' in r), r[:120])
                shot(pg, f'{tag}_ch_tier2', full=(w == 390))
                click(pg, '#pa-ic-B button[data-v="16"]')
                r = pg.inner_text('#pa-chres')
                check(f'{tag} tier 1: met and minimal', Z('as little as possible', '改动最少') in r, r[:120])
                check(f'{tag} tier 1: goal rows stay inside the task card', pg.evaluate(goal_fit))
                check(f'{tag} chart marks start, goal and your pick', pg.evaluate("()=>{const c=document.querySelector('#pa-chres .cmp'); return !!c&&!!c.querySelector('.cmp-you')&&!!c.querySelector('.cmp-tag')&&!!c.querySelector('.cmp-axis')}"))
                check(f'{tag} status goal met', 'ok' in pg.get_attribute('#pa-status', 'class') and pg.inner_text('#pa-status') == Z('Goal met', '达成目标'))
                if w == 390:
                    pg.wait_for_timeout(350)
                    d = pg.evaluate(dockfit)
                    check(f'{tag} mini result bar once met: both numbers, not cut', not d['hidden'] and d['fits'] and d['uncut'] and d['text'].startswith(Z('96.3% stores tokens while requests wait; block tables average 40.4 entries', '排队时 96.3% 存着 token，block table 平均 40.4 条')), str(d))
                    shot(pg, f'{tag}_ch_met_dock')
                shot(pg, f'{tag}_ch_tier1')
                if w == 1440 and scheme == 'light':
                    # formula terms in the challenge: a locked parameter explains its fixed value; block size jumps to the card
                    pg.evaluate("()=>{const d=document.getElementById('pa-math'); d.open=true;}")
                    pg.wait_for_timeout(100)
                    pg.evaluate("()=>document.querySelector('#pa-mathbody .t[data-ctl=\"B\"]').scrollIntoView({block:'center'})")
                    pg.wait_for_timeout(150)
                    pg.click('#pa-mathbody .t[data-ctl="B"]')
                    pg.wait_for_timeout(700)
                    check(f'{tag} challenge: block-size term jumps to the control in the task card', pg.evaluate("()=>!!document.activeElement.closest('#pa-ic-B')"))
                    pg.evaluate("()=>document.querySelector('#pa-mathbody .t[data-ctl=\"n\"]').scrollIntoView({block:'center'})")
                    pg.wait_for_timeout(150)
                    pg.click('#pa-mathbody .t[data-ctl="n"]')
                    pg.wait_for_timeout(200)
                    check(f'{tag} challenge: a locked term says what it is fixed to', Z('Fixed during the challenge', '挑战中固定不变') in tip(pg), tip(pg)[:80])
                    pg.evaluate("()=>document.getElementById('pa-chres').scrollIntoView({block:'start'})")
                    pg.wait_for_timeout(200)
                    shot(pg, f'{tag}_ch_tier1_chart')
                # language switch in the challenge keeps the answer and the feedback
                pg.click(f'.langsw button[data-lang="{other}"]')
                pg.wait_for_timeout(300)
                st = state(pg)
                r2 = pg.inner_text('#pa-chres')
                check(f'{tag} language switch keeps the challenge result', st['mode'] == 'challenge' and st['S']['B'] == 16 and st['chSubmitted'] and (('as little as possible' in r2) if zh else ('改动最少' in r2)), r2[:80])
                pg.click(f'.langsw button[data-lang="{lang}"]')
                pg.wait_for_timeout(300)
                check(f'{tag} no overflow in challenge', overflow(pg) <= 0, str(overflow(pg)))
                click(pg, '#pa-ic-B button[data-v="64"]', wait=60)
                click(pg, '#pa-ic-B button[data-v="16"]', wait=60)
                had = pg.evaluate(badges)
                # --- free explore (a mode button in the header also reveals the rail)
                click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])', wait=60)
                check(f'{tag} a mode switch clears the badges and the dashed line, even when the run is unchanged', any(had) and pg.evaluate(badges) == [''] * 5 and ghost(pg) == '0' and state(pg)['S'] == {'strat': 'paged', 'B': 16, 'mu': 256, 'lim': 1536}, f'{had} {ghost(pg)}')
                in_view(pg, '#pa-stage' if w <= 760 else '#pa-rail', f'{tag} a mode switch brings the rail into view' + (' (phones: the stage)' if w <= 760 else ''), h)
                st = state(pg)
                check(f'{tag} leaving the challenge restores the settings', st['mode'] == 'sandbox' and st['S'] == {'strat': 'paged', 'B': 16, 'mu': 256, 'lim': 1536}, str(st['S']))   # step 7 had set paged, 16
                names = pg.evaluate("()=>({eyebrow:document.querySelector('#pa-rail .railcard .eyebrow').innerText,title:document.querySelector('#pa-rail .rh').innerText,presets:[...document.querySelectorAll('#pa-rail [data-preset] b')].map(x=>x.innerText),settings:document.querySelector('#pa-controls>.eyebrow').innerText,hint:!!document.querySelector('#pa-controls .ctl-hint'),desc:(document.querySelector('#view-dyn [data-flow-nav] button[aria-pressed=true]')||{dataset:{}}).dataset.tipx||''})")
                check(f'{tag} scenarios: the mode, the card title and real workloads', names['eyebrow'].lower() == Z('scenarios', '应用场景') and names['title'] == Z('Pick a scenario', '选一个应用场景')
                      and names['presets'] == Z(['Customer support', 'Chat service', 'Long-form writing', 'API with a large default cap', 'Summaries'], ['客服问答', '聊天服务', '长文写作', 'API 服务，默认上限很大', '摘要']), str(names))
                intro = text(pg, '#pa-rail .railcard p.small')
                check(f'{tag} the scenario intro is a lead-in that says what to pick and what to watch (G1, the owner 2026-10-08)', intro.startswith(Z('Pick a scenario and see how much of the pool stores tokens while requests wait.', '选一个场景，看看有请求排队时，显存池有多少存着 token。')) and Z('do not stack', '不会叠加') not in intro, intro[:60])
                pd = pg.evaluate("()=>[...document.querySelectorAll('#pa-rail [data-pa-pd]')].map(x=>x.innerText)")
                check(f'{tag} every scenario line gives its units', pd[1] == Z('answer lengths drawn at random with a mean of 256 tokens, capped at 1,536 tokens (the initial setup’s workload)', '回答长度按均值 256 个 token 随机抽取，上限 1,536 个 token（和初始设置的负载相同）')
                      and all(Z('tokens', '个 token') in x for x in pd), str(pd[:2]))
                check(f'{tag} the settings panel is called Settings, with no sentence about the page', names['settings'].lower() == Z('settings', '自由调参') and not names['hint'], str(names['settings']))
                check(f'{tag} scenario mode description', names['desc'] == Z('Pick a real workload and see how the three allocation strategies do under it.', '选一种真实的负载，看三种分配方式在它下面表现如何。'), names['desc'])
                pos = pg.evaluate("()=>{const n=document.getElementById('pa-sbnote'), a=n.previousElementSibling, z=n.nextElementSibling; return {prev:a?a.className:'',next:z?(z.id||z.className):'',empty:n.innerText===''}}")
                check(f'{tag} the scenario note sits directly under the scenario list, above the allocation strategy', pos == {'prev': 'presets', 'next': 'pa-ic-strat', 'empty': False}, str(pos))
                # right after the mode switch the list is in view: click as a reader would, without scrolling first
                v0 = pg.evaluate(mvals)
                pg.click('#view-dyn [data-preset="0"]')
                pg.wait_for_timeout(60)
                dl, v1 = pg.evaluate(badges), pg.evaluate(mvals)
                check(f'{tag} scenario badges reproduce from the shown values (block table 40.4 → 30.9 is −24%)', v0[4] == '40.4' and v1[4] == '30.9' and dl[4].startswith('▼ −24%') and not badges_match(v0, v1, dl), str(badges_match(v0, v1, dl) or dl))
                ns = note_on_screen(pg)
                check(f'{tag} after a scenario click the note is on screen', ns['ok'], str(ns))
                check(f'{tag} a scenario that changes the numbers: one rounded share, the decode steps against reserving the maximum in words', text(pg, '#pa-sbnote') == Z('Selected: Customer support. With paging, about 96% of the pool stores tokens while requests wait, so the requests finish in about a third of the decode steps that reserving the maximum needs.',
                      '已选：客服问答。分页时，排队期间显存池约 96% 存着 token，所以跑完这些请求用的 decode 步只有按最大长度预留时的约三分之一。') and pg.evaluate(mvals)[2] == '488', text(pg, '#pa-sbnote'))
                shot(pg, f'{tag}_sb_s0_paged')
                pg.click('#view-dyn [data-preset="3"]')
                ns = note_on_screen(pg)
                check(f'{tag} after a scenario click that changes nothing the note is on screen', ns['ok'], str(ns))
                check(f'{tag} a scenario that changes no number leads with that, says why, and where reserving the maximum differs', text(pg, '#pa-sbnote') == Z('Selected: API with a large default cap. No change from “Customer support”: paging hands out the next block only when the last one is full, and no answer here reaches even the lower cap, so raising the cap changes no number. Reserving the maximum is different: each request then reserves the higher cap, so the run takes many more decode steps.',
                      '已选：API 服务，默认上限很大。和“客服问答”相比没有变化：分页只在写满一个 block 后才分配下一个，这些回答连较低的那个上限都没写到，所以上限提高后，数字一个都没变。按最大长度预留时就不同了：每个请求都按更高的上限占空间，跑完要多得多的 decode 步。'), text(pg, '#pa-sbnote'))
                check(f'{tag} a click that changes no number shows no badge', pg.evaluate(badges) == [''] * 5, str(pg.evaluate(badges)))
                shot(pg, f'{tag}_sb_s3_paged')
                v0 = pg.evaluate(mvals)
                pg.click('#view-dyn [data-preset="1"]')
                pg.wait_for_timeout(60)
                dl, v1 = pg.evaluate(badges), pg.evaluate(mvals)
                check(f'{tag} scenario badges reproduce from the shown values (30.9 → 40.4 is +31%)', dl[4].startswith('▲ +31%') and not badges_match(v0, v1, dl), str(badges_match(v0, v1, dl) or dl))
                ns = note_on_screen(pg)
                check(f'{tag} the note is on screen after the third click too', ns['ok'], str(ns))
                check(f'{tag} the scenario card compares the strategies in place, in sync with Settings', pg.evaluate("()=>[...document.querySelectorAll('#pa-rail .seg-ctl[data-key=strat] button')].length") == 3)
                sl = pg.evaluate("()=>[...document.querySelectorAll('.seg-ctl[data-key=strat]')].filter(g=>g.offsetParent).map(g=>({dir:getComputedStyle(g).flexDirection,labels:[...g.querySelectorAll('button')].map(b=>b.innerText),fit:[...g.querySelectorAll('button')].every(b=>b.scrollWidth<=b.clientWidth+1&&b.getBoundingClientRect().height<48)}))")
                check(f'{tag} the strategy buttons use the names the text uses, stacked so each fits on one line', len(sl) == 2 and all(x['dir'] == 'column' and x['fit'] and x['labels'] == Z(['Reserve the maximum', 'Reserve the exact length', 'Paging'], ['按最大长度预留', '按实际长度预留', '分页']) for x in sl), str(sl))
                click(pg, '#pa-rail .seg-ctl[data-key=strat] button[data-v="max"]')
                check(f'{tag} picking a strategy in the scenario card', state(pg)['S']['strat'] == 'max' and pg.get_attribute('#pa-ctl-strat button[data-v="max"]', 'aria-pressed') == 'true')
                click(pg, '#pa-rail .seg-ctl[data-key=strat] button[data-v="paged"]')
                link = pg.evaluate("()=>{const a=document.querySelector('#pa-rail .railcard p [data-jump=controls]'); return a?a.innerText:''}")
                check(f'{tag} the scenario card names “自由调参” / Settings as a link', link == Z('Settings', '自由调参'), link)
                click(pg, '#pa-rail .railcard p [data-jump=controls]', wait=900)
                y = top_of(pg, '#pa-controls')
                check(f'{tag} that link brings Settings into view', y is not None and -40 <= y <= h * 0.6, str(y))
                steps_of = lambda mu, lim: pg.evaluate(f"()=>window.__pa.stats({{strat:'paged',B:16,mu:{mu},lim:{lim}}}).T")
                sel_typ = pg.inner_text('#pa-sbnote')
                click(pg, '#view-dyn [data-preset="4"]')
                sel_tight = pg.inner_text('#pa-sbnote')
                ns = note_on_screen(pg)
                check(f'{tag} a scenario click far down the page still ends with the note on screen', ns['ok'], str(ns))
                check(f'{tag} the selected line adds one baseline: the same requests reserving the maximum', sel_typ == Z('Selected: Chat service. With paging, about 96% of the pool stores tokens while requests wait, so the requests finish in about a third of the decode steps that reserving the maximum needs.',
                      '已选：聊天服务。分页时，排队期间显存池约 96% 存着 token，所以跑完这些请求用的 decode 步只有按最大长度预留时的约三分之一。'), sel_typ)
                check(f'{tag} the selected line compares the decode steps in words, so similar presets differ; the steps card has the count', Z('about a third of the decode steps', '约三分之一') in sel_typ and Z('about half of the decode steps', '约一半') in sel_tight and sel_typ != sel_tight
                      and pg.evaluate(mvals)[2] == f'{steps_of(256, 512):,}', sel_tight[-90:])
                click(pg, '#view-dyn [data-preset="0"]')
                click(pg, '#view-dyn [data-preset="3"]')
                check(f'{tag} a scenario click that changes no number says so and why', pg.inner_text('#pa-sbnote').startswith(Z('Selected: API with a large default cap. No change from “Customer support”: ', '已选：API 服务，默认上限很大。和“客服问答”相比没有变化：')), pg.inner_text('#pa-sbnote')[:120])
                nocap = Z('Changing the cap changed nothing', '改上限没有变化')
                check(f'{tag} no cap note after picking a scenario', nocap not in pg.inner_text('#pa-sbnote'), pg.inner_text('#pa-sbnote')[-120:])
                pressed = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-preset]')].filter(x=>x.getAttribute('aria-pressed')==='true').map(x=>+x.dataset.preset)")
                check(f'{tag} preset pressed', pressed == [3], str(pressed))
                click(pg, '#pa-ctl-lim button[data-v="4096"]')
                m4k = pg.evaluate("()=>window.__pa.stats({strat:'max',B:16,mu:128,lim:4096}).T")
                m4k = f'{m4k:,}'
                fewer = Z('far fewer', '少得多的') if int(m4k.replace(',', '')) <= 0.5 * 6074 else Z('fewer', '更少的')   # the words the page picks from the two runs' decode steps
                check(f'{tag} a cap change that changes nothing says so and why, and where reserving the maximum differs', pg.inner_text('#pa-sbnote') == Z(f'No scenario is selected: answer lengths with a mean of 128 tokens and a cap of 4,096 tokens match none of them. Changing the cap changed nothing: paging hands out the next block only when the last one is full, and no answer here reaches even the lower cap, so lowering the cap changes no number. Reserving the maximum is different: each request then reserves the lower cap, so the run takes {fewer} decode steps.',
                      f'没有选中场景：回答长度按均值 128 个 token 抽取、上限 4,096 个 token，和哪个场景都不一样。改上限没有变化：分页只在写满一个 block 后才分配下一个，这些回答连较低的那个上限都没写到，所以上限降低后，数字一个都没变。按最大长度预留时就不同了：每个请求都按更低的上限占空间，跑完要{fewer} decode 步。'), pg.inner_text('#pa-sbnote'))
                click(pg, '#pa-ctl-lim button[data-v="8192"]')
                click(pg, '#pa-ctl-strat button[data-v="max"]')
                check(f'{tag} the cap note goes away with the next change', nocap not in pg.inner_text('#pa-sbnote'))
                st = pg.inner_text('#pa-status')
                u = pg.evaluate("()=>{const u=window.__pa.stats({strat:'max',mu:128,lim:8192}).util*100; return Math.round(u*10)/10}")
                check(f'{tag} huge cap with reserve-the-maximum is wasteful', u < 10 and st == Z(f'While requests wait: {u:g}% stores tokens on average', f'排队时平均 {u:g}% 存着 token'), st)
                shot(pg, f'{tag}_sandbox_huge')
                click(pg, '#pa-ctl-mu button[data-v="512"]')
                pressed = pg.evaluate("()=>[...document.querySelectorAll('#view-dyn [data-preset]')].filter(x=>x.getAttribute('aria-pressed')==='true').length")
                check(f'{tag} scenario unpressed when settings leave it', pressed == 0 and Z('No scenario is selected', '没有选中场景') in pg.inner_text('#pa-sbnote'))
                check(f'{tag} no overflow in free explore', overflow(pg) <= 0, str(overflow(pg)))
                shot(pg, f'{tag}_sandbox')
                pg.close()

    # --- hash round-trip
    pg = open_page(b, 1440, 900, 'en')
    click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
    click(pg, '#pa-ctl-strat button[data-v="paged"]')
    click(pg, '#pa-ctl-B button[data-v="32"]')
    click(pg, '#pa-ctl-lim button[data-v="2048"]')
    pg.evaluate("()=>{const i=document.getElementById('pa-st-time-in'); i.value=123; i.dispatchEvent(new Event('input',{bubbles:true}))}")
    pg.wait_for_timeout(150)
    hsh = pg.evaluate("()=>location.hash")
    check('hash encodes the setup', hsh == '#paged-attention~paged.32.256.2048.123', hsh)
    pg2 = b.new_page(viewport={'width': 1440, 'height': 900})
    pg2.on('pageerror', lambda e: errs.append(f'hash: {e}'))
    pg2.goto(URL + hsh)
    pg2.wait_for_timeout(500)
    st = state(pg2)
    check('hash round-trip restores mode, settings and step', st['mode'] == 'sandbox' and st['S'] == {'strat': 'paged', 'B': 32, 'mu': 256, 'lim': 2048} and st['t'] == 123, str(st))
    check('hash round-trip redraws the same pool (decode step numbered from 1)', pg2.inner_text('#pa-st-time .pa-tread').startswith('Decode step 124 of'), pg2.inner_text('#pa-st-time .pa-tread'))
    pg2.close()
    pg.close()
    # --- tooltips, jumps, keyboard (EN and ZH at 1440)
    for lang in ['en', 'zh']:
        zh = lang == 'zh'
        Z = lambda en, zh_: zh_ if zh else en
        pg = open_page(b, 1440, 900, lang)
        click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')     # guided hides the formulas until step 3
        pg.evaluate("()=>document.getElementById('pa-math').open=true")
        pg.wait_for_timeout(100)
        pg.evaluate("()=>document.querySelector('#pa-mathbody .t[data-ctl=\"lim\"]').scrollIntoView({block:'center'})")
        pg.wait_for_timeout(200)      # a scroll hides the tooltip on purpose; hover after it settles
        pg.hover('#pa-mathbody .t[data-ctl="lim"]')
        pg.wait_for_timeout(150)
        tv = tip(pg)
        check(f'{lang} formula term tooltip: source and action', Z('Click to jump', '点击跳到') in tv, tv.replace('\n', ' | ')[:100])
        mb = pg.inner_text('#pa-mathbody')
        check(f'{lang} math: full-width punctuation in Chinese, units on every result', ('（在有请求排队的' in mb and '负载：' in mb and '个 token 的空间（请求 1）；40 个请求平均 1,232.5 个' in mb) if zh else ('(averaged over the' in mb and 'Workload: ' in mb and 'tokens of room (request 1); 1,232.5 on average over the 40 requests' in mb), mb[-200:])
        pg.click('#pa-mathbody .t[data-ctl="lim"]')
        pg.wait_for_timeout(700)
        check(f'{lang} formula term click jumps to the control', pg.evaluate("()=>!!document.activeElement.closest('#pa-ctl-lim')"))
        hover(pg, '#pa-legend [data-jump="hl-intf"]')
        check(f'{lang} legend tooltip explains the state and what the server knows', Z('A real server cannot tell it from “reserved for later tokens”, because it does not know the final length.', '真实服务时不知道请求多长，所以分不清它和“为后面的 token 预留”。') in tip(pg), tip(pg)[-80:])
        hover(pg, '#pa-legend [data-jump="hl-ext"]')
        check(f'{lang} external fragmentation tooltip: while a request waits, no piece big enough', tip(pg).startswith(Z('External fragmentation is room nobody holds while a request is waiting: the request at the head of the queue needs one contiguous piece, and no piece is big enough.', '外部碎片是有请求在排队、却没人占用的空间：排在最前面的请求要一整段连续空间，而每一段都不够大。')), tip(pg)[:100])
        hover(pg, '#pa-legend [data-jump="hl-free"]')
        check(f'{lang} free tooltip: not counted as waste, and when', tip(pg).startswith(Z('Free room is room nobody holds that does not count as waste:', '空闲是没人占用、也不算浪费的空间：')), tip(pg)[:100])
        subj = pg.evaluate("()=>[...document.querySelectorAll('#pa-legend [data-tipx]')].map(e=>e.dataset.tipx)")
        pre = Z(['Stored tokens are', 'This is room', 'Internal fragmentation is', 'External fragmentation is', 'Free room is', 'The thick line marks'], ['存着的 token 是', '这是请求', '内部碎片是', '外部碎片是', '空闲是', '粗竖线标出'])
        check(f'{lang} every legend tooltip opens with its subject', len(subj) == 6 and all(x.startswith(p_) for x, p_ in zip(subj, pre)), str([x[:16] for x in subj]))
        pg.click('#pa-legend [data-jump="hl-intf"]')
        pg.wait_for_timeout(150)
        check(f'{lang} legend click highlights', state(pg)['hl'] == 'intf')
        pg.click('#pa-legend [data-jump="hl-intf"]')
        pg.wait_for_timeout(150)
        check(f'{lang} legend click again shows all', state(pg)['hl'] == 'all')
        # a highlight with nothing to show says why (reserve the maximum, a waiting step: every free slot is external fragmentation)
        pg.click('#pa-legend [data-jump="hl-free"]')
        pg.wait_for_timeout(200)
        check(f'{lang} empty highlight explains itself', Z('Nothing to highlight', '没有可高亮的') in tip(pg), tip(pg)[:100])
        pg.click('#pa-legend [data-jump="hl-free"]')
        pg.wait_for_timeout(150)
        hover(pg, '#pa-metrics [data-k="table"]')
        check(f'{lang} metric tooltip with jump', 'block table' in tip(pg), tip(pg)[:80])
        hover(pg, '#pa-metrics [data-k="steps"]')
        check(f'{lang} steps tooltip: what it counts, and where the timing assumption is explained', tip(pg).startswith(Z('This is the number of decode steps until all 40 requests finish. Every decode step is assumed to take the same time (see the notes at the end of the page for why).', '这个数是 40 个请求全部完成用的 decode 步数。这里假设每个 decode 步耗时相同（原因见页末的“简化、假设和来源”）。')), tip(pg)[:120])
        mt = pg.evaluate("()=>['util','run','unused','table'].map(k=>{const e=document.querySelector(`#pa-metrics [data-k=${k}]`); return window.ASP.t(e.dataset.tip)})")
        check(f'{lang} every metric tooltip opens with its subject', all(x.startswith(Z('This is', '这个数是')) for x in mt), str([x[:12] for x in mt]))
        hover(pg, '#pa-status')
        check(f'{lang} status tooltip names the badge and explains the colours', Z('This badge shows the share of the pool that stores tokens, averaged over the decode steps in which requests wait.', '这个标记显示有请求排队时，显存池平均有多大比例存着 token。') in tip(pg), tip(pg)[:80])
        pg.evaluate("()=>document.getElementById('pa-cv').scrollIntoView({block:'center'})")
        pg.wait_for_timeout(200)
        box = pg.query_selector('#pa-cv').bounding_box()
        pg.mouse.move(box['x'] + 6, box['y'] + 6)
        pg.wait_for_timeout(200)
        tv = tip(pg)
        check(f'{lang} pool hover names the request and whose answer', tv.startswith(Z('Request 1: a ', '请求 1：prompt ')) and Z('its answer will be', '回答最终会生成') in tv and Z('generated so far', '目前已生成') in tv, tv.replace('\n', ' | ')[:160])
        shot(pg, f'1440_{lang}_pool_hover')
        pg.mouse.move(5, 5)
        pg.focus('#pa-cv')
        pg.wait_for_timeout(150)
        t1 = tip(pg)
        pg.keyboard.press('ArrowDown')
        pg.keyboard.press('ArrowDown')
        pg.wait_for_timeout(150)
        check(f'{lang} keyboard on the pool shows cells', t1 != '' and tip(pg) != '')
        click(pg, '#pa-ctl-strat button[data-v="paged"]')
        pg.evaluate("()=>document.getElementById('pa-legend').scrollIntoView({block:'center'})")
        pg.wait_for_timeout(200)
        pg.click('#pa-legend [data-jump="hl-ext"]')
        pg.wait_for_timeout(200)
        check(f'{lang} paging has no external fragmentation to highlight', Z('paging never leaves external fragmentation', '分页不会产生外部碎片') in tip(pg), tip(pg)[:100])
        pg.locator('#paged-attention-fid').scroll_into_view_if_needed()
        pg.wait_for_timeout(150)
        fid = pg.inner_text('#paged-attention-fid')
        check(f'{lang} notes at the page end: assumed, sources, free queue, timing, seeds', all(s in fid.lower() for s in (['假设', '来源', '先进先出', '2.76', '2.88', '14 gib', '188'] if zh else ['assumed', 'sources', 'first in, first out', '2.76', '2.88', '14 gib', '188']))
              and pg.evaluate("()=>!!document.querySelector('#paged-attention-fid a[href*=\"2309.06180\"]')") and 'block_pool.py' in fid, fid[:120])
        shot(pg, f'1440_{lang}_fidelity')
        copy_rules(pg, f'1440_{lang} fidelity card', fid)
        # the 前置知识 popover went with the header buttons (owner, 2026-10-11 19:28 UTC)
        click(pg, '#view-dyn [data-act="drill-show"]')
        copy_rules(pg, f'1440_{lang} connect and drill answer', text(pg, '#view-dyn .connect'))
        pg.close()
    # the run ends while part 3's card is off screen, and Show me: plays at Fast to the last decode step, then the result appears
    pg = open_page(b, 390, 844, 'en')
    click(pg, '#view-dyn [data-go="2"]')
    pg.evaluate("()=>document.getElementById('pa-cv').scrollIntoView({block:'center'})")
    pg.wait_for_timeout(300)
    pg.evaluate("()=>{const i=document.getElementById('pa-st-time-in'); i.value=i.max; i.dispatchEvent(new Event('input',{bubbles:true}))}")
    pg.wait_for_timeout(300)
    dn = text(pg, '#pa-done')
    check('390 run finished off screen: a line leads back to part 3', 'Run finished: the result is in part 3' in dn, dn)
    click(pg, '#pa-done')
    in_view(pg, '#view-dyn [data-obs]', '390 that line brings the result into view', 844)
    pg.close()
    pg = open_page(b, 1440, 900, 'en')
    click(pg, '#view-dyn [data-go="2"]')
    click(pg, '#view-dyn [data-nav="showme"]')
    pg.wait_for_timeout(1500)
    st = state(pg)
    spd = pg.evaluate("()=>[...document.querySelectorAll('.seg-ctl[data-key=speed] button[aria-pressed=true]')].map(x=>x.dataset.v)")
    check('Show me plays the run fast and leaves the learner\'s speed on Medium', st['playing'] and 300 < st['t'] < 2808 and obs(pg) == '' and st['speed'] == 'med' and spd == ['med', 'med'], f"{st['t']} {st['speed']} {spd}")
    for _ in range(40):
        if state(pg)['t'] >= 2808:
            break
        pg.wait_for_timeout(500)
    pg.wait_for_timeout(300)
    st = state(pg)
    check('Show me reaches the last step and shows the result (rounded share in the text, the exact 24.8% on the card)', st['t'] == 2808 and not st['playing'] and 'about 25%' in obs(pg) and pg.evaluate(mvals)[0] == '24.8%' and next_disabled(pg) is False, f"{st['t']} {pg.evaluate(mvals)[0]}")
    set_time(pg, 0)
    click(pg, '#view-dyn #pa-ic-time [data-pa="play"]', wait=0)
    t0 = state(pg)['t']
    pg.wait_for_timeout(1000)
    t1 = state(pg)['t']
    click(pg, '#view-dyn #pa-ic-time [data-pa="play"]')
    check('after Show me, Play runs at the learner\'s Medium speed again', 60 <= t1 - t0 <= 160 and state(pg)['speed'] == 'med', f'{t1 - t0} decode steps in 1 s')
    pg.close()
    # reduced motion: Play jumps to the end state
    pg = open_page(b, 1440, 900, 'en', reduced_motion='reduce')
    click(pg, '#view-dyn [data-go="2"]')
    click(pg, '#view-dyn #pa-ic-time [data-pa="play"]')
    st = state(pg)
    check('reduced motion: Play jumps to the last step and shows the result (rounded share in the text, the exact 24.8% on the card)', st['t'] == 2808 and not st['playing'] and 'about 25%' in obs(pg) and pg.evaluate(mvals)[0] == '24.8%', f"{st['t']} {pg.evaluate(mvals)[0]}")
    pg.close()
    # animation runs and stops
    pg = open_page(b, 1440, 900, 'en')
    click(pg, '#view-dyn [data-go="2"]')
    click(pg, '#view-dyn #pa-ic-time [data-pa="play"]')
    pg.wait_for_timeout(700)
    st = state(pg)
    check('Play advances time', st['playing'] and st['t'] > 0, str(st['t']))
    click(pg, '#view-dyn #pa-ic-time [data-pa="play"]')
    t_stop = state(pg)['t']
    pg.wait_for_timeout(300)
    check('Pause stops time', state(pg)['t'] == t_stop and not state(pg)['playing'])
    pg.close()
    # a changed setup names “自由调参” / Settings as a link; the decode-steps metric jumps to the notes at the page end
    for lang in ['en', 'zh']:
        zh = lang == 'zh'
        Z = lambda en, zh_: zh_ if zh else en
        pg = open_page(b, 1440, 900, lang)
        click(pg, '#view-dyn [data-go="4"]')
        click(pg, '#view-dyn [data-lever="st-paged"]')
        click(pg, '#pa-ctl-B button[data-v="32"]')
        o = obs(pg)
        check(f'{lang} a changed setup refers to Settings by a link', o.split('\n', 1)[-1].strip().startswith(Z('You changed the setup under Settings.', '你在“自由调参”里改了设置。'))
              and pg.evaluate("()=>{const a=document.querySelector('#view-dyn [data-obs] [data-jump=controls]'); return a?a.innerText:''}") == Z('Settings', '自由调参'), o[:120])
        hover(pg, '#pa-metrics [data-k="steps"]')
        check(f'{lang} the decode-steps tooltip ends with where the notes are', tip(pg).endswith(Z('Click to jump to the notes at the end of the page.', '点击跳到页末的“简化、假设和来源”。')), tip(pg)[-80:])
        click(pg, '#pa-metrics [data-k="steps"]', wait=900)
        check(f'{lang} clicking it brings the notes at the page end on screen', pg.evaluate("()=>{const r=document.getElementById('paged-attention-fid').getBoundingClientRect(); return r.top<innerHeight-40&&r.bottom>60}"))
        pg.close()
    # playback speed: three speeds, pressed in both copies, a tooltip, and a change while playing takes effect at once (time-based)
    for lang in ['en', 'zh']:
        zh = lang == 'zh'
        Z = lambda en, zh_: zh_ if zh else en
        pg = open_page(b, 1440, 900, lang)
        click(pg, '#view-dyn [data-go="2"]')
        pressed = lambda: pg.evaluate("()=>[...document.querySelectorAll('.seg-ctl[data-key=speed]')].map(g=>[...g.querySelectorAll('button')].filter(x=>x.getAttribute('aria-pressed')==='true').map(x=>x.dataset.v).join())")
        check(f'{lang} speed: three speeds in the card and in the stage', [pg.evaluate(f"()=>[...document.querySelectorAll('#{i} .seg-ctl[data-key=speed] button')].map(x=>x.textContent)") for i in ('pa-ic-time', 'pa-st-time')]
              == [Z(['Slow', 'Medium', 'Fast'], ['慢', '中', '快'])] * 2)
        check(f'{lang} speed: Medium by default, pressed in both places', pressed() == ['med', 'med'] and state(pg)['speed'] == 'med', str(pressed()))
        hover(pg, '#pa-ic-time .pa-speed-l')
        check(f'{lang} speed: tooltip', tip(pg) == Z('Slow plays 25 decode steps per second, Medium 100 and Fast 400.', '慢速每秒播放 25 个 decode 步，中速 100 个，快速 400 个。'), tip(pg))
        # R3.20 (owner, 2026-10-04): nothing replays the run on its own; Play is its replay control, and its tooltip says so
        hover(pg, '#pa-ic-time [data-pa="play"]')
        check(f'{lang} Play: its tooltip says that at the last decode step it starts the run again (R3.20)', tip(pg) == Z('Plays the run from the decode step shown. At the last decode step, Play starts the run again from decode step 1.', '从当前显示的 decode 步开始播放；停在最后一个 decode 步时，点“播放”会从第 1 个 decode 步重新开始。'), tip(pg))
        click(pg, '#pa-ic-time .seg-ctl[data-key=speed] button[data-v="slow"]')
        check(f'{lang} speed: Slow pressed in both places', pressed() == ['slow', 'slow'] and state(pg)['speed'] == 'slow', str(pressed()))
        click(pg, '#view-dyn #pa-ic-time [data-pa="play"]', wait=0)
        t0 = state(pg)['t']
        pg.wait_for_timeout(1000)
        t1 = state(pg)['t']
        click(pg, '#pa-st-time .seg-ctl[data-key=speed] button[data-v="fast"]', wait=0)
        st_ = state(pg)
        pg.wait_for_timeout(1000)
        st2 = state(pg)
        slow_, fast_ = t1 - t0, st2['t'] - st_['t']
        check(f'{lang} speed: a change while playing keeps playing and takes effect at once', st_['playing'] and st2['playing'] and pressed() == ['fast', 'fast'] and 8 <= slow_ <= 45 and fast_ >= 6 * slow_, f'slow {slow_} in 1 s, fast {fast_} in 1 s')
        click(pg, '#view-dyn #pa-ic-time [data-pa="play"]')
        shot(pg, f'1440_{lang}_speed')
        pg.close()
    # R3.20 (owner, 2026-10-04): part 3's run, already taken to its last decode step in Scenarios, shows its whole-run result at once
    # in part 3 and Next is open; a different run watched there does not count, and part 3 still waits for its own
    for lang in ['en', 'zh']:
        pg = open_page(b, 1440, 900, lang, hash_='paged-attention~paged.16.256.1536.0')
        def watch():
            click(pg, '#pa-st-time .seg-ctl[data-key=speed] button[data-v="fast"]', wait=0)
            click(pg, '#pa-st-time [data-pa="play"]', wait=300)
            for _ in range(80):
                if not state(pg)['playing']: break
                pg.wait_for_timeout(200)
        watch()
        click(pg, '#view-dyn .modes button[data-mode="guided"]'); click(pg, '#view-dyn [data-go="2"]')
        st = state(pg)
        check(f'{lang} R3.20 part 3: after another run was watched in Scenarios, part 3 still waits for its own run', st['step'] == 2 and st['t'] == 0 and not st['playing'] and next_disabled(pg) is True and not obs(pg), str(st)[:200])
        click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
        st = state(pg)
        check(f'{lang} R3.20 Scenarios: part 3’s run (each request reserves the maximum), nothing playing', st['mode'] == 'sandbox' and st['S']['strat'] == 'max' and not st['playing'], str(st)[:200])
        watch()
        click(pg, '#view-dyn .modes button[data-mode="guided"]', wait=300)
        st = state(pg)
        check(f'{lang} R3.20 part 3: the same run already watched to its last decode step in Scenarios shows its whole-run result at once, nothing plays, Next is open', st['step'] == 2 and not st['playing'] and next_disabled(pg) is False and bool(obs(pg)), str(st)[:200])
        pg.close()
    # other widths: no overflow in any mode, stage and rail side by side above 760
    for (w, hh) in [(1280, 800), (900, 900), (720, 900)]:
        pg = open_page(b, w, hh, 'zh')
        click(pg, '#view-dyn [data-go="4"]')
        click(pg, '#view-dyn [data-lever="st-paged"]')
        ov = [overflow(pg)]
        shot(pg, f'{w}_zh_step5')
        click(pg, '#view-dyn .modes button[data-mode="challenge"]')
        click(pg, '#view-dyn [data-act="submit"]')
        ov.append(overflow(pg))
        shot(pg, f'{w}_zh_challenge')
        click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
        ov.append(overflow(pg))
        check(f'{w} no overflow in all modes', max(ov) <= 0, str(ov))
        if w > 760:
            same_row = pg.evaluate("()=>{const r=document.getElementById('pa-rail').getBoundingClientRect(), s=document.getElementById('pa-stage').getBoundingClientRect(); return s.left>r.left+50}")
            check(f'{w} steps beside the stage', same_row)
        pg.close()
    # home tile teaser: the spec's label, options, caption, legend and sentences; notes behind the shared button; the Formula chip
    SPEC = {'en': dict(label='How each request’s KV cache is stored', opts=['Reserve the maximum', '16-token blocks'],
                       cap='8 requests, one bar each; the pool holds 12,288 tokens', leg=['tokens it ends up storing', 'reserved, never used', 'does not fit, waits'],
                       max='Each request reserves room for its prompt plus the longest answer allowed: not all 8 requests fit, and 74% of the reserved room is never used.',
                       paged='With 16-token blocks, handed out one at a time as they fill: every request fits, and 99% of the room they hold stores tokens.',
                       notes=['The pool: GPU memory set aside for the KV cache, room for 12,288 tokens (128 KiB per token for Llama-3.1-8B, 1.5 GiB in total), as in the lab.',
                              'These are the first 8 requests of the lab’s workload: prompts of 100 to 600 tokens, answers of at most 1,536 tokens. Each bar is drawn at the request’s final length.',
                              'The vLLM paper measured that in earlier systems only 20.4% to 38.2% of KV cache memory actually held tokens (Kwon et al., 2023).']),
            'zh': dict(label='每个请求的 KV cache 怎么存', opts=['按最大长度预留', '16 个 token 一个 block'],
                       cap='8 个请求，每条是一个请求；显存池能放 12,288 个 token', leg=['最终存着的 token', '预留了但没用上', '放不下，排队'],
                       max='每个请求都按 prompt 加回答上限预留空间：显存池放不下全部 8 个请求，预留的空间有 74% 到结束也没用上。',
                       paged='按 16 个 token 一个 block，写满一个再分配下一个：每个请求都放得下，占用的空间有 99% 存着 token。',
                       notes=['显存池：一张 GPU 留给 KV cache 的显存，能放 12,288 个 token（Llama-3.1-8B 每个 token 128 KiB，共 1.5 GiB），和 Lab 里一样。',
                              '这 8 个请求是 Lab 负载里的前 8 个：prompt 100 到 600 个 token，回答最多 1,536 个 token。每条按请求结束时的长度画。',
                              'vLLM 论文测得，以前的系统里 KV cache 显存只有 20.4% 到 38.2% 真正存着 token（Kwon 等，2023）。'])}
    for (w, lang, scheme) in [(1440, 'en', 'light'), (390, 'zh', 'light'), (1440, 'zh', 'light'), (1440, 'en', 'dark'), (390, 'en', 'dark'), (390, 'zh', 'dark')]:
        zh = lang == 'zh'
        Z = lambda en, zh_: zh_ if zh else en
        sp = SPEC[lang]
        tg = f'teaser {w} {lang} {scheme}'
        pg = open_page(b, w, 900, lang, hash_='map', color_scheme=scheme)
        pg.evaluate("()=>document.getElementById('tile-paged-attention').scrollIntoView({block:'center'})")
        pg.wait_for_timeout(200)
        read1 = pg.inner_text('#tile-paged-attention .tz-read')
        check(f'{tg}: chip says Formula', pg.inner_text('#tile-paged-attention .tz-kind') == Z('Formula', '公式计算'))
        parts = pg.evaluate("()=>{const t=document.getElementById('tile-paged-attention'); return {label:t.querySelector('.tz .lbl').textContent,opts:[...t.querySelectorAll('.tz .seg-ctl button')].map(x=>x.textContent),cap:t.querySelector('.tz-pa-cap').textContent,leg:[...t.querySelectorAll('.tz-leg>span')].map(x=>x.textContent)}}")
        check(f'{tg}: label, options, caption and legend as specified', parts == {'label': sp['label'], 'opts': sp['opts'], 'cap': sp['cap'], 'leg': sp['leg']}, str(parts))
        check(f'{tg}: the reserve-the-maximum sentence is the spec\'s', read1 == sp['max'], read1)
        check(f'{tg}: legend with the waiting item in reserve-max mode', pg.evaluate("()=>[...document.querySelectorAll('#tile-paged-attention .tz-leg>span')].filter(x=>x.offsetParent).length") == 3)
        pg.click('#tile-paged-attention .seg-ctl button[data-v="paged"]')
        pg.wait_for_timeout(150)
        read2 = pg.inner_text('#tile-paged-attention .tz-read')
        check(f'{tg}: the 16-token-block sentence is the spec\'s', read2 == sp['paged'], read2)
        check(f'{tg}: legend hides the waiting item when all fit', pg.evaluate("()=>[...document.querySelectorAll('#tile-paged-attention .tz-leg>span')].filter(x=>x.offsetParent).length") == 2)
        check(f'{tg}: pressed state', pg.get_attribute('#tile-paged-attention .seg-ctl button[data-v="paged"]', 'aria-pressed') == 'true' and pg.get_attribute('#tile-paged-attention .seg-ctl button[data-v="max"]', 'aria-pressed') == 'false')
        notes = pg.evaluate("()=>[...document.querySelectorAll('#tile-paged-attention .tz-note')].map(x=>({shown:!!x.offsetParent,links:x.querySelectorAll('a,button,[data-tipx]').length,text:x.textContent}))")
        check(f'{tg}: three notes as specified, hidden on the tile, plain text', [n['text'] for n in notes] == sp['notes'] and not any(n['shown'] or n['links'] for n in notes), str(notes)[:200])
        check(f'{tg}: no “Assumptions & sources” on the tile (owner 2026-10-13: the tile is one button, no “Assumptions & sources” on it)', pg.evaluate("()=>!document.querySelector('#tile-paged-attention .tz-more')&&!/假设和来源|Assumptions (&|and) sources/.test(document.querySelector('#tile-paged-attention').innerText)"))
        hgt = pg.evaluate("()=>document.querySelector('#tile-paged-attention .tz-pa-rows').getBoundingClientRect().height")
        check(f'{tg}: visual at most 130 px tall', hgt <= 130, str(hgt))
        check(f'{tg}: no overflow', overflow(pg) <= 0)
        tile_txt = pg.evaluate("()=>{const t=document.getElementById('tile-paged-attention'); return t.innerText+' '+[...t.querySelectorAll('.tz-note')].map(x=>x.textContent).join(' ')+' '+t.querySelector('.tz-vis').getAttribute('aria-label')}")
        copy_rules(pg, tg, tile_txt + ' ' + read1)
        pg.mouse.move(2, 2)
        pg.wait_for_timeout(100)
        pg.locator('#tile-paged-attention').screenshot(path=str(SH / f'teaser_{w}_{lang}_{scheme}_paged.png'))
        pg.click('#tile-paged-attention .seg-ctl button[data-v="max"]')
        pg.wait_for_timeout(150)
        pg.locator('#tile-paged-attention').screenshot(path=str(SH / f'teaser_{w}_{lang}_{scheme}_max.png'))
        pg.close()
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
        _pg.evaluate("()=>{location.hash='paged-attention'}"); _pg.wait_for_timeout(500)
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


# polish round 5 (2026-10-17, R3.2, R9.7, R7.9, R7.12, the owner's B4; as tests/moe_flow.py and tests/grpo_flow.py): the reader's walk on
# the whole site (built into out/ and wrapped with a doctype; the one-lab page lacks the “Read first” line, which moves every card 28 px),
# in the lab's real layout (window.__aspPrimer, as tests/kit_flow.py), entering each part as the polish probe stage.py does: Next when it
# is on, else the part's title. Before, at 1440 × 900 the stage scrolled inside in English at parts 2–7 (35, 35, 68, 68, 29, 29 px) and
# had 18 px to spare in Chinese at parts 4–7; entering part 7 in Chinese pushed it up to −91 px. Now, at every part in both languages:
# the stage's top at its sticky line (70 px then; 12 px since refit step 2, where the top bar scrolls away on a desktop lab page) or
# lower, its head below the top bar, no scrolling inside, at least 30 px to spare in Chinese
# (none needed in English), and the pool (the main figure) whole on the screen. Spare = the stage's max-height minus what its content
# and borders need. Part 1: the pool is drawn for the stage's width as it is entered (it was drawn for a 554 px stage and stretched,
# then grew 69 px at the first hover: R7.9), so hovering it leaves the stage's height alone; on this wide stage a cell is 0.8 as tall
# as it is wide (paged-attention.css), square on a phone.
subprocess.run([sys.executable, str(P / 'build.py'), 'out/site-paged-attention.html'], check=True, cwd=P, capture_output=True)
(P / 'out' / 'wrapped-site-paged-attention.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>'
                                                       + (P / 'out' / 'site-paged-attention.html').read_text() + '</body></html>')
SITE = 'file://' + str(P / 'out' / 'wrapped-site-paged-attention.html')
WALK = """()=>{const s=document.querySelector('#pa-stage'), r=s.getBoundingClientRect(), bar=document.querySelector('.topbar').getBoundingClientRect().bottom, cs=getComputedStyle(s),
  cur=document.querySelector('#pa-rail .steps>.step.current'), h=s.querySelector('.stage-head').getBoundingClientRect(), f=document.getElementById('pa-cv').getBoundingClientRect();
  return {part:cur?[...cur.parentNode.children].indexOf(cur)+1:0, top:Math.round(r.top), head:Math.round(h.top), bar:Math.round(bar), scroll:s.scrollHeight-s.clientHeight,
    spare:Math.round(parseFloat(cs.maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), fig:[Math.round(f.top),Math.round(f.bottom)], vh:innerHeight,
    stick:cs.position==='sticky'?Math.round(parseFloat(cs.top)):70};}"""
POOL = """()=>{const c=document.getElementById('pa-cv'), s=document.getElementById('pa-stage'); return {drawn:Math.round(c.width/(devicePixelRatio||1)), w:c.clientWidth, h:c.clientHeight, stage:s.offsetHeight}}"""
with sync_playwright() as _p:
    _b = _p.chromium.launch()
    for _lang in ('en', 'zh'):
        _pg = _b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); _pg.add_init_script('window.__aspPrimer=true')
        _pg.on('pageerror', lambda e: errs.append(f'1440 walk: {e}'))
        _pg.goto(SITE); _pg.wait_for_timeout(600); _pg.click(f'.langsw button[data-lang="{_lang}"]'); _pg.wait_for_timeout(80)
        _pg.evaluate("()=>{location.hash='paged-attention'}"); _pg.wait_for_timeout(500)
        _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
        if _g.count(): _g.first.click(); _pg.wait_for_timeout(700)
        _n = _pg.evaluate("()=>document.querySelectorAll('#pa-rail .steps>.step').length"); _rows = [_pg.evaluate(WALK)]
        _a = _pg.evaluate(POOL); _bx = _pg.query_selector('#pa-cv').bounding_box()
        _pg.mouse.move(_bx['x'] + 30, _bx['y'] + 6); _pg.wait_for_timeout(250); _c = _pg.evaluate(POOL); _pg.mouse.move(2, 2); _pg.wait_for_timeout(200)
        _cs = (_a['w'] - 47 * 2) / 48
        check(f'1440 {_lang} part 1: the pool is drawn for the stage’s width, its cells 0.8 as tall as they are wide, and hovering it leaves the stage’s height alone (R7.9)',
              _a['drawn'] == _a['w'] and abs(_a['h'] - (16 * 0.8 * _cs + 15 * 2)) <= 1 and _c['h'] == _a['h'] and _c['stage'] == _a['stage'], (_a, _c))
        for _i in range(1, _n):
            _nb = _pg.locator('.pbar [data-nav=next]:visible:not([disabled])')
            if _nb.count(): _nb.first.click()
            else: _pg.evaluate(f"()=>document.querySelector('#pa-rail .steps>.step:nth-child({_i + 1}) .step-h').click()")
            _pg.wait_for_timeout(1500); _rows.append(_pg.evaluate(WALK))
        check(f'1440 {_lang} walk: entering every part, the stage’s top at its sticky line or lower and its head below the top bar (R3.2, R9.7, B4)',
              len(_rows) == _n == 7 and all(r['part'] == i + 1 and r['top'] >= max(r['stick'] - 2, r['bar'] - 1) and r['head'] >= r['bar'] - 1 for i, r in enumerate(_rows)), [(r['part'], r['top']) for r in _rows])
        for _r in _rows:
            check(f'1440 {_lang} walk part {_r["part"]}: the stage needs no scrolling inside' + (', 30 px to spare' if _lang == 'zh' else '') + ', the pool whole on the screen (B4, R7.12)',
                  _r['scroll'] <= 0 and _r['spare'] >= (30 if _lang == 'zh' else 0) and _r['fig'][0] >= _r['bar'] - 1 and _r['fig'][1] <= _r['vh'] + 1, _r)
        # part 7, whose stage the bench's end pushed up before (−91 px in Chinese, 65 in English): keepStage brings it back. Since the
        # 2026-10-22 refit's shorter lab head the reveal often has no need to scroll here, so the stage may stay below its sticky line;
        # it must never be above it, and the pool must be whole on the screen
        _r = _rows[6] if len(_rows) >= 7 else {}
        check(f'1440 {_lang} walk part 7: the stage’s top at its sticky line or lower and the pool whole on the screen (it was at ' + ('−91' if _lang == 'zh' else '65') + ' px before keepStage)',
              bool(_r) and _r['top'] >= _r['stick'] - 2 and _r['fig'][0] >= _r['bar'] - 1 and _r['fig'][1] <= _r['vh'] + 1, _r)
        _pg.close()
    # a phone keeps square cells (the layout at 390 is unchanged)
    _pg = _b.new_page(viewport={'width': 390, 'height': 844}, reduced_motion='reduce'); _pg.add_init_script('window.__aspPrimer=true')
    _pg.goto(SITE); _pg.wait_for_timeout(600); _pg.evaluate("()=>{location.hash='paged-attention'}"); _pg.wait_for_timeout(600)
    _g = _pg.locator('[data-flow-nav] button[data-flow=guided]:visible')
    if _g.count(): _g.first.click(); _pg.wait_for_timeout(700)
    _a = _pg.evaluate(POOL); _cs = (_a['w'] - 31) / 32
    check('390 part 1: the pool keeps square cells on a phone', _a['drawn'] == _a['w'] and abs(_a['h'] - (24 * _cs + 23)) <= 1, _a)
    _pg.close()
    _b.close()

for line in out:
    print(line)
real_errs = [e for e in errs if 'ERR_TUNNEL' not in e]
print('page errors:', real_errs)
print(f'{len(out) - len(fails)} passed, {len(fails)} failed')
sys.exit(1 if fails or real_errs else 0)
