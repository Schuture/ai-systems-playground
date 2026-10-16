"""Flow test for the admission lab (capacity capstone), in English and Chinese, at 1440 and 390 wide. Filters: en|zh, 1440|390
(python3 tests/admission_flow.py zh 390).
- the guided parts in order; Next waits with the kit's why line where a part asks for a choice (parts 2-7); the prediction gate
  (part 4: pick, then Show the result, which locks the answer);
- R3.20: part 1 plays nothing; a new choice plays its run, the same choice or one already watched replays nothing; Back and Next
  again replay nothing; Replay plays from the start; Play pauses and resumes; the speed control changes the speed;
- part 1 states the goals and the service targets in the text, and its stage title names no rule yet;
- each part's observation says what happened (4 GPUs: above capacity; 110%; the 10-minute queue; the rate limit; paid first, with free
  requests getting 429; preemption; θ);
- part 8: what to remember, misconceptions, real systems, the link to the scenarios;
- Scenarios: five, each shows its insight; a new one plays, one already watched does not; the hash round trip;
- the challenge: starts at 6 GPUs with no rules; it misses, says why and offers Show the answer (chart and why); 5 GPUs with the
  rate limit, paid first and θ = 100% but no free queue limit misses (above capacity at the peak), and the stage says so; with a free
  queue limit of 200 it is met on the fewest GPU hours; 6 GPUs the same is told 5 are enough; the exits; no company name in the exits;
- the mini result bar (dock) at 390 when the stage is off screen;
- the home tile: the switch changes the sentence and the chart, chip Toy run, picture at most 130 px;
- no horizontal page scroll, no page errors, the text rules (no “toy”, no “一次只能选 / 不会叠加 / 第 N 部分里”, no 万/亿, no two
  colons in one sentence).
Run: python3 tests/admission_flow.py [en|zh] [1440|390]"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
(P / 'out' / 'admission').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/admission.html', '--only', 'admission'], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / 'admission' / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'admission.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'admission' / 'flow-wrapped.html')
SH = P / 'shots' / 'admission'; SH.mkdir(parents=True, exist_ok=True)
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


# polish round 3 (2026-10-16, R7.12, the owner's B4; as tests/data-pipeline_flow.py part 4): on entering a listed part at 1440 × 900 the
# stage needs no scrolling inside itself and could still grow by 30 px in Chinese (its max-height minus the height its content and borders
# need; in English it only needs no scrolling), and the part's figures lie inside it and on the screen (below the top bar and the mini
# result bar, above the window's bottom)
FIT = """([st,sels])=>{const s=document.querySelector(st), r=s.getBoundingClientRect(), tb=document.querySelector('.topbar'), dk=document.querySelector('.dock');
 const top=Math.max(tb?tb.getBoundingClientRect().bottom:0, dk&&getComputedStyle(dk).display!=='none'?dk.getBoundingClientRect().bottom:0);
 const figs=sels.map(x=>{const e=document.querySelector(x); if(!e||!e.getClientRects().length) return null; const q=e.getBoundingClientRect(); return [Math.round(q.top),Math.round(q.bottom)];});
 return {spare:Math.round(parseFloat(getComputedStyle(s).maxHeight)-(s.scrollHeight+s.offsetHeight-s.clientHeight)), scroll:s.scrollHeight-s.clientHeight, top:Math.round(r.top), figs,
   inside:figs.every(q=>q&&q[0]>=r.top-1&&q[1]<=r.bottom+1), shown:figs.every(q=>q&&q[0]>=top-1&&q[1]<=innerHeight+1)};}"""


def fits(pg, tag, part, lang, sels):
    f = pg.evaluate(FIT, ['#adm-stage', sels])
    check(f'{tag} part {part}: the stage fits without scrolling inside' + (', 30 px to spare' if lang == 'zh' else '') + ', its figures inside it and on the screen (R7.12, B4)',
          f['scroll'] <= 0 and f['spare'] >= (30 if lang == 'zh' else 0) and f['inside'] and f['shown'], f)


def page(b, w, h, lang, hash_='admission'):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(500)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__adm.state()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' '):''}", sel)
cur = lambda pg: text(pg, '#adm-rail .step.current')
next_dis = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#adm-rail .step.current [data-nav=\"next\"]'); return b?b.disabled:null}")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#adm-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent)")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#adm-run .seg-ctl[data-key=\"spd\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait)


def finish(pg):
    """scrub the run to its end, as a reader can with the progress bar"""
    pg.evaluate("()=>{const r=document.querySelector('#adm-prog'); r.value='1000'; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true}));}")
    pg.wait_for_timeout(150)


def nav(pg, which):
    click(pg, f'#adm-rail .step.current [data-nav="{which}"]', 350)


def mode(pg, m):
    click(pg, f'#adm-app .modes button[data-mode="{m}"]', 400)


RULES = """(sel)=>{const root=document.querySelector(sel); if(!root) return [];
 const out=[], tw=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
 const bad=[[/ · (?![0-9])/,'“·” joining parts'],[/起点/,'起点'],[/starting point/i,'starting point'],[/控件/,'控件'],[/\\bControls\\b/,'Controls'],
   [/mini result bar|迷你结果条/i,'sentence about the mini result bar'],[/\\bhover\\b|悬停/i,'hover instruction'],
   [/in this toy\\b|在这个玩具里|\\btoy\\b(?! run)|玩具(?!实跑)/i,'“toy”'],[/一次只能选|不会叠加|第 [0-9一二三四五六七八] 部分里/,'banned phrase'],
   [/updates as you|随之更新|会立刻更新|updates at once/i,'sentence about the page updating'],[/\\bNaN\\b|undefined|\\[object|Infinity/,'broken value'],
   [/亿|万/,'亿 or 万'],[/(?<![这那两])个数(?![值量])/,'“个数”'],
   [/深色|浅色|浅灰|深灰|\\b(?:dark|light|pale)(?:er)? (?:bar|line|block|part|cell)/i,'colour named by lightness'],
   [/不是哪家|不是实测|不是测出来|不是对真实|假设，不是|自己的假设|不是真实录制|不是真实 agent|不是真的分词器|not a vendor|not a quote|, not a measurement|its own assumption|not a recording|not a real tokenizer/i,'negative disclaimer (the owner, 2026-10-14: state the assumption positively)'],[/SDE → LLM Serving|学习路线|learning path/i,'a learning path the site no longer has (the owner, 2026-10-14)']];
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
    for sel in ('#adm-app', '#tile-admission'):
        for h in pg.evaluate(RULES, sel): rule_hits.append(f'{where}: {h}')
        for h in pg.evaluate(COLONS, sel): rule_hits.append(f'{where}: two colons: {h}')
    ov = overflow(pg); check(f'{where}: no horizontal page scroll', ov <= 0, ov)




def lever(pg, v, wait=300):
    click(pg, f'#adm-rail .step.current [data-lever="{v}"]', wait)


def guided(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    title = text(pg, '#adm-app .labtitle')
    check(f'{tag} title', title == Z(lang, 'How many GPUs does your traffic need, and who should wait?', '你的流量要几张卡，谁该排队？'), title)
    # the owner (2026-10-14): assumptions stated positively, “本 Lab 假设每 GPU 小时 3 美元”, in the GPU-hour tooltip and in the end notes
    tips = pg.evaluate("()=>[...document.querySelectorAll('#adm-app [data-tipx]')].map(e=>e.getAttribute('data-tipx')).join('\\n')")
    ends = pg.evaluate("()=>{const v=document.querySelector('#view-dyn:not([hidden])')||document.querySelector('#view-lab:not([hidden])'); const e=v&&v.querySelector('[data-endnotes]'); return e?e.textContent:''}")
    check(f'{tag} the GPU-hour tooltip and the end notes state the price positively', Z(lang, 'This lab assumes $3 per GPU hour.', '本 Lab 假设每 GPU 小时 3 美元。') in tips
          and Z(lang, 'it assumes $3 per GPU hour', '本 Lab 假设每 GPU 小时 3 美元') in ends and not any(x in tips + ends for x in ('not a vendor', 'its own assumption', '不是哪家', '自己的假设', 'both percentages are assumptions', '两个百分比都是假设')),
          ([t for t in tips.split('\n') if 'GPU' in t][:2], ends[:160]))
    s = state(pg); check(f'{tag} part 1: guided, nothing plays on the first part', s['mode'] == 'guided' and s['step'] == 0 and s['plays'] == 0 and not s['playing'], s)
    day = pg.evaluate("()=>document.querySelectorAll('#adm-day svg rect.adm-pa').length")
    check(f'{tag} part 1: the day chart has 24 hours, the capacity line', day == 24 and pg.query_selector('#adm-day .adm-cap') is not None, day)
    t1 = cur(pg); st1 = text(pg, '#adm-st-title')
    check(f'{tag} part 1: the targets are in the text and the stage title names no rule yet', Z(lang, 'at least 90% of paid requests', '付费请求至少 90% 要达标') in t1 and Z(lang, 'at least 80% of free ones', '免费请求至少 80%') in t1 and Z(lang, 'no rules at all', '没有任何规则') in st1 and Z(lang, 'threshold', '门槛') not in st1, (t1[:300], st1))
    if w == 1440: fits(pg, tag, 1, lang, ['#adm-day svg', '#adm-q svg'])
    rules(pg, f'{tag} part 1')
    # part 2
    nav(pg, 'next'); s = state(pg)
    if w == 1440: fits(pg, tag, 2, lang, ['#adm-day svg', '#adm-q svg'])
    check(f'{tag} part 2: Next waits for a fleet, with the why line', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick a fleet to continue.', '选一种卡数才能继续。')}, why(pg))
    t2c = cur(pg)
    check(f'{tag} part 2: quantization and speculative decoding raise the goodput per GPU; the P/D lab uses it to compare setups', Z(lang, 'the P/D disaggregation lab uses it to compare setups', 'P/D 分离 Lab 也用它比较方案') in t2c
          and Z(lang, 'speculative decoding all raise this number', '投机解码这类 Lab 提高的都是这个数') in t2c, t2c[:400])
    p0 = s['plays']; lever(pg, '4'); s = state(pg)
    check(f'{tag} part 2: 4 GPUs plays the new run at fast speed (R3.20, R3.9)', s['playing'] and s['plays'] == p0 + 1 and speed(pg) == 'fast' and s['p2n'] == 4, (s, speed(pg)))
    click(pg, '#adm-play', 200); s1 = state(pg); pg.wait_for_timeout(300); s2 = state(pg)
    check(f'{tag} part 2: Play pauses the run where it is', not s1['playing'] and s1['ph'] == s2['ph'] and s1['ph'] > 0, (s1['ph'], s2['ph']))
    click(pg, '#adm-play', 300); s3 = state(pg)
    check(f'{tag} part 2: Play resumes from there', s3['playing'] and s3['ph'] >= s1['ph'] and s3['plays'] == p0 + 2, s3)
    click(pg, '#adm-run .seg-ctl[data-key="spd"] button[data-v="slow"]', 150)
    check(f'{tag} part 2: the speed control switches to slow', speed(pg) == 'slow' and state(pg)['spd'] == 'slow', speed(pg))
    click(pg, '#adm-run .seg-ctl[data-key="spd"] button[data-v="fast"]', 150)
    finish(pg); t2 = cur(pg)
    check(f'{tag} part 2: the observation quotes the capacity of 4 GPUs', Z(lang, 'With 4 GPUs, the traffic is above capacity from morning to evening', '4 张卡时，从上午到晚上，流量一直超过容量') in t2 and next_dis(pg) is False, t2[-300:])
    lever(pg, '6'); finish(pg); p1 = state(pg)['plays']; lever(pg, '4'); s = state(pg)
    check(f'{tag} part 2: back to the watched 4 GPUs replays nothing (R3.20)', not s['playing'] and s['plays'] == p1, s)
    lever(pg, '4'); s = state(pg)
    check(f'{tag} part 2: the same choice again replays nothing (R3.20)', not s['playing'] and s['plays'] == p1, s)
    rules(pg, f'{tag} part 2')
    p0 = state(pg)['plays']; nav(pg, 'back'); nav(pg, 'next'); s = state(pg)
    check(f'{tag} part 2 again: Back and Next replay nothing (R3.20)', s['step'] == 1 and not s['playing'] and s['plays'] == p0, s)
    click(pg, '#adm-replay', 200); s = state(pg)
    check(f'{tag} part 2: Replay plays the watched run from the start', s['playing'] and s['plays'] == p0 + 1 and s['ph'] < 5, s)
    finish(pg)
    # part 3
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 3: Next waits for 110%', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try 110% to continue.', '试一下 110% 才能继续。')}, why(pg))
    lever(pg, '0.7'); finish(pg); lever(pg, '1.1'); s = state(pg); finish(pg); t3 = cur(pg)
    check(f'{tag} part 3: 110% plays and the queue keeps growing', s['p3f'] == 1.1 and Z(lang, 'At 110%, requests arrive faster than the GPU finishes them', '110% 时，请求来得比卡做完得快') in t3 and next_dis(pg) is False, t3[-300:])
    rules(pg, f'{tag} part 3')
    # part 4: the prediction
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 4: Next waits for an answer and its result', next_dis(pg) and set(why(pg)) == {Z(lang, 'Pick an answer and show the result to continue.', '选一个答案并看结果才能继续。')}, why(pg))
    click(pg, '#adm-rail [data-pred="slow20"]'); t4 = cur(pg)
    check(f'{tag} part 4: an answer is saved, Next still waits for the result', Z(lang, 'Saved.', '已记下') in t4 and next_dis(pg), t4[-300:])
    p0 = state(pg)['plays']; click(pg, '#adm-rail [data-act="pred-show"]', 400); s = state(pg); finish(pg); t4 = cur(pg)
    dis = pg.evaluate("()=>[...document.querySelectorAll('#adm-rail [data-pred]')].every(b=>b.disabled)")
    check(f'{tag} part 4: Show the result plays the 15-minute run, names the answer and locks the choices', s['predShown'] and s['plays'] == p0 + 1 and Z(lang, 'The answer: the wait keeps growing', '答案是：等待越来越长') in t4 and dis and next_dis(pg) is False, t4[-400:])
    rules(pg, f'{tag} part 4')
    # part 5
    nav(pg, 'next'); finish(pg)
    if w == 1440: fits(pg, tag, 5, lang, ['#adm-day svg', '#adm-q svg'])
    t5 = cur(pg)
    check(f'{tag} part 5: Next waits for paid first; admission control is named', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try paid first to continue.', '试一下付费优先才能继续。')} and Z(lang, 'admission control', '准入控制') in t5, why(pg))
    lever(pg, 'none'); finish(pg); lever(pg, 'q'); finish(pg); lever(pg, 'rl'); finish(pg); t5 = cur(pg)
    check(f'{tag} part 5: the rate limit lets the script through at 1 a second; paid still miss; Next still waits', Z(lang, 'the script gets only 1 request a second through', '脚本每秒只能过 1 个请求') in t5 and Z(lang, 'Now try paid first.', '再让付费请求先走试试。') in t5 and next_dis(pg), t5[-400:])
    lever(pg, 'pf'); finish(pg); t5 = cur(pg); st5 = text(pg, '#adm-st-title')
    check(f'{tag} part 5: paid first meets the paid goal at the peak, and the extra free requests get 429', Z(lang, 'With paid first, 100% of paid requests', '付费请求先走以后，100% 达标') in t5 and Z(lang, 'a new free request gets 429 at once', '新来的免费请求马上拿到 429') in t5 and next_dis(pg) is False and Z(lang, 'paid first', '付费优先') in st5 and Z(lang, 'threshold', '门槛') not in st5, (t5[-400:], st5))
    rules(pg, f'{tag} part 5')
    # part 6
    nav(pg, 'next'); finish(pg)
    check(f'{tag} part 6: Next waits for preemption', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try preemption to continue.', '试一下抢占才能继续。')}, why(pg))
    lever(pg, 'fcfs'); finish(pg); lever(pg, 'pre'); finish(pg); t6 = cur(pg)
    check(f'{tag} part 6: preemption meets both goals and counts its cost', Z(lang, 'With preemption, 100% of paid', '加上抢占，付费请求 100% 达标') in t6 and next_dis(pg) is False, t6[-400:])
    rules(pg, f'{tag} part 6')
    # part 7
    nav(pg, 'next'); finish(pg)
    if w == 1440: fits(pg, tag, 7, lang, ['#adm-day svg', '#adm-q svg'])
    check(f'{tag} part 7: Next waits for θ = 100%', next_dis(pg) and set(why(pg)) == {Z(lang, 'Try θ = 100% to continue.', '试一下 θ = 100% 才能继续。')}, why(pg))
    lever(pg, '60'); finish(pg); t7a = cur(pg); lever(pg, '100'); finish(pg); t7 = cur(pg)
    check(f'{tag} part 7: θ = 60% falls short, θ = 100% has more capacity than jobs: all are done, the rest is spare (R14), and it preempts at night', Z(lang, 'less than half of those handed in, so the rest are left undone', '不到交上来的一半，剩下的做不完') in t7a and Z(lang, 'that is the day’s batch capacity. It is more than the jobs handed in, so every job is done, and the capacity left over is spare', '这是全天批处理容量。它比交上来的任务多，所以任务全部做完，多出来的容量是余量') in t7 and Z(lang, 'preemptions in these two minutes', '这两分钟里就抢占了') in t7 and next_dis(pg) is False, (t7a[-200:], t7[-400:]))
    pg.screenshot(path=str(SH / f'flow-{tag}-p7.png'))
    rules(pg, f'{tag} part 7')
    # part 8
    nav(pg, 'next'); finish(pg); t8 = pg.evaluate("()=>document.querySelector('#adm-rail .step.current').textContent")
    if w == 1440: fits(pg, tag, 8, lang, ['#adm-day svg', '#adm-q svg'])
    check(f'{tag} part 8: what to remember (the extra peak traffic falls on free users as 429), misconceptions, real systems', all(x in t8 for x in (Z(lang, 'Size the fleet for the peak', '按高峰配卡'), Z(lang, 'The free users absorb the extra', '多出来的这部分由免费请求承担'), Z(lang, 'Common misconceptions', '常见误解'), Z(lang, 'What real systems do', '真实系统怎么做'), 'vLLM', 'SGLang', 'llm-d')), t8[:200])
    rules(pg, f'{tag} part 8')
    pg.screenshot(path=str(SH / f'flow-{tag}-p8.png'))
    # first-time reader (2026-10-14): Show the math states the price before the cost line, in the same words as the metric card
    pg.evaluate("()=>{const d=document.getElementById('adm-math'); if(d) d.open=true;}"); pg.wait_for_timeout(400)
    cost = pg.evaluate("""()=>{const b=[...document.querySelectorAll('#adm-mathbody .adm-mb')].find(x=>/What the day costs|一天要花多少/.test((x.querySelector('p')||{}).textContent||'')); if(!b) return null;
      const r=b.querySelector('.adm-mb-res'); return {lead:b.querySelector('p').textContent, res:r?r.textContent:''};}""")
    pg.evaluate("()=>{const d=document.getElementById('adm-math'); if(d) d.open=false;}"); pg.wait_for_timeout(150)
    check(f'{tag} Show the math: the price is stated in the lead of “what the day costs”, before the cost line', cost is not None
          and Z(lang, 'this lab assumes $3 per GPU hour', '本 Lab 假设每 GPU 小时 3 美元') in cost['lead'] and Z(lang, 'assumes', '假设') not in cost['res'], cost)
    pg.close()


def sandbox(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'sandbox'); finish(pg)
    pres = pg.evaluate("()=>[...document.querySelectorAll('#adm-app [data-preset]')].map(e=>e.textContent.trim().slice(0,40))")
    check(f'{tag} scenarios: five', len(pres) == 5, pres)
    for i in range(5):
        p0 = state(pg)['plays']; click(pg, f'#adm-app [data-preset="{i}"]', 400); s = state(pg)
        ins = text(pg, '#adm-insight')
        # scenario 1 (6 GPUs, no rules) is the sandbox's own start, already shown: it does not replay (R3.20)
        check(f'{tag} scenario {i}: {"the run on screen does not replay" if i == 1 else "a new run plays"} and its insight shows', s['plays'] == p0 + (0 if i == 1 else 1) and '%' in ins, (s['plays'], p0, ins[:120]))
        finish(pg)
    p0 = state(pg)['plays']; click(pg, '#adm-app [data-preset="0"]', 300); s = state(pg)
    check(f'{tag} scenarios: one already watched replays nothing (R3.20)', s['plays'] == p0 and not s['playing'], s)
    click(pg, '#adm-app [data-preset="4"]', 300)
    hs = pg.evaluate('()=>location.hash')
    check(f'{tag} sandbox writes its settings into the link', hs == '#admission~7_0_60_1_80', hs)
    click(pg, '#adm-controls .seg-ctl[data-key="th"] button[data-v="100"]', 300); s = state(pg)
    check(f'{tag} settings: a new threshold plays briefly and lands in the link', s['th'] == 100 and pg.evaluate('()=>location.hash') == '#admission~7_0_60_1_100', (s['th'], pg.evaluate('()=>location.hash')))
    finish(pg)
    pg.evaluate("()=>{const r=document.querySelector('#adm-ctl-n-in'); r.value='5'; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true}));}"); pg.wait_for_timeout(400)
    check(f'{tag} settings: the GPU slider sets the fleet', state(pg)['n'] == 5 and '5' in text(pg, '#adm-st-title'), (state(pg)['n'], text(pg, '#adm-st-title')))
    finish(pg)
    rules(pg, f'{tag} sandbox')
    if w == 390:
        pg.evaluate("()=>{const r=document.querySelector('#adm-rail'); window.scrollTo(0, r.getBoundingClientRect().top+scrollY+r.offsetHeight-200)}"); pg.wait_for_timeout(400)
        dk = pg.evaluate("()=>{const d=document.querySelector('#dock'); if(!d) return null; const r=d.getBoundingClientRect(); return {vis:getComputedStyle(d).display!=='none'&&r.height>0&&r.bottom<=innerHeight+1, t:d.innerText}}")
        check(f'{tag}: the mini result bar shows paid and free when the stage is off screen', dk and dk['vis'] and '%' in dk['t'], dk)
    pg.close()
    pg = page(b, w, h, lang, hash_='admission~5_50_60_1_100'); s = state(pg)
    check(f'{tag} hash round trip: a link opens the sandbox with its settings', s['mode'] == 'sandbox' and (s['n'], s['q'], s['rl'], s['prio'], s['th']) == (5, 50, 60, 1, 100), s)
    pg.close()


def challenge(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang)
    mode(pg, 'challenge'); finish(pg); s = state(pg)
    check(f'{tag} challenge: starts at 6 GPUs with no rules', s['mode'] == 'challenge' and (s['n'], s['q'], s['rl'], s['prio'], s['th']) == (6, 0, 0, 0, 60), s)
    check(f'{tag} challenge: the task card shows 144 GPU hours', '144' in text(pg, '#adm-ch-g'), text(pg, '#adm-ch-g'))
    click(pg, '#adm-app [data-act="submit"]', 300)
    pg.wait_for_function('()=>window.__adm.state().chReady', timeout=180000); pg.wait_for_timeout(400); r = text(pg, '#adm-ch-res')
    check(f'{tag} challenge: the initial setup misses, says why, offers the answer', Z(lang, 'Not met.', '没达标。') in r and Z(lang, 'of paid requests meet their goals', '的付费请求达标') in r and Z(lang, 'Show the answer', '直接看答案') in r and not pg.query_selector('#adm-ch-res .cmp'), r)
    click(pg, '#adm-ch-res [data-act="reveal"]', 400); r = text(pg, '#adm-ch-res')
    check(f'{tag} challenge: Show the answer shows the chart (your pick, the initial setup) and why', Z(lang, 'your pick', '你的选择') in r and Z(lang, 'initial setup', '初始设置') in r and Z(lang, 'The cheapest settings use 5 GPUs', '最省的设置用 5 张卡') in r, r[:400])
    for k, v in (('chn', '5'), ('chr', '60'), ('chp', '1'), ('cht', '100')):
        click(pg, f'#adm-app .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250)
    pg.wait_for_timeout(300); r = text(pg, '#adm-ch-res'); stt = text(pg, '#adm-status'); gn = text(pg, '#adm-goals .adm-gnote')
    check(f'{tag} challenge: 5 GPUs, rate limit, paid first, θ = 100% with no free queue limit misses: above capacity at the peak (B1)', Z(lang, 'Not met.', '没达标。') in r and Z(lang, 'more than these GPUs can take', '比这几张卡的容量多') in r and stt == Z(lang, 'misses if the peak lasts', '高峰持续下去就达不到') and Z(lang, 'the queue keeps growing', '排队会一直涨') in gn, (r[:300], stt, gn))
    click(pg, '#adm-app .seg-ctl[data-key="chq"] button[data-v="200"]', 300); r = text(pg, '#adm-ch-res')
    check(f'{tag} challenge: with a free queue of 200 it is met on the fewest GPU hours', Z(lang, 'Met, on the fewest GPU hours: 120 a day', '达标，而且 GPU 小时最少：每天 120') in r and text(pg, '#adm-status') == Z(lang, 'chat targets met', '聊天达到服务目标'), r[:300])
    mj, gb = text(pg, '#adm-metrics [data-k="jobs"]'), text(pg, '#adm-goals')
    check(f'{tag} challenge: the 189,000 is named the capacity; 150,000 of 150,000 are done and 39,000 is spare (R14)', Z(lang, 'Batch capacity in a day (at most)', '全天批处理容量（最多能处理）') in mj and '189,000' in mj and Z(lang, '150,000 handed in, all done; 39,000 spare', '交上来 150,000 个，全部做完；余量 39,000 个') in mj
          and Z(lang, '150,000 of 150,000', '150,000 / 150,000') in gb and '189,000' not in gb and Z(lang, 'so all 150,000 handed in are done, and the capacity left over is spare', '所以交上来的 150,000 个全部做完，多出来的容量是余量') in r, (mj, gb, r[-200:]))
    click(pg, '#adm-app .seg-ctl[data-key="chn"] button[data-v="6"]', 300); r = text(pg, '#adm-ch-res')
    check(f'{tag} challenge: the same on 6 GPUs is told 5 are enough', Z(lang, 'Met, on 144 GPU hours a day.', '达标，每天 144 GPU 小时。') in r and Z(lang, 'But 5 GPUs are enough too', '不过用 5 张卡也能过关') in r, r[:300])
    click(pg, '#adm-app .seg-ctl[data-key="chn"] button[data-v="5"]', 200); click(pg, '#adm-app .seg-ctl[data-key="chp"] button[data-v="0"]', 300); click(pg, '#adm-app .seg-ctl[data-key="chq"] button[data-v="50"]', 300); r = text(pg, '#adm-ch-res')
    check(f'{tag} challenge: 5 GPUs first come with a free queue of 50 misses at night', Z(lang, 'In the quietest hour only', '凌晨 3 点只有') in r, r[:300])
    ex = pg.evaluate("()=>[...document.querySelectorAll('#adm-app [data-exits] button')].length")
    check(f'{tag} challenge: the exits follow the result', ex >= 3, ex)
    co = pg.evaluate("()=>{const e=document.querySelector('#adm-app [data-exits]'); return e?e.textContent:''}")
    check(f'{tag} challenge: no company name in the exits', not __import__('re').search(r'OpenAI|Anthropic|DeepMind|Google|Meta\b|Microsoft', co), co[:200])
    pg.screenshot(path=str(SH / f'flow-{tag}-challenge.png'), full_page=False)
    rules(pg, f'{tag} challenge')
    pg.close()


def tile(b, w, h, lang):
    tag = f'{w}_{lang}'; pg = page(b, w, h, lang, hash_='map')
    pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="inference"], #domStrip [data-world="inference"]')].find(e=>e.offsetParent);
      if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="inference"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(500)
    t = pg.query_selector('#tile-admission')
    check(f'{tag} tile: the admission tile is on the inference branch', t is not None)
    if not t: pg.close(); return
    t.scroll_into_view_if_needed(); pg.wait_for_timeout(700)
    r0 = text(pg, '#tile-admission .tz-read')
    click(pg, '#tile-admission .seg-ctl button[data-v="4r"]', 900)
    r1 = text(pg, '#tile-admission .tz-read')
    check(f'{tag} tile: 4 GPUs fall behind; with rules 100% of paid chats are on time', Z(lang, '4 GPUs fall behind', '每天下午都接不住') in r0 and '100%' in r1, (r0, r1))
    chip = text(pg, '#tile-admission .tz-kind')
    check(f'{tag} tile: the chip says Toy run, as the note does', chip == Z(lang, 'Toy run', '玩具实跑'), chip)
    hgt = pg.evaluate("()=>document.querySelector('#tile-admission .tz-adm-vis').getBoundingClientRect().height")
    check(f'{tag} tile: the picture is at most 130 px tall', 0 < hgt <= 130, hgt)
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
for x in sorted(set(h.split(': ',1)[1] for h in rule_hits)): print('RULE', x[:220])
check('text rules', not rule_hits, rule_hits[:8])
print(f'{n - len(fails)}/{n} checks passed')
print('FAILED:', fails if fails else 'none')
sys.exit(1 if fails else 0)
