"""Flow test for the tool-scheduling lab (CONTRACT §6), at 1440×900 and 390×844, in EN and ZH.

- no page errors, no horizontal overflow, a tooltip for every help cursor, no text past the stage at 390;
- part 1: Next waits for a prediction (kit why line); the one-at-a-time run plays at 快 (R3.9) and the observation names 10 model
  calls and 27.2 s only once it has played; no stage status before the prediction (R3.8);
- part 2: Next waits for two calls; picking “book flight” highlights what it waits for;
- part 3: the verdict on the prediction comes first; all at once plays with a red call (R5.11); the three levers give 6.0 / 14.3 /
  16.3 s; Next waits for “by dependency, with a lock”;
- R3.20: Back and Next again replays nothing; the same lever again replays nothing; a new lever plays; 重放 replays from the start;
- part 4: weather ×½ keeps 16.3 s, flights ×½ 15.8 s (0.5 s less), flight booking ×½ 14.8 s (1.5 s less); part 5: k = 1 19.7 s, k = 2 16.8 s;
- parts 3, 4, 6: an option shows its result only once tried (or current); graph labels never overlap (trip and bug fix);
- part 6: hotel search timeout 20.8 s; flight booking without a key: two tickets (red); with a key: right, 21.3 s;
- part 7: the scenarios button and the next-lab footer;
- Scenarios: five, the insight of each, the hash round trip, the language switch keeps the setup;
- the challenge: no time and no verdict before Submit; the initial arrangement misses; the best arrangement meets the goal at
  11.9 s; a 13.4-s arrangement gets the middle tier; a wrong one names the tests running too early; the brief chart; the capability card;
- the dock at 390; the writing rules on every visible text and tooltip (no “toy” outside the Toy run label, no two colons in one
  Chinese sentence, no “一次只能选 / 不会叠加 / 第 N 部分里”).
Screenshots: shots/tool-scheduling/.
Run: python3 tests/tool-scheduling_flow.py [en|zh] [1440|390]  (filters for a partial run)"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from _m2 import open_drawers

P = pathlib.Path(__file__).resolve().parent.parent
LAB = 'tool-scheduling'
(P / 'out' / LAB).mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), f'out/{LAB}.html', '--only', LAB], check=True, cwd=P, stdout=subprocess.DEVNULL)
(P / 'out' / LAB / 'flow-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / f'{LAB}.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / LAB / 'flow-wrapped.html')
SH = P / 'shots' / LAB; SH.mkdir(parents=True, exist_ok=True)
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


def page(b, w, h, lang, hash_=LAB):
    pg = b.new_page(viewport={'width': w, 'height': h})
    open_drawers(pg)
    pg.on('pageerror', lambda e: errs.append(f'{w} {lang}: {e}'))
    pg.on('console', lambda m: errs.append(f'{w} {lang} console: {m.text}') if m.type == 'error' and 'ERR_' not in m.text and 'fonts.g' not in m.text and 'Failed to load resource' not in m.text else None)
    pg.goto(URL); pg.wait_for_timeout(250)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(80)
    pg.evaluate(f"()=>{{location.hash='{hash_}'}}"); pg.wait_for_timeout(450)
    return pg


overflow = lambda pg: pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth")
state = lambda pg: pg.evaluate("()=>window.__tls.state()")
anim = lambda pg: pg.evaluate("()=>window.__tls.anim()")
text = lambda pg, sel: pg.evaluate("s=>{const e=document.querySelector(s); return e?e.innerText.replace(/\\u00a0/g,' ').replace(/\\u2011/g,'-').replace(/\\u2212/g,'-'):''}", sel)
nexts = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tls-rail [data-nav=\"next\"]')].filter(b=>b.getClientRects().length).map(b=>b.disabled)")
why = lambda pg: pg.evaluate("()=>[...document.querySelectorAll('#tls-rail [data-next-why]')].filter(p=>!p.hidden&&getComputedStyle(p).visibility!=='hidden'&&p.getClientRects().length).map(p=>p.textContent).join(' ')")
speed = lambda pg: pg.evaluate("()=>{const b=document.querySelector('#tls-transport .seg-ctl[data-key=\"speed\"] button[aria-pressed=\"true\"]'); return b?b.dataset.v:null}")


def settle(pg, ms=2000):
    last, t = None, 0
    while t < ms:
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120); t += 120


def click(pg, sel, wait=250):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(wait); settle(pg)


CLICK_ANIM = """async (sel)=>{document.querySelector(sel).click(); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); return window.__tls.anim();}"""


def click_anim(pg, sel):
    a = pg.evaluate(CLICK_ANIM, sel); pg.wait_for_timeout(150); settle(pg); return a


def wait_anim(pg, timeout=25000):
    pg.wait_for_function("()=>!window.__tls.anim()", timeout=timeout); pg.wait_for_timeout(150)


def mode(pg, m):
    click(pg, f'#tls-app .modes button[data-mode="{m}"]', 350)


LVR = "()=>Object.fromEntries([...document.querySelectorAll('#tls-rail .step.current button[data-lever]')].map(b=>[b.dataset.lever,b.querySelector('.lv-r').textContent.trim()]))"
NODES = """()=>{const bad=[]; document.querySelectorAll('#tls-graph .tls-node').forEach(g=>{const r=g.querySelector('rect').getBBox(), a=g.querySelector('.tls-nt').getBBox(), b=g.querySelector('.tls-ns').getBBox();
  const inside=x=>x.x>=r.x-0.5&&x.y>=r.y-0.5&&x.x+x.width<=r.x+r.width+0.5&&x.y+x.height<=r.y+r.height+0.5;
  const meet=!(a.x+a.width<=b.x+0.5||b.x+b.width<=a.x+0.5||a.y+a.height<=b.y+0.5||b.y+b.height<=a.y+0.5);
  if(meet||!inside(a)||!inside(b)) bad.push(g.querySelector('.tls-nt').textContent);}); return [document.querySelectorAll('#tls-graph .tls-node').length,bad]}"""


def nodes_ok(pg, name):
    n, bad = pg.evaluate(NODES)
    check(f'{name}: {n} graph nodes, name and time apart and inside the box', n >= 6 and not bad, bad)


def nxt(pg):
    click(pg, '#tls-rail .step.current [data-nav="next"]', 300)


def shot(pg, name, full=False):
    vp = pg.viewport_size; pg.mouse.move(vp['width'] - 2, vp['height'] - 2); pg.wait_for_timeout(80)
    pg.screenshot(path=str(SH / f'{name}.png'), full_page=full)


def all_text(pg):
    return pg.evaluate("""()=>{const out=[]; for(const s of ['#tls-app','#dock']){const e=document.querySelector(s); if(e&&e.offsetParent!==null) out.push(e.innerText);}
      document.querySelectorAll('#tls-app [data-tipx], #dock [data-tipx]').forEach(e=>out.push(e.getAttribute('data-tipx'))); return out.join('\\n').replace(/\\u2212/g,'-').replace(/\\u2011/g,'-')}""")


TOY = re.compile(r'in this toy\b|在这个玩具里|\btoy\b(?! run)|玩具(?!实跑)', re.I)
BANNED = re.compile(r'一次只能选|不会叠加|第 ?[0-9一二三四五六七八] ?部分里|吃|喂|扛|吞')


def writing(pg, tag, lang):
    t = all_text(pg)
    m = TOY.search(t); check(f'{tag} no toy wording', not m, m and t[max(0, m.start() - 60):m.end() + 40])
    m = BANNED.search(t); check(f'{tag} no banned wording', not m, m and t[max(0, m.start() - 60):m.end() + 40])
    if lang == 'zh':
        bad = [s for s in re.split(r'[。！？\n]', t) if s.count('：') >= 3]
        check(f'{tag} no sentence with three colons', not bad, bad[:2])
    helpless = pg.evaluate("""()=>[...document.querySelectorAll('#tls-app *')].filter(e=>e.getClientRects().length&&getComputedStyle(e).cursor==='help'&&!e.dataset.tipx&&!e.dataset.tip&&!e.closest('[data-tipx],[data-tip]')).map(e=>e.className).slice(0,4)""")
    check(f'{tag} every help cursor has a tooltip', not helpless, helpless)
    check(f'{tag} no horizontal overflow', overflow(pg) <= 1, overflow(pg))


def run(b, w, h, lang):
    tag = f'{w} {lang}'
    pg = page(b, w, h, lang)
    pg.evaluate("()=>{const t=document.querySelector('#tls-transport .seg-ctl[data-key=\"speed\"] button[data-v=\"fast\"]');}")
    # part 1
    st = state(pg); check(f'{tag} starts in guided part 1', st['mode'] == 'guided' and st['step'] == 0, st)
    check(f'{tag} p1 Next disabled before a prediction', all(nexts(pg)), nexts(pg))
    check(f'{tag} p1 why line', Z(lang, 'Pick an answer', '选一个答案') in why(pg), why(pg))
    check(f'{tag} p1 run plays at fast', anim(pg) is not None and speed(pg) == 'fast', (anim(pg), speed(pg)))
    check(f'{tag} p1 no observation while it plays', not pg.evaluate("()=>{const b=document.querySelector('#tls-rail [data-obs]'); return b&&!b.hidden&&b.innerText.length>0}"))
    check(f'{tag} p1 no status before the prediction', pg.evaluate("()=>document.querySelector('#tls-status').hidden"))
    shot(pg, f'p1-{w}-{lang}')
    nodes_ok(pg, f'{tag} p1 trip graph')
    wait_anim(pg, 20000)
    ob = text(pg, '#tls-rail [data-obs]')
    check(f'{tag} p1 observation 10 calls 27.2 s', '10' in ob and '27.2' in ob, ob)
    check(f'{tag} p1 agent-anatomy linked', pg.evaluate("(lg)=>{const l=window.ASP.catalog.LAB['agent-anatomy']; return !!document.querySelector('#tls-rail a[href=\"#agent-anatomy\"]')||document.querySelector('#tls-rail').innerText.includes(l.s[lg])}", lang))
    click(pg, '#tls-rail [data-pred="ok"]')
    check(f'{tag} p1 Next enabled after a prediction', not any(nexts(pg)), nexts(pg))
    # R09: Settings put the stage on another setup: the card says so and names both; its own numbers stay on this part's setup
    check(f'{tag} p1 the observation is labelled as this part’s setup (R09)', Z(lang, 'This part’s setup', '这一部分的设置').lower() in ob.lower(), ob)   # the eyebrow is set in capitals
    click(pg, '#tls-controls .seg-ctl[data-key="sch"] button[data-v="all"]'); wait_anim(pg, 20000)
    nw, ob = text(pg, '#tls-rail [data-now]'), text(pg, '#tls-rail [data-obs]')
    check(f'{tag} p1 Settings on all at once: a line says the timeline shows another setup, names both, and the card keeps 27.2 s (R09)',
          Z(lang, 'The timeline now shows another setup than this card', '时间轴上现在不是这张卡片讲的设置') in nw and Z(lang, 'All at once', '一次全发') in nw and Z(lang, 'One at a time', '一个一个来') in nw and '27.2' in ob, (nw, ob))
    click(pg, '#tls-rail [data-act="card"]')
    check(f'{tag} p1 Back to this part’s setup: one at a time again, and the line goes (R09)', state(pg)['sch'] == 'seq' and not pg.evaluate("()=>{const b=document.querySelector('#tls-rail [data-now]'); return !!b&&!b.hidden}"), state(pg)['sch'])
    writing(pg, f'{tag} p1', lang)
    # part 2
    nxt(pg)
    check(f'{tag} p2 Next waits for two calls', all(nexts(pg)))
    click(pg, '#tls-ic-pick2 button[data-v="bfl"]')
    check(f'{tag} p2 highlight', pg.evaluate("()=>document.querySelectorAll('#tls-graph .tls-node.up').length") == 4)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p2 observation names the comparison', Z(lang, 'Compare prices', '比价') in ob, ob)
    click(pg, '#tls-ic-pick2 button[data-v="wea"]')
    check(f'{tag} p2 Next enabled', not any(nexts(pg)))
    shot(pg, f'p2-{w}-{lang}')
    writing(pg, f'{tag} p2', lang)
    # part 3
    a = None
    nxt(pg); a = anim(pg)
    lv = pg.evaluate(LVR); check(f'{tag} p3 only the current option shows a result', lv.get('all') and not lv.get('dep0') and not lv.get('dep1'), lv)
    card = text(pg, '#tls-rail .step.current')
    check(f'{tag} p3 verdict first', card.find(Z(lang, 'Not quite', '不对')) >= 0 and card.find(Z(lang, 'Not quite', '不对')) < card.find(Z(lang, 'One reply', '模型的一次回复')), card[:200])
    check(f'{tag} p3 all at once plays', a is not None)
    wait_anim(pg)
    check(f'{tag} p3 red call', pg.evaluate("()=>document.querySelectorAll('#tls-gantt .tls-bar.bad').length") >= 4)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p3 all at once 6.0 s', '6.0' in ob, ob)
    shot(pg, f'p3-all-{w}-{lang}')
    a = click_anim(pg, '#tls-rail button[data-lever="dep0"]'); check(f'{tag} p3 new lever plays', a is not None); wait_anim(pg)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p3 no lock 14.3 s', '14.3' in ob, ob)
    a = click_anim(pg, '#tls-rail button[data-lever="dep0"]'); check(f'{tag} p3 same lever replays nothing (R3.20)', a is None, a)
    check(f'{tag} p3 Next waits for the lock', all(nexts(pg)))
    click_anim(pg, '#tls-rail button[data-lever="dep1"]'); wait_anim(pg)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p3 lock 16.3 s, 10.9 s saved', '16.3' in ob and '10.9' in ob, ob)
    check(f'{tag} p3 Next enabled', not any(nexts(pg)))
    lv = pg.evaluate(LVR); check(f'{tag} p3 tried options show results', all(lv.get(k) for k in ('all', 'dep0', 'dep1')), lv)
    shot(pg, f'p3-{w}-{lang}')
    writing(pg, f'{tag} p3', lang)
    # Back and Next: nothing replays
    click(pg, '#tls-rail .step.current [data-nav="back"]'); a = pg.evaluate(CLICK_ANIM, '#tls-rail .step.current [data-nav="next"]'); settle(pg)
    check(f'{tag} back and next replays nothing (R3.20)', a is None, a)
    # replay button
    if w > 760 or True:
        a = pg.evaluate("async ()=>{document.querySelector('#tls-transport [data-act=\"replay\"]').click(); await new Promise(r=>requestAnimationFrame(r)); return window.__tls.anim();}")
        check(f'{tag} replay plays from the start', a is not None and a['t'] < 1.0, a)
        wait_anim(pg)
    # part 4
    lv3 = state(pg); nxt(pg)
    lv = pg.evaluate(LVR); check(f'{tag} p4 untried options empty', lv.get('-1') and not any(lv.get(k) for k in ('3', '1', '5')), lv)
    for v, want, says in (('3', '16.3', '16.3'), ('1', '15.8', '0.5'), ('5', '14.8', '1.5')):
        click(pg, f'#tls-rail button[data-lever="{v}"]', 200); wait_anim(pg)
        ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p4 half {v} observation says {says}', says in ob, ob)
        mt = text(pg, '#tls-metrics'); check(f'{tag} p4 half {v} total {want}', want in mt, mt)
    check(f'{tag} p4 critical path outlined', pg.evaluate("()=>document.querySelectorAll('#tls-gantt .tls-bar.crit').length") >= 4)
    check(f'{tag} p4 Next enabled', not any(nexts(pg)))
    shot(pg, f'p4-{w}-{lang}')
    pg.evaluate("()=>{document.querySelector('#tls-math').open=true}"); pg.wait_for_timeout(100)
    m = text(pg, '#tls-mathbody'); check(f'{tag} p4 math has the total', '14.8' in m, m[:200])
    writing(pg, f'{tag} p4', lang)
    # part 5
    nxt(pg)
    for v, want in (('1', '19.7'), ('2', '16.8')):
        click(pg, f'#tls-ic-k button[data-v="{v}"]', 200); wait_anim(pg)
        ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p5 k {v} gives {want}', want in ob, ob)
    check(f'{tag} p5 Next enabled', not any(nexts(pg)))
    shot(pg, f'p5-{w}-{lang}')
    # part 6
    nxt(pg)
    lv = pg.evaluate(LVR); check(f'{tag} p6 no result before a try', not any(lv.values()), lv)
    click(pg, '#tls-rail button[data-lever="sho"]', 200); wait_anim(pg)
    lv = pg.evaluate(LVR); check(f'{tag} p6 only the tried option shows a result', lv.get('sho') and not lv.get('bfl0') and not lv.get('bfl1'), lv)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p6 hotel timeout 20.8', '20.8' in ob, ob)
    check(f'{tag} p6 hung attempt drawn', pg.evaluate("()=>document.querySelectorAll('#tls-gantt .tls-hang').length") == 1)
    click(pg, '#tls-rail button[data-lever="bfl0"]', 200); wait_anim(pg)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p6 two tickets', Z(lang, 'two tickets', '两张票') in ob, ob)
    check(f'{tag} p6 the booking is red', pg.evaluate("()=>document.querySelectorAll('#tls-gantt .tls-bar.bad').length") == 1)
    shot(pg, f'p6-{w}-{lang}')
    click(pg, '#tls-rail button[data-lever="bfl1"]', 200); wait_anim(pg)
    ob = text(pg, '#tls-rail [data-obs]'); check(f'{tag} p6 with a key one ticket', '21.3' in ob and Z(lang, 'one ticket', '一张票') in ob, ob)
    check(f'{tag} p6 Next enabled', not any(nexts(pg)))
    writing(pg, f'{tag} p6', lang)
    # part 7
    nxt(pg); wait_anim(pg)
    card = text(pg, '#tls-rail')
    check(f'{tag} p7 scenarios button', pg.evaluate("()=>!!document.querySelector('#tls-rail [data-mode-go=\"sandbox\"]')"))
    check(f'{tag} p7 next lab footer', pg.evaluate("()=>!!document.querySelector('#tls-rail .next-lab, .next-lab-end')"))
    shot(pg, f'p7-{w}-{lang}')
    writing(pg, f'{tag} p7', lang)
    # scenarios
    mode(pg, 'sandbox')
    n_pre = pg.evaluate("()=>document.querySelectorAll('#tls-rail [data-preset]').length"); check(f'{tag} five scenarios', n_pre == 5, n_pre)
    for i, want in ((1, '27.2'), (2, '16.8'), (3, '23.8'), (4, '21.3')):
        click(pg, f'#tls-rail [data-preset="{i}"]', 200)
        ins = text(pg, '#tls-insight'); check(f'{tag} scenario {i} insight {want}', want in ins, ins)
        wait_anim(pg)
    a = click_anim(pg, '#tls-rail [data-preset="2"]'); check(f'{tag} a watched scenario does not replay', a is None, a)
    hs = pg.evaluate("()=>location.hash"); check(f'{tag} hash carries the setup', hs.startswith(f'#{LAB}~') and '_2_' in hs, hs)
    shot(pg, f'sandbox-{w}-{lang}')
    writing(pg, f'{tag} sandbox', lang)
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(300)
    st = state(pg); check(f'{tag} language switch keeps the setup', st['mode'] == 'sandbox' and st['k'] == 2, st)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(300)
    # the dock at 390
    if w <= 760:
        pg.evaluate("()=>window.scrollTo(0, document.querySelector('#tls-controls').getBoundingClientRect().top+scrollY)"); pg.wait_for_timeout(400)
        check(f'{tag} dock shows', pg.evaluate("()=>{const d=document.getElementById('dock'); return !!d&&!d.hidden&&d.getBoundingClientRect().height>0}"))
        pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(200)
    # hash round trip
    pg2 = page(b, w, h, lang, f'{LAB}~seq_1_0_0_1_1p5')
    st = state(pg2); check(f'{tag} hash round trip', st['mode'] == 'sandbox' and st['sch'] == 'seq', st); pg2.close()
    # challenge
    mode(pg, 'challenge')
    nodes_ok(pg, f'{tag} ch bug-fix graph')
    t = text(pg, '#tls-rail .railcard')
    check(f'{tag} ch no time before submit', Z(lang, 'shown after you submit', '提交后才知道') in t, t[:300])
    check(f'{tag} ch no Gantt before submit', pg.evaluate("()=>!document.querySelector('#tls-g-ph').hidden"))
    click(pg, '#tls-rail [data-act="submit"]', 300); wait_anim(pg)
    r = text(pg, '#tls-ch-res'); check(f'{tag} ch initial misses', Z(lang, 'Not yet', '还没达到') in r and '17.1' in r, r[:300])
    for i, v in enumerate([1, 1, 1, 2, 2, 2]):
        click(pg, f'#tls-ch-{i} button[data-v="{v}"]', 120)
    click(pg, '#tls-rail [data-act="submit"]', 300); wait_anim(pg)
    r = text(pg, '#tls-ch-res'); check(f'{tag} ch tests too early', Z(lang, 'before', '之前') in r and Z(lang, 'F run the tests', 'F 跑测试') in r, r[:300])
    click(pg, '#tls-ch-5 button[data-v="3"]', 120)
    click(pg, '#tls-rail [data-act="submit"]', 300); wait_anim(pg)
    r = text(pg, '#tls-ch-res'); check(f'{tag} ch best met', Z(lang, 'Goal met', '达成目标') in r and '11.9' in r and Z(lang, 'none is faster', '没有哪种更快') in r, r[:300])
    check(f'{tag} ch brief chart', pg.evaluate("()=>!!document.querySelector('#tls-ch-res .cmp')"))
    shot(pg, f'ch-best-{w}-{lang}')
    for i, v in enumerate([1, 2, 2, 3, 3, 4]):
        click(pg, f'#tls-ch-{i} button[data-v="{v}"]', 120)
    click(pg, '#tls-rail [data-act="submit"]', 300); wait_anim(pg)
    r = text(pg, '#tls-ch-res'); check(f'{tag} ch middle tier', Z(lang, 'Goal met', '达成目标') in r and '13.4' in r and '11.9' in r, r[:300])
    check(f'{tag} ch capability card', pg.evaluate("()=>!!document.querySelector('#tls-rail [data-exits]')"))
    shot(pg, f'ch-{w}-{lang}')
    writing(pg, f'{tag} challenge', lang)
    pg.close()


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in SIZES:
        for lang in LANGS:
            run(b, w, h, lang)
    b.close()
for e in errs[:10]: print('ERR', e)
print(f'{n - len(fails)}/{n} passed; page errors: {len(errs)}')
sys.exit(1 if fails or errs else 0)
