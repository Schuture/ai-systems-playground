"""Smoke checks of the Architecture overview v2 pages (Batch 1 after Codex review 1: wide stage with the model switch on it, five guided
parts that light up the picture, optional deep dives, the challenge one question at a time, the phone template).

python3 tests/architecture_flow.py
Builds out/architecture-build.html (this lab only; wrapped as out/architecture-flow.html). EN and 中文 at 1440 × 900 and 390 × 844: the home hero (six stage names, one lit, no buttons since 2026-10-11 19:28 UTC; one expanded,
fits the first screen at 1440, no sideways scroll, animation on screen only, none with reduced motion; the pointer crossing it on the
first visit does not cut a pass short), the lab (overview, hot spots,
the five pop-ups and their steppers and controls, guided parts, scenarios, challenge), no page errors. Screenshots in shots/architecture-v2/.
The causal switch (guided part 5): Next waits for the switch, the switch toggles the leak arrows and the grid's hatched cells, the result
box states the result with its baseline, the stage fits, the language switch keeps the state, reduced motion draws no animation.
Its screenshots (mask on and off, EN and 中文, 1440 and 390) are in shots/architecture/.
python3 tests/architecture_flow.py [--mask]   (--mask: only the causal switch, for quick runs)"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
SH = P / 'shots' / 'architecture-v2'; SH.mkdir(parents=True, exist_ok=True)
SM = P / 'shots' / 'architecture'; SM.mkdir(parents=True, exist_ok=True)
ONLY_MASK = '--mask' in sys.argv
subprocess.run([sys.executable, str(P / 'build.py'), 'out/architecture-build.html', '--only', 'architecture'], check=True, cwd=P, stdout=subprocess.DEVNULL)
PAGE = P / 'out' / 'architecture-flow.html'
PAGE.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'architecture-build.html').read_text() + '</body></html>')
URL = 'file://' + str(PAGE)
n, fails = 0, []


def check(name, ok, info=''):
    global n
    n += 1
    if not ok:
        fails.append(f'{name} :: {info}')
        print('FAIL', name, '::', str(info)[:300])


def st(pg):
    return pg.evaluate("()=>window.__ar.state()")


def page(b, w, h, lang, errs, reduce=False):
    ctx = b.new_context(viewport={'width': w, 'height': h}, reduced_motion='reduce' if reduce else 'no-preference')
    pg = ctx.new_page()
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(500)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(250)
    return pg


def click(pg, sel):
    pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(250)


ON_SCREEN = """s=>{const e=document.querySelector(s); if(!e) return {ok:false,why:'missing'}; const r=e.getBoundingClientRect(), bar=document.querySelector('.topbar'), dk=document.querySelector('#dock');
  const top=Math.max(bar?bar.getBoundingClientRect().bottom:0, dk&&!dk.hidden?dk.getBoundingClientRect().bottom:0);
  return {ok:r.height>0&&r.top>=top-1&&r.bottom<=innerHeight+1, top:Math.round(r.top), bottom:Math.round(r.bottom), limit:Math.round(top)}}"""


def rail_jump(pg, tag, w, lang):
    """review L1 (and L10 on phones): from part 1, jump straight to part 5 with the rail's own button (a real click, the header on a
    desktop, the part bar's dot on a phone): the card's title and the stage's caption are on screen, clear of the top bar and the mini result bar"""
    click(pg, '#ar-rail .steps [data-go="0"]'); pg.wait_for_timeout(900)
    sel = '#ar-rail .steps [data-go="4"]' if w > 760 else '#ar-rail .pbar [data-go="4"]'
    pg.evaluate("s=>{window.scrollTo(0,0); const e=document.querySelector(s); if(e.getBoundingClientRect().bottom>innerHeight-10) e.scrollIntoView({block:'end'})}", sel); pg.wait_for_timeout(300)
    pg.click(sel); pg.wait_for_timeout(1800)
    title = pg.evaluate(ON_SCREEN, '#ar-rail .step.current .step-h' if w > 760 else '#ar-rail .pbar')
    cap = pg.evaluate(ON_SCREEN, '#ar-cap')
    check(f'{tag} part 5 by the rail’s own button: the card title (phone: the part bar) and the stage caption are on screen', st(pg)['step'] == 4 and title['ok'] and cap['ok'], (title, cap))
    check(f'{tag} R3.20 part 5: the causal switch draws itself in on its first showing', not pg.evaluate(MK_STILL))
    pg.screenshot(path=str(SM / f'mask_railjump_{w}_{lang}.png'))


MK_STILL = "()=>{const e=document.querySelector('#ar-mk'); return !!e&&e.classList.contains('ar-still')}"
MK_MARK = "()=>{const s=document.querySelector('#ar-mk svg'); if(s) s._r320=1}"
# not drawn again (the same chart element) or drawn finished
MK_CALM = "()=>{const e=document.querySelector('#ar-mk'), s=e&&e.querySelector('svg'); return !!e&&(e.classList.contains('ar-still')||!!(s&&s._r320))}"
def replay_checks(pg, tag, w, lang):
    """R3.20: a pop-up step animates when it is new to the screen; the same step drawn again (Pause, the end of a play-through, the option
    already chosen, a language switch) is drawn finished; a new draw animates only the drawn token; at the last step Play says Replay"""
    still = "()=>document.querySelector('#ar-dlg-body').classList.contains('ar-still')"
    fly_still = "()=>document.querySelector('#ar-dlg-body').classList.contains('ar-still-fly')"
    mark = "()=>{document.querySelector('#ar-dlg-body').firstElementChild._r320=1}"
    kept = "()=>!!document.querySelector('#ar-dlg-body').firstElementChild._r320"
    play = '#ar-stepper [data-act="play"]'
    click(pg, '#ar-ov [data-pop="attn"]')
    check(f'{tag} R3.20: a pop-up opens with its first step animating', st(pg)['dlg'] == 'attn' and not pg.evaluate(still))
    click(pg, '#ar-stepper [data-step="1"]')
    check(f'{tag} R3.20: the next step animates', st(pg)['dst'] == 1 and not pg.evaluate(still))
    for _ in range(5): click(pg, '#ar-stepper [data-step="1"]')
    m = st(pg)['aMask']
    pg.evaluate(mark); click(pg, f'#ar-modal [data-key="amask"] [data-v="{m}"]')
    check(f'{tag} R3.20: the mask already chosen redraws nothing', pg.evaluate(kept), m)
    other = 'causal' if m != 'causal' else 'window'
    click(pg, f'#ar-modal [data-key="amask"] [data-v="{other}"]')
    check(f'{tag} R3.20: another mask animates its step', not pg.evaluate(kept) and not pg.evaluate(still))
    click(pg, f'#ar-modal [data-key="amask"] [data-v="{m}"]')
    pg.wait_for_timeout(2600)   # this mask's step has played its entrance to the end
    click(pg, f'#ar-modal [data-key="amask"] [data-v="{other}"]'); pg.wait_for_timeout(2600)
    pg.evaluate(mark); click(pg, f'#ar-modal [data-key="amask"] [data-v="{m}"]')
    check(f'{tag} R3.20: a mask switched back (A → B → A) to a step already watched to its end is drawn finished, no replay', not pg.evaluate(kept) and pg.evaluate(still), m)
    b = pg.evaluate("s=>{const e=document.querySelector(s); return {t:e.textContent, tip:e.dataset.tipx||''}}", play)
    check(f'{tag} R3.20: at the last step the play button says Replay and its tooltip says so', ('Replay' in b['t'] or '重放' in b['t']) and ('step 1' in b['tip'] or '第 1 步' in b['tip']), b)
    click(pg, play)
    check(f'{tag} R3.20: Replay starts again from step 1, animating', st(pg)['dst'] == 0 and not pg.evaluate(still), st(pg)['dst'])
    click(pg, play)
    check(f'{tag} R3.20: Pause draws the same step finished, no replay', st(pg)['dst'] == 0 and pg.evaluate(still))
    other_lang = 'zh' if lang == 'en' else 'en'
    pg.evaluate("l=>document.querySelector(`.langsw button[data-lang=\"${l}\"]`).click()", other_lang); pg.wait_for_timeout(400)
    check(f'{tag} R3.20: a language switch keeps the pop-up and draws its step finished', st(pg)['dlg'] == 'attn' and st(pg)['dst'] == 0 and pg.evaluate(still), st(pg)['dlg'])
    pg.evaluate("l=>document.querySelector(`.langsw button[data-lang=\"${l}\"]`).click()", lang); pg.wait_for_timeout(400)
    click(pg, '#ar-stepper [data-step="1"]'); click(pg, '#ar-stepper [data-step="-1"]')
    check(f'{tag} R3.20: ◀ back to a step already watched (drawn finished on Pause) draws it finished, no replay', st(pg)['dst'] == 0 and pg.evaluate(still), st(pg)['dst'])
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    click(pg, '#ar-ov [data-pop="attn"]')
    check(f'{tag} R3.20: the pop-up opened again at a step already watched draws it finished, no replay', st(pg)['dlg'] == 'attn' and st(pg)['dst'] == 0 and pg.evaluate(still), st(pg)['dst'])
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    if w == 1440:   # a play-through to the end draws the last step once (about 5 s)
        click(pg, '#ar-ov [data-pop="mlp"]'); click(pg, '#ar-stepper [data-step="1"]'); click(pg, '#ar-stepper [data-step="1"]')
        click(pg, play); pg.wait_for_timeout(1500)
        check(f'{tag} R3.20: playing on, the last step animates', st(pg)['dst'] == 3 and not pg.evaluate(still), st(pg)['dst'])
        pg.wait_for_timeout(3000)
        b = pg.evaluate("s=>{const e=document.querySelector(s); return {t:e.textContent, p:e.getAttribute('aria-pressed')}}", play)
        check(f'{tag} R3.20: at the end of the play-through the last step is not played again, and Play says Replay', st(pg)['dst'] == 3 and pg.evaluate(still) and b['p'] == 'false' and ('Replay' in b['t'] or '重放' in b['t']), b)
        pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    click(pg, '#ar-ov [data-pop="out"]'); click(pg, '#ar-stepper [data-step="1"]'); click(pg, '#ar-stepper [data-step="1"]')
    click(pg, '#ar-modal [data-act="sample"]')
    check(f'{tag} R3.20: a draw animates the drawn token only, not the step around it', pg.evaluate(still) and not pg.evaluate(fly_still) and pg.evaluate("()=>!!document.querySelector('#ar-modal .ar-ofly.on')"))
    pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
def mask_part(pg, tag, w, lang):
    """guided part 5, the causal switch: gating, the switch, the result box, the stage; screenshots on and off"""
    pg.evaluate(MK_MARK)
    click(pg, '#ar-rail .steps [data-go="4"]')
    wait = "()=>{const e=document.querySelector('#ar-rail #ar-mkwait'); return e&&e.getClientRects().length?e.textContent:''}"
    check(f'{tag} part 5: the disabled Next says why, under the buttons, and names the mask', ('mask' in pg.evaluate(wait)) and
          pg.evaluate("()=>[...document.querySelectorAll('#ar-rail [data-nav=\"next\"]')].every(b=>b.getAttribute('aria-describedby')==='ar-mkwait')"), pg.evaluate(wait))
    # R1.1: every help cursor in the card and the stage has a hover text; R7.8: every text fits its capsule
    helpless = pg.evaluate("()=>[...document.querySelectorAll('#ar-rail .step.current *, #ar-stage *')].filter(e=>getComputedStyle(e).cursor==='help'&&!e.closest('[data-tipx],[data-tip],.t,.def')).map(e=>e.tagName+'.'+(e.getAttribute('class')||'')).slice(0,5)")
    check(f'{tag} part 5: every help cursor has a hover text (R1.1)', not helpless, helpless)
    tight = pg.evaluate("()=>[...document.querySelectorAll('#ar-mk .ar-mk-pill')].filter(g=>{const r=g.querySelector('rect'), t=g.querySelector('text'); return t.getComputedTextLength()>r.getBBox().width-8}).map(g=>g.textContent)")
    btns = pg.evaluate("()=>[...document.querySelectorAll('#ar-rail .step.current button')].filter(b=>b.getClientRects().length&&b.scrollWidth>b.clientWidth+1).map(b=>b.textContent)")
    check(f'{tag} part 5: every token fits its capsule and every button its box (R7.8)', not tight and not btns, (tight, btns))
    nexts = "()=>[...document.querySelectorAll('#ar-rail [data-nav=\"next\"]')].map(b=>b.disabled)"
    leaks = "()=>document.querySelectorAll('#ar-mk .ar-mk-leak').length"
    hatch = "()=>document.querySelectorAll('#ar-mk .ar-mkc.hid').length"
    stage_fit = "()=>{const s=document.querySelector('#ar-stage'); return getComputedStyle(s).position!=='sticky'||s.scrollHeight<=s.clientHeight+2}"
    check(f'{tag} part 5: the stage shows the causal switch instead of the overview', pg.is_visible('#ar-mk svg') and not pg.is_visible('#ar-ov') and not pg.is_visible('#ar-stage .ar-mseg'))
    check(f'{tag} part 5: Next (card and phone bar) waits until the switch is tried', all(pg.evaluate(nexts)), pg.evaluate(nexts))
    check(f'{tag} part 5: mask on, no leak arrows and 15 hatched cells (the upper triangle of 6 × 6)', pg.evaluate(leaks) == 0 and pg.evaluate(hatch) == 15, (pg.evaluate(leaks), pg.evaluate(hatch)))
    check(f'{tag} part 5: no result box before the switch is tried', pg.query_selector('#ar-rail .ar-mkobs') is None)
    click(pg, '#ar-rail [data-key="mkmask"] [data-v="on"]')
    check(f'{tag} part 5: pressing the switch that is already on does not count as trying it', all(pg.evaluate(nexts)) and not st(pg)['mk']['tried'])
    check(f'{tag} R3.20 part 5: the part clicked again and the switch already on replay nothing', pg.evaluate(MK_CALM))
    pg.evaluate(MK_MARK); pg.set_viewport_size({'width': w - 24, 'height': 900 if w > 760 else 844}); pg.wait_for_timeout(500)
    check(f'{tag} R3.20 part 5: a resize replays nothing (the chart is kept, or drawn again finished)', pg.evaluate(MK_CALM))
    pg.set_viewport_size({'width': w, 'height': 900 if w > 760 else 844}); pg.wait_for_timeout(500)
    pg.screenshot(path=str(SM / f'mask_on_{w}_{lang}.png'), full_page=w <= 760)
    click(pg, '#ar-rail [data-key="mkmask"] [data-v="off"]')
    check(f'{tag} R3.20 part 5: the mask turned off is new, so the chart draws itself in', not pg.evaluate(MK_STILL))
    pg.wait_for_timeout(650)
    s = st(pg)['mk']
    check(f'{tag} part 5: the switch turns the mask off and Next opens', s == {'on': False, 'tried': True} and not any(pg.evaluate(nexts)), (s, pg.evaluate(nexts)))
    check(f'{tag} part 5: mask off, 5 leak arrows in training and 1 in generation, no hatched cell, the answer cells red',
          pg.evaluate(leaks) == 6 and pg.evaluate("()=>document.querySelectorAll('#ar-mk .ar-mk-gen').length") == 1 and pg.evaluate(hatch) == 0
          and pg.evaluate("()=>document.querySelectorAll('#ar-mk .ar-mkc.fut').length") == 15, (pg.evaluate(leaks), pg.evaluate(hatch)))
    check(f'{tag} part 5: once the switch is tried, the reason under Next is gone', not pg.evaluate(wait))
    t = pg.inner_text('#ar-rail .ar-mkobs')
    check(f'{tag} part 5: mask off, “training loss”, 1.31 and 33.5% explain themselves on hover', pg.evaluate("()=>[...document.querySelectorAll('#ar-rail .ar-mkobs .def')].map(e=>e.textContent)") == (['training loss', '1.31', '77.8%', '33.5%'] if lang == 'en' else ['训练 loss', '1.31', '77.8%', '33.5%']),
          pg.evaluate("()=>[...document.querySelectorAll('#ar-rail .ar-mkobs .def')].map(e=>e.textContent)"))
    check(f'{tag} part 5: mask off, the switch and the generation panel say it is about training', ('during training' in pg.inner_text('#ar-rail #ar-mkseg') or '训练时' in pg.inner_text('#ar-rail #ar-mkseg'))
          and ('trained without the mask' in pg.evaluate("()=>document.querySelector('#ar-mk svg').textContent") or '去掉 mask 训练出来的模型' in pg.evaluate("()=>document.querySelector('#ar-mk svg').textContent")))
    check(f'{tag} part 5: mask off, the result box states the result with its baseline (loss 1.31 → nearly 0, “ Paris” 77.8% → 33.5%)',
          all(x in t for x in ['1.31', '77.8%', '33.5%']) and ('nearly 0' in t or '接近 0' in t), t)
    svg = pg.evaluate("()=>document.querySelector('#ar-mk svg').textContent")
    check(f'{tag} part 5: mask off, the stage shows the loss below 0.01 and “ Paris” at 33.5%', '< 0.01' in svg and '33.5%' in svg, svg[-200:])
    check(f'{tag} part 5: the stage caption says the mask is off', ('off' in pg.inner_text('#ar-cap')) or ('关' in pg.inner_text('#ar-cap')), pg.inner_text('#ar-cap'))
    check(f'{tag} part 5: no sideways scroll, and the sticky stage does not scroll inside', pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth") <= 0 and pg.evaluate(stage_fit))
    pg.screenshot(path=str(SM / f'mask_off_{w}_{lang}.png'), full_page=w <= 760)
    # the language switch keeps the part and the switch
    other = 'zh' if lang == 'en' else 'en'
    pg.click(f'.langsw button[data-lang="{other}"]'); pg.wait_for_timeout(400)
    check(f'{tag} part 5: the language switch keeps the part and the switch', st(pg)['step'] == 4 and st(pg)['mk'] == {'on': False, 'tried': True} and pg.evaluate(leaks) == 6, st(pg)['mk'])
    check(f'{tag} R3.20 part 5: the language switch draws the chart finished', pg.evaluate(MK_STILL))
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(400)
    click(pg, '#ar-rail [data-key="mkmask"] [data-v="on"]')
    check(f'{tag} R3.20 part 5: the mask on again (on → off → on) is a chart already watched, so it is drawn finished', pg.evaluate(MK_STILL))
    pg.wait_for_timeout(300)
    t = pg.inner_text('#ar-rail .ar-mkobs')
    check(f'{tag} part 5: mask on again, the leak is gone and the result box compares with the mask off',
          pg.evaluate(leaks) == 0 and pg.evaluate(hatch) == 15 and all(x in t for x in ['1.31', '77.8%', '33.5%']), t)
    if w <= 760:   # on phones the card sits under the stage: with the stage out of view, the mini result bar carries the switch's result
        pg.evaluate("()=>{const c=document.querySelector('#ar-rail .ar-mkobs'); window.scrollTo(0,c.getBoundingClientRect().top+scrollY-150)}"); pg.wait_for_timeout(600)
        dk = pg.evaluate("()=>{const d=document.querySelector('#dock'); return {hidden:d.hidden, st:(document.querySelector('#dkStatus')||{}).textContent, read:(document.querySelector('#dkRead')||{}).textContent}}")
        check(f'{tag} part 5: the mini result bar reads the switch (mask on, loss 1.31, “ Paris” 77.8%)', not dk['hidden'] and '1.31' in dk['read'] and '77.8%' in dk['read'], dk)
    click(pg, '#ar-rail .step.current [data-nav="next"]')
    check(f'{tag} part 6: What to remember, the overview is back, the mask in the third idea', st(pg)['step'] == 5 and pg.is_visible('#ar-ov') and not pg.is_visible('#ar-mk')
          and ('causal mask' in pg.inner_text('#ar-rail .ar-sum') or '因果 mask' in pg.inner_text('#ar-rail .ar-sum')))
    click(pg, '#ar-rail .steps [data-go="4"]')
    check(f'{tag} R3.20 part 5: back in the part, the chart it showed is drawn finished', st(pg)['step'] == 4 and pg.evaluate(MK_STILL))
    click(pg, '#ar-rail .steps [data-go="0"]')


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        for (w, h) in [(1440, 900), (390, 844)]:
            tag = f'{lang} {w}'
            errs = []
            pg = page(b, w, h, lang, errs)
            if ONLY_MASK:
                pg.evaluate("()=>{location.hash='architecture'}"); pg.wait_for_timeout(900)
                rail_jump(pg, tag, w, lang)
                mask_part(pg, tag, w, lang)
                check(f'{tag} no page errors', not errs, errs[:3])
                pg.close(); continue
            # the hero
            # the owner's decision of 2026-10-11 19:28 UTC: the hero plays on its own, with no next button and no clickable stage
            check(f'{tag} hero: six stage names, one lit, nothing to click', pg.evaluate("()=>document.querySelectorAll('#heroLab .ar-hb').length") == 6
                  and pg.evaluate("()=>document.querySelectorAll('#heroLab .ar-hb.on').length") == 1
                  and pg.evaluate("()=>document.querySelectorAll('#heroLab button, #heroLab [data-hs], #heroLab [role=button]').length") == 0)
            check(f'{tag} hero: animating while on screen', st(pg)['hero']['on'])
            r = pg.evaluate("()=>{const e=document.querySelector('#heroLab'); const q=e.getBoundingClientRect(); return {bottom:q.bottom, sw:document.documentElement.scrollWidth-innerWidth}}")
            check(f'{tag} hero: no sideways scroll', r['sw'] <= 0, r)
            if w == 1440:
                check(f'{tag} hero: inside the first screen', r['bottom'] <= h, r)
            pg.screenshot(path=str(SH / f'flow_hero_{w}_{lang}.png'))
            pg.evaluate("()=>window.scrollTo(0,document.body.scrollHeight)"); pg.wait_for_timeout(500)
            check(f'{tag} hero: paused offscreen', not st(pg)['hero']['on'])
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(300)
            # the lab
            pg.evaluate("()=>{location.hash='architecture'}"); pg.wait_for_timeout(900)
            check(f'{tag} lab: 13 components to hover and click', pg.evaluate("()=>document.querySelectorAll('#ar-ov .ar-hot').length") == 13)
            check(f'{tag} lab: no sideways scroll', pg.evaluate("()=>document.documentElement.scrollWidth-innerWidth") <= 0)
            check(f'{tag} lab: no Settings panel; the model switch is on the stage', pg.query_selector('#ar-controls') is None and pg.evaluate("()=>document.querySelectorAll('#ar-stage .ar-mseg button').length") == 4)
            check(f'{tag} lab: the fidelity boundary is always on the stage', pg.is_visible('#ar-stage .ar-fchip') and ('Illustrative' in pg.inner_text('#ar-stage .ar-fchip') or '示意' in pg.inner_text('#ar-stage .ar-fchip')))
            if w <= 760:
                r = pg.evaluate("()=>{const s=document.querySelector('#ar-stage').getBoundingClientRect(), b=document.querySelector('#ar-rail .pbar [data-nav=\"next\"]').getBoundingClientRect(); return {stageTop:s.top, next:b.bottom, h:innerHeight}}")
                check(f'{tag} phone: the first screen shows the part bar with Next and the top of the picture', r['next'] < r['h'] and r['stageTop'] < r['h'] - 200, r)
            pg.screenshot(path=str(SH / f'flow_lab_{w}_{lang}.png'))
            for k, steps in [('emb', 6), ('attn', 7), ('mlp', 4), ('layer', 1), ('out', 3)]:
                click(pg, f'#ar-ov [data-pop="{k}"]')
                check(f'{tag} pop-up {k} opens', st(pg)['dlg'] == k and pg.is_visible('#ar-modal'))
                for s in range(1, steps):
                    click(pg, '#ar-stepper [data-step="1"]')
                check(f'{tag} pop-up {k}: the stepper reaches the last step', st(pg)['dst'] == steps - 1, st(pg)['dst'])
                if k == 'attn':
                    click(pg, '#ar-modal [data-key="amask"] [data-v="window"]')
                    t = pg.inner_text('#ar-dlg-body')
                    check(f'{tag} masks: the sliding window computes fewer pairs', '58' in t and '256' in t, t[:200])
                    click(pg, '#ar-modal [data-act="a1m"]')
                    check(f'{tag} masks: at 1M tokens, 128× fewer', '128' in pg.inner_text('#ar-dlg-body'))
                if k == 'layer':
                    t0 = pg.inner_text('#ar-dlg-body'); click(pg, '#ar-modal [data-act="nonorm"]'); t1 = pg.inner_text('#ar-dlg-body')
                    check(f'{tag} layer: without RMSNorm the size explodes', ('stable' in t0 or '稳定' in t0) and ('explodes' in t1 or '爆炸' in t1), (t0[-80:], t1[-80:]))
                    click(pg, '#ar-modal [data-act="nonorm"]')
                if k == 'out':
                    # step 1 is the LM head (one logit per vocabulary token), step 2 the probabilities, step 3 the pick
                    click(pg, '#ar-stepper [data-step="-1"]')
                    check(f'{tag} output step 2: logit sliders and temperature', pg.evaluate("()=>document.querySelectorAll('#ar-modal [data-z]').length") == 5 and pg.is_visible('#ar-modal #ar-temp-in'))
                    pg.evaluate("()=>{const i=document.querySelector('#ar-modal [data-z=\"2\"]'); i.value=10; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(150)
                    click(pg, '#ar-stepper [data-step="1"]')
                    g = pg.inner_text('#ar-modal .ar-greedy b')
                    check(f'{tag} output: a dragged logit changes the greedy pick', g.strip() == 'the', g)
                    click(pg, '#ar-modal [data-act="sample"]')
                    check(f'{tag} output: sampling picks a token', pg.evaluate("()=>!!document.querySelector('#ar-modal .ar-ofly.on')"))
                pg.screenshot(path=str(SH / f'flow_pop_{k}_{w}_{lang}.png'))
                pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
                check(f'{tag} pop-up {k}: Escape closes it', st(pg)['dlg'] is None)
            replay_checks(pg, tag, w, lang)
            # LM head, the probabilities and the next token open their own steps of the output pop-up (review 11)
            for hk, want in [('head', 0), ('prob', 1), ('next', 2)]:
                click(pg, f'#ar-ov [data-k="{hk}"]')
                check(f'{tag} {hk} opens the output pop-up at step {want + 1}', st(pg)['dlg'] == 'out' and st(pg)['dst'] == want, st(pg))
                pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
            # guided, scenarios, challenge
            check(f'{tag} guided: six parts', pg.evaluate("()=>document.querySelectorAll('#ar-rail .steps [data-go]').length") == 6)
            click(pg, '#ar-rail .steps [data-go="1"]')
            check(f'{tag} guided part 2: the picture lights the flow in turn', st(pg)['ov']['focus'] == ['tok', 'emb', 'stack', 'normF', 'head', 'prob', 'next'], st(pg)['ov'])
            click(pg, '#ar-rail .ar-deep [data-open="emb"]')
            check(f'{tag} guided part 2: a deep dive opens, titled as one', st(pg)['dlg'] == 'emb' and ('Deep dive' in pg.inner_text('#ar-dlg-title') or '深入' in pg.inner_text('#ar-dlg-title')), pg.inner_text('#ar-dlg-title'))
            pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
            check(f'{tag} guided part 2: the opened deep dive is ticked', '✓' in pg.inner_text('#ar-rail .ar-deep'))
            click(pg, '#ar-rail .steps [data-go="2"]')
            check(f'{tag} guided part 3: only attention lights up', st(pg)['ov']['focus'] == ['attn'], st(pg)['ov'])
            click(pg, '#ar-rail .steps [data-go="0"]')
            click(pg, '#ar-rail [data-pred="attn"]')
            check(f'{tag} guided part 1: the prediction gets an answer', pg.evaluate("()=>!!document.querySelector('#ar-rail .ar-fb.ok')"))
            rail_jump(pg, tag, w, lang)
            mask_part(pg, tag, w, lang)
            click(pg, '#view-dyn .modes button[data-mode="sandbox"]:not([data-flow=free])')
            click(pg, '#ar-rail [data-preset="3"]'); pg.wait_for_timeout(300)
            check(f'{tag} scenarios: the 405B caption', '126' in pg.inner_text('#ar-cap'), pg.inner_text('#ar-cap'))
            click(pg, '#view-dyn .modes button[data-mode="challenge"]')
            check(f'{tag} challenge: one question at a time', pg.evaluate("()=>document.querySelectorAll('#ar-rail .ar-chq').length") == 1)
            ans = pg.evaluate("()=>window.__ar.CHQ_CORE.map(q=>q.a).concat(window.__ar.CHQ.map(q=>q.a))")
            check(f'{tag} challenge: three core and six deeper questions', len(ans) == 9, ans)
            click(pg, '#ar-rail [data-cq="0"][data-v="0"]')
            check(f'{tag} challenge: a wrong pick is marked at once, with why', pg.evaluate("()=>!!document.querySelector('#ar-rail .ar-chr.no')"))
            for i, a in enumerate(ans):
                click(pg, f'#ar-rail [data-chi="{i}"]')
                click(pg, f'#ar-rail [data-cq="{i}"][data-v="{a}"]')
                if i == 1:
                    click(pg, '#ar-rail [data-show="1"]')
                    check(f'{tag} challenge: “Show me” for the position question opens attention step 1 (RoPE)', st(pg)['dlg'] == 'attn' and st(pg)['dst'] == 0, st(pg))
                    pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
            sc = pg.inner_text('#ar-rail .ar-chs')
            check(f'{tag} challenge: all right, core and deeper counted apart', ('3 of 3' in sc and '6 of 6' in sc) or ('3 题对了 3 题' in sc and '6 题对了 6 题' in sc), sc)
            # the owner's review of batch 8 (2026-10-07): the verdict comes first (✓ / ✗, then why); no listing of every option
            fr = pg.evaluate("()=>{const r=document.querySelector('#ar-rail .ar-chr'); return r?r.textContent.trim().slice(0,3):''}")
            check(f'{tag} challenge: the answer starts with the verdict', fr.startswith('✓'), fr)
            # the owner's clarification (2026-10-07 23:37 UTC): no hover only expands a shown number to its exact value
            ex = pg.evaluate("()=>[...document.querySelectorAll('[data-tipx]')].filter(e=>!e.closest('.mathbody')&&/精确值是|Exactly/.test(e.getAttribute('data-tipx')||'')).map(e=>e.getAttribute('data-tipx').slice(0,90))")
            check(f'{tag} no hover that only gives a shown number’s exact value (精确值是 / Exactly)', not ex, ex)
            click(pg, '#ar-rail [data-show="8"]')
            check(f'{tag} challenge: “Show me” for training without the mask opens the causal mask step', st(pg)['dlg'] == 'attn' and st(pg)['dst'] == 3, st(pg))
            pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
            click(pg, '#ar-rail [data-chi="7"]')   # the sliding-window question (no longer the last one since the ninth was added)
            click(pg, '#ar-rail [data-show="7"]')
            check(f'{tag} challenge: “Show me” for the sliding window opens the masks', st(pg)['dlg'] == 'attn' and st(pg)['dst'] == 6, st(pg))
            pg.keyboard.press('Escape'); pg.wait_for_timeout(150)
            check(f'{tag} no page errors', not errs, errs[:3])
            pg.close()
    # the owner's report (2026-10-04): on the first visit the pointer crossed the hero during the first tokenization pass, the stage
    # clock restarted, and the second pass was cut after about a second. The hero now moves on only where a pass of the stage ends.
    errs = []
    pg = page(b, 1440, 900, 'en', errs)
    hb = pg.evaluate("()=>{const r=document.querySelector('#heroLab').getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]}")
    h0 = st(pg)['hero']
    pg.mouse.move(hb[0], hb[1]); pg.wait_for_timeout(1000); pg.mouse.move(2, 2)
    seen, prev = None, None
    for _ in range(240):   # up to about 12 s
        h = st(pg)['hero']
        if h['cur'] != h0['cur']:
            seen = (prev, h); break
        prev = h; pg.wait_for_timeout(50)
    check('first visit: the pointer crossing the hero does not cut the first pass short; the stage changes where a pass ends',
          seen is not None and seen[0] is not None and seen[0]['ms'] % seen[0]['ht'] >= seen[0]['ht'] - 400 and seen[1]['cur'] == h0['cur'] + 1, seen)
    check('first visit: no page errors', not errs, errs[:3])
    pg.close()
    # reduced motion: one still frame, no auto-advance
    errs = []
    pg = page(b, 1440, 900, 'en', errs, reduce=True)
    c0 = st(pg)['hero']['cur']; pg.wait_for_timeout(5000)
    check('reduced motion: the hero does not animate or advance', not st(pg)['hero']['on'] and st(pg)['hero']['cur'] == c0, st(pg)['hero'])
    pg.evaluate("()=>{location.hash='architecture'}"); pg.wait_for_timeout(900)
    click(pg, '#ar-rail .steps [data-go="4"]'); click(pg, '#ar-rail [data-key="mkmask"] [data-v="off"]')
    an = pg.evaluate("()=>[...document.querySelectorAll('#ar-mk .ar-mkdraw,#ar-mk .ar-grow')].map(e=>getComputedStyle(e).animationName)")
    check('reduced motion: the causal switch draws the leak and the bars without animation', len(an) > 5 and all(a == 'none' for a in an), an[:4])
    check('reduced motion: no page errors', not errs, errs[:3])
    pg.close()
    b.close()
# G1 and G2, the owner's site-wide decisions of 2026-10-08 (sweep): option lead-ins instead of “one at a time” notes, and no sentence
# that leans on a detail of an earlier part. Read from the lab's source, so both languages are covered.
_G_SRC = (P / 'labs' / 'architecture.js').read_text()
for _bad in ['一次只能选', '一次选一个', '不会叠加', 'do not stack', 'One at a time', 'one at a time']:
    check(f'G1: no “{_bad}” left in the lab (EN and ZH)', _bad not in _G_SRC)
for _lead in ['Pick a model size and watch the drawing change.', '选一种模型大小，看图怎么变。']:
    check(f'G1: the lead-in “{_lead[:36]}…” is there', _lead in _G_SRC)
for _old, _new in [('in part 5 the probabilities and the training loss', 'the probabilities and the training loss of the causal-mask switch'), ('第 5 部分的概率和训练 loss', '因果 mask 开关那一步的概率和训练 loss')]:
    check(f'G2: “{_old[:30]}…” is rewritten as “{_new[:30]}…”', _old not in _G_SRC and _new in _G_SRC)
# the owner's decisions of 2026-10-09 (H1): every formula typeset with A.ui.math; none of the kinds converted left as plain text in the
# lab's source outside comments (the pop-up caption of the score step carries its formula in fx and shows it typeset)
import re as _re
_H_CODE = _re.sub(r'/\*.*?\*/', '', _G_SRC, flags=_re.S)
for _bad in ['divided by √d_head', '再除以 √d_head', 'q · k ÷ √d_head', '−ln p', "logits ÷ ${MK_K}", "logit ÷ ${MK_K}", 'logits ÷ 4', 'logit ÷ 4', '∝ √d']:
    check(f'H1: “{_bad}” is gone from the lab (typeset or said in words)', _bad not in _H_CODE)
check('H1: the score step’s caption formula is typeset (fx, A.ui.math in the caption)', "fx:'s = q \\\\cdot k \\\\div \\\\sqrt{d_{\\\\text{head}}}'" in _H_CODE and "pc.insertAdjacentHTML('beforeend',' '+A.ui.math(sp.fx))" in _H_CODE)
check('H1: the loss and the logit hovers use A.ui.math', _H_CODE.count("A.ui.math('-\\\\ln p')") >= 6 and _H_CODE.count("\\\\text{logit} \\\\div") >= 4, (_H_CODE.count("A.ui.math('-\\\\ln p')"), _H_CODE.count("\\\\text{logit} \\\\div")))
check('H1: the typeset “logit ÷ k” takes the value of k, not the text “${MK_K}”', "\\\\div ${MK_K}')" not in _H_CODE and _H_CODE.count("\\\\div '+MK_K)") == 4)
print(f'{n} checks, {len(fails)} failed')
sys.exit(1 if fails else 0)
