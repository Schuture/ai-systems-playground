"""Dump every piece of text a learner meets in the parallelism lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/parallelism_copy_deck.py  ->  out/parallelism/copy-en.md and out/parallelism/copy-zh.md

Order: home tile (the sentence at every GPU count, the notes), lab header and its popovers, each guided part (card before and after
its action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge
(every feedback tier, your setting beside the best one, the settings that pass), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text. Page errors are printed and fail the run."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'parallelism'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/parallelism.html', '--only', 'parallelism'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'parallelism.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
seen, errs = set(), []


def txt(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?e.innerText:''}", sel)


def lines(t):
    return [clean(x) for x in t.split('\n') if clean(x)]


def tips(pg, root):
    out = []
    for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>[(e.innerText||'').trim().slice(0,40),e.getAttribute('data-tipx')])", root):
        k = clean(t[1])
        if k and k not in seen:
            seen.add(k); out.append(f'- hover “{clean(t[0])}” → {k}')
    return out


def stage(pg):
    out = [clean(txt(pg, '#zr-st-title')), clean(txt(pg, '#zr-st-sub')), 'Status: ' + clean(txt(pg, '#zr-status'))]
    for sel in ('#zr-mem-cap', '#zr-cols', '#zr-mem-leg', '#zr-tl-cap', '#zr-now', '#zr-tl-leg'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#zr-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def caption(pg):
    """the caption above the legend in each pass: scrub into the forward pass, the backward pass, then back to the end"""
    out = []
    for v, name in ((80, 'forward'), (650, 'backward'), (1000, 'end')):
        pg.evaluate(f"""()=>{{const i=document.querySelector('#zr-player input[data-key="scrub"]'); if(!i) return; i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}));}}"""); pg.wait_for_timeout(120)
        out.append(f'- {name}: ' + ' / '.join(clean(x) for x in txt(pg, '#zr-now').split('\n') if clean(x)))
    return out


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def finish(pg):
    """let the step's replay end, so the part's result shows"""
    pg.evaluate("()=>{const b=document.querySelector('#zr-spd button[data-v=\"fast\"]'); if(b) b.click()}")
    try: pg.wait_for_function("()=>{const s=window.__zr.state(); return !s.playing}", timeout=9000)
    except Exception: pass
    pg.wait_for_timeout(200)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(f'{lang}: {e}'))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    # the tile lives in the training world of the home tree: open it, so the label, legend, notes and help text are read too
    pg.evaluate("()=>{const b=document.querySelector('button[data-world=\"training\"]'); if(b) b.click()}"); pg.wait_for_timeout(400)
    for n in (1, 2, 4, 8, 16, 32, 64):
        D.append(f'- {clean(pg.evaluate("n=>window.__tzZr.read(n)", n)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-zr'); return t?[...t.querySelectorAll('.ctl .lbl,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    D += [f'- hover tile → {clean(x)}' for x in pg.evaluate("()=>[...document.querySelectorAll('.tz-zr [data-tipx]')].map(e=>e.getAttribute('data-tipx'))")]
    pg.evaluate("()=>{location.hash='parallelism'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#zr-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#zr-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#zr-app .labhead')
    for i in range(7):
        sid = pg.evaluate("()=>window.__zr.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#zr-rail .step.current') or txt(pg, '#zr-rail'))
        if i == 1: finish(pg)
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#zr-controls'))
        D += tips(pg, '#zr-app')
        if i == 0: click(pg, '#zr-rail [data-pred="q18"]')
        elif i == 2: click(pg, '#zr-rail .seg-ctl[data-key="st13"] button[data-v="z1"]')
        elif i == 3: click(pg, '#zr-rail .seg-ctl[data-key="st24"] button[data-v="z2"]')
        elif i == 4:
            finish(pg); D.append('\n## Fetch when needed (before the pick)'); D += lines(txt(pg, '#zr-rail [data-obs]'))
            click(pg, '#zr-rail [data-lever="ov"]'); finish(pg)
        elif i == 5:
            finish(pg); D.append('\n## ZeRO-3 over 64 GPUs (before the pick)'); D += lines(txt(pg, '#zr-rail [data-obs]'))
            click(pg, '#zr-rail [data-lever="tok"]'); finish(pg)
            D.append('\n## More tokens'); D += lines(txt(pg, '#zr-rail [data-obs]'))
            click(pg, '#zr-rail [data-lever="net"]'); finish(pg)
        if i < 6:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#zr-rail .step.current') or txt(pg, '#zr-rail'))
            D.append('## Stage'); D += stage(pg)
            if i >= 1: D.append('## The caption above the legend, in each pass'); D += caption(pg)
            pg.evaluate("()=>{document.getElementById('zr-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#zr-mathbody'))
            D += tips(pg, '#zr-app')
            click(pg, '#zr-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#zr-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#zr-rail'))
    for i in range(4):
        click(pg, f'#zr-rail [data-preset="{i}"]', 400); finish(pg)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#zr-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#zr-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#zr-controls'))
    D += tips(pg, '#zr-controls')
    D.append('\n# Challenge')
    click(pg, '#zr-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#zr-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#zr-rail [data-act="submit"]')
    D.append('\n## Tier 3: ZeRO-2, 4,096 tokens'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-rail [data-act="reveal"]')
    D.append('\n## Show the best setting: your setting beside the best one'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-ch-res details.zr-all summary')
    D.append('\n## See all settings that pass (opened)'); D += lines(txt(pg, '#zr-ch-res .zr-allb'))
    D += tips(pg, '#zr-ch-res')
    click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="65536"]')
    D.append('\n## Tier 3: ZeRO-2, 65,536 tokens'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-rail .seg-ctl[data-key="chs"] button[data-v="z3"]'); click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="4096"]')
    D.append('\n## Tier 3: ZeRO-3, 4,096 tokens'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="32768"]')
    D.append('\n## Tier 3: ZeRO-3, 32,768 tokens'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="16384"]')
    D.append('\n## Tier 2: ZeRO-3, 16,384 tokens'); D += lines(txt(pg, '#zr-ch-res'))
    click(pg, '#zr-rail .seg-ctl[data-key="cht"] button[data-v="8192"]')
    D.append('\n## Tier 1: ZeRO-3, 8,192 tokens'); D += lines(txt(pg, '#zr-ch-res'))[:4]
    D += tips(pg, '#zr-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#zr-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#zr-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#zr-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#zr-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#zr-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#zr-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#zr-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#zr-app .connect'))
    D += tips(pg, '#zr-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
print('page errors:', errs if errs else 'none')
sys.exit(1 if errs else 0)
