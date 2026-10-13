"""Dump every piece of text a learner meets in the eval-ci lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/eval-ci_copy_deck.py  ->  out/eval-ci/copy-en.md and out/eval-ci/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'eval-ci'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-ci.html', '--only', 'eval-ci'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'eval-ci.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
seen = set()


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
    out = [clean(txt(pg, '#evc-st-title')), clean(txt(pg, '#evc-st-sub')), 'Status: ' + clean(txt(pg, '#evc-status'))]
    for sel in ('#evc-phone', '#evc-ax-cap', '#evc-ax', '#evc-pile', '#evc-player', '#evc-grid-cap', '#evc-grid', '#evc-gleg', '#evc-mix-cap', '#evc-mix', '#evc-kc-cap', '#evc-kc', '#evc-df-cap', '#evc-df'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#evc-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for i in range(7):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzEvc.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-evc'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='eval-ci'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#evc-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#evc-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#evc-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__evc.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#evc-rail .step.current') or txt(pg, '#evc-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#evc-controls'))
        D += tips(pg, '#evc-app')
        # the part's action
        if i == 0: click(pg, '#evc-rail [data-pred="three"]')
        elif i == 1: click(pg, '#evc-rail [data-nav="showme"]', 600)
        elif i == 2:
            pg.evaluate("()=>{const i=document.querySelector('#evc-ic-n-in'); i.value=12; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
        elif i == 3:
            D.append('\n## Each on its own (before the pick)'); D += lines(txt(pg, '#evc-rail [data-obs]'))
            click(pg, '#evc-rail [data-lever="pair"]')
        elif i == 4: click(pg, '#evc-ic-k .seg-ctl button[data-v="16"]')
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#evc-rail .step.current') or txt(pg, '#evc-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('evc-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#evc-mathbody'))
            D += tips(pg, '#evc-app')
            click(pg, '#evc-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#evc-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#evc-rail'))
    for i in range(4):
        click(pg, f'#evc-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#evc-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#evc-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#evc-controls'))
    D.append('\n# Challenge')
    click(pg, '#evc-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#evc-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#evc-rail [data-act="submit"]')
    D.append('\n## Tier 3: 200, each on its own, 1 answer'); D += lines(txt(pg, '#evc-ch-res'))
    click(pg, '#evc-rail [data-act="reveal"]')
    D.append('\n## Show the answer'); D += lines(txt(pg, '#evc-ch-res'))
    click(pg, '#evc-rail .seg-ctl[data-key="chn"] button[data-v="4000"]')
    D.append('\n## Tier 2: 4,000, each on its own, 1 answer'); D += lines(txt(pg, '#evc-ch-res'))[:4]
    click(pg, '#evc-rail .seg-ctl[data-key="chn"] button[data-v="500"]'); click(pg, '#evc-rail .seg-ctl[data-key="chm"] button[data-v="1"]'); click(pg, '#evc-rail .seg-ctl[data-key="chk"] button[data-v="4"]')
    D.append('\n## Tier 1: 500, question by question, 4 answers'); D += lines(txt(pg, '#evc-ch-res'))[:4]
    D += tips(pg, '#evc-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#evc-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#evc-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#evc-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#evc-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#evc-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#evc-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#evc-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#evc-app .connect'))
    D += tips(pg, '#evc-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
