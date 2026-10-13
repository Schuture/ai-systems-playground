"""Dump every piece of text a learner meets in the agent-anatomy lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/agent-anatomy_copy_deck.py  ->  out/agent-anatomy/copy-en.md and out/agent-anatomy/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-anatomy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-anatomy.html', '--only', 'agent-anatomy'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-anatomy.html').read_text() + '</body></html>')
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
    out = [clean(txt(pg, '#ana-st-title')), clean(txt(pg, '#ana-st-sub')), 'Status: ' + clean(txt(pg, '#ana-status'))]
    for sel in ('#ana-phone', '#ana-loop', '#ana-cap2', '#ana-conv-cap', '#ana-conv-leg', '#ana-conv', '#ana-bars-cap', '#ana-bars-leg', '#ana-bars', '#ana-time-cap', '#ana-time-leg', '#ana-time', '#ana-ch-cap', '#ana-ch-leg'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#ana-metrics .metric')].map(m=>m.innerText)")))
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
    for i in range(5):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzAna.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-ana'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='agent-anatomy'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#ana-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ana-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#ana-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__ana.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#ana-rail .step.current') or txt(pg, '#ana-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#ana-controls'))
        D += tips(pg, '#ana-app')
        if i == 0: click(pg, '#ana-rail [data-pred="mid"]')
        elif i == 1:
            pg.wait_for_timeout(1200); D.append('\n## Replay, mid-way'); D.append('### Stage'); D += stage(pg); D += tips(pg, '#ana-app')
            click(pg, '#ana-rail [data-nav="showme"]', 600)
        elif i == 2:
            for k in (2, 5, 8):
                pg.evaluate(f"()=>{{const i=document.querySelector('#ana-ic-call-in'); i.value={k}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(300)
                if k < 8: D.append(f'\n## Call {k}'); D += lines(txt(pg, '#ana-rail [data-obs]')); D += stage(pg)[3:6]
        elif i == 3:
            D.append('\n## Send the error back (before the pick)'); D += lines(txt(pg, '#ana-rail [data-obs]'))
            click(pg, '#ana-ic-fx [data-lever="stop"]'); D.append('\n## Stop on any tool error'); D += lines(txt(pg, '#ana-rail [data-obs]')); D.append('### Stage'); D += stage(pg); D += tips(pg, '#ana-app')
            click(pg, '#ana-ic-fx [data-lever="terse"]')
        elif i == 4:
            D.append('\n## One tool per call (before the pick)'); D += lines(txt(pg, '#ana-rail [data-obs]'))
            click(pg, '#ana-ic-par [data-lever="par"]')
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#ana-rail .step.current') or txt(pg, '#ana-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('ana-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#ana-mathbody'))
            D += tips(pg, '#ana-app')
            click(pg, '#ana-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#ana-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#ana-rail'))
    for i in range(4):
        click(pg, f'#ana-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#ana-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#ana-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#ana-controls'))
    D += tips(pg, '#ana-controls')
    D.append('\n# Challenge')
    click(pg, '#ana-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#ana-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    pg.evaluate("()=>{document.getElementById('ana-math').open=true}"); pg.wait_for_timeout(100)
    D.append('## Show the math'); D += lines(txt(pg, '#ana-mathbody'))
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="100000"]')
    click(pg, '#ana-rail [data-act="submit"]')
    D.append('\n## Tier 3: limit 50, budget 100,000'); D += lines(txt(pg, '#ana-ch-res'))
    click(pg, '#ana-rail [data-act="reveal"]')
    D.append('\n## Show the answer'); D += lines(txt(pg, '#ana-ch-res'))
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="300000"]')
    D.append('\n## Tier 2: limit 50, budget 300,000 (the initial setup)'); D += lines(txt(pg, '#ana-ch-res'))[:4]
    click(pg, '#ana-rail .seg-ctl[data-key="chb"] button[data-v="150000"]')
    D.append('\n## Tier 1: limit 50, budget 150,000'); D += lines(txt(pg, '#ana-ch-res'))[:4]
    D.append('## Stage'); D += stage(pg)
    D += tips(pg, '#ana-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#ana-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#ana-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#ana-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#ana-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#ana-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#ana-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#ana-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#ana-app .connect'))
    D += tips(pg, '#ana-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
