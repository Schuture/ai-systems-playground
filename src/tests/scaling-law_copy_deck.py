"""Dump every piece of text a learner meets in the scaling-law lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/scaling-law_copy_deck.py  ->  out/scaling-law/copy-en.md and out/scaling-law/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'scaling-law'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/scaling-law.html', '--only', 'scaling-law'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'scaling-law.html').read_text() + '</body></html>')
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
    out = [clean(txt(pg, '#sl-st-title')), clean(txt(pg, '#sl-st-sub')), 'Status: ' + clean(txt(pg, '#sl-status'))]
    for sel in ('#sl-vsw', '#sl-chart', '#sl-terms', '#sl-leg'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#sl-metrics .metric')].map(m=>m.innerText)")))
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
    for i in range(9):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzSl.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-sl'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='scaling-law'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#sl-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#sl-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#sl-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__sl.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#sl-rail .step.current') or txt(pg, '#sl-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#sl-controls'))
        D += tips(pg, '#sl-app')
        # the part's action
        if i == 0: click(pg, '#sl-rail [data-pred="small"]')
        elif i == 1: click(pg, '#sl-rail [data-nav="showme"]', 600)
        elif i == 2:
            for k in ('big', 'small'):
                click(pg, f'#sl-rail [data-lever="{k}"]'); D.append(f'\n## After {k}'); D += lines(txt(pg, '#sl-rail [data-obs]'))
            click(pg, '#sl-rail [data-lever="mid"]')
        elif i == 3:
            D.append('\n## Too small (before the move)'); D += lines(txt(pg, '#sl-rail [data-obs]'))
            pg.evaluate("()=>{const i=document.querySelector('#sl-ic-ln-in'); i.value=60; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
            D.append('\n## Too big'); D += lines(txt(pg, '#sl-rail [data-obs]'))
            click(pg, '#sl-rail [data-nav="showme"]', 600)
        elif i == 4:
            for v in ('1', '2'):
                click(pg, f'#sl-ic-T button[data-v="{v}"]'); D.append(f'\n## Served tokens option {v}'); D += lines(txt(pg, '#sl-rail [data-obs]')); D.append('### Stage'); D += stage(pg)
            click(pg, '#sl-ic-T button[data-v="3"]')
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#sl-rail .step.current') or txt(pg, '#sl-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('sl-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#sl-mathbody'))
            D += tips(pg, '#sl-app')
            click(pg, '#sl-rail [data-nav="next"]', 500)
    for k in ('iso', 'tot', 'nd'):
        click(pg, f'#sl-rail [data-rem="{k}"]'); D.append(f'\n## Part 6, show in the chart: {k}'); D += stage(pg); D += tips(pg, '#sl-app')
    D.append('\n# Scenarios')
    click(pg, '#sl-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#sl-rail'))
    for i in range(4):
        click(pg, f'#sl-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#sl-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#sl-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#sl-controls'))
    D.append('\n# Challenge')
    click(pg, '#sl-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#sl-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#sl-rail [data-act="submit"]')
    D.append('\n## Tier 3: 8B on 100B tokens'); D += lines(txt(pg, '#sl-ch-res'))
    click(pg, '#sl-rail [data-act="reveal"]')
    D.append('\n## Show the cheapest plan (the summary: your plan, the cheapest, the counterexample)'); D += lines(txt(pg, '#sl-ch-res')); D += tips(pg, '#sl-ch-res')
    click(pg, '#sl-rail .seg-ctl[data-key="chn"] button[data-v="3"]'); click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="1"]')
    D.append('\n## Tier 2: 20B on 300B tokens'); D += lines(txt(pg, '#sl-ch-res'))
    click(pg, '#sl-rail .seg-ctl[data-key="chn"] button[data-v="2"]'); click(pg, '#sl-rail .seg-ctl[data-key="chd"] button[data-v="2"]')
    D.append('\n## Tier 1: 8B on 1T tokens'); D += lines(txt(pg, '#sl-ch-res'))
    pg.evaluate("()=>{const d=document.querySelector('#sl-ch-res details.sl-all'); if(d) d.open=true}"); pg.wait_for_timeout(200)
    D.append('\n## See all settings that pass (opened)'); D += lines(txt(pg, '#sl-ch-res details.sl-all'))
    D.append('## Stage'); D += stage(pg)
    D += tips(pg, '#sl-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#sl-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#sl-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#sl-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#sl-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#sl-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#sl-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#sl-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#sl-app .connect'))
    D += tips(pg, '#sl-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
