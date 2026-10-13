"""Dump every piece of text a learner meets in the training-memory lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/training-memory_copy_deck.py  ->  out/training-memory/copy-en.md and out/training-memory/copy-zh.md

Order: home tile (the sentence and the hover at each of the four models, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier, the summary beside the fastest plan, the list of plans that pass opened), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'training-memory'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/training-memory.html', '--only', 'training-memory'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'training-memory.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
seen = set()


def txt(pg, sel):
    return pg.evaluate("s=>{const e=document.querySelector(s); return e&&e.getClientRects().length?e.innerText:''}", sel)


def lines(t):
    return [clean(x) for x in t.split('\n') if clean(x)]


def tips(pg, root):
    out = []
    for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>[(e.textContent||'').trim().slice(0,40),e.getAttribute('data-tipx')])", root):
        k = clean(t[1])
        if k and k not in seen:
            seen.add(k); out.append(f'- hover “{clean(t[0])}” → {k}')
    return out


def stage(pg):
    out = [clean(txt(pg, '#tmem-st-title')), clean(txt(pg, '#tmem-st-sub')), 'Status: ' + clean(txt(pg, '#tmem-status'))]
    for sel in ('#tmem-col-cap', '#tmem-cv-cap', '#tmem-cv-leg', '#tmem-phase'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Column: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tmem-col text')].map(t=>t.textContent)")))
    out.append('Curve: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tmem-curve text')].map(t=>t.textContent)") if not clean(x).isdigit()))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#tmem-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def to_end(pg):
    pg.evaluate("()=>{const i=document.querySelector('#tmem-player input[data-key=\"scrub\"]'); if(i){i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}}"); pg.wait_for_timeout(250)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for i in range(pg.evaluate("()=>window.__tzTmem.SIZES.length")):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzTmem.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
        D.append(f'  - hover: {clean(pg.evaluate("i=>window.__tzTmem.help(i)", i))}')
    pg.click('#domStrip .ds-btn[data-world="training"]'); pg.wait_for_timeout(300)
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-tmem'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='training-memory'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#tmem-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#tmem-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#tmem-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__tmem.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#tmem-rail .step.current') or txt(pg, '#tmem-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#tmem-controls'))
        D += tips(pg, '#tmem-app')
        if i == 0: click(pg, '#tmem-rail [data-pred="80-100"]')
        elif i == 1:
            click(pg, '#tmem-rail [data-nav="showme"]', 600)
            D.append('\n## After the step played'); D += lines(txt(pg, '#tmem-rail [data-obs]'))
            for o in ('adam8', 'sgdm'):
                click(pg, f'#tmem-rail [data-lever="opt:{o}"]')
                D.append(f'\n## Optimizer: {o}'); D += lines(txt(pg, '#tmem-rail [data-obs]'))
            click(pg, '#tmem-rail [data-lever="opt:adamw"]')
        elif i == 2:
            pg.evaluate("()=>{const i=document.querySelector('#tmem-ic-seq-in'); i.value=4; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
        elif i == 3:
            D.append('\n## Keep everything (before the pick)'); D += lines(txt(pg, '#tmem-rail [data-obs]'))
            click(pg, '#tmem-rail [data-lever="rc:1"]'); to_end(pg)
        elif i == 4:
            D.append('\n## All 8 at once (before the pick)'); D += lines(txt(pg, '#tmem-rail [data-obs]'))
            click(pg, '#tmem-rail [data-lever="ac:a2x4"]'); to_end(pg)
            D.append('\n## 4 micro-batches of 2'); D += lines(txt(pg, '#tmem-rail [data-obs]'))
            click(pg, '#tmem-rail [data-lever="ac:all"]'); to_end(pg)
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#tmem-rail .step.current') or txt(pg, '#tmem-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('tmem-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#tmem-mathbody'))
            D += tips(pg, '#tmem-app')
            click(pg, '#tmem-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#tmem-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#tmem-rail'))
    for i in range(4):
        click(pg, f'#tmem-rail [data-preset="{i}"]', 400); to_end(pg)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#tmem-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#tmem-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#tmem-controls'))
    D.append('\n# Challenge')
    click(pg, '#tmem-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#tmem-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#tmem-rail [data-act="submit"]')
    D.append('\n## Tier 3: AdamW, no recomputation, 1 at a time'); D += lines(txt(pg, '#tmem-ch-res'))
    click(pg, '#tmem-rail [data-act="reveal"]')
    D.append('\n## Show the fastest plan'); D += lines(txt(pg, '#tmem-ch-res'))
    pg.evaluate("()=>{const d=document.querySelector('#tmem-ch-res details.tmem-all'); if(d) d.open=true}"); pg.wait_for_timeout(150)
    D.append('\n## See all settings that pass (opened)'); D += lines(txt(pg, '#tmem-ch-res details.tmem-all'))
    D += tips(pg, '#tmem-ch-res')
    click(pg, '#tmem-rail .seg-ctl[data-key="cho"] button[data-v="adam8"]'); click(pg, '#tmem-rail .seg-ctl[data-key="chr"] button[data-v="1"]'); click(pg, '#tmem-rail .seg-ctl[data-key="chm"] button[data-v="4"]')
    D.append('\n## Tier 2: 8-bit AdamW, recompute, 4 at a time'); D += lines(txt(pg, '#tmem-ch-res'))
    click(pg, '#tmem-rail .seg-ctl[data-key="chr"] button[data-v="0"]'); click(pg, '#tmem-rail .seg-ctl[data-key="chm"] button[data-v="2"]')
    D.append('\n## Tier 1: 8-bit AdamW, no recomputation, 2 at a time'); D += lines(txt(pg, '#tmem-ch-res'))
    D += tips(pg, '#tmem-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#tmem-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#tmem-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#tmem-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#tmem-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#tmem-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#tmem-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#tmem-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#tmem-app .connect'))
    D += tips(pg, '#tmem-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
