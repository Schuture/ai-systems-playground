"""Dump every piece of text a learner meets in the checkpointing lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/checkpointing_copy_deck.py  ->  out/checkpointing/copy-en.md and out/checkpointing/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'checkpointing'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/checkpointing.html', '--only', 'checkpointing'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'checkpointing.html').read_text() + '</body></html>')
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
    out = [clean(txt(pg, '#ckp-st-title')), clean(txt(pg, '#ckp-st-sub')), 'Status: ' + clean(txt(pg, '#ckp-status'))]
    for sel in ('#ckp-terms', '#ckp-box', '#ckp-leg', '#ckp-phase'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Chart: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#ckp-chart text')].map(t=>t.textContent)") if not clean(x).replace('.', '').isdigit()))
    fs = txt(pg, '#ckp-chart .ckp-fs')
    if fs: out.append('Files: ' + ' | '.join(lines(fs)))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#ckp-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def to_end(pg):
    pg.evaluate("()=>{const i=document.querySelector('#ckp-player input[data-key=\"scrub\"]'); if(i){i.value=1000; i.dispatchEvent(new Event('input',{bubbles:true}));}}"); pg.wait_for_timeout(250)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for i in range(11):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzCkp.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-ckp'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='checkpointing'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#ckp-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#ckp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#ckp-app .labhead')
    for i in range(6):
        sid = pg.evaluate("()=>window.__ckp.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#ckp-rail .step.current') or txt(pg, '#ckp-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#ckp-controls'))
        D += tips(pg, '#ckp-app')
        if i == 0: click(pg, '#ckp-rail [data-pred="bump"]')
        elif i == 1:
            to_end(pg)
            D.append('\n## After the run played (weights only)'); D += lines(txt(pg, '#ckp-rail [data-obs]'))
            for k in ('o', 's', 'd'):
                click(pg, f'#ckp-rail [data-part="{k}"]'); to_end(pg)
                D.append(f'\n## Added: {k}'); D += lines(txt(pg, '#ckp-rail [data-obs]'))
            click(pg, '#ckp-rail [data-part="r"]'); to_end(pg)
        elif i == 2:
            D.append('\n## Overwrite (before the pick)'); D += lines(txt(pg, '#ckp-rail [data-obs]')); D += stage(pg)
            click(pg, '#ckp-rail [data-lever="atom:atomic"]')
        elif i == 3:
            D.append('\n## Synchronous every 3 h (before the action)'); D += lines(txt(pg, '#ckp-rail [data-obs]'))
            pg.evaluate("()=>{const i=document.querySelector('#ckp-ic-T-in'); i.value=2; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
            click(pg, '#ckp-rail .seg-ctl[data-key="mode"] button[data-v="async"]')
        elif i == 4:
            to_end(pg)
            D.append('\n## Latest (500)'); D += lines(txt(pg, '#ckp-rail [data-obs]')); D += stage(pg)
            click(pg, '#ckp-rail [data-lever="fix:back"]'); to_end(pg)
            D.append('\n## Back to 400'); D += lines(txt(pg, '#ckp-rail [data-obs]'))
            click(pg, '#ckp-rail [data-lever="fix:skip"]'); to_end(pg)
        if i < 5:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#ckp-rail .step.current') or txt(pg, '#ckp-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('ckp-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#ckp-mathbody'))
            D += tips(pg, '#ckp-app')
            click(pg, '#ckp-rail [data-nav="next"]', 500)
        else:
            for k in ('atomic', 'interval', 'weights'):
                click(pg, f'#ckp-rail [data-rem="{k}"]'); D.append(f'## Show in the chart: {k}'); D += stage(pg)
    D.append('\n# Scenarios')
    click(pg, '#ckp-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#ckp-rail'))
    for i in range(4):
        click(pg, f'#ckp-rail [data-preset="{i}"]', 400); to_end(pg)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#ckp-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#ckp-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#ckp-controls'))
    D.append('\n# Challenge')
    click(pg, '#ckp-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#ckp-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#ckp-rail [data-act="submit"]')
    D.append('\n## Not yet: synchronous, every hour, keep 2'); D += lines(txt(pg, '#ckp-ch-res'))
    click(pg, '#ckp-rail [data-act="reveal"]')
    D.append('\n## The plan that wastes least'); D += lines(txt(pg, '#ckp-ch-res'))
    pg.evaluate("()=>{const d=document.querySelector('#ckp-ch-res details.ckp-all'); if(d) d.open=true}"); pg.wait_for_timeout(150)
    D.append('\n## All settings that pass (opened)'); D += lines(txt(pg, '#ckp-ch-res details.ckp-all'))
    click(pg, '#ckp-rail .seg-ctl[data-key="chm"] button[data-v="async"]'); click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="5"]')
    D.append('\n## Not yet: async, 5 min, keep 2'); D += lines(txt(pg, '#ckp-ch-res'))[:2]
    click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="20"]'); click(pg, '#ckp-rail .seg-ctl[data-key="chk"] button[data-v="4"]')
    D.append('\n## Goal met, a better plan: async, 20 min, keep 4'); D += lines(txt(pg, '#ckp-ch-res'))[:3]
    click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="15"]'); click(pg, '#ckp-rail .seg-ctl[data-key="chk"] button[data-v="5"]')
    D.append('\n## The best: async, 15 min, keep 5'); D += lines(txt(pg, '#ckp-ch-res'))[:3]
    click(pg, '#ckp-rail .seg-ctl[data-key="chT"] button[data-v="60"]'); click(pg, '#ckp-rail .seg-ctl[data-key="chk"] button[data-v="6"]')
    D.append('\n## Not yet: async, 1 h, keep 6 (over 30 TB)'); D += lines(txt(pg, '#ckp-ch-res'))[:2]
    D.append('## Stage'); D += stage(pg)
    D += tips(pg, '#ckp-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#ckp-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#ckp-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#ckp-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#ckp-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#ckp-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#ckp-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#ckp-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#ckp-app .connect'))
    D += tips(pg, '#ckp-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
