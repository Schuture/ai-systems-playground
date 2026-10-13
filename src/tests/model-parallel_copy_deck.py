"""Dump every piece of text a learner meets in the model-parallel lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/model-parallel_copy_deck.py  ->  out/model-parallel/copy-en.md and out/model-parallel/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not printed before, Show the math), the scenarios, the settings panel, the challenge (every
feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'model-parallel'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/model-parallel.html', '--only', 'model-parallel'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'model-parallel.html').read_text() + '</body></html>')
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
    out = [clean(txt(pg, '#mp-st-title')), clean(txt(pg, '#mp-st-sub')), 'Status: ' + clean(txt(pg, '#mp-status'))]
    for sel in ('#mp-phone', '#mp-cl-cap', '#mp-mem', '#mp-g-cap', '#mp-gleg', '#mp-player', '#mp-l-cap', '#mp-tp', '#mp-lt-cap', '#mp-lt', '#mp-ltleg'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#mp-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def finish(pg):
    pg.evaluate("()=>{const b=document.querySelector('#mp-player .seg-ctl[data-key=spd] button[data-v=fast]'); if(b) b.click();}")
    try: pg.wait_for_function("()=>window.__mp.state().ph>=1&&!window.__mp.state().playing", timeout=12000)
    except Exception: pass


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for i in range(6):
        D.append(f'- {clean(pg.evaluate("i=>window.__tzMp.read(i)", i)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-mp'); return t?[...t.querySelectorAll('.tz-lab span,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='model-parallel'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#mp-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#mp-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#mp-app .labhead')
    for i in range(7):
        sid = pg.evaluate("()=>window.__mp.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#mp-rail .step.current') or txt(pg, '#mp-rail'))
        D.append('## Stage'); D += stage(pg)
        D.append('## Settings'); D += lines(txt(pg, '#mp-controls'))
        D += tips(pg, '#mp-app')
        if i == 0: click(pg, '#mp-rail [data-pred="one"]')
        elif i == 1: finish(pg); click(pg, '#mp-ic-m button[data-v="32"]', 300); finish(pg)
        elif i == 2: finish(pg); click(pg, '#mp-ic-lv3 [data-lever="1f1b"]', 300); finish(pg)
        elif i == 3: click(pg, '#mp-ic-t button[data-v="8"]')
        elif i == 4:
            D.append('\n## In one machine (before the pick)'); D += lines(txt(pg, '#mp-rail [data-obs]'))
            click(pg, '#mp-ic-lv5 [data-lever="out"]')
        if i < 6:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#mp-rail .step.current') or txt(pg, '#mp-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('mp-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#mp-mathbody'))
            D += tips(pg, '#mp-app')
            click(pg, '#mp-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#mp-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#mp-rail'))
    for i in range(4):
        click(pg, f'#mp-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#mp-insight')); D.append('### Stage'); D += stage(pg)
        pg.evaluate("()=>{document.getElementById('mp-math').open=true}"); pg.wait_for_timeout(100)
        D.append('### Show the math'); D += lines(txt(pg, '#mp-mathbody'))
        D += tips(pg, '#mp-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#mp-controls'))
    D.append('\n# Challenge')
    click(pg, '#mp-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#mp-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#mp-rail [data-act="submit"]')
    D.append('\n## Does not fit: the start, TP 8 × PP 1, 8 micro-batches'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-rail [data-act="reveal"]')
    D.append('\n## Show the fastest layout (R1 summary)'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-ic-chm button[data-v="64"]')
    D.append('\n## Does not fit, one change makes it fit: TP 8 × PP 1, 64 micro-batches (G7)'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-ic-chp button[data-v="4"]'); click(pg, '#mp-ic-chm button[data-v="256"]')
    D.append('\n## Fits, not the fastest: TP 8 × PP 4, 256 micro-batches'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-ic-chp button[data-v="2"]')
    D.append('\n## Fits, no single change helps: TP 8 × PP 2, 128 micro-batches (the Codex pick, G7)'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-ic-chm button[data-v="64"]')
    D.append('\n## Fits, one change helps a little: TP 8 × PP 2, 64 micro-batches (G7)'); D += lines(txt(pg, '#mp-ch-res'))
    click(pg, '#mp-ic-chp button[data-v="4"]'); click(pg, '#mp-ic-cht button[data-v="4"]'); click(pg, '#mp-ic-chm button[data-v="128"]')
    D.append('\n## Goal met: TP 4 × PP 4, 128 micro-batches (the disclosure opened)'); pg.evaluate("()=>{const d=document.querySelector('#mp-ch-res details'); if(d) d.open=true}"); D += lines(txt(pg, '#mp-ch-res'))
    D += tips(pg, '#mp-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#mp-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#mp-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#mp-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#mp-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#mp-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#mp-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#mp-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#mp-app .connect'))
    D += tips(pg, '#mp-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
