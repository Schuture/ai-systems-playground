"""Dump every piece of text a learner meets in the lora lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/lora_copy_deck.py  ->  out/lora/copy-en.md and out/lora/copy-zh.md

Order: home tile (the sentence for every method, the legend, the notes), lab header and its popovers, each guided part (card before
and after its action, the stage, the tooltips in view not printed before), Show the math, the scenarios, the settings panel, the
challenge (every feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'lora'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/lora.html', '--only', 'lora'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'lora.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
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
    out = [clean(txt(pg, '#lora-st-title')), clean(txt(pg, '#lora-st-sub')), 'Status: ' + clean(txt(pg, '#lora-status'))]
    for sel in ('#lora-mem-cap', '#lora-mem-leg', '#lora-mat-cap', '#lora-mat-note'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Memory chart: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#lora-mem text')].map(t=>t.textContent)")))
    out.append('Matrix: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#lora-mat text')].map(t=>t.textContent)")))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#lora-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def slide(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(300)


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    for m in ('full', 'lora', 'qlora'):
        D.append(f'- {clean(pg.evaluate("m=>window.__tzLora.read(m)", m)).replace("<b>", "").replace("</b>", "")}')
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-lora'); return t?[...t.querySelectorAll('.seg-ctl,.tz-leg,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    pg.evaluate("()=>{location.hash='lora'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#lora-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#lora-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#lora-app .labhead')
    for i in range(8):
        sid = pg.evaluate("()=>window.__lora.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#lora-rail .step.current') or txt(pg, '#lora-rail'))
        D.append('## Stage'); D += stage(pg)
        pg.evaluate("()=>{const d=document.getElementById('lora-asm'); if(d) d.open=true}"); pg.wait_for_timeout(100)
        D.append('## Settings'); D += lines(txt(pg, '#lora-controls'))
        D += tips(pg, '#lora-app')
        if i == 0: pass
        elif i == 1:
            click(pg, '#lora-rail [data-nav="showme"]', 400)
        elif i == 2: click(pg, '#lora-rail [data-pred="10-30"]')
        elif i == 3:
            click(pg, '#lora-rail [data-lever="l3:lora"]')
        elif i == 4:
            slide(pg, '#lora-ic-r-in', 4)
            for tg in ('qv', 'attn', 'all'):
                click(pg, f'#lora-rail .seg-ctl[data-key="tg"] button[data-v="{tg}"]'); click(pg, '#lora-mat-segw button[data-v="k"]')
                D.append(f'\n## LoRA on {tg}: matrix k'); D += lines(txt(pg, '#lora-mat-cap')) + lines(txt(pg, '#lora-mat-note')) + tips(pg, '#lora-mat')
        elif i == 5:
            slide(pg, '#lora-ic-seq-in', 4)
            D.append('\n## 16,384 tokens, no recomputation'); D += lines(txt(pg, '#lora-rail [data-obs]'))
            click(pg, '#lora-rail .seg-ctl[data-key="rc"] button[data-v="1"]')
        elif i == 6:
            click(pg, '#lora-rail [data-lever="l6:qnorc"]')
            D.append('\n## QLoRA without recomputation'); D += lines(txt(pg, '#lora-rail [data-obs]'))
            click(pg, '#lora-rail [data-lever="l6:qlora"]')
        elif i == 7:
            for k in ('small', 'acts', 'rank', 'q4'):
                click(pg, f'#lora-rail [data-rem="{k}"]')
                D.append(f'\n## Show in the chart: {k}'); D += stage(pg)
        if i < 7:
            D.append(f'\n# Guided part {sid + 1}: after the action')
            D += lines(txt(pg, '#lora-rail .step.current') or txt(pg, '#lora-rail'))
            D.append('## Stage'); D += stage(pg)
            pg.evaluate("()=>{document.getElementById('lora-math').open=true}"); pg.wait_for_timeout(100)
            D.append('## Show the math'); D += lines(txt(pg, '#lora-mathbody'))
            D += tips(pg, '#lora-app')
            click(pg, '#lora-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#lora-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#lora-rail'))
    for i in range(4):
        click(pg, f'#lora-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#lora-insight')); D.append('### Stage'); D += stage(pg)
        D += tips(pg, '#lora-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#lora-controls'))
    D.append('\n# Challenge')
    click(pg, '#lora-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#lora-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#lora-rail [data-act="submit"]')
    D.append('\n## Tier 3: the initial setup'); D += lines(txt(pg, '#lora-ch-res'))
    click(pg, '#lora-rail .seg-ctl[data-key="chm"] button[data-v="qlora"]')
    D.append('\n## Tier 3: QLoRA, no recomputation'); D += lines(txt(pg, '#lora-ch-res'))
    click(pg, '#lora-rail .seg-ctl[data-key="chm"] button[data-v="lora"]')
    click(pg, '#lora-rail [data-act="reveal"]')
    D.append('\n## Show the answer'); D += lines(txt(pg, '#lora-ch-res'))
    pg.evaluate("()=>{const d=document.getElementById('lora-allp'); if(d) d.open=true}"); pg.wait_for_timeout(200)
    D.append('\n## See all settings that pass (opened)'); D += lines(txt(pg, '#lora-allp'))
    click(pg, '#lora-rail .seg-ctl[data-key="chm"] button[data-v="qlora"]'); click(pg, '#lora-rail .seg-ctl[data-key="chc"] button[data-v="1"]'); click(pg, '#lora-rail .seg-ctl[data-key="cht"] button[data-v="all"]'); slide(pg, '#lora-ic-chr-in', 2)
    D.append('\n## Tier 2: QLoRA, recompute, all 7, r = 16'); D += [x for x in lines(txt(pg, '#lora-ch-res')) if x][:24]
    slide(pg, '#lora-ic-chr-in', 1)
    D.append('\n## Tier 1: r = 8'); D += [x for x in lines(txt(pg, '#lora-ch-res')) if x][:18]
    D += tips(pg, '#lora-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#lora-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#lora-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#lora-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#lora-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#lora-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#lora-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#lora-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#lora-app .connect'))
    D += tips(pg, '#lora-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
