"""Dump every piece of text a learner meets in the train-capstone lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/train-capstone_copy_deck.py  ->  out/train-capstone/copy-en.md and out/train-capstone/copy-zh.md

Order: home tile (the sentence at every position, the notes), lab header and its popovers, each guided part (card before and after
its actions, every option's result, the stage, the tooltips in view not printed before, Show the math), the scenarios, the settings
panel, the challenge (every feedback tier and the chart), the card "Can you use it on a new case?" (each tab), and Connect with the
drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'train-capstone'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-capstone.html', '--only', 'train-capstone'], cwd=P, check=True, capture_output=True)
(OUT / 'wrapped-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'train-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'wrapped-deck.html')
clean = lambda s: ' '.join(str(s).replace('‑', '-').replace(' ', ' ').split())
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
    out = [clean(txt(pg, '#tcap-st-title')), clean(txt(pg, '#tcap-st-sub')), 'Status: ' + clean(txt(pg, '#tcap-status'))]
    for sel in ('#tcap-phone', '#tcap-cl-cap', '#tcap-cl-note', '#tcap-mem-cap', '#tcap-mem', '#tcap-step-cap', '#tcap-step-list', '#tcap-run-cap', '#tcap-run-list', '#tcap-real-cap', '#tcap-real'):
        t = txt(pg, sel)
        if t: out.append(clean(t))
    out.append('Metrics: ' + ' || '.join(clean(m) for m in pg.evaluate("()=>[...document.querySelectorAll('#tcap-metrics .metric')].map(m=>m.innerText)")))
    return [x for x in out if x]


def click(pg, sel, wait=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(wait)


def show_math(pg, D, head='## Show the math'):
    pg.evaluate("()=>{document.getElementById('tcap-math').open=true}"); pg.wait_for_timeout(100)
    D.append(head); D += lines(txt(pg, '#tcap-mathbody'))


ACTIONS = {0: [('pred', '1to3')], 1: [('tcap-ic-lv2', '0.3'), ('tcap-ic-lv2', '0.5')], 2: [('tcap-ic-lv3', 'z3'), ('tcap-ic-lv3', 'tp8z3'), ('tcap-ic-lv3', 'tp8pp8z1')],
           3: [('tcap-ic-lv4', 'rc'), ('tcap-ic-lv4', 'tp4pp16'), ('tcap-ic-lv4', 'tp8z3'), ('tcap-ic-lv4', 'tp8z1')], 4: [('tcap-ic-lv5', '1s'), ('tcap-ic-lv5', '15s'), ('tcap-ic-lv5', '15a')]}


def deck(b, lang):
    D = []
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
    seen.clear()
    D.append('# Home tile')
    pg.evaluate("()=>{const b=document.querySelector('[data-world=\"training\"]'); if(b) b.click();}"); pg.wait_for_timeout(400)
    tile = pg.evaluate("()=>{const t=document.querySelector('.tz-tcap'); return t?[...t.querySelectorAll('.tz-lab span,.tz-note')].map(e=>e.textContent):[]}")
    D += [f'- {clean(x)}' for x in tile if clean(x)]
    for i in range(7):
        pg.evaluate(f"()=>{{const r=document.querySelector('#tz-tcap-in'); if(r){{r.value={i}; r.dispatchEvent(new Event('input'));}}}}"); pg.wait_for_timeout(60)
        D.append(f'- {clean(txt(pg, ".tz-tcap .tz-read"))}')
    tz_tip = pg.evaluate("()=>{const v=document.querySelector('.tz-tcap-vis'); return v?v.getAttribute('data-tipx'):''}")
    D.append(f'- tooltip: {clean(tz_tip)}')
    pg.evaluate("()=>{location.hash='train-capstone'}"); pg.wait_for_timeout(500)
    D.append('\n# Lab header')
    D += lines(txt(pg, '#tcap-app .labhead'))
    for pid in pg.evaluate("()=>[...document.querySelectorAll('#tcap-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
        D.append(f'\n## Popover {pid}')
        D += lines(pg.evaluate(f"()=>document.getElementById('{pid}').innerText||document.getElementById('{pid}').textContent"))
    D += tips(pg, '#tcap-app .labhead')
    for i in range(7):
        sid = pg.evaluate("()=>window.__tcap.state().step")
        D.append(f'\n# Guided part {sid + 1}: card')
        D += lines(txt(pg, '#tcap-rail .step.current') or txt(pg, '#tcap-rail'))
        D.append('## Stage'); D += stage(pg)
        if i == 0:
            D.append('## Settings'); D += lines(txt(pg, '#tcap-controls'))
        D += tips(pg, '#tcap-app')
        for box, v in ACTIONS.get(i, []):
            if box == 'pred': click(pg, f'#tcap-rail [data-pred="{v}"]')
            else: click(pg, f'#{box} [data-lever="{v}"]')
            D.append(f'\n## Part {sid + 1} after picking “{v}”')
            D += lines(txt(pg, '#tcap-rail [data-obs]') if box != 'pred' else txt(pg, '#tcap-rail .step.current'))
            D.append('### Stage'); D += stage(pg)
            D += tips(pg, '#tcap-app')
        if i in ACTIONS:
            D.append('\n## Options with their results'); D += lines(txt(pg, '#tcap-rail .tcap-lv')) if i > 0 else []
        show_math(pg, D)
        D += tips(pg, '#tcap-app')
        if i < 6: click(pg, '#tcap-rail [data-nav="next"]', 500)
    D.append('\n# Scenarios')
    click(pg, '#tcap-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 500)
    D += lines(txt(pg, '#tcap-rail'))
    for i in range(5):
        click(pg, f'#tcap-rail [data-preset="{i}"]', 400)
        D.append(f'\n## Scenario {i + 1}'); D += lines(txt(pg, '#tcap-insight')); D.append('### Stage'); D += stage(pg)
        show_math(pg, D, '### Show the math')
        D += tips(pg, '#tcap-app')
    D.append('\n# Settings panel (Scenarios)'); D += lines(txt(pg, '#tcap-controls'))
    D += tips(pg, '#tcap-app')
    D.append('\n# Challenge')
    click(pg, '#tcap-app .modes button[data-mode="challenge"]', 500)
    D += lines(txt(pg, '#tcap-rail .railcard'))
    D.append('## Stage'); D += stage(pg)
    click(pg, '#tcap-rail [data-act="submit"]')
    D.append('\n## Not yet: the start, no split (does not fit)'); D += lines(txt(pg, '#tcap-ch-res'))
    D += tips(pg, '#tcap-ch-res')

    def pick(t, p, z, rc, ivl, asy):
        for k, v in (('t', t), ('p', p), ('z', z), ('rc', rc), ('ivl', ivl), ('asy', asy)):
            click(pg, f'#tcap-ic-ch{k} button[data-v="{v}"]', 150)
    pick(8, 1, 3, 0, '0.25', 1)
    D.append('\n## Not yet: TP 8, ZeRO-3, every 15 minutes async (fits, too slow)'); D += lines(txt(pg, '#tcap-ch-res'))
    click(pg, '#tcap-rail [data-act="reveal"]')
    D.append('\n## The same, after “Show the recommended plan”'); D += lines(txt(pg, '#tcap-ch-res'))
    D += tips(pg, '#tcap-ch-res')
    pick(8, 1, 1, 0, '1', 0)
    D.append('\n## Met, not the cheapest: TP 8, ZeRO-1, every hour sync'); D += lines(txt(pg, '#tcap-ch-res'))
    pick(8, 4, 1, 0, '0.25', 1)
    D.append('\n## Met, not the cheapest: the review’s plan, TP 8 × PP 4, ZeRO-1, every 15 minutes async (Codex review b9v3, 应修 5)'); D += lines(txt(pg, '#tcap-ch-res'))
    D += tips(pg, '#tcap-ch-res')
    pg.evaluate("()=>{const d=document.querySelector('#tcap-ch-res details.tcap-all'); if(d) d.open=true;}"); pg.wait_for_timeout(250)
    D.append('\n### “See all plans”, opened (the first rows of the list)')
    body = lines(txt(pg, '#tcap-ch-res .tcap-allb'))
    D += body[:20] + ['…'] + [x for x in body if not x[:1].isdigit()][-3:]
    pg.evaluate("()=>{const d=document.querySelector('#tcap-ch-res details.tcap-all'); if(d) d.open=false;}"); pg.wait_for_timeout(150)
    pick(2, 1, 3, 0, '0.25', 1)
    D.append('\n## Goal met, the cheapest: TP 2, ZeRO-3, every 15 minutes async'); D += lines(txt(pg, '#tcap-ch-res'))
    D.append('## Stage'); D += stage(pg)
    D += tips(pg, '#tcap-app')
    D.append('\n# Can you use it on a new case?')
    tabs = pg.evaluate("()=>document.querySelectorAll('#tcap-rail [data-exits] .ex-tabs button').length")
    D += lines(txt(pg, '#tcap-rail [data-exits]'))
    for t in range(tabs):
        pg.evaluate(f"()=>document.querySelectorAll('#tcap-rail [data-exits] .ex-tabs button')[{t}].click()"); pg.wait_for_timeout(200)
        D.append(f'## Tab {t + 1}'); D += lines(txt(pg, '#tcap-rail [data-exits]'))
        for q in range(3):
            nx = pg.evaluate("()=>{const b=[...document.querySelectorAll('#tcap-rail [data-exits] button')].find(b=>/Next question|下一题/.test(b.textContent)&&b.getClientRects().length); if(b){b.click(); return true} return false}")
            if not nx: break
            pg.wait_for_timeout(200); D += lines(txt(pg, '#tcap-rail [data-exits]'))
    D.append('\n# Connect')
    click(pg, '#tcap-app [data-act="drill-reveal"]')
    D += lines(txt(pg, '#tcap-app .connect'))
    D += tips(pg, '#tcap-app')
    pg.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(D) + '\n')
    return len(D)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('en', 'zh'):
        print(lang, deck(b, lang), 'lines ->', OUT / f'copy-{lang}.md')
    b.close()
