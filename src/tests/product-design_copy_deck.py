"""Dump every piece of text a learner meets in the product design capstone lab, in reading order, in Chinese and English.

python3 tests/product-design_copy_deck.py [tag]  ->  out/pdz/copy/<tag>-zh.md and out/pdz/copy/<tag>-en.md (default tag: now; the
'now' deck is also written to out/product-design-copy/<lang>.md for tests/numdense.py)

Order: lab header with the mode descriptions, each guided part (card before and after its actions, the stage, the tooltips in view that
were not shown before; part 1's message component by component), "Show the math", the scenarios card for every preset, the settings
panel, the challenge before and after submitting (every feedback tier and the chart), the capability layer and the connect section, the
drill. A tooltip is printed the first time it appears with a given text.
"""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = P / 'out' / 'pdz' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/product-design.html', '--only', 'product-design'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'pdz' / 'wrapped-copy.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'product-design.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'pdz' / 'wrapped-copy.html')


def txt(pg, sel):
    el = pg.query_selector(sel)
    return el.inner_text().strip() if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#pdz-st-title'), txt(pg, '#pdz-status')] + [txt(pg, f'#pdz-{x}') for x in ('trace', 'prompt', 'fleet', 'sheet')]
    svg = pg.evaluate("()=>[...document.querySelectorAll('#pdz-wf svg text')].filter(t=>t.getClientRects().length).map(t=>t.textContent.trim()).filter(Boolean).join(' / ')")
    mets = pg.evaluate("()=>[...document.querySelectorAll('#pdz-metrics .metric')].filter(m=>m.getClientRects().length).map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ([f'Drawing text: {svg}'] if svg else []) + (['Metrics:'] + [f'- {m}' for m in mets] if mets else []))


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .def'):
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:50]
                t = ' '.join((e.get_attribute('data-tipx') or e.get_attribute('data-tip') or '').split())
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {label} ⟶ {t}')
        return out


def card(pg): return txt(pg, '#pdz-rail .step.current') or txt(pg, '#pdz-rail')
def click(pg, sel, ms=350): pg.evaluate("s=>{const b=document.querySelector(s); if(b) b.click()}", sel); pg.wait_for_timeout(ms)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('zh', 'en'):
        pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); T = Tips(); L = []
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='product-design'}"); pg.wait_for_timeout(900)
        L += ['# Header', txt(pg, '#pdz-app .lab-head') or txt(pg, '#pdz-app header')]
        L += ['## Mode descriptions'] + [f'- {x}' for x in pg.evaluate("()=>[...document.querySelectorAll('#pdz-app .modes [data-mode]')].map(b=>b.innerText+': '+(b.dataset.tipx||''))")]
        acts = {1: ["[data-lkey='stream'][data-lever='0']", "[data-lkey='stream'][data-lever='1']"], 2: ["[data-lkey='cache'][data-lever='1']", "[data-lkey='cache'][data-lever='0']"],
                3: ["[data-pred='p37']"], 4: ["[data-lkey='share'][data-lever='0.7']", "[data-lkey='share'][data-lever='0.9']"],
                5: ["[data-lkey='filt'][data-lever='after']", "[data-lkey='filt'][data-lever='whole']", "[data-lkey='filt'][data-lever='16']", "[data-lkey='filt'][data-lever='4']"],
                6: ["[data-lkey='dau'][data-lever='100000']", "[data-lkey='dau'][data-lever='10000000']"]}
        for i in range(8):
            click(pg, f'#pdz-rail [data-go="{i}"]')
            L += [f'# Part {i + 1}', '## Card', card(pg), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#pdz-app')
            if i == 0:   # the message, component by component
                pg.evaluate("()=>{const r=document.querySelector('#pdz-pos'); r.value=0; r.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(150)
                for f in range(14):
                    L += [f'### Message, step {f}', txt(pg, '#pdz-st-title')]
                    if pg.evaluate("()=>window.__pdz.state().done"): break
                    click(pg, '#pdz-player [data-act="fwd"]', 150)
            for a in acts.get(i, []):
                click(pg, f'#pdz-rail {a}')
                L += [f'## After {a}', '### Card', card(pg), '### Stage', stage(pg)] + T.grab(pg, '#pdz-app')
        pg.evaluate("()=>{const m=document.querySelector('#pdz-math'); if(m) m.open=true}"); pg.wait_for_timeout(200)
        L += ['# Show the math (part 8)', txt(pg, '#pdz-mathbody')]
        click(pg, '#pdz-rail [data-mode-go="sandbox"]', 500)
        L += ['# Scenarios', txt(pg, '#pdz-rail')]
        for i in range(5):
            click(pg, f'#pdz-rail [data-preset="{i}"]', 450)
            L += [f'## Scenario {i + 1}', txt(pg, '#pdz-insight'), '### Stage', stage(pg)] + T.grab(pg, '#pdz-app')
        pg.evaluate("()=>{const m=document.querySelector('#pdz-math'); if(m) m.open=true}"); pg.wait_for_timeout(200)
        L += ['## Show the math (scenario 5)', txt(pg, '#pdz-mathbody')]
        L += ['# Settings', txt(pg, '#pdz-controls')] + T.grab(pg, '#pdz-controls')
        click(pg, '#pdz-app .modes [data-mode="challenge"]', 500)
        L += ['# Challenge', txt(pg, '#pdz-rail'), '## Stage', stage(pg)]
        for sel, name in [('[data-act="submit"]', 'submitted at the start'), ('#pdz-ic-stream button[data-v="0"]', 'sent when done'), ('#pdz-ic-stream button[data-v="1"]', 'streaming'),
                          ('#pdz-ic-filt button[data-v="after"]', 'checked after sending'), ('#pdz-ic-filt button[data-v="4"]', 'chunks of 4, no cache'),
                          ('#pdz-ic-share button[data-v="0.9"]', '90% small'), ('#pdz-ic-cache button[data-v="1"]', 'cache, 90%'), ('[data-act="reveal"]', 'all designs'),
                          ('#pdz-ic-share button[data-v="0.5"]', 'cache, 50%, chunks of 4 (met)'), ('#pdz-ic-share button[data-v="0.7"]', 'cache, 70%, chunks of 4 (best)')]:
            click(pg, f'#pdz-rail {sel}')
            L += [f'## Challenge feedback: {name}', txt(pg, '#pdz-ch-res')]
        L += ['# Capability layer', txt(pg, '#pdz-rail [data-exits]')]
        L += ['# Connect', txt(pg, '#pdz-app .connect')]
        click(pg, '#pdz-app [data-act="drill-reveal"]')
        L += ['# Drill', txt(pg, '#pdz-drill-q'), txt(pg, '#pdz-drill-a')]
        (OUT / f'{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x) + '\n')
        print('wrote', OUT / f'{tag}-{lang}.md')
        if tag == 'now':   # the deck tests/numdense.py reads
            ND = P / 'out' / 'product-design-copy'; ND.mkdir(parents=True, exist_ok=True)
            (ND / f'{lang}.md').write_text('\n\n'.join(x for x in L if x) + '\n')
        pg.close()
    b.close()
