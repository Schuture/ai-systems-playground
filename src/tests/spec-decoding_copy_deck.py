"""Dump every piece of text a learner meets in the speculative decoding lab, in reading order, in Chinese and English.

python3 tests/spec-decoding_copy_deck.py [tag]  ->  out/spd/copy/<tag>-zh.md and out/spd/copy/<tag>-en.md (default tag: now)

Order: lab header with the mode descriptions, each guided part (card before and after its actions, the stage, the tooltips in view that
were not shown before; part 3's round frame by frame), "Show the math", the scenarios card for every preset, the settings panel, the
challenge before and after submitting (every feedback tier and the chart), the capability layer and the connect section. The home tile
stays in teasers/set-a.js (not this lab's file), so it is not dumped here. A tooltip is printed the first time it appears with a given text.
"""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = P / 'out' / 'spd' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/spec-decoding.html', '--only', 'spec-decoding'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'spd' / 'wrapped-spd.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'spec-decoding.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'spd' / 'wrapped-spd.html')


def txt(pg, sel):
    el = pg.query_selector(sel)
    return el.inner_text().strip() if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#spd-st-title'), txt(pg, '#spd-status')] + [txt(pg, f'#spd-{x}') for x in ('cost', 'round', 'acc', 'kcurve', 'bcurve', 'time')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#spd-metrics .metric')].filter(m=>m.getClientRects().length).map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + (['Metrics:'] + [f'- {m}' for m in mets] if mets else []))


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


def card(pg): return txt(pg, '#spd-rail .step.current') or txt(pg, '#spd-rail')
def click(pg, sel, ms=300): pg.evaluate("s=>{const b=document.querySelector(s); if(b) b.click()}", sel); pg.wait_for_timeout(ms)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ('zh', 'en'):
        pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce'); T = Tips(); L = []
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        pg.evaluate("()=>{location.hash='spec-decoding'}"); pg.wait_for_timeout(600)
        L += ['# Header', txt(pg, '#spd-app .lab-head') or txt(pg, '#spd-app header')]
        L += ['## Mode descriptions'] + [f'- {x}' for x in pg.evaluate("()=>[...document.querySelectorAll('#spd-app .modes [data-mode]')].map(b=>b.innerText+': '+(b.dataset.tipx||''))")]
        acts = {1: ["#spd-rail [data-lever='5']", "#spd-rail [data-lever='80']", "#spd-rail [data-lever='640']"], 3: ["#spd-rail [data-pred='a34']"],
                4: ["#spd-rail [data-lever='2']", "#spd-rail [data-lever='8']"], 5: ["#spd-rail [data-lever='greedy']", "#spd-rail [data-lever='sample']", "#spd-rail [data-lever='nocheck']"],
                6: ["#spd-rail [data-lever='128']", "#spd-rail [data-lever='32']"]}
        for i in range(8):
            click(pg, f'#spd-rail [data-go="{i}"]')
            L += [f'# Part {i + 1}', '## Card', card(pg), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#spd-app')
            if i == 2:   # the round, frame by frame
                n = pg.evaluate("()=>+document.querySelector('#spd-rp-pos').max")
                for f in range(n + 1):
                    pg.evaluate(f"()=>{{const r=document.querySelector('#spd-rp-pos'); r.value={f}; r.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(120)
                    L += [f'### Round, frame {f}', txt(pg, '#spd-st-title'), txt(pg, '#spd-rows') or txt(pg, '#spd-round')]
            for a in acts.get(i, []):
                click(pg, a)
                L += [f'## After {a.split("[")[-1].rstrip("]")}', '### Card', card(pg), '### Stage', stage(pg)] + T.grab(pg, '#spd-app')
        pg.evaluate("()=>{document.querySelector('#spd-math').open=true}"); pg.wait_for_timeout(200)
        L += ['# Show the math (part 8)', txt(pg, '#spd-mathbody')]
        click(pg, '#spd-rail [data-mode-go="sandbox"]', 400)
        L += ['# Scenarios', txt(pg, '#spd-rail')]
        for i in range(5):
            click(pg, f'#spd-rail [data-preset="{i}"]')
            L += [f'## Scenario {i + 1}', txt(pg, '#spd-insight'), '### Stage', stage(pg)] + T.grab(pg, '#spd-app')
        pg.evaluate("()=>{document.querySelector('#spd-math').open=true}"); pg.wait_for_timeout(200)
        L += ['## Show the math (scenario 5)', txt(pg, '#spd-mathbody')]
        L += ['# Settings', txt(pg, '#spd-controls')] + T.grab(pg, '#spd-controls')
        click(pg, '#spd-app .modes [data-mode="challenge"]', 400)
        L += ['# Challenge', txt(pg, '#spd-rail'), '## Stage', stage(pg)]
        for sel, name in [('[data-act="submit"]', 'submitted at the start (3B, k = 4)'), ('#spd-ic-dr button[data-v="ngram"]', 'n-gram, k = 4'), ('#spd-ic-dr button[data-v="none"]', 'no guessing'),
                          ('#spd-ic-dr button[data-v="d1"]', '1B, k = 4'), ('#spd-ic-k button[data-v="8"]', '1B, k = 8'), ('#spd-ic-k button[data-v="1"]', '1B, k = 1'),
                          ('#spd-ic-dr button[data-v="d3"]', '3B, k = 1'), ('#spd-ic-k button[data-v="2"]', '3B, k = 2'), ('[data-act="reveal"]', 'all options'), ('#spd-ic-dr button[data-v="d1"]', '1B, k = 2 (best)')]:
            click(pg, f'#spd-rail {sel}')
            L += [f'## Challenge feedback: {name}', txt(pg, '#spd-ch-res')]
        L += ['# Capability layer', txt(pg, '#spd-rail [data-exits]')]
        L += ['# Connect', txt(pg, '#spd-app .connect')]
        (OUT / f'{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x) + '\n')
        print('wrote', OUT / f'{tag}-{lang}.md')
        pg.close()
    b.close()
