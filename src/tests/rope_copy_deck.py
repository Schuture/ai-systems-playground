"""Dump every piece of text a learner meets in the rope lab and its home tile, in reading order, in English and Chinese
(CONTRACT §4.0: export the whole text in reading order and read it before reporting).

python3 tests/rope_copy_deck.py  ->  out/rope-copy/en.md and out/rope-copy/zh.md

Order: home tile (the sentence at several settings, the legend, the notes), lab header (title, outcome, mode descriptions, fidelity and
primer popovers), each of the 7 guided parts (the card before and after each action, the stage: titles, status, legend, caption,
note, the comparison of part 2, metric cards with their tooltips, Show the math), part 4's verdict for a wrong and for the right pick,
every scenario, the settings panel, the challenge before and after submitting (every tier and the chart), the capability card (every
question, a verdict for every option), and Connect with the drill's answer. A tooltip is printed the first time its text appears."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'rope-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rope-deck.html', '--only', 'rope'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-rope-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rope-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-rope-deck.html')
clean = lambda s: ' '.join((s or '').replace('‑', '-').replace(' ', ' ').split())


def txt(pg, sel):
    return clean(pg.evaluate("s=>{const e=document.querySelector(s); return e&&e.offsetParent!==null?e.innerText:''}", sel))


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root='#rp-app'):
        out = []
        for t in pg.evaluate(f"()=>[...document.querySelectorAll('{root} [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))"):
            t = clean(t)
            if t and t not in self.seen: self.seen.add(t); out.append('  - tip: ' + t)
        return out


def stage(pg, tips):
    L = [f"Stage: {txt(pg, '#rp-st-title')} / {txt(pg, '#rp-st-sub')} [status: {txt(pg, '#rp-status')}]",
         'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#rp-legend > span')].map(s=>s.innerText)")),
         'Caption: ' + txt(pg, '#rp-vis .rp-cap')]
    note = txt(pg, '#rp-vis .rp-note')
    if note: L.append('Note: ' + note)
    L.append('SVG text: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('#rp-vis svg text')].map(t=>t.textContent).join(' | ')"))[:900])
    cmp_ = txt(pg, '#rp-cmp')
    if cmp_: L.append('Comparison: ' + cmp_)
    L.append('Scope line: ' + txt(pg, '.rp-scope'))
    for m in pg.evaluate("()=>[...document.querySelectorAll('#rp-metrics .metric')].map(m=>[m.querySelector('.k').innerText,m.querySelector('.v').innerText,m.querySelector('.sub').innerText,m.querySelector('.d').innerText,m.dataset.tipx])"):
        L.append(f'- Card: {clean(m[0])} = {clean(m[1])} ({clean(m[2])}){" " + clean(m[3]) if m[3] else ""}; tip: {clean(m[4])}')
    pg.evaluate("()=>{const d=document.getElementById('rp-math'); d.open=true;}")
    L.append('Show the math: ' + txt(pg, '#rp-mathbody'))
    pg.evaluate("()=>{const d=document.getElementById('rp-math'); d.open=false;}")
    dock = pg.evaluate("()=>{const d=document.getElementById('dock'); if(!d) return ''; const r=d.querySelector('.dk-read')||d; return r.innerText}")
    L += tips.grab(pg)
    return '\n'.join(L)


def deck(lang):
    lines, tips = [], Tips()
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        # the home tile
        pg.evaluate("()=>{location.hash='map~models'}"); pg.wait_for_timeout(700)
        tile = pg.query_selector('#tile-rope')
        lines.append('# Home tile')
        if tile:
            lines.append('Title: ' + clean(tile.query_selector('.tile-q').inner_text()))
            for k_, v in enumerate(('off', 'on')):
                pg.evaluate(f"()=>document.querySelectorAll('#tile-rope .seg-ctl button')[{k_}].click()"); pg.wait_for_timeout(80)
                lines.append(f'- RoPE {v}: ' + clean(tile.query_selector('.tz-read').inner_text()))
            lines.append('Legend: ' + clean(tile.query_selector('.tz-leg').inner_text()))
            lines += ['Note: ' + clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tile-rope .tz-note')].map(n=>n.textContent)")]
            lines.append('Visual tip: ' + clean(pg.evaluate("()=>document.querySelector('#tile-rope svg').getAttribute('data-tipx')")))
        else:
            lines.append('(tile not found)')
        # the lab header
        pg.evaluate("()=>{location.hash='rope'}"); pg.wait_for_timeout(600)
        lines += ['', '# Header', 'Breadcrumb: ' + txt(pg, '#rp-app .crumbs'), 'Title: ' + txt(pg, '#rp-app .labtitle'), 'Outcome: ' + txt(pg, '#rp-app .outcome') if pg.query_selector('#rp-app .outcome') else '',
                  'Tags: ' + txt(pg, '#rp-app .labmeta'), 'Mode description: ' + txt(pg, '#rp-app [data-mode-desc]')]
        lines += ['Mode tips: ' + ' || '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#rp-app .modes button')].map(b=>b.dataset.tipx)"))]
        lines += ['Notes at the page end: ' + clean(pg.evaluate("()=>{const e=document.getElementById('rope-fid'); return e?e.innerText:''}"))]   # no Primer popover since 2026-10-11
        # the guided parts
        seg = lambda k, v: ("click", f'#rp-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]')
        acts = {1: [seg('hv', 'pairs'), seg('hv', 'arrows')], 2: [seg('rope3', 'on')],
                3: [("click", '#rp-rail [data-pred="little"]'), ("slide", '#rp-ic-shift-in', 100), ("click", '#rp-rail [data-pred="same"]')],
                4: [("slide", '#rp-ic-l5-in', 0), seg('b5', '10000'), ("slide", '#rp-ic-l5-in', 5)],
                5: [("slide", '#rp-ic-l6-in', 4), seg('s6', '1'), ("slide", '#rp-ic-l6-in', 3)]}
        for i in range(7):
            pg.evaluate(f"()=>document.querySelector('#rp-rail [data-go=\"{i}\"]').click()"); pg.wait_for_timeout(250)
            lines += ['', f'# Guided part {i + 1}', 'Bar: ' + txt(pg, '#rp-rail .pbar .pb-t'), 'Card: ' + txt(pg, '#rp-rail .step.current .step-b')]
            lines += tips.grab(pg, '#rp-rail')
            lines.append(stage(pg, tips))
            if i == 3 and not pg.evaluate("()=>window.__rp.state().pred"):
                pass
            for a in acts.get(i, []):
                if a[0] == 'click': pg.evaluate(f"()=>document.querySelector('{a[1]}').click()")
                else: pg.evaluate(f"()=>{{const i=document.querySelector('{a[1]}'); i.value={a[2]}; i.dispatchEvent(new Event('input',{{bubbles:true}}));}}")
                pg.wait_for_timeout(250)
                lines += [f'## after {a[1]} {a[2] if len(a) > 2 else ""}', 'Result: ' + txt(pg, '#rp-rail [data-obs]'), stage(pg, tips)]
                if i == 3: lines.append('Card: ' + txt(pg, '#rp-rail .step.current .step-b'))
        # scenarios
        pg.evaluate("()=>document.querySelector('#rp-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(350)
        lines += ['', '# Scenarios', 'Card: ' + txt(pg, '#rp-rail .railcard')]
        for i in range(5):
            pg.evaluate(f"()=>document.querySelector('#rp-rail [data-preset=\"{i}\"]').click()"); pg.wait_for_timeout(250)
            lines += [f'## Scenario {i + 1}', 'Result: ' + txt(pg, '#rp-insight'), stage(pg, tips)]
        lines += ['', '# Settings', txt(pg, '#rp-controls')] + tips.grab(pg, '#rp-controls')
        # the challenge
        pg.evaluate("()=>document.querySelector('#rp-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(350)
        lines += ['', '# Challenge', 'Card: ' + txt(pg, '#rp-rail .railcard'), stage(pg, tips), 'Settings: ' + txt(pg, '#rp-controls')]
        pg.evaluate("()=>document.querySelector('#rp-rail [data-act=\"submit\"]').click()"); pg.wait_for_timeout(250)
        lines += ['## Submitted at 10,000', txt(pg, '#rp-ch-res')]
        pg.evaluate("()=>document.querySelector('#rp-ch-res [data-act=\"reveal\"]').click()"); pg.wait_for_timeout(250)
        for bb in (100000, 1000000, 500000):
            pg.evaluate(f"()=>document.querySelector('#rp-ic-ch [data-lever=\"{bb}\"]').click()"); pg.wait_for_timeout(250)
            lines += [f'## Base {bb}', txt(pg, '#rp-ch-res')]
        lines += tips.grab(pg, '#rp-rail')
        # the capability card
        lines += ['', '# Can you use it on a new case?']
        for t in range(3):
            pg.evaluate(f"()=>document.querySelector('[data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(200)
            if t == 2:
                lines.append(txt(pg, '[data-exits] .ex-body')); break
            nq = pg.evaluate("()=>document.querySelectorAll('[data-exits] .ex-qs .pb-dot').length") or 1
            for j in range(nq):
                pg.evaluate(f"()=>{{const d=document.querySelectorAll('[data-exits] .ex-qs .pb-dot')[{j}]; if(d) d.click();}}"); pg.wait_for_timeout(150)
                opts = pg.evaluate("()=>[...document.querySelectorAll('[data-exits] .ex-opts .choice')].map(b=>b.dataset.v)")
                lines.append(f'## Tab {t + 1}, question {j + 1}: ' + txt(pg, '[data-exits] .ex-body > p'))
                for v in opts:
                    pg.evaluate(f"()=>document.querySelector('[data-exits] .ex-opts .choice[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
                    lines.append(f'- pick {v}: ' + txt(pg, '[data-exits] .ex-res'))
        # connect
        pg.evaluate("()=>document.querySelector('#rp-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(150)
        lines += ['', '# Connect', txt(pg, '#rp-app .connect')]
        lines += tips.grab(pg)
        b.close()
    (OUT / f'{lang}.md').write_text('\n'.join(x for x in lines if x is not None))
    return len(lines)


for lang in ('en', 'zh'):
    print(lang, deck(lang), 'lines ->', OUT / f'{lang}.md')
