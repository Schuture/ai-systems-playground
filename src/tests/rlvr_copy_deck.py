"""Dump every piece of text a learner meets in the rlvr lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/rlvr_copy_deck.py  ->  out/rlvr/copy-en.md and out/rlvr/copy-zh.md

Order: home tile (both choices, the notes), lab header and both popovers, each guided part (card before and after its action, the stage,
the metrics), Show the math, the scenarios (each), Settings, the challenge (every feedback tier and the chart), the card "Can you use it
on a new case?" (every question, every option's verdict), and Connect with the drill answer. A tooltip is printed the first time it
appears with a given text. The page runs with reduced motion, so every run and group is shown at its end at once."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'rlvr'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/rlvr.html', '--only', 'rlvr'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'rlvr.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''


def txt(pg, sel):
    return pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).map(e=>e.innerText).join('\\n')", sel)


class Deck:
    def __init__(self): self.lines, self.tips = [], set()

    def h(self, t): self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = (t or '').strip()
        if t: self.lines.append(f'**{label}**\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            if c and c not in self.tips: self.tips.add(c); new.append('- ' + c)
        if new: self.lines.append('Tooltips (new):\n' + '\n'.join(new) + '\n')


def stage(d, pg):
    d.add('Stage', '\n'.join([txt(pg, '#rlvr-st-title'), txt(pg, '#rlvr-st-sub'), 'Status: ' + txt(pg, '#rlvr-status'), txt(pg, '#rlvr-chk'), txt(pg, '#rlvr-grp .rlvr-q'), txt(pg, '#rlvr-gstat'), txt(pg, '#rlvr-rows').replace('\n', ' '),
                             txt(pg, '#rlvr-sel'), txt(pg, '#rlvr-mix'), txt(pg, '#rlvr-p1'), txt(pg, '#rlvr-len'), txt(pg, '#rlvr-pk'), txt(pg, '.rlvr-scope')]))
    d.add('Metrics', '\n'.join(' | '.join(x for x in m.split('\n') if x.strip()) for m in pg.evaluate("()=>[...document.querySelectorAll('#rlvr-metrics .metric')].filter(m=>m.getClientRects().length).map(m=>m.innerText)")))
    d.tipsOf(pg, '#rlvr-stage')


def run(lang):
    d = Deck(); d.lines.append(f'# rlvr copy deck ({lang})\n')
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
        cl = lambda s: (pg.evaluate(f"()=>{{const e=document.querySelector('{s}'); if(e) e.click();}}"), pg.wait_for_timeout(250))
        d.h('Home tile')
        for m in ('strict', 'loose'):
            d.add(f'Tile, {m} checker', pg.evaluate(f"()=>window.__tzRlvr?window.__tzRlvr.read('{m}').replace(/<[^>]+>/g,''):''"))
        d.add('Tile notes and legend', pg.evaluate("()=>window.__tzRlvr?window.__tzRlvr.notes().join('\\n')+'\\nLegend: '+window.__tzRlvr.legend().join(' | '):''"))
        pg.evaluate("()=>{location.hash='rlvr'}"); pg.wait_for_timeout(500)
        d.h('Header')
        d.add('Header', txt(pg, '#rlvr-app .labhead') or txt(pg, '#rlvr-app header'))
        for k in ('fid', 'pri'):
            pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"rlvr-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(200)
            d.add(f'Popover {k}', pg.evaluate(f"()=>{{const p=document.getElementById('rlvr-{k}'); return p?p.innerText:''}}"))
            pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"rlvr-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
        steps = [('mission', [('pick “step by step grows”', '#rlvr-rail [data-pred="S"]')]),
                 ('checker', [('step by step, right', '#rlvr-rail [data-lever="1"]'), ('answer only, wrong', '#rlvr-rail [data-lever="0"]'), ('candidate numbers', '#rlvr-rail [data-lever="3"]')]),
                 ('group', [('next problem', '#rlvr-rail [data-act="grp-next"]'), ('next problem', '#rlvr-rail [data-act="grp-next"]')]),
                 ('train', [('run end', '#rlvr-rplay')]),
                 ('loose', [('loose', '#rlvr-rail [data-lever="loose"]'), ('strict', '#rlvr-rail [data-lever="strict"]')]),
                 ('format', [('format reward on', '#rlvr-rail [data-lever="1"]')]), ('passk', []), ('remember', [])]
        for i, (sid, acts) in enumerate(steps):
            if i: cl('#rlvr-rail [data-nav="next"]')
            d.h(f'Part {i + 1}: {sid}')
            d.add('Card', txt(pg, '#rlvr-rail .step.current')); d.tipsOf(pg, '#rlvr-rail'); stage(d, pg)
            for name, sel in acts:
                cl(sel); pg.wait_for_timeout(200)
                d.add(f'After “{name}”: result box', txt(pg, '#rlvr-rail [data-obs]') or txt(pg, '#rlvr-rail .step.current')); d.tipsOf(pg, '#rlvr-rail'); stage(d, pg)
        d.h('Show the math')
        pg.evaluate("()=>{document.getElementById('rlvr-math').open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#rlvr-mathbody'))
        d.h('Scenarios')
        pg.evaluate("()=>document.querySelector('#rlvr-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#rlvr-rail'))
        for k in range(4):
            cl(f'#rlvr-rail [data-preset="{k}"]'); d.add(f'Scenario {k + 1}', txt(pg, '#rlvr-insight')); stage(d, pg)
        d.h('Settings'); d.add('Settings (assumed values closed)', txt(pg, '#rlvr-controls'))
        pg.evaluate("()=>{const e=document.getElementById('rlvr-asm-d'); if(e) e.open=true}"); pg.wait_for_timeout(150)
        d.add('Settings (assumed values opened)', txt(pg, '#rlvr-asm-d')); d.tipsOf(pg, '#rlvr-controls')
        d.h('Challenge')
        pg.evaluate("()=>document.querySelector('#rlvr-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#rlvr-rail .railcard')); stage(d, pg)
        cl('#rlvr-rail [data-act="submit"]'); d.add('Not met (loose)', txt(pg, '#rlvr-ch-res'))
        cl('#rlvr-rail .seg-ctl[data-key="chC"] button[data-v="strict"]'); d.add('Not met (strict)', txt(pg, '#rlvr-ch-res'))
        cl('#rlvr-ch-res [data-act="reveal"]'); d.add('Show the answer', txt(pg, '#rlvr-ch-res'))
        cl('#rlvr-rail .seg-ctl[data-key="chD"] button[data-v="1"]'); cl('#rlvr-rail .seg-ctl[data-key="chF"] button[data-v="1"]'); d.add('Met, but a smaller change exists', txt(pg, '#rlvr-ch-res'))
        cl('#rlvr-rail .seg-ctl[data-key="chF"] button[data-v="0"]'); d.add('Met with the fewest changes', txt(pg, '#rlvr-ch-res')); stage(d, pg); d.tipsOf(pg, '#rlvr-rail')
        d.h('Can you use it on a new case?')
        for t in range(3):
            pg.evaluate(f"()=>{{const b=document.querySelector('[data-exits] [data-ex=\"tab\"][data-i=\"{t}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(200)
            if t == 2:
                d.add('Tab 3 (Go further)', txt(pg, '[data-exits]')); break
            for j in range(2):
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-exits] [data-ex=\"q\"][data-t=\"{t}\"][data-j=\"{j}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
                nopt = pg.evaluate(f"()=>document.querySelectorAll('[data-exits] [data-ex=\"ans\"][data-t=\"{t}\"]').length")
                for v in range(nopt):
                    pg.evaluate(f"()=>{{const b=document.querySelector('[data-exits] [data-ex=\"ans\"][data-t=\"{t}\"][data-v=\"{v}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(120)
                    d.add(f'Tab {t + 1}, question {j + 1}, option {v + 1}', txt(pg, '[data-exits]'))
        d.h('Connect')
        pg.evaluate("()=>{const b=document.querySelector('#rlvr-app [data-act=\"drill-reveal\"]'); if(b) b.click();}"); pg.wait_for_timeout(150)
        d.add('Connect', txt(pg, '#rlvr-app .connect'))
        b.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(d.lines))
    print('wrote', OUT / f'copy-{lang}.md', sum(len(x) for x in d.lines), 'chars')


for lang in [x for x in ('en', 'zh') if x in sys.argv[1:]] or ['en', 'zh']: run(lang)
