"""Dump every piece of text a learner meets in the reward-model lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/reward-model_copy_deck.py  ->  out/reward-model/copy-en.md and out/reward-model/copy-zh.md

Order: home tile (every step of its slider, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the metrics), Show the math, the scenarios (each), Settings, the challenge (every feedback tier and the chart), the
card "Can you use it on a new case?" (every question, every option's verdict), and Connect with the drill answer. A tooltip is printed
the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'reward-model'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/reward-model.html', '--only', 'reward-model'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'reward-model.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''


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
    d.add('Stage', '\n'.join([txt(pg, '#rm-st-title'), txt(pg, '#rm-st-sub'), 'Status: ' + txt(pg, '#rm-status'), txt(pg, '#rm-pair'), txt(pg, '#rm-sig .rm-cap'), txt(pg, '#rm-w'), txt(pg, '#rm-tr-cap'), txt(pg, '#rm-tr-charts figcaption'), txt(pg, '#rm-bon .rm-cap'), txt(pg, '#rm-bon figcaption'), txt(pg, '#rm-bon .rm-bleg').replace('\n', ' | '), txt(pg, '.rm-scope')]))
    d.add('Metrics', '\n'.join(' | '.join(x for x in m.split('\n') if x.strip()) for m in pg.evaluate("()=>[...document.querySelectorAll('#rm-metrics .metric')].map(m=>m.innerText)")))
    d.tipsOf(pg, '#rm-stage')


def run(lang):
    d = Deck(); d.lines.append(f'# reward-model copy deck ({lang})\n')
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
        cl = lambda s: (pg.evaluate(f"()=>document.querySelector('{s}').click()"), pg.wait_for_timeout(250))
        end = lambda: (pg.evaluate("()=>{const s=document.querySelector('#rm-player input[data-key=\"scrub\"]'); if(s){s.value=s.max; s.dispatchEvent(new Event('input',{bubbles:true}));}}"), pg.wait_for_timeout(250))
        d.h('Home tile')
        for i in range(7):
            d.add(f'Tile, slider position {i + 1}', pg.evaluate(f"()=>window.__tzRm.read({i}).replace(/<[^>]+>/g,'')"))
        d.add('Tile notes', '\n'.join(pg.evaluate("()=>window.__tzRm.notes()")))
        pg.evaluate("()=>{location.hash='reward-model'}"); pg.wait_for_timeout(500)
        d.h('Header')
        d.add('Header', txt(pg, '#rm-app .labhead') or txt(pg, '#rm-app header'))
        for k in ('fid', 'primer'):
            pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"reward-model-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(200)
            d.add(f'Popover {k}', pg.evaluate(f"()=>{{const p=document.getElementById('reward-model-{k}'); return p?p.innerText:''}}"))
            pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"reward-model-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
        steps = [('mission', []), ('bt', [('pick 65% to 80%', '#rm-rail [data-pred="b"]')]), ('train', [('play to the end', None)]),
                 ('shift', [('show me: +2', '#rm-rail [data-nav="showme"]')]), ('weights', [('length unrelated to quality', '#rm-rail [data-lever="none"]')]),
                 ('hack', [('n = 64', 'n64')]), ('fix', [('50 pairs', '#rm-rail [data-lever="50"]'), ('200 pairs', '#rm-rail [data-lever="200"]')]), ('remember', [])]
        for i, (sid, acts) in enumerate(steps):
            if i: cl('#rm-rail [data-nav="next"]')
            d.h(f'Part {i + 1}: {sid}')
            d.add('Card', txt(pg, '#rm-rail .step.current')); d.tipsOf(pg, '#rm-rail'); stage(d, pg)
            for name, sel in acts:
                if sel == 'n64':
                    pg.evaluate("()=>{const s=document.querySelector('#rm-ic-n-in'); s.value=6; s.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)
                elif sel: cl(sel)
                end()
                d.add(f'After “{name}”: result box', txt(pg, '#rm-rail [data-obs]') or txt(pg, '#rm-rail .step.current')); d.tipsOf(pg, '#rm-rail'); stage(d, pg)
        d.h('Show the math')
        pg.evaluate("()=>{document.getElementById('rm-math').open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#rm-mathbody'))
        d.h('Scenarios')
        pg.evaluate("()=>document.querySelector('#rm-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#rm-rail'))
        for k in range(4):
            cl(f'#rm-rail [data-preset="{k}"]'); end(); d.add(f'Scenario {k + 1}', txt(pg, '#rm-insight')); stage(d, pg)
        d.h('Settings'); d.add('Settings', txt(pg, '#rm-controls')); d.tipsOf(pg, '#rm-controls')
        d.h('Challenge')
        pg.evaluate("()=>document.querySelector('#rm-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#rm-rail .railcard')); stage(d, pg)
        cl('#rm-rail [data-act="submit"]'); d.add('Not met', txt(pg, '#rm-ch-res'))
        cl('#rm-ch-res [data-act="reveal"]'); d.add('Show the answer', txt(pg, '#rm-ch-res'))
        cl('#rm-rail .seg-ctl[data-key="chB"] button[data-v="200"]'); cl('#rm-rail .seg-ctl[data-key="chS"] button[data-v="400"]'); d.add('Met with more pairs than needed (200 pairs, 400 steps)', txt(pg, '#rm-ch-res'))
        cl('#rm-rail .seg-ctl[data-key="chS"] button[data-v="50"]'); d.add('Met, but a smaller change exists', txt(pg, '#rm-ch-res'))
        cl('#rm-rail .seg-ctl[data-key="chB"] button[data-v="100"]'); cl('#rm-rail .seg-ctl[data-key="chS"] button[data-v="100"]'); d.add('Met, steps changed', txt(pg, '#rm-ch-res'))
        cl('#rm-rail .seg-ctl[data-key="chS"] button[data-v="200"]'); d.add('Met with the smallest change', txt(pg, '#rm-ch-res')); stage(d, pg); d.tipsOf(pg, '#rm-rail')
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
        pg.evaluate("()=>document.querySelector('#rm-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(150)
        d.add('Connect', txt(pg, '#rm-app .connect'))
        b.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(d.lines))
    print('wrote', OUT / f'copy-{lang}.md', sum(len(x) for x in d.lines), 'chars')


for lang in [x for x in ('en', 'zh') if x in sys.argv[1:]] or ['en', 'zh']: run(lang)
