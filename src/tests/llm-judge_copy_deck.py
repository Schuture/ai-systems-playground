"""Dump every piece of text a learner meets in the llm-judge lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/llm-judge_copy_deck.py  ->  out/llm-judge/copy-en.md and out/llm-judge/copy-zh.md

Order: home tile (both choices, the notes), lab header and both popovers, each guided part (card before and after its action, the stage,
the metrics, the mini result bar), Show the math, the scenarios (each), Settings, the challenge (every feedback tier and the chart), the
card "Can you use it on a new case?" (every question, every option's verdict), and Connect with the drill answer. A tooltip is printed the
first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'llm-judge'; OUT.mkdir(parents=True, exist_ok=True)
(P / 'out' / 'llm-judge').mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/llm-judge.html', '--only', 'llm-judge'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'llm-judge' / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'llm-judge.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'llm-judge' / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''


def txt(pg, sel):
    return pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).map(e=>e.innerText).join('\\n')", sel)


class Deck:
    def __init__(self): self.lines, self.tips = [], set()

    def h(self, t): self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = t.strip()
        if t: self.lines.append(f'**{label}**\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(t)
            if c and c not in self.tips: self.tips.add(c); new.append('- ' + c)
        if new: self.lines.append('Tooltips (new):\n' + '\n'.join(new) + '\n')


def stage(d, pg):
    d.add('Stage', '\n'.join([txt(pg, '#ljg-st-title'), txt(pg, '#ljg-st-sub'), 'Status: ' + txt(pg, '#ljg-status'), txt(pg, '#ljg-pair'), txt(pg, '#ljg-wf-cap'), 'Beside the marks: ' + txt(pg, '#ljg-wf-leg').replace('\n', ' | '), txt(pg, '#ljg-bars'), txt(pg, '.ljg-scope')]))
    d.add('Metrics', '\n'.join(' | '.join(x for x in m.split('\n') if x.strip()) for m in pg.evaluate("()=>[...document.querySelectorAll('#ljg-metrics .metric')].map(m=>m.innerText)")))
    d.tipsOf(pg, '#ljg-stage')


def run(lang):
    d = Deck(); d.lines.append(f'# llm-judge copy deck ({lang})\n')
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
        cl = lambda s: (pg.evaluate(f"()=>document.querySelector('{s}').click()"), pg.wait_for_timeout(250))
        # the home tile, rendered by the teaser itself in a scratch box
        d.h('Home tile')
        for m in ('once', 'both'):
            d.add(f'Tile, {m}', pg.evaluate(f"""()=>{{const el=document.createElement('div'); el.style.width='320px'; document.body.appendChild(el); const T=window.ASP;
              const def=(window.__tzLjg&&true); return def?window.__tzLjg.read('{m}').replace(/<[^>]+>/g,''):''}}"""))
        pg.evaluate("()=>{location.hash='llm-judge'}"); pg.wait_for_timeout(500)
        d.h('Header')
        d.add('Header', txt(pg, '#ljg-app .labhead') or txt(pg, '#ljg-app header'))
        for k in ('fid', 'primer'):
            try:
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"llm-judge-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(200)
                d.add(f'Popover {k}', pg.evaluate(f"()=>{{const p=document.getElementById('llm-judge-{k}'); return p?p.innerText:''}}"))
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"llm-judge-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
            except Exception as e: d.add(f'Popover {k}', f'(not opened: {e})')
        steps = [('mission', [('pick 60%', '#ljg-rail [data-pred="60"]')]), ('how', [('next pair', '#ljg-rail [data-act="pair-next"]')]),
                 ('order', [('A first', '#ljg-rail [data-lever="A"]'), ('random order', '#ljg-rail [data-lever="R"]'), ('both orders', '#ljg-rail [data-lever="both"]')]),
                 ('length', [('take length out', '#ljg-rail .seg-ctl[data-key="lc"] button[data-v="1"]')]),
                 ('check', [('large judge', '#ljg-rail [data-lever="big"]'), ('small, both orders', '#ljg-rail [data-lever="both"]')]), ('remember', [('show part 5', '#ljg-rail [data-rem="check"]')])]
        for i, (sid, acts) in enumerate(steps):
            if i: cl('#ljg-rail [data-nav="next"]')
            d.h(f'Part {i + 1}: {sid}')
            d.add('Card', txt(pg, '#ljg-rail .step.current')); d.tipsOf(pg, '#ljg-rail'); stage(d, pg)
            for name, sel in acts:
                cl(sel); d.add(f'After “{name}”: result box', txt(pg, '#ljg-rail [data-obs]') or txt(pg, '#ljg-rail .step.current')); d.tipsOf(pg, '#ljg-rail'); stage(d, pg)
        d.h('Show the math')
        pg.evaluate("()=>{document.getElementById('ljg-math').open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#ljg-mathbody'))
        d.h('Scenarios')
        pg.evaluate("()=>document.querySelector('#ljg-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#ljg-rail'))
        for k in range(4):
            cl(f'#ljg-rail [data-preset="{k}"]'); d.add(f'Scenario {k + 1}', txt(pg, '#ljg-insight')); stage(d, pg)
        d.h('Settings'); d.add('Settings', txt(pg, '#ljg-controls')); d.tipsOf(pg, '#ljg-controls')
        d.h('Challenge')
        pg.evaluate("()=>document.querySelector('#ljg-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#ljg-rail .railcard')); stage(d, pg)
        cl('#ljg-rail [data-act="submit"]'); d.add('Not met', txt(pg, '#ljg-ch-res'))
        cl('#ljg-ch-res [data-act="reveal"]'); d.add('Show the answer', txt(pg, '#ljg-ch-res'))
        for s in ('#ljg-rail .seg-ctl[data-key="chJ"] button[data-v="big"]', '#ljg-rail .seg-ctl[data-key="chO"] button[data-v="both"]', '#ljg-rail .seg-ctl[data-key="chL"] button[data-v="1"]'): cl(s)
        d.add('Met, but cheaper exists', txt(pg, '#ljg-ch-res'))
        cl('#ljg-rail .seg-ctl[data-key="chJ"] button[data-v="small"]'); d.add('Met, small judge in both orders (random order is cheaper)', txt(pg, '#ljg-ch-res'))
        cl('#ljg-rail .seg-ctl[data-key="chO"] button[data-v="R"]'); d.add('Met and cheapest', txt(pg, '#ljg-ch-res')); stage(d, pg); d.tipsOf(pg, '#ljg-rail')
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
                if t == 2: break
        d.h('Connect')
        pg.evaluate("()=>document.querySelector('#ljg-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(150)
        d.add('Connect', txt(pg, '#ljg-app .connect'))
        b.close()
    (OUT / f'copy-{lang}.md').write_text('\n'.join(d.lines))
    print('wrote', OUT / f'copy-{lang}.md', sum(len(x) for x in d.lines), 'chars')


for lang in [x for x in ('en', 'zh') if x in sys.argv[1:]] or ['en', 'zh']: run(lang)
