"""Dump every piece of text a learner meets in the eval-capstone lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/eval-capstone_copy_deck.py [en|zh]  ->  out/eval-capstone-copy/en.md and out/eval-capstone-copy/zh.md  (read by tests/numdense.py)

Order: home tile (both choices), lab header and both popovers, each guided part (card before and after each action, the stage, the
metrics), Show the math, the scenarios (each), Settings, the challenge (every feedback tier and the chart), the card “Can you use it
on a new case?” (every question, every option's verdict), and Connect with the drill answer. A tooltip is printed the first time it
appears with a given text, indented under “_New tooltips:_”."""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'eval-capstone-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/eval-capstone.html', '--only', 'eval-capstone'], cwd=P, check=True, capture_output=True)
(OUT / 'deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'eval-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(OUT / 'deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split()) if s else ''


def detag(t, lang):
    """A tooltip written as a list (a title, then name and value pairs) read as one line: “name: value; name: value”."""
    if '<' not in t: return t
    sep, end = ('：', '；') if lang == 'zh' else (': ', '; ')
    t = re.sub(r'</span>\s*<b>', sep, t)
    t = re.sub(r'</(?:div|p)>', end, t)
    t = re.sub(r'<[^>]+>', '', t)
    return re.sub(r'(?:[；;]\s*)+', end, t).strip().rstrip('；; ')


def txt(pg, sel):
    return pg.evaluate("s=>[...document.querySelectorAll(s)].filter(e=>e.getClientRects().length).map(e=>e.innerText).join('\\n')", sel)


class Deck:
    def __init__(self, lang='en'): self.lines, self.tips, self.lang = [], set(), lang

    def h(self, t): self.lines.append(f'\n## {t}\n')

    def add(self, label, t):
        t = (t or '').strip()
        if t: self.lines.append(f'**{label}**\n\n' + '\n'.join(clean(x) for x in t.split('\n') if clean(x)) + '\n')

    def tipsOf(self, pg, root):
        new = []
        for t in pg.evaluate("r=>[...document.querySelectorAll(r+' [data-tipx]')].filter(e=>e.getClientRects().length).map(e=>e.getAttribute('data-tipx'))", root):
            c = clean(detag(t, self.lang))
            if c and c not in self.tips: self.tips.add(c); new.append('  - ' + c)
        if new: self.lines.append('_New tooltips:_\n' + '\n'.join(new) + '\n')


def stage(d, pg):
    d.add('Stage', '\n'.join([txt(pg, '#ecap-st-title'), txt(pg, '#ecap-st-sub'), 'Status: ' + txt(pg, '#ecap-status'), txt(pg, '#ecap-leg').replace('\n', ' | '), txt(pg, '#ecap-fp')]))
    d.add('Metrics', '\n'.join(' | '.join(x for x in m.split('\n') if x.strip()) for m in pg.evaluate("()=>[...document.querySelectorAll('#ecap-metrics .metric')].map(m=>m.innerText)")))
    d.tipsOf(pg, '#ecap-stage')


def seg(key, v, root='#ecap-rail'):
    return f'{root} .seg-ctl[data-key="{key}"] button[data-v="{v}"]'


def run(lang):
    d = Deck(lang); d.lines.append(f'# eval-capstone copy deck ({lang})\n')
    with sync_playwright() as pw:
        b = pw.chromium.launch(); pg = b.new_page(viewport={'width': 1440, 'height': 900})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
        cl = lambda s: (pg.evaluate("s=>document.querySelector(s).click()", s), pg.wait_for_timeout(220))
        d.h('Home tile')
        for m in ('tot', 'cat'):
            d.add(f'Tile, {m}', pg.evaluate(f"()=>window.__tzEcap?window.__tzEcap.read('{m}').replace(/<[^>]+>/g,''):''"))
        d.add('Tile, as drawn', txt(pg, '.tz-ecap, [data-teaser="eval-capstone"]'))
        pg.evaluate("()=>{location.hash='eval-capstone'}"); pg.wait_for_timeout(500)
        d.h('Header')
        d.add('Header', txt(pg, '#ecap-app .labhead') or txt(pg, '#ecap-app header'))
        for k in ('fid', 'primer'):
            try:
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"eval-capstone-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(200)
                d.add(f'Popover {k}', pg.evaluate(f"()=>{{const p=document.getElementById('eval-capstone-{k}'); return p?p.innerText:''}}"))
                pg.evaluate(f"()=>{{const b=document.querySelector('[data-pop=\"eval-capstone-{k}\"]'); if(b) b.click();}}"); pg.wait_for_timeout(150)
            except Exception as e: d.add(f'Popover {k}', f'(not opened: {e})')
        steps = [('mission', [('pick “Not yet”', '#ecap-rail [data-pred="split"]')]),
                 ('split', [('each category', seg('split', '1'))]),
                 ('size', [('1,000 items', seg('gsize', '1000')), ('100 items', seg('gsize', '100')), ('3,000 items', seg('gsize', '3000'))]),
                 ('judge', [('judge once', '#ecap-rail [data-lever="once"]'), ('both orders', '#ecap-rail [data-lever="both"]')]),
                 ('calib', [('100 labels', seg('gcal', '100')), ('300 labels', seg('gcal', '300')), ('1,000 labels', seg('gcal', '1000'))]),
                 ('online', [('3 days', seg('abd', '3')), ('7 days', seg('abd', '7'))]),
                 ('remember', [])]
        for i, (sid, acts) in enumerate(steps):
            if i: cl('#ecap-rail [data-nav="next"]')
            d.h(f'Part {i + 1}: {sid}')
            d.add('Card', txt(pg, '#ecap-rail .step.current')); d.tipsOf(pg, '#ecap-rail'); stage(d, pg)
            for name, sel in acts:
                cl(sel); d.add(f'After “{name}”: result box', txt(pg, '#ecap-rail [data-obs]') or txt(pg, '#ecap-rail .step.current')); d.tipsOf(pg, '#ecap-rail'); stage(d, pg)
            if sid in ('split', 'size', 'judge', 'calib'):
                pg.evaluate("()=>{document.getElementById('ecap-math').open=true}"); pg.wait_for_timeout(150)
                d.h(f'Show the math (part {i + 1})'); d.add('Show the math', txt(pg, '#ecap-mathbody'))
                pg.evaluate("()=>{document.getElementById('ecap-math').open=false}")
        d.h('Show the math (online)')
        cl('#ecap-rail [data-go="5"]')
        pg.evaluate("()=>{document.getElementById('ecap-math').open=true}"); pg.wait_for_timeout(150); d.add('Show the math', txt(pg, '#ecap-mathbody'))
        d.h('Scenarios')
        pg.evaluate("()=>document.querySelector('#ecap-app .modes button[data-mode=\"sandbox\"]:not([data-flow=free])').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#ecap-rail'))
        for k in range(4):
            cl(f'#ecap-rail [data-preset="{k}"]'); d.add(f'Scenario {k + 1}', txt(pg, '#ecap-insight')); stage(d, pg)
        d.h('Settings'); d.add('Settings', txt(pg, '#ecap-controls')); d.tipsOf(pg, '#ecap-controls')
        d.h('Challenge')
        pg.evaluate("()=>document.querySelector('#ecap-app .modes button[data-mode=\"challenge\"]').click()"); pg.wait_for_timeout(300)
        d.add('Card', txt(pg, '#ecap-rail .railcard')); stage(d, pg)
        cl('#ecap-rail [data-act="submit"]'); d.add('Not met', txt(pg, '#ecap-ch-res'))
        cl('#ecap-ch-res [data-act="reveal"]'); d.add('Show the answer', txt(pg, '#ecap-ch-res'))
        for s in (seg('m-write', 'judge'), seg('n-write', '3000'), seg('ctl', '1'), seg('cal', '300'), seg('n-safe', '300')): cl(s)
        d.add('Met, but a cheaper plan exists', txt(pg, '#ecap-ch-res'))
        cl(seg('m-safe', 'human')); d.add('Met and cheapest', txt(pg, '#ecap-ch-res')); stage(d, pg); d.tipsOf(pg, '#ecap-rail')
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
        d.add('Connect', pg.evaluate("()=>{const c=document.querySelector('#ecap-app .connect').cloneNode(true); c.querySelectorAll('pre, #ecap-drill-q, #ecap-drill-a').forEach(e=>e.remove()); document.body.appendChild(c); const t=c.innerText; c.remove(); return t}"))
        d.lines.append('```\n' + pg.evaluate("()=>document.getElementById('ecap-code').textContent") + '\n```\n')
        d.h('Drill (an exercise with its numbers)')
        pg.evaluate("()=>document.querySelector('#ecap-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(150)
        d.add('Drill', txt(pg, '#ecap-drill-q') + '\n' + txt(pg, '#ecap-drill-a'))
        b.close()
    (OUT / f'{lang}.md').write_text('\n'.join(d.lines))
    print('wrote', OUT / f'{lang}.md', sum(len(x) for x in d.lines), 'chars')


for lang in [x for x in ('en', 'zh') if x in sys.argv[1:]] or ['en', 'zh']: run(lang)
