"""Dump every piece of text a learner meets in the tokenization-cost lab, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/tokenization-cost_copy_deck.py  ->  out/tokenization-cost-copy/en.md and out/tokenization-cost-copy/zh.md

Order: the lab header with both popovers; each guided part (the card before and after its action, the stage, the tooltips in view
that were not shown before); the prediction's reveal for a wrong pick (EN) and the right pick (ZH); Show the math; the scenarios card
for every scenario; the settings panel; the challenge before and after submitting (every feedback tier and the chart); the card
"Can you use it on a new case?" (every question, a verdict for every option); the connect section with the drill answer; the mini
result bar at 390 px. A tooltip is printed the first time it appears with a given text.
Round 1: one line per paragraph (so tests/numdense.py counts numbers per paragraph, R6.20); the challenge's comparison chart in a
section of its own, apart from the feedback before it and the explanation under it; built with the next lab (mha), so “Next lab”
reads as on the full site; part 7's “Show in the chart” buttons."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'tokenization-cost-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tokenization-cost-deck.html', '--only', 'tokenization-cost,mha'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'tokenization-cost-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tokenization-cost-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tokenization-cost-deck-wrapped.html')
clean = lambda s: ' '.join(s.replace('‑', '-').replace(' ', ' ').split())
paras = lambda s: '\n'.join(x for x in (clean(l) for l in (s or '').split('\n')) if x)


def txt(pg, sel):
    el = pg.query_selector(sel)
    return paras(el.inner_text()) if el and el.is_visible() else ''


def chres(pg):
    """the challenge's result: the feedback before the comparison chart, the chart, and the explanation under it"""
    parts = pg.evaluate("()=>[...document.querySelectorAll('#tk-ch-res > *')].filter(e=>e.getClientRects().length).map(e=>[e.classList.contains('cmp'), e.innerText])")
    i = next((j for j, (c, _) in enumerate(parts) if c), len(parts))
    return (paras('\n'.join(t for _, t in parts[:i])), paras(parts[i][1]) if i < len(parts) else '', paras('\n'.join(t for _, t in parts[i + 1:])))


def stage(pg):
    parts = [txt(pg, '#tk-st-title'), txt(pg, '#tk-st-sub'), 'Status: ' + txt(pg, '#tk-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tk-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             'Strips: ' + txt(pg, '#tk-strips-cap') + ' :: ' + ' / '.join(txt(pg, f'.tk-strip[data-l="{l}"] .tk-sh') for l in ('en', 'zh')),
             'Hex caption: ' + txt(pg, '#tk-hexcap'), 'Bytes or characters: ' + txt(pg, '#tk-vs'),
             'Merges: ' + txt(pg, '#tk-mg-cap') + ' :: ' + txt(pg, '#tk-mg-sum'), 'First merges: ' + txt(pg, '#tk-mg-first'), 'Hex note: ' + txt(pg, '#tk-mg-hexnote'),
             'Merges list: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tk-mg-list li')].map(l=>l.innerText)")),
             'Cost: ' + txt(pg, '#tk-cost-cap') + ' :: ' + ' / '.join(txt(pg, f'.tk-crow[data-l="{l}"]') for l in ('en', 'zh')) + ' :: ' + txt(pg, '#tk-ctx')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#tk-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p and not p.endswith(': ')] + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx], {root} .t, {root} [data-jump]'):
            try:
                if not e.is_visible(): continue
                t = clean(e.get_attribute('data-tipx') or '')
                if not t:
                    bb = e.bounding_box()
                    if not bb: continue
                    e.scroll_into_view_if_needed(timeout=1500); bb = e.bounding_box()
                    pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(110)
                    t = clean(pg.inner_text('#tip')) if pg.is_visible('#tip') else ''
                    pg.mouse.move(1, 1)
                label = clean(e.inner_text() or '')[:50]
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {label} ⟶ {t}')
        return out


def calm(pg):
    last = None
    for _ in range(25):
        y = pg.evaluate('()=>scrollY')
        if y == last: break
        last = y; pg.wait_for_timeout(120)
    pg.wait_for_timeout(350)


def clk(pg, sel): pg.evaluate("s=>document.querySelector(s).click()", sel); pg.wait_for_timeout(250); calm(pg)


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(250); calm(pg)


def exits(pg):
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in (0, 1):
        pg.evaluate(f"()=>{{const c={card}; c.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()}}"); pg.wait_for_timeout(150)
        nq = pg.evaluate(f"()=>{{const c={card}; return c.querySelectorAll('.ex-qs .pb-dot').length||1}}")
        for j in range(nq):
            pg.evaluate(f"()=>{{const c={card}; const d=c.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]'); if(d) d.click()}}"); pg.wait_for_timeout(150)
            q = pg.evaluate(f"()=>{{const c={card}; return c.querySelector('.ex-body p b').innerText}}")
            opts = pg.evaluate(f"()=>{{const c={card}; return [...c.querySelectorAll('.ex-opts .choice')].map(b=>[b.dataset.v,b.innerText])}}")
            out.append(f'\n**{["Explain it", "A new case"][t]} {j + 1}.** {clean(q)}')
            for v, o in opts:
                pg.evaluate(f"()=>{{const c={card}; c.querySelector('.ex-opts .choice[data-v=\"{v}\"]').click()}}"); pg.wait_for_timeout(150)
                res = pg.evaluate(f"()=>{{const c={card}; const r=c.querySelector('.ex-res'); return r?r.innerText:''}}")
                out.append(f'- ({v}) {clean(o)}\n  ⟶ {clean(res)}')
    pg.evaluate(f"()=>{{const c={card}; c.querySelector('[data-ex=\"tab\"][data-i=\"2\"]').click()}}"); pg.wait_for_timeout(150)
    out.append('\n**Go further.** ' + clean(pg.evaluate(f"()=>{{const c={card}; return c.querySelector('.ex-body').innerText}}")))
    return out


def deck(lang):
    L = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300)
        pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(100)
        pg.evaluate("()=>{location.hash='tokenization-cost'}"); pg.wait_for_timeout(600)
        tips = Tips()
        L += ['# Header', txt(pg, '#tk-app .labhead .crumbs'), txt(pg, '#tk-app .labtitle'), txt(pg, '#tk-app .lab-outcome'), txt(pg, '#tk-app .labmeta'), txt(pg, '#tk-app [data-mode-desc]')]
        L += ['Mode buttons: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tk-app .modes button')].map(b=>b.innerText+' ⟶ '+b.getAttribute('data-tipx'))"))]
        for pid in pg.evaluate("()=>[...document.querySelectorAll('#tk-app .labhead [data-pop]')].map(e=>e.getAttribute('data-pop'))"):
            pg.evaluate(f"()=>document.querySelector('#tk-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(150)
            L += [f'\n## Popover {pid}', clean(pg.inner_text(f'#{pid}'))] + tips.grab(pg, f'#{pid}')
            pg.evaluate(f"()=>document.querySelector('#tk-app .labhead [data-pop=\"{pid}\"]').click()"); pg.wait_for_timeout(100)
        steps = pg.evaluate("()=>window.__tk.consts.STEP_IDS")
        for i, sid in enumerate(steps):
            if i: clk(pg, '#tk-rail .step.current [data-nav="next"]')
            L += [f'\n# Part {i + 1} ({sid})', 'Rail list: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#tk-rail .step-h')].map(h=>h.innerText)")),
                  'Lead (part bar): ' + clean(pg.evaluate("()=>{const e=document.querySelector('#tk-rail .pbar'); return e?e.textContent:''}")),
                  '## Card', txt(pg, '#tk-rail .step.current .step-b'), '## Stage', stage(pg), '## Tooltips'] + tips.grab(pg, '#tk-app .bench')
            act = {'predict': lambda: clk(pg, f'#tk-rail [data-pred="{"9" if lang == "en" else "27"}"]'), 'bytes': lambda: clk(pg, '#tk-rail .seg-ctl[data-key="hex"] button[data-v="1"]'),
                   'merges': lambda: (rng(pg, '#tk-ic-k-in', 30), L.append('## After 30 merges\n' + txt(pg, '#tk-rail [data-obs]')), rng(pg, '#tk-ic-k-in', 100)),
                   'mix': lambda: [(clk(pg, f'#tk-rail [data-lever="{m}"]'), L.append(f'## After {m}\n' + txt(pg, '#tk-rail [data-obs]'))) for m in ('en', 'zh', 'bal')],
                   'cost': lambda: (clk(pg, '#tk-rail .seg-ctl[data-key="mix"] button[data-v="bal"]'), L.append('## After half and half\n' + txt(pg, '#tk-rail [data-obs]') + '\n' + txt(pg, '#tk-ctx')), clk(pg, '#tk-rail .seg-ctl[data-key="mix"] button[data-v="tile"]'))}.get(sid)
            if act:
                act(); L += ['## Card after the action', txt(pg, '#tk-rail .step.current .step-b'), '## Stage after the action', stage(pg), '## New tooltips'] + tips.grab(pg, '#tk-app .bench')
        rem = []
        for k in ('bytes', 'mix', 'cost'):
            clk(pg, f'#tk-rail [data-rem="{k}"]'); rem.append(f'- after “{k}”: ' + txt(pg, '#tk-st-sub') + ' / button reads: ' + clean(pg.evaluate(f"()=>document.querySelector('#tk-rail [data-rem={k}]').innerText")))
        L += ['## Part 7: the “Show in the chart” buttons, one at a time'] + rem
        pg.evaluate("()=>{document.getElementById('tk-math').open=true}"); pg.wait_for_timeout(150)
        L += ['\n# Show the math (part 7 state)', clean(pg.inner_text('#tk-mathbody'))] + tips.grab(pg, '#tk-mathbody')
        clk(pg, '#tk-rail [data-mode-go="sandbox"]')
        L += ['\n# Scenarios', txt(pg, '#tk-rail .railcard')]
        for i in range(4):
            clk(pg, f'#tk-rail [data-preset="{i}"]'); L += [f'## Scenario {i + 1}', txt(pg, '#tk-insight')]
        L += ['## Stage', stage(pg), '\n# Settings', txt(pg, '#tk-controls')] + tips.grab(pg, '#tk-controls')
        rng(pg, '#tk-ctl-k-in', 250); L += ['## After a change in Settings', txt(pg, '#tk-insight')]
        clk(pg, '#tk-rail [data-mode-go="challenge"]')
        L += ['\n# Challenge (before submitting)', txt(pg, '#tk-rail .railcard'), '## Stage', stage(pg), '## Settings', txt(pg, '#tk-controls')] + tips.grab(pg, '#tk-app .bench')
        clk(pg, '#tk-rail [data-act="submit"]'); L += ['## Submitted: mostly English, 100 merges', chres(pg)[0]]
        clk(pg, '#tk-rail .seg-ctl[data-key="chmix"] button[data-v="zh"]'); L += ['## Mostly Chinese, 100 merges', chres(pg)[0]]
        clk(pg, '#tk-ch-res [data-act="reveal"]'); a, c, w = chres(pg)
        L += ['## All options: the feedback', a, '## All options: the comparison chart', c, '## All options: the explanation under the comparison', w]
        clk(pg, '#tk-rail .seg-ctl[data-key="chmix"] button[data-v="bal"]'); clk(pg, '#tk-rail .seg-ctl[data-key="chk"] button[data-v="400"]'); L += ['## Half and half, 400 merges: the feedback', chres(pg)[0]]
        clk(pg, '#tk-rail .seg-ctl[data-key="chk"] button[data-v="100"]'); L += ['## Half and half, 100 merges: the feedback', chres(pg)[0]]
        L += ['\n# Can you use it on a new case?'] + exits(pg)
        L += ['\n# Next lab', txt(pg, '#tk-rail .next-lab')]
        pg.evaluate("()=>document.querySelector('#tk-app [data-act=\"drill-reveal\"]').click()"); pg.wait_for_timeout(150)
        con, code = txt(pg, '#tk-app .connect'), pg.evaluate("()=>{const p=document.querySelector('#tk-app .connect pre'); return p?p.innerText:''}")
        if code and paras(code) in con: con = con.replace(paras(code), '```\n' + code.rstrip() + '\n```')   # fenced, so its # comments are not headings
        L += ['\n# Connect', con] + tips.grab(pg, '#tk-app .connect')
        pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(300)
        clk(pg, '#tk-rail .seg-ctl[data-key="chk"] button[data-v="100"]')   # a render after the resize fills the bar
        pg.evaluate("()=>{const r=document.querySelector('#tk-rail .railcard'); r.scrollIntoView({block:'start'}); window.scrollBy(0,400);}"); pg.wait_for_timeout(500)
        L += ['\n# Mini result bar (390 px)', clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(not shown)']
        b.close()
    (OUT / f'{lang}.md').write_text('\n'.join(x for x in L if x is not None))
    print('wrote', OUT / f'{lang}.md', len(L), 'blocks')


for lg in ('en', 'zh'): deck(lg)
