"""Dump every piece of text a learner meets in the kv-basics lab and its home tile, in reading order, in English and Chinese.

python3 tests/kv-basics_copy_deck.py  ->  out/kv-basics-copy/en.md and out/kv-basics-copy/zh.md

Order: home tile (sentence at several token counts, legend, the notes behind "Assumptions & sources", chip tooltips), lab header
with the mode tooltips and both popovers, each guided part (card before and after its action, the stage, the tooltips in view that
were not shown before, the mini result bar), Show the math in both views, the scenarios card for every scenario, the settings panel,
the challenge before and after submitting (every feedback tier and the chart), and the connect section with the drill answer.
A tooltip is printed the first time it appears with a given text, so a repeated legend does not drown the rest.
"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'kv-basics-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/kv-basics.html', '--only', 'kv-basics'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-kv-basics.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'kv-basics.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-kv-basics.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').strip()

def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''

def stage(pg):
    """The stage in reading order: head, legend, the chart's words (not every grid cell), metrics."""
    parts = [txt(pg, '#kvb-st-title'), txt(pg, '#kvb-st-sub'), 'Status: ' + txt(pg, '#kvb-status'), 'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#kvb-legend > span')].map(s=>s.innerText)"))]
    if pg.is_visible('#kvb-grid'):
        parts.append('Grid: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-lcap')].map(e=>e.innerText).join(' / ')")))
        parts.append('Tokens: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-tok b')].map(e=>e.textContent).join(' ')")))
        rows = pg.evaluate("()=>[...document.querySelectorAll('#kvb-grid .kvb-chk')].filter(r=>!r.hidden).map(r=>[...r.children].map(c=>c.textContent).join(' '))")
        if rows: parts.append('Check rows: ' + ' / '.join(clean(r) for r in rows))
        parts.append(txt(pg, '#kvb-toy .kvb-note'))
    if pg.is_visible('#kvb-sp'):
        parts += [txt(pg, '#kvb-sp-title'), clean(pg.evaluate("()=>[...document.querySelectorAll('#kvb-sp .kvb-sph')].map(e=>e.innerText.replace(/\\n/g,': ')).join(' / ')")), txt(pg, '#kvb-mem-title'), txt(pg, '#kvb-real-note')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#kvb-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {clean(m)}' for m in mets])

class Tips:
    """Hover every visible element that has a tooltip and record the text the first time it appears."""
    def __init__(self): self.seen = set()
    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn], {root} [data-jump]'):
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(40)
                # the first line box of the element: a link that wraps has a bounding box whose corner can sit on other text
                bb = e.evaluate("el=>{const r=el.getClientRects()[0]; return r?{x:r.x,y:r.y,width:r.width,height:r.height}:null}")
                if not bb: continue
                pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {clean(label)} ⟶ {clean(t)}')
            pg.mouse.move(1, 1)
        return out

def dock(pg):
    """Scroll the stage's chart out of view (on a phone-sized window) and read the mini result bar."""
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const r=document.querySelector('#kvb-rail .step.current, #kvb-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('.bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())

def calm(pg):
    """wait for the lab's correction that keeps the sticky stage below the top bar, then for the scroll to stop"""
    try: pg.wait_for_function("()=>!window.__kvb.state().stageBusy", timeout=4000)
    except Exception: pass
    last = None
    for _ in range(25):
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120)

def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(200)

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        pg.evaluate("()=>document.querySelector('#tile-kv-basics').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        Lines += ['# Home tile', txt(pg, '#tile-kv-basics')]
        for v in (1, 9, 64):
            rng(pg, '#tile-kv-basics input[type=range]', v)
            Lines += [f'## Tile sentence at {v} tokens', txt(pg, '#tile-kv-basics .tz-read')]
        Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-kv-basics .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-kv-basics')
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='kv-basics'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        Lines += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        # ---------- guided parts ----------
        ids = pg.evaluate("()=>window.__kvb.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            pg.evaluate(f"()=>document.querySelector('#kvb-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(350); calm(pg)
            Lines += [f'# Guided part {k + 1}', '## Card', txt(pg, '#kvb-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench')
            acted = True
            if sid == 'predict': pg.click('#kvb-rail [data-pred="200"]')
            elif sid == 'steps':
                for _ in range(4): pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(80)
            elif sid == 'check': pg.click('#kvb-ic-mask button[data-v="0"]')
            elif sid == 'keep': pg.click('#kvb-ic-cache button[data-v="1"]')
            elif sid == 'real': rng(pg, '#kvb-ic-ctx-in', 11)
            else: acted = False
            if acted:
                pg.wait_for_timeout(350); calm(pg)
                Lines += ['## Card after the action', txt(pg, '#kvb-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '.dyn-root .bench')
            if sid == 'steps':
                for _ in range(5): pg.click('#kvb-rail [data-act="gen"]'); pg.wait_for_timeout(60)
                Lines += ['## Result at 9 steps', txt(pg, '#kvb-rail [data-obs]')]
            if sid == 'real':
                rng(pg, '#kvb-ic-ctx-in', 0)
                Lines += ['## Result at 256 tokens', txt(pg, '#kvb-rail [data-obs]')]
                rng(pg, '#kvb-ic-ctx-in', 7)
            Lines += [dock(pg)]
        # ---------- Show the math ----------
        for tok, name in [('t_9_1_0_8192', 'small model, 9 steps, cache on, mask off'), ('r_9_1_1_8192', 'Llama-3.1-8B at 8K')]:
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
            pg.evaluate(f"()=>{{location.hash='kv-basics~{tok}'}}"); pg.wait_for_timeout(500)
            pg.evaluate("()=>{document.getElementById('kvb-math').open=true}"); pg.wait_for_timeout(150)
            Lines += [f'# Show the math ({name})', txt(pg, '#kvb-mathbody'), '## Math term tooltips (new)'] + T.grab(pg, '#kvb-mathbody')
        # ---------- scenarios ----------
        pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#kvb-rail')]
        for i in range(4):
            pg.click(f'#kvb-rail [data-preset="{i}"]'); pg.wait_for_timeout(250)
            name = clean(' '.join(pg.inner_text(f'#kvb-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#kvb-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#kvb-insight') + T.grab(pg, '.dyn-root #kvb-stage') + [dock(pg)]
        rng(pg, '#kvb-ctl-ctx-in', 7)
        Lines += ['## After changing the context under Settings (no scenario matches)', txt(pg, '#kvb-insight')]
        # ---------- settings panel ----------
        Lines += ['# Settings panel', txt(pg, '#kvb-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#kvb-controls')
        # ---------- challenge ----------
        pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#kvb-rail'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#kvb-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench') + [dock(pg)]
        pg.click('#kvb-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## After submitting at 4K (missed)', txt(pg, '#kvb-ch-res'), 'Status: ' + txt(pg, '#kvb-status')]
        rng(pg, '#kvb-ic-chctx-in', 2)
        Lines += ['## 1,024 tokens (met, a longer context works too)', txt(pg, '#kvb-ch-res')]
        rng(pg, '#kvb-ic-chctx-in', 3)
        Lines += ['## 1,536 tokens (met, the longest)', txt(pg, '#kvb-ch-res'), 'Status: ' + txt(pg, '#kvb-status'), dock(pg)]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#kvb-rail')
        # ---------- connect ----------
        pg.click('.dyn-root [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '.dyn-root .connect'), '## Connect tooltips (new)'] + T.grab(pg, '.dyn-root .connect')
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
