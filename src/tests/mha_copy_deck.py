"""Dump every piece of text a learner meets in the MHA lab and its home tile, in reading order, in English and Chinese.

python3 tests/mha_copy_deck.py  ->  out/mha-copy/en.md and out/mha-copy/zh.md

Order: home tile (sentence at 1 and 2 heads, the notes behind "Assumptions & sources"), lab header with the mode tooltips and
both popovers, each guided part (card before and after its action, the stage, the metrics, the tooltips in view that were not
shown before, the mini result bar), "Show the math" (1 head at “sat” and “it”, 2 heads), the scenarios card for every scenario, the settings
panel with the hand-set weights note, the challenge before and after submitting (every feedback tier and the chart), the capability
layer under it (every question's options, why and wrong-pick notes, from window.__mha.exits()), and the connect section with the
drill answer. A tooltip is printed the first time its text appears, so a repeated legend does not
drown the rest.
"""
import pathlib, re, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'mha-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/mha.html', '--only', 'mha'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-mha.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'mha.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-mha.html')


def txt(pg, sel):
    el = pg.query_selector(sel)
    return el.inner_text().strip() if el and el.is_visible() else ''


def stage(pg):
    """The stage in reading order: title, badge, legend, the grid as one line per head, captions, metrics."""
    parts = [txt(pg, '#mha-st-k'), txt(pg, '#mha-st-title'), 'Badge: ' + txt(pg, '#mha-status'), 'Legend: ' + ' | '.join(pg.evaluate("()=>[...document.querySelectorAll('#mha-stage .legend-row > span')].filter(e=>e.offsetParent).map(e=>e.innerText.trim())"))]
    toks = pg.evaluate("()=>[...document.querySelectorAll('#mha-grid .mha-tok')].map(t=>(t.classList.contains('q')?'['+t.innerText+']':t.innerText)+(t.classList.contains('need')?'*':''))")
    parts.append('Tokens: ' + ' '.join(toks))
    for row in pg.query_selector_all('#mha-grid .mha-row'):
        lab = row.query_selector('.mha-rlab').inner_text().strip()
        vals = [c.inner_text().strip() or '·' for c in row.query_selector_all('.mha-cell')]
        parts.append(f'  {lab}: ' + ' '.join(vals))
    parts += [txt(pg, '#mha-cap'), txt(pg, '#mha-mats .mha-mleg'), txt(pg, '#mha-mats .mha-cap')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#mha-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {m}' for m in mets])


class Tips:
    """Hover every visible element that has a tooltip and record the text the first time it appears."""
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        els = pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn], {root} [data-jump]')
        for e in els:
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500)
                bb = e.evaluate("e=>{const r=e.getClientRects()[0]||e.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height}}")   # the first line box: a term may wrap
                if not bb: continue
                pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {label} ⟶ {t}')
            pg.mouse.move(1, 1)
        return out


def dock(pg):
    """Scroll the stage out of view at 1440 x 1000 so that the mini result bar shows; read it; scroll back."""
    pg.evaluate("()=>{document.querySelector('.connect').scrollIntoView({block:'start'}); window.dispatchEvent(new Event('scroll'))}"); pg.wait_for_timeout(400)
    t = ' '.join(pg.inner_text('#dock').split()) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.evaluate("()=>{window.scrollTo(0,0); window.dispatchEvent(new Event('scroll'))}"); pg.wait_for_timeout(250)
    return 'Mini result bar: ' + t


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#mha-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(300)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000}); pg.set_default_timeout(8000)
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); L = []
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        pg.evaluate("()=>document.querySelector('#tile-mha').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        L += ['# Home tile', txt(pg, '#tile-mha')]
        for v in ('1', '2'):
            pg.click(f'#tile-mha .seg-ctl button[data-v="{v}"]'); pg.wait_for_timeout(120)
            L += [f'## Tile sentence with {v} head(s)', txt(pg, '#tile-mha .tz-read'), txt(pg, '#tile-mha .mha-tz')]
        L += ['## Assumptions & sources', pg.evaluate("()=>[...document.querySelectorAll('#tile-mha .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')")]
        L += ['## Tile tooltips'] + T.grab(pg, '#tile-mha')
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='mha'}"); pg.wait_for_timeout(700)
        L += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        L += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            try:
                pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
                L += [f'## Popover: {" ".join(sel.inner_text().split())}', pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid)]
                sel.click(); pg.wait_for_timeout(150)
            except Exception as e:
                L += [f'(popover failed: {e})']
        # ---------- guided parts ----------
        pg.click('.dyn-root .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
        for k, sid in enumerate(['mission', 'predict', 'look', 'one', 'two', 'four', 'real', 'remember']):
            go(pg, k)
            L += [f'# Guided part {k + 1}', '## Card', txt(pg, '#mha-rail .step.current'), '## Stage', stage(pg)]
            L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench')
            acted = True
            if sid == 'predict':
                pg.click('#mha-rail [data-pred="one"]')
            elif sid == 'look':
                pg.click('#mha-ic-q button[data-q="3"]')
            elif sid == 'one':
                pg.click('#mha-ic-q button[data-q="5"]'); pg.wait_for_timeout(300)   # another word before “it” gets its own result, no verdict (N8, M3)
                L += ['## Card after picking “the” / “上” first', txt(pg, '#mha-rail .step.current')]
                pg.click('#mha-ic-q button[data-q="8"]')
            elif sid == 'two':
                pg.click('#mha-ctl-H button[data-v="4"]'); pg.wait_for_timeout(300)   # 4 heads (in Settings: the card offers 1 and 2) before 2: that state, and a pointer to 2 heads
                L += ['## Card after picking “4 heads” in Settings first', txt(pg, '#mha-rail .step.current')]
                pg.click('#mha-ic-H button[data-v="2"]')
            elif sid == 'four':
                pg.click('#mha-ic-H button[data-v="4"]')
            else:
                acted = False
            if acted:
                pg.wait_for_timeout(350)
                L += ['## Card after the action', txt(pg, '#mha-rail .step.current'), '## Stage after the action', stage(pg)]
                L += ['## Tooltips in view after the action (new)'] + T.grab(pg, '.dyn-root .bench')
            if sid == 'look':
                for q in ('0', '1', '7'):
                    pg.click(f'#mha-ic-q button[data-q="{q}"]'); pg.wait_for_timeout(250)
                    L += [f'## Observation with word {q}', txt(pg, '#mha-rail [data-obs]')]
                pg.click('#mha-ic-q button[data-q="8"]', force=True); pg.wait_for_timeout(250)
                L += ['## A click on “it” in part 3 (kept for part 4)', 'Tip: ' + (' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else '(no tip)')]
                pg.mouse.move(1, 1)
                pg.click('#mha-ic-q button[data-q="3"]'); pg.wait_for_timeout(250)
                pg.evaluate("()=>{const d=document.getElementById('mha-math'); d.open=true; d.querySelector('.mha-mxs').open=true}"); pg.wait_for_timeout(200)
                L += ['## Show the math (“sat” / “在”, 1 head)', txt(pg, '#mha-mathbody')]
                L += ['## Show the math: tooltips (new)'] + T.grab(pg, '#mha-mathbody')
                pg.evaluate("()=>{document.getElementById('mha-math').open=false}")
            if sid == 'one':
                pg.evaluate("()=>{document.getElementById('mha-math').open=true}"); pg.wait_for_timeout(200)
                L += ['## Show the math (“it”, 1 head)', txt(pg, '#mha-mathbody')]
                pg.evaluate("()=>{document.getElementById('mha-math').open=false}")
            if sid == 'two':
                pg.evaluate("()=>{document.getElementById('mha-math').open=true}"); pg.wait_for_timeout(200)
                L += ['## Show the math (“it”, 2 heads, head A)', txt(pg, '#mha-mathbody')]
                pg.evaluate("()=>{document.getElementById('mha-math').open=false}")
            L += [dock(pg)]
        # ---------- part 4 with the other answers: the wrong prediction, and a skipped one (N9) ----------
        for how in ('both', 'skip'):
            q = b.new_page(viewport={'width': 1440, 'height': 1000}); q.set_default_timeout(8000)
            q.goto(URL); q.wait_for_timeout(300); q.click(f'.langsw button[data-lang="{lang}"]'); q.wait_for_timeout(150)
            q.evaluate("()=>{location.hash='mha'}"); q.wait_for_timeout(500); go(q, 1)
            q.click('#mha-rail [data-nav="skip"]' if how == 'skip' else '#mha-rail [data-pred="both"]'); q.wait_for_timeout(200); go(q, 3)
            q.click('#mha-ic-q button[data-q="8"]'); q.wait_for_timeout(300)
            L += [f'# Part 4 after {"skipping the prediction" if how == "skip" else "predicting “Both can get 80% or more”"}', txt(q, '#mha-rail [data-obs]')]
            q.close()
        # ---------- scenarios ----------
        pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        L += ['# Scenarios', txt(pg, '#mha-rail')]
        n = len(pg.query_selector_all('#mha-rail [data-preset]'))
        for i in range(n):
            pg.click(f'#mha-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = ' '.join(pg.inner_text(f'#mha-rail [data-preset="{i}"]').split())
            L += [f'## Scenario {i + 1}: {name}', txt(pg, '#mha-insight'), '## Stage', stage(pg)]
            L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root #mha-stage') + [dock(pg)]
        pg.click('#mha-ctl-q button[data-q="5"]'); pg.wait_for_timeout(250)
        L += ['## After picking word 5 in the settings (no scenario matches)', txt(pg, '#mha-insight')]
        pg.click('#mha-ctl-q button[data-q="8"]'); pg.wait_for_timeout(200); pg.click('#mha-ctl-H button[data-v="4"]'); pg.wait_for_timeout(200)
        pg.click('#mha-ctl-on button[data-on="2"]'); pg.wait_for_timeout(250)
        L += ['## After turning H3 off in the settings', txt(pg, '#mha-insight'), stage(pg)]
        # ---------- settings panel ----------
        pg.evaluate("()=>{const d=document.getElementById('mha-adv'); if(d) d.open=true}"); pg.wait_for_timeout(150)
        L += ['# Settings panel', txt(pg, '#mha-controls')]
        L += ['## Settings tooltips (new)'] + T.grab(pg, '#mha-controls')
        d42 = pg.query_selector('#mha-adv .def')   # R6.18: the real block's size, rounded in the text, exact on hover
        if d42:
            d42.scroll_into_view_if_needed(); d42.hover(); pg.wait_for_timeout(250)
            L += ['## Hover on the real block’s size', ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else '(no tip)']
            pg.mouse.move(1, 1)
        # ---------- challenge ----------
        pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        L += ['# Challenge', '## Card before submitting', txt(pg, '#mha-rail'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#mha-controls')]
        L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench') + [dock(pg)]
        pg.click('#mha-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        L += ['## Submitted with all 4 heads on', txt(pg, '#mha-ch-res'), 'Badge: ' + txt(pg, '#mha-status'), dock(pg)]
        pg.click('#mha-ic-on button[data-on="2"]'); pg.wait_for_timeout(300)
        L += ['## H3 off', txt(pg, '#mha-ch-res'), 'Badge: ' + txt(pg, '#mha-status'), dock(pg)]
        rv = pg.query_selector('#mha-ch-res [data-act="reveal"]')
        if rv: rv.click(); pg.wait_for_timeout(300)
        L += ['## After “show all options”', txt(pg, '#mha-ch-res')]
        pg.click('#mha-ic-on button[data-on="1"]'); pg.wait_for_timeout(200); pg.click('#mha-ic-on button[data-on="2"]'); pg.wait_for_timeout(200); pg.click('#mha-ic-on button[data-on="3"]'); pg.wait_for_timeout(300)
        L += ['## H2 and H4 off (only H1 and H3 on)', txt(pg, '#mha-ch-res'), 'Badge: ' + txt(pg, '#mha-status'), dock(pg)]
        L += ['## Tooltips in view (new)'] + T.grab(pg, '#mha-rail')
        # ---------- capability layer: each question's options, why and the note on each wrong pick (read-only hook; numdense scans them) ----------
        ex = pg.evaluate("()=>window.__mha.exits()")
        strip = lambda s: re.sub(r'<[^>]+>', '', s).replace('&lt;', '<').replace('&gt;', '>')
        for key, head in (('explain', 'Capability layer: Explain it'), ('transfer', 'Capability layer: A new case')):
            L += [f'# {head}']
            for i, q in enumerate(ex[key]):
                L += [f'{i + 1}. {strip(q["q"])}', '\n'.join(f'- {"(right) " if j == q["a"] else ""}{strip(o)}' for j, o in enumerate(q['o'])), f'Why: {strip(q["why"])}']
                L += [f'Not quite (option {k + 1}): {strip(n)}' for k, n in enumerate(q.get('n') or []) if n]
        # ---------- connect ----------
        pg.click('.dyn-root [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        L += ['# Connect', txt(pg, '.dyn-root .connect')]
        L += ['## Connect tooltips (new)'] + T.grab(pg, '.dyn-root .connect')
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in L if x))
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
