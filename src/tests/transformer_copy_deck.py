"""Dump every piece of text a learner meets in the Transformer ① lab and its home tile, in reading order, in English and Chinese.

python3 tests/transformer_copy_deck.py  ->  out/transformer-copy/en.md and out/transformer-copy/zh.md

Order: home tile (for each model the picture's caption, the name a screen reader gives the picture, the sentence, the legend with its
tooltips and the picture's tooltips; the notes behind "Assumptions & sources"), lab header with the mode tooltips and
both popovers, each guided part (card before and after its action, the stage with legend, labels, share bar and metrics, the tooltips in
view that were not shown before, the mini result bar), every stop of part 3, "Show the math", the scenarios card for every scenario, the
settings panel with its notes, the challenge before and after submitting (every tier and the chart), and the connect section with the
drill answer. A tooltip is printed the first time it appears with a given text, so a repeated legend does not drown the rest.
"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'transformer-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/transformer.html', '--only', 'transformer'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-transformer.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'transformer.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-transformer.html')
plain = lambda s: s.replace('‑', '-').replace(' ', ' ')


def txt(pg, sel):
    el = pg.query_selector(sel)
    return plain(el.inner_text().strip()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#tf-st-title'), txt(pg, '#tf-st-sub'), txt(pg, '#tf-status'), txt(pg, '#tf-stage .tf-cap'), 'Legend: ' + ' | '.join(pg.evaluate("()=>[...document.querySelectorAll('#tf-legend > span')].map(s=>s.innerText.trim())")),
             txt(pg, '#tf-now'), txt(pg, '#tf-stack'), txt(pg, '#tf-share')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#tf-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {plain(m)}' for m in mets])


class Tips:
    """Hover every visible element with a tooltip and record the text the first time it appears."""
    def __init__(self):
        self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn], {root} [data-jump]'):
            try:
                if not e.is_visible():
                    continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); bb = e.bounding_box()
                if not bb:
                    continue
                pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {plain(label)} ⟶ {plain(t)}')
            pg.mouse.move(1, 1)
        return out


def dock(pg):
    """the mini result bar only appears once the stack scrolls out of view (on a phone); for the deck, render it in place through the kit and read it"""
    t = pg.evaluate("""()=>{const d=document.getElementById('dock'), h=d.hidden; d.hidden=false; window.ASP.dock.refresh();
      const t=d.innerText.replace(/\\s+/g,' ').trim(); d.hidden=h; window.ASP.dock.update(); return t}""")
    return 'Mini result bar: ' + plain(t or '(empty)')


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lx = []
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="models"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        pg.evaluate("()=>document.querySelector('#tile-transformer').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        Lx += ['# Home tile', txt(pg, '#tile-transformer')]
        fr = '#tile-transformer .tf-tz-frame'
        for k in ['llama32-1b', 'llama31-8b', 'llama31-70b']:
            pg.click(f'#tile-transformer .seg-ctl button[data-v="{k}"]'); pg.wait_for_timeout(1100)   # the re-tiling ends by 0.96 s
            Lx += [f'## Tile, {k}', 'Caption: ' + txt(pg, '#tile-transformer .tf-tz-cap'),
                   'Picture, as a screen reader names it: ' + plain(pg.get_attribute(fr, 'aria-label') or '(no name)'),
                   'Sentence: ' + txt(pg, '#tile-transformer .tz-read'),
                   'Legend: ' + ' | '.join(plain(x) for x in pg.eval_on_selector_all('#tile-transformer .tz-leg > span', 'ss=>ss.map(s=>s.innerText.trim())'))]
            # each legend item's tooltip says what its part of the model does (home review v4 A #7: the definitions that were the first note)
            Lx += ['Legend tooltips:'] + [f'- {plain(a_)} ⟶ {plain(t_)}' for a_, t_ in pg.eval_on_selector_all('#tile-transformer .tz-leg > span', 'ss=>ss.map(s=>[s.innerText.trim(), s.dataset.tipx||"(none)"])')]
            # the picture's tooltips, hovered with the mouse: the caption's number, the embedding table, the first and the last layer (the
            # layers' tooltips differ only in the layer's number, so hovering all 80 would bury the rest), and the LM head
            tips = []
            for name, sel, nth in [('caption number', '#tile-transformer .tf-tz-cap .num-big', 0), ('embedding table', f'{fr} .tf-tz-emb', 0),
                                   ('first layer', f'{fr} .tf-tz-grid:not(.tf-tz-old) .tf-tz-cell', 0),
                                   ('last layer', f'{fr} .tf-tz-grid:not(.tf-tz-old) .tf-tz-cell', -1), ('LM head', f'{fr} .tf-tz-head[data-tipx]', 0)]:
                els = pg.query_selector_all(sel)
                if not els:
                    tips.append(f'- {name} ⟶ (no block of its own: tied to the embedding table)'); continue
                bb = els[nth].bounding_box(); pg.mouse.move(bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2); pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else '(no tooltip)'
                T.seen.add(t); tips.append(f'- {name} ⟶ {plain(t)}'); pg.mouse.move(1, 1); pg.wait_for_timeout(60)
            Lx += ['Tooltips:'] + tips
        Lx += ['## Assumptions & sources (70B)', plain(pg.evaluate("()=>[...document.querySelectorAll('#tile-transformer .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        Lx += ['## Other tile tooltips'] + T.grab(pg, '#tile-transformer .tile-meta') + T.grab(pg, '#tile-transformer .tile-q')
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='transformer'}"); pg.wait_for_timeout(700)
        Lx += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        Lx += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            try:
                pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
                pop = plain(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))
                Lx += [f'## Popover: {" ".join(sel.inner_text().split())}', pop]
                if pg.evaluate("(id)=>{const p=document.getElementById(id); return !!p&&!p.hidden}", pid):
                    sel.click(); pg.wait_for_timeout(150)
            except Exception as e:
                Lx += [f'(popover failed: {e})']
        # ---------- guided parts ----------
        ids = pg.evaluate("()=>window.__tf.STEP_IDS")
        for k, sid in enumerate(ids):
            pg.evaluate(f"()=>document.querySelector('#tf-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(300)
            pg.evaluate("()=>window.scrollTo(0,0)"); pg.wait_for_timeout(100)
            Lx += [f'# Guided part {k + 1}', '## Card', txt(pg, '#tf-rail .step.current'), '## Stage', stage(pg)]
            Lx += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench')
            acted = False
            if sid == 'predict':
                pg.click('#tf-rail [data-pred="q2"]'); acted = True
            elif sid == 'walk':
                for i in range(6):
                    pg.click('#tf-rail [data-walk="down"]'); pg.wait_for_timeout(150)
                    Lx += [f'## Stop {i + 2}', txt(pg, '#tf-rail [data-obs]'), 'Stage: ' + txt(pg, '#tf-now'), dock(pg)]
                    Lx += ['### Tooltips in view (new)'] + T.grab(pg, '.dyn-root #tf-rail')
                acted = True
            elif sid == 'vocab':
                pg.click('#tf-ic-model button[data-v="mistral-7b"]'); pg.wait_for_timeout(250)
                Lx += ['## Card after picking Mistral-7B-v0.1', txt(pg, '#tf-rail .step.current'), '## Stage', stage(pg)]
                # the vocabulary's cost is a line in Show the math (R6.20)
                pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=true}"); pg.wait_for_timeout(150)
                Lx += ['## Show the math (Mistral-7B-v0.1)', txt(pg, '#tf-mathbody'), '## Show the math: term tooltips (new)'] + T.grab(pg, '#tf-mathbody')
                pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=false}")
                pg.click('#tf-ic-model button[data-v="llama31-8b"]'); acted = True
            elif sid == 'config':
                for i in (1, 2):
                    pg.click(f'#tf-rail [data-lever="{i}"]'); pg.wait_for_timeout(300)
                    Lx += [f'## Card after option {i + 1}', txt(pg, '#tf-rail .step.current'), '## Stage', stage(pg)]
                pg.click('#tf-rail [data-lever="0"]'); acted = True
            if acted:
                pg.wait_for_timeout(300)
                Lx += ['## Card after the action', txt(pg, '#tf-rail .step.current'), '## Stage after the action', stage(pg)]
                Lx += ['## Tooltips in view after the action (new)'] + T.grab(pg, '.dyn-root .bench')
            if sid in ('mission', 'split'):
                pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=true}"); pg.wait_for_timeout(150)
                Lx += [f'## Show the math ({"hidden split" if sid == "mission" else "revealed"})', txt(pg, '#tf-mathbody')]
                Lx += ['## Show the math: term tooltips (new)'] + T.grab(pg, '#tf-mathbody')
                pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=false}")
            Lx += [dock(pg)]
        # ---------- scenarios ----------
        pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lx += ['# Scenarios', txt(pg, '#tf-rail')]
        for i in range(5):
            pg.click(f'#tf-rail [data-preset="{i}"]'); pg.wait_for_timeout(250)
            name = ' '.join(pg.inner_text(f'#tf-rail [data-preset="{i}"]').split())
            Lx += [f'## Scenario {i + 1}: {plain(name)}', txt(pg, '#tf-insight'), '## Stage', stage(pg)]
            Lx += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root #tf-stage') + [dock(pg)]
        pg.click('#tf-ctl-p button[data-v="fp8"]'); pg.wait_for_timeout(250)
        Lx += ['## After changing the weight precision in the settings (no scenario matches)', txt(pg, '#tf-insight')]
        # ---------- settings panel ----------
        Lx += ['# Settings panel', txt(pg, '#tf-controls')]
        Lx += ['## Settings tooltips (new)'] + T.grab(pg, '#tf-controls')
        # ---------- challenge ----------
        pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lx += ['# Challenge', '## Card before submitting', txt(pg, '#tf-rail'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#tf-controls')]
        Lx += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root .bench') + [dock(pg)]
        setL = lambda L: (pg.evaluate(f"()=>{{const i=document.getElementById('tf-ic-L-in'); i.value={L}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"), pg.wait_for_timeout(200))
        setL(48); pg.click('#tf-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lx += ['## Submitted with 48 layers', txt(pg, '#tf-ch-res'), txt(pg, '#tf-status'), dock(pg)]
        setL(40)
        Lx += ['## 40 layers', txt(pg, '#tf-ch-res'), txt(pg, '#tf-status')]
        setL(44)
        Lx += ['## 44 layers', txt(pg, '#tf-ch-res'), txt(pg, '#tf-status'), dock(pg)]
        Lx += ['## Tooltips in view (new)'] + T.grab(pg, '#tf-rail')
        # once the chart shows, Show the math has the division behind the verdict (R6.20)
        pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=true}"); pg.wait_for_timeout(150)
        Lx += ['## Show the math once the chart shows', txt(pg, '#tf-mathbody'), '## Show the math: term tooltips (new)'] + T.grab(pg, '#tf-mathbody')
        pg.evaluate("()=>{const d=document.getElementById('tf-math'); if(d) d.open=false}")
        # ---------- connect ----------
        pg.click('.dyn-root [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lx += ['# Connect', txt(pg, '.dyn-root .connect')]
        Lx += ['## Connect tooltips (new)'] + T.grab(pg, '.dyn-root .connect')
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lx if x))
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
