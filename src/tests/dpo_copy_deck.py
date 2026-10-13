"""Dump every piece of text a learner meets in the DPO lab and its home tile, in reading order, in Chinese and English.

python3 tests/dpo_copy_deck.py [tag]  ->  out/dpo-copy/<tag>-zh.md and out/dpo-copy/<tag>-en.md
(without a tag: now-zh.md and now-en.md, and also zh.md and en.md, the files tests/numdense.py reads)

Order: home tile (sentence at several step counts, legend, the notes behind "Assumptions & sources"), lab header with the
mode tooltips and both popovers, each guided part (card before and after its action, the stage, the tooltips in view that
were not shown before, the mini result bar), "Show the math", the scenarios card for every preset, the settings panel,
the challenge before and after submitting (every feedback tier and the chart), and the connect section with the drill answer.
A tooltip is printed the first time it appears with a given text, so a repeated legend does not drown the rest.
"""
import pathlib, sys, subprocess
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'now'
OUT = P / 'out' / 'dpo-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/dpo.html', '--only', 'dpo'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-dpo.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'dpo.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-dpo.html')

def txt(pg, sel):
    el = pg.query_selector(sel)
    return el.inner_text().strip() if el and el.is_visible() else ''

def stage(pg):
    """The stage in reading order, without the chart's tick labels."""
    parts = [txt(pg, '#dpo-st-title'), txt(pg, '#dpo-status'), txt(pg, '#dpo-st-sub'), txt(pg, '#dpo-transport'), txt(pg, '#dpo-stage .dpo-prompt'),
             txt(pg, '#dpo-stage .legend-row'), txt(pg, '#dpo-bars'), txt(pg, '#dpo-curve-title'), txt(pg, '#dpo-view'), txt(pg, '#dpo-curve-key')]
    extra = []
    mets = pg.evaluate("()=>[...document.querySelectorAll('#dpo-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + extra + ['Metrics:'] + [f'- {m}' for m in mets])

class Tips:
    """Hover every visible element that has a tooltip and record the text the first time it appears."""
    def __init__(self): self.seen = set()
    def grab(self, pg, root):
        out = []
        els = pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn]')
        for e in els:
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.inner_text() or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); bb = e.bounding_box()
                if not bb: continue
                pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            key = t
            if t and key not in self.seen:
                self.seen.add(key); out.append(f'- {label} ⟶ {t}')
            pg.mouse.move(1, 1)
        return out

def dock(pg):
    """Hide the answer bars (open the math and scroll the sticky stage to its end) so the mini result bar shows; read it; restore."""
    was = pg.evaluate("()=>{const d=document.getElementById('dpo-math'), o=d.open; d.open=true; const st=document.getElementById('dpo-stage'); st.scrollTop=st.scrollHeight; st.dispatchEvent(new Event('scroll')); window.dispatchEvent(new Event('scroll')); return o}"); pg.wait_for_timeout(400)
    t = ' '.join(pg.inner_text('#dock').split()) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.evaluate("(o)=>{const d=document.getElementById('dpo-math'); d.open=o; const st=document.getElementById('dpo-stage'); st.scrollTop=0; st.dispatchEvent(new Event('scroll')); window.dispatchEvent(new Event('scroll'))}", was); pg.wait_for_timeout(250)
    return 'Mini result bar: ' + t

def wait_next(pg, limit=160):
    for _ in range(limit):
        nb = pg.query_selector('#dpo-rail [data-nav="next"]')
        if nb and not nb.is_disabled(): return
        pg.wait_for_timeout(150)

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['zh', 'en']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); L = []
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="post"], #domStrip [data-world="post"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="post"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        pg.evaluate("()=>document.querySelector('#tile-dpo').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
        L += ['# Home tile', txt(pg, '#tile-dpo')]
        for v in (0, 1, 40):
            pg.evaluate(f"()=>{{const i=document.querySelector('#tile-dpo input[type=range]'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}}))}}"); pg.wait_for_timeout(120)
            L += [f'## Tile sentence at {v} steps', txt(pg, '#tile-dpo .tz-read')]
        L += ['## Assumptions & sources', pg.evaluate("()=>[...document.querySelectorAll('#tile-dpo .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')")]
        L += ['## Tile tooltips'] + T.grab(pg, '#tile-dpo')
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='dpo'}"); pg.wait_for_timeout(700)
        L += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        L += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            try:
                pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
                pop = pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid)
                L += [f'## Popover: {" ".join(sel.inner_text().split())}', pop]
                if pg.evaluate("(id)=>{const p=document.getElementById(id); return !!p&&!p.hidden}", pid): sel.click(); pg.wait_for_timeout(150)
            except Exception as e:
                L += [f'(popover failed: {e})']
        # ---------- guided parts ----------
        pg.click('.dyn-root .modes button[data-mode="guided"]'); pg.wait_for_timeout(300)
        ids = pg.evaluate("()=>(window.__dpo&&window.__dpo.consts.STEP_IDS)||['mission','predict','steps','beta','loss','fade','shared','remember']")
        for k, sid in enumerate(ids):
            pg.evaluate(f"()=>document.querySelector('#dpo-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(350)
            L += [f'# Guided part {k + 1}', '## Card', txt(pg, '#dpo-rail .step.current'), '## Stage', stage(pg)]
            L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root')
            acted = False
            if sid == 'predict':
                pg.click('#dpo-rail [data-pred="down"]'); acted = True
            elif sid == 'steps':
                pg.click('#dpo-rail [data-act="step10"]'); pg.wait_for_timeout(200); pg.click('#dpo-rail [data-act="step1"]'); acted = True
            elif sid == 'beta':
                pg.click('#dpo-ic-beta button[data-v="1"]'); pg.wait_for_timeout(250)
                L += ['## Card after picking β = 1', txt(pg, '#dpo-rail .step.current')]
                pg.click('#dpo-ic-beta button[data-v="2"]'); acted = True
            elif sid == 'fade':
                sm = pg.query_selector('#dpo-rail [data-nav="showme"]')
                if sm and sm.is_visible(): sm.click()
                wait_next(pg)
                for _ in range(80):
                    if not pg.evaluate("()=>window.__dpo.state().playing"): break
                    pg.wait_for_timeout(150)
                pg.wait_for_timeout(300); acted = True
            elif sid == 'shared':
                pg.click('#dpo-ic-mode button[data-v="shared"]'); acted = True
            if acted:
                pg.wait_for_timeout(350)
                L += ['## Card after the action', txt(pg, '#dpo-rail .step.current'), '## Stage after the action', stage(pg)]
                L += ['## Tooltips in view after the action (new)'] + T.grab(pg, '.dyn-root')
            if sid == 'steps':   # the other variant of the observation: after a full run C and D have fallen below where they started
                pg.evaluate("()=>{const i=document.getElementById('dpo-ic-steps-in'); i.value=100; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(300)
                L += ['## Observation at 100 steps', txt(pg, '#dpo-rail [data-obs]')]
            if sid == 'loss':
                pg.evaluate("()=>{const d=document.getElementById('dpo-math'); if(d) d.open=true}"); pg.wait_for_timeout(200)
                L += ['## Show the math (40 steps, β 0.1)', txt(pg, '#dpo-mathbody')]
                L += ['## Show the math: term tooltips (new)'] + T.grab(pg, '#dpo-mathbody')
            L += [dock(pg)]
        # ---------- scenarios ----------
        pg.click('.dyn-root .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        L += ['# Scenarios', txt(pg, '#dpo-rail')]
        n = len(pg.query_selector_all('#dpo-rail [data-preset]'))
        for i in range(n):
            pg.click(f'#dpo-rail [data-preset="{i}"]'); pg.wait_for_timeout(250)
            name = ' '.join(pg.inner_text(f'#dpo-rail [data-preset="{i}"]').split())
            L += [f'## Scenario {i + 1}: {name}', txt(pg, '#dpo-insight'), '## Stage', stage(pg)]
            L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root #dpo-stage') + [dock(pg)]
        pg.click('#dpo-ctl-beta button[data-v="0.5"]'); pg.wait_for_timeout(250)
        L += ['## After changing β in the settings (no scenario matches)', txt(pg, '#dpo-insight')]
        # ---------- settings panel ----------
        pg.evaluate("()=>{const d=document.getElementById('dpo-adv'); if(d) d.open=true}"); pg.wait_for_timeout(150)
        L += ['# Settings panel', txt(pg, '#dpo-controls')]
        L += ['## Settings tooltips (new)'] + T.grab(pg, '#dpo-controls')
        # ---------- challenge ----------
        pg.click('.dyn-root .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        L += ['# Challenge', '## Card before submitting', txt(pg, '#dpo-rail'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#dpo-controls')]
        L += ['## Tooltips in view (new)'] + T.grab(pg, '.dyn-root') + [dock(pg)]
        pg.click('#dpo-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        L += ['## After submitting with β = 0.1', txt(pg, '#dpo-ch-res'), txt(pg, '#dpo-status'), dock(pg)]
        pg.click('#dpo-ic-beta button[data-v="2"]'); pg.wait_for_timeout(250)
        L += ['## β = 2', txt(pg, '#dpo-ch-res'), txt(pg, '#dpo-status'), dock(pg)]
        rv = pg.query_selector('#dpo-ch-res [data-act="reveal"]')
        if rv: rv.click(); pg.wait_for_timeout(250)
        L += ['## After “Show the answer”', txt(pg, '#dpo-ch-res')]
        pg.click('#dpo-ic-beta button[data-v="0.5"]'); pg.wait_for_timeout(250)
        L += ['## β = 0.5', txt(pg, '#dpo-ch-res'), txt(pg, '#dpo-status')]
        pg.click('#dpo-ic-beta button[data-v="1"]'); pg.wait_for_timeout(250)
        L += ['## β = 1', txt(pg, '#dpo-ch-res'), txt(pg, '#dpo-status'), dock(pg)]
        L += ['## Tooltips in view (new)'] + T.grab(pg, '#dpo-rail')
        # ---------- connect ----------
        pg.click('.dyn-root [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        L += ['# Connect', txt(pg, '.dyn-root .connect')]
        L += ['## Connect tooltips (new)'] + T.grab(pg, '.dyn-root .connect')
        (OUT / f'{tag}-{lang}.md').write_text('\n\n'.join(x for x in L if x))
        if len(sys.argv) <= 1: (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in L if x))   # the name tests/numdense.py reads
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{tag}-{l}.md') for l in ['zh', 'en']])
