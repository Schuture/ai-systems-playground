"""Dump every piece of text a learner meets in the roofline lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/roofline_copy_deck.py  ->  out/roofline-copy/en.md and out/roofline-copy/zh.md

Order: home tile (sentence at a few batches, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), part 3 at several batches, part 5 with each change,
Show the math, the scenarios (each), the settings panel, the challenge (every feedback tier and the chart), the card "Can you use it on
a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'roofline-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/roofline-deck.html', '--only', 'roofline'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'roofline-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'roofline-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'roofline-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#rf-st-title'), txt(pg, '#rf-st-sub'), 'Status: ' + txt(pg, '#rf-status'),
             'Key under the roofline: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#rf-pkey > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             'Key under the lanes: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#rf-lkey > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    svg = clean(pg.evaluate("()=>[...document.querySelectorAll('#rf-plot svg text')].map(t=>t.textContent).join(' | ')"))
    parts.append('Chart labels: ' + svg)
    t = txt(pg, '#rf-time')
    if t: parts.append(t)
    mets = pg.evaluate("()=>[...document.querySelectorAll('#rf-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    ph = clean(pg.evaluate("()=>{const e=document.getElementById('rf-phone'); return e?e.textContent.trim():''}"))   # phones only: the part's control or jump at the top of the chart
    return '\n'.join([p for p in parts if p] + (['Phones only, at the top of the chart: ' + ph] if ph else []) + ['Scope: ' + txt(pg, '.rf-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx], {root} .t, {root} [data-jump]'):
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.evaluate("el=>el.innerText||el.textContent||''") or '').split())[:50]
                e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(30)
                bb = e.evaluate("el=>{const r=el.getClientRects()[0]; return r?{x:r.x,y:r.y,width:r.width,height:r.height}:null}")
                if not bb: continue
                pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6)); pg.wait_for_timeout(100)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {clean(label)} ⟶ {clean(t)}')
            pg.mouse.move(1, 1)
        return out


def dock(pg):
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const r=document.querySelector('#rf-rail .step.current, #rf-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#rf-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#rf-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


def exits(pg):
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        name = clean(pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent"))
        if t == 2:
            out += [f'## {name}', clean(pg.evaluate(f"()=>{card}.querySelector('.ex-body').innerText"))]; continue
        nq = pg.evaluate(f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j: pg.evaluate(f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = clean(pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-body > p > b'); return e?e.innerText:''}}"))
            opts = pg.evaluate(f"()=>[...{card}.querySelectorAll('.ex-opts .choice')].map(b=>({{v:+b.dataset.v, t:b.innerText}}))")
            out += [f'## {name}, question {j + 1}', q] + [f'- ({o["v"]}) {clean(o["t"])}' for o in opts]
            for o in opts:
                pg.evaluate(f"()=>{card}.querySelector('.ex-opts .choice[data-v=\"{o['v']}\"]').click()"); pg.wait_for_timeout(200)
                r = pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-res'); return e?{{ok:e.classList.contains('ok'),t:e.innerText}}:null}}")
                if not r: continue
                out += [f'### After picking ({o["v"]})', clean(r['t'])]
                if r['ok']: break
    return out


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        # home tile
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="roots"], #domStrip [data-world="roots"]')].find(e=>e.offsetParent);
          if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-roofline'):
            pg.evaluate("()=>document.querySelector('#tile-roofline').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-roofline')]
            bst = pg.evaluate("()=>window.__tzRf.BST")
            for B in (1, 16, 64, 256, 512):
                rng(pg, '#tile-roofline input[type=range]', bst.index(B))
                Lines += [f'## Tile sentence at batch {B}', txt(pg, '#tile-roofline .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-roofline .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='roofline'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__rf.consts.STEP_IDS")
        bst = pg.evaluate("()=>window.__rf.consts.BST"); mst = pg.evaluate("()=>window.__rf.consts.MST")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#rf-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#rf-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#rf-app .bench')
            acted = True
            if sid == 'predict': pg.click('#rf-rail [data-pred="few"]')
            elif sid == 'batch': rng(pg, '#rf-ic-B-in', bst.index(16))
            elif sid == 'matmul': rng(pg, '#rf-ic-n-in', mst.index(512))
            elif sid == 'lever': pg.click('#rf-rail [data-lever="a100"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#rf-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#rf-app .bench')
            if sid == 'batch':
                for v in (64, 256, 512):
                    rng(pg, '#rf-ic-B-in', bst.index(v)); Lines += [f'## Result at batch {v}', txt(pg, '#rf-rail [data-obs]')]
                Lines += ['## Stage at batch 512', stage(pg)]
                pg.evaluate("()=>{document.getElementById('rf-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math at batch 512', txt(pg, '#rf-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#rf-mathbody')
                pg.evaluate("()=>{document.getElementById('rf-math').open=false}")
                go(pg, 1); pg.click('#rf-rail [data-pred="same"]'); pg.wait_for_timeout(200); go(pg, 2); rng(pg, '#rf-ic-B-in', bst.index(64))
                Lines += ['## Part 3 after the right prediction (batch 64)', txt(pg, '#rf-rail [data-obs]')]
            if sid == 'matmul':
                rng(pg, '#rf-ic-n-in', mst.index(1024)); Lines += ['## Result at n = 1,024', txt(pg, '#rf-rail [data-obs]'), '## Stage', stage(pg)]
            if sid == 'lever':
                pg.click('#rf-rail [data-lever="fp8"]'); pg.wait_for_timeout(300)
                Lines += ['## Part 5 with FP8 weights', txt(pg, '#rf-rail .step.current'), '## Stage', stage(pg)]
                pg.evaluate("()=>{document.getElementById('rf-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5, FP8 weights)', txt(pg, '#rf-mathbody')]
                pg.evaluate("()=>{document.getElementById('rf-math').open=false}")
            Lines += [dock(pg)]
        pg.click('#rf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#rf-rail')]
        for i in range(5):
            pg.click(f'#rf-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#rf-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#rf-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#rf-stage') + [dock(pg)]
        pg.click('#rf-ctl-gpu button[data-v="a100"]'); pg.wait_for_timeout(300)
        Lines += ['## After changing the GPU under Settings (no scenario matches)', txt(pg, '#rf-insight'), '## Stage', stage(pg)]
        Lines += ['# Settings panel', txt(pg, '#rf-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#rf-controls')
        pg.click('#rf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#rf-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#rf-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#rf-app .bench') + [dock(pg)]
        seg = lambda k, v: (pg.click(f'#rf-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'), pg.wait_for_timeout(250))
        pg.click('#rf-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (goal missed)', txt(pg, '#rf-ch-res')]
        seg('chB', '64'); Lines += ['## Batch 64, BF16 (goal missed)', txt(pg, '#rf-ch-res')]
        pg.click('#rf-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “Show the answer”', txt(pg, '#rf-ch-res')]
        seg('chB', '256'); Lines += ['## Batch 256, BF16 (met, more requests than needed)', txt(pg, '#rf-ch-res')]
        seg('chB', '64'); seg('chP', 'fp8'); Lines += ['## Batch 64, FP8 (met, the fewest requests)', txt(pg, '#rf-ch-res'), 'Status: ' + txt(pg, '#rf-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('rf-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#rf-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#rf-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#rf-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#rf-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#rf-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
