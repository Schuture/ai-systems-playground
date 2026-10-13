"""Dump every piece of text a learner meets in the prefix-caching lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/prefix-caching_copy_deck.py  ->  out/prefix-caching-copy/en.md and out/prefix-caching-copy/zh.md

Order: home tile (the sentence for every option, the notes behind "Assumptions & sources"), lab header with its tooltips and both
popovers, each guided part (card before and after its action, the stage, the tooltips in view that were not shown before, the mini
result bar), the prediction's reveal for a wrong and for the right pick, every lever of parts 4 and 6, part 5's experiment (the card
before and after a position and a prediction, each position's result and stage, the verdict for each prediction at each position that
tells them apart or not, block by block at every stop, the phone's question under the positions and the line to the result, its Show
the math), Show the math, the scenarios
card for every scenario, the settings panel, the challenge before and after submitting (every feedback tier and the chart), the card
"Can you use it on a new case?" (every question, a verdict for every option), and the connect section with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'prefix-caching-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefix-caching-deck.html', '--only', 'prefix-caching,set-a'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-prefix-caching-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefix-caching-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefix-caching-deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#px-st-title'), txt(pg, '#px-st-sub'), 'Status: ' + txt(pg, '#px-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             'Lead: ' + pg.evaluate("()=>document.getElementById('px-graph').dataset.lead")]
    parts += ['Rows: ' + txt(pg, '#px-rows-cap')] + ['- ' + clean(' '.join(x.split())) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-rows .px-row')].filter(r=>r.offsetParent).map(r=>r.innerText)")] + [txt(pg, '#px-sum')]
    pg.wait_for_timeout(60)   # the pinned labels are placed on the next frame
    parts += ['Strip: ' + txt(pg, '#px-strip-cap'), 'Band: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-band .px-bs')].filter(s=>s.offsetParent).map(s=>s.innerText||'(pinned)')")),
              'Pinned above: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-pins-top .px-pin')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
              'Pinned below: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-pins .px-pin')].filter(s=>s.offsetParent).map(s=>s.innerText)")), txt(pg, '#px-strip-foot')]
    parts += ['Timeline: ' + txt(pg, '#px-tl-cap')] + [clean(' '.join(x.split())) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-tl .px-tll')].filter(r=>r.offsetParent).map(r=>r.innerText)")]
    parts += ['Axis: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('.px-axis span')].filter(s=>s.offsetParent).map(s=>s.textContent).join('  ')")), 'Scope: ' + txt(pg, '.px-scope')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#px-metrics .metric')].filter(m=>m.offsetParent).map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + pair(pg) + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


def pair(pg):
    """part 5 (Batch 4): request 1 and request 2 block by block, the keys, the positions, the counters"""
    if not pg.is_visible('#px-pairbox'): return []
    out = ['Pair: ' + txt(pg, '#px-pair-cap'), 'Rows: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-pair .px-plab')].map(e=>e.innerText).filter(Boolean)")),
           'Band: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-pband .px-bs')].map(s=>getComputedStyle(s.querySelector('em')).visibility==='hidden'?'(pinned)':s.innerText)")),
           'Pinned: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-ppins .px-pin')].map(s=>s.innerText)")),
           'Request 2 cover: ' + txt(pg, '#px-pcover'),
           'Keys: ' + ' | '.join(clean(' '.join(x.split())) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-lane .px-kp')].filter(e=>!e.hidden).map(e=>e.innerText)")),
           'Positions: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#px-ppos .px-pbtn')].map(b=>b.innerText+(b.getAttribute('aria-pressed')==='true'?' (picked)':''))")),
           'Label: ' + (txt(pg, '#px-same') if pg.evaluate("()=>getComputedStyle(document.getElementById('px-same')).visibility")=='visible' else ''),
           'Progress: ' + txt(pg, '#px-prog'), 'Button: ' + txt(pg, '#px-otstep'), 'Counters:']
    out += [f'- {clean(m)}' for m in pg.evaluate("()=>[...document.querySelectorAll('#px-ctrs .px-ctr')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")]
    return out


def sweep(pg):
    try: pg.wait_for_function("()=>window.__px.state().sweep===null", timeout=8000)
    except Exception: pass
    pg.wait_for_timeout(150); calm(pg)


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn], {root} [data-jump]'):
            try:
                if not e.is_visible(): continue
                # a part too narrow for its label: its pinned label under or above the strip carries the same tooltip
                if e.evaluate("el=>el.classList.contains('px-bs')&&getComputedStyle(el.querySelector('em')).visibility==='hidden'"): continue
                label = ' '.join((e.evaluate("el=>el.innerText||el.textContent||''") or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(40)
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


def calm(pg):
    try: pg.wait_for_function("()=>!window.__px.state().stageBusy", timeout=4000)
    except Exception: pass
    last = None
    for _ in range(25):
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120)


def dock(pg):
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const r=document.querySelector('#px-rail .step.current, #px-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-60); document.querySelector('#px-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def click(pg, sel, w=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(w); calm(pg)


def go(pg, k): click(pg, f'#px-rail [data-go="{k}"]', 350)


def exits(pg):
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        name = clean(pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent"))
        if t == 2:
            out += [f'## {name}', clean(pg.evaluate(f"()=>{card}.querySelector('.ex-body').innerText"))]
            continue
        nq = pg.evaluate(f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j:
                pg.evaluate(f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = clean(pg.evaluate(f"()=>{{const c={card}, e=c.querySelector('.ex-body > p > b'); return e?e.innerText:''}}"))
            opts = pg.evaluate(f"()=>[...{card}.querySelectorAll('.ex-opts .choice')].map(b=>({{v:+b.dataset.v, t:b.innerText}}))")
            out += [f'## {name}, question {j + 1}', q] + [f'- ({o["v"]}) {clean(o["t"])}' for o in opts]
            for o in opts:
                pg.evaluate(f"()=>{card}.querySelector('.ex-opts .choice[data-v=\"{o['v']}\"]').click()"); pg.wait_for_timeout(200)
                r = pg.evaluate(f"()=>{{const e={card}.querySelector('.ex-res'); return e?{{ok:e.classList.contains('ok'),t:e.innerText}}:null}}")
                if not r: continue
                out += [f'### After picking ({o["v"]})', clean(r['t'])]
    return out


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="inference"], #domStrip [data-world="inference"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="inference"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-prefix-caching'):
            pg.evaluate("()=>document.querySelector('#tile-prefix-caching').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-prefix-caching')]
            for v in pg.evaluate("()=>[...document.querySelectorAll('#tile-prefix-caching .seg-ctl button')].map(b=>b.dataset.v)"):
                pg.evaluate(f"()=>document.querySelector('#tile-prefix-caching .seg-ctl button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(150)
                Lines += [f'## Tile sentence at {v} requests', txt(pg, '#tile-prefix-caching .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-prefix-caching .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='prefix-caching'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead'), '## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        # ---------- guided parts ----------
        ids = pg.evaluate("()=>window.__px.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Lead (the phone’s parts bar): ' + clean(pg.evaluate("()=>{const e=document.querySelector('#px-rail .pbar .pb-t'); return e?e.textContent:''}"))]
            Lines += ['## Card', txt(pg, '#px-rail .step.current'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#px-app .bench')
            if sid == 'predict':
                click(pg, '#px-rail [data-pred="half"]', 200); Lines += ['## Card after a pick', txt(pg, '#px-rail .step.current')]
            if sid == 'send':
                click(pg, '#px-rail [data-act="send"]', 900)
                Lines += ['## Card after sending (prediction: less, but still long)', txt(pg, '#px-rail .step.current'), '## Stage after sending', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
                go(pg, 1); click(pg, '#px-rail [data-pred="few"]', 200); go(pg, 2)
                Lines += ['## Part 3 after the right prediction', txt(pg, '#px-rail [data-obs]')]
            if sid == 'break':
                for v in ['dateStart', 'dateAfter', 'edit', 'nameStart']:
                    click(pg, f'#px-rail [data-lever="{v}"]', 300)
                    Lines += [f'## Lever {v}', txt(pg, '#px-rail [data-obs]'), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
                Lines += ['## Levers with their results', txt(pg, '#px-rail .levers')]
            if sid == 'onetoken':
                click(pg, '#px-stage [data-otpos="middle"]', 300)
                Lines += ['## After picking a position in the chart (the middle)', txt(pg, '#px-rail .step.current'), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
                click(pg, '#px-rail [data-otpred="none"]', 200); sweep(pg)
                Lines += ['## After the prediction “nothing” and the sweep (the middle)', txt(pg, '#px-rail .step.current'), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
                for v in ['front', 'end']:
                    click(pg, f'#px-stage [data-otpos="{v}"]', 300); sweep(pg)
                    Lines += [f'## Position {v}', txt(pg, '#px-rail [data-obs]'), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
                Lines += ['## Positions with their results', txt(pg, '#px-rail .levers')]
                click(pg, '#px-otstep', 150)
                Lines += ['## Block by block, before the first click (the end position)', '## Stage', stage(pg)]
                for j in range(3):
                    click(pg, '#px-otstep', 150)
                    Lines += [f'## Block by block, stop {j + 1}', '## Stage', stage(pg)]
                click(pg, '#px-stage [data-otpos="middle"]', 300); sweep(pg)
                pg.evaluate("()=>{document.getElementById('px-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5)', txt(pg, '#px-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#px-mathbody')
                pg.evaluate("()=>{document.getElementById('px-math').open=false}")
                # the verdict for the other predictions, each on a fresh page (a prediction is final once the result shows)
                for pk, pos in [('before', 'middle'), ('block', 'middle'), ('block', 'front'), ('none', 'front'), ('before', 'front'), ('block', 'end'), ('none', 'end'), ('before', 'end')]:
                    p2 = b.new_page(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
                    p2.goto(URL); p2.wait_for_timeout(300); p2.click(f'.langsw button[data-lang="{lang}"]'); p2.wait_for_timeout(100)
                    p2.evaluate("()=>{location.hash='prefix-caching'}"); p2.wait_for_timeout(600)
                    p2.evaluate(f"()=>{{document.querySelector('#px-rail [data-go=\"{k}\"]').click(); document.querySelector('#px-stage [data-otpos=\"{pos}\"]').click(); document.querySelector('#px-rail [data-otpred=\"{pk}\"]').click()}}"); p2.wait_for_timeout(400)
                    Lines += [f'## Part 5 result after the prediction “{pk}” at {pos}', txt(p2, '#px-rail [data-obs]')]
                    p2.close()
                # the phone (390 px): the question under the positions in the chart, then the line under the chart that leads to the result
                p3 = b.new_page(viewport={'width': 390, 'height': 844}, reduced_motion='reduce')
                p3.goto(URL); p3.wait_for_timeout(300); p3.click(f'.langsw button[data-lang="{lang}"]'); p3.wait_for_timeout(100)
                p3.evaluate("()=>{location.hash='prefix-caching'}"); p3.wait_for_timeout(600)
                p3.evaluate(f"()=>{{document.querySelector('#px-rail [data-go=\"{k}\"]').click(); document.querySelector('#px-stage [data-otpos=\"end\"]').click()}}"); p3.wait_for_timeout(400)
                Lines += ['## Phone (390 px): after picking the end in the chart', 'Status: ' + txt(p3, '#px-status'), 'Title: ' + txt(p3, '#px-st-sub'), 'Request 2 cover: ' + txt(p3, '#px-pcover'), 'Question in the chart: ' + txt(p3, '#px-otqs')]
                p3.evaluate("()=>document.querySelector('#px-stage [data-otpred=\"block\"]').click()"); p3.wait_for_timeout(500)
                Lines += ['## Phone: after the prediction', 'Line under the chart: ' + txt(p3, '#px-prog'), 'Button: ' + txt(p3, '#px-otstep')]
                p3.close()
            if sid == 'cost':
                for v in ['2', '1', 'pool']:
                    click(pg, f'#px-rail [data-lever="{v}"]', 300)
                    Lines += [f'## Memory {v}', txt(pg, '#px-rail [data-obs]'), '## Stage', stage(pg)]
                Lines += ['## Tooltips (new)'] + T.grab(pg, '#px-app .bench')
            Lines += [dock(pg)]
        # ---------- Show the math ----------
        for tok, name in [('10000_200_2_same_none_1_pool', 'request 2 of the mission'), ('10000_200_2_same_edit_1_pool', 'one word changed')]:
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
            pg.evaluate(f"()=>{{location.hash='prefix-caching~{tok}'}}"); pg.wait_for_timeout(500)
            pg.evaluate("()=>{document.getElementById('px-math').open=true}"); pg.wait_for_timeout(150)
            Lines += [f'# Show the math ({name})', txt(pg, '#px-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#px-mathbody')
        # ---------- scenarios ----------
        Lines += ['# Scenarios', txt(pg, '#px-rail')]
        for i in range(4):
            click(pg, f'#px-rail [data-preset="{i}"]', 300)
            name = clean(' '.join(pg.inner_text(f'#px-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#px-insight'), '## Stage', stage(pg), '## Tooltips (new)'] + T.grab(pg, '#px-stage') + [dock(pg)]
        pg.evaluate("()=>{const i=document.querySelector('#px-ctl-O-in'); i.value=0; i.dispatchEvent(new Event('input',{bubbles:true}));}"); pg.wait_for_timeout(250)
        Lines += ['## After changing the own tokens under Settings (no scenario matches)', txt(pg, '#px-insight')]
        Lines += ['# Settings panel', txt(pg, '#px-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#px-controls')
        # ---------- challenge ----------
        click(pg, '#px-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#px-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#px-controls')]
        Lines += ['## Tooltips (new)'] + T.grab(pg, '#px-app .bench') + [dock(pg)]
        click(pg, '#px-rail [data-act="submit"]', 300)
        Lines += ['## Submitted the initial order (goal missed)', txt(pg, '#px-ch-res')]
        click(pg, '#px-rail [data-act="reveal"]', 300)
        Lines += ['## After “show all options”', txt(pg, '#px-ch-res')]
        for target, nm in [(['shots', 'sys', 'name', 'hist', 'date', 'msg'], 'met, 3 parts moved'), (['sys', 'shots', 'name', 'hist', 'msg', 'date'], 'met, 2 parts moved')]:
            for i, p_ in enumerate(target):
                for _ in range(6):
                    if pg.evaluate("()=>window.__px.state().chOrder").index(p_) <= i: break
                    pg.evaluate(f"()=>document.querySelector('#px-order [data-part=\"{p_}\"][data-act=\"up\"]').click()"); pg.wait_for_timeout(60)
            pg.wait_for_timeout(250)
            Lines += [f'## Order {", ".join(target)} ({nm})', txt(pg, '#px-rail .railcard'), '## Stage', stage(pg)]
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#px-app [data-act="drill-reveal"]', 200)
        Lines += ['# Connect', txt(pg, '#px-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#px-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
