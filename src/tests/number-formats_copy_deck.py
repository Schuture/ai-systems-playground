"""Dump every piece of text a learner meets in the number-formats lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/number-formats_copy_deck.py  ->  out/number-formats-copy/en.md and out/number-formats-copy/zh.md

Order: home tile (sentence at a few values, the notes), lab header and its popovers, each guided part (card before and after its action,
the stage, the tooltips in view not shown before, the mini result bar), the reveal for the right pick, part 5 with each fix, part 6 with
each picture, Show the math, the scenarios (each), the settings panel, the challenge (every feedback tier and the chart), the card
"Can you use it on a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'number-formats-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/number-formats-deck.html', '--only', 'number-formats'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'number-formats-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'number-formats-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'number-formats-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#nf-st-title'), txt(pg, '#nf-st-sub'), 'Status: ' + txt(pg, '#nf-status'),
             'Keys under the charts: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('.nf-key > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#nf-a-cap', '#nf-rows', '#nf-range', '#nf-b-cap', '#nf-zoom', '#nf-c-cap', '#nf-paths', '#nf-chart-cap'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()) if sel in ('#nf-rows', '#nf-range', '#nf-zoom', '#nf-paths') else t)
    mets = pg.evaluate("()=>[...document.querySelectorAll('#nf-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    ph = clean(pg.evaluate("()=>{const e=document.getElementById('nf-phone'); return e?e.textContent.trim():''}"))
    return '\n'.join([p for p in parts if p] + (['Phones only, at the top of the chart: ' + ph] if ph else []) + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#nf-rail .step.current, #nf-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#nf-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#nf-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="roots"], #domStrip [data-world="roots"]')].find(e=>e.offsetParent); if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-number-formats'):
            pg.evaluate("()=>document.querySelector('#tile-number-formats').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-number-formats')]
            xs = pg.evaluate('()=>window.__tzNf.xs')
            for v in (1e-8, 5e-8, 1e-4, 0.02, 1.0, 500, 1e5):
                rng(pg, '#tile-number-formats input[type=range]', xs.index(v))
                Lines += [f'## Tile sentence at {v}', txt(pg, '#tile-number-formats .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-number-formats .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='number-formats'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__nf.consts.STEP_IDS")
        XS = pg.evaluate("()=>window.__nf.consts.XS"); US = pg.evaluate("()=>window.__nf.consts.US")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#nf-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#nf-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#nf-app .bench')
            acted = True
            if sid == 'predict': pg.click('#nf-rail [data-pred="kept"]')
            elif sid == 'range': rng(pg, '#nf-ic-x-in', XS.index(1e-8))
            elif sid == 'prec': rng(pg, '#nf-ic-u-in', US.index(1e-3))
            elif sid == 'fix': pg.click('#nf-rail [data-lever="fp16ls"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#nf-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#nf-app .bench')
            if sid == 'range':
                for v in (5e-8, 1e5, 2e-3):
                    rng(pg, '#nf-ic-x-in', XS.index(v)); Lines += [f'## Result at {v}', txt(pg, '#nf-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('nf-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 3, 2e-3)', txt(pg, '#nf-mathbody')]
                pg.evaluate("()=>{document.getElementById('nf-math').open=false}")
                go(pg, 1); pg.click('#nf-rail [data-pred="zero"]'); pg.wait_for_timeout(200); go(pg, 2); rng(pg, '#nf-ic-x-in', XS.index(1e-8))
                Lines += ['## Part 3 after the right prediction (1e-8)', txt(pg, '#nf-rail [data-obs]')]
            if sid == 'prec':
                for v in (3e-4, 1e-2):
                    rng(pg, '#nf-ic-u-in', US.index(v)); Lines += [f'## Result at {v}', txt(pg, '#nf-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('nf-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 4)', txt(pg, '#nf-mathbody')]
                pg.evaluate("()=>{document.getElementById('nf-math').open=false}")
            if sid == 'fix':
                for lv in ('fp16m', 'fp16lsm'):
                    pg.click(f'#nf-rail [data-lever="{lv}"]'); pg.wait_for_timeout(300)
                    Lines += [f'## Part 5 with {lv}', txt(pg, '#nf-rail .step.current'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#nf-stage')
                pg.evaluate("()=>{document.getElementById('nf-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5)', txt(pg, '#nf-mathbody')]
                pg.evaluate("()=>{document.getElementById('nf-math').open=false}")
            if sid == 'remember':
                for kk in ('b', 'c', 'a'):
                    pg.click(f'#nf-rail [data-rem="{kk}"]'); pg.wait_for_timeout(350)
                    Lines += [f'## Part 6 after “Show in the chart” ({kk})', txt(pg, '#nf-rail .step.current'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#nf-app .bench')
            Lines += [dock(pg)]
        pg.click('#nf-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#nf-rail')]
        for i in range(4):
            pg.click(f'#nf-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#nf-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#nf-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#nf-stage') + [dock(pg)]
        rng(pg, '#nf-ctl-x-in', XS.index(500))
        Lines += ['## After changing the value under Settings (no scenario matches)', txt(pg, '#nf-insight'), '## Stage', stage(pg)]
        pg.click('#nf-ctl-blk-wrap [data-v="0"]'); pg.wait_for_timeout(250)
        Lines += ['## 500 with no block scale', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#nf-stage')
        Lines += ['# Settings panel', txt(pg, '#nf-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#nf-controls')
        pg.click('#nf-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#nf-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#nf-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#nf-app .bench') + [dock(pg)]
        pg.click('#nf-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup, all FP32 (goal missed)', txt(pg, '#nf-ch-res')]
        for lv, name in (('fp16', 'All FP16 (missed)'), ('bf16', 'All BF16 (missed)')):
            pg.click(f'#nf-rail #nf-ic-ch [data-lever="{lv}"]'); pg.wait_for_timeout(300); Lines += [f'## {name}', txt(pg, '#nf-ch-res')]
        pg.click('#nf-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “Show the answer”', txt(pg, '#nf-ch-res')]
        pg.click('#nf-rail #nf-ic-ch [data-lever="fp16lsm"]'); pg.wait_for_timeout(300); Lines += ['## FP16 + loss scaling + master copy (met, two extra parts)', txt(pg, '#nf-ch-res')]
        pg.click('#nf-rail #nf-ic-ch [data-lever="bf16m"]'); pg.wait_for_timeout(300)
        Lines += ['## BF16 + master copy (met, the fewest)', txt(pg, '#nf-ch-res'), 'Status: ' + txt(pg, '#nf-status'), '## Live checks', txt(pg, '#nf-ch-g'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('nf-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#nf-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#nf-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#nf-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#nf-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#nf-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
