"""Dump every piece of text a learner meets in the sampling lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/sampling_copy_deck.py  ->  out/sampling-copy/en.md and out/sampling-copy/zh.md

Order: home tile (sentence at a few temperatures, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), the reveal for the right pick, part 4 with each rule,
part 5 at a rare word, Show the math, the scenarios (each, and the tree view), the settings panel, the challenge (every feedback tier and
the chart), the card "Can you use it on a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'sampling-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/sampling-deck.html', '--only', 'sampling,set-b'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'sampling-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'sampling-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'sampling-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#sp-st-title'), txt(pg, '#sp-st-sub'), 'Status: ' + txt(pg, '#sp-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#sp-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#sp-phone', '#sp-a-cap', '#sp-rows', '#sp-cum-cap', '#sp-cumbar', '#sp-cum-marks', '#sp-draw-cap', '#sp-strip-cap', '#sp-duo', '#sp-b-cap', '#sp-tree', '#sp-sent'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()) if sel in ('#sp-tree', '#sp-cumbar', '#sp-cum-marks') else t)
    mets = pg.evaluate("()=>[...document.querySelectorAll('#sp-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    ph = clean(pg.evaluate("()=>{const e=document.getElementById('sp-phone'); return e?e.textContent.trim():''}"))   # phones only: the part's control or jump at the top of the chart
    return '\n'.join([p for p in parts if p] + (['Phones only, at the top of the chart: ' + ph] if ph else []) + ['Scope: ' + txt(pg, '.sp-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#sp-rail .step.current, #sp-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#sp-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#sp-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="models"], #domStrip [data-world="models"]')].find(e=>e.offsetParent);
          if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-sampling'):
            pg.evaluate("()=>document.querySelector('#tile-sampling').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-sampling')]
            for v in (1, 5, 10, 20):
                rng(pg, '#tile-sampling input[type=range]', v)
                Lines += [f'## Tile sentence at temperature {v / 10}', txt(pg, '#tile-sampling .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-sampling .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='sampling'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__sp.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#sp-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#sp-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#sp-app .bench')
            acted = True
            if sid == 'predict': pg.click('#sp-rail [data-pred="few"]')
            elif sid == 'temp': rng(pg, '#sp-ic-T-in', 5)
            elif sid == 'cut': pg.click('#sp-rail [data-lever="k3"]')
            elif sid == 'tree': rng(pg, '#sp-ic-u1-in', 97)
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#sp-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#sp-app .bench')
            if sid == 'temp':
                for v in (15, 0):
                    rng(pg, '#sp-ic-T-in', v); Lines += [f'## Result at temperature {v / 10}', txt(pg, '#sp-rail [data-obs]')]
                Lines += ['## Stage at temperature 0', stage(pg)]
                pg.evaluate("()=>{document.getElementById('sp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math at temperature 0', txt(pg, '#sp-mathbody')]
                rng(pg, '#sp-ic-T-in', 5)
                Lines += ['## Show the math at temperature 0.5', txt(pg, '#sp-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#sp-mathbody')
                pg.evaluate("()=>{document.getElementById('sp-math').open=false}")
                go(pg, 1); pg.click('#sp-rail [data-pred="one"]'); pg.wait_for_timeout(200); go(pg, 2); rng(pg, '#sp-ic-T-in', 0)
                Lines += ['## Part 3 after the right prediction (temperature 0)', txt(pg, '#sp-rail [data-obs]')]
            if sid == 'cut':
                pg.click('#sp-rail [data-lever="p9"]'); pg.wait_for_timeout(300)
                Lines += ['## Part 4 with top-p 0.9', txt(pg, '#sp-rail .step.current'), '## Stage', stage(pg)]
                pg.evaluate("()=>{document.getElementById('sp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 4, top-p 0.9)', txt(pg, '#sp-mathbody')]
                pg.evaluate("()=>{document.getElementById('sp-math').open=false}")
            if sid == 'remember':
                for k in ('temp', 'cut', 'tree', 'runs'):
                    pg.click(f'#sp-rail [data-rem="{k}"]'); pg.wait_for_timeout(350)
                    Lines += [f'## Part 6 after “Show in the chart” ({k})', txt(pg, '#sp-rail .step.current'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sp-app .bench')
            if sid == 'tree':
                rng(pg, '#sp-ic-u1-in', 99); Lines += ['## Result at 0.99', txt(pg, '#sp-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('sp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5)', txt(pg, '#sp-mathbody')]
                pg.evaluate("()=>{document.getElementById('sp-math').open=false}")
            Lines += [dock(pg)]
        pg.click('#sp-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#sp-rail')]
        for i in range(4):
            pg.click(f'#sp-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#sp-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#sp-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sp-stage') + [dock(pg)]
        pg.click('#sp-view button[data-view="b"]'); pg.wait_for_timeout(300)
        Lines += ['## Scenario 4 as a sentence tree', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sp-stage')
        pg.click('#sp-view button[data-view="a"]'); pg.wait_for_timeout(200)
        rng(pg, '#sp-ctl-P-in', 12)
        Lines += ['## After changing top-p under Settings (no scenario matches)', txt(pg, '#sp-insight')]
        pg.evaluate("()=>{const d=document.querySelector('#sp-controls .sp-zlist'); if(d) d.open=true}"); pg.wait_for_timeout(150)
        Lines += ['# Settings panel', txt(pg, '#sp-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#sp-controls')
        pg.click('#sp-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#sp-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#sp-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sp-app .bench') + [dock(pg)]
        seg = lambda k, v: (pg.click(f'#sp-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'), pg.wait_for_timeout(250))
        pg.click('#sp-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (goal missed: wrong words)', txt(pg, '#sp-ch-res')]
        seg('chT', '0.5'); Lines += ['## Temperature 0.5, no top-p (goal missed: too few words)', txt(pg, '#sp-ch-res')]
        pg.click('#sp-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “show all options”', txt(pg, '#sp-ch-res')]
        seg('chT', '0.7'); seg('chP', '0.9'); Lines += ['## Temperature 0.7, top-p 0.9 (met, two changes)', txt(pg, '#sp-ch-res')]
        seg('chT', '1'); seg('chP', '0.8'); Lines += ['## Temperature 1.0, top-p 0.8 (met, cuts too much)', txt(pg, '#sp-ch-res')]
        seg('chP', '0.9'); Lines += ['## Temperature 1.0, top-p 0.9 (met, the smallest change)', txt(pg, '#sp-ch-res'), 'Status: ' + txt(pg, '#sp-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('sp-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#sp-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sp-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#sp-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#sp-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#sp-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
