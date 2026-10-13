"""Dump every piece of text a learner meets in the optimizers lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/optimizers_copy_deck.py  ->  out/optimizers-copy/en.md and out/optimizers-copy/zh.md

Order: home tile (the sentence, legend and picture description for SGD and for Adam, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), part 3 at 0.08 and 0.10 and the verdict for the right
pick, Show the math, the scenarios (each), the settings panel, the challenge (every feedback tier and the chart), the card "Can you use
it on a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text. The page runs with reduced motion, so the stage shows each run's end
state (the player still reads ▶ Replay); the folded “In numbers” boxes are printed opened."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'optimizers-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/optimizers-deck.html', '--only', 'optimizers'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'optimizers-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'optimizers-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'optimizers-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    pg.evaluate("s=>{const e=document.querySelector(s); if(e) e.querySelectorAll('details.op-more').forEach(d=>d.open=true)}", sel)
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


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


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#op-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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



def stage(pg):
    parts = [txt(pg, '#op-st-title'), txt(pg, '#op-st-sub'), 'Status: ' + txt(pg, '#op-status'),
             'Names on the charts: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#op-map .op-pl, #op-loss .op-ll, #op-loss .op-gl')].map(s=>s.textContent)"))]
    for sel in ('#op-map-cap', '#op-player', '#op-bars-cap', '#op-bars', '#op-loss-cap', '#op-wd'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()))
    svg = pg.evaluate("()=>[...document.querySelectorAll('#op-map text, #op-loss text')].map(t=>t.textContent).join(' | ')")
    mets = pg.evaluate("()=>[...document.querySelectorAll('#op-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Chart labels: ' + clean(svg), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


def dock(pg):
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const g=document.getElementById('op-graph'); window.scrollTo(0, g.getBoundingClientRect().bottom+scrollY+60)}"); pg.wait_for_timeout(450)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#op-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


def math(pg, title):
    pg.evaluate("()=>{document.getElementById('op-math').open=true}"); pg.wait_for_timeout(150)
    t = txt(pg, '#op-mathbody')
    pg.evaluate("()=>{document.getElementById('op-math').open=false}")
    return [f'## Show the math ({title})', t]


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="roots"], #domStrip [data-world="roots"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="roots"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-optimizers .tz-op'):
            pg.evaluate("()=>document.querySelector('#tile-optimizers').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-optimizers')]
            for o in pg.evaluate("()=>window.__tzOp.opts"):   # SGD / Adam: the sentence, the legend and the picture's description for each pick
                pg.click(f'#tile-optimizers .seg-ctl button[data-v="{o}"]'); pg.wait_for_timeout(250)
                Lines += [f'## Tile with {clean(pg.inner_text(f"#tile-optimizers .seg-ctl button[data-v={o}]"))} picked', txt(pg, '#tile-optimizers .tz-read'),
                          'Legend: ' + ' | '.join(clean(' '.join(x.split())) for x in pg.evaluate("()=>[...document.querySelectorAll('#tile-optimizers .tz-leg > span')].map(s=>s.innerText)")),
                          'Picture: ' + clean(pg.get_attribute('#tile-optimizers svg.tz-op-vis', 'aria-label') or '')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-optimizers .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-optimizers')
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='optimizers'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__op.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#op-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#op-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#op-app .bench')
            acted = True
            if sid == 'predict': pg.click('#op-rail [data-pred="fewer"]')
            elif sid == 'sgd': rng(pg, '#op-ic-lr-in', 11)
            elif sid == 'mom': rng(pg, '#op-ic-beta-in', 12)
            elif sid == 'adam': rng(pg, '#op-ic-lrA-in', 5)
            elif sid == 'bias': pg.click('#op-rail [data-lever="off"]')
            elif sid == 'adamw': pg.click('#op-rail [data-lever="adamw"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350); pg.evaluate("()=>document.querySelectorAll('#op-rail details.op-more').forEach(d=>d.open=true)")
                Lines += ['## Card after the action', txt(pg, '#op-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#op-app .bench')
            if sid == 'sgd':
                for v, name in ((1, '0.01'), (15, '0.08'), (19, '0.10')):
                    rng(pg, '#op-ic-lr-in', v); Lines += [f'## Result at learning rate {name}', txt(pg, '#op-rail [data-obs]')]
                Lines += ['## Stage at 0.10', stage(pg)] + math(pg, 'part 3, 0.10')
                go(pg, 1); pg.click('#op-rail [data-pred="never"]'); pg.wait_for_timeout(200); go(pg, 2); rng(pg, '#op-ic-lr-in', 19)
                Lines += ['## Part 3 after the right prediction (0.10)', txt(pg, '#op-rail [data-obs]')]
            if sid == 'mom':
                for v in (0, 1, 13, 15, 18, 19):   # β 0, 0.05 (the swing is not slower), 0.65 (the best β), 0.75 (past it), 0.9, 0.95
                    rng(pg, '#op-ic-beta-in', v); Lines += [f'## Result at β index {v}', txt(pg, '#op-rail [data-obs]')]
            if sid == 'adam':
                for v, name in ((3, '0.3'), (4, '0.5'), (0, '0.05')):
                    rng(pg, '#op-ic-lrA-in', v); Lines += [f'## Result at learning rate {name}', txt(pg, '#op-rail [data-obs]')]
                pg.evaluate("()=>{const i=document.getElementById('op-k'); i.value=40; i.dispatchEvent(new Event('input',{bubbles:true}))}"); pg.wait_for_timeout(200)
                Lines += ['## Stage at step 40', stage(pg)] + math(pg, 'part 5, step 40')
            if sid == 'bias': Lines += math(pg, 'part 6')
            if sid == 'adamw':
                pg.click('#op-rail [data-lever="l2"]'); pg.wait_for_timeout(250)
                Lines += ['## Result with Adam + L2 picked again', txt(pg, '#op-rail [data-obs]'), '## Stage', stage(pg)] + math(pg, 'part 7')
            if sid == 'predict': Lines += math(pg, 'part 2, before the answer is shown')
            Lines += [dock(pg)]
        pg.click('#op-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#op-rail')]
        for i in range(4):
            pg.click(f'#op-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#op-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#op-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#op-stage') + [dock(pg)]
        pg.click('#op-ctl-opt [data-v="sgd"]'); pg.wait_for_timeout(250)
        Lines += ['## After changing the optimizer under Settings (no scenario matches)', txt(pg, '#op-insight')]
        Lines += math(pg, 'Scenarios, SGD on the curved valley')
        Lines += ['# Settings panel', txt(pg, '#op-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#op-controls')
        pg.click('#op-ctl-opt [data-v="adamw"]'); pg.wait_for_timeout(250)
        Lines += ['## Settings panel with AdamW', txt(pg, '#op-controls')]
        pg.click('#op-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#op-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#op-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#op-app .bench') + [dock(pg)]
        seg = lambda k, v: (pg.click(f'#op-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]'), pg.wait_for_timeout(250))
        pg.click('#op-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (goal missed)', txt(pg, '#op-ch-res')]
        seg('chL', '0.03'); Lines += ['## SGD 0.03 (diverges)', txt(pg, '#op-ch-res')]
        pg.click('#op-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “Show all options”', txt(pg, '#op-ch-res')]
        seg('chO', 'adam'); seg('chL', '0.3'); Lines += ['## Adam 0.3 (met, two changes)', txt(pg, '#op-ch-res')]
        seg('chO', 'mom'); seg('chL', '0.01'); Lines += ['## Momentum 0.9 at 0.01 (met, the smallest change)', txt(pg, '#op-ch-res'), 'Status: ' + txt(pg, '#op-status'), '## Stage', stage(pg), dock(pg)]
        Lines += math(pg, 'challenge')
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#op-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#op-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#op-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#op-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
