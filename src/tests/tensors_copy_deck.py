"""Dump every piece of text a learner meets in the tensors lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/tensors_copy_deck.py  ->  out/tensors-copy/en.md and out/tensors-copy/zh.md

Order: home tile (each pair of shapes, the notes), lab header and both popovers, each guided part (card before and after its action,
the stage, the tooltips in view not shown before, the mini result bar), part 3 at an error and at a missing axis, part 4 with each
string, part 5 at D 1,024, part 6 with each "Show in the chart", Show the math, the scenarios (each), the settings panel, the challenge
(every feedback tier and the chart), the card "Can you use it on a new case?" (every question, a verdict for every option), and Connect
with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'tensors-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/tensors-deck.html', '--only', 'tensors'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'tensors-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'tensors-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'tensors-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#ts-st-title'), txt(pg, '#ts-st-sub'), 'Status: ' + txt(pg, '#ts-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#ts-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#ts-bc', '#ts-es', '#ts-nn'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#ts-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    ph = clean(pg.evaluate("()=>{const e=document.getElementById('ts-phone'); return e?e.textContent.trim():''}"))   # phones only: the part's control or jump at the top of the chart
    return '\n'.join([p for p in parts if p] + (['Phones only, at the top of the chart: ' + ph] if ph else []) + ['Scope: ' + txt(pg, '.ts-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#ts-rail .step.current, #ts-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#ts-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#ts-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(300)


def math(pg, title):
    pg.evaluate("()=>{document.getElementById('ts-math').open=true}"); pg.wait_for_timeout(150)
    t = txt(pg, '#ts-mathbody'); pg.evaluate("()=>{document.getElementById('ts-math').open=false}")
    return [f'## Show the math ({title})', t]


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
        if pg.query_selector('#tile-tensors'):
            pg.evaluate("()=>document.querySelector('#tile-tensors').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-tensors')]
            for v in ('p2', 'p3', 'p4', 'p1'):
                click(pg, f'#tile-tensors .seg-ctl button[data-v="{v}"]')
                Lines += [f'## Tile at {v}', txt(pg, '#tile-tensors .tz-tensors-vis'), txt(pg, '#tile-tensors .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-tensors .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='tensors'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__ts.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#ts-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#ts-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#ts-app .bench')
            acted = True
            if sid == 'predict': click(pg, '#ts-rail [data-pred="err"]')
            elif sid == 'bcast': click(pg, '#ts-ic-a button[data-v="4x3"]'); click(pg, '#ts-ic-b button[data-v="4"]')
            elif sid == 'einsum': click(pg, '#ts-ic-es [data-lever="sc"]')
            elif sid == 'nn': rng(pg, '#ts-ic-D-in', 4)
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#ts-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#ts-app .bench')
            if sid == 'bcast':
                Lines += math(pg, 'part 3, (4, 3) + (4,)')
                for a, bb in (('4x3', '3'), ('2x3', '2x1'), ('3x1', '4')):
                    click(pg, f'#ts-ic-a button[data-v="{a}"]'); click(pg, f'#ts-ic-b button[data-v="{bb}"]')
                    Lines += [f'## Result at {a} + {bb}', txt(pg, '#ts-rail [data-obs]'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#ts-stage')
                Lines += math(pg, 'part 3, (3, 1) + (4,)')
            if sid == 'einsum':
                for e in ('ov', 'bm', 'mm'):
                    click(pg, f'#ts-ic-es [data-lever="{e}"]')
                    Lines += [f'## Result at {e}', txt(pg, '#ts-rail [data-obs]'), '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#ts-stage')
                Lines += math(pg, 'part 4, matrix multiply')
            if sid == 'nn':
                rng(pg, '#ts-ic-D-in', 0); Lines += ['## Result at D 50', txt(pg, '#ts-rail [data-obs]')]
                Lines += math(pg, 'part 5, D 50') + ['## Math tooltips (new)'] + T.grab(pg, '#ts-mathbody')
            if sid == 'remember':
                for r in ('es', 'nn', 'bc'):
                    click(pg, f'#ts-rail [data-rem="{r}"]')
                    Lines += [f'## Part 6 after “Show in the chart” ({r})', '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#ts-app .bench')
            Lines += [dock(pg)]
        pg.click('#ts-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#ts-rail')]
        for i in range(4):
            pg.click(f'#ts-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#ts-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#ts-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ts-stage') + [dock(pg)]
        rng(pg, '#ts-ctl-D-in', 5)
        Lines += ['## After changing D under Settings (no scenario matches)', txt(pg, '#ts-insight')]
        Lines += ['# Settings panel', txt(pg, '#ts-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#ts-controls')
        pg.click('#ts-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#ts-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#ts-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ts-app .bench') + [dock(pg)]
        pg.click('#ts-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (goal missed: the loop)', txt(pg, '#ts-ch-res')]
        for o, what in (('bc', 'goal missed: memory'), ('c1', 'goal missed: the loop'), ('c256', 'goal missed: memory')):
            click(pg, f'#ts-ic-ch [data-lever="{o}"]'); Lines += [f'## {o} ({what})', txt(pg, '#ts-ch-res')]
        click(pg, '#ts-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#ts-ch-res')]
        click(pg, '#ts-ic-ch [data-lever="c32"]'); Lines += ['## Chunks of 32 (met, not the smallest)', txt(pg, '#ts-ch-res')]
        click(pg, '#ts-ic-ch [data-lever="mm"]'); Lines += ['## Matmul expansion (met, the smallest)', txt(pg, '#ts-ch-res'), 'Status: ' + txt(pg, '#ts-status'), '## Stage', stage(pg), dock(pg)]
        Lines += math(pg, 'challenge, matmul expansion')
        click(pg, '#ts-ic-ch [data-lever="c32"]'); Lines += math(pg, 'challenge, chunks of 32')
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ts-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#ts-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#ts-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#ts-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
