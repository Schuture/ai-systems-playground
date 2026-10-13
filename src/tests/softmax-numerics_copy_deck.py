"""Dump every piece of text a learner meets in the softmax-numerics lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/softmax-numerics_copy_deck.py  ->  out/softmax-numerics-copy/en.md and out/softmax-numerics-copy/zh.md

Order: home tile (sentence at a few largest logits, the notes), lab header and both popovers, each guided part (card before and after
its action, the stage, the tooltips in view not shown before, the mini result bar), part 3 at several logits and after the right
prediction, part 4 in each format, part 5 at a few gaps, part 6 block by block, Show the math, the scenarios (each, and −Inf), the
settings panel, the challenge (every feedback tier and the chart), the card "Can you use it on a new case?" (every question, a verdict
for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'softmax-numerics-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/softmax-numerics-deck.html', '--only', 'softmax-numerics'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'softmax-numerics-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'softmax-numerics-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'softmax-numerics-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#sm-st-title'), txt(pg, '#sm-st-sub'), 'Status: ' + txt(pg, '#sm-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#sm-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#sm-nl', '#sm-tab', '#sm-ce', '#sm-on'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#sm-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    ph = clean(pg.evaluate("()=>{const e=document.getElementById('sm-phone'); return e?e.textContent.trim():''}"))   # phones only: the part's control at the top of the chart
    return '\n'.join([p for p in parts if p] + (['Phones only, at the top of the chart: ' + ph] if ph else []) + ['Scope: ' + txt(pg, '.sm-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx], {root} [data-tip], {root} .t, {root} [data-jump]'):
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
    pg.evaluate("()=>{const r=document.querySelector('#sm-rail .step.current, #sm-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#sm-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#sm-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        if pg.query_selector('#tile-softmax-numerics'):
            pg.evaluate("()=>document.querySelector('#tile-softmax-numerics').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-softmax-numerics')]
            for v in (10, 11, 5, 50, 100):
                rng(pg, '#tile-softmax-numerics input[type=range]', v)
                Lines += [f'## Tile sentence at temperature {v / 100:.2f}', txt(pg, '#tile-softmax-numerics .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-softmax-numerics .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='softmax-numerics'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__sm.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#sm-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#sm-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#sm-app .bench')
            if sid in ('mission', 'predict'):
                pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
                Lines += [f'## Show the math ({sid})', txt(pg, '#sm-mathbody')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=false}")
            acted = True
            if sid == 'predict': pg.click('#sm-rail [data-pred="flat"]')
            elif sid == 'shift': rng(pg, '#sm-ic-top-in', 178)
            elif sid == 'fmt': pg.click('#sm-rail .seg-ctl[data-key="fmt"] button[data-v="fp16"]')
            elif sid == 'ce': rng(pg, '#sm-ic-gap-in', 103)
            elif sid == 'online': pg.click('#sm-rail [data-on="1"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#sm-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#sm-app .bench')
            if sid == 'shift':
                for v in (176, 120, -180, -200, -208):
                    rng(pg, '#sm-ic-top-in', v); Lines += [f'## Result at a largest logit of {v / 2}', txt(pg, '#sm-rail [data-obs]'), '## Stage', stage(pg)]
                pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math at −104', txt(pg, '#sm-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#sm-mathbody')
                pg.evaluate("()=>{document.getElementById('sm-math').open=false}")
                go(pg, 1); pg.click('#sm-rail [data-pred="same"]'); pg.wait_for_timeout(200); go(pg, 2); rng(pg, '#sm-ic-top-in', 120)
                Lines += ['## Part 3 after the right prediction (largest logit 60)', txt(pg, '#sm-rail [data-obs]')]
            if sid == 'fmt':
                pg.click('#sm-rail .seg-ctl[data-key="fmt"] button[data-v="bf16"]'); pg.wait_for_timeout(300)
                Lines += ['## Part 4 in BF16', txt(pg, '#sm-rail [data-obs]'), '## Stage', stage(pg)]
                pg.click('#sm-rail .seg-ctl[data-key="fmt"] button[data-v="fp32"]'); pg.wait_for_timeout(300)
                Lines += ['## Part 4 back in FP32', txt(pg, '#sm-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 4, FP32)', txt(pg, '#sm-mathbody')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=false}")
            if sid == 'ce':
                for v in (60, 0, 140):
                    rng(pg, '#sm-ic-gap-in', v); Lines += [f'## Result at a gap of {v}', txt(pg, '#sm-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5, gap 140)', txt(pg, '#sm-mathbody')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=false}")
            if sid == 'online':
                for k2 in range(2, 6):
                    pg.click('#sm-rail [data-on="1"]'); pg.wait_for_timeout(300)
                    Lines += [f'## After ▸ number {k2}', txt(pg, '#sm-rail [data-obs]')]
                Lines += ['## Stage at the end', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sm-app .bench')
                pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 6)', txt(pg, '#sm-mathbody')]
                pg.evaluate("()=>{document.getElementById('sm-math').open=false}")
            if sid == 'remember':
                for k2 in ('ce', 'on', 'shift'):
                    pg.click(f'#sm-rail [data-rem="{k2}"]'); pg.wait_for_timeout(350)
                    Lines += [f'## Part 7 after “Show in the chart” ({k2})', '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#sm-app .bench')
            Lines += [dock(pg)]
        pg.click('#sm-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#sm-rail')]
        for i in range(4):
            pg.click(f'#sm-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#sm-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#sm-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sm-stage') + [dock(pg)]
        pg.click('#sm-controls .seg-ctl[data-key="mv"] button[data-v="inf"]'); pg.wait_for_timeout(300)
        Lines += ['## Scenario 4 with −Inf', txt(pg, '#sm-insight'), '## Stage', stage(pg)]
        Lines += ['# Settings panel (scenario 4)', txt(pg, '#sm-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#sm-controls')
        pg.click('#sm-rail [data-preset="0"]'); pg.wait_for_timeout(300)
        rng(pg, '#sm-ctl-top-in', 100)
        Lines += ['## After changing the largest logit under Settings (no scenario matches)', txt(pg, '#sm-insight')]
        Lines += ['# Settings panel (sampler)', txt(pg, '#sm-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#sm-controls')
        pg.click('#sm-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#sm-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#sm-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sm-app .bench') + [dock(pg)]
        pg.click('#sm-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted the initial setup (goal missed: NaN)', txt(pg, '#sm-ch-res')]
        for o in ('bf16', 'half', 'clamp'):
            pg.click(f'#sm-rail [data-lever="{o}"]'); pg.wait_for_timeout(300)
            Lines += [f'## Fix {o} (goal missed)', txt(pg, '#sm-ch-res'), '## Stage', stage(pg)]
        pg.click('#sm-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “Show the answer”', txt(pg, '#sm-ch-res')]
        for o in ('fp32', 'both', 'max'):
            pg.click(f'#sm-rail [data-lever="{o}"]'); pg.wait_for_timeout(300)
            Lines += [f'## Fix {o} (goal met)', txt(pg, '#sm-ch-res')]
        Lines += ['Status: ' + txt(pg, '#sm-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('sm-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#sm-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#sm-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#sm-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#sm-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#sm-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
