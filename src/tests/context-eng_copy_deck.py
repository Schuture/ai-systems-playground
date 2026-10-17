"""Dump every piece of text a learner meets in the context-eng lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).
python3 tests/context-eng_copy_deck.py  ->  out/context-eng/copy/en.md and out/context-eng/copy/zh.md
Order: home tile (the sentence for each choice, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), Show the math, the scenarios, the settings panel, the
challenge (every feedback tier and the chart), the card “Can you use it on a new case?” (every question, a verdict for every option),
and Connect with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright
P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'context-eng' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/context-eng.html', '--only', 'context-eng'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'context-eng' / 'deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'context-eng.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'context-eng' / 'deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#cte-st-title'), txt(pg, '#cte-st-sub'), 'Status: ' + txt(pg, '#cte-status'), txt(pg, '#cte-cap'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#cte-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             txt(pg, '#cte-lanes-h'),
             'Legend of the rows: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#cte-legend2 > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             txt(pg, '#cte-tally')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#cte-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '.cte-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx], {root} .t, {root} [data-jump], {root} .def'):
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
    pg.evaluate("()=>{const r=document.querySelector('#cte-rail [data-exits]')||document.querySelector('#cte-rail .step.current, #cte-rail .railcard'); if(r) window.scrollTo(0, r.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#cte-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


def click(pg, sel, ms=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(ms)


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


def end(pg): click(pg, '#cte-player [data-act="end"]', 300)
def lever(pg, v): click(pg, f'#cte-rail [data-lever="{v}"]', 300); end(pg)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        # home tile
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="agents"], #domStrip [data-world="agents"]')].find(e=>e.offsetParent);
          if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-context-eng'):
            pg.evaluate("()=>document.querySelector('#tile-context-eng').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-context-eng')]
            for v in ('all', 'last', 'notes'):
                click(pg, f'#tile-context-eng .tz-cte-seg button[data-v="{v}"]', 200)
                Lines += [f'## Tile with “{v}”', txt(pg, '#tile-context-eng .tz-read'), clean(pg.evaluate("()=>{const v=document.querySelector('#tile-context-eng .tz-cte-vis'); return v?v.innerText+' / tooltip: '+v.getAttribute('data-tipx'):''}"))]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-context-eng .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='context-eng'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__cte.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#cte-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#cte-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#cte-app .bench')
            if sid == 'mission':
                click(pg, '#cte-rail [data-pred="b"]', 1200)
                Lines += ['## Card while the run plays', txt(pg, '#cte-rail .step.current'), '## Stage while the run plays', stage(pg)]
                end(pg)
                Lines += ['## Card after the run (wrong prediction)', txt(pg, '#cte-rail .step.current'), '## Stage after the run', stage(pg)]
                Lines += ['## Tooltips in view after the run (new)'] + T.grab(pg, '#cte-app .bench')
                click(pg, '#cte-rail [data-pred="c"]', 300)
                Lines += ['## Result after the right prediction', txt(pg, '#cte-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('cte-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 1)', txt(pg, '#cte-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#cte-mathbody')
                pg.evaluate("()=>{document.getElementById('cte-math').open=false}")
            elif sid == 'fills':
                for v in ('head', 'conv', 'tool'):
                    click(pg, f'#cte-rail [data-lever="{v}"]')
                    Lines += [f'## Card after “{v}”', txt(pg, '#cte-rail .step.current')]
            elif sid == 'trim':
                for v in ('trim', 'slide'):
                    lever(pg, v); Lines += [f'## Card after “{v}”', txt(pg, '#cte-rail .step.current'), '## Stage', stage(pg)]
                click(pg, '#cte-controls .seg-ctl[data-key="N"] button[data-v="20"]', 300); end(pg)
                Lines += ['## Result after the last 20 steps under Settings (no option matches)', txt(pg, '#cte-rail [data-obs]')]
            elif sid == 'notes':
                for v in ('sum', 'notes'):
                    lever(pg, v); Lines += [f'## Card after “{v}”', txt(pg, '#cte-rail .step.current')]
                lever(pg, 'sum')
                pg.evaluate("()=>{document.getElementById('cte-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (a summary every 20 steps)', txt(pg, '#cte-mathbody')]
                pg.evaluate("()=>{document.getElementById('cte-math').open=false}")
            elif sid == 'sub':
                lever(pg, 'sub'); Lines += ['## Card after “sub”', txt(pg, '#cte-rail .step.current'), '## Stage', stage(pg)]
            elif sid == 'prod':
                for r in ('slide', 'comp', 'sum', 'subx'):
                    click(pg, f'#cte-rail [data-rem="{r}"]', 350); end(pg)
                    Lines += [f'## Part {k + 1} after “Show in the chart” ({r})', '## Stage', stage(pg)]
                Lines += ['## Source links (page-end sources since polish round 3)', '\n'.join(pg.evaluate("()=>[...document.querySelectorAll('[data-endnotes] a')].map(a=>a.textContent+' → '+a.href)"))]
            elif sid == 'remember':
                for r in ('trim', 'slide', 'sum'):
                    click(pg, f'#cte-rail [data-rem="{r}"]', 350); end(pg)
                    Lines += [f'## Part {k + 1} after “Show in the chart” ({r})', txt(pg, '#cte-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#cte-app .bench') + [dock(pg)]
        click(pg, '#cte-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        Lines += ['# Scenarios', txt(pg, '#cte-rail')]
        for i in range(4):
            click(pg, f'#cte-rail [data-preset="{i}"]'); end(pg)
            name = clean(' '.join(pg.inner_text(f'#cte-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#cte-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#cte-stage') + [dock(pg)]
        click(pg, '#cte-controls .seg-ctl[data-key="W"] button[data-v="262144"]', 300)
        Lines += ['## After changing the window under Settings (no scenario matches)', txt(pg, '#cte-insight')]
        Lines += ['# Settings panel', txt(pg, '#cte-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#cte-controls')
        click(pg, '#cte-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#cte-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#cte-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#cte-app .bench') + [dock(pg)]
        seg = lambda k, v: click(pg, f'#cte-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250)
        click(pg, '#cte-rail [data-act="submit"]')
        Lines += ['## Submitted the initial setup (goal missed: full at step 45)', txt(pg, '#cte-ch-res')]
        seg('chT', 1); Lines += ['## Trim (goal missed: the reload line is cut)', txt(pg, '#cte-ch-res')]
        seg('chT', 0); seg('chH', 2); Lines += ['## The last 10 steps (goal missed: facts lost)', txt(pg, '#cte-ch-res')]
        seg('chN', 1); Lines += ['## The last 10 steps and notes (goal missed: fact ③)', txt(pg, '#cte-ch-res')]
        seg('chH', 6); Lines += ['## A summary every 20 steps and notes (goal missed: fact ③ only likely)', txt(pg, '#cte-ch-res')]
        click(pg, '#cte-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#cte-ch-res')]
        seg('chH', 4); Lines += ['## The last 30 steps and notes (met, not the cheapest)', txt(pg, '#cte-ch-res')]
        seg('chH', 3); seg('chS', 1); Lines += ['## The last 20 steps, notes and a sub-agent (met, the cheapest)', txt(pg, '#cte-ch-res'), 'Status: ' + txt(pg, '#cte-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('cte-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#cte-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#cte-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#cte-app [data-act="drill-reveal"]', 200)
        Lines += ['# Connect', txt(pg, '#cte-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#cte-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(l for l in Lines if l is not None) + '\n')
        print(lang, len(Lines), 'blocks', 'errors:', len(errs))
        pg.close()
    b.close()
