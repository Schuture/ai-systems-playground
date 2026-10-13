"""Dump every piece of text a learner meets in the agent-reliability lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/agent-reliability_copy_deck.py  ->  out/agent-reliability/copy/en.md and out/agent-reliability/copy/zh.md

Order: home tile (sentence at a few step counts, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), Show the math, the scenarios, the settings panel, the
challenge (every feedback tier and the chart), the card “Can you use it on a new case?” (every question, a verdict for every option),
and Connect with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-reliability' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-reliability.html', '--only', 'agent-reliability'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'agent-reliability' / 'deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-reliability.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'agent-reliability' / 'deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#arl-st-title'), txt(pg, '#arl-st-sub'), 'Status: ' + txt(pg, '#arl-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#arl-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             txt(pg, '#arl-batch'), txt(pg, '#arl-tally'), txt(pg, '#arl-curve-cap')]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#arl-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '.arl-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#arl-rail [data-exits]')||document.querySelector('#arl-rail .step.current, #arl-rail .railcard'); if(r) window.scrollTo(0, r.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#arl-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        if pg.query_selector('#tile-agent-reliability'):
            pg.evaluate("()=>document.querySelector('#tile-agent-reliability').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-agent-reliability')]
            for v in (1, 10, 20, 40):
                rng(pg, '#tile-agent-reliability input[type=range]', v)
                Lines += [f'## Tile sentence at {v} steps', txt(pg, '#tile-agent-reliability .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-agent-reliability .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltip', clean(pg.evaluate("()=>{const v=document.querySelector('#tile-agent-reliability .tz-arl-vis'); return v?v.getAttribute('data-tipx'):''}"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='agent-reliability'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__arl.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#arl-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#arl-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#arl-app .bench')
            if sid == 'mission':
                click(pg, '#arl-rail [data-pred="b"]', 1200)
                Lines += ['## Card while the tasks run', txt(pg, '#arl-rail .step.current'), '## Stage while the tasks run', stage(pg)]
                click(pg, '#arl-player [data-act="end"]', 400)
                Lines += ['## Card after the runs (wrong prediction)', txt(pg, '#arl-rail .step.current'), '## Stage after the runs', stage(pg)]
                Lines += ['## Tooltips in view after the runs (new)'] + T.grab(pg, '#arl-app .bench')
                click(pg, '#arl-rail [data-pred="c"]', 300)
                Lines += ['## Result after the right prediction', txt(pg, '#arl-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('arl-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 1)', txt(pg, '#arl-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#arl-mathbody')
                pg.evaluate("()=>{document.getElementById('arl-math').open=false}")
            elif sid == 'chain':
                for v in ('half', 'steady'):
                    click(pg, f'#arl-rail [data-lever="{v}"]')
                    Lines += [f'## Card after “{v}”', txt(pg, '#arl-rail .step.current')]
                rng(pg, '#arl-ctl-n-in', 30)
                Lines += ['## Result after changing the steps under Settings (no option matches)', txt(pg, '#arl-rail [data-obs]')]
            elif sid == 'loud':
                click(pg, '#arl-rail .seg-ctl[data-key="R"] button[data-v="2"]')
                Lines += ['## Card after 2 retries', txt(pg, '#arl-rail .step.current'), '## Stage', stage(pg)]
                rng(pg, '#arl-rail input[data-key="r"]', 3)
                Lines += ['## Result at 30% loud', txt(pg, '#arl-rail [data-obs]')]
                rng(pg, '#arl-rail input[data-key="r"]', 10)
                Lines += ['## Result at 100% loud', txt(pg, '#arl-rail [data-obs]')]
                click(pg, '#arl-rail .seg-ctl[data-key="R"] button[data-v="0"]')
                Lines += ['## Result with no retries', txt(pg, '#arl-rail [data-obs]')]
            elif sid == 'check':
                for v in ('ck', 'ck0'):
                    click(pg, f'#arl-rail [data-lever="{v}"]')
                    Lines += [f'## Card after “{v}”', txt(pg, '#arl-rail .step.current')]
                click(pg, '#arl-rail [data-lever="ck"]')
                pg.evaluate("()=>{document.getElementById('arl-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (a check, 2 retries)', txt(pg, '#arl-mathbody')]
                pg.evaluate("()=>{document.getElementById('arl-math').open=false}")
            elif sid == 'retry':
                for v in ('blind', 'none'):
                    click(pg, f'#arl-rail [data-lever="{v}"]')
                    Lines += [f'## Card after “{v}”', txt(pg, '#arl-rail .step.current')]
                click(pg, '#arl-rail [data-lever="blind"]')
                pg.evaluate("()=>{document.getElementById('arl-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (same-call retries)', txt(pg, '#arl-mathbody')]
                pg.evaluate("()=>{document.getElementById('arl-math').open=false}")
            elif sid == 'remember':
                for r in ('chain', 'loud', 'check'):
                    click(pg, f'#arl-rail [data-rem="{r}"]', 350)
                    Lines += [f'## Part 6 after “Show in the chart” ({r})', txt(pg, '#arl-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#arl-app .bench') + [dock(pg)]
        click(pg, '#arl-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        Lines += ['# Scenarios', txt(pg, '#arl-rail')]
        for i in range(4):
            click(pg, f'#arl-rail [data-preset="{i}"]')
            name = clean(' '.join(pg.inner_text(f'#arl-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#arl-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#arl-stage') + [dock(pg)]
        rng(pg, '#arl-ctl-n-in', 12)
        Lines += ['## After changing the steps under Settings (no scenario matches)', txt(pg, '#arl-insight')]
        Lines += ['# Settings panel', txt(pg, '#arl-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#arl-controls')
        click(pg, '#arl-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#arl-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#arl-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#arl-app .bench') + [dock(pg)]
        seg = lambda k, v: click(pg, f'#arl-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250)
        click(pg, '#arl-rail [data-act="submit"]')
        Lines += ['## Submitted the initial setup (goal missed)', txt(pg, '#arl-ch-res')]
        seg('chR', 3); Lines += ['## 3 retries (goal missed: silent failures)', txt(pg, '#arl-ch-res')]
        seg('chC', 1); Lines += ['## A check and 3 retries, 20 steps (over the budget)', txt(pg, '#arl-ch-res')]
        seg('chR', 0); Lines += ['## A check, no retries', txt(pg, '#arl-ch-res')]
        seg('chC', 0); seg('chM', 1); seg('chR', 1); Lines += ['## The 98% model, 1 retry, 20 steps', txt(pg, '#arl-ch-res')]
        seg('chN', 12); Lines += ['## The 98% model, 1 retry, 12 steps', txt(pg, '#arl-ch-res')]
        click(pg, '#arl-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#arl-ch-res')]
        seg('chM', 0); seg('chC', 1); seg('chR', 2); Lines += ['## The 95% model, a check, 2 retries, 12 steps (met, not the cheapest)', txt(pg, '#arl-ch-res')]
        seg('chR', 1); Lines += ['## The 95% model, a check, 1 retry, 12 steps (met, the cheapest)', txt(pg, '#arl-ch-res'), 'Status: ' + txt(pg, '#arl-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('arl-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#arl-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#arl-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#arl-app [data-act="drill-reveal"]', 200)
        Lines += ['# Connect', txt(pg, '#arl-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#arl-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(l for l in Lines if l is not None) + '\n')
        print(lang, len(Lines), 'blocks', 'errors:', len(errs))
        pg.close()
    b.close()
