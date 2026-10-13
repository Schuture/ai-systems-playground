"""Dump every piece of text a learner meets in the agent-capstone lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/agent-capstone_copy_deck.py  ->  out/agent-capstone/copy/en.md and out/agent-capstone/copy/zh.md

Order: home tile (both harnesses, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), Show the math, the scenarios, the settings panel, the
challenge (every feedback tier and the chart), the card “Can you use it on a new case?” (every question, a verdict for every option),
and Connect with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'agent-capstone' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/agent-capstone.html', '--only', 'agent-capstone'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'agent-capstone' / 'deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'agent-capstone.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'agent-capstone' / 'deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#acp-st-title'), txt(pg, '#acp-st-sub'), 'Status: ' + txt(pg, '#acp-status'), txt(pg, '#acp-score-cap'),
             'Score strip: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('#acp-score svg text')].map(t=>t.textContent).join(' | ')")),
             txt(pg, '#acp-grid-cap'), 'Kinds: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#acp-kinds .acp-kind')].map(s=>s.innerText.replace(/\\s+/g,' '))")),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#acp-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")),
             txt(pg, '#acp-tally'), txt(pg, '#acp-cause-cap'), 'Causes: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#acp-causes .acp-cz')].map(s=>s.innerText.replace(/\\s+/g,' '))"))]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#acp-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '.acp-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#acp-rail [data-exits]')||document.querySelector('#acp-rail .step.current, #acp-rail .railcard'); if(r) window.scrollTo(0, r.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#acp-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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


def endp(pg, ms=350):
    pg.evaluate("()=>{const b=document.querySelector('#acp-player [data-act=end]'); if(b) b.click()}"); pg.wait_for_timeout(ms)


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
        if pg.query_selector('#tile-agent-capstone'):
            pg.evaluate("()=>document.querySelector('#tile-agent-capstone').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-agent-capstone')]
            for v in ('new', 'old'):
                click(pg, f'#tile-agent-capstone .tz-acp-seg button[data-v="{v}"]', 200)
                Lines += [f'## Tile sentence ({v})', txt(pg, '#tile-agent-capstone .tz-read'),
                          '## Tile tooltip', clean(pg.evaluate("()=>{const v=document.querySelector('#tile-agent-capstone .tz-acp-vis'); return v?v.getAttribute('data-tipx'):''}"))]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-agent-capstone .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='agent-capstone'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__acp.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#acp-rail .step.current')]
            if sid != 'mission': endp(pg)
            Lines += ['## Stage', stage(pg), '## Settings panel', txt(pg, '#acp-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#acp-app .bench')
            if sid == 'mission':
                click(pg, '#acp-rail [data-pred="d"]', 1200)
                Lines += ['## Card while the runs play', txt(pg, '#acp-rail .step.current'), '## Stage while the runs play', stage(pg)]
                endp(pg, 500)
                Lines += ['## Card after the runs (wrong prediction)', txt(pg, '#acp-rail .step.current'), '## Stage after the runs', stage(pg)]
                Lines += ['## Tooltips in view after the runs (new)'] + T.grab(pg, '#acp-app .bench')
                click(pg, '#acp-rail [data-pred="a"]', 300)
                Lines += ['## Result after the right prediction', txt(pg, '#acp-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('acp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 1)', txt(pg, '#acp-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#acp-mathbody')
                pg.evaluate("()=>{document.getElementById('acp-math').open=false}")
            elif sid in ('loud', 'context', 'gate'):
                opts = {'loud': ('r2', 'r2c'), 'context': ('last', 'notes'), 'gate': ('detect', 'confirm')}[sid]
                for v in opts:
                    click(pg, f'#acp-rail [data-lever="{v}"]', 500)
                    Lines += [f'## Card while “{v}” runs', txt(pg, '#acp-rail [data-obs]')]
                    endp(pg)
                    Lines += [f'## Card after “{v}”', txt(pg, '#acp-rail .step.current'), '## Stage', stage(pg)]
                if sid == 'loud':
                    click(pg, '#acp-controls .seg-ctl[data-key="R"] button[data-v="1"]', 300); endp(pg)
                    Lines += ['## Result after changing the retries under Settings (no option matches)', txt(pg, '#acp-rail [data-obs]')]
            elif sid == 'prove':
                for v in (1, 5, 3):
                    click(pg, f'#acp-rail .seg-ctl[data-key="K"] button[data-v="{v}"]', 300); endp(pg)
                    Lines += [f'## Card after {v} runs per task', txt(pg, '#acp-rail [data-obs]'), '## Stage', stage(pg)]
                pg.evaluate("()=>{document.getElementById('acp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 5, 3 runs)', txt(pg, '#acp-mathbody')]
                pg.evaluate("()=>{document.getElementById('acp-math').open=false}")
            elif sid == 'remember':
                for r in ('one', 'det', 'k1'):
                    click(pg, f'#acp-rail [data-rem="{r}"]', 350); endp(pg)
                    Lines += [f'## Part 6 after “Show” ({r})', '## Stage', stage(pg)]
            Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#acp-app .bench') + [dock(pg)]
        click(pg, '#acp-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        Lines += ['# Scenarios', txt(pg, '#acp-rail')]
        for i in range(4):
            click(pg, f'#acp-rail [data-preset="{i}"]', 300)
            name = clean(' '.join(pg.inner_text(f'#acp-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name} (while the runs play)', txt(pg, '#acp-insight')]
            endp(pg)
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#acp-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#acp-stage') + [dock(pg)]
        click(pg, '#acp-controls .seg-ctl[data-key="K"] button[data-v="5"]', 300); endp(pg)
        Lines += ['## After changing the runs per task under Settings (no scenario matches)', txt(pg, '#acp-insight')]
        Lines += ['# Settings panel', txt(pg, '#acp-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#acp-controls')
        click(pg, '#acp-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#acp-rail .railcard')]
        endp(pg)
        Lines += ['## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#acp-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#acp-app .bench') + [dock(pg)]
        seg = lambda k, v: (click(pg, f'#acp-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250), endp(pg))
        click(pg, '#acp-rail [data-act="submit"]')
        Lines += ['## Submitted the initial setup (goal missed)', txt(pg, '#acp-ch-res')]
        seg('chG', 'detect'); Lines += ['## A detector (goal missed)', txt(pg, '#acp-ch-res')]
        seg('chG', 'confirm'); Lines += ['## A person confirms, keep all (goal missed)', txt(pg, '#acp-ch-res')]
        seg('chX', 'last'); Lines += ['## Last 8 steps (goal missed)', txt(pg, '#acp-ch-res')]
        seg('chX', 'notes'); Lines += ['## Trim + notes, no retries (goal missed)', txt(pg, '#acp-ch-res')]
        seg('chC', 1); seg('chR', 2); Lines += ['## The guided parts’ harness (over the budget)', txt(pg, '#acp-ch-res')]
        seg('chM', 'strong'); seg('chC', 0); Lines += ['## The stronger model, 2 retries, no check (over the budget)', txt(pg, '#acp-ch-res')]
        click(pg, '#acp-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#acp-ch-res')]
        seg('chM', 'cheap'); Lines += ['## Cheap, 2 retries, no check, trim + notes, a person confirms (met, not the cheapest)', txt(pg, '#acp-ch-res')]
        seg('chR', 1); Lines += ['## Cheap, 1 retry (met, the cheapest)', txt(pg, '#acp-ch-res'), 'Status: ' + txt(pg, '#acp-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('acp-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#acp-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#acp-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#acp-app [data-act="drill-reveal"]', 200)
        Lines += ['# Connect', txt(pg, '#acp-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#acp-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(l for l in Lines if l is not None) + '\n')
        print(lang, len(Lines), 'blocks', 'errors:', len(errs))
        pg.close()
    b.close()
