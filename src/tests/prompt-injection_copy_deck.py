"""Dump every piece of text a learner meets in the prompt-injection lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/prompt-injection_copy_deck.py  ->  out/prompt-injection/copy/en.md and out/prompt-injection/copy/zh.md

Order: home tile (both states, the notes), lab header and its popovers, each guided part (card before and after its action, the stage,
the replay at the end, the tooltips in view not shown before, the mini result bar), Show the math, the scenarios, the settings panel,
the challenge (every feedback tier and the chart), the card “Can you use it on a new case?” (every question, a verdict for every
option), and Connect with the drill answer. A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'prompt-injection' / 'copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prompt-injection.html', '--only', 'prompt-injection'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'prompt-injection' / 'deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prompt-injection.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'prompt-injection' / 'deck.html')
clean = lambda s: s.replace('\u2011', '-').replace('\u00a0', ' ').replace('\u202f', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#pin-st-title'), txt(pg, '#pin-st-sub'), 'Status: ' + txt(pg, '#pin-status'), txt(pg, '#pin-tools'), txt(pg, '#pin-rp-cap'),
             '### What the model reads', txt(pg, '#pin-ctx'), 'Legend: ' + txt(pg, '#pin-ctx-leg'), '### What the agent does', txt(pg, '#pin-log'), 'Legend: ' + txt(pg, '#pin-log-leg'),
             txt(pg, '#pin-grid-cap'), 'Grid summary: ' + clean(pg.evaluate("()=>document.querySelector('#pin-grid').getAttribute('aria-label')||''")), 'Legend: ' + txt(pg, '#pin-grid-leg')]
    cells = pg.evaluate("()=>[...new Set([...document.querySelectorAll('#pin-grid .pin-c[data-tipx]')].map(c=>c.getAttribute('data-tipx').replace(/^[^,，]*[,，]/,'')))]")
    parts += ['Cell tooltips (distinct, task number dropped):'] + [f'- {clean(c)}' for c in cells]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#pin-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {clean(m)}' for m in mets] + ['Scope: ' + txt(pg, '.pin-scope')])   # the cards sit under the picture (R7.12)


class Tips:
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tipx]:not(.pin-c), {root} .t, {root} [data-jump], {root} .def'):
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
    pg.evaluate("()=>{const r=document.querySelector('#pin-rail [data-exits]')||document.querySelector('#pin-rail .step.current, #pin-rail .railcard'); if(r) window.scrollTo(0, r.getBoundingClientRect().top+scrollY-60)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#pin-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="agents"], #domStrip [data-world="agents"]')].find(e=>e.offsetParent);
          if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-prompt-injection'):
            pg.evaluate("()=>document.querySelector('#tile-prompt-injection').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-prompt-injection')]
            for v in ('1', '0'):
                click(pg, f'#tile-prompt-injection .seg-ctl button[data-v="{v}"]', 200)
                Lines += [f'## Tile with send_email = {v}', txt(pg, '#tile-prompt-injection .tz-vis'), txt(pg, '#tile-prompt-injection .tz-read'),
                          'Tooltip: ' + clean(pg.evaluate("()=>{const v=document.querySelector('#tile-prompt-injection .tz-pin-vis'); return v?v.getAttribute('data-tipx'):''}"))]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-prompt-injection .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='prompt-injection'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__pin.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#pin-rail .step.current'), '## Next is disabled because', txt(pg, '#pin-rail .nx-why'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#pin-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#pin-app .bench')
            if sid == 'mission':
                click(pg, '#pin-rail [data-pred="b"]', 900)
                Lines += ['## Card while the replay plays', txt(pg, '#pin-rail .step.current'), '## Stage while the replay plays', stage(pg)]
                click(pg, '#pin-player [data-act="end"]', 400)
                Lines += ['## Card after the replay (wrong prediction)', txt(pg, '#pin-rail .step.current'), '## Stage after the replay', stage(pg)]
                Lines += ['## Tooltips in view after the replay (new)'] + T.grab(pg, '#pin-app .bench')
                click(pg, '#pin-rail [data-pred="c"]', 300)
                Lines += ['## Result after the right prediction', txt(pg, '#pin-rail [data-obs]')]
                pg.evaluate("()=>{document.getElementById('pin-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 1)', txt(pg, '#pin-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#pin-mathbody')
                pg.evaluate("()=>{document.getElementById('pin-math').open=false}")
            elif sid == 'why':
                click(pg, '#pin-rail .seg-ctl[data-key="view"] button[data-v="text"]', 300)
                Lines += ['## Card after “as the model reads it”', txt(pg, '#pin-rail .step.current'), '## Stage', stage(pg)]
            elif sid == 'combo':
                for kk in ('send', 'files', 'inp'):
                    click(pg, f'#pin-rail .seg-ctl[data-key="{kk}"] button[data-v="0"]', 300)
                    Lines += [f'## Card with {kk} off', txt(pg, '#pin-rail [data-obs]'), '## Stage', stage(pg)]
                    click(pg, f'#pin-rail .seg-ctl[data-key="{kk}"] button[data-v="1"]', 300)
                click(pg, '#pin-rail .seg-ctl[data-key="send"] button[data-v="0"]', 200); click(pg, '#pin-rail .seg-ctl[data-key="files"] button[data-v="0"]', 300)
                Lines += ['## Card with two tools off', txt(pg, '#pin-rail [data-obs]')]
            elif sid == 'bound':
                for v in ('det', 'prm', 'scope', 'ask'):
                    click(pg, f'#pin-rail [data-lever="{v}"]')
                    Lines += [f'## Card after “{v}”', txt(pg, '#pin-rail .step.current'), '## Stage', stage(pg)]
                pg.evaluate("()=>{document.getElementById('pin-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (ask before every email)', txt(pg, '#pin-mathbody')]
                pg.evaluate("()=>{document.getElementById('pin-math').open=false}")
                click(pg, '#pin-ctl-prm button[data-v="1"]', 300)
                Lines += ['## Result after a change no option matches', txt(pg, '#pin-rail [data-obs]')]
            Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#pin-app .bench') + [dock(pg)]
        click(pg, '#pin-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        Lines += ['# Scenarios', txt(pg, '#pin-rail')]
        for i in range(4):
            click(pg, f'#pin-rail [data-preset="{i}"]')
            pg.evaluate("()=>document.querySelector('#pin-player [data-act=end]').click()"); pg.wait_for_timeout(250)
            name = clean(' '.join(pg.inner_text(f'#pin-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#pin-insight'), '## Stage (replay at the end)', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pin-stage') + [dock(pg)]
        click(pg, '#pin-ctl-det button[data-v="1"]', 300)
        Lines += ['## After a change under Settings (no scenario matches)', txt(pg, '#pin-insight'), '## Stage', stage(pg)]
        Lines += ['# Settings panel', txt(pg, '#pin-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#pin-controls')
        click(pg, '#pin-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#pin-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#pin-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pin-app .bench') + [dock(pg)]
        seg = lambda k, v: click(pg, f'#pin-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]', 250)
        click(pg, '#pin-rail [data-act="submit"]')
        Lines += ['## Submitted the initial setup (goal missed)', txt(pg, '#pin-ch-res')]
        seg('chD', 1); Lines += ['## Detector only (goal missed)', txt(pg, '#pin-ch-res')]
        seg('chD', 0); seg('chS', 0); Lines += ['## Email off (goal missed: tasks)', txt(pg, '#pin-ch-res')]
        seg('chS', 1); seg('chR', 'allow'); Lines += ['## Contacts only (goal missed: tasks)', txt(pg, '#pin-ch-res')]
        click(pg, '#pin-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#pin-ch-res')]
        seg('chR', 'askAll'); Lines += ['## Ask before every send (met, not the fewest asks)', txt(pg, '#pin-ch-res')]
        seg('chD', 1); seg('chR', 'allowAsk'); Lines += ['## Detector, contacts freely, ask for others (met, the fewest asks)', txt(pg, '#pin-ch-res'), 'Status: ' + txt(pg, '#pin-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('pin-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#pin-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pin-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#pin-app [data-act="drill-reveal"]', 200)
        Lines += ['# Connect', txt(pg, '#pin-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#pin-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(l for l in Lines if l is not None) + '\n')
        print(lang, len(Lines), 'blocks', 'errors:', len(errs))
        pg.close()
    b.close()
