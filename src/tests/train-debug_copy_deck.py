"""Dump every piece of text a learner meets in the train-debug lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/train-debug_copy_deck.py  ->  out/train-debug-copy/en.md and out/train-debug-copy/zh.md

Order: home tile (the sentence for each bug, the notes), lab header and its popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), each option of parts 4 to 6, Show the math, the seven
scenarios, the settings panel, the challenge (every feedback tier and the candidates' table, every mystery run's answer), the card
"Can you use it on a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'train-debug-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/train-debug-deck.html', '--only', 'train-debug'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'train-debug-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'train-debug-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'train-debug-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#td-st-title'), txt(pg, '#td-st-sub'), 'Status: ' + txt(pg, '#td-status'),
             'Key under the region map: ' + txt(pg, '#td-reg .td-regkey')]
    if txt(pg, '#td-reg .td-regnote'): parts.append('Note under the region map: ' + txt(pg, '#td-reg .td-regnote'))
    for sel in ('#td-curve-cap', '#td-reg-cap', '#td-tbl'):
        t = txt(pg, sel)
        if t: parts.append(t)
    marks = pg.evaluate("()=>[...document.querySelectorAll('#td-curve text.td-mk, #td-curve text.td-lnl, #td-curve text.td-cl, #td-reg svg.td-regsvg text')].map(t=>t.textContent)")
    if marks: parts.append('Chart labels: ' + ' | '.join(marks))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#td-metrics .metric')].filter(m=>m.offsetParent).map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '.td-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const s=document.getElementById('td-stage'); window.scrollTo(0, s.getBoundingClientRect().bottom+scrollY+40)}"); pg.wait_for_timeout(500)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def click(pg, sel, ms=300):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(ms)


def go(pg, k):
    click(pg, f'#td-rail [data-go="{k}"]', 400)


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


def math(pg, title):
    pg.evaluate("()=>{document.getElementById('td-math').open=true}"); pg.wait_for_timeout(150)
    t = txt(pg, '#td-mathbody')
    pg.evaluate("()=>{document.getElementById('td-math').open=false}")
    return [f'## Show the math ({title})', t]


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        # home tile
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('.ds-btn[data-world="roots"], #domGrid [data-world="roots"]')].find(e=>e.offsetParent); if(g) g.click();}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-train-debug'):
            pg.evaluate("()=>document.querySelector('#tile-train-debug').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-train-debug')]
            for v in ('high', 'nozero', 'none'):
                click(pg, f'#tile-train-debug .seg-ctl button[data-v="{v}"]', 200)
                Lines += [f'## Tile sentence: {v}', txt(pg, '#tile-train-debug .tz-read')]
            Lines += ['## Tile tooltip', clean(pg.evaluate("()=>{const e=document.querySelector('#tile-train-debug .tz-vis'); return e?e.getAttribute('data-tipx')||'':''}"))]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-train-debug .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='train-debug'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__td.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#td-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#td-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#td-app .bench')
            if sid == 'mission': Lines += math(pg, 'part 1, the learning rate hidden')
            acts = {'predict': ['#td-rail [data-pred="nozero"]'], 'healthy': ['#td-rail .seg-ctl[data-key="steps"] button[data-v="30"]', '#td-rail .seg-ctl[data-key="steps"] button[data-v="300"]'],
                    'lr': ['#td-ic-lr [data-lever="10"]', '#td-ic-lr [data-lever="0.003"]'], 'bugs': ['#td-ic-bug [data-lever="nozero"]', '#td-ic-bug [data-lever="twice"]', '#td-ic-bug [data-lever="shuffle"]'],
                    'checks': ['#td-ic-chk [data-lever="c1"]', '#td-ic-chk [data-lever="c2"]', '#td-ic-chk [data-lever="c3"]']}.get(sid, [])
            for a in acts:
                click(pg, a, 350)
                Lines += [f'## Card after clicking {a.split("[")[-1].rstrip("]")}', txt(pg, '#td-rail .step.current'), '## Stage', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#td-app .bench')
            if sid == 'lr': Lines += math(pg, 'part 4, learning rate 0.003')
            if sid == 'bugs':
                go(pg, 1); click(pg, '#td-rail [data-pred="shuffle"]'); go(pg, 4); click(pg, '#td-ic-bug [data-lever="shuffle"]', 350)
                Lines += ['## Part 5 after the right prediction (labels shuffled)', txt(pg, '#td-rail [data-obs]')]
            Lines += [dock(pg)]
        click(pg, '#td-app .modes button[data-mode="sandbox"]:not([data-flow=free])', 400)
        Lines += ['# Scenarios', txt(pg, '#td-rail')]
        for i in range(pg.evaluate("()=>document.querySelectorAll('#td-rail [data-preset]').length")):
            click(pg, f'#td-rail [data-preset="{i}"]')
            name = clean(' '.join(pg.inner_text(f'#td-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#td-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#td-stage') + [dock(pg)]
        click(pg, '#td-ctl-bug button[data-v="twice"]')
        Lines += ['## After changing the bug under Settings (no scenario matches)', txt(pg, '#td-insight')]
        Lines += ['# Settings panel', txt(pg, '#td-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#td-controls')
        click(pg, '#td-app .modes button[data-mode="challenge"]', 400)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#td-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#td-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#td-app .bench') + [dock(pg)] + math(pg, 'challenge')
        click(pg, '#td-rail [data-ans="low"]'); click(pg, '#td-rail [data-act="submit"]')
        Lines += ['## Wrong: learning rate too low, no check opened', txt(pg, '#td-ch-res')]
        click(pg, '#td-rail [data-act="reveal"]')
        Lines += ['## After “Show all five candidates”', txt(pg, '#td-ch-res')]
        click(pg, '#td-rail [data-ans="twice"]')
        Lines += ['## Right but no check opened (a guess)', txt(pg, '#td-ch-res')]
        click(pg, '#td-rail [data-ck="c3"]')
        Lines += ['## Right with check 3 (the fewest)', txt(pg, '#td-ch-res'), '## Stage with check 3 open', stage(pg)]
        click(pg, '#td-rail [data-ck="c1"]'); click(pg, '#td-rail [data-ck="c2"]')
        Lines += ['## Right with all three checks', txt(pg, '#td-ch-res'), '## Stage with every check open', stage(pg)]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#td-app .bench')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        for m in range(1, 5):
            click(pg, '#td-rail [data-act="next-myst"]')
            k = pg.evaluate("()=>window.__td.state().myst")
            click(pg, f'#td-rail [data-ans="{k}"]'); click(pg, '#td-rail [data-ck="c3"]'); click(pg, '#td-rail [data-act="submit"]')
            Lines += [f'## Mystery run {m + 1} ({k}), right with check 3', txt(pg, '#td-ch-res'), '## Stage', stage(pg)]
        click(pg, '#td-app [data-act="drill-reveal"]')
        Lines += ['# Connect', txt(pg, '#td-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#td-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(Lines) + '\n')
        print(lang, len(Lines), 'blocks;', 'errors:', errs[:3])
        pg.close()
    b.close()
