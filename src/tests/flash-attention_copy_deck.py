"""Dump every piece of text a learner meets in the flash-attention lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/flash-attention_copy_deck.py  ->  out/flash-attention-copy/en.md and out/flash-attention-copy/zh.md

Order: home tile (sentence at three lengths, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), Show the math, the scenarios (each, and the 12-token
example), the settings panel, the challenge (every feedback tier and the chart), the card "Can you use it on a new case?" (every
question, a verdict for every option), and Connect with the drill answer. A tooltip is printed the first time it appears."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'flash-attention-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/flash-attention-deck.html', '--only', 'flash-attention'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'flash-attention-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'flash-attention-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'flash-attention-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#fa-st-title'), txt(pg, '#fa-st-sub'), 'Status: ' + txt(pg, '#fa-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#fa-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#fa-hbm', '#fa-xfer', '#fa-chipm', '#fa-row', '#fa-cnt', '#fa-real-cap', '#fa-bars', '#fa-room', '#fa-curve .fa-cvcap', '#fa-gauge'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()))
    cl = pg.evaluate("()=>[...document.querySelectorAll('#fa-curve text.fa-cv-lab')].filter(t=>t.getBBox().width>0).map(t=>t.textContent)")
    if cl: parts.append('Curve names at the line ends: ' + ' | '.join(cl))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#fa-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '#fa-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#fa-rail .step.current, #fa-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120)}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#fa-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


def click(pg, sel):
    pg.evaluate(f"()=>document.querySelector('{sel}').click()"); pg.wait_for_timeout(300)


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


def math(pg, label):
    pg.evaluate("()=>{document.getElementById('fa-math').open=true}"); pg.wait_for_timeout(150)
    t = txt(pg, '#fa-mathbody')
    pg.evaluate("()=>{document.getElementById('fa-math').open=false}")
    return [f'## Show the math ({label})', t]


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ['en', 'zh']:
        pg = b.new_page(viewport={'width': 1440, 'height': 1000})
        errs = []; pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300); pg.click(f'.langsw button[data-lang="{lang}"]'); pg.wait_for_timeout(200)
        T = Tips(); Lines = []
        pg.evaluate("()=>{const g=document.querySelector('.world[data-world=\"models\"]'); if(g) g.dispatchEvent(new MouseEvent('click',{bubbles:true}))}"); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-flash-attention'):
            pg.evaluate("()=>document.querySelector('#tile-flash-attention').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-flash-attention')]
            for v in (0, 3, 7):
                rng(pg, '#tz-fa-n', v)
                Lines += [f'## Tile sentence at step {v}', txt(pg, '#tile-flash-attention .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-flash-attention .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-flash-attention')
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='flash-attention'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__fa.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#fa-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#fa-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#fa-app .bench')
            acted = True
            if sid == 'predict':
                click(pg, '#fa-rail [data-pred="exp"]'); Lines += ['## Card after a pick', txt(pg, '#fa-rail .step.current')]
            elif sid in ('std', 'fa'):
                click(pg, '#fa-rail [data-act="fwd"]'); Lines += ['## After one step', txt(pg, '#fa-rail [data-obs]'), '## Stage', stage(pg)]
                if sid == 'fa':
                    for _ in range(2): click(pg, '#fa-rail [data-act="fwd"]')
                    Lines += ['## After three tiles', txt(pg, '#fa-rail [data-obs]'), '## Stage', stage(pg)] + math(pg, 'tile 3')
                click(pg, '#fa-rail [data-nav="showme"]')
            elif sid == 'len': rng(pg, '#fa-ic-N input', 7)
            elif sid == 'tile':
                for v in (32, 64):
                    click(pg, f'#fa-rail [data-lever="{v}"]'); Lines += [f'## Tiles of {v}', txt(pg, '#fa-rail [data-obs]')]
                click(pg, '#fa-rail [data-lever="256"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#fa-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#fa-app .bench')
            if sid == 'std':
                Lines += math(pg, 'part 3, end')
                go(pg, 1); click(pg, '#fa-rail [data-pred="move"]'); go(pg, 2)
                Lines += ['## Part 3 after the right prediction', txt(pg, '#fa-rail [data-obs]')]
            if sid in ('len', 'tile'): Lines += math(pg, f'part {k + 1}')
            Lines += [dock(pg)]
        click(pg, '#fa-app .modes button[data-mode="sandbox"]:not([data-flow=free])')
        Lines += ['# Scenarios', txt(pg, '#fa-rail')]
        for i in range(4):
            click(pg, f'#fa-rail [data-preset="{i}"]')
            name = clean(' '.join(pg.inner_text(f'#fa-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#fa-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#fa-stage') + [dock(pg)]
        click(pg, '#fa-ctl-Br button[data-v="64"]')
        Lines += ['## After changing the Q tile under Settings (no scenario matches)', txt(pg, '#fa-insight')]
        click(pg, '#fa-view button[data-view="toy"]'); click(pg, '#fa-pathsw button[data-path="fa"]')
        Lines += ['## The 12-token example in Scenarios', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#fa-stage')
        click(pg, '#fa-view button[data-view="real"]')
        Lines += ['# Settings panel', txt(pg, '#fa-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#fa-controls')
        click(pg, '#fa-app .modes button[data-mode="challenge"]')
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#fa-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#fa-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#fa-app .bench') + [dock(pg)]
        seg = lambda k, v: click(pg, f'#fa-rail .seg-ctl[data-key="{k}"] button[data-v="{v}"]')
        click(pg, '#fa-rail [data-act="submit"]')
        Lines += ['## Submitted the initial setup (too much traffic)', txt(pg, '#fa-ch-res')]
        seg('chBc', '128'); seg('chBr', '256'); Lines += ['## 256 × 128 (does not fit)', txt(pg, '#fa-ch-res')]
        click(pg, '#fa-rail [data-act="reveal"]')
        Lines += ['## After “Show the answer”', txt(pg, '#fa-ch-res')]
        seg('chBc', '32'); Lines += ['## 256 × 32 (met, two changes)', txt(pg, '#fa-ch-res')]
        seg('chBc', '64'); Lines += ['## 256 × 64 (met, the smallest change)', txt(pg, '#fa-ch-res'), 'Status: ' + txt(pg, '#fa-status'), '## Stage', stage(pg), dock(pg)]
        Lines += math(pg, 'challenge')
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#fa-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        click(pg, '#fa-app [data-act="drill-reveal"]')
        Lines += ['# Connect', txt(pg, '#fa-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#fa-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x is not None) + '\n')
        print(lang, len(Lines), 'blocks', 'errors:', len(errs))
        pg.close()
    b.close()
