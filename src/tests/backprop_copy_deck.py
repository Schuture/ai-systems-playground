"""Dump every piece of text a learner meets in the backprop lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/backprop_copy_deck.py  ->  out/backprop-copy/en.md and out/backprop-copy/zh.md

Order: home tile (sentence at a few temperatures, the notes), lab header and both popovers, each guided part (card before and after its
action, the stage, the tooltips in view not shown before, the mini result bar), the reveal for the right pick, part 4 with each rule,
part 5 at a rare word, Show the math, the scenarios (each, and the tree view), the settings panel, the challenge (every feedback tier and
the chart), the card "Can you use it on a new case?" (every question, a verdict for every option), and Connect with the drill answer.
A tooltip is printed the first time it appears with a given text."""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'backprop-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/backprop-deck.html', '--only', 'backprop'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'backprop-deck-wrapped.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'backprop-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'backprop-deck-wrapped.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    parts = [txt(pg, '#bp-st-title'), txt(pg, '#bp-st-sub'), 'Status: ' + txt(pg, '#bp-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#bp-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    for sel in ('#bp-err', '#bp-ck-cap', '#bp-cknum', '#bp-chips', '#bp-read'):
        t = txt(pg, sel)
        if t: parts.append(' '.join(t.split()))
    # the formula box beside the graph (review round 3), its MathML written out on one line
    fx = pg.evaluate("""()=>{const b=document.getElementById('bp-fx'); if(!b||b.hidden) return [];
      const t=n=>n.nodeType===3?n.textContent:n.nodeType!==1?'':n.localName==='mo'?' '+n.textContent+' ':n.localName==='mfrac'?t(n.children[0])+'/'+t(n.children[1])
        :n.localName==='msup'?t(n.children[0])+'^'+t(n.children[1]):n.localName==='msub'?t(n.children[0])+'_'+t(n.children[1]):[...n.childNodes].map(t).join('');
      return [...b.children].map(e=>t(e).replace(/\\s+/g,' ').trim()).filter(Boolean)}""")
    if fx: parts.append('Formulas: ' + ' | '.join(clean(x) for x in fx))
    parts.append('Graph: ' + ' / '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#bp-svg .bp-n')].map(g=>[...g.querySelectorAll('text')].map(t=>t.textContent).join(', '))")))
    extra = [clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#bp-svg .bp-tag, #bp-svg .bp-fant, #bp-svg .bp-cl text')].map(t=>t.textContent)")]
    if extra: parts.append('Graph labels: ' + ' | '.join(extra))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#bp-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Scope: ' + txt(pg, '.bp-scope'), 'Metrics:'] + [f'- {clean(m)}' for m in mets])


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
    pg.evaluate("()=>{const r=document.querySelector('#bp-rail .step.current, #bp-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#bp-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#bp-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(400)


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
        if pg.query_selector('#tile-backprop'):
            pg.evaluate("()=>document.querySelector('#tile-backprop').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-backprop')]
            for v in (-20, 0, 10, 15, 20):
                rng(pg, '#tile-backprop input[type=range]', v)
                Lines += [f'## Tile sentence at x = {v / 10}', txt(pg, '#tile-backprop .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-backprop .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-backprop')
        else:
            Lines += ['# Home tile', '(tile not found)']
        pg.evaluate("()=>{location.hash='backprop'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')] + ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        ids = pg.evaluate("()=>window.__bp.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#bp-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Settings panel', txt(pg, '#bp-controls'), '## Tooltips in view (new)'] + T.grab(pg, '#bp-app .bench')
            acted = True
            if sid == 'predict': pg.click('#bp-rail [data-pred="few"]')
            elif sid == 'walk':
                for _ in range(3): pg.click('#bp-ic-play [data-play="fwd"]'); pg.wait_for_timeout(120)
                Lines += ['## Stage after 3 steps', stage(pg), '## Tooltips in view after 3 steps (new)'] + T.grab(pg, '#bp-stage')
                for _ in range(3): pg.click('#bp-ic-play [data-play="fwd"]'); pg.wait_for_timeout(120)
            elif sid == 'check': pg.click('#bp-ic-h button[data-v="3"]')
            elif sid == 'fan': pg.click('#bp-rail [data-lever="over"]')
            elif sid == 'inplace': pg.click('#bp-rail [data-lever="pt"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350)
                Lines += ['## Card after the action', txt(pg, '#bp-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#bp-app .bench')
            if sid == 'walk':
                pg.evaluate("()=>{document.getElementById('bp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 3)', txt(pg, '#bp-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#bp-mathbody')
                pg.evaluate("()=>{document.getElementById('bp-math').open=false}")
                go(pg, 1); pg.click('#bp-rail [data-pred="one"]'); pg.wait_for_timeout(200); go(pg, 2)
                for _ in range(6): pg.evaluate("()=>{const b=document.querySelector('#bp-ic-play [data-play=\"fwd\"]'); if(b&&!b.disabled) b.click()}"); pg.wait_for_timeout(120)
                pg.wait_for_timeout(300)
                Lines += ['## Part 3 after the right prediction', txt(pg, '#bp-rail [data-obs]')]
            if sid == 'check':
                for v in ('1', '2'):
                    pg.click(f'#bp-ic-h button[data-v="{v}"]'); pg.wait_for_timeout(250)
                    Lines += [f'## Result at h choice {v}', txt(pg, '#bp-rail [data-obs]')]
                pg.click('#bp-ic-h button[data-v="3"]'); pg.wait_for_timeout(200)
                pg.evaluate("()=>{document.getElementById('bp-math').open=true}"); pg.wait_for_timeout(150)
                Lines += ['## Show the math (part 4)', txt(pg, '#bp-mathbody')]
                pg.evaluate("()=>{document.getElementById('bp-math').open=false}")
            if sid == 'inplace':
                pg.click('#bp-rail [data-lever="eng"]'); pg.wait_for_timeout(300)
                Lines += ['## Part 6 with the small engine', txt(pg, '#bp-rail .step.current'), '## Stage', stage(pg)]
            if sid == 'remember':
                for k2 in ('walk', 'check', 'fan', 'inpl'):
                    pg.click(f'#bp-rail [data-rem="{k2}"]'); pg.wait_for_timeout(350)
                    Lines += [f'## Part 7 after “Show in the chart” ({k2})', '## Stage', stage(pg), '## Tooltips in view (new)'] + T.grab(pg, '#bp-app .bench')
            Lines += [dock(pg)]
        pg.click('#bp-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#bp-rail')]
        for i in range(4):
            pg.click(f'#bp-rail [data-preset="{i}"]'); pg.wait_for_timeout(300)
            name = clean(' '.join(pg.inner_text(f'#bp-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#bp-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#bp-stage') + [dock(pg)]
        pg.click('#bp-ctl-h button[data-v="1"]'); pg.wait_for_timeout(300)
        Lines += ['## After changing h under Settings (no scenario matches)', txt(pg, '#bp-insight')]
        pg.click('#bp-ctl-tz button[data-v="1"]'); pg.wait_for_timeout(300)
        Lines += ['## With the tanh′ typo', txt(pg, '#bp-insight'), '## Stage', stage(pg)]
        Lines += ['# Settings panel', txt(pg, '#bp-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#bp-controls')
        pg.click('#bp-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400)
        hid = pg.evaluate("()=>window.__bp.hidden(window.__bp.state().chSeed)"); mn = pg.evaluate("h=>window.__bp.minChecks(h)", hid)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#bp-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#bp-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#bp-app .bench') + [dock(pg)]
        pg.click('#bp-chks [data-chk="c"]'); pg.wait_for_timeout(250)
        pg.click(f'#bp-ch-pick [data-lever="{(hid + 1) % 4}"]'); pg.wait_for_timeout(200); pg.click('#bp-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Wrong bug after one check of c', txt(pg, '#bp-ch-res')]
        pg.click('#bp-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
        Lines += ['## After “show all options”', txt(pg, '#bp-ch-res')]
        for p_ in ('w', 'b', 'v'):
            if not pg.evaluate(f"()=>document.querySelector('#bp-chks [data-chk=\"{p_}\"]').disabled"): pg.click(f'#bp-chks [data-chk="{p_}"]'); pg.wait_for_timeout(200)
        pg.click(f'#bp-ch-pick [data-lever="{hid}"]'); pg.wait_for_timeout(200); pg.click('#bp-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Right bug with every check', txt(pg, '#bp-ch-res')]
        pg.click('#bp-rail [data-act="another"]'); pg.wait_for_timeout(300)
        hid = pg.evaluate("()=>window.__bp.hidden(window.__bp.state().chSeed)"); mn = pg.evaluate("h=>window.__bp.minChecks(h)", hid)
        for p_ in mn: pg.click(f'#bp-chks [data-chk="{p_}"]'); pg.wait_for_timeout(200)
        pg.click(f'#bp-ch-pick [data-lever="{hid}"]'); pg.wait_for_timeout(200); pg.click('#bp-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Another hidden bug, right with the fewest checks', txt(pg, '#bp-ch-res'), 'Status: ' + txt(pg, '#bp-status'), '## Stage', stage(pg), dock(pg)]
        pg.evaluate("()=>{document.getElementById('bp-math').open=true}"); pg.wait_for_timeout(150)
        Lines += ['## Show the math (challenge)', txt(pg, '#bp-mathbody')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#bp-rail')
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        pg.click('#bp-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#bp-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#bp-app .connect')
        if errs: Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
