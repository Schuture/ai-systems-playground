"""Dump every piece of text a learner meets in the chunked prefill lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/chunked-prefill_copy_deck.py  ->  out/chunked-prefill-copy/en.md and out/chunked-prefill-copy/zh.md

Order: home tile (the sentence for every chunk size, the notes, tooltips), lab header with its tooltips and both popovers, each guided
part (card before and after its action, the stage, the tooltips in view that were not shown before, the mini result bar), the replay's
clock lines, the prediction's reveal for a wrong and for the right pick, the deep dive, Show the math for three setups, the scenarios
card for every scenario, the settings panel, the challenge before and after submitting (every feedback tier and the chart), the card
"Can you use it on a new case?" (every question, a verdict for every option), and the connect section with the drill answer. A tooltip
is printed the first time it appears with a given text, so a repeated legend does not drown the rest.
"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'chunked-prefill-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/chunked-prefill-deck.html', '--only', 'chunked-prefill,set-a'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-chunked-prefill-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'chunked-prefill-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-chunked-prefill-deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace('\xa0', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    """The stage in reading order: head, legend, timeline, clock, the two limits (or the deep dive), scope, metrics."""
    parts = [txt(pg, '#ck-st-title'), txt(pg, '#ck-st-sub'), 'Status: ' + txt(pg, '#ck-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#ck-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    if pg.is_visible('#ck-tl'):
        parts += ['Timeline: ' + txt(pg, '#ck-tl-cap'),
                  'Users’ row: ' + ' / '.join(x for x in [txt(pg, '#ck-lh-u'), txt(pg, '#ck-lh-u2')] if x) + ' :: bracket: ' + (txt(pg, '#ck-lg em') or txt(pg, '#ck-wt em')),
                  'Prompt’s row: ' + ' / '.join(x for x in [txt(pg, '#ck-lh-p'), txt(pg, '#ck-lh-p2')] if x) + ' :: ' + ' / '.join(x for x in [txt(pg, '.ck-arr-a'), txt(pg, '.ck-arr-f'), txt(pg, '.ck-pin')] if x),
                  'Axis: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('#ck-axis span')].map(s=>s.textContent).join('  ')")),
                  'Clock: ' + txt(pg, '#ck-clock')]
    if pg.is_visible('#ck-lim'):
        parts.append('Limits: ' + txt(pg, '#ck-lim-cap'))
        parts.append('Limits legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#ck-limleg > span')].filter(s=>s.offsetParent).map(s=>s.innerText)")))
        parts += [clean(x) for x in pg.evaluate("""()=>[...document.querySelectorAll('#ck-lim .ck-ph')].filter(p=>p.offsetParent).map(p=>{
            const h=p.querySelector('.ck-ph-h').innerText.replace(/\\n/g,' ');
            return '- '+h+' :: '+[...p.querySelectorAll('.ck-row')].map(r=>{const t=r.querySelector('.ck-settag'); return r.querySelector('.ck-rl').innerText+' '+r.querySelector('.ck-rv').innerText+(t&&!t.hidden?' ['+t.innerText+']':'')}).join(' / ')})""")]
    if pg.is_visible('#ck-deep'):
        parts += ['Deep dive: ' + ' '.join(txt(pg, '#ck-deep').split()), 'Deep dive chart text: ' + clean(' '.join(pg.evaluate("()=>document.querySelector('#ck-deep svg').textContent").split()))]
    parts.append('Scope: ' + txt(pg, '#ck-scope'))
    mets = pg.evaluate("()=>[...document.querySelectorAll('#ck-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    """Hover every visible element that has a tooltip and record the text the first time it appears."""
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        sel = f'{root} [data-tip], {root} [data-tipx]:not(.ck-seg), {root} .t, {root} [data-tipfn], {root} [data-jump], {root} .ck-row, {root} #ck-tu, {root} #ck-tp, {root} .ck-fpt'
        els = pg.query_selector_all(sel)
        segs = pg.query_selector_all(f'{root} .ck-seg[data-tipx]')
        if segs: els += [segs[0], segs[-1]] if len(segs) > 1 else segs
        for e in els:
            try:
                if not e.is_visible(): continue
                label = ' '.join((e.evaluate("el=>el.innerText||el.textContent||''") or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(40)
                bb = e.evaluate("""el=>{const s=el instanceof SVGGElement?el.querySelector('circle,rect'):null; if(s){const r=s.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height,mid:1};}
                  const r=el.getClientRects()[0]; return r?{x:r.x,y:r.y,width:r.width,height:r.height}:null}""")
                if not bb: continue
                if bb.get('mid'): pg.mouse.move(bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2)
                else: pg.mouse.move(bb['x'] + min(bb['width'] / 2, 12), bb['y'] + min(bb['height'] / 2, 6))
                pg.wait_for_timeout(110)
                t = ' '.join(pg.inner_text('#tip').split()) if pg.is_visible('#tip') else ''
            except Exception:
                continue
            if t and t not in self.seen:
                self.seen.add(t); out.append(f'- {clean(label)} ⟶ {clean(t)}')
            pg.mouse.move(1, 1)
        return out


def calm(pg):
    try: pg.wait_for_function("()=>!window.__ck.state().stageBusy", timeout=4000)
    except Exception: pass
    last = None
    for _ in range(25):
        y = pg.evaluate('()=>scrollY')
        if y == last: return
        last = y; pg.wait_for_timeout(120)


def dock(pg):
    """Scroll the stage out of view on a phone-sized window and read the mini result bar."""
    vp = pg.viewport_size
    pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(250)
    pg.evaluate("()=>{const r=document.querySelector('#ck-rail .step.current, #ck-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#ck-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def chunk(pg, where, c):
    pg.evaluate(f"()=>document.querySelector('{where} .seg-ctl[data-key=\"C\"] button[data-v=\"{c}\"]').click()"); pg.wait_for_timeout(300); calm(pg)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#ck-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(350); calm(pg)


def exits(pg):
    """The card “Can you use it on a new case?”: every question, and the verdict for every option in turn (wrong ones first)."""
    out = []
    card = "[...document.querySelectorAll('[data-exits]')].find(e=>e.offsetParent!==null)"
    for t in range(3):
        pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').click()"); pg.wait_for_timeout(150)
        name = clean(pg.evaluate(f"()=>{card}.querySelector('[data-ex=\"tab\"][data-i=\"{t}\"]').textContent"))
        if t == 2:
            out += [f'## {name}', clean(pg.evaluate(f"()=>{card}.querySelector('.ex-body, .ex-links')?.innerText||{card}.innerText"))]
            continue
        nq = pg.evaluate(f"()=>{card}.querySelectorAll('.ex-qs .pb-dot').length||1")
        for j in range(nq):
            if j:
                pg.evaluate(f"()=>{card}.querySelector('.ex-qs .pb-dot[data-j=\"{j}\"]').click()"); pg.wait_for_timeout(150)
            q = clean(pg.evaluate(f"()=>{{const c={card}, e=c.querySelector('.ex-body > p > b'); return e?e.innerText:''}}"))
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
        # ---------- home tile ----------
        pg.evaluate("""()=>{const g=[...document.querySelectorAll('#domGrid [data-world="inference"], #domStrip [data-world="inference"]')].find(e=>e.offsetParent);
          if(g) g.click(); else {const m=document.querySelector('#mapSvg .world[data-world="inference"]'); if(m) m.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}"""); pg.wait_for_timeout(400)
        if pg.query_selector('#tile-chunked-prefill'):
            pg.evaluate("()=>document.querySelector('#tile-chunked-prefill').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-chunked-prefill')]
            for v in pg.evaluate("()=>[...document.querySelectorAll('#tile-chunked-prefill .seg-ctl button')].map(b=>b.dataset.v)"):
                pg.evaluate(f"()=>document.querySelector('#tile-chunked-prefill .seg-ctl button[data-v=\"{v}\"]').click()"); pg.wait_for_timeout(200)
                Lines += [f'## Tile sentence at chunk {v}', txt(pg, '#tile-chunked-prefill .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-chunked-prefill .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-chunked-prefill')
        else:
            Lines += ['# Home tile', '(tile not found)']
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='chunked-prefill'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        Lines += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        # ---------- guided parts ----------
        ids = pg.evaluate("()=>window.__ck.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#ck-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ck-app .bench')
            acted = True
            if sid == 'predict': pg.click('#ck-rail [data-pred="c"]')
            elif sid == 'stall':
                for v, name in ((30, 'before the prompt arrives'), (400, 'during the prefill step'), (900, 'after the first token')):
                    rng(pg, '#ck-scrub', v); Lines += [f'## Replay at {v / 10:.0f}% of the timeline ({name})', 'Clock: ' + txt(pg, '#ck-clock'), 'Users’ row: ' + (txt(pg, '#ck-wt em') or '(no bracket)'), 'Second box: ' + txt(pg, '.ck-ph[data-ph="s"] .ck-ph-h'), 'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#ck-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
                rng(pg, '#ck-scrub', 1000); acted = False
            elif sid == 'chunk': chunk(pg, '#ck-ic-C', 2048)
            elif sid == 'free': chunk(pg, '#ck-ic-C', 256)
            else: acted = False
            if acted:
                pg.wait_for_timeout(350); calm(pg)
                Lines += ['## Card after the action', txt(pg, '#ck-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#ck-app .bench')
            if sid == 'chunk':
                for c in (1024, 512, 256, 128, 32):
                    chunk(pg, '#ck-ic-C', c); Lines += [f'## Result at {c}-token chunks', txt(pg, '#ck-rail [data-obs]')]
                chunk(pg, '#ck-ic-C', 2048)
                go(pg, 1); pg.click('#ck-rail [data-pred="a"]'); pg.wait_for_timeout(200)
                Lines += ['## Part 2 card after picking the right answer', txt(pg, '#ck-rail .step.current')]
                go(pg, 3); chunk(pg, '#ck-ic-C', 2048)
                Lines += ['## Part 4 card after the right prediction (2K)', txt(pg, '#ck-rail .step.current')]
            if sid == 'free':
                for c in (1024, 512, 128, 64, 32):
                    chunk(pg, '#ck-ic-C', c); Lines += [f'## Result at {c}-token chunks', txt(pg, '#ck-rail [data-obs]')]
                Lines += ['## Stage at 32-token chunks', stage(pg), '## Tooltips in view at 32-token chunks (new)'] + T.grab(pg, '#ck-app .bench')
                pg.click('#ck-rail details.ck-why summary'); pg.wait_for_timeout(200)
                Lines += ['## The “why” opened', txt(pg, '#ck-rail details.ck-why')]
                chunk(pg, '#ck-ic-C', 256)
                pg.click('#ck-rail [data-act="deep"]'); pg.wait_for_timeout(350); calm(pg)
                Lines += ['## Deep dive: the free chunk', '## Stage', stage(pg)]
                Lines += ['## Deep dive tooltips (new)'] + T.grab(pg, '#ck-deep')
                pg.click('#ck-deep [data-act="deep-close"]'); pg.wait_for_timeout(250)
            Lines += [dock(pg)]
        # ---------- Show the math ----------
        for tok, name in [('16_4096_32768_0', 'the mission: no chunking'), ('16_4096_32768_2048', '2K-token chunks'), ('16_4096_32768_256', '256-token chunks')]:
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
            pg.evaluate(f"()=>{{location.hash='chunked-prefill~{tok}'}}"); pg.wait_for_timeout(500)
            pg.evaluate("()=>{document.getElementById('ck-math').open=true}"); pg.wait_for_timeout(150)
            Lines += [f'# Show the math ({name})', txt(pg, '#ck-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#ck-mathbody')
        # ---------- scenarios ----------
        Lines += ['# Scenarios', txt(pg, '#ck-rail')]
        chunk(pg, '#ck-sb-C', 2048)
        for i in range(4):
            pg.click(f'#ck-rail [data-preset="{i}"]'); pg.wait_for_timeout(250); calm(pg)
            name = clean(' '.join(pg.inner_text(f'#ck-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name} (2K-token chunks)', txt(pg, '#ck-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ck-stage') + [dock(pg)]
            if i in (1, 2):
                chunk(pg, '#ck-sb-C', 0); Lines += [f'## Scenario {i + 1} without chunking', txt(pg, '#ck-insight')]; chunk(pg, '#ck-sb-C', 2048)
        rng(pg, '#ck-ctl-U-in', 2)
        Lines += ['## After changing the users under Settings (no scenario matches)', txt(pg, '#ck-insight')]
        # ---------- settings panel ----------
        Lines += ['# Settings panel', txt(pg, '#ck-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#ck-controls')
        # ---------- challenge ----------
        pg.click('#ck-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400); calm(pg)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#ck-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#ck-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ck-app .bench') + [dock(pg)]
        chunk(pg, '#ck-ic-chC', 1024); pg.click('#ck-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted 1K-token chunks (goal missed)', txt(pg, '#ck-ch-res'), 'Status: ' + txt(pg, '#ck-status')]
        if pg.query_selector('#ck-rail [data-act="reveal"]'):
            pg.click('#ck-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
            Lines += ['## After “Show the answer”', txt(pg, '#ck-ch-res')]
        chunk(pg, '#ck-ic-chC', 256)
        Lines += ['## 256-token chunks (met, not the earliest)', txt(pg, '#ck-ch-res')]
        chunk(pg, '#ck-ic-chC', 512)
        Lines += ['## 512-token chunks (met, the earliest)', txt(pg, '#ck-ch-res'), 'Status: ' + txt(pg, '#ck-status'), '## Stage', stage(pg), dock(pg)]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#ck-rail')
        # ---------- can you use it on a new case? ----------
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        # ---------- connect ----------
        pg.click('#ck-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#ck-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#ck-app .connect')
        if errs:
            Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
