"""Dump every piece of text a learner meets in the prefill-decode lab and its home tile, in reading order, in English and Chinese
(CONTRACT §0: export the whole text in reading order and read it before reporting).

python3 tests/prefill-decode_copy_deck.py  ->  out/prefill-decode-copy/en.md and out/prefill-decode-copy/zh.md

Order: home tile (sentence at every prompt length, the notes behind "Assumptions & sources", tooltips), lab header with its tooltips
and both popovers, each guided part (card before and after its action, the stage, the tooltips in view that were not shown before,
the mini result bar), the prediction's reveal for a wrong and for the right pick, the roofline deep dive, part 6 (the experiment:
before the prediction, then every GPU on both loads after a wrong pick, Show the math for each doubling, and the verdict after the right
pick), Show the math for two setups, the scenarios card for every scenario, the settings panel, the challenge before and after submitting (every feedback tier
and the chart), the card "Can you use it on a new case?" (every question, a verdict for every option), and the connect section
with the drill answer. A tooltip is printed the first time it appears with a given text, so a repeated legend does not drown the rest.
"""
import pathlib, subprocess, sys
from playwright.sync_api import sync_playwright

P = pathlib.Path(__file__).resolve().parent.parent
OUT = P / 'out' / 'prefill-decode-copy'; OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(P / 'build.py'), 'out/prefill-decode-deck.html', '--only', 'prefill-decode,set-a'], cwd=P, check=True, capture_output=True)
(P / 'out' / 'wrapped-prefill-decode-deck.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}[hidden]{display:none!important}</style></head><body>' + (P / 'out' / 'prefill-decode-deck.html').read_text() + '</body></html>')
URL = 'file://' + str(P / 'out' / 'wrapped-prefill-decode-deck.html')
clean = lambda s: s.replace('‑', '-').replace(' ', ' ').replace(' ', ' ').strip()


def txt(pg, sel):
    el = pg.query_selector(sel)
    return clean(el.inner_text()) if el and el.is_visible() else ''


def stage(pg):
    """The stage in reading order: head, legend, timeline, the two limits of each phase, memory bar, roofline, metrics."""
    parts = [txt(pg, '#pd-st-title'), txt(pg, '#pd-st-sub'), 'Status: ' + txt(pg, '#pd-status'),
             'Legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#pd-legend > span')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    if pg.is_visible('#pd-tl'):
        parts += ['Timeline: ' + txt(pg, '#pd-tl-cap'), ' / '.join(x for x in [txt(pg, '.pd-tl-lpre'), txt(pg, '.pd-tl-ldec')] if x),
                  'In the prefill block: ' + (txt(pg, '.pd-tl-pin') or '(none)') + ('; ' + txt(pg, '.pd-tl-mask') if txt(pg, '.pd-tl-mask') else ''),
                  'Axis: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('.pd-tl-axis span')].map(s=>s.textContent).join('  ')"))]
    if pg.is_visible('#pd-xa'):
        parts += ['Three timelines: ' + txt(pg, '#pd-xa-cap'), 'Label: ' + txt(pg, '.pd-xa-ideal')]
        parts += [clean(x) for x in pg.evaluate("""()=>[...document.querySelectorAll('.pd-xrow')].map(r=>'- '+(r.getAttribute('aria-pressed')==='true'?'(picked) ':'')+
            r.querySelector('.pd-xrow-n').innerText+' :: '+(r.querySelector('.pd-xmask').hidden?r.querySelector('.pd-xrow-v').innerText.replace(/\\n/g,' / '):r.querySelector('.pd-xmask').innerText))""")]
        parts += ['Axis: ' + clean(pg.evaluate("()=>[...document.querySelectorAll('.pd-xaxis span')].map(s=>s.textContent).join('  ')"))]
        if pg.is_visible('.pd-xgo'): parts += ['Button: ' + txt(pg, '.pd-xgo')]
        if pg.is_visible('#pd-xzoom'):
            parts += ['Zoom: ' + txt(pg, '.pd-xzoom-cap')] + [clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('.pd-xzrow')].filter(r=>r.offsetParent).map(r=>'- '+r.querySelector('.pd-xzn').innerText+' :: '+r.querySelector('.pd-xzv').innerText)")]
    if pg.is_visible('#pd-lim'):
        parts.append('Limits: ' + txt(pg, '#pd-lim-cap'))
        parts += [clean(x) for x in pg.evaluate("""()=>[...document.querySelectorAll('#pd-lim .pd-ph')].filter(p=>p.offsetParent).map(p=>{
            const h=p.querySelector('.pd-ph-h').innerText.replace(/\\n/g,' '), m=p.querySelector('.pd-mask');
            if(m&&!m.hidden) return '- '+h+' :: '+m.innerText;
            return '- '+h+' :: '+[...p.querySelectorAll('.pd-row')].map(r=>{const t=r.querySelector('.pd-settag'); return r.querySelector('.pd-rl').innerText+' '+r.querySelector('.pd-rv').innerText+(t&&!t.hidden?' ['+t.innerText+']':'')}).join(' / ')})""")]
    if pg.is_visible('#pd-mem'):
        parts += ['Memory: ' + txt(pg, '#pd-mem-cap'), 'Memory legend: ' + ' | '.join(clean(x) for x in pg.evaluate("()=>[...document.querySelectorAll('#pd-memleg > *')].filter(s=>s.offsetParent).map(s=>s.innerText)"))]
    if pg.is_visible('#pd-roof'):
        parts += ['Roofline: ' + ' '.join(txt(pg, '#pd-roof').split())]
    mets = pg.evaluate("()=>[...document.querySelectorAll('#pd-metrics .metric')].map(m=>[...m.children].map(c=>c.innerText.trim()).filter(Boolean).join(' | '))")
    return '\n'.join([p for p in parts if p] + ['Metrics:'] + [f'- {clean(m)}' for m in mets])


class Tips:
    """Hover every visible element that has a tooltip and record the text the first time it appears."""
    def __init__(self): self.seen = set()

    def grab(self, pg, root):
        out = []
        for e in pg.query_selector_all(f'{root} [data-tip], {root} [data-tipx], {root} .t, {root} [data-tipfn], {root} [data-jump], {root} .pd-row, {root} .pd-rrg, {root} .pd-rpt, {root} #pd-tl-track, {root} #pd-membar'):
            try:
                if not e.is_visible(): continue
                # innerText does not exist on SVG elements (the roofline's points), so fall back to textContent
                label = ' '.join((e.evaluate("el=>el.innerText||el.textContent||''") or e.get_attribute('aria-label') or '').split())[:60]
                e.scroll_into_view_if_needed(timeout=1500); pg.wait_for_timeout(40)
                # an SVG group (a roofline point or the ridge) is hovered at the middle of its shape, other elements on their first line box
                bb = e.evaluate("""el=>{const s=el instanceof SVGGElement?el.querySelector('circle,rect,.pd-rhit'):null; if(s){const r=s.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height,mid:1};}
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
    """wait for the lab's correction that keeps the stage in view, then for the scroll to stop"""
    try: pg.wait_for_function("()=>!window.__pd.state().stageBusy", timeout=4000)
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
    pg.evaluate("()=>{const r=document.querySelector('#pd-rail .step.current, #pd-rail .railcard'); if(r) r.scrollIntoView({block:'start'}); window.scrollBy(0,-120); document.querySelector('#pd-app .bench').click()}"); pg.wait_for_timeout(350)
    t = clean(pg.inner_text('#dock')) if pg.is_visible('#dock') else '(mini result bar not shown)'
    pg.set_viewport_size(vp); pg.wait_for_timeout(200)
    return 'Mini result bar (390 px): ' + ' '.join(t.split())


def rng(pg, sel, v):
    pg.evaluate(f"()=>{{const i=document.querySelector('{sel}'); i.value={v}; i.dispatchEvent(new Event('input',{{bubbles:true}})); i.dispatchEvent(new Event('change',{{bubbles:true}}))}}"); pg.wait_for_timeout(220)


def go(pg, k):
    pg.evaluate(f"()=>document.querySelector('#pd-rail [data-go=\"{k}\"]').click()"); pg.wait_for_timeout(350); calm(pg)


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
        if pg.query_selector('#tile-prefill-decode'):
            pg.evaluate("()=>document.querySelector('#tile-prefill-decode').scrollIntoView({block:'center'})"); pg.wait_for_timeout(200)
            Lines += ['# Home tile', txt(pg, '#tile-prefill-decode')]
            nmax = int(pg.evaluate("()=>document.querySelector('#tile-prefill-decode input[type=range]').max"))
            for v in range(nmax + 1):
                rng(pg, '#tile-prefill-decode input[type=range]', v)
                Lines += [f'## Tile sentence at step {v}: ' + txt(pg, '#tile-prefill-decode output'), txt(pg, '#tile-prefill-decode .tz-read')]
            Lines += ['## Assumptions & sources', clean(pg.evaluate("()=>[...document.querySelectorAll('#tile-prefill-decode .tz-note')].map(n=>n.textContent.replace(/\\s+/g,' ').trim()).join('\\n\\n')"))]
            Lines += ['## Tile tooltips'] + T.grab(pg, '#tile-prefill-decode')
        else:
            Lines += ['# Home tile', '(tile not found)']
        # ---------- lab header ----------
        pg.evaluate("()=>{location.hash='prefill-decode'}"); pg.wait_for_timeout(700)
        Lines += ['# Lab header', txt(pg, '.dyn-root .labhead')]
        Lines += ['## Header tooltips'] + T.grab(pg, '.dyn-root .labhead')
        for sel in pg.query_selector_all('.dyn-root .labhead [data-pop]'):
            pid = sel.get_attribute('data-pop'); sel.click(); pg.wait_for_timeout(200)
            Lines += [f'## Popover: {clean(" ".join(sel.inner_text().split()))}', clean(pg.evaluate("(id)=>{const p=document.getElementById(id); return p?p.innerText.trim():''}", pid))]
            Lines += ['## Popover tooltips'] + T.grab(pg, f'#{pid}')
            sel.click(); pg.wait_for_timeout(150)
        # ---------- guided parts ----------
        ids = pg.evaluate("()=>window.__pd.consts.STEP_IDS")
        for k, sid in enumerate(ids):
            go(pg, k)
            Lines += [f'# Guided part {k + 1} ({sid})', '## Card', txt(pg, '#pd-rail .step.current'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pd-app .bench')
            acted = True
            if sid == 'predict': pg.click('#pd-rail [data-pred="later"]')
            elif sid == 'prefill': rng(pg, '#pd-ic-N-in', 4)
            elif sid == 'decode': rng(pg, '#pd-ic-N-in', 2)
            elif sid == 'batch': rng(pg, '#pd-ic-B-in', 64)
            elif sid == 'double': pg.click('#pd-rail [data-xpred="both"]')
            else: acted = False
            if acted:
                pg.wait_for_timeout(350); calm(pg)
                Lines += ['## Card after the action', txt(pg, '#pd-rail .step.current'), '## Stage after the action', stage(pg)]
                Lines += ['## Tooltips in view after the action (new)'] + T.grab(pg, '#pd-app .bench')
            if sid == 'prefill':
                for v, name in ((0, '512 tokens'), (6, '32K tokens')):
                    rng(pg, '#pd-ic-N-in', v); Lines += [f'## Result at {name}', txt(pg, '#pd-rail [data-obs]')]
                rng(pg, '#pd-ic-N-in', 4)
            if sid == 'decode':
                rng(pg, '#pd-ic-N-in', 6); Lines += ['## Result at 32K tokens of context', txt(pg, '#pd-rail [data-obs]')]
                # the reveal for the right prediction: back to part 2, pick it, then part 4 again (at 8K, as the prediction asked)
                go(pg, 1); pg.click('#pd-rail [data-pred="first"]'); pg.wait_for_timeout(200)
                Lines += ['## Part 2 card after picking the right answer', txt(pg, '#pd-rail .step.current')]
                go(pg, 2); rng(pg, '#pd-ic-N-in', 4); go(pg, 3)
                Lines += ['## Part 4 card after the right prediction (8K)', txt(pg, '#pd-rail .step.current')]
            if sid == 'batch':
                pg.click('#pd-rail [data-act="roof"]'); pg.wait_for_timeout(350); calm(pg)
                Lines += ['## Deep dive: the roofline (64 requests)', '## Card', txt(pg, '#pd-rail .step.current'), '## Stage', stage(pg)]
                Lines += ['## Roofline tooltips (new)'] + T.grab(pg, '#pd-roof')
                pg.click('#pd-roof [data-act="roof-close"]'); pg.wait_for_timeout(250)
                lim = pg.evaluate("()=>document.querySelector('#pd-ic-B-in').max")
                rng(pg, '#pd-ic-B-in', lim); Lines += [f'## Result at the memory limit ({lim} requests)', txt(pg, '#pd-rail [data-obs]')]
                rng(pg, '#pd-ic-B-in', 64)
            if sid == 'double':
                # every GPU on both loads (after a wrong pick), then Show the math for each doubling, then the right pick's verdict
                for l in ('chat', 'doc'):
                    pg.evaluate(f"()=>document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"] button[data-v=\"{l}\"]').click()"); pg.wait_for_timeout(200)
                    for v in ('c2', 'b2', 'orig'):
                        pg.evaluate(f"()=>document.querySelector('#pd-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(250); calm(pg)
                        Lines += [f'## Part 6, load {l}, GPU {v}: card', txt(pg, '#pd-rail .step.current'), f'## Part 6, load {l}, GPU {v}: stage', stage(pg)]
                        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pd-app .bench')
                pg.evaluate("()=>document.querySelector('#pd-rail .seg-ctl[data-key=\"xl\"] button[data-v=\"chat\"]').click()"); pg.wait_for_timeout(150)
                for v in ('c2', 'b2'):
                    pg.evaluate(f"()=>document.querySelector('#pd-rail [data-lever=\"{v}\"]').click()"); pg.wait_for_timeout(200)
                    pg.evaluate("()=>{document.getElementById('pd-math').open=true}"); pg.wait_for_timeout(150)
                    Lines += [f'## Part 6: Show the math ({v}, chat)', txt(pg, '#pd-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#pd-mathbody')
                pg.evaluate("()=>{document.getElementById('pd-math').open=false}")
                p2 = b.new_page(viewport={'width': 1440, 'height': 1000}); p2.goto(URL); p2.wait_for_timeout(300); p2.click(f'.langsw button[data-lang="{lang}"]'); p2.wait_for_timeout(150)
                p2.evaluate("()=>{location.hash='prefill-decode'}"); p2.wait_for_timeout(600); go(p2, k)
                p2.click('#pd-rail [data-xpred="first"]'); p2.wait_for_timeout(200); p2.click('#pd-rail [data-lever="c2"]'); p2.wait_for_timeout(300)
                Lines += ['## Part 6 card after the right prediction (compute × 2, chat)', txt(p2, '#pd-rail .step.current')]
                p2.close()
            Lines += [dock(pg)]
        # ---------- Show the math ----------
        for tok, name in [('2048_300_1', 'the mission: 2K prompt, 300 tokens, 1 request'), ('4096_300_46', 'the challenge’s answer: 4K, 46 requests')]:
            pg.evaluate("()=>{location.hash='map'}"); pg.wait_for_timeout(150)
            pg.evaluate(f"()=>{{location.hash='prefill-decode~{tok}'}}"); pg.wait_for_timeout(500)
            pg.evaluate("()=>{document.getElementById('pd-math').open=true}"); pg.wait_for_timeout(150)
            Lines += [f'# Show the math ({name})', txt(pg, '#pd-mathbody'), '## Math tooltips (new)'] + T.grab(pg, '#pd-mathbody')
        # ---------- scenarios ----------
        pg.click('#pd-app .modes button[data-mode="sandbox"]:not([data-flow=free])'); pg.wait_for_timeout(400)
        Lines += ['# Scenarios', txt(pg, '#pd-rail')]
        for i in range(4):
            pg.click(f'#pd-rail [data-preset="{i}"]'); pg.wait_for_timeout(250); calm(pg)
            name = clean(' '.join(pg.inner_text(f'#pd-rail [data-preset="{i}"]').split()))
            Lines += [f'## Scenario {i + 1}: {name}', txt(pg, '#pd-insight'), '## Stage', stage(pg)]
            Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pd-stage') + [dock(pg)]
        rng(pg, '#pd-ctl-M-in', 0)
        Lines += ['## After changing the answer length under Settings (no scenario matches)', txt(pg, '#pd-insight')]
        # ---------- settings panel ----------
        Lines += ['# Settings panel', txt(pg, '#pd-controls'), '## Settings tooltips (new)'] + T.grab(pg, '#pd-controls')
        # ---------- challenge ----------
        pg.click('#pd-app .modes button[data-mode="challenge"]'); pg.wait_for_timeout(400); calm(pg)
        Lines += ['# Challenge', '## Card before submitting', txt(pg, '#pd-rail .railcard'), '## Stage', stage(pg), '## Settings panel in the challenge', txt(pg, '#pd-controls')]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pd-app .bench') + [dock(pg)]
        rng(pg, '#pd-ic-chB-in', 64); pg.click('#pd-rail [data-act="submit"]'); pg.wait_for_timeout(300)
        Lines += ['## Submitted 64 users (goal missed)', txt(pg, '#pd-ch-res'), 'Status: ' + txt(pg, '#pd-status')]
        rng(pg, '#pd-ic-chB-in', 32)
        Lines += ['## 32 users (met, more fit)', txt(pg, '#pd-ch-res')]
        if pg.query_selector('#pd-rail [data-act="reveal"]'):
            pg.click('#pd-rail [data-act="reveal"]'); pg.wait_for_timeout(300)
            Lines += ['## After “Show the answer”', txt(pg, '#pd-ch-res')]
        rng(pg, '#pd-ic-chB-in', 46)
        Lines += ['## 46 users (met, the most)', txt(pg, '#pd-ch-res'), 'Status: ' + txt(pg, '#pd-status'), '## Stage', stage(pg), dock(pg)]
        Lines += ['## Tooltips in view (new)'] + T.grab(pg, '#pd-rail')
        # ---------- can you use it on a new case? ----------
        Lines += ['# Can you use it on a new case?'] + exits(pg)
        # ---------- connect ----------
        pg.click('#pd-app [data-act="drill-reveal"]'); pg.wait_for_timeout(200)
        Lines += ['# Connect', txt(pg, '#pd-app .connect'), '## Connect tooltips (new)'] + T.grab(pg, '#pd-app .connect')
        if errs:
            Lines += ['# Page errors'] + errs
        (OUT / f'{lang}.md').write_text('\n\n'.join(x for x in Lines if x) + '\n')
        pg.close()
    b.close()
print('wrote', [str(OUT / f'{l}.md') for l in ['en', 'zh']])
